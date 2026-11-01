# FlyDSL K5 opt BV Tune

Model untuned/tuned CSVs live under `aiter/configs/model_configs/` (`qwen3_5_*_chunk_gdn_h_opt_{un,t}uned.csv`). Untuned tables are shape-only; `cu_num` lives in tuned rows. Newly tuned rows belong in the per-model tuned table, not in the canonical one. Prefill cases are in `op_tests/test_flydsl_linear_attention_prefill.py`.

Two header-only files under `aiter/configs/` back the merge that `AITER_CONFIG_GDN_K5_OPT` performs (and that the opt AOT reads):

- `chunk_gdn_h_opt_tuned.csv` -- merge anchor. Must be a readable csv: with no per-model table present, `get_config_file` returns this path as-is.
- `chunk_gdn_h_opt_untuned.csv` -- supplies the duplicate-detection keys, so two tables claiming the same lookup key fail the merge instead of silently coexisting. Its columns must stay equal to the tuner's `LOOKUP_KEYS`; using shape-only columns would mark every per-batch row a duplicate and collapse the table.

```bash
# Tune (write candidates under /tmp before merging)
python3 csrc/gdn_k5/chunk_gdn_h_opt_tune.py \
  -i aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_opt_untuned.csv \
  -o /tmp/qwen3_5_35b_chunk_gdn_h_opt_tuned.candidate.csv \
  --case 'Qwen3.5-35B-dense-tp4-bf16snap'

# Replay tuned rows (default 5% us drift tolerance; CI test_run_config)
python3 csrc/gdn_k5/chunk_gdn_h_opt_tune.py \
  --run_config aiter/configs/model_configs/qwen3_5_397b_chunk_gdn_h_opt_tuned.csv
```

Options: `-i`/`-o`, `--run_config`, `--compare --update_improved`, `--case REGEX ...`, `--list-cases`,
`--single-seq-chunks LO:HI`, `--min-gain-pct P`.

- `--single-seq-chunks LO:HI` adds one varlen sequence of every chunk count in `LO..HI` (TP1, h0 + final state,
  bf16 snapshots, fp32 state: what vLLM's `aiter_flydsl` backend runs for a single prompt chunk) for each varlen
  shape in `-i`. The case catalog has only a few prompt lengths, and the winning BV can change at any chunk count.
  `--run_config` rebuilds these cases from the tuned rows.
- `--min-gain-pct P` writes a row only when the best BV beats the CU/LDS rule's BV by at least `P` percent, so the
  table carries real overrides only (the rule depends on the CU count, so it is re-checked on each GPU).

```bash
# Per-GPU table for one model, e.g. Qwen3.8-27B on a 128-CU gfx950: single prompts up to 32768 tokens
# plus the catalog's batched cases, keeping rows that beat the rule by >= 2%
python3 csrc/gdn_k5/chunk_gdn_h_opt_tune.py   -i aiter/configs/model_configs/qwen3_8_27b_chunk_gdn_h_opt_untuned.csv   -o /tmp/qwen3_8_27b_chunk_gdn_h_opt_tuned.candidate.csv   --case 'Qwen3.8-27B-vllm-tp1|single-H48' --single-seq-chunks 1:512 --min-gain-pct 2
```

Missing tuned lookup keys fall back to the CU/LDS BV rule until new rows are merged.
