# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL K5 opt BV tuner (TunerCommon). See ``csrc/gdn_k5/README.md``."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import logger
from aiter.jit.core import (
    AITER_CONFIG_GDN_K5_OPT,
    AITER_CONFIGS,
    AITER_ROOT_DIR,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.linear_attention_prefill_kernels import _hipeq_select_bv
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_opt as k5,
)
from aiter.utility.base_tuner import TunerCommon, _read_csv

CHUNK_SIZE = 64
BV_CANDIDATES = (16, 32, 64)
_K5_TEST_PATH = (
    Path(AITER_ROOT_DIR) / "op_tests" / "test_flydsl_linear_attention_prefill.py"
)
LOOKUP_KEYS = (
    "gfx",
    "cu_num",
    "H",
    "Hg",
    "V",
    "is_varlen",
    "use_h0",
    "store_fs",
    "snapshot_bf16",
    "state_bf16",
    "total_chunks",
    "max_seq_chunks",
)
TUNED_EXTRA_COLS = ("dtype", "K", "V", "BT", "T_flat", "N", "BV", "us")
TUNED_COLUMNS = LOOKUP_KEYS + TUNED_EXTRA_COLS
# On-disk column order matches shipped model_configs CSVs (dtype/shape before
# lookup tail). ``self.columns`` keeps LOOKUP_KEYS order for dedup; exports use
# ``_CSV_COLUMN_ORDER`` so ``result_to_csv`` does not scramble headers.
_CSV_COLUMN_ORDER = (
    "gfx",
    "cu_num",
    "dtype",
    "K",
    "V",
    "BT",
    "H",
    "Hg",
    "is_varlen",
    "use_h0",
    "store_fs",
    "snapshot_bf16",
    "state_bf16",
    "T_flat",
    "N",
    "total_chunks",
    "max_seq_chunks",
    "BV",
    "us",
)

_RUN_CONFIG_TOL_PCT = 5.0
_RESULT_COLS = [c for c in TUNED_COLUMNS if c not in LOOKUP_KEYS]
_DEFAULT_UNTUNED = (
    f"{AITER_ROOT_DIR}/aiter/configs/model_configs/"
    "qwen3_5_35b_chunk_gdn_h_opt_untuned.csv"
)


def _resolve_tuned_config_for_read(path: str) -> str:
    """Default tuned path reads the merged runtime table; writes stay on ``-o``."""
    if os.getenv("AITER_CONFIG_GDN_K5_OPT") or path != f"{AITER_CONFIG_GDN_K5_OPT}":
        return path
    return AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE


_K5_CASES_MODULE = None


def _load_k5_module():
    global _K5_CASES_MODULE
    if _K5_CASES_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "_gdn_k5_prefill_cases", _K5_TEST_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _K5_CASES_MODULE = module
    return _K5_CASES_MODULE


def load_k5_cases():
    module = _load_k5_module()
    return list(zip(module.PREFILL_TEST_IDS, module.PREFILL_PARAMS, strict=True))


def _single_seq_case(K, V, BT, H, Hg, output_final_state, t_flat):
    """One varlen sequence of ``t_flat`` tokens at TP1, as vLLM's
    aiter_flydsl backend runs a single prompt chunk (h0 + bf16 snapshots,
    fp32 state)."""
    module = _load_k5_module()
    return module.PrefillArgs(
        K=K,
        V=V,
        Hk=Hg,
        Hv=H,
        tp=1,
        full_prompt_len=t_flat,
        model_name=f"single-H{H}-Hg{Hg}",
        BT=BT,
        max_num_batched_tokens=t_flat,
        is_varlen=True,
        output_final_state=output_final_state,
    )


def parse_chunk_range(value: str) -> tuple[int, int]:
    lo, sep, hi = value.partition(":")
    if not sep:
        lo = hi = value
    lo_i, hi_i = int(lo), int(hi)
    if lo_i < 1 or hi_i < lo_i:
        raise ValueError(
            f"bad chunk range {value!r}; expected LO:HI with 1 <= LO <= HI"
        )
    return lo_i, hi_i


def single_seq_cases(untuned_rows, chunk_range: tuple[int, int]):
    """Single-sequence varlen cases for every chunk count in ``chunk_range``,
    one set per varlen untuned shape row. The case catalog only has a few
    prompt lengths; the BV that wins can change at any chunk count."""
    lo, hi = chunk_range
    out = []
    for row in untuned_rows:
        if not bool_cell(row["is_varlen"]):
            continue
        bt = int(row.get("BT") or 64)
        for chunks in range(lo, hi + 1):
            case = _single_seq_case(
                int(row["K"]),
                int(row["V"]),
                bt,
                int(row["H"]),
                int(row["Hg"]),
                bool_cell(row["store_fs"]),
                chunks * bt,
            )
            out.append((repr(case), case))
    return out


def case_from_tuned_row(row: dict[str, Any]):
    """Rebuild a single-sequence varlen case from a tuned row, for rows that
    no catalog case produces (e.g. from ``--single-seq-chunks``)."""
    if int(row["N"]) != 1 or not bool_cell(row["is_varlen"]):
        return None
    if not bool_cell(row["snapshot_bf16"]) or bool_cell(row["state_bf16"]):
        return None  # _single_seq_case builds bf16 snapshots, fp32 state
    return _single_seq_case(
        int(row["K"]),
        int(row["V"]),
        int(row.get("BT") or 64),
        int(row["H"]),
        int(row["Hg"]),
        bool_cell(row["store_fs"]),
        int(row["T_flat"]),
    )


def read_csv_rows(path: str) -> list[dict[str, str]]:
    df = _read_csv(path, comment="#")
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            {
                key: (
                    ""
                    if val is None or (isinstance(val, float) and math.isnan(val))
                    else str(val)
                )
                for key, val in row.items()
            }
        )
    return rows


def bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip() == "True"


def chunk_counts(context_lens, batch):
    per_seq = [(n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in context_lens]
    return sum(per_seq) * batch, max(per_seq)


def case_snapshot_dtype(case):
    return case.snapshot_dtype or case.dtype


def case_matches_untuned_shape(case, row: dict[str, str]) -> bool:
    bt = int(row.get("BT") or case.BT or 64)
    use_h0 = bool_cell(row.get("use_h0") or "True")
    return (
        case.K == int(row["K"])
        and case.V == int(row["V"])
        and case.BT == bt
        and case.H == int(row["H"])
        and case.Hg == int(row["Hg"])
        and case.is_varlen == bool_cell(row["is_varlen"])
        and use_h0
        and case.output_final_state == bool_cell(row["store_fs"])
    )


def select_cases(cases, untuned_rows, case_patterns: list[str]):
    patterns = [re.compile(p) for p in case_patterns] if case_patterns else []
    selected = []
    for case_id, case in cases:
        if not any(case_matches_untuned_shape(case, row) for row in untuned_rows):
            continue
        if patterns and not any(p.search(case_id) for p in patterns):
            continue
        selected.append((case_id, case))
    return selected


def build_k5_inputs(case, snapshot_dtype, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    context_lens = case.resolve_context_lens()
    t_flat = sum(context_lens)
    h, hg = case.H, case.Hg
    batch = 1 if case.is_varlen else case.dense_batch
    num_states = len(context_lens) if case.is_varlen else batch

    cu_seqlens = None
    if case.is_varlen:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(context_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=dev,
        )

    args = {
        "k": torch.randn((batch, t_flat, hg, case.K), device=dev, dtype=case.dtype),
        "w": torch.randn((batch, h, t_flat, case.K), device=dev, dtype=case.dtype),
        "u": torch.randn((batch, h, t_flat, case.V), device=dev, dtype=case.dtype),
        "g": torch.randn((batch, h, t_flat), device=dev, dtype=torch.float32) * -0.1,
        "initial_state": torch.zeros(
            (num_states, h, case.V, case.K), device=dev, dtype=case.ssm_state_dtype
        ),
        "output_final_state": case.output_final_state,
        "cu_seqlens": cu_seqlens,
        "state_dtype": case.ssm_state_dtype,
        "snapshot_dtype": snapshot_dtype,
        "g_head_major": True,
    }
    total_chunks, max_seq_chunks = chunk_counts(context_lens, batch)
    return args, t_flat, num_states, total_chunks, max_seq_chunks


def bench_us(args, bv: int, warmup: int, iters: int) -> float:
    # The op reads FLYDSL_K5_OPT_BV once at import (_OPT_BV_ENV), so setting the
    # env var here alone would time the rule's BV for every candidate.
    from aiter.ops.flydsl import linear_attention_prefill_kernels as _op

    os.environ["FLYDSL_K5_OPT_BV"] = str(bv)
    _op._OPT_BV_ENV = str(bv)
    try:
        for _ in range(warmup):
            k5(**args)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            k5(**args)
            ends[i].record()
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    finally:
        os.environ.pop("FLYDSL_K5_OPT_BV", None)
        _op._OPT_BV_ENV = None


def lookup_key_from_case(
    case, snapshot_dtype, total_chunks, max_seq_chunks, *, gfx: str, cu_num: int
) -> tuple:
    return (
        gfx,
        cu_num,
        case.H,
        case.Hg,
        case.V,
        case.is_varlen,
        True,
        case.output_final_state,
        snapshot_dtype is torch.bfloat16,
        case.ssm_state_dtype is torch.bfloat16,
        total_chunks,
        max_seq_chunks,
    )


def case_to_tuned_row(
    case,
    snapshot_dtype,
    t_flat,
    n,
    total_chunks,
    max_seq_chunks,
    bv,
    us,
    *,
    gfx: str,
    cu_num: int,
) -> dict[str, Any]:
    return {
        "gfx": gfx,
        "cu_num": int(cu_num),
        "dtype": str(case.dtype),
        "K": int(case.K),
        "V": int(case.V),
        "BT": int(case.BT),
        "H": int(case.H),
        "Hg": int(case.Hg),
        "is_varlen": case.is_varlen,
        "use_h0": True,
        "store_fs": case.output_final_state,
        "snapshot_bf16": snapshot_dtype is torch.bfloat16,
        "state_bf16": case.ssm_state_dtype is torch.bfloat16,
        "T_flat": int(t_flat),
        "N": int(n),
        "total_chunks": int(total_chunks),
        "max_seq_chunks": int(max_seq_chunks),
        "BV": int(bv),
        "us": float(f"{us:.1f}"),
    }


def sweep_case_row(
    case_id,
    case,
    warmup: int,
    iters: int,
    *,
    gfx: str,
    cu_num: int,
    min_gain_pct: float = 0.0,
    below_gain: list | None = None,
) -> dict[str, Any] | None:
    """Time every BV candidate; return the winner's tuned row, or None.

    With ``min_gain_pct`` > 0 a row is only returned when the winner beats
    the CU/LDS rule's BV by at least that much (the case id is appended to
    ``below_gain`` otherwise), so the table only carries real overrides.
    """
    if case.K != 128 or case.V != 128 or case.BT != 64:
        print(f"{case_id:58s} skipped (kernel supports K=V=128, BT=64 only)")
        return None

    snapshot_dtype = case_snapshot_dtype(case)
    inputs, t_flat, n, total_chunks, max_seq_chunks = build_k5_inputs(
        case, snapshot_dtype
    )

    times = {}
    for bv in BV_CANDIDATES:
        if case.V % bv:
            continue
        times[bv] = bench_us(inputs, bv, warmup, iters)

    if not times:
        return None

    best = min(times, key=times.get)
    rule = _hipeq_select_bv(
        torch.device("cuda:0"), case.H, total_chunks, max_seq_chunks
    )
    gain = (times[rule] - times[best]) / times[rule] * 100 if rule in times else 0.0
    cells = " ".join(f"{times.get(bv, float('nan')):8.1f}" for bv in BV_CANDIDATES)
    keep = min_gain_pct <= 0 or gain >= min_gain_pct
    print(
        f"{case_id:58s} {total_chunks:7d} {cells} {best:5d} {rule:5d} {gain:6.1f}"
        + ("" if keep else "  (below --min-gain-pct, not written)")
    )
    if not keep:
        if below_gain is not None:
            below_gain.append(case_id)
        return None

    return case_to_tuned_row(
        case,
        snapshot_dtype,
        t_flat,
        n,
        total_chunks,
        max_seq_chunks,
        best,
        times[best],
        gfx=gfx,
        cu_num=cu_num,
    )


def case_matches_row(case, row: dict[str, Any]) -> bool:
    snapshot_dtype = case_snapshot_dtype(case)
    batch = 1 if case.is_varlen else case.dense_batch
    total_chunks, max_seq_chunks = chunk_counts(case.resolve_context_lens(), batch)
    return (
        case.H == int(row["H"])
        and case.Hg == int(row["Hg"])
        and case.K == int(row["K"])
        and case.V == int(row["V"])
        and case.is_varlen == bool_cell(row["is_varlen"])
        and case.output_final_state == bool_cell(row["store_fs"])
        and (snapshot_dtype is torch.bfloat16) == bool_cell(row["snapshot_bf16"])
        and (case.ssm_state_dtype is torch.bfloat16) == bool_cell(row["state_bf16"])
        and int(row["total_chunks"]) == total_chunks
        and int(row["max_seq_chunks"]) == max_seq_chunks
    )


def find_case_for_row(cases, row: dict[str, Any]):
    for _, case in cases:
        if case_matches_row(case, row):
            return case
    case = case_from_tuned_row(row)
    if case is not None and case_matches_row(case, row):
        return case
    return None


def dataframe_from_cases(
    selected: list[tuple[str, Any]], *, gfx: str, cu_num: int
) -> pd.DataFrame:
    rows = []
    for case_id, case in selected:
        snapshot_dtype = case_snapshot_dtype(case)
        batch = 1 if case.is_varlen else case.dense_batch
        total_chunks, max_seq_chunks = chunk_counts(case.resolve_context_lens(), batch)
        row = case_to_tuned_row(
            case,
            snapshot_dtype,
            sum(case.resolve_context_lens()),
            len(case.resolve_context_lens()) if case.is_varlen else batch,
            total_chunks,
            max_seq_chunks,
            bv=0,
            us=0.0,
            gfx=gfx,
            cu_num=cu_num,
        )
        row["_case_id"] = case_id
        row["BV"] = pd.NA
        row["us"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


class K5BvTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": _DEFAULT_UNTUNED,
        "tune_file": f"{AITER_CONFIG_GDN_K5_OPT}",
        "config_env_name": "AITER_CONFIG_GDN_K5_OPT",
        "warmup": 5,
        "iters": 20,
        "batch": 100,
        "sort": False,
        "min_gain_pct": 0.0,
    }

    def __init__(self):
        super().__init__(
            "chunk_gdn_h_opt_tuned",
            list(LOOKUP_KEYS),
            _RESULT_COLS,
            "FlyDSL K5 opt BV tuner",
        )
        self._cases: list[tuple[str, Any]] = []
        self._case_by_id: dict[str, Any] = {}
        self._below_gain: list[str] = []
        self.run_config_failed = False

    def _clear_op_caches(self):
        from aiter.ops.flydsl import linear_attention_prefill_kernels as _op

        _op.reload_tuned_bv_table()
        _op._get_or_compile_opt.cache_clear()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        try:
            from aiter.jit import core as jit_core

            jit_core.AITER_CONFIGS.get_config_file.cache_clear()
        except ImportError:
            pass
        self._clear_op_caches()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--case",
            nargs="+",
            default=[],
            help="optional regex filters on pytest case ids (after untuned shape filter)",
        )
        self.parser.add_argument(
            "--list-cases",
            action="store_true",
            help="print PrefillGroup case ids and exit",
        )
        self.parser.add_argument(
            "--single-seq-chunks",
            type=parse_chunk_range,
            default=None,
            metavar="LO:HI",
            help="also tune one varlen sequence of every chunk count in LO..HI "
            "(TP1, h0 + final state, bf16 snapshots, fp32 state) for each "
            "varlen shape in -i, e.g. 1:512 for prompts up to 32768 tokens",
        )
        self.parser.add_argument(
            "--min-gain-pct",
            dest="min_gain_pct",
            type=float,
            default=self.get_arg_defaults()["min_gain_pct"],
            help="write a row only when the best BV beats the CU/LDS rule's BV "
            "by at least this percent (default 0: write every tuned row)",
        )

    def pre_process(self, args):
        self._gfx = get_gfx_runtime()
        self._cu_num = get_cu_num()
        if args.all:
            self.get_retune_gemm_list(args)
            return

        untuned_rows = read_csv_rows(args.untune_file)
        cases = load_k5_cases()
        if args.single_seq_chunks:
            cases += single_seq_cases(untuned_rows, args.single_seq_chunks)
        self._cases = select_cases(cases, untuned_rows, args.case)
        self._case_by_id = {case_id: case for case_id, case in self._cases}
        tuned_read = _resolve_tuned_config_for_read(args.tune_file)

        if not self._cases:
            self.untunedf = pd.DataFrame(columns=list(TUNED_COLUMNS) + ["_case_id"])
            self.tunedf = self.get_tuned_gemm_list(tuned_read)
            return

        self.untunedf = dataframe_from_cases(
            self._cases, gfx=self._gfx, cu_num=self._cu_num
        )
        # Catalog and --single-seq-chunks cases can share a lookup key.
        self.untunedf = self.untunedf.drop_duplicates(
            subset=list(self.keys)
        ).reset_index(drop=True)
        self.tunedf = self.get_tuned_gemm_list(tuned_read)

        if self.tunedf is not None and not self.tunedf.empty:
            dedup_cols = [c for c in self.keys if c in self.tunedf.columns]
            if len(dedup_cols) == len(self.keys):
                tuned_keys = set(self.tunedf[dedup_cols].apply(tuple, axis=1))
                mask = self.untunedf[dedup_cols].apply(tuple, axis=1).isin(tuned_keys)
                if mask.any() and args.verbose:
                    print(f"skip {mask.sum()} shapes already present in tuned csv")
                self.untunedf = self.untunedf[~mask].reset_index(drop=True)
                self._cases = [
                    (row["_case_id"], self._case_by_id[row["_case_id"]])
                    for _, row in self.untunedf.iterrows()
                ]

    def tune(self, untunedf, tunedf, args):
        if untunedf.empty:
            return []

        if not hasattr(self, "_printed_tune_header"):
            header = (
                f"{'case':58s} {'chunks':>7s} "
                + " ".join(f"BV{bv:<7d}" for bv in BV_CANDIDATES)
            ) + f" {'best':>5s} {'rule':>5s} {'gain%':>6s}"
            print(header)
            print("-" * len(header))
            self._printed_tune_header = True

        emitted: dict[tuple, dict[str, Any]] = {}
        for _, row in untunedf.iterrows():
            case_id = row["_case_id"]
            case = self._case_by_id[case_id]
            tuned_row = sweep_case_row(
                case_id,
                case,
                args.warmup,
                args.iters,
                gfx=self._gfx,
                cu_num=self._cu_num,
                min_gain_pct=args.min_gain_pct,
                below_gain=self._below_gain,
            )
            if tuned_row is None:
                continue
            snapshot_dtype = case_snapshot_dtype(case)
            batch = 1 if case.is_varlen else case.dense_batch
            total_chunks, max_seq_chunks = chunk_counts(
                case.resolve_context_lens(), batch
            )
            key = lookup_key_from_case(
                case,
                snapshot_dtype,
                total_chunks,
                max_seq_chunks,
                gfx=self._gfx,
                cu_num=self._cu_num,
            )
            emitted[key] = tuned_row
            torch.cuda.empty_cache()

        if not emitted:
            return []
        return pd.DataFrame(list(emitted.values()), columns=self.columns).to_dict(
            "records"
        )

    def post_process(self, results, args, topk=-1, fast_mode=False):
        if isinstance(results, list):
            results = pd.DataFrame(results, columns=self.columns)
        if isinstance(results, pd.DataFrame):
            if results.empty:
                return results
            return (
                results.sort_values("us")
                .drop_duplicates(subset=self.keys, keep="first")
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=self.columns)

    def result_to_csv(self, results, file, concat=False):
        old_tunedf = self.get_tuned_gemm_list(file)
        for col in self.columns:
            if col not in old_tunedf.columns:
                old_tunedf[col] = pd.NA
        resultdf = self.update_tunedf(old_tunedf, results.loc[:, self.columns])
        self.success = pd.concat([self.success, results], ignore_index=True)
        if results is not None and not results.empty:
            resultdf = resultdf.astype(str).drop_duplicates(
                subset=self.keys, keep="last"
            )
        ordered_cols = [c for c in _CSV_COLUMN_ORDER if c in resultdf.columns]
        ordered_cols.extend(c for c in resultdf.columns if c not in ordered_cols)
        resultdf = resultdf[ordered_cols]
        resultdf.to_csv(file, index=False)

    def run_config(self, args):
        tol = _RUN_CONFIG_TOL_PCT
        cases = load_k5_cases()
        results = []
        print("Shape | e2e_us | Status")
        print("-" * 60)

        for _, row in self.untunedf.iterrows():
            row_dict = row.to_dict()
            case = find_case_for_row(cases, row_dict)
            label = f"H={row_dict['H']}/Hg={row_dict['Hg']}"
            shape = f"({label}, tc={row_dict['total_chunks']}, BV={row_dict['BV']})"
            if case is None:
                print(f"{shape} | {'-1':>10} | ERROR")
                print("reason: no matching K5 prefill case")
                results.append({"shape": shape, "us": -1.0, "status": "error:no case"})
                continue

            snapshot_dtype = case_snapshot_dtype(case)
            inputs, *_rest = build_k5_inputs(case, snapshot_dtype)
            bv = int(row_dict["BV"])
            csv_us = float(row_dict["us"])
            try:
                live_us = bench_us(inputs, bv, args.warmup, args.iters)
                delta = (live_us - csv_us) / csv_us * 100 if csv_us > 0 else 0.0
                if abs(delta) <= max(tol, 0.0):
                    status = "ok"
                    print(f"{shape} | {live_us:>10.1f} | OK")
                else:
                    status = (
                        f"mismatch: live_us drift {delta:.1f}% vs csv (tol {tol:.1f}%)"
                    )
                    print(f"{shape} | {live_us:>10.1f} | MISMATCH")
                    print(f"reason: {status[len('mismatch:'):].strip()}")
            except Exception as exc:  # noqa: BLE001
                live_us = -1.0
                status = f"error: {exc}"
                print(f"{shape} | {'-1':>10} | ERROR")
                print(f"reason: {exc}")
            results.append({"shape": shape, "us": live_us, "status": status})
            del inputs
            torch.cuda.empty_cache()
        return results

    def run(self, args, fast_mode=False):
        if args.list_cases:
            for case_id, _ in load_k5_cases():
                print(case_id)
            return pd.DataFrame()

        self.pre_process(args)

        run_config_file = args.run_config if isinstance(args.run_config, str) else None
        if args.run_config and run_config_file:
            tunedf = self.get_tuned_gemm_list(run_config_file)
            if not tunedf.empty and self.keys[0] in tunedf.columns:
                self.untunedf = tunedf.drop_duplicates(subset=self.keys).reset_index(
                    drop=True
                )

        if args.run_config:
            if self.untunedf.empty:
                print("No shapes to benchmark, nothing to run")
                return pd.DataFrame()

            env_name = self.get_arg_defaults().get("config_env_name")
            if run_config_file:
                old_val, old_rebuild = self._set_config_env_for_run_config(
                    args, config_file=run_config_file
                )
                try:
                    results = self.run_config(args)
                finally:
                    self._restore_config_env(env_name, old_val, old_rebuild)
            else:
                results = self.run_config(args)

            self.run_config_failed = any(
                not str(r.get("status", "")).startswith("ok") for r in results
            )
            return self.tunedf if self.tunedf is not None else pd.DataFrame()

        if hasattr(self, "_printed_tune_header"):
            del self._printed_tune_header
        out = super().run(args, fast_mode=fast_mode)
        if not self.untunedf.empty:
            print(f"\n{len(self.success)} tuned rows")
        return out

    def tune_summary(self, status):
        tuning_time = round(time.time() - getattr(self, "tune_start_time", 0), 4)
        logger.info("============= Tuning results Summary: ==============")
        logger.info(
            f"Tuning {status}. tune {len(self.success)} shapes, "
            f"total tuning time is {tuning_time} seconds"
        )
        if not self.success.empty:
            logger.info("Successfully tuned shapes:")
            print(self.success, flush=True)
        if not self.failed.empty:
            logger.info("Failed shapes:")
            print(self.failed, flush=True)
            sys.exit(1)
        if self._below_gain:
            logger.info(
                f"{len(self._below_gain)} shapes not written: the CU/LDS rule's BV "
                "is within --min-gain-pct of the best"
            )
        if (
            self.success.empty
            and not self.untunedf.empty
            and len(self._below_gain) < len(self.untunedf)
        ):
            logger.error("\033[91m[Tuning not Finished]\033[0m no shapes were tuned")
            sys.exit(1)

    def getKernelName(self, kernel_id):
        return f"BV{kernel_id}"

    def calculate(self, results, inbpe=2, outbpe=2):
        return 0, 0

    def result_to_df(self, rets):
        if isinstance(rets, pd.DataFrame):
            return rets
        return pd.DataFrame(columns=self.columns)


if __name__ == "__main__":
    tuner = K5BvTuner()
    args = tuner.parse_args()
    tuner.run(args, fast_mode=False)
    if args.run_config and tuner.run_config_failed:
        sys.exit(1)
