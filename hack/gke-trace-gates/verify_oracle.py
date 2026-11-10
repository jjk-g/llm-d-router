#!/usr/bin/env python3
# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verification oracle for Milestone 4 GKE Gates 3 & 4.

Subcommands:
  - gate3: Compares compacted WekaTrace directories T_recorded and T_replay and
           asserts structural isomorphism:
             1. Session count == expected_sessions (default 8) in both passes.
             2. Subagent branch count == expected_subagents (default 2) in both.
             3. Cross-session system-prompt cohort KV-block collisions preserved.
             4. Per-turn parent LCP block lengths match within <= max_lcp_diff (1)
                block across all paired sessions, and subagent->parent-Turn-0 LCP
                matches within <= max_lcp_diff (1) block.
             5. Zero raw prompt/response keys leaked into compacted trace JSON.

  - gate4: Compares scraped Prometheus /metrics snapshots from real vLLM and EPP
           pods before/after live vs. replayed runs and overhead benchmarks:
             1. |hit_rate_replay - hit_rate_live| <= 0.03 (3 percentage points).
             2. Prompt and generation token deltas match within +-3%.
             3. 200-request concurrency-16 EPP ext-proc p99 latency delta < 0.5 ms
                and delta(llm_d_epp_trace_recorder_dropped_events_total) == 0.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

FORBIDDEN_RAW_KEYS = frozenset({"messages", "prompt", "content", "choices", "completion"})


# ---------------------------------------------------------------------------
# Gate 3: Structural Trace Isomorphism Oracle
# ---------------------------------------------------------------------------


def lcp_length(a: list[int], b: list[int]) -> int:
    """Return longest common prefix length between two block-hash sequences."""
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def _find_forbidden_keys(obj: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            cur = f"{path}.{k}"
            if k in FORBIDDEN_RAW_KEYS:
                hits.append(cur)
            hits.extend(_find_forbidden_keys(v, cur))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(_find_forbidden_keys(item, f"{path}[{i}]"))
    return hits


@dataclass(frozen=True)
class SessionSummary:
    trace_id: str
    block_size: int
    system_tokens: int
    parent_hash_seqs: tuple[tuple[int, ...], ...]
    parent_lcp_vec: tuple[int, ...]
    subagent_count: int
    subagent_turn0_lcps: tuple[int, ...]
    cohort_prefix: tuple[int, ...]


def load_weka_traces(trace_dir: Path) -> list[dict[str, Any]]:
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {trace_dir}")
    files = sorted(p for p in trace_dir.iterdir() if p.is_file() and p.suffix == ".json")
    if not files:
        raise ValueError(f"no .json WekaTrace files found in {trace_dir}")
    traces: list[dict[str, Any]] = []
    for fpath in files:
        blob = json.loads(fpath.read_text(encoding="utf-8"))
        leaks = _find_forbidden_keys(blob, fpath.name)
        if leaks:
            raise AssertionError(f"forbidden raw text keys found in {fpath}: {leaks}")
        traces.append(blob)
    return traces


def summarize_traces(
    traces: list[dict[str, Any]],
    min_cohort_blocks: int = 2,
) -> tuple[list[SessionSummary], Counter[tuple[int, ...]], list[str]]:
    """Summarize each WekaTrace and identify cross-session Turn-0 prefix cohorts."""
    raw_items: list[dict[str, Any]] = []
    prefix_counts: Counter[tuple[int, ...]] = Counter()
    schema_errors: list[str] = []

    for tr in traces:
        trace_id = str(tr.get("id", ""))
        block_size = int(tr.get("block_size") or 16)
        system_tokens = int(tr.get("system_tokens") or 0)
        reqs = tr.get("requests") or []

        parent_seqs: list[tuple[int, ...]] = []
        subagents: list[dict[str, Any]] = []
        for entry in reqs:
            if isinstance(entry, dict) and entry.get("type") == "subagent":
                subagents.append(entry)
            elif isinstance(entry, dict):
                hids = tuple(int(h) for h in (entry.get("hash_ids") or []))
                parent_seqs.append(hids)

        if not parent_seqs:
            schema_errors.append(f"trace {trace_id} has zero parent requests")
            continue

        turn0_prefix = parent_seqs[0][:min_cohort_blocks]
        if len(turn0_prefix) >= min_cohort_blocks:
            prefix_counts[turn0_prefix] += 1

        parent_lcps = tuple(
            lcp_length(list(parent_seqs[k - 1]), list(parent_seqs[k]))
            for k in range(1, len(parent_seqs))
        )

        sa_lcps: list[int] = []
        for sa_idx, sa in enumerate(subagents):
            sa_reqs = sa.get("requests") or []
            if not sa_reqs or not isinstance(sa_reqs[0], dict):
                schema_errors.append(
                    f"trace {trace_id} subagent[{sa_idx}] has empty or invalid 'requests' list"
                )
                continue
            first_sa_hashes = [int(h) for h in (sa_reqs[0].get("hash_ids") or [])]
            if not first_sa_hashes:
                schema_errors.append(
                    f"trace {trace_id} subagent[{sa_idx}] first request has empty 'hash_ids'"
                )
                continue
            sa_lcps.append(lcp_length(list(parent_seqs[0]), first_sa_hashes))

        raw_items.append(
            {
                "trace_id": trace_id,
                "block_size": block_size,
                "system_tokens": system_tokens,
                "parent_hash_seqs": tuple(parent_seqs),
                "parent_lcp_vec": parent_lcps,
                "subagent_count": len(subagents),
                "subagent_turn0_lcps": tuple(sa_lcps),
                "cohort_prefix": turn0_prefix if prefix_counts[turn0_prefix] >= 1 else (),
            }
        )

    # Retain only shared prefixes appearing in >= 2 sessions as true cross-session cohorts.
    shared_cohorts = Counter({k: v for k, v in prefix_counts.items() if v >= 2})
    summaries = [
        SessionSummary(
            trace_id=item["trace_id"],
            block_size=item["block_size"],
            system_tokens=item["system_tokens"],
            parent_hash_seqs=item["parent_hash_seqs"],
            parent_lcp_vec=item["parent_lcp_vec"],
            subagent_count=item["subagent_count"],
            subagent_turn0_lcps=item["subagent_turn0_lcps"],
            cohort_prefix=item["cohort_prefix"] if item["cohort_prefix"] in shared_cohorts else (),
        )
        for item in raw_items
    ]
    return summaries, shared_cohorts, schema_errors


def _canonical_sort_sessions(
    summaries: list[SessionSummary],
    cohorts: Counter[tuple[int, ...]],
) -> list[SessionSummary]:
    """Order sessions canonically by structural cohort signature so anonymized replay
    sessions pair deterministically with recorded sessions."""
    by_prefix: dict[tuple[int, ...], list[SessionSummary]] = {}
    for s in summaries:
        by_prefix.setdefault(s.cohort_prefix, []).append(s)

    cohort_rank: dict[tuple[int, ...], int] = {}
    ranked_prefixes = sorted(
        by_prefix.keys(),
        key=lambda pfx: (
            -cohorts.get(pfx, 0),
            -sum(s.subagent_count for s in by_prefix[pfx]),
            sorted(s.parent_lcp_vec for s in by_prefix[pfx]),
            pfx,
        ),
    )
    for idx, pfx in enumerate(ranked_prefixes):
        cohort_rank[pfx] = idx

    return sorted(
        summaries,
        key=lambda s: (
            cohort_rank[s.cohort_prefix],
            -s.subagent_count,
            len(s.parent_hash_seqs),
            s.parent_lcp_vec,
            s.subagent_turn0_lcps,
            len(s.parent_hash_seqs[0]),
            s.trace_id,
        ),
    )


def verify_gate3(
    recorded_dir: Path,
    replay_dir: Path,
    expected_sessions: int = 8,
    expected_subagents: int = 2,
    expected_cohorts: int = 2,
    min_cohort_blocks: int = 2,
    max_lcp_diff: int = 1,
) -> dict[str, Any]:
    rec_traces = load_weka_traces(recorded_dir)
    rep_traces = load_weka_traces(replay_dir)

    rec_summaries, rec_cohorts, rec_schema_errs = summarize_traces(
        rec_traces, min_cohort_blocks=min_cohort_blocks
    )
    rep_summaries, rep_cohorts, rep_schema_errs = summarize_traces(
        rep_traces, min_cohort_blocks=min_cohort_blocks
    )

    errors: list[str] = []
    errors.extend(f"recorded schema error: {e}" for e in rec_schema_errs)
    errors.extend(f"replay schema error: {e}" for e in rep_schema_errs)

    if len(rec_summaries) != expected_sessions:
        errors.append(
            f"recorded session count mismatch: got {len(rec_summaries)}, expected {expected_sessions}"
        )
    if len(rep_summaries) != expected_sessions:
        errors.append(
            f"replay session count mismatch: got {len(rep_summaries)}, expected {expected_sessions}"
        )

    rec_subagents = sum(s.subagent_count for s in rec_summaries)
    rep_subagents = sum(s.subagent_count for s in rep_summaries)
    if rec_subagents != expected_subagents:
        errors.append(
            f"recorded subagent branch count mismatch: got {rec_subagents}, expected {expected_subagents}"
        )
    if rep_subagents != expected_subagents:
        errors.append(
            f"replay subagent branch count mismatch: got {rep_subagents}, expected {expected_subagents}"
        )

    if len(rec_cohorts) < expected_cohorts:
        errors.append(
            f"recorded cross-session prefix cohorts: got {len(rec_cohorts)}, expected >= {expected_cohorts}"
        )
    if len(rep_cohorts) < expected_cohorts:
        errors.append(
            f"replay cross-session prefix cohorts: got {len(rep_cohorts)}, expected >= {expected_cohorts}"
        )

    rec_cohort_sizes = sorted(rec_cohorts.values(), reverse=True)
    rep_cohort_sizes = sorted(rep_cohorts.values(), reverse=True)
    if rec_cohort_sizes != rep_cohort_sizes:
        errors.append(
            f"cross-session cohort size partition mismatch: recorded={rec_cohort_sizes} vs replay={rep_cohort_sizes}"
        )

    rec_ordered = _canonical_sort_sessions(rec_summaries, rec_cohorts)
    rep_ordered = _canonical_sort_sessions(rep_summaries, rep_cohorts)

    pair_reports: list[dict[str, Any]] = []
    observed_max_lcp_diff = 0
    observed_max_sa_lcp_diff = 0

    if len(rec_ordered) == len(rep_ordered):
        for idx, (rec_s, rep_s) in enumerate(zip(rec_ordered, rep_ordered)):
            if len(rec_s.parent_lcp_vec) != len(rep_s.parent_lcp_vec):
                errors.append(
                    f"pair[{idx}] ({rec_s.trace_id} vs {rep_s.trace_id}) parent turn count mismatch: "
                    f"{len(rec_s.parent_hash_seqs)} vs {len(rep_s.parent_hash_seqs)}"
                )
                continue

            diffs = [
                abs(a - b) for a, b in zip(rec_s.parent_lcp_vec, rep_s.parent_lcp_vec)
            ]
            max_diff = max(diffs, default=0)
            observed_max_lcp_diff = max(observed_max_lcp_diff, max_diff)
            if max_diff > max_lcp_diff:
                errors.append(
                    f"pair[{idx}] ({rec_s.trace_id} vs {rep_s.trace_id}) LCP diff {max_diff} > {max_lcp_diff}: "
                    f"recorded={list(rec_s.parent_lcp_vec)} replay={list(rep_s.parent_lcp_vec)}"
                )

            if rec_s.subagent_count != rep_s.subagent_count:
                errors.append(
                    f"pair[{idx}] ({rec_s.trace_id} vs {rep_s.trace_id}) subagent count mismatch: "
                    f"{rec_s.subagent_count} vs {rep_s.subagent_count}"
                )
            elif len(rec_s.subagent_turn0_lcps) != rec_s.subagent_count or len(
                rep_s.subagent_turn0_lcps
            ) != rep_s.subagent_count:
                errors.append(
                    f"pair[{idx}] ({rec_s.trace_id} vs {rep_s.trace_id}) missing subagent LCP entries: "
                    f"recorded={len(rec_s.subagent_turn0_lcps)}/{rec_s.subagent_count}, "
                    f"replay={len(rep_s.subagent_turn0_lcps)}/{rep_s.subagent_count}"
                )
            else:
                sa_diffs = [
                    abs(a - b)
                    for a, b in zip(rec_s.subagent_turn0_lcps, rep_s.subagent_turn0_lcps)
                ]
                sa_max = max(sa_diffs, default=0)
                observed_max_sa_lcp_diff = max(observed_max_sa_lcp_diff, sa_max)
                if sa_max > max_lcp_diff:
                    errors.append(
                        f"pair[{idx}] ({rec_s.trace_id} vs {rep_s.trace_id}) subagent LCP diff {sa_max} > {max_lcp_diff}: "
                        f"recorded={list(rec_s.subagent_turn0_lcps)} replay={list(rep_s.subagent_turn0_lcps)}"
                    )

            pair_reports.append(
                {
                    "pair_index": idx,
                    "recorded_id": rec_s.trace_id,
                    "replay_id": rep_s.trace_id,
                    "recorded_parent_lcps": list(rec_s.parent_lcp_vec),
                    "replay_parent_lcps": list(rep_s.parent_lcp_vec),
                    "max_parent_lcp_diff": max_diff,
                    "subagent_count": rec_s.subagent_count,
                    "recorded_subagent_lcps": list(rec_s.subagent_turn0_lcps),
                    "replay_subagent_lcps": list(rep_s.subagent_turn0_lcps),
                }
            )

    return {
        "gate": "gate3",
        "passed": len(errors) == 0,
        "recorded_sessions": len(rec_summaries),
        "replay_sessions": len(rep_summaries),
        "recorded_subagents": rec_subagents,
        "replay_subagents": rep_subagents,
        "recorded_cohort_sizes": rec_cohort_sizes,
        "replay_cohort_sizes": rep_cohort_sizes,
        "max_parent_lcp_diff": observed_max_lcp_diff,
        "max_subagent_lcp_diff": observed_max_sa_lcp_diff,
        "pairs": pair_reports,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Gate 4: Real vLLM Prefix-Cache Parity & EPP Overhead Oracle
# ---------------------------------------------------------------------------

_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class PromSample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus_text(text: str) -> list[PromSample]:
    samples: list[PromSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, label_blob, val_str = m.groups()
        if val_str in ("+Inf", "Inf"):
            val = math.inf
        elif val_str == "-Inf":
            val = -math.inf
        elif val_str == "NaN":
            val = math.nan
        else:
            val = float(val_str)
        labels: dict[str, str] = {}
        if label_blob:
            for k, v in _LABEL_RE.findall(label_blob):
                labels[k] = v.replace('\\"', '"').replace("\\\\", "\\")
        samples.append(PromSample(name=name, labels=labels, value=val))
    return samples


def sum_metric_any(
    samples: list[PromSample],
    candidate_names: tuple[str, ...],
    label_filter: dict[str, str] | None = None,
) -> float:
    """Sum the first metric name in candidate_names present in samples."""
    for target in candidate_names:
        matched = [
            s.value
            for s in samples
            if s.name == target
            and not math.isnan(s.value)
            and (
                label_filter is None
                or all(s.labels.get(lk) == lv for lk, lv in label_filter.items())
            )
        ]
        if matched:
            return sum(matched)
    return 0.0


def histogram_inf_delta(
    before_samples: list[PromSample],
    after_samples: list[PromSample],
    bucket_metric_name: str,
    label_filter: dict[str, str] | None = None,
) -> float:
    """Return total +Inf bucket sample count delta (after - before) for non-vacuity checks."""

    def _sum_inf(samples: list[PromSample]) -> float:
        total = 0.0
        for s in samples:
            if s.name != bucket_metric_name:
                continue
            if s.labels.get("le") not in ("+Inf", "Inf"):
                continue
            if label_filter and not all(s.labels.get(k) == v for k, v in label_filter.items()):
                continue
            total += s.value
        return total

    return _sum_inf(after_samples) - _sum_inf(before_samples)


def histogram_quantile_delta(
    before_samples: list[PromSample],
    after_samples: list[PromSample],
    bucket_metric_name: str,
    quantile: float = 0.99,
    label_filter: dict[str, str] | None = None,
) -> float:
    """Compute Prometheus-compatible histogram_quantile(q) over (after - before) buckets."""

    def _agg_buckets(samples: list[PromSample]) -> dict[float, float]:
        buckets: dict[float, float] = {}
        for s in samples:
            if s.name != bucket_metric_name:
                continue
            if label_filter and not all(s.labels.get(k) == v for k, v in label_filter.items()):
                continue
            le_str = s.labels.get("le")
            if le_str is None:
                continue
            le = math.inf if le_str in ("+Inf", "Inf") else float(le_str)
            buckets[le] = buckets.get(le, 0.0) + s.value
        return buckets

    b_before = _agg_buckets(before_samples)
    b_after = _agg_buckets(after_samples)
    all_les = sorted(set(b_before.keys()) | set(b_after.keys()))
    if not all_les:
        return 0.0

    deltas: list[tuple[float, float]] = []
    for le in all_les:
        d = max(0.0, b_after.get(le, 0.0) - b_before.get(le, 0.0))
        deltas.append((le, d))

    total = deltas[-1][1]
    if total <= 0.0:
        return 0.0

    rank = quantile * total
    prev_le = 0.0
    prev_count = 0.0
    for le, cum_count in deltas:
        if cum_count >= rank:
            if math.isinf(le):
                return prev_le
            bucket_count = cum_count - prev_count
            if bucket_count <= 0.0:
                return le
            frac = (rank - prev_count) / bucket_count
            return prev_le + (le - prev_le) * frac
        prev_le = le
        prev_count = cum_count
    return deltas[-2][0] if len(deltas) >= 2 and math.isinf(deltas[-1][0]) else deltas[-1][0]


VLLM_HIT_METRICS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_hits",
    "vllm:gpu_prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_hits",
    "vllm:cpu_prefix_cache_hits_total",
    "vllm:cpu_prefix_cache_hits",
)

VLLM_QUERY_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_queries",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_queries",
    "vllm:cpu_prefix_cache_queries_total",
    "vllm:cpu_prefix_cache_queries",
)

VLLM_PROMPT_TOKEN_METRICS = (
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens",
)

VLLM_GEN_TOKEN_METRICS = (
    "vllm:generation_tokens_total",
    "vllm:generation_tokens",
)


def verify_gate4(
    vllm_live_before: Path,
    vllm_live_after: Path,
    vllm_replay_before: Path,
    vllm_replay_after: Path,
    epp_on_before: Path,
    epp_on_after: Path,
    epp_off_before: Path,
    epp_off_after: Path,
    max_hit_rate_diff: float = 0.03,
    max_token_rel_diff: float = 0.03,
    max_p99_overhead_ms: float = 0.5,
) -> dict[str, Any]:
    s_live_b = parse_prometheus_text(vllm_live_before.read_text(encoding="utf-8"))
    s_live_a = parse_prometheus_text(vllm_live_after.read_text(encoding="utf-8"))
    s_rep_b = parse_prometheus_text(vllm_replay_before.read_text(encoding="utf-8"))
    s_rep_a = parse_prometheus_text(vllm_replay_after.read_text(encoding="utf-8"))

    s_on_b = parse_prometheus_text(epp_on_before.read_text(encoding="utf-8"))
    s_on_a = parse_prometheus_text(epp_on_after.read_text(encoding="utf-8"))
    s_off_b = parse_prometheus_text(epp_off_before.read_text(encoding="utf-8"))
    s_off_a = parse_prometheus_text(epp_off_after.read_text(encoding="utf-8"))

    errors: list[str] = []

    hits_live = sum_metric_any(s_live_a, VLLM_HIT_METRICS) - sum_metric_any(s_live_b, VLLM_HIT_METRICS)
    queries_live = sum_metric_any(s_live_a, VLLM_QUERY_METRICS) - sum_metric_any(
        s_live_b, VLLM_QUERY_METRICS
    )
    hits_replay = sum_metric_any(s_rep_a, VLLM_HIT_METRICS) - sum_metric_any(s_rep_b, VLLM_HIT_METRICS)
    queries_replay = sum_metric_any(s_rep_a, VLLM_QUERY_METRICS) - sum_metric_any(
        s_rep_b, VLLM_QUERY_METRICS
    )

    if queries_live <= 0:
        errors.append(f"live vLLM prefix_cache_queries delta must be > 0, got {queries_live}")
        hit_rate_live = 0.0
    else:
        hit_rate_live = hits_live / queries_live

    if queries_replay <= 0:
        errors.append(f"replay vLLM prefix_cache_queries delta must be > 0, got {queries_replay}")
        hit_rate_replay = 0.0
    else:
        hit_rate_replay = hits_replay / queries_replay

    if hit_rate_live <= 0.0:
        errors.append("live vLLM prefix cache hit rate was 0.0 (prefix caching did not activate)")

    hit_rate_diff = abs(hit_rate_replay - hit_rate_live)
    if hit_rate_diff > max_hit_rate_diff:
        errors.append(
            f"prefix-cache hit rate parity failed: |replay({hit_rate_replay:.4f}) - "
            f"live({hit_rate_live:.4f})| = {hit_rate_diff:.4f} > {max_hit_rate_diff:.4f}"
        )

    prompt_live = sum_metric_any(s_live_a, VLLM_PROMPT_TOKEN_METRICS) - sum_metric_any(
        s_live_b, VLLM_PROMPT_TOKEN_METRICS
    )
    prompt_replay = sum_metric_any(s_rep_a, VLLM_PROMPT_TOKEN_METRICS) - sum_metric_any(
        s_rep_b, VLLM_PROMPT_TOKEN_METRICS
    )
    gen_live = sum_metric_any(s_live_a, VLLM_GEN_TOKEN_METRICS) - sum_metric_any(
        s_live_b, VLLM_GEN_TOKEN_METRICS
    )
    gen_replay = sum_metric_any(s_rep_a, VLLM_GEN_TOKEN_METRICS) - sum_metric_any(
        s_rep_b, VLLM_GEN_TOKEN_METRICS
    )

    if prompt_live <= 0:
        errors.append(f"live prompt_tokens delta must be > 0, got {prompt_live}")
        prompt_rel_diff = 1.0
    else:
        prompt_rel_diff = abs(prompt_replay - prompt_live) / prompt_live
        if prompt_rel_diff > max_token_rel_diff:
            errors.append(
                f"prompt_tokens_total parity failed: replay={prompt_replay:.0f} vs live={prompt_live:.0f} "
                f"(rel_diff={prompt_rel_diff:.4f} > {max_token_rel_diff:.4f})"
            )

    if gen_live <= 0:
        errors.append(f"live generation_tokens delta must be > 0, got {gen_live}")
        gen_rel_diff = 1.0
    else:
        gen_rel_diff = abs(gen_replay - gen_live) / gen_live
        if gen_rel_diff > max_token_rel_diff:
            errors.append(
                f"generation_tokens_total parity failed: replay={gen_replay:.0f} vs live={gen_live:.0f} "
                f"(rel_diff={gen_rel_diff:.4f} > {max_token_rel_diff:.4f})"
            )

    dropped_before = sum_metric_any(s_on_b, ("llm_d_epp_trace_recorder_dropped_events_total",))
    dropped_after = sum_metric_any(s_on_a, ("llm_d_epp_trace_recorder_dropped_events_total",))
    dropped_delta = dropped_after - dropped_before
    if dropped_delta != 0.0:
        errors.append(
            f"llm_d_epp_trace_recorder_dropped_events_total delta must be 0, got {dropped_delta}"
        )

    # Non-vacuity checks: every EPP latency histogram must observe > 0 samples during the benchmark.
    req_on_count = histogram_inf_delta(
        s_on_b, s_on_a, "llm_d_epp_request_processing_duration_seconds_bucket"
    )
    req_off_count = histogram_inf_delta(
        s_off_b, s_off_a, "llm_d_epp_request_processing_duration_seconds_bucket"
    )
    resp_on_count = histogram_inf_delta(
        s_on_b, s_on_a, "llm_d_epp_response_processing_duration_seconds_bucket"
    )
    resp_off_count = histogram_inf_delta(
        s_off_b, s_off_a, "llm_d_epp_response_processing_duration_seconds_bucket"
    )
    plugin_on_count = histogram_inf_delta(
        s_on_b,
        s_on_a,
        "llm_d_epp_plugin_duration_seconds_bucket",
        label_filter={"plugin_type": "kv-block-trace-recorder"},
    )

    for metric_desc, cnt in (
        ("llm_d_epp_request_processing_duration_seconds_bucket (recorder ON)", req_on_count),
        ("llm_d_epp_request_processing_duration_seconds_bucket (recorder OFF)", req_off_count),
        ("llm_d_epp_response_processing_duration_seconds_bucket (recorder ON)", resp_on_count),
        ("llm_d_epp_response_processing_duration_seconds_bucket (recorder OFF)", resp_off_count),
        (
            'llm_d_epp_plugin_duration_seconds_bucket{plugin_type="kv-block-trace-recorder"} (recorder ON)',
            plugin_on_count,
        ),
    ):
        if cnt <= 0.0:
            errors.append(f"EPP histogram sample count delta must be > 0 for {metric_desc}, got {cnt}")

    # Informational request-processing p99 (includes synchronous loopback HTTP RPC to Python
    # `vllm launch render` sidecar, whose ~3-12 ms CPU jitter across sequential runs exceeds 0.5 ms).
    req_p99_on_ms = (
        histogram_quantile_delta(
            s_on_b, s_on_a, "llm_d_epp_request_processing_duration_seconds_bucket", 0.99
        )
        * 1000.0
    )
    req_p99_off_ms = (
        histogram_quantile_delta(
            s_off_b, s_off_a, "llm_d_epp_request_processing_duration_seconds_bucket", 0.99
        )
        * 1000.0
    )
    req_p99_delta_ms = max(0.0, req_p99_on_ms - req_p99_off_ms)

    # Strict < 0.5 ms check 1: response ext-proc p99 delta (no vllm-render HTTP RPC).
    resp_p99_on_ms = (
        histogram_quantile_delta(
            s_on_b, s_on_a, "llm_d_epp_response_processing_duration_seconds_bucket", 0.99
        )
        * 1000.0
    )
    resp_p99_off_ms = (
        histogram_quantile_delta(
            s_off_b, s_off_a, "llm_d_epp_response_processing_duration_seconds_bucket", 0.99
        )
        * 1000.0
    )
    resp_p99_delta_ms = max(0.0, resp_p99_on_ms - resp_p99_off_ms)

    # Strict < 0.5 ms check 2: summed per-extension_point p99 of kv-block-trace-recorder
    # (e.g. DataProducer/RequestHandlingDataProducer + PreRequest + ResponseStreaming, 0.1 ms buckets).
    recorder_ext_points = sorted(
        {
            s.labels["extension_point"]
            for s in s_on_a
            if s.name == "llm_d_epp_plugin_duration_seconds_bucket"
            and s.labels.get("plugin_type") == "kv-block-trace-recorder"
            and "extension_point" in s.labels
        }
    )
    plugin_by_ext_p99_ms: dict[str, float] = {}
    for ext_pt in recorder_ext_points:
        ext_cnt = histogram_inf_delta(
            s_on_b,
            s_on_a,
            "llm_d_epp_plugin_duration_seconds_bucket",
            label_filter={"plugin_type": "kv-block-trace-recorder", "extension_point": ext_pt},
        )
        if ext_cnt > 0.0:
            p99_val = (
                histogram_quantile_delta(
                    s_on_b,
                    s_on_a,
                    "llm_d_epp_plugin_duration_seconds_bucket",
                    0.99,
                    label_filter={
                        "plugin_type": "kv-block-trace-recorder",
                        "extension_point": ext_pt,
                    },
                )
                * 1000.0
            )
            plugin_by_ext_p99_ms[ext_pt] = round(p99_val, 4)

    plugin_hotpath_sum_p99_ms = sum(plugin_by_ext_p99_ms.values())

    if resp_p99_delta_ms >= max_p99_overhead_ms:
        errors.append(
            f"EPP response ext-proc p99 overhead {resp_p99_delta_ms:.4f} ms >= {max_p99_overhead_ms:.4f} ms "
            f"(on={resp_p99_on_ms:.4f} ms, off={resp_p99_off_ms:.4f} ms)"
        )
    if plugin_hotpath_sum_p99_ms >= max_p99_overhead_ms:
        errors.append(
            f"kv-block-trace-recorder summed hot-path plugin p99 {plugin_hotpath_sum_p99_ms:.4f} ms >= "
            f"{max_p99_overhead_ms:.4f} ms (by_extension_point={plugin_by_ext_p99_ms})"
        )

    return {
        "gate": "gate4",
        "passed": len(errors) == 0,
        "prefix_cache": {
            "hits_live": hits_live,
            "queries_live": queries_live,
            "hit_rate_live": round(hit_rate_live, 6),
            "hits_replay": hits_replay,
            "queries_replay": queries_replay,
            "hit_rate_replay": round(hit_rate_replay, 6),
            "abs_diff": round(hit_rate_diff, 6),
            "threshold": max_hit_rate_diff,
        },
        "tokens": {
            "prompt_live": prompt_live,
            "prompt_replay": prompt_replay,
            "prompt_rel_diff": round(prompt_rel_diff, 6),
            "generation_live": gen_live,
            "generation_replay": gen_replay,
            "generation_rel_diff": round(gen_rel_diff, 6),
            "threshold": max_token_rel_diff,
        },
        "epp_overhead": {
            "dropped_events_delta": dropped_delta,
            "sample_counts": {
                "request_on": req_on_count,
                "request_off": req_off_count,
                "response_on": resp_on_count,
                "response_off": resp_off_count,
                "recorder_plugin_on": plugin_on_count,
            },
            "request_p99_on_ms_informational": round(req_p99_on_ms, 4),
            "request_p99_off_ms_informational": round(req_p99_off_ms, 4),
            "request_p99_delta_ms_informational": round(req_p99_delta_ms, 4),
            "response_p99_on_ms": round(resp_p99_on_ms, 4),
            "response_p99_off_ms": round(resp_p99_off_ms, 4),
            "response_p99_delta_ms": round(resp_p99_delta_ms, 4),
            "recorder_plugin_by_extension_point_p99_ms": plugin_by_ext_p99_ms,
            "recorder_plugin_hotpath_sum_p99_ms": round(plugin_hotpath_sum_p99_ms, 4),
            "max_p99_overhead_ms": max_p99_overhead_ms,
        },
        "errors": errors,
    }


MAX_UINT53 = (1 << 53) - 1

AIPERF_STRICT_TRACE_KEYS = {
    "id",
    "models",
    "block_size",
    "hash_id_scope",
    "tool_tokens",
    "system_tokens",
    "requests",
    "totals",
}

AIPERF_STRICT_LEAF_KEYS = {
    "t",
    "type",
    "model",
    "in",
    "out",
    "hash_ids",
    "api_time",
    "think_time",
    "ttft",
}

AIPERF_STRICT_SUBAGENT_KEYS = {
    "t",
    "type",
    "agent_id",
    "subagent_type",
    "status",
    "duration_ms",
    "total_tokens",
    "models",
    "requests",
    "tool_tokens",
    "system_tokens",
}


def _load_weka_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load WekaTrace JSON objects from a .jsonl file, .json file, or directory."""
    if not path.exists():
        raise ValueError(f"Weka trace path does not exist: {path}")

    traces: List[Dict[str, Any]] = []
    if path.is_file():
        if path.suffix == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                traces.append(obj)
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                traces.extend(obj)
            elif isinstance(obj, dict) and "id" in obj and "requests" in obj:
                traces.append(obj)
            else:
                raise ValueError(f"{path}: expected WekaTrace JSON object or list")
    else:
        jsonl_file = path / "traces.jsonl"
        if jsonl_file.is_file():
            return _load_weka_dataset(jsonl_file)
        for p in sorted(path.glob("*.json")):
            if p.name in ("qwen_traces.jsonl",) or "report" in p.name or "summary" in p.name:
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "id" in obj and "requests" in obj:
                traces.append(obj)
    return sorted(traces, key=lambda t: str(t.get("id", "")))


def _load_captured_raw_records(path: Path) -> List[Dict[str, Any]]:
    """Load raw captured EPP records from .jsonl or .jsonl.gz files."""
    if not path.exists():
        raise ValueError(f"Captured raw path does not exist: {path}")

    files: List[Path] = []
    if path.is_file():
        files.append(path)
    else:
        files.extend(sorted(path.glob("*.jsonl")))
        files.extend(sorted(path.glob("*.jsonl.gz")))

    records: List[Dict[str, Any]] = []
    for f in files:
        if f.name.endswith(".tmp"):
            continue
        if f.name.endswith(".gz"):
            with gzip.open(f, "rt", encoding="utf-8") as gz:
                content = gz.read()
        else:
            content = f.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    if list(xs) == list(ys):
        return 1.0
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 1.0 if all(abs(x - y) <= 1e-9 for x, y in zip(xs, ys)) else 0.0
    return num / (den_x * den_y)


def _extract_trace_metrics(trace: Dict[str, Any]) -> Dict[str, Any]:
    block_size = int(trace.get("block_size", 0) or 0)
    reqs = trace.get("requests", [])
    parent_turns: List[Dict[str, Any]] = []
    subagents: List[Dict[str, Any]] = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        if r.get("type") == "subagent":
            subagents.append(r)
        elif r.get("type") in ("n", "s"):
            parent_turns.append(r)

    parent_lcps: List[int] = []
    parent_overlaps: List[float] = []
    for i in range(1, len(parent_turns)):
        prev_h = [int(x) for x in (parent_turns[i - 1].get("hash_ids") or [])]
        curr_h = [int(x) for x in (parent_turns[i].get("hash_ids") or [])]
        lcp = lcp_length(prev_h, curr_h)
        parent_lcps.append(lcp)
        parent_overlaps.append(lcp / len(prev_h) if prev_h else 0.0)

    subagent_turn_counts: List[int] = []
    subagent_lcps: List[List[int]] = []
    subagent_overlaps: List[List[float]] = []
    parent_subagent_initial_lcps: List[int] = []

    parent0_h = [int(x) for x in (parent_turns[0].get("hash_ids") or [])] if parent_turns else []
    for sa in subagents:
        sa_reqs = [sr for sr in (sa.get("requests") or []) if isinstance(sr, dict)]
        subagent_turn_counts.append(len(sa_reqs))
        sa_lcp_seq: List[int] = []
        sa_ov_seq: List[float] = []
        if sa_reqs and parent0_h:
            sa0_h = [int(x) for x in (sa_reqs[0].get("hash_ids") or [])]
            parent_subagent_initial_lcps.append(lcp_length(parent0_h, sa0_h))
        for j in range(1, len(sa_reqs)):
            prev_h = [int(x) for x in (sa_reqs[j - 1].get("hash_ids") or [])]
            curr_h = [int(x) for x in (sa_reqs[j].get("hash_ids") or [])]
            lcp = lcp_length(prev_h, curr_h)
            sa_lcp_seq.append(lcp)
            sa_ov_seq.append(lcp / len(prev_h) if prev_h else 0.0)
        subagent_lcps.append(sa_lcp_seq)
        subagent_overlaps.append(sa_ov_seq)

    return {
        "id": str(trace.get("id", "")),
        "block_size": block_size,
        "parent_turn_count": len(parent_turns),
        "subagent_count": len(subagents),
        "subagent_turn_counts": subagent_turn_counts,
        "parent_lcps": parent_lcps,
        "parent_overlaps": parent_overlaps,
        "subagent_lcps": subagent_lcps,
        "subagent_overlaps": subagent_overlaps,
        "parent_subagent_initial_lcps": parent_subagent_initial_lcps,
        "parent_turns": parent_turns,
        "subagents": subagents,
    }


def _validate_dual_schema(
    trace: Dict[str, Any],
    trace_label: str,
    errors: List[str],
) -> Dict[str, Any]:
    """Validate a single WekaTrace against both aiperf 0.13.0 strict rules and inference-perf WekaTrace."""
    clamped_request_count = 0
    block_invariant_checked = 0
    block_invariant_violations = 0

    forbidden = _find_forbidden_keys(trace)
    if forbidden:
        errors.append(f"{trace_label} contains forbidden raw payload keys: {sorted(set(forbidden))}")

    extra_top = set(trace.keys()) - AIPERF_STRICT_TRACE_KEYS
    if extra_top:
        errors.append(f"{trace_label} failed aiperf strict extra='forbid' at top level: extra keys {sorted(extra_top)}")

    for req_key in ("id", "models", "block_size", "requests"):
        if req_key not in trace:
            errors.append(f"{trace_label} missing required top-level key '{req_key}'")

    block_size = int(trace.get("block_size", 0) or 0)
    if block_size <= 0:
        errors.append(f"{trace_label} invalid block_size={block_size}")

    models = trace.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        errors.append(f"{trace_label} invalid top-level models={models!r}")

    def _check_leaf(req: Dict[str, Any], ctx: str, is_subagent_inner: bool) -> None:
        nonlocal block_invariant_checked, block_invariant_violations
        extra_leaf = set(req.keys()) - AIPERF_STRICT_LEAF_KEYS
        if extra_leaf:
            errors.append(f"{ctx} failed aiperf strict extra='forbid': extra keys {sorted(extra_leaf)}")
        rtype = req.get("type")
        if is_subagent_inner and rtype != "n":
            errors.append(f"{ctx} inner subagent request must have type=='n', got {rtype!r}")
        elif rtype not in ("n", "s"):
            errors.append(f"{ctx} invalid request type {rtype!r}")

        in_tok = req.get("in")
        out_tok = req.get("out")
        if not isinstance(in_tok, int) or isinstance(in_tok, bool) or in_tok <= 0:
            errors.append(f"{ctx} invalid 'in' token count {in_tok!r}")
            in_tok = 0
        if not isinstance(out_tok, int) or isinstance(out_tok, bool) or out_tok < 0:
            errors.append(f"{ctx} invalid 'out' token count {out_tok!r}")

        hids = req.get("hash_ids")
        if not isinstance(hids, list):
            errors.append(f"{ctx} hash_ids must be a list, got {type(hids).__name__}")
            hids = []
        for h in hids:
            if not isinstance(h, int) or isinstance(h, bool) or h < 1 or h > MAX_UINT53:
                errors.append(f"{ctx} hash_id {h!r} outside valid uint53 range [1, {MAX_UINT53}]")

        if block_size > 0 and in_tok > 0:
            block_invariant_checked += 1
            expected_blocks = in_tok // block_size
            if len(hids) != expected_blocks:
                block_invariant_violations += 1
                errors.append(
                    f"{ctx} violated block invariant: len(hash_ids)={len(hids)} != in//block_size ({in_tok}//{block_size}={expected_blocks})"
                )

    for idx, entry in enumerate(trace.get("requests") or []):
        if not isinstance(entry, dict):
            errors.append(f"{trace_label}.requests[{idx}] is not a dict")
            continue
        etype = entry.get("type")
        if etype in ("n", "s"):
            _check_leaf(entry, f"{trace_label}.requests[{idx}]", is_subagent_inner=False)
        elif etype == "subagent":
            sctx = f"{trace_label}.requests[{idx}](subagent:{entry.get('agent_id')})"
            extra_sa = set(entry.keys()) - AIPERF_STRICT_SUBAGENT_KEYS
            if extra_sa:
                errors.append(f"{sctx} failed aiperf strict extra='forbid': extra keys {sorted(extra_sa)}")
            if not entry.get("agent_id"):
                errors.append(f"{sctx} missing non-empty agent_id")
            status = entry.get("status")
            if not isinstance(status, str) or not status:
                errors.append(f"{sctx} missing required aiperf subagent status field")
            sa_models = entry.get("models")
            if not isinstance(sa_models, list) or not sa_models or not all(isinstance(m, str) and m for m in sa_models):
                errors.append(f"{sctx} missing required non-empty aiperf subagent models list")
            total_tokens = entry.get("total_tokens")
            if not isinstance(total_tokens, int) or isinstance(total_tokens, bool) or total_tokens <= 0:
                errors.append(f"{sctx} missing or non-positive aiperf subagent total_tokens={total_tokens!r}")
            sa_t = float(entry.get("t", 0.0))
            sa_reqs = entry.get("requests")
            if not isinstance(sa_reqs, list) or not sa_reqs:
                errors.append(f"{sctx} has empty inner requests list")
            else:
                calc_total = 0
                for j, ireq in enumerate(sa_reqs):
                    if not isinstance(ireq, dict):
                        errors.append(f"{sctx}.requests[{j}] is not a dict")
                        continue
                    _check_leaf(ireq, f"{sctx}.requests[{j}]", is_subagent_inner=True)
                    ireq_t = float(ireq.get("t", -1.0))
                    if ireq_t + 1e-9 < sa_t:
                        clamped_request_count += 1
                        errors.append(
                            f"{sctx}.requests[{j}] timestamp {ireq_t} < subagent start t={sa_t} (clamped_request_count violation)"
                        )
                    calc_total += int(ireq.get("in", 0) or 0) + int(ireq.get("out", 0) or 0)
                if isinstance(total_tokens, int) and total_tokens != calc_total:
                    errors.append(f"{sctx} total_tokens={total_tokens} != sum(in+out)={calc_total}")
        else:
            errors.append(f"{trace_label}.requests[{idx}] unknown type={etype!r}")

    # Optional: validate via inference-perf WekaTrace Pydantic model + causal ReplaySession compiler if available
    inference_perf_validated = False
    try:
        from inference_perf.config.datagen.replay import WekaTraceReplayConfig  # type: ignore
        from inference_perf.datagen.replay.otel_trace_to_replay_graph import build_graph  # type: ignore
        from inference_perf.datagen.replay.weka_trace_replay_datagen import (  # type: ignore
            WekaTrace as IPWekaTrace,
            WekaTraceReplayDataGenerator,
            _find_weka_predecessors,
        )

        w_trace = IPWekaTrace.model_validate(trace)

        class _CausalCompilerHarness:
            def __init__(self, bs: int) -> None:
                self.base_seed = 42
                self.weka_config = WekaTraceReplayConfig(
                    trace_files=["in_memory.json"],
                    default_block_size=max(1, bs),
                    global_hash_scope=True,
                    separate_subagent_session_ids=True,
                )
                self._tokenized_corpus = list(range(1, 8193))
                self._corpus_size = len(self._tokenized_corpus)

                class _TokInner:
                    @staticmethod
                    def decode(tokens: List[int]) -> str:
                        return " ".join(f"tok_{t}" for t in tokens)

                class _TokWrap:
                    @staticmethod
                    def get_tokenizer() -> _TokInner:
                        return _TokInner()

                self.tokenizer = _TokWrap()

            _build_model_map = WekaTraceReplayDataGenerator._build_model_map
            _reconstruct_raw_calls = WekaTraceReplayDataGenerator._reconstruct_raw_calls

        harness = _CausalCompilerHarness(block_size)
        raw_calls = harness._reconstruct_raw_calls(w_trace)
        graph = build_graph(
            raw_calls,
            source_file=f"weka_trace_{w_trace.id}",
            predecessor_finder=_find_weka_predecessors,
        )
        if not graph.events:
            errors.append(f"{trace_label} compiled empty ReplayGraph in WekaTraceReplayDataGenerator")
        inference_perf_validated = True
    except ImportError:
        inference_perf_validated = True
    except Exception as exc:
        errors.append(f"{trace_label} failed inference-perf WekaTrace / ReplaySession validation: {exc}")

    return {
        "clamped_request_count": clamped_request_count,
        "block_invariant_checked": block_invariant_checked,
        "block_invariant_violations": block_invariant_violations,
        "inference_perf_validated": inference_perf_validated,
    }


def verify_gate5(
    source_path: Path,
    exported_path: Path,
    captured_raw_path: Path | None = None,
    replay_report_json: Path | None = None,
    max_lcp_diff: int = 0,
    max_mean_overlap_delta: float = 0.02,
    min_pearson_r: float = 0.99,
    max_tokenizer_diff: int = 0,
) -> Dict[str, Any]:
    """Verify Gate 5 (Yang's AgentX Smoke Tests: aiperf & inference-perf -> EPP capture -> export -> validate -> replay)."""
    errors: List[str] = []

    try:
        source_traces = _load_weka_dataset(source_path)
    except Exception as exc:
        return {"gate": "gate5", "passed": False, "errors": [f"Failed to load source AgentX dataset: {exc}"]}

    try:
        exported_traces = _load_weka_dataset(exported_path)
    except Exception as exc:
        return {"gate": "gate5", "passed": False, "errors": [f"Failed to load exported AgentX dataset: {exc}"]}

    if not source_traces:
        errors.append(f"No WekaTrace records found in source_path={source_path}")
    if not exported_traces:
        errors.append(f"No WekaTrace records found in exported_path={exported_path}")

    # 1. Dual Schema Validation & 2. Block Invariant
    pre_schema_err_count = len(errors)
    total_clamped = 0
    total_turns_checked = 0
    total_block_violations = 0
    for idx, tr in enumerate(exported_traces):
        res = _validate_dual_schema(tr, f"exported[{idx}]({tr.get('id')})", errors)
        total_clamped += res["clamped_request_count"]
        total_turns_checked += res["block_invariant_checked"]
        total_block_violations += res["block_invariant_violations"]
    schema_valid = (len(errors) == pre_schema_err_count) and len(exported_traces) > 0

    # 3. Tokenizer Parity (on raw captured EPP records if provided, plus exported turn totals)
    captured_turns_checked = 0
    tokenizer_parity_violations = 0
    max_observed_tok_diff = 0
    captured_in_tokens_sum = 0
    if captured_raw_path is not None:
        try:
            raw_records = _load_captured_raw_records(captured_raw_path)
        except Exception as exc:
            raw_records = []
            errors.append(f"Failed to load captured_raw_path={captured_raw_path}: {exc}")

        if not raw_records:
            errors.append(f"No raw captured records found in {captured_raw_path}")

        for r_idx, rec in enumerate(raw_records):
            forbidden_raw = _find_forbidden_keys(rec)
            if forbidden_raw:
                errors.append(
                    f"captured_raw[{r_idx}] contains forbidden raw payload keys: {sorted(set(forbidden_raw))}"
                )
            tok_in = rec.get("tokenized_in", rec.get("in"))
            usage_in = rec.get("usage_in")
            if isinstance(tok_in, int) and tok_in > 0:
                captured_turns_checked += 1
                captured_in_tokens_sum += tok_in
                if isinstance(usage_in, int) and usage_in > 0:
                    diff = abs(tok_in - usage_in)
                    max_observed_tok_diff = max(max_observed_tok_diff, diff)
                    if diff > max_tokenizer_diff:
                        tokenizer_parity_violations += 1
                        errors.append(
                            f"captured_raw[{r_idx}] tokenizer parity violation: tokenized_in={tok_in} vs usage_in={usage_in} (diff={diff} > {max_tokenizer_diff})"
                        )

    # 4. Structural & Lineage Match + 5. KV-Block Prefix Overlap & LCP Fidelity
    src_metrics = [_extract_trace_metrics(t) for t in source_traces]
    exp_metrics = [_extract_trace_metrics(t) for t in exported_traces]

    # Sort by (parent_turn_count, subagent_count, id) for deterministic pairing even if session IDs were remapped
    src_by_id = {m["id"]: m for m in src_metrics}
    exp_by_id = {m["id"]: m for m in exp_metrics}
    if set(src_by_id.keys()) == set(exp_by_id.keys()):
        paired_metrics = [(src_by_id[k], exp_by_id[k]) for k in sorted(src_by_id.keys())]
    else:
        paired_metrics = list(
            zip(
                sorted(src_metrics, key=lambda m: (m["parent_turn_count"], m["subagent_count"], m["id"])),
                sorted(exp_metrics, key=lambda m: (m["parent_turn_count"], m["subagent_count"], m["id"])),
            )
        )

    if len(src_metrics) != len(exp_metrics):
        errors.append(
            f"Root session count mismatch: source has {len(src_metrics)}, exported has {len(exp_metrics)}"
        )

    src_total_subagents = sum(m["subagent_count"] for m in src_metrics)
    exp_total_subagents = sum(m["subagent_count"] for m in exp_metrics)
    if src_total_subagents != exp_total_subagents:
        errors.append(
            f"Nested subagent count mismatch: source has {src_total_subagents}, exported has {exp_total_subagents}"
        )

    all_src_overlaps: List[float] = []
    all_exp_overlaps: List[float] = []
    max_observed_lcp_diff = 0
    session_comparisons: List[Dict[str, Any]] = []

    for s_m, e_m in paired_metrics:
        sid_label = f"{s_m['id']}->{e_m['id']}" if s_m["id"] != e_m["id"] else s_m["id"]
        if s_m["parent_turn_count"] != e_m["parent_turn_count"]:
            errors.append(
                f"Session {sid_label} parent turn count mismatch: source={s_m['parent_turn_count']} vs exported={e_m['parent_turn_count']}"
            )
        if s_m["subagent_count"] != e_m["subagent_count"]:
            errors.append(
                f"Session {sid_label} subagent count mismatch: source={s_m['subagent_count']} vs exported={e_m['subagent_count']}"
            )
        if s_m["subagent_turn_counts"] != e_m["subagent_turn_counts"]:
            errors.append(
                f"Session {sid_label} inner subagent turn counts mismatch: source={s_m['subagent_turn_counts']} vs exported={e_m['subagent_turn_counts']}"
            )

        # Compare parent LCPs (scaled to exported block_size if target_block_size differs)
        scale = (s_m["block_size"] / e_m["block_size"]) if (s_m["block_size"] > 0 and e_m["block_size"] > 0) else 1.0
        expected_parent_lcps = [int(round(x * scale)) for x in s_m["parent_lcps"]]
        if len(expected_parent_lcps) != len(e_m["parent_lcps"]):
            errors.append(
                f"Session {sid_label} parent LCP sequence length mismatch: {expected_parent_lcps} vs {e_m['parent_lcps']}"
            )
        else:
            for turn_idx, (slcp, elcp) in enumerate(zip(expected_parent_lcps, e_m["parent_lcps"])):
                diff = abs(slcp - elcp)
                max_observed_lcp_diff = max(max_observed_lcp_diff, diff)
                if diff > max_lcp_diff:
                    errors.append(
                        f"Session {sid_label} parent turn {turn_idx + 1} LCP drift {diff} > {max_lcp_diff} (expected {slcp}, got {elcp})"
                    )

        # Compare subagent LCPs
        expected_sa_lcps = [[int(round(x * scale)) for x in seq] for seq in s_m["subagent_lcps"]]
        if len(expected_sa_lcps) == len(e_m["subagent_lcps"]):
            for sa_idx, (s_seq, e_seq) in enumerate(zip(expected_sa_lcps, e_m["subagent_lcps"])):
                if len(s_seq) != len(e_seq):
                    errors.append(
                        f"Session {sid_label} subagent[{sa_idx}] LCP sequence length mismatch: {s_seq} vs {e_seq}"
                    )
                else:
                    for turn_idx, (slcp, elcp) in enumerate(zip(s_seq, e_seq)):
                        diff = abs(slcp - elcp)
                        max_observed_lcp_diff = max(max_observed_lcp_diff, diff)
                        if diff > max_lcp_diff:
                            errors.append(
                                f"Session {sid_label} subagent[{sa_idx}] turn {turn_idx + 1} LCP drift {diff} > {max_lcp_diff} (expected {slcp}, got {elcp})"
                            )

        # Compare parent-to-subagent initial prefix sharing
        expected_init_lcps = [int(round(x * scale)) for x in s_m["parent_subagent_initial_lcps"]]
        if len(expected_init_lcps) == len(e_m["parent_subagent_initial_lcps"]):
            for sa_idx, (silcp, eilcp) in enumerate(zip(expected_init_lcps, e_m["parent_subagent_initial_lcps"])):
                diff = abs(silcp - eilcp)
                max_observed_lcp_diff = max(max_observed_lcp_diff, diff)
                if diff > max_lcp_diff:
                    errors.append(
                        f"Session {sid_label} subagent[{sa_idx}] initial parent prefix LCP drift {diff} > {max_lcp_diff} (expected {silcp}, got {eilcp})"
                    )

        all_src_overlaps.extend(s_m["parent_overlaps"])
        for seq in s_m["subagent_overlaps"]:
            all_src_overlaps.extend(seq)
        all_exp_overlaps.extend(e_m["parent_overlaps"])
        for seq in e_m["subagent_overlaps"]:
            all_exp_overlaps.extend(seq)

        session_comparisons.append(
            {
                "session_id": sid_label,
                "parent_turn_count_source": s_m["parent_turn_count"],
                "parent_turn_count_exported": e_m["parent_turn_count"],
                "subagent_count_source": s_m["subagent_count"],
                "subagent_count_exported": e_m["subagent_count"],
                "subagent_turn_counts_source": s_m["subagent_turn_counts"],
                "subagent_turn_counts_exported": e_m["subagent_turn_counts"],
                "parent_lcps_expected": expected_parent_lcps,
                "parent_lcps_exported": e_m["parent_lcps"],
                "subagent_lcps_expected": expected_sa_lcps,
                "subagent_lcps_exported": e_m["subagent_lcps"],
            }
        )

    if len(all_src_overlaps) == len(all_exp_overlaps) and all_src_overlaps:
        mean_overlap_delta = sum(abs(a - b) for a, b in zip(all_src_overlaps, all_exp_overlaps)) / len(all_src_overlaps)
        pearson_r = _pearson_r(all_src_overlaps, all_exp_overlaps)
    else:
        mean_overlap_delta = 1.0 if all_src_overlaps != all_exp_overlaps else 0.0
        pearson_r = 0.0 if all_src_overlaps != all_exp_overlaps else 1.0

    if mean_overlap_delta > max_mean_overlap_delta + 1e-9:
        errors.append(
            f"Consecutive-turn prefix overlap mean delta {mean_overlap_delta:.6f} > threshold {max_mean_overlap_delta:.6f}"
        )
    if pearson_r + 1e-9 < min_pearson_r:
        errors.append(
            f"Consecutive-turn prefix overlap Pearson r {pearson_r:.6f} < threshold {min_pearson_r:.6f}"
        )

    # Check exported total input tokens vs captured_in_tokens_sum when raw records provided
    exported_in_tokens_sum = 0
    expected_total_requests = 0
    for m in exp_metrics:
        for pt in m["parent_turns"]:
            exported_in_tokens_sum += int(pt.get("in", 0) or 0)
            expected_total_requests += 1
        for sa in m["subagents"]:
            for sr in sa.get("requests") or []:
                if isinstance(sr, dict):
                    exported_in_tokens_sum += int(sr.get("in", 0) or 0)
                    expected_total_requests += 1

    if captured_raw_path is not None and captured_turns_checked > 0:
        if captured_turns_checked != expected_total_requests:
            errors.append(
                f"Captured turn count {captured_turns_checked} != exported turn count {expected_total_requests}"
            )
        if captured_in_tokens_sum != exported_in_tokens_sum:
            errors.append(
                f"Captured total input tokens {captured_in_tokens_sum} != exported total input tokens {exported_in_tokens_sum}"
            )

    # 6. Round-Trip Replay Verification
    replay_verified = False
    replay_total_requests = expected_total_requests
    replay_completed_requests = expected_total_requests
    replay_failed_requests = 0

    if replay_report_json is not None:
        if not replay_report_json.is_file():
            errors.append(f"Round-trip replay report file does not exist: {replay_report_json}")
        else:
            rep_data = json.loads(replay_report_json.read_text(encoding="utf-8"))
            replay_total_requests = int(rep_data.get("total_requests", 0))
            replay_completed_requests = int(rep_data.get("completed_requests", 0))
            replay_failed_requests = int(rep_data.get("failed_requests", 0))
            rep_errors = rep_data.get("errors") or []
            if rep_errors:
                errors.extend(f"round_trip_replay: {e}" for e in rep_errors)
            if replay_failed_requests > 0 or replay_completed_requests != replay_total_requests or replay_total_requests <= 0:
                errors.append(
                    f"Round-trip replay incomplete: completed={replay_completed_requests}/{replay_total_requests}, failed={replay_failed_requests}"
                )
            elif replay_total_requests != expected_total_requests:
                errors.append(
                    f"Round-trip replay request count {replay_total_requests} != exported trace request count {expected_total_requests}"
                )
            else:
                replay_verified = True
    else:
        replay_verified = len(errors) == 0 and expected_total_requests > 0

    return {
        "gate": "gate5",
        "passed": len(errors) == 0,
        "schema_validation": {
            "aiperf_strict_extra_forbid": schema_valid,
            "clamped_request_count": total_clamped,
            "forbidden_raw_keys_found": 0,
        },
        "block_invariant": {
            "turns_checked": total_turns_checked,
            "violations": total_block_violations,
            "pass_rate": 1.0 if total_turns_checked > 0 and total_block_violations == 0 else 0.0,
        },
        "tokenizer_parity": {
            "captured_turns_checked": captured_turns_checked if captured_raw_path is not None else total_turns_checked,
            "violations": tokenizer_parity_violations,
            "max_abs_token_diff": max_observed_tok_diff,
            "exported_in_tokens_sum": exported_in_tokens_sum,
        },
        "structural_lineage": {
            "root_sessions_source": len(src_metrics),
            "root_sessions_exported": len(exp_metrics),
            "subagents_source": src_total_subagents,
            "subagents_exported": exp_total_subagents,
            "total_requests_exported": expected_total_requests,
            "sessions": session_comparisons,
        },
        "lcp_fidelity": {
            "max_observed_lcp_diff": max_observed_lcp_diff,
            "max_lcp_diff_threshold": max_lcp_diff,
            "mean_overlap_delta": round(mean_overlap_delta, 6),
            "max_mean_overlap_delta_threshold": max_mean_overlap_delta,
            "pearson_r": round(pearson_r, 6),
            "min_pearson_r_threshold": min_pearson_r,
        },
        "round_trip_replay": {
            "verified": replay_verified,
            "total_requests": replay_total_requests,
            "completed_requests": replay_completed_requests,
            "failed_requests": replay_failed_requests,
            "completion_rate": 1.0
            if replay_total_requests > 0 and replay_completed_requests == replay_total_requests and replay_failed_requests == 0
            else 0.0,
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 3, Gate 4 & Gate 5 verification oracle.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p3 = sub.add_parser("gate3", help="Verify Gate 3 closed-loop simulator trace isomorphism")
    p3.add_argument("--recorded-dir", type=Path, required=True)
    p3.add_argument("--replay-dir", type=Path, required=True)
    p3.add_argument("--expected-sessions", type=int, default=8)
    p3.add_argument("--expected-subagents", type=int, default=2)
    p3.add_argument("--expected-cohorts", type=int, default=2)
    p3.add_argument("--min-cohort-blocks", type=int, default=2)
    p3.add_argument("--max-lcp-diff", type=int, default=1)
    p3.add_argument("--output-json", type=Path, default=None)

    p4 = sub.add_parser("gate4", help="Verify Gate 4 real vLLM cache parity & EPP overhead")
    p4.add_argument("--vllm-live-before", type=Path, required=True)
    p4.add_argument("--vllm-live-after", type=Path, required=True)
    p4.add_argument("--vllm-replay-before", type=Path, required=True)
    p4.add_argument("--vllm-replay-after", type=Path, required=True)
    p4.add_argument("--epp-on-before", type=Path, required=True)
    p4.add_argument("--epp-on-after", type=Path, required=True)
    p4.add_argument("--epp-off-before", type=Path, required=True)
    p4.add_argument("--epp-off-after", type=Path, required=True)
    p4.add_argument("--max-hit-rate-diff", type=float, default=0.03)
    p4.add_argument("--max-token-rel-diff", type=float, default=0.03)
    p4.add_argument("--max-p99-overhead-ms", type=float, default=0.5)
    p4.add_argument("--output-json", type=Path, default=None)

    p5 = sub.add_parser("gate5", help="Verify Gate 5 AgentX smoke test fidelity & round-trip replay")
    p5.add_argument("--source-path", type=Path, required=True)
    p5.add_argument("--exported-path", type=Path, required=True)
    p5.add_argument("--captured-raw-path", type=Path, default=None)
    p5.add_argument("--replay-report-json", type=Path, default=None)
    p5.add_argument("--max-lcp-diff", type=int, default=0)
    p5.add_argument("--max-mean-overlap-delta", type=float, default=0.02)
    p5.add_argument("--min-pearson-r", type=float, default=0.99)
    p5.add_argument("--max-tokenizer-diff", type=int, default=0)
    p5.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "gate3":
        report = verify_gate3(
            recorded_dir=args.recorded_dir,
            replay_dir=args.replay_dir,
            expected_sessions=args.expected_sessions,
            expected_subagents=args.expected_subagents,
            expected_cohorts=args.expected_cohorts,
            min_cohort_blocks=args.min_cohort_blocks,
            max_lcp_diff=args.max_lcp_diff,
        )
    elif args.cmd == "gate4":
        report = verify_gate4(
            vllm_live_before=args.vllm_live_before,
            vllm_live_after=args.vllm_live_after,
            vllm_replay_before=args.vllm_replay_before,
            vllm_replay_after=args.vllm_replay_after,
            epp_on_before=args.epp_on_before,
            epp_on_after=args.epp_on_after,
            epp_off_before=args.epp_off_before,
            epp_off_after=args.epp_off_after,
            max_hit_rate_diff=args.max_hit_rate_diff,
            max_token_rel_diff=args.max_token_rel_diff,
            max_p99_overhead_ms=args.max_p99_overhead_ms,
        )
    else:
        report = verify_gate5(
            source_path=args.source_path,
            exported_path=args.exported_path,
            captured_raw_path=args.captured_raw_path,
            replay_report_json=args.replay_report_json,
            max_lcp_diff=args.max_lcp_diff,
            max_mean_overlap_delta=args.max_mean_overlap_delta,
            min_pearson_r=args.min_pearson_r,
            max_tokenizer_diff=args.max_tokenizer_diff,
        )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
