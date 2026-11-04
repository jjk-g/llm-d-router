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


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 3 & Gate 4 verification oracle.")
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
    else:
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

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
