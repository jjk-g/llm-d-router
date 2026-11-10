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

"""Live multi-turn OpenAI /v1/chat/completions traffic generator for GKE Gates 3 & 4.

Modes:
  - gate3:    8 multi-turn sessions (4 parent turns each), 2 shared system-prompt
              cohorts (4 sessions per cohort), and 2 synchronous subagent branches
              (sessions 0 and 4 spawn a subagent turn between parent turns 1 and 2,
              diverging after the shared system prompt before parent turn 2 resumes
              the pre-fork parent prefix).
  - gate4:    6 multi-turn chat sessions (4 turns each) sharing a common system
              prompt and OpenAI tool definitions catalog.
  - overhead: 200 streaming requests at concurrency 16 for Gate 4 p99 ext-proc
              latency overhead and zero-dropped-event validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from typing import Any
import urllib.error
import urllib.request

COHORT_A_SYSTEM_PROMPT = (
    "You are Cohort-Alpha Principal Kubernetes SRE Assistant operating inside the "
    "llm-d distributed inference routing plane. Always analyze Envoy ext-proc scheduling, "
    "KV-cache block alignment on 16-token boundaries, ZMQ cache invalidation streams, "
    "session-affinity fairness queuing, and tail latency budgets with crisp technical precision. "
    "Cite concrete metrics, pod topologies, and deterministic remediation steps in every reply."
)

COHORT_B_SYSTEM_PROMPT = (
    "You are Cohort-Beta Compiler & Distributed Systems Architect specializing in "
    "privacy-preserving telemetry pipelines, HMAC-SHA256 structural block anonymization, "
    "streaming Server-Sent Events usage extraction, and directed acyclic session replay graphs. "
    "Provide rigorous invariants, asymptotic complexity bounds, and reproducible verification "
    "checks for every architectural question asked by the operator."
)

GATE4_SHARED_SYSTEM_PROMPT = (
    "You are the Production Diagnostics Copilot for llm-d distributed inference clusters on "
    "Google Kubernetes Engine. Every conversation in this workload cohort shares an identical "
    "standing system specification and tool catalog so vLLM prefix caching reuses the initial "
    "16-token KV cache blocks across concurrent multi-turn sessions and subsequent replays.\n\n"
    "Operational Architecture & Routing Reference:\n"
    "1. Ingress & Envoy ext-proc Plane: Client OpenAI /v1/chat/completions traffic enters the "
    "standalone Envoy proxy Service and streams through active-active Endpoint Picker (EPP) "
    "replicas over gRPC ext-proc without sticky control-plane state. Each EPP pod runs a "
    "co-located vllm launch render HTTP sidecar so chat templates and tool schemas tokenize "
    "identically to the serving vLLM engine on exact 16-token block boundaries.\n"
    "2. Privacy-Preserving KV-Block Trace Recording: The kv-block-trace-recorder plugin "
    "observes tokenized block hashes, prompt token counts, and final SSE completion token "
    "usage asynchronously via a non-blocking ring buffer. Session, agent, and block-hash "
    "identifiers are keyed with a cluster-shared 256-bit Kubernetes Secret using HMAC-SHA256 "
    "domain-separated tags so raw prompt text, tool arguments, and model completions never "
    "touch staging disk or Google Cloud Storage.\n"
    "3. Atomic Segment Persistence & Offline Compaction: Active gzip segments write to "
    "pod-local emptyDir staging (/tmp/epp-traces) and atomically seal into GCS FUSE CSI "
    "mounts (/traces) on rotation or graceful SIGTERM drain. Offline compaction via "
    "epp-trace-compact merges multi-pod segments chronologically, reconstructs parent and "
    "synchronous subagent DAG branches, infers shared system-prompt block cohorts, and emits "
    "strictly validated WekaTrace session JSON artifacts.\n"
    "4. Closed-Loop Replay & Cache Parity Guidelines: During replay with global_hash_scope "
    "enabled, identical hash_ids across distinct sessions synthesize deterministic token "
    "blocks so vLLM prefix-cache hit ratios, prompt token totals, and generation token "
    "totals match live production behavior within tight statistical tolerances. Always "
    "respond with clear, structured engineering prose summarizing cluster health indicators, "
    "KV block reuse mechanics, and tail-latency invariants without invoking tools unless "
    "explicitly instructed by the operator."
)

GATE4_SHARED_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_epp_prometheus_metrics",
            "description": (
                "Query live Prometheus counters and histograms exported by llm-d-router "
                "Endpoint Picker pods, including scheduler latency, prefix indexer hit ratio, "
                "and trace recorder dropped-event counters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Fully qualified Prometheus metric name to inspect.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace hosting the router release.",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["sum", "p50", "p99", "rate"],
                    },
                },
                "required": ["metric_name", "namespace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_gcs_trace_segments",
            "description": (
                "List sealed gzipped JSONL trace segments in Google Cloud Storage and verify "
                "atomic rotation, block-hash schema version, and zero raw prompt leakage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket": {"type": "string"},
                    "prefix": {"type": "string"},
                    "min_segments": {"type": "integer"},
                },
                "required": ["bucket", "prefix"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_vllm_prefix_cache_telemetry",
            "description": (
                "Scrape Prometheus /metrics from serving vLLM pods to evaluate KV-cache block "
                "hit ratios, total prefix cache queries, GPU KV block utilization, active "
                "running/waiting sequence queues, and cumulative prefill/decode token counters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pod_selector": {
                        "type": "string",
                        "description": "Kubernetes label selector matching target vLLM model server pods.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace containing the vLLM Deployment and Service.",
                    },
                    "metrics_port": {
                        "type": "integer",
                        "description": "HTTP port exposing vLLM Prometheus metrics (typically 8000).",
                    },
                    "include_histograms": {
                        "type": "boolean",
                        "description": "Whether to return time-to-first-token and inter-token latency buckets.",
                    },
                },
                "required": ["pod_selector", "namespace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_envoy_ext_proc_clusters",
            "description": (
                "Query Envoy admin /clusters endpoint to verify STRICT_DNS resolution and active "
                "gRPC ExternalProcessor health check status across all active-active EPP pod IPs "
                "before dispatching latency-sensitive replay or overhead benchmark traffic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "proxy_deployment": {
                        "type": "string",
                        "description": "Name of the standalone Envoy proxy Deployment in Kubernetes.",
                    },
                    "admin_port": {
                        "type": "integer",
                        "description": "Envoy localhost admin HTTP port (default 19000).",
                    },
                    "expected_epp_replicas": {
                        "type": "integer",
                        "description": "Expected count of healthy active-active EPP gRPC endpoints.",
                    },
                    "require_zero_pending_hc": {
                        "type": "boolean",
                        "description": "Require zero endpoints in pending_active_hc or failed_active_hc state.",
                    },
                },
                "required": ["proxy_deployment", "expected_epp_replicas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_compacted_weka_traces",
            "description": (
                "Run structural and statistical invariants over compacted WekaTrace JSON files "
                "produced by epp-trace-compact, checking dense global hash_ids remapping, "
                "inferred system_tokens cohort sizes, causal turn timestamps, and subagent trees."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trace_directory": {
                        "type": "string",
                        "description": "Filesystem path containing compacted per-session WekaTrace JSON files.",
                    },
                    "block_size_tokens": {
                        "type": "integer",
                        "description": "Expected KV-cache token block size (16 for vLLM parity gates).",
                    },
                    "expected_sessions": {
                        "type": "integer",
                        "description": "Expected number of distinct parent conversation sessions.",
                    },
                    "expected_subagent_branches": {
                        "type": "integer",
                        "description": "Expected count of extracted WekaSubagentEntry child branches.",
                    },
                    "verify_hmac_pseudonyms": {
                        "type": "boolean",
                        "description": "Assert all session and agent IDs use s_ and a_ HMAC pseudonyms.",
                    },
                },
                "required": ["trace_directory", "block_size_tokens", "expected_sessions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_zmq_kv_event_subscriptions",
            "description": (
                "Inspect ZMQ PUB/SUB KV-cache block event connectivity between serving vLLM or "
                "llm-d-inference-sim pods and active-active Endpoint Picker indexers on TCP 5557, "
                "verifying sequence monotonicity, block eviction watermarks, and scorer weights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "epp_service_fqdn": {
                        "type": "string",
                        "description": "Cluster-local FQDN of the EPP ZMQ subscriber Service.",
                    },
                    "zmq_port": {
                        "type": "integer",
                        "description": "TCP port bound by precise-prefix-cache-producer (default 5557).",
                    },
                    "window_seconds": {
                        "type": "integer",
                        "description": "Observation window in seconds for block stored/removed rates.",
                    },
                },
                "required": ["epp_service_fqdn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_live_and_replay_parity_report",
            "description": (
                "Compute prefix-cache hit-rate delta, prompt and generation token relative drift, "
                "and per-extension-point EPP plugin p99 latency overhead across live and replayed "
                "GKE benchmark runs to certify Milestone 4 release readiness."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "live_metrics_snapshot": {"type": "string"},
                    "replay_metrics_snapshot": {"type": "string"},
                    "max_hit_rate_abs_diff": {"type": "number"},
                    "max_token_rel_diff": {"type": "number"},
                    "max_plugin_p99_ms": {"type": "number"},
                },
                "required": ["live_metrics_snapshot", "replay_metrics_snapshot"],
            },
        },
    },
]


def _send_streaming_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    headers: dict[str, str],
    max_tokens: int = 28,
    tools: list[dict[str, Any]] | None = None,
    ignore_eos: bool = False,
    timeout_sec: float = 90.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "none"

    body = json.dumps(payload).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **headers,
    }
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")

    t0 = time.monotonic()
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    chunk_count = 0

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        status = resp.status
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            chunk_count += 1
            if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)

    elapsed_s = time.monotonic() - t0
    assistant_text = "".join(content_parts)
    if not assistant_text:
        # Fallback only when an echo-less mock emits zero content deltas; never pad
        # non-empty model responses after EPP has already recorded SSE usage.completion_tokens.
        assistant_text = (
            "Deterministic fallback assistant response when upstream mock emits empty deltas."
        )

    return {
        "status": status,
        "elapsed_s": elapsed_s,
        "chunk_count": chunk_count,
        "usage": usage,
        "assistant_text": assistant_text,
    }


def _user_turn_prompt(session_idx: int, turn_idx: int, label: str) -> str:
    return (
        f"[{label} session={session_idx} turn={turn_idx}] "
        f"Evaluate routing state transition #{turn_idx} for workload partition {session_idx}: "
        f"detail how 16-token KV cache blocks accumulate across turn history, how prefix "
        f"scoring selects the warm replica, and what telemetry confirms zero dropped trace "
        f"records under concurrent streaming load. Provide a thorough multi-sentence explanation."
    )


def run_gate3_workload(base_url: str, model: str, run_id: str, concurrency: int) -> dict[str, Any]:
    """8 sessions x 4 parent turns, 2 system-prompt cohorts, 2 synchronous subagent branches."""

    def _run_one_session(session_idx: int) -> dict[str, Any]:
        cohort = "A" if session_idx < 4 else "B"
        sys_prompt = COHORT_A_SYSTEM_PROMPT if cohort == "A" else COHORT_B_SYSTEM_PROMPT
        has_subagent = session_idx in (0, 4)
        session_id = f"{run_id}-gate3-sess-{session_idx:02d}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        parent_calls = 0
        subagent_calls = 0

        for turn_idx in range(4):
            messages.append(
                {"role": "user", "content": _user_turn_prompt(session_idx, turn_idx, f"cohort-{cohort}")}
            )
            res = _send_streaming_chat(
                base_url=base_url,
                model=model,
                messages=messages,
                headers={"x-llm-d-inference-fairness-id": session_id},
                max_tokens=32,
                ignore_eos=True,
            )
            parent_calls += 1
            messages.append({"role": "assistant", "content": res["assistant_text"]})
            time.sleep(0.25)

            # After parent turn 1 (second parent request), spawn synchronous subagent on sessions 0 & 4.
            if has_subagent and turn_idx == 1:
                sub_messages = [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"[subagent-explorer session={session_idx}] Inspect auxiliary shard "
                            f"diagnostics for session {session_idx} independently of the parent "
                            f"transcript, summarizing ZMQ block announcements, staging segment "
                            f"rotation health, and cross-pod fairness queue balance in detail."
                        ),
                    },
                ]
                _send_streaming_chat(
                    base_url=base_url,
                    model=model,
                    messages=sub_messages,
                    headers={
                        "x-llm-d-inference-fairness-id": session_id,
                        "x-agent-id": f"explorer-{session_idx:02d}",
                        "x-subagent-type": "explorer",
                    },
                    max_tokens=32,
                    ignore_eos=True,
                )
                subagent_calls += 1
                time.sleep(0.25)

        return {
            "session_idx": session_idx,
            "session_id": session_id,
            "cohort": cohort,
            "parent_calls": parent_calls,
            "subagent_calls": subagent_calls,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(concurrency, 4)) as pool:
        futs = [pool.submit(_run_one_session, idx) for idx in range(8)]
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["session_idx"])
    return {
        "mode": "gate3",
        "sessions": len(results),
        "total_parent_calls": sum(r["parent_calls"] for r in results),
        "total_subagent_calls": sum(r["subagent_calls"] for r in results),
        "details": results,
    }


def run_gate4_workload(base_url: str, model: str, run_id: str, concurrency: int) -> dict[str, Any]:
    """6 multi-turn chat sessions x 4 turns sharing system prompt + tool definitions."""

    def _run_one_session(session_idx: int) -> dict[str, Any]:
        session_id = f"{run_id}-gate4-sess-{session_idx:02d}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": GATE4_SHARED_SYSTEM_PROMPT}]
        prompt_tokens_sum = 0
        completion_tokens_sum = 0

        for turn_idx in range(4):
            messages.append(
                {
                    "role": "user",
                    "content": _user_turn_prompt(session_idx, turn_idx, "gate4-live"),
                }
            )
            res = _send_streaming_chat(
                base_url=base_url,
                model=model,
                messages=messages,
                headers={"x-llm-d-inference-fairness-id": session_id},
                max_tokens=32,
                tools=GATE4_SHARED_TOOLS,
                ignore_eos=True,
            )
            usage = res.get("usage") or {}
            prompt_tokens_sum += int(usage.get("prompt_tokens") or 0)
            completion_tokens_sum += int(usage.get("completion_tokens") or 0)
            messages.append({"role": "assistant", "content": res["assistant_text"]})
            time.sleep(0.15)

        return {
            "session_idx": session_idx,
            "session_id": session_id,
            "turns": 4,
            "prompt_tokens": prompt_tokens_sum,
            "completion_tokens": completion_tokens_sum,
        }

    # Seed session 0 turn 0..3 first (or run with controlled concurrency) so the shared
    # system+tools prefix is populated in vLLM's KV cache before peer sessions query it.
    first = _run_one_session(0)
    results: list[dict[str, Any]] = [first]
    if concurrency <= 1:
        for idx in range(1, 6):
            results.append(_run_one_session(idx))
    else:
        with ThreadPoolExecutor(max_workers=min(concurrency, 3)) as pool:
            futs = [pool.submit(_run_one_session, idx) for idx in range(1, 6)]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: r["session_idx"])
    return {
        "mode": "gate4",
        "sessions": len(results),
        "total_requests": sum(r["turns"] for r in results),
        "reported_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "reported_completion_tokens": sum(r["completion_tokens"] for r in results),
        "details": results,
    }


def run_overhead_workload(
    base_url: str,
    model: str,
    run_id: str,
    num_requests: int,
    concurrency: int,
) -> dict[str, Any]:
    """200 streaming requests at concurrency 16 for p99 ext-proc latency & drop check."""

    def _one_req(req_idx: int) -> float:
        session_id = f"{run_id}-overhead-{req_idx % 16:02d}"
        messages = [
            {"role": "system", "content": GATE4_SHARED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Benchmark probe #{req_idx}: summarize router status in two sentences.",
            },
        ]
        res = _send_streaming_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            headers={"x-llm-d-inference-fairness-id": session_id},
            max_tokens=16,
            ignore_eos=True,
        )
        return float(res["elapsed_s"])

    latencies: list[float] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_one_req, i) for i in range(num_requests)]
        for fut in as_completed(futs):
            latencies.append(fut.result())
    wall_s = time.monotonic() - t0
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] if latencies else 0.0
    return {
        "mode": "overhead",
        "num_requests": len(latencies),
        "concurrency": concurrency,
        "wall_seconds": round(wall_s, 4),
        "client_p50_seconds": round(p50, 4),
        "client_p99_seconds": round(p99, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Send live multi-turn workloads for Gates 3 & 4.")
    parser.add_argument("--mode", choices=["gate3", "gate4", "overhead"], required=True)
    parser.add_argument("--base-url", required=True, help="Envoy HTTP base URL (e.g. http://127.0.0.1:18081)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--run-id", default="local-run")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--output-json", default="", help="Optional path to write summary JSON")
    args = parser.parse_args()

    try:
        if args.mode == "gate3":
            summary = run_gate3_workload(args.base_url, args.model, args.run_id, args.concurrency)
        elif args.mode == "gate4":
            summary = run_gate4_workload(args.base_url, args.model, args.run_id, args.concurrency)
        else:
            summary = run_overhead_workload(
                args.base_url,
                args.model,
                args.run_id,
                num_requests=args.num_requests,
                concurrency=args.concurrency,
            )
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        print(f"ERROR: HTTP {exc.code} from {args.base_url}: {err_body}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: workload {args.mode} failed: {exc}", file=sys.stderr)
        return 1

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
