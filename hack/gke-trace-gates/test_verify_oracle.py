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

"""Unit tests for hack/gke-trace-gates/verify_oracle.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_oracle  # noqa: E402


def _write_gate3_fixture(
    out_dir: Path,
    *,
    id_prefix: str = "s_rec",
    cohort_a_blocks: tuple[int, ...] = (101, 102, 103),
    cohort_b_blocks: tuple[int, ...] = (201, 202, 203),
    lcp_shift: int = 0,
    sa_lcp_shift: int = 0,
    include_subagents: int = 2,
    empty_subagent_requests: bool = False,
    num_sessions: int = 8,
    inject_forbidden: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    next_unique = 1000
    for idx in range(num_sessions):
        sys_blocks = list(cohort_a_blocks if idx < 4 else cohort_b_blocks)
        requests: list[dict] = []
        cur_blocks = list(sys_blocks) + [next_unique, next_unique + 1]
        next_unique += 10

        for turn in range(4):
            if turn > 0:
                base = cur_blocks[:-lcp_shift] if (lcp_shift > 0 and turn == 2) else list(cur_blocks)
                cur_blocks = base + [next_unique, next_unique + 1, next_unique + 2]
                next_unique += 10
            req_entry: dict = {
                "type": "n",
                "t": round(idx * 2.0 + turn * 0.4, 3),
                "in": len(cur_blocks) * 16,
                "out": 32,
                "hash_ids": list(cur_blocks),
            }
            if inject_forbidden and idx == 0 and turn == 0:
                req_entry["messages"] = [{"role": "user", "content": "LEAKED_CANARY"}]
            requests.append(req_entry)

            if turn == 1 and (
                (idx == 0 and include_subagents >= 1) or (idx == 4 and include_subagents >= 2)
            ):
                sa_prefix = sys_blocks[:-sa_lcp_shift] if sa_lcp_shift > 0 else list(sys_blocks)
                sa_reqs = (
                    []
                    if empty_subagent_requests
                    else [
                        {
                            "type": "n",
                            "t": 0.0,
                            "in": (len(sa_prefix) + 2) * 16,
                            "out": 32,
                            "hash_ids": list(sa_prefix) + [next_unique + 90, next_unique + 91],
                        }
                    ]
                )
                requests.append(
                    {
                        "type": "subagent",
                        "t": round(idx * 2.0 + turn * 0.4 + 0.2, 3),
                        "agent_id": f"a_{idx:04d}",
                        "subagent_type": "explorer",
                        "duration_ms": 120,
                        "tool_tokens": 0,
                        "system_tokens": len(sys_blocks) * 16,
                        "requests": sa_reqs,
                    }
                )

        trace = {
            "id": f"{id_prefix}_{idx:02d}",
            "block_size": 16,
            "system_tokens": len(sys_blocks) * 16,
            "requests": requests,
        }
        (out_dir / f"{trace['id']}.json").write_text(json.dumps(trace), encoding="utf-8")


def _vllm_prom(hits: int, queries: int, prompt_tok: int, gen_tok: int) -> str:
    return "\n".join(
        [
            "# TYPE vllm:prefix_cache_hits_total counter",
            f'vllm:prefix_cache_hits_total{{model_name="Qwen/Qwen2.5-0.5B-Instruct"}} {hits}',
            "# TYPE vllm:prefix_cache_queries_total counter",
            f'vllm:prefix_cache_queries_total{{model_name="Qwen/Qwen2.5-0.5B-Instruct"}} {queries}',
            f'vllm:prompt_tokens_total{{model_name="Qwen/Qwen2.5-0.5B-Instruct"}} {prompt_tok}',
            f'vllm:generation_tokens_total{{model_name="Qwen/Qwen2.5-0.5B-Instruct"}} {gen_tok}',
        ]
    )


def _epp_prom(
    *,
    dropped: int,
    req_count: int,
    resp_fast_count: int,
    resp_slow_count: int = 0,
    plugin_fast_count: int = 200,
    plugin_slow_count: int = 0,
) -> str:
    tot_resp = resp_fast_count + resp_slow_count
    tot_plugin = plugin_fast_count + plugin_slow_count
    lines = [
        f'llm_d_epp_trace_recorder_dropped_events_total{{plugin_name="trace-recorder"}} {dropped}',
        f'llm_d_epp_request_processing_duration_seconds_bucket{{api_surface="chat",le="0.005"}} {req_count}',
        f'llm_d_epp_request_processing_duration_seconds_bucket{{api_surface="chat",le="0.01"}} {req_count}',
        f'llm_d_epp_request_processing_duration_seconds_bucket{{api_surface="chat",le="+Inf"}} {req_count}',
        f'llm_d_epp_response_processing_duration_seconds_bucket{{api_surface="chat",le="0.00025"}} {resp_fast_count}',
        f'llm_d_epp_response_processing_duration_seconds_bucket{{api_surface="chat",le="0.0005"}} {resp_fast_count}',
        f'llm_d_epp_response_processing_duration_seconds_bucket{{api_surface="chat",le="0.005"}} {tot_resp}',
        f'llm_d_epp_response_processing_duration_seconds_bucket{{api_surface="chat",le="+Inf"}} {tot_resp}',
    ]
    for ext_pt in ("DataProducer", "PreRequest", "ResponseStreaming"):
        lines.extend(
            [
                f'llm_d_epp_plugin_duration_seconds_bucket{{extension_point="{ext_pt}",plugin_name="trace-recorder",plugin_type="kv-block-trace-recorder",le="0.0001"}} {plugin_fast_count}',
                f'llm_d_epp_plugin_duration_seconds_bucket{{extension_point="{ext_pt}",plugin_name="trace-recorder",plugin_type="kv-block-trace-recorder",le="0.0002"}} {plugin_fast_count}',
                f'llm_d_epp_plugin_duration_seconds_bucket{{extension_point="{ext_pt}",plugin_name="trace-recorder",plugin_type="kv-block-trace-recorder",le="0.0005"}} {plugin_fast_count}',
                f'llm_d_epp_plugin_duration_seconds_bucket{{extension_point="{ext_pt}",plugin_name="trace-recorder",plugin_type="kv-block-trace-recorder",le="0.005"}} {tot_plugin}',
                f'llm_d_epp_plugin_duration_seconds_bucket{{extension_point="{ext_pt}",plugin_name="trace-recorder",plugin_type="kv-block-trace-recorder",le="+Inf"}} {tot_plugin}',
            ]
        )
    return "\n".join(lines)


class VerifyOracleTests(unittest.TestCase):
    def test_gate3_passes_with_anonymized_ids_and_one_block_lcp_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec_dir = root / "rec"
            rep_dir = root / "rep"
            _write_gate3_fixture(rec_dir, id_prefix="s_alpha", lcp_shift=0, sa_lcp_shift=0)
            _write_gate3_fixture(
                rep_dir,
                id_prefix="s_replayed_zzz",
                cohort_a_blocks=(501, 502, 503),
                cohort_b_blocks=(601, 602, 603),
                lcp_shift=1,
                sa_lcp_shift=1,
            )
            res = verify_oracle.verify_gate3(rec_dir, rep_dir)
            self.assertTrue(res["passed"], msg=res["errors"])
            self.assertEqual(res["recorded_sessions"], 8)
            self.assertEqual(res["replay_sessions"], 8)
            self.assertEqual(res["recorded_subagents"], 2)
            self.assertEqual(res["replay_subagents"], 2)
            self.assertEqual(res["recorded_cohort_sizes"], [4, 4])
            self.assertEqual(res["replay_cohort_sizes"], [4, 4])
            self.assertEqual(res["max_parent_lcp_diff"], 1)
            self.assertEqual(res["max_subagent_lcp_diff"], 1)
            sa_pairs = [p for p in res["pairs"] if p["subagent_count"] > 0]
            self.assertEqual(len(sa_pairs), 2)
            for sp in sa_pairs:
                self.assertEqual(sp["recorded_subagent_lcps"], [3])
                self.assertEqual(sp["replay_subagent_lcps"], [2])

    def test_gate3_fails_when_lcp_diff_exceeds_one_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec_dir = root / "rec"
            rep_dir = root / "rep"
            _write_gate3_fixture(rec_dir, lcp_shift=0)
            _write_gate3_fixture(rep_dir, lcp_shift=2)
            res = verify_oracle.verify_gate3(rec_dir, rep_dir)
            self.assertFalse(res["passed"])
            self.assertGreaterEqual(res["max_parent_lcp_diff"], 2)

    def test_gate3_fails_when_subagent_requests_empty_or_lost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec_dir = root / "rec"
            rep_dir = root / "rep"
            _write_gate3_fixture(rec_dir, include_subagents=2)
            _write_gate3_fixture(rep_dir, include_subagents=2, empty_subagent_requests=True)
            res = verify_oracle.verify_gate3(rec_dir, rep_dir)
            self.assertFalse(res["passed"])
            self.assertTrue(any("empty or invalid 'requests'" in e for e in res["errors"]))

    def test_gate3_rejects_raw_prompt_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec_dir = root / "rec"
            rep_dir = root / "rep"
            _write_gate3_fixture(rec_dir, inject_forbidden=True)
            _write_gate3_fixture(rep_dir)
            with self.assertRaises(AssertionError):
                verify_oracle.verify_gate3(rec_dir, rep_dir)

    def test_gate4_passes_within_3pct_and_sub_half_ms_overhead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "live_b": _vllm_prom(100, 200, 1000, 500),
                "live_a": _vllm_prom(720, 1200, 7000, 1268),  # hit_rate = 0.620
                "rep_b": _vllm_prom(720, 1200, 7000, 1268),
                "rep_a": _vllm_prom(1335, 2200, 13090, 2036),  # hit_rate = 0.615
                "on_b": _epp_prom(dropped=0, req_count=10, resp_fast_count=10, plugin_fast_count=10),
                "on_a": _epp_prom(dropped=0, req_count=210, resp_fast_count=210, plugin_fast_count=210),
                "off_b": _epp_prom(dropped=0, req_count=5, resp_fast_count=5, plugin_fast_count=0),
                "off_a": _epp_prom(dropped=0, req_count=205, resp_fast_count=205, plugin_fast_count=0),
            }
            paths: dict[str, Path] = {}
            for k, content in files.items():
                p = root / f"{k}.prom"
                p.write_text(content, encoding="utf-8")
                paths[k] = p

            res = verify_oracle.verify_gate4(
                vllm_live_before=paths["live_b"],
                vllm_live_after=paths["live_a"],
                vllm_replay_before=paths["rep_b"],
                vllm_replay_after=paths["rep_a"],
                epp_on_before=paths["on_b"],
                epp_on_after=paths["on_a"],
                epp_off_before=paths["off_b"],
                epp_off_after=paths["off_a"],
            )
            self.assertTrue(res["passed"], msg=res["errors"])
            self.assertLessEqual(res["prefix_cache"]["abs_diff"], 0.03)
            # 3 extension points * ~0.099 ms each = ~0.297 ms < 0.5 ms
            self.assertGreater(res["epp_overhead"]["recorder_plugin_hotpath_sum_p99_ms"], 0.0)
            self.assertLess(res["epp_overhead"]["recorder_plugin_hotpath_sum_p99_ms"], 0.5)

    def test_gate4_fails_on_empty_or_zero_epp_histograms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "live_b": _vllm_prom(100, 200, 1000, 500),
                "live_a": _vllm_prom(720, 1200, 7000, 1268),
                "rep_b": _vllm_prom(720, 1200, 7000, 1268),
                "rep_a": _vllm_prom(1335, 2200, 13090, 2036),
                "on_b": "",
                "on_a": "",
                "off_b": "",
                "off_a": "",
            }
            paths = {}
            for k, content in files.items():
                p = root / f"{k}.prom"
                p.write_text(content, encoding="utf-8")
                paths[k] = p

            res = verify_oracle.verify_gate4(
                vllm_live_before=paths["live_b"],
                vllm_live_after=paths["live_a"],
                vllm_replay_before=paths["rep_b"],
                vllm_replay_after=paths["rep_a"],
                epp_on_before=paths["on_b"],
                epp_on_after=paths["on_a"],
                epp_off_before=paths["off_b"],
                epp_off_after=paths["off_a"],
            )
            self.assertFalse(res["passed"])
            self.assertGreaterEqual(
                len([e for e in res["errors"] if "sample count delta must be > 0" in e]), 5
            )

    def test_gate4_fails_on_dropped_events_or_hit_rate_drift_or_plugin_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "live_b": _vllm_prom(0, 0, 0, 0),
                "live_a": _vllm_prom(700, 1000, 6000, 600),
                "rep_b": _vllm_prom(0, 0, 0, 0),
                "rep_a": _vllm_prom(500, 1000, 6000, 600),
                "on_b": _epp_prom(dropped=0, req_count=0, resp_fast_count=0, plugin_fast_count=0),
                "on_a": _epp_prom(
                    dropped=3,
                    req_count=200,
                    resp_fast_count=180,
                    resp_slow_count=20,
                    plugin_fast_count=180,
                    plugin_slow_count=20,
                ),
                "off_b": _epp_prom(dropped=0, req_count=0, resp_fast_count=0, plugin_fast_count=0),
                "off_a": _epp_prom(dropped=0, req_count=200, resp_fast_count=200, plugin_fast_count=0),
            }
            paths = {}
            for k, content in files.items():
                p = root / f"{k}.prom"
                p.write_text(content, encoding="utf-8")
                paths[k] = p

            res = verify_oracle.verify_gate4(
                vllm_live_before=paths["live_b"],
                vllm_live_after=paths["live_a"],
                vllm_replay_before=paths["rep_b"],
                vllm_replay_after=paths["rep_a"],
                epp_on_before=paths["on_b"],
                epp_on_after=paths["on_a"],
                epp_off_before=paths["off_b"],
                epp_off_after=paths["off_a"],
            )
            self.assertFalse(res["passed"])
            self.assertTrue(any("hit rate parity failed" in e for e in res["errors"]))
            self.assertTrue(any("dropped_events_total" in e for e in res["errors"]))
            self.assertTrue(any("response ext-proc p99 overhead" in e for e in res["errors"]))
            self.assertTrue(any("summed hot-path plugin p99" in e for e in res["errors"]))

    def test_gate5_passes_on_valid_agentx_smoke_fixture(self) -> None:
        fixture = Path(__file__).resolve().parent / "agentx_smoke_fixture.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            # Synthesize raw captured EPP records matching the 10 turns in agentx_smoke_fixture.jsonl
            traces = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
            raw_records = []
            for tr in traces:
                for req in tr["requests"]:
                    if req["type"] == "subagent":
                        for ireq in req["requests"]:
                            raw_records.append(
                                {
                                    "v": 1,
                                    "session_id": tr["id"],
                                    "agent_id": req["agent_id"],
                                    "subagent_type": req["subagent_type"],
                                    "block_size": tr["block_size"],
                                    "in": ireq["in"],
                                    "tokenized_in": ireq["in"],
                                    "usage_in": ireq["in"],
                                    "out": ireq["out"],
                                    "hash_ids": ireq["hash_ids"],
                                }
                            )
                    else:
                        raw_records.append(
                            {
                                "v": 1,
                                "session_id": tr["id"],
                                "block_size": tr["block_size"],
                                "in": req["in"],
                                "tokenized_in": req["in"],
                                "usage_in": req["in"],
                                "out": req["out"],
                                "hash_ids": req["hash_ids"],
                            }
                        )
            (raw_dir / "capture.jsonl").write_text(
                "\n".join(json.dumps(r) for r in raw_records) + "\n", encoding="utf-8"
            )
            replay_report = root / "replay_report.json"
            replay_report.write_text(
                json.dumps(
                    {
                        "total_requests": 10,
                        "completed_requests": 10,
                        "failed_requests": 0,
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )

            res = verify_oracle.verify_gate5(
                source_path=fixture,
                exported_path=fixture,
                captured_raw_path=raw_dir,
                replay_report_json=replay_report,
            )
            self.assertTrue(res["passed"], res["errors"])
            self.assertEqual(res["schema_validation"]["clamped_request_count"], 0)
            self.assertEqual(res["block_invariant"]["turns_checked"], 10)
            self.assertEqual(res["block_invariant"]["violations"], 0)
            self.assertEqual(res["tokenizer_parity"]["violations"], 0)
            self.assertEqual(res["structural_lineage"]["root_sessions_exported"], 2)
            self.assertEqual(res["structural_lineage"]["subagents_exported"], 2)
            self.assertEqual(res["lcp_fidelity"]["max_observed_lcp_diff"], 0)
            self.assertAlmostEqual(res["lcp_fidelity"]["pearson_r"], 1.0)
            self.assertTrue(res["round_trip_replay"]["verified"])

    def test_gate5_fails_on_schema_clamped_block_tokenizer_lcp_and_replay_violations(self) -> None:
        fixture = Path(__file__).resolve().parent / "agentx_smoke_fixture.jsonl"
        traces = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
        bad_traces = json.loads(json.dumps(traces))
        # 1. Extra forbidden aiperf field & forbidden raw key
        bad_traces[0]["unexpected_extra_field"] = "bad"
        bad_traces[0]["requests"][0]["messages"] = [{"role": "user", "content": "leak"}]
        # 2. Block invariant violation (in=64 at block_size=16 expects 4 hashes, give 3)
        bad_traces[0]["requests"][0]["hash_ids"] = [1, 2, 3]
        # 3. Subagent clamped timestamp violation (req.t < subagent.t) & wrong inner type
        bad_traces[0]["requests"][1]["requests"][0]["t"] = 0.05  # < subagent.t=0.1
        bad_traces[0]["requests"][1]["requests"][0]["type"] = "s"
        # 4. LCP drift on parent turn 1
        bad_traces[0]["requests"][2]["hash_ids"] = [91, 92, 93, 94, 95, 96]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_file = root / "bad_exported.jsonl"
            exp_file.write_text("\n".join(json.dumps(t) for t in bad_traces) + "\n", encoding="utf-8")

            raw_file = root / "bad_raw.jsonl"
            raw_file.write_text(
                json.dumps({"v": 1, "tokenized_in": 64, "usage_in": 50, "prompt": "secret"}) + "\n",
                encoding="utf-8",
            )
            replay_report = root / "bad_replay.json"
            replay_report.write_text(
                json.dumps({"total_requests": 10, "completed_requests": 8, "failed_requests": 2, "errors": ["HTTP 500"]}),
                encoding="utf-8",
            )

            res = verify_oracle.verify_gate5(
                source_path=fixture,
                exported_path=exp_file,
                captured_raw_path=raw_file,
                replay_report_json=replay_report,
            )
            self.assertFalse(res["passed"])
            self.assertGreaterEqual(res["schema_validation"]["clamped_request_count"], 1)
            self.assertGreaterEqual(res["block_invariant"]["violations"], 1)
            self.assertGreaterEqual(res["tokenizer_parity"]["violations"], 1)
            self.assertTrue(any("extra='forbid'" in e for e in res["errors"]))
            self.assertTrue(any("forbidden raw payload keys" in e for e in res["errors"]))
            self.assertTrue(any("clamped_request_count violation" in e for e in res["errors"]))
            self.assertTrue(any("violated block invariant" in e for e in res["errors"]))
            self.assertTrue(any("LCP drift" in e for e in res["errors"]))
            self.assertTrue(any("Round-trip replay incomplete" in e for e in res["errors"]))


if __name__ == "__main__":
    unittest.main()
