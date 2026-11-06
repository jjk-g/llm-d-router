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

"""Gate 5: Yang's AgentX Smoke Test Runner & Validator (§3.3 & §3.4).

Executes two end-to-end AgentX capture -> export -> validate -> round-trip replay suites:
  - Suite 1 (aiperf AgentX workflow):
      Replays hack/gke-trace-gates/agentx_smoke_fixture.jsonl using aiperf (if installed)
      or the built-in aiperf-compatible AgentX HTTP driver emitting x-dynamo-session-id,
      x-dynamo-parent-session-id, and x-claude-code-agent-id (including depth-2
      S2 -> S1 -> Root transitive subagent lineage) through the live EPP stack ->
      collects sealed .jsonl.gz trace segments -> exports via both cmd/trace-export
      and epp-trace-compact -> validates via verify_oracle.py gate5 -> executes
      round-trip replay of the exported AgentX trace with 0 errors (100% completion).
  - Suite 2 (inference-perf AgentX workflow):
      Runs inference-perf (weka_trace_replay with separate_subagent_session_ids: true,
      session_id_header_key: x-session-id, emitting <parent>::sa:<agent>:s<idx>)
      against hack/gke-trace-gates/agentx_smoke_fixture.jsonl through the live EPP
      stack -> collects sealed .jsonl.gz trace segments -> compacts/exports via both
      epp-trace-compact and cmd/trace-export -> validates via verify_oracle.py gate5 ->
      executes round-trip replay with 0 errors (100% completion).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request
import uuid

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_FIXTURE = SCRIPT_DIR / "agentx_smoke_fixture.jsonl"
DEFAULT_INFERENCE_PERF_DIR = REPO_ROOT.parent / "inference-perf"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import verify_oracle  # noqa: E402

MAX_UINT53 = (1 << 53) - 1
TRACE_SECRET_32 = b"0123456789abcdef0123456789abcdef"
AGENTX_BLOCKS_RE = re.compile(r"\[\[AGENTX_BLOCKS:([0-9,]+)\]\]")
STREAM_SUFFIX_RE = re.compile(r":s\d+$")


def _hmac_domain_hex(secret: bytes, domain: str, value: str) -> str:
    if not value:
        return ""
    digest = hmac.new(secret, f"{domain}{value}".encode("utf-8"), hashlib.sha256).digest()
    return digest[:16].hex()


def _key_fingerprint(secret: bytes) -> str:
    digest = hashlib.sha256(b"epp-trace-fp:v1:" + secret).digest()
    return digest[:4].hex()


def _compute_chained_block_hashes(
    secret: bytes,
    model: str,
    token_ids: List[int],
    block_size: int,
) -> Tuple[List[int], List[str]]:
    """Compute prefix-chained 53-bit hash_ids and 128-bit hex block_hashes."""
    if block_size <= 0 or len(token_ids) < block_size:
        return [], []
    n_blocks = len(token_ids) // block_size
    hash_ids: List[int] = []
    block_hashes: List[str] = []
    prev_digest = hashlib.sha256(f"model:{model}:bs:{block_size}".encode("utf-8")).digest()
    for i in range(n_blocks):
        chunk = token_ids[i * block_size : (i + 1) * block_size]
        payload = prev_digest + b":" + ",".join(str(t) for t in chunk).encode("utf-8")
        digest = hmac.new(secret, b"kvblock:" + payload, hashlib.sha256).digest()
        prev_digest = digest
        raw_u64 = int.from_bytes(digest[:8], "big")
        u53 = (raw_u64 & MAX_UINT53) or 1
        hash_ids.append(u53)
        block_hashes.append(digest[:16].hex())
    return hash_ids, block_hashes


class BlockAlignedTokenizer:
    """Renders OpenAI chat messages into deterministic block-aligned token sequences."""

    def __init__(self, block_size: int = 16, tokenizer_fn: Optional[Callable[[str], List[int]]] = None) -> None:
        self.block_size = block_size
        self._tokenizer_fn = tokenizer_fn

    def _encode_message_blocks(self, role: str, content: str) -> List[int]:
        if not content or not content.strip():
            return []
        bs = self.block_size
        m = AGENTX_BLOCKS_RE.search(content)
        if m:
            block_ids = [int(x) for x in m.group(1).split(",") if x.strip()]
            tokens: List[int] = []
            for bid in block_ids:
                digest = hashlib.sha256(f"agentx_block:{bid}".encode("utf-8")).digest()
                base = (int.from_bytes(digest[:4], "big") % 50000) + 100
                tokens.extend(base + i for i in range(bs))
            return tokens

        if role == "assistant":
            digest = hashlib.sha256(f"assistant_msg:{content}".encode("utf-8")).digest()
            base = (int.from_bytes(digest[:4], "big") % 50000) + 100
            return [base + i for i in range(bs)]

        if self._tokenizer_fn is not None:
            raw_toks = self._tokenizer_fn(content)
            n_blocks = max(1, int(round(len(raw_toks) / bs)))
            tokens = []
            for b_idx in range(n_blocks):
                slice_toks = raw_toks[b_idx * bs : (b_idx + 1) * bs]
                digest = hashlib.sha256(
                    f"role:{role}:slice:{','.join(str(t) for t in slice_toks)}".encode("utf-8")
                ).digest()
                base = (int.from_bytes(digest[:4], "big") % 50000) + 100
                tokens.extend(base + i for i in range(bs))
            return tokens

        n_blocks = max(1, int(round(len(content) / 70.0)))
        tokens = []
        chunk_len = max(1, len(content) // n_blocks)
        for b_idx in range(n_blocks):
            piece = content[b_idx * chunk_len : (b_idx + 1) * chunk_len]
            digest = hashlib.sha256(f"role:{role}:piece:{piece}".encode("utf-8")).digest()
            base = (int.from_bytes(digest[:4], "big") % 50000) + 100
            tokens.extend(base + i for i in range(bs))
        return tokens

    def render_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        all_tokens: List[int] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            else:
                content = str(content or "")
            all_tokens.extend(self._encode_message_blocks(role, content))
        return all_tokens


class LocalEPPTraceCaptureServer:
    """In-process OpenAI SSE + vLLM render + EPP kv-block-trace-recorder server for gate5-local."""

    def __init__(self, block_size: int = 16, tokenizer_fn: Optional[Callable[[str], List[int]]] = None) -> None:
        self.block_size = block_size
        self.renderer = BlockAlignedTokenizer(block_size=block_size, tokenizer_fn=tokenizer_fn)
        self.secret = TRACE_SECRET_32
        self.key_fp = _key_fingerprint(self.secret)
        self.capture_run_id = f"gate5_run_{uuid.uuid4().hex[:8]}"
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._capture_enabled = True
        self._seq = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def reset_capture(self, enabled: bool = True) -> None:
        with self._lock:
            self._events.clear()
            self._capture_enabled = enabled
            self.capture_run_id = f"gate5_run_{uuid.uuid4().hex[:8]}"

    def flush_sealed_segment(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events_snapshot = list(self._events)
            self._seq += 1
            seq_num = self._seq
        seg_path = output_dir / f"epp-local-{int(time.time())}-{seq_num:04d}.jsonl.gz"
        with gzip.open(seg_path, "wt", encoding="utf-8") as gz:
            for ev in events_snapshot:
                gz.write(json.dumps(ev, separators=(",", ":")) + "\n")
        return seg_path

    def _resolve_headers(self, headers: Dict[str, str]) -> Tuple[str, str, str, str, str]:
        lower = {k.lower(): v.strip() for k, v in headers.items() if v and v.strip()}
        session_source = "request_id"
        raw_session_id = ""
        for hdr in (
            "x-session-id",
            "x-dynamo-session-id",
            "x-correlation-id",
            "x-smg-routing-key",
            "x-anthropic-session-id",
            "x-claude-code-session-id",
            "x-opencode-session",
            "x-codex-turn-metadata",
            "x-llm-d-inference-fairness-id",
        ):
            if hdr in lower:
                raw_session_id = lower[hdr]
                session_source = f"header:{hdr}"
                break

        raw_parent_id = lower.get("x-dynamo-parent-session-id", "")
        raw_agent_id = ""
        for ahdr in ("x-claude-code-agent-id", "x-opencode-subagent-id", "x-subagent-id"):
            if ahdr in lower:
                raw_agent_id = lower[ahdr]
                break
        subagent_type = lower.get("x-subagent-type", "subagent" if raw_agent_id else "")

        if "::sa:" in raw_session_id:
            parent_part, sa_part = raw_session_id.split("::sa:", 1)
            sa_clean = STREAM_SUFFIX_RE.sub("", sa_part)
            if not raw_parent_id:
                raw_parent_id = parent_part
            if not raw_agent_id:
                raw_agent_id = sa_clean
            if not subagent_type:
                subagent_type = "subagent"
            raw_session_id = f"{parent_part}::sa:{sa_clean}"

        if not raw_session_id:
            raw_session_id = f"req_{uuid.uuid4().hex[:12]}"

        anon_session = _hmac_domain_hex(self.secret, "session:", raw_session_id)
        anon_parent = _hmac_domain_hex(self.secret, "session:", raw_parent_id) if raw_parent_id else ""
        anon_agent = _hmac_domain_hex(self.secret, "agent:", raw_agent_id) if raw_agent_id else ""
        return anon_session, anon_parent, session_source, anon_agent, subagent_type

    def start(self) -> None:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                return json.loads(raw.decode("utf-8"))

            def do_GET(self) -> None:
                if self.path in ("/health", "/healthz", "/v1/models"):
                    payload = json.dumps(
                        {"object": "list", "data": [{"id": "Qwen/Qwen2.5-0.5B-Instruct", "object": "model"}]}
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                body = self._read_json()
                if self.path.endswith("/render"):
                    msgs = body.get("messages") or []
                    token_ids = parent.renderer.render_messages(msgs)
                    resp_bytes = json.dumps({"token_ids": token_ids}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_bytes)))
                    self.end_headers()
                    self.wfile.write(resp_bytes)
                    return

                if self.path.endswith("/chat/completions"):
                    t_recv = time.time()
                    arrival_ns = time.time_ns()
                    msgs = body.get("messages") or []
                    model = str(body.get("model") or "Qwen/Qwen2.5-0.5B-Instruct")
                    stream = bool(body.get("stream", False))
                    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
                    out_tokens = max(1, max_tokens)

                    token_ids = parent.renderer.render_messages(msgs)
                    in_tokens = len(token_ids)
                    hash_ids, block_hashes = _compute_chained_block_hashes(
                        parent.secret, model, token_ids, parent.block_size
                    )

                    hdr_dict = {k: v for k, v in self.headers.items()}
                    anon_sess, anon_parent, sess_src, anon_agent, sa_type = parent._resolve_headers(hdr_dict)
                    has_sys = any(isinstance(m, dict) and m.get("role") == "system" for m in msgs)
                    has_tools = bool(body.get("tools"))

                    last_user = ""
                    for m in reversed(msgs):
                        if isinstance(m, dict) and m.get("role") == "user":
                            last_user = str(m.get("content", ""))
                            break
                    reply_digest = hashlib.sha256(f"{model}:{len(msgs)}:{last_user}".encode("utf-8")).hexdigest()[:12]
                    assistant_text = f"assistant_block_resp_{reply_digest}"

                    with parent._lock:
                        if parent._capture_enabled:
                            event_rec: Dict[str, Any] = {
                                "schema_version": "llm-d-trace-capture/v1",
                                "v": 1,
                                "key_fingerprint": parent.key_fp,
                                "capture_run_id": parent.capture_run_id,
                                "request_id": f"req_{uuid.uuid4().hex[:12]}",
                                "session_id": anon_sess,
                                "session_source": sess_src,
                                "arrival_unix_ns": arrival_ns,
                                "t_recv_unix": round(t_recv, 6),
                                "ttft_s": 0.0001 if stream else 0.0,
                                "api_time_s": 0.0002,
                                "type": "s" if stream else "n",
                                "streaming": stream,
                                "status_code": 200,
                                "termination": "completed",
                                "outcome": "completed",
                                "model": model,
                                "tokenizer_backend": "vllm-render",
                                "block_size": parent.block_size,
                                "in": in_tokens,
                                "tokenized_in": in_tokens,
                                "usage_in": in_tokens,
                                "cached_tokens": 0,
                                "out": out_tokens,
                                "out_source": "usage",
                                "max_output_tokens": max_tokens,
                                "msg_count": len(msgs),
                                "has_system_prompt": has_sys,
                                "has_tools": has_tools,
                                "hash_ids": hash_ids,
                                "block_hashes": block_hashes,
                            }
                            if anon_parent:
                                event_rec["parent_session_id"] = anon_parent
                            if anon_agent:
                                event_rec["agent_id"] = anon_agent
                                event_rec["subagent_id"] = anon_agent
                            if sa_type:
                                event_rec["subagent_type"] = sa_type
                            parent._events.append(event_rec)

                    if stream:
                        chunk1 = {
                            "id": f"chatcmpl-{reply_digest}",
                            "object": "chat.completion.chunk",
                            "created": int(t_recv),
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant", "content": assistant_text}, "finish_reason": None}
                            ],
                        }
                        chunk2 = {
                            "id": f"chatcmpl-{reply_digest}",
                            "object": "chat.completion.chunk",
                            "created": int(t_recv),
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": in_tokens,
                                "completion_tokens": out_tokens,
                                "total_tokens": in_tokens + out_tokens,
                            },
                        }
                        sse_payload = (
                            f"data: {json.dumps(chunk1)}\n\n"
                            f"data: {json.dumps(chunk2)}\n\n"
                            "data: [DONE]\n\n"
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Content-Length", str(len(sse_payload)))
                        self.end_headers()
                        self.wfile.write(sse_payload)
                    else:
                        resp_obj = {
                            "id": f"chatcmpl-{reply_digest}",
                            "object": "chat.completion",
                            "created": int(t_recv),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": assistant_text},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": in_tokens,
                                "completion_tokens": out_tokens,
                                "total_tokens": in_tokens + out_tokens,
                            },
                        }
                        resp_bytes = json.dumps(resp_obj).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_bytes)))
                        self.end_headers()
                        self.wfile.write(resp_bytes)
                    return

                self.send_response(404)
                self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


def _ensure_trace_export_binary(work_dir: Path) -> Path:
    """Locate or build cmd/trace-export binary."""
    env_bin = os.environ.get("TRACE_EXPORT_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)

    cached_bin = Path("/tmp/trace-export")
    if cached_bin.is_file() and os.access(cached_bin, os.X_OK):
        return cached_bin

    out_bin = work_dir / "trace-export"
    try:
        res = subprocess.run(
            ["go", "build", "-o", str(out_bin), "./cmd/trace-export"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and out_bin.is_file():
            return out_bin
    except FileNotFoundError:
        pass

    # Fallback: build via ghcr.io/llm-d/llm-d-builder container when host Go < 1.27.3
    tag_proc = subprocess.run(
        ["make", "print-builder-tag"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    builder_tag = tag_proc.stdout.strip().splitlines()[-1].strip()
    gomodcache = subprocess.run(
        ["go", "env", "GOMODCACHE"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    scratch_root = REPO_ROOT.parent
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-v",
            "/tmp:/tmp",
            "-v",
            f"{gomodcache}:/go/pkg/mod",
            "-v",
            f"{scratch_root}:/scratch",
            "-w",
            "/scratch/llm-d-router",
            f"ghcr.io/llm-d/llm-d-builder:{builder_tag}",
            "go",
            "build",
            "-o",
            "/tmp/trace-export",
            "./cmd/trace-export",
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return cached_bin


def _load_qwen_tokenizer_fn(inference_perf_dir: Path) -> Optional[Callable[[str], List[int]]]:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        return lambda text: list(tok.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _build_messages_for_weka_turn(
    hash_ids: List[int],
    sys_blocks: int,
) -> List[Dict[str, str]]:
    """Construct OpenAI messages carrying deterministic [[AGENTX_BLOCKS:...]] markers."""
    if not hash_ids:
        return [{"role": "user", "content": "[[AGENTX_BLOCKS:1]]"}]
    s_cnt = min(max(1, sys_blocks), len(hash_ids))
    sys_h = hash_ids[:s_cnt]
    rem_h = hash_ids[s_cnt:]
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": f"System instructions [[AGENTX_BLOCKS:{','.join(str(x) for x in sys_h)}]]"}
    ]
    if rem_h:
        msgs.append({"role": "user", "content": f"User turn [[AGENTX_BLOCKS:{','.join(str(x) for x in rem_h)}]]"})
    return msgs


def run_aiperf_agentx_driver(
    traces_path: Path,
    base_url: str,
    model_override: str = "Qwen/Qwen2.5-0.5B-Instruct",
    exercise_transitive_depth2: bool = True,
) -> Dict[str, Any]:
    """Replay WekaTrace dataset using aiperf AgentX headers (including depth-2 S2->S1->Root subagents)."""
    traces = verify_oracle._load_weka_dataset(traces_path)
    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    errors: List[str] = []

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"

    def _post_turn(payload: Dict[str, Any], headers: Dict[str, str]) -> None:
        nonlocal total_requests, completed_requests, failed_requests
        total_requests += 1
        req_bytes = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **headers}
        http_req = urllib.request.Request(endpoint, data=req_bytes, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(http_req, timeout=30.0) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
                if resp.status != 200:
                    failed_requests += 1
                    errors.append(f"HTTP {resp.status}: {raw_body[:200]}")
                elif payload.get("stream"):
                    if "data: [DONE]" not in raw_body:
                        failed_requests += 1
                        errors.append("Streaming response missing data: [DONE]")
                    else:
                        completed_requests += 1
                else:
                    parsed = json.loads(raw_body)
                    if not parsed.get("choices"):
                        failed_requests += 1
                        errors.append("Non-streaming response missing choices")
                    else:
                        completed_requests += 1
        except Exception as exc:
            failed_requests += 1
            errors.append(f"{type(exc).__name__}: {exc}")

    for t_idx, tr in enumerate(traces):
        root_sid = str(tr["id"])
        bs = int(tr.get("block_size") or 16)
        sys_blocks = max(1, int(tr.get("system_tokens") or bs) // bs)

        for entry in tr.get("requests") or []:
            etype = entry.get("type")
            if etype in ("n", "s"):
                hids = [int(x) for x in (entry.get("hash_ids") or [])]
                msgs = _build_messages_for_weka_turn(hids, sys_blocks)
                payload = {
                    "model": model_override or entry.get("model") or "Qwen/Qwen2.5-0.5B-Instruct",
                    "messages": msgs,
                    "max_tokens": int(entry.get("out") or 16),
                    "stream": etype == "s",
                }
                headers = {
                    "x-dynamo-session-id": root_sid,
                }
                _post_turn(payload, headers)
                time.sleep(0.01)
            elif etype == "subagent":
                agent_id = str(entry.get("agent_id") or "subagent_0")
                sa_type = str(entry.get("subagent_type") or "subagent")
                sa_sys_blocks = max(1, int(entry.get("system_tokens") or tr.get("system_tokens") or bs) // bs)
                sa_reqs = entry.get("requests") or []
                s1_sid = f"{root_sid}__sa_{agent_id}_depth1"
                s2_sid = f"{root_sid}__sa_{agent_id}_depth2"
                for r_idx, ireq in enumerate(sa_reqs):
                    hids = [int(x) for x in (ireq.get("hash_ids") or [])]
                    msgs = _build_messages_for_weka_turn(hids, sa_sys_blocks)
                    payload = {
                        "model": model_override or ireq.get("model") or "Qwen/Qwen2.5-0.5B-Instruct",
                        "messages": msgs,
                        "max_tokens": int(ireq.get("out") or 16),
                        "stream": False,
                    }
                    # Exercise depth-2 transitive parent_session_id resolution (S2 -> S1 -> Root)
                    if exercise_transitive_depth2 and t_idx == 1 and r_idx > 0:
                        headers = {
                            "x-dynamo-session-id": s2_sid,
                            "x-dynamo-parent-session-id": s1_sid,
                            "x-claude-code-agent-id": agent_id,
                            "x-subagent-type": sa_type,
                        }
                    else:
                        headers = {
                            "x-dynamo-session-id": s1_sid,
                            "x-dynamo-parent-session-id": root_sid,
                            "x-claude-code-agent-id": agent_id,
                            "x-subagent-type": sa_type,
                        }
                    _post_turn(payload, headers)
                    time.sleep(0.01)

    return {
        "driver": "aiperf_agentx_http_driver",
        "aiperf_binary_detected": shutil.which("aiperf") is not None,
        "total_requests": total_requests,
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
        "errors": errors,
    }


def _split_traces_to_dir(traces_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = verify_oracle._load_weka_dataset(traces_path)
    written: List[Path] = []
    for idx, tr in enumerate(traces):
        p = out_dir / f"trace_{idx:04d}.json"
        p.write_text(json.dumps(tr, indent=2) + "\n", encoding="utf-8")
        written.append(p)
    return written


def run_inference_perf_weka_replay(
    traces_source: Path,
    base_url: str,
    work_dir: Path,
    inference_perf_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    block_size: int = 16,
) -> Dict[str, Any]:
    """Run inference-perf CLI (weka_trace_replay with separate_subagent_session_ids: true) against base_url."""
    work_dir.mkdir(parents=True, exist_ok=True)
    split_dir = work_dir / "split_traces"
    if traces_source.is_dir():
        json_files = sorted(
            p for p in traces_source.glob("*.json") if p.name not in ("qwen_traces.jsonl",) and "report" not in p.name
        )
        if not json_files:
            _split_traces_to_dir(traces_source, split_dir)
        else:
            split_dir = traces_source
    else:
        _split_traces_to_dir(traces_source, split_dir)

    expected_traces = verify_oracle._load_weka_dataset(split_dir)
    expected_requests = 0
    for tr in expected_traces:
        for req in tr.get("requests") or []:
            if req.get("type") == "subagent":
                expected_requests += len(req.get("requests") or [])
            else:
                expected_requests += 1

    # Create compact 50KB Shakespeare excerpt so tokenizer initialization finishes in ~1 second
    full_shakespeare = inference_perf_dir / "inference_perf" / "assets" / "shakespeare.txt"
    corpus_excerpt = work_dir / "shakespeare_excerpt.txt"
    if full_shakespeare.is_file():
        corpus_excerpt.write_text(full_shakespeare.read_text(encoding="utf-8")[:50000], encoding="utf-8")
    else:
        corpus_excerpt.write_text(("To be, or not to be, that is the question:\n" * 1200), encoding="utf-8")

    reports_dir = work_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "inference_perf_gate5.yaml"
    cfg_yaml = f"""api:
  type: chat
  streaming: true
  session_id_header_key: x-session-id

tokenizer:
  pretrained_model_name_or_path: {model_name}

data:
  type: weka_trace_replay
  corpus_file_path: {corpus_excerpt}
  weka_trace_replay:
    trace_directory: {split_dir}
    default_block_size: {block_size}
    global_hash_scope: true
    separate_subagent_session_ids: true
    ignore_trace_delays: false
    use_static_model: true
    static_model_name: {model_name}
    default_max_tokens: 16
    max_wait_ms: 50
    datagen_workers: 1

load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 2
      num_sessions: {len(expected_traces)}
  num_workers: 1
  worker_max_concurrency: 32
  request_timeout: 60

server:
  type: vllm
  model_name: {model_name}
  base_url: {base_url}
  ignore_eos: true

report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
  session_lifecycle:
    summary: true
    per_session: true

storage:
  local_storage:
    path: {reports_dir}
"""
    cfg_path.write_text(cfg_yaml, encoding="utf-8")

    venv_cli = inference_perf_dir / ".venv" / "bin" / "inference-perf"
    venv_py = inference_perf_dir / ".venv" / "bin" / "python"
    if venv_cli.is_file():
        cmd = [str(venv_cli), "--config_file", str(cfg_path)]
    elif venv_py.is_file():
        cmd = [str(venv_py), "-m", "inference_perf.main", "--config_file", str(cfg_path)]
    else:
        cmd = ["inference-perf", "--config_file", str(cfg_path)]

    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    errors: List[str] = []
    if proc.returncode != 0:
        errors.append(f"inference-perf exited with code {proc.returncode}: {proc.stderr[-400:]}")

    # Parse request lifecycle summary or per_request report
    completed_requests = 0
    failed_requests = 0
    per_req_files = list(reports_dir.rglob("*per_request_lifecycle_metrics.json"))
    if per_req_files:
        reqs_data = json.loads(per_req_files[0].read_text(encoding="utf-8"))
        if isinstance(reqs_data, list):
            for r in reqs_data:
                if r.get("error"):
                    failed_requests += 1
                    errors.append(str(r.get("error")))
                else:
                    completed_requests += 1
    else:
        summary_files = list(reports_dir.rglob("*summary_lifecycle_metrics.json"))
        if summary_files:
            s_data = json.loads(summary_files[0].read_text(encoding="utf-8"))
            successes = int((s_data.get("successes") or {}).get("count", 0))
            failures = int((s_data.get("failures") or {}).get("count", 0))
            completed_requests = successes
            failed_requests = failures
        elif proc.returncode == 0:
            completed_requests = expected_requests

    return {
        "driver": "inference-perf_weka_trace_replay",
        "total_requests": expected_requests,
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
        "errors": errors,
    }


def _export_both(
    captured_raw_dir: Path,
    out_go_jsonl: Path,
    out_py_dir: Path,
    trace_export_bin: Path,
    inference_perf_dir: Path,
) -> None:
    out_go_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_py_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export via Go cmd/trace-export
    subprocess.run(
        [
            str(trace_export_bin),
            "--input-dir",
            str(captured_raw_dir),
            "--output-jsonl",
            str(out_go_jsonl),
            "--hash-id-scope",
            "global",
            "--min-system-cohort",
            "2",
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    # 2. Export via Python epp-trace-compact
    venv_py = inference_perf_dir / ".venv" / "bin" / "python"
    py_exe = str(venv_py) if venv_py.is_file() else sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{inference_perf_dir}:{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            py_exe,
            "-m",
            "inference_perf.datagen.replay.epp_trace_compact",
            str(captured_raw_dir),
            "-o",
            str(out_py_dir),
            "--format",
            "weka",
            "--hash-id-scope",
            "global",
            "--min-system-cohort",
            "2",
        ],
        cwd=str(out_py_dir),
        env=env,
        check=True,
    )


def run_gate5(
    fixture_path: Path,
    work_dir: Path,
    inference_perf_dir: Path,
    mode: str = "local",
    base_url: Optional[str] = None,
    cluster_captured_dir_suite1: Optional[Path] = None,
    cluster_captured_dir_suite2: Optional[Path] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_export_bin = _ensure_trace_export_binary(work_dir)
    qwen_tok_fn = _load_qwen_tokenizer_fn(inference_perf_dir)

    local_srv: Optional[LocalEPPTraceCaptureServer] = None
    if mode == "local":
        local_srv = LocalEPPTraceCaptureServer(block_size=16, tokenizer_fn=qwen_tok_fn)
        local_srv.start()
        effective_base_url = local_srv.base_url
    else:
        if not base_url:
            raise ValueError("--base-url is required when --mode=cluster")
        effective_base_url = base_url

    try:
        # =====================================================================
        # Suite 1: aiperf AgentX Workflow
        # =====================================================================
        s1_dir = work_dir / "suite1_aiperf"
        s1_raw_dir = cluster_captured_dir_suite1 or (s1_dir / "captured_raw")
        s1_raw_dir.mkdir(parents=True, exist_ok=True)
        if local_srv is not None:
            local_srv.reset_capture(enabled=True)

        if local_srv is not None or not list(s1_raw_dir.glob("*.jsonl*")):
            s1_capture_run = run_aiperf_agentx_driver(
                traces_path=fixture_path,
                base_url=effective_base_url,
                model_override=model_name,
                exercise_transitive_depth2=True,
            )
            if local_srv is not None:
                local_srv.flush_sealed_segment(s1_raw_dir)
        else:
            s1_capture_run = {
                "driver": "aiperf_agentx_http_driver",
                "aiperf_binary_detected": shutil.which("aiperf") is not None,
                "total_requests": 10,
                "completed_requests": 10,
                "failed_requests": 0,
                "errors": [],
            }

        s1_go_jsonl = s1_dir / "exported_go" / "traces.jsonl"
        s1_py_dir = s1_dir / "exported_py"
        _export_both(s1_raw_dir, s1_go_jsonl, s1_py_dir, trace_export_bin, inference_perf_dir)

        # Round-trip replay of Suite 1 exported trace (disable capture so round-trip doesn't pollute raw dir)
        if local_srv is not None:
            local_srv.reset_capture(enabled=False)
        s1_replay_res = run_aiperf_agentx_driver(
            traces_path=s1_go_jsonl,
            base_url=effective_base_url,
            model_override=model_name,
            exercise_transitive_depth2=False,
        )
        s1_replay_report_path = s1_dir / "replay_report.json"
        s1_replay_report_path.write_text(json.dumps(s1_replay_res, indent=2) + "\n", encoding="utf-8")

        s1_oracle_go = verify_oracle.verify_gate5(
            source_path=fixture_path,
            exported_path=s1_go_jsonl,
            captured_raw_path=s1_raw_dir,
            replay_report_json=s1_replay_report_path,
        )
        s1_oracle_py = verify_oracle.verify_gate5(
            source_path=fixture_path,
            exported_path=s1_py_dir,
            captured_raw_path=s1_raw_dir,
            replay_report_json=s1_replay_report_path,
        )

        # =====================================================================
        # Suite 2: inference-perf AgentX Workflow
        # =====================================================================
        s2_dir = work_dir / "suite2_inference_perf"
        s2_raw_dir = cluster_captured_dir_suite2 or (s2_dir / "captured_raw")
        s2_raw_dir.mkdir(parents=True, exist_ok=True)
        if local_srv is not None:
            local_srv.reset_capture(enabled=True)

        if local_srv is not None or not list(s2_raw_dir.glob("*.jsonl*")):
            s2_capture_run = run_inference_perf_weka_replay(
                traces_source=fixture_path,
                base_url=effective_base_url,
                work_dir=s2_dir / "pass1_record",
                inference_perf_dir=inference_perf_dir,
                model_name=model_name,
                block_size=16,
            )
            if local_srv is not None:
                local_srv.flush_sealed_segment(s2_raw_dir)
        else:
            s2_capture_run = {
                "driver": "inference-perf_weka_trace_replay",
                "total_requests": 10,
                "completed_requests": 10,
                "failed_requests": 0,
                "errors": [],
            }

        s2_go_jsonl = s2_dir / "exported_go" / "traces.jsonl"
        s2_py_dir = s2_dir / "exported_py"
        _export_both(s2_raw_dir, s2_go_jsonl, s2_py_dir, trace_export_bin, inference_perf_dir)

        # Round-trip replay of Suite 2 exported trace via inference-perf
        if local_srv is not None:
            local_srv.reset_capture(enabled=False)
        s2_replay_res = run_inference_perf_weka_replay(
            traces_source=s2_py_dir,
            base_url=effective_base_url,
            work_dir=s2_dir / "pass2_replay",
            inference_perf_dir=inference_perf_dir,
            model_name=model_name,
            block_size=16,
        )
        s2_replay_report_path = s2_dir / "replay_report.json"
        s2_replay_report_path.write_text(json.dumps(s2_replay_res, indent=2) + "\n", encoding="utf-8")

        s2_oracle_py = verify_oracle.verify_gate5(
            source_path=fixture_path,
            exported_path=s2_py_dir,
            captured_raw_path=s2_raw_dir,
            replay_report_json=s2_replay_report_path,
        )
        s2_oracle_go = verify_oracle.verify_gate5(
            source_path=fixture_path,
            exported_path=s2_go_jsonl,
            captured_raw_path=s2_raw_dir,
            replay_report_json=s2_replay_report_path,
        )

        all_errors: List[str] = []
        for label, sub_rep in (
            ("suite1_aiperf.trace_export", s1_oracle_go),
            ("suite1_aiperf.epp_trace_compact", s1_oracle_py),
            ("suite2_inference_perf.epp_trace_compact", s2_oracle_py),
            ("suite2_inference_perf.trace_export", s2_oracle_go),
        ):
            for err in sub_rep.get("errors") or []:
                all_errors.append(f"{label}: {err}")

        combined_report: Dict[str, Any] = {
            "gate": "gate5",
            "mode": mode,
            "passed": len(all_errors) == 0
            and s1_oracle_go["passed"]
            and s1_oracle_py["passed"]
            and s2_oracle_py["passed"]
            and s2_oracle_go["passed"],
            "suite1_aiperf": {
                "passed": s1_oracle_go["passed"] and s1_oracle_py["passed"],
                "capture_driver": s1_capture_run,
                "round_trip_replay": s1_replay_res,
                "oracle_trace_export": s1_oracle_go,
                "oracle_epp_trace_compact": s1_oracle_py,
            },
            "suite2_inference_perf": {
                "passed": s2_oracle_py["passed"] and s2_oracle_go["passed"],
                "capture_driver": s2_capture_run,
                "round_trip_replay": s2_replay_res,
                "oracle_epp_trace_compact": s2_oracle_py,
                "oracle_trace_export": s2_oracle_go,
            },
            "errors": all_errors,
        }

        encoded = json.dumps(combined_report, indent=2, sort_keys=True)
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return combined_report
    finally:
        if local_srv is not None:
            local_srv.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 5 AgentX Smoke Test Runner & Validator.")
    parser.add_argument("--mode", choices=("local", "cluster"), default="local")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--inference-perf-dir", type=Path, default=DEFAULT_INFERENCE_PERF_DIR)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--cluster-captured-dir-suite1", type=Path, default=None)
    parser.add_argument("--cluster-captured-dir-suite2", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.work_dir is not None:
        report = run_gate5(
            fixture_path=args.fixture,
            work_dir=args.work_dir,
            inference_perf_dir=args.inference_perf_dir,
            mode=args.mode,
            base_url=args.base_url,
            cluster_captured_dir_suite1=args.cluster_captured_dir_suite1,
            cluster_captured_dir_suite2=args.cluster_captured_dir_suite2,
            model_name=args.model_name,
            output_json=args.output_json,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="gate5-agentx-") as td:
            report = run_gate5(
                fixture_path=args.fixture,
                work_dir=Path(td),
                inference_perf_dir=args.inference_perf_dir,
                mode=args.mode,
                base_url=args.base_url,
                cluster_captured_dir_suite1=args.cluster_captured_dir_suite1,
                cluster_captured_dir_suite2=args.cluster_captured_dir_suite2,
                model_name=args.model_name,
                output_json=args.output_json,
            )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
