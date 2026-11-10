/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package tracerecorder implements the kv-block-trace-recorder EPP plugin,
// capturing anonymized KV-cache block hashes and request lifecycle telemetry
// into sealed .jsonl.gz trace segments with zero prompt or completion text.
package tracerecorder

// SchemaVersion is the wire schema version emitted on every trace event.
const SchemaVersion = 1

// Session resolution source tags recorded in Event.SessionSource.
const (
	SessionSourceHeader         = "header"
	SessionSourcePrefixFallback = "prefix_fallback"
	SessionSourceRequestID      = "request_id"
)

// Output token count resolution source tags recorded in Event.OutSource.
const (
	OutSourceUsage  = "usage"
	OutSourceEvents = "events"
)

// TerminationNatural indicates a response completed normally without abort or error.
const TerminationNatural = "natural"

// Event represents a single anonymized per-request KV-block trace record
// serialized as one JSON line inside a sealed .jsonl.gz segment.
type Event struct {
	Version          int      `json:"v"`
	KeyFingerprint   string   `json:"key_fingerprint"`
	CaptureRunID     string   `json:"capture_run_id,omitempty"`
	SessionID        string   `json:"session_id"`
	ParentSessionID  string   `json:"parent_session_id,omitempty"`
	SessionSource    string   `json:"session_source"`
	CacheSaltID      string   `json:"cache_salt_id"`
	AgentID          string   `json:"agent_id"`
	SubagentType     string   `json:"subagent_type"`
	TRecvUnix        float64  `json:"t_recv_unix"`
	TTFTS            float64  `json:"ttft_s"`
	APITimeS         float64  `json:"api_time_s"`
	Streaming        bool     `json:"streaming"`
	StatusCode       int      `json:"status_code"`
	Termination      string   `json:"termination"`
	Model            string   `json:"model"`
	TokenizerBackend string   `json:"tokenizer_backend"`
	BlockSize        int      `json:"block_size"`
	In               int      `json:"in"`
	UsageIn          int      `json:"usage_in,omitempty"`
	CachedTokens     int      `json:"cached_tokens,omitempty"`
	Out              int      `json:"out"`
	OutSource        string   `json:"out_source"`
	MaxOutputTokens  *int64   `json:"max_output_tokens,omitempty"`
	MsgCount         int      `json:"msg_count"`
	HasSystemPrompt  bool     `json:"has_system_prompt"`
	HasTools         bool     `json:"has_tools"`
	HashIDs          []uint64 `json:"hash_ids"`
}
