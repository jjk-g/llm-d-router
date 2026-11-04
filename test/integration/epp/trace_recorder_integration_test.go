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

package epp

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/tracerecorder"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
)

const envTraceGoldenOutDir = "EPP_TRACE_GOLDEN_OUT_DIR"

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// TestTraceRecorder_Step2A_StreamingMultiTurnAndGoldenExport exercises Gate 2 Step 2A:
// in-process Envoy ext-proc gRPC streaming capture across 3 multi-turn sessions sharing a
// 2-block (32-token, block_size=16) system prompt via a deterministic vLLM /render stub,
// including:
//   - multi-turn prefix growth across sessions,
//   - 1 synchronous subagent branch in Session 1 (Turn 0 -> diverging subagent -> Turn 1 resuming Turn 0),
//   - 1 HTTP-200 mid-prefill client disconnect (out == 0, termination == "client-disconnect"),
//   - 1 HTTP-500 upstream error (status_code == 500, termination == "error"),
//   - SSE Usage.CompletionTokens ("usage") and SSE StreamedEvents ("events") fallback,
//   - zero canary leakage and atomic .jsonl.gz sealing (plus EPP_TRACE_GOLDEN_OUT_DIR export).
func TestTraceRecorder_Step2A_StreamingMultiTurnAndGoldenExport(t *testing.T) {
	ctx := t.Context()

	renderSrv := newDeterministicRenderServer(t)
	t.Cleanup(renderSrv.Close)

	workDir := t.TempDir()
	outDir := filepath.Join(workDir, "traces_out")
	stageDir := filepath.Join(workDir, "traces_staging")
	secFile := filepath.Join(workDir, "capture_secret")
	const step2ASecret = "step2a-hermetic-hmac-secret-32by!"
	require.NoError(t, os.WriteFile(secFile, []byte(step2ASecret+"\n"), 0o600))

	cfgYAML := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: mock-metrics-source
  - type: agent-identity
  - type: token-producer
    parameters:
      modelName: %s
      vllm:
        url: %q
  - name: trace-recorder
    type: kv-block-trace-recorder
    parameters:
      outputDir: %q
      stagingDir: %q
      podName: "epp-step2a"
      secretFilePath: %q
      blockSizeTokens: 16
      sampleRateBPS: 10000
schedulingProfiles:
  - name: default
requestHandler:
  parsers:
    - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
    - pluginRef: mock-metrics-source
`, modelMyModelTarget, renderSrv.URL, outDir, stageDir, secFile)

	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(cfgYAML)).WithBaseResources()
	pods := []PodState{P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	recPlugin := h.Runner.PluginHandle.Plugin("trace-recorder")
	require.NotNil(t, recPlugin, "expected trace-recorder plugin in Runner.PluginHandle")
	rec, ok := recPlugin.(*tracerecorder.Recorder)
	require.True(t, ok, "expected *tracerecorder.Recorder")

	// Canary strings that MUST NEVER appear in sealed .jsonl.gz output.
	const (
		sharedSysPrompt = "CANARY_SHARED_COHORT_SYSTEM_PROMPT_DO_NOT_LEAK_7741"
		toolFnName      = "CANARY_TOOL_SCHEMA_FN_DO_NOT_LEAK_9912"
		rawSess1        = "CANARY_RAW_SESSION_ALPHA_001"
		rawSess2        = "CANARY_RAW_SESSION_BETA_002"
		rawSess3        = "CANARY_RAW_SESSION_GAMMA_003"
	)

	var reqSeq atomic.Int64
	buildChatReq := func(rawSession string, msgs []chatMsg) []*extProcPb.ProcessingRequest {
		seq := reqSeq.Add(1)
		payload := map[string]any{
			"model":    modelMyModel,
			"stream":   true,
			"messages": msgs,
			"tools": []map[string]any{
				{
					"type": "function",
					"function": map[string]any{
						"name":        toolFnName,
						"description": "Canary tool definition",
					},
				},
			},
		}
		rawBody, err := json.Marshal(payload)
		require.NoError(t, err)

		headers := map[string]string{
			":path":                               "/v1/chat/completions",
			"content-type":                        "application/json",
			metadata.ObjectiveKey:                 modelMyModel,
			metadata.ModelNameRewriteKey:          modelMyModelTarget,
			reqcommon.RequestIDHeaderKey:          fmt.Sprintf("step2a-req-%d", seq),
			agentidentity.ClaudeCodeSessionHeader: rawSession,
		}
		return fwkepp.ReqRaw(headers, string(rawBody))
	}

	streamSSEWithUsage := func(rawSession string, msgs []chatMsg, promptTokens, completionTokens int) {
		reqs := buildChatReq(rawSession, msgs)
		respReqs := ReqResponseOnly(
			map[string]string{"content-type": "text/event-stream", ":status": "200"},
			fmt.Sprintf("data: {\"id\":\"cmpl\",\"object\":\"chat.completion.chunk\",\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tok \"}}]}\n\n", modelMyModelTarget),
			fmt.Sprintf("data: {\"id\":\"cmpl\",\"object\":\"chat.completion.chunk\",\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
				modelMyModelTarget, promptTokens, completionTokens, promptTokens+completionTokens),
			"data: [DONE]\n\n",
		)
		reqs = append(reqs, respReqs...)

		streamClient, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(ctx)
		require.NoError(t, err)
		responses, err := fwkepp.StreamedRequest(t, streamClient, reqs, len(reqs))
		require.NoError(t, err)
		require.Len(t, responses, len(reqs))
		time.Sleep(20 * time.Millisecond)
	}

	streamSSEEventsOnly := func(rawSession string, msgs []chatMsg, eventCount int) {
		reqs := buildChatReq(rawSession, msgs)
		chunks := make([]string, 0, eventCount+1)
		for i := range eventCount {
			chunks = append(chunks, fmt.Sprintf(
				"data: {\"id\":\"cmpl-%d\",\"object\":\"chat.completion.chunk\",\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"w%d \"}}]}\n\n",
				i, modelMyModelTarget, i,
			))
		}
		chunks = append(chunks, "data: [DONE]\n\n")
		respReqs := ReqResponseOnly(
			map[string]string{"content-type": "text/event-stream", ":status": "200"},
			chunks...,
		)
		reqs = append(reqs, respReqs...)

		streamClient, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(ctx)
		require.NoError(t, err)
		responses, err := fwkepp.StreamedRequest(t, streamClient, reqs, len(reqs))
		require.NoError(t, err)
		require.Len(t, responses, len(reqs))
		time.Sleep(20 * time.Millisecond)
	}

	// --- Session 1: Parent Turn 0 -> Synchronous Subagent Branch -> Parent Turn 1 (resuming Turn 0) ---
	s1Turn0Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S1_USER_TURN_0_ANALYZE_REPO"},
	}
	streamSSEWithUsage(rawSess1, s1Turn0Msgs, 64, 16)

	s1SubagentMsgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S1_SUBAGENT_BRANCH_SEARCH_SYMBOLS"},
	}
	streamSSEWithUsage(rawSess1, s1SubagentMsgs, 64, 12)

	s1Turn1Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S1_USER_TURN_0_ANALYZE_REPO"},
		{Role: "assistant", Content: "CANARY_S1_ASSISTANT_REPLY_0"},
		{Role: "user", Content: "CANARY_S1_USER_TURN_1_FOLLOWUP_PATCH"},
	}
	streamSSEWithUsage(rawSess1, s1Turn1Msgs, 112, 16)

	// --- Session 2: Turn 0 -> Turn 1 (StreamedEvents fallback) -> Turn 2 (HTTP-200 mid-prefill disconnect) ---
	s2Turn0Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S2_USER_TURN_0_EXPLAIN_CACHE"},
	}
	streamSSEWithUsage(rawSess2, s2Turn0Msgs, 64, 16)

	s2Turn1Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S2_USER_TURN_0_EXPLAIN_CACHE"},
		{Role: "assistant", Content: "CANARY_S2_ASSISTANT_REPLY_0"},
		{Role: "user", Content: "CANARY_S2_USER_TURN_1_BENCHMARK_PLAN"},
	}
	streamSSEEventsOnly(rawSess2, s2Turn1Msgs, 16)

	// Session 2 Turn 2: HTTP-200 response headers arrive, then client disconnects mid-prefill before any SSE chunk!
	s2Turn2Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S2_USER_TURN_0_EXPLAIN_CACHE"},
		{Role: "assistant", Content: "CANARY_S2_ASSISTANT_REPLY_0"},
		{Role: "user", Content: "CANARY_S2_USER_TURN_1_BENCHMARK_PLAN"},
		{Role: "assistant", Content: "CANARY_S2_ASSISTANT_REPLY_1"},
		{Role: "user", Content: "CANARY_S2_USER_TURN_2_CLIENT_DISCONNECTS"},
	}
	{
		discCtx, discCancel := context.WithCancel(ctx)
		discClient, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(discCtx)
		require.NoError(t, err)

		discReqs := buildChatReq(rawSess2, s2Turn2Msgs)
		discReqs = append(discReqs, ReqResponseOnly(
			map[string]string{"content-type": "text/event-stream", ":status": "200"},
		)...)

		responses, err := fwkepp.StreamedRequest(t, discClient, discReqs, 3)
		require.NoError(t, err)
		require.Len(t, responses, 3)

		// Tear down the gRPC stream mid-prefill so StreamingServer's defer records client-disconnect (out == 0).
		discCancel()
		require.Eventually(t, func() bool {
			return rec.RecordedRecords() == 6
		}, 5*time.Second, 10*time.Millisecond, "timed out waiting for mid-prefill disconnect event to finalize")
		time.Sleep(20 * time.Millisecond)
	}

	// --- Session 3: Turn 0 -> Turn 1 -> Turn 2 (HTTP-500 upstream error) ---
	s3Turn0Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S3_USER_TURN_0_DEBUG_LATENCY"},
	}
	streamSSEWithUsage(rawSess3, s3Turn0Msgs, 64, 16)

	s3Turn1Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S3_USER_TURN_0_DEBUG_LATENCY"},
		{Role: "assistant", Content: "CANARY_S3_ASSISTANT_REPLY_0"},
		{Role: "user", Content: "CANARY_S3_USER_TURN_1_CHECK_METRICS"},
	}
	streamSSEWithUsage(rawSess3, s3Turn1Msgs, 112, 16)

	s3Turn2Msgs := []chatMsg{
		{Role: "system", Content: sharedSysPrompt},
		{Role: "user", Content: "CANARY_S3_USER_TURN_0_DEBUG_LATENCY"},
		{Role: "assistant", Content: "CANARY_S3_ASSISTANT_REPLY_0"},
		{Role: "user", Content: "CANARY_S3_USER_TURN_1_CHECK_METRICS"},
		{Role: "assistant", Content: "CANARY_S3_ASSISTANT_REPLY_1"},
		{Role: "user", Content: "CANARY_S3_USER_TURN_2_UPSTREAM_500"},
	}
	{
		errReqs := buildChatReq(rawSess3, s3Turn2Msgs)
		errReqs = append(errReqs, ReqResponseOnly(
			map[string]string{"content-type": "application/json", ":status": "500"},
			`{"error":{"message":"upstream internal error","type":"server_error"}}`,
		)...)

		errClient, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(ctx)
		require.NoError(t, err)
		responses, err := fwkepp.StreamedRequest(t, errClient, errReqs, 4)
		require.NoError(t, err)
		require.Len(t, responses, 4)
	}

	require.Eventually(t, func() bool {
		return rec.RecordedRecords() == 9
	}, 5*time.Second, 10*time.Millisecond)
	assert.Zero(t, rec.DroppedRecords())

	// Close the sink to flush and atomically seal active staging writes into outputDir.
	require.NoError(t, rec.Close(ctx))

	// Verify stagingDir has zero lingering .tmp files and outputDir has sealed .jsonl.gz segment(s).
	stageEntries, err := os.ReadDir(stageDir)
	require.NoError(t, err)
	assert.Empty(t, stageEntries, "stagingDir must be empty after Close()")

	events, gzFiles, rawText := readSealedTraceSegments(t, outDir)
	require.NotEmpty(t, gzFiles, "expected at least 1 sealed .jsonl.gz segment")
	require.Len(t, events, 9, "expected exactly 9 recorded events across 3 sessions")

	// Zero canary leak check across raw uncompressed JSONL.
	for _, canary := range []string{
		sharedSysPrompt, toolFnName, rawSess1, rawSess2, rawSess3,
		"CANARY_S1_", "CANARY_S2_", "CANARY_S3_",
	} {
		assert.NotContains(t, rawText, canary, "canary leaked into sealed .jsonl.gz")
	}

	// Validate common invariants across all 9 events.
	sharedSysPrefix := events[0].HashIDs[:2]
	require.Len(t, sharedSysPrefix, 2)
	for idx, ev := range events {
		assert.Equal(t, tracerecorder.SchemaVersion, ev.Version)
		assert.Equal(t, rec.KeyFingerprint(), ev.KeyFingerprint)
		assert.Equal(t, tracerecorder.SessionSourceHeader, ev.SessionSource)
		assert.Equal(t, tracerecorder.EmptyCacheSaltID, ev.CacheSaltID)
		assert.Equal(t, modelMyModelTarget, ev.Model)
		assert.Equal(t, "vllm", ev.TokenizerBackend)
		assert.Equal(t, 16, ev.BlockSize)
		assert.True(t, ev.Streaming)
		assert.True(t, ev.HasSystemPrompt)
		assert.True(t, ev.HasTools)
		assert.Equal(t, ev.In/16, len(ev.HashIDs), "event %d full-block hash count mismatch", idx)
		assert.Equal(t, sharedSysPrefix, ev.HashIDs[:2], "event %d must share the 2-block system prompt prefix", idx)
		for _, hVal := range ev.HashIDs {
			assert.GreaterOrEqual(t, hVal, uint64(1))
			assert.LessOrEqual(t, hVal, tracerecorder.MaxSafeUint53)
		}
	}

	// Group events by session_id preserving arrival order.
	bySession := make(map[string][]tracerecorder.Event)
	for _, ev := range events {
		bySession[ev.SessionID] = append(bySession[ev.SessionID], ev)
	}
	require.Len(t, bySession, 3, "expected 3 distinct anonymized session_ids")

	// Session 1 assertions: Turn 0 (4 blocks) -> Subagent branch (4 blocks, LCP=2) -> Turn 1 (7 blocks, LCP(Turn0,Turn1)=4).
	s1 := bySession[events[0].SessionID]
	require.Len(t, s1, 3)
	assert.Len(t, s1[0].HashIDs, 4)
	assert.Len(t, s1[1].HashIDs, 4)
	assert.Len(t, s1[2].HashIDs, 7)
	assert.Equal(t, 2, lcpLen(s1[0].HashIDs, s1[1].HashIDs), "subagent branch must share only the 2-block system prompt with Turn 0")
	assert.Equal(t, 4, lcpLen(s1[0].HashIDs, s1[2].HashIDs), "Parent Turn 1 must resume all 4 blocks of Parent Turn 0")

	// Session 2 assertions: Turn 0 ("usage") -> Turn 1 ("events", out=16) -> Turn 2 (HTTP 200, out=0, "client-disconnect").
	s2 := bySession[events[3].SessionID]
	require.Len(t, s2, 3)
	assert.Equal(t, 16, s2[0].Out)
	assert.Equal(t, tracerecorder.OutSourceUsage, s2[0].OutSource)
	assert.Equal(t, 16, s2[1].Out)
	assert.Equal(t, tracerecorder.OutSourceEvents, s2[1].OutSource)
	assert.Equal(t, 7, lcpLen(s2[1].HashIDs, s2[2].HashIDs))
	assert.Equal(t, 200, s2[2].StatusCode)
	assert.Equal(t, 0, s2[2].Out)
	assert.Equal(t, string(fwkrh.TerminationCauseClientDisconnect), s2[2].Termination)

	// Session 3 assertions: Turn 0 -> Turn 1 -> Turn 2 (HTTP 500, termination="error").
	s3 := bySession[events[6].SessionID]
	require.Len(t, s3, 3)
	assert.Equal(t, 200, s3[0].StatusCode)
	assert.Equal(t, 200, s3[1].StatusCode)
	assert.Equal(t, 500, s3[2].StatusCode)
	assert.Equal(t, string(fwkrh.TerminationCauseError), s3[2].Termination)

	// Export sealed .jsonl.gz segment(s) when EPP_TRACE_GOLDEN_OUT_DIR is set (Gate 2 Step 2A -> Step 2B handoff).
	if goldenDir := strings.TrimSpace(os.Getenv(envTraceGoldenOutDir)); goldenDir != "" {
		exportGoldenSegments(t, gzFiles, goldenDir)
	}
}

// newDeterministicRenderServer stubs vLLM's /v1/chat/completions/render (plus warmup endpoints)
// so each message deterministically maps to fixed-width uint32 token blocks:
//   - role "system":    32 tokens (2 full 16-token blocks)
//   - role "user":      32 tokens (2 full 16-token blocks)
//   - role "assistant": 16 tokens (1 full 16-token block)
func newDeterministicRenderServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/v1/messages/render", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"token_ids":[1,2,3,4]}`))
	})
	mux.HandleFunc("/v1/responses/render", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"token_ids":[1,2,3,4]}`))
	})
	mux.HandleFunc("/v1/completions/render", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"token_ids":[1,2,3,4]}`))
	})
	mux.HandleFunc("/v1/chat/completions/render", func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var req struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		var tokenIDs []uint32
		for _, m := range req.Messages {
			var contentStr string
			if err := json.Unmarshal(m.Content, &contentStr); err != nil {
				contentStr = string(m.Content)
			}
			nTokens := 16
			switch m.Role {
			case "system", "developer", "user":
				nTokens = 32
			case "assistant":
				nTokens = 16
			}
			tokenIDs = append(tokenIDs, deterministicTokensForMessage(m.Role, contentStr, nTokens)...)
		}
		if len(tokenIDs) == 0 {
			tokenIDs = []uint32{1, 2, 3, 4}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": tokenIDs})
	})

	return httptest.NewServer(mux)
}

func deterministicTokensForMessage(role, content string, count int) []uint32 {
	out := make([]uint32, 0, count)
	counter := uint32(0)
	for len(out) < count {
		h := sha256.New()
		_, _ = h.Write([]byte(role))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(content))
		var ctrBuf [4]byte
		binary.BigEndian.PutUint32(ctrBuf[:], counter)
		_, _ = h.Write(ctrBuf[:])
		sum := h.Sum(nil)
		for off := 0; off+4 <= len(sum) && len(out) < count; off += 4 {
			tok := (binary.BigEndian.Uint32(sum[off:off+4]) % 100000) + 100
			out = append(out, tok)
		}
		counter++
	}
	return out
}

func readSealedTraceSegments(t *testing.T, dir string) ([]tracerecorder.Event, []string, string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	require.NoError(t, err)

	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl.gz") {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(files)

	var events []tracerecorder.Event
	var rawBuilder strings.Builder
	for _, path := range files {
		f, err := os.Open(path) // #nosec G304 -- test directory path
		require.NoError(t, err)
		gz, err := gzip.NewReader(f)
		require.NoError(t, err)
		rawBytes, err := io.ReadAll(gz)
		require.NoError(t, err)
		require.NoError(t, gz.Close())
		require.NoError(t, f.Close())

		rawBuilder.Write(rawBytes)
		sc := bufio.NewScanner(bytes.NewReader(rawBytes))
		sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for sc.Scan() {
			line := bytes.TrimSpace(sc.Bytes())
			if len(line) == 0 {
				continue
			}
			var ev tracerecorder.Event
			require.NoError(t, json.Unmarshal(line, &ev))
			events = append(events, ev)
		}
		require.NoError(t, sc.Err())
	}
	return events, files, rawBuilder.String()
}

func lcpLen(a, b []uint64) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func exportGoldenSegments(t *testing.T, gzFiles []string, goldenDir string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(goldenDir, 0o750))

	existing, err := os.ReadDir(goldenDir)
	require.NoError(t, err)
	for _, e := range existing {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl.gz") {
			require.NoError(t, os.Remove(filepath.Join(goldenDir, e.Name())))
		}
	}

	for _, src := range gzFiles {
		data, err := os.ReadFile(src) // #nosec G304 -- test segment source
		require.NoError(t, err)
		dst := filepath.Join(goldenDir, filepath.Base(src))
		require.NoError(t, os.WriteFile(dst, data, 0o640))
	}
}
