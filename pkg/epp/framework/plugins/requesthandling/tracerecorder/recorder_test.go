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

package tracerecorder

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionid/constants"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

const testSecret32 = "0123456789abcdef0123456789abcdef" // 32 ASCII bytes

func writeSecretFile(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "capture_secret")
	require.NoError(t, os.WriteFile(p, []byte(content), 0o600))
	return p
}

func makeTokens(start uint32, count int) []uint32 {
	out := make([]uint32, count)
	for i := range count {
		out[i] = start + uint32(i) // #nosec G115 -- bounded test token slice
	}
	return out
}

func completeRequestLifecycle(
	t *testing.T,
	rec *Recorder,
	req *scheduling.InferenceRequest,
	resp *fwkrh.Response,
) {
	t.Helper()
	ctx := context.Background()
	require.NoError(t, rec.Produce(ctx, req, nil))
	require.NoError(t, rec.PreRequest(ctx, req, nil))
	rec.ResponseBody(ctx, req, resp, nil)
}

func TestStartupSecretValidation_FailClosedAndFingerprint(t *testing.T) {
	// Uses t.Setenv, so do not mark top-level t.Parallel().
	t.Setenv(EnvTraceHMACSecret, "")

	sink := NewMemoryTraceSink(0)

	// 1. Fail-closed when no secret is configured and allowEphemeralSecretForDev is false.
	_, err := NewRecorder(Config{}, sink, time.Now)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "capture secret required")

	// Also verify PluginFactory fails closed before creating outputDir.
	nonExistentOut := filepath.Join(t.TempDir(), "should-not-exist", "traces")
	rawCfg, err := json.Marshal(map[string]any{"outputDir": nonExistentOut})
	require.NoError(t, err)
	_, err = PluginFactory("recorder", json.NewDecoder(bytes.NewReader(rawCfg)), nil)
	require.Error(t, err)
	_, statErr := os.Stat(nonExistentOut)
	assert.True(t, os.IsNotExist(statErr), "PluginFactory must fail closed before creating directories")

	// 2. Reject short secrets (< 32 bytes) from file or env, even if allowEphemeralSecretForDev=true.
	shortFile := writeSecretFile(t, "too-short-secret-16b")
	_, err = NewRecorder(Config{
		SecretFilePath:             shortFile,
		AllowEphemeralSecretForDev: true,
	}, sink, time.Now)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least 32 bytes")

	t.Setenv(EnvTraceHMACSecret, "short-env-secret")
	_, err = NewRecorder(Config{AllowEphemeralSecretForDev: true}, sink, time.Now)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least 32 bytes")
	t.Setenv(EnvTraceHMACSecret, "")

	// 3. Ephemeral secret succeeds when explicitly enabled for single-process dev/tests.
	devRec, err := NewRecorder(Config{AllowEphemeralSecretForDev: true}, sink, time.Now)
	require.NoError(t, err)
	assert.Len(t, devRec.KeyFingerprint(), 8)

	// 4. Deterministic 8-hex-char key_fingerprint matching hex(SHA256("epp-trace-fp:v1:" || secret)[:4]),
	// including K8s Secret trailing newline normalization.
	secFileWithNewline := writeSecretFile(t, testSecret32+"\n")
	recFromFile, err := NewRecorder(Config{SecretFilePath: secFileWithNewline}, sink, time.Now)
	require.NoError(t, err)

	t.Setenv(EnvTraceHMACSecret, testSecret32)
	recFromEnv, err := NewRecorder(Config{}, sink, time.Now)
	require.NoError(t, err)
	t.Setenv(EnvTraceHMACSecret, "")

	expectedSum := sha256.Sum256([]byte("epp-trace-fp:v1:" + testSecret32))
	expectedFP := hex.EncodeToString(expectedSum[:4])
	assert.Len(t, expectedFP, 8)
	assert.Equal(t, expectedFP, recFromFile.KeyFingerprint())
	assert.Equal(t, expectedFP, recFromEnv.KeyFingerprint())

	otherFile := writeSecretFile(t, "fedcba9876543210fedcba9876543210")
	recOther, err := NewRecorder(Config{SecretFilePath: otherFile}, sink, time.Now)
	require.NoError(t, err)
	assert.NotEqual(t, recFromFile.KeyFingerprint(), recOther.KeyFingerprint())
}

func TestSessionResolution_PrecedenceAndFallbackMetadata(t *testing.T) {
	t.Parallel()

	secFile := writeSecretFile(t, testSecret32)
	sink := NewMemoryTraceSink(0)
	rec, err := NewRecorder(Config{SecretFilePath: secFile}, sink, time.Now)
	require.NoError(t, err)

	defaultEndResp := &fwkrh.Response{
		EndOfStream: true,
		Headers:     map[string]string{":status": "200"},
		Usage:       fwkrequest.Usage{PromptTokens: 32, CompletionTokens: 8, TotalTokens: 40},
	}

	// Case 1: "header" wins over conversation prefix when agent-identity attribute is populated,
	// and explicit agent_id + subagent_type headers are anonymized.
	reqHeaderAttr := &scheduling.InferenceRequest{
		RequestID:   "req-1",
		TargetModel: "model-a",
		FairnessID:  "fairness-ignored",
		Headers: map[string]string{
			"x-agent-id":      "subagent-alpha",
			"x-subagent-type": "code-search",
		},
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{
					{Role: "system", Content: fwkrequest.Content{Raw: "You are helpful."}},
					{Role: "user", Content: fwkrequest.Content{Raw: "Hello"}},
				},
				Tools: []any{map[string]any{"type": "function"}},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(10, 32)}},
			},
		},
	}
	reqHeaderAttr.PutAttribute(agentidentity.AgentIdentityKey, "claude-sess-42")
	completeRequestLifecycle(t, rec, reqHeaderAttr, defaultEndResp)

	// Case 2: "::sa:<agent>:s0" child stream suffix on fairness header preserves parent session_id
	// and populates anonymized agent_id.
	reqSubagentStream := &scheduling.InferenceRequest{
		RequestID:   "req-1b",
		TargetModel: "model-a",
		FairnessID:  "claude-sess-42::sa:subagent-alpha:s0",
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{
					{Role: "user", Content: fwkrequest.Content{Raw: "Subagent task"}},
				},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(100, 32)}},
			},
		},
	}
	completeRequestLifecycle(t, rec, reqSubagentStream, defaultEndResp)

	// Case 3: session-id-producer attribute also resolves as "header".
	reqSessionProducer := &scheduling.InferenceRequest{
		RequestID:   "req-1c",
		TargetModel: "model-a",
		FairnessID:  reqcommon.DefaultFairnessID,
		Body: &fwkrequest.InferenceRequestBody{
			Messages: &fwkrequest.MessagesRequest{
				System: fwkrequest.AnthropicContent{Raw: "Anthropic system prompt"},
				Messages: []fwkrequest.AnthropicMessage{
					{Role: "user", Content: fwkrequest.AnthropicContent{Raw: "Hi"}},
				},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(200, 32)}},
			},
		},
	}
	reqSessionProducer.PutAttribute(
		attrsession.SessionIDDataKey.WithNonEmptyProducerName(sessionidconstants.SessionIDProducerType),
		attrsession.SessionID("cookie-sess-99"),
	)
	completeRequestLifecycle(t, rec, reqSessionProducer, defaultEndResp)

	// Case 4: Headerless multi-turn chat (with DefaultFairnessID set by Director!) falls through to
	// "prefix_fallback", yielding identical session_id across Turn 0 (msg_count=2) and Turn 1 (msg_count=4).
	turn0Fallback := &scheduling.InferenceRequest{
		RequestID:   "req-2-t0",
		TargetModel: "model-a",
		FairnessID:  reqcommon.DefaultFairnessID, // must NOT count as explicit header!
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{
					{Role: "system", Content: fwkrequest.Content{Raw: "Sys prompt"}},
					{Role: "user", Content: fwkrequest.Content{Raw: "First user turn"}},
				},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(300, 32)}},
			},
		},
	}
	turn1Fallback := &scheduling.InferenceRequest{
		RequestID:   "req-2-t1",
		TargetModel: "model-a",
		FairnessID:  reqcommon.DefaultFairnessID,
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{
					{Role: "system", Content: fwkrequest.Content{Raw: "Sys prompt"}},
					{Role: "user", Content: fwkrequest.Content{Raw: "First user turn"}},
					{Role: "assistant", Content: fwkrequest.Content{Raw: "First reply"}},
					{Role: "user", Content: fwkrequest.Content{Raw: "Second user turn"}},
				},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(300, 64)}},
			},
		},
	}
	completeRequestLifecycle(t, rec, turn0Fallback, defaultEndResp)
	completeRequestLifecycle(t, rec, turn1Fallback, defaultEndResp)

	// Case 5: Single-shot /v1/completions without headers or turn boundary falls back to "request_id".
	reqCompletions := &scheduling.InferenceRequest{
		RequestID:   "req-3-single",
		TargetModel: "model-a",
		FairnessID:  reqcommon.DefaultFairnessID,
		Body: &fwkrequest.InferenceRequestBody{
			Completions: &fwkrequest.CompletionsRequest{
				Prompt: fwkrequest.Prompt{Raw: "Once upon a time"},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(500, 16)}},
			},
		},
	}
	completeRequestLifecycle(t, rec, reqCompletions, defaultEndResp)

	evs := sink.Events()
	require.Len(t, evs, 6)

	// Verify Case 1 & 2 share session_id and agent_id via ::sa: parsing.
	assert.Equal(t, SessionSourceHeader, evs[0].SessionSource)
	assert.True(t, evs[0].HasSystemPrompt)
	assert.True(t, evs[0].HasTools)
	assert.Equal(t, 2, evs[0].MsgCount)
	assert.NotEmpty(t, evs[0].AgentID)
	assert.NotEmpty(t, evs[0].SubagentType)

	assert.Equal(t, SessionSourceHeader, evs[1].SessionSource)
	assert.Equal(t, evs[0].SessionID, evs[1].SessionID, "::sa: child stream must resolve to parent session_id")
	assert.Equal(t, evs[0].AgentID, evs[1].AgentID, "::sa: agent_id must match explicit agent_id")
	assert.False(t, evs[1].HasSystemPrompt)
	assert.False(t, evs[1].HasTools)
	assert.Equal(t, 1, evs[1].MsgCount)

	// Verify Case 3 (Anthropic system prompt + session-id-producer).
	assert.Equal(t, SessionSourceHeader, evs[2].SessionSource)
	assert.True(t, evs[2].HasSystemPrompt)
	assert.False(t, evs[2].HasTools)
	assert.Equal(t, 1, evs[2].MsgCount)

	// Verify Case 4 ("prefix_fallback" stable session_id + msg_count growth).
	assert.Equal(t, SessionSourcePrefixFallback, evs[3].SessionSource)
	assert.Equal(t, SessionSourcePrefixFallback, evs[4].SessionSource)
	assert.Equal(t, evs[3].SessionID, evs[4].SessionID)
	assert.Equal(t, 2, evs[3].MsgCount)
	assert.Equal(t, 4, evs[4].MsgCount)

	// Verify Case 5 ("request_id" fallback).
	assert.Equal(t, SessionSourceRequestID, evs[5].SessionSource)
	assert.Equal(t, 1, evs[5].MsgCount)
	assert.False(t, evs[5].HasSystemPrompt)
}

func TestAnonymization_ZeroCanaryLeakAnd53BitBounds(t *testing.T) {
	t.Parallel()

	secFile := writeSecretFile(t, testSecret32)
	baseDir := t.TempDir()
	outDir := filepath.Join(baseDir, "out")
	stageDir := filepath.Join(baseDir, "stage")

	gzSink, err := NewSealedGzipSink(SealedGzipSinkConfig{
		OutputDir:       outDir,
		StagingDir:      stageDir,
		PodName:         "epp-sec",
		MaxSegmentBytes: 1024 * 1024,
		FlushInterval:   time.Hour,
		BufferSize:      64,
	})
	require.NoError(t, err)

	rec, err := NewRecorder(Config{
		SecretFilePath:  secFile,
		BlockSizeTokens: 16,
	}, gzSink, time.Now)
	require.NoError(t, err)

	canaries := []string{
		"CANARY_SYSTEM_PROMPT_SECRET_99182",
		"CANARY_USER_PII_SSN_000_11_2222",
		"CANARY_TOOL_SCHEMA_INTERNAL_FN_771",
		"CANARY_CACHE_SALT_TENANT_ALPHA_552",
		"CANARY_SESSION_HEADER_RAW_88412",
		"CANARY_AGENT_HEADER_RAW_33109",
		"CANARY_SUBAGENT_TYPE_RAW_66501",
	}

	tokens := makeTokens(42, 64) // 4 full blocks of 16
	req := &scheduling.InferenceRequest{
		RequestID:   "req-canary",
		TargetModel: "Qwen/Qwen2.5-0.5B-Instruct",
		Headers: map[string]string{
			metadata.FlowFairnessIDKey: canaries[4],
			"x-agent-id":               canaries[5],
			"x-subagent-type":          canaries[6],
		},
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				CacheSalt: canaries[3],
				Messages: []fwkrequest.Message{
					{Role: "system", Content: fwkrequest.Content{Raw: canaries[0]}},
					{Role: "user", Content: fwkrequest.Content{Raw: canaries[1]}},
				},
				Tools: []any{map[string]any{"name": canaries[2]}},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				CacheSalt: canaries[3],
				Prompts:   []fwkrequest.PromptTokens{{TokenIDs: tokens}},
			},
		},
	}

	completeRequestLifecycle(t, rec, req, &fwkrh.Response{
		EndOfStream: true,
		Headers:     map[string]string{":status": "200"},
		Usage:       fwkrequest.Usage{PromptTokens: 64, CompletionTokens: 16, TotalTokens: 80},
	})
	require.NoError(t, rec.Close(context.Background()))

	// Compute raw unkeyed digests (cbor-fnv, xxh64, and prefixhash xxhash64) to verify NONE appear in output.
	var forbiddenHashStrings []string
	for _, alg := range []string{kvblock.HashAlgorithmCBORFNV, kvblock.HashAlgorithmXXH64} {
		tp, tpErr := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
			BlockSizeTokens: 16,
			HashAlgorithm:   alg,
		})
		require.NoError(t, tpErr)
		extra := foldFirstBlockCacheSalt(nil, canaries[3], 4)
		keys, kErr := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, req.TargetModel, extra)
		require.NoError(t, kErr)
		for _, k := range keys {
			forbiddenHashStrings = append(forbiddenHashStrings,
				strconv.FormatUint(uint64(k), 10),
				strconv.FormatUint(uint64(k), 16),
			)
		}
	}
	phBlocks := prefixhash.GetBlockHashes(context.Background(), req, 16, math.MaxInt)
	for _, list := range phBlocks {
		for _, h := range list {
			forbiddenHashStrings = append(forbiddenHashStrings,
				strconv.FormatUint(uint64(h), 10),
				strconv.FormatUint(uint64(h), 16),
			)
		}
	}

	evs, files := readAllGzipEvents(t, outDir)
	require.Len(t, files, 1)
	require.Len(t, evs, 1)

	rawCompressed, err := os.ReadFile(files[0]) // #nosec G304 -- test output file
	require.NoError(t, err)
	gzReader, err := gzip.NewReader(bytes.NewReader(rawCompressed))
	require.NoError(t, err)
	uncompressedBytes, err := io.ReadAll(gzReader)
	require.NoError(t, err)
	uncompressedText := string(uncompressedBytes)

	for _, c := range canaries {
		assert.NotContains(t, uncompressedText, c, "canary string leaked into .jsonl.gz")
		assert.False(t, bytes.Contains(rawCompressed, []byte(c)), "canary string leaked into compressed bytes")
	}
	for _, rawHashStr := range forbiddenHashStrings {
		assert.NotContains(t, uncompressedText, rawHashStr, "unkeyed raw block hash leaked into .jsonl.gz")
	}

	// Verify exact JSON schema keys and 53-bit integer bounds.
	var rawMap map[string]any
	require.NoError(t, json.Unmarshal(bytes.TrimSpace(uncompressedBytes), &rawMap))
	expectedKeys := []string{
		"v", "key_fingerprint", "capture_run_id", "session_id", "session_source", "cache_salt_id",
		"agent_id", "subagent_type", "t_recv_unix", "ttft_s", "api_time_s",
		"streaming", "status_code", "termination", "model", "tokenizer_backend",
		"block_size", "in", "usage_in", "out", "out_source", "msg_count",
		"has_system_prompt", "has_tools", "hash_ids",
	}
	assert.Len(t, rawMap, len(expectedKeys))
	for _, k := range expectedKeys {
		assert.Contains(t, rawMap, k, "missing required JSON schema key %q", k)
	}

	got := evs[0]
	require.Len(t, got.HashIDs, 4)
	for i, h := range got.HashIDs {
		assert.GreaterOrEqual(t, h, uint64(1), "hash_ids[%d] must be >= 1", i)
		assert.LessOrEqual(t, h, MaxSafeUint53, "hash_ids[%d] must be <= 2^53-1", i)
	}
	assert.True(t, strings.HasPrefix(got.CacheSaltID, "c_"))
	assert.NotEqual(t, EmptyCacheSaltID, got.CacheSaltID)
	assert.True(t, strings.HasPrefix(got.SessionID, "s_"))
	assert.True(t, strings.HasPrefix(got.AgentID, "a_"))
	assert.True(t, strings.HasPrefix(got.SubagentType, "st_"))
}

func TestCohortSampling_CrossReplicaDeterminism(t *testing.T) {
	t.Parallel()

	secFile := writeSecretFile(t, testSecret32)
	rate10Pct := 1000 // 10% = 1000 bps

	sinkA := NewMemoryTraceSink(0)
	sinkB := NewMemoryTraceSink(0)
	replicaA, err := NewRecorder(Config{SecretFilePath: secFile, SampleRateBPS: &rate10Pct}, sinkA, time.Now)
	require.NoError(t, err)
	replicaB, err := NewRecorder(Config{SecretFilePath: secFile, SampleRateBPS: &rate10Pct}, sinkB, time.Now)
	require.NoError(t, err)

	const totalSessions = 500
	const turnsPerSession = 4 // 3 parent turns + 1 ::sa: subagent turn

	for s := range totalSessions {
		rawSess := fmt.Sprintf("user-session-%04d", s)
		for turn := range turnsPerSession {
			headerVal := rawSess
			if turn == 2 {
				headerVal = rawSess + "::sa:helper:s0"
			}
			req := &scheduling.InferenceRequest{
				RequestID:   fmt.Sprintf("req-%d-%d", s, turn),
				TargetModel: "model-x",
				Headers:     map[string]string{metadata.FlowFairnessIDKey: headerVal},
				Body: &fwkrequest.InferenceRequestBody{
					ChatCompletions: &fwkrequest.ChatCompletionsRequest{
						Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Raw: "turn"}}},
					},
					TokenizedRequest: &fwkrequest.TokenizedRequest{
						Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, (turn+1)*16)}},
					},
				},
			}
			resp := &fwkrh.Response{
				EndOfStream: true,
				Headers:     map[string]string{":status": "200"},
				Usage:       fwkrequest.Usage{PromptTokens: (turn + 1) * 16, CompletionTokens: 8, TotalTokens: (turn+1)*16 + 8},
			}
			// Alternate turns across active-active replicas A and B!
			targetReplica := replicaA
			if (s+turn)%2 == 1 {
				targetReplica = replicaB
			}
			completeRequestLifecycle(t, targetReplica, req, resp)
		}
	}

	countsBySession := make(map[string]int)
	for _, ev := range append(sinkA.Events(), sinkB.Events()...) {
		countsBySession[ev.SessionID]++
	}

	// Every sampled session MUST have all 4 turns captured (100% all-or-nothing cohort determinism).
	assert.GreaterOrEqual(t, len(countsBySession), 25, "10% of 500 sessions should capture >= 25 sessions")
	assert.LessOrEqual(t, len(countsBySession), 85, "10% of 500 sessions should capture <= 85 sessions")
	for sid, count := range countsBySession {
		assert.Equal(t, turnsPerSession, count, "session %s had partial turn capture across replicas", sid)
	}

	// Explicit 0 bps captures zero events.
	zeroRate := 0
	sinkZero := NewMemoryTraceSink(0)
	recZero, err := NewRecorder(Config{SecretFilePath: secFile, SampleRateBPS: &zeroRate}, sinkZero, time.Now)
	require.NoError(t, err)
	completeRequestLifecycle(t, recZero, &scheduling.InferenceRequest{
		RequestID: "req-zero",
		Headers:   map[string]string{metadata.FlowFairnessIDKey: "sess-any"},
		Body:      &fwkrequest.InferenceRequestBody{},
	}, &fwkrh.Response{EndOfStream: true})
	assert.Empty(t, sinkZero.Events())
}

func TestTokenizerPolicyAndFullBlockTruncation(t *testing.T) {
	t.Parallel()

	secFile := writeSecretFile(t, testSecret32)
	ctx := context.Background()

	// 1. Startup DAG validation rejects token-producer/estimate when allowEstimateTokenizer=false.
	handle := plugin.NewEppHandle(ctx, nil)
	estProducerPlg, fErr := tokenproducer.PluginFactory(tokenproducer.PluginType, nil, handle)
	require.NoError(t, fErr)

	strictSink := NewMemoryTraceSink(0)
	strictRec, err := NewRecorder(Config{
		SecretFilePath:           secFile,
		AllowEstimateTokenizer:   false,
		TokenizerBackendOverride: tokenproducer.BackendEstimate,
	}, strictSink, time.Now)
	require.NoError(t, err)

	_, dagErr := datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{estProducerPlg, strictRec})
	require.Error(t, dagErr, "DAG validation must reject estimate tokenizer when allowEstimateTokenizer=false")

	// Runtime Produce also rejects non-vllm backend when allowEstimateTokenizer=false.
	req70 := &scheduling.InferenceRequest{
		RequestID:   "req-est-70",
		TargetModel: "model-est",
		Headers:     map[string]string{metadata.FlowFairnessIDKey: "sess-est"},
		Body: &fwkrequest.InferenceRequestBody{
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Raw: "hello"}}},
			},
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, 70)}},
			},
		},
	}
	err = strictRec.Produce(ctx, req70, nil)
	require.Error(t, err)
	assert.Empty(t, strictSink.Events())

	// 2. With allowEstimateTokenizer=true, DAG validation succeeds AND in_tokens=70, block_size=16
	// emits exactly 4 full-block hashes (slicing off prefixhash's 5th partial block!).
	estSink := NewMemoryTraceSink(0)
	estRec, err := NewRecorder(Config{
		SecretFilePath:           secFile,
		BlockSizeTokens:          16,
		AllowEstimateTokenizer:   true,
		TokenizerBackendOverride: tokenproducer.BackendEstimate,
	}, estSink, time.Now)
	require.NoError(t, err)

	_, dagErr = datalayer.ValidateAndOrderDataDependencies([]plugin.Plugin{estProducerPlg, estRec})
	require.NoError(t, dagErr, "DAG validation must accept estimate tokenizer when allowEstimateTokenizer=true")

	// Verify prefixhash itself returns 5 blocks (4 full + 1 partial) for 70 tokens,
	// while Recorder truncates to 70 // 16 = 4 full blocks.
	rawPH := prefixhash.GetBlockHashes(ctx, req70, 16, math.MaxInt)
	require.Len(t, rawPH, 1)
	require.Len(t, rawPH[0], 5, "prefixhash produces 4 full + 1 partial block for 70 tokens")

	completeRequestLifecycle(t, estRec, req70, &fwkrh.Response{
		EndOfStream:    true,
		Headers:        map[string]string{":status": "200"},
		StreamedEvents: 10,
	})

	evs := estSink.Events()
	require.Len(t, evs, 1)
	assert.Equal(t, tokenproducer.BackendEstimate, evs[0].TokenizerBackend)
	assert.Equal(t, 70, evs[0].In)
	require.Len(t, evs[0].HashIDs, 4, "70 tokens at block_size=16 must emit exactly 4 full-block hashes")

	// Also verify response.Usage.PromptTokens override in estimate mode (e.g. PromptTokens=48 -> 3 hashes)
	// and < 16 tokens -> non-nil empty [] slice.
	req70WithUsage := &scheduling.InferenceRequest{
		RequestID:   "req-est-usage",
		TargetModel: "model-est",
		Headers:     map[string]string{metadata.FlowFairnessIDKey: "sess-est"},
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, 70)}},
			},
		},
	}
	completeRequestLifecycle(t, estRec, req70WithUsage, &fwkrh.Response{
		EndOfStream: true,
		Headers:     map[string]string{":status": "200"},
		Usage:       fwkrequest.Usage{PromptTokens: 48, CompletionTokens: 5, TotalTokens: 53},
	})
	reqShort := &scheduling.InferenceRequest{
		RequestID:   "req-short",
		TargetModel: "model-est",
		Headers:     map[string]string{metadata.FlowFairnessIDKey: "sess-est"},
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, 9)}},
			},
		},
	}
	completeRequestLifecycle(t, estRec, reqShort, &fwkrh.Response{
		EndOfStream: true,
		Headers:     map[string]string{":status": "200"},
	})

	evs = estSink.Events()
	require.Len(t, evs, 3)
	assert.Equal(t, 48, evs[1].In)
	assert.Len(t, evs[1].HashIDs, 3)
	require.NotNil(t, evs[2].HashIDs)
	assert.Empty(t, evs[2].HashIDs)
}

func TestResponseLifecycle_ReadOnlyOutFallbackAndNoStateLeaks(t *testing.T) {
	t.Parallel()

	secFile := writeSecretFile(t, testSecret32)
	clk := newFakeClock(time.Unix(1762121500, 0))
	sink := NewMemoryTraceSink(0)
	rec, err := NewRecorder(Config{
		SecretFilePath:           secFile,
		InjectStreamIncludeUsage: false, // read-only default
	}, sink, clk.Now)
	require.NoError(t, err)

	ctx := context.Background()
	makeReq := func(id string) *scheduling.InferenceRequest {
		return &scheduling.InferenceRequest{
			RequestID:   id,
			TargetModel: "Qwen/Qwen2.5-0.5B-Instruct",
			Headers:     map[string]string{metadata.FlowFairnessIDKey: "sess-lifecycle"},
			Body: &fwkrequest.InferenceRequestBody{
				Stream:  true,
				Payload: fwkrequest.PayloadMap{"model": "Qwen/Qwen2.5-0.5B-Instruct", "stream": true},
				ChatCompletions: &fwkrequest.ChatCompletionsRequest{
					Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Raw: "hi"}}},
				},
				TokenizedRequest: &fwkrequest.TokenizedRequest{
					Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, 32)}},
				},
			},
		}
	}

	// 1. Usage.CompletionTokens ("usage") + read-only PreRequest (Body.Mutated == false).
	reqUsage := makeReq("req-usage")
	require.NoError(t, rec.Produce(ctx, reqUsage, nil))
	require.True(t, rec.HasPendingState("req-usage"))
	require.NoError(t, rec.PreRequest(ctx, reqUsage, nil))
	assert.False(t, reqUsage.Body.Mutated, "PreRequest must be read-only by default")

	// Intermediate streaming chunk (EndOfStream: false) must NOT finalize or delete state.
	rec.ResponseBody(ctx, reqUsage, &fwkrh.Response{
		EndOfStream: false,
		Streaming:   true,
	}, nil)
	require.True(t, rec.HasPendingState("req-usage"))

	tRecv := clk.Now()
	tFirst := tRecv.Add(45 * time.Millisecond)
	tDone := tRecv.Add(320 * time.Millisecond)
	rec.ResponseBody(ctx, reqUsage, &fwkrh.Response{
		EndOfStream:         true,
		Streaming:           true,
		Headers:             map[string]string{":status": "200"},
		ReceivedTimestamp:   tRecv,
		FirstTokenTimestamp: tFirst,
		CompleteTimestamp:   tDone,
		StreamedEvents:      25,
		Usage:               fwkrequest.Usage{PromptTokens: 32, CompletionTokens: 21, TotalTokens: 53},
	}, nil)
	assert.False(t, rec.HasPendingState("req-usage"), "PluginState must be deleted on EndOfStream")

	// 2. StreamedEvents fallback ("events") when Usage is absent.
	reqEvents := makeReq("req-events")
	completeRequestLifecycle(t, rec, reqEvents, &fwkrh.Response{
		EndOfStream:    true,
		Streaming:      true,
		Headers:        map[string]string{":status": "200"},
		StreamedEvents: 14,
	})
	assert.False(t, rec.HasPendingState("req-events"))

	// 3. Client disconnect mid-prefill (HTTP 200, out=0, termination="client-disconnect").
	reqDisc := makeReq("req-disconnect")
	completeRequestLifecycle(t, rec, reqDisc, &fwkrh.Response{
		EndOfStream:      true,
		Streaming:        true,
		Headers:          map[string]string{":status": "200"},
		TerminationCause: fwkrh.TerminationCauseClientDisconnect,
		StreamedEvents:   0,
	})
	assert.False(t, rec.HasPendingState("req-disconnect"))

	// 4. Upstream HTTP 503 error (status_code >= 400).
	reqErr := makeReq("req-503")
	completeRequestLifecycle(t, rec, reqErr, &fwkrh.Response{
		EndOfStream: true,
		Headers:     map[string]string{":status": "503"},
	})
	assert.False(t, rec.HasPendingState("req-503"))

	// 5. RequestAbandoned deletes PluginState immediately without emitting an event.
	reqAband := makeReq("req-abandoned")
	require.NoError(t, rec.Produce(ctx, reqAband, nil))
	require.True(t, rec.HasPendingState("req-abandoned"))
	rec.RequestAbandoned(ctx, reqAband)
	assert.False(t, rec.HasPendingState("req-abandoned"), "PluginState must be deleted on RequestAbandoned")

	evs := sink.Events()
	require.Len(t, evs, 4)

	assert.Equal(t, 21, evs[0].Out)
	assert.Equal(t, OutSourceUsage, evs[0].OutSource)
	assert.InDelta(t, 0.045, evs[0].TTFTS, 1e-6)
	assert.InDelta(t, 0.320, evs[0].APITimeS, 1e-6)
	assert.Equal(t, TerminationNatural, evs[0].Termination)

	assert.Equal(t, 14, evs[1].Out)
	assert.Equal(t, OutSourceEvents, evs[1].OutSource)

	assert.Equal(t, 0, evs[2].Out)
	assert.Equal(t, OutSourceEvents, evs[2].OutSource)
	assert.Equal(t, 200, evs[2].StatusCode)
	assert.Equal(t, string(fwkrh.TerminationCauseClientDisconnect), evs[2].Termination)

	assert.Equal(t, 503, evs[3].StatusCode)
	assert.Equal(t, string(fwkrh.TerminationCauseError), evs[3].Termination)

	// 6. Opt-in injectStreamIncludeUsage=true mutates PayloadMap cleanly.
	recMut, err := NewRecorder(Config{
		SecretFilePath:           secFile,
		InjectStreamIncludeUsage: true,
	}, NewMemoryTraceSink(0), clk.Now)
	require.NoError(t, err)
	reqOptIn := makeReq("req-opt-in")
	require.NoError(t, recMut.Produce(ctx, reqOptIn, nil))
	require.NoError(t, recMut.PreRequest(ctx, reqOptIn, nil))
	assert.True(t, reqOptIn.Body.Mutated)
	pm, ok := reqOptIn.Body.Payload.AsMap()
	require.True(t, ok)
	assert.Equal(t, map[string]any{"include_usage": true}, pm[reqcommon.FieldStreamOptions])
}

func TestParentSessionHeaders_StreamCoalescing_AndLifecycleHardening(t *testing.T) {
	t.Parallel()

	secFile := writeSecretFile(t, testSecret32)
	sink := NewMemoryTraceSink(0)
	rec, err := NewRecorder(Config{SecretFilePath: secFile}, sink, time.Now)
	require.NoError(t, err)
	assert.Len(t, rec.CaptureRunID(), 32)

	maxOut32 := int64(32)
	rootReq := &scheduling.InferenceRequest{
		RequestID:   "req-root",
		TargetModel: "model-a",
		Headers: map[string]string{
			HeaderDynamoSessionID: "dyn-root-session",
		},
		Body: &fwkrequest.InferenceRequestBody{
			MaxOutputTokens: &maxOut32,
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, 32)}},
			},
		},
	}
	completeRequestLifecycle(t, rec, rootReq, &fwkrh.Response{
		EndOfStream: true,
		Headers:     map[string]string{":status": "200"},
		Usage: fwkrequest.Usage{
			PromptTokens:       32,
			CompletionTokens:   8,
			TotalTokens:        40,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 16},
		},
	})

	// Explicit parent session header + parallel streams (:s0 and :s1) coalescing.
	for i, streamSuffix := range []string{"worker-7:s0", "worker-7:s1"} {
		childReq := &scheduling.InferenceRequest{
			RequestID:   fmt.Sprintf("req-child-%d", i),
			TargetModel: "model-a",
			Headers: map[string]string{
				HeaderDynamoSessionID:       fmt.Sprintf("dyn-child-session-%d", i),
				HeaderDynamoParentSessionID: "dyn-root-session",
				HeaderClaudeCodeAgentID:     streamSuffix,
			},
			Body: &fwkrequest.InferenceRequestBody{
				TokenizedRequest: &fwkrequest.TokenizedRequest{
					Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(100, 32)}},
				},
			},
		}
		completeRequestLifecycle(t, rec, childReq, &fwkrh.Response{
			EndOfStream: true,
			Headers:     map[string]string{":status": "200"},
			Usage:       fwkrequest.Usage{PromptTokens: 32, CompletionTokens: 6, TotalTokens: 38},
		})
	}

	evs := sink.Events()
	require.Len(t, evs, 3)

	// Root event has empty ParentSessionID, CachedTokens=16, UsageIn=32, MaxOutputTokens=32.
	assert.Empty(t, evs[0].ParentSessionID)
	assert.Equal(t, 32, evs[0].UsageIn)
	assert.Equal(t, 16, evs[0].CachedTokens)
	require.NotNil(t, evs[0].MaxOutputTokens)
	assert.Equal(t, int64(32), *evs[0].MaxOutputTokens)

	// Child events have ParentSessionID joining directly to root SessionID, and identical AgentID across :s0 and :s1.
	assert.Equal(t, evs[0].SessionID, evs[1].ParentSessionID, "child parent_session_id must join directly to parent session_id")
	assert.Equal(t, evs[0].SessionID, evs[2].ParentSessionID)
	assert.NotEmpty(t, evs[1].AgentID)
	assert.Equal(t, evs[1].AgentID, evs[2].AgentID, "parallel subagent streams :s0 and :s1 must coalesce to identical agent_id")

	// Verify BindLiveness + OnEvicted holds in-flight state across staleness sweeps while liveCtx is active,
	// and evicts + decrements inFlight once liveCtx cancels.
	rec.pluginState = plugin.NewPluginState(
		context.Background(),
		plugin.WithStalenessThreshold(10*time.Millisecond),
		plugin.WithCleanupInterval(5*time.Millisecond),
	)
	liveCtx, cancel := context.WithCancel(context.Background())
	inFlightReq := &scheduling.InferenceRequest{
		RequestID:   "req-inflight-cancel",
		TargetModel: "model-a",
		Headers:     map[string]string{HeaderCorrelationID: "corr-99"},
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: &fwkrequest.TokenizedRequest{
				Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeTokens(1, 16)}},
			},
		},
	}
	require.NoError(t, rec.Produce(liveCtx, inFlightReq, nil))
	assert.Equal(t, int64(1), rec.InFlightRequests())

	// Even after >2x staleness threshold, BindLiveness keeps the entry alive while liveCtx is not done.
	time.Sleep(25 * time.Millisecond)
	assert.Equal(t, int64(1), rec.InFlightRequests())
	assert.True(t, rec.HasPendingState("req-inflight-cancel"))

	rawDump, err := rec.DumpState()
	require.NoError(t, err)
	var dumped map[string]any
	require.NoError(t, json.Unmarshal(rawDump, &dumped))
	assert.Equal(t, rec.CaptureRunID(), dumped["captureRunId"])
	assert.Equal(t, float64(1), dumped["inFlight"])

	cancel()
	require.Eventually(t, func() bool {
		return rec.InFlightRequests() == 0 && !rec.HasPendingState("req-inflight-cancel")
	}, 2*time.Second, 5*time.Millisecond, "BindLiveness cancellation must allow janitor to evict pendingRecord and decrement inFlight")
}

func TestDownsampleBlockHashes_MultiType(t *testing.T) {
	t.Parallel()

	u64s := []uint64{10, 20, 30, 40, 50, 60, 70, 80, 90}
	dsU64, err := DownsampleBlockHashes(u64s, 16, 64)
	require.NoError(t, err)
	assert.Equal(t, []uint64{40, 80}, dsU64)

	i64s := []int64{11, 22, 33, 44, 55, 66}
	dsI64, err := DownsampleBlockHashes(i64s, 16, 32)
	require.NoError(t, err)
	assert.Equal(t, []int64{22, 44, 66}, dsI64)

	strs := []string{"h0", "h1", "h2", "h3", "h4"}
	dsStr, err := DownsampleBlockHashes(strs, 16, 32)
	require.NoError(t, err)
	assert.Equal(t, []string{"h1", "h3"}, dsStr)

	_, err = DownsampleBlockHashes(strs, 16, 24)
	require.Error(t, err)
}
