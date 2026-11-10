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
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAssembleWekaTraces_TransitiveSubagents_LCPChaining_AndDualSchema(t *testing.T) {
	t.Parallel()

	maxOut32 := int64(32)
	maxOut512 := int64(512)

	// 1. Transitive subagent tree (S2 -> S1 -> Root) + cycle guard + classifier call filtering + out==0->1 clamping.
	records := []CapturedRecord{
		{
			RequestID:         "r-root-0",
			SessionID:         "s_root",
			ArrivalUnixNano:   1_000_000_000,
			APITimeSeconds:    0.200,
			StreamType:        "s",
			FirstTokenSeconds: 0.025,
			Model:             "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:         16,
			TokenizedIn:       64,
			OutputTokens:      16,
			MaxOutputTokens:   &maxOut512,
			HasTools:          true,
			BlockHashes:       []string{"sys0", "sys1", "u0", "u1"},
		},
		// Classifier call (max_output_tokens <= 64 && !has_tools): must be dropped when DropClassifierCalls=true.
		{
			RequestID:       "r-classifier",
			SessionID:       "s_root",
			ArrivalUnixNano: 1_100_000_000,
			APITimeSeconds:  0.050,
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     32,
			OutputTokens:    4,
			MaxOutputTokens: &maxOut32,
			HasTools:        false,
			BlockHashes:     []string{"sys0", "sys1"},
		},
		// Level-1 subagent S1 -> s_root
		{
			RequestID:       "r-sub-s1",
			SessionID:       "s_child_1",
			ParentSessionID: "s_root",
			SubagentID:      "a_sub1",
			ArrivalUnixNano: 1_300_000_000,
			APITimeSeconds:  0.150,
			StreamType:      "s",
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     48,
			OutputTokens:    0, // must clamp to 1
			BlockHashes:     []string{"sys0", "sys1", "sub1"},
		},
		// Level-2 transitive subagent S2 -> S1 -> s_root
		{
			RequestID:       "r-sub-s2",
			SessionID:       "s_child_2",
			ParentSessionID: "s_child_1",
			SubagentID:      "a_sub2",
			ArrivalUnixNano: 1_500_000_000,
			APITimeSeconds:  0.100,
			StreamType:      "s",
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     48,
			OutputTokens:    10,
			BlockHashes:     []string{"sys0", "sys1", "sub2"},
		},
		// Root Turn 1
		{
			RequestID:       "r-root-1",
			SessionID:       "s_root",
			ArrivalUnixNano: 2_000_000_000,
			APITimeSeconds:  0.250,
			StreamType:      "s",
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     96,
			OutputTokens:    20,
			BlockHashes:     []string{"sys0", "sys1", "u0", "u1", "a0", "u2"},
		},
		// Sessionless LCP chain (2 turns sharing >=90% prefix)
		{
			RequestID:       "r-lcp-0",
			ArrivalUnixNano: 3_000_000_000,
			APITimeSeconds:  0.100,
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     64,
			OutputTokens:    12,
			BlockHashes:     []string{"sys0", "sys1", "lcp0", "lcp1"},
		},
		{
			RequestID:       "r-lcp-1",
			ArrivalUnixNano: 3_500_000_000,
			APITimeSeconds:  0.120,
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     96,
			OutputTokens:    14,
			BlockHashes:     []string{"sys0", "sys1", "lcp0", "lcp1", "lcp2", "lcp3"},
		},
		// 3rd session with untagged synchronous subagent branch (Turn 0 -> diverging branch -> Turn 1 resuming Turn 0)
		{
			RequestID:       "r-s3-t0",
			SessionID:       "s_untagged",
			ArrivalUnixNano: 4_000_000_000,
			APITimeSeconds:  0.100,
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     64,
			OutputTokens:    8,
			BlockHashes:     []string{"sys0", "sys1", "p0", "p1"},
		},
		{
			RequestID:       "r-s3-sub",
			SessionID:       "s_untagged",
			ArrivalUnixNano: 4_200_000_000,
			APITimeSeconds:  0.100,
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     64,
			OutputTokens:    6,
			BlockHashes:     []string{"sys0", "sys1", "branch0", "branch1"},
		},
		{
			RequestID:       "r-s3-t1",
			SessionID:       "s_untagged",
			ArrivalUnixNano: 4_500_000_000,
			APITimeSeconds:  0.100,
			Model:           "Qwen/Qwen2.5-0.5B-Instruct",
			BlockSize:       16,
			TokenizedIn:     96,
			OutputTokens:    9,
			BlockHashes:     []string{"sys0", "sys1", "p0", "p1", "p2", "p3"},
		},
	}

	traces, err := AssembleWekaTraces(records, AssembleOptions{
		TargetBlockSize:     32, // downsample 16 -> 32 (k=2)
		HashIDScope:         HashIDScopeLocal,
		LCPOverlapThreshold: 0.90,
		DropClassifierCalls: true,
		MinSystemCohort:     3,
	})
	require.NoError(t, err)
	require.Len(t, traces, 3)

	// Cohort of 3 traces shares 2 blocks of 16 tokens (= 1 downsampled block of 32 tokens: "sys1").
	for _, tr := range traces {
		assert.Equal(t, 32, tr.BlockSize)
		assert.Equal(t, 32, tr.SystemTokens, "cohort system_tokens should infer 1 block * 32 tokens = 32")
	}

	// Trace 0: 2 parent turns + 2 transitive subagents (classifier call dropped).
	tr0 := traces[0]
	require.Len(t, tr0.Requests, 4)
	require.NotNil(t, tr0.Requests[0].Turn)
	require.NotNil(t, tr0.Requests[1].Subagent)
	require.NotNil(t, tr0.Requests[2].Subagent)
	require.NotNil(t, tr0.Requests[3].Turn)
	assert.Equal(t, 2, tr0.Totals.SubagentCount)
	assert.Equal(t, 1, tr0.Requests[1].Subagent.Requests[0].Out, "out == 0 must clamp to 1")
	assert.Equal(t, "n", tr0.Requests[1].Subagent.Requests[0].Type, "inner subagent request type must be 'n'")
	assert.Equal(t, "completed", tr0.Requests[1].Subagent.Status)
	require.NotNil(t, tr0.Requests[1].Subagent.Requests[0].ThinkTime)

	// Trace 1: sessionless LCP-chained conversation (2 turns).
	tr1 := traces[1]
	require.Len(t, tr1.Requests, 2)
	assert.Equal(t, []int{1, 2}, tr1.Requests[0].Turn.HashIDs)
	assert.Equal(t, []int{1, 2, 3}, tr1.Requests[1].Turn.HashIDs)

	// Trace 2: untagged synchronous subagent lookahead detected 1 subagent + 2 parent turns.
	tr2 := traces[2]
	require.Len(t, tr2.Requests, 3)
	require.NotNil(t, tr2.Requests[0].Turn)
	require.NotNil(t, tr2.Requests[1].Subagent)
	require.NotNil(t, tr2.Requests[2].Turn)
	assert.Equal(t, 1, tr2.Totals.SubagentCount)
}

func TestLoadCapturedRecords_BothEventGzipAndCapturedRecordJSONL(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	gzPath := filepath.Join(tmpDir, "seg-001.jsonl.gz")
	f, err := os.Create(gzPath)
	require.NoError(t, err)
	gz := gzip.NewWriter(f)
	ev := Event{
		Version:          SchemaVersion,
		KeyFingerprint:   "abcd1234",
		SessionID:        "s_1111222233334444",
		SessionSource:    SessionSourceHeader,
		CacheSaltID:      EmptyCacheSaltID,
		TRecvUnix:        1762121400.5,
		TTFTS:            0.02,
		APITimeS:         0.15,
		Streaming:        true,
		StatusCode:       200,
		Termination:      TerminationNatural,
		Model:            "Qwen/Qwen2.5-0.5B-Instruct",
		TokenizerBackend: "vllm",
		BlockSize:        16,
		In:               32,
		Out:              10,
		OutSource:        OutSourceUsage,
		HashIDs:          []uint64{1001, 1002},
	}
	b, err := json.Marshal(ev)
	require.NoError(t, err)
	_, err = gz.Write(append(b, '\n'))
	require.NoError(t, err)
	require.NoError(t, gz.Close())
	require.NoError(t, f.Close())

	loaded, err := LoadCapturedRecords([]string{gzPath})
	require.NoError(t, err)
	require.Len(t, loaded, 1)
	assert.Equal(t, []string{"1001", "1002"}, loaded[0].BlockHashes)
	assert.Equal(t, 32, loaded[0].TokenizedIn)

	traces, err := AssembleWekaTraces(loaded, AssembleOptions{HashIDScope: HashIDScopeGlobal})
	require.NoError(t, err)
	require.Len(t, traces, 1)
	assert.Equal(t, []int{1, 2}, traces[0].Requests[0].Turn.HashIDs)
}
