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

package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/tracerecorder"
)

func TestTraceExportCLI_JSONLAndGzipInputs(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	inDir := filepath.Join(tmpDir, "in")
	require.NoError(t, os.MkdirAll(inDir, 0o750))

	// Write one .jsonl.gz Event file and one .jsonl CapturedRecord file.
	gzPath := filepath.Join(inDir, "pod-a-1000-000001.jsonl.gz")
	f, err := os.Create(gzPath)
	require.NoError(t, err)
	gz := gzip.NewWriter(f)
	ev := tracerecorder.Event{
		Version:          tracerecorder.SchemaVersion,
		KeyFingerprint:   "abcd1234",
		SessionID:        "s_aaaaaaaaaaaaaaaa",
		SessionSource:    tracerecorder.SessionSourceHeader,
		CacheSaltID:      tracerecorder.EmptyCacheSaltID,
		TRecvUnix:        1762121400.0,
		TTFTS:            0.02,
		APITimeS:         0.10,
		Streaming:        true,
		StatusCode:       200,
		Termination:      tracerecorder.TerminationNatural,
		Model:            "Qwen/Qwen2.5-0.5B-Instruct",
		TokenizerBackend: "vllm",
		BlockSize:        16,
		In:               32,
		Out:              12,
		OutSource:        tracerecorder.OutSourceUsage,
		HashIDs:          []uint64{101, 102},
	}
	rawEv, err := json.Marshal(ev)
	require.NoError(t, err)
	_, err = gz.Write(append(rawEv, '\n'))
	require.NoError(t, err)
	require.NoError(t, gz.Close())
	require.NoError(t, f.Close())

	outJSONL := filepath.Join(tmpDir, "out", "traces.jsonl")
	var stdout bytes.Buffer
	err = run([]string{
		"--input-dir", inDir,
		"--output-jsonl", outJSONL,
		"--hash-id-scope", "local",
	}, &stdout)
	require.NoError(t, err)
	assert.Contains(t, stdout.String(), "exported 1 traces from 1 captured records")

	content, err := os.ReadFile(outJSONL)
	require.NoError(t, err)
	var tr tracerecorder.WekaTrace
	require.NoError(t, json.Unmarshal(bytes.TrimSpace(content), &tr))
	assert.Equal(t, "trace_0000", tr.ID)
	require.Len(t, tr.Requests, 1)
	assert.Equal(t, []int{1, 2}, tr.Requests[0].Turn.HashIDs)

	// Also verify --output-dir mode.
	outTraceDir := filepath.Join(tmpDir, "out_dir")
	stdout.Reset()
	err = run([]string{
		"--input-dir", inDir,
		"--output-dir", outTraceDir,
	}, &stdout)
	require.NoError(t, err)
	_, statErr := os.Stat(filepath.Join(outTraceDir, "trace_0000.json"))
	require.NoError(t, statErr)
}
