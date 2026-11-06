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
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock(initial time.Time) *fakeClock {
	return &fakeClock{now: initial}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func readAllGzipEvents(t *testing.T, dir string) ([]Event, []string) {
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

	var events []Event
	for _, path := range files {
		f, err := os.Open(path) // #nosec G304 -- test directory path
		require.NoError(t, err)
		gz, err := gzip.NewReader(f)
		require.NoError(t, err)

		scanner := bufio.NewScanner(gz)
		for scanner.Scan() {
			var ev Event
			require.NoError(t, json.Unmarshal(scanner.Bytes(), &ev))
			events = append(events, ev)
		}
		require.NoError(t, scanner.Err())
		require.NoError(t, gz.Close())
		require.NoError(t, f.Close())
	}
	return events, files
}

func sampleEvent(seq int) Event {
	return Event{
		Version:          SchemaVersion,
		KeyFingerprint:   "0123abcd",
		SessionID:        "s_1122334455667788",
		SessionSource:    SessionSourceHeader,
		CacheSaltID:      EmptyCacheSaltID,
		TRecvUnix:        1762121400.0 + float64(seq),
		TTFTS:            0.025,
		APITimeS:         0.250,
		Streaming:        true,
		StatusCode:       200,
		Termination:      TerminationNatural,
		Model:            "Qwen/Qwen2.5-0.5B-Instruct",
		TokenizerBackend: "vllm",
		BlockSize:        16,
		In:               32,
		Out:              12,
		OutSource:        OutSourceUsage,
		MsgCount:         2,
		HasSystemPrompt:  true,
		HasTools:         false,
		HashIDs:          []uint64{101, 202},
	}
}

func TestTraceSink_SharedContract(t *testing.T) {
	t.Parallel()

	type sinkHarness struct {
		name  string
		build func(t *testing.T) (TraceSink, func() []Event)
	}

	harnesses := []sinkHarness{
		{
			name: "MemoryTraceSink",
			build: func(_ *testing.T) (TraceSink, func() []Event) {
				s := NewMemoryTraceSink(0)
				return s, s.Events
			},
		},
		{
			name: "SealedGzipSink",
			build: func(t *testing.T) (TraceSink, func() []Event) {
				outDir := filepath.Join(t.TempDir(), "out")
				stageDir := filepath.Join(t.TempDir(), "stage")
				clk := newFakeClock(time.Unix(1762121400, 0))
				s, err := NewSealedGzipSink(SealedGzipSinkConfig{
					OutputDir:       outDir,
					StagingDir:      stageDir,
					PodName:         "epp-pod-a",
					MaxSegmentBytes: 10 * 1024 * 1024,
					FlushInterval:   time.Hour,
					BufferSize:      64,
					Clock:           clk.Now,
				})
				require.NoError(t, err)
				return s, func() []Event {
					evs, _ := readAllGzipEvents(t, outDir)
					return evs
				}
			},
		},
	}

	for _, h := range harnesses {
		t.Run(h.name, func(t *testing.T) {
			t.Parallel()
			sink, readEvents := h.build(t)
			for i := range 5 {
				ev := sampleEvent(i)
				if i == 0 {
					ev.HashIDs = nil // verify empty slice normalization to []
				}
				require.True(t, sink.TryEnqueue(ev))
			}
			require.NoError(t, sink.Close(context.Background()))
			// Enqueue after Close must return false without blocking or panicking.
			require.False(t, sink.TryEnqueue(sampleEvent(99)))

			got := readEvents()
			require.Len(t, got, 5)
			require.NotNil(t, got[0].HashIDs)
			assert.Empty(t, got[0].HashIDs)
			assert.Equal(t, []uint64{101, 202}, got[1].HashIDs)
		})
	}
}

func TestSealedGzipSink_LocalStagingAndBackpressure(t *testing.T) {
	t.Parallel()

	t.Run("active_writes_stay_in_staging_until_rotation_or_close", func(t *testing.T) {
		t.Parallel()
		baseDir := t.TempDir()
		stageDir := filepath.Join(baseDir, "staging")
		outDir := filepath.Join(baseDir, "traces")
		clk := newFakeClock(time.UnixMilli(1762121400123))

		sink, err := NewSealedGzipSink(SealedGzipSinkConfig{
			OutputDir:       outDir,
			StagingDir:      stageDir,
			PodName:         "epp-deployment-7f9b-x2k",
			MaxSegmentBytes: 1024 * 1024, // 1 MiB so single event stays open in staging
			FlushInterval:   30 * time.Second,
			BufferSize:      64,
			Clock:           clk.Now,
		})
		require.NoError(t, err)

		require.True(t, sink.TryEnqueue(sampleEvent(1)))
		// Synchronously drain the queue without advancing clock past flushInterval.
		require.NoError(t, sink.CheckRotation(context.Background()))

		// OutputDir MUST still be empty (GCS FUSE close-to-publish semantics), while stagingDir holds the .tmp file.
		outEntries, err := os.ReadDir(outDir)
		require.NoError(t, err)
		assert.Empty(t, outEntries, "unsealed segment must not appear in outputDir")

		stageEntries, err := os.ReadDir(stageDir)
		require.NoError(t, err)
		require.Len(t, stageEntries, 1, "active segment must exist in stagingDir")

		// Advance injected clock past FlushInterval and trigger rotation without sleeping.
		clk.Advance(31 * time.Second)
		require.NoError(t, sink.CheckRotation(context.Background()))

		eventsAfterTimeRotate, sealedFiles := readAllGzipEvents(t, outDir)
		require.Len(t, sealedFiles, 1)
		assert.Contains(t, filepath.Base(sealedFiles[0]), "epp-deployment-7f9b-x2k-1762121431123-000001.jsonl.gz")
		require.Len(t, eventsAfterTimeRotate, 1)

		// Staging directory should be clean after sealing.
		stageEntries, err = os.ReadDir(stageDir)
		require.NoError(t, err)
		assert.Empty(t, stageEntries)

		// Now test size-based rotation and final Close sealing.
		sizeSinkOut := filepath.Join(baseDir, "traces_size")
		sizeSinkStage := filepath.Join(baseDir, "staging_size")
		sizeSink, err := NewSealedGzipSink(SealedGzipSinkConfig{
			OutputDir:       sizeSinkOut,
			StagingDir:      sizeSinkStage,
			PodName:         "epp-size",
			MaxSegmentBytes: 400, // ~1-2 JSONL lines per segment
			FlushInterval:   time.Hour,
			BufferSize:      64,
			Clock:           clk.Now,
		})
		require.NoError(t, err)
		for i := range 5 {
			clk.Advance(time.Millisecond)
			require.True(t, sizeSink.TryEnqueue(sampleEvent(i)))
		}
		require.NoError(t, sizeSink.Close(context.Background()))

		sizeEvents, sizeFiles := readAllGzipEvents(t, sizeSinkOut)
		require.Len(t, sizeEvents, 5)
		assert.GreaterOrEqual(t, len(sizeFiles), 2, "expected multiple rotated segments")
		assert.Zero(t, sizeSink.SealErrors())

		require.NoError(t, sink.Close(context.Background()))
	})

	t.Run("full_ring_buffer_drops_non_blocking_under_1ms", func(t *testing.T) {
		t.Parallel()
		baseDir := t.TempDir()
		blockCh := make(chan struct{})

		sink, err := NewSealedGzipSink(SealedGzipSinkConfig{
			OutputDir:       filepath.Join(baseDir, "out"),
			StagingDir:      filepath.Join(baseDir, "stage"),
			PodName:         "epp-bp",
			MaxSegmentBytes: 1024 * 1024,
			FlushInterval:   time.Hour,
			BufferSize:      8,
			BlockConsumerCh: blockCh,
		})
		require.NoError(t, err)

		for i := range 8 {
			require.True(t, sink.TryEnqueue(sampleEvent(i)))
		}

		// Buffer is completely full and consumer is paused; 100 overflow calls must drop non-blocking in < 1ms.
		overflowEv := sampleEvent(999)
		var anyAccepted bool
		start := time.Now()
		for range 100 {
			if sink.TryEnqueue(overflowEv) {
				anyAccepted = true
			}
		}
		elapsed := time.Since(start)
		require.False(t, anyAccepted)
		assert.Less(t, elapsed, time.Millisecond, "non-blocking drop path exceeded 1ms for 100 calls")
		assert.Equal(t, uint64(100), sink.Dropped())

		close(blockCh)
		require.NoError(t, sink.Close(context.Background()))

		evs, _ := readAllGzipEvents(t, filepath.Join(baseDir, "out"))
		require.Len(t, evs, 8)
	})

	t.Run("ensureSinkDir_tolerates_permission_error_on_preexisting_dir_and_prunes_maxRotatedFiles", func(t *testing.T) {
		t.Parallel()
		baseDir := t.TempDir()
		existingDir := filepath.Join(baseDir, "k8s-mount")
		require.NoError(t, os.MkdirAll(existingDir, 0o750))

		// Simulate Kubernetes emptyDir/PVC root-owned mountpoint where Chmod returns os.ErrPermission.
		err := ensureSinkDir(existingDir, func(_ string, _ os.FileMode) error {
			return os.ErrPermission
		})
		require.NoError(t, err)

		// Verify MaxRotatedFiles pruning retains only the newest N sealed .jsonl.gz files.
		pruneOut := filepath.Join(baseDir, "prune_out")
		pruneStage := filepath.Join(baseDir, "prune_stage")
		clk := newFakeClock(time.UnixMilli(1762121500000))
		pruneSink, err := NewSealedGzipSink(SealedGzipSinkConfig{
			OutputDir:       pruneOut,
			StagingDir:      pruneStage,
			PodName:         "epp-prune",
			MaxSegmentBytes: 100, // rotate after every event
			MaxRotatedFiles: 2,
			FlushInterval:   time.Hour,
			BufferSize:      32,
			Clock:           clk.Now,
		})
		require.NoError(t, err)

		for i := range 5 {
			clk.Advance(10 * time.Millisecond)
			require.True(t, pruneSink.TryEnqueue(sampleEvent(i)))
		}
		require.NoError(t, pruneSink.Close(context.Background()))
		assert.Equal(t, uint64(5), pruneSink.WrittenRecords())
		assert.Equal(t, uint64(5), pruneSink.Rotations())

		evs, files := readAllGzipEvents(t, pruneOut)
		require.Len(t, files, 2, "MaxRotatedFiles=2 must retain only the 2 newest rotated .jsonl.gz files")
		require.Len(t, evs, 2)
	})
}
