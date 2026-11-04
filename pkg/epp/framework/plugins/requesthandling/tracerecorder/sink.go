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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// DefaultBufferSize is the default non-blocking event channel capacity.
	DefaultBufferSize = 4096

	// DefaultMaxSegmentBytes is the default uncompressed byte threshold before segment rotation (64 MiB).
	DefaultMaxSegmentBytes int64 = 64 * 1024 * 1024

	// DefaultFlushInterval is the default max age of an open staging segment before sealing.
	DefaultFlushInterval = 30 * time.Second
)

// TraceSink consumes anonymized trace events off the request hot path.
type TraceSink interface {
	// TryEnqueue attempts to enqueue ev without blocking. Returns false if the
	// sink buffer is full or the sink has been closed.
	TryEnqueue(ev Event) bool

	// Close flushes and seals any buffered events, waiting until completion or ctx cancellation.
	Close(ctx context.Context) error
}

// MemoryTraceSink is an in-memory TraceSink used in unit and contract tests.
type MemoryTraceSink struct {
	mu       sync.Mutex
	events   []Event
	capacity int
	closed   bool
	dropped  atomic.Uint64
}

// NewMemoryTraceSink constructs an unbounded MemoryTraceSink (or bounded when capacity > 0).
func NewMemoryTraceSink(capacity int) *MemoryTraceSink {
	return &MemoryTraceSink{capacity: capacity}
}

// TryEnqueue appends ev to the in-memory slice if not closed and within capacity.
func (m *MemoryTraceSink) TryEnqueue(ev Event) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		m.dropped.Add(1)
		return false
	}
	if m.capacity > 0 && len(m.events) >= m.capacity {
		m.dropped.Add(1)
		return false
	}
	// Defensive copy of HashIDs so caller mutations cannot race with readers.
	if ev.HashIDs != nil {
		copied := make([]uint64, len(ev.HashIDs))
		copy(copied, ev.HashIDs)
		ev.HashIDs = copied
	} else {
		ev.HashIDs = []uint64{}
	}
	m.events = append(m.events, ev)
	return true
}

// Close marks the sink closed.
func (m *MemoryTraceSink) Close(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

// Events returns a snapshot copy of all recorded events.
func (m *MemoryTraceSink) Events() []Event {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Event, len(m.events))
	copy(out, m.events)
	return out
}

// Dropped returns the count of events rejected by TryEnqueue.
func (m *MemoryTraceSink) Dropped() uint64 {
	return m.dropped.Load()
}

// SealedGzipSinkConfig configures SealedGzipSink.
type SealedGzipSinkConfig struct {
	OutputDir       string
	StagingDir      string
	PodName         string
	MaxSegmentBytes int64
	FlushInterval   time.Duration
	BufferSize      int
	Clock           func() time.Time
	// BlockConsumerCh, if non-nil, pauses the background writer loop until closed
	// or signaled; test-only hook for deterministic backpressure verification.
	BlockConsumerCh <-chan struct{}
}

// SealedGzipSink streams active JSONL+gzip writes to a pod-local StagingDir and
// seals completed segments (<pod>-<epoch_ms>-<seq>.jsonl.gz) into OutputDir on
// size rotation, time rotation, or graceful Close. Single-close semantics into
// OutputDir ensure full compatibility with GKE GCS FUSE CSI volumes.
type SealedGzipSink struct {
	outputDir       string
	stagingDir      string
	podName         string
	maxSegmentBytes int64
	flushInterval   time.Duration
	clock           func() time.Time
	blockConsumerCh <-chan struct{}

	enqueueMu sync.RWMutex
	closed    bool
	queue     chan Event
	flushReq  chan chan error
	stopCh    chan struct{}
	doneCh    chan struct{}

	dropped  atomic.Uint64
	sealErrs atomic.Uint64

	// Writer-goroutine-owned state:
	seq               uint64
	activeFile        *os.File
	activeGzip        *gzip.Writer
	activePath        string
	activeOpenedAt    time.Time
	activeUncompBytes int64
	lastErr           error
}

// NewSealedGzipSink creates staging and output directories and starts the background writer goroutine.
func NewSealedGzipSink(cfg SealedGzipSinkConfig) (*SealedGzipSink, error) {
	if strings.TrimSpace(cfg.OutputDir) == "" {
		return nil, errors.New("SealedGzipSink: outputDir is required")
	}
	stagingDir := strings.TrimSpace(cfg.StagingDir)
	if stagingDir == "" {
		stagingDir = filepath.Join(os.TempDir(), "epp-traces-staging")
	}
	if err := os.MkdirAll(stagingDir, 0o750); err != nil {
		return nil, fmt.Errorf("SealedGzipSink: failed to create stagingDir %q: %w", stagingDir, err)
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o750); err != nil {
		return nil, fmt.Errorf("SealedGzipSink: failed to create outputDir %q: %w", cfg.OutputDir, err)
	}

	podName := sanitizePodName(cfg.PodName)
	if podName == "" {
		podName = sanitizePodName(os.Getenv("POD_NAME"))
	}
	if podName == "" {
		podName = sanitizePodName(os.Getenv("HOSTNAME"))
	}
	if podName == "" {
		podName = "epp"
	}

	maxBytes := cfg.MaxSegmentBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxSegmentBytes
	}
	flushInterval := cfg.FlushInterval
	if flushInterval <= 0 {
		flushInterval = DefaultFlushInterval
	}
	bufSize := cfg.BufferSize
	if bufSize <= 0 {
		bufSize = DefaultBufferSize
	}
	clock := cfg.Clock
	if clock == nil {
		clock = time.Now
	}

	s := &SealedGzipSink{
		outputDir:       cfg.OutputDir,
		stagingDir:      stagingDir,
		podName:         podName,
		maxSegmentBytes: maxBytes,
		flushInterval:   flushInterval,
		clock:           clock,
		blockConsumerCh: cfg.BlockConsumerCh,
		queue:           make(chan Event, bufSize),
		flushReq:        make(chan chan error, 1),
		stopCh:          make(chan struct{}),
		doneCh:          make(chan struct{}),
	}
	go s.writerLoop()
	return s, nil
}

// TryEnqueue enqueues ev without blocking. If the ring/channel buffer is full or
// the sink is closed, it increments the dropped counter and returns false immediately.
func (s *SealedGzipSink) TryEnqueue(ev Event) bool {
	s.enqueueMu.RLock()
	defer s.enqueueMu.RUnlock()
	if s.closed {
		s.dropped.Add(1)
		return false
	}
	select {
	case s.queue <- ev:
		return true
	default:
		s.dropped.Add(1)
		return false
	}
}

// CheckRotation synchronously drains queued events and triggers time-based rotation
// if clock() has advanced past flushInterval. Useful in tests with a fake clock.
func (s *SealedGzipSink) CheckRotation(ctx context.Context) error {
	s.enqueueMu.RLock()
	if s.closed {
		s.enqueueMu.RUnlock()
		return errors.New("SealedGzipSink is closed")
	}
	reply := make(chan error, 1)
	select {
	case s.flushReq <- reply:
		s.enqueueMu.RUnlock()
	case <-ctx.Done():
		s.enqueueMu.RUnlock()
		return ctx.Err()
	}
	select {
	case err := <-reply:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close stops accepting new events, drains all queued events, seals the active
// staging segment into OutputDir, and waits for completion or ctx cancellation.
func (s *SealedGzipSink) Close(ctx context.Context) error {
	s.enqueueMu.Lock()
	if !s.closed {
		s.closed = true
		close(s.stopCh)
	}
	s.enqueueMu.Unlock()

	select {
	case <-s.doneCh:
		return s.lastErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Dropped returns the total count of dropped events due to backpressure or post-close enqueue.
func (s *SealedGzipSink) Dropped() uint64 {
	return s.dropped.Load()
}

// SealErrors returns the total count of segment seal errors encountered.
func (s *SealedGzipSink) SealErrors() uint64 {
	return s.sealErrs.Load()
}

func (s *SealedGzipSink) writerLoop() {
	defer close(s.doneCh)

	if s.blockConsumerCh != nil {
		select {
		case <-s.blockConsumerCh:
		case <-s.stopCh:
		}
	}

	tickInterval := min(s.flushInterval, time.Second)
	if tickInterval <= 0 {
		tickInterval = 100 * time.Millisecond
	}
	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()

	for {
		select {
		case ev := <-s.queue:
			if err := s.writeEvent(ev); err != nil {
				s.lastErr = err
				s.sealErrs.Add(1)
			}
		case reply := <-s.flushReq:
			s.drainAvailable()
			var err error
			if s.activeFile != nil && s.clock().Sub(s.activeOpenedAt) >= s.flushInterval {
				err = s.sealActiveSegment()
				if err != nil {
					s.lastErr = err
					s.sealErrs.Add(1)
				}
			}
			reply <- err
		case <-ticker.C:
			if s.activeFile != nil && s.clock().Sub(s.activeOpenedAt) >= s.flushInterval {
				if err := s.sealActiveSegment(); err != nil {
					s.lastErr = err
					s.sealErrs.Add(1)
				}
			}
		case <-s.stopCh:
			s.drainAvailable()
			if s.activeFile != nil {
				if err := s.sealActiveSegment(); err != nil {
					s.lastErr = err
					s.sealErrs.Add(1)
				}
			}
			return
		}
	}
}

func (s *SealedGzipSink) drainAvailable() {
	for {
		select {
		case ev := <-s.queue:
			if err := s.writeEvent(ev); err != nil {
				s.lastErr = err
				s.sealErrs.Add(1)
			}
		default:
			return
		}
	}
}

func (s *SealedGzipSink) writeEvent(ev Event) error {
	if ev.HashIDs == nil {
		ev.HashIDs = []uint64{}
	}
	line, err := json.Marshal(ev)
	if err != nil {
		return fmt.Errorf("failed to marshal trace event: %w", err)
	}
	line = append(line, '\n')

	if s.activeFile == nil {
		if err := s.openStagingSegment(); err != nil {
			return err
		}
	}

	n, err := s.activeGzip.Write(line)
	if err != nil {
		return fmt.Errorf("failed to write gzip trace event: %w", err)
	}
	s.activeUncompBytes += int64(n)

	if s.activeUncompBytes >= s.maxSegmentBytes || s.clock().Sub(s.activeOpenedAt) >= s.flushInterval {
		return s.sealActiveSegment()
	}
	return nil
}

func (s *SealedGzipSink) openStagingSegment() error {
	s.seq++
	now := s.clock()
	stagingName := fmt.Sprintf(".staging-%s-%d-%06d.jsonl.gz.tmp", s.podName, now.UnixNano(), s.seq)
	stagingPath := filepath.Join(s.stagingDir, stagingName)
	f, err := os.OpenFile(stagingPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600) // #nosec G304 -- internal staging path
	if err != nil {
		return fmt.Errorf("failed to open staging trace segment %q: %w", stagingPath, err)
	}
	s.activeFile = f
	s.activeGzip = gzip.NewWriter(f)
	s.activePath = stagingPath
	s.activeOpenedAt = now
	s.activeUncompBytes = 0
	return nil
}

func (s *SealedGzipSink) sealActiveSegment() error {
	if s.activeFile == nil {
		return nil
	}
	gz := s.activeGzip
	f := s.activeFile
	stagingPath := s.activePath
	seq := s.seq

	s.activeGzip = nil
	s.activeFile = nil
	s.activePath = ""
	s.activeUncompBytes = 0

	gzErr := gz.Close()
	closeErr := f.Close()
	if gzErr != nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("failed to close gzip writer for %q: %w", stagingPath, gzErr)
	}
	if closeErr != nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("failed to close staging file %q: %w", stagingPath, closeErr)
	}

	sealedName := fmt.Sprintf("%s-%d-%06d.jsonl.gz", s.podName, s.clock().UnixMilli(), seq)
	sealedPath := filepath.Join(s.outputDir, sealedName)

	if err := moveOrCopySealedFile(stagingPath, sealedPath); err != nil {
		return fmt.Errorf("failed to seal segment %q to %q: %w", stagingPath, sealedPath, err)
	}
	return nil
}

// moveOrCopySealedFile renames src to dst when on the same mount, and falls back
// to a single-open-write-close stream copy when StagingDir (/tmp emptyDir) and
// OutputDir (/traces GCS FUSE CSI) reside on separate filesystems (EXDEV).
func moveOrCopySealedFile(src, dst string) error {
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	in, err := os.Open(src) // #nosec G304 -- internal staging path
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600) // #nosec G304 -- configured trace output path
	if err != nil {
		return err
	}
	if _, copyErr := io.Copy(out, in); copyErr != nil {
		_ = out.Close()
		_ = os.Remove(dst)
		return copyErr
	}
	if closeErr := out.Close(); closeErr != nil {
		_ = os.Remove(dst)
		return closeErr
	}
	_ = in.Close()
	_ = os.Remove(src)
	return nil
}

func sanitizePodName(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}
