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

	sinkDirPerm  = 0o750
	sinkFilePerm = 0o600
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
	PluginName      string
	MaxSegmentBytes int64
	MaxRotatedFiles int
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
	outputRoot      *os.Root
	stagingRoot     *os.Root
	podName         string
	pluginName      string
	maxSegmentBytes int64
	maxRotatedFiles int
	flushInterval   time.Duration
	clock           func() time.Time
	blockConsumerCh <-chan struct{}

	enqueueMu sync.RWMutex
	closed    bool
	queue     chan Event
	flushReq  chan chan error
	stopCh    chan struct{}
	doneCh    chan struct{}

	writtenRecords atomic.Uint64
	dropped        atomic.Uint64
	sealErrs       atomic.Uint64
	rotations      atomic.Uint64
	activeName     atomic.Value // string

	// Writer-goroutine-owned state:
	seq               uint64
	activeFile        *os.File
	activeGzip        *gzip.Writer
	activePath        string
	activeStagingName string
	activeOpenedAt    time.Time
	activeUncompBytes int64
	rotatedFiles      []string
	lastErr           error
}

// NewSealedGzipSink creates staging and output directories and starts the background writer goroutine.
func NewSealedGzipSink(cfg SealedGzipSinkConfig) (*SealedGzipSink, error) {
	outputDir := strings.TrimSpace(cfg.OutputDir)
	if outputDir == "" {
		return nil, errors.New("SealedGzipSink: outputDir is required")
	}
	outputDir = filepath.Clean(outputDir)

	stagingDir := strings.TrimSpace(cfg.StagingDir)
	if stagingDir == "" {
		stagingDir = filepath.Join(os.TempDir(), "epp-traces-staging")
	}
	stagingDir = filepath.Clean(stagingDir)

	if err := ensureSinkDir(stagingDir, os.Chmod); err != nil {
		return nil, fmt.Errorf("SealedGzipSink: failed to prepare stagingDir %q: %w", stagingDir, err)
	}
	if err := ensureSinkDir(outputDir, os.Chmod); err != nil {
		return nil, fmt.Errorf("SealedGzipSink: failed to prepare outputDir %q: %w", outputDir, err)
	}

	stagingRoot, err := os.OpenRoot(stagingDir)
	if err != nil {
		return nil, fmt.Errorf("SealedGzipSink: failed to open stagingDir root %q: %w", stagingDir, err)
	}
	outputRoot, err := os.OpenRoot(outputDir)
	if err != nil {
		_ = stagingRoot.Close()
		return nil, fmt.Errorf("SealedGzipSink: failed to open outputDir root %q: %w", outputDir, err)
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

	pluginName := strings.TrimSpace(cfg.PluginName)
	if pluginName == "" {
		pluginName = PluginType
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

	registerRecorderMetrics()

	s := &SealedGzipSink{
		outputDir:       outputDir,
		stagingDir:      stagingDir,
		outputRoot:      outputRoot,
		stagingRoot:     stagingRoot,
		podName:         podName,
		pluginName:      pluginName,
		maxSegmentBytes: maxBytes,
		maxRotatedFiles: max(0, cfg.MaxRotatedFiles),
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

func ensureSinkDir(cleanDir string, chmodFn func(string, os.FileMode) error) error {
	info, statErr := os.Stat(cleanDir)
	preExisting := statErr == nil && info.IsDir()
	if err := os.MkdirAll(cleanDir, sinkDirPerm); err != nil {
		return err
	}
	if err := chmodFn(cleanDir, sinkDirPerm); err != nil {
		// Pre-existing volume mountpoints (e.g. Kubernetes emptyDir/PVC with fsGroup)
		// are owned by root while writable by the container GID; POSIX forbids non-owner chmod.
		if preExisting && errors.Is(err, os.ErrPermission) {
			return nil
		}
		return err
	}
	return nil
}

// TryEnqueue enqueues ev without blocking. If the ring/channel buffer is full or
// the sink is closed, it increments the dropped counter and returns false immediately.
func (s *SealedGzipSink) TryEnqueue(ev Event) bool {
	s.enqueueMu.RLock()
	defer s.enqueueMu.RUnlock()
	if s.closed {
		s.dropped.Add(1)
		droppedRecordsMetric.WithLabelValues(s.pluginName, dropReasonSinkClosed).Inc()
		return false
	}
	select {
	case s.queue <- ev:
		return true
	default:
		s.dropped.Add(1)
		droppedRecordsMetric.WithLabelValues(s.pluginName, dropReasonBufferFull).Inc()
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

// WrittenRecords returns the total count of events written to segment files.
func (s *SealedGzipSink) WrittenRecords() uint64 {
	return s.writtenRecords.Load()
}

// SealErrors returns the total count of segment seal or write errors encountered.
func (s *SealedGzipSink) SealErrors() uint64 {
	return s.sealErrs.Load()
}

// Rotations returns the total count of sealed segment files produced.
func (s *SealedGzipSink) Rotations() uint64 {
	return s.rotations.Load()
}

// ActiveFile returns the basename of the current staging or last sealed segment file.
func (s *SealedGzipSink) ActiveFile() string {
	v, _ := s.activeName.Load().(string)
	return v
}

func (s *SealedGzipSink) writerLoop() {
	defer func() {
		if s.stagingRoot != nil {
			_ = s.stagingRoot.Close()
		}
		if s.outputRoot != nil {
			_ = s.outputRoot.Close()
		}
		close(s.doneCh)
	}()

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
				s.recordWriteError(err)
			}
		case reply := <-s.flushReq:
			s.drainAvailable()
			var err error
			if s.activeFile != nil && s.clock().Sub(s.activeOpenedAt) >= s.flushInterval {
				err = s.sealActiveSegment()
				if err != nil {
					s.recordWriteError(err)
				}
			}
			reply <- err
		case <-ticker.C:
			if s.activeFile != nil && s.clock().Sub(s.activeOpenedAt) >= s.flushInterval {
				if err := s.sealActiveSegment(); err != nil {
					s.recordWriteError(err)
				}
			}
		case <-s.stopCh:
			s.drainAvailable()
			if s.activeFile != nil {
				if err := s.sealActiveSegment(); err != nil {
					s.recordWriteError(err)
				}
			}
			return
		}
	}
}

func (s *SealedGzipSink) recordWriteError(err error) {
	s.lastErr = err
	s.sealErrs.Add(1)
	droppedRecordsMetric.WithLabelValues(s.pluginName, dropReasonWriteError).Inc()
}

func (s *SealedGzipSink) drainAvailable() {
	for {
		select {
		case ev := <-s.queue:
			if err := s.writeEvent(ev); err != nil {
				s.recordWriteError(err)
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
	s.writtenRecords.Add(1)
	writtenEventsMetric.WithLabelValues(s.pluginName).Inc()

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
	f, err := s.stagingRoot.OpenFile(stagingName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, sinkFilePerm)
	if err != nil {
		return fmt.Errorf("failed to open staging trace segment %q: %w", stagingPath, err)
	}
	s.activeFile = f
	s.activeGzip = gzip.NewWriter(f)
	s.activePath = stagingPath
	s.activeStagingName = stagingName
	s.activeOpenedAt = now
	s.activeUncompBytes = 0
	s.activeName.Store(stagingName)
	return nil
}

func (s *SealedGzipSink) sealActiveSegment() error {
	if s.activeFile == nil {
		return nil
	}
	gz := s.activeGzip
	f := s.activeFile
	stagingPath := s.activePath
	stagingName := s.activeStagingName
	seq := s.seq

	s.activeGzip = nil
	s.activeFile = nil
	s.activePath = ""
	s.activeStagingName = ""
	s.activeUncompBytes = 0

	gzErr := gz.Close()
	closeErr := f.Close()
	if gzErr != nil {
		_ = s.stagingRoot.Remove(stagingName)
		return fmt.Errorf("failed to close gzip writer for %q: %w", stagingPath, gzErr)
	}
	if closeErr != nil {
		_ = s.stagingRoot.Remove(stagingName)
		return fmt.Errorf("failed to close staging file %q: %w", stagingPath, closeErr)
	}

	sealedName := fmt.Sprintf("%s-%d-%06d.jsonl.gz", s.podName, s.clock().UnixMilli(), seq)
	sealedPath := filepath.Join(s.outputDir, sealedName)

	if err := s.moveOrCopySealedFile(stagingPath, stagingName, sealedPath, sealedName); err != nil {
		return fmt.Errorf("failed to seal segment %q to %q: %w", stagingPath, sealedPath, err)
	}

	s.activeName.Store(sealedName)
	s.rotations.Add(1)
	rotationsMetric.WithLabelValues(s.pluginName).Inc()

	if s.maxRotatedFiles > 0 {
		s.rotatedFiles = append(s.rotatedFiles, sealedName)
		for len(s.rotatedFiles) > s.maxRotatedFiles {
			oldest := s.rotatedFiles[0]
			s.rotatedFiles = s.rotatedFiles[1:]
			if err := s.outputRoot.Remove(oldest); err != nil && !errors.Is(err, os.ErrNotExist) {
				s.lastErr = err
			}
		}
	}
	return nil
}

// moveOrCopySealedFile renames src to dst when on the same mount, and falls back
// to a single-open-write-close stream copy confined to stagingRoot and outputRoot
// when StagingDir (/tmp emptyDir) and OutputDir (/traces GCS FUSE CSI) reside on
// separate filesystems (EXDEV).
func (s *SealedGzipSink) moveOrCopySealedFile(srcPath, srcName, dstPath, dstName string) error {
	if err := os.Rename(srcPath, dstPath); err == nil {
		return nil
	}
	in, err := s.stagingRoot.Open(srcName)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	out, err := s.outputRoot.OpenFile(dstName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, sinkFilePerm)
	if err != nil {
		return err
	}
	if _, copyErr := io.Copy(out, in); copyErr != nil {
		_ = out.Close()
		_ = s.outputRoot.Remove(dstName)
		return copyErr
	}
	if closeErr := out.Close(); closeErr != nil {
		_ = s.outputRoot.Remove(dstName)
		return closeErr
	}
	_ = in.Close()
	_ = s.stagingRoot.Remove(srcName)
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
