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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/prefixhash"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionid/constants"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

const (
	// PluginType is the registered EPP plugin type for the KV-block trace recorder.
	PluginType = "kv-block-trace-recorder"

	// DefaultBlockSizeTokens is the default KV-cache block size in tokens (matching vLLM).
	DefaultBlockSizeTokens = 16

	// Client and harness session/subagent correlation headers not defined in shared router packages.
	HeaderDynamoSessionID       = "x-dynamo-session-id"
	HeaderDynamoParentSessionID = "x-dynamo-parent-session-id"
	HeaderCorrelationID         = "x-correlation-id"
	HeaderSMGRoutingKey         = "x-smg-routing-key"
	HeaderClaudeCodeAgentID     = "x-claude-code-agent-id"

	subagentSessionDelimiter = "::sa:"

	dropReasonBufferFull = "buffer_full"
	dropReasonSinkClosed = "sink_closed"
	dropReasonWriteError = "write_error"

	skipReasonSampledOut   = "sampled_out"
	skipReasonNoTokens     = "no_tokens"
	skipReasonNonCompleted = "non_completed"
)

// Explicit request headers inspected for subagent_type resolution.
var (
	explicitSubagentTypeHeaders = []string{
		"x-subagent-type",
		"x-claude-code-subagent-type",
		"x-llm-d-subagent-type",
	}
)

// DefaultSessionSources returns the priority-ordered session sources used when
// Config.SessionSources is unset.
func DefaultSessionSources() []sessionaffinity.SessionIDSource {
	return []sessionaffinity.SessionIDSource{
		{Attribute: agentidentity.AgentIdentityKey.DataType()},
		{Attribute: attrsession.SessionIDDataKey.DataType(), Producer: sessionidconstants.SessionIDProducerType},
		{Header: sessionaffinity.DefaultSessionIDHeader},
		{Header: HeaderDynamoSessionID},
		{Header: HeaderCorrelationID},
		{Header: HeaderSMGRoutingKey},
		{Header: agentidentity.ClaudeCodeSessionHeader},
		{Header: agentidentity.OpenCodeSessionHeader},
		{Header: agentidentity.CodexSessionHeader},
		{Header: agentidentity.CodexSessionHeaderLegacy},
		{Header: metadata.FlowFairnessIDKey},
	}
}

// DefaultParentSessionSources returns the priority-ordered parent session sources
// used when Config.ParentSessionSources is unset.
func DefaultParentSessionSources() []sessionaffinity.SessionIDSource {
	return []sessionaffinity.SessionIDSource{
		{Header: HeaderDynamoParentSessionID},
	}
}

// DefaultSubagentSources returns the priority-ordered subagent identifier sources
// used when Config.SubagentSources is unset.
func DefaultSubagentSources() []sessionaffinity.SessionIDSource {
	return []sessionaffinity.SessionIDSource{
		{Header: "x-agent-id"},
		{Header: HeaderClaudeCodeAgentID},
		{Header: "x-llm-d-agent-id"},
	}
}

var (
	registerMetricsOnce sync.Once

	droppedEventsMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "llm_d_epp",
			Name:      "trace_recorder_dropped_events_total",
			Help:      "Total number of anonymized KV-block trace events dropped due to sink backpressure.",
		},
		[]string{"plugin_name"},
	)

	writtenEventsMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "llm_d_epp",
			Name:      "trace_recorder_written_events_total",
			Help:      "Total number of anonymized KV-block trace events persisted by the trace sink.",
		},
		[]string{"plugin_name"},
	)

	droppedRecordsMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "llm_d_epp",
			Name:      "trace_recorder_dropped_records_total",
			Help:      "Total number of anonymized KV-block trace records dropped before persistence, by reason.",
		},
		[]string{"plugin_name", "reason"},
	)

	skippedRequestsMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "llm_d_epp",
			Name:      "trace_recorder_skipped_requests_total",
			Help:      "Total number of inference requests skipped by the trace recorder, by reason.",
		},
		[]string{"plugin_name", "reason"},
	)

	rotationsMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: "llm_d_epp",
			Name:      "trace_recorder_rotations_total",
			Help:      "Total number of sealed trace segment rotations performed by the trace sink.",
		},
		[]string{"plugin_name"},
	)
)

func registerRecorderMetrics() {
	registerMetricsOnce.Do(func() {
		registerRecorderMetricsOn(ctrlmetrics.Registry)
	})
}

func registerRecorderMetricsOn(registerer prometheus.Registerer) {
	if registerer == nil {
		return
	}
	for _, c := range []prometheus.Collector{
		droppedEventsMetric,
		writtenEventsMetric,
		droppedRecordsMetric,
		skippedRequestsMetric,
		rotationsMetric,
	} {
		_ = registerer.Register(c)
	}
}

// Config configures the kv-block-trace-recorder plugin.
type Config struct {
	// OutputDir is the directory where sealed <pod>-<epoch_ms>-<seq>.jsonl.gz segments are written.
	OutputDir string `json:"outputDir"`
	// StagingDir is the pod-local directory used for active gzip writes prior to sealing.
	StagingDir string `json:"stagingDir,omitempty"`
	// PodName overrides POD_NAME / HOSTNAME in sealed segment filenames.
	PodName string `json:"podName,omitempty"`

	// SecretFilePath points to a mounted Kubernetes Secret file (>= 32 bytes) used for HMAC anonymization.
	SecretFilePath string `json:"secretFilePath,omitempty"`
	// AllowEphemeralSecretForDev allows generating an ephemeral 32-byte random secret when neither
	// secretFilePath nor EPP_TRACE_HMAC_SECRET is set. Must remain false in multi-replica production.
	AllowEphemeralSecretForDev bool `json:"allowEphemeralSecretForDev,omitempty"`

	// SampleRateBPS is the deterministic session-cohort sample rate in basis points [0, 10000].
	// Nil defaults to 10000 (100%); explicit 0 disables sampling (0%).
	SampleRateBPS *int `json:"sampleRateBPS,omitempty"`

	// BlockSizeTokens is the KV block size in tokens (defaults to 16).
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	// HashSeed aligns initial block hashing with vLLM's PYTHONHASHSEED when using vllm tokenization.
	HashSeed string `json:"hashSeed,omitempty"`
	// HashAlgorithm selects "cbor-fnv" (default) or "xxh64" for vllm block hashing.
	HashAlgorithm string `json:"hashAlgorithm,omitempty"`

	// AllowEstimateTokenizer permits running alongside token-producer's byte-estimate backend.
	// Defaults to false (requiring backend: vllm).
	AllowEstimateTokenizer bool `json:"allowEstimateTokenizer,omitempty"`

	// InjectStreamIncludeUsage opts into mutating streaming request bodies in PreRequest to set
	// stream_options.include_usage = true. Defaults to false (strictly read-only).
	InjectStreamIncludeUsage bool `json:"injectStreamIncludeUsage,omitempty"`

	// MaxSegmentBytes rotates the active staging segment once uncompressed JSONL bytes reach this size.
	MaxSegmentBytes int64 `json:"maxSegmentBytes,omitempty"`
	// MaxRotatedFiles bounds how many sealed .jsonl.gz files are retained in OutputDir (0 = unlimited).
	MaxRotatedFiles int `json:"maxRotatedFiles,omitempty"`
	// FlushInterval string duration (e.g. "30s") after which an open non-empty segment is sealed.
	FlushInterval string `json:"flushInterval,omitempty"`
	// BufferSize sets the non-blocking sink queue capacity (defaults to 4096).
	BufferSize int `json:"bufferSize,omitempty"`

	// SessionSources resolves the session identifier in priority order.
	SessionSources []sessionaffinity.SessionIDSource `json:"sessionSources,omitempty"`
	// ParentSessionSources resolves the parent session identifier for subagent requests.
	ParentSessionSources []sessionaffinity.SessionIDSource `json:"parentSessionSources,omitempty"`
	// SubagentSources resolves the subagent identifier within a parent session.
	SubagentSources []sessionaffinity.SessionIDSource `json:"subagentSources,omitempty"`

	// TokenizerBackendOverride is an optional test-only override ("vllm" or "estimate") used when
	// constructing Recorder directly via NewRecorder without a plugin.Handle.
	TokenizerBackendOverride string `json:"-"`
}

func (c Config) effectiveSampleRateBPS() int {
	if c.SampleRateBPS == nil {
		return MaxSampleRateBPS
	}
	if *c.SampleRateBPS < 0 {
		return 0
	}
	if *c.SampleRateBPS > MaxSampleRateBPS {
		return MaxSampleRateBPS
	}
	return *c.SampleRateBPS
}

func (c Config) effectiveBlockSize() int {
	if c.BlockSizeTokens <= 0 {
		return DefaultBlockSizeTokens
	}
	return c.BlockSizeTokens
}

func (c Config) effectiveSessionSources() ([]sessionaffinity.SessionIDSource, bool) {
	if len(c.SessionSources) == 0 {
		return DefaultSessionSources(), true
	}
	return c.SessionSources, false
}

func (c Config) effectiveParentSessionSources() []sessionaffinity.SessionIDSource {
	if len(c.ParentSessionSources) == 0 {
		return DefaultParentSessionSources()
	}
	return c.ParentSessionSources
}

func (c Config) effectiveSubagentSources() []sessionaffinity.SessionIDSource {
	if len(c.SubagentSources) == 0 {
		return DefaultSubagentSources()
	}
	return c.SubagentSources
}

type pendingRecord struct {
	sessionID        string
	parentSessionID  string
	sessionSource    string
	cacheSaltID      string
	agentID          string
	subagentType     string
	model            string
	tokenizerBackend string
	blockSize        int
	inTokens         int
	maxOutputTokens  *int64
	msgCount         int
	hasSystemPrompt  bool
	hasTools         bool
	streaming        bool
	dispatched       bool
	producedAt       time.Time
	// hashIDs holds already-anonymized full-block hashes in [1, 2^53-1].
	// In estimate mode, len(hashIDs) equals tokReq.TokenCount() / blockSize and
	// may be further sliced in ResponseBody if response.Usage.PromptTokens < inTokens.
	hashIDs []uint64

	evictOnce sync.Once
	onEvict   func()
}

func (r *pendingRecord) Clone() plugin.StateData {
	if r == nil {
		return nil
	}
	var maxOut *int64
	if r.maxOutputTokens != nil {
		v := *r.maxOutputTokens
		maxOut = &v
	}
	cp := &pendingRecord{
		sessionID:        r.sessionID,
		parentSessionID:  r.parentSessionID,
		sessionSource:    r.sessionSource,
		cacheSaltID:      r.cacheSaltID,
		agentID:          r.agentID,
		subagentType:     r.subagentType,
		model:            r.model,
		tokenizerBackend: r.tokenizerBackend,
		blockSize:        r.blockSize,
		inTokens:         r.inTokens,
		maxOutputTokens:  maxOut,
		msgCount:         r.msgCount,
		hasSystemPrompt:  r.hasSystemPrompt,
		hasTools:         r.hasTools,
		streaming:        r.streaming,
		dispatched:       r.dispatched,
		producedAt:       r.producedAt,
	}
	if r.hashIDs != nil {
		cp.hashIDs = make([]uint64, len(r.hashIDs))
		copy(cp.hashIDs, r.hashIDs)
	}
	return cp
}

// OnEvicted decrements the Recorder's inFlight counter exactly once when a pendingRecord
// is deleted or evicted from PluginState.
func (r *pendingRecord) OnEvicted(_ string, _ plugin.StateKey) {
	if r == nil {
		return
	}
	r.evictOnce.Do(func() {
		if r.onEvict != nil {
			r.onEvict()
		}
	})
}

// Recorder implements the kv-block-trace-recorder EPP plugin.
type Recorder struct {
	typedName                 plugin.TypedName
	cfg                       Config
	secret                    []byte
	keyFingerprint            string
	captureRunID              string
	sampleRateBPS             int
	blockSize                 int
	sessionSources            []sessionaffinity.SessionIDSource
	parentSessionSources      []sessionaffinity.SessionIDSource
	subagentSources           []sessionaffinity.SessionIDSource
	useDefaultSessionFallback bool
	tokenProc                 kvblock.TokenProcessor
	sink                      TraceSink
	clock                     func() time.Time
	pluginState               *plugin.PluginState
	handle                    plugin.Handle

	backendOnce     sync.Once
	resolvedBackend string

	inFlight      atomic.Int64
	recordedCount atomic.Uint64
	droppedCount  atomic.Uint64
}

// Compile-time interface assertions.
var (
	_ plugin.Plugin               = &Recorder{}
	_ plugin.ConsumerPlugin       = &Recorder{}
	_ plugin.StateDumper          = &Recorder{}
	_ fwkrh.DataProducer          = &Recorder{}
	_ fwkrh.RequestAbandoned      = &Recorder{}
	_ fwkrh.PreRequest            = &Recorder{}
	_ fwkrh.ResponseBodyProcessor = &Recorder{}
)

// PluginFactory instantiates Recorder from EPP YAML/JSON plugin configuration.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	var cfg Config
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("%s: failed to decode parameters: %w", PluginType, err)
		}
	}

	var flushInterval time.Duration
	if cfg.FlushInterval != "" {
		parsed, err := time.ParseDuration(cfg.FlushInterval)
		if err != nil {
			return nil, fmt.Errorf("%s: invalid flushInterval %q: %w", PluginType, cfg.FlushInterval, err)
		}
		flushInterval = parsed
	}

	// Validate capture secret before creating directories/goroutines so fail-closed
	// startup leaves zero side effects.
	if _, err := LoadCaptureSecret(cfg.SecretFilePath, cfg.AllowEphemeralSecretForDev); err != nil {
		return nil, fmt.Errorf("%s: %w", PluginType, err)
	}

	if name == "" {
		name = PluginType
	}

	sink, err := NewSealedGzipSink(SealedGzipSinkConfig{
		OutputDir:       cfg.OutputDir,
		StagingDir:      cfg.StagingDir,
		PodName:         cfg.PodName,
		PluginName:      name,
		MaxSegmentBytes: cfg.MaxSegmentBytes,
		MaxRotatedFiles: cfg.MaxRotatedFiles,
		FlushInterval:   flushInterval,
		BufferSize:      cfg.BufferSize,
		Clock:           time.Now,
	})
	if err != nil {
		return nil, fmt.Errorf("%s: %w", PluginType, err)
	}

	ctx := context.Background()
	if handle != nil && handle.Context() != nil {
		ctx = handle.Context()
	}

	rec, err := newRecorderWithHandle(ctx, name, cfg, sink, time.Now, handle)
	if err != nil {
		_ = sink.Close(context.Background())
		return nil, err
	}

	if handle != nil && handle.Context() != nil {
		context.AfterFunc(handle.Context(), func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = rec.Close(shutdownCtx)
		})
	}
	return rec, nil
}

// NewRecorder constructs a Recorder with an injected TraceSink and clock for testing
// or programmatic embedding.
func NewRecorder(cfg Config, sink TraceSink, clock func() time.Time) (*Recorder, error) {
	return newRecorderWithHandle(context.Background(), PluginType, cfg, sink, clock, nil)
}

func newRecorderWithHandle(
	ctx context.Context,
	name string,
	cfg Config,
	sink TraceSink,
	clock func() time.Time,
	handle plugin.Handle,
) (*Recorder, error) {
	if sink == nil {
		return nil, errors.New("trace sink must not be nil")
	}
	if clock == nil {
		clock = time.Now
	}
	if name == "" {
		name = PluginType
	}

	secret, err := LoadCaptureSecret(cfg.SecretFilePath, cfg.AllowEphemeralSecretForDev)
	if err != nil {
		return nil, err
	}

	captureRunID, err := GenerateCaptureRunID()
	if err != nil {
		return nil, err
	}

	sessionSources, usedDefaultSession := cfg.effectiveSessionSources()
	if err := sessionaffinity.ValidateSources(sessionSources); err != nil {
		return nil, fmt.Errorf("invalid sessionSources: %w", err)
	}
	parentSessionSources := cfg.effectiveParentSessionSources()
	if err := sessionaffinity.ValidateSources(parentSessionSources); err != nil {
		return nil, fmt.Errorf("invalid parentSessionSources: %w", err)
	}
	subagentSources := cfg.effectiveSubagentSources()
	if err := sessionaffinity.ValidateSources(subagentSources); err != nil {
		return nil, fmt.Errorf("invalid subagentSources: %w", err)
	}

	blockSize := cfg.effectiveBlockSize()
	tokenProc, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: blockSize,
		HashSeed:        cfg.HashSeed,
		HashAlgorithm:   cfg.HashAlgorithm,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize KV block token processor: %w", err)
	}

	registerRecorderMetrics()
	if handle != nil && handle.Metrics() != nil {
		registerRecorderMetricsOn(handle.Metrics())
	}

	return &Recorder{
		typedName:                 plugin.TypedName{Type: PluginType, Name: name},
		cfg:                       cfg,
		secret:                    secret,
		keyFingerprint:            KeyFingerprint(secret),
		captureRunID:              captureRunID,
		sampleRateBPS:             cfg.effectiveSampleRateBPS(),
		blockSize:                 blockSize,
		sessionSources:            sessionSources,
		parentSessionSources:      parentSessionSources,
		subagentSources:           subagentSources,
		useDefaultSessionFallback: usedDefaultSession,
		tokenProc:                 tokenProc,
		sink:                      sink,
		clock:                     clock,
		pluginState:               plugin.NewPluginState(ctx),
		handle:                    handle,
	}, nil
}

// TypedName returns the plugin type and instance name.
func (r *Recorder) TypedName() plugin.TypedName {
	return r.typedName
}

// KeyFingerprint returns the 8-hex-char SHA-256 fingerprint of the active capture secret.
func (r *Recorder) KeyFingerprint() string {
	return r.keyFingerprint
}

// CaptureRunID returns the 32-hex-char public identifier of this Recorder instance.
func (r *Recorder) CaptureRunID() string {
	return r.captureRunID
}

// InFlightRequests returns the current count of in-flight requests tracked in PluginState.
func (r *Recorder) InFlightRequests() int64 {
	return r.inFlight.Load()
}

// RecordedRecords returns the count of events successfully enqueued into the sink.
func (r *Recorder) RecordedRecords() uint64 {
	return r.recordedCount.Load()
}

// DroppedRecords returns the count of events dropped due to sink backpressure.
func (r *Recorder) DroppedRecords() uint64 {
	return r.droppedCount.Load()
}

// HasPendingState reports whether requestID currently holds transient state in PluginState.
func (r *Recorder) HasPendingState(requestID string) bool {
	_, err := r.pluginState.Read(requestID, plugin.StateKey(r.typedName.String()))
	return err == nil
}

// Close flushes and closes the underlying TraceSink.
func (r *Recorder) Close(ctx context.Context) error {
	return r.sink.Close(ctx)
}

type recorderDebugState struct {
	CaptureRunID    string `json:"captureRunId"`
	KeyFingerprint  string `json:"keyFingerprint"`
	BlockSizeTokens int    `json:"blockSizeTokens"`
	SampleRateBPS   int    `json:"sampleRateBps"`
	ActiveFile      string `json:"activeFile,omitempty"`
	InFlight        int64  `json:"inFlight"`
	RecordsWritten  uint64 `json:"recordsWritten"`
	RecordsDropped  uint64 `json:"recordsDropped"`
	WriteErrors     uint64 `json:"writeErrors"`
	FileRotations   uint64 `json:"fileRotations"`
}

// DumpState implements plugin.StateDumper with operational counters only.
// No HMAC key material, session identifiers, or block digests are exposed.
func (r *Recorder) DumpState() (json.RawMessage, error) {
	state := recorderDebugState{
		CaptureRunID:    r.captureRunID,
		KeyFingerprint:  r.keyFingerprint,
		BlockSizeTokens: r.blockSize,
		SampleRateBPS:   r.sampleRateBPS,
		InFlight:        r.inFlight.Load(),
		RecordsWritten:  r.recordedCount.Load(),
		RecordsDropped:  r.droppedCount.Load(),
	}
	if sealed, ok := r.sink.(*SealedGzipSink); ok {
		state.ActiveFile = sealed.ActiveFile()
		if w := sealed.WrittenRecords(); w > 0 {
			state.RecordsWritten = w
		}
		if d := sealed.Dropped(); d > state.RecordsDropped {
			state.RecordsDropped = d
		}
		state.WriteErrors = sealed.SealErrors()
		state.FileRotations = sealed.Rotations()
	}
	return json.Marshal(state)
}

// Produces declares no downstream data keys; Recorder acts as a terminal DataProducer observer.
func (r *Recorder) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{}
}

// Consumes declares a required dependency on TokenizedRequest (enforcing backend: vllm
// unless allowEstimateTokenizer is true) and optional dependencies on agent-identity,
// session-id-producer, and any attributes referenced by configured session/subagent sources.
func (r *Recorder) Consumes() plugin.DataDependencies {
	var consumedTokenized plugin.ConsumedData
	if !r.cfg.AllowEstimateTokenizer {
		consumedTokenized = plugin.ConsumedData{
			Type:    fwkrequest.TokenizedRequest{},
			Matches: plugin.MatchKeyValues(map[string]string{"backend": tokenproducer.BackendVLLM}),
		}
	} else {
		consumedTokenized = plugin.ConsumedData{
			Type: fwkrequest.TokenizedRequest{},
			Matches: func(attrs map[string]string) error {
				b := attrs["backend"]
				if b != tokenproducer.BackendVLLM && b != tokenproducer.BackendEstimate {
					return fmt.Errorf("unsupported token-producer backend %q", b)
				}
				return nil
			},
		}
	}

	optional := map[plugin.DataKey]any{
		agentidentity.AgentIdentityKey: "",
		attrsession.SessionIDDataKey.WithNonEmptyProducerName(sessionidconstants.SessionIDProducerType): attrsession.SessionID(""),
	}
	mergeOptional := func(deps plugin.DataDependencies) {
		for k, v := range deps.Optional {
			if _, exists := optional[k]; !exists {
				optional[k] = v
			}
		}
	}
	mergeOptional(sessionaffinity.SourcesConsumes(r.sessionSources))
	mergeOptional(sessionaffinity.SourcesConsumes(r.parentSessionSources))
	mergeOptional(sessionaffinity.SourcesConsumes(r.subagentSources))

	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			tokenproducer.TokenizedPromptDataKey: consumedTokenized,
		},
		Optional: optional,
	}
}

// Produce resolves session lineage, applies stateless cohort sampling, computes anonymized
// full-block KV hashes, and stashes transient state in PluginState[request.RequestID].
func (r *Recorder) Produce(ctx context.Context, request *scheduling.InferenceRequest, _ []scheduling.Endpoint) error {
	if request == nil || request.RequestID == "" || request.Body == nil || request.Body.RenderRequest {
		return nil
	}

	backend := r.detectTokenizerBackend()
	if backend != tokenproducer.BackendVLLM && !r.cfg.AllowEstimateTokenizer {
		return fmt.Errorf("%s requires token-producer backend %q (got %q); set allowEstimateTokenizer=true to permit byte-estimated hashes",
			PluginType, tokenproducer.BackendVLLM, backend)
	}

	rawSessionKey, rawParentSessionID, sessionSource, rawAgentID, rawSubagentType := r.resolveSessionAndSubagent(ctx, request)

	// Key sampling on root session when present so a sampled agent trace retains all subagents.
	sampleKey := rootSessionPrefix(rawParentSessionID)
	if sampleKey == "" {
		sampleKey = rootSessionPrefix(rawSessionKey)
	}
	if sampleKey == "" {
		sampleKey = "req:" + request.RequestID
	}
	if !ShouldSample(r.secret, sampleKey, r.sampleRateBPS) {
		skippedRequestsMetric.WithLabelValues(r.typedName.Name, skipReasonSampledOut).Inc()
		return nil
	}

	tokReq := request.Body.TokenizedRequest
	inTokens := 0
	var rawSalt string
	if tokReq != nil {
		inTokens = tokReq.TokenCount()
		rawSalt = tokReq.CacheSalt
	}
	if rawSalt == "" {
		rawSalt = tokenproducer.CacheSaltFromBody(request.Body)
	}

	model := request.TargetModel
	if model == "" {
		model = request.Body.Model
	}

	hashIDs, err := r.computeAnonymizedFullBlockHashes(ctx, request, backend, model, rawSalt, inTokens)
	if err != nil {
		return err
	}

	msgCount, hasSystemPrompt, hasTools := extractStructuralHints(request.Body)
	maxOut := extractMaxOutputTokens(request.Body)

	rec := &pendingRecord{
		sessionID:        AnonymizeSessionID(r.secret, rawSessionKey),
		parentSessionID:  AnonymizeSessionID(r.secret, rawParentSessionID),
		sessionSource:    sessionSource,
		cacheSaltID:      AnonymizeCacheSalt(r.secret, rawSalt),
		agentID:          AnonymizeAgentID(r.secret, rawAgentID),
		subagentType:     AnonymizeSubagentType(r.secret, rawSubagentType),
		model:            model,
		tokenizerBackend: backend,
		blockSize:        r.blockSize,
		inTokens:         inTokens,
		maxOutputTokens:  maxOut,
		msgCount:         msgCount,
		hasSystemPrompt:  hasSystemPrompt,
		hasTools:         hasTools,
		streaming:        request.Body.Stream,
		producedAt:       r.clock(),
		hashIDs:          hashIDs,
		onEvict: func() {
			r.inFlight.Add(-1)
		},
	}
	stateKey := plugin.StateKey(r.typedName.String())
	if _, existed := r.pluginState.ReadOrWrite(request.RequestID, stateKey, rec); !existed {
		r.inFlight.Add(1)
	} else {
		r.pluginState.Write(request.RequestID, stateKey, rec)
	}
	r.pluginState.BindLiveness(ctx, request.RequestID)
	return nil
}

// RequestAbandoned clears PluginState when admission control rejects a request prior to PreRequest.
func (r *Recorder) RequestAbandoned(_ context.Context, request *scheduling.InferenceRequest) {
	if request == nil || request.RequestID == "" {
		return
	}
	r.pluginState.Delete(request.RequestID)
}

// PreRequest marks the request dispatched and optionally injects stream_options.include_usage=true.
// It always returns nil so trace capture never disrupts live traffic.
func (r *Recorder) PreRequest(_ context.Context, request *scheduling.InferenceRequest, _ *scheduling.SchedulingResult) error {
	if request == nil || request.RequestID == "" {
		return nil
	}
	stateData, err := r.pluginState.Read(request.RequestID, plugin.StateKey(r.typedName.String()))
	if err == nil {
		if rec, ok := stateData.(*pendingRecord); ok {
			rec.dispatched = true
		}
	}

	if r.cfg.InjectStreamIncludeUsage && request.Body != nil && request.Body.Stream {
		request.Body.MutatePayloadMap(func(m fwkrequest.PayloadMap) {
			if existing, ok := m[reqcommon.FieldStreamOptions].(map[string]any); ok {
				existing["include_usage"] = true
			} else {
				m[reqcommon.FieldStreamOptions] = map[string]any{"include_usage": true}
			}
		})
	}
	return nil
}

// ResponseBody finalizes and enqueues the anonymized Event when response.EndOfStream is true,
// unconditionally deleting PluginState[request.RequestID].
func (r *Recorder) ResponseBody(
	_ context.Context,
	request *scheduling.InferenceRequest,
	response *fwkrh.Response,
	_ *fwkdl.EndpointMetadata,
) {
	if response == nil || !response.EndOfStream || request == nil || request.RequestID == "" {
		return
	}
	defer r.pluginState.Delete(request.RequestID)

	stateData, err := r.pluginState.Read(request.RequestID, plugin.StateKey(r.typedName.String()))
	if err != nil {
		return
	}
	rec, ok := stateData.(*pendingRecord)
	if !ok || rec == nil {
		return
	}

	now := r.clock()
	recvTS := response.ReceivedTimestamp
	if recvTS.IsZero() {
		recvTS = rec.producedAt
	}
	completeTS := response.CompleteTimestamp
	if completeTS.IsZero() {
		completeTS = now
	}

	tRecvUnix := float64(recvTS.UnixNano()) / 1e9
	var ttftS float64
	if !response.FirstTokenTimestamp.IsZero() && !response.FirstTokenTimestamp.Before(recvTS) {
		ttftS = response.FirstTokenTimestamp.Sub(recvTS).Seconds()
	}
	var apiTimeS float64
	if !completeTS.Before(recvTS) {
		apiTimeS = completeTS.Sub(recvTS).Seconds()
	}

	statusCode := parseStatusCode(response.Headers)
	termination := resolveTermination(response.TerminationCause, statusCode)

	inTokens := rec.inTokens
	hashIDs := rec.hashIDs
	if rec.tokenizerBackend == tokenproducer.BackendEstimate && response.Usage.PromptTokens > 0 {
		inTokens = response.Usage.PromptTokens
		maxFullBlocks := inTokens / r.blockSize
		if len(hashIDs) > maxFullBlocks {
			hashIDs = hashIDs[:maxFullBlocks]
		}
	} else if inTokens == 0 && response.Usage.PromptTokens > 0 {
		inTokens = response.Usage.PromptTokens
	}
	if hashIDs == nil {
		hashIDs = []uint64{}
	}

	outTokens, outSource := resolveOutputTokens(response)

	usageIn := max(0, response.Usage.PromptTokens)
	var cachedTokens int
	if response.Usage.PromptTokenDetails != nil {
		cachedTokens = max(0, response.Usage.PromptTokenDetails.CachedTokens)
	}

	ev := Event{
		Version:          SchemaVersion,
		KeyFingerprint:   r.keyFingerprint,
		CaptureRunID:     r.captureRunID,
		SessionID:        rec.sessionID,
		ParentSessionID:  rec.parentSessionID,
		SessionSource:    rec.sessionSource,
		CacheSaltID:      rec.cacheSaltID,
		AgentID:          rec.agentID,
		SubagentType:     rec.subagentType,
		TRecvUnix:        tRecvUnix,
		TTFTS:            ttftS,
		APITimeS:         apiTimeS,
		Streaming:        rec.streaming || response.Streaming,
		StatusCode:       statusCode,
		Termination:      termination,
		Model:            rec.model,
		TokenizerBackend: rec.tokenizerBackend,
		BlockSize:        rec.blockSize,
		In:               inTokens,
		UsageIn:          usageIn,
		CachedTokens:     cachedTokens,
		Out:              outTokens,
		OutSource:        outSource,
		MaxOutputTokens:  rec.maxOutputTokens,
		MsgCount:         rec.msgCount,
		HasSystemPrompt:  rec.hasSystemPrompt,
		HasTools:         rec.hasTools,
		HashIDs:          hashIDs,
	}

	if !r.sink.TryEnqueue(ev) {
		r.droppedCount.Add(1)
		droppedEventsMetric.WithLabelValues(r.typedName.Name).Inc()
		if _, isSealed := r.sink.(*SealedGzipSink); !isSealed {
			droppedRecordsMetric.WithLabelValues(r.typedName.Name, dropReasonBufferFull).Inc()
		}
		return
	}
	if _, isSealed := r.sink.(*SealedGzipSink); !isSealed {
		writtenEventsMetric.WithLabelValues(r.typedName.Name).Inc()
	}
	r.recordedCount.Add(1)
}

func (r *Recorder) detectTokenizerBackend() string {
	r.backendOnce.Do(func() {
		if r.cfg.TokenizerBackendOverride != "" {
			r.resolvedBackend = r.cfg.TokenizerBackendOverride
			return
		}
		if r.handle != nil {
			for _, p := range r.handle.GetAllPlugins() {
				if tp, ok := p.(*tokenproducer.Plugin); ok {
					r.resolvedBackend = tp.Backend()
					return
				}
			}
		}
		r.resolvedBackend = tokenproducer.BackendVLLM
	})
	return r.resolvedBackend
}

func (r *Recorder) computeAnonymizedFullBlockHashes(
	ctx context.Context,
	request *scheduling.InferenceRequest,
	backend string,
	model string,
	cacheSalt string,
	inTokens int,
) ([]uint64, error) {
	tokReq := request.Body.TokenizedRequest
	if tokReq == nil || len(tokReq.Prompts) == 0 || inTokens < r.blockSize {
		return []uint64{}, nil
	}

	if backend == tokenproducer.BackendEstimate {
		promptHashes, promptTokenCounts := prefixhash.GetBlockHashesWithPromptTokens(ctx, request, r.blockSize, math.MaxInt)
		if len(promptHashes) == 0 {
			return []uint64{}, nil
		}
		out := make([]uint64, 0, inTokens/r.blockSize)
		for i, hashes := range promptHashes {
			pTokens := 0
			if i < len(promptTokenCounts) {
				pTokens = promptTokenCounts[i]
			}
			fullBlocks := min(len(hashes), pTokens/r.blockSize)
			for _, h := range hashes[:fullBlocks] {
				out = append(out, AnonymizeBlockHash(r.secret, uint64(h)))
			}
		}
		return out, nil
	}

	// Default vLLM tokenized path: hash full blocks only via kvblock.TokenProcessor,
	// folding CacheSalt into block 0 and preserving multimodal extra features.
	out := make([]uint64, 0, inTokens/r.blockSize)
	for _, prompt := range tokReq.Prompts {
		if len(prompt.TokenIDs) < r.blockSize {
			continue
		}
		numBlocks := len(prompt.TokenIDs) / r.blockSize
		var extraFeatures []*kvblock.BlockExtraFeatures
		if len(prompt.MultiModalFeatures) > 0 {
			mmHashes, mmPlaceholders := tokenproducer.ConvertMMFeaturesFromUpstream(prompt.MultiModalFeatures)
			extraFeatures = kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, r.blockSize, len(prompt.TokenIDs))
		}
		extraFeatures = foldFirstBlockCacheSalt(extraFeatures, cacheSalt, numBlocks)

		keys, err := r.tokenProc.TokensToKVBlockKeys(kvblock.EmptyBlockHash, prompt.TokenIDs, model, extraFeatures)
		if err != nil {
			return nil, fmt.Errorf("failed to compute KV block keys: %w", err)
		}
		for _, k := range keys {
			out = append(out, AnonymizeBlockHash(r.secret, uint64(k)))
		}
	}
	return out, nil
}

func foldFirstBlockCacheSalt(extraFeatures []*kvblock.BlockExtraFeatures, salt string, numBlocks int) []*kvblock.BlockExtraFeatures {
	if salt == "" || numBlocks <= 0 {
		return extraFeatures
	}
	if extraFeatures == nil {
		extraFeatures = make([]*kvblock.BlockExtraFeatures, numBlocks)
	}
	if extraFeatures[0] == nil {
		extraFeatures[0] = &kvblock.BlockExtraFeatures{}
	}
	extraFeatures[0].MMHashes = append(extraFeatures[0].MMHashes, kvblock.MMHash{Hash: salt})
	return extraFeatures
}

func (r *Recorder) resolveSessionAndSubagent(
	ctx context.Context,
	request *scheduling.InferenceRequest,
) (rawSessionKey, rawParentSessionID, sessionSource, rawAgentID, rawSubagentType string) {
	rawSubagentType = firstHeaderValue(request.Headers, explicitSubagentTypeHeaders)
	rawAgentID = stripStreamIndexSuffix(resolveFromSources(ctx, request, r.subagentSources))
	rawParentSessionID = resolveFromSources(ctx, request, r.parentSessionSources)

	// Priority 1: "header" — configured/default SessionSources (agent-identity attribute,
	// session-id-producer attribute, explicit session/fairness headers) or non-default FairnessID.
	rawSessionKey = resolveFromSources(ctx, request, r.sessionSources)
	if rawSessionKey == "" && r.useDefaultSessionFallback {
		if fid := strings.TrimSpace(request.FairnessID); fid != "" && fid != reqcommon.DefaultFairnessID {
			rawSessionKey = fid
		}
	}
	if rawSessionKey != "" {
		// Support Weka/inference-perf "<parent>::sa:<agent_id>[:s<stream_idx>]" child stream suffixes
		// so parent and subagent requests share cohort sampling and parent session_id, and parallel
		// streams of one subagent coalesce via stripStreamIndexSuffix.
		if parent, agentBase := splitCompositeSubagentSessionID(rawSessionKey); parent != "" {
			rawSessionKey = rootSessionPrefix(parent)
			if rawParentSessionID == "" {
				rawParentSessionID = parent
			}
			if rawAgentID == "" && agentBase != "" {
				rawAgentID = agentBase
			}
		}
		return rawSessionKey, rawParentSessionID, SessionSourceHeader, rawAgentID, rawSubagentType
	}

	// Priority 2: "prefix_fallback" — digest of CacheSalt + system prompt + Turn-0 user message.
	var prefixBuf bytes.Buffer
	if request.Body != nil && request.Body.WriteConversationPrefix(&prefixBuf) {
		sum := sha256.Sum256(prefixBuf.Bytes())
		return "pf:" + hex.EncodeToString(sum[:]), rawParentSessionID, SessionSourcePrefixFallback, rawAgentID, rawSubagentType
	}

	// Priority 3: "request_id" — single-turn fallback.
	return "req:" + request.RequestID, rawParentSessionID, SessionSourceRequestID, rawAgentID, rawSubagentType
}

func resolveFromSources(
	ctx context.Context,
	request *scheduling.InferenceRequest,
	sources []sessionaffinity.SessionIDSource,
) string {
	if request == nil || len(sources) == 0 {
		return ""
	}
	for _, src := range sources {
		if src.Header != "" {
			if val := firstHeaderValue(request.Headers, []string{src.Header}); val != "" {
				return val
			}
			continue
		}
		if val := strings.TrimSpace(sessionaffinity.ResolveSessionID(ctx, request, []sessionaffinity.SessionIDSource{src})); val != "" {
			return val
		}
	}
	return ""
}

// splitCompositeSubagentSessionID extracts the immediate parent session ID and
// normalized subagent identifier from "<parent>::sa:<agent_id>[:s<stream_index>]"
// composite session identifiers emitted by inference-perf WEKA trace replay.
func splitCompositeSubagentSessionID(sessionID string) (parentID, agentBase string) {
	idx := strings.LastIndex(sessionID, subagentSessionDelimiter)
	if idx <= 0 {
		return "", ""
	}
	parent := sessionID[:idx]
	tail := sessionID[idx+len(subagentSessionDelimiter):]
	if tail == "" {
		return "", ""
	}
	base := stripStreamIndexSuffix(tail)
	if base == "" {
		return "", ""
	}
	return parent, base
}

func stripStreamIndexSuffix(s string) string {
	idx := strings.LastIndex(s, ":s")
	if idx <= 0 || idx+2 >= len(s) {
		return s
	}
	for i := idx + 2; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return s
		}
	}
	return s[:idx]
}

func rootSessionPrefix(sessionID string) string {
	if idx := strings.Index(sessionID, subagentSessionDelimiter); idx > 0 {
		return sessionID[:idx]
	}
	return sessionID
}

func extractMaxOutputTokens(body *fwkrequest.InferenceRequestBody) *int64 {
	if body == nil {
		return nil
	}
	if body.MaxOutputTokens != nil {
		v := *body.MaxOutputTokens
		return &v
	}
	if body.Payload != nil {
		if pm, ok := body.Payload.AsMap(); ok {
			for _, key := range []string{"max_completion_tokens", "max_tokens", "max_output_tokens"} {
				switch raw := pm[key].(type) {
				case int:
					v := int64(raw)
					return &v
				case int64:
					v := raw
					return &v
				case float64:
					v := int64(raw)
					return &v
				case json.Number:
					if parsed, err := raw.Int64(); err == nil {
						return &parsed
					}
				}
			}
		}
	}
	return nil
}

func firstHeaderValue(headers map[string]string, keys []string) string {
	if len(headers) == 0 {
		return ""
	}
	for _, k := range keys {
		if v := strings.TrimSpace(headers[k]); v != "" {
			return v
		}
		// Fallback case-insensitive scan when callers construct InferenceRequest.Headers directly in tests.
		for hk, hv := range headers {
			if strings.EqualFold(hk, k) {
				if trimmed := strings.TrimSpace(hv); trimmed != "" {
					return trimmed
				}
			}
		}
	}
	return ""
}

func extractStructuralHints(body *fwkrequest.InferenceRequestBody) (msgCount int, hasSystemPrompt bool, hasTools bool) {
	if body == nil {
		return 0, false, false
	}
	switch {
	case body.ChatCompletions != nil:
		cc := body.ChatCompletions
		msgCount = len(cc.Messages)
		hasTools = len(cc.Tools) > 0
		for _, m := range cc.Messages {
			if m.Role == reqcommon.RoleSystem || m.Role == "developer" {
				hasSystemPrompt = true
				break
			}
		}
	case body.Messages != nil:
		mr := body.Messages
		msgCount = len(mr.Messages)
		hasTools = len(mr.Tools) > 0
		hasSystemPrompt = mr.System.Raw != "" || len(mr.System.Structured) > 0
	case body.Responses != nil:
		rr := body.Responses
		switch inst := rr.Instructions.(type) {
		case string:
			hasSystemPrompt = inst != ""
		case nil:
		default:
			hasSystemPrompt = true
		}
		switch tools := rr.Tools.(type) {
		case []any:
			hasTools = len(tools) > 0
		case nil:
		default:
			hasTools = true
		}
		switch in := rr.Input.(type) {
		case string:
			if in != "" {
				msgCount = 1
			}
		case []any:
			msgCount = len(in)
			for _, item := range in {
				if m, ok := item.(map[string]any); ok {
					if role, _ := m[reqcommon.FieldRole].(string); role == reqcommon.RoleSystem || role == "developer" {
						hasSystemPrompt = true
					}
				}
			}
		}
	case body.Conversations != nil:
		msgCount = len(body.Conversations.Items)
		for _, item := range body.Conversations.Items {
			if item.Role == reqcommon.RoleSystem || item.Role == "developer" {
				hasSystemPrompt = true
				break
			}
		}
	case body.Completions != nil:
		if !body.Completions.Prompt.IsEmpty() {
			msgCount = 1
		}
	}
	return msgCount, hasSystemPrompt, hasTools
}

func parseStatusCode(headers map[string]string) int {
	if len(headers) == 0 {
		return 200
	}
	for _, key := range []string{reqcommon.PseudoHeaderStatus, reqcommon.HeaderStatus} {
		if raw := strings.TrimSpace(headers[key]); raw != "" {
			if code, err := strconv.Atoi(raw); err == nil && code > 0 {
				return code
			}
		}
	}
	return 200
}

func resolveTermination(cause fwkrh.TerminationCause, statusCode int) string {
	if cause != "" && cause != fwkrh.TerminationCauseNatural {
		return string(cause)
	}
	if statusCode >= 400 {
		return string(fwkrh.TerminationCauseError)
	}
	return TerminationNatural
}

func resolveOutputTokens(response *fwkrh.Response) (int, string) {
	if response.Usage.CompletionTokens > 0 || response.Usage.TotalTokens > 0 {
		return max(0, response.Usage.CompletionTokens), OutSourceUsage
	}
	return max(0, response.StreamedEvents), OutSourceEvents
}
