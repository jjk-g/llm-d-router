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
	"cmp"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

const (
	// HashIDScopeLocal remaps block digests to 1-based integers independently per trace.
	HashIDScopeLocal = "local"
	// HashIDScopeGlobal remaps block digests to 1-based integers across all assembled traces.
	HashIDScopeGlobal = "global"

	// DefaultLCPOverlapThreshold is the default block-prefix overlap fraction
	// used to chain sessionless requests into reconstructed conversations.
	DefaultLCPOverlapThreshold = 0.90

	// DefaultMinSystemCohort is the minimum number of traces sharing a (model, cache_salt_id)
	// cohort required to infer shared system_tokens from Turn-0 block prefixes.
	DefaultMinSystemCohort = 3

	// DefaultSubagentLookahead is the default number of subsequent turns inspected
	// within a session to detect untagged synchronous subagent branches.
	DefaultSubagentLookahead = 8

	// ClassifierMaxOutputTokens is the max_output_tokens cap below or at which
	// tool-free auxiliary requests are classified as lightweight utility calls.
	ClassifierMaxOutputTokens = 64

	outcomeCompleted = "completed"
	outcomeEvicted   = "evicted"
	outcomeCancelled = "cancelled"
	outcomeFailed    = "failed"

	streamTypeStreaming = "s"
	streamTypeNormal    = "n"
	entryTypeSubagent   = "subagent"
	maxScannerLine      = 32 << 20 // 32 MiB max JSONL line for ultra-long-context prompts
	exportWriterBufSize = 64 * 1024
)

// CapturedRecord represents a unified per-request trace record supporting both
// kv-block-trace-recorder Event JSONL (.jsonl / .jsonl.gz) and trace-capture-producer
// CapturedRecord JSONL schemas.
type CapturedRecord struct {
	SchemaVersion     string   `json:"schema_version,omitempty"`
	Version           int      `json:"v,omitempty"`
	KeyFingerprint    string   `json:"key_fingerprint,omitempty"`
	CaptureRunID      string   `json:"capture_run_id,omitempty"`
	RequestID         string   `json:"request_id,omitempty"`
	SessionID         string   `json:"session_id,omitempty"`
	ParentSessionID   string   `json:"parent_session_id,omitempty"`
	SessionSource     string   `json:"session_source,omitempty"`
	CacheSaltID       string   `json:"cache_salt_id,omitempty"`
	SubagentID        string   `json:"subagent_id,omitempty"`
	AgentID           string   `json:"agent_id,omitempty"`
	SubagentType      string   `json:"subagent_type,omitempty"`
	ArrivalUnixNano   int64    `json:"arrival_unix_ns,omitempty"`
	TRecvUnix         float64  `json:"t_recv_unix,omitempty"`
	FirstTokenSeconds float64  `json:"ttft_s,omitempty"`
	APITimeSeconds    float64  `json:"api_time_s,omitempty"`
	StreamType        string   `json:"type,omitempty"`
	Streaming         bool     `json:"streaming,omitempty"`
	StatusCode        int      `json:"status_code,omitempty"`
	Termination       string   `json:"termination,omitempty"`
	Model             string   `json:"model,omitempty"`
	TokenizerBackend  string   `json:"tokenizer_backend,omitempty"`
	BlockSize         int      `json:"block_size"`
	TokenizedIn       int      `json:"tokenized_in,omitempty"`
	In                int      `json:"in,omitempty"`
	UsageIn           int      `json:"usage_in,omitempty"`
	CachedTokens      int      `json:"cached_tokens,omitempty"`
	OutputTokens      int      `json:"out"`
	OutSource         string   `json:"out_source,omitempty"`
	MaxOutputTokens   *int64   `json:"max_output_tokens,omitempty"`
	MsgCount          int      `json:"msg_count,omitempty"`
	HasSystemPrompt   bool     `json:"has_system_prompt,omitempty"`
	HasTools          bool     `json:"has_tools,omitempty"`
	Outcome           string   `json:"outcome,omitempty"`
	BlockHashes       []string `json:"block_hashes,omitempty"`
	HashIDs           []uint64 `json:"hash_ids,omitempty"`
}

// normalize populates canonical fields from either Event or CapturedRecord representations.
func (r *CapturedRecord) normalize() {
	if r.ArrivalUnixNano == 0 && r.TRecvUnix > 0 {
		r.ArrivalUnixNano = int64(math.Round(r.TRecvUnix * 1e9))
	}
	if r.TokenizedIn == 0 && r.In > 0 {
		r.TokenizedIn = r.In
	}
	if r.SubagentID == "" && r.AgentID != "" {
		r.SubagentID = r.AgentID
	}
	if r.StreamType == "" {
		if r.Streaming {
			r.StreamType = streamTypeStreaming
		} else {
			r.StreamType = streamTypeNormal
		}
	}
	if len(r.BlockHashes) == 0 && len(r.HashIDs) > 0 {
		hashes := make([]string, len(r.HashIDs))
		for i, id := range r.HashIDs {
			hashes[i] = strconv.FormatUint(id, 10)
		}
		r.BlockHashes = hashes
	}
	if r.SessionSource == SessionSourceRequestID {
		// Treat request_id fallback sessions as sessionless so LCP chaining can reconstruct multi-turn traces.
		r.SessionID = ""
	}
}

// WekaTrace is a single multi-turn session in the WEKA agentic trace format,
// compatible with both aiperf 0.13.0 and inference-perf 0.7.0 WEKA trace replay engines.
type WekaTrace struct {
	ID           string      `json:"id"`
	Models       []string    `json:"models"`
	BlockSize    int         `json:"block_size"`
	HashIDScope  string      `json:"hash_id_scope"`
	ToolTokens   int         `json:"tool_tokens"`
	SystemTokens int         `json:"system_tokens"`
	Requests     []WekaEntry `json:"requests"`
	Totals       *WekaTotals `json:"totals,omitempty"`
}

// WekaEntry is one discriminated element ("n", "s", or "subagent") in WekaTrace.Requests.
type WekaEntry struct {
	Turn     *WekaTurnRequest
	Subagent *WekaSubagentEntry
}

// MarshalJSON serializes the underlying Turn or Subagent variant directly.
func (e WekaEntry) MarshalJSON() ([]byte, error) {
	switch {
	case e.Subagent != nil:
		return json.Marshal(e.Subagent)
	case e.Turn != nil:
		return json.Marshal(e.Turn)
	default:
		return nil, errors.New("empty WekaEntry")
	}
}

// UnmarshalJSON decodes a WekaEntry based on its "type" discriminator.
func (e *WekaEntry) UnmarshalJSON(data []byte) error {
	var disc struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &disc); err != nil {
		return err
	}
	if disc.Type == entryTypeSubagent {
		var sub WekaSubagentEntry
		if err := json.Unmarshal(data, &sub); err != nil {
			return err
		}
		e.Subagent = &sub
		e.Turn = nil
		return nil
	}
	var turn WekaTurnRequest
	if err := json.Unmarshal(data, &turn); err != nil {
		return err
	}
	e.Turn = &turn
	e.Subagent = nil
	return nil
}

// WekaTurnRequest represents a normal ("n") or streaming ("s") LLM turn.
type WekaTurnRequest struct {
	T         float64  `json:"t"`
	Type      string   `json:"type"`
	Model     string   `json:"model"`
	In        int      `json:"in"`
	Out       int      `json:"out"`
	HashIDs   []int    `json:"hash_ids"`
	APITime   *float64 `json:"api_time,omitempty"`
	ThinkTime *float64 `json:"think_time,omitempty"`
	TTFT      *float64 `json:"ttft,omitempty"`
}

// WekaSubagentEntry represents a nested subagent span within a parent trace.
// Inner Requests use Type "n" and include leaf ThinkTime + root-elapsed T >= Subagent.T
// to satisfy both aiperf 0.13.0 and inference-perf 0.7.0 validation.
type WekaSubagentEntry struct {
	T            float64           `json:"t"`
	Type         string            `json:"type"`
	AgentID      string            `json:"agent_id"`
	SubagentType string            `json:"subagent_type"`
	Status       string            `json:"status"`
	DurationMs   *int              `json:"duration_ms,omitempty"`
	TotalTokens  *int              `json:"total_tokens,omitempty"`
	Models       []string          `json:"models"`
	Requests     []WekaTurnRequest `json:"requests"`
	ToolTokens   int               `json:"tool_tokens"`
	SystemTokens int               `json:"system_tokens"`
}

// WekaTokenPair holds aggregated input/output token counts.
type WekaTokenPair struct {
	Input  int `json:"input"`
	Output int `json:"output"`
}

// WekaTotals summarizes token usage across parent and subagent turns.
type WekaTotals struct {
	ParentTokens   WekaTokenPair `json:"parent_tokens"`
	SubagentTokens WekaTokenPair `json:"subagent_tokens"`
	CombinedTokens WekaTokenPair `json:"combined_tokens"`
	SubagentCount  int           `json:"subagent_count"`
}

// AssembleOptions configures offline conversion of CapturedRecord slices into WekaTrace entries.
type AssembleOptions struct {
	// TargetBlockSize downsamples captured block digests to an integer multiple of the captured block size.
	// Zero retains the captured block size.
	TargetBlockSize int
	// HashIDScope selects "local" (default, 1-based per trace) or "global" (1-based across dataset).
	HashIDScope string
	// LCPOverlapThreshold sets the minimum block-prefix overlap ratio in (0, 1] for chaining
	// sessionless requests into conversations. Zero defaults to 0.90.
	LCPOverlapThreshold float64
	// MaxIdleGapSeconds splits sessions or limits LCP recovery across idle gaps when > 0.
	MaxIdleGapSeconds float64
	// DropClassifierCalls filters out requests with max_output_tokens <= 64 and has_tools == false.
	DropClassifierCalls bool
	// IncludeNonCompleted retains non-completed records instead of filtering them out.
	IncludeNonCompleted bool
	// MinSystemCohort is the minimum cohort size to infer system_tokens across traces (0 defaults to 3; <0 disables).
	MinSystemCohort int
	// SubagentLookahead is the lookahead window for untagged synchronous subagent detection (0 defaults to 8; <0 disables).
	SubagentLookahead int
}

type preparedRecord struct {
	rec    CapturedRecord
	hashes []string
}

type subagentGroup struct {
	agentID      string
	subagentType string
	records      []preparedRecord
	startNs      int64
	endNs        int64
}

type sessionBucket struct {
	key         string
	cacheSaltID string
	parent      []preparedRecord
	subagents   map[string]*subagentGroup
	subOrder    []string
	startNs     int64
}

// AssembleWekaTraces groups per-request CapturedRecord entries into multi-turn
// WEKA traces sorted chronologically by trace start time.
func AssembleWekaTraces(records []CapturedRecord, opts AssembleOptions) ([]WekaTrace, error) {
	scope := opts.HashIDScope
	if scope == "" {
		scope = HashIDScopeLocal
	}
	if scope != HashIDScopeLocal && scope != HashIDScopeGlobal {
		return nil, fmt.Errorf("invalid hash_id_scope %q: must be %q or %q", scope, HashIDScopeLocal, HashIDScopeGlobal)
	}
	lcpThreshold := opts.LCPOverlapThreshold
	if lcpThreshold == 0 {
		lcpThreshold = DefaultLCPOverlapThreshold
	}
	if lcpThreshold <= 0 || lcpThreshold > 1 || math.IsNaN(lcpThreshold) {
		return nil, fmt.Errorf("lcpOverlapThreshold must be in (0, 1], got %v", lcpThreshold)
	}
	minCohort := opts.MinSystemCohort
	if minCohort == 0 {
		minCohort = DefaultMinSystemCohort
	}
	lookahead := opts.SubagentLookahead
	if lookahead == 0 {
		lookahead = DefaultSubagentLookahead
	}

	filtered, err := filterAndSortRecords(records, opts)
	if err != nil {
		return nil, err
	}
	if len(filtered) == 0 {
		return nil, nil
	}

	effectiveBlockSize := opts.TargetBlockSize
	if effectiveBlockSize == 0 {
		effectiveBlockSize = filtered[0].BlockSize
	}
	if effectiveBlockSize <= 0 {
		return nil, fmt.Errorf("invalid block_size %d", effectiveBlockSize)
	}

	prepared := make([]preparedRecord, 0, len(filtered))
	for _, r := range filtered {
		if r.BlockSize <= 0 {
			return nil, fmt.Errorf("record %q has invalid block_size %d", r.RequestID, r.BlockSize)
		}
		hashes, dsErr := DownsampleBlockHashes(r.BlockHashes, r.BlockSize, effectiveBlockSize)
		if dsErr != nil {
			return nil, fmt.Errorf("record %q: %w", r.RequestID, dsErr)
		}
		prepared = append(prepared, preparedRecord{rec: r, hashes: hashes})
	}

	buckets := groupIntoSessionBuckets(prepared, lcpThreshold, opts.MaxIdleGapSeconds, lookahead)
	slices.SortFunc(buckets, func(a, b *sessionBucket) int {
		if c := cmp.Compare(a.startNs, b.startNs); c != 0 {
			return c
		}
		return cmp.Compare(a.key, b.key)
	})

	sysBlocksByBucket := inferCohortSystemBlocks(buckets, minCohort)

	globalMap := make(map[string]int)
	traces := make([]WekaTrace, 0, len(buckets))
	for idx, b := range buckets {
		idMap := globalMap
		if scope == HashIDScopeLocal {
			idMap = make(map[string]int)
		}
		sysTokens := sysBlocksByBucket[idx] * effectiveBlockSize
		traces = append(traces, buildWekaTrace(idx, b, effectiveBlockSize, scope, sysTokens, idMap))
	}
	return traces, nil
}

func filterAndSortRecords(records []CapturedRecord, opts AssembleOptions) ([]CapturedRecord, error) {
	seenReq := make(map[string]struct{}, len(records))
	var activeFingerprint string
	out := make([]CapturedRecord, 0, len(records))
	for _, r := range records {
		r.normalize()
		if r.KeyFingerprint != "" {
			if activeFingerprint == "" {
				activeFingerprint = r.KeyFingerprint
			} else if r.KeyFingerprint != activeFingerprint {
				return nil, fmt.Errorf("key_fingerprint mismatch across segments: %q vs %q", activeFingerprint, r.KeyFingerprint)
			}
		}
		if r.RequestID != "" {
			if _, dup := seenReq[r.RequestID]; dup {
				continue
			}
			seenReq[r.RequestID] = struct{}{}
		}
		if !opts.IncludeNonCompleted {
			if r.Outcome != "" && r.Outcome != outcomeCompleted {
				continue
			}
			if r.Outcome == "" && (r.StatusCode >= 400 || r.Termination == "error" || r.Termination == outcomeFailed || r.Termination == outcomeEvicted) {
				continue
			}
		}
		if opts.DropClassifierCalls && r.MaxOutputTokens != nil && *r.MaxOutputTokens <= ClassifierMaxOutputTokens && !r.HasTools {
			continue
		}
		out = append(out, r)
	}
	slices.SortFunc(out, func(a, b CapturedRecord) int {
		if c := cmp.Compare(a.ArrivalUnixNano, b.ArrivalUnixNano); c != 0 {
			return c
		}
		return cmp.Compare(a.RequestID, b.RequestID)
	})
	return out, nil
}

type lcpCandidate struct {
	bucket     *sessionBucket
	lastHashes []string
	lastEndNs  int64
}

func groupIntoSessionBuckets(records []preparedRecord, lcpThreshold, maxIdleGapSec float64, lookahead int) []*sessionBucket {
	bySession := make(map[string]*sessionBucket)
	var buckets []*sessionBucket
	var lcpActive []*lcpCandidate

	getOrCreate := func(key string, arrivalNs int64, saltID string) *sessionBucket {
		if b, ok := bySession[key]; ok {
			if arrivalNs < b.startNs {
				b.startNs = arrivalNs
			}
			if b.cacheSaltID == "" && saltID != "" {
				b.cacheSaltID = saltID
			}
			return b
		}
		b := &sessionBucket{
			key:         key,
			cacheSaltID: saltID,
			subagents:   make(map[string]*subagentGroup),
			startNs:     arrivalNs,
		}
		bySession[key] = b
		buckets = append(buckets, b)
		return b
	}

	parentOf := make(map[string]string)
	for _, pr := range records {
		r := pr.rec
		if r.SessionID != "" && r.ParentSessionID != "" && r.ParentSessionID != r.SessionID {
			if _, exists := parentOf[r.SessionID]; !exists {
				parentOf[r.SessionID] = r.ParentSessionID
			}
		}
	}
	resolveRootSession := func(id string) string {
		if id == "" {
			return ""
		}
		seen := map[string]struct{}{id: {}}
		cur := id
		for {
			nxt := parentOf[cur]
			if nxt == "" || nxt == cur {
				return cur
			}
			if _, cycle := seen[nxt]; cycle {
				return id
			}
			seen[nxt] = struct{}{}
			cur = nxt
		}
	}

	lcpSeq := 0
	for _, pr := range records {
		r := pr.rec
		hasDistinctParent := r.ParentSessionID != "" && r.ParentSessionID != r.SessionID
		isSubagent := hasDistinctParent || r.SubagentID != ""
		if isSubagent {
			parentKey := ""
			if hasDistinctParent {
				parentKey = resolveRootSession(r.ParentSessionID)
			}
			if parentKey == "" {
				parentKey = resolveRootSession(r.SessionID)
			}
			if parentKey == "" {
				parentKey = "subagent-root-" + r.SubagentID
			}
			agentKey := r.SubagentID
			if agentKey == "" {
				agentKey = r.SessionID
			}
			if agentKey == "" {
				agentKey = r.RequestID
			}
			b := getOrCreate("sess:"+parentKey, r.ArrivalUnixNano, r.CacheSaltID)
			addSubagentRecord(b, agentKey, r.SubagentType, pr)
			continue
		}

		if r.SessionID != "" {
			b := getOrCreate("sess:"+r.SessionID, r.ArrivalUnixNano, r.CacheSaltID)
			b.parent = append(b.parent, pr)
			continue
		}

		// Sessionless request: match via longest-common-prefix against active sessionless chains.
		bestIdx := -1
		bestLCP := 0
		for i, cand := range lcpActive {
			if maxIdleGapSec > 0 && float64(r.ArrivalUnixNano-cand.lastEndNs)/1e9 > maxIdleGapSec {
				continue
			}
			if len(cand.lastHashes) == 0 || len(pr.hashes) == 0 {
				continue
			}
			lcp := longestCommonPrefix(cand.lastHashes, pr.hashes)
			overlap := float64(lcp) / float64(len(cand.lastHashes))
			if overlap < lcpThreshold && len(cand.lastHashes) >= 2 && lcp >= len(cand.lastHashes)-1 {
				// 1-block boundary tolerance for trailing turn-template tokens on the previous turn.
				overlap = float64(lcp) / float64(len(cand.lastHashes)-1)
			}
			if overlap >= lcpThreshold && (lcp > bestLCP || (lcp == bestLCP && bestIdx >= 0 && cand.lastEndNs > lcpActive[bestIdx].lastEndNs)) {
				bestLCP = lcp
				bestIdx = i
			}
		}

		endNs := r.ArrivalUnixNano + int64(r.APITimeSeconds*1e9)
		if bestIdx >= 0 {
			cand := lcpActive[bestIdx]
			cand.bucket.parent = append(cand.bucket.parent, pr)
			cand.lastHashes = pr.hashes
			cand.lastEndNs = endNs
		} else {
			key := fmt.Sprintf("lcp:%06d", lcpSeq)
			lcpSeq++
			b := getOrCreate(key, r.ArrivalUnixNano, r.CacheSaltID)
			b.parent = append(b.parent, pr)
			lcpActive = append(lcpActive, &lcpCandidate{
				bucket:     b,
				lastHashes: pr.hashes,
				lastEndNs:  endNs,
			})
		}
	}

	if lookahead > 0 {
		for _, b := range buckets {
			if len(b.subOrder) == 0 && len(b.parent) >= 3 {
				extractUntaggedSubagents(b, lookahead)
			}
		}
	}

	return buckets
}

func addSubagentRecord(b *sessionBucket, agentKey, subagentType string, pr preparedRecord) {
	r := pr.rec
	grp, ok := b.subagents[agentKey]
	endNs := r.ArrivalUnixNano + int64(r.APITimeSeconds*1e9)
	if !ok {
		st := subagentType
		if st == "" {
			st = "captured"
		}
		grp = &subagentGroup{
			agentID:      shortenID(agentKey),
			subagentType: st,
			startNs:      r.ArrivalUnixNano,
			endNs:        endNs,
		}
		b.subagents[agentKey] = grp
		b.subOrder = append(b.subOrder, agentKey)
	} else {
		if r.ArrivalUnixNano < grp.startNs {
			grp.startNs = r.ArrivalUnixNano
		}
		if endNs > grp.endNs {
			grp.endNs = endNs
		}
		if grp.subagentType == "captured" && subagentType != "" {
			grp.subagentType = subagentType
		}
	}
	grp.records = append(grp.records, pr)
}

// extractUntaggedSubagents detects untagged synchronous or concurrent subagent branches
// within a session when turn i diverges from turn i-1's KV block prefix and a subsequent
// turn k within lookahead resumes turn i-1's prefix (within a 1-block boundary tolerance).
func extractUntaggedSubagents(b *sessionBucket, lookahead int) {
	var keptParent []preparedRecord
	turns := b.parent
	i := 0
	subSeq := 0
	for i < len(turns) {
		if len(keptParent) == 0 {
			keptParent = append(keptParent, turns[i])
			i++
			continue
		}
		prev := keptParent[len(keptParent)-1]
		prevLen := len(prev.hashes)
		curr := turns[i]
		lcpPrev := longestCommonPrefix(prev.hashes, curr.hashes)
		prevEndNs := prev.rec.ArrivalUnixNano + int64(prev.rec.APITimeSeconds*1e9)

		// If curr completely extends prev, keep on parent timeline.
		if prevLen == 0 || lcpPrev >= prevLen {
			keptParent = append(keptParent, curr)
			i++
			continue
		}

		// Check if a future turn within lookahead resumes prev's prefix (within 1-block boundary tolerance).
		minResume := max(1, prevLen-1)
		resumeIdx := -1
		limit := min(len(turns), i+1+lookahead)
		for k := i + 1; k < limit; k++ {
			lcpK := longestCommonPrefix(prev.hashes, turns[k].hashes)
			if lcpK >= minResume && lcpK > lcpPrev {
				resumeIdx = k
				break
			}
		}

		isConcurrent := curr.rec.ArrivalUnixNano < prevEndNs-1_000_000 // >1ms overlap
		if resumeIdx > i {
			subSeq++
			agentKey := fmt.Sprintf("sa_inferred_%d", subSeq)
			for j := i; j < resumeIdx; j++ {
				addSubagentRecord(b, agentKey, "inferred", turns[j])
			}
			i = resumeIdx
			continue
		}
		if isConcurrent {
			subSeq++
			agentKey := fmt.Sprintf("sa_inferred_%d", subSeq)
			addSubagentRecord(b, agentKey, "inferred", curr)
			i++
			continue
		}

		keptParent = append(keptParent, curr)
		i++
	}
	b.parent = keptParent
}

func inferCohortSystemBlocks(buckets []*sessionBucket, minCohort int) []int {
	out := make([]int, len(buckets))
	if minCohort <= 1 || len(buckets) < minCohort {
		return out
	}

	type cohortKey struct {
		model string
		salt  string
	}
	cohorts := make(map[cohortKey][]int)
	for idx, b := range buckets {
		first := firstBucketRecord(b)
		if first == nil || len(first.hashes) == 0 {
			continue
		}
		k := cohortKey{model: first.rec.Model, salt: b.cacheSaltID}
		cohorts[k] = append(cohorts[k], idx)
	}

	for _, idxs := range cohorts {
		if len(idxs) < minCohort {
			continue
		}
		common := firstBucketRecord(buckets[idxs[0]]).hashes
		for _, bi := range idxs[1:] {
			h := firstBucketRecord(buckets[bi]).hashes
			lcp := longestCommonPrefix(common, h)
			common = common[:lcp]
			if len(common) == 0 {
				break
			}
		}
		if len(common) > 0 {
			for _, bi := range idxs {
				out[bi] = len(common)
			}
		}
	}
	return out
}

func firstBucketRecord(b *sessionBucket) *preparedRecord {
	if len(b.parent) > 0 {
		return &b.parent[0]
	}
	for _, k := range b.subOrder {
		if grp := b.subagents[k]; grp != nil && len(grp.records) > 0 {
			return &grp.records[0]
		}
	}
	return nil
}

type rawTimelineItem struct {
	startNs  int64
	orderIdx int
	parent   *preparedRecord
	subagent *subagentGroup
}

func buildWekaTrace(idx int, b *sessionBucket, blockSize int, scope string, systemTokens int, idMap map[string]int) WekaTrace {
	items := make([]rawTimelineItem, 0, len(b.parent)+len(b.subOrder))
	order := 0
	for i := range b.parent {
		items = append(items, rawTimelineItem{
			startNs:  b.parent[i].rec.ArrivalUnixNano,
			orderIdx: order,
			parent:   &b.parent[i],
		})
		order++
	}
	for _, subKey := range b.subOrder {
		grp := b.subagents[subKey]
		items = append(items, rawTimelineItem{
			startNs:  grp.startNs,
			orderIdx: order,
			subagent: grp,
		})
		order++
	}
	slices.SortFunc(items, func(a, b rawTimelineItem) int {
		if c := cmp.Compare(a.startNs, b.startNs); c != 0 {
			return c
		}
		return cmp.Compare(a.orderIdx, b.orderIdx)
	})

	traceStartNs := b.startNs
	if len(items) > 0 && items[0].startNs < traceStartNs {
		traceStartNs = items[0].startNs
	}

	assignHashIDs := func(hexHashes []string) []int {
		ids := make([]int, len(hexHashes))
		for i, h := range hexHashes {
			id, ok := idMap[h]
			if !ok {
				id = len(idMap) + 1
				idMap[h] = id
			}
			ids[i] = id
		}
		return ids
	}

	var models []string
	seenModel := make(map[string]struct{})
	recordModel := func(m string) string {
		if m == "" {
			m = "unknown"
		}
		if _, ok := seenModel[m]; !ok {
			seenModel[m] = struct{}{}
			models = append(models, m)
		}
		return m
	}

	var entries []WekaEntry
	var totals WekaTotals
	var prevParentEndSec float64
	hasPrevParent := false

	for _, item := range items {
		if item.parent != nil {
			pr := item.parent
			r := pr.rec
			tSec := roundSeconds(max(0.0, float64(r.ArrivalUnixNano-traceStartNs)/1e9))
			apiSec := roundSeconds(max(0.0, r.APITimeSeconds))
			thinkSec := 0.0
			if hasPrevParent {
				thinkSec = roundSeconds(max(0.0, tSec-prevParentEndSec))
			}
			prevParentEndSec = tSec + apiSec
			hasPrevParent = true

			inTokens := r.TokenizedIn
			if inTokens == 0 && r.UsageIn > 0 {
				inTokens = r.UsageIn
			}
			outTokens := max(1, r.OutputTokens)
			totals.ParentTokens.Input += inTokens
			totals.ParentTokens.Output += outTokens

			sType := r.StreamType
			if sType != streamTypeStreaming {
				sType = streamTypeNormal
			}
			var ttftPtr *float64
			if sType == streamTypeStreaming && r.FirstTokenSeconds > 0 {
				ttft := roundSeconds(r.FirstTokenSeconds)
				ttftPtr = &ttft
			}

			turn := &WekaTurnRequest{
				T:         tSec,
				Type:      sType,
				Model:     recordModel(r.Model),
				In:        inTokens,
				Out:       outTokens,
				HashIDs:   assignHashIDs(pr.hashes),
				APITime:   &apiSec,
				ThinkTime: &thinkSec,
				TTFT:      ttftPtr,
			}
			entries = append(entries, WekaEntry{Turn: turn})
			continue
		}

		grp := item.subagent
		totals.SubagentCount++
		subStartSec := roundSeconds(max(0.0, float64(grp.startNs-traceStartNs)/1e9))
		durMs := int(math.Round(max(0.0, float64(grp.endNs-grp.startNs)/1e6)))

		inner := make([]WekaTurnRequest, 0, len(grp.records))
		var subModels []string
		seenSubModel := make(map[string]struct{})
		subTotalTokens := 0
		var prevSubEndSec float64
		for i, pr := range grp.records {
			r := pr.rec
			innerT := max(subStartSec, roundSeconds(max(0.0, float64(r.ArrivalUnixNano-traceStartNs)/1e9)))
			apiSec := roundSeconds(max(0.0, r.APITimeSeconds))
			thinkSec := 0.0
			if i > 0 {
				thinkSec = roundSeconds(max(0.0, innerT-prevSubEndSec))
			}
			prevSubEndSec = innerT + apiSec

			inTokens := r.TokenizedIn
			if inTokens == 0 && r.UsageIn > 0 {
				inTokens = r.UsageIn
			}
			outTokens := max(1, r.OutputTokens)
			totals.SubagentTokens.Input += inTokens
			totals.SubagentTokens.Output += outTokens
			subTotalTokens += inTokens + outTokens

			m := recordModel(r.Model)
			if _, ok := seenSubModel[m]; !ok {
				seenSubModel[m] = struct{}{}
				subModels = append(subModels, m)
			}

			// WekaSubagentEntry.requests in inference-perf accepts WekaNormalRequest (type "n")
			// only, and aiperf accepts WekaNormalRequest | WekaStreamingRequest; emitting "n"
			// with think_time and root-elapsed t >= subStartSec satisfies both loaders.
			inner = append(inner, WekaTurnRequest{
				T:         innerT,
				Type:      streamTypeNormal,
				Model:     m,
				In:        inTokens,
				Out:       outTokens,
				HashIDs:   assignHashIDs(pr.hashes),
				APITime:   &apiSec,
				ThinkTime: &thinkSec,
			})
		}
		if len(subModels) == 0 {
			subModels = []string{"unknown"}
		}
		subType := grp.subagentType
		if subType == "" {
			subType = "captured"
		}

		entries = append(entries, WekaEntry{
			Subagent: &WekaSubagentEntry{
				T:            subStartSec,
				Type:         entryTypeSubagent,
				AgentID:      grp.agentID,
				SubagentType: subType,
				Status:       outcomeCompleted,
				DurationMs:   &durMs,
				TotalTokens:  &subTotalTokens,
				Models:       subModels,
				Requests:     inner,
				ToolTokens:   0,
				SystemTokens: 0,
			},
		})
	}

	if len(models) == 0 {
		models = []string{"unknown"}
	}
	if entries == nil {
		entries = []WekaEntry{}
	}
	totals.CombinedTokens = WekaTokenPair{
		Input:  totals.ParentTokens.Input + totals.SubagentTokens.Input,
		Output: totals.ParentTokens.Output + totals.SubagentTokens.Output,
	}

	return WekaTrace{
		ID:           fmt.Sprintf("trace_%04d", idx),
		Models:       models,
		BlockSize:    blockSize,
		HashIDScope:  scope,
		ToolTokens:   0,
		SystemTokens: systemTokens,
		Requests:     entries,
		Totals:       &totals,
	}
}

// LoadCapturedRecords reads CapturedRecord / Event JSONL lines from one or more
// .jsonl or .jsonl.gz file paths.
func LoadCapturedRecords(paths []string) ([]CapturedRecord, error) {
	var all []CapturedRecord
	for _, p := range paths {
		clean := filepath.Clean(p)
		root, err := os.OpenRoot(filepath.Dir(clean))
		if err != nil {
			return nil, fmt.Errorf("open directory for %q: %w", clean, err)
		}
		f, err := root.Open(filepath.Base(clean))
		_ = root.Close()
		if err != nil {
			return nil, fmt.Errorf("open %q: %w", clean, err)
		}

		var reader io.Reader = f
		var gz *gzip.Reader
		if strings.HasSuffix(strings.ToLower(clean), ".gz") {
			gz, err = gzip.NewReader(f)
			if err != nil {
				_ = f.Close()
				return nil, fmt.Errorf("open gzip %q: %w", clean, err)
			}
			reader = gz
		}

		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 64*1024), maxScannerLine)
		lineNo := 0
		for scanner.Scan() {
			lineNo++
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var rec CapturedRecord
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				if gz != nil {
					_ = gz.Close()
				}
				_ = f.Close()
				return nil, fmt.Errorf("%s:%d: %w", clean, lineNo, err)
			}
			rec.normalize()
			all = append(all, rec)
		}
		scanErr := scanner.Err()
		if gz != nil {
			_ = gz.Close()
		}
		_ = f.Close()
		if scanErr != nil {
			return nil, fmt.Errorf("read %q: %w", clean, scanErr)
		}
	}
	return all, nil
}

// WriteWekaTracesJSONL writes traces as one JSON object per line to path (mode 0600).
func WriteWekaTracesJSONL(path string, traces []WekaTrace) error {
	clean := filepath.Clean(path)
	dir := filepath.Dir(clean)
	if err := os.MkdirAll(dir, sinkDirPerm); err != nil {
		return fmt.Errorf("create output directory %q: %w", dir, err)
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return fmt.Errorf("open output directory %q: %w", dir, err)
	}
	defer func() { _ = root.Close() }()

	f, err := root.OpenFile(filepath.Base(clean), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, sinkFilePerm)
	if err != nil {
		return fmt.Errorf("open %q: %w", clean, err)
	}
	bw := bufio.NewWriterSize(f, exportWriterBufSize)
	for _, tr := range traces {
		b, err := json.Marshal(tr)
		if err != nil {
			_ = f.Close()
			return err
		}
		if _, err := bw.Write(b); err != nil {
			_ = f.Close()
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			_ = f.Close()
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// WriteWekaTracesDir writes each trace as <trace.ID>.json inside dir (mode 0750/0600).
func WriteWekaTracesDir(dir string, traces []WekaTrace) error {
	cleanDir := filepath.Clean(dir)
	if err := os.MkdirAll(cleanDir, sinkDirPerm); err != nil {
		return fmt.Errorf("create output directory %q: %w", cleanDir, err)
	}
	root, err := os.OpenRoot(cleanDir)
	if err != nil {
		return fmt.Errorf("open output directory %q: %w", cleanDir, err)
	}
	defer func() { _ = root.Close() }()

	for _, tr := range traces {
		name := sanitizePodName(tr.ID) + ".json"
		b, err := json.Marshal(tr)
		if err != nil {
			return err
		}
		f, err := root.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, sinkFilePerm)
		if err != nil {
			return fmt.Errorf("open %q: %w", name, err)
		}
		_, writeErr := f.Write(b)
		closeErr := f.Close()
		if writeErr != nil {
			return writeErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func longestCommonPrefix(a, b []string) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func shortenID(id string) string {
	if len(id) <= 12 {
		return id
	}
	return id[:12]
}

func roundSeconds(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
