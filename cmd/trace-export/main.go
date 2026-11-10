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

// Command trace-export assembles per-request .jsonl and .jsonl.gz files emitted by
// kv-block-trace-recorder into WEKA multi-turn agentic trace datasets.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/tracerecorder"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "trace-export: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("trace-export", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	var (
		inputDir            string
		outputJSONL         string
		outputDir           string
		targetBlockSize     int
		hashIDScope         string
		lcpOverlap          float64
		maxIdleGap          float64
		dropClassifierCalls bool
		includeNonCompleted bool
		minSystemCohort     int
		subagentLookahead   int
	)

	fs.StringVar(&inputDir, "input-dir", "", "Directory containing trace .jsonl or .jsonl.gz files")
	fs.StringVar(&outputJSONL, "output-jsonl", "", "Output file path for WEKA traces.jsonl")
	fs.StringVar(&outputDir, "output-dir", "", "Output directory path for per-trace trace_NNNN.json files")
	fs.IntVar(&targetBlockSize, "target-block-size", 0, "Target KV block size in tokens (integer multiple of captured block size; 0 keeps captured)")
	fs.StringVar(&hashIDScope, "hash-id-scope", tracerecorder.HashIDScopeLocal, "Hash ID remapping scope: local or global")
	fs.Float64Var(&lcpOverlap, "lcp-overlap-threshold", tracerecorder.DefaultLCPOverlapThreshold, "Block-prefix overlap threshold in (0, 1] for sessionless LCP chaining")
	fs.Float64Var(&maxIdleGap, "max-idle-gap-seconds", 0, "Maximum idle gap in seconds for sessionless LCP chaining (0 = unlimited)")
	fs.BoolVar(&dropClassifierCalls, "drop-classifier-calls", false, "Filter out tool-free requests with max_output_tokens <= 64")
	fs.BoolVar(&includeNonCompleted, "include-non-completed", false, "Include non-completed requests in assembled traces")
	fs.IntVar(&minSystemCohort, "min-system-cohort", tracerecorder.DefaultMinSystemCohort, "Minimum cohort size to infer system_tokens across traces (<0 disables)")
	fs.IntVar(&subagentLookahead, "subagent-lookahead", tracerecorder.DefaultSubagentLookahead, "Lookahead window for untagged synchronous subagent detection (<0 disables)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	var paths []string
	collectDir := func(dir string) error {
		cleanDir := filepath.Clean(dir)
		entries, err := os.ReadDir(cleanDir)
		if err != nil {
			return fmt.Errorf("read input-dir %q: %w", cleanDir, err)
		}
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() || strings.HasPrefix(name, ".") || strings.HasSuffix(name, ".tmp") {
				continue
			}
			if strings.HasSuffix(name, ".jsonl") || strings.HasSuffix(name, ".jsonl.gz") {
				paths = append(paths, filepath.Join(cleanDir, name))
			}
		}
		return nil
	}

	if inputDir != "" {
		if err := collectDir(inputDir); err != nil {
			return err
		}
	}
	for _, arg := range fs.Args() {
		info, err := os.Stat(arg)
		if err == nil && info.IsDir() {
			if err := collectDir(arg); err != nil {
				return err
			}
			continue
		}
		paths = append(paths, arg)
	}
	slices.Sort(paths)
	paths = slices.Compact(paths)
	if len(paths) == 0 {
		return errors.New("specify --input-dir or at least one capture .jsonl/.jsonl.gz file argument")
	}
	if (outputJSONL == "") == (outputDir == "") {
		return errors.New("specify exactly one of --output-jsonl or --output-dir")
	}

	records, err := tracerecorder.LoadCapturedRecords(paths)
	if err != nil {
		return err
	}

	traces, err := tracerecorder.AssembleWekaTraces(records, tracerecorder.AssembleOptions{
		TargetBlockSize:     targetBlockSize,
		HashIDScope:         hashIDScope,
		LCPOverlapThreshold: lcpOverlap,
		MaxIdleGapSeconds:   maxIdleGap,
		DropClassifierCalls: dropClassifierCalls,
		IncludeNonCompleted: includeNonCompleted,
		MinSystemCohort:     minSystemCohort,
		SubagentLookahead:   subagentLookahead,
	})
	if err != nil {
		return err
	}

	if outputJSONL != "" {
		if err := tracerecorder.WriteWekaTracesJSONL(outputJSONL, traces); err != nil {
			return err
		}
	} else {
		if err := tracerecorder.WriteWekaTracesDir(outputDir, traces); err != nil {
			return err
		}
	}

	_, err = fmt.Fprintf(stdout, "exported %d traces from %d captured records\n", len(traces), len(records))
	return err
}
