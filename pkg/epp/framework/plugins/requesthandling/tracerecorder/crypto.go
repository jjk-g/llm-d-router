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
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
)

const (
	// EnvTraceHMACSecret is the fallback environment variable carrying the HMAC secret.
	EnvTraceHMACSecret = "EPP_TRACE_HMAC_SECRET" // #nosec G101 -- env var name, not a credential

	// MinSecretBytes is the minimum required HMAC key length (256 bits).
	MinSecretBytes = 32

	// EmptyCacheSaltID is the canonical sentinel emitted when CacheSalt is unset.
	EmptyCacheSaltID = "c_0000000000000000"

	// MaxSafeUint53 is 2^53 - 1 (IEEE-754 exact integer ceiling across JSON parsers).
	MaxSafeUint53 uint64 = (uint64(1) << 53) - 1

	// MaxSampleRateBPS is 100% in basis points (10,000 bps).
	MaxSampleRateBPS = 10000

	fingerprintDomainPrefix = "epp-trace-fp:v1:"
)

// LoadCaptureSecret loads and validates the HMAC capture secret from secretFilePath
// or EPP_TRACE_HMAC_SECRET. It fails closed when no secret is configured or when
// the secret is shorter than MinSecretBytes, unless allowEphemeralDev is true and
// neither secret source was provided.
func LoadCaptureSecret(secretFilePath string, allowEphemeralDev bool) ([]byte, error) {
	var raw []byte
	switch {
	case secretFilePath != "":
		data, err := os.ReadFile(secretFilePath) // #nosec G304 -- operator-configured K8s Secret mount path
		if err != nil {
			return nil, fmt.Errorf("failed to read capture secret file %q: %w", secretFilePath, err)
		}
		raw = trimOptionalASCIINewline(data)
	case os.Getenv(EnvTraceHMACSecret) != "":
		raw = []byte(os.Getenv(EnvTraceHMACSecret))
	case allowEphemeralDev:
		ephemeral := make([]byte, MinSecretBytes)
		if _, err := rand.Read(ephemeral); err != nil {
			return nil, fmt.Errorf("failed to generate ephemeral capture secret: %w", err)
		}
		return ephemeral, nil
	default:
		return nil, errors.New("capture secret required: set secretFilePath or EPP_TRACE_HMAC_SECRET (>= 32 bytes), or set allowEphemeralSecretForDev=true for single-process tests")
	}

	if len(raw) < MinSecretBytes {
		return nil, fmt.Errorf("capture secret must be at least %d bytes, got %d bytes", MinSecretBytes, len(raw))
	}
	out := make([]byte, len(raw))
	copy(out, raw)
	return out, nil
}

// trimOptionalASCIINewline strips a single trailing \n or \r\n when the preceding
// bytes are printable ASCII, matching mounted Kubernetes Secret files created from
// string literals without truncating arbitrary binary keys.
func trimOptionalASCIINewline(b []byte) []byte {
	trimmed := bytes.TrimSuffix(b, []byte("\n"))
	trimmed = bytes.TrimSuffix(trimmed, []byte("\r"))
	if len(trimmed) == len(b) || len(trimmed) == 0 {
		return b
	}
	for _, c := range trimmed {
		if c < 0x20 || c > 0x7e {
			return b
		}
	}
	return trimmed
}

// KeyFingerprint computes hex(SHA256("epp-trace-fp:v1:" || secret)[:4]) (8 lowercase hex chars)
// so offline compaction can detect accidental mixing of segments keyed with different secrets.
func KeyFingerprint(secret []byte) string {
	h := sha256.New()
	_, _ = h.Write([]byte(fingerprintDomainPrefix))
	_, _ = h.Write(secret)
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:4])
}

// ShouldSample performs stateless deterministic session-cohort sampling in basis points [0, 10000].
func ShouldSample(secret []byte, rawSessionKey string, sampleRateBPS int) bool {
	if sampleRateBPS >= MaxSampleRateBPS {
		return true
	}
	if sampleRateBPS <= 0 {
		return false
	}
	mac := hmacDomain(secret, "sample:", []byte(rawSessionKey))
	bucket := binary.BigEndian.Uint64(mac[:8]) % uint64(MaxSampleRateBPS)
	return bucket < uint64(sampleRateBPS) // #nosec G115 -- sampleRateBPS is verified > 0 above
}

// AnonymizeSessionID maps a raw session key to "s_<16-hex>".
func AnonymizeSessionID(secret []byte, rawSessionKey string) string {
	mac := hmacDomain(secret, "session:", []byte(rawSessionKey))
	return "s_" + hex.EncodeToString(mac[:8])
}

// AnonymizeCacheSalt maps a raw cache_salt string to "c_<16-hex>", or returns
// EmptyCacheSaltID when salt is empty.
func AnonymizeCacheSalt(secret []byte, salt string) string {
	if salt == "" {
		return EmptyCacheSaltID
	}
	mac := hmacDomain(secret, "salt:", []byte(salt))
	return "c_" + hex.EncodeToString(mac[:8])
}

// AnonymizeAgentID maps a non-empty raw agent identifier to "a_<16-hex>" so raw
// header values never appear in trace segments.
func AnonymizeAgentID(secret []byte, rawAgentID string) string {
	if rawAgentID == "" {
		return ""
	}
	mac := hmacDomain(secret, "agent:", []byte(rawAgentID))
	return "a_" + hex.EncodeToString(mac[:8])
}

// AnonymizeSubagentType maps a non-empty raw subagent_type string to "st_<16-hex>".
func AnonymizeSubagentType(secret []byte, rawSubagentType string) string {
	if rawSubagentType == "" {
		return ""
	}
	mac := hmacDomain(secret, "subagent_type:", []byte(rawSubagentType))
	return "st_" + hex.EncodeToString(mac[:8])
}

// AnonymizeBlockHash maps a 64-bit KV block digest to an HMAC-keyed integer in [1, 2^53 - 1].
func AnonymizeBlockHash(secret []byte, rawHash uint64) uint64 {
	var rawBytes [8]byte
	binary.LittleEndian.PutUint64(rawBytes[:], rawHash)
	mac := hmacDomain(secret, "block:", rawBytes[:])
	val := binary.BigEndian.Uint64(mac[:8]) & MaxSafeUint53
	if val == 0 {
		return 1
	}
	return val
}

func hmacDomain(secret []byte, domain string, payload []byte) [sha256.Size]byte {
	mac := hmac.New(sha256.New, secret)
	_, _ = mac.Write([]byte(domain))
	_, _ = mac.Write(payload)
	var out [sha256.Size]byte
	_ = mac.Sum(out[:0])
	return out
}
