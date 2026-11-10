# GKE Verification Harness for `kv-block-trace-recorder` (Gates 3, 4 & 5)

Self-contained Google Kubernetes Engine (GKE) verification harness for the `kv-block-trace-recorder` Endpoint Picker (EPP) plugin (`llm-d-router`), `cmd/trace-export`, and offline `epp-trace-compact` + `weka_trace_replay` / `aiperf` AgentX pipeline (`inference-perf`).

## Environment Configuration

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | *(required for cloud commands)* | Target GCP project ID (e.g. `my-gcp-project`) |
| `TRACE_GCS_BUCKET` | *(required for cloud commands)* | Uniform-bucket-level-access GCS bucket for sealed `.jsonl.gz` trace segments (e.g. `my-llm-d-trace-bucket`) |
| `GCP_REGION` | `us-central1` | Region for GCS bucket creation and default Artifact Registry host |
| `K8S_CONTEXT` | current `kubectl` context | Target GKE cluster context (`GcsFuseCsiDriver` enabled; NVIDIA GPUs for `VLLM_DEVICE=gpu`) |
| `EPP_IMAGE_REPO` | `${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/llm-d/llm-d-router-endpoint-picker` | Artifact Registry repository for the custom EPP image |
| `EPP_IMAGE_TAG` | `v0.0.1-trace-gates` | Image tag built from `Dockerfile.epp` |
| `VLLM_DEVICE` | `gpu` | Real `vLLM` runtime for Gate 4 (`gpu` or `cpu`) |
| `NAMESPACE` | `llm-d-trace-gates` | Ephemeral Kubernetes namespace cleaned up automatically unless `KEEP_NAMESPACE=true` |
| `INFERENCE_PERF_DIR` | `../inference-perf` | Path to a local `inference-perf` checkout with `epp-trace-compact` and `weka_trace_replay` |

## Usage

```bash
# Offline Helm template structural checks + Python oracle unit tests (no cluster required)
./scripts/run-gke-gates.sh validate-local

# Export target cloud coordinates
export GCP_PROJECT="my-gcp-project"
export TRACE_GCS_BUCKET="my-llm-d-trace-bucket"
export GCP_REGION="us-central1"

# 1. Verify gcloud/ADC auth, GKE reachability, GCS FUSE CSI driver, and Workload Identity IAM
./scripts/run-gke-gates.sh preflight

# 2. Create gs://${TRACE_GCS_BUCKET} with uniform bucket-level access (idempotent)
./scripts/run-gke-gates.sh create-bucket

# 3. Build & push EPP container image from Dockerfile.epp to Artifact Registry
./scripts/run-gke-gates.sh build-push

# 4. Run Gate 3 (2x llm-d-inference-sim + 2x active-active EPP + Pass 1 Record -> Pass 2 Replay isomorphism)
./scripts/run-gke-gates.sh gate3

# 5. Run Gate 4 (1x real vLLM + live vs. replay prefix-cache hit-rate parity + 200-req p99 overhead)
./scripts/run-gke-gates.sh gate4

# 6. Run Gate 5 locally (hermetic live-EPP + aiperf & inference-perf AgentX smoke tests + round-trip replay)
./scripts/run-gke-gates.sh gate5-local

# 7. Run Gate 5 on GKE cluster (live GKE EPP stack + aiperf & inference-perf AgentX smoke tests)
./scripts/run-gke-gates.sh gate5

# 8. Run full end-to-end suite (create-bucket -> build-push -> preflight -> gate3 -> gate4 -> gate5)
./scripts/run-gke-gates.sh all
```

## Context-Aware Access & Workload Identity Notes

- **Context-Aware Access Fallback**: In environments where Context-Aware Access restrictions block non-interactive user credential refresh on `gcloud`, `scripts/run-gke-gates.sh` automatically probes client-certificate mode (`CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=True`) and Application Default Credentials (`gcloud auth application-default print-access-token` via in-memory process substitution plus an ephemeral `gke-gcloud-auth-plugin --use_application_default_credentials` shim) without modifying `~/.kube` or `~/.config/gcloud`.
- **GCS FUSE Workload Identity IAM**: The harness never mutates IAM policies automatically. If the EPP Kubernetes ServiceAccount (`ns/${NAMESPACE}/sa/llm-d-router-epp`) lacks object read/write access (`roles/storage.objectUser`) on `gs://${TRACE_GCS_BUCKET}`, `preflight` prints the exact `gcloud storage buckets add-iam-policy-binding` command for an operator to execute.
