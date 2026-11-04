#!/usr/bin/env bash
# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Self-contained GKE verification harness for kv-block-trace-recorder (Gates 3 & 4).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS_DIR="${REPO_ROOT}/hack/gke-trace-gates"
CHART_DIR="${REPO_ROOT}/config/charts/llm-d-router-standalone"

export PYTHONDONTWRITEBYTECODE=1

# Canonical environment coordinates (all overridable)
export GCP_PROJECT="${GCP_PROJECT:-}"
export GCP_REGION="${GCP_REGION:-us-central1}"
export K8S_CONTEXT="${K8S_CONTEXT:-$(kubectl config current-context 2>/dev/null || true)}"
export TRACE_GCS_BUCKET="${TRACE_GCS_BUCKET:-}"
export EPP_IMAGE_REPO="${EPP_IMAGE_REPO:-}"
export EPP_IMAGE_TAG="${EPP_IMAGE_TAG:-v0.0.1-trace-gates}"
export VLLM_DEVICE="${VLLM_DEVICE:-gpu}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
export RELEASE_NAME="${RELEASE_NAME:-llm-d-router}"
export NAMESPACE="${NAMESPACE:-llm-d-trace-gates}"
export KEEP_NAMESPACE="${KEEP_NAMESPACE:-false}"
export RUN_ID="${RUN_ID:-run-$(date -u +%Y%m%dT%H%M%SZ)}"

SCRATCH_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
export WORK_ROOT="${WORK_ROOT:-${TMPDIR:-/tmp}/llm-d-trace-gates}"
RUN_DIR="${WORK_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

HELM_BIN="${HELM_BIN:-$(command -v helm || echo "${HOME}/.local/bin/helm")}"
INFERENCE_PERF_DIR="${INFERENCE_PERF_DIR:-${SCRATCH_PARENT}/inference-perf}"

GCLOUD_AUTH_MODE="standard"
NAMESPACE_TOUCHED="false"
PORT_FORWARD_PIDS=()

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"
}

err() {
  printf '[%s] ERROR: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2
}

cleanup() {
  local exit_code=$?
  rm -rf "${RUN_DIR}/docker-config" 2>/dev/null || true
  for pid in "${PORT_FORWARD_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ "${NAMESPACE_TOUCHED}" == "true" && "${KEEP_NAMESPACE}" != "true" ]]; then
    log "Cleaning up ephemeral namespace ${NAMESPACE} (set KEEP_NAMESPACE=true to retain)..."
    kubectl --context="${K8S_CONTEXT}" delete namespace "${NAMESPACE}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

start_port_forward() {
  local target="$1"
  local port_map="$2"
  local log_file="$3"
  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" port-forward "${target}" "${port_map}" >"${log_file}" 2>&1 &
  local pf_pid=$!
  PORT_FORWARD_PIDS+=("${pf_pid}")
  local local_port="${port_map%%:*}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${pf_pid}" 2>/dev/null; then
      err "port-forward ${target} (${port_map}) exited prematurely; see ${log_file}"
      return 1
    fi
    if python3 - "${local_port}" <<'EOF'
import socket, sys
s = socket.socket()
s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
EOF
    then
      return 0
    fi
    sleep 0.5
  done
  err "Timed out waiting for port-forward ${target} on 127.0.0.1:${local_port}"
  return 1
}

stop_port_forwards() {
  for pid in "${PORT_FORWARD_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  PORT_FORWARD_PIDS=()
}

# ---------------------------------------------------------------------------
# Non-interactive GCP / CAA / ADC Authentication Setup
# ---------------------------------------------------------------------------

run_gcloud() {
  if [[ "${GCLOUD_AUTH_MODE}" == "caa_cert" ]]; then
    CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=True gcloud "$@"
  elif [[ "${GCLOUD_AUTH_MODE}" == "adc" ]]; then
    gcloud --access-token-file=<(gcloud auth application-default print-access-token) "$@"
  else
    gcloud "$@"
  fi
}

require_cloud_env() {
  if [[ -z "${GCP_PROJECT}" ]]; then
    err "GCP_PROJECT must be set for cloud commands (e.g. export GCP_PROJECT=my-gcp-project)."
    return 1
  fi
  if [[ -z "${TRACE_GCS_BUCKET}" ]]; then
    err "TRACE_GCS_BUCKET must be set for cloud commands (e.g. export TRACE_GCS_BUCKET=my-llm-d-trace-bucket)."
    return 1
  fi
  if [[ -z "${EPP_IMAGE_REPO}" ]]; then
    EPP_IMAGE_REPO="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/llm-d/llm-d-router-endpoint-picker"
    export EPP_IMAGE_REPO
  fi
}

setup_gcp_auth() {
  require_cloud_env
  log "Probing non-interactive GCP auth for project=${GCP_PROJECT}..."
  local probe_err="${RUN_DIR}/gcloud_probe.err"

  if timeout 15s env CLOUDSDK_CORE_HTTP_TIMEOUT=10 gcloud projects describe "${GCP_PROJECT}" --format="value(projectNumber)" >"${RUN_DIR}/project_number.txt" 2>"${probe_err}"; then
    GCLOUD_AUTH_MODE="standard"
    log "Standard gcloud user credentials succeeded."
  elif timeout 15s env CLOUDSDK_CORE_HTTP_TIMEOUT=10 CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=True gcloud projects describe "${GCP_PROJECT}" --format="value(projectNumber)" >"${RUN_DIR}/project_number.txt" 2>"${probe_err}"; then
    GCLOUD_AUTH_MODE="caa_cert"
    export CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=True
    log "gcloud succeeded with CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE=True."
  elif timeout 15s gcloud auth application-default print-access-token >/dev/null 2>"${probe_err}"; then
    GCLOUD_AUTH_MODE="adc"
    mkdir -p "${RUN_DIR}/bin"
    # Resolve real gke-gcloud-auth-plugin path before prepending RUN_DIR/bin to PATH.
    local real_plugin
    real_plugin="$(command -v gke-gcloud-auth-plugin || echo /usr/bin/gke-gcloud-auth-plugin)"
    cat >"${RUN_DIR}/bin/gke-gcloud-auth-plugin" <<SHIM
#!/usr/bin/env bash
exec "${real_plugin}" --use_application_default_credentials "\$@"
SHIM
    chmod +x "${RUN_DIR}/bin/gke-gcloud-auth-plugin"
    export PATH="${RUN_DIR}/bin:${PATH}"
    CLOUDSDK_CORE_HTTP_TIMEOUT=10 run_gcloud projects describe "${GCP_PROJECT}" --format="value(projectNumber)" >"${RUN_DIR}/project_number.txt"
    log "Detected Context-Aware Access restrictions on user credentials; switched gcloud and gke-gcloud-auth-plugin to Application Default Credentials (ADC)."
  else
    err "Non-interactive GCP authentication is blocked (Context-Aware Access restrictions on user credentials or expired ADC)."
    cat >&2 <<EOF

================================================================================
ACTION REQUIRED: Refresh GCP Application Default Credentials in an interactive
browser terminal (NEVER run interactive login inside headless automation):

  gcloud auth application-default login

Then re-run:
  GCP_PROJECT=${GCP_PROJECT} TRACE_GCS_BUCKET=${TRACE_GCS_BUCKET} ./scripts/run-gke-gates.sh preflight
================================================================================
EOF
    return 1
  fi

  PROJECT_NUMBER="$(tr -d '[:space:]' <"${RUN_DIR}/project_number.txt")"
  export PROJECT_NUMBER
  log "Resolved GCP_PROJECT=${GCP_PROJECT} (projectNumber=${PROJECT_NUMBER})"
}

# ---------------------------------------------------------------------------
# Local Helm & Python Validation
# ---------------------------------------------------------------------------

ensure_helm_deps() {
  if [[ ! -x "${HELM_BIN}" ]]; then
    err "helm binary not found at ${HELM_BIN}"
    return 1
  fi
  EFFECTIVE_CHART_DIR="${RUN_DIR}/charts-staging/llm-d-router-standalone"
  if [[ ! -d "${EFFECTIVE_CHART_DIR}/charts" ]] || [[ -z "$(ls -A "${EFFECTIVE_CHART_DIR}/charts" 2>/dev/null || true)" ]]; then
    mkdir -p "${RUN_DIR}/charts-staging"
    cp -R "${REPO_ROOT}/config/charts/routerlib" "${RUN_DIR}/charts-staging/routerlib"
    cp -R "${REPO_ROOT}/config/charts/llm-d-router-standalone" "${EFFECTIVE_CHART_DIR}"
    log "Building Helm chart dependencies in ${EFFECTIVE_CHART_DIR}..."
    "${HELM_BIN}" dependency build "${EFFECTIVE_CHART_DIR}" >/dev/null
  fi
}

validate_local() {
  log "Validating local Helm templates and oracle unit tests..."
  ensure_helm_deps

  "${HELM_BIN}" template "${RELEASE_NAME}" "${EFFECTIVE_CHART_DIR}" \
    --namespace "${NAMESPACE}" \
    -f "${HARNESS_DIR}/values-trace-router.yaml" >"${RUN_DIR}/rendered-trace-router.yaml"

  "${HELM_BIN}" template "${RELEASE_NAME}" "${EFFECTIVE_CHART_DIR}" \
    --namespace "${NAMESPACE}" \
    -f "${HARNESS_DIR}/values-trace-router.yaml" \
    -f "${HARNESS_DIR}/values-no-recorder.yaml" >"${RUN_DIR}/rendered-no-recorder.yaml"

  python3 - "${RUN_DIR}/rendered-trace-router.yaml" "${RUN_DIR}/rendered-no-recorder.yaml" <<'EOF'
import sys, yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    docs_on = [d for d in yaml.safe_load_all(fh) if d]
with open(sys.argv[2], encoding="utf-8") as fh:
    docs_off = [d for d in yaml.safe_load_all(fh) if d]

epp_deps = [d for d in docs_on if d.get("kind") == "Deployment" and d["metadata"]["name"].endswith("-epp")]
proxy_deps = [d for d in docs_on if d.get("kind") == "Deployment" and d["metadata"]["name"].endswith("-proxy")]
assert len(epp_deps) == 1, f"expected 1 EPP Deployment, got {len(epp_deps)}"
assert len(proxy_deps) == 1, f"expected 1 service-mode Proxy Deployment, got {len(proxy_deps)}"

epp = epp_deps[0]
assert epp["spec"]["replicas"] == 2, f"expected 2 active-active EPP replicas, got {epp['spec']['replicas']}"
pod_meta = epp["spec"]["template"]["metadata"]
assert pod_meta.get("annotations", {}).get("gke-gcsfuse/volumes") == "true", "missing gke-gcsfuse/volumes annotation"

pod_spec = epp["spec"]["template"]["spec"]
containers = {c["name"]: c for c in pod_spec["containers"]}
assert "epp" in containers and "vllm-render" in containers, f"expected epp + vllm-render containers, got {list(containers)}"
assert "--config-file" in containers["epp"]["args"]
vols = {v["name"]: v for v in pod_spec.get("volumes", [])}
assert vols["gcs-traces"]["csi"]["driver"] == "gcsfuse.csi.storage.gke.io"
assert "emptyDir" in vols["trace-staging"]
assert vols["trace-hmac-secret"]["secret"]["secretName"] == "epp-trace-hmac-secret"

epp_off = [d for d in docs_off if d.get("kind") == "Deployment" and d["metadata"]["name"].endswith("-epp")][0]
off_args = epp_off["spec"]["template"]["spec"]["containers"][0]["args"]
assert "/config/no-recorder-plugins.yaml" in off_args, f"expected no-recorder config in off overlay, got {off_args}"
print("Helm template structural assertions PASSED.")
EOF

  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "${HARNESS_DIR}" -p "test_*.py" -v
  log "Local validation PASSED (artifacts in ${RUN_DIR})."
}

# ---------------------------------------------------------------------------
# Bucket Creation, Preflight & Image Build/Push
# ---------------------------------------------------------------------------

create_bucket() {
  setup_gcp_auth
  if run_gcloud storage buckets describe "gs://${TRACE_GCS_BUCKET}" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
    log "GCS bucket gs://${TRACE_GCS_BUCKET} already exists in ${GCP_PROJECT}."
  else
    log "Creating uniform-bucket-level-access GCS bucket gs://${TRACE_GCS_BUCKET} in ${GCP_REGION}..."
    run_gcloud storage buckets create "gs://${TRACE_GCS_BUCKET}" \
      --project="${GCP_PROJECT}" \
      --location="${GCP_REGION}" \
      --uniform-bucket-level-access
    log "Created gs://${TRACE_GCS_BUCKET}."
  fi
}

check_workload_identity_iam() {
  local ksa_name="${RELEASE_NAME}-epp"
  local expected_principal="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${GCP_PROJECT}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${ksa_name}"
  local ns_principal_set="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${GCP_PROJECT}.svc.id.goog/attribute.namespace/${NAMESPACE}"
  local policy_file="${RUN_DIR}/bucket_iam_policy.json"

  run_gcloud storage buckets get-iam-policy "gs://${TRACE_GCS_BUCKET}" \
    --project="${GCP_PROJECT}" \
    --format=json >"${policy_file}"

  if python3 - "${policy_file}" "${expected_principal}" "${ns_principal_set}" <<'EOF'
import json, sys

policy = json.load(open(sys.argv[1], encoding="utf-8"))
ksa_principal = sys.argv[2]
ns_principal = sys.argv[3]
rw_roles = {
    "roles/storage.objectUser",
    "roles/storage.objectAdmin",
    "roles/storage.admin",
    "roles/storage.legacyBucketWriter",
    "roles/storage.legacyBucketOwner",
}
for b in policy.get("bindings", []):
    if b.get("role") in rw_roles:
        members = set(b.get("members", []))
        if ksa_principal in members or ns_principal in members:
            sys.exit(0)
sys.exit(1)
EOF
  then
    log "Verified Workload Identity GCS object RW binding on gs://${TRACE_GCS_BUCKET} for ${ksa_name}."
    return 0
  fi

  err "Missing Workload Identity IAM binding on gs://${TRACE_GCS_BUCKET} for KSA ${NAMESPACE}/${ksa_name}."
  cat >&2 <<EOF

================================================================================
ACTION REQUIRED (IAM Safety Guardrail — harness never mutates IAM policies):
Grant GCS object read/write to the EPP Kubernetes ServiceAccount so GKE's
GCS FUSE CSI driver can mount gs://${TRACE_GCS_BUCKET} at /traces:

  gcloud storage buckets add-iam-policy-binding gs://${TRACE_GCS_BUCKET} \\
    --project=${GCP_PROJECT} \\
    --member="${expected_principal}" \\
    --role="roles/storage.objectUser"

(Tip: if Context-Aware Access restricts standard user credentials, add:
  --access-token-file=<(gcloud auth application-default print-access-token)
Or grant namespace-wide for custom NAMESPACE overrides:
  --member="${ns_principal_set}")
================================================================================
EOF
  return 2
}

preflight() {
  validate_local
  setup_gcp_auth

  if [[ -z "${K8S_CONTEXT}" ]]; then
    err "K8S_CONTEXT is empty and no active kubectl current-context is configured."
    return 1
  fi
  log "Checking GKE cluster reachability for K8S_CONTEXT=${K8S_CONTEXT}..."
  kubectl --context="${K8S_CONTEXT}" --request-timeout=15s cluster-info >/dev/null

  log "Checking GCS FUSE CSI driver (gcsfuse.csi.storage.gke.io) on cluster..."
  if ! kubectl --context="${K8S_CONTEXT}" --request-timeout=15s get csidriver gcsfuse.csi.storage.gke.io >/dev/null 2>&1; then
    err "CSIDriver gcsfuse.csi.storage.gke.io not installed on ${K8S_CONTEXT}."
    err "Enable via: gcloud container clusters update <CLUSTER> --region=${GCP_REGION} --project=${GCP_PROJECT} --update-addons=GcsFuseCsiDriver=ENABLED"
    return 1
  fi
  log "Verified CSIDriver gcsfuse.csi.storage.gke.io is present."

  if [[ "${VLLM_DEVICE}" == "gpu" ]]; then
    local gpu_nodes
    gpu_nodes="$(kubectl --context="${K8S_CONTEXT}" get nodes -o json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(int(n.get("status",{}).get("allocatable",{}).get("nvidia.com/gpu","0")) for n in d.get("items",[])))')"
    if [[ "${gpu_nodes}" -lt 1 ]]; then
      err "VLLM_DEVICE=gpu requested, but cluster reports 0 allocatable nvidia.com/gpu (set VLLM_DEVICE=cpu to use CPU fallback)."
      return 1
    fi
    log "Verified cluster has ${gpu_nodes} allocatable nvidia.com/gpu accelerator(s)."
  fi

  local registry_host="${EPP_IMAGE_REPO%%/*}"
  local docker_cfg="${RUN_DIR}/docker-config"
  mkdir -p "${docker_cfg}"
  log "Checking Docker authentication to ${registry_host}..."
  if [[ "${GCLOUD_AUTH_MODE}" == "adc" ]]; then
    gcloud auth application-default print-access-token | DOCKER_CONFIG="${docker_cfg}" docker login -u oauth2accesstoken --password-stdin "https://${registry_host}" >/dev/null
  else
    run_gcloud auth print-access-token | DOCKER_CONFIG="${docker_cfg}" docker login -u oauth2accesstoken --password-stdin "https://${registry_host}" >/dev/null
  fi
  rm -rf "${docker_cfg}"
  log "Verified Docker registry authentication to ${registry_host}."

  if ! run_gcloud storage buckets describe "gs://${TRACE_GCS_BUCKET}" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
    err "Bucket gs://${TRACE_GCS_BUCKET} does not exist yet. Run: ./scripts/run-gke-gates.sh create-bucket"
    return 1
  fi
  log "Verified GCS bucket gs://${TRACE_GCS_BUCKET} exists."

  check_workload_identity_iam
  log "Preflight checks PASSED."
}

build_push() {
  setup_gcp_auth
  local registry_host="${EPP_IMAGE_REPO%%/*}"
  local full_image="${EPP_IMAGE_REPO}:${EPP_IMAGE_TAG}"
  local docker_cfg="${RUN_DIR}/docker-config"
  mkdir -p "${docker_cfg}"

  log "Building EPP image ${full_image} from ${REPO_ROOT}/Dockerfile.epp..."
  docker build \
    --progress=plain \
    --platform linux/amd64 \
    -f "${REPO_ROOT}/Dockerfile.epp" \
    --build-arg COMMIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)" \
    --build-arg BUILD_REF="${EPP_IMAGE_TAG}" \
    -t "${full_image}" \
    "${REPO_ROOT}"

  log "Authenticating Docker (isolated config) with ${registry_host} and pushing ${full_image}..."
  if [[ "${GCLOUD_AUTH_MODE}" == "adc" ]]; then
    gcloud auth application-default print-access-token | DOCKER_CONFIG="${docker_cfg}" docker login -u oauth2accesstoken --password-stdin "https://${registry_host}" >/dev/null
  else
    run_gcloud auth print-access-token | DOCKER_CONFIG="${docker_cfg}" docker login -u oauth2accesstoken --password-stdin "https://${registry_host}" >/dev/null
  fi
  DOCKER_CONFIG="${docker_cfg}" docker push "${full_image}"
  rm -rf "${docker_cfg}"
  log "Successfully built and pushed ${full_image}."
}

# ---------------------------------------------------------------------------
# Cluster Deployment & Trace Compaction Helpers
# ---------------------------------------------------------------------------

resolve_python_cli() {
  local venv_py="${INFERENCE_PERF_DIR}/.venv/bin/python"
  if [[ -x "${venv_py}" ]]; then
    echo "${venv_py}"
  else
    command -v python3
  fi
}

ensure_namespace_and_secret() {
  NAMESPACE_TOUCHED="true"
  kubectl --context="${K8S_CONTEXT}" create namespace "${NAMESPACE}" --dry-run=client -o yaml | \
    kubectl --context="${K8S_CONTEXT}" apply -f - >/dev/null

  if ! kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" get secret epp-trace-hmac-secret >/dev/null 2>&1; then
    log "Creating 64-byte random Kubernetes Secret epp-trace-hmac-secret in ${NAMESPACE}..."
    local rand_secret
    rand_secret="$(python3 -c 'import secrets; print(secrets.token_hex(32), end="")')"
    kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" create secret generic epp-trace-hmac-secret \
      --from-literal="capture_secret=${rand_secret}" >/dev/null
  fi
}

render_runtime_values() {
  local gcs_subpath="$1"
  local plugins_file="$2"
  local out_values="$3"
  local img_registry="${EPP_IMAGE_REPO%%/*}"
  local img_repository="${EPP_IMAGE_REPO#*/}"

  python3 - "${HARNESS_DIR}/values-trace-router.yaml" "${out_values}" \
    "${img_registry}" "${img_repository}" "${EPP_IMAGE_TAG}" \
    "${TRACE_GCS_BUCKET}" "/traces/${gcs_subpath}" "${MODEL_NAME}" "${plugins_file}" <<'EOF'
import sys, yaml

base_path, out_path, reg, repo, tag, bucket, out_dir, model_name, plugins_file = sys.argv[1:]
cfg = yaml.safe_load(open(base_path, encoding="utf-8"))

cfg["router"]["tokenizer"]["modelName"] = model_name
epp = cfg["router"]["epp"]
epp["image"]["registry"] = reg
epp["image"]["repository"] = repo
epp["image"]["tag"] = tag
epp["pluginsConfigFile"] = plugins_file

for vol in epp["volumes"]:
    if vol["name"] == "gcs-traces":
        vol["csi"]["volumeAttributes"]["bucketName"] = bucket

for key in ("trace-gate-plugins.yaml", "no-recorder-plugins.yaml"):
    pcfg = yaml.safe_load(epp["pluginsCustomConfig"][key])
    for pl in pcfg["plugins"]:
        if pl.get("type") == "token-producer":
            pl["parameters"]["modelName"] = model_name
        if pl.get("type") == "kv-block-trace-recorder":
            pl["parameters"]["outputDir"] = out_dir
    epp["pluginsCustomConfig"][key] = yaml.safe_dump(pcfg, sort_keys=False)

with open(out_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
EOF

  "${HELM_BIN}" template "${RELEASE_NAME}" "${EFFECTIVE_CHART_DIR}" \
    --namespace "${NAMESPACE}" \
    -f "${out_values}" >/dev/null
}

wait_for_envoy_ext_proc_ready() {
  local epp_selector="llm-d-router-standalone=${RELEASE_NAME}-epp"
  local expected_ips=""
  local attempt

  for attempt in $(seq 1 90); do
    expected_ips="$(kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" get pods -l "${epp_selector}" -o json | python3 -c '
import json, sys
items = json.load(sys.stdin).get("items", [])
if any(p.get("metadata", {}).get("deletionTimestamp") for p in items):
    sys.exit(0)
ips = sorted(
    p.get("status", {}).get("podIP", "")
    for p in items
    if p.get("status", {}).get("phase") == "Running" and p.get("status", {}).get("podIP")
)
if ips:
    print(",".join(ips))
')"
    if [[ -n "${expected_ips}" ]]; then
      break
    fi
    sleep 1
  done

  if [[ -z "${expected_ips}" ]]; then
    err "Timed out waiting for non-terminating EPP pod IPs (${epp_selector})."
    return 1
  fi

  log "Waiting for Envoy ext_proc cluster to converge on healthy EPP endpoints [${expected_ips}]..."
  start_port_forward "deployment/${RELEASE_NAME}-proxy" "29000:19000" "${RUN_DIR}/pf-envoy-admin.log"
  local converged="false"
  for attempt in $(seq 1 35); do
    if python3 - "http://127.0.0.1:29000/clusters" "${expected_ips}" <<'EOF'
import sys, urllib.request

url, expected_csv = sys.argv[1], sys.argv[2]
expected = set(expected_csv.split(","))
try:
    txt = urllib.request.urlopen(url, timeout=5).read().decode("utf-8", errors="replace")
except Exception:
    sys.exit(1)

all_eps = {}
for line in txt.splitlines():
    if line.startswith("ext_proc::") and "::health_flags::" in line:
        parts = line.split("::")
        if len(parts) >= 4:
            ip = parts[1].split(":")[0]
            all_eps[ip] = parts[3].strip()

if set(all_eps.keys()) == expected and all(v == "healthy" for v in all_eps.values()):
    sys.exit(0)
sys.exit(1)
EOF
    then
      converged="true"
      break
    fi
    sleep 2
  done
  stop_port_forwards

  if [[ "${converged}" != "true" ]]; then
    err "Timed out waiting for Envoy ext_proc endpoints to match [${expected_ips}] and become healthy."
    return 1
  fi
  log "Envoy ext_proc converged on healthy EPP endpoints [${expected_ips}]."
}

deploy_router() {
  local gcs_subpath="$1"
  local plugins_file="${2:-trace-gate-plugins.yaml}"
  local runtime_values="${RUN_DIR}/values-${plugins_file%.yaml}-$(echo "${gcs_subpath}" | tr '/' '_').yaml"

  ensure_helm_deps
  ensure_namespace_and_secret
  render_runtime_values "${gcs_subpath}" "${plugins_file}" "${runtime_values}"

  local rel_status
  rel_status="$("${HELM_BIN}" status "${RELEASE_NAME}" --kube-context "${K8S_CONTEXT}" --namespace "${NAMESPACE}" -o json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("info",{}).get("status",""))' 2>/dev/null || true)"
  if [[ "${rel_status}" == pending-* ]]; then
    log "WARNING: Helm release ${RELEASE_NAME} is stuck in ${rel_status}; uninstalling before re-deploy..."
    "${HELM_BIN}" uninstall "${RELEASE_NAME}" --kube-context "${K8S_CONTEXT}" --namespace "${NAMESPACE}" --wait || true
  fi

  log "Deploying ${RELEASE_NAME} (plugins=${plugins_file}, outputDir=/traces/${gcs_subpath})..."
  "${HELM_BIN}" upgrade --install "${RELEASE_NAME}" "${EFFECTIVE_CHART_DIR}" \
    --kube-context "${K8S_CONTEXT}" \
    --namespace "${NAMESPACE}" \
    -f "${runtime_values}" \
    --wait --timeout 8m

  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" rollout status "deployment/${RELEASE_NAME}-epp" --timeout=5m
  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" rollout status "deployment/${RELEASE_NAME}-proxy" --timeout=5m
  wait_for_envoy_ext_proc_ready
}

deploy_model_backend() {
  local manifest_template="$1"
  local rendered_manifest="${RUN_DIR}/$(basename "${manifest_template}")"
  local epp_svc="${RELEASE_NAME}-epp"

  python3 - "${manifest_template}" "${rendered_manifest}" "${MODEL_NAME}" "${epp_svc}" "${NAMESPACE}" <<'EOF'
import sys
src, dst, model, epp_svc, ns = sys.argv[1:]
txt = open(src, encoding="utf-8").read()
txt = txt.replace("__MODEL_NAME__", model).replace("__EPP_SERVICE_NAME__", epp_svc).replace("__NAMESPACE__", ns)
open(dst, "w", encoding="utf-8").write(txt)
EOF

  log "Applying backend manifest $(basename "${manifest_template}") in ${NAMESPACE}..."
  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" apply -f "${rendered_manifest}"
  local dep_name
  dep_name="$(python3 -c 'import sys,yaml; docs=list(yaml.safe_load_all(open(sys.argv[1]))); print([d["metadata"]["name"] for d in docs if d and d.get("kind")=="Deployment"][0])' "${rendered_manifest}")"
  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" rollout status "deployment/${dep_name}" --timeout=10m
}

seal_and_fetch_gcs_traces() {
  local gcs_subpath="$1"
  local local_raw_dir="$2"
  local min_distinct_pods="${3:-1}"
  local restore_replicas="${4:-true}"
  mkdir -p "${local_raw_dir}"

  log "Flushing active EPP trace segments into gs://${TRACE_GCS_BUCKET}/${gcs_subpath}/..."
  sleep 5
  # Gracefully terminate EPP replicas to guarantee deterministic Close() flush on all pods.
  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" scale "deployment/${RELEASE_NAME}-epp" --replicas=0
  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" wait --for=delete pod -l "llm-d-router-standalone=${RELEASE_NAME}-epp" --timeout=120s >/dev/null 2>&1 || sleep 5
  if [[ "${restore_replicas}" == "true" ]]; then
    kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" scale "deployment/${RELEASE_NAME}-epp" --replicas=2
    kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" rollout status "deployment/${RELEASE_NAME}-epp" --timeout=6m
    wait_for_envoy_ext_proc_ready
  fi

  run_gcloud storage cp "gs://${TRACE_GCS_BUCKET}/${gcs_subpath}/*.jsonl.gz" "${local_raw_dir}/"
  python3 - "${local_raw_dir}" "${min_distinct_pods}" <<'EOF'
import pathlib, re, sys

raw_dir = pathlib.Path(sys.argv[1])
min_pods = int(sys.argv[2])
segs = sorted(raw_dir.glob("*.jsonl.gz"))
if not segs:
    sys.exit("ERROR: zero sealed .jsonl.gz segments downloaded")

# Segment naming convention (<pod>-<epoch_ms>-<seq>.jsonl.gz)
pods = {re.sub(r"-\d+-\d+\.jsonl\.gz$", "", p.name) for p in segs}
if len(pods) < min_pods:
    sys.exit(
        f"ERROR: expected segments from >= {min_pods} distinct active-active EPP pods, "
        f"got {len(pods)} ({sorted(pods)}) across {[p.name for p in segs]}"
    )
print(f"Verified {len(segs)} sealed .jsonl.gz segment(s) across {len(pods)} EPP pod(s): {sorted(pods)}")
EOF
}

compact_traces() {
  local raw_dir="$1"
  local compacted_dir="$2"
  local report_json="$3"
  local py_bin
  py_bin="$(resolve_python_cli)"
  mkdir -p "${compacted_dir}"

  log "Compacting ${raw_dir} -> ${compacted_dir} via epp_trace_compact..."
  (
    cd "${RUN_DIR}"
    PYTHONPATH="${INFERENCE_PERF_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${py_bin}" \
      -m inference_perf.datagen.replay.epp_trace_compact \
      --input-dir "${raw_dir}" \
      --output-dir "${compacted_dir}"
  )
  python3 - "${raw_dir}" "${compacted_dir}" "${report_json}" <<'EOF'
import json, pathlib, sys
raw_dir, out_dir, report_path = sys.argv[1:]
traces = sorted(p.name for p in pathlib.Path(out_dir).glob("*.json"))
summary = {
    "input_dir": raw_dir,
    "output_dir": out_dir,
    "compacted_sessions": len(traces),
    "trace_files": traces,
}
pathlib.Path(report_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
EOF
}

run_inference_perf_replay() {
  local trace_dir="$1"
  local base_url="$2"
  local num_sessions="$3"
  local concurrent_sessions="$4"
  local stage_name="$5"
  local cfg_file="${RUN_DIR}/inference-perf-${stage_name}.yaml"
  local py_bin
  py_bin="$(resolve_python_cli)"

  python3 - "${HARNESS_DIR}/inference-perf-replay.yaml.tmpl" "${cfg_file}" \
    "${trace_dir}" "${base_url}" "${MODEL_NAME}" "${num_sessions}" "${concurrent_sessions}" <<'EOF'
import sys
tmpl, dst, tdir, url, model, nsess, csess = sys.argv[1:]
nworkers = "1" if int(csess) <= 1 else "2"
txt = open(tmpl, encoding="utf-8").read()
txt = (
    txt.replace("__TRACE_DIR__", tdir)
    .replace("__BASE_URL__", url)
    .replace("__MODEL_NAME__", model)
    .replace("__NUM_SESSIONS__", nsess)
    .replace("__CONCURRENT_SESSIONS__", csess)
    .replace("__NUM_WORKERS__", nworkers)
)
open(dst, "w", encoding="utf-8").write(txt)
EOF

  log "Running inference-perf replay (${stage_name}: sessions=${num_sessions}, concurrency=${concurrent_sessions})..."
  (
    cd "${RUN_DIR}"
    PYTHONPATH="${INFERENCE_PERF_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${py_bin}" \
      -m inference_perf.main --config_file "${cfg_file}"
  )
}

scrape_pod_metrics() {
  local label_selector="$1"
  local metrics_port="$2"
  local out_file="$3"
  : >"${out_file}"

  local pods
  pods="$(kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" get pods -l "${label_selector}" -o json | python3 -c '
import json, sys
items = json.load(sys.stdin).get("items", [])
active = [
    p["metadata"]["name"]
    for p in items
    if not p.get("metadata", {}).get("deletionTimestamp")
    and p.get("status", {}).get("phase") == "Running"
]
print(" ".join(sorted(active)))
')"
  if [[ -z "${pods}" ]]; then
    err "scrape_pod_metrics: zero active Running pods matched selector ${label_selector}"
    return 1
  fi
  local offset=0
  for pod in ${pods}; do
    local local_port=$((29100 + offset))
    offset=$((offset + 1))
    start_port_forward "pod/${pod}" "${local_port}:${metrics_port}" "${RUN_DIR}/pf-metrics-${pod}.log"
    python3 - "http://127.0.0.1:${local_port}/metrics" >>"${out_file}" <<'EOF'
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=15) as r:
    sys.stdout.write(r.read().decode("utf-8", errors="replace") + "\n")
EOF
    stop_port_forwards
  done
}

reset_vllm_prefix_cache() {
  log "Resetting real vLLM prefix cache before replay..."
  start_port_forward "svc/vllm-trace-gate" "18000:8000" "${RUN_DIR}/pf-vllm-reset.log"
  if python3 - <<'EOF'
import sys, urllib.request
req = urllib.request.Request("http://127.0.0.1:18000/reset_prefix_cache", data=b"", method="POST")
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
EOF
  then
    stop_port_forwards
    log "POST /reset_prefix_cache returned 200 OK."
  else
    stop_port_forwards
    log "POST /reset_prefix_cache unavailable; restarting deployment/vllm-trace-real to guarantee cold KV cache..."
    kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" rollout restart deployment/vllm-trace-real
    kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" rollout status deployment/vllm-trace-real --timeout=10m
  fi
}

# ---------------------------------------------------------------------------
# Gate 3 & Gate 4 End-to-End Orchestration
# ---------------------------------------------------------------------------

run_gate3() {
  preflight
  ensure_namespace_and_secret

  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" delete deployment vllm-trace-real --ignore-not-found --wait=true >/dev/null
  deploy_router "${RUN_ID}/gate3/pass1" "trace-gate-plugins.yaml"
  deploy_model_backend "${HARNESS_DIR}/manifests/vllm-sim.yaml"

  log "=== Gate 3 Pass 1 (Record): 8 multi-turn sessions + 2 cohorts + 2 subagents ==="
  start_port_forward "svc/${RELEASE_NAME}-proxy" "18081:8081" "${RUN_DIR}/pf-envoy-g3-pass1.log"
  python3 "${HARNESS_DIR}/send_live_workload.py" \
    --mode gate3 \
    --base-url "http://127.0.0.1:18081" \
    --model "${MODEL_NAME}" \
    --run-id "${RUN_ID}" \
    --output-json "${RUN_DIR}/gate3_pass1_live.json"
  stop_port_forwards

  seal_and_fetch_gcs_traces "${RUN_ID}/gate3/pass1" "${RUN_DIR}/gate3_pass1_raw" 2 false
  compact_traces "${RUN_DIR}/gate3_pass1_raw" "${RUN_DIR}/gate3_T_recorded" "${RUN_DIR}/gate3_pass1_compact.json"

  log "=== Gate 3 Pass 2 (Replay): Replaying T_recorded through Envoy & re-recording ==="
  deploy_router "${RUN_ID}/gate3/pass2" "trace-gate-plugins.yaml"
  start_port_forward "svc/${RELEASE_NAME}-proxy" "18081:8081" "${RUN_DIR}/pf-envoy-g3-pass2.log"
  run_inference_perf_replay "${RUN_DIR}/gate3_T_recorded" "http://127.0.0.1:18081" 8 4 "gate3-pass2"
  stop_port_forwards

  seal_and_fetch_gcs_traces "${RUN_ID}/gate3/pass2" "${RUN_DIR}/gate3_pass2_raw" 2 false
  compact_traces "${RUN_DIR}/gate3_pass2_raw" "${RUN_DIR}/gate3_T_replay" "${RUN_DIR}/gate3_pass2_compact.json"

  log "=== Gate 3 Oracle Verification ==="
  python3 "${HARNESS_DIR}/verify_oracle.py" gate3 \
    --recorded-dir "${RUN_DIR}/gate3_T_recorded" \
    --replay-dir "${RUN_DIR}/gate3_T_replay" \
    --output-json "${RUN_DIR}/gate3_oracle_report.json"
  log "Gate 3 PASSED! Report: ${RUN_DIR}/gate3_oracle_report.json"
}

run_gate4() {
  preflight
  ensure_namespace_and_secret

  kubectl --context="${K8S_CONTEXT}" -n "${NAMESPACE}" delete deployment vllm-trace-sim --ignore-not-found --wait=true >/dev/null
  deploy_router "${RUN_ID}/gate4/live" "trace-gate-plugins.yaml"
  deploy_model_backend "${HARNESS_DIR}/manifests/vllm-${VLLM_DEVICE}.yaml"

  log "=== Gate 4 Phase 1 (Cold-Cache Reset + Live 6x4 Multi-Turn Workload) ==="
  reset_vllm_prefix_cache
  scrape_pod_metrics "app=vllm-trace-gate" 8000 "${RUN_DIR}/vllm_live_before.prom"

  start_port_forward "svc/${RELEASE_NAME}-proxy" "18081:8081" "${RUN_DIR}/pf-envoy-g4-live.log"
  python3 "${HARNESS_DIR}/send_live_workload.py" \
    --mode gate4 \
    --base-url "http://127.0.0.1:18081" \
    --model "${MODEL_NAME}" \
    --run-id "${RUN_ID}" \
    --concurrency 1 \
    --output-json "${RUN_DIR}/gate4_live_workload.json"
  stop_port_forwards

  scrape_pod_metrics "app=vllm-trace-gate" 8000 "${RUN_DIR}/vllm_live_after.prom"
  seal_and_fetch_gcs_traces "${RUN_ID}/gate4/live" "${RUN_DIR}/gate4_live_raw" 1 false
  compact_traces "${RUN_DIR}/gate4_live_raw" "${RUN_DIR}/gate4_T_recorded" "${RUN_DIR}/gate4_compact.json"

  log "=== Gate 4 Phase 2 (Prefix Cache Reset + Replay Parity Scrape) ==="
  reset_vllm_prefix_cache
  deploy_router "${RUN_ID}/gate4/replay" "trace-gate-plugins.yaml"
  scrape_pod_metrics "app=vllm-trace-gate" 8000 "${RUN_DIR}/vllm_replay_before.prom"

  start_port_forward "svc/${RELEASE_NAME}-proxy" "18081:8081" "${RUN_DIR}/pf-envoy-g4-replay.log"
  run_inference_perf_replay "${RUN_DIR}/gate4_T_recorded" "http://127.0.0.1:18081" 6 1 "gate4-replay"
  stop_port_forwards

  scrape_pod_metrics "app=vllm-trace-gate" 8000 "${RUN_DIR}/vllm_replay_after.prom"

  log "=== Gate 4 Phase 3 (200-Req Concurrency-16 Overhead: Recorder ON vs OFF) ==="
  local epp_selector="llm-d-router-standalone=${RELEASE_NAME}-epp"

  scrape_pod_metrics "${epp_selector}" 9090 "${RUN_DIR}/epp_on_before.prom"
  start_port_forward "svc/${RELEASE_NAME}-proxy" "18081:8081" "${RUN_DIR}/pf-envoy-g4-on.log"
  python3 "${HARNESS_DIR}/send_live_workload.py" \
    --mode overhead \
    --base-url "http://127.0.0.1:18081" \
    --model "${MODEL_NAME}" \
    --run-id "${RUN_ID}-on" \
    --num-requests 200 \
    --concurrency 16 \
    --output-json "${RUN_DIR}/gate4_overhead_on.json"
  stop_port_forwards
  scrape_pod_metrics "${epp_selector}" 9090 "${RUN_DIR}/epp_on_after.prom"

  deploy_router "${RUN_ID}/gate4/off" "no-recorder-plugins.yaml"
  scrape_pod_metrics "${epp_selector}" 9090 "${RUN_DIR}/epp_off_before.prom"
  start_port_forward "svc/${RELEASE_NAME}-proxy" "18081:8081" "${RUN_DIR}/pf-envoy-g4-off.log"
  python3 "${HARNESS_DIR}/send_live_workload.py" \
    --mode overhead \
    --base-url "http://127.0.0.1:18081" \
    --model "${MODEL_NAME}" \
    --run-id "${RUN_ID}-off" \
    --num-requests 200 \
    --concurrency 16 \
    --output-json "${RUN_DIR}/gate4_overhead_off.json"
  stop_port_forwards
  scrape_pod_metrics "${epp_selector}" 9090 "${RUN_DIR}/epp_off_after.prom"

  log "=== Gate 4 Oracle Verification ==="
  python3 "${HARNESS_DIR}/verify_oracle.py" gate4 \
    --vllm-live-before "${RUN_DIR}/vllm_live_before.prom" \
    --vllm-live-after "${RUN_DIR}/vllm_live_after.prom" \
    --vllm-replay-before "${RUN_DIR}/vllm_replay_before.prom" \
    --vllm-replay-after "${RUN_DIR}/vllm_replay_after.prom" \
    --epp-on-before "${RUN_DIR}/epp_on_before.prom" \
    --epp-on-after "${RUN_DIR}/epp_on_after.prom" \
    --epp-off-before "${RUN_DIR}/epp_off_before.prom" \
    --epp-off-after "${RUN_DIR}/epp_off_after.prom" \
    --output-json "${RUN_DIR}/gate4_oracle_report.json"
  log "Gate 4 PASSED! Report: ${RUN_DIR}/gate4_oracle_report.json"
}

usage() {
  local bucket_disp="${TRACE_GCS_BUCKET:-<TRACE_GCS_BUCKET>}"
  local project_disp="${GCP_PROJECT:-<GCP_PROJECT>}"
  local repo_disp="${EPP_IMAGE_REPO:-${GCP_REGION}-docker.pkg.dev/${project_disp}/llm-d/llm-d-router-endpoint-picker}"
  cat <<EOF
Usage: GCP_PROJECT=my-gcp-project TRACE_GCS_BUCKET=my-llm-d-trace-bucket $(basename "$0") <subcommand>

Subcommands:
  preflight       Validate local Helm/oracle tests, GCP/ADC auth, cluster, GCS FUSE CSI, and IAM
  create-bucket   Idempotently create gs://${bucket_disp} in ${project_disp}
  build-push      Build Dockerfile.epp and push to ${repo_disp}:${EPP_IMAGE_TAG}
  validate-local  Run offline helm template + oracle unit tests only
  gate3           Run Gate 3 (2x llm-d-inference-sim closed-loop trace isomorphism)
  gate4           Run Gate 4 (1x real vLLM prefix-cache parity & 200-req p99 overhead)
  all             Run create-bucket -> build-push -> preflight -> gate3 -> gate4
EOF
}

main() {
  local subcmd="${1:-}"
  case "${subcmd}" in
    preflight)
      preflight
      ;;
    create-bucket)
      create_bucket
      ;;
    build-push)
      build_push
      ;;
    validate-local)
      validate_local
      ;;
    gate3)
      run_gate3
      ;;
    gate4)
      run_gate4
      ;;
    all)
      create_bucket
      build_push
      preflight
      KEEP_NAMESPACE="true" run_gate3
      run_gate4
      ;;
    -h|--help|"")
      usage
      [[ -n "${subcmd}" ]] && exit 0 || exit 1
      ;;
    *)
      err "Unknown subcommand: ${subcmd}"
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
