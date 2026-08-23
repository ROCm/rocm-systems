#!/usr/bin/env bash
# Verify GIN-SDMA AllGather manifest files match between two branches.
# Usage (from repo root):
#   ddai-artifacts/scripts/gin-sdma-ag-sync-check.bash [REF_A] [REF_B]
# Defaults: users/dondai/gin-stage3b-sdma-ag vs users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "error: run from inside the rocm-systems git repository" >&2
  exit 1
fi
cd "${REPO_ROOT}"

REF_A="${1:-users/dondai/gin-stage3b-sdma-ag}"
REF_B="${2:-users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip}"
MANIFEST="${REPO_ROOT}/ddai-artifacts/scripts/gin-sdma-ag-manifest.txt"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

for ref in "${REF_A}" "${REF_B}"; do
  if ! git rev-parse --verify -q "${ref}^{commit}" >/dev/null; then
    echo "error: unknown git ref: ${ref}" >&2
    exit 1
  fi
done

echo "GIN-SDMA AllGather manifest sync check"
echo "  A: ${REF_A}"
echo "  B: ${REF_B}"
echo

fail=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%%#*}"
  line="$(echo "${line}" | xargs)"
  [[ -z "${line}" ]] && continue

  if ! git cat-file -e "${REF_A}:${line}" 2>/dev/null; then
    echo "MISSING on ${REF_A}: ${line}"
    fail=1
    continue
  fi
  if ! git cat-file -e "${REF_B}:${line}" 2>/dev/null; then
    echo "MISSING on ${REF_B}: ${line}"
    fail=1
    continue
  fi

  hash_a="$(git rev-parse "${REF_A}:${line}")"
  hash_b="$(git rev-parse "${REF_B}:${line}")"
  if [[ "${hash_a}" != "${hash_b}" ]]; then
    echo "DIFF: ${line}"
    fail=1
  else
    echo "OK:   ${line}"
  fi
done < "${MANIFEST}"

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "FAIL: manifest files differ between ${REF_A} and ${REF_B}" >&2
  exit 1
fi
echo "PASS: all manifest files match"
