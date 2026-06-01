#!/usr/bin/env bash
# apply-opencode-patches.sh — apply .patches/*.patch to the opencode submodule.
#
# See .patches/README.md for the patch catalog. Per-patch motivation is
# summarized in the patch header and in each patch's docstring inside
# the changed file.
#
# Every patch must `git apply --check` cleanly against its predecessor's
# output. Any failure short-circuits with a non-zero exit and a message
# naming the offending patch.
#
# Usage:
#   bash experimental/python/perfxpert/scripts/apply-opencode-patches.sh
#
# Environment:
#   PERFXPERT_PATCH_DIR  override the default .patches dir (for testing)
#   PERFXPERT_PATCH_MANIFEST override the default SHA256SUMS manifest
#   PERFXPERT_OPENCODE_DIR override the default submodule dir
#   PERFXPERT_PATCH_CHECK_ONLY=1  run only `git apply --check`, do NOT apply

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFXPERT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PATCH_DIR="${PERFXPERT_PATCH_DIR:-${PERFXPERT_ROOT}/.patches}"
PATCH_MANIFEST="${PERFXPERT_PATCH_MANIFEST:-${PATCH_DIR}/SHA256SUMS}"
OPENCODE_DIR="${PERFXPERT_OPENCODE_DIR:-${PERFXPERT_ROOT}/opencode}"
CHECK_ONLY="${PERFXPERT_PATCH_CHECK_ONLY:-0}"

if [[ ! -d "${PATCH_DIR}" ]]; then
  echo "apply-opencode-patches: patch dir not found: ${PATCH_DIR}" >&2
  exit 2
fi

if [[ ! -d "${OPENCODE_DIR}" ]]; then
  echo "apply-opencode-patches: opencode submodule not found: ${OPENCODE_DIR}" >&2
  echo "  Run: git submodule update --init --recursive" >&2
  exit 2
fi

# Collect patches in lexicographic order. Using a glob here rather than
# `ls | sort` because glob expansion is already lexicographic in bash.
shopt -s nullglob
patches=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
  echo "apply-opencode-patches: no patches found in ${PATCH_DIR} — nothing to do" >&2
  exit 0
fi

if [[ ! -f "${PATCH_MANIFEST}" ]]; then
  echo "apply-opencode-patches: patch manifest not found: ${PATCH_MANIFEST}" >&2
  exit 2
fi

manifest_names=()
while read -r checksum relpath _rest; do
  [[ -n "${checksum}" ]] || continue
  name="$(basename "${relpath}")"
  manifest_names+=( "${name}" )
done < "${PATCH_MANIFEST}"

if [[ ${#manifest_names[@]} -eq 0 ]]; then
  echo "apply-opencode-patches: patch manifest is empty: ${PATCH_MANIFEST}" >&2
  exit 2
fi

for patch in "${patches[@]}"; do
  name="$(basename "${patch}")"
  if ! printf '%s\n' "${manifest_names[@]}" | grep -Fxq "${name}"; then
    echo "apply-opencode-patches: manifest missing entry for ${name}" >&2
    exit 1
  fi
done

for name in "${manifest_names[@]}"; do
  if [[ ! -f "${PATCH_DIR}/${name}" ]]; then
    echo "apply-opencode-patches: manifest references missing patch ${name}" >&2
    exit 1
  fi
done

if [[ ${#manifest_names[@]} -ne ${#patches[@]} ]]; then
  echo "apply-opencode-patches: manifest/patch set mismatch in ${PATCH_DIR}" >&2
  exit 1
fi

if ! (
  cd "${PATCH_DIR}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --check --status "${PATCH_MANIFEST}"
  elif command -v shasum >/dev/null 2>&1; then
    while read -r expected relpath _rest; do
      [[ -n "${expected}" ]] || continue
      actual="$(shasum -a 256 "${relpath}" | awk '{print $1}')"
      [[ "${actual}" == "${expected}" ]] || exit 1
    done < "${PATCH_MANIFEST}"
  else
    echo "apply-opencode-patches: need sha256sum or shasum for manifest verification" >&2
    exit 2
  fi
); then
  echo "apply-opencode-patches: checksum verification failed for ${PATCH_MANIFEST}" >&2
  exit 1
fi

echo "apply-opencode-patches: ${#patches[@]} patches to apply (check_only=${CHECK_ONLY})"
echo "apply-opencode-patches: manifest OK — ${PATCH_MANIFEST}"

cd "${OPENCODE_DIR}"

# Check mode requires a clean submodule since we apply & reset.
if ! git diff --quiet HEAD --; then
  echo "apply-opencode-patches: refusing to run: submodule has uncommitted changes" >&2
  echo "  Run: (cd ${OPENCODE_DIR} && git checkout HEAD -- .)" >&2
  exit 1
fi

applied=()
for patch in "${patches[@]}"; do
  name="$(basename "${patch}")"
  # In both modes we apply in sequence: a later patch may depend on an
  # earlier one, so a pristine `git apply --check` is insufficient.
  echo "  checking  ${name}"
  if ! git apply --check "${patch}"; then
    echo "apply-opencode-patches: FAILED --check for ${name}" >&2
    echo "  already-applied patches: ${applied[*]:-<none>}" >&2
    # Roll back
    git checkout HEAD -- . >/dev/null 2>&1 || true
    exit 1
  fi
  echo "  applying  ${name}"
  if ! git apply "${patch}"; then
    echo "apply-opencode-patches: FAILED apply for ${name}" >&2
    git checkout HEAD -- . >/dev/null 2>&1 || true
    exit 1
  fi
  applied+=( "${name}" )
done

if [[ "${CHECK_ONLY}" == "1" || "${CHECK_ONLY}" == "true" ]]; then
  # Check-only mode: revert everything we just applied so the submodule
  # goes back to pristine state.
  git checkout HEAD -- . >/dev/null 2>&1 || true
  echo "apply-opencode-patches: CHECK OK — ${#patches[@]} patch(es) verified (reverted)"
else
  echo "apply-opencode-patches: OK — ${#patches[@]} patch(es) applied"
fi
