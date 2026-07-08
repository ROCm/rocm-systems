#!/usr/bin/env bash
# Resolve + download a ROCm dist tarball from TheRock and write .ci-out/rocm.env
# (ROCM_PATH, ROCM_RELEASE) for later stages. Bash (not
# install_rocm_from_artifacts.py) because the head node only has Python 3.9.
#
# Instead of a single fixed channel, this resolves the "best" tarball for a
# requested version using a stability-first cascade. For each channel (in order)
# it tries the multi-arch tarball first, then the specific GPU family:
#
#   1. release      CDN  repo.amd.com/rocm/tarball            (stable X.Y.Z only)
#   2. prerelease   CDN  rocm.prereleases.amd.com/tarball     (rc/prerelease)
#   3. prerelease-artifacts  S3 therock-prerelease-artifacts  (run-id keyed;
#        holds RC builds + the multi-arch tarball before CDN promotion. The run
#        id is discovered from the ROCm/rockrel "multi_arch_release.yml" run for
#        branch release/therock-<MAJOR.MINOR>; requires a GitHub token.)
#   4. nightly      S3   therock-nightly-tarball              (dated alphas)
#
# The FIRST (channel, family) pair that has a matching tarball wins. So a release
# build is preferred even if a newer nightly exists ("greatest, not latest"),
# and within a channel the combined multi-arch tarball is preferred over the
# single-family one.
#
# Tarball naming (all channels): therock-dist-linux-<family>-<version>.tar.gz
# where <family> is e.g. "multiarch" or "gfx950-dcgpu".
#
# Environment:
#   ROCM_RELEASE              Version QUERY. Matched as a prefix, so "7.14"
#                            selects the newest 7.14.* in the first channel that
#                            has it; "7.14.0rc0" pins that exact build; empty =>
#                            newest available per channel. (default: empty)
#   ROCM_AMDGPU_FAMILY        Specific GPU family fallback (default: gfx950-dcgpu)
#   RCCL_ROCM_CHANNELS        Space-separated cascade override
#                            (default: "release prerelease prerelease-artifacts nightly")
#   RCCL_ROCM_TRY_MULTIARCH   1 (default) to try the multiarch tarball first; 0 to skip
#   GH_TOKEN / GITHUB_TOKEN   Token with read access to ROCm/rockrel Actions, used
#                            only for the prerelease-artifacts leg. Without it,
#                            that leg is skipped (with a warning).
#   RCCL_DEVICE_API_CACHE     Cache root (default: /apps/rccl-ci)
#   ROCM_PATH_OVERRIDE        Use this existing ROCm tree; skips resolve+download
#   ROCM_REF_TOKEN            If set, register a reference for GC (see rocm-ref.sh)
#   RCCL_DEVICE_API_WORKDIR / WORKDIR   Workspace root (for .ci-out output)
#   RCCL_ROCM_RESOLVE_ONLY    1 => print the resolved channel/family/version/url
#                            and exit WITHOUT downloading (for testing/preview)
#   RCCL_CI_TRACE             1 => enable `set -x` tracing (kept off by default so
#                            the GitHub token never lands in logs)
#
# Per-channel base-URL overrides (rarely needed):
#   RELEASE_TARBALL_URL, PRERELEASE_TARBALL_URL,
#   PRERELEASE_ARTIFACTS_URL, NIGHTLY_TARBALL_URL, DEV_TARBALL_URL
#
# Legacy: ROCM_RELEASE_CHANNEL (release|prerelease|nightly|dev) still works. When
# set (and RCCL_ROCM_CHANNELS is not), it pins selection to just that single
# channel, preserving the previous one-fixed-channel behavior.

set -euo pipefail
if [[ -n "${RCCL_CI_TRACE:-}" ]]; then set -x; fi

VERSION_QUERY="${ROCM_RELEASE:-}"
SPECIFIC_FAMILY="${ROCM_AMDGPU_FAMILY:-gfx950-dcgpu}"

RELEASE_TARBALL_URL="${RELEASE_TARBALL_URL:-https://repo.amd.com/rocm/tarball}"
PRERELEASE_TARBALL_URL="${PRERELEASE_TARBALL_URL:-https://rocm.prereleases.amd.com/tarball}"
PRERELEASE_ARTIFACTS_URL="${PRERELEASE_ARTIFACTS_URL:-https://therock-prerelease-artifacts.s3.amazonaws.com}"
NIGHTLY_TARBALL_URL="${NIGHTLY_TARBALL_URL:-https://therock-nightly-tarball.s3.amazonaws.com}"
DEV_TARBALL_URL="${DEV_TARBALL_URL:-https://therock-dev-tarball.s3.amazonaws.com}"

ROCKREL_REPO="${ROCKREL_REPO:-ROCm/rockrel}"
ROCKREL_WORKFLOW="${ROCKREL_WORKFLOW:-multi_arch_release.yml}"

# Channel selection:
#   - RCCL_ROCM_CHANNELS set  -> use it verbatim (the cascade).
#   - else ROCM_RELEASE_CHANNEL set (legacy single-channel) -> use just that one,
#     preserving the old "one fixed channel" behavior for existing callers.
#   - else -> the default stability-first cascade.
if [[ -n "${RCCL_ROCM_CHANNELS:-}" ]]; then
  read -ra CHANNELS <<< "${RCCL_ROCM_CHANNELS}"
elif [[ -n "${ROCM_RELEASE_CHANNEL:-}" ]]; then
  read -ra CHANNELS <<< "${ROCM_RELEASE_CHANNEL}"
else
  read -ra CHANNELS <<< "release prerelease prerelease-artifacts nightly"
fi

# Family preference: multiarch first (combined, all-arch), then the specific one.
FAMILIES=()
if [[ "${RCCL_ROCM_TRY_MULTIARCH:-1}" == "1" && "${SPECIFIC_FAMILY}" != "multiarch" ]]; then
  FAMILIES+=("multiarch")
fi
FAMILIES+=("${SPECIFIC_FAMILY}")

# MAJOR.MINOR drives the rockrel release branch (release/therock-<MM>).
MM=""
if [[ -n "${VERSION_QUERY}" ]]; then
  MM="$(printf '%s' "${VERSION_QUERY}" | grep -oE '^[0-9]+\.[0-9]+' || true)"
fi

WORKDIR="${RCCL_DEVICE_API_WORKDIR:-${WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  WORKDIR="$(cd "${script_dir}/../../../.." && pwd)"
fi

CACHE_DIR="${RCCL_DEVICE_API_CACHE:-/apps/rccl-ci}"
env_out="${WORKDIR}/.ci-out/rocm.env"

# --- Helpers ----------------------------------------------------------------

# Read tarball listing text (HTML index or S3 XML) on stdin and print the newest
# matching <version> for the given family, honoring VERSION_QUERY (prefix) and
# an optional stable-only (X.Y.Z) filter. Prints nothing if there is no match.
#   $1 = family   $2 = stable_only(0|1)
_pick_latest_version() {
  local family="$1" stable_only="$2"
  local prefix="therock-dist-linux-${family}-"
  local esc="${VERSION_QUERY//./\\.}"
  # [^"<]* stops at an HTML quote or an XML tag boundary.
  grep -oE "${prefix}[0-9][^\"<]*\.tar\.gz" 2>/dev/null \
    | grep -v -- '-tests-' \
    | sed -E "s/^${prefix}(.+)\.tar\.gz$/\1/" \
    | { if [[ -n "${esc}" ]]; then grep -E "^${esc}([.a-z-]|$)" || true; else cat; fi } \
    | { if [[ "${stable_only}" == "1" ]]; then grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true; else cat; fi } \
    | sort -V | tail -n1
}

# Resolve against a flat listing endpoint (CDN index or S3 root bucket).
#   $1 = base_url  $2 = stable_only  $3 = family  $4 = list-suffix (e.g. "/" or "/?list-type=2&prefix=...")
# Prints "<version>\t<download_url>" or nothing.
_resolve_flat() {
  local base_url="$1" stable_only="$2" family="$3" list_suffix="$4"
  local listing ver
  listing="$(curl -fsSL "${base_url}${list_suffix}" 2>/dev/null || true)"
  [[ -n "${listing}" ]] || return 0
  ver="$(printf '%s' "${listing}" | _pick_latest_version "${family}" "${stable_only}")"
  [[ -n "${ver}" ]] || return 0
  printf '%s\t%s\n' "${ver}" "${base_url}/therock-dist-linux-${family}-${ver}.tar.gz"
}

# Query the newest successful rockrel release run for release/therock-<MM>.
# Token handling is wrapped so `set -x` never echoes it. Prints the run id.
_rockrel_latest_run_id() {
  local token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  [[ -n "${token}" && -n "${MM}" ]] || return 0
  local branch="release/therock-${MM}"
  # Fetch several recent successful runs; the same workflow publishes dev /
  # prerelease / release "types" to DIFFERENT buckets, so we must pick the newest
  # *prerelease* run (identified by its display title) for this bucket.
  local url="https://api.github.com/repos/${ROCKREL_REPO}/actions/workflows/${ROCKREL_WORKFLOW}/runs?branch=${branch}&status=success&per_page=30"
  local json trace_was_on=""
  case "$-" in *x*) trace_was_on=1 ;; esac
  { set +x; } 2>/dev/null
  json="$(curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${url}" 2>/dev/null || true)"
  [[ -n "${trace_was_on}" ]] && set -x
  [[ -n "${json}" ]] || return 0
  printf '%s' "${json}" | python3 -c 'import sys, json
try:
    runs = json.load(sys.stdin).get("workflow_runs", [])
except Exception:
    runs = []
# Runs are newest-first; keep the first whose title marks it a prerelease build.
for r in runs:
    if "prerelease" in (r.get("display_title") or "").lower():
        print(r["id"]); break' 2>/dev/null || true
}

# Resolve against the run-id-keyed prerelease artifacts bucket.
#   $1 = family    Prints "<version>\t<download_url>" or nothing.
_resolve_prerelease_artifacts() {
  local family="$1"
  local run_id
  run_id="$(_rockrel_latest_run_id)"
  if [[ -z "${run_id}" ]]; then
    echo "==> prerelease-artifacts: no rockrel run id (missing GH token or no run for ${MM:-<no version>}); skipping" >&2
    return 0
  fi
  local key_prefix="${run_id}-linux/tarballs/"
  local listing ver
  listing="$(curl -fsSL "${PRERELEASE_ARTIFACTS_URL}/?list-type=2&prefix=${key_prefix}therock-dist-linux-${family}-" 2>/dev/null || true)"
  [[ -n "${listing}" ]] || return 0
  ver="$(printf '%s' "${listing}" | _pick_latest_version "${family}" "0")"
  [[ -n "${ver}" ]] || return 0
  printf '%s\t%s\n' "${ver}" "${PRERELEASE_ARTIFACTS_URL}/${key_prefix}therock-dist-linux-${family}-${ver}.tar.gz"
}

# Dispatch one (channel, family) probe. Prints "<version>\t<url>" or nothing.
_resolve_one() {
  local channel="$1" family="$2"
  case "${channel}" in
    release)              _resolve_flat "${RELEASE_TARBALL_URL}"    1 "${family}" "/" ;;
    prerelease)           _resolve_flat "${PRERELEASE_TARBALL_URL}" 0 "${family}" "/" ;;
    nightly)              _resolve_flat "${NIGHTLY_TARBALL_URL}"    0 "${family}" "/?list-type=2&prefix=therock-dist-linux-${family}-" ;;
    dev)                  _resolve_flat "${DEV_TARBALL_URL}"        0 "${family}" "/?list-type=2&prefix=therock-dist-linux-${family}-" ;;
    prerelease-artifacts) _resolve_prerelease_artifacts "${family}" ;;
    *) echo "ERROR: unknown channel '${channel}' in RCCL_ROCM_CHANNELS" >&2; return 1 ;;
  esac
}

# --- ROCM_PATH_OVERRIDE: explicit tree, no resolve/download -----------------

mkdir -p "${WORKDIR}/.ci-out"
if [[ -n "${ROCM_PATH_OVERRIDE:-}" ]]; then
  ROCM_PATH="${ROCM_PATH_OVERRIDE}"
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: ROCM_PATH_OVERRIDE=${ROCM_PATH} has no bin/hipcc" >&2
    exit 1
  fi
  echo "==> Using ROCM_PATH_OVERRIDE=${ROCM_PATH} (no fetch)"
  {
    printf 'export ROCM_RELEASE=%q\n' "${ROCM_RELEASE:-}"
    printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}"
  } > "${env_out}"
  exit 0
fi

# --- Resolve via the cascade ------------------------------------------------

echo "==> Resolving ROCm: version-query='${VERSION_QUERY:-<latest>}' families='${FAMILIES[*]}'"
echo "==> Cascade: ${CHANNELS[*]}"

WIN_VER="" WIN_URL="" WIN_CHANNEL="" WIN_FAMILY=""
for channel in "${CHANNELS[@]}"; do
  for family in "${FAMILIES[@]}"; do
    res="$(_resolve_one "${channel}" "${family}" || true)"
    if [[ -n "${res}" ]]; then
      WIN_VER="${res%%$'\t'*}"
      WIN_URL="${res#*$'\t'}"
      WIN_CHANNEL="${channel}"
      WIN_FAMILY="${family}"
      echo "==> MATCH: channel=${channel} family=${family} version=${WIN_VER}"
      break 2
    fi
    echo "    miss: channel=${channel} family=${family}"
  done
done

if [[ -z "${WIN_URL}" ]]; then
  echo "ERROR: no ROCm tarball found for version-query='${VERSION_QUERY:-<latest>}'" >&2
  echo "       across channels [${CHANNELS[*]}] and families [${FAMILIES[*]}]" >&2
  exit 1
fi

echo "==> Selected: ${WIN_CHANNEL}/${WIN_FAMILY} ROCm ${WIN_VER}"
echo "==>       url: ${WIN_URL}"

if [[ -n "${RCCL_ROCM_RESOLVE_ONLY:-}" ]]; then
  echo "==> RCCL_ROCM_RESOLVE_ONLY set; not downloading."
  exit 0
fi

# --- Download + cache -------------------------------------------------------

if ! mkdir -p "${CACHE_DIR}/rocm" "${CACHE_DIR}/downloads"; then
  echo "ERROR: cannot create cache dirs under ${CACHE_DIR} (is /apps writable?)" >&2
  exit 1
fi

ROCM_RELEASE="${WIN_VER}"
# Version-only cache path (must match rocm-ref.sh for refcounting/GC). The
# resolved family/channel are recorded in .stamp, not the path.
ROCM_PATH="${CACHE_DIR}/rocm/rocm-${ROCM_RELEASE}"
refs_dir="${ROCM_PATH}.refs"
ref_token="${ROCM_REF_TOKEN:-}"
tarball="$(basename "${WIN_URL}")"

cache_hit() {
  [[ -x "${ROCM_PATH}/bin/hipcc" ]] \
    && grep -qx "family=${WIN_FAMILY}" "${ROCM_PATH}/.stamp" 2>/dev/null \
    && grep -qx "channel=${WIN_CHANNEL}" "${ROCM_PATH}/.stamp" 2>/dev/null \
    && grep -qx "release=${ROCM_RELEASE}" "${ROCM_PATH}/.stamp" 2>/dev/null
}

do_fetch() {
  echo "==> Fetching ROCm ${ROCM_RELEASE} (${WIN_CHANNEL}/${WIN_FAMILY}) from ${WIN_URL}"
  local dl="${CACHE_DIR}/downloads/${tarball}.$$.part"
  rm -f "${dl}"
  # -f makes HTTP errors fail (instead of saving an XML/HTML error body as the tarball).
  curl -fL --retry 5 --retry-delay 5 --retry-connrefused -o "${dl}" "${WIN_URL}"
  rm -rf "${ROCM_PATH}"
  mkdir -p "${ROCM_PATH}"
  tar -xzf "${dl}" -C "${ROCM_PATH}"
  rm -f "${dl}"
  if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    echo "ERROR: extracted ROCm at ${ROCM_PATH} has no bin/hipcc" >&2
    ls -la "${ROCM_PATH}" >&2 || true
    exit 1
  fi
  printf 'release=%s\nfamily=%s\nchannel=%s\n' \
    "${ROCM_RELEASE}" "${WIN_FAMILY}" "${WIN_CHANNEL}" > "${ROCM_PATH}/.stamp"
}

acquire_ref() {
  [[ -n "${ref_token}" ]] || return 0
  mkdir -p "${refs_dir}"
  : > "${refs_dir}/${ref_token}"
  echo "==> Registered ROCm reference ${ref_token} (channel=${WIN_CHANNEL})"
}

# A per-version lock serializes parallel jobs; the inner re-check lets the loser
# reuse the winner's tree. With a ref token we always lock (even on a cache hit)
# so registration is ordered against GC.
if cache_hit && [[ -z "${ref_token}" ]]; then
  echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
else
  exec {lock_fd}>"${CACHE_DIR}/downloads/.lock-${ROCM_RELEASE}"
  flock "${lock_fd}"
  acquire_ref
  if cache_hit; then
    echo "==> Reusing cached ROCm ${ROCM_RELEASE} at ${ROCM_PATH}"
  else
    do_fetch
  fi
  flock -u "${lock_fd}"
fi

{
  printf 'export ROCM_RELEASE=%q\n' "${ROCM_RELEASE}"
  printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}"
} > "${env_out}"
echo "==> Wrote ${env_out}"
