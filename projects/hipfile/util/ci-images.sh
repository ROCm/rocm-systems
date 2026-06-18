#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Single source of truth for the ci_images jq used by the Precheck job in
# .github/workflows/hipfile-ci-toplevel.yml ("Compute CI image names" step).
# ci_images consumes build_ci_image_matrix.include and emits the keyed map of
# image names (and caches) used by the downstream build/test jobs.
#
# Two ways to use it:
#   Sourced (CI):  source ci-images.sh
#                  -> sets shell var ci_images from the env vars REGISTRY_URL,
#                     tag, BUILD_CI_IMAGE_MATRIX
#   Tests:         bash ci-images.sh --test
#
# Because the same jq is both shipped to CI and exercised by --test, the two
# can never drift. The tests are self-contained: they feed literal
# build_ci_image_matrix fixtures into compute_ci_images and assert the map, so
# they do not depend on ci-matrices.sh or on any env vars.

# --- pure functions: the jq, one copy -------------------------------------

# Consume build_ci_image_matrix.include directly: it is already the
# deduplicated set of (platform, version) pairs to build. Single source of
# truth -- no parallel exclude/include logic here. A partial-axis include row
# (missing rocm_versions) produces a malformed image URL (e.g. "...:latest-rocm"
# with no version); the resulting docker-pull failure surfaces the malformed
# include to the maintainer.
# Args: registry tag build_ci_image_matrix (JSON)
compute_ci_images() {
  local registry="$1" tag="$2" build_ci_image_matrix="$3"
  jq -c -n \
    --arg registry "${registry}" \
    --arg tag "${tag}" \
    --argjson build_matrix "${build_ci_image_matrix}" \
    '
      $build_matrix.include
      | map({ platform: .supported_platforms,
              version:  .rocm_versions })
      | reduce .[] as $combo (
          {}; . + {
            ($combo.platform + "-" + $combo.version): (
              ( $combo
                | .image_name    = ($registry + "/hipfile/ais_ci_" + .platform)
                | .image         = (.image_name + ":" + $tag + .version)
                | .cache         = (.image_name + ":latest-rocm" + .version + "-cache")
                | .image_nvidia  = (.image_name + ":" + $tag + .version + "-nvidia")
                | .cache_nvidia  = (.image_name + ":latest-rocm" + .version + "-nvidia-cache")
              ) | {
                ci_image:              (.image        | ascii_downcase),
                ci_image_cache:        (.cache        | ascii_downcase),
                ci_image_nvidia:       (.image_nvidia | ascii_downcase),
                ci_image_nvidia_cache: (.cache_nvidia | ascii_downcase)
              }
            )
          }
        )
    '
}

# --- test suite -----------------------------------------------------------

_assert_eq() {
  local label="$1" actual="$2" expected="$3"
  # ci_images is a plain object map; normalize by sorting object keys (-S),
  # compact (-c).
  local a_norm e_norm
  a_norm=$(printf '%s' "${actual}"   | jq -cS .)
  e_norm=$(printf '%s' "${expected}" | jq -cS .)
  if [ "${a_norm}" = "${e_norm}" ]; then
    printf '  PASS  %s\n        -> %s\n' "${label}" "${a_norm}"
  else
    printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
      "${label}" "${e_norm}" "${a_norm}"
    return 1
  fi
}

_run_tests() {
  local failures=0 bm out

  echo "Test 1: single (ubuntu, 7.2.2) row -> one keyed entry, all 4 URL fields"
  bm='{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'
  out=$(compute_ci_images "ghcr.io/testorg" "latest-rocm" "$bm")
  _assert_eq "  output" "$out" \
    '{"ubuntu-7.2.2":{"ci_image":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2","ci_image_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-cache","ci_image_nvidia":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-nvidia","ci_image_nvidia_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-nvidia-cache"}}' || failures=$((failures+1))

  echo
  echo "Test 2: multiple rows -> one entry per (platform, version)"
  bm='{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"}]}'
  out=$(compute_ci_images "ghcr.io/testorg" "latest-rocm" "$bm")
  _assert_eq "  output" "$out" \
    '{"rocky-7.2.2":{"ci_image":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm7.2.2","ci_image_cache":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm7.2.2-cache","ci_image_nvidia":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm7.2.2-nvidia","ci_image_nvidia_cache":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm7.2.2-nvidia-cache"},"ubuntu-7.13.0":{"ci_image":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.13.0","ci_image_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.13.0-cache","ci_image_nvidia":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.13.0-nvidia","ci_image_nvidia_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.13.0-nvidia-cache"}}' || failures=$((failures+1))

  echo
  echo "Test 3: changed-dockerfile tag (sanitized ref) flows into ci_image URLs"
  # When a DOCKERFILE changes, CI passes a per-ref tag instead of latest-rocm.
  # The cache fields are pinned to latest-rocm regardless (warm-cache reuse).
  bm='{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'
  out=$(compute_ci_images "ghcr.io/testorg" "refs-pull-42-merge-rocm" "$bm")
  _assert_eq "  output" "$out" \
    '{"ubuntu-7.2.2":{"ci_image":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:refs-pull-42-merge-rocm7.2.2","ci_image_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-cache","ci_image_nvidia":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:refs-pull-42-merge-rocm7.2.2-nvidia","ci_image_nvidia_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-nvidia-cache"}}' || failures=$((failures+1))

  echo
  echo "Test 4: uppercase registry/platform are downcased in URLs, but NOT in the map key"
  # Docker refs must be lowercase, hence ascii_downcase on the URLs. The map
  # key is built from the raw matrix values so it round-trips with the GHA
  # format('{0}-{1}', ...) lookup, which also uses raw matrix values.
  bm='{"include":[{"supported_platforms":"Ubuntu","rocm_versions":"7.2.2"}]}'
  out=$(compute_ci_images "ghcr.io/TestOrg" "latest-rocm" "$bm")
  _assert_eq "  output" "$out" \
    '{"Ubuntu-7.2.2":{"ci_image":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2","ci_image_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-cache","ci_image_nvidia":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-nvidia","ci_image_nvidia_cache":"ghcr.io/testorg/hipfile/ais_ci_ubuntu:latest-rocm7.2.2-nvidia-cache"}}' || failures=$((failures+1))

  echo
  echo "Test 5: empty include -> empty map"
  bm='{"include":[]}'
  out=$(compute_ci_images "ghcr.io/testorg" "latest-rocm" "$bm")
  _assert_eq "  output" "$out" '{}' || failures=$((failures+1))

  echo
  echo "Test 6: partial-axis row (missing rocm_versions) -> present entry, malformed URL"
  # A malformed include propagates as a 'platform-' key with a versionless
  # image ref. The pull fails downstream, surfacing the bad include rather
  # than silently dropping it.
  bm='{"include":[{"supported_platforms":"rocky"}]}'
  out=$(compute_ci_images "ghcr.io/testorg" "latest-rocm" "$bm")
  _assert_eq "  output" "$out" \
    '{"rocky-":{"ci_image":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm","ci_image_cache":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm-cache","ci_image_nvidia":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm-nvidia","ci_image_nvidia_cache":"ghcr.io/testorg/hipfile/ais_ci_rocky:latest-rocm-nvidia-cache"}}' || failures=$((failures+1))

  echo
  if [ "${failures}" -eq 0 ]; then
    echo "All tests passed."
  else
    echo "${failures} test(s) FAILED."
    return 1
  fi
}

# --- dispatch -------------------------------------------------------------
# When sourced, ${BASH_SOURCE[0]} (this file) differs from ${0} (the caller's
# shell); when executed they are equal. So "!=" means "we are being sourced".
# Do NOT set shell options here: sourcing must not mutate the caller's shell.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  # SC2034: ci_images is consumed by the sourcing CI step.
  # SC2153: BUILD_CI_IMAGE_MATRIX is a CI env var, not a typo of build_ci_image_matrix.
  # shellcheck disable=SC2034,SC2153
  ci_images=$(compute_ci_images "${REGISTRY_URL}" "${tag}" "${BUILD_CI_IMAGE_MATRIX}")
else
  set -euo pipefail
  case "${1:-}" in
    --test) _run_tests ;;
    *) echo "usage: source ci-images.sh | bash ci-images.sh --test" >&2; exit 2 ;;
  esac
fi
