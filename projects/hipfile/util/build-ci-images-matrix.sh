#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Single source of truth for the "which images do we build" logic used by the
# Precheck job in .github/workflows/hipfile-ci-toplevel.yml ("Compute images to
# build" step). compute_build_set decides the matrix of images to (re)build:
#   - a Dockerfile change rebuilds the whole catalog;
#   - otherwise only nightly rows may need building (other versions reuse their
#     latest-rocm<ver> images from hipfile-image-update.yml), and a nightly row
#     is built only when its base or nvidia image isn't already published.
#
# Two ways to use it:
#   Sourced (CI):  source build-ci-images-matrix.sh
#                  -> sets shell var build_CI_images_matrix from the env vars
#                     CHANGED_DOCKERFILE, FULL_CI_IMAGES_MATRIX, CI_IMAGES
#                     (querying GHCR for nightly rows)
#   Tests:         bash build-ci-images-matrix.sh --test
#
# The only impurity is the registry-existence check, isolated behind
# _image_published so --test can override it and run offline. Everything else is
# pure; because the same code is shipped to CI and exercised by --test, the two
# can never drift.

# --- pure helpers: the jq, one copy each ----------------------------------

# Keep only the nightly rows of a matrix (rocm_versions == "nightly") -- the
# only rows whose target moves daily and so may need (re)building this run.
# Args: matrix (JSON {include:[...]})
filter_nightly_rows() {
  local matrix="$1"
  printf '%s' "${matrix}" \
    | jq -c '{include: [.include[] | select(.rocm_versions == "nightly")]}'
}

# Keep only candidate rows whose "<supported_platforms>-<rocm_versions>" key is
# present in to_build.
# Args: candidate (JSON {include:[...]}) to_build (space-separated keys)
select_to_build() {
  local candidate="$1" to_build="$2"
  printf '%s' "${candidate}" | jq -c --arg tb "${to_build}" \
    '($tb | split(" ") | map(select(length > 0))) as $b
     | {include: [.include[]
         | select((.supported_platforms + "-" + .rocm_versions) as $k | $b | index($k))]}'
}

# --- impurity seam --------------------------------------------------------

# Returns 0 if the image ref already exists in the registry. This is the one
# I/O point; --test overrides it to keep the suite offline.
# Args: image_ref
_image_published() {
  docker buildx imagetools inspect "$1" >/dev/null 2>&1
}

# --- orchestration --------------------------------------------------------

# Emit the matrix of images to (re)build on stdout. Progress notes go to stderr
# so stdout carries only the JSON result.
# Args: changed_dockerfile ("1" or other) full_ci_images_matrix (JSON)
#       ci_images (JSON map: "<sp>-<rv>" -> {ci_image, ci_image_nvidia, ...})
compute_build_set() {
  local changed_dockerfile="$1" full_ci_images_matrix="$2" ci_images="$3"

  if [ "${changed_dockerfile}" = "1" ]; then
    printf '%s' "${full_ci_images_matrix}"
    return
  fi

  local nightly_matrix to_build="" key img img_nvidia
  nightly_matrix=$(filter_nightly_rows "${full_ci_images_matrix}")

  # A nightly row is (re)built when its base OR nvidia image is missing.
  for key in $(printf '%s' "${nightly_matrix}" \
      | jq -r '.include[] | "\(.supported_platforms)-\(.rocm_versions)"'); do
    img=$(printf '%s' "${ci_images}" | jq -r --arg k "${key}" '.[$k].ci_image')
    img_nvidia=$(printf '%s' "${ci_images}" | jq -r --arg k "${key}" '.[$k].ci_image_nvidia')
    if ! _image_published "${img}" || ! _image_published "${img_nvidia}"; then
      echo "Nightly image not yet published, will build: ${key}" >&2
      to_build="${to_build}${key} "
    fi
  done

  select_to_build "${nightly_matrix}" "${to_build}"
}

# --- test suite -----------------------------------------------------------

_assert_eq() {
  local label="$1" actual="$2" expected="$3"
  local a_norm e_norm
  a_norm=$(printf '%s' "${actual}"   | jq -cS '.include |= sort')
  e_norm=$(printf '%s' "${expected}" | jq -cS '.include |= sort')
  if [ "${a_norm}" = "${e_norm}" ]; then
    printf '  PASS  %s\n        -> %s\n' "${label}" "${a_norm}"
  else
    printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
      "${label}" "${e_norm}" "${a_norm}"
    return 1
  fi
}

_run_tests() {
  set -uo pipefail
  local failures=0

  # Offline seam: treat refs listed in _PUBLISHED as already published, so the
  # compute_build_set tests need no registry. Visible to _image_published via
  # bash dynamic scoping.
  local _PUBLISHED=""
  _image_published() {
    case " ${_PUBLISHED} " in
      *" $1 "*) return 0 ;;
      *)        return 1 ;;
    esac
  }

  local CI_IMAGES_FIX='{
    "ubuntu-nightly":{"ci_image":"reg/ubuntu:nightly","ci_image_nvidia":"reg/ubuntu:nightly-nvidia"},
    "rocky-nightly":{"ci_image":"reg/rocky:nightly","ci_image_nvidia":"reg/rocky:nightly-nvidia"}
  }'
  local FULL='{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}'
  local TWO_NIGHTLY='{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"},{"supported_platforms":"rocky","rocm_versions":"nightly"}]}'

  echo "Test 1: filter_nightly_rows keeps only nightly rows"
  _assert_eq "  output" \
    "$(filter_nightly_rows "${FULL}")" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' || failures=$((failures+1))

  echo
  echo "Test 2: filter_nightly_rows with no nightly row -> empty"
  _assert_eq "  output" \
    "$(filter_nightly_rows '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}')" \
    '{"include":[]}' || failures=$((failures+1))

  echo
  echo "Test 3: select_to_build keeps only rows whose key is in to_build"
  _assert_eq "  output" \
    "$(select_to_build "${TWO_NIGHTLY}" 'rocky-nightly')" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"nightly"}]}' || failures=$((failures+1))

  echo
  echo "Test 4: select_to_build with empty to_build -> empty"
  _assert_eq "  output" \
    "$(select_to_build "${TWO_NIGHTLY}" '')" \
    '{"include":[]}' || failures=$((failures+1))

  echo
  echo "Test 5: select_to_build ignores keys not in the candidate set"
  _assert_eq "  output" \
    "$(select_to_build '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' 'rocky-nightly')" \
    '{"include":[]}' || failures=$((failures+1))

  echo
  echo "Test 6: compute_build_set with Dockerfile change -> full catalog"
  _PUBLISHED="reg/ubuntu:nightly reg/ubuntu:nightly-nvidia"
  _assert_eq "  output" \
    "$(compute_build_set "1" "${FULL}" "${CI_IMAGES_FIX}")" \
    "${FULL}" || failures=$((failures+1))

  echo
  echo "Test 7: compute_build_set, nightly fully published -> empty (skip)"
  _PUBLISHED="reg/ubuntu:nightly reg/ubuntu:nightly-nvidia"
  _assert_eq "  output" \
    "$(compute_build_set "0" "${FULL}" "${CI_IMAGES_FIX}")" \
    '{"include":[]}' || failures=$((failures+1))

  echo
  echo "Test 8: compute_build_set, nightly base missing -> build it"
  _PUBLISHED="reg/ubuntu:nightly-nvidia"
  _assert_eq "  output" \
    "$(compute_build_set "0" "${FULL}" "${CI_IMAGES_FIX}")" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' || failures=$((failures+1))

  echo
  echo "Test 9: compute_build_set, nightly nvidia missing -> build it"
  _PUBLISHED="reg/ubuntu:nightly"
  _assert_eq "  output" \
    "$(compute_build_set "0" "${FULL}" "${CI_IMAGES_FIX}")" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' || failures=$((failures+1))

  echo
  echo "Test 10: compute_build_set, two nightly rows, only the unpublished one is built"
  _PUBLISHED="reg/rocky:nightly reg/rocky:nightly-nvidia"
  _assert_eq "  output" \
    "$(compute_build_set "0" "${TWO_NIGHTLY}" "${CI_IMAGES_FIX}")" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' || failures=$((failures+1))

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
# Do NOT set shell options at top level: sourcing must not mutate the caller.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  # SC2034: build_CI_images_matrix is consumed by the sourcing CI step.
  # SC2153: the UPPER_CASE names are CI env vars, not typos of the lowercase args.
  # shellcheck disable=SC2034,SC2153
  build_CI_images_matrix=$(compute_build_set "${CHANGED_DOCKERFILE}" "${FULL_CI_IMAGES_MATRIX}" "${CI_IMAGES}")
else
  case "${1:-}" in
    --test) _run_tests ;;
    *) echo "usage: source build-ci-images-matrix.sh | bash build-ci-images-matrix.sh --test" >&2; exit 2 ;;
  esac
fi
