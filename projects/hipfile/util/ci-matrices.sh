#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Single source of truth for the ci_matrix + build_ci_image_matrix jq used by
# the Precheck job in .github/workflows/hipfile-ci-toplevel.yml ("Compute
# matrix configurations" step).
#
# Two ways to use it:
#   Sourced (CI):  source ci-matrices.sh
#                  -> sets shell vars ci_matrix and build_ci_image_matrix from
#                     the env vars SUPPORTED_PLATFORMS, ROCM_VERSIONS,
#                     MATRIX_INCLUDE, MATRIX_EXCLUDE
#   Tests:         bash ci-matrices.sh --test
#
# Because the same jq is both shipped to CI and exercised by --test, the two
# can never drift.

# --- pure functions: the jq, one copy each --------------------------------

# Build the full strategy.matrix object handed to the test (Linux) job. GHA
# expands the cross-product and applies the include/exclude keys itself.
# Args: supported_platforms rocm_versions matrix_include matrix_exclude (JSON)
compute_ci_matrix() {
  local supported_platforms="$1" rocm_versions="$2" matrix_include="$3" matrix_exclude="$4"
  jq -c --null-input \
    --argjson supported_platforms "${supported_platforms}" \
    --argjson rocm_versions "${rocm_versions}" \
    --argjson matrix_include "${matrix_include}" \
    --argjson matrix_exclude "${matrix_exclude}" \
    '{
      supported_platforms: $supported_platforms,
      rocm_versions: $rocm_versions,
      include: $matrix_include,
      exclude: $matrix_exclude
    }'
}

# Derive the image-build matrix from ci_matrix. Simulates GHA's matrix
# expansion (cross-product -> exclude -> include with merge semantics),
# projects each resulting combo down to (supported_platforms, rocm_versions),
# and dedupes. This way future test-only axes on ci_matrix (CXX, compiler,
# ...) do not multiply images, and an exclude that removes all test legs for a
# given (platform, version) correctly removes that image. Emitted in
# include-only form so GHA spawns exactly one image-build job per row.
# Arg: ci_matrix (JSON)
compute_build_ci_image_matrix() {
  local ci_matrix="$1"
  printf '%s' "${ci_matrix}" | jq -c '
    . as $ci
    | ($ci
        | with_entries(select(.key != "include" and .key != "exclude"))
      ) as $base
    | ($ci.include // []) as $includes
    | ($ci.exclude // []) as $excludes

    # Cross-product of all base axes (auto-extends when axes added)
    | (reduce ($base | to_entries)[] as $axis (
        [{}];
        [ .[] as $combo
        | $axis.value[] as $v
        | $combo + { ($axis.key): $v }
        ]
      )) as $cross

    # Apply excludes (partial match: every pinned field must match)
    | ($cross | map(. as $c | select(
        [ $excludes[] | . as $e
        | ($e | to_entries | all($c[.key] == .value))
        ] | any | not
      ))) as $after_exclude

    # Includes that cannot merge into any original combo create new combos.
    # An include merges if none of its base-axis values would overwrite the
    # combos value; non-base-axis fields are always mergeable.
    | ($includes | map(select(. as $incl
        | [ $after_exclude[] | . as $c
          | ($incl | to_entries | all(
              ((.key as $k | $base | has($k)) | not)
              or ($c[.key] == .value)
            ))
          ] | any | not
      ))) as $new_combos

    # Project to image-identity axes, drop empty rows, dedupe
    | ($after_exclude + $new_combos
        | map({supported_platforms, rocm_versions}
              | with_entries(select(.value != null)))
        | map(select(length > 0))
        | unique
      ) as $image_pairs

    | { include: $image_pairs }
  '
}

# --- test suite -----------------------------------------------------------

_assert_eq() {
  local label="$1" actual="$2" expected="$3"
  # Normalize: sort .include array (order is irrelevant for GHA), sort object
  # keys (-S), compact (-c).
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
  local failures=0 ci out

  echo "Test 1: today's ci_matrix (no extra axes) -> all cross-product pairs"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 2: real-world today (rocky,rocky8,suse,ubuntu) x (7.2.2), include ubuntu/7.13.0"
  ci='{"supported_platforms":["rocky","rocky8","suse","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"}],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"suse","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 3: cxx=[17] only, exclude (rocky8, cxx:17) -> NO rocky8 image"
  # rocky8 has only one test leg in ci_matrix and it's excluded. Therefore no
  # rocky8 image is needed.
  ci='{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 4: cxx=[17,20], exclude (rocky8, cxx:17) -> rocky8 STILL needed"
  # rocky8 has two test legs; only one is excluded. Image still required for
  # the cxx=20 leg.
  ci='{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 5: include with extra cxx field -> still adds the (sp, rv) pair"
  # include {ubuntu, 7.13.0, cxx:17} can't merge (rv=7.13.0 not in base), so
  # new combo; projection adds the pair.
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0","cxx_standard":17}],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 6: augmenting include {cxx:20} -> no new (sp, rv) pairs"
  # {cxx:20} merges into every combo (no overwrite), creates no new combos.
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"cxx_standard":20}],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 7: include duplicating an existing combo -> dedup, no spurious entry"
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 8: partial-axis include {sp:rocky} when rocky in base -> merges, no new"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 9: partial-axis include {sp:rocky} when rocky NOT in base -> creates partial entry"
  # The phantom case the maintainer accepted. We don't paper over it.
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 10: exclude (rocky, 7.2.2) -> removes that pair"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2","7.3.0"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.3.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.3.0"}]}' || failures=$((failures+1))

  echo
  echo "Test 11: exclude then re-include same pair"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 12: future axes, no excludes/includes -> dedup down to (sp,rv) projection"
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"compiler":["clang","gcc"],"include":[],"exclude":[]}'
  out=$(compute_build_ci_image_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

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
  # SC2034: ci_matrix/build_ci_image_matrix are consumed by the sourcing CI step.
  # SC2153: the UPPER_CASE names are CI env vars, not typos of the lowercase args.
  # shellcheck disable=SC2034,SC2153
  ci_matrix=$(compute_ci_matrix "${SUPPORTED_PLATFORMS}" "${ROCM_VERSIONS}" "${MATRIX_INCLUDE}" "${MATRIX_EXCLUDE}")
  # shellcheck disable=SC2034
  build_ci_image_matrix=$(compute_build_ci_image_matrix "${ci_matrix}")
else
  set -euo pipefail
  case "${1:-}" in
    --test) _run_tests ;;
    *) echo "usage: source ci-matrices.sh | bash ci-matrices.sh --test" >&2; exit 2 ;;
  esac
fi
