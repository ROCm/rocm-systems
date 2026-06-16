#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Single source of truth for the ci_matrix + full_CI_images_matrix jq used by
# the Precheck job in .github/workflows/hipfile-ci-toplevel.yml ("Compute
# matrix configurations" step).
#
# Two ways to use it:
#   Sourced (CI):  source ci-matrices.sh
#                  -> sets shell vars ci_matrix and full_CI_images_matrix from
#                     the env vars CXX_COMPILER, CXX_STANDARD,
#                     SUPPORTED_PLATFORMS, ROCM_VERSIONS, MATRIX_INCLUDE,
#                     MATRIX_EXCLUDE
#   Tests:         bash ci-matrices.sh --test
#
# Because the same jq is both shipped to CI and exercised by --test, the two
# can never drift.

# --- pure functions: the jq, one copy each --------------------------------

# Build the full strategy.matrix object handed to the test (Linux) job. GHA
# expands the cross-product and applies the include/exclude keys itself.
# Args: cxx_compiler cxx_standard supported_platforms rocm_versions matrix_include matrix_exclude (JSON)
compute_ci_matrix() {
  local cxx_compiler="$1" cxx_standard="$2" supported_platforms="$3" rocm_versions="$4" matrix_include="$5" matrix_exclude="$6"
  jq -c --null-input \
    --argjson cxx_compiler "${cxx_compiler}" \
    --argjson cxx_standard "${cxx_standard}" \
    --argjson supported_platforms "${supported_platforms}" \
    --argjson rocm_versions "${rocm_versions}" \
    --argjson matrix_include "${matrix_include}" \
    --argjson matrix_exclude "${matrix_exclude}" \
    '{
      cxx_compiler: $cxx_compiler,
      cxx_standard: $cxx_standard,
      supported_platforms: $supported_platforms,
      rocm_versions: $rocm_versions,
      include: $matrix_include,
      exclude: $matrix_exclude
    }'
}

# Calculate the catalog of every CI image this run references. It is
# derived by simulating GHA's matrix expansion (cross-product -> exclude
# -> include with merge semantics) on ci_matrix, projecting each resulting 
# combo down to (supported_platforms, rocm_versions), and deduping.
# This way the test-only cxx axes (and any future axes) on ci_matrix
# do not multiply images, and an exclude that removes all test legs for a
# given (platform, version) correctly removes that image. Emitted in
# include-only form so GHA spawns exactly one image-build job per row.
# Arg: ci_matrix (JSON)
compute_full_CI_images_matrix() {
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

  echo "Test 0: compute_ci_matrix emits every axis (incl. cxx_compiler/cxx_standard)"
  # Guards against silently dropping an axis from the jq object: a missing
  # cxx_compiler/cxx_standard here means matrix.cxx_* resolves empty in GHA.
  out=$(compute_ci_matrix \
    '["amdclang++","clang++","g++"]' \
    '[17,20]' \
    '["rocky8","ubuntu"]' \
    '["7.2.2"]' \
    '[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0","cxx_compiler":"amdclang++","cxx_standard":17}]' \
    '[{"supported_platforms":"rocky8","cxx_standard":20}]')
  local ci_matrix_actual ci_matrix_expected
  ci_matrix_actual=$(printf '%s' "${out}" | jq -cS .)
  ci_matrix_expected=$(printf '%s' '{
    "cxx_compiler":["amdclang++","clang++","g++"],
    "cxx_standard":[17,20],
    "supported_platforms":["rocky8","ubuntu"],
    "rocm_versions":["7.2.2"],
    "include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0","cxx_compiler":"amdclang++","cxx_standard":17}],
    "exclude":[{"supported_platforms":"rocky8","cxx_standard":20}]
  }' | jq -cS .)
  if [ "${ci_matrix_actual}" = "${ci_matrix_expected}" ]; then
    printf '  PASS    compute_ci_matrix\n        -> %s\n' "${ci_matrix_actual}"
  else
    printf '  FAIL    compute_ci_matrix\n        expected: %s\n        got:      %s\n' \
      "${ci_matrix_expected}" "${ci_matrix_actual}"
    failures=$((failures+1))
  fi

  echo
  echo "Test 1: today's ci_matrix (no extra axes) -> all cross-product pairs"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 2: include-pattern scenario (rocky,rocky8,suse,ubuntu) x (7.2.2), include ubuntu/7.13.0"
  ci='{"supported_platforms":["rocky","rocky8","suse","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"}],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"suse","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 3: cxx=[17] only, exclude (rocky8, cxx:17) -> NO rocky8 image"
  # rocky8 has only one test leg in ci_matrix and it's excluded. Therefore no
  # rocky8 image is needed.
  ci='{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 4: cxx=[17,20], exclude (rocky8, cxx:17) -> rocky8 STILL needed"
  # rocky8 has two test legs; only one is excluded. Image still required for
  # the cxx=20 leg.
  ci='{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 5: include with extra cxx field -> still adds the (sp, rv) pair"
  # include {ubuntu, 7.13.0, cxx:17} can't merge (rv=7.13.0 not in base), so
  # new combo; projection adds the pair.
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0","cxx_standard":17}],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 6: augmenting include {cxx:20} -> no new (sp, rv) pairs"
  # {cxx:20} merges into every combo (no overwrite), creates no new combos.
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"cxx_standard":20}],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 7: include duplicating an existing combo -> dedup, no spurious entry"
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 8: partial-axis include {sp:rocky} when rocky in base -> merges, no new"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 9: partial-axis include {sp:rocky} when rocky NOT in base -> creates partial entry"
  # The phantom case the maintainer accepted. We don't paper over it.
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 10: exclude (rocky, 7.2.2) -> removes that pair"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2","7.3.0"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.3.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.3.0"}]}' || failures=$((failures+1))

  echo
  echo "Test 11: exclude then re-include same pair"
  ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 12: cxx axes, no excludes/includes -> dedup down to (sp,rv) projection"
  ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"cxx_compiler":["amdclang++","clang++","g++"],"include":[],"exclude":[]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 13: production matrix (cxx_compiler x cxx_standard, rocky8/cxx20 excluded)"
  # rocky8 keeps only cxx_standard 17 legs but still has image-needing legs, so every
  # (sp, rv) still resolves to exactly one image despite the cxx fan-out.
  ci='{"cxx_compiler":["amdclang++","clang++","g++"],"cxx_standard":[17,20],"supported_platforms":["rocky","rocky8","suse","ubuntu"],"rocm_versions":["7.2.2"],"include":[],"exclude":[{"supported_platforms":"rocky8","cxx_standard":20}]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"suse","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}' || failures=$((failures+1))

  echo
  echo "Test 14: production exclude-pattern (ubuntu-only 7.13.0/nightly via excludes)"
  # Mirrors hipfile-ci-toplevel.yml: 7.13.0 and nightly live in the base
  # ROCM_VERSIONS axis and are removed from every non-ubuntu distro via
  # supported_platforms+rocm_versions excludes, with the cxx fan-out present.
  # Each surviving (sp, rv) still resolves to exactly one image.
  ci='{"cxx_compiler":["amdclang++","clang++","g++"],"cxx_standard":[17,20],"supported_platforms":["rocky","rocky8","suse","ubuntu"],"rocm_versions":["7.2.2","7.13.0","nightly"],"include":[],"exclude":[{"supported_platforms":"rocky8","cxx_standard":20},{"supported_platforms":"rocky","rocm_versions":"7.13.0"},{"supported_platforms":"rocky","rocm_versions":"nightly"},{"supported_platforms":"rocky8","rocm_versions":"7.13.0"},{"supported_platforms":"rocky8","rocm_versions":"nightly"},{"supported_platforms":"suse","rocm_versions":"7.13.0"},{"supported_platforms":"suse","rocm_versions":"nightly"}]}'
  out=$(compute_full_CI_images_matrix "$ci")
  _assert_eq "  output" "$out" \
    '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"suse","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"nightly"}]}' || failures=$((failures+1))

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
  # SC2034: ci_matrix/full_CI_images_matrix are consumed by the sourcing CI step.
  # SC2153: the UPPER_CASE names are CI env vars, not typos of the lowercase args.
  # shellcheck disable=SC2034,SC2153
  ci_matrix=$(compute_ci_matrix "${CXX_COMPILER}" "${CXX_STANDARD}" "${SUPPORTED_PLATFORMS}" "${ROCM_VERSIONS}" "${MATRIX_INCLUDE}" "${MATRIX_EXCLUDE}")
  # shellcheck disable=SC2034
  full_CI_images_matrix=$(compute_full_CI_images_matrix "${ci_matrix}")
else
  set -euo pipefail
  case "${1:-}" in
    --test) _run_tests ;;
    *) echo "usage: source ci-matrices.sh | bash ci-matrices.sh --test" >&2; exit 2 ;;
  esac
fi
