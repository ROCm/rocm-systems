#!/usr/bin/env bash
#
# Tests the ci_images jq in .github/workflows/hipfile-ci-toplevel.yml
# (Precheck job -> "Compute CI image names" step). ci_images consumes
# build_ci_image_matrix.include and emits the keyed map of image names
# used by downstream jobs. This script covers the same 12 scenarios as
# test-build-ci-image-matrix.sh to confirm both stay in sync.
#
# The functions `compute_build_ci_image_matrix` and
# `compute_ci_images_refactored` are verbatim copies of the workflow's
# jq. If you edit either, mirror the change here and re-run.
#
# Usage: bash projects/hipfile/util/test-ci-images.sh
#
set -uo pipefail  # not -e: keep going on failures

# Reference implementation of build_ci_image_matrix (verbatim from the workflow).
compute_build_ci_image_matrix() {
  local ci_matrix="$1"
  printf '%s' "$ci_matrix" | jq -c '
    . as $ci
    | ($ci
        | with_entries(select(.key != "include" and .key != "exclude"))
      ) as $base
    | ($ci.include // []) as $includes
    | ($ci.exclude // []) as $excludes

    | (reduce ($base | to_entries)[] as $axis (
        [{}];
        [ .[] as $combo
        | $axis.value[] as $v
        | $combo + { ($axis.key): $v }
        ]
      )) as $cross

    | ($cross | map(. as $c | select(
        [ $excludes[] | . as $e
        | ($e | to_entries | all($c[.key] == .value))
        ] | any | not
      ))) as $after_exclude

    | ($includes | map(select(. as $incl
        | [ $after_exclude[] | . as $c
          | ($incl | to_entries | all(
              ((.key as $k | $base | has($k)) | not)
              or ($c[.key] == .value)
            ))
          ] | any | not
      ))) as $new_combos

    | ($after_exclude + $new_combos
        | map({supported_platforms, rocm_versions}
              | with_entries(select(.value != null)))
        | map(select(length > 0))
        | unique
      ) as $image_pairs

    | { include: $image_pairs }
  '
}

# REFACTORED ci_images: iterate build_ci_image_matrix.include and emit the
# map. No cross-product/exclude/include logic -- single source of truth.
# jq's null-as-additive-identity for strings handles partial-axis rows
# safely; the resulting map entries have malformed image URLs that fail at
# docker-pull time, surfacing the malformed include to the maintainer.
compute_ci_images_refactored() {
  local build_matrix="$1"
  jq -c -n \
    --arg registry "ghcr.io/testorg" \
    --arg tag "latest-rocm" \
    --argjson build_matrix "$build_matrix" \
    '
      $build_matrix.include
      | map({ platform: .supported_platforms, version: .rocm_versions })
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

# Extract (sp, rv) pairs from a ci_images map for set comparison. Trailing
# dash + empty version part becomes a partial-axis entry.
keys_as_build_matrix() {
  printf '%s' "$1" | jq -c '
    {
      include: (
        keys
        | map(split("-")
              | { supported_platforms: .[0], rocm_versions: .[1] }
              | with_entries(select(.value != null and .value != "")))
      )
    }
  '
}

assert_eq() {
  local label="$1" actual="$2" expected="$3"
  local a_norm e_norm
  a_norm=$(printf '%s' "$actual"   | jq -cS '.include |= sort')
  e_norm=$(printf '%s' "$expected" | jq -cS '.include |= sort')
  if [ "$a_norm" = "$e_norm" ]; then
    printf '  PASS  %s\n        -> %s\n' "$label" "$a_norm"
  else
    printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
      "$label" "$e_norm" "$a_norm"
  fi
}

run_scenario() {
  local label="$1" ci_matrix="$2" expected="$3"
  local bm images extracted
  bm=$(compute_build_ci_image_matrix "$ci_matrix")
  images=$(compute_ci_images_refactored "$bm") || {
    printf '  FAIL  %s (refactored ci_images errored)\n' "$label"
    return
  }
  extracted=$(keys_as_build_matrix "$images")
  assert_eq "$label" "$extracted" "$expected"
}

echo "Test 1: today's ci_matrix (no extra axes)"
run_scenario "  output" \
  '{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[],"exclude":[]}' \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 2: real-world today, include ubuntu/7.13.0"
run_scenario "  output" \
  '{"supported_platforms":["rocky","rocky8","suse","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"}],"exclude":[]}' \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"suse","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 3: cxx=[17] only, exclude (rocky8, cxx:17) -> NO rocky8 image"
run_scenario "  output" \
  '{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}' \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 4: cxx=[17,20], exclude (rocky8, cxx:17) -> rocky8 STILL needed"
run_scenario "  output" \
  '{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}' \
  '{"include":[{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 5: include with extra cxx field"
run_scenario "  output" \
  '{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0","cxx_standard":17}],"exclude":[]}' \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 6: augmenting include {cxx:20}"
run_scenario "  output" \
  '{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"cxx_standard":20}],"exclude":[]}' \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 7: include duplicating an existing combo"
run_scenario "  output" \
  '{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}],"exclude":[]}' \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 8: partial-axis include {sp:rocky} when rocky in base"
run_scenario "  output" \
  '{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}' \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 9: partial-axis include {sp:rocky} when rocky NOT in base -> phantom propagates"
run_scenario "  output" \
  '{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}' \
  '{"include":[{"supported_platforms":"rocky"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 10: exclude (rocky, 7.2.2)"
run_scenario "  output" \
  '{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2","7.3.0"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[]}' \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.3.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.3.0"}]}'

echo
echo "Test 11: exclude then re-include same pair"
run_scenario "  output" \
  '{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}]}' \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 12: future axes, no excludes/includes"
run_scenario "  output" \
  '{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"compiler":["clang","gcc"],"include":[],"exclude":[]}' \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Done."
