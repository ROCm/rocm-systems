#!/usr/bin/env bash
#
# Tests the jq logic that derives build_ci_image_matrix from ci_matrix in
# .github/workflows/hipfile-ci-toplevel.yml (Precheck job -> "Compute matrix
# configurations" step). That logic simulates GHA's matrix expansion
# (cross-product -> exclude -> include with merge semantics), projects each
# resulting combo down to (supported_platforms, rocm_versions), and dedupes.
#
# The function `project_image_matrix` below is a verbatim copy of the
# workflow's jq pipeline. If you edit the workflow's jq, mirror the change
# here and re-run. Conversely, if a test scenario surprises you, fix the
# workflow accordingly.
#
# Usage: bash projects/hipfile/util/test-build-ci-image-matrix.sh
#
set -euo pipefail

project_image_matrix() {
  local ci_matrix="$1"
  printf '%s' "$ci_matrix" | jq -c '
    . as $ci
    | ($ci
        | with_entries(select(.key != "include" and .key != "exclude"))
      ) as $base
    | ($ci.include // []) as $includes
    | ($ci.exclude // []) as $excludes

    # Cross-product of base axes
    | (reduce ($base | to_entries)[] as $axis (
        [{}];
        [ .[] as $combo
        | $axis.value[] as $v
        | $combo + { ($axis.key): $v }
        ]
      )) as $cross

    # Apply excludes (partial match: every pinned field must match the combo)
    | ($cross | map(. as $c | select(
        [ $excludes[] | . as $e
        | ($e | to_entries | all($c[.key] == .value))
        ] | any | not
      ))) as $after_exclude

    # Includes that DO NOT merge into any original combo create new combos.
    # An include can merge if none of its base-axis fields would overwrite
    # the combos value; non-base-axis fields are always mergeable.
    | ($includes | map(select(. as $incl
        | [ $after_exclude[] | . as $c
          | ($incl | to_entries | all(
              ((.key as $k | $base | has($k)) | not)
              or ($c[.key] == .value)
            ))
          ] | any | not
      ))) as $new_combos

    # Project all resulting combos to image-identity axes, drop empties,
    # dedupe.
    | ($after_exclude + $new_combos
        | map({supported_platforms, rocm_versions}
              | with_entries(select(.value != null)))
        | map(select(length > 0))
        | unique
      ) as $image_pairs

    | { include: $image_pairs }
  '
}

assert_eq() {
  local label="$1" actual="$2" expected="$3"
  # Normalize both: sort .include array (order is irrelevant for GHA),
  # then sort object keys (-S), compact (-c).
  local a_norm e_norm
  a_norm=$(printf '%s' "$actual"   | jq -cS '.include |= sort')
  e_norm=$(printf '%s' "$expected" | jq -cS '.include |= sort')
  if [ "$a_norm" = "$e_norm" ]; then
    printf '  PASS  %s\n        -> %s\n' "$label" "$a_norm"
  else
    printf '  FAIL  %s\n        expected: %s\n        got:      %s\n' \
      "$label" "$e_norm" "$a_norm"
    return 1
  fi
}

echo "Test 1: today's ci_matrix (no extra axes) -> all cross-product pairs"
ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 2: real-world today (rocky,rocky8,suse,ubuntu) x (7.2.2), include ubuntu/7.13.0"
ci='{"supported_platforms":["rocky","rocky8","suse","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"}],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"suse","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 3: YOUR SCENARIO -- cxx=[17] only, exclude (rocky8, cxx:17) -> NO rocky8 image"
# rocky8 has only one test leg in ci_matrix and it's excluded. Therefore no
# rocky8 image is needed.
ci='{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 4: contrast scenario -- cxx=[17,20], exclude (rocky8, cxx:17) -> rocky8 STILL needed"
# rocky8 has two test legs; only one is excluded. Image is still required
# for the cxx=20 leg.
ci='{"supported_platforms":["rocky8","ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"exclude":[{"supported_platforms":"rocky8","cxx_standard":17}],"include":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky8","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 5: include with extra cxx field -> still adds the (sp, rv) pair"
# include {ubuntu, 7.13.0, cxx:17} can't merge (rv=7.13.0 not in base), so
# new combo; projection adds the pair.
ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0","cxx_standard":17}],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.13.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 6: augmenting include {cxx:20} -> no new (sp, rv) pairs"
# {cxx:20} merges into every combo (no overwrite), creates no new combos.
ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17],"include":[{"cxx_standard":20}],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 7: include duplicating an existing combo -> dedup, no spurious entry"
ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 8: partial-axis include {sp:rocky} when rocky is in base -> merges, no new"
ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 9: partial-axis include {sp:rocky} when rocky NOT in base -> creates partial entry"
# This is the phantom case the user accepted. We don't paper over it.
ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"include":[{"supported_platforms":"rocky"}],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 10: exclude (rocky, 7.2.2) -> removes that pair"
ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2","7.3.0"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.3.0"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.3.0"}]}'

echo
echo "Test 11: exclude then re-include same pair"
ci='{"supported_platforms":["rocky","ubuntu"],"rocm_versions":["7.2.2"],"exclude":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}],"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"}]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"rocky","rocm_versions":"7.2.2"},{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "Test 12: future axes, no excludes/includes -> dedup down to (sp,rv) projection"
ci='{"supported_platforms":["ubuntu"],"rocm_versions":["7.2.2"],"cxx_standard":[17,20],"compiler":["clang","gcc"],"include":[],"exclude":[]}'
out=$(project_image_matrix "$ci")
assert_eq "  output" "$out" \
  '{"include":[{"supported_platforms":"ubuntu","rocm_versions":"7.2.2"}]}'

echo
echo "All tests passed."
