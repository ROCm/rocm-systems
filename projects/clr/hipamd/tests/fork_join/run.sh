#!/usr/bin/env bash
# Build and run the fork/join command-buffer tests against the locally-built CLR.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLR_INSTALL="${CLR_INSTALL:-/home/agodavar/rocm-systems/projects/clr/build/install}"
ARCH="${ARCH:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || echo gfx950)}"

# The locally-built CLR links against the repo's ROCR runtime (HSA 1.21+, which
# exports hsa_amd_queue_create); put it ahead of the system/opt-rocm HSA.
ROCR_INSTALL="${ROCR_INSTALL:-/home/agodavar/rocm-systems/projects/rocr-runtime/build/install}"
export LD_LIBRARY_PATH="${CLR_INSTALL}/lib:${ROCR_INSTALL}/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

TESTS=(
  test_fork_join_basic
  test_fork_join_wide
  test_fork_join_diamond_in_if
  test_fork_join_child
  test_fork_join_chain
  test_fork_join_nested_cond
  test_fork_join_parallel_while
)

rc=0
for t in "${TESTS[@]}"; do
  echo "=================== $t ==================="
  /opt/rocm/bin/hipcc --offload-arch="${ARCH}" \
      -I"${CLR_INSTALL}/include" \
      "${HERE}/${t}.cpp" -o "${HERE}/${t}" \
      -L"${CLR_INSTALL}/lib" -lamdhip64 2>&1 | tail -20
  if [[ ! -x "${HERE}/${t}" ]]; then
    echo "BUILD FAILED: $t"; rc=1; continue
  fi
  "${HERE}/${t}"
  if [[ $? -ne 0 ]]; then echo "RUN FAILED: $t"; rc=1; fi
done

echo "======================================="
[[ $rc -eq 0 ]] && echo "ALL FORK/JOIN TESTS PASSED" || echo "SOME TESTS FAILED"
exit $rc
