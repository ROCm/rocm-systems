#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Run the MPI basic API matrix within an existing multi-node Slurm allocation.
# The selected profile keeps C API coverage proportional to its rank count.
# The four-rank profile runs the core matrix. The eight-rank profile also runs
# the cross-dimension regressions, PACK split/group gates, and the
# benchmark validation matrix. Multi-dimensional placement sweeps run in both
# profiles; the eight-rank profile exercises two nontrivial shard axes.

set -euo pipefail

: "${NCCL_HOME:?NCCL_HOME must be set}"
: "${NCCL_M2N_MPI_TEST_PROFILE:?NCCL_M2N_MPI_TEST_PROFILE must be set}"
: "${NCCL_M2N_MPI_NNODES:?NCCL_M2N_MPI_NNODES must be set}"
: "${NCCL_M2N_MPI_NTASKS_PER_NODE:?NCCL_M2N_MPI_NTASKS_PER_NODE must be set}"

if ! [[ "${NCCL_M2N_MPI_NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NCCL_M2N_MPI_NNODES must be a positive integer." >&2
  exit 2
fi
if ! [[ "${NCCL_M2N_MPI_NTASKS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NCCL_M2N_MPI_NTASKS_PER_NODE must be a positive integer." >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This test runner requires an existing Slurm allocation." >&2
  exit 2
fi
if [[ "${SLURM_NNODES:-}" != "${NCCL_M2N_MPI_NNODES}" ]]; then
  echo "Expected ${NCCL_M2N_MPI_NNODES} allocated nodes, got ${SLURM_NNODES:-unset}." >&2
  exit 2
fi

expectedRanks=$((NCCL_M2N_MPI_NNODES * NCCL_M2N_MPI_NTASKS_PER_NODE))
splitReverseMeshActivated=0
binary="${NCCL_HOME}/bin/basic_api_test_mpi"
if [[ ! -x "${binary}" ]]; then
  echo "Missing executable basic MPI API test: ${binary}" >&2
  exit 1
fi
multiDimBinary="${NCCL_HOME}/bin/multi_dim_test_mpi"
if [[ ! -x "${multiDimBinary}" ]]; then
  echo "Missing executable multi-dimensional MPI test: ${multiDimBinary}" >&2
  exit 1
fi
benchmark="${NCCL_HOME}/bin/reshard_bench"
if [[ ! -x "${benchmark}" ]]; then
  echo "Missing executable M2N benchmark: ${benchmark}" >&2
  exit 1
fi

case "${NCCL_M2N_MPI_TEST_PROFILE}" in
  four_rank) requiredRanks=4 ;;
  eight_rank) requiredRanks=8 ;;
  *)
    echo "Unknown NCCL_M2N_MPI_TEST_PROFILE: ${NCCL_M2N_MPI_TEST_PROFILE}" >&2
    exit 2
    ;;
esac
if [[ "${expectedRanks}" -ne "${requiredRanks}" ]]; then
  echo "The ${NCCL_M2N_MPI_TEST_PROFILE} profile requires ${requiredRanks} ranks." >&2
  exit 2
fi

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
fi

runMpiProgram() {
  local program="$1"
  local log="$2"
  local distribution="${NCCL_M2N_MPI_DISTRIBUTION:-block}"
  local rc
  shift 2

  set +e
  timeout -k 30 600 srun \
    -N "${NCCL_M2N_MPI_NNODES}" \
    -n "${expectedRanks}" \
    --ntasks-per-node "${NCCL_M2N_MPI_NTASKS_PER_NODE}" \
    --distribution="${distribution}" \
    --exclusive \
    --mpi=pmix \
    --kill-on-bad-exit=1 \
    --cpu-bind=none \
    --export=ALL \
    "${program}" "$@" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

runCase() {
  local label="$1"
  shift
  local log
  log="$(mktemp)"

  echo "=== basic_api_test_mpi: ${label} ==="
  if runMpiProgram "${binary}" "${log}" "$@"; then
    :
  else
    local rc=$?
    echo "basic_api_test_mpi [${label}] failed with exit code ${rc}." >&2
    rm -f "${log}"
    return "${rc}"
  fi

  if ! grep -Eq "^worldSize=${expectedRanks}, devices=[1-9][0-9]*," "${log}"; then
    echo "basic_api_test_mpi [${label}] did not run with ${expectedRanks} ranks." >&2
    rm -f "${log}"
    return 1
  fi
  if ! grep -Eq '^\[  PASSED  \] [1-9][0-9]* tests?\.' "${log}"; then
    echo "basic_api_test_mpi [${label}] completed without passing test cases." >&2
    rm -f "${log}"
    return 1
  fi
  rm -f "${log}"
}

runPackSplitCase() {
  local caseFilter="$1"
  local requireSplit="${2:-0}"
  local log
  log="$(mktemp)"

  echo "=== basic_api_test_mpi: ${caseFilter}, PACK split ==="
  # The node-aware load-balance mode makes the split path eligible. Domain
  # sizes stay topology-discovered rather than assuming a particular machine.
  # AUTO_UNIFORM_BCAST=0 keeps a fully replicated transfer on the node-aware
  # mode, which the fully replicated split case needs to reach the split path.
  # The default block rank distribution is deliberate: it keeps each mesh's
  # ranks within a node so the NVLink domains are real. Round-robin placement
  # scatters them one per node and collapses every domain to a single GPU.
  if NCCL_RESHARD_SPLIT_COMM=1 \
     NCCL_RESHARD_AUTO_UNIFORM_BCAST=0 \
     NCCL_RESHARD_LOG_LEVEL=INFO \
     runMpiProgram "${binary}" "${log}" \
       --filter "${caseFilter}" --algorithm ring --api default --copy-algorithm pack --lb-mode node \
       --gtest_filter='Matrix/BasicApiMpiTest.*'; then
    :
  else
    local rc=$?
    echo "basic_api_test_mpi [${caseFilter}, PACK split] failed with exit code ${rc}." >&2
    cat "${log}"
    rm -f "${log}"
    return "${rc}"
  fi

  if ! grep -Eq "^worldSize=${expectedRanks}, devices=[1-9][0-9]*," "${log}" ||
     ! grep -Eq '^\[  PASSED  \] [1-9][0-9]* tests?\.' "${log}" ||
     grep -Fq '[  SKIPPED ]' "${log}"; then
    echo "basic_api_test_mpi [${caseFilter}, PACK split] did not complete a non-skipped passing case." >&2
    cat "${log}"
    rm -f "${log}"
    return 1
  fi
  # Which path a reshard takes is a property of the fabric, not of the rank
  # count, so derive the expectation from the run instead of from the profile.
  # The single-domain host-RMA path returns before the split block is reached,
  # and it activates only when the communicator has exactly one LSA team. Where
  # it announces itself the split path was never reachable, so requiring the
  # split kernel there would assert something impossible. Taking neither path
  # is always a failure, which is what catches a silently disabled split.
  #
  # Match the activation line exactly: four other messages share this prefix
  # and every one of them means the fast path was declined.
  if grep -Fq 'pack-lsa-hput: ranks=' "${log}"; then
    if [[ "${requireSplit}" -eq 1 ]]; then
      echo "basic_api_test_mpi [${caseFilter}, PACK split] did not activate split." >&2
      cat "${log}"
      rm -f "${log}"
      return 1
    fi
    echo "    one NVLink domain spans this allocation; the single-domain" \
         "host-RMA path claimed this case before the split path"
  elif grep -Fq 'split-launch:' "${log}"; then
    if [[ "${caseFilter}" == split_reverse_mesh ]]; then
      splitReverseMeshActivated=1
    fi
  else
    echo "basic_api_test_mpi [${caseFilter}, PACK split] took neither the split path nor the single-domain host-RMA path." >&2
    cat "${log}"
    rm -f "${log}"
    return 1
  fi
  rm -f "${log}"
}

runNonBlockingCase() {
  NCCL_COMM_BLOCKING=0 runCase "$@"
}

runNonBlockingCoverage() {
  echo "=== basic_api_test_mpi: bounded nonblocking coverage ==="
  runNonBlockingCase "nonblocking DIRECT/default API" \
    --filter tiny_contribution --algorithm direct --api default --lb-mode uniform
  runNonBlockingCase "nonblocking PACK/default API" \
    --filter tiny_contribution --algorithm ring --api default --copy-algorithm pack --lb-mode uniform
  runNonBlockingCase "nonblocking PACK/window API" \
    --filter tiny_contribution --algorithm ring --api window --copy-algorithm pack --lb-mode uniform
  NCCL_RESHARD_PIPE_NET_MODE=HOST_RMA runNonBlockingCase "nonblocking PIPE host-RMA/default API" \
    --filter tiny_contribution --algorithm ring --api default --copy-algorithm pipe --lb-mode uniform
  if [[ "${NCCL_M2N_MPI_TEST_PROFILE}" == eight_rank ]]; then
    NCCL_COMM_BLOCKING=0 runPackGroupCase
  fi
  NCCL_COMM_BLOCKING=0 runPackSplitCase split_reverse_mesh "${splitReverseMeshActivated}"
}

runPackGroupCase() {
  local log
  log="$(mktemp)"

  echo "=== basic_api_test_mpi: mixed communicator groups, PACK ==="
  if NCCL_RESHARD_SPLIT_COMM=0 \
     NCCL_RESHARD_LOG_LEVEL=INFO \
     runMpiProgram "${binary}" "${log}" \
       --filter group_mixed_context \
       --gtest_filter='M2nGroupMpiTest.OverlappingCommunicatorsPreserveBucketOrder' \
       --algorithm ring --api default --copy-algorithm pack --lb-mode uniform; then
    :
  else
    local rc=$?
    echo "basic_api_test_mpi [mixed communicator groups, PACK] failed with exit code ${rc}." >&2
    cat "${log}"
    rm -f "${log}"
    return "${rc}"
  fi

  if ! grep -Eq "^worldSize=${expectedRanks}, devices=[1-9][0-9]*," "${log}" ||
     ! grep -Eq '^\[  PASSED  \] [1-9][0-9]* tests?\.' "${log}" ||
     ! grep -Eq 'entries=2 bins=[0-9]+ fusedBins=1 maxBinEntries=2' "${log}" ||
     grep -Fq '[  SKIPPED ]' "${log}"; then
    echo "basic_api_test_mpi [mixed communicator groups, PACK] did not prove group fusion." >&2
    cat "${log}"
    rm -f "${log}"
    return 1
  fi
  rm -f "${log}"
}

runMultiDimCase() {
  local api="$1"
  local grouped="$2"
  local nonblocking="$3"
  local args=(--src-rows 2 --dst-rows 2)
  local log
  local meshCols=$((expectedRanks / 4))
  local expected="MULTI_DIM_GPU_PASS cases=356 api=${api} grouped=${grouped} nonblocking=${nonblocking} src=2x${meshCols} dst=2x${meshCols}"
  if [[ "${api}" == window ]]; then args+=(--window); fi
  if [[ "${grouped}" -eq 1 ]]; then args+=(--grouped); fi
  if [[ "${nonblocking}" -eq 1 ]]; then args+=(--nonblocking); fi
  log="$(mktemp)"
  echo "=== multi_dim_test_mpi: copy=${NCCL_RESHARD_COPY_ALGORITHM} algorithm=${NCCL_RESHARD_ALGORITHM} lb=${NCCL_RESHARD_LB_MODE} api=${api} grouped=${grouped} nonblocking=${nonblocking} ==="
  if runMpiProgram "${multiDimBinary}" "${log}" "${args[@]}"; then
    :
  else
    local rc=$?
    echo "multi_dim_test_mpi failed with exit code ${rc}." >&2
    rm -f "${log}"
    return "${rc}"
  fi
  if ! grep -Fxq "${expected}" "${log}"; then
    echo "multi_dim_test_mpi did not complete all 356 cases with the expected API and meshes: ${expected}" >&2
    rm -f "${log}"
    return 1
  fi
  rm -f "${log}"
}

runMultiDimCoverage() {
  if [[ "${NCCL_M2N_MPI_TEST_PROFILE}" == four_rank ]]; then
    NCCL_RESHARD_COPY_ALGORITHM=DIRECT NCCL_RESHARD_ALGORITHM=RING NCCL_RESHARD_LB_MODE=UNIFORM \
      runMultiDimCase default 0 0
    NCCL_RESHARD_COPY_ALGORITHM=PACK NCCL_RESHARD_ALGORITHM=RING NCCL_RESHARD_LB_MODE=NODE_AWARE \
      runMultiDimCase window 1 0
    return
  fi

  local copy algorithm
  for copy in DIRECT PACK; do
    for algorithm in RING DIRECT; do
      NCCL_RESHARD_COPY_ALGORITHM="${copy}" NCCL_RESHARD_ALGORITHM="${algorithm}" NCCL_RESHARD_LB_MODE=UNIFORM \
        runMultiDimCase default 0 0
      NCCL_RESHARD_COPY_ALGORITHM="${copy}" NCCL_RESHARD_ALGORITHM="${algorithm}" NCCL_RESHARD_LB_MODE=NODE_AWARE \
        runMultiDimCase window 1 0
    done
  done
  NCCL_RESHARD_COPY_ALGORITHM=PACK NCCL_RESHARD_ALGORITHM=RING NCCL_RESHARD_LB_MODE=NODE_AWARE \
    runMultiDimCase default 0 1
  NCCL_RESHARD_COPY_ALGORITHM=PACK NCCL_RESHARD_ALGORITHM=RING NCCL_RESHARD_LB_MODE=NODE_AWARE \
    runMultiDimCase window 0 1
}

runMultiDimCoverage

runCase "mixed communicator groups" --filter group_mixed_context \
  --algorithm ring --api default --lb-mode uniform

runEightRankDefaultMatrix() {
  local log
  log="$(mktemp)"

  echo "=== basic_api_test_mpi: eight-rank default API matrix ==="
  if runMpiProgram "${binary}" "${log}" \
    --algorithm all --api default --lb-mode uniform --min-world 5 --max-world 8 \
    --gtest_filter='Matrix/BasicApiMpiTest.*'; then
    :
  else
    local rc=$?
    echo "basic_api_test_mpi [eight-rank default API matrix] failed with exit code ${rc}." >&2
    rm -f "${log}"
    return "${rc}"
  fi
  if ! grep -Eq "^worldSize=${expectedRanks}, devices=[1-9][0-9]*," "${log}"; then
    echo "basic_api_test_mpi [eight-rank default API matrix] did not run with ${expectedRanks} ranks." >&2
    rm -f "${log}"
    return 1
  fi

  local expectedCasesPerAlgorithm=141
  local ringPasses
  local directPasses
  local totalPasses
  local skipped
  ringPasses=$(grep -F '[       OK ] Matrix/BasicApiMpiTest' "${log}" | grep -c 'RING_default_' || true)
  directPasses=$(grep -F '[       OK ] Matrix/BasicApiMpiTest' "${log}" | grep -c 'DIRECT_default_' || true)
  totalPasses=$(grep -Fc '[       OK ] Matrix/BasicApiMpiTest' "${log}" || true)
  skipped=$(grep -Fc '[  SKIPPED ]' "${log}" || true)
  if [[ "${ringPasses}" -ne "${expectedCasesPerAlgorithm}" ||
        "${directPasses}" -ne "${expectedCasesPerAlgorithm}" ||
        "${totalPasses}" -ne "$((2 * expectedCasesPerAlgorithm))" ||
        "${skipped}" -ne 0 ]]; then
    echo "Unexpected eight-rank default-API matrix: ring=${ringPasses} direct=${directPasses} total=${totalPasses} skipped=${skipped}." >&2
    rm -f "${log}"
    return 1
  fi
  rm -f "${log}"

  runCase "PIPE LSA fanout reuse" \
    --filter pipe_lsa_fanout_reuse --algorithm ring --api window --copy-algorithm pipe \
    --gtest_filter='Matrix/BasicApiMpiTest.*'
}

runBenchmark() {
  local label="$1"
  shift
  local srcLayout="${BENCH_SRC_LAYOUT:-1 reps x 4 shards}"
  local dstLayout="${BENCH_DST_LAYOUT:-1 reps x 4 shards}"
  local log
  log="$(mktemp)"

  echo "=== reshard_bench: ${label} ==="
  if runMpiProgram "${benchmark}" "${log}" "$@"; then
    :
  else
    local rc=$?
    echo "reshard_bench [${label}] failed with exit code ${rc}." >&2
    rm -f "${log}"
    return "${rc}"
  fi
  if ! grep -Fq "Source: 4 ranks = ${srcLayout}" "${log}" ||
     ! grep -Fq "Dest: 4 ranks = ${dstLayout}" "${log}" ||
     ! grep -Fxq '*** VALIDATION PASSED ***' "${log}"; then
    echo "reshard_bench [${label}] did not complete the expected 8-rank validation." >&2
    rm -f "${log}"
    return 1
  fi
  rm -f "${log}"
}

case "${NCCL_M2N_MPI_TEST_PROFILE}" in
  four_rank)
    runCase "ring, all APIs" --algorithm ring --api all --lb-mode uniform
    runCase "direct, all APIs" --algorithm direct --api all --lb-mode uniform
    runCase "pack, default API" --algorithm ring --api default \
      --copy-algorithm pack --lb-mode uniform
    runCase "pack reduced-bucket regressions" --filter pack_reduced_bucket \
      --algorithm ring --api default --copy-algorithm pack --lb-mode uniform \
      --gtest_filter='PackMpiTest.ReducedBucket*'
    NCCL_RESHARD_STAGING_CHANNEL_SIZE=4194304 \
      runCase "staging slot pressure" --filter staging_slot_pressure \
      --algorithm ring --api default --lb-mode uniform
    runPackSplitCase split_tiny_contribution
    runPackSplitCase split_reverse_mesh
    ;;
  eight_rank)
    runCase "all algorithms, source parity" --algorithm all --lb-mode uniform
    runEightRankDefaultMatrix
    runCase "cross-dimension regressions" --filter cross_dim_regression \
      --algorithm all --api all --lb-mode uniform
    runPackSplitCase split_tiny_contribution
    runPackSplitCase split_reverse_mesh
    runPackGroupCase
    for algorithm in ring direct; do
      for dstShardDim in 0 1; do
        runBenchmark "${algorithm}, dst shard ${dstShardDim}" \
          --src-mesh-dims 1,4 --dst-mesh-dims 1,4 --tensor-dims 256,128,64 \
          --src-shard-dim 0 --dst-shard-dim "${dstShardDim}" --validate \
          --algorithm "${algorithm}" --iterations 10 --warmup 2
      done
    done
    # LSA followers pull 12-byte forwarded pieces, which are not a multiple
    # of the 16-byte bulk-copy granule.
    BENCH_DST_LAYOUT="2 reps x 2 shards" \
      runBenchmark "PIPE LSA fan-out, unaligned pieces" \
      --src-mesh-dims 1,4 --dst-mesh-dims 2,2 --tensor-dims 2,24 \
      --src-shard-dim 1 --dst-shard-dim 1 --validate \
      --copy-algorithm pipe --iterations 3 --warmup 1
    ;;
esac

runNonBlockingCoverage
