/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard Benchmark
 *
 * Benchmarks ncclReshard or the ncclReshardWithWindow entry point.
 * The window-mode setup is retained for API regression coverage; the library
 * ignores that caller window and uses the configured copy transport.
 *
 * Usage:
 *   mpirun -np <N> reshard_bench [options]
 *
 * Example:
 *   mpirun -np 6 reshard_bench \
 *       --src-mesh-dims 1,4 --dst-mesh-dims 1,2 \
 *       --tensor-dims 256,128,64 \
 *       --src-shard-dim 0 --dst-shard-dim 0 \
 *       --validate --verbose
 *
 ************************************************************************/

#include "bench_common.h"
#include "bench_common_kernels.h"
#include "bench_metrics.h"

#include "nccl_m2n.h"

#include <exception>

static void printUsage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("\nRequired:\n");
  printf("  --src-mesh-dims <rep>,<shard>    Source mesh dimensions\n");
  printf("  --dst-mesh-dims <rep>,<shard>    Dest mesh dimensions\n");
  printf("  --tensor-dims <d0>,<d1>[,<d2>]   Global tensor dims (2D or 3D)\n");
  printf("  --src-shard-dim <0|1|2>          Source sharding dimension\n");
  printf("  --dst-shard-dim <0|1|2>          Dest sharding dimension (can "
         "differ!)\n");
  printf("\nOptional:\n");
  printf("  --iterations <N>                 Timed iterations (default: 10)\n");
  printf("  --warmup <N>                     Warmup iterations (default: 2)\n");
  printf("  --validate                       Validate data correctness\n");
  printf("  --algorithm <algo>               Legacy compatibility setting: 'ring' "
         "(default) or 'direct'\n");
  printf("  --api <window|default>           Public entry point (default: default)\n");
  printf("  --copy-algorithm <a>             Advanced staging override: "
         "'pack',\n");
  printf("                                   'pipe', or 'direct' (default: pack)\n");
  printf("  --lb-mode <node|uniform>         Load balancing: 'node' "
         "(default) or 'uniform'\n");
  printf("  --verbose                        Enable debug output\n");
  printf("  --print-all-ranks                Print per-rank timing\n");
  printf("  --metrics-output <path>          Write structured rank-max latency samples\n");
  printf("  --use-default-stream             Pass nullptr to the selected "
         "reshard API so the\n");
  printf("                                   library substitutes a stream from "
         "its internal\n");
  printf("                                   pool — exercises the "
         "default-stream code path.\n");
  printf("\nExamples:\n");
  printf("  # Same-dim sharding (partial overlap)\n");
  printf("  mpirun -np 6 %s --src-mesh-dims 1,4 --dst-mesh-dims 1,2 \\\n", prog);
  printf("         --tensor-dims 256,128,64 --src-shard-dim 0 --dst-shard-dim 0 "
         "--validate\n");
  printf("\n  # Cross-dim sharding (all-to-all)\n");
  printf("  mpirun -np 6 %s --src-mesh-dims 1,4 --dst-mesh-dims 1,2 \\\n", prog);
  printf("         --tensor-dims 256,128,64 --src-shard-dim 0 --dst-shard-dim 1 "
         "--validate\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));

  int mpiRank, mpiSize;
  MPICHECK(MPI_Comm_rank(benchMpiWorld(), &mpiRank));
  MPICHECK(MPI_Comm_size(benchMpiWorld(), &mpiSize));

  // Default parameters
  int srcMeshDims[2] = {0, 0};
  int dstMeshDims[2] = {0, 0};
  size_t globalTensorDims[3] = {0, 0, 0};
  int ndims = 0;
  int srcShardDim = -1;
  int dstShardDim = -1;
  int iterations = 10;
  int warmup = 2;
  bool validate = false;
  bool verbose = false;
  bool printAllRanks = false;
  bool useDefaultStream = false;
  ReshardApiMode apiMode = ReshardApiMode::Default;
  const char* algorithm = "RING";
  const char* lbMode = "NODE_AWARE";
  const char* copyAlgorithm = nullptr;
  const char* pMetricsOutput = nullptr;

  BenchArgParser parser(argc, argv, mpiRank);
  parser.meshDims("--src-mesh-dims", srcMeshDims)
      .meshDims("--dst-mesh-dims", dstMeshDims)
      .tensorDims("--tensor-dims", globalTensorDims, &ndims)
      .integer("--src-shard-dim", &srcShardDim)
      .integer("--dst-shard-dim", &dstShardDim)
      .integer("--iterations", &iterations)
      .integer("--warmup", &warmup)
      .flag("--validate", [&] { validate = true; })
      .flag("--verbose", [&] { verbose = true; })
      .flag("--print-all-ranks", [&] { printAllRanks = true; })
      .value("--metrics-output", [&](const char* value) {
        pMetricsOutput = value;
        return BenchParseResult::Success;
      })
      .flag("--use-default-stream", [&] { useDefaultStream = true; })
      .enumValue("--algorithm", &algorithm, {{"direct", "DIRECT"}, {"ring", "RING"}},
          "ERROR: Unknown algorithm '%s'. Use 'ring' or 'direct'\n")
      .apiMode("--api", &apiMode)
    .enumValue("--copy-algorithm", &copyAlgorithm,
               {{"direct", "DIRECT"}, {"pack", "PACK"}, {"pipe", "PIPE"}},
               "ERROR: Unknown copy-algorithm '%s'. Use 'pack', 'pipe', or 'direct'\n")
    .enumValue("--lb-mode", &lbMode, {{"node", "NODE_AWARE"}, {"uniform", "UNIFORM"}},
               "ERROR: Unknown lb-mode '%s'. Use 'uniform' or 'node'\n")
    .help(printUsage);

  int parseExit = benchParseExitCode(parser.parse());
  if (parseExit >= 0) {
    return parseExit;
  }
  benchConfigureCopyAlgorithm(copyAlgorithm);

  if (iterations <= 0 || warmup < 0) {
    if (mpiRank == 0) {
      printf("ERROR: --iterations must be > 0 and --warmup must be >= 0\n");
    }
    MPI_Finalize();
    return 1;
  }

  // Configure reshard library via env vars (applied in ncclM2nInit).
  if (verbose) {
    benchSetEnv("NCCL_RESHARD_LOG_LEVEL", "DEBUG");
  }
  benchSetEnv("NCCL_RESHARD_ALGORITHM", algorithm);
  benchSetEnv("NCCL_RESHARD_LB_MODE", lbMode);
  ncclM2nHandle_t m2nHandle = nullptr;

  // Validate required parameters
  if (srcMeshDims[0] <= 0 || srcMeshDims[1] <= 0 || dstMeshDims[0] <= 0 || dstMeshDims[1] <= 0 || ndims < 2 ||
      ndims > 3 || srcShardDim < 0 || srcShardDim >= ndims || dstShardDim < 0 || dstShardDim >= ndims) {
    if (mpiRank == 0) {
      printf("ERROR: Missing or invalid required parameters\n");
      printUsage(argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  // Calculate total ranks needed
  int srcTotal = srcMeshDims[0] * srcMeshDims[1];
  int dstTotal = dstMeshDims[0] * dstMeshDims[1];
  int totalExpected = srcTotal + dstTotal;

  if (mpiSize != totalExpected) {
    if (mpiRank == 0) {
      printf("ERROR: Expected %d processes (src=%d + dst=%d), got %d\n", totalExpected, srcTotal, dstTotal, mpiSize);
    }
    MPI_Finalize();
    return 1;
  }

  NCCLCHECK(ncclM2nInit(&m2nHandle, NULL));

  // Determine role
  bool isSource = (mpiRank < srcTotal);
  bool isDest = (mpiRank >= srcTotal);

  // Compute shard counts
  int srcShardCount = srcMeshDims[1]; // Sharding is on mesh dim 1
  int dstShardCount = dstMeshDims[1];

  // Compute local tensor dimensions
  size_t srcLocalDims[3], dstLocalDims[3];
  for (int d = 0; d < ndims; d++) {
    if (d == srcShardDim) {
      srcLocalDims[d] = globalTensorDims[d] / srcShardCount;
    } else {
      srcLocalDims[d] = globalTensorDims[d];
    }
    if (d == dstShardDim) {
      dstLocalDims[d] = globalTensorDims[d] / dstShardCount;
    } else {
      dstLocalDims[d] = globalTensorDims[d];
    }
  }

  // Print configuration
  if (mpiRank == 0) {
    printf("=== Tensor Reshard Benchmark ===\n");
    if (apiMode == ReshardApiMode::Default) {
      printf("Using: ncclReshard (default API, copy-algorithm=%s)\n", benchResolvedCopyAlgorithm());
    } else {
      printf("Using: ncclReshardWithWindow (window entry point, copy-algorithm=%s)\n",
             benchResolvedCopyAlgorithm());
    }
    printf("Global tensor: [%zu", globalTensorDims[0]);
    for (int d = 1; d < ndims; d++) {
      printf(", %zu", globalTensorDims[d]);
    }
    printf("] (%dD)\n", ndims);
    printf("Source shard dim: %d, Dest shard dim: %d%s\n", srcShardDim, dstShardDim,
           srcShardDim == dstShardDim ? " (same-dim)" : " (CROSS-DIM!)");
    printf("Source: %d ranks = %d reps x %d shards, local=[%zu", srcTotal, srcMeshDims[0], srcMeshDims[1],
           srcLocalDims[0]);
    for (int d = 1; d < ndims; d++) {
      printf(", %zu", srcLocalDims[d]);
    }
    printf("]\n");
    printf("Dest: %d ranks = %d reps x %d shards, local=[%zu", dstTotal, dstMeshDims[0], dstMeshDims[1],
           dstLocalDims[0]);
    for (int d = 1; d < ndims; d++) {
      printf(", %zu", dstLocalDims[d]);
    }
    printf("]\n");
    printf("Algorithm setting (compatibility): %s\n", algorithm);
    if (strcmp(algorithm, "RING") == 0) {
      printf("Load Balance Mode: %s\n", lbMode);
    }
    printf("Iterations: %d (warmup: %d), Validate: %s\n", iterations, warmup, validate ? "yes" : "no");
    fflush(stdout);
  }

  // Setup CUDA device
  int numDevices;
  CUDACHECK(cudaGetDeviceCount(&numDevices));
  CUDACHECK(cudaSetDevice(mpiRank % numDevices));

  // Create NCCL communicator
  ncclUniqueId worldId;
  if (mpiRank == 0) {
    NCCLCHECK(ncclGetUniqueId(&worldId));
  }
  MPICHECK(MPI_Bcast(&worldId, sizeof(worldId), benchMpiByte(), 0, benchMpiWorld()));

  ncclComm_t worldComm;
  NCCLCHECK(ncclCommInitRank(&worldComm, mpiSize, worldId, mpiRank));

  // Compatibility mode retains its historical symmetric allocation and window
  // setup for API regression coverage. The library ignores that caller window;
  // the default mode uses the ordinary cudaMalloc integration contract.
  size_t srcBufferSize = 1, dstBufferSize = 1;
  for (int d = 0; d < ndims; d++) {
    srcBufferSize *= srcLocalDims[d];
    dstBufferSize *= dstLocalDims[d];
  }
  size_t allocSize = std::max(srcBufferSize, dstBufferSize);
  const size_t NCCL_MIN_ALLOC = 4096;
  if (allocSize < NCCL_MIN_ALLOC) {
    allocSize = NCCL_MIN_ALLOC;
  }

  void* buffer = nullptr;
  if (apiMode == ReshardApiMode::Default) {
    CUDACHECK(cudaMalloc(&buffer, allocSize));
  } else {
    NCCLCHECK(ncclMemAlloc(&buffer, allocSize));
  }
  CUDACHECK(cudaMemset(buffer, 0xDE, allocSize)); // Initialize with pattern

  // Retain the legacy caller-window setup for compatibility-mode coverage.
  ncclWindow_t window = nullptr;
  if (apiMode == ReshardApiMode::Window) {
    NCCLCHECK(benchWaitCommReady(worldComm));
    NCCLCHECK(ncclCommWindowRegister(worldComm, buffer, allocSize, &window, NCCL_WIN_COLL_SYMMETRIC));
  }

  // Setup mesh structures (topology only).  Per-tensor placement is
  // set on the DistTensor below.
  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  srcMesh.ndims = 2;
  srcMesh.dims = srcMeshDims;
  ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
  dstMesh.ndims = 2;
  dstMesh.dims = dstMeshDims;
  dstMesh.startRank = srcTotal;

  // Create CUDA stream
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Stream we pass to the selected reshard API. When
  // --use-default-stream is set we pass nullptr, exercising the
  // library's internal stream pool; otherwise we hand the explicit
  // stream through.  Init / validation kernels still use the
  // explicit stream regardless.
  cudaStream_t reshardStream = useDefaultStream ? (cudaStream_t)0 : stream;

  // Initialize source data for validation
  if (isSource && validate) {
    // Compute shard index: position within shard dimension of mesh
    int localRank = mpiRank - srcMesh.startRank;
    int shardIdx = localRank % srcMeshDims[1]; // Shard dim is mesh dim 1

    benchInitSourceData((char*)buffer, srcLocalDims, ndims, srcShardDim, shardIdx, srcShardCount, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  MPICHECK(MPI_Barrier(benchMpiWorld()));

  MPICHECK(MPI_Barrier(benchMpiWorld()));

  // Build src/dst descriptors once. dataPtr=NULL is the role signal;
  // localShape metadata is still required on inactive sides.
  size_t srcLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  int srcPlacements[] = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_SHARD(srcShardDim)};
  ncclDistTensor_t srcTensor = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  srcTensor.dataPtr = isSource ? buffer : nullptr;
  srcTensor.localShape = srcLocalShape;
  srcTensor.ndims = ndims;
  srcTensor.dtype = ncclInt8; // bench validates byte patterns
  srcTensor.mesh = &srcMesh;
  srcTensor.placements = srcPlacements;
  for (int d = 0; d < ndims; d++) {
    srcTensor.localShape[d] = srcLocalDims[d];
  }

  size_t dstLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
  int dstPlacements[] = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_SHARD(dstShardDim)};
  ncclDistTensor_t dstTensor = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  dstTensor.dataPtr = isDest ? buffer : nullptr;
  dstTensor.localShape = dstLocalShape;
  dstTensor.ndims = ndims;
  dstTensor.dtype = ncclInt8;
  dstTensor.mesh = &dstMesh;
  dstTensor.placements = dstPlacements;
  for (int d = 0; d < ndims; d++) {
    dstTensor.localShape[d] = dstLocalDims[d];
  }

  // Lambda for running one iteration. NCCLCHECK aborts on any non-success
  // return so a contract violation (null window, mismatched offsets, etc.)
  // fails the bench instead of being silently dropped.
  auto runOneIteration = [&]() {
    if (apiMode == ReshardApiMode::Default) {
      NCCLCHECK(ncclReshard(m2nHandle, worldComm, &srcTensor, &dstTensor, reshardStream));
    } else {
      NCCLCHECK(ncclReshardWithWindow(m2nHandle, worldComm, window, &srcTensor, &dstTensor, reshardStream));
    }
  };

  // Finish registration before the first iteration reads the window handle.
  if (apiMode == ReshardApiMode::Window) NCCLCHECK(benchWaitCommReady(worldComm));

  // Warmup
  if (mpiRank == 0) {
    printf("\nRunning %d warmup iterations...\n", warmup);
  }

  for (int i = 0; i < warmup; i++) {
    runOneIteration();
    CUDACHECK(cudaStreamSynchronize(reshardStream));
    MPICHECK(MPI_Barrier(benchMpiWorld()));
  }

  if (mpiRank == 0) {
    printf("Warmup complete.\n");
  }

  // Validation (after warmup). Result is propagated to the process exit
  // code so a corrupted reshard fails the bench instead of silently
  // printing FAILED while returning success.
  int validationRc = 0;
  if (validate) {
    bool localValid = true;

    if (isDest) {
      int localRank = mpiRank - dstMesh.startRank;
      int shardIdx = localRank % dstMeshDims[1];

      localValid = benchValidateDestData((const char*)buffer, dstLocalDims, ndims, dstShardDim, shardIdx, dstShardCount,
                                         mpiRank, stream);
      if (localValid) {
        printf("[Rank %d] VALIDATION PASSED: %zu bytes correct\n", mpiRank, dstBufferSize);
      }
    }

    int localResult = localValid ? 1 : 0;
    int globalResult = 0;
    MPICHECK(MPI_Allreduce(&localResult, &globalResult, 1, benchMpiInt(), benchMpiMin(), benchMpiWorld()));

    if (globalResult == 0) {
      if (mpiRank == 0) {
        printf("\n*** VALIDATION FAILED ***\n\n");
      }
      validationRc = 1;
    } else {
      if (mpiRank == 0) {
        printf("\n*** VALIDATION PASSED ***\n\n");
      }
    }

    // Reset dest buffer for timing runs
    if (isDest) {
      CUDACHECK(cudaMemset(buffer, 0xDE, dstBufferSize));
    }
    MPICHECK(MPI_Barrier(benchMpiWorld()));
  }

  // Timed iterations
  if (mpiRank == 0) {
    printf("\nRunning %d timed iterations...\n", iterations);
  }

  const bool bCollectMetrics = pMetricsOutput != nullptr;
  std::vector<double> localIterationMs;
  int metricsWriteRc = 0;
  if (bCollectMetrics) {
    int localMetricsSetupRc = 0;
    try {
      localIterationMs.resize(iterations);
    } catch (const std::exception& error) {
      fprintf(stderr, "[Rank %d] ERROR: unable to allocate local metrics samples: %s\n", mpiRank, error.what());
      localMetricsSetupRc = 1;
    } catch (...) {
      fprintf(stderr, "[Rank %d] ERROR: unable to allocate local metrics samples: unknown exception\n", mpiRank);
      localMetricsSetupRc = 1;
    }
    MPICHECK(MPI_Allreduce(&localMetricsSetupRc, &metricsWriteRc, 1, benchMpiInt(), benchMpiMax(),
                           benchMpiWorld()));
  }

  MPICHECK(MPI_Barrier(benchMpiWorld()));
  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iterations; iter++) {
    std::chrono::steady_clock::time_point iterationStart;
    if (bCollectMetrics && metricsWriteRc == 0) {
      iterationStart = std::chrono::steady_clock::now();
    }
    runOneIteration();
    CUDACHECK(cudaStreamSynchronize(reshardStream));
    if (bCollectMetrics && metricsWriteRc == 0) {
      const auto iterationEnd = std::chrono::steady_clock::now();
      localIterationMs[iter] = std::chrono::duration<double, std::milli>(iterationEnd - iterationStart).count();
    }
    MPICHECK(MPI_Barrier(benchMpiWorld()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
  double avgTimeMs = elapsedMs / iterations;

  // Compute bandwidth statistics
  size_t totalData = 1;
  for (int d = 0; d < ndims; d++) {
    totalData *= globalTensorDims[d];
  }
  double bandwidthGbps = ((double)totalData / (avgTimeMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

  size_t myData = isSource ? srcBufferSize : dstBufferSize;
  double myBwGbps = ((double)myData / (avgTimeMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

  // Gather statistics
  double bwMin, bwMax, bwSum;
  MPICHECK(MPI_Reduce(&myBwGbps, &bwMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&myBwGbps, &bwMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&myBwGbps, &bwSum, 1, benchMpiDouble(), benchMpiSum(), 0, benchMpiWorld()));

  double timeMin, timeMax;
  MPICHECK(MPI_Reduce(&avgTimeMs, &timeMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&avgTimeMs, &timeMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));

  // Reduce after the timed loop so structured metrics add no collective to
  // the measured region. Each sample is the slowest rank's local interval
  // from immediately before API enqueue through CUDA stream completion; the
  // existing per-iteration MPI barrier is excluded. The clock reads themselves
  // remain inside the legacy elapsed-time region when collection is enabled.
  std::vector<double> rankMaxIterationMs;
  if (bCollectMetrics) {
    if (mpiRank == 0 && metricsWriteRc == 0) {
      try {
        rankMaxIterationMs.resize(iterations);
      } catch (const std::exception& error) {
        fprintf(stderr, "ERROR: unable to allocate metrics samples: %s\n", error.what());
        metricsWriteRc = 1;
      } catch (...) {
        fprintf(stderr, "ERROR: unable to allocate metrics samples: unknown exception\n");
        metricsWriteRc = 1;
      }
    }
    // Coordinate root-only allocation failure before peers enter the reduce.
    MPICHECK(MPI_Bcast(&metricsWriteRc, 1, benchMpiInt(), 0, benchMpiWorld()));
    if (metricsWriteRc == 0) {
      MPICHECK(MPI_Reduce(localIterationMs.data(), mpiRank == 0 ? rankMaxIterationMs.data() : nullptr, iterations,
                          benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
    }
  }

  if (mpiRank == 0 && bCollectMetrics && metricsWriteRc == 0) {
    try {
      BenchMetrics("reshard_bench", mpiSize, iterations)
          .addSamples("rank_max_enqueue_to_cuda_completion", "milliseconds", rankMaxIterationMs,
              {
                  {"interval_start", "immediately_before_reshard_api_call"},
                  {"interval_stop", "after_cuda_stream_synchronize"},
                  {"rank_reduction", "max"},
                  {"mpi_barrier_included", false},
              })
          .write(pMetricsOutput);
    } catch (const std::exception& error) {
      fprintf(stderr, "ERROR: unable to write metrics output '%s': %s\n", pMetricsOutput, error.what());
      metricsWriteRc = 1;
    } catch (...) {
      fprintf(stderr, "ERROR: unable to write metrics output '%s': unknown exception\n", pMetricsOutput);
      metricsWriteRc = 1;
    }
  }
  if (bCollectMetrics) {
    // Propagate root-only JSON/allocation/stream failures before continuing.
    MPICHECK(MPI_Bcast(&metricsWriteRc, 1, benchMpiInt(), 0, benchMpiWorld()));
  }

  // Source-only stats
  double trainerBwForMin = isSource ? myBwGbps : 1e20;
  double trainerBwForMax = isSource ? myBwGbps : -1e20;
  double trainerBwForSum = isSource ? myBwGbps : 0.0;
  double trainerTimeForMin = isSource ? avgTimeMs : 1e20;
  double trainerTimeForMax = isSource ? avgTimeMs : -1e20;

  double trainerBwMin, trainerBwMax, trainerBwSum;
  double trainerTimeMin, trainerTimeMax;
  MPICHECK(MPI_Reduce(&trainerBwForMin, &trainerBwMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerBwForMax, &trainerBwMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerBwForSum, &trainerBwSum, 1, benchMpiDouble(), benchMpiSum(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerTimeForMin, &trainerTimeMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerTimeForMax, &trainerTimeMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));

  // Dest-only stats
  double genBwForMin = isDest ? myBwGbps : 1e20;
  double genBwForMax = isDest ? myBwGbps : -1e20;
  double genBwForSum = isDest ? myBwGbps : 0.0;
  double genTimeForMin = isDest ? avgTimeMs : 1e20;
  double genTimeForMax = isDest ? avgTimeMs : -1e20;

  double genBwMin, genBwMax, genBwSum;
  double genTimeMin, genTimeMax;
  MPICHECK(MPI_Reduce(&genBwForMin, &genBwMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genBwForMax, &genBwMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genBwForSum, &genBwSum, 1, benchMpiDouble(), benchMpiSum(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genTimeForMin, &genTimeMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genTimeForMax, &genTimeMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));

  // Print per-rank stats if requested
  if (printAllRanks) {
    for (int r = 0; r < mpiSize; r++) {
      if (mpiRank == r) {
        printf("[Rank %3d] %s: time=%.3f ms, bw=%.2f GB/s\n", mpiRank, isSource ? "Source" : "Dest  ", avgTimeMs,
               myBwGbps);
        fflush(stdout);
      }
      MPICHECK(MPI_Barrier(benchMpiWorld()));
    }
  }

  // Print summary
  if (mpiRank == 0) {
    printf("\n=================================\n");
    printf("       BENCHMARK RESULTS\n");
    printf("=================================\n");
    printf("Iterations: %d (warmup: %d)\n", iterations, warmup);
    printf("Total data: %zu bytes (%.2f MB)\n", totalData, (double)totalData / (1024.0 * 1024.0));
    printf("Sources: %d ranks, Destinations: %d ranks\n", srcTotal, dstTotal);
    printf("Sharding: src_dim=%d, dst_dim=%d (%s)\n", srcShardDim, dstShardDim,
           srcShardDim == dstShardDim ? "same-dim" : "cross-dim");

    printf("\n--- Overall (all ranks) ---\n");
    printf("Time per iteration (ms):  Min=%.3f  Max=%.3f\n", timeMin, timeMax);
    printf("Bandwidth (GB/s):         Min=%.2f  Max=%.2f  Avg=%.2f\n", bwMin, bwMax, bwSum / mpiSize);

    printf("\n--- Sources only (%d ranks) ---\n", srcTotal);
    printf("Time per iteration (ms):  Min=%.3f  Max=%.3f\n", trainerTimeMin, trainerTimeMax);
    printf("Bandwidth (GB/s):         Min=%.2f  Max=%.2f  Avg=%.2f\n", trainerBwMin, trainerBwMax,
           trainerBwSum / srcTotal);

    printf("\n--- Destinations only (%d ranks) ---\n", dstTotal);
    printf("Time per iteration (ms):  Min=%.3f  Max=%.3f\n", genTimeMin, genTimeMax);
    printf("Bandwidth (GB/s):         Min=%.2f  Max=%.2f  Avg=%.2f\n", genBwMin, genBwMax, genBwSum / dstTotal);

    printf("\n--- Effective bandwidth ---\n");
    printf("Total data throughput: %.2f GB/s\n", bandwidthGbps);
    printf("=================================\n");
    fflush(stdout);
  }

  // Reshard is asynchronous. Complete the work before releasing its resources.
  CUDACHECK(cudaStreamSynchronize(reshardStream));
  if (window != nullptr) {
    NCCLCHECK(benchWaitCommReady(worldComm));
    NCCLCHECK(ncclCommWindowDeregister(worldComm, window));
  }
  // M2N finalization also releases windows on this communicator.
  NCCLCHECK(benchWaitCommReady(worldComm));
  NCCLCHECK(ncclM2nFinalize(m2nHandle));
  if (apiMode == ReshardApiMode::Default) {
    CUDACHECK(cudaFree(buffer));
  } else {
    NCCLCHECK(ncclMemFree(buffer));
  }
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(benchWaitCommReady(worldComm));
  NCCLCHECK(ncclCommDestroy(worldComm));

  MPICHECK(MPI_Finalize());

  if (mpiRank == 0) {
    if (validationRc == 0 && metricsWriteRc == 0) {
      printf("\nBenchmark completed successfully!\n");
    } else if (validationRc != 0) {
      printf("\nBenchmark completed with VALIDATION FAILURES.\n");
    } else {
      printf("\nBenchmark completed with METRICS OUTPUT FAILURE.\n");
    }
  }

  return validationRc != 0 ? validationRc : metricsWriteRc;
}
