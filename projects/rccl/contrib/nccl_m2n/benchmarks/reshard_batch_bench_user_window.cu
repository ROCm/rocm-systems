/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Batched Transfer Benchmark — User Window API
 *
 * Sweeps over tensor sizes, shard-dim patterns, and communicator counts.
 * For each configuration, compares N tensor reshards issued sequentially
 * versus concurrently across C communicators.
 *
 *   Sequential:  tensor[0]→sync, tensor[1]→sync, …, tensor[N-1]→sync
 *   Concurrent:  launch all N on streams[0..N-1], then sync all streams
 *
 *   Tensor i uses comm[i % numComms] and bufs[i].
 *
 *   Speedup    = seqTime / conTime
 *   Efficiency = Speedup / N           (100% = perfectly parallel)
 *
 * Differences from the legacy reshard_batch_bench.cu (MR !16):
 *   - Calls the ncclReshardWithWindow entry point and retains a caller-managed
 *     window per (comm, buffer) pair for API regression coverage. The library
 *     ignores the window and uses the same configured copy transport as
 *     ncclReshard.
 *   - No per-iteration ncclM2nFinalize; the handle is created once and
 *     reused for the whole sweep.
 *
 * Usage:
 *   mpirun -np <worldSize> reshard_batch_bench_user_window [options]
 *
 * Example (16 ranks, 2 comms, tensor + shard sweep):
 *   mpirun -np 16 reshard_batch_bench_user_window \
 *       --src-mesh-dims 1,8 --dst-mesh-dims 1,8 \
 *       --tensor-dims 256,256:1024,1024:4096,4096 \
 *       --src-shard-dims 0,0 --dst-shard-dims 0,1 \
 *       --num-comms 2 --num-tensors 4 --iterations 20 --warmup 4
 ************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "bench_common.h"
#include "nccl_m2n.h"

// ============================================================================
// Validation kernels (lifted from reshard_bench.cu — same byte pattern)
// ============================================================================

// guide §4.1.4 multi-param: bare `(` / `)` on own lines
// clang-format off
__global__ void
benchInitSourceDataKernel
(
    char    *pBuffer,
    size_t   dim0,
    size_t   dim1,
    size_t   dim2,
    int      nDims,
    size_t   globalStart0,
    size_t   globalStart1,
    size_t   globalStart2,
    size_t   globalDim1,
    size_t   globalDim2,
    unsigned salt
)
// clang-format on
{
  size_t total = dim0 * dim1 * (nDims == 3 ? dim2 : 1);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = (nDims == 3) ? (i % dim2) : 0;
    size_t rem = (nDims == 3) ? (i / dim2) : i;
    size_t d1 = rem % dim1;
    size_t d0 = rem / dim1;
    size_t g0 = globalStart0 + d0;
    size_t g1 = globalStart1 + d1;
    size_t g2 = globalStart2 + d2;
    size_t globalIdx = g0 + g1 * globalDim1 + g2 * globalDim1 * globalDim2;
    pBuffer[i] = (char)((globalIdx + salt) % 256);
  }
}

// guide §4.1.4 multi-param: bare `(` / `)` on own lines
// clang-format off
__global__ void
benchValidateDestDataKernel
(
    const char         *pBuffer,
    size_t              dim0,
    size_t              dim1,
    size_t              dim2,
    int                 nDims,
    size_t              globalStart0,
    size_t              globalStart1,
    size_t              globalStart2,
    size_t              globalDim1,
    size_t              globalDim2,
    unsigned            salt,
    unsigned long long *pErrorCount
)
// clang-format on
{
  size_t total = dim0 * dim1 * (nDims == 3 ? dim2 : 1);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = (nDims == 3) ? (i % dim2) : 0;
    size_t rem = (nDims == 3) ? (i / dim2) : i;
    size_t d1 = rem % dim1;
    size_t d0 = rem / dim1;
    size_t g0 = globalStart0 + d0;
    size_t g1 = globalStart1 + d1;
    size_t g2 = globalStart2 + d2;
    size_t globalIdx = g0 + g1 * globalDim1 + g2 * globalDim1 * globalDim2;
    char expected = (char)((globalIdx + salt) % 256);
    if (pBuffer[i] != expected) atomicAdd(pErrorCount, 1ULL);
  }
}

// guide §4.1.4 multi-param: bare `(` / `)` on own lines
// clang-format off
static void
benchInitSourceData
(
    char         *pBuffer,
    const size_t  pLocalDims[],
    int           nDims,
    int           shardDim,
    int           shardIdx,
    int           shardCount,
    cudaStream_t  stream,
    int           iteration = 0,
    int           bufferId  = 0
)
// clang-format on
{
  size_t globalStart[3] = {0, 0, 0};
  size_t globalDims[3] = {pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1};
  globalStart[shardDim] = shardIdx * pLocalDims[shardDim];
  globalDims[shardDim] = pLocalDims[shardDim] * shardCount;

  unsigned salt = (unsigned)iteration * 37U + (unsigned)bufferId * 131U;

  int blockSize = 256;
  size_t total = pLocalDims[0] * pLocalDims[1] * (nDims == 3 ? pLocalDims[2] : 1);
  int numBlocks = (total + blockSize - 1) / blockSize;
  benchInitSourceDataKernel<<<numBlocks, blockSize, 0, stream>>>(
    pBuffer, pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1, nDims, globalStart[0], globalStart[1],
    globalStart[2], globalDims[1], globalDims[2], salt);
}

// guide §4.1.4 multi-param: bare `(` / `)` on own lines
// clang-format off
static bool
benchValidateDestData
(
    const char   *pBuffer,
    const size_t  pLocalDims[],
    int           nDims,
    int           shardDim,
    int           shardIdx,
    int           shardCount,
    int           worldRank,
    cudaStream_t  stream,
    int           iteration = 0,
    int           bufferId  = 0
)
// clang-format on
{
  size_t globalStart[3] = {0, 0, 0};
  size_t globalDims[3] = {pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1};
  globalStart[shardDim] = shardIdx * pLocalDims[shardDim];
  globalDims[shardDim] = pLocalDims[shardDim] * shardCount;

  unsigned salt = (unsigned)iteration * 37U + (unsigned)bufferId * 131U;

  unsigned long long* pDevErr;
  CUDACHECK(cudaMalloc(&pDevErr, sizeof(unsigned long long)));
  CUDACHECK(cudaMemsetAsync(pDevErr, 0, sizeof(unsigned long long), stream));

  int blockSize = 256;
  size_t total = pLocalDims[0] * pLocalDims[1] * (nDims == 3 ? pLocalDims[2] : 1);
  int numBlocks = (total + blockSize - 1) / blockSize;
  benchValidateDestDataKernel<<<numBlocks, blockSize, 0, stream>>>(
    pBuffer, pLocalDims[0], pLocalDims[1], nDims == 3 ? pLocalDims[2] : 1, nDims, globalStart[0], globalStart[1],
    globalStart[2], globalDims[1], globalDims[2], salt, pDevErr);

  unsigned long long hErr;
  CUDACHECK(cudaMemcpyAsync(&hErr, pDevErr, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaFree(pDevErr));
  if (hErr > 0) {
    printf("[Rank %d] VALIDATION FAILED: %llu mismatches\n", worldRank, hErr);
    return false;
  }
  return true;
}

// ============================================================================
// CLI parsing helpers
// ============================================================================

struct TensorCfg {
  size_t dims[3];
  int nDims;
};
struct ShardCfg {
  int srcSd;
  int dstSd;
};

static bool benchParseSingleTensorDims(const char* value, TensorCfg* out) {
  TensorCfg parsed = {};
  if (!benchParseTensorDims(value, parsed.dims, &parsed.nDims)) return false;
  *out = parsed;
  return true;
}

// "d0,d1:d0,d1:..." → list of TensorCfg
static bool benchParseTensorDimsList(const char* value, std::vector<TensorCfg>* out) {
  if (value == nullptr || out == nullptr || *value == '\0') return false;
  std::vector<TensorCfg> parsed;
  const std::string input(value);
  size_t begin = 0;
  while (begin < input.size()) {
    const size_t end = input.find(':', begin);
    const std::string item = input.substr(begin, end - begin);
    TensorCfg cfg = {};
    if (item.empty() || !benchParseSingleTensorDims(item.c_str(), &cfg)) return false;
    parsed.push_back(cfg);
    if (end == std::string::npos) {
      *out = std::move(parsed);
      return true;
    }
    begin = end + 1;
  }
  return false;
}

static void benchPrintUsage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("\nRequired:\n");
  printf("  --src-mesh-dims <rep>,<shard>          Source mesh\n");
  printf("  --dst-mesh-dims <rep>,<shard>          Dest mesh\n");
  printf("  --tensor-dims <d0,d1[:d0,d1:...]>      Colon-separated tensor "
         "specs\n");
  printf("  --src-shard-dims <s0[,s1,...]>         Comma-separated src shard "
         "dims\n");
  printf("  --dst-shard-dims <d0[,d1,...]>         Comma-separated dst shard "
         "dims\n");
  printf("  (src/dst shard-dim lists are zipped into patterns)\n");
  printf("\nOptional:\n");
  printf("  --num-comms <N>                        Independent NCCL comms "
         "(default: 1)\n");
  printf("  --num-tensors <N>                      Tensors per batch (default: "
         "4)\n");
  printf("  --iterations <N>                       Timed iterations (default: "
         "20)\n");
  printf("  --warmup <N>                           Warmup iterations (default: "
         "4)\n");
  printf("  --validate                             Check data correctness\n");
  printf("  --algorithm <ring|direct>              Legacy compatibility setting "
         "(default: ring)\n");
  printf("  --api <default|window>                 Public entry point (default: default)\n");
  printf("  --copy-algorithm <a>                   Advanced staging override: "
         "'pack',\n");
  printf("                                           'pipe', or 'direct' "
         "(default: pack)\n");
  printf("  --lb-mode <node|uniform>               Load balance (default: "
         "node)\n");
  printf("  --print-all-ranks                      Per-rank timing\n");
  printf("  --verbose                              Debug output\n");
  printf("\nExample:\n");
  printf("  mpirun -np 16 %s \\\n", prog);
  printf("      --src-mesh-dims 1,8 --dst-mesh-dims 1,8 \\\n");
  printf("      --tensor-dims 256,256:1024,1024:4096,4096 \\\n");
  printf("      --src-shard-dims 0,0 --dst-shard-dims 0,1 \\\n");
  printf("      --num-comms 2 --num-tensors 4 --validate\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  MPICHECK(MPI_Init(&argc, &argv));
  int mpiRank, mpiSize;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));

  int numComms = 1;
  int numTensors = 4;
  int srcMdims[NCCL_RESHARD_MESH_NDIMS] = {0, 0};
  int dstMdims[NCCL_RESHARD_MESH_NDIMS] = {0, 0};
  int iterations = 20;
  int warmup = 4;
  bool bValidate = false;
  bool bVerbose = false;
  bool bPrintAllRanks = false;
  const char* algorithm = "RING";
  ReshardApiMode apiMode = ReshardApiMode::Default;
  const char* copyAlgorithm = nullptr;
  const char* lbMode = "UNIFORM";

  std::vector<TensorCfg> tensorCfgs;
  std::vector<int> srcSdRaw;
  std::vector<int> dstSdRaw;

  auto parseTensorCfgs = [&](const char* value) {
    if (!benchParseTensorDimsList(value, &tensorCfgs)) {
      if (mpiRank == 0) {
        printf("ERROR: invalid --tensor-dims value '%s'\n", value);
      }
      return BenchParseResult::Error;
    }
    return BenchParseResult::Success;
  };

  auto parseShardDims = [&](const char* name, std::vector<int>* out) {
    return [name, out, mpiRank](const char* value) {
      if (!benchParseNonNegativeIntList(value, out)) {
        if (mpiRank == 0) {
          printf("ERROR: invalid %s value '%s'\n", name, value);
        }
        return BenchParseResult::Error;
      }
      return BenchParseResult::Success;
    };
  };

  BenchArgParser parser(argc, argv, mpiRank);
  parser.integer("--num-comms", "--num-comms", &numComms)
    .integer("--num-tensors", "--num-tensors", &numTensors)
    .value("--tensor-dims", parseTensorCfgs)
    .meshDims("--src-mesh-dims", srcMdims)
    .meshDims("--dst-mesh-dims", dstMdims)
    .value("--src-shard-dims", parseShardDims("--src-shard-dims", &srcSdRaw))
    .value("--src-shard-dim", parseShardDims("--src-shard-dim", &srcSdRaw))
    .value("--dst-shard-dims", parseShardDims("--dst-shard-dims", &dstSdRaw))
    .value("--dst-shard-dim", parseShardDims("--dst-shard-dim", &dstSdRaw))
    .integer("--iterations", "--iterations", &iterations)
    .integer("--warmup", "--warmup", &warmup)
    .flag("--validate", [&] { bValidate = true; })
    .flag("--verbose", [&] { bVerbose = true; })
    .flag("--print-all-ranks", [&] { bPrintAllRanks = true; })
    .enumValue("--algorithm", &algorithm, {{"direct", "DIRECT"}, {"ring", "RING"}},
               "ERROR: unknown algorithm '%s' (use 'ring' or 'direct')\n")
    .apiMode("--api", &apiMode)
    .enumValue("--copy-algorithm", &copyAlgorithm,
               {{"direct", "DIRECT"}, {"pack", "PACK"}, {"pipe", "PIPE"}},
               "ERROR: unknown --copy-algorithm '%s' (use 'pack', 'pipe', or 'direct')\n")
    .enumValue("--lb-mode", &lbMode, {{"node", "NODE_AWARE"}, {"uniform", "UNIFORM"}},
               "ERROR: unknown --lb-mode '%s' (use 'uniform' or 'node')\n")
    .help(benchPrintUsage);

  int parseExit = benchParseExitCode(parser.parse());
  if (parseExit >= 0) {
    return parseExit;
  }

  if (iterations <= 0 || warmup < 0) {
    if (mpiRank == 0) {
      printf("ERROR: --iterations must be > 0 and --warmup must be >= 0\n");
    }
    MPI_Finalize();
    return 1;
  }
  benchConfigureCopyAlgorithm(copyAlgorithm);

  if (srcMdims[0] <= 0 || srcMdims[1] <= 0 || dstMdims[0] <= 0 || dstMdims[1] <= 0 || tensorCfgs.empty() ||
      srcSdRaw.empty() || dstSdRaw.empty()) {
    if (mpiRank == 0) {
      printf("ERROR: missing required parameters\n");
      benchPrintUsage(argv[0]);
    }
    MPI_Finalize();
    return 1;
  }
  if (srcSdRaw.size() != dstSdRaw.size()) {
    if (mpiRank == 0) {
      printf("ERROR: --src-shard-dims and --dst-shard-dims must have equal "
             "entry counts\n");
    }
    MPI_Finalize();
    return 1;
  }
  if (numComms < 1 || numTensors < 1) {
    if (mpiRank == 0) printf("ERROR: --num-comms and --num-tensors must be >= 1\n");
    MPI_Finalize();
    return 1;
  }

  std::vector<ShardCfg> shardCfgs;
  for (size_t i = 0; i < srcSdRaw.size(); i++) shardCfgs.push_back({srcSdRaw[i], dstSdRaw[i]});

  int srcTotal = srcMdims[0] * srcMdims[1];
  int dstTotal = dstMdims[0] * dstMdims[1];
  if (mpiSize != srcTotal + dstTotal) {
    if (mpiRank == 0) printf("ERROR: worldSize=%d but src+dst=%d\n", mpiSize, srcTotal + dstTotal);
    MPI_Finalize();
    return 1;
  }

  bool bIsSource = (mpiRank < srcTotal);
  bool bIsDest = !bIsSource;

  int numDevices;
  CUDACHECK(cudaGetDeviceCount(&numDevices));
  CUDACHECK(cudaSetDevice(mpiRank % numDevices));

  if (bVerbose) benchSetEnv("NCCL_RESHARD_LOG_LEVEL", "DEBUG");
  benchSetEnv("NCCL_RESHARD_ALGORITHM", algorithm);
  benchSetEnv("NCCL_RESHARD_LB_MODE", lbMode);
  ncclM2nHandle_t m2nHandle = nullptr;
  NCCLCHECK(ncclM2nInit(&m2nHandle, NULL));

  // ------------------------------------------------------------------------
  // Create numComms independent NCCL communicators (each covers all ranks)
  // ------------------------------------------------------------------------
  std::vector<ncclComm_t> comms(numComms);
  for (int c = 0; c < numComms; c++) {
    ncclUniqueId uid;
    if (mpiRank == 0) NCCLCHECK(ncclGetUniqueId(&uid));
    MPICHECK(MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&comms[c], mpiSize, uid, mpiRank));
  }

  // ------------------------------------------------------------------------
  // Buffer sizing — pre-allocate to the max across (tensor × shard) combos
  // so the same buffers (and registered windows) work for every config.
  // ------------------------------------------------------------------------
  const size_t NCCL_MIN_ALLOC = 4096U;
  size_t maxAlloc = NCCL_MIN_ALLOC;
  for (auto& tc : tensorCfgs) {
    for (auto& sc : shardCfgs) {
      if (sc.srcSd >= tc.nDims || sc.dstSd >= tc.nDims) continue;
      size_t sb = 1;
      size_t db = 1;
      for (int d = 0; d < tc.nDims; d++) {
        sb *= (d == sc.srcSd) ? tc.dims[d] / srcMdims[1] : tc.dims[d];
        db *= (d == sc.dstSd) ? tc.dims[d] / dstMdims[1] : tc.dims[d];
      }
      maxAlloc = std::max({maxAlloc, sb, db});
    }
  }

  // ------------------------------------------------------------------------
  // Per-tensor symmetric buffer + ncclWindow_t on its assigned comm.
  // Tensor i is bound to comms[i % numComms] for all subsequent reshards.
  // ------------------------------------------------------------------------
  std::vector<void*> bufs(numTensors);
  std::vector<ncclWindow_t> windows(numTensors, nullptr);
  for (int i = 0; i < numTensors; i++) {
    if (apiMode == ReshardApiMode::Default) {
      CUDACHECK(cudaMalloc(&bufs[i], maxAlloc));
    } else if (apiMode == ReshardApiMode::Window) {
      NCCLCHECK(ncclMemAlloc(&bufs[i], maxAlloc));
    }
    CUDACHECK(cudaMemset(bufs[i], 0xDE, maxAlloc));
    if (apiMode == ReshardApiMode::Window) {
      NCCLCHECK(benchWaitCommReady(comms[i % numComms]));
      NCCLCHECK(ncclCommWindowRegister(comms[i % numComms], bufs[i], maxAlloc, &windows[i],
                                                   NCCL_WIN_COLL_SYMMETRIC));
    }
  }

  std::vector<cudaStream_t> streams(numTensors);
  for (int i = 0; i < numTensors; i++) CUDACHECK(cudaStreamCreate(&streams[i]));
  cudaStream_t seqStream;
  CUDACHECK(cudaStreamCreate(&seqStream));

  if (mpiRank == 0) {
    printf("=== Batched Transfer Benchmark — User Window API ===\n");
    printf("Comms     : %d\n", numComms);
    printf("Tensors   : %d per batch\n", numTensors);
    printf("Src mesh  : %dx%d\n", srcMdims[0], srcMdims[1]);
    printf("Dst mesh  : %dx%d\n", dstMdims[0], dstMdims[1]);
    printf("Algorithm setting (compatibility): %s\n", algorithm);
    printf("Iters     : %d  warmup=%d\n", iterations, warmup);
    printf("\n");
    printf("%-24s %-12s %9s %9s %8s %7s %9s %9s\n", "Tensor dims", "Pattern", "Seq ms", "Con ms", "Speedup", "Eff%",
           "Seq GB/s", "Con GB/s");
    printf("%-24s %-12s %9s %9s %8s %7s %9s %9s\n", "------------------------", "------------", "---------",
           "---------", "--------", "-------", "---------", "---------");
    fflush(stdout);
  }

  // Finish registration before transfers read the window handles.
  if (apiMode == ReshardApiMode::Window) {
    for (auto& c : comms) NCCLCHECK(benchWaitCommReady(c));
  }

  int overallRc = 0;

  // ------------------------------------------------------------------------
  // Sweep: shard pattern × tensor size
  // ------------------------------------------------------------------------
  for (auto& sc : shardCfgs) {
    ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
    srcMesh.ndims = 2;
    srcMesh.dims = srcMdims;
    ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
    dstMesh.ndims = 2;
    dstMesh.dims = dstMdims;
    dstMesh.startRank = srcTotal;
    const char* pattern = (sc.srcSd == sc.dstSd) ? "same-dim" : "cross-dim";

    for (auto& tc : tensorCfgs) {
      if (sc.srcSd >= tc.nDims || sc.dstSd >= tc.nDims) {
        if (mpiRank == 0) {
          printf("SKIP %-20s shard_dims=%d/%d out of range for %dD "
                 "tensor\n",
                 "", sc.srcSd, sc.dstSd, tc.nDims);
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        continue;
      }

      size_t srcLocal[3] = {0, 0, 0};
      size_t dstLocal[3] = {0, 0, 0};
      for (int d = 0; d < tc.nDims; d++) {
        srcLocal[d] = (d == sc.srcSd) ? tc.dims[d] / srcMdims[1] : tc.dims[d];
        dstLocal[d] = (d == sc.dstSd) ? tc.dims[d] / dstMdims[1] : tc.dims[d];
      }
      size_t srcBufBytes = 1;
      size_t dstBufBytes = 1;
      for (int d = 0; d < tc.nDims; d++) {
        srcBufBytes *= srcLocal[d];
        dstBufBytes *= dstLocal[d];
      }
      int srcSc = srcMdims[1];
      int dstSc = dstMdims[1];

      if (bIsSource && bValidate) {
        int sidx = mpiRank % srcSc;
        for (int i = 0; i < numTensors; i++)
          benchInitSourceData((char*)bufs[i], srcLocal, tc.nDims, sc.srcSd, sidx, srcSc, streams[i]);
        for (int i = 0; i < numTensors; i++) CUDACHECK(cudaStreamSynchronize(streams[i]));
      }
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

      // --- one reshard for tensor i on stream s, comm[i % numComms]
      auto oneTransfer = [&](int i, cudaStream_t s) {
        size_t srcLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
        int srcPlacements[] = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_SHARD(sc.srcSd)};
        ncclDistTensor_t srcT = NCCL_M2N_DIST_TENSOR_INITIALIZER;
        srcT.localShape = srcLocalShape;
        srcT.ndims = tc.nDims;
        srcT.dtype = ncclInt8; // bench validates byte patterns
        srcT.mesh = &srcMesh;
        srcT.placements = srcPlacements;
        srcT.dataPtr = bIsSource ? bufs[i] : nullptr;
        for (int d = 0; d < tc.nDims; d++) {
          srcT.localShape[d] = srcLocal[d];
        }

        size_t dstLocalShape[NCCL_RESHARD_MAX_TENSOR_DIMS] = {};
        int dstPlacements[] = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_SHARD(sc.dstSd)};
        ncclDistTensor_t dstT = NCCL_M2N_DIST_TENSOR_INITIALIZER;
        dstT.localShape = dstLocalShape;
        dstT.ndims = tc.nDims;
        dstT.dtype = ncclInt8;
        dstT.mesh = &dstMesh;
        dstT.placements = dstPlacements;
        dstT.dataPtr = bIsDest ? bufs[i] : nullptr;
        for (int d = 0; d < tc.nDims; d++) {
          dstT.localShape[d] = dstLocal[d];
        }

        if (apiMode == ReshardApiMode::Default) {
          NCCLCHECK(ncclReshard(m2nHandle, comms[i % numComms], &srcT, &dstT, s));
        } else {
          NCCLCHECK(ncclReshardWithWindow(m2nHandle, comms[i % numComms], windows[i], &srcT, &dstT, s));
        }
      };

      auto runSequential = [&]() {
        for (int i = 0; i < numTensors; i++) {
          oneTransfer(i, seqStream);
          CUDACHECK(cudaStreamSynchronize(seqStream));
        }
      };

      auto runConcurrent = [&]() {
        for (int i = 0; i < numTensors; i++) oneTransfer(i, streams[i]);
        for (int i = 0; i < numTensors; i++) CUDACHECK(cudaStreamSynchronize(streams[i]));
      };

      // Warmup (mix both patterns so the devComm caches stay warm)
      for (int it = 0; it < warmup; it++) {
        runSequential();
        runConcurrent();
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }

      // Optional validation between warmup and timed runs
      if (bValidate) {
        bool ok = true;
        if (bIsDest) {
          int didx = (mpiRank - srcTotal) % dstSc;
          for (int i = 0; i < numTensors; i++) {
            ok &= benchValidateDestData((const char*)bufs[i], dstLocal, tc.nDims, sc.dstSd, didx, dstSc, mpiRank,
                                        streams[i]);
          }
        }
        int okInt = ok ? 1 : 0;
        int bAllOk = 0;
        MPICHECK(MPI_Allreduce(&okInt, &bAllOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (mpiRank == 0 && !bAllOk) {
          char tstr[64];
          int off = snprintf(tstr, sizeof(tstr), "[%zu", tc.dims[0]);
          for (int d = 1; d < tc.nDims; d++) off += snprintf(tstr + off, sizeof(tstr) - off, ",%zu", tc.dims[d]);
          snprintf(tstr + off, sizeof(tstr) - off, "]");
          printf("*** VALIDATION FAILED  tensor=%-16s pattern=%s ***\n", tstr, pattern);
          fflush(stdout);
        }
        if (!bAllOk) {
          overallRc = 1;
          MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
          continue;
        }
        if (bIsDest)
          for (int i = 0; i < numTensors; i++) CUDACHECK(cudaMemset(bufs[i], 0xDE, maxAlloc));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      }

      // --- timed runs
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      auto t0s = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < iterations; it++) runSequential();
      auto t1s = std::chrono::high_resolution_clock::now();
      double seqMs = std::chrono::duration<double, std::milli>(t1s - t0s).count() / iterations;

      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      auto t0c = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < iterations; it++) runConcurrent();
      auto t1c = std::chrono::high_resolution_clock::now();
      double conMs = std::chrono::duration<double, std::milli>(t1c - t0c).count() / iterations;

      double seqWall = 0;
      double conWall = 0;
      MPICHECK(MPI_Reduce(&seqMs, &seqWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
      MPICHECK(MPI_Reduce(&conMs, &conWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

      if (bPrintAllRanks) {
        for (int r = 0; r < mpiSize; r++) {
          if (mpiRank == r) {
            printf("[Rank %3d] %s  seq=%.3f ms  con=%.3f ms\n", r, bIsSource ? "src" : "dst", seqMs, conMs);
            fflush(stdout);
          }
          MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
      }

      if (mpiRank == 0) {
        double payload = (double)numTensors * (double)(bIsSource ? srcBufBytes : dstBufBytes);
        double seqBw = (payload / (seqWall / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        double conBw = (payload / (conWall / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        double speedup = (conWall > 0) ? (seqWall / conWall) : 0.0;
        // Effective concurrency cap is min(numComms, numTensors).
        double effPct = (speedup / std::min(numComms, numTensors)) * 100.0;

        char tstr[64];
        int off = snprintf(tstr, sizeof(tstr), "[%zu", tc.dims[0]);
        for (int d = 1; d < tc.nDims; d++) off += snprintf(tstr + off, sizeof(tstr) - off, ",%zu", tc.dims[d]);
        snprintf(tstr + off, sizeof(tstr) - off, "]");

        printf("%-24s %-12s %9.3f %9.3f %8.2f %6.1f%% %9.2f %9.2f\n", tstr, pattern, seqWall, conWall, speedup, effPct,
               seqBw, conBw);
        fflush(stdout);
      }

      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  if (mpiRank == 0) {
    printf("\n");
    if (overallRc == 0) printf("Benchmark completed successfully!\n");
    else printf("Benchmark completed with VALIDATION FAILURES.\n");
  }

  // ------------------------------------------------------------------------
  // Teardown — complete work and deregister windows before destroying comms.
  // ------------------------------------------------------------------------
  CUDACHECK(cudaStreamSynchronize(seqStream));
  for (auto& s : streams) CUDACHECK(cudaStreamSynchronize(s));
  if (apiMode == ReshardApiMode::Window) {
    for (int i = 0; i < numTensors; i++) {
      NCCLCHECK(benchWaitCommReady(comms[i % numComms]));
      NCCLCHECK(ncclCommWindowDeregister(comms[i % numComms], windows[i]));
    }
  }
  // Complete deregistration before releasing the buffers.
  if (apiMode == ReshardApiMode::Window) {
    for (auto& c : comms) NCCLCHECK(benchWaitCommReady(c));
  }
  NCCLCHECK(ncclM2nFinalize(m2nHandle));
  for (auto& b : bufs) {
    if (apiMode == ReshardApiMode::Default) {
      CUDACHECK(cudaFree(b));
    } else if (apiMode == ReshardApiMode::Window) {
      NCCLCHECK(ncclMemFree(b));
    }
  }
  for (auto& s : streams) CUDACHECK(cudaStreamDestroy(s));
  CUDACHECK(cudaStreamDestroy(seqStream));
  for (auto& c : comms) {
    NCCLCHECK(benchWaitCommReady(c));
    NCCLCHECK(ncclCommDestroy(c));
  }

  MPICHECK(MPI_Finalize());
  return overallRc;
}
