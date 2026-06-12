/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * RcclSmokeTest — standalone RCCL functionality smoke test.
 *
 * Tests: AllReduce, AllGather, ReduceScatter, AllToAll, Broadcast
 * Sizes: 1 KB, 4 MB, 1 GB
 * Datatypes: reduction collectives (AllReduce, ReduceScatter) run both
 *            float32 and bfloat16; data-movement collectives run float32 only.
 *            fp32 rows run first, then bf16, to minimize GPU buffer refills.
 * Reduction op: ncclSum  (where applicable)
 *
 * Modes:
 *   Single-node: ./RcclSmokeTest
 *     Auto-detects all local GPUs; one process manages all GPUs.
 *
 *   Multi-node:  NCCL_COMM_ID=<root_host>:<port> \
 *                  ./RcclSmokeTest --nranks N --rank R [--device D]
 *     One process per GPU; NCCL_COMM_ID bootstraps the communicator.
 */

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ─── version ──────────────────────────────────────────────────────────────

#ifndef SMOKETEST_VERSION_MAJOR
#define SMOKETEST_VERSION_MAJOR 0
#endif
#ifndef SMOKETEST_VERSION_MINOR
#define SMOKETEST_VERSION_MINOR 0
#endif
#ifndef SMOKETEST_VERSION_PATCH
#define SMOKETEST_VERSION_PATCH 0
#endif
#ifndef SMOKETEST_GIT_COMMIT
#define SMOKETEST_GIT_COMMIT "unknown"
#endif

#define SMOKETEST_STR2(x) #x
#define SMOKETEST_STR(x)  SMOKETEST_STR2(x)
#define SMOKETEST_VERSION \
  SMOKETEST_STR(SMOKETEST_VERSION_MAJOR) "." \
  SMOKETEST_STR(SMOKETEST_VERSION_MINOR) "." \
  SMOKETEST_STR(SMOKETEST_VERSION_PATCH) \
  " (" SMOKETEST_GIT_COMMIT ")"

// ─── error-check macros ────────────────────────────────────────────────────

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t _e = (cmd);                                                     \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "[ERROR] HIP %s:%d — %s\n",                             \
              __FILE__, __LINE__, hipGetErrorString(_e));                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define RCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t _r = (cmd);                                                   \
    if (_r != ncclSuccess) {                                                   \
      fprintf(stderr, "[ERROR] RCCL %s:%d — %s\n",                            \
              __FILE__, __LINE__, ncclGetErrorString(_r));                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ─── test dimensions ────────────────────────────────────────────────────────

static const size_t MSG_BYTES[]   = { 1024UL, 4UL << 20, 1UL << 30 };
static const char*  SIZE_LABELS[] = { "1 KB", "4 MB", "1 GB" };
static constexpr int N_SIZES = 3;

enum Collective {
  ALLREDUCE     = 0,
  ALLGATHER     = 1,
  REDUCESCATTER = 2,
  ALLTOALL      = 3,
  BROADCAST     = 4,
};

static const char* COLL_NAMES[] = {
  "AllReduce", "AllGather", "ReduceScatter", "AllToAll", "Broadcast"
};

// ─── datatype descriptor ────────────────────────────────────────────────────

struct DTypeInfo {
  ncclDataType_t ncclType;
  size_t         elemBytes;
  const char*    name;
};

static const DTypeInfo DTYPES[] = {
  { ncclFloat,    sizeof(float),    "float32"  },
  { ncclBfloat16, sizeof(uint16_t), "bfloat16" },
};


// ─── test row: one (collective, dtype) pair ──────────────────────────────────

struct TestRow {
  Collective       coll;
  const DTypeInfo* dt;
  char             label[32];        // e.g. "ReduceScatter (bfloat16)"
  char             results[N_SIZES]; // 'P'=pass  'F'=fail  'S'=OOM  '-'=not run
};

// ─── output ─────────────────────────────────────────────────────────────────

static void printTable(int nRanks, const std::vector<TestRow>& rows) {
  printf("\n");
  printf("  Legend: P=Pass  F=Fail  S=Skip(OOM)  -=Not run\n\n");
  printf("  %-26s | %6s | %6s | %6s |\n",
         "Collective", SIZE_LABELS[0], SIZE_LABELS[1], SIZE_LABELS[2]);
  printf("  --------------------------+---------+--------+--------+\n");
  for (const auto& row : rows) {
    printf("  %-26s |   %c   |   %c   |   %c   |\n",
           row.label,
           row.results[0], row.results[1], row.results[2]);
  }
  printf("\n");
}

static bool anyFailed(const std::vector<TestRow>& rows) {
  for (const auto& row : rows)
    for (int s = 0; s < N_SIZES; s++)
      if (row.results[s] == 'F' || row.results[s] == 'S') return true;
  return false;
}

// BF16 is the upper 16 bits of a float32 IEEE-754 word.
static uint16_t floatToBF16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even before truncating.
  uint32_t lsb = (bits >> 16) & 1u;
  bits         += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// ─── data helpers ───────────────────────────────────────────────────────────

// Fill host buffer with (rank+1) + (i%16) in the target datatype.
// Max fill value is nRanks+15; BF16 exactly represents integers up to 256,
// so fill values are exact for up to ~241 GPUs.
static void fillHost(void* buf, size_t n, int rank, const DTypeInfo& dt) {
  if (dt.ncclType == ncclFloat) {
    float* p = static_cast<float*>(buf);
    for (size_t i = 0; i < n; i++) p[i] = (float)(rank + 1) + (float)(i % 16);
  } else {
    uint16_t* p = static_cast<uint16_t*>(buf);
    for (size_t i = 0; i < n; i++)
      p[i] = floatToBF16((float)(rank + 1) + (float)(i % 16));
  }
}

// GPU fill kernel: writes (rank+1) + (i%16) into dSend for n elements.
// Used for the 1 GB size to avoid a 1 GB H→D PCIe transfer.
__global__ static void fillSendKernel(void* dSend, size_t n, int rank, int isBF16) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float val = (float)(rank + 1) + (float)(i % 16);
  if (!isBF16) {
    ((float*)dSend)[i] = val;
  } else {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    ((uint16_t*)dSend)[i] = (uint16_t)(bits >> 16);
  }
}

// ─── GPU validation kernels ──────────────────────────────────────────────────
//
// Each kernel checks one element per thread. On mismatch, atomicOr sets dFail.
// isBF16=1 reads uint16_t and expands to float; isBF16=0 reads float directly.
// relTol: 0.0f for float32, 1/64f for bfloat16.

__device__ static float elemToFloat(const void* buf, size_t i, int isBF16) {
  if (!isBF16) return ((const float*)buf)[i];
  uint32_t bits = (uint32_t)((const uint16_t*)buf)[i] << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

__device__ static float checkTolDev(float expected, float relTol) {
  return fmaxf(0.5f, fabsf(expected) * relTol);
}

// AllReduce: element i = nRanks*(nRanks+1)/2 + nRanks*(i%16)
__global__ static void checkAllReduceKernel(const void* buf, size_t n,
                                            int nRanks, float relTol,
                                            int isBF16, int* dFail) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float base     = (float)(nRanks * (nRanks + 1) / 2);
  float expected = base + (float)nRanks * (float)(i % 16);
  float got      = elemToFloat(buf, i, isBF16);
  if (fabsf(got - expected) > checkTolDev(expected, relTol))
    atomicOr(dFail, 1);
}

// AllGather: chunk g, element j = (g+1) + j%16
__global__ static void checkAllGatherKernel(const void* buf, size_t nPerRank,
                                            int nRanks, float relTol,
                                            int isBF16, int* dFail) {
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = nPerRank * (size_t)nRanks;
  if (idx >= total) return;
  int    g        = (int)(idx / nPerRank);
  size_t j        = idx % nPerRank;
  float  expected = (float)(g + 1) + (float)(j % 16);
  float  got      = elemToFloat(buf, idx, isBF16);
  if (fabsf(got - expected) > checkTolDev(expected, relTol))
    atomicOr(dFail, 1);
}

// ReduceScatter: rank r receives [r*nPerRank .. (r+1)*nPerRank).
// gi = rank*nPerRank + i; expected = base + nRanks*(gi%16)
__global__ static void checkReduceScatterKernel(const void* buf, size_t nPerRank,
                                                int rank, int nRanks, float relTol,
                                                int isBF16, int* dFail) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nPerRank) return;
  float  base     = (float)(nRanks * (nRanks + 1) / 2);
  size_t gi       = (size_t)rank * nPerRank + i;
  float  expected = base + (float)nRanks * (float)(gi % 16);
  float  got      = elemToFloat(buf, i, isBF16);
  if (fabsf(got - expected) > checkTolDev(expected, relTol))
    atomicOr(dFail, 1);
}

// AllToAll: chunk from sender s, element j: expected = (s+1) + (rank*nPerRank+j)%16
__global__ static void checkAllToAllKernel(const void* buf, size_t nPerRank,
                                           int rank, int nRanks, float relTol,
                                           int isBF16, int* dFail) {
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = nPerRank * (size_t)nRanks;
  if (idx >= total) return;
  int    s        = (int)(idx / nPerRank);
  size_t j        = idx % nPerRank;
  float  expected = (float)(s + 1) + (float)(((size_t)rank * nPerRank + j) % 16);
  float  got      = elemToFloat(buf, idx, isBF16);
  if (fabsf(got - expected) > checkTolDev(expected, relTol))
    atomicOr(dFail, 1);
}

// Broadcast: root is rank 0, so element i = 1.0 + i%16
__global__ static void checkBroadcastKernel(const void* buf, size_t n,
                                            float relTol, int isBF16, int* dFail) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float expected = 1.0f + (float)(i % 16);
  float got      = elemToFloat(buf, i, isBF16);
  if (fabsf(got - expected) > checkTolDev(expected, relTol))
    atomicOr(dFail, 1);
}

// Dispatch: zero dFail on stream, launch appropriate kernel, caller syncs stream.
static void launchCheckKernel(int c, const void* dRecv, size_t recvE,
                               size_t nPerRank, int rank, int nRanks,
                               float relTol, int isBF16,
                               int* dFail, hipStream_t stream) {
  HIP_CHECK(hipMemsetAsync(dFail, 0, sizeof(int), stream));
  dim3 block(256);
  dim3 grid((recvE + 255) / 256);
  switch (c) {
    case ALLREDUCE:
      checkAllReduceKernel<<<grid, block, 0, stream>>>(
        dRecv, recvE, nRanks, relTol, isBF16, dFail);
      break;
    case ALLGATHER:
      checkAllGatherKernel<<<grid, block, 0, stream>>>(
        dRecv, nPerRank, nRanks, relTol, isBF16, dFail);
      break;
    case REDUCESCATTER:
      checkReduceScatterKernel<<<grid, block, 0, stream>>>(
        dRecv, nPerRank, rank, nRanks, relTol, isBF16, dFail);
      break;
    case ALLTOALL:
      checkAllToAllKernel<<<grid, block, 0, stream>>>(
        dRecv, nPerRank, rank, nRanks, relTol, isBF16, dFail);
      break;
    case BROADCAST:
      checkBroadcastKernel<<<grid, block, 0, stream>>>(
        dRecv, recvE, relTol, isBF16, dFail);
      break;
  }
}

// ─── test runner ─────────────────────────────────────────────────────────────
//
// Single-node: nGpus=N, myRank=0, nRanks=N — one process owns all GPUs.
// Multi-node:  nGpus=1, myRank=globalRank, nRanks=totalRanks — one GPU per process.
//
// doAggregate=true in multi-node: after local validation, ncclAllReduce the dFail
// flag so rank 0 sees the global result without extra sockets.

static void runTests(int nGpus, int myRank, int nRanks, int deviceBase,
                     std::vector<ncclComm_t>& comms,
                     std::vector<hipStream_t>& streams,
                     std::vector<TestRow>& rows,
                     const bool sizeEnabled[N_SIZES]) {
  // GPU buffers are 1 GB each (largest test size).
  // hSend is only 4 MB — covers the 1 KB and 4 MB sizes via H→D copy.
  // The 1 GB size uses fillSendKernel to write dSend directly on the GPU,
  // avoiding a 1 GB PCIe transfer.
  const size_t MAX_BYTES  = 1UL << 30;
  const size_t FILL_BYTES = 4UL << 20;  // 4 MB: largest size filled via H→D

  // In multi-node mode, aggregate dFail across all ranks via ncclAllReduce
  // so rank 0 sees the global pass/fail without any extra sockets.
  const bool doAggregate = (nRanks > nGpus);

  struct GpuBufs { void* dSend; void* dRecv; int* dFail; };
  std::vector<GpuBufs> bufs(nGpus);
  std::vector<std::vector<uint8_t>> hSend(nGpus);

  for (int g = 0; g < nGpus; g++) {
    HIP_CHECK(hipSetDevice(deviceBase + g));
    HIP_CHECK(hipMalloc(&bufs[g].dSend, MAX_BYTES));
    HIP_CHECK(hipMalloc(&bufs[g].dRecv, MAX_BYTES));
    HIP_CHECK(hipMalloc(&bufs[g].dFail, sizeof(int)));
    hSend[g].resize(FILL_BYTES);
  }

  const DTypeInfo* lastFillDtype = nullptr;

  for (auto& row : rows) {
    const DTypeInfo& dt = *row.dt;
    int c = row.coll;

    // On dtype change: fill 4 MB on the host and copy H→D.
    // Covers 1 KB and 4 MB sizes. The 1 GB size re-fills dSend via kernel below.
    if (row.dt != lastFillDtype) {
      size_t fillElems = FILL_BYTES / dt.elemBytes;
      for (int g = 0; g < nGpus; g++) {
        HIP_CHECK(hipSetDevice(deviceBase + g));
        fillHost(hSend[g].data(), fillElems, myRank + g, dt);
        HIP_CHECK(hipMemcpy(bufs[g].dSend, hSend[g].data(),
                            FILL_BYTES, hipMemcpyHostToDevice));
      }
      lastFillDtype = row.dt;
    }

    float relTol = (dt.ncclType == ncclFloat) ? 0.0f : (1.0f / 64.0f);
    int   isBF16 = (dt.ncclType == ncclBfloat16) ? 1 : 0;

    for (int s = 0; s < N_SIZES; s++) {
      if (!sizeEnabled[s]) continue;
      size_t msgBytes = MSG_BYTES[s];
      size_t nElems   = msgBytes / dt.elemBytes;
      size_t nPerRank = (nElems / (size_t)nRanks) ? (nElems / (size_t)nRanks) : 1;

      size_t recvE;
      switch (c) {
        case ALLGATHER:     recvE = nPerRank * (size_t)nRanks; break;
        case REDUCESCATTER: recvE = nPerRank;                  break;
        default:            recvE = nElems;                    break;
      }

      // For the 1 GB size, fill dSend on the GPU (avoids 1 GB PCIe transfer).
      // Queued on each GPU's stream so ordering with the collective is guaranteed.
      if (s == N_SIZES - 1) {
        size_t maxElems = MAX_BYTES / dt.elemBytes;
        dim3 blk(256), grd((maxElems + 255) / 256);
        for (int g = 0; g < nGpus; g++) {
          HIP_CHECK(hipSetDevice(deviceBase + g));
          fillSendKernel<<<grd, blk, 0, streams[g]>>>(
            bufs[g].dSend, maxElems, myRank + g, isBF16);
        }
      }

      // Zero recv buffer; dSend is already filled (H→D for small sizes, kernel for 1 GB).
      for (int g = 0; g < nGpus; g++) {
        HIP_CHECK(hipSetDevice(deviceBase + g));
        HIP_CHECK(hipMemset(bufs[g].dRecv, 0, recvE * dt.elemBytes));
      }

      // Launch collective across all GPU comms in a group.
      RCCL_CHECK(ncclGroupStart());
      for (int g = 0; g < nGpus; g++) {
        int rank = myRank + g;
        switch (c) {
          case ALLREDUCE:
            RCCL_CHECK(ncclAllReduce(bufs[g].dSend, bufs[g].dRecv,
                                     nElems, dt.ncclType, ncclSum,
                                     comms[g], streams[g]));
            break;
          case ALLGATHER:
            RCCL_CHECK(ncclAllGather(bufs[g].dSend, bufs[g].dRecv,
                                     nPerRank, dt.ncclType,
                                     comms[g], streams[g]));
            break;
          case REDUCESCATTER:
            RCCL_CHECK(ncclReduceScatter(bufs[g].dSend, bufs[g].dRecv,
                                         nPerRank, dt.ncclType, ncclSum,
                                         comms[g], streams[g]));
            break;
          case ALLTOALL:
            for (int peer = 0; peer < nRanks; peer++) {
              RCCL_CHECK(ncclSend(
                static_cast<uint8_t*>(bufs[g].dSend) + (size_t)peer * nPerRank * dt.elemBytes,
                nPerRank, dt.ncclType, peer, comms[g], streams[g]));
              RCCL_CHECK(ncclRecv(
                static_cast<uint8_t*>(bufs[g].dRecv) + (size_t)peer * nPerRank * dt.elemBytes,
                nPerRank, dt.ncclType, peer, comms[g], streams[g]));
            }
            break;
          case BROADCAST:
            RCCL_CHECK(ncclBroadcast(bufs[g].dSend, bufs[g].dRecv,
                                     nElems, dt.ncclType, /*root=*/0,
                                     comms[g], streams[g]));
            break;
        }
        (void)rank;
      }
      RCCL_CHECK(ncclGroupEnd());

      // Launch GPU validation kernel; in multi-node mode, allreduce the flag
      // so rank 0 sees whether any rank failed. Only rank 0 records the result.
      bool allPass = true;
      for (int g = 0; g < nGpus; g++) {
        int rank = myRank + g;
        HIP_CHECK(hipSetDevice(deviceBase + g));
        launchCheckKernel(c, bufs[g].dRecv, recvE, nPerRank, rank, nRanks,
                          relTol, isBF16, bufs[g].dFail, streams[g]);
        if (doAggregate) {
          RCCL_CHECK(ncclAllReduce(bufs[g].dFail, bufs[g].dFail,
                                   1, ncclInt32, ncclMax,
                                   comms[g], streams[g]));
        }
        HIP_CHECK(hipStreamSynchronize(streams[g]));
        if (myRank == 0) {
          int flag = 0;
          HIP_CHECK(hipMemcpy(&flag, bufs[g].dFail, sizeof(int), hipMemcpyDeviceToHost));
          if (flag) {
            fprintf(stderr, "[FAIL] %s  size=%s  rank=%d\n",
                    row.label, SIZE_LABELS[s], rank);
            allPass = false;
          }
        }
      }
      if (myRank == 0)
        row.results[s] = allPass ? 'P' : 'F';
    }
  }

  for (int g = 0; g < nGpus; g++) {
    HIP_CHECK(hipSetDevice(deviceBase + g));
    HIP_CHECK(hipFree(bufs[g].dSend));
    HIP_CHECK(hipFree(bufs[g].dRecv));
    HIP_CHECK(hipFree(bufs[g].dFail));
  }
}

// ─── logging setup ──────────────────────────────────────────────────────────

// Sets up NCCL debug logging to a per-rank file.
// Controlled by two env vars the caller may set before invoking the binary:
//   NCCL_LOGGING=0   — disable logging (default: enabled)
//   NCCL_LOG_DIR=<d> — directory for log files (default: ./logs)
// File name: rccl_smoketest_<hostname>_rank<N>.log
// Uses setenv with overwrite=0 so NCCL_DEBUG / NCCL_DEBUG_FILE already set
// in the environment by the user are never overridden.
static void setupLogging(int rank) {
  const char* loggingEnv = getenv("NCCL_LOGGING");
  if (loggingEnv && (strcmp(loggingEnv, "0") == 0 || strcmp(loggingEnv, "off") == 0 ||
                     strcmp(loggingEnv, "OFF") == 0 || strcmp(loggingEnv, "false") == 0)) {
    if (!getenv("_RCCL_ST_LOG_QUIET"))
      printf("NCCL log  : disabled\n");
    return;
  }

  const char* logDir = getenv("NCCL_LOG_DIR");
  if (!logDir || logDir[0] == '\0') logDir = "./logs";

  if (mkdir(logDir, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "WARNING: Could not create log directory '%s': %s\n",
            logDir, strerror(errno));
    return;
  }

  char hostname[256] = "unknown";
  gethostname(hostname, sizeof(hostname));

  char timestamp[32]; 
  time_t now = time(nullptr);
  strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

  char logPath[1024];
  snprintf(logPath, sizeof(logPath), "%s/rccl_smoketest_%s_rank%d_%s.log",
           logDir, hostname, rank, timestamp);

  setenv("NCCL_DEBUG",      "INFO",    0);
  setenv("NCCL_DEBUG_FILE", logPath,   0);
  // Suppress print when launched via LaunchSmokeTest.sh (script prints it).
  if (!getenv("_RCCL_ST_LOG_QUIET"))
    printf("NCCL log  : %s\n", logPath);
}

// ─── arg parsing ────────────────────────────────────────────────────────────

static void usage(const char* prog) {
  printf(
    "Usage:\n"
    "  %s                                    — single-node (auto-detect GPUs)\n"
    "  %s --nranks N --rank R [--device D]   — multi-node (one process per GPU)\n\n"
    "Multi-node requires NCCL_COMM_ID=<root_host>:<port> to be set in the environment.\n"
    "To pick a free port:  shuf -i 10000-65535 -n 1\n\n"
    "Options:\n"
    "  --coll <names>  Comma-separated collectives to run (default: all)\n"
    "                  Names: allreduce, allgather, reducescatter, alltoall, broadcast\n"
    "  --size <sizes>  Comma-separated message sizes to run (default: all)\n"
    "                  Sizes: 1kb, 4mb, 1gb\n"
    "  --dtype <types> Comma-separated datatypes to run (default: all)\n"
    "                  Types: float32, bfloat16\n"
    "                  Note: bfloat16 applies only to reduction collectives\n"
    "                        (allreduce, reducescatter); other collectives run\n"
    "                        float32 only and are unaffected by --dtype bfloat16\n"
    "  --version       Print version and exit\n"
    "  --help          Show this help\n\n"
    "Multi-node only options:\n"
    "  --nranks N      Total number of ranks across all nodes\n"
    "  --rank R        This process's global rank (0 … N-1)\n"
    "  --device D      Local GPU index to use (default: 0)\n"
    "Environment variables:\n"
    "  NCCL_COMM_ID=host:port  — required for multi-node\n"
    "  NCCL_LOGGING=0          — disable RCCL debug logging (default: enabled)\n"
    "  NCCL_LOG_DIR=<dir>      — directory for log files (default: ./logs)\n\n",
    prog, prog);
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  int         argNRanks = 0;
  int         argRank   = -1;
  int         argDevice = 0;
  std::string argColl;   // empty = all
  std::string argSize;   // empty = all
  std::string argDtype;  // empty = all

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(argv[0]); return 0;
    } else if (strcmp(argv[i], "--version") == 0) {
      printf("RcclSmokeTest v" SMOKETEST_VERSION "\n"); return 0;
    } else if (strcmp(argv[i], "--nranks") == 0 && i + 1 < argc) {
      argNRanks = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
      argRank = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      argDevice = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--coll") == 0 && i + 1 < argc) {
      argColl = argv[++i];
    } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      argSize = argv[++i];
    } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
      argDtype = argv[++i];
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]); return 1;
    }
  }

  bool multiNode = (argNRanks > 0 && argRank >= 0);

  if (multiNode && !getenv("NCCL_COMM_ID")) {
    fprintf(stderr,
      "[ERROR] Multi-node mode requires NCCL_COMM_ID to be set.\n"
      "  Example: NCCL_COMM_ID=<root_host>:<port> %s --nranks N --rank R\n"
      "  Pick a free port: shuf -i 10000-65535 -n 1\n",
      argv[0]);
    return 1;
  }

  if (!multiNode && (argNRanks > 0 || argRank >= 0)) {
    fprintf(stderr, "[ERROR] --nranks and --rank must both be specified for multi-node mode.\n");
    return 1;
  }

  const DTypeInfo* fp32 = &DTYPES[0];
  const DTypeInfo* bf16 = &DTYPES[1];

  auto makeRow = [](Collective c, const DTypeInfo* dt, bool showDtype) -> TestRow {
    TestRow r{};
    r.coll = c;
    r.dt   = dt;
    if (showDtype)
      snprintf(r.label, sizeof(r.label), "%s (%s)", COLL_NAMES[c], dt->name);
    else
      snprintf(r.label, sizeof(r.label), "%s", COLL_NAMES[c]);
    memset(r.results, '-', sizeof(r.results));
    return r;
  };

  // fp32 rows first, then bf16 — minimizes dSend refills (2 total).
  std::vector<TestRow> rows;
  rows.push_back(makeRow(ALLREDUCE,     fp32, true));
  rows.push_back(makeRow(ALLGATHER,     fp32, false));
  rows.push_back(makeRow(REDUCESCATTER, fp32, true));
  rows.push_back(makeRow(ALLTOALL,      fp32, false));
  rows.push_back(makeRow(BROADCAST,     fp32, false));
  rows.push_back(makeRow(ALLREDUCE,     bf16, true));
  rows.push_back(makeRow(REDUCESCATTER, bf16, true));

  // Helper: lowercase a string.
  auto toLower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
  };

  // Build sizeEnabled[] from --size (default: all enabled).
  bool sizeEnabled[N_SIZES];
  std::fill(sizeEnabled, sizeEnabled + N_SIZES, true);
  if (!argSize.empty()) {
    std::fill(sizeEnabled, sizeEnabled + N_SIZES, false);
    // SIZE_LABELS: "1 KB", "4 MB", "1 GB" — canonical tokens: "1kb","4mb","1gb"
    static const char* SIZE_TOKENS[] = { "1kb", "4mb", "1gb" };
    std::string spec = toLower(argSize);
    // strip spaces so "1 kb" and "1kb" both match
    spec.erase(std::remove(spec.begin(), spec.end(), ' '), spec.end());
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t comma = spec.find(',', pos);
      std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      bool matched = false;
      for (int s = 0; s < N_SIZES; s++) {
        if (tok == SIZE_TOKENS[s]) { sizeEnabled[s] = true; matched = true; }
      }
      if (!matched) {
        fprintf(stderr, "Unknown size '%s'. Valid sizes: 1kb, 4mb, 1gb\n", tok.c_str());
        return 1;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }

  // Filter rows by --coll (default: all).
  if (!argColl.empty()) {
    std::string spec = toLower(argColl);
    // Build set of requested collective names.
    std::vector<std::string> requested;
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t comma = spec.find(',', pos);
      std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      // Validate against known collective names.
      bool matched = false;
      for (int c = 0; c <= BROADCAST; c++) {
        if (tok == toLower(COLL_NAMES[c])) { matched = true; break; }
      }
      if (!matched) {
        fprintf(stderr, "Unknown collective '%s'. Valid names: allreduce, allgather, "
                        "reducescatter, alltoall, broadcast\n", tok.c_str());
        return 1;
      }
      requested.push_back(tok);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const TestRow& r) {
      std::string name = toLower(COLL_NAMES[r.coll]);
      return std::find(requested.begin(), requested.end(), name) == requested.end();
    }), rows.end());
    if (rows.empty()) {
      fprintf(stderr, "No rows match the requested collectives.\n");
      return 1;
    }
  }

  // Filter rows by --dtype (default: all).
  if (!argDtype.empty()) {
    std::string spec = toLower(argDtype);
    std::vector<std::string> requested;
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t comma = spec.find(',', pos);
      std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      bool matched = false;
      for (const auto& dt : DTYPES) {
        if (tok == toLower(dt.name)) { matched = true; break; }
      }
      if (!matched) {
        fprintf(stderr, "Unknown dtype '%s'. Valid types: float32, bfloat16\n", tok.c_str());
        return 1;
      }
      requested.push_back(tok);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const TestRow& r) {
      return std::find(requested.begin(), requested.end(), toLower(r.dt->name)) == requested.end();
    }), rows.end());
    if (rows.empty()) {
      fprintf(stderr, "No rows match the requested datatypes.\n"
                      "Note: bfloat16 applies only to allreduce and reducescatter.\n");
      return 1;
    }
  }

  auto t0 = std::chrono::steady_clock::now();

  if (multiNode) {
    // ── Multi-node path ──────────────────────────────────────────────────────
    // One process per GPU. NCCL_COMM_ID bootstraps ncclUniqueId across nodes.
    HIP_CHECK(hipSetDevice(argDevice));
    setupLogging(argRank);

    ncclUniqueId id;
    RCCL_CHECK(ncclGetUniqueId(&id));

    ncclComm_t  comm;
    hipStream_t stream;
    RCCL_CHECK(ncclCommInitRank(&comm, argNRanks, id, argRank));
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<ncclComm_t>  comms   = { comm };
    std::vector<hipStream_t> streams = { stream };
    runTests(1, argRank, argNRanks, argDevice, comms, streams, rows, sizeEnabled);

    HIP_CHECK(hipStreamDestroy(stream));
    RCCL_CHECK(ncclCommDestroy(comm));

    auto t1 = std::chrono::steady_clock::now();
    double elapsedSec = std::chrono::duration<double>(t1 - t0).count();

    if (argRank == 0) {
      printTable(argNRanks, rows);
      bool fail = anyFailed(rows);
      printf("  %s\n", fail ? "RESULT: SOME TESTS FAILED OR WERE SKIPPED" : "RESULT: ALL TESTS PASSED!");
      printf("  Test Time: %.1f s\n\n", elapsedSec);
      return fail ? 1 : 0;
    }
    return 0;

  } else {
    // ── Single-node path ─────────────────────────────────────────────────────
    // One process owns all local GPUs via ncclCommInitAll.
    int nGpus = 0;
    HIP_CHECK(hipGetDeviceCount(&nGpus));
    if (nGpus == 0) { fprintf(stderr, "No GPUs found.\n"); return 1; }

    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname));
    printf("RcclSmokeTest v" SMOKETEST_VERSION "\n");
    printf("Running single-node smoke test\n");
    printf("Host      : %s\n", hostname);
    printf("GPUs      : %d\n", nGpus);
    printf("Ranks     : %d\n", nGpus);
    setupLogging(/*rank=*/0);

    std::vector<int>         devices(nGpus);
    std::vector<ncclComm_t>  comms(nGpus);
    std::vector<hipStream_t> streams(nGpus);

    for (int g = 0; g < nGpus; g++) devices[g] = g;
    RCCL_CHECK(ncclCommInitAll(comms.data(), nGpus, devices.data()));
    for (int g = 0; g < nGpus; g++) {
      HIP_CHECK(hipSetDevice(g));
      HIP_CHECK(hipStreamCreate(&streams[g]));
    }

    runTests(nGpus, /*myRank=*/0, /*nRanks=*/nGpus, /*deviceBase=*/0, comms, streams, rows, sizeEnabled);

    for (int g = 0; g < nGpus; g++) {
      HIP_CHECK(hipSetDevice(g));
      HIP_CHECK(hipStreamDestroy(streams[g]));
      RCCL_CHECK(ncclCommDestroy(comms[g]));
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsedSec = std::chrono::duration<double>(t1 - t0).count();

    printTable(nGpus, rows);
    bool fail = anyFailed(rows);
    printf("  %s\n", fail ? "RESULT: SOME TESTS FAILED OR WERE SKIPPED" : "RESULT: ALL TESTS PASSED!");
    printf("  Test Time: %.1f s\n\n", elapsedSec);
    return fail ? 1 : 0;
  }
}
