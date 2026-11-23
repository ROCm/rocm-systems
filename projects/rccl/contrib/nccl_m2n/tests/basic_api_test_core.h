/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * basic_api_test_core.h — Shared test matrix + per-case execution.
 *
 * Mirrors an external pytest reference suite at the C level: each test
 * case from the pytest parametrize matrix becomes a TestCase entry.
 * The bootstrap layer (MPI vs single-process pthreads) is injected via
 * the TestEnv struct's function pointers so the same matrix runs from
 * both binaries.
 *
 * Mesh encoding follows the Python reshape:
 *   mesh dims[0] = N_axis0 (the first arg to reshape)
 *   mesh dims[1] = total / N_axis0
 *   placement decides whether axis 0 is shard or replicate
 *
 * shardIdx for a rank inside a mesh of shape [d0, d1] (startRank=s):
 *   PL_RS  (placement = {REPL, SHARD}):  shardIdx = (rank - s) % d1
 *   PL_SR  (placement = {SHARD, REPL}):  shardIdx = (rank - s) / d1
 *   PL_REPL                          :   shardIdx = 0
 ************************************************************************/

#ifndef TESTS_BASIC_API_TEST_CORE_H_
#define TESTS_BASIC_API_TEST_CORE_H_

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ostream>
#include <string>
#include <strings.h>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <nccl.h>

#include "nccl_m2n.h"
#ifdef NCCL_M2N_TESTING
#include "reshard_internal.h"
#endif
#include "test_helpers.h"

#ifndef GTEST_SKIP
#error "nccl_m2n tests require GoogleTest >= 1.10.0 (GTEST_SKIP support)"
#endif

/* ======================================================================
 * API path selector — both resharding APIs are exercised from the same
 * test matrix.
 *   ApiKind::Window -> env->buffer (ncclMemAlloc'd) + ncclCommWindowRegister
 *                      + ncclReshardWithWindow.
 *   ApiKind::Default -> env->copyBuffer (cudaMalloc'd) + ncclReshard
 *                       (no symmetric-window contract — the staging path
 *                       manages its own pool internally).
 * ====================================================================*/

enum class ApiKind {
  Window = 0,
  Default = 1
};

/* ======================================================================
 * Bootstrap-agnostic test environment.
 * ====================================================================*/

struct TestEnv {
  int rank;
  int worldSize;
  int device;
  ncclComm_t comm;
  cudaStream_t stream;
  cudaStream_t alternateStream;
  ncclM2nHandle_t m2nHandle;
  void* buffer;
  size_t bufferBytes;
  void* copyBuffer;
  size_t copyBufferBytes;
  ApiKind apiKind;
  bool expectPack;
  bool verbose;

  void (*barrier)(TestEnv* env);
  int (*allreduceMinInt)(TestEnv* env, int local);
  bool (*isRank0Printer)(TestEnv* env);
  void* ctx;
};

/* ======================================================================
 * Test descriptor.
 * ====================================================================*/

enum PlacementKind {
  PL_RS = 0, /* {REPLICATE, SHARD(d)} — Python "rs" */
  PL_SR = 1, /* {SHARD(d), REPLICATE} — Python "sr" */
  PL_REPL = 2, /* {REPLICATE, REPLICATE} — full-replicate, 1D mesh */
};

struct TestCase {
  std::string group;
  std::string name; /* full case name; populated by builders */
  int ndims;
  size_t globalDims[3];

  int srcDim0, dstDim0; /* mesh axis-0 size; 0 = "1D mesh, axis-0 is 1" */
  int srcShardDim; /* tensor dim sharded; -1 if PL_REPL */
  int dstShardDim;
  PlacementKind srcPl, dstPl;

  size_t elementSize; /* 1, 2, 4 */

  int worldMin; /* skip if worldSize < this */
  int worldMax; /* skip if worldSize > this; 0 = unbounded */
  int worldDivisor; /* skip if worldSize % this != 0 */

  int srcRatioNum, dstRatioNum; /* (0,0) ⇒ even split */
  bool dstFirst; /* place destination mesh before source mesh in parent rank order */
  bool bAsyncOrdering; /* keep source fill and destination validation on the caller stream */
  bool bGraphCapture;
  bool bGraphCapturePrewarmed;
  bool bNullWindow; /* pass NULL to ncclReshardWithWindow */
  int serialRepeats; /* reshard calls with a stream sync and validation after each call */
  const char* requiredCopyAlgorithm; /* nullptr = any copy algorithm */
};

static inline void printTo(const TestCase& tc, std::ostream* os) {
  *os << tc.name;
}

struct BasicApiCliArgs {
  int requestedRanks = 0; /* 0 = use cudaGetDeviceCount() */
  bool listOnly = false;
  bool verbose = false;
  const char* filter = nullptr;
  const char* algorithm = "ring";
  const char* copyAlgorithm = nullptr;
  const char* lbMode = "uniform";
  const char* api = "window"; /* "window", "default", or "all" */
  int maxWorld = 0; /* 0 = unrestricted */
  int minWorld = 0; /* 0 = unrestricted */
  int serialRepeats = 0; /* 0 = use the case default */
};

static void basicApiPrintUsage(const char* prog, const char* usageFmt, bool allowRankCount, bool allowAlgorithmAll) {
  printf("Usage: ");
  printf(usageFmt, prog);
  printf("\n\nOptions:\n");
  if (allowRankCount) {
    printf("  -N <ranks>                   Number of ranks/threads (<= device "
           "count)\n");
  }
  printf("  --list                       List test cases (with min world) and "
         "exit\n");
  printf("  --filter <substr>            Run only cases whose name contains "
         "<substr>\n");
  printf("  --max-world <N>              Skip cases whose minimum world > N\n");
  printf("                               (lets a CI script pre-filter by "
         "allocation)\n");
  printf("  --min-world <N>              Skip cases whose minimum world < N\n");
  printf("                               (combine with --max-world to bin by "
         "rank tier)\n");
  printf("  --algorithm ring|direct%s  Legacy scenario label (default: ring%s)\n", allowAlgorithmAll ? "|all" : "   ",
         allowAlgorithmAll ? "; 'all' registers one gtest case per label" : "");
  printf("  --copy-algorithm direct|pack|pipe\n");
  printf("                               Copy transport for every selected API\n");
  printf("                               (default: PACK for ring, DIRECT for direct)\n");
  printf("  --api  window|default|all    Reshard API surface (default: "
         "window;\n");
  printf("                               'all' runs both window and default)\n");
  printf("  --lb-mode  uniform|node      Load-balance mode (default: "
         "uniform)\n");
  printf("  --serial-repeats <N>         Override serial reuse count for cases "
         "that define it\n");
  printf("  --verbose                    Verbose / per-rank output\n");
  printf("  --help                       Print this help\n");
}

static bool basicApiConsumeGtestArg(int argc, char** argv, int* i) {
  const char* arg = argv[*i];
  if (strncmp(arg, "--gtest_", 8) != 0) return false;
  if (strchr(arg, '=') == nullptr && *i + 1 < argc && argv[*i + 1][0] != '-') ++(*i);
  return true;
}

static const char* basicApiRequireValue(int argc, char** argv, int* i) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "Missing value for %s\n", argv[*i]);
    _Exit(2);
  }
  return argv[++(*i)];
}

static int basicApiParseIntArg(const char* value) {
  char* end = nullptr;
  long n = strtol(value, &end, 10);
  if (end == value) return 0;
  if (n < INT_MIN) return INT_MIN;
  if (n > INT_MAX) return INT_MAX;
  return (int)n;
}

static BasicApiCliArgs basicApiParseCli(int argc, char** argv, const char* usageFmt, bool allowRankCount,
                                        bool allowAlgorithmAll) {
  BasicApiCliArgs a;
  for (int i = 1; i < argc; i++) {
    const char* k = argv[i];
    if (basicApiConsumeGtestArg(argc, argv, &i)) continue;
    if (allowRankCount && strcmp(k, "-N") == 0) {
      a.requestedRanks = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
    } else if (strcmp(k, "--list") == 0) {
      a.listOnly = true;
    } else if (strcmp(k, "--verbose") == 0) {
      a.verbose = true;
    } else if (strcmp(k, "--filter") == 0) {
      a.filter = basicApiRequireValue(argc, argv, &i);
    } else if (strcmp(k, "--max-world") == 0) {
      a.maxWorld = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
    } else if (strcmp(k, "--min-world") == 0) {
      a.minWorld = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
    } else if (strcmp(k, "--serial-repeats") == 0) {
      a.serialRepeats = basicApiParseIntArg(basicApiRequireValue(argc, argv, &i));
      if (a.serialRepeats <= 0) {
        fprintf(stderr, "--serial-repeats must be positive\n");
        _Exit(2);
      }
    } else if (strcmp(k, "--algorithm") == 0) {
      a.algorithm = basicApiRequireValue(argc, argv, &i);
      if (!allowAlgorithmAll && strcmp(a.algorithm, "all") == 0) {
        fprintf(stderr, "--algorithm all is supported by MPI only\n");
        _Exit(2);
      }
    } else if (strcmp(k, "--copy-algorithm") == 0) {
      a.copyAlgorithm = basicApiRequireValue(argc, argv, &i);
      if (strcmp(a.copyAlgorithm, "direct") != 0 && strcmp(a.copyAlgorithm, "pack") != 0 &&
          strcmp(a.copyAlgorithm, "pipe") != 0) {
        fprintf(stderr, "Unknown copy algorithm '%s'. Use 'direct', 'pack', or 'pipe'\n", a.copyAlgorithm);
        _Exit(2);
      }
    } else if (strcmp(k, "--api") == 0) {
      a.api = basicApiRequireValue(argc, argv, &i);
      if (strcmp(a.api, "window") != 0 && strcmp(a.api, "default") != 0 && strcmp(a.api, "all") != 0) {
        fprintf(stderr, "Unknown api '%s'. Use 'window', 'default', or 'all'\n", a.api);
        _Exit(2);
      }
    } else if (strcmp(k, "--lb-mode") == 0) {
      a.lbMode = basicApiRequireValue(argc, argv, &i);
    } else if (strcmp(k, "--help") == 0) {
      basicApiPrintUsage(argv[0], usageFmt, allowRankCount, allowAlgorithmAll);
      _Exit(0);
    } else {
      fprintf(stderr, "Unknown argument: %s\n", k);
      basicApiPrintUsage(argv[0], usageFmt, allowRankCount, allowAlgorithmAll);
      _Exit(2);
    }
  }
  return a;
}

/* ======================================================================
 * Naming helpers.
 * ====================================================================*/

static const char* plName(PlacementKind p) {
  switch (p) {
  case PL_RS:
    return "rs";
  case PL_SR:
    return "sr";
  case PL_REPL:
    return "repl";
  }
  return "?";
}

static std::string formatGlobalDims(int ndims, const size_t gd[3]) {
  char buf[64];
  if (ndims == 1) snprintf(buf, sizeof(buf), "%zu", gd[0]);
  else if (ndims == 2) snprintf(buf, sizeof(buf), "%zux%zu", gd[0], gd[1]);
  else snprintf(buf, sizeof(buf), "%zux%zux%zu", gd[0], gd[1], gd[2]);
  return buf;
}

static std::string buildCaseName(const TestCase& tc) {
  char buf[256];
  std::string gd = formatGlobalDims(tc.ndims, tc.globalDims);

  if (tc.srcPl == PL_REPL && tc.dstPl == PL_REPL) {
    snprintf(buf, sizeof(buf), "%s[gd=%s,esz=%zu]", tc.group.c_str(), gd.c_str(), tc.elementSize);
  } else if (tc.srcDim0 == 0 && tc.dstDim0 == 0) {
    /* 1D mesh per side */
    snprintf(buf, sizeof(buf), "%s[gd=%s,sd=%d/%d,esz=%zu]", tc.group.c_str(), gd.c_str(), tc.srcShardDim,
             tc.dstShardDim, tc.elementSize);
  } else if (tc.srcRatioNum == 0 && tc.dstRatioNum == 0) {
    snprintf(buf, sizeof(buf), "%s[gd=%s,m=%dx%d_%s/%s,sd=%d/%d,esz=%zu]", tc.group.c_str(), gd.c_str(), tc.srcDim0,
             tc.dstDim0, plName(tc.srcPl), plName(tc.dstPl), tc.srcShardDim, tc.dstShardDim, tc.elementSize);
  } else {
    snprintf(buf, sizeof(buf), "%s[gd=%s,m=%dx%d_%s/%s,sd=%d/%d,esz=%zu,ratio=%d:%d]", tc.group.c_str(), gd.c_str(),
             tc.srcDim0, tc.dstDim0, plName(tc.srcPl), plName(tc.dstPl), tc.srcShardDim, tc.dstShardDim, tc.elementSize,
             tc.srcRatioNum, tc.dstRatioNum);
  }
  std::string name(buf);
  if (tc.dstFirst) name += "[dst-first]";
  if (tc.bNullWindow) name += "[null-window]";
  return name;
}

/* ======================================================================
 * Builders — one per Python test method.
 * ====================================================================*/

static void emitFullReplication(std::vector<TestCase>& cases) {
  /*  test_basic_api_full_replication
   *  - 1D mesh per side, both Replicate()
   *  - Tensor (200, 200), dtypes fp32 / bf16 / uint8
   *  - Pytest skips world < 8; we only need world >= 4 (2 src + 2 dst)
   *    since the C kernel has no inherent 8-rank requirement.
   */
  const size_t esz_list[] = {
    4, /* fp32 (size table) */
    2, /* bf16 (size table) */
    1, /* uint8 (size table default for esz=1) */
  };
  for (size_t esz : esz_list) {
    TestCase tc{};
    tc.group = "full_replication";
    tc.ndims = 2;
    tc.globalDims[0] = 200;
    tc.globalDims[1] = 200;
    tc.srcDim0 = 0; /* 1D mesh */
    tc.dstDim0 = 0;
    tc.srcShardDim = -1;
    tc.dstShardDim = -1;
    tc.srcPl = PL_REPL;
    tc.dstPl = PL_REPL;
    tc.elementSize = esz;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static void emitFullSharding(std::vector<TestCase>& cases) {
  /*  test_basic_api_full_sharding
   *  - 1D mesh per side, both Shard(d)
   *  - sharding_dims ∈ {(0,0),(0,1),(1,0),(1,1)}
   *  - Tensor (200, 200), dtypes fp32 / bf16 / uint8
   *  - Skip if world < 8 or world % 2 != 0
   */
  const int sd_list[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const size_t esz_list[] = {
    4, /* fp32 */
    2, /* bf16 */
    1, /* uint8 */
  };
  for (auto& sd : sd_list) {
    for (size_t esz : esz_list) {
      TestCase tc{};
      tc.group = "full_sharding";
      tc.ndims = 2;
      tc.globalDims[0] = 200;
      tc.globalDims[1] = 200;
      tc.srcDim0 = 0;
      tc.dstDim0 = 0;
      tc.srcShardDim = sd[0];
      tc.dstShardDim = sd[1];
      tc.srcPl = PL_RS; /* 1D mesh: dims={1,N}, placement={REPL, SHARD(d)} */
      tc.dstPl = PL_RS;
      tc.elementSize = esz;
      tc.worldMin = 4;
      tc.worldDivisor = 2;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }
}

static void emit2dPlacementMatrix(std::vector<TestCase>& cases, const char* group, size_t global0, size_t global1,
                                  int nShardsSrc, int nShardsDst, int ratioNumSrc, int ratioNumDst, size_t esz) {
  /* Inner helper used by 2d_placement, uneven_ratio, and
     tensor_size_sensitivity. */
  const int sd_list[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const PlacementKind pl_list[][2] = {
    {PL_SR, PL_SR},
    {PL_RS, PL_RS},
    {PL_SR, PL_RS},
    {PL_RS, PL_SR},
  };
  const int totalRatio = (ratioNumSrc + ratioNumDst);
  const bool even = (ratioNumSrc == 0 && ratioNumDst == 0);

  /* Pytest enforces worldSize >= 8; the C kernel only needs the mesh
   * shapes to divide cleanly, so we use 4 as the floor and let runtime
   * feasibility checks (n_shards divides total, global divides shards)
   * skip configs that don't fit at small world sizes. The divisor is the
   * src+dst ratio sum (pytest's `world % (a+b)` rule), or 2 for an even
   * split.
   */
  int worldMin = 4;
  int worldDivisor;
  if (even) worldDivisor = 2;
  else worldDivisor = totalRatio;

  for (auto& sd : sd_list) {
    for (auto& pl : pl_list) {
      TestCase tc{};
      tc.group = group;
      tc.ndims = 2;
      tc.globalDims[0] = global0;
      tc.globalDims[1] = global1;
      tc.srcDim0 = nShardsSrc;
      tc.dstDim0 = nShardsDst;
      tc.srcShardDim = sd[0];
      tc.dstShardDim = sd[1];
      tc.srcPl = pl[0];
      tc.dstPl = pl[1];
      tc.elementSize = esz;
      tc.worldMin = worldMin;
      tc.worldDivisor = worldDivisor;
      tc.srcRatioNum = ratioNumSrc;
      tc.dstRatioNum = ratioNumDst;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }
}

static void emit2dPlacement(std::vector<TestCase>& cases) {
  /*  test_basic_api_2d_placement
   *  - ratio (1,1)
   *  - n_shards ∈ {(2,4),(4,2),(2,2)}
   *  - Tensor (200, 200), dtype bf16
   */
  const int ns_list[][2] = {{2, 4}, {4, 2}, {2, 2}};
  for (auto& ns : ns_list) {
    emit2dPlacementMatrix(cases, "2d_placement", 200, 200, ns[0], ns[1], 0, 0, /* even split */
                          2); /* bf16 */
  }
}

static void emitUnevenRatio(std::vector<TestCase>& cases) {
  /*  test_basic_api_uneven_ratio
   *  - ratio ∈ {(3,1),(1,3)}
   *  - n_shards (2,2)
   *  - Tensor (240, 240), dtype bf16
   */
  const int ratio_list[][2] = {{3, 1}, {1, 3}};
  for (auto& r : ratio_list) emit2dPlacementMatrix(cases, "uneven_ratio", 240, 240, 2, 2, r[0], r[1], 2);
}

static void emitTensorSizeSensitivity(std::vector<TestCase>& cases) {
  /*  test_basic_api_tensor_size_sensitivity
   *  - n_shards (4,4)
   *  - global_shape ∈ {(576,576),(3072,6144),(3072,3072)}
   *  - dtype bf16
   *  - Skip the (rs,rs) large-shape cases on world < 32 (pytest does too,
   *    but our runtime skips on worldSize < worldMin=8 already; the
   *    fine-grained large-shape skip is mirrored by checking buffer size).
   */
  const size_t shape_list[][2] = {
    {576, 576},
    {3072, 6144},
    {3072, 3072},
  };
  for (auto& s : shape_list) emit2dPlacementMatrix(cases, "tensor_size_sensitivity", s[0], s[1], 4, 4, 0, 0, 2);
}

static void emitNdTensors(std::vector<TestCase>& cases) {
  /*  test_nd_tensors
   *  - 1D mesh per side, Shard(d), single shard
   *  - global_shape ∈ {(64,128,128),(128,64,64)}  (3D only; 4D skipped)
   *  - sharding_dims ∈ {(0,0),(0,1),(0,2),(2,0),(1,0),(1,1),(1,2)}
   *    except the historical (64,128,128), sd=(0,1) case owned by
   *    cross_dim_regression.
   *  - dtype bf16
   *  - Pytest requires world >= 32; the C kernel works with 2 ranks per
   *    side and the chosen 3D shapes are even-divisible by small shard
   *    counts, so we lower the floor to 4 and let runtime feasibility
   *    handle the rest.
   */
  const int sd_list[][2] = {
    {0, 0}, {0, 1}, {0, 2}, {2, 0}, {1, 0}, {1, 1}, {1, 2},
  };
  const size_t shape_list[][3] = {
    {64, 128, 128},
    {128, 64, 64},
  };
  for (auto& shape : shape_list) {
    for (auto& sd : sd_list) {
      if (shape[0] == 64 && shape[1] == 128 && shape[2] == 128 && sd[0] == 0 && sd[1] == 1) continue;
      /* Skip combos that exceed tensor ndim (mirrors pytest skip). */
      if (sd[0] >= 3 || sd[1] >= 3) continue;
      TestCase tc{};
      tc.group = "nd_tensors";
      tc.ndims = 3;
      tc.globalDims[0] = shape[0];
      tc.globalDims[1] = shape[1];
      tc.globalDims[2] = shape[2];
      tc.srcDim0 = 0;
      tc.dstDim0 = 0;
      tc.srcShardDim = sd[0];
      tc.dstShardDim = sd[1];
      tc.srcPl = PL_RS;
      tc.dstPl = PL_RS;
      tc.elementSize = 2; /* bf16 */
      tc.worldMin = 4;
      tc.worldDivisor = 2;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }
}

static void emitStagingSlotPressure(std::vector<TestCase>& cases) {
  /* 2-node CI reproducer for the staging-buffer slot-sizing regression.
   *
   * Shape/placement mirrors the dsv3 expert transfer pattern at small scale:
   * train-style Shard(0) to gen-style Shard(innermost). With the CI allocation
   * of 2 nodes * 2 ranks/node, srcTotal=2 and dstTotal=2. Running this case
   * with NCCL_RESHARD_STAGING_CHANNEL_SIZE=4 MiB leaves each peer slightly less than
   * the old fixed 1 MiB chunk size after the staging-region split, reproducing
   * the same prepare-time invalid-argument failure without a 256-rank tensor.
   */
  TestCase tc{};
  tc.group = "staging_slot_pressure";
  tc.ndims = 3;
  tc.globalDims[0] = 64;
  tc.globalDims[1] = 128;
  tc.globalDims[2] = 128;
  tc.srcDim0 = 0;
  tc.dstDim0 = 0;
  tc.srcShardDim = 0;
  tc.dstShardDim = 2;
  tc.srcPl = PL_RS;
  tc.dstPl = PL_RS;
  tc.elementSize = 2; /* bf16 */
  tc.worldMin = 4;
  tc.worldMax = 4;
  tc.worldDivisor = 2;
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static void emitPipeLsaFanoutReuse(std::vector<TestCase>& cases) {
  /* Four source shards feed four destination replicas on an eight-rank,
   * two-node allocation. Each destination rank owns one RDMA source and
   * receives the remaining three through LSA fan-out. Repeating with a
   * stream sync after each call exercises the persistent flow-control state
   * without relying on overlap between calls. */
  TestCase tc{};
  tc.group = "pipe_lsa_fanout_reuse";
  tc.ndims = 3;
  tc.globalDims[0] = 256;
  tc.globalDims[1] = 128;
  tc.globalDims[2] = 128;
  tc.srcDim0 = 4;
  tc.dstDim0 = 4;
  tc.srcShardDim = 0;
  tc.dstShardDim = 2;
  tc.srcPl = PL_SR;
  tc.dstPl = PL_RS;
  tc.elementSize = 2;
  tc.worldMin = 8;
  tc.worldMax = 8;
  tc.worldDivisor = 8;
  tc.serialRepeats = 8;
  tc.requiredCopyAlgorithm = "PIPE";
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static void emitGraphCapture(std::vector<TestCase>& cases) {
  for (bool bPrewarmed : {false, true}) {
    TestCase tc{};
    tc.group = "graph_capture";
    tc.ndims = 2;
    tc.globalDims[0] = 64;
    tc.globalDims[1] = 64;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = 4;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.bGraphCapture = true;
    tc.bGraphCapturePrewarmed = bPrewarmed;
    tc.name = buildCaseName(tc) + (bPrewarmed ? "_prewarmed" : "_first_use");
    cases.push_back(std::move(tc));
  }
}

/* ======================================================================
 * 1D tensor variants — placement / sharding coverage with ndims=1 and
 * sharding_dim always 0 (only one tensor axis).
 * The pytest source does not exercise 1D; we add coverage here because
 * the lib advertises 1 ≤ ndims ≤ 3 and 1D shards hit kernel paths that
 * the 2D/3D matrix doesn't.
 * ====================================================================*/

static void emit1dFullSharding(std::vector<TestCase>& cases) {
  /* Only sd=(0,0) is meaningful for ndims=1. */
  const size_t esz_list[] = {4, 2, 1};
  for (size_t esz : esz_list) {
    TestCase tc{};
    tc.group = "1d_full_sharding";
    tc.ndims = 1;
    tc.globalDims[0] = 8192;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = esz;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

/* Inner helper for the 2D-mesh-on-1D-tensor groups. Mirrors
 * emit2dPlacementMatrix but with ndims=1 and sd fixed to (0,0).
 */
static void emit1dPlacementMatrix(std::vector<TestCase>& cases, const char* group, size_t global0, int nShardsSrc,
                                  int nShardsDst, int ratioNumSrc, int ratioNumDst, size_t esz) {
  const PlacementKind pl_list[][2] = {
    {PL_SR, PL_SR},
    {PL_RS, PL_RS},
    {PL_SR, PL_RS},
    {PL_RS, PL_SR},
  };
  const int totalRatio = (ratioNumSrc + ratioNumDst);
  const bool even = (ratioNumSrc == 0 && ratioNumDst == 0);

  int worldMin = 4;
  int worldDivisor = even ? 2 : totalRatio;

  for (auto& pl : pl_list) {
    TestCase tc{};
    tc.group = group;
    tc.ndims = 1;
    tc.globalDims[0] = global0;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = nShardsSrc;
    tc.dstDim0 = nShardsDst;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = pl[0];
    tc.dstPl = pl[1];
    tc.elementSize = esz;
    tc.worldMin = worldMin;
    tc.worldDivisor = worldDivisor;
    tc.srcRatioNum = ratioNumSrc;
    tc.dstRatioNum = ratioNumDst;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static void emit1d2dPlacement(std::vector<TestCase>& cases) {
  const int ns_list[][2] = {{2, 4}, {4, 2}, {2, 2}};
  for (auto& ns : ns_list) emit1dPlacementMatrix(cases, "1d_2d_placement", 8192, ns[0], ns[1], 0, 0, 2); /* bf16 */
}

static void emit1dUnevenRatio(std::vector<TestCase>& cases) {
  const int ratio_list[][2] = {{3, 1}, {1, 3}};
  for (auto& r : ratio_list) emit1dPlacementMatrix(cases, "1d_uneven_ratio", 16384, 2, 2, r[0], r[1], 2);
}

static void emit1dTensorSizeSensitivity(std::vector<TestCase>& cases) {
  const size_t shape_list[] = {16384, 1048576, 4194304};
  for (size_t s : shape_list) emit1dPlacementMatrix(cases, "1d_tensor_size_sensitivity", s, 4, 4, 0, 0, 2);
}

/* ======================================================================
 * Cross-dim regression group — hand-picked shapes from historical bugs.
 *
 * Each case is a cross-dim layout that previously exposed a data-layout or
 * transfer-path bug. Curated, intentionally small — meant to be
 * run quickly via `--filter cross_dim_regression` as a fast targeted
 * gate.  Not a substitute for the full 2d_placement / nd_tensors
 * matrices; complements them.
 *
 * Mapping:
 *   issue !4 — 3D, sd=0/1, dstShardDim != ndims-1.
 *   issue !5 — 2D, sd=0/1, mesh 2x4, rs/rs placement.
 *   staging high-fanout — 3D, train-style Shard(0) to gen-style
 *              Shard(innermost) with >INT32_MAX global elements.
 * ====================================================================*/

static void emitTinyContribution(std::vector<TestCase>& cases) {
  /* Every CTA must signal even when a transfer contribution is smaller than
   * the configured CTA count. Full replication keeps each contribution at
   * exactly four bytes regardless of the allocation size. */
  TestCase tc{};
  tc.group = "tiny_contribution";
  tc.ndims = 1;
  tc.globalDims[0] = 4;
  tc.globalDims[1] = 1;
  tc.globalDims[2] = 1;
  tc.srcDim0 = 0;
  tc.dstDim0 = 0;
  tc.srcShardDim = -1;
  tc.dstShardDim = -1;
  tc.srcPl = PL_REPL;
  tc.dstPl = PL_REPL;
  tc.elementSize = 1;
  tc.worldMin = 4;
  tc.worldDivisor = 2;
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static void emitStreamChurn(std::vector<TestCase>& cases) {
  TestCase tc{};
  tc.group = "stream_churn";
  tc.ndims = 2;
  tc.globalDims[0] = 64;
  tc.globalDims[1] = 64;
  tc.srcDim0 = 0;
  tc.dstDim0 = 0;
  tc.srcShardDim = 0;
  tc.dstShardDim = 0;
  tc.srcPl = PL_RS;
  tc.dstPl = PL_RS;
  tc.elementSize = 4;
  tc.worldMin = 4;
  tc.worldDivisor = 2;
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static void emitStreamOrdering(std::vector<TestCase>& cases) {
  TestCase tc{};
  tc.group = "stream_ordering";
  tc.ndims = 2;
  tc.globalDims[0] = 64;
  tc.globalDims[1] = 64;
  tc.srcDim0 = 0;
  tc.dstDim0 = 0;
  tc.srcShardDim = 0;
  tc.dstShardDim = 0;
  tc.srcPl = PL_RS;
  tc.dstPl = PL_RS;
  tc.elementSize = 4;
  tc.worldMin = 4;
  tc.worldDivisor = 2;
  tc.bAsyncOrdering = true;
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static void emitSplitRegressions(std::vector<TestCase>& cases) {
  {
    TestCase tc{};
    tc.group = "split_tiny_contribution";
    tc.ndims = 1;
    tc.globalDims[0] = 4;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = -1;
    tc.dstShardDim = -1;
    tc.srcPl = PL_REPL;
    tc.dstPl = PL_REPL;
    tc.elementSize = 1;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }

  {
    TestCase tc{};
    tc.group = "split_reverse_mesh";
    tc.ndims = 1;
    tc.globalDims[0] = 4096;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = 1;
    tc.worldMin = 4;
    tc.worldDivisor = 2;
    tc.dstFirst = true;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static void emitPackRegressions(std::vector<TestCase>& cases) {
  /* Minimal forward-order source/destination pair for the single-LSA host-RMA
   * activation gate. Two element sizes make the focused local run reuse the
   * same communicator and PACK staging state across consecutive tests. */
  for (size_t elementSize : {2UL, 1UL}) {
    TestCase tc{};
    tc.group = "pack_lsa_hput";
    tc.ndims = 1;
    tc.globalDims[0] = 4096;
    tc.globalDims[1] = 1;
    tc.globalDims[2] = 1;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 0;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = elementSize;
    tc.worldMin = 2;
    tc.worldMax = 2;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }

  /* The non-split PACK producer and consumer must use the same
   * source-relative GIN signal bank. With two source shards, source rank 1
   * uses the second bank instead of aliasing source rank 0. The tiny transfer
   * also leaves trailing CTAs idle, which verifies that each still signals. */
  TestCase tc{};
  tc.group = "pack_signal_bank";
  tc.ndims = 1;
  tc.globalDims[0] = 4;
  tc.globalDims[1] = 1;
  tc.globalDims[2] = 1;
  tc.srcDim0 = 0;
  tc.dstDim0 = 0;
  tc.srcShardDim = 0;
  tc.dstShardDim = 0;
  tc.srcPl = PL_RS;
  tc.dstPl = PL_RS;
  tc.elementSize = 2;
  tc.worldMin = 4;
  tc.worldMax = 4;
  tc.worldDivisor = 2;
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static void emitCrossDimRegression(std::vector<TestCase>& cases) {
  /* issue !5: 2D mesh 2x4, sd=0/1, four placement permutations.
   * worldMin = src_shards * dst_shards = 2 * 4 = 8 (even split → /2). */
  {
    const PlacementKind pl_list[][2] = {
      {PL_SR, PL_SR},
      {PL_RS, PL_RS},
      {PL_SR, PL_RS},
      {PL_RS, PL_SR},
    };
    for (auto& pl : pl_list) {
      TestCase tc{};
      tc.group = "cross_dim_regression";
      tc.ndims = 2;
      tc.globalDims[0] = 200;
      tc.globalDims[1] = 200;
      tc.srcDim0 = 2;
      tc.dstDim0 = 4;
      tc.srcShardDim = 0;
      tc.dstShardDim = 1;
      tc.srcPl = pl[0];
      tc.dstPl = pl[1];
      tc.elementSize = 2; /* bf16 */
      tc.worldMin = 8;
      tc.worldDivisor = 2;
      tc.name = buildCaseName(tc);
      cases.push_back(std::move(tc));
    }
  }

  /* issue !4: 3D, sd=0/1, dstShardDim is not innermost. 1D mesh per
   * side (dim0=0) with rs/rs placement; each side has 4 shards, so
   * worldMin = 8. */
  {
    TestCase tc{};
    tc.group = "cross_dim_regression";
    tc.ndims = 3;
    tc.globalDims[0] = 64;
    tc.globalDims[1] = 128;
    tc.globalDims[2] = 128;
    tc.srcDim0 = 0;
    tc.dstDim0 = 0;
    tc.srcShardDim = 0;
    tc.dstShardDim = 1;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = 2;
    tc.worldMin = 8;
    tc.worldDivisor = 2;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }

  /* dsv3-style MoE expert transfer: train [DP8, EP16] Shard(0) to
   * gen [DP4, TP32] Shard(2).  Global elements =
   * 256 * 7168 * 2048 = 3,758,096,384, above INT32_MAX. */
  {
    TestCase tc{};
    tc.group = "cross_dim_regression";
    tc.ndims = 3;
    tc.globalDims[0] = 256;
    tc.globalDims[1] = 7168;
    tc.globalDims[2] = 2048;
    tc.srcDim0 = 8;
    tc.dstDim0 = 4;
    tc.srcShardDim = 0;
    tc.dstShardDim = 2;
    tc.srcPl = PL_RS;
    tc.dstPl = PL_RS;
    tc.elementSize = 2; /* bf16 */
    tc.worldMin = 256;
    tc.worldMax = 256;
    tc.worldDivisor = 256;
    tc.name = buildCaseName(tc);
    cases.push_back(std::move(tc));
  }
}

static void emitWindowNullCase(std::vector<TestCase>& cases) {
  TestCase tc{};
  tc.group = "window_null";
  tc.ndims = 1;
  tc.globalDims[0] = 4096;
  tc.globalDims[1] = 1;
  tc.globalDims[2] = 1;
  tc.srcDim0 = 0;
  tc.dstDim0 = 0;
  tc.srcShardDim = 0;
  tc.dstShardDim = 0;
  tc.srcPl = PL_RS;
  tc.dstPl = PL_RS;
  tc.elementSize = 1;
  tc.worldMin = 2;
  tc.worldDivisor = 2;
  tc.bNullWindow = true;
  tc.name = buildCaseName(tc);
  cases.push_back(std::move(tc));
}

static std::vector<TestCase> buildAllTestCases() {
  std::vector<TestCase> cases;
  emitFullReplication(cases);
  emitFullSharding(cases);
  emit2dPlacement(cases);
  emitUnevenRatio(cases);
  emitTensorSizeSensitivity(cases);
  emitNdTensors(cases);
  emitStagingSlotPressure(cases);
  emitPipeLsaFanoutReuse(cases);
  emitGraphCapture(cases);
  /* 1D tensor groups (extends pytest matrix). */
  emit1dFullSharding(cases);
  emit1d2dPlacement(cases);
  emit1dUnevenRatio(cases);
  emit1dTensorSizeSensitivity(cases);
  /* Targeted regression coverage for historical cross-dim bugs. */
  emitCrossDimRegression(cases);
  emitWindowNullCase(cases);
  emitTinyContribution(cases);
  emitStreamChurn(cases);
  emitStreamOrdering(cases);
  emitSplitRegressions(cases);
  emitPackRegressions(cases);
  return cases;
}

/* ======================================================================
 * Mesh / shard math (no NCCL calls).
 * ====================================================================*/

struct MeshLayout {
  int dims[NCCL_RESHARD_MESH_NDIMS];
  int placement[NCCL_RESHARD_MESH_NDIMS];
  int startRank;
  int shardCount; /* 1 if PL_REPL */
  int shardDim; /* -1 if PL_REPL */
};

static int shardCountForMesh(const TestCase& tc, bool isSrc, int dim0, int dim1) {
  PlacementKind pl = isSrc ? tc.srcPl : tc.dstPl;
  switch (pl) {
  case PL_REPL:
    return 1;
  case PL_RS:
    return dim1; /* axis 1 is shard */
  case PL_SR:
    return dim0; /* axis 0 is shard */
  }
  return 1;
}

static int shardIdxForRank(const MeshLayout& m, PlacementKind pl, int rank) {
  int local = rank - m.startRank;
  switch (pl) {
  case PL_REPL:
    return 0;
  case PL_RS:
    return local % m.dims[1];
  case PL_SR:
    return local / m.dims[1];
  }
  return 0;
}

static void buildMesh(MeshLayout* out, PlacementKind pl, int shardDim, int dim0, int dim1, int startRank) {
  out->dims[0] = dim0;
  out->dims[1] = dim1;
  out->startRank = startRank;
  out->shardDim = (pl == PL_REPL) ? -1 : shardDim;
  if (pl == PL_REPL) {
    /* Encode "full replication" as a 1-shard PL_RS layout: every
     * rank still owns the full tensor, shardCount = 1 keeps the
     * expected global-range math simple, and the kernel goes through
     * the well-tested sharded path. A {REPLICATE, REPLICATE} mesh
     * lands in a degenerate prepare branch that the test suite does
     * not currently exercise.
     */
    out->placement[0] = NCCL_RESHARD_REPLICATE;
    out->placement[1] = NCCL_RESHARD_SHARD(0);
    out->shardCount = 1;
  } else if (pl == PL_RS) {
    out->placement[0] = NCCL_RESHARD_REPLICATE;
    out->placement[1] = NCCL_RESHARD_SHARD(shardDim);
    out->shardCount = dim1;
  } else { /* PL_SR */
    out->placement[0] = NCCL_RESHARD_SHARD(shardDim);
    out->placement[1] = NCCL_RESHARD_REPLICATE;
    out->shardCount = dim0;
  }
}

/* ======================================================================
 * Per-case execution result.
 * ====================================================================*/

enum CaseStatus {
  CASE_PASS,
  CASE_FAIL,
  CASE_SKIP
};

struct CaseResult {
  CaseStatus status;
  const char* skipReason; /* set when status == CASE_SKIP */
  const char* failReason; /* set when status == CASE_FAIL */
};

static inline CaseResult makeSkip(const char* reason) {
  return CaseResult{CASE_SKIP, reason, nullptr};
}
static inline CaseResult makeFail(const char* reason) {
  return CaseResult{CASE_FAIL, nullptr, reason};
}
static inline CaseResult makePass() {
  return CaseResult{CASE_PASS, nullptr, nullptr};
}

/* ======================================================================
 * Feasibility helpers — shared between runOneCase() and
 * computeMinWorldForCase().
 * ====================================================================*/

struct CaseShape {
  int srcTotal, dstTotal;
  int srcDim0, srcDim1;
  int dstDim0, dstDim1;
  int srcShardCount, dstShardCount;
};

/* Returns true if test case `tc` is feasible at worldSize W. On false,
 * `*skipReason` (if non-null) is set to a static string describing why.
 * On true, `*shape` (if non-null) is filled with the resolved layout.
 *
 * This is the single source of truth for "does this case run at world W?".
 * runOneCase() uses it for the feasibility phase; computeMinWorldForCase()
 * iterates over W until it returns true.
 */
static bool caseFeasibleAt(const TestCase& tc, int w, CaseShape* shape = nullptr, const char** skipReason = nullptr) {
  auto fail = [&](const char* r) {
    if (skipReason != nullptr) *skipReason = r;
    return false;
  };

  if (w < tc.worldMin) return fail("worldSize below minimum");
  if (tc.worldMax > 0 && w > tc.worldMax) {
    return fail("worldSize above maximum");
  }
  if (tc.worldDivisor != 0 && (w % tc.worldDivisor) != 0) return fail("worldSize not divisible by required factor");

  int srcTotal, dstTotal;
  if (tc.srcRatioNum == 0 && tc.dstRatioNum == 0) {
    srcTotal = w / 2;
    dstTotal = w - srcTotal;
  } else {
    int totalRatio = tc.srcRatioNum + tc.dstRatioNum;
    srcTotal = w * tc.srcRatioNum / totalRatio;
    dstTotal = w - srcTotal;
  }
  if (srcTotal + dstTotal != w || srcTotal == 0 || dstTotal == 0) return fail("ratio yields empty side");

  int srcDim0, srcDim1, dstDim0, dstDim1;
  if (tc.srcPl == PL_REPL) {
    srcDim0 = srcTotal;
    srcDim1 = 1;
  } else {
    srcDim0 = (tc.srcDim0 == 0) ? 1 : tc.srcDim0;
    if (srcTotal % srcDim0 != 0) return fail("srcTotal not divisible by srcDim0");
    srcDim1 = srcTotal / srcDim0;
  }
  if (tc.dstPl == PL_REPL) {
    dstDim0 = dstTotal;
    dstDim1 = 1;
  } else {
    dstDim0 = (tc.dstDim0 == 0) ? 1 : tc.dstDim0;
    if (dstTotal % dstDim0 != 0) return fail("dstTotal not divisible by dstDim0");
    dstDim1 = dstTotal / dstDim0;
  }

  int srcShardCount = shardCountForMesh(tc, /*isSrc=*/true, srcDim0, srcDim1);
  int dstShardCount = shardCountForMesh(tc, /*isSrc=*/false, dstDim0, dstDim1);

  if (tc.srcShardDim >= 0 && tc.globalDims[tc.srcShardDim] % (size_t)srcShardCount != 0)
    return fail("global dim not divisible by src shard count");
  if (tc.dstShardDim >= 0 && tc.globalDims[tc.dstShardDim] % (size_t)dstShardCount != 0)
    return fail("global dim not divisible by dst shard count");

  if (shape != nullptr) {
    shape->srcTotal = srcTotal;
    shape->dstTotal = dstTotal;
    shape->srcDim0 = srcDim0;
    shape->srcDim1 = srcDim1;
    shape->dstDim0 = dstDim0;
    shape->dstDim1 = dstDim1;
    shape->srcShardCount = srcShardCount;
    shape->dstShardCount = dstShardCount;
  }
  return true;
}

/* Smallest world W (>= worldMin, <= bound) at which this case is
 * feasible. Returns -1 if no W up to `bound` works.
 */
static int computeMinWorldForCase(const TestCase& tc, int bound = 4096) {
  int divisor = tc.worldDivisor > 0 ? tc.worldDivisor : 1;
  int start = tc.worldMin;
  /* Round start up to the next multiple of divisor. */
  if (start % divisor != 0) start += divisor - (start % divisor);
  if (start < divisor) start = divisor;
  int stop = (tc.worldMax > 0 && tc.worldMax < bound) ? tc.worldMax : bound;
  for (int w = start; w <= stop; w += divisor) {
    if (caseFeasibleAt(tc, w)) return w;
  }
  return -1;
}

/* ======================================================================
 * Buffer-size estimator.
 * ====================================================================*/

static size_t localBytesForShard(const TestCase& tc, int shardDim, int shardCount) {
  size_t localDims[3] = {tc.globalDims[0], tc.globalDims[1], tc.globalDims[2]};
  if (shardDim >= 0) {
    if (shardCount <= 0) {
      return 0;
    }
    localDims[shardDim] /= (size_t)shardCount;
  }

  int innermost = tc.ndims - 1;
  localDims[innermost] *= tc.elementSize;

  size_t total = localDims[0];
  for (int d = 1; d < tc.ndims; d++) {
    total *= localDims[d];
  }
  return total;
}

static size_t maxLocalBytes(const TestCase& tc, int worldSize) {
  CaseShape shape = {};
  if (!caseFeasibleAt(tc, worldSize, &shape)) {
    return 0;
  }

  size_t srcBytes = localBytesForShard(tc, tc.srcShardDim, shape.srcShardCount);
  size_t dstBytes = localBytesForShard(tc, tc.dstShardDim, shape.dstShardCount);
  return std::max(srcBytes, dstBytes);
}

static size_t computeMaxBufferBytes(const std::vector<TestCase>& cases, int worldSize) {
  size_t mx = 4096; /* NCCL min alloc */
  for (auto& tc : cases) {
    size_t need = maxLocalBytes(tc, worldSize);
    if (need > mx) mx = need;
  }
#ifdef NCCL_M2N_TESTING
  /* Keep two disjoint tensor regions for the PACK fusion assertion. */
  mx *= 2;
#endif
  /* Round up to 4 KiB. */
  const size_t pg = 4096;
  return ((mx + pg - 1) / pg) * pg;
}

static bool caseMatchesSelection(const TestCase& tc, const char* filter, int minWorld, int maxWorld) {
  if (filter != nullptr && filter[0] != '\0' && strstr(tc.name.c_str(), filter) == nullptr) return false;

  if (maxWorld > 0 || minWorld > 0) {
    int mw = computeMinWorldForCase(tc);
    if (mw < 0) return false;
    if (maxWorld > 0 && mw > maxWorld) return false;
    if (minWorld > 0 && mw < minWorld) return false;
  }
  return true;
}

static const char* basicApiSelectedCopyAlgorithm(const BasicApiCliArgs& cli) {
  if (cli.copyAlgorithm != nullptr) {
    if (strcmp(cli.copyAlgorithm, "direct") == 0) return "DIRECT";
    if (strcmp(cli.copyAlgorithm, "pipe") == 0) return "PIPE";
    return "PACK";
  }

  // NOLINTNEXTLINE(concurrency-mt-unsafe) — test selection runs before worker threads start
  const char* copyAlgorithmEnv = getenv("NCCL_RESHARD_COPY_ALGORITHM");
  if (copyAlgorithmEnv != nullptr) {
    if (strcasecmp(copyAlgorithmEnv, "DIRECT") == 0) return "DIRECT";
    if (strcasecmp(copyAlgorithmEnv, "PIPE") == 0) return "PIPE";
    return "PACK";
  }
  return strcmp(cli.algorithm, "direct") == 0 ? "DIRECT" : "PACK";
}

static std::vector<TestCase> basicApiSelectCases(const std::vector<TestCase>& cases, const BasicApiCliArgs& cli) {
  std::vector<TestCase> selected;
  const char* copyAlgorithm = basicApiSelectedCopyAlgorithm(cli);
  for (const TestCase& tc : cases) {
    if (!caseMatchesSelection(tc, cli.filter, cli.minWorld, cli.maxWorld)) {
      continue;
    }
    if (tc.requiredCopyAlgorithm != nullptr && strcasecmp(tc.requiredCopyAlgorithm, copyAlgorithm) != 0) {
      continue;
    }
    TestCase selectedCase = tc;
    if (cli.serialRepeats > 0 && selectedCase.serialRepeats > 1) {
      selectedCase.serialRepeats = cli.serialRepeats;
    }
    selected.push_back(std::move(selectedCase));
  }
  return selected;
}

static std::string basicApiGtestCaseName(const std::string& caseName, size_t index, const char* prefix) {
  char indexBuf[48];
  if (prefix != nullptr) snprintf(indexBuf, sizeof(indexBuf), "%s_case%04zu_", prefix, index);
  else snprintf(indexBuf, sizeof(indexBuf), "case%04zu_", index);

  std::string out = indexBuf;
  for (unsigned char ch : caseName) out.push_back((std::isalnum(ch) != 0) ? (char)ch : '_');
  return out;
}

static void basicApiPrintCaseList(const std::vector<TestCase>& cases, const BasicApiCliArgs& cli, bool shouldPrint) {
  if (!shouldPrint) return;

  printf("# total_cases=%zu\n", cases.size());
  printf("# columns: idx minWorld name\n");
  for (size_t i = 0; i < cases.size(); ++i) {
    const TestCase& tc = cases[i];
    if (!caseMatchesSelection(tc, cli.filter, cli.minWorld, cli.maxWorld)) continue;
    int mw = computeMinWorldForCase(tc);
    if (mw > 0) printf("[%4zu] %4d %s\n", i, mw, tc.name.c_str());
    else printf("[%4zu]    - %s  // no feasible world\n", i, tc.name.c_str());
  }
}

static bool basicApiRunAllAlgorithms(const BasicApiCliArgs& cli) {
  return strcmp(cli.algorithm, "all") == 0;
}

static bool basicApiRunAllApis(const BasicApiCliArgs& cli) {
  return strcmp(cli.api, "all") == 0;
}

static ApiKind basicApiRequestedApi(const BasicApiCliArgs& cli) {
  if (strcmp(cli.api, "default") == 0) {
    return ApiKind::Default;
  }
  return ApiKind::Window;
}

static const char* basicApiRequestedAlgorithmEnv(const BasicApiCliArgs& cli, bool shouldPrintUnknown) {
  if (strcmp(cli.algorithm, "direct") == 0) return "DIRECT";
  if (strcmp(cli.algorithm, "ring") == 0 || basicApiRunAllAlgorithms(cli)) return "RING";

  if (shouldPrintUnknown) fprintf(stderr, "Unknown algorithm '%s', defaulting to ring\n", cli.algorithm);
  return "RING";
}

static const char* basicApiConfigureReshardEnv(const BasicApiCliArgs& cli, const char* algorithmEnv) {
  testSetEnv("NCCL_RESHARD_ALGORITHM", algorithmEnv);
  static bool inheritedCopyAlgorithmCaptured = false;
  static const char* inheritedCopyAlgorithm = nullptr;
  if (!inheritedCopyAlgorithmCaptured) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe) — serialized test configuration
    const char* copyAlgorithmEnv = getenv("NCCL_RESHARD_COPY_ALGORITHM");
    if (copyAlgorithmEnv != nullptr) {
      if (strcasecmp(copyAlgorithmEnv, "DIRECT") == 0) {
        inheritedCopyAlgorithm = "DIRECT";
      } else if (strcasecmp(copyAlgorithmEnv, "PIPE") == 0) {
        inheritedCopyAlgorithm = "PIPE";
      } else {
        inheritedCopyAlgorithm = "PACK";
      }
    }
    inheritedCopyAlgorithmCaptured = true;
  }

  const char* copyAlgorithm = nullptr;
  if (cli.copyAlgorithm != nullptr) {
    if (strcmp(cli.copyAlgorithm, "direct") == 0) {
      copyAlgorithm = "DIRECT";
    } else if (strcmp(cli.copyAlgorithm, "pipe") == 0) {
      copyAlgorithm = "PIPE";
    } else {
      copyAlgorithm = "PACK";
    }
    testSetEnv("NCCL_RESHARD_COPY_ALGORITHM", copyAlgorithm);
  } else if (inheritedCopyAlgorithm != nullptr) {
    copyAlgorithm = inheritedCopyAlgorithm;
  } else {
    copyAlgorithm = strcmp(algorithmEnv, "DIRECT") == 0 ? "DIRECT" : "PACK";
    testSetEnv("NCCL_RESHARD_COPY_ALGORITHM", copyAlgorithm);
  }
  testSetEnv("NCCL_RESHARD_LB_MODE", strcmp(cli.lbMode, "node") == 0 ? "NODE_AWARE" : "UNIFORM");
  if (cli.verbose) testSetEnv("NCCL_RESHARD_LOG_LEVEL", "DEBUG");
  return copyAlgorithm;
}

static void basicApiPrintRuntimeSummary(const char* title, int worldSize, int deviceCount,
                                        const BasicApiCliArgs& cli, const char* copyAlgorithm, size_t bufferBytes,
                                        const char* countLabel, size_t count, bool shouldPrint) {
  if (!shouldPrint) return;

  printf("=== %s ===\n", title);
  printf("worldSize=%d, devices=%d, algo=%s, copy=%s, lb=%s, api=%s\n", worldSize, deviceCount, cli.algorithm,
         copyAlgorithm, cli.lbMode, cli.api);
  printf("bufferBytes=%zu, %s=%zu\n", bufferBytes, countLabel, count);
  if (cli.filter != nullptr) printf("filter='%s'\n", cli.filter);
  if (cli.maxWorld > 0) printf("maxWorld=%d\n", cli.maxWorld);
  if (cli.minWorld > 0) printf("minWorld=%d\n", cli.minWorld);
  printf("\n");
  fflush(stdout);
}

/* ======================================================================
 * Per-case driver.
 *
 * Returns CaseResult; rank-aggregation happens in the bootstrap loop.
 * ====================================================================*/

static inline bool isPipeHostRmaAlgo() {
  const char* copyAlgorithm = getenv("NCCL_RESHARD_COPY_ALGORITHM");
  const char* pipeNetMode = getenv("NCCL_RESHARD_PIPE_NET_MODE");
  return copyAlgorithm != nullptr && strcasecmp(copyAlgorithm, "PIPE") == 0 && pipeNetMode != nullptr &&
         (strcasecmp(pipeNetMode, "HOST_RMA") == 0 || strcasecmp(pipeNetMode, "HOST-RMA") == 0 ||
          strcasecmp(pipeNetMode, "HOSTRMA") == 0);
}

/* ncclReshard rejects PIPE host-RMA before NCCL 2.31.2; see #139. */
static inline bool skipPipeHostRmaTest() {
  int ncclVersion = 0;
  return ncclGetVersion(&ncclVersion) != ncclSuccess || ncclVersion < NCCL_VERSION(2, 31, 2);
}

static inline CaseResult checkPipeHostRmaSupport(TestEnv* env) {
  if (isPipeHostRmaAlgo()) {
    int driverVersion = 0;
    if (cudaDriverGetVersion(&driverVersion) != cudaSuccess) driverVersion = -1;
    const int minDriverVersion = env->allreduceMinInt(env, driverVersion);
    if (minDriverVersion < 0) return makeFail("failed to query the CUDA driver version");
    if (minDriverVersion < 12050) return makeSkip("PIPE host-RMA requires CUDA driver API 12.5 or newer");
    if (skipPipeHostRmaTest()) {
      return makeSkip("PIPE host-RMA requires NCCL 2.31.2 or newer; older NCCL can hang on idle ranks");
    }
  }
  return makePass();
}

static CaseResult runOneCase(const TestCase& tc, TestEnv* env) {
  const bool asyncOrdering = tc.bAsyncOrdering && env->alternateStream != nullptr;
  /* ----- 1. feasibility check (single source of truth, see
   * caseFeasibleAt). On skip, also include the minimum world that
   * would let the case run so the user can plan a larger allocation.
   */
  CaseShape shape;
  const char* baseReason = nullptr;
  if (!caseFeasibleAt(tc, env->worldSize, &shape, &baseReason)) {
    static thread_local char buf[160];
    int minWorld = computeMinWorldForCase(tc);
    if (minWorld > 0) {
      snprintf(buf, sizeof(buf), "%s (needs world >= %d)", (baseReason != nullptr) ? baseReason : "infeasible",
               minWorld);
    } else {
      snprintf(buf, sizeof(buf), "%s (no feasible world)", (baseReason != nullptr) ? baseReason : "infeasible");
    }
    return makeSkip(buf);
  }
  const CaseResult support = checkPipeHostRmaSupport(env);
  if (support.status != CASE_PASS) return support;

  int srcTotal = shape.srcTotal;
  int dstTotal = shape.dstTotal;
  int srcDim0 = shape.srcDim0, srcDim1 = shape.srcDim1;
  int dstDim0 = shape.dstDim0, dstDim1 = shape.dstDim1;
  int srcShardCount = shape.srcShardCount;
  int dstShardCount = shape.dstShardCount;

  /* ----- 2. build mesh layouts ----- */
  MeshLayout srcLayout, dstLayout;
  const int srcStart = tc.dstFirst ? dstTotal : 0;
  const int dstStart = tc.dstFirst ? 0 : srcTotal;
  buildMesh(&srcLayout, tc.srcPl, tc.srcShardDim, srcDim0, srcDim1,
            /*startRank=*/srcStart);
  buildMesh(&dstLayout, tc.dstPl, tc.dstShardDim, dstDim0, dstDim1,
            /*startRank=*/dstStart);

  /* ----- 3. determine role and per-rank local dims (in elements) ----- */
  bool isSrc = (env->rank >= srcStart && env->rank < srcStart + srcTotal);
  bool isDst = (env->rank >= dstStart && env->rank < dstStart + dstTotal);

  size_t srcLocalDimsElems[3] = {tc.globalDims[0], tc.globalDims[1], tc.globalDims[2]};
  size_t dstLocalDimsElems[3] = {tc.globalDims[0], tc.globalDims[1], tc.globalDims[2]};
  if (tc.srcShardDim >= 0) srcLocalDimsElems[tc.srcShardDim] /= (size_t)srcShardCount;
  if (tc.dstShardDim >= 0) dstLocalDimsElems[tc.dstShardDim] /= (size_t)dstShardCount;

  /* Local *byte* dims used by the validator: multiply innermost dim by
   * elementSize (the validator works at byte granularity).
   */
  size_t srcLocalBytesDims[3] = {srcLocalDimsElems[0], srcLocalDimsElems[1], srcLocalDimsElems[2]};
  size_t dstLocalBytesDims[3] = {dstLocalDimsElems[0], dstLocalDimsElems[1], dstLocalDimsElems[2]};
  int innermost = tc.ndims - 1;
  srcLocalBytesDims[innermost] *= tc.elementSize;
  dstLocalBytesDims[innermost] *= tc.elementSize;

  size_t myBytes = isSrc ? srcLocalBytesDims[0] * srcLocalBytesDims[1] * (tc.ndims == 3 ? srcLocalBytesDims[2] : 1) :
                           dstLocalBytesDims[0] * dstLocalBytesDims[1] * (tc.ndims == 3 ? dstLocalBytesDims[2] : 1);

  /* ----- 5b. select active buffer ----- */
  void* activeBuffer;
  size_t activeBufferBytes;
  if (env->apiKind == ApiKind::Default) {
    activeBuffer = env->copyBuffer;
    activeBufferBytes = env->copyBufferBytes;
    if (activeBuffer == nullptr) {
      return makeSkip("Default API selected but copyBuffer not allocated");
    }
  } else {
    activeBuffer = env->buffer;
    activeBufferBytes = env->bufferBytes;
  }
#ifdef NCCL_M2N_TESTING
  /* Selects the grouped two-tensor fusion scenario, which halves the buffer and
   * issues a grouped pair through the selected entry point. The window entry
   * point is a compatibility alias for the default one, so it must fuse the
   * same way -- that is what this MR claims, so both kinds run the scenario. */
  const bool testPackFusion = env->expectPack && !asyncOrdering && tc.serialRepeats <= 1;
#else
  const bool testPackFusion = false;
#endif
  const size_t tensorRegionBytes = testPackFusion ? activeBufferBytes / 2 : activeBufferBytes;
  if (myBytes > tensorRegionBytes) {
    return makeSkip("local buffer exceeds preallocated max");
  }

  /* ----- 6. window registration (window path only) ----- */
  ncclWindow_t window = nullptr;
  if (env->apiKind == ApiKind::Window && !tc.bNullWindow) {
    TEST_NCCL_COMM_CHECK(env->comm, ncclCommWindowRegister(env->comm, activeBuffer, activeBufferBytes, &window,
                                                           NCCL_WIN_COLL_SYMMETRIC));
  }
  TEST_CUDACHECK(cudaMemsetAsync(activeBuffer, 0xDE, activeBufferBytes, env->stream));

  /* ----- 7. init source data ----- */
  if (isSrc) {
    int srcShardIdx = shardIdxForRank(srcLayout, tc.srcPl, env->rank);
    int sd = (tc.srcShardDim >= 0) ? tc.srcShardDim : -1;
    int sc = (tc.srcShardDim >= 0) ? srcShardCount : 1;
    testInitSourceData((char*)activeBuffer, srcLocalBytesDims, tc.ndims, sd, srcShardIdx, sc, env->stream);
#ifdef NCCL_M2N_TESTING
    if (testPackFusion) {
      testInitSourceData((char*)activeBuffer + tensorRegionBytes, srcLocalBytesDims, tc.ndims, sd, srcShardIdx, sc,
                         env->stream);
    }
#endif
  }
  if (!asyncOrdering) {
    TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
  }
  env->barrier(env);

  /* ----- 8. resharding call ----- */
  int srcMeshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {srcLayout.dims[0], srcLayout.dims[1]};
  int dstMeshDims[NCCL_RESHARD_MAX_MESH_DIMS] = {dstLayout.dims[0], dstLayout.dims[1]};
  ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
  ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
  srcMesh.ndims = NCCL_RESHARD_MAX_MESH_DIMS;
  srcMesh.dims = srcMeshDims;
  srcMesh.startRank = srcLayout.startRank;
  dstMesh.ndims = NCCL_RESHARD_MAX_MESH_DIMS;
  dstMesh.dims = dstMeshDims;
  dstMesh.startRank = dstLayout.startRank;

  /* The harness works at byte granularity, so pass the dtype whose
   * size matches tc.elementSize (1 / 2 / 4 / 8). */
  static const ncclDataType_t dtype_for_size[] = {
    /* 0 */ ncclInt8,
    /* 1 */ ncclInt8,
    /* 2 */ ncclBfloat16,
    /* 3 */ ncclInt8, /* unreachable */
    /* 4 */ ncclFloat32,
    /* 5 */ ncclInt8,     ncclInt8, ncclInt8,
    /* 8 */ ncclFloat64,
  };
  ncclDataType_t dtype = (tc.elementSize <= 8) ? dtype_for_size[tc.elementSize] : ncclBfloat16;

  int srcPlacements[NCCL_RESHARD_MAX_MESH_DIMS] = {srcLayout.placement[0], srcLayout.placement[1]};
  int dstPlacements[NCCL_RESHARD_MAX_MESH_DIMS] = {dstLayout.placement[0], dstLayout.placement[1]};
  ncclDistTensor_t srcT = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  srcT.dataPtr = isSrc ? activeBuffer : nullptr;
  srcT.localShape = srcLocalDimsElems;
  srcT.ndims = tc.ndims;
  srcT.dtype = dtype;
  srcT.mesh = &srcMesh;
  srcT.placements = srcPlacements;

  ncclDistTensor_t dstT = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  dstT.dataPtr = isDst ? activeBuffer : nullptr;
  dstT.localShape = dstLocalDimsElems;
  dstT.ndims = tc.ndims;
  dstT.dtype = dtype;
  dstT.mesh = &dstMesh;
  dstT.placements = dstPlacements;

  /* Every call below goes through this one helper, so a scenario that issues
   * several calls stays on the selected entry point for all of them. */
  auto reshardTensors = [&](ncclDistTensor_t* src, ncclDistTensor_t* dst, cudaStream_t callStream) {
    if (env->apiKind == ApiKind::Default) {
      return ncclReshard(env->m2nHandle, env->comm, src, dst, callStream);
    }
    return ncclReshardWithWindow(env->m2nHandle, env->comm, tc.bNullWindow ? nullptr : window, src, dst, callStream);
  };
  auto reshard = [&](cudaStream_t callStream) { return reshardTensors(&srcT, &dstT, callStream); };

  auto validateDest = [&]() {
    if (!isDst) {
      return true;
    }
    int dstShardIdx = shardIdxForRank(dstLayout, tc.dstPl, env->rank);
    int sd = (tc.dstShardDim >= 0) ? tc.dstShardDim : -1;
    int sc = (tc.dstShardDim >= 0) ? dstShardCount : 1;
    return testValidateDestData((const char*)activeBuffer, dstLocalBytesDims, tc.ndims, sd, dstShardIdx, sc,
                                env->rank, env->stream, nullptr);
  };

  cudaGraph_t graph = nullptr;
  ncclResult_t r = ncclSuccess;
  bool serialValidationPassed = true;
  if (tc.bGraphCapture) {
    if (tc.bGraphCapturePrewarmed) {
      r = reshard(env->stream);
      if (r == ncclSuccess) {
        if (testUsesNonBlockingComm()) {
          TEST_NCCL_ASYNC_CHECK(env->comm, env->stream);
        } else {
          TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
        }
        env->barrier(env);
      }
    }
    if (r == ncclSuccess) {
      TEST_CUDACHECK(cudaStreamBeginCapture(env->stream, cudaStreamCaptureModeGlobal));
      r = reshard(env->stream);
      TEST_CUDACHECK(cudaStreamEndCapture(env->stream, &graph));
    }
    if (graph != nullptr) {
      TEST_CUDACHECK(cudaGraphDestroy(graph));
    }
    if (window != nullptr) {
      TEST_NCCL_COMM_CHECK(env->comm, ncclCommWindowDeregister(env->comm, window));
    }
    const bool bActionableError = strstr(ncclM2nGetLastError(), "does not support CUDA graph capture") != nullptr;
    return (r == ncclInvalidUsage && bActionableError) ? makePass() : makeFail("graph capture was not rejected");
  } else if (asyncOrdering) {
    for (int call = 0; call < 3; call++) {
      cudaStream_t callStream = (call % 2 == 0) ? env->stream : env->alternateStream;
      r = reshard(callStream);
      if (call + 1 < 3) {
        env->barrier(env);
      }
      if (r != ncclSuccess) {
        break;
      }
    }
  } else if (tc.serialRepeats > 1) {
    for (int call = 0; call < tc.serialRepeats; call++) {
      r = reshard(env->stream);
      const int localCallOk = (r == ncclSuccess) ? 1 : 0;
      if (env->allreduceMinInt(env, localCallOk) == 0) {
        r = ncclSystemError;
        break;
      }
      TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
      env->barrier(env);
      const int localDataOk = validateDest() ? 1 : 0;
      if (env->allreduceMinInt(env, localDataOk) == 0) {
        serialValidationPassed = false;
        break;
      }
      env->barrier(env);
    }
#ifdef NCCL_M2N_TESTING
  } else if (testPackFusion) {
    /* Both entries go through reshardTensors, so the window kind submits two
     * grouped ncclReshardWithWindow calls and the assertion below observes the
     * alias fusing rather than the default entry point standing in for it. */
    ncclDistTensor_t srcT2 = srcT;
    ncclDistTensor_t dstT2 = dstT;
    if (srcT2.dataPtr != nullptr) srcT2.dataPtr = (char*)srcT2.dataPtr + tensorRegionBytes;
    if (dstT2.dataPtr != nullptr) dstT2.dataPtr = (char*)dstT2.dataPtr + tensorRegionBytes;
    r = ncclM2nGroupStart();
    if (r == ncclSuccess) r = reshardTensors(&srcT, &dstT, env->stream);
    if (r == ncclSuccess) r = reshardTensors(&srcT2, &dstT2, env->stream);
    if (r == ncclSuccess) r = ncclM2nGroupEnd();
    if (r != ncclSuccess) (void)ncclM2nGroupAbort();
#endif
  } else {
    r = reshard(env->stream);
  }

  if (r != ncclSuccess) {
    if (window != nullptr) {
      TEST_NCCL_COMM_CHECK(env->comm, ncclCommWindowDeregister(env->comm, window));
    }
    return makeFail("reshard call returned error");
  }

  if (testUsesNonBlockingComm()) {
    TEST_NCCL_ASYNC_CHECK(env->comm, env->stream);
  } else if (!asyncOrdering) {
    TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
  }
  /* These observations are rank-local, so a rank must not return on them before
   * the barrier below -- its peers would still enter, and the harness would
   * deadlock rather than report the failure. Record and report after. */
  const char* instrumentationFailure = nullptr;
#ifdef NCCL_M2N_TESTING
  if (testPackFusion) {
    if (reshardGetFusedSubmissionCountForTest() != 1) {
      instrumentationFailure = "compatible grouped tensors did not produce exactly one fused PACK submission";
    }
  } else if (!asyncOrdering && env->expectPack &&
             reshardGetLastCompletedCopyAlgorithmForTest() != RESHARD_COPY_ALGO_PACK) {
    instrumentationFailure = "selected API did not execute the selected PACK path";
  }
#endif
  env->barrier(env);

  /* ----- 9. validate dest ----- */
  int localOk = 1;
  if (isDst) {
    bool ok = validateDest();
#ifdef NCCL_M2N_TESTING
    if (testPackFusion) {
      int dstShardIdx = shardIdxForRank(dstLayout, tc.dstPl, env->rank);
      int sd = (tc.dstShardDim >= 0) ? tc.dstShardDim : -1;
      int sc = (tc.dstShardDim >= 0) ? dstShardCount : 1;
      ok = ok && testValidateDestData((const char*)activeBuffer + tensorRegionBytes, dstLocalBytesDims, tc.ndims, sd,
                                      dstShardIdx, sc, env->rank, env->stream, nullptr);
    }
#endif
    localOk = ok ? 1 : 0;
  }
  if (asyncOrdering) {
    TEST_CUDACHECK(cudaStreamSynchronize(env->stream));
  }
  /* Fold the rank-local instrumentation result into the same reduction as data
   * validation. Returning on it separately would leave a failing rank out of
   * the allreduce its peers still enter. */
  if (instrumentationFailure != nullptr) {
    localOk = 0;
  }
  if (!serialValidationPassed) {
    localOk = 0;
  }
  int globalOk = env->allreduceMinInt(env, localOk);

  if (window != nullptr) {
    TEST_NCCL_COMM_CHECK(env->comm, ncclCommWindowDeregister(env->comm, window));
  }

  if (instrumentationFailure != nullptr) {
    return makeFail(instrumentationFailure);
  }

  if (env->verbose && env->isRank0Printer(env))
    printf("    [rank %d] localOk=%d, globalOk=%d, myBytes=%zu\n", env->rank, localOk, globalOk, myBytes);

  return (globalOk != 0) ? makePass() : makeFail("byte-pattern mismatch");
}

#endif /* TESTS_BASIC_API_TEST_CORE_H_ */
