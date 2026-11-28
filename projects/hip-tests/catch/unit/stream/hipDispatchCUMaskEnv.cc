/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipDispatchCUMaskEnv hipDispatchCUMaskEnv
 * @{
 * @ingroup StreamTest
 * Validates the per-dispatch CU mask driven by the HIP_DISPATCH_CU_MASK runtime flag.
 *
 * The flag holds the packed launch descriptor cu_enable DWORD (hex):
 * START [3:0], CNT [7:4] (0 = all CUs in the SE), SE_EN [9:8]. It is read while the runtime
 * initializes, so every mask value runs the workload in a fresh child of this binary.
 *
 * The child launches one kernel in which every thread records where it ran, using __smid().
 * On gfx1250 __smid() returns WGP [3:0], shader array [4], SE [8:5], XCC [12:9].
 *
 * Only SE restriction and subset/ordering of the used WGP sets are checked. The number of WGPs
 * is not compared against CNT, because the CU index used by START/CNT does not map one to one
 * onto WGP ids.
 *
 * Requires gfx1250 (launch descriptor / format 4 dispatch packet). AMD + Linux only.
 */

#include <hip_test_common.hh>

#if HT_AMD && HT_LINUX

#include <hip_test_aqllog.hh>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace aqllog = hip::aqllog;

constexpr char kMaskEnv[] = "HIP_DISPATCH_CU_MASK";
constexpr char kResultToken[] = "CUMASK_RESULT";
constexpr char kChildTest[] = "Unit_hipDispatchCUMaskEnv_ChildWorkload";

constexpr uint32_t kNumSE = 2;
constexpr uint32_t kThreads = 1024 * 1024;
constexpr uint32_t kBlockSize = 256;
constexpr uint16_t kNotWritten = 0xFFFF;

// Every thread bumps an atomic counter and records its WGP, shader array and SE ids.
__global__ void recordPlacement(unsigned long long* counter, uint16_t* placement, uint32_t n) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  atomicAdd(counter, 1ULL);
  placement[tid] = static_cast<uint16_t>(__smid() & 0x1FF);
}

struct Placement {
  uint32_t wgps[kNumSE] = {};     //!< Used WGPs per SE, bit = shader array * 16 + WGP id
  uint32_t threads[kNumSE] = {};  //!< Threads that ran on each SE
  uint32_t strayThreads = 0;      //!< Threads never written or on an unknown SE
  unsigned long long count = 0;   //!< Atomic counter value
  std::string log;                //!< Full child output, including the AQL trace

  bool usesSE(uint32_t se) const { return wgps[se] != 0 && threads[se] != 0; }
  bool avoidsSE(uint32_t se) const { return wgps[se] == 0 && threads[se] == 0; }
};

bool isSubset(uint32_t subset, uint32_t superset) { return (subset & ~superset) == 0; }

int numWgps(uint32_t wgps) { return __builtin_popcount(wgps); }

bool isGfx1250() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  return strncmp(prop.gcnArchName, "gfx1250", 7) == 0;
}

// Runs the workload in a child with HIP_DISPATCH_CU_MASK=|mask| and parses what it observed.
// An empty mask leaves the feature off.
Placement runWithMask(const std::string& mask) {
  Placement result;
  result.log = aqllog::RunChildTest(kChildTest, {{kMaskEnv, mask}});

  const size_t pos = result.log.find(std::string{kResultToken} + " ");
  INFO("HIP_DISPATCH_CU_MASK='" << mask << "'");
  REQUIRE(pos != std::string::npos);
  const int parsed = sscanf(result.log.c_str() + pos,
                            "CUMASK_RESULT se0=0x%x se1=0x%x thr0=%u thr1=%u stray=%u count=%llu",
                            &result.wgps[0], &result.wgps[1], &result.threads[0],
                            &result.threads[1], &result.strayThreads, &result.count);
  REQUIRE(parsed == 6);

  // Masking must never lose or duplicate work.
  REQUIRE(result.count == kThreads);
  REQUIRE(result.strayThreads == 0);
  return result;
}

// True when the child's AQL trace holds a launch descriptor carrying |cuEnable|.
bool traceHasCUEnable(const std::string& log, uint32_t cuEnable) {
  char token[32];
  snprintf(token, sizeof(token), "cu_enable=0x%x,", cuEnable);
  return log.find(token) != std::string::npos;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Internal child workload (only runs when driven by a parent case).
 *    Launches one kernel under the inherited HIP_DISPATCH_CU_MASK and prints which SEs and
 *    WGPs its threads ran on.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipDispatchCUMaskEnv.cc
 */
HIP_TEST_CASE(Unit_hipDispatchCUMaskEnv_ChildWorkload) {
  if (!aqllog::RunningAsChild()) {
    SUCCEED("Skipping child workload outside of parent-driven run");
    return;
  }

  unsigned long long* d_counter = nullptr;
  uint16_t* d_placement = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(*d_counter)));
  HIP_CHECK(hipMalloc(&d_placement, kThreads * sizeof(uint16_t)));
  HIP_CHECK(hipMemset(d_counter, 0, sizeof(*d_counter)));
  HIP_CHECK(hipMemset(d_placement, 0xFF, kThreads * sizeof(uint16_t)));

  hipLaunchKernelGGL(recordPlacement, dim3(kThreads / kBlockSize), dim3(kBlockSize), 0, nullptr,
                     d_counter, d_placement, kThreads);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  Placement result;
  std::vector<uint16_t> placement(kThreads);
  HIP_CHECK(hipMemcpy(&result.count, d_counter, sizeof(result.count), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(placement.data(), d_placement, kThreads * sizeof(uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipFree(d_placement));

  for (const uint16_t p : placement) {
    const uint32_t wgp = p & 0xF;
    const uint32_t sa = (p >> 4) & 0x1;
    const uint32_t se = (p >> 5) & 0xF;
    if (p == kNotWritten || se >= kNumSE) {
      ++result.strayThreads;
      continue;
    }
    result.wgps[se] |= 1u << (sa * 16 + wgp);
    ++result.threads[se];
  }

  std::fflush(nullptr);
  fprintf(stderr, "\n%s se0=0x%x se1=0x%x thr0=%u thr1=%u stray=%u count=%llu\n", kResultToken,
          result.wgps[0], result.wgps[1], result.threads[0], result.threads[1],
          result.strayThreads, result.count);
  std::fflush(stderr);
}

/**
 * Test Description
 * ------------------------
 *  - SE_EN restricts every thread of the dispatch to the enabled Shader Engines:
 *    no mask and SE_EN=0x3 use both SEs, SE_EN=0x1 only SE0, SE_EN=0x2 only SE1.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipDispatchCUMaskEnv.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend, gfx1250
 */
HIP_TEST_CASE(Unit_hipDispatchCUMaskEnv_SEIsolation) {
  if (aqllog::RunningAsChild()) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }
  if (!isGfx1250()) {
    HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const Placement unmasked = runWithMask("");
  CHECK(unmasked.usesSE(0));
  CHECK(unmasked.usesSE(1));

  const Placement se0 = runWithMask("0x100");
  CHECK(se0.usesSE(0));
  CHECK(se0.avoidsSE(1));
  CHECK(se0.wgps[0] == unmasked.wgps[0]);

  const Placement se1 = runWithMask("0x200");
  CHECK(se1.avoidsSE(0));
  CHECK(se1.usesSE(1));
  CHECK(se1.wgps[1] == unmasked.wgps[1]);

  const Placement both = runWithMask("0x300");
  CHECK(both.usesSE(0));
  CHECK(both.usesSE(1));
}

/**
 * Test Description
 * ------------------------
 *  - START and CNT narrow the dispatch to a range of CUs inside the enabled SE: a smaller CNT
 *    uses a strict subset of the WGPs of a larger one, and a non-zero START moves the range
 *    away from the first WGP.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipDispatchCUMaskEnv.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend, gfx1250
 */
HIP_TEST_CASE(Unit_hipDispatchCUMaskEnv_CURange) {
  if (aqllog::RunningAsChild()) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }
  if (!isGfx1250()) {
    HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const Placement allCUs = runWithMask("0x100");   // SE0, all CUs
  const Placement twoCUs = runWithMask("0x120");   // SE0, START=0, CNT=2
  const Placement fourCUs = runWithMask("0x140");  // SE0, START=0, CNT=4
  const Placement offset = runWithMask("0x142");   // SE0, START=2, CNT=4

  for (const Placement* p : {&twoCUs, &fourCUs, &offset}) {
    CHECK(p->usesSE(0));
    CHECK(p->avoidsSE(1));
    CHECK(isSubset(p->wgps[0], allCUs.wgps[0]));
    CHECK(numWgps(p->wgps[0]) < numWgps(allCUs.wgps[0]));
  }

  // Growing CNT from the same START only adds WGPs.
  CHECK(isSubset(twoCUs.wgps[0], fourCUs.wgps[0]));
  CHECK(numWgps(twoCUs.wgps[0]) < numWgps(fourCUs.wgps[0]));

  // START=0 ranges include the first WGP of shader array 0; START=2 does not.
  CHECK((twoCUs.wgps[0] & 0x1) != 0);
  CHECK((fourCUs.wgps[0] & 0x1) != 0);
  CHECK((offset.wgps[0] & 0x1) == 0);
  CHECK(offset.wgps[0] != fourCUs.wgps[0]);
}

/**
 * Test Description
 * ------------------------
 *  - The mask reaches the hardware through the launch descriptor: the AQL trace of a masked
 *    run shows the flag value in cu_enable, and an unmasked run does not.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipDispatchCUMaskEnv.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend, gfx1250
 */
HIP_TEST_CASE(Unit_hipDispatchCUMaskEnv_LaunchDescriptor) {
  if (aqllog::RunningAsChild()) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }
  if (!isGfx1250()) {
    HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  CHECK(traceHasCUEnable(runWithMask("0x120").log, 0x120));
  CHECK(traceHasCUEnable(runWithMask("0x231").log, 0x231));
  CHECK_FALSE(traceHasCUEnable(runWithMask("").log, 0x120));
}

/**
 * Test Description
 * ------------------------
 *  - Values that are not a valid cu_enable DWORD (reserved bits set, SE_EN == 0, not hex) are
 *    ignored: the dispatch uses both SEs and the same WGPs as an unmasked run.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipDispatchCUMaskEnv.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend, gfx1250
 */
HIP_TEST_CASE(Unit_hipDispatchCUMaskEnv_InvalidValueIgnored) {
  if (aqllog::RunningAsChild()) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }
  if (!isGfx1250()) {
    HIP_SKIP_TEST("Per-dispatch CU mask requires gfx1250");
    return;
  }

  const Placement unmasked = runWithMask("");
  for (const char* mask : {"0x400", "0x20", "zzz"}) {
    const Placement ignored = runWithMask(mask);
    INFO("HIP_DISPATCH_CU_MASK='" << mask << "'");
    CHECK(ignored.wgps[0] == unmasked.wgps[0]);
    CHECK(ignored.wgps[1] == unmasked.wgps[1]);
  }
}

#endif  // HT_AMD && HT_LINUX

/**
 * End doxygen group StreamTest.
 * @}
 */
