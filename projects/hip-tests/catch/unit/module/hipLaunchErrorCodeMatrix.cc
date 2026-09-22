/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipLaunchErrorCodeMatrix hipLaunchErrorCodeMatrix
 * @{
 * @ingroup ModuleTest
 *
 * CHARACTERIZATION / GOLDEN-MASTER TEST -- read before touching this file.
 *
 * This test pins the `hipError_t` returned by every HIP kernel-launch entry point for a fixed
 * matrix of invalid (and a few intentionally-simultaneous-invalid) launch configurations. The
 * expected values below are OBSERVED behaviour of the current runtime (see
 * /tmp/launch_matrix/results.csv, gathered by driving each cell in its own subprocess), NOT
 * "desired"/spec behaviour. Some cells encode known defects (marked `GOLDEN-MASTER` below, with a
 * pointer into clr/consolidate_grid_size_plan.md). Do NOT "fix" a value here to match what you
 * think the API *should* return -- if this file starts failing during a refactor, that means the
 * refactor changed the public API contract, which is exactly what this test exists to catch.
 *
 * Case legend (matches /tmp/launch_matrix/harness.cc):
 *   c1  gridDim.x == 0
 *   c2  gridDim.y == 0
 *   c3  gridDim.z == 0
 *   c4  blockDim.x == 0
 *   c5  blockDim.y == 0
 *   c6  blockDim.z == 0
 *   c7  blockDim.x == 65536
 *   c8  blockDim.x == 66560
 *   c9  gridDim.x * blockDim.x > 2^32
 *   c10 blockDim.x * blockDim.y * blockDim.z > maxThreadsPerBlock
 *   c11 sharedMemBytes > maxSharedMemoryPerBlock
 *   c12 sharedMemBytes == 2^32 (only meaningful where the parameter is `size_t`)
 *   c13 clusterDim does not divide gridDim
 *   c14 clusterDim.x == 256 (uint8 overflow)
 *   c15 gridDim.x == maxGridDimX + 1
 *   c16 blockDim.x == 0 AND sharedMemBytes oversized (simultaneous failure - pins check order)
 *   c17 blockDim product > max AND sharedMemBytes oversized (simultaneous failure)
 *   c18 gridDim.x == 0 AND blockDim.x == 0 (simultaneous failure)
 *
 * Widening beyond the source CSV: c12 is marked N/A in the CSV for `hipLaunchKernel`,
 * `hipExtLaunchKernel`, and the `<<<>>>` triple-chevron launch, because the harness that produced
 * the CSV didn't drive it there. All three of those APIs take `size_t sharedMemBytes` (unlike, say,
 * `hipModuleLaunchKernel`'s `unsigned int`), so the case is meaningful and was newly driven here.
 * This is the single most valuable cell in the matrix: it is the one that would catch a
 * size_t -> uint32_t truncation bug silently swallowing the top 32 bits of sharedMemBytes.
 *
 * The widening found exactly that bug. All three newly-driven cells observed hipSuccess, not an
 * error: sharedMemBytes == 2^32 truncates to 0 on all three paths and the launch silently
 * "succeeds" with zero shared memory instead of being rejected -- the same class of defect already
 * known for hipHccModuleLaunchKernel's c12 (see below). Kept as observed and annotated
 * GOLDEN-MASTER at each of the three cells.
 *
 * OpenCL-style entry points (`hipExtModuleLaunchKernel`, `hipHccModuleLaunchKernel`) take
 * globalWorkSize (gridDim * blockDim), not gridDim directly, so "the same case id" is not always a
 * bit-for-bit-identical input across every row -- it is still a valid, meaningful per-cell
 * regression pin for that row, using the same CaseSpec construction as the module-launch rows.
 *
 * Case 15 (gridDim.x > maxGridDimX) safety note:
 * -----------------------------------------------
 * The characterization harness had to run every cell in a SEPARATE SUBPROCESS because, on the
 * entry points that launched the harness's own memory-touching kernel (`hostInc`, which does
 * `atomicAdd` through a pointer), c15 triggered an asynchronous HSA_STATUS_ERROR_MEMORY_FAULT that
 * poisoned the HSA queue -- every subsequent cell run in that process then returned
 * hipErrorIllegalAddress regardless of input. A Catch2 binary runs every SECTION in one process,
 * so that hazard would corrupt every entry point's results after the first poisoned one.
 *
 * This test avoids the hazard by construction (choice (a) from the task, verified empirically):
 * every kernel driven here -- `NOPKernel` (loaded from the module fixture) and the file-local
 * `GoldenNoOpKernel` (used for the compile-time launch APIs) -- touches no memory at all and takes
 * no arguments. There is no pointer for a bad launch configuration to dereference, so there is
 * nothing for an out-of-range grid to fault against; c15 is driven normally, in-line, for every
 * entry point. This was verified by running the full `Unit_hipLaunchErrorCodeMatrix_Golden` test
 * case three consecutive times in one process and confirming identical, all-green results each
 * time (see the PR description / commit message for the run transcript).
 *
 * Known-defect cells (kept as observed, not "corrected" -- see clr/consolidate_grid_size_plan.md):
 *   - hipHccModuleLaunchKernel   c8  -> hipSuccess   (plan Sec6.1: blockDim.x=66560 narrows to
 *                                                     1024 in NDRange16 and silently launches with
 *                                                     the wrong dimensions instead of failing)
 *   - hipHccModuleLaunchKernel   c12 -> hipSuccess   (sharedMemBytes==2^32 truncates to 0 in the
 *                                                     uint32_t narrowing on this path)
 *   - hipDrvLaunchKernelEx(clusterDim attr) c13/c14 -> hipSuccess (plan Sec6.2: the cluster
 *                                                     construction path never consults
 *                                                     IsValidConfig(), so an indivisible/overflowed
 *                                                     clusterDim is never rejected)
 *   - hipGraphAddKernelNode c15 -> hipSuccess (matches the CSV and plan Sec6.3a)
 *
 * DISAGREEMENT WITH THE SOURCE CSV (investigated, not papered over): the CSV records
 * hipGraphKernelNodeSetParams and hipGraphExecKernelNodeSetParams c15 as hipErrorInvalidValue,
 * asymmetric with hipGraphAddKernelNode's hipSuccess (plan Sec6.3a). Reproducing that exact cell
 * against this runtime -- with a valid (1,1,1) node re-parented via SetParams / ExecSetParams to
 * gridDim.x == maxGridDimX + 1, and independently via a standalone (non-Catch2) probe program --
 * both return hipSuccess here, i.e. no asymmetry: all three Add/Set/ExecSet APIs currently accept
 * the oversized grid. Encoded as observed (hipSuccess) below; see the commit message / task report
 * for the reproduction. This does not contradict plan Sec6.3a's root-cause analysis (nothing in
 * IsValidConfig() consults gridDim.x against maxGridDimX on any of these three paths) -- it means
 * the CSV's SetParams/ExecSetParams cells no longer reproduce on this runtime build.
 *
 * Test source
 * ------------------------
 *  - unit/module/hipLaunchErrorCodeMatrix.cc
 */

#include <hip_test_common.hh>
#include <hip/hip_ext.h>
#include <resource_guards.hh>
#include <utils.hh>

#include "hip_module_launch_kernel_common.hh"

namespace {

// Sentinel for "not applicable" cells -- never equal to a real hipError_t (hipError_t's
// underlying range is [0, 2047]; no defined error code uses this value).
constexpr hipError_t kNA = static_cast<hipError_t>(2046);

struct CaseSpec {
  int id;
  unsigned gx, gy, gz;
  unsigned bx, by, bz;
  unsigned long long shared;  // may exceed 32 bits (c12)
  bool useCluster;
  unsigned cx, cy, cz;
};

std::vector<CaseSpec> BuildCases(unsigned maxThreadsPerBlock, unsigned maxSharedMemPerBlock,
                                  unsigned maxGridDimX) {
  std::vector<CaseSpec> cases;
  cases.push_back({1, 0, 1, 1, 1, 1, 1, 0, false, 0, 0, 0});
  cases.push_back({2, 1, 0, 1, 1, 1, 1, 0, false, 0, 0, 0});
  cases.push_back({3, 1, 1, 0, 1, 1, 1, 0, false, 0, 0, 0});
  cases.push_back({4, 1, 1, 1, 0, 1, 1, 0, false, 0, 0, 0});
  cases.push_back({5, 1, 1, 1, 1, 0, 1, 0, false, 0, 0, 0});
  cases.push_back({6, 1, 1, 1, 1, 1, 0, 0, false, 0, 0, 0});
  cases.push_back({7, 1, 1, 1, 65536, 1, 1, 0, false, 0, 0, 0});
  cases.push_back({8, 1, 1, 1, 66560, 1, 1, 0, false, 0, 0, 0});
  cases.push_back({9, 2147483648u, 1, 1, 4, 1, 1, 0, false, 0, 0, 0});

  unsigned cube = 1;
  while (static_cast<unsigned long long>(cube) * cube * cube < maxThreadsPerBlock) cube++;
  cube += 1;  // ensure the product exceeds max
  cases.push_back({10, 1, 1, 1, cube, cube, cube, 0, false, 0, 0, 0});

  cases.push_back(
      {11, 1, 1, 1, 1, 1, 1, static_cast<unsigned long long>(maxSharedMemPerBlock) + 1, false, 0, 0, 0});
  cases.push_back({12, 1, 1, 1, 1, 1, 1, 4294967296ULL, false, 0, 0, 0});
  cases.push_back({13, 3, 1, 1, 1, 1, 1, 0, true, 2, 1, 1});
  cases.push_back({14, 256, 1, 1, 1, 1, 1, 0, true, 256, 1, 1});
  cases.push_back({15, maxGridDimX + 1u, 1, 1, 1, 1, 1, 0, false, 0, 0, 0});
  cases.push_back(
      {16, 1, 1, 1, 0, 1, 1, static_cast<unsigned long long>(maxSharedMemPerBlock) + 1, false, 0, 0, 0});
  cases.push_back(
      {17, 1, 1, 1, cube, cube, cube, static_cast<unsigned long long>(maxSharedMemPerBlock) + 1,
       false, 0, 0, 0});
  cases.push_back({18, 0, 1, 1, 0, 1, 1, 0, false, 0, 0, 0});
  return cases;
}

// Golden-master check: reports mismatches via CHECK (not REQUIRE) so one bad cell doesn't hide
// the rest of the matrix in a single test run.
void GoldenCheck(const char* entryPoint, int caseId, hipError_t actual, hipError_t expected) {
  INFO("entry point: " << entryPoint << "  case: c" << caseId
                        << "  expected: " << hipGetErrorName(expected)
                        << "  actual: " << hipGetErrorName(actual));
  CHECK(actual == expected);
}

// Zero-argument, memory-touch-free kernel for the compile-time launch entry points
// (hipLaunchKernel, hipExtLaunchKernel, hipLaunchCooperativeKernel, <<<>>>) and for the graph
// kernel-node entry points, which take a compiled function pointer rather than a module
// hipFunction_t. Deliberately distinct from the module fixture's NOPKernel (different translation
// unit / compiled kernel object) but semantically identical: it does nothing and touches nothing,
// so it cannot fault regardless of how badly the launch configuration is mis-shapen. See the c15
// safety note above.
__global__ void GoldenNoOpKernel() {}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Golden-master / characterization test pinning the hipError_t returned by every HIP
 *    kernel-launch entry point (module launch, driver launch, runtime launch, graph kernel-node
 *    APIs, cooperative and multi-device launch, and the <<<>>> syntax) for 18 invalid or
 *    simultaneously-invalid launch-configuration cases. See the file header for full context.
 * Test source
 * ------------------------
 *  - unit/module/hipLaunchErrorCodeMatrix.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
HIP_TEST_CASE(Unit_hipLaunchErrorCodeMatrix_Golden) {
  const unsigned maxThreadsPerBlock =
      static_cast<unsigned>(GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0));
  const unsigned maxSharedMemPerBlock =
      static_cast<unsigned>(GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  const unsigned maxGridDimX =
      static_cast<unsigned>(GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0));
  const std::vector<CaseSpec> cases = BuildCases(maxThreadsPerBlock, maxSharedMemPerBlock,
                                                  maxGridDimX);

  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  hipFunction_t f_noop = GetKernel(mg.module(), "NOPKernel");

  SECTION("hipModuleLaunchKernel") {
    // c12/c13/c14 N/A: sharedMemBytes is `unsigned int` here (no size_t widening), and this API
    // has no cluster-dimension concept.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidValue},  {2, hipErrorInvalidValue},  {3, hipErrorInvalidValue},
        {4, hipErrorInvalidValue},  {5, hipErrorInvalidValue},  {6, hipErrorInvalidValue},
        {7, hipErrorInvalidValue},  {8, hipErrorInvalidValue},  {9, hipErrorInvalidValue},
        {10, hipErrorInvalidValue}, {11, hipErrorInvalidValue}, {12, kNA},
        {13, kNA},                  {14, kNA},                  {15, hipSuccess},
        {16, hipErrorInvalidValue}, {17, hipErrorInvalidValue}, {18, hipErrorInvalidValue},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      hipError_t e = hipModuleLaunchKernel(f_noop, c.gx, c.gy, c.gz, c.bx, c.by, c.bz,
                                            static_cast<unsigned>(c.shared), nullptr, nullptr,
                                            nullptr);
      GoldenCheck("hipModuleLaunchKernel", c.id, e, exp);
    }
  }

  SECTION("hipExtModuleLaunchKernel") {
    // c13/c14 N/A: no cluster-dimension concept on this API.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidValue},         {2, hipErrorInvalidValue},
        {3, hipErrorInvalidValue},         {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidValue},         {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, hipErrorInvalidValue},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidConfiguration}, {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      unsigned long long gwx = static_cast<unsigned long long>(c.gx) * c.bx;
      unsigned long long gwy = static_cast<unsigned long long>(c.gy) * c.by;
      unsigned long long gwz = static_cast<unsigned long long>(c.gz) * c.bz;
      hipError_t e = hipExtModuleLaunchKernel(
          f_noop, static_cast<uint32_t>(gwx), static_cast<uint32_t>(gwy),
          static_cast<uint32_t>(gwz), c.bx, c.by, c.bz, static_cast<size_t>(c.shared), nullptr,
          nullptr, nullptr);
      GoldenCheck("hipExtModuleLaunchKernel", c.id, e, exp);
    }
  }

  SECTION("hipHccModuleLaunchKernel") {
    // c13/c14 N/A: no cluster-dimension concept on this API.
    // GOLDEN-MASTER (clr/consolidate_grid_size_plan.md Sec6.1): c8 -> hipSuccess. blockDim.x=66560
    // narrows to 1024 in NDRange16 and silently launches with the wrong dimensions instead of
    // being rejected.
    // GOLDEN-MASTER: c12 -> hipSuccess. sharedMemBytes==2^32 truncates to 0 on this path (this API
    // never consults IsValidConfig()/validConfig_ at all -- see plan Sec6.1).
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipSuccess},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, hipSuccess},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      unsigned long long gwx = static_cast<unsigned long long>(c.gx) * c.bx;
      unsigned long long gwy = static_cast<unsigned long long>(c.gy) * c.by;
      unsigned long long gwz = static_cast<unsigned long long>(c.gz) * c.bz;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      hipError_t e = hipHccModuleLaunchKernel(
          f_noop, static_cast<uint32_t>(gwx), static_cast<uint32_t>(gwy),
          static_cast<uint32_t>(gwz), c.bx, c.by, c.bz, static_cast<size_t>(c.shared), nullptr,
          nullptr, nullptr, nullptr, nullptr);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
      GoldenCheck("hipHccModuleLaunchKernel", c.id, e, exp);
    }
  }

  SECTION("hipModuleLaunchCooperativeKernel") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
    // c12/c13/c14 N/A: sharedMemBytes is `unsigned int`; no cluster-dimension concept.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidValue},  {2, hipErrorInvalidValue},
        {3, hipErrorInvalidValue},  {4, hipErrorInvalidValue},
        {5, hipErrorInvalidValue},  {6, hipErrorInvalidValue},
        {7, hipErrorInvalidValue},  {8, hipErrorInvalidValue},
        {9, hipErrorInvalidValue},  {10, hipErrorInvalidValue},
        {11, hipErrorInvalidValue}, {12, kNA},
        {13, kNA},                  {14, kNA},
        {15, hipErrorCooperativeLaunchTooLarge}, {16, hipErrorInvalidValue},
        {17, hipErrorInvalidValue}, {18, hipErrorInvalidValue},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      hipError_t e = hipModuleLaunchCooperativeKernel(f_noop, c.gx, c.gy, c.gz, c.bx, c.by, c.bz,
                                                        static_cast<unsigned>(c.shared), nullptr,
                                                        nullptr);
      GoldenCheck("hipModuleLaunchCooperativeKernel", c.id, e, exp);
    }
  }

  SECTION("hipLaunchCooperativeKernel") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
    // c12/c13/c14 N/A: sharedMemBytes is `unsigned int`; no cluster-dimension concept.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration},  {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration},  {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration},  {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration},  {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration},  {10, hipErrorInvalidConfiguration},
        {11, hipErrorCooperativeLaunchTooLarge}, {12, kNA},
        {13, kNA},                           {14, kNA},
        {15, hipErrorCooperativeLaunchTooLarge}, {16, hipErrorCooperativeLaunchTooLarge},
        {17, hipErrorInvalidConfiguration},  {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      void* args[] = {};
      hipError_t e = hipLaunchCooperativeKernel((const void*)GoldenNoOpKernel,
                                                 dim3(c.gx, c.gy, c.gz), dim3(c.bx, c.by, c.bz),
                                                 args, static_cast<unsigned>(c.shared), nullptr);
      GoldenCheck("hipLaunchCooperativeKernel", c.id, e, exp);
    }
  }

  SECTION("hipDrvLaunchKernelEx(no attrs)") {
    // c12 N/A: not widened per task scope (only hipLaunchKernel / hipExtLaunchKernel / <<<>>> are
    // widened). c13/c14 N/A: no cluster attribute attached in this row.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, kNA},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      HIP_LAUNCH_CONFIG cfg{};
      cfg.gridDimX = c.gx;
      cfg.gridDimY = c.gy;
      cfg.gridDimZ = c.gz;
      cfg.blockDimX = c.bx;
      cfg.blockDimY = c.by;
      cfg.blockDimZ = c.bz;
      cfg.sharedMemBytes = static_cast<unsigned>(c.shared);
      cfg.hStream = nullptr;
      cfg.attrs = nullptr;
      cfg.numAttrs = 0;
      void* params[] = {};
      hipError_t e = hipDrvLaunchKernelEx(&cfg, f_noop, params, nullptr);
      GoldenCheck("hipDrvLaunchKernelEx(no attrs)", c.id, e, exp);
    }
  }

  SECTION("hipDrvLaunchKernelEx(clusterDim attr)") {
    // c12 N/A: not widened per task scope.
    // GOLDEN-MASTER (clr/consolidate_grid_size_plan.md Sec6.2): c13/c14 -> hipSuccess. The cluster
    // construction path (`launch_params_cluster`, hip_module.cpp:1492) never consults
    // IsValidConfig(), so an indivisible clusterDim (c13) or an overflowed clusterDim.x==256 (c14)
    // is never rejected.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, kNA},
        {13, hipSuccess},                  {14, hipSuccess},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      hipLaunchAttribute attr{};
      attr.id = hipLaunchAttributeClusterDimension;
      if (c.useCluster) {
        attr.value.clusterDim.x = c.cx;
        attr.value.clusterDim.y = c.cy;
        attr.value.clusterDim.z = c.cz;
      } else {
        attr.value.clusterDim.x = 1;
        attr.value.clusterDim.y = 1;
        attr.value.clusterDim.z = 1;
      }
      HIP_LAUNCH_CONFIG cfg{};
      cfg.gridDimX = c.gx;
      cfg.gridDimY = c.gy;
      cfg.gridDimZ = c.gz;
      cfg.blockDimX = c.bx;
      cfg.blockDimY = c.by;
      cfg.blockDimZ = c.bz;
      cfg.sharedMemBytes = static_cast<unsigned>(c.shared);
      cfg.hStream = nullptr;
      cfg.attrs = &attr;
      cfg.numAttrs = 1;
      void* params[] = {};
      hipError_t e = hipDrvLaunchKernelEx(&cfg, f_noop, params, nullptr);
      GoldenCheck("hipDrvLaunchKernelEx(clusterDim attr)", c.id, e, exp);
    }
  }

  SECTION("hipLaunchKernel") {
    // c13/c14 N/A: no cluster-dimension concept on this API.
    // c12 WIDENED (required by task scope): sharedMemBytes is `size_t` here, so the 2^32 case is
    // meaningful and is newly driven (the source CSV marked it N/A because the harness that
    // produced it didn't drive this cell).
    // GOLDEN-MASTER (newly discovered by this widening): c12 -> hipSuccess. sharedMemBytes ==
    // 2^32 truncates to 0 on this path and the launch silently "succeeds" with zero shared memory
    // instead of being rejected -- the same class of defect as hipHccModuleLaunchKernel's c12.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, hipSuccess},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      void* args[] = {};
      hipError_t launchErr = hipLaunchKernel((const void*)GoldenNoOpKernel,
                                              dim3(c.gx, c.gy, c.gz), dim3(c.bx, c.by, c.bz), args,
                                              static_cast<size_t>(c.shared), nullptr);
      (void)launchErr;
      hipError_t e = hipGetLastError();
      GoldenCheck("hipLaunchKernel", c.id, e, exp);
    }
  }

  SECTION("hipExtLaunchKernel") {
    // c13/c14 N/A: no cluster-dimension concept on this API.
    // c12 WIDENED (required by task scope): sharedMemBytes is `size_t` here.
    // GOLDEN-MASTER (newly discovered by this widening): c12 -> hipSuccess. Same
    // sharedMemBytes==2^32 -> truncates-to-0 defect as hipLaunchKernel above.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, hipSuccess},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      void* args[] = {};
      hipError_t e = hipExtLaunchKernel((const void*)GoldenNoOpKernel, dim3(c.gx, c.gy, c.gz),
                                         dim3(c.bx, c.by, c.bz), args,
                                         static_cast<size_t>(c.shared), nullptr, nullptr, nullptr,
                                         0);
      GoldenCheck("hipExtLaunchKernel", c.id, e, exp);
    }
  }

  SECTION("hipGraphAddKernelNode") {
    // c12/c13/c14 N/A: hipKernelNodeParams::sharedMemBytes is `unsigned int`; no
    // cluster-dimension concept.
    // GOLDEN-MASTER (clr/consolidate_grid_size_plan.md Sec6.3a add-vs-set asymmetry): c15 ->
    // hipSuccess here, but the identical params fed through hipGraphKernelNodeSetParams /
    // hipGraphExecKernelNodeSetParams below return hipErrorInvalidValue.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, kNA},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      hipGraph_t graph;
      HIP_CHECK(hipGraphCreate(&graph, 0));
      hipKernelNodeParams p{};
      p.func = reinterpret_cast<void*>(GoldenNoOpKernel);
      p.gridDim = dim3(c.gx, c.gy, c.gz);
      p.blockDim = dim3(c.bx, c.by, c.bz);
      p.sharedMemBytes = static_cast<unsigned>(c.shared);
      p.kernelParams = nullptr;
      p.extra = nullptr;
      hipGraphNode_t node;
      hipError_t e = hipGraphAddKernelNode(&node, graph, nullptr, 0, &p);
      GoldenCheck("hipGraphAddKernelNode", c.id, e, exp);
      HIP_CHECK(hipGraphDestroy(graph));
    }
  }

  SECTION("hipGraphKernelNodeSetParams") {
    // c12/c13/c14 N/A: see hipGraphAddKernelNode above.
    // DISAGREEMENT WITH THE SOURCE CSV (see file header): the CSV records c15 as
    // hipErrorInvalidValue here (an asymmetry vs. hipGraphAddKernelNode's hipSuccess, plan
    // Sec6.3a). Reproduced against this runtime -- both in this test and independently via a
    // standalone probe program -- c15 returns hipSuccess, matching hipGraphAddKernelNode. Encoded
    // as observed; investigated, not papered over.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, kNA},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      hipGraph_t graph;
      HIP_CHECK(hipGraphCreate(&graph, 0));
      hipKernelNodeParams pInit{};
      pInit.func = reinterpret_cast<void*>(GoldenNoOpKernel);
      pInit.gridDim = dim3(1, 1, 1);
      pInit.blockDim = dim3(1, 1, 1);
      pInit.sharedMemBytes = 0;
      pInit.kernelParams = nullptr;
      pInit.extra = nullptr;
      hipGraphNode_t node;
      HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &pInit));

      hipKernelNodeParams p{};
      p.func = reinterpret_cast<void*>(GoldenNoOpKernel);
      p.gridDim = dim3(c.gx, c.gy, c.gz);
      p.blockDim = dim3(c.bx, c.by, c.bz);
      p.sharedMemBytes = static_cast<unsigned>(c.shared);
      p.kernelParams = nullptr;
      p.extra = nullptr;
      hipError_t e = hipGraphKernelNodeSetParams(node, &p);
      GoldenCheck("hipGraphKernelNodeSetParams", c.id, e, exp);
      HIP_CHECK(hipGraphDestroy(graph));
    }
  }

  SECTION("hipGraphExecKernelNodeSetParams") {
    // c12/c13/c14 N/A: see hipGraphAddKernelNode above.
    // DISAGREEMENT WITH THE SOURCE CSV (see file header and hipGraphKernelNodeSetParams above):
    // the CSV records c15 as hipErrorInvalidValue; reproduced against this runtime it returns
    // hipSuccess. Encoded as observed.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, kNA},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      hipGraph_t graph;
      HIP_CHECK(hipGraphCreate(&graph, 0));
      hipKernelNodeParams pInit{};
      pInit.func = reinterpret_cast<void*>(GoldenNoOpKernel);
      pInit.gridDim = dim3(1, 1, 1);
      pInit.blockDim = dim3(1, 1, 1);
      pInit.sharedMemBytes = 0;
      pInit.kernelParams = nullptr;
      pInit.extra = nullptr;
      hipGraphNode_t node;
      HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &pInit));
      hipGraphExec_t exec;
      HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

      hipKernelNodeParams p{};
      p.func = reinterpret_cast<void*>(GoldenNoOpKernel);
      p.gridDim = dim3(c.gx, c.gy, c.gz);
      p.blockDim = dim3(c.bx, c.by, c.bz);
      p.sharedMemBytes = static_cast<unsigned>(c.shared);
      p.kernelParams = nullptr;
      p.extra = nullptr;
      hipError_t e = hipGraphExecKernelNodeSetParams(exec, node, &p);
      GoldenCheck("hipGraphExecKernelNodeSetParams", c.id, e, exp);
      HIP_CHECK(hipGraphExecDestroy(exec));
      HIP_CHECK(hipGraphDestroy(graph));
    }
  }

  SECTION("hipModuleLaunchCooperativeKernelMultiDevice") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
    // c12/c13/c14 N/A: hipFunctionLaunchParams::sharedMemBytes is `unsigned int`; no
    // cluster-dimension concept.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, kNA},
        {13, kNA},                         {14, kNA},
        {15, hipErrorCooperativeLaunchTooLarge}, {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };

    const int deviceCount = HipTest::getDeviceCount();
    std::vector<hipModule_t> modules(deviceCount);
    std::vector<hipStream_t> streams(deviceCount);
    std::vector<hipFunctionLaunchParams> params(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipModuleLoad(&modules[i], "launch_kernel_module.code"));
      HIP_CHECK(hipStreamCreate(&streams[i]));
    }
    HIP_CHECK(hipSetDevice(0));

    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      for (int i = 0; i < deviceCount; i++) {
        hipFunction_t fn;
        HIP_CHECK(hipModuleGetFunction(&fn, modules[i], "NOPKernel"));
        params[i].function = fn;
        params[i].gridDimX = c.gx;
        params[i].gridDimY = c.gy;
        params[i].gridDimZ = c.gz;
        params[i].blockDimX = c.bx;
        params[i].blockDimY = c.by;
        params[i].blockDimZ = c.bz;
        params[i].sharedMemBytes = static_cast<unsigned>(c.shared);
        params[i].kernelParams = nullptr;
        params[i].hStream = streams[i];
      }
      hipError_t e = hipModuleLaunchCooperativeKernelMultiDevice(params.data(), deviceCount, 0u);
      GoldenCheck("hipModuleLaunchCooperativeKernelMultiDevice", c.id, e, exp);
    }

    for (int i = 0; i < deviceCount; i++) {
      HIP_CHECK(hipStreamDestroy(streams[i]));
      HIP_CHECK(hipModuleUnload(modules[i]));
    }
    HIP_CHECK(hipSetDevice(0));
  }

  SECTION("<<<>>> (hipLaunchByPtr)") {
    // c13/c14 N/A: no cluster-dimension concept via this syntax.
    // c12 WIDENED (required by task scope): the compiler lowers <<<>>> to a call taking
    // `size_t sharedMemBytes`, so the 2^32 case is meaningful and is newly driven.
    // GOLDEN-MASTER (newly discovered by this widening): c12 -> hipSuccess. Same
    // sharedMemBytes==2^32 -> truncates-to-0 defect as hipLaunchKernel/hipExtLaunchKernel above.
    const std::map<int, hipError_t> expected = {
        {1, hipErrorInvalidConfiguration}, {2, hipErrorInvalidConfiguration},
        {3, hipErrorInvalidConfiguration}, {4, hipErrorInvalidConfiguration},
        {5, hipErrorInvalidConfiguration}, {6, hipErrorInvalidConfiguration},
        {7, hipErrorInvalidConfiguration}, {8, hipErrorInvalidConfiguration},
        {9, hipErrorInvalidConfiguration}, {10, hipErrorInvalidConfiguration},
        {11, hipErrorInvalidValue},        {12, hipSuccess},
        {13, kNA},                         {14, kNA},
        {15, hipSuccess},                  {16, hipErrorInvalidConfiguration},
        {17, hipErrorInvalidValue},        {18, hipErrorInvalidConfiguration},
    };
    for (const auto& c : cases) {
      hipError_t exp = expected.at(c.id);
      if (exp == kNA) continue;
      GoldenNoOpKernel<<<dim3(c.gx, c.gy, c.gz), dim3(c.bx, c.by, c.bz),
                          static_cast<size_t>(c.shared), 0>>>();
      hipError_t e = hipGetLastError();
      GoldenCheck("<<<>>> (hipLaunchByPtr)", c.id, e, exp);
    }
  }
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
