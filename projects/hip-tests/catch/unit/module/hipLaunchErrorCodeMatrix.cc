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
 * Pins the current HIP error code behaviour, this is not the ideal behaviour. The test is to
 * identify API breaks and allow this to be made consistent in the future.
 *
 * The globalWorkSize entry points (`hipExtModuleLaunchKernel`, `hipHccModuleLaunchKernel`) take
 * gridDim * blockDim rather than gridDim, so a case id is not a bit-identical input on every row.
 *
 * Every kernel driven here takes no arguments and touches no memory, so an out-of-range grid (c15)
 * has no pointer to fault against. That keeps all sections safe in one process: a memory-touching
 * kernel would raise an async memory fault on c15 and poison the HSA queue for every later cell.
 *
 * Cells recording a known defect -- hipSuccess where an error is expected -- kept as observed, see
 * clr/consolidate_grid_size_plan.md:
 *   - kHccModule c8: blockDim.x=66560 narrows to 1024 and launches mis-shapen (Sec6.1)
 *   - kHccModule / kLaunch / kExtLaunch / kByPtr c12: sharedMemBytes 2^32 truncates to 0
 *   - kDrvExCluster c13/c14: the cluster path never consults IsValidConfig() (Sec6.2)
 *   - kGraphAdd / kGraphSet / kGraphExecSet c15: nothing checks gridDim.x vs maxGridDimX (Sec6.3a)
 */

#include <array>

#include <hip_test_common.hh>
#include <hip/hip_ext.h>
#include <resource_guards.hh>
#include <utils.hh>

#include "hip_module_launch_kernel_common.hh"

namespace {

constexpr int kCaseCount = 18;

// Short aliases so one entry point's whole row fits on a line. NA marks a case the entry point
// cannot express (sharedMemBytes is `unsigned int` there, or it has no cluster parameter); 2046 is
// outside hipError_t's [0, 2047] range, so it can never collide with a real code.
constexpr hipError_t OK = hipSuccess;
constexpr hipError_t IC = hipErrorInvalidConfiguration;
constexpr hipError_t IV = hipErrorInvalidValue;
constexpr hipError_t TL = hipErrorCooperativeLaunchTooLarge;
constexpr hipError_t NA = static_cast<hipError_t>(2046);

// Indexed by case id - 1. Fixed size, so a new case forces every row to be revisited to compile.
using Row = std::array<hipError_t, kCaseCount>;

// Exempt from clang-format: the column alignment is the point, and pinning it keeps a re-recorded
// cell to a one-token diff.
// clang-format off
//                                c1  c2  c3  c4  c5  c6  c7  c8  c9 c10 c11 c12 c13 c14 c15 c16 c17 c18
constexpr Row kModule        = {IV, IV, IV, IV, IV, IV, IV, IV, IV, IV, IV, NA, NA, NA, OK, IV, IV, IV};
constexpr Row kExtModule     = {IV, IV, IV, IC, IC, IC, IC, IC, IV, IC, IV, IV, NA, NA, OK, IC, IC, IC};
constexpr Row kHccModule     = {IC, IC, IC, IC, IC, IC, IC, OK, IC, IC, IV, OK, NA, NA, OK, IC, IV, IC};
constexpr Row kModuleCoop    = {IV, IV, IV, IV, IV, IV, IV, IV, IV, IV, IV, NA, NA, NA, TL, IV, IV, IV};
constexpr Row kCoop          = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, TL, NA, NA, NA, TL, TL, IC, IC};
constexpr Row kDrvEx         = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, NA, NA, NA, OK, IC, IV, IC};
constexpr Row kDrvExCluster  = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, NA, OK, OK, OK, IC, IV, IC};
constexpr Row kLaunch        = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, OK, NA, NA, OK, IC, IV, IC};
constexpr Row kExtLaunch     = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, OK, NA, NA, OK, IC, IV, IC};
constexpr Row kGraphAdd      = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, NA, NA, NA, OK, IC, IV, IC};
constexpr Row kGraphSet      = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, NA, NA, NA, OK, IC, IV, IC};
constexpr Row kGraphExecSet  = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, NA, NA, NA, OK, IC, IV, IC};
constexpr Row kCoopMultiDev  = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, NA, NA, NA, TL, IC, IV, IC};
constexpr Row kByPtr         = {IC, IC, IC, IC, IC, IC, IC, IC, IC, IC, IV, OK, NA, NA, OK, IC, IV, IC};
// clang-format on

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
  const unsigned long long overMaxShared =
      static_cast<unsigned long long>(maxSharedMemPerBlock) + 1;

  unsigned cube = 1;
  while (static_cast<unsigned long long>(cube) * cube * cube < maxThreadsPerBlock) cube++;
  cube += 1;  // ensure the product exceeds max

  // {id, grid xyz, block xyz, sharedMemBytes, useCluster, cluster xyz}
  return {
      {1, 0, 1, 1, 1, 1, 1, 0, false, 0, 0, 0},
      {2, 1, 0, 1, 1, 1, 1, 0, false, 0, 0, 0},
      {3, 1, 1, 0, 1, 1, 1, 0, false, 0, 0, 0},
      {4, 1, 1, 1, 0, 1, 1, 0, false, 0, 0, 0},
      {5, 1, 1, 1, 1, 0, 1, 0, false, 0, 0, 0},
      {6, 1, 1, 1, 1, 1, 0, 0, false, 0, 0, 0},
      {7, 1, 1, 1, 65536, 1, 1, 0, false, 0, 0, 0},
      {8, 1, 1, 1, 66560, 1, 1, 0, false, 0, 0, 0},
      {9, 2147483648u, 1, 1, 4, 1, 1, 0, false, 0, 0, 0},  // grid.x * block.x > 2^32
      {10, 1, 1, 1, cube, cube, cube, 0, false, 0, 0, 0},  // block product > maxThreadsPerBlock
      {11, 1, 1, 1, 1, 1, 1, overMaxShared, false, 0, 0, 0},
      {12, 1, 1, 1, 1, 1, 1, 4294967296ULL, false, 0, 0, 0},
      {13, 3, 1, 1, 1, 1, 1, 0, true, 2, 1, 1},
      {14, 256, 1, 1, 1, 1, 1, 0, true, 256, 1, 1},
      {15, maxGridDimX + 1u, 1, 1, 1, 1, 1, 0, false, 0, 0, 0},
      // c16-c18 fail two ways at once, pinning the order the checks run in.
      {16, 1, 1, 1, 0, 1, 1, overMaxShared, false, 0, 0, 0},
      {17, 1, 1, 1, cube, cube, cube, overMaxShared, false, 0, 0, 0},
      {18, 0, 1, 1, 0, 1, 1, 0, false, 0, 0, 0},
  };
}

// CHECK, not REQUIRE, so one bad cell doesn't hide the rest of the matrix. Catch2 already prints
// the enclosing SECTION, so the entry point is not repeated here.
void GoldenCheck(int caseId, hipError_t actual, hipError_t expected) {
  INFO("case: c" << caseId << "  expected: " << hipGetErrorName(expected)
                 << "  actual: " << hipGetErrorName(actual));
  CHECK(actual == expected);
}

// The globalWorkSize entry points take grid * block rather than gridDim.
uint32_t GlobalWork(unsigned grid, unsigned block) {
  return static_cast<uint32_t>(static_cast<unsigned long long>(grid) * block);
}

// For the entry points taking a compiled function pointer rather than a module hipFunction_t.
// Argument-free and memory-free, so no launch configuration can make it fault.
__global__ void GoldenNoOpKernel() {}

HIP_LAUNCH_CONFIG MakeLaunchConfig(const CaseSpec& c, hipLaunchAttribute* attrs,
                                   unsigned numAttrs) {
  HIP_LAUNCH_CONFIG cfg{};
  cfg.gridDimX = c.gx;
  cfg.gridDimY = c.gy;
  cfg.gridDimZ = c.gz;
  cfg.blockDimX = c.bx;
  cfg.blockDimY = c.by;
  cfg.blockDimZ = c.bz;
  cfg.sharedMemBytes = static_cast<unsigned>(c.shared);
  cfg.hStream = nullptr;
  cfg.attrs = attrs;
  cfg.numAttrs = numAttrs;
  return cfg;
}

hipKernelNodeParams MakeNodeParams(const dim3& grid, const dim3& block, unsigned sharedMemBytes) {
  hipKernelNodeParams p{};
  p.func = reinterpret_cast<void*>(GoldenNoOpKernel);
  p.gridDim = grid;
  p.blockDim = block;
  p.sharedMemBytes = sharedMemBytes;
  p.kernelParams = nullptr;
  p.extra = nullptr;
  return p;
}

hipKernelNodeParams MakeNodeParams(const CaseSpec& c) {
  return MakeNodeParams(dim3(c.gx, c.gy, c.gz), dim3(c.bx, c.by, c.bz),
                        static_cast<unsigned>(c.shared));
}

// A graph holding one valid (1,1,1) no-op node, for the SetParams rows to re-parent.
struct GraphWithNode {
  GraphWithNode() {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    const hipKernelNodeParams valid = MakeNodeParams(dim3(1, 1, 1), dim3(1, 1, 1), 0);
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &valid));
  }
  ~GraphWithNode() { static_cast<void>(hipGraphDestroy(graph)); }

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
};

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Pins the hipError_t returned by every HIP kernel-launch entry point (module, driver, runtime,
 *    graph kernel-node, cooperative, multi-device, and <<<>>>) for 18 invalid or
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
  const std::vector<CaseSpec> cases =
      BuildCases(maxThreadsPerBlock, maxSharedMemPerBlock, maxGridDimX);

  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  hipFunction_t f_noop = GetKernel(mg.module(), "NOPKernel");

  // Drive every case this entry point can express. \a launch returns the code it reported for \a c.
  auto RunRow = [&cases](const Row& expected, auto&& launch) {
    for (const auto& c : cases) {
      const hipError_t exp = expected[c.id - 1];
      if (exp == NA) continue;
      GoldenCheck(c.id, launch(c), exp);
    }
  };

  SECTION("hipModuleLaunchKernel") {
    RunRow(kModule, [&](const CaseSpec& c) {
      return hipModuleLaunchKernel(f_noop, c.gx, c.gy, c.gz, c.bx, c.by, c.bz,
                                   static_cast<unsigned>(c.shared), nullptr, nullptr, nullptr);
    });
  }

  SECTION("hipExtModuleLaunchKernel") {
    RunRow(kExtModule, [&](const CaseSpec& c) {
      return hipExtModuleLaunchKernel(f_noop, GlobalWork(c.gx, c.bx), GlobalWork(c.gy, c.by),
                                      GlobalWork(c.gz, c.bz), c.bx, c.by, c.bz,
                                      static_cast<size_t>(c.shared), nullptr, nullptr, nullptr);
    });
  }

  SECTION("hipHccModuleLaunchKernel") {
    RunRow(kHccModule, [&](const CaseSpec& c) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      return hipHccModuleLaunchKernel(
          f_noop, GlobalWork(c.gx, c.bx), GlobalWork(c.gy, c.by), GlobalWork(c.gz, c.bz), c.bx,
          c.by, c.bz, static_cast<size_t>(c.shared), nullptr, nullptr, nullptr, nullptr, nullptr);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    });
  }

  SECTION("hipModuleLaunchCooperativeKernel") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
    RunRow(kModuleCoop, [&](const CaseSpec& c) {
      return hipModuleLaunchCooperativeKernel(f_noop, c.gx, c.gy, c.gz, c.bx, c.by, c.bz,
                                              static_cast<unsigned>(c.shared), nullptr, nullptr);
    });
  }

  SECTION("hipLaunchCooperativeKernel") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
    RunRow(kCoop, [&](const CaseSpec& c) {
      void* args[] = {};
      return hipLaunchCooperativeKernel((const void*)GoldenNoOpKernel, dim3(c.gx, c.gy, c.gz),
                                        dim3(c.bx, c.by, c.bz), args,
                                        static_cast<unsigned>(c.shared), nullptr);
    });
  }

  SECTION("hipDrvLaunchKernelEx(no attrs)") {
    RunRow(kDrvEx, [&](const CaseSpec& c) {
      HIP_LAUNCH_CONFIG cfg = MakeLaunchConfig(c, nullptr, 0);
      void* params[] = {};
      return hipDrvLaunchKernelEx(&cfg, f_noop, params, nullptr);
    });
  }

  SECTION("hipDrvLaunchKernelEx(clusterDim attr)") {
    RunRow(kDrvExCluster, [&](const CaseSpec& c) {
      hipLaunchAttribute attr{};
      attr.id = hipLaunchAttributeClusterDimension;
      attr.value.clusterDim.x = c.useCluster ? c.cx : 1;
      attr.value.clusterDim.y = c.useCluster ? c.cy : 1;
      attr.value.clusterDim.z = c.useCluster ? c.cz : 1;
      HIP_LAUNCH_CONFIG cfg = MakeLaunchConfig(c, &attr, 1);
      void* params[] = {};
      return hipDrvLaunchKernelEx(&cfg, f_noop, params, nullptr);
    });
  }

  SECTION("hipLaunchKernel") {
    RunRow(kLaunch, [&](const CaseSpec& c) {
      void* args[] = {};
      // The launch's own return is discarded: this entry point reports the configuration error
      // through hipGetLastError().
      static_cast<void>(hipLaunchKernel((const void*)GoldenNoOpKernel, dim3(c.gx, c.gy, c.gz),
                                        dim3(c.bx, c.by, c.bz), args, static_cast<size_t>(c.shared),
                                        nullptr));
      return hipGetLastError();
    });
  }

  SECTION("hipExtLaunchKernel") {
    RunRow(kExtLaunch, [&](const CaseSpec& c) {
      void* args[] = {};
      return hipExtLaunchKernel((const void*)GoldenNoOpKernel, dim3(c.gx, c.gy, c.gz),
                                dim3(c.bx, c.by, c.bz), args, static_cast<size_t>(c.shared),
                                nullptr, nullptr, nullptr, 0);
    });
  }

  SECTION("hipGraphAddKernelNode") {
    RunRow(kGraphAdd, [&](const CaseSpec& c) {
      hipGraph_t graph;
      HIP_CHECK(hipGraphCreate(&graph, 0));
      hipKernelNodeParams p = MakeNodeParams(c);
      hipGraphNode_t node;
      const hipError_t e = hipGraphAddKernelNode(&node, graph, nullptr, 0, &p);
      HIP_CHECK(hipGraphDestroy(graph));
      return e;
    });
  }

  SECTION("hipGraphKernelNodeSetParams") {
    RunRow(kGraphSet, [&](const CaseSpec& c) {
      GraphWithNode g;
      hipKernelNodeParams p = MakeNodeParams(c);
      return hipGraphKernelNodeSetParams(g.node, &p);
    });
  }

  SECTION("hipGraphExecKernelNodeSetParams") {
    RunRow(kGraphExecSet, [&](const CaseSpec& c) {
      GraphWithNode g;
      hipGraphExec_t exec;
      HIP_CHECK(hipGraphInstantiate(&exec, g.graph, nullptr, nullptr, 0));
      hipKernelNodeParams p = MakeNodeParams(c);
      const hipError_t e = hipGraphExecKernelNodeSetParams(exec, g.node, &p);
      HIP_CHECK(hipGraphExecDestroy(exec));
      return e;
    });
  }

  SECTION("hipModuleLaunchCooperativeKernelMultiDevice") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
      HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    }
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

    RunRow(kCoopMultiDev, [&](const CaseSpec& c) {
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
      return hipModuleLaunchCooperativeKernelMultiDevice(params.data(), deviceCount, 0u);
    });

    for (int i = 0; i < deviceCount; i++) {
      HIP_CHECK(hipStreamDestroy(streams[i]));
      HIP_CHECK(hipModuleUnload(modules[i]));
    }
    HIP_CHECK(hipSetDevice(0));
  }

  SECTION("<<<>>> (hipLaunchByPtr)") {
    RunRow(kByPtr, [&](const CaseSpec& c) {
      GoldenNoOpKernel<<<dim3(c.gx, c.gy, c.gz), dim3(c.bx, c.by, c.bz),
                         static_cast<size_t>(c.shared), 0>>>();
      return hipGetLastError();
    });
  }
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
