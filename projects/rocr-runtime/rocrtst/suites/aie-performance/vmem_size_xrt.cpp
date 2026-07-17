// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Microbenchmark comparing XRT buffer-object allocation cost across a sweep of
// allocation sizes (8 KiB, 16 KiB, 32 KiB, 64 KiB) on the NPU.
//
// XRT has no virtual-memory reserve/map API; its allocation primitive is
// xrt::bo. Two cases are provided:
//   BoAllocXRT     - buffer construct/destruct only.
//   BoAllocMapXRT  - construct + map to a host pointer + destruct.

#include <benchmark/benchmark.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <cstdint>
#include <string>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static std::string g_xclbin_path = STRINGIFY(DEFAULT_XCLBIN_PATH);

namespace {

/// @brief Group id of the host-only input argument in the suite kernel, used to
/// allocate benchmark buffer objects (matches the existing XRT benchmarks).
constexpr int INPUT_GROUP_ARG = 3;

/// @brief Holds the XRT objects needed to allocate buffer objects, kept alive
/// for the duration of a benchmark case.
struct XrtContext {
  xrt::device device;
  xrt::kernel kernel;
  int group = 0;
};

/// @brief Open device 0, load the suite xclbin, and construct the kernel so a
/// valid memory group id is available for buffer allocation.
///
/// All work here is one-time setup performed outside the timed loop. On any
/// failure the benchmark case is skipped (not failed) so the suite can run on
/// machines lacking the device or toolchain artifacts.
///
/// @param[in,out] state  Benchmark state; skip status is written here.
/// @param[out]    ctx    Receives the device/kernel/group on success.
/// @return true if setup succeeded, false otherwise.
bool setup_xrt(benchmark::State& state, XrtContext* ctx) {
  try {
    ctx->device = xrt::device(0);
    auto xclbin = xrt::xclbin(g_xclbin_path);

    auto xkernels = xclbin.get_kernels();
    auto xkernel_it = std::find_if(xkernels.begin(), xkernels.end(), [](const auto& k) {
      return k.get_name().rfind("MLIR_AIE", 0) == 0;
    });
    if (xkernel_it == xkernels.end()) {
      state.SkipWithError("Kernel MLIR_AIE not found in xclbin");
      return false;
    }

    ctx->device.register_xclbin(xclbin);
    xrt::hw_context context(ctx->device, xclbin.get_uuid());
    ctx->kernel = xrt::kernel(context, xkernel_it->get_name());
    ctx->group = ctx->kernel.group_id(INPUT_GROUP_ARG);
  } catch (const std::exception& e) {
    state.SkipWithError((std::string("XRT setup failed: ") + e.what()).c_str());
    return false;
  }
  return true;
}

}  // namespace

/// @brief Time xrt::bo construction (allocation) and destruction for a single
/// host-only buffer of `state.range(0)` bytes.
static void BoAllocXRT(benchmark::State& state) {
  XrtContext ctx;
  if (!setup_xrt(state, &ctx)) {
    return;
  }

  const auto size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    xrt::bo bo(ctx.device, size, XRT_BO_FLAGS_HOST_ONLY, ctx.group);
    benchmark::DoNotOptimize(bo);
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(size));
}

/// @brief Time xrt::bo construction, mapping to a host pointer, and destruction
/// for a single host-only buffer of `state.range(0)` bytes.
static void BoAllocMapXRT(benchmark::State& state) {
  XrtContext ctx;
  if (!setup_xrt(state, &ctx)) {
    return;
  }

  const auto size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    xrt::bo bo(ctx.device, size, XRT_BO_FLAGS_HOST_ONLY, ctx.group);
    void* ptr = bo.map<void*>();
    benchmark::DoNotOptimize(ptr);
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(size));
}

BENCHMARK(BoAllocXRT)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(BoAllocMapXRT)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
