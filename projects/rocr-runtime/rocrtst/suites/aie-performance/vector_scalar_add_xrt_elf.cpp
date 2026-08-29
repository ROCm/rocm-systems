// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via XRT (full-ELF API).
// Measures combined: sync input -> dispatch -> wait -> sync output.

#include <benchmark/benchmark.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static std::string g_elf_path = STRINGIFY(DEFAULT_ELF_PATH);
static std::string g_kernel_name = DEFAULT_KERNEL_NAME;

static constexpr int N = 1024;

namespace {

// Per-benchmark fixture: load the ELF, open the device and allocate one input
// and one output buffer per dispatch.
struct ElfHarness {
  xrt::elf elf;
  xrt::device device;
  xrt::hw_context context;
  xrt::ext::kernel kernel;
  std::vector<xrt::bo> bo_ins;
  std::vector<xrt::bo> bo_outs;

  explicit ElfHarness(std::int32_t num_dispatches)
      : elf(g_elf_path),
        device(0),
        // The ELF configures the hardware context directly; no xclbin needed.
        context(device, elf),
        kernel(context, g_kernel_name) {
    bo_ins.reserve(num_dispatches);
    bo_outs.reserve(num_dispatches);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      // xrt::ext::bo is host-only, matching the flags the xclbin benchmark uses.
      bo_ins.emplace_back(xrt::ext::bo{device, N * sizeof(int32_t)});
      bo_outs.emplace_back(xrt::ext::bo{device, N * sizeof(int32_t)});

      // Initialize input: [1, 2, ..., 1024]
      auto* buf_in = bo_ins[i].map<int32_t*>();
      std::iota(buf_in, buf_in + N, 1);

      // Initialize output: all zeros. The mapping is cached, so the write has
      // to be flushed before the first dispatch -- otherwise the dirty lines
      // write back over the DMA'd result and the head of the output reads as
      // zero.
      auto* buf_out = bo_outs[i].map<int32_t*>();
      std::fill_n(buf_out, N, 0);
      bo_outs[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  // Verify the kernel actually ran; otherwise the numbers are timing a no-op
  // dispatch.
  bool verify() const {
    for (const auto& bo_out : bo_outs) {
      const auto* buf_out = bo_out.map<const int32_t*>();
      for (int i = 0; i < N; ++i) {
        if (buf_out[i] != i + 2) return false;
      }
    }
    return true;
  }
};

}  // namespace

static void VectorScalarAddELFKernel(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    ElfHarness h(num_dispatches);

    // Benchmark loop: sync in -> dispatch -> wait -> sync out (all dispatches)
    std::vector<xrt::run> runs(num_dispatches);
    for (auto _ : state) {
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        h.bo_ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        xrt::run run(h.kernel);
        run.set_arg(0, h.bo_ins[i]);
        run.set_arg(1, h.bo_outs[i]);
        run.start();
        runs[i] = std::move(run);
      }

      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        runs[i].wait();
        h.bo_outs[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      }

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");
  } catch (const std::exception& e) {
    // Google Benchmark does not catch, so without this a missing artifact or a
    // failed dispatch terminates the whole run instead of failing one case.
    state.SkipWithError(e.what());
  }
}

static void VectorScalarAddELFRunlist(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    ElfHarness h(num_dispatches);

    // Bind once outside the loop; the runlist is re-executed each iteration.
    xrt::runlist runlist(h.context);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      xrt::run run(h.kernel);
      run.set_arg(0, h.bo_ins[i]);
      run.set_arg(1, h.bo_outs[i]);
      runlist.add(std::move(run));
    }

    // Benchmark loop: sync inputs -> execute -> wait -> sync outputs
    for (auto _ : state) {
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        h.bo_ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }

      runlist.execute();
      runlist.wait();

      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        h.bo_outs[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      }

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

BENCHMARK(VectorScalarAddELFKernel)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
BENCHMARK(VectorScalarAddELFRunlist)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
