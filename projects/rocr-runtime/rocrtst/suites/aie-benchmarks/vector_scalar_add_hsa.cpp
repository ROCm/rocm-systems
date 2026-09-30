// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via HSA (ROCR), PDI + instruction sequence.
//
// The counterpart of vector_scalar_add_hsa_elf.cpp, which prices the same design dispatched from
// a full ELF. Everything except which hsaco is loaded lives in aie_hsa_harness.h.

#include <benchmark/benchmark.h>

#include "aie_hsa_harness.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <vector>

namespace {

using aie_bench::dispatch_packet;
using aie_bench::HsaBumpAllocator;
using aie_bench::HsaHarness;
using aie_bench::KERNARG_ENTRIES;

const std::filesystem::path g_hsaco_path = STRINGIFY(DEFAULT_HSACO_PATH);
constexpr const char* g_kernel_name = DEFAULT_HSACO_KERNEL_NAME;

}  // namespace

// Dispatch N PDI+insts packets per iteration with the kernargs prepared up front.
static void VectorScalarAddHSA(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    HsaHarness h(num_dispatches, g_hsaco_path, g_kernel_name);

    std::vector<std::uint64_t*> kernargs(num_dispatches, nullptr);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      if (hsa_amd_memory_pool_allocate(h.kernarg_pool, KERNARG_ENTRIES * sizeof(std::uint64_t), 0,
                                       reinterpret_cast<void**>(&kernargs[i])) !=
          HSA_STATUS_SUCCESS) {
        state.SkipWithError("Failed to allocate kernarg buffer");
        return;
      }
    }

    for (auto _ : state) {
      std::uint64_t last_wr_idx = 0;
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        last_wr_idx =
            dispatch_packet(h.kernel_object, h.inputs[i], h.outputs[i], kernargs[i], h.queue);
      }

      // The doorbell store submits and waits, so the whole batch is timed.
      hsa_signal_store_screlease(h.queue->doorbell_signal, last_wr_idx);

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");

    for (auto* p : kernargs) {
      if (p) hsa_amd_memory_pool_free(p);
    }
  } catch (const std::exception& e) {
    // Google Benchmark does not catch, so without this a missing artifact or a failed allocation
    // terminates the whole run instead of failing one case.
    state.SkipWithError(e.what());
  }
}

// Same, but the kernargs are carved out of a bump allocator inside the loop, so the numbers
// include the cost of preparing arguments per dispatch.
static void VectorScalarAddHSAAllocKernargs(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    HsaHarness h(num_dispatches, g_hsaco_path, g_kernel_name);

    // Declared after the harness, so it is destroyed before it -- its pool memory must be freed
    // while the runtime is still up.
    HsaBumpAllocator kernarg_alloc(h.kernarg_pool,
                                   num_dispatches * KERNARG_ENTRIES * sizeof(std::uint64_t));

    for (auto _ : state) {
      kernarg_alloc.reset();
      std::uint64_t last_wr_idx = 0;
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        auto* kernargs = kernarg_alloc.allocate<std::uint64_t>(KERNARG_ENTRIES);
        last_wr_idx =
            dispatch_packet(h.kernel_object, h.inputs[i], h.outputs[i], kernargs, h.queue);
      }

      hsa_signal_store_screlease(h.queue->doorbell_signal, last_wr_idx);

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

BENCHMARK(VectorScalarAddHSA)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(VectorScalarAddHSAAllocKernargs)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
