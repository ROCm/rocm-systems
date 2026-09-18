// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via HSA (ROCR), full ELF.
//
// The counterpart of vector_scalar_add_hsa.cpp, which prices the same design dispatched from a
// PDI plus an instruction sequence. Everything except which hsaco is loaded, and the architecture
// it requires, lives in aie_hsa_harness.h.

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

const std::filesystem::path g_hsaco_path = STRINGIFY(DEFAULT_ELF_HSACO_PATH);
const char* const g_kernel_name = DEFAULT_ELF_KERNEL_NAME;

// Full-ELF dispatch is aie2p only; aie2 has neither the firmware command nor the preemption
// support it is built on.
constexpr const char* g_required_arch = "aie2p";

}  // namespace

// Dispatch N full-ELF packets per iteration with the kernargs prepared up front.
// The counterpart of VectorScalarAddHSA.
static void VectorScalarAddHSAELF(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    HsaHarness h(num_dispatches, g_hsaco_path, g_kernel_name, g_required_arch);

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
    // Google Benchmark does not catch, so without this a missing artifact or an unsupported agent
    // terminates the whole run instead of failing one case.
    state.SkipWithError(e.what());
  }
}

// Same, but the kernargs are carved out of a bump allocator inside the loop, so the numbers
// include the cost of preparing arguments per dispatch. The counterpart of
// VectorScalarAddHSAAllocKernargs.
static void VectorScalarAddHSAELFAllocKernargs(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    HsaHarness h(num_dispatches, g_hsaco_path, g_kernel_name, g_required_arch);

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

BENCHMARK(VectorScalarAddHSAELF)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(VectorScalarAddHSAELFAllocKernargs)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
