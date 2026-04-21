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

#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static std::string g_elf_path = STRINGIFY(DEFAULT_ELF_PATH);
static std::string g_kernel_name = STRINGIFY(DEFAULT_KERNEL_NAME);

static constexpr int N = 1024;

static void VectorScalarAddELF(benchmark::State &state) {
    // Load full ELF (contains PDI + instructions + metadata)
    xrt::elf elf(g_elf_path);

    // Open device; ELF configures the hardware context directly
    auto device = xrt::device(0);
    xrt::hw_context context(device, elf);

    // Create kernel from the ELF-configured context
    auto kernel = xrt::ext::kernel(context, g_kernel_name);

    // Allocate buffer objects (no group_id needed with ext::bo)
    xrt::bo bo_in = xrt::ext::bo{device, N * sizeof(int32_t)};
    xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(int32_t)};

    // Initialize input: [1, 2, 3, ..., 1024]
    auto *buf_in = bo_in.map<int32_t *>();
    std::iota(buf_in, buf_in + N, 1);

    // Zero output
    auto *buf_out = bo_out.map<int32_t *>();
    std::memset(buf_out, 0, N * sizeof(int32_t));

    // Benchmark loop: sync in -> dispatch -> wait -> sync out
    for (auto _ : state) {
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kernel(opcode, 0, 0, bo_in, bo_out);
        run.wait2();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        benchmark::ClobberMemory();
    }
}

BENCHMARK(VectorScalarAddELF)->Unit(benchmark::kMicrosecond);
