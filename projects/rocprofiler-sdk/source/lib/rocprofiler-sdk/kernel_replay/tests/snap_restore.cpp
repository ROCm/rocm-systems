// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Note on assertions: real interception records EVERY device allocation, including runtime-internal
// ones from kernel launches / hipMemcpy (the known over-capture behavior). Tests therefore assert
// deltas around their own hipMalloc/hipFree and check pointer membership, never absolute inventory
// sizes.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace rocprofiler;
namespace mt   = kernel_replay::memory_tracker;
namespace msnp = kernel_replay::memory_snapshot;

namespace
{
namespace kernel
{
__global__ void
fill(float* d, float val, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        d[i] = val;
}

// d[i] = base + i : a non-uniform pattern so a stale/partial restore is easy to catch.
__global__ void
iota(float* d, float base, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        d[i] = base + static_cast<float>(i);
}

// In-place read-write kernel: the canonical case restore must protect (y is both read and written).
__global__ void
saxpy(float* y, const float* x, float a, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

// In-place add (another read-write mutation).
__global__ void
add(float* d, float delta, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        d[i] = d[i] + delta;
}
}  // namespace kernel

constexpr int NUM_THREADS = 1024;

int
blocks_for(int n)
{
    return (n + NUM_THREADS - 1) / NUM_THREADS;
}

rocprofiler_context_id_t g_ctx{};

void
tracing_noop(rocprofiler_callback_tracing_record_t /*record*/,
             rocprofiler_user_data_t* /*user_data*/,
             void* /*callback_data*/)
{}

int
tool_init(rocprofiler_client_finalize_t /*fini*/, void* /*tool_data*/)
{
    if(rocprofiler_create_context(&g_ctx) != ROCPROFILER_STATUS_SUCCESS) return -1;
    rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API, nullptr, 0, tracing_noop, nullptr);
    rocprofiler_start_context(g_ctx);
    return 0;
}

void
tool_fini(void* /*tool_data*/)
{}

rocprofiler_tool_configure_result_t*
configure(uint32_t /*version*/,
          const char* /*runtime_version*/,
          uint32_t /*priority*/,
          rocprofiler_client_id_t* id)
{
    id->name        = "kernel-replay-snapshot-test";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}

bool
ensure_live_tracking()
{
    static bool ok = [] {
        if(rocprofiler_force_configure(&configure) != ROCPROFILER_STATUS_SUCCESS) return false;
        if(hipInit(0) != hipSuccess) return false;
        int devs = 0;
        if(hipGetDeviceCount(&devs) != hipSuccess || devs == 0) return false;
        return true;
    }();
    // Tracking is normally enabled when a tool configures the KERNEL_REPLAY callback-tracing
    // service (rocprofiler_configure_callback_tracing_service). This unit test drives snap/restore
    // directly without that service, so enable it here (same statically-linked instance).
    if(ok) mt::set_tracking_enabled(true);
    return ok;
}

// ------------------------- device helpers -------------------------
bool
inventory_contains(void* p)
{
    auto inv = mt::snap_inventory();
    return inv.find(p) != inv.end();
}

std::vector<float>
read_device(const float* d, int n)
{
    std::vector<float> out(n);
    EXPECT_EQ(
        hipMemcpy(out.data(), d, static_cast<size_t>(n) * sizeof(float), hipMemcpyDeviceToHost),
        hipSuccess);
    return out;
}

void
sync_ok()
{
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}

void
launch_fill(float* d, float val, int n)
{
    kernel::fill<<<blocks_for(n), NUM_THREADS>>>(d, val, n);
    sync_ok();
}

void
launch_iota(float* d, float base, int n)
{
    kernel::iota<<<blocks_for(n), NUM_THREADS>>>(d, base, n);
    sync_ok();
}

void
launch_saxpy(float* y, const float* x, float a, int n)
{
    kernel::saxpy<<<blocks_for(n), NUM_THREADS>>>(y, x, a, n);
    sync_ok();
}

void
launch_add(float* d, float delta, int n)
{
    kernel::add<<<blocks_for(n), NUM_THREADS>>>(d, delta, n);
    sync_ok();
}

constexpr size_t N_ELEMS = 64U * 1024U * 1024U;
}  // namespace

// A plain hipMalloc must be captured automatically by the tracker the SDK installed on the live HSA
// table (no manual record_alloc), and hipFree must auto-remove it.
TEST(kernel_replay_snapshot, hipmalloc_autocaptured_and_freed)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const size_t before_alloc = mt::snap_inventory().size();
    const size_t bytes        = N_ELEMS * sizeof(float);

    float* buffer = nullptr;

    ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);
    ASSERT_NE(buffer, nullptr);

    EXPECT_EQ(mt::snap_inventory().size(), before_alloc + 1)
        << "hipMalloc was not auto-captured by the live-table tracker";
    EXPECT_TRUE(inventory_contains(buffer))
        << "our device pointer is not in the auto-populated inventory";

    const size_t before_free = mt::snap_inventory().size();
    ASSERT_EQ(hipFree(buffer), hipSuccess);
    EXPECT_EQ(mt::snap_inventory().size(), before_free - 1) << "hipFree was not auto-removed";
    EXPECT_FALSE(inventory_contains(buffer)) << "freed pointer still present in inventory";
}

// iota A -> snapshot -> (sanity A) -> in-place kernel mutation (sanity mutated) -> restore -> A.
TEST(kernel_replay_snapshot, restore_reverts_device_memory)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const size_t bytes = N_ELEMS * sizeof(float);

    float* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);  // auto-captured
    ASSERT_NE(buffer, nullptr);
    ASSERT_TRUE(inventory_contains(buffer));

    launch_iota(buffer, 1.0f, N_ELEMS);
    {
        // sanity: device holds A
        auto a = read_device(buffer, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(a[i], 1.0f + i) << "pre-mutation elem " << i;
    }

    msnp::Snapshot snapshot{};
    snapshot.snap();

    launch_add(buffer, 9000.0f, N_ELEMS);
    {
        // control: mutation landed
        auto b = read_device(buffer, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(b[i], 9001.0f + i) << "mutated elem " << i;
    }

    snapshot.restore();
    {
        // restore reverted to A
        auto a = read_device(buffer, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(a[i], 1.0f + i) << "post-restore elem " << i;
    }

    ASSERT_EQ(hipFree(buffer), hipSuccess);
}

// Restoring between passes must stop an in-place kernel from accumulating: N saxpy passes with a
// restore each should net one application (y0 + a), not N. A no-op restore would leave y0 + N*a.
TEST(kernel_replay_snapshot, restore_prevents_inplace_accumulation_across_passes)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    constexpr int   passes = 5;
    constexpr float y0     = 100.0f;
    constexpr float a      = 2.0f;  // x == 1 -> each saxpy pass adds exactly `a`
    const size_t    bytes  = N_ELEMS * sizeof(float);

    float* x = nullptr;
    float* y = nullptr;
    ASSERT_EQ(hipMalloc(&x, bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&y, bytes), hipSuccess);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);

    launch_fill(x, 1.0f, N_ELEMS);
    launch_fill(y, y0, N_ELEMS);

    msnp::Snapshot snapshot{};
    snapshot.snap();

    for(int pass = 0; pass < passes; ++pass)
    {
        launch_saxpy(y, x, a, N_ELEMS);
        {
            // sensitivity: mutation landed this pass
            auto mutated = read_device(y, N_ELEMS);
            for(size_t i = 0; i < N_ELEMS; ++i)
                ASSERT_FLOAT_EQ(mutated[i], y0 + a) << "pass " << pass << " elem " << i;
        }

        snapshot.restore();
        {
            // restore reverts to snapped inputs -- no accumulation into the next pass
            auto reverted = read_device(y, N_ELEMS);
            for(size_t i = 0; i < N_ELEMS; ++i)
                ASSERT_FLOAT_EQ(reverted[i], y0) << "post-restore pass " << pass << " elem " << i;
        }
    }

    {  // after N passes the buffer is exactly the snapped inputs, not y0 + N*a
        auto final_state = read_device(y, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(final_state[i], y0) << "final elem " << i;
    }

    ASSERT_EQ(hipFree(y), hipSuccess);
    ASSERT_EQ(hipFree(x), hipSuccess);
}

// A single snapshot must revert every tracked device allocation the test created, not just one.
TEST(kernel_replay_snapshot, restore_reverts_multiple_buffers)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const size_t bytes = N_ELEMS * sizeof(float);

    constexpr int                  kBufs = 3;
    std::array<float*, kBufs>      buffers{};
    const std::array<float, kBufs> base = {1.0f, 100.0f, 10000.0f};

    for(int b = 0; b < kBufs; ++b)
    {
        ASSERT_EQ(hipMalloc(&buffers[b], bytes), hipSuccess);  // auto-captured
        ASSERT_NE(buffers[b], nullptr);
        launch_iota(buffers[b], base[b], N_ELEMS);
    }

    msnp::Snapshot snapshot{};
    snapshot.snap();

    for(int b = 0; b < kBufs; ++b)
    {
        launch_add(buffers[b], 77000.0f, N_ELEMS);
    }

    for(int b = 0; b < kBufs; ++b)
    {
        // sensitivity: mutations landed on every buffer
        auto mutated = read_device(buffers[b], N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(mutated[i], base[b] + 77000.0f + i) << "buf " << b << " elem " << i;
    }

    snapshot.restore();

    for(int b = 0; b < kBufs; ++b)
    {
        // every buffer reverted to its own pattern
        auto reverted = read_device(buffers[b], N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(reverted[i], base[b] + i) << "restored buf " << b << " elem " << i;
    }

    for(int b = 0; b < kBufs; ++b)
        ASSERT_EQ(hipFree(buffers[b]), hipSuccess);
}

// The tracking gate must fully suppress inventory population: a hipMalloc made while tracking is
// disabled must not be recorded (the fast-path check in the wrappers).
TEST(kernel_replay_snapshot, disabled_tracking_records_nothing)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const size_t bytes = 4096 * sizeof(float);

    ASSERT_EQ(mt::set_tracking_enabled(false), false);
    const size_t before = mt::snap_inventory().size();

    float* d = nullptr;
    ASSERT_EQ(hipMalloc(&d, bytes), hipSuccess);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(mt::snap_inventory().size(), before) << "allocation recorded while tracking disabled";
    EXPECT_FALSE(inventory_contains(d)) << "disabled tracking still recorded the pointer";

    // restore the gate before freeing (and for subsequent tests)
    ASSERT_EQ(mt::set_tracking_enabled(true), true);
    ASSERT_EQ(hipFree(d), hipSuccess);
}
