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

#include "snap_kernels.hpp"

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/utils.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace rocprofiler;
namespace mt   = kernel_replay::memory_tracker;
namespace msnp = kernel_replay::memory_snapshot;

namespace
{
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
// The GPU agent that owns the test's device allocations. Tests run with a single visible device, so
// the first GPU agent is the one hipMalloc allocates on.
hsa_agent_t
gpu_agent()
{
    for(const auto* rocp_agent : rocprofiler::agent::get_agents())
    {
        if(rocp_agent == nullptr || rocp_agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        if(auto hsa = rocprofiler::agent::get_hsa_agent(rocp_agent); hsa.has_value()) return *hsa;
    }
    return hsa_agent_t{.handle = 0};
}

bool
inventory_contains(void* p, hsa_agent_t agent)
{
    auto inv = mt::snap_inventory(agent);
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
    kernel_launch::fill(d, val, n);
    sync_ok();
}

void
launch_iota(float* d, float base, int n)
{
    kernel_launch::iota(d, base, n);
    sync_ok();
}

void
launch_saxpy(float* y, const float* x, float a, int n)
{
    kernel_launch::saxpy(y, x, a, n);
    sync_ok();
}

void
launch_add(float* d, float delta, int n)
{
    kernel_launch::add(d, delta, n);
    sync_ok();
}

// 32 MB/buffer (8Mi floats). Large enough to exercise real multi-MB DMA snap/restore, yet well
// under the FP32 exact-integer limit (2^24) so `base + i` compares exactly. These snapshot tests
// also share a ctest RESOURCE_LOCK (see tests/CMakeLists.txt) so they never run concurrently with
// each other, bounding the 3-buffer test to ~192 MB device+host. At 256 MB/buffer, and unlocked,
// the 3-buffer restore_reverts_multiple_buffers test intermittently OOM'd/partial-copied on shared
// CI GPUs (single/double-buffer variants passed), producing zeroed tail regions after restore.
constexpr size_t N_ELEMS = 8U * 1024U * 1024U;
}  // namespace

// A plain hipMalloc must be captured automatically by the tracker the SDK installed on the live HSA
// table (no manual record_alloc), and hipFree must auto-remove it.
TEST(kernel_replay_snapshot, hipmalloc_autocaptured_and_freed)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    const size_t before_alloc = mt::snap_inventory(agent).size();
    const size_t bytes        = N_ELEMS * sizeof(float);

    float* buffer = nullptr;

    ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);
    ASSERT_NE(buffer, nullptr);

    EXPECT_EQ(mt::snap_inventory(agent).size(), before_alloc + 1)
        << "hipMalloc was not auto-captured by the live-table tracker";
    EXPECT_TRUE(inventory_contains(buffer, agent))
        << "our device pointer is not in the auto-populated inventory";

    const size_t before_free = mt::snap_inventory(agent).size();
    ASSERT_EQ(hipFree(buffer), hipSuccess);
    EXPECT_EQ(mt::snap_inventory(agent).size(), before_free - 1) << "hipFree was not auto-removed";
    EXPECT_FALSE(inventory_contains(buffer, agent)) << "freed pointer still present in inventory";
}

// iota A -> snapshot -> (sanity A) -> in-place kernel mutation (sanity mutated) -> restore -> A.
TEST(kernel_replay_snapshot, restore_reverts_device_memory)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    const size_t bytes = N_ELEMS * sizeof(float);

    float* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);  // auto-captured
    ASSERT_NE(buffer, nullptr);
    ASSERT_TRUE(inventory_contains(buffer, agent));

    launch_iota(buffer, 1.0f, N_ELEMS);
    {
        // sanity: device holds A
        auto a = read_device(buffer, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(a[i], 1.0f + i) << "pre-mutation elem " << i;
    }

    auto snapshot = msnp::snap(agent);

    launch_add(buffer, 9000.0f, N_ELEMS);
    {
        // control: mutation landed
        auto b = read_device(buffer, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(b[i], 9001.0f + i) << "mutated elem " << i;
    }

    ASSERT_TRUE(msnp::restore(snapshot));
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
// The saxpy kernel also bumps a __device__ module global once per pass, so this covers both tracked
// buffer restore AND untracked module-variable restore (snap()'s HSA_SYMBOL_KIND_VARIABLE path).
TEST(kernel_replay_snapshot, restore_prevents_inplace_accumulation_across_passes)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

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

    // Seed the __device__ module counter so snap() captures a known baseline (0); the saxpy kernel
    // bumps it once per pass, so a working module-variable restore keeps it at 0 across passes.
    kernel_launch::set_module_counter(0);

    auto snapshot = msnp::snap(agent);

    for(int pass = 0; pass < passes; ++pass)
    {
        launch_saxpy(y, x, a, N_ELEMS);
        {
            // sensitivity: buffer mutation landed this pass
            auto mutated = read_device(y, N_ELEMS);
            for(size_t i = 0; i < N_ELEMS; ++i)
                ASSERT_FLOAT_EQ(mutated[i], y0 + a) << "pass " << pass << " elem " << i;
            // and the __device__ module global was bumped this pass
            ASSERT_EQ(kernel_launch::read_module_counter(), 1)
                << "module counter after saxpy, pass " << pass;
        }

        ASSERT_TRUE(msnp::restore(snapshot));
        {
            // restore reverts to snapped inputs -- no accumulation into the next pass
            auto reverted = read_device(y, N_ELEMS);
            for(size_t i = 0; i < N_ELEMS; ++i)
                ASSERT_FLOAT_EQ(reverted[i], y0) << "post-restore pass " << pass << " elem " << i;
            // the module global was reverted too (not just the buffer)
            ASSERT_EQ(kernel_launch::read_module_counter(), 0)
                << "module counter post-restore, pass " << pass;
        }
    }

    {  // after N passes the buffer is exactly the snapped inputs, not y0 + N*a
        auto final_state = read_device(y, N_ELEMS);
        for(size_t i = 0; i < N_ELEMS; ++i)
            ASSERT_FLOAT_EQ(final_state[i], y0) << "final elem " << i;
    }
    EXPECT_EQ(kernel_launch::read_module_counter(), 0)
        << "final module counter accumulated across passes instead of being restored";

    ASSERT_EQ(hipFree(y), hipSuccess);
    ASSERT_EQ(hipFree(x), hipSuccess);
}

// A single snapshot must revert every tracked device allocation the test created, not just one.
TEST(kernel_replay_snapshot, restore_reverts_multiple_buffers)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

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

    auto snapshot = msnp::snap(agent);

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

    ASSERT_TRUE(msnp::restore(snapshot));

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

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    const size_t bytes = 4096 * sizeof(float);

    ASSERT_EQ(mt::set_tracking_enabled(false), false);
    const size_t before = mt::snap_inventory(agent).size();

    float* d = nullptr;
    ASSERT_EQ(hipMalloc(&d, bytes), hipSuccess);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(mt::snap_inventory(agent).size(), before)
        << "allocation recorded while tracking disabled";
    EXPECT_FALSE(inventory_contains(d, agent)) << "disabled tracking still recorded the pointer";

    // restore the gate before freeing (and for subsequent tests)
    ASSERT_EQ(mt::set_tracking_enabled(true), true);
    ASSERT_EQ(hipFree(d), hipSuccess);
}

// Snapshots must exclude kernarg (restoring it mid-kernel faults the GPU) and host/CPU memory.
// query_alloc drops kernarg/fine-grained at record time; snap_inventory() scopes out other agents.
TEST(kernel_replay_snapshot, pool_filter_excludes_kernarg_and_cpu)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    // device VRAM: trackable, lands in the snapshot.
    float* dev = nullptr;
    ASSERT_EQ(hipMalloc(&dev, 4096), hipSuccess);
    ASSERT_NE(dev, nullptr);
    EXPECT_TRUE(mt::query_alloc(dev).trackable) << "device VRAM should be snapshot-trackable";
    EXPECT_TRUE(inventory_contains(dev, agent)) << "device VRAM missing from the agent snapshot";

    // host memory: goes through interception, must not land in the GPU agent's snapshot.
    float* host = nullptr;
    ASSERT_EQ(hipHostMalloc(reinterpret_cast<void**>(&host), 4096), hipSuccess);
    ASSERT_NE(host, nullptr);
    EXPECT_FALSE(inventory_contains(host, agent)) << "host memory leaked into the agent snapshot";

    // kernarg: nothing routes it through the tracker, so classify a real kernarg pointer directly.
    auto cache = agent::get_agent_cache(agent);
    ASSERT_TRUE(cache.has_value()) << "no agent cache for the GPU agent";
    const auto kernarg_pool = cache->kernarg_pool();
    ASSERT_NE(kernarg_pool.handle, 0U) << "GPU agent has no kernarg pool";

    void* karg = nullptr;
    ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, 256, 0, &karg), HSA_STATUS_SUCCESS);
    ASSERT_NE(karg, nullptr);
    EXPECT_FALSE(mt::query_alloc(karg).trackable)
        << "kernarg memory must be excluded from snapshots (a stale restore faults the GPU)";
    EXPECT_FALSE(inventory_contains(karg, agent))
        << "kernarg pointer present in the agent snapshot";
    EXPECT_EQ(hsa_amd_memory_pool_free(karg), HSA_STATUS_SUCCESS);

    EXPECT_EQ(hipHostFree(host), hipSuccess);
    ASSERT_EQ(hipFree(dev), hipSuccess);
}

// Snapshots must exclude allocations carrying HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG from the main
// inventory (HIP kernarg pools / profiler buffers share that flag). When the allocation is
// otherwise trackable (coarse device VRAM -- what a direct-HSA app may put behind the flag), it is
// recorded in the unsupported side inventory. snap() does not decline on that side inventory: the
// HIP runtime routinely keeps trackable+executable allocations live, so declining would disable
// replay for ordinary HIP apps. Assert the inventory exclusion (and side-table bookkeeping)
// directly via the intercepted HSA table so the check does not depend on replay-window
// serialization.
TEST(kernel_replay_snapshot, executable_flag_excluded_from_inventory)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    struct pool_pick_t
    {
        hsa_amd_memory_pool_t pool{};
        bool                  found = false;
    } pick{};

    (void) hsa_amd_agent_iterate_memory_pools(
        agent,
        [](hsa_amd_memory_pool_t p, void* data) -> hsa_status_t {
            auto*             out = static_cast<pool_pick_t*>(data);
            hsa_amd_segment_t segment{};
            if(hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) !=
               HSA_STATUS_SUCCESS)
                return HSA_STATUS_SUCCESS;
            if(segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
            uint32_t flags = 0;
            if(hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) !=
               HSA_STATUS_SUCCESS)
                return HSA_STATUS_SUCCESS;
            constexpr uint32_t kFine    = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
            constexpr uint32_t kKernarg = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
            if((flags & kFine) != 0 || (flags & kKernarg) != 0) return HSA_STATUS_SUCCESS;
            out->pool  = p;
            out->found = true;
            return HSA_STATUS_INFO_BREAK;
        },
        &pick);

    if(!pick.found) GTEST_SKIP() << "no coarse global memory pool on agent";

    // Interceptor wrappers are installed on the runtime-facing HSA table, not on
    // get_amd_ext_table()'s internal (unwrapped) copy. tracking_pool_allocate/free invoke those
    // same wrappers so this unit test exercises the executable-flag path directly.
    constexpr uint32_t kExecutableFlag = (1U << 2);  // HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG
    void*              ptr             = nullptr;
    ASSERT_EQ(mt::tracking_pool_allocate(pick.pool, 4096, kExecutableFlag, &ptr),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(ptr, nullptr);

    EXPECT_FALSE(inventory_contains(ptr, agent))
        << "executable-flag allocation must not enter the snapshot inventory";

    const auto q = mt::query_alloc(ptr);
    if(q.trackable)
    {
        EXPECT_EQ(mt::unsupported_executable(agent).count(ptr), 1U)
            << "trackable+executable allocation must enter the unsupported side inventory";
        EXPECT_TRUE(mt::agent_has_unsupported_executable(agent));
    }

    ASSERT_EQ(mt::tracking_pool_free(ptr), HSA_STATUS_SUCCESS);
    EXPECT_EQ(mt::unsupported_executable(agent).count(ptr), 0U)
        << "side inventory must clear this pointer on free";
}

// A __device__ module global mutated by a kernel must be reverted by restore(). It is not a
// hipMalloc allocation -- it lives in the loaded executable's data segment and is captured only by
// snap()'s HSA_SYMBOL_KIND_VARIABLE path. Without that path this reads kBase+1 after restore.
TEST(kernel_replay_snapshot, restore_reverts_module_variable)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    constexpr int kBase = 7;
    kernel_launch::set_module_counter(kBase);
    ASSERT_EQ(kernel_launch::read_module_counter(), kBase) << "failed to seed __device__ global";

    auto snapshot = msnp::snap(agent);

    kernel_launch::bump_module_counter();
    sync_ok();
    ASSERT_EQ(kernel_launch::read_module_counter(), kBase + 1) << "kernel mutation did not land";

    ASSERT_TRUE(msnp::restore(snapshot));
    EXPECT_EQ(kernel_launch::read_module_counter(), kBase)
        << "module variable (__device__ global) was not restored by snapshot/restore";
}

// The replay scenario for a module global: N passes each bump it; a restore between passes must
// prevent accumulation, so the final value stays kBase (each pass reverts kBase+1 -> kBase), never
// kBase + N. A no-op (missing) module-variable restore would leave kBase + N.
TEST(kernel_replay_snapshot, restore_prevents_module_variable_accumulation_across_passes)
{
    if(!ensure_live_tracking()) GTEST_SKIP() << "could not activate rocprofiler / no HIP GPU";

    const auto agent = gpu_agent();
    ASSERT_NE(agent.handle, 0U) << "no GPU agent found";

    constexpr int kBase  = 0;
    constexpr int passes = 5;
    kernel_launch::set_module_counter(kBase);

    auto snapshot = msnp::snap(agent);

    for(int pass = 0; pass < passes; ++pass)
    {
        kernel_launch::bump_module_counter();
        sync_ok();
        ASSERT_EQ(kernel_launch::read_module_counter(), kBase + 1) << "mutation, pass " << pass;

        ASSERT_TRUE(msnp::restore(snapshot));
        ASSERT_EQ(kernel_launch::read_module_counter(), kBase) << "post-restore, pass " << pass;
    }

    EXPECT_EQ(kernel_launch::read_module_counter(), kBase)
        << "module variable accumulated across passes instead of being restored";
}
