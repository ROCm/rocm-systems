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

// Unit test for kernel_replay single-agent memory snap/restore, exercised through real HSA table
// interception (the counters/tests test_init() pattern): build the real HSA tables, copy them into
// the SDK's internal tables, then install the tracker's wrappers on top. Device allocations made
// through the wrapped amd_ext table are recorded into the inventory automatically -- no manual
// record_alloc.
//
// A kernel that writes a buffer is, for snap/restore purposes, just "something that changes device
// memory", so the test simulates the mutation with a device copy: fill -> snapshot -> mutate ->
// restore must revert to the snapped contents (and the mutated-readback proves the test is
// sensitive to restore).

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdlib>
#include <vector>

namespace mt = rocprofiler::kernel_replay::memory_tracker;
using namespace rocprofiler::counters::test_constants;

namespace
{
// Build real HSA tables, copy them into the SDK internal tables (so get_core_table()->...copy_fn is
// live), bring up the agent cache, then install the kernel_replay tracker wrappers and enable it.
void
init_tracking()
{
    // Surface the kernel_replay logs: enable the debug gate and lower the absl log level so
    // ROCP_INFO is emitted to stderr.
    setenv("ROCPROFILER_KERNEL_REPLAY_DEBUG", "1", 1);
    absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

    HsaApiTable table{};
    table.core_    = &get_api_table();
    table.amd_ext_ = &get_ext_table();
    rocprofiler::hsa::copy_table(table.core_, 0);
    rocprofiler::hsa::copy_table(table.amd_ext_, 0);

    rocprofiler::agent::construct_agent_cache(&table);
    ASSERT_NE(rocprofiler::hsa::get_queue_controller(), nullptr);
    rocprofiler::hsa::get_queue_controller()->init(get_api_table(), get_ext_table());

    // install our inventory wrappers on top of the SDK internal tables
    mt::update_table(rocprofiler::hsa::get_core_table());
    mt::update_table(rocprofiler::hsa::get_amd_ext_table());
    ASSERT_EQ(mt::set_tracking_enabled(true), true);
}

// First GPU agent that has both a device pool and a host-accessible pool.
bool
find_gpu_agent(hsa_amd_memory_pool_t& gpu_pool, hsa_amd_memory_pool_t& cpu_pool)
{
    for(const auto& [_, agent] : rocprofiler::hsa::get_queue_controller()->get_supported_agents())
    {
        if(agent.gpu_pool().handle != 0 && agent.cpu_pool().handle != 0)
        {
            gpu_pool = agent.gpu_pool();
            cpu_pool = agent.cpu_pool();
            return true;
        }
    }
    return false;
}

// Allocate device memory through the WRAPPED amd_ext table so the tracker records it.
void*
alloc_tracked_device(hsa_amd_memory_pool_t pool, size_t bytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(rocprofiler::hsa::get_amd_ext_table()->hsa_amd_memory_pool_allocate_fn(
                  pool, bytes, 0, &ptr),
              HSA_STATUS_SUCCESS);
    return ptr;
}

// Host staging from the (real) cpu pool; deliberately NOT through the wrapped table so it does not
// pollute the inventory.
void*
alloc_host(hsa_amd_memory_pool_t cpu_pool, size_t bytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(hsa_amd_memory_pool_allocate(cpu_pool, bytes, 0, &ptr), HSA_STATUS_SUCCESS);
    return ptr;
}
}  // namespace

// fill A -> snapshot -> (sanity A) -> overwrite B (sanity B) -> restore -> must read back A.
TEST(kernel_replay_snapshot, restore_reverts_device_memory)
{
    init_tracking();

    hsa_amd_memory_pool_t gpu_pool{}, cpu_pool{};
    if(!find_gpu_agent(gpu_pool, cpu_pool)) GTEST_SKIP() << "no GPU agent with usable pools";

    constexpr int n     = 4096;
    constexpr int check = 256;
    const size_t  bytes = n * sizeof(float);

    auto* d = static_cast<float*>(alloc_tracked_device(gpu_pool, bytes));
    auto* h = static_cast<float*>(alloc_host(cpu_pool, bytes));
    ASSERT_NE(d, nullptr);
    ASSERT_NE(h, nullptr);

    auto read_back = [&]() {
        std::vector<float> out(n);
        EXPECT_EQ(hsa_memory_copy(out.data(), d, bytes), HSA_STATUS_SUCCESS);
        return out;
    };
    auto write_device = [&](float base) {
        for(int i = 0; i < n; ++i)
            h[i] = base + static_cast<float>(i);
        EXPECT_EQ(hsa_memory_copy(d, h, bytes), HSA_STATUS_SUCCESS);
    };

    // pattern A
    write_device(1.0f);

    rocprofiler::kernel_replay::memory_snapshot::Snapshot snapshot{};

    {  // sanity: device currently holds A
        auto a = read_back();
        for(int i = 0; i < check; ++i)
            ASSERT_FLOAT_EQ(a[i], 1.0f + i) << "pre-mutation elem " << i;
    }

    snapshot.snap();

    // mutate to pattern B (stand-in for a kernel writing the buffer)
    write_device(9000.0f);
    {  // control: mutation actually landed (so the restore check can't pass trivially)
        auto b = read_back();
        for(int i = 0; i < check; ++i)
            ASSERT_FLOAT_EQ(b[i], 9000.0f + i) << "mutated elem " << i;
    }

    snapshot.restore();

    {  // restore must have reverted device memory to pattern A
        auto a = read_back();
        for(int i = 0; i < check; ++i)
            ASSERT_FLOAT_EQ(a[i], 1.0f + i) << "post-restore elem " << i;
    }

    EXPECT_EQ(rocprofiler::hsa::get_amd_ext_table()->hsa_amd_memory_pool_free_fn(d),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(h), HSA_STATUS_SUCCESS);
}
