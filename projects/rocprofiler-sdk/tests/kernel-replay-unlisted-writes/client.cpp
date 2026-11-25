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

/**
 * @file tests/kernel-replay-unlisted-writes/client.cpp
 *
 * @brief Client tool for the kernel replay test of writes outside a reused restore list.
 *
 * LD_PRELOAD-ed alongside main.cpp, which registers a pinned host int through
 * kr_unlisted_register_tile(). The tool replays the workload's kernel (workgroup size 128) with
 * four passes and stores the pass index in that int at each PASS PHASE_ENTER, so every pass writes
 * a different tile.
 */

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
constexpr uint32_t kWriteBlockX = 128;
constexpr int      kPasses      = 4;

rocprofiler_context_id_t g_context{0};
std::atomic<int*>        g_tile{nullptr};
std::atomic<int>         g_replays{0};
std::atomic<int>         g_passes{0};

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(p->dispatch_info.workgroup_size.x != kWriteBlockX) return;

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        p->replay_pass_count = replay_pass_count;
        g_replays.fetch_add(1);
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(auto* tile = g_tile.load()) *tile = static_cast<int>(p->current_pass);
        g_passes.fetch_add(1);
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    if(rocprofiler_create_context(&g_context) != ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_configure_callback_tracing_service(g_context,
                                                      ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                      nullptr,
                                                      0,
                                                      kernel_replay_cb,
                                                      nullptr) != ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_start_context(g_context) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "[unlisted-writes tool] FAIL: could not configure kernel replay\n");
        return -1;
    }
    return 0;
}

void
tool_fini(void*)
{
    std::fprintf(
        stderr, "[unlisted-writes tool] replays=%d passes=%d\n", g_replays.load(), g_passes.load());
    if(g_replays.load() != 1 || g_passes.load() != kPasses)
    {
        std::fprintf(
            stderr, "[unlisted-writes tool] FAIL: expected 1 replay of %d passes\n", kPasses);
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}
}  // namespace

// Default visibility, since the tests build hides symbols and the workload finds this one with
// dlsym(RTLD_DEFAULT, "kr_unlisted_register_tile").
extern "C" __attribute__((visibility("default"))) void
kr_unlisted_register_tile(int* tile)
{
    g_tile.store(tile);
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-unlisted-writes";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
