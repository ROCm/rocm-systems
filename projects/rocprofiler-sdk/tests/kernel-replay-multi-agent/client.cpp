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
 * @file tests/kernel-replay-multi-agent/client.cpp
 *
 * @brief Client tool for the multi-agent kernel replay test.
 *
 * LD_PRELOAD-ed alongside main.cpp. It replays every dispatch with workgroup size 192, the
 * workload's accumulate kernel, with three passes on whichever GPU it runs. KR_SNAPSHOT=host-pinned
 * or KR_SNAPSHOT=device limits where the SDK keeps the snapshot. At finalize it checks that two
 * GPUs each replayed KR_LAUNCHES dispatches, every one with three passes and one CONFIG
 * PHASE_EXIT.
 */

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace
{
constexpr uint32_t kAccumulateBlockX = 192;
constexpr int      kPasses           = 3;
constexpr size_t   kAgents           = 2;

struct agent_counts_t
{
    int replays      = 0;
    int config_exits = 0;
    int passes       = 0;
};

rocprofiler_context_id_t           g_context{0};
std::mutex                         g_mutex{};
std::map<uint64_t, agent_counts_t> g_counts{};

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

bool
placement_is(const char* name)
{
    const char* placement = std::getenv("KR_SNAPSHOT");
    return placement != nullptr && std::strcmp(placement, name) == 0;
}

// The options are static because the SDK reads them after the CONFIG callback returns.
const rocprofiler_kernel_replay_options_t*
snapshot_options()
{
    static const auto options = []() {
        auto flags = uint64_t{ROCPROFILER_KERNEL_REPLAY_OPTION_NONE};
        if(placement_is("host-pinned"))
            flags = ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_HOST_PINNED;
        else if(placement_is("device"))
            flags = ROCPROFILER_KERNEL_REPLAY_OPTION_SNAPSHOT_DEVICE;
        return rocprofiler_kernel_replay_options_t{sizeof(rocprofiler_kernel_replay_options_t),
                                                   flags};
    }();
    return &options;
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(p->dispatch_info.workgroup_size.x != kAccumulateBlockX) return;

    const auto agent = p->dispatch_info.agent_id.handle;
    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        p->replay_pass_count = replay_pass_count;
        // An older SDK's record ends before replay_options.
        if(p->size >= offsetof(rocprofiler_callback_tracing_kernel_replay_data_t, replay_options) +
                          sizeof(p->replay_options))
            p->replay_options = snapshot_options();
        auto lock = std::lock_guard{g_mutex};
        ++g_counts[agent].replays;
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG)
    {
        auto lock = std::lock_guard{g_mutex};
        ++g_counts[agent].config_exits;
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        auto lock = std::lock_guard{g_mutex};
        ++g_counts[agent].passes;
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
        std::fprintf(stderr, "[multi-agent tool] FAIL: could not configure kernel replay\n");
        return -1;
    }
    return 0;
}

void
tool_fini(void*)
{
    const char* launches_env = std::getenv("KR_LAUNCHES");
    const int   launches     = launches_env != nullptr ? std::atoi(launches_env) : 4;

    auto lock = std::lock_guard{g_mutex};
    auto ok   = g_counts.size() == kAgents;
    for(const auto& [agent, counts] : g_counts)
    {
        std::fprintf(stderr,
                     "[multi-agent tool] agent %lu: replays=%d config_exits=%d passes=%d\n",
                     static_cast<unsigned long>(agent),
                     counts.replays,
                     counts.config_exits,
                     counts.passes);
        ok = ok && counts.replays == launches && counts.config_exits == launches &&
             counts.passes == launches * kPasses;
    }
    if(!ok)
    {
        std::fprintf(
            stderr,
            "[multi-agent tool] FAIL: expected %zu GPUs with %d replays of %d passes each\n",
            kAgents,
            launches,
            kPasses);
        // An abort would make ctest report a failure even when SKIP_REGULAR_EXPRESSION matches, and
        // _Exit skips the flush of the application's buffered output.
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-multi-agent";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
