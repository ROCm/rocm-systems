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
 * @file tests/range-replay-concurrency/client.cpp
 *
 * @brief LD_PRELOAD tool that replays the application's range and times a second thread's graph
 * launches into the replay window.
 *
 * The window restores the agent's whole tracked inventory before every pass and once more at the
 * end, so the second thread's buffers are rewound with everything else. Its graphs must therefore
 * wait for the window to close rather than run inside it: a launch that ran inside the window would
 * have its writes restored away, and its buffer would end short of the count the application
 * expects. The handshake makes that overlap happen on every run:
 *   - PASS enter, pass 1: the window is open (the range was taken off the thread, so a launch can
 * no longer decline it as a concurrent dispatch); tell the graph thread to start.
 *   - PASS enter, last pass: wait until the graph thread has begun launching, then give the launch
 *     time to reach the queue while the window is still held.
 *   - CLOSE: tell the graph thread the range is over, whether or not it was replayed.
 */

#include "range.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-concurrency] error '%s' @%d: %s\n",                                       \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

// Exported with default visibility on purpose: the tests build compiles with hidden visibility,
// but the application resolves this symbol with dlsym(RTLD_DEFAULT, "rr_coord_get").
extern "C" __attribute__((visibility("default"))) rr_coord_t*
rr_coord_get()
{
    static rr_coord_t coord{};
    return &coord;
}

namespace
{
rocprofiler_context_id_t g_ctx{0};

std::atomic<int>      g_configs{0};
std::atomic<int>      g_passes{0};
std::atomic<int>      g_closes{0};
std::atomic<int>      g_status{ROCPROFILER_RANGE_REPLAY_STATUS_LAST};
std::atomic<uint64_t> g_dispatches{0};
std::atomic<uint64_t> g_divergence{0};
std::atomic<int>      g_overlapped{0};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

void
await_first_launch()
{
    auto*      coord    = rr_coord_get();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while(coord->launches_started.load() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});

    if(coord->launches_started.load() == 0) return;

    // The flag is raised just before the first hipGraphLaunch; this is the time that launch has to
    // reach the queue and wait on the agent's replay lock while the window still holds it.
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    g_overlapped.store(1);
}

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    auto* data = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(data->range_id != kRangeId) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            data->pass_count_cb = pass_count;
            g_configs.fetch_add(1);
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS:
            g_passes.fetch_add(1);
            if(data->current_pass == 1) rr_coord_get()->window_open.store(1);
            if(data->current_pass == kPasses - 1) await_first_launch();
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_status.store(data->status);
            g_dispatches.store(data->dispatch_count);
            g_divergence.store(data->divergence_count);
            g_closes.fetch_add(1);
            rr_coord_get()->range_closed.store(1);
            break;
        default: break;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    RC(rocprofiler_create_context(&g_ctx));
    RC(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, nullptr, 0, range_replay_cb, nullptr));
    RC(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    if(const char* reason = std::getenv(kSkipEnv); reason != nullptr)
    {
        fprintf(stderr, "[rr-concurrency] SKIP: %s\n", reason);
        return;
    }

    fprintf(stderr,
            "[rr-concurrency] configs=%d passes=%d closes=%d status=%d dispatches=%lu "
            "divergence=%lu overlapped=%d\n",
            g_configs.load(),
            g_passes.load(),
            g_closes.load(),
            g_status.load(),
            static_cast<unsigned long>(g_dispatches.load()),
            static_cast<unsigned long>(g_divergence.load()),
            g_overlapped.load());

    auto ok = true;
    if(g_configs.load() != 1 || g_closes.load() != 1)
    {
        fprintf(stderr, "[rr-concurrency] FAIL: expected exactly one CONFIG and one CLOSE\n");
        ok = false;
    }
    if(g_status.load() != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED))
    {
        fprintf(stderr, "[rr-concurrency] FAIL: the range was not replayed\n");
        ok = false;
    }
    if(g_passes.load() != static_cast<int>(kPasses - 1))
    {
        fprintf(stderr,
                "[rr-concurrency] FAIL: expected %lu replayed passes\n",
                static_cast<unsigned long>(kPasses - 1));
        ok = false;
    }
    if(g_dispatches.load() != kRangeDispatches)
    {
        fprintf(stderr, "[rr-concurrency] FAIL: CLOSE reported the wrong dispatch count\n");
        ok = false;
    }
    // A graph that ran inside the window would have written between the exit snapshot and the
    // final pass's snapshot.
    if(std::getenv("ROCPROF_RANGE_REPLAY_VERIFY") != nullptr && g_divergence.load() != 0)
    {
        fprintf(stderr,
                "[rr-concurrency] FAIL: the final pass diverged from the application's run\n");
        ok = false;
    }
    if(g_overlapped.load() == 0)
    {
        fprintf(stderr,
                "[rr-concurrency] FAIL: the graph thread never started launching inside the replay "
                "window, so nothing was tested\n");
        ok = false;
    }

    fprintf(stderr, ok ? "[rr-concurrency] PASS\n" : "[rr-concurrency] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-concurrency";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
