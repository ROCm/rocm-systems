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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/**
 * @file tests/range-replay-host/fence_timeout_client.cpp
 *
 * @brief LD_PRELOAD tool for fence_timeout.cpp. Asks for host replay of both of its ranges and
 * checks that the one opened while a kernel still ran declined with CONCURRENT_DISPATCH without a
 * replayed pass, and that the one opened after it replayed every pass.
 */

#include "fence_timeout.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-fence] error '%s' @%d: %s\n",                                             \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
struct range_seen_t
{
    std::atomic<int>      closes{0};
    std::atomic<int>      status{-1};
    std::atomic<int>      pass_enters{0};
    std::atomic<int>      pass_exits{0};
    std::atomic<uint64_t> divergence{0};
};

rocprofiler_context_id_t g_ctx{0};
range_seen_t             g_blocked{};
range_seen_t             g_replayed{};

range_seen_t*
seen(uint64_t range_id)
{
    if(range_id == kBlockedRangeId) return &g_blocked;
    if(range_id == kReplayedRangeId) return &g_replayed;
    return nullptr;
}

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    auto* p     = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    auto* range = (p != nullptr) ? seen(p->range_id) : nullptr;
    if(range == nullptr) return;

    if(record.operation == ROCPROFILER_RANGE_REPLAY_PASS)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            range->pass_enters.fetch_add(1);
        else
            range->pass_exits.fetch_add(1);
        return;
    }

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            p->pass_count_cb = pass_count;
            p->host_replay   = 1;
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            range->closes.fetch_add(1);
            range->status.store(static_cast<int>(p->status));
            range->divergence.store(p->divergence_count);
            break;
        default: break;
    }
}

bool
check_range(const char* name, const range_seen_t& range, int want_status, int want_passes)
{
    const auto status = range.status.load();
    fprintf(stderr,
            "[rr-fence] %s range: closes=%d status=%d pass_enters=%d pass_exits=%d "
            "divergence=%lu\n",
            name,
            range.closes.load(),
            status,
            range.pass_enters.load(),
            range.pass_exits.load(),
            static_cast<unsigned long>(range.divergence.load()));

    auto ok = true;
    if(range.closes.load() != 1)
    {
        fprintf(stderr, "[rr-fence] FAIL: expected exactly one CLOSE for the %s range\n", name);
        ok = false;
    }
    if(status != want_status)
    {
        fprintf(stderr,
                "[rr-fence] FAIL: the %s range closed with status %d, expected %d\n",
                name,
                status,
                want_status);
        ok = false;
    }
    if(range.pass_enters.load() != want_passes || range.pass_exits.load() != want_passes)
    {
        fprintf(stderr,
                "[rr-fence] FAIL: expected %d replayed passes of the %s range\n",
                want_passes,
                name);
        ok = false;
    }
    if(range.divergence.load() != 0)
    {
        fprintf(stderr, "[rr-fence] FAIL: replayed passes of the %s range diverged\n", name);
        ok = false;
    }
    return ok;
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
    const bool blocked =
        check_range("blocked", g_blocked, ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH, 0);
    const bool replayed =
        check_range("replayed", g_replayed, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, kPasses - 1);
    fprintf(stderr, blocked && replayed ? "[rr-fence] PASS\n" : "[rr-fence] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-fence";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
