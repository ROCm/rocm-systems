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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Minimal kernel-replay tool library shared by the reproducers. LD_PRELOAD it next to
// any HIP application. Behaviour is chosen with KR_REPRO_MODE:
//
//   fixed       (default) replay every dispatch KR_REPRO_PASSES times, default 4
//   indefinite  return 0 from pass_count_cb and always continue -> unbounded loop (R4)
//
// It prints one line per replay phase to stderr so a reproducer script can see how far
// the loop got before a hang or a crash.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define RC(...)                                                                          \
    do                                                                                   \
    {                                                                                    \
        auto _s = (__VA_ARGS__);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                 \
            fprintf(stderr, "[repro-client] %s failed: %d\n", #__VA_ARGS__, (int) _s);     \
            abort();                                                                      \
        }                                                                                 \
    } while(0)

namespace
{
rocprofiler_context_id_t g_ctx{0};
std::atomic<long>        g_configs{0};
std::atomic<long>        g_passes{0};
bool                     g_indefinite = false;
uint64_t                 g_passes_req = 4;

uint64_t
repro_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    // 0 requests an indefinite loop, which is only legal with a continue callback.
    return g_indefinite ? 0 : g_passes_req;
}

int
repro_continue(rocprofiler_kernel_dispatch_info_t, uint64_t current, uint64_t total,
               rocprofiler_user_data_t)
{
    if(g_indefinite)
    {
        // Deliberately never stop. A bounded implementation must still terminate.
        if(current % 1000 == 0)
            fprintf(stderr, "[repro-client] indefinite loop at pass %lu\n",
                    (unsigned long) current);
        return 1;
    }
    return (current + 1 < total) ? 1 : 0;
}

void
replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* data = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        data->pass_count_cb      = repro_pass_count;
        data->replay_continue_cb = repro_continue;
        g_configs.fetch_add(1, std::memory_order_relaxed);
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        g_passes.fetch_add(1, std::memory_order_relaxed);
        fprintf(stderr, "[repro-client] pass %lu\n", (unsigned long) data->current_pass);
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    if(const char* mode = getenv("KR_REPRO_MODE"))
        g_indefinite = (strcmp(mode, "indefinite") == 0);
    if(const char* n = getenv("KR_REPRO_PASSES")) g_passes_req = strtoull(n, nullptr, 10);

    fprintf(stderr, "[repro-client] mode=%s passes=%lu\n",
            g_indefinite ? "indefinite" : "fixed", (unsigned long) g_passes_req);

    RC(rocprofiler_create_context(&g_ctx));
    RC(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY, nullptr, 0, replay_cb, nullptr));
    RC(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    fprintf(stderr, "[repro-client] fini configs=%ld passes=%ld\n", g_configs.load(),
            g_passes.load());
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name        = "kernel-replay-reproducer";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
