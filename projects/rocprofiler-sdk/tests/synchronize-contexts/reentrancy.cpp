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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Regression test for the reentrancy guard on rocprofiler_synchronize_contexts.
//
// The call waits for record production to catch up. A thread that produces records cannot wait
// for itself, so the SDK marks those threads and the call returns
// ROCPROFILER_STATUS_ERROR_REENTRANT_CALL instead of blocking. Threads the SDK does NOT wait for
// -- the application's own, running an API tracing callback -- must keep working, because calling
// from a hipDeviceSynchronize EXIT callback is the feature's primary use.
//
// So the guard is wrong in two directions and this test asserts both: too little marking hangs the
// application, too much disables the API for its main caller. The covered set is listed by name in
// the report this prints, so it is derivable here rather than from a comment somewhere else.
//
// If a marker is deleted, the matching probe stops returning REENTRANT_CALL: either it reports a
// different status and the assertion fails, or it blocks and the test times out. Both are failures.

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_ROCP(...)                                                                            \
    {                                                                                              \
        auto _status = (__VA_ARGS__);                                                              \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[reentrancy] %s failed at line %i: %s\n",                                     \
                    #__VA_ARGS__,                                                                  \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_status));                                       \
            std::abort();                                                                          \
        }                                                                                          \
    }

#define CHECK_HIP(...)                                                                             \
    {                                                                                              \
        auto _err = (__VA_ARGS__);                                                                 \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[reentrancy] %s failed at line %i: %s\n",                                     \
                    #__VA_ARGS__,                                                                  \
                    __LINE__,                                                                      \
                    hipGetErrorString(_err));                                                      \
            std::abort();                                                                          \
        }                                                                                          \
    }

namespace
{
// One probe per thread class under test. `expect_reentrant` is the assertion: true for threads the
// SDK produces records on, false for the application's own thread.
struct probe
{
    const char*           thread_description;
    bool                  expect_reentrant;
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> matched{0};
    std::atomic<uint64_t> mismatched{0};
    std::atomic<int>      last_mismatch{ROCPROFILER_STATUS_SUCCESS};
};

// The marked set, named. A reader learns which threads are covered from this list, and a reviewer
// comparing it against the source needs no other document.
probe g_buffer_probe{"buffer delivery callback -- buffer.cpp flush task", true};
probe g_dispatch_probe{"kernel dispatch callback -- hsa/queue.cpp:AsyncSignalHandler", true};
probe g_counter_probe{
    "counter record callback -- counters/sample_processing.cpp:process_completed_cb",
    true};
probe g_app_thread_probe{"HIP API tracing callback -- the application's own thread", false};

rocprofiler_context_id_t g_trace_ctx   = {0};
rocprofiler_context_id_t g_counter_ctx = {0};
rocprofiler_buffer_id_t  g_buffer      = {0};

// The drain must never re-enter a probe. Without this a traced call made underneath the drain
// would recurse instead of reporting.
thread_local bool g_inside_probe = false;

void
run_probe(probe& p, rocprofiler_context_id_t ctx)
{
    if(g_inside_probe) return;
    g_inside_probe = true;

    const auto _status       = rocprofiler_synchronize_contexts(&ctx, 1);
    const bool _is_reentrant = (_status == ROCPROFILER_STATUS_ERROR_REENTRANT_CALL);

    p.calls.fetch_add(1, std::memory_order_relaxed);
    if(_is_reentrant == p.expect_reentrant)
    {
        p.matched.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        p.mismatched.fetch_add(1, std::memory_order_relaxed);
        p.last_mismatch.store(_status, std::memory_order_relaxed);
    }

    g_inside_probe = false;
}

void
buffer_callback(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t**,
                size_t num_headers,
                void*,
                uint64_t)
{
    if(num_headers > 0) run_probe(g_buffer_probe, g_trace_ctx);
}

void
dispatch_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*,
                          void*)
{
    // Select on the operation, not the phase. COMPLETE arrives as a phase-NONE callback from the
    // completion handler, which is the marked thread under test. The ENQUEUE operation's
    // enter/exit pair runs on the application's own thread and asserts the opposite claim.
    if(record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE)
        run_probe(g_dispatch_probe, g_trace_ctx);
}

// Deliberately NOT a queue-lifecycle API. hsa_queue_create/destroy callbacks can be holding a queue
// gate lock that the drain acquires, which is an open question tracked separately; a probe there
// would be reporting that rather than reporting on the guard.
void
hip_api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) run_probe(g_app_thread_probe, g_trace_ctx);
}

void
counter_record_callback(rocprofiler_dispatch_counting_service_data_t,
                        rocprofiler_counter_record_t*,
                        size_t,
                        rocprofiler_user_data_t,
                        void*)
{
    run_probe(g_counter_probe, g_counter_ctx);
}

void
counter_dispatch_callback(rocprofiler_dispatch_counting_service_data_t data,
                          rocprofiler_counter_config_id_t*             config,
                          rocprofiler_user_data_t*,
                          void*)
{
    static auto _mutex = std::mutex{};
    static auto _cache = std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>{};

    const auto _agent = data.dispatch_info.agent_id;
    {
        auto _lk = std::lock_guard<std::mutex>{_mutex};
        if(auto itr = _cache.find(_agent.handle); itr != _cache.end())
        {
            *config = itr->second;
            return;
        }
    }

    auto _all = std::vector<rocprofiler_counter_id_t>{};
    CHECK_ROCP(rocprofiler_iterate_agent_supported_counters(
        _agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* _v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for(size_t i = 0; i < n; ++i)
                _v->push_back(cs[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &_all));

    auto _want = std::vector<rocprofiler_counter_id_t>{};
    for(auto citr : _all)
    {
        auto _info = rocprofiler_counter_info_v0_t{};
        CHECK_ROCP(
            rocprofiler_query_counter_info(citr, ROCPROFILER_COUNTER_INFO_VERSION_0, &_info));
        if(_info.name && std::string{_info.name} == "SQ_WAVES") _want.push_back(citr);
    }

    if(_want.empty())
    {
        fprintf(stderr, "[reentrancy] SQ_WAVES is not available on this agent\n");
        std::abort();
    }

    auto _cfg = rocprofiler_counter_config_id_t{.handle = 0};
    CHECK_ROCP(rocprofiler_create_counter_config(_agent, _want.data(), _want.size(), &_cfg));
    {
        auto _lk = std::lock_guard<std::mutex>{_mutex};
        _cache.emplace(_agent.handle, _cfg);
    }
    *config = _cfg;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    CHECK_ROCP(rocprofiler_create_context(&g_trace_ctx));
    CHECK_ROCP(rocprofiler_create_buffer(g_trace_ctx,
                                         256 * 1024,
                                         128 * 1024,
                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                         buffer_callback,
                                         nullptr,
                                         &g_buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_tracing_service(
        g_trace_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, g_buffer));
    CHECK_ROCP(
        rocprofiler_configure_callback_tracing_service(g_trace_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                       nullptr,
                                                       0,
                                                       dispatch_tracing_callback,
                                                       nullptr));

    // One operation rather than the whole domain: every traced HIP call would otherwise run a
    // full drain, which turns a guard test into a throughput test.
    auto _hip_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_HIP_RUNTIME_API_ID_hipDeviceSynchronize};
    CHECK_ROCP(
        rocprofiler_configure_callback_tracing_service(g_trace_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                       _hip_ops.data(),
                                                       _hip_ops.size(),
                                                       hip_api_callback,
                                                       nullptr));

    // Counter collection gets its own context: the counter record callback is the only way to
    // reach the consumer thread, and the dispatch path it replaces is the one traced above.
    CHECK_ROCP(rocprofiler_create_context(&g_counter_ctx));
    CHECK_ROCP(rocprofiler_configure_callback_dispatch_counting_service(
        g_counter_ctx, counter_dispatch_callback, nullptr, counter_record_callback, nullptr));

    CHECK_ROCP(rocprofiler_start_context(g_trace_ctx));
    CHECK_ROCP(rocprofiler_start_context(g_counter_ctx));
    return 0;
}

void
tool_fini(void*)
{}
}  // namespace

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name         = "synchronize-contexts-reentrancy";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

namespace
{
__global__ void
touch_kernel(float* ptr)
{
    ptr[threadIdx.x] = static_cast<float>(threadIdx.x);
}

// A probe that never ran proves nothing, so an unexercised path is a failure rather than a pass.
bool
report(const probe& p)
{
    const auto  _calls      = p.calls.load();
    const auto  _matched    = p.matched.load();
    const auto  _mismatched = p.mismatched.load();
    const char* _expected   = p.expect_reentrant ? "REENTRANT_CALL" : "not REENTRANT_CALL";

    if(_calls == 0)
    {
        fprintf(stderr,
                "FAIL  %s\n        never exercised; expected %s\n",
                p.thread_description,
                _expected);
        return false;
    }
    if(_mismatched > 0)
    {
        fprintf(stderr,
                "FAIL  %s\n        expected %s on every call; %lu of %lu disagreed, last was %s\n",
                p.thread_description,
                _expected,
                static_cast<unsigned long>(_mismatched),
                static_cast<unsigned long>(_calls),
                rocprofiler_get_status_string(
                    static_cast<rocprofiler_status_t>(p.last_mismatch.load())));
        return false;
    }
    fprintf(stdout,
            "ok    %s\n        %s on all %lu calls\n",
            p.thread_description,
            _expected,
            static_cast<unsigned long>(_matched));
    return true;
}
}  // namespace

int
main()
{
    float* _dev = nullptr;
    CHECK_HIP(hipMalloc(&_dev, 64 * sizeof(float)));

    // Several iterations so each covered path is entered more than once and a single lucky status
    // cannot carry the run.
    for(int i = 0; i < 16; ++i)
    {
        hipLaunchKernelGGL(touch_kernel, dim3(1), dim3(64), 0, nullptr, _dev);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_ROCP(rocprofiler_flush_buffer(g_buffer));
    }

    CHECK_HIP(hipFree(_dev));

    fprintf(stdout, "\nreentrancy guard coverage\n-------------------------\n");
    bool _ok = true;
    _ok &= report(g_buffer_probe);
    _ok &= report(g_dispatch_probe);
    _ok &= report(g_counter_probe);
    _ok &= report(g_app_thread_probe);

    fprintf(stdout, "\n%s\n", _ok ? "PASS" : "FAILED");
    return _ok ? 0 : 1;
}
