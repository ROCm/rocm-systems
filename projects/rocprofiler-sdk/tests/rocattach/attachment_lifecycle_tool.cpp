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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "../common/defines.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#include <dlfcn.h>

extern "C" int
rocprofiler_is_current_client_attachment(void);

namespace
{
auto configure_count      = std::atomic<int>{0};
auto initialize_count     = std::atomic<int>{0};
auto attach_count         = std::atomic<int>{0};
auto detach_count         = std::atomic<int>{0};
auto callbackless_context = rocprofiler_context_id_t{};
auto failure_contexts     = std::array<rocprofiler_context_id_t, 2>{};

void
check_count(const char* name, int actual, int expected)
{
    if(actual == expected) return;

    std::fprintf(stderr,
                 "Attachment lifecycle test FAILED: %s count was %d, expected %d\n",
                 name,
                 actual,
                 expected);
    std::abort();
}

int
tool_initialize(rocprofiler_client_finalize_t, void*)
{
    check_count("initialize", ++initialize_count, 1);
#if defined(ROCPROFILER_TEST_STARTUP_CLIENT)
    check_count("attachment role", rocprofiler_is_current_client_attachment(), 0);
#else
    check_count("attachment role", rocprofiler_is_current_client_attachment(), 1);
#endif
#if defined(ROCPROFILER_TEST_CALLBACKLESS_CLIENT)
    if(rocprofiler_create_context(&callbackless_context) != ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_start_context(callbackless_context) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "Attachment lifecycle test FAILED: callback-less context setup "
                     "failed\n");
        std::abort();
    }
#endif
    if(auto* value = std::getenv("ROCPROFILER_TEST_FAIL_ATTACH");
       value != nullptr && std::atoi(value) != 0)
    {
        for(auto& context : failure_contexts)
        {
            if(rocprofiler_create_context(&context) != ROCPROFILER_STATUS_SUCCESS)
            {
                std::fprintf(stderr,
                             "Attachment lifecycle test FAILED: failure context setup "
                             "failed\n");
                std::abort();
            }
        }
    }
    return 0;
}

void
tool_finalize(void*)
{
#if defined(ROCPROFILER_TEST_CALLBACKLESS_CLIENT)
    auto active = int{-1};
    if(rocprofiler_context_is_active(callbackless_context, &active) != ROCPROFILER_STATUS_SUCCESS ||
       active != 0)
    {
        std::fprintf(stderr,
                     "Attachment lifecycle test FAILED: callback-less context remained "
                     "active at finalization\n");
        std::abort();
    }
    std::printf("Callback-less attachment lifecycle test PASSED\n");
#elif defined(ROCPROFILER_TEST_STARTUP_CLIENT)
    check_count("configure", configure_count.load(), 1);
    check_count("initialize", initialize_count.load(), 1);
    check_count("attach", attach_count.load(), 0);
    check_count("detach", detach_count.load(), 0);
    std::printf("Startup client test PASSED: attach/detach were not invoked\n");
#else
    auto expected = 1;
    if(auto* value = std::getenv("ROCPROFILER_TEST_EXPECT_ATTACH_COUNT"))
        expected = std::atoi(value);
    auto expected_detach = expected;
    if(auto* value = std::getenv("ROCPROFILER_TEST_EXPECT_DETACH_COUNT"))
        expected_detach = std::atoi(value);

    check_count("configure", configure_count.load(), 1);
    check_count("initialize", initialize_count.load(), 1);
    check_count("attach", attach_count.load(), expected);
    check_count("detach", detach_count.load(), expected_detach);
    std::printf("Attachment lifecycle test PASSED: %d session(s)\n", expected);
#endif
}

#if !defined(ROCPROFILER_TEST_CALLBACKLESS_CLIENT)
int
tool_attach(rocprofiler_client_detach_t, rocprofiler_context_id_t*, uint64_t, void*)
{
#    if defined(ROCPROFILER_TEST_STARTUP_CLIENT)
    std::fprintf(stderr, "Attachment lifecycle test FAILED: startup client received tool_attach\n");
    std::abort();
#    else
    if(auto* startup_tool_path = std::getenv("ROCP_TOOL_LIBRARIES"))
    {
        auto* handle = dlopen(startup_tool_path, RTLD_LOCAL | RTLD_LAZY | RTLD_NOLOAD);
        auto* symbol =
            (handle != nullptr) ? dlsym(handle, "rocprofiler_test_startup_client_active") : nullptr;
        using startup_client_active_t = int (*)();
        auto startup_client_active    = reinterpret_cast<startup_client_active_t>(symbol);
        if(startup_client_active == nullptr || startup_client_active() != 1)
        {
            std::fprintf(stderr,
                         "Attachment lifecycle test FAILED: startup client was not active\n");
            std::abort();
        }
        dlclose(handle);
    }

    auto count = ++attach_count;
    std::printf("Attachment lifecycle: attach %d\n", count);
    const auto fail_attach = []() {
        auto* value = std::getenv("ROCPROFILER_TEST_FAIL_ATTACH");
        return (value != nullptr && std::atoi(value) != 0);
    }();
    if(failure_contexts.front().handle != 0)
    {
        for(auto context : failure_contexts)
        {
            auto active = int{-1};
            if(rocprofiler_context_is_active(context, &active) != ROCPROFILER_STATUS_SUCCESS ||
               active != 0)
            {
                std::fprintf(stderr,
                             "Attachment lifecycle test FAILED: failure context was not "
                             "clean before attach\n");
                std::abort();
            }
        }
        if(rocprofiler_start_context(failure_contexts.front()) != ROCPROFILER_STATUS_SUCCESS)
            std::abort();
    }
    if(fail_attach) return 1;
    if(failure_contexts.front().handle != 0 &&
       rocprofiler_start_context(failure_contexts.back()) != ROCPROFILER_STATUS_SUCCESS)
        std::abort();
    return 0;
#    endif
}

void
tool_detach(void*)
{
#    if defined(ROCPROFILER_TEST_STARTUP_CLIENT)
    std::fprintf(stderr, "Attachment lifecycle test FAILED: startup client received tool_detach\n");
    std::abort();
#    else
    auto count = ++detach_count;
    std::printf("Attachment lifecycle: detach %d\n", count);
    if(failure_contexts.front().handle != 0)
    {
        for(auto context : failure_contexts)
        {
            auto active = int{-1};
            if(rocprofiler_context_is_active(context, &active) != ROCPROFILER_STATUS_SUCCESS ||
               active != 0)
            {
                std::fprintf(stderr,
                             "Attachment lifecycle test FAILED: failure context remained "
                             "active during detach\n");
                std::abort();
            }
        }
    }
#    endif
}
#endif
}  // namespace

#if defined(ROCPROFILER_TEST_CALLBACKLESS_CLIENT)
extern "C" int
rocprofiler_test_callbackless_context_active() ROCPROFILER_TEST_PUBLIC_API;

extern "C" int
rocprofiler_test_callbackless_context_active()
{
    auto active = int{-1};
    return (rocprofiler_context_is_active(callbackless_context, &active) ==
            ROCPROFILER_STATUS_SUCCESS)
               ? active
               : -1;
}
#endif

#if defined(ROCPROFILER_TEST_STARTUP_CLIENT)
extern "C" int
rocprofiler_test_startup_client_active() ROCPROFILER_TEST_PUBLIC_API;

extern "C" int
rocprofiler_test_startup_client_active()
{
    return (configure_count.load() == 1 && initialize_count.load() == 1 &&
            attach_count.load() == 0 && detach_count.load() == 0)
               ? 1
               : 0;
}
#endif

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* client_id)
    ROCPROFILER_TEST_PUBLIC_API;

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* client_id)
{
    check_count("configure", ++configure_count, 1);
#if defined(ROCPROFILER_TEST_STARTUP_CLIENT)
    check_count("client priority", static_cast<int>(priority), 0);
    client_id->name = "startup-lifecycle-tool";
#else
    auto expected_priority = 0;
    if(auto* value = std::getenv("ROCPROFILER_TEST_EXPECT_CLIENT_PRIORITY"))
        expected_priority = std::atoi(value);
    check_count("client priority", static_cast<int>(priority), expected_priority);
    client_id->name = "attachment-lifecycle-tool";
#endif

    static auto result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        &tool_initialize,
        &tool_finalize,
        nullptr,
    };
    return &result;
}

#if !defined(ROCPROFILER_TEST_CALLBACKLESS_CLIENT)
extern "C" rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
    ROCPROFILER_TEST_PUBLIC_API;

extern "C" rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    static auto result = rocprofiler_tool_configure_attach_result_t{
        sizeof(rocprofiler_tool_configure_attach_result_t),
        &tool_attach,
        &tool_detach,
        nullptr,
    };
    return &result;
}
#endif
