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

// Checks which HIP runtime dispatch table entries rocprofiler-sdk wraps for a given set of
// registered contexts. The SDK's HIP graph and HIP event wrappers add work to every
// hipGraphLaunch / hipEventRecord / hipStreamWaitEvent, so they must only be installed when a
// context consumes them; thread trace alone must leave the launch path untouched.
//
// The HIP runtime table is inspected through rocprofiler_at_intercept_table_registration, which
// is invoked after the SDK has installed its own wrappers. An entry still resolving into the HIP
// runtime library (via dladdr) is unwrapped. Entries taking a hipStream_t (hipGraphLaunch,
// hipEventRecord, ...) may also be wrapped by the HIP stream wrappers, so the positive checks use
// hipGraphInstantiate and hipEventDestroy, which only the graph and event modules wrap.

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/tests/common.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hip/amd_detail/hip_api_trace.hpp>

#include <dlfcn.h>
#include <map>
#include <string>

namespace
{
using setup_func_t = void (*)(rocprofiler_context_id_t);

struct test_data
{
    setup_func_t                       setup            = nullptr;
    rocprofiler_context_id_t           ctx              = {};
    bool                               setup_supported  = true;
    bool                               table_registered = false;
    std::map<std::string, std::string> entry_libs       = {};  // table entry -> library path
};

auto&
get_test_data()
{
    static auto _v = test_data{};
    return _v;
}

std::string
library_of(const void* func)
{
    auto info = Dl_info{};
    if(func != nullptr && dladdr(func, &info) != 0 && info.dli_fname != nullptr)
        return std::string{info.dli_fname};
    return std::string{};
}

void
hip_table_registration(rocprofiler_intercept_table_t type,
                       uint64_t /*lib_version*/,
                       uint64_t /*lib_instance*/,
                       void**   tables,
                       uint64_t num_tables,
                       void* /*user_data*/)
{
    ASSERT_EQ(type, ROCPROFILER_HIP_RUNTIME_TABLE);
    ASSERT_EQ(num_tables, 1);

    const auto* table = static_cast<const HipDispatchTable*>(tables[0]);
    auto&       data  = get_test_data();

    data.table_registered = true;
    data.entry_libs       = {
        {"hipGraphLaunch", library_of(reinterpret_cast<const void*>(table->hipGraphLaunch_fn))},
        {"hipGraphInstantiate",
         library_of(reinterpret_cast<const void*>(table->hipGraphInstantiate_fn))},
        {"hipEventRecord", library_of(reinterpret_cast<const void*>(table->hipEventRecord_fn))},
        {"hipEventDestroy", library_of(reinterpret_cast<const void*>(table->hipEventDestroy_fn))},
        {"hipLaunchKernel", library_of(reinterpret_cast<const void*>(table->hipLaunchKernel_fn))}};
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    auto& data = get_test_data();
    ROCPROFILER_CALL(rocprofiler_create_context(&data.ctx), "failed to create context");
    // Only registration matters for which wrappers are installed, so the context is never
    // started and no hardware is programmed.
    data.setup(data.ctx);
    return 0;
}

// Registers the test's contexts, then initializes the HIP runtime (loaded at runtime so the test
// binary does not depend on HIP) so it hands its dispatch table to rocprofiler-sdk.
void
run_with_contexts(setup_func_t setup)
{
    get_test_data().setup = setup;

    static auto cfg_result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr, nullptr};

    static rocprofiler_configure_func_t configure =
        [](uint32_t,
           const char*,
           uint32_t,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        ROCPROFILER_CALL(rocprofiler_at_intercept_table_registration(
                             hip_table_registration, ROCPROFILER_HIP_RUNTIME_TABLE, nullptr),
                         "failed to register for HIP runtime table");
        return &cfg_result;
    };

    ASSERT_EQ(rocprofiler_force_configure(configure), ROCPROFILER_STATUS_SUCCESS);

    void* hip_lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    if(hip_lib == nullptr) GTEST_SKIP() << "HIP runtime not available: " << dlerror();

    using get_device_count_t = hipError_t (*)(int*);
    auto* get_device_count =
        reinterpret_cast<get_device_count_t>(dlsym(hip_lib, "hipGetDeviceCount"));
    ASSERT_NE(get_device_count, nullptr);

    // Any HIP API call initializes the runtime; the result itself does not matter here.
    int num_devices = 0;
    static_cast<void>(get_device_count(&num_devices));

    ASSERT_TRUE(get_test_data().table_registered) << "HIP runtime table was not registered";
}

void
expect_unwrapped(const char* name)
{
    const auto& lib = get_test_data().entry_libs.at(name);
    EXPECT_NE(lib.find("amdhip64"), std::string::npos)
        << name << " was wrapped (resolves into '" << lib << "')";
}

void
expect_wrapped(const char* name)
{
    const auto& lib = get_test_data().entry_libs.at(name);
    EXPECT_EQ(lib.find("amdhip64"), std::string::npos)
        << name << " was not wrapped (resolves into '" << lib << "')";
}
}  // namespace

TEST(rocprofiler_lib, hip_table_wrappers_thread_trace_only)
{
    run_with_contexts([](rocprofiler_context_id_t ctx) {
        static auto _ctx             = ctx;
        auto        configure_agents = [](rocprofiler_agent_version_t,
                                   const void** agents,
                                   size_t       num_agents,
                                   void*) -> rocprofiler_status_t {
            for(size_t i = 0; i < num_agents; ++i)
            {
                const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

                auto status = rocprofiler_configure_device_thread_trace_service(
                    _ctx,
                    agent->id,
                    nullptr,
                    0,
                    [](rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {},
                    rocprofiler_user_data_t{});
                if(status != ROCPROFILER_STATUS_SUCCESS) get_test_data().setup_supported = false;
                return ROCPROFILER_STATUS_SUCCESS;
            }
            get_test_data().setup_supported = false;
            return ROCPROFILER_STATUS_SUCCESS;
        };

        ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                            configure_agents,
                                                            sizeof(rocprofiler_agent_v0_t),
                                                            nullptr),
                         "failed to query agents");
    });

    if(!get_test_data().setup_supported)
        GTEST_SKIP() << "device thread trace is not supported on this system";

    for(const auto& itr : get_test_data().entry_libs)
        expect_unwrapped(itr.first.c_str());
}

// Positive control: contexts that consume the HIP graph and HIP event wrappers get them, so the
// test above cannot pass just because the table inspection failed to see the wrappers.
TEST(rocprofiler_lib, hip_table_wrappers_graph_and_event_tracing)
{
    run_with_contexts([](rocprofiler_context_id_t ctx) {
        for(auto kind :
            {ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH, ROCPROFILER_CALLBACK_TRACING_HIP_EVENT})
        {
            ROCPROFILER_CALL(
                rocprofiler_configure_callback_tracing_service(
                    ctx,
                    kind,
                    nullptr,
                    0,
                    [](rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*) {},
                    nullptr),
                "failed to configure callback tracing");
        }
    });

    expect_wrapped("hipGraphInstantiate");
    expect_wrapped("hipEventDestroy");
}

// Kernel replay relies on the HIP graph launch state to avoid replaying graph dispatches, so it
// must get the graph wrappers even without HIP_GRAPH tracing.
TEST(rocprofiler_lib, hip_table_wrappers_kernel_replay)
{
    run_with_contexts([](rocprofiler_context_id_t ctx) {
        auto status = rocprofiler_configure_callback_tracing_service(
            ctx,
            ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
            nullptr,
            0,
            [](rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*) {},
            nullptr);
        if(status != ROCPROFILER_STATUS_SUCCESS) get_test_data().setup_supported = false;
    });

    if(!get_test_data().setup_supported)
        GTEST_SKIP() << "kernel replay is not supported by this rocprofiler-sdk build";

    expect_wrapped("hipGraphInstantiate");
}
