// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler/context.h>
#include <rocprofiler/fwd.h>
#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/common/defines.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <dlfcn.h>
#include <pthread.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string_view>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS)                                             \
            << MSG << " (" << _status << "=" << rocprofiler_get_status_string(_status)             \
            << ") :: " << #ARG;                                                                    \
    }

#define ROCPROFILER_CALL_EXPECT(ARG, MSG, STATUS)                                                  \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, STATUS) << MSG << " (" << _status << "="                                \
                                   << rocprofiler_get_status_string(_status) << ") :: " << #ARG;   \
    }

namespace
{
struct callback_data
{
    using callback_count_data_t = std::pair<uint64_t, uint64_t>;
    using callback_count_map_t  = std::map<std::string_view, callback_count_data_t>;

    rocprofiler_client_id_t*      client_id             = nullptr;
    rocprofiler_client_finalize_t client_fini_func      = nullptr;
    rocprofiler_context_id_t      client_ctx            = {};
    rocprofiler_buffer_id_t       client_buffer         = {};
    rocprofiler_callback_thread_t client_thread         = {};
    uint64_t                      client_workflow_count = {};
    uint64_t                      client_elapsed        = {};
    int64_t                       current_depth         = 0;
    int64_t                       max_depth             = 0;
    callback_count_map_t          client_callback_count = {};
};

struct agent_data
{
    uint64_t                       agent_count = 0;
    std::vector<hsa_device_type_t> agents      = {};
};

using callback_kind_names_t = std::map<rocprofiler_callback_tracing_kind_t, const char*>;
using callback_kind_operation_names_t =
    std::map<rocprofiler_callback_tracing_kind_t, std::map<uint32_t, const char*>>;
using wrap_count_t = std::pair<std::string_view, size_t>;

struct callback_name_info
{
    callback_kind_names_t           kind_names      = {};
    callback_kind_operation_names_t operation_names = {};
};

auto&
get_preincrement()
{
    static size_t _v = 0;
    return _v;
}

auto&
get_postincrement()
{
    static size_t _v = 0;
    return _v;
}

auto&
get_client_callback_data()
{
    static auto _v = callback_data{};
    return _v;
}

auto
get_callback_tracing_names()
{
    auto cb_name_info = callback_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_callback_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<callback_name_info*>(data_v);

            if(kindv == ROCPROFILER_CALLBACK_TRACING_HSA_API)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query callback tracing kind operation name");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }
            return 0;
        };

    //
    //  callback for each callback kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<callback_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query callback tracing kind operation name");
        if(name) name_info_v->kind_names[kind] = name;

        if(kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating callback tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb,
                                                                static_cast<void*>(&cb_name_info)),
                     "iterating callback tracing kinds");

    return cb_name_info;
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* /*user_data*/,
                      void* client_data)
{
    auto* cb_data = static_cast<callback_data*>(client_data);

    static auto name_map = get_callback_tracing_names();
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        cb_data->client_callback_count[name_map.operation_names[record.kind][record.operation]]
            .second += get_preincrement();
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        cb_data->client_callback_count[name_map.operation_names[record.kind][record.operation]]
            .second += get_postincrement();
    }
}

template <size_t Idx, typename RetT, typename... Args>
auto
generate_wrapper(const char* name, RetT (*func)(Args...))
{
    using functor_type = RetT (*)(Args...);

    static functor_type underlying_func = func;
    static auto         func_name       = std::string_view{name};

    get_client_callback_data().client_callback_count.emplace(
        std::string_view{name}, callback_data::callback_count_data_t{0, 0});

    static functor_type wrapped_func = [](Args... args) -> RetT {
        get_client_callback_data().client_callback_count.at(func_name).first +=
            get_preincrement() + get_postincrement();
        if(underlying_func) return underlying_func(args...);
        if constexpr(!std::is_void<RetT>::value) return RetT{};
    };

    return wrapped_func;
}

#define GENERATE_WRAPPER(TABLE, FUNC)                                                              \
    TABLE->FUNC##_fn = generate_wrapper<__COUNTER__>(#FUNC, TABLE->FUNC##_fn)

void
api_registration_callback(rocprofiler_runtime_library_t type,
                          uint64_t                      lib_version,
                          uint64_t                      lib_instance,
                          void**                        tables,
                          uint64_t                      num_tables,
                          void*                         client_data)
{
    auto* cb_data = static_cast<callback_data*>(client_data);

    cb_data->client_workflow_count++;

    EXPECT_EQ(type, ROCPROFILER_HSA_LIBRARY) << "unexpected library type: " << type;
    EXPECT_EQ(lib_instance, 0) << "multiple instances of HSA runtime library";
    EXPECT_EQ(num_tables, 1) << "expected only one table of type HsaApiTable";
    EXPECT_GT(lib_version, 0) << "expected library version > 0";

    auto* hsa_api_table = static_cast<HsaApiTable*>(tables[0]);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_init);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_agent_get_info);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_iterate_agents);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_shut_down);
}
}  // namespace

TEST(rocprofiler_lib, intercept_table_and_callback_tracing)
{
    // test layering of tool interception of API table on top of rocprofiler API tracing.
    // This test enables both rocprofiler API tracing and a tool installing it's own
    // wrappers via the HsaApiTable. With both active, one should see the same
    // number of calls to "hsa_init", "hsa_iterate_agents", "hsa_agent_get_info", and
    // "hsa_shut_down"

    using init_func_t             = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t             = void (*)(void*);
    using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    get_preincrement()  = 1;
    get_postincrement() = 0;

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        auto operations = std::vector<uint32_t>{ROCPROFILER_HSA_API_ID_hsa_init,
                                                ROCPROFILER_HSA_API_ID_hsa_iterate_agents,
                                                ROCPROFILER_HSA_API_ID_hsa_agent_get_info,
                                                ROCPROFILER_HSA_API_ID_hsa_shut_down};

        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(cb_data->client_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_HSA_API,
                                                           operations.data(),
                                                           operations.size(),
                                                           tool_tracing_callback,
                                                           client_data),
            "callback tracing service failed to configure");

        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(cb_data->client_ctx, &valid_ctx),
                         "failure checking context validity");

        EXPECT_EQ(valid_ctx, 1);

        ROCPROFILER_CALL(rocprofiler_start_context(cb_data->client_ctx),
                         "rocprofiler context start failed");

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        ROCPROFILER_CALL(rocprofiler_stop_context(cb_data->client_ctx),
                         "rocprofiler context stop failed");

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto& cb_data = get_client_callback_data();

    static auto cfg_result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            tool_init,
                                            tool_fini,
                                            static_cast<void*>(&cb_data)};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t                 version,
           const char*              runtime_version,
           uint32_t                 prio,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        auto expected_version = ROCPROFILER_VERSION;
        EXPECT_EQ(expected_version, version);
        EXPECT_EQ(std::string_view{runtime_version}, std::string_view{ROCPROFILER_VERSION_STRING});
        EXPECT_EQ(prio, 0);
        EXPECT_EQ(client_id->name, nullptr);
        cb_data.client_id       = client_id;
        cb_data.client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

        ROCPROFILER_CALL_EXPECT(
            rocprofiler_at_runtime_api_registration(api_registration_callback,
                                                    ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY |
                                                        ROCPROFILER_HIP_LIBRARY |
                                                        ROCPROFILER_MARKER_LIBRARY,
                                                    static_cast<void*>(&cb_data)),
            "function should return invalid argument if ROCPROFILER_LIBRARY included",
            ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

        using init_list_t = std::initializer_list<int>;
        for(auto itr : init_list_t{
                (ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY | ROCPROFILER_MARKER_LIBRARY),
                (ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY),
                (ROCPROFILER_HIP_LIBRARY),
                (ROCPROFILER_HSA_LIBRARY | ROCPROFILER_MARKER_LIBRARY),
                (ROCPROFILER_MARKER_LIBRARY)})
        {
            ROCPROFILER_CALL_EXPECT(
                rocprofiler_at_runtime_api_registration(
                    api_registration_callback, itr, static_cast<void*>(&cb_data)),
                "test should be updated if new (non-HSA) intercept table is supported",
                ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);
        }

        ROCPROFILER_CALL_EXPECT(rocprofiler_at_runtime_api_registration(
                                    api_registration_callback,
                                    ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY,
                                    static_cast<void*>(&cb_data)),
                                "test should be updated if HIP intercept table is supported",
                                ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);

        ROCPROFILER_CALL_EXPECT(
            rocprofiler_at_runtime_api_registration(
                api_registration_callback, ROCPROFILER_HIP_LIBRARY, static_cast<void*>(&cb_data)),
            "test should be updated if HIP intercept table is supported",
            ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);

        ROCPROFILER_CALL_EXPECT(rocprofiler_at_runtime_api_registration(
                                    api_registration_callback,
                                    ROCPROFILER_HSA_LIBRARY | ROCPROFILER_MARKER_LIBRARY,
                                    static_cast<void*>(&cb_data)),
                                "test should be updated if ROCTx intercept table is supported",
                                ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);

        ROCPROFILER_CALL(
            rocprofiler_at_runtime_api_registration(
                api_registration_callback, ROCPROFILER_HSA_LIBRARY, static_cast<void*>(&cb_data)),
            "HSA API intercept table registration failed");

        return &cfg_result;
    };

    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);

    hsa_iterate_agents_cb_t agent_cb = [](hsa_agent_t agent, void* data) {
        static_cast<agent_data*>(data)->agent_count++;

        auto status     = HSA_STATUS_SUCCESS;
        auto agent_type = hsa_device_type_t{};
        if((status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_type)) ==
           HSA_STATUS_SUCCESS)
            static_cast<agent_data*>(data)->agents.emplace_back(agent_type);

        return status;
    };

    hsa_init();
    auto         _agent_data = agent_data{};
    hsa_status_t itr_status  = hsa_iterate_agents(agent_cb, static_cast<void*>(&_agent_data));
    hsa_shut_down();

    EXPECT_EQ(itr_status, HSA_STATUS_SUCCESS);
    EXPECT_GT(_agent_data.agent_count, 0);
    EXPECT_EQ(_agent_data.agent_count, _agent_data.agents.size());

    ASSERT_NE(cb_data.client_id, nullptr);
    ASSERT_NE(cb_data.client_fini_func, nullptr);

    cb_data.client_fini_func(*cb_data.client_id);

    EXPECT_EQ(cb_data.client_workflow_count, 3);

    for(auto itr : cb_data.client_callback_count)
    {
        EXPECT_EQ(itr.second.first, itr.second.second)
            << "mismatched wrap counts for " << itr.first
            << " (lhs=tool_wrapper, rhs=rocprofiler_wrapper)";
        if(itr.first != "hsa_init")
        {
            EXPECT_GT(itr.second.first, 0) << itr.first << " not wrapped";
        }
        else
        {
            EXPECT_EQ(itr.second.first, 0) << itr.first
                                           << " was wrapped. If hsa runtime has been updated to "
                                              "include first call to hsa_init, update this test";
        }
    }

    auto get_count = [](std::string_view func_name) {
        // we already checked that first == second so we can just check first here
        return cb_data.client_callback_count.at(func_name).first;
    };

    EXPECT_EQ(get_count("hsa_init"), 0);
    EXPECT_EQ(get_count("hsa_iterate_agents"), 1);
    EXPECT_EQ(get_count("hsa_agent_get_info"), _agent_data.agent_count);
    EXPECT_EQ(get_count("hsa_shut_down"), 1);
}

TEST(rocprofiler_lib, intercept_table_and_callback_tracing_disable_context)
{
    // test layering of tool interception of API table on top of rocprofiler API tracing.
    // Similar to intercept_table_and_callback_tracing test except on the
    // first call to hsa_iterate_agents the context is disabled. As a result,
    // one should only see the rocprofiler API tracing for hsa_iterate_agents
    // and not for hsa_agent_get_info or hsa_shut_down.

    using init_func_t             = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t             = void (*)(void*);
    using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    get_preincrement()  = 1;
    get_postincrement() = 1;

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        auto operations = std::vector<uint32_t>{ROCPROFILER_HSA_API_ID_hsa_init,
                                                ROCPROFILER_HSA_API_ID_hsa_iterate_agents,
                                                ROCPROFILER_HSA_API_ID_hsa_agent_get_info,
                                                ROCPROFILER_HSA_API_ID_hsa_shut_down};

        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(cb_data->client_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_HSA_API,
                                                           operations.data(),
                                                           operations.size(),
                                                           tool_tracing_callback,
                                                           client_data),
            "callback tracing service failed to configure");

        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(cb_data->client_ctx, &valid_ctx),
                         "failure checking context validity");

        EXPECT_EQ(valid_ctx, 1);

        ROCPROFILER_CALL(rocprofiler_start_context(cb_data->client_ctx),
                         "rocprofiler context start failed");

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        ROCPROFILER_CALL_EXPECT(rocprofiler_stop_context(cb_data->client_ctx),
                                "rocprofiler context stop",
                                ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);

        ROCPROFILER_CALL_EXPECT(rocprofiler_start_context(cb_data->client_ctx),
                                "rocprofiler context start",
                                ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED);

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto& cb_data = get_client_callback_data();
    cb_data              = callback_data{};

    static auto cfg_result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            tool_init,
                                            tool_fini,
                                            static_cast<void*>(&cb_data)};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t                 version,
           const char*              runtime_version,
           uint32_t                 prio,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        auto expected_version = ROCPROFILER_VERSION;
        EXPECT_EQ(expected_version, version);
        EXPECT_EQ(std::string_view{runtime_version}, std::string_view{ROCPROFILER_VERSION_STRING});
        EXPECT_EQ(prio, 0);
        EXPECT_EQ(client_id->name, nullptr);
        cb_data.client_id       = client_id;
        cb_data.client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

        ROCPROFILER_CALL_EXPECT(
            rocprofiler_at_runtime_api_registration(api_registration_callback,
                                                    ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY |
                                                        ROCPROFILER_HIP_LIBRARY |
                                                        ROCPROFILER_MARKER_LIBRARY,
                                                    static_cast<void*>(&cb_data)),
            "function should return invalid argument if ROCPROFILER_LIBRARY included",
            ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

        using init_list_t = std::initializer_list<int>;
        for(auto itr : init_list_t{
                (ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY | ROCPROFILER_MARKER_LIBRARY),
                (ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY),
                (ROCPROFILER_HIP_LIBRARY),
                (ROCPROFILER_HSA_LIBRARY | ROCPROFILER_MARKER_LIBRARY),
                (ROCPROFILER_MARKER_LIBRARY)})
        {
            ROCPROFILER_CALL_EXPECT(
                rocprofiler_at_runtime_api_registration(
                    api_registration_callback, itr, static_cast<void*>(&cb_data)),
                "test should be updated if new (non-HSA) intercept table is supported",
                ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);
        }

        ROCPROFILER_CALL_EXPECT(rocprofiler_at_runtime_api_registration(
                                    api_registration_callback,
                                    ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY,
                                    static_cast<void*>(&cb_data)),
                                "test should be updated if HIP intercept table is supported",
                                ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);

        ROCPROFILER_CALL_EXPECT(
            rocprofiler_at_runtime_api_registration(
                api_registration_callback, ROCPROFILER_HIP_LIBRARY, static_cast<void*>(&cb_data)),
            "test should be updated if HIP intercept table is supported",
            ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);

        ROCPROFILER_CALL_EXPECT(rocprofiler_at_runtime_api_registration(
                                    api_registration_callback,
                                    ROCPROFILER_HSA_LIBRARY | ROCPROFILER_MARKER_LIBRARY,
                                    static_cast<void*>(&cb_data)),
                                "test should be updated if ROCTx intercept table is supported",
                                ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED);

        ROCPROFILER_CALL(
            rocprofiler_at_runtime_api_registration(
                api_registration_callback, ROCPROFILER_HSA_LIBRARY, static_cast<void*>(&cb_data)),
            "HSA API intercept table registration failed");

        return &cfg_result;
    };

    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);

    hsa_iterate_agents_cb_t agent_cb = [](hsa_agent_t agent, void* data) {
        auto* _data = static_cast<std::pair<agent_data*, callback_data*>*>(data);
        _data->first->agent_count++;

        if(int _is_active = 0;
           rocprofiler_context_is_active(_data->second->client_ctx, &_is_active) ==
               ROCPROFILER_STATUS_SUCCESS &&
           _is_active != 0)
        {
            ROCPROFILER_CALL(rocprofiler_stop_context(_data->second->client_ctx),
                             "rocprofiler context stop failed");
        }

        auto status     = HSA_STATUS_SUCCESS;
        auto agent_type = hsa_device_type_t{};
        if((status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_type)) ==
           HSA_STATUS_SUCCESS)
            _data->first->agents.emplace_back(agent_type);

        return status;
    };

    hsa_init();
    auto         _agent_data = agent_data{};
    auto         _pair_data  = std::make_pair(&_agent_data, &cb_data);
    hsa_status_t itr_status  = hsa_iterate_agents(agent_cb, static_cast<void*>(&_pair_data));
    hsa_shut_down();

    EXPECT_EQ(itr_status, HSA_STATUS_SUCCESS);
    EXPECT_GT(_agent_data.agent_count, 0);
    EXPECT_EQ(_agent_data.agent_count, _agent_data.agents.size());

    ASSERT_NE(cb_data.client_id, nullptr);
    ASSERT_NE(cb_data.client_fini_func, nullptr);

    cb_data.client_fini_func(*cb_data.client_id);

    EXPECT_EQ(cb_data.client_workflow_count, 3);

    auto get_tool_count = [](std::string_view func_name) {
        // we already checked that first == second so we can just check first here
        return cb_data.client_callback_count.at(func_name).first;
    };

    auto get_rocp_count = [](std::string_view func_name) {
        // we already checked that first == second so we can just check first here
        return cb_data.client_callback_count.at(func_name).second;
    };

    EXPECT_EQ(get_tool_count("hsa_init"), 0);
    EXPECT_EQ(get_tool_count("hsa_iterate_agents"), 2);
    EXPECT_EQ(get_tool_count("hsa_agent_get_info"), 2 * _agent_data.agent_count);
    EXPECT_EQ(get_tool_count("hsa_shut_down"), 2);

    EXPECT_EQ(get_rocp_count("hsa_init"), 0);
    EXPECT_EQ(get_rocp_count("hsa_iterate_agents"), 2)
        << "if equal to 1, then ENTER phase was invoked but EXIT phase was not (incorrect)";
    EXPECT_EQ(get_rocp_count("hsa_agent_get_info"), 0);
    EXPECT_EQ(get_rocp_count("hsa_shut_down"), 0);
}
