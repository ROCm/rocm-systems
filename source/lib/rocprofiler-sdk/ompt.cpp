// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/ompt/ompt.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/ompt/details/format.hpp"  // NOLINT(unused-includes)
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/ompt.h>
#include <rocprofiler-sdk/ompt/api_args.h>
#include <rocprofiler-sdk/ompt/omp-tools.h>

#include <fmt/core.h>
#include <cstdint>
#include <iostream>

namespace rocprofiler
{
namespace ompt
{
namespace
{
ompt_start_tool_result_t*
get_start_tool_result()
{
    static auto*& obj = common::static_object<ompt_start_tool_result_t>::construct();
    return obj;
}

namespace
{
auto                                  init_status            = std::atomic<int>{0};
auto                                  fini_status            = std::atomic<int>{0};
ompt_finalize_tool_t                  tool_finalize          = nullptr;
ompt_set_callback_t                   set_callback           = nullptr;
rocprofiler_ompt_callback_functions_t ompt_callback_table    = {};
ompt_get_thread_data_t                real_get_thread_data   = nullptr;
ompt_get_parallel_info_t              real_get_parallel_info = nullptr;
ompt_get_task_info_t                  real_get_task_info     = nullptr;
ompt_get_target_info_t                real_get_target_info   = nullptr;
}  // namespace

void
set_ompt_callbacks()
{
    // set all the ompt callbacks that might be used

    auto cb = [](const char* cbname, ompt_callback_t* cbf, int cbnum) {
        ompt_set_result_t result = set_callback(static_cast<ompt_callbacks_t>(cbnum), *cbf);
        ROCP_WARNING_IF(result != ompt_set_always)
            << "rocprofiler-sdk OpenMP tools set_callback returned " << result
            << fmt::format(" (set result = {})", result) << " for " << cbname << " (id=" << cbnum
            << ")";
    };
    ompt::update_table(cb);
    ompt::update_callback(ompt_callback_table);
}

// proxies for some entry points that use ompt_data_t *
ompt_data_t*
proxy_get_thread_data()
{
    ompt_data_t* real = real_get_thread_data();
    auto*        ret  = ompt::proxy_data_ptr(real);
    return ret;
}

int
proxy_get_parallel_info(int ancestor_level, ompt_data_t** parallel_data, int* team_size)
{
    ompt_data_t* tdata;
    int          tteam_size;
    int          ret = real_get_parallel_info(ancestor_level, &tdata, &tteam_size);
    if(ret != 2) return ret;
    if(team_size != nullptr) *team_size = tteam_size;
    if(parallel_data != nullptr) *parallel_data = ompt::proxy_data_ptr(tdata);
    return ret;
}

int
proxy_get_task_info(int            ancestor_level,
                    int*           flags,
                    ompt_data_t**  task_data,
                    ompt_frame_t** task_frame,
                    ompt_data_t**  parallel_data,
                    int*           thread_num)
{
    int           tflags, tthread_num;
    ompt_data_t * ttask_data, *tparallel_data;
    ompt_frame_t* ttask_frame;
    int           ret = real_get_task_info(
        ancestor_level, &tflags, &ttask_data, &ttask_frame, &tparallel_data, &tthread_num);
    if(ret != 2) return ret;
    if(flags != nullptr) *flags = tflags;
    if(task_data != nullptr) *task_data = ompt::proxy_data_ptr(ttask_data);
    if(task_frame != nullptr) *task_frame = ttask_frame;
    if(parallel_data != nullptr) *parallel_data = ompt::proxy_data_ptr(tparallel_data);
    if(thread_num != nullptr) *thread_num = tthread_num;
    return ret;
}

int
proxy_get_target_info(uint64_t* device_num, ompt_id_t* target_id, ompt_id_t* host_op_id)
{
    uint64_t  tdevice_num;
    ompt_id_t ttarget_id, thost_op_id;
    int       ret = real_get_target_info(&tdevice_num, &ttarget_id, &thost_op_id);
    if(ret != 1) return ret;
    if(device_num != nullptr) *device_num = tdevice_num;
    if(target_id != nullptr) *target_id = ttarget_id;
    if(host_op_id != nullptr) *host_op_id = thost_op_id;
    return ret;
}

#define SETCB(name)                                                                                \
    ompt_callback_table.ompt_##name = reinterpret_cast<ompt_##name##_t>(lookup("ompt_" #name));    \
    LOG_IF(FATAL, ompt_callback_table.ompt_##name == nullptr)                                      \
        << "rocprofiler-sdk OMPT cannot find "                                                     \
           "ompt_" #name

#define PROXYCB(name)                                                                              \
    SETCB(name);                                                                                   \
    real_##name                     = ompt_callback_table.ompt_##name;                             \
    ompt_callback_table.ompt_##name = proxy_##name

int
initialize(ompt_function_lookup_t lookup, int /*initial_device_num*/, ompt_data_t* /*tool_data*/)
{
    init_status.store(-1);

    tool_finalize = static_cast<ompt_finalize_tool_t>(lookup("ompt_finalize_tool"));
    LOG_IF(FATAL, tool_finalize == nullptr)
        << "rocprofiler-sdk OMPT cannot find ompt_finalize_tool";

    set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    LOG_IF(FATAL, set_callback == nullptr) << "rocprofiler-sdk OMPT cannot find ompt_set_callback";

    SETCB(enumerate_states);
    SETCB(enumerate_mutex_impls);
    PROXYCB(get_thread_data);  // <- need proxy
    SETCB(get_num_places);
    SETCB(get_place_proc_ids);
    SETCB(get_place_num);
    SETCB(get_partition_place_nums);
    SETCB(get_proc_id);
    SETCB(get_state);
    PROXYCB(get_parallel_info);  // <= need proxy
    PROXYCB(get_task_info);      // <= need proxy
    SETCB(get_task_memory);
    SETCB(get_num_devices);
    SETCB(get_num_procs);
    PROXYCB(get_target_info);  // <= need proxy
    SETCB(get_unique_id);

    set_ompt_callbacks();
    init_status.store(1);

    return 1;  // bizarre abberation in the OMPT spec, not 0
}
#undef SETCB

void
finalize(ompt_data_t* /*tool_data*/)
{
    fini_status.store(-1);

    // do whatever for finalization

    fini_status.store(1);
}
}  // namespace

void
finalize_ompt()
{
    // As per OMPT spec, this is no need for the tool to call finalize, it is called by the
    // runtime when the tool is unloaded.
    // However, we can call it here to ensure that the tool is finalized properly.
    // But temporarily disable it to avoid issues with the race condition between the runtime
    // and the tool unloading.
    // if(rocprofiler::ompt::tool_finalize) rocprofiler::ompt::tool_finalize();
}
}  // namespace ompt
}  // namespace rocprofiler

extern "C" {

rocprofiler_status_t
rocprofiler_ompt_is_initialized(int* status)
{
    *status = ::rocprofiler::ompt::init_status.load();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_ompt_is_finalized(int* status)
{
    *status = ::rocprofiler::ompt::fini_status.load();
    return ROCPROFILER_STATUS_SUCCESS;
}

ompt_start_tool_result_t*
rocprofiler_ompt_start_tool(unsigned int /*omp_version*/, const char* /*runtime_version*/)
{
    // log to clog since logging probably won't be initialized here
    auto _init_status = ::rocprofiler::ompt::init_status.load();
    if(_init_status != 0)
    {
        std::clog << "ERROR: rocprofiler-sdk OMPT backend has already been initialized: "
                  << _init_status << '\n';
        return nullptr;
    }

    // don't check contexts here, client tool may not be initialized
    auto* _result = ::rocprofiler::ompt::get_start_tool_result();

    if(_result)
    {
        _result->initialize = ::rocprofiler::ompt::initialize;
        _result->finalize   = ::rocprofiler::ompt::finalize;
    }

    return _result;
}

ompt_start_tool_result_t*
ompt_start_tool(unsigned int omp_version, const char* runtime_version) ROCPROFILER_PUBLIC_API;

ompt_start_tool_result_t*
ompt_start_tool(unsigned int omp_version, const char* runtime_version)
{
    ::rocprofiler::registration::initialize();
    return rocprofiler_ompt_start_tool(omp_version, runtime_version);
}
}
