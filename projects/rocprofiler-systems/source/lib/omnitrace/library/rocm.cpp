// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/rocm.hpp"
#include "library/components/rocprofiler.hpp"
#include "library/components/roctracer.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/dynamic_library.hpp"
#include "library/gpu.hpp"
#include "library/rocm/hsa_rsrc_factory.hpp"
#include "library/rocm_smi.hpp"
#include "library/rocprofiler.hpp"
#include "library/roctracer.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/cpu.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/utility/types.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <tuple>

#define HIP_PROF_HIP_API_STRING 1

#include <roctracer_ext.h>
#include <roctracer_hcc.h>
#include <roctracer_hip.h>

#define AMD_INTERNAL_BUILD 1
#include <roctracer_hsa.h>

#if __has_include(<hip/amd_detail/hip_prof_str.h>) || (defined(OMNITRACE_USE_HIP) && OMNITRACE_USE_HIP > 0)
#    include <hip/amd_detail/hip_prof_str.h>
#    define OMNITRACE_HIP_API_ARGS 1
#else
#    define OMNITRACE_HIP_API_ARGS 0
#endif

#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
#    include <rocprofiler.h>
#endif

using namespace omnitrace;

namespace omnitrace
{
namespace rocm
{
std::mutex rocm_mutex    = {};
bool       is_loaded     = false;
bool       on_load_trace = (get_env<int>("ROCP_ONLOAD_TRACE", 0) > 0);
}  // namespace rocm
}  // namespace omnitrace

std::ostream&
operator<<(std::ostream& _os, const rocprofiler_settings_t& _v)
{
#define ROCPROF_SETTING_FIELD_STR(NAME) JOIN('=', #NAME, _v.NAME)

    _os << JOIN(
        ", ", ROCPROF_SETTING_FIELD_STR(intercept_mode),
        ROCPROF_SETTING_FIELD_STR(code_obj_tracking),
        ROCPROF_SETTING_FIELD_STR(memcopy_tracking),
        ROCPROF_SETTING_FIELD_STR(trace_size), ROCPROF_SETTING_FIELD_STR(trace_local),
        ROCPROF_SETTING_FIELD_STR(timeout), ROCPROF_SETTING_FIELD_STR(timestamp_on),
        ROCPROF_SETTING_FIELD_STR(hsa_intercepting),
        ROCPROF_SETTING_FIELD_STR(k_concurrent), ROCPROF_SETTING_FIELD_STR(opt_mode),
        ROCPROF_SETTING_FIELD_STR(obj_dumping));
    return _os;
}

// HSA-runtime tool on-load method
extern "C"
{
#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
    void OnUnloadTool()
    {
        OMNITRACE_BASIC_VERBOSE_F(2 || rocm::on_load_trace, "Unloading...\n");

        rocm::lock_t _lk{ rocm::rocm_mutex, std::defer_lock };
        if(!_lk.owns_lock()) _lk.lock();

        if(!rocm::is_loaded)
        {
            OMNITRACE_BASIC_VERBOSE_F(1 || rocm::on_load_trace,
                                      "rocprofiler is not loaded\n");
            return;
        }
        rocm::is_loaded = false;

        _lk.unlock();

        // stop_top_level_timer_if_necessary();
        // Final resources cleanup
        omnitrace::rocprofiler::rocm_cleanup();
    }

    void OnLoadToolProp(rocprofiler_settings_t* settings)
    {
        OMNITRACE_BASIC_VERBOSE_F(2 || rocm::on_load_trace, "Loading...\n");

        rocm::lock_t _lk{ rocm::rocm_mutex, std::defer_lock };
        if(!_lk.owns_lock()) _lk.lock();

        if(rocm::is_loaded)
        {
            OMNITRACE_BASIC_VERBOSE_F(1 || rocm::on_load_trace,
                                      "rocprofiler is already loaded\n");
            return;
        }
        rocm::is_loaded = true;

        _lk.unlock();

        // Enable timestamping
        settings->timestamp_on     = 1;
        settings->intercept_mode   = 1;
        settings->hsa_intercepting = 1;
        settings->k_concurrent     = 0;
        settings->obj_dumping      = 0;
        // settings->code_obj_tracking = 0;
        // settings->memcopy_tracking  = 0;
        // settings->trace_local       = 1;
        // settings->opt_mode          = 1;
        // settings->trace_size        = 0;
        // settings->timeout           = 0;

        OMNITRACE_BASIC_VERBOSE_F(1 || rocm::on_load_trace, "rocprofiler settings: %s\n",
                                  JOIN("", *settings).c_str());

        // Initialize profiling
        omnitrace::rocprofiler::rocm_initialize();
        HsaRsrcFactory::Instance().PrintGpuAgents("ROCm");
    }
#endif

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names)
    {
        tim::consume_parameters(table, runtime_version, failed_tool_count,
                                failed_tool_names);

        OMNITRACE_BASIC_VERBOSE_F(2 || rocm::on_load_trace, "Loading...\n");
        OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

        if(!tim::get_env("OMNITRACE_INIT_TOOLING", true)) return true;
        if(!tim::settings::enabled()) return true;

        roctracer_is_init() = true;
        OMNITRACE_BASIC_VERBOSE_F(1 || rocm::on_load_trace, "Loading ROCm tooling...\n");

        if(!config::settings_are_configured() && get_state() < State::Active)
            omnitrace_init_tooling_hidden();

        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

        static auto _setup = [=]() {
            try
            {
                OMNITRACE_VERBOSE(1 || rocm::on_load_trace,
                                  "[OnLoad] setting up HSA...\n");

                // const char* output_prefix = getenv("ROCP_OUTPUT_DIR");
                const char* output_prefix = nullptr;

                bool trace_hsa_api = get_trace_hsa_api();

                // Enable HSA API callbacks/activity
                if(trace_hsa_api)
                {
                    std::vector<std::string> hsa_api_vec =
                        tim::delimit(get_trace_hsa_api_types());

                    // initialize HSA tracing
                    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*) table);

                    if(!hsa_api_vec.empty())
                    {
                        for(const auto& itr : hsa_api_vec)
                        {
                            uint32_t    cid = HSA_API_ID_NUMBER;
                            const char* api = itr.c_str();
                            ROCTRACER_CALL(roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api,
                                                             &cid, nullptr));
                            ROCTRACER_CALL(roctracer_enable_op_callback(
                                ACTIVITY_DOMAIN_HSA_API, cid, hsa_api_callback, nullptr));

                            OMNITRACE_VERBOSE(1 || rocm::on_load_trace,
                                              "    HSA-trace(%s)", api);
                        }
                    }
                    else
                    {
                        OMNITRACE_VERBOSE(1 || rocm::on_load_trace, "    HSA-trace()\n");
                        ROCTRACER_CALL(roctracer_enable_domain_callback(
                            ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, nullptr));
                    }
                }

                bool trace_hsa_activity = get_trace_hsa_activity();
                // Enable HSA GPU activity
                if(trace_hsa_activity)
                {
                    // initialize HSA tracing
                    ::roctracer::hsa_ops_properties_t ops_properties{
                        table,
                        reinterpret_cast<activity_async_callback_t>(
                            hsa_activity_callback),
                        nullptr, output_prefix
                    };
                    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);

                    OMNITRACE_VERBOSE(1 || rocm::on_load_trace,
                                      "    HSA-activity-trace()\n");
                    ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS,
                                                                HSA_OP_ID_COPY));
                }
            } catch(std::exception& _e)
            {
                OMNITRACE_BASIC_PRINT("Exception was thrown in HSA setup: %s\n",
                                      _e.what());
            }
        };

        static auto _shutdown = []() {
            OMNITRACE_DEBUG_F("roctracer_disable_domain_callback\n");
            ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));

            OMNITRACE_DEBUG_F("roctracer_disable_op_activity\n");
            ROCTRACER_CALL(
                roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
        };

        OMNITRACE_VERBOSE_F(1 || rocm::on_load_trace,
                            "Computing the roctracer clock skew...\n");
        (void) omnitrace::get_clock_skew();

        comp::roctracer::add_setup("hsa", _setup);
        comp::roctracer::add_shutdown("hsa", _shutdown);

        OMNITRACE_VERBOSE_F(1 || rocm::on_load_trace,
                            "Setting rocm_smi state to active...\n");
        rocm_smi::set_state(State::Active);

        OMNITRACE_VERBOSE_F(1 || rocm::on_load_trace,
                            "Requesting roctracer to setup...\n");
        comp::roctracer::setup();

#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
        bool _force_rocprofiler_init =
            tim::get_env("OMNITRACE_FORCE_ROCPROFILER_INIT", false, false);
#else
        bool _force_rocprofiler_init = false;
#endif

        bool _success = true;
        bool _is_empty =
            (config::settings_are_configured() && config::get_rocm_events().empty());
        if(_force_rocprofiler_init || (get_use_rocprofiler() && !_is_empty))
        {
            auto _rocprof =
                dynamic_library{ "OMNITRACE_ROCPROFILER_LIBRARY",
                                 find_library_path("librocprofiler64.so",
                                                   { "OMNITRACE_ROCM_PATH", "ROCM_PATH" },
                                                   { OMNITRACE_DEFAULT_ROCM_PATH },
                                                   { "lib", "lib64", "rocprofiler/lib",
                                                     "rocprofiler/lib64" }),
                                 (RTLD_LAZY | RTLD_GLOBAL), false };

            OMNITRACE_VERBOSE_F(1 || rocm::on_load_trace,
                                "Loading rocprofiler library (%s=%s)...\n",
                                _rocprof.envname.c_str(), _rocprof.filename.c_str());
            _rocprof.open();

            on_load_t _rocprof_load = nullptr;
            _success = _rocprof.invoke("OnLoad", _rocprof_load, table, runtime_version,
                                       failed_tool_count, failed_tool_names);
            OMNITRACE_CONDITIONAL_PRINT_F(!_success,
                                          "Warning! Invoking rocprofiler's OnLoad "
                                          "failed! OMNITRACE_ROCPROFILER_LIBRARY=%s\n",
                                          _rocprof.filename.c_str());
            OMNITRACE_CI_THROW(!_success,
                               "Warning! Invoking rocprofiler's OnLoad "
                               "failed! OMNITRACE_ROCPROFILER_LIBRARY=%s\n",
                               _rocprof.filename.c_str());
        }
        else
        {
            HsaRsrcFactory::Instance().PrintGpuAgents("ROCm");
        }

        gpu::add_hip_device_metadata();

        OMNITRACE_BASIC_VERBOSE_F(2 || rocm::on_load_trace, "Loading... %s\n",
                                  (_success) ? "Done" : "Failed");
        return _success;
    }

    // HSA-runtime on-unload method
    void OnUnload()
    {
        OMNITRACE_BASIC_VERBOSE_F(2 || rocm::on_load_trace, "Unloading...\n");
        omnitrace_finalize_hidden();
        OMNITRACE_BASIC_VERBOSE_F(2 || rocm::on_load_trace, "Unloading... Done\n");
    }
}
