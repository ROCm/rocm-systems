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
#include "library.hpp"
#include "library/components/rocm_smi.hpp"
#include "library/components/rocprofiler.hpp"
#include "library/components/roctracer.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/rocprofiler.hpp"
#include "library/rocprofiler/hsa_rsrc_factory.hpp"
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

#include <rocprofiler.h>

using namespace omnitrace;

namespace omnitrace
{
namespace rocm
{
std::mutex rocm_mutex = {};
bool       is_loaded  = false;
}  // namespace rocm
}  // namespace omnitrace

// HSA-runtime tool on-load method
extern "C"
{
#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
    void OnUnloadTool()
    {
        OMNITRACE_BASIC_VERBOSE(2, "Inside %s\n", __FUNCTION__);

        rocm::lock_t _lk{ rocm::rocm_mutex, std::defer_lock };
        if(!_lk.owns_lock()) _lk.lock();

        if(!rocm::is_loaded) return;
        rocm::is_loaded = false;

        _lk.unlock();

        // stop_top_level_timer_if_necessary();
        // Final resources cleanup
        omnitrace::rocprofiler::rocm_cleanup();
    }

    void OnLoadToolProp(rocprofiler_settings_t* settings)
    {
        OMNITRACE_BASIC_VERBOSE(2, "Inside %s\n", __FUNCTION__);

        rocm::lock_t _lk{ rocm::rocm_mutex, std::defer_lock };
        if(!_lk.owns_lock()) _lk.lock();

        if(rocm::is_loaded) return;
        rocm::is_loaded = true;

        _lk.unlock();

        // Enable timestamping
        settings->timestamp_on = 1u;

        // Initialize profiling
        omnitrace::rocprofiler::rocm_initialize();
        HsaRsrcFactory::Instance().PrintGpuAgents("ROCm");
    }
#endif

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names)
    {
        OMNITRACE_BASIC_VERBOSE(2, "Inside %s\n", __FUNCTION__);

        if(!tim::get_env("OMNITRACE_INIT_TOOLING", true)) return true;
        if(!tim::settings::enabled()) return true;

        roctracer_is_init() = true;
        pthread_gotcha::push_enable_sampling_on_child_threads(false);
        OMNITRACE_BASIC_VERBOSE_F(1, "\n");

        tim::consume_parameters(table, runtime_version, failed_tool_count,
                                failed_tool_names);

        if(!config::settings_are_configured() && get_state() < State::Active)
            omnitrace_init_tooling_hidden();

        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);

        static auto _setup = [=]() {
            try
            {
                OMNITRACE_VERBOSE(1, "[OnLoad] setting up HSA...\n");

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

                    OMNITRACE_VERBOSE(1, "    HSA-trace(");
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

                            OMNITRACE_VERBOSE(1, " %s", api);
                        }
                    }
                    else
                    {
                        ROCTRACER_CALL(roctracer_enable_domain_callback(
                            ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, nullptr));
                    }
                    OMNITRACE_VERBOSE(1, " )\n");
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

                    OMNITRACE_VERBOSE(1, "    HSA-activity-trace()\n");
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

        (void) omnitrace::get_clock_skew();

        comp::roctracer::add_setup("hsa", _setup);
        comp::roctracer::add_shutdown("hsa", _shutdown);

        rocm_smi::set_state(State::Active);
        comp::roctracer::setup();

#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
        bool _force_rocprofiler_init =
            tim::get_env("OMNITRACE_FORCE_ROCPROFILE_INIT", false, false);
#else
        bool _force_rocprofiler_init = false;
#endif

        bool _success = true;
        bool _is_empty =
            (config::settings_are_configured() && config::get_rocm_events().empty());
        if(_force_rocprofiler_init || (get_use_rocprofiler() && !_is_empty))
        {
            auto _rocprof =
                dynamic_library{ "OMNITRACE_ROCPROFILER_LIBRARY", "librocprofiler64.so",
                                 (RTLD_LAZY | RTLD_GLOBAL), true };

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
        pthread_gotcha::pop_enable_sampling_on_child_threads();
        return _success;
    }

    // HSA-runtime on-unload method
    void OnUnload()
    {
        OMNITRACE_BASIC_VERBOSE(2, "Inside %s\n", __FUNCTION__);
        rocm_smi::set_state(State::Finalized);
        comp::roctracer::shutdown();
        comp::rocprofiler::shutdown();
        omnitrace_finalize_hidden();
    }
}
