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

#include "library/components/roctracer.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/components/rocm_smi.hpp"
#include "library/components/roctracer_callbacks.hpp"
#include "library/config.hpp"
#include "library/defines.hpp"
#include "library/redirect.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"

namespace rocm_smi = omnitrace::rocm_smi;
using namespace omnitrace;

namespace tim
{
namespace component
{
void
roctracer::preinit()
{
    roctracer_data::label()       = "roctracer";
    roctracer_data::description() = "ROCm tracer (activity API)";
}

void
roctracer::start()
{
    if(tracker_type::start() == 0) setup();
}

void
roctracer::stop()
{
    if(tracker_type::stop() == 0) shutdown();
}

bool
roctracer::is_setup()
{
    return roctracer_is_setup();
}

void
roctracer::add_setup(const std::string& _lbl, std::function<void()>&& _func)
{
    roctracer_setup_routines().emplace_back(_lbl, std::move(_func));
}

void
roctracer::add_shutdown(const std::string& _lbl, std::function<void()>&& _func)
{
    roctracer_shutdown_routines().emplace_back(_lbl, std::move(_func));
}

void
roctracer::remove_setup(const std::string& _lbl)
{
    auto& _data = roctracer_setup_routines();
    for(auto itr = _data.begin(); itr != _data.end(); ++itr)
    {
        if(itr->first == _lbl)
        {
            _data.erase(itr);
            break;
        }
    }
}

void
roctracer::remove_shutdown(const std::string& _lbl)
{
    auto& _data = roctracer_setup_routines();
    for(auto itr = _data.begin(); itr != _data.end(); ++itr)
    {
        if(itr->first == _lbl)
        {
            _data.erase(itr);
            break;
        }
    }
}

void
roctracer::setup()
{
    trait::runtime_enabled<roctracer_data>::set(get_use_timemory());

    if(!get_use_roctracer()) return;

    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(roctracer_is_setup()) return;
    roctracer_is_setup() = true;

    OMNITRACE_DEBUG("[%s]\n", __FUNCTION__);

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR < 4
    auto _kfdwrapper = dynamic_library{ "OMNITRACE_ROCTRACER_LIBKFDWRAPPER",
                                        OMNITRACE_ROCTRACER_LIBKFDWRAPPER };
#endif

    ROCTRACER_CALL(
        roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, (void*) hip_api_callback));

    if(roctracer_default_pool() == nullptr)
    {
        // Allocating tracing pool
        roctracer_properties_t properties{};
        memset(&properties, 0, sizeof(roctracer_properties_t));
        // properties.mode                = 0x1000;
        properties.buffer_size         = 0x1000;
        properties.buffer_callback_fun = hip_activity_callback;
        ROCTRACER_CALL(roctracer_open_pool(&properties));
    }

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR >= 4 &&              \
    OMNITRACE_HIP_VERSION_MINOR <= 5
    // HIP 4.5.0 has an invalid warning
    redirect _rd{ std::cerr, "roctracer_enable_callback(), get_op_end(), invalid domain "
                             "ID(4)  in: roctracer_enable_callback(hip_api_callback, "
                             "nullptr)roctracer_enable_activity_expl(), get_op_end(), "
                             "invalid domain ID(4)  in: roctracer_enable_activity()" };
#endif

    // Enable API callbacks, all domains
    ROCTRACER_CALL(roctracer_enable_callback(hip_api_callback, nullptr));
    // Enable activity tracing, all domains
    ROCTRACER_CALL(roctracer_enable_activity());

    // callback for HSA
    for(auto& itr : roctracer_setup_routines())
        itr.second();
}

void
roctracer::shutdown()
{
    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(!roctracer_is_setup()) return;
    roctracer_is_setup() = false;
    OMNITRACE_DEBUG("[%s]\n", __FUNCTION__);

    // flush all the activity
    if(roctracer_default_pool() != nullptr)
    {
        OMNITRACE_DEBUG("[%s] roctracer_flush_activity\n", __FUNCTION__);
        ROCTRACER_CALL(roctracer_flush_activity());

        // flush all buffers
        OMNITRACE_DEBUG("[%s] roctracer_flush_buf\n", __FUNCTION__);
        roctracer_flush_buf();
    }

    OMNITRACE_DEBUG("[%s] executing hip_exec_activity_callbacks\n", __FUNCTION__);
    // make sure all async operations are executed
    for(size_t i = 0; i < max_supported_threads; ++i)
        hip_exec_activity_callbacks(i);

    // callback for hsa
    OMNITRACE_DEBUG("[%s] executing roctracer_shutdown_routines...\n", __FUNCTION__);
    for(auto& itr : roctracer_shutdown_routines())
        itr.second();

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR >= 4 &&              \
    OMNITRACE_HIP_VERSION_MINOR <= 5
    OMNITRACE_DEBUG("[%s] redirecting roctracer warnings\n", __FUNCTION__);
    // HIP 4.5.0 has an invalid warning
    redirect _rd{
        std::cerr, "roctracer_disable_callback(), get_op_end(), invalid domain ID(4)  "
                   "in: roctracer_disable_callback()roctracer_disable_activity(), "
                   "get_op_end(), invalid domain ID(4)  in: roctracer_disable_activity()"
    };
#endif

    // Disable tracing and closing the pool
    OMNITRACE_DEBUG("[%s] roctracer_disable_callback\n", __FUNCTION__);
    ROCTRACER_CALL(roctracer_disable_callback());

    OMNITRACE_DEBUG("[%s] roctracer_disable_activity\n", __FUNCTION__);
    ROCTRACER_CALL(roctracer_disable_activity());

    OMNITRACE_DEBUG("[%s] roctracer_close_pool\n", __FUNCTION__);
    ROCTRACER_CALL(roctracer_close_pool());

    OMNITRACE_DEBUG("[%s] roctracer is shutdown\n", __FUNCTION__);
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer, false, void)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer_data, true, double)

// HSA-runtime tool on-load method
extern "C"
{
    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names) OMNITRACE_VISIBILITY("default");
    void OnUnload() OMNITRACE_VISIBILITY("default");

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names)
    {
        pthread_gotcha::enable_sampling_on_child_threads() = false;
        OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug_env() || get_verbose_env() > 0,
                                          "[%s]\n", __FUNCTION__);
        tim::consume_parameters(table, runtime_version, failed_tool_count,
                                failed_tool_names);

        auto _setup = [=]() {
            try
            {
                OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                  "[%s] setting up HSA...\n",
                                                  __FUNCTION__);

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

                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                      "    HSA-trace(");
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

                            OMNITRACE_CONDITIONAL_BASIC_PRINT(
                                get_debug() || get_verbose() > 1, " %s", api);
                        }
                    }
                    else
                    {
                        ROCTRACER_CALL(roctracer_enable_domain_callback(
                            ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, nullptr));
                    }
                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                      "\n");
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

                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
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

        auto _shutdown = []() {
            OMNITRACE_DEBUG("[%s] roctracer_disable_domain_callback\n", __FUNCTION__);
            ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));

            OMNITRACE_DEBUG("[%s] roctracer_disable_op_activity\n", __FUNCTION__);
            ROCTRACER_CALL(
                roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
        };

        comp::roctracer::add_setup("hsa", std::move(_setup));
        comp::roctracer::add_shutdown("hsa", std::move(_shutdown));

        rocm_smi::set_state(State::Active);
        comp::roctracer::setup();

        pthread_gotcha::enable_sampling_on_child_threads() = true;
        return true;
    }

    // HSA-runtime on-unload method
    void OnUnload()
    {
        OMNITRACE_DEBUG("[%s]\n", __FUNCTION__);
        rocm_smi::set_state(State::Finalized);
        comp::roctracer::shutdown();
    }
}
