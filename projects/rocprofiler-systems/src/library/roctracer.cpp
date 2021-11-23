// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/roctracer.hpp"
#include "library/config.hpp"
#include "library/defines.hpp"
#include "library/roctracer_callbacks.hpp"
#include "library/thread_data.hpp"

namespace tim
{
namespace component
{
void
roctracer::preinit()
{
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
    roctracer_data::label()       = "roctracer";
    roctracer_data::description() = "ROCm tracer (activity API)";
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
roctracer::add_tear_down(const std::string& _lbl, std::function<void()>&& _func)
{
    roctracer_tear_down_routines().emplace_back(_lbl, std::move(_func));
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
roctracer::remove_tear_down(const std::string& _lbl)
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
    if(!get_use_timemory() && !get_use_perfetto()) return;

    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(roctracer_is_setup()) return;
    roctracer_is_setup() = true;
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

    tim::set_env("HSA_TOOLS_LIB", "libhosttrace.so", 0);

    auto _kfdwrapper = dynamic_library{ "HOSTTRACE_ROCTRACER_LIBKFDWRAPPER",
                                        HOSTTRACE_ROCTRACER_LIBKFDWRAPPER };

    ROCTRACER_CALL(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr));

    // if(roctracer_default_pool() == nullptr)
    {
        // Allocating tracing pool
        roctracer_properties_t properties{};
        memset(&properties, 0, sizeof(roctracer_properties_t));
        properties.mode                = 0x1000;
        properties.buffer_size         = 0x1000;
        properties.buffer_callback_fun = hip_activity_callback;
        ROCTRACER_CALL(roctracer_open_pool(&properties));
    }

    // Enable API callbacks, all domains
    ROCTRACER_CALL(roctracer_enable_callback(hip_api_callback, nullptr));
    // Enable activity tracing, all domains
    ROCTRACER_CALL(roctracer_enable_activity());

    // callback for HSA
    for(auto& itr : roctracer_setup_routines())
        itr.second();
}

void
roctracer::tear_down()
{
    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(!roctracer_is_setup()) return;
    roctracer_is_setup() = false;
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

    // flush all the activity
    if(roctracer_default_pool() != nullptr)
    {
        ROCTRACER_CALL(roctracer_flush_activity());
    }
    // flush all buffers
    roctracer_flush_buf();

    // make sure all async operations are executed
    for(size_t i = 0; i < max_supported_threads; ++i)
        hip_exec_activity_callbacks(i);

    // callback for hsa
    for(auto& itr : roctracer_tear_down_routines())
        itr.second();

    // Disable tracing and closing the pool
    ROCTRACER_CALL(roctracer_disable_callback());
    ROCTRACER_CALL(roctracer_disable_activity());
    ROCTRACER_CALL(roctracer_close_pool());
}

void
roctracer::start()
{
    if(tracker_type::start() == 0) setup();
}

void
roctracer::stop()
{
    if(tracker_type::stop() == 0) tear_down();
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer, false, void)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer_data, true, double)

// HSA-runtime tool on-load method
extern "C"
{
    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names) TIMEMORY_VISIBILITY("default");
    void OnUnload() TIMEMORY_VISIBILITY("default");

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names)
    {
        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
        tim::consume_parameters(table, runtime_version, failed_tool_count,
                                failed_tool_names);

        // ONLOAD_TRACE_BEG();
        // on_exit(exit_handler, nullptr);

        auto _setup = [=]() {
            get_hsa_timer() =
                std::make_unique<hsa_timer_t>(table->core_->hsa_system_get_info_fn);

            // const char* output_prefix = getenv("ROCP_OUTPUT_DIR");
            const char* output_prefix = nullptr;

            // App begin timestamp begin_ts_file.txt
            // begin_ts_file_handle = open_output_file(output_prefix,
            // "begin_ts_file.txt"); const timestamp_t app_start_time =
            // timer->timestamp_fn_ns(); fprintf(begin_ts_file_handle, "%lu\n",
            // app_start_time);

            bool                     trace_hsa_api = get_trace_hsa_api();
            std::vector<std::string> hsa_api_vec =
                tim::delimit(get_trace_hsa_api_types());

            // Enable HSA API callbacks/activity
            if(trace_hsa_api)
            {
                // hsa_api_file_handle = open_output_file(output_prefix,
                // "hsa_api_trace.txt");

                // initialize HSA tracing
                roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*) table);

                HOSTTRACE_DEBUG("    HSA-trace(");
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

                        HOSTTRACE_DEBUG(" %s", api);
                    }
                }
                else
                {
                    ROCTRACER_CALL(roctracer_enable_domain_callback(
                        ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, nullptr));
                }
                HOSTTRACE_DEBUG("\n");
            }

            bool trace_hsa_activity = get_trace_hsa_activity();
            // Enable HSA GPU activity
            if(trace_hsa_activity)
            {
                // initialize HSA tracing
                ::roctracer::hsa_ops_properties_t ops_properties{
                    table,
                    reinterpret_cast<activity_async_callback_t>(hsa_activity_callback),
                    nullptr, output_prefix
                };
                roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);

                HOSTTRACE_DEBUG("    HSA-activity-trace()\n");
                ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS,
                                                            HSA_OP_ID_COPY));
            }
        };

        auto _tear_down = []() {
            ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));

            ROCTRACER_CALL(
                roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
        };

        if(comp::roctracer::is_setup()) _setup();

        comp::roctracer::add_setup("hsa", std::move(_setup));
        comp::roctracer::add_tear_down("hsa", std::move(_tear_down));

        return true;
    }

    // HSA-runtime on-unload method
    void OnUnload()
    {
        HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
        // ONLOAD_TRACE("");
    }
}
