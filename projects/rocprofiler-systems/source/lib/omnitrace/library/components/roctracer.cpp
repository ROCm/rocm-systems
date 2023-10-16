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
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/dynamic_library.hpp"
#include "core/redirect.hpp"
#include "library/roctracer.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <roctracer.h>

#define HIP_PROF_HIP_API_STRING 1

#include <roctracer_ext.h>
#include <roctracer_hip.h>

#if OMNITRACE_HIP_VERSION < 50300
#    include <roctracer_hcc.h>
#endif

#define AMD_INTERNAL_BUILD 1
#include <roctracer_hsa.h>

namespace omnitrace
{
namespace component
{
namespace
{
auto&
roctracer_activity_count()
{
    static std::atomic<int64_t> _v{ 0 };
    return _v;
}
}  // namespace

void
roctracer::preinit()
{
    roctracer_data::label()       = "roctracer";
    roctracer_data::description() = "ROCm tracer (activity API)";
}

void
roctracer::start()
{
    if(tracker_type::start() == 0) setup(nullptr);
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
roctracer::setup(void* table, bool on_load_trace)
{
    if(!get_use_roctracer()) return;

    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(roctracer_is_setup()) return;
    roctracer_is_setup() = true;

    OMNITRACE_VERBOSE_F(1, "setting up roctracer...\n");
    OMNITRACE_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    dynamic_library _amdhip64{ "OMNITRACE_ROCTRACER_LIBAMDHIP64",
                               find_library_path("libamdhip64.so",
                                                 { "OMNITRACE_ROCM_PATH", "ROCM_PATH" },
                                                 { OMNITRACE_DEFAULT_ROCM_PATH }) };

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR < 4
    dynamic_library _kfdwrapper{
        "OMNITRACE_ROCTRACER_LIBKFDWRAPPER",
        find_library_path("libkfdwrapper64.so", { "OMNITRACE_ROCM_PATH", "ROCM_PATH" },
                          { OMNITRACE_DEFAULT_ROCM_PATH },
                          { "roctracer/lib", "roctracer/lib64", "lib", "lib64" })
    };
#endif

    OMNITRACE_ROCTRACER_CALL(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr));

    // Allocating tracing pool
    roctracer_properties_t properties{};
    memset(&properties, 0, sizeof(roctracer_properties_t));
    // properties.mode                = 0x1000;
    properties.buffer_size         = 0x100;
    properties.buffer_callback_fun = hip_activity_callback;
    OMNITRACE_ROCTRACER_CALL(roctracer_open_pool(&properties));

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR >= 4
    // HIP 4.5.0 has an invalid warning
    redirect _rd{ std::cerr, "roctracer_enable_callback(), get_op_end(), invalid domain "
                             "ID(4)  in: roctracer_enable_callback(hip_api_callback, "
                             "nullptr)roctracer_enable_activity_expl(), get_op_end(), "
                             "invalid domain ID(4)  in: roctracer_enable_activity()" };
#endif

    if(get_trace_hip_api())
    {
        OMNITRACE_ROCTRACER_CALL(roctracer_enable_domain_callback(
            ACTIVITY_DOMAIN_HIP_API, hip_api_callback, nullptr));
    }

    if(get_use_roctx())
    {
        OMNITRACE_ROCTRACER_CALL(roctracer_enable_domain_callback(
            ACTIVITY_DOMAIN_ROCTX, roctx_api_callback, nullptr));
    }

    if(get_trace_hip_activity())
    {
        // Enable HIP activity tracing
        OMNITRACE_ROCTRACER_CALL(
            roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
    }

    if(table != nullptr)
    {
        OMNITRACE_VERBOSE(1 || on_load_trace, "[OnLoad] setting up HSA...\n");

        bool trace_hsa_api = get_trace_hsa_api();

        // Enable HSA API callbacks/activity
        if(trace_hsa_api)
        {
            std::vector<std::string> hsa_api_vec =
                tim::delimit(get_trace_hsa_api_types());

            // initialize HSA tracing
            roctracer_set_properties(
                static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HSA_API), (void*) table);

            if(!hsa_api_vec.empty())
            {
                for(const auto& itr : hsa_api_vec)
                {
                    uint32_t    cid = HSA_API_ID_NUMBER;
                    const char* api = itr.c_str();
                    OMNITRACE_ROCTRACER_CALL(roctracer_op_code(
                        static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HSA_API), api,
                        &cid, nullptr));
                    OMNITRACE_ROCTRACER_CALL(roctracer_enable_op_callback(
                        static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HSA_API), cid,
                        hsa_api_callback, nullptr));

                    OMNITRACE_VERBOSE(1 || on_load_trace, "    HSA-trace(%s)", api);
                }
            }
            else
            {
                OMNITRACE_VERBOSE(1 || on_load_trace, "    HSA-trace()\n");
                OMNITRACE_ROCTRACER_CALL(roctracer_enable_domain_callback(
                    static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HSA_API),
                    hsa_api_callback, nullptr));
            }
        }

        bool trace_hsa_activity = get_trace_hsa_activity();
        // Enable HSA GPU activity
        if(trace_hsa_activity)
        {
#if OMNITRACE_HIP_VERSION < 50300
            using namespace roctracer;
            // initialize HSA tracing
            const char*          output_prefix = nullptr;
            hsa_ops_properties_t ops_properties{
                table, reinterpret_cast<activity_async_callback_t>(hsa_activity_callback),
                nullptr, output_prefix
            };
#elif OMNITRACE_HIP_VERSION < 50301
            hsa_ops_properties_t ops_properties;
            ops_properties.table        = table;
            ops_properties.reserved1[0] = reinterpret_cast<void*>(&hsa_activity_callback);
            ops_properties.reserved1[1] = nullptr;
            ops_properties.reserved1[2] = nullptr;
#else
            hsa_ops_properties_t ops_properties{
                table, reinterpret_cast<void*>(&hsa_activity_callback), nullptr, nullptr
            };
#endif
            roctracer_set_properties(
                static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HSA_OPS), &ops_properties);

            OMNITRACE_VERBOSE(1 || on_load_trace, "    HSA-activity-trace()\n");
            OMNITRACE_ROCTRACER_CALL(roctracer_enable_op_activity(
                static_cast<activity_domain_t>(ACTIVITY_DOMAIN_HSA_OPS), HSA_OP_ID_COPY));
        }
    }

    // callback for HSA
    for(auto& itr : roctracer_setup_routines())
        itr.second();

    // make sure all async callbacks are allocated
    for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
        hip_exec_activity_callbacks(i);

    OMNITRACE_VERBOSE_F(1, "roctracer is setup\n");
}

void
roctracer::shutdown()
{
    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(!roctracer_is_setup())
    {
        if(!roctracer_is_init() && tim::storage<comp::roctracer_data>::instance())
            tim::storage<comp::roctracer_data>::instance()->reset();
        return;
    }
    roctracer_is_setup() = false;

    OMNITRACE_VERBOSE_F(1, "shutting down roctracer...\n");

    OMNITRACE_VERBOSE_F(2, "executing hip_exec_activity_callbacks(0..%zu)\n",
                        thread_info::get_peak_num_threads());
    // make sure all async operations are executed
    for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
        hip_exec_activity_callbacks(i);

    // callback for hsa
    OMNITRACE_VERBOSE_F(2, "executing %zu roctracer_shutdown_routines...\n",
                        roctracer_shutdown_routines().size());
    for(auto& itr : roctracer_shutdown_routines())
        itr.second();

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR >= 4
    OMNITRACE_DEBUG_F("redirecting roctracer warnings\n");
    // HIP 4.5.0 has an invalid warning
    redirect _rd{
        std::cerr, "roctracer_disable_callback(), get_op_end(), invalid domain ID(4)  "
                   "in: roctracer_disable_callback()roctracer_disable_activity(), "
                   "get_op_end(), invalid domain ID(4)  in: roctracer_disable_activity()"
    };
#endif

    if(get_trace_hip_api())
    {
        OMNITRACE_VERBOSE_F(
            2,
            "executing roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API)...\n");
        OMNITRACE_ROCTRACER_CALL(
            roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    }

    if(get_use_roctx())
    {
        OMNITRACE_VERBOSE_F(
            2, "executing roctracer_disable_domain_activity(ACTIVITY_DOMAIN_ROCTX)...\n");
        OMNITRACE_ROCTRACER_CALL(
            roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
    }

    if(get_trace_hip_activity())
    {
        OMNITRACE_VERBOSE_F(
            2,
            "executing roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS)...\n");
        OMNITRACE_ROCTRACER_CALL(
            roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
    }

    if(get_trace_hsa_api())
    {
        OMNITRACE_VERBOSE_F(
            2,
            "executing roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_API)...\n");
        OMNITRACE_ROCTRACER_CALL(
            roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
    }

    if(get_trace_hsa_api())
    {
        OMNITRACE_VERBOSE_F(
            2, "executing roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, "
               "HSA_OP_ID_COPY)...\n");
        OMNITRACE_ROCTRACER_CALL(
            roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
    }

    if(roctracer_activity_count() == 0)
    {
        OMNITRACE_VERBOSE_F(2, "executing roctracer_flush_activity()...\n");
        OMNITRACE_ROCTRACER_CALL(roctracer_flush_activity());
    }
    else
    {
        OMNITRACE_CI_FAIL(true,
                          "roctracer_activity_count() != 0 (== %li). "
                          "roctracer::shutdown() most likely called during abort",
                          roctracer_activity_count().load());
    }

    OMNITRACE_VERBOSE_F(1, "roctracer is shutdown\n");
}

scope::transient_destructor
roctracer::protect_flush_activity()
{
    return scope::transient_destructor([]() { --roctracer_activity_count(); },
                                       []() { ++roctracer_activity_count(); });
}
}  // namespace component
}  // namespace omnitrace

OMNITRACE_INSTANTIATE_EXTERN_COMPONENT(roctracer, false, void)
OMNITRACE_INSTANTIATE_EXTERN_COMPONENT(roctracer_data, true, double)
