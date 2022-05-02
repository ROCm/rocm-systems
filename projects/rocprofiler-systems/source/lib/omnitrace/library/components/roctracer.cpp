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
#include "library/components/roctracer_callbacks.hpp"
#include "library/config.hpp"
#include "library/defines.hpp"
#include "library/redirect.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"

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
    if(!get_use_roctracer()) return;

    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(roctracer_is_setup()) return;
    roctracer_is_setup() = true;

    OMNITRACE_VERBOSE_F(1, "setting up roctracer...\n");

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR < 4
    auto _kfdwrapper = dynamic_library{ "OMNITRACE_ROCTRACER_LIBKFDWRAPPER",
                                        OMNITRACE_ROCTRACER_LIBKFDWRAPPER };
#endif

    ROCTRACER_CALL(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr));

    // Allocating tracing pool
    roctracer_properties_t properties{};
    memset(&properties, 0, sizeof(roctracer_properties_t));
    // properties.mode                = 0x1000;
    properties.buffer_size         = 0x100;
    properties.buffer_callback_fun = hip_activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));

#if OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR >= 4
    // HIP 4.5.0 has an invalid warning
    redirect _rd{ std::cerr, "roctracer_enable_callback(), get_op_end(), invalid domain "
                             "ID(4)  in: roctracer_enable_callback(hip_api_callback, "
                             "nullptr)roctracer_enable_activity_expl(), get_op_end(), "
                             "invalid domain ID(4)  in: roctracer_enable_activity()" };
#endif

    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API,
                                                    hip_api_callback, nullptr));
    // ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX,
    //                                                hip_api_callback, nullptr));
    // Enable HIP activity tracing
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));

    // callback for HSA
    for(auto& itr : roctracer_setup_routines())
        itr.second();

    OMNITRACE_VERBOSE_F(1, "roctracer is setup\n");
}

void
roctracer::shutdown()
{
    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(!roctracer_is_setup()) return;
    roctracer_is_setup() = false;

    OMNITRACE_VERBOSE_F(1, "shutting down roctracer...\n");

    OMNITRACE_DEBUG_F("executing hip_exec_activity_callbacks\n");
    // make sure all async operations are executed
    for(size_t i = 0; i < max_supported_threads; ++i)
        hip_exec_activity_callbacks(i);

    // callback for hsa
    OMNITRACE_DEBUG_F("executing roctracer_shutdown_routines...\n");
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

    // ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
    ROCTRACER_CALL(roctracer_flush_activity());

    OMNITRACE_VERBOSE_F(1, "roctracer is shutdown\n");
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer, false, void)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer_data, true, double)
