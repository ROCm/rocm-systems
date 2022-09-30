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

#include "api.hpp"
#include "library/common.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"

#include <timemory/defines.h>

#if defined(OMNITRACE_USE_OMPT) && OMNITRACE_USE_OMPT > 0

#    include "library/components/category_region.hpp"
#    include "library/components/fwd.hpp"

#    include <timemory/components/ompt.hpp>
#    include <timemory/components/ompt/extern.hpp>
#    include <timemory/timemory.hpp>

#    include <memory>

using api_t          = TIMEMORY_API;
using ompt_handle_t  = tim::component::ompt_handle<api_t>;
using ompt_context_t = tim::openmp::context_handler<api_t>;
using ompt_toolset_t = typename ompt_handle_t::toolset_type;
using ompt_bundle_t  = tim::component_tuple<ompt_handle_t>;

extern "C"
{
    ompt_start_tool_result_t* ompt_start_tool(unsigned int,
                                              const char*) OMNITRACE_PUBLIC_API;
}

namespace omnitrace
{
namespace ompt
{
namespace
{
std::unique_ptr<ompt_bundle_t> f_bundle = {};
bool _init_toolset_off = (trait::runtime_enabled<ompt_toolset_t>::set(false),
                          trait::runtime_enabled<ompt_context_t>::set(false), true);
}  // namespace

void
setup()
{
    if(!tim::settings::enabled()) return;
    trait::runtime_enabled<ompt_toolset_t>::set(true);
    trait::runtime_enabled<ompt_context_t>::set(true);
    comp::user_ompt_bundle::global_init();
    comp::user_ompt_bundle::reset();
    tim::auto_lock_t lk{ tim::type_mutex<ompt_handle_t>() };
    comp::user_ompt_bundle::configure<component::local_category_region<category::ompt>>();
    f_bundle = std::make_unique<ompt_bundle_t>("omnitrace/ompt",
                                               quirk::config<quirk::auto_start>{});
}

void
shutdown()
{
    if(f_bundle)
    {
        f_bundle->stop();
        ompt_context_t::cleanup();
        trait::runtime_enabled<ompt_toolset_t>::set(false);
        trait::runtime_enabled<ompt_context_t>::set(false);
        comp::user_ompt_bundle::reset();
    }
    f_bundle.reset();
}
}  // namespace ompt
}  // namespace omnitrace

extern "C" ompt_start_tool_result_t*
ompt_start_tool(unsigned int omp_version, const char* runtime_version)
{
    TIMEMORY_PRINTF(stderr, "OpenMP version: %u, runtime version: %s\n", omp_version,
                    runtime_version);

    OMNITRACE_METADATA("OMP_VERSION", omp_version);
    OMNITRACE_METADATA("OMP_RUNTIME_VERSION", runtime_version);

    if(!omnitrace::settings_are_configured())
    {
        OMNITRACE_BASIC_WARNING(
            0,
            "[%s] invoked before omnitrace was initialized. In instrumentation mode, "
            "settings exported to the environment have not been propagated yet...\n",
            __FUNCTION__);
        omnitrace::configure_settings();
    }

    static bool _use_ompt       = omnitrace::config::get_use_ompt();
    static auto ompt_initialize = [](ompt_function_lookup_t lookup,
                                     int                    initial_device_num,
                                     ompt_data_t*           tool_data) -> int {
        _use_ompt = omnitrace::config::get_use_ompt();
        if(_use_ompt)
        {
            TIMEMORY_PRINTF(stderr, "OpenMP-tools configuring for initial device %i\n\n",
                            initial_device_num);
            tim::ompt::configure<TIMEMORY_OMPT_API_TAG>(lookup, initial_device_num,
                                                        tool_data);
        }
        return 1;  // success
    };

    static auto ompt_finalize = [](ompt_data_t* tool_data) {
        if(_use_ompt)
        {
            TIMEMORY_PRINTF(stderr, "OpenMP-tools finalized\n\n");
            tim::consume_parameters(tool_data);
        }
    };

    static auto data = ompt_start_tool_result_t{ ompt_initialize, ompt_finalize, { 0 } };
    return (ompt_start_tool_result_t*) &data;
}

#else
namespace omnitrace
{
namespace ompt
{
void
setup()
{}

void
shutdown()
{}
}  // namespace ompt
}  // namespace omnitrace

#endif
