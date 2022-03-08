// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "library/defines.hpp"

#include <timemory/defines.h>

#if defined(TIMEMORY_USE_OMPT)

#    include "library/components/fwd.hpp"
#    include "library/components/user_region.hpp"
#    include "library/config.hpp"
#    include "library/debug.hpp"

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
bool _init_toolset_off = (trait::runtime_enabled<ompt_toolset_t>::set(false), true);
}  // namespace

void
setup()
{
    OMNITRACE_VERBOSE(1, "Setting up OMPT...\n");
    trait::runtime_enabled<ompt_toolset_t>::set(config::get_use_ompt());
    comp::user_ompt_bundle::global_init();
    comp::user_ompt_bundle::reset();
    // provide environment variable for enabling/disabling
    if(config::get_use_ompt())
    {
        tim::auto_lock_t lk{ tim::type_mutex<ompt_handle_t>() };
        comp::user_ompt_bundle::configure<omnitrace::component::user_region>();
        f_bundle =
            std::make_unique<ompt_bundle_t>("ompt", quirk::config<quirk::auto_start>{});
    }
}

void
shutdown()
{
    OMNITRACE_VERBOSE(1, "Shutting down OMPT...\n");
    if(f_bundle)
    {
        f_bundle->stop();
        ompt_context_t::cleanup();
        trait::runtime_enabled<ompt_toolset_t>::set(false);
    }
    f_bundle.reset();
}
}  // namespace ompt
}  // namespace omnitrace

// include the ompt_start_tool definition
#    include <timemory/components/ompt/start_tool.hpp>

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
