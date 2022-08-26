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

#include "library/components/rcclp.hpp"
#include "library/components/category_region.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"

#include <timemory/timemory.hpp>

#if OMNITRACE_HIP_VERSION == 0 || OMNITRACE_HIP_VERSION >= 50200
#    include <rccl/rccl.h>
#else
#    include <rccl.h>
#endif

#include <dlfcn.h>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

static uint64_t global_id      = std::numeric_limits<uint64_t>::max();
static void*    librccl_handle = nullptr;

namespace omnitrace
{
namespace rcclp
{
void
configure()
{}

void
setup()
{
    configure();

    // make sure the symbols are loaded to be wrapped
    auto libpath   = tim::get_env<std::string>("OMNITRACE_RCCL_LIBRARY", "librccl.so");
    librccl_handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if(!librccl_handle) fprintf(stderr, "%s\n", dlerror());
    dlerror();  // Clear any existing error

    auto _use_data = tim::get_env("OMNITRACE_RCCLP_COMM_DATA", get_use_timemory());
    if(!get_use_timemory())
    {
        trait::runtime_enabled<comp::comm_data>::set(false);
        trait::runtime_enabled<comp::comm_data_tracker_t>::set(false);
    }
    else
    {
        trait::runtime_enabled<comp::comm_data>::set(_use_data);
        trait::runtime_enabled<comp::comm_data_tracker_t>::set(_use_data);
    }

    comp::configure_rcclp();
    global_id = comp::activate_rcclp();
    if(librccl_handle) dlclose(librccl_handle);
}

void
shutdown()
{
    if(global_id < std::numeric_limits<uint64_t>::max())
        comp::deactivate_rcclp(global_id);
}
}  // namespace rcclp
}  // namespace omnitrace
