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
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/dynamic_library.hpp"
#include "core/rccl.hpp"
#include "core/timemory.hpp"
#include "library/components/category_region.hpp"

#include <timemory/timemory.hpp>

#include <dlfcn.h>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

namespace
{
uint64_t global_id = std::numeric_limits<uint64_t>::max();
}

namespace rocprofsys
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
    dynamic_library _librccl{
        "ROCPROFSYS_RCCL_LIBRARY", "librccl.so", RTLD_NOW | RTLD_GLOBAL, true, true, true
    };

    auto _use_data = tim::get_env("ROCPROFSYS_RCCLP_COMM_DATA", get_use_timemory());
    if(!get_use_timemory())
    {
        trait::runtime_enabled<component::comm_data>::set(false);
        trait::runtime_enabled<component::comm_data_tracker_t>::set(false);
    }
    else
    {
        trait::runtime_enabled<component::comm_data>::set(_use_data);
        trait::runtime_enabled<component::comm_data_tracker_t>::set(_use_data);
    }

    component::configure_rcclp();
    global_id = component::activate_rcclp();
}

void
shutdown()
{
    if(global_id < std::numeric_limits<uint64_t>::max())
        component::deactivate_rcclp(global_id);
}
}  // namespace rcclp
}  // namespace rocprofsys
