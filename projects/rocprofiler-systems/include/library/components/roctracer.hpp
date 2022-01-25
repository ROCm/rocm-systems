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

#pragma once

#include "library/components/fwd.hpp"
#include "library/defines.hpp"

#include <timemory/api.hpp>
#include <timemory/components/base.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/enum.h>
#include <timemory/macros/os.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>

namespace tim
{
namespace component
{
using roctracer_data = data_tracker<double, roctracer>;

struct roctracer
: base<roctracer, void>
, private policy::instance_tracker<roctracer, false>
{
    using value_type   = void;
    using base_type    = base<roctracer, void>;
    using tracker_type = policy::instance_tracker<roctracer, false>;

    TIMEMORY_DEFAULT_OBJECT(roctracer)

    static void preinit();
    static void global_init() { setup(); }
    static void global_finalize() { tear_down(); }

    static bool is_setup();
    static void setup();
    static void tear_down();
    static void add_setup(const std::string&, std::function<void()>&&);
    static void add_tear_down(const std::string&, std::function<void()>&&);
    static void remove_setup(const std::string&);
    static void remove_tear_down(const std::string&);

    void start();
    void stop();
    void set_prefix(const char* _v) { m_prefix = _v; }

private:
    const char* m_prefix = nullptr;
};
}  // namespace component
}  // namespace tim

TIMEMORY_SET_COMPONENT_API(component::roctracer_data, project::timemory, category::timing,
                           os::supports_unix)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::roctracer_data, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::roctracer_data, true_type)

#if !defined(OMNITRACE_EXTERN_COMPONENTS) ||                                             \
    (defined(OMNITRACE_EXTERN_COMPONENTS) && OMNITRACE_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

TIMEMORY_DECLARE_EXTERN_COMPONENT(roctracer, false, void)
TIMEMORY_DECLARE_EXTERN_COMPONENT(roctracer_data, true, double)

#endif
