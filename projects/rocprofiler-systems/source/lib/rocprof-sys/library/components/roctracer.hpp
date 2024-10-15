// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"

#include <timemory/api.hpp>
#include <timemory/components/base.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/enum.h>
#include <timemory/macros/os.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/transient_function.hpp>

ROCPROFSYS_COMPONENT_ALIAS(roctracer_data,
                           ::tim::component::data_tracker<double, roctracer>)

namespace rocprofsys
{
namespace component
{
struct roctracer
: base<roctracer, void>
, private policy::instance_tracker<roctracer, false>
{
    using value_type   = void;
    using base_type    = base<roctracer, void>;
    using tracker_type = policy::instance_tracker<roctracer, false>;

    ROCPROFSYS_DEFAULT_OBJECT(roctracer)

    static void preinit();
    static void global_finalize() { shutdown(); }

    static bool is_setup();
    static void setup(void* hsa_api_table, bool on_load_trace = false);
    static void flush();
    static void shutdown();
    static void add_setup(const std::string&, std::function<void()>&&);
    static void add_shutdown(const std::string&, std::function<void()>&&);
    static void remove_setup(const std::string&);
    static void remove_shutdown(const std::string&);

    void start();
    void stop();

    // this function protects roctracer_flush_activty from being called
    // when rocprof-sys exits during a callback
    [[nodiscard]] static scope::transient_destructor protect_flush_activity();
};

#if !defined(ROCPROFSYS_USE_ROCTRACER)
inline void
roctracer::setup(void*, bool)
{}

inline void
roctracer::flush()
{}

inline void
roctracer::shutdown()
{}

inline bool
roctracer::is_setup()
{
    return false;
}
#endif
}  // namespace component
}  // namespace rocprofsys

#if !defined(ROCPROFSYS_USE_ROCTRACER)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::roctracer_data, false_type)
#endif

TIMEMORY_SET_COMPONENT_API(rocprofsys::component::roctracer_data, project::timemory,
                           category::timing, os::supports_unix)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_timing_category, component::roctracer_data, true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::roctracer_data, true_type)

#if defined(ROCPROFSYS_USE_ROCTRACER) && ROCPROFSYS_USE_ROCTRACER > 0
#    if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                        \
        (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#        include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(roctracer, false, void)
ROCPROFSYS_DECLARE_EXTERN_COMPONENT(roctracer_data, true, double)

#    endif
#endif
