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

#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "library/thread_data.hpp"

#include <timemory/api.hpp>
#include <timemory/backends/hardware_counters.hpp>
#include <timemory/components/base.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/enum.h>
#include <timemory/macros.hpp>
#include <timemory/macros/os.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/macros.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/transient_function.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rocprofsys
{
namespace component
{
using rocm_metric_type   = unsigned long long;
using rocm_info_entry    = ::tim::hardware_counters::info;
using rocm_feature_value = std::variant<uint32_t, float, uint64_t, double>;

struct rocm_counter
{
    std::array<rocm_metric_type, ROCPROFSYS_ROCM_MAX_COUNTERS> counters;
};

struct rocm_event
{
    using value_type = rocm_feature_value;

    uint32_t                        device_id      = 0;
    uint32_t                        thread_id      = 0;
    uint32_t                        queue_id       = 0;
    rocm_metric_type                entry          = 0;
    rocm_metric_type                exit           = 0;
    std::string                     name           = {};
    std::vector<size_t>             feature_names  = {};
    std::vector<rocm_feature_value> feature_values = {};

    rocm_event() = default;
    rocm_event(uint32_t _dev, uint32_t _thr, uint32_t _queue, std::string _event_name,
               rocm_metric_type begin, rocm_metric_type end, uint32_t _feature_count,
               void* _features);

    std::string as_string() const;

    friend std::ostream& operator<<(std::ostream& _os, const rocm_event& _v)
    {
        return (_os << _v.as_string());
    }

    friend bool operator<(const rocm_event& _lhs, const rocm_event& _rhs)
    {
        return std::tie(_lhs.device_id, _lhs.queue_id, _lhs.entry, _lhs.thread_id) <
               std::tie(_rhs.device_id, _rhs.queue_id, _rhs.entry, _rhs.thread_id);
    }
};

using rocm_data_t       = std::vector<rocm_event>;
using rocm_data_tracker = data_tracker<rocm_feature_value, rocm_event>;

rocprofsys::unique_ptr_t<rocm_data_t>&
rocm_data(int64_t _tid = threading::get_id());

using rocprofiler_value = typename rocm_event::value_type;
using rocprofiler_data  = data_tracker<rocprofiler_value, rocprofiler>;

struct rocprofiler
: base<rocprofiler, void>
, private policy::instance_tracker<rocprofiler, false>
{
    using value_type   = void;
    using base_type    = base<rocprofiler, void>;
    using tracker_type = policy::instance_tracker<rocprofiler, false>;

    ROCPROFSYS_DEFAULT_OBJECT(rocprofiler)

    static void preinit();
    static void global_init() { setup(); }
    static void global_finalize() { shutdown(); }

    static bool is_setup();
    static void setup();
    static void shutdown();
    static void add_setup(const std::string&, std::function<void()>&&);
    static void add_shutdown(const std::string&, std::function<void()>&&);
    static void remove_setup(const std::string&);
    static void remove_shutdown(const std::string&);

    void start();
    void stop();

    // this function protects rocprofiler_flush_activty from being called
    // when rocprof-sys exits during a callback
    [[nodiscard]] static scope::transient_destructor protect_flush_activity();
};

#if !defined(ROCPROFSYS_USE_ROCPROFILER)
inline void
rocprofiler::setup()
{}

inline void
rocprofiler::shutdown()
{}

inline bool
rocprofiler::is_setup()
{
    return false;
}
#endif
}  // namespace component
}  // namespace rocprofsys

namespace tim
{
namespace component
{
using ::rocprofsys::component::rocm_data_tracker;
using ::rocprofsys::component::rocm_feature_value;
using ::rocprofsys::component::rocprofiler_data;
using ::rocprofsys::component::rocprofiler_value;
}  // namespace component
}  // namespace tim

namespace tim
{
namespace operation
{
template <>
struct set_storage<component::rocm_data_tracker>
{
    using T                             = component::rocm_data_tracker;
    static constexpr size_t max_threads = 4096;
    using type                          = T;
    using storage_array_t               = std::array<storage<type>*, max_threads>;
    friend struct get_storage<component::rocm_data_tracker>;

    ROCPROFSYS_DEFAULT_OBJECT(set_storage)

    auto operator()(storage<type>*, size_t) const {}
    auto operator()(type&, size_t) const {}
    auto operator()(storage<type>* _v) const { get().fill(_v); }

private:
    static storage_array_t& get()
    {
        static storage_array_t _v = { nullptr };
        return _v;
    }
};

template <>
struct get_storage<component::rocm_data_tracker>
{
    using type = component::rocm_data_tracker;

    ROCPROFSYS_DEFAULT_OBJECT(get_storage)

    auto operator()(const type&) const
    {
        return operation::set_storage<type>::get().at(0);
    }

    auto operator()() const
    {
        type _obj{};
        return (*this)(_obj);
    }

    auto operator()(size_t _idx) const
    {
        return operation::set_storage<type>::get().at(_idx);
    }

    auto operator()(type&, size_t _idx) const { return (*this)(_idx); }
};
}  // namespace operation
}  // namespace tim

#if !defined(ROCPROFSYS_USE_ROCPROFILER)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_available, component::rocprofiler_data, false_type)
#endif

TIMEMORY_SET_COMPONENT_API(component::rocprofiler_data, project::timemory,
                           category::timing, os::supports_unix)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(is_timing_category, component::rocprofiler_data,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::rocprofiler_data,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_units, component::rocprofiler_data, false_type)
TIMEMORY_STATISTICS_TYPE(component::rocprofiler_data, component::rocprofiler_value)
TIMEMORY_STATISTICS_TYPE(component::rocm_data_tracker, component::rocm_feature_value)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_units, component::rocm_data_tracker, false_type)

#if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                            \
    (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(rocprofiler, false, void)
ROCPROFSYS_DECLARE_EXTERN_COMPONENT(rocprofiler_data, true, double)

#endif
