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

#include "library/components/rocprofiler.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/dynamic_library.hpp"
#include "core/perfetto.hpp"
#include "core/redirect.hpp"
#include "library/rocprofiler.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"

#include <timemory/storage/types.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic/functional.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>

#include <rocprofiler.h>

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace rocprofsys
{
namespace component
{
namespace
{
auto&
rocprofiler_activity_count()
{
    static std::atomic<int64_t> _v{ 0 };
    return _v;
}
}  // namespace

unique_ptr_t<rocm_data_t>&
rocm_data(int64_t _tid)
{
    using thread_data_t = thread_data<rocm_data_t, rocm_event>;
    return thread_data_t::instance(construct_on_thread{ _tid });
}

rocm_event::rocm_event(uint32_t _dev, uint32_t _thr, uint32_t _queue,
                       std::string _event_name, rocm_metric_type _begin,
                       rocm_metric_type _end, uint32_t _feature_count, void* _features_v)
: device_id{ _dev }
, thread_id{ _thr }
, queue_id{ _queue }
, entry{ _begin }
, exit{ _end }
, name(std::move(_event_name))
{
    feature_values.reserve(_feature_count);
    feature_names.reserve(_feature_count);
    auto* _features = static_cast<rocprofiler_feature_t*>(_features_v);
    for(uint32_t i = 0; i < _feature_count; ++i)
    {
        const rocprofiler_feature_t* p = &_features[i];
        feature_names.emplace_back(i);
        switch(p->data.kind)
        {
            // Output metrics results
            case ROCPROFILER_DATA_KIND_UNINIT: break;
            case ROCPROFILER_DATA_KIND_BYTES:
                feature_values.emplace_back(
                    rocm_feature_value{ p->data.result_bytes.size });
                break;
            case ROCPROFILER_DATA_KIND_INT32:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_int32 });
                break;
            case ROCPROFILER_DATA_KIND_FLOAT:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_float });
                break;
            case ROCPROFILER_DATA_KIND_DOUBLE:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_double });
                break;
            case ROCPROFILER_DATA_KIND_INT64:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_int64 });
                break;
        }
    }
}

std::string
rocm_event::as_string() const
{
    std::stringstream _ss{};
    _ss << name << ", device: " << device_id << ", queue: " << queue_id
        << ", thread: " << thread_id << ", entry: " << entry << ", exit = " << exit;
    _ss.precision(3);
    _ss << std::fixed;
    for(size_t i = 0; i < feature_names.size(); ++i)
    {
        auto _name = rocprofsys::rocprofiler::get_data_labels().at(device_id).at(
            feature_names.at(i));
        _ss << ", " << _name << " = ";
        auto _as_string = [&_ss](auto&& itr) { _ss << std::setw(4) << itr; };
        std::visit(_as_string, feature_values.at(i));
    }
    return _ss.str();
}

void
rocprofiler::preinit()
{
    rocprofiler_data::label()       = "rocprofiler";
    rocprofiler_data::description() = "ROCm hardware counters";
}

void
rocprofiler::start()
{
    if(tracker_type::start() == 0) setup();
}

void
rocprofiler::stop()
{
    if(tracker_type::stop() == 0) shutdown();
}

bool
rocprofiler::is_setup()
{
    return rocprofsys::rocprofiler::is_setup();
}

void
rocprofiler::add_setup(const std::string&, std::function<void()>&&)
{}

void
rocprofiler::add_shutdown(const std::string&, std::function<void()>&&)
{}

void
rocprofiler::remove_setup(const std::string&)
{}

void
rocprofiler::remove_shutdown(const std::string&)
{}

void
rocprofiler::setup()
{
    ROCPROFSYS_VERBOSE_F(1, "rocprofiler is setup\n");
}

void
rocprofiler::shutdown()
{
    rocprofsys::rocprofiler::post_process();
    rocprofsys::rocprofiler::rocm_cleanup();
    ROCPROFSYS_VERBOSE_F(1, "rocprofiler is shutdown\n");
}

scope::transient_destructor
rocprofiler::protect_flush_activity()
{
    return scope::transient_destructor([]() { --rocprofiler_activity_count(); },
                                       []() { ++rocprofiler_activity_count(); });
}
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(rocprofiler, false, void)
ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(rocprofiler_data, true,
                                        tim::component::rocprofiler_value)
