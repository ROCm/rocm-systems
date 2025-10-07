// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/components/fwd.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "library/amd_smi/common.hpp"
#include "timemory/components/timing/backends.hpp"
#include <algorithm>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif
namespace rocprofsys
{
namespace amd_smi
{

using get_timestamp_t = std::function<unsigned long()>;

template <typename smi_service_factory, typename settings_api, typename perfetto_api,
          typename rocpd_api>
struct amd_smi_impl
{
    using smi_service        = typename smi_service_factory::smi_service;
    using processor_vector_t = typename smi_service::processor_vector_t;
    using processor_t        = typename smi_service::processor_t;

    void setup()
    {
        m_smi_service = smi_service_factory::create_smi_service();
        auto _version = m_smi_service->get_version();
        // ROCPROFSYS_VERBOSE_F(0, "AMD SMI version: %u.%u.%u - str: %s.\n",
        //                      _version.numeric_representation.major,
        //                      _version.numeric_representation.minor,
        //                      _version.numeric_representation.release,
        //                      _version.string_representation.c_str());

        m_gpu_processors =
            m_smi_service->get_processors([](const processor_vector_t& processors) {
                auto               filter = settings_api::get_device_filter();
                processor_vector_t result{};
                switch(filter.mode)
                {
                    case device_selection_mode::all: return processors;
                    case device_selection_mode::none: break;
                    case device_selection_mode::specific:
                        std::copy_if(
                            processors.begin(), processors.end(),
                            std::back_inserter(result), [&](const auto& device) {
                                return (filter.indices.count(device->get_index()) > 0);
                            });
                        break;
                }
                return result;
            });

        for(const auto& processor : m_gpu_processors)
        {
            perfetto_api::init_storage(processor->get_index());
        }
    }

    void config()
    {
        auto _enabled_metrics = settings_api::get_enabled_metrics();
        rocpd_api::initialize_category_metadata();

        std::for_each(
            m_gpu_processors.begin(), m_gpu_processors.end(), [&](const auto& device) {
                auto device_index = device->get_index();
                perfetto_api::setup_counter_tracks(device->get_index(), _enabled_metrics);
                rocpd_api::initialize_smi_tracks_metadata(device_index);
                rocpd_api::initialize_smi_pmc_metadata(device_index);
            });
    }

    void sample(const get_timestamp_t& get_timestamp)
    {
        auto _enabled_metrics = settings_api::get_enabled_metrics();

        for(auto it = m_gpu_processors.begin(); it != m_gpu_processors.end();)
        {
            auto& processor  = *it;
            auto  _timestamp = get_timestamp();
            assert(_timestamp < std::numeric_limits<int64_t>::max());

            try
            {
                auto _supported_metrics = processor->get_supported_metrics();
                auto _smi_metrics       = processor->get_smi_metrics();
                auto _device_id         = processor->get_index();

                rocpd_api::store_sample(_device_id, _supported_metrics, _enabled_metrics,
                                        _smi_metrics, _timestamp);
                perfetto_api::store_sample(_device_id, _smi_metrics, _timestamp);
                ++it;
            } catch(const std::runtime_error& e)
            {
                ROCPROFSYS_WARNING(
                    0,
                    "Reading metrics failed for device with ID %zu. Error: %s. "
                    "Disabling device!\n",
                    processor->get_index(), e.what());
                auto device_to_remove = std::find(m_gpu_processors.begin(),
                                                  m_gpu_processors.end(), processor);
                m_gpu_processors.erase(device_to_remove);
            }
        }
    }

    void post_process()
    {
        auto _enabled_metrics = settings_api::get_enabled_metrics();
        for(const auto& processor : m_gpu_processors)
        {
            perfetto_api::post_process(processor->get_index(), _enabled_metrics,
                                       processor->get_supported_metrics());
        }
    }

private:
    processor_vector_t           m_gpu_processors;
    std::shared_ptr<smi_service> m_smi_service;
};

void
setup();

void
config();

void
sample();

void
shutdown();

void
post_process();

void set_state(State);

#if !defined(ROCPROFSYS_USE_ROCM) || ROCPROFSYS_USE_ROCM == 0

inline void
setup()
{}

inline void
config()
{}

inline void
sample()
{}

inline void
shutdown()
{}

inline void
post_process()
{}

inline void
set_state(State)
{}
#endif
}  // namespace amd_smi
}  // namespace rocprofsys

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
#    if !defined(ROCPROFSYS_EXTERN_COMPONENTS) ||                                        \
        (defined(ROCPROFSYS_EXTERN_COMPONENTS) && ROCPROFSYS_EXTERN_COMPONENTS > 0)

#        include <timemory/components/base.hpp>
#        include <timemory/components/data_tracker/components.hpp>
#        include <timemory/operations.hpp>

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_gfx>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_umc>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_mm>),
    true, double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_vcn>), true,
    double)

ROCPROFSYS_DECLARE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_jpeg>), true,
    double)

#    endif
#endif
