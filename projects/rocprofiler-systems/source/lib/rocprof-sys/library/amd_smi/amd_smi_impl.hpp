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

#include "core/debug.hpp"
#include "library/amd_smi/common.hpp"

#include <timemory/components/timing/backends.hpp>

#include <cassert>
#include <functional>
#include <memory>

namespace rocprofsys
{
namespace amd_smi
{

#if ROCPROFSYS_USE_ROCM > 0

using get_timestamp_t = std::function<unsigned long()>;

template <typename Config>
struct amd_smi_impl
{
    using SmiServiceFactory  = typename Config::SmiServiceFactory;
    using SettingsApi        = typename Config::SettingsApi;
    using PerfettoApi        = typename Config::PerfettoApi;
    using CacheApi           = typename Config::RocpdApi;
    using smi_service        = typename SmiServiceFactory::smi_service;
    using processor_vector_t = typename smi_service::processor_vector_t;
    using processor_t        = typename smi_service::processor_t;

    void setup()
    {
        m_smi_service = SmiServiceFactory::create_smi_service();
        auto _version = m_smi_service->get_version();

        ROCPROFSYS_VERBOSE_F(0, "AMD SMI version: %u.%u.%u - str: %s.\n",
                             _version.numeric_representation.major,
                             _version.numeric_representation.minor,
                             _version.numeric_representation.release,
                             _version.string_representation.c_str());

        m_gpu_processors =
            m_smi_service->get_processors([](const processor_vector_t& processors) {
                auto               filter = SettingsApi::get_device_filter();
                processor_vector_t result{};

                switch(filter.mode)
                {
                    case device_selection_mode::ALL: return processors;
                    case device_selection_mode::NONE: break;
                    case device_selection_mode::SPECIFIC:
                        std::copy_if(processors.begin(), processors.end(),
                                     std::back_inserter(result), [&](const auto& device) {
                                         return filter.indices.count(
                                                    device->get_index()) > 0;
                                     });
                        break;
                }
                return result;
            });

        m_enabled_metrics = SettingsApi::get_enabled_metrics();

        ROCPROFSYS_VERBOSE_F(1, "Enabled %zu GPU processors for AMD SMI sampling\n",
                             m_gpu_processors.size());

        for(const auto& processor : m_gpu_processors)
        {
            PerfettoApi::init_storage(processor->get_index());
        }
    }

    void config()
    {
        auto _enabled_metrics = SettingsApi::get_enabled_metrics();
        CacheApi::initialize_category_metadata();

        for(const auto& device : m_gpu_processors)
        {
            auto device_index = device->get_index();
            PerfettoApi::setup_counter_tracks(device_index, _enabled_metrics);
            CacheApi::initialize_smi_tracks_metadata(device_index);
            CacheApi::initialize_smi_pmc_metadata(device_index);
        }
    }

    void sample(const get_timestamp_t& get_timestamp)
    {
        auto _enabled_metrics = SettingsApi::get_enabled_metrics();

        for(auto it = m_gpu_processors.begin(); it != m_gpu_processors.end();)
        {
            auto& processor  = *it;
            auto  _timestamp = get_timestamp();
            assert(_timestamp <
                   static_cast<unsigned long>(std::numeric_limits<int64_t>::max()));

            try
            {
                auto _supported_metrics = processor->get_supported_metrics();
                auto _smi_metrics       = processor->get_smi_metrics();
                auto _device_id         = processor->get_index();

                CacheApi::store_sample(_device_id, _supported_metrics, _enabled_metrics,
                                       _smi_metrics, _timestamp);
                PerfettoApi::store_sample(_device_id, _smi_metrics, _timestamp);
                ++it;
            } catch(const std::runtime_error& e)
            {
                ROCPROFSYS_WARNING(
                    0,
                    "Reading metrics failed for device with ID %zu. Error: %s. "
                    "Disabling device!\n",
                    processor->get_index(), e.what());
                it = m_gpu_processors.erase(it);
            }
        }
    }

    void post_process()
    {
        for(const auto& processor : m_gpu_processors)
        {
            PerfettoApi::post_process(processor->get_index(), m_enabled_metrics,
                                      processor->get_supported_metrics());
        }
    }

    const processor_vector_t& get_processors() const { return m_gpu_processors; }

    size_t get_processor_count() const { return m_gpu_processors.size(); }

    void shutdown()
    {
        if(m_smi_service)
        {
            m_smi_service->shutdown();
            m_smi_service.reset();
        }
    }

private:
    processor_vector_t           m_gpu_processors;
    std::shared_ptr<smi_service> m_smi_service;
    enabled_metric               m_enabled_metrics{};
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace amd_smi
}  // namespace rocprofsys
