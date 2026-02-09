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

#include <functional>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{

/**
 * @brief Type-erased collector slice - non-owning view of any collector type.
 *
 * This class provides a lightweight type erasure mechanism for PMC collectors.
 * It allows storing heterogeneous collector types (GPU, NIC, CPU) in a single
 * container without requiring virtual inheritance or a common base class.
 *
 * The collector_slice is a non-owning view (like std::string_view or
 * std::span). The actual collector object must outlive the slice.
 *
 * Any type T can be wrapped in a collector_slice as long as it provides the
 * required interface methods: setup(), config(), sample(), post_process(),
 * shutdown()
 *
 * Example usage:
 * @code
 *     pmc::collectors::gpu::collector gpu_collector(device_mgr);
 *     pmc::collectors::nic::collector nic_collector(device_mgr);
 *
 *     std::vector<pmc::collectors::collector_slice> slices;
 *     slices.emplace_back(gpu_collector);  // Creates slice to gpu_collector
 *     slices.emplace_back(nic_collector);  // Creates slice to nic_collector
 *
 *     for (auto& slice : slices) {
 *         slice.setup();     // Calls appropriate collector's setup()
 *         slice.sample();    // Calls appropriate collector's sample()
 *     }
 * @endcode
 */
class collector_slice
{
public:
    /**
     * @brief Construct a collector_slice from any collector type.
     *
     * @tparam T Collector type (must have setup, config, sample, post_process, shutdown
     * methods)
     * @param obj Reference to the collector object (must outlive the slice)
     */
    template <typename T>
    explicit collector_slice(T& obj)
    : m_object{ &obj }
    , m_setup_impl{ [](void* ptr) { static_cast<T*>(ptr)->setup(); } }
    , m_config_impl{ [](void* ptr) { static_cast<T*>(ptr)->config(); } }
    , m_sample_impl{ [](void* ptr) { static_cast<T*>(ptr)->sample(); } }
    , m_post_process_impl{ [](void* ptr) { static_cast<T*>(ptr)->post_process(); } }
    , m_shutdown_impl{ [](void* ptr) { static_cast<T*>(ptr)->shutdown(); } }
    {}

    /**
     * @brief Setup the collector.
     *
     * Calls the underlying collector's setup() method.
     */
    void setup() { m_setup_impl(m_object); }

    /**
     * @brief Configure the collector.
     *
     * Calls the underlying collector's config() method.
     */
    void config() { m_config_impl(m_object); }

    /**
     * @brief Sample metrics from the collector.
     *
     * Calls the underlying collector's sample() method.
     */
    void sample() { m_sample_impl(m_object); }

    /**
     * @brief Post-process collected metrics.
     *
     * Calls the underlying collector's post_process() method.
     */
    void post_process() { m_post_process_impl(m_object); }

    /**
     * @brief Shutdown the collector.
     *
     * Calls the underlying collector's shutdown() method.
     */
    void shutdown() { m_shutdown_impl(m_object); }

private:
    void*                      m_object;      /**< Non-owning pointer to collector */
    std::function<void(void*)> m_setup_impl;  /**< Type-erased setup function */
    std::function<void(void*)> m_config_impl; /**< Type-erased config function */
    std::function<void(void*)> m_sample_impl; /**< Type-erased sample function */
    std::function<void(void*)>
        m_post_process_impl;                    /**< Type-erased post_process function */
    std::function<void(void*)> m_shutdown_impl; /**< Type-erased shutdown function */
};

}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
