// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocprofsys::pmc::collectors
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
 * required interface methods: setup(), config(), sample(timestamp),
 * post_process(), shutdown()
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
 *     auto timestamp = get_clock_now();
 *     for (auto& slice : slices) {
 *         slice.setup();             // Calls appropriate collector's setup()
 *         slice.sample(timestamp);   // Calls appropriate collector's sample()
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
    , m_sample_impl{ [](void* ptr, int64_t timestamp) {
        static_cast<T*>(ptr)->sample(timestamp);
    } }
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
     * @param timestamp Current timestamp in nanoseconds.
     * Calls the underlying collector's sample() method.
     */
    void sample(int64_t timestamp) { m_sample_impl(m_object, timestamp); }

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
    using setup_fn_t        = void (*)(void*);
    using config_fn_t       = void (*)(void*);
    using sample_fn_t       = void (*)(void*, int64_t);
    using post_process_fn_t = void (*)(void*);
    using shutdown_fn_t     = void (*)(void*);

    void*             m_object;            /**< Non-owning pointer to collector */
    setup_fn_t        m_setup_impl;        /**< Type-erased setup function */
    config_fn_t       m_config_impl;       /**< Type-erased config function */
    sample_fn_t       m_sample_impl;       /**< Type-erased sample function */
    post_process_fn_t m_post_process_impl; /**< Type-erased post_process function */
    shutdown_fn_t     m_shutdown_impl;     /**< Type-erased shutdown function */
};

}  // namespace rocprofsys::pmc::collectors
