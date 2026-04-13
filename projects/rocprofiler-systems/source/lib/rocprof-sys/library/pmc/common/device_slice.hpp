// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/common/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rocprofsys::pmc
{

/**
 * @brief Type-erased device slice - owning wrapper for any device type.
 *
 * This class provides a lightweight type erasure mechanism for PMC devices.
 * It allows storing heterogeneous device types (GPU, NIC) in a single
 * container without requiring virtual inheritance or a common base class.
 *
 * Any type T can be wrapped in a device_slice as long as it provides the
 * required interface methods: get_index(), get_name(), get_product_name(),
 * get_vendor_name(), is_supported()
 *
 * Example usage:
 * @code
 *     auto gpu_dev = std::make_shared<gpu::device<Driver>>(...);
 *     auto nic_dev = std::make_shared<nic::device<Driver>>(...);
 *
 *     std::vector<device_slice> devices;
 *     devices.emplace_back(gpu_dev, device_type::GPU);
 *     devices.emplace_back(nic_dev, device_type::NIC);
 *
 *     for (auto& dev : devices) {
 *         std::cout << dev.get_name() << ": " << dev.get_product_name() << "\n";
 *     }
 *
 *     // Type recovery when needed
 *     if (auto gpu = devices[0].as<gpu::device<Driver>>()) {
 *         auto metrics = gpu->get_gpu_metrics(enabled, timestamp);
 *     }
 * @endcode
 */
class device_slice
{
public:
    /**
     * @brief Construct a device_slice from any device type.
     *
     * @tparam T Device type (must have get_index, get_name, get_product_name,
     *           get_vendor_name, is_supported methods)
     * @param dev Shared pointer to the device object
     * @param dev_type The device type (GPU, NIC)
     */
    template <typename T>
    device_slice(std::shared_ptr<T> dev, device_type dev_type)
    : m_device{ std::move(dev) }
    , m_type{ dev_type }
    , m_get_index_impl{ [](void* ptr) -> size_t {
        return static_cast<T*>(ptr)->get_index();
    } }
    , m_get_name_impl{ [](void* ptr) -> const std::string& {
        return static_cast<T*>(ptr)->get_name();
    } }
    , m_get_product_name_impl{ [](void* ptr) -> const std::string& {
        return static_cast<T*>(ptr)->get_product_name();
    } }
    , m_get_vendor_name_impl{ [](void* ptr) -> const std::string& {
        return static_cast<T*>(ptr)->get_vendor_name();
    } }
    , m_is_supported_impl{ [](void* ptr) -> bool {
        return static_cast<T*>(ptr)->is_supported();
    } }
    {}

    /**
     * @brief Get the device index.
     * @return Device index.
     */
    [[nodiscard]] size_t get_index() const { return m_get_index_impl(m_device.get()); }

    /**
     * @brief Get the device name (e.g., "GPU0", "nic0").
     * @return Const reference to the device name.
     */
    [[nodiscard]] const std::string& get_name() const
    {
        return m_get_name_impl(m_device.get());
    }

    /**
     * @brief Get the product name (e.g., "AMD Instinct MI300X").
     * @return Const reference to the product name.
     */
    [[nodiscard]] const std::string& get_product_name() const
    {
        return m_get_product_name_impl(m_device.get());
    }

    /**
     * @brief Get the vendor name (e.g., "AMD").
     * @return Const reference to the vendor name.
     */
    [[nodiscard]] const std::string& get_vendor_name() const
    {
        return m_get_vendor_name_impl(m_device.get());
    }

    /**
     * @brief Check if the device is supported.
     * @return True if the device supports metrics collection.
     */
    [[nodiscard]] bool is_supported() const
    {
        return m_is_supported_impl(m_device.get());
    }

    /**
     * @brief Get the device type.
     * @return The device type (GPU, NIC).
     */
    [[nodiscard]] device_type type() const noexcept { return m_type; }

private:
    std::shared_ptr<void> m_device; /**< Owning pointer to device (type-erased) */
    device_type           m_type;   /**< Device type (GPU, NIC) */

    std::function<size_t(void*)>             m_get_index_impl;
    std::function<const std::string&(void*)> m_get_name_impl;
    std::function<const std::string&(void*)> m_get_product_name_impl;
    std::function<const std::string&(void*)> m_get_vendor_name_impl;
    std::function<bool(void*)>               m_is_supported_impl;
};

}  // namespace rocprofsys::pmc
