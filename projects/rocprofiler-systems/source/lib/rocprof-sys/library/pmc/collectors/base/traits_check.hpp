// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocprofsys::pmc::collectors::base
{

/**
 * @brief Concept satisfied when Traits exposes all required nested type aliases.
 *
 * Required types:
 * - metrics_t: The metrics data structure for this device type
 * - enabled_metrics_t: Bitset/struct indicating which metrics are enabled
 * - device_t: The device class type
 * - device_ptr_t: Smart pointer type for device (typically shared_ptr<device_t>)
 * - container_t: Container type for storing devices (vector or set)
 */
template <typename Traits>
concept has_required_types = requires {
    typename Traits::metrics_t;
    typename Traits::enabled_metrics_t;
    typename Traits::device_t;
    typename Traits::device_ptr_t;
    typename Traits::container_t;
};

/**
 * @brief Concept satisfied when Traits exposes the device_name constant.
 */
template <typename Traits>
concept has_device_name = requires { Traits::device_name; };

namespace detail
{
struct dummy_settings
{};
struct dummy_provider
{};
}  // namespace detail

/**
 * @brief Concept satisfied when Traits exposes a callable enumerate_devices template.
 *
 * Expected signature:
 *   template <typename Settings, typename Provider>
 *   static std::vector<device_entry> enumerate_devices(std::shared_ptr<Provider>)
 *
 * enumerate_devices is itself a template, so the concept resolves it against dummy
 * Settings/Provider types just to confirm a valid instantiation exists.
 */
template <typename Traits>
concept has_enumerate_devices = requires(
    std::shared_ptr<detail::dummy_provider> provider) {
    Traits::template enumerate_devices<detail::dummy_settings, detail::dummy_provider>(
        provider);
};

/**
 * @brief Composite concept: Traits is usable as a collector traits parameter.
 */
template <typename Traits>
concept valid_collector_traits = has_required_types<Traits> && has_device_name<Traits> &&
                                 has_enumerate_devices<Traits>;

}  // namespace rocprofsys::pmc::collectors::base
