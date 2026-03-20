// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace rocprofsys::pmc::collectors::base
{

/**
 * @brief Type trait to check if Traits defines all required type aliases.
 *
 * Required types:
 * - metrics_t: The metrics data structure for this device type
 * - enabled_metrics_t: Bitset/struct indicating which metrics are enabled
 * - device_t: The device class type
 * - device_ptr_t: Smart pointer type for device (typically shared_ptr<device_t>)
 * - container_t: Container type for storing devices (vector or set)
 */
template <typename Traits, typename = void>
struct has_required_types : std::false_type
{};

template <typename Traits>
struct has_required_types<
    Traits, std::void_t<typename Traits::metrics_t, typename Traits::enabled_metrics_t,
                        typename Traits::device_t, typename Traits::device_ptr_t,
                        typename Traits::container_t>> : std::true_type
{};

template <typename Traits>
inline constexpr bool has_required_types_v = has_required_types<Traits>::value;

/**
 * @brief Type trait to check if Traits defines the device_name constant.
 */
template <typename Traits, typename = void>
struct has_device_name : std::false_type
{};

template <typename Traits>
struct has_device_name<Traits, std::void_t<decltype(Traits::device_name)>>
: std::true_type
{};

template <typename Traits>
inline constexpr bool has_device_name_v = has_device_name<Traits>::value;

/**
 * @brief Type trait to check if Traits defines enumerate_devices().
 *
 * Expected signature:
 *   template <typename Settings, typename Provider>
 *   static std::vector<device_entry> enumerate_devices(std::shared_ptr<Provider>)
 *
 * Since enumerate_devices is a template function, we cannot use
 * &Traits::enumerate_devices to detect it (the compiler cannot resolve which
 * instantiation to take the address of). Instead, we use SFINAE with dummy types to check
 * if a valid instantiation exists.
 */
template <typename Traits, typename = void>
struct has_enumerate_devices : std::false_type
{};

namespace detail
{
struct dummy_settings
{};
struct dummy_provider
{};
}  // namespace detail

template <typename Traits>
struct has_enumerate_devices<
    Traits, std::void_t<decltype(Traits::template enumerate_devices<
                                 detail::dummy_settings, detail::dummy_provider>(
                std::declval<std::shared_ptr<detail::dummy_provider>>()))>>
: std::true_type
{};

template <typename Traits>
inline constexpr bool has_enumerate_devices_v = has_enumerate_devices<Traits>::value;

}  // namespace rocprofsys::pmc::collectors::base
