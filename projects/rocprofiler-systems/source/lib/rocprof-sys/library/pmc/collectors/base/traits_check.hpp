// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <type_traits>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace base
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
 * @brief Type trait to check if Traits defines the multi_device constant.
 */
template <typename Traits, typename = void>
struct has_multi_device : std::false_type
{};

template <typename Traits>
struct has_multi_device<Traits, std::void_t<decltype(Traits::multi_device)>>
: std::true_type
{};

template <typename Traits>
inline constexpr bool has_multi_device_v = has_multi_device<Traits>::value;

/**
 * @brief Type trait to check if Traits defines enumerate_devices().
 *
 * Expected signature:
 *   template <typename Settings, typename Provider>
 *   static std::vector<device_entry> enumerate_devices(std::shared_ptr<Provider>)
 *
 * Since enumerate_devices is a template function, we cannot use &Traits::enumerate_devices
 * to detect it (the compiler cannot resolve which instantiation to take the address of).
 * Instead, we use SFINAE with dummy types to check if a valid instantiation exists.
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
    Traits,
    std::void_t<decltype(Traits::template enumerate_devices<detail::dummy_settings,
                                                            detail::dummy_provider>(
        std::declval<std::shared_ptr<detail::dummy_provider>>()))>> : std::true_type
{};

template <typename Traits>
inline constexpr bool has_enumerate_devices_v = has_enumerate_devices<Traits>::value;

/**
 * @brief Validate that a Traits type satisfies all collector requirements.
 *
 * This function uses static_assert to provide clear compile-time error messages
 * when a traits type is missing required members. Call this at the beginning of
 * the collector template to catch errors early.
 *
 * Required Traits interface:
 * - Types: metrics_t, enabled_metrics_t, device_t, device_ptr_t, container_t
 * - Constants: device_name (const char*), multi_device (bool)
 *
 * @tparam Traits The traits type to validate
 */
template <typename Traits>
constexpr void
validate_traits()
{
    static_assert(has_required_types_v<Traits>,
                  "Traits must define: metrics_t, enabled_metrics_t, device_t, "
                  "device_ptr_t, container_t");

    static_assert(has_device_name_v<Traits>,
                  "Traits must define: static constexpr const char* device_name");

    static_assert(has_multi_device_v<Traits>,
                  "Traits must define: static constexpr bool multi_device");
}

}  // namespace base
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
