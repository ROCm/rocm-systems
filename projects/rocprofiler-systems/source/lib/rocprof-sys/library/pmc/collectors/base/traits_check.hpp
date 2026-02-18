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
 * The enumerate_devices function should be a static template function that
 * takes a device provider and device filter, returning a vector of device entries.
 */
template <typename Traits, typename = void>
struct has_enumerate_devices : std::false_type
{};

template <typename Traits>
struct has_enumerate_devices<
    Traits, std::void_t<decltype(Traits::device_entry::device),
                        decltype(Traits::device_entry::supported_metrics)>> : std::true_type
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
