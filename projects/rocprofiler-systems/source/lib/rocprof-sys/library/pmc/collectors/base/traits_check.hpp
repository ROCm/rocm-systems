// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
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

/**
 * @brief Concept satisfied when DeviceProvider exposes shutdown().
 *
 * collector calls provider->shutdown() during teardown; the rest of the
 * provider API surface is reached through Traits::enumerate_devices and is
 * therefore policy-specific - it stays unconstrained at the base level.
 */
template <typename Provider>
concept device_provider = requires(Provider provider) { provider.shutdown(); };

/**
 * @brief Concept satisfied when SettingsApi exposes the perfetto-legacy switch.
 *
 * Per-Traits Settings methods (get_device_filter, get_enabled_metrics, etc.)
 * vary by collector kind (GPU / NIC / CPU) so they are not constrained here.
 */
template <typename Settings>
concept settings_api = requires {
    { Settings::get_use_perfetto_legacy_metrics() } -> std::convertible_to<bool>;
};

/**
 * @brief Concept satisfied when CacheApi exposes the metadata-init hooks.
 *
 * The store_sample signature is collector-specific (different metrics types per
 * device kind), so it is checked at the call-site instantiation rather than in
 * the concept.
 */
template <typename Cache>
concept cache_api = requires {
    Cache::initialize_category_metadata();
    Cache::initialize_tracks_metadata();
};

/**
 * @brief Composite concept: Config exposes the three nested policy aliases
 *        (SettingsApi, PerfettoApi, CacheApi), each satisfying its own concept
 *        where the contract is collector-agnostic.
 *
 * PerfettoApi has no required no-arg static methods at the base level (the
 * init_storage / store_sample / post_process signatures vary by collector
 * kind), so it is constrained only by existence.
 */
template <typename Config>
concept valid_collector_config = requires {
    typename Config::SettingsApi;
    typename Config::PerfettoApi;
    typename Config::CacheApi;
} && settings_api<typename Config::SettingsApi> && cache_api<typename Config::CacheApi>;

}  // namespace rocprofsys::pmc::collectors::base
