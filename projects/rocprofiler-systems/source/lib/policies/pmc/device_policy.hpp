// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace rocprofsys::policies::pmc
{
/** A device that reports its index (GPU, CPU socket or NIC index). */
template <typename Device>
concept indexed_device = requires(const Device& device) {
    { device.get_index() } -> std::convertible_to<std::size_t>;
};

/** A GPU read through AMD SMI; support_reading() is the reading taken at device init. */
template <typename Device, typename Enabled>
concept gpu_smi_device =
    indexed_device<Device> &&
    requires(Device& device, const Enabled& enabled, std::uint64_t timestamp) {
        device.get_metrics(enabled, timestamp);
        device.support_reading();
    };

/** A NIC read through AMD SMI. */
template <typename Device>
concept nic_device =
    indexed_device<Device> && requires(Device& device) { device.get_nic_metrics(); };

/** A CPU socket: one reading covers the process and its monitored cores. */
template <typename Device, typename Enabled>
concept cpu_device =
    indexed_device<Device> && requires(Device& device, const Enabled& enabled) {
        device.get_cpu_metrics(enabled);
        device.get_monitored_cpus();
    };

/** A GPU whose hipFile I/O statistics are read. */
template <typename Device, typename Enabled>
concept hipfile_device =
    indexed_device<Device> &&
    requires(Device& device, const Enabled& enabled, std::uint64_t timestamp) {
        device.get_metrics(enabled, timestamp);
    };

/** A GPU whose hardware counters are sampled through the profiling SDK. */
template <typename Device, typename Enabled>
concept counter_device =
    indexed_device<Device> &&
    requires(Device& device, const Enabled& enabled, std::uint64_t timestamp) {
        device.sample_metrics(enabled, timestamp);
        device.get_counter_metadata();
    };

/** A provider that has to be shut down by its last user. */
template <typename Provider>
concept stoppable_provider = requires(Provider& provider) { provider.shutdown(); };
}  // namespace rocprofsys::policies::pmc
