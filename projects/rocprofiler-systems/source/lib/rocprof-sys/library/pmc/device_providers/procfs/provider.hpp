// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/device_providers/procfs/drivers/driver.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <unistd.h>
#include <vector>

namespace rocprofsys::pmc::device_providers::procfs
{

/**
 * @brief Procfs device provider for CPU enumeration.
 *
 * Unlike the AMD SMI provider, procfs requires no initialization or
 * shutdown -- the kernel filesystems are always available.
 *
 * Encapsulates driver internals and socket topology. External code
 * accesses devices via get_devices<Device>().
 *
 * @tparam DriverFactory Factory for creating procfs driver instances.
 */
template <typename DriverFactory>
class provider
{
public:
    using driver_t = typename DriverFactory::driver_t;

    provider()
    : m_cpu_count(static_cast<size_t>(std::max(0L, sysconf(_SC_NPROCESSORS_ONLN))))
    , m_driver(DriverFactory::create_driver(m_cpu_count))
    {}

    ~provider() = default;

    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = default;
    provider& operator=(provider&&)      = default;

    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices()
    {
        std::vector<std::shared_ptr<Device>> devices;
        const auto&                          topology = m_driver->get_socket_topology();

        for(const auto& [socket_id, cpu_set] : topology)
        {
            devices.push_back(std::make_shared<Device>(m_driver, socket_id, cpu_set));
        }

        LOG_INFO("Detected {} CPU socket(s), {} online CPUs", topology.size(),
                 m_cpu_count);
        return devices;
    }

    [[nodiscard]] size_t get_cpu_count() const noexcept { return m_cpu_count; }

    void init() {}
    void shutdown() {}

private:
    size_t                    m_cpu_count = 0;
    std::shared_ptr<driver_t> m_driver;
};

/**
 * @brief Factory for creating procfs provider instances.
 *
 * @tparam DriverFactory Factory type for creating procfs driver instances.
 */
template <typename DriverFactory>
struct provider_factory
{
    using provider_t = provider<DriverFactory>;

    static std::shared_ptr<provider_t> create() { return std::make_shared<provider_t>(); }
};

}  // namespace rocprofsys::pmc::device_providers::procfs
