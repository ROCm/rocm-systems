// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <unistd.h>

namespace rocprofsys::pmc::device_providers::procfs
{

/**
 * @brief Procfs device provider for CPU enumeration.
 *
 * Unlike the AMD SMI provider, procfs requires no initialization or
 * shutdown -- the kernel filesystems are always available.
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

    [[nodiscard]] const std::shared_ptr<driver_t>& get_driver() const noexcept
    {
        return m_driver;
    }

    [[nodiscard]] size_t get_cpu_count() const noexcept { return m_cpu_count; }

    [[nodiscard]] size_t get_socket_count() const noexcept
    {
        return m_driver->get_socket_count();
    }

    [[nodiscard]] const drivers::procfs::socket_topology_t& get_socket_topology()
        const noexcept
    {
        return m_driver->get_socket_topology();
    }

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
