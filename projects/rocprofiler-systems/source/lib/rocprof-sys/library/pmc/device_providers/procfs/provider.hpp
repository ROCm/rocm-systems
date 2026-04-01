// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <cstddef>
#include <memory>

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
    : m_driver(DriverFactory::create_driver())
    , m_cpu_count(m_driver->get_cpu_count())
    {}

    ~provider() = default;

    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = default;
    provider& operator=(provider&&)      = default;

    [[nodiscard]] std::shared_ptr<driver_t> get_driver() const noexcept
    {
        return m_driver;
    }

    [[nodiscard]] size_t get_cpu_count() const noexcept { return m_cpu_count; }

    void init() {}
    void shutdown() {}

private:
    std::shared_ptr<driver_t> m_driver;
    size_t                    m_cpu_count = 0;
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
