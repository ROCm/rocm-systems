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

#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <cstddef>
#include <memory>

namespace rocprofsys
{
namespace pmc
{
namespace device_providers
{
namespace procfs
{

/**
 * @brief Procfs device provider for CPU enumeration.
 *
 * Provides access to the procfs driver and CPU enumeration services.
 * Unlike the AMD SMI provider, procfs requires no initialization or
 * shutdown — the kernel filesystems are always available.
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

    // Non-copyable, but movable
    provider(const provider&)            = delete;
    provider& operator=(const provider&) = delete;
    provider(provider&&)                 = default;
    provider& operator=(provider&&)      = default;

    /**
     * @brief Get driver instance.
     * @return Shared pointer to the procfs driver.
     */
    [[nodiscard]] std::shared_ptr<driver_t> get_driver() const noexcept
    {
        return m_driver;
    }

    /**
     * @brief Get the number of online CPUs.
     * @return CPU count cached at construction time.
     */
    [[nodiscard]] size_t get_cpu_count() const noexcept { return m_cpu_count; }

    /**
     * @brief No-op init (procfs is always available).
     */
    void init() {}

    /**
     * @brief No-op shutdown (procfs has no cleanup).
     */
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

    static std::shared_ptr<provider_t> create()
    {
        return std::make_shared<provider_t>();
    }
};

}  // namespace procfs
}  // namespace device_providers
}  // namespace pmc
}  // namespace rocprofsys
