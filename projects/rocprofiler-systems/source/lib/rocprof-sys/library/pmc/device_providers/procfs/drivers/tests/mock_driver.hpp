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

#include <gmock/gmock.h>

#include <map>
#include <memory>
#include <utility>

namespace rocprofsys
{
namespace pmc
{
namespace drivers
{
namespace procfs
{
namespace testing
{

/**
 * @brief Mock implementation of procfs driver for unit testing.
 *
 * Used by device and collector tests to inject synthetic CPU data
 * without touching the filesystem.
 */
class mock_driver
{
public:
    MOCK_METHOD((std::map<size_t, cpu_jiffies>), read_proc_stat, ());
    MOCK_METHOD((std::map<size_t, float>), read_cpu_frequencies, ());
    MOCK_METHOD(rusage_snapshot, read_rusage, ());
    MOCK_METHOD(size_t, get_cpu_count, ());

    /**
     * @brief Set up default mock behaviors for common operations.
     *
     * Configures the mock to return 4 CPUs with reasonable default values:
     * - 4 CPUs (0-3) with 2000 MHz frequencies
     * - Moderate jiffies (10000 total, ~5% load)
     * - Basic process resource usage
     */
    void set_up_defaults()
    {
        using ::testing::Return;

        // Default: 4 CPUs
        ON_CALL(*this, get_cpu_count()).WillByDefault(Return(4));

        // Default jiffies: ~5% load (500 active out of 10000 total)
        std::map<size_t, cpu_jiffies> default_jiffies;
        for(size_t i = 0; i < 4; ++i)
        {
            cpu_jiffies j;
            j.user             = 200;
            j.nice             = 10;
            j.system           = 150;
            j.idle             = 9500;
            j.iowait           = 50;
            j.irq              = 30;
            j.softirq          = 60;
            default_jiffies[i] = j;
        }
        ON_CALL(*this, read_proc_stat()).WillByDefault(Return(default_jiffies));

        // Default frequencies: 2000 MHz per CPU
        std::map<size_t, float> default_freqs;
        for(size_t i = 0; i < 4; ++i)
        {
            default_freqs[i] = 2000.0f;
        }
        ON_CALL(*this, read_cpu_frequencies()).WillByDefault(Return(default_freqs));

        // Default rusage
        rusage_snapshot default_rusage;
        default_rusage.page_rss         = 50 * 1024 * 1024;   // 50 MB
        default_rusage.virt_mem         = 200 * 1024 * 1024;  // 200 MB
        default_rusage.peak_rss         = 60 * 1024 * 1024;   // 60 MB
        default_rusage.context_switches = 1000;
        default_rusage.page_faults      = 500;
        default_rusage.user_mode_time   = 5000000;  // 5 seconds
        default_rusage.kernel_mode_time = 1000000;  // 1 second
        ON_CALL(*this, read_rusage()).WillByDefault(Return(default_rusage));
    }
};

/**
 * @brief Factory for creating and injecting mock driver instances in tests.
 *
 * Allows tests to inject a mock_driver instance that will be
 * used by the code under test via set_mock_driver().
 */
struct mock_driver_factory
{
    using driver_t = mock_driver;

    static std::shared_ptr<driver_t> s_mock_driver;

    static std::shared_ptr<driver_t> create_driver() { return s_mock_driver; }

    static void set_mock_driver(std::shared_ptr<driver_t> driver)
    {
        s_mock_driver = std::move(driver);
    }
};

/// Global mock driver instance shared across tests
inline std::shared_ptr<mock_driver> mock_driver_factory::s_mock_driver = nullptr;

}  // namespace testing
}  // namespace procfs
}  // namespace drivers
}  // namespace pmc
}  // namespace rocprofsys
