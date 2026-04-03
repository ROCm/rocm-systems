// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/device_providers/procfs/drivers/driver.hpp"

#include <gmock/gmock.h>

#include <map>
#include <memory>
#include <utility>

namespace rocprofsys::pmc::drivers::procfs::testing
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
     * @brief Set up default mock behaviors.
     *
     * Configures: 4 CPUs with 2000 MHz, moderate jiffies, basic process usage.
     */
    void set_up_defaults()
    {
        using ::testing::Return;

        ON_CALL(*this, get_cpu_count()).WillByDefault(Return(4));

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

        std::map<size_t, float> default_freqs;
        for(size_t i = 0; i < 4; ++i)
        {
            default_freqs[i] = 2000.0f;
        }
        ON_CALL(*this, read_cpu_frequencies()).WillByDefault(Return(default_freqs));

        rusage_snapshot default_rusage;
        default_rusage.page_rss         = 50 * 1024 * 1024;
        default_rusage.virt_mem         = 200 * 1024 * 1024;
        default_rusage.peak_rss         = 60 * 1024 * 1024;
        default_rusage.context_switches = 1000;
        default_rusage.page_faults      = 500;
        default_rusage.user_mode_time   = 5000000;
        default_rusage.kernel_mode_time = 1000000;
        ON_CALL(*this, read_rusage()).WillByDefault(Return(default_rusage));
    }
};

/**
 * @brief Factory for creating and injecting mock driver instances in tests.
 */
struct mock_driver_factory
{
    using driver_t = mock_driver;

    static std::shared_ptr<driver_t> s_mock_driver;

    static std::shared_ptr<driver_t> create_driver([[maybe_unused]] size_t cpu_count = 0)
    {
        return s_mock_driver;
    }

    static void set_mock_driver(std::shared_ptr<driver_t> driver)
    {
        s_mock_driver = std::move(driver);
    }
};

inline std::shared_ptr<mock_driver> mock_driver_factory::s_mock_driver = nullptr;

}  // namespace rocprofsys::pmc::drivers::procfs::testing
