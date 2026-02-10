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

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

#include <sys/resource.h>

namespace rocprofsys
{
namespace pmc
{
namespace drivers
{
namespace procfs
{

/**
 * @brief CPU time counters from /proc/stat for a single CPU core.
 *
 * Values are in jiffies (clock ticks). Use total() and active() to compute
 * CPU load percentage from the delta between two snapshots.
 */
struct cpu_jiffies
{
    uint64_t user    = 0;
    uint64_t nice    = 0;
    uint64_t system  = 0;
    uint64_t idle    = 0;
    uint64_t iowait  = 0;
    uint64_t irq     = 0;
    uint64_t softirq = 0;

    uint64_t total() const noexcept
    {
        return user + nice + system + idle + iowait + irq + softirq;
    }

    uint64_t active() const noexcept { return user + nice + system + irq + softirq; }
};

/**
 * @brief Process-level resource usage snapshot.
 *
 * Collected via getrusage(RUSAGE_SELF) and /proc/self/statm.
 */
struct rusage_snapshot
{
    int64_t page_rss         = 0;  // bytes (physical memory RSS)
    int64_t virt_mem         = 0;  // bytes (virtual memory)
    int64_t peak_rss         = 0;  // bytes (peak memory HWM)
    int64_t context_switches = 0;  // count (voluntary + involuntary)
    int64_t page_faults      = 0;  // count (major + minor)
    int64_t user_mode_time   = 0;  // microseconds
    int64_t kernel_mode_time = 0;  // microseconds
};

/**
 * @brief Driver wrapping Linux procfs and getrusage for CPU metrics.
 *
 * Provides static methods for reading CPU data from /proc/stat, /proc/cpuinfo,
 * /proc/self/statm, and getrusage(). This abstraction enables dependency
 * injection and mocking for unit testing.
 */
struct driver
{
    /**
     * @brief Read per-CPU jiffies from /proc/stat.
     * @return Map of CPU ID to cpu_jiffies. Empty map on failure.
     */
    static std::map<size_t, cpu_jiffies> read_proc_stat()
    {
        std::map<size_t, cpu_jiffies> result;
        std::ifstream                 ifs("/proc/stat");
        if(!ifs.is_open()) return result;

        std::string line;
        while(std::getline(ifs, line))
        {
            // Match lines like "cpu0 ...", "cpu1 ...", skip aggregate "cpu ..."
            if(line.size() < 4 || line.substr(0, 3) != "cpu" || !std::isdigit(line[3]))
                continue;

            std::istringstream iss(line);
            std::string        cpu_label;
            cpu_jiffies        jiffies;

            iss >> cpu_label >> jiffies.user >> jiffies.nice >> jiffies.system >>
                jiffies.idle >> jiffies.iowait >> jiffies.irq >> jiffies.softirq;

            size_t cpu_id = std::stoull(cpu_label.substr(3));
            result[cpu_id] = jiffies;
        }
        return result;
    }

    /**
     * @brief Read per-CPU frequencies from /proc/cpuinfo.
     * @return Map of CPU ID to frequency in MHz. Empty map on failure.
     */
    static std::map<size_t, float> read_cpu_frequencies()
    {
        std::map<size_t, float> result;
        std::ifstream           ifs("/proc/cpuinfo");
        if(!ifs.is_open()) return result;

        size_t      current_processor = 0;
        bool        has_processor     = false;
        std::string line;

        while(std::getline(ifs, line))
        {
            if(line.substr(0, 9) == "processor")
            {
                auto pos = line.find(':');
                if(pos != std::string::npos)
                {
                    current_processor = std::stoull(line.substr(pos + 1));
                    has_processor     = true;
                }
            }
            else if(has_processor && line.substr(0, 7) == "cpu MHz")
            {
                auto pos = line.find(':');
                if(pos != std::string::npos)
                {
                    result[current_processor] = std::stof(line.substr(pos + 1));
                }
            }
        }
        return result;
    }

    /**
     * @brief Read process-level resource usage via getrusage and /proc/self/statm.
     * @return rusage_snapshot with current process metrics.
     */
    static rusage_snapshot read_rusage()
    {
        rusage_snapshot snap;

        // getrusage for timing, context switches, page faults, peak RSS
        struct rusage usage;
        std::memset(&usage, 0, sizeof(usage));
        if(getrusage(RUSAGE_SELF, &usage) == 0)
        {
            snap.peak_rss = static_cast<int64_t>(usage.ru_maxrss) *
                            1024;  // ru_maxrss is in KB on Linux
            snap.context_switches =
                static_cast<int64_t>(usage.ru_nvcsw + usage.ru_nivcsw);
            snap.page_faults =
                static_cast<int64_t>(usage.ru_majflt + usage.ru_minflt);
            snap.user_mode_time = static_cast<int64_t>(usage.ru_utime.tv_sec) * 1000000 +
                                  static_cast<int64_t>(usage.ru_utime.tv_usec);
            snap.kernel_mode_time =
                static_cast<int64_t>(usage.ru_stime.tv_sec) * 1000000 +
                static_cast<int64_t>(usage.ru_stime.tv_usec);
        }

        // /proc/self/statm for RSS and virtual memory
        // Fields: size resident shared text lib data dt (all in pages)
        std::ifstream statm("/proc/self/statm");
        if(statm.is_open())
        {
            size_t virt_pages = 0;
            size_t rss_pages  = 0;
            statm >> virt_pages >> rss_pages;
            long page_size = sysconf(_SC_PAGESIZE);
            snap.page_rss  = static_cast<int64_t>(rss_pages) * page_size;
            snap.virt_mem  = static_cast<int64_t>(virt_pages) * page_size;
        }

        return snap;
    }

    /**
     * @brief Get the number of online CPUs.
     * @return Number of online CPUs via sysconf.
     */
    static size_t get_cpu_count()
    {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return (n > 0) ? static_cast<size_t>(n) : 0;
    }
};

/**
 * @brief Factory for creating procfs driver instances.
 *
 * Enables dependency injection for testing with mock drivers.
 */
struct driver_factory
{
    using driver_t = driver;

    static std::shared_ptr<driver_t> create_driver()
    {
        return std::make_shared<driver_t>();
    }
};

}  // namespace procfs
}  // namespace drivers
}  // namespace pmc
}  // namespace rocprofsys
