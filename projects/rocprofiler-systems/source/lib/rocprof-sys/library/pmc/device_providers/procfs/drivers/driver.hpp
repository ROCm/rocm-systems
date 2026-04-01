// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

namespace rocprofsys::pmc::drivers::procfs
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

    [[nodiscard]] uint64_t total() const noexcept
    {
        return user + nice + system + idle + iowait + irq + softirq;
    }

    [[nodiscard]] uint64_t active() const noexcept
    {
        return user + nice + system + irq + softirq;
    }
};

/**
 * @brief Process-level resource usage snapshot.
 *
 * Collected via getrusage(RUSAGE_SELF) and /proc/self/statm.
 */
struct rusage_snapshot
{
    int64_t page_rss         = 0;  // bytes
    int64_t virt_mem         = 0;  // bytes
    int64_t peak_rss         = 0;  // bytes
    int64_t context_switches = 0;  // count (voluntary + involuntary)
    int64_t page_faults      = 0;  // count (major + minor)
    int64_t user_mode_time   = 0;  // microseconds
    int64_t kernel_mode_time = 0;  // microseconds
};

/**
 * @brief Driver wrapping Linux procfs and getrusage for CPU metrics.
 *
 * Performance-optimized: persistent file descriptors, lseek+read,
 * pre-allocated buffers, zero-copy string_view parsing.
 */
class driver
{
public:
    driver()
    : m_proc_stat_fd(-1)
    , m_proc_cpuinfo_fd(-1)
    , m_proc_statm_fd(-1)
    , m_cpu_count(0)
    {
        m_stat_buffer.resize(16384);
        m_cpuinfo_buffer.resize(32768);
        m_statm_buffer.resize(256);
    }

    ~driver() noexcept { close_fds(); }

    driver(const driver&)            = delete;
    driver& operator=(const driver&) = delete;

    driver(driver&& other) noexcept
    : m_proc_stat_fd(other.m_proc_stat_fd)
    , m_proc_cpuinfo_fd(other.m_proc_cpuinfo_fd)
    , m_proc_statm_fd(other.m_proc_statm_fd)
    , m_cpu_count(other.m_cpu_count)
    , m_stat_buffer(std::move(other.m_stat_buffer))
    , m_cpuinfo_buffer(std::move(other.m_cpuinfo_buffer))
    , m_statm_buffer(std::move(other.m_statm_buffer))
    {
        other.m_proc_stat_fd    = -1;
        other.m_proc_cpuinfo_fd = -1;
        other.m_proc_statm_fd   = -1;
    }

    driver& operator=(driver&& other) noexcept
    {
        if(this != &other)
        {
            close_fds();
            m_proc_stat_fd    = other.m_proc_stat_fd;
            m_proc_cpuinfo_fd = other.m_proc_cpuinfo_fd;
            m_proc_statm_fd   = other.m_proc_statm_fd;
            m_cpu_count       = other.m_cpu_count;
            m_stat_buffer     = std::move(other.m_stat_buffer);
            m_cpuinfo_buffer  = std::move(other.m_cpuinfo_buffer);
            m_statm_buffer    = std::move(other.m_statm_buffer);

            other.m_proc_stat_fd    = -1;
            other.m_proc_cpuinfo_fd = -1;
            other.m_proc_statm_fd   = -1;
        }
        return *this;
    }

    /**
     * @brief Read per-CPU jiffies from /proc/stat.
     * @return Map of CPU ID to cpu_jiffies. Empty map on failure.
     */
    std::map<size_t, cpu_jiffies> read_proc_stat()
    {
        std::map<size_t, cpu_jiffies> result;

        if(m_proc_stat_fd < 0)
        {
            m_proc_stat_fd = ::open("/proc/stat", O_RDONLY);
            if(m_proc_stat_fd < 0) return result;
        }

        ::lseek(m_proc_stat_fd, 0, SEEK_SET);
        auto bytes =
            ::read(m_proc_stat_fd, m_stat_buffer.data(), m_stat_buffer.size() - 1);
        if(bytes <= 0) return result;

        m_stat_buffer[static_cast<size_t>(bytes)] = '\0';
        parse_proc_stat(
            std::string_view(m_stat_buffer.data(), static_cast<size_t>(bytes)), result);
        return result;
    }

    /**
     * @brief Read per-CPU frequencies from /proc/cpuinfo.
     * @return Map of CPU ID to frequency in MHz. Empty map on failure.
     */
    std::map<size_t, float> read_cpu_frequencies()
    {
        std::map<size_t, float> result;

        if(m_proc_cpuinfo_fd < 0)
        {
            m_proc_cpuinfo_fd = ::open("/proc/cpuinfo", O_RDONLY);
            if(m_proc_cpuinfo_fd < 0) return result;
        }

        ::lseek(m_proc_cpuinfo_fd, 0, SEEK_SET);
        auto bytes = ::read(m_proc_cpuinfo_fd, m_cpuinfo_buffer.data(),
                            m_cpuinfo_buffer.size() - 1);
        if(bytes <= 0) return result;

        m_cpuinfo_buffer[static_cast<size_t>(bytes)] = '\0';
        parse_cpuinfo(
            std::string_view(m_cpuinfo_buffer.data(), static_cast<size_t>(bytes)),
            result);
        return result;
    }

    /**
     * @brief Read process-level resource usage via getrusage and /proc/self/statm.
     * @return rusage_snapshot with current process metrics.
     */
    rusage_snapshot read_rusage()
    {
        rusage_snapshot snap;

        struct rusage usage;
        std::memset(&usage, 0, sizeof(usage));
        if(getrusage(RUSAGE_SELF, &usage) == 0)
        {
            // ru_maxrss is in KB on Linux
            snap.peak_rss = static_cast<int64_t>(usage.ru_maxrss) * 1024;
            snap.context_switches =
                static_cast<int64_t>(usage.ru_nvcsw + usage.ru_nivcsw);
            snap.page_faults    = static_cast<int64_t>(usage.ru_majflt + usage.ru_minflt);
            snap.user_mode_time = static_cast<int64_t>(usage.ru_utime.tv_sec) * 1000000 +
                                  static_cast<int64_t>(usage.ru_utime.tv_usec);
            snap.kernel_mode_time =
                static_cast<int64_t>(usage.ru_stime.tv_sec) * 1000000 +
                static_cast<int64_t>(usage.ru_stime.tv_usec);
        }

        if(m_proc_statm_fd < 0)
        {
            m_proc_statm_fd = ::open("/proc/self/statm", O_RDONLY);
            if(m_proc_statm_fd < 0) return snap;
        }

        ::lseek(m_proc_statm_fd, 0, SEEK_SET);
        auto bytes =
            ::read(m_proc_statm_fd, m_statm_buffer.data(), m_statm_buffer.size() - 1);
        if(bytes > 0)
        {
            m_statm_buffer[static_cast<size_t>(bytes)] = '\0';
            parse_statm(
                std::string_view(m_statm_buffer.data(), static_cast<size_t>(bytes)),
                snap);
        }

        return snap;
    }

    /**
     * @brief Get the number of online CPUs (cached after first call).
     */
    size_t get_cpu_count()
    {
        if(m_cpu_count == 0)
        {
            long n      = sysconf(_SC_NPROCESSORS_ONLN);
            m_cpu_count = (n > 0) ? static_cast<size_t>(n) : 0;
        }
        return m_cpu_count;
    }

private:
    void close_fds() noexcept
    {
        if(m_proc_stat_fd >= 0)
        {
            ::close(m_proc_stat_fd);
            m_proc_stat_fd = -1;
        }
        if(m_proc_cpuinfo_fd >= 0)
        {
            ::close(m_proc_cpuinfo_fd);
            m_proc_cpuinfo_fd = -1;
        }
        if(m_proc_statm_fd >= 0)
        {
            ::close(m_proc_statm_fd);
            m_proc_statm_fd = -1;
        }
    }

    static void parse_proc_stat(std::string_view               content,
                                std::map<size_t, cpu_jiffies>& result)
    {
        size_t line_start = 0;
        while(line_start < content.size())
        {
            size_t line_end = content.find('\n', line_start);
            if(line_end == std::string_view::npos) line_end = content.size();
            auto line = content.substr(line_start, line_end - line_start);

            if(line.size() >= 4 && line.substr(0, 3) == "cpu" && std::isdigit(line[3]))
            {
                size_t space = line.find(' ');
                if(space == std::string_view::npos)
                {
                    line_start = line_end + 1;
                    continue;
                }

                size_t cpu_id = 0;
                auto [ptr, ec] =
                    std::from_chars(line.data() + 3, line.data() + space, cpu_id);
                if(ec != std::errc())
                {
                    line_start = line_end + 1;
                    continue;
                }

                cpu_jiffies j;
                uint64_t*   fields[] = { &j.user,   &j.nice, &j.system, &j.idle,
                                         &j.iowait, &j.irq,  &j.softirq };
                size_t      pos      = space + 1;

                for(size_t i = 0; i < 7 && pos < line.size(); ++i)
                {
                    while(pos < line.size() && std::isspace(line[pos]))
                        ++pos;
                    auto [p, e] = std::from_chars(line.data() + pos,
                                                  line.data() + line.size(), *fields[i]);
                    if(e != std::errc()) break;
                    pos = static_cast<size_t>(p - line.data());
                }
                result[cpu_id] = j;
            }
            line_start = line_end + 1;
        }
    }

    static void parse_cpuinfo(std::string_view content, std::map<size_t, float>& result)
    {
        size_t current_cpu = 0;
        bool   has_cpu     = false;
        size_t line_start  = 0;

        while(line_start < content.size())
        {
            size_t line_end = content.find('\n', line_start);
            if(line_end == std::string_view::npos) line_end = content.size();
            auto line = content.substr(line_start, line_end - line_start);

            if(line.size() > 9 && line.substr(0, 9) == "processor")
            {
                size_t colon = line.find(':');
                if(colon != std::string_view::npos)
                {
                    size_t pos = colon + 1;
                    while(pos < line.size() && std::isspace(line[pos]))
                        ++pos;
                    auto [p, e] = std::from_chars(line.data() + pos,
                                                  line.data() + line.size(), current_cpu);
                    if(e == std::errc()) has_cpu = true;
                }
            }
            else if(has_cpu && line.size() > 7 && line.substr(0, 7) == "cpu MHz")
            {
                size_t colon = line.find(':');
                if(colon != std::string_view::npos)
                {
                    size_t pos = colon + 1;
                    while(pos < line.size() && std::isspace(line[pos]))
                        ++pos;
                    size_t end_pos = line.find_first_of(" \t\r\n", pos);
                    if(end_pos == std::string_view::npos) end_pos = line.size();
                    std::string temp(line.data() + pos, end_pos - pos);
                    char*       end  = nullptr;
                    float       freq = std::strtof(temp.c_str(), &end);
                    if(end != temp.c_str()) result[current_cpu] = freq;
                }
            }
            line_start = line_end + 1;
        }
    }

    static void parse_statm(std::string_view content, rusage_snapshot& snap)
    {
        size_t virt_pages = 0;
        size_t rss_pages  = 0;
        size_t pos        = 0;
        while(pos < content.size() && std::isspace(content[pos]))
            ++pos;

        auto [p1, e1] = std::from_chars(content.data() + pos,
                                        content.data() + content.size(), virt_pages);
        if(e1 != std::errc()) return;
        pos = static_cast<size_t>(p1 - content.data());
        while(pos < content.size() && std::isspace(content[pos]))
            ++pos;

        auto [p2, e2] = std::from_chars(content.data() + pos,
                                        content.data() + content.size(), rss_pages);
        if(e2 != std::errc()) return;

        long page_size = sysconf(_SC_PAGESIZE);
        snap.page_rss  = static_cast<int64_t>(rss_pages) * page_size;
        snap.virt_mem  = static_cast<int64_t>(virt_pages) * page_size;
    }

    int               m_proc_stat_fd;
    int               m_proc_cpuinfo_fd;
    int               m_proc_statm_fd;
    size_t            m_cpu_count;
    std::vector<char> m_stat_buffer;
    std::vector<char> m_cpuinfo_buffer;
    std::vector<char> m_statm_buffer;
};

/**
 * @brief Factory for creating procfs driver instances.
 */
struct driver_factory
{
    using driver_t = driver;

    static std::shared_ptr<driver_t> create_driver()
    {
        return std::make_shared<driver_t>();
    }
};

}  // namespace rocprofsys::pmc::drivers::procfs
