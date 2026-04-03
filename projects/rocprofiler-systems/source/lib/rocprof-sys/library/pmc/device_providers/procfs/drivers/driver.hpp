// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::drivers::procfs
{
// /proc/stat: ~120 bytes per CPU line. Default 16 KB covers ~128 CPUs.
static constexpr size_t DEFAULT_STAT_BUFFER_SIZE = 16384;
// /proc/self/statm: 7 space-separated numbers, always < 256 bytes.
static constexpr size_t STATM_BUFFER_SIZE = 256;
// Approximate bytes per /proc/stat CPU line (for dynamic sizing).
static constexpr size_t BYTES_PER_STAT_LINE = 120;
// sysfs frequency file: value in kHz, always < 32 bytes.
static constexpr size_t SYSFS_FREQ_BUFFER_SIZE = 32;
// ru_maxrss is in KB on Linux.
static constexpr int64_t KB_TO_BYTES = 1024;
// Microseconds per second (for timeval conversion).
static constexpr int64_t US_PER_SECOND = 1'000'000;
// kHz to MHz conversion for sysfs scaling_cur_freq.
static constexpr float KHZ_TO_MHZ = 0.001f;

class unique_fd
{
public:
    unique_fd() noexcept = default;
    explicit unique_fd(int fd) noexcept
    : m_fd(fd)
    {}
    ~unique_fd() noexcept
    {
        if(m_fd >= 0) ::close(m_fd);
    }

    unique_fd(unique_fd&& other) noexcept
    : m_fd(std::exchange(other.m_fd, -1))
    {}

    unique_fd& operator=(unique_fd&& other) noexcept
    {
        if(this != &other)
        {
            if(m_fd >= 0) ::close(m_fd);
            m_fd = std::exchange(other.m_fd, -1);
        }
        return *this;
    }

    unique_fd(const unique_fd&)            = delete;
    unique_fd& operator=(const unique_fd&) = delete;

    [[nodiscard]] int      get() const noexcept { return m_fd; }
    [[nodiscard]] explicit operator bool() const noexcept { return m_fd >= 0; }

    void reset(int fd = -1) noexcept
    {
        if(m_fd >= 0) ::close(m_fd);
        m_fd = fd;
    }

private:
    int m_fd = -1;
};

constexpr bool
starts_with(std::string_view str, std::string_view prefix) noexcept
{
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}

inline std::string_view
ltrim(std::string_view str) noexcept
{
    const auto* const it = std::find_if_not(
        str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); });
    return str.substr(static_cast<size_t>(it - str.begin()));
}

template <typename Fn>
void
for_each_line(std::string_view content, Fn&& fn)
{
    size_t line_start = 0;
    while(line_start < content.size())
    {
        auto line_end = content.find('\n', line_start);
        if(line_end == std::string_view::npos) line_end = content.size();
        fn(content.substr(line_start, line_end - line_start));
        line_start = line_end + 1;
    }
}

struct cpu_jiffies
{
    uint64_t user    = 0;
    uint64_t nice    = 0;
    uint64_t system  = 0;
    uint64_t idle    = 0;
    uint64_t iowait  = 0;
    uint64_t irq     = 0;
    uint64_t softirq = 0;

    [[nodiscard]] constexpr uint64_t total() const noexcept
    {
        return user + nice + system + idle + iowait + irq + softirq;
    }

    [[nodiscard]] constexpr uint64_t active() const noexcept
    {
        return user + nice + system + irq + softirq;
    }
};

struct rusage_snapshot
{
    int64_t page_rss         = 0;
    int64_t virt_mem         = 0;
    int64_t peak_rss         = 0;
    int64_t context_switches = 0;
    int64_t page_faults      = 0;
    int64_t user_mode_time   = 0;
    int64_t kernel_mode_time = 0;
};

// virt_mem (first) and page_rss (second) from /proc/self/statm, in bytes.
using statm_data = std::pair<int64_t, int64_t>;

/**
 * @brief Driver wrapping Linux procfs/sysfs and getrusage for CPU metrics.
 *
 * Persistent file descriptors via unique_fd (RAII), lseek+read,
 * pre-allocated buffers, zero-copy string_view parsing.
 * CPU frequency read from sysfs scaling_cur_freq (per-CPU FDs).
 */
class driver
{
public:
    /**
     * @param cpu_count Number of online CPUs. Sizes buffers and opens
     *        per-CPU sysfs FDs for frequency reading.
     */
    explicit driver(size_t cpu_count)
    : m_stat_buffer(
          std::max(DEFAULT_STAT_BUFFER_SIZE, (cpu_count * BYTES_PER_STAT_LINE) + 256))
    , m_statm_buffer(STATM_BUFFER_SIZE)
    , m_freq_buffer(SYSFS_FREQ_BUFFER_SIZE)
    {
        for(size_t i = 0; i < cpu_count; ++i)
        {
            const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                              "/cpufreq/scaling_cur_freq";
            unique_fd fd{ ::open(path.c_str(), O_RDONLY) };
            if(fd) m_sysfs_freq_fds.emplace(i, std::move(fd));
        }
        m_use_sysfs_freq = !m_sysfs_freq_fds.empty();
    }

    // Rule of Zero: unique_fd and map<.., unique_fd> handle cleanup.

    [[nodiscard]] std::map<size_t, cpu_jiffies> read_proc_stat()
    {
        const auto content = read_file(m_proc_stat_fd, "/proc/stat", m_stat_buffer);
        if(content.empty()) return {};
        return parse_proc_stat(content);
    }

    [[nodiscard]] std::map<size_t, float> read_cpu_frequencies()
    {
        if(m_use_sysfs_freq) return read_sysfs_frequencies();
        return {};
    }

    [[nodiscard]] rusage_snapshot read_rusage()
    {
        rusage_snapshot snap;

        struct rusage usage = {};
        if(getrusage(RUSAGE_SELF, &usage) == 0)
        {
            snap.peak_rss = static_cast<int64_t>(usage.ru_maxrss) * KB_TO_BYTES;
            snap.context_switches =
                static_cast<int64_t>(usage.ru_nvcsw + usage.ru_nivcsw);
            snap.page_faults = static_cast<int64_t>(usage.ru_majflt + usage.ru_minflt);
            snap.user_mode_time =
                static_cast<int64_t>(usage.ru_utime.tv_sec) * US_PER_SECOND +
                static_cast<int64_t>(usage.ru_utime.tv_usec);
            snap.kernel_mode_time =
                static_cast<int64_t>(usage.ru_stime.tv_sec) * US_PER_SECOND +
                static_cast<int64_t>(usage.ru_stime.tv_usec);
        }

        const auto content =
            read_file(m_proc_statm_fd, "/proc/self/statm", m_statm_buffer);
        if(!content.empty())
        {
            if(const auto data = parse_statm(content))
            {
                snap.virt_mem = data->first;
                snap.page_rss = data->second;
            }
        }

        return snap;
    }

    [[nodiscard]] size_t get_cpu_count() const noexcept { return m_cpu_count; }

private:
    [[nodiscard]] static std::string_view read_file(unique_fd& fd, const char* path,
                                                    std::vector<char>& buffer)
    {
        if(!fd)
        {
            fd.reset(::open(path, O_RDONLY));
            if(!fd) return {};
        }
        if(::lseek(fd.get(), 0, SEEK_SET) == static_cast<off_t>(-1)) return {};

        const auto bytes = ::read(fd.get(), buffer.data(), buffer.size());
        if(bytes <= 0) return {};

        return { buffer.data(), static_cast<size_t>(bytes) };
    }

    [[nodiscard]] std::map<size_t, float> read_sysfs_frequencies()
    {
        std::map<size_t, float> result;
        for(auto& [cpu_id, fd] : m_sysfs_freq_fds)
        {
            if(::lseek(fd.get(), 0, SEEK_SET) == static_cast<off_t>(-1)) continue;
            const auto bytes =
                ::read(fd.get(), m_freq_buffer.data(), m_freq_buffer.size());
            if(bytes <= 0) continue;

            unsigned long khz = 0;
            const auto [p, e] =
                std::from_chars(m_freq_buffer.data(), m_freq_buffer.data() + bytes, khz);
            if(e == std::errc()) result[cpu_id] = static_cast<float>(khz) * KHZ_TO_MHZ;
        }
        return result;
    }

    [[nodiscard]] static std::map<size_t, cpu_jiffies> parse_proc_stat(
        std::string_view content)
    {
        std::map<size_t, cpu_jiffies> result;

        for_each_line(content, [&](std::string_view line) {
            if(!starts_with(line, "cpu") || line.size() < 4 ||
               !std::isdigit(static_cast<unsigned char>(line[3])))
                return;

            const auto space = line.find(' ');
            if(space == std::string_view::npos) return;

            size_t cpu_id = 0;
            const auto [ptr, ec] =
                std::from_chars(line.data() + 3, line.data() + space, cpu_id);
            if(ec != std::errc()) return;

            cpu_jiffies j;
            uint64_t*   fields[]  = { &j.user,   &j.nice, &j.system, &j.idle,
                                      &j.iowait, &j.irq,  &j.softirq };
            auto        remaining = ltrim(line.substr(space));

            for(size_t i = 0; i < 7 && !remaining.empty(); ++i)
            {
                const auto [p, e] = std::from_chars(
                    remaining.data(), remaining.data() + remaining.size(), *fields[i]);
                if(e != std::errc()) break;
                remaining = ltrim(
                    { p, static_cast<size_t>(remaining.data() + remaining.size() - p) });
            }
            result[cpu_id] = j;
        });

        return result;
    }

    [[nodiscard]] static std::optional<statm_data> parse_statm(std::string_view content)
    {
        static const long page_size = sysconf(_SC_PAGESIZE);

        auto remaining = ltrim(content);

        size_t virt_pages   = 0;
        const auto [p1, e1] = std::from_chars(
            remaining.data(), remaining.data() + remaining.size(), virt_pages);
        if(e1 != std::errc()) return std::nullopt;

        remaining =
            ltrim({ p1, static_cast<size_t>(remaining.data() + remaining.size() - p1) });

        size_t rss_pages    = 0;
        const auto [p2, e2] = std::from_chars(
            remaining.data(), remaining.data() + remaining.size(), rss_pages);
        if(e2 != std::errc()) return std::nullopt;

        return statm_data{ static_cast<int64_t>(virt_pages) * page_size,
                           static_cast<int64_t>(rss_pages) * page_size };
    }

    unique_fd                   m_proc_stat_fd;
    unique_fd                   m_proc_statm_fd;
    std::map<size_t, unique_fd> m_sysfs_freq_fds;
    bool                        m_use_sysfs_freq = false;
    size_t                      m_cpu_count      = 0;
    std::vector<char>           m_stat_buffer;
    std::vector<char>           m_statm_buffer;
    std::vector<char>           m_freq_buffer;
};

/**
 * @brief Factory for creating procfs driver instances.
 */
struct driver_factory
{
    using driver_t = driver;

    static std::shared_ptr<driver_t> create_driver(size_t cpu_count)
    {
        return std::make_shared<driver_t>(cpu_count);
    }
};

}  // namespace rocprofsys::pmc::drivers::procfs
