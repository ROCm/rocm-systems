// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/uuid_v7.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/sha256.hpp"

#include <fmt/format.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
uint64_t
get_process_start_ticks_since_boot(pid_t pid)
{
    auto line = std::string{};

    // Read the stat file
    if(auto stat_file = std::ifstream{fmt::format("/proc/{}/stat", pid)}; !stat_file.is_open())
    {
        ROCP_CI_LOG(WARNING) << fmt::format("failed to open /proc/{}/stat for process start time",
                                            pid);
        return 0;
    }
    else
    {
        // Read entire line
        std::getline(stat_file, line);
    }

    // Locate the end of the comm field (")")
    size_t rparen = line.rfind(')');
    if(rparen == std::string::npos)
    {
        ROCP_CI_LOG(WARNING) << fmt::format("Malformed stat file for pid {}", pid);
        return 0;
    }

    // Tokenize fields after ") "
    auto iss   = std::istringstream{line.substr(rparen + 2)};
    auto token = std::string{};
    // Skip fields 3 through 21
    for(int i = 0; i < 20; ++i)
    {
        if(!(iss >> token))
        {
            ROCP_CI_LOG(WARNING) << fmt::format("Unexpected end of /proc/{}/stat", pid);
            return 0;
        }
    }

    // Field 22: starttime in clock ticks since boot
    uint64_t start_ticks = 0;
    if(!(iss >> start_ticks))
    {
        ROCP_CI_LOG(WARNING) << fmt::format(
            "Unexpected end of /proc/{}/stat. Failed to read start ticks", pid);
        return 0;
    }

    return start_ticks;
}

uint64_t
compute_system_seed(std::string_view machine_id, pid_t pid, pid_t ppid, uint64_t pstart_ticks)
{
    // If no machine_id provided, read from /etc/machine-id
    ROCP_CI_LOG_IF(WARNING, machine_id.empty())
        << fmt::format("compute_system_seed provided empty machine id");

    // Hash for seed value
    return std::hash<std::string>{}(
        sha256{fmt::format("{}|{}|{}|{}", machine_id, pid, ppid, pstart_ticks)}.hexdigest());
}

std::string
generate_uuid_v7(uint64_t timestamp_ns, uint64_t seed, std::string_view delim)
{
    constexpr auto nanosec_per_millisec = std::nano::den / std::milli::den;
    auto           timestamp_ms         = timestamp_ns / nanosec_per_millisec;

    auto uuid = std::array<uint8_t, 16>{};

    // First 6 bytes = timestamp
    for(int i = 0; i < 6; ++i)
    {
        uuid[i] = static_cast<uint8_t>((timestamp_ms >> (40 - 8 * i)) & 0xFF);
    }

    // Set version to 7 with ordering based on timestamp.
    uuid[6] = static_cast<uint8_t>((timestamp_ms >> 8) & 0x0F);
    uuid[6] |= 0x70;

    uuid[7] = static_cast<uint8_t>(timestamp_ms & 0xFF);

    // Seeded RNG
    auto rand64 = std::mt19937_64{seed}();

    for(int i = 0; i < 8; ++i)
    {
        uuid[8 + i] = static_cast<uint8_t>((rand64 >> (56 - 8 * i)) & 0xFF);
    }

    // Set variant to RFC 4122
    uuid[8] = (uuid[8] & 0x3F) | 0x80;

    // Format as UUID string
    auto oss = std::ostringstream{};
    oss << std::hex << std::setfill('0');
    for(int i = 0; i < 16; ++i)
    {
        oss << std::setw(2) << static_cast<int>(uuid[i]);
        if(i == 3 || i == 5 || i == 7 || i == 9) oss << delim;
    }

    return oss.str();
}
}  // namespace common
}  // namespace rocprofiler
