// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cache_directory.hpp"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace profiler_hub
{
namespace
{
constexpr uint64_t k_fnv_offset = 0xcbf29ce484222325ULL;
constexpr uint64_t k_fnv_prime  = 0x100000001b3ULL;

void
hash_bytes(uint64_t& hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for(size_t index = 0; index < size; ++index)
    {
        hash ^= bytes[index];
        hash *= k_fnv_prime;
    }
}

std::string
hex16(uint64_t value)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}
}  // namespace

std::string
trace_cache_directory(std::string_view                    cache_root,
                      const std::vector<std::string>&     source_paths)
{
    if(cache_root.empty() || source_paths.empty()) return {};

    uint64_t hash = k_fnv_offset;
    for(const std::string& source_path : source_paths)
    {
        const std::filesystem::path file{ source_path };
        if(!std::filesystem::is_regular_file(file)) return {};
        const uint64_t size = static_cast<uint64_t>(std::filesystem::file_size(file));
        const int64_t  mtime = static_cast<int64_t>(
            std::filesystem::last_write_time(file).time_since_epoch().count());
        const uint64_t path_size = source_path.size();
        hash_bytes(hash, &path_size, sizeof(path_size));
        hash_bytes(hash, source_path.data(), source_path.size());
        hash_bytes(hash, &size, sizeof(size));
        hash_bytes(hash, &mtime, sizeof(mtime));
    }
    return (std::filesystem::path{ cache_root } / hex16(hash)).string();
}

}  // namespace profiler_hub
