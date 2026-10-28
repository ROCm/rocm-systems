#pragma once

/**
 * @file symbol_preload.hpp
 * @brief One-shot addr2line preload across every unique IP seen in a
 *        tid->samples map. Avoids the per-IP symbolizer round-trip
 *        that would otherwise dominate post-processing.
 */

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t in caller-provided map

#include "cpu_snap/core/raw_sample.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"

namespace cs_examples
{

/**
 * Walk every per-tid raw_sample vector, collect the unique frame IPs,
 * and hand the deduped vector to sym.preload(). Subsequent
 * sym.resolve() calls then hit the in-memory cache instead of forking
 * addr2line.
 */
inline void
preload_symbols_from(
    cpu_snap::processing::symbols::symbolizer&                                        sym,
    const std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>>& per_tid)
{
    std::vector<std::uint64_t>        unique_ips;
    std::unordered_set<std::uint64_t> seen;
    for(const auto& [tid, vec] : per_tid)
    {
        for(const auto& sample : vec)
        {
            for(std::uint32_t i = 0; i < sample.frames.size(); ++i)
            {
                if(seen.insert(sample.frames[i]).second)
                {
                    unique_ips.push_back(sample.frames[i]);
                }
            }
        }
    }
    sym.preload(unique_ips);
}

}  // namespace cs_examples
