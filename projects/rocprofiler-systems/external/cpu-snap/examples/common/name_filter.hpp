#pragma once

/**
 * @file name_filter.hpp
 * @brief Cheap heuristic that flags symbol names which have NOT yet
 *        been resolved into a readable form (raw hex, mangled, or the
 *        addr2line "??" sentinel). Used by demos to filter out frames
 *        that would render as noise in the Perfetto trace.
 */

#include <string_view>

namespace cs_examples
{

inline bool
name_looks_resolved(std::string_view name) noexcept
{
    if(name.empty()) return false;
    if(name == "??") return false;
    if(name.starts_with("0x")) return false;
    if(name.starts_with("_Z")) return false;
    if(name.find("+0x") != std::string_view::npos) return false;
    return true;
}

}  // namespace cs_examples
