// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Resolves raw instruction-pointer addresses to demangled symbol names.
// Uses dladdr() (POSIX) for PC → mangled name, then routes through
// rocprofsys::utility::demangler for cached demangling.
// Safe to call at post_process time (not async-signal-safe — uses malloc).
// Caches results so repeated resolution of the same PC is O(1).

#include "core/demangler.hpp"

#include <cstdint>
#include <dlfcn.h>
#include <string>
#include <unordered_map>

namespace rocprofsys::sampling
{

class symbol_resolver
{
public:
    // Returns demangled symbol name for pc, or "" on failure.
    // pc == 0 always returns "".
    [[nodiscard]] std::string resolve(uintptr_t pc)
    {
        if(pc == 0) return {};

        auto it = m_cache.find(pc);
        if(it != m_cache.end()) return it->second;

        std::string result;

        ::Dl_info info{};
        if(::dladdr(reinterpret_cast<void const*>(pc), &info) != 0 &&
           info.dli_sname != nullptr && info.dli_sname[0] != '\0')
        {
            result = m_demangler.demangle(info.dli_sname);
        }

        m_cache.emplace(pc, result);
        return result;
    }

private:
    std::unordered_map<uintptr_t, std::string> m_cache;
    rocprofsys::utility::demangler<>           m_demangler;
};

}  // namespace rocprofsys::sampling
