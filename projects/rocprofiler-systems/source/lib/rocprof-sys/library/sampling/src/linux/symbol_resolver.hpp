// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Resolves raw instruction-pointer addresses to demangled symbol names.
// Uses dladdr() (POSIX) for PC → mangled name, then routes through
// rocprofsys::utility::demangler for cached demangling.
// Safe to call at post_process time (not async-signal-safe — uses malloc).
// Caches results so repeated resolution of the same PC is O(1).
//
// dladdr only finds dynamic-symbol-table entries (exported functions). For
// static binary symbols (e.g. fib/run in parallel-overhead) callers should
// fall back to a libunwind-based lookup at the call site — see
// production hooks emit_resolved_to_trace_cache (TF-4).

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

    // Inject a name for a pc when the caller has a better lookup result
    // (e.g. libunwind unw_get_proc_name_by_ip succeeded after dladdr failed).
    void inject(uintptr_t pc, std::string const& name)
    {
        if(pc == 0 || name.empty()) return;
        m_cache[pc] = m_demangler.demangle(name);
    }

private:
    std::unordered_map<uintptr_t, std::string> m_cache;
    rocprofsys::utility::demangler<>           m_demangler;
};

}  // namespace rocprofsys::sampling
