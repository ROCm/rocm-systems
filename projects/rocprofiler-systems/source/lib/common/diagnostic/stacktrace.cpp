// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/stacktrace.hpp"
#include "common/diagnostic/color.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <cxxabi.h>
#include <dlfcn.h>
#include <dwarf.h>
#include <elfutils/libdwfl.h>
#include <libgen.h>
#include <unistd.h>

// Compile flags already define UNW_LOCAL_ONLY for the project; redefining
// would trip -Werror. Guard so the header still builds in isolation.
#ifndef UNW_LOCAL_ONLY
#    define UNW_LOCAL_ONLY
#endif
#include <libunwind.h>

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
namespace
{
// ---------------------------------------------------------------------
// Demangle cache. __cxa_demangle is hot enough on long traces that we
// cache by mangled-string pointer (addresses are stable within a process).
// ---------------------------------------------------------------------
struct demangle_cache
{
    std::mutex                                   mu;
    std::unordered_map<std::string, std::string> map;
};

demangle_cache&
demangler()
{
    static demangle_cache c;
    return c;
}

// gcc emits link-time-only suffixes on certain optimized symbols:
//   `.cold`        - cold-section split
//   `.constprop.N` - constant-propagation clone
//   `.isra.N`      - interprocedural scalar replacement
//   `.part.N`      - function partition
//   `.lto_priv.N`  - LTO private clone
// Strip them so the trace shows the underlying user-visible function name.
std::string
strip_optimizer_suffix(const std::string& s)
{
    static const char* const suffixes[] = {
        ".cold", ".constprop", ".isra", ".part", ".lto_priv",
    };
    std::size_t cut = std::string::npos;
    for(const char* sfx : suffixes)
    {
        auto pos = s.find(sfx);
        if(pos != std::string::npos && pos < cut)
        {
            cut = pos;
        }
    }
    if(cut == std::string::npos) return s;
    return s.substr(0, cut);
}

std::string
demangle(const char* mangled)
{
    if(mangled == nullptr || *mangled == '\0')
    {
        return {};
    }

    auto& dc = demangler();
    {
        std::lock_guard lock{ dc.mu };
        if(auto it = dc.map.find(mangled); it != dc.map.end())
        {
            return it->second;
        }
    }

    // Optimizer suffixes break demangling. Strip first, then demangle.
    std::string base = strip_optimizer_suffix(mangled);

    int   status = 0;
    char* out    = abi::__cxa_demangle(base.c_str(), nullptr, nullptr, &status);
    auto  result = (status == 0 && out != nullptr) ? std::string{ out } : base;
    std::free(out);

    std::lock_guard lock{ dc.mu };
    dc.map.emplace(mangled, result);
    return result;
}

// ---------------------------------------------------------------------
// Process-wide Dwfl session. dwfl_begin/end is heavy; we keep a singleton
// guarded by a mutex for the lifetime of the process. dwfl_report_*
// rebinds modules on each access in case dlopen happened.
// ---------------------------------------------------------------------
struct dwfl_session
{
    Dwfl*          handle = nullptr;
    Dwfl_Callbacks callbacks{};
    std::mutex     mu;
    char*          debuginfo_path = nullptr;

    dwfl_session()
    {
        callbacks.find_elf        = dwfl_linux_proc_find_elf;
        callbacks.find_debuginfo  = dwfl_standard_find_debuginfo;
        callbacks.section_address = dwfl_offline_section_address;
        callbacks.debuginfo_path  = &debuginfo_path;
        handle                    = dwfl_begin(&callbacks);
        if(handle != nullptr)
        {
            // Loads modules from /proc/self/maps. Refreshed on each
            // symbolize pass so newly dlopen'd .so files appear.
            dwfl_linux_proc_report(handle, ::getpid());
            dwfl_report_end(handle, nullptr, nullptr);
        }
    }

    dwfl_session(const dwfl_session&)            = delete;
    dwfl_session& operator=(const dwfl_session&) = delete;
    dwfl_session(dwfl_session&&)                 = delete;
    dwfl_session& operator=(dwfl_session&&)      = delete;
};

dwfl_session&
dwfl_singleton()
{
    static dwfl_session s;
    return s;
}

// Refresh module map (cheap if no new dlopens). Caller holds the dwfl mu.
void
refresh_dwfl(Dwfl* handle)
{
    if(handle == nullptr)
    {
        return;
    }
    dwfl_report_begin_add(handle);
    dwfl_linux_proc_report(handle, ::getpid());
    dwfl_report_end(handle, nullptr, nullptr);
}

// ---------------------------------------------------------------------
// Resolve a single PC into 1+ frames (multiple if there's an inline chain).
// Order of returned frames is outer->inner, matching the format spec.
// ---------------------------------------------------------------------
std::vector<stacktrace_frame>
resolve_address(std::uintptr_t addr)
{
    std::vector<stacktrace_frame> frames;
    if(addr == 0)
    {
        return frames;
    }

    // First pass: dladdr. Always works for symbol+module if the symbol
    // is exported; misses static/local symbols.
    Dl_info info{};
    bool    have_dladdr = (::dladdr(reinterpret_cast<void*>(addr), &info) != 0);

    auto&           sess = dwfl_singleton();
    Dwfl_Module*    mod  = nullptr;
    std::lock_guard lock{ sess.mu };
    if(sess.handle != nullptr)
    {
        mod = dwfl_addrmodule(sess.handle, addr);
        if(mod == nullptr)
        {
            refresh_dwfl(sess.handle);
            mod = dwfl_addrmodule(sess.handle, addr);
        }
    }

    // Build the outer frame from libdw (preferred) or dladdr (fallback).
    stacktrace_frame outer{};
    outer.address = addr;

    if(have_dladdr && info.dli_fname != nullptr)
    {
        outer.module = ::basename(const_cast<char*>(info.dli_fname));
    }

    if(mod != nullptr)
    {
        // Module name from dwfl; preferred over dladdr's dli_fname when
        // both differ (dwfl resolves /proc/self/exe etc.).
        const char* mn = dwfl_module_info(mod, nullptr, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, nullptr);
        if(mn != nullptr && outer.module.empty())
        {
            outer.module = ::basename(const_cast<char*>(mn));
        }

        // Function name + offset.
        GElf_Sym    sym{};
        const char* fn = dwfl_module_addrname(mod, addr);
        if(fn != nullptr)
        {
            outer.function = demangle(fn);
        }

        // Resolve symbol-relative offset; dwfl_module_addrsym fills sym.
        GElf_Off    off_in_sym   = 0;
        GElf_Word   shndx_unused = 0;
        const char* sym_name     = dwfl_module_addrinfo(mod, addr, &off_in_sym, &sym,
                                                        &shndx_unused, nullptr, nullptr);
        if(sym_name != nullptr && outer.function.empty())
        {
            outer.function = demangle(sym_name);
        }
        outer.offset = static_cast<std::uint32_t>(off_in_sym);

        // File / line. dwfl_module_getsrc returns the source line for the
        // outer (real, non-inlined) frame.
        Dwfl_Line* line = dwfl_module_getsrc(mod, addr);
        if(line != nullptr)
        {
            int         linenum = 0;
            int         colnum  = 0;
            const char* src =
                dwfl_lineinfo(line, nullptr, &linenum, &colnum, nullptr, nullptr);
            if(src != nullptr)
            {
                outer.file = src;
            }
            outer.line   = static_cast<std::uint32_t>(linenum > 0 ? linenum : 0);
            outer.column = static_cast<std::uint32_t>(colnum > 0 ? colnum : 0);
        }
    }
    else if(have_dladdr)
    {
        if(info.dli_sname != nullptr)
        {
            outer.function = demangle(info.dli_sname);
            if(info.dli_saddr != nullptr)
            {
                outer.offset = static_cast<std::uint32_t>(
                    addr - reinterpret_cast<std::uintptr_t>(info.dli_saddr));
            }
        }
    }

    // Inline-frame walking is intentionally NOT done here. The outer
    // resolution above already gives the enclosing subprogram + the actual
    // source line of the running PC (via dwfl_module_getsrc, which is
    // inline-aware). Walking DW_TAG_inlined_subroutine separately produces
    // confusing duplicates under aggressive inlining (RAII destructors,
    // `.cold` split-out blocks, etc.) without adding much information.
    // Inline-aware multi-frame rendering is a Phase 2 follow-up.
    frames.push_back(std::move(outer));
    return frames;
}

// ---------------------------------------------------------------------
// Per-process exe path cache, used to suppress the module suffix when
// the frame's module is the main exe.
// ---------------------------------------------------------------------
const std::string&
exe_basename()
{
    static const std::string name = []() {
        char    buf[4096];
        ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if(n <= 0)
        {
            return std::string{};
        }
        buf[n] = '\0';
        return std::string{ ::basename(buf) };
    }();
    return name;
}

// ---------------------------------------------------------------------
// Env-var driven runtime overrides.
// ---------------------------------------------------------------------
struct runtime_overrides
{
    bool verbose   = false;
    bool no_filter = false;
};

const runtime_overrides&
get_overrides()
{
    static const runtime_overrides r = []() {
        runtime_overrides o;
        if(const char* v = std::getenv("ROCPROFSYS_TRACE_VERBOSE");
           v != nullptr && std::strcmp(v, "1") == 0)
        {
            o.verbose = true;
        }
        if(const char* v = std::getenv("ROCPROFSYS_TRACE_NO_FILTER");
           v != nullptr && std::strcmp(v, "1") == 0)
        {
            o.no_filter = true;
        }
        return o;
    }();
    return r;
}

bool
should_skip(const std::string& fn, const std::vector<std::string>& filters)
{
    if(get_overrides().no_filter)
    {
        return false;
    }
    for(const auto& sub : filters)
    {
        if(fn.find(sub) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

// Try to render a path as project-relative. We don't have access to
// PROJECT_SOURCE_DIR at this layer, so we fall back to "shorten common
// prefixes" - drop leading "/" segment count over 3 with "..".
std::string
display_path(const std::string& p)
{
    if(p.empty()) return p;

    // If the cwd is a strict prefix, render relative.
    char cwd[4096];
    if(::getcwd(cwd, sizeof(cwd)) != nullptr)
    {
        const std::string c{ cwd };
        if(p.compare(0, c.size(), c) == 0 && p.size() > c.size() && p[c.size()] == '/')
        {
            return p.substr(c.size() + 1);
        }
    }
    return p;
}

// Number of decimal digits needed to render `n` (minimum 1).
std::size_t
decimal_width(std::size_t n)
{
    std::size_t w = 1;
    while(n >= 10)
    {
        n /= 10;
        ++w;
    }
    return w;
}
}  // namespace

const std::vector<std::string>&
stacktrace::default_skip_filters()
{
    static const std::vector<std::string> filters = {
        "rocprofsys::diagnostic::stacktrace::capture",
        "rocprofsys::common::diagnostic::stacktrace::capture",
        "rocprofsys::common::diagnostic::format_exception",
        "rocprofsys::common::diagnostic::print_exception",
        "__libc_start_main",
        "__libc_init_first",
        "_start",
        "__GI___",
        "__cxa_throw",
        "__cxa_rethrow",
        "_Unwind_RaiseException",
        "__sanitizer::",
        "__asan_",
        "__ubsan_",
        "__tsan_",
        "std::__throw_",
        "std::__detail::",
    };
    return filters;
}

stacktrace
stacktrace::capture(int skip_frames, int max_frames) noexcept
{
    stacktrace out;
    if(max_frames <= 0)
    {
        return out;
    }
    out.m_raw.reserve(static_cast<std::size_t>(max_frames));

    unw_context_t ctx;
    if(unw_getcontext(&ctx) < 0)
    {
        return out;
    }

    unw_cursor_t cur;
    if(unw_init_local(&cur, &ctx) < 0)
    {
        return out;
    }

    // unw_step advances PAST the current frame, so the first iteration
    // already yields capture's caller. skip_frames is how many additional
    // frames the caller wants dropped on top of that.
    int to_skip = skip_frames;
    while(unw_step(&cur) > 0 && static_cast<int>(out.m_raw.size()) < max_frames)
    {
        if(to_skip > 0)
        {
            --to_skip;
            continue;
        }
        unw_word_t ip = 0;
        if(unw_get_reg(&cur, UNW_REG_IP, &ip) < 0)
        {
            break;
        }
        if(ip == 0)
        {
            break;
        }
        out.m_raw.push_back(static_cast<std::uintptr_t>(ip));
    }
    return out;
}

bool
stacktrace::empty() const noexcept
{
    return m_raw.empty();
}

std::pair<const std::uintptr_t*, std::size_t>
stacktrace::raw() const noexcept
{
    return { m_raw.data(), m_raw.size() };
}

const std::vector<stacktrace_frame>&
stacktrace::frames() const
{
    if(m_frames.has_value())
    {
        return *m_frames;
    }
    std::vector<stacktrace_frame> v;
    v.reserve(m_raw.size());
    for(auto addr : m_raw)
    {
        auto resolved = resolve_address(addr);
        if(resolved.empty())
        {
            stacktrace_frame f{};
            f.address = addr;
            v.push_back(std::move(f));
        }
        else
        {
            for(auto& f : resolved)
            {
                v.push_back(std::move(f));
            }
        }
    }
    m_frames.emplace(std::move(v));
    return *m_frames;
}

std::string
stacktrace::to_string(format_options opt) const
{
    // If caller didn't override, use defaults.
    if(opt.skip_substrings.empty())
    {
        opt.skip_substrings = default_skip_filters();
    }

    // Apply env-driven overrides (verbose, no_filter).
    const auto& ov = get_overrides();
    if(ov.verbose)
    {
        opt.with_offset = true;
        opt.with_module = true;
        if(opt.max_frames_shown < 64) opt.max_frames_shown = 64;
    }
    if(ov.no_filter)
    {
        opt.skip_substrings.clear();
    }

    bool color_on = false;
    switch(opt.color)
    {
        case color_mode::off: color_on = false; break;
        case color_mode::on: color_on = true; break;
        case color_mode::auto_detect: color_on = color::color_supported_for(2); break;
    }

    auto col = [color_on](const char* code) -> const char* {
        return color_on ? code : "";
    };

    const auto& fs = frames();

    // Filter pass.
    std::vector<const stacktrace_frame*> visible;
    visible.reserve(fs.size());
    std::size_t skipped = 0;
    for(const auto& f : fs)
    {
        if(should_skip(f.function, opt.skip_substrings))
        {
            ++skipped;
            continue;
        }
        visible.push_back(&f);
    }

    const std::size_t shown = std::min(visible.size(), opt.max_frames_shown);
    const std::size_t more  = visible.size() > shown ? visible.size() - shown : 0;

    // Compute padding widths once. Frame index uses the largest index that
    // will appear (shown - 1). Function-name column pads to the longest name
    // among visible frames, capped by max_function_width.
    const std::size_t idx_width = (shown == 0) ? 1 : decimal_width(shown - 1);

    std::size_t fn_col_width = 0;
    for(std::size_t i = 0; i < shown; ++i)
    {
        const auto& f      = *visible[i];
        std::size_t fn_len = f.function.empty() ? 18  // hex address: 0x + 16 hex digits
                                                : f.function.size();
        if(fn_len <= opt.max_function_width)
        {
            fn_col_width = std::max(fn_col_width, fn_len);
        }
    }

    std::ostringstream out;
    const std::string& exe = exe_basename();

    for(std::size_t i = 0; i < shown; ++i)
    {
        const auto& f = *visible[i];

        // Frame index: "  #N  " with N right-padded to idx_width.
        char idx_buf[16];
        std::snprintf(idx_buf, sizeof(idx_buf), "#%-*zu", static_cast<int>(idx_width), i);
        out << "  " << col(color::frame_idx) << idx_buf << col(color::reset) << "  ";

        // Function name (or hex address if unresolved).
        std::string fn_text;
        if(!f.function.empty())
        {
            fn_text = f.function;
        }
        else
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "0x%016lx",
                          static_cast<unsigned long>(f.address));
            fn_text = buf;
        }

        const bool oversized = fn_text.size() > opt.max_function_width;
        if(oversized)
        {
            // Don't pad oversized names; single space + `in ...` follows.
            out << col(color::fn_name) << fn_text << col(color::reset);
        }
        else
        {
            out << col(color::fn_name) << fn_text << col(color::reset);
            if(fn_text.size() < fn_col_width)
            {
                out << std::string(fn_col_width - fn_text.size(), ' ');
            }
        }

        // Suffixes attached to function-name slot: offset + module tag.
        if(opt.with_offset && f.offset != 0)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), " [+0x%x]", f.offset);
            out << col(color::tag) << buf << col(color::reset);
        }
        if(opt.with_module && !f.module.empty() && f.module != exe)
        {
            out << ' ' << col(color::tag) << '(' << f.module << ')' << col(color::reset);
        }

        // Location: `  in <file>:<line>` on the same line.
        if(opt.with_file_line && !f.file.empty())
        {
            const std::string p = display_path(f.file);
            out << "  " << col(color::keyword_dim) << "in" << col(color::reset) << ' '
                << col(color::file_path) << p << col(color::reset);
            if(f.line > 0)
            {
                out << col(color::line_num) << ':' << f.line << col(color::reset);
            }
        }

        if(f.inlined)
        {
            out << ' ' << col(color::tag) << "[inlined]" << col(color::reset);
        }

        out << '\n';
    }

    if(skipped > 0)
    {
        out << "       " << col(color::trailer) << "... " << skipped
            << " frames skipped (libc, runtime) ..." << col(color::reset) << '\n';
    }
    if(more > 0)
    {
        out << "       " << col(color::trailer) << "... " << more << " more ..."
            << col(color::reset) << '\n';
    }

    return out.str();
}
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
