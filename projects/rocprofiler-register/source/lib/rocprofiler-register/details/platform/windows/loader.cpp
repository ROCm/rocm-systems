// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// WINDOWS-DIVERGENCE: This file replaces the POSIX dlopen/dlsym/dlclose +
// /proc/<pid>/maps machinery with the Win32 module loader and PSAPI module
// enumeration. It is built only on Windows; gnulinux/loader.cpp covers Linux.

#include "details/platform/loader.hpp"

#include "details/filesystem.hpp"
#include "details/platform/windows/encoding.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef NOGDI
#    define NOGDI
#endif
#include <windows.h>
//
#include <psapi.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler_register
{
namespace platform
{
namespace
{
using encoding::utf8_to_wide;
using encoding::wide_to_utf8;

// Return the absolute filesystem path of the DLL that contains this code.
// Used by the multi-attempt fallback to look for sibling DLLs alongside
// rocprofiler-register itself.
std::wstring
this_module_directory()
{
    auto handle = HMODULE{ nullptr };
    if(::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&this_module_directory),
                            &handle) == 0)
    {
        return std::wstring{};
    }

    auto buffer = std::wstring(MAX_PATH, L'\0');
    while(true)
    {
        auto written =
            ::GetModuleFileNameW(handle, buffer.data(), static_cast<DWORD>(buffer.size()));
        if(written == 0) return std::wstring{};
        if(written < buffer.size())
        {
            buffer.resize(written);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }

    auto last_sep = buffer.find_last_of(L"\\/");
    if(last_sep == std::wstring::npos) return std::wstring{};
    buffer.resize(last_sep);
    return buffer;
}

// Append ".dll" to name if it doesn't already end with .dll (case-insensitive).
std::wstring
ensure_dll_suffix(const std::wstring& name)
{
    constexpr std::wstring_view kSuffix = L".dll";
    if(name.size() >= kSuffix.size())
    {
        auto tail = std::wstring_view{ name }.substr(name.size() - kSuffix.size());
        auto match = true;
        for(size_t i = 0; i < kSuffix.size(); ++i)
        {
            if(::towlower(tail[i]) != kSuffix[i])
            {
                match = false;
                break;
            }
        }
        if(match) return name;
    }
    return name + std::wstring{ kSuffix };
}

// Read the absolute path of a loaded module, in UTF-8.
std::string
module_path_for_handle(HMODULE handle)
{
    if(handle == nullptr) return std::string{};
    auto buffer = std::wstring(MAX_PATH, L'\0');
    while(true)
    {
        auto written =
            ::GetModuleFileNameW(handle, buffer.data(), static_cast<DWORD>(buffer.size()));
        if(written == 0) return std::string{};
        if(written < buffer.size())
        {
            return wide_to_utf8(buffer.data(), static_cast<int>(written));
        }
        buffer.resize(buffer.size() * 2);
    }
}

}  // namespace

module_handle_t
module_open(const char* name) noexcept
{
    if(name == nullptr) return nullptr;
    auto wide = utf8_to_wide(name);
    if(wide.empty()) return nullptr;
    return reinterpret_cast<module_handle_t>(::LoadLibraryW(wide.c_str()));
}

module_handle_t
module_open_already_loaded(const char* name) noexcept
{
    if(name == nullptr) return nullptr;
    auto wide = utf8_to_wide(name);
    if(wide.empty()) return nullptr;
    return reinterpret_cast<module_handle_t>(::GetModuleHandleW(wide.c_str()));
}

module_handle_t
module_open_with_fallback(const char* name) noexcept
{
    if(name == nullptr) return nullptr;
    auto wide = utf8_to_wide(name);
    if(wide.empty()) return nullptr;

    // 1) Try as-given.
    auto* handle = ::LoadLibraryW(wide.c_str());
    if(handle != nullptr) return reinterpret_cast<module_handle_t>(handle);

    // 2) Try with ".dll" suffix appended if not already present.
    auto with_suffix = ensure_dll_suffix(wide);
    if(with_suffix != wide)
    {
        handle = ::LoadLibraryW(with_suffix.c_str());
        if(handle != nullptr) return reinterpret_cast<module_handle_t>(handle);
    }

    // 3) Try CWD-prefixed path.
    auto cwd_buf = std::wstring(MAX_PATH, L'\0');
    while(true)
    {
        auto needed =
            ::GetCurrentDirectoryW(static_cast<DWORD>(cwd_buf.size()), cwd_buf.data());
        if(needed == 0)
        {
            cwd_buf.clear();
            break;
        }
        if(needed <= cwd_buf.size())
        {
            cwd_buf.resize(needed);
            break;
        }
        cwd_buf.resize(needed);
    }
    if(!cwd_buf.empty())
    {
        auto cwd_path = cwd_buf + L"\\" + with_suffix;
        handle        = ::LoadLibraryW(cwd_path.c_str());
        if(handle != nullptr) return reinterpret_cast<module_handle_t>(handle);
    }

    // 4) Try alongside the rocprofiler-register DLL itself.
    auto reg_dir = this_module_directory();
    if(!reg_dir.empty())
    {
        auto sibling = reg_dir + L"\\" + with_suffix;
        handle       = ::LoadLibraryW(sibling.c_str());
        if(handle != nullptr) return reinterpret_cast<module_handle_t>(handle);
    }

    return nullptr;
}

void*
module_sym(module_handle_t handle, const char* sym) noexcept
{
    if(handle == nullptr || sym == nullptr) return nullptr;
    auto* proc =
        ::GetProcAddress(reinterpret_cast<HMODULE>(handle), sym);
    return reinterpret_cast<void*>(proc);
}

void*
module_sym_default(const char* sym) noexcept
{
    // WINDOWS-DIVERGENCE: PE/COFF has no RTLD_DEFAULT equivalent. Enumerate
    // every loaded module and return the first GetProcAddress hit. Callers
    // that need wrong-tool-wins safety MUST use module_sym(handle, sym) with
    // a specific tool handle; see plan Q9.
    if(sym == nullptr) return nullptr;
    auto* process = ::GetCurrentProcess();

    // Per MSDN, EnumProcessModulesEx may report a `needed` larger than the
    // supplied buffer if modules are loaded between the probe call and the
    // fill call. Retry up to a small bound to cover that benign race; cap
    // the retry count to avoid pathological loops if the host process is
    // pathologically loading DLLs in a tight loop.
    constexpr int kMaxRetries = 3;
    auto          modules     = std::vector<HMODULE>{};
    auto          count       = std::size_t{ 0 };
    for(int attempt = 0; attempt < kMaxRetries; ++attempt)
    {
        auto needed = DWORD{ 0 };
        if(::EnumProcessModulesEx(process, nullptr, 0, &needed, LIST_MODULES_ALL) == 0 &&
           needed == 0)
        {
            return nullptr;
        }
        modules.assign(needed / sizeof(HMODULE), HMODULE{ nullptr });
        auto cb        = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        auto cb_needed = DWORD{ 0 };
        if(::EnumProcessModulesEx(
               process, modules.data(), cb, &cb_needed, LIST_MODULES_ALL) == 0)
        {
            return nullptr;
        }
        if(cb_needed <= cb)
        {
            count = cb_needed / sizeof(HMODULE);
            break;
        }
        // Buffer was too small for the snapshot at the time of the second
        // call; retry with the freshly reported size.
        if(attempt + 1 == kMaxRetries)
        {
            // Bound the loop: accept a partial snapshot rather than spinning
            // forever. The first `cb / sizeof(HMODULE)` entries are valid.
            count = cb / sizeof(HMODULE);
        }
    }

    for(std::size_t i = 0; i < count; ++i)
    {
        if(modules[i] == nullptr) continue;  // snapshot race — slot not filled
        auto* proc = ::GetProcAddress(modules[i], sym);
        if(proc != nullptr) return reinterpret_cast<void*>(proc);
    }
    return nullptr;
}

void
module_close(module_handle_t handle) noexcept
{
    if(handle == nullptr) return;
    ::FreeLibrary(reinterpret_cast<HMODULE>(handle));
}

std::string
module_path(module_handle_t handle) noexcept
{
    if(handle == nullptr) return std::string{};
    auto path = module_path_for_handle(reinterpret_cast<HMODULE>(handle));
    if(path.empty()) return std::string{};
    auto absolute = fs::absolute(fs::path{ path });
    return absolute.string();
}

std::vector<module_segments>
get_segment_addresses(std::uint32_t pid)
{
    auto data = std::vector<module_segments>{};

    auto* current_handle = ::GetCurrentProcess();
    auto  current_pid    = ::GetCurrentProcessId();

    auto  process_handle = HANDLE{ current_handle };
    auto  needs_close    = false;
    if(pid != current_pid)
    {
        process_handle = ::OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
        if(process_handle == nullptr) return data;
        needs_close = true;
    }

    // See module_sym_default for the rationale on the resize-retry loop:
    // EnumProcessModulesEx can report a larger `needed` on the second call
    // if modules are loaded between the probe and the fill. Cap retries.
    constexpr int kMaxRetries = 3;
    auto          modules     = std::vector<HMODULE>{};
    auto          count       = std::size_t{ 0 };
    for(int attempt = 0; attempt < kMaxRetries; ++attempt)
    {
        auto needed = DWORD{ 0 };
        if(::EnumProcessModulesEx(
               process_handle, nullptr, 0, &needed, LIST_MODULES_ALL) == 0 &&
           needed == 0)
        {
            if(needs_close) ::CloseHandle(process_handle);
            return data;
        }
        modules.assign(needed / sizeof(HMODULE), HMODULE{ nullptr });
        auto cb        = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        auto cb_needed = DWORD{ 0 };
        if(::EnumProcessModulesEx(
               process_handle, modules.data(), cb, &cb_needed, LIST_MODULES_ALL) == 0)
        {
            if(needs_close) ::CloseHandle(process_handle);
            return data;
        }
        if(cb_needed <= cb)
        {
            count = cb_needed / sizeof(HMODULE);
            break;
        }
        if(attempt + 1 == kMaxRetries)
        {
            count = cb / sizeof(HMODULE);
        }
    }
    data.reserve(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        if(modules[i] == nullptr) continue;  // snapshot race — slot not filled
        auto info = MODULEINFO{};
        if(::GetModuleInformation(process_handle, modules[i], &info, sizeof(info)) == 0)
            continue;
        // Skip forwarder DLLs and zero-image MEM_IMAGE entries. Without this
        // guard, last = start + 0 - 1 wraps to UINTPTR_MAX and the resulting
        // [start, UINTPTR_MAX) range matches every address, breaking secure mode.
        if(info.SizeOfImage == 0) continue;

        auto name_buffer = std::wstring(MAX_PATH, L'\0');
        auto written     = DWORD{ 0 };
        while(true)
        {
            written = ::GetModuleFileNameExW(process_handle,
                                             modules[i],
                                             name_buffer.data(),
                                             static_cast<DWORD>(name_buffer.size()));
            if(written == 0)
            {
                break;
            }
            if(written < name_buffer.size())
            {
                break;
            }
            name_buffer.resize(name_buffer.size() * 2);
        }
        if(written == 0) continue;

        auto entry     = module_segments{};
        entry.filepath = wide_to_utf8(name_buffer.data(), static_cast<int>(written));
        auto start     = reinterpret_cast<std::uintptr_t>(info.lpBaseOfDll);
        // Exclusive upper bound: matches the half-open [start, last) semantics
        // used by _in_address_range (addr >= start && addr < last).
        auto last      = start + static_cast<std::uintptr_t>(info.SizeOfImage);
        entry.ranges.emplace_back(module_address_range{ start, last });
        data.emplace_back(std::move(entry));
    }

    if(needs_close) ::CloseHandle(process_handle);
    return data;
}

std::vector<module_segments>
get_segment_addresses()
{
    return get_segment_addresses(static_cast<std::uint32_t>(::GetCurrentProcessId()));
}

}  // namespace platform
}  // namespace rocprofiler_register
