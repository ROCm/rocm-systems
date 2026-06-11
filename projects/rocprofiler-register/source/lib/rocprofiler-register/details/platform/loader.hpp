// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rocprofiler_register
{
namespace platform
{
// Opaque module handle. On POSIX this is the value returned by dlopen()
// (already void*); on Windows it is an HMODULE reinterpreted to void* so
// callers do not need to include windows.h.
using module_handle_t = void*;

// Address range covering a single contiguous segment of a loaded module.
struct module_address_range
{
    std::uintptr_t start = 0;
    std::uintptr_t last  = 0;
};

// Collection of address ranges for one loaded module, identified by
// absolute filesystem path. Linux modules typically expose multiple ranges
// (one per ELF segment); Windows modules expose a single range covering
// the full SizeOfImage.
struct module_segments
{
    std::string                       filepath = {};
    std::vector<module_address_range> ranges   = {};
};

// Open a shared library by name. Linux: dlopen(name, RTLD_NOW | RTLD_LOCAL).
// Windows: LoadLibraryW(utf8_to_wide(name)).
// Returns nullptr on failure.
module_handle_t
module_open(const char* name) noexcept;

// Look up a module that is already loaded into the process without bringing
// in a new copy. Linux: dlopen(name, RTLD_LAZY | RTLD_NOLOAD). Windows:
// GetModuleHandleW(utf8_to_wide(name)). Returns nullptr if not currently
// loaded.
module_handle_t
module_open_already_loaded(const char* name) noexcept;

// Multi-attempt open. Linux: tries (RTLD_LAZY|RTLD_NOLOAD) -> RTLD_NOW ->
// (RTLD_NOW|RTLD_GLOBAL) -> versioned-name fallback. Windows: tries
// LoadLibraryW(name) -> LoadLibraryW(name + ".dll") -> CWD-prefixed ->
// rocprofiler-register-DLL-directory-prefixed.
//
// Returns the first handle that resolves; nullptr otherwise.
module_handle_t
module_open_with_fallback(const char* name) noexcept;

// Look up a symbol within a specific module handle.
void*
module_sym(module_handle_t handle, const char* sym) noexcept;

// Look up a symbol across all currently-loaded modules. Linux:
// dlsym(RTLD_DEFAULT, sym). Windows: enumerate all modules via
// EnumProcessModulesEx and return the first GetProcAddress hit.
void*
module_sym_default(const char* sym) noexcept;

// Release a module handle. Linux: dlclose(handle). Windows:
// FreeLibrary(handle). Safe to call with nullptr.
void
module_close(module_handle_t handle) noexcept;

// Resolve the absolute filesystem path of a loaded module. Returns an empty
// string if the path cannot be determined.
std::string
module_path(module_handle_t handle) noexcept;

// Enumerate the address ranges of every module loaded into the given process.
// Linux: parses /proc/<pid>/maps. Windows: EnumProcessModulesEx +
// GetModuleInformation + GetModuleFileNameW.
std::vector<module_segments>
get_segment_addresses(std::uint32_t pid);

// Convenience overload: query the calling process.
std::vector<module_segments>
get_segment_addresses();

}  // namespace platform
}  // namespace rocprofiler_register
