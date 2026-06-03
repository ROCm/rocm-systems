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

// WINDOWS-DIVERGENCE: This file provides per-page memory protection introspection
// using VirtualQuery + EnumProcessModulesEx. Linux exposes this information via
// /proc/self/maps; on Windows the equivalent metadata only comes from VirtualQuery.

#include "details/platform/pe_parser.hpp"

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

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler_register
{
namespace platform
{
namespace
{
using encoding::wide_to_utf8;

bool
prot_is_readable(DWORD protect) noexcept
{
    return (protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ |
                       PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_WRITECOPY)) != 0;
}

bool
prot_is_writable(DWORD protect) noexcept
{
    return (protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_WRITECOPY)) != 0;
}

bool
prot_is_executable(DWORD protect) noexcept
{
    return (protect &
            (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
             PAGE_EXECUTE_WRITECOPY)) != 0;
}

std::string
module_path_for_base(HMODULE base)
{
    if(base == nullptr) return std::string{};
    auto buffer = std::wstring(MAX_PATH, L'\0');
    while(true)
    {
        auto written = ::GetModuleFileNameExW(::GetCurrentProcess(),
                                              base,
                                              buffer.data(),
                                              static_cast<DWORD>(buffer.size()));
        if(written == 0) return std::string{};
        if(written < buffer.size())
        {
            return wide_to_utf8(buffer.data(), static_cast<int>(written));
        }
        buffer.resize(buffer.size() * 2);
    }
}

}  // namespace

std::vector<memory_region>
get_memory_regions()
{
    auto  regions = std::vector<memory_region>{};
    auto* address = static_cast<const std::uint8_t*>(nullptr);
    auto  mbi     = MEMORY_BASIC_INFORMATION{};

    while(::VirtualQuery(address, &mbi, sizeof(mbi)) == sizeof(mbi))
    {
        if(mbi.State != MEM_FREE)
        {
            auto region          = memory_region{};
            region.base_address  = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress);
            region.size          = static_cast<std::size_t>(mbi.RegionSize);
            region.protection    = static_cast<std::uint32_t>(mbi.Protect);
            region.state         = static_cast<std::uint32_t>(mbi.State);
            region.type          = static_cast<std::uint32_t>(mbi.Type);
            region.is_readable   = prot_is_readable(mbi.Protect);
            region.is_writable   = prot_is_writable(mbi.Protect);
            region.is_executable = prot_is_executable(mbi.Protect);
            if(mbi.Type == MEM_IMAGE && mbi.State == MEM_COMMIT)
            {
                region.module_path =
                    module_path_for_base(reinterpret_cast<HMODULE>(mbi.AllocationBase));
            }
            regions.emplace_back(std::move(region));
        }
        address = static_cast<const std::uint8_t*>(mbi.BaseAddress) + mbi.RegionSize;
    }

    return regions;
}

std::uint32_t
get_memory_protection(const void* address)
{
    if(address == nullptr) return 0;
    auto mbi = MEMORY_BASIC_INFORMATION{};
    if(::VirtualQuery(address, &mbi, sizeof(mbi)) != sizeof(mbi)) return 0;
    return static_cast<std::uint32_t>(mbi.Protect);
}

bool
is_address_executable(const void* address)
{
    if(address == nullptr) return false;
    auto mbi = MEMORY_BASIC_INFORMATION{};
    if(::VirtualQuery(address, &mbi, sizeof(mbi)) != sizeof(mbi)) return false;
    if(mbi.State != MEM_COMMIT) return false;
    return prot_is_executable(mbi.Protect);
}

bool
is_address_readable(const void* address)
{
    if(address == nullptr) return false;
    auto mbi = MEMORY_BASIC_INFORMATION{};
    if(::VirtualQuery(address, &mbi, sizeof(mbi)) != sizeof(mbi)) return false;
    if(mbi.State != MEM_COMMIT) return false;
    return prot_is_readable(mbi.Protect);
}

void*
find_module_for_address(const void* address)
{
    if(address == nullptr) return nullptr;
    auto mbi = MEMORY_BASIC_INFORMATION{};
    if(::VirtualQuery(address, &mbi, sizeof(mbi)) != sizeof(mbi)) return nullptr;
    if(mbi.Type != MEM_IMAGE) return nullptr;
    return mbi.AllocationBase;
}

}  // namespace platform
}  // namespace rocprofiler_register
