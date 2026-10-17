// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/platform/windows/d3dkmt_loader.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

// clang-format off
// Third-party WDK D3DKMT thunk headers. <windows.h> must come first so the base
// Windows types (UINT/WCHAR/LUID/NTSTATUS/NTDDI_VERSION/...) the vendored
// headers reference are defined. These includes are isolated to this .cpp so
// clang-tidy never lints the vendored headers via a project TU.
#include <windows.h>
#include "lib/rocprofiler-sdk/platform/windows/external/d3dkmt/d3dkmthk.h"
// clang-format on

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace windows
{
namespace
{
// D3DKMT NTSTATUS success is STATUS_SUCCESS == 0.
constexpr ::NTSTATUS kNtSuccess = 0;

// Prioritized module list to resolve the D3DKMT entry points from. The env
// override comes first so a test (or a future GPU simulator) can swap in an
// alternate DLL without rebuilding; gdi32.dll is the real implementation on
// native Windows. The list is intentionally trivial to extend with future
// simulator DLLs (dtif64a.dll / hsakmtmodel.dll / hsakmt-sim.dll) - append more
// fallbacks here, no other code changes required.
constexpr const char*    kEnvModuleOverride = "ROCPROFILER_D3DKMT_MODULE";
constexpr const wchar_t* kDefaultModule     = L"gdi32.dll";

// Resolve a function pointer of type Fn from module, logging on failure.
template <typename Fn>
Fn
resolve(::HMODULE module, const char* name)
{
    auto* proc = ::GetProcAddress(module, name);
    if(proc == nullptr)
    {
        ROCP_INFO << fmt::format("d3dkmt_loader: symbol '{}' not exported by module", name);
        return nullptr;
    }
    return reinterpret_cast<Fn>(reinterpret_cast<void*>(proc));
}

std::wstring
utf8_to_wide(const std::string& src)
{
    if(src.empty()) return std::wstring{};
    int needed =
        ::MultiByteToWideChar(CP_UTF8, 0, src.c_str(), static_cast<int>(src.size()), nullptr, 0);
    if(needed <= 0) return std::wstring{};
    auto out = std::wstring(static_cast<size_t>(needed), L'\0');
    ::MultiByteToWideChar(
        CP_UTF8, 0, src.c_str(), static_cast<int>(src.size()), out.data(), needed);
    return out;
}
}  // namespace

// Concrete dispatch table. Function pointers use the PFND3DKMT_* typedefs from
// the vendored WDK header so the APIENTRY (__stdcall) calling convention is
// preserved correctly - a raw void(*)() would drop the convention annotation.
struct d3dkmt_loader::impl_t
{
    ::HMODULE                    hmodule       = nullptr;
    bool                         owns_module   = false;
    ::PFND3DKMT_ENUMADAPTERS3    enum_adapters = nullptr;
    ::PFND3DKMT_QUERYADAPTERINFO query_adapter = nullptr;
    ::PFND3DKMT_CLOSEADAPTER     close_adapter = nullptr;

    bool ready() const
    {
        return hmodule != nullptr && enum_adapters != nullptr && query_adapter != nullptr &&
               close_adapter != nullptr;
    }
};

namespace
{
// Resolve D3DKMT entry points from a single named module. GetModuleHandleW
// first (no refcount bump, no FreeLibrary needed), LoadLibraryW as a fallback
// (refcounted; recorded so the destructor FreeLibrary()s it). Returns true and
// fills out on success; false and leaves out unchanged on any failure.
bool
open_d3dkmt_module(const std::wstring& name, d3dkmt_loader::impl_t& out)
{
    if(name.empty()) return false;

    bool      owns    = false;
    ::HMODULE hmodule = ::GetModuleHandleW(name.c_str());
    if(hmodule == nullptr)
    {
        hmodule = ::LoadLibraryW(name.c_str());
        owns    = (hmodule != nullptr);
    }
    if(hmodule == nullptr)
    {
        ROCP_INFO << "d3dkmt_loader: could not obtain a handle for module";
        return false;
    }

    auto enum_fn  = resolve<::PFND3DKMT_ENUMADAPTERS3>(hmodule, "D3DKMTEnumAdapters3");
    auto query_fn = resolve<::PFND3DKMT_QUERYADAPTERINFO>(hmodule, "D3DKMTQueryAdapterInfo");
    auto close_fn = resolve<::PFND3DKMT_CLOSEADAPTER>(hmodule, "D3DKMTCloseAdapter");

    if(enum_fn == nullptr || query_fn == nullptr || close_fn == nullptr)
    {
        if(owns) ::FreeLibrary(hmodule);
        return false;
    }

    out.hmodule       = hmodule;
    out.owns_module   = owns;
    out.enum_adapters = enum_fn;
    out.query_adapter = query_fn;
    out.close_adapter = close_fn;
    return true;
}

// Resolve D3DKMT entry points. When ROCPROFILER_D3DKMT_MODULE is set the
// env_override is tried exclusively — no fallback to gdi32.dll — so a test
// that forces a bogus module name reliably produces a failed load. Without the
// env_override, gdi32.dll is the only candidate (the real implementation on
// native Windows).
void
open_d3dkmt(d3dkmt_loader::impl_t& out)
{
    auto env_override = common::get_env(kEnvModuleOverride, std::string{});
    if(!env_override.empty())
    {
        if(!open_d3dkmt_module(utf8_to_wide(env_override), out))
            ROCP_INFO << fmt::format(
                "d3dkmt_loader: {} override '{}' failed to load or missing D3DKMT symbols; "
                "not falling back to gdi32.dll",
                kEnvModuleOverride,
                env_override);
        return;
    }
    open_d3dkmt_module(std::wstring{kDefaultModule}, out);
}

// Single KMTQAITYPE query into a caller-provided buffer.
bool
query_one(const d3dkmt_loader::impl_t& impl,
          uint32_t                     hadapter,
          ::KMTQUERYADAPTERINFOTYPE    type,
          void*                        out,
          uint32_t                     out_size)
{
    auto q                  = ::D3DKMT_QUERYADAPTERINFO{};
    q.hAdapter              = static_cast<::D3DKMT_HANDLE>(hadapter);
    q.Type                  = type;
    q.pPrivateDriverData    = out;
    q.PrivateDriverDataSize = out_size;

    auto st = impl.query_adapter(&q);
    if(st != kNtSuccess)
    {
        ROCP_INFO << fmt::format(
            "d3dkmt_loader: D3DKMTQueryAdapterInfo type={} failed status=0x{:08x}",
            static_cast<uint32_t>(type),
            static_cast<uint32_t>(st));
        return false;
    }
    return true;
}
}  // namespace

d3dkmt_loader::d3dkmt_loader()
: m_impl{new impl_t{}}
{
    open_d3dkmt(*m_impl);
}

d3dkmt_loader::~d3dkmt_loader()
{
    if(m_impl != nullptr)
    {
        if(m_impl->owns_module && m_impl->hmodule != nullptr) ::FreeLibrary(m_impl->hmodule);
        delete m_impl;
        m_impl = nullptr;
    }
}

bool
d3dkmt_loader::ready() const
{
    return m_impl != nullptr && m_impl->ready();
}

std::vector<d3dkmt_adapter>
d3dkmt_loader::enumerate_adapters() const
{
    auto out = std::vector<d3dkmt_adapter>{};
    if(!ready()) return out;

    auto e                      = ::D3DKMT_ENUMADAPTERS3{};
    e.Filter.IncludeComputeOnly = 1;

    auto st = m_impl->enum_adapters(&e);
    if(st != kNtSuccess)
    {
        ROCP_WARNING << fmt::format(
            "d3dkmt_loader: D3DKMTEnumAdapters3 (count) failed status=0x{:08x}",
            static_cast<uint32_t>(st));
        return out;
    }

    if(e.NumAdapters == 0)
    {
        ROCP_INFO << "d3dkmt_loader: zero adapters reported by D3DKMTEnumAdapters3";
        return out;
    }

    auto infos  = std::vector<::D3DKMT_ADAPTERINFO>(e.NumAdapters);
    e.pAdapters = infos.data();
    st          = m_impl->enum_adapters(&e);
    if(st != kNtSuccess)
    {
        ROCP_WARNING << fmt::format(
            "d3dkmt_loader: D3DKMTEnumAdapters3 (fill) failed status=0x{:08x}",
            static_cast<uint32_t>(st));
        return out;
    }

    // Cap iteration at the buffer we actually sized in the count pass. The fill
    // pass may report a larger NumAdapters if the adapter set grew between the
    // two calls (TOCTOU); reading past infos.size() would be an overrun, so
    // clamp and ignore any newcomers until the next enumeration (WSL caveat).
    const auto count = std::min<size_t>(e.NumAdapters, infos.size());
    out.reserve(count);
    for(size_t i = 0; i < count; ++i)
    {
        auto a      = d3dkmt_adapter{};
        a.hadapter  = static_cast<uint32_t>(infos[i].hAdapter);
        a.luid_low  = static_cast<uint32_t>(infos[i].AdapterLuid.LowPart);
        a.luid_high = static_cast<int32_t>(infos[i].AdapterLuid.HighPart);
        out.emplace_back(a);
    }
    return out;
}

bool
d3dkmt_loader::query_device_ids(uint32_t hadapter, d3dkmt_device_ids& out) const
{
    if(!ready()) return false;

    auto devids = ::D3DKMT_QUERY_DEVICE_IDS{};
    if(!query_one(*m_impl, hadapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &devids, sizeof(devids)))
    {
        return false;
    }

    out.vendor_id    = devids.DeviceIds.VendorID;
    out.device_id    = devids.DeviceIds.DeviceID;
    out.subvendor_id = devids.DeviceIds.SubVendorID;
    out.subsystem_id = devids.DeviceIds.SubSystemID;
    out.revision_id  = devids.DeviceIds.RevisionID;
    out.bus_type     = devids.DeviceIds.BusType;
    return true;
}

bool
d3dkmt_loader::query_adapter_address(uint32_t hadapter, d3dkmt_adapter_address& out) const
{
    if(!ready()) return false;

    auto addr = ::D3DKMT_ADAPTERADDRESS{};
    if(!query_one(*m_impl, hadapter, KMTQAITYPE_ADAPTERADDRESS, &addr, sizeof(addr)))
    {
        return false;
    }

    out.bus_number      = addr.BusNumber;
    out.device_number   = addr.DeviceNumber;
    out.function_number = addr.FunctionNumber;
    return true;
}

bool
d3dkmt_loader::query_segment_size(uint32_t hadapter, d3dkmt_segment_size& out) const
{
    if(!ready()) return false;

    auto seg = ::D3DKMT_SEGMENTSIZEINFO{};
    if(!query_one(*m_impl, hadapter, KMTQAITYPE_GETSEGMENTSIZE, &seg, sizeof(seg)))
    {
        return false;
    }

    out.dedicated_video_memory  = seg.DedicatedVideoMemorySize;
    out.dedicated_system_memory = seg.DedicatedSystemMemorySize;
    out.shared_system_memory    = seg.SharedSystemMemorySize;
    return true;
}

void
d3dkmt_loader::close_adapter(uint32_t hadapter) const
{
    if(!ready()) return;

    auto cl     = ::D3DKMT_CLOSEADAPTER{};
    cl.hAdapter = static_cast<::D3DKMT_HANDLE>(hadapter);
    m_impl->close_adapter(&cl);
}
}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler
