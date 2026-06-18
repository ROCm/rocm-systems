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

#include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <fmt/core.h>
#include <fmt/format.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
namespace
{
using ::rocprofiler::agent::update_agent_runtime_visibility;

using NTSTATUS                = int32_t;
constexpr NTSTATUS kNtSuccess = 0;

using D3DKMT_HANDLE = uint32_t;
// Linux libdxcore.so widens the original Windows WCHAR (UTF-16, 16-bit) to
// 32-bit Unicode code points, so we use char32_t directly rather than relying
// on Linux's wchar_t happening to be 4 bytes.
using DXC_WCHAR          = char32_t;
constexpr size_t kMaxStr = 260;

// Local re-declarations of the D3DKMT ABI consumed via libdxcore.so. The DDK
// headers (d3dkmthk.h / d3dukmdt.h) ship with the Windows SDK and are not
// available on Linux toolchains, so we duplicate just the subset of structs
// and enums this file calls into. Sizes are pinned with static_asserts below
// to catch any future drift.
struct DxcLuid
{
    uint32_t LowPart;
    int32_t  HighPart;
};

union DxcEnumAdaptersFilter
{
    struct
    {
        uint64_t IncludeComputeOnly : 1;
        uint64_t IncludeDisplayOnly : 1;
        uint64_t Reserved           : 62;
    } bits;
    uint64_t Value;
};

struct DxcAdapterInfo
{
    D3DKMT_HANDLE hAdapter;
    DxcLuid       AdapterLuid;
    uint32_t      NumOfSources;
    int32_t       bPrecisePresentRegionsPreferred;
};

struct DxcEnumAdapters3
{
    DxcEnumAdaptersFilter Filter;
    uint32_t              NumAdapters;
    DxcAdapterInfo*       pAdapters;
};

struct DxcCloseAdapter
{
    D3DKMT_HANDLE hAdapter;
};

enum DxcKmtQaiType : uint32_t
{
    DXC_KMTQAITYPE_GETSEGMENTSIZE           = 3,
    DXC_KMTQAITYPE_ADAPTERADDRESS           = 6,
    DXC_KMTQAITYPE_ADAPTERREGISTRYINFO      = 8,
    DXC_KMTQAITYPE_PHYSICALADAPTERDEVICEIDS = 31,
};

struct DxcQueryAdapterInfo
{
    D3DKMT_HANDLE hAdapter;
    uint32_t      Type;
    void*         pPrivateDriverData;
    uint32_t      PrivateDriverDataSize;
};

struct DxcDeviceIds
{
    uint32_t VendorID;
    uint32_t DeviceID;
    uint32_t SubVendorID;
    uint32_t SubSystemID;
    uint32_t RevisionID;
    uint32_t BusType;
};

struct DxcQueryDeviceIds
{
    uint32_t     PhysicalAdapterIndex;
    DxcDeviceIds DeviceIds;
};

struct DxcAdapterAddress
{
    uint32_t BusNumber;
    uint32_t DeviceNumber;
    uint32_t FunctionNumber;
};

struct DxcAdapterRegistryInfo
{
    DXC_WCHAR AdapterString[kMaxStr];
    DXC_WCHAR BiosString[kMaxStr];
    DXC_WCHAR DacType[kMaxStr];
    DXC_WCHAR ChipType[kMaxStr];
};

struct DxcSegmentSizeInfo
{
    uint64_t DedicatedVideoMemorySize;
    uint64_t DedicatedSystemMemorySize;
    uint64_t SharedSystemMemorySize;
};

static_assert(sizeof(DxcAdapterInfo) == 20, "DxcAdapterInfo ABI mismatch");
static_assert(sizeof(DxcSegmentSizeInfo) == 24, "DxcSegmentSizeInfo ABI mismatch");
static_assert(sizeof(DxcDeviceIds) == 24, "DxcDeviceIds ABI mismatch");

using PFN_D3DKMTEnumAdapters3    = NTSTATUS (*)(DxcEnumAdapters3*);
using PFN_D3DKMTQueryAdapterInfo = NTSTATUS (*)(DxcQueryAdapterInfo*);
using PFN_D3DKMTCloseAdapter     = NTSTATUS (*)(const DxcCloseAdapter*);

// Try the unqualified soname first so users can override via LD_LIBRARY_PATH
// (e.g. a packaged copy or a debug build). Fall back to the canonical WSL
// install path that ships the library by default.
constexpr const char* kLibDxcoreSoname  = "libdxcore.so";
constexpr const char* kLibDxcoreWslPath = "/usr/lib/wsl/lib/libdxcore.so";

void*
open_libdxcore()
{
    void* h = ::dlopen(kLibDxcoreSoname, RTLD_NOW | RTLD_LOCAL);
    if(!h) h = ::dlopen(kLibDxcoreWslPath, RTLD_NOW | RTLD_LOCAL);
    return h;
}

bool
probe_libdxcore()
{
    static const bool _v = []() {
        if(::access("/dev/dxg", F_OK) != 0)
        {
            ROCP_INFO << "wsl::is_available: /dev/dxg not present; not a WSL GPU environment";
            return false;
        }
        // /dev/dxg passed, so we are inside WSL with the GPU paravirt driver
        // loaded; anything missing from here on is genuinely unexpected and
        // worth surfacing as a warning rather than swallowing at INFO.
        void* h = open_libdxcore();
        if(!h)
        {
            ROCP_WARNING << "wsl::is_available: /dev/dxg present but dlopen(libdxcore.so) failed: "
                         << ::dlerror();
            return false;
        }
        void* sym = ::dlsym(h, "D3DKMTEnumAdapters3");
        if(!sym)
        {
            ROCP_WARNING
                << "wsl::is_available: libdxcore.so loaded but D3DKMTEnumAdapters3 not exported";
            ::dlclose(h);
            return false;
        }
        ::dlclose(h);
        return true;
    }();
    return _v;
}

// Random per-process offset applied to rocprofiler_agent_id_t.handle. Kept
// identical to the gnulinux path so agent IDs are non-stable across runs and
// downstream code cannot accidentally treat them as ordinals.
uint64_t
get_agent_offset()
{
    static const uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}

// UTF-8 encoding constants per RFC 3629. Boundary thresholds delimit the
// 1/2/3/4-byte ranges; lead-byte prefixes mark how many continuation bytes
// follow; the continuation prefix tags every trailing byte; the payload mask
// extracts the 6 data bits each continuation byte carries.
constexpr uint32_t kUtf8OneByteMax    = 0x80;     // < 0x80         => 1 byte
constexpr uint32_t kUtf8TwoByteMax    = 0x800;    // < 0x800        => 2 bytes
constexpr uint32_t kUtf8ThreeByteMax  = 0x10000;  // < 0x10000      => 3 bytes
constexpr uint8_t  kUtf8LeadTwoByte   = 0xC0;
constexpr uint8_t  kUtf8LeadThreeByte = 0xE0;
constexpr uint8_t  kUtf8LeadFourByte  = 0xF0;
constexpr uint8_t  kUtf8ContPrefix    = 0x80;
constexpr uint32_t kUtf8ContPayload   = 0x3F;

std::string
wchar_to_utf8(const DXC_WCHAR* src, size_t max_len)
{
    std::string out;
    out.reserve(max_len);
    for(size_t i = 0; i < max_len && src[i] != 0; ++i)
    {
        auto cp = static_cast<uint32_t>(src[i]);
        if(cp < kUtf8OneByteMax)
        {
            out.push_back(static_cast<char>(cp));
        }
        else if(cp < kUtf8TwoByteMax)
        {
            out.push_back(static_cast<char>(kUtf8LeadTwoByte | (cp >> 6)));
            out.push_back(static_cast<char>(kUtf8ContPrefix | (cp & kUtf8ContPayload)));
        }
        else if(cp < kUtf8ThreeByteMax)
        {
            out.push_back(static_cast<char>(kUtf8LeadThreeByte | (cp >> 12)));
            out.push_back(static_cast<char>(kUtf8ContPrefix | ((cp >> 6) & kUtf8ContPayload)));
            out.push_back(static_cast<char>(kUtf8ContPrefix | (cp & kUtf8ContPayload)));
        }
        else
        {
            out.push_back(static_cast<char>(kUtf8LeadFourByte | (cp >> 18)));
            out.push_back(static_cast<char>(kUtf8ContPrefix | ((cp >> 12) & kUtf8ContPayload)));
            out.push_back(static_cast<char>(kUtf8ContPrefix | ((cp >> 6) & kUtf8ContPayload)));
            out.push_back(static_cast<char>(kUtf8ContPrefix | (cp & kUtf8ContPayload)));
        }
    }
    return out;
}

NTSTATUS
query_one(PFN_D3DKMTQueryAdapterInfo query_fn,
          D3DKMT_HANDLE              hAdapter,
          DxcKmtQaiType              type,
          void*                      out,
          uint32_t                   out_size)
{
    DxcQueryAdapterInfo q{};
    q.hAdapter              = hAdapter;
    q.Type                  = type;
    q.pPrivateDriverData    = out;
    q.PrivateDriverDataSize = out_size;
    auto st                 = query_fn(&q);
    if(st != kNtSuccess)
    {
        ROCP_INFO << fmt::format(
            "wsl::enumerate: D3DKMTQueryAdapterInfo type={} failed status=0x{:08x}",
            static_cast<uint32_t>(type),
            static_cast<uint32_t>(st));
    }
    return st;
}

// RAII wrapper for the dlopen'd libdxcore.so handle plus resolved symbols.
struct DxcoreHandle
{
    void*                      handle        = nullptr;
    PFN_D3DKMTEnumAdapters3    enum_adapters = nullptr;
    PFN_D3DKMTQueryAdapterInfo query_adapter = nullptr;
    PFN_D3DKMTCloseAdapter     close_adapter = nullptr;

    DxcoreHandle()
    {
        handle = open_libdxcore();
        if(!handle) return;
        enum_adapters =
            reinterpret_cast<PFN_D3DKMTEnumAdapters3>(::dlsym(handle, "D3DKMTEnumAdapters3"));
        query_adapter =
            reinterpret_cast<PFN_D3DKMTQueryAdapterInfo>(::dlsym(handle, "D3DKMTQueryAdapterInfo"));
        close_adapter =
            reinterpret_cast<PFN_D3DKMTCloseAdapter>(::dlsym(handle, "D3DKMTCloseAdapter"));
    }

    ~DxcoreHandle()
    {
        if(handle) ::dlclose(handle);
    }

    DxcoreHandle(const DxcoreHandle&) = delete;
    DxcoreHandle& operator=(const DxcoreHandle&) = delete;

    bool ready() const
    {
        return handle != nullptr && enum_adapters != nullptr && query_adapter != nullptr &&
               close_adapter != nullptr;
    }
};
}  // namespace

bool
is_available()
{
    return probe_libdxcore();
}

std::vector<unique_agent_t>
enumerate()
{
    std::vector<unique_agent_t> out;

    DxcoreHandle dxc;
    if(!dxc.handle)
    {
        ROCP_WARNING << "wsl::enumerate: libdxcore.so not available; returning empty topology";
        return out;
    }
    if(!dxc.ready())
    {
        ROCP_WARNING << "wsl::enumerate: required D3DKMT* symbols missing in libdxcore.so";
        return out;
    }

    DxcEnumAdapters3 e{};
    e.Filter.bits.IncludeComputeOnly = 1;

    NTSTATUS st = dxc.enum_adapters(&e);
    if(st != kNtSuccess)
    {
        ROCP_WARNING << fmt::format(
            "wsl::enumerate: D3DKMTEnumAdapters3 (count) failed status=0x{:08x}",
            static_cast<uint32_t>(st));
        return out;
    }

    if(e.NumAdapters == 0)
    {
        ROCP_INFO << "wsl::enumerate: zero adapters reported by D3DKMTEnumAdapters3";
        return out;
    }

    std::vector<DxcAdapterInfo> infos(e.NumAdapters);
    e.pAdapters = infos.data();
    st          = dxc.enum_adapters(&e);
    if(st != kNtSuccess)
    {
        ROCP_WARNING << fmt::format(
            "wsl::enumerate: D3DKMTEnumAdapters3 (fill) failed status=0x{:08x}",
            static_cast<uint32_t>(st));
        return out;
    }

    const auto offset = get_agent_offset();
    // Every adapter enumerated through DXCore is a GPU, so the logical node
    // id and the per-type id move in lockstep.
    uint64_t logical = 0;

    for(uint32_t i = 0; i < e.NumAdapters; ++i)
    {
        const auto& a = infos[i];

        // Close the adapter on every exit path (early continues, query
        // failures, exceptions from new below).
        auto _closer = common::scope_destructor{[&dxc, h = a.hAdapter]() {
            DxcCloseAdapter cl{h};
            dxc.close_adapter(&cl);
        }};

        DxcQueryDeviceIds devids{};
        if(query_one(dxc.query_adapter,
                     a.hAdapter,
                     DXC_KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
                     &devids,
                     sizeof(devids)) != kNtSuccess)
        {
            continue;
        }

        if(devids.DeviceIds.VendorID != 0x1002)
        {
            ROCP_INFO << fmt::format(
                "wsl::enumerate: skipping non-AMD adapter (vendor=0x{:04x} device=0x{:04x})",
                devids.DeviceIds.VendorID,
                devids.DeviceIds.DeviceID);
            continue;
        }

        // The remaining queries populate fields the agent struct cannot
        // sensibly do without (BDF / adapter name / VRAM size). Treat any
        // failure as a reason to discard the adapter rather than publish a
        // half-filled rocprofiler_agent_t.
        DxcAdapterAddress addr{};
        if(query_one(
               dxc.query_adapter, a.hAdapter, DXC_KMTQAITYPE_ADAPTERADDRESS, &addr, sizeof(addr)) !=
           kNtSuccess)
        {
            ROCP_WARNING << fmt::format("wsl::enumerate: discarding adapter {} "
                                        "(vendor=0x{:04x} device=0x{:04x}): "
                                        "ADAPTERADDRESS query failed",
                                        i,
                                        devids.DeviceIds.VendorID,
                                        devids.DeviceIds.DeviceID);
            continue;
        }

        DxcAdapterRegistryInfo reg{};
        if(query_one(dxc.query_adapter,
                     a.hAdapter,
                     DXC_KMTQAITYPE_ADAPTERREGISTRYINFO,
                     &reg,
                     sizeof(reg)) != kNtSuccess)
        {
            ROCP_WARNING << fmt::format("wsl::enumerate: discarding adapter {} "
                                        "(vendor=0x{:04x} device=0x{:04x}): "
                                        "ADAPTERREGISTRYINFO query failed",
                                        i,
                                        devids.DeviceIds.VendorID,
                                        devids.DeviceIds.DeviceID);
            continue;
        }

        DxcSegmentSizeInfo seg{};
        if(query_one(
               dxc.query_adapter, a.hAdapter, DXC_KMTQAITYPE_GETSEGMENTSIZE, &seg, sizeof(seg)) !=
           kNtSuccess)
        {
            ROCP_WARNING << fmt::format("wsl::enumerate: discarding adapter {} "
                                        "(vendor=0x{:04x} device=0x{:04x}): "
                                        "GETSEGMENTSIZE query failed",
                                        i,
                                        devids.DeviceIds.VendorID,
                                        devids.DeviceIds.DeviceID);
            continue;
        }

        auto info                 = common::init_public_api_struct(rocprofiler_agent_t{});
        info.type                 = ROCPROFILER_AGENT_TYPE_GPU;
        info.logical_node_id      = logical;
        info.node_id              = static_cast<uint32_t>(logical);
        info.id.handle            = logical + offset;
        info.logical_node_type_id = logical;
        ++logical;

        info.vendor_id   = devids.DeviceIds.VendorID;
        info.device_id   = devids.DeviceIds.DeviceID;
        info.location_id = ((addr.BusNumber & 0xFF) << 8) | ((addr.DeviceNumber & 0x1F) << 3) |
                           (addr.FunctionNumber & 0x7);
        info.domain         = 0;
        info.local_mem_size = seg.DedicatedVideoMemorySize;
        // DXCore on WSL does not expose XCC topology; consumer-class adapters
        // shipped on WSL today are single-XCC, so 1 is the only correct value
        // here. Multi-XCC datacenter parts are not supported on this path.
        info.num_xcc = 1;

        auto adapter_name = wchar_to_utf8(reg.AdapterString, kMaxStr);
        if(adapter_name.empty()) adapter_name = "unknown";

        info.product_name = common::get_string_entry(adapter_name)->c_str();
        info.vendor_name  = common::get_string_entry("AMD")->c_str();
        info.name         = info.product_name;
        info.model_name   = common::get_string_entry("")->c_str();

        info.mem_banks_count = 0;
        info.caches_count    = 0;
        info.io_links_count  = 0;
        info.mem_banks       = nullptr;
        info.caches          = nullptr;
        info.io_links        = nullptr;

        std::memset(&info.uuid.bytes, 0, sizeof(info.uuid.bytes));

        update_agent_runtime_visibility(info);

        ROCP_INFO << fmt::format(
            "wsl::enumerate: enumerated adapter {} vendor=0x{:04x} device=0x{:04x} "
            "BDF={:02x}:{:02x}.{:x} dedicated_vram={} '{}'",
            i,
            devids.DeviceIds.VendorID,
            devids.DeviceIds.DeviceID,
            addr.BusNumber,
            addr.DeviceNumber,
            addr.FunctionNumber,
            seg.DedicatedVideoMemorySize,
            adapter_name);

        out.emplace_back(new rocprofiler_agent_t{info}, [](rocprofiler_agent_t* p) {
            if(p)
            {
                delete[] p->mem_banks;
                delete[] p->caches;
                delete[] p->io_links;
            }
            delete p;
        });
    }

    return out;
}

}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
