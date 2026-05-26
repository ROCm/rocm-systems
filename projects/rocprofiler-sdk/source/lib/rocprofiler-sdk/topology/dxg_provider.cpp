// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/topology/dxg_provider.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent_internal.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <fmt/core.h>
#include <fmt/format.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <cwchar>
#include <random>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace topology
{
namespace
{
using ::rocprofiler::agent::update_agent_runtime_visibility;

using NTSTATUS                = int32_t;
constexpr NTSTATUS kNtSuccess = 0;

using D3DKMT_HANDLE      = uint32_t;
using DXC_WCHAR          = wchar_t;
constexpr size_t kMaxStr = 260;

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
static_assert(sizeof(wchar_t) == 4, "DxcAdapterRegistryInfo expects 32-bit wchar_t on Linux");

using PFN_D3DKMTEnumAdapters3    = NTSTATUS (*)(DxcEnumAdapters3*);
using PFN_D3DKMTQueryAdapterInfo = NTSTATUS (*)(DxcQueryAdapterInfo*);
using PFN_D3DKMTCloseAdapter     = NTSTATUS (*)(const DxcCloseAdapter*);

constexpr const char* kLibDxcorePrimary  = "/usr/lib/wsl/lib/libdxcore.so";
constexpr const char* kLibDxcoreFallback = "libdxcore.so";

void*
open_libdxcore()
{
    void* h = ::dlopen(kLibDxcorePrimary, RTLD_NOW | RTLD_LOCAL);
    if(!h) h = ::dlopen(kLibDxcoreFallback, RTLD_NOW | RTLD_LOCAL);
    return h;
}

bool
probe_libdxcore()
{
    static const bool _v = []() {
        if(::access("/dev/dxg", F_OK) != 0)
        {
            ROCP_INFO << "DxgProvider: /dev/dxg not present; not a WSL GPU environment";
            return false;
        }
        void* h = open_libdxcore();
        if(!h)
        {
            ROCP_INFO << "DxgProvider: dlopen(libdxcore.so) failed: " << ::dlerror();
            return false;
        }
        void* sym = ::dlsym(h, "D3DKMTEnumAdapters3");
        if(!sym)
        {
            ROCP_INFO << "DxgProvider: dlsym(D3DKMTEnumAdapters3) failed";
            ::dlclose(h);
            return false;
        }
        ::dlclose(h);
        return true;
    }();
    return _v;
}

uint64_t
get_agent_offset()
{
    static uint64_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                           std::numeric_limits<uint16_t>::max()};
        return rng(gen);
    }();
    return _v;
}

std::string
wchar_to_utf8(const DXC_WCHAR* src, size_t max_len)
{
    std::string out;
    out.reserve(max_len);
    for(size_t i = 0; i < max_len && src[i] != 0; ++i)
    {
        auto cp = static_cast<uint32_t>(src[i]);
        if(cp < 0x80)
        {
            out.push_back(static_cast<char>(cp));
        }
        else if(cp < 0x800)
        {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if(cp < 0x10000)
        {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
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
            "DxgProvider: D3DKMTQueryAdapterInfo type={} failed status=0x{:08x}",
            static_cast<uint32_t>(type),
            static_cast<uint32_t>(st));
    }
    return st;
}
}  // namespace

struct DxgProvider::Impl
{
    void*                      dxcore_handle = nullptr;
    PFN_D3DKMTEnumAdapters3    enum_adapters = nullptr;
    PFN_D3DKMTQueryAdapterInfo query_adapter = nullptr;
    PFN_D3DKMTCloseAdapter     close_adapter = nullptr;
};

bool
DxgProvider::is_available()
{
    return probe_libdxcore();
}

DxgProvider::DxgProvider()
: m_impl(std::make_unique<Impl>())
{
    m_impl->dxcore_handle = open_libdxcore();
    if(!m_impl->dxcore_handle) return;

    m_impl->enum_adapters = reinterpret_cast<PFN_D3DKMTEnumAdapters3>(
        ::dlsym(m_impl->dxcore_handle, "D3DKMTEnumAdapters3"));
    m_impl->query_adapter = reinterpret_cast<PFN_D3DKMTQueryAdapterInfo>(
        ::dlsym(m_impl->dxcore_handle, "D3DKMTQueryAdapterInfo"));
    m_impl->close_adapter = reinterpret_cast<PFN_D3DKMTCloseAdapter>(
        ::dlsym(m_impl->dxcore_handle, "D3DKMTCloseAdapter"));
}

DxgProvider::~DxgProvider()
{
    if(m_impl && m_impl->dxcore_handle) ::dlclose(m_impl->dxcore_handle);
}

std::vector<unique_agent_t>
DxgProvider::enumerate()
{
    std::vector<unique_agent_t> out;

    if(!m_impl || !m_impl->dxcore_handle)
    {
        ROCP_WARNING << "DxgProvider: libdxcore.so not available; returning empty topology";
        return out;
    }

    if(!m_impl->enum_adapters || !m_impl->query_adapter || !m_impl->close_adapter)
    {
        ROCP_WARNING << "DxgProvider: required D3DKMT* symbols missing in libdxcore.so";
        return out;
    }

    DxcEnumAdapters3 e{};
    e.Filter.bits.IncludeComputeOnly = 1;

    NTSTATUS st = m_impl->enum_adapters(&e);
    if(st != kNtSuccess)
    {
        ROCP_WARNING << fmt::format(
            "DxgProvider: D3DKMTEnumAdapters3 (count) failed status=0x{:08x}",
            static_cast<uint32_t>(st));
        return out;
    }

    if(e.NumAdapters == 0)
    {
        ROCP_INFO << "DxgProvider: zero adapters reported by D3DKMTEnumAdapters3";
        return out;
    }

    std::vector<DxcAdapterInfo> infos(e.NumAdapters);
    e.pAdapters = infos.data();
    st          = m_impl->enum_adapters(&e);
    if(st != kNtSuccess)
    {
        ROCP_WARNING << fmt::format(
            "DxgProvider: D3DKMTEnumAdapters3 (fill) failed status=0x{:08x}",
            static_cast<uint32_t>(st));
        return out;
    }

    const auto offset    = get_agent_offset();
    uint64_t   logical   = 0;
    uint64_t   gpu_count = 0;

    for(uint32_t i = 0; i < e.NumAdapters; ++i)
    {
        const auto& a = infos[i];

        DxcQueryDeviceIds devids{};
        if(query_one(m_impl->query_adapter,
                     a.hAdapter,
                     DXC_KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
                     &devids,
                     sizeof(devids)) != kNtSuccess)
        {
            DxcCloseAdapter cl{a.hAdapter};
            m_impl->close_adapter(&cl);
            continue;
        }

        if(devids.DeviceIds.VendorID != 0x1002)
        {
            ROCP_INFO << fmt::format(
                "DxgProvider: skipping non-AMD adapter (vendor=0x{:04x} device=0x{:04x})",
                devids.DeviceIds.VendorID,
                devids.DeviceIds.DeviceID);
            DxcCloseAdapter cl{a.hAdapter};
            m_impl->close_adapter(&cl);
            continue;
        }

        DxcAdapterAddress addr{};
        query_one(
            m_impl->query_adapter, a.hAdapter, DXC_KMTQAITYPE_ADAPTERADDRESS, &addr, sizeof(addr));

        DxcAdapterRegistryInfo reg{};
        query_one(m_impl->query_adapter,
                  a.hAdapter,
                  DXC_KMTQAITYPE_ADAPTERREGISTRYINFO,
                  &reg,
                  sizeof(reg));

        DxcSegmentSizeInfo seg{};
        query_one(
            m_impl->query_adapter, a.hAdapter, DXC_KMTQAITYPE_GETSEGMENTSIZE, &seg, sizeof(seg));

        auto info                 = common::init_public_api_struct(rocprofiler_agent_t{});
        info.type                 = ROCPROFILER_AGENT_TYPE_GPU;
        info.logical_node_id      = logical;
        info.node_id              = static_cast<uint32_t>(logical);
        info.id.handle            = logical + offset;
        info.logical_node_type_id = gpu_count++;
        ++logical;

        info.vendor_id   = devids.DeviceIds.VendorID;
        info.device_id   = devids.DeviceIds.DeviceID;
        info.location_id = ((addr.BusNumber & 0xFF) << 8) | ((addr.DeviceNumber & 0x1F) << 3) |
                           (addr.FunctionNumber & 0x7);
        info.domain         = 0;
        info.local_mem_size = seg.DedicatedVideoMemorySize;
        info.num_xcc        = 1;

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
            "DxgProvider: enumerated adapter {} vendor=0x{:04x} device=0x{:04x} "
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

        DxcCloseAdapter cl{a.hAdapter};
        m_impl->close_adapter(&cl);
    }

    return out;
}

}  // namespace topology
}  // namespace rocprofiler
