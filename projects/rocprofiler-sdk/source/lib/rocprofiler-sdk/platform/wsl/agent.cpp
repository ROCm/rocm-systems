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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

// libhsakmt-windows (librocdxg) headers. These are only available when the shim
// is built/linked, which today is the native-Windows path (gdi32 + matching
// KMD). On the WSL2 / Linux build ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS is left
// undefined, so the shim call below compiles out and enumerate() keeps using
// the hardcoded per-arch fallback.
#if defined(ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS)
#    include <hsakmt/hsakmt.h>
#    include <hsakmt/hsakmttypes.h>
#endif

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

// === libhsakmt-windows topology bridge ===
//
// Holds KFD-equivalent topology fields that DXCore does not surface. Populated
// by fetch_libhsakmt_topology() from HsaNodeProperties on systems where the
// libhsakmt-windows shim (librocdxg) is available; otherwise the caller falls
// back to a hardcoded per-arch table (see enumerate() below).
struct WslTopology
{
    uint32_t cu_count                = 0;
    uint32_t num_shader_banks        = 0;  // HsaNodeProperties.NumShaderBanks
    uint32_t array_count             = 0;  // HsaNodeProperties.NumArrays
    uint32_t simd_arrays_per_engine  = 0;
    uint32_t cu_per_simd_array       = 0;
    uint32_t simd_per_cu             = 0;
    uint32_t simd_count              = 0;  // HsaNodeProperties.NumFComputeCores
    uint32_t wave_front_size         = 0;
    uint32_t max_waves_per_simd      = 0;
    uint32_t max_engine_clk_fcompute = 0;  // HsaNodeProperties.MaxEngineClockMhzFCompute
    uint32_t max_engine_clk_ccompute = 0;  // HsaNodeProperties.MaxEngineClockMhzCCompute
    uint32_t engine_id_major         = 0;  // HsaNodeProperties.EngineId.ui32.Major
    uint32_t engine_id_minor         = 0;
    uint32_t engine_id_stepping      = 0;
};

// Populate the topology fields that DXCore does not expose by invoking the
// libhsakmt-windows shim. DXCore (D3DKMTQueryAdapterInfo) surfaces vendor/device
// id, BDF and local memory size, but not the KFD-equivalent compute topology
// (NumArrays / NumShaderBanks / NumFComputeCores / EngineId / engine clocks).
// The shim exposes the same private KFD escape ABI that Linux uses — populated
// under the hood via D3DKMTEscape — so a single hsaKmtGetNodeProperties() call
// returns the full HsaNodeProperties struct, with no HSA runtime dependency.
//
// `luid` is the DXCore adapter's Windows LUID (DxcAdapterInfo.AdapterLuid); the
// matching KFD node is found by comparing it against HsaNodeProperties.LuidLow/
// HighPart. `adapter` is reserved for future escape calls keyed by the handle.
//
// Returns true and fills `out` on success; false if the shim is not linked in
// this build, the opt-in env var ROCPROFILER_USE_LIBROCDXG is not set, no KFD
// node matches the adapter LUID, or any hsaKmt* call fails.
//
// The shim only links on native Windows (gdi32 + the matching KMD). On the
// WSL2 / Linux build ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS is undefined and this
// unconditionally returns false, so enumerate() keeps using the hardcoded
// per-arch fallback and today's behavior is unchanged.
[[maybe_unused]] bool
fetch_libhsakmt_topology([[maybe_unused]] D3DKMT_HANDLE  adapter,
                         [[maybe_unused]] const DxcLuid& luid,
                         [[maybe_unused]] WslTopology&   out)
{
    // Opt-in only.
    const char* enable = std::getenv("ROCPROFILER_USE_LIBROCDXG");
    if(!enable || std::string{enable} != "1") return false;

#if !defined(ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS)
    // Shim not available in this build (e.g. WSL2 / Linux). No-op.
    return false;
#else
    if(auto _st = hsaKmtOpenKFD(); _st != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_INFO << fmt::format(
            "[libhsakmt-windows] hsaKmtOpenKFD failed status={}; falling back to hardcoded "
            "topology",
            static_cast<int>(_st));
        return false;
    }

    // RAII close so every early return releases the KFD handle / snapshot.
    struct KfdGuard
    {
        ~KfdGuard() { hsaKmtCloseKFD(); }
    } kfd_guard;

    HsaSystemProperties sys{};
    if(auto _st = hsaKmtAcquireSystemProperties(&sys); _st != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_INFO << fmt::format(
            "[libhsakmt-windows] hsaKmtAcquireSystemProperties failed status={}; falling back to "
            "hardcoded topology",
            static_cast<int>(_st));
        return false;
    }

    struct SysGuard
    {
        ~SysGuard() { hsaKmtReleaseSystemProperties(); }
    } sys_guard;

    for(HSAuint32 i = 0; i < sys.NumNodes; ++i)
    {
        HsaNodeProperties n{};
        if(hsaKmtGetNodeProperties(i, &n) != HSAKMT_STATUS_SUCCESS) continue;

        // CPU-only nodes carry no FCompute cores; skip them.
        if(n.NumFComputeCores == 0) continue;

        // Match the KFD node to the DXCore adapter by Windows LUID. KFD reports
        // the same LUID that D3DKMTQueryAdapterInfo returns for the adapter.
        if(n.LuidLowPart != luid.LowPart ||
           n.LuidHighPart != static_cast<HSAuint32>(luid.HighPart))
            continue;

        const uint32_t simd_per_cu = (n.NumSIMDPerCU != 0) ? n.NumSIMDPerCU : 1;

        out.simd_count             = n.NumFComputeCores;               // total SIMDs
        out.cu_count               = n.NumFComputeCores / simd_per_cu; // SIMDs / SIMD-per-CU
        out.simd_per_cu            = n.NumSIMDPerCU;
        out.cu_per_simd_array      = n.NumCUPerArray;
        out.num_shader_banks       = n.NumShaderBanks;                 // shader engines
        // KFD reports NumArrays as SIMD-arrays *per engine*. On single-SE parts
        // (gfx1150) array_count and simd_arrays_per_engine coincide; on multi-SE
        // parts confirm whether the synth-agent table expects a total
        // (NumShaderBanks * NumArrays) here before relying on it.
        out.array_count            = n.NumArrays;
        out.simd_arrays_per_engine = n.NumArrays;
        out.wave_front_size        = n.WaveFrontSize;
        out.max_waves_per_simd     = n.MaxWavesPerSIMD;
        out.max_engine_clk_fcompute = n.MaxEngineClockMhzFCompute;
        out.max_engine_clk_ccompute = n.MaxEngineClockMhzCCompute;
        out.engine_id_major        = n.EngineId.ui32.Major;
        out.engine_id_minor        = n.EngineId.ui32.Minor;
        out.engine_id_stepping     = n.EngineId.ui32.Stepping;

        ROCP_INFO << fmt::format(
            "[libhsakmt-windows] KFD node {} matched adapter LUID; topology: "
            "gfx{}.{}.{} cu_count={} simd_count={} num_shader_banks={} array_count={} "
            "cu_per_simd_array={} simd_per_cu={} wave_front_size={} max_waves_per_simd={} "
            "max_engine_clk_fcompute={}",
            i, out.engine_id_major, out.engine_id_minor, out.engine_id_stepping, out.cu_count,
            out.simd_count, out.num_shader_banks, out.array_count, out.cu_per_simd_array,
            out.simd_per_cu, out.wave_front_size, out.max_waves_per_simd,
            out.max_engine_clk_fcompute);

        return true;
    }

    ROCP_INFO << "fetch_libhsakmt_topology: no KFD node matched the DXCore "
                 "adapter LUID; falling back to hardcoded topology";
    return false;
#endif
}
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

        // DXCore does not expose KFD topology (NumArrays, NumShaderBanks,
        // NumFComputeCores, MaxEngineClockMhz*, etc.). Without these
        // aql_profile::Gfx11Factory::Init() SIGFPEs on integer divide-by-zero.
        //
        // First try to populate them via the libhsakmt-windows shim
        // (hsaKmtOpenKFD + hsaKmtAcquireSystemProperties +
        // hsaKmtGetNodeProperties; see fetch_libhsakmt_topology() above for
        // the planned wiring). When the shim is unavailable, fall back to a
        // hardcoded gfx1150 (RDNA 3.5) topology — gated by the same
        // FORCE_GFX env knob.
        WslTopology topo{};
        if(fetch_libhsakmt_topology(a.hAdapter, a.AdapterLuid, topo))
        {
            info.cu_count                = topo.cu_count;
            info.num_shader_banks        = topo.num_shader_banks;
            info.array_count             = topo.array_count;
            info.simd_arrays_per_engine  = topo.simd_arrays_per_engine;
            info.cu_per_simd_array       = topo.cu_per_simd_array;
            info.simd_per_cu             = topo.simd_per_cu;
            info.simd_count              = topo.simd_count;
            info.wave_front_size         = topo.wave_front_size;
            info.max_waves_per_simd      = topo.max_waves_per_simd;
            info.max_engine_clk_fcompute = topo.max_engine_clk_fcompute;
            info.max_engine_clk_ccompute = topo.max_engine_clk_ccompute;
            // NOTE: gfx_target_version is still derived from ROCPROFILER_FORCE_GFX
            // below, since that env-var override is the canonical source of
            // truth for which per-arch counter YAML rocprofiler-sdk loads. The
            // EngineId from HsaNodeProperties is captured into `topo` for
            // future use (e.g. cross-checking the env override).
        }
        else
        {
            // Hardcoded fallback: gfx1150 (RDNA 3.5) topology.
            info.cu_count               = 16;  // 8 WGPs * 2 CUs
            info.num_shader_banks       = 1;   // 1 SE
            info.simd_arrays_per_engine = 2;   // 2 SAs per SE
            info.array_count            = 2;   // 1 SE * 2 SA
            info.cu_per_simd_array      = 8;   // 16 CUs / 2 SAs
            info.simd_per_cu            = 2;   // RDNA: 2 SIMDs per CU
            info.simd_count             = 32;  // 16 CUs * 2 SIMDs
            info.wave_front_size        = 32;  // RDNA wave32
            info.max_waves_per_simd     = 16;
        }

        auto adapter_name = wchar_to_utf8(reg.AdapterString, kMaxStr);
        if(adapter_name.empty()) adapter_name = "unknown";

        info.product_name = common::get_string_entry(adapter_name)->c_str();
        info.vendor_name  = common::get_string_entry("AMD")->c_str();
        // Counter YAML (config.yaml) is keyed by gfx target (e.g. "gfx1150"),
        // not the marketing product name. WSL DXCore does not expose KFD topology so
        // we can't reliably derive gfx from the PCI device ID; honor an override env
        // var and default to gfx1150 (RDNA 3.5).
        const char* forced_gfx = std::getenv("ROCPROFILER_FORCE_GFX");
        const char* gfx_name   = (forced_gfx && *forced_gfx) ? forced_gfx : "gfx1150";
        info.name              = common::get_string_entry(gfx_name)->c_str();
        info.model_name        = common::get_string_entry("")->c_str();

        // gfx_target_version = major*10000 + minor*100 + step.
        // gfx1150 -> 110500. Parse trailing digits from gfx_name suffix.
        {
            uint32_t    maj = 11, min = 5, stp = 0;
            const char* p = gfx_name;
            if(p[0] == 'g' && p[1] == 'f' && p[2] == 'x') p += 3;
            size_t n = std::strlen(p);
            if(n >= 3)
            {
                stp = static_cast<uint32_t>(p[n - 1] - '0');
                min = static_cast<uint32_t>(p[n - 2] - '0');
                maj = 0;
                for(size_t k = 0; k + 2 < n; ++k)
                    maj = maj * 10 + static_cast<uint32_t>(p[k] - '0');
            }
            info.gfx_target_version = maj * 10000 + min * 100 + stp;
        }

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
