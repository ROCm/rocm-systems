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
#include <cwchar>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

// libhsakmt-windows (librocdxg) headers. These are only available when the shim
// is built/linked, which today is the native-Windows path (gdi32 + matching
// KMD). On the WSL2 / Linux build ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS is left
// undefined, so the shim call below compiles out and enumerate() seeds documented
// gfx1150 defaults, which the HSA refinement in agent::construct_agent_cache()
// then overrides with real per-device values at runtime.
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

// Read the CPU marketing string from /proc/cpuinfo "model name". On WSL the HSA
// runtime reports this same string for both the CPU agent's name and its
// device_mkt_name, so mirroring it keeps the synthesized CPU agent aligned with
// HSA (see tests/agent.cpp). Returns an empty string if it cannot be read.
std::string
read_cpu_model_name()
{
    auto ifs = std::ifstream{"/proc/cpuinfo"};
    if(!ifs) return {};

    std::string line;
    while(std::getline(ifs, line))
    {
        if(line.rfind("model name", 0) != 0) continue;
        auto pos = line.find(':');
        if(pos == std::string::npos) continue;
        auto val = line.substr(pos + 1);
        auto b   = val.find_first_not_of(" \t\r\n\v\f");
        auto e   = val.find_last_not_of(" \t\r\n\v\f");
        if(b == std::string::npos) return {};
        return val.substr(b, e - b + 1);
    }
    return {};
}

// Number of online logical CPUs. On WSL this matches the HSA CPU agent's
// reported compute_unit count, which rocprofiler exposes as cu_count /
// cpu_cores_count for CPU agents.
uint32_t
read_cpu_core_count()
{
    long n = ::sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? static_cast<uint32_t>(n) : 0;
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
// libhsakmt-windows shim (librocdxg) is available; otherwise the caller seeds
// documented defaults that the HSA refinement in agent::construct_agent_cache()
// overrides at runtime (see enumerate() below).
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
    // Identity fields the shim also reports, so agent info needs no HSA runtime.
    uint32_t family_id  = 0;  // HsaNodeProperties.FamilyID
    uint32_t num_xcc    = 0;  // HsaNodeProperties.NumXcc
    uint32_t domain     = 0;  // HsaNodeProperties.Domain
    uint32_t fw_ucode   = 0;  // HsaNodeProperties.EngineId.ui32.uCode
    uint32_t sdma_ucode = 0;  // HsaNodeProperties.uCodeEngineVersions.uCodeSDMA
    bool     valid      = false;
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
// unconditionally returns false, so enumerate() seeds documented gfx1150 defaults
// that the HSA refinement in agent::construct_agent_cache() overrides at runtime.
[[maybe_unused]] bool
fetch_libhsakmt_topology([[maybe_unused]] D3DKMT_HANDLE  adapter,
                         [[maybe_unused]] const DxcLuid& luid,
                         [[maybe_unused]] uint32_t       device_id,
                         [[maybe_unused]] WslTopology&   out)
{
    // Default-on when the shim is compiled in (the intended primary source of
    // topology on WSL); set ROCPROFILER_USE_LIBROCDXG=0 to force the pre-HSA
    // placeholder + HSA-backfill fallback path instead.
    if(const char* enable = std::getenv("ROCPROFILER_USE_LIBROCDXG");
       enable != nullptr && std::string{enable} == "0")
        return false;

#if !defined(ROCPROFILER_HAVE_LIBHSAKMT_WINDOWS)
    // Shim not available in this build (e.g. WSL2 / Linux). No-op.
    return false;
#else
    if(auto _st = hsaKmtOpenKFD(); _st != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_INFO << fmt::format(
            "[libhsakmt-windows] hsaKmtOpenKFD failed status={}; falling back to documented "
            "defaults (refined from HSA at runtime)",
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
            "documented defaults (refined from HSA at runtime)",
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

        // Match the KFD node to the DXCore adapter. Prefer the Windows LUID (the
        // multi-GPU-safe key: KFD reports the same LUID D3DKMTQueryAdapterInfo
        // returns for the adapter). Some shim builds do not populate the node LUID
        // yet (LuidLowPart/HighPart == 0); in that case fall back to matching by PCI
        // DeviceId, which the shim always fills. WSL is typically single-GPU, so the
        // DeviceId fallback is unambiguous there. Skip nodes that match neither.
        const bool node_has_luid   = (n.LuidLowPart != 0 || n.LuidHighPart != 0);
        const bool adapter_has_luid = (luid.LowPart != 0 || luid.HighPart != 0);
        if(node_has_luid && adapter_has_luid)
        {
            if(n.LuidLowPart != luid.LowPart ||
               n.LuidHighPart != static_cast<HSAuint32>(luid.HighPart))
                continue;
        }
        else if(device_id != 0)
        {
            // LUID unavailable on the node or adapter: fall back to DeviceId.
            if(static_cast<uint32_t>(n.DeviceId) != device_id) continue;
        }
        else
        {
            // Neither a usable LUID nor a DeviceId to match on.
            continue;
        }

        const uint32_t simd_per_cu = (n.NumSIMDPerCU != 0) ? n.NumSIMDPerCU : 1;

        out.simd_count        = n.NumFComputeCores;                // total SIMDs
        out.cu_count          = n.NumFComputeCores / simd_per_cu;  // SIMDs / SIMD-per-CU
        out.simd_per_cu       = n.NumSIMDPerCU;
        out.cu_per_simd_array = n.NumCUPerArray;
        out.num_shader_banks  = n.NumShaderBanks;  // shader engines
        // KFD reports NumArrays as SIMD-arrays *per engine*. On single-SE parts
        // (gfx1150) array_count and simd_arrays_per_engine coincide; on multi-SE
        // parts confirm whether the synth-agent table expects a total
        // (NumShaderBanks * NumArrays) here before relying on it.
        out.array_count             = n.NumArrays;
        out.simd_arrays_per_engine  = n.NumArrays;
        out.wave_front_size         = n.WaveFrontSize;
        out.max_waves_per_simd      = n.MaxWavesPerSIMD;
        out.max_engine_clk_fcompute = n.MaxEngineClockMhzFCompute;
        out.max_engine_clk_ccompute = n.MaxEngineClockMhzCCompute;
        out.engine_id_major         = n.EngineId.ui32.Major;
        out.engine_id_minor         = n.EngineId.ui32.Minor;
        out.engine_id_stepping      = n.EngineId.ui32.Stepping;
        // Identity fields — the shim reports these too, so the agent needs no HSA
        // runtime to be fully populated up front.
        out.family_id  = n.FamilyID;
        out.num_xcc    = n.NumXcc;
        out.domain     = n.Domain;
        out.fw_ucode   = n.EngineId.ui32.uCode;
        out.sdma_ucode = n.uCodeEngineVersions.uCodeSDMA;
        out.valid      = true;

        ROCP_INFO << fmt::format(
            "[libhsakmt-windows] KFD node {} matched adapter (LUID or DeviceId); topology: "
            "gfx{}.{}.{} cu_count={} simd_count={} num_shader_banks={} array_count={} "
            "cu_per_simd_array={} simd_per_cu={} wave_front_size={} max_waves_per_simd={} "
            "max_engine_clk_fcompute={}",
            i,
            out.engine_id_major,
            out.engine_id_minor,
            out.engine_id_stepping,
            out.cu_count,
            out.simd_count,
            out.num_shader_banks,
            out.array_count,
            out.cu_per_simd_array,
            out.simd_per_cu,
            out.wave_front_size,
            out.max_waves_per_simd,
            out.max_engine_clk_fcompute);

        return true;
    }

    ROCP_INFO << "fetch_libhsakmt_topology: no KFD node matched the DXCore adapter "
                 "(by LUID or DeviceId); falling back to documented defaults (refined from "
                 "HSA at runtime)";
    return false;
#endif
}
}  // namespace

bool
is_available()
{
    const bool avail = probe_libdxcore();

    // On WSL the dxg/libhsakmt path only honors the vendor-specific PM4 IB
    // packets that aqlprofile emits for hardware counter collection when
    // WSLKMT_VENDOR_PACKET is set; otherwise the embedded PM4 IB is silently
    // dropped and every Counter_Value reads back zero. Opt in by default on WSL
    // so counters work out of the box, using overwrite=0 so an explicit user
    // setting always wins. This must run before the HSA runtime / libhsakmt
    // initializes its dxg runtime (which reads the variable once per process);
    // is_available() is called during agent discovery at rocprofiler init, well
    // before any profiling queue is created. The function-local static ensures
    // the setenv happens at most once. NOTE: setenv mutates the global environ
    // and is not itself thread-safe; this is safe here only because it runs on
    // the single-threaded rocprofiler init path before any worker threads exist.
    if(avail)
    {
        static const bool _vendor_packet_enabled = []() {
            ::setenv("WSLKMT_VENDOR_PACKET", "1", /*overwrite=*/0);
            return true;
        }();
        (void) _vendor_packet_enabled;
    }

    return avail;
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

    // Deleter shared by every agent we emplace into `out`. Frees the
    // heap-allocated topology arrays (when present) and the agent struct.
    auto agent_deleter = [](rocprofiler_agent_t* p) {
        if(p)
        {
            delete[] p->mem_banks;
            delete[] p->caches;
            delete[] p->io_links;
        }
        delete p;
    };

    // Synthesize a CPU agent at logical_node_id=0.
    //
    // The HSA runtime on WSL always enumerates a CPU agent (internal_node_id=0)
    // in addition to the GPU (internal_node_id=1). rocprofiler-sdk asserts that
    // every HSA agent's internal node id maps to a rocprofiler agent's
    // logical_node_id (see source/lib/rocprofiler-sdk/agent.cpp); without a CPU
    // agent here the GPU ends up at logical_node_id=0 and HSA's GPU node id (1)
    // has no match, aborting with a fatal node-id mismatch. DXCore only reports
    // GPU adapters, so we add the CPU agent ourselves and shift the GPU agents
    // to start at logical_node_id=1 to stay aligned with HSA enumeration.
    {
        auto cpu            = common::init_public_api_struct(rocprofiler_agent_t{});
        cpu.type            = ROCPROFILER_AGENT_TYPE_CPU;
        cpu.logical_node_id = logical;
        cpu.node_id         = static_cast<uint32_t>(logical);
        cpu.id.handle       = logical + offset;
        // logical_node_type_id is an index within agents of the same type; this
        // is the first (and only) CPU agent.
        cpu.logical_node_type_id = 0;
        ++logical;

        // HSA reports the CPU agent's vendor as "CPU" and uses the /proc/cpuinfo
        // "model name" for both its name and marketing name. Match that so the
        // agent-info checks (cu_count, name, product_name, Cpu_Cores_Count) pass.
        const auto        cpu_cores = read_cpu_core_count();
        const std::string cpu_name  = []() {
            auto m = read_cpu_model_name();
            return m.empty() ? std::string{"CPU"} : m;
        }();

        cpu.vendor_name  = common::get_string_entry("CPU")->c_str();
        cpu.product_name = common::get_string_entry(cpu_name)->c_str();
        cpu.name         = common::get_string_entry(cpu_name)->c_str();
        cpu.model_name   = common::get_string_entry("")->c_str();

        // HSA's CPU agent reports compute_unit == online logical core count;
        // rocprofiler exposes that as cu_count and cpu_cores_count for CPUs.
        cpu.cpu_cores_count = cpu_cores;
        cpu.cu_count        = cpu_cores;

        cpu.mem_banks_count = 0;
        cpu.caches_count    = 0;
        cpu.io_links_count  = 0;
        cpu.mem_banks       = nullptr;
        cpu.caches          = nullptr;
        cpu.io_links        = nullptr;

        std::memset(&cpu.uuid.bytes, 0, sizeof(cpu.uuid.bytes));

        update_agent_runtime_visibility(cpu);

        ROCP_INFO << fmt::format(
            "wsl::enumerate: synthesized CPU agent at logical_node_id={} (cores={}, name='{}') to "
            "align with HSA runtime enumeration (CPU internal_node_id=0)",
            cpu.logical_node_id,
            cpu_cores,
            cpu_name);

        out.emplace_back(new rocprofiler_agent_t{cpu}, agent_deleter);
    }

    // Index within GPU-type agents only (0-based). Distinct from logical_node_id,
    // which is shared across all agent types (the synthesized CPU occupies 0, so
    // the GPU's logical_node_id starts at 1). VISIBLE_DEVICES filtering matches on
    // logical_node_type_id, so the first GPU must be 0 here regardless of the CPU.
    uint32_t gpu_type_index = 0;

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
        info.logical_node_type_id = gpu_type_index;
        ++logical;
        ++gpu_type_index;

        info.vendor_id   = devids.DeviceIds.VendorID;
        info.device_id   = devids.DeviceIds.DeviceID;
        info.location_id = ((addr.BusNumber & 0xFF) << 8) | ((addr.DeviceNumber & 0x1F) << 3) |
                           (addr.FunctionNumber & 0x7);
        info.local_mem_size = seg.DedicatedVideoMemorySize;
        // num_xcc / domain are not surfaced by DXCore. Seed documented defaults at
        // enumeration time so pre-HSA consumers (rocprofv3-avail) are not zero; the
        // HSA refinement in agent::construct_agent_cache() overrides them at runtime.
        info.num_xcc = 1;
        info.domain  = 0;

        // Resolve the gfx target name + version AT ENUMERATION TIME. These must
        // never be empty/zero for an enumerated GPU agent: counter-metric
        // (config.yaml) resolution and pre-HSA tools such as rocprofv3-avail (which
        // list agents without starting the HSA runtime, so they never run the HSA
        // refinement in agent::construct_agent_cache()) read agent->name /
        // gfx_target_version directly. An empty architecture makes metric lookup
        // fail with "Agent HW architecture is not supported".
        //
        // DXCore does not expose the gfx target, so default to the only
        // WSL-validated target (gfx1150, RDNA 3.5); ROCPROFILER_FORCE_GFX overrides
        // it (see docs/ for the WSL env vars). For other GPUs run without
        // FORCE_GFX, the HSA refinement later corrects name/version at runtime; the
        // gfx1150 default only governs the brief pre-HSA window (and avail).
        static constexpr std::string_view kDefaultGfxName = "gfx1150";

        std::string_view gfx_name = kDefaultGfxName;
        if(const char* forced_gfx = std::getenv("ROCPROFILER_FORCE_GFX");
           forced_gfx != nullptr && *forced_gfx != '\0')
        {
            if(::rocprofiler::agent::parse_gfx_target_version(forced_gfx))
                gfx_name = forced_gfx;
            else
                ROCP_WARNING << "Ignoring malformed ROCPROFILER_FORCE_GFX='" << forced_gfx
                             << "'; expected gfx<NNN> with >=3 decimal digits. Falling back to "
                             << kDefaultGfxName;
        }

        info.name               = common::get_string_entry(std::string{gfx_name})->c_str();
        info.gfx_target_version = ::rocprofiler::agent::parse_gfx_target_version(gfx_name).value();

        // DXCore does not expose KFD topology (NumArrays, NumShaderBanks,
        // NumFComputeCores, MaxEngineClockMhz*, etc.). Without non-zero topology
        // aql_profile::Gfx11Factory::Init() SIGFPEs on integer divide-by-zero, and
        // pre-HSA tools (rocprofv3-avail) print zeros.
        //
        // First try the libhsakmt-windows shim (hsaKmtOpenKFD +
        // hsaKmtAcquireSystemProperties + hsaKmtGetNodeProperties). When the shim
        // is unavailable (the plain WSL2 / Linux build), seed the documented
        // gfx1150 (RDNA 3.5) defaults so the fields are never zero at enumeration
        // time; the HSA refinement in agent::construct_agent_cache() then overrides
        // them with the real per-device values once the HSA agent is known. (HSA is
        // not yet initialized here, so it cannot be queried at this point.)
        WslTopology topo{};
        if(fetch_libhsakmt_topology(a.hAdapter, a.AdapterLuid, info.device_id, topo))
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

            // Identity fields from the shim too, so the agent is fully populated
            // without the HSA runtime. The gfx target name/version come from the
            // shim's EngineId (major.minor.stepping); an explicit, valid
            // ROCPROFILER_FORCE_GFX (resolved into gfx_name above) still wins.
            info.num_xcc = (topo.num_xcc != 0) ? topo.num_xcc : info.num_xcc;
            info.domain  = topo.domain;
            if(topo.family_id != 0) info.family_id = topo.family_id;
            if(topo.fw_ucode != 0) info.fw_version.ui32.uCode = topo.fw_ucode;
            if(topo.sdma_ucode != 0) info.sdma_fw_version.uCodeSDMA = topo.sdma_ucode;

            const bool force_gfx =
                (std::getenv("ROCPROFILER_FORCE_GFX") != nullptr &&
                 *std::getenv("ROCPROFILER_FORCE_GFX") != '\0' &&
                 ::rocprofiler::agent::parse_gfx_target_version(std::getenv("ROCPROFILER_FORCE_GFX"))
                     .has_value());
            if(!force_gfx && topo.engine_id_major != 0)
            {
                auto shim_gfx = fmt::format(
                    "gfx{}{}{:x}", topo.engine_id_major, topo.engine_id_minor, topo.engine_id_stepping);
                if(auto _ver = ::rocprofiler::agent::parse_gfx_target_version(shim_gfx))
                {
                    info.name               = common::get_string_entry(shim_gfx)->c_str();
                    info.gfx_target_version = *_ver;
                }
            }

            // Real topology obtained up front — no HSA dependency. Tell
            // construct_agent_cache() to skip its WSL HSA backfill.
            ::rocprofiler::agent::set_wsl_topology_from_shim(true);
        }
        else
        {
            // Neutral placeholder topology — NOT a real device layout and
            // intentionally does NOT impersonate any architecture (e.g. gfx1150
            // CU/SIMD counts would yield absurd counter values on gfx940).
            // DXCore cannot expose KFD shader counts, so these are minimal
            // non-zero sentinels whose only job is to keep enumerate()'s derived
            // fields (cu_per_engine, max_waves_per_cu) and aqlprofile agent
            // registration (GpuPmcBuilder) from dividing by zero before HSA is
            // initialized. agent::construct_agent_cache() overrides every field
            // below with the real per-device topology from HSA at runtime, so
            // profiling never relies on these values. The agent name /
            // gfx_target_version resolved above still select the counter YAML for
            // pre-HSA rocprofv3-avail; they deliberately do NOT assume a layout.
            //
            // cu_count must be >= 2 (one WGP): aqlprofile's GpuPmcBuilder derives
            // wgp_per_sa_ as (cu_num/2 + ...) / ..., so cu_num == 1 yields zero
            // WGPs and a divide-by-zero (SIGFPE) when iterating event coordinates.
            // Everything else is the smallest internally consistent layout (1 SE,
            // 1 SA per SE, 1 WGP), which is the minimum that stays divide-safe.
            info.cu_count               = 2;
            info.num_shader_banks       = 1;
            info.simd_arrays_per_engine = 1;
            info.array_count            = 1;
            info.cu_per_simd_array      = 2;
            info.simd_per_cu            = 1;
            info.simd_count             = 2;
            info.wave_front_size        = 1;
            info.max_waves_per_simd     = 1;
        }

        // Fields not surfaced by DXCore but reported by the HSA runtime for
        // gfx11 (RDNA3/3.5). Without these the rocprofiler agent diverges from
        // HSA enumeration and tests/agent.cpp fails its field-by-field compare.
        //
        // max_waves_per_cu and cu_per_engine are computed the same way the
        // gnulinux KFD path computes them, from the topology fields populated
        // above.
        if(info.simd_per_cu > 0) info.max_waves_per_cu = info.simd_per_cu * info.max_waves_per_simd;
        if(info.simd_per_cu > 0 && info.num_shader_banks > 0)
            info.cu_per_engine = (info.simd_count / info.simd_per_cu) / info.num_shader_banks;

        // Register the topology properties we populated above so the counter
        // subsystem can expose them as constants. get_constants() in
        // metrics.cpp iterates get_agent_available_properties() to build the
        // constant pseudo-metrics (simd_count, simd_per_cu, ...) that counter
        // expressions reference. The gnulinux/KFD path registers these as a
        // side effect of read_property(); on the synthesized WSL path the
        // fields are assigned directly, so the names must be registered
        // explicitly here. Without this the constants are never created and
        // evaluation fails with "Unable to lookup metric <name>".
        for(const char* prop : {"array_count",
                                "simd_count",
                                "wave_front_size",
                                "simd_arrays_per_engine",
                                "cu_per_simd_array",
                                "simd_per_cu",
                                "max_waves_per_simd"})
        {
            ::rocprofiler::agent::get_agent_available_properties().insert(prop);
        }

        // workgroup/grid limits are not exposed by DXCore but are fixed by the
        // HSA runtime for gfx11; seed them here so pre-HSA consumers are non-zero.
        // The HSA refinement in agent::construct_agent_cache() overrides them at
        // runtime.
        info.workgroup_max_size = 1024;
        info.workgroup_max_dim  = {1024, 1024, 1024};
        info.grid_max_size      = std::numeric_limits<uint32_t>::max();
        info.grid_max_dim       = {2147483647u, 65535u, 65535u};

        // family code and firmware versions: on the shim path these were already
        // filled from HsaNodeProperties above, so preserve them. Only when the shim
        // did not supply them (fallback path) leave them zero rather than impersonate
        // a specific architecture; agent::construct_agent_cache() then fills them from
        // HSA at runtime, which is also what tests/agent.cpp compares against.
        if(!::rocprofiler::agent::wsl_topology_from_shim())
        {
            info.family_id                 = 0;
            info.fw_version.ui32.uCode     = 0;
            info.sdma_fw_version.uCodeSDMA = 0;
        }

        auto adapter_name = wchar_to_utf8(reg.AdapterString, kMaxStr);
        if(adapter_name.empty()) adapter_name = "unknown";

        info.product_name = common::get_string_entry(adapter_name)->c_str();
        info.vendor_name  = common::get_string_entry("AMD")->c_str();
        // info.name (the gfx target string config.yaml is keyed by) was resolved
        // at enumeration time above (gfx1150 default or ROCPROFILER_FORCE_GFX).
        info.model_name = common::get_string_entry("")->c_str();

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
