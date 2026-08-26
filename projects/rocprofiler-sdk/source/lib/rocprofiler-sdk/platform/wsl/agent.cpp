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
#include "lib/rocprofiler-sdk/platform/wsl/dxg_topology.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

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
#include <set>
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

// Read the CPU marketing string from /proc/cpuinfo "model name". On WSL the HSA
// runtime reports this same string for both the CPU agent's name and its
// device_mkt_name, so mirroring it keeps the synthesized CPU agent aligned with
// HSA (see tests/agent.cpp). Returns an empty string if it cannot be read.
std::string
read_cpu_model_name()
{
    auto ifs = std::ifstream{"/proc/cpuinfo"};
    if(!ifs) return {};

    constexpr auto whitespace = " \t\r\n\v\f";

    std::string line;
    while(std::getline(ifs, line))
    {
        if(line.rfind("model name", 0) != 0) continue;
        auto pos = line.find(':');
        if(pos == std::string::npos) continue;
        auto val = line.substr(pos + 1);
        // parse::strip() returns an all-whitespace value unchanged, so a
        // "model name" line carrying no value has to be rejected here for the
        // caller to fall back to "CPU" instead of naming the agent blank.
        if(val.find_first_not_of(whitespace) == std::string::npos) return {};
        return ::rocprofiler::sdk::parse::strip(std::move(val), whitespace);
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

}  // namespace

bool
is_available()
{
    return probe_libdxcore();
}

std::vector<unique_agent_t>
enumerate()
{
    if(is_late_attach_mode())
    {
        ROCP_WARNING
            << "wsl::enumerate: GPU agent enumeration is disabled for rocprof-attach until "
               "librocdxg provides reference-counted topology snapshots; continuing without "
               "WSL GPU agents";
        return {};
    }

    // On WSL the dxg/libhsakmt path only honors the vendor-specific PM4 IB
    // packets that aqlprofile emits for hardware counter collection when
    // WSLKMT_VENDOR_PACKET is set; otherwise the embedded PM4 IB is silently
    // dropped and every Counter_Value reads back zero. Opt in by default on WSL
    // so counters work out of the box, using overwrite=0 so an explicit user
    // setting always wins.
    //
    // Here rather than in is_available(), because select_platform() does not always ask
    // that question: ROCPROFILER_FORCE_PLATFORM=wsl returns before the probe, and on bare
    // metal gnulinux::is_available() answers first. enumerate() runs exactly when this
    // platform was selected, by either route.
    //
    // This must run before the HSA runtime / libhsakmt initializes its dxg runtime (which
    // reads the variable once per process); agent discovery happens at rocprofiler init,
    // well before any profiling queue is created. The function-local static ensures the
    // setenv happens at most once. NOTE: setenv mutates the global environ and is not
    // itself thread-safe; this is safe here only because it runs on the single-threaded
    // rocprofiler init path before any worker threads exist.
    static const bool _vendor_packet_enabled = []() {
        ::setenv("WSLKMT_VENDOR_PACKET", "1", /*overwrite=*/0);
        return true;
    }();
    (void) _vendor_packet_enabled;

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

    // Read the KMT topology up front, through the same librocdxg interface the
    // HSA runtime uses. Everything a published agent record needs is taken from
    // here, so the records are complete before any consumer can observe them
    // and never change afterwards.
    const auto gpu_nodes = read_dxg_gpu_topology();
    if(gpu_nodes.empty())
    {
        ROCP_WARNING << "wsl::enumerate: the DXG thunk reported no GPU nodes; no GPU agents will "
                        "be published. GPU operation through the HSA runtime requires the same "
                        "thunk, so this environment cannot profile GPU work.";
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

    // KMT nodes already paired with an adapter. Matching is stateless, so
    // without this two adapters that cannot be distinguished by LUID would both
    // resolve to the first node reporting their device id.
    auto consumed_nodes = std::set<uint32_t>{};

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

        // Pair the adapter with its KMT node before anything is assigned. Every
        // compute-topology field below comes from that node; DXCore only
        // contributes the adapter identity, its marketing string and the
        // dedicated VRAM size (which the WSL thunk reports as a memory bank
        // rather than in the node record).
        const auto match = match_node_to_adapter(gpu_nodes,
                                                 consumed_nodes,
                                                 a.AdapterLuid.LowPart,
                                                 a.AdapterLuid.HighPart,
                                                 devids.DeviceIds.DeviceID);
        if(match.ambiguous)
        {
            ROCP_WARNING << fmt::format(
                "wsl::enumerate: discarding adapter {} (vendor=0x{:04x} device=0x{:04x}): several "
                "DXG topology nodes report this device id and neither they nor the adapter carry "
                "a LUID, so this adapter cannot be told apart from an identical one. Guessing "
                "would attribute one GPU's counters to another. A librocdxg that reports adapter "
                "LUIDs resolves this",
                i,
                devids.DeviceIds.VendorID,
                devids.DeviceIds.DeviceID);
            continue;
        }

        const auto* node = match.node;
        if(node == nullptr)
        {
            ROCP_WARNING << fmt::format(
                "wsl::enumerate: discarding adapter {} (vendor=0x{:04x} device=0x{:04x}): no DXG "
                "topology node matched it by LUID or device id, so its compute topology is "
                "unknown",
                i,
                devids.DeviceIds.VendorID,
                devids.DeviceIds.DeviceID);
            continue;
        }

        // Counter collection derives everything from these, so a node that
        // cannot describe them fully is not publishable: aqlprofile divides by
        // simd_per_cu / num_shader_banks / cu_count, and counter definitions
        // are keyed by the gfx target name.
        auto       info        = common::init_public_api_struct(rocprofiler_agent_t{});
        const auto gfx_name    = resolve_gfx_name(node->props);
        const auto gfx_version = ::rocprofiler::agent::parse_gfx_target_version(gfx_name);
        if(!gfx_version)
        {
            ROCP_WARNING << fmt::format(
                "wsl::enumerate: discarding adapter {} (vendor=0x{:04x} device=0x{:04x}): DXG node "
                "{} names gfx target '{}', which this build cannot parse; the node topology itself "
                "is not at fault",
                i,
                devids.DeviceIds.VendorID,
                devids.DeviceIds.DeviceID,
                node->node_id,
                gfx_name);
            continue;
        }

        if(!apply_node_topology(*node, info))
        {
            ROCP_WARNING << fmt::format(
                "wsl::enumerate: discarding adapter {} (vendor=0x{:04x} device=0x{:04x}): DXG node "
                "topology is incomplete (simd_count={} simd_per_cu={} shader_banks={} "
                "arrays_per_engine={} wave_front_size={})",
                i,
                devids.DeviceIds.VendorID,
                devids.DeviceIds.DeviceID,
                node->props.NumFComputeCores,
                node->props.NumSIMDPerCU,
                node->props.NumShaderBanks,
                node->props.NumArrays,
                node->props.WaveFrontSize);
            continue;
        }

        // Claim the node so a later adapter cannot be paired with it too.
        consumed_nodes.emplace(node->node_id);

        info.type = ROCPROFILER_AGENT_TYPE_GPU;
        // logical_node_id is rocprofiler's own dense ordinal across all agent
        // types; node_id is the driver's. Keeping the KMT node id here rather
        // than reusing the ordinal means a skipped adapter shifts the ordinals
        // without renaming the GPUs that were published.
        info.logical_node_id      = logical;
        info.node_id              = node->node_id;
        info.id.handle            = logical + offset;
        info.logical_node_type_id = gpu_type_index;
        ++logical;
        ++gpu_type_index;

        info.name               = common::get_string_entry(gfx_name)->c_str();
        info.gfx_target_version = *gfx_version;

        // Fall back to the DXCore adapter address only if the node left its BDF
        // unset; the node's LocationId is the same value the HSA runtime
        // publishes.
        if(info.location_id == 0)
            info.location_id = ((addr.BusNumber & 0xFF) << 8) | ((addr.DeviceNumber & 0x1F) << 3) |
                               (addr.FunctionNumber & 0x7);
        // The WSL thunk leaves HsaNodeProperties::LocalMemSize zero and reports
        // VRAM as a memory bank instead, so take the size DXCore reports.
        info.local_mem_size = seg.DedicatedVideoMemorySize;

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

        auto adapter_name = wchar_to_utf8(reg.AdapterString, kMaxStr);
        if(adapter_name.empty()) adapter_name = "unknown";

        info.product_name = common::get_string_entry(adapter_name)->c_str();
        info.vendor_name  = common::get_string_entry("AMD")->c_str();
        info.model_name   = common::get_string_entry("")->c_str();

        // Memory banks, caches and IO links are not published on this path; the
        // agent record carries no arrays for them.
        info.mem_banks_count = 0;
        info.caches_count    = 0;
        info.io_links_count  = 0;
        info.mem_banks       = nullptr;
        info.caches          = nullptr;
        info.io_links        = nullptr;

        update_agent_runtime_visibility(info);

        ROCP_INFO << fmt::format(
            "wsl::enumerate: enumerated adapter {} as {} (device=0x{:04x} BDF={:02x}:{:02x}.{:x} "
            "dedicated_vram={} '{}'): cu_count={} simd_count={} simd_per_cu={} shader_banks={} "
            "arrays_per_engine={} array_count={} cu_per_simd_array={} cu_per_engine={} "
            "wave_front_size={} max_waves_per_simd={} num_xcc={} family_id={}",
            i,
            gfx_name,
            devids.DeviceIds.DeviceID,
            addr.BusNumber,
            addr.DeviceNumber,
            addr.FunctionNumber,
            seg.DedicatedVideoMemorySize,
            adapter_name,
            info.cu_count,
            info.simd_count,
            info.simd_per_cu,
            info.num_shader_banks,
            info.simd_arrays_per_engine,
            info.array_count,
            info.cu_per_simd_array,
            info.cu_per_engine,
            info.wave_front_size,
            info.max_waves_per_simd,
            info.num_xcc,
            info.family_id);

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
