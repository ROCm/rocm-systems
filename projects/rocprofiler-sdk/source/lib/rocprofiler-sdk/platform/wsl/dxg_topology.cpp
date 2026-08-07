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

#include "lib/rocprofiler-sdk/platform/wsl/dxg_topology.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
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
using PFN_DxgAbiCheck                = int32_t (*)(DxgStructureSizes*);
using PFN_DxgGetNodeTopology         = int32_t (*)(uint32_t, uint32_t, DxgNodeTopology*);
using PFN_DxgAcquireTopologySnapshot = int32_t (*)(uint32_t*);
using PFN_DxgReleaseTopologySnapshot = int32_t (*)();
using PFN_hsaKmtOpenKFD              = int32_t (*)();
using PFN_hsaKmtCloseKFD             = int32_t (*)();

// The unversioned soname, matching ThunkLoader::whoami() in the HSA runtime.
// Never a versioned name (librocdxg.so.1 / .so.7): the ABI is negotiated
// through DxgAbiCheck, and hard-coding a soversion would silently stop
// resolving the very object the HSA runtime has loaded.
constexpr const char* kLibRocdxgSoname = "librocdxg.so";

// RAII handle to librocdxg plus the entry points the topology read needs.
//
// The library is normally already resident: the HSA runtime loads it before
// tools are loaded, so agent enumeration during a profiling run finds it with
// RTLD_NOLOAD and just takes a reference on the existing object. Pre-HSA
// consumers (rocprofv3-avail, tool initialization) are the first to touch it
// and fall through to a real dlopen. Either way the handle is refcounted, so
// the dlclose below cannot pull the object out from under the HSA runtime.
struct RocdxgHandle
{
    void* handle = nullptr;

    PFN_DxgAbiCheck                abi_check        = nullptr;
    PFN_DxgGetNodeTopology         get_node         = nullptr;
    PFN_DxgAcquireTopologySnapshot acquire_snapshot = nullptr;
    PFN_DxgReleaseTopologySnapshot release_snapshot = nullptr;
    PFN_hsaKmtOpenKFD              open_kfd         = nullptr;
    PFN_hsaKmtCloseKFD             close_kfd        = nullptr;

    RocdxgHandle()
    {
        handle = ::dlopen(kLibRocdxgSoname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if(!handle) handle = ::dlopen(kLibRocdxgSoname, RTLD_NOW | RTLD_LOCAL);
        if(!handle)
        {
            ROCP_WARNING << fmt::format(
                "wsl topology: dlopen({}) failed: {}. GPU agents cannot be enumerated without "
                "the DXG thunk, which the HSA runtime also requires for GPU operation.",
                kLibRocdxgSoname,
                ::dlerror());
            return;
        }

        auto resolve = [this](const char* name) {
            void* sym = ::dlsym(handle, name);
            if(!sym)
                ROCP_WARNING << fmt::format(
                    "wsl topology: {} does not export {}", kLibRocdxgSoname, name);
            return sym;
        };

        abi_check = reinterpret_cast<PFN_DxgAbiCheck>(resolve("DxgAbiCheck"));
        get_node  = reinterpret_cast<PFN_DxgGetNodeTopology>(resolve("DxgGetNodeTopology"));
        acquire_snapshot =
            reinterpret_cast<PFN_DxgAcquireTopologySnapshot>(resolve("DxgAcquireTopologySnapshot"));
        release_snapshot =
            reinterpret_cast<PFN_DxgReleaseTopologySnapshot>(resolve("DxgReleaseTopologySnapshot"));
        open_kfd  = reinterpret_cast<PFN_hsaKmtOpenKFD>(resolve("hsaKmtOpenKFD"));
        close_kfd = reinterpret_cast<PFN_hsaKmtCloseKFD>(resolve("hsaKmtCloseKFD"));
    }

    ~RocdxgHandle()
    {
        if(handle) ::dlclose(handle);
    }

    RocdxgHandle(const RocdxgHandle&) = delete;
    RocdxgHandle& operator=(const RocdxgHandle&) = delete;

    bool ready() const
    {
        return handle != nullptr && abi_check != nullptr && get_node != nullptr &&
               acquire_snapshot != nullptr && release_snapshot != nullptr && open_kfd != nullptr &&
               close_kfd != nullptr;
    }
};

// Confirm the thunk speaks the ABI this build was compiled against before any
// structure crosses the boundary. Zero means "this caller never exchanges that
// structure": rocprofiler-sdk only uses the size-aware DxgGetNodeTopology, so
// it advertises nothing beyond the size of the descriptor block itself.
bool
check_abi(const RocdxgHandle& dxg)
{
    auto sizes           = DxgStructureSizes{};
    sizes.StructureSizes = sizeof(DxgStructureSizes);

    if(auto st = dxg.abi_check(&sizes); st != kHsaKmtStatusSuccess)
    {
        ROCP_WARNING << fmt::format(
            "wsl topology: DxgAbiCheck rejected this build (status={}); refusing to read the KMT "
            "topology from an ABI-incompatible {}",
            st,
            kLibRocdxgSoname);
        return false;
    }
    return true;
}
}  // namespace

std::vector<DxgNodeTopology>
read_dxg_gpu_topology()
{
    auto out = std::vector<DxgNodeTopology>{};

    const auto dxg = RocdxgHandle{};
    if(!dxg.ready()) return out;
    if(!check_abi(dxg)) return out;

    // The thunk refcounts this: SUCCESS means we opened it, KERNEL_ALREADY_OPENED
    // means the HSA runtime (or another consumer) had it open and we took an
    // additional reference. Both require the matching close below.
    if(auto st = dxg.open_kfd();
       st != kHsaKmtStatusSuccess && st != kHsaKmtStatusKernelAlreadyOpened)
    {
        ROCP_WARNING << fmt::format("wsl topology: hsaKmtOpenKFD failed (status={})", st);
        return out;
    }

    struct OpenGuard
    {
        const RocdxgHandle& dxg;
        ~OpenGuard() { dxg.close_kfd(); }
    } _open_guard{dxg};

    uint32_t num_nodes = 0;
    if(auto st = dxg.acquire_snapshot(&num_nodes); st != kHsaKmtStatusSuccess)
    {
        ROCP_WARNING << fmt::format("wsl topology: DxgAcquireTopologySnapshot failed (status={})",
                                    st);
        return out;
    }

    struct SnapshotGuard
    {
        const RocdxgHandle& dxg;
        ~SnapshotGuard() { dxg.release_snapshot(); }
    } _snapshot_guard{dxg};

    for(uint32_t node_id = 0; node_id < num_nodes; ++node_id)
    {
        auto node = DxgNodeTopology{};
        if(auto st = dxg.get_node(node_id, sizeof(node), &node); st != kHsaKmtStatusSuccess)
        {
            ROCP_WARNING << fmt::format(
                "wsl topology: DxgGetNodeTopology(node={}) failed (status={})", node_id, st);
            continue;
        }

        // The thunk reports what it actually wrote. A short write means it is
        // an older revision than this build expects, so the trailing fields
        // are untouched rather than wrong - reject instead of publishing them.
        if(node.AbiVersion != kDxgNodeTopologyAbiVersion || node.StructSize != sizeof(node))
        {
            ROCP_WARNING << fmt::format(
                "wsl topology: node {} returned topology abi_version={} size={}, this build "
                "expects abi_version={} size={}",
                node_id,
                node.AbiVersion,
                node.StructSize,
                kDxgNodeTopologyAbiVersion,
                sizeof(node));
            continue;
        }

        if(node.NumFComputeCores == 0) continue;  // CPU-only node

        out.emplace_back(node);
    }

    ROCP_INFO << fmt::format(
        "wsl topology: {} of {} KMT nodes report FCompute cores", out.size(), num_nodes);
    return out;
}

const DxgNodeTopology*
match_node_to_adapter(const std::vector<DxgNodeTopology>& nodes,
                      uint32_t                            luid_low,
                      int32_t                             luid_high,
                      uint32_t                            device_id)
{
    const bool adapter_has_luid = (luid_low != 0 || luid_high != 0);

    for(const auto& node : nodes)
    {
        const bool node_has_luid = (node.LuidLowPart != 0 || node.LuidHighPart != 0);

        if(node_has_luid && adapter_has_luid)
        {
            if(node.LuidLowPart == luid_low &&
               node.LuidHighPart == static_cast<uint32_t>(luid_high))
                return &node;
            continue;
        }

        if(device_id != 0 && node.DeviceId == device_id) return &node;
    }

    return nullptr;
}

std::string
resolve_gfx_name(const DxgNodeTopology& node)
{
    if(const char* forced = std::getenv("ROCPROFILER_FORCE_GFX");
       forced != nullptr && *forced != '\0')
    {
        if(::rocprofiler::agent::parse_gfx_target_version(forced)) return std::string{forced};

        ROCP_WARNING << "Ignoring malformed ROCPROFILER_FORCE_GFX='" << forced
                     << "'; expected gfx<NNN> with >=3 decimal digits";
    }

    const bool overridden = (node.OverrideEngineIdMajor != 0);
    const auto major      = overridden ? node.OverrideEngineIdMajor : node.EngineIdMajor;
    const auto minor      = overridden ? node.OverrideEngineIdMinor : node.EngineIdMinor;
    const auto stepping   = overridden ? node.OverrideEngineIdStepping : node.EngineIdStepping;

    if(major == 0) return {};

    return fmt::format("gfx{}{}{:x}", major, minor, stepping);
}

bool
apply_node_topology(const DxgNodeTopology& node, rocprofiler_agent_t& info)
{
    if(node.NumFComputeCores == 0 || node.NumSIMDPerCU == 0 || node.NumShaderBanks == 0 ||
       node.NumArrays == 0 || node.WaveFrontSize == 0)
        return false;

    // NumArrays is per shader engine, so the total array count is the product -
    // the same relation the KFD path inverts when it derives num_shader_banks
    // from array_count / simd_arrays_per_engine.
    info.simd_count             = node.NumFComputeCores;
    info.simd_per_cu            = node.NumSIMDPerCU;
    info.cu_count               = node.NumFComputeCores / node.NumSIMDPerCU;
    info.num_shader_banks       = node.NumShaderBanks;
    info.simd_arrays_per_engine = node.NumArrays;
    info.array_count            = node.NumShaderBanks * node.NumArrays;
    info.cu_per_simd_array      = node.NumCUPerArray;
    info.cu_per_engine          = info.cu_count / node.NumShaderBanks;
    info.wave_front_size        = node.WaveFrontSize;
    info.max_waves_per_simd     = node.MaxWavesPerSIMD;
    info.max_waves_per_cu       = node.NumSIMDPerCU * node.MaxWavesPerSIMD;
    info.max_slots_scratch_cu   = node.MaxSlotsScratchCU;
    info.lds_size_in_kb         = node.LDSSizeInKB;
    info.gds_size_in_kb         = node.GDSSizeInKB;
    info.num_gws                = node.NumGws;
    // Every GPU has at least one XCC and aqlprofile divides instance counts by
    // it, so treat an unreported value the way the KFD path treats a missing
    // num_xcc sysfs property.
    info.num_xcc = (node.NumXcc != 0) ? node.NumXcc : 1;

    info.num_sdma_engines           = node.NumSdmaEngines;
    info.num_sdma_xgmi_engines      = node.NumSdmaXgmiEngines;
    info.num_sdma_queues_per_engine = node.NumSdmaQueuesPerEngine;
    info.num_cp_queues              = node.NumCpQueues;
    info.max_engine_clk_fcompute    = node.MaxEngineClockMhzFCompute;
    info.max_engine_clk_ccompute    = node.MaxEngineClockMhzCCompute;

    info.vendor_id                 = static_cast<uint16_t>(node.VendorId);
    info.device_id                 = static_cast<uint16_t>(node.DeviceId);
    info.location_id               = node.LocationId;
    info.domain                    = node.Domain;
    info.family_id                 = node.FamilyID;
    info.capability.Value          = node.Capability;
    info.hive_id                   = node.HiveID;
    info.gpu_id                    = node.KFDGpuID;
    info.fw_version.ui32.uCode     = node.EngineIdUCode;
    info.sdma_fw_version.uCodeSDMA = node.SdmaUCode;

    auto _uuid       = ::rocprofiler::agent::uuid_view_t{};
    _uuid.value64[0] = node.UniqueID;
    info.uuid        = static_cast<rocprofiler_uuid_t>(_uuid);

    // Workgroup and grid limits are architectural constants the HSA runtime
    // itself hardcodes (see amd_gpu_agent.cpp) rather than topology the KMT
    // driver reports, so they are the same values the KFD path uses.
    constexpr auto workgrp_max = 1024;
    constexpr auto grid_max    = std::numeric_limits<int32_t>::max();
    constexpr auto grid_max_y  = std::numeric_limits<uint16_t>::max();
    constexpr auto grid_max_z  = std::numeric_limits<uint16_t>::max();

    info.workgroup_max_size = workgrp_max;
    info.workgroup_max_dim  = {workgrp_max, workgrp_max, workgrp_max};
    info.grid_max_size      = grid_max;
    info.grid_max_dim       = {grid_max, grid_max_y, grid_max_z};

    return true;
}
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
