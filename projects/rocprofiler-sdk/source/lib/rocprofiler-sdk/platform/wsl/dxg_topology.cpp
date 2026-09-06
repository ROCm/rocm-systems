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
#include "lib/rocprofiler-sdk/platform/wsl/dxg_thunk.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <dlfcn.h>

#include <cstddef>
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
// RAII handle to librocdxg.
//
// The library is normally already resident: the HSA runtime loads it before
// tools are loaded, so agent enumeration during a profiling run finds it with
// RTLD_NOLOAD and just takes a reference on the existing object. Pre-HSA
// consumers (rocprofv3-avail, tool initialization) are the first to touch it
// and fall through to a real dlopen. Either way the handle is refcounted, so
// the close below cannot pull the object out from under the HSA runtime.
struct RocdxgHandle
{
    const DxgLoaderOps& ops;
    void*               handle = nullptr;
    DxgThunk            thunk  = {};

    explicit RocdxgHandle(const DxgLoaderOps& ops_v)
    : ops{ops_v}
    {
        handle = ops.open(kLibRocdxgSoname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if(!handle) handle = ops.open(kLibRocdxgSoname, RTLD_NOW | RTLD_LOCAL);
        if(!handle)
        {
            ROCP_WARNING << fmt::format(
                "wsl topology: dlopen({}) failed: {}. GPU agents cannot be enumerated without "
                "the DXG thunk, which the HSA runtime also requires for GPU operation.",
                kLibRocdxgSoname,
                ops.error());
            return;
        }

        thunk = resolve_dxg_thunk(handle, ops);
    }

    ~RocdxgHandle()
    {
        if(handle) ops.close(handle);
    }

    RocdxgHandle(const RocdxgHandle&) = delete;
    RocdxgHandle& operator=(const RocdxgHandle&) = delete;

    bool ready() const { return handle != nullptr && thunk.complete(); }
};

// Ask the thunk whether it writes HsaNodeProperties to the layout this build
// reads, before any record is read.
//
// hsaKmtGetNodeProperties() has no size parameter: the thunk writes
// sizeof(HsaNodeProperties) bytes as *it* knows the type, into storage sized as
// *this build* knows the type. A longer thunk record overruns the destination
// and a rearranged one misassigns fields, and neither shows up as a failed
// call. HsaStructureSizes is the hsakmt type that exists to negotiate this, so
// asking costs no new interface.
//
// Only an explicit disagreement is fatal. A thunk that does not export the
// handshake predates it, which is a supported configuration and the one this
// port ran in until now.
//
// A matching size is necessary but not sufficient: HsaNodeProperties has also
// been rearranged at an unchanged sizeof - KFDGpuID and FamilyID swapped
// offsets, and apply_node_topology() reads both - and no size comparison can
// see that.
bool
layout_agrees(const DxgThunk& dxg)
{
    if(dxg.abi_check == nullptr)
    {
        ROCP_INFO << fmt::format(
            "wsl topology: {} does not export DxgAbiCheck, so the record layout cannot be "
            "negotiated; proceeding on the assumption that it writes the {}-byte "
            "HsaNodeProperties this build reads",
            kLibRocdxgSoname,
            sizeof(HsaNodeProperties));
        return true;
    }

    // SizeOfHsaExternalHandleDesc is left zero: the thunk reads a zero as "the
    // caller never exchanges that structure", which is true here.
    auto sizes                    = HsaStructureSizes{};
    sizes.StructureSizes          = static_cast<uint16_t>(sizeof(HsaStructureSizes));
    sizes.SizeOfHsaNodeProperties = static_cast<uint16_t>(sizeof(HsaNodeProperties));

    if(auto st = dxg.abi_check(&sizes); st != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_ERROR << fmt::format(
            "wsl topology: {} rejected the HsaNodeProperties layout this build was compiled "
            "against ({} bytes, DxgAbiCheck status={}). No GPU agents will be enumerated from "
            "the KMT topology: a record written to a different layout cannot be interpreted "
            "here, and reading it anyway would attribute whatever the thunk wrote to the wrong "
            "fields. Build rocprofiler-sdk against the hsakmt headers shipped with the ROCr "
            "package that provides this thunk.",
            kLibRocdxgSoname,
            sizeof(HsaNodeProperties),
            static_cast<int>(st));
        return false;
    }

    return true;
}

// Destination for one hsaKmtGetNodeProperties() call, deliberately longer than
// the record this build expects.
//
// The handshake above turns away a thunk that admits to a different size, but a
// thunk that does not export it is taken at its word, and the call bounds
// nothing. The slack means such a thunk overruns padding instead of the rest of
// the stack frame. It does nothing about a same-size rearrangement, which stays
// undetectable from here.
constexpr auto kNodePropsSlack = size_t{128};

struct NodePropsScratch
{
    HsaNodeProperties props                    = {};
    std::byte         overrun[kNodePropsSlack] = {};
};

static_assert(offsetof(NodePropsScratch, overrun) == sizeof(HsaNodeProperties),
              "the slack must directly follow the record it is there to absorb overruns into");
}  // namespace

bool
is_late_attach_mode()
{
    // rocprofiler-register owns this marker and sets it before invoking tool
    // registration. Presence is the signal; do not reinterpret a value copied
    // from the attaching process as permission to enter librocdxg.
    return std::getenv("ROCPROFILER_REGISTER_TOOL_ATTACHED") != nullptr;
}

const DxgLoaderOps&
default_loader_ops()
{
    static const auto ops =
        DxgLoaderOps{[](const char* soname, int flags) { return ::dlopen(soname, flags); },
                     [](void* handle, const char* name) { return ::dlsym(handle, name); },
                     [](void* handle) { return ::dlclose(handle); },
                     []() { return static_cast<const char*>(::dlerror()); }};
    return ops;
}

DxgThunk
resolve_dxg_thunk(void* handle, const DxgLoaderOps& ops)
{
    auto resolve = [&handle, &ops](const char* name) {
        void* sym = ops.sym(handle, name);
        if(!sym)
            ROCP_WARNING << fmt::format(
                "wsl topology: {} does not export {}", kLibRocdxgSoname, name);
        return sym;
    };

    auto thunk             = DxgThunk{};
    thunk.open_kfd         = reinterpret_cast<PFN_hsaKmtOpenKFD>(resolve("hsaKmtOpenKFD"));
    thunk.acquire_snapshot = reinterpret_cast<PFN_hsaKmtAcquireSystemProperties>(
        resolve("hsaKmtAcquireSystemProperties"));
    thunk.get_node =
        reinterpret_cast<PFN_hsaKmtGetNodeProperties>(resolve("hsaKmtGetNodeProperties"));
    thunk.release_snapshot = reinterpret_cast<PFN_hsaKmtReleaseSystemProperties>(
        resolve("hsaKmtReleaseSystemProperties"));
    thunk.close_kfd = reinterpret_cast<PFN_hsaKmtCloseKFD>(resolve("hsaKmtCloseKFD"));

    // Not through resolve(): an older thunk legitimately does not export this,
    // so its absence is reported by layout_agrees() at info level rather than
    // as a missing entry point.
    thunk.abi_check = reinterpret_cast<PFN_DxgAbiCheck>(ops.sym(handle, "DxgAbiCheck"));
    return thunk;
}

std::vector<DxgNode>
read_dxg_gpu_topology(const DxgLoaderOps& ops)
{
    // Keep the late-attach refusal ahead of dlopen as well as every KMT call.
    if(is_late_attach_mode()) return {};

    const auto dxg = RocdxgHandle{ops};
    if(!dxg.ready()) return {};
    return read_dxg_gpu_topology(dxg.thunk);
}

std::vector<DxgNode>
read_dxg_gpu_topology()
{
    return read_dxg_gpu_topology(default_loader_ops());
}

std::vector<DxgNode>
read_dxg_gpu_topology(const DxgThunk& dxg)
{
    auto out = std::vector<DxgNode>{};

    // This overload is the unit-test seam and may also be used by future
    // already-resolved callers. Preserve the same pre-call safety boundary.
    if(is_late_attach_mode()) return out;

    if(!dxg.complete()) return out;

    if(!layout_agrees(dxg)) return out;

    // The thunk refcounts this: SUCCESS means we opened it, KERNEL_ALREADY_OPENED
    // means the HSA runtime (or another consumer) had it open and we took an
    // additional reference. Both require the matching close below.
    if(auto st = dxg.open_kfd();
       st != HSAKMT_STATUS_SUCCESS && st != HSAKMT_STATUS_KERNEL_ALREADY_OPENED)
    {
        ROCP_WARNING << fmt::format("wsl topology: hsaKmtOpenKFD failed (status={})",
                                    static_cast<int>(st));
        return out;
    }

    struct OpenGuard
    {
        const DxgThunk& dxg;
        ~OpenGuard() { dxg.close_kfd(); }
    } _open_guard{dxg};

    // Released external librocdxg packages do not refcount this, unlike the
    // open above. Acquire copies out their one global snapshot without
    // recording a holder, and release drops it outright: topology_drop_snapshot()
    // frees it and deletes the WDDM devices for every consumer, not just this
    // one. (Those packages likewise reset the suballocator on every
    // hsaKmtOpenKFD(), including one that reports KERNEL_ALREADY_OPENED.) The
    // prerequisite in-tree runtime fixes both ownership defects, but exposes no
    // run-time capability bit by which this caller can distinguish it.
    //
    // Safe here with an older external thunk only because normal rocprofv3
    // startup is constructor-ordered: this sequence completes before main(),
    // and so before hsa_init(). rocprof-attach injects into a process that is
    // already running, so is_late_attach_mode() refuses it above before any
    // librocdxg load or call. PR #10034 adds the refcounted snapshot ownership
    // needed to remove that late-attach restriction in a follow-up; it is not a
    // prerequisite for constructor-ordered enumeration in PR #7016.
    //
    // The explicit attach marker is essential because librocdxg exposes no
    // run-time snapshot-ownership capability:
    //
    //   - There is no capability signal to read. DxgAbiCheck negotiates
    //     sizeof(HsaNodeProperties) and nothing else, and which thunks export
    //     it runs the wrong way round for this purpose: the released librocdxg
    //     packages (v1.1.2 through 1.2.x) do export it, while a thunk built
    //     from the in-tree sources does not. Reading it as a refcount bit would
    //     report exactly the shipped thunks - the ones keeping one global
    //     snapshot with no holder count - as the capable ones.
    //   - KERNEL_ALREADY_OPENED counts something else. It reports that
    //     hsaKmtOpenKFD()'s open count was already non-zero, which any second
    //     consumer produces; what matters is whether a snapshot already exists,
    //     and hsaKmtAcquireSystemProperties() answers SUCCESS whether it took a
    //     fresh one or handed back the live one.
    //   - Probing is not free. hsaKmtOpenKFD() resets the shared suballocator
    //     before it returns, on the already-opened path too, and
    //     hsaKmtCloseKFD() does not undo that. Once the acquire below has
    //     happened there is no per-caller release to unwind with at all.
    //
    // Therefore this sequence stays unconditional once the explicit
    // constructor-versus-late-attach decision above has admitted the call.
    auto sys_props = HsaSystemProperties{};
    if(auto st = dxg.acquire_snapshot(&sys_props); st != HSAKMT_STATUS_SUCCESS)
    {
        ROCP_WARNING << fmt::format(
            "wsl topology: hsaKmtAcquireSystemProperties failed (status={})", static_cast<int>(st));
        return out;
    }

    struct SnapshotGuard
    {
        const DxgThunk& dxg;
        ~SnapshotGuard() { dxg.release_snapshot(); }
    } _snapshot_guard{dxg};

    const auto num_nodes = sys_props.NumNodes;

    for(uint32_t node_id = 0; node_id < num_nodes; ++node_id)
    {
        auto scratch = NodePropsScratch{};
        if(auto st = dxg.get_node(node_id, &scratch.props); st != HSAKMT_STATUS_SUCCESS)
        {
            ROCP_WARNING << fmt::format(
                "wsl topology: hsaKmtGetNodeProperties(node={}) failed (status={})",
                node_id,
                static_cast<int>(st));
            continue;
        }

        auto node = DxgNode{node_id, scratch.props};

        if(node.props.NumFComputeCores == 0) continue;  // CPU-only node

        out.emplace_back(node);
    }

    ROCP_INFO << fmt::format(
        "wsl topology: {} of {} KMT nodes report FCompute cores", out.size(), num_nodes);
    return out;
}

NodeMatch
match_node_to_adapter(const std::vector<DxgNode>& nodes,
                      const std::set<uint32_t>&   consumed_node_ids,
                      uint32_t                    luid_low,
                      int32_t                     luid_high,
                      uint32_t                    device_id)
{
    const bool adapter_has_luid = (luid_low != 0 || luid_high != 0);

    auto available = [&consumed_node_ids](const DxgNode& node) {
        return consumed_node_ids.count(node.node_id) == 0;
    };
    auto has_luid = [](const DxgNode& node) {
        return node.props.LuidLowPart != 0 || node.props.LuidHighPart != 0;
    };

    if(adapter_has_luid)
    {
        for(const auto& node : nodes)
        {
            if(!available(node) || !has_luid(node)) continue;
            if(node.props.LuidLowPart == luid_low &&
               node.props.LuidHighPart == static_cast<uint32_t>(luid_high))
                return NodeMatch{&node, false};
        }
    }

    if(device_id == 0) return NodeMatch{};

    // Only nodes a LUID cannot already speak for are eligible. If both sides
    // report LUIDs and they did not match above, they are different GPUs and
    // the shared device id says nothing.
    const DxgNode* candidate = nullptr;
    for(const auto& node : nodes)
    {
        if(!available(node) || node.props.DeviceId != device_id) continue;
        if(adapter_has_luid && has_luid(node)) continue;

        if(candidate != nullptr) return NodeMatch{nullptr, true};
        candidate = &node;
    }

    return NodeMatch{candidate, false};
}

std::string
resolve_gfx_name(const HsaNodeProperties& props)
{
    if(const char* forced = std::getenv("ROCPROFILER_FORCE_GFX");
       forced != nullptr && *forced != '\0')
    {
        if(::rocprofiler::agent::parse_gfx_target_version(forced)) return std::string{forced};

        ROCP_WARNING << "Ignoring malformed ROCPROFILER_FORCE_GFX='" << forced
                     << "'; expected gfx<NNN> with >=3 digits, the last of which may be hex";
    }

    // EngineId and OverrideEngineId are unions over the same bit-field layout,
    // so the version digits come out of the .ui32 view of whichever one speaks.
    const auto& reported   = props.EngineId.ui32;
    const auto& overriding = props.OverrideEngineId.ui32;
    const bool  overridden = (overriding.Major != 0);

    const uint32_t major    = overridden ? overriding.Major : reported.Major;
    const uint32_t minor    = overridden ? overriding.Minor : reported.Minor;
    const uint32_t stepping = overridden ? overriding.Stepping : reported.Stepping;

    if(major == 0) return {};

    return fmt::format("gfx{}{}{:x}", major, minor, stepping);
}

bool
apply_node_topology(const DxgNode& node, rocprofiler_agent_t& info)
{
    const auto& props = node.props;

    if(props.NumFComputeCores == 0 || props.NumSIMDPerCU == 0 || props.NumShaderBanks == 0 ||
       props.NumArrays == 0 || props.WaveFrontSize == 0)
        return false;

    // NumArrays is per shader engine, so the total array count is the product -
    // the same relation the KFD path inverts when it derives num_shader_banks
    // from array_count / simd_arrays_per_engine.
    info.simd_count             = props.NumFComputeCores;
    info.simd_per_cu            = props.NumSIMDPerCU;
    info.cu_count               = props.NumFComputeCores / props.NumSIMDPerCU;
    info.num_shader_banks       = props.NumShaderBanks;
    info.simd_arrays_per_engine = props.NumArrays;
    info.array_count            = props.NumShaderBanks * props.NumArrays;
    info.cu_per_engine          = info.cu_count / props.NumShaderBanks;

    // Derived rather than copied from NumCUPerArray. The two must agree, but
    // this record is published as immutable and counter collection divides by
    // it, so it is defined here by the same relation the rest of these fields
    // satisfy instead of inherited from whatever the thunk computed. A thunk
    // that disagrees is reporting a bug in itself; say so and keep going with
    // the self-consistent value.
    info.cu_per_simd_array = info.cu_count / info.array_count;
    if(props.NumCUPerArray != info.cu_per_simd_array)
        ROCP_WARNING << fmt::format(
            "wsl topology: node {} reports NumCUPerArray={} but {} compute units across {} shader "
            "arrays ({} engines x {}) is {} per array; using the derived value",
            node.node_id,
            props.NumCUPerArray,
            info.cu_count,
            info.array_count,
            props.NumShaderBanks,
            props.NumArrays,
            info.cu_per_simd_array);
    info.wave_front_size      = props.WaveFrontSize;
    info.max_waves_per_simd   = props.MaxWavesPerSIMD;
    info.max_waves_per_cu     = props.NumSIMDPerCU * props.MaxWavesPerSIMD;
    info.max_slots_scratch_cu = props.MaxSlotsScratchCU;
    info.lds_size_in_kb       = props.LDSSizeInKB;
    info.gds_size_in_kb       = props.GDSSizeInKB;
    info.num_gws              = props.NumGws;
    // Every GPU has at least one XCC and aqlprofile divides instance counts by
    // it, so treat an unreported value the way the KFD path treats a missing
    // num_xcc sysfs property.
    info.num_xcc = (props.NumXcc != 0) ? props.NumXcc : 1;

    info.num_sdma_engines           = props.NumSdmaEngines;
    info.num_sdma_xgmi_engines      = props.NumSdmaXgmiEngines;
    info.num_sdma_queues_per_engine = props.NumSdmaQueuesPerEngine;
    info.num_cp_queues              = props.NumCpQueues;
    info.max_engine_clk_fcompute    = props.MaxEngineClockMhzFCompute;
    info.max_engine_clk_ccompute    = props.MaxEngineClockMhzCCompute;

    info.vendor_id                 = props.VendorId;
    info.device_id                 = props.DeviceId;
    info.location_id               = props.LocationId;
    info.domain                    = props.Domain;
    info.family_id                 = props.FamilyID;
    info.capability.Value          = props.Capability.Value;
    info.hive_id                   = props.HiveID;
    info.gpu_id                    = props.KFDGpuID;
    info.fw_version.ui32.uCode     = props.EngineId.ui32.uCode;
    info.sdma_fw_version.uCodeSDMA = props.uCodeEngineVersions.uCodeSDMA;

    auto _uuid       = ::rocprofiler::agent::uuid_view_t{};
    _uuid.value64[0] = props.UniqueID;
    info.uuid        = static_cast<rocprofiler_uuid_t>(_uuid);

    // Workgroup and grid limits are architectural constants the HSA runtime
    // itself hardcodes rather than topology the KMT driver reports, so they are
    // the same values the KFD path publishes. ROCr answers
    // HSA_AGENT_INFO_GRID_MAX_SIZE as min(kern_cluster_max_dim_.x, INT32_MAX),
    // and kern_cluster_max_dim_.x is itself constructed as INT32_MAX and never
    // refreshed from the driver (amd_gpu_agent.cpp), so INT32_MAX is exactly
    // what HSA reports rather than an approximation of it.
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
