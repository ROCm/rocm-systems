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

#include "lib/rocprofiler-sdk/platform/windows/agent.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/platform/windows/d3dkmt_loader.hpp"
#include "lib/rocprofiler-sdk/platform/windows/wkmi_bridge.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace windows
{
#ifdef _WIN32
namespace
{
using ::rocprofiler::agent::update_agent_runtime_visibility;
using ::rocprofiler::agent::uuid_view_t;

constexpr uint32_t kAmdVendorId = 0x1002;

// Random per-process offset applied to rocprofiler_agent_id_t.handle. Kept
// identical to the gnulinux/wsl paths so agent IDs are non-stable across runs
// and downstream code cannot accidentally treat them as ordinals.
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
}  // namespace

bool
is_available()
{
    // The native Windows path is viable iff the D3DKMT entry points resolve.
    // Cache the probe in a function-local static so repeated calls do not
    // re-LoadLibrary/GetProcAddress every time; this mirrors the WSL cached
    // probe (dlopen + symbol check) and respects the ROCPROFILER_D3DKMT_MODULE
    // override (resolved once, on first call).
    static const bool _v = []() {
        auto loader = d3dkmt_loader{};
        return loader.ready();
    }();
    return _v;
}

std::vector<unique_agent_t>
enumerate()
{
    auto out = std::vector<unique_agent_t>{};

    auto loader = d3dkmt_loader{};
    if(!loader.ready())
    {
        ROCP_WARNING << "windows::enumerate: D3DKMT entry points unavailable; returning empty "
                        "topology";
        return out;
    }

    ROCP_INFO << fmt::format("windows::enumerate: wkmi {} this build",
                             wkmi_is_present() ? "linked into" : "absent from");

    auto adapters = loader.enumerate_adapters();
    if(adapters.empty())
    {
        ROCP_INFO << "windows::enumerate: no adapters enumerated by D3DKMTEnumAdapters3";
        return out;
    }

    const auto offset = get_agent_offset();
    // Every adapter enumerated through compute-only D3DKMT is a GPU, so the
    // logical node id and the per-type id move in lockstep (WSL parity).
    uint64_t logical = 0;

    for(uint32_t i = 0; i < adapters.size(); ++i)
    {
        const auto& a = adapters[i];

        // Close the adapter on every exit path (early continues, query
        // failures, exceptions from new below).
        auto _closer =
            common::scope_destructor{[&loader, h = a.hadapter]() { loader.close_adapter(h); }};

        auto devids = d3dkmt_device_ids{};
        if(!loader.query_device_ids(a.hadapter, devids))
        {
            continue;
        }

        if(devids.vendor_id != kAmdVendorId)
        {
            ROCP_INFO << fmt::format(
                "windows::enumerate: skipping non-AMD adapter (vendor=0x{:04x} device=0x{:04x})",
                devids.vendor_id,
                devids.device_id);
            continue;
        }

        // wkmi_adapter_supported() returns false in a stub build (wkmi absent),
        // which would silently discard every adapter and contradict the intended
        // D3DKMT-only degraded mode. Only apply the wkmi support gate when wkmi
        // is actually linked.
        if(wkmi_is_present() && !wkmi_adapter_supported(devids.device_id))
        {
            ROCP_INFO << fmt::format(
                "windows::enumerate: skipping unsupported AMD adapter (device=0x{:04x})",
                devids.device_id);
            continue;
        }

        // Basic fields straight from D3DKMT; wkmi values override below when
        // available. BDF/VRAM are required to publish a sensible agent, so a
        // failure here discards the adapter rather than half-filling it (WSL
        // parity).
        auto addr = d3dkmt_adapter_address{};
        if(!loader.query_adapter_address(a.hadapter, addr))
        {
            ROCP_WARNING << fmt::format("windows::enumerate: discarding adapter {} "
                                        "(device=0x{:04x}): ADAPTERADDRESS query failed",
                                        i,
                                        devids.device_id);
            continue;
        }

        auto seg = d3dkmt_segment_size{};
        if(!loader.query_segment_size(a.hadapter, seg))
        {
            ROCP_WARNING << fmt::format("windows::enumerate: discarding adapter {} "
                                        "(device=0x{:04x}): GETSEGMENTSIZE query failed",
                                        i,
                                        devids.device_id);
            continue;
        }

        // Detailed fields from wkmi. wkmi is the source of truth for the KFD-
        // parity topology; a parse failure (or a stub build) leaves wk default-
        // constructed and the agent falls back to D3DKMT basics only.
        auto wk             = wkmi_device_info{};
        bool have_wkmi_info = wkmi_parse_adapter(a.hadapter, wk);

        auto info                 = common::init_public_api_struct(rocprofiler_agent_t{});
        info.type                 = ROCPROFILER_AGENT_TYPE_GPU;
        info.logical_node_id      = logical;
        info.node_id              = static_cast<uint32_t>(logical);
        info.id.handle            = logical + offset;
        info.logical_node_type_id = logical;
        ++logical;

        // ---- basic identity / addressing (D3DKMT, wkmi preferred) ----
        info.vendor_id = devids.vendor_id;
        info.device_id = (have_wkmi_info && wk.device_id != 0) ? wk.device_id : devids.device_id;

        // location_id is packed bus/device/function. Prefer wkmi's already-
        // packed pci_bus_addr; fall back to the D3DKMT ADAPTERADDRESS triple.
        if(have_wkmi_info && wk.pci_bus_addr != 0)
        {
            info.location_id = wk.pci_bus_addr;
        }
        else
        {
            info.location_id = ((addr.bus_number & 0xFF) << 8) |
                               ((addr.device_number & 0x1F) << 3) | (addr.function_number & 0x7);
        }

        info.domain = have_wkmi_info ? wk.domain : 0;

        // local_mem_size: prefer wkmi visible+invisible heaps; fall back to the
        // D3DKMT dedicated video memory size.
        if(have_wkmi_info && (wk.local_visible_heap_size != 0 || wk.local_invisible_heap_size != 0))
        {
            info.local_mem_size = wk.local_visible_heap_size + wk.local_invisible_heap_size;
        }
        else
        {
            info.local_mem_size = seg.dedicated_video_memory;
        }

        info.num_xcc = (have_wkmi_info && wk.num_xcc > 0) ? wk.num_xcc : 1;

        // ---- detailed compute topology (wkmi) ----
        if(have_wkmi_info)
        {
            info.gfx_target_version =
                (wk.major * 10000) + (wk.minor * 100) + static_cast<uint32_t>(wk.stepping);

            info.family_id       = wk.family;
            info.simd_per_cu     = wk.simd_per_cu;
            info.cu_count        = wk.compute_unit_count;
            info.simd_count      = wk.compute_unit_count * wk.simd_per_cu;
            info.wave_front_size = wk.wavefront_size;

            // wkmi.h documents wave_per_cu as "Maximum waves per compute unit",
            // matching max_waves_per_cu directly. max_waves_per_simd is derived
            // from it (gnulinux reads max_waves_per_simd from KFD sysfs and
            // derives max_waves_per_cu = simd_per_cu * max_waves_per_simd; here
            // the direction is reversed but the semantics are equivalent).
            info.max_waves_per_cu = wk.wave_per_cu;

            // Guard divisions against zero exactly like gnulinux.
            if(wk.simd_per_cu > 0) info.max_waves_per_simd = wk.wave_per_cu / wk.simd_per_cu;

            // array_count = shader engines * shader arrays per engine.
            info.array_count            = wk.num_shader_engine * wk.shader_array_per_shader_engine;
            info.simd_arrays_per_engine = wk.shader_array_per_shader_engine;

            // cu_per_simd_array: CUs distributed evenly across all shader
            // arrays. gnulinux reads this from KFD directly; wkmi does not
            // expose it, so derive it. Guard the divisor against zero.
            if(info.array_count > 0)
                info.cu_per_simd_array = wk.compute_unit_count / info.array_count;

            // num_shader_banks = array_count / simd_arrays_per_engine (gnulinux
            // semantics). Guard against a zero divisor.
            if(info.simd_arrays_per_engine > 0)
                info.num_shader_banks = info.array_count / info.simd_arrays_per_engine;

            // cu_per_engine = CUs per shader engine. gnulinux computes
            // (simd_count / simd_per_cu) / num_shader_banks; since
            // simd_count / simd_per_cu == compute_unit_count here, this reduces
            // to compute_unit_count / num_shader_banks. Guard the divisor.
            if(info.num_shader_banks > 0)
                info.cu_per_engine = wk.compute_unit_count / info.num_shader_banks;

            info.max_engine_clk_fcompute = wk.max_engine_clock_mhz;
            info.lds_size_in_kb          = wk.lds_size / 1024;
            info.max_slots_scratch_cu    = wk.max_scratch_slots_per_cu;
            info.num_gws                 = wk.num_gws;
            info.num_cp_queues           = wk.num_cp_queues;

            // Firmware versions: KFD masks to the low 10 bits (gnulinux parity).
            info.fw_version.Value      = wk.mec_fw_version & 0x3ff;
            info.sdma_fw_version.Value = wk.sdma_fw_version & 0x3ff;

            // info.capability: gnulinux reads capability.Value from the KFD
            // sysfs node. Neither D3DKMT nor the current wkmi surface exposes an
            // equivalent, so it is intentionally left at the
            // init_public_api_struct default (0). TODO: investigate a wkmi /
            // KMTQAITYPE source for the HSA capability bits and populate here.
        }

        // ---- names ----
        // name is the gfx string (gnulinux semantics), product_name is the
        // marketing string. They are kept distinct.
        if(info.gfx_target_version >= 10000)
        {
            auto major = (info.gfx_target_version / 10000) % 100;
            auto minor = (info.gfx_target_version / 100) % 100;
            auto step  = (info.gfx_target_version % 100);
            info.name =
                common::get_string_entry(fmt::format("gfx{}{}{:x}", major, minor, step))->c_str();
        }
        else
        {
            info.name = common::get_string_entry("")->c_str();
        }

        auto product =
            (have_wkmi_info && !wk.product_name.empty()) ? wk.product_name : std::string{"unknown"};
        info.product_name = common::get_string_entry(product)->c_str();
        info.vendor_name  = common::get_string_entry("AMD")->c_str();
        info.model_name   = common::get_string_entry("")->c_str();

        // ---- workgroup / grid maxima (hardcoded in hsa-runtime, gnulinux) ----
        constexpr auto workgrp_max = 1024;
        constexpr auto grid_max    = std::numeric_limits<uint32_t>::max();
        constexpr auto grid_max_x  = std::numeric_limits<int32_t>::max();
        constexpr auto grid_max_y  = std::numeric_limits<uint16_t>::max();
        constexpr auto grid_max_z  = std::numeric_limits<uint16_t>::max();
        info.workgroup_max_size    = workgrp_max;
        info.workgroup_max_dim     = {workgrp_max, workgrp_max, workgrp_max};
        info.grid_max_size         = grid_max;
        info.grid_max_dim          = {grid_max_x, grid_max_y, grid_max_z};

        // ---- uuid ----
        auto _uuid       = uuid_view_t{};
        _uuid.value64[0] = have_wkmi_info ? wk.uuid : 0;
        info.uuid        = static_cast<rocprofiler_uuid_t>(_uuid);

        // ---- follow-up topology (WSL parity: not populated yet) ----
        info.mem_banks_count = 0;
        info.caches_count    = 0;
        info.io_links_count  = 0;
        info.mem_banks       = nullptr;
        info.caches          = nullptr;
        info.io_links        = nullptr;

        update_agent_runtime_visibility(info);

        ROCP_INFO << fmt::format(
            "windows::enumerate: enumerated adapter {} vendor=0x{:04x} device=0x{:04x} "
            "BDF={:02x}:{:02x}.{:x} name='{}' product='{}' vram={} gfx={}",
            i,
            info.vendor_id,
            info.device_id,
            addr.bus_number,
            addr.device_number,
            addr.function_number,
            info.name,
            product,
            info.local_mem_size,
            info.gfx_target_version);

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
#else
bool
is_available()
{
    return false;
}

std::vector<unique_agent_t>
enumerate()
{
    ROCP_WARNING << "windows::enumerate: not available on this platform";
    return {};
}
#endif

}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler
