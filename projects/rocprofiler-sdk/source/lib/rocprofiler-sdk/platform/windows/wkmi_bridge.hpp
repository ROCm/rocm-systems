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

#pragma once

// Narrow bridge over the AMD-private wkmi library. The wkmi headers (and the
// WDK headers they transitively need) are confined to wkmi_bridge.cpp so the
// enumerator in agent.cpp depends only on the plain struct + free functions
// declared here. When wkmi is unavailable (option OFF / submodule absent) a
// stub TU provides the same surface returning false, so agent.cpp compiles and
// degrades to D3DKMT-only basic fields without any conditional compilation of
// its own.

#include <array>
#include <cstdint>
#include <string>

namespace rocprofiler
{
namespace platform
{
namespace windows
{
// Subset of Wkmi::DeviceInfo that the agent enumerator maps onto
// rocprofiler_agent_t. Field names mirror the wkmi DeviceInfo members they are
// copied from (see wkmi_bridge.cpp) so the mapping stays auditable.
struct wkmi_device_info
{
    // Architecture identification
    int32_t     major        = 0;  // GFX IP major
    int32_t     minor        = 0;  // GFX IP minor
    int32_t     stepping     = 0;  // GFX IP stepping
    bool        is_dgpu      = false;
    std::string product_name = {};
    uint64_t    uuid         = 0;
    uint32_t    family       = 0;  // GPU family id
    uint32_t    device_id    = 0;

    // Compute capabilities
    uint32_t wavefront_size                 = 0;
    uint32_t compute_unit_count             = 0;
    uint32_t wave_per_cu                    = 0;  // Maximum waves per compute unit (wkmi.h)
    uint32_t simd_per_cu                    = 0;
    uint32_t max_scratch_slots_per_cu       = 0;
    uint32_t num_shader_engine              = 0;
    uint32_t shader_array_per_shader_engine = 0;

    // Clocks
    uint32_t max_engine_clock_mhz = 0;
    uint32_t max_memory_clock_mhz = 0;

    // Hardware features / addressing
    uint32_t pci_bus_addr     = 0;  // bus << 8 | device << 3 | function
    uint32_t memory_bus_width = 0;
    uint32_t domain           = 0;
    uint32_t num_gws          = 0;
    uint32_t asic_revision    = 0;

    // Memory
    uint64_t local_visible_heap_size   = 0;
    uint64_t local_invisible_heap_size = 0;
    uint32_t lds_size                  = 0;  // bytes

    // Firmware
    uint32_t mec_fw_version  = 0;
    uint32_t sdma_fw_version = 0;

    // Hardware scheduling
    uint32_t num_cp_queues = 0;
    uint32_t num_xcc       = 0;
};

// Maps a D3DKMT adapter handle to a populated wkmi_device_info. Returns false
// (and leaves out unspecified) if wkmi is unavailable or the parse fails.
// Internally calls Wkmi::ParseAdapterInfo and free()s the adapter_info blob it
// mallocs, per the libhsakmt DestroyDeviceInfo contract.
bool
wkmi_parse_adapter(uint32_t d3dkmt_handle, wkmi_device_info& out);

// Whether wkmi recognizes/supports the given PCI device id. Returns false when
// wkmi is unavailable. Gate adapters on this before publishing them.
bool
wkmi_adapter_supported(uint32_t device_id);

// Whether this build links a real wkmi (vs. the stub). Lets the enumerator log
// which code path is active.
bool
wkmi_is_present();
}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler
