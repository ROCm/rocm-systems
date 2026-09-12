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

#include "lib/rocprofiler-sdk/platform/windows/wkmi_bridge.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"

#include <cstdint>

// ROCPROFILER_WINDOWS_USE_WKMI is defined to 1 by the CMake target when a real
// wkmi (source build or prebuilt .lib) is being linked, and 0 otherwise. The
// two implementations below present the identical surface so agent.cpp needs no
// conditional compilation of its own.

#if defined(ROCPROFILER_WINDOWS_USE_WKMI) && ROCPROFILER_WINDOWS_USE_WKMI

// clang-format off
// Third-party headers. <windows.h> first for the base Windows types, then the
// vendored WDK D3DKMT thunk header (for D3DKMT_HANDLE / NTSTATUS / MAX_PATH /
// HANDLE that wkmi.h references), then wkmi.h itself. Confined to this TU so
// clang-tidy never lints the third-party headers via a project TU.
#include <windows.h>
#include "lib/rocprofiler-sdk/platform/windows/external/d3dkmt/d3dkmthk.h"
// wkmi.h is resolved via the wkmi include dir (.../wkmi/lib/inc) carried by the
// rocprofiler-sdk-wkmi INTERFACE target, mirroring the libhsakmt CMake wiring.
#include <wkmi.h>
// clang-format on

#    include <cstdlib>

namespace rocprofiler
{
namespace platform
{
namespace windows
{
namespace
{
constexpr ::NTSTATUS kNtSuccess = 0;
}  // namespace

bool
wkmi_parse_adapter(uint32_t d3dkmt_handle, wkmi_device_info& out)
{
    auto info = ::Wkmi::DeviceInfo{};

    auto st = ::Wkmi::ParseAdapterInfo(static_cast<::D3DKMT_HANDLE>(d3dkmt_handle), &info);

    // ParseAdapterInfo malloc()s info.adapter_info; release it on every exit
    // path (early return, normal return, AND exceptions thrown by the field
    // mapping below such as std::string{info.product_name}), mirroring
    // libhsakmt WDDMDevice::DestroyDeviceInfo. A scope_destructor guarantees
    // the free runs even when the mapping throws std::bad_alloc.
    auto _free_adapter_info = common::scope_destructor{[&info]() {
        if(info.adapter_info != nullptr)
        {
            ::free(info.adapter_info);
            info.adapter_info = nullptr;
        }
    }};

    if(st != kNtSuccess)
    {
        ROCP_INFO << "wkmi_bridge: Wkmi::ParseAdapterInfo failed";
        return false;
    }

    out.major    = info.major;
    out.minor    = info.minor;
    out.stepping = info.stepping;
    out.is_dgpu  = info.is_dgpu;
    // product_name is a fixed-size char[MAX_PATH] buffer; bound the copy to
    // MAX_PATH to guard against a missing NUL terminator.
    out.product_name = std::string{info.product_name, ::strnlen(info.product_name, MAX_PATH)};
    out.uuid         = info.uuid;
    out.family       = info.family;
    out.device_id    = info.device_id;

    out.wavefront_size                 = info.wavefront_size;
    out.compute_unit_count             = info.compute_unit_count;
    out.wave_per_cu                    = info.wave_per_cu;
    out.simd_per_cu                    = info.simd_per_cu;
    out.max_scratch_slots_per_cu       = info.max_scratch_slots_per_cu;
    out.num_shader_engine              = info.num_shader_engine;
    out.shader_array_per_shader_engine = info.shader_array_per_shader_engine;

    out.max_engine_clock_mhz = info.max_engine_clock_mhz;
    out.max_memory_clock_mhz = info.max_memory_clock_mhz;

    out.pci_bus_addr     = info.pci_bus_addr;
    out.memory_bus_width = info.memory_bus_width;
    out.domain           = info.domain;
    out.num_gws          = info.num_gws;
    out.asic_revision    = info.asic_revision;

    out.local_visible_heap_size   = info.local_visible_heap_size;
    out.local_invisible_heap_size = info.local_invisible_heap_size;
    out.lds_size                  = info.lds_size;

    out.mec_fw_version  = info.mec_fw_version;
    out.sdma_fw_version = info.sdma_fw_version;

    out.num_cp_queues = info.num_cp_queues;
    out.num_xcc       = info.num_xcc;

    return true;
}

bool
wkmi_adapter_supported(uint32_t device_id)
{
    return ::Wkmi::QueryAdapterSupported(static_cast<unsigned int>(device_id));
}

bool
wkmi_is_present()
{
    return true;
}
}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler

#else  // !ROCPROFILER_WINDOWS_USE_WKMI

namespace rocprofiler
{
namespace platform
{
namespace windows
{
bool
wkmi_parse_adapter(uint32_t /*d3dkmt_handle*/, wkmi_device_info& out)
{
    out = wkmi_device_info{};
    ROCP_INFO << "wkmi_bridge: built without wkmi; wkmi_parse_adapter is a no-op";
    return false;
}

bool wkmi_adapter_supported(uint32_t /*device_id*/) { return false; }

bool
wkmi_is_present()
{
    return false;
}
}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler

#endif  // ROCPROFILER_WINDOWS_USE_WKMI
