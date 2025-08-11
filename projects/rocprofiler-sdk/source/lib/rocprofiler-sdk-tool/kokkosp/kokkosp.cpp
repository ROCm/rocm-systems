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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/environment.hpp"

#include <rocprofiler-sdk-roctx/roctx.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" {
struct Kokkos_Tools_ToolSettings
{
    bool requires_global_fencing;
    bool padding[255];
};

struct Kokkos_Profiling_KokkosPDeviceInfo
{
    size_t deviceID;
};
}

namespace
{
struct Section
{
    Section()                   = default;
    ~Section()                  = default;
    Section(const Section&)     = default;
    Section(Section&&) noexcept = default;
    Section& operator=(const Section&) = default;
    Section& operator=(Section&&) noexcept = default;

    std::string      label = {};
    roctx_range_id_t id    = std::numeric_limits<roctx_range_id_t>::max();
};

bool tool_globfences  = false;
auto kokkosp_sections = std::vector<Section>{};
}  // namespace

extern "C" {
void
kokkosp_request_tool_settings(const uint32_t, Kokkos_Tools_ToolSettings* settings) ROCTX_PUBLIC_API;

void
kokkosp_init_library(const int /*loadSeq*/,
                     const uint64_t /*interfaceVer*/,
                     const uint32_t /*devInfoCount*/,
                     Kokkos_Profiling_KokkosPDeviceInfo* /*deviceInfo*/) ROCTX_PUBLIC_API;

void
kokkosp_finalize_library() ROCTX_PUBLIC_API;

void
kokkosp_begin_parallel_for(const char* /*name*/,
                           const uint32_t /*devID*/,
                           uint64_t* /*kID*/) ROCTX_PUBLIC_API;
void
kokkosp_end_parallel_for(const uint64_t /*kID*/) ROCTX_PUBLIC_API;

void
kokkosp_begin_parallel_scan(const char* /*name*/,
                            const uint32_t /*devID*/,
                            uint64_t* /*kID*/) ROCTX_PUBLIC_API;
void
kokkosp_end_parallel_scan(const uint64_t /*kID*/) ROCTX_PUBLIC_API;

void
kokkosp_begin_parallel_reduce(const char* /*name*/,
                              const uint32_t /*devID*/,
                              uint64_t* /*kID*/) ROCTX_PUBLIC_API;

void
kokkosp_end_parallel_reduce(const uint64_t /*kID*/) ROCTX_PUBLIC_API;

void
kokkosp_push_profile_region(const char* /*name*/) ROCTX_PUBLIC_API;

void
kokkosp_pop_profile_region() ROCTX_PUBLIC_API;

void
kokkosp_create_profile_section(const char* /*name*/, uint32_t* /*secid*/) ROCTX_PUBLIC_API;

void
kokkosp_start_profile_section(const uint32_t /*secid*/) ROCTX_PUBLIC_API;

void
kokkosp_stop_profile_section(const uint32_t /*secid*/) ROCTX_PUBLIC_API;

void
kokkosp_destroy_profile_section(const uint32_t /*secid*/) ROCTX_PUBLIC_API;

void
kokkosp_profile_event(const char* /*name*/) ROCTX_PUBLIC_API;

void
kokkosp_begin_fence(const char* /*name*/,
                    const uint32_t /*devID*/,
                    uint64_t* /*fID*/) ROCTX_PUBLIC_API;

void
kokkosp_end_fence(const uint64_t /*fID*/) ROCTX_PUBLIC_API;
}

//
//
//          IMPLEMENTATION
//
//
extern "C" {
void
kokkosp_request_tool_settings(const uint32_t, Kokkos_Tools_ToolSettings* settings)
{
    if(tool_globfences)
    {
        settings->requires_global_fencing = true;
    }
    else
    {
        settings->requires_global_fencing = false;
    }
}

void
kokkosp_init_library(const int      loadSeq,
                     const uint64_t interfaceVer,
                     const uint32_t /*devInfoCount*/,
                     Kokkos_Profiling_KokkosPDeviceInfo* /*deviceInfo*/)
{
    tool_globfences = ::rocprofiler::common::get_env("KOKKOS_TOOLS_GLOBALFENCES", false);

    std::cout << "-----------------------------------------------------------\n"
              << "KokkosP: rocprofv3 Connector (sequence is " << loadSeq
              << ", version: " << interfaceVer << ")\n"
              << "-----------------------------------------------------------\n";

    roctxMark("Kokkos::Initialization Complete");
}

void
kokkosp_finalize_library()
{
    std::cout << R"(
-----------------------------------------------------------
KokkosP: Finalization of rocprofv3 Connector. Complete.
-----------------------------------------------------------
)";

    roctxMark("Kokkos::Finalization Complete");
}

void
kokkosp_begin_parallel_for(const char* name, const uint32_t /*devID*/, uint64_t* /*kID*/)
{
    roctxRangePush(name);
}

void
kokkosp_end_parallel_for(const uint64_t /*kID*/)
{
    roctxRangePop();
}

void
kokkosp_begin_parallel_scan(const char* name, const uint32_t /*devID*/, uint64_t* /*kID*/)
{
    roctxRangePush(name);
}

void
kokkosp_end_parallel_scan(const uint64_t /*kID*/)
{
    roctxRangePop();
}

void
kokkosp_begin_parallel_reduce(const char* name, const uint32_t /*devID*/, uint64_t* /*kID*/)
{
    roctxRangePush(name);
}

void
kokkosp_end_parallel_reduce(const uint64_t /*kID*/)
{
    roctxRangePop();
}

void
kokkosp_push_profile_region(const char* name)
{
    roctxRangePush(name);
}

void
kokkosp_pop_profile_region()
{
    roctxRangePop();
}

void
kokkosp_create_profile_section(const char* name, uint32_t* secid)
{
    *secid = kokkosp_sections.size();
    kokkosp_sections.emplace_back(
        Section{std::string{name}, std::numeric_limits<roctx_range_id_t>::max()});
}

void
kokkosp_start_profile_section(const uint32_t secid)
{
    auto& section = kokkosp_sections[secid];
    section.id    = roctxRangeStart(section.label.c_str());
}

void
kokkosp_stop_profile_section(const uint32_t secid)
{
    auto const& section = kokkosp_sections[secid];
    roctxRangeStop(section.id);
}

void
kokkosp_destroy_profile_section(const uint32_t)
{
    // do nothing
}

void
kokkosp_profile_event(const char* name)
{
    roctxMark(name);
}

void
kokkosp_begin_fence(const char* name, const uint32_t /*devID*/, uint64_t* fID)
{
    *fID = roctxRangeStart(name);
}

void
kokkosp_end_fence(const uint64_t fID)
{
    roctxRangeStop(fID);
}
}
