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

#include "lib/rocprofiler-sdk/topology/d3dkmt_provider.hpp"

#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace topology
{
#ifdef _WIN32
// TODO(win-d3dkmt): include <d3dkmthk.h> / <ntddvdeo.h> equivalents from
// the WDK / Windows SDK. The POC at bgopesh/win-agent-info-poc demonstrates
// the exact link target and which headers are used.

struct D3dkmtProvider::AdapterRecord
{
    // Filled by enumerate_adapters_ from D3DKMT_ADAPTERINFO:
    //   HANDLE   hAdapter;
    //   LUID     AdapterLuid;
    //   uint32_t NumOfSources;
    // Filled by query_*:
    //   uint32_t vendor_id, device_id, sub_vendor_id, sub_device_id;
    //   uint32_t pci_segment, bus, device, function;
    //   uint32_t num_engines_3d, num_engines_compute, num_engines_copy;
    //   uint64_t dedicated_video_memory;
    //   uint32_t gfx_target_version, num_xcc, simd_count, cu_count;
    //   wchar_t  driver_description[256];
};

bool
D3dkmtProvider::enumerate_adapters_(std::vector<AdapterRecord>&)
{
    // TODO: D3DKMTEnumAdapters3 (preferred; fall back to D3DKMTEnumAdapters2).
    return false;
}

bool
D3dkmtProvider::query_node_metadata_(AdapterRecord&)
{
    // TODO: loop ordinal 0..NumOfSources-1; D3DKMTQueryAdapterInfo with
    //       KMTQAITYPE_NODEMETADATA. Classify engines by EngineType.
    return false;
}

bool
D3dkmtProvider::query_umd_private_(AdapterRecord&)
{
    // TODO: D3DKMTQueryAdapterInfo with KMTQAITYPE_UMDRIVERPRIVATE; cast the
    //       returned blob to the AMD UMD struct (sourced from rocm-libs/PAL).
    return false;
}

bool
D3dkmtProvider::query_physical_adapter_(AdapterRecord&)
{
    // TODO: KMTQAITYPE_PHYSICALADAPTERCOUNT + KMTQAITYPE_PHYSICALADAPTERDEVICEIDS
    //       for vendor/device IDs and PCI BDF.
    return false;
}

unique_agent_t
D3dkmtProvider::adapter_to_agent_(const AdapterRecord&, uint64_t)
{
    return {nullptr, [](rocprofiler_agent_t*) {}};
}

D3dkmtProvider::D3dkmtProvider()  = default;
D3dkmtProvider::~D3dkmtProvider() = default;

std::vector<unique_agent_t>
D3dkmtProvider::enumerate()
{
    ROCP_WARNING << "D3dkmtProvider::enumerate(): native Windows path not yet "
                    "implemented (skeleton only)";
    return {};
}
#else
// Non-Windows: every method is a no-op. Class exists so the factory and
// CMake plumbing compile cleanly on Linux.
struct D3dkmtProvider::AdapterRecord
{};

D3dkmtProvider::D3dkmtProvider()  = default;
D3dkmtProvider::~D3dkmtProvider() = default;

std::vector<unique_agent_t>
D3dkmtProvider::enumerate()
{
    ROCP_WARNING << "D3dkmtProvider::enumerate(): not available on this platform";
    return {};
}

// Stubs so the header's method declarations have linkage when referenced
// (they won't be -- but defining them keeps -Wmissing-declarations quiet if
// the header is included).
bool
D3dkmtProvider::enumerate_adapters_(std::vector<AdapterRecord>&)
{
    return false;
}

bool
D3dkmtProvider::query_node_metadata_(AdapterRecord&)
{
    return false;
}

bool
D3dkmtProvider::query_umd_private_(AdapterRecord&)
{
    return false;
}

bool
D3dkmtProvider::query_physical_adapter_(AdapterRecord&)
{
    return false;
}

unique_agent_t
D3dkmtProvider::adapter_to_agent_(const AdapterRecord&, uint64_t)
{
    return {nullptr, [](rocprofiler_agent_t*) {}};
}
#endif

}  // namespace topology
}  // namespace rocprofiler
