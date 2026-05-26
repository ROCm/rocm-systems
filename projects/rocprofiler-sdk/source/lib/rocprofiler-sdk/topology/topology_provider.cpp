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

#include "lib/rocprofiler-sdk/topology/topology_provider.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/topology/d3dkmt_provider.hpp"
#include "lib/rocprofiler-sdk/topology/dxg_provider.hpp"
#include "lib/rocprofiler-sdk/topology/linux_kfd_provider.hpp"

#include <initializer_list>
#include <string>

namespace rocprofiler
{
namespace topology
{
namespace
{
namespace fs = ::rocprofiler::common::filesystem;

bool
sysfs_topology_present()
{
    for(const std::string& p : std::initializer_list<std::string>{
            common::get_env("ROCPROFILER_KFD_TOPOLOGY", ""),
            common::get_env("AMD_KFD_TOPOLOGY", ""),
            common::get_env("HSA_MODEL_TOPOLOGY", ""),
            std::string{"/sys/devices/virtual/kfd/kfd/topology/nodes"},
            std::string{"/sys/class/kfd/kfd/topology/nodes"}})
    {
        if(!p.empty() && fs::exists(fs::path{p}) && fs::is_directory(fs::path{p})) return true;
    }
    return false;
}
}  // namespace

std::unique_ptr<TopologyProvider>
make_default_provider()
{
    const bool force_dxg    = common::get_env("ROCPROFILER_FORCE_DXG", false);
    const bool force_d3dkmt = common::get_env("ROCPROFILER_FORCE_D3DKMT", false);

    if(force_dxg)
    {
        ROCP_INFO << "TopologyProvider: forced DxgProvider via ROCPROFILER_FORCE_DXG";
        return std::make_unique<DxgProvider>();
    }
    if(force_d3dkmt)
    {
#ifdef _WIN32
        ROCP_INFO << "TopologyProvider: forced D3dkmtProvider via ROCPROFILER_FORCE_D3DKMT";
        return std::make_unique<D3dkmtProvider>();
#else
        ROCP_WARNING << "TopologyProvider: ROCPROFILER_FORCE_D3DKMT ignored on non-Windows build";
#endif
    }
    if(sysfs_topology_present())
    {
        ROCP_INFO << "TopologyProvider: selected LinuxKfdProvider (sysfs present)";
        return std::make_unique<LinuxKfdProvider>();
    }
    if(DxgProvider::is_available())
    {
        ROCP_INFO << "TopologyProvider: selected DxgProvider (libdxcore.so present)";
        return std::make_unique<DxgProvider>();
    }
#ifdef _WIN32
    ROCP_INFO << "TopologyProvider: selected D3dkmtProvider (native Windows)";
    return std::make_unique<D3dkmtProvider>();
#else
    ROCP_WARNING << "TopologyProvider: no provider matched; falling back to LinuxKfdProvider "
                    "(will return empty)";
    return std::make_unique<LinuxKfdProvider>();
#endif
}

}  // namespace topology
}  // namespace rocprofiler
