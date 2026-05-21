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

#pragma once

#include <rocprofiler-sdk/agent.h>

#include <memory>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace topology
{
// Owning handle for a rocprofiler_agent_t allocated by a provider. The custom
// deleter is responsible for freeing the mem_banks/caches/io_links arrays in
// addition to the struct itself.
using unique_agent_t = std::unique_ptr<rocprofiler_agent_t, void (*)(rocprofiler_agent_t*)>;

class TopologyProvider
{
public:
    virtual ~TopologyProvider() = default;

    // Return the freshly enumerated topology. Empty vector = no agents detected
    // (NOT an error - caller decides). Must be safe to call multiple times.
    virtual std::vector<unique_agent_t> enumerate() = 0;

    // Short identifier for logging ("linux-kfd", "wsl-dxcore", "win-d3dkmt").
    virtual std::string_view name() const = 0;
};

// Factory: inspects environment + filesystem and returns the active provider.
// Selection precedence (first match wins):
//   1. ROCPROFILER_FORCE_DXG=1     -> DxgProvider
//   2. ROCPROFILER_FORCE_D3DKMT=1  -> D3dkmtProvider (Windows only)
//   3. /sys/class/kfd/kfd/topology/nodes OR
//      /sys/devices/virtual/kfd/kfd/topology/nodes OR
//      $ROCPROFILER_KFD_TOPOLOGY / $AMD_KFD_TOPOLOGY / $HSA_MODEL_TOPOLOGY
//      exists                      -> LinuxKfdProvider
//   4. libdxcore.so loadable + /dev/dxg exists -> DxgProvider
//   5. compiled for _WIN32         -> D3dkmtProvider
//   6. fallback                    -> LinuxKfdProvider (legacy behaviour: it
//      will log a WARNING and return empty)
std::unique_ptr<TopologyProvider>
make_default_provider();

}  // namespace topology
}  // namespace rocprofiler
