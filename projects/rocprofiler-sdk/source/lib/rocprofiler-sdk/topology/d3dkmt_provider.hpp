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

#include "lib/rocprofiler-sdk/topology/topology_provider.hpp"

#include <cstdint>

namespace rocprofiler
{
namespace topology
{
class D3dkmtProvider final : public TopologyProvider
{
public:
    D3dkmtProvider();
    ~D3dkmtProvider() override;

    std::vector<unique_agent_t> enumerate() override;
    std::string_view            name() const override { return "win-d3dkmt"; }

private:
    // Helpers declared here so the eventual implementation has a place to land.
    // All bodies are #ifdef _WIN32 guarded in the .cpp.
    struct AdapterRecord;  // opaque; defined in .cpp under _WIN32

    bool           enumerate_adapters_(std::vector<AdapterRecord>& out);
    bool           query_node_metadata_(AdapterRecord& adapter);
    bool           query_umd_private_(AdapterRecord& adapter);
    bool           query_physical_adapter_(AdapterRecord& adapter);
    unique_agent_t adapter_to_agent_(const AdapterRecord& adapter, uint64_t logical_node_id);
};
}  // namespace topology
}  // namespace rocprofiler
