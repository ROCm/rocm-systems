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

#include <memory>

namespace rocprofiler
{
namespace topology
{
class DxgProvider final : public TopologyProvider
{
public:
    DxgProvider();
    ~DxgProvider() override;

    std::vector<unique_agent_t> enumerate() override;
    std::string_view            name() const override { return "wsl-dxcore"; }

    // True if libdxcore.so can be dlopened and D3DKMTEnumAdapters3 resolved.
    // Cached on first call. Used by the factory for runtime selection.
    static bool is_available();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace topology
}  // namespace rocprofiler
