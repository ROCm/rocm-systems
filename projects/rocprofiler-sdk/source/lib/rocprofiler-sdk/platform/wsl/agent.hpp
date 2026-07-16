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

#include "lib/rocprofiler-sdk/platform/agent.hpp"

#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
// True if running inside WSL with a usable DXCore driver: /dev/dxg present and
// libdxcore.so dlopens with D3DKMTEnumAdapters3 resolvable. Result is cached
// on first call.
bool
is_available();

// Enumerates AMD adapters via libdxcore.so (D3DKMTEnumAdapters3 +
// D3DKMTQueryAdapterInfo). Returns empty vector if WSL/DXCore is not present
// or no AMD adapter was found.
std::vector<unique_agent_t>
enumerate();

constexpr std::string_view name = "wsl-dxcore";
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
