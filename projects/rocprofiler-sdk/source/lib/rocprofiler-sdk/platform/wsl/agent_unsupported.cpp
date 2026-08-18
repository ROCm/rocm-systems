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

// platform::wsl for builds whose hsakmt headers cannot describe a DXG KMT node.
//
// agent.cpp and dxg_topology.cpp read the node's adapter LUID, negotiate the
// record layout through HsaStructureSizes and honour the thunk's gfx version
// override. Older hsakmt headers declare none of these, and every ROCm shipping
// such headers ships no librocdxg either, so on those releases there is no
// topology to read and no thunk to read it through. This file is compiled in
// their place (see the CMakeLists in this directory) so that platform::wsl
// still exists as an enumerator and answers the one question agent.cpp asks of
// it.
//
// Reporting unavailable is the same answer the real enumerator would give on
// such a ROCm: is_available() there also fails once nothing provides the thunk.
// select_platform() therefore never chooses this platform, and a caller that
// forces it with ROCPROFILER_FORCE_PLATFORM=wsl gets an empty agent list rather
// than a silently wrong one.

#include "lib/rocprofiler-sdk/platform/wsl/agent.hpp"

#include "lib/common/logging.hpp"

#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
bool
is_available()
{
    ROCP_INFO << "agent topology: " << name
              << " is not built into this binary; profiling on WSL requires rocprofiler-sdk "
                 "to be built against hsakmt headers that describe a DXG KMT node";
    return false;
}

std::vector<unique_agent_t>
enumerate()
{
    return {};
}
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
