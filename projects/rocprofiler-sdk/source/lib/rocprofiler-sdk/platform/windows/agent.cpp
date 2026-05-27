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

#include "lib/rocprofiler-sdk/platform/windows/agent.hpp"

#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace platform
{
namespace windows
{
#ifdef _WIN32
// TODO(win-d3dkmt): include <d3dkmthk.h> / <ntddvdeo.h> equivalents from the
// WDK / Windows SDK. The POC at bgopesh/win-agent-info-poc demonstrates the
// exact link target and which headers are used. Real implementation will
// mirror platform/wsl/agent.cpp against the WDK D3DKMT entry points.

bool
is_available()
{
    return true;
}

std::vector<unique_agent_t>
enumerate()
{
    ROCP_WARNING << "windows::enumerate: native Windows path not yet implemented (skeleton only)";
    return {};
}
#else
bool
is_available()
{
    return false;
}

std::vector<unique_agent_t>
enumerate()
{
    ROCP_WARNING << "windows::enumerate: not available on this platform";
    return {};
}
#endif

}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler
