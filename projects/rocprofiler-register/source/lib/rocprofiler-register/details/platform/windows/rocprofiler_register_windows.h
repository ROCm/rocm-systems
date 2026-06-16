// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

// WINDOWS-DIVERGENCE: Central include for the Win32 SDK headers used by
// rocprofiler-register's Windows platform layer. Include this header instead of
// spelling out the WIN32_LEAN_AND_MEAN / NOMINMAX / NOGDI guards everywhere.
//
// The three guard macros strip parts of windows.h that conflict with the C++
// standard library or rocprofiler's own definitions:
//   WIN32_LEAN_AND_MEAN  — omits Winsock 1, OLE, COM, RPC, and other subsystems
//                          that are not used by rocprofiler-register and that
//                          pull in incompatible typedefs (e.g. SOCKET, BOOL).
//   NOMINMAX             — prevents windows.h from #defining min/max macros that
//                          shadow std::min / std::max (C4003 / shadowing errors).
//   NOGDI                — omits the GDI drawing API; it defines ERROR as 0,
//                          which collides with rocprofiler's ERROR enum values.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef NOGDI
#    define NOGDI
#endif
#include <windows.h>
