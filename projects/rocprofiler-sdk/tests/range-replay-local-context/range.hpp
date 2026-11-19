// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <cstdint>

// The contract between main.cpp, which places the range, and client.cpp, which counts what each
// service delivered for it.

constexpr uint64_t kRangeId = 0x5E4C1C;

// Dispatches main.cpp submits inside the range.
constexpr uint64_t kRangeDispatches = 3;

// Name of main.cpp's kernel. The tool finds its kernel ids by this name as code objects load,
// because the thread trace dispatch callback reports only a kernel id, and range replay's CONFIG
// callback, unlike kernel replay's, carries no dispatch to learn one from.
constexpr const char* kKernelName = "rr_local_context_step";

// acc = acc*3 + add for add = 1, 2, 3 from a zeroed buffer: 0 -> 1 -> 5 -> 18.
constexpr int kExpectedResult = 18;
