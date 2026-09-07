/*
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>

// Parse a comma-separated list of device indices, e.g. the value of
// ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES. The argument is typically the
// result of std::getenv(); that pointer aliases the process environment and
// must not be written through (undefined behaviour, C11 6.22.4.6 / C++
// [c.strings]). Comma-delimited tokens are converted with std::atoi; empty
// tokens (leading/trailing/consecutive commas) are skipped. The result is not
// sorted; callers apply their own ordering.
static inline std::vector<int> ParseVisibleDevicesCsv(const char* env) {
    std::vector<int> devices;
    if (env == nullptr) {
        return devices;
    }
    // NOTE: std::strtok writes a NUL over each delimiter in its first argument.
    // When `env` is a std::getenv() return value this modifies the environment
    // in place -- see the accompanying non-mutation test, which fails here.
    for (char* token = std::strtok(const_cast<char*>(env), ","); token != nullptr;
         token = std::strtok(nullptr, ",")) {
        devices.push_back(std::atoi(token));
    }
    return devices;
}
