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
#include <string>
#include <vector>

// Parse a comma-separated list of device indices, e.g. the value of
// ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES. The argument is typically the
// result of std::getenv(); that pointer aliases the process environment and
// must not be written through (undefined behaviour, C11 6.22.4.6 / C++
// [c.strings]). The input is copied before tokenising so the caller's buffer is
// never modified. Comma-delimited tokens are converted with std::atoi; empty
// tokens (leading/trailing/consecutive commas) are skipped. The result is not
// sorted; callers apply their own ordering.
static inline std::vector<int> ParseVisibleDevicesCsv(const char* env) {
    std::vector<int> devices;
    if (env == nullptr) {
        return devices;
    }
    const std::string csv(env);  // copy: never write through the caller's pointer
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = csv.find(',', start);
        const std::size_t end = (comma == std::string::npos) ? csv.size() : comma;
        if (end > start) {  // skip empty tokens
            devices.push_back(std::atoi(csv.substr(start, end - start).c_str()));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return devices;
}
