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

// Host-only table-driven test for ParseVisibleDevicesCsv (no GPU / HIP / VAAPI).
// Build/run:
//   c++ -std=c++17 -I projects/rocjpeg/src \
//       projects/rocjpeg/test/vaapi_parse_visible_devices_test.cpp \
//       -o t && ./t

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rocjpeg_parse_helpers.h"

namespace {

struct Case {
    const char*      description;  // what this row exercises + expected outcome
    const char*      input;        // the CSV string
    std::vector<int> expected;     // expected parsed indices (unsorted)
};

// Parse from a writable buffer so the test is well-defined regardless of whether
// the helper mutates its argument (pre-fix) or copies it (fixed).
std::vector<int> parse(const char* s) {
    std::vector<char> buf(s, s + std::strlen(s) + 1);
    return ParseVisibleDevicesCsv(buf.data());
}

}  // namespace

int main() {
    int failures = 0;

    const Case cases[] = {
        // ---- positive ----
        {"positive: single index '0' -> {0}",              "0",        {0}},
        {"positive: '0,1,2,3' -> {0,1,2,3}",               "0,1,2,3",  {0, 1, 2, 3}},
        {"positive: unsorted '3,1,2' -> {3,1,2} (no sort)", "3,1,2",   {3, 1, 2}},
        {"positive: duplicates kept '5,5,7' -> {5,5,7}",   "5,5,7",    {5, 5, 7}},
        {"positive: multi-digit '10,20,30' -> {10,20,30}", "10,20,30", {10, 20, 30}},
        // ---- boundary ----
        {"boundary: '0' -> {0}",                           "0",        {0}},
        {"boundary: three-digit '100,200' -> {100,200}",   "100,200",  {100, 200}},
        // ---- negative ----
        {"negative: empty string -> {}",                   "",         {}},
        // ---- corner ----
        {"corner: empty tokens '0,,1' -> {0,1}",           "0,,1",     {0, 1}},
        {"corner: leading/trailing commas ',5,' -> {5}",   ",5,",      {5}},
    };

    for (const Case& c : cases) {
        std::vector<int> got = parse(c.input);
        if (got != c.expected) {
            ++failures;
            std::printf("FAIL: %s (got %zu values)\n", c.description, got.size());
        }
    }

    // negative: nullptr -> empty
    if (!ParseVisibleDevicesCsv(nullptr).empty()) {
        ++failures;
        std::printf("FAIL: negative: nullptr -> {}\n");
    }

    // regression (the bug): the parser must NOT modify its input buffer.
    // strtok-on-getenv writes NULs over the delimiters; this row fails pre-fix.
    {
        const char* original = "0,1,2";
        std::vector<char> buf(original, original + std::strlen(original) + 1);
        (void)ParseVisibleDevicesCsv(buf.data());
        if (std::strcmp(buf.data(), original) != 0) {
            ++failures;
            std::printf("FAIL: regression: input buffer was mutated by the parser\n");
        }
    }

    if (failures != 0) {
        std::printf("%d FAILED\n", failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
