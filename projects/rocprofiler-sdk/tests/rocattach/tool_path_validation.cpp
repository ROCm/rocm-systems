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

#include <rocprofiler-sdk-rocattach/rocattach.h>

#include <unistd.h>

#include <iostream>
#include <string_view>

int
main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <tool-library-path|--empty>\n";
        return 1;
    }

    auto tool_library_path = std::string_view{argv[1]};
    if(tool_library_path == "--empty") tool_library_path = "";

    setenv("ROCPROF_ATTACH_TOOL_LIBRARY", tool_library_path.data(), 1);
    // Invalid absolute paths should be rejected before rocattach needs a real
    // attach-enabled target thread, keeping this negative test fast.
    auto status = rocattach_attach(getpid());
    if(status != ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT)
    {
        std::cerr << "Expected ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT, got " << status << '\n';
        return 1;
    }

    std::cout << "Test PASSED: invalid absolute tool library path rejected before attach setup\n";
    return 0;
}
