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

#include <rocprofiler-sdk/registration.h>

#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

namespace
{
namespace fs = std::filesystem;

struct remove_path
{
    fs::path value;

    ~remove_path()
    {
        auto ec = std::error_code{};
        fs::remove(value, ec);
    }
};

rocprofiler_tool_configure_result_t*
empty_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    return nullptr;
}

int
run_registration_test(const fs::path& alias_path)
{
    if(setenv("ROCPROFILER_REGISTER_LIBRARY", alias_path.c_str(), 1) != 0)
    {
        std::cerr << "Test FAILED: could not set ROCPROFILER_REGISTER_LIBRARY\n";
        return 1;
    }

    auto status = rocprofiler_force_configure(&empty_configure);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Test FAILED: rocprofiler_force_configure returned " << status << '\n';
        return 1;
    }

    auto initialized = 0;
    status           = rocprofiler_is_initialized(&initialized);
    if(status != ROCPROFILER_STATUS_SUCCESS || initialized != 1)
    {
        std::cerr << "Test FAILED: rocprofiler-sdk did not initialize\n";
        return 1;
    }

    std::cout << "Test PASSED: equivalent ROCPROFILER_REGISTER_LIBRARY alias accepted\n";
    return 0;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <rocprofiler-sdk-library-path>\n";
        return 1;
    }

    auto ec           = std::error_code{};
    auto library_path = fs::canonical(argv[1], ec);
    if(ec)
    {
        std::cerr << "Test FAILED: could not resolve rocprofiler-sdk library path: " << ec.message()
                  << '\n';
        return 1;
    }

    auto alias_path =
        library_path.parent_path() / ("rocattach-equivalent-" + library_path.filename().string() +
                                      "-" + std::to_string(getpid()));

    fs::remove(alias_path, ec);
    ec.clear();
    fs::create_hard_link(library_path, alias_path, ec);
    if(ec)
    {
        std::cerr << "Test FAILED: could not create hard-link library alias beside " << library_path
                  << ": " << ec.message() << '\n';
        return 1;
    }
    auto cleanup = remove_path{alias_path};

    if(!fs::equivalent(library_path, alias_path, ec) || ec)
    {
        std::cerr << "Test FAILED: library and hard-link alias are not equivalent\n";
        return 1;
    }

    auto child_pid = fork();
    if(child_pid < 0)
    {
        std::cerr << "Test FAILED: could not fork registration subprocess\n";
        return 1;
    }
    if(child_pid == 0) return run_registration_test(alias_path);

    auto child_status = 0;
    if(waitpid(child_pid, &child_status, 0) != child_pid)
    {
        std::cerr << "Test FAILED: could not wait for registration subprocess\n";
        return 1;
    }

    if(!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)
    {
        std::cerr << "Test FAILED: registration subprocess did not exit successfully\n";
        return 1;
    }

    return 0;
}
