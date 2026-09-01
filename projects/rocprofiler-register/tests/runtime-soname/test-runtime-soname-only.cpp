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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hsa-runtime.hpp"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace
{
namespace fs = std::filesystem;

struct remove_directory
{
    fs::path value;

    ~remove_directory()
    {
        auto ec = std::error_code{};
        fs::remove_all(value, ec);
    }
};

bool
copy_library(const fs::path& source, const fs::path& destination)
{
    auto ec = std::error_code{};
    if(!fs::is_regular_file(source, ec) || ec)
    {
        std::cerr << "Test FAILED: fixture library does not exist: " << source << '\n';
        return false;
    }

    fs::copy_file(source, destination, fs::copy_options::overwrite_existing, ec);
    if(ec)
    {
        std::cerr << "Test FAILED: could not copy " << source << " to " << destination
                  << ": " << ec.message() << '\n';
        return false;
    }
    return true;
}

bool
verify_loaded_library(const char* symbol_name, const fs::path& expected_library)
{
    dlerror();
    auto* symbol = dlsym(RTLD_DEFAULT, symbol_name);
    if(auto* error = dlerror(); error != nullptr || symbol == nullptr)
    {
        std::cerr << "Test FAILED: could not find " << symbol_name << ": "
                  << ((error != nullptr) ? error : "symbol address was null") << '\n';
        return false;
    }

    auto info = Dl_info{};
    if(dladdr(symbol, &info) == 0 || info.dli_fname == nullptr)
    {
        std::cerr << "Test FAILED: could not identify the library providing "
                  << symbol_name << '\n';
        return false;
    }

    auto ec = std::error_code{};
    if(!fs::equivalent(fs::path{ info.dli_fname }, expected_library, ec) || ec)
    {
        std::cerr << "Test FAILED: " << symbol_name << " was loaded from "
                  << info.dli_fname << " instead of the expected runtime library at "
                  << expected_library << '\n';
        return false;
    }

    std::cout << "Verified " << symbol_name << " loaded from " << info.dli_fname << '\n';
    return true;
}

int
run_child(std::string_view mode,
          const fs::path&  register_library,
          const fs::path&  expected_library)
{
    if(!verify_loaded_library("rocprofiler_register_library_api_table", register_library))
        return 1;

    hsa_init();

    if(mode == "sdk")
    {
        if(!verify_loaded_library("rocprofiler_set_api_table", expected_library))
            return 1;
    }
    else if(mode == "attach")
    {
        if(!verify_loaded_library("rocprofiler_attach_set_api_table", expected_library))
            return 1;
    }
    else
    {
        std::cerr << "Test FAILED: unknown child mode " << mode << '\n';
        return 1;
    }

    return 0;
}

bool
set_child_environment(const char*     mode,
                      const fs::path& runtime_directory,
                      const fs::path& register_library,
                      const fs::path& additional_library_directory,
                      const fs::path& register_library_override)
{
    unsetenv("ROCP_TOOL_ATTACH");
    unsetenv("ROCP_TOOL_LIBRARIES");
    unsetenv("ROCPROFILER_REGISTER_FORCE_LOAD");
    unsetenv("ROCPROFILER_REGISTER_LIBRARY");

    auto status = 0;
    if(std::string_view{ mode } == "sdk")
        status = setenv("ROCPROFILER_REGISTER_FORCE_LOAD", "1", 1);
    else if(std::string_view{ mode } == "attach")
        status = setenv("ROCP_TOOL_ATTACH", "1", 1);
    else
        return false;

    auto library_path = runtime_directory.string();
    if(!additional_library_directory.empty())
        library_path =
            additional_library_directory.string() + ":" + std::move(library_path);

    auto preload = register_library.string();
    if(auto* current = std::getenv("LD_PRELOAD"); current != nullptr && *current != '\0')
        preload = std::string{ current } + ":" + preload;

    if(status != 0 || setenv("LD_LIBRARY_PATH", library_path.c_str(), 1) != 0 ||
       setenv("LD_PRELOAD", preload.c_str(), 1) != 0 ||
       setenv("ROCPROFILER_REGISTER_LOG_LEVEL", "info", 1) != 0)
        return false;

    return register_library_override.empty() ||
           setenv("ROCPROFILER_REGISTER_LIBRARY", register_library_override.c_str(), 1) ==
               0;
}

bool
execute_child(const fs::path& executable,
              const char*     mode,
              const fs::path& runtime_directory,
              const fs::path& register_library,
              const fs::path& expected_library,
              const fs::path& additional_library_directory = {},
              const fs::path& register_library_override    = {},
              bool            expect_success               = true)
{
    auto child_pid = fork();
    if(child_pid < 0)
    {
        std::cerr << "Test FAILED: fork failed: " << std::strerror(errno) << '\n';
        return false;
    }

    if(child_pid == 0)
    {
        if(!set_child_environment(mode,
                                  runtime_directory,
                                  register_library,
                                  additional_library_directory,
                                  register_library_override))
        {
            std::fprintf(stderr, "Test FAILED: could not configure child environment\n");
            _exit(127);
        }

        execl(executable.c_str(),
              executable.c_str(),
              "--child",
              mode,
              register_library.c_str(),
              expected_library.c_str(),
              nullptr);
        std::fprintf(stderr, "Test FAILED: exec failed: %s\n", std::strerror(errno));
        _exit(127);
    }

    auto status = 0;
    if(waitpid(child_pid, &status, 0) != child_pid)
    {
        std::cerr << "Test FAILED: waitpid failed: " << std::strerror(errno) << '\n';
        return false;
    }

    if(expect_success && (!WIFEXITED(status) || WEXITSTATUS(status) != 0))
    {
        std::cerr << "Test FAILED: " << mode << " child did not exit successfully";
        if(WIFSIGNALED(status)) std::cerr << " (signal " << WTERMSIG(status) << ')';
        std::cerr << '\n';
        return false;
    }
    if(!expect_success && (!WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT))
    {
        std::cerr << "Test FAILED: " << mode
                  << " child did not abort after the invalid explicit override";
        if(WIFEXITED(status)) std::cerr << " (exit code " << WEXITSTATUS(status) << ')';
        if(WIFSIGNALED(status)) std::cerr << " (signal " << WTERMSIG(status) << ')';
        std::cerr << '\n';
        return false;
    }
    return true;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc == 5 && std::string_view{ argv[1] } == "--child")
        return run_child(argv[2], argv[3], argv[4]);

    if(argc != 12)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <layout> <register-library> <register-soname> <sdk-v1-library> "
                     "<sdk-v1-soname> <sdk-unversioned> <sdk-v0-library> "
                     "<sdk-v0-soname> <attach-library> <attach-soname> "
                     "<attach-unversioned>\n";
        return 1;
    }

    auto layout = std::string_view{ argv[1] };

    auto ec = std::error_code{};
    auto runtime_directory_template =
        (fs::temp_directory_path(ec) / "rocprofiler-register-runtime-soname-XXXXXX")
            .string();
    if(ec)
    {
        std::cerr << "Test FAILED: could not locate temporary directory: " << ec.message()
                  << '\n';
        return 1;
    }

    auto* runtime_directory_value = mkdtemp(runtime_directory_template.data());
    if(runtime_directory_value == nullptr)
    {
        std::cerr << "Test FAILED: could not create temporary directory: "
                  << std::strerror(errno) << '\n';
        return 1;
    }
    auto runtime_directory = fs::path{ runtime_directory_value };
    auto cleanup           = remove_directory{ runtime_directory };

    auto register_library = runtime_directory / argv[3];
    if(!copy_library(argv[2], register_library)) return 1;

    auto sdk_expected                 = fs::path{};
    auto attach_expected              = fs::path{};
    auto run_sdk                      = false;
    auto run_attach                   = false;
    auto additional_library_directory = fs::path{};
    auto register_library_override    = fs::path{};
    auto expect_sdk_success           = true;

    if(layout == "soname-v1")
    {
        sdk_expected    = runtime_directory / argv[5];
        attach_expected = runtime_directory / argv[10];
        run_sdk         = true;
        run_attach      = true;
        if(!copy_library(argv[4], sdk_expected) ||
           !copy_library(argv[9], attach_expected))
            return 1;
    }
    else if(layout == "unversioned")
    {
        sdk_expected    = runtime_directory / argv[6];
        attach_expected = runtime_directory / argv[11];
        run_sdk         = true;
        run_attach      = true;
        if(!copy_library(argv[4], sdk_expected) ||
           !copy_library(argv[9], attach_expected))
            return 1;
    }
    else if(layout == "sdk-soversion-zero")
    {
        sdk_expected = runtime_directory / argv[8];
        run_sdk      = true;
        if(!copy_library(argv[7], sdk_expected)) return 1;
    }
    else if(layout == "sdk-soversion-precedence")
    {
        sdk_expected = runtime_directory / argv[5];
        run_sdk      = true;
        if(!copy_library(argv[7], runtime_directory / argv[8]) ||
           !copy_library(argv[4], sdk_expected))
            return 1;
    }
    else if(layout == "sdk-entrypoint-fallback")
    {
        sdk_expected = runtime_directory / argv[5];
        run_sdk      = true;
        if(!copy_library(argv[9], runtime_directory / argv[6]) ||
           !copy_library(argv[4], sdk_expected))
            return 1;
    }
    else if(layout == "sdk-colocated-precedence")
    {
        sdk_expected                 = runtime_directory / argv[5];
        additional_library_directory = runtime_directory / "global";
        run_sdk                      = true;

        fs::create_directory(additional_library_directory, ec);
        if(ec)
        {
            std::cerr << "Test FAILED: could not create conflicting library directory: "
                      << ec.message() << '\n';
            return 1;
        }

        if(!copy_library(argv[7], additional_library_directory / argv[6]) ||
           !copy_library(argv[4], sdk_expected))
            return 1;
    }
    else if(layout == "sdk-explicit-override-no-fallback")
    {
        sdk_expected              = runtime_directory / argv[5];
        register_library_override = runtime_directory / "invalid-sdk-library";
        expect_sdk_success        = false;
        run_sdk                   = true;
        if(!copy_library(argv[9], register_library_override) ||
           !copy_library(argv[4], sdk_expected))
            return 1;
    }
    else
    {
        std::cerr << "Test FAILED: unknown layout " << layout << '\n';
        return 1;
    }

    auto executable = fs::canonical("/proc/self/exe", ec);
    if(ec)
    {
        std::cerr << "Test FAILED: could not resolve test executable: " << ec.message()
                  << '\n';
        return 1;
    }

    if(run_sdk && !execute_child(executable,
                                 "sdk",
                                 runtime_directory,
                                 register_library,
                                 sdk_expected,
                                 additional_library_directory,
                                 register_library_override,
                                 expect_sdk_success))
        return 1;
    if(run_attach &&
       !execute_child(
           executable, "attach", runtime_directory, register_library, attach_expected))
        return 1;

    std::cout << "Test PASSED: " << layout << '\n';
    return 0;
}
