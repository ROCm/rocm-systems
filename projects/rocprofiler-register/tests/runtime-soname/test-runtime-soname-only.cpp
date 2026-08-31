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

constexpr auto sdk_soname         = std::string_view{ "librocprofiler-sdk.so.1" };
constexpr auto attach_soname      = std::string_view{ "librocprofiler-sdk-attach.so.1" };
constexpr auto sdk_unversioned    = std::string_view{ "librocprofiler-sdk.so" };
constexpr auto attach_unversioned = std::string_view{ "librocprofiler-sdk-attach.so" };

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
                  << info.dli_fname << " instead of the SONAME-only runtime directory at "
                  << expected_library << '\n';
        return false;
    }

    std::cout << "Verified " << symbol_name << " loaded from " << info.dli_fname << '\n';
    return true;
}

int
run_child(std::string_view mode,
          const fs::path&  runtime_directory,
          const fs::path&  register_library)
{
    if(fs::exists(runtime_directory / sdk_unversioned) ||
       fs::exists(runtime_directory / attach_unversioned))
    {
        std::cerr << "Test FAILED: runtime directory contains an unversioned development "
                     "symlink\n";
        return 1;
    }

    if(!verify_loaded_library("rocprofiler_register_library_api_table", register_library))
        return 1;

    hsa_init();

    if(mode == "sdk")
    {
        if(!verify_loaded_library("rocprofiler_set_api_table",
                                  runtime_directory / sdk_soname))
            return 1;
    }
    else if(mode == "attach")
    {
        if(!verify_loaded_library("rocprofiler_attach_set_api_table",
                                  runtime_directory / attach_soname))
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
                      const fs::path& register_library)
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
    auto preload      = register_library.string();
    if(auto* current = std::getenv("LD_PRELOAD"); current != nullptr && *current != '\0')
        preload = std::string{ current } + ":" + preload;

    return status == 0 && setenv("LD_LIBRARY_PATH", library_path.c_str(), 1) == 0 &&
           setenv("LD_PRELOAD", preload.c_str(), 1) == 0 &&
           setenv("ROCPROFILER_REGISTER_LOG_LEVEL", "info", 1) == 0;
}

bool
execute_child(const fs::path& executable,
              const char*     mode,
              const fs::path& runtime_directory,
              const fs::path& register_library)
{
    auto child_pid = fork();
    if(child_pid < 0)
    {
        std::cerr << "Test FAILED: fork failed: " << std::strerror(errno) << '\n';
        return false;
    }

    if(child_pid == 0)
    {
        if(!set_child_environment(mode, runtime_directory, register_library))
        {
            std::fprintf(stderr, "Test FAILED: could not configure child environment\n");
            _exit(127);
        }

        execl(executable.c_str(),
              executable.c_str(),
              "--child",
              mode,
              runtime_directory.c_str(),
              register_library.c_str(),
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

    if(!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        std::cerr << "Test FAILED: " << mode << " child did not exit successfully";
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

    if(argc != 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <register-library> <register-soname> <sdk-library> <sdk-soname> "
                     "<attach-library> <attach-soname>\n";
        return 1;
    }

    if(std::string_view{ argv[4] } != sdk_soname ||
       std::string_view{ argv[6] } != attach_soname)
    {
        std::cerr << "Test FAILED: fixtures do not use the expected SDK ABI 1 SONAMEs\n";
        return 1;
    }

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

    auto register_library = runtime_directory / argv[2];
    if(!copy_library(argv[1], register_library) ||
       !copy_library(argv[3], runtime_directory / argv[4]) ||
       !copy_library(argv[5], runtime_directory / argv[6]))
        return 1;

    auto executable = fs::canonical("/proc/self/exe", ec);
    if(ec)
    {
        std::cerr << "Test FAILED: could not resolve test executable: " << ec.message()
                  << '\n';
        return 1;
    }

    if(!execute_child(executable, "sdk", runtime_directory, register_library) ||
       !execute_child(executable, "attach", runtime_directory, register_library))
        return 1;

    std::cout << "Test PASSED: SDK and attach libraries loaded from SONAME-only runtime "
                 "directory\n";
    return 0;
}
