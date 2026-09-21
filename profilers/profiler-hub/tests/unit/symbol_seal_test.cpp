// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Catch regressions where bundled sqlite3_* symbols leak out of libprofiler-hub.so.
//
// SQLite is compiled with -fvisibility=hidden and the shared library uses
// --exclude-libs on ELF so we do not step on another sqlite3 in the same process
// (TheRock has one). We parse nm/dumpbin output on the built artifact, not at
// runtime. PROFILER_HUB_SHARED_LIB comes from CMake. Skip if the host has no
// suitable tool.

#include <gtest/gtest.h>

#if !defined(_WIN32)
#    include <sys/wait.h>
#endif

#include <array>
#include <cstdio>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace
{

// Set by CMake (absolute path to libprofiler-hub.so).
constexpr const char* k_shared_library = PROFILER_HUB_SHARED_LIB;

#if defined(_WIN32)
#    define PHUB_POPEN  _popen
#    define PHUB_PCLOSE _pclose
#else
#    define PHUB_POPEN  popen
#    define PHUB_PCLOSE pclose
#endif

struct file_op
{
    void operator()(FILE* pipe) const noexcept
    {
        if(pipe != nullptr)
        {
            PHUB_PCLOSE(pipe);
        }
    }
};

// Run command, collect stdout. Returns the child exit status via `status`
// (-1 if the shell could not run it). False only when the pipe could not open.
bool
run_capture(const std::string& command, std::string& output, int& status)
{
    output.clear();
    status = -1;
    std::unique_ptr<FILE, file_op> pipe(PHUB_POPEN(command.c_str(), "r"));
    if(!pipe)
    {
        return false;
    }

    std::array<char, 4096> buffer{};
    while(std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) !=
          nullptr)
    {
        output += buffer.data();
    }

    status = PHUB_PCLOSE(pipe.release());
    return true;
}

// Convenience overload: success means the command ran and exited zero.
bool
run_capture(const std::string& command, std::string& output)
{
    int status = -1;
    if(!run_capture(command, output, status))
    {
        return false;
    }
#if defined(_WIN32)
    // _pclose gives the child exit code directly (-1 on error).
    return status == 0;
#else
    return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

struct symbol_holder
{
    std::string command;
    std::regex  pattern;
};

// A probe only needs to prove the tool exists and runs. Some tools (notably
// dumpbin /?) print their banner to stdout but exit non-zero, so presence is
// decided by captured output, not by the exit status.
bool
tool_available(const std::string& probe)
{
    std::string out;
    int         status = -1;
    return run_capture(probe, out, status) && !out.empty();
}

// First nm/llvm-nm/dumpbin that works on this platform, or false.
bool
pick_tool(const std::string& library, symbol_holder& out)
{
    const std::string quoted = "\"" + library + "\"";

#if defined(_WIN32)
    // dumpbin export table: ordinal, hint, RVA, name.
    if(tool_available("dumpbin /? 2>NUL"))
    {
        out = {
            "dumpbin /nologo /exports " + quoted + " 2>NUL",
            std::regex{
                R"(^\s*[0-9]+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(sqlite3_[A-Za-z0-9_]+))" }
        };
        return true;
    }
    if(tool_available("llvm-nm --version 2>NUL"))
    {
        out = { "llvm-nm --defined-only " + quoted + " 2>NUL",
                std::regex{ R"(\b[A-Za-z] (sqlite3_[A-Za-z0-9_]+)$)" } };
        return true;
    }
#else
    for(const char* candidate : { "nm", "llvm-nm" })
    {
        if(tool_available(std::string{ candidate } + " --version 2>/dev/null"))
        {
            out = {
                std::string{ candidate } + " -D --defined-only " + quoted +
                    " 2>/dev/null",
                std::regex{
                    R"(^[0-9a-fA-F]+ [A-Za-z] (sqlite3_[A-Za-z0-9_]+(?:@{1,2}[A-Za-z0-9_.-]+)?)$)" }
            };
            return true;
        }
    }
#endif

    return false;
}

TEST(sqlite3_symbol_seal, shared_library_exports_no_sqlite3_symbols)
{
    symbol_holder lister;
    if(!pick_tool(k_shared_library, lister))
    {
        GTEST_SKIP() << "need nm, llvm-nm, or dumpbin to inspect " << k_shared_library;
    }

    std::string output;
    ASSERT_TRUE(run_capture(lister.command, output))
        << "failed to inspect exported symbols of " << k_shared_library;

    std::vector<std::string> leaked;
    std::istringstream       stream(output);
    std::string              line;
    while(std::getline(stream, line))
    {
        if(!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::smatch match;
        if(std::regex_search(line, match, lister.pattern))
        {
            leaked.push_back(match[1].str());
        }
    }

    std::string detail;
    for(const auto& symbol : leaked)
    {
        detail += "\n  " + symbol;
    }

    EXPECT_TRUE(leaked.empty())
        << k_shared_library << " exports bundled sqlite3 symbol(s):" << detail
        << " (expected none, heck -fvisibility=hidden on profiler-hub-sqlite3-static "
           "and --exclude-libs on the profiler-hub shared link)";
}

}  // namespace
