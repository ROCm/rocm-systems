// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression guard (GTest): the bundled SQLite must stay sealed.
//
// profiler-hub bundles its own static SQLite compiled with hidden visibility
// (-fvisibility=hidden) and links libprofiler-hub.so with --exclude-libs, so no
// sqlite3_* symbol is exported from the shared library and therefore cannot
// collide with (or be interposed by) other sqlite3 versions present on TheRock.
//
// This test inspects the BUILT shared library with `nm` and fails if any
// defined, exported sqlite3_* dynamic symbol reappears. It asserts a property
// of the build artifact, not runtime behavior; the absolute path of the library
// is injected by CMake via PROFILER_HUB_SHARED_LIB.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace
{

// Absolute path to the built libprofiler-hub.so, injected by CMake.
constexpr const char* k_shared_library = PROFILER_HUB_SHARED_LIB;

// Closes a popen() stream; used as a unique_ptr deleter. A dedicated functor
// (rather than decltype(&pclose)) avoids -Wignored-attributes on libc symbols.
struct file_op
{
    void operator()(FILE* pipe) const noexcept
    {
        if(pipe != nullptr)
        {
            pclose(pipe);
        }
    }
};

// Run a shell command and capture its stdout. Returns false if the command
// could not be launched.
bool
run_capture(const std::string& command, std::string& output)
{
    output.clear();
    std::unique_ptr<FILE, file_op> pipe(popen(command.c_str(), "r"));
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
    return true;
}

// Locate an `nm`-like tool; empty string if none is available.
std::string
find_nm()
{
    for(const char* candidate : { "nm", "llvm-nm" })
    {
        std::string out;
        if(run_capture(std::string{ candidate } + " --version 2>/dev/null", out) &&
           !out.empty())
        {
            return candidate;
        }
    }
    return {};
}

TEST(sqlite3_symbol_seal, shared_library_exports_no_sqlite3_symbols)
{
    const std::string nm = find_nm();
    if(nm.empty())
    {
        GTEST_SKIP() << "no 'nm'/'llvm-nm' available to inspect exported symbols";
    }

    // -D: dynamic (exported) symbols, --defined-only: only symbols defined here.
    std::string nm_output;
    ASSERT_TRUE(run_capture(
        nm + " -D --defined-only \"" + k_shared_library + "\" 2>/dev/null", nm_output))
        << "failed to run " << nm << " on " << k_shared_library;

    // Match exact sqlite3_* symbol names, e.g. "0000... T sqlite3_open"; ignore
    // mangled C++ names that merely contain the substring "sqlite3_".
    const std::regex exported_sqlite{
        R"(^[0-9a-fA-F]+ [A-Za-z] (sqlite3_[A-Za-z0-9_]+)$)"
    };

    std::vector<std::string> leaked;
    std::istringstream       stream(nm_output);
    std::string              line;
    while(std::getline(stream, line))
    {
        if(!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::smatch match;
        if(std::regex_match(line, match, exported_sqlite))
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
        << "'" << k_shared_library << "' exports " << leaked.size()
        << " bundled sqlite3_* symbol(s):" << detail
        << "\nThese must remain local (hidden) so they cannot collide with other "
           "sqlite3 versions on TheRock. Check the -fvisibility=hidden compile "
           "option on profiler-hub-sqlite3-static and the --exclude-libs link "
           "option on the profiler-hub shared library.";
}

}  // namespace
