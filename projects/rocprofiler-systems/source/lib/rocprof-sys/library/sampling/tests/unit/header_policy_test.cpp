// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Static grep-gate tests.
// NFR-PORT-3: no platform headers (<signal.h>, <ucontext.h>, <pthread.h>,
//             <linux/perf_event.h>, <sys/...>) in include/sampling/
// NFR-D-1:    no `virtual` keyword (as code) in include/sampling/ headers.
//
// Paths are baked in at compile time via ROCPROFSYS_SAMPLING_INCLUDE_DIR.
// The tests read each .hpp file from the filesystem and grep its content.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const fs::path k_include_dir{ ROCPROFSYS_SAMPLING_INCLUDE_DIR };

// Collect all .hpp files under include/sampling/ recursively.
static std::vector<fs::path>
all_sampling_headers()
{
    std::vector<fs::path> out;
    for(auto const& entry : fs::recursive_directory_iterator(k_include_dir))
    {
        if(entry.is_regular_file() && entry.path().extension() == ".hpp")
            out.push_back(entry.path());
    }
    return out;
}

static std::string
read_file(fs::path const& path)
{
    std::ifstream     ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Strip C++ line comments and block comments before searching for keywords.
// This prevents grep-gate from flagging "// no virtual" as a violation.
static std::string
strip_comments(std::string const& src)
{
    // Block comments: /* ... */
    static const std::regex k_block_comment(R"(/\*[\s\S]*?\*/)");
    // Line comments: // ...
    static const std::regex k_line_comment(R"(//[^\n]*)");
    std::string             out = std::regex_replace(src, k_block_comment, " ");
    out                         = std::regex_replace(out, k_line_comment, "");
    return out;
}

// ─── NFR-PORT-3: no platform headers in include/sampling/ ────────────────────

TEST(header_policy, no_platform_headers_in_public_sampling_includes)
{
    static const std::vector<std::string> k_forbidden_includes = {
        "#include <signal.h>",           "#include <ucontext.h>", "#include <pthread.h>",
        "#include <linux/perf_event.h>", "#include <sys/",        "#include <asm/",
    };

    auto        headers  = all_sampling_headers();
    bool        any_fail = false;
    std::string failures;

    for(auto const& hdr : headers)
    {
        auto content = read_file(hdr);
        for(auto const& forbidden : k_forbidden_includes)
        {
            if(content.find(forbidden) != std::string::npos)
            {
                any_fail = true;
                failures += "\n  " + hdr.string() + " contains: " + forbidden;
            }
        }
    }

    EXPECT_FALSE(any_fail)
        << "NFR-PORT-3 violation — platform header(s) in include/sampling/:" << failures;
}

// ─── NFR-D-1: no `virtual` keyword (as code) in include/sampling/ ─────────────

TEST(header_policy, no_virtual_keyword_in_public_sampling_headers)
{
    // Match `virtual` as a standalone word in code (not in comments or strings).
    static const std::regex k_virtual_word(R"(\bvirtual\b)");

    auto        headers  = all_sampling_headers();
    bool        any_fail = false;
    std::string failures;

    for(auto const& hdr : headers)
    {
        auto content   = read_file(hdr);
        auto code_only = strip_comments(content);

        if(std::regex_search(code_only, k_virtual_word))
        {
            any_fail = true;
            failures += "\n  " + hdr.string();
        }
    }

    EXPECT_FALSE(any_fail)
        << "NFR-D-1 violation — `virtual` found in include/sampling/ headers:"
        << failures;
}

// ─── Sanity: the include directory contains at least the expected headers ──────

TEST(header_policy, sampling_include_dir_contains_expected_headers)
{
    auto headers = all_sampling_headers();
    ASSERT_FALSE(headers.empty())
        << "ROCPROFSYS_SAMPLING_INCLUDE_DIR must point to include/sampling/ "
           "and contain at least one .hpp";

    // Spot-check that sampling_service.hpp is present.
    bool found_service = false;
    for(auto const& hdr : headers)
    {
        if(hdr.filename() == "sampling_service.hpp")
        {
            found_service = true;
            break;
        }
    }
    EXPECT_TRUE(found_service) << "include/sampling/sampling_service.hpp must exist";
}
