// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/stacktrace.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

using rocprofsys::common::diagnostic::stacktrace;

namespace
{
// Marker function so the captured trace contains an unambiguous name.
// noinline alone is insufficient under -O3: gcc TCO/NRVO away the body
// and the resulting trace skips the marker. Pass an out-parameter and
// add a memory barrier between the capture and the return so the marker
// function leaves a real frame on the stack.
[[gnu::noinline]] void
diagnostic_test_marker_capture(stacktrace* out)
{
    asm volatile("" ::: "memory");
    *out = stacktrace::capture();
    asm volatile("" ::"r"(out) : "memory");
}

[[gnu::noinline]] void
deep_recurse(int n, stacktrace* out)
{
    asm volatile("" ::: "memory");
    if(n <= 0)
    {
        *out = stacktrace::capture();
        asm volatile("" ::"r"(out) : "memory");
        return;
    }
    deep_recurse(n - 1, out);
    asm volatile("" ::"r"(out) : "memory");
}

stacktrace
make_marker_trace()
{
    stacktrace t;
    diagnostic_test_marker_capture(&t);
    return t;
}

stacktrace
make_deep_trace(int n)
{
    stacktrace t;
    deep_recurse(n, &t);
    return t;
}

class env_guard
{
public:
    env_guard(const char* name, const char* value)
    : m_name{ name }
    {
        if(const char* prev = std::getenv(name); prev != nullptr)
        {
            m_had   = true;
            m_saved = prev;
        }
        if(value != nullptr)
        {
            ::setenv(name, value, 1);
        }
        else
        {
            ::unsetenv(name);
        }
    }

    ~env_guard()
    {
        if(m_had)
        {
            ::setenv(m_name, m_saved.c_str(), 1);
        }
        else
        {
            ::unsetenv(m_name);
        }
    }

    env_guard(const env_guard&)            = delete;
    env_guard& operator=(const env_guard&) = delete;
    env_guard(env_guard&&)                 = delete;
    env_guard& operator=(env_guard&&)      = delete;

private:
    const char* m_name = nullptr;
    bool        m_had  = false;
    std::string m_saved;
};

// Count occurrences of `needle` in `hay`.
std::size_t
count_occurrences(const std::string& hay, const std::string& needle)
{
    if(needle.empty()) return 0;
    std::size_t n = 0;
    for(std::size_t pos = 0; (pos = hay.find(needle, pos)) != std::string::npos;
        pos += needle.size())
    {
        ++n;
    }
    return n;
}

stacktrace::format_options
plain_opts()
{
    stacktrace::format_options o;
    o.with_color = false;
    return o;
}
}  // namespace

TEST(diagnostic_stacktrace_test, CaptureReturnsNonEmpty)
{
    auto t = make_marker_trace();
    EXPECT_FALSE(t.empty());
    auto [data, n] = t.raw();
    EXPECT_GT(n, 0u);
    EXPECT_NE(data, nullptr);
}

TEST(diagnostic_stacktrace_test, CaptureIncludesCallerName)
{
    auto t = make_marker_trace();
    auto s = t.to_string(plain_opts());
    EXPECT_NE(s.find("diagnostic_test_marker_capture"), std::string::npos)
        << "trace was:\n"
        << s;
}

TEST(diagnostic_stacktrace_test, FrameZeroAppears)
{
    auto t = make_marker_trace();
    auto s = t.to_string(plain_opts());
    EXPECT_NE(s.find("#0 "), std::string::npos) << "trace was:\n" << s;
}

TEST(diagnostic_stacktrace_test, FramesNumberedMonotonically)
{
    auto t = make_marker_trace();
    auto s = t.to_string(plain_opts());

    std::size_t prev = std::string::npos;
    for(std::size_t i = 0; i < 4; ++i)
    {
        const std::string tag = "#" + std::to_string(i) + " ";
        auto              pos = s.find(tag);
        ASSERT_NE(pos, std::string::npos) << "frame index " << i << " missing in:\n" << s;
        if(prev != std::string::npos)
        {
            EXPECT_GT(pos, prev) << "frame indices not in order: " << s;
        }
        prev = pos;
    }
}

TEST(diagnostic_stacktrace_test, SingleLinePerFrame)
{
    auto t = make_marker_trace();
    auto s = t.to_string(plain_opts());

    // One newline per `#N` token. Trailing trailers (... skipped ..., ... more
    // ...) add their own lines; subtract them.
    std::size_t newlines = std::count(s.begin(), s.end(), '\n');
    std::size_t trailers = count_occurrences(s, "...");

    // Total `#N` tokens. Walk frame indices 0..N-1.
    std::size_t frame_lines = 0;
    while(s.find("#" + std::to_string(frame_lines) + " ") != std::string::npos)
    {
        ++frame_lines;
    }
    ASSERT_GT(frame_lines, 0u);
    EXPECT_EQ(newlines, frame_lines + trailers / 2)
        << "expected one line per `#N` frame plus trailers; trace was:\n"
        << s;
}

TEST(diagnostic_stacktrace_test, DefaultIsColored)
{
    auto t = make_marker_trace();
    auto s = t.to_string();  // defaults: with_color = true
    EXPECT_NE(s.find("\033["), std::string::npos)
        << "default options must produce ANSI escapes; trace was:\n"
        << s;
}

TEST(diagnostic_stacktrace_test, WithColorFalseNoEscapes)
{
    auto t = make_marker_trace();
    auto s = t.to_string(plain_opts());
    EXPECT_EQ(s.find("\033["), std::string::npos);
}

TEST(diagnostic_stacktrace_test, DefaultFiltersDropLibcStartMain)
{
    auto t = make_marker_trace();
    auto s = t.to_string(plain_opts());
    EXPECT_EQ(s.find("__libc_start_main"), std::string::npos) << "trace was:\n" << s;
}

TEST(diagnostic_stacktrace_test, NoFilterEnvKeepsLibcFrames)
{
    env_guard g{ "ROCPROFSYS_TRACE_NO_FILTER", "1" };
    auto      t          = make_marker_trace();
    auto      opts       = plain_opts();
    opts.skip_substrings = { "__ZZZ_no_match_ZZZ__" };
    auto s               = t.to_string(opts);
    EXPECT_FALSE(s.empty());
}

TEST(diagnostic_stacktrace_test, TruncatesAtMaxFramesShown)
{
    auto t                = make_deep_trace(60);
    auto opts             = plain_opts();
    opts.max_frames_shown = 8;
    auto s                = t.to_string(opts);
    EXPECT_NE(s.find("more"), std::string::npos) << "trace was:\n" << s;
}

TEST(diagnostic_stacktrace_test, ModuleBasenameOnly)
{
    auto t           = make_marker_trace();
    auto opts        = plain_opts();
    opts.with_module = true;
    auto s           = t.to_string(opts);

    // Frame for the marker is in the test exe; its line must NOT carry a
    // module suffix.
    auto pos = s.find("diagnostic_test_marker_capture");
    ASSERT_NE(pos, std::string::npos);
    auto eol = s.find('\n', pos);
    ASSERT_NE(eol, std::string::npos);
    auto line = s.substr(pos, eol - pos);
    EXPECT_EQ(line.find(".so"), std::string::npos)
        << "exe-local frame must not have a .so module suffix; line was:\n"
        << line;
    EXPECT_EQ(line.find("(rocprof-sys-unit-tests)"), std::string::npos)
        << "exe-local frame must not echo the exe basename; line was:\n"
        << line;

    // No module string anywhere in the formatted output should contain a '/'.
    for(std::size_t open = 0; (open = s.find('(', open)) != std::string::npos; ++open)
    {
        auto close = s.find(')', open);
        if(close == std::string::npos) break;
        auto inside = s.substr(open + 1, close - open - 1);
        EXPECT_EQ(inside.find('/'), std::string::npos)
            << "module shown with full path; substring was: " << inside;
    }
}
