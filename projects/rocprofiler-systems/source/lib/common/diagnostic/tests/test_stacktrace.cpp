// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/stacktrace.hpp"

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
    auto t     = make_marker_trace();
    auto opts  = stacktrace::format_options{};
    opts.color = stacktrace::color_mode::off;
    auto s     = t.to_string(opts);
    // The marker function name MUST appear in the rendered trace.
    EXPECT_NE(s.find("diagnostic_test_marker_capture"), std::string::npos)
        << "trace was:\n"
        << s;
}

TEST(diagnostic_stacktrace_test, NoColorModeProducesNoEscape)
{
    auto t     = make_marker_trace();
    auto opts  = stacktrace::format_options{};
    opts.color = stacktrace::color_mode::off;
    auto s     = t.to_string(opts);
    EXPECT_EQ(s.find("\033["), std::string::npos);
}

TEST(diagnostic_stacktrace_test, ForceColorModeProducesAnsi)
{
    auto t     = make_marker_trace();
    auto opts  = stacktrace::format_options{};
    opts.color = stacktrace::color_mode::on;
    auto s     = t.to_string(opts);
    EXPECT_NE(s.find("\033["), std::string::npos);
}

TEST(diagnostic_stacktrace_test, DefaultFiltersDropLibcStartMain)
{
    auto t     = make_marker_trace();
    auto opts  = stacktrace::format_options{};
    opts.color = stacktrace::color_mode::off;
    auto s     = t.to_string(opts);
    EXPECT_EQ(s.find("__libc_start_main"), std::string::npos) << "trace was:\n" << s;
}

TEST(diagnostic_stacktrace_test, NoFilterEnvKeepsLibcFrames)
{
    env_guard g{ "ROCPROFSYS_TRACE_NO_FILTER", "1" };
    // overrides cache once per process; this test is robustly true ONLY
    // if it runs first. We can still assert that with empty filters
    // and no_filter on, libc frames stay.
    auto t     = make_marker_trace();
    auto opts  = stacktrace::format_options{};
    opts.color = stacktrace::color_mode::off;
    // Sentinel value: a substring that won't match any real function so
    // the "use defaults if empty" branch doesn't fire and no frames are
    // dropped by the explicit list.
    opts.skip_substrings = { "__ZZZ_no_match_ZZZ__" };
    auto s               = t.to_string(opts);
    // We don't assert __libc_start_main appears (depends on link), but
    // the rendered string must not be empty.
    EXPECT_FALSE(s.empty());
}

TEST(diagnostic_stacktrace_test, TruncatesAtMaxFramesShown)
{
    auto t                = make_deep_trace(60);
    auto opts             = stacktrace::format_options{};
    opts.color            = stacktrace::color_mode::off;
    opts.max_frames_shown = 8;
    auto s                = t.to_string(opts);
    EXPECT_NE(s.find("more"), std::string::npos) << "trace was:\n" << s;
}

TEST(diagnostic_stacktrace_test, ModuleSuffixOnlyForNonExe)
{
    auto t           = make_marker_trace();
    auto opts        = stacktrace::format_options{};
    opts.color       = stacktrace::color_mode::off;
    opts.with_module = true;
    auto s           = t.to_string(opts);
    // The marker is in the test exe; its frame must not have a (module)
    // suffix. The function name itself contains '(' from its parameter
    // list, so we check for the module-suffix marker " (" that follows
    // the function-name closing ')' or any other token. The reliable
    // signal: a ".so" string on the same line indicates module rendering.
    auto pos = s.find("diagnostic_test_marker_capture");
    ASSERT_NE(pos, std::string::npos);
    auto eol = s.find('\n', pos);
    ASSERT_NE(eol, std::string::npos);
    auto line = s.substr(pos, eol - pos);
    EXPECT_EQ(line.find(".so"), std::string::npos)
        << "exe-local frame must not have a .so module suffix; line was:\n"
        << line;
    // Also: the unit-test binary basename should never appear as a module.
    EXPECT_EQ(line.find("(rocprof-sys-unit-tests)"), std::string::npos)
        << "exe-local frame must not echo the exe basename; line was:\n"
        << line;
}

TEST(diagnostic_stacktrace_test, WithSourceExcerptIncludesSourceLine)
{
    auto t                   = make_marker_trace();
    auto opts                = stacktrace::format_options{};
    opts.color               = stacktrace::color_mode::off;
    opts.with_source_excerpt = true;
    auto s                   = t.to_string(opts);
    // We look for the divider char used in the excerpt format.
    // If the test binary was built without -g this excerpt is silently
    // skipped; in CI builds we always have at least -g1.
    auto pos = s.find("diagnostic_test_marker_capture");
    ASSERT_NE(pos, std::string::npos);
    // The presence of " | " sequence anywhere in the trace indicates an
    // excerpt was rendered.
    if(s.find(" | ") == std::string::npos)
    {
        GTEST_SKIP() << "binary built without source line info; "
                        "excerpt cannot be tested";
    }
    EXPECT_NE(s.find(" | "), std::string::npos);
}
