// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/process_environment_info.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <stdlib.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace
{
nlohmann::json
filter(const std::vector<const char*>& block, std::string_view home = {})
{
    return rocprofsys::filter_environment(block, home);
}

// Sets an environment variable for one test and restores the previous state on exit.
class scoped_env
{
public:
    scoped_env(const char* name, const char* value)
    : m_name{ name }
    {
        if(const char* previous = ::getenv(name); previous != nullptr)
        {
            m_previous = previous;
        }
        ::setenv(name, value, 1);
    }

    ~scoped_env()
    {
        if(m_previous.has_value())
        {
            ::setenv(m_name.c_str(), m_previous->c_str(), 1);
        }
        else
        {
            ::unsetenv(m_name.c_str());
        }
    }

    scoped_env(const scoped_env&)            = delete;
    scoped_env& operator=(const scoped_env&) = delete;

private:
    std::string                m_name;
    std::optional<std::string> m_previous;
};
}  // namespace

TEST(process_environment_info_test, keeps_only_allowlisted_variables)
{
    // Names a keyword blocklist would miss, near-misses of allowed names and entries
    // that are not "NAME=value": none of them may reach a shared database.
    const auto environment = filter({
        "ROCPROFSYS_OPTIONS=a=b=c",  // kept; the value may contain '='
        "LD_PRELOAD=libfoo.so",
        "PATH=/usr/bin:/opt/rocm/bin",
        "SLURM_PROCID=2",
        "DATABASE_URL=postgres://user:pw@host/db",
        "AWS_SESSION=abc",
        "SLURM_JOB_NAME=secret-project",
        "ROCPROFSYS=1",           // a prefix rule needs its trailing '_'
        "PMI_RANKS=1",            // an exact rule must not act as a prefix
        "MPI_COMM_WORLD_RANK=7",  // synthesized at MPI_Init, never read from the env
        "NOEQUALS",
        "=NONAME",
    });

    const auto expected = nlohmann::json{
        { "ROCPROFSYS_OPTIONS", "a=b=c" },
        { "LD_PRELOAD", "libfoo.so" },
        { "PATH", "/usr/bin:/opt/rocm/bin" },
        { "SLURM_PROCID", "2" },
    };
    EXPECT_EQ(environment, expected);
}

TEST(process_environment_info_test, yields_an_empty_object_for_an_empty_environment)
{
    // The stored column is a JSON object; "null" would break readers of the database.
    EXPECT_EQ(rocprofsys::dump_environment(filter({})), "{}");
}

TEST(process_environment_info_test, redacts_every_whole_component_occurrence_of_home)
{
    const auto environment = filter(
        {
            "LD_LIBRARY_PATH=/home/bob/lib:/opt/rocm/lib:/home/bob",
            "ROCPROFSYS_OPTIONS=--out=/home/bob/a,/home/bob/b",
            "ROCPROFSYS_TRACE_PATH=/home/bob,/tmp /home/bob \"/home/bob\"",
        },
        "/home/bob/");  // a trailing slash is tolerated

    const auto expected = nlohmann::json{
        { "LD_LIBRARY_PATH", "~/lib:/opt/rocm/lib:~" },
        { "ROCPROFSYS_OPTIONS", "--out=~/a,~/b" },
        { "ROCPROFSYS_TRACE_PATH", "~,/tmp ~ \"~\"" },
    };
    EXPECT_EQ(environment, expected);
}

TEST(process_environment_info_test, does_not_redact_a_name_that_merely_contains_home)
{
    // A match that continues a longer path component, or sits inside another path, is
    // not the home directory.
    const auto environment = filter(
        { "LD_LIBRARY_PATH=/home/bobby/lib:/home/bob2:/mnt/home/bob/lib" }, "/home/bob");

    const auto expected = nlohmann::json{
        { "LD_LIBRARY_PATH", "/home/bobby/lib:/home/bob2:/mnt/home/bob/lib" },
    };
    EXPECT_EQ(environment, expected);
}

TEST(process_environment_info_test, unset_or_root_home_disables_redaction)
{
    // HOME is commonly unset in containers; it must not rewrite every path to "~...".
    const std::vector<const char*> block{ "LD_LIBRARY_PATH=/home/alice/lib" };
    const auto expected = nlohmann::json{ { "LD_LIBRARY_PATH", "/home/alice/lib" } };

    EXPECT_EQ(filter(block, ""), expected);
    EXPECT_EQ(filter(block, "/"), expected);
}

TEST(process_environment_info_test, replaces_invalid_utf8_instead_of_throwing)
{
    // Runs during process start-up: a stray byte in a value must not kill the profiled
    // application.
    const auto environment = filter({ "LD_LIBRARY_PATH=/opt/caf\xe9/lib" });

    std::string json;
    ASSERT_NO_THROW(json = rocprofsys::dump_environment(environment));
    EXPECT_EQ(json, "{\"LD_LIBRARY_PATH\":\"/opt/caf\xEF\xBF\xBD/lib\"}");
}

TEST(process_environment_info_test, capture_reads_the_real_environment_and_home)
{
    const scoped_env home{ "HOME", "/home/capture_tester" };
    const scoped_env allowed{ "ROCPROFSYS_CAPTURE_TEST", "/home/capture_tester/out" };
    const scoped_env unlisted{ "CAPTURE_TEST_SECRET", "hunter2" };

    const auto environment = rocprofsys::capture_process_environment();

    EXPECT_EQ(environment.value("ROCPROFSYS_CAPTURE_TEST", ""), "~/out");
    EXPECT_FALSE(environment.contains("CAPTURE_TEST_SECRET"));
    EXPECT_FALSE(environment.contains("HOME"));
}
