// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launcher.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cerrno>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::common_utils::tool_mode;
using rocprofsys::rocsys::env_op;
using rocprofsys::rocsys::launch_plan;
using rocprofsys::rocsys::launcher;

namespace
{
class gmock_system
{
public:
    MOCK_METHOD(int, set_env, (const char* key, const char* value));
    MOCK_METHOD(int, unset_env, (const char* key));
    MOCK_METHOD(bool, exists, (const char* path));
    MOCK_METHOD(std::string, executable_directory, ());
    MOCK_METHOD(std::vector<std::string>, environ, ());
    MOCK_METHOD(int, run_tool, (int argc, char** argv, tool_mode mode));
    MOCK_METHOD(int, exec, (const char* path, char** argv, char** envp));
    MOCK_METHOD(void, write_error, (std::string_view message));
};

// Forwards to the test's global mock so launcher<bound_system> stays a value type.
std::unique_ptr<::testing::StrictMock<gmock_system>> g_mock;

struct bound_system
{
    int set_env(const char* key, const char* value) const
    {
        return g_mock->set_env(key, value);
    }
    int         unset_env(const char* key) const { return g_mock->unset_env(key); }
    bool        exists(const char* path) const { return g_mock->exists(path); }
    std::string executable_directory() const { return g_mock->executable_directory(); }
    std::vector<std::string> environ() const { return g_mock->environ(); }
    int                      run_tool(int argc, char** argv, tool_mode mode) const
    {
        return g_mock->run_tool(argc, argv, mode);
    }
    int exec(const char* path, char** argv, char** envp) const
    {
        return g_mock->exec(path, argv, envp);
    }
    void write_error(std::string_view message) const { g_mock->write_error(message); }
};

static_assert(rocprofsys::rocsys::system_ops<bound_system>);

[[nodiscard]] std::vector<std::string>
copy_pointers(char** values)
{
    std::vector<std::string> copied;
    for(char** entry = values; entry != nullptr && *entry != nullptr; ++entry)
    {
        copied.emplace_back(*entry);
    }
    return copied;
}

class launcher_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock = std::make_unique<::testing::StrictMock<gmock_system>>();
    }

    void TearDown() override { g_mock.reset(); }
};

}  // namespace

TEST_F(launcher_test, profile_sets_env_then_runs_the_tool)
{
    launch_plan              planned{ .in_process = true,
                                      .program    = "rocsys",
                                      .argv       = { "rocsys", "--", "./app" },
                                      .env  = { env_op{ .key   = "ROCPROFSYS_ROCM_DOMAINS",
                                                        .value = "kernel_dispatch" } },
                                      .mode = tool_mode::run };
    std::vector<std::string> seen_argv;

    ::testing::InSequence order;
    EXPECT_CALL(*g_mock, set_env(::testing::StrEq("ROCPROFSYS_ROCM_DOMAINS"),
                                 ::testing::StrEq("kernel_dispatch")))
        .WillOnce(::testing::Return(0));
    EXPECT_CALL(*g_mock, run_tool(::testing::_, ::testing::NotNull(), tool_mode::run))
        .WillOnce([&seen_argv](int, char** argv, tool_mode) {
            seen_argv = copy_pointers(argv);
            return 0;
        });

    EXPECT_EQ(launcher<bound_system>{}.launch(planned, "rocsys"), 0);
    EXPECT_EQ(seen_argv, planned.argv);
}

TEST_F(launcher_test, profile_with_an_empty_plan_does_not_change_the_environment)
{
    launch_plan planned{ .in_process = true,
                         .program    = "rocsys",
                         .argv       = { "rocsys", "--", "./app" },
                         .mode       = tool_mode::run };
    EXPECT_CALL(*g_mock, run_tool(::testing::_, ::testing::NotNull(), tool_mode::run))
        .WillOnce(::testing::Return(11));

    EXPECT_EQ(launcher<bound_system>{}.launch(planned, "rocsys"), 11);
}

TEST_F(launcher_test, missing_sibling_is_reported_without_exec)
{
    launch_plan planned{ .program = "rocprof-sys-instrument",
                         .argv    = { "rocprof-sys-instrument", "--help" } };
    std::string reported;

    ::testing::InSequence order;
    EXPECT_CALL(*g_mock, executable_directory())
        .WillOnce(::testing::Return("/opt/rocm/bin"));
    EXPECT_CALL(*g_mock, exists(::testing::StrEq("/opt/rocm/bin/rocprof-sys-instrument")))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*g_mock, write_error(::testing::_))
        .WillOnce(::testing::SaveArg<0>(&reported));

    EXPECT_EQ(launcher<bound_system>{}.launch(planned, "rocsys"), 127);
    EXPECT_NE(reported.find("rocprof-sys-instrument"), std::string::npos);
    EXPECT_NE(reported.find("was not found"), std::string::npos);
    EXPECT_NE(reported.find("/opt/rocm/bin/rocprof-sys-instrument"), std::string::npos);
    EXPECT_NE(reported.find("hint:"), std::string::npos);
}

TEST_F(launcher_test, unresolved_install_directory_keeps_the_existing_message)
{
    launch_plan planned{ .program = "rocprof-sys-avail",
                         .argv    = { "rocprof-sys-avail" } };
    std::string reported;
    EXPECT_CALL(*g_mock, executable_directory()).WillOnce(::testing::Return(""));
    EXPECT_CALL(*g_mock, write_error(::testing::_))
        .WillOnce(::testing::SaveArg<0>(&reported));

    EXPECT_EQ(launcher<bound_system>{}.launch(planned, "rocsys"), 127);
    EXPECT_NE(reported.find("failed to resolve install directory"), std::string::npos);
}

TEST_F(launcher_test, exec_merges_env_and_does_not_setenv)
{
    launch_plan              planned{ .program = "rocprof-sys-instrument",
                                      .argv = { "rocprof-sys-instrument", "--", "./app" },
                                      .env = { env_op{ .key = "FOO", .value = "new" } } };
    std::vector<std::string> seen_argv;
    std::vector<std::string> seen_env;
    std::string              seen_path;
    std::string              reported;

    ::testing::InSequence order;
    EXPECT_CALL(*g_mock, executable_directory())
        .WillOnce(::testing::Return("/opt/rocm/bin"));
    EXPECT_CALL(*g_mock, exists(::testing::StrEq("/opt/rocm/bin/rocprof-sys-instrument")))
        .WillOnce(::testing::Return(true));
    EXPECT_CALL(*g_mock, environ())
        .WillOnce(::testing::Return(
            std::vector<std::string>{ "FOO=old", "FOOBAR=keep", "OTHER=1" }));
    EXPECT_CALL(*g_mock, exec(::testing::_, ::testing::NotNull(), ::testing::NotNull()))
        .WillOnce([&](const char* path, char** argv, char** envp) {
            seen_path = path;
            seen_argv = copy_pointers(argv);
            seen_env  = copy_pointers(envp);
            return EACCES;
        });
    EXPECT_CALL(*g_mock, write_error(::testing::_))
        .WillOnce(::testing::SaveArg<0>(&reported));

    EXPECT_EQ(launcher<bound_system>{}.launch(planned, "rocsys"), 127);
    EXPECT_EQ(seen_path, "/opt/rocm/bin/rocprof-sys-instrument");
    EXPECT_EQ(seen_argv, planned.argv);
    EXPECT_EQ(seen_env,
              (std::vector<std::string>{ "FOOBAR=keep", "OTHER=1", "FOO=new" }));
    EXPECT_NE(reported.find("failed to execute"), std::string::npos);
}

TEST_F(launcher_test, set_env_failure_stops_before_run_tool)
{
    launch_plan planned{ .in_process = true,
                         .program    = "rocsys",
                         .argv       = { "rocsys", "--", "./app" },
                         .env        = { env_op{ .key = "FOO", .value = "bar" } },
                         .mode       = tool_mode::run };
    EXPECT_CALL(*g_mock, set_env(::testing::StrEq("FOO"), ::testing::StrEq("bar")))
        .WillOnce(::testing::Return(EPERM));
    EXPECT_CALL(*g_mock, write_error(::testing::_));

    EXPECT_EQ(launcher<bound_system>{}.launch(planned, "rocsys"), EXIT_FAILURE);
}
