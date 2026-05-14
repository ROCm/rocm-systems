// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_tool_setup.h"

#include "rocprofiler_compute_tool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

using namespace rocprofiler_compute_tool;

namespace
{
bool contains_set_env_call(const std::vector<MockEnvironmentSetUp::set_env_call>& calls,
                           const std::string&                                     key,
                           const std::string&                                     value)
{
    return std::any_of(calls.begin(),
                       calls.end(),
                       [&](const MockEnvironmentSetUp::set_env_call& call)
                       { return call.key == key && call.value == value; });
}

bool contains_set_env_key(const std::vector<MockEnvironmentSetUp::set_env_call>& calls,
                          const std::string&                                     key)
{
    return std::any_of(calls.begin(),
                       calls.end(),
                       [&](const MockEnvironmentSetUp::set_env_call& call)
                       { return call.key == key; });
}
}  // namespace

TEST_F(TestToolSetUp, OnConfigure_InvokesToolSetupSetUp)
{
    rocprofiler_configure(1, "", 1, &m_client_id);
    EXPECT_EQ(m_tool_setup->get_setup_call_count(), 1);
}

TEST_F(TestToolSetUp, OnConfigureCalledTwice_InvokesSetUpEachTime)
{
    rocprofiler_configure(1, "", 1, &m_client_id);
    rocprofiler_configure(1, "", 1, &m_client_id);
    EXPECT_EQ(m_tool_setup->get_setup_call_count(), 2);
}

//////////////////////////////////////////////////////////////////////////
/// TestToolSetUp
void TestToolSetUp::SetUp()
{
    m_input_parameters = std::make_shared<MockInputParameters>();
    m_sdk_wrapper      = std::make_shared<MockSdkWrapper>();
    m_counters_writer  = std::make_shared<MockCountersWriter>();
    m_tool_setup       = std::make_shared<MockToolSetUp>();

    test_knobs::set_input_parameters(m_input_parameters);
    test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    test_knobs::set_csv_writer(m_counters_writer);
    test_knobs::set_tool_setup(m_tool_setup);
}

void TestToolSetUp::TearDown()
{
    test_knobs::reset_cfg();
    test_knobs::reset_tool_setup();
}

//////////////////////////////////////////////////////////////////////////
/// TestEnvironmentSetUp
void TestEnvironmentSetUp::SetUp()
{
    m_tool_setup = std::make_shared<MockEnvironmentSetUp>();
}

TEST_F(TestEnvironmentSetUp, IsShellTarget_WhenRocprofShellTargetIsOne_ReturnsTrue)
{
    m_tool_setup->set_test_env({"ROCPROF_SHELL_TARGET=1"});
    m_tool_setup->set_up();
    EXPECT_TRUE(m_tool_setup->is_shell_target());
}

TEST_F(TestEnvironmentSetUp, IsShellTarget_WhenRocprofShellTargetIsZero_ReturnsFalse)
{
    for (const auto* shell_target_value : {"0", "", "true", "yes", " 1", "1 "})
    {
        SCOPED_TRACE(std::string{"ROCPROF_SHELL_TARGET="} + shell_target_value);
        m_tool_setup->set_test_env({std::string{"ROCPROF_SHELL_TARGET="} + shell_target_value});
        m_tool_setup->set_up();
        EXPECT_FALSE(m_tool_setup->is_shell_target());
    }
}

TEST_F(TestEnvironmentSetUp, IsShellTarget_WhenRocprofShellTargetMissing_ReturnsFalse)
{
    m_tool_setup->set_test_env({"HOME=/home/user", "ROCPROF_FOO=bar"});
    m_tool_setup->set_up();
    EXPECT_FALSE(m_tool_setup->is_shell_target());
}

TEST_F(TestEnvironmentSetUp, RepublishRocprofEnv_RepublishesAllRocprofVars)
{
    m_tool_setup->set_test_env({"ROCPROF_FOO=bar", "ROCPROF_BAZ=qux", "HOME=/x"});
    m_tool_setup->set_up();
    m_tool_setup->republish_rocprof_env();

    const auto& calls = m_tool_setup->get_set_env_calls();
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROF_FOO", "bar"));
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROF_BAZ", "qux"));
    EXPECT_FALSE(contains_set_env_key(calls, "HOME"));
}

TEST_F(TestEnvironmentSetUp, RepublishRocprofEnv_AlsoRepublishesRocprofilerPrefixedVars)
{
    m_tool_setup->set_test_env({
        "ROCPROFILER_TOOL_PATH=/path/to/tool.so",
        "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1",
        "ROCPROF_FOO=bar",
        "PATH=/usr/bin",
    });
    m_tool_setup->set_up();
    m_tool_setup->republish_rocprof_env();

    const auto& calls = m_tool_setup->get_set_env_calls();
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROFILER_TOOL_PATH", "/path/to/tool.so"));
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROFILER_PC_SAMPLING_BETA_ENABLED", "1"));
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROF_FOO", "bar"));
    EXPECT_FALSE(contains_set_env_key(calls, "PATH"));
}

TEST_F(TestEnvironmentSetUp, RepublishRocprofEnv_UsesValuesFromCache)
{
    m_tool_setup->set_test_env({"ROCPROF_FOO=original"});
    m_tool_setup->set_up();
    m_tool_setup->republish_rocprof_env();

    const auto& calls = m_tool_setup->get_set_env_calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].key, "ROCPROF_FOO");
    EXPECT_EQ(calls[0].value, "original");
}

TEST_F(TestEnvironmentSetUp, SetUp_WhenShellTarget_InvokesRepublish)
{
    m_tool_setup->set_test_env({"ROCPROF_SHELL_TARGET=1", "ROCPROF_FOO=bar"});
    m_tool_setup->set_up();

    const auto& calls = m_tool_setup->get_set_env_calls();
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROF_SHELL_TARGET", "1"));
    EXPECT_TRUE(contains_set_env_call(calls, "ROCPROF_FOO", "bar"));
}

TEST_F(TestEnvironmentSetUp, SetUp_WhenNotShellTarget_DoesNotInvokeRepublish)
{
    m_tool_setup->set_test_env({"ROCPROF_FOO=bar"});
    m_tool_setup->set_up();

    EXPECT_TRUE(m_tool_setup->get_set_env_calls().empty());
}

TEST_F(TestEnvironmentSetUp, SetUp_AlwaysBuildsEnvCache)
{
    m_tool_setup->set_test_env({"ROCPROF_FOO=bar"});
    m_tool_setup->set_up();
    m_tool_setup->set_up();

    EXPECT_EQ(m_tool_setup->get_build_env_cache_call_count(), 2);
}
