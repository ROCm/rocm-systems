// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#define ROCPROFILER_SDK_EXPERIMENTAL

#include "test_pc_sampling_input.h"

#include "environ_cache.h"
#include "input_parameters.h"
#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace rocprofiler_compute_tool;

TEST_F(TestPcSamplingInput, EnvInputParameters_PcSamplingInterval_ReturnInjectedValue)
{
    Envp               envp{{"ROCPROF_PC_SAMPLING_INTERVAL=1048576"}};
    EnvInputParameters input_parameters{std::make_shared<EnvironCache>(envp.data())};

    EXPECT_EQ(input_parameters.get_pc_sampling_interval(), std::string_view{"1048576"});
}

TEST_F(TestPcSamplingInput, EnvInputParameters_PcSamplingIntervalUnset_ReturnEmpty)
{
    Envp               envp{{}};
    EnvInputParameters input_parameters{std::make_shared<EnvironCache>(envp.data())};

    EXPECT_EQ(input_parameters.get_pc_sampling_interval(), std::string_view{""});
}

TEST_F(TestPcSamplingInput, MockInputParameters_SetPcSamplingInterval_RoundTrips)
{
    m_input_parameters->set_pc_sampling_interval("256");
    EXPECT_EQ(m_input_parameters->get_pc_sampling_interval(), std::string_view{"256"});
}

TEST_F(TestPcSamplingInput, MockInputParameters_PcSamplingInterval_DefaultEmpty)
{
    EXPECT_EQ(m_input_parameters->get_pc_sampling_interval(), std::string_view{""});
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_SupportingAgent_CreatesBufferAndConfiguresService)
{
    constexpr rocprofiler_agent_id_t agent{42};
    m_input_parameters->set_pc_sampling_beta_enabled("1");
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_sdk_wrapper->set_available_gpu_agents({agent});
    m_sdk_wrapper->set_pc_sampling_config(/*min_interval=*/1,
                                          /*max_interval=*/1000,
                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);
    m_sdk_wrapper->set_configure_pc_sampling_status(ROCPROFILER_STATUS_SUCCESS);

    drive_hsa_runtime_loaded();

    // Two buffers are created: one for PC samples and one for the buffered
    // kernel-dispatch tracing service (the timestamp source for kernel_dispatch
    // records in the results JSON).
    ASSERT_EQ(m_sdk_wrapper->get_create_buffer_info().size(), 2u);
    ASSERT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info().size(), 1u);
    EXPECT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info()[0].agent.handle, agent.handle);

    // The kernel-dispatch buffer tracing service was configured exactly once.
    ASSERT_EQ(m_sdk_wrapper->get_buffer_tracing_service_info().size(), 1u);
    EXPECT_EQ(m_sdk_wrapper->get_buffer_tracing_service_info()[0].kind,
              ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_ConfigureReturnsError_DoesNotThrowAndStillAttempts)
{
    constexpr rocprofiler_agent_id_t agent{7};
    m_input_parameters->set_pc_sampling_beta_enabled("1");
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_sdk_wrapper->set_available_gpu_agents({agent});
    m_sdk_wrapper->set_pc_sampling_config(/*min_interval=*/1,
                                          /*max_interval=*/1000,
                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);
    m_sdk_wrapper->set_configure_pc_sampling_status(ROCPROFILER_STATUS_ERROR);

    EXPECT_NO_THROW(drive_hsa_runtime_loaded());

    // The configure call was attempted (recorded) even though it failed.
    EXPECT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info().size(), 1u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalEnvUnset_UsesValueWithinAdvertisedRange)
{
    constexpr rocprofiler_agent_id_t agent{99};
    constexpr uint64_t               min_interval = 64;
    constexpr uint64_t               max_interval = 4096;

    // Interval env intentionally left unset (default empty).
    m_input_parameters->set_pc_sampling_beta_enabled("1");
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_input_parameters->set_pc_sampling_interval("");
    m_sdk_wrapper->set_available_gpu_agents({agent});
    m_sdk_wrapper->set_pc_sampling_config(min_interval,
                                          max_interval,
                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);
    m_sdk_wrapper->set_configure_pc_sampling_status(ROCPROFILER_STATUS_SUCCESS);

    drive_hsa_runtime_loaded();

    ASSERT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info().size(), 1u);
    const auto chosen_interval = m_sdk_wrapper->get_configure_pc_sampling_info()[0].interval;
    EXPECT_GE(chosen_interval, min_interval);
    EXPECT_LE(chosen_interval, max_interval);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalInRange_PassedThrough)
{
    EXPECT_EQ(configured_interval_for("256", 64, 4096), 256u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalAboveMax_ClampedToMax)
{
    EXPECT_EQ(configured_interval_for("100000", 64, 4096), 4096u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalBelowMin_RaisedToMin)
{
    EXPECT_EQ(configured_interval_for("1", 64, 4096), 64u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalNonNumeric_FallsBackToMin)
{
    EXPECT_EQ(configured_interval_for("abc", 64, 4096), 64u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalNegative_FallsBackToMin)
{
    // "-1" must not wrap to a huge value then clamp to max; it is invalid input.
    EXPECT_EQ(configured_interval_for("-1", 64, 4096), 64u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_IntervalTrailingGarbage_FallsBackToMin)
{
    EXPECT_EQ(configured_interval_for("100abc", 64, 4096), 64u);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_MultipleSupportingAgents_ConfiguresEach)
{
    constexpr rocprofiler_agent_id_t agent_a{11};
    constexpr rocprofiler_agent_id_t agent_b{22};
    m_input_parameters->set_pc_sampling_beta_enabled("1");
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_sdk_wrapper->set_available_gpu_agents({agent_a, agent_b});
    m_sdk_wrapper->set_pc_sampling_config(1,
                                          1000,
                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);
    m_sdk_wrapper->set_configure_pc_sampling_status(ROCPROFILER_STATUS_SUCCESS);

    drive_hsa_runtime_loaded();

    ASSERT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info().size(), 2u);
    EXPECT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info()[0].agent.handle, agent_a.handle);
    EXPECT_EQ(m_sdk_wrapper->get_configure_pc_sampling_info()[1].agent.handle, agent_b.handle);
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_AgentLacksRequestedMethod_SkipsConfigure)
{
    constexpr rocprofiler_agent_id_t agent{33};
    m_input_parameters->set_pc_sampling_beta_enabled("1");
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_sdk_wrapper->set_available_gpu_agents({agent});
    // Agent advertises STOCHASTIC only; requested method is HOST_TRAP.
    m_sdk_wrapper->set_pc_sampling_config(1,
                                          1000,
                                          ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);

    EXPECT_NO_THROW(drive_hsa_runtime_loaded());

    EXPECT_TRUE(m_sdk_wrapper->get_configure_pc_sampling_info().empty());
}

TEST_F(TestPcSamplingInput, OnHsaRuntimeLoaded_BetaDisabled_ConfiguresNothing)
{
    constexpr rocprofiler_agent_id_t agent{44};
    // Beta intentionally left unset.
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_sdk_wrapper->set_available_gpu_agents({agent});
    m_sdk_wrapper->set_pc_sampling_config(1,
                                          1000,
                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);

    drive_hsa_runtime_loaded();

    EXPECT_TRUE(m_sdk_wrapper->get_create_buffer_info().empty());
    EXPECT_TRUE(m_sdk_wrapper->get_configure_pc_sampling_info().empty());
}

void TestPcSamplingInput::SetUp()
{
    m_input_parameters = std::make_shared<MockInputParameters>();
    m_sdk_wrapper      = std::make_shared<MockSdkWrapper>();
    m_counters_writer  = std::make_shared<MockCountersWriter>();

    test_knobs::set_input_parameters(m_input_parameters);
    test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    test_knobs::set_csv_writer(m_counters_writer);
}

void TestPcSamplingInput::TearDown()
{
    test_knobs::reset_cfg();
}

tool_data_t* TestPcSamplingInput::get_tool_data(const rocprofiler_tool_configure_result_t* cfg)
{
    return (static_cast<std::unique_ptr<tool_data_t>*>(cfg->tool_data))->get();
}

void TestPcSamplingInput::drive_hsa_runtime_loaded()
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    ASSERT_EQ(m_sdk_wrapper->get_hsa_intercept_registration_info().size(), 1u);
    const auto reg = m_sdk_wrapper->get_hsa_intercept_registration_info()[0];
    reg.callback(ROCPROFILER_HSA_TABLE, 0, 0, nullptr, 0, reg.user_data);
}

uint64_t TestPcSamplingInput::configured_interval_for(const std::string& env_interval,
                                                      uint64_t           min_interval,
                                                      uint64_t           max_interval)
{
    constexpr rocprofiler_agent_id_t agent{101};
    m_input_parameters->set_pc_sampling_beta_enabled("1");
    m_input_parameters->set_pc_sampling_method("host_trap");
    m_input_parameters->set_pc_sampling_interval(env_interval);
    m_sdk_wrapper->set_available_gpu_agents({agent});
    m_sdk_wrapper->set_pc_sampling_config(min_interval,
                                          max_interval,
                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                          ROCPROFILER_PC_SAMPLING_UNIT_CYCLES);
    m_sdk_wrapper->set_configure_pc_sampling_status(ROCPROFILER_STATUS_SUCCESS);
    drive_hsa_runtime_loaded();
    return m_sdk_wrapper->get_configure_pc_sampling_info().at(0).interval;
}
