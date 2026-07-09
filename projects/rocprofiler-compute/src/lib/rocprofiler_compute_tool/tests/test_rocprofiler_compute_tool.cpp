// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_rocprofiler_compute_tool.h"

#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <gtest/gtest.h>

using namespace rocprofiler_compute_tool;

TEST_F(TestRocprofilerComputeTool, ProvidedEmptyOutputPath_UsesDefault)
{
    m_input_parameters->set_output_path("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->output_filename.find(EnvInputParameters::kDefaultOutputPath) !=
                std::string::npos);
    EXPECT_TRUE(tool_data->output_filename.find(
                    std::to_string(getpid()) + "_native_counter_collection.csv") != std::string::npos);
}

TEST_F(TestRocprofilerComputeTool, ProvidedPcSamplingMethod_EnablesPcSampling)
{
    m_input_parameters->set_pc_sampling_method("host_trap");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->pc_sampling.enabled());
    EXPECT_EQ(tool_data->pc_sampling.mode(), PcSamplingMode::HostTrap);
}

TEST_F(TestRocprofilerComputeTool, ProvidedNoPcSamplingMethod_DoesNotEnablePcSampling)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_FALSE(tool_data->pc_sampling.enabled());
}

TEST_F(TestRocprofilerComputeTool, ProvidedNoRequestedCounters_Throws)
{
    m_input_parameters->set_requested_counters("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(TestRocprofilerComputeTool, ProvidedEmptyInterationMultiplexingMode_DoesntThrow)
{
    m_input_parameters->set_iteration_multiplexing_mode("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(TestRocprofilerComputeTool, ProvidedEmptyKernelFilterIncludeRegex_DoesntThrow)
{
    m_input_parameters->set_kernel_filter_include_regex("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(TestRocprofilerComputeTool, ProvidedEmptyKernelFilterRange_DoesntThrow)
{
    m_input_parameters->set_kernel_filter_range("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(TestRocprofilerComputeTool, ProvidedUnsetOutputPath_UsesDefault)
{
    m_input_parameters->unset_output_path();
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->output_filename.find(EnvInputParameters::kDefaultOutputPath) !=
                std::string::npos);
    EXPECT_TRUE(tool_data->output_filename.find(
                    std::to_string(getpid()) + "_native_counter_collection.csv") != std::string::npos);
}

TEST_F(TestRocprofilerComputeTool, ProvidedUnsetRequestedCounters_UsesDefault)
{
    m_input_parameters->unset_requested_counters();
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->requested_counters, EnvInputParameters::kDefaultRequestedCounters);
}

TEST_F(TestRocprofilerComputeTool, ProvidedUnsetIterationMultiplexingMode_UsesDefault)
{
    m_input_parameters->unset_iteration_multiplexing_mode();
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode,
              iteration_multiplexing_mode(EnvInputParameters::kDefaultIterationMultiplexingMode));
}

TEST_F(TestRocprofilerComputeTool, ProvidedUnsetKernelFilterIncludeRegex_UsesDefault)
{
    m_input_parameters->unset_kernel_filter_include_regex();
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_include_regex,
              EnvInputParameters::kDefaultKernelFilterIncludeRegex);
}

TEST_F(TestRocprofilerComputeTool, ProvidedUnsetKernelFilterRange_UsesDefault)
{
    m_input_parameters->unset_kernel_filter_range();
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(EnvInputParameters::kDefaultKernelFilterRange.empty());
    EXPECT_TRUE(tool_data->kernel_filter_ranges.empty());
}

TEST_F(TestRocprofilerComputeTool, ProvidedNonEmptyOutputPath_ReturnsItExtended)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->output_filename.find(m_input_parameters->get_output_path()) !=
                std::string::npos);
    EXPECT_TRUE(tool_data->output_filename.find(
                    std::to_string(getpid()) + "_native_counter_collection.csv") != std::string::npos);
}

TEST_F(TestRocprofilerComputeTool, ProvidedRequestedCounters_ReturnsIt)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->requested_counters, m_input_parameters->get_requested_counters());
}

TEST_F(TestRocprofilerComputeTool, ProvidedIncorrectIterationMultiplexingMode_ReturnsDisabled)
{
    m_input_parameters->set_iteration_multiplexing_mode("incorrect");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode, iteration_multiplexing_mode_t::DISABLED);
}

TEST_F(TestRocprofilerComputeTool, ProvidedKernelIterationMultiplexingMode_ReturnsIt)
{
    m_input_parameters->set_iteration_multiplexing_mode("kernel");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode, iteration_multiplexing_mode_t::KERNEL);
}

TEST_F(TestRocprofilerComputeTool, ProvidedKernelLauncParamsIterationMultiplexingMode_ReturnsIt)
{
    m_input_parameters->set_iteration_multiplexing_mode("kernel_launch_params");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode, iteration_multiplexing_mode_t::LAUNCH);
}

TEST_F(TestRocprofilerComputeTool, ProvidedKernelFilterIncludeRegex_ReturnsIt)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_include_regex,
              m_input_parameters->get_kernel_filter_include_regex());
}

TEST_F(TestRocprofilerComputeTool, ProvidedIncorrectKernelFilterRange_ReturnsEmpty)
{
    m_input_parameters->set_kernel_filter_range("invalid");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->kernel_filter_ranges.empty());
}

TEST_F(TestRocprofilerComputeTool, ProvidedSingleRangeWithSquareBrackets_ReturnsRangeWithoutBrackets)
{
    m_input_parameters->set_kernel_filter_range("[4]");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_ranges.size(), 1);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].first, 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].second, 4);
}

TEST_F(TestRocprofilerComputeTool, ProvidedSingleRangeWithoutSquareBrackets_ReturnsRange)
{
    m_input_parameters->set_kernel_filter_range("4");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_ranges.size(), 1);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].first, 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].second, 4);
}

TEST_F(TestRocprofilerComputeTool, ProvidedMixOfRanges_ReturnsThem)
{
    m_input_parameters->set_kernel_filter_range("4, 10-11, 12-23, 5");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_ranges.size(), 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].first, 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].second, 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[1].first, 10);
    EXPECT_EQ(tool_data->kernel_filter_ranges[1].second, 11);
    EXPECT_EQ(tool_data->kernel_filter_ranges[2].first, 12);
    EXPECT_EQ(tool_data->kernel_filter_ranges[2].second, 23);
    EXPECT_EQ(tool_data->kernel_filter_ranges[3].first, 5);
    EXPECT_EQ(tool_data->kernel_filter_ranges[3].second, 5);
}

TEST_F(TestRocprofilerComputeTool, DISABLED_ProvidedInvalidRangeWithEndSmallerStart_Throws)
{
    m_input_parameters->set_kernel_filter_range("10-5");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(TestRocprofilerComputeTool, DISABLED_ProvidedIncompleteRange_Throws)
{
    m_input_parameters->set_kernel_filter_range("-5");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
    m_input_parameters->set_kernel_filter_range("5-");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(TestRocprofilerComputeTool, DISABLED_ProvidedIntersectingRanges_Throws)
{
    m_input_parameters->set_kernel_filter_range("2-5, 3-6");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(TestRocprofilerComputeTool, OnToolInit_CreatesContext)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    cfg->initialize(nullptr, cfg->tool_data);
    EXPECT_EQ(m_sdk_wrapper->get_created_contexts().size(), 1u);
}

TEST_F(TestRocprofilerComputeTool, OnToolInit_ConfiguresDispatchCountingService)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    ASSERT_EQ(m_sdk_wrapper->get_dispatch_counting_service_info().size(), 1u);
    const auto& args = m_sdk_wrapper->get_dispatch_counting_service_info()[0];
    EXPECT_EQ(args.context, m_sdk_wrapper->get_created_contexts()[0]);
    EXPECT_TRUE(args.dispatch_callback != nullptr);
    EXPECT_TRUE(args.dispatch_callback_args != nullptr);
    EXPECT_TRUE(args.record_callback != nullptr);
    EXPECT_TRUE(args.record_callback_args != nullptr);
}

TEST_F(TestRocprofilerComputeTool, OnFiniEmptyCounterRecords_DoesntWriteCounters)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    cfg->finalize(cfg->tool_data);
    EXPECT_EQ(m_counters_writer->get_write_counters_info().size(), 0);
}

TEST_F(TestRocprofilerComputeTool, OnFiniWithNonEmptyCounterRecords_WritesCounters)
{
    const auto         cfg        = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto         tool_data  = get_tool_data(cfg);
    constexpr uint64_t counter_id = 20;
    constexpr uint64_t kernel_id  = 11;
    tool_data->counter_records.push_back(create_counter_record(counter_id, kernel_id));
    cfg->finalize(cfg->tool_data);
    EXPECT_EQ(m_counters_writer->get_write_counters_info().size(), 1);
    EXPECT_EQ(m_counters_writer->get_write_counters_info()[0].counter_ids, std::vector{counter_id});
}

TEST_F(TestRocprofilerComputeTool, OnFiniWithNonEmptyCountersAndKernelFiltering_WriteOnlyFilteredCounters)
{
    const auto         cfg        = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto         tool_data  = get_tool_data(cfg);
    constexpr uint64_t counter_id = 20;
    constexpr uint64_t kernel_id0 = 11;
    constexpr uint64_t kernel_id1 = 22;
    tool_data->counter_records.push_back(create_counter_record(counter_id, kernel_id0));
    tool_data->counter_records.push_back(create_counter_record(counter_id, kernel_id1));
    tool_data->target_kernel_ids.insert(kernel_id0);
    cfg->finalize(cfg->tool_data);
    EXPECT_EQ(m_counters_writer->get_write_counters_info().size(), 1);
    EXPECT_EQ(m_counters_writer->get_write_counters_info()[0].counter_ids, std::vector{counter_id});
    EXPECT_EQ(m_counters_writer->get_write_counters_info()[0].kernel_id, std::vector{kernel_id0});
}

// Counter intent is declared in tool_init so the SDK queue-interposition gate
// sees it before HSA loads. The HSA-touching start_context must stay deferred
// to the intercept callback (regression guard for the fork deadlock when the
// tool is preloaded into non-GPU processes).
TEST_F(TestRocprofilerComputeTool, ToolInit_ConfiguresCountingServiceButDoesNotStartContext)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    EXPECT_EQ(m_sdk_wrapper->get_dispatch_counting_service_info().size(), 1u);
    EXPECT_TRUE(m_sdk_wrapper->get_started_contexts().empty());
}

// Without requested counters there is nothing to collect, so tool_init must not
// configure the counting service.
TEST_F(TestRocprofilerComputeTool, ToolInit_WithoutRequestedCounters_DoesNotConfigureCountingService)
{
    m_input_parameters->set_requested_counters("");
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    EXPECT_TRUE(m_sdk_wrapper->get_dispatch_counting_service_info().empty());
    EXPECT_TRUE(m_sdk_wrapper->get_started_contexts().empty());
}

TEST_F(TestRocprofilerComputeTool, RocprofilerConfigure_RegistersHsaInterceptCallback)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(m_sdk_wrapper->get_hsa_intercept_registration_info().size(), 1u);
    EXPECT_NE(m_sdk_wrapper->get_hsa_intercept_registration_info()[0].callback, nullptr);
    EXPECT_EQ(m_sdk_wrapper->get_hsa_intercept_registration_info()[0].user_data, cfg->tool_data);
}

TEST_F(TestRocprofilerComputeTool, HsaInterceptCallback_StartsContext)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    ASSERT_EQ(m_sdk_wrapper->get_hsa_intercept_registration_info().size(), 1u);
    const auto reg = m_sdk_wrapper->get_hsa_intercept_registration_info()[0];
    reg.callback(ROCPROFILER_HSA_TABLE, 0, 0, nullptr, 0, reg.user_data);
    EXPECT_EQ(m_sdk_wrapper->get_started_contexts().size(), 1u);
    compare_counter_config_ids(m_sdk_wrapper->get_created_contexts(),
                               m_sdk_wrapper->get_started_contexts());
}

// The SDK may fire the HSA intercept callback more than once. Context start
// must run exactly once per process.
TEST_F(TestRocprofilerComputeTool, HsaInterceptCallback_IsIdempotent)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    ASSERT_EQ(m_sdk_wrapper->get_hsa_intercept_registration_info().size(), 1u);
    const auto reg = m_sdk_wrapper->get_hsa_intercept_registration_info()[0];
    reg.callback(ROCPROFILER_HSA_TABLE, 0, 0, nullptr, 0, reg.user_data);
    reg.callback(ROCPROFILER_HSA_TABLE, 0, 0, nullptr, 0, reg.user_data);
    EXPECT_EQ(m_sdk_wrapper->get_started_contexts().size(), 1u);
}

// If HSA loads after tool_fini has run, the callback must not start a context
// or dereference the freed tool_data pointer it captured at registration time.
TEST_F(TestRocprofilerComputeTool, HsaInterceptCallback_AfterToolFini_IsNoOp)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    ASSERT_EQ(cfg->initialize(nullptr, cfg->tool_data), 0);
    ASSERT_EQ(m_sdk_wrapper->get_hsa_intercept_registration_info().size(), 1u);
    cfg->finalize(cfg->tool_data);
    const auto reg = m_sdk_wrapper->get_hsa_intercept_registration_info()[0];
    reg.callback(ROCPROFILER_HSA_TABLE, 0, 0, nullptr, 0, reg.user_data);
    EXPECT_TRUE(m_sdk_wrapper->get_started_contexts().empty());
}

//////////////////////////////////////////////////////////////////////////
/// TestRocprofilerComputeTool
void TestRocprofilerComputeTool::SetUp()
{
    m_input_parameters = std::make_shared<MockInputParameters>();
    m_sdk_wrapper      = std::make_shared<MockSdkWrapper>();
    m_counters_writer  = std::make_shared<MockCountersWriter>();

    test_knobs::set_input_parameters(m_input_parameters);
    test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    test_knobs::set_csv_writer(m_counters_writer);
}

void TestRocprofilerComputeTool::TearDown()
{
    test_knobs::reset_cfg();
}

tool_data_t* TestRocprofilerComputeTool::get_tool_data(const rocprofiler_tool_configure_result_t* cfg)
{
    return (static_cast<std::unique_ptr<tool_data_t>*>(cfg->tool_data))->get();
}

void TestRocprofilerComputeTool::compare_counter_config_ids(const std::vector<uint64_t>& expected,
                                                            const std::vector<uint64_t>& actual)
{
    EXPECT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << "Counter config ID at index " << i << " does not match";
    }
}

counter_info_record_t TestRocprofilerComputeTool::create_counter_record(uint64_t counter_id,
                                                                        uint64_t kernel_id)
{
    counter_info_record_t record = {};
    record.counter_id            = counter_id;
    record.kernel_id             = kernel_id;
    return record;
}
