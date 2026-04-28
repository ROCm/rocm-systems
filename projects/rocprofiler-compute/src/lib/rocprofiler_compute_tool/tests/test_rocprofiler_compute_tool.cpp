// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_rocprofiler_compute_tool.h"

#include "gtest/gtest.h"
#include "mocks.h"
#include "rocprofiler_compute_tool.h"

using namespace rocm_compute;

TEST_F(test_rocprofiler_compute_tool_t, ProvidedEmptyOutputPath_Throws)
{
    m_env_parameters->set_output_path("");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedNoRequestedCounters_Throws)
{
    m_env_parameters->set_requested_counters("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedEmptyInterationMultiplexingMode_DoesntThrow)
{
    m_env_parameters->set_iteration_multiplexing_mode("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedEmptyKernelFilterIncludeRegex_DoesntThrow)
{
    m_env_parameters->set_kernel_filter_include_regex("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedEmptyKernelFilterRange_DoesntThrow)
{
    m_env_parameters->set_kernel_filter_range("");
    EXPECT_NO_THROW(rocprofiler_configure(1, "", 1, &m_client_id));
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedNonEmptyOutputPath_ReturnsItExtended)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->output_filename.find(m_env_parameters->get_output_path()) != std::string::npos);
    EXPECT_TRUE(tool_data->output_filename.find(
                    std::to_string(getpid()) + "_native_counter_collection.csv") != std::string::npos);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedRequestedCounters_ReturnsIt)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->requested_counters, m_env_parameters->get_requested_counters());
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedIncorrectIterationMultiplexingMode_ReturnsDisabled)
{
    m_env_parameters->set_iteration_multiplexing_mode("incorrect");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode, IterationMultiplexingMode::Disabled);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedKernelIterationMultiplexingMode_ReturnsIt)
{
    m_env_parameters->set_iteration_multiplexing_mode("kernel");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode, IterationMultiplexingMode::Kernel);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedKernelLauncParamsIterationMultiplexingMode_ReturnsIt)
{
    m_env_parameters->set_iteration_multiplexing_mode("kernel_launch_params");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->iteration_multiplexing_mode, IterationMultiplexingMode::Launch);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedKernelFilterIncludeRegex_ReturnsIt)
{
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_include_regex,
              m_env_parameters->get_kernel_filter_include_regex());
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedIncorrectKernelFilterRange_ReturnsEmpty)
{
    m_env_parameters->set_kernel_filter_range("invalid");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_TRUE(tool_data->kernel_filter_ranges.empty());
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedSingleRangeWithSquareBrackets_ReturnsRangeWithoutBrackets)
{
    m_env_parameters->set_kernel_filter_range("[4]");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_ranges.size(), 1);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].first, 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].second, 4);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedSingleRangeWithoutSquareBrackets_ReturnsRange)
{
    m_env_parameters->set_kernel_filter_range("4");
    const auto cfg       = rocprofiler_configure(1, "", 1, &m_client_id);
    const auto tool_data = get_tool_data(cfg);
    EXPECT_EQ(tool_data->kernel_filter_ranges.size(), 1);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].first, 4);
    EXPECT_EQ(tool_data->kernel_filter_ranges[0].second, 4);
}

TEST_F(test_rocprofiler_compute_tool_t, ProvidedMixOfRanges_ReturnsThem)
{
    m_env_parameters->set_kernel_filter_range("4, 10-11, 12-23, 5");
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

TEST_F(test_rocprofiler_compute_tool_t, DISABLED_ProvidedInvalidRangeWithEndSmallerStart_Throws)
{
    m_env_parameters->set_kernel_filter_range("10-5");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(test_rocprofiler_compute_tool_t, DISABLED_ProvidedIncompleteRange_Throws)
{
    m_env_parameters->set_kernel_filter_range("-5");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
    m_env_parameters->set_kernel_filter_range("5-");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(test_rocprofiler_compute_tool_t, DISABLED_ProvidedIntersectingRanges_Throws)
{
    m_env_parameters->set_kernel_filter_range("2-5, 3-6");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
}

TEST_F(test_rocprofiler_compute_tool_t, OnToolInit_CreatesAndStartsContext)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    cfg->initialize(nullptr, cfg->tool_data);
    compare_counter_config_ids(m_sdk_wrapper->get_created_contexts(),
                               m_sdk_wrapper->get_started_contexts());
}

TEST_F(test_rocprofiler_compute_tool_t, OnToolInit_ConfiguresDispatchCountingService)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    cfg->initialize(nullptr, cfg->tool_data);
    EXPECT_EQ(m_sdk_wrapper->get_dispatch_counting_service_info().size(), 1);
    const auto& args = m_sdk_wrapper->get_dispatch_counting_service_info()[0];
    EXPECT_EQ(args.context, m_sdk_wrapper->get_created_contexts()[0]);
    EXPECT_TRUE(args.dispatch_callback != nullptr);
    EXPECT_TRUE(args.dispatch_callback_args != nullptr);
    EXPECT_TRUE(args.record_callback != nullptr);
    EXPECT_TRUE(args.record_callback_args != nullptr);
}

TEST_F(test_rocprofiler_compute_tool_t, OnFiniEmptyCounterRecords_DoesntWriteCounters)
{
    const auto cfg = rocprofiler_configure(1, "", 1, &m_client_id);
    cfg->finalize(cfg->tool_data);
    EXPECT_EQ(m_counters_writer->get_write_counters_info().size(), 0);
}

TEST_F(test_rocprofiler_compute_tool_t, OnFiniWithNonEmptyCounterRecords_WritesCounters)
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

TEST_F(test_rocprofiler_compute_tool_t,
       OnFiniWithNonEmptyCountersAndKernelFiltering_WriteOnlyFilteredCounters)
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

TEST_F(test_rocprofiler_compute_tool_t, OnDispatchCallback_ForwardsToSdkCallbacks)
{
    tool_data_t tool_data{};
    tool_data.sdk_callbacks = m_sdk_callbacks;

    rocprofiler_dispatch_counting_service_data_t dispatch_data{};
    dispatch_data.dispatch_info.kernel_id = 42;
    rocprofiler_counter_config_id_t config{};

    dispatch_callback(dispatch_data, &config, nullptr, &tool_data);

    const auto& calls = m_sdk_callbacks->get_dispatch_callback_info();
    EXPECT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].dispatch_data.dispatch_info.kernel_id, dispatch_data.dispatch_info.kernel_id);
    EXPECT_EQ(calls[0].config, &config);
}

TEST_F(test_rocprofiler_compute_tool_t, OnRecordCallback_ForwardsToSdkCallbacks)
{
    rocprofiler_dispatch_counting_service_data_t dispatch_data{};
    dispatch_data.dispatch_info.kernel_id = 7;
    rocprofiler_counter_record_t      record{};
    constexpr size_t                  record_count = 1;
    constexpr rocprofiler_user_data_t user_data{};

    record_callback(dispatch_data, &record, record_count, user_data, &m_tool_data);

    const auto& calls = m_sdk_callbacks->get_record_callback_info();
    EXPECT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].dispatch_data.dispatch_info.kernel_id, dispatch_data.dispatch_info.kernel_id);
    EXPECT_EQ(calls[0].record_data, &record);
    EXPECT_EQ(calls[0].record_count, record_count);
}

TEST_F(test_rocprofiler_compute_tool_t, OnToolTracingCallback_ForwardsToSdkCallbacks)
{
    tool_data_t tool_data{};
    tool_data.sdk_callbacks = m_sdk_callbacks;

    rocprofiler_callback_tracing_record_t record{};
    record.kind = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;

    tool_tracing_callback(m_pc_sampling_record, nullptr, &m_tool_data);

    const auto& calls = m_sdk_callbacks->get_tracing_callback_info();
    EXPECT_EQ(calls.size(), 1);
    EXPECT_EQ(calls[0].record.kind, record.kind);
}

TEST_F(test_rocprofiler_compute_tool_t, OnCodeObjectTracingIfPcSamplingEnabled_ForwardsToSdkCallbacks)
{
    m_payload.code_object_id = 1;
    m_env_parameters->set_pc_sampling_mode("host_trap");
    code_object_tracing_callback(m_pc_sampling_record, nullptr, &m_tool_data);

    m_payload.code_object_id = 2;
    m_env_parameters->set_pc_sampling_mode("stochastic");
    code_object_tracing_callback(m_pc_sampling_record, nullptr, &m_tool_data);

    const auto& calls = m_pc_sampling_collector->get_on_code_object_load_info();
    EXPECT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0].code_object_id, 1);
    EXPECT_EQ(calls[1].code_object_id, 2);
}

TEST_F(test_rocprofiler_compute_tool_t, OnCodeObjectTracingIfPcSamplingDisabled_DoesntForwardToSdkCallbacks)
{
    m_env_parameters->set_pc_sampling_mode("disabled");
    code_object_tracing_callback(m_pc_sampling_record, nullptr, &m_tool_data);

    const auto& calls = m_pc_sampling_collector->get_on_code_object_load_info();
    EXPECT_EQ(calls.size(), 0);
}

//////////////////////////////////////////////////////////////////////////
/// test_rocprofiler_compute_tool_t
void test_rocprofiler_compute_tool_t::SetUp()
{
    m_payload.code_object_id = 1;
    m_pc_sampling_record     = create_code_object_load_info_with_payload(m_payload);

    m_env_parameters        = std::make_shared<mock_env_parameters_t>();
    m_sdk_wrapper           = std::make_shared<mock_sdk_wrapper_t>();
    m_counters_writer       = std::make_shared<mock_counters_writer_t>();
    m_sdk_callbacks         = std::make_shared<mock_sdk_callbacks_t>();
    m_pc_sampling_collector = std::make_shared<mock_pc_sampling_collector_t>();

    m_tool_data.pc_sampling_collector.wlock([&](auto& ptr) { ptr = m_pc_sampling_collector; });
    m_tool_data.sdk_callbacks = m_sdk_callbacks;

    test_knobs::set_env_parameters(m_env_parameters);
    test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    test_knobs::set_csv_writer(m_counters_writer);
}

void test_rocprofiler_compute_tool_t::TearDown()
{
    test_knobs::reset_cfg();
}

tool_data_t* test_rocprofiler_compute_tool_t::get_tool_data(const rocprofiler_tool_configure_result_t* cfg)
{
    return static_cast<tool_data_t*>(cfg->tool_data);
}

void test_rocprofiler_compute_tool_t::compare_counter_config_ids(const std::vector<uint64_t>& expected,
                                                                 const std::vector<uint64_t>& actual)
{
    EXPECT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << "Counter config ID at index " << i << " does not match";
    }
}

counter_info_record_t test_rocprofiler_compute_tool_t::create_counter_record(uint64_t counter_id,
                                                                             uint64_t kernel_id)
{
    counter_info_record_t record = {};
    record.counter_id            = counter_id;
    record.kernel_id             = kernel_id;
    return record;
}

rocprofiler_callback_tracing_record_t test_rocprofiler_compute_tool_t::create_code_object_load_info_with_payload(
    rocprofiler_callback_tracing_code_object_load_data_t& payload)
{
    rocprofiler_callback_tracing_record_t record = {};
    record.kind                                  = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    record.operation                             = ROCPROFILER_CODE_OBJECT_LOAD;
    record.phase                                 = ROCPROFILER_CALLBACK_PHASE_LOAD;
    record.payload                               = &m_payload;

    return record;
}
