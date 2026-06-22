// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_sdk_callbacks.h"

#include "fmt/format.h"
#include "rocprofiler_compute_tool.h"

using namespace rocprofiler_compute_tool;

TEST_F(TestSdkCallbacks, ProvidedSameKernelWithMultiplexingDisabled_DispatchCbReturnsFirstPmc)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::DISABLED;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
}

TEST_F(TestSdkCallbacks, ProvidedDifferentKernelsWithMultiplexingDisabled_DispatchCbReturnsFirstPmc)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::DISABLED;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(2, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
}

TEST_F(TestSdkCallbacks, ProvidedSameKernelWithKernelMultiplexing_DispatchCbReturnsEachPmc)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::KERNEL;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_LT(config_index_0, created_config_info.size());
    EXPECT_LT(config_index_1, created_config_info.size());
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedDifferentKernelsWithKernelMultiplexing_DispatchCbReturnsFirstPmc)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::KERNEL;
    const auto config_index_0 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_id(2, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_2 = dispatch_kernel_with_id(1, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_3 = dispatch_kernel_with_id(2, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_2].counter_names, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index_3].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedSameKernelSameParamsWithLaunchMultiplexing_DispatchCbReturnsEachPmc)
{
    m_tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode_t::LAUNCH;
    constexpr kernel_dispatch_info_t info    = {1, 2, {3, 3, 3}, {4, 4, 4}, 5};

    const auto config_index_0 = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    const auto config_index_1 = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(created_config_info[config_index_0].counter_names, m_counters_pmc0);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedSameKernelDifferentParamsWithLaunchMultiplexing_DispatchCbReturnsSamePmc)
{
    m_tool_data->iteration_multiplexing_mode   = iteration_multiplexing_mode_t::LAUNCH;
    kernel_dispatch_info_t info                = {1, 2, {3, 3, 3}, {4, 4, 4}, 5};
    const auto&            created_config_info = m_sdk_wrapper->get_create_counter_config_info();

    auto config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.kernel_id++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.queue_id++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.workgroup_size.x++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.grid_size.x++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);

    info.LDS_memory_size++;
    config_index = dispatch_kernel_with_dispatch_info(info, m_counters_pmc0, m_counters_pmc1);
    EXPECT_EQ(created_config_info[config_index].counter_names, m_counters_pmc0);
}

TEST_P(TestSdkCallbacksMultiplexing, DISABLED_ProvidedCountersNotAvailable_DispatchCbReturnsNoConfig)
{
    // FIXME: This test currently disabled because current implementation of dispatch_callback
    // tries to create counters config even if there are no counters available for it.
    m_tool_data->requested_counters = convert_counters_per_pmc_to_str({m_counters_pmc0, m_counters_pmc1});

    constexpr rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    rocprofiler_counter_config_id_t                        config{m_invalid_config_id};
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);

    EXPECT_EQ(config.handle, m_invalid_config_id);
}

TEST_P(TestSdkCallbacksMultiplexing, ProvidedKernelIdsOfInterest_DispatchCbReturnsResultForThemOnly)
{
    m_tool_data->target_kernel_ids.insert(2);

    m_tool_data->iteration_multiplexing_mode = m_multiplexing_mode;
    const auto config_index_0                = dispatch_kernel_with_id(1, m_counters_pmc0);
    const auto config_index_1                = dispatch_kernel_with_id(2, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(config_index_0, m_invalid_config_id);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_P(TestSdkCallbacksMultiplexing, ProvidedKernelDispatchRanges_DispatchCbReturnsResultForThemOnly)
{
    m_tool_data->kernel_filter_ranges.emplace_back(2, 2);

    m_tool_data->iteration_multiplexing_mode = m_multiplexing_mode;
    const auto config_index_0                = dispatch_kernel_with_id(1, m_counters_pmc0);
    const auto config_index_1                = dispatch_kernel_with_id(1, m_counters_pmc1);

    const auto& created_config_info = m_sdk_wrapper->get_create_counter_config_info();
    EXPECT_EQ(config_index_0, m_invalid_config_id);
    EXPECT_EQ(created_config_info[config_index_1].counter_names, m_counters_pmc1);
}

TEST_F(TestSdkCallbacks, ProvidedCounterRecord_RecordCbReturnsCorrectData)
{
    constexpr uint64_t counter_id    = 10;
    const std::string  counter_name  = "counter10";
    constexpr double   counter_value = 11.;

    invoke_record_callback(counter_id, counter_name, counter_value);

    EXPECT_EQ(m_tool_data->counter_records[0].counter_id, counter_id);
    EXPECT_EQ(m_tool_data->counter_records[0].counter_name, counter_name);
    EXPECT_EQ(m_tool_data->counter_records[0].counter_value, counter_value);
}

TEST_F(TestSdkCallbacks, ProvidedTracingRecord_ToolTracingCbReturnsKernelIdsFromIt)
{
    constexpr uint64_t kernel_id_0 = 10;
    constexpr uint64_t kernel_id_1 = 20;

    invoke_tool_tracing_callback(kernel_id_0);
    invoke_tool_tracing_callback(kernel_id_1);

    EXPECT_EQ(m_tool_data->target_kernel_ids.size(), 2);
    EXPECT_EQ(*m_tool_data->target_kernel_ids.cbegin(), kernel_id_0);
    EXPECT_EQ(*(++m_tool_data->target_kernel_ids.cbegin()), kernel_id_1);
}

TEST_F(TestSdkCallbacks, ProvidedCodeObjectLoadWithPcSamplingEnabled_ForwardsToCollector)
{
    auto collector = std::make_shared<MockPcSamplingCollector>();
    m_tool_data->pc_sampling = pc_sampling_feature_t{PcSamplingMode::HostTrap, "unused.json", collector};

    rocprofiler_callback_tracing_record_t                record  = {};
    rocprofiler_callback_tracing_code_object_load_data_t payload = {};
    record.phase                                                 = ROCPROFILER_CALLBACK_PHASE_LOAD;
    record.kind      = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    record.operation = ROCPROFILER_CODE_OBJECT_LOAD;
    record.payload   = &payload;

    m_sdk_callbacks->tool_tracing_callback(record, &m_tool_data);

    EXPECT_EQ(collector->load_count, 1);
}

TEST_P(TestSdkCallbacksKernelFiltering, ProvidedKernelFilteringEnabled_ReturnsKernelIdsOnlyForMathing)
{
    constexpr uint64_t kernel_id_0           = 10;
    constexpr uint64_t kernel_id_1           = 20;
    const std::string  target_kernel_name    = m_filtering_params.mangled_kernel_name;
    const std ::string nontarget_kernel_name = "non_matching";

    m_tool_data->kernel_filter_include_regex = m_filtering_params.kernel_regex;
    invoke_tool_tracing_callback(kernel_id_0, target_kernel_name);
    invoke_tool_tracing_callback(kernel_id_1, nontarget_kernel_name);

    EXPECT_EQ(m_tool_data->target_kernel_ids.size(), 1);
    EXPECT_EQ(*m_tool_data->target_kernel_ids.cbegin(), kernel_id_0);
}

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacks
void TestSdkCallbacks::SetUp()
{
    m_sdk_wrapper = std::make_shared<MockSdkWrapper>();
    test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    m_sdk_callbacks = std::make_shared<SdkCallbacksImpl>(m_sdk_wrapper);
    m_tool_data     = std::make_unique<tool_data_t>();
}

uint64_t TestSdkCallbacks::dispatch_kernel_with_dispatch_info(const kernel_dispatch_info_t& dispatch_info,
                                                              const std::vector<std::string>& counters_pmc0,
                                                              const std::vector<std::string>& counters_pmc1)
{
    m_tool_data->requested_counters = convert_counters_per_pmc_to_str({counters_pmc0, counters_pmc1});
    m_sdk_wrapper->set_available_counters(concat_counters(counters_pmc0, counters_pmc1));

    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.kernel_id                      = dispatch_info.kernel_id;
    dispatch_data.dispatch_info.queue_id.handle                = dispatch_info.queue_id;
    dispatch_data.dispatch_info.workgroup_size                 = dispatch_info.workgroup_size;
    dispatch_data.dispatch_info.grid_size                      = dispatch_info.grid_size;
    dispatch_data.dispatch_info.group_segment_size             = dispatch_info.LDS_memory_size;
    dispatch_data.dispatch_info.agent_id.handle                = 0xff;
    rocprofiler_counter_config_id_t config{m_invalid_config_id};
    m_sdk_callbacks->dispatch_callback(dispatch_data, &config, &m_tool_data);
    return config.handle;
}

uint64_t TestSdkCallbacks::dispatch_kernel_with_id(uint64_t                        kernel_id,
                                                   const std::vector<std::string>& counters_pmc0,
                                                   const std::vector<std::string>& counters_pmc1)
{
    kernel_dispatch_info_t dispatch_info = {};
    dispatch_info.kernel_id              = kernel_id;
    return dispatch_kernel_with_dispatch_info(dispatch_info, counters_pmc0, counters_pmc1);
}

void TestSdkCallbacks::invoke_record_callback(uint64_t           counter_id,
                                              const std::string& counter_name,
                                              double             counter_value)
{
    rocprofiler_dispatch_counting_service_data_t dispatch_data = {};
    dispatch_data.dispatch_info.dispatch_id                    = 100;
    dispatch_data.dispatch_info.agent_id.handle                = 200;
    dispatch_data.dispatch_info.kernel_id                      = 300;
    dispatch_data.dispatch_info.group_segment_size             = 400;

    std::vector<rocprofiler_counter_record_t> record_data(2);
    record_data[0].counter_value = counter_value;
    record_data[0].id            = counter_id;

    m_tool_data->counter_id_name_map[counter_id] = counter_name;

    m_sdk_callbacks->record_callback(dispatch_data, record_data.data(), record_data.size(), &m_tool_data);

    const auto query_record_info = m_sdk_wrapper->get_query_counter_record_info();
    EXPECT_EQ(m_tool_data->counter_records.size(), record_data.size());
    EXPECT_EQ(m_tool_data->counter_records.size(), query_record_info.size());
    EXPECT_EQ(m_tool_data->counter_records[0].dispatch_id, dispatch_data.dispatch_info.dispatch_id);
    EXPECT_EQ(m_tool_data->counter_records[0].agent_id, dispatch_data.dispatch_info.agent_id.handle);
    EXPECT_EQ(m_tool_data->counter_records[0].kernel_id, dispatch_data.dispatch_info.kernel_id);
    EXPECT_EQ(m_tool_data->counter_records[0].LDS_memory_size,
              dispatch_data.dispatch_info.group_segment_size);
}

void TestSdkCallbacks::invoke_tool_tracing_callback(uint64_t kernel_id, const std::string& kernel_name)
{
    rocprofiler_callback_tracing_record_t                                  record  = {};
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t payload = {};
    record.phase     = ROCPROFILER_CALLBACK_PHASE_LOAD;
    record.kind      = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    record.operation = ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    record.payload   = &payload;

    payload.kernel_id   = kernel_id;
    payload.kernel_name = kernel_name.c_str();
    m_sdk_callbacks->tool_tracing_callback(record, &m_tool_data);
}

std::string TestSdkCallbacks::convert_counters_per_pmc_to_str(
    const std::vector<std::vector<std::string>>& counters_per_pmc)
{
    std::string result;
    for (const auto& counters : counters_per_pmc)
    {
        result += convert_counters_to_str(counters);
        result += ",";
    }
    return remove_trailing_comma(result);
}

std::string TestSdkCallbacks::convert_counters_to_str(const std::vector<std::string>& counters)
{
    if (counters.empty())
        return "";

    std::string result = "pmc: ";
    for (const auto& counter : counters)
    {
        result += fmt::format("{} ", counter);
    }
    return result;
}

std::string TestSdkCallbacks::remove_trailing_comma(const std::string& str)
{
    std::string result = str;
    if (!result.empty() && result.back() == ',')
    {
        result.pop_back();
    }
    return result;
}

std::vector<std::string> TestSdkCallbacks::concat_counters(const std::vector<std::string>& v0,
                                                           const std::vector<std::string>& v1)
{
    auto result = v0;
    result.insert(result.end(), v1.begin(), v1.end());
    return result;
}

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacksMultiplexing
void TestSdkCallbacksMultiplexing::SetUp()
{
    TestSdkCallbacks::SetUp();
    m_multiplexing_mode = GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    Multiplexing,
    TestSdkCallbacksMultiplexing,
    ::testing::Values(rocprofiler_compute_tool::iteration_multiplexing_mode_t::DISABLED,
                      rocprofiler_compute_tool::iteration_multiplexing_mode_t::KERNEL,
                      rocprofiler_compute_tool::iteration_multiplexing_mode_t::LAUNCH));

//////////////////////////////////////////////////////////////////////////
/// TestSdkCallbacksKernelFiltering
void TestSdkCallbacksKernelFiltering::SetUp()
{
    TestSdkCallbacks::SetUp();
    m_filtering_params = GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    KernelFiltering,
    TestSdkCallbacksKernelFiltering,
    ::testing::Values(
        kernel_filtering_test_params_t{"_Z10my_kernel", "my_kernel()", ".*my_kernel.*"},
        kernel_filtering_test_params_t{"_ZN3hip12vector_add_1Ev", "hip::vector_add_1()", ".*vector_add.*"},
        kernel_filtering_test_params_t{"_Z6kernelIiEvv", "void kernel<int>()", ".*kernel.*"},
        kernel_filtering_test_params_t{
            "_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_15gpu_kernel_implIZZZNS0_31direct_"
            "copy_"
            "kernel_cuda_gpu_nuERKNS_10TensorIterEEENKUlvE_clEvEUlvE_EEvS5_T_EUlfE_EEvS5_SB_",
            "void at::native::elementwise_kernel<128, 4, "
            "at::native::gpu_kernel_impl<direct_copy_kernel_cuda_gpu_nu(at::TensorIter "
            "const&)::{lambda()#1}::operator()() const::{lambda()#1}>(at::TensorIter const&, "
            "{lambda()#1})::{lambda(float)#1}>(at::TensorIter const&, {lambda(float)#1})",
            ".*elementwise_kernel.*"}));
