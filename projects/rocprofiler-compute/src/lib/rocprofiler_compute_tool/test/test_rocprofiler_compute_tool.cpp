#include "mocks.h"
#include "rocprofiler_compute_tool.h"

#include <gtest/gtest.h>

using namespace rocprofiler_compute_tool;

class TestRocprofilerComputeTool : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_input_parameters = std::make_shared<MockInputParameters>();
        m_sdk_callbacks    = std::make_shared<MockSdkCallbacks>();
        m_sdk_wrapper      = std::make_shared<MockSdkWrapper>();

        test_knobs::set_input_parameters(m_input_parameters);
        test_knobs::set_sdk_callbacks(m_sdk_callbacks);
        test_knobs::set_sdk_wrapper(m_sdk_wrapper);
    }

    void TearDown() override { test_knobs::reset_cfg(); }

    static tool_data_t* get_tool_data(const rocprofiler_tool_configure_result_t* cfg)
    {
        return (static_cast<std::unique_ptr<tool_data_t>*>(cfg->tool_data))->get();
    }

    rocprofiler_client_id_t              m_client_id{};
    std::shared_ptr<MockInputParameters> m_input_parameters;
    std::shared_ptr<MockSdkCallbacks>    m_sdk_callbacks;
    std::shared_ptr<MockSdkWrapper>      m_sdk_wrapper;
};

TEST_F(TestRocprofilerComputeTool, ProvidedEmptyOutputPath_Throws)
{
    m_input_parameters->set_output_path("");
    EXPECT_THROW(rocprofiler_configure(1, "", 1, &m_client_id), std::runtime_error);
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
