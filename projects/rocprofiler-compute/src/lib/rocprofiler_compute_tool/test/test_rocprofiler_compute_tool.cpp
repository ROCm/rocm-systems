#include "input_parameters.h"
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

    rocprofiler_client_id_t              m_client_id{};
    std::shared_ptr<MockInputParameters> m_input_parameters;
    std::shared_ptr<MockSdkCallbacks>    m_sdk_callbacks;
    std::shared_ptr<MockSdkWrapper>      m_sdk_wrapper;
};

TEST_F(TestRocprofilerComputeTool, TestExample)
{
    rocprofiler_configure(1, "", 1, &m_client_id);
}

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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
