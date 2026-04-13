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
        m_mock_parameters    = std::make_shared<MockInputParameters>();
        m_mock_sdk_callbacks = std::make_shared<MockSdkCallbacks>();
        m_mock_sdk_wrapper   = std::make_shared<MockSdkWrapper>();

        test_knobs::set_input_parameters(m_mock_parameters);
        test_knobs::set_sdk_callbacks(m_mock_sdk_callbacks);
        test_knobs::set_sdk_wrapper(m_mock_sdk_wrapper);
    }

    const char*                          non_null_input_parameter = "parameter";
    std::shared_ptr<MockInputParameters> m_mock_parameters;
    std::shared_ptr<MockSdkCallbacks>    m_mock_sdk_callbacks;
    std::shared_ptr<MockSdkWrapper>      m_mock_sdk_wrapper;
};

TEST_F(TestRocprofilerComputeTool, TestExample)
{
    rocprofiler_client_id_t client_id{};
    rocprofiler_configure(1, "", 1, &client_id);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
