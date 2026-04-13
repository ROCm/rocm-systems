#include "rocprofiler_compute_tool.h"
#include "input_parameters.h"
#include "mocks.h"

#include <gtest/gtest.h>

class TestRocprofilerComputeTool : public ::testing::Test
{
protected:

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
