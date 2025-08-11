#include "gtest/gtest.h"
#include "rocm_version.h"

TEST(ROCmVersionTest, GetROCmVersion) {
    unsigned int major, minor, patch;
    EXPECT_EQ(getROCmVersion(&major, &minor, &patch), VerSuccess);
    // Check that the version numbers are as expected
    EXPECT_EQ(major, ROCM_VERSION_MAJOR);
    EXPECT_EQ(minor, ROCM_VERSION_MINOR);
    EXPECT_EQ(patch, ROCM_VERSION_PATCH);
}


TEST(ROCmVersionTest, PrintBuildInfo) {
    // This test assumes that printBuildInfo always returns VerSuccess
    EXPECT_EQ(printBuildInfo(), VerSuccess);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

