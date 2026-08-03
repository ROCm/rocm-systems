/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "kernel_config.h"
#include "gtest/gtest.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

namespace RcclUnitTesting {

class IommuPassthroughTest : public ::testing::Test {
protected:
  std::string tempPath;

  void SetUp() override {
    char path[] = "/tmp/rccl-kernel-config-test-XXXXXX";
    int fd = mkstemp(path);
    ASSERT_NE(fd, -1);
    close(fd);
    tempPath = path;
  }

  void TearDown() override {
    if (!tempPath.empty()) {
      std::remove(tempPath.c_str());
    }
  }

  void writeConfig(const std::string& content) {
    std::ofstream out(tempPath);
    ASSERT_TRUE(out.is_open());
    out << content;
  }
};

TEST_F(IommuPassthroughTest, ContentHasOptionOnSingleLine) {
  const std::string content =
      "CONFIG_64BIT=y\n"
      "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y\n"
      "CONFIG_SMP=y\n";
  EXPECT_TRUE(ncclKernelConfigContentHasOption(content, "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y"));
}

TEST_F(IommuPassthroughTest, ContentFindsOptionSplitAcrossReads) {
  // Simulate a token that would be missed by fixed-size fgets buffers.
  std::string content(300, 'x');
  content += "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y";
  EXPECT_TRUE(ncclKernelConfigContentHasOption(content, "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y"));
}

TEST_F(IommuPassthroughTest, ContentMissingOption) {
  const std::string content = "CONFIG_64BIT=y\nCONFIG_SMP=y\n";
  EXPECT_FALSE(ncclKernelConfigContentHasOption(content, "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y"));
}

TEST_F(IommuPassthroughTest, ReadFileFindsOption) {
  writeConfig("CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y\n");
  std::string content;
  ASSERT_TRUE(ncclKernelConfigReadFile(tempPath.c_str(), &content));
  EXPECT_TRUE(ncclKernelConfigContentHasOption(content, "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y"));
}

TEST_F(IommuPassthroughTest, IommuPassthroughOkWithCmdline) {
  EXPECT_TRUE(ncclIommuPassthroughOk("quiet splash iommu=pt amd_iommu=on"));
  EXPECT_TRUE(ncclIommuPassthroughOk("iommu=pt"));
}

TEST_F(IommuPassthroughTest, ReadFileMissingOption) {
  writeConfig("CONFIG_64BIT=y\n");
  std::string content;
  ASSERT_TRUE(ncclKernelConfigReadFile(tempPath.c_str(), &content));
  EXPECT_FALSE(ncclKernelConfigContentHasOption(content, "CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y"));
}

}  // namespace RcclUnitTesting
