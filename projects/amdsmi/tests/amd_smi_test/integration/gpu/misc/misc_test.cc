// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_npm_info (node handle) ----------------
TEST_F(GpuIntegration, GetNpmInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_npm_info", "node=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_npm_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetNpmInfo_InvalidHandle) {
  amdsmi_npm_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_npm_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_npm_info(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetNpmInfo_Node) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_npm_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  // NPM is node-scoped: it needs a node handle, and amdsmi_get_node_handle only
  // resolves one from the OAM ID 0 processor.
  bool resolved_any = false;
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_node_handle node = nullptr;
    if (amdsmi_get_node_handle(gpus()[i], &node) != AMDSMI_STATUS_SUCCESS) continue;
    resolved_any = true;
    amdsmi_npm_info_t info;
    memset(&info, 0, sizeof(info));
    const std::string in = "node from gpu=" + std::to_string(i);
    DISPLAY_AMDSMI_API("amdsmi_get_npm_info", in, kVerbose);
    amdsmi_status_t err = amdsmi_get_npm_info(node, &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive(in, err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
  if (!resolved_any) GTEST_SKIP() << "No node handle on this system";
}
