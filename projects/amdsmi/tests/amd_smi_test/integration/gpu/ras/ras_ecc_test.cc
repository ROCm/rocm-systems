// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

static constexpr amdsmi_gpu_block_t kGpuBlocks[] = {
    AMDSMI_GPU_BLOCK_UMC,   AMDSMI_GPU_BLOCK_SDMA,      AMDSMI_GPU_BLOCK_GFX,
    AMDSMI_GPU_BLOCK_MMHUB, AMDSMI_GPU_BLOCK_ATHUB,     AMDSMI_GPU_BLOCK_PCIE_BIF,
    AMDSMI_GPU_BLOCK_HDP,   AMDSMI_GPU_BLOCK_XGMI_WAFL, AMDSMI_GPU_BLOCK_DF,
    AMDSMI_GPU_BLOCK_SMN,   AMDSMI_GPU_BLOCK_SEM,       AMDSMI_GPU_BLOCK_MP0,
    AMDSMI_GPU_BLOCK_MP1,   AMDSMI_GPU_BLOCK_FUSE,      AMDSMI_GPU_BLOCK_MCA,
    AMDSMI_GPU_BLOCK_VCN,   AMDSMI_GPU_BLOCK_JPEG,      AMDSMI_GPU_BLOCK_IH,
    AMDSMI_GPU_BLOCK_MPIO};

// ---------------- amdsmi_get_gpu_ras_block_features_enabled (enum) ----------------
TEST_F(GpuIntegration, GetRasBlockFeaturesEnabled_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ras_block_features_enabled", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_ras_block_features_enabled(any_gpu(), AMDSMI_GPU_BLOCK_UMC, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetRasBlockFeaturesEnabled_InvalidHandle) {
  amdsmi_ras_err_state_t state;
  memset(&state, 0, sizeof(state));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ras_block_features_enabled", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_ras_block_features_enabled(kInvalidHandle, AMDSMI_GPU_BLOCK_UMC, &state);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetRasBlockFeaturesEnabled_AllGpusAllBlocks) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ras_block_features_enabled");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto blk : kGpuBlocks) {
      amdsmi_ras_err_state_t state;
      memset(&state, 0, sizeof(state));
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_ras_block_features_enabled",
                         "gpu=" + std::to_string(i) + " block=" + std::to_string(blk), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_ras_block_features_enabled(gpus()[i], blk, &state);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " block=" + std::to_string(blk), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_ecc_count (enum) ----------------
TEST_F(GpuIntegration, GetEccCount_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_count", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ecc_count(any_gpu(), AMDSMI_GPU_BLOCK_UMC, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetEccCount_InvalidHandle) {
  amdsmi_error_count_t ec;
  memset(&ec, 0, sizeof(ec));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_count", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ecc_count(kInvalidHandle, AMDSMI_GPU_BLOCK_UMC, &ec);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetEccCount_AllGpusAllBlocks) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ecc_count");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto blk : kGpuBlocks) {
      amdsmi_error_count_t ec;
      memset(&ec, 0, sizeof(ec));
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_count",
                         "gpu=" + std::to_string(i) + " block=" + std::to_string(blk), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_ecc_count(gpus()[i], blk, &ec);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " block=" + std::to_string(blk), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_ecc_status (enum) ----------------
TEST_F(GpuIntegration, GetEccStatus_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_status", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ecc_status(any_gpu(), AMDSMI_GPU_BLOCK_UMC, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetEccStatus_InvalidHandle) {
  amdsmi_ras_err_state_t state;
  memset(&state, 0, sizeof(state));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ecc_status(kInvalidHandle, AMDSMI_GPU_BLOCK_UMC, &state);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetEccStatus_AllGpusAllBlocks) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ecc_status");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto blk : kGpuBlocks) {
      amdsmi_ras_err_state_t state;
      memset(&state, 0, sizeof(state));
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_status",
                         "gpu=" + std::to_string(i) + " block=" + std::to_string(blk), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_ecc_status(gpus()[i], blk, &state);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " block=" + std::to_string(blk), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_ecc_enabled ----------------
TEST_F(GpuIntegration, GetEccEnabled_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_enabled", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ecc_enabled(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetEccEnabled_InvalidHandle) {
  uint64_t blocks = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_enabled", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ecc_enabled(kInvalidHandle, &blocks);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetEccEnabled_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ecc_enabled");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint64_t blocks = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_enabled", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_ecc_enabled(gpus()[i], &blocks);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_total_ecc_count ----------------
TEST_F(GpuIntegration, GetTotalEccCount_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_total_ecc_count", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_total_ecc_count(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetTotalEccCount_InvalidHandle) {
  amdsmi_error_count_t ec;
  memset(&ec, 0, sizeof(ec));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_total_ecc_count", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_total_ecc_count(kInvalidHandle, &ec);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetTotalEccCount_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_total_ecc_count");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_error_count_t ec;
    memset(&ec, 0, sizeof(ec));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_total_ecc_count", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_total_ecc_count(gpus()[i], &ec);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_ras_feature_info ----------------
TEST_F(GpuIntegration, GetRasFeatureInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ras_feature_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ras_feature_info(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetRasFeatureInfo_InvalidHandle) {
  amdsmi_ras_feature_t rf;
  memset(&rf, 0, sizeof(rf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ras_feature_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ras_feature_info(kInvalidHandle, &rf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetRasFeatureInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ras_feature_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_ras_feature_t rf;
    memset(&rf, 0, sizeof(rf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ras_feature_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_ras_feature_info(gpus()[i], &rf);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_gpu_validate_ras_eeprom (action) ----------------
TEST_F(GpuIntegration, ValidateRasEeprom_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_gpu_validate_ras_eeprom", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_validate_ras_eeprom(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, ValidateRasEeprom_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_gpu_validate_ras_eeprom");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_gpu_validate_ras_eeprom", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_gpu_validate_ras_eeprom(gpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_cper_entries ----------------
TEST_F(GpuIntegration, GetCperEntries_NullOutput) {
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_gpu_cper_entries returns OUT_OF_RESOURCES instead of INVAL for null "
         "pointer; library input-validation bug";
  char cper_data[4096];
  memset(cper_data, 0, sizeof(cper_data));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_cper_entries", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries(any_gpu(), 0xFFFFFFFF, cper_data, nullptr,
                                                    nullptr, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetCperEntries_InvalidHandle) {
  char cper_data[4096];
  memset(cper_data, 0, sizeof(cper_data));
  uint64_t buf_size = sizeof(cper_data);
  amdsmi_cper_hdr_t* hdrs[16];
  memset(hdrs, 0, sizeof(hdrs));
  uint64_t entry_count = 16, cursor = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_cper_entries", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries(kInvalidHandle, 0xFFFFFFFF, cper_data,
                                                    &buf_size, hdrs, &entry_count, &cursor);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetCperEntries_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_cper_entries");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char cper_data[4096];
    memset(cper_data, 0, sizeof(cper_data));
    uint64_t buf_size = sizeof(cper_data);
    amdsmi_cper_hdr_t* hdrs[16];
    memset(hdrs, 0, sizeof(hdrs));
    uint64_t entry_count = 16, cursor = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_cper_entries", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_cper_entries(gpus()[i], 0xFFFFFFFF, cper_data, &buf_size,
                                                      hdrs, &entry_count, &cursor);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_afids_from_cper (no handle) ----------------
TEST_F(GpuIntegration, GetAfidsFromCper_NullOutput) {
  char cper_buffer[256];
  memset(cper_buffer, 0, sizeof(cper_buffer));
  DISPLAY_AMDSMI_API("amdsmi_get_afids_from_cper", "out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_afids_from_cper(cper_buffer, sizeof(cper_buffer), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetAfidsFromCper_DummyBuffer) {
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_afids_from_cper returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
         "unknown, under investigation";

  char cper_buffer[256];
  memset(cper_buffer, 0, sizeof(cper_buffer));
  uint64_t afids[16];
  memset(afids, 0, sizeof(afids));
  uint32_t num_afids = 16;
  DISPLAY_AMDSMI_API("amdsmi_get_afids_from_cper", "dummy buffer", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_afids_from_cper(cper_buffer, sizeof(cper_buffer), afids, &num_afids);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_INVAL);
}
