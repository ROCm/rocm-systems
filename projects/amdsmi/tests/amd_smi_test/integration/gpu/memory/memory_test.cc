// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

static constexpr amdsmi_memory_type_t kMemTypes[] = {AMDSMI_MEM_TYPE_VRAM, AMDSMI_MEM_TYPE_VIS_VRAM,
                                                     AMDSMI_MEM_TYPE_GTT};

// ---------------- amdsmi_get_gpu_memory_total (enum) ----------------
TEST_F(GpuIntegration, GetMemoryTotal_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_total", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_total(any_gpu(), AMDSMI_MEM_TYPE_VRAM, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMemoryTotal_InvalidHandle) {
  uint64_t total = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_total", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_total(kInvalidHandle, AMDSMI_MEM_TYPE_VRAM, &total);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMemoryTotal_AllGpusAllTypes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_total");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto mt : kMemTypes) {
      uint64_t total = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_total",
                         "gpu=" + std::to_string(i) + " type=" + std::to_string(mt), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_memory_total(gpus()[i], mt, &total);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " type=" + std::to_string(mt), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_memory_usage (enum) ----------------
TEST_F(GpuIntegration, GetMemoryUsage_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_usage", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_usage(any_gpu(), AMDSMI_MEM_TYPE_VRAM, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMemoryUsage_InvalidHandle) {
  uint64_t used = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_usage", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_usage(kInvalidHandle, AMDSMI_MEM_TYPE_VRAM, &used);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMemoryUsage_AllGpusAllTypes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_usage");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto mt : kMemTypes) {
      uint64_t used = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_usage",
                         "gpu=" + std::to_string(i) + " type=" + std::to_string(mt), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_memory_usage(gpus()[i], mt, &used);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " type=" + std::to_string(mt), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_vram_usage ----------------
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetVramUsage, amdsmi_get_gpu_vram_usage, amdsmi_vram_usage_t)

// ---------------- amdsmi_get_gpu_bad_page_info ----------------
TEST_F(GpuIntegration, GetBadPageInfo_NullNumPages) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", "gpu=0 num_pages=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_info(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetBadPageInfo_InvalidHandle) {
  uint32_t num = 16;
  amdsmi_retired_page_record_t recs[16];
  memset(recs, 0, sizeof(recs));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_info(kInvalidHandle, &num, recs);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetBadPageInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_bad_page_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    const std::string in = "gpu=" + std::to_string(i);
    // Two-call pattern per amdsmi.h: size first, then fill a buffer of that
    // size. A fixed array would overflow on a GPU with more bad pages than it
    // holds, because the API reports the true count rather than clamping.
    uint32_t num = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", in + " out=nullptr", kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_bad_page_info(gpus()[i], &num, nullptr);
    if (err == AMDSMI_STATUS_SUCCESS && num > 0) {
      std::vector<amdsmi_retired_page_record_t> recs(num);
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", in, kVerbose);
      err = amdsmi_get_gpu_bad_page_info(gpus()[i], &num, recs.data());
    }
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive(in, err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_bad_page_threshold ----------------
TEST_F(GpuIntegration, GetBadPageThreshold_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_threshold", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_threshold(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetBadPageThreshold_InvalidHandle) {
  uint32_t threshold = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_threshold", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_threshold(kInvalidHandle, &threshold);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetBadPageThreshold_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_bad_page_threshold");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t threshold = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_threshold", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_bad_page_threshold(gpus()[i], &threshold);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_memory_reserved_pages ----------------
TEST_F(GpuIntegration, GetMemoryReservedPages_NullNumPages) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "gpu=0 num_pages=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_reserved_pages(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMemoryReservedPages_InvalidHandle) {
  uint32_t num = 16;
  amdsmi_retired_page_record_t recs[16];
  memset(recs, 0, sizeof(recs));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_reserved_pages(kInvalidHandle, &num, recs);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMemoryReservedPages_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_reserved_pages");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t num = 16;
    amdsmi_retired_page_record_t recs[16];
    memset(recs, 0, sizeof(recs));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_memory_reserved_pages(gpus()[i], &num, recs);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_cache_info ----------------
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetCacheInfo, amdsmi_get_gpu_cache_info,
                                     amdsmi_gpu_cache_info_t)

// ---------------- amdsmi_get_gpu_uma_carveout_info ----------------
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetUmaCarveoutInfo, amdsmi_get_gpu_uma_carveout_info,
                                     amdsmi_uma_carveout_info_t)

// ---------------- amdsmi_get_ttm_info (no handle) ----------------
TEST_F(GpuIntegration, GetTtmInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_ttm_info", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_ttm_info(nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetTtmInfo_Valid) {
  amdsmi_ttm_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_ttm_info", "valid", kVerbose);
  amdsmi_status_t err = amdsmi_get_ttm_info(&info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---------------- amdsmi_set_gpu_uma_carveout (SET) ----------------
TEST_F(GpuIntegration, SetUmaCarveout_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_uma_carveout(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_clean_gpu_local_data (action) ----------------
TEST_F(GpuIntegration, CleanLocalData_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_clean_gpu_local_data", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_clean_gpu_local_data(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
