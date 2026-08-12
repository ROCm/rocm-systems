/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstring>
#include <string>

#include "unit/unit_test_framework.h"

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

static constexpr amdsmi_memory_type_t kMemTypes[] = {AMDSMI_MEM_TYPE_VRAM, AMDSMI_MEM_TYPE_VIS_VRAM,
                                                     AMDSMI_MEM_TYPE_GTT};

// ---------------- amdsmi_get_gpu_memory_total (enum) ----------------
TEST_F(GpuUnit, GetMemoryTotal_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_total", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_total(gpus()[0], AMDSMI_MEM_TYPE_VRAM, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetMemoryTotal_InvalidHandle) {
  uint64_t total = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_total", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_total(kInvalidHandle, AMDSMI_MEM_TYPE_VRAM, &total);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMemoryTotal_AllGpusAllTypes) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_total");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto mt : kMemTypes) {
      uint64_t total = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_total",
                         "gpu=" + std::to_string(i) + " type=" + std::to_string(mt), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_memory_total(gpus()[i], mt, &total);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i) + " type=" + std::to_string(mt), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_memory_usage (enum) ----------------
TEST_F(GpuUnit, GetMemoryUsage_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_usage", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_usage(gpus()[0], AMDSMI_MEM_TYPE_VRAM, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetMemoryUsage_InvalidHandle) {
  uint64_t used = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_usage", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_usage(kInvalidHandle, AMDSMI_MEM_TYPE_VRAM, &used);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMemoryUsage_AllGpusAllTypes) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_usage");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto mt : kMemTypes) {
      uint64_t used = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_usage",
                         "gpu=" + std::to_string(i) + " type=" + std::to_string(mt), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_memory_usage(gpus()[i], mt, &used);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i) + " type=" + std::to_string(mt), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_vram_usage ----------------
TEST_F(GpuUnit, GetVramUsage_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_usage", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_vram_usage(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetVramUsage_InvalidHandle) {
  amdsmi_vram_usage_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_usage", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_vram_usage(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetVramUsage_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_vram_usage");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_vram_usage_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_usage", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_vram_usage(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_bad_page_info ----------------
TEST_F(GpuUnit, GetBadPageInfo_NullNumPages) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", "gpu=0 num_pages=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_info(gpus()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetBadPageInfo_InvalidHandle) {
  uint32_t num = 16;
  amdsmi_retired_page_record_t recs[16];
  memset(recs, 0, sizeof(recs));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_info(kInvalidHandle, &num, recs);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetBadPageInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_bad_page_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t num = 16;
    amdsmi_retired_page_record_t recs[16];
    memset(recs, 0, sizeof(recs));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_bad_page_info(gpus()[i], &num, recs);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_bad_page_threshold ----------------
TEST_F(GpuUnit, GetBadPageThreshold_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_threshold", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_threshold(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetBadPageThreshold_InvalidHandle) {
  uint32_t threshold = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_threshold", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_bad_page_threshold(kInvalidHandle, &threshold);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetBadPageThreshold_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_bad_page_threshold");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t threshold = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bad_page_threshold", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_bad_page_threshold(gpus()[i], &threshold);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_memory_reserved_pages ----------------
TEST_F(GpuUnit, GetMemoryReservedPages_NullNumPages) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "gpu=0 num_pages=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_reserved_pages(gpus()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetMemoryReservedPages_InvalidHandle) {
  uint32_t num = 16;
  amdsmi_retired_page_record_t recs[16];
  memset(recs, 0, sizeof(recs));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_reserved_pages(kInvalidHandle, &num, recs);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMemoryReservedPages_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_reserved_pages");
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
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_cache_info ----------------
TEST_F(GpuUnit, GetCacheInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_cache_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_cache_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetCacheInfo_InvalidHandle) {
  amdsmi_gpu_cache_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_cache_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_cache_info(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetCacheInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_cache_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_gpu_cache_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_cache_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_cache_info(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_uma_carveout_info ----------------
TEST_F(GpuUnit, GetUmaCarveoutInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_uma_carveout_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_uma_carveout_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetUmaCarveoutInfo_InvalidHandle) {
  amdsmi_uma_carveout_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_uma_carveout_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_uma_carveout_info(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetUmaCarveoutInfo_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_uma_carveout_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_uma_carveout_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_uma_carveout_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_uma_carveout_info(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_ttm_info (no handle) ----------------
TEST_F(GpuUnit, GetTtmInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_ttm_info", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_ttm_info(nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetTtmInfo_Valid) {
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
TEST_F(GpuUnit, SetUmaCarveout_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_uma_carveout(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetUmaCarveout_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_uma_carveout");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_uma_carveout(gpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(
        "gpu=" + std::to_string(i), err,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_clean_gpu_local_data (action) ----------------
TEST_F(GpuUnit, CleanLocalData_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_clean_gpu_local_data", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_clean_gpu_local_data(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, CleanLocalData_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_clean_gpu_local_data");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_clean_gpu_local_data", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_clean_gpu_local_data(gpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_set_ttm_pages_limit / reset (no handle, SET) ----------------
TEST_F(GpuUnit, SetTtmPagesLimit_Valid) {
  amdsmi_ttm_info_t info;
  memset(&info, 0, sizeof(info));
  if (amdsmi_get_ttm_info(&info) != AMDSMI_STATUS_SUCCESS) GTEST_SKIP() << "TTM info not available";
  DISPLAY_AMDSMI_API("amdsmi_set_ttm_pages_limit", "restore current", kVerbose);
  amdsmi_status_t err = amdsmi_set_ttm_pages_limit(info.current_pages);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                       AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, ResetTtmPagesLimit_Valid) {
  DISPLAY_AMDSMI_API("amdsmi_reset_ttm_pages_limit", "valid", kVerbose);
  amdsmi_status_t err = amdsmi_reset_ttm_pages_limit();
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_NO_PERM);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM);
}
