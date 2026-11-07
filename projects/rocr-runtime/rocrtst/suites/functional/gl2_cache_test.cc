/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <sstream>
#include <vector>

#include "suites/functional/gl2_cache_test.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

static const char kSubTestSeparator[] = "  **************************";

static void PrintSubtestHeader(const char *header) {
  std::cout << "  *** GL2 Cache Subtest: " << header << " ***" << std::endl;
}

GL2CacheTest::GL2CacheTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("  *** GL2 Persisting Cache Tests ***");
  set_description("  *** Tests GL2 persisting L2 cache residency control APIs ***");
}

GL2CacheTest::~GL2CacheTest(void) {
}

void GL2CacheTest::SetUp(void) {
  TestBase::SetUp();
  if (test_skipped_) return;
  std::cout << "  *** Initialize ROCr Runtime for GL2 cache tests ***"
            << std::endl;
}

void GL2CacheTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
  if (subtest_) {
    subtest_();
  }
}

void GL2CacheTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void GL2CacheTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  std::cout << std::endl;
  for (const auto& result : resultList_) {
    std::cout << result << std::endl;
  }
}

void GL2CacheTest::Close() {
  TestBase::Close();
}

void GL2CacheTest::QueryMaxPersistingCacheSize() {
  hsa_status_t err;
  PrintSubtestHeader("Query Max Persisting L2 Cache Size");

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    size_t maxSize = 0;
    err = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
        &maxSize);

    std::stringstream ss;
    if (err != HSA_STATUS_SUCCESS) {
      ss << "    GPU[" << idx
         << "] Max persisting L2 cache query returned error (status="
         << err << ") — feature not supported on this kernel";
      resultList_.push_back(ss.str());
      std::cout << ss.str() << std::endl;
      continue;
    }

    ss << "    GPU[" << idx << "] Max persisting L2 cache size: "
       << maxSize << " bytes";
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;
  }

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}

void GL2CacheTest::QueryRequestPersistingCacheSize() {
  hsa_status_t err;
  PrintSubtestHeader("Query Requested Persisting L2 Cache Size (default)");

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    size_t requestedSize = 0xDEADBEEF;
    err = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE,
        &requestedSize);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    std::stringstream ss;
    ss << "    GPU[" << idx << "] Default requested persisting L2 cache size: "
       << requestedSize << " bytes";
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;
  }

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}

void GL2CacheTest::SetPersistingCacheSize() {
  hsa_status_t err;
  PrintSubtestHeader("Set Persisting L2 Cache Size");

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    size_t maxSize = 0;
    err = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
        &maxSize);

    if (err != HSA_STATUS_SUCCESS || maxSize == 0) {
      std::cout << "    GPU[" << idx
                << "] does not support persisting L2 cache, skipping set test"
                << std::endl;
      continue;
    }

    size_t setSize = maxSize / 2;
    err = hsa_amd_agent_set_attribute(gpus[idx],
        HSA_AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE,
        &setSize);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    std::stringstream ss;
    ss << "    GPU[" << idx << "] Set persisting L2 cache size to "
       << setSize << " bytes (max=" << maxSize << ")";
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;
  }

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}

void GL2CacheTest::SetAndReadBackPersistingCacheSize() {
  hsa_status_t err;
  PrintSubtestHeader("Set and Read-Back Persisting L2 Cache Size");

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    size_t maxSize = 0;
    err = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
        &maxSize);

    if (err != HSA_STATUS_SUCCESS || maxSize == 0) {
      std::cout << "    GPU[" << idx
                << "] does not support persisting L2 cache, skipping"
                << std::endl;
      continue;
    }

    size_t setSize = maxSize / 4;
    err = hsa_amd_agent_set_attribute(gpus[idx],
        HSA_AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE,
        &setSize);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    size_t readBack = 0;
    err = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE,
        &readBack);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    ASSERT_EQ(readBack, setSize);

    std::stringstream ss;
    ss << "    GPU[" << idx << "] Set " << setSize
       << " bytes, read back " << readBack << " bytes - MATCH";
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;
  }

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}

void GL2CacheTest::NegativeInvalidSize() {
  hsa_status_t err;
  PrintSubtestHeader("Negative: Set Invalid (oversized) Cache Size");

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    size_t maxSize = 0;
    err = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
        &maxSize);

    if (err != HSA_STATUS_SUCCESS || maxSize == 0) {
      std::cout << "    GPU[" << idx
                << "] does not support persisting L2 cache, skipping"
                << std::endl;
      continue;
    }

    size_t overSize = maxSize + 1;
    err = hsa_amd_agent_set_attribute(gpus[idx],
        HSA_AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE,
        &overSize);
    ASSERT_NE(err, HSA_STATUS_SUCCESS);

    std::stringstream ss;
    ss << "    GPU[" << idx << "] Oversized request (" << overSize
       << " > max " << maxSize << ") correctly rejected";
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;
  }

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}

void GL2CacheTest::QueryStatusConsistency() {
  hsa_status_t err;
  PrintSubtestHeader("Query Status Consistency (max vs current)");

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  for (uint32_t idx = 0; idx < gpus.size(); ++idx) {
    size_t maxSize = 0;
    hsa_status_t maxErr = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
        &maxSize);

    size_t currentSize = 0;
    hsa_status_t currentErr = hsa_agent_get_info(gpus[idx],
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE,
        &currentSize);

    std::stringstream ss;

    if (maxErr != HSA_STATUS_SUCCESS && currentErr != HSA_STATUS_SUCCESS) {
      // Both failed: unsupported hardware — consistent
      ss << "    GPU[" << idx
         << "] Both queries failed (max status=" << maxErr
         << ", current status=" << currentErr
         << ") — consistent (unsupported HW)";
      resultList_.push_back(ss.str());
      std::cout << ss.str() << std::endl;
      continue;
    }

    if (maxErr == HSA_STATUS_SUCCESS) {
      // If max query succeeds, the value must not be zero. A zero max under
      // SUCCESS would let SetAgentAttribute reject every request silently
      // (review comment #6).
      ASSERT_GT(maxSize, 0u) << "GPU[" << idx
          << "] max query returned SUCCESS but maxSize is 0";
      ss << "    GPU[" << idx << "] Max query: status=SUCCESS, maxSize="
         << maxSize << " bytes (non-zero verified)";
    } else {
      ss << "    GPU[" << idx << "] Max query: status=" << maxErr
         << " (error propagated, not masked as 0)";
    }
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;

    ss.str("");
    if (currentErr == HSA_STATUS_SUCCESS) {
      // Current size can legitimately be 0 (nothing set yet), but status
      // must be SUCCESS only if the kernel actually supports the query.
      ss << "    GPU[" << idx << "] Current query: status=SUCCESS, currentSize="
         << currentSize << " bytes";
    } else {
      ss << "    GPU[" << idx << "] Current query: status=" << currentErr
         << " (error propagated)";
    }
    resultList_.push_back(ss.str());
    std::cout << ss.str() << std::endl;

    // If max succeeded but current failed (or vice versa), that's fine —
    // each field is independently queried. But if max succeeded, verify
    // that set+readback also works.
    if (maxErr == HSA_STATUS_SUCCESS && currentErr == HSA_STATUS_SUCCESS) {
      size_t setSize = maxSize / 2;
      err = hsa_amd_agent_set_attribute(gpus[idx],
          HSA_AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE,
          &setSize);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      size_t readBack = 0;
      err = hsa_agent_get_info(gpus[idx],
          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE,
          &readBack);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      ASSERT_EQ(readBack, setSize);

      ss.str("");
      ss << "    GPU[" << idx << "] Set " << setSize
         << ", read back " << readBack << " — consistency verified";
      resultList_.push_back(ss.str());
      std::cout << ss.str() << std::endl;
    }
  }

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}

void GL2CacheTest::NegativeCPUAgent() {
  hsa_status_t err;
  PrintSubtestHeader("Negative: Query/Set on CPU Agent");

  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(cpus.size(), 0u);

  size_t value = 0;
  err = hsa_agent_get_info(cpus[0],
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE,
      &value);
  ASSERT_NE(err, HSA_STATUS_SUCCESS);

  std::stringstream ss;
  ss << "    CPU agent query correctly rejected with status " << err;
  resultList_.push_back(ss.str());
  std::cout << ss.str() << std::endl;

  std::cout << "    Subtest finished" << std::endl;
  std::cout << kSubTestSeparator << std::endl;
}
