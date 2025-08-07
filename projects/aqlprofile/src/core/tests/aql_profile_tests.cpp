// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "core/aql_profile.hpp"
#include "core/pm4_factory.h"
// header for memcpy
#include <cstring>


//#include "core/counter_dimensions.hpp"
//#include "core/include/aql_profile_v2.h"

using namespace aql_profile;
using namespace testing;

namespace aql_profile {
bool Pm4Factory::concurrent_create_mode_ = false;
bool Pm4Factory::spm_kfd_mode_ = false;
}

// Mock classes to simulate Pm4Factory and related functionality
class MockPm4Factory : public Pm4Factory {
public:
  MockPm4Factory() : Pm4Factory(BlockInfoMap(nullptr, 0)) {}
  MOCK_METHOD(const GpuBlockInfo*, GetBlockInfo, (const hsa_ven_amd_aqlprofile_event_t*), (const));
  MOCK_METHOD(bool, IsGFX9, (), (const));
  MOCK_METHOD(bool, IsConcurrent, (), (const));
  MOCK_METHOD(int, GetNumWGPs, (), (const));
  MOCK_METHOD(bool, SpmKfdMode, (), (const));
  MOCK_METHOD(bool, SPISkip, (uint32_t, uint32_t), (const));
};


// Helper to create a mock GpuBlockInfo
GpuBlockInfo* CreateBlockInfo(uint32_t id, uint32_t counter_count, uint32_t attr = 0) {
  auto* info = new GpuBlockInfo();
  info->id = id;
  info->counter_count = counter_count;
  info->attr = attr;
  info->instance_count = 1;
  return info;
}

// Helper to create a profile with specified events
hsa_ven_amd_aqlprofile_profile_t* CreateProfile(const std::vector<hsa_ven_amd_aqlprofile_event_t>& events) {
  auto* profile = new hsa_ven_amd_aqlprofile_profile_t();
  profile->event_count = events.size();
  if (!events.empty()) {
      memcpy(reinterpret_cast<void*>(&profile->events), &events, sizeof(hsa_ven_amd_aqlprofile_event_t));
  } else {
    profile->events = nullptr;
  }
  return profile;
}

void DeleteProfile(hsa_ven_amd_aqlprofile_profile_t* profile) {
  if (profile) {
    delete[] profile->events;
    delete profile;
  }
}

hsa_status_t DefaultTracedataCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                      hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                      void* callback_data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  hsa_ven_amd_aqlprofile_info_data_t* passed_data =
      reinterpret_cast<hsa_ven_amd_aqlprofile_info_data_t*>(callback_data);

  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA) {
    if (info_data->sample_id == passed_data->sample_id) {
      passed_data->trace_data = info_data->trace_data;
      status = HSA_STATUS_INFO_BREAK;
    }
  }

  return status;
}

// Test fixture for CountersVec tests
class CountersVecTest : public Test {
protected:
  void SetUp() override {
    pm4_factory = new NiceMock<MockPm4Factory>();
    ON_CALL(*pm4_factory, IsGFX9()).WillByDefault(Return(true));
  }
  void TearDown() override {
    delete pm4_factory;
  }
  NiceMock<MockPm4Factory>* pm4_factory;

  pm4_builder::counters_vector CountersVec(const profile_t* profile, const Pm4Factory* pm4_factory);
};

pm4_builder::counters_vector CountersVecTest::CountersVec(const profile_t* profile,
                                                       const Pm4Factory* pm4_factory) {
  pm4_builder::counters_vector vec;
  std::map<block_des_t, uint32_t, lt_block_des> index_map;
  for (const hsa_ven_amd_aqlprofile_event_t* p = profile->events;
       p < profile->events + profile->event_count; ++p) {
    const GpuBlockInfo* block_info = pm4_factory->GetBlockInfo(p);
    const block_des_t block_des = {pm4_factory->GetBlockInfo(p)->id, p->block_index};

    // Counting counter register index per block
    const auto ret = index_map.insert({block_des, 0});
    uint32_t& reg_index = ret.first->second;

    if (pm4_builder::SPISkip(block_info->attr, p->counter_id))
    {
      vec.push_back({p->counter_id, reg_index, block_des, block_info});
      continue;
    }

    if (reg_index >= block_info->counter_count) {
      throw event_exception("Event is out of block counter registers number limit, ", *p);
    }

    vec.push_back({p->counter_id, reg_index, block_des, block_info});

    ++reg_index;
  }
  return vec;
}

//Test case: Empty profile (no events)
TEST_F(CountersVecTest, EmptyProfile) {
  auto profile = CreateProfile({});
  auto counters = CountersVec(profile, pm4_factory);
  EXPECT_TRUE(counters.empty());
  DeleteProfile(profile);
}

// Test case: Profile with regular events
TEST_F(CountersVecTest, RegularEvents) {
  GpuBlockInfo* block_info1 = CreateBlockInfo(1, 4);
  GpuBlockInfo* block_info2 = CreateBlockInfo(2, 2);

  std::vector<hsa_ven_amd_aqlprofile_event_t> events = {
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4}
  };

  auto profile = CreateProfile(events);
  EXPECT_NE(profile,nullptr);
  EXPECT_EQ(profile->events->block_index, 0);
  EXPECT_EQ(profile->events->counter_id, 4);

  pm4_factory->GetBlockInfo(profile->events);
  bool is_gfx9 = pm4_factory->IsGFX9();
  EXPECT_TRUE(is_gfx9);
 
}

// Test fixture for the DefaultTracedataCallback function
class DefaultTracedataCallbackTest : public Test {
protected:
  hsa_ven_amd_aqlprofile_info_data_t CreateInfoData(uint32_t sample_id) {
    hsa_ven_amd_aqlprofile_info_data_t data{};
    data.sample_id = sample_id;
    data.trace_data.ptr = reinterpret_cast<void*>(0x1000 + sample_id);
    data.trace_data.size = 0x100 + sample_id;
    return data;
  }
};

//Test case: DefaultTracedataCallback with matching sample IDs
TEST_F(DefaultTracedataCallbackTest, MatchingSampleId) {
  auto info_data = CreateInfoData(42);
  hsa_ven_amd_aqlprofile_info_data_t callback_data{};
  callback_data.sample_id = 42;
  callback_data.trace_data.ptr = nullptr;
  callback_data.trace_data.size = 0;

  hsa_status_t status = DefaultTracedataCallback(
      HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA,
      &info_data,
      &callback_data);

  EXPECT_EQ(status, HSA_STATUS_INFO_BREAK);
  EXPECT_EQ(callback_data.trace_data.ptr, info_data.trace_data.ptr);
  EXPECT_EQ(callback_data.trace_data.size, info_data.trace_data.size);
}

// Test case: DefaultTracedataCallback with non-matching sample IDs
TEST_F(DefaultTracedataCallbackTest, NonMatchingSampleId) {
  auto info_data = CreateInfoData(42);
  hsa_ven_amd_aqlprofile_info_data_t callback_data{};
  callback_data.sample_id = 24;
  void* original_ptr = nullptr;
  size_t original_size = 0;
  callback_data.trace_data.ptr = original_ptr;
  callback_data.trace_data.size = original_size;

  hsa_status_t status = DefaultTracedataCallback(
      HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA,
      &info_data,
      &callback_data);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(callback_data.trace_data.ptr, original_ptr);
  EXPECT_EQ(callback_data.trace_data.size, original_size);
}

// Test case: DefaultTracedataCallback with non-trace info type
TEST_F(DefaultTracedataCallbackTest, NonTraceInfoType) {
  auto info_data = CreateInfoData(42);
  hsa_ven_amd_aqlprofile_info_data_t callback_data{};
  callback_data.sample_id = 42;
  void* original_ptr = nullptr;
  size_t original_size = 0;
  callback_data.trace_data.ptr = original_ptr;
  callback_data.trace_data.size = original_size;

  hsa_status_t status = DefaultTracedataCallback(
      HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA,
      &info_data,
      &callback_data);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(callback_data.trace_data.ptr, original_ptr);
  EXPECT_EQ(callback_data.trace_data.size, original_size);
}
