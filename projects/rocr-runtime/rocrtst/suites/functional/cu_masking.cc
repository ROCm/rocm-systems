/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2021-2021, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include "suites/functional/cu_masking.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/os.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include <string>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <chrono>

CU_Masking::CU_Masking() : TestBase() {
  std::string name;
  std::string desc;

  name = "CU Masking";
  desc = "This test checks CU masking functionality via hsa_amd_queue_cu_get(set)_mask and HSA_CU_MASK.";

  set_title(name);
  set_description(desc);

  set_kernel_file_name("cu_mask_kernels.hsaco");
}

void CU_Masking::Run() {
  hsa_status_t err;
  TestBase::Run();

  printf("Running %lu iterations\n", RealIterationNum());

  // Random source
  std::mt19937 rand(std::chrono::system_clock::now().time_since_epoch().count());

  // Store cu masking variable
  std::string mask_var;
  char* temp = getenv("HSA_CU_MASK");
  if(temp!=nullptr)
    mask_var = temp;
  unsetenv("HSA_CU_MASK");

  std::string mask_init_var;
  temp = getenv("HSA_CU_MASK_SKIP_INIT");
  if(temp!=nullptr)
    mask_init_var = temp;
  unsetenv("HSA_CU_MASK_SKIP_INIT");

  // Loop over and test all GPUs
  uint32_t idx = 0;
  while(true) {
    Device* gpu;
    CodeObject* obj;
    Kernel kern;

    struct args_t {
      uint32_t* hw_ids;
      OCLHiddenArgs _;
    };
    args_t* args;

    hsa_signal_t signal;
    hsa_queue_t* q;

    uint32_t cu_count;
    uint32_t group_size;
    uint32_t max_grid_size;
    uint32_t threads;

    auto init = [&]() {
      System::Init();
      if(idx == System::gpu().size())
        return false;

      gpu = &System::gpu()[idx];
      std::string filename = rocrtst::LocateKernelFile(kernel_file_name(), gpu->agent);

      obj = new CodeObject(filename, *gpu);

      err = hsa_agent_get_info(gpu->agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cu_count);
      CHECK(err);

      err = hsa_agent_get_info(gpu->agent, (hsa_agent_info_t)HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &group_size);
      CHECK(err);

      err = hsa_agent_get_info(gpu->agent, (hsa_agent_info_t)HSA_AGENT_INFO_GRID_MAX_SIZE, &max_grid_size);
      CHECK(err);

      uint64_t max_threads = uint64_t(cu_count)*group_size*10;
      threads = max_threads < max_grid_size ? max_threads : max_grid_size;
      threads = (threads / group_size) * group_size;

      // All CU enabled check
      if(!obj->GetKernel("get_hw_id", kern)) {
        ADD_FAILURE();
        return false;
      }

      args = (args_t*)hsaMalloc(sizeof(args_t), System::kernarg());
      memset(args, 0, sizeof(args_t));

      args->hw_ids = (uint32_t*)hsaMalloc(sizeof(uint32_t)*threads, System::kernarg());

      err = hsa_signal_create(1, 0, nullptr, &signal);
      CHECK(err);

      err = hsa_queue_create(gpu->agent, 4096, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &q);
      CHECK(err);

      return true;
    };

    auto fini = [&]() {
      err = hsa_queue_destroy(q);
      CHECK(err);
      err = hsa_signal_destroy(signal);
      CHECK(err);
      err = hsa_memory_free(args->hw_ids);
      CHECK(err);
      err = hsa_memory_free(args);
      CHECK(err);
      delete obj;
      gpu = nullptr;
      System::Shutdown();
    };

    auto dispatch = [&]() {
      memset(args->hw_ids, 0, sizeof(uint32_t)*threads);

      Aql pkt = { };
      pkt.header.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
      pkt.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
      pkt.header.release = HSA_FENCE_SCOPE_SYSTEM;
      pkt.dispatch.kernel_object = kern.handle;
      pkt.dispatch.private_segment_size = kern.scratch;
      pkt.dispatch.group_segment_size = kern.group;
      pkt.dispatch.setup = 1;
      pkt.dispatch.workgroup_size_x = group_size;
      pkt.dispatch.workgroup_size_y = 1;
      pkt.dispatch.workgroup_size_z = 1;
      pkt.dispatch.grid_size_x = threads;
      pkt.dispatch.grid_size_y = 1;
      pkt.dispatch.grid_size_z = 1;
      pkt.dispatch.kernarg_address = args;
      pkt.dispatch.completion_signal = signal;

      SubmitPacket(q, pkt);

      hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, -1ull, HSA_WAIT_STATE_BLOCKED);
      hsa_signal_store_relaxed(signal, 1);
    };

    auto getHwIds = [&](std::vector<uint32_t>& ids){
      dispatch();
      std::sort(&args->hw_ids[0], &args->hw_ids[threads]);
      uint32_t* end = std::unique(&args->hw_ids[0], &args->hw_ids[threads]);
      ids.clear();
      ids.insert(ids.begin(), &args->hw_ids[0], end);
    };

    // Check fully unconstrained.
    unsetenv("HSA_CU_MASK_SKIP_INIT");
    setenv("HSA_CU_MASK_SKIP_INIT", "1", 1);

    if(!init())
      break;
    
    {
      char name[64];
      hsa_agent_get_info(gpu->agent, HSA_AGENT_INFO_NAME, name);
      name[63]='\0';
      printf("Testing gpu index %u, %s\n", idx, name);
    }

    std::vector<uint32_t> left, right, isect;

    // Check unconstrained cu set.
    getHwIds(left);
    printf("Expecting %u CUs, found %lu with HSA_CU_MASK_SKIP_INIT.\n", cu_count, left.size());
    ASSERT_EQ(cu_count, left.size());
    fini();
    unsetenv("HSA_CU_MASK_SKIP_INIT");

    // Check fully enabled, but mask used, set.
    setenv("HSA_CU_MASK", (std::to_string(idx)+":0-"+std::to_string(cu_count-1)).c_str(), 1);
    init();
    getHwIds(right);
    printf("Expecting %u CUs, found %lu with HSA_CU_MASK.\n", cu_count, right.size());
    if(cu_count != right.size()) {
      isect.resize(left.size());
      auto isect_end = std::set_difference(left.begin(), left.end(), right.begin(), right.end(), isect.begin());
      isect.resize(isect_end - isect.begin());
      printf("Missing CUs: ");
      for(auto cu : isect)
        printf("%u ", cu);
      printf("\n");
    }
    ASSERT_EQ(cu_count, right.size());
    fini();
    unsetenv("HSA_CU_MASK");

    // Check rocr default mask.
    init();
    getHwIds(right);
    printf("Expecting %u CUs, found %lu.\n", cu_count, right.size());
    if(cu_count != right.size()) {
      isect.resize(left.size());
      auto isect_end = std::set_difference(left.begin(), left.end(), right.begin(), right.end(), isect.begin());
      isect.resize(isect_end - isect.begin());
      printf("Missing CUs: ");
      for(auto cu : isect)
        printf("%u ", cu);
      printf("\n");
    }
    ASSERT_EQ(cu_count, right.size());
    fini();

    std::vector<uint32_t> bits;
    for(uint32_t i=0; i<cu_count; i++)
      bits.push_back(i);
    
    std::vector<uint32_t> bitmask, resultmask;
    uint32_t dwords = (cu_count + 31) / 32;

    bitmask.resize(dwords);
    resultmask.resize(dwords);

    for(size_t iteration=0; iteration<RealIterationNum(); iteration++) {

      auto setBits = [&](uint32_t start, uint32_t stop, std::vector<uint32_t>& array) {
        assert(array.size() == dwords && "Bitmask array has incorrect size.");
        for(uint32_t i=0; i<dwords; i++)
          array[i] = 0;
        for(uint32_t i=start; i<stop; i++) {
          int dword = bits[i] / 32;
          int offset = bits[i] % 32;
          array[dword] |= (1 << offset);
        }
      };

      auto getMasks = [&](uint32_t start, uint32_t stop, std::vector<uint32_t>& hw_ids) {
        setBits(start, stop, bitmask);
        err = hsa_amd_queue_cu_set_mask(q, dwords*32, &bitmask[0]);
        if((err!=HSA_STATUS_SUCCESS) && (err!=(hsa_status_t)HSA_STATUS_CU_MASK_REDUCED))
          CHECK(err);
        err = hsa_amd_queue_cu_get_mask(q, dwords*32, &resultmask[0]);
        CHECK(err);
        getHwIds(hw_ids);
      };

      auto getIsect = [&]() {
        isect.resize(left.size());
        auto isect_end = std::set_intersection(left.begin(), left.end(), right.begin(), right.end(), isect.begin());
        isect.resize(isect_end - isect.begin());
      };

      auto printMask = [](std::vector<uint32_t>& mask) {
        printf("0x");
        for(size_t i=1; i<mask.size()+1; i++)
          printf("%08X", mask[mask.size()-i]);
      };

      auto printMasks = [&]() {
        printf("Set mask: ");
        printMask(bitmask);
        printf("\n");
        printf("Get mask: ");
        printMask(resultmask);
        printf("\n");
      };

      // CU set API check, no overlap
      std::shuffle(bits.begin(), bits.end(), rand);
      uint32_t split_index = (rand() % (cu_count - 2)) + 1;

      init();

      getMasks(0, split_index, left);
      printMasks();
      printf("Observed %lu CUs.\n", left.size());
      for(uint32_t i=0; i<dwords; i++)
        ASSERT_EQ(bitmask[i], resultmask[i]);
      ASSERT_EQ(split_index, left.size());

      getMasks(split_index, cu_count, right);
      printMasks();
      printf("Observed %lu CUs.\n", right.size());
      for(uint32_t i=0; i<dwords; i++)
        ASSERT_EQ(bitmask[i], resultmask[i]);
      ASSERT_EQ(cu_count-split_index, right.size());

      getIsect();
      printf("Overlap of %lu CUs.\n", isect.size());
      ASSERT_EQ(0u, isect.size());
      
      // CU set API check, overlap possible
      uint32_t high_split_index = (rand() % (cu_count - 2)) + 1;

      if(high_split_index < split_index)
        std::swap(high_split_index, split_index);

      getMasks(0, high_split_index, left);
      printMasks();
      printf("Observed %lu CUs.\n", left.size());
      for(uint32_t i=0; i<dwords; i++)
        ASSERT_EQ(bitmask[i], resultmask[i]);
      ASSERT_EQ(high_split_index, left.size());

      getMasks(split_index, cu_count, right);
      printMasks();
      printf("Observed %lu CUs.\n", right.size());
      for(uint32_t i=0; i<dwords; i++)
        ASSERT_EQ(bitmask[i], resultmask[i]);
      ASSERT_EQ(cu_count-split_index, right.size());

      getIsect();
      printf("Overlap of %lu CUs.\n", isect.size());
      ASSERT_EQ(high_split_index - split_index, isect.size());
      
      // HSA_CU_MASK check, default
      fini();
      
      // Pick masking bits for env var
      std::shuffle(bits.begin(), bits.end(), rand);
      uint32_t mask_index = (rand() % (cu_count - 2)) + 1;
      std::vector<uint32_t> env_mask(&bits[0], &bits[mask_index]);

      // Convert to string range syntax
      std::sort(env_mask.begin(), env_mask.end());
      uint32_t start, stop;
      start=stop=env_mask[0];
      std::vector<std::string> ranges;
      // Append invalid bit so that final loop will emit the last range.
      env_mask.push_back(-1);
      for(size_t j=1; j<env_mask.size(); j++) {
        uint32_t index = env_mask[j];
        if(index != stop+1) {
          if(start==stop)
            ranges.push_back(std::to_string(start));
          else
            ranges.push_back(std::to_string(start)+"-"+std::to_string(stop));
          start=stop=index;
        } else {
          stop = index;
        }
      }
      env_mask.pop_back();
      // Shuffle ranges
      std::shuffle(ranges.begin(), ranges.end(), rand);
      // Assemble final env var string.
      std::string env_var = std::to_string(idx) + ":";
      env_var += ranges[0];
      for(uint32_t i=1; i<ranges.size(); i++)
        env_var += ", " + ranges[i];

      // Set env var and check that default queues are masked.
      //env_var = "0:41-44, 104-107, 47-50, 67-68, 77-100, 61, 102, 19-24, 109, 70-75, 52-59, 63-65, 0-17, 27-39";
      setenv("HSA_CU_MASK", env_var.c_str(), 1);
      printf("HSA_CU_MASK = %s\n", env_var.c_str());
      env_mask.clear();
      env_mask.resize(dwords);
      setBits(0, mask_index, env_mask);
      printf("  HSA_CU_MASK => ");
      printMask(env_mask);
      printf("\n");

      init();
      
      getHwIds(left);
      printf("Expecting %u CUs, found %lu\n", mask_index, left.size());
      ASSERT_EQ(left.size(), mask_index);

      // Check that HSA_CU_MASK constrains the API
      // Find at least partially enabled CU mask.
      [&]() {
        while(true) {
          std::shuffle(bits.begin(), bits.end(), rand);
          split_index = (rand() % (cu_count - 2)) + 1;
          setBits(0, split_index, bitmask);
          for(uint32_t i=0; i<dwords; i++) {
            if((bitmask[i] & env_mask[i]) != 0)
              return;
          }
        }
      }();

      getMasks(0, split_index, left);
      printMasks();
      printf("Observed %lu CUs.\n", left.size());
      uint32_t enabledCus = 0;
      for(uint32_t i=0; i<dwords; i++) {
        bitmask[i] &= env_mask[i];
        enabledCus += rocrtst::popcount(bitmask[i]);
        ASSERT_EQ(bitmask[i], resultmask[i]);
      }
      ASSERT_EQ(enabledCus, left.size());
      ASSERT_LE(enabledCus, mask_index);

      fini();
      unsetenv("HSA_CU_MASK");

      // Todo: Hex syntax.  Syntax errors.  Above hw limit bits.

    }
    idx++;
  }

  if(!mask_var.empty())
    setenv("HSA_CU_MASK", mask_var.c_str(), 1);
  if(!mask_init_var.empty())
    setenv("HSA_CU_MASK_SKIP_INIT", mask_var.c_str(), 1);
}
