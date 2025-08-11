// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef TEST_PGEN_TEST_PGEN_SPM_H_
#define TEST_PGEN_TEST_PGEN_SPM_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "pgen/test_pgen.h"
#include "util/test_assert.h"

typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> callback_data_t;

hsa_status_t TestPGenSpmCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                 hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                 void* callback_data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  reinterpret_cast<callback_data_t*>(callback_data)->push_back(*info_data);
  return status;
}

// Class implements SPM profiling
class TestPGenSpm : public TestPGen {
 public:
  explicit TestPGenSpm(TestAql* t) : TestPGen(t) {
    std::clog << "Test: PGen SPM" << std::endl;
    profile_ = hsa_ven_amd_aqlprofile_profile_t{};
  }

  bool Initialize(int arg_cnt, char** arg_list) {
    std::vector<hsa_ven_amd_aqlprofile_event_t> event_vec;

    for (int i = 0; i < arg_cnt; ++i) {
      unsigned block_id = 0;
      unsigned block_index = 0;
      unsigned event_id = 0;
      sscanf(arg_list[i], "%u:%u:%u", &block_id, &block_index, &event_id);
      const hsa_ven_amd_aqlprofile_event_t event = {
          static_cast<hsa_ven_amd_aqlprofile_block_name_t>(block_id), block_index, event_id};
      event_vec.push_back(event);
    }

    if (!TestPMgr::Initialize(arg_cnt, arg_list)) return false;

    hsa_status_t status;
    hsa_agent_t agent;
    uint32_t command_buffer_alignment;
    uint32_t command_buffer_size;
    uint32_t output_buffer_alignment;
    uint32_t output_buffer_size;

    // GPU identificator
    agent = GetAgentInfo()->dev_id;

    // Preparing events vector
    std::vector<hsa_ven_amd_aqlprofile_event_t> event_vec_filtered;
    for (auto it = event_vec.begin(); it != event_vec.end(); ++it) {
      bool result = false;
      hsa_status_t status = api_->hsa_ven_amd_aqlprofile_validate_event(agent, &(*it), &result);
      if (status != HSA_STATUS_SUCCESS) {
        const char* str = "";
        api_->hsa_ven_amd_aqlprofile_error_string(&str);
        std::cerr << "aqlprofile err: " << str << std::endl;
      }
      if (!result) {
        std::cerr << "Bad event: block (" << it->block_name << "_" << it->block_index << ") id ("
                  << it->counter_id << ")" << std::endl;
      } else {
        event_vec_filtered.push_back(*it);
      }
    }
    const size_t event_count = event_vec_filtered.size();
    hsa_ven_amd_aqlprofile_event_t* events = new hsa_ven_amd_aqlprofile_event_t[event_count];
    for (uint32_t i = 0; i < event_count; ++i) {
      events[i] = event_vec_filtered.at(i);
    }

    // Initialization of the profile
    memset(&profile_, 0, sizeof(profile_));
    profile_.agent = agent;
    profile_.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE;

    // Set sample rate parameter
    hsa_ven_amd_aqlprofile_parameter_t parameter;
    parameter.parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE;
    parameter.value = spm_sample_rate_;
    profile_.parameters = &parameter;
    profile_.parameter_count = 1;

    // Set enabled events list
    profile_.events = events;
    profile_.event_count = event_count;

    // Profile buffers attributes
    command_buffer_alignment = buffer_alignment_;
    status = api_->hsa_ven_amd_aqlprofile_get_info(
        &profile_, HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE, &command_buffer_size);
    TEST_ASSERT(status == HSA_STATUS_SUCCESS);

    output_buffer_alignment = buffer_alignment_;
    output_buffer_size = buffer_size_;

    // Application is allocating the command buffer
    // AllocateSystem(command_buffer_alignment, command_buffer_size,
    //                MODE_HOST_ACC|MODE_DEV_ACC|MODE_EXEC_DATA)
    profile_.command_buffer.ptr =
        GetRsrcFactory()->AllocateCmdMemory(GetAgentInfo(), command_buffer_size);
    TEST_ASSERT(profile_.command_buffer.ptr != NULL);
    profile_.command_buffer.size = command_buffer_size;
    TEST_ASSERT((reinterpret_cast<uintptr_t>(profile_.command_buffer.ptr) &
                 (command_buffer_alignment - 1)) == 0);

    // Application is allocating the output buffer
    // AllocateLocal(output_buffer_alignment, output_buffer_size,
    //               MODE_DEV_ACC)
    profile_.output_buffer.ptr =
        GetRsrcFactory()->AllocateLocalMemory(GetAgentInfo(), output_buffer_size);
    TEST_ASSERT(profile_.output_buffer.ptr != NULL);
    profile_.output_buffer.size = output_buffer_size;
    TEST_ASSERT((reinterpret_cast<uintptr_t>(profile_.output_buffer.ptr) &
                 (output_buffer_alignment - 1)) == 0);

    // Populating the AQL start packet
    status = api_->hsa_ven_amd_aqlprofile_start(&profile_, PrePacket());
    TEST_ASSERT(status == HSA_STATUS_SUCCESS);
    if (status != HSA_STATUS_SUCCESS) return false;

    // Populating the AQL stop packet
    status = api_->hsa_ven_amd_aqlprofile_stop(&profile_, PostPacket());
    TEST_ASSERT(status == HSA_STATUS_SUCCESS);

    return (status == HSA_STATUS_SUCCESS);
  }

  int GetMode() { return RUN_MODE; }
  bool BuildPackets() { return true; }

  bool DumpData() {
    std::clog << "TestPGenSpm::DumpData :" << std::endl;

    callback_data_t data;
    api_->hsa_ven_amd_aqlprofile_iterate_data(&profile_, TestPGenSpmCallback, &data);
    for (callback_data_t::iterator it = data.begin(); it != data.end(); ++it) {
      std::cout << "sample(" << std::dec << it->sample_id << ") size(" << std::dec
                << it->trace_data.size << ") ptr(" << std::hex << it->trace_data.ptr << ")"
                << std::endl;

      void* sys_buf = GetRsrcFactory()->AllocateSysMemory(GetAgentInfo(), it->trace_data.size);
      TEST_ASSERT(sys_buf != NULL);
      if (sys_buf == NULL) return false;

      hsa_status_t status = hsa_memory_copy(sys_buf, it->trace_data.ptr, it->trace_data.size);
      TEST_ASSERT(status == HSA_STATUS_SUCCESS);
      if (status != HSA_STATUS_SUCCESS) return false;

      std::string file_name;
      file_name.append("spm_dump_");
      file_name.append(std::to_string(it->sample_id));
      file_name.append(".txt");
      std::ofstream out_file;
      out_file.open(file_name);

      // Write the buffer in terms of shorts (16 bits)
      uint16_t* trace_data = reinterpret_cast<uint16_t*>(sys_buf);
      for (unsigned i = 0; i < (it->trace_data.size / sizeof(uint16_t)); ++i) {
        out_file << std::setw(4) << std::setfill('0') << std::hex << trace_data[i] << "\n";
      }

      out_file.close();
    }

    return true;
  }

  static const uint32_t buffer_alignment_ = 0x1000;  // 4K
  static const uint32_t buffer_size_ = 0x2000000;    // 32M
  static const uint32_t spm_sample_rate_ = 10000;    // default SPM sample rate

  hsa_ven_amd_aqlprofile_profile_t profile_;
};

#endif  // TEST_PGEN_TEST_PGEN_SPM_H_
