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


#ifndef TEST_PGEN_TEST_PGEN_PCSMP_H_
#define TEST_PGEN_TEST_PGEN_PCSMP_H_

#include <cstdint>
#include <iostream>

#include "pgen/test_pgen.h"
#include "pgen/test_pgen_sqtt.h"
#include "util/test_assert.h"

// PC sampling callback data
struct pcsmp_callback_data_t {
  const char* kernel_name;  // sampled kernel name
  void* data_buffer;        // host buffer for tracing data
  uint64_t id;              // sample id
  uint64_t cycle;           // sample cycle
  uint64_t pc;              // sample PC
};

hsa_status_t TestPGenPcsmpCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                   hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                   void* callback_data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  pcsmp_callback_data_t* pcsmp_data = reinterpret_cast<pcsmp_callback_data_t*>(callback_data);
  std::cout << "id(" << std::dec << pcsmp_data->id << ") cycle(" << std::dec << pcsmp_data->cycle
            << ") pc(0x" << std::hex << pcsmp_data->pc << ") name(\"" << pcsmp_data->kernel_name
            << "\")" << std::dec << std::endl
            << std::flush;
  return status;
}

// Class implements SQTT profiling
class TestPGenPcsmp : public TestPGenSqtt {
 public:
  explicit TestPGenPcsmp(TestAql* t) : TestPGenSqtt(t) {
    std::clog << "Test: PGen PC sampling" << std::endl;
  }

  bool DumpData() {
    std::clog << "TestPGenPcsmp::DumpData :" << std::endl;

    TEST_ASSERT(profile_.event_count == 0);
    profile_.event_count = UINT32_MAX;
    pcsmp_callback_data_t data{};
    data.kernel_name = Name();

    // allocate host space
    void* sys_buf = GetRsrcFactory()->AllocateSysMemory(GetAgentInfo(), TestPGenSqtt::buffer_size_);
    TEST_ASSERT(sys_buf != NULL);
    if (sys_buf == NULL) return false;
    data.data_buffer = sys_buf;

    api_->hsa_ven_amd_aqlprofile_iterate_data(&profile_, TestPGenPcsmpCallback, &data);

    return true;
  }
};

#endif  // TEST_PGEN_TEST_PGEN_PCSMP_H_
