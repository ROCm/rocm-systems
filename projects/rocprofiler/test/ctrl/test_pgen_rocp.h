/******************************************************************************

Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

#ifndef TEST_CTRL_TEST_PGEN_ROCP_H_
#define TEST_CTRL_TEST_PGEN_ROCP_H_

#include <list>
#include <vector>

#include "ctrl/test_pgen.h"
#include "util/test_assert.h"

hsa_status_t TestPGenRocpCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                 hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                 void* callback_data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> passed_data_t;
  reinterpret_cast<passed_data_t*>(callback_data)->push_back(*info_data);
  return status;
}

// Class implements PMC profiling
class TestPGenRocp : public TestPGen {
 public:
  explicit TestPGenRocp(TestAql* t) : TestPGen(t) { std::clog << "Test: PGen ROCP" << std::endl; }

  bool Initialize(int /*arg_cnt*/, char** /*arg_list*/) {
    status = rocprofiler_on_dispatch(&profile_, PrePacket(), PostPacket());
    TEST_STATUS(status != HSA_STATUS_SUCCESS);
    return (status == HSA_STATUS_SUCCESS);
  }

 private:
  bool BuildPackets() { return true; }

  bool DumpData() {
    std::clog << "TestPGenRocp::DumpData :" << std::endl;

    typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> callback_data_t;

    callback_data_t data;
    api_.hsa_ven_amd_aqlprofile_iterate_data(&profile_, TestPGenRocpCallback, &data);
    for (callback_data_t::iterator it = data.begin(); it != data.end(); ++it) {
      std::cout << std::dec << "event(block(" << it->pmc_data.event.block_name << "_"
                << it->pmc_data.event.block_index << "), id(" << it->pmc_data.event.counter_id
                << ")), sample(" << it->sample_id << "), result(" << it->pmc_data.result << ")"
                << std::endl;
    }

    return true;
  }
};

#endif  // TEST_CTRL_TEST_PGEN_ROCP_H_
