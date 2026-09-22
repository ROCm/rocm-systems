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


#ifndef TEST_PGEN_TEST_PMGR_H_
#define TEST_PGEN_TEST_PMGR_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <atomic>

#include "ctrl/test_aql.h"

enum Mode { SETUP_MODE, RUN_MODE, UNKNOWN };

// Class implements profiling manager
class TestPMgr : public TestAql {
 public:
  typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
  explicit TestPMgr(TestAql* t);
  ~TestPMgr();
  bool Setup();
  bool Run();

 protected:
  packet_t pre_packet_;
  packet_t post_packet_;
  hsa_signal_t dummy_signal_;
  hsa_signal_t packet_signal_;

  const hsa_ven_amd_aqlprofile_pfn_t* api_;

  virtual int GetMode() { return UNKNOWN; }
  virtual bool BuildPackets() { return false; }
  virtual bool DumpData() { return false; }
  virtual bool Initialize(int argc, char** argv);

 private:
  enum {
    SLOT_PM4_SIZE_DW = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(uint32_t),
    SLOT_PM4_SIZE_AQLP = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(packet_t)
  };
  struct slot_pm4_t {
    uint32_t words[SLOT_PM4_SIZE_DW];
  };

  bool AddPacket(const packet_t* packet);
  bool AddWaitPacket(packet_t* packet, hsa_signal_t signal);
};

#endif  // TEST_PGEN_TEST_PMGR_H_
