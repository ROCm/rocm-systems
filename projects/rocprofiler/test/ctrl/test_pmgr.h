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

#ifndef TEST_CTRL_TEST_PMGR_H_
#define TEST_CTRL_TEST_PMGR_H_

#include <hsa.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <atomic>

#include "ctrl/test_aql.h"

// Class implements profiling manager
class TestPMgr : public TestAql {
 public:
  typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
  explicit TestPMgr(TestAql* t);
  bool Run();

 protected:
  packet_t pre_packet_;
  packet_t post_packet_;
  hsa_signal_t dummy_signal_;
  hsa_signal_t post_signal_;

  HsaRsrcFactory::aqlprofile_pfn_t* api_;

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
  bool AddPacketGfx8(const packet_t* packet);
  bool AddPacketGfx9(const packet_t* packet);
};

#endif  // TEST_CTRL_TEST_PMGR_H_
