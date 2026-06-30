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


#include "pgen/test_pmgr.h"

#include <atomic>

#include "util/test_assert.h"

bool TestPMgr::AddPacket(const packet_t* packet) {
  GetRsrcFactory()->Submit(GetQueue(), packet);
  return true;
}

bool TestPMgr::AddWaitPacket(packet_t* packet, hsa_signal_t signal) {
  // Set packet completion signal
  packet->completion_signal = signal;

  // Submit Dispatch Aql packet
  bool result = AddPacket(packet);

  // Wait for Dispatch packet to complete
  hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1, HSA_WAIT_STATE_BLOCKED);

  hsa_signal_store_relaxed(signal, 1);

  return result;
}

bool TestPMgr::Setup() {
  // Build Aql Pkts
  const int mode = GetMode();
  if (mode == SETUP_MODE) {
    // Submit Pre-Dispatch Aql packet
    AddWaitPacket(&pre_packet_, packet_signal_);
  }

  Test()->Setup();

  if (mode == SETUP_MODE) {
    // Submit Post-Dispatch Aql packet
    AddWaitPacket(&post_packet_, packet_signal_);

    // Dumping profiling data
    DumpData();
  }

  return true;
}

bool TestPMgr::Run() {
  // Build Aql Pkts
  const int mode = GetMode();
  if (mode == RUN_MODE) {
    // Submit Pre-Dispatch Aql packet
    AddWaitPacket(&pre_packet_, packet_signal_);
  }

  Test()->Run();
  if (getenv("AQLPROFILE_SDMA") != NULL) Test()->RunSdma(0x1000);

  if (mode == RUN_MODE) {
    // Submit Post-Dispatch Aql packet
    AddWaitPacket(&post_packet_, packet_signal_);

    // Dumping profiling data
    DumpData();
  }

  return true;
}

bool TestPMgr::Initialize(int argc, char** argv) {
  TestAql::Initialize(argc, argv);

  hsa_status_t status = hsa_signal_create(1, 0, NULL, &packet_signal_);
  TEST_ASSERT(status == HSA_STATUS_SUCCESS);
  api_ = HsaRsrcFactory::Instance().AqlProfileApi();

  return true;
}

TestPMgr::TestPMgr(TestAql* t) : TestAql(t), api_(NULL) {
  memset(&pre_packet_, 0, sizeof(pre_packet_));
  memset(&post_packet_, 0, sizeof(post_packet_));
  dummy_signal_.handle = 0;
  packet_signal_ = dummy_signal_;
}

TestPMgr::~TestPMgr() {
  if (packet_signal_.handle != 0) {
    hsa_status_t status = hsa_signal_destroy(packet_signal_);
    TEST_ASSERT(status == HSA_STATUS_SUCCESS);
  }
}
