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

#include "ctrl/test_pmgr.h"

#include <atomic>

#include "ctrl/test_assert.h"

bool TestPMgr::AddPacketGfx9(const packet_t* packet) {
  packet_t aql_packet = *packet;

  // Compute the write index of queue and copy Aql packet into it
  uint64_t que_idx = hsa_queue_load_write_index_relaxed(GetQueue());
  const uint32_t mask = GetQueue()->size - 1;
  packet_t* slot = (reinterpret_cast<packet_t*>(GetQueue()->base_address)) + (que_idx & mask);

  // Disable packet so that submission to HW is complete
  const auto header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  aql_packet.header &= (~((1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1)) << HSA_PACKET_HEADER_TYPE;
  aql_packet.header |= HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;

  // Copy Aql packet into queue buffer
  *slot = aql_packet;
  // After AQL packet is fully copied into queue buffer
  // update packet header from invalid state to valid state
  auto header_atomic_ptr = reinterpret_cast<std::atomic<uint16_t>*>(&slot->header);
  header_atomic_ptr->store(header, std::memory_order_release);

  // Increment the write index and ring the doorbell to dispatch the kernel.
  hsa_queue_store_write_index_relaxed(GetQueue(), (que_idx + 1));
  hsa_signal_store_relaxed(GetQueue()->doorbell_signal, que_idx);

  return true;
}

bool TestPMgr::AddPacketGfx8(const packet_t* packet) {
  // Create legacy devices PM4 data
  const hsa_ext_amd_aql_pm4_packet_t* aql_packet = (const hsa_ext_amd_aql_pm4_packet_t*)packet;
  slot_pm4_t data;
  api_.hsa_ven_amd_aqlprofile_legacy_get_pm4(aql_packet, reinterpret_cast<void*>(data.words));

  // Compute the write index of queue and copy Aql packet into it
  uint64_t que_idx = hsa_queue_load_write_index_relaxed(GetQueue());
  const uint32_t mask = GetQueue()->size - 1;

  // Copy Aql/Pm4 blob into queue buffer
  packet_t* ptr = (reinterpret_cast<packet_t*>(GetQueue()->base_address)) + (que_idx & mask);
  slot_pm4_t* slot = reinterpret_cast<slot_pm4_t*>(ptr);
  for (unsigned i = 1; i < SLOT_PM4_SIZE_DW; ++i) {
    slot->words[i] = data.words[i];
  }
  // To maintain global order to ensure the prior copy of the packet contents is made visible
  // before the header is updated.
  // With in-order CP it will wait until the first packet in the blob will be valid
  std::atomic<uint32_t>* header_atomic_ptr =
      reinterpret_cast<std::atomic<uint32_t>*>(&slot->words[0]);
  header_atomic_ptr->store(data.words[0], std::memory_order_release);

  // Increment the write index and ring the doorbell to dispatch the kernel.
  que_idx += SLOT_PM4_SIZE_AQLP - 1;
  hsa_queue_store_write_index_relaxed(GetQueue(), (que_idx + 1));
  hsa_signal_store_relaxed(GetQueue()->doorbell_signal, que_idx);

  return true;
}

bool TestPMgr::AddPacket(const packet_t* packet) {
  const char* agent_name = GetAgentInfo()->name;
  return (strncmp(agent_name, "gfx8", 4) == 0) ? AddPacketGfx8(packet) : AddPacketGfx9(packet);
}

bool TestPMgr::Run() {
  // Build Aql Pkts
  const bool active = BuildPackets();
  if (active) {
    // Submit Pre-Dispatch Aql packet
    AddPacket(&pre_packet_);
  }

  Test()->Run();

  if (active) {
    // Set post packet completion signal
    post_packet_.completion_signal = post_signal_;

    // Submit Post-Dispatch Aql packet
    AddPacket(&post_packet_);

    // Wait for Post-Dispatch packet to complete
    hsa_signal_wait_acquire(post_signal_, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                            HSA_WAIT_STATE_BLOCKED);

    // Dumping profiling data
    DumpData();
  }

  return true;
}

bool TestPMgr::Initialize(int argc, char** argv) {
  TestAql::Initialize(argc, argv);

  hsa_status_t status = HSA_STATUS_ERROR;
  status = hsa_signal_create(1, 0, NULL, &post_signal_);
  TEST_ASSERT(status == HSA_STATUS_SUCCESS);
  status = hsa_system_get_extension_table(HSA_EXTENSION_AMD_AQLPROFILE, 1, 0, &api_);
  TEST_ASSERT(status == HSA_STATUS_SUCCESS);

  return true;
}

TestPMgr::TestPMgr(TestAql* t) : TestAql(t), api_({0}) {
  memset(&pre_packet_, 0, sizeof(pre_packet_));
  memset(&post_packet_, 0, sizeof(post_packet_));
  dummy_signal_.handle = 0;
  post_signal_ = dummy_signal_;
  memset(&api_, 0, sizeof(api_));
}
