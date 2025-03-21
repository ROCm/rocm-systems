////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_aie_aql_queue.h"
#include "core/inc/amd_xdna_driver.h"

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#endif

#include <cstring>

#include "core/inc/amd_xdna_driver.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {

AieAqlQueue::AieAqlQueue(AieAgent *agent, size_t req_size_pkts,
                         uint32_t node_id)
    : Queue(0, 0), LocalSignal(0, false), DoorbellSignal(signal()),
      agent_(*agent), active_(false) {
  if (agent_.device_type() != core::Agent::DeviceType::kAmdAieDevice) {
    throw AMD::hsa_exception(
        HSA_STATUS_ERROR_INVALID_AGENT,
        "Attempting to create an AIE queue on a non-AIE agent.");
  }
  queue_size_bytes_ = req_size_pkts * sizeof(core::AqlPacket);
  ring_buf_ = agent_.system_allocator()(queue_size_bytes_, 4096,
                                        core::MemoryRegion::AllocateNoFlags);

  if (!ring_buf_) {
    throw AMD::hsa_exception(
        HSA_STATUS_ERROR_INVALID_QUEUE_CREATION,
        "Could not allocate a ring buffer for an AIE queue.");
  }

  // Populate hsa_queue_t fields.
  amd_queue_.hsa_queue.type = HSA_QUEUE_TYPE_SINGLE;
  amd_queue_.hsa_queue.id = INVALID_QUEUEID;
  amd_queue_.hsa_queue.doorbell_signal = Signal::Convert(this);
  amd_queue_.hsa_queue.size = req_size_pkts;
  amd_queue_.hsa_queue.base_address = ring_buf_;
  // Populate AMD queue fields.
  amd_queue_.write_dispatch_id = 0;
  amd_queue_.read_dispatch_id = 0;

  signal_.hardware_doorbell_ptr = nullptr;
  signal_.kind = AMD_SIGNAL_KIND_DOORBELL;
  signal_.queue_ptr = &amd_queue_;
  active_ = true;

  auto &drv = static_cast<XdnaDriver &>(agent_.driver());
  drv.CreateQueue(*this);
}

AieAqlQueue::~AieAqlQueue() {
  AieAqlQueue::Inactivate();
  if (ring_buf_) {
    agent_.system_deallocator()(ring_buf_);
  }
}

hsa_status_t AieAqlQueue::Inactivate() {
  bool active(active_.exchange(false, std::memory_order_relaxed));
  hsa_status_t status(HSA_STATUS_SUCCESS);

  if (active) {
    auto &drv = static_cast<XdnaDriver &>(agent_.driver());
    status = drv.DestroyQueue(*this);
    hw_ctx_handle_ = std::numeric_limits<uint32_t>::max();
  }

  return status;
}

hsa_status_t AieAqlQueue::SetPriority(HSA_QUEUE_PRIORITY priority) {
  return HSA_STATUS_SUCCESS;
}

void AieAqlQueue::Destroy() { delete this; }

// Atomic Reads/Writes
uint64_t AieAqlQueue::LoadReadIndexRelaxed() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieAqlQueue::LoadReadIndexAcquire() {
  return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
}

uint64_t AieAqlQueue::LoadWriteIndexRelaxed() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
}

uint64_t AieAqlQueue::LoadWriteIndexAcquire() {
  return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
}

void AieAqlQueue::StoreWriteIndexRelaxed(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_relaxed);
}

void AieAqlQueue::StoreWriteIndexRelease(uint64_t value) {
  atomic::Store(&amd_queue_.write_dispatch_id, value,
                std::memory_order_release);
}

uint64_t AieAqlQueue::CasWriteIndexRelaxed(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_relaxed);
}

uint64_t AieAqlQueue::CasWriteIndexAcquire(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acquire);
}

uint64_t AieAqlQueue::CasWriteIndexRelease(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_release);
}

uint64_t AieAqlQueue::CasWriteIndexAcqRel(uint64_t expected, uint64_t value) {
  return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected,
                     std::memory_order_acq_rel);
}

uint64_t AieAqlQueue::AddWriteIndexRelaxed(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_relaxed);
}

uint64_t AieAqlQueue::AddWriteIndexAcquire(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acquire);
}

uint64_t AieAqlQueue::AddWriteIndexRelease(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_release);
}

uint64_t AieAqlQueue::AddWriteIndexAcqRel(uint64_t value) {
  return atomic::Add(&amd_queue_.write_dispatch_id, value,
                     std::memory_order_acq_rel);
}

void AieAqlQueue::StoreRelaxed(hsa_signal_value_t value) {
  SubmitCmd(amd_queue_.hsa_queue.base_address, amd_queue_.read_dispatch_id,
            amd_queue_.write_dispatch_id);
}

hsa_status_t AieAqlQueue::SubmitCmd(void* queue_base, uint64_t read_dispatch_id,
                                    uint64_t write_dispatch_id) {
  auto& driver = static_cast<XdnaDriver&>(agent_.driver());
  uint64_t cur_id = read_dispatch_id;
  while (cur_id < write_dispatch_id) {
    hsa_amd_aie_ert_packet_t* pkt = static_cast<hsa_amd_aie_ert_packet_t*>(queue_base) + cur_id;

    // Get the packet header information
    if (pkt->header.header != HSA_PACKET_TYPE_VENDOR_SPECIFIC ||
        pkt->header.AmdFormat != HSA_AMD_PACKET_TYPE_AIE_ERT)
      return HSA_STATUS_ERROR;

    // Get the payload information
    switch (pkt->opcode) {
      case HSA_AMD_AIE_ERT_START_CU: {
        // Iterating over future packets and seeing how many contiguous HSA_AMD_AIE_ERT_START_CU
        // packets there are. All can be combined into a single chain.
        int num_cont_start_cu_pkts = 1;
        int num_operands = 0;
        for (int peak_pkt_id = cur_id + 1; peak_pkt_id < write_dispatch_id; peak_pkt_id++) {
          hsa_amd_aie_ert_packet_t* peak_pkt =
              static_cast<hsa_amd_aie_ert_packet_t*>(queue_base) + peak_pkt_id;
          if (peak_pkt->opcode != HSA_AMD_AIE_ERT_START_CU) {
            break;
          }
          num_operands += GetOperandCount(peak_pkt->count);
          num_cont_start_cu_pkts++;
        }

        // Call into the driver to submit from cur_id to write_dispatch_id
        hsa_status_t status =
            driver.SubmitCmdChain(pkt, num_cont_start_cu_pkts, num_operands, hw_ctx_handle_);
        if (status != HSA_STATUS_SUCCESS) {
          return status;
        }

        cur_id += num_cont_start_cu_pkts;
        break;
      }
      default: {
        return HSA_STATUS_ERROR;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

void AieAqlQueue::StoreRelease(hsa_signal_value_t value) {
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}

hsa_status_t AieAqlQueue::GetInfo(hsa_queue_info_attribute_t attribute,
                                  void *value) {
  switch (attribute) {
    case HSA_AMD_QUEUE_INFO_AGENT:
      *static_cast<hsa_agent_t*>(value) = agent_.public_handle();
      break;
    case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
      // Hardware doorbell supports AQL semantics.
      *static_cast<uint64_t*>(value) = reinterpret_cast<uint64_t>(signal_.hardware_doorbell_ptr);
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieAqlQueue::GetCUMasking(uint32_t num_cu_mask_count,
                                       uint32_t *cu_mask) {
  assert(false && "AIE AQL queue does not support CU masking.");
  return HSA_STATUS_ERROR;
}

hsa_status_t AieAqlQueue::SetCUMasking(uint32_t num_cu_mask_count,
                                       const uint32_t *cu_mask) {
  assert(false && "AIE AQL queue does not support CU masking.");
  return HSA_STATUS_ERROR;
}

void AieAqlQueue::ExecutePM4(uint32_t *cmd_data, size_t cmd_size_b,
                             hsa_fence_scope_t acquireFence,
                             hsa_fence_scope_t releaseFence,
                             hsa_signal_t *signal) {
  assert(false && "AIE AQL queue does not support PM4 packets.");
}

} // namespace AMD
} // namespace rocr
