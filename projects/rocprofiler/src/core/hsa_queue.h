/******************************************************************************
MIT License

Copyright (c) 2018 ROCm Core Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#ifndef _SRC_CORE_HSA_QUEUE_H
#define _SRC_CORE_HSA_QUEUE_H

#include <atomic>

#include "core/queue.h"
#include "core/types.h"

namespace rocprofiler {

class HsaQueue : public Queue {
 public:
  typedef void (HsaQueue::*submit_fptr_t)(const packet_t* packet);
  enum {
    LEGACY_SLOT_SIZE_W = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(packet_word_t),
    LEGACY_SLOT_SIZE_P = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(packet_t)
  };
  struct slot_pm4_t {
    packet_word_t words[LEGACY_SLOT_SIZE_W];
  };

  HsaQueue(const util::AgentInfo* agent_info, hsa_queue_t* queue) : queue_(queue) {}

  void Submit(const packet_t* packet) {
    // Compute the write index of queue and copy Aql packet into it
    const uint64_t que_idx = hsa_queue_load_write_index_relaxed(queue_);
    // Increment the write index
    hsa_queue_store_write_index_relaxed(queue_, que_idx + 1);

    const uint32_t mask = queue_->size - 1;

    // Copy packet to the queue
    const packet_word_t* src = reinterpret_cast<const packet_word_t*>(packet);
    packet_t* slot = reinterpret_cast<packet_t*>(queue_->base_address) + (que_idx & mask);
    packet_word_t* dst = reinterpret_cast<packet_word_t*>(slot);
    const uint32_t nwords = sizeof(packet_t) / sizeof(packet_word_t);
    for (unsigned i = 1; i < nwords; ++i) {
      dst[i] = src[i];
    }

    // To maintain global order to ensure the prior copy of the packet contents is made visible
    // before the header is updated.
    // With in-order CP it will wait until the first packet in the blob will be valid
    std::atomic<packet_word_t>* header_atomic_ptr =
        reinterpret_cast<std::atomic<packet_word_t>*>(&dst[0]);
    header_atomic_ptr->store(src[0], std::memory_order_release);

    // Doorbell signaling
    hsa_signal_store_relaxed(queue_->doorbell_signal, que_idx);
  }

 private:
  hsa_queue_t* queue_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_HSA_QUEUE_H
