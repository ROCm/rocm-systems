/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef _SRC_CORE_SIMPLE_PROXY_QUEUE_H
#define _SRC_CORE_SIMPLE_PROXY_QUEUE_H

#include <hsa/hsa.h>
#include <atomic>
#include <map>
#include <mutex>

#include "core/proxy_queue.h"
#include "core/types.h"
#include "util/hsa_rsrc_factory.h"

#ifndef ROCP_PROXY_LOCK
#define ROCP_PROXY_LOCK 1
#endif

namespace rocprofiler {
extern decltype(::hsa_queue_create)* hsa_queue_create_fn;
extern decltype(::hsa_queue_destroy)* hsa_queue_destroy_fn;

extern decltype(::hsa_signal_store_relaxed)* hsa_signal_store_relaxed_fn;
extern decltype(::hsa_signal_store_relaxed)* hsa_signal_store_screlease_fn;

extern decltype(::hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed_fn;
extern decltype(::hsa_queue_store_write_index_relaxed)* hsa_queue_store_write_index_relaxed_fn;
extern decltype(::hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed_fn;

extern decltype(::hsa_queue_load_write_index_scacquire)* hsa_queue_load_write_index_scacquire_fn;
extern decltype(::hsa_queue_store_write_index_screlease)* hsa_queue_store_write_index_screlease_fn;
extern decltype(::hsa_queue_load_read_index_scacquire)* hsa_queue_load_read_index_scacquire_fn;

typedef decltype(hsa_signal_t::handle) signal_handle_t;


class SimpleProxyQueue : public ProxyQueue {
 public:
  static void HsaIntercept(HsaApiTable* table);

  static void SignalStore(hsa_signal_t signal, hsa_signal_value_t que_idx) {
    auto it = queue_map_->find(signal.handle);
    if (it != queue_map_->end()) {
      SimpleProxyQueue* instance = it->second;
      instance->mutex_lock();
      const uint64_t begin = instance->submit_index_;
      const uint64_t end = que_idx + 1;
      instance->submit_index_ = end;
      instance->mutex_unlock();
      for (uint64_t j = begin; j < end; ++j) {
        // Submited packet
        const uint32_t idx = j & instance->queue_mask_;
        packet_t* packet = reinterpret_cast<packet_t*>(instance->queue_->base_address) + idx;
        if (instance->on_submit_cb_ != NULL)
          instance->on_submit_cb_(packet, 1, j, instance->on_submit_cb_data_, NULL);
        else
          instance->Submit(packet);
      }
    } else {
      hsa_signal_store_relaxed_fn(signal, que_idx);
    }
  }

  static uint64_t GetSubmitIndex(const hsa_queue_t* queue) {
    uint64_t index = 0;
    auto it = queue_map_->find(queue->doorbell_signal.handle);
    if (it != queue_map_->end()) {
      SimpleProxyQueue* instance = it->second;
      index = instance->submit_index_;
    } else {
      index = hsa_queue_load_read_index_relaxed_fn(queue);
    }
    return index;
  }

  static uint64_t GetQueueIndex(const hsa_queue_t* queue) {
    uint64_t index = 0;
    auto it = queue_map_->find(queue->doorbell_signal.handle);
    if (it != queue_map_->end()) {
      SimpleProxyQueue* instance = it->second;
      instance->mutex_lock();
      index = instance->queue_index_;
    } else {
      index = hsa_queue_load_write_index_relaxed_fn(queue);
    }
    return index;
  }

  static void SetQueueIndex(const hsa_queue_t* queue, uint64_t value) {
    auto it = queue_map_->find(queue->doorbell_signal.handle);
    if (it != queue_map_->end()) {
      SimpleProxyQueue* instance = it->second;
      instance->queue_index_ = value;
      instance->mutex_unlock();
    } else {
      hsa_queue_store_write_index_relaxed_fn(queue, value);
    }
  }

  hsa_status_t SetInterceptCB(on_submit_cb_t on_submit_cb, void* data) {
    on_submit_cb_ = on_submit_cb;
    on_submit_cb_data_ = data;
    return HSA_STATUS_SUCCESS;
  }

  void Submit(const packet_t* packet) {
    // Compute the write index of queue
    const uint64_t que_idx = hsa_queue_load_write_index_relaxed_fn(queue_);

    // Waiting untill there is a free space in the queue
    while (que_idx >= (hsa_queue_load_read_index_relaxed_fn(queue_) + size_))
      ;

    // Increment the write index
    hsa_queue_store_write_index_relaxed_fn(queue_, que_idx + 1);

    const uint32_t mask = queue_->size - 1;
    const uint32_t idx = que_idx & mask;

    // Copy packet to the queue
    const packet_word_t* src = reinterpret_cast<const packet_word_t*>(packet);
    packet_word_t* dst = reinterpret_cast<packet_word_t*>(base_address_ + idx);
    for (unsigned i = 1; i < sizeof(packet_t) / sizeof(packet_word_t); ++i) {
      dst[i] = src[i];
    }

    // To maintain global order to ensure the prior copy of the packet contents is made visible
    // before the header is updated.
    // With in-order CP it will wait until the first packet in the blob will be valid.
    std::atomic<packet_word_t>* header_atomic_ptr =
        reinterpret_cast<std::atomic<packet_word_t>*>(&dst[0]);
    header_atomic_ptr->store(src[0], std::memory_order_release);

    // Doorbell signaling to submit the packet
    hsa_signal_store_relaxed_fn(doorbell_signal_, que_idx);
  }

  SimpleProxyQueue()
      : agent_info_(NULL),
        queue_(NULL),
        base_address_(NULL),
        doorbell_signal_({}),
        queue_index_(0),
        queue_mask_(0),
        submit_index_(0),
        on_submit_cb_(NULL),
        on_submit_cb_data_(NULL) {
    printf("ROCProfiler: SimpleProxyQueue is enabled\n");
    fflush(stdout);
  }

  ~SimpleProxyQueue() {}

 private:
  typedef std::map<signal_handle_t, SimpleProxyQueue*> queue_map_t;

  hsa_status_t Init(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                    void* data, uint32_t private_segment_size, uint32_t group_segment_size,
                    hsa_queue_t** queue) {
    size_ = size;
    auto status = Init(agent, size);
    *queue = queue_;
    return status;
  }

  hsa_status_t Init(hsa_agent_t agent, uint32_t size) {
    hsa_status_t status = HSA_STATUS_ERROR;
    agent_info_ = util::HsaRsrcFactory::Instance().GetAgentInfo(agent);
    if (agent_info_ != NULL) {
      if (agent_info_->dev_type == HSA_DEVICE_TYPE_GPU) {
        status = hsa_queue_create_fn(agent, size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX,
                                     UINT32_MAX, &queue_);
        if (status == HSA_STATUS_SUCCESS) {
          base_address_ = reinterpret_cast<packet_t*>(queue_->base_address);
          doorbell_signal_ = queue_->doorbell_signal;
          data_array_ = calloc(size + 1, sizeof(packet_t));
          uintptr_t addr = (uintptr_t)data_array_;
          queue_->base_address = (void*)((addr + align_mask_) & ~align_mask_);
          status = hsa_signal_create(1, 0, NULL, &(queue_->doorbell_signal));
          if (status != HSA_STATUS_SUCCESS) abort();
          queue_mask_ = size - 1;

          if (queue_map_ == NULL) queue_map_ = new queue_map_t;
          (*queue_map_)[queue_->doorbell_signal.handle] = this;
        } else
          abort();
      }
    }
    if (status != HSA_STATUS_SUCCESS) abort();
    return status;
  }

  hsa_status_t Cleanup() const {
    hsa_status_t status = HSA_STATUS_ERROR;
    hsa_signal_t queue_signal = queue_->doorbell_signal;

    // Destroy original HSA queue
    queue_->base_address = base_address_;
    queue_->doorbell_signal = doorbell_signal_;
    status = hsa_queue_destroy_fn(queue_);
    if (status != HSA_STATUS_SUCCESS) abort();

    // Destroy overloaded virtual queue data and signal
    free(data_array_);
    status = hsa_signal_destroy(queue_signal);
    if (status != HSA_STATUS_SUCCESS) abort();

    return status;
  }

  void mutex_lock() {
#if ROCP_PROXY_LOCK
    mutex_.lock();
#endif
  }

  void mutex_unlock() {
#if ROCP_PROXY_LOCK
    mutex_.unlock();
#endif
  }

  uint32_t size_;
  static queue_map_t* queue_map_;
  const util::AgentInfo* agent_info_;
  hsa_queue_t* queue_;
  static const uintptr_t align_mask_ = sizeof(packet_t) - 1;
  packet_t* base_address_;
  hsa_signal_t doorbell_signal_;
  uint64_t queue_index_;
  uint64_t queue_mask_;
  uint64_t submit_index_;
  std::mutex mutex_;
  on_submit_cb_t on_submit_cb_;
  void* on_submit_cb_data_;
  void* data_array_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_SIMPLE_PROXY_QUEUE_H
