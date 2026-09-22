/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef SRC_CORE_MEMORY_GENERIC_BUFFER_H_
#define SRC_CORE_MEMORY_GENERIC_BUFFER_H_
#include "rocprofiler.h"

#include <bitset>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#define ASSERTM(exp, msg) assert(((void)msg, exp))

namespace Memory {

class GenericBuffer {
 public:
  GenericBuffer(rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t id, size_t buffer_size,
                rocprofiler_buffer_callback_t flush_function);
  ~GenericBuffer();

  GenericBuffer(const GenericBuffer&) = delete;
  GenericBuffer& operator=(const GenericBuffer&) = delete;

  rocprofiler_buffer_id_t GetId();
  rocprofiler_session_id_t GetSessionId();

  bool Flush();

  template <typename Record, typename Functor = std::function<void(Record& record, const void*)>>
  void AddRecord(Record&& record, const void* data, size_t data_size, Functor&& store_data) {
    if (!is_valid_) return;
    assert(data != nullptr || data_size == 0);  // If data is null, then data_size must be 0

    std::lock_guard producer_lock(producer_mutex_);

    // The amount of memory reserved in the buffer to store data. If the data
    // cannot fit because it is larger than the buffer size minus one record,
    // then the data won't be copied into the buffer.
    size_t reserve_data_size = data_size <= (buffer_size_ - sizeof(Record)) ? data_size : 0;

    std::byte* next_record = record_ptr_ + sizeof(Record);
    if (next_record > (data_ptr_ - reserve_data_size)) {
      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
      next_record = record_ptr_ + sizeof(Record);
      assert(next_record <= buffer_end_ && "buffer size is less then the record size");
    }

    // Store data in the record. Copy the data first if it fits in the buffer
    if (reserve_data_size != 0) {
      data_ptr_ -= data_size;
      ::memcpy(data_ptr_, data, data_size);
      store_data(record, data_ptr_);
    } else if (data != nullptr) {
      store_data(record, data);
    }

    // Store the record into the buffer, and increment the write pointer.
    ::memcpy(record_ptr_, &record, sizeof(Record));
    record_ptr_ = next_record;

    // If the data does not fit in the buffer, flush the buffer with the record
    // as is. We don't copy the data so we make sure that the record and its
    // data are processed by waiting until the flush is complete.
    if (data != nullptr && reserve_data_size == 0) {
      NotifyConsumerThread(buffer_begin_, record_ptr_);
      SwitchBuffers();
      {
        std::unique_lock consumer_lock(consumer_mutex_);
        consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid; });
      }
    }
  }
  template <typename Record> bool AddRecord(Record&& record) {
    using DataPtr = void*;
    AddRecord(std::forward<Record>(record), DataPtr(nullptr), 0, {});
    return true;
  }

  void SetProperties(rocprofiler_buffer_property_t* buffer_properties,
                     uint32_t buffer_properties_count);


  bool GetPeriodicFlushFlag(rocprofiler_session_id_t session_id);

  bool IsValid();

  std::mutex& GetBufferLock();

 private:
  std::atomic<bool> consumerRunning{false};
  void SwitchBuffers();
  void ConsumerThreadLoop(std::promise<void> ready);
  void NotifyConsumerThread(const std::byte* data_begin, const std::byte* data_end);
  void AllocateMemory(std::byte** ptr, size_t size) const;

  // Memory Pool and Buffers
  size_t buffer_size_;
  size_t available_space_;
  std::byte* pool_begin_;
  std::byte* pool_end_;
  std::byte* buffer_begin_;
  std::byte* buffer_end_;
  std::byte* record_ptr_;
  std::byte* data_ptr_;
  std::mutex producer_mutex_;

  // Session related Information
  rocprofiler_buffer_id_t id_;
  rocprofiler_buffer_callback_t flush_function_;
  rocprofiler_session_id_t session_id_;
  std::vector<rocprofiler_buffer_property_t> properties_;
  std::atomic<bool> is_valid_{false};

  // Flush Period
  std::mutex periodic_flush_threads_map_lock_;
  std::map<uint64_t, std::thread> periodic_flush_threads_;
  std::mutex periodic_flush_flags_map_lock_;
  std::map<uint64_t, std::atomic<bool>> periodic_flush_flags_;

  // Consumer thread
  std::thread consumer_thread_;
  struct {
    const std::byte* begin;
    const std::byte* end;
    bool valid = false;
  } consumer_arg_;

  std::mutex consumer_mutex_;
  std::condition_variable consumer_cond_;

  std::mutex buffer_lock_;
};

bool GetNextRecord(const rocprofiler_record_header_t* record,
                   const rocprofiler_record_header_t** next);

}  // namespace Memory
#endif  // SRC_CORE_MEMORY_GENERIC_BUFFER_H_
