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

#include "generic_buffer.h"

#include <algorithm>
#include <atomic>

#include "rocprofiler.h"
#include "src/api/rocprofiler_singleton.h"

namespace Memory {

void periodic_flush_buffer_fn(rocprofiler_session_id_t session_id);

GenericBuffer::GenericBuffer(rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t id,
                             size_t buffer_size, rocprofiler_buffer_callback_t flush_function)
    : buffer_size_(buffer_size),
      available_space_(buffer_size_),
      id_(id),
      flush_function_(flush_function),
      session_id_(session_id) {
  if (!is_valid_.load(std::memory_order_acquire)) {
    // Pool definition: The memory pool is split in 2 buffers of equal size. When
    // first initialized, the write pointer points to the first element of the
    // first buffer. When a buffer is full,  or when Flush() is called, the write
    // pointer moves to the other buffer. Each buffer should be large enough to
    // hold at least 2 activity records, as record pairs may be written when
    // external correlation ids are used.
    const size_t allocation_size =
        2 * std::max(2 * sizeof(rocprofiler_record_header_t), buffer_size);
    pool_begin_ = nullptr;
    AllocateMemory(&pool_begin_, allocation_size);
    assert(pool_begin_ != nullptr && "pool allocator failed");

    pool_end_ = pool_begin_ + allocation_size;
    buffer_begin_ = pool_begin_;
    buffer_end_ = buffer_begin_ + buffer_size;
    record_ptr_ = buffer_begin_;
    data_ptr_ = buffer_end_;

    // Create a consumer thread and wait for it to be ready to accept work.
    std::promise<void> ready;
    std::future<void> future = ready.get_future();
    consumer_thread_ = std::thread(&GenericBuffer::ConsumerThreadLoop, this, std::move(ready));
    future.wait();

    is_valid_.exchange(true, std::memory_order_release);
  }
}

GenericBuffer::~GenericBuffer() {
  if (is_valid_.load(std::memory_order_acquire)) {
    std::lock_guard lock(buffer_lock_);

    Flush();

    // Wait for the previous flush to complete, then send the exit signal.
    NotifyConsumerThread(nullptr, nullptr);
    consumer_thread_.join();

    // Free the pool's buffer memory.
    AllocateMemory(&pool_begin_, 0);

    is_valid_.exchange(false, std::memory_order_release);
  }
}

bool GenericBuffer::Flush() {
  {
    std::lock_guard producer_lock(producer_mutex_);
    if (record_ptr_ == buffer_begin_) return true;

    NotifyConsumerThread(buffer_begin_, record_ptr_);
    SwitchBuffers();
  }
  {
    // Wait for the current operation to complete.
    std::unique_lock consumer_lock(consumer_mutex_);
    consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid || !consumerRunning; });
  }
  return true;
}

void GenericBuffer::SetProperties(rocprofiler_buffer_property_t* buffer_properties,
                                  uint32_t buffer_properties_count) {
  // TODO(aelwazir): Change it to do real work
  for (uint32_t i = 0; i < buffer_properties_count; i++)
    properties_.emplace_back(buffer_properties[i]);
}

void GenericBuffer::SwitchBuffers() {
  buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
  buffer_end_ = buffer_begin_ + buffer_size_;
  record_ptr_ = buffer_begin_;
  data_ptr_ = buffer_end_;
}

void GenericBuffer::ConsumerThreadLoop(std::promise<void> ready) {
  std::unique_lock consumer_lock(consumer_mutex_);

  // This consumer is now ready to accept work.
  consumerRunning.store(true);
  ready.set_value();

  while (true) {
    consumer_cond_.wait(consumer_lock, [this]() { return consumer_arg_.valid; });

    // begin == end == nullptr means the thread needs to exit.
    if (consumer_arg_.begin == nullptr && consumer_arg_.end == nullptr) break;

    flush_function_(reinterpret_cast<const rocprofiler_record_header_t*>(consumer_arg_.begin),
                    reinterpret_cast<const rocprofiler_record_header_t*>(consumer_arg_.end),
                    session_id_, id_);

    // Mark this operation as complete (valid=false) and notify all producers
    // that may be waiting for this operation to finish, or to start a new
    // operation. See comment below in NotifyConsumerThread().
    consumer_arg_.valid = false;
    consumer_cond_.notify_all();
  }
  consumerRunning.store(false);
}

void GenericBuffer::NotifyConsumerThread(const std::byte* data_begin, const std::byte* data_end) {
  std::unique_lock consumer_lock(consumer_mutex_);

  // If consumer_arg_ is still in use (valid=true), then wait for the consumer
  // thread to finish processing the current operation. Multiple producers may
  // wait here, one will be allowed to continue once the consumer thread is
  // idle and valid=false. This prevents a race condition where operations
  // would be lost if multiple producers could enter this critical section
  // (sequentially) before the consumer thread could re-acquire the
  // consumer_mutex_ lock.
  consumer_cond_.wait(consumer_lock, [this]() { return !consumer_arg_.valid || !consumerRunning; });

  consumer_arg_.begin = data_begin;
  consumer_arg_.end = data_end;

  consumer_arg_.valid = true;
  consumer_cond_.notify_all();
}

void GenericBuffer::AllocateMemory(std::byte** ptr, size_t size) const {
  // Allocate using the default malloc/realloc/free allocator.
  if (*ptr == nullptr && size > 0) {
    *ptr = static_cast<std::byte*>(malloc(size));
  } else if (size != 0) {
    *ptr = static_cast<std::byte*>(realloc(*ptr, size));
  } else {
    if (*ptr) free(*ptr);
    *ptr = nullptr;
  }
}

rocprofiler_session_id_t GenericBuffer::GetSessionId() {
  if (is_valid_) return session_id_;
  return rocprofiler_session_id_t{0};
}

bool GenericBuffer::IsValid() { return is_valid_.load(std::memory_order_acquire); }

rocprofiler_buffer_id_t GenericBuffer::GetId() {
  if (is_valid_) return id_;
  return rocprofiler_buffer_id_t{0};
}

std::mutex& GenericBuffer::GetBufferLock() { return buffer_lock_; }

bool GetNextRecord(const rocprofiler_record_header_t* record,
                   const rocprofiler_record_header_t** next) {
  // size_t size_to_add = sizeof(rocprofiler_record_header_t);
  switch (record->kind) {
    case ROCPROFILER_PROFILER_RECORD: {
      const rocprofiler_record_profiler_t* profiler_record =
          reinterpret_cast<const rocprofiler_record_profiler_t*>(record);
      // size_to_add = sizeof(rocprofiler_record_profiler_t);
      // if (profiler_record->counters_count.value > 0) {
      //   size_to_add += (profiler_record->counters_count.value *
      //                   sizeof(rocprofiler_record_counter_instance_t));
      // }
      *next = reinterpret_cast<const rocprofiler_record_header_t*>(profiler_record + 1);
      break;
    }
    case ROCPROFILER_SPM_RECORD: {
      const rocprofiler_record_spm_t* spm_record =
          reinterpret_cast<const rocprofiler_record_spm_t*>(record);
      *next = reinterpret_cast<const rocprofiler_record_header_t*>(spm_record + 1);
      break;
    }
    case ROCPROFILER_TRACER_RECORD: {
      const rocprofiler_record_tracer_t* tracer_record =
          reinterpret_cast<const rocprofiler_record_tracer_t*>(record);
      // size_to_add = sizeof(rocprofiler_record_tracer_t);
      // if (tracer_record->api_data_handle.size > 0) {
      //   size_to_add += tracer_record->api_data_handle.size;
      // }
      *next = reinterpret_cast<const rocprofiler_record_header_t*>(tracer_record + 1);
      break;
    }
    case ROCPROFILER_ATT_TRACER_RECORD: {
      const rocprofiler_record_att_tracer_t* att_tracer_record =
          reinterpret_cast<const rocprofiler_record_att_tracer_t*>(record);
      *next = reinterpret_cast<const rocprofiler_record_header_t*>(att_tracer_record + 1);
      break;
    }
    case ROCPROFILER_COUNTERS_SAMPLER_RECORD: {
      const rocprofiler_record_counters_sampler_t* sampler_record =
          reinterpret_cast<const rocprofiler_record_counters_sampler_t*>(record);
      *next = reinterpret_cast<const rocprofiler_record_header_t*>(sampler_record + 1);
      break;
    }
    default:
      const rocprofiler_record_tracer_t* tracer_record =
          reinterpret_cast<const rocprofiler_record_tracer_t*>(record);
      *next = reinterpret_cast<const rocprofiler_record_header_t*>(tracer_record + 1);
      // size_to_add = sizeof(rocprofiler_record_header_t);
      break;
  }
  // const std::byte* ptr = reinterpret_cast<const std::byte*>(record);
  // ptr += size_to_add;
  // *next = reinterpret_cast<const rocprofiler_record_header_t*>(ptr);

  return true;
}

}  // namespace Memory
