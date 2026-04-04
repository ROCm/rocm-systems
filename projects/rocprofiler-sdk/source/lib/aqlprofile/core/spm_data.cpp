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

#include "hsa/hsa_ext_amd.h"

#include <thread>
#include <condition_variable>

#include "core/logger.h"
#include "core/pm4_factory.h"

// C++11's solution for std::format()
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

#define DEBUG_SPM   0
#define SUPPORT_XCC 1

struct spm_set_dest_buffer_args {
  hsa_agent_t agent;
  size_t buf_size;
  uint32_t timeout;
  uint32_t size_copied;
  void* dest_buf;
  bool is_data_loss;
};

struct spm_state_t : public spm_set_dest_buffer_args {
  std::thread* manager_thread;
  std::mutex work_mutex;
  std::condition_variable work_cond;
  std::atomic<bool> data_ready;

  std::atomic<bool> stop_prod_thread;
  std::atomic<bool> stop_cons_thread;
  void* prod_buf;
  void* cons_buf;
  uint32_t num_xcc;
  size_t buf_size_xcc;

  // Parameters from spm_iterate_data
  const hsa_ven_amd_aqlprofile_profile_t* profile;
  hsa_ven_amd_aqlprofile_data_callback_t callback;
  void* data;
};

#if DEBUG_SPM >= 2
static int data_ready_check[2] = {};
#endif

inline static hsa_status_t HsaSpmSetDestBuffer(spm_set_dest_buffer_args& args) {
  return hsa_amd_spm_set_dest_buffer(args.agent, args.buf_size, &args.timeout, &args.size_copied,
                                     args.dest_buf, &args.is_data_loss);
}

static void producer(spm_state_t* s) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  spm_set_dest_buffer_args args = *s;
  bool exiting = false;
  int count_down = 0;

  args.timeout = s->timeout;
  do {
    args.size_copied = 0;
    args.dest_buf = s->prod_buf;
    // s->stop_prod_thread should be set after SPM End() sequence is submitted, this is the
    // handshake protocal between app/library and aqlprofile.
    // If s->stop_prod_thread is set in current loop, producer thread will exit after all
    // SPM counters are drained (args.size_copied == 0) which could be at least one
    // HsaSpmSetDestBuffer() call or maybe more than one.
    if (s->stop_prod_thread)
      exiting = true;
    status = HsaSpmSetDestBuffer(args);
    if (status != HSA_STATUS_SUCCESS) {
      ERR_LOGGING << "hsa_amd_spm_set_dest_buffer() error";
      goto exit_;
    }
#if DEBUG_SPM >= 2
    if (s->data_ready) data_ready_check[0]++;
#endif
    std::unique_lock<std::mutex> lock(s->work_mutex);
    void* tmp = s->prod_buf;
    s->prod_buf = s->cons_buf;
    s->cons_buf = s->dest_buf;
    s->dest_buf = tmp;
    s->size_copied = args.size_copied;
    s->is_data_loss = args.is_data_loss;
    s->data_ready = true;
    s->work_cond.notify_one();
    lock.unlock();
#if DEBUG_SPM >= 2
    if (s->data_ready) data_ready_check[1]++;
#endif
    // We must make sure consumer_thread owns s->work_mutex before we proceed to next loop in
    // producer_thread
    while (s->data_ready) {
      if (lock.try_lock()) lock.unlock();
    }

    // We cannot directly use s->stop_prod_thread here, otherwise we might miss the last
    // HsaSpmSetDestBuffer() call if s->stop_prod_thread is set after the HsaSpmSetDestBuffer()
    // call from this loop!
    //
    if (exiting && !s->size_copied) break;
    // Forced exit: This happens when we want to stop SPM but not the app. This should be
    // improved by getting the hint from caller instead of a hardcoded number. Will consider this
    // in the new SPM api design
    #define MAX_EXTRA_CALLS_AFTER_FORCED_EXIT 5
    if (exiting && s->size_copied) {
      count_down++;
      if (count_down > MAX_EXTRA_CALLS_AFTER_FORCED_EXIT) {
        printf("Forced exit after %d extra hsa_amd_spm_set_dest_buffer() calls\n", count_down);
        break;
      }
    }
    if (s->stop_cons_thread) break;
  } while (1);
exit_:
  if (status != HSA_STATUS_SUCCESS) {
    // Even when HsaSpmSetDestBuffer() fails, we still need to fulfill the handshake protocal
    // between producer and consumer
    std::unique_lock<std::mutex> lock(s->work_mutex);
    s->size_copied = 0;
    s->data_ready = true;
    s->work_cond.notify_one();
  }
  s->stop_cons_thread = true;
}

static void consumer(spm_state_t* s) {
  do {
    std::unique_lock<std::mutex> lock(s->work_mutex);
    while (!s->data_ready) s->work_cond.wait(lock);
    s->data_ready = false;

    hsa_status_t status = HSA_STATUS_SUCCESS;
    hsa_ven_amd_aqlprofile_info_data_t sample_info{};
#if SUPPORT_XCC
    char* base = (char*)s->cons_buf;
    for (int i = 0; i < s->num_xcc; i++) {
      auto buf_info = (struct kfd_ioctl_spm_buffer_header*)base;
      if (buf_info->bytes_copied) {
        sample_info.sample_id = i;
        sample_info.trace_data.ptr = base + sizeof(struct kfd_ioctl_spm_buffer_header);
        sample_info.trace_data.size = buf_info->bytes_copied;
        hsa_status_t status =
            s->callback(HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA, &sample_info, s->data);
      }
      base += s->buf_size_xcc;
    }
#else
    if (s->size_copied) {
      sample_info.trace_data.ptr = s->cons_buf;
      sample_info.trace_data.size = s->size_copied;

      hsa_status_t status =
          s->callback(HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA, &sample_info, s->data);
    }
#endif

    if (status != HSA_STATUS_SUCCESS) {
      ERR_LOGGING << "SPM consumer callback failed";
      s->stop_cons_thread = true;
    }
  } while (!s->stop_cons_thread);
}

static void manager(spm_state_t* s) {
  // spm threads
  std::thread producer_thread(producer, s);
  std::thread consumer_thread(consumer, s);

  producer_thread.join();
  consumer_thread.join();
}

hsa_status_t start_spm_threads(spm_state_t& s) {
  hsa_status_t status = hsa_amd_spm_acquire(s.profile->agent);
  if (status != HSA_STATUS_SUCCESS) {
    ERR_LOGGING << "hsa_amd_spm_acquire() error";
    abort();
    return status;
  }

  // The first page of output_buffer is reserved for SpmBufferDesc
  char* buf_ptr = (char*)(s.profile->output_buffer.ptr) + SPM_DESC_SIZE;
  size_t buf_size = (s.profile->output_buffer.size - SPM_DESC_SIZE) / 3;
  SpmBufferDesc* desc = (SpmBufferDesc*)s.profile->output_buffer.ptr;
  size_t seg_size = (desc->global_num_line + desc->se_num_line * desc->num_se) * 32;
  // Align buf_size to the exact multiples of segments, so that every HsaSpmSetDestBuffer
  // will always return complete segments
  if (!desc->num_xcc) desc->num_xcc = 1;
#if SUPPORT_XCC
  buf_size /= desc->num_xcc;
  if (seg_size) {
    buf_size = (buf_size - sizeof(struct kfd_ioctl_spm_buffer_header)) / seg_size * seg_size +
               sizeof(struct kfd_ioctl_spm_buffer_header);
  }
  buf_size *= desc->num_xcc;
#else
  if (seg_size) buf_size = buf_size / seg_size * seg_size;
#endif
#if DEBUG_SPM >= 3
  FILE* fp = fopen("spm_header.bin", "wb");
  if (fp) {
    fwrite(s.profile->output_buffer.ptr, 1, 0x1000, fp);
    fclose(fp);
  }
  std::clog << string_format("Buffer Size = %d (%x) bytes\n", buf_size, buf_size);
  std::clog << string_format("Segment Size = %d bytes\n", seg_size);
  for (int i = 0; i < s.profile->event_count; i++) {
    auto it = &s.profile->events[i];
    std::clog << string_format("block (%d_%d) id (%d) at offset %d\n", it->block_name,
                               it->block_index, it->counter_id, desc->counter_map[i]);
  }
#endif

  // Args for hsa_amd_spm_set_dest_buffer
  s.agent = s.profile->agent;
  s.buf_size = buf_size;
  s.timeout = 1000;  // 1sec
  s.dest_buf = buf_ptr;

  s.prod_buf = buf_ptr + buf_size;
  s.cons_buf = buf_ptr + buf_size * 2;
  s.num_xcc = desc->num_xcc;
  s.buf_size_xcc = s.buf_size / desc->num_xcc;

  // This non-blocking (timeout = 0) HsaSpmSetDestBuffer() call will clear up all the
  // residual counters from previous SPM runs. Most of the time, nothing will be copied.
  // This call will also trigger KFD to call spm_start() function. We must make sure
  // spm_start() is finished before we give back the control to caller of
  // start_spm_threads().
  spm_set_dest_buffer_args args = s;
  args.size_copied = 0;
  args.timeout = 0;
  status = HsaSpmSetDestBuffer(args);
  if (status != HSA_STATUS_SUCCESS) {
    ERR_LOGGING << "hsa_amd_spm_set_dest_buffer() init error";
    abort();
    return status;
  }
  if (args.size_copied) {
    std::clog << string_format("HsaSpmSetDestBuffer().data_size=%d (init)\n", args.size_copied);
  }

  s.manager_thread = new std::thread(manager, &s);

  if (!s.manager_thread) {
    hsa_amd_spm_release(s.profile->agent);
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

void stop_spm_threads(spm_state_t& s) {
  s.stop_prod_thread = true;
  s.manager_thread->join();
  hsa_amd_spm_release(s.profile->agent);
  delete s.manager_thread;
  s.manager_thread = nullptr;
#if DEBUG_SPM >= 2
  printf("data_ready_check = %d, %d\n", data_ready_check[0], data_ready_check[1]);
#endif
}

typedef std::mutex spm_mutex_t;
spm_mutex_t spm_mutex;

// Getting SPM data using driver API
hsa_status_t spm_iterate_data(const hsa_ven_amd_aqlprofile_profile_t* profile,
                              hsa_ven_amd_aqlprofile_data_callback_t callback, void* data) {
  std::lock_guard<spm_mutex_t> lck(spm_mutex);
  static spm_state_t s{};

  if (data && !s.manager_thread) {
    s.profile = profile;
    s.callback = callback;
    s.data = data;
    return start_spm_threads(s);
  } else if (!data && s.manager_thread)
    stop_spm_threads(s);

  return HSA_STATUS_SUCCESS;
}
