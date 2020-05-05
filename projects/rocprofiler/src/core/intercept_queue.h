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

#ifndef _SRC_CORE_INTERCEPT_QUEUE_H
#define _SRC_CORE_INTERCEPT_QUEUE_H

#include <amd_hsa_kernel_code.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <sys/syscall.h>

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>

#include "core/context.h"
#include "core/proxy_queue.h"
#include "core/tracker.h"
#include "core/types.h"
#include "inc/rocprofiler.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
extern decltype(hsa_queue_create)* hsa_queue_create_fn;
extern decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;

class InterceptQueue {
 public:
  typedef std::recursive_mutex mutex_t;
  typedef std::map<uint64_t, InterceptQueue*> obj_map_t;
  typedef hsa_status_t (*queue_callback_t)(hsa_queue_t*, void* data);
  typedef void (*queue_event_callback_t)(hsa_status_t status, hsa_queue_t *queue, void *arg);
  typedef uint32_t queue_id_t;

  static void HsaIntercept(HsaApiTable* table);

  static hsa_status_t InterceptQueueCreate(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                  void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                   void* data),
                                  void* data, uint32_t private_segment_size,
                                  uint32_t group_segment_size, hsa_queue_t** queue,
                                  const bool& tracker_on) {
    std::lock_guard<mutex_t> lck(mutex_);
    hsa_status_t status = HSA_STATUS_ERROR;

    if (in_create_call_) EXC_ABORT(status, "recursive InterceptQueueCreate()");
    in_create_call_ = true;

    ProxyQueue* proxy = ProxyQueue::Create(agent, size, type, queue_event_callback, data, private_segment_size,
                                           group_segment_size, queue, &status);
    if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "ProxyQueue::Create()");

    if (tracker_on || tracker_on_) {
      if (tracker_ == NULL) tracker_ = &Tracker::Instance();
      status = rocprofiler::util::HsaRsrcFactory::HsaApi()->hsa_amd_profiling_set_profiler_enabled(*queue, true);
      if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "hsa_amd_profiling_set_profiler_enabled()");
    }

    if (!obj_map_) obj_map_ = new obj_map_t;
    InterceptQueue* obj = new InterceptQueue(agent, *queue, proxy);
    (*obj_map_)[(uint64_t)(*queue)] = obj;
    status = proxy->SetInterceptCB(OnSubmitCB, obj);
    obj->queue_event_callback_ = callback;
    obj->queue_id = current_queue_id;
    ++current_queue_id;

    if (callbacks_.create != NULL) {
      status = callbacks_.create(*queue, callback_data_);
    }

    in_create_call_ = false;
    return status;
  }

  static hsa_status_t QueueCreate(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                  void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                   void* data),
                                  void* data, uint32_t private_segment_size,
                                  uint32_t group_segment_size, hsa_queue_t** queue) {
    return InterceptQueueCreate(agent, size, type, callback, data, private_segment_size, group_segment_size, queue, false);
  }

  static hsa_status_t QueueCreateTracked(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                  void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                   void* data),
                                  void* data, uint32_t private_segment_size,
                                  uint32_t group_segment_size, hsa_queue_t** queue) {
    return InterceptQueueCreate(agent, size, type, callback, data, private_segment_size, group_segment_size, queue, true);
  }

  static hsa_status_t QueueDestroy(hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(mutex_);
    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (callbacks_.destroy != NULL) {
      status = callbacks_.destroy(queue, callback_data_);
    }

    if (status == HSA_STATUS_SUCCESS) {
      status = DelObj(queue);
    }

    return status;
  }

  static void OnSubmitCB(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data,
                         hsa_amd_queue_intercept_packet_writer writer) {
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptQueue* obj = reinterpret_cast<InterceptQueue*>(data);
    Queue* proxy = obj->proxy_;

    if (submit_callback_fun_) {
      mutex_.lock();
      auto* callback_fun = submit_callback_fun_;
      void* callback_arg = submit_callback_arg_;
      mutex_.unlock();

      if (callback_fun) {
        for (uint64_t j = 0; j < count; ++j) {
          const packet_t* packet = &packets_arr[j];
          const hsa_kernel_dispatch_packet_t* dispatch_packet =
              reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);

          const char* kernel_name = NULL;
          if (GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
            uint64_t kernel_object = dispatch_packet->kernel_object;
            const amd_kernel_code_t* kernel_code = GetKernelCode(kernel_object);
            kernel_name = (GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) ?
              QueryKernelName(kernel_object, kernel_code) : NULL;
          }

          // Prepareing submit callback data
          rocprofiler_hsa_callback_data_t data{};
          data.submit.packet = (void*)packet;
          data.submit.kernel_name = kernel_name;
          data.submit.queue = obj->queue_;
          data.submit.device_type = obj->agent_info_->dev_type;
          data.submit.device_id = obj->agent_info_->dev_index;

          callback_fun(ROCPROFILER_HSA_CB_ID_SUBMIT, &data, callback_arg);
        }
      }
    }

    // Travers input packets
    for (uint64_t j = 0; j < count; ++j) {
      const packet_t* packet = &packets_arr[j];
      bool to_submit = true;

      // Checking for dispatch packet type
      if ((GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) &&
          (dispatch_callback_.load(std::memory_order_acquire) != NULL)) {
        const hsa_kernel_dispatch_packet_t* dispatch_packet =
            reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);
        const hsa_signal_t completion_signal = dispatch_packet->completion_signal;

        // Adding kernel timing tracker
        Tracker::entry_t* tracker_entry = NULL;
        if (tracker_ != NULL) {
          tracker_entry = tracker_->Alloc(obj->agent_info_->dev_id, dispatch_packet->completion_signal);
          const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal = tracker_entry->signal;
        }

        // Prepareing dispatch callback data
        uint64_t kernel_object = dispatch_packet->kernel_object;
        const amd_kernel_code_t* kernel_code = GetKernelCode(kernel_object);
        const char* kernel_name = QueryKernelName(kernel_object, kernel_code);

        rocprofiler_callback_data_t data = {obj->agent_info_->dev_id,
                                            obj->agent_info_->dev_index,
                                            obj->queue_,
                                            user_que_idx,
                                            obj->queue_id,
                                            completion_signal,
                                            dispatch_packet,
                                            kernel_name,
                                            kernel_object,
                                            kernel_code,
                                            (uint32_t)syscall(__NR_gettid),
                                            (tracker_entry) ? tracker_entry->record : NULL};

        // Calling dispatch callback
        rocprofiler_group_t group = {};
        hsa_status_t status = (dispatch_callback_.load())(&data, callback_data_, &group);
        free(const_cast<char*>(kernel_name));
        // Injecting profiling start/stop packets
        if ((status != HSA_STATUS_SUCCESS) || (group.context == NULL)) {
          if (tracker_entry != NULL) {
            const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal = tracker_entry->orig;
            tracker_->Delete(tracker_entry);
          }
        } else {
          Context* context = reinterpret_cast<Context*>(group.context);

          if (group.feature_count != 0) {
            if (tracker_entry != NULL) {
              Group* context_group = context->GetGroup(group.index);
              context_group->IncrRefsCount();
              tracker_->EnableContext(tracker_entry, Context::Handler, reinterpret_cast<void*>(context_group));
            }

            const pkt_vector_t& start_vector = context->StartPackets(group.index);
            const pkt_vector_t& stop_vector = context->StopPackets(group.index);
            pkt_vector_t packets = start_vector;
            packets.insert(packets.end(), *packet);
            packets.insert(packets.end(), stop_vector.begin(), stop_vector.end());
            if (writer != NULL) {
              writer(&packets[0], packets.size());
            } else {
              proxy->Submit(&packets[0], packets.size());
            }
            to_submit = false;
          } else {
            if (tracker_entry != NULL) {
              void* context_handler_arg = NULL;
              rocprofiler_handler_t context_handler_fun = context->GetHandler(&context_handler_arg);
              tracker_->EnableDispatch(tracker_entry, context_handler_fun, context_handler_arg);
            }
          }
        }
      }

      // Submitting the original packets if profiling was not enabled
      if (to_submit) {
        if (writer != NULL) {
          writer(packet, 1);
        } else {
          proxy->Submit(packet, 1);
        }
      }
    }
  }

  static void SetCallbacks(rocprofiler_queue_callbacks_t callbacks, void* data) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (callback_data_ != NULL) {
      EXC_ABORT(HSA_STATUS_ERROR, "reassigning queue callbacks - not supported");
    }
    callbacks_ = callbacks;
    callback_data_ = data;
    Start();
  }

  static void RemoveCallbacks() {
    std::lock_guard<mutex_t> lck(mutex_);
    callbacks_ = {};
    Stop();
  }

  static inline void Start() { dispatch_callback_.store(callbacks_.dispatch, std::memory_order_release); }
  static inline void Stop() { dispatch_callback_.store(NULL, std::memory_order_relaxed); }

  static void SetSubmitCallback(rocprofiler_hsa_callback_fun_t fun, void* arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    submit_callback_fun_ = fun;
    submit_callback_arg_ = arg;
  }

  static void TrackerOn(bool on) { tracker_on_ = on; }
  static bool IsTrackerOn() { return tracker_on_; }

 private:
  static void queue_event_callback(hsa_status_t status, hsa_queue_t *queue, void *arg) {
    if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "queue error handling is not supported");
    InterceptQueue* obj = GetObj(queue);
    if (obj->queue_event_callback_) obj->queue_event_callback_(status, obj->queue_, arg);
  }

  static hsa_packet_type_t GetHeaderType(const packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) & header_type_mask);
  }

  static const amd_kernel_code_t* GetKernelCode(uint64_t kernel_object) {
    const amd_kernel_code_t* kernel_code = NULL;
    hsa_status_t status =
        util::HsaRsrcFactory::Instance().LoaderApi()->hsa_ven_amd_loader_query_host_address(
            reinterpret_cast<const void*>(kernel_object),
            reinterpret_cast<const void**>(&kernel_code));
    if (HSA_STATUS_SUCCESS != status) {
      kernel_code = reinterpret_cast<amd_kernel_code_t*>(kernel_object);
    }
    return kernel_code;
  }

  static const char* GetKernelName(const uint64_t kernel_symbol) {
    amd_runtime_loader_debug_info_t* dbg_info =
        reinterpret_cast<amd_runtime_loader_debug_info_t*>(kernel_symbol);
    return (dbg_info != NULL) ? dbg_info->kernel_name : NULL;
  }

  // Demangle C++ symbol name
  static const char* cpp_demangle(const char* symname) {
    size_t size = 0;
    int status;
    const char* ret = abi::__cxa_demangle(symname, NULL, &size, &status);
    return (ret != 0) ? ret : strdup(symname);
  }

  static const char* QueryKernelName(uint64_t kernel_object, const amd_kernel_code_t* kernel_code) {
    const uint16_t kernel_object_flag = *((uint64_t*)kernel_code + 1);
    if (kernel_object_flag == 0) {
      if (!util::HsaRsrcFactory::IsExecutableTracking()) {
        EXC_ABORT(HSA_STATUS_ERROR, "Error: V3 code object detected - code objects tracking should be enabled\n");
      }
    }
    const char* kernel_symname = (util::HsaRsrcFactory::IsExecutableTracking()) ?
      util::HsaRsrcFactory::GetKernelNameRef(kernel_object) :
      GetKernelName(kernel_code->runtime_loader_kernel_symbol);
    return cpp_demangle(kernel_symname);
  }

  // method to get an intercept queue object
  static InterceptQueue* GetObj(const hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(mutex_);
    InterceptQueue* obj = NULL;
    obj_map_t::const_iterator it = obj_map_->find((uint64_t)queue);
    if (it != obj_map_->end()) {
      obj = it->second;
      assert(queue == obj->queue_);
    }
    return obj;
  }

  // method to delete an intercept queue object
  static hsa_status_t DelObj(const hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(mutex_);
    hsa_status_t status = HSA_STATUS_ERROR;
    obj_map_t::const_iterator it = obj_map_->find((uint64_t)queue);
    if (it != obj_map_->end()) {
      const InterceptQueue* obj = it->second;
      assert(queue == obj->queue_);
      delete obj;
      obj_map_->erase(it);
      status = HSA_STATUS_SUCCESS;;
    }
    return status;
  }

  InterceptQueue(const hsa_agent_t& agent, hsa_queue_t* const queue, ProxyQueue* proxy) :
    queue_(queue),
    proxy_(proxy)
  {
    agent_info_ = util::HsaRsrcFactory::Instance().GetAgentInfo(agent);
    queue_event_callback_ = NULL;
  }

  ~InterceptQueue() {
    ProxyQueue::Destroy(proxy_);
  }

  static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;

  static mutex_t mutex_;
  static rocprofiler_queue_callbacks_t callbacks_;
  static void* callback_data_;
  static std::atomic<rocprofiler_callback_t> dispatch_callback_;

  static obj_map_t* obj_map_;
  static const char* kernel_none_;
  static Tracker* tracker_;
  static bool tracker_on_;
  static bool in_create_call_;
  static queue_id_t current_queue_id;

  static rocprofiler_hsa_callback_fun_t submit_callback_fun_;
  static void* submit_callback_arg_;

  hsa_queue_t* const queue_;
  ProxyQueue* const proxy_;
  const util::AgentInfo* agent_info_;
  queue_event_callback_t queue_event_callback_;
  queue_id_t queue_id;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_INTERCEPT_QUEUE_H
