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

#include <hsa/amd_hsa_kernel_code.h>
#include <dlfcn.h>
#include <sys/syscall.h>

#include <atomic>
#include <cassert>
#include <iostream>
#include <map>
#include <mutex>

#include "core/context.h"
#include "core/proxy_queue.h"
#include "core/tracker.h"
#include "core/types.h"
#include "rocprofiler.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
enum { K_CONC_OFF = 0, K_CONC_PMC = 1, K_CONC_TRACE = 2 };

extern decltype(::hsa_queue_create)* hsa_queue_create_fn;
extern decltype(::hsa_queue_destroy)* hsa_queue_destroy_fn;

static inline void print_packet(const void* in_p, const uint32_t& in_n,
                                const uint32_t& w_n = UINT32_MAX) {
  const uint32_t size32 = util::HsaRsrcFactory::CMD_SLOT_SIZE_B / 4;
  const uint32_t* beg = (const uint32_t*)in_p;
  const uint32_t* end = beg + (in_n * size32);
  const uint32_t p_n = (w_n != UINT32_MAX) ? w_n : size32;

  printf("Packets(%p, %u):\n", beg, in_n);
  const uint32_t* p = beg;
  while (p < end) {
    const uint32_t ind = (p - beg) / size32;
    printf("%u, packet(%p):\n", ind, p);
    const uint32_t p_size = (*p == 0) ? size32 : p_n;
    for (const uint32_t* u = p; u < p + p_size; ++u) {
      printf("  %p: 0x%08x\n", u, *u);
    }
    p += size32;
  }
  fflush(stdout);
}

static std::mutex ctx_a_mutex;
typedef std::map<Context*, bool> ctx_a_map_t;
static ctx_a_map_t* ctx_a_map = NULL;
static bool ck_ctx_inactive(Context* context) {
  std::lock_guard<std::mutex> lock(ctx_a_mutex);
  if (ctx_a_map == NULL) ctx_a_map = new ctx_a_map_t;
  auto ret = ctx_a_map->insert({context, true});
  if (ret.second == false) ctx_a_map->erase(context);
  return ret.second;
}

class InterceptQueue {
 public:
  typedef std::recursive_mutex mutex_t;
  typedef std::map<uint64_t, std::unique_ptr<InterceptQueue>> obj_map_t;
  typedef hsa_status_t (*queue_callback_t)(hsa_queue_t*, void* data);
  typedef void (*queue_event_callback_t)(hsa_status_t status, hsa_queue_t* queue, void* arg);
  typedef uint32_t queue_id_t;

  InterceptQueue(const hsa_agent_t& agent, hsa_queue_t* const queue, ProxyQueue* proxy)
      : queue_(queue), proxy_(proxy) {
    agent_info_ = util::HsaRsrcFactory::Instance().GetAgentInfo(agent);
    queue_event_callback_ = NULL;
  }
  ~InterceptQueue() { ProxyQueue::Destroy(proxy_); }

  static void HsaIntercept(HsaApiTable* table);

  static hsa_status_t InterceptQueueCreate(
      hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
      void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
      uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue,
      const bool& tracker_on) {
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    hsa_status_t status = HSA_STATUS_ERROR;

    if (in_create_call_) EXC_ABORT(status, "recursive InterceptQueueCreate()");
    in_create_call_ = true;

    ProxyQueue* proxy =
        ProxyQueue::Create(agent, size, type, queue_event_callback, data, private_segment_size,
                           group_segment_size, queue, &status);
    if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "ProxyQueue::Create()");

    if (tracker_on || tracker_on_) {
      if (tracker_ == NULL) tracker_ = &Tracker::Instance();
      status = rocprofiler::util::HsaRsrcFactory::HsaApi()->hsa_amd_profiling_set_profiler_enabled(
          *queue, true);
      if (status != HSA_STATUS_SUCCESS)
        EXC_ABORT(status, "hsa_amd_profiling_set_profiler_enabled()");
    }

    auto obj = std::make_unique<InterceptQueue>(agent, *queue, proxy);
    if (k_concurrent_ == K_CONC_TRACE) {
      status = proxy->SetInterceptCB(OnSubmitCB_ctrace, obj.get());
    } else if (opt_mode_) {
      status = proxy->SetInterceptCB(OnSubmitCB_opt, obj.get());
    } else {
      status = proxy->SetInterceptCB(OnSubmitCB, obj.get());
    }
    obj->queue_event_callback_ = callback;
    obj->queue_id = current_queue_id;
    ++current_queue_id;

    if (get_persistent().callbacks_.create != NULL) {
      status = get_persistent().callbacks_.create(*queue, get_persistent().callback_data_);
    }

    in_create_call_ = false;
    get_persistent().obj_map_[(uint64_t)(*queue)] = std::move(obj);
    return status;
  }

  static hsa_status_t QueueCreate(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                  void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                   void* data),
                                  void* data, uint32_t private_segment_size,
                                  uint32_t group_segment_size, hsa_queue_t** queue) {
    return InterceptQueueCreate(agent, size, type, callback, data, private_segment_size,
                                group_segment_size, queue, false);
  }

  static hsa_status_t QueueCreateTracked(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                         void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                          void* data),
                                         void* data, uint32_t private_segment_size,
                                         uint32_t group_segment_size, hsa_queue_t** queue) {
    return InterceptQueueCreate(agent, size, type, callback, data, private_segment_size,
                                group_segment_size, queue, true);
  }

  static hsa_status_t QueueDestroy(hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (GetObj(queue) == nullptr) {
      /* This isn't an intercept queue managed by the rocprofiler, call the original function to
         destroy this queue. */
      return hsa_queue_destroy_fn(queue);
    }

    if (get_persistent().callbacks_.destroy != NULL) {
      status = get_persistent().callbacks_.destroy(queue, get_persistent().callback_data_);
    }

    if (status == HSA_STATUS_SUCCESS) {
      status = DelObj(queue);
    }

    return status;
  }

  static void OnSubmitCB_opt(const void* in_packets, uint64_t count, uint64_t user_que_idx,
                             void* data, hsa_amd_queue_intercept_packet_writer writer) {
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptQueue* obj = reinterpret_cast<InterceptQueue*>(data);
    Queue* proxy = obj->proxy_;

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

        rocprofiler_callback_data_t data = {obj->agent_info_->dev_id,
                                            obj->agent_info_->dev_index,
                                            obj->queue_,
                                            user_que_idx,
                                            obj->queue_id,
                                            completion_signal,
                                            dispatch_packet,
                                            NULL,   // kernel_name
                                            0,      // kernel_object
                                            NULL,   // kernel_code
                                            0,      // (uint32_t)syscall(__NR_gettid),
                                            NULL};  // record

        // Calling dispatch callback
        rocprofiler_group_t group = {};
        hsa_status_t status = (dispatch_callback_.load())(&data, get_persistent().callback_data_, &group);
        Context* context = reinterpret_cast<Context*>(group.context);
        // Injecting profiling start/stop packets
        if ((status == HSA_STATUS_SUCCESS) && (context != NULL)) {
          if (group.feature_count != 0) {
            if (tracker_ != NULL) {
              Group* context_group = context->GetGroup(group.index);
              const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal =
                  context_group->GetDispatchSignal();
              Tracker::Enable_opt(context_group, completion_signal);
              context_group->IncrRefsCount();
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

  static void OnSubmitCB(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data,
                         hsa_amd_queue_intercept_packet_writer writer) {
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptQueue* obj = reinterpret_cast<InterceptQueue*>(data);
    Queue* proxy = obj->proxy_;

    ////////////////////////////////////////////////
#if INTERCEPT_QUEUE_TRACE
    const uint32_t header_val = *(uint32_t*)in_packets;
    const uint32_t pid = syscall(__NR_getpid);
    const uint32_t tid = syscall(__NR_gettid);
    hsa_queue_t* qptr = obj->queue_;
    const void* slot_ptr = util::HsaRsrcFactory::GetSlotPointer(qptr, user_que_idx);
    printf("OnSubmitCB: %u:%u queue(%p:%lu) in(%p, %p, %lu) hdr(%u)\n", pid, tid, qptr,
           user_que_idx, in_packets, slot_ptr, count, header_val);
    fflush(stdout);
    print_packet(in_packets, count);
    abort();
#endif
    ////////////////////////////////////////////////

    if (submit_callback_fun_) {
      get_persistent().mutex_.lock();
      auto* callback_fun = submit_callback_fun_;
      void* callback_arg = submit_callback_arg_;
      get_persistent().mutex_.unlock();

      if (callback_fun) {
        for (uint64_t j = 0; j < count; ++j) {
          const packet_t* packet = &packets_arr[j];
          const hsa_kernel_dispatch_packet_t* dispatch_packet =
              reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);

          const char* kernel_name = NULL;
          if (GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
            uint64_t kernel_object = dispatch_packet->kernel_object;
            const amd_kernel_code_t* kernel_code = GetKernelCode(kernel_object);
            kernel_name = (GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH)
                ? QueryKernelName(kernel_object, kernel_code)
                : NULL;
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

        const bool is_serial = (k_concurrent_ == K_CONC_OFF);
        if (tracker_ != NULL) {
          tracker_entry = tracker_->Alloc(obj->agent_info_->dev_id,
                                          dispatch_packet->completion_signal, is_serial);
          if (is_serial)
            const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal =
                tracker_entry->signal;
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
        hsa_status_t status = (dispatch_callback_.load())(&data, get_persistent().callback_data_, &group);
        // Injecting profiling start/stop/read packets
        if ((status != HSA_STATUS_SUCCESS) || (group.context == NULL)) {
          if (tracker_entry != NULL) {
            if (is_serial)
              const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal =
                  tracker_entry->orig;
            tracker_->Delete(tracker_entry);
          }
        } else {
          Context* context = reinterpret_cast<Context*>(group.context);

          if (group.feature_count != 0) {
            const pkt_vector_t& start_vector = context->StartPackets(group.index);
            const pkt_vector_t& stop_vector = context->StopPackets(group.index);
            const pkt_vector_t& read_vector = context->ReadPackets(group.index);
            pkt_vector_t packets;

            if (is_serial) {  // serial
              packets = start_vector;
              packets.insert(packets.end(), *packet);
              packets.insert(packets.end(), stop_vector.begin(), stop_vector.end());
            } else {  // concurrent
              // Insert start packets once
              auto inject_start = [&packets](const pkt_vector_t& starts) mutable {
                packets = starts;
              };
              std::call_once(once_flag_, inject_start, start_vector);
              // Reads at both kernel start and end (also with barriers)
              assert(read_vector.size() >= 2 * start_vector.size());
              auto mid = read_vector.begin() + read_vector.size() / 2;
              // Read at kernel start
              packets.insert(packets.end(), read_vector.begin(), mid);
              // Kernel dispatch packet
              assert(tracker_entry != NULL);
              // Bind dispatch and barrier signals with tracker entry
              tracker_->SetHandler(tracker_entry, context->GetGroup(group.index));
              const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal =
                  context->GetGroup(group.index)->GetDispatchSignal();
              packets.insert(packets.end(), *packet);
              // Read at kernel end
              packets.insert(packets.end(), mid, read_vector.end());
            }

            if (tracker_entry != NULL) {
              Group* context_group = context->GetGroup(group.index);
              context_group->IncrRefsCount();
              tracker_->EnableContext(tracker_entry, Context::Handler,
                                      reinterpret_cast<void*>(context_group));
            }

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

  static void OnSubmitCB_ctrace(const void* in_packets, uint64_t count, uint64_t user_que_idx,
                                void* data, hsa_amd_queue_intercept_packet_writer writer) {
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptQueue* obj = reinterpret_cast<InterceptQueue*>(data);
    Queue* proxy = obj->proxy_;

    if (submit_callback_fun_) {
      get_persistent().mutex_.lock();
      auto* callback_fun = submit_callback_fun_;
      void* callback_arg = submit_callback_arg_;
      get_persistent().mutex_.unlock();

      if (callback_fun) {
        for (uint64_t j = 0; j < count; ++j) {
          const packet_t* packet = &packets_arr[j];
          const hsa_kernel_dispatch_packet_t* dispatch_packet =
              reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);

          const char* kernel_name = NULL;
          if (GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
            uint64_t kernel_object = dispatch_packet->kernel_object;
            const amd_kernel_code_t* kernel_code = GetKernelCode(kernel_object);
            kernel_name = (GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH)
                ? QueryKernelName(kernel_object, kernel_code)
                : NULL;
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
                                            NULL};

        // Calling dispatch callback
        rocprofiler_group_t group = {};
        hsa_status_t status = (dispatch_callback_.load())(&data, get_persistent().callback_data_, &group);

        // Injecting profiling start/stop packets
        if ((status == HSA_STATUS_SUCCESS) && (group.context != NULL)) {
          Context* context = reinterpret_cast<Context*>(group.context);
          const bool ctx_inactive = ck_ctx_inactive(context);

          const pkt_vector_t& start_vector = context->StartPackets(group.index);
          const pkt_vector_t& stop_vector = context->StopPackets(group.index);
          pkt_vector_t packets;
          if (ctx_inactive) packets = start_vector;
          packets.insert(packets.end(), *packet);
          if (!ctx_inactive) packets.insert(packets.end(), stop_vector.begin(), stop_vector.end());
          if (writer != NULL) {
            writer(&packets[0], packets.size());
          } else {
            proxy->Submit(&packets[0], packets.size());
          }
          to_submit = false;
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
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    if (get_persistent().callback_data_ != NULL) {
      EXC_ABORT(HSA_STATUS_ERROR, "reassigning queue callbacks - not supported");
    }
    get_persistent().callbacks_ = callbacks;
    get_persistent().callback_data_ = data;
    Start();
  }

  static void RemoveCallbacks() {
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    get_persistent().callbacks_ = {};
    Stop();
  }

  static inline void Start() {
    dispatch_callback_.store(get_persistent().callbacks_.dispatch, std::memory_order_release);
  }
  static inline void Stop() { dispatch_callback_.store(NULL, std::memory_order_relaxed); }

  static void SetSubmitCallback(rocprofiler_hsa_callback_fun_t fun, void* arg) {
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    submit_callback_fun_ = fun;
    submit_callback_arg_ = arg;
  }

  static void TrackerOn(bool on) { tracker_on_ = on; }
  static bool IsTrackerOn() { return tracker_on_; }

  static bool opt_mode_;
  static uint32_t k_concurrent_;

 private:
  static void queue_event_callback(hsa_status_t status, hsa_queue_t* queue, void* arg) {
    if (status != HSA_STATUS_SUCCESS) {
      uint32_t* read_ptr32 = (uint32_t*)util::HsaRsrcFactory::GetReadPointer(queue);
      print_packet(read_ptr32, 1);
      EXC_ABORT(status, "queue(" << queue << ":" << read_ptr32 << ")");
    }
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

  static const char* QueryKernelName(uint64_t kernel_object, const amd_kernel_code_t* kernel_code) {
    const uint16_t kernel_object_flag = *((uint64_t*)kernel_code + 1);
    if (kernel_object_flag == 0) {
      if (!util::HsaRsrcFactory::IsExecutableTracking()) {
        EXC_ABORT(HSA_STATUS_ERROR,
                  "Error: V3 code object detected - code objects tracking should be enabled\n");
      }
    }
    const char* kernel_symname = (util::HsaRsrcFactory::IsExecutableTracking())
        ? util::HsaRsrcFactory::GetKernelNameRef(kernel_object)
        : GetKernelName(kernel_code->runtime_loader_kernel_symbol);
    return kernel_symname;
  }

  // method to get an intercept queue object
  static InterceptQueue* GetObj(const hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    InterceptQueue* obj = NULL;
    obj_map_t::const_iterator it = get_persistent().obj_map_.find((uint64_t)queue);
    if (it != get_persistent().obj_map_.end()) {
      obj = it->second.get();
      if (obj)
        assert(queue == obj->queue_);
    }
    return obj;
  }

  // method to delete an intercept queue object
  static hsa_status_t DelObj(const hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(get_persistent().mutex_);
    auto& obj_map_ = get_persistent().obj_map_;
    hsa_status_t status = HSA_STATUS_ERROR;
    obj_map_t::iterator it = obj_map_.find((uint64_t)queue);
    if (it != obj_map_.end()) {
      assert(queue == it->second->queue_);
      obj_map_.erase(it);
      status = HSA_STATUS_SUCCESS;
    }

    return status;
  }

  static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;

  typedef struct
  {
    mutex_t mutex_{};
    obj_map_t obj_map_{};
    rocprofiler_queue_callbacks_t callbacks_{};
    void* callback_data_ = nullptr;
  } persistent_objects_t;

  static persistent_objects_t& get_persistent() {
    static auto* obj = new persistent_objects_t{};
    return *obj;
  }

  static std::atomic<rocprofiler_callback_t> dispatch_callback_;

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

  static std::once_flag once_flag_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_INTERCEPT_QUEUE_H
