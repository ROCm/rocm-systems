#ifndef _SRC_CORE_INTERCEPT_QUEUE_H
#define _SRC_CORE_INTERCEPT_QUEUE_H

#include <amd_hsa_kernel_code.h>
#include <cxxabi.h>
#include <dlfcn.h>

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>

#include "core/context.h"
#include "core/proxy_queue.h"
#include "core/types.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
extern decltype(hsa_queue_create)* hsa_queue_create_fn;
extern decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;

class InterceptQueue {
 public:
  typedef std::recursive_mutex mutex_t;
  typedef std::map<uint64_t, InterceptQueue*> obj_map_t;
  typedef hsa_status_t (*queue_callback_t)(hsa_queue_t*, void* data);

  static void HsaIntercept(HsaApiTable* table);

  static hsa_status_t QueueCreate(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                  void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                   void* data),
                                  void* data, uint32_t private_segment_size,
                                  uint32_t group_segment_size, hsa_queue_t** queue) {
    hsa_status_t status = HSA_STATUS_ERROR;
    std::lock_guard<mutex_t> lck(mutex_);

    if (!obj_map_) obj_map_ = new obj_map_t;

    ProxyQueue* proxy = ProxyQueue::Create(agent, size, type, callback, data, private_segment_size,
                                           group_segment_size, queue, &status);
    if (status == HSA_STATUS_SUCCESS) {
      InterceptQueue* obj = new InterceptQueue(agent, *queue, proxy);
      (*obj_map_)[(uint64_t)(*queue)] = obj;
      status = proxy->SetInterceptCB(OnSubmitCB, obj);
    }

    if (status != HSA_STATUS_SUCCESS) abort();

    return status;
  }

  static hsa_status_t QueueDestroy(hsa_queue_t* queue) {
    std::lock_guard<mutex_t> lck(mutex_);
    hsa_status_t status = HSA_STATUS_ERROR;

   if (destroy_callback_ != NULL) {
     status = destroy_callback_(queue, callback_data_);
     if (status != HSA_STATUS_SUCCESS) return status;
   }

    obj_map_t::iterator it = obj_map_->find((uint64_t)queue);
    if (it != obj_map_->end()) {
      const InterceptQueue* obj = it->second;
      assert(queue == obj->queue_);
      delete obj;
      obj_map_->erase(it);
      status = HSA_STATUS_SUCCESS;
    }

    return status;
  }

  static void OnSubmitCB(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data,
                         hsa_amd_queue_intercept_packet_writer writer) {
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptQueue* obj = reinterpret_cast<InterceptQueue*>(data);
    Queue* proxy = obj->proxy_;

    for (uint64_t j = 0; j < count; ++j) {
      bool to_submit = true;
      const packet_t* packet = &packets_arr[j];

      if ((GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) && (dispatch_callback_ != NULL)) {
        rocprofiler_group_t group = {};
        const hsa_kernel_dispatch_packet_t* dispatch_packet =
            reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);
        const char* kernel_name = GetKernelName(dispatch_packet);
        rocprofiler_callback_data_t data = {obj->agent_info_->dev_id,
                                            obj->agent_info_->dev_index,
                                            obj->queue_,
                                            user_que_idx,
                                            dispatch_packet->kernel_object,
                                            kernel_name};
        hsa_status_t status = dispatch_callback_(&data, callback_data_, &group);
        free(const_cast<char*>(kernel_name));
        if ((status == HSA_STATUS_SUCCESS) && (group.context != NULL)) {
          Context* context = reinterpret_cast<Context*>(group.context);
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

      if (to_submit) {
        if (writer != NULL) {
          writer(packet, 1);
        } else {
          proxy->Submit(packet, 1);
        }
      }

      packet += 1;
    }
  }

  static void SetCallbacks(rocprofiler_callback_t dispatch_callback, queue_callback_t destroy_callback, void* data) {
    std::lock_guard<mutex_t> lck(mutex_);
    callback_data_ = data;
    dispatch_callback_ = dispatch_callback;
    destroy_callback_ = destroy_callback;
  }

 private:
  InterceptQueue(const hsa_agent_t& agent, hsa_queue_t* const queue, ProxyQueue* proxy) :
    queue_(queue),
    proxy_(proxy)
  {
    agent_info_ = util::HsaRsrcFactory::Instance().GetAgentInfo(agent);
  }
  ~InterceptQueue() { ProxyQueue::Destroy(proxy_); }

  static packet_word_t GetHeaderType(const packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return (*header >> HSA_PACKET_HEADER_TYPE) & header_type_mask;
  }

  static const char* GetKernelName(const hsa_kernel_dispatch_packet_t* dispatch_packet) {
    const amd_kernel_code_t* kernel_code = NULL;
    hsa_status_t status =
        util::HsaRsrcFactory::Instance().LoaderApi()->hsa_ven_amd_loader_query_host_address(
            reinterpret_cast<const void*>(dispatch_packet->kernel_object),
            reinterpret_cast<const void**>(&kernel_code));
    if (HSA_STATUS_SUCCESS != status) {
      kernel_code = reinterpret_cast<amd_kernel_code_t*>(dispatch_packet->kernel_object);
    }
    amd_runtime_loader_debug_info_t* dbg_info = reinterpret_cast<amd_runtime_loader_debug_info_t*>(
        kernel_code->runtime_loader_kernel_symbol);
    const char* kernel_name = (dbg_info != NULL) ? dbg_info->kernel_name : NULL;

    // Kernel name is mangled name
    // apply __cxa_demangle() to demangle it
    const char* funcname = NULL;
    if (kernel_name != NULL) {
      size_t funcnamesize = 0;
      int status;
      const char* ret = abi::__cxa_demangle(kernel_name, NULL, &funcnamesize, &status);
      funcname = (ret != 0) ? ret : strdup(kernel_name);
    }
    if (funcname == NULL) funcname = strdup(kernel_none_);

    return funcname;
  }

  static mutex_t mutex_;
  static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
  static rocprofiler_callback_t dispatch_callback_;
  static queue_callback_t destroy_callback_;
  static void* callback_data_;
  static obj_map_t* obj_map_;
  static const char* kernel_none_;

  hsa_queue_t* const queue_;
  ProxyQueue* const proxy_;
  const util::AgentInfo* agent_info_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_INTERCEPT_QUEUE_H
