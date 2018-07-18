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

#ifndef _SRC_CORE_PROF_LISTENER_H
#define _SRC_CORE_PROF_LISTENER_H

#include <amd_hsa_kernel_code.h>
#include <cxxabi.h>
#include <dlfcn.h>

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

class ProfListener {
 public:
  static void Callback(const void* in_packets, uint64_t count, uint64_t user_que_idx, void* data,
                         hsa_amd_queue_intercept_packet_writer writer) {
    const packet_t* packets_arr = reinterpret_cast<const packet_t*>(in_packets);
    InterceptQueue* obj = reinterpret_cast<InterceptQueue*>(data);
    Queue* proxy = obj->proxy_;

    // Travers input packets
    for (uint64_t j = 0; j < count; ++j) {
      const packet_t* packet = &packets_arr[j];
      bool to_submit = true;

      // Checking for dispatch packet type
      if ((GetHeaderType(packet) == HSA_PACKET_TYPE_KERNEL_DISPATCH) && (dispatch_callback_ != NULL)) {
        const hsa_kernel_dispatch_packet_t* dispatch_packet =
            reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);

        // Adding kernel timing tracker
        const rocprofiler_dispatch_record_t* record = NULL;
        if (tracker_ != NULL) {
          const auto* entry = tracker_->Add(obj->agent_info_->dev_id, dispatch_packet->completion_signal);
          const_cast<hsa_kernel_dispatch_packet_t*>(dispatch_packet)->completion_signal = entry->signal;
          record = entry->record;
        }

        // Prepareing dispatch callback data
        const char* kernel_name = GetKernelName(dispatch_packet);
        rocprofiler_callback_data_t data = {obj->agent_info_->dev_id,
                                            obj->agent_info_->dev_index,
                                            obj->queue_,
                                            user_que_idx,
                                            dispatch_packet,
                                            kernel_name,
                                            record};

        // Calling dispatch callback
        rocprofiler_group_t group = {};
        hsa_status_t status = dispatch_callback_(&data, callback_data_, &group);
        free(const_cast<char*>(kernel_name));
        // Injecting profiling start/stop packets
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

  static void SetProfCallbacks(rocprofiler_callback_t dispatch_callback, queue_callback_t destroy_callback, void* data) {
    std::lock_guard<mutex_t> lck(mutex_);
    callback_data_ = data;
    dispatch_callback_ = dispatch_callback;
    destroy_callback_ = destroy_callback;
  }

  void TrackerOn(bool on) { tracker_on_ = on; }
  bool IsTrackerOn() { return tracker_on_; }

 private:
  static hsa_packet_type_t GetHeaderType(const packet_t* packet) {
    const packet_word_t* header = reinterpret_cast<const packet_word_t*>(packet);
    return static_cast<hsa_packet_type_t>((*header >> HSA_PACKET_HEADER_TYPE) & header_type_mask);
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

  InterceptQueue(const hsa_agent_t& agent, hsa_queue_t* const queue, ProxyQueue* proxy) :
    queue_(queue),
    proxy_(proxy)
  {
    agent_info_ = util::HsaRsrcFactory::Instance().GetAgentInfo(agent);
  }
  ~InterceptQueue() { ProxyQueue::Destroy(proxy_); }

  static mutex_t mutex_;
  static const packet_word_t header_type_mask = (1ul << HSA_PACKET_HEADER_WIDTH_TYPE) - 1;
  static rocprofiler_callback_t dispatch_callback_;
  static queue_callback_t destroy_callback_;
  static void* callback_data_;
  static obj_map_t* obj_map_;
  static const char* kernel_none_;
  static uint64_t timeout_;
  static Tracker* tracker_;
  static bool tracker_on_;
  static bool in_constr_call_;

  const util::AgentInfo* agent_info_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_PROF_LISTENER_H
