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

#ifndef SRC_CORE_SESSION_ATT_ATT_H_
#define SRC_CORE_SESSION_ATT_ATT_H_

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>

#include "rocprofiler.h"
#include "src/utils/helper.h"
#include "src/core/proxy_queue.h"
#include "src/core/hsa/hsa_support.h"
#include "src/core/hsa/queues/queue.h"
#include "att_header.h"

namespace rocprofiler {

typedef struct {
  uint64_t kernel_descriptor;
  hsa_signal_t original_signal;
  hsa_signal_t new_signal;
  rocprofiler_session_id_t session_id;
  rocprofiler_buffer_id_t buffer_id;
  hsa_ven_amd_aqlprofile_profile_t* profile;
  rocprofiler_kernel_properties_t kernel_properties;
  uint32_t thread_id;
  uint64_t queue_index;
} att_pending_signal_t;


namespace att {

struct ATTRecordSignal
{
  size_t record_id;
  size_t writer_id;
  size_t last_kernel_exec;
  rocprofiler_session_id_t session_id_snapshot;
  hsa_ext_amd_aql_pm4_packet_t stop_packet;
};

class AttTracer {
public:
  AttTracer(
    rocprofiler_buffer_id_t buffer_id,
    rocprofiler_filter_id_t filter_id,
    rocprofiler_session_id_t session_id
  );

  void AddPendingSignals(size_t writer_id, uint64_t kernel_object,
                         const hsa_signal_t& original_completion_signal,
                         const hsa_signal_t& new_completion_signal,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id,
                         hsa_ven_amd_aqlprofile_profile_t* profile,
                         rocprofiler_kernel_properties_t kernel_properties, uint32_t thread_id,
                         uint64_t queue_index);

  bool InsertPacketStart(
    std::vector<packet_t>& transformed_packets,
    queue::Queue& queue_info,
    size_t writer_id,
    rocprofiler_buffer_id_t buffer_id,
    size_t stop_location,
    const std::string& kernel_name
  );

  void InsertPacketStop(
    std::vector<packet_t>& transformed,
    const ATTRecordSignal& rsignal,
    queue::Queue& queue,
    uint64_t agent_handle
  );

  std::vector<att_pending_signal_t> MovePendingSignals(size_t writer_id);

  bool ATTWriteInterceptor(
    const void* packets,
    uint64_t pkt_count,
    uint64_t user_pkt_index,
    queue::Queue& queue_info,
    hsa_amd_queue_intercept_packet_writer writer,
    rocprofiler_buffer_id_t buffer_id
  );

  void InsertMarker(
    std::vector<packet_t>& transformed_packets,
    hsa_agent_t agent,
    uint32_t data,
    hsa_ven_amd_aqlprofile_att_marker_channel_t channel
  );
  void InsertUnloadMarker(
    std::vector<packet_t>& transformed_packets,
    hsa_agent_t agent,
    uint32_t data
  );
  void InsertLoadMarker(
    std::vector<packet_t>& transformed_packets,
    hsa_agent_t agent,
    rocprofiler_intercepted_codeobj_t codeobj,
    bool bFromStart
  );

  void SetParameters(const std::vector<rocprofiler_att_parameter_t>& params) {
    att_parameters_data = params;
  }
  void SetDispatchIds(const std::vector<std::pair<uint64_t,uint64_t>>& ids) {
    kernel_profile_dispatch_ids = ids;
  }
  void SetCountersNames(const std::vector<std::string>& names) {
    att_counters_names = names;
  }
  void SetKernelsNames(const std::vector<std::string>& names) {
    kernel_profile_names = names;
  }
  std::optional<std::pair<size_t, size_t>> RequiresStartPacket(size_t rstart, size_t size);

  static void signalAsyncHandlerATT(const hsa_signal_t& signal, void* data);
  static bool AsyncSignalHandlerATT(hsa_signal_value_t /* signal */, void* data);

  static hsa_status_t attTraceDataCallback(
    hsa_ven_amd_aqlprofile_info_type_t info_type,
    hsa_ven_amd_aqlprofile_info_data_t* info_data,
    void* data
  );

  bool HasActiveTracerATT(uint64_t agent_handle) const {
    return pending_stop_packets.find(agent_handle) != pending_stop_packets.end();
  }

  void WaitForPendingAndDestroy();

protected:
  using packet_t = hsa_ext_amd_aql_pm4_packet_t;
  static std::unordered_map<uint64_t, ATTRecordSignal> pending_stop_packets;
  static std::mutex att_enable_disable_mutex;

private:
  uint32_t codeobj_event_cnt = 0;

  static void AddAttRecord(
    rocprofiler_record_att_tracer_t* record,
    hsa_agent_t gpu_agent,
    att_pending_signal_t& pending
  );

  std::pair<hsa_ven_amd_aqlprofile_profile_t*, rocprofiler_codeobj_capture_mode_t>
  ProcessATTParams(
    hsa_ext_amd_aql_pm4_packet_t& start_packet,
    hsa_ext_amd_aql_pm4_packet_t& stop_packet,
    queue::Queue& queue_info,
    rocprofiler::HSAAgentInfo& agentInfo
  );

  bool ATTSingleWriteInterceptor(
    const void* packets,
    uint64_t pkt_count,
    uint64_t user_pkt_index,
    queue::Queue& queue_info,
    hsa_amd_queue_intercept_packet_writer writer,
    rocprofiler_buffer_id_t buffer_id
  );

  bool ATTContiguousWriteInterceptor(
    const void* packets,
    uint64_t pkt_count,
    queue::Queue& queue_info,
    hsa_amd_queue_intercept_packet_writer writer,
    rocprofiler_buffer_id_t buffer_id
  );

  static void CreateSignal(uint32_t attribute, hsa_signal_t* signal) {
    HSASupport_Singleton::GetInstance().CreateSignal(attribute, signal);
  }

  std::pair<std::vector<bool>, bool> GetAllowedProfilesList(const void* packets, int pkt_count);

  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_filter_id_t filter_id_;
  rocprofiler_session_id_t session_id_;
  std::atomic<size_t> WRITER_ID{0};

  std::vector<std::string> kernel_profile_names;
  std::vector<std::pair<uint64_t,uint64_t>> kernel_profile_dispatch_ids;
  std::vector<std::string> att_counters_names;
  std::vector<rocprofiler_att_parameter_t> att_parameters_data;

  std::mutex sessions_pending_signals_lock_;
  std::map<size_t, std::vector<att_pending_signal_t>> sessions_pending_signals_;
  std::condition_variable has_session_pending_cv;
  std::atomic<bool> bIsSessionDestroying{false};

  rocprofiler_record_id_t capture_id;
  std::unordered_set<size_t> active_capture_event_ids;
};

}  // namespace att

}  // namespace rocprofiler


#endif  // SRC_CORE_SESSION_ATT_ATT_H_
