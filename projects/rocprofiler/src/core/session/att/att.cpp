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

#include "att.h"
#include <cassert>
#include <atomic>

#include "src/core/hsa/packets/packets_generator.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/core/isa_capture/code_object_track.hpp"

namespace rocprofiler {

namespace att {

AttTracer::AttTracer(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
                     rocprofiler_session_id_t session_id)
    : buffer_id_(buffer_id), filter_id_(filter_id), session_id_(session_id) {}

void AttTracer::AddPendingSignals(
    size_t writer_id,
    uint64_t kernel_object,
    const hsa_signal_t& original_completion_signal,
    const hsa_signal_t& new_completion_signal,
    rocprofiler_session_id_t session_id,
    rocprofiler_buffer_id_t buffer_id,
    hsa_ven_amd_aqlprofile_profile_t* profile,
    rocprofiler_kernel_properties_t kernel_properties,
    uint32_t thread_id, uint64_t queue_index
) {
  std::lock_guard<std::mutex> lock(sessions_pending_signals_lock_);
  if (bIsSessionDestroying.load())
    return;

  auto pending = sessions_pending_signals_.find(writer_id);
  if (pending == sessions_pending_signals_.end())
    pending = sessions_pending_signals_.emplace(writer_id, std::vector<att_pending_signal_t>()).first;

  pending->second.emplace_back(att_pending_signal_t{
    kernel_object,
    original_completion_signal,
    new_completion_signal,
    session_id,
    buffer_id,
    profile,
    kernel_properties,
    thread_id,
    queue_index
  });
}

std::vector<att_pending_signal_t> AttTracer::MovePendingSignals(size_t writer_id)
{
  std::lock_guard<std::mutex> lock(sessions_pending_signals_lock_);
  auto it = sessions_pending_signals_.find(writer_id);
  if (it == sessions_pending_signals_.end())
  {
    rocprofiler::warning("writer_id is not found in the pending_signals");
    return {};
  }

  auto move_pending = std::move(it->second);
  sessions_pending_signals_.erase(writer_id);
  if (bIsSessionDestroying.load() && sessions_pending_signals_.size() == 0)
    has_session_pending_cv.notify_all();
  return move_pending;
}

#define DEFAULT_ATT_BUFFER_SIZE 0x40000000

std::pair<hsa_ven_amd_aqlprofile_profile_t*, rocprofiler_codeobj_capture_mode_t>
AttTracer::ProcessATTParams(
  hsa_ext_amd_aql_pm4_packet_t& start_packet,
  hsa_ext_amd_aql_pm4_packet_t& stop_packet,
  queue::Queue& queue_info,
  rocprofiler::HSAAgentInfo& agentInfo
) {
  std::vector<hsa_ven_amd_aqlprofile_parameter_t> att_params;
  int num_att_counters = 0;
  uint32_t att_buffer_size = DEFAULT_ATT_BUFFER_SIZE;
  rocprofiler_codeobj_capture_mode_t capture_mode = ROCPROFILER_CAPTURE_SYMBOLS_ONLY;

  for (rocprofiler_att_parameter_t& param : att_parameters_data) {
    switch (param.parameter_name) {
      case ROCPROFILER_ATT_PERFCOUNTER_NAME:
        break;
      case ROCPROFILER_ATT_CAPTURE_MODE:
        capture_mode = static_cast<rocprofiler_codeobj_capture_mode_t>(param.value);
        break;
      case ROCPROFILER_ATT_BUFFER_SIZE:
        att_buffer_size =
            std::max(96l << 10l, std::min(int64_t(param.value) << 20l, (1l << 32l) - (3l << 20)));
        break;  // Clip to [96KB, 4GB)
      case ROCPROFILER_ATT_PERFCOUNTER:
        num_att_counters += 1;
        break;
      default:
        att_params.push_back(
            {static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(param.parameter_name)),
             param.value});
    }
  }

  if (att_counters_names.size() > 0) {
    MetricsDict* metrics_dict_ = MetricsDict::Create(&agentInfo);

    for (const std::string& counter_name : att_counters_names) {
      const Metric* metric = metrics_dict_->Get(counter_name);
      const BaseMetric* base = dynamic_cast<const BaseMetric*>(metric);
      if (!base) rocprofiler::fatal("Invalid base metric value: %s\n", counter_name.c_str());

      std::vector<const counter_t*> counters;
      base->GetCounters(counters);
      hsa_ven_amd_aqlprofile_event_t event = counters[0]->event;
      if (event.block_name != HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ)
        rocprofiler::fatal("Only events from the SQ block can be selected for ATT.\n");

      att_params.push_back(
          {static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(ROCPROFILER_ATT_PERFCOUNTER)),
           event.counter_id | (event.counter_id ? (0xF << 24) : 0)});
      num_att_counters += 1;
    }

    hsa_ven_amd_aqlprofile_parameter_t zero_perf = {
        static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(ROCPROFILER_ATT_PERFCOUNTER)), 0};

    // Fill other perfcounters with 0's
    for (; num_att_counters < 16; num_att_counters++) att_params.push_back(zero_perf);
  }
  // Get the PM4 Packets using packets_generator
  return {Packet::GenerateATTPackets(
                                    queue_info.GetCPUAgent(),
                                    queue_info.GetGPUAgent(),
                                    att_params,
                                    &start_packet,
                                    &stop_packet,
                                    att_buffer_size
          ),
          capture_mode};

}

bool AttTracer::ATTWriteInterceptor(
  const void* packets,
  uint64_t pkt_count,
  uint64_t user_pkt_index,
  queue::Queue& queue,
  hsa_amd_queue_intercept_packet_writer writer,
  rocprofiler_buffer_id_t buffer_id
) {
  bool IsSingleDispatchMode = kernel_profile_dispatch_ids.size() == 0;

  if (session_id_.handle == 0 ||
    pkt_count == 0 ||
    att_parameters_data.size() == 0
  ) return false;

  if (IsSingleDispatchMode)
    return ATTSingleWriteInterceptor(packets, pkt_count, user_pkt_index, queue, writer, buffer_id);
  else
    return ATTContiguousWriteInterceptor(packets, pkt_count, queue, writer, buffer_id);
}

void AttTracer::signalAsyncHandlerATT(const hsa_signal_t& signal, void* data) {
  hsa_status_t status =
      HSASupport_Singleton::GetInstance().GetAmdExtTable().hsa_amd_signal_async_handler_fn(
          signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandlerATT, data);
  if (status != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("Error: hsa_amd_signal_async_handler for ATT failed");
}

bool AttTracer::AsyncSignalHandlerATT(hsa_signal_value_t /* signal */, void* data)
{
  auto queue_info_session = static_cast<queue::queue_info_session_t*>(data);
  rocprofiler::ROCProfiler_Singleton& rocprofiler_singleton =
      rocprofiler::ROCProfiler_Singleton::GetInstance();

  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();

  if (!queue_info_session || !rocprofiler_singleton.GetSession(queue_info_session->session_id) ||
      !rocprofiler_singleton.GetSession(queue_info_session->session_id)->GetAttTracer())
    return true;

  rocprofiler::Session* session = rocprofiler_singleton.GetSession(queue_info_session->session_id);
  std::lock_guard<std::mutex> lock(session->GetSessionLock());
  rocprofiler::att::AttTracer* att_tracer = session->GetAttTracer();

  if (!session->GetAttTracer()) return true;
  auto pending_signals = att_tracer->MovePendingSignals(queue_info_session->writer_id);

  for (auto& pending : pending_signals)
  {
    rocprofiler_record_att_tracer_t record{};
    record.kernel_id = rocprofiler_kernel_id_t{pending.kernel_descriptor};
    record.gpu_id = rocprofiler_agent_id_t{(uint64_t)queue_info_session->gpu_index};
    record.kernel_properties = pending.kernel_properties;
    record.thread_id = rocprofiler_thread_id_t{pending.thread_id};
    record.queue_idx = rocprofiler_queue_index_t{pending.queue_index};
    record.queue_id = rocprofiler_queue_id_t{queue_info_session->queue_id};
    record.writer_id = queue_info_session->writer_id;

    if (pending.profile)
      AddAttRecord(&record, queue_info_session->agent, pending);

    // July/01/2023 -> Changed this to queue_info_session->writer_id
    // so we can correlate to dispatches. kernel_id already has the descriptor.
    record.header = {ROCPROFILER_ATT_TRACER_RECORD,
                      rocprofiler_record_id_t{pending.kernel_descriptor}};

    record.intercept_list = codeobj_record::get_capture(record.header.id);
    std::atomic_thread_fence(std::memory_order_release);

    if (pending.session_id.handle == 0)
      pending.session_id = rocprofiler_singleton.GetCurrentSessionId();

    if (session->FindBuffer(pending.buffer_id)) {
      Memory::GenericBuffer* buffer = session->GetBuffer(pending.buffer_id);
      buffer->AddRecord(record);
      buffer->Flush();
    }
    codeobj_record::free_capture(record.header.id);

    hsa_status_t status = hsasupport_singleton.GetAmdExtTable().hsa_amd_memory_pool_free_fn(
        (pending.profile->output_buffer.ptr));
    if (status != HSA_STATUS_SUCCESS)
      rocprofiler::warning("Error: Couldn't free output buffer memory");

    status = hsasupport_singleton.GetAmdExtTable().hsa_amd_memory_pool_free_fn(
        (pending.profile->command_buffer.ptr));
    if (status != HSA_STATUS_SUCCESS)
      rocprofiler::warning("Error: Couldn't free command buffer memory");

    if (pending.profile->parameters)
      delete[] pending.profile->parameters;
    delete pending.profile;
  }
  delete queue_info_session;
  return false;
}

void AttTracer::AddAttRecord(
  rocprofiler_record_att_tracer_t* record,
  hsa_agent_t gpu_agent,
  att_pending_signal_t& pending
) {
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  HSAAgentInfo agent_info = hsasupport_singleton.GetHSAAgentInfo(gpu_agent.handle);
  std::vector<hsa_ven_amd_aqlprofile_info_data_t> data;
  hsa_status_t status =
      hsa_ven_amd_aqlprofile_iterate_data(pending.profile, attTraceDataCallback, &data);

  if ((status & HSA_STATUS_ERROR_OUT_OF_RESOURCES) == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
    rocprofiler::warning("Warning: ATT buffer full!\n");
  if ((status & HSA_STATUS_ERROR_EXCEPTION) == HSA_STATUS_ERROR_EXCEPTION)
    rocprofiler::warning("Warning: ATT received a UTC memory error!\n");
  if (status == HSA_STATUS_ERROR) fatal("Thread Trace Error!");

  size_t max_sample_id = 0;
  for (auto& trace_data_it : data)
    max_sample_id = std::max<size_t>(max_sample_id, trace_data_it.sample_id+1);

  // Allocate memory for shader_engine_data
  record->shader_engine_data_count = max_sample_id;
  record->shader_engine_data = static_cast<rocprofiler_record_se_att_data_t*>(calloc(
    max_sample_id,
    sizeof(rocprofiler_record_se_att_data_t)
  ));

  // iterate over the trace data collected from each shader engine
  for (auto& trace_data_it : data)
  {
    auto& trace = trace_data_it.trace_data;

    void* buffer = NULL;
    if (trace.ptr && trace.size) {
      // Allocate buffer on CPU to copy out trace data
      buffer = Packet::AllocateSysMemory(gpu_agent, trace.size, &agent_info.cpu_pool_);
      if (buffer == NULL) fatal("Trace data buffer allocation failed");

      auto status =
          hsasupport_singleton.GetCoreApiTable().hsa_memory_copy_fn(buffer, trace.ptr, trace.size);
      if (status != HSA_STATUS_SUCCESS) fatal("Trace data memcopy to host failed");

      record->shader_engine_data[trace_data_it.sample_id].buffer_ptr = buffer;
      record->shader_engine_data[trace_data_it.sample_id].buffer_size = trace.size;
      // TODO: clear output buffers after copying
    }
  }
}

hsa_status_t AttTracer::attTraceDataCallback(
  hsa_ven_amd_aqlprofile_info_type_t info_type,
  hsa_ven_amd_aqlprofile_info_data_t* info_data,
  void* data
) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  auto* passed_data = reinterpret_cast<std::vector<hsa_ven_amd_aqlprofile_info_data_t>*>(data);
  passed_data->push_back(*info_data);
  // TODO: clear output buffers after copying
  // either copy here or in ::AddAttRecord

  return status;
}

void AttTracer::WaitForPendingAndDestroy()
{
  bIsSessionDestroying.store(true);
  std::unique_lock<std::mutex> lk(sessions_pending_signals_lock_);
  if (sessions_pending_signals_.size() == 0)
    return;

  has_session_pending_cv.wait_for(lk, std::chrono::seconds(2), [this] () {
    return this->sessions_pending_signals_.size() == 0;
  });
}

std::unordered_map<uint64_t, ATTRecordSignal> AttTracer::pending_stop_packets;
std::mutex AttTracer::att_enable_disable_mutex;


}  // namespace att

}  // namespace rocprofiler
