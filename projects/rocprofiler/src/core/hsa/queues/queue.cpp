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

#include "queue.h"

#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "src/api/rocmtool.h"
#include "src/core/hsa/packets/packets_generator.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/helper.h"

#define __NR_gettid 186
#define MAX_ATT_PROFILES 16

std::mutex sessions_pending_signal_lock;

namespace rocmtools {

std::atomic<uint32_t> ACTIVE_INTERRUPT_SIGNAL_COUNT{0};

uint32_t GetCurrentActiveInterruptSignalsCount() {
  return ACTIVE_INTERRUPT_SIGNAL_COUNT.load(std::memory_order_relaxed);
}

typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> pmc_callback_data_t;

static inline bool IsEventMatch(const hsa_ven_amd_aqlprofile_event_t& event1,
                                const hsa_ven_amd_aqlprofile_event_t& event2) {
  return (event1.block_name == event2.block_name) && (event1.block_index == event2.block_index) &&
      (event1.counter_id == event2.counter_id);
}

typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> att_trace_callback_data_t;

static std::mutex ksymbol_map_lock;
static std::map<uint64_t, std::string>* ksymbols;
static std::atomic<bool> ksymbols_flag{true};
void AddKernelName(uint64_t handle, std::string name) {
  std::lock_guard<std::mutex> lock(ksymbol_map_lock);
  ksymbols->emplace(handle, name);
}
void RemoveKernelName(uint64_t handle) {
  std::lock_guard<std::mutex> lock(ksymbol_map_lock);
  ksymbols->erase(handle);
}
std::string GetKernelNameFromKsymbols(uint64_t handle) {
  std::lock_guard<std::mutex> lock(ksymbol_map_lock);
  if(ksymbols->find(handle)!=ksymbols->end())
    return ksymbols->at(handle);
  else
    return "Unknown Kernel!";
}

static std::mutex kernel_names_map_lock;
static std::map<std::string, std::vector<uint64_t>>* kernel_names;
static std::atomic<bool> kernel_names_flag{true};
void AddKernelNameWithDispatchID(std::string name, uint64_t id) {
  std::lock_guard<std::mutex> lock(kernel_names_map_lock);
  if(kernel_names->find(name) == kernel_names->end())
    kernel_names->emplace(name, std::vector<uint64_t>());
  kernel_names->at(name).push_back(id);
}
std::string GetKernelNameUsingDispatchID(uint64_t given_id) {
  std::lock_guard<std::mutex> lock(kernel_names_map_lock);
  for(auto kernel_name : (*kernel_names)) {
    for(auto dispatch_id : kernel_name.second) {
      if(dispatch_id == given_id)
        return kernel_name.first;
    }
  }
  return "Unknown Kernel!";
}

void InitKsymbols() {
  if (ksymbols_flag.load(std::memory_order_relaxed)) {
    {
      std::lock_guard<std::mutex> lock(ksymbol_map_lock);
      ksymbols = new std::map<uint64_t, std::string>();
      ksymbols_flag.exchange(false, std::memory_order_release);
    }
    {
      std::lock_guard<std::mutex> lock(kernel_names_map_lock);
      kernel_names = new std::map<std::string, std::vector<uint64_t>>();
      kernel_names_flag.exchange(false, std::memory_order_release);
    }
  }
}
void FinitKsymbols() {
  if (!ksymbols_flag.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(ksymbol_map_lock);
    ksymbols->clear();
    delete ksymbols;
    ksymbols_flag.exchange(true, std::memory_order_release);
  }
  if (!kernel_names_flag.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(kernel_names_map_lock);
    kernel_names->clear();
    delete kernel_names;
    kernel_names_flag.exchange(true, std::memory_order_release);
  }
}



struct kernel_descriptor_t {
  uint8_t reserved0[16];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint8_t reserved2[6];
};
// AMD Compute Program Resource Register Three.
typedef uint32_t amd_compute_pgm_rsrc_three32_t;
enum amd_compute_gfx9_pgm_rsrc_three_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET, 0, 5),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TG_SPLIT, 16, 1)
};
enum amd_compute_gfx10_gfx11_pgm_rsrc_three_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_SHARED_VGPR_COUNT, 0, 4),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_INST_PREF_SIZE, 4, 6),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_START, 10, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_END, 11, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_IMAGE_OP, 31, 1)
};

// Kernel code properties.
enum amd_kernel_code_property_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 0,
                                   1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 2, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 3, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 4, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 6, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED0, 7, 3),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 10,
                                   1),  // GFX10+
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 11, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED1, 12, 4),
};

static const kernel_descriptor_t* GetKernelCode(uint64_t kernel_object) {
  const kernel_descriptor_t* kernel_code = NULL;
  hsa_status_t status = hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_query_host_address(
      reinterpret_cast<const void*>(kernel_object), reinterpret_cast<const void**>(&kernel_code));
  if (HSA_STATUS_SUCCESS != status) {
    kernel_code = reinterpret_cast<kernel_descriptor_t*>(kernel_object);
  }
  return kernel_code;
}

static uint32_t arch_vgpr_count(Agent::AgentInfo& info, const kernel_descriptor_t& kernel_code) {
  const std::string_view& name = info.getName();
  std::string info_name(name.data(), name.size());
  if (strcmp(name.data(), "gfx90a") == 0 || strcmp(name.data(), "gfx940") == 0)
    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc3,
                             AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET) +
            1) *
        4;

  return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                           AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
          1) *
      (AMD_HSA_BITS_GET(kernel_code.kernel_code_properties,
                        AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
           ? 8
           : 4);
}
static uint32_t accum_vgpr_count(Agent::AgentInfo& info, const kernel_descriptor_t& kernel_code) {
  const std::string_view& name = info.getName();
  std::string info_name(name.data(), name.size());
  if (strcmp(info_name.c_str(), "gfx908") == 0) return arch_vgpr_count(info, kernel_code);
  if (strcmp(info_name.c_str(), "gfx90a") == 0 || strcmp(info_name.c_str(), "gfx940") == 0)
    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                             AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
            1) *
        8 -
        arch_vgpr_count(info, kernel_code);

  return 0;
}

static uint32_t sgpr_count(Agent::AgentInfo& info, const kernel_descriptor_t& kernel_code) {
  // GFX10 and later always allocate 128 sgprs.
  const std::string_view name = info.getName();
  // TODO(srnagara): Recheck the extraction of gfxip from gpu name
  const char* name_data = name.data();
  const size_t gfxip_label_len = std::min(name.size() - 2, size_t{63});
  if (gfxip_label_len > 0 && strlen(name_data) >= gfxip_label_len) {
    char gfxip[gfxip_label_len];
    memcpy(gfxip, name_data, gfxip_label_len);
    // TODO(srnagara): Check if it is hardcoded
    if (std::atoi(&gfxip[3]) >= 10) return 128;
    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                             AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT) /
                2 +
            1) *
        16;
  } else {
    return 0;
  }
}

rocprofiler_kernel_properties_t set_kernel_properties(hsa_kernel_dispatch_packet_t packet,
                                                    hsa_agent_t agent) {
  const uint64_t kernel_object = packet.kernel_object;
  rocprofiler_kernel_properties_t kernel_properties_ptr = {};
  const kernel_descriptor_t* kernel_code = GetKernelCode(kernel_object);
  uint64_t grid_size = packet.grid_size_x * packet.grid_size_y * packet.grid_size_z;
  if (grid_size > UINT32_MAX) abort();
  kernel_properties_ptr.grid_size = grid_size;
  uint64_t workgroup_size =
      packet.workgroup_size_x * packet.workgroup_size_y * packet.workgroup_size_z;
  if (workgroup_size > UINT32_MAX) abort();
  kernel_properties_ptr.workgroup_size = (uint32_t)workgroup_size;
  kernel_properties_ptr.lds_size = packet.group_segment_size;
  kernel_properties_ptr.scratch_size = packet.private_segment_size;
  Agent::AgentInfo agent_info = hsa_support::GetAgentInfo(agent.handle);
  kernel_properties_ptr.arch_vgpr_count = arch_vgpr_count(agent_info, *kernel_code);
  kernel_properties_ptr.accum_vgpr_count = accum_vgpr_count(agent_info, *kernel_code);
  kernel_properties_ptr.sgpr_count = sgpr_count(agent_info, *kernel_code);
  kernel_properties_ptr.wave_size =
      AMD_HSA_BITS_GET(kernel_code->kernel_code_properties,
                       AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
      ? 32
      : 64;
  kernel_properties_ptr.signal_handle = packet.completion_signal.handle;

  return kernel_properties_ptr;
}

namespace queue {

using rocmtools::GetROCMToolObj;

hsa_status_t pmcCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                         hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  pmc_callback_data_t* passed_data = reinterpret_cast<pmc_callback_data_t*>(data);

  pmc_callback_data_t::iterator data_it;
  if (info_data->sample_id == 0) {
    passed_data->emplace_back(*info_data);
  } else {
    for (data_it = passed_data->begin(); data_it != passed_data->end(); ++data_it) {
      if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
        if (IsEventMatch(info_data->pmc_data.event, data_it->pmc_data.event)) {
          data_it->pmc_data.result += info_data->pmc_data.result;
        }
      }
    }
  }
  return status;
}

hsa_status_t attTraceDataCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                   hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  att_trace_callback_data_t* passed_data = reinterpret_cast<att_trace_callback_data_t*>(data);
  passed_data->push_back(*info_data);
  // TODO: clear output buffers after copying
  // either copy here or in AddattRecord

  return status;
}

void AddRecordCounters(rocprofiler_record_profiler_t* record, const pending_signal_t& pending) {
  rocmtools::metrics::GetCounterData(pending.profile, pending.context->results_list);
  rocmtools::metrics::GetMetricsData(pending.context->results_map, pending.context->metrics_list);

  std::vector<rocprofiler_record_counter_instance_t> counters_vec;
  for (size_t i = 0; i < pending.context->metrics_list.size(); i++) {
    const rocmtools::Metric* metric = pending.context->metrics_list[i];
    double value = 0;
    std::string metric_name = metric->GetName();
    auto it = pending.context->results_map.find(metric_name);
    if (it != pending.context->results_map.end()) {
      value = it->second->val_double;
    }
    counters_vec.emplace_back(rocprofiler_record_counter_instance_t{
        // TODO(aelwazir): Moving to span once C++20 is adopted, strdup can be
        // removed after that
        rocprofiler_counter_id_t{rocmtools::profiler::GetCounterID(metric_name)},
        rocprofiler_record_counter_value_t{value}});
  }
  record->counters = static_cast<rocprofiler_record_counter_instance_t*>(
      malloc(counters_vec.size() * sizeof(rocprofiler_record_counter_instance_t)));
  ::memcpy(record->counters, &(counters_vec)[0],
           counters_vec.size() * sizeof(rocprofiler_record_counter_instance_t));
  record->counters_count = rocprofiler_record_counters_instances_count_t{counters_vec.size()};
}

void AddAttRecord(rocprofiler_record_att_tracer_t* record, hsa_agent_t gpu_agent,
                   att_pending_signal_t& pending) {
  att_trace_callback_data_t data;
  hsa_ven_amd_aqlprofile_iterate_data(pending.profile, attTraceDataCallback, &data);

  // Get CPU and GPU memory pools
  Packet::att_memory_pools_t* att_mem_pools = Packet::GetAttMemPools(gpu_agent);

  // Allocate memory for shader_engine_data
  record->shader_engine_data = static_cast<rocprofiler_record_se_att_data_t*>(
      calloc(data.size(), sizeof(rocprofiler_record_se_att_data_t)));

  att_trace_callback_data_t::iterator trace_data_it;

  uint32_t se_index = 0;
  // iterate over the trace data collected from each shader engine
  for (trace_data_it = data.begin(); trace_data_it != data.end(); trace_data_it++) {
    const void* data_ptr = trace_data_it->trace_data.ptr;
    const uint32_t data_size = trace_data_it->trace_data.size;
    // fprintf(arg->file, "    SE(%u) size(%u)\n", data.sample_id, data_size);

    void* buffer = NULL;
    if (data_size != 0) {
      // Allocate buffer on CPU to copy out trace data
      buffer = Packet::AllocateSysMemory(gpu_agent, data_size, &att_mem_pools->cpu_mem_pool);
      if (buffer == NULL) fatal("Trace data buffer allocation failed");

      auto status =
          rocmtools::hsa_support::GetCoreApiTable().hsa_memory_copy_fn(buffer, data_ptr, data_size);
      if (status != HSA_STATUS_SUCCESS) fatal("Trace data memcopy to host failed");

      record->shader_engine_data[se_index].buffer_ptr = buffer;
      record->shader_engine_data[se_index].buffer_size = data_size;
      ++se_index;

      // TODO: clear output buffers after copying
    }
  }
  record->shader_engine_data_count = data.size();
}

// static const size_t MEM_PAGE_BYTES = 0x1000;
// static const size_t MEM_PAGE_MASK = MEM_PAGE_BYTES - 1;
// static std::mutex begin_signal_lock;

// bool BeginSignalHandler(hsa_signal_value_t signal_value, void* data) {
//   std::lock_guard<std::mutex> lock(begin_signal_lock);
//   auto profiling_context =
//       static_cast<std::pair<rocmtools::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>*>(
//           data);
//   if (!profiling_context->first->begin_completed.load(std::memory_order_relaxed)) {
//     std::cout << "BeginSignalHandler is called" << std::endl;
//     hsa_status_t status = HSA_STATUS_ERROR;
//     size_t size = profiling_context->second->command_buffer.size;
//     size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
//     status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
//         Packet::GetCommandPool(), size, 0,
//         reinterpret_cast<void**>(&(profiling_context->second->command_buffer.ptr)));

//     // Both the CPU and GPU can access the memory
//     if (status == HSA_STATUS_SUCCESS) {
//       hsa_agent_t ag_list[1] = {profiling_context->first->gpu_agent};
//       status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
//           1, ag_list, NULL, profiling_context->second->command_buffer.ptr);

//       if (status != HSA_STATUS_SUCCESS) {
//         printf("Error: Can't allow access for both agents to Command Buffer\n");
//       }
//     } else if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
//       printf("Error: Ran out of GPU memory to allocate Command Buffer\n");
//     } else {
//       const char* hsa_err_str = NULL;
//       if (hsa_status_string(status, &hsa_err_str) != HSA_STATUS_SUCCESS) hsa_err_str = "Unknown";
//       printf("Error: Allocating command Buffer (Size=%lu) (%s)\n", size, hsa_err_str);
//     }

//     status = HSA_STATUS_ERROR;
//     size = profiling_context->second->output_buffer.size;
//     size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
//     status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
//         Packet::GetOutputPool(), size, 0,
//         reinterpret_cast<void**>(&profiling_context->second->output_buffer.ptr));

//     if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
//       printf("Error: Ran out of GPU memory to allocate Output Buffer\n");
//     }

//     if (status == HSA_STATUS_SUCCESS) {
//       hsa_agent_t ag_list[1] = {profiling_context->first->gpu_agent};
//       status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
//           1, ag_list, NULL, profiling_context->second->output_buffer.ptr);

//       if (status == HSA_STATUS_SUCCESS) {
//         memset(profiling_context->second->output_buffer.ptr, 0x0,
//                profiling_context->second->output_buffer.size);

//         // Creating the start/stop/read packets
//         status = hsa_ven_amd_aqlprofile_start(profiling_context->second,
//                                               profiling_context->first->start_packet);
//         status = hsa_ven_amd_aqlprofile_stop(profiling_context->second,
//                                              profiling_context->first->stop_packet);
//         status = hsa_ven_amd_aqlprofile_read(profiling_context->second,
//                                              profiling_context->first->read_packet);
//       } else {
//         printf("Error: Can't allow access for both agents to output Buffer\n");
//       }
//     } else {
//       const char* hsa_err_str = NULL;
//       if (hsa_status_string(status, &hsa_err_str) != HSA_STATUS_SUCCESS) hsa_err_str = "Unknown";
//       printf("Error: Allocating output Buffer (%s)\n", hsa_err_str);
//     }

//     profiling_context->first->begin_completed.exchange(true, std::memory_order_relaxed);
//   }
//   return true;
// }

bool AsyncSignalHandler(hsa_signal_value_t signal_value, void* data) {
  auto queue_info_session = static_cast<queue_info_session_t*>(data);
  if (!queue_info_session || !GetROCMToolObj() ||
      !GetROCMToolObj()->GetSession(queue_info_session->session_id) ||
      !GetROCMToolObj()->GetSession(queue_info_session->session_id)->GetProfiler())
    return true;
  rocmtools::Session* session = GetROCMToolObj()->GetSession(queue_info_session->session_id);
  rocmtools::profiler::Profiler* profiler = session->GetProfiler();
  std::vector<pending_signal_t>& pending_signals = const_cast<std::vector<pending_signal_t>&>(
      profiler->GetPendingSignals(queue_info_session->writer_id));

  if (!pending_signals.empty()) {
    for (auto it = pending_signals.begin(); it != pending_signals.end();
         it = pending_signals.erase(it)) {
      auto& pending = *it;
      std::lock_guard<std::mutex> lock(session->GetSessionLock());
      if (hsa_support::GetCoreApiTable().hsa_signal_load_relaxed_fn(pending.signal)) return true;
      hsa_amd_profiling_dispatch_time_t time;
      hsa_support::GetAmdExtTable().hsa_amd_profiling_get_dispatch_time_fn(
          queue_info_session->agent, pending.signal, &time);
      rocprofiler_record_profiler_t record{};
      record.gpu_id = rocprofiler_agent_id_t{
          (uint64_t)hsa_support::GetAgentInfo(queue_info_session->agent.handle).getIndex()};
      record.kernel_properties = pending.kernel_properties;
      record.thread_id = rocprofiler_thread_id_t{pending.thread_id};
      record.queue_idx = rocprofiler_queue_index_t{pending.queue_index};
      record.timestamps = rocprofiler_record_header_timestamp_t{time.start, time.end};
      record.queue_id = rocprofiler_queue_id_t{queue_info_session->queue_id};
      if (pending.counters_count > 0 && pending.context->metrics_list.size() > 0 &&
          pending.profile) {
        AddRecordCounters(&record, pending);
      }
      record.header = {ROCPROFILER_PROFILER_RECORD,
                       rocprofiler_record_id_t{GetROCMToolObj()->GetUniqueRecordId()}};
      record.kernel_id = rocprofiler_kernel_id_t{pending.kernel_descriptor};

      if (pending.session_id.handle == 0) {
        pending.session_id = GetROCMToolObj()->GetCurrentSessionId();
      }
      if (session->FindBuffer(pending.buffer_id)) {
        Memory::GenericBuffer* buffer = session->GetBuffer(pending.buffer_id);
        if (pending.profile && pending.counters_count > 0) {
          rocprofiler_record_counter_instance_t* record_counters = record.counters;
          buffer->AddRecord(
              record, record.counters,
              (record.counters_count.value * (sizeof(rocprofiler_record_counter_instance_t) + 1)),
              [](auto& record, const void* data) {
                record.counters = const_cast<rocprofiler_record_counter_instance_t*>(
                    static_cast<const rocprofiler_record_counter_instance_t*>(data));
              });
          free(record_counters);
        } else {
          buffer->AddRecord(record);
        }
      }
      if (pending.counters_count > 0 && pending.profile && pending.profile->events) {
        // TODO(aelwazir): we need a better way of distributing events and free them
        // free(const_cast<hsa_ven_amd_aqlprofile_event_t*>(pending.profile->events));
        hsa_status_t status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_free_fn(
            (pending.profile->output_buffer.ptr));
        if (status != HSA_STATUS_SUCCESS) {
          printf("Error: Couldn't free output buffer memory\n");
        }
        status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_free_fn(
            (pending.profile->command_buffer.ptr));
        if (status != HSA_STATUS_SUCCESS) {
          printf("Error: Couldn't free command buffer memory\n");
        }
        delete pending.profile;
      }
      if (pending.signal.handle)
        hsa_support::GetCoreApiTable().hsa_signal_destroy_fn(pending.signal);
      if (queue_info_session->interrupt_signal.handle)
        hsa_support::GetCoreApiTable().hsa_signal_destroy_fn(queue_info_session->interrupt_signal);
    }
  }
  delete queue_info_session;
  ACTIVE_INTERRUPT_SIGNAL_COUNT.fetch_sub(1, std::memory_order_relaxed);
  return false;
}

bool AsyncSignalHandlerATT(hsa_signal_value_t /* signal */, void* data) {
  // TODO: finish implementation to iterate trace data and add it to rocmtools record
  // and generic buffer

  auto queue_info_session = static_cast<queue_info_session_t*>(data);
  if (!queue_info_session || !GetROCMToolObj() ||
      !GetROCMToolObj()->GetSession(queue_info_session->session_id) ||
      !GetROCMToolObj()->GetSession(queue_info_session->session_id)->GetAttTracer())
    return true;
  rocmtools::Session* session = GetROCMToolObj()->GetSession(queue_info_session->session_id);
  rocmtools::att::AttTracer* att_tracer = session->GetAttTracer();
  std::vector<att_pending_signal_t>& pending_signals =
      const_cast<std::vector<att_pending_signal_t>&>(
          att_tracer->GetPendingSignals(queue_info_session->writer_id));

  if (!pending_signals.empty()) {
    for (auto it = pending_signals.begin(); it != pending_signals.end();
         it = pending_signals.erase(it)) {
      auto& pending = *it;
      std::lock_guard<std::mutex> lock(session->GetSessionLock());
      if (hsa_support::GetCoreApiTable().hsa_signal_load_relaxed_fn(pending.signal)) return true;
      rocprofiler_record_att_tracer_t record{};
      record.kernel_id = rocprofiler_kernel_id_t{pending.kernel_descriptor};
      record.gpu_id = rocprofiler_agent_id_t{
          (uint64_t)hsa_support::GetAgentInfo(queue_info_session->agent.handle).getIndex()};
      record.kernel_properties = pending.kernel_properties;
      record.thread_id = rocprofiler_thread_id_t{pending.thread_id};
      record.queue_idx = rocprofiler_queue_index_t{pending.queue_index};
      record.queue_id = rocprofiler_queue_id_t{queue_info_session->queue_id};
      if (/*pending.counters_count > 0 && */ pending.profile) {
        AddAttRecord(&record, queue_info_session->agent, pending);
      }
      record.header = {ROCPROFILER_ATT_TRACER_RECORD,
                       rocprofiler_record_id_t{GetROCMToolObj()->GetUniqueRecordId()}};

      if (pending.session_id.handle == 0) {
        pending.session_id = GetROCMToolObj()->GetCurrentSessionId();
      }
      if (session->FindBuffer(pending.buffer_id)) {
        Memory::GenericBuffer* buffer = session->GetBuffer(pending.buffer_id);
        buffer->AddRecord(record);
      }
      hsa_status_t status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_free_fn(
          (pending.profile->output_buffer.ptr));
      if (status != HSA_STATUS_SUCCESS) {
        printf("Error: Couldn't free output buffer memory\n");
      }
      status = rocmtools::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_free_fn(
          (pending.profile->command_buffer.ptr));
      if (status != HSA_STATUS_SUCCESS) {
        printf("Error: Couldn't free command buffer memory\n");
      }
      delete pending.profile;
    }
  }
  delete queue_info_session;

  return false;
}

void CreateBarrierPacket(const hsa_signal_t& packet_completion_signal,
                         std::vector<Packet::packet_t>* transformed_packets) {
  hsa_barrier_and_packet_t barrier{0};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
  barrier.dep_signal[0] = packet_completion_signal;
  void* barrier_ptr = &barrier;
  transformed_packets->emplace_back(*reinterpret_cast<Packet::packet_t*>(barrier_ptr));
}

void AddVendorSpecificPacket(const Packet::packet_t* packet,
                             std::vector<Packet::packet_t>* transformed_packets,
                             const hsa_signal_t& packet_completion_signal) {
  transformed_packets->emplace_back(*packet).completion_signal = packet_completion_signal;
}

void SignalAsyncHandler(const hsa_signal_t& signal, void* data) {
  hsa_status_t status = hsa_support::GetAmdExtTable().hsa_amd_signal_async_handler_fn(
      signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandler, data);
  if (status != HSA_STATUS_SUCCESS) fatal("hsa_amd_signal_async_handler failed");
}

void signalAsyncHandlerATT(const hsa_signal_t& signal, void* data) {
  hsa_status_t status = hsa_support::GetAmdExtTable().hsa_amd_signal_async_handler_fn(
      signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandlerATT, data);
  if (status != HSA_STATUS_SUCCESS) fatal("hsa_amd_signal_async_handler failed");
}

void CreateSignal(uint32_t attribute, hsa_signal_t* signal) {
  hsa_status_t status =
      hsa_support::GetAmdExtTable().hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
  if (status != HSA_STATUS_SUCCESS) fatal("hsa_amd_signal_create failed");
}

template <typename Integral = uint64_t> constexpr Integral bit_mask(int first, int last) {
  assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
  size_t num_bits = last - first + 1;
  return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                             /* num_bits exceed the size of Integral */
                                             : ((Integral{1} << num_bits) - 1))
      << first;
}

/* Extract bits [last:first] from t.  */
template <typename Integral> constexpr Integral bit_extract(Integral x, int first, int last) {
  return (x >> first) & bit_mask<Integral>(0, last - first);
}

static int KernelInterceptCount = 0;
std::atomic<uint32_t> WRITER_ID{0};
/**
 * @brief This function is a queue write interceptor. It intercepts the
 * packet write function. Creates an instance of packet class with the raw
 * pointer. invoke the populate function of the packet class which returns a
 * pointer to the packet. This packet is written into the queue by this
 * interceptor by invoking the writer function.
 */
void WriteInterceptor(const void* packets, uint64_t pkt_count, uint64_t user_pkt_index, void* data,
                      hsa_amd_queue_intercept_packet_writer writer) {
  const Packet::packet_t* packets_arr = reinterpret_cast<const Packet::packet_t*>(packets);
  std::vector<Packet::packet_t> transformed_packets;
  rocprofiler_session_id_t session_id;
  if (GetROCMToolObj())
    // Getting Session ID
    session_id = GetROCMToolObj()->GetCurrentSessionId();
  else
    session_id = {0};

  // Counter Names declaration
  std::vector<std::string> session_data;

  rocprofiler_buffer_id_t buffer_id;

  uint64_t session_data_count = 0;

  bool is_counter_collection_mode = false;
  bool is_timestamp_collection_mode = false;
  bool is_att_collection_mode = false;
  bool is_pc_sampling_collection_mode = false;
  std::vector<rocprofiler_att_parameter_t> att_parameters_data;
  uint32_t replay_mode_count = 0;
  std::vector<std::string> kernel_profile_names;
  std::vector<std::string> att_counters_names;

  rocmtools::Session* session = nullptr;

  // Getting Counters count from the Session
  if (session_id.handle > 0 && GetROCMToolObj()) {
    session = GetROCMToolObj()->GetSession(session_id);
    if (session && session->FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION)) {
      rocprofiler_filter_id_t filter_id = session->GetFilterIdWithKind(ROCPROFILER_COUNTERS_COLLECTION);
      rocmtools::Filter* filter = session->GetFilter(filter_id);
      session_data = filter->GetCounterData();
      is_counter_collection_mode = true;
      session_data_count = session_data.size();
      buffer_id = filter->GetBufferId();
    } else if (session && session->FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION)) {
      is_timestamp_collection_mode = true;
      rocprofiler_filter_id_t filter_id =
          session->GetFilterIdWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION);
      rocmtools::Filter* filter = session->GetFilter(filter_id);
      buffer_id = filter->GetBufferId();
    } else if (session && session->FindFilterWithKind(ROCPROFILER_ATT_TRACE_COLLECTION)) {
      rocprofiler_filter_id_t filter_id =
          session->GetFilterIdWithKind(ROCPROFILER_ATT_TRACE_COLLECTION);
      rocmtools::Filter* filter = session->GetFilter(filter_id);
      att_parameters_data = filter->GetAttParametersData();
      is_att_collection_mode = true;
      buffer_id = session->GetFilter(session->GetFilterIdWithKind(ROCPROFILER_ATT_TRACE_COLLECTION))
                      ->GetBufferId();

      att_counters_names = filter->GetCounterData();
      kernel_profile_names = std::get<std::vector<std::string>>(filter->GetProperty(ROCPROFILER_FILTER_KERNEL_NAMES));
    } else if (session && session->FindFilterWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION)) {
      is_pc_sampling_collection_mode = true;
    }
  }

  if (session_id.handle > 0 && pkt_count > 0 &&
      (is_counter_collection_mode || is_timestamp_collection_mode ||
       is_pc_sampling_collection_mode) &&
      session) {
    // Getting Queue Data and Information
    auto& queue_info = *static_cast<Queue*>(data);
    std::lock_guard<std::mutex> lk(queue_info.qw_mutex);


    // hsa_ven_amd_aqlprofile_profile_t* profile;
    std::vector<std::pair<rocmtools::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>*
        profiles = nullptr;


    // Searching accross all the packets given during this write
    for (size_t i = 0; i < pkt_count; ++i) {
      auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];

      // +Skip kernel dispatch IDs not wanted
      // Skip packets other than kernel dispatch packets.
      if (bit_extract(original_packet.header, HSA_PACKET_HEADER_TYPE,
                      HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1) !=
          HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        transformed_packets.emplace_back(packets_arr[i]);
        continue;
      }

      // If counters found in the session
      if (session_data_count > 0 && is_counter_collection_mode) {
        // Get the PM4 Packets using packets_generator
        profiles = Packet::InitializeAqlPackets(queue_info.GetCPUAgent(), queue_info.GetGPUAgent(),
                                                session_data);
        replay_mode_count = profiles->size();
      }

      uint32_t profile_id = 0;
      hsa_signal_t interrupt_signal;
      do {
        std::pair<rocmtools::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*> profile;
        if (profiles && replay_mode_count > 0) profile = profiles->at(profile_id);

        uint32_t writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);

        if (session_data_count > 0 && is_counter_collection_mode && profiles &&
            replay_mode_count > 0) {
          // hsa_signal_t begin_signal{};
          // CreateSignal(0, &begin_signal);
          // hsa_barrier_and_packet_t barrier{0};
          // barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
          // CreateSignal(0, &barrier.completion_signal);
          // barrier.dep_signal[0] = hsa_signal_t{};
          // Packet::packet_t* __attribute__((__may_alias__)) pkt =
          //     (reinterpret_cast<Packet::packet_t*>(&barrier));
          // transformed_packets.emplace_back(*pkt);
          // hsa_status_t status = hsa_support::GetAmdExtTable().hsa_amd_signal_async_handler_fn(
          //     barrier.completion_signal, HSA_SIGNAL_CONDITION_GTE, 1, BeginSignalHandler,
          //     &profiles->at(profile_id));
          // if (status != HSA_STATUS_SUCCESS)
          //   fatal("hsa_amd_signal_async_handler failed for begin signal");

          // Adding start packet and its barrier with a dummy signal
          hsa_signal_t dummy_signal{};
          dummy_signal.handle = 0;
          profile.first->start_packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
              << HSA_PACKET_HEADER_TYPE;
          AddVendorSpecificPacket(profile.first->start_packet, &transformed_packets, dummy_signal);

          CreateBarrierPacket(profile.first->start_packet->completion_signal, &transformed_packets);
        }

        auto& packet = transformed_packets.emplace_back(packets_arr[i]);
        auto& dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t&>(packet);

        /*
         * Only PC sampling relies on this right now, so it would be better to
         * only generate an ID if PC sampling is active to conserve IDs, but it's
         * unlikely 64 bits' worth of identifiers will be exhausted during the
         * lifetime of the ROCMToolObj.
         */
        dispatch_packet.reserved2 = GetROCMToolObj()->GetUniqueKernelDispatchId();

        CreateSignal(HSA_AMD_SIGNAL_AMD_GPU_ONLY, &packet.completion_signal);
        // Adding the dispatch packet newly created signal to the pending signals
        // list to be processed by the signal interrupt
        rocprofiler_kernel_properties_t kernel_properties =
            set_kernel_properties(dispatch_packet, queue_info.GetGPUAgent());
        if (session) {
          uint64_t record_id = GetROCMToolObj()->GetUniqueRecordId();
          AddKernelNameWithDispatchID(GetKernelNameFromKsymbols(dispatch_packet.kernel_object), record_id);
          if (profiles && replay_mode_count > 0) {
            session->GetProfiler()->AddPendingSignals(
                writer_id, record_id, dispatch_packet.completion_signal,
                session_id, buffer_id, profile.first, profile.first->metrics_list.size(),
                profile.second, kernel_properties, (uint32_t)syscall(__NR_gettid), user_pkt_index);
          } else {
            session->GetProfiler()->AddPendingSignals(
                writer_id, record_id, dispatch_packet.completion_signal,
                session_id, buffer_id, nullptr, 0, nullptr, kernel_properties,
                (uint32_t)syscall(__NR_gettid), user_pkt_index);
          }
        }

        // Make a copy of the original packet, adding its signal to a barrier
        // packet and create a new signal for it to get timestamps
        if (original_packet.completion_signal.handle) {
          hsa_barrier_and_packet_t barrier{0};
          barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
          Packet::packet_t* __attribute__((__may_alias__)) pkt =
              (reinterpret_cast<Packet::packet_t*>(&barrier));
          transformed_packets.emplace_back(*pkt).completion_signal =
              original_packet.completion_signal;
        }

        // Adding a barrier packet with the original packet's completion signal.
        CreateSignal(0, &interrupt_signal);

        // Adding Stop and Read PM4 Packets
        if (session_data_count > 0 && is_counter_collection_mode && profiles &&
            replay_mode_count > 0) {
          hsa_signal_t dummy_signal{};
          profile.first->stop_packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
              << HSA_PACKET_HEADER_TYPE;
          AddVendorSpecificPacket(profile.first->stop_packet, &transformed_packets, dummy_signal);
          profile.first->read_packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC
              << HSA_PACKET_HEADER_TYPE;
          AddVendorSpecificPacket(profile.first->read_packet, &transformed_packets,
                                  interrupt_signal);

          // Added Interrupt Signal with barrier and provided handler for it
          CreateBarrierPacket(interrupt_signal, &transformed_packets);
        } else {
          hsa_barrier_and_packet_t barrier{0};
          barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
          barrier.completion_signal = interrupt_signal;
          Packet::packet_t* __attribute__((__may_alias__)) pkt =
              (reinterpret_cast<Packet::packet_t*>(&barrier));
          transformed_packets.emplace_back(*pkt);
        }
        //  Creating Async Handler to be called every time the interrupt signal is
        //  marked complete
        SignalAsyncHandler(interrupt_signal,
                           new queue_info_session_t{queue_info.GetGPUAgent(), session_id,
                                                    queue_info.GetQueueID(), writer_id});
        ACTIVE_INTERRUPT_SIGNAL_COUNT.fetch_add(1, std::memory_order_relaxed);
        profile_id++;
      } while (replay_mode_count > 0 && profile_id < replay_mode_count);  // Profiles loop end
    }
    /* Write the transformed packets to the hardware queue.  */
    writer(&transformed_packets[0], transformed_packets.size());
  } else if (session_id.handle > 0 && pkt_count > 0 &&
            is_att_collection_mode && session &&
            KernelInterceptCount < MAX_ATT_PROFILES
  ) {
    // att start
    // Getting Queue Data and Information
    auto& queue_info = *static_cast<Queue*>(data);
    std::lock_guard<std::mutex> lk(queue_info.qw_mutex);
    Agent::AgentInfo* agentInfo = &(hsa_support::GetAgentInfo(queue_info.GetGPUAgent().handle));

    bool can_profile_anypacket = false;
    std::vector<bool> can_profile_packet;

    for (size_t i = 0; i < pkt_count; ++i) {
      auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];
      bool b_profile_this_object = false;

      // Skip packets other than kernel dispatch packets.
      if (bit_extract(original_packet.header, HSA_PACKET_HEADER_TYPE,
                      HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1) ==
          HSA_PACKET_TYPE_KERNEL_DISPATCH) {

        auto& kdispatch = static_cast<const hsa_kernel_dispatch_packet_s*>(packets)[i];
        uint64_t kernel_object = kdispatch.kernel_object;

        // Try to match the mangled kernel name with given matches in input.txt
        try {
          std::lock_guard<std::mutex> lock(ksymbol_map_lock);
          assert(ksymbols);
          const std::string& kernel_name = ksymbols->at(kernel_object);

          // We want to initiate att profiling only if a match exists
          for(const std::string& kernel_matches : kernel_profile_names) {
            if (kernel_name.find(kernel_matches) != std::string::npos) {
              b_profile_this_object = true;
              break;
            }
          }
          if (!b_profile_this_object) printf("Skipping: %s\n", kernel_name.c_str());
        } catch (...) {
          printf("Warning: Unknown name for object %lu\n", kernel_object);
        }
      }

      if (b_profile_this_object)
        can_profile_anypacket = true;
      can_profile_packet.push_back(b_profile_this_object);
    }

    if (!can_profile_anypacket) {
      /* Write the original packets to the hardware if no patch will be profiled */
      writer(packets, pkt_count);
      return;
    }

    // Preparing att Packets
    Packet::packet_t start_packet{};
    Packet::packet_t stop_packet{};
    hsa_ven_amd_aqlprofile_profile_t* profile = nullptr;

    if (att_parameters_data.size() > 0 && is_att_collection_mode) {
      // TODO sauverma: convert att_parameters_data to pass to generateattPackets
      std::vector<hsa_ven_amd_aqlprofile_parameter_t> att_params;
      int num_att_counters = 0;

      for (rocprofiler_att_parameter_t& param : att_parameters_data) {
        att_params.push_back({
          static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(param.parameter_name)),
          param.value
        });
        num_att_counters += param.parameter_name == ROCPROFILER_ATT_PERFCOUNTER;
      }

      if (att_counters_names.size() > 0) {
        MetricsDict* metrics_dict_ = MetricsDict::Create(agentInfo);

        for (const std::string& counter_name : att_counters_names) {
          const Metric* metric = metrics_dict_->Get(counter_name);
          const BaseMetric* base = dynamic_cast<const BaseMetric*>(metric);
          if (!base) {
            printf("Invalid base metric value: %s\n", counter_name.c_str());
            exit(1);
          }
          std::vector<const counter_t*> counters;
          base->GetCounters(counters);
          hsa_ven_amd_aqlprofile_event_t event = counters[0]->event;
          if (event.block_name != HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ) {
            printf("Only events from the SQ block can be selected for ATT.");
            exit(1);
          }
          att_params.push_back({
            static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(ROCPROFILER_ATT_PERFCOUNTER)),
            event.counter_id | (event.counter_id ? (0xF<<24) : 0)
          });
          num_att_counters += 1;
        }

        hsa_ven_amd_aqlprofile_parameter_t zero_perf = {
          static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(int(ROCPROFILER_ATT_PERFCOUNTER)), 0};

        // Fill other perfcounters with 0's
        for(; num_att_counters<16; num_att_counters++) att_params.push_back(zero_perf);
      }

      // Get the PM4 Packets using packets_generator
      profile = Packet::GenerateATTPackets(queue_info.GetCPUAgent(), queue_info.GetGPUAgent(),
                                            att_params, &start_packet, &stop_packet);
    }

    // Searching across all the packets given during this write
    for (size_t i = 0; i < pkt_count; ++i) {
      auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];

      // Skip all packets marked with !can_profile
      if (i >= can_profile_packet.size() || can_profile_packet[i] == false) {
        transformed_packets.emplace_back(packets_arr[i]);
        continue;
      }
      KernelInterceptCount += 1;

      uint32_t writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);


      if (att_parameters_data.size() > 0 && is_att_collection_mode && profile) {
        // Adding start packet and its barrier with a dummy signal
        hsa_signal_t dummy_signal{};
        dummy_signal.handle = 0;
        start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(&start_packet, &transformed_packets, dummy_signal);
        CreateBarrierPacket(start_packet.completion_signal, &transformed_packets);
      }

      auto& packet = transformed_packets.emplace_back(packets_arr[i]);
      auto& dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t&>(packet);

      CreateSignal(HSA_AMD_SIGNAL_AMD_GPU_ONLY, &packet.completion_signal);
      // Adding the dispatch packet newly created signal to the pending signals
      // list to be processed by the signal interrupt
      rocprofiler_kernel_properties_t kernel_properties =
          set_kernel_properties(dispatch_packet, queue_info.GetGPUAgent());
      uint64_t record_id = GetROCMToolObj()->GetUniqueRecordId();
      AddKernelNameWithDispatchID(GetKernelNameFromKsymbols(dispatch_packet.kernel_object), record_id);
      if (session && profile) {
        session->GetAttTracer()->AddPendingSignals(
            writer_id, record_id, dispatch_packet.completion_signal, session_id,
            buffer_id, profile, kernel_properties, (uint32_t)syscall(__NR_gettid), user_pkt_index);
      } else {
        session->GetAttTracer()->AddPendingSignals(
            writer_id, record_id, dispatch_packet.completion_signal, session_id,
            buffer_id, nullptr, kernel_properties, (uint32_t)syscall(__NR_gettid), user_pkt_index);
      }

      // Make a copy of the original packet, adding its signal to a barrier packet
      if (original_packet.completion_signal.handle) {
        hsa_barrier_and_packet_t barrier{0};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        Packet::packet_t* __attribute__((__may_alias__)) pkt =
            (reinterpret_cast<Packet::packet_t*>(&barrier));
        transformed_packets.emplace_back(*pkt).completion_signal =
            original_packet.completion_signal;
      }

      // Adding a barrier packet with the original packet's completion signal.
      hsa_signal_t interrupt_signal;
      CreateSignal(0, &interrupt_signal);

      // Adding Stop PM4 Packets
      if (att_parameters_data.size() > 0 && is_att_collection_mode && profile) {
        stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        AddVendorSpecificPacket(&stop_packet, &transformed_packets, interrupt_signal);

        // Added Interrupt Signal with barrier and provided handler for it
        CreateBarrierPacket(interrupt_signal, &transformed_packets);
      } else {
        hsa_barrier_and_packet_t barrier{0};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.completion_signal = interrupt_signal;
        Packet::packet_t* __attribute__((__may_alias__)) pkt =
            (reinterpret_cast<Packet::packet_t*>(&barrier));
        transformed_packets.emplace_back(*pkt);
      }

      // Creating Async Handler to be called every time the interrupt signal is
      // marked complete
      signalAsyncHandlerATT(
          interrupt_signal,
          new queue_info_session_t{queue_info.GetGPUAgent(), session_id, queue_info.GetQueueID(),
                                   writer_id, interrupt_signal});
    }
    /* Write the transformed packets to the hardware queue.  */
    writer(&transformed_packets[0], transformed_packets.size());
    // ATT end
  } else {
    /* Write the original packets to the hardware queue if no profiling session
     * is active  */
    writer(packets, pkt_count);
  }
}


Queue::Queue(const hsa_agent_t& cpu_agent, const hsa_agent_t& gpu_agent, uint32_t size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
             uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue)
    : cpu_agent_(cpu_agent), gpu_agent_(gpu_agent) {
  [[maybe_unused]] hsa_status_t status =
      hsa_support::GetAmdExtTable().hsa_amd_queue_intercept_create_fn(
          gpu_agent, size, type, callback, data, private_segment_size, group_segment_size,
          &intercept_queue_);
  assert(status == HSA_STATUS_SUCCESS);

  status = hsa_support::GetAmdExtTable().hsa_amd_profiling_set_profiler_enabled_fn(intercept_queue_,
                                                                                   true);
  assert(status == HSA_STATUS_SUCCESS);

  hsa_support::GetAmdExtTable().hsa_amd_queue_intercept_register_fn(intercept_queue_,
                                                                    WriteInterceptor, this);
  assert(status == HSA_STATUS_SUCCESS);

  *queue = intercept_queue_;
}

// Queue::~Queue() { std::lock_guard<std::mutex> lk(mutex_); }

hsa_queue_t* Queue::GetCurrentInterceptQueue() { return intercept_queue_; }

hsa_agent_t Queue::GetGPUAgent() { return gpu_agent_; }

hsa_agent_t Queue::GetCPUAgent() { return cpu_agent_; }

uint64_t Queue::GetQueueID() { return intercept_queue_->id; }

void InitializePools(hsa_agent_t cpu_agent) { Packet::InitializePools(cpu_agent); }

}  // namespace queue
}  // namespace rocmtools
