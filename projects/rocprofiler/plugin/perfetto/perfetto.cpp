/* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
 THE SOFTWARE. */

#include "perfetto.h"
#include "rocprofiler.h"

#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <functional>
#include <iostream>
#include <string_view>

#include <cxxabi.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "perfetto/sdk/perfetto.h"
#include "rocprofiler_plugin.h"
#include "../utils.h"

#include "src/utils/filesystem.hpp"

#define STREAM_CONSTANT 98736677
#define QUEUE_CONSTANT 18746479

namespace fs = rocprofiler::common::filesystem;

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("GENERIC").SetDescription("GENERAL_CATEGORY"),
    perfetto::Category("ROCTX_MARK").SetDescription("ACTIVITY_DOMAIN_ROCTX_API"),
    perfetto::Category("ROCTX_RANGE_PUSH_POP").SetDescription("ACTIVITY_DOMAIN_ROCTX_API"),
    perfetto::Category("ROCTX_RANGE_START_STOP").SetDescription("ACTIVITY_DOMAIN_ROCTX_API"),
    perfetto::Category("HSA_API").SetDescription("ACTIVITY_DOMAIN_HSA_API"),
    perfetto::Category("HIP_API").SetDescription("ACTIVITY_DOMAIN_HIP_API"),
    perfetto::Category("External_API").SetDescription("ACTIVITY_DOMAIN_EXT_API"),
    perfetto::Category("HIP_OPS").SetDescription("ACTIVITY_DOMAIN_HIP_OPS"),
    perfetto::Category("HSA_OPS").SetDescription("ACTIVITY_DOMAIN_HSA_OPS"),
    perfetto::Category("MEM_COPIES").SetDescription("MEMORY_COPY_ASYNCHRONOUS_ACTIVITY"),
    perfetto::Category("KERNELS").SetDescription("KERNEL_DISPATCHES"),
    perfetto::Category("COUNTERS").SetDescription("PERFORMANCE_COUNTERS"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();


enum class TrackType {
  DEVICE = 2,
  MCOPY,
  HIPAPI,
  HSAAPI,
  ROCTX_MARK,
  ROCTX_PUSH_RANGE,
  ROCTX_START_RANGE,
  TRACER_DEV_ID,
  PROFILER_DEV_ID,
};

struct TrackID {
  TrackID(TrackType type, uint64_t machine, uint64_t device, uint64_t stream)
      : type(type), machine(machine), dev(device), stream(stream) {};
  TrackType type;
  uint64_t machine;
  uint64_t dev;
  uint64_t stream;
  bool operator==(const TrackID& other) const {
    return machine == other.machine && dev == other.dev && stream == other.stream &&
        type == other.type;
  }
};

template <> struct std::hash<TrackID> {
  uint64_t operator()(const TrackID& s) const {
    return static_cast<uint64_t>(s.type) ^ (s.machine + 1) ^ (s.dev << 32) ^ (s.dev >> 32) ^
        (s.stream << 48) ^ (s.stream >> 16);
  }
};

namespace {
std::mutex writing_lock{};

std::string process_name;

std::string get_kernel_name(rocprofiler_record_profiler_t& profiler_record) {
  std::string kernel_name = "";
  size_t name_length = 1;
  CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                       profiler_record.kernel_id, &name_length));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wstringop-overread"
  if (name_length > 1) {
    const char* kernel_name_c = nullptr;
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                    profiler_record.kernel_id, &kernel_name_c));
    if (kernel_name_c && strlen(kernel_name_c) > 1)
      kernel_name = rocprofiler::truncate_name(rocprofiler::cxx_demangle(kernel_name_c));
  }
#pragma GCC diagnostic pop
  return kernel_name;
}

class perfetto_plugin_t {
 public:
  perfetto_plugin_t(int filename_ext) {
    const char* output_dir = getenv("OUTPUT_PATH");
    const char* temp_file_name = getenv("OUT_FILE_NAME");
    std::string output_file_name = temp_file_name ? std::string(temp_file_name) + "_" : "";
    if (filename_ext > 0) output_file_name += "_run" + std::to_string(filename_ext) + "_";

    if (output_dir == nullptr) output_dir = "./";

    output_prefix_ = output_dir;
    if (!fs::is_directory(fs::status(output_prefix_))) {
      if (!stream_.fail()) rocprofiler::warning("Cannot open output directory '%s'", output_dir);
      stream_.setstate(std::ios_base::failbit);
      return;
    }

    machine_id_ = gethostid();
    hostname_.resize(1024);
    gethostname(hostname_.data(), hostname_.size());

    perfetto::TracingInitArgs args;
    args.backends |= perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    track_event_cfg.add_enabled_categories("*");

    perfetto::TraceConfig trace_cfg;

    auto buffer_cfg = trace_cfg.add_buffers();
    uint32_t max_buffer_size = 1024 * 1024;  // Default max buffer size is 1 GB
    buffer_cfg->set_fill_policy(
        perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_RING_BUFFER);
    const char* max_buffer_size_str = getenv("rocprofiler_PERFETTO_MAX_BUFFER_SIZE_KIB");
    if (max_buffer_size_str && std::atol(max_buffer_size_str) > 0)
      max_buffer_size = std::atol(max_buffer_size_str);
    // Record up to max buffer size determined by user or the 10 GB (default value)
    buffer_cfg->set_size_kb(max_buffer_size);

    auto* data_source_cfg = trace_cfg.add_data_sources()->mutable_config();
    data_source_cfg->set_name("track_event");

    output_prefix_.append(output_file_name + std::to_string(GetPid()) + "_output.pftrace");
    file_descriptor_ = open(output_prefix_.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (file_descriptor_ == -1) rocprofiler::warning("Can't open output file\n");

    tracing_session_ = perfetto::Tracing::NewTrace();
    trace_cfg.set_unique_session_name(output_prefix_.string());
    tracing_session_->Setup(trace_cfg, file_descriptor_);
    tracing_session_->StartBlocking();

    // Give a custom name for the traced process.
    perfetto::ProcessTrack process_track = perfetto::ProcessTrack::Current();
    perfetto::protos::gen::TrackDescriptor desc = process_track.Serialize();
    desc.mutable_process()->set_process_name("Node: " + hostname_);
    perfetto::TrackEvent::SetTrackDescriptor(process_track, desc);

    is_valid_ = true;
  }

  bool delete_perfetto_plugin() {
    if (is_valid_ && tracing_session_) {
      tracing_session_->StopBlocking();
      is_valid_ = false;
      close(file_descriptor_);
      tracing_session_.reset();
      return true;
    }
    return false;
  }

  const char* GetDomainName(rocprofiler_tracer_activity_domain_t domain) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX:
        return "ROCTX_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HIP_API:
        return "HIP_API_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HIP_OPS:
        return "HIP_OPS_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_API:
        return "HSA_API_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_OPS:
        return "HSA_OPS_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_EVT:
        return "HSA_EVT_DOMAIN";
        break;
      default:
        return "";
    }
  }

  int FlushProfilerRecord(rocprofiler_record_profiler_t profiler_record,
                          rocprofiler_session_id_t session_id) {
    // ToDO: rename this variable?
    if (!tracing_session_) rocprofiler::warning("Tracing session is deleted!\n");

    const uint64_t device_id = profiler_record.gpu_id.handle;
    const uint64_t queue_id = profiler_record.queue_id.handle;
    const uint64_t correlation_id = profiler_record.correlation_id.value;

    uint64_t queue_track_id =
        getTrackID(TrackType::PROFILER_DEV_ID, machine_id_, device_id, queue_id);

    auto queue_track_it = queue_tracks_.find(queue_track_id);
    if (queue_track_it == queue_tracks_.end()) {
      /* Create a new perfetto::Track */
      queue_track_it = queue_tracks_.emplace(queue_track_id, perfetto::Track(queue_track_id)).first;

      auto queue_desc = queue_track_it->second.Serialize();
      std::stringstream ss;
      ss << "Dev " << device_id << " Queue " << queue_id;
      queue_desc.set_name(ss.str());
      perfetto::TrackEvent::SetTrackDescriptor(queue_track_it->second, queue_desc);
      track_ids_used_.emplace_back(queue_track_id);
    }
    auto& queue_track = queue_track_it->second;

    // Taken from rocprofiler: The size hasn't changed in  recent past
    static const uint32_t lds_block_size = 128 * 4;

    std::string full_kernel_name = get_kernel_name(profiler_record);

    if (correlation_id) {
      TRACE_EVENT_BEGIN("KERNELS", perfetto::DynamicString(full_kernel_name.c_str()), queue_track,
                        profiler_record.timestamps.begin.value, "Full Kernel Name",
                        full_kernel_name.c_str(), "Agent ID", device_id, "Queue ID",
                        profiler_record.queue_id.handle, "GRD",
                        profiler_record.kernel_properties.grid_size, "WGR",
                        profiler_record.kernel_properties.workgroup_size, "LDS",
                        (((profiler_record.kernel_properties.lds_size + (lds_block_size - 1)) &
                          ~(lds_block_size - 1))),
                        "SCR", profiler_record.kernel_properties.scratch_size, "Arch. VGPR",
                        profiler_record.kernel_properties.arch_vgpr_count, "Accumulation Vgpr",
                        profiler_record.kernel_properties.accum_vgpr_count, "SGPR",
                        profiler_record.kernel_properties.sgpr_count, "Wave Size",
                        profiler_record.kernel_properties.wave_size, "Signal",
                        profiler_record.kernel_properties.signal_handle,
                        perfetto::Flow::ProcessScoped(correlation_id));
    } else {
      TRACE_EVENT_BEGIN("KERNELS", perfetto::DynamicString(full_kernel_name.c_str()), queue_track,
                        profiler_record.timestamps.begin.value, "Full Kernel Name",
                        full_kernel_name.c_str(), "Agent ID", device_id, "Queue ID",
                        profiler_record.queue_id.handle, "GRD",
                        profiler_record.kernel_properties.grid_size, "WGR",
                        profiler_record.kernel_properties.workgroup_size, "LDS",
                        (((profiler_record.kernel_properties.lds_size + (lds_block_size - 1)) &
                          ~(lds_block_size - 1))),
                        "SCR", profiler_record.kernel_properties.scratch_size, "Arch. VGPR",
                        profiler_record.kernel_properties.arch_vgpr_count, "Accumulation Vgpr",
                        profiler_record.kernel_properties.accum_vgpr_count, "SGPR",
                        profiler_record.kernel_properties.sgpr_count, "Wave Size",
                        profiler_record.kernel_properties.wave_size, "Signal",
                        profiler_record.kernel_properties.signal_handle);
    }

    TRACE_EVENT_END("KERNELS", queue_track, profiler_record.timestamps.end.value);

    auto get_counter_track_fn = [&](size_t i, rocprofiler_counter_id_t counter_handler) {
      auto& ctrack = counter_tracks_[device_id];

      if (i < ctrack.size()) return;

      const char* name_c = nullptr;
      CHECK_ROCPROFILER(rocprofiler_query_counter_info(session_id, ROCPROFILER_COUNTER_NAME,
                                                       counter_handler, &name_c));

      std::stringstream ss;
      ss << "Dev " << device_id << " Counter " << name_c;
      ctrack.push_back(perfetto::CounterTrack(ss.str().c_str()));
      auto counter_track_desc = ctrack.back().Serialize();
      counter_track_desc.set_name(ss.str());
      perfetto::TrackEvent::SetTrackDescriptor(ctrack.back(), counter_track_desc);
    };

    // For Counters
    if (!profiler_record.counters) return 0;

    for (uint64_t i = 0; i < profiler_record.counters_count.value; i++) {
      if (profiler_record.counters[i].counter_handler.handle == 0) continue;

      get_counter_track_fn(i, profiler_record.counters[i].counter_handler);
      auto& counters_track = counter_tracks_.at(device_id).at(i);
      TRACE_COUNTER("COUNTERS", counters_track, profiler_record.timestamps.begin.value,
                    profiler_record.counters[i].value.value);
      // Added an extra zero event for maintaining start-end of the counter
      TRACE_COUNTER("COUNTERS", counters_track, profiler_record.timestamps.end.value, 0);
    }

    return 0;
  }

  int FlushTracerRecord(rocprofiler_record_tracer_t tracer_record,
                        rocprofiler_session_id_t session_id) {
    if (!tracing_session_) rocprofiler::warning("Tracing session is deleted!\n");
    uint64_t device_id = tracer_record.agent_id.handle;
    const char* operation_name_c = nullptr;
    // ROCTX domain Operation ID doesn't have a name
    // It depends on the user input of the roctx functions.
    // ROCTX message is the tracer_record.name
    if (tracer_record.domain != ACTIVITY_DOMAIN_ROCTX) {
      CHECK_ROCPROFILER(rocprofiler_query_tracer_operation_name(
          tracer_record.domain, tracer_record.operation_id, &operation_name_c));
      if (!operation_name_c) operation_name_c = "Unknown Operation";
    }
    std::string roctx_message;
    uint64_t roctx_id = 0;
    uint64_t thread_id = tracer_record.thread_id.value;
    std::unordered_map<uint64_t, perfetto::Track>::iterator hip_stream_tracks_it;

    std::unordered_map<uint64_t, perfetto::Track>::iterator mem_copies_track_it;
    if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_OPS ||
        tracer_record.domain == ACTIVITY_DOMAIN_HSA_OPS) {
      bool bIsHSAQueue = tracer_record.domain == ACTIVITY_DOMAIN_HSA_OPS;
      uint64_t qID = tracer_record.queue_id.handle;

      uint64_t hip_track_id = getTrackID(TrackType::TRACER_DEV_ID, machine_id_, device_id, qID);
      hip_stream_tracks_it = hip_stream_tracks.find(hip_track_id);
      if (hip_stream_tracks_it == hip_stream_tracks.end()) {
        /* Create a new perfetto::Track (Sub-Track) */
        hip_stream_tracks_it =
            hip_stream_tracks.emplace(hip_track_id, perfetto::Track(hip_track_id)).first;
        auto gpu_desc = hip_stream_tracks_it->second.Serialize();
        std::string queue_str = (bIsHSAQueue ? "Stream " : "HipStream ") + std::to_string(qID);
        gpu_desc.set_name(queue_str);
        perfetto::TrackEvent::SetTrackDescriptor(hip_stream_tracks_it->second, gpu_desc);
        track_ids_used_.emplace_back(hip_track_id);
      }

      {
        uint64_t mcpy_track_id = getTrackID(TrackType::MCOPY, machine_id_, 0, thread_id);
        mem_copies_track_it = mem_copies_tracks_.find(mcpy_track_id);
        if (mem_copies_track_it == mem_copies_tracks_.end()) {
          mem_copies_track_it =
              mem_copies_tracks_.emplace(mcpy_track_id, perfetto::Track(mcpy_track_id)).first;

          auto mem_copies_track_desc = mem_copies_track_it->second.Serialize();
          std::string mem_copies_track_str =
              rocprofiler::string_printf("MEM COPIES(%lu): ", thread_id);
          mem_copies_track_desc.set_name(mem_copies_track_str);
          perfetto::TrackEvent::SetTrackDescriptor(mem_copies_track_it->second,
                                                   mem_copies_track_desc);
        }
      }
    }

    auto& gpu_track = hip_stream_tracks_it->second;
    auto& mem_copies_track = mem_copies_track_it->second;
    switch (tracer_record.domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        roctx_id = tracer_record.external_id.id;
        roctx_message = tracer_record.name ? tracer_record.name : "";
        switch (tracer_record.operation_id.id) {
          case 0: {
            std::unordered_map<uint64_t, perfetto::Track>::iterator roctx_track_it;
            {
              uint64_t rtx_track_id = getTrackID(TrackType::ROCTX_MARK, machine_id_, 0, thread_id);
              roctx_track_it = roctx_mark_tracks_.find(rtx_track_id);
              if (roctx_track_it == roctx_mark_tracks_.end()) {
                roctx_track_it =
                    roctx_mark_tracks_.emplace(rtx_track_id, perfetto::Track(rtx_track_id)).first;

                auto roctx_track_desc = roctx_track_it->second.Serialize();
                std::string roctx_track_str = rocprofiler::string_printf("ROCTX Markers");
                roctx_track_desc.set_name(roctx_track_str);
                perfetto::TrackEvent::SetTrackDescriptor(roctx_track_it->second, roctx_track_desc);
              }
            }
            auto& roctx_track = roctx_track_it->second;
            perfetto::DynamicString roctx_message_pft(
                (!roctx_message.empty() ? roctx_message.c_str() : ""));
            TRACE_EVENT_INSTANT("ROCTX_MARK", roctx_message_pft, roctx_track,
                                tracer_record.timestamps.begin.value, "Timestamp(ns)",
                                tracer_record.timestamps.begin.value, "RocTx ID", roctx_id);
            break;
          }
          case 1: {
            std::unordered_map<uint64_t, perfetto::Track>::iterator roctx_track_it;
            {
              uint64_t rtx_track_id = getTrackID(TrackType::ROCTX_PUSH_RANGE, machine_id_, 0, thread_id);
              roctx_track_it = roctx_push_tracks_.find(rtx_track_id);
              if (roctx_track_it == roctx_push_tracks_.end()) {
                roctx_track_it =
                    roctx_push_tracks_.emplace(rtx_track_id, perfetto::Track(rtx_track_id)).first;

                auto roctx_track_desc = roctx_track_it->second.Serialize();
                std::string roctx_track_str = rocprofiler::string_printf("ROCTX Push/Pop Ranges");
                roctx_track_desc.set_name(roctx_track_str);
                perfetto::TrackEvent::SetTrackDescriptor(roctx_track_it->second, roctx_track_desc);
              }
            }
            auto& roctx_track = roctx_track_it->second;
            perfetto::DynamicString roctx_message_pft(
                (!roctx_message.empty() ? roctx_message.c_str() : ""));
            TRACE_EVENT_BEGIN("ROCTX_RANGE_PUSH_POP", roctx_message_pft, roctx_track,
                              tracer_record.timestamps.begin.value, "Timestamp(ns)",
                              tracer_record.timestamps.begin.value, "RocTx ID", roctx_id);
            roctx_push_track_entries_++;
            break;
          }
          case 3: {
            std::unordered_map<uint64_t, perfetto::Track>::iterator roctx_track_it;
            {
              uint64_t rtx_track_id = getTrackID(TrackType::ROCTX_START_RANGE, machine_id_, 0, thread_id);
              roctx_track_it = roctx_start_tracks_.find(rtx_track_id);
              if (roctx_track_it == roctx_start_tracks_.end()) {
                roctx_track_it =
                    roctx_start_tracks_.emplace(rtx_track_id, perfetto::Track(rtx_track_id)).first;

                auto roctx_track_desc = roctx_track_it->second.Serialize();
                std::string roctx_track_str = rocprofiler::string_printf("ROCTX Start/Stop Ranges");
                roctx_track_desc.set_name(roctx_track_str);
                perfetto::TrackEvent::SetTrackDescriptor(roctx_track_it->second, roctx_track_desc);
              }
            }
            auto& roctx_track = roctx_track_it->second;
            perfetto::DynamicString roctx_message_pft(
                (!roctx_message.empty() ? roctx_message.c_str() : ""));
            TRACE_EVENT_BEGIN("ROCTX_RANGE_START_STOP", roctx_message_pft, roctx_track,
                              tracer_record.timestamps.begin.value, "Timestamp(ns)",
                              tracer_record.timestamps.begin.value, "RocTx ID", roctx_id);
            roctx_start_track_entries_++;
            break;
          }
          case 2: {
            std::unordered_map<uint64_t, perfetto::Track>::iterator roctx_track_it;
            {
              uint64_t rtx_track_id = getTrackID(TrackType::ROCTX_PUSH_RANGE, machine_id_, 0, thread_id);
              roctx_track_it = roctx_push_tracks_.find(rtx_track_id);
              if (roctx_track_it == roctx_push_tracks_.end()) {
                roctx_track_it =
                    roctx_push_tracks_.emplace(rtx_track_id, perfetto::Track(rtx_track_id)).first;

                auto roctx_track_desc = roctx_track_it->second.Serialize();
                std::string roctx_track_str = rocprofiler::string_printf("ROCTX Push/Pop Ranges");
                roctx_track_desc.set_name(roctx_track_str);
                perfetto::TrackEvent::SetTrackDescriptor(roctx_track_it->second, roctx_track_desc);
              }
            }
            auto& roctx_track = roctx_track_it->second;
            TRACE_EVENT_END("ROCTX_RANGE_PUSH_POP", roctx_track,
                            tracer_record.timestamps.begin.value);
            roctx_push_track_entries_--;
            break;
          }
          case 4: {
            std::unordered_map<uint64_t, perfetto::Track>::iterator roctx_track_it;
            {
              uint64_t rtx_track_id = getTrackID(TrackType::ROCTX_START_RANGE, machine_id_, 0, thread_id);
              roctx_track_it = roctx_start_tracks_.find(rtx_track_id);
              if (roctx_track_it == roctx_start_tracks_.end()) {
                roctx_track_it =
                    roctx_start_tracks_.emplace(rtx_track_id, perfetto::Track(rtx_track_id)).first;

                auto roctx_track_desc = roctx_track_it->second.Serialize();
                std::string roctx_track_str = rocprofiler::string_printf("ROCTX Start/Stop Ranges");
                roctx_track_desc.set_name(roctx_track_str);
                perfetto::TrackEvent::SetTrackDescriptor(roctx_track_it->second, roctx_track_desc);
              }
            }
            auto& roctx_track = roctx_track_it->second;
            TRACE_EVENT_END("ROCTX_RANGE_START_STOP", roctx_track,
                            tracer_record.timestamps.begin.value);
            roctx_start_track_entries_--;
            break;
          }
          default: {
            rocprofiler::warning("ROCPROFILER_TOOL: Wrong ROCTx Operation ID!");
          }
        }
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        std::unordered_map<uint64_t, perfetto::Track>::iterator hsa_track_it;
        {
          uint64_t hsa_track_id = getTrackID(TrackType::HSAAPI, machine_id_, 0, thread_id);
          hsa_track_it = hsa_tracks_.find(hsa_track_id);
          if (hsa_track_it == hsa_tracks_.end()) {
            hsa_track_it = hsa_tracks_.emplace(hsa_track_id, perfetto::Track(hsa_track_id)).first;
            auto hsa_track_desc = hsa_track_it->second.Serialize();
            std::stringstream hsa_track_str;
            hsa_track_str << "HSA API " << thread_id;
            hsa_track_desc.set_name(hsa_track_str.str());
            perfetto::TrackEvent::SetTrackDescriptor(hsa_track_it->second, hsa_track_desc);
          }
        }
        auto& hsa_track = hsa_track_it->second;
        if (tracer_record.phase == ROCPROFILER_PHASE_ENTER)
          TRACE_EVENT_BEGIN("HSA_API", perfetto::DynamicString(operation_name_c), hsa_track,
                            tracer_record.timestamps.begin.value,
                            perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
        if (tracer_record.phase == ROCPROFILER_PHASE_EXIT)
          TRACE_EVENT_END("HSA_API", hsa_track, tracer_record.timestamps.end.value);
        if (tracer_record.phase == ROCPROFILER_PHASE_NONE) {
          TRACE_EVENT_BEGIN("HSA_API", perfetto::DynamicString(operation_name_c), hsa_track,
                            tracer_record.timestamps.begin.value,
                            perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
          TRACE_EVENT_END("HSA_API", hsa_track, tracer_record.timestamps.end.value);
        }
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        std::unordered_map<uint64_t, perfetto::Track>::iterator hip_track_it;
        {
          uint64_t hipapi_track_id = getTrackID(TrackType::HIPAPI, machine_id_, 0, thread_id);
          hip_track_it = hip_tracks_.find(hipapi_track_id);
          if (hip_track_it == hip_tracks_.end()) {
            hip_track_it =
                hip_tracks_.emplace(hipapi_track_id, perfetto::Track(hipapi_track_id)).first;

            auto hip_track_desc = hip_track_it->second.Serialize();
            std::stringstream hip_track_str;
            hip_track_str << "HIP API " << thread_id;
            hip_track_desc.set_name(hip_track_str.str());
            perfetto::TrackEvent::SetTrackDescriptor(hip_track_it->second, hip_track_desc);
          }
        }
        auto& hip_track = hip_track_it->second;
        if (tracer_record.phase == ROCPROFILER_PHASE_ENTER)
          TRACE_EVENT_BEGIN("HIP_API", perfetto::DynamicString(operation_name_c), hip_track,
                            tracer_record.timestamps.begin.value,
                            perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
        if (tracer_record.phase == ROCPROFILER_PHASE_EXIT)
          TRACE_EVENT_END("HIP_API", hip_track, tracer_record.timestamps.end.value);
        if (tracer_record.phase == ROCPROFILER_PHASE_NONE) {
          TRACE_EVENT_BEGIN("HIP_API", perfetto::DynamicString(operation_name_c), hip_track,
                            tracer_record.timestamps.begin.value,
                            perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
          TRACE_EVENT_END("HIP_API", hip_track, tracer_record.timestamps.end.value);
        }
        break;
      }
      case ACTIVITY_DOMAIN_EXT_API: {
        printf("Warning: External API is not supported!\n");
        break;
      }
      case ACTIVITY_DOMAIN_HIP_OPS: {
        std::size_t pos = std::string::npos;
        if (tracer_record.name) {
          auto kernel_name_it = kernel_names_map.find(tracer_record.name);
          if (kernel_name_it == kernel_names_map.end()) {
            kernel_name_it =
                kernel_names_map
                    .emplace(
                        tracer_record.name,
                        rocprofiler::truncate_name(rocprofiler::cxx_demangle(tracer_record.name)))
                    .first;
          }
          TRACE_EVENT_BEGIN("HIP_OPS", perfetto::DynamicString(kernel_name_it->second.c_str()),
                            gpu_track, tracer_record.timestamps.begin.value, "Agent ID",
                            tracer_record.agent_id.handle, "Process ID", GetPid(),
                            perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
          TRACE_EVENT_END("HIP_OPS", gpu_track, tracer_record.timestamps.end.value);
        } else {
          // MEM Copies are not correlated to GPUs, so they need a special track
          pos = operation_name_c ? std::string_view(operation_name_c).find("Copy")
                                 : std::string::npos;

          if (std::string::npos == pos) {
            TRACE_EVENT_BEGIN("HIP_OPS", perfetto::DynamicString(operation_name_c), gpu_track,
                              tracer_record.timestamps.begin.value, "Process ID", GetPid(),
                              perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
            TRACE_EVENT_END("HIP_OPS", gpu_track, tracer_record.timestamps.end.value);
          } else {
            TRACE_EVENT_BEGIN("MEM_COPIES", perfetto::DynamicString(operation_name_c),
                              mem_copies_track, tracer_record.timestamps.begin.value, "Process ID",
                              GetPid(),
                              perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
            TRACE_EVENT_END("MEM_COPIES", mem_copies_track, tracer_record.timestamps.end.value);
          }
        }
        break;
      }
      case ACTIVITY_DOMAIN_HSA_OPS: {
        TRACE_EVENT_BEGIN("MEM_COPIES", perfetto::DynamicString(operation_name_c), mem_copies_track,
                          tracer_record.timestamps.begin.value, "Process ID", GetPid(),
                          perfetto::Flow::ProcessScoped(tracer_record.correlation_id.value));
        TRACE_EVENT_END("MEM_COPIES", mem_copies_track, tracer_record.timestamps.end.value);
        break;
      }
      default: {
        rocprofiler::warning("Ignored record for domain %d", tracer_record.domain);
        break;
      }
    }
    return 0;
  }

  int WriteBufferRecords(const rocprofiler_record_header_t* begin,
                         const rocprofiler_record_header_t* end,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    if (!tracing_session_) rocprofiler::warning("Tracing session is deleted!\n");
    while (begin < end) {
      if (!begin) return 0;
      switch (begin->kind) {
        case ROCPROFILER_PROFILER_RECORD: {
          rocprofiler_record_profiler_t* profiler_record =
              const_cast<rocprofiler_record_profiler_t*>(
                  reinterpret_cast<const rocprofiler_record_profiler_t*>(begin));
          FlushProfilerRecord(*profiler_record, session_id);
          break;
        }
        case ROCPROFILER_TRACER_RECORD: {
          rocprofiler_record_tracer_t* tracer_record = const_cast<rocprofiler_record_tracer_t*>(
              reinterpret_cast<const rocprofiler_record_tracer_t*>(begin));
          FlushTracerRecord(*tracer_record, session_id);
          break;
        }
        default:
          break;
      }
      rocprofiler_next_record(begin, &begin, session_id, buffer_id);
    }
    return 0;
  }

  bool IsValid() const { return is_valid_; }

 private:
  std::unique_ptr<perfetto::TracingSession> tracing_session_;
  fs::path output_prefix_;
  int file_descriptor_;
  bool is_valid_{false};
  size_t roctx_start_track_entries_{0};
  size_t roctx_push_track_entries_{0};

  // Correlate stream id(s) with correlation id(s) to identify the stream id of every HIP activity
  std::unordered_map<uint64_t, uint64_t> stream_ids_;

  // Callback Tracks
  std::unordered_map<uint64_t, perfetto::Track> roctx_mark_tracks_, roctx_push_tracks_,
      roctx_start_tracks_, hsa_tracks_, hip_tracks_, hip_ext_tracks_, mem_copies_tracks_;

  // Activity Tracks
  std::unordered_map<uint64_t, perfetto::Track> queue_tracks_;

  std::unordered_map<uint64_t, std::vector<perfetto::CounterTrack>> counter_tracks_;

  std::atomic<uint64_t> track_counter_{GetPid()};
  std::vector<uint64_t> track_ids_used_;

  std::string hostname_;
  uint64_t machine_id_;

  std::ofstream stream_;

  std::unordered_map<TrackID, uint64_t> track_ids;
  std::unordered_map<uint64_t, perfetto::Track> device_tracks;
  std::unordered_map<uint64_t, perfetto::Track> hip_stream_tracks;
  std::unordered_map<std::string, std::string> kernel_names_map;

  uint64_t getTrackID(TrackType type, uint64_t machine, uint64_t device, uint64_t queue) {
    TrackID id(type, machine, device, queue);

    auto it = track_ids.find(id);
    if (it == track_ids.end()) it = track_ids.emplace(id, getUniqueID()).first;

    return it->second;
  }

  uint64_t getUniqueID() { return cur_unique_id.fetch_add(1); };
  std::atomic<uint64_t> cur_unique_id{uint64_t(GetPid()) << 30};
};

perfetto_plugin_t* perfetto_plugin = nullptr;

}  // namespace

int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                  uint32_t rocprofiler_minor_version, void* data) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version > ROCPROFILER_VERSION_MINOR)
    return -1;

  // if (perfetto_plugin != nullptr && perfetto_plugin->IsValid()) return -1;

  std::lock_guard<std::mutex> lock(writing_lock);
  if (perfetto_plugin != nullptr) perfetto_plugin->delete_perfetto_plugin();

  static int perfetto_init_count = 0;
  perfetto_plugin = new perfetto_plugin_t(perfetto_init_count++);

  if (perfetto_plugin->IsValid()) return 0;

  // delete perfetto_plugin;
  // perfetto_plugin = nullptr;
  return -1;
}

void rocprofiler_plugin_finalize() {
  std::lock_guard<std::mutex> lock(writing_lock);
  if (!perfetto_plugin) return;
  if (perfetto_plugin->delete_perfetto_plugin()) {
    delete perfetto_plugin;
    perfetto_plugin = nullptr;
  }
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  std::lock_guard<std::mutex> lock(writing_lock);
  if (!perfetto_plugin || !perfetto_plugin->IsValid()) return -1;
  return perfetto_plugin->WriteBufferRecords(begin, end, session_id, buffer_id);
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record) {
  if (record.header.id.handle == 0) return 0;

  std::lock_guard<std::mutex> lock(writing_lock);
  if (!perfetto_plugin || !perfetto_plugin->IsValid()) return -1;
  return perfetto_plugin->FlushTracerRecord(record, rocprofiler_session_id_t{0});
}
