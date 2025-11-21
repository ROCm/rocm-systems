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

#include <cassert>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <string>
#include <memory>
#include <limits>
#include <fstream>
#include <time.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include "hsa_prof_str.h"

#include <hip/hip_runtime.h>
#include <hip/amd_detail/hip_prof_str.h>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"
#include "../utils.h"

#include "barectf.h"
#include "barectf_event_record.h"
#include "barectf_tracer.h"
#include "plugin.h"

#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;

namespace rocm_ctf {
namespace {

// Abstract tracer event record using the barectf context type `CtxT`.
template <typename CtxT> class TracerEventRecord : public BarectfEventRecord<CtxT> {
 protected:
  explicit TracerEventRecord(const rocprofiler_record_tracer_t& record,
                             const std::uint64_t clock_val)
      : BarectfEventRecord<CtxT>{clock_val},
        op_{record.operation_id.id},
        thread_id_{record.thread_id.value},
        queue_id_{record.queue_id.handle},
        agent_id_{record.agent_id.handle},
        correlation_id_{record.correlation_id.value} {}

  std::uint32_t GetOp() const noexcept { return op_; }
  std::uint32_t GetThreadId() const noexcept { return thread_id_; }
  std::uint64_t GetQueueId() const noexcept { return queue_id_; }
  std::uint64_t GetAgentId() const noexcept { return agent_id_; }
  std::uint64_t GetCorrelationId() const noexcept { return correlation_id_; }

 private:
  std::uint32_t op_;
  std::uint32_t thread_id_;
  std::uint64_t queue_id_;
  std::uint64_t agent_id_;
  std::uint64_t correlation_id_;
};

// Returns the beginning clock value of the tracer or profiler record
// `record`.
template <typename RecordT> std::uint64_t GetRecordBeginClockVal(const RecordT& record) {
  return record.timestamps.begin.value;
}

// Returns the end clock value of the tracer or profiler record
// `record`.
template <typename RecordT> std::uint64_t GetRecordEndClockVal(const RecordT& record) {
  return record.timestamps.end.value;
}

// Queries allocated string data using the size query function
// `query_size_func` and the data query function `query_data_func`,
// returning the corresponding string and freeing temporary allocated
// memory.
//
// Returns an empty string if anything goes wrong.
template <typename QuerySizeFuncT, typename QueryDataFuncT>
std::string QueryAllocStr(QuerySizeFuncT&& query_size_func, QueryDataFuncT&& query_data_func) {
  // Query size first.
  std::size_t size = 0;
  [[maybe_unused]] auto ret = query_size_func(&size);

  assert(ret == ROCPROFILER_STATUS_SUCCESS && "Query size");

  if (size == 0) {
    // No size: return empty string.
    return {};
  }

  // Query data (allocated by query_data_func()).
  char* alloc_str = nullptr;

  ret = query_data_func(&alloc_str);
  assert(ret == ROCPROFILER_STATUS_SUCCESS && "Query data");

  if (!alloc_str) {
    // No data: return empty string.
    return {};
  }

  // Allocate return value.
  std::string str_ret{alloc_str};

  // Free allocated data.
  std::free(alloc_str);

  // Return string object.
  return str_ret;
}

// rocTX event record.
class RocTxEventRecord final : public TracerEventRecord<barectf_roctx_ctx> {
 public:
  explicit RocTxEventRecord(const rocprofiler_record_tracer_t& record,
                            const rocprofiler_session_id_t session_id)
      : TracerEventRecord<barectf_roctx_ctx>{record, GetRecordBeginClockVal(record)},
        id_{record.external_id.id} {
    msg_ = record.name ? record.name : "";
  }

  void Write(barectf_roctx_ctx& barectf_ctx) const override {
    barectf_roctx_trace_roctx(&barectf_ctx, GetThreadId(), id_, msg_.c_str());
  }

 private:
  std::uint64_t id_;
  std::string msg_;
};

// Abstract HSA API event record.
class HsaApiEventRecord : public TracerEventRecord<barectf_hsa_api_ctx> {
 protected:
  explicit HsaApiEventRecord(const rocprofiler_record_tracer_t& record,
                             const rocprofiler_session_id_t session_id,
                             const std::uint64_t clock_val)
      : TracerEventRecord<barectf_hsa_api_ctx>{record, clock_val} {
    if (record.api_data.hsa) api_data_ = *(record.api_data.hsa);
  }
  explicit HsaApiEventRecord(const rocprofiler_record_tracer_t& record,
                             const std::uint64_t clock_val, hsa_api_data_t& api_data)
      : TracerEventRecord<barectf_hsa_api_ctx>{record, clock_val}, api_data_(api_data) {}
  const hsa_api_data_t& GetApiData() const noexcept { return api_data_; }

 private:
  hsa_api_data_t api_data_;
};

// HSA API event record (beginning).
class HsaApiEventRecordBegin final : public HsaApiEventRecord {
 public:
  explicit HsaApiEventRecordBegin(const rocprofiler_record_tracer_t& record,
                                  const rocprofiler_session_id_t session_id)
      : HsaApiEventRecord{record, session_id, GetRecordBeginClockVal(record)} {}
  explicit HsaApiEventRecordBegin(const rocprofiler_record_tracer_t& record,
                                  hsa_api_data_t& api_data)
      : HsaApiEventRecord{record, GetRecordBeginClockVal(record), api_data} {}

  void Write(barectf_hsa_api_ctx& barectf_ctx) const override {
    // Include generated switch statement.
#include "hsa_begin.cpp.i"
  }
};

// HSA API event record (end).
class HsaApiEventRecordEnd final : public HsaApiEventRecord {
 public:
  explicit HsaApiEventRecordEnd(const rocprofiler_record_tracer_t& record,
                                const rocprofiler_session_id_t session_id)
      : HsaApiEventRecord{record, session_id, GetRecordEndClockVal(record)} {}
  explicit HsaApiEventRecordEnd(const rocprofiler_record_tracer_t& record, hsa_api_data_t& api_data)
      : HsaApiEventRecord{record, GetRecordBeginClockVal(record), api_data} {}

  void Write(barectf_hsa_api_ctx& barectf_ctx) const override {
    // Include generated switch statement.
#include "hsa_end.cpp.i"
  }
};

// Abstract HIP API event record.
class HipApiEventRecord : public TracerEventRecord<barectf_hip_api_ctx> {
 protected:
  explicit HipApiEventRecord(const rocprofiler_record_tracer_t& record,
                             const rocprofiler_session_id_t session_id,
                             const std::uint64_t clock_val)
      : TracerEventRecord<barectf_hip_api_ctx>{record, clock_val},
        api_data_{record.api_data.hip ? *(record.api_data.hip) : hip_api_data_t{}},
        kernel_name_{record.name ? record.name : std::string{}} {}
  explicit HipApiEventRecord(const rocprofiler_record_tracer_t& record,
                             const std::uint64_t clock_val, hip_api_data_t& api_data,
                             std::string kernel_name)
      : TracerEventRecord<barectf_hip_api_ctx>{record, clock_val},
        api_data_{api_data},
        kernel_name_{kernel_name} {}
  const hip_api_data_t& GetApiData() const noexcept { return api_data_; }
  const std::string& GetKernelName() const noexcept { return kernel_name_; }

 private:
  hip_api_data_t api_data_;
  std::string kernel_name_;
};

// HIP API event record (beginning).
class HipApiEventRecordBegin final : public HipApiEventRecord {
 public:
  explicit HipApiEventRecordBegin(const rocprofiler_record_tracer_t& record,
                                  const rocprofiler_session_id_t session_id)
      : HipApiEventRecord{record, session_id, GetRecordBeginClockVal(record)} {}
  explicit HipApiEventRecordBegin(const rocprofiler_record_tracer_t& record,
                                  hip_api_data_t& api_data, std::string kernel_name)
      : HipApiEventRecord{record, GetRecordBeginClockVal(record), api_data, kernel_name} {}

  void Write(barectf_hip_api_ctx& barectf_ctx) const override {
    // Include generated switch statement.
#include "hip_begin.cpp.i"
  }
};

// HIP API event record (end).
class HipApiEventRecordEnd final : public HipApiEventRecord {
 public:
  explicit HipApiEventRecordEnd(const rocprofiler_record_tracer_t& record,
                                const rocprofiler_session_id_t session_id)
      : HipApiEventRecord{record, session_id, GetRecordEndClockVal(record)} {}
  explicit HipApiEventRecordEnd(const rocprofiler_record_tracer_t& record, hip_api_data_t& api_data,
                                std::string kernel_name)
      : HipApiEventRecord{record, GetRecordBeginClockVal(record), api_data, kernel_name} {}

  void Write(barectf_hip_api_ctx& barectf_ctx) const override {
    // Include generated switch statement.
#include "hip_end.cpp.i"
  }
};

// HSA API handle type event record.
class HsaHandleTypeEventRecord final : public BarectfEventRecord<barectf_hsa_handles_ctx> {
 public:
  enum class Type {
    CPU = 0,
    GPU = 1,
  };

  explicit HsaHandleTypeEventRecord(const std::uint64_t handle, const Type type)
      : BarectfEventRecord<barectf_hsa_handles_ctx>{0}, handle_{handle}, type_{type} {}

  void Write(barectf_hsa_handles_ctx& barectf_ctx) const override {
    barectf_hsa_handles_trace_hsa_handle_type(&barectf_ctx, handle_,
                                              static_cast<std::uint8_t>(type_));
  }

 private:
  std::uint64_t handle_;
  Type type_;
};

// Abstract API operation event record.
class ApiOpEventRecord : public TracerEventRecord<barectf_api_ops_ctx> {
 protected:
  explicit ApiOpEventRecord(const rocprofiler_record_tracer_t& record,
                            const std::uint64_t clock_val)
      : TracerEventRecord<barectf_api_ops_ctx>{record, clock_val} {}
};

// HSA API operation event record (beginning).
class HsaOpEventRecordBegin final : public ApiOpEventRecord {
 public:
  explicit HsaOpEventRecordBegin(const rocprofiler_record_tracer_t& record)
      : ApiOpEventRecord{record, GetRecordBeginClockVal(record)} {}

  void Write(barectf_api_ops_ctx& barectf_ctx) const override {
    barectf_api_ops_trace_hsa_op_begin(&barectf_ctx, GetThreadId(), GetQueueId(), GetAgentId(),
                                       GetCorrelationId());
  }
};

// HSA API operation event record (end).
class HsaOpEventRecordEnd final : public ApiOpEventRecord {
 public:
  explicit HsaOpEventRecordEnd(const rocprofiler_record_tracer_t& record)
      : ApiOpEventRecord{record, GetRecordEndClockVal(record)} {}

  void Write(barectf_api_ops_ctx& barectf_ctx) const override {
    barectf_api_ops_trace_hsa_op_end(&barectf_ctx, GetThreadId(), GetQueueId(), GetAgentId(),
                                     GetCorrelationId());
  }
};

// HIP API operation event record (beginning).
class HipOpEventRecordBegin final : public ApiOpEventRecord {
 public:
  explicit HipOpEventRecordBegin(const rocprofiler_record_tracer_t& record)
      : ApiOpEventRecord{record, GetRecordBeginClockVal(record)},
        kernel_name_{QueryKernelName(record)} {}

  void Write(barectf_api_ops_ctx& barectf_ctx) const override {
    barectf_api_ops_trace_hip_op_begin(&barectf_ctx, GetThreadId(), GetQueueId(), GetAgentId(),
                                       GetCorrelationId(), kernel_name_.c_str());
  }

 private:
  // Queries and returns the kernel name of the record `record`.
  //
  // Returns an empty string if not available.
  static std::string QueryKernelName(const rocprofiler_record_tracer_t& record) {
    if (record.name) {
      // Return demangled version.
      return rocprofiler::cxx_demangle(record.name);
    }

    return {};
  }

  std::string kernel_name_;
};

// HIP API operation event record (end).
class HipOpEventRecordEnd final : public ApiOpEventRecord {
 public:
  explicit HipOpEventRecordEnd(const rocprofiler_record_tracer_t& record)
      : ApiOpEventRecord{record, GetRecordEndClockVal(record)} {}

  void Write(barectf_api_ops_ctx& barectf_ctx) const override {
    barectf_api_ops_trace_hip_op_end(&barectf_ctx, GetThreadId(), GetQueueId(), GetAgentId(),
                                     GetCorrelationId());
  }
};

// Profiler record base.
class ProfilerEventRecord : public BarectfEventRecord<barectf_profiler_ctx> {
 public:
  explicit ProfilerEventRecord(const rocprofiler_record_profiler_t& record,
                               const rocprofiler_session_id_t session_id)
      : BarectfEventRecord<barectf_profiler_ctx>{GetRecordBeginClockVal(record)},
        dispatch_{record.header.id.handle},
        gpu_id_{record.gpu_id.handle},
        queue_id_{record.queue_id.handle},
        queue_index_{record.queue_idx.value},
        process_id_{GetPid()},
        thread_id_{record.thread_id.value},
        kernel_id_{record.kernel_id.handle},
        kernel_name_{QueryKernelName(record)},
        counter_infos_{QueryCounterInfos(record, session_id)} {}

  void Write(barectf_profiler_ctx& barectf_ctx) const override {
    barectf_profiler_trace_profiler_record(
        &barectf_ctx, dispatch_, gpu_id_, queue_id_, queue_index_, process_id_, thread_id_,
        kernel_id_, kernel_name_.c_str(), counter_infos_.names.size(), counter_infos_.names.data(),
        counter_infos_.values.size(), counter_infos_.values.data());
  }

 protected:
  // Counter infos.
  //
  // `names[i]` names the counter value `values[i]`.
  struct CounterInfos final {
    // `names_storage` owns the strings while the elements of `names`
    // point to the internal C strings of `names_storage`.
    //
    // This is needed because barectf expects an array of contiguous
    // C string pointers.
    std::vector<std::string> names_storage;
    std::vector<const char*> names;

    // Counter values.
    std::vector<std::uint64_t> values;
  };

  std::uint64_t GetDispatch() const noexcept { return dispatch_; }
  std::uint64_t GetGpuId() const noexcept { return gpu_id_; }
  std::uint64_t GetQueueId() const noexcept { return queue_id_; }
  std::uint64_t GetQueueIndex() const noexcept { return queue_index_; }
  std::uint32_t GetProcessId() const noexcept { return process_id_; }
  std::uint32_t GetThreadId() const noexcept { return thread_id_; }
  std::uint64_t GetKernelId() const noexcept { return kernel_id_; }
  const std::string& GetKernelName() const noexcept { return kernel_name_; }
  const CounterInfos& GetCounterInfos() const noexcept { return counter_infos_; }

 private:
  // Queries and returns the kernel name of the record `record`.
  //
  // Returns an empty string if not available.
  static std::string QueryKernelName(const rocprofiler_record_profiler_t& record) {
    const auto kernel_name = QueryAllocStr(
        [&record](const auto size) {
          return rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME, record.kernel_id,
                                                    size);
        },
        [&record](const auto str) {
          return rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME, record.kernel_id,
                                               const_cast<const char**>(str));
        });

    if (kernel_name.size() <= 1) {
      return {};
    }

    // Return truncated and demangled version.
    return rocprofiler::truncate_name(rocprofiler::cxx_demangle(kernel_name));
  }

  // Queries and returns the counter infos of the record `record` and
  // session ID `session_id`.
  static CounterInfos QueryCounterInfos(const rocprofiler_record_profiler_t& record,
                                        const rocprofiler_session_id_t session_id) {
    if (!record.counters) {
      // No counters.
      return {};
    }

    CounterInfos infos;

    for (std::size_t i = 0; i < record.counters_count.value; ++i) {
      auto& counter = record.counters[i];

      if (counter.counter_handler.handle == 0) {
        // Not available: continue.
        continue;
      }

      // Query counter name size first
      std::size_t counter_name_size = 0;
      [[maybe_unused]] auto ret = rocprofiler_query_counter_info_size(
          session_id, ROCPROFILER_COUNTER_NAME, counter.counter_handler, &counter_name_size);

      assert(ret == ROCPROFILER_STATUS_SUCCESS && "Query counter name size");

      if (counter_name_size == 0) {
        // No size: continue.
        continue;
      }

      // Query counter name (borrowed from `record`: no need to free).
      const char* counter_name = nullptr;

      ret = rocprofiler_query_counter_info(session_id, ROCPROFILER_COUNTER_NAME,
                                           counter.counter_handler, &counter_name);
      assert(ret == ROCPROFILER_STATUS_SUCCESS && "Query counter name");

      if (!counter_name) {
        // Not available: continue.
        continue;
      }

      // Push back infos.
      infos.names_storage.emplace_back(counter_name);
      infos.names.push_back(infos.names_storage.back().c_str());
      infos.values.push_back(counter.value.value);
    }

    return infos;
  }

  std::uint64_t dispatch_;
  std::uint64_t gpu_id_;
  std::uint64_t queue_id_;
  std::uint64_t queue_index_;
  std::uint32_t process_id_;
  std::uint32_t thread_id_;
  std::uint64_t kernel_id_;
  std::string kernel_name_;
  CounterInfos counter_infos_;
};

// Profiler record base.
class ProfilerWithKernelPropsEventRecord final : public ProfilerEventRecord {
 private:
  // According to `plugin/file/file.cpp`:
  //
  // > Taken from rocprofiler: The size hasn't changed in recent past
  static constexpr std::uint32_t lds_block_size_ = 128 * 4;

 public:
  explicit ProfilerWithKernelPropsEventRecord(const rocprofiler_record_profiler_t& record,
                                              const rocprofiler_session_id_t session_id)
      : ProfilerEventRecord{record, session_id},
        grid_size_{record.kernel_properties.grid_size},
        workgroup_size_{record.kernel_properties.workgroup_size},
        lds_size_{
            ((record.kernel_properties.lds_size + (lds_block_size_ - 1)) & ~(lds_block_size_ - 1))},
        scratch_size_{record.kernel_properties.scratch_size},
        arch_vgpr_count_{record.kernel_properties.arch_vgpr_count},
        accum_vgpr_count_{record.kernel_properties.accum_vgpr_count},
        sgpr_count_{record.kernel_properties.sgpr_count},
        wave_size_{record.kernel_properties.wave_size},
        signal_handle_{record.kernel_properties.signal_handle} {}

  void Write(barectf_profiler_ctx& barectf_ctx) const override {
    barectf_profiler_trace_profiler_record_with_kernel_properties(
        &barectf_ctx, GetDispatch(), GetGpuId(), GetQueueId(), GetQueueIndex(), GetProcessId(),
        GetThreadId(), GetKernelId(), GetKernelName().c_str(), GetCounterInfos().names.size(),
        GetCounterInfos().names.data(), GetCounterInfos().values.size(),
        GetCounterInfos().values.data(), grid_size_, workgroup_size_, lds_size_, scratch_size_,
        arch_vgpr_count_, accum_vgpr_count_, sgpr_count_, wave_size_, signal_handle_);
  }

 private:
  std::uint64_t grid_size_;
  std::uint64_t workgroup_size_;
  std::uint64_t lds_size_;
  std::uint64_t scratch_size_;
  std::uint64_t arch_vgpr_count_;
  std::uint64_t accum_vgpr_count_;
  std::uint64_t sgpr_count_;
  std::uint64_t wave_size_;
  std::uint64_t signal_handle_;
};

}  // namespace

Plugin::Plugin(const std::size_t packet_size, const fs::path& trace_dir,
               const fs::path& metadata_stream_path)
    : roctx_tracer_{packet_size, trace_dir, "roctx_"},
      hsa_api_tracer_{packet_size, trace_dir, "hsa_api_"},
      hip_api_tracer_{packet_size, trace_dir, "hip_api_"},
      api_ops_tracer_{packet_size, trace_dir, "api_ops_"},
      hsa_handles_tracer_{packet_size, trace_dir, "hsa_handles_"},
      profiler_tracer_{packet_size, trace_dir, "profiler_"} {
  // Make sure the trace directory doesn't exist.
  if (fs::exists(trace_dir)) {
    std::ostringstream ss;

    ss << "CTF trace directory `" << trace_dir.string() << "` already exists";
    throw std::runtime_error{ss.str()};
  }

  // Make sure the metadata stream file exists.
  if (!fs::exists(metadata_stream_path)) {
    std::ostringstream ss;

    ss << "CTF metadata stream file `" << metadata_stream_path.string() << "` doesn't exist";
    throw std::runtime_error{ss.str()};
  }

  // Create trace directory.
  if (!fs::create_directory(trace_dir)) {
    std::ostringstream ss;

    ss << "Cannot create the CTF trace directory `" << trace_dir.string() << "`";
    throw std::runtime_error{ss.str()};
  }

  // Copy adjusted metadata stream file to trace directory.
  try {
    CopyAdjustedMetadataStreamFile(metadata_stream_path, trace_dir);
  } catch (const std::exception& exc) {
    std::ostringstream ss;

    ss << "Cannot adjust and copy metadata stream file `" << metadata_stream_path.string()
       << "` to the CTF trace directory `" << trace_dir.string() << "`: " << exc.what();
    throw std::runtime_error{ss.str()};
  }

  // Write HSA handle type event records.
  WriteHsaHandleTypes();
}

void Plugin::HandleTracerRecord(const rocprofiler_record_tracer_t& record,
                                const rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock{lock_};

  // Depending on the domain, create and add an event record to the
  // corresponding tracer.
  switch (record.domain) {
    case ACTIVITY_DOMAIN_ROCTX:
      roctx_tracer_.AddEventRecord(std::make_shared<const RocTxEventRecord>(record, session_id));
      break;
    case ACTIVITY_DOMAIN_HSA_API: {
      /*If data is nullptr then the call is asynchromous*/
      if (record.api_data.hsa == nullptr) {
        hsa_api_tracer_.AddEventRecord(
            std::make_shared<const HsaApiEventRecordBegin>(record, session_id));
        hsa_api_tracer_.AddEventRecord(
            std::make_shared<const HsaApiEventRecordEnd>(record, session_id));
      } else {
        hsa_api_data_t hsa_api_data = *(record.api_data.hsa);
        hsa_api_tracer_.AddEventRecord(
            std::make_shared<const HsaApiEventRecordBegin>(record, hsa_api_data));
        hsa_api_tracer_.AddEventRecord(
            std::make_shared<const HsaApiEventRecordEnd>(record, hsa_api_data));
      }
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      /*If data is nullptr then the call is asynchromous*/
      if (record.api_data.hip == nullptr) {
        hip_api_tracer_.AddEventRecord(
            std::make_shared<const HipApiEventRecordBegin>(record, session_id));
        hip_api_tracer_.AddEventRecord(
            std::make_shared<const HipApiEventRecordEnd>(record, session_id));
      } else {
        std::string kernel_name;
        hip_api_data_t hip_api_data = *(record.api_data.hip);
        if (record.name != nullptr)
          kernel_name = rocprofiler::truncate_name(rocprofiler::cxx_demangle(std::string(record.name)));
        else
          kernel_name = "";
        hip_api_tracer_.AddEventRecord(
            std::make_shared<const HipApiEventRecordBegin>(record, hip_api_data, kernel_name));
        hip_api_tracer_.AddEventRecord(
            std::make_shared<const HipApiEventRecordEnd>(record, hip_api_data, kernel_name));
      }
      break;
    }
    case ACTIVITY_DOMAIN_HSA_OPS:
      api_ops_tracer_.AddEventRecord(std::make_shared<const HsaOpEventRecordBegin>(record));
      api_ops_tracer_.AddEventRecord(std::make_shared<const HsaOpEventRecordEnd>(record));
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      api_ops_tracer_.AddEventRecord(std::make_shared<const HipOpEventRecordBegin>(record));
      api_ops_tracer_.AddEventRecord(std::make_shared<const HipOpEventRecordEnd>(record));
      break;
    default:
      // Warn
      std::cerr << "rocm_ctf::Plugin::HandleTracerRecord(): "
                << "ignoring record for unknown domain #" << record.domain << std::endl;
      break;
  }
}

void Plugin::HandleProfilerRecord(const rocprofiler_record_profiler_t& record,
                                  const rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock{lock_};
  profiler_tracer_.AddEventRecord(
      std::make_shared<const ProfilerWithKernelPropsEventRecord>(record, session_id));
}

void Plugin::HandleBufferRecords(const rocprofiler_record_header_t* begin,
                                 const rocprofiler_record_header_t* const end,
                                 const rocprofiler_session_id_t session_id,
                                 const rocprofiler_buffer_id_t buffer_id) {
  while (begin && begin < end) {
    if (begin->kind == ROCPROFILER_TRACER_RECORD) {
      HandleTracerRecord(*reinterpret_cast<const rocprofiler_record_tracer_t*>(begin), session_id);
    } else {
      assert(begin->kind == ROCPROFILER_PROFILER_RECORD);
      HandleProfilerRecord(*reinterpret_cast<const rocprofiler_record_profiler_t*>(begin),
                           session_id);
    }

    rocprofiler_next_record(begin, &begin, session_id, buffer_id);
  }
}

void Plugin::WriteHsaHandleTypes() {
  [[maybe_unused]] const auto status = hsa_iterate_agents(
      [](const auto agent, const auto user_data) {
        auto& tracer = *static_cast<HsaHandlesTracer*>(user_data);
        hsa_device_type_t type;

        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS) {
          return HSA_STATUS_ERROR;
        }

        using Type = HsaHandleTypeEventRecord::Type;

        auto event_record = std::make_shared<HsaHandleTypeEventRecord>(
            agent.handle, type == HSA_DEVICE_TYPE_CPU ? Type::CPU : Type::GPU);

        tracer.AddEventRecord(std::move(event_record));
        return HSA_STATUS_SUCCESS;
      },
      &hsa_handles_tracer_);

  assert(status == HSA_STATUS_SUCCESS && "Iterate HSA agents");
}

namespace {

constexpr std::uint64_t ns_per_s = 1'000'000'000ULL;

// Samples the ROCProfiler clock and returns the value.
std::uint64_t GetClkVal() {
  rocprofiler_timestamp_t ts;
  [[maybe_unused]] const auto ret = rocprofiler_get_timestamp(&ts);

  assert(ret == ROCPROFILER_STATUS_SUCCESS && "Get timestamp");
  return ts.value;
}

// Updates `offset` and `delta`, if needed, to a more accurate clock
// class offset and a smaller ROCProfiler clock value delta.
//
// This function samples the ROCProfiler clock twice, also sampling the
// real-time clock in between, and uses the average ROCProfiler clock
// value to approximate the actual clock class offset.
//
// This strategy is based on the measure_single_clock_offset() function
// of the LTTng-tools project <https://lttng.org/>.
void UpdateClkClsOffsetAndDelta(std::uint64_t& offset, std::uint64_t& delta) {
  // Sample ROCProfiler clock (first time).
  const auto rocm_clk_val1 = GetClkVal();

  // Sample real-time clock.
  timespec realtime_spec = {0, 0};
  [[maybe_unused]] const auto ret = clock_gettime(CLOCK_REALTIME, &realtime_spec);

  assert(ret == 0);

  // Sample ROCProfiler clock (second time).
  const auto rocm_clk_val2 = GetClkVal();

  // Compute the current ROCProfiler clock value delta.
  const auto this_delta = rocm_clk_val2 - rocm_clk_val1;

  if (this_delta > delta) {
    // Discard larger delta.
    return;
  }

  // Compute the average ROCProfiler clock value.
  const auto rocm_clk_val_avg = (rocm_clk_val1 + rocm_clk_val2) >> 1;

  // Compute the real-time clock value in nanoseconds.
  const auto realtime_ns =
      (static_cast<std::uint64_t>(realtime_spec.tv_sec) * ns_per_s) + realtime_spec.tv_nsec;

  // Update clock class offset and delta.
  assert(rocm_clk_val_avg < realtime_ns);
  offset = realtime_ns - rocm_clk_val_avg;
  delta = this_delta;
}

// Computes and returns the most possible accurate clock class offset.
std::uint64_t GetMetadataClkClsOffset() {
  std::uint64_t offset = 0;
  std::uint64_t delta = std::numeric_limits<std::uint64_t>::max();

  // Best effort to find the most accurate offset.
  for (auto i = 0U; i < 50U; ++i) {
    UpdateClkClsOffsetAndDelta(offset, delta);
  }

  return offset;
}

}  // namespace


static const char* LOOP_MPI_RANK(const std::vector<const char*>& mpivars) {
  for (const char* env : mpivars)
    if (const char* envvar = getenv(env)) return envvar;
  return nullptr;
}

static void insert_meta_to_stream(std::stringstream& stream, const char* field, const char* value) {
  if (!field || !value) return;
  stream << "\n\t" << std::string(field) << " = " << std::string(value) << ';';
}

void Plugin::CopyAdjustedMetadataStreamFile(const fs::path& metadata_stream_path,
                                            const fs::path& trace_dir) {
  // Load installed metadata stream file contents.
  std::string metadata;
  std::getline(std::ifstream{metadata_stream_path}, metadata, '\0');

  // Replace the original `offset` property.
  {
    static constexpr auto offset_term = "offset = 0;";
    std::ostringstream ss;

    ss << "offset = " << GetMetadataClkClsOffset() << ';';
    metadata.replace(metadata.find(offset_term), std::strlen(offset_term), ss.str());
  }

  std::stringstream data_stream;
  const char* rank = LOOP_MPI_RANK({"MPI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"});
  // Add MPI information to metadata
  if (rank) {
    insert_meta_to_stream(data_stream, "rank", rank);
    insert_meta_to_stream(data_stream, "node_rank", getenv("OMPI_COMM_WORLD_NODE_RANK"));

    const char* local = LOOP_MPI_RANK({"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"});
    insert_meta_to_stream(data_stream, "local_rank", local);

    std::string data_ins = data_stream.str();
    size_t env_pos = metadata.find("env {");
    if (env_pos != std::string::npos)
      metadata.insert(metadata.begin() + env_pos + 5, data_ins.begin(), data_ins.end());
    else
      std::cerr << "Failed to insert MPI metadata!" << std::endl;
  }

  // Write adjusted metadata stream to trace directory.
  {
    std::ofstream output{trace_dir / "metadata"};

    output.write(metadata.data(), metadata.size());
  }
}

}  // namespace rocm_ctf
