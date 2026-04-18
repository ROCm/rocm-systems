/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/activity.hpp"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/command_utils.hpp"

#include <atomic>

namespace amd::activity_prof {

decltype(report_activity) report_activity{nullptr};

// Reserved sentinel pointer value (0x1) used to signal roctracer that CLR commits
// to delivering an activity record for this operation. roctracer's TracerCallback
// distinguishes this from an IsEnabled query (nullptr) and a real record (valid ptr).
// See TracerCallback in roctracer.cpp, ACTIVITY_DOMAIN_HIP_OPS case.
static void* const kCommitRecordSentinel = reinterpret_cast<void*>(uintptr_t{1});

void CommitRecord(OpId operation_id) {
  auto function = report_activity.load(std::memory_order_acquire);
  if (function) function(ACTIVITY_DOMAIN_HIP_OPS, operation_id, kCommitRecordSentinel);
}

#if defined(__linux__)
__thread activity_correlation_id_t correlation_id __attribute__((tls_model("initial-exec"))) = 0;
#elif defined(_WIN32)
__declspec(thread) activity_correlation_id_t correlation_id = 0;
#endif  // defined(_WIN32)

static inline size_t linearSize(const amd::Coord3D& size3d) {
  size_t size = size3d[0];
  if (size3d[1] != 0) size *= size3d[1];
  if (size3d[2] != 0) size *= size3d[2];
  return size;
}

bool IsEnabled(OpId operation_id) {
  if (operation_id < OP_ID_NUMBER)
    if (auto report = report_activity.load(std::memory_order_acquire))
      return report(ACTIVITY_DOMAIN_HIP_OPS, operation_id, nullptr) == 0;
  return false;
}

void ReportActivity(const amd::Command& command) {
  assert(command.profilingInfo().enabled_ && "Profiling must be enabled for this command");
  activity_op_t operation_id = OperationId(command.type());
  if (operation_id >= OP_ID_NUMBER) {
    // This command does not translate into a profiler activity (dispatch, memcopy, etc...), there
    // is nothing to report to the profiler.
    return;
  }

  auto function = report_activity.load(std::memory_order_acquire);
  if (!function) return;

  const auto* queue = command.queue();
  assert(queue != nullptr);
  activity_record_t record{
      ACTIVITY_DOMAIN_HIP_OPS,                  // activity domain
      static_cast<activity_kind_t>(command.type()),  // activity kind
      operation_id,                             // operation id
      command.profilingInfo().correlation_id_,  // activity correlation id
      command.profilingInfo().start_,           // begin timestamp, ns
      command.profilingInfo().end_,             // end timestamp, ns
      {{
          static_cast<int>(queue->device().info().driverNodeId_),  // device id
          queue->vdev()->index()                                   // queue id
      }},
      {}  // copied data size for memcpy, or kernel name for dispatch
  };

  switch (command.type()) {
    case amd::CommandType::NdRangeKernel:
      record.kernel_name =
          static_cast<const amd::NDRangeKernelCommand&>(command).kernel().name().c_str();
      break;
    case amd::CommandType::ReadBuffer:
    case amd::CommandType::ReadBufferRect:
      record.bytes = linearSize(static_cast<const amd::ReadMemoryCommand&>(command).size());
      break;
    case amd::CommandType::WriteBuffer:
    case amd::CommandType::WriteBufferRect:
      record.bytes = linearSize(static_cast<const amd::WriteMemoryCommand&>(command).size());
      break;
    case amd::CommandType::CopyBuffer:
    case amd::CommandType::CopyBufferRect:
      record.bytes = linearSize(static_cast<const amd::CopyMemoryCommand&>(command).size());
      break;
    case amd::CommandType::FillBuffer:
      record.bytes = linearSize(static_cast<const amd::FillMemoryCommand&>(command).size());
      break;
    default:
      break;
  }

  if (command.type() == amd::CommandType::Task) {
    auto timestamps = static_cast<const amd::AccumulateCommand&>(command).getTimestamps();
    const auto& kernel_names =
        static_cast<const amd::AccumulateCommand&>(command).getKernelNames();
    for (uint32_t i = 0; i < timestamps.size() && i < kernel_names.size(); i++) {
      auto it = timestamps[i];
      record.begin_ns = it.first;
      record.end_ns = it.second;
      record.kernel_name = kernel_names[i] != nullptr ? kernel_names[i]->c_str() : "";
      function(ACTIVITY_DOMAIN_HIP_OPS, operation_id, &record);
    }
  } else {
    record.begin_ns = command.profilingInfo().start_;
    record.end_ns = command.profilingInfo().end_;
    function(ACTIVITY_DOMAIN_HIP_OPS, operation_id, &record);
  }
}


#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    return #C

const char* getOclCommandKindString(amd::CommandType commandType) {
  switch (commandType) {
    CASE_STRING(static_cast<amd::CommandType>(0), InternalMarker);
    CASE_STRING(amd::CommandType::Marker, Marker);
    CASE_STRING(amd::CommandType::NdRangeKernel, KernelExecution);
    CASE_STRING(amd::CommandType::ReadBuffer, CopyDeviceToHost);
    CASE_STRING(amd::CommandType::WriteBuffer, CopyHostToDevice);
    CASE_STRING(amd::CommandType::CopyBuffer, CopyDeviceToDevice);
    CASE_STRING(amd::CommandType::ReadBufferRect, CopyDeviceToHost2D);
    CASE_STRING(amd::CommandType::WriteBufferRect, CopyHostToDevice2D);
    CASE_STRING(amd::CommandType::CopyBufferRect, CopyDeviceToDevice2D);
    CASE_STRING(amd::CommandType::FillBuffer, FillBuffer);
    CASE_STRING(amd::CommandType::Task, Task);
    CASE_STRING(amd::CommandType::NativeKernel, NativeKernel);
    CASE_STRING(amd::CommandType::ReadImage, ReadImage);
    CASE_STRING(amd::CommandType::WriteImage, WriteImage);
    CASE_STRING(amd::CommandType::CopyImage, CopyImage);
    CASE_STRING(amd::CommandType::CopyImageToBuffer, CopyImageToBuffer);
    CASE_STRING(amd::CommandType::CopyBufferToImage, CopyBufferToImage);
    CASE_STRING(amd::CommandType::MapBuffer, MapBuffer);
    CASE_STRING(amd::CommandType::MapImage, MapImage);
    CASE_STRING(amd::CommandType::UnmapMemObject, UnmapMemObject);
    CASE_STRING(amd::CommandType::AcquireGlObjects, AcquireGLObjects);
    CASE_STRING(amd::CommandType::ReleaseGlObjects, ReleaseGLObjects);
    CASE_STRING(amd::CommandType::User, User);
    CASE_STRING(amd::CommandType::Barrier, Barrier);
    CASE_STRING(amd::CommandType::MigrateMemObjects, MigrateMemObjects);
    CASE_STRING(amd::CommandType::FillImage, FillImage);
    CASE_STRING(amd::CommandType::SvmFree, SvmFree);
    CASE_STRING(amd::CommandType::SvmMemcpy, SvmMemcpy);
    CASE_STRING(amd::CommandType::SvmMemfill, SvmMemFill);
    CASE_STRING(amd::CommandType::SvmMap, SvmMap);
    CASE_STRING(amd::CommandType::SvmUnmap, SvmUnmap);
    CASE_STRING(static_cast<amd::CommandType>(ROCCLR_COMMAND_STREAM_WAIT_VALUE), StreamWait);
    CASE_STRING(static_cast<amd::CommandType>(ROCCLR_COMMAND_STREAM_WRITE_VALUE), StreamWrite);
    default:
      break;
  };
  return "Unknown command kind";
};
}  // namespace amd::activity_prof