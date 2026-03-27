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
      command.type(),                           // activity kind
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
    case CL_COMMAND_NDRANGE_KERNEL:
      record.kernel_name =
          static_cast<const amd::NDRangeKernelCommand&>(command).kernel().name().c_str();
      break;
    case CL_COMMAND_READ_BUFFER:
    case CL_COMMAND_READ_BUFFER_RECT:
      record.bytes = linearSize(static_cast<const amd::ReadMemoryCommand&>(command).size());
      break;
    case CL_COMMAND_WRITE_BUFFER:
    case CL_COMMAND_WRITE_BUFFER_RECT:
      record.bytes = linearSize(static_cast<const amd::WriteMemoryCommand&>(command).size());
      break;
    case CL_COMMAND_COPY_BUFFER:
    case CL_COMMAND_COPY_BUFFER_RECT:
      record.bytes = linearSize(static_cast<const amd::CopyMemoryCommand&>(command).size());
      break;
    case CL_COMMAND_FILL_BUFFER:
      record.bytes = linearSize(static_cast<const amd::FillMemoryCommand&>(command).size());
      break;
    default:
      break;
  }

  if (command.type() == CL_COMMAND_TASK) {
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

const char* getOclCommandKindString(cl_command_type commandType) {
  switch (commandType) {
    CASE_STRING(0, InternalMarker);
    CASE_STRING(CL_COMMAND_MARKER, Marker);
    CASE_STRING(CL_COMMAND_NDRANGE_KERNEL, KernelExecution);
    CASE_STRING(CL_COMMAND_READ_BUFFER, CopyDeviceToHost);
    CASE_STRING(CL_COMMAND_WRITE_BUFFER, CopyHostToDevice);
    CASE_STRING(CL_COMMAND_COPY_BUFFER, CopyDeviceToDevice);
    CASE_STRING(CL_COMMAND_READ_BUFFER_RECT, CopyDeviceToHost2D);
    CASE_STRING(CL_COMMAND_WRITE_BUFFER_RECT, CopyHostToDevice2D);
    CASE_STRING(CL_COMMAND_COPY_BUFFER_RECT, CopyDeviceToDevice2D);
    CASE_STRING(CL_COMMAND_FILL_BUFFER, FillBuffer);
    CASE_STRING(CL_COMMAND_TASK, Task);
    CASE_STRING(CL_COMMAND_NATIVE_KERNEL, NativeKernel);
    CASE_STRING(CL_COMMAND_READ_IMAGE, ReadImage);
    CASE_STRING(CL_COMMAND_WRITE_IMAGE, WriteImage);
    CASE_STRING(CL_COMMAND_COPY_IMAGE, CopyImage);
    CASE_STRING(CL_COMMAND_COPY_IMAGE_TO_BUFFER, CopyImageToBuffer);
    CASE_STRING(CL_COMMAND_COPY_BUFFER_TO_IMAGE, CopyBufferToImage);
    CASE_STRING(CL_COMMAND_MAP_BUFFER, MapBuffer);
    CASE_STRING(CL_COMMAND_MAP_IMAGE, MapImage);
    CASE_STRING(CL_COMMAND_UNMAP_MEM_OBJECT, UnmapMemObject);
    CASE_STRING(CL_COMMAND_ACQUIRE_GL_OBJECTS, AcquireGLObjects);
    CASE_STRING(CL_COMMAND_RELEASE_GL_OBJECTS, ReleaseGLObjects);
    CASE_STRING(CL_COMMAND_USER, User);
    CASE_STRING(CL_COMMAND_BARRIER, Barrier);
    CASE_STRING(CL_COMMAND_MIGRATE_MEM_OBJECTS, MigrateMemObjects);
    CASE_STRING(CL_COMMAND_FILL_IMAGE, FillImage);
    CASE_STRING(CL_COMMAND_SVM_FREE, SvmFree);
    CASE_STRING(CL_COMMAND_SVM_MEMCPY, SvmMemcpy);
    CASE_STRING(CL_COMMAND_SVM_MEMFILL, SvmMemFill);
    CASE_STRING(CL_COMMAND_SVM_MAP, SvmMap);
    CASE_STRING(CL_COMMAND_SVM_UNMAP, SvmUnmap);
    CASE_STRING(ROCCLR_COMMAND_STREAM_WAIT_VALUE, StreamWait);
    CASE_STRING(ROCCLR_COMMAND_STREAM_WRITE_VALUE, StreamWrite);
    default:
      break;
  };
  return "Unknown command kind";
};
}  // namespace amd::activity_prof