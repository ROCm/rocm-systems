/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/activity.hpp"
#include "platform/clr_prof_event_bus.hpp"
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
  // Legacy path: signal roctracer via the old sentinel protocol.
  auto function = report_activity.load(std::memory_order_acquire);
  if (function) function(ACTIVITY_DOMAIN_HIP_OPS, operation_id, kCommitRecordSentinel);
  // New path: no commit-record concept; the EventBus delivers records directly
  // on GPU completion without a pre-commitment step.
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
  if (operation_id >= OP_ID_NUMBER) return false;
  // New path: check the EventBus subscriber count (O(1), no callback).
  if (amd::clr_prof::EventBus::instance().gpu_activity_enabled()) return true;
  // Legacy path: probe the roctracer callback with data=nullptr.
  if (auto report = report_activity.load(std::memory_order_acquire))
    return report(ACTIVITY_DOMAIN_HIP_OPS, operation_id, nullptr) == 0;
  return false;
}

// Build a clr_prof_gpu_record_t from a completed command and emit it to the
// EventBus.  Also forward to the legacy roctracer callback when registered.
static void EmitGpuRecord(const amd::Command& command, activity_op_t operation_id,
                           uint64_t begin_ns, uint64_t end_ns,
                           const char* kernel_name, size_t bytes,
                           int device_id, uint64_t queue_id) {
  auto& bus = amd::clr_prof::EventBus::instance();

  // ── New path: EventBus ───────────────────────────────────────────────────
  if (bus.gpu_activity_enabled()) {
    clr_prof_gpu_record_t rec{};
    rec.struct_size    = sizeof(clr_prof_gpu_record_t);
    rec.op             = static_cast<clr_prof_gpu_op_t>(operation_id);
    rec.correlation_id = command.profilingInfo().correlation_id_;
    rec.begin_ns       = begin_ns;
    rec.end_ns         = end_ns;
    rec.device_id      = device_id;
    rec.queue_id       = queue_id;
    if (operation_id == OP_ID_DISPATCH)
      rec.kernel_name = kernel_name;
    else
      rec.bytes = bytes;
    bus.emit_gpu(rec);
  }

  // ── Legacy path: roctracer callback ─────────────────────────────────────
  auto function = report_activity.load(std::memory_order_acquire);
  if (function) {
    activity_record_t legacy{};
    legacy.domain         = ACTIVITY_DOMAIN_HIP_OPS;
    legacy.kind           = command.type();
    legacy.op             = operation_id;
    legacy.correlation_id = command.profilingInfo().correlation_id_;
    legacy.begin_ns       = begin_ns;
    legacy.end_ns         = end_ns;
    legacy.device_id      = device_id;
    legacy.queue_id       = queue_id;
    if (operation_id == OP_ID_DISPATCH)
      legacy.kernel_name = kernel_name;
    else
      legacy.bytes = bytes;
    function(ACTIVITY_DOMAIN_HIP_OPS, operation_id, &legacy);
  }
}

void ReportActivity(const amd::Command& command) {
  assert(command.profilingInfo().enabled_ && "Profiling must be enabled for this command");
  activity_op_t operation_id = OperationId(command.type());
  if (operation_id >= OP_ID_NUMBER) {
    // This command does not translate into a profiler activity; nothing to report.
    return;
  }

  // Early-out when nothing is listening (fast path).
  bool has_new = amd::clr_prof::EventBus::instance().gpu_activity_enabled();
  bool has_legacy = report_activity.load(std::memory_order_acquire) != nullptr;
  if (!has_new && !has_legacy) return;

  const auto* queue = command.queue();
  assert(queue != nullptr);

  const int      device_id = static_cast<int>(queue->device().info().driverNodeId_);
  const uint64_t queue_id  = queue->vdev()->index();

  const char* kernel_name = nullptr;
  size_t      bytes       = 0;

  switch (command.type()) {
    case CL_COMMAND_NDRANGE_KERNEL:
      kernel_name =
          static_cast<const amd::NDRangeKernelCommand&>(command).kernel().name().c_str();
      break;
    case CL_COMMAND_READ_BUFFER:
    case CL_COMMAND_READ_BUFFER_RECT:
      bytes = linearSize(static_cast<const amd::ReadMemoryCommand&>(command).size());
      break;
    case CL_COMMAND_WRITE_BUFFER:
    case CL_COMMAND_WRITE_BUFFER_RECT:
      bytes = linearSize(static_cast<const amd::WriteMemoryCommand&>(command).size());
      break;
    case CL_COMMAND_COPY_BUFFER:
    case CL_COMMAND_COPY_BUFFER_RECT:
      bytes = linearSize(static_cast<const amd::CopyMemoryCommand&>(command).size());
      break;
    case CL_COMMAND_FILL_BUFFER:
      bytes = linearSize(static_cast<const amd::FillMemoryCommand&>(command).size());
      break;
    default:
      break;
  }

  if (command.type() == CL_COMMAND_TASK) {
    // Batched graph commands: each sub-kernel has its own timestamp pair.
    auto timestamps = static_cast<const amd::AccumulateCommand&>(command).getTimestamps();
    const auto& kernel_names =
        static_cast<const amd::AccumulateCommand&>(command).getKernelNames();
    for (uint32_t i = 0; i < timestamps.size() && i < kernel_names.size(); ++i) {
      const char* kn = kernel_names[i] != nullptr ? kernel_names[i]->c_str() : "";
      EmitGpuRecord(command, operation_id,
                    timestamps[i].first, timestamps[i].second,
                    kn, 0, device_id, queue_id);
    }
  } else {
    EmitGpuRecord(command, operation_id,
                  command.profilingInfo().start_, command.profilingInfo().end_,
                  kernel_name, bytes, device_id, queue_id);
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