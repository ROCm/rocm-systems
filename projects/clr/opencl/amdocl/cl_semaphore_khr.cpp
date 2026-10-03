/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

//
// cl_khr_semaphore / cl_khr_external_semaphore
//
// A semaphore is either imported from an external handle, or created device-owned so that
// clGetSemaphoreHandleForTypeKHR can create an OS handle from it. Export is only offered for
// handle types the device lists in CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR.
//
// An invalid cl_semaphore_khr is reported as CL_INVALID_VALUE rather than
// CL_INVALID_SEMAPHORE_KHR. CL_INVALID_SEMAPHORE_KHR was added by a later revision of
// the extension; the conformance tests we gate on still require CL_INVALID_VALUE
// (test_vulkan_api_consistency.cpp, consistency_external_semaphore).
//

#include "cl_common.hpp"

#include "platform/command.hpp"
#include "platform/context.hpp"
#include "platform/semaphore.hpp"

#include <algorithm>
#include <vector>

namespace {

//! Maps a cl_external_semaphore_handle_type_khr token onto the rocclr enum.
//! Returns false for handle types this implementation cannot import.
bool MapSemaphoreHandleType(cl_semaphore_properties_khr token,
                            amd::ExternalSemaphoreHandleType* type) {
  switch (token) {
    case CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR:
      *type = amd::ExternalSemaphoreHandleType::OpaqueFd;
      return true;
    case CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR:
      *type = amd::ExternalSemaphoreHandleType::OpaqueWin32;
      return true;
    case CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR:
      *type = amd::ExternalSemaphoreHandleType::OpaqueWin32Kmt;
      return true;
    default:
      return false;
  }
}

//! Unpacks a property-list slot into an amd::Os::FileDesc. The property list is
//! cl_ulong-wide, while FileDesc is void* on Windows and int on POSIX, so the two
//! platforms need different casts.
inline amd::Os::FileDesc ToFileDesc(cl_semaphore_properties_khr value) {
#if defined(_WIN32)
  return reinterpret_cast<amd::Os::FileDesc>(static_cast<uintptr_t>(value));
#else
  return static_cast<amd::Os::FileDesc>(value);
#endif
}

//! Shared body of clEnqueueWaitSemaphoresKHR / clEnqueueSignalSemaphoresKHR.
cl_int EnqueueSemaphoreCommands(cl_command_queue command_queue, cl_uint num_sema_objects,
                                const cl_semaphore_khr* sema_objects,
                                const cl_semaphore_payload_khr* sema_payload_list,
                                cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                cl_event* event, cl_command_type cmd_type,
                                amd::ExternalSemaphoreCmd::ExternalSemaphoreCmdType sem_cmd_type) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (queue == nullptr) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (num_sema_objects == 0 || sema_objects == nullptr) {
    return CL_INVALID_VALUE;
  }

  // Resolve every semaphore up front so a bad handle fails before anything is enqueued.
  std::vector<void*> devSemaphores(num_sema_objects);
  for (cl_uint i = 0; i < num_sema_objects; ++i) {
    if (!is_valid(sema_objects[i])) {
      return CL_INVALID_VALUE;
    }
    amd::Semaphore* semaphore = as_amd(sema_objects[i]);
    if (&semaphore->context() != &hostQueue.context()) {
      return CL_INVALID_CONTEXT;
    }
    void* devSemaphore = semaphore->getDeviceSemaphore(hostQueue.device());
    if (devSemaphore == nullptr) {
      // The semaphore was not imported on the device this queue runs on.
      return CL_INVALID_VALUE;
    }
    devSemaphores[i] = devSemaphore;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list,
                                       event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }
  const amd::Command::EventWaitList emptyWaitList;

  // One command per semaphore. The caller's wait list gates the first command only;
  // the rest are ordered behind it by the in-order queue, and the last command's event
  // is the completion point reported back to the caller.
  amd::Command* lastCommand = nullptr;
  for (cl_uint i = 0; i < num_sema_objects; ++i) {
    const uint64_t fence = (sema_payload_list != nullptr) ? sema_payload_list[i] : 0;

    auto* command = new amd::ExternalSemaphoreCmd(
        hostQueue, (i == 0) ? eventWaitList : emptyWaitList, cmd_type, devSemaphores[i], fence,
        sem_cmd_type);
    if (command == nullptr) {
      if (lastCommand != nullptr) {
        lastCommand->release();
      }
      return CL_OUT_OF_HOST_MEMORY;
    }

    command->enqueue();

    if (sem_cmd_type == amd::ExternalSemaphoreCmd::COMMAND_SIGNAL_EXTSEMAPHORE) {
      as_amd(sema_objects[i])->setPayload(fence);
    }

    // Keep only the last command alive for the caller's event.
    if (lastCommand != nullptr) {
      lastCommand->release();
    }
    lastCommand = command;
  }

  *not_null(event) = as_cl(&lastCommand->event());
  if (event == nullptr) {
    lastCommand->release();
  }
  return CL_SUCCESS;
}

}  // namespace

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup cl_khr_semaphore
 *  @{
 */

/*! \brief Creates a semaphore object, importing an external semaphore handle.
 *
 *  \return the created cl_semaphore_khr, or NULL with \a errcode_ret set to
 *  - CL_INVALID_CONTEXT if \a context is not valid.
 *  - CL_INVALID_VALUE if \a sema_props is NULL, malformed, names a property twice,
 *    or omits CL_SEMAPHORE_TYPE_KHR.
 *  - CL_INVALID_DEVICE if the device handle list names a device outside \a context.
 *  - CL_INVALID_OPERATION if no external handle was supplied, or the import failed.
 */
RUNTIME_ENTRY_RET(cl_semaphore_khr, clCreateSemaphoreWithPropertiesKHR,
                  (cl_context context, const cl_semaphore_properties_khr* sema_props,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return nullptr;
  }
  if (sema_props == nullptr) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return nullptr;
  }
  amd::Context& amdContext = *as_amd(context);

  cl_semaphore_type_khr type = CL_SEMAPHORE_TYPE_BINARY_KHR;
  bool typeSeen = false;

  amd::ExternalSemaphoreHandleType handleType = amd::ExternalSemaphoreHandleType::OpaqueFd;
  amd::Os::FileDesc handle = amd::Os::FDescInit();
  bool handleSeen = false;

  std::vector<amd::Device*> devices;
  bool deviceListSeen = false;
  bool exportListSeen = false;
  std::vector<cl_external_semaphore_handle_type_khr> exportTypes;

  // Keep a verbatim copy for CL_SEMAPHORE_PROPERTIES_KHR, including the 0 terminator.
  std::vector<cl_semaphore_properties_khr> propertiesCopy;

  for (const cl_semaphore_properties_khr* p = sema_props; *p != 0;) {
    const cl_semaphore_properties_khr name = *p++;
    propertiesCopy.push_back(name);

    switch (name) {
      case CL_SEMAPHORE_TYPE_KHR: {
        if (typeSeen) {
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          return nullptr;
        }
        typeSeen = true;
        type = static_cast<cl_semaphore_type_khr>(*p);
        propertiesCopy.push_back(*p++);
        if (type != CL_SEMAPHORE_TYPE_BINARY_KHR) {
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          return nullptr;
        }
        break;
      }

      case CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR:
      case CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR:
      case CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR:
      case CL_SEMAPHORE_HANDLE_SYNC_FD_KHR:
      case CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR: {
        if (handleSeen) {
          // At most one external handle may be specified.
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          return nullptr;
        }
        handleSeen = true;
        if (!MapSemaphoreHandleType(name, &handleType)) {
          // A legal token this implementation cannot import (sync_fd, D3D12 fence).
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          return nullptr;
        }
        handle = ToFileDesc(*p);
        propertiesCopy.push_back(*p++);
        break;
      }

      case CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR: {
        if (deviceListSeen) {
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          return nullptr;
        }
        deviceListSeen = true;
        while (*p != CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR) {
          cl_device_id device = reinterpret_cast<cl_device_id>(static_cast<uintptr_t>(*p));
          if (!is_valid(device) || !amdContext.containsDevice(as_amd(device))) {
            *not_null(errcode_ret) = CL_INVALID_DEVICE;
            return nullptr;
          }
          devices.push_back(as_amd(device));
          propertiesCopy.push_back(*p++);
        }
        propertiesCopy.push_back(*p++);  // the LIST_END sentinel
        break;
      }

      case CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR: {
        if (exportListSeen) {
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          return nullptr;
        }
        exportListSeen = true;
        while (*p != CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR) {
          const auto requested = static_cast<cl_external_semaphore_handle_type_khr>(*p);
          amd::ExternalSemaphoreHandleType unused;
          if (!MapSemaphoreHandleType(requested, &unused)) {
            // A legal token this implementation cannot export.
            *not_null(errcode_ret) = CL_INVALID_VALUE;
            return nullptr;
          }
          exportTypes.push_back(requested);
          propertiesCopy.push_back(*p++);
        }
        propertiesCopy.push_back(*p++);  // the LIST_END sentinel
        break;
      }

      default:
        *not_null(errcode_ret) = CL_INVALID_VALUE;
        return nullptr;
    }
  }
  propertiesCopy.push_back(0);

  if (!typeSeen) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return nullptr;
  }
  if (!handleSeen && exportTypes.empty()) {
    // Neither imported nor exportable: nothing can back this semaphore.
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return nullptr;
  }
  if (handleSeen && !exportTypes.empty()) {
    // Importing an existing handle and creating new ones are different objects; asking for
    // both in one create has no defined meaning here.
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return nullptr;
  }

  if (devices.empty()) {
    devices = amdContext.devices();
  }

  if (!exportTypes.empty()) {
    // create() makes one independent semaphore per device, but a handle can only be exported
    // from one of them - a signal on device A would be invisible to a handle exported for
    // device B. Refuse rather than hand back silently unsynchronised objects.
    if (devices.size() != 1) {
      *not_null(errcode_ret) = CL_INVALID_DEVICE;
      return nullptr;
    }
    // Every requested type must be one the device can actually export.
    const auto& supported = devices[0]->info().semaphoreExportHandleTypes_;
    for (const auto& requested : exportTypes) {
      if (std::find(supported.begin(), supported.end(), requested) == supported.end()) {
        *not_null(errcode_ret) = CL_INVALID_DEVICE;
        return nullptr;
      }
    }
    // The device object's flavour is fixed now; the first requested type decides it.
    MapSemaphoreHandleType(exportTypes[0], &handleType);
  }

  auto* semaphore = new amd::Semaphore(amdContext, type, propertiesCopy, devices, handleType,
                                       handle, handleSeen, exportTypes);
  if (semaphore == nullptr) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return nullptr;
  }
  if (!semaphore->create()) {
    semaphore->release();
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return nullptr;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(semaphore);
}
RUNTIME_EXIT

/*! \brief Increments the cl_semaphore_khr reference count. */
RUNTIME_ENTRY(cl_int, clRetainSemaphoreKHR, (cl_semaphore_khr sema_object)) {
  if (!is_valid(sema_object)) {
    return CL_INVALID_VALUE;
  }
  as_amd(sema_object)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrements the cl_semaphore_khr reference count. */
RUNTIME_ENTRY(cl_int, clReleaseSemaphoreKHR, (cl_semaphore_khr sema_object)) {
  if (!is_valid(sema_object)) {
    return CL_INVALID_VALUE;
  }
  as_amd(sema_object)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Queries information about a cl_semaphore_khr. */
RUNTIME_ENTRY(cl_int, clGetSemaphoreInfoKHR,
              (cl_semaphore_khr sema_object, cl_semaphore_info_khr param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(sema_object)) {
    return CL_INVALID_VALUE;
  }
  amd::Semaphore* semaphore = as_amd(sema_object);

  switch (param_name) {
    case CL_SEMAPHORE_CONTEXT_KHR: {
      cl_context context = as_cl(&semaphore->context());
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SEMAPHORE_REFERENCE_COUNT_KHR: {
      cl_uint count = semaphore->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SEMAPHORE_TYPE_KHR: {
      cl_semaphore_type_khr type = semaphore->type();
      return amd::clGetInfo(type, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SEMAPHORE_PAYLOAD_KHR: {
      cl_semaphore_payload_khr payload = semaphore->payload();
      return amd::clGetInfo(payload, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SEMAPHORE_EXPORTABLE_KHR: {
      cl_bool exportable = semaphore->exportable() ? CL_TRUE : CL_FALSE;
      return amd::clGetInfo(exportable, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SEMAPHORE_PROPERTIES_KHR: {
      const auto& properties = semaphore->properties();
      const size_t valueSize = properties.size() * sizeof(cl_semaphore_properties_khr);
      *not_null(param_value_size_ret) = valueSize;
      if (param_value != nullptr) {
        if (param_value_size < valueSize) {
          return CL_INVALID_VALUE;
        }
        ::memcpy(param_value, properties.data(), valueSize);
      }
      return CL_SUCCESS;
    }
    case CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR: {
      const auto& devices = semaphore->devices();
      const size_t valueSize = devices.size() * sizeof(cl_device_id);
      *not_null(param_value_size_ret) = valueSize;
      if (param_value != nullptr) {
        if (param_value_size < valueSize) {
          return CL_INVALID_VALUE;
        }
        auto* out = static_cast<cl_device_id*>(param_value);
        for (const auto& device : devices) {
          *out++ = as_cl(device);
        }
      }
      return CL_SUCCESS;
    }
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief Exports a semaphore as an external handle.
 *
 *  Only semaphores created with a CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR list can be exported,
 *  and only as one of the types in that list. The returned handle belongs to the caller: the
 *  runtime never closes it, because the importer (Vulkan) does not take ownership and may
 *  outlive the cl_semaphore.
 */
RUNTIME_ENTRY(cl_int, clGetSemaphoreHandleForTypeKHR,
              (cl_semaphore_khr sema_object, cl_device_id device,
               cl_external_semaphore_handle_type_khr handle_type, size_t handle_size,
               void* handle_ptr, size_t* handle_size_ret)) {
  if (!is_valid(sema_object)) {
    return CL_INVALID_VALUE;
  }
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }
  amd::Semaphore* semaphore = as_amd(sema_object);

  // The device must be one this semaphore was created on.
  const auto& devices = semaphore->devices();
  if (std::find(devices.begin(), devices.end(), as_amd(device)) == devices.end()) {
    return CL_INVALID_DEVICE;
  }

  const auto& exportTypes = semaphore->exportHandleTypes();
  if (std::find(exportTypes.begin(), exportTypes.end(), handle_type) == exportTypes.end()) {
    return CL_INVALID_VALUE;
  }

  *not_null(handle_size_ret) = sizeof(amd::Os::FileDesc);
  if (handle_ptr == nullptr) {
    // Size query only.
    return CL_SUCCESS;
  }
  if (handle_size < sizeof(amd::Os::FileDesc)) {
    return CL_INVALID_VALUE;
  }

  amd::Os::FileDesc handle = amd::Os::FDescInit();
  if (!semaphore->getHandleForType(*as_amd(device), handle_type, &handle)) {
    return CL_INVALID_OPERATION;
  }
  *reinterpret_cast<amd::Os::FileDesc*>(handle_ptr) = handle;
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueues a wait on one or more semaphores. */
RUNTIME_ENTRY(cl_int, clEnqueueWaitSemaphoresKHR,
              (cl_command_queue command_queue, cl_uint num_sema_objects,
               const cl_semaphore_khr* sema_objects,
               const cl_semaphore_payload_khr* sema_payload_list, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  return EnqueueSemaphoreCommands(command_queue, num_sema_objects, sema_objects, sema_payload_list,
                                  num_events_in_wait_list, event_wait_list, event,
                                  CL_COMMAND_SEMAPHORE_WAIT_KHR,
                                  amd::ExternalSemaphoreCmd::COMMAND_WAIT_EXTSEMAPHORE);
}
RUNTIME_EXIT

/*! \brief Enqueues a signal of one or more semaphores. */
RUNTIME_ENTRY(cl_int, clEnqueueSignalSemaphoresKHR,
              (cl_command_queue command_queue, cl_uint num_sema_objects,
               const cl_semaphore_khr* sema_objects,
               const cl_semaphore_payload_khr* sema_payload_list, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  return EnqueueSemaphoreCommands(command_queue, num_sema_objects, sema_objects, sema_payload_list,
                                  num_events_in_wait_list, event_wait_list, event,
                                  CL_COMMAND_SEMAPHORE_SIGNAL_KHR,
                                  amd::ExternalSemaphoreCmd::COMMAND_SIGNAL_EXTSEMAPHORE);
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
