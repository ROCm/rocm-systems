/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_

#include "top.hpp"
#include "platform/object.hpp"
#include "platform/context.hpp"
#include "device/device.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace amd {

//! Abstraction layer object backing cl_semaphore_khr.
//!
//! A semaphore is backed by a device object in one of two ways: it either imports an
//! external handle (importExtSemaphore) or is created device-owned so that an OS handle can
//! later be exported from it (createExtSemaphore + exportExtSemaphore). Both produce the same
//! opaque per-device object, so signalling, waiting and destruction are identical afterwards.
//! A semaphore that is neither imported nor exportable has no device counterpart and
//! create() fails.
class Semaphore : public RuntimeObject {
 public:
  typedef std::unordered_map<Device*, void*> DeviceSemaphores;

 private:
  Context& context_;          //!< OpenCL context associated with this semaphore
  cl_semaphore_type_khr type_;  //!< Semaphore type, always CL_SEMAPHORE_TYPE_BINARY_KHR today
  //! Verbatim copy of the property list, returned by CL_SEMAPHORE_PROPERTIES_KHR
  std::vector<cl_semaphore_properties_khr> properties_;
  std::vector<Device*> devices_;  //!< Devices this semaphore is valid on
  DeviceSemaphores deviceSemaphores_;  //!< Opaque per-device handle from importExtSemaphore

  ExternalSemaphoreHandleType handleType_;  //!< External handle flavour
  Os::FileDesc handle_;                     //!< The external handle being imported
  bool imported_;                           //!< False if no external handle was supplied
  //! Handle types this semaphore may be exported as, from
  //! CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR. Empty for an ordinary imported semaphore.
  std::vector<cl_external_semaphore_handle_type_khr> exportHandleTypes_;

  //! Last payload passed to clEnqueueSignalSemaphoresKHR, reported by
  //! CL_SEMAPHORE_PAYLOAD_KHR.
  cl_semaphore_payload_khr payload_;

 public:
  Semaphore(Context& context, cl_semaphore_type_khr type,
            const std::vector<cl_semaphore_properties_khr>& properties,
            const std::vector<Device*>& devices, ExternalSemaphoreHandleType handleType,
            Os::FileDesc handle, bool imported,
            const std::vector<cl_external_semaphore_handle_type_khr>& exportHandleTypes = {})
      : context_(context),
        type_(type),
        properties_(properties),
        devices_(devices),
        handleType_(handleType),
        handle_(handle),
        imported_(imported),
        exportHandleTypes_(exportHandleTypes),
        payload_(0) {}

  virtual ~Semaphore() {
    for (const auto& it : deviceSemaphores_) {
      it.first->DestroyExtSemaphore(it.second);
    }
  }

  //! Backs this semaphore with a device object on every device it covers: by importing the
  //! external handle, or - when an export list was requested - by creating a device-owned
  //! semaphore an OS handle can be exported from later.
  bool create() {
    if (!imported_ && exportHandleTypes_.empty()) {
      // Neither an import nor an export: nothing can back this object.
      return false;
    }
    for (const auto& dev : devices_) {
      void* devSemaphore = nullptr;
      if (imported_) {
        if (!dev->importExtSemaphore(&devSemaphore, handle_, handleType_)) {
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_RESOURCE,
                  "External semaphore import failed for device: %p \n", dev);
          return false;
        }
      } else {
        // The handle flavour is fixed at creation; exportExtSemaphore() later creates a handle
        // of whichever requested type the caller asks for.
        const bool timeline = (type_ != CL_SEMAPHORE_TYPE_BINARY_KHR);
        if (!dev->createExtSemaphore(&devSemaphore, handleType_, timeline)) {
          ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_RESOURCE,
                  "Exportable semaphore creation failed for device: %p \n", dev);
          return false;
        }
      }
      deviceSemaphores_[dev] = devSemaphore;
    }
    return true;
  }

  //! Creates a new OS handle of \p type for \p dev. The caller owns the handle.
  bool getHandleForType(Device& dev, cl_external_semaphore_handle_type_khr type,
                        Os::FileDesc* handle) const {
    if (handle == nullptr ||
        std::find(exportHandleTypes_.begin(), exportHandleTypes_.end(), type) ==
            exportHandleTypes_.end()) {
      return false;
    }
    void* devSemaphore = getDeviceSemaphore(dev);
    if (devSemaphore == nullptr) {
      return false;
    }
    return dev.exportExtSemaphore(handle, devSemaphore, handleType_);
  }

  //! Returns the opaque device handle for \p dev, or nullptr if this semaphore
  //! was not imported on it.
  void* getDeviceSemaphore(Device& dev) const {
    auto it = deviceSemaphores_.find(&dev);
    return (it != deviceSemaphores_.end()) ? it->second : nullptr;
  }

  //! Accessor functions
  Context& context() const { return context_; }
  cl_semaphore_type_khr type() const { return type_; }
  const std::vector<cl_semaphore_properties_khr>& properties() const { return properties_; }
  const std::vector<Device*>& devices() const { return devices_; }
  ExternalSemaphoreHandleType handleType() const { return handleType_; }
  bool imported() const { return imported_; }
  bool exportable() const { return !exportHandleTypes_.empty(); }
  const std::vector<cl_external_semaphore_handle_type_khr>& exportHandleTypes() const {
    return exportHandleTypes_;
  }

  cl_semaphore_payload_khr payload() const { return payload_; }
  void setPayload(cl_semaphore_payload_khr payload) { payload_ = payload; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeSemaphore; }
};

}  // namespace amd

#endif /*SEMAPHORE_HPP_*/
