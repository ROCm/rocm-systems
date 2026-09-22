/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "platform/object.hpp"
#include "platform/agent.hpp"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Contexts
 *  @{
 */

class GLFunctions;
class DeviceQueue;

class Context : public RuntimeObject {
  std::vector<Device*> devices_;

 public:
  enum DeviceFlagIdx {
    GLDeviceKhrIdx = 0,   //!< GL
    D3D10DeviceKhrIdx,    //!< D3D10
    OfflineDevicesIdx,    //!< Offline devices
    CommandInterceptIdx,  //!< (Deprecated) Command intercept
    D3D11DeviceKhrIdx,    //!< D3D11
    InteropUserSyncIdx,   //!< Interop user sync enabled
    D3D9DeviceKhrIdx,     //!< d3d9 device
    D3D9DeviceEXKhrIdx,   //!< d3d9EX device
    D3D9DeviceVAKhrIdx,   //!< d3d9VA device
    EGLDeviceKhrIdx,      //!< EGL device
    LastDeviceFlagIdx
  };

  enum Flags {
    GLDeviceKhr = 1 << GLDeviceKhrIdx,          //!< GL
    D3D10DeviceKhr = 1 << D3D10DeviceKhrIdx,    //!< D3D10
    OfflineDevices = 1 << OfflineDevicesIdx,    //!< Offline devices
    D3D11DeviceKhr = 1 << D3D11DeviceKhrIdx,    //!< D3D11
    InteropUserSync = 1 << InteropUserSyncIdx,  //!< Interop user sync enabled
    D3D9DeviceKhr = 1 << D3D9DeviceKhrIdx,      //!< d3d9 device
    D3D9DeviceEXKhr = 1 << D3D9DeviceEXKhrIdx,  //!< d3d9EX device
    D3D9DeviceVAKhr = 1 << D3D9DeviceVAKhrIdx,  //!< d3d9VA device
    EGLDeviceKhr = 1 << EGLDeviceKhrIdx,        //!< EGL device
  };

  //! Context info structure
  struct Info {
    uint flags_;                     //!< Context info flags
    void* hDev_[LastDeviceFlagIdx];  //!< Device object reference
    void* hCtx_;                     //!< Context object reference
    size_t propertiesSize_;          //!< Size of the original properties in bytes
  };

  struct DeviceQueueInfo {
    DeviceQueue* defDeviceQueue_;  //!< Default device queue
    uint deviceQueueCnt_;          //!< The number of device queues
    DeviceQueueInfo() : defDeviceQueue_(NULL), deviceQueueCnt_(0) {}
  };

 private:
  // Copying a Context is not allowed
  Context(const Context&);
  Context& operator=(const Context&);

 protected:
  bool terminate() {
    if (Agent::shouldPostContextEvents()) {
      Agent::postContextFree(as_cl(this));
    }
    return true;
  }
  //! Context destructor
  ~Context();

 public:
  /*! \brief Helper function to check the context properties and initialize
   *  context info structure
   *
   *  \return An errcode if invalid, CL_SUCCESS if valid
   */
  static int checkProperties(const cl_context_properties* properties,  //!< Properties
                             Info* info                                //!< Info structure
  );

  //! Default constructor
  Context(const std::vector<Device*>& devices,  //!< List of all devices
          const Info& info                      //!< Context info structure
  );

  //! Compare two Context instances.
  bool operator==(const Context& rhs) const { return this == &rhs; }
  bool operator!=(const Context& rhs) const { return !(*this == rhs); }

  /*! Creates the context
   *
   *  \return An errcode if runtime fails the context creation,
   *          CL_SUCCESS otherwise
   */
  int create(const intptr_t* properties  //!< Original context properties
  );

  /**
   * Allocate host memory using either a custom device allocator or a generic
   * OS allocator
   *
   * @param size Allocation size, in bytes
   * @param alignment Desired alignment, in bytes
   * @param atomics The buffer should support platform (SVM) atomics
   */
  void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const;

  /**
   * Release host memory
   * @param ptr Pointer allocated using ::hostAlloc. If the pointer has been
   * allocated elsewhere, the behavior is undefined
   */
  void hostFree(void* ptr) const;

  /**
   * Allocate SVM buffer
   *
   * @param size Allocation size, in bytes
   * @param alignment Desired alignment, in bytes
   * @param flags The flags to create a svm space
   * @param curDev The current device
   */
  void* svmAlloc(size_t size, size_t alignment, cl_svm_mem_flags flags = CL_MEM_READ_WRITE,
                 const amd::Device* curDev = nullptr, void* svmPtr = nullptr);

  /**
   * Release SVM buffer
   * @param ptr Pointer allocated using ::svmAlloc. If the pointer has been
   * allocated elsewhere, the behavior is undefined
   */
  void svmFree(void* ptr) const;

  //! Return the devices associated with this context.
  const std::vector<Device*>& devices() const { return devices_; }

  //! Return the SVM capable devices associated with this context.
  const std::vector<Device*>& svmDevices() const { return svmAllocDevice_; }

  //! Returns true if the given device is associated with this context.
  bool containsDevice(const Device* device) const;

  //! Returns the context info structure
  const Info& info() const { return info_; }

  void setInfo(Info info) {
    info_ = info;
    return;
  }

  //! Returns a pointer to the original properties
  const cl_context_properties* properties() const { return properties_; }

  //! Returns a pointer to the OpenGL context
  GLFunctions* glenv() const { return glenv_; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeContext; }

  //! Returns context lock for the serialized access to the context
  Monitor& lock() { return ctxLock_; }

  //! Returns TRUE if runtime succesfully added a device queue
  DeviceQueue* defDeviceQueue(const Device& dev) const;

  //! Returns TRUE if runtime succesfully added a device queue
  bool isDevQueuePossible(const Device& dev);

  //! Returns TRUE if runtime succesfully added a device queue
  void addDeviceQueue(const Device& dev,   //!< Device object
                      DeviceQueue* queue,  //!< Device queue
                      bool defDevQueue     //!< Added device queue will be the default queue
  );

  //! Removes a device queue from the list of queues
  void removeDeviceQueue(const Device& dev,  //!< Device object
                         DeviceQueue* queue  //!< Device queue
  );

  //! Set the default device queue
  void setDefDeviceQueue(const Device& dev, DeviceQueue* queue) {
    deviceQueues_[&dev].defDeviceQueue_ = queue;
  };

 public:
#ifdef _WIN32
  // D3D11 interop extension cache: ID3D11Device* (as void*) -> {pExt, pCLExt}
  // pCLExt is IAmdDxExtCLInterop*, pExt is IAmdDxExt* kept alive as long as pCLExt is used.
  struct D3D11ExtEntry {
    void* pExt   = nullptr;   //!< IAmdDxExt*
    void* pCLExt = nullptr;   //!< IAmdDxExtCLInterop*
  };
  void setD3D11Ext(void* d3d11Device, void* pExt, void* pCLExt) {
    std::lock_guard<std::mutex> lock(d3d11ExtCacheLock_);
    d3d11ExtCache_[d3d11Device] = {pExt, pCLExt};
  }
  void* getD3D11Ext(void* d3d11Device) const {
    std::lock_guard<std::mutex> lock(d3d11ExtCacheLock_);
    auto it = d3d11ExtCache_.find(d3d11Device);
    return (it != d3d11ExtCache_.end()) ? it->second.pCLExt : nullptr;
  }
  bool removeD3D11Ext(void* d3d11Device, void** ppExt, void** ppCLExt) {
    std::lock_guard<std::mutex> lock(d3d11ExtCacheLock_);
    auto it = d3d11ExtCache_.find(d3d11Device);
    if (it == d3d11ExtCache_.end()) return false;
    if (ppExt)   *ppExt   = it->second.pExt;
    if (ppCLExt) *ppCLExt = it->second.pCLExt;
    d3d11ExtCache_.erase(it);
    return true;
  }
  void setD3D10Ext(void* d3d10Device, void* pExt, void* pCLExt) {
    std::lock_guard<std::mutex> lock(d3d10ExtCacheLock_);
    d3d10ExtCache_[d3d10Device] = {pExt, pCLExt};
  }
  void* getD3D10Ext(void* d3d10Device) const {
    std::lock_guard<std::mutex> lock(d3d10ExtCacheLock_);
    auto it = d3d10ExtCache_.find(d3d10Device);
    return (it != d3d10ExtCache_.end()) ? it->second.pCLExt : nullptr;
  }
  bool removeD3D10Ext(void* d3d10Device, void** ppExt, void** ppCLExt) {
    std::lock_guard<std::mutex> lock(d3d10ExtCacheLock_);
    auto it = d3d10ExtCache_.find(d3d10Device);
    if (it == d3d10ExtCache_.end()) return false;
    if (ppExt)   *ppExt   = it->second.pExt;
    if (ppCLExt) *ppCLExt = it->second.pCLExt;
    d3d10ExtCache_.erase(it);
    return true;
  }
#endif  // _WIN32

 private:
  Info info_;                            //!< Context info structure
  cl_context_properties* properties_;    //!< Original properties
  GLFunctions* glenv_;                   //!< OpenGL context
  Device* customHostAllocDevice_;        //!< Device responsible for host allocations
  std::vector<Device*> svmAllocDevice_;  //!< Devices can support SVM allocations
  std::unordered_map<const Device*, DeviceQueueInfo> deviceQueues_;  //!< Device queues mapping
  mutable Monitor ctxLock_;  //!< Lock for the context access
#ifdef _WIN32
  std::unordered_map<void*, D3D11ExtEntry> d3d11ExtCache_;
  mutable std::mutex d3d11ExtCacheLock_;
  std::unordered_map<void*, D3D11ExtEntry> d3d10ExtCache_;
  mutable std::mutex d3d10ExtCacheLock_;
#endif
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*CONTEXT_HPP_*/
