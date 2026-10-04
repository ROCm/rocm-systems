/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_GLOBAL_HPP
#define HIP_GLOBAL_HPP

#include <cstdint>
#include <vector>
#include <string>

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_fatbin.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/memory.hpp"

namespace hip {

// Forward Declaration
class CodeObject;

// Cast helpers replacing the old DeviceFunc/DeviceVar wrappers
inline amd::Kernel* asKernel(hipFunction_t f) { return reinterpret_cast<amd::Kernel*>(f); }
inline amd::Kernel* asKernel(hipKernel_t k)   { return reinterpret_cast<amd::Kernel*>(k); }
inline hipFunction_t asHipFunction(amd::Kernel* k) { return reinterpret_cast<hipFunction_t>(k); }
inline hipDeviceptr_t memDevPtr(const amd::Memory* m) {
  return reinterpret_cast<hipDeviceptr_t>(m->getSvmPtr());
}

// Abstract Structures
class Function {
 public:
  Function(const std::string& name, FatBinaryInfo** modules = nullptr);
  ~Function();

  hipError_t GetDynFunc(hipFunction_t* hfunc, hipModule_t hmod);
  hipError_t GetStatFunc(hipFunction_t* hfunc, int deviceId);
  hipError_t GetStatFuncAttr(hipFuncAttributes* func_attr, int deviceId);
  void ResizeDFunc(size_t size) { dFunc_.resize(size); }
  FatBinaryInfo** ModuleInfo() { return modules_; }
  const std::string& GetName() const { return name_; }

 private:
  amd::Kernel* BuildKernel(hipModule_t hmod) const;

  std::vector<amd::Kernel*> dFunc_;  //!< Per-device kernel objects; index matches g_devices
  std::string name_;                 //!< Symbol name for kernel lookup in the program
  FatBinaryInfo** modules_;          //!< Owning fat binary; nullptr for dynamic COs
};

class Var {
 public:
  // Types of variable
  enum DeviceVarKind { DVK_Variable = 0, DVK_Surface, DVK_Texture, DVK_Managed };

  Var(const std::string& name, DeviceVarKind dVarKind, size_t size, int type, int norm,
      FatBinaryInfo** modules = nullptr);

  Var(const std::string& name, DeviceVarKind dVarKind, void* pointer, size_t size, unsigned align,
      FatBinaryInfo** modules = nullptr);

  ~Var();

  hipError_t GetDeviceVar(amd::Memory** mem, int deviceId, hipModule_t hmod);
  hipError_t GetStatDeviceVar(amd::Memory** mem, int deviceId);
  hipError_t GetDeviceVarPtr(amd::Memory** mem, int deviceId);

  hipError_t EnsureManagedStorageMapped();

  void ResizeDVar(size_t size) {
    dMem_.resize(size);
    managedSlotWritten_.resize(size, 0);
  }

  //! Whether this managed variable's device-side pointer slot has been filled on
  //! deviceId. Guarded by StatCO::sclock_.
  bool IsManagedSlotWritten(int deviceId) const {
    return static_cast<size_t>(deviceId) < managedSlotWritten_.size() &&
        managedSlotWritten_[deviceId] != 0;
  }
  void SetManagedSlotWritten(int deviceId) {
    if (static_cast<size_t>(deviceId) < managedSlotWritten_.size()) {
      managedSlotWritten_[deviceId] = 1;
    }
  }

  FatBinaryInfo** ModuleInfo() { return modules_; }
  DeviceVarKind GetVarKind() const { return dVarKind_; }
  size_t GetSize() const { return size_; }
  size_t GetAlignment() const { return align_; }
  const std::string& GetName() const { return name_; }

  void* shadowVptr = nullptr;  //!< Host-side textureReference shadow; device-independent

  //! Two meanings by provenance: for `__hipRegisterManagedVar` this is the address of the
  //! application's host shadow pointer and callers dereference it to reach the storage; for
  //! a `DynCO` variable it is the storage address itself, since a dynamically loaded module
  //! has no shadow pointer to fill. `StatCO::managedVars_` and `DynCO::vars_` hold the two
  //! populations separately; a consumer that guesses wrong is off by one indirection.
  void* GetManagedVarPtr() const { return managedVarPtr_; }
  void SetManagedVarInfo(void* pointer, size_t size) {
    managedVarPtr_ = pointer;
    size_ = size;
    dVarKind_ = DVK_Managed;
  }
  bool IsSvmOwned() const { return svmOwned_; }
  void SetSvmOwned(bool val) { svmOwned_ = val; }

 private:
  std::vector<amd::Memory*> dMem_;  //!< Per-device memory objects; index matches g_devices
  //! Per-device "pointer slot written" state (DVK_Managed); index matches g_devices
  std::vector<uint8_t> managedSlotWritten_;
  std::string name_;                //!< Symbol name for code-object lookup (not a unique key)
  DeviceVarKind dVarKind_;          //!< Classification: regular, surface, texture, or managed
  size_t size_;                     //!< Size of the variable in bytes
  int type_;                        //!< Channel type (textures/surfaces only)
  int norm_;                        //!< Normalisation flag (textures/surfaces only)
  FatBinaryInfo** modules_;         //!< Owning fat binary; nullptr for dynamic COs
  void* managedVarPtr_;             //!< Managed storage, or its host shadow slot (DVK_Managed)
  size_t align_;                    //!< Alignment of the managed allocation in bytes
  bool svmOwned_;                   //!< Release path: ihipFree if set, else Os::releaseMemory
};

};  // namespace hip
#endif /* HIP_GLOBAL_HPP */
