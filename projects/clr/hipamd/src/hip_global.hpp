/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
THE SOFTWARE.
*/

#ifndef HIP_GLOBAL_HPP
#define HIP_GLOBAL_HPP

#include <vector>
#include <string>

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_fatbin.hpp"
#include "platform/program.hpp"

namespace hip {

// Forward Declaration
class CodeObject;

// Device Structures
class DeviceVar {
 public:
  DeviceVar(std::string name, hipModule_t hmod, int deviceId);
  ~DeviceVar();

  // Accessors for device ptr and size, populated during constructor.
  hipDeviceptr_t device_ptr() const { return device_ptr_; }
  size_t size() const { return size_; }
  std::string name() const { return name_; }
  void* shadowVptr;

 private:
  std::string name_;           // Name of the var
  amd::Memory* amd_mem_obj_;   // amd_mem_obj abstraction
  hipDeviceptr_t device_ptr_;  // Device Pointer
  size_t size_;                // Size of the var
};

class DeviceFunc {
 public:
  DeviceFunc(std::string name, hipModule_t hmod);
  ~DeviceFunc();

  amd::Monitor dflock_;

  // Converts DeviceFunc to hipFunction_t(used by app) and vice versa.
  hipFunction_t asHipFunction() { return reinterpret_cast<hipFunction_t>(this); }
  static DeviceFunc* asFunction(hipFunction_t f) { return reinterpret_cast<DeviceFunc*>(f); }

  // Accessor for kernel_ and name_ populated during constructor.
  std::string name() const { return name_; }
  amd::Kernel* kernel() const { return kernel_; }

 private:
  std::string name_;     // name of the func(not unique identifier)
  amd::Kernel* kernel_;  // Kernel ptr referencing to ROCclr Symbol
};

// Abstract Structures
class Function {
 public:
  Function(const std::string& name, FatBinaryInfo** modules = nullptr);
  ~Function();

  // Return DeviceFunc for this this dynamically loaded module
  hipError_t getDynFunc(hipFunction_t* hfunc, hipModule_t hmod);
  bool isValidDynFunc(const void* hfunc);
  // Return Device Func & attr . Generate/build if not already done so.
  hipError_t getStatFunc(hipFunction_t* hfunc, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, int deviceId);
  void resize_dFunc(size_t size) { dFunc_.resize(size); }
  FatBinaryInfo** moduleInfo() { return modules_; }
  const std::string& name() const { return name_; }

 private:
  std::vector<DeviceFunc*> dFunc_;  //!< DeviceFuncObj per Device
  std::string name_;                //!< name of the func(not unique identifier)
  FatBinaryInfo** modules_;         //!< static module where it is referenced
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

  // Return DeviceVar for this dynamically loaded module
  hipError_t getDeviceVar(DeviceVar** dvar, int deviceId, hipModule_t hmod);

  // Return DeviceVar for module Generate/build if not already done so.
  hipError_t getStatDeviceVar(DeviceVar** dvar, int deviceId);

  hipError_t getDeviceVarPtr(DeviceVar** dvar, int deviceId);

  hipError_t allocateManagedVarPtr();

  void resize_dVar(size_t size) { dVar_.resize(size); }
  // bool isEmpty_dVar() const { return dVar_.empty(); }


  FatBinaryInfo** moduleInfo() { return modules_; };
  DeviceVarKind getVarKind() const { return dVarKind_; }
  size_t getSize() const { return size_; }
  size_t getAlignment() const { return align_; }
  std::string getName() const { return name_; }

  void* getManagedVarPtr() const { return managedVarPtr_; }
  void setManagedVarInfo(void* pointer, size_t size) {
    managedVarPtr_ = pointer;
    size_ = size;
    dVarKind_ = DVK_Managed;
  }
  bool getAllocFlag() const { return allocFlag_; }
  void setAllocFlag(bool val) { allocFlag_ = val; }

 private:
  std::vector<DeviceVar*> dVar_;  // DeviceVarObj per Device
  std::string name_;              // Variable name (not unique identifier)
  DeviceVarKind dVarKind_;        // Variable kind
  size_t size_;                   // Size of the variable
  int type_;                      // Type(Textures/Surfaces only)
  int norm_;                      // Type(Textures/Surfaces only)
  FatBinaryInfo** modules_;       // static module where it is referenced
  void* managedVarPtr_;           // Managed memory pointer with size_ & align_
  size_t align_;                  // Managed memory alignment
  bool allocFlag_;                // 0 : host alloc, 1: managed alloc
};

};  // namespace hip
#endif /* HIP_GLOBAL_HPP */
