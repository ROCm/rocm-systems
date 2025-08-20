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

#include "hip_global.hpp"

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_code_object.hpp"
#include "platform/program.hpp"
#include <hip/hip_version.h>

const char* amd_dbgapi_get_build_name(void) { return HIP_VERSION_BUILD_NAME; }

const char* amd_dbgapi_get_git_hash() { return HIP_VERSION_GITHASH; }

size_t amd_dbgapi_get_build_id() { return HIP_VERSION_BUILD_ID; }

#ifdef __HIP_ENABLE_PCH
extern const char __hip_pch_wave32[];
extern const char __hip_pch_wave64[];
extern unsigned __hip_pch_wave32_size;
extern unsigned __hip_pch_wave64_size;
void __hipGetPCH(const char** pch, unsigned int* size) {
  hipDeviceProp_t deviceProp;
  int deviceId;
  hipError_t error = hipGetDevice(&deviceId);
  error = hipGetDeviceProperties(&deviceProp, deviceId);
  if (deviceProp.warpSize == 32) {
    *pch = __hip_pch_wave32;
    *size = __hip_pch_wave32_size;
  } else {
    *pch = __hip_pch_wave64;
    *size = __hip_pch_wave64_size;
  }
}
#endif
namespace hip {

// forward declaration of methods required for managed variables
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align = 0, bool use_host_ptr = 0);

// Device Vars
DeviceVar::DeviceVar(std::string name, hipModule_t hmod, int deviceId)
    : shadowVptr(nullptr), name_(name), amd_mem_obj_(nullptr), device_ptr_(nullptr), size_(0) {
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));
  device::Program* dev_program = program->getDeviceProgram(*g_devices.at(deviceId)->devices()[0]);

  guarantee(dev_program != nullptr, "Cannot get Device Program for module: 0x%x", hmod);

  if (!dev_program->createGlobalVarObj(&amd_mem_obj_, &device_ptr_, &size_, name.c_str())) {
    guarantee(false, "Cannot create GlobalVar Obj for symbol: %s", name.c_str());
  }

  // Handle size 0 symbols
  if (size_ != 0) {
    if (amd_mem_obj_ == nullptr || device_ptr_ == nullptr) {
      LogPrintfError("Cannot get memory for creating device Var: %s", name.c_str());
      guarantee(false, "Cannot get memory for creating device var");
    }
    amd::MemObjMap::AddMemObj(device_ptr_, amd_mem_obj_);
  }
}

DeviceVar::~DeviceVar() {
  // device_ptr_ is being removed and its amd:Memory obj is being released/deleted during
  // ihipFree in hip::StatCO::removeFatBinary however in DynCO path, it seems to bypass
  // ihipFree and hence it needs to be removed+released here. In order to avoid issue with
  // StatCO, It is better to check if mem obj is found.
  if (amd::MemObjMap::FindMemObj(device_ptr_) != nullptr && amd_mem_obj_ != nullptr) {
    amd::MemObjMap::RemoveMemObj(device_ptr_);
    amd_mem_obj_->release();
  }
  if (shadowVptr != nullptr) {
    textureReference* texRef = reinterpret_cast<textureReference*>(shadowVptr);
    hipError_t err = ihipUnbindTexture(texRef);
    delete texRef;
    shadowVptr = nullptr;
  }

  device_ptr_ = nullptr;
  size_ = 0;
}

// Device Functions
DeviceFunc::DeviceFunc(std::string name, hipModule_t hmod)
    : dflock_("function lock"), name_(name), kernel_(nullptr) {
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  const amd::Symbol* symbol = program->findSymbol(name.c_str());
  guarantee(symbol != nullptr, "Cannot find Symbol with name: %s", name.c_str());

  kernel_ = new amd::Kernel(*program, *symbol, name);
  guarantee(kernel_ != nullptr, "Cannot Create kernel with name: %s", name.c_str());
}

DeviceFunc::~DeviceFunc() {
  if (kernel_ != nullptr) {
    kernel_->release();
  }
}

// Abstract functions
Function::Function(const std::string& name, FatBinaryInfo** modules)
    : name_(name), modules_(modules) {
  dFunc_.resize(g_devices.size());
}

Function::~Function() {
  for (auto& elem : dFunc_) {
    delete elem;
  }
  name_ = "";
  modules_ = nullptr;
}

hipError_t Function::getDynFunc(hipFunction_t* hfunc, hipModule_t hmod) {
  guarantee((dFunc_.size() == g_devices.size()), "dFunc Size mismatch");
  if (dFunc_[ihipGetDevice()] == nullptr) {
    dFunc_[ihipGetDevice()] = new DeviceFunc(name_, hmod);
  }
  *hfunc = dFunc_[ihipGetDevice()]->asHipFunction();

  return hipSuccess;
}

bool Function::isValidDynFunc(const void* hfunc) {
  return (hfunc == dFunc_[ihipGetDevice()]->asHipFunction());
}

hipError_t Function::getStatFunc(hipFunction_t* hfunc, int deviceId) {
  if (deviceId >= dFunc_.size()) {
    return hipErrorNoBinaryForGpu;
  }
  if (dFunc_[deviceId] != nullptr) {
    *hfunc = dFunc_[deviceId]->asHipFunction();
    return hipSuccess;
  }
  amd::ScopedLock lock((*modules_)->FatBinaryLock());
  // Check for the compiled kernel again, to make sure only one thread does compilation
  if (dFunc_[deviceId] != nullptr) {
    *hfunc = dFunc_[deviceId]->asHipFunction();
    return hipSuccess;
  }
  hipModule_t hmod = nullptr;
  IHIP_RETURN_ONFAIL((*modules_)->BuildProgram(deviceId));
  IHIP_RETURN_ONFAIL((*modules_)->GetModule(deviceId, &hmod));
  dFunc_[deviceId] = new DeviceFunc(name_, hmod);
  *hfunc = dFunc_[deviceId]->asHipFunction();
  return hipSuccess;
}

hipError_t Function::getStatFuncAttr(hipFuncAttributes* func_attr, int deviceId) {
  if (modules_ == nullptr || *modules_ == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }

  hipModule_t hmod = nullptr;
  IHIP_RETURN_ONFAIL((*modules_)->BuildProgram(deviceId));
  IHIP_RETURN_ONFAIL((*modules_)->GetModule(deviceId, &hmod));

  if (dFunc_[deviceId] == nullptr) {
    dFunc_[deviceId] = new DeviceFunc(name_, hmod);
  }

  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  amd::Kernel* kernel = dFunc_[deviceId]->kernel();
  auto* device_handle = devices[deviceId];
  const device::Kernel::WorkGroupInfo* wginfo =
      kernel->getDeviceKernel(*device_handle)->workGroupInfo();
  int binaryVersion =
      device_handle->isa().versionMajor() * 10 + device_handle->isa().versionMinor();
  func_attr->sharedSizeBytes = static_cast<int>(wginfo->localMemSize_);
  func_attr->binaryVersion = binaryVersion;
  func_attr->cacheModeCA = 0;
  func_attr->constSizeBytes = wginfo->constMemSize_ - 1;
  func_attr->localSizeBytes = wginfo->privateMemSize_;
  func_attr->maxDynamicSharedSizeBytes = wginfo->maxDynamicSharedSizeBytes_;
  func_attr->maxThreadsPerBlock = static_cast<int>(wginfo->size_);
  func_attr->numRegs = static_cast<int>(wginfo->usedVGPRs_);
  func_attr->preferredShmemCarveout = 0;
  func_attr->ptxVersion = binaryVersion;
  return hipSuccess;
}

// Abstract Vars
Var::Var(const std::string& name, DeviceVarKind dVarKind, size_t size, int type, int norm,
         FatBinaryInfo** modules)
    : name_(name),
      dVarKind_(dVarKind),
      size_(size),
      type_(type),
      norm_(norm),
      modules_(modules),
      managedVarPtr_(nullptr),
      align_(0) {
  dVar_.resize(g_devices.size());
}

Var::Var(const std::string& name, DeviceVarKind dVarKind, void* pointer, size_t size,
         unsigned align, FatBinaryInfo** modules)
    : name_(name),
      dVarKind_(dVarKind),
      size_(size),
      modules_(modules),
      managedVarPtr_(pointer),
      allocFlag_(false),
      align_(align),
      type_(0),
      norm_(0) {
  dVar_.resize(g_devices.size());
}

Var::~Var() {
  for (auto& elem : dVar_) {
    delete elem;
  }
  modules_ = nullptr;
}

hipError_t Var::getDeviceVarPtr(DeviceVar** dvar, int deviceId) {
  guarantee((deviceId >= 0), "Invalid DeviceId, less than zero");
  guarantee((static_cast<size_t>(deviceId) < g_devices.size()),
            "Invalid DeviceId, greater than no of code objects");
  guarantee((dVar_.size() == g_devices.size()), "Device Var not initialized to size");
  *dvar = dVar_[deviceId];
  return hipSuccess;
}

hipError_t Var::getDeviceVar(DeviceVar** dvar, int deviceId, hipModule_t hmod) {
  guarantee((deviceId >= 0), "Invalid DeviceId, less than zero");
  guarantee((static_cast<size_t>(deviceId) < g_devices.size()),
            "Invalid DeviceId, greater than no of code objects");
  guarantee((dVar_.size() == g_devices.size()), "Device Var not initialized to size");

  if (dVar_[deviceId] == nullptr) {
    dVar_[deviceId] = new DeviceVar(name_, hmod, deviceId);
  }

  *dvar = dVar_[deviceId];
  return hipSuccess;
}

hipError_t Var::getStatDeviceVar(DeviceVar** dvar, int deviceId) {
  guarantee((deviceId >= 0), "Invalid DeviceId, less than zero");
  guarantee((static_cast<size_t>(deviceId) < g_devices.size()),
            "Invalid DeviceId, greater than no of code objects");
  if (dVar_[deviceId] == nullptr) {
    hipModule_t hmod = nullptr;
    IHIP_RETURN_ONFAIL((*modules_)->BuildProgram(deviceId));
    IHIP_RETURN_ONFAIL((*modules_)->GetModule(deviceId, &hmod));
    dVar_[deviceId] = new DeviceVar(name_, hmod, deviceId);
  }
  *dvar = dVar_[deviceId];
  return hipSuccess;
}
// this method is added for allocation of managed var
hipError_t Var::allocateManagedVarPtr() {
  void** pointer = static_cast<void**>(managedVarPtr_);
  // check if it is deffered allocation
  if (!allocFlag_) {
    // Allocate managed memory for this var
    const bool use_host_ptr = true;
    IHIP_RETURN_ONFAIL(ihipMallocManaged(pointer, size_, align_, use_host_ptr));
    allocFlag_ = true;
  }
  if (dVar_.empty()) {
    resize_dVar(g_devices.size());
  }
  return hipSuccess;
}
};  // namespace hip
