/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include "hip_code_object.hpp"
#include "amd_hsa_elf.hpp"

#include <cstring>

#include <hip/driver_types.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include <elf/elf.hpp>
#include "comgrctx.hpp"
#include "hip_comgr_helper.hpp"
#include "hip_platform.hpp"

namespace hip {
hipError_t ihipFree(void* ptr);
// forward declaration of methods required for managed variables
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align = 0, bool use_host_ptr = 0);

hipError_t DynCO::loadCodeObject(const char* fname, const void* image) {
  amd::ScopedLock lock(dclock_);

  // Number of devices = 1 in dynamic code object
  fb_info_ = new FatBinaryInfo(fname, image);
  std::vector<hip::Device*> devices = {g_devices[ihipGetDevice()]};
  IHIP_RETURN_ONFAIL(fb_info_->ExtractFatBinaryUsingCOMGR(devices));

  // No Lazy loading for DynCO
  IHIP_RETURN_ONFAIL(fb_info_->BuildProgram(ihipGetDevice()));

  module_ = fb_info_->Module(device_id_);

  // Define Global variables
  IHIP_RETURN_ONFAIL(populateDynGlobalVars());

  // Define Global functions
  IHIP_RETURN_ONFAIL(populateDynGlobalFuncs());

  return hipSuccess;
}

// Dynamic Code Object
DynCO::~DynCO() {
  amd::ScopedLock lock(dclock_);

  for (auto& elem : vars_) {
    if (elem.second->getVarKind() == Var::DVK_Managed) {
      hipError_t err = ihipFree(elem.second->getManagedVarPtr());
      assert(err == hipSuccess);
    }

    if (elem.second->getVarKind() == Var::DVK_Variable) {
      for (auto dev : g_devices) {
        DeviceVar* dvar = nullptr;
        hipError_t err = elem.second->getDeviceVarPtr(&dvar, dev->deviceId());
        assert(err == hipSuccess);
        if (dvar != nullptr) {
          // free also deletes the device ptr
          err = ihipFree(dvar->device_ptr());
          assert(err == hipSuccess);
        }
      }
    }
    delete elem.second;
  }
  vars_.clear();

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  delete fb_info_;
}

hipError_t DynCO::getDeviceVar(DeviceVar** dvar, std::string var_name) {
  amd::ScopedLock lock(dclock_);

  auto it = vars_.find(var_name);
  if (it == vars_.end()) {
    LogPrintfError("Cannot find the Var: %s ", var_name.c_str());
    return hipErrorNotFound;
  }

  hipError_t err = it->second->getDeviceVar(dvar, device_id_, module_);
  return err;
}

hipError_t DynCO::getDynFunc(hipFunction_t* hfunc, std::string func_name) {
  amd::ScopedLock lock(dclock_);

  if (hfunc == nullptr) {
    return hipErrorInvalidValue;
  }

  auto it = functions_.find(func_name);
  if (it == functions_.end()) {
    LogPrintfError("Cannot find the function: %s ", func_name.c_str());
    return hipErrorNotFound;
  }

  /* See if this could be solved */
  return it->second->getDynFunc(hfunc, module_);
}

hipError_t DynCO::getFuncCount(unsigned int* count) {
  amd::ScopedLock lock(dclock_);
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = functions_.size();
  return hipSuccess;
}

bool DynCO::isValidDynFunc(const void* hfunc) {
  amd::ScopedLock lock(dclock_);
  return std::any_of(functions_.begin(), functions_.end(),
                     [&](auto& it) { return it.second->isValidDynFunc(hfunc); });
}

hipError_t DynCO::initDynManagedVars(const std::string& managedVar) {
  amd::ScopedLock lock(dclock_);
  DeviceVar* dvar;
  void* pointer = nullptr;
  hipError_t status = hipSuccess;
  // To get size of the managed variable
  status = getDeviceVar(&dvar, managedVar + ".managed");
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get .managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // Allocate managed memory for these symbols
  status = ihipMallocManaged(&pointer, dvar->size(), 0, 0);
  guarantee(status == hipSuccess, "Status %d, failed to allocate managed memory", status);

  // update as manager variable and set managed memory pointer and size
  auto it = vars_.find(managedVar);
  it->second->setManagedVarInfo(pointer, dvar->size());

  // copy initial value to the managed variable to the managed memory allocated
  hip::Stream* stream = hip::getNullStream();
  if (stream != nullptr) {
    status = ihipMemcpy(pointer, reinterpret_cast<address>(dvar->device_ptr()), dvar->size(),
                        hipMemcpyDeviceToDevice, *stream);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
              managedVar.c_str());
      return status;
    }
  } else {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
    return hipErrorInvalidResourceHandle;
  }

  // Get deivce ptr to initialize with managed memory pointer
  status = getDeviceVar(&dvar, managedVar);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // copy managed memory pointer to the managed device variable
  status = ihipMemcpy(reinterpret_cast<address>(dvar->device_ptr()), &pointer, dvar->size(),
                      hipMemcpyHostToDevice, *stream);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
            managedVar.c_str());
    return status;
  }
  return status;
}

hipError_t DynCO::populateDynGlobalVars() {
  amd::ScopedLock lock(dclock_);
  hipError_t err = hipSuccess;
  std::vector<std::string> var_names;
  std::string managedVarExt = ".managed";
  // For Dynamic Modules there is only one hipFatBinaryDevInfo_
  device::Program* dev_program = fb_info_->GetProgram(ihipGetDevice())
                                     ->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program->getGlobalVarFromCodeObj(&var_names)) {
    LogPrintfError("Could not get Global vars from Code Obj for Module: 0x%x", module_);
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : var_names) {
    vars_.insert(
        std::make_pair(elem, new Var(elem, Var::DeviceVarKind::DVK_Variable, 0, 0, 0, nullptr)));
  }

  for (auto& elem : var_names) {
    if (elem.find(managedVarExt) != std::string::npos) {
      std::string managedVar = elem;
      managedVar.erase(managedVar.length() - managedVarExt.length(), managedVarExt.length());
      err = initDynManagedVars(managedVar);
    }
  }
  return err;
}

hipError_t DynCO::populateDynGlobalFuncs() {
  amd::ScopedLock lock(dclock_);

  std::vector<std::string> func_names;
  device::Program* dev_program = fb_info_->GetProgram(ihipGetDevice())
                                     ->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  // Get all the global func names from COMGR
  if (!dev_program->getGlobalFuncFromCodeObj(&func_names)) {
    LogPrintfError("Could not get Global Funcs from Code Obj for Module: 0x%x", module_);
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : func_names) {
    functions_.insert(std::make_pair(elem, new Function(elem)));
  }

  return hipSuccess;
}

// Static Code Object
StatCO::StatCO() {}

StatCO::~StatCO() {
  amd::ScopedLock lock(sclock_);

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  for (auto& elem : vars_) {
    delete elem.second;
  }
  vars_.clear();
}

hipError_t StatCO::digestFatBinary(const void* data, FatBinaryInfo*& programs) {
  amd::ScopedLock lock(sclock_);

  if (programs != nullptr) {
    return hipSuccess;
  }

  // Fat binary wrapper structure (matches hip_platform.cpp definition)
  // Defined locally to keep kpack integration as implementation detail
  struct __CudaFatBinaryWrapper {
    unsigned int magic;
    unsigned int version;
    void* binary;
    void* dummy1;  // reserved1: bundle index for multi-TU binaries
  };

  // Check if this is a kpack'd binary (HIPK magic)
  const auto* wrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  if (wrapper->magic == symbols::kHipkMagic && wrapper->version == 1) {
    // Discover binary path from the wrapper address using existing CLR utility
    std::string binary_path;
    size_t file_offset = 0;
    if (!amd::Os::FindFileNameFromAddress(data, &binary_path, &file_offset)) {
      LogError("Failed to discover binary path for kpack loading");
      return hipErrorNoBinaryForGpu;
    }

    // Get bundle index from wrapper->dummy1 (reserved1 field)
    // For multi-TU binaries, this identifies which bundle this wrapper corresponds to
    uint64_t bundle_index = reinterpret_cast<uintptr_t>(wrapper->dummy1);

    // wrapper->binary points to msgpack metadata
    // ExtractKpackBinary will error if ROCM_KPACK_ENABLED=OFF
    FatBinaryInfo* fatBinaryInfo = new FatBinaryInfo(
        FatBinaryInfo::KpackParams{wrapper->binary, std::move(binary_path), bundle_index});
    hipError_t err = fatBinaryInfo->ExtractKpackBinary(g_devices);
    programs = fatBinaryInfo;
    return err;
  }

  // Create a new fat binary object and extract the fat binary for all devices.
  FatBinaryInfo* fatBinaryInfo = new FatBinaryInfo(nullptr, data);
  hipError_t err = fatBinaryInfo->ExtractFatBinaryUsingCOMGR(g_devices);
  programs = fatBinaryInfo;
  return err;
}

FatBinaryInfo** StatCO::addFatBinary(const void* data, bool initialized, bool& success) {
  amd::ScopedLock lock(sclock_);
  module_to_hostModule_.insert(std::make_pair(&modules_[data], data));

  if (initialized == false) {
    success = true;
    return &modules_[data];
  }

  hipError_t err = digestFatBinary(data, modules_[data]);

  success = (err == hipSuccess);
  return &modules_[data];
}

FatBinaryInfo** StatCO::addKpackBinary(const void* hipk_metadata, const void* wrapper_addr,
                                        bool initialized, bool& success) {
  amd::ScopedLock lock(sclock_);

  // Use wrapper_addr as the key (same as data pointer for normal path)
  // This allows digestFatBinary to access the wrapper and detect HIPK magic
  module_to_hostModule_.insert(std::make_pair(&modules_[wrapper_addr], wrapper_addr));

  if (!initialized) {
    // Deferred loading: modules_[wrapper_addr] is nullptr, digestFatBinary will handle it later
    success = true;
    return &modules_[wrapper_addr];
  }

  // Immediate loading: call digestFatBinary which handles kpack detection
  hipError_t err = digestFatBinary(wrapper_addr, modules_[wrapper_addr]);
  success = (err == hipSuccess);
  return &modules_[wrapper_addr];
}

hipError_t StatCO::removeFatBinary(FatBinaryInfo** module) {
  amd::ScopedLock lock(sclock_);

  auto hostVarsIter = module_to_hostVars_.find(module);
  if (hostVarsIter != module_to_hostVars_.end()) {
    for (auto& hostVar : hostVarsIter->second) {
      auto varIter = vars_.find(hostVar);
      if (varIter == vars_.end()) {
        LogPrintfError("removeFatBinary: Unable to find module 0x%x hostVar 0x%x", module, hostVar);
      } else {
        delete varIter->second;
        vars_.erase(varIter);
      }
    }
    module_to_hostVars_.erase(hostVarsIter);
  }

  auto managedVarsIter = managedVars_.find(module);
  if (managedVarsIter != managedVars_.end()) {
    for (auto& managedVar : managedVarsIter->second) {
      hipError_t err = hipSuccess;
      for (auto dev : g_devices) {
        DeviceVar* dvar = nullptr;
        IHIP_RETURN_ONFAIL(managedVar->getDeviceVarPtr(&dvar, dev->deviceId()));
        if (dvar != nullptr) {
          // free also deletes the device ptr
          err = ihipFree(dvar->device_ptr());
          assert(err == hipSuccess);
        }
      }
      if (managedVar->getAllocFlag()) {  // check if it is a managed or host alloc
        err = ihipFree(*(static_cast<void**>(managedVar->getManagedVarPtr())));
      } else {
        void** pointer = static_cast<void**>(managedVar->getManagedVarPtr());
        amd::Os::releaseMemory(*pointer, managedVar->getSize());
      }
      assert(err == hipSuccess);
      delete managedVar;
    }
    managedVars_.erase(managedVarsIter);
  }

  auto hostFuncsIter = module_to_hostFunctions_.find(module);
  if (hostFuncsIter != module_to_hostFunctions_.end()) {
    for (auto& hostFunc : hostFuncsIter->second) {
      auto funcIter = functions_.find(hostFunc);
      if (funcIter == functions_.end()) {
        LogPrintfError("removeFatBinary: Unable to find module 0x%x hostFunc 0x%x", module,
                       hostFunc);
      } else {
        delete funcIter->second;
        functions_.erase(funcIter);
      }
    }
    module_to_hostFunctions_.erase(hostFuncsIter);
  }

  auto hostModuleIter = module_to_hostModule_.find(module);
  if (hostModuleIter != module_to_hostModule_.end()) {
    auto hostModule = hostModuleIter->second;
    auto moduleIter = modules_.find(hostModule);
    if (moduleIter != modules_.end()) {
      delete moduleIter->second;
      modules_.erase(moduleIter);
    } else {
      LogPrintfError("removeFatBinary: Unable to find module 0x%x via hostModule 0x%x", module,
                     hostModule);
    }
    module_to_hostModule_.erase(hostModuleIter);
  }

  return hipSuccess;
}

// =================================================================================================
void StatCO::RemoveAllFatBinaries() {
  amd::ScopedLock lock(sclock_);

  // Clear mapping tables that associate modules with host-side constructs
  module_to_hostModule_.clear();
  module_to_hostFunctions_.clear();
  module_to_hostVars_.clear();

  // Delete all registered variables and clear the container
  for (auto const& [_, var] : vars_) {
    delete var;
  }
  vars_.clear();

  // Clean up managed variables - these require special handling for memory on each device
  for (auto& [_, managed_vars] : managedVars_) {
    for (auto& managed_var : managed_vars) {
      // Free device-specific allocations across all devices
      for (auto dev : g_devices) {
        DeviceVar* dvar = nullptr;
        if (managed_var->getDeviceVarPtr(&dvar, dev->deviceId()) == hipSuccess && dvar) {
          // Free device memory (also deletes the device ptr)
          [[maybe_unused]] hipError_t err = ihipFree(dvar->device_ptr());
          assert(err == hipSuccess);
        }
      }

      // Free the managed memory allocation itself
      void** managed_ptr = static_cast<void**>(managed_var->getManagedVarPtr());
      if (managed_var->getAllocFlag()) {
        // Memory was allocated with ihipMallocManaged - use ihipFree
        [[maybe_unused]] hipError_t err = ihipFree(*managed_ptr);
        assert(err == hipSuccess);
      } else {
        // Memory was allocated with OS-level allocator - use OS release
        amd::Os::releaseMemory(*managed_ptr, managed_var->getSize());
      }
      delete managed_var;
    }
  }
  managedVars_.clear();

  // Delete all registered functions and clear the container
  for (auto const& [_, func] : functions_) {
    delete func;
  }
  functions_.clear();

  // Delete all fat binary info objects and clear the modules container
  for (auto const& [_, fb_info] : modules_) {
    delete fb_info;
  }
  modules_.clear();
}

hipError_t StatCO::registerStatFunction(const void* hostFunction, Function* func) {
  amd::ScopedLock lock(sclock_);

  if (functions_.find(hostFunction) != functions_.end()) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
             "hostFunctionPtr: 0x%x already exists", hostFunction);
    delete func;
  } else {
    functions_.insert(std::make_pair(hostFunction, func));
    module_to_hostFunctions_[func->moduleInfo()].push_back(hostFunction);
  }

  return hipSuccess;
}

const char* StatCO::getStatFuncName(const void* hostFunction) {
  amd::ScopedLock lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return nullptr;
  }
  return it->second->name().c_str();
}

hipError_t StatCO::getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId) {
  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return hipErrorInvalidSymbol;
  }

  // Lazy load
  FatBinaryInfo** module = it->second->moduleInfo();
  if (module != nullptr) {
    amd::ScopedLock lock(sclock_);
    if (*(module) == nullptr) {
      hipError_t err = digestFatBinary(module_to_hostModule_[module], *module);

      if (err != hipSuccess) {
        return err;
      }
    }
  } else {
    // Module was nullptr
    return hipErrorInvalidDeviceFunction;
  }

  return it->second->getStatFunc(hfunc, deviceId);
}

hipError_t StatCO::getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction,
                                   int deviceId) {
  amd::ScopedLock lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return hipErrorInvalidSymbol;
  }

  // Lazy load
  FatBinaryInfo** module = it->second->moduleInfo();
  if (*(module) == nullptr) {
    std::ignore = digestFatBinary(module_to_hostModule_[module], *module);
  }

  return it->second->getStatFuncAttr(func_attr, deviceId);
}

hipError_t StatCO::registerStatGlobalVar(const void* hostVar, Var* var) {
  amd::ScopedLock lock(sclock_);

  auto var_it = vars_.find(hostVar);
  if ((var_it != vars_.end()) && (var_it->second->getName() != var->getName())) {
    return hipErrorInvalidSymbol;
  }

  vars_.insert(std::make_pair(hostVar, var));
  module_to_hostVars_[var->moduleInfo()].push_back(hostVar);
  return hipSuccess;
}

hipError_t StatCO::getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                                    size_t* size_ptr) {
  amd::ScopedLock lock(sclock_);

  const auto it = vars_.find(hostVar);
  if (it == vars_.end()) {
    return hipErrorInvalidSymbol;
  }

  // Lazy load
  FatBinaryInfo** module = it->second->moduleInfo();
  if (*(module) == nullptr) {
    std::ignore = digestFatBinary(module_to_hostModule_[module], *module);
  }

  DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getStatDeviceVar(&dvar, deviceId));

  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();
  return hipSuccess;
}

hipError_t StatCO::registerStatManagedVar(Var* var) {
  managedVars_[var->moduleInfo()].push_back(var);
  return hipSuccess;
}

hipError_t StatCO::initStatManagedVarDevicePtr(int deviceId) {
  amd::ScopedLock lock(sclock_);
  hipError_t err = hipSuccess;
  if (managedVarsDevicePtrInitalized_.find(deviceId) == managedVarsDevicePtrInitalized_.end() ||
      !managedVarsDevicePtrInitalized_[deviceId]) {
    for (auto& vecIter : managedVars_) {
      for (auto& var : vecIter.second) {
        // Lazy load
        FatBinaryInfo** module = var->moduleInfo();
        if (*(module) == nullptr) {
          std::ignore = digestFatBinary(module_to_hostModule_[module], *module);
        }
        hip::Stream* stream = g_devices.at(deviceId)->NullStream();
        if (stream == nullptr) {
          ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
          return hipErrorInvalidResourceHandle;
        }
        // Allocate managed var for deferred loading
        IHIP_RETURN_ONFAIL(var->allocateManagedVarPtr());
        // Copy from managed var host to device ptr
        DeviceVar* dvar = nullptr;
        IHIP_RETURN_ONFAIL(var->getStatDeviceVar(&dvar, deviceId));
        err = ihipMemcpy(reinterpret_cast<address>(dvar->device_ptr()), var->getManagedVarPtr(),
                         dvar->size(), hipMemcpyHostToDevice, *stream);
      }
    }
    managedVarsDevicePtrInitalized_[deviceId] = true;
  }
  return err;
}
}  // namespace hip
