/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
  std::scoped_lock lock(dclock_);

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
  std::scoped_lock lock(dclock_);

  for (auto& elem : vars_) {
    if (elem.second->GetVarKind() == Var::DVK_Managed) {
      hipError_t err = ihipFree(elem.second->GetManagedVarPtr());
      assert(err == hipSuccess);
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

hipError_t DynCO::GetDeviceVar(amd::Memory** mem, const std::string& var_name) {
  std::scoped_lock lock(dclock_);

  auto it = vars_.find(var_name);
  if (it == vars_.end()) {
    LogPrintfError("Cannot find the Var: %s ", var_name.c_str());
    return hipErrorNotFound;
  }

  return it->second->GetDeviceVar(mem, device_id_, module_);
}

hip::Var* DynCO::getVar(const std::string& var_name) {
  std::scoped_lock lock(dclock_);
  auto it = vars_.find(var_name);
  return (it != vars_.end()) ? it->second : nullptr;
}

hipError_t DynCO::GetGlobal(const std::string& name, void** dptr, size_t* bytes) {
  std::scoped_lock lock(dclock_);
  if (dptr != nullptr) {
    *dptr = nullptr;
  }
  IHIP_RETURN_ONFAIL(getManagedVarPointer(name, dptr, bytes));
  if ((dptr != nullptr && *dptr == nullptr) || (bytes != nullptr && *bytes == 0)) {
    amd::Memory* mem = nullptr;
    IHIP_RETURN_ONFAIL(GetDeviceVar(&mem, name));
    if (dptr != nullptr) {
      *dptr = memDevPtr(mem);
    }
    if (bytes != nullptr) {
      *bytes = mem->getSize();
    }
  }
  return hipSuccess;
}

hipError_t DynCO::GetManaged(const std::string& name, void** dptr, size_t* bytes) {
  std::scoped_lock lock(dclock_);
  auto it = vars_.find(name);
  if (it == vars_.end() || it->second->GetVarKind() != Var::DVK_Managed) {
    return hipErrorNotFound;
  }
  if (dptr != nullptr) {
    *dptr = it->second->GetManagedVarPtr();
  }
  if (bytes != nullptr) {
    *bytes = it->second->GetSize();
  }
  return hipSuccess;
}

std::vector<std::string> DynCO::getFunctionNames() {
  std::scoped_lock lock(dclock_);
  std::vector<std::string> names;
  names.reserve(functions_.size());
  for (const auto& kv : functions_) {
    names.push_back(kv.first);
  }
  return names;
}

hipError_t DynCO::getDynFunc(hipFunction_t* hfunc, const std::string& func_name) {
  std::scoped_lock lock(dclock_);

  if (hfunc == nullptr) {
    return hipErrorInvalidValue;
  }

  auto it = functions_.find(func_name);
  if (it == functions_.end()) {
    LogPrintfInfo("Cannot find the function: %s", func_name.c_str());
    return hipErrorNotFound;
  }

  /* See if this could be solved */
  return it->second->GetDynFunc(hfunc, module_);
}

hipError_t DynCO::getFuncCount(unsigned int* count) {
  std::scoped_lock lock(dclock_);
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = functions_.size();
  return hipSuccess;
}

bool DynCO::isValidDynFunc(const void* hfunc) {
  std::scoped_lock lock(dclock_);
  return std::any_of(functions_.begin(), functions_.end(),
                     [&](auto& it) { return it.second->IsValidDynFunc(hfunc); });
}

hipError_t DynCO::initDynManagedVars(const std::string& managedVar) {
  std::scoped_lock lock(dclock_);
  amd::Memory* mem = nullptr;
  void* pointer = nullptr;
  hipError_t status = hipSuccess;
  // To get size of the managed variable
  status = GetDeviceVar(&mem, managedVar + ".managed");
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get .managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // Allocate managed memory for these symbols
  status = ihipMallocManaged(&pointer, mem->getSize(), 0, 0);
  guarantee(status == hipSuccess, "Status %d, failed to allocate managed memory", status);

  // update as manager variable and set managed memory pointer and size
  auto it = vars_.find(managedVar);
  it->second->SetManagedVarInfo(pointer, mem->getSize());

  // copy initial value to the managed variable to the managed memory allocated
  hip::Stream* stream = hip::getNullStream();
  if (stream != nullptr) {
    status = ihipMemcpy(pointer, reinterpret_cast<address>(memDevPtr(mem)), mem->getSize(),
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

  // Get device ptr to initialize with managed memory pointer
  status = GetDeviceVar(&mem, managedVar);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // copy managed memory pointer to the managed device variable
  status = ihipMemcpy(reinterpret_cast<address>(memDevPtr(mem)), &pointer, mem->getSize(),
                      hipMemcpyHostToDevice, *stream);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
            managedVar.c_str());
    return status;
  }
  return status;
}

hipError_t DynCO::populateDynGlobalVars() {
  std::scoped_lock lock(dclock_);
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
  std::scoped_lock lock(dclock_);

  std::vector<std::string> func_names;
  device::Program* dev_program = fb_info_->GetProgram(ihipGetDevice())
                                     ->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  // Get all the global func names from COMGR
  if (!dev_program->getGlobalFuncFromCodeObj(&func_names)) {
    LogPrintfError("Could not get Global Funcs from Code Obj for Module: 0x%x", module_);
    return hipErrorSharedObjectSymbolNotFound;
  }

  // COMGR returns every AMD_COMGR_SYMBOL_TYPE_FUNC entry in the ELF, which
  // includes C++ ABI helpers like __cxa_deleted_virtual that have no HSA
  // kernel symbol. Filter to user-callable kernels — anything else would
  // SIGABRT inside Function::BuildKernel's findSymbol guarantee when later
  // resolved via hipLibraryEnumerateKernels / hipLibraryGetKernel.
  amd::Program* amd_program = as_amd(reinterpret_cast<cl_program>(module_));
  for (auto& elem : func_names) {
    if (amd_program == nullptr || amd_program->findSymbol(elem.c_str()) == nullptr) continue;
    functions_.insert(std::make_pair(elem, new Function(elem)));
  }

  return hipSuccess;
}

// Static Code Object
StatCO::StatCO(const PlatformState& owner) : owner_(owner) {}

StatCO::~StatCO() {
  std::scoped_lock lock(sclock_);

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  for (auto& elem : vars_) {
    delete elem.second;
  }
  vars_.clear();
}

hipError_t StatCO::DigestFatBinary(const void* data, FatBinaryInfo*& programs) {
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

FatBinaryInfo** StatCO::AddFatBinary(const void* data, bool& success) {
  std::scoped_lock lock(sclock_);
  module_to_hostModule_.insert(std::make_pair(&modules_[data], data));

  if (!owner_.IsInitialized()) {
    success = true;
    return &modules_[data];
  }

  hipError_t err = DigestFatBinary(data, modules_[data]);

  success = (err == hipSuccess);
  return &modules_[data];
}

FatBinaryInfo** StatCO::AddKpackBinary(const void* hipk_metadata, const void* wrapper_addr,
                                       bool& success) {
  std::scoped_lock lock(sclock_);

  // Use wrapper_addr as the key (same as data pointer for normal path)
  // This allows DigestFatBinary to access the wrapper and detect HIPK magic
  module_to_hostModule_.insert(std::make_pair(&modules_[wrapper_addr], wrapper_addr));

  if (!owner_.IsInitialized()) {
    // Deferred loading: modules_[wrapper_addr] is nullptr, DigestFatBinary will handle it later
    success = true;
    return &modules_[wrapper_addr];
  }

  // Immediate loading: call DigestFatBinary which handles kpack detection
  hipError_t err = DigestFatBinary(wrapper_addr, modules_[wrapper_addr]);
  success = (err == hipSuccess);
  return &modules_[wrapper_addr];
}

hipError_t StatCO::RemoveFatBinary(FatBinaryInfo** module) {
  std::scoped_lock lock(sclock_);

  // Host-asynchronous initialization can still reference a variable's host shadow and device
  // symbol when its fat binary is unloaded.
  if (!HIP_SKIP_ABORT_ON_GPU_ERROR || !amd::Device::IsGPUInError()) {
    for (auto& [deviceId, state] : managedVarInitialization_.deviceStates) {
      if (state.phase != DeferredInitManagedVarState::Phase::InProgress ||
          state.modulesInProgress.count(module) == 0) {
        continue;
      }

      assert(state.completion != nullptr);
      if (state.completion->awaitCompletion()) {
        const uint64_t initializedSequenceNumber = state.queuedUpToSequenceNumber;
        state.MarkCompleted();
        managedVarInitialization_.initializedUpToSequenceNumberByDevice[deviceId].store(
            initializedSequenceNumber, std::memory_order_release);
      } else {
        state.MarkFailed(hipErrorUnknown);
      }
    }
  }

  auto hostVarsIter = module_to_hostVars_.find(module);
  if (hostVarsIter != module_to_hostVars_.end()) {
    for (auto& hostVar : hostVarsIter->second) {
      auto varIter = vars_.find(hostVar);
      if (varIter == vars_.end()) {
        LogPrintfError("RemoveFatBinary: Unable to find module 0x%x hostVar 0x%x", module, hostVar);
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
      if (managedVar->GetAllocFlag()) {  // check if it is a managed or host alloc
        err = ihipFree(*(static_cast<void**>(managedVar->GetManagedVarPtr())));
      } else {
        void** pointer = static_cast<void**>(managedVar->GetManagedVarPtr());
        amd::Os::releaseMemory(*pointer, managedVar->GetSize());
      }
      assert(err == hipSuccess);
      managedVarInitialization_.sequenceNumberByVariable.erase(managedVar);
      delete managedVar;
    }
    managedVars_.erase(managedVarsIter);
  }

  auto hostFuncsIter = module_to_hostFunctions_.find(module);
  if (hostFuncsIter != module_to_hostFunctions_.end()) {
    for (auto& hostFunc : hostFuncsIter->second) {
      auto funcIter = functions_.find(hostFunc);
      if (funcIter == functions_.end()) {
        LogPrintfError("RemoveFatBinary: Unable to find module 0x%x hostFunc 0x%x", module,
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
      LogPrintfError("RemoveFatBinary: Unable to find module 0x%x via hostModule 0x%x", module,
                     hostModule);
    }
    module_to_hostModule_.erase(hostModuleIter);
  }

  return hipSuccess;
}

// =================================================================================================
void StatCO::RemoveAllFatBinaries() {
  std::scoped_lock lock(sclock_);

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
        amd::Memory* mem = nullptr;
        if (managed_var->GetDeviceVarPtr(&mem, dev->deviceId()) == hipSuccess && mem) {
          // Free device memory (also deletes the device ptr)
          [[maybe_unused]] hipError_t err = ihipFree(memDevPtr(mem));
          assert(err == hipSuccess);
        }
      }

      // Free the managed memory allocation itself
      void** managed_ptr = static_cast<void**>(managed_var->GetManagedVarPtr());
      if (managed_var->GetAllocFlag()) {
        // Memory was allocated with ihipMallocManaged - use ihipFree
        [[maybe_unused]] hipError_t err = ihipFree(*managed_ptr);
        assert(err == hipSuccess);
      } else {
        // Memory was allocated with OS-level allocator - use OS release
        amd::Os::releaseMemory(*managed_ptr, managed_var->GetSize());
      }
      delete managed_var;
    }
  }
  managedVars_.clear();
  managedVarInitialization_.sequenceNumberByVariable.clear();
  managedVarInitialization_.deviceStates.clear();
  for (auto& sequenceNumber : managedVarInitialization_.initializedUpToSequenceNumberByDevice) {
    sequenceNumber.store(0, std::memory_order_relaxed);
  }
  managedVarInitialization_.lastAssignedSequenceNumber.store(0, std::memory_order_release);

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

namespace {

// Helper function finds entries and returns error codes
template <typename MapT> hipError_t FindRegistered(MapT& map, const void* key,
                                                   hipError_t notFoundErr, hipError_t badModuleErr,
                                                   typename MapT::mapped_type& value,
                                                   FatBinaryInfo**& module) {
  auto it = map.find(key);
  if (it == map.end()) {
    return notFoundErr;
  }
  value = it->second;
  module = value->ModuleInfo();
  if (module == nullptr) {
    return badModuleErr;
  }
  return hipSuccess;
}
}  // namespace

hipError_t StatCO::RegisterFunction(const void* hostFunction, Function* func) {
  std::scoped_lock lock(sclock_);

  if (functions_.find(hostFunction) != functions_.end()) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
             "hostFunctionPtr: 0x%x already exists", hostFunction);
    delete func;
  } else {
    functions_.insert(std::make_pair(hostFunction, func));
    module_to_hostFunctions_[func->ModuleInfo()].push_back(hostFunction);
  }

  return hipSuccess;
}

const char* StatCO::GetFuncName(const void* hostFunction) {
  std::shared_lock<std::shared_mutex> lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return nullptr;
  }
  return it->second->GetName().c_str();
}

hipError_t StatCO::GetFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId) {
  Function* func = nullptr;
  FatBinaryInfo** module = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(sclock_);

    hipError_t err = FindRegistered(functions_, hostFunction, hipErrorInvalidSymbol,
                                    hipErrorInvalidDeviceFunction, func, module);
    if (err != hipSuccess) {
      return err;
    }
    if (*module != nullptr) {
      return func->GetStatFunc(hfunc, deviceId);
    }
  }

  // Lazy load: upgrade to exclusive lock.
  std::unique_lock<std::shared_mutex> wlock(sclock_);
  // Re-find after acquiring exclusive lock (concurrent removal may have occurred).
  hipError_t err = FindRegistered(functions_, hostFunction, hipErrorInvalidSymbol,
                                  hipErrorInvalidDeviceFunction, func, module);
  if (err != hipSuccess) {
    return err;
  }
  if (*module == nullptr) {
    err = DigestFatBinary(module_to_hostModule_[module], *module);
    if (err != hipSuccess) {
      return err;
    }
  }
  return func->GetStatFunc(hfunc, deviceId);
}

hipError_t StatCO::GetFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction,
                               int deviceId) {
  Function* func = nullptr;
  FatBinaryInfo** module = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(sclock_);

    hipError_t err = FindRegistered(functions_, hostFunction, hipErrorInvalidSymbol,
                                    hipErrorInvalidDeviceFunction, func, module);
    if (err != hipSuccess) {
      return err;
    }
    if (*module != nullptr) {
      return func->GetStatFuncAttr(func_attr, deviceId);
    }
  }

  // Lazy load: upgrade to exclusive lock.
  std::unique_lock<std::shared_mutex> wlock(sclock_);
  // Re-find after acquiring exclusive lock (concurrent removal may have occurred).
  hipError_t err = FindRegistered(functions_, hostFunction, hipErrorInvalidSymbol,
                                  hipErrorInvalidDeviceFunction, func, module);
  if (err != hipSuccess) {
    return err;
  }
  if (*module == nullptr) {
    std::ignore = DigestFatBinary(module_to_hostModule_[module], *module);
  }
  return func->GetStatFuncAttr(func_attr, deviceId);
}

hipError_t StatCO::RegisterGlobalVar(const void* hostVar, Var* var) {
  std::scoped_lock lock(sclock_);

  auto var_it = vars_.find(hostVar);
  if ((var_it != vars_.end()) && (var_it->second->GetName() != var->GetName())) {
    return hipErrorInvalidSymbol;
  }

  vars_.insert(std::make_pair(hostVar, var));
  module_to_hostVars_[var->ModuleInfo()].push_back(hostVar);
  return hipSuccess;
}

hipError_t StatCO::GetGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                                size_t* size_ptr) {
  Var* var = nullptr;
  FatBinaryInfo** module = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(sclock_);

    hipError_t err =
        FindRegistered(vars_, hostVar, hipErrorInvalidSymbol, hipErrorInvalidSymbol, var, module);
    if (err != hipSuccess) {
      return err;
    }
    if (*module != nullptr) {
      // Read-only fast path: GetDeviceVarPtr only reads dMem_[deviceId], it does not lazily
      // initialize it. Lazy init writes dMem_ and must run under the exclusive lock (see the
      // slow path below), so it is not safe to call GetStatDeviceVar here under the shared lock.
      amd::Memory* mem = nullptr;
      IHIP_RETURN_ONFAIL(var->GetDeviceVarPtr(&mem, deviceId));
      if (mem != nullptr) {
        *dev_ptr = memDevPtr(mem);
        *size_ptr = mem->getSize();
        return hipSuccess;
      }
      // mem == nullptr: either not yet initialized, or a size-0 global. Fall through to the
      // exclusive-lock slow path, which lazily initializes dMem_ via GetStatDeviceVar.
    }
  }

  // Lazy load: upgrade to exclusive lock.
  std::unique_lock<std::shared_mutex> wlock(sclock_);
  // Re-find after acquiring exclusive lock (concurrent removal may have occurred).
  hipError_t err =
      FindRegistered(vars_, hostVar, hipErrorInvalidSymbol, hipErrorInvalidSymbol, var, module);
  if (err != hipSuccess) {
    return err;
  }
  if (*module == nullptr) {
    std::ignore = DigestFatBinary(module_to_hostModule_[module], *module);
  }
  amd::Memory* mem = nullptr;
  IHIP_RETURN_ONFAIL(var->GetStatDeviceVar(&mem, deviceId));

  // Handle size-0 globals: return null device pointer and size 0.
  *dev_ptr = (mem == nullptr) ? 0 : memDevPtr(mem);
  *size_ptr = (mem == nullptr) ? 0 : mem->getSize();
  return hipSuccess;
}

hipError_t StatCO::RegisterManagedVar(Var* var) {
  std::scoped_lock lock(sclock_);
  managedVars_[var->ModuleInfo()].push_back(var);
  const uint64_t sequenceNumber =
      managedVarInitialization_.lastAssignedSequenceNumber.load(std::memory_order_relaxed) + 1;
  managedVarInitialization_.sequenceNumberByVariable.emplace(var, sequenceNumber);
  managedVarInitialization_.lastAssignedSequenceNumber.store(sequenceNumber,
                                                             std::memory_order_release);
  return hipSuccess;
}

// ================================================================================================
void StatCO::ResizeForDevices(size_t device_count) {
  std::scoped_lock lock(sclock_);
  for (const auto& it : vars_) {
    it.second->ResizeDVar(device_count);
  }
  for (const auto& it : managedVars_) {
    for (const auto& var : it.second) {
      var->ResizeDVar(device_count);
    }
  }
  for (const auto& it : functions_) {
    it.second->ResizeDFunc(device_count);
  }
  managedVarInitialization_.InitializeDevices(device_count);
}

// ================================================================================================
hipError_t StatCO::InitManagedVarDevicePtr(int deviceId, hip::Stream* orderStream) {
  assert(deviceId >= 0 &&
         static_cast<size_t>(deviceId) <
             managedVarInitialization_.initializedUpToSequenceNumberByDevice.size());

  // FAST PATH (lock-free): already initialized for this device
  const uint64_t targetSequenceNumber =
      managedVarInitialization_.lastAssignedSequenceNumber.load(
          std::memory_order_acquire);
  if (managedVarInitialization_.initializedUpToSequenceNumberByDevice[deviceId].load(
          std::memory_order_acquire) == targetSequenceNumber) {
    return hipSuccess;
  }

  // SLOW PATH: acquire exclusive lock, re-check, then do init work
  std::unique_lock<std::shared_mutex> lock(sclock_);
  // A variable can be registered while this thread waits for the lock, so the sequence number
  // observed on the fast path is stale here and must be reloaded before comparing.
  const uint64_t currentSequenceNumber =
      managedVarInitialization_.lastAssignedSequenceNumber.load(
          std::memory_order_relaxed);
  if (managedVarInitialization_.initializedUpToSequenceNumberByDevice[deviceId].load(
          std::memory_order_relaxed) == currentSequenceNumber) {
    return hipSuccess;
  }

  using Phase = DeferredInitManagedVarState::Phase;
  DeferredInitManagedVarState& devInitState = managedVarInitialization_.deviceStates[deviceId];

  if (devInitState.phase == Phase::Failed) {
    return devInitState.terminalError;
  }

  if (devInitState.phase == Phase::InProgress) {
    const int32_t completionStatus = devInitState.completion->status();
    if (completionStatus < CL_COMPLETE) {
      devInitState.MarkFailed(hipErrorUnknown);
      return hipErrorUnknown;
    }
    if (completionStatus == CL_COMPLETE) {
      devInitState.MarkCompleted();
    }
  }

  if (devInitState.queuedUpToSequenceNumber < currentSequenceNumber) {
    hip::Stream* stream = g_devices.at(deviceId)->NullStream();
    if (stream == nullptr) {
      ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
      return hipErrorInvalidResourceHandle;
    }

    bool queuedWork = false;
    auto trackQueuedWork = [&](uint64_t accountedSequenceNumber) -> hipError_t {
      if (!queuedWork) {
        return hipSuccess;
      }

      amd::Command* completion = new amd::Marker(*stream, /*userVisible=*/false);
      if (completion == nullptr) {
        // Preserve lifetime safety if bookkeeping allocation fails after copies were queued.
        if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
          devInitState.MarkFailed(hipErrorUnknown);
          return hipErrorOutOfMemory;
        }
        stream->finish();
        devInitState.queuedUpToSequenceNumber = accountedSequenceNumber;
        devInitState.MarkCompleted();
        managedVarInitialization_.initializedUpToSequenceNumberByDevice[deviceId].store(
            accountedSequenceNumber, std::memory_order_release);
        return hipErrorOutOfMemory;
      }
      completion->enqueue();
      devInitState.completion.reset(completion);
      devInitState.phase = Phase::InProgress;
      devInitState.queuedUpToSequenceNumber = accountedSequenceNumber;
      return hipSuccess;
    };
    auto returnAfterTrackingQueuedWork = [&](hipError_t error) -> hipError_t {
      // A failed snapshot may have gaps because managedVars_ is unordered. Retry the whole
      // unaccounted range after the queued work reaches a terminal state.
      const hipError_t trackingStatus =
          trackQueuedWork(devInitState.queuedUpToSequenceNumber);
      return trackingStatus == hipSuccess ? error : trackingStatus;
    };

    for (auto& vecIter : managedVars_) {
      for (auto& var : vecIter.second) {

        const uint64_t sequenceNumber =
            managedVarInitialization_.sequenceNumberByVariable.at(var);
        assert(sequenceNumber <= currentSequenceNumber);
        if (sequenceNumber <= devInitState.queuedUpToSequenceNumber) {
          continue;
        }

        // Lazy load
        FatBinaryInfo** module = var->ModuleInfo();
        if (*(module) == nullptr) {
          std::ignore = DigestFatBinary(module_to_hostModule_[module], *module);
        }

        hipError_t status = var->AllocateManagedVarPtr();
        if (status != hipSuccess) {
          return returnAfterTrackingQueuedWork(status);
        }

        amd::Memory* mem = nullptr;
        status = var->GetStatDeviceVar(&mem, deviceId);
        if (status != hipSuccess) {
          return returnAfterTrackingQueuedWork(status);
        }

        // Preserve host-asynchronous API semantics: deferred initialization must not make the
        // calling host thread wait.
        status = ihipMemcpy(reinterpret_cast<address>(memDevPtr(mem)), var->GetManagedVarPtr(),
                            mem->getSize(), hipMemcpyHostToDevice, *stream,
                            /*isHostAsync=*/true, /*isGPUAsync=*/true);
        if (status != hipSuccess) {
          return returnAfterTrackingQueuedWork(status);
        }
        queuedWork = true;
        devInitState.modulesInProgress.insert(module);
      }
    }

    if (queuedWork) {
      // Copies are profiled individually; keep this aggregate bookkeeping marker internal.
      IHIP_RETURN_ONFAIL(trackQueuedWork(currentSequenceNumber));
    } else if (devInitState.phase == Phase::NotStarted) {
      // No copy was needed, so there is no completion marker to track.
      devInitState.MarkCompleted();
    }
  } else if (devInitState.phase == Phase::NotStarted) {
    devInitState.MarkCompleted();
  }

  if (devInitState.phase == Phase::InProgress) {
    OrderStreamAfterManagedVarInitialization(deviceId, orderStream, devInitState);
  } else {
    // Failed returns earlier and NotStarted transitions above, so only Completed reaches here.
    managedVarInitialization_.initializedUpToSequenceNumberByDevice[deviceId].store(
        devInitState.queuedUpToSequenceNumber, std::memory_order_release);
  }
  return hipSuccess;
}

// Nonblocking and per-thread streams do not implicitly wait for legacy null-stream work. Order
// them explicitly so kernels cannot use a managed symbol before its device pointer is initialized.
void StatCO::OrderStreamAfterManagedVarInitialization(int deviceId, hip::Stream* orderStream,
                                                      const DeferredInitManagedVarState& state) {
  if (state.completion == nullptr || orderStream == nullptr ||
      orderStream == g_devices.at(deviceId)->GetNullStream()) {
    return;
  }

  amd::Command::EventWaitList waitList{state.completion.get()};
  amd::Command* waitMarker =
      new amd::Marker(*orderStream, /*userVisible=*/false, waitList);
  waitMarker->enqueue();
  waitMarker->release();
}

// ================================================================================================
Var* StatCO::FindDeferredManagedVar(const void* ptr) {
  std::shared_lock<std::shared_mutex> lock(sclock_);
  for (const auto& [_, vars] : managedVars_) {
    for (auto* var : vars) {
      void* base = *static_cast<void**>(var->GetManagedVarPtr());
      if (base != nullptr && base == ptr) return var;
    }
  }
  return nullptr;
}

// ================================================================================================
void StatCO::ForEachFatBinaryBlob(void (*cb)(const void*)) const {
  std::shared_lock<std::shared_mutex> lock(sclock_);
  for (const auto& [data, _] : modules_) {
    cb(data);
  }
}
}  // namespace hip
