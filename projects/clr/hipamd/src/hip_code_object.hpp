/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_CODE_OBJECT_HPP
#define HIP_CODE_OBJECT_HPP

#include "hip_global.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "device/device.hpp"
#include "platform/program.hpp"

namespace hip {
namespace symbols {
// In uncompressed mode
constexpr char kOffloadBundleUncompressedMagicStr[] = "__CLANG_OFFLOAD_BUNDLE__";
static constexpr size_t kOffloadBundleUncompressedMagicStrSize =
    sizeof(kOffloadBundleUncompressedMagicStr);

// In compressed mode
constexpr char kOffloadBundleCompressedMagicStr[] = "CCOB";
static constexpr size_t kOffloadBundleCompressedMagicStrSize =
    sizeof(kOffloadBundleCompressedMagicStr);

constexpr char kOffloadKindHip[] = "hip";
constexpr char kOffloadKindHipv4[] = "hipv4";
constexpr char kOffloadKindHcc[] = "hcc";
constexpr char kAmdgcnTargetTriple[] = "amdgcn-amd-amdhsa-";
constexpr char kHipFatBinName[] = "hipfatbin";
constexpr char kHipFatBinName_[] = "hipfatbin-";
constexpr char kOffloadKindHipv4_[] = "hipv4-";  // bundled code objects need the prefix
constexpr char kOffloadHipV4FatBinName_[] = "hipfatbin-hipv4-";

// Fat binary wrapper magic values
constexpr uint32_t kHipfMagic = 0x48495046;  // "HIPF" little-endian (normal fat binary)
constexpr uint32_t kHipkMagic = 0x4B504948;  // "HIPK" little-endian (kpack'd binary)

// Clang Offload bundler description & Header in uncompressed mode.
struct ClangOffloadBundleInfo {
  uint64_t offset;
  uint64_t size;
  uint64_t bundleEntryIdSize;
  const char bundleEntryId[1];
};

struct ClangOffloadBundleUncompressedHeader {
  const char magic[kOffloadBundleUncompressedMagicStrSize - 1];
  uint64_t numOfCodeObjects;
  ClangOffloadBundleInfo desc[1];
};

// Clang Offload bundler description & Header in compressed mode.
struct ClangOffloadBundleCompressedHeader {
  const char magic[kOffloadBundleCompressedMagicStrSize - 1];
  uint16_t versionNumber;
  uint16_t compressionMethod;
  uint32_t totalSize;
  uint32_t uncompressedBinarySize;
  uint64_t Hash;
  const char compressedBinarydesc[1];
};
}  // namespace symbols

// Forward Declaration for friend usage
class PlatformState;

// Code Object base class
class CodeObject {
 public:
  virtual ~CodeObject() {}

 protected:
  CodeObject() {}

 private:
  friend const std::vector<hipModule_t>& modules();
};

// Dynamic Code Object
class DynCO : public CodeObject {
  // Guards Dynamic Code object
  std::recursive_mutex dclock_;

 public:
  DynCO() : device_id_(ihipGetDevice()), fb_info_(nullptr), module_(nullptr) {}
  virtual ~DynCO();

  // LoadsCodeObject and its data
  hipError_t loadCodeObject(const char* fname, const void* image = nullptr);
  hipModule_t getModule() const { return module_; };

  // Device the code object was loaded for at construction. Callers that key
  // per-device caches (e.g., LibraryContainer::kernels_) must use this and
  // not ihipGetDevice(): the active device may differ at use time, but the
  // loaded module is single-device.
  int getDeviceId() const { return device_id_; }

  // Gets GlobalVar/Functions from a dynamically loaded code object
  hipError_t getDynFunc(hipFunction_t* hfunc, const std::string& func_name);
  hipError_t getFuncCount(unsigned int* count);
  bool isValidDynFunc(const void* hfunc);
  hipError_t GetDeviceVar(amd::Memory** mem, const std::string& var_name);
  hip::Var* getVar(const std::string& var_name);

  hipError_t getManagedVarPointer(std::string name, void** pointer, size_t* size_ptr) const {
    auto it = vars_.find(name);
    if (it != vars_.end() && it->second->GetVarKind() == Var::DVK_Managed) {
      if (pointer != nullptr) {
        *pointer = it->second->GetManagedVarPtr();
      }
      if (size_ptr != nullptr) {
        *size_ptr = it->second->GetSize();
      }
    }
    return hipSuccess;
  }

  // Common entry point used by both hipModuleGetGlobal and hipLibraryGetGlobal.
  // Returns the managed host pointer if name is a __managed__ var, otherwise the
  // device pointer of the __device__ global. hipErrorNotFound if name is missing.
  hipError_t GetGlobal(const std::string& name, void** dptr, size_t* bytes);

  // Strict managed-only lookup used by hipLibraryGetManaged.
  // hipErrorNotFound if name is missing or not DVK_Managed.
  hipError_t GetManaged(const std::string& name, void** dptr, size_t* bytes);

  // Names of all loaded kernel functions; used by LibraryContainer::EnumerateKernels.
  std::vector<std::string> getFunctionNames();

 private:
  int device_id_;
  FatBinaryInfo* fb_info_;
  hipModule_t module_;

  // Maps for vars/funcs, could be keyed in with std::string name
  std::unordered_map<std::string, Function*> functions_;
  std::unordered_map<std::string, Var*> vars_;

  // Populate Global Vars/Funcs from an code object(@ module_load)
  hipError_t populateDynGlobalFuncs();
  hipError_t populateDynGlobalVars();
  hipError_t initDynManagedVars(const std::string& managedVar);
};

// Static Code Object
class StatCO : public CodeObject {
 public:
  explicit StatCO(const PlatformState& owner);
  virtual ~StatCO();

  // Add/Remove/Digest Fat Binaries passed to us from "__hipRegisterFatBinary"
  FatBinaryInfo** AddFatBinary(const void* data, bool& success);
  FatBinaryInfo** AddKpackBinary(const void* hipk_metadata, const void* wrapper_addr,
                                 bool& success);
  hipError_t RemoveFatBinary(FatBinaryInfo** module);
  void RemoveAllFatBinaries();

  // Register vars/funcs given to use from __hipRegister[Var/Func/ManagedVar]
  hipError_t RegisterFunction(const void* hostFunction, Function* func);
  hipError_t RegisterGlobalVar(const void* hostVar, Var* var);
  hipError_t RegisterManagedVar(Var* var);

  // Retrive Vars/Funcs for a given hostSidePtr(const void*), unless stated otherwise.
  const char* GetFuncName(const void* hostFunction);
  hipError_t GetFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId);
  hipError_t GetFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId);
  hipError_t GetGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                          size_t* size_ptr);

  // Writes each managed allocation's device pointer into its code-object symbol.
  // When non-null, orderStream receives a dependency on these deferred writes.
  hipError_t InitManagedVarDevicePtr(int deviceId, hip::Stream* orderStream);

  // Find a deferred managed var whose mmap address equals ptr
  Var* FindDeferredManagedVar(const void* ptr);

  // Resize device-specific data structures for all registered functions and variables
  void ResizeForDevices(size_t device_count);

  // Iterate all registered fat binary data pointers — for HRR capture post-registration sweep.
  void ForEachFatBinaryBlob(void (*cb)(const void*)) const;

 private:
  // Lock ordering: sclock_ is always acquired before FatBinaryLock, never the reverse.
  mutable std::shared_mutex sclock_;       //!< Guards Static Code object

  // Precondition: caller must hold sclock_ exclusively.
  hipError_t DigestFatBinary(const void* data, FatBinaryInfo*& programs);
  const PlatformState& owner_;             //!< Reference to owning PlatformState
  //! Populated during __hipRegisterFatBinary
  std::unordered_map<const void*, FatBinaryInfo*> modules_;
  //! Populated during __hipRegisterFuncs
  std::unordered_map<const void*, Function*> functions_;
  std::unordered_map<const void*, Var*> vars_;               //!< Populated during __hipRegisterVars
  //! Populated during __hipRegisterManagedVar
  std::unordered_map<FatBinaryInfo**, std::vector<Var*> > managedVars_;
  //! Reverse mapping of modules to speed up removal
  std::unordered_map<FatBinaryInfo**, const void*> module_to_hostModule_;
  //! Reverse mapping of functions
  std::unordered_map<FatBinaryInfo**, std::vector<const void*> > module_to_hostFunctions_;
  //! Reverse mapping of vars
  std::unordered_map<FatBinaryInfo**, std::vector<const void*> > module_to_hostVars_;

  // Managed variables can be added when a HIP fat binary is loaded, including after the first
  // kernel launch. Each receives a monotonically increasing sequence number that is not reused
  // when a fat binary is unloaded. A device is current when initializedUpToSequenceNumberByDevice
  // equals lastAssignedManagedVarSequenceNumber; otherwise only variables above its queued number
  // need initializing.
  struct DeferredInitManagedVarState {
    enum class Phase { NotStarted, InProgress, Completed, Failed };

    struct CommandDeleter {
      void operator()(amd::Command* command) const {
        if (command != nullptr) {
          command->release();
        }
      }
    };

    Phase phase = Phase::NotStarted;
    std::unique_ptr<amd::Command, CommandDeleter> completion;
    hipError_t terminalError = hipSuccess;
    // Highest managed-variable sequence number queued for initialization on this device.
    uint64_t queuedUpToSequenceNumber = 0;
    // Fat binaries whose symbols are referenced by the current completion marker.
    std::unordered_set<FatBinaryInfo**> modulesInProgress;

    void MarkCompleted() {
      completion.reset();
      modulesInProgress.clear();
      phase = Phase::Completed;
    }

    void MarkFailed(hipError_t error) {
      assert(error != hipSuccess);
      assert(phase != Phase::Failed);
      completion.reset();
      modulesInProgress.clear();
      terminalError = error;
      phase = Phase::Failed;
    }
  };

  struct ManagedVarInitializationTracker {
    void InitializeDevices(size_t deviceCount) {
      // Construct at the final size because std::atomic cannot be moved during vector growth.
      std::vector<std::atomic<uint64_t>> initialized(deviceCount);
      for (auto& sequenceNumber : initialized) {
        sequenceNumber.store(0, std::memory_order_relaxed);
      }
      initializedUpToSequenceNumberByDevice.swap(initialized);
    }

    // Highest sequence number assigned; atomic to support the lock-free launch-path check.
    std::atomic<uint64_t> lastAssignedSequenceNumber{0};
    // Identifies variables added after a device last queued initialization.
    std::unordered_map<const Var*, uint64_t> sequenceNumberByVariable;
    // Tracks initialization work and failures separately for each device.
    std::unordered_map<int, DeferredInitManagedVarState> deviceStates;
    // Highest sequence number fully initialized on each device.
    std::vector<std::atomic<uint64_t>> initializedUpToSequenceNumberByDevice;
  } managedVarInitialization_;

  void OrderStreamAfterManagedVarInitialization(int deviceId, hip::Stream* orderStream,
                                                const DeferredInitManagedVarState& state);
};

};  // namespace hip

#endif /* HIP_CODE_OBJECT_HPP */
