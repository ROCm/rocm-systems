/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */
#pragma once

#include "hip_internal.hpp"
#include "hip_fatbin.hpp"
#include "device/device.hpp"
#include "hip_code_object.hpp"

namespace hip_impl {

hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* maxBlocksPerCU, int* numBlocksPerGrid, int* bestBlockSize, const amd::Device& device,
    hipFunction_t func, int inputBlockSize, size_t dynamicSMemSize, bool bCalcPotentialBlkSz);
}  // namespace hip_impl

// Unique file descriptor class
struct UniqueFD {
  UniqueFD(const std::string& fpath, amd::Os::FileDesc fdesc, size_t fsize)
      : fpath_(fpath), fdesc_(fdesc), fsize_(fsize) {}

  const std::string fpath_;        //!< File path of this unique file
  const amd::Os::FileDesc fdesc_;  //!< File Descriptor
  const size_t fsize_;             //!< File Size
};

namespace hip {
class PlatformState {
 public:
  void Init();

  // Dynamic Code Objects functions
  hipError_t LoadModule(hipModule_t* module, const char* fname, const void* image = nullptr);
  hipError_t UnloadModule(hipModule_t hmod);
  bool IsValidDynFunc(const void* hfunc);
  hipError_t GetDynFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* func_name);
  hipError_t GetFuncCount(unsigned int* count, hipModule_t hmod);
  hipError_t GetDynGlobalVar(const char* hostVar, hipModule_t hmod, hipDeviceptr_t* dev_ptr,
                             size_t* size_ptr);
  hipError_t GetDynTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef);

  hipError_t RegisterTexRef(textureReference* texRef, hipModule_t hmod, std::string name);
  hipError_t GetDynTexGlobalVar(textureReference* texRef, hipDeviceptr_t* dev_ptr,
                                size_t* size_ptr);

  // Singleton instance
  static PlatformState& Instance() {
    if (platform_ == nullptr) {
      // __hipRegisterFatBinary() will call this when app starts, thus
      // there is no multiple entry issue here.
      platform_ = new PlatformState();
    }
    return *platform_;
  }

  // Load hip dynamic library
  void* GetDynamicLibraryHandle();
  void SetDynamicLibraryHandle(void* handle);

  // Exec Functions
  void SetupArgument(const void* arg, size_t size, size_t offset);
  void ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);
  void PopExec(ihipExec_t& exec);

  std::shared_ptr<UniqueFD> GetUniqueFileHandle(const std::string& file_path);
  bool CloseUniqueFileHandle(const std::shared_ptr<UniqueFD>& ufd);

  // Logging lock accessor
  amd::Monitor& GetLogLock() { return lg_lock_; }

  // Friend functions for logging access
  friend hipError_t hipExtEnableLogging();
  friend hipError_t hipExtDisableLogging();
  friend hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask);

  inline bool RegisterLibraryFunction(const hipKernel_t f, const hipLibrary_t l) {
    amd::ScopedLock lock(lock_);
    return library_functions_.try_emplace(f, l).second;
  }

  inline bool UnregisterLibraryFunction(const hipKernel_t f) {
    amd::ScopedLock lock(lock_);
    return library_functions_.erase(f) > 0;
  }

  inline bool GetFunctionLibrary(const hipKernel_t f, hipLibrary_t* lib) {
    amd::ScopedLock lock(lock_);
    auto it = library_functions_.find(f);
    if (it != library_functions_.end()) {
      *lib = it->second;
      return true;
    }
    return false;
  }

  hip::StatCO& StatCO() { return statCO_; }  //!< Static Code object var
  bool IsInitialized() const { return initialized_; }

 private:
  PlatformState() : statCO_(*this), log_level_(0), log_size_(0), log_mask_(0) {}
  ~PlatformState() {}

  amd::Monitor lock_{true};         //!< Guards PlatformState globals
  amd::Monitor ufd_lock_{true};     //!< Unique FD Store Lock
  amd::Monitor lg_lock_{true};      //!< Lock for logging operations
  static PlatformState* platform_;  //!< Singleton instance

  //! Dynamic Code Object map, keyin module to get the corresponding object
  std::unordered_map<hipModule_t, hip::DynCO*> dynCO_map_;
  hip::StatCO statCO_;              //!< Static Code object var
  bool initialized_{false};         //!< Platform initialization state
  //! Texture reference map: texRef -> (module, name)
  std::unordered_map<textureReference*, std::pair<hipModule_t, std::string>> texRef_map_;
  //! Unique File Descriptor Map
  std::unordered_map<std::string, std::shared_ptr<UniqueFD>> ufd_map_;
  void* dynamicLibraryHandle_{nullptr};  //!< Handle to dynamic library
  //! Library function map: kernel -> library
  std::unordered_map<hipKernel_t, hipLibrary_t> library_functions_;
  size_t log_level_;  //!< Logging level
  size_t log_size_;   //!< Logging buffer size
  size_t log_mask_;   //!< Logging mask
};
}  // namespace hip
