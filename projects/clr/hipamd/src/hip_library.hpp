/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <hip/hip_runtime.h>

#include "hip_code_object.hpp"

namespace hip {
// An abstract Library container.
//
// Owns a hip::DynCO under the hood; all kernel/global/managed lookups delegate
// to the DynCO so that hipModuleGet* and hipLibraryGet* share a single
// implementation of code-object symbol resolution.
class LibraryContainer {
 public:
  // Create from pointer
  explicit LibraryContainer(const char* code_object);  // from pointer
  // Create from file
  explicit LibraryContainer(const std::string &file_name);  // deep copy from file
  ~LibraryContainer();

  // Load and build the library
  hipError_t BuildIt();

  // Get the total Kernel count in Library
  size_t KernelCount();

  // Get the Kernel from name
  hipError_t Kernel(hipKernel_t* k, const std::string &name);

  // Register the kernel function, make an entry in global state
  void Register(const std::string &name, int device, hipKernel_t k);

  // Enumerate atmost maxKernels kernel handles in this library
  hipError_t EnumerateKernels(hipKernel_t* k, unsigned int maxKernels);
  hipError_t GetKernelName(const char** name, hipKernel_t kernel);

  // Variable lookups for hipLibraryGetGlobal / hipLibraryGetManaged
  hipError_t GetGlobal(const std::string& name, void** dptr, size_t* bytes);
  hipError_t GetManaged(const std::string& name, void** dptr, size_t* bytes);

 private:
  LibraryContainer() = delete;
  LibraryContainer(const LibraryContainer&) = delete;
  LibraryContainer(const LibraryContainer&&) = delete;
  LibraryContainer& operator=(const LibraryContainer&) = delete;
  LibraryContainer& operator=(const LibraryContainer&&) = delete;

  std::mutex lib_mutex_;
  std::atomic_bool built_ = false;
  std::unique_ptr<hip::DynCO> dynco_;
  // Construction args saved until the lazy BuildIt() runs.
  std::string filename_;          // empty when loading from image
  const char* image_ = nullptr;   // valid only when filename_ is empty
  // Cache of hipKernel_t handles keyed by (name, device).
  std::map<std::pair<std::string /* name */, int /* device */>, hipKernel_t> kernels_;
};
}  // namespace hip
