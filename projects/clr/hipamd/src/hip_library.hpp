/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include <hip/hip_runtime.h>

#include "hip_code_object.hpp"
#include "hip_fatbin.hpp"

namespace hip {
// An abstract Library container
class LibraryContainer {
 public:
  // Create from pointer
  explicit LibraryContainer(const char* code_object);  // from pointer
  // Create from file
  explicit LibraryContainer(const std::string file_name);  // deep copy from file
  ~LibraryContainer();

  // Load and build the library
  hipError_t BuildIt();

  // Get the total Kernel count in Library
  size_t KernelCount() const { return functions_.size(); }

  // Get the Kernel from name
  hipError_t Kernel(hipKernel_t* k, std::string name);

  // Get Fatbin pointer
  inline FatBinaryInfo* FatBin() { return fatbin_.get(); }

  // Register the kernel function, make an entry in global state
  void Register(std::string name, int device, hipKernel_t k);

  // Enumerate atmost maxKernels kernel handles in this library
  hipError_t EnumerateKernels(hipKernel_t* k, unsigned int maxKernels);
  hipError_t GetKernelName(const char** name, hipKernel_t kernel);

 private:
  LibraryContainer() = delete;
  LibraryContainer(const LibraryContainer&) = delete;
  LibraryContainer(const LibraryContainer&&) = delete;
  LibraryContainer& operator=(const LibraryContainer&) = delete;
  LibraryContainer& operator=(const LibraryContainer&&) = delete;

  std::mutex lib_mutex_;
  std::atomic_bool built_ = false;
  std::shared_ptr<FatBinaryInfo> fatbin_;
  std::map<std::string, std::shared_ptr<hip::Function>> functions_;
  // Store already looked up kernels for certain devices
  std::map<std::pair<std::string /* name */, int /* device */>, hipKernel_t> kernels_;
};
}  // namespace hip
