/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_LOADER_PLATFORM_H
#define HIP_LOADER_PLATFORM_H

#include <string>

namespace hip_loader {

class DynamicLibrary {
 public:
  DynamicLibrary() = default;
  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  ~DynamicLibrary();

  bool open(const char* path, std::string* error);
  void* symbol(const char* name) const;

 private:
  void* handle_ = nullptr;
};

void log(const char* format, ...);

}  // namespace hip_loader

#endif
