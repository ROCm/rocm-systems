/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_loader_platform.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace hip_loader {

DynamicLibrary::~DynamicLibrary() {
#if defined(_WIN32)
  if (handle_ != nullptr) {
    FreeLibrary(static_cast<HMODULE>(handle_));
  }
#else
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
#endif
}

bool DynamicLibrary::open(const char* path, std::string* error) {
#if defined(_WIN32)
  handle_ = LoadLibraryA(path);
  if (handle_ == nullptr && error != nullptr) {
    *error = "LoadLibraryA failed";
  }
#else
  handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (handle_ == nullptr && error != nullptr) {
    const char* dl_error = dlerror();
    *error = dl_error == nullptr ? "dlopen failed" : dl_error;
  }
#endif
  return handle_ != nullptr;
}

void* DynamicLibrary::symbol(const char* name) const {
  if (handle_ == nullptr) {
    return nullptr;
  }
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

void log(const char* format, ...) {
  const char* setting = std::getenv("HIP_LOADER_LOG");
  if (setting == nullptr || setting[0] == '\0' || setting[0] == '0') {
    return;
  }

  std::fprintf(stderr, "hip-loader: ");
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

}  // namespace hip_loader
