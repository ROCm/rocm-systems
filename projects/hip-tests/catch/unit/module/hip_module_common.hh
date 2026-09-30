/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <vector>

#include <hip_test_common.hh>

class ModuleGuard {
 public:
  ~ModuleGuard() { static_cast<void>(hipModuleUnload(module_)); }

  ModuleGuard(const ModuleGuard&) = delete;
  ModuleGuard(ModuleGuard&&) = delete;

  static ModuleGuard LoadModule(const char* fname);

  static ModuleGuard InitModule(const char* fname);

  static ModuleGuard LoadModuleDataFile(const char* fname);

  static ModuleGuard LoadModuleDataRTC(const char* code);

  hipModule_t module() const { return module_; }

 private:
  ModuleGuard(const hipModule_t module) : module_{module} {}
  hipModule_t module_ = nullptr;
};

// Load module into buffer instead of mapping file to avoid platform specific mechanisms
std::vector<char> LoadModuleIntoBuffer(const char* path_string);

std::vector<char> CreateRTCCharArray(const char* src);

inline hipFunction_t GetKernel(const hipModule_t module, const char* kname) {
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, kname));
  return kernel;
}