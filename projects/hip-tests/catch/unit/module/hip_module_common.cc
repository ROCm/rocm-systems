/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"

#include <fstream>

#include <hip_test_common.hh>
#include <hip_test_filesystem.hh>
#include <hip/hiprtc.h>

ModuleGuard ModuleGuard::LoadModule(const char* fname) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, fname));
  return ModuleGuard{module};
}

ModuleGuard ModuleGuard::InitModule(const char* fname) {
  HIP_CHECK(hipFree(nullptr));
  return LoadModule(fname);
}

ModuleGuard ModuleGuard::LoadModuleDataFile(const char* fname) {
  const auto loaded_module = LoadModuleIntoBuffer(fname);
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
  return ModuleGuard{module};
}

ModuleGuard ModuleGuard::LoadModuleDataRTC(const char* code) {
  const auto rtc = CreateRTCCharArray(code);
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, rtc.data()));
  return ModuleGuard{module};
}

// Load module into buffer instead of mapping file to avoid platform specific mechanisms
std::vector<char> LoadModuleIntoBuffer(const char* path_string) {
  std::ifstream file_stream(path_string, std::ios::binary | std::ios::in);
  REQUIRE(file_stream);
  std::vector<char> empty_module((std::istreambuf_iterator<char>(file_stream)),
                                 std::istreambuf_iterator<char>());
  file_stream.close();
  empty_module.push_back('\0');
  return empty_module;
}

std::vector<char> CreateRTCCharArray(const char* src) {
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, src, "prog", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 0, nullptr));
  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size, '\0');
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return code;
}
