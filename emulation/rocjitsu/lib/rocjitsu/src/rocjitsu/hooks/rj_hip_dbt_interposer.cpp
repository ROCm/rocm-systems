// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hip_dbt_interposer.cpp
/// @brief Route file-backed HIP guest modules through the memory-reader DBT path.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/config/dbt_guest_config.h"

#define __HIP_PLATFORM_AMD__ 1
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hip/hip_runtime_api.h>
RJ_DIAGNOSTIC_POP
#undef __HIP_PLATFORM_AMD__

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <vector>

namespace {

using HipModuleLoad = hipError_t (*)(hipModule_t *, const char *);
using HipModuleLoadData = hipError_t (*)(hipModule_t *, const void *);

[[nodiscard]] uint32_t detect_mach(const std::vector<uint8_t> &bytes) {
  if (bytes.size() < sizeof(rocjitsu::Elf64_Ehdr))
    return 0;
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0 ||
      header.e_ident[rocjitsu::EI_CLASS] != rocjitsu::ELFCLASS64 ||
      header.e_machine != rocjitsu::EM_AMDGPU)
    return 0;
  return header.e_flags & rocjitsu::EF_AMDGPU_MACH;
}

[[nodiscard]] std::optional<std::vector<uint8_t>> read_file(const char *path) {
  if (path == nullptr)
    return std::nullopt;
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    return std::nullopt;
  const std::streamsize size = file.tellg();
  if (size <= 0)
    return std::nullopt;
  file.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char *>(bytes.data()), size))
    return std::nullopt;
  return bytes;
}

[[nodiscard]] std::optional<std::vector<uint8_t>>
prepare_guest_module(const char *path, const rocjitsu::config::DbtGuestConfig &config) {
  std::optional<std::vector<uint8_t>> bytes = read_file(path);
  if (!bytes)
    return std::nullopt;

  const uint32_t guest_mach = rocjitsu::elf_mach_for_name(config.guest_isa);
  const uint32_t detected_mach = detect_mach(*bytes);
  if (detected_mach == 0 || detected_mach != guest_mach)
    return std::nullopt;
  if (rocjitsu::arch_for_elf_mach(guest_mach) == ROCJITSU_CODE_ARCH_INVALID) {
    std::fprintf(stderr, "[rocjitsu-hip-dbt] invalid guest ISA '%s' for %s\n",
                 config.guest_isa.c_str(), path);
    return std::vector<uint8_t>{};
  }

  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, bytes->data(), sizeof(header));
  header.e_flags = rocjitsu::elf_flags_for_target(header.e_flags, guest_mach);
  std::memcpy(bytes->data(), &header, sizeof(header));
  if (config.log_level > 0) {
    std::fprintf(stderr,
                 "[rocjitsu-hip-dbt] loading file-backed %s module through DBT memory "
                 "reader: %s\n",
                 rocjitsu::elf_mach_name(guest_mach), path);
  }
  return bytes;
}

} // namespace

extern "C" RJ_INTERPOSER_EXPORT hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {
  auto *original = reinterpret_cast<HipModuleLoad>(dlsym(RTLD_NEXT, "hipModuleLoad"));
  if (original == nullptr)
    return hipErrorUnknown;

  try {
    std::optional<rocjitsu::config::DbtGuestConfig> config =
        rocjitsu::config::load_dbt_guest_config_from_runtime_config();
    if (!config || !config->enabled)
      return original(module, fname);

    std::optional<std::vector<uint8_t>> guest_module = prepare_guest_module(fname, *config);
    if (!guest_module)
      return original(module, fname);
    if (guest_module->empty())
      return hipErrorNoBinaryForGpu;

    auto *load_data = reinterpret_cast<HipModuleLoadData>(dlsym(RTLD_NEXT, "hipModuleLoadData"));
    if (load_data == nullptr)
      return hipErrorUnknown;
    const hipError_t status = load_data(module, guest_module->data());
    if (status != hipSuccess) {
      std::fprintf(stderr,
                   "[rocjitsu-hip-dbt] memory-backed guest module load failed for %s: hip error "
                   "%d\n",
                   fname ? fname : "<null>", static_cast<int>(status));
    }
    return status;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[rocjitsu-hip-dbt] module preparation failed for %s: %s\n",
                 fname ? fname : "<null>", e.what());
    return hipErrorNoBinaryForGpu;
  } catch (...) {
    std::fprintf(stderr, "[rocjitsu-hip-dbt] module preparation failed for %s\n",
                 fname ? fname : "<null>");
    return hipErrorNoBinaryForGpu;
  }
}
