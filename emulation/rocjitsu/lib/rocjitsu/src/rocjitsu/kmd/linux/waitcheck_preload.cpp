// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file waitcheck_preload.cpp
/// @brief LD_PRELOAD entrypoint for checking AMDGPU code objects as HSA loads them.

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/amdgpu_code_object.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include "hsa/hsa.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

extern "C" __attribute__((visibility("default"))) hsa_status_t HSA_API
hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader);

namespace {

using CreateFromFileFn = hsa_status_t (*)(hsa_file_t, hsa_code_object_reader_t *);
using CreateFromMemoryFn = hsa_status_t (*)(const void *, size_t, hsa_code_object_reader_t *);
using CreateFromFileWithOffsetSizeFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                        hsa_code_object_reader_t *);
using SystemGetExtensionTableFn = hsa_status_t (*)(uint16_t, uint16_t, uint16_t, void *);
using SystemGetMajorExtensionTableFn = hsa_status_t (*)(uint16_t, uint16_t, size_t, void *);

std::atomic<bool> g_in_waitcheck{false};
CreateFromFileWithOffsetSizeFn g_real_create_from_file_with_offset_size_from_table = nullptr;

struct AmdLoaderTablePrefix {
  void *query_host_address = nullptr;
  void *query_segment_descriptors = nullptr;
  void *query_executable = nullptr;
  void *executable_iterate_loaded_code_objects = nullptr;
  void *loaded_code_object_get_info = nullptr;
  CreateFromFileWithOffsetSizeFn create_from_file_with_offset_size = nullptr;
};

[[nodiscard]] bool env_enabled(const char *name, bool default_value) {
  const char *value = std::getenv(name);
  if (!value)
    return default_value;
  std::string_view text(value);
  return !(text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF");
}

[[nodiscard]] CreateFromFileFn real_create_from_file() {
  static auto *real = reinterpret_cast<CreateFromFileFn>(
      dlsym(RTLD_NEXT, "hsa_code_object_reader_create_from_file"));
  return real;
}

[[nodiscard]] CreateFromMemoryFn real_create_from_memory() {
  static auto *real = reinterpret_cast<CreateFromMemoryFn>(
      dlsym(RTLD_NEXT, "hsa_code_object_reader_create_from_memory"));
  return real;
}

[[nodiscard]] CreateFromFileWithOffsetSizeFn real_create_from_file_with_offset_size() {
  if (g_real_create_from_file_with_offset_size_from_table)
    return g_real_create_from_file_with_offset_size_from_table;
  static auto *real = reinterpret_cast<CreateFromFileWithOffsetSizeFn>(
      dlsym(RTLD_NEXT, "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size"));
  return real;
}

[[nodiscard]] SystemGetExtensionTableFn real_system_get_extension_table() {
  static auto *real = reinterpret_cast<SystemGetExtensionTableFn>(
      dlsym(RTLD_NEXT, "hsa_system_get_extension_table"));
  return real;
}

[[nodiscard]] SystemGetMajorExtensionTableFn real_system_get_major_extension_table() {
  static auto *real = reinterpret_cast<SystemGetMajorExtensionTableFn>(
      dlsym(RTLD_NEXT, "hsa_system_get_major_extension_table"));
  return real;
}

[[nodiscard]] const char *target_name(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX942:
    return "gfx942";
  case ROCJITSU_CODE_TARGET_GFX950:
    return "gfx950";
  case ROCJITSU_CODE_TARGET_GFX1100:
    return "gfx1100";
  case ROCJITSU_CODE_TARGET_GFX1200:
    return "gfx1200";
  case ROCJITSU_CODE_TARGET_GFX1201:
    return "gfx1201";
  case ROCJITSU_CODE_TARGET_GFX1250:
    return "gfx1250";
  default:
    return "unsupported";
  }
}

void print_report(const rocjitsu::AmdGpuCodeObject &code_object,
                  const rocjitsu::WaitcheckReport &report) {
  if (report.passed())
    return;

  if (report.diagnostics_truncated) {
    std::fprintf(stderr, "rocjitsu-waitcheck: at least %zu waitcnt hazard(s) in %s code object\n",
                 report.diagnostics_observed, target_name(code_object.target_id()));
  } else {
    std::fprintf(stderr, "rocjitsu-waitcheck: %zu waitcnt hazard(s) in %s code object\n",
                 report.diagnostics_observed, target_name(code_object.target_id()));
  }

  constexpr size_t kMaxDiagnostics = 32;
  const size_t limit = std::min(kMaxDiagnostics, report.diagnostics.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto &diag = report.diagnostics[i];
    std::fprintf(stderr, "rocjitsu-waitcheck: %s+0x%llx: %s; producer %s+0x%llx: %s\n",
                 diag.section_name.c_str(), static_cast<unsigned long long>(diag.section_offset),
                 diag.message.c_str(), diag.section_name.c_str(),
                 static_cast<unsigned long long>(diag.producer_section_offset),
                 diag.producer_instruction.c_str());
    std::fprintf(stderr, "rocjitsu-waitcheck:   consumer: %s\n", diag.instruction.c_str());
  }
  if (report.diagnostics.size() > limit) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted %zu additional diagnostic(s)\n",
                 report.diagnostics.size() - limit);
  } else if (report.diagnostics_truncated) {
    std::fprintf(stderr, "rocjitsu-waitcheck: omitted additional diagnostic(s) after limit\n");
  }
}

void print_analysis_failure(const rocjitsu::AmdGpuCodeObject &code_object,
                            const rocjitsu::WaitcheckReport &report) {
  std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed for %s code object",
               target_name(code_object.target_id()));
  if (!report.analysis_error.empty())
    std::fprintf(stderr, ": %s", report.analysis_error.c_str());
  std::fprintf(stderr, "\n");
}

[[nodiscard]] bool amd_loader_table_has_file_offset_reader(size_t table_length) {
  constexpr size_t reader_offset =
      offsetof(AmdLoaderTablePrefix, create_from_file_with_offset_size);
  return table_length >= reader_offset + sizeof(CreateFromFileWithOffsetSizeFn);
}

void patch_amd_loader_extension_table(uint16_t extension, uint16_t version_major,
                                      size_t table_length, void *table) {
  if (extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr ||
      !amd_loader_table_has_file_offset_reader(table_length)) {
    return;
  }

  auto *loader = static_cast<AmdLoaderTablePrefix *>(table);
  CreateFromFileWithOffsetSizeFn original = loader->create_from_file_with_offset_size;
  if (!original)
    return;

  if (original != hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size)
    g_real_create_from_file_with_offset_size_from_table = original;
  loader->create_from_file_with_offset_size =
      hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size;
}

[[nodiscard]] size_t deprecated_amd_loader_table_length(uint16_t version_minor) {
  if (version_minor >= 2)
    return sizeof(AmdLoaderTablePrefix);
  return 0;
}

[[nodiscard]] std::optional<std::vector<uint8_t>>
read_regular_file_range(hsa_file_t file, size_t range_offset, size_t range_size) {
  struct stat st {};
  if (fstat(file, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0)
    return std::nullopt;
  const size_t file_size = static_cast<size_t>(st.st_size);
  if (range_size == 0 || range_offset > file_size || range_size > file_size - range_offset)
    return std::nullopt;
  if (range_offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
    return std::nullopt;

  std::vector<uint8_t> data(range_size);
  size_t offset = 0;
  while (offset < data.size()) {
    const size_t absolute_offset = range_offset + offset;
    if (absolute_offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
      return std::nullopt;
    const size_t chunk =
        std::min(data.size() - offset, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t nread =
        pread(file, data.data() + offset, chunk, static_cast<off_t>(absolute_offset));
    if (nread < 0) {
      if (errno == EINTR)
        continue;
      return std::nullopt;
    }
    if (nread == 0)
      return std::nullopt;
    offset += static_cast<size_t>(nread);
  }
  return data;
}

[[nodiscard]] std::optional<std::vector<uint8_t>> read_regular_file(hsa_file_t file) {
  struct stat st {};
  if (fstat(file, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0)
    return std::nullopt;
  return read_regular_file_range(file, 0, static_cast<size_t>(st.st_size));
}

[[nodiscard]] bool check_code_object(const void *code_object, size_t size) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true) || !code_object || size == 0)
    return true;

  bool expected = false;
  if (!g_in_waitcheck.compare_exchange_strong(expected, true))
    return true;

  bool passed = true;
  try {
    rocjitsu::AmdGpuCodeObject parsed(static_cast<const uint8_t *>(code_object), size);
    const rj_code_arch_t arch = rocjitsu::waitcheck_arch_for_target(parsed.target_id());
    if (parsed.is_valid() && arch != ROCJITSU_CODE_ARCH_INVALID) {
      rocjitsu::WaitcheckOptions options;
      options.max_diagnostics = 32;
      auto report = rocjitsu::analyze_waitcnts(parsed, arch, options);
      if (!report.supported) {
        print_analysis_failure(parsed, report);
        passed = false;
      } else {
        print_report(parsed, report);
        passed = report.passed();
      }
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed: %s\n", ex.what());
  } catch (...) {
    std::fprintf(stderr, "rocjitsu-waitcheck: analysis failed\n");
  }

  g_in_waitcheck.store(false);
  return passed;
}

[[nodiscard]] bool check_code_object_file(hsa_file_t file) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true))
    return true;

  std::optional<std::vector<uint8_t>> code_object = read_regular_file(file);
  if (!code_object)
    return true;
  return check_code_object(code_object->data(), code_object->size());
}

[[nodiscard]] bool check_code_object_file_range(hsa_file_t file, size_t offset, size_t size) {
  if (!env_enabled("ROCJITSU_WAITCHECK", true))
    return true;

  std::optional<std::vector<uint8_t>> code_object = read_regular_file_range(file, offset, size);
  if (!code_object)
    return true;
  return check_code_object(code_object->data(), code_object->size());
}

} // namespace

extern "C" __attribute__((visibility("default"))) hsa_status_t HSA_API
hsa_system_get_extension_table(uint16_t extension, uint16_t version_major, uint16_t version_minor,
                               void *table) {
  auto *real = real_system_get_extension_table();
  if (!real)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  hsa_status_t status = real(extension, version_major, version_minor, table);
  if (status == HSA_STATUS_SUCCESS) {
    patch_amd_loader_extension_table(extension, version_major,
                                     deprecated_amd_loader_table_length(version_minor), table);
  }
  return status;
}

extern "C" __attribute__((visibility("default"))) hsa_status_t HSA_API
hsa_system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                     size_t table_length, void *table) {
  auto *real = real_system_get_major_extension_table();
  if (!real)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  hsa_status_t status = real(extension, version_major, table_length, table);
  if (status == HSA_STATUS_SUCCESS)
    patch_amd_loader_extension_table(extension, version_major, table_length, table);
  return status;
}

extern "C" __attribute__((visibility("default"))) hsa_status_t HSA_API
hsa_code_object_reader_create_from_file(hsa_file_t file,
                                        hsa_code_object_reader_t *code_object_reader) {
  auto *real = real_create_from_file();
  if (!real)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  const bool passed = check_code_object_file(file);
  if (!passed && env_enabled("ROCJITSU_WAITCHECK_FAIL", false))
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

  return real(file, code_object_reader);
}

extern "C" __attribute__((visibility("default"))) hsa_status_t HSA_API
hsa_code_object_reader_create_from_memory(const void *code_object, size_t size,
                                          hsa_code_object_reader_t *code_object_reader) {
  auto *real = real_create_from_memory();
  if (!real)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  const bool passed = check_code_object(code_object, size);
  if (!passed && env_enabled("ROCJITSU_WAITCHECK_FAIL", false))
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

  return real(code_object, size, code_object_reader);
}

extern "C" __attribute__((visibility("default"))) hsa_status_t HSA_API
hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader) {
  if (!code_object_reader)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const bool passed = check_code_object_file_range(file, offset, size);
  if (!passed && env_enabled("ROCJITSU_WAITCHECK_FAIL", false))
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

  auto *real = real_create_from_file_with_offset_size();
  if (!real)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  return real(file, offset, size, code_object_reader);
}
