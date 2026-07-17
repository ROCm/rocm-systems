// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_comgr_hotswap.cpp
/// @brief Minimal COMGR HotSwap ABI backed by rocjitsu DBT.
///
/// ROCr's HotSwap path resolves six COMGR entry points from one explicitly
/// opened shared object.  This library implements only that narrow contract;
/// it is not a replacement for general-purpose COMGR compilation APIs.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

// Keep these ABI mirrors in sync with amd_comgr.h. ROCr mirrors the same
// declarations locally so neither side needs a build-time COMGR dependency.
enum ComgrStatus : int {
  kComgrStatusSuccess = 0,
  kComgrStatusError = 1,
  kComgrStatusInvalidArgument = 2,
  kComgrStatusOutOfResources = 3,
};

enum ComgrDataKind : int {
  kComgrDataKindExecutable = 0x8,
};

struct ComgrData {
  uint64_t handle;
};

struct DataObject {
  int kind = 0;
  std::vector<uint8_t> bytes;
};

std::mutex g_data_mutex;
std::unordered_set<DataObject *> g_data_objects;
std::atomic<uint64_t> g_dump_sequence{0};

[[nodiscard]] bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return false;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized != "0" && normalized != "off" && normalized != "false" &&
         normalized != "no";
}

template <typename... Args> void verbose_log(const char *format, Args... args) {
  if (!env_flag_enabled("HSA_HOTSWAP_VERBOSE"))
    return;
  std::fputs("[rocjitsu-comgr] ", stderr);
  std::fprintf(stderr, format, args...);
}

void dump_failed_input(const DataObject &object) {
  const char *directory = std::getenv("ROCJITSU_HOTSWAP_DUMP_DIR");
  if (directory == nullptr || directory[0] == '\0')
    return;

  const uint64_t sequence = g_dump_sequence.fetch_add(1, std::memory_order_relaxed);
  char path[4096];
  const int length = std::snprintf(path, sizeof(path), "%s/rocjitsu-hotswap-%ld-%llu.hsaco",
                                   directory, static_cast<long>(getpid()),
                                   static_cast<unsigned long long>(sequence));
  if (length < 0 || static_cast<size_t>(length) >= sizeof(path)) {
    verbose_log("failed-input dump path is too long\n");
    return;
  }

  FILE *file = std::fopen(path, "wb");
  if (file == nullptr) {
    verbose_log("could not open failed-input dump %s\n", path);
    return;
  }
  const size_t written = std::fwrite(object.bytes.data(), 1, object.bytes.size(), file);
  const int close_status = std::fclose(file);
  if (written != object.bytes.size() || close_status != 0) {
    verbose_log("failed to write complete failed-input dump %s\n", path);
    return;
  }
  verbose_log("dumped failed input to %s\n", path);
}

[[nodiscard]] DataObject *lookup_data(ComgrData data) {
  auto *object = reinterpret_cast<DataObject *>(static_cast<uintptr_t>(data.handle));
  if (object == nullptr)
    return nullptr;
  std::lock_guard lock(g_data_mutex);
  return g_data_objects.contains(object) ? object : nullptr;
}

[[nodiscard]] ComgrStatus allocate_data(int kind, ComgrData *data) {
  if (data == nullptr || kind != kComgrDataKindExecutable)
    return kComgrStatusInvalidArgument;

  auto object = std::unique_ptr<DataObject>(new (std::nothrow) DataObject());
  if (!object)
    return kComgrStatusOutOfResources;
  object->kind = kind;
  {
    std::lock_guard lock(g_data_mutex);
    try {
      g_data_objects.insert(object.get());
    } catch (const std::bad_alloc &) {
      return kComgrStatusOutOfResources;
    }
  }
  data->handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(object.release()));
  return kComgrStatusSuccess;
}

[[nodiscard]] bool is_gfx1250_b0_to_a0(std::string_view source, std::string_view target) {
  constexpr std::string_view kProcessor = "--gfx1250";
  constexpr std::string_view kB0 = ":gfx1250-b0-specific+";
  constexpr std::string_view kA0 = ":gfx1250-b0-specific-";
  return source.find(kProcessor) != std::string_view::npos &&
         target.find(kProcessor) != std::string_view::npos && source.find(kB0) != source.npos &&
         target.find(kA0) != target.npos;
}

[[nodiscard]] bool is_gfx1250_already_a0(std::string_view source, std::string_view target) {
  constexpr std::string_view kProcessor = "--gfx1250";
  constexpr std::string_view kA0 = ":gfx1250-b0-specific-";
  return source.find(kProcessor) != std::string_view::npos &&
         target.find(kProcessor) != std::string_view::npos && source.find(kA0) != source.npos &&
         target.find(kA0) != target.npos;
}

[[nodiscard]] std::string isa_name_from_elf(const DataObject &object) {
  using namespace rocjitsu;
  if (object.bytes.size() < sizeof(Elf64_Ehdr))
    return {};
  Elf64_Ehdr header{};
  std::memcpy(&header, object.bytes.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || header.e_machine != EM_AMDGPU)
    return {};
  const char *processor = elf_mach_name(header.e_flags);
  if (std::string_view(processor) == "unknown")
    return {};
  return std::string("amdgcn-amd-amdhsa--") + processor;
}

void print_diagnostics(const std::vector<rocjitsu::TranslationDiagnostic> &diagnostics) {
  if (!env_flag_enabled("HSA_HOTSWAP_VERBOSE"))
    return;
  for (const auto &diagnostic : diagnostics) {
    std::fprintf(stderr,
                 "[rocjitsu-comgr] diagnostic kind=%d severity=%d offset=0x%llx mnemonic=%s: %s\n",
                 static_cast<int>(diagnostic.kind), static_cast<int>(diagnostic.severity),
                 static_cast<unsigned long long>(diagnostic.guest_offset.value_or(0)),
                 diagnostic.mnemonic.empty() ? "<none>" : diagnostic.mnemonic.c_str(),
                 diagnostic.message.c_str());
  }
}

} // namespace

extern "C" RJ_API_EXPORT int amd_comgr_create_data(int kind, ComgrData *data) {
  return allocate_data(kind, data);
}

extern "C" RJ_API_EXPORT int amd_comgr_release_data(ComgrData data) {
  auto *object = reinterpret_cast<DataObject *>(static_cast<uintptr_t>(data.handle));
  if (object == nullptr)
    return kComgrStatusInvalidArgument;
  {
    std::lock_guard lock(g_data_mutex);
    auto it = g_data_objects.find(object);
    if (it == g_data_objects.end())
      return kComgrStatusInvalidArgument;
    g_data_objects.erase(it);
  }
  delete object;
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_set_data(ComgrData data, size_t size, const char *bytes) {
  DataObject *object = lookup_data(data);
  if (object == nullptr || size == 0 || bytes == nullptr)
    return kComgrStatusInvalidArgument;
  try {
    object->bytes.assign(reinterpret_cast<const uint8_t *>(bytes),
                         reinterpret_cast<const uint8_t *>(bytes) + size);
  } catch (const std::bad_alloc &) {
    return kComgrStatusOutOfResources;
  }
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_get_data(ComgrData data, size_t *size, char *bytes) {
  DataObject *object = lookup_data(data);
  if (object == nullptr || size == nullptr || object->bytes.empty())
    return kComgrStatusInvalidArgument;
  const size_t required = object->bytes.size();
  if (bytes != nullptr)
    std::memcpy(bytes, object->bytes.data(), std::min(*size, required));
  *size = required;
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_get_data_isa_name(ComgrData data, size_t *size,
                                                          char *isa_name) {
  DataObject *object = lookup_data(data);
  if (object == nullptr || size == nullptr || object->kind != kComgrDataKindExecutable)
    return kComgrStatusInvalidArgument;
  const std::string isa = isa_name_from_elf(*object);
  if (isa.empty())
    return kComgrStatusError;
  const size_t required = isa.size() + 1;
  if (isa_name != nullptr && *size != 0)
    std::memcpy(isa_name, isa.c_str(), std::min(*size, required));
  *size = required;
  return kComgrStatusSuccess;
}

extern "C" RJ_API_EXPORT int amd_comgr_hotswap_rewrite(ComgrData input,
                                                        const char *source_isa_name,
                                                        const char *target_isa_name,
                                                        ComgrData *output) {
  DataObject *input_object = lookup_data(input);
  if (input_object == nullptr || input_object->kind != kComgrDataKindExecutable ||
      input_object->bytes.empty() || source_isa_name == nullptr || target_isa_name == nullptr ||
      output == nullptr)
    return kComgrStatusInvalidArgument;
  output->handle = 0;

  const bool needs_b0_to_a0 = is_gfx1250_b0_to_a0(source_isa_name, target_isa_name);
  const bool already_a0 = is_gfx1250_already_a0(source_isa_name, target_isa_name);
  if (!needs_b0_to_a0 && !already_a0) {
    verbose_log("rejecting unsupported rewrite %s -> %s\n", source_isa_name, target_isa_name);
    return kComgrStatusInvalidArgument;
  }

  try {
    rocjitsu::AmdGpuCodeObject source(input_object->bytes.data(), input_object->bytes.size());
    if (!source.is_valid() || source.target_id() != ROCJITSU_CODE_TARGET_GFX1250) {
      verbose_log("input is not a valid gfx1250 HSA code object\n");
      return kComgrStatusInvalidArgument;
    }

    if (already_a0) {
      ComgrData result{};
      ComgrStatus status = allocate_data(kComgrDataKindExecutable, &result);
      if (status != kComgrStatusSuccess)
        return status;
      DataObject *result_object = lookup_data(result);
      if (result_object == nullptr) {
        (void)amd_comgr_release_data(result);
        return kComgrStatusError;
      }
      result_object->bytes = input_object->bytes;
      *output = result;
      verbose_log("input is already gfx1250 A0-compatible (%s); leaving %zu bytes unchanged\n",
                  source_isa_name, input_object->bytes.size());
      return kComgrStatusSuccess;
    }

    verbose_log("confirmed gfx1250 B0 input (%s); translating to A0 (%s)\n", source_isa_name,
                target_isa_name);

    rocjitsu::BinaryTranslatorOptions options;
    options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
    options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
    options.skip_failed_kernels = false;
    rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250,
                                          ROCJITSU_CODE_ARCH_GFX1250,
                                          rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250, options);
    rocjitsu::TranslatedCodeObject translated = translator.translate(source);
    print_diagnostics(translated.diagnostics);
    if (!translated.ok() || translated.elf_bytes.empty()) {
      dump_failed_input(*input_object);
      verbose_log("B0 -> A0 translation failed\n");
      return kComgrStatusError;
    }

    ComgrData result{};
    ComgrStatus status = allocate_data(kComgrDataKindExecutable, &result);
    if (status != kComgrStatusSuccess)
      return status;
    DataObject *result_object = lookup_data(result);
    if (result_object == nullptr) {
      (void)amd_comgr_release_data(result);
      return kComgrStatusError;
    }
    result_object->bytes = std::move(translated.elf_bytes);
    *output = result;
    verbose_log("translated %zu input bytes to %zu output bytes\n", input_object->bytes.size(),
                result_object->bytes.size());
    return kComgrStatusSuccess;
  } catch (const std::bad_alloc &) {
    return kComgrStatusOutOfResources;
  } catch (const std::exception &error) {
    dump_failed_input(*input_object);
    verbose_log("translation threw: %s\n", error.what());
    return kComgrStatusError;
  } catch (...) {
    dump_failed_input(*input_object);
    verbose_log("translation threw an unknown exception\n");
    return kComgrStatusError;
  }
}
