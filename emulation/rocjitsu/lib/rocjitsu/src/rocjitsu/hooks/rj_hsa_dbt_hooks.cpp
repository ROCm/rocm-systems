// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbt_hooks.cpp
/// @brief HSA tools load-time DBT hook for translating AMDGPU code objects.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. The hook saves the API table entries that are present at
/// `OnLoad()` time, installs wrappers for code-object readers, HSA ISA queries,
/// and agent code-object loads, and invokes rocjitsu DBT before ROCR sees a
/// guest code object. The hook is deliberately strict: when translation is
/// requested and fails, it returns an HSA error instead of retrying the original
/// reader, because the original ELF may target a different GPU ISA.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using rocjitsu::AmdGpuCodeObject;
using rocjitsu::BinaryTranslator;
using rocjitsu::BinaryTranslatorOptions;
using rocjitsu::DiagnosticKind;
using rocjitsu::DiagnosticSeverity;
using rocjitsu::EF_AMDGPU_MACH;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1100;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX940;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX941;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950;
using rocjitsu::Elf64_Ehdr;
using rocjitsu::ELFCLASS64;
using rocjitsu::EM_AMDGPU;
using rocjitsu::has_error_diagnostic;
using rocjitsu::TranslationDiagnostic;

enum HookLogLevel : int {
  kLogDisabled = 0,
  kLogInfo = 1,
  kLogVerbose = 2,
  kLogDebug = 3,
};

int g_log_level = kLogDisabled;

struct TargetInfo {
  std::string_view name;
  rj_code_arch_t arch;
  uint32_t mach;
};

struct TargetId {
  TargetInfo target{};
  std::string hsa_isa_name;
};

constexpr std::array<TargetInfo, 9> kTargets = {{
    {"gfx942", ROCJITSU_CODE_ARCH_CDNA3, EF_AMDGPU_MACH_AMDGCN_GFX942},
    {"gfx950", ROCJITSU_CODE_ARCH_CDNA4, EF_AMDGPU_MACH_AMDGCN_GFX950},
    {"gfx1100", ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
    {"gfx1200", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1200},
    {"gfx1201", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1201},
    {"cdna3", ROCJITSU_CODE_ARCH_CDNA3, EF_AMDGPU_MACH_AMDGCN_GFX942},
    {"cdna4", ROCJITSU_CODE_ARCH_CDNA4, EF_AMDGPU_MACH_AMDGCN_GFX950},
    {"rdna3", ROCJITSU_CODE_ARCH_RDNA3, EF_AMDGPU_MACH_AMDGCN_GFX1100},
    {"rdna4", ROCJITSU_CODE_ARCH_RDNA4, EF_AMDGPU_MACH_AMDGCN_GFX1200},
}};

struct HookConfig {
  TargetInfo target{};
  std::optional<TargetId> source_override;
  int log_level = kLogDisabled;
};

using AmdLoaderCreateFromFileWithOffsetSizeFn = hsa_status_t
HSA_API (*)(hsa_file_t file, size_t offset, size_t size,
            hsa_code_object_reader_t *code_object_reader);

struct AmdLoaderTablePrefix {
  void *query_host_address = nullptr;
  void *query_segment_descriptors = nullptr;
  void *query_executable = nullptr;
  void *executable_iterate_loaded_code_objects = nullptr;
  void *loaded_code_object_get_info = nullptr;
  AmdLoaderCreateFromFileWithOffsetSizeFn create_from_file_with_offset_size = nullptr;
};

[[nodiscard]] std::optional<TargetInfo> parse_target(std::string_view value) {
  for (const TargetInfo &target : kTargets) {
    if (value == target.name)
      return target;
  }
  return std::nullopt;
}

[[nodiscard]] std::string hsa_isa_name(std::string_view target_id) {
  constexpr std::string_view kPrefix = "amdgcn-amd-amdhsa--";
  if (target_id.starts_with(kPrefix))
    return std::string(target_id);
  return std::string(kPrefix) + std::string(target_id);
}

[[nodiscard]] std::string_view target_id_processor(std::string_view value) {
  constexpr std::string_view kPrefix = "amdgcn-amd-amdhsa--";
  if (value.starts_with(kPrefix))
    value.remove_prefix(kPrefix.size());
  const size_t feature_pos = value.find(':');
  if (feature_pos != std::string_view::npos)
    value = value.substr(0, feature_pos);
  return value;
}

[[nodiscard]] std::optional<TargetId> parse_target_id(std::string_view value) {
  auto target = parse_target(target_id_processor(value));
  if (!target)
    return std::nullopt;
  return TargetId{*target, hsa_isa_name(value)};
}

[[nodiscard]] int parse_log_level() {
  const char *value = std::getenv("RJ_DBT_LOG");
  if (value == nullptr || *value == '\0')
    return kLogDisabled;

  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value)
    return kLogDisabled;
  if (parsed < 0)
    return kLogDisabled;
  if (parsed > kLogDebug)
    return kLogDebug;
  return static_cast<int>(parsed);
}

[[nodiscard]] std::optional<HookConfig> parse_config() {
  HookConfig config;
  config.log_level = parse_log_level();

  const char *target_value = std::getenv("RJ_DBT_TARGET_ISA");
  if (target_value == nullptr || *target_value == '\0') {
    std::fprintf(stderr,
                 "[rocjitsu-hooks] RJ_DBT_TARGET_ISA is required for the DBT HSA tools hook\n");
    return std::nullopt;
  }

  auto target = parse_target_id(target_value);
  if (!target) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid RJ_DBT_TARGET_ISA='%s'\n", target_value);
    return std::nullopt;
  }
  config.target = target->target;

  const char *source_value = std::getenv("RJ_DBT_SOURCE_ISA");
  if (source_value != nullptr && *source_value != '\0') {
    auto source = parse_target_id(source_value);
    if (!source) {
      std::fprintf(stderr, "[rocjitsu-hooks] invalid RJ_DBT_SOURCE_ISA='%s'\n", source_value);
      return std::nullopt;
    }
    config.source_override = *source;
  }

  return config;
}

void log_message(int required_level, const char *format, ...) {
  if (g_log_level < required_level)
    return;

  std::fprintf(stderr, "[rocjitsu-hooks] ");
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

[[nodiscard]] const char *arch_name(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "rdna3";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "rdna4";
  default:
    return "invalid";
  }
}

[[nodiscard]] const char *diagnostic_severity_name(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  }
  return "diagnostic";
}

[[nodiscard]] const char *diagnostic_kind_name(DiagnosticKind kind) {
  switch (kind) {
  case DiagnosticKind::UnsupportedGuestArch:
    return "unsupported-guest-arch";
  case DiagnosticKind::KernelDescriptor:
    return "kernel-descriptor";
  case DiagnosticKind::Legalization:
    return "legalization";
  case DiagnosticKind::ExpandMissing:
    return "expand-missing";
  case DiagnosticKind::ExpandFailed:
    return "expand-failed";
  case DiagnosticKind::ResourceLimit:
    return "resource-limit";
  }
  return "unknown";
}

/// @brief Print one structured DBT diagnostic in the same compact style as the CLI.
void print_diagnostic(FILE *stream, const TranslationDiagnostic &diagnostic) {
  std::fprintf(stream, "[rocjitsu-dbt] %s: %s", diagnostic_severity_name(diagnostic.severity),
               diagnostic_kind_name(diagnostic.kind));
  if (diagnostic.guest_offset)
    std::fprintf(stream, " .text+0x%llx",
                 static_cast<unsigned long long>(*diagnostic.guest_offset));
  if (!diagnostic.mnemonic.empty())
    std::fprintf(stream, " %s", diagnostic.mnemonic.c_str());
  std::fprintf(stream, ": %s\n", diagnostic.message.c_str());
  for (const std::string &item : diagnostic.required_work)
    std::fprintf(stream, "[rocjitsu-dbt]   required: %s\n", item.c_str());
}

/// @brief Print errors unconditionally and lower-severity diagnostics when requested.
void print_diagnostics(FILE *stream, std::span<const TranslationDiagnostic> diagnostics,
                       bool include_warnings) {
  for (const TranslationDiagnostic &diagnostic : diagnostics) {
    if (diagnostic.severity == DiagnosticSeverity::Error || include_warnings)
      print_diagnostic(stream, diagnostic);
  }
}

void dump_source_if_requested(const uint8_t *bytes, size_t size) {
  const char *dir = std::getenv("RJ_DBT_DUMP_SOURCE_DIR");
  if (dir == nullptr || *dir == '\0' || bytes == nullptr || size == 0)
    return;

  static std::atomic<uint32_t> index{0};
  const uint32_t current = index.fetch_add(1, std::memory_order_relaxed);
  char path[4096];
  const int written =
      std::snprintf(path, sizeof(path), "%s/rj-dbt-source-%04u-%zu.hsaco", dir, current, size);
  if (written <= 0 || static_cast<size_t>(written) >= sizeof(path))
    return;

  FILE *file = std::fopen(path, "wb");
  if (file == nullptr)
    return;
  const size_t done = std::fwrite(bytes, 1, size, file);
  std::fclose(file);
  if (done == size)
    log_message(kLogInfo, "dumped source code object %s", path);
}

/// @brief Detect the rocjitsu ISA family from an AMDGPU ELF header.
///
/// @details HSA code-object readers are opaque once created, so source ISA
/// detection has to use the ELF bytes captured at reader creation time. The
/// helper checks only ELF identity, machine type, and `EF_AMDGPU_MACH`; semantic
/// validation remains the job of `AmdGpuCodeObject`.
[[nodiscard]] rj_code_arch_t detect_arch_from_elf(const uint8_t *bytes, size_t size) {
  if (bytes == nullptr || size < sizeof(Elf64_Ehdr))
    return ROCJITSU_CODE_ARCH_INVALID;

  Elf64_Ehdr header{};
  std::memcpy(&header, bytes, sizeof(header));
  if (std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0)
    return ROCJITSU_CODE_ARCH_INVALID;
  if (header.e_ident[rocjitsu::EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU)
    return ROCJITSU_CODE_ARCH_INVALID;

  switch (header.e_flags & EF_AMDGPU_MACH) {
  case EF_AMDGPU_MACH_AMDGCN_GFX940:
  case EF_AMDGPU_MACH_AMDGCN_GFX941:
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  case EF_AMDGPU_MACH_AMDGCN_GFX1100:
    return ROCJITSU_CODE_ARCH_RDNA3;
  case EF_AMDGPU_MACH_AMDGCN_GFX1200:
  case EF_AMDGPU_MACH_AMDGCN_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

[[nodiscard]] std::unique_ptr<std::vector<uint8_t>>
read_file_region(hsa_file_t file, size_t offset, size_t size, std::string_view context) {
#if defined(_WIN32)
  (void)file;
  (void)offset;
  (void)size;
  log_message(kLogInfo, "%.*s reader capture is not implemented on this platform",
              static_cast<int>(context.size()), context.data());
  return nullptr;
#else
  if (file < 0 || size == 0)
    return nullptr;

  auto bytes = std::make_unique<std::vector<uint8_t>>(size);
  size_t done = 0;
  while (done < size) {
    const size_t remaining = size - done;
    const ssize_t n =
        pread(file, bytes->data() + done, remaining, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR)
        continue;
      log_message(kLogInfo, "failed to read %.*s code object fd=%d offset=%zu size=%zu errno=%d",
                  static_cast<int>(context.size()), context.data(), file, offset, size, errno);
      return nullptr;
    }
    if (n == 0) {
      log_message(kLogInfo, "short read for %.*s code object fd=%d offset=%zu size=%zu got=%zu",
                  static_cast<int>(context.size()), context.data(), file, offset, size, done);
      return nullptr;
    }
    done += static_cast<size_t>(n);
  }
  return bytes;
#endif
}

[[nodiscard]] std::unique_ptr<std::vector<uint8_t>> read_whole_file(hsa_file_t file,
                                                                    std::string_view context) {
#if defined(_WIN32)
  (void)file;
  (void)context;
  return nullptr;
#else
  struct stat st {};
  if (fstat(file, &st) != 0 || st.st_size <= 0) {
    log_message(kLogInfo, "failed to stat %.*s code object fd=%d errno=%d",
                static_cast<int>(context.size()), context.data(), file, errno);
    return nullptr;
  }
  return read_file_region(file, 0, static_cast<size_t>(st.st_size), context);
#endif
}

/// @brief Process-local map from HSA code-object reader handles to ELF bytes.
///
/// @details `hsa_executable_load_agent_code_object()` receives only an opaque
/// reader handle. The create wrapper records memory-backed reader bytes here so
/// the load wrapper can translate the original ELF. Entries for application
/// readers are non-owning and rely on the application's reader lifetime. Entries
/// for hidden translated readers own a vector so ROCR's memory-reader pointer
/// remains valid while the translated load is in progress.
class CodeObjectReaderRegistry {
public:
  /// @brief Return the singleton registry used by all hook wrappers.
  static CodeObjectReaderRegistry &instance() {
    // ROCR may retain memory-reader backing storage until executable/runtime
    // teardown. Keep the registry alive until process exit instead of relying
    // on tool unload ordering.
    static auto *registry = new CodeObjectReaderRegistry();
    return *registry;
  }

  /// @brief Record bytes backing a code-object reader.
  /// @param reader HSA reader handle used as the lookup key.
  /// @param bytes Start of the ELF image.
  /// @param size Size of the ELF image in bytes.
  /// @param owned Optional owned storage for translated ELF bytes.
  void store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size,
             std::unique_ptr<std::vector<uint8_t>> owned) {
    std::unique_lock lock(mutex_);
    entries_[reader.handle] = Entry{bytes, size, std::move(owned)};
  }

  /// @brief Record bytes owned by the registry under a code-object reader.
  /// @param reader HSA reader handle used as the lookup key.
  /// @param owned Owned ELF bytes that must outlive ROCR loading.
  void store_owned(hsa_code_object_reader_t reader, std::unique_ptr<std::vector<uint8_t>> owned) {
    if (!owned)
      return;
    const uint8_t *bytes = owned->data();
    const size_t size = owned->size();
    store(reader, bytes, size, std::move(owned));
  }

  /// @brief Find bytes previously recorded for @p reader.
  /// @returns true when @p bytes and @p size were populated.
  bool lookup(hsa_code_object_reader_t reader, const uint8_t **bytes, size_t *size) const {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(reader.handle);
    if (it == entries_.end())
      return false;
    *bytes = it->second.bytes;
    *size = it->second.size;
    return true;
  }

  /// @brief Remove one reader entry and release owned translated bytes if any.
  void remove(hsa_code_object_reader_t reader) {
    std::unique_lock lock(mutex_);
    entries_.erase(reader.handle);
  }

  /// @brief Clear all reader entries during tool unload.
  void clear() {
    std::unique_lock lock(mutex_);
    entries_.clear();
  }

private:
  struct Entry {
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::unique_ptr<std::vector<uint8_t>> owned;
  };

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint64_t, Entry> entries_;
};

struct SyntheticIsaRecord {
  hsa_agent_t agent{};
  hsa_isa_t real_isa{};
  std::string name;
};

/// @brief Process-local map from synthetic HSA ISA handles to real host ISAs.
class SyntheticIsaRegistry {
public:
  static SyntheticIsaRegistry &instance() {
    static SyntheticIsaRegistry registry;
    return registry;
  }

  hsa_isa_t get_or_create(hsa_agent_t agent, hsa_isa_t real_isa, std::string_view source_name) {
    std::lock_guard lock(mutex_);
    for (const auto &[handle, record] : entries_) {
      if (record.agent.handle == agent.handle)
        return hsa_isa_t{handle};
    }

    const uint64_t handle = kSyntheticIsaBase | next_id_++;
    entries_[handle] = SyntheticIsaRecord{agent, real_isa, std::string(source_name)};
    log_message(kLogInfo, "spoofing agent=%llu isa=%s synthetic=%llu real=%llu",
                static_cast<unsigned long long>(agent.handle), entries_[handle].name.c_str(),
                static_cast<unsigned long long>(handle),
                static_cast<unsigned long long>(real_isa.handle));
    return hsa_isa_t{handle};
  }

  bool lookup(hsa_isa_t isa, SyntheticIsaRecord *record) const {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(isa.handle);
    if (it == entries_.end())
      return false;
    *record = it->second;
    return true;
  }

  void clear() {
    std::lock_guard lock(mutex_);
    entries_.clear();
    next_id_ = 1;
  }

private:
  static constexpr uint64_t kSyntheticIsaBase = 0x724a000000000000ull;

  mutable std::mutex mutex_;
  uint64_t next_id_ = 1;
  std::unordered_map<uint64_t, SyntheticIsaRecord> entries_;
};

struct FirstIsaCapture {
  bool found = false;
  hsa_isa_t isa{};
};

hsa_status_t HSA_API capture_first_isa(hsa_isa_t isa, void *data) {
  auto *capture = static_cast<FirstIsaCapture *>(data);
  if (!capture->found) {
    capture->found = true;
    capture->isa = isa;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);
hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);
hsa_status_t HSA_API rj_agent_iterate_isas(hsa_agent_t agent,
                                           hsa_status_t (*callback)(hsa_isa_t isa, void *data),
                                           void *data);
hsa_status_t HSA_API rj_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void *value);
hsa_status_t HSA_API rj_system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                         size_t table_length, void *table);
hsa_status_t HSA_API rj_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                     void (*callback)(hsa_status_t status, hsa_queue_t *source,
                                                      void *data),
                                     void *data, uint32_t private_segment_size,
                                     uint32_t group_segment_size, hsa_queue_t **queue);
hsa_status_t HSA_API rj_amd_loader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader);

/// @brief Process-local HSA API table patch state for the rocjitsu DBT tool.
///
/// @details Tool chaining depends on saving the function pointers that are
/// present at `OnLoad()` time and calling those saved pointers from wrappers.
/// They may already point at another tool's wrapper. `OnUnload()` restores only
/// entries that still point at rocjitsu wrappers so later tools are not
/// accidentally overwritten.
class RjHsaLayer {
public:
  /// @brief Validate the incoming table, save original entries, and install wrappers.
  bool install(HsaApiTable *table, HookConfig config) {
    std::lock_guard lock(mutex_);
    if (active_) {
      std::fprintf(stderr, "[rocjitsu-hooks] OnLoad called while hook is already active\n");
      return false;
    }
    if (!validate_table(table))
      return false;

    table_ = table;
    core_ = table->core_;
    g_log_level = config.log_level;
    config_ = std::move(config);
    original_agent_get_info_ = core_->hsa_agent_get_info_fn;
    original_queue_create_ = core_->hsa_queue_create_fn;
    original_system_get_major_extension_table_ = core_->hsa_system_get_major_extension_table_fn;
    original_agent_iterate_isas_ = core_->hsa_agent_iterate_isas_fn;
    original_isa_get_info_alt_ = core_->hsa_isa_get_info_alt_fn;
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_load_agent_code_object_ = core_->hsa_executable_load_agent_code_object_fn;

    if (original_agent_get_info_ == nullptr || original_queue_create_ == nullptr ||
        original_system_get_major_extension_table_ == nullptr ||
        original_agent_iterate_isas_ == nullptr || original_isa_get_info_alt_ == nullptr ||
        original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_load_agent_code_object_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] HSA core table contains null code-object entries\n");
      clear_unlocked();
      return false;
    }

    core_->hsa_agent_iterate_isas_fn = rj_agent_iterate_isas;
    core_->hsa_isa_get_info_alt_fn = rj_isa_get_info_alt;
    core_->hsa_system_get_major_extension_table_fn = rj_system_get_major_extension_table;
    core_->hsa_code_object_reader_create_from_file_fn = rj_code_object_reader_create_from_file;
    core_->hsa_code_object_reader_create_from_memory_fn = rj_code_object_reader_create_from_memory;
    core_->hsa_code_object_reader_destroy_fn = rj_code_object_reader_destroy;
    core_->hsa_executable_load_agent_code_object_fn = rj_executable_load_agent_code_object;
    core_->hsa_queue_create_fn = rj_queue_create;
    active_ = true;

    if (config_->source_override) {
      log_message(kLogInfo, "installed DBT hook source=%s target=%s arch=%s mach=0x%x",
                  config_->source_override->hsa_isa_name.c_str(), config_->target.name.data(),
                  arch_name(config_->target.arch), config_->target.mach);
    } else {
      log_message(kLogInfo, "installed DBT hook target=%s arch=%s mach=0x%x",
                  config_->target.name.data(), arch_name(config_->target.arch),
                  config_->target.mach);
    }
    return true;
  }

  /// @brief Restore rocjitsu wrappers if still installed and clear owned state.
  void uninstall() {
    std::lock_guard lock(mutex_);
    if (active_ && core_ != nullptr) {
      if (core_->hsa_agent_iterate_isas_fn == rj_agent_iterate_isas)
        core_->hsa_agent_iterate_isas_fn = original_agent_iterate_isas_;
      if (core_->hsa_isa_get_info_alt_fn == rj_isa_get_info_alt)
        core_->hsa_isa_get_info_alt_fn = original_isa_get_info_alt_;
      if (core_->hsa_system_get_major_extension_table_fn == rj_system_get_major_extension_table)
        core_->hsa_system_get_major_extension_table_fn = original_system_get_major_extension_table_;
      if (core_->hsa_code_object_reader_create_from_file_fn ==
          rj_code_object_reader_create_from_file)
        core_->hsa_code_object_reader_create_from_file_fn = original_create_from_file_;
      if (core_->hsa_code_object_reader_create_from_memory_fn ==
          rj_code_object_reader_create_from_memory)
        core_->hsa_code_object_reader_create_from_memory_fn = original_create_from_memory_;
      if (core_->hsa_code_object_reader_destroy_fn == rj_code_object_reader_destroy)
        core_->hsa_code_object_reader_destroy_fn = original_destroy_;
      if (core_->hsa_executable_load_agent_code_object_fn == rj_executable_load_agent_code_object)
        core_->hsa_executable_load_agent_code_object_fn = original_load_agent_code_object_;
      if (core_->hsa_queue_create_fn == rj_queue_create)
        core_->hsa_queue_create_fn = original_queue_create_;
    }

    SyntheticIsaRegistry::instance().clear();
    clear_unlocked();
  }

  [[nodiscard]] int log_level() const {
    std::lock_guard lock(mutex_);
    return config_ ? config_->log_level : kLogDisabled;
  }

  [[nodiscard]] HookConfig config() const {
    std::lock_guard lock(mutex_);
    return *config_;
  }

  [[nodiscard]] decltype(hsa_agent_get_info) *agent_get_info() const {
    std::lock_guard lock(mutex_);
    return original_agent_get_info_;
  }

  [[nodiscard]] decltype(hsa_queue_create) *queue_create() const {
    std::lock_guard lock(mutex_);
    return original_queue_create_;
  }

  [[nodiscard]] decltype(hsa_system_get_major_extension_table) *
  system_get_major_extension_table() const {
    std::lock_guard lock(mutex_);
    return original_system_get_major_extension_table_;
  }

  [[nodiscard]] decltype(hsa_agent_iterate_isas) *agent_iterate_isas() const {
    std::lock_guard lock(mutex_);
    return original_agent_iterate_isas_;
  }

  [[nodiscard]] decltype(hsa_isa_get_info_alt) *isa_get_info_alt() const {
    std::lock_guard lock(mutex_);
    return original_isa_get_info_alt_;
  }

  [[nodiscard]] decltype(hsa_code_object_reader_create_from_file) *create_from_file() const {
    std::lock_guard lock(mutex_);
    return original_create_from_file_;
  }

  [[nodiscard]] decltype(hsa_code_object_reader_create_from_memory) *create_from_memory() const {
    std::lock_guard lock(mutex_);
    return original_create_from_memory_;
  }

  [[nodiscard]] decltype(hsa_code_object_reader_destroy) *destroy() const {
    std::lock_guard lock(mutex_);
    return original_destroy_;
  }

  [[nodiscard]] decltype(hsa_executable_load_agent_code_object) *load_agent_code_object() const {
    std::lock_guard lock(mutex_);
    return original_load_agent_code_object_;
  }

  [[nodiscard]] AmdLoaderCreateFromFileWithOffsetSizeFn
  amd_loader_create_from_file_with_offset_size() const {
    std::lock_guard lock(mutex_);
    return original_amd_loader_create_from_file_with_offset_size_;
  }

  bool install_amd_loader_wrapper(void *table, size_t table_length) {
    std::lock_guard lock(mutex_);
    if (table == nullptr)
      return false;
    constexpr size_t required_size =
        offsetof(AmdLoaderTablePrefix, create_from_file_with_offset_size) +
        sizeof(AmdLoaderTablePrefix::create_from_file_with_offset_size);
    if (table_length < required_size)
      return false;

    auto *loader = static_cast<AmdLoaderTablePrefix *>(table);
    if (loader->create_from_file_with_offset_size == nullptr)
      return false;
    if (loader->create_from_file_with_offset_size !=
        rj_amd_loader_create_from_file_with_offset_size)
      original_amd_loader_create_from_file_with_offset_size_ =
          loader->create_from_file_with_offset_size;
    loader->create_from_file_with_offset_size = rj_amd_loader_create_from_file_with_offset_size;
    return true;
  }

private:
  [[nodiscard]] static bool validate_table(HsaApiTable *table) {
    if (table == nullptr || table->core_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] invalid HSA API table passed to OnLoad\n");
      return false;
    }

    constexpr size_t required_size =
        offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
        sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn);
    if (table->core_->version.minor_id < required_size) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] HSA core table too small: got %u bytes, need %zu bytes\n",
                   table->core_->version.minor_id, required_size);
      return false;
    }
    return true;
  }

  void clear_unlocked() {
    active_ = false;
    g_log_level = kLogDisabled;
    table_ = nullptr;
    core_ = nullptr;
    config_.reset();
    original_agent_get_info_ = nullptr;
    original_queue_create_ = nullptr;
    original_system_get_major_extension_table_ = nullptr;
    original_agent_iterate_isas_ = nullptr;
    original_isa_get_info_alt_ = nullptr;
    original_create_from_file_ = nullptr;
    original_create_from_memory_ = nullptr;
    original_destroy_ = nullptr;
    original_load_agent_code_object_ = nullptr;
    original_amd_loader_create_from_file_with_offset_size_ = nullptr;
  }

  mutable std::mutex mutex_;
  HsaApiTable *table_ = nullptr;
  CoreApiTable *core_ = nullptr;
  std::optional<HookConfig> config_;
  bool active_ = false;
  decltype(hsa_agent_get_info) *original_agent_get_info_ = nullptr;
  decltype(hsa_queue_create) *original_queue_create_ = nullptr;
  decltype(hsa_system_get_major_extension_table) *original_system_get_major_extension_table_ =
      nullptr;
  decltype(hsa_agent_iterate_isas) *original_agent_iterate_isas_ = nullptr;
  decltype(hsa_isa_get_info_alt) *original_isa_get_info_alt_ = nullptr;
  decltype(hsa_code_object_reader_create_from_file) *original_create_from_file_ = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *original_create_from_memory_ = nullptr;
  decltype(hsa_code_object_reader_destroy) *original_destroy_ = nullptr;
  decltype(hsa_executable_load_agent_code_object) *original_load_agent_code_object_ = nullptr;
  AmdLoaderCreateFromFileWithOffsetSizeFn original_amd_loader_create_from_file_with_offset_size_ =
      nullptr;
};

RjHsaLayer &layer() {
  static RjHsaLayer state;
  return state;
}

hsa_status_t HSA_API rj_agent_iterate_isas(hsa_agent_t agent,
                                           hsa_status_t (*callback)(hsa_isa_t isa, void *data),
                                           void *data) {
  auto *original_iterate = layer().agent_iterate_isas();
  if (original_iterate == nullptr)
    return HSA_STATUS_ERROR;

  HookConfig config = layer().config();
  if (!config.source_override)
    return original_iterate(agent, callback, data);

  auto *original_agent_get_info = layer().agent_get_info();
  if (original_agent_get_info == nullptr)
    return HSA_STATUS_ERROR;

  hsa_device_type_t device_type{};
  hsa_status_t status = original_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS || device_type != HSA_DEVICE_TYPE_GPU)
    return original_iterate(agent, callback, data);

  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  FirstIsaCapture capture;
  status = original_iterate(agent, capture_first_isa, &capture);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  if (!capture.found)
    return HSA_STATUS_SUCCESS;

  hsa_isa_t synthetic = SyntheticIsaRegistry::instance().get_or_create(
      agent, capture.isa, config.source_override->hsa_isa_name);
  return callback(synthetic, data);
}

hsa_status_t HSA_API rj_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
  auto *original = layer().isa_get_info_alt();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  SyntheticIsaRecord record;
  if (!SyntheticIsaRegistry::instance().lookup(isa, &record))
    return original(isa, attribute, value);

  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  switch (attribute) {
  case HSA_ISA_INFO_NAME_LENGTH:
    *static_cast<uint32_t *>(value) = static_cast<uint32_t>(record.name.size());
    return HSA_STATUS_SUCCESS;
  case HSA_ISA_INFO_NAME:
    std::memcpy(value, record.name.data(), record.name.size());
    return HSA_STATUS_SUCCESS;
  default:
    return original(record.real_isa, attribute, value);
  }
}

void remember_owned_reader(hsa_code_object_reader_t reader,
                           std::unique_ptr<std::vector<uint8_t>> bytes, std::string_view source) {
  if (!bytes || bytes->empty()) {
    log_message(kLogInfo, "no bytes captured for %.*s reader=%llu", static_cast<int>(source.size()),
                source.data(), static_cast<unsigned long long>(reader.handle));
    return;
  }

  log_message(kLogDebug, "registered %.*s reader=%llu bytes=%zu", static_cast<int>(source.size()),
              source.data(), static_cast<unsigned long long>(reader.handle), bytes->size());
  CodeObjectReaderRegistry::instance().store_owned(reader, std::move(bytes));
}

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    CodeObjectReaderRegistry::instance().store(
        *code_object_reader, static_cast<const uint8_t *>(code_object), size, nullptr);
    log_message(kLogDebug, "registered reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), size);
  }
  return status;
}

hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    remember_owned_reader(*code_object_reader, read_whole_file(file, "file"), "file");
  }
  return status;
}

hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  CodeObjectReaderRegistry::instance().remove(code_object_reader);

  auto *original = layer().destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  return original(code_object_reader);
}

[[nodiscard]] hsa_status_t create_translated_reader(std::vector<uint8_t> translated,
                                                    hsa_code_object_reader_t *translated_reader) {
  auto owned = std::make_unique<std::vector<uint8_t>>(std::move(translated));
  auto *original_create = layer().create_from_memory();
  if (original_create == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original_create(owned->data(), owned->size(), translated_reader);
  if (status != HSA_STATUS_SUCCESS)
    return status;

  CodeObjectReaderRegistry::instance().store(*translated_reader, owned->data(), owned->size(),
                                             std::move(owned));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API rj_system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                         size_t table_length, void *table) {
  auto *original = layer().system_get_major_extension_table();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(extension, version_major, table_length, table);
  if (status == HSA_STATUS_SUCCESS && extension == HSA_EXTENSION_AMD_LOADER && version_major == 1) {
    if (layer().install_amd_loader_wrapper(table, table_length)) {
      log_message(kLogInfo, "installed AMD loader file-region reader hook");
    }
  }
  return status;
}

hsa_status_t HSA_API rj_amd_loader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().amd_loader_create_from_file_with_offset_size();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, offset, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    remember_owned_reader(*code_object_reader, read_file_region(file, offset, size, "file-region"),
                          "file-region");
  }
  return status;
}

hsa_status_t HSA_API rj_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                     void (*callback)(hsa_status_t status, hsa_queue_t *source,
                                                      void *data),
                                     void *data, uint32_t private_segment_size,
                                     uint32_t group_segment_size, hsa_queue_t **queue) {
  auto *original = layer().queue_create();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  return original(agent, size, type, callback, data, private_segment_size, group_segment_size,
                  queue);
}

hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original_load = layer().load_agent_code_object();
  if (original_load == nullptr)
    return HSA_STATUS_ERROR;

  HookConfig config = layer().config();
  const uint8_t *bytes = nullptr;
  size_t size = 0;
  if (!CodeObjectReaderRegistry::instance().lookup(code_object_reader, &bytes, &size)) {
    if (config.source_override) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] no bytes registered for reader=%llu while source spoofing is "
                   "enabled; refusing original code object\n",
                   static_cast<unsigned long long>(code_object_reader.handle));
      return HSA_STATUS_ERROR;
    }
    log_message(kLogInfo, "no bytes registered for reader=%llu; passing through",
                static_cast<unsigned long long>(code_object_reader.handle));
    return original_load(executable, agent, code_object_reader, options, loaded_code_object);
  }

  const rj_code_arch_t actual_arch = detect_arch_from_elf(bytes, size);
  if (actual_arch == ROCJITSU_CODE_ARCH_INVALID) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to detect source ISA from code-object ELF\n");
    return HSA_STATUS_ERROR;
  }

  if (config.source_override && actual_arch != config.source_override->target.arch) {
    if (actual_arch == config.target.arch) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] refusing native target-ISA code object: actual=%s "
                   "expected_guest=%s target=%s\n",
                   arch_name(actual_arch), arch_name(config.source_override->target.arch),
                   arch_name(config.target.arch));
      return HSA_STATUS_ERROR;
    }
    std::fprintf(stderr,
                 "[rocjitsu-hooks] refusing unexpected source ISA code object: actual=%s "
                 "expected_guest=%s target=%s\n",
                 arch_name(actual_arch), arch_name(config.source_override->target.arch),
                 arch_name(config.target.arch));
    return HSA_STATUS_ERROR;
  }

  const rj_code_arch_t source_arch =
      config.source_override ? config.source_override->target.arch : actual_arch;
  if (source_arch == config.target.arch) {
    log_message(kLogInfo, "source arch %s already matches target arch %s; passing through",
                arch_name(source_arch), arch_name(config.target.arch));
    return original_load(executable, agent, code_object_reader, options, loaded_code_object);
  }

  log_message(kLogInfo, "translating reader=%llu %s -> %s mach=0x%x",
              static_cast<unsigned long long>(code_object_reader.handle), arch_name(source_arch),
              arch_name(config.target.arch), config.target.mach);
  dump_source_if_requested(bytes, size);

  AmdGpuCodeObject source(bytes, size);
  if (!source.is_valid()) {
    std::fprintf(stderr, "[rocjitsu-hooks] source bytes are not a valid AMDGPU code object\n");
    return HSA_STATUS_ERROR;
  }

  rocjitsu::TranslatedCodeObject translated;
  try {
    BinaryTranslatorOptions translator_options;
    BinaryTranslator translator(source_arch, config.target.arch, config.target.mach,
                                translator_options);
    translated = translator.translate(source);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[rocjitsu-hooks] translation threw exception: %s\n", e.what());
    return HSA_STATUS_ERROR;
  }

  print_diagnostics(stderr, translated.diagnostics, config.log_level > kLogDisabled);
  if (translated.elf_bytes.empty() || has_error_diagnostic(translated.diagnostics)) {
    std::fprintf(stderr, "[rocjitsu-hooks] translation failed; refusing original code object\n");
    return HSA_STATUS_ERROR;
  }

  hsa_code_object_reader_t translated_reader{};
  hsa_status_t status =
      create_translated_reader(std::move(translated.elf_bytes), &translated_reader);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to create translated code-object reader: %d\n",
                 static_cast<int>(status));
    return status;
  }

  status = original_load(executable, agent, translated_reader, options, loaded_code_object);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] translated code-object load failed: %d\n",
                 static_cast<int>(status));
  }
  // Keep the translated reader and its memory-backed ELF alive for the process.
  // Some loader paths retain code-object backing storage past this call and only
  // finish with it during executable/runtime teardown.
  return status;
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

/// @brief ROCR HSA tools entry point.
///
/// @details Saves the incoming `CoreApiTable` function pointers and installs
/// DBT load-time wrappers when `RJ_DBT_TARGET_ISA` is configured. The failed
/// tool list is not modified; ROCR owns that state and passes it for diagnostics
/// only.
extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;

  auto config = parse_config();
  if (!config)
    return false;
  return layer().install(table, std::move(*config));
}

/// @brief ROCR HSA tools unload entry point.
///
/// @details Restores rocjitsu wrappers that are still installed. Translated
/// memory-reader storage is intentionally process-lifetime because ROCR loader
/// teardown can outlive the HSA tools unload callback.
extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
