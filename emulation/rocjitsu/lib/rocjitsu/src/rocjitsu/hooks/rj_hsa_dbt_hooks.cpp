// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbt_hooks.cpp
/// @brief HSA tools load-time DBT hook for translating AMDGPU code objects.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. The hook saves the API table entries that are present at
/// `OnLoad()` time, installs wrappers for code-object reader creation/destruction
/// and agent code-object loads, and invokes rocjitsu DBT before ROCR sees a
/// memory-backed guest code object. The MVP is deliberately strict: when
/// translation is requested and fails, the hook returns an HSA error instead of
/// retrying the original reader, because the original ELF may target a different
/// GPU ISA.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"
#include "util/log.h"

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using rocjitsu::AmdGpuCodeObject;
using rocjitsu::arch_for_elf_mach;
using rocjitsu::BinaryTranslator;
using rocjitsu::BinaryTranslatorOptions;
using rocjitsu::DiagnosticKind;
using rocjitsu::DiagnosticSeverity;
using rocjitsu::EF_AMDGPU_MACH;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1100;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950;
using rocjitsu::Elf64_Ehdr;
using rocjitsu::elf_mach_for_arch;
using rocjitsu::elf_mach_name;
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

std::atomic<int> g_log_level{kLogDisabled};

struct TargetInfo {
  std::string_view name;
  rj_code_arch_t arch;
  uint32_t mach;
};

constexpr std::array<uint32_t, 5> kAcceptedConcreteTargetMachs = {
    EF_AMDGPU_MACH_AMDGCN_GFX942, EF_AMDGPU_MACH_AMDGCN_GFX950, EF_AMDGPU_MACH_AMDGCN_GFX1100,
    EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201};

constexpr std::array<TargetInfo, 4> kArchAliases = {{
    {"cdna3", ROCJITSU_CODE_ARCH_CDNA3, elf_mach_for_arch(ROCJITSU_CODE_ARCH_CDNA3)},
    {"cdna4", ROCJITSU_CODE_ARCH_CDNA4, elf_mach_for_arch(ROCJITSU_CODE_ARCH_CDNA4)},
    {"rdna3", ROCJITSU_CODE_ARCH_RDNA3, elf_mach_for_arch(ROCJITSU_CODE_ARCH_RDNA3)},
    {"rdna4", ROCJITSU_CODE_ARCH_RDNA4, elf_mach_for_arch(ROCJITSU_CODE_ARCH_RDNA4)},
}};

struct HookConfig {
  TargetInfo target{};
  std::optional<TargetInfo> source_override;
  int log_level = kLogDisabled;
};

[[nodiscard]] std::optional<TargetInfo> parse_target(std::string_view value) {
  for (uint32_t mach : kAcceptedConcreteTargetMachs) {
    std::string_view name = elf_mach_name(mach);
    if (value == name) {
      const rj_code_arch_t arch = arch_for_elf_mach(mach);
      if (arch != ROCJITSU_CODE_ARCH_INVALID)
        return TargetInfo{name, arch, mach};
    }
  }
  for (const TargetInfo &target : kArchAliases) {
    if (value == target.name)
      return target;
  }
  return std::nullopt;
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

  auto target = parse_target(target_value);
  if (!target) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid RJ_DBT_TARGET_ISA='%s'\n", target_value);
    return std::nullopt;
  }
  config.target = *target;

  const char *source_value = std::getenv("RJ_DBT_SOURCE_ISA");
  if (source_value != nullptr && *source_value != '\0') {
    auto source = parse_target(source_value);
    if (!source) {
      std::fprintf(stderr, "[rocjitsu-hooks] invalid RJ_DBT_SOURCE_ISA='%s'\n", source_value);
      return std::nullopt;
    }
    config.source_override = *source;
  }

  return config;
}

void log_message(int required_level, const char *format, ...) {
  if (g_log_level.load(std::memory_order_relaxed) < required_level)
    return;

  std::array<char, 512> message{};
  va_list args;
  va_start(args, format);
  std::vsnprintf(message.data(), message.size(), format, args);
  va_end(args);

  util::Logger::dbt_hooks(message.data());
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

struct DetectedElfTarget {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t mach = 0;
};

/// @brief Detect the rocjitsu ISA family and exact ELF MACH from an AMDGPU ELF header.
///
/// @details HSA code-object readers are opaque once created, so source ISA
/// detection has to use the ELF bytes captured at reader creation time. The
/// helper checks only ELF identity, machine type, and `EF_AMDGPU_MACH`. The
/// exact MACH is kept because ROCR rejects same-family-but-different-stepping
/// code objects such as gfx1200 ELFs loaded on gfx1201 agents.
[[nodiscard]] DetectedElfTarget detect_target_from_elf(const uint8_t *bytes, size_t size) {
  if (bytes == nullptr || size < sizeof(Elf64_Ehdr))
    return {};

  Elf64_Ehdr header{};
  std::memcpy(&header, bytes, sizeof(header));
  if (std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0)
    return {};
  if (header.e_ident[rocjitsu::EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU)
    return {};

  const uint32_t mach = header.e_flags & EF_AMDGPU_MACH;
  return DetectedElfTarget{arch_for_elf_mach(mach), mach};
}

/// @brief Process-local map from HSA code-object reader handles to ELF bytes.
///
/// @details `hsa_executable_load_agent_code_object()` receives only an opaque
/// reader handle. The create wrapper records memory-backed reader bytes here so
/// the load wrapper can translate the original ELF. The registry uses rocjitsu's
/// intrusive list and fixed-block arena so this C ABI path can report registry
/// exhaustion as an HSA status instead of depending on throwing STL allocation.
/// Entries for application readers are non-owning and rely on the application's
/// reader lifetime. Entries for hidden translated readers own a vector so ROCR's
/// memory-reader pointer remains valid while the translated load is in progress.
class CodeObjectReaderRegistry {
public:
  /// @brief Return the singleton registry used by all hook wrappers.
  static CodeObjectReaderRegistry &instance() {
    static CodeObjectReaderRegistry registry;
    return registry;
  }

  /// @brief Record bytes backing a code-object reader.
  /// @param reader HSA reader handle used as the lookup key.
  /// @param bytes Start of the ELF image.
  /// @param size Size of the ELF image in bytes.
  /// @param owned Optional owned storage for translated ELF bytes.
  [[nodiscard]] bool store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size,
                           std::vector<uint8_t> *owned) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        delete entry->owned;
        entry->bytes = bytes;
        entry->size = size;
        entry->owned = owned;
        return true;
      }
    }

    void *storage = entry_pool_.try_allocate(sizeof(Entry));
    if (storage == nullptr)
      return false;
    auto *entry = new (storage) Entry(reader.handle, bytes, size, owned);
    entries_.push_front(*entry);
    return true;
  }

  /// @brief Find bytes previously recorded for @p reader.
  /// @returns true when @p bytes and @p size were populated.
  bool lookup(hsa_code_object_reader_t reader, const uint8_t **bytes, size_t *size) {
    std::shared_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        *bytes = entry->bytes;
        *size = entry->size;
        return true;
      }
    }
    return false;
  }

  /// @brief Remove one reader entry and release owned translated bytes if any.
  void remove(hsa_code_object_reader_t reader) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        it = entries_.erase(it);
        destroy_entry(entry);
        return;
      }
      ++it;
    }
  }

  /// @brief Clear all reader entries during tool unload.
  void clear() {
    std::unique_lock lock(mutex_);
    while (!entries_.empty()) {
      auto it = entries_.begin();
      auto *entry = static_cast<Entry *>(it.node_pointer());
      entries_.erase(it);
      destroy_entry(entry);
    }
  }

private:
  struct Entry : util::IListNode<Entry> {
    Entry(uint64_t h, const uint8_t *b, size_t s, std::vector<uint8_t> *o)
        : handle(h), bytes(b), size(s), owned(o) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::vector<uint8_t> *owned = nullptr;
  };

  void destroy_entry(Entry *entry) {
    delete entry->owned;
    entry->~Entry();
    entry_pool_.deallocate(entry);
  }

  mutable std::shared_mutex mutex_;
  util::ArenaAlloc<sizeof(Entry), 256, alignof(Entry)> entry_pool_;
  util::IntrusiveList<Entry> entries_;
};

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);
hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);

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
    g_log_level.store(config.log_level, std::memory_order_relaxed);
    config_ = std::move(config);
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_load_agent_code_object_ = core_->hsa_executable_load_agent_code_object_fn;

    if (original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_load_agent_code_object_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] HSA core table contains null code-object entries\n");
      clear_unlocked();
      return false;
    }

    core_->hsa_code_object_reader_create_from_file_fn = rj_code_object_reader_create_from_file;
    core_->hsa_code_object_reader_create_from_memory_fn = rj_code_object_reader_create_from_memory;
    core_->hsa_code_object_reader_destroy_fn = rj_code_object_reader_destroy;
    core_->hsa_executable_load_agent_code_object_fn = rj_executable_load_agent_code_object;
    active_ = true;

    log_message(kLogInfo, "installed DBT hook target=%s arch=%s mach=0x%x",
                config_->target.name.data(), arch_name(config_->target.arch), config_->target.mach);
    return true;
  }

  /// @brief Restore rocjitsu wrappers if still installed and clear owned state.
  void uninstall() {
    std::lock_guard lock(mutex_);
    if (active_ && core_ != nullptr) {
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
    }

    CodeObjectReaderRegistry::instance().clear();
    clear_unlocked();
  }

  [[nodiscard]] int log_level() const {
    std::lock_guard lock(mutex_);
    return config_ ? config_->log_level : kLogDisabled;
  }

  [[nodiscard]] std::optional<HookConfig> config() const {
    std::lock_guard lock(mutex_);
    return config_;
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
    g_log_level.store(kLogDisabled, std::memory_order_relaxed);
    table_ = nullptr;
    core_ = nullptr;
    config_.reset();
    original_create_from_file_ = nullptr;
    original_create_from_memory_ = nullptr;
    original_destroy_ = nullptr;
    original_load_agent_code_object_ = nullptr;
  }

  mutable std::mutex mutex_;
  HsaApiTable *table_ = nullptr;
  CoreApiTable *core_ = nullptr;
  std::optional<HookConfig> config_;
  bool active_ = false;
  decltype(hsa_code_object_reader_create_from_file) *original_create_from_file_ = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *original_create_from_memory_ = nullptr;
  decltype(hsa_code_object_reader_destroy) *original_destroy_ = nullptr;
  decltype(hsa_executable_load_agent_code_object) *original_load_agent_code_object_ = nullptr;
};

RjHsaLayer &layer() {
  static RjHsaLayer state;
  return state;
}

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    if (!CodeObjectReaderRegistry::instance().store(
            *code_object_reader, static_cast<const uint8_t *>(code_object), size, nullptr)) {
      if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
        (void)original_destroy(*code_object_reader);
      *code_object_reader = {};
      std::fprintf(stderr, "[rocjitsu-hooks] failed to track memory-backed code-object reader\n");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
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
    // File-backed readers do not expose stable ELF bytes through the later
    // load-agent callback. Hook the create path anyway so users get a direct
    // warning instead of an unexplained INVALID_ISA pass-through failure.
    std::fprintf(stderr,
                 "[rocjitsu-hooks] file-backed code-object reader=%llu is not translated; use "
                 "hsa_code_object_reader_create_from_memory for DBT hook translation\n",
                 static_cast<unsigned long long>(code_object_reader->handle));
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
  auto *owned = new (std::nothrow) std::vector<uint8_t>(std::move(translated));
  if (owned == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto *original_create = layer().create_from_memory();
  if (original_create == nullptr) {
    delete owned;
    return HSA_STATUS_ERROR;
  }

  const hsa_status_t status = original_create(owned->data(), owned->size(), translated_reader);
  if (status != HSA_STATUS_SUCCESS) {
    delete owned;
    return status;
  }

  if (!CodeObjectReaderRegistry::instance().store(*translated_reader, owned->data(), owned->size(),
                                                  owned)) {
    if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
      (void)original_destroy(*translated_reader);
    *translated_reader = {};
    delete owned;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original_load = layer().load_agent_code_object();
  if (original_load == nullptr)
    return HSA_STATUS_ERROR;

  const uint8_t *bytes = nullptr;
  size_t size = 0;
  if (!CodeObjectReaderRegistry::instance().lookup(code_object_reader, &bytes, &size)) {
    log_message(kLogInfo, "no memory bytes registered for reader=%llu; passing through",
                static_cast<unsigned long long>(code_object_reader.handle));
    return original_load(executable, agent, code_object_reader, options, loaded_code_object);
  }

  auto config = layer().config();
  if (!config) {
    // `OnUnload()` should not normally race active ROCR callbacks, but returning
    // an HSA error here is safer than dereferencing cleared hook state.
    std::fprintf(stderr, "[rocjitsu-hooks] DBT hook layer is inactive during code-object load\n");
    return HSA_STATUS_ERROR;
  }

  const DetectedElfTarget detected = detect_target_from_elf(bytes, size);
  DetectedElfTarget source_target = detected;
  if (config->source_override) {
    source_target.arch = config->source_override->arch;
    source_target.mach = config->source_override->mach;
  }
  if (source_target.arch == ROCJITSU_CODE_ARCH_INVALID || source_target.mach == 0) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to detect source ISA from code-object ELF\n");
    return HSA_STATUS_ERROR;
  }

  if (source_target.arch == config->target.arch && source_target.mach == config->target.mach) {
    log_message(kLogInfo,
                "source target %s arch %s already matches requested target; passing through",
                elf_mach_name(source_target.mach), arch_name(source_target.arch));
    return original_load(executable, agent, code_object_reader, options, loaded_code_object);
  }

  log_message(kLogInfo, "translating reader=%llu %s/%s -> %s/%s mach=0x%x",
              static_cast<unsigned long long>(code_object_reader.handle),
              elf_mach_name(source_target.mach), arch_name(source_target.arch),
              config->target.name.data(), arch_name(config->target.arch), config->target.mach);

  AmdGpuCodeObject source_object(bytes, size);
  if (!source_object.is_valid()) {
    std::fprintf(stderr, "[rocjitsu-hooks] source bytes are not a valid AMDGPU code object\n");
    return HSA_STATUS_ERROR;
  }

  rocjitsu::TranslatedCodeObject translated;
  BinaryTranslatorOptions translator_options;
  BinaryTranslator translator(source_target.arch, config->target.arch, config->target.mach,
                              translator_options);
  translated = translator.translate(source_object);

  print_diagnostics(stderr, translated.diagnostics, config->log_level > kLogDisabled);
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
  CodeObjectReaderRegistry::instance().remove(translated_reader);
  if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
    (void)original_destroy(translated_reader);

  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] translated code-object load failed: %d\n",
                 static_cast<int>(status));
  }
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
/// @details Restores rocjitsu wrappers that are still installed and clears
/// process-local reader state owned by the hook. ROCR also resets the API table
/// after unloading tools, but the hook does its own cleanup so tests and future
/// in-process reload paths do not retain stale translated ELF buffers.
extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
