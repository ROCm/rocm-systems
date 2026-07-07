// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbi_hooks.cpp
/// @brief HSA tools load-time hook for opt-in rocJITsu DBI instrumentation.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. This initial DBI hook only parses configuration, installs the
/// code-object reader/load wrappers, logs observed loads when requested, and
/// passes every original code object through unchanged. Later SuperCollider
/// slices will route memory-backed reader bytes through the DBI patcher.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/patch/supercollider_dbi.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"

#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace {

enum HookLogLevel : int {
  kLogDisabled = 0,
  kLogInfo = 1,
  kLogVerbose = 2,
  kLogDebug = 3,
};

enum class CheckTrapMode : uint8_t {
  Lds,
  Flat,
};

std::atomic<int> g_log_level{kLogDisabled};
std::atomic<uint64_t> g_dump_sequence{0};

struct HookConfig {
  bool supercollider = false;
  bool fail_closed = false;
  bool require_patch = false;
  bool probe_nop = false;
  bool probe_trampoline_nop = false;
  bool probe_endpgm = false;
  bool probe_lds_endpgm = false;
  CheckTrapMode check_trap_mode = CheckTrapMode::Lds;
  bool probe_lds_check_trap = false;
  bool probe_flat_check_trap = false;
  bool probe_flat_trap = false;
  bool fault_drop_barrier = false;
  uint32_t fault_barrier_index = 0;
  rocjitsu::SuperColliderDbiDelayMode delay_mode = rocjitsu::SuperColliderDbiDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  uint32_t delay_nops = 0;
  uint32_t max_patches = 1;
  int log_level = kLogDisabled;
  std::string dump_dir;
};

[[nodiscard]] const char *preflight_action_name(rocjitsu::SuperColliderDbiPreflightAction action) {
  switch (action) {
  case rocjitsu::SuperColliderDbiPreflightAction::NotRun:
    return "not-run";
  case rocjitsu::SuperColliderDbiPreflightAction::Candidate:
    return "candidate";
  case rocjitsu::SuperColliderDbiPreflightAction::Skip:
    return "skip";
  case rocjitsu::SuperColliderDbiPreflightAction::Reject:
    return "reject";
  }
  return "unknown";
}

[[nodiscard]] const char *patch_kind_name(rocjitsu::SuperColliderDbiPatchKind kind) {
  switch (kind) {
  case rocjitsu::SuperColliderDbiPatchKind::InlineNopRewrite:
    return "inline-nop-rewrite";
  case rocjitsu::SuperColliderDbiPatchKind::InlineEndpgmRewrite:
    return "inline-endpgm-rewrite";
  case rocjitsu::SuperColliderDbiPatchKind::InlineLdsEndpgmRewrite:
    return "inline-lds-endpgm-rewrite";
  case rocjitsu::SuperColliderDbiPatchKind::InlineLdsLoadCheckTrap:
    return "inline-lds-load-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::InlineLdsStoreCheckTrap:
    return "inline-lds-store-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::LocalCaveLdsLoadCheckTrap:
    return "local-cave-lds-load-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::LocalCaveLdsStoreCheckTrap:
    return "local-cave-lds-store-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::InlineFlatLoadCheckTrap:
    return "inline-flat-load-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::InlineFlatStoreCheckTrap:
    return "inline-flat-store-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::LocalCaveFlatLoadCheckTrap:
    return "local-cave-flat-load-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::LocalCaveFlatStoreCheckTrap:
    return "local-cave-flat-store-check-trap";
  case rocjitsu::SuperColliderDbiPatchKind::InlineFlatTrapRewrite:
    return "inline-flat-trap-rewrite";
  case rocjitsu::SuperColliderDbiPatchKind::InlineBarrierNopRewrite:
    return "inline-barrier-nop-rewrite";
  case rocjitsu::SuperColliderDbiPatchKind::TrampolineNop:
    return "trampoline-nop";
  }
  return "unknown";
}

[[nodiscard]] const char *delay_mode_name(rocjitsu::SuperColliderDbiDelayMode mode) {
  switch (mode) {
  case rocjitsu::SuperColliderDbiDelayMode::Nop:
    return "nop";
  case rocjitsu::SuperColliderDbiDelayMode::Sleep:
    return "sleep";
  case rocjitsu::SuperColliderDbiDelayMode::SleepVar:
    return "sleep_var";
  }
  return "unknown";
}

[[nodiscard]] const char *check_trap_mode_name(CheckTrapMode mode) {
  switch (mode) {
  case CheckTrapMode::Lds:
    return "lds";
  case CheckTrapMode::Flat:
    return "flat";
  }
  return "unknown";
}

[[nodiscard]] const char *lds_access_kind_name(rocjitsu::SuperColliderDbiLdsAccessKind kind) {
  switch (kind) {
  case rocjitsu::SuperColliderDbiLdsAccessKind::Read:
    return "read";
  case rocjitsu::SuperColliderDbiLdsAccessKind::Write:
    return "write";
  case rocjitsu::SuperColliderDbiLdsAccessKind::Atomic:
    return "atomic";
  case rocjitsu::SuperColliderDbiLdsAccessKind::Other:
    return "other";
  }
  return "unknown";
}

[[nodiscard]] const char *
flat_address_space_hint_name(rocjitsu::SuperColliderDbiFlatAddressSpaceHint hint) {
  switch (hint) {
  case rocjitsu::SuperColliderDbiFlatAddressSpaceHint::Unknown:
    return "unknown";
  case rocjitsu::SuperColliderDbiFlatAddressSpaceHint::Group:
    return "group";
  case rocjitsu::SuperColliderDbiFlatAddressSpaceHint::Private:
    return "private";
  case rocjitsu::SuperColliderDbiFlatAddressSpaceHint::MaybeGroup:
    return "maybe-group";
  case rocjitsu::SuperColliderDbiFlatAddressSpaceHint::MaybePrivate:
    return "maybe-private";
  case rocjitsu::SuperColliderDbiFlatAddressSpaceHint::Global:
    return "global";
  }
  return "unknown";
}

[[nodiscard]] bool ascii_iequals(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto l = static_cast<unsigned char>(lhs[i]);
    const auto r = static_cast<unsigned char>(rhs[i]);
    if (std::tolower(l) != std::tolower(r))
      return false;
  }
  return true;
}

[[nodiscard]] std::optional<bool> parse_bool_value(std::string_view value) {
  if (value == "1" || ascii_iequals(value, "true") || ascii_iequals(value, "on") ||
      ascii_iequals(value, "yes"))
    return true;
  if (value == "0" || ascii_iequals(value, "false") || ascii_iequals(value, "off") ||
      ascii_iequals(value, "no"))
    return false;
  return std::nullopt;
}

[[nodiscard]] bool parse_bool_env(const char *name, bool default_value, bool *out) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    *out = default_value;
    return true;
  }

  auto parsed = parse_bool_value(value);
  if (!parsed) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected boolean\n", name, value);
    return false;
  }
  *out = *parsed;
  return true;
}

[[nodiscard]] bool parse_u32_env(const char *name, uint32_t default_value, uint32_t *out) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    *out = default_value;
    return true;
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected uint32\n", name, value);
    return false;
  }

  *out = static_cast<uint32_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_delay_mode_env(rocjitsu::SuperColliderDbiDelayMode *out) {
  const char *value = std::getenv("RJ_DBI_SC_DELAY_MODE");
  if (value == nullptr || *value == '\0') {
    *out = rocjitsu::SuperColliderDbiDelayMode::Nop;
    return true;
  }
  if (ascii_iequals(value, "nop")) {
    *out = rocjitsu::SuperColliderDbiDelayMode::Nop;
    return true;
  }
  if (ascii_iequals(value, "sleep")) {
    *out = rocjitsu::SuperColliderDbiDelayMode::Sleep;
    return true;
  }
  if (ascii_iequals(value, "sleep_var") || ascii_iequals(value, "sleep-var")) {
    *out = rocjitsu::SuperColliderDbiDelayMode::SleepVar;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_DBI_SC_DELAY_MODE='%s'; "
               "expected nop|sleep|sleep_var\n",
               value);
  return false;
}

[[nodiscard]] bool parse_check_trap_mode_env(CheckTrapMode *out) {
  const char *value = std::getenv("RJ_DBI_SC_CHECK_TRAP_MODE");
  if (value == nullptr || *value == '\0') {
    *out = CheckTrapMode::Lds;
    return true;
  }
  if (ascii_iequals(value, "lds") || ascii_iequals(value, "ds") ||
      ascii_iequals(value, "native-lds")) {
    *out = CheckTrapMode::Lds;
    return true;
  }
  if (ascii_iequals(value, "flat") || ascii_iequals(value, "vflat") ||
      ascii_iequals(value, "generic")) {
    *out = CheckTrapMode::Flat;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_DBI_SC_CHECK_TRAP_MODE='%s'; "
               "expected lds|flat\n",
               value);
  return false;
}

[[nodiscard]] bool parse_log_level(int *out) {
  const char *value = std::getenv("RJ_DBI_LOG");
  if (value == nullptr || *value == '\0') {
    *out = kLogDisabled;
    return true;
  }

  if (auto parsed_bool = parse_bool_value(value)) {
    *out = *parsed_bool ? kLogInfo : kLogDisabled;
    return true;
  }

  errno = 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || errno == ERANGE || parsed < 0) {
    std::fprintf(
        stderr, "[rocjitsu-dbi-hooks] invalid RJ_DBI_LOG='%s'; expected boolean or level\n", value);
    return false;
  }

  *out = parsed > kLogDebug ? kLogDebug : static_cast<int>(parsed);
  return true;
}

[[nodiscard]] bool has_explicit_primary_probe(const HookConfig &config) {
  return config.probe_nop || config.probe_trampoline_nop || config.probe_endpgm ||
         config.probe_lds_endpgm || config.probe_flat_trap;
}

[[nodiscard]] std::optional<HookConfig> parse_config() {
  HookConfig config;
  if (!parse_log_level(&config.log_level))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SUPERCOLLIDER", false, &config.supercollider))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_FAIL_CLOSED", false, &config.fail_closed))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SC_REQUIRE_PATCH", false, &config.require_patch))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SC_PROBE_NOP", false, &config.probe_nop))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SC_PROBE_TRAMPOLINE_NOP", false, &config.probe_trampoline_nop))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SC_PROBE_ENDPGM", false, &config.probe_endpgm))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SC_PROBE_LDS_ENDPGM", false, &config.probe_lds_endpgm))
    return std::nullopt;
  if (!parse_check_trap_mode_env(&config.check_trap_mode))
    return std::nullopt;
  if (!parse_bool_env("RJ_DBI_SC_PROBE_FLAT_TRAP", false, &config.probe_flat_trap))
    return std::nullopt;
  if (!has_explicit_primary_probe(config)) {
    config.probe_lds_check_trap = config.check_trap_mode == CheckTrapMode::Lds;
    config.probe_flat_check_trap = config.check_trap_mode == CheckTrapMode::Flat;
  }
  if (!parse_bool_env("RJ_DBI_SC_FAULT_DROP_BARRIER", false, &config.fault_drop_barrier))
    return std::nullopt;
  if (!parse_u32_env("RJ_DBI_SC_FAULT_BARRIER_INDEX", 0, &config.fault_barrier_index))
    return std::nullopt;
  if (!parse_delay_mode_env(&config.delay_mode))
    return std::nullopt;
  if (!parse_u32_env("RJ_DBI_SC_DELAY", 0, &config.delay_nops))
    return std::nullopt;
  if (!parse_u32_env("RJ_DBI_SC_MAX_PATCHES", 1, &config.max_patches))
    return std::nullopt;
  if (config.max_patches == 0) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid RJ_DBI_SC_MAX_PATCHES='0'; expected >=1\n");
    return std::nullopt;
  }
  uint32_t delay_var_ssrc = 106;
  if (!parse_u32_env("RJ_DBI_SC_DELAY_VAR_SSRC", 106, &delay_var_ssrc))
    return std::nullopt;
  if (delay_var_ssrc > 255) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_DBI_SC_DELAY_VAR_SSRC='%s'; expected 0..255\n",
                 std::getenv("RJ_DBI_SC_DELAY_VAR_SSRC"));
    return std::nullopt;
  }
  config.delay_var_ssrc = static_cast<uint16_t>(delay_var_ssrc);
  if (const char *value = std::getenv("RJ_DBI_DUMP_DIR"); value != nullptr && *value != '\0')
    config.dump_dir = value;
  uint32_t scratch_vgpr = 0;
  if (const char *value = std::getenv("RJ_DBI_SC_TMP_VGPR"); value != nullptr && *value != '\0') {
    if (!parse_u32_env("RJ_DBI_SC_TMP_VGPR", 0, &scratch_vgpr))
      return std::nullopt;
    if (scratch_vgpr > 255) {
      std::fprintf(
          stderr, "[rocjitsu-dbi-hooks] invalid RJ_DBI_SC_TMP_VGPR='%s'; expected 0..255\n", value);
      return std::nullopt;
    }
    config.scratch_vgpr = static_cast<uint16_t>(scratch_vgpr);
  }
  return config;
}

[[nodiscard]] bool require_patch_applies_to(const rocjitsu::SuperColliderDbiResult &result) {
  for (const rocjitsu::SuperColliderDbiKernelInfo &kernel : result.kernels) {
    for (const rocjitsu::SuperColliderDbiLdsSite &site : kernel.lds_sites) {
      if (site.supported_mvp &&
          (site.mnemonic == "ds_load_b32" || site.mnemonic == "ds_load_b64" ||
           site.mnemonic == "ds_load_b128" || site.mnemonic == "ds_load_2addr_b32" ||
           site.mnemonic == "ds_load_2addr_b64" || site.mnemonic == "ds_load_2addr_stride64_b32" ||
           site.mnemonic == "ds_load_2addr_stride64_b64" || site.mnemonic == "ds_load_u16_d16" ||
           site.mnemonic == "ds_load_u16_d16_hi" || site.mnemonic == "ds_store_b32" ||
           site.mnemonic == "ds_store_b64" || site.mnemonic == "ds_store_b128"))
        return true;
    }
  }
  return false;
}

std::mutex &log_mutex() {
  static std::mutex mutex;
  return mutex;
}

void log_message(int required_level, const char *format, ...) {
  if (g_log_level.load(std::memory_order_relaxed) < required_level)
    return;

  std::lock_guard lock(log_mutex());
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] ");

  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);

  std::fprintf(stderr, "\n");
}

void dump_code_object_bytes(const HookConfig &config, uint64_t dump_id, uint64_t reader,
                            std::string_view tag, std::span<const uint8_t> bytes) {
  if (config.dump_dir.empty() || bytes.empty())
    return;

  if (::mkdir(config.dump_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    log_message(kLogInfo, "failed to create RJ_DBI_DUMP_DIR='%s': %s", config.dump_dir.c_str(),
                std::strerror(errno));
    return;
  }

  std::array<char, 4096> path{};
  const int written = std::snprintf(
      path.data(), path.size(), "%s/rj-dbi-%06llu-reader-%llu-%.*s.hsaco", config.dump_dir.c_str(),
      static_cast<unsigned long long>(dump_id), static_cast<unsigned long long>(reader),
      static_cast<int>(tag.size()), tag.data());
  if (written < 0 || static_cast<size_t>(written) >= path.size()) {
    log_message(kLogInfo, "RJ_DBI_DUMP_DIR path is too long: %s", config.dump_dir.c_str());
    return;
  }

  FILE *file = std::fopen(path.data(), "wb");
  if (file == nullptr) {
    log_message(kLogInfo, "failed to open DBI dump '%s': %s", path.data(), std::strerror(errno));
    return;
  }
  const size_t stored = std::fwrite(bytes.data(), 1, bytes.size(), file);
  const int close_status = std::fclose(file);
  if (stored != bytes.size() || close_status != 0) {
    log_message(kLogInfo, "failed to write complete DBI dump '%s'", path.data());
    return;
  }

  log_message(kLogInfo, "dumped DBI %.*s code object reader=%llu bytes=%zu path=%s",
              static_cast<int>(tag.size()), tag.data(), static_cast<unsigned long long>(reader),
              bytes.size(), path.data());
}

/// @brief Process-local map from HSA code-object reader handles to ELF bytes.
///
/// @details `hsa_executable_load_agent_code_object()` receives only an opaque
/// reader handle. The create wrapper records memory-backed reader bytes here so
/// the load wrapper can later hand those bytes to the DBI patcher. Session 2
/// only logs and passes through, but this registry is the Session 3 handoff.
class CodeObjectReaderRegistry {
public:
  static CodeObjectReaderRegistry &instance() {
    static CodeObjectReaderRegistry registry;
    return registry;
  }

  [[nodiscard]] bool store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        entry->bytes = bytes;
        entry->size = size;
        return true;
      }
    }

    void *storage = entry_pool_.try_allocate(sizeof(Entry));
    if (storage == nullptr)
      return false;
    auto *entry = new (storage) Entry(reader.handle, bytes, size);
    entries_.push_front(*entry);
    return true;
  }

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
    Entry(uint64_t h, const uint8_t *b, size_t s) : handle(h), bytes(b), size(s) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
  };

  void destroy_entry(Entry *entry) {
    entry->~Entry();
    entry_pool_.deallocate(entry);
  }

  mutable std::shared_mutex mutex_;
  util::ArenaAlloc<sizeof(Entry), 256, alignof(Entry)> entry_pool_;
  util::IntrusiveList<Entry> entries_;
};

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);
hsa_status_t HSA_API rj_dbi_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);

class RjDbiHsaLayer {
public:
  bool install(HsaApiTable *table, HookConfig config) {
    std::lock_guard lock(mutex_);
    if (active_) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] OnLoad called while hook is already active\n");
      return false;
    }
    if (!validate_table(table))
      return false;

    table_ = table;
    core_ = table->core_;
    g_log_level.store(config.log_level, std::memory_order_relaxed);
    config_ = config;
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_load_agent_code_object_ = core_->hsa_executable_load_agent_code_object_fn;

    if (original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_load_agent_code_object_ == nullptr) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] HSA core table contains null code-object entries\n");
      clear_unlocked();
      return false;
    }

    core_->hsa_code_object_reader_create_from_file_fn = rj_dbi_code_object_reader_create_from_file;
    core_->hsa_code_object_reader_create_from_memory_fn =
        rj_dbi_code_object_reader_create_from_memory;
    core_->hsa_code_object_reader_destroy_fn = rj_dbi_code_object_reader_destroy;
    core_->hsa_executable_load_agent_code_object_fn = rj_dbi_executable_load_agent_code_object;
    active_ = true;

    log_message(
        kLogInfo,
        "installed SuperCollider DBI hook delay_nops=%u fail_closed=%s require_patch=%s "
        "probe_nop=%s probe_trampoline_nop=%s probe_endpgm=%s probe_lds_endpgm=%s "
        "check_trap_mode=%s probe_lds_check_trap=%s probe_flat_check_trap=%s probe_flat_trap=%s "
        "fault_drop_barrier=%s fault_barrier_index=%u delay_mode=%s delay_var_ssrc=%u "
        "max_patches=%u tmp_vgpr=%s mode=%s",
        config.delay_nops, config.fail_closed ? "true" : "false",
        config.require_patch ? "true" : "false", config.probe_nop ? "true" : "false",
        config.probe_trampoline_nop ? "true" : "false", config.probe_endpgm ? "true" : "false",
        config.probe_lds_endpgm ? "true" : "false", check_trap_mode_name(config.check_trap_mode),
        config.probe_lds_check_trap ? "true" : "false",
        config.probe_flat_check_trap ? "true" : "false", config.probe_flat_trap ? "true" : "false",
        config.fault_drop_barrier ? "true" : "false", config.fault_barrier_index,
        delay_mode_name(config.delay_mode), config.delay_var_ssrc, config.max_patches,
        config.scratch_vgpr ? std::to_string(*config.scratch_vgpr).c_str() : "auto",
        config.fault_drop_barrier
            ? (config.probe_lds_check_trap    ? "proof-lds-check-trap+fault-drop-barrier"
               : config.probe_flat_check_trap ? "proof-flat-check-trap+fault-drop-barrier"
                                              : "fault-drop-barrier")
        : config.probe_lds_check_trap ? "proof-lds-check-trap"
        : config.probe_flat_check_trap
            ? "proof-flat-check-trap"
            : (config.probe_flat_trap
                   ? "proof-flat-trap"
                   : (config.probe_lds_endpgm
                          ? "proof-lds-endpgm"
                          : (config.probe_endpgm
                                 ? "proof-endpgm"
                                 : (config.probe_trampoline_nop
                                        ? "proof-trampoline-nop"
                                        : (config.probe_nop ? "proof-nop" : "pass-through"))))));
    if (!config.dump_dir.empty())
      log_message(kLogInfo, "DBI code-object dumps enabled dir=%s", config.dump_dir.c_str());
    return true;
  }

  void uninstall() {
    std::lock_guard lock(mutex_);
    if (active_ && core_ != nullptr) {
      if (core_->hsa_code_object_reader_create_from_file_fn ==
          rj_dbi_code_object_reader_create_from_file)
        core_->hsa_code_object_reader_create_from_file_fn = original_create_from_file_;
      if (core_->hsa_code_object_reader_create_from_memory_fn ==
          rj_dbi_code_object_reader_create_from_memory)
        core_->hsa_code_object_reader_create_from_memory_fn = original_create_from_memory_;
      if (core_->hsa_code_object_reader_destroy_fn == rj_dbi_code_object_reader_destroy)
        core_->hsa_code_object_reader_destroy_fn = original_destroy_;
      if (core_->hsa_executable_load_agent_code_object_fn ==
          rj_dbi_executable_load_agent_code_object)
        core_->hsa_executable_load_agent_code_object_fn = original_load_agent_code_object_;
    }

    CodeObjectReaderRegistry::instance().clear();
    clear_unlocked();
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
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid HSA API table passed to OnLoad\n");
      return false;
    }

    constexpr size_t required_size =
        offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
        sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn);
    if (table->core_->version.minor_id < required_size) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] HSA core table too small: got %u bytes, need %zu bytes\n",
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

RjDbiHsaLayer &layer() {
  static RjDbiHsaLayer state;
  return state;
}

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    if (!CodeObjectReaderRegistry::instance().store(
            *code_object_reader, static_cast<const uint8_t *>(code_object), size)) {
      if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
        (void)original_destroy(*code_object_reader);
      *code_object_reader = {};
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] failed to track memory-backed code-object reader\n");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    log_message(kLogDebug, "registered memory reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), size);
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    log_message(kLogVerbose, "file-backed reader=%llu will pass through unchanged",
                static_cast<unsigned long long>(code_object_reader->handle));
  }
  return status;
}

hsa_status_t HSA_API
rj_dbi_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  CodeObjectReaderRegistry::instance().remove(code_object_reader);

  auto *original = layer().destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  return original(code_object_reader);
}

hsa_status_t HSA_API rj_dbi_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original_load = layer().load_agent_code_object();
  if (original_load == nullptr)
    return HSA_STATUS_ERROR;

  auto config = layer().config();
  if (!config) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] DBI hook layer is inactive during load\n");
    return HSA_STATUS_ERROR;
  }

  hsa_code_object_reader_t reader_to_load = code_object_reader;
  hsa_code_object_reader_t replacement_reader{};
  bool using_replacement_reader = false;
  std::optional<rocjitsu::SuperColliderDbiResult> patch_result_storage;

  const uint8_t *bytes = nullptr;
  size_t size = 0;
  if (CodeObjectReaderRegistry::instance().lookup(code_object_reader, &bytes, &size)) {
    const uint64_t dump_id =
        config->dump_dir.empty() ? 0 : g_dump_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    dump_code_object_bytes(*config, dump_id, code_object_reader.handle, "original",
                           std::span<const uint8_t>(bytes, size));

    rocjitsu::SuperColliderDbiOptions patch_options;
    patch_options.enabled = config->supercollider;
    patch_options.fail_closed = config->fail_closed;
    patch_options.probe_nop = config->probe_nop;
    patch_options.probe_trampoline_nop = config->probe_trampoline_nop;
    patch_options.probe_endpgm = config->probe_endpgm;
    patch_options.probe_lds_endpgm = config->probe_lds_endpgm;
    patch_options.probe_lds_check_trap = config->probe_lds_check_trap;
    patch_options.probe_flat_check_trap = config->probe_flat_check_trap;
    patch_options.probe_flat_trap = config->probe_flat_trap;
    patch_options.fault_drop_barrier = config->fault_drop_barrier;
    patch_options.fault_barrier_index = config->fault_barrier_index;
    patch_options.delay_mode = config->delay_mode;
    patch_options.delay_var_ssrc = config->delay_var_ssrc;
    patch_options.scratch_vgpr = config->scratch_vgpr;
    patch_options.delay_nops = config->delay_nops;
    patch_options.max_patches = config->max_patches;

    log_message(kLogInfo, "SuperCollider DBI patch begin reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader.handle), size);
    patch_result_storage =
        rocjitsu::try_patch_supercollider_dbi(std::span<const uint8_t>(bytes, size), patch_options);
    const rocjitsu::SuperColliderDbiResult &patch_result = *patch_result_storage;
    log_message(kLogInfo,
                "SuperCollider DBI patch end reader=%llu visited=%s modified=%s errors=%zu "
                "warnings=%zu patches=%zu",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.visited_code_object ? "true" : "false",
                patch_result.modified ? "true" : "false", patch_result.errors.size(),
                patch_result.warnings.size(), patch_result.patches.size());

    if (!patch_result.errors.empty()) {
      for (const std::string &error : patch_result.errors)
        std::fprintf(stderr, "[rocjitsu-dbi-hooks] %s\n", error.c_str());
      if (config->fail_closed)
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    for (const std::string &warning : patch_result.warnings)
      log_message(kLogInfo, "%s", warning.c_str());

    log_message(
        kLogInfo,
        "SuperCollider DBI inventory reader=%llu bytes=%zu visited=%s modified=%s "
        "delay_nops=%u fail_closed=%s probe_nop=%s probe_trampoline_nop=%s "
        "probe_endpgm=%s probe_lds_endpgm=%s check_trap_mode=%s probe_lds_check_trap=%s "
        "probe_flat_check_trap=%s probe_flat_trap=%s fault_drop_barrier=%s "
        "fault_barrier_index=%u delay_mode=%s delay_var_ssrc=%u max_patches=%u tmp_vgpr=%s "
        "require_patch=%s",
        static_cast<unsigned long long>(code_object_reader.handle), patch_result.input_size,
        patch_result.visited_code_object ? "true" : "false",
        patch_result.modified ? "true" : "false", config->delay_nops,
        config->fail_closed ? "true" : "false", config->probe_nop ? "true" : "false",
        config->probe_trampoline_nop ? "true" : "false", config->probe_endpgm ? "true" : "false",
        config->probe_lds_endpgm ? "true" : "false", check_trap_mode_name(config->check_trap_mode),
        config->probe_lds_check_trap ? "true" : "false",
        config->probe_flat_check_trap ? "true" : "false",
        config->probe_flat_trap ? "true" : "false", config->fault_drop_barrier ? "true" : "false",
        config->fault_barrier_index, delay_mode_name(config->delay_mode), config->delay_var_ssrc,
        config->max_patches,
        config->scratch_vgpr ? std::to_string(*config->scratch_vgpr).c_str() : "auto",
        config->require_patch ? "true" : "false");
    if (!patch_result.target_name.empty()) {
      log_message(kLogInfo,
                  "SuperCollider DBI code-object reader=%llu target=%s arch=%s text_sections=%zu "
                  "kernels=%zu functions=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result.target_name.c_str(), patch_result.arch_name.c_str(),
                  patch_result.text_sections.size(), patch_result.kernels.size(),
                  patch_result.functions.size());
    }
    size_t candidate_kernel_count = 0;
    size_t skipped_kernel_count = 0;
    size_t rejected_kernel_count = 0;
    size_t supported_lds_site_count = 0;
    size_t flat_site_count = 0;
    size_t flat_group_hint_count = 0;
    size_t flat_private_hint_count = 0;
    size_t flat_maybe_group_hint_count = 0;
    size_t flat_maybe_private_hint_count = 0;
    size_t flat_global_hint_count = 0;
    size_t flat_unknown_hint_count = 0;
    size_t function_lds_site_count = 0;
    size_t function_supported_lds_site_count = 0;
    size_t function_flat_site_count = 0;
    size_t function_flat_group_hint_count = 0;
    size_t function_flat_private_hint_count = 0;
    size_t function_flat_maybe_group_hint_count = 0;
    size_t function_flat_maybe_private_hint_count = 0;
    size_t function_flat_global_hint_count = 0;
    size_t function_flat_unknown_hint_count = 0;
    for (const rocjitsu::SuperColliderDbiKernelInfo &kernel : patch_result.kernels) {
      switch (kernel.preflight_action) {
      case rocjitsu::SuperColliderDbiPreflightAction::Candidate:
        ++candidate_kernel_count;
        break;
      case rocjitsu::SuperColliderDbiPreflightAction::Skip:
        ++skipped_kernel_count;
        break;
      case rocjitsu::SuperColliderDbiPreflightAction::Reject:
        ++rejected_kernel_count;
        break;
      case rocjitsu::SuperColliderDbiPreflightAction::NotRun:
        break;
      }
      for (const rocjitsu::SuperColliderDbiLdsSite &site : kernel.lds_sites) {
        if (site.supported_mvp)
          ++supported_lds_site_count;
      }
      flat_site_count += kernel.flat_sites.size();
      flat_group_hint_count += kernel.stats.flat_group_hint_count;
      flat_private_hint_count += kernel.stats.flat_private_hint_count;
      flat_maybe_group_hint_count += kernel.stats.flat_maybe_group_hint_count;
      flat_maybe_private_hint_count += kernel.stats.flat_maybe_private_hint_count;
      flat_global_hint_count += kernel.stats.flat_global_hint_count;
      flat_unknown_hint_count += kernel.stats.flat_unknown_hint_count;
    }
    for (const rocjitsu::SuperColliderDbiFunctionInfo &function : patch_result.functions) {
      function_lds_site_count += function.lds_sites.size();
      for (const rocjitsu::SuperColliderDbiLdsSite &site : function.lds_sites) {
        if (site.supported_mvp)
          ++function_supported_lds_site_count;
      }
      function_flat_site_count += function.flat_sites.size();
      function_flat_group_hint_count += function.stats.flat_group_hint_count;
      function_flat_private_hint_count += function.stats.flat_private_hint_count;
      function_flat_maybe_group_hint_count += function.stats.flat_maybe_group_hint_count;
      function_flat_maybe_private_hint_count += function.stats.flat_maybe_private_hint_count;
      function_flat_global_hint_count += function.stats.flat_global_hint_count;
      function_flat_unknown_hint_count += function.stats.flat_unknown_hint_count;
    }
    log_message(kLogInfo,
                "SuperCollider DBI summary reader=%llu kernels=%zu candidates=%zu skips=%zu "
                "rejects=%zu supported_lds_sites=%zu flat_sites=%zu flat_group_hints=%zu "
                "flat_private_hints=%zu flat_maybe_group_hints=%zu "
                "flat_maybe_private_hints=%zu flat_global_hints=%zu "
                "flat_unknown_hints=%zu functions=%zu function_lds_sites=%zu "
                "function_supported_lds_sites=%zu function_flat_sites=%zu "
                "function_flat_group_hints=%zu function_flat_private_hints=%zu "
                "function_flat_maybe_group_hints=%zu function_flat_maybe_private_hints=%zu "
                "function_flat_global_hints=%zu function_flat_unknown_hints=%zu patches=%zu "
                "modified=%s",
                static_cast<unsigned long long>(code_object_reader.handle),
                patch_result.kernels.size(), candidate_kernel_count, skipped_kernel_count,
                rejected_kernel_count, supported_lds_site_count, flat_site_count,
                flat_group_hint_count, flat_private_hint_count, flat_maybe_group_hint_count,
                flat_maybe_private_hint_count, flat_global_hint_count, flat_unknown_hint_count,
                patch_result.functions.size(), function_lds_site_count,
                function_supported_lds_site_count, function_flat_site_count,
                function_flat_group_hint_count, function_flat_private_hint_count,
                function_flat_maybe_group_hint_count, function_flat_maybe_private_hint_count,
                function_flat_global_hint_count, function_flat_unknown_hint_count,
                patch_result.patches.size(), patch_result.modified ? "true" : "false");
    for (const rocjitsu::SuperColliderDbiTextSection &text : patch_result.text_sections) {
      log_message(kLogVerbose,
                  "SuperCollider DBI text reader=%llu name=%s file=0x%llx vaddr=0x%llx size=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle), text.name.c_str(),
                  static_cast<unsigned long long>(text.file_offset),
                  static_cast<unsigned long long>(text.virtual_address),
                  static_cast<unsigned long long>(text.size));
    }
    for (const rocjitsu::SuperColliderDbiKernelInfo &kernel : patch_result.kernels) {
      if (kernel.has_text_range) {
        log_message(kLogInfo,
                    "SuperCollider DBI kernel reader=%llu name=%s kd_file=0x%llx "
                    "text_file=0x%llx entry_text=0x%llx code_size=%llu decoded=%s "
                    "insts=%llu lds_reads=%llu lds_writes=%llu lds_atomics=%llu ds_other=%llu "
                    "flat_reads=%llu flat_writes=%llu flat_atomics=%llu flat_group_hints=%llu "
                    "flat_private_hints=%llu flat_maybe_group_hints=%llu "
                    "flat_maybe_private_hints=%llu flat_global_hints=%llu "
                    "flat_unknown_hints=%llu global_mem=%llu scratch_mem=%llu barriers=%llu "
                    "waits=%llu fences=%llu decode_errors=%llu "
                    "preflight=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    static_cast<unsigned long long>(kernel.descriptor_file_offset),
                    static_cast<unsigned long long>(kernel.text_file_offset),
                    static_cast<unsigned long long>(kernel.entry_text_offset),
                    static_cast<unsigned long long>(kernel.code_size),
                    kernel.decoded ? "true" : "false",
                    static_cast<unsigned long long>(kernel.stats.instruction_count),
                    static_cast<unsigned long long>(kernel.stats.lds_read_count),
                    static_cast<unsigned long long>(kernel.stats.lds_write_count),
                    static_cast<unsigned long long>(kernel.stats.lds_atomic_count),
                    static_cast<unsigned long long>(kernel.stats.ds_other_count),
                    static_cast<unsigned long long>(kernel.stats.flat_read_count),
                    static_cast<unsigned long long>(kernel.stats.flat_write_count),
                    static_cast<unsigned long long>(kernel.stats.flat_atomic_count),
                    static_cast<unsigned long long>(kernel.stats.flat_group_hint_count),
                    static_cast<unsigned long long>(kernel.stats.flat_private_hint_count),
                    static_cast<unsigned long long>(kernel.stats.flat_maybe_group_hint_count),
                    static_cast<unsigned long long>(kernel.stats.flat_maybe_private_hint_count),
                    static_cast<unsigned long long>(kernel.stats.flat_global_hint_count),
                    static_cast<unsigned long long>(kernel.stats.flat_unknown_hint_count),
                    static_cast<unsigned long long>(kernel.stats.global_memory_count),
                    static_cast<unsigned long long>(kernel.stats.scratch_memory_count),
                    static_cast<unsigned long long>(kernel.stats.barrier_count),
                    static_cast<unsigned long long>(kernel.stats.wait_count),
                    static_cast<unsigned long long>(kernel.stats.fence_like_count),
                    static_cast<unsigned long long>(kernel.stats.decode_error_count),
                    preflight_action_name(kernel.preflight_action));
      } else {
        log_message(kLogInfo,
                    "SuperCollider DBI kernel reader=%llu name=%s kd_file=0x%llx "
                    "text_range=unavailable decoded=%s decode_errors=%llu preflight=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    static_cast<unsigned long long>(kernel.descriptor_file_offset),
                    kernel.decoded ? "true" : "false",
                    static_cast<unsigned long long>(kernel.stats.decode_error_count),
                    preflight_action_name(kernel.preflight_action));
      }
      for (const std::string &reason : kernel.preflight_reasons) {
        log_message(kLogInfo,
                    "SuperCollider DBI preflight reader=%llu kernel=%s action=%s reason=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    preflight_action_name(kernel.preflight_action), reason.c_str());
      }
      for (const rocjitsu::SuperColliderDbiLdsSite &site : kernel.lds_sites) {
        log_message(kLogVerbose,
                    "SuperCollider DBI lds-site reader=%llu kernel=%s kind=%s supported=%s "
                    "mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    lds_access_kind_name(site.kind), site.supported_mvp ? "true" : "false",
                    site.mnemonic.c_str(), static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-");
      }
      for (const rocjitsu::SuperColliderDbiFlatSite &site : kernel.flat_sites) {
        log_message(kLogVerbose,
                    "SuperCollider DBI flat-site reader=%llu kernel=%s kind=%s "
                    "mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s addr_hint=%s raw_saddr=%s "
                    "raw_vaddr=%s raw_vsrc=%s raw_vdst=%s raw_ioffset=%s "
                    "raw_scope=%s raw_th=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    lds_access_kind_name(site.kind), site.mnemonic.c_str(),
                    static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-",
                    flat_address_space_hint_name(site.address_space_hint),
                    site.raw_saddr ? std::to_string(*site.raw_saddr).c_str() : "-",
                    site.raw_vaddr ? std::to_string(*site.raw_vaddr).c_str() : "-",
                    site.raw_vsrc ? std::to_string(*site.raw_vsrc).c_str() : "-",
                    site.raw_vdst ? std::to_string(*site.raw_vdst).c_str() : "-",
                    site.raw_ioffset ? std::to_string(*site.raw_ioffset).c_str() : "-",
                    site.raw_scope ? std::to_string(*site.raw_scope).c_str() : "-",
                    site.raw_th ? std::to_string(*site.raw_th).c_str() : "-");
      }
    }
    for (const rocjitsu::SuperColliderDbiFunctionInfo &function : patch_result.functions) {
      log_message(kLogVerbose,
                  "SuperCollider DBI function reader=%llu name=%s text_file=0x%llx "
                  "entry_text=0x%llx code_size=%llu decoded=%s insts=%llu lds_reads=%llu "
                  "lds_writes=%llu lds_atomics=%llu ds_other=%llu flat_reads=%llu "
                  "flat_writes=%llu flat_atomics=%llu flat_group_hints=%llu "
                  "flat_private_hints=%llu flat_maybe_group_hints=%llu "
                  "flat_maybe_private_hints=%llu flat_global_hints=%llu "
                  "flat_unknown_hints=%llu global_mem=%llu scratch_mem=%llu barriers=%llu "
                  "waits=%llu fences=%llu decode_errors=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle), function.name.c_str(),
                  static_cast<unsigned long long>(function.text_file_offset),
                  static_cast<unsigned long long>(function.entry_text_offset),
                  static_cast<unsigned long long>(function.code_size),
                  function.decoded ? "true" : "false",
                  static_cast<unsigned long long>(function.stats.instruction_count),
                  static_cast<unsigned long long>(function.stats.lds_read_count),
                  static_cast<unsigned long long>(function.stats.lds_write_count),
                  static_cast<unsigned long long>(function.stats.lds_atomic_count),
                  static_cast<unsigned long long>(function.stats.ds_other_count),
                  static_cast<unsigned long long>(function.stats.flat_read_count),
                  static_cast<unsigned long long>(function.stats.flat_write_count),
                  static_cast<unsigned long long>(function.stats.flat_atomic_count),
                  static_cast<unsigned long long>(function.stats.flat_group_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_private_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_maybe_group_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_maybe_private_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_global_hint_count),
                  static_cast<unsigned long long>(function.stats.flat_unknown_hint_count),
                  static_cast<unsigned long long>(function.stats.global_memory_count),
                  static_cast<unsigned long long>(function.stats.scratch_memory_count),
                  static_cast<unsigned long long>(function.stats.barrier_count),
                  static_cast<unsigned long long>(function.stats.wait_count),
                  static_cast<unsigned long long>(function.stats.fence_like_count),
                  static_cast<unsigned long long>(function.stats.decode_error_count));
      for (const rocjitsu::SuperColliderDbiLdsSite &site : function.lds_sites) {
        log_message(kLogVerbose,
                    "SuperCollider DBI function-lds-site reader=%llu function=%s kind=%s "
                    "supported=%s mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    function.name.c_str(), lds_access_kind_name(site.kind),
                    site.supported_mvp ? "true" : "false", site.mnemonic.c_str(),
                    static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-");
      }
      for (const rocjitsu::SuperColliderDbiFlatSite &site : function.flat_sites) {
        log_message(kLogVerbose,
                    "SuperCollider DBI function-flat-site reader=%llu function=%s kind=%s "
                    "mnemonic=%s text=0x%llx file=0x%llx size=%u width=%u "
                    "dst_vgpr=%s addr_vgpr=%s data_vgpr=%s addr_hint=%s raw_saddr=%s "
                    "raw_vaddr=%s raw_vsrc=%s raw_vdst=%s raw_ioffset=%s raw_scope=%s "
                    "raw_th=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    function.name.c_str(), lds_access_kind_name(site.kind), site.mnemonic.c_str(),
                    static_cast<unsigned long long>(site.text_offset),
                    static_cast<unsigned long long>(site.file_offset), site.size, site.width_bits,
                    site.dst_vgpr ? std::to_string(*site.dst_vgpr).c_str() : "-",
                    site.addr_vgpr ? std::to_string(*site.addr_vgpr).c_str() : "-",
                    site.data_vgpr ? std::to_string(*site.data_vgpr).c_str() : "-",
                    flat_address_space_hint_name(site.address_space_hint),
                    site.raw_saddr ? std::to_string(*site.raw_saddr).c_str() : "-",
                    site.raw_vaddr ? std::to_string(*site.raw_vaddr).c_str() : "-",
                    site.raw_vsrc ? std::to_string(*site.raw_vsrc).c_str() : "-",
                    site.raw_vdst ? std::to_string(*site.raw_vdst).c_str() : "-",
                    site.raw_ioffset ? std::to_string(*site.raw_ioffset).c_str() : "-",
                    site.raw_scope ? std::to_string(*site.raw_scope).c_str() : "-",
                    site.raw_th ? std::to_string(*site.raw_th).c_str() : "-");
      }
    }
    for (const rocjitsu::SuperColliderDbiPatchInfo &patch : patch_result.patches) {
      const std::string scratch_vgpr =
          patch.scratch_vgpr ? std::to_string(*patch.scratch_vgpr) : "-";
      log_message(kLogInfo,
                  "SuperCollider DBI proof patch reader=%llu kind=%s anchor=0x%llx "
                  "trampoline=0x%llx original_size=%u scratch_vgpr=%s",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_kind_name(patch.kind), static_cast<unsigned long long>(patch.anchor_offset),
                  static_cast<unsigned long long>(patch.trampoline_offset), patch.original_size,
                  scratch_vgpr.c_str());
    }
    if (config->require_patch && !patch_result.modified && require_patch_applies_to(patch_result)) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_DBI_SC_REQUIRE_PATCH requested, but no patch was "
                   "applied to a code object with supported LDS sites\n");
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    if (patch_result.modified && !patch_result.elf_bytes.empty()) {
      dump_code_object_bytes(
          *config, dump_id, code_object_reader.handle, "patched",
          std::span<const uint8_t>(patch_result.elf_bytes.data(), patch_result.elf_bytes.size()));
    }
  } else {
    log_message(kLogInfo, "SuperCollider DBI pass-through reader=%llu bytes=unavailable",
                static_cast<unsigned long long>(code_object_reader.handle));
  }

  if (patch_result_storage && patch_result_storage->modified &&
      !patch_result_storage->elf_bytes.empty()) {
    auto *original_create = layer().create_from_memory();
    if (original_create == nullptr)
      return HSA_STATUS_ERROR;

    const hsa_status_t reader_status =
        original_create(patch_result_storage->elf_bytes.data(),
                        patch_result_storage->elf_bytes.size(), &replacement_reader);
    if (reader_status != HSA_STATUS_SUCCESS) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] failed to create replacement patched reader: %d\n",
                   static_cast<int>(reader_status));
      if (config->fail_closed)
        return reader_status;
    } else {
      reader_to_load = replacement_reader;
      using_replacement_reader = true;
      log_message(kLogInfo,
                  "SuperCollider DBI replacement reader=%llu original_reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(replacement_reader.handle),
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result_storage->elf_bytes.size());
    }
  }

  const hsa_status_t load_status =
      original_load(executable, agent, reader_to_load, options, loaded_code_object);
  if (using_replacement_reader) {
    auto *original_destroy = layer().destroy();
    if (original_destroy != nullptr)
      (void)original_destroy(replacement_reader);
  }
  return load_status;
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;

  auto config = parse_config();
  if (!config)
    return false;

  g_log_level.store(config->log_level, std::memory_order_relaxed);
  if (!config->supercollider) {
    log_message(kLogInfo, "RJ_DBI_SUPERCOLLIDER is disabled; not installing wrappers");
    return true;
  }

  return layer().install(table, *config);
}

extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
