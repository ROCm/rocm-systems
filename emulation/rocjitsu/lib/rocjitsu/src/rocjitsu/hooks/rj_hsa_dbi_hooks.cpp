// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbi_hooks.cpp
/// @brief HSA tools load-time hook for opt-in rocJITsu DBI instrumentation.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. This initial DBI hook only parses configuration, installs the
/// code-object reader/load wrappers, logs observed loads when requested, and
/// routes memory-backed reader bytes through the selected ConSan DBI flavor.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/patch/consan.h"
#include "rocjitsu/code/patch/consan_moi.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"

#include <algorithm>
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
  All,
  Lds,
  Flat,
};

std::atomic<int> g_log_level{kLogDisabled};
std::atomic<uint64_t> g_dump_sequence{0};

struct HookConfig {
  std::optional<rocjitsu::ConSanFlavor> flavor;
  rocjitsu::ConSanMoiEngine moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
  rocjitsu::ConSanMoiOwnerSource moi_owner_source = rocjitsu::ConSanMoiOwnerSource::WorkitemId;
  bool fail_closed = false;
  bool require_patch = false;
  bool probe_nop = false;
  bool probe_trampoline_nop = false;
  bool probe_endpgm = false;
  bool probe_lds_endpgm = false;
  CheckTrapMode check_trap_mode = CheckTrapMode::All;
  bool probe_lds_check_trap = false;
  bool probe_flat_check_trap = false;
  bool probe_flat_trap = false;
  bool fault_drop_barrier = false;
  bool moi_init_owner_epoch = false;
  bool moi_track_barriers = false;
  bool moi_track_atomics = false;
  bool moi_dynamic_access_records = false;
  bool test_force_vgpr_spill = false;
  std::string test_kernel_name_filter;
  bool moi_require_records = false;
  bool moi_require_diagnostics = false;
  bool moi_forbid_diagnostics = false;
  bool moi_require_replay_conflict = false;
  uint32_t fault_barrier_index = 0;
  rocjitsu::ConSanDelayMode delay_mode = rocjitsu::ConSanDelayMode::Nop;
  uint16_t delay_var_ssrc = 106;
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  std::optional<uint16_t> moi_owner_sgpr;
  std::optional<uint16_t> moi_owner_vgpr;
  std::optional<uint16_t> moi_epoch_vgpr;
  std::optional<uint64_t> report_buffer_address;
  std::optional<uint64_t> moi_report_buffer_address;
  uint64_t moi_report_buffer_size = 0;
  uint64_t moi_auto_report_buffer_size = 0;
  uint32_t delay_nops = 0;
  uint32_t max_patches = 1;
  uint32_t moi_sample_stride = 1;
  uint32_t moi_sample_offset = 0;
  uint32_t report_marker = 1;
  int log_level = kLogDisabled;
  std::string dump_dir;
};

[[nodiscard]] const char *preflight_action_name(rocjitsu::ConSanPreflightAction action) {
  switch (action) {
  case rocjitsu::ConSanPreflightAction::NotRun:
    return "not-run";
  case rocjitsu::ConSanPreflightAction::Candidate:
    return "candidate";
  case rocjitsu::ConSanPreflightAction::Skip:
    return "skip";
  case rocjitsu::ConSanPreflightAction::Reject:
    return "reject";
  }
  return "unknown";
}

[[nodiscard]] const char *patch_kind_name(rocjitsu::ConSanPatchKind kind) {
  switch (kind) {
  case rocjitsu::ConSanPatchKind::InlineNopRewrite:
    return "inline-nop-rewrite";
  case rocjitsu::ConSanPatchKind::InlineEndpgmRewrite:
    return "inline-endpgm-rewrite";
  case rocjitsu::ConSanPatchKind::InlineLdsEndpgmRewrite:
    return "inline-lds-endpgm-rewrite";
  case rocjitsu::ConSanPatchKind::InlineLdsLoadCheckTrap:
    return "inline-lds-load-check-trap";
  case rocjitsu::ConSanPatchKind::InlineLdsStoreCheckTrap:
    return "inline-lds-store-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
    return "local-cave-lds-load-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
    return "local-cave-lds-store-check-trap";
  case rocjitsu::ConSanPatchKind::InlineFlatLoadCheckTrap:
    return "inline-flat-load-check-trap";
  case rocjitsu::ConSanPatchKind::InlineFlatStoreCheckTrap:
    return "inline-flat-store-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
    return "local-cave-flat-load-check-trap";
  case rocjitsu::ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
    return "local-cave-flat-store-check-trap";
  case rocjitsu::ConSanPatchKind::InlineFlatTrapRewrite:
    return "inline-flat-trap-rewrite";
  case rocjitsu::ConSanPatchKind::InlineBarrierNopRewrite:
    return "inline-barrier-nop-rewrite";
  case rocjitsu::ConSanPatchKind::InlineMoiAccessRecordStore:
    return "inline-moi-access-record-store";
  case rocjitsu::ConSanPatchKind::TrampolineMoiAccessRecordStore:
    return "trampoline-moi-access-record-store";
  case rocjitsu::ConSanPatchKind::InlineMoiExactShadowStore:
    return "inline-moi-exact-shadow-store";
  case rocjitsu::ConSanPatchKind::TrampolineMoiExactShadowStore:
    return "trampoline-moi-exact-shadow-store";
  case rocjitsu::ConSanPatchKind::InlineMoiSampledWatchpointStore:
    return "inline-moi-sampled-watchpoint-store";
  case rocjitsu::ConSanPatchKind::TrampolineMoiSampledWatchpointStore:
    return "trampoline-moi-sampled-watchpoint-store";
  case rocjitsu::ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue:
    return "kernel-entry-moi-owner-epoch-prologue";
  case rocjitsu::ConSanPatchKind::TrampolineMoiBarrierRecord:
    return "trampoline-moi-barrier-record";
  case rocjitsu::ConSanPatchKind::TrampolineMoiInlineEpochBarrier:
    return "trampoline-moi-inline-epoch-barrier";
  case rocjitsu::ConSanPatchKind::TrampolineMoiInlineAtomicOrdering:
    return "trampoline-moi-inline-atomic-ordering";
  case rocjitsu::ConSanPatchKind::TrampolineMoiAtomicRecord:
    return "trampoline-moi-atomic-record";
  case rocjitsu::ConSanPatchKind::TrampolineNop:
    return "trampoline-nop";
  }
  return "unknown";
}

[[nodiscard]] const char *moi_candidate_source_name(rocjitsu::ConSanMoiCandidateSource source) {
  switch (source) {
  case rocjitsu::ConSanMoiCandidateSource::NativeLds:
    return "native-lds";
  case rocjitsu::ConSanMoiCandidateSource::FlatGroup:
    return "flat-group";
  case rocjitsu::ConSanMoiCandidateSource::FlatMaybeGroup:
    return "flat-maybe-group";
  }
  return "unknown";
}

[[nodiscard]] const char *delay_mode_name(rocjitsu::ConSanDelayMode mode) {
  switch (mode) {
  case rocjitsu::ConSanDelayMode::Nop:
    return "nop";
  case rocjitsu::ConSanDelayMode::Sleep:
    return "sleep";
  case rocjitsu::ConSanDelayMode::SleepVar:
    return "sleep_var";
  }
  return "unknown";
}

[[nodiscard]] const char *owner_source_name(rocjitsu::ConSanMoiOwnerSource source) {
  switch (source) {
  case rocjitsu::ConSanMoiOwnerSource::WorkitemId:
    return "workitem_id";
  case rocjitsu::ConSanMoiOwnerSource::HwId:
    return "hw_id";
  }
  return "unknown";
}

[[nodiscard]] const char *flavor_name(rocjitsu::ConSanFlavor flavor) {
  return rocjitsu::consan_flavor_name(flavor);
}

[[nodiscard]] const char *check_trap_mode_name(CheckTrapMode mode) {
  switch (mode) {
  case CheckTrapMode::All:
    return "all";
  case CheckTrapMode::Lds:
    return "lds";
  case CheckTrapMode::Flat:
    return "flat";
  }
  return "unknown";
}

[[nodiscard]] const char *lds_access_kind_name(rocjitsu::ConSanLdsAccessKind kind) {
  switch (kind) {
  case rocjitsu::ConSanLdsAccessKind::Read:
    return "read";
  case rocjitsu::ConSanLdsAccessKind::Write:
    return "write";
  case rocjitsu::ConSanLdsAccessKind::Atomic:
    return "atomic";
  case rocjitsu::ConSanLdsAccessKind::Other:
    return "other";
  }
  return "unknown";
}

[[nodiscard]] const char *flat_address_space_hint_name(rocjitsu::ConSanFlatAddressSpaceHint hint) {
  switch (hint) {
  case rocjitsu::ConSanFlatAddressSpaceHint::Unknown:
    return "unknown";
  case rocjitsu::ConSanFlatAddressSpaceHint::Group:
    return "group";
  case rocjitsu::ConSanFlatAddressSpaceHint::Private:
    return "private";
  case rocjitsu::ConSanFlatAddressSpaceHint::MaybeGroup:
    return "maybe-group";
  case rocjitsu::ConSanFlatAddressSpaceHint::MaybePrivate:
    return "maybe-private";
  case rocjitsu::ConSanFlatAddressSpaceHint::Global:
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

[[nodiscard]] bool parse_optional_vgpr_env(const char *name, std::optional<uint16_t> *out) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    out->reset();
    return true;
  }

  uint32_t parsed = 0;
  if (!parse_u32_env(name, 0, &parsed))
    return false;
  if (parsed > 255) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected 0..255\n", name, value);
    return false;
  }
  *out = static_cast<uint16_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_optional_sgpr_env(const char *name, std::optional<uint16_t> *out) {
  const char *value = std::getenv(name);
  out->reset();
  if (value == nullptr || *value == '\0')
    return true;
  uint32_t parsed = 0;
  if (!parse_u32_env(name, 0, &parsed))
    return false;
  if (parsed > 105) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected SGPR 0..105\n", name,
                 value);
    return false;
  }
  *out = static_cast<uint16_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_optional_sgpr_pair_env(const char *name, std::optional<uint16_t> *out) {
  const char *value = std::getenv(name);
  out->reset();
  if (value == nullptr || *value == '\0')
    return true;
  uint32_t parsed = 0;
  if (!parse_u32_env(name, 0, &parsed))
    return false;
  if (parsed > 104 || parsed % 2 != 0) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid %s='%s'; expected an even SGPR pair base "
                 "in 0..104\n",
                 name, value);
    return false;
  }
  *out = static_cast<uint16_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_u64_env(const char *name, uint64_t default_value, uint64_t *out) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    *out = default_value;
    return true;
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 0);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed > std::numeric_limits<uint64_t>::max()) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected uint64\n", name, value);
    return false;
  }

  *out = static_cast<uint64_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_delay_mode_env(rocjitsu::ConSanDelayMode *out) {
  const char *value = std::getenv("RJ_CONSAN_DELAY_MODE");
  if (value == nullptr || *value == '\0') {
    *out = rocjitsu::ConSanDelayMode::Nop;
    return true;
  }
  if (ascii_iequals(value, "nop")) {
    *out = rocjitsu::ConSanDelayMode::Nop;
    return true;
  }
  if (ascii_iequals(value, "sleep")) {
    *out = rocjitsu::ConSanDelayMode::Sleep;
    return true;
  }
  if (ascii_iequals(value, "sleep_var") || ascii_iequals(value, "sleep-var")) {
    *out = rocjitsu::ConSanDelayMode::SleepVar;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_DELAY_MODE='%s'; "
               "expected nop|sleep|sleep_var\n",
               value);
  return false;
}

[[nodiscard]] bool parse_flavor_env(std::optional<rocjitsu::ConSanFlavor> *out) {
  const char *value = std::getenv("RJ_CONSAN_FLAVOR");
  if (value == nullptr || *value == '\0') {
    out->reset();
    return true;
  }
  if (auto parsed = rocjitsu::parse_consan_flavor(value)) {
    *out = *parsed;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_FLAVOR='%s'; "
               "expected supercollider|moi\n",
               value);
  return false;
}

[[nodiscard]] bool parse_moi_engine_env(rocjitsu::ConSanMoiEngine *out) {
  const char *value = std::getenv("RJ_CONSAN_MOI_ENGINE");
  if (value != nullptr && *value != '\0') {
    if (auto parsed = rocjitsu::parse_consan_moi_engine(value)) {
      *out = *parsed;
      return true;
    }

    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_ENGINE='%s'; "
                 "expected record_replay|inline_shadow|sampled\n",
                 value);
    return false;
  }

  const char *legacy_value = std::getenv("RJ_CONSAN_MOI_BACKEND");
  if (legacy_value == nullptr || *legacy_value == '\0') {
    *out = rocjitsu::ConSanMoiEngine::RecordReplay;
    return true;
  }
  if (auto parsed = rocjitsu::parse_consan_moi_engine(legacy_value)) {
    *out = *parsed;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_BACKEND='%s'; "
               "expected record_replay|sampled or legacy context|sampled_watchpoint\n",
               legacy_value);
  return false;
}

[[nodiscard]] bool parse_moi_owner_source_env(rocjitsu::ConSanMoiOwnerSource *out) {
  const char *value = std::getenv("RJ_CONSAN_MOI_OWNER_SOURCE");
  if (value == nullptr || *value == '\0') {
    *out = rocjitsu::ConSanMoiOwnerSource::WorkitemId;
    return true;
  }
  const std::string_view mode(value);
  if (mode == "workitem_id" || mode == "workitem") {
    *out = rocjitsu::ConSanMoiOwnerSource::WorkitemId;
    return true;
  }
  if (mode == "hw_id" || mode == "hwid") {
    *out = rocjitsu::ConSanMoiOwnerSource::HwId;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_OWNER_SOURCE='%s'; "
               "expected workitem_id|hw_id\n",
               value);
  return false;
}

[[nodiscard]] bool parse_check_trap_mode_env(CheckTrapMode *out) {
  const char *value = std::getenv("RJ_CONSAN_CHECK_TRAP_MODE");
  if (value == nullptr || *value == '\0') {
    *out = CheckTrapMode::All;
    return true;
  }
  if (ascii_iequals(value, "all") || ascii_iequals(value, "both") ||
      ascii_iequals(value, "default")) {
    *out = CheckTrapMode::All;
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
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_CHECK_TRAP_MODE='%s'; "
               "expected all|lds|flat\n",
               value);
  return false;
}

[[nodiscard]] bool parse_log_level(int *out) {
  const char *value = std::getenv("RJ_CONSAN_LOG");
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
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_LOG='%s'; expected boolean or level\n",
                 value);
    return false;
  }

  *out = parsed > kLogDebug ? kLogDebug : static_cast<int>(parsed);
  return true;
}

[[nodiscard]] bool has_explicit_primary_probe(const HookConfig &config) {
  return config.probe_nop || config.probe_trampoline_nop || config.probe_endpgm ||
         config.probe_lds_endpgm || config.probe_flat_trap;
}

[[nodiscard]] bool env_has_value(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

void warn_ignored_env(const char *name, const char *reason) {
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] warning: %s is ignored: %s\n", name, reason);
}

void warn_env(const char *name, const char *message) {
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] warning: %s: %s\n", name, message);
}

void warn_irrelevant_env_combinations(const HookConfig &config) {
  if (env_has_value("RJ_CONSAN_MOI_ENGINE") && env_has_value("RJ_CONSAN_MOI_BACKEND"))
    warn_ignored_env("RJ_CONSAN_MOI_BACKEND", "RJ_CONSAN_MOI_ENGINE takes precedence");

  if (config.flavor == rocjitsu::ConSanFlavor::Moi) {
    if (config.moi_engine != rocjitsu::ConSanMoiEngine::Sampled) {
      constexpr const char *kSampledOnlyKnobs[] = {
          "RJ_CONSAN_MOI_SAMPLE_STRIDE",
          "RJ_CONSAN_MOI_SAMPLE_OFFSET",
      };
      for (const char *name : kSampledOnlyKnobs) {
        if (env_has_value(name))
          warn_ignored_env(name, "only applies to RJ_CONSAN_MOI_ENGINE=sampled");
      }
    }
    if (config.moi_engine != rocjitsu::ConSanMoiEngine::RecordReplay &&
        (env_has_value("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS") ||
         env_has_value("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT"))) {
      if (env_has_value("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS"))
        warn_ignored_env("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS",
                         "only applies to RJ_CONSAN_MOI_ENGINE=record_replay");
      if (env_has_value("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT"))
        warn_ignored_env("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT",
                         "only applies to RJ_CONSAN_MOI_ENGINE=record_replay");
    }
    if (!config.moi_init_owner_epoch && (env_has_value("RJ_CONSAN_MOI_OWNER_SOURCE") ||
                                         env_has_value("RJ_CONSAN_MOI_OWNER_SGPR"))) {
      if (env_has_value("RJ_CONSAN_MOI_OWNER_SOURCE"))
        warn_ignored_env("RJ_CONSAN_MOI_OWNER_SOURCE",
                         "only affects RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1");
      if (env_has_value("RJ_CONSAN_MOI_OWNER_SGPR"))
        warn_ignored_env("RJ_CONSAN_MOI_OWNER_SGPR",
                         "only affects RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1");
    }
    return;
  }

  constexpr const char *kMoiOnlyKnobs[] = {
      "RJ_CONSAN_MOI_ENGINE",
      "RJ_CONSAN_MOI_BACKEND",
      "RJ_CONSAN_MOI_REPORT_BUFFER",
      "RJ_CONSAN_MOI_REPORT_BUFFER_SIZE",
      "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE",
      "RJ_CONSAN_MOI_REQUIRE_RECORDS",
      "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS",
      "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS",
      "RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT",
      "RJ_CONSAN_MOI_INIT_OWNER_EPOCH",
      "RJ_CONSAN_MOI_TRACK_BARRIERS",
      "RJ_CONSAN_MOI_TRACK_ATOMICS",
      "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS",
      "RJ_CONSAN_MOI_EXEC_SAVE_SGPR",
      "RJ_CONSAN_MOI_OWNER_SOURCE",
      "RJ_CONSAN_MOI_OWNER_SGPR",
      "RJ_CONSAN_MOI_OWNER_VGPR",
      "RJ_CONSAN_MOI_EPOCH_VGPR",
      "RJ_CONSAN_MOI_SAMPLE_STRIDE",
      "RJ_CONSAN_MOI_SAMPLE_OFFSET",
  };
  for (const char *name : kMoiOnlyKnobs) {
    if (env_has_value(name))
      warn_ignored_env(name, "RJ_CONSAN_FLAVOR is not moi");
  }
}

[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config);

[[nodiscard]] std::optional<HookConfig> parse_config() {
  HookConfig config;
  if (!parse_log_level(&config.log_level))
    return std::nullopt;
  if (!parse_flavor_env(&config.flavor))
    return std::nullopt;
  if (!parse_moi_engine_env(&config.moi_engine))
    return std::nullopt;
  if (!parse_moi_owner_source_env(&config.moi_owner_source))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAIL_CLOSED", false, &config.fail_closed))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_REQUIRE_PATCH", false, &config.require_patch))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_NOP", false, &config.probe_nop))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_TRAMPOLINE_NOP", false, &config.probe_trampoline_nop))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_ENDPGM", false, &config.probe_endpgm))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_LDS_ENDPGM", false, &config.probe_lds_endpgm))
    return std::nullopt;
  if (!parse_check_trap_mode_env(&config.check_trap_mode))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_FLAT_TRAP", false, &config.probe_flat_trap))
    return std::nullopt;
  if (config.flavor == rocjitsu::ConSanFlavor::SuperCollider &&
      !has_explicit_primary_probe(config)) {
    config.probe_lds_check_trap = config.check_trap_mode == CheckTrapMode::All ||
                                  config.check_trap_mode == CheckTrapMode::Lds;
    config.probe_flat_check_trap = config.check_trap_mode == CheckTrapMode::All ||
                                   config.check_trap_mode == CheckTrapMode::Flat;
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_DROP_BARRIER", false, &config.fault_drop_barrier))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_INIT_OWNER_EPOCH", false, &config.moi_init_owner_epoch))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_TRACK_BARRIERS", false, &config.moi_track_barriers))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_TRACK_ATOMICS", false, &config.moi_track_atomics))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS", false,
                      &config.moi_dynamic_access_records))
    return std::nullopt;
  // Deliberately test-only: this is not part of the public ConSan knob set.
  if (!parse_bool_env("RJ_CONSAN_TEST_FORCE_VGPR_SPILL", false, &config.test_force_vgpr_spill))
    return std::nullopt;
  if (const char *test_filter = std::getenv("RJ_CONSAN_TEST_KERNEL_FILTER"))
    config.test_kernel_name_filter = test_filter;
  if (!parse_bool_env("RJ_CONSAN_MOI_REQUIRE_RECORDS", false, &config.moi_require_records))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS", false, &config.moi_require_diagnostics))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", false, &config.moi_forbid_diagnostics))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT", false,
                      &config.moi_require_replay_conflict))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_FAULT_BARRIER_INDEX", 0, &config.fault_barrier_index))
    return std::nullopt;
  if (!parse_delay_mode_env(&config.delay_mode))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_DELAY", 0, &config.delay_nops))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_MAX_PATCHES", 1, &config.max_patches))
    return std::nullopt;
  if (config.max_patches == 0) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MAX_PATCHES='0'; expected >=1\n");
    return std::nullopt;
  }
  if (!parse_u32_env("RJ_CONSAN_MOI_SAMPLE_STRIDE", 1, &config.moi_sample_stride))
    return std::nullopt;
  if (config.moi_sample_stride == 0) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_SAMPLE_STRIDE='0'; expected >=1\n");
    return std::nullopt;
  }
  if (!parse_u32_env("RJ_CONSAN_MOI_SAMPLE_OFFSET", 0, &config.moi_sample_offset))
    return std::nullopt;
  if (config.moi_sample_offset >= config.moi_sample_stride) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_SAMPLE_OFFSET='%s'; expected < "
                 "RJ_CONSAN_MOI_SAMPLE_STRIDE (%u)\n",
                 std::getenv("RJ_CONSAN_MOI_SAMPLE_OFFSET"), config.moi_sample_stride);
    return std::nullopt;
  }
  if (!refresh_report_config_from_env(&config))
    return std::nullopt;
  uint32_t delay_var_ssrc = 106;
  if (!parse_u32_env("RJ_CONSAN_DELAY_VAR_SSRC", 106, &delay_var_ssrc))
    return std::nullopt;
  if (delay_var_ssrc > 255) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_DELAY_VAR_SSRC='%s'; expected 0..255\n",
                 std::getenv("RJ_CONSAN_DELAY_VAR_SSRC"));
    return std::nullopt;
  }
  config.delay_var_ssrc = static_cast<uint16_t>(delay_var_ssrc);
  if (const char *value = std::getenv("RJ_CONSAN_DUMP_DIR"); value != nullptr && *value != '\0')
    config.dump_dir = value;
  uint32_t scratch_vgpr = 0;
  if (const char *value = std::getenv("RJ_CONSAN_TMP_VGPR"); value != nullptr && *value != '\0') {
    if (!parse_u32_env("RJ_CONSAN_TMP_VGPR", 0, &scratch_vgpr))
      return std::nullopt;
    if (scratch_vgpr > 255) {
      std::fprintf(
          stderr, "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_TMP_VGPR='%s'; expected 0..255\n", value);
      return std::nullopt;
    }
    config.scratch_vgpr = static_cast<uint16_t>(scratch_vgpr);
  }
  if (!parse_optional_vgpr_env("RJ_CONSAN_MOI_OWNER_VGPR", &config.moi_owner_vgpr))
    return std::nullopt;
  if (!parse_optional_vgpr_env("RJ_CONSAN_MOI_EPOCH_VGPR", &config.moi_epoch_vgpr))
    return std::nullopt;
  if (!parse_optional_sgpr_env("RJ_CONSAN_MOI_OWNER_SGPR", &config.moi_owner_sgpr))
    return std::nullopt;
  if (!parse_optional_sgpr_pair_env("RJ_CONSAN_MOI_EXEC_SAVE_SGPR", &config.moi_exec_save_sgpr))
    return std::nullopt;
  if (config.flavor == rocjitsu::ConSanFlavor::Moi && config.moi_require_records &&
      config.moi_auto_report_buffer_size == 0) {
    warn_env("RJ_CONSAN_MOI_REQUIRE_RECORDS",
             "this guard only checks HSA-tool-owned auto report buffers");
  }
  if (config.flavor == rocjitsu::ConSanFlavor::Moi && config.moi_require_replay_conflict &&
      config.moi_auto_report_buffer_size == 0) {
    warn_env("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT",
             "this guard only checks HSA-tool-owned auto report buffers");
  }
  if (config.flavor == rocjitsu::ConSanFlavor::Moi && config.moi_require_diagnostics &&
      config.moi_auto_report_buffer_size == 0) {
    warn_env("RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS",
             "this guard only checks HSA-tool-owned auto report buffers");
  }
  if (config.flavor == rocjitsu::ConSanFlavor::Moi && config.moi_forbid_diagnostics &&
      config.moi_auto_report_buffer_size == 0) {
    warn_env("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS",
             "this guard only checks HSA-tool-owned auto report buffers");
  }
  if (config.flavor == rocjitsu::ConSanFlavor::Moi && config.moi_require_diagnostics &&
      config.moi_forbid_diagnostics) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS and "
                         "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS cannot both be enabled\n");
    return std::nullopt;
  }
  warn_irrelevant_env_combinations(config);
  return config;
}

[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config) {
  uint64_t report_buffer_address = 0;
  if (const char *value = std::getenv("RJ_CONSAN_REPORT_BUFFER");
      value != nullptr && *value != '\0') {
    if (!parse_u64_env("RJ_CONSAN_REPORT_BUFFER", 0, &report_buffer_address))
      return false;
    if (report_buffer_address == 0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_REPORT_BUFFER='0'; expected nonzero "
                   "device-visible address\n");
      return false;
    }
    config->report_buffer_address = report_buffer_address;
  } else {
    config->report_buffer_address.reset();
  }

  uint64_t moi_report_buffer_address = 0;
  if (const char *value = std::getenv("RJ_CONSAN_MOI_REPORT_BUFFER");
      value != nullptr && *value != '\0') {
    if (!parse_u64_env("RJ_CONSAN_MOI_REPORT_BUFFER", 0, &moi_report_buffer_address))
      return false;
    if (moi_report_buffer_address == 0) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_REPORT_BUFFER='0'; expected "
                           "nonzero device-visible address\n");
      return false;
    }
    config->moi_report_buffer_address = moi_report_buffer_address;
  } else {
    config->moi_report_buffer_address.reset();
  }
  if (!parse_u64_env("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", 0, &config->moi_report_buffer_size))
    return false;
  if (!parse_u64_env("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", 0,
                     &config->moi_auto_report_buffer_size))
    return false;
  if (config->moi_auto_report_buffer_size != 0 &&
      config->moi_auto_report_buffer_size <
          rocjitsu::consan_moi_report_buffer_min_bytes(1, 0, 0, 0)) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE='%s'; "
                 "expected 0 or at least %zu bytes\n",
                 std::getenv("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE"),
                 rocjitsu::consan_moi_report_buffer_min_bytes(1, 0, 0, 0));
    return false;
  }

  return parse_u32_env("RJ_CONSAN_REPORT_MARKER", 1, &config->report_marker);
}

[[nodiscard]] bool is_supported_require_patch_flat_site(const rocjitsu::ConSanFlatSite &site) {
  if (site.kind != rocjitsu::ConSanLdsAccessKind::Read &&
      site.kind != rocjitsu::ConSanLdsAccessKind::Write)
    return false;
  if (site.size != 3u * sizeof(uint32_t) || !site.addr_vgpr)
    return false;
  if (site.width_bits != 32u && site.width_bits != 64u && site.width_bits != 128u)
    return false;
  if (site.kind == rocjitsu::ConSanLdsAccessKind::Read) {
    if (site.mnemonic != "flat_load_b32" && site.mnemonic != "flat_load_b64" &&
        site.mnemonic != "flat_load_b128")
      return false;
    if (!site.dst_vgpr)
      return false;
  } else {
    if (site.mnemonic != "flat_store_b32" && site.mnemonic != "flat_store_b64" &&
        site.mnemonic != "flat_store_b128")
      return false;
    if (!site.data_vgpr)
      return false;
  }
  return site.address_space_hint == rocjitsu::ConSanFlatAddressSpaceHint::Group ||
         site.address_space_hint == rocjitsu::ConSanFlatAddressSpaceHint::MaybeGroup;
}

[[nodiscard]] bool require_patch_applies_to(const rocjitsu::ConSanResult &result,
                                            const HookConfig &config) {
  for (const rocjitsu::ConSanKernelInfo &kernel : result.kernels) {
    if (config.probe_lds_check_trap) {
      for (const rocjitsu::ConSanLdsSite &site : kernel.lds_sites) {
        if (site.supported_mvp &&
            (site.mnemonic == "ds_load_b32" || site.mnemonic == "ds_load_b64" ||
             site.mnemonic == "ds_load_b128" || site.mnemonic == "ds_load_2addr_b32" ||
             site.mnemonic == "ds_load_2addr_b64" ||
             site.mnemonic == "ds_load_2addr_stride64_b32" ||
             site.mnemonic == "ds_load_2addr_stride64_b64" || site.mnemonic == "ds_load_u16_d16" ||
             site.mnemonic == "ds_load_u16_d16_hi" || site.mnemonic == "ds_store_b32" ||
             site.mnemonic == "ds_store_b64" || site.mnemonic == "ds_store_b128"))
          return true;
      }
    }
    if (config.probe_flat_check_trap) {
      for (const rocjitsu::ConSanFlatSite &site : kernel.flat_sites) {
        if (is_supported_require_patch_flat_site(site))
          return true;
      }
    }
  }
  if (config.probe_flat_check_trap) {
    for (const rocjitsu::ConSanFunctionInfo &function : result.functions) {
      for (const rocjitsu::ConSanFlatSite &site : function.flat_sites) {
        if (is_supported_require_patch_flat_site(site))
          return true;
      }
    }
  }
  return false;
}

[[nodiscard]] bool
is_supported_require_patch_moi_candidate(const rocjitsu::ConSanMoiCandidate &candidate) {
  if (candidate.kind != rocjitsu::ConSanLdsAccessKind::Read &&
      candidate.kind != rocjitsu::ConSanLdsAccessKind::Write)
    return false;
  if (candidate.size == 0 || candidate.size % sizeof(uint32_t) != 0)
    return false;
  if (candidate.width_bits == 0 || candidate.width_bits % 8u != 0)
    return false;
  if (!candidate.addr_vgpr)
    return false;

  if (candidate.source == rocjitsu::ConSanMoiCandidateSource::NativeLds)
    return rocjitsu::consan_moi_supports_native_lds_record_replay_mnemonic(candidate.mnemonic);

  if (candidate.source != rocjitsu::ConSanMoiCandidateSource::FlatGroup &&
      candidate.source != rocjitsu::ConSanMoiCandidateSource::FlatMaybeGroup)
    return false;
  if (candidate.size != 3u * sizeof(uint32_t))
    return false;
  if (!candidate.raw_ioffset || *candidate.raw_ioffset != 0)
    return false;
  if (*candidate.addr_vgpr >= 255)
    return false;
  return candidate.mnemonic == "flat_load_b32" || candidate.mnemonic == "flat_load_b64" ||
         candidate.mnemonic == "flat_load_b128" || candidate.mnemonic == "flat_store_b32" ||
         candidate.mnemonic == "flat_store_b64" || candidate.mnemonic == "flat_store_b128";
}

[[nodiscard]] bool require_moi_patch_applies_to(const rocjitsu::ConSanResult &result) {
  return std::any_of(result.moi_candidates.begin(), result.moi_candidates.end(),
                     is_supported_require_patch_moi_candidate);
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
    log_message(kLogInfo, "failed to create RJ_CONSAN_DUMP_DIR='%s': %s", config.dump_dir.c_str(),
                std::strerror(errno));
    return;
  }

  std::array<char, 4096> path{};
  const int written = std::snprintf(
      path.data(), path.size(), "%s/rj-dbi-%06llu-reader-%llu-%.*s.hsaco", config.dump_dir.c_str(),
      static_cast<unsigned long long>(dump_id), static_cast<unsigned long long>(reader),
      static_cast<int>(tag.size()), tag.data());
  if (written < 0 || static_cast<size_t>(written) >= path.size()) {
    log_message(kLogInfo, "RJ_CONSAN_DUMP_DIR path is too long: %s", config.dump_dir.c_str());
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

class AutoMoiReportBufferRegistry {
public:
  static AutoMoiReportBufferRegistry &instance() {
    static AutoMoiReportBufferRegistry registry;
    return registry;
  }

  [[nodiscard]] bool allocate(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                              uint64_t requested_size, bool direct_sampled, bool inline_shadow,
                              bool track_barriers, bool track_atomics, uint64_t *address,
                              uint64_t *registered_size) {
    if (core == nullptr || core->hsa_agent_iterate_regions_fn == nullptr ||
        core->hsa_region_get_info_fn == nullptr || core->hsa_memory_allocate_fn == nullptr ||
        core->hsa_memory_free_fn == nullptr) {
      log_message(
          kLogInfo,
          "ConSan MOI auto report buffer requested but HSA allocation APIs are unavailable");
      return false;
    }
    if (requested_size > std::numeric_limits<size_t>::max()) {
      log_message(kLogInfo, "ConSan MOI auto report buffer size is too large: %llu",
                  static_cast<unsigned long long>(requested_size));
      return false;
    }

    const size_t requested = static_cast<size_t>(requested_size);
    const rocjitsu::ConSanMoiReportBufferLayout layout =
        inline_shadow
            ? rocjitsu::consan_moi_inline_shadow_report_buffer_layout_for_bytes(requested_size)
        : direct_sampled
            ? rocjitsu::consan_moi_direct_sampled_report_buffer_layout_for_bytes(requested_size)
            : rocjitsu::consan_moi_report_buffer_layout_for_bytes(requested_size, track_barriers,
                                                                  track_atomics);
    if (requested < sizeof(rocjitsu::ConSanMoiReportHeader) ||
        (!direct_sampled && !inline_shadow && layout.access_record_capacity == 0) ||
        (direct_sampled && layout.sampled_watchpoint_capacity == 0) ||
        (inline_shadow &&
         (layout.diagnostic_capacity == 0 || layout.exact_shadow_entry_capacity == 0)) ||
        (track_barriers && !inline_shadow && layout.barrier_record_capacity == 0) ||
        (track_atomics && !inline_shadow && layout.atomic_record_capacity == 0) ||
        (track_atomics && inline_shadow &&
         layout.inline_atomic_release_slots_offset == layout.sampled_watchpoints_offset)) {
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer is too small reader=%llu bytes=%zu "
                  "direct_sampled=%s inline_shadow=%s track_barriers=%s track_atomics=%s",
                  static_cast<unsigned long long>(reader), requested,
                  direct_sampled ? "true" : "false", inline_shadow ? "true" : "false",
                  track_barriers ? "true" : "false", track_atomics ? "true" : "false");
      return false;
    }

    RegionSearch search;
    search.core = core;
    search.requested_size = requested;
    const hsa_status_t iterate_status =
        core->hsa_agent_iterate_regions_fn(agent, select_region, &search);
    if (iterate_status != HSA_STATUS_SUCCESS && iterate_status != HSA_STATUS_INFO_BREAK) {
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer region iteration failed reader=%llu status=%d",
                  static_cast<unsigned long long>(reader), static_cast<int>(iterate_status));
      return false;
    }
    if (!search.found) {
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer found no allocatable global HSA region "
                  "reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(reader), requested);
      return false;
    }

    void *ptr = nullptr;
    const hsa_status_t status = core->hsa_memory_allocate_fn(search.region, requested, &ptr);
    if (status != HSA_STATUS_SUCCESS) {
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer hsa_memory_allocate failed reader=%llu "
                  "status=%d bytes=%zu",
                  static_cast<unsigned long long>(reader), static_cast<int>(status), requested);
      return false;
    }
    std::memset(ptr, 0, requested);

    if (core->hsa_memory_assign_agent_fn != nullptr) {
      const hsa_status_t assign_status =
          core->hsa_memory_assign_agent_fn(ptr, agent, HSA_ACCESS_PERMISSION_RW);
      if (assign_status != HSA_STATUS_SUCCESS) {
        log_message(kLogInfo,
                    "ConSan MOI auto report buffer hsa_memory_assign_agent failed reader=%llu "
                    "status=%d",
                    static_cast<unsigned long long>(reader), static_cast<int>(assign_status));
        (void)core->hsa_memory_free_fn(ptr);
        return false;
      }
    }

    const uint64_t generation = next_generation_.fetch_add(1, std::memory_order_relaxed) + 1u;
    auto *header = static_cast<rocjitsu::ConSanMoiReportHeader *>(ptr);
    *header = rocjitsu::make_consan_moi_report_header(
        generation, /*dispatch_id=*/reader, layout.access_record_capacity,
        layout.diagnostic_capacity, layout.exact_shadow_entry_capacity,
        layout.sampled_watchpoint_capacity, layout.barrier_record_capacity,
        layout.atomic_record_capacity);

    {
      std::lock_guard lock(mutex_);
      if (entry_count_ >= entries_.size()) {
        log_message(kLogInfo, "ConSan MOI auto report buffer registry is full");
        (void)core->hsa_memory_free_fn(ptr);
        return false;
      }
      entries_[entry_count_++] = Entry{reader,
                                       ptr,
                                       requested,
                                       generation,
                                       layout.access_record_capacity,
                                       layout.barrier_record_capacity,
                                       layout.atomic_record_capacity,
                                       layout.diagnostic_capacity,
                                       layout.exact_shadow_entry_capacity,
                                       layout.sampled_watchpoint_capacity,
                                       search.fine_grained};
    }
    *address = reinterpret_cast<uint64_t>(ptr);
    *registered_size = requested;
    log_message(kLogInfo,
                "ConSan MOI auto report buffer reader=%llu addr=0x%llx bytes=%zu "
                "access_record_capacity=%u barrier_record_capacity=%u atomic_record_capacity=%u "
                "diagnostic_capacity=%u exact_shadow_entry_capacity=%u "
                "sampled_watchpoint_capacity=%u generation=%llu fine_grained=%s",
                static_cast<unsigned long long>(reader), static_cast<unsigned long long>(*address),
                requested, layout.access_record_capacity, layout.barrier_record_capacity,
                layout.atomic_record_capacity, layout.diagnostic_capacity,
                layout.exact_shadow_entry_capacity, layout.sampled_watchpoint_capacity,
                static_cast<unsigned long long>(generation),
                search.fine_grained ? "true" : "false");
    return true;
  }

  struct Summary {
    uint64_t buffer_count = 0;
    uint64_t visible_access_record_count = 0;
    uint64_t visible_barrier_record_count = 0;
    uint64_t visible_atomic_record_count = 0;
    uint64_t visible_diagnostic_record_count = 0;
    uint64_t visible_exact_shadow_entry_count = 0;
    uint64_t visible_sampled_watchpoint_count = 0;
    uint64_t dropped_access_record_count = 0;
    uint64_t dropped_barrier_record_count = 0;
    uint64_t dropped_atomic_record_count = 0;
    uint64_t replay_conflict_count = 0;
    uint64_t replay_diagnostic_count = 0;
    uint64_t sampled_conflict_count = 0;
  };

  Summary summarize_and_clear(CoreApiTable *core) {
    Summary total;
    std::lock_guard lock(mutex_);
    for (size_t i = 0; i < entry_count_; ++i) {
      const Summary entry_summary = summarize(core, entries_[i]);
      total.buffer_count += entry_summary.buffer_count;
      total.visible_access_record_count += entry_summary.visible_access_record_count;
      total.visible_barrier_record_count += entry_summary.visible_barrier_record_count;
      total.visible_atomic_record_count += entry_summary.visible_atomic_record_count;
      total.visible_diagnostic_record_count += entry_summary.visible_diagnostic_record_count;
      total.visible_exact_shadow_entry_count += entry_summary.visible_exact_shadow_entry_count;
      total.visible_sampled_watchpoint_count += entry_summary.visible_sampled_watchpoint_count;
      total.dropped_access_record_count += entry_summary.dropped_access_record_count;
      total.dropped_barrier_record_count += entry_summary.dropped_barrier_record_count;
      total.dropped_atomic_record_count += entry_summary.dropped_atomic_record_count;
      total.replay_conflict_count += entry_summary.replay_conflict_count;
      total.replay_diagnostic_count += entry_summary.replay_diagnostic_count;
      total.sampled_conflict_count += entry_summary.sampled_conflict_count;
      if (core != nullptr && core->hsa_memory_free_fn != nullptr && entries_[i].ptr != nullptr)
        (void)core->hsa_memory_free_fn(entries_[i].ptr);
      entries_[i] = Entry{};
    }
    entry_count_ = 0;
    return total;
  }

private:
  struct RegionSearch {
    CoreApiTable *core = nullptr;
    size_t requested_size = 0;
    hsa_region_t region{};
    bool found = false;
    bool fine_grained = false;
  };

  struct Entry {
    uint64_t reader = 0;
    void *ptr = nullptr;
    size_t size = 0;
    uint64_t generation = 0;
    uint32_t access_record_capacity = 0;
    uint32_t barrier_record_capacity = 0;
    uint32_t atomic_record_capacity = 0;
    uint32_t diagnostic_capacity = 0;
    uint32_t exact_shadow_entry_capacity = 0;
    uint32_t sampled_watchpoint_capacity = 0;
    bool fine_grained = false;
  };

  static hsa_status_t HSA_API select_region(hsa_region_t region, void *data) {
    auto *search = static_cast<RegionSearch *>(data);
    hsa_region_segment_t segment{};
    hsa_status_t status =
        search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (segment != HSA_REGION_SEGMENT_GLOBAL)
      return HSA_STATUS_SUCCESS;

    bool alloc_allowed = false;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                                  &alloc_allowed);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (!alloc_allowed)
      return HSA_STATUS_SUCCESS;

    size_t max_size = 0;
    status =
        search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &max_size);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (max_size < search->requested_size)
      return HSA_STATUS_SUCCESS;

    uint32_t flags = 0;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    const bool fine_grained = (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0;
    const bool coarse_grained = (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    if (fine_grained) {
      search->region = region;
      search->found = true;
      search->fine_grained = true;
      return HSA_STATUS_INFO_BREAK;
    }
    if (!search->found && coarse_grained) {
      search->region = region;
      search->found = true;
      search->fine_grained = false;
    }
    return HSA_STATUS_SUCCESS;
  }

  static Summary summarize(CoreApiTable *core, const Entry &entry) {
    Summary summary;
    summary.buffer_count = 1;
    std::vector<uint8_t> snapshot;
    const void *report_ptr = entry.ptr;
    if (!entry.fine_grained) {
      if (core == nullptr || core->hsa_memory_copy_fn == nullptr) {
        log_message(kLogInfo,
                    "ConSan MOI auto report reader=%llu needs hsa_memory_copy for "
                    "coarse-grained summary",
                    static_cast<unsigned long long>(entry.reader));
        return summary;
      }
      snapshot.resize(entry.size);
      const hsa_status_t copy_status =
          core->hsa_memory_copy_fn(snapshot.data(), entry.ptr, entry.size);
      if (copy_status != HSA_STATUS_SUCCESS) {
        log_message(kLogInfo, "ConSan MOI auto report reader=%llu hsa_memory_copy failed status=%d",
                    static_cast<unsigned long long>(entry.reader), static_cast<int>(copy_status));
        return summary;
      }
      report_ptr = snapshot.data();
    }

    const auto *header = static_cast<const rocjitsu::ConSanMoiReportHeader *>(report_ptr);
    if (header->magic != rocjitsu::kConSanMoiReportMagic ||
        header->abi_version != rocjitsu::kConSanMoiReportAbiVersion ||
        header->header_size != sizeof(rocjitsu::ConSanMoiReportHeader)) {
      log_message(kLogInfo,
                  "ConSan MOI auto report reader=%llu has invalid header magic=0x%08x "
                  "abi=%u header_size=%u",
                  static_cast<unsigned long long>(entry.reader), header->magic, header->abi_version,
                  header->header_size);
      return summary;
    }

    const uint32_t visible_records =
        std::min(header->access_record_count, header->access_record_capacity);
    const uint32_t visible_barriers =
        std::min(header->barrier_record_count, header->barrier_record_capacity);
    const uint32_t visible_atomics =
        std::min(header->atomic_record_count, header->atomic_record_capacity);
    const uint32_t visible_diagnostics =
        std::min(header->diagnostic_count, header->diagnostic_capacity);
    const uint32_t dropped_records = header->access_record_count > visible_records
                                         ? header->access_record_count - visible_records
                                         : 0;
    const uint32_t dropped_barriers = header->barrier_record_count > visible_barriers
                                          ? header->barrier_record_count - visible_barriers
                                          : 0;
    const uint32_t dropped_atomics = header->atomic_record_count > visible_atomics
                                         ? header->atomic_record_count - visible_atomics
                                         : 0;
    const auto *exact_shadow = reinterpret_cast<const uint64_t *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord) +
        static_cast<size_t>(header->barrier_record_capacity) *
            sizeof(rocjitsu::ConSanMoiBarrierRecord) +
        static_cast<size_t>(header->atomic_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAtomicRecord) +
        static_cast<size_t>(header->diagnostic_capacity) *
            sizeof(rocjitsu::ConSanMoiDiagnosticRecord));
    const auto *sampled = exact_shadow + static_cast<size_t>(header->exact_shadow_entry_capacity);
    struct ExactShadowEntry {
      uint32_t index = 0;
      rocjitsu::ConSanMoiExactShadowEntry entry;
    };
    std::vector<ExactShadowEntry> visible_exact_shadow;
    for (uint32_t i = 0; i < header->exact_shadow_entry_capacity; ++i) {
      if (exact_shadow[i] == 0)
        continue;
      visible_exact_shadow.push_back(
          {i, rocjitsu::decode_consan_moi_exact_shadow_entry(exact_shadow[i])});
    }
    struct SampledEntry {
      uint32_t index = 0;
      rocjitsu::ConSanMoiSampledWatchpointEntry entry;
    };
    std::vector<SampledEntry> visible_sampled;
    for (uint32_t i = 0; i < header->sampled_watchpoint_capacity; ++i) {
      const rocjitsu::ConSanMoiSampledWatchpointEntry entry =
          rocjitsu::decode_consan_moi_sampled_watchpoint_entry(sampled[i]);
      if (entry.valid)
        visible_sampled.push_back({i, entry});
    }
    uint32_t sampled_conflicts = 0;
    std::optional<std::pair<SampledEntry, SampledEntry>> first_sampled_conflict;
    for (size_t i = 0; i < visible_sampled.size(); ++i) {
      const SampledEntry &current = visible_sampled[i];
      for (size_t prior_index = 0; prior_index < i; ++prior_index) {
        const SampledEntry &prior = visible_sampled[prior_index];
        if (!rocjitsu::consan_moi_sampled_watchpoints_conflict(current.entry, prior.entry))
          continue;
        if (sampled_conflicts != std::numeric_limits<uint32_t>::max())
          ++sampled_conflicts;
        if (!first_sampled_conflict)
          first_sampled_conflict = std::make_pair(prior, current);
      }
    }
    summary.visible_access_record_count = visible_records;
    summary.visible_barrier_record_count = visible_barriers;
    summary.visible_atomic_record_count = visible_atomics;
    summary.visible_diagnostic_record_count = visible_diagnostics;
    summary.visible_exact_shadow_entry_count = visible_exact_shadow.size();
    summary.visible_sampled_watchpoint_count = visible_sampled.size();
    summary.dropped_access_record_count = dropped_records;
    summary.dropped_barrier_record_count = dropped_barriers;
    summary.dropped_atomic_record_count = dropped_atomics;
    summary.sampled_conflict_count = sampled_conflicts;
    log_message(kLogInfo,
                "ConSan MOI auto report reader=%llu addr=0x%llx bytes=%zu generation=%llu "
                "event_counter=%u access_records=%u visible_records=%u dropped_records=%u "
                "capacity=%u "
                "barrier_records=%u visible_barriers=%u dropped_barriers=%u barrier_capacity=%u "
                "atomic_records=%u visible_atomics=%u dropped_atomics=%u atomic_capacity=%u "
                "diagnostics=%u visible_diagnostics=%u diagnostic_capacity=%u "
                "exact_shadow_capacity=%u visible_exact_shadow=%zu "
                "sampled_watchpoints=%u visible_sampled=%zu sampled_conflicts=%u "
                "fine_grained=%s",
                static_cast<unsigned long long>(entry.reader),
                static_cast<unsigned long long>(reinterpret_cast<uint64_t>(entry.ptr)), entry.size,
                static_cast<unsigned long long>(header->generation), header->event_counter,
                header->access_record_count, visible_records, dropped_records,
                header->access_record_capacity, header->barrier_record_count, visible_barriers,
                dropped_barriers, header->barrier_record_capacity, header->atomic_record_count,
                visible_atomics, dropped_atomics, header->atomic_record_capacity,
                header->diagnostic_count, visible_diagnostics, header->diagnostic_capacity,
                header->exact_shadow_entry_capacity, visible_exact_shadow.size(),
                header->sampled_watchpoint_capacity, visible_sampled.size(), sampled_conflicts,
                entry.fine_grained ? "true" : "false");

    const auto *records = reinterpret_cast<const rocjitsu::ConSanMoiAccessRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader));
    const auto *barriers = reinterpret_cast<const rocjitsu::ConSanMoiBarrierRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord));
    const auto *atomics = reinterpret_cast<const rocjitsu::ConSanMoiAtomicRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord) +
        static_cast<size_t>(header->barrier_record_capacity) *
            sizeof(rocjitsu::ConSanMoiBarrierRecord));
    const auto *diagnostics = reinterpret_cast<const rocjitsu::ConSanMoiDiagnosticRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord) +
        static_cast<size_t>(header->barrier_record_capacity) *
            sizeof(rocjitsu::ConSanMoiBarrierRecord) +
        static_cast<size_t>(header->atomic_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAtomicRecord));
    if (visible_records != 0 || visible_barriers != 0 || visible_atomics != 0) {
      uint64_t required_shadow_entries = 0;
      for (uint32_t i = 0; i < visible_records; ++i) {
        const rocjitsu::ConSanMoiAccessRecord &record = records[i];
        uint64_t record_end = static_cast<uint64_t>(record.start_cell) + record.cell_count;
        if (record_end == 0 && record.lds_byte_count != 0) {
          const rocjitsu::ConSanMoiLdsCellRange range =
              rocjitsu::consan_moi_lds_cell_range_for_bytes(record.lds_byte_offset,
                                                            record.lds_byte_count);
          record_end = static_cast<uint64_t>(range.start_cell) + range.cell_count;
        }
        required_shadow_entries = std::max(required_shadow_entries, record_end);
      }
      const uint64_t kMaxAutoReplayShadowEntries = 1u << 20u;
      if (required_shadow_entries > kMaxAutoReplayShadowEntries) {
        log_message(kLogInfo,
                    "ConSan MOI auto replay reader=%llu skipped required_shadow_entries=%llu "
                    "limit=%llu",
                    static_cast<unsigned long long>(entry.reader),
                    static_cast<unsigned long long>(required_shadow_entries),
                    static_cast<unsigned long long>(kMaxAutoReplayShadowEntries));
      } else {
        rocjitsu::ConSanMoiReportHeader replay_header = *header;
        replay_header.diagnostic_count = 0;
        replay_header.diagnostic_capacity = 4;
        std::vector<rocjitsu::ConSanMoiDiagnosticRecord> diagnostics(
            replay_header.diagnostic_capacity);
        std::vector<uint64_t> exact_shadow_entries(
            static_cast<size_t>(std::max<uint64_t>(required_shadow_entries, 1u)));
        const rocjitsu::ConSanMoiRecordReplayResult replay =
            rocjitsu::consan_moi_record_replay_access_records(
                replay_header,
                std::span<const rocjitsu::ConSanMoiAccessRecord>(records, visible_records),
                std::span<const rocjitsu::ConSanMoiBarrierRecord>(barriers, visible_barriers),
                std::span<const rocjitsu::ConSanMoiRecordReplayAtomicEvent>(atomics,
                                                                            visible_atomics),
                diagnostics, exact_shadow_entries);
        summary.replay_conflict_count = replay.conflict ? 1u : 0u;
        summary.replay_diagnostic_count = replay.emitted_diagnostic_count;
        log_message(kLogInfo,
                    "ConSan MOI auto replay reader=%llu processed_access=%u processed_barriers=%u "
                    "processed_atomics=%u dropped_access=%u dropped_barriers=%u "
                    "unsupported_access=%u unsupported_atomics=%u diagnostics=%u "
                    "conflict=%s metadata_full=%s diagnostic_capacity_exhausted=%s "
                    "shadow_entries=%zu",
                    static_cast<unsigned long long>(entry.reader), replay.processed_access_count,
                    replay.processed_barrier_count, replay.processed_atomic_count,
                    replay.dropped_access_count, replay.dropped_barrier_count,
                    replay.unsupported_access_count, replay.unsupported_atomic_count,
                    replay.emitted_diagnostic_count, replay.conflict ? "true" : "false",
                    replay.metadata_full ? "true" : "false",
                    replay.diagnostic_capacity_exhausted ? "true" : "false",
                    exact_shadow_entries.size());
        if (replay.emitted_diagnostic_count != 0) {
          const rocjitsu::ConSanMoiDiagnosticRecord &diagnostic = diagnostics[0];
          log_message(kLogInfo,
                      "ConSan MOI auto replay diagnostic reader=%llu kind=%u generation=%llu "
                      "epoch=%u first_owner=%u second_owner=%u first_inst=0x%x "
                      "second_inst=0x%x first_lds_known=%s first_lds=[%u,%u) "
                      "second_lds=[%u,%u) first_kind=%u second_kind=%u "
                      "first_lane_mask=0x%llx second_lane_mask=0x%llx",
                      static_cast<unsigned long long>(entry.reader), diagnostic.kind,
                      static_cast<unsigned long long>(diagnostic.generation), diagnostic.epoch,
                      diagnostic.first_owner_id, diagnostic.second_owner_id,
                      diagnostic.first_instruction_offset, diagnostic.second_instruction_offset,
                      diagnostic.first_lds_byte_count != 0 ? "true" : "false",
                      diagnostic.first_lds_byte_offset,
                      diagnostic.first_lds_byte_offset + diagnostic.first_lds_byte_count,
                      diagnostic.second_lds_byte_offset,
                      diagnostic.second_lds_byte_offset + diagnostic.second_lds_byte_count,
                      diagnostic.first_access_kind, diagnostic.second_access_kind,
                      static_cast<unsigned long long>(diagnostic.first_lane_mask),
                      static_cast<unsigned long long>(diagnostic.second_lane_mask));
        }
      }
    }

    const uint32_t sample_count = std::min<uint32_t>(visible_records, 4u);
    for (uint32_t i = 0; i < sample_count; ++i) {
      const rocjitsu::ConSanMoiAccessRecord &record = records[i];
      log_message(kLogInfo,
                  "ConSan MOI auto record reader=%llu index=%u event_index=%u kind=%u wave=%u "
                  "epoch=%u inst=0x%x lds_offset=%u lds_bytes=%u cells=[%u,%u) "
                  "lane_mask=0x%llx",
                  static_cast<unsigned long long>(entry.reader), i, record.event_index,
                  record.access_kind, record.wave_id, record.epoch, record.instruction_offset,
                  record.lds_byte_offset, record.lds_byte_count, record.start_cell,
                  record.start_cell + record.cell_count,
                  static_cast<unsigned long long>(record.lane_mask));
    }

    const uint32_t barrier_sample_count = std::min<uint32_t>(visible_barriers, 4u);
    for (uint32_t i = 0; i < barrier_sample_count; ++i) {
      const rocjitsu::ConSanMoiBarrierRecord &record = barriers[i];
      log_message(kLogInfo,
                  "ConSan MOI auto barrier reader=%llu index=%u event_index=%u wave=%u "
                  "inst=0x%x lane_mask=0x%llx",
                  static_cast<unsigned long long>(entry.reader), i, record.event_index,
                  record.wave_id, record.instruction_offset,
                  static_cast<unsigned long long>(record.lane_mask));
    }

    const uint32_t atomic_sample_count = std::min<uint32_t>(visible_atomics, 4u);
    for (uint32_t i = 0; i < atomic_sample_count; ++i) {
      const rocjitsu::ConSanMoiAtomicRecord &record = atomics[i];
      log_message(kLogInfo,
                  "ConSan MOI auto atomic reader=%llu index=%u event_index=%u kind=%u owner=%u "
                  "epoch=%u inst=0x%x address=0x%llx scope=%u semantics=%u",
                  static_cast<unsigned long long>(entry.reader), i, record.event_index,
                  static_cast<uint32_t>(record.kind), record.owner_id, record.epoch,
                  record.instruction_offset, static_cast<unsigned long long>(record.atomic_address),
                  record.scope, record.semantics);
    }
    const uint32_t diagnostic_sample_count = std::min<uint32_t>(visible_diagnostics, 4u);
    for (uint32_t i = 0; i < diagnostic_sample_count; ++i) {
      const rocjitsu::ConSanMoiDiagnosticRecord &record = diagnostics[i];
      log_message(kLogInfo,
                  "ConSan MOI auto diagnostic reader=%llu index=%u backend=%u kind=%u "
                  "generation=%llu epoch=%u first_owner=%u second_owner=%u first_inst=0x%x "
                  "second_inst=0x%x first_kind=%u second_kind=%u second_lds=[%u,%u)",
                  static_cast<unsigned long long>(entry.reader), i, record.backend, record.kind,
                  static_cast<unsigned long long>(record.generation), record.epoch,
                  record.first_owner_id, record.second_owner_id, record.first_instruction_offset,
                  record.second_instruction_offset, record.first_access_kind,
                  record.second_access_kind, record.second_lds_byte_offset,
                  record.second_lds_byte_offset + record.second_lds_byte_count);
    }
    for (uint32_t i = 0; i < std::min<size_t>(visible_exact_shadow.size(), 4u); ++i) {
      const ExactShadowEntry &shadow_entry = visible_exact_shadow[i];
      log_message(kLogInfo,
                  "ConSan MOI auto exact-shadow reader=%llu index=%u kind=%u owner=%u epoch=%u "
                  "generation=%u inst=0x%x",
                  static_cast<unsigned long long>(entry.reader), shadow_entry.index,
                  static_cast<uint32_t>(shadow_entry.entry.kind), shadow_entry.entry.owner_id,
                  shadow_entry.entry.epoch, shadow_entry.entry.generation,
                  shadow_entry.entry.instruction_offset);
    }
    for (uint32_t i = 0; i < std::min<size_t>(visible_sampled.size(), 4u); ++i) {
      const SampledEntry &sampled_entry = visible_sampled[i];
      log_message(kLogInfo,
                  "ConSan MOI auto sampled reader=%llu index=%u kind=%u owner=%u epoch=%u "
                  "generation=%u cells=[%u,%u) consumed=%s",
                  static_cast<unsigned long long>(entry.reader), sampled_entry.index,
                  static_cast<uint32_t>(sampled_entry.entry.kind), sampled_entry.entry.owner_id,
                  sampled_entry.entry.epoch, sampled_entry.entry.generation,
                  sampled_entry.entry.start_cell,
                  sampled_entry.entry.start_cell + sampled_entry.entry.cell_count,
                  sampled_entry.entry.consumed ? "true" : "false");
    }
    if (first_sampled_conflict) {
      const SampledEntry &first = first_sampled_conflict->first;
      const SampledEntry &second = first_sampled_conflict->second;
      const uint32_t first_end = first.entry.start_cell + first.entry.cell_count;
      const uint32_t second_end = second.entry.start_cell + second.entry.cell_count;
      log_message(kLogInfo,
                  "ConSan MOI auto sampled conflict reader=%llu first_index=%u second_index=%u "
                  "first_kind=%u second_kind=%u first_owner=%u second_owner=%u epoch=%u "
                  "generation=%u first_cells=[%u,%u) second_cells=[%u,%u)",
                  static_cast<unsigned long long>(entry.reader), first.index, second.index,
                  static_cast<uint32_t>(first.entry.kind), static_cast<uint32_t>(second.entry.kind),
                  first.entry.owner_id, second.entry.owner_id, second.entry.epoch,
                  second.entry.generation, first.entry.start_cell, first_end,
                  second.entry.start_cell, second_end);
    }
    return summary;
  }

  std::mutex mutex_;
  std::array<Entry, 256> entries_{};
  size_t entry_count_ = 0;
  std::atomic<uint64_t> next_generation_{0};
};

class KernelPrivateDispatchRegistry {
public:
  static KernelPrivateDispatchRegistry &instance() {
    // The HSA runtime may call OnUnload from a shared-library finalizer after
    // ordinary function-local statics have already been destroyed. Keep the
    // registry alive for the process lifetime and clear its contents explicitly
    // when the hook layer is uninstalled.
    static auto *registry = new KernelPrivateDispatchRegistry;
    return *registry;
  }

  void note_patch_requirements(hsa_executable_t executable, const rocjitsu::ConSanResult &result) {
    std::lock_guard lock(mutex_);
    for (const rocjitsu::ConSanPatchInfo &patch : result.patches) {
      if (patch.required_private_segment_size == 0)
        continue;
      const auto kernel = std::ranges::find_if(result.kernels, [&](const auto &candidate) {
        return candidate.has_text_range && patch.anchor_offset >= candidate.entry_text_offset &&
               patch.anchor_offset - candidate.entry_text_offset < candidate.code_size;
      });
      if (kernel == result.kernels.end())
        continue;
      const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
        return candidate.executable == executable.handle && candidate.kernel_name == kernel->name;
      });
      if (pending == pending_.end()) {
        pending_.push_back({executable.handle, kernel->name, patch.required_private_segment_size});
      } else {
        pending->required_private_bytes =
            std::max(pending->required_private_bytes, patch.required_private_segment_size);
      }
    }
  }

  void bind_symbol(hsa_executable_t executable, std::string_view symbol_name,
                   hsa_executable_symbol_t symbol,
                   decltype(hsa_executable_symbol_get_info) *original_get_info) {
    if (original_get_info == nullptr)
      return;
    std::lock_guard lock(mutex_);
    const std::string_view normalized = normalize_kernel_name(symbol_name);
    const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
      return candidate.executable == executable.handle &&
             normalize_kernel_name(candidate.kernel_name) == normalized;
    });
    if (pending == pending_.end())
      return;

    uint64_t kernel_object = 0;
    if (original_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object) !=
        HSA_STATUS_SUCCESS) {
      return;
    }
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.symbol == symbol.handle; });
    if (bound == bound_.end()) {
      bound_.push_back({symbol.handle, kernel_object, pending->required_private_bytes});
    } else {
      bound->kernel_object = kernel_object;
      bound->required_private_bytes =
          std::max(bound->required_private_bytes, pending->required_private_bytes);
    }
    log_message(kLogInfo,
                "ConSan dispatch-private binding executable=%llu symbol=%llu kernel_object=0x%llx "
                "private_bytes=%u",
                static_cast<unsigned long long>(executable.handle),
                static_cast<unsigned long long>(symbol.handle),
                static_cast<unsigned long long>(kernel_object), pending->required_private_bytes);
  }

  [[nodiscard]] std::optional<uint32_t> required_for_symbol(hsa_executable_symbol_t symbol) const {
    std::lock_guard lock(mutex_);
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.symbol == symbol.handle; });
    return bound == bound_.end() ? std::nullopt
                                 : std::optional<uint32_t>(bound->required_private_bytes);
  }

  [[nodiscard]] std::optional<uint32_t> required_for_kernel_object(uint64_t kernel_object) const {
    std::lock_guard lock(mutex_);
    const auto bound = std::ranges::find_if(
        bound_, [&](const Bound &candidate) { return candidate.kernel_object == kernel_object; });
    return bound == bound_.end() ? std::nullopt
                                 : std::optional<uint32_t>(bound->required_private_bytes);
  }

  void clear() {
    std::lock_guard lock(mutex_);
    pending_.clear();
    bound_.clear();
  }

private:
  struct Pending {
    uint64_t executable = 0;
    std::string kernel_name;
    uint32_t required_private_bytes = 0;
  };
  struct Bound {
    uint64_t symbol = 0;
    uint64_t kernel_object = 0;
    uint32_t required_private_bytes = 0;
  };

  [[nodiscard]] static std::string_view normalize_kernel_name(std::string_view name) {
    if (name.ends_with(".kd"))
      name.remove_suffix(3);
    return name;
  }

  mutable std::mutex mutex_;
  std::vector<Pending> pending_;
  std::vector<Bound> bound_;
};

hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);
hsa_status_t HSA_API rj_dbi_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);
hsa_status_t HSA_API rj_dbi_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);
hsa_status_t HSA_API rj_dbi_executable_get_symbol_by_name(hsa_executable_t executable,
                                                          const char *symbol_name,
                                                          const hsa_agent_t *agent,
                                                          hsa_executable_symbol_t *symbol);
hsa_status_t HSA_API rj_dbi_executable_symbol_get_info(hsa_executable_symbol_t symbol,
                                                       hsa_executable_symbol_info_t attribute,
                                                       void *value);
hsa_status_t HSA_API rj_dbi_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                         void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                         void *data, uint32_t private_segment_size,
                                         uint32_t group_segment_size, hsa_queue_t **queue);

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
    amd_ext_ = table->amd_ext_;
    g_log_level.store(config.log_level, std::memory_order_relaxed);
    config_ = config;
    original_create_from_file_ = core_->hsa_code_object_reader_create_from_file_fn;
    original_create_from_memory_ = core_->hsa_code_object_reader_create_from_memory_fn;
    original_destroy_ = core_->hsa_code_object_reader_destroy_fn;
    original_load_agent_code_object_ = core_->hsa_executable_load_agent_code_object_fn;
    original_get_symbol_by_name_ = core_->hsa_executable_get_symbol_by_name_fn;
    original_symbol_get_info_ = core_->hsa_executable_symbol_get_info_fn;
    original_queue_create_ = core_->hsa_queue_create_fn;
    intercept_dispatch_private_ =
        config.flavor.value_or(rocjitsu::ConSanFlavor::None) == rocjitsu::ConSanFlavor::Moi;
    const bool amd_intercept_table_valid =
        amd_ext_ != nullptr &&
        amd_ext_->version.minor_id >= offsetof(AmdExtTable, hsa_amd_queue_intercept_register_fn) +
                                          sizeof(AmdExtTable::hsa_amd_queue_intercept_register_fn);

    if (original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_load_agent_code_object_ == nullptr ||
        (intercept_dispatch_private_ &&
         (original_get_symbol_by_name_ == nullptr || original_symbol_get_info_ == nullptr ||
          original_queue_create_ == nullptr || !amd_intercept_table_valid ||
          amd_ext_->hsa_amd_queue_intercept_create_fn == nullptr ||
          amd_ext_->hsa_amd_queue_intercept_register_fn == nullptr))) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] HSA API table lacks a required DBI entry\n");
      clear_unlocked();
      return false;
    }

    core_->hsa_code_object_reader_create_from_file_fn = rj_dbi_code_object_reader_create_from_file;
    core_->hsa_code_object_reader_create_from_memory_fn =
        rj_dbi_code_object_reader_create_from_memory;
    core_->hsa_code_object_reader_destroy_fn = rj_dbi_code_object_reader_destroy;
    core_->hsa_executable_load_agent_code_object_fn = rj_dbi_executable_load_agent_code_object;
    if (intercept_dispatch_private_) {
      core_->hsa_executable_get_symbol_by_name_fn = rj_dbi_executable_get_symbol_by_name;
      core_->hsa_executable_symbol_get_info_fn = rj_dbi_executable_symbol_get_info;
      core_->hsa_queue_create_fn = rj_dbi_queue_create;
    }
    active_ = true;

    log_message(
        kLogInfo,
        "installed ConSan hook flavor=%s moi_engine=%s delay_nops=%u fail_closed=%s "
        "require_patch=%s "
        "probe_nop=%s probe_trampoline_nop=%s probe_endpgm=%s probe_lds_endpgm=%s "
        "check_trap_mode=%s probe_lds_check_trap=%s probe_flat_check_trap=%s probe_flat_trap=%s "
        "fault_drop_barrier=%s moi_init_owner_epoch=%s moi_track_barriers=%s "
        "moi_track_atomics=%s moi_dynamic_access_records=%s moi_require_records=%s "
        "moi_require_diagnostics=%s moi_forbid_diagnostics=%s "
        "moi_require_replay_conflict=%s "
        "fault_barrier_index=%u "
        "delay_mode=%s delay_var_ssrc=%u "
        "max_patches=%u tmp_vgpr=%s moi_exec_save_sgpr=%s "
        "moi_owner_source=%s moi_owner_sgpr=%s moi_owner_vgpr=%s moi_epoch_vgpr=%s "
        "moi_report_buffer=%s moi_report_buffer_size=%llu "
        "moi_auto_report_buffer_size=%llu mode=%s",
        flavor_name(config.flavor.value_or(rocjitsu::ConSanFlavor::None)),
        rocjitsu::consan_moi_engine_name(config.moi_engine), config.delay_nops,
        config.fail_closed ? "true" : "false", config.require_patch ? "true" : "false",
        config.probe_nop ? "true" : "false", config.probe_trampoline_nop ? "true" : "false",
        config.probe_endpgm ? "true" : "false", config.probe_lds_endpgm ? "true" : "false",
        check_trap_mode_name(config.check_trap_mode),
        config.probe_lds_check_trap ? "true" : "false",
        config.probe_flat_check_trap ? "true" : "false", config.probe_flat_trap ? "true" : "false",
        config.fault_drop_barrier ? "true" : "false",
        config.moi_init_owner_epoch ? "true" : "false",
        config.moi_track_barriers ? "true" : "false", config.moi_track_atomics ? "true" : "false",
        config.moi_dynamic_access_records ? "true" : "false",
        config.moi_require_records ? "true" : "false",
        config.moi_require_diagnostics ? "true" : "false",
        config.moi_forbid_diagnostics ? "true" : "false",
        config.moi_require_replay_conflict ? "true" : "false", config.fault_barrier_index,
        delay_mode_name(config.delay_mode), config.delay_var_ssrc, config.max_patches,
        config.scratch_vgpr ? std::to_string(*config.scratch_vgpr).c_str() : "auto",
        config.moi_exec_save_sgpr ? std::to_string(*config.moi_exec_save_sgpr).c_str() : "unset",
        owner_source_name(config.moi_owner_source),
        config.moi_owner_sgpr ? std::to_string(*config.moi_owner_sgpr).c_str() : "unset",
        config.moi_owner_vgpr ? std::to_string(*config.moi_owner_vgpr).c_str() : "unset",
        config.moi_epoch_vgpr ? std::to_string(*config.moi_epoch_vgpr).c_str() : "unset",
        config.moi_report_buffer_address ? std::to_string(*config.moi_report_buffer_address).c_str()
                                         : "disabled",
        static_cast<unsigned long long>(config.moi_report_buffer_size),
        static_cast<unsigned long long>(config.moi_auto_report_buffer_size),
        config.fault_drop_barrier
            ? (config.probe_lds_check_trap && config.probe_flat_check_trap
                   ? "proof-check-trap-all+fault-drop-barrier"
               : config.probe_lds_check_trap  ? "proof-lds-check-trap+fault-drop-barrier"
               : config.probe_flat_check_trap ? "proof-flat-check-trap+fault-drop-barrier"
                                              : "fault-drop-barrier")
        : config.probe_lds_check_trap && config.probe_flat_check_trap ? "proof-check-trap-all"
        : config.probe_lds_check_trap                                 ? "proof-lds-check-trap"
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
      if (core_->hsa_executable_get_symbol_by_name_fn == rj_dbi_executable_get_symbol_by_name)
        core_->hsa_executable_get_symbol_by_name_fn = original_get_symbol_by_name_;
      if (core_->hsa_executable_symbol_get_info_fn == rj_dbi_executable_symbol_get_info)
        core_->hsa_executable_symbol_get_info_fn = original_symbol_get_info_;
      if (core_->hsa_queue_create_fn == rj_dbi_queue_create)
        core_->hsa_queue_create_fn = original_queue_create_;
    }

    const AutoMoiReportBufferRegistry::Summary moi_report_summary =
        AutoMoiReportBufferRegistry::instance().summarize_and_clear(core_);
    const bool moi_require_records = config_ && config_->moi_require_records;
    const bool moi_require_diagnostics = config_ && config_->moi_require_diagnostics;
    const bool moi_forbid_diagnostics = config_ && config_->moi_forbid_diagnostics;
    const bool moi_require_replay_conflict = config_ && config_->moi_require_replay_conflict;
    CodeObjectReaderRegistry::instance().clear();
    KernelPrivateDispatchRegistry::instance().clear();
    clear_unlocked();
    if (moi_require_records && moi_report_summary.visible_access_record_count +
                                       moi_report_summary.visible_barrier_record_count +
                                       moi_report_summary.visible_atomic_record_count +
                                       moi_report_summary.visible_diagnostic_record_count +
                                       moi_report_summary.visible_exact_shadow_entry_count +
                                       moi_report_summary.visible_sampled_watchpoint_count ==
                                   0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but %llu auto "
                   "MOI report buffer(s) contained zero visible "
                   "access/barrier/atomic/diagnostic/exact-shadow/sampled records\n",
                   static_cast<unsigned long long>(moi_report_summary.buffer_count));
      std::fflush(stderr);
      std::_Exit(86);
    }
    if (moi_require_replay_conflict && moi_report_summary.replay_conflict_count == 0) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT requested, but "
          "%llu auto MOI report buffer(s) produced zero replay conflicts "
          "(visible access=%llu barrier=%llu atomic=%llu diagnostics=%llu sampled=%llu, replay "
          "diagnostics=%llu)\n",
          static_cast<unsigned long long>(moi_report_summary.buffer_count),
          static_cast<unsigned long long>(moi_report_summary.visible_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_sampled_watchpoint_count),
          static_cast<unsigned long long>(moi_report_summary.replay_diagnostic_count));
      std::fflush(stderr);
      std::_Exit(87);
    }
    const bool moi_has_diagnostics = moi_report_summary.visible_diagnostic_record_count +
                                         moi_report_summary.replay_diagnostic_count +
                                         moi_report_summary.sampled_conflict_count >
                                     0;
    if (moi_require_diagnostics && !moi_has_diagnostics) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS requested, but "
          "%llu auto MOI report buffer(s) produced zero visible/replay diagnostics or sampled "
          "conflicts "
          "(visible access=%llu barrier=%llu atomic=%llu diagnostics=%llu "
          "exact-shadow=%llu sampled=%llu, replay diagnostics=%llu sampled_conflicts=%llu)\n",
          static_cast<unsigned long long>(moi_report_summary.buffer_count),
          static_cast<unsigned long long>(moi_report_summary.visible_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_exact_shadow_entry_count),
          static_cast<unsigned long long>(moi_report_summary.visible_sampled_watchpoint_count),
          static_cast<unsigned long long>(moi_report_summary.replay_diagnostic_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_conflict_count));
      std::fflush(stderr);
      std::_Exit(88);
    }
    if (moi_forbid_diagnostics && moi_has_diagnostics) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] RJ_CONSAN_MOI_FORBID_DIAGNOSTICS requested, but "
          "%llu auto MOI report buffer(s) produced visible/replay diagnostics or sampled "
          "conflicts "
          "(visible access=%llu barrier=%llu atomic=%llu diagnostics=%llu "
          "exact-shadow=%llu sampled=%llu, replay diagnostics=%llu sampled_conflicts=%llu)\n",
          static_cast<unsigned long long>(moi_report_summary.buffer_count),
          static_cast<unsigned long long>(moi_report_summary.visible_access_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_barrier_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_atomic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_diagnostic_record_count),
          static_cast<unsigned long long>(moi_report_summary.visible_exact_shadow_entry_count),
          static_cast<unsigned long long>(moi_report_summary.visible_sampled_watchpoint_count),
          static_cast<unsigned long long>(moi_report_summary.replay_diagnostic_count),
          static_cast<unsigned long long>(moi_report_summary.sampled_conflict_count));
      std::fflush(stderr);
      std::_Exit(89);
    }
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

  [[nodiscard]] decltype(hsa_executable_get_symbol_by_name) *get_symbol_by_name() const {
    std::lock_guard lock(mutex_);
    return original_get_symbol_by_name_;
  }

  [[nodiscard]] decltype(hsa_executable_symbol_get_info) *symbol_get_info() const {
    std::lock_guard lock(mutex_);
    return original_symbol_get_info_;
  }

  [[nodiscard]] decltype(hsa_queue_create) *queue_create() const {
    std::lock_guard lock(mutex_);
    return original_queue_create_;
  }

  [[nodiscard]] hsa_amd_queue_intercept_create_fn_t queue_intercept_create() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_intercept_create_fn;
  }

  [[nodiscard]] hsa_amd_queue_intercept_register_fn_t queue_intercept_register() const {
    std::lock_guard lock(mutex_);
    return amd_ext_ == nullptr ? nullptr : amd_ext_->hsa_amd_queue_intercept_register_fn;
  }

  [[nodiscard]] CoreApiTable *core_table() const {
    std::lock_guard lock(mutex_);
    return core_;
  }

private:
  [[nodiscard]] static bool validate_table(HsaApiTable *table) {
    if (table == nullptr || table->core_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid HSA API table passed to OnLoad\n");
      return false;
    }

    constexpr size_t required_size =
        std::max({offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
                      sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn),
                  offsetof(CoreApiTable, hsa_executable_get_symbol_by_name_fn) +
                      sizeof(CoreApiTable::hsa_executable_get_symbol_by_name_fn),
                  offsetof(CoreApiTable, hsa_executable_symbol_get_info_fn) +
                      sizeof(CoreApiTable::hsa_executable_symbol_get_info_fn)});
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
    amd_ext_ = nullptr;
    config_.reset();
    original_create_from_file_ = nullptr;
    original_create_from_memory_ = nullptr;
    original_destroy_ = nullptr;
    original_load_agent_code_object_ = nullptr;
    original_get_symbol_by_name_ = nullptr;
    original_symbol_get_info_ = nullptr;
    original_queue_create_ = nullptr;
    intercept_dispatch_private_ = false;
  }

  mutable std::mutex mutex_;
  HsaApiTable *table_ = nullptr;
  CoreApiTable *core_ = nullptr;
  AmdExtTable *amd_ext_ = nullptr;
  std::optional<HookConfig> config_;
  bool active_ = false;
  decltype(hsa_code_object_reader_create_from_file) *original_create_from_file_ = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *original_create_from_memory_ = nullptr;
  decltype(hsa_code_object_reader_destroy) *original_destroy_ = nullptr;
  decltype(hsa_executable_load_agent_code_object) *original_load_agent_code_object_ = nullptr;
  decltype(hsa_executable_get_symbol_by_name) *original_get_symbol_by_name_ = nullptr;
  decltype(hsa_executable_symbol_get_info) *original_symbol_get_info_ = nullptr;
  decltype(hsa_queue_create) *original_queue_create_ = nullptr;
  bool intercept_dispatch_private_ = false;
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

struct alignas(8) InterceptPacket {
  std::array<uint8_t, sizeof(hsa_kernel_dispatch_packet_t)> bytes{};
};
static_assert(sizeof(InterceptPacket) == sizeof(hsa_kernel_dispatch_packet_t));
static_assert(sizeof(InterceptPacket) == 64);

void rj_dbi_queue_write_interceptor(const void *packets, uint64_t packet_count,
                                    uint64_t user_packet_index, void *data,
                                    hsa_amd_queue_intercept_packet_writer_t writer) {
  (void)user_packet_index;
  (void)data;
  if (packets == nullptr || writer == nullptr || packet_count == 0) {
    if (writer != nullptr)
      writer(packets, packet_count);
    return;
  }
  if (packet_count > std::numeric_limits<size_t>::max() / sizeof(InterceptPacket)) {
    writer(packets, packet_count);
    return;
  }

  std::vector<InterceptPacket> rewritten(static_cast<size_t>(packet_count));
  std::memcpy(rewritten.data(), packets,
              static_cast<size_t>(packet_count) * sizeof(InterceptPacket));
  for (InterceptPacket &packet_bytes : rewritten) {
    auto *packet = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(packet_bytes.bytes.data());
    const uint16_t type =
        static_cast<uint16_t>((packet->header >> HSA_PACKET_HEADER_TYPE) &
                              ((uint16_t{1} << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u));
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
      continue;
    const auto required =
        KernelPrivateDispatchRegistry::instance().required_for_kernel_object(packet->kernel_object);
    if (!required || *required <= packet->private_segment_size)
      continue;
    log_message(kLogInfo, "ConSan dispatch-private grow kernel_object=0x%llx private_bytes=%u->%u",
                static_cast<unsigned long long>(packet->kernel_object),
                packet->private_segment_size, *required);
    packet->private_segment_size = *required;
  }
  writer(rewritten.data(), packet_count);
}

hsa_status_t HSA_API rj_dbi_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                         void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                         void *data, uint32_t private_segment_size,
                                         uint32_t group_segment_size, hsa_queue_t **queue) {
  auto *intercept_create = layer().queue_intercept_create();
  auto *intercept_register = layer().queue_intercept_register();
  if (intercept_create == nullptr || intercept_register == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t create_status = intercept_create(
      agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
  if (create_status != HSA_STATUS_SUCCESS || queue == nullptr || *queue == nullptr)
    return create_status;
  const hsa_status_t register_status =
      intercept_register(*queue, rj_dbi_queue_write_interceptor, nullptr);
  if (register_status != HSA_STATUS_SUCCESS) {
    CoreApiTable *core = layer().core_table();
    if (core != nullptr && core->hsa_queue_destroy_fn != nullptr)
      (void)core->hsa_queue_destroy_fn(*queue);
    *queue = nullptr;
  }
  return register_status;
}

hsa_status_t HSA_API rj_dbi_executable_get_symbol_by_name(hsa_executable_t executable,
                                                          const char *symbol_name,
                                                          const hsa_agent_t *agent,
                                                          hsa_executable_symbol_t *symbol) {
  auto *original = layer().get_symbol_by_name();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(executable, symbol_name, agent, symbol);
  if (status == HSA_STATUS_SUCCESS && symbol_name != nullptr && symbol != nullptr) {
    KernelPrivateDispatchRegistry::instance().bind_symbol(executable, symbol_name, *symbol,
                                                          layer().symbol_get_info());
  }
  return status;
}

hsa_status_t HSA_API rj_dbi_executable_symbol_get_info(hsa_executable_symbol_t symbol,
                                                       hsa_executable_symbol_info_t attribute,
                                                       void *value) {
  auto *original = layer().symbol_get_info();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  const hsa_status_t status = original(symbol, attribute, value);
  if (status == HSA_STATUS_SUCCESS && value != nullptr &&
      attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) {
    const auto required = KernelPrivateDispatchRegistry::instance().required_for_symbol(symbol);
    if (required) {
      auto *private_bytes = static_cast<uint32_t *>(value);
      *private_bytes = std::max(*private_bytes, *required);
    }
  }
  return status;
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
  if (!refresh_report_config_from_env(&*config))
    return HSA_STATUS_ERROR;

  hsa_code_object_reader_t reader_to_load = code_object_reader;
  hsa_code_object_reader_t replacement_reader{};
  bool using_replacement_reader = false;
  std::optional<rocjitsu::ConSanResult> patch_result_storage;

  const uint8_t *bytes = nullptr;
  size_t size = 0;
  if (CodeObjectReaderRegistry::instance().lookup(code_object_reader, &bytes, &size)) {
    const uint64_t dump_id =
        config->dump_dir.empty() ? 0 : g_dump_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    dump_code_object_bytes(*config, dump_id, code_object_reader.handle, "original",
                           std::span<const uint8_t>(bytes, size));

    rocjitsu::ConSanOptions patch_options;
    patch_options.flavor = config->flavor.value_or(rocjitsu::ConSanFlavor::None);
    patch_options.moi_engine = config->moi_engine;
    patch_options.moi_owner_source = config->moi_owner_source;
    patch_options.fail_closed = config->fail_closed;
    patch_options.probe_nop = config->probe_nop;
    patch_options.probe_trampoline_nop = config->probe_trampoline_nop;
    patch_options.probe_endpgm = config->probe_endpgm;
    patch_options.probe_lds_endpgm = config->probe_lds_endpgm;
    patch_options.probe_lds_check_trap = config->probe_lds_check_trap;
    patch_options.probe_flat_check_trap = config->probe_flat_check_trap;
    patch_options.probe_flat_trap = config->probe_flat_trap;
    patch_options.fault_drop_barrier = config->fault_drop_barrier;
    patch_options.moi_init_owner_epoch = config->moi_init_owner_epoch;
    patch_options.moi_track_barriers = config->moi_track_barriers;
    patch_options.moi_track_atomics = config->moi_track_atomics;
    patch_options.moi_dynamic_access_records = config->moi_dynamic_access_records;
    patch_options.force_vgpr_spill = config->test_force_vgpr_spill;
    patch_options.test_kernel_name_filter = config->test_kernel_name_filter;
    patch_options.fault_barrier_index = config->fault_barrier_index;
    patch_options.delay_mode = config->delay_mode;
    patch_options.delay_var_ssrc = config->delay_var_ssrc;
    patch_options.scratch_vgpr = config->scratch_vgpr;
    patch_options.moi_exec_save_sgpr = config->moi_exec_save_sgpr;
    patch_options.moi_owner_sgpr = config->moi_owner_sgpr;
    patch_options.moi_owner_vgpr = config->moi_owner_vgpr;
    patch_options.moi_epoch_vgpr = config->moi_epoch_vgpr;
    patch_options.report_buffer_address = config->report_buffer_address;
    patch_options.moi_report_buffer_address = config->moi_report_buffer_address;
    patch_options.moi_report_buffer_size = config->moi_report_buffer_size;
    patch_options.delay_nops = config->delay_nops;
    patch_options.max_patches = config->max_patches;
    patch_options.moi_sample_stride = config->moi_sample_stride;
    patch_options.moi_sample_offset = config->moi_sample_offset;
    patch_options.report_marker = config->report_marker;
    if (patch_options.flavor == rocjitsu::ConSanFlavor::Moi &&
        !patch_options.moi_report_buffer_address && config->moi_auto_report_buffer_size != 0) {
      uint64_t auto_report_address = 0;
      uint64_t auto_report_size = 0;
      const bool direct_sampled = config->moi_engine == rocjitsu::ConSanMoiEngine::Sampled;
      const bool inline_shadow = config->moi_engine == rocjitsu::ConSanMoiEngine::InlineShadow;
      if (AutoMoiReportBufferRegistry::instance().allocate(
              layer().core_table(), agent, code_object_reader.handle,
              config->moi_auto_report_buffer_size, direct_sampled, inline_shadow,
              config->moi_track_barriers, config->moi_track_atomics, &auto_report_address,
              &auto_report_size)) {
        patch_options.moi_report_buffer_address = auto_report_address;
        patch_options.moi_report_buffer_size = auto_report_size;
      } else if (config->fail_closed) {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }
    }

    log_message(kLogInfo, "ConSan patch begin reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader.handle), size);
    patch_result_storage =
        rocjitsu::try_patch_consan(std::span<const uint8_t>(bytes, size), patch_options);
    const rocjitsu::ConSanResult &patch_result = *patch_result_storage;
    log_message(kLogInfo,
                "ConSan patch end reader=%llu visited=%s modified=%s errors=%zu "
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
        "ConSan inventory reader=%llu flavor=%s moi_engine=%s bytes=%zu visited=%s modified=%s "
        "delay_nops=%u fail_closed=%s probe_nop=%s probe_trampoline_nop=%s "
        "probe_endpgm=%s probe_lds_endpgm=%s check_trap_mode=%s probe_lds_check_trap=%s "
        "probe_flat_check_trap=%s probe_flat_trap=%s fault_drop_barrier=%s "
        "moi_init_owner_epoch=%s moi_track_barriers=%s moi_track_atomics=%s "
        "moi_dynamic_access_records=%s "
        "fault_barrier_index=%u "
        "delay_mode=%s delay_var_ssrc=%u "
        "max_patches=%u tmp_vgpr=%s moi_exec_save_sgpr=%s "
        "moi_owner_source=%s moi_owner_sgpr=%s moi_owner_vgpr=%s moi_epoch_vgpr=%s "
        "report_buffer=%s report_marker=%u "
        "moi_report_buffer=%s moi_report_buffer_size=%llu "
        "moi_auto_report_buffer_size=%llu require_patch=%s",
        static_cast<unsigned long long>(code_object_reader.handle),
        flavor_name(patch_options.flavor),
        rocjitsu::consan_moi_engine_name(patch_options.moi_engine), patch_result.input_size,
        patch_result.visited_code_object ? "true" : "false",
        patch_result.modified ? "true" : "false", config->delay_nops,
        config->fail_closed ? "true" : "false", config->probe_nop ? "true" : "false",
        config->probe_trampoline_nop ? "true" : "false", config->probe_endpgm ? "true" : "false",
        config->probe_lds_endpgm ? "true" : "false", check_trap_mode_name(config->check_trap_mode),
        config->probe_lds_check_trap ? "true" : "false",
        config->probe_flat_check_trap ? "true" : "false",
        config->probe_flat_trap ? "true" : "false", config->fault_drop_barrier ? "true" : "false",
        config->moi_init_owner_epoch ? "true" : "false",
        config->moi_track_barriers ? "true" : "false", config->moi_track_atomics ? "true" : "false",
        config->moi_dynamic_access_records ? "true" : "false", config->fault_barrier_index,
        delay_mode_name(config->delay_mode), config->delay_var_ssrc, config->max_patches,
        config->scratch_vgpr ? std::to_string(*config->scratch_vgpr).c_str() : "auto",
        config->moi_exec_save_sgpr ? std::to_string(*config->moi_exec_save_sgpr).c_str() : "unset",
        owner_source_name(config->moi_owner_source),
        config->moi_owner_sgpr ? std::to_string(*config->moi_owner_sgpr).c_str() : "unset",
        config->moi_owner_vgpr ? std::to_string(*config->moi_owner_vgpr).c_str() : "unset",
        config->moi_epoch_vgpr ? std::to_string(*config->moi_epoch_vgpr).c_str() : "unset",
        config->report_buffer_address ? std::to_string(*config->report_buffer_address).c_str()
                                      : "disabled",
        config->report_marker,
        patch_options.moi_report_buffer_address
            ? std::to_string(*patch_options.moi_report_buffer_address).c_str()
            : "disabled",
        static_cast<unsigned long long>(patch_options.moi_report_buffer_size),
        static_cast<unsigned long long>(config->moi_auto_report_buffer_size),
        config->require_patch ? "true" : "false");
    if (!patch_result.target_name.empty()) {
      log_message(kLogInfo,
                  "ConSan code-object reader=%llu target=%s arch=%s text_sections=%zu "
                  "kernels=%zu functions=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result.target_name.c_str(), patch_result.arch_name.c_str(),
                  patch_result.text_sections.size(), patch_result.kernels.size(),
                  patch_result.functions.size());
    }
    if (patch_options.flavor == rocjitsu::ConSanFlavor::Moi) {
      log_message(kLogInfo, "ConSan MOI inventory reader=%llu candidates=%zu",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result.moi_candidates.size());
      for (const rocjitsu::ConSanMoiCandidate &candidate : patch_result.moi_candidates) {
        log_message(kLogDebug,
                    "ConSan MOI candidate reader=%llu source=%s container=%s container_kind=%s "
                    "kind=%s mnemonic=%s text_offset=0x%llx file_offset=0x%llx size=%u "
                    "width_bits=%u dst_vgpr=%s addr_vgpr=%s data_vgpr=%s flat_hint=%s "
                    "raw_saddr=%s raw_vaddr=%s raw_vsrc=%s raw_vdst=%s raw_ioffset=%s "
                    "raw_scope=%s raw_th=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    moi_candidate_source_name(candidate.source), candidate.container_name.c_str(),
                    candidate.in_kernel ? "kernel" : "function",
                    lds_access_kind_name(candidate.kind), candidate.mnemonic.c_str(),
                    static_cast<unsigned long long>(candidate.text_offset),
                    static_cast<unsigned long long>(candidate.file_offset), candidate.size,
                    candidate.width_bits,
                    candidate.dst_vgpr ? std::to_string(*candidate.dst_vgpr).c_str() : "-",
                    candidate.addr_vgpr ? std::to_string(*candidate.addr_vgpr).c_str() : "-",
                    candidate.data_vgpr ? std::to_string(*candidate.data_vgpr).c_str() : "-",
                    flat_address_space_hint_name(candidate.flat_address_space_hint),
                    candidate.raw_saddr ? std::to_string(*candidate.raw_saddr).c_str() : "-",
                    candidate.raw_vaddr ? std::to_string(*candidate.raw_vaddr).c_str() : "-",
                    candidate.raw_vsrc ? std::to_string(*candidate.raw_vsrc).c_str() : "-",
                    candidate.raw_vdst ? std::to_string(*candidate.raw_vdst).c_str() : "-",
                    candidate.raw_ioffset ? std::to_string(*candidate.raw_ioffset).c_str() : "-",
                    candidate.raw_scope ? std::to_string(*candidate.raw_scope).c_str() : "-",
                    candidate.raw_th ? std::to_string(*candidate.raw_th).c_str() : "-");
      }
      for (const rocjitsu::ConSanCandidateResourcePlan &plan : patch_result.resource_plans) {
        log_message(
            kLogInfo,
            "ConSan MOI resource reader=%llu candidate=%zu text_offset=0x%llx "
            "source=%u reason=%u owners=%zu scratch_vgpr=%s scratch_count=%u "
            "current_vgprs=%u max_referenced_vgprs=%u required_vgprs=%u "
            "private_bytes=%u",
            static_cast<unsigned long long>(code_object_reader.handle), plan.candidate_index,
            static_cast<unsigned long long>(plan.text_offset), static_cast<unsigned>(plan.source),
            static_cast<unsigned>(plan.reason), plan.owner_descriptor_file_offsets.size(),
            plan.scratch_vgpr ? std::to_string(*plan.scratch_vgpr).c_str() : "-",
            plan.scratch_vgpr_count, plan.current_vgpr_count, plan.max_referenced_vgpr_count,
            plan.required_vgpr_count, plan.original_private_segment_size);
      }
      if (patch_result.resolved_moi_owner_vgpr || patch_result.resolved_moi_epoch_vgpr) {
        log_message(kLogInfo,
                    "ConSan MOI persistent reader=%llu owner_vgpr=%s epoch_vgpr=%s automatic=%s",
                    static_cast<unsigned long long>(code_object_reader.handle),
                    patch_result.resolved_moi_owner_vgpr
                        ? std::to_string(*patch_result.resolved_moi_owner_vgpr).c_str()
                        : "-",
                    patch_result.resolved_moi_epoch_vgpr
                        ? std::to_string(*patch_result.resolved_moi_epoch_vgpr).c_str()
                        : "-",
                    patch_result.moi_persistent_vgprs_automatic ? "true" : "false");
      }
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
    for (const rocjitsu::ConSanKernelInfo &kernel : patch_result.kernels) {
      switch (kernel.preflight_action) {
      case rocjitsu::ConSanPreflightAction::Candidate:
        ++candidate_kernel_count;
        break;
      case rocjitsu::ConSanPreflightAction::Skip:
        ++skipped_kernel_count;
        break;
      case rocjitsu::ConSanPreflightAction::Reject:
        ++rejected_kernel_count;
        break;
      case rocjitsu::ConSanPreflightAction::NotRun:
        break;
      }
      for (const rocjitsu::ConSanLdsSite &site : kernel.lds_sites) {
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
    for (const rocjitsu::ConSanFunctionInfo &function : patch_result.functions) {
      function_lds_site_count += function.lds_sites.size();
      for (const rocjitsu::ConSanLdsSite &site : function.lds_sites) {
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
                "ConSan summary reader=%llu kernels=%zu candidates=%zu skips=%zu "
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
    for (const rocjitsu::ConSanTextSection &text : patch_result.text_sections) {
      log_message(kLogVerbose, "ConSan text reader=%llu name=%s file=0x%llx vaddr=0x%llx size=%llu",
                  static_cast<unsigned long long>(code_object_reader.handle), text.name.c_str(),
                  static_cast<unsigned long long>(text.file_offset),
                  static_cast<unsigned long long>(text.virtual_address),
                  static_cast<unsigned long long>(text.size));
    }
    for (const rocjitsu::ConSanKernelInfo &kernel : patch_result.kernels) {
      if (kernel.has_text_range) {
        log_message(
            kLogInfo,
            "ConSan kernel reader=%llu name=%s kd_file=0x%llx "
            "text_file=0x%llx entry_text=0x%llx code_size=%llu decoded=%s "
            "dynamic_stack=%s "
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
            static_cast<unsigned long long>(kernel.code_size), kernel.decoded ? "true" : "false",
            kernel.uses_dynamic_stack ? (*kernel.uses_dynamic_stack ? "true" : "false") : "unknown",
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
                    "ConSan kernel reader=%llu name=%s kd_file=0x%llx "
                    "text_range=unavailable decoded=%s dynamic_stack=%s decode_errors=%llu "
                    "preflight=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    static_cast<unsigned long long>(kernel.descriptor_file_offset),
                    kernel.decoded ? "true" : "false",
                    kernel.uses_dynamic_stack ? (*kernel.uses_dynamic_stack ? "true" : "false")
                                              : "unknown",
                    static_cast<unsigned long long>(kernel.stats.decode_error_count),
                    preflight_action_name(kernel.preflight_action));
      }
      for (const std::string &reason : kernel.preflight_reasons) {
        log_message(kLogInfo, "ConSan preflight reader=%llu kernel=%s action=%s reason=%s",
                    static_cast<unsigned long long>(code_object_reader.handle), kernel.name.c_str(),
                    preflight_action_name(kernel.preflight_action), reason.c_str());
      }
      for (const rocjitsu::ConSanLdsSite &site : kernel.lds_sites) {
        log_message(kLogVerbose,
                    "ConSan lds-site reader=%llu kernel=%s kind=%s supported=%s "
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
      for (const rocjitsu::ConSanFlatSite &site : kernel.flat_sites) {
        log_message(kLogVerbose,
                    "ConSan flat-site reader=%llu kernel=%s kind=%s "
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
    for (const rocjitsu::ConSanFunctionInfo &function : patch_result.functions) {
      log_message(kLogVerbose,
                  "ConSan function reader=%llu name=%s text_file=0x%llx "
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
      for (const rocjitsu::ConSanLdsSite &site : function.lds_sites) {
        log_message(kLogVerbose,
                    "ConSan function-lds-site reader=%llu function=%s kind=%s "
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
      for (const rocjitsu::ConSanFlatSite &site : function.flat_sites) {
        log_message(kLogVerbose,
                    "ConSan function-flat-site reader=%llu function=%s kind=%s "
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
    for (const rocjitsu::ConSanPatchInfo &patch : patch_result.patches) {
      const std::string scratch_vgpr =
          patch.scratch_vgpr ? std::to_string(*patch.scratch_vgpr) : "-";
      log_message(kLogInfo,
                  "ConSan proof patch reader=%llu kind=%s anchor=0x%llx "
                  "trampoline=0x%llx original_size=%u scratch_vgpr=%s "
                  "spilled_vgprs=%u private_bytes=%u",
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_kind_name(patch.kind), static_cast<unsigned long long>(patch.anchor_offset),
                  static_cast<unsigned long long>(patch.trampoline_offset), patch.original_size,
                  scratch_vgpr.c_str(), patch.spilled_vgpr_count,
                  patch.required_private_segment_size);
    }
    if (config->require_patch && !patch_result.modified) {
      const bool required = (patch_options.flavor == rocjitsu::ConSanFlavor::SuperCollider &&
                             require_patch_applies_to(patch_result, *config)) ||
                            (patch_options.flavor == rocjitsu::ConSanFlavor::Moi &&
                             require_moi_patch_applies_to(patch_result));
      if (required) {
        std::fprintf(stderr,
                     "[rocjitsu-dbi-hooks] RJ_CONSAN_REQUIRE_PATCH requested, but no patch was "
                     "applied to a code object with supported ConSan sites\n");
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
    }

    if (patch_result.modified && !patch_result.elf_bytes.empty()) {
      dump_code_object_bytes(
          *config, dump_id, code_object_reader.handle, "patched",
          std::span<const uint8_t>(patch_result.elf_bytes.data(), patch_result.elf_bytes.size()));
    }
  } else {
    log_message(kLogInfo, "ConSan pass-through reader=%llu bytes=unavailable",
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
      log_message(kLogInfo, "ConSan replacement reader=%llu original_reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(replacement_reader.handle),
                  static_cast<unsigned long long>(code_object_reader.handle),
                  patch_result_storage->elf_bytes.size());
    }
  }

  const hsa_status_t load_status =
      original_load(executable, agent, reader_to_load, options, loaded_code_object);
  if (load_status == HSA_STATUS_SUCCESS && patch_result_storage && patch_result_storage->modified) {
    KernelPrivateDispatchRegistry::instance().note_patch_requirements(executable,
                                                                      *patch_result_storage);
  }
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
  if (!config->flavor) {
    log_message(kLogInfo, "RJ_CONSAN_FLAVOR is unset; not installing wrappers");
    return true;
  }

  return layer().install(table, *config);
}

extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
