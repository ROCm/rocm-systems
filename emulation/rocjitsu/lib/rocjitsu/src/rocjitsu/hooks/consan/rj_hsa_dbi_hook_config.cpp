// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace rocjitsu::consan_hook {

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
  if (!std::isdigit(static_cast<unsigned char>(*value))) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected uint32\n", name, value);
    return false;
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

[[nodiscard]] bool parse_optional_i32_env(const char *name, std::optional<int32_t> *out) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    out->reset();
    return true;
  }

  errno = 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max()) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected int32\n", name, value);
    return false;
  }
  *out = static_cast<int32_t>(parsed);
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

[[nodiscard]] bool parse_u64_env(const char *name, uint64_t default_value, uint64_t *out,
                                 int base = 0) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    *out = default_value;
    return true;
  }
  if (!std::isdigit(static_cast<unsigned char>(*value))) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected uint64\n", name, value);
    return false;
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, base);
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

[[nodiscard]] bool parse_barrier_move_direction_env(rocjitsu::ConSanBarrierMoveDirection *out) {
  const char *value = std::getenv("RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION");
  if (value == nullptr || *value == '\0' || std::strcmp(value, "legacy-marker") == 0 ||
      std::strcmp(value, "legacy_marker") == 0) {
    *out = rocjitsu::ConSanBarrierMoveDirection::LegacyMarker;
    return true;
  }
  if (std::strcmp(value, "earlier") == 0) {
    *out = rocjitsu::ConSanBarrierMoveDirection::Earlier;
    return true;
  }
  if (std::strcmp(value, "later") == 0) {
    *out = rocjitsu::ConSanBarrierMoveDirection::Later;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION='%s'; "
               "expected legacy-marker, earlier, or later\n",
               value);
  return false;
}

[[nodiscard]] bool parse_sc_perturb_kind_env(rocjitsu::ConSanPerturbationKind *out) {
  const char *value = std::getenv("RJ_CONSAN_SC_PERTURB_KIND");
  if (value == nullptr || *value == '\0' || ascii_iequals(value, "none")) {
    *out = rocjitsu::ConSanPerturbationKind::None;
    return true;
  }
  if (ascii_iequals(value, "barrier")) {
    *out = rocjitsu::ConSanPerturbationKind::Barrier;
    return true;
  }
  if (ascii_iequals(value, "atomic")) {
    *out = rocjitsu::ConSanPerturbationKind::Atomic;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_SC_PERTURB_KIND='%s'; "
               "expected none, barrier, or atomic\n",
               value);
  return false;
}

[[nodiscard]] bool parse_sc_perturb_edge_env(rocjitsu::ConSanPerturbationEdge *out) {
  const char *value = std::getenv("RJ_CONSAN_SC_PERTURB_EDGE");
  if (value == nullptr || *value == '\0' || ascii_iequals(value, "release")) {
    *out = rocjitsu::ConSanPerturbationEdge::Release;
    return true;
  }
  if (ascii_iequals(value, "acquire")) {
    *out = rocjitsu::ConSanPerturbationEdge::Acquire;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_SC_PERTURB_EDGE='%s'; "
               "expected release or acquire\n",
               value);
  return false;
}

[[nodiscard]] bool parse_atomic_order_edge_env(rocjitsu::ConSanAtomicOrderEdge *out) {
  const char *value = std::getenv("RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE");
  if (value == nullptr || *value == '\0' || ascii_iequals(value, "any")) {
    *out = rocjitsu::ConSanAtomicOrderEdge::Any;
    return true;
  }
  if (ascii_iequals(value, "release")) {
    *out = rocjitsu::ConSanAtomicOrderEdge::Release;
    return true;
  }
  if (ascii_iequals(value, "acquire")) {
    *out = rocjitsu::ConSanAtomicOrderEdge::Acquire;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE='%s'; "
               "expected any, release, or acquire\n",
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
    *out = rocjitsu::ConSanMoiOwnerSource::Automatic;
    return true;
  }
  const std::string_view mode(value);
  if (mode == "automatic" || mode == "auto") {
    *out = rocjitsu::ConSanMoiOwnerSource::Automatic;
    return true;
  }
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
               "expected automatic|workitem_id|hw_id\n",
               value);
  return false;
}

[[nodiscard]] bool parse_flat_provenance_mode_env(rocjitsu::ConSanFlatProvenanceMode *out) {
  const char *value = std::getenv("RJ_CONSAN_FLAT_PROVENANCE");
  if (value == nullptr || *value == '\0' || ascii_iequals(value, "likely") ||
      ascii_iequals(value, "default")) {
    *out = rocjitsu::ConSanFlatProvenanceMode::Likely;
    return true;
  }
  if (ascii_iequals(value, "strict") || ascii_iequals(value, "group")) {
    *out = rocjitsu::ConSanFlatProvenanceMode::Strict;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_FLAT_PROVENANCE='%s'; "
               "expected likely|strict\n",
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

[[nodiscard]] bool parse_sc_report_mode_env(ScReportMode *out) {
  const char *value = std::getenv("RJ_CONSAN_SC_REPORT_MODE");
  if (value == nullptr || *value == '\0' || ascii_iequals(value, "auto") ||
      ascii_iequals(value, "default")) {
    *out = ScReportMode::Auto;
    return true;
  }
  if (ascii_iequals(value, "trap")) {
    *out = ScReportMode::Trap;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_SC_REPORT_MODE='%s'; "
               "expected auto|trap\n",
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
         config.probe_lds_endpgm || config.probe_lds_check_trap || config.probe_flat_check_trap ||
         config.probe_flat_trap;
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

void warn_deprecated_env(const char *name, const char *replacement) {
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] warning: %s is deprecated; use %s instead\n", name,
               replacement);
}

[[nodiscard]] bool parse_mode_env(HookConfig *config) {
  const char *value = std::getenv("RJ_CONSAN_MODE");
  if (value == nullptr || *value == '\0') {
    if (env_has_value("RJ_CONSAN_FLAVOR"))
      warn_deprecated_env("RJ_CONSAN_FLAVOR", "RJ_CONSAN_MODE");
    if (env_has_value("RJ_CONSAN_MOI_ENGINE"))
      warn_deprecated_env("RJ_CONSAN_MOI_ENGINE", "RJ_CONSAN_MODE");
    if (env_has_value("RJ_CONSAN_MOI_BACKEND"))
      warn_deprecated_env("RJ_CONSAN_MOI_BACKEND", "RJ_CONSAN_MODE");

    if (!parse_flavor_env(&config->flavor))
      return false;
    if (!config->flavor)
      config->flavor = rocjitsu::ConSanFlavor::Moi;
    return parse_moi_engine_env(&config->moi_engine);
  }

  constexpr const char *kLegacySelectionVariables[] = {
      "RJ_CONSAN_FLAVOR",
      "RJ_CONSAN_MOI_ENGINE",
      "RJ_CONSAN_MOI_BACKEND",
  };
  for (const char *name : kLegacySelectionVariables) {
    if (env_has_value(name)) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] %s and RJ_CONSAN_MODE cannot both be set\n", name);
      return false;
    }
  }

  if (ascii_iequals(value, "record-replay") || ascii_iequals(value, "record_replay")) {
    config->flavor = rocjitsu::ConSanFlavor::Moi;
    config->moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
    return true;
  }
  if (ascii_iequals(value, "inline-shadow") || ascii_iequals(value, "inline_shadow")) {
    config->flavor = rocjitsu::ConSanFlavor::Moi;
    config->moi_engine = rocjitsu::ConSanMoiEngine::InlineShadow;
    return true;
  }
  if (ascii_iequals(value, "sampled")) {
    config->flavor = rocjitsu::ConSanFlavor::Moi;
    config->moi_engine = rocjitsu::ConSanMoiEngine::Sampled;
    return true;
  }
  if (ascii_iequals(value, "supercollider")) {
    config->flavor = rocjitsu::ConSanFlavor::SuperCollider;
    config->moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MODE='%s'; expected "
               "record-replay|inline-shadow|sampled|supercollider\n",
               value);
  return false;
}

[[nodiscard]] bool parse_policy_env(HookPolicy *out) {
  const char *value = std::getenv("RJ_CONSAN_POLICY");
  if (value == nullptr || *value == '\0' || ascii_iequals(value, "default")) {
    *out = HookPolicy::Default;
    return true;
  }
  if (ascii_iequals(value, "strict")) {
    *out = HookPolicy::Strict;
    return true;
  }
  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_POLICY='%s'; expected default|strict\n",
               value);
  return false;
}

void warn_irrelevant_env_combinations(const HookConfig &config) {
  if (env_has_value("RJ_CONSAN_MOI_ENGINE") && env_has_value("RJ_CONSAN_MOI_BACKEND"))
    warn_ignored_env("RJ_CONSAN_MOI_BACKEND", "RJ_CONSAN_MOI_ENGINE takes precedence");

  if (config.process_concurrent_transform_limit_bytes) {
    const std::optional<ConSanTransformReservationEstimate> minimum_reservation =
        consan_transform_major_image_reservation(1, config.patched_image_growth_limit);
    if (!minimum_reservation) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] warning: "
          "RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES=%llu cannot admit any nonempty "
          "code object because the configured per-object growth policy makes the smallest "
          "major-image reservation overflow uint64\n",
          static_cast<unsigned long long>(*config.process_concurrent_transform_limit_bytes));
    } else if (*config.process_concurrent_transform_limit_bytes <
               minimum_reservation->reservation_bytes) {
      std::fprintf(
          stderr,
          "[rocjitsu-dbi-hooks] warning: "
          "RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES=%llu cannot admit any nonempty "
          "code object; the smallest possible reservation is %llu bytes "
          "(phase=%s: %llu * input bytes + %llu * "
          "(input bytes + maximum growth bytes))\n",
          static_cast<unsigned long long>(*config.process_concurrent_transform_limit_bytes),
          static_cast<unsigned long long>(minimum_reservation->reservation_bytes),
          minimum_reservation->phase_name(),
          static_cast<unsigned long long>(minimum_reservation->input_image_copies()),
          static_cast<unsigned long long>(minimum_reservation->maximum_image_copies()));
    }
  }

  if (config.flavor == rocjitsu::ConSanFlavor::Moi) {
    if (env_has_value("RJ_CONSAN_SC_REPORT_MODE"))
      warn_ignored_env("RJ_CONSAN_SC_REPORT_MODE", "only applies to RJ_CONSAN_MODE=supercollider");
    if (config.moi_engine != rocjitsu::ConSanMoiEngine::Sampled) {
      constexpr const char *kSampledOnlyKnobs[] = {
          "RJ_CONSAN_MOI_SAMPLE_STRIDE",         "RJ_CONSAN_MOI_SAMPLE_OFFSET",
          "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
          "RJ_CONSAN_MOI_SAMPLED_CHECK",
      };
      for (const char *name : kSampledOnlyKnobs) {
        if (env_has_value(name))
          warn_ignored_env(name, "only applies to RJ_CONSAN_MODE=sampled");
      }
    }
    if (config.moi_engine != rocjitsu::ConSanMoiEngine::RecordReplay &&
        (env_has_value("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS") ||
         env_has_value("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT"))) {
      if (env_has_value("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS"))
        warn_ignored_env("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS",
                         "only applies to RJ_CONSAN_MODE=record-replay");
      if (env_has_value("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT"))
        warn_ignored_env("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT",
                         "only applies to RJ_CONSAN_MODE=record-replay");
    }
    const bool probe_local_owner = config.moi_engine == rocjitsu::ConSanMoiEngine::InlineShadow;
    if (!config.moi_init_owner_epoch && !probe_local_owner &&
        (env_has_value("RJ_CONSAN_MOI_OWNER_SOURCE") ||
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
      "RJ_CONSAN_MOI_FORBID_OVERFLOW",
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
      "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
      "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
      "RJ_CONSAN_MOI_SAMPLED_CHECK",
  };
  for (const char *name : kMoiOnlyKnobs) {
    if (env_has_value(name))
      warn_ignored_env(name, "RJ_CONSAN_MODE does not select an MOI engine");
  }
}

[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config);

[[nodiscard]] std::optional<HookConfig> parse_config() {
  HookConfig config;
  if (!parse_log_level(&config.log_level))
    return std::nullopt;

  if (!parse_mode_env(&config))
    return std::nullopt;
  if (!parse_policy_env(&config.policy))
    return std::nullopt;
  config.enabled = true;
  if (!parse_flat_provenance_mode_env(&config.flat_provenance_mode))
    return std::nullopt;
  if (!parse_moi_owner_source_env(&config.moi_owner_source))
    return std::nullopt;
  const bool strict_policy = config.policy == HookPolicy::Strict;
  if (!parse_bool_env("RJ_CONSAN_FAIL_CLOSED", strict_policy, &config.fail_closed))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_REQUIRE_PATCH", strict_policy, &config.require_patch))
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
  if (!parse_sc_report_mode_env(&config.sc_report_mode))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_LDS_CHECK_TRAP", false, &config.probe_lds_check_trap))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_FLAT_CHECK_TRAP", false, &config.probe_flat_check_trap))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT", false,
                      &config.abort_unmatched_barrier_wait))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_PROBE_FLAT_TRAP", false, &config.probe_flat_trap))
    return std::nullopt;
  if (!parse_sc_perturb_kind_env(&config.sc_perturb_kind) ||
      !parse_sc_perturb_edge_env(&config.sc_perturb_edge) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_INDEX", 0, &config.sc_perturb_index) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_MAX", 1, &config.sc_perturb_max) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_SLEEP", 1, &config.sc_perturb_sleep) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_REQUIRED_COUNT", 0, &config.sc_perturb_required_count))
    return std::nullopt;
  if (const char *identity = std::getenv("RJ_CONSAN_SC_PERTURB_IDENTITY"))
    config.sc_perturb_identity = identity;
  if (config.sc_perturb_kind != rocjitsu::ConSanPerturbationKind::None &&
      config.flavor != rocjitsu::ConSanFlavor::SuperCollider) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_SC_PERTURB_KIND requires "
                         "RJ_CONSAN_MODE=supercollider\n");
    return std::nullopt;
  }
  if (config.sc_perturb_max == 0 || config.sc_perturb_max > 2 || config.sc_perturb_sleep == 0 ||
      config.sc_perturb_sleep > 15 || config.sc_perturb_required_count > config.sc_perturb_max) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] SC perturb controls require max=1..2, "
                         "sleep=1..15, and required-count<=max\n");
    return std::nullopt;
  }
  if (config.flavor == rocjitsu::ConSanFlavor::SuperCollider &&
      !has_explicit_primary_probe(config) &&
      config.sc_perturb_kind == rocjitsu::ConSanPerturbationKind::None) {
    config.probe_lds_check_trap = config.check_trap_mode == CheckTrapMode::All ||
                                  config.check_trap_mode == CheckTrapMode::Lds;
    config.probe_flat_check_trap = config.check_trap_mode == CheckTrapMode::All ||
                                   config.check_trap_mode == CheckTrapMode::Flat;
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_DROP_BARRIER", false, &config.fault_drop_barrier))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_INCOMPLETE_BARRIER_DROP", false,
                      &config.fault_allow_destructive_incomplete_barrier_drop))
    return std::nullopt;
  if (config.fault_allow_destructive_incomplete_barrier_drop && !config.fault_drop_barrier) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] "
                         "RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_INCOMPLETE_BARRIER_DROP requires "
                         "RJ_CONSAN_FAULT_DROP_BARRIER=1\n");
    return std::nullopt;
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_MOVE_BARRIER", false, &config.fault_move_barrier))
    return std::nullopt;
  if (!parse_barrier_move_direction_env(&config.fault_barrier_move_direction))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ALLOW_COMPLETING_CONDITIONAL_BARRIER_MOVE", false,
                      &config.fault_allow_completing_conditional_barrier_move))
    return std::nullopt;
  if (config.fault_allow_completing_conditional_barrier_move &&
      (!config.fault_move_barrier ||
       config.fault_barrier_move_direction != rocjitsu::ConSanBarrierMoveDirection::Earlier)) {
    std::fprintf(
        stderr,
        "[rocjitsu-dbi-hooks] "
        "RJ_CONSAN_FAULT_ALLOW_COMPLETING_CONDITIONAL_BARRIER_MOVE requires "
        "RJ_CONSAN_FAULT_MOVE_BARRIER=1 and RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION=earlier\n");
    return std::nullopt;
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_DIVERGENT_BARRIER_MOVE", false,
                      &config.fault_allow_destructive_divergent_barrier_move))
    return std::nullopt;
  if (config.fault_allow_destructive_divergent_barrier_move &&
      (!config.fault_move_barrier ||
       config.fault_barrier_move_direction != rocjitsu::ConSanBarrierMoveDirection::Earlier)) {
    std::fprintf(
        stderr,
        "[rocjitsu-dbi-hooks] "
        "RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_DIVERGENT_BARRIER_MOVE requires "
        "RJ_CONSAN_FAULT_MOVE_BARRIER=1 and RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION=earlier\n");
    return std::nullopt;
  }
  if (const char *identity = std::getenv("RJ_CONSAN_FAULT_BARRIER_DESTINATION_IDENTITY"))
    config.fault_barrier_destination_identity = identity;
  if (!parse_bool_env("RJ_CONSAN_FAULT_MUTATE_BARRIER_ID_SCOPE", false,
                      &config.fault_mutate_barrier_id_scope))
    return std::nullopt;
  if (const char *identity = std::getenv("RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY"))
    config.fault_barrier_sequence_identity = identity;
  if (const char *identity = std::getenv("RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY"))
    config.fault_barrier_companion_site_identity = identity;
  if (const char *identity = std::getenv("RJ_CONSAN_FAULT_BARRIER_COMPANION_SEQUENCE_IDENTITY"))
    config.fault_barrier_companion_sequence_identity = identity;
  if (config.fault_barrier_companion_site_identity.empty() !=
      config.fault_barrier_companion_sequence_identity.empty()) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] grouped barrier drop requires both "
                         "RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY and "
                         "RJ_CONSAN_FAULT_BARRIER_COMPANION_SEQUENCE_IDENTITY\n");
    return std::nullopt;
  }
  if (!parse_optional_i32_env("RJ_CONSAN_FAULT_BARRIER_TARGET_ID", &config.fault_barrier_target_id))
    return std::nullopt;
  if (config.fault_mutate_barrier_id_scope &&
      (config.fault_barrier_sequence_identity.empty() || !config.fault_barrier_target_id)) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_MUTATE_BARRIER_ID_SCOPE requires "
                         "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY and "
                         "RJ_CONSAN_FAULT_BARRIER_TARGET_ID\n");
    return std::nullopt;
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_MUTATE_BARRIER_PARTICIPANTS", false,
                      &config.fault_mutate_barrier_participants))
    return std::nullopt;
  if (std::getenv("RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_COUNT") != nullptr) {
    uint32_t count = 0;
    if (!parse_u32_env("RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_COUNT", 0, &count))
      return std::nullopt;
    config.fault_barrier_target_participant_count = count;
  }
  if (std::getenv("RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_MASK") != nullptr) {
    uint64_t mask = 0;
    if (!parse_u64_env("RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_MASK", 0, &mask))
      return std::nullopt;
    config.fault_barrier_target_participant_mask = mask;
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS", false,
                      &config.fault_atomic_wrong_address))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER", false,
                      &config.fault_atomic_weaken_order))
    return std::nullopt;
  if (!parse_atomic_order_edge_env(&config.fault_atomic_order_edge))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE", false,
                      &config.fault_atomic_weaken_scope))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS", false, &config.fault_lds_wrong_address))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ORDINARY_WEAKEN_ORDER", false,
                      &config.fault_ordinary_weaken_order))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ORDINARY_WEAKEN_SCOPE", false,
                      &config.fault_ordinary_weaken_scope))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS", false,
                      &config.fault_ordinary_wrong_address))
    return std::nullopt;
  if (config.fault_atomic_wrong_address) {
    if (std::getenv("RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA") == nullptr) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS requires explicit "
                   "valid padded storage via RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA\n");
      return std::nullopt;
    }
    if (!parse_u32_env("RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA", 0,
                       &config.fault_atomic_address_delta) ||
        config.fault_atomic_address_delta == 0 || config.fault_atomic_address_delta % 4u != 0 ||
        config.fault_atomic_address_delta > 0x7fffffu) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA must be a "
                   "positive aligned signed-24-bit byte offset\n");
      return std::nullopt;
    }
  }
  if (config.fault_lds_wrong_address) {
    if (std::getenv("RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR") == nullptr ||
        !parse_u32_env("RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR", 0, &config.fault_lds_address_vgpr) ||
        config.fault_lds_address_vgpr > 255u) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS requires "
                           "RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR in the range 0..255\n");
      return std::nullopt;
    }
  }
  if (config.fault_ordinary_wrong_address) {
    if (std::getenv("RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA") == nullptr) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS requires "
                           "explicit valid storage via "
                           "RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA\n");
      return std::nullopt;
    }
    if (!parse_u32_env("RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA", 0,
                       &config.fault_ordinary_address_delta) ||
        config.fault_ordinary_address_delta == 0 || config.fault_ordinary_address_delta % 4u != 0 ||
        config.fault_ordinary_address_delta > 0x7fffffu) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] "
                   "RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA must be a positive aligned "
                   "signed-24-bit byte offset\n");
      return std::nullopt;
    }
  }
  if (!parse_bool_env("RJ_CONSAN_FAULT_DRY_RUN", false, &config.fault_dry_run))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", false,
                      &config.fault_require_exactly_one))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS",
                     kConSanDefaultFaultReservationTimeoutMs, &config.fault_reservation_timeout_ms))
    return std::nullopt;
  if (config.fault_reservation_timeout_ms == 0) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS='0'; "
                         "expected >=1\n");
    return std::nullopt;
  }
  // Synchronization implemented by an MOI engine is part of its ordinary
  // profile, not an expert opt-in. Engine-specific lowering still reports
  // unsupported sites honestly. Explicit false values remain useful for
  // focused compatibility and bring-up tests.
  const bool persistent_owner_defaults =
      config.flavor == rocjitsu::ConSanFlavor::Moi &&
      (config.moi_engine == rocjitsu::ConSanMoiEngine::RecordReplay ||
       config.moi_engine == rocjitsu::ConSanMoiEngine::Sampled);
  const bool ordinary_moi_defaults = config.flavor == rocjitsu::ConSanFlavor::Moi;
  if (!parse_bool_env("RJ_CONSAN_MOI_INIT_OWNER_EPOCH", persistent_owner_defaults,
                      &config.moi_init_owner_epoch))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_TRACK_BARRIERS", ordinary_moi_defaults,
                      &config.moi_track_barriers))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_TRACK_ATOMICS", ordinary_moi_defaults,
                      &config.moi_track_atomics))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS", false,
                      &config.moi_dynamic_access_records))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_SAMPLED_CHECK", false, &config.moi_sampled_check))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_PARTITION_MASK_DEBUG", false,
                      &config.moi_partition_mask_debug))
    return std::nullopt;
  // Deliberately test-only: this is not part of the public ConSan knob set.
  if (!parse_bool_env("RJ_CONSAN_TEST_FORCE_VGPR_SPILL", false, &config.test_force_vgpr_spill))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_TEST_FORCE_PRIVATE_EPOCH", false,
                      &config.test_force_private_epoch))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_TEST_SEED_INLINE_EXACT_ODD", false,
                      &config.test_seed_inline_exact_odd))
    return std::nullopt;
  if (config.test_seed_inline_exact_odd &&
      (config.flavor != rocjitsu::ConSanFlavor::Moi ||
       config.moi_engine != rocjitsu::ConSanMoiEngine::InlineShadow)) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_TEST_SEED_INLINE_EXACT_ODD requires "
                         "MOI inline_shadow\n");
    return std::nullopt;
  }
  if (const char *test_filter = std::getenv("RJ_CONSAN_TEST_KERNEL_FILTER"))
    config.test_kernel_name_filter = test_filter;
  const bool strict_moi_policy = strict_policy && config.flavor == rocjitsu::ConSanFlavor::Moi;
  if (!parse_bool_env("RJ_CONSAN_MOI_REQUIRE_RECORDS", strict_moi_policy,
                      &config.moi_require_records))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS", false, &config.moi_require_diagnostics))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", false, &config.moi_forbid_diagnostics))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT", false,
                      &config.moi_require_replay_conflict))
    return std::nullopt;
  if (!parse_bool_env("RJ_CONSAN_MOI_FORBID_OVERFLOW", strict_moi_policy,
                      &config.moi_forbid_overflow))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_FAULT_BARRIER_INDEX", 0, &config.fault_barrier_index))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_FAULT_ATOMIC_INDEX", 0, &config.fault_atomic_index))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_FAULT_LDS_INDEX", 0, &config.fault_lds_index))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_FAULT_ORDINARY_INDEX", 0, &config.fault_ordinary_index))
    return std::nullopt;
  if (const char *identity = std::getenv("RJ_CONSAN_FAULT_SITE_IDENTITY"))
    config.fault_site_identity = identity;
  if (!config.fault_barrier_companion_site_identity.empty() &&
      (!config.fault_drop_barrier || config.fault_site_identity.empty() ||
       config.fault_barrier_sequence_identity.empty())) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] grouped barrier drop requires "
                         "RJ_CONSAN_FAULT_DROP_BARRIER=1 plus exact primary site and sequence "
                         "identities\n");
    return std::nullopt;
  }
  if (std::getenv("RJ_CONSAN_FAULT_LOAD_OCCURRENCE") != nullptr) {
    uint32_t occurrence = 0;
    if (!parse_u32_env("RJ_CONSAN_FAULT_LOAD_OCCURRENCE", 0, &occurrence) || occurrence == 0) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_LOAD_OCCURRENCE must be a "
                           "positive one-based integer\n");
      return std::nullopt;
    }
    config.fault_load_occurrence = occurrence;
  }
  if (config.fault_load_occurrence && (config.fault_site_identity.empty() ||
                                       !config.fault_require_exactly_one || config.fault_dry_run)) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_LOAD_OCCURRENCE requires an exact "
                 "site identity, RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1, and a live fault row\n");
    return std::nullopt;
  }
  if (!parse_delay_mode_env(&config.delay_mode))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_DELAY", 0, &config.delay_nops))
    return std::nullopt;
  const bool absolute_growth_limit = env_has_value("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES");
  const bool relative_growth_limit = env_has_value("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT");
  if (absolute_growth_limit && relative_growth_limit) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES and "
                         "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT cannot both be set\n");
    return std::nullopt;
  }
  if (absolute_growth_limit) {
    config.patched_image_growth_limit.kind =
        rocjitsu::ConSanPatchedImageGrowthLimitKind::AbsoluteBytes;
    if (!parse_u64_env("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES",
                       rocjitsu::kConSanDefaultMaxPatchedImageGrowthBytes,
                       &config.patched_image_growth_limit.absolute_bytes, 10)) {
      return std::nullopt;
    }
  } else if (relative_growth_limit) {
    config.patched_image_growth_limit.kind =
        rocjitsu::ConSanPatchedImageGrowthLimitKind::InputPercent;
    if (!parse_u32_env("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", 0,
                       &config.patched_image_growth_limit.input_percent)) {
      return std::nullopt;
    }
  }
  if (env_has_value("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES")) {
    uint64_t process_limit = 0;
    if (!parse_u64_env("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", 0, &process_limit, 10))
      return std::nullopt;
    config.process_patched_image_growth_limit_bytes = process_limit;
  }
  if (env_has_value("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES")) {
    uint64_t process_limit = 0;
    if (!parse_u64_env("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", 0, &process_limit, 10))
      return std::nullopt;
    config.process_patched_image_limit_bytes = process_limit;
  }
  if (env_has_value("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES")) {
    uint64_t process_limit = 0;
    if (!parse_u64_env("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES", 0, &process_limit, 10))
      return std::nullopt;
    config.process_concurrent_transform_limit_bytes = process_limit;
  }
  config.max_patches_explicit = env_has_value("RJ_CONSAN_MAX_PATCHES");
  if (!parse_u32_env("RJ_CONSAN_MAX_PATCHES", kConSanAllSupportedPatchBudget, &config.max_patches))
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
  config.moi_runtime_sample_stride_explicit = env_has_value("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE");
  uint32_t runtime_sample_stride_default = 1u;
  if (config.flavor == rocjitsu::ConSanFlavor::Moi) {
    if (config.moi_engine == rocjitsu::ConSanMoiEngine::RecordReplay)
      runtime_sample_stride_default = kMoiRecordReplayStandardRuntimeStride;
    else if (config.moi_engine == rocjitsu::ConSanMoiEngine::Sampled)
      runtime_sample_stride_default = kMoiSampledStandardRuntimeStride;
  }
  if (!parse_u32_env("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", runtime_sample_stride_default,
                     &config.moi_runtime_sample_stride))
    return std::nullopt;
  if (config.moi_runtime_sample_stride == 0 || config.moi_runtime_sample_stride > (1u << 24u) ||
      (config.moi_runtime_sample_stride & (config.moi_runtime_sample_stride - 1u)) != 0) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE='%s'; "
                 "expected a power of two in 1..16777216\n",
                 std::getenv("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"));
    return std::nullopt;
  }
  if (!parse_u32_env("RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET", 0, &config.moi_runtime_sample_offset))
    return std::nullopt;
  if (config.moi_runtime_sample_offset >= config.moi_runtime_sample_stride) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET='%s'; "
                 "expected < RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE (%u)\n",
                 std::getenv("RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET"),
                 config.moi_runtime_sample_stride);
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
  if (config.flavor == rocjitsu::ConSanFlavor::Moi && config.moi_forbid_overflow &&
      config.moi_auto_report_buffer_size == 0) {
    warn_env("RJ_CONSAN_MOI_FORBID_OVERFLOW",
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
  config->moi_auto_report_buffer_size_explicit =
      env_has_value("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE");
  if (config->moi_auto_report_buffer_size_explicit) {
    if (!parse_u64_env("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", 0,
                       &config->moi_auto_report_buffer_size))
      return false;
  } else {
    config->moi_auto_report_buffer_size =
        config->flavor == rocjitsu::ConSanFlavor::Moi && !config->moi_report_buffer_address
            ? rocjitsu::consan_moi_auto_report_buffer_ceiling_bytes(config->moi_engine)
            : 0;
  }
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
  const uint64_t auto_report_buffer_ceiling =
      rocjitsu::consan_moi_auto_report_buffer_ceiling_bytes(config->moi_engine);
  if (config->moi_auto_report_buffer_size > auto_report_buffer_ceiling) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE='%s'; "
                 "maximum auto-report cap is %llu bytes\n",
                 std::getenv("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE"),
                 static_cast<unsigned long long>(auto_report_buffer_ceiling));
    return false;
  }

  return parse_u32_env("RJ_CONSAN_REPORT_MARKER", 1, &config->report_marker);
}

} // namespace rocjitsu::consan_hook
