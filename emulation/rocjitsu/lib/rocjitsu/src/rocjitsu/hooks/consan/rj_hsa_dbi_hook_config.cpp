// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

namespace rocjitsu::consan::hook {

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

struct BoolEnvBinding {
  const char *name;
  bool default_value;
  bool *output;
};

[[nodiscard]] bool parse_bool_envs(std::initializer_list<BoolEnvBinding> bindings) {
  return std::ranges::all_of(bindings, [](const BoolEnvBinding &binding) {
    return parse_bool_env(binding.name, binding.default_value, binding.output);
  });
}

[[nodiscard]] bool parse_positive_decimal(std::string_view text, uint64_t *out) {
  if (text.empty() ||
      !std::ranges::all_of(text, [](unsigned char c) { return std::isdigit(c) != 0; }))
    return false;
  errno = 0;
  char *end = nullptr;
  const std::string owned(text);
  const unsigned long long value = std::strtoull(owned.c_str(), &end, 10);
  if (errno == ERANGE || end == owned.c_str() || *end != '\0' || value == 0)
    return false;
  *out = static_cast<uint64_t>(value);
  return true;
}

[[nodiscard]] bool parse_epoch_analysis_env(HookConfig::EpochAnalysisPolicy *policy) {
  const char *raw = std::getenv("RJ_CONSAN_EPOCH_ANALYSIS");
  if (raw == nullptr || *raw == '\0') {
    *policy = {};
    return true;
  }
  const std::string_view value(raw);
  if (ascii_iequals(value, "every")) {
    *policy = {};
    return true;
  }
  if (ascii_iequals(value, "manual")) {
    policy->kind = HookConfig::EpochAnalysisKind::Manual;
    policy->value = 1;
    policy->offset = 1;
    return true;
  }

  const size_t first_colon = value.find(':');
  const std::string_view kind = value.substr(0, first_colon);
  std::string_view arguments =
      first_colon == std::string_view::npos ? std::string_view{} : value.substr(first_colon + 1);
  if (ascii_iequals(kind, "nth")) {
    uint64_t epoch = 0;
    if (arguments.find(':') == std::string_view::npos &&
        parse_positive_decimal(arguments, &epoch)) {
      policy->kind = HookConfig::EpochAnalysisKind::Nth;
      policy->value = epoch;
      policy->offset = epoch;
      return true;
    }
  } else if (ascii_iequals(kind, "periodic")) {
    const size_t second_colon = arguments.find(':');
    const std::string_view period_text = arguments.substr(0, second_colon);
    const std::string_view offset_text = second_colon == std::string_view::npos
                                             ? std::string_view{}
                                             : arguments.substr(second_colon + 1);
    uint64_t period = 0;
    uint64_t offset = 0;
    const bool has_explicit_offset = second_colon != std::string_view::npos;
    if (parse_positive_decimal(period_text, &period) &&
        (has_explicit_offset ? parse_positive_decimal(offset_text, &offset)
                             : (offset = period, true)) &&
        offset <= period) {
      policy->kind = HookConfig::EpochAnalysisKind::Periodic;
      policy->value = period;
      policy->offset = offset;
      return true;
    }
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_EPOCH_ANALYSIS='%s'; expected "
               "every, nth:N, periodic:N, periodic:N:OFFSET, or manual\n",
               raw);
  return false;
}

[[nodiscard]] bool append_kernel_allowlist_name(std::string_view item, const char *source,
                                                std::vector<std::string> *out) {
  while (!item.empty() && std::isspace(static_cast<unsigned char>(item.front())))
    item.remove_prefix(1);
  while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back())))
    item.remove_suffix(1);
  if (item.empty()) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s; expected nonempty exact kernel names\n",
                 source);
    return false;
  }
  std::string normalized(item);
  if (normalized.ends_with(".kd"))
    normalized.resize(normalized.size() - 3u);
  if (std::ranges::find(*out, normalized) == out->end())
    out->push_back(std::move(normalized));
  return true;
}

[[nodiscard]] bool parse_kernel_allowlist_env(std::vector<std::string> *out) {
  out->clear();
  const char *value = std::getenv("RJ_CONSAN_KERNEL_ALLOWLIST");
  const char *file_path = std::getenv("RJ_CONSAN_KERNEL_ALLOWLIST_FILE");
  const bool has_value = value != nullptr && *value != '\0';
  const bool has_file = file_path != nullptr && *file_path != '\0';
  if (has_value && has_file) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_KERNEL_ALLOWLIST and "
                         "RJ_CONSAN_KERNEL_ALLOWLIST_FILE are mutually exclusive\n");
    return false;
  }
  if (has_file) {
    std::ifstream input(file_path);
    if (!input) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] could not read "
                   "RJ_CONSAN_KERNEL_ALLOWLIST_FILE='%s'\n",
                   file_path);
      return false;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      const std::string source = "RJ_CONSAN_KERNEL_ALLOWLIST_FILE='" + std::string(file_path) +
                                 "' line " + std::to_string(line_number);
      if (!append_kernel_allowlist_name(line, source.c_str(), out)) {
        out->clear();
        return false;
      }
    }
    if (!input.eof() || out->empty()) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_KERNEL_ALLOWLIST_FILE='%s'; "
                   "expected one exact kernel name per line\n",
                   file_path);
      out->clear();
      return false;
    }
    return true;
  }
  if (!has_value)
    return true;
  std::string_view remaining(value);
  while (!remaining.empty()) {
    const size_t comma = remaining.find(',');
    const std::string_view item = remaining.substr(0, comma);
    if (!append_kernel_allowlist_name(item, "RJ_CONSAN_KERNEL_ALLOWLIST", out)) {
      out->clear();
      return false;
    }
    if (comma == std::string_view::npos)
      break;
    remaining.remove_prefix(comma + 1u);
    if (remaining.empty()) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_KERNEL_ALLOWLIST='%s'; "
                   "expected comma-separated nonempty exact kernel names\n",
                   value);
      out->clear();
      return false;
    }
  }
  return true;
}

template <typename UInt>
[[nodiscard]] bool parse_unsigned_env(const char *name, UInt default_value, UInt *out,
                                      int base = 0) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    *out = default_value;
    return true;
  }
  if (!std::isdigit(static_cast<unsigned char>(*value))) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected uint%zu\n", name, value,
                 sizeof(UInt) * 8u);
    return false;
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, base);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed > std::numeric_limits<UInt>::max()) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected uint%zu\n", name, value,
                 sizeof(UInt) * 8u);
    return false;
  }

  *out = static_cast<UInt>(parsed);
  return true;
}

[[nodiscard]] bool parse_u32_env(const char *name, uint32_t default_value, uint32_t *out) {
  return parse_unsigned_env(name, default_value, out, 10);
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

[[nodiscard]] bool parse_optional_u16_env(const char *name, uint16_t maximum, bool require_even,
                                          const char *expected, std::optional<uint16_t> *out) {
  const char *value = std::getenv(name);
  out->reset();
  if (value == nullptr || *value == '\0') {
    return true;
  }

  uint32_t parsed = 0;
  if (!parse_u32_env(name, 0, &parsed))
    return false;
  if (parsed > maximum || (require_even && parsed % 2u != 0u)) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected %s\n", name, value,
                 expected);
    return false;
  }
  *out = static_cast<uint16_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_optional_vgpr_env(const char *name, std::optional<uint16_t> *out) {
  return parse_optional_u16_env(name, 255u, false, "0..255", out);
}

[[nodiscard]] bool parse_optional_sgpr_env(const char *name, std::optional<uint16_t> *out) {
  return parse_optional_u16_env(name, 105u, false, "SGPR 0..105", out);
}

[[nodiscard]] bool parse_optional_sgpr_pair_env(const char *name, std::optional<uint16_t> *out) {
  return parse_optional_u16_env(name, 104u, true, "an even SGPR pair base in 0..104", out);
}

[[nodiscard]] bool parse_u64_env(const char *name, uint64_t default_value, uint64_t *out,
                                 int base = 0) {
  return parse_unsigned_env(name, default_value, out, base);
}

template <typename Enum>
[[nodiscard]] bool
parse_enum_env(const char *name, Enum default_value, Enum *out,
               std::initializer_list<std::pair<std::string_view, Enum>> vocabulary,
               const char *expected, bool case_insensitive = true) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    *out = default_value;
    return true;
  }
  const auto entry = std::ranges::find_if(vocabulary, [&](const auto &candidate) {
    return case_insensitive ? ascii_iequals(value, candidate.first) : value == candidate.first;
  });
  if (entry != vocabulary.end()) {
    *out = entry->second;
    return true;
  }
  std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid %s='%s'; expected %s\n", name, value,
               expected);
  return false;
}

[[nodiscard]] bool parse_supercollider_delay_mode_env(SuperColliderDelayMode *out) {
  using E = SuperColliderDelayMode;
  return parse_enum_env("RJ_CONSAN_SC_DELAY_MODE", E::Nop, out,
                        {{"nop", E::Nop},
                         {"sleep", E::Sleep},
                         {"sleep_var", E::SleepVar},
                         {"sleep-var", E::SleepVar}},
                        "nop|sleep|sleep_var");
}

[[nodiscard]] bool parse_barrier_move_direction_env(BarrierMoveDirection *out) {
  using E = BarrierMoveDirection;
  return parse_enum_env("RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION", E::LegacyMarker, out,
                        {{"legacy-marker", E::LegacyMarker},
                         {"legacy_marker", E::LegacyMarker},
                         {"earlier", E::Earlier},
                         {"later", E::Later}},
                        "legacy-marker, earlier, or later", false);
}

[[nodiscard]] bool parse_supercollider_perturb_kind_env(SuperColliderPerturbationKind *out) {
  using E = SuperColliderPerturbationKind;
  return parse_enum_env("RJ_CONSAN_SC_PERTURB_KIND", E::None, out,
                        {{"none", E::None}, {"barrier", E::Barrier}, {"atomic", E::Atomic}},
                        "none, barrier, or atomic");
}

[[nodiscard]] bool parse_supercollider_perturb_edge_env(SuperColliderPerturbationEdge *out) {
  using E = SuperColliderPerturbationEdge;
  return parse_enum_env("RJ_CONSAN_SC_PERTURB_EDGE", E::Release, out,
                        {{"release", E::Release}, {"acquire", E::Acquire}}, "release or acquire");
}

[[nodiscard]] bool parse_atomic_order_edge_env(AtomicOrderEdge *out) {
  using E = AtomicOrderEdge;
  return parse_enum_env("RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE", E::Any, out,
                        {{"any", E::Any}, {"release", E::Release}, {"acquire", E::Acquire}},
                        "any, release, or acquire");
}

[[nodiscard]] bool parse_owner_source_env(OwnerSource *out) {
  using E = OwnerSource;
  return parse_enum_env("RJ_CONSAN_OWNER_SOURCE", E::Automatic, out,
                        {{"automatic", E::Automatic},
                         {"auto", E::Automatic},
                         {"workitem_id", E::WorkitemId},
                         {"workitem", E::WorkitemId},
                         {"hw_id", E::HwId},
                         {"hwid", E::HwId}},
                        "automatic|workitem_id|hw_id", false);
}

[[nodiscard]] bool parse_flat_provenance_mode_env(FlatProvenanceMode *out) {
  using E = FlatProvenanceMode;
  return parse_enum_env(
      "RJ_CONSAN_FLAT_PROVENANCE", E::Likely, out,
      {{"likely", E::Likely}, {"default", E::Likely}, {"strict", E::Strict}, {"group", E::Strict}},
      "likely|strict");
}

[[nodiscard]] bool parse_check_trap_mode_env(CheckTrapMode *out) {
  return parse_enum_env("RJ_CONSAN_CHECK_TRAP_MODE", CheckTrapMode::All, out,
                        {{"all", CheckTrapMode::All},
                         {"both", CheckTrapMode::All},
                         {"default", CheckTrapMode::All},
                         {"lds", CheckTrapMode::Lds},
                         {"ds", CheckTrapMode::Lds},
                         {"native-lds", CheckTrapMode::Lds},
                         {"flat", CheckTrapMode::Flat},
                         {"vflat", CheckTrapMode::Flat},
                         {"generic", CheckTrapMode::Flat}},
                        "all|lds|flat");
}

[[nodiscard]] bool parse_supercollider_report_mode_env(SuperColliderReportMode *out) {
  return parse_enum_env("RJ_CONSAN_SC_REPORT_MODE", SuperColliderReportMode::Auto, out,
                        {{"auto", SuperColliderReportMode::Auto},
                         {"default", SuperColliderReportMode::Auto},
                         {"trap", SuperColliderReportMode::Trap}},
                        "auto|trap");
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
  return config.probe_lds_check_trap || config.probe_flat_check_trap;
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

[[nodiscard]] bool parse_mode_env(HookConfig *config) {
  const char *value = std::getenv("RJ_CONSAN_MODE");
  if (value == nullptr || *value == '\0') {
    config->mode = Mode::Default;
    return true;
  }

  if (ascii_iequals(value, "default")) {
    config->mode = Mode::Default;
    return true;
  }
  if (ascii_iequals(value, "supercollider")) {
    config->mode = Mode::SuperCollider;
    return true;
  }

  std::fprintf(stderr,
               "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_MODE='%s'; expected "
               "default|supercollider\n",
               value);
  return false;
}

[[nodiscard]] bool parse_policy_env(HookPolicy *out) {
  return parse_enum_env("RJ_CONSAN_POLICY", HookPolicy::Default, out,
                        {{"default", HookPolicy::Default}, {"strict", HookPolicy::Strict}},
                        "default|strict");
}

void warn_irrelevant_env_combinations(const HookConfig &config) {
  if (config.process_concurrent_transform_limit_bytes) {
    const std::optional<TransformReservationEstimate> minimum_reservation =
        transform_major_image_reservation(1, config.patched_image_growth_limit);
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

  if (config.mode == Mode::Default) {
    if (env_has_value("RJ_CONSAN_SC_REPORT_MODE"))
      warn_ignored_env("RJ_CONSAN_SC_REPORT_MODE", "only applies to RJ_CONSAN_MODE=supercollider");
    if (!config.init_owner_epoch &&
        (env_has_value("RJ_CONSAN_OWNER_SOURCE") || env_has_value("RJ_CONSAN_OWNER_SGPR"))) {
      if (env_has_value("RJ_CONSAN_OWNER_SOURCE"))
        warn_ignored_env("RJ_CONSAN_OWNER_SOURCE", "only affects RJ_CONSAN_INIT_OWNER_EPOCH=1");
      if (env_has_value("RJ_CONSAN_OWNER_SGPR"))
        warn_ignored_env("RJ_CONSAN_OWNER_SGPR", "only affects RJ_CONSAN_INIT_OWNER_EPOCH=1");
    }
    return;
  }

  constexpr const char *kOnlyKnobs[] = {
      "RJ_CONSAN_REPORT_BUFFER",
      "RJ_CONSAN_REPORT_BUFFER_SIZE",
      "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE",
      "RJ_CONSAN_REQUIRE_RECORDS",
      "RJ_CONSAN_REQUIRE_DIAGNOSTICS",
      "RJ_CONSAN_FORBID_DIAGNOSTICS",
      "RJ_CONSAN_FORBID_OVERFLOW",
      "RJ_CONSAN_INIT_OWNER_EPOCH",
      "RJ_CONSAN_TRACK_BARRIERS",
      "RJ_CONSAN_TRACK_ATOMICS",
      "RJ_CONSAN_EXEC_SAVE_SGPR",
      "RJ_CONSAN_OWNER_SOURCE",
      "RJ_CONSAN_OWNER_SGPR",
      "RJ_CONSAN_OWNER_VGPR",
      "RJ_CONSAN_EPOCH_VGPR",
      "RJ_CONSAN_SAMPLE_STRIDE",
      "RJ_CONSAN_SAMPLE_OFFSET",
      "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE",
      "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET",
      "RJ_CONSAN_DEVICE_CONFLICT_CHECK",
      "RJ_CONSAN_CONFLICT_LIMIT",
      "RJ_CONSAN_TOTAL_CONFLICT_LIMIT",
      "RJ_CONSAN_EPOCH_ANALYSIS",
  };
  for (const char *name : kOnlyKnobs) {
    if (env_has_value(name))
      warn_ignored_env(name, "RJ_CONSAN_MODE does not select the default mode");
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
  if (!parse_owner_source_env(&config.owner_source))
    return std::nullopt;
  const bool strict_policy = config.policy == HookPolicy::Strict;
  if (!parse_check_trap_mode_env(&config.check_trap_mode))
    return std::nullopt;
  if (!parse_supercollider_report_mode_env(&config.supercollider_report_mode))
    return std::nullopt;
  config.supercollider_evidence_mode =
      config.supercollider_report_mode == SuperColliderReportMode::Trap
          ? SuperColliderEvidenceMode::TrapOnly
          : SuperColliderEvidenceMode::StickyMarker;
  // ConSan synchronization is part of its ordinary
  // profile, not an expert opt-in. Explicit false values remain useful for
  // focused compatibility and bring-up tests.
  const bool ordinary_defaults = config.mode == Mode::Default;
  const bool strict_report_policy = strict_policy && ordinary_defaults;
  if (!parse_bool_envs({
          {"RJ_CONSAN_FAIL_CLOSED", strict_policy, &config.fail_closed},
          {"RJ_CONSAN_REQUIRE_PATCH", strict_policy, &config.require_patch},
          {"RJ_CONSAN_PROBE_LDS_CHECK_TRAP", false, &config.probe_lds_check_trap},
          {"RJ_CONSAN_PROBE_FLAT_CHECK_TRAP", false, &config.probe_flat_check_trap},
          {"RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT", false, &config.abort_unmatched_barrier_wait},
          {"RJ_CONSAN_FAULT_DROP_BARRIER", false, &config.fault_drop_barrier},
          {"RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_INCOMPLETE_BARRIER_DROP", false,
           &config.fault_allow_destructive_incomplete_barrier_drop},
          {"RJ_CONSAN_FAULT_MOVE_BARRIER", false, &config.fault_move_barrier},
          {"RJ_CONSAN_FAULT_ALLOW_COMPLETING_CONDITIONAL_BARRIER_MOVE", false,
           &config.fault_allow_completing_conditional_barrier_move},
          {"RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_DIVERGENT_BARRIER_MOVE", false,
           &config.fault_allow_destructive_divergent_barrier_move},
          {"RJ_CONSAN_FAULT_MUTATE_BARRIER_ID_SCOPE", false, &config.fault_mutate_barrier_id_scope},
          {"RJ_CONSAN_FAULT_MUTATE_BARRIER_PARTICIPANTS", false,
           &config.fault_mutate_barrier_participants},
          {"RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS", false, &config.fault_atomic_wrong_address},
          {"RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER", false, &config.fault_atomic_weaken_order},
          {"RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE", false, &config.fault_atomic_weaken_scope},
          {"RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS", false, &config.fault_lds_wrong_address},
          {"RJ_CONSAN_FAULT_ORDINARY_WEAKEN_ORDER", false, &config.fault_ordinary_weaken_order},
          {"RJ_CONSAN_FAULT_ORDINARY_WEAKEN_SCOPE", false, &config.fault_ordinary_weaken_scope},
          {"RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS", false, &config.fault_ordinary_wrong_address},
          {"RJ_CONSAN_FAULT_DRY_RUN", false, &config.fault_dry_run},
          {"RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE", false, &config.fault_require_exactly_one},
          {"RJ_CONSAN_INIT_OWNER_EPOCH", ordinary_defaults, &config.init_owner_epoch},
          {"RJ_CONSAN_TRACK_BARRIERS", ordinary_defaults, &config.track_barriers},
          {"RJ_CONSAN_TRACK_ATOMICS", ordinary_defaults, &config.track_atomics},
          {"RJ_CONSAN_DEVICE_CONFLICT_CHECK", false, &config.device_conflict_check},
          {"RJ_CONSAN_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES", false,
           &config.allow_uniform_lds_stores},
          // Deliberately test-only: these are not part of the public ConSan knob set.
          {"RJ_CONSAN_TEST_FORCE_VGPR_SPILL", false, &config.test_force_vgpr_spill},
          {"RJ_CONSAN_TEST_FORCE_PRIVATE_EPOCH", false, &config.test_force_private_epoch},
          {"RJ_CONSAN_REQUIRE_RECORDS", strict_report_policy, &config.require_records},
          {"RJ_CONSAN_REQUIRE_DIAGNOSTICS", false, &config.require_diagnostics},
          {"RJ_CONSAN_FORBID_DIAGNOSTICS", false, &config.forbid_diagnostics},
          {"RJ_CONSAN_FORBID_OVERFLOW", strict_report_policy, &config.forbid_overflow},
      }))
    return std::nullopt;
  if (!parse_supercollider_perturb_kind_env(&config.supercollider_perturb_kind) ||
      !parse_supercollider_perturb_edge_env(&config.supercollider_perturb_edge) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_INDEX", 0, &config.supercollider_perturb_index) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_MAX", 1, &config.supercollider_perturb_max) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_SLEEP", 1, &config.supercollider_perturb_sleep) ||
      !parse_u32_env("RJ_CONSAN_SC_PERTURB_REQUIRED_COUNT", 0,
                     &config.supercollider_perturb_required_count))
    return std::nullopt;
  if (const char *identity = std::getenv("RJ_CONSAN_SC_PERTURB_IDENTITY"))
    config.supercollider_perturb_identity = identity;
  if (config.mode == Mode::SuperCollider && !has_explicit_primary_probe(config) &&
      config.supercollider_perturb_kind == SuperColliderPerturbationKind::None) {
    config.probe_lds_check_trap = config.check_trap_mode == CheckTrapMode::All ||
                                  config.check_trap_mode == CheckTrapMode::Lds;
    config.probe_flat_check_trap = config.check_trap_mode == CheckTrapMode::All ||
                                   config.check_trap_mode == CheckTrapMode::Flat;
  }
  if (!parse_barrier_move_direction_env(&config.fault_barrier_move_direction))
    return std::nullopt;
  if (const char *identity = std::getenv("RJ_CONSAN_FAULT_BARRIER_DESTINATION_IDENTITY"))
    config.fault_barrier_destination_identity = identity;
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
  if (!parse_atomic_order_edge_env(&config.fault_atomic_order_edge))
    return std::nullopt;
  if (config.fault_atomic_wrong_address) {
    if (std::getenv("RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA") == nullptr) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS requires explicit "
                   "valid padded storage via RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA\n");
      return std::nullopt;
    }
    if (!parse_u32_env("RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA", 0,
                       &config.fault_atomic_address_delta)) {
      return std::nullopt;
    }
  }
  if (config.fault_lds_wrong_address) {
    uint32_t address_vgpr = 0;
    if (std::getenv("RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR") == nullptr ||
        !parse_u32_env("RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR", 0, &address_vgpr) ||
        address_vgpr > 255u) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS requires "
                           "RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR in the range 0..255\n");
      return std::nullopt;
    }
    config.fault_lds_address_vgpr = static_cast<uint16_t>(address_vgpr);
  }
  if (config.fault_ordinary_wrong_address) {
    if (std::getenv("RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA") == nullptr) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS requires "
                           "explicit valid storage via "
                           "RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA\n");
      return std::nullopt;
    }
    if (!parse_u32_env("RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA", 0,
                       &config.fault_ordinary_address_delta)) {
      return std::nullopt;
    }
  }
  if (!parse_u32_env("RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS", kDefaultFaultReservationTimeoutMs,
                     &config.fault_reservation_timeout_ms))
    return std::nullopt;
  if (const char *test_filter = std::getenv("RJ_CONSAN_TEST_KERNEL_FILTER"))
    config.test_kernel_name_filter = test_filter;
  if (!parse_kernel_allowlist_env(&config.kernel_name_allowlist))
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
  if (!parse_supercollider_delay_mode_env(&config.supercollider_delay_mode))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_SC_DELAY", 0, &config.supercollider_delay_nops))
    return std::nullopt;
  const bool absolute_growth_limit = env_has_value("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES");
  const bool relative_growth_limit = env_has_value("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT");
  if (absolute_growth_limit && relative_growth_limit) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES and "
                         "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT cannot both be set\n");
    return std::nullopt;
  }
  if (absolute_growth_limit) {
    config.patched_image_growth_limit.kind = PatchedImageGrowthLimitKind::AbsoluteBytes;
    if (!parse_u64_env("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES",
                       kDefaultMaxPatchedImageGrowthBytes,
                       &config.patched_image_growth_limit.absolute_bytes, 10)) {
      return std::nullopt;
    }
  } else if (relative_growth_limit) {
    config.patched_image_growth_limit.kind = PatchedImageGrowthLimitKind::InputPercent;
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
  if (!parse_u32_env("RJ_CONSAN_MAX_PATCHES", kAllSupportedPatchBudget, &config.max_patches))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_SAMPLE_STRIDE", 1, &config.sample_stride))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_SAMPLE_OFFSET", 0, &config.sample_offset))
    return std::nullopt;
  // Presets supply selector defaults only. Explicit selectors keep their
  // existing precedence and validation; the default preset preserves coupled
  // selection exactly, including legacy offset overrides.
  uint32_t workgroup_default = kDefaultRuntimeSampleStride;
  uint32_t cell_default = workgroup_default;
  if (const char *preset = std::getenv("RJ_CONSAN_PRESET"); preset != nullptr && *preset != '\0') {
    if (config.mode != Mode::Default) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] RJ_CONSAN_PRESET requires ConSan\n");
      return std::nullopt;
    }
    if (ascii_iequals(preset, "default")) {
      config.preset = "default";
    } else if (ascii_iequals(preset, "low")) {
      config.preset = "low";
      workgroup_default = cell_default = 1024;
    } else if (ascii_iequals(preset, "high")) {
      config.preset = "high";
      workgroup_default = 1;
      cell_default = 4;
    } else if (ascii_iequals(preset, "max")) {
      config.preset = "max";
      workgroup_default = cell_default = 1;
    } else {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_PRESET='%s'; "
                   "expected low|default|high|max\n",
                   preset);
      return std::nullopt;
    }
  }
  const bool legacy_sample_selection = env_has_value("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE") ||
                                       env_has_value("RJ_CONSAN_RUNTIME_SAMPLE_OFFSET");
  const bool explicit_independent_selection = env_has_value("RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE") ||
                                              env_has_value("RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET") ||
                                              env_has_value("RJ_CONSAN_CELL_SAMPLE_STRIDE") ||
                                              env_has_value("RJ_CONSAN_CELL_SAMPLE_OFFSET");
  config.runtime_sample_stride_explicit = env_has_value("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE");
  const uint32_t runtime_sample_stride_default =
      config.mode == Mode::Default ? workgroup_default : 1u;
  if (!parse_u32_env("RJ_CONSAN_RUNTIME_SAMPLE_STRIDE", runtime_sample_stride_default,
                     &config.runtime_sample_stride) ||
      !parse_u32_env("RJ_CONSAN_RUNTIME_SAMPLE_OFFSET", 0, &config.runtime_sample_offset))
    return std::nullopt;
  const bool independent_sample_selection =
      explicit_independent_selection ||
      (!legacy_sample_selection && workgroup_default != cell_default);
  if (independent_sample_selection) {
    if (config.mode != Mode::Default || legacy_sample_selection) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] independent workgroup/cell selectors require "
                           "ConSan and cannot be combined with legacy runtime selectors\n");
      return std::nullopt;
    }
    SampleSelector cell;
    if (!parse_u32_env("RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE", workgroup_default,
                       &config.runtime_sample_stride) ||
        !parse_u32_env("RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET", 0, &config.runtime_sample_offset) ||
        !parse_u32_env("RJ_CONSAN_CELL_SAMPLE_STRIDE", cell_default, &cell.stride) ||
        !parse_u32_env("RJ_CONSAN_CELL_SAMPLE_OFFSET", 0, &cell.offset))
      return std::nullopt;
    config.cell_selection = cell;
    config.runtime_sample_stride_explicit = explicit_independent_selection;
  }
  if (!parse_u32_env("RJ_CONSAN_WATCHPOINT_BANKS", 0, &config.watchpoint_banks))
    return std::nullopt;
  if (!parse_u32_env("RJ_CONSAN_CONFLICT_LIMIT", 8, &config.conflict_limit) ||
      !parse_u32_env("RJ_CONSAN_TOTAL_CONFLICT_LIMIT", 64, &config.total_conflict_limit))
    return std::nullopt;
  if (config.conflict_limit > 1024 || config.total_conflict_limit > 65536) {
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] sampled conflict limits exceed bounds "
                         "(per report: 1024, per hook session: 65536)\n");
    return std::nullopt;
  }
  if (!parse_epoch_analysis_env(&config.epoch_analysis))
    return std::nullopt;
  if (!refresh_report_config_from_env(&config))
    return std::nullopt;
  uint32_t supercollider_delay_var_ssrc = 106;
  if (!parse_u32_env("RJ_CONSAN_SC_DELAY_VAR_SSRC", 106, &supercollider_delay_var_ssrc))
    return std::nullopt;
  if (supercollider_delay_var_ssrc > 255) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_SC_DELAY_VAR_SSRC='%s'; expected 0..255\n",
                 std::getenv("RJ_CONSAN_SC_DELAY_VAR_SSRC"));
    return std::nullopt;
  }
  config.supercollider_delay_var_ssrc = static_cast<uint16_t>(supercollider_delay_var_ssrc);
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
  if (!parse_optional_vgpr_env("RJ_CONSAN_OWNER_VGPR", &config.requested_owner_vgpr))
    return std::nullopt;
  if (!parse_optional_vgpr_env("RJ_CONSAN_EPOCH_VGPR", &config.requested_epoch_vgpr))
    return std::nullopt;
  if (!parse_optional_sgpr_env("RJ_CONSAN_OWNER_SGPR", &config.requested_owner_sgpr))
    return std::nullopt;
  if (!parse_optional_sgpr_pair_env("RJ_CONSAN_EXEC_SAVE_SGPR", &config.requested_exec_save_sgpr))
    return std::nullopt;
  if (config.mode == Mode::Default && config.auto_report_buffer_size == 0) {
    const std::pair<bool, const char *> report_guards[] = {
        {config.require_records, "RJ_CONSAN_REQUIRE_RECORDS"},
        {config.require_diagnostics, "RJ_CONSAN_REQUIRE_DIAGNOSTICS"},
        {config.forbid_diagnostics, "RJ_CONSAN_FORBID_DIAGNOSTICS"},
        {config.forbid_overflow, "RJ_CONSAN_FORBID_OVERFLOW"},
    };
    for (const auto &[enabled, name] : report_guards) {
      if (enabled)
        warn_env(name, "this guard only checks HSA-tool-owned auto report buffers");
    }
  }
  config.max_patches_is_expert_limit = config.max_patches_explicit;
  // Keep this parser responsible for environment syntax and hook-only
  // provenance constraints. Cross-field semantics belong to the typed
  // contracts so every configuration source observes the same rules.
  const auto reject_contract = [](ContractIssue issue) {
    if (issue == ContractIssue::None)
      return false;
    const std::string_view name = contract_issue_name(issue);
    std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid ConSan typed configuration: %.*s\n",
                 static_cast<int>(name.size()), name.data());
    return true;
  };
  if (reject_contract(validate_configuration(config, config, config, config, config, config))) {
    return std::nullopt;
  }
  warn_irrelevant_env_combinations(config);
  return config;
}

[[nodiscard]] bool refresh_report_config_from_env(HookConfig *config) {
  uint64_t supercollider_report_buffer_address = 0;
  if (const char *value = std::getenv("RJ_CONSAN_SC_REPORT_BUFFER");
      value != nullptr && *value != '\0') {
    if (!parse_u64_env("RJ_CONSAN_SC_REPORT_BUFFER", 0, &supercollider_report_buffer_address))
      return false;
    if (supercollider_report_buffer_address == 0) {
      std::fprintf(stderr,
                   "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_SC_REPORT_BUFFER='0'; expected nonzero "
                   "device-visible address\n");
      return false;
    }
    config->supercollider_report_buffer_address = supercollider_report_buffer_address;
  } else {
    config->supercollider_report_buffer_address.reset();
  }

  uint64_t report_buffer_address = 0;
  if (const char *value = std::getenv("RJ_CONSAN_REPORT_BUFFER");
      value != nullptr && *value != '\0') {
    if (!parse_u64_env("RJ_CONSAN_REPORT_BUFFER", 0, &report_buffer_address))
      return false;
    if (report_buffer_address == 0) {
      std::fprintf(stderr, "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_REPORT_BUFFER='0'; expected "
                           "nonzero device-visible address\n");
      return false;
    }
    config->report_buffer_address = report_buffer_address;
  } else {
    config->report_buffer_address.reset();
  }
  if (!parse_u64_env("RJ_CONSAN_REPORT_BUFFER_SIZE", 0, &config->report_buffer_size))
    return false;
  config->auto_report_buffer_size_explicit = env_has_value("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE");
  if (config->auto_report_buffer_size_explicit) {
    if (!parse_u64_env("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE", 0, &config->auto_report_buffer_size))
      return false;
  } else {
    config->auto_report_buffer_size =
        config->mode == Mode::Default && !config->report_buffer_address
            ? kOrdinaryAutoReportBufferCeilingBytes
            : 0;
  }
  if (config->auto_report_buffer_size != 0 &&
      config->auto_report_buffer_size < sizeof(ReportHeader)) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE='%s'; "
                 "expected 0 or at least %zu bytes\n",
                 std::getenv("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE"), sizeof(ReportHeader));
    return false;
  }
  const uint64_t auto_report_buffer_ceiling = kOrdinaryAutoReportBufferCeilingBytes;
  if (config->auto_report_buffer_size > auto_report_buffer_ceiling) {
    std::fprintf(stderr,
                 "[rocjitsu-dbi-hooks] invalid RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE='%s'; "
                 "maximum auto-report cap is %llu bytes\n",
                 std::getenv("RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE"),
                 static_cast<unsigned long long>(auto_report_buffer_ceiling));
    return false;
  }

  config->scope =
      config->bound() ? RuntimeResourceScope::CodeObject : RuntimeResourceScope::Unbound;
  return parse_u32_env("RJ_CONSAN_SC_REPORT_MARKER", 1, &config->supercollider_report_marker);
}

} // namespace rocjitsu::consan::hook
