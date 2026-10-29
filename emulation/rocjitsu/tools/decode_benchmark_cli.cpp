// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_benchmark_cli.h"
#include "cli_choices.h"

#include <array>
#include <limits>
#include <string>

namespace rocjitsu::tools {
namespace {

constexpr std::array<NamedChoice<rj_code_target_id_t>, 6> kTargets{{
    {"gfx942", ROCJITSU_CODE_TARGET_GFX942},
    {"gfx950", ROCJITSU_CODE_TARGET_GFX950},
    {"gfx1200", ROCJITSU_CODE_TARGET_GFX1200},
    {"gfx1201", ROCJITSU_CODE_TARGET_GFX1201},
    {"gfx1250", ROCJITSU_CODE_TARGET_GFX1250},
    {"gfx1251", ROCJITSU_CODE_TARGET_GFX1251},
}};

} // namespace

std::string decode_target_description() {
  std::string description = "Target: auto";
  for (const NamedChoice<rj_code_target_id_t> &target : kTargets) {
    description += ", ";
    description += target.first;
  }
  return description;
}

util::Result DecodeTargetFlag::set_value(std::optional<rj_code_target_id_t> value,
                                         util::DiagnosticEmitter diagnostic) {
  if (!value) {
    value_.reset();
    return util::Result::success();
  }
  if (choice_name<rj_code_target_id_t>(kTargets, *value)) {
    value_ = value;
    return util::Result::success();
  }
  return diagnostic.emit() << "unsupported decoder target";
}

std::string DecodeTargetFlag::format_value() const {
  if (value_) {
    std::optional<std::string_view> name = choice_name<rj_code_target_id_t>(kTargets, *value_);
    if (name)
      return std::string(*name);
  }
  return "auto";
}

util::Result DecodeTargetFlag::parse_value(std::string_view text,
                                           util::DiagnosticEmitter diagnostic) {
  if (text == "auto")
    return set_value(std::nullopt, diagnostic);
  std::optional<rj_code_target_id_t> target = find_choice<rj_code_target_id_t>(kTargets, text);
  if (target)
    return set_value(*target, diagnostic);
  return diagnostic.emit() << "unsupported decoder target '" << text << "'";
}

util::Result DecodeCountFlag::set_value(uint64_t value, util::DiagnosticEmitter diagnostic) {
  if (value < minimum_ || value > std::numeric_limits<size_t>::max())
    return diagnostic.emit() << "expected an integer in [" << minimum_ << ", "
                             << std::numeric_limits<size_t>::max() << "]";
  value_ = static_cast<size_t>(value);
  return util::Result::success();
}

std::string DecodeCountFlag::format_value() const { return std::to_string(value_); }

util::Result DecodeCountFlag::parse_value(std::string_view text,
                                          util::DiagnosticEmitter diagnostic) {
  auto parsed = cli::parse_uint(text, diagnostic);
  if (parsed.failed())
    return util::Result::failure();
  return set_value(parsed.value(), diagnostic);
}

util::Result register_decode_benchmark_flags(cli::Command &command,
                                             const DecodeBenchmarkFlags &flags,
                                             util::DiagnosticEmitter diagnostic) {
  if (command
          .add_flags("Decoder settings", {flags.target, flags.iterations, flags.invalid_limit},
                     diagnostic)
          .failed())
    return util::Result::failure();
  return command.add_flag(flags.help, diagnostic);
}

} // namespace rocjitsu::tools
