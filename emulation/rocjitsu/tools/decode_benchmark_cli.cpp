// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_benchmark_cli.h"
#include "cli_choices.h"

#include <array>
#include <limits>

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

cli::Result DecodeTargetFlag::set_value(std::optional<rj_code_target_id_t> value,
                                        cli::DiagnosticEmitter diagnostic) {
  if (!value) {
    value_.reset();
    return cli::Result::success();
  }
  if (choice_name<rj_code_target_id_t>(kTargets, *value)) {
    value_ = value;
    return cli::Result::success();
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

cli::Result DecodeTargetFlag::parse_value(std::string_view text,
                                          cli::DiagnosticEmitter diagnostic) {
  if (text == "auto")
    return set_value(std::nullopt, diagnostic);
  std::optional<rj_code_target_id_t> target = find_choice<rj_code_target_id_t>(kTargets, text);
  if (target)
    return set_value(*target, diagnostic);
  return diagnostic.emit() << "unsupported decoder target '" << text << "'";
}

cli::Result DecodeCountFlag::set_value(uint64_t value, cli::DiagnosticEmitter diagnostic) {
  if (value < minimum_ || value > std::numeric_limits<size_t>::max())
    return diagnostic.emit() << "expected an integer in [" << minimum_ << ", "
                             << std::numeric_limits<size_t>::max() << "]";
  value_ = static_cast<size_t>(value);
  return cli::Result::success();
}

std::string DecodeCountFlag::format_value() const { return std::to_string(value_); }

cli::Result DecodeCountFlag::parse_value(std::string_view text, cli::DiagnosticEmitter diagnostic) {
  auto parsed = cli::parse_uint(text, diagnostic);
  if (parsed.failed())
    return cli::Result::failure();
  return set_value(parsed.value(), diagnostic);
}

cli::Result register_decode_benchmark_flags(cli::Command &command,
                                            const DecodeBenchmarkFlags &flags,
                                            cli::DiagnosticEmitter diagnostic) {
  if (command
          .add_flags("Decoder settings", {flags.target, flags.iterations, flags.invalid_limit},
                     diagnostic)
          .failed())
    return cli::Result::failure();
  return command.add_flag(flags.help, diagnostic);
}

} // namespace rocjitsu::tools
