// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/cli.h"
#include "rocjitsu/code/rj_code.h"

#include <optional>
#include <string>

namespace rocjitsu::tools {
namespace cli = ::rocjitsu::cli;

class DecodeTargetFlag final : public cli::Flag {
public:
  using Flag::Flag;
  std::optional<rj_code_target_id_t> value() const { return value_; }
  util::Result set_value(std::optional<rj_code_target_id_t> value,
                         util::DiagnosticEmitter diagnostic);
  std::string format_value() const override;

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override;
  std::optional<rj_code_target_id_t> value_;
};

// Constrains both typed and textual updates to the benchmark's native size range.
class DecodeCountFlag final : public cli::Flag {
public:
  DecodeCountFlag(cli::FlagInfo info, size_t initial, size_t minimum)
      : Flag(std::move(info)), value_(initial), minimum_(minimum) {}
  size_t value() const { return value_; }
  util::Result set_value(uint64_t value, util::DiagnosticEmitter diagnostic);
  std::string format_value() const override;

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override;
  size_t value_;
  size_t minimum_;
};

struct DecodeBenchmarkFlags {
  DecodeBenchmarkFlags();

  std::shared_ptr<DecodeTargetFlag> target;
  std::shared_ptr<DecodeCountFlag> iterations = std::make_shared<DecodeCountFlag>(
      cli::FlagInfo{"--iterations", {"-n"}, "Positive iteration count (default: 1)"}, 1, 1);
  std::shared_ptr<DecodeCountFlag> invalid_limit = std::make_shared<DecodeCountFlag>(
      cli::FlagInfo{"--invalid-limit", {}, "Maximum invalid samples (default: 65536)"}, 65536, 0);
  std::shared_ptr<cli::BoolFlag> help = std::make_shared<cli::BoolFlag>(
      cli::FlagInfo{"--help", {"-h"}, "Show help", cli::ValueMode::Switch});
};

util::Result register_decode_benchmark_flags(cli::Command &command,
                                             const DecodeBenchmarkFlags &flags,
                                             util::DiagnosticEmitter diagnostic);

} // namespace rocjitsu::tools
