// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/cli.h"
#include "rocjitsu/config/config_common.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::tools {
namespace ucli = ::rocjitsu::cli;

class NonEmptyStringFlag final : public ucli::Flag {
public:
  explicit NonEmptyStringFlag(ucli::FlagInfo info, std::string requirement)
      : ucli::Flag(std::move(info)), requirement_(std::move(requirement)) {}
  const std::string &value() const { return value_; }
  util::Result set_value(std::string value, util::DiagnosticEmitter diagnostic) {
    if (value.empty())
      return diagnostic.emit() << requirement_;
    value_ = std::move(value);
    return util::Result::success();
  }
  std::string format_value() const override { return value_; }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    return set_value(std::string(text), diagnostic);
  }
  std::string requirement_;
  std::string value_;
};

class StringListFlag final : public ucli::Flag {
public:
  explicit StringListFlag(ucli::FlagInfo info, std::string requirement)
      : ucli::Flag(std::move(info)), requirement_(std::move(requirement)) {}
  const std::vector<std::string> &values() const { return values_; }
  util::Result add_value(std::string value, util::DiagnosticEmitter diagnostic) {
    if (value.empty())
      return diagnostic.emit() << requirement_;
    values_.push_back(std::move(value));
    return util::Result::success();
  }
  std::string format_value() const override {
    std::string text;
    for (const auto &value : values_) {
      if (!text.empty())
        text += ",";
      text += value;
    }
    return text;
  }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    return add_value(std::string(text), diagnostic);
  }
  std::string requirement_;
  std::vector<std::string> values_;
};

// Absence is a value: an omitted override must not replace a JSON setting.
class OptionalUInt32Flag final : public ucli::Flag {
public:
  explicit OptionalUInt32Flag(ucli::FlagInfo info, uint32_t maximum = UINT32_MAX)
      : ucli::Flag(std::move(info)), maximum_(maximum) {}
  std::optional<uint32_t> value() const { return value_; }
  util::Result set_value(std::optional<uint32_t> value, util::DiagnosticEmitter diagnostic) {
    if (value && *value > maximum_)
      return diagnostic.emit() << "expected an unsigned integer at most " << maximum_;
    value_ = value;
    return util::Result::success();
  }
  std::string format_value() const override { return value_ ? std::to_string(*value_) : "unset"; }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    auto parsed = ucli::parse_uint(text, diagnostic);
    if (parsed.failed())
      return util::Result::failure();
    if (parsed.value() > maximum_)
      return diagnostic.emit() << "expected an unsigned integer at most " << maximum_;
    return set_value(static_cast<uint32_t>(parsed.value()), diagnostic);
  }
  std::optional<uint32_t> value_;
  uint32_t maximum_;
};

} // namespace rocjitsu::tools

namespace rocjitsu::launcher {
namespace rjtools = ::rocjitsu::tools;
namespace ucli = ::rocjitsu::cli;

class WaitCheckingFlag final : public ucli::Flag {
public:
  using ucli::Flag::Flag;
  std::optional<config::WaitChecking> value() const { return value_; }
  util::Result set_value(std::optional<config::WaitChecking> value,
                         util::DiagnosticEmitter diagnostic) {
    if (value && *value != config::WaitChecking::Off && *value != config::WaitChecking::On &&
        *value != config::WaitChecking::All)
      return diagnostic.emit() << "requires on, off, or all";
    value_ = value;
    return util::Result::success();
  }
  std::string format_value() const override {
    if (!value_)
      return "unset";
    switch (*value_) {
    case config::WaitChecking::Off:
      return "off";
    case config::WaitChecking::On:
      return "on";
    case config::WaitChecking::All:
      return "all";
    }
    return "unset";
  }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    const auto value = config::parse_wait_checking(text);
    if (!value)
      return diagnostic.emit() << "requires on, off, or all";
    return set_value(value, diagnostic);
  }
  std::optional<config::WaitChecking> value_;
};

struct LaunchFlags {
  std::shared_ptr<rjtools::NonEmptyStringFlag> config =
      std::make_shared<rjtools::NonEmptyStringFlag>(
          ucli::FlagInfo{"--config", {}, "Simulation config JSON (required)"},
          "requires a nonempty path");
  std::shared_ptr<rjtools::NonEmptyStringFlag> node = std::make_shared<rjtools::NonEmptyStringFlag>(
      ucli::FlagInfo{
          "--node",
          {},
          "Select one local node from a system manifest. Required for multi-node launches. "
          "Not available with --attach"},
      "requires a nonempty ID");
  std::shared_ptr<rjtools::OptionalUInt32Flag> cpu_thread_budget =
      std::make_shared<rjtools::OptionalUInt32Flag>(ucli::FlagInfo{
          "--cpu-thread-budget",
          {},
          "Override CPU thread budget for this launch; zero selects automatic sizing. "
          "Omission preserves JSON and launch overrides do not rewrite the source "
          "config. Not available with --attach"});
  std::shared_ptr<WaitCheckingFlag> wait_checking =
      std::make_shared<WaitCheckingFlag>(ucli::FlagInfo{
          "--wait-checking",
          {},
          "Override wait checks: off disables checks, on enables ordinary checks, and all includes "
          "gfx1250 XCNT. Omission preserves JSON and launch overrides do not rewrite the source "
          "config. Not available with --attach"});
  std::shared_ptr<ucli::BoolFlag> thread_budget_table = std::make_shared<ucli::BoolFlag>(
      ucli::FlagInfo{"--thread-budget-table",
                     {},
                     "Print engine/dispatch allocations without running a VM",
                     ucli::ValueMode::Switch});
  std::shared_ptr<ucli::BoolFlag> daemon = std::make_shared<ucli::BoolFlag>(
      ucli::FlagInfo{"--daemon",
                     {},
                     "Run a daemon, optionally launching an application",
                     ucli::ValueMode::Switch});
  std::shared_ptr<ucli::BoolFlag> attach = std::make_shared<ucli::BoolFlag>(ucli::FlagInfo{
      "--attach", {}, "Attach the application to a running daemon", ucli::ValueMode::Switch});
  std::shared_ptr<rjtools::StringListFlag> vfio_socket = std::make_shared<rjtools::StringListFlag>(
      ucli::FlagInfo{
          "--vfio-socket",
          {},
          "Serve a VMM on this AF_UNIX socket. Requires ROCJITSU_ENABLE_VFIO=ON. Repeat once "
          "per local GPU in ordinal order"},
      "requires a nonempty path");
  std::shared_ptr<rjtools::OptionalUInt32Flag> vfio_ready_fd =
      std::make_shared<rjtools::OptionalUInt32Flag>(
          ucli::FlagInfo{
              "--vfio-ready-fd", {}, "Write readiness to this nonnegative file descriptor"},
          std::numeric_limits<int>::max());
  std::shared_ptr<ucli::BoolFlag> check_vfio_user =
      std::make_shared<ucli::BoolFlag>(ucli::FlagInfo{"--check-vfio-user",
                                                      {},
                                                      "Report whether VFIO-user support is enabled",
                                                      ucli::ValueMode::Switch});
  std::shared_ptr<ucli::BoolFlag> version = std::make_shared<ucli::BoolFlag>(
      ucli::FlagInfo{"--version", {"-v"}, "Print version", ucli::ValueMode::Switch});
  std::shared_ptr<ucli::BoolFlag> help = std::make_shared<ucli::BoolFlag>(
      ucli::FlagInfo{"--help", {"-h"}, "Show help", ucli::ValueMode::Switch});

  util::Result register_flags(ucli::Command &command, util::DiagnosticEmitter diagnostic) const {
    if (command
            .add_flags("Simulation configuration",
                       {config, node, cpu_thread_budget, wait_checking, thread_budget_table},
                       diagnostic)
            .failed() ||
        command.add_flags("Launch modes", {daemon, attach, vfio_socket, vfio_ready_fd}, diagnostic)
            .failed())
      return util::Result::failure();
    return command.add_flags("Information", {check_vfio_user, version, help}, diagnostic);
  }
};

} // namespace rocjitsu::launcher
