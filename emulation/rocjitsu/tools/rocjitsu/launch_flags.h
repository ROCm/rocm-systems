// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/cli.h"
#include "rocjitsu/config/config_common.h"

#include <limits>
#include <optional>

namespace rocjitsu::cli {

class LaunchPathFlag final : public Flag {
public:
  using Flag::Flag;
  const std::string &value() const { return value_; }
  Result set_value(std::string value, DiagnosticEmitter diagnostic) {
    if (value.empty())
      return diagnostic.emit() << "requires a nonempty path";
    value_ = std::move(value);
    return Result::success();
  }
  std::string format_value() const override { return value_; }

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override {
    return set_value(std::string(text), diagnostic);
  }
  std::string value_;
};

// Absence is a value: an omitted override must not replace a JSON setting.
class OptionalUInt32Flag final : public Flag {
public:
  explicit OptionalUInt32Flag(FlagInfo info, uint32_t maximum = UINT32_MAX)
      : Flag(std::move(info)), maximum_(maximum) {}
  std::optional<uint32_t> value() const { return value_; }
  Result set_value(std::optional<uint32_t> value, DiagnosticEmitter diagnostic) {
    if (value && *value > maximum_)
      return diagnostic.emit() << "expected an unsigned integer at most " << maximum_;
    value_ = value;
    return Result::success();
  }
  std::string format_value() const override { return value_ ? std::to_string(*value_) : "unset"; }

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override {
    auto parsed = parse_uint(text, diagnostic);
    if (parsed.failed())
      return Result::failure();
    if (parsed.value() > maximum_)
      return diagnostic.emit() << "expected an unsigned integer at most " << maximum_;
    return set_value(static_cast<uint32_t>(parsed.value()), diagnostic);
  }
  std::optional<uint32_t> value_;
  uint32_t maximum_;
};

class WaitCheckingFlag final : public Flag {
public:
  using Flag::Flag;
  std::optional<config::WaitChecking> value() const { return value_; }
  Result set_value(std::optional<config::WaitChecking> value, DiagnosticEmitter diagnostic) {
    if (value && *value != config::WaitChecking::Off && *value != config::WaitChecking::On &&
        *value != config::WaitChecking::All)
      return diagnostic.emit() << "requires on, off, or all";
    value_ = value;
    return Result::success();
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
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override {
    const auto value = config::parse_wait_checking(text);
    if (!value)
      return diagnostic.emit() << "requires on, off, or all";
    return set_value(value, diagnostic);
  }
  std::optional<config::WaitChecking> value_;
};

struct LaunchFlags {
  std::shared_ptr<LaunchPathFlag> config = std::make_shared<LaunchPathFlag>(
      FlagInfo{"--config", {}, "Simulation config JSON (required)"});
  std::shared_ptr<OptionalUInt32Flag> cpu_budget = std::make_shared<OptionalUInt32Flag>(
      FlagInfo{"--cpu-thread-budget",
               {},
               "Override CPU thread budget for this launch; zero selects automatic sizing. "
               "Omission preserves JSON. Not available with --attach"});
  std::shared_ptr<WaitCheckingFlag> wait = std::make_shared<WaitCheckingFlag>(
      FlagInfo{"--wait-checking",
               {},
               "Override checks: on, off, or all (includes gfx1250 XCNT). Omission preserves JSON. "
               "Not available with --attach"});
  std::shared_ptr<BoolFlag> thread_table =
      std::make_shared<BoolFlag>(FlagInfo{"--thread-budget-table",
                                          {},
                                          "Print engine/dispatch allocations without running a VM",
                                          ValueMode::Switch});
  std::shared_ptr<BoolFlag> daemon = std::make_shared<BoolFlag>(FlagInfo{
      "--daemon", {}, "Run a daemon, optionally launching an application", ValueMode::Switch});
  std::shared_ptr<BoolFlag> attach = std::make_shared<BoolFlag>(
      FlagInfo{"--attach", {}, "Attach the application to a running daemon", ValueMode::Switch});
  std::shared_ptr<LaunchPathFlag> vfio_socket = std::make_shared<LaunchPathFlag>(
      FlagInfo{"--vfio-socket",
               {},
               "Serve a VMM on this AF_UNIX socket (requires VFIO support and shared guest RAM)"});
  std::shared_ptr<OptionalUInt32Flag> vfio_ready_fd = std::make_shared<OptionalUInt32Flag>(
      FlagInfo{"--vfio-ready-fd", {}, "Write readiness to this nonnegative file descriptor"},
      std::numeric_limits<int>::max());
  std::shared_ptr<BoolFlag> check_vfio = std::make_shared<BoolFlag>(FlagInfo{
      "--check-vfio-user", {}, "Report whether VFIO-user support is enabled", ValueMode::Switch});
  std::shared_ptr<BoolFlag> version =
      std::make_shared<BoolFlag>(FlagInfo{"--version", {"-v"}, "Print version", ValueMode::Switch});
  std::shared_ptr<BoolFlag> help =
      std::make_shared<BoolFlag>(FlagInfo{"--help", {"-h"}, "Show help", ValueMode::Switch});

  Result register_flags(Command &command, DiagnosticEmitter diagnostic) const {
    if (command
            .add_flags("Simulation configuration", {config, cpu_budget, wait, thread_table},
                       diagnostic)
            .failed() ||
        command.add_flags("Launch modes", {daemon, attach, vfio_socket, vfio_ready_fd}, diagnostic)
            .failed())
      return Result::failure();
    return command.add_flags("Information", {check_vfio, version, help}, diagnostic);
  }
};

} // namespace rocjitsu::cli
