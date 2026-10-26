// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/diagnostic.h"
#include "util/result.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace util::cli {

using util::DiagnosticEmitter;
using util::FailureOr;
using util::Result;

/// @brief Whether a flag consumes an argument or acts as a boolean switch.
///
/// A switch consumes no following argument and passes "true" unless an
/// explicit =value is given.
enum class ValueMode { Switch, Required };

/// @brief Static metadata for one flag spelling and its help entry.
struct FlagInfo {
  std::string name;
  std::vector<std::string> aliases;
  std::string description;
  ValueMode value_mode = ValueMode::Required;
};

/// @brief Base class for one registered flag and its strongly typed value.
///
/// A flag object owns immutable metadata and its current value. Commands store
/// shared ownership so callers can keep typed handles while generic parsing and
/// help formatting use the base class.
class Flag {
public:
  explicit Flag(FlagInfo info) : info_(std::move(info)) {}
  virtual ~Flag() = default;
  Flag(const Flag &) = delete;
  Flag &operator=(const Flag &) = delete;
  const FlagInfo &info() const { return info_; }
  virtual std::string format_value() const = 0;
  Result parse_and_set(std::string_view text, DiagnosticEmitter diagnostic);

protected:
  virtual Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) = 0;

private:
  FlagInfo info_;
};

/// @brief Parse a signed decimal integer for custom flags.
///
/// These helpers do not require flag metadata; callers provide the diagnostic
/// context they want users to see.
FailureOr<int64_t> parse_int(std::string_view text, DiagnosticEmitter diagnostic);
/// @brief Parse an unsigned decimal integer for custom flags.
FailureOr<uint64_t> parse_uint(std::string_view text, DiagnosticEmitter diagnostic);

class BoolFlag final : public Flag {
public:
  explicit BoolFlag(FlagInfo info, bool initial = false) : Flag(std::move(info)), value_(initial) {}
  bool value() const { return value_; }
  Result set_value(bool value, DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override;
  bool value_;
};

class IntFlag final : public Flag {
public:
  explicit IntFlag(FlagInfo info, int64_t initial = 0) : Flag(std::move(info)), value_(initial) {}
  int64_t value() const { return value_; }
  Result set_value(int64_t value, DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override;
  int64_t value_;
};

class UIntFlag final : public Flag {
public:
  explicit UIntFlag(FlagInfo info, uint64_t initial = 0) : Flag(std::move(info)), value_(initial) {}
  uint64_t value() const { return value_; }
  Result set_value(uint64_t value, DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override;
  uint64_t value_;
};

class StringFlag final : public Flag {
public:
  explicit StringFlag(FlagInfo info, std::string initial = {})
      : Flag(std::move(info)), value_(std::move(initial)) {}
  const std::string &value() const { return value_; }
  Result set_value(std::string value, DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override;
  std::string value_;
};

/// @brief A help section, not a scope.
///
/// Every group shares its command's flag namespace.
struct FlagGroup {
  std::string description;
  std::vector<std::shared_ptr<Flag>> flags;
};

enum class PositionalMode { Anywhere, AfterSeparator };

/// @brief A command scope containing child commands and flags.
///
/// Command and Flag addresses stay stable as registration adds entries. All
/// operations require external synchronization. Metadata is read-only to
/// callers. Parsing seals registration for the command tree, but values remain
/// writable through their typed flag handles.
class Command {
public:
  explicit Command(std::string name, PositionalMode positional_mode = PositionalMode::Anywhere)
      : name_(std::move(name)), positional_mode_(positional_mode) {}
  Command(const Command &) = delete;
  Command &operator=(const Command &) = delete;

  const std::string &name() const { return name_; }
  std::string path() const;
  PositionalMode positional_mode() const { return positional_mode_; }
  const std::string &positional_usage() const { return positional_usage_; }
  void set_positional_usage(std::string usage) { positional_usage_ = std::move(usage); }
  FailureOr<Command *> add_command(std::string name, DiagnosticEmitter diagnostic,
                                   PositionalMode positional_mode = PositionalMode::Anywhere);
  /// @brief Register one flag before parsing.
  ///
  /// Before parsing, the same object is a no-op; conflicting spellings diagnose.
  Result add_flag(std::shared_ptr<Flag> flag, DiagnosticEmitter diagnostic);
  Result add_flags(std::string description, std::vector<std::shared_ptr<Flag>> flags,
                   DiagnosticEmitter diagnostic);
  std::shared_ptr<Flag> find_flag(std::string_view spelling) const;
  Command *find_command(std::string_view name);
  const Command *find_command(std::string_view name) const;
  std::vector<std::shared_ptr<Flag>> flags() const;
  /// @brief Return registered help sections in registration order.
  ///
  /// This borrowed view can be invalidated by subsequent registration.
  std::span<const FlagGroup> groups() const { return groups_; }
  std::vector<const Command *> commands() const;

private:
  friend FailureOr<struct ParsedArguments> parse(Command &, std::span<const std::string_view>,
                                                 DiagnosticEmitter);
  void seal();
  std::string name_;
  Command *parent_ = nullptr;
  bool sealed_ = false;
  PositionalMode positional_mode_;
  std::string positional_usage_ = "[arguments...]";
  std::vector<FlagGroup> groups_;
  std::vector<std::unique_ptr<Command>> commands_;
};

struct ParsedArguments {
  Command *command;
  std::vector<std::string> positional;
  bool saw_separator = false;
};

/// @brief Parse arguments for a command tree.
///
/// Excludes argv[0]. Seals registration, retains current values, and stops at
/// the first failure. Parent flags must precede child selection. -- ends all
/// recognition. Repeated scalar flags apply in order; the last successful value
/// wins unless a custom flag implements append semantics. A command without a
/// child selection is returned to the caller for dispatch/help.
FailureOr<ParsedArguments> parse(Command &root, std::span<const std::string_view> arguments,
                                 DiagnosticEmitter diagnostic);

/// @brief Format help from registered command and flag metadata.
///
/// Pure formatting: no output streams or process control owned by the library.
std::string format_help(const Command &command);

} // namespace util::cli
