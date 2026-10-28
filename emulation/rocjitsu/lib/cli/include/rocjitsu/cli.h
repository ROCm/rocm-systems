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

namespace rocjitsu::cli {

/// @brief Whether a flag consumes an argument or acts as a boolean switch.
///
/// A switch consumes no following argument and passes "true" unless an
/// explicit =value is given.
enum class ValueMode { Switch, Required };

/// @brief Static metadata for one flag spelling and its help entry.
struct FlagInfo {
  /// @brief Primary spelling, such as --output.
  std::string name;
  /// @brief Additional spellings that parse the same value.
  std::vector<std::string> aliases;
  /// @brief Help text shown below this flag.
  std::string description;
  /// @brief Whether this flag consumes a value.
  ValueMode value_mode = ValueMode::Required;
  /// @brief Help placeholder for required values.
  std::string value_placeholder = "VALUE";
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
  util::Result parse_and_set(std::string_view text, util::DiagnosticEmitter diagnostic);

protected:
  virtual util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) = 0;

private:
  FlagInfo info_;
};

/// @brief Parse a signed decimal integer for custom flags.
///
/// These helpers do not require flag metadata; callers provide the diagnostic
/// context they want users to see.
util::FailureOr<int64_t> parse_int(std::string_view text, util::DiagnosticEmitter diagnostic);
/// @brief Parse an unsigned decimal integer for custom flags.
util::FailureOr<uint64_t> parse_uint(std::string_view text, util::DiagnosticEmitter diagnostic);

class BoolFlag final : public Flag {
public:
  explicit BoolFlag(FlagInfo info, bool initial = false) : Flag(std::move(info)), value_(initial) {}
  bool value() const { return value_; }
  util::Result set_value(bool value, util::DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override;
  bool value_;
};

class Int64Flag final : public Flag {
public:
  explicit Int64Flag(FlagInfo info, int64_t initial = 0) : Flag(std::move(info)), value_(initial) {}
  int64_t value() const { return value_; }
  util::Result set_value(int64_t value, util::DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override;
  int64_t value_;
};

class UInt64Flag final : public Flag {
public:
  explicit UInt64Flag(FlagInfo info, uint64_t initial = 0)
      : Flag(std::move(info)), value_(initial) {}
  uint64_t value() const { return value_; }
  util::Result set_value(uint64_t value, util::DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override;
  uint64_t value_;
};

class StringFlag final : public Flag {
public:
  explicit StringFlag(FlagInfo info, std::string initial = {})
      : Flag(std::move(info)), value_(std::move(initial)) {}
  const std::string &value() const { return value_; }
  util::Result set_value(std::string value, util::DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override;
  std::string value_;
};

/// @brief A group of flags shown together in formatted help.
///
/// Every group contributes entries to its command's flag spelling namespace.
struct FlagGroup {
  std::string description;
  std::vector<std::shared_ptr<Flag>> flags;
};

/// @brief Where positional arguments are accepted for a command.
///
/// Anywhere accepts bare positional arguments after option parsing. AfterSeparator
/// requires callers to pass -- before any positional argument so launcher commands
/// can forward application argv without parsing application options as their own.
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
  /// @brief Return the help-only positional usage suffix.
  const std::string &positional_usage() const { return positional_usage_; }
  /// @brief Set the help-only positional usage suffix.
  ///
  /// This does not validate argument counts. An empty string suppresses the
  /// positional suffix in formatted help.
  void set_positional_usage(std::string usage) { positional_usage_ = std::move(usage); }
  /// @brief Add a child command before parsing seals the tree.
  ///
  /// The returned pointer is borrowed from this command and remains stable until
  /// this command is destroyed.
  util::FailureOr<Command *> add_command(std::string name, util::DiagnosticEmitter diagnostic,
                                         PositionalMode positional_mode = PositionalMode::Anywhere);
  /// @brief Register one flag before parsing.
  ///
  /// Before parsing, the same object is a no-op; conflicting spellings diagnose.
  util::Result add_flag(std::shared_ptr<Flag> flag, util::DiagnosticEmitter diagnostic);
  /// @brief Register a help section of flags before parsing.
  ///
  /// All flags still share the command's spelling namespace. The same flag
  /// object can be registered repeatedly as a no-op; spelling conflicts fail the
  /// whole batch without partially adding new flags.
  util::Result add_flags(std::string description, std::vector<std::shared_ptr<Flag>> flags,
                         util::DiagnosticEmitter diagnostic);
  /// @brief Find a flag by primary spelling or alias.
  ///
  /// Returns a retaining handle to the registered flag, or null if no spelling
  /// matches.
  std::shared_ptr<Flag> find_flag(std::string_view spelling) const;
  /// @brief Find a child command by name, or null if absent.
  Command *find_command(std::string_view name);
  /// @brief Find a child command by name, or null if absent.
  const Command *find_command(std::string_view name) const;
  /// @brief Return retaining handles for every flag on this command.
  std::vector<std::shared_ptr<Flag>> flags() const;
  /// @brief Return registered help sections in registration order.
  ///
  /// This borrowed view can be invalidated by subsequent registration.
  std::span<const FlagGroup> groups() const { return groups_; }
  /// @brief Return borrowed child-command pointers in registration order.
  std::vector<const Command *> commands() const;

private:
  friend util::FailureOr<struct ParsedArguments> parse(Command &, std::span<const std::string_view>,
                                                       util::DiagnosticEmitter);
  void seal();
  std::string name_;
  Command *parent_ = nullptr;
  bool sealed_ = false;
  PositionalMode positional_mode_;
  std::string positional_usage_ = "[arguments...]";
  std::vector<FlagGroup> groups_;
  std::vector<std::unique_ptr<Command>> commands_;
};

/// @brief Successful parse result.
struct ParsedArguments {
  /// @brief Borrowed selected command for caller dispatch.
  Command *command;
  /// @brief Owned positional arguments that remain valid independently.
  std::vector<std::string> positional;
  /// @brief Whether parsing encountered -- before collecting positional arguments.
  bool saw_separator = false;
};

/// @brief Parse arguments for a command tree.
///
/// Excludes argv[0]. Seals registration, retains current values, and stops at
/// the first failure. Parent flags must precede child selection. -- ends all
/// recognition. Repeated scalar flags apply in order; the last successful value
/// wins unless a custom flag implements append semantics. A command without a
/// child selection is returned to the caller for dispatch/help.
util::FailureOr<ParsedArguments> parse(Command &root, std::span<const std::string_view> arguments,
                                       util::DiagnosticEmitter diagnostic);

/// @brief Format help from registered command and flag metadata.
///
/// Pure formatting: no output streams or process control owned by the library.
std::string format_help(const Command &command);

} // namespace rocjitsu::cli
