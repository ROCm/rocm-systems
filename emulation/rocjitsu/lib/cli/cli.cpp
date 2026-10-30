// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/cli.h"

#include <algorithm>
#include <charconv>
#include <sstream>
#include <unordered_set>

namespace rocjitsu::cli {

using util::DiagnosticEmitter;
using util::FailureOr;
using util::Result;

namespace {

bool valid_word(std::string_view word) {
  return !word.empty() && std::ranges::all_of(word, [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
           c == '_';
  });
}

bool valid_spelling(std::string_view spelling) {
  if (spelling.starts_with("--")) {
    spelling.remove_prefix(2);
    return valid_word(spelling) && spelling.front() != '-';
  }
  return spelling.size() == 2 && spelling.front() == '-' && valid_word(spelling.substr(1)) &&
         spelling.back() != '-';
}

bool equals_ignore_case(std::string_view lhs, std::string_view rhs) {
  auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
  return lhs.size() == rhs.size() && std::ranges::equal(lhs, rhs, {}, lower, lower);
}

template <typename T>
FailureOr<T> parse_integer(std::string_view text, DiagnosticEmitter diagnostic,
                           std::string_view kind) {
  if (text.empty())
    return diagnostic.emit() << "expected " << kind << " integer, got an empty value";
  T parsed{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size())
    return diagnostic.emit() << "expected " << kind << " decimal integer, got '" << text << "'";
  return parsed;
}

} // namespace

FailureOr<int64_t> parse_int(std::string_view text, DiagnosticEmitter diagnostic) {
  return parse_integer<int64_t>(text, diagnostic, "signed");
}

FailureOr<uint64_t> parse_uint(std::string_view text, DiagnosticEmitter diagnostic) {
  return parse_integer<uint64_t>(text, diagnostic, "unsigned");
}

Result BoolFlag::set_value(bool value, DiagnosticEmitter) {
  value_ = value;
  return Result::success();
}

std::string BoolFlag::format_value() const { return value_ ? "true" : "false"; }

Result BoolFlag::parse_value(std::string_view text, DiagnosticEmitter diagnostic) {
  if (equals_ignore_case(text, "1") || equals_ignore_case(text, "yes") ||
      equals_ignore_case(text, "y") || equals_ignore_case(text, "true") ||
      equals_ignore_case(text, "t") || equals_ignore_case(text, "on"))
    return set_value(true, diagnostic);
  if (equals_ignore_case(text, "0") || equals_ignore_case(text, "no") ||
      equals_ignore_case(text, "n") || equals_ignore_case(text, "false") ||
      equals_ignore_case(text, "f") || equals_ignore_case(text, "off"))
    return set_value(false, diagnostic);
  return diagnostic.emit() << "expected boolean value, got '" << text << "'";
}

Result Int64Flag::set_value(int64_t value, DiagnosticEmitter) {
  value_ = value;
  return Result::success();
}

std::string Int64Flag::format_value() const { return std::to_string(value_); }

Result Int64Flag::parse_value(std::string_view text, DiagnosticEmitter diagnostic) {
  FailureOr<int64_t> value = parse_int(text, diagnostic);
  if (value.failed())
    return Result::failure();
  return set_value(value.value(), diagnostic);
}

Result UInt64Flag::set_value(uint64_t value, DiagnosticEmitter) {
  value_ = value;
  return Result::success();
}

std::string UInt64Flag::format_value() const { return std::to_string(value_); }

Result UInt64Flag::parse_value(std::string_view text, DiagnosticEmitter diagnostic) {
  FailureOr<uint64_t> value = parse_uint(text, diagnostic);
  if (value.failed())
    return Result::failure();
  return set_value(value.value(), diagnostic);
}

Result StringFlag::set_value(std::string value, DiagnosticEmitter) {
  value_ = std::move(value);
  return Result::success();
}

std::string StringFlag::format_value() const { return value_; }

Result StringFlag::parse_value(std::string_view text, DiagnosticEmitter diagnostic) {
  return set_value(std::string(text), diagnostic);
}

Result Flag::parse_and_set(std::string_view text, DiagnosticEmitter diagnostic) {
  bool emitted = false;
  auto contextual = [&](std::string_view message) {
    emitted = true;
    (void)(diagnostic.emit() << info_.name << ": " << message).emit();
  };
  Result result = parse_value(text, DiagnosticEmitter(contextual));
  if (result.failed() && !emitted)
    return diagnostic.emit() << info_.name << ": invalid value '" << text << "'";
  return result;
}

std::string Command::path() const { return parent_ ? parent_->path() + " " + name_ : name_; }

FailureOr<Command *> Command::add_command(std::string name, DiagnosticEmitter diagnostic,
                                          PositionalMode positional_mode) {
  if (sealed_)
    return diagnostic.emit() << path() << ": registration is closed";
  if (!valid_word(name) || name.front() == '-')
    return diagnostic.emit() << path() << ": invalid command name '" << name << "'";
  if (find_command(name))
    return diagnostic.emit() << path() << ": duplicate command '" << name << "'";
  auto command = std::make_unique<Command>(std::move(name), positional_mode);
  command->parent_ = this;
  Command *result = command.get();
  commands_.push_back(std::move(command));
  return result;
}

Result Command::add_flag(std::shared_ptr<Flag> flag, DiagnosticEmitter diagnostic) {
  return add_flags({}, {std::move(flag)}, diagnostic);
}

Result Command::add_flags(std::string description, std::vector<std::shared_ptr<Flag>> flags,
                          DiagnosticEmitter diagnostic) {
  if (sealed_)
    return diagnostic.emit() << path() << ": registration is closed";
  std::unordered_set<std::string_view> unique;
  std::unordered_set<const Flag *> objects;
  std::vector<std::shared_ptr<Flag>> additions;
  for (const auto &flag : flags) {
    if (!flag)
      return diagnostic.emit() << path() << ": null flag";
    const FlagInfo &info = flag->info();
    if (info.value_mode == ValueMode::Switch && !flag->allows_switch_mode())
      return diagnostic.emit() << path() << ": " << info.name
                               << " cannot use switch mode with this flag type";
    if (find_flag(info.name).get() == flag.get() || !objects.insert(flag.get()).second)
      continue;
    additions.push_back(flag);
    std::vector<std::string_view> names{info.name};
    for (const auto &alias : info.aliases)
      names.push_back(alias);
    for (std::string_view name : names) {
      if (!valid_spelling(name))
        return diagnostic.emit() << path() << ": invalid flag spelling '" << name << "'";
      if (!unique.insert(name).second)
        return diagnostic.emit() << path() << ": repeated spelling '" << name << "'";
      if (std::shared_ptr<Flag> existing = find_flag(name))
        return diagnostic.emit() << path() << ": spelling '" << name << "' for " << info.name
                                 << " conflicts with " << existing->info().name;
    }
  }
  if (additions.empty())
    return Result::success();
  // Consecutive ungrouped additions share an Options section. Named batches
  // always start a section; preserve registration order, even across plugins.
  if (description.empty() && !groups_.empty() && groups_.back().description.empty()) {
    std::vector<std::shared_ptr<Flag>> &entries = groups_.back().flags;
    entries.insert(entries.end(), additions.begin(), additions.end());
  } else {
    groups_.push_back({std::move(description), std::move(additions)});
  }
  return Result::success();
}

std::shared_ptr<Flag> Command::find_flag(std::string_view spelling) const {
  for (const auto &group : groups_)
    for (const auto &flag : group.flags) {
      const FlagInfo &info = flag->info();
      if (info.name == spelling || std::ranges::find(info.aliases, spelling) != info.aliases.end())
        return flag;
    }
  return nullptr;
}

Command *Command::find_command(std::string_view name) {
  for (const auto &command : commands_)
    if (command->name() == name)
      return command.get();
  return nullptr;
}

const Command *Command::find_command(std::string_view name) const {
  for (const auto &command : commands_)
    if (command->name() == name)
      return command.get();
  return nullptr;
}

std::vector<std::shared_ptr<Flag>> Command::flags() const {
  std::vector<std::shared_ptr<Flag>> result;
  for (const auto &group : groups_)
    result.insert(result.end(), group.flags.begin(), group.flags.end());
  return result;
}

std::vector<const Command *> Command::commands() const {
  std::vector<const Command *> result;
  for (const auto &command : commands_)
    result.push_back(command.get());
  return result;
}

void Command::seal() {
  sealed_ = true;
  for (const auto &command : commands_)
    command->seal();
}

FailureOr<ParsedArguments> parse(Command &root, std::span<const std::string_view> arguments,
                                 DiagnosticEmitter diagnostic) {
  root.seal();
  ParsedArguments result{&root, {}};
  bool positional_only = false;
  for (size_t i = 0; i < arguments.size(); ++i) {
    std::string_view arg = arguments[i];
    if (positional_only) {
      result.positional.emplace_back(arg);
      continue;
    }
    if (arg == "--") {
      positional_only = true;
      result.saw_separator = true;
      continue;
    }
    if (arg.size() > 1 && arg.front() == '-') {
      const size_t equals = arg.find('=');
      const std::string_view spelling = arg.substr(0, equals);
      std::shared_ptr<Flag> flag = result.command->find_flag(spelling);
      if (!flag)
        return diagnostic.emit() << result.command->path() << ": unknown flag '" << spelling << "'";
      std::string_view value;
      if (equals != std::string_view::npos)
        value = arg.substr(equals + 1);
      else if (flag->info().value_mode == ValueMode::Switch)
        value = "true";
      else {
        if (i + 1 == arguments.size() || arguments[i + 1] == "--")
          return diagnostic.emit()
                 << result.command->path() << ": " << spelling << " requires a value";
        value = arguments[++i];
      }
      auto contextual = [&](std::string_view message) {
        (void)(diagnostic.emit() << result.command->path() << ": " << message).emit();
      };
      if (flag->parse_and_set(value, DiagnosticEmitter(contextual)).failed())
        return Result::failure();
      continue;
    }
    if (!result.command->commands_.empty()) {
      Command *next = result.command->find_command(arg);
      if (!next)
        return diagnostic.emit() << result.command->path() << ": unknown command '" << arg << "'";
      result.command = next;
    } else {
      if (result.command->positional_mode_ == PositionalMode::AfterSeparator)
        return diagnostic.emit() << result.command->path()
                                 << ": expected -- before application argument '" << arg << "'";
      result.positional.emplace_back(arg);
    }
  }
  return result;
}

std::string format_help(const Command &command) {
  std::ostringstream out;
  const auto write_positional_usage = [&]() {
    out << (command.positional_mode() == PositionalMode::AfterSeparator ? " --" : " [--]") << ' '
        << command.positional_usage();
  };
  out << "Usage: " << command.path() << " [options]";
  const bool has_commands = !command.commands().empty();
  if (has_commands)
    out << " <command>";
  if (!command.positional_usage().empty()) {
    if (has_commands)
      out << "\n       " << command.path() << " [options]";
    write_positional_usage();
  }
  out << "\n";
  for (const auto &group : command.groups()) {
    out << "\n" << (group.description.empty() ? "Options" : group.description) << ":\n";
    for (const auto &flag : group.flags) {
      out << "  " << flag->info().name;
      for (const std::string &alias : flag->info().aliases)
        out << ", " << alias;
      if (flag->info().value_mode == ValueMode::Required)
        out << ' '
            << (flag->info().value_placeholder.empty() ? "VALUE" : flag->info().value_placeholder);
      out << "\n      " << flag->info().description << '\n';
    }
  }
  if (!command.commands().empty()) {
    out << "\nCommands:\n";
    for (const Command *child : command.commands())
      out << "  " << child->name() << '\n';
  }
  return out.str();
}

} // namespace rocjitsu::cli
