// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugin_host.h"

#include <iostream>

int main(int argc, char **argv) {
  using namespace util::cli;
  auto report = [](std::string_view message) { std::cerr << message << '\n'; };
  DiagnosticEmitter diagnostic(report);
  Command root{"rj_cli_probe"};
  auto run = root.add_command("run", diagnostic);
  if (run.failed())
    return 1;
  auto limit =
      std::make_shared<IntFlag>(FlagInfo{"--limit", {"-l", "--max-items"}, "Item limit"}, 64);
  auto label = std::make_shared<StringFlag>(FlagInfo{"--label", {}, "Output label"});
  auto help =
      std::make_shared<BoolFlag>(FlagInfo{"--help", {"-h"}, "Show help", ValueMode::Switch});
  for (Command *command : {&root, run.value()}) {
    if (command->add_flag(help, diagnostic).failed())
      return 1;
  }
  if (run.value()->add_flags("Run settings", {limit, label}, diagnostic).failed())
    return 1;

  // Deliberately explicit bootstrap prefix. Plugin options are parsed only after
  // all libraries have registered; unknown option arity is never guessed.
  int first = 1;
  while (first < argc && std::string_view(argv[first]) == "--plugin") {
    if (++first == argc) {
      std::cerr << "--plugin requires a library path\n";
      return 1;
    }
    if (load_plugin(argv[first++], root, diagnostic).failed())
      return 1;
  }
  std::vector<std::string_view> args;
  for (int i = first; i < argc; ++i)
    args.emplace_back(argv[i]);
  auto parsed = parse(root, args, diagnostic);
  if (parsed.failed())
    return 1;
  Command &selected = *parsed.value().command;
  auto requested_help = selected.find_flag("--help");
  if (help->value() || (requested_help && requested_help->format_value() == "true")) {
    std::cout << format_help(selected);
    return 0;
  }
  if (!selected.commands().empty()) {
    std::cerr << selected.path() << ": a subcommand is required\n";
    return 1;
  }
  // A real consumer can keep the typed handle and observe generic updates.
  if (&selected == run.value() && limit->value() < 0) {
    std::cerr << "limit must be nonnegative for this command\n";
    return 1;
  }
  std::cout << "command=" << selected.path() << '\n';
  for (const auto &flag : selected.flags())
    std::cout << flag->info().name << '=' << flag->format_value() << '\n';
  for (const auto &positional : parsed.value().positional)
    std::cout << "positional=" << positional << '\n';
  return 0;
}
