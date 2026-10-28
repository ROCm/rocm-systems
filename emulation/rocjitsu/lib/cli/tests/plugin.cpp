// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugin_api.h"

namespace {

enum class Format { Text, Json };

class FormatFlag final : public rocjitsu::cli::Flag {
public:
  FormatFlag() : Flag({"--format", {"-f"}, "Output format: text or json"}) {}
  Format value() const { return value_; }
  util::Result set_value(Format value, util::DiagnosticEmitter diagnostic) {
    if (value != Format::Text && value != Format::Json)
      return diagnostic.emit() << "unsupported format";
    value_ = value;
    return util::Result::success();
  }
  std::string format_value() const override { return value() == Format::Text ? "text" : "json"; }

private:
  util::Result parse_value(std::string_view text, util::DiagnosticEmitter diagnostic) override {
    if (text == "text")
      return set_value(Format::Text, diagnostic);
    if (text == "json")
      return set_value(Format::Json, diagnostic);
    return diagnostic.emit() << "expected text or json";
  }

  Format value_ = Format::Text;
};

util::Result register_cli_plugin_impl(rocjitsu::cli::Command &root, const PluginHost &host,
                                      util::DiagnosticEmitter diagnostic) {
  using namespace rocjitsu::cli;
  auto plugin = root.add_command("plugin", diagnostic);
  if (plugin.failed())
    return Result::failure();
  auto inspect = plugin.value()->add_command("inspect", diagnostic);
  if (inspect.failed())
    return Result::failure();
  for (Command *command : {plugin.value(), inspect.value()}) {
    if (command
            ->add_flag(
                host.retain(host.context,
                            new BoolFlag({"--help", {"-h"}, "Show help", ValueMode::Switch})),
                diagnostic)
            .failed())
      return Result::failure();
  }
  return inspect.value()->add_flags("Output formatting",
                                    {host.retain(host.context, new FormatFlag)}, diagnostic);
}

} // namespace

extern "C" __attribute__((visibility("default"))) RegisterPlugin register_cli_plugin() {
  return register_cli_plugin_impl;
}
