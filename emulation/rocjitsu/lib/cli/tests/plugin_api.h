// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/cli.h"

// Prototype-only, same-build plugin interface, not a public binary ABI.
// The host owns the deleter so unloading occurs after plugin destruction returns.
struct PluginHost {
  void *context;
  std::shared_ptr<rocjitsu::cli::Flag> (*retain)(void *, rocjitsu::cli::Flag *);
};

using RegisterPlugin = util::Result (*)(rocjitsu::cli::Command &, const PluginHost &,
                                        util::DiagnosticEmitter);
