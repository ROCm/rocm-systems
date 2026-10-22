// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/cli.h"

// Prototype-only, same-build plugin interface, not a public binary ABI.
// The host owns the deleter so unloading occurs after plugin destruction returns.
struct PluginHost {
  void *context;
  std::shared_ptr<util::cli::Flag> (*retain)(void *, util::cli::Flag *);
};

using RegisterPlugin = util::Result (*)(util::cli::Command &, const PluginHost &,
                                        util::DiagnosticEmitter);
