// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "plugin_api.h"

#include <dlfcn.h>

inline util::FailureOr<std::shared_ptr<void>>
load_plugin(const char *path, util::cli::Command &root, util::DiagnosticEmitter diagnostic) {
  void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return diagnostic.emit() << "cannot load plugin '" << path << "': " << dlerror();
  std::shared_ptr<void> library(handle, [](void *p) { dlclose(p); });
  auto entry = reinterpret_cast<RegisterPluginEntry>(dlsym(handle, "register_cli_plugin"));
  if (!entry)
    return diagnostic.emit() << "missing register_cli_plugin: " << dlerror();
  RegisterPlugin registration = entry();
  if (!registration)
    return diagnostic.emit() << "invalid register_cli_plugin entry";

  PluginHost host{
      &library,
      [](void *context, util::cli::Flag *flag) {
        auto pin = *static_cast<std::shared_ptr<void> *>(context);
        return std::shared_ptr<util::cli::Flag>(
            flag, [pin = std::move(pin)](util::cli::Flag *value) { delete value; });
      },
  };
  if (registration(root, host, diagnostic).failed())
    return util::Result::failure();
  return library;
}
