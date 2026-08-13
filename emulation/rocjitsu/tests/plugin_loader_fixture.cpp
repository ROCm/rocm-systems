// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/plugin_exports.h"

#include <cstdio>
#include <cstdlib>

#ifndef PLUGIN_FIXTURE_NAME
#error "PLUGIN_FIXTURE_NAME must be defined"
#endif

namespace {

void trace(const char *event) {
  const char *path = std::getenv("ROCJITSU_PLUGIN_TEST_TRACE");
  if (!path)
    return;
  if (FILE *file = std::fopen(path, "a")) {
    std::fprintf(file, "%s:%s\n", PLUGIN_FIXTURE_NAME, event);
    std::fclose(file);
  }
}

class BoundaryPlugin final : public rocjitsu::ExecutionPlugin {
public:
  BoundaryPlugin() : ExecutionPlugin("boundary") { trace("create"); }
  ~BoundaryPlugin() override { trace("destroy"); }

  void onInit() override {
    trace("init");
    sink().write("boundary:init\n");
  }
};

__attribute__((destructor)) void on_unload() { trace("unload"); }

} // namespace

extern "C" ROCJITSU_PLUGIN_EXPORT const rocjitsu::PluginMetadata *rocjitsu_plugin_metadata() {
  trace("metadata");
  static const rocjitsu::PluginMetadata metadata{PLUGIN_FIXTURE_NAME, "{}"};
  return &metadata;
}

extern "C" ROCJITSU_PLUGIN_EXPORT rocjitsu::PluginHandle rocjitsu_plugin_create(const char *) {
  return new BoundaryPlugin();
}

#ifndef PLUGIN_FIXTURE_MISSING_DESTROY
extern "C" ROCJITSU_PLUGIN_EXPORT void rocjitsu_plugin_destroy(rocjitsu::PluginHandle handle) {
  delete static_cast<rocjitsu::ExecutionPlugin *>(handle);
}
#endif
