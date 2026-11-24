// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/execution_plugin_group.h"

namespace rocjitsu {

std::shared_ptr<ExecutionPluginGroup> ExecutionPluginGroup::empty_group() {
#if defined(RJ_DISABLE_INTERPOSER)
  // Explicit hosts finish VM teardown before unloading the library.
  static auto instance = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  return instance;
#else
  // The interposer may defer joining its engine until HIP/ROCr close their
  // KFD references, after static destructors run. Retain the default group
  // through that process-teardown interval.
  static std::shared_ptr<ExecutionPluginGroup> *instance =
      new std::shared_ptr<ExecutionPluginGroup>(
          std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  return *instance;
#endif
}

} // namespace rocjitsu
