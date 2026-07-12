// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Internal: a loaded plugin instance (dlopen handle + vtable + ctx + cached
// enumerated devices and metrics).

#ifndef GPUMETRICS_SRC_PLUGIN_HOST_H_
#define GPUMETRICS_SRC_PLUGIN_HOST_H_

#include <memory>
#include <string>
#include <vector>

#include "gpumetrics/plugin_abi.h"

namespace gpumetrics {

// One dlopen'd plugin.
class LoadedPlugin {
 public:
  ~LoadedPlugin();
  LoadedPlugin(const LoadedPlugin&) = delete;
  LoadedPlugin& operator=(const LoadedPlugin&) = delete;
  LoadedPlugin(LoadedPlugin&&) noexcept;

  // Load, dlsym the entry, verify ABI, init(). nullptr on failure (reason in
  // *err if non-null).
  static std::unique_ptr<LoadedPlugin> Open(const std::string& path, std::string* err);

  const std::string& name() const { return name_; }
  const std::string& path() const { return path_; }

  // Cached enumerate()/list_metrics() results, fetched once at open.
  const std::vector<gpum_device_desc>& devices() const { return devices_; }
  const std::vector<gpum_metric_desc>& metrics() const { return metrics_; }

  // Batch read passthrough.
  gpum_status read(const std::vector<gpum_read_req>& reqs, std::vector<gpum_sample>& out);

 private:
  LoadedPlugin() = default;
  void* dl_ = nullptr;
  const gpum_plugin_v1* vt_ = nullptr;
  gpum_plugin_ctx* ctx_ = nullptr;
  std::string name_;
  std::string path_;
  std::vector<gpum_device_desc> devices_;
  std::vector<gpum_metric_desc> metrics_;
};

// Discover candidate plugin .so paths from search dirs + explicit names.
std::vector<std::string> DiscoverPluginPaths(const std::vector<std::string>& search_dirs,
                                             const std::vector<std::string>& explicit_names);

}  // namespace gpumetrics

#endif  // GPUMETRICS_SRC_PLUGIN_HOST_H_
