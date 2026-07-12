// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "plugin_host.h"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace gpumetrics {

namespace {

// Some backends abort the process on init against unsupported configs (e.g.
// rocprofiler-sdk LOG(FATAL)s when a partitioned GPU exposes more agents than
// its node-id encoding allows). An abort can't be caught in-process, so probe
// the plugin in a forked child (dlopen + init + enumerate) and skip it if the
// child dies on a signal or non-zero exit. Returns true on clean init.
bool ProbePluginInChild(const std::string& path) {
  if (std::getenv("GPUMETRICS_NO_PROBE")) return true;
  pid_t pid = fork();
  if (pid < 0) return true;  // cannot fork; fall through to direct load
  if (pid == 0) {
    // Child: silence output, then attempt the risky init.
    if (!::freopen("/dev/null", "w", stderr)) { /* ignore */ }
    if (!::freopen("/dev/null", "w", stdout)) { /* ignore */ }
    void* dl = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dl) _exit(2);
    auto entry = reinterpret_cast<gpum_plugin_entry_v1_fn>(dlsym(dl, GPUM_PLUGIN_ENTRY_SYMBOL));
    if (!entry) _exit(3);
    const gpum_plugin_v1* vt = entry();
    if (!vt || vt->abi_version != GPUM_PLUGIN_ABI_V1 || !vt->init) _exit(4);
    gpum_plugin_ctx* ctx = nullptr;
    if (vt->init(&ctx) != GPUM_OK) _exit(5);
    const gpum_device_desc* devs = nullptr;
    uint32_t ndev = 0;
    if (vt->enumerate && vt->enumerate(ctx, &devs, &ndev) != GPUM_OK) _exit(6);
    _exit(0);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace

LoadedPlugin::LoadedPlugin(LoadedPlugin&& o) noexcept
    : dl_(o.dl_),
      vt_(o.vt_),
      ctx_(o.ctx_),
      name_(std::move(o.name_)),
      path_(std::move(o.path_)),
      devices_(std::move(o.devices_)),
      metrics_(std::move(o.metrics_)) {
  o.dl_ = nullptr;
  o.vt_ = nullptr;
  o.ctx_ = nullptr;
}

LoadedPlugin::~LoadedPlugin() {
  if (vt_ && ctx_ && vt_->shutdown) {
    vt_->shutdown(ctx_);
  }
  // rocprofiler-backed plugins register process-global HSA/rocprofiler state
  // that can't be cleanly reversed, so we intentionally do NOT dlclose():
  // leaking the handle at teardown is safer than tearing down live runtime
  // state.
}

std::unique_ptr<LoadedPlugin> LoadedPlugin::Open(const std::string& path, std::string* err) {
  auto set_err = [&](const std::string& m) {
    if (err) *err = m;
  };

  // Probe in a child first so a backend that aborts on init can't take down the
  // whole tool (see ProbePluginInChild).
  if (!ProbePluginInChild(path)) {
    set_err("plugin aborted during probe; skipping");
    return nullptr;
  }

  // RTLD_GLOBAL is mandatory: rocprofiler-sdk resolves the rocprofiler plugin's
  // rocprofiler_configure from the global symbol namespace at hsa_init().
  void* dl = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!dl) {
    set_err(std::string("dlopen failed: ") + (dlerror() ? dlerror() : "unknown"));
    return nullptr;
  }

  auto entry = reinterpret_cast<gpum_plugin_entry_v1_fn>(dlsym(dl, GPUM_PLUGIN_ENTRY_SYMBOL));
  if (!entry) {
    set_err(std::string("missing symbol ") + GPUM_PLUGIN_ENTRY_SYMBOL);
    dlclose(dl);
    return nullptr;
  }

  const gpum_plugin_v1* vt = entry();
  if (!vt || vt->abi_version != GPUM_PLUGIN_ABI_V1) {
    set_err("plugin ABI mismatch");
    dlclose(dl);
    return nullptr;
  }
  if (!vt->init || !vt->shutdown || !vt->enumerate || !vt->list_metrics || !vt->read) {
    set_err("plugin vtable incomplete");
    dlclose(dl);
    return nullptr;
  }

  gpum_plugin_ctx* ctx = nullptr;
  gpum_status st = vt->init(&ctx);
  if (st != GPUM_OK) {
    set_err(std::string("plugin init failed: ") + gpum_status_string(st));
    dlclose(dl);
    return nullptr;
  }

  auto p = std::unique_ptr<LoadedPlugin>(new LoadedPlugin());
  p->dl_ = dl;
  p->vt_ = vt;
  p->ctx_ = ctx;
  p->path_ = path;
  p->name_ = vt->name ? vt->name : "unknown";

  // Cache enumerate()/list_metrics() once.
  const gpum_device_desc* devs = nullptr;
  uint32_t ndev = 0;
  if (vt->enumerate(ctx, &devs, &ndev) == GPUM_OK && devs) {
    p->devices_.assign(devs, devs + ndev);
  }
  const gpum_metric_desc* mets = nullptr;
  uint32_t nmet = 0;
  if (vt->list_metrics(ctx, &mets, &nmet) == GPUM_OK && mets) {
    p->metrics_.assign(mets, mets + nmet);
  }
  return p;
}

gpum_status LoadedPlugin::read(const std::vector<gpum_read_req>& reqs,
                               std::vector<gpum_sample>& out) {
  out.assign(reqs.size(), gpum_sample{});
  if (reqs.empty()) return GPUM_OK;
  return vt_->read(ctx_, reqs.data(), static_cast<uint32_t>(reqs.size()), out.data());
}

std::vector<std::string> DiscoverPluginPaths(const std::vector<std::string>& search_dirs,
                                             const std::vector<std::string>& explicit_names) {
  std::vector<std::string> out;
  auto consider = [&](const fs::path& p) {
    std::error_code ec;
    if (!fs::exists(p, ec) || fs::is_directory(p, ec)) return;
    const std::string s = p.string();
    for (const auto& e : out)
      if (e == s) return;
    out.push_back(s);
  };

  // Canonical plugin filename pattern: libgpumetrics_<name>.so
  const char* kPrefix = "libgpumetrics_";
  const char* kSuffix = ".so";

  for (const auto& dir : search_dirs) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) continue;
    if (!explicit_names.empty()) {
      for (const auto& n : explicit_names) {
        // Accept a bare backend name ("amdsmi"), a filename, or a full path.
        consider(fs::path(dir) / n);
        consider(fs::path(dir) / (std::string(kPrefix) + n + kSuffix));
      }
      continue;
    }
    for (const auto& de : fs::directory_iterator(dir, ec)) {
      const std::string fn = de.path().filename().string();
      if (fn.rfind(kPrefix, 0) == 0 && fn.size() > 3 &&
          fn.compare(fn.size() - 3, 3, kSuffix) == 0) {
        consider(de.path());
      }
    }
  }
  // Explicit paths given directly.
  for (const auto& n : explicit_names) {
    if (n.find('/') != std::string::npos) consider(n);
  }
  return out;
}

}  // namespace gpumetrics
