/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Loads the recorder profiler plugin for an MPI test; call Load() before the first communicator.

#ifndef RCCL_TEST_PROFILER_RECORDER_HELPERS_HPP_
#define RCCL_TEST_PROFILER_RECORDER_HELPERS_HPP_

#include <dlfcn.h>
#include <mpi.h>
#include <stdlib.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "gtest/gtest.h"

#include "MPIHelpers.hpp"
#include "TestChecks.hpp"
#include "plugin/profiler_kernelch_recorder.h"

class ProfilerRecorder {
 public:
  ProfilerRecorder() = default;
  ProfilerRecorder(const ProfilerRecorder&) = delete;
  ProfilerRecorder& operator=(const ProfilerRecorder&) = delete;
  ~ProfilerRecorder() {
    Unload();
  }

  // All ranks agree: a recorder not built is a skip, one that will not load is a failure.
  void Load() {
    std::string skip;
    std::string fail;
#ifdef RCCL_TEST_KCH_RECORDER_NAME
    // Installed beside the test binary in both build and install trees.
    std::error_code ec;
    std::filesystem::path exe =
        std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
      path_ = (exe.parent_path() / RCCL_TEST_KCH_RECORDER_NAME).string();
    }
#endif
    if (path_.empty()) {
      skip = "KernelCh recorder plugin was not built into this binary";
    } else if (path_.size() >= kPluginPathLimit) {
      // RCCL ignores a longer NCCL_PROFILER_PLUGIN without saying so.
      skip = "recorder plugin path exceeds RCCL's plugin path limit: " + path_;
    } else {
      fail = Open();
    }
    if (MPIHelpers::anyRankTrue(!fail.empty())) {
      failReason_ =
          fail.empty()
              ? "KernelCh recorder plugin failed to load on another rank"
              : fail;
      return;
    }
    skipReason_ = mpiCoordinatedSkipReason(!skip.empty(), skip.c_str());
    if (!skipReason_.empty()) {
      return;
    }
    pluginEnv_ = std::make_unique<MPIHelpers::MpiEnvGuard>(
        "NCCL_PROFILER_PLUGIN", path_.c_str());
    reset_();
  }

  // Call after every communicator is destroyed: RCCL holds handles into the plugin.
  void Unload() {
    pluginEnv_.reset();
    if (handle_ != nullptr) {
      dlclose(handle_);
      handle_ = nullptr;
    }
  }

  // Every record so far, in start order.
  std::vector<RcclKchRecord> Snapshot() const {
    size_t total = snapshot_(nullptr, 0);
    std::vector<RcclKchRecord> recs(total + 1024);
    size_t got = snapshot_(recs.data(), recs.size());
    recs.resize(std::min(got, recs.size()));
    return recs;
  }

  // Empty when RCCL loaded the recorder on every rank. Every rank must call it.
  std::string NotLoadedOnAnyRank() const {
    const bool notLoaded = initCount_() == 0;
    if (!MPIHelpers::anyRankTrue(notLoaded)) {
      return {};
    }
    return notLoaded ? "RCCL did not load the recorder plugin: it rejected it, "
                       "or an earlier communicator in this process settled "
                       "the profiler without it. Run the test in a process of "
                       "its own"
                     : "RCCL did not load the recorder plugin on another rank";
  }

  const std::string& skipReason() const {
    return skipReason_;
  }
  const std::string& failReason() const {
    return failReason_;
  }
  void Reset() const {
    reset_();
  }
  uint64_t ProxyOpCount() const {
    return proxyOps_();
  }
  uint64_t Anomalies() const {
    return anomalies_();
  }
  size_t ProxyThreads(long* out, size_t cap) const {
    return proxyThreads_(out, cap);
  }

 private:
  using ResetFn = void (*)();
  using SnapshotFn = size_t (*)(RcclKchRecord*, size_t);
  using CounterFn = uint64_t (*)();
  using ThreadsFn = size_t (*)(long*, size_t);

  static constexpr size_t kPluginPathLimit = 255;

  template <typename Fn>
  Fn Symbol(const char* name) {
    return reinterpret_cast<Fn>(dlsym(handle_, name));
  }

  // Empty on success, otherwise why the plugin cannot be used.
  std::string Open() {
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* err = dlerror();
      return "cannot dlopen " + path_ + ": " + (err ? err : "no loader error");
    }
    reset_ = Symbol<ResetFn>("rcclKchRecorderReset");
    snapshot_ = Symbol<SnapshotFn>("rcclKchRecorderSnapshot");
    initCount_ = Symbol<CounterFn>("rcclKchRecorderInitCount");
    proxyOps_ = Symbol<CounterFn>("rcclKchRecorderProxyOpCount");
    anomalies_ = Symbol<CounterFn>("rcclKchRecorderAnomalies");
    proxyThreads_ = Symbol<ThreadsFn>("rcclKchRecorderProxyThreads");
    if (!reset_ || !snapshot_ || !initCount_ || !proxyOps_ || !anomalies_ ||
        !proxyThreads_) {
      return "KernelCh recorder plugin " + path_ +
             " is missing its query symbols";
    }
    return {};
  }

  std::string skipReason_;
  std::string failReason_;
  ResetFn reset_ = nullptr;
  SnapshotFn snapshot_ = nullptr;
  CounterFn initCount_ = nullptr;
  CounterFn proxyOps_ = nullptr;
  CounterFn anomalies_ = nullptr;
  ThreadsFn proxyThreads_ = nullptr;
  std::string path_;
  void* handle_ = nullptr;
  std::unique_ptr<MPIHelpers::MpiEnvGuard> pluginEnv_;
};

// Fails the test when the recorder will not load and skips it when it was not built.
#define RECORDER_REQUIRE(recorder)             \
  do {                                         \
    if (!(recorder).failReason().empty()) {    \
      FAIL() << (recorder).failReason();       \
    }                                          \
    if (!(recorder).skipReason().empty()) {    \
      GTEST_SKIP() << (recorder).skipReason(); \
    }                                          \
  } while (0)

#endif  // RCCL_TEST_PROFILER_RECORDER_HELPERS_HPP_
