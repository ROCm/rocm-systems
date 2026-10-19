/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by the rccl-UnitTestsMicroLinux test TUs: the base fixture, a log recorder and a scripted
// getifaddrs() list, all over the shared libc, nccl, env and HIP fakes.
#pragma once

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>

#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/nccl_fakes.h"

// What production logged, from any thread, as "WARN|INFO|ATTN <message>\n" lines.
class LogRecord {
 public:
  bool Has(const std::string& needle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return text_.find(needle) != std::string::npos;
  }
  std::string Text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return text_;
  }
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    text_.clear();
  }

 private:
  void Append(ncclDebugLogLevel level, const char* msg) {
    const char* tag = level == NCCL_LOG_WARN ? "WARN" : level == NCCL_LOG_ATTN ? "ATTN" : "INFO";
    std::lock_guard<std::mutex> lock(mutex_);
    text_ += std::string(tag) + " " + msg + "\n";
  }
  mutable std::mutex mutex_;
  std::string text_;
  RcclUnitTesting::ScopedDebugLogging everyLevel_{-1, ~uint64_t(0)};
  ScopedHook<void(ncclDebugLogLevel, unsigned long, const char*, int, const char*)> hook_{
      g_ncclDebugLog,
      [this](ncclDebugLogLevel level, unsigned long, const char*, int, const char* msg) { Append(level, msg); }};
};

// Params reach the units through param_redirect.h's g_loadParam; one not Set() keeps its default.
class ParamScript {
 public:
  void Set(const char* name, int64_t value) { values_[name] = value; }

 private:
  std::map<std::string, int64_t> values_;
  ScopedHook<int64_t(const char*, int64_t)> hook_{g_loadParam, [this](const char* name, int64_t deftVal) {
                                                     auto it = values_.find(name);
                                                     return it == values_.end() ? deftVal : it->second;
                                                   }};
};

// Real libc behind every seam, an empty scripted environment, and socket retries that do not sleep.
class LinuxTestBase : public ::testing::Test {
 protected:
  LinuxTestBase() {
    UseRealLibc();
    ResetEnvFakes();
    params_.Set("SOCKET_RETRY_SLEEP_MSEC", 0);
  }
  ~LinuxTestBase() override {
    ResetEnvFakes();
    UseRealLibc();
  }
  LogRecord log_;
  ParamScript params_;
};

// A scripted getifaddrs() list, installed for the object's lifetime.
class LinuxFakeIfaddrs {
 public:
  void Add(const char* name, int family, const char* addr = nullptr, const char* mask = nullptr,
           unsigned flags = IFF_UP | IFF_RUNNING, uint32_t scope = 0) {
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.name = name;
    n.ifa.ifa_name = const_cast<char*>(n.name.c_str());
    n.ifa.ifa_flags = flags;
    if (family == AF_UNSPEC) return;
    Fill(&n.addr, family, addr, scope);
    n.ifa.ifa_addr = (sockaddr*)&n.addr;
    if (mask) {
      Fill(&n.mask, family, mask, 0);
      n.ifa.ifa_netmask = (sockaddr*)&n.mask;
    }
  }
  static void Fill(sockaddr_storage* ss, int family, const char* text, uint32_t scope) {
    std::memset(ss, 0, sizeof(*ss));
    ss->ss_family = family;
    if (family == AF_INET) inet_pton(AF_INET, text, &((sockaddr_in*)ss)->sin_addr);
    if (family == AF_INET6) {
      inet_pton(AF_INET6, text, &((sockaddr_in6*)ss)->sin6_addr);
      ((sockaddr_in6*)ss)->sin6_scope_id = scope;
    }
  }
  int Listed() const { return list_.calls; }
  int Freed() const { return free_.calls; }

 private:
  struct Node {
    ifaddrs ifa{};
    std::string name;
    sockaddr_storage addr{}, mask{};
  };
  std::deque<Node> nodes_;
  ScopedHook<int(ifaddrs**)> list_{g_getifaddrs, [this](ifaddrs** out) {
                                     ifaddrs* next = nullptr;
                                     for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
                                       it->ifa.ifa_next = next;
                                       next = &it->ifa;
                                     }
                                     *out = next;
                                     return 0;
                                   }};
  ScopedHook<void(ifaddrs*)> free_{g_freeifaddrs, [](ifaddrs*) {}};
};
