/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by the rccl-UnitTestsMicroLinux test TUs. The binary links no shared fakes:
// linux-test.cc defines the log capture, the env-backed param loader and the lx:: hooks declared here.
#pragma once

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <poll.h>
#include <sys/socket.h>

#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>

// Every WARN/INFO/ATTN line, as "WARN <message>\n". A WARN under NOWARN is logged as INFO, as debug.cc does.
extern std::string g_linuxLog;
inline bool LinuxLogged(const char* needle) { return g_linuxLog.find(needle) != std::string::npos; }

// NCCL_PARAM/RCCL_PARAM read their env var once and cache it; this forgets every cached value.
void LinuxResetParams();

// Sets (nullptr: unsets) an env var for one scope, so params re-read it.
class LinuxScopedEnv {
 public:
  LinuxScopedEnv(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old) old_ = old, hadOld_ = true;
    Set(value);
  }
  ~LinuxScopedEnv() { Set(hadOld_ ? old_.c_str() : nullptr); }

 private:
  void Set(const char* value) {
    if (value) setenv(name_.c_str(), value, 1);
    else unsetenv(name_.c_str());
    LinuxResetParams();
  }
  std::string name_, old_;
  bool hadOld_ = false;
};

namespace lx {
// std::function for a libc function's type, minus the noexcept glibc puts on it.
template <typename T>
struct Hook;
template <typename R, typename... A>
struct Hook<R (*)(A...)> {
  using type = std::function<R(A...)>;
};
template <typename R, typename... A>
struct Hook<R (*)(A...) noexcept> {
  using type = std::function<R(A...)>;
};

// The os/linux.cc hooks other TUs drive; Reset() restores every lx:: hook to the real call.
extern Hook<decltype(&::getifaddrs)>::type getifaddrs;
extern Hook<decltype(&::freeifaddrs)>::type freeifaddrs;
extern Hook<decltype(&::setsockopt)>::type setsockopt;
extern Hook<decltype(&::connect)>::type connect;
extern Hook<decltype(&::poll)>::type poll;
void Reset();
}  // namespace lx

// A scripted getifaddrs() list. Install() routes os/linux.cc to it until lx::Reset().
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
  void Install() {
    lx::getifaddrs = [this](ifaddrs** out) {
      ifaddrs* next = nullptr;
      for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
        it->ifa.ifa_next = next;
        next = &it->ifa;
      }
      *out = next;
      listed++;
      return 0;
    };
    lx::freeifaddrs = [this](ifaddrs*) { freed++; };
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
  int listed = 0, freed = 0;

 private:
  struct Node {
    ifaddrs ifa{};
    std::string name;
    sockaddr_storage addr{}, mask{};
  };
  std::deque<Node> nodes_;
};
