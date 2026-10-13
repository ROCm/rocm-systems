/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the libc interposers. See libc_interposers.h.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // RTLD_NEXT
#endif
#include <dlfcn.h>

#include "libc_interposers.h"

#include <unistd.h>

#include <cerrno>

// Arming gethostname failure + reaching fillInfo latches getHostName's hostHash call_once, poisoning later tests.
namespace {
bool g_gethostnameFail = false;
bool g_dladdrFail = false;
size_t g_lastGethostnameLen = 0;
}  // namespace

void SetGethostnameFail(bool fail) { g_gethostnameFail = fail; }
void SetDladdrFail(bool fail) { g_dladdrFail = fail; }
size_t LastGethostnameLen() { return g_lastGethostnameLen; }

extern "C" int gethostname(char* name, size_t len) {
  using Fn = int (*)(char*, size_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "gethostname"));
  g_lastGethostnameLen = len;
  if (g_gethostnameFail) {
    errno = ENAMETOOLONG;
    return -1;
  }
  return real ? real(name, len) : -1;
}

extern "C" int dladdr(const void* addr, Dl_info* info) {
  using Fn = int (*)(const void*, Dl_info*);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "dladdr"));
  if (g_dladdrFail) return 0;  // dladdr reports failure as 0, not -1
  return real ? real(addr, info) : 0;
}

void ResetLibcInterposers() {
  g_gethostnameFail = false;
  g_dladdrFail = false;
  g_lastGethostnameLen = 0;
}
