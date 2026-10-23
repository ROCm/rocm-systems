/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Helpers for the handler-level tests (hrr-handler-tests), which link the
// playback handlers in and call them with crafted records.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <csignal>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace hrr_test {

// Run `fn` with stderr redirected to a temporary file and return what it wrote.
template <typename Fn>
std::string capture_stderr(Fn&& fn) {
  char path[] = "/tmp/hrr_handler_stderrXXXXXX";
  int tmp = ::mkstemp(path);
  REQUIRE(tmp >= 0);
  fflush(stderr);
  int saved = ::dup(2);
  ::dup2(tmp, 2);
  fn();
  fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  std::string out;
  ::lseek(tmp, 0, SEEK_SET);
  char buf[512];
  ssize_t n;
  while ((n = ::read(tmp, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
  ::close(tmp);
  ::unlink(path);
  return out;
}

// What a call made in a child process left behind.
struct Isolated {
  bool exited = false;  // false: the child died of `signal`
  int signal = 0;
  std::string out;  // the child's stderr, then what `fn` returned
};

// Run `fn` in a forked child, with the child's stderr sent back ahead of the
// string `fn` returns, so a crash still leaves what was said before it. For a call whose
// failure mode is a crash: a handler that wrongly reaches the HIP runtime with a
// bad argument would otherwise take the whole test binary down with SIGSEGV
// instead of failing the one test. The child is torn down with _exit.
template <typename Fn>
Isolated run_isolated(Fn&& fn) {
  int fds[2];
  REQUIRE(::pipe(fds) == 0);
  fflush(nullptr);
  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    ::close(fds[0]);
    ::dup2(fds[1], 2);
    // Catch2 turns a fatal signal into a report; here the parent wants to see it.
    for (int sig : {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL}) ::signal(sig, SIG_DFL);
    const std::string out = fn();
    fflush(stderr);
    size_t off = 0;
    while (off < out.size()) {
      ssize_t n = ::write(fds[1], out.data() + off, out.size() - off);
      if (n <= 0) break;
      off += static_cast<size_t>(n);
    }
    ::_exit(0);
  }
  ::close(fds[1]);
  Isolated r;
  char buf[512];
  ssize_t n;
  while ((n = ::read(fds[0], buf, sizeof(buf))) > 0) r.out.append(buf, static_cast<size_t>(n));
  ::close(fds[0]);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {}
  r.exited = WIFEXITED(status);
  r.signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
  return r;
}

// A record held in a buffer of exactly sizeof(T), so a read past it is a heap
// overflow AddressSanitizer can see.
template <typename T>
struct ExactRecord {
  std::vector<uint8_t> buf;
  ExactRecord() : buf(sizeof(T), 0) {}
  T* get() { return reinterpret_cast<T*>(buf.data()); }
  const uint8_t* bytes() const { return buf.data(); }
};

}  // namespace hrr_test
