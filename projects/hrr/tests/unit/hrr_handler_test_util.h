/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Helpers for the handler-level tests (hrr-handler-tests), which link the
// playback handlers in and call them with crafted records.

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"

#include <cstdint>
#include <cstdio>
#include <string>
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

// Call the handler the dispatch table holds for `id` with a crafted record.
inline hipError_t dispatch(PlaybackContext& ctx, hrr_api_id_t id, const uint8_t* record) {
  REQUIRE(hrr_playback_dispatch[id] != nullptr);
  return hrr_playback_dispatch[id](ctx, record);
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
