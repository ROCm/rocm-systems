// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simd_ab.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>

namespace simd_ab {
namespace {

const char *dump_path() {
  static const char *p = std::getenv("RJ_SIMD_DUMP");
  return (p && p[0]) ? p : nullptr;
}

// Single process-wide sink, truncated on first open. Each A/B run writes its
// own file (distinct RJ_SIMD_DUMP path), so truncate-once is correct.
std::ofstream &sink() {
  static std::ofstream f(dump_path(), std::ios::out | std::ios::trunc);
  return f;
}

std::mutex &mu() {
  static std::mutex m;
  return m;
}

} // namespace

bool dumping() { return dump_path() != nullptr; }

void record(std::string_view sublabel, uint64_t exec, const uint32_t *dst, std::size_t n) {
  if (!dumping())
    return;
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::lock_guard<std::mutex> lk(mu());
  std::ofstream &f = sink();
  if (info)
    f << info->test_suite_name() << '.' << info->name() << '|';
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(exec));
  f << sublabel << "|exec=" << buf << '|';
  for (std::size_t i = 0; i < n; ++i) {
    std::snprintf(buf, sizeof(buf), "%08x", dst[i]);
    f << buf << (i + 1 < n ? " " : "");
  }
  f << '\n';
}

} // namespace simd_ab
