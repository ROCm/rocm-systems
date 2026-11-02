/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by the src/misc/*wrap.cc tests, whose wrappers call through a global table of
// function pointers.

#ifndef RCCL_TEST_HOST_MISC_WRAP_TABLE_H_
#define RCCL_TEST_HOST_MISC_WRAP_TABLE_H_

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "nccl.h"

// Zeroes a global symbol table for its lifetime and restores it after.
template <typename Table>
class ScopedZeroedTable {
 public:
  explicit ScopedZeroedTable(Table& table) : table_(table), saved_(table) { std::memset(&table_, 0, sizeof(table_)); }
  ~ScopedZeroedTable() { table_ = saved_; }
  ScopedZeroedTable(const ScopedZeroedTable&) = delete;
  ScopedZeroedTable& operator=(const ScopedZeroedTable&) = delete;

 private:
  Table& table_;
  Table saved_;
};

// One wrapper and the fake verb behind it; the fake checks the arguments it was forwarded.
struct WrapCase {
  const char* name;
  void (*install)();
  ncclResult_t (*call)();
};

template <typename Case>
std::string CaseName(const ::testing::TestParamInfo<Case>& info) {
  return info.param.name;
}

#endif  // RCCL_TEST_HOST_MISC_WRAP_TABLE_H_
