// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_test_utils.h
/// @brief Shared helpers for DBT hardware tests.

#ifndef ROCJITSU_TESTS_DBT_HSA_TEST_UTILS_H_
#define ROCJITSU_TESTS_DBT_HSA_TEST_UTILS_H_

#include "rocjitsu/code/rj_code.h"
#include "tools/hsa_run_kernel.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu::dbt_test {

struct HostTarget {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t mach = 0;
  std::string isa_name;
};

std::string kernel_path(const char *name);

HostTarget detect_hsa_host_target();

template <typename T> std::vector<uint8_t> scalar_arg_bytes(T value) {
  std::vector<uint8_t> bytes(sizeof(T));
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

template <typename T> std::vector<uint8_t> bytes_of(const std::vector<T> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty())
    std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

tools::KernelArgPatch ptr_arg(size_t offset, std::string buffer_name);

tools::KernelArgPatch u32_arg(size_t offset, uint32_t value);

const tools::HsaBufferOutput *find_output(const tools::HsaRunOutput &output,
                                          const std::string &name);

} // namespace rocjitsu::dbt_test

#endif // ROCJITSU_TESTS_DBT_HSA_TEST_UTILS_H_
