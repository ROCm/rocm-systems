// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_test_utils.h
/// @brief Shared ELF fixtures for DBT unit and integration tests.

#ifndef ROCJITSU_TESTS_DBT_TEST_UTILS_H_
#define ROCJITSU_TESTS_DBT_TEST_UTILS_H_

#include "rocjitsu/code/code_object.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace rocjitsu::dbt_test {

std::vector<uint8_t> make_minimal_amdgpu_elf_with_text_and_rodata();

std::vector<uint8_t> make_minimal_amdgpu_elf_with_load_segments();

std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text();

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors();

std::vector<uint8_t> make_minimal_amdgpu_elf_with_relocation_after_text();

std::vector<uint8_t> make_large_amdgpu_elf_with_waitcnt_entry();

const Section *find_section(const CodeObject &co, std::string_view name);

} // namespace rocjitsu::dbt_test

#endif // ROCJITSU_TESTS_DBT_TEST_UTILS_H_
