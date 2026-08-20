// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file translate_test_support.h
/// @brief Shared fixtures for CPU-only DBT translation tests.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/instruction.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test_support {

using TestKernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
inline constexpr size_t kKernelDescriptorSize = sizeof(TestKernelDescriptor);

enum class TestRuntimeTextRelocation : uint8_t {
  Abs64,
  Relative64,
};

struct TestRuntimeTextReference {
  TestRuntimeTextRelocation relocation = TestRuntimeTextRelocation::Abs64;
  uint32_t relocation_type = R_AMDGPU_ABS64;
  uint64_t target_text_offset = 0;
};

void write_kernel_descriptor_entry_offset(void *descriptor, int64_t entry_offset);
[[nodiscard]] int64_t read_kernel_descriptor_entry_offset(const void *descriptor);
[[nodiscard]] TestKernelDescriptor read_kernel_descriptor_for_test(const void *descriptor);
void write_kernel_descriptor_for_test(void *descriptor, const TestKernelDescriptor &kd);
uint16_t append_elf_section_for_test(std::vector<uint8_t> &image, Elf64_Shdr section,
                                     std::span<const uint8_t> contents);

/// @param export_text_function Give the `.text` function global binding and a name of its own.
/// @details A device function is LOCAL and shares the fixture's "kernel" name by default, which
/// object_defines_only_kernels() then pairs with the "kernel.kd" descriptor and counts as a kernel.
/// Set this when the object must look like one that defines a device function a host could have
/// taken the address of -- the shape the kernarg admission has to handle.
[[nodiscard]] std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text(
    const std::vector<uint32_t> &text_words,
    std::optional<size_t> text_function_words = std::nullopt, size_t text_function_offset_words = 0,
    std::optional<size_t> function_pointer_table_target_words = std::nullopt,
    bool name_function_pointer_table_with_symbol = true, bool export_text_function = false);
[[nodiscard]] std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text();
/// @brief One sized `STT_FUNC` body in the fixture's `.text`.
struct TestTextFunction {
  size_t offset_word = 0;
  size_t words = 0;
};

/// @brief Two kernels, N sized local device functions, and one pointer slot naming each.
///
/// @details The two-descriptor helper above has no `.text` symbols and no relocation table, and
/// the single-descriptor helper has both but only one scope. Three properties are needed together
/// here: two scopes, so a body can be adopted by one and reached from the other; sized `STT_FUNC`
/// symbols, because adoption now requires a declared entry; and `R_AMDGPU_RELATIVE64` slots, which
/// are what make a body address-taken to begin with.
///
/// Kernel 0 entry is word 0 and kernel 1 entry is @p kernel1_entry_word.

[[nodiscard]] std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernels_and_function_pointers(
    const std::vector<uint32_t> &text_words, size_t kernel1_entry_word,
    const std::vector<TestTextFunction> &functions);

[[nodiscard]] std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors(
    const std::vector<uint32_t> &text_words = {0xBF810000u, 0xBF810000u},
    std::optional<TestRuntimeTextReference> runtime_text_reference = std::nullopt);
[[nodiscard]] std::vector<uint8_t> make_large_amdgpu_elf_with_waitcnt_entry();

[[nodiscard]] std::unique_ptr<Instruction> decode_one(uint32_t word, rj_code_arch_t arch);
[[nodiscard]] bool has_error_containing(const TranslatedCodeObject &result, DiagnosticKind kind,
                                        std::string_view message);
/// @brief Whether a warning of @p kind naming @p guest_offset contains @p message.
/// @details Pins the offset as well as the text so a diagnostic cannot silently move to a
/// different block while the message still matches.
[[nodiscard]] bool has_warning_at(const TranslatedCodeObject &result, DiagnosticKind kind,
                                  std::string_view message, uint64_t guest_offset);
void enable_workgroup_id_x_sgpr(std::vector<uint8_t> &image);

} // namespace rocjitsu::test_support
