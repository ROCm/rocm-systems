// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_input_layout.h
/// @brief Structural preconditions shared by ConSan entry and final validation.

#pragma once

#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;

/// Validate the ELF and AMDGPU metadata geometry on which ConSan relies.
///
/// Descriptor-entry redirection is permitted only while validating a
/// replacement whose committed patch proof explicitly records such a
/// redirection. Ordinary input validation must retain the default.
[[nodiscard]] std::vector<std::string>
validate_consan_input_layout(const AmdGpuCodeObject &code_object,
                             bool allow_descriptor_entry_redirect = false);

} // namespace rocjitsu
