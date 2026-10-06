// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/mca/registers.hpp"

#include <string_view>

namespace addc::mca
{

/// Return whether permitted raw fields identify the Base bad-page-retirement
/// policy condition. This deliberately exposes no restricted bank semantics.
[[nodiscard]] bool is_bad_page_retirement(std::string_view project,
                                          bool error_threshold_exceeded,
                                          McaRegisters registers) noexcept;

} // namespace addc::mca
