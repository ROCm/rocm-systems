// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_internal.h
/// @brief Private, directly testable invariants shared by MOI lowering paths.

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

struct ConSanOptions;

namespace consan_detail {

/// Validate the site-local VGPR half of scalar-persistent MOI state before
/// emission. This remains release-active because ConSan rewrites untrusted
/// code objects and must fail cleanly if placement and emission ever diverge.
[[nodiscard]] bool validate_scalar_state_temporaries(const ConSanOptions &options,
                                                     std::string_view consumer,
                                                     std::vector<std::string> &errors);

} // namespace consan_detail
} // namespace rocjitsu
