// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// Re-run production's independent final proof against a deliberately
/// corrupted lowerer artifact. The expected dispatch identity must match the
/// value used for lowering when a fixture supplies a nonzero literal.
[[nodiscard]] std::vector<std::string>
validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                             const ConSanTransformArtifacts &modified_result,
                             uint64_t expected_moi_report_dispatch_id = 0);

} // namespace rocjitsu
