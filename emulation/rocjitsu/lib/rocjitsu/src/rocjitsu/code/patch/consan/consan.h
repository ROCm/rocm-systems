// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan.h
/// @brief Shared ConSan contracts and prototype mechanism data.
///
/// Production transformation enters through `consan_pipeline.h`. No library
/// header exposes the mutable compatibility result entry used by explicitly
/// mechanism-level tests while the lowerer is decomposed.

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "rocjitsu/code/kernel_symbol.h"
#include "rocjitsu/code/patch/consan/consan_access_classifier.h"
#include "rocjitsu/code/patch/consan/consan_access_shape.h"
#include "rocjitsu/code/patch/consan/consan_atomic_classifier.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_dispatch_prologue_effect.h"
#include "rocjitsu/code/patch/consan/consan_entry_scalar_backup.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis_encoding.h"
#include "rocjitsu/code/patch/consan/consan_scalar_vcc_spill.h"
#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/code/patch/consan/consan_indirect_jump_sgprs.h.inc"

#include "rocjitsu/code/patch/consan/consan_options.h.inc"

#include "rocjitsu/code/patch/consan/consan_vgpr_state_effect.h.inc"

#include "rocjitsu/code/patch/consan/consan_private_state_layout.h.inc"

#include "rocjitsu/code/patch/consan/consan_request_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_code_object_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_site_identity.h.inc"

#include "rocjitsu/code/patch/consan/consan_fault_sync_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_program_inventory.h.inc"

#include "rocjitsu/code/patch/consan/consan_observation_plan.h.inc"

#include "rocjitsu/code/patch/consan/consan_resource_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_result.h.inc"

} // namespace rocjitsu::consan
