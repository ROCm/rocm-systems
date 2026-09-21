// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file processor_revision.h
/// @brief Silicon revisions that affect DBT translation policy.

#pragma once

namespace rocjitsu {

/// @brief Silicon revision associated with one side of a translation.
///
/// @details This is deliberately separate from the ELF EF_AMDGPU_MACH value.
/// gfx1250 A0 and B0 use the same ELF machine ID, while their explicit
/// revisions select the corresponding translation profile.
enum class ProcessorRevision {
  Unspecified,
  Gfx1250A0,
  Gfx1250B0,
};

} // namespace rocjitsu
