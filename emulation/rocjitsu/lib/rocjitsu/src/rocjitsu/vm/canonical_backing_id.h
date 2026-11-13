// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file canonical_backing_id.h
/// @brief Process-wide identity for a canonical GPU memory backing.

#pragma once

#include <cstdint>

namespace rocjitsu {

/// Allocate a nonzero ID shared by KFD and PCI backings in this process.
/// An exhausted sequence terminates rather than reusing a retired identity.
[[nodiscard]] uint64_t allocate_canonical_backing_id() noexcept;

} // namespace rocjitsu
