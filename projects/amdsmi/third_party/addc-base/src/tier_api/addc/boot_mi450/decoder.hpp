// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace addc::boot_mi450
{

/// Map a raw MI450 EAM boot/runtime message to the AFID knowledge available
/// from the public register definition.
[[nodiscard]] int decode_afid(uint64_t value) noexcept;

} // namespace addc::boot_mi450
