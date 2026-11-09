/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace hrr_cap {
namespace scrub {

// Zeroes, in place, the .comment and .debug* sections of a code object about to be stored:
// an ELF64 image, or every entry of an uncompressed offload bundle. Returns the number of sections zeroed.
size_t scrub_code_object(uint8_t* image, size_t size);

}  // namespace scrub
}  // namespace hrr_cap
