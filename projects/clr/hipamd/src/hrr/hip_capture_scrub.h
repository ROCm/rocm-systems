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

// Zeroes, in place, the contents of the .comment and .debug* sections of a code
// object about to be stored: an ELF64 image, or every entry of an uncompressed
// clang offload bundle. Those sections carry the compiler build, absolute build
// paths and source file names; the GPU never loads them, so replay is unchanged.
// A string the GPU does load, such as the file name in a device assert(), stays.
// Each zeroed section is retyped SHT_NOBITS, so debuggers and profilers see it as
// absent. Sizes and offsets stay as they are, so a bundle's entry table stays valid.
// A compressed bundle, or anything that does not parse, is left as it is.
// Returns how many sections were zeroed.
size_t scrub_code_object(uint8_t* image, size_t size);

}  // namespace scrub
}  // namespace hrr_cap
