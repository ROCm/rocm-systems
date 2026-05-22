// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocjitsu {
namespace amdgpu {

/// Resolve the kernel symbol name from an in-memory AMDHSA code object.
///
/// Parses the ELF's PT_DYNAMIC segment to locate .dynsym/.dynstr and matches
/// the kernel descriptor offset against symbol values with ".kd" suffix. Falls
/// back to scanning PT_NOTE msgpack metadata if .dynsym lookup fails.
///
/// All pointer dereferences are bounds-checked; returns empty on any failure.
///
/// @param kernel_object_ptr  Host pointer to the kernel descriptor.
/// @param elf_base           Host pointer to the ELF header of the code object.
/// @param elf_accessible     Number of accessible bytes from elf_base.
/// @returns Kernel name (without ".kd" suffix), or empty string on failure.
std::string find_kernel_symbol(const uint8_t *kernel_object_ptr, const uint8_t *elf_base,
                               uint64_t elf_accessible);

} // namespace amdgpu
} // namespace rocjitsu
