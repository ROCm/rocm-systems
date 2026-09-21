// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_BUFFER_FORMAT_H_
#define ROCJITSU_VM_AMDGPU_BUFFER_FORMAT_H_

#include <array>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
struct VectorMemState;

enum class BufferFormatEncoding : uint8_t { Gfx9, Rdna1, Rdna2, Gfx11 };

/// GFX9 combines DFMT | (NFMT << 4); RDNA uses its generation's FORMAT table.
/// The memory footprint is independent of the instruction's VGPR count.
uint32_t buffer_format_bytes(uint32_t format,
                             BufferFormatEncoding encoding = BufferFormatEncoding::Gfx11);
std::array<uint32_t, 4>
unpack_buffer_format(uint32_t format, uint32_t selectors, std::span<const uint8_t> bytes,
                     BufferFormatEncoding encoding = BufferFormatEncoding::Gfx11);
void pack_buffer_format(uint32_t format, uint32_t selectors, std::span<const uint32_t> components,
                        std::span<uint8_t> bytes,
                        BufferFormatEncoding encoding = BufferFormatEncoding::Gfx11);

/// Snapshot the descriptor before issuing the memory request. A negative format
/// selects the resource's FORMAT/DST_SEL fields; MTBUF supplies its own format.
bool prepare_buffer_format(Wavefront &wf, VectorMemState &state, uint32_t resource, int format,
                           uint32_t components);
void capture_buffer_format_store(Wavefront &wf, VectorMemState &state, uint32_t data_base);
void complete_buffer_format_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);
} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_BUFFER_FORMAT_H_
