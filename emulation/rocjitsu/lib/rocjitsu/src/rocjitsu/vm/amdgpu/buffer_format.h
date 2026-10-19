// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_BUFFER_FORMAT_H_
#define ROCJITSU_VM_AMDGPU_BUFFER_FORMAT_H_

#include "util/result.h"

#include <array>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
class VectorMemState;

enum class BufferFormatEncoding : uint8_t { Gfx9, Rdna1, Rdna2, Gfx11 };

/// Independent of numeric format encoding: native gfx1201 replicates X and
/// gfx1100 zero-fills missing channels. The zero-fill policy for CDNA1-4,
/// RDNA1/2 and RDNA3.5 is extrapolated; those families have not been measured.
enum class BufferStoreMissingChannel { Zero, ReplicateFirst };

enum class BufferNumberFormat { Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float };

/// Decoded once at instruction setup and retained through memory completion.
struct BufferFormat {
  std::array<uint32_t, 4> widths{};
  BufferNumberFormat number = BufferNumberFormat::Uint;
};

/// @brief Decode @p state format/encoding, retaining channel widths and texel size.
/// @returns False for an invalid resource format and failure for reserved encodings.
util::FailureOr<bool> prepare_buffer_format_state(VectorMemState &state);

/// GFX9 combines DFMT | (NFMT << 4); RDNA uses its generation's FORMAT table.
/// The memory footprint is independent of the instruction's VGPR count.
util::FailureOr<uint32_t>
buffer_format_bytes(uint32_t format, BufferFormatEncoding encoding = BufferFormatEncoding::Gfx11);
util::FailureOr<std::array<uint32_t, 4>>
unpack_buffer_format(uint32_t format, uint32_t selectors, std::span<const uint8_t> bytes,
                     BufferFormatEncoding encoding = BufferFormatEncoding::Gfx11);
/// Packs an untyped descriptor-selected store, including DST_SEL qualification.
util::Result
pack_buffer_format(uint32_t format, uint32_t selectors, std::span<const uint32_t> components,
                   std::span<uint8_t> bytes,
                   BufferFormatEncoding encoding = BufferFormatEncoding::Gfx11,
                   BufferStoreMissingChannel missing = BufferStoreMissingChannel::Zero);

/// Snapshot the descriptor before issuing the memory request. A negative format
/// selects the resource's FORMAT/DST_SEL fields; MTBUF supplies its own format.
util::FailureOr<bool> prepare_buffer_format(Wavefront &wf, VectorMemState &state, uint32_t resource,
                                            int format, uint32_t components);
void capture_buffer_format_store(Wavefront &wf, VectorMemState &state, uint32_t data_base);
void complete_buffer_format_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);
} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_BUFFER_FORMAT_H_
