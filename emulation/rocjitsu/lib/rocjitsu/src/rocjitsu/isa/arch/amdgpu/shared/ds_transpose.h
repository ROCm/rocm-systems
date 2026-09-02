// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_DS_TRANSPOSE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_DS_TRANSPOSE_H_

/// @file ds_transpose.h
/// @brief Cross-lane transpose logic for transpose-load instructions.
///
/// @details Transpose-load instructions read raw data per-lane, then apply a
/// cross-lane shuffle to produce the transposed matrix layout expected by
/// matrix instructions.
///
/// B64_TR_B8 is the CDNA4 MFMA byte transpose within 16-lane groups.
/// WMMA_TR_B8 is used by RDNA4 and CDNA5 global loads. CDNA5_DS_TR_B8
/// handles the four-row, two-column-half request grouping of gfx1250 DS loads.
/// TR16_B128 (gfx1250 ds/global_load_tr16_b128, RDNA4 global_load_tr_b128)
/// transposes 16-bit elements within groups of 8 consecutive lanes.
/// B64_TR_B16 (CDNA4 ds_read_b64_tr_b16) is a 4x16-lane halfword transpose
/// with stride-4 source lanes.

#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

enum class TransposeKind : uint8_t {
  NONE = 0,
  TR_B4 = 1,
  TR_B6 = 2,
  B64_TR_B8 = 3,
  TR16_B128 = 4,
  B64_TR_B16 = 5,
  WMMA_TR_B8 = 6,
  CDNA5_DS_TR_B8 = 7,
};

/// @brief CDNA4 MFMA B64 byte-level transpose (B64_TR_B8 only).
///
/// Within each 16-lane group, destination lane `l` byte `n` comes from source
/// lane `((l & ~0xf) | ((l >> 3) & 1)) + 2 * n`, byte `l & 7`.
/// B64_TR_B8 always carries exactly two dwords (eight bytes) per lane.
inline void transpose_b64_tr_b8(std::vector<uint8_t> &response_data, uint32_t num_elems,
                                uint32_t wf_size) {
  assert(num_elems == 2 && "B64_TR_B8 requires an eight-byte lane payload");
  const uint32_t bytes_per_lane_total = num_elems * 4;

  std::vector<uint8_t> output(response_data.size(), 0);

  for (uint32_t dest_lane = 0; dest_lane < wf_size; ++dest_lane) {
    const uint32_t source_byte = dest_lane & 7u;
    const uint32_t source_base = (dest_lane & ~0xfu) | ((dest_lane >> 3) & 1u);
    for (uint32_t dest_byte = 0; dest_byte < bytes_per_lane_total; ++dest_byte) {
      const uint32_t source_lane = source_base + 2u * dest_byte;
      const uint32_t source_offset = source_lane * bytes_per_lane_total + source_byte;
      const uint32_t dest_offset = dest_lane * bytes_per_lane_total + dest_byte;
      if (source_offset < response_data.size() && dest_offset < output.size())
        output[dest_offset] = response_data[source_offset];
    }
  }

  response_data = std::move(output);
}

/// @brief RDNA4 and CDNA5 global-load 16x16 8-bit WMMA transpose layout.
///
/// Each of source lanes 0..31 reads eight adjacent matrix rows for one K
/// coordinate. Wave32 writes two VGPRs per lane. RDNA4 Wave64 consumes the
/// same 32 source addresses but writes one VGPR across 64 destination lanes.
inline void transpose_wmma_tr_b8(std::vector<uint8_t> &response_data, uint32_t &num_elems,
                                 uint32_t wf_size) {
  constexpr uint32_t source_lanes = 32;
  constexpr uint32_t source_bytes_per_lane = 8;

  const uint32_t source_stride = num_elems * 4;
  const uint32_t dest_stride = (wf_size == 64) ? 4u : 8u;
  std::vector<uint8_t> output(wf_size * dest_stride, 0);

  for (uint32_t source_lane = 0; source_lane < source_lanes; ++source_lane) {
    const uint32_t k_lo = source_lane & 7u;
    const uint32_t k_hi = source_lane >> 4;
    const uint32_t matrix_row_base = 8u * ((source_lane >> 3) & 1u);
    for (uint32_t byte = 0; byte < source_bytes_per_lane; ++byte) {
      const uint32_t matrix_row = matrix_row_base + byte;
      uint32_t dest_lane;
      uint32_t dest_byte;
      if (wf_size == 64) {
        dest_lane = matrix_row + 16u * k_hi + 32u * ((k_lo >> 2) & 1u);
        dest_byte = k_lo & 3u;
      } else {
        dest_lane = matrix_row + 16u * k_hi;
        dest_byte = k_lo;
      }

      const uint32_t source_offset = source_lane * source_stride + byte;
      const uint32_t dest_offset = dest_lane * dest_stride + dest_byte;
      if (source_offset < response_data.size() && dest_offset < output.size())
        output[dest_offset] = response_data[source_offset];
    }
  }

  response_data = std::move(output);
  num_elems = dest_stride / 4;
}

/// @brief gfx1250 DS B64 byte-level transpose.
/// @details Destination lane `l` byte `n` comes from source lane
/// `((l & ~0xf) | (((l >> 3) & 1) * 4)) + (n & 3) + 8 * (n >> 2)`, byte
/// `l & 7`. The instruction has an eight-byte lane payload and requires wave32.
inline void transpose_cdna5_ds_tr_b8(std::vector<uint8_t> &response_data, uint32_t num_elems,
                                     uint32_t wf_size) {
  assert(num_elems == 2 && "CDNA5 DS TR_B8 requires an eight-byte lane payload");
  assert(wf_size == 32 && "CDNA5 DS transpose loads require wave32");
  const uint32_t bytes_per_lane = num_elems * 4;
  std::vector<uint8_t> output(response_data.size(), 0);

  // Each 16-lane half loads an 8x16 matrix. Four adjacent lanes address
  // consecutive rows of one eight-column half; lanes 8..15 address rows 4..7.
  // Destination lane half*16+column receives rows 0..7 of that column.
  for (uint32_t dest_lane = 0; dest_lane < wf_size; ++dest_lane) {
    const uint32_t source_byte = dest_lane & 7u;
    const uint32_t source_base = (dest_lane & ~0xfu) | (((dest_lane >> 3) & 1u) * 4u);
    for (uint32_t dest_byte = 0; dest_byte < bytes_per_lane; ++dest_byte) {
      const uint32_t source_lane = source_base + (dest_byte & 3u) + 8u * (dest_byte >> 2);
      output[dest_lane * bytes_per_lane + dest_byte] =
          response_data[source_lane * bytes_per_lane + source_byte];
    }
  }

  response_data = std::move(output);
}

/// @brief Return the lanes that issue a transpose-load memory request.
/// @details Wave64 B8 transpose loads issue through lanes 0-31; other forms use all active lanes.
/// @param d Vector-memory state of the access.
/// @param wf_size Width of the issuing wavefront. Taken separately because
///        VectorMemState::wf_size is set by the pipeline, and a caller that
///        runs before the pipeline has it would take the wave64 branch for a
///        wave32 access.
inline uint64_t transpose_request_lane_mask(const VectorMemState &d, uint32_t wf_size) {
  const auto kind = static_cast<TransposeKind>(d.transpose);
  if (wf_size == 64 && (kind == TransposeKind::WMMA_TR_B8 ||
                        (d.tag() == GLOBAL_MEM && kind == TransposeKind::TR16_B128)))
    return d.lane_mask & 0xFFFF'FFFFULL;
  return d.lane_mask;
}

/// @brief Return the lanes that issue a transpose-load memory request, using
///        the width recorded in @p d.
inline uint64_t transpose_request_lane_mask(const VectorMemState &d) {
  return transpose_request_lane_mask(d, d.wf_size);
}

/// @brief TR16_B128: 16-bit element transpose for B128 transpose loads
/// (gfx1250 ds_load_tr16_b128 / global_load_tr16_b128, RDNA4 global_load_tr_b128).
///
/// Groups of 8 source lanes, each reading 8 halfwords. Destination lane
/// `group_start + halfword_index` receives that halfword from each source lane.
inline void transpose_tr16_b128(std::vector<uint8_t> &response_data, uint32_t num_elems,
                                uint32_t wf_size, bool compact_wave64 = false) {
  constexpr uint32_t lanes_per_half = 32;

  const uint32_t source_bytes_per_lane = num_elems * 4;
  compact_wave64 = compact_wave64 && wf_size == 64;
  const uint32_t destination_bytes_per_lane =
      compact_wave64 ? source_bytes_per_lane / 2 : source_bytes_per_lane;
  const uint32_t halfwords_per_source_lane = source_bytes_per_lane / 2;
  const uint32_t group_size = halfwords_per_source_lane;
  const uint32_t num_halves = compact_wave64 ? 1u : (wf_size > lanes_per_half ? 2u : 1u);

  std::vector<uint8_t> output(static_cast<size_t>(wf_size) * destination_bytes_per_lane, 0);

  for (uint32_t half_index = 0; half_index < num_halves; ++half_index) {
    const uint32_t lane_base = half_index * lanes_per_half;

    for (uint32_t group_start = 0; group_start < lanes_per_half; group_start += group_size) {
      for (uint32_t halfword_index = 0; halfword_index < halfwords_per_source_lane;
           ++halfword_index) {
        const uint32_t dest_lane = lane_base + group_start + halfword_index;
        for (uint32_t lane_in_group = 0; lane_in_group < group_size; ++lane_in_group) {
          const uint32_t source_lane = lane_base + group_start + lane_in_group;
          const uint32_t source_offset = source_lane * source_bytes_per_lane + halfword_index * 2;
          const uint32_t dest_half = compact_wave64 ? lane_in_group / (group_size / 2) : 0;
          const uint32_t dest_halfword =
              compact_wave64 ? lane_in_group % (group_size / 2) : lane_in_group;
          const uint32_t physical_dest_lane = dest_lane + dest_half * lanes_per_half;
          const uint32_t dest_offset =
              physical_dest_lane * destination_bytes_per_lane + dest_halfword * 2;

          if (source_offset + 2 <= response_data.size() && dest_offset + 2 <= output.size())
            std::memcpy(&output[dest_offset], &response_data[source_offset], 2);
        }
      }
    }
  }

  response_data = std::move(output);
}

/// @brief B64_TR_B16: 4x16-lane halfword transpose (CDNA4 ds_read_b64_tr_b16).
///
/// Within each 16-lane group, destination lane `l` halfword `n` comes from
/// source lane `((l & ~0xf) | ((l >> 2) & 3)) + 4 * n`, halfword `l & 3`.
inline void transpose_b64_tr_b16(std::vector<uint8_t> &response_data, uint32_t num_elems,
                                 uint32_t wf_size) {
  constexpr uint32_t source_lane_stride = 4;

  const uint32_t bytes_per_lane_total = num_elems * 4;
  const uint32_t halfwords_per_lane = bytes_per_lane_total / 2;

  std::vector<uint8_t> output(response_data.size(), 0);

  for (uint32_t dest_lane = 0; dest_lane < wf_size; ++dest_lane) {
    const uint32_t source_halfword = dest_lane & 3;
    const uint32_t source_base = (dest_lane & ~0xfu) | ((dest_lane >> 2) & 3);
    for (uint32_t halfword_index = 0; halfword_index < halfwords_per_lane; ++halfword_index) {
      const uint32_t source_lane = source_base + source_lane_stride * halfword_index;
      const uint32_t source_offset = source_lane * bytes_per_lane_total + source_halfword * 2;
      const uint32_t dest_offset = dest_lane * bytes_per_lane_total + halfword_index * 2;

      if (source_offset + 2 <= response_data.size() && dest_offset + 2 <= output.size())
        std::memcpy(&output[dest_offset], &response_data[source_offset], 2);
    }
  }

  response_data = std::move(output);
}

/// @brief TR_B4: 4-bit element transpose for B128 family.
///
/// Groups of 8 source lanes, each reading 8 bytes (64 bits). Extracts
/// 4-bit nibbles and transposes them across lanes.
inline void transpose_b4(std::vector<uint8_t> &response_data, uint32_t num_elems,
                         uint32_t wf_size) {
  constexpr uint32_t lanes_per_half = 32;
  constexpr uint32_t source_group_size = 8;
  constexpr uint32_t bytes_per_source_lane = 8;
  constexpr uint32_t nibbles_per_group = 16;

  const uint32_t bytes_per_lane_total = num_elems * 4;
  const uint32_t num_halves = (wf_size > lanes_per_half) ? 2u : 1u;

  std::vector<uint8_t> output(response_data.size(), 0);

  for (uint32_t half_index = 0; half_index < num_halves; ++half_index) {
    const uint32_t lane_base = half_index * lanes_per_half;

    for (uint32_t group_start = 0; group_start < lanes_per_half; group_start += source_group_size) {
      uint8_t source_bytes[source_group_size][bytes_per_source_lane] = {};
      for (uint32_t lane_in_group = 0; lane_in_group < source_group_size; ++lane_in_group) {
        uint32_t source_offset = (lane_base + group_start + lane_in_group) * bytes_per_lane_total;
        for (uint32_t byte = 0;
             byte < bytes_per_source_lane && source_offset + byte < response_data.size(); ++byte)
          source_bytes[lane_in_group][byte] = response_data[source_offset + byte];
      }

      for (uint32_t nibble_index = 0; nibble_index < nibbles_per_group; ++nibble_index) {
        uint32_t nibble_shift = (nibble_index & 1) * 4;
        uint32_t packed_dword = 0;
        for (uint32_t lane_in_group = 0; lane_in_group < source_group_size; ++lane_in_group) {
          uint32_t nibble_value =
              (static_cast<uint32_t>(source_bytes[lane_in_group][nibble_index / 2]) >>
               nibble_shift) &
              0xf;
          packed_dword |= nibble_value << (lane_in_group * 4);
        }

        uint32_t dest_vgpr_index = (group_start / 16) & 1;
        uint32_t dest_lane = lane_base + (group_start & 8) * 2 + nibble_index;
        uint32_t dest_offset = dest_lane * bytes_per_lane_total + dest_vgpr_index * 4;

        if (dest_offset + 4 <= output.size())
          std::memcpy(&output[dest_offset], &packed_dword, 4);
      }
    }
  }

  response_data = std::move(output);
}

/// @brief TR_B6: 6-bit element transpose for B96 family.
///
/// 16x16 6-bit permutation in 2 passes over 32 lanes. Each pass reads
/// 12 bytes (96 bits) from 16 source lanes and transposes 6-bit fields.
inline void transpose_b6(std::vector<uint8_t> &response_data, uint32_t num_elems,
                         uint32_t wf_size) {
  constexpr uint32_t lanes_per_half = 32;
  constexpr uint32_t lanes_per_pass = 16;
  constexpr uint32_t dwords_per_lane = 3;
  constexpr uint32_t bits_per_element = 6;
  constexpr uint32_t bits_per_dword = 32;

  const uint32_t bytes_per_lane_total = num_elems * 4;
  const uint32_t num_halves = (wf_size > lanes_per_half) ? 2u : 1u;

  std::vector<uint8_t> output(response_data.size(), 0);

  for (uint32_t half_index = 0; half_index < num_halves; ++half_index) {
    const uint32_t lane_base = half_index * lanes_per_half;

    for (uint32_t pass = 0; pass < 2; ++pass) {
      uint32_t source_dwords[lanes_per_pass][dwords_per_lane] = {};

      for (uint32_t input_index = 0; input_index < lanes_per_pass; ++input_index) {
        uint32_t source_lane = 8 * (input_index / 4) + (input_index % 4) + pass * 4;
        uint32_t source_offset = (lane_base + source_lane) * bytes_per_lane_total;
        for (uint32_t dword = 0; dword < dwords_per_lane; ++dword) {
          if (source_offset + dword * 4 + 4 <= response_data.size())
            std::memcpy(&source_dwords[input_index][dword],
                        &response_data[source_offset + dword * 4], 4);
        }
      }

      for (uint32_t output_lane_in_pass = 0, input_bit_position = 0;
           output_lane_in_pass < lanes_per_pass;
           ++output_lane_in_pass, input_bit_position += bits_per_element) {

        uint32_t output_dwords[dwords_per_lane] = {};

        for (uint32_t input_lane_in_pass = 0, output_bit_position = 0;
             input_lane_in_pass < lanes_per_pass;
             ++input_lane_in_pass, output_bit_position += bits_per_element) {

          uint32_t dword_index = input_bit_position / bits_per_dword;
          uint32_t bit_offset = input_bit_position % bits_per_dword;
          uint32_t element = source_dwords[input_lane_in_pass][dword_index] >> bit_offset;
          if (bit_offset > (bits_per_dword - bits_per_element))
            element |= source_dwords[input_lane_in_pass][dword_index + 1]
                       << (bits_per_dword - bit_offset);
          element &= 0x3f;

          uint32_t out_dword_index = output_bit_position / bits_per_dword;
          uint32_t out_bit_offset = output_bit_position % bits_per_dword;
          output_dwords[out_dword_index] |= (element << out_bit_offset);
          if (out_bit_offset > (bits_per_dword - bits_per_element))
            output_dwords[out_dword_index + 1] |= (element >> (bits_per_dword - out_bit_offset));
        }

        uint32_t dest_lane = lane_base + output_lane_in_pass + pass * lanes_per_pass;
        for (uint32_t dword = 0; dword < dwords_per_lane; ++dword) {
          uint32_t dest_offset = dest_lane * bytes_per_lane_total + dword * 4;
          if (dest_offset + 4 <= output.size())
            std::memcpy(&output[dest_offset], &output_dwords[dword], 4);
        }
      }
    }
  }

  response_data = std::move(output);
}

inline void transpose_response(VectorMemState &d) {
  auto kind = static_cast<TransposeKind>(d.transpose);
  const bool compact_wave64 =
      d.tag() == GLOBAL_MEM && d.wf_size == 64 && kind == TransposeKind::TR16_B128;
  switch (kind) {
  case TransposeKind::B64_TR_B8:
    transpose_b64_tr_b8(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::WMMA_TR_B8:
    transpose_wmma_tr_b8(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::CDNA5_DS_TR_B8:
    transpose_cdna5_ds_tr_b8(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::TR16_B128:
    transpose_tr16_b128(d.response_data, d.num_elems, d.wf_size, compact_wave64);
    break;
  case TransposeKind::B64_TR_B16:
    transpose_b64_tr_b16(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::TR_B4:
    transpose_b4(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::TR_B6:
    transpose_b6(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::NONE:
    break;
  }
  if (compact_wave64)
    d.num_elems /= 2;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_DS_TRANSPOSE_H_
