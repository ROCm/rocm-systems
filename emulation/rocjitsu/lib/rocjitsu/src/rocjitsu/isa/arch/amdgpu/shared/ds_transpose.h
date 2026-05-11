// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_DS_TRANSPOSE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_DS_TRANSPOSE_H_

/// @file ds_transpose.h
/// @brief Cross-lane transpose logic for DS_READ_B64_TR_* instructions.
///
/// @details CDNA4 transpose read instructions read raw data from LDS per-lane,
/// then apply a cross-lane shuffle to produce the transposed matrix layout
/// expected by MFMA instructions.
///
/// TR_B8 uses byte-level transpose with groups of 4 consecutive lanes.
/// TR_B16 uses word-level transpose with stride-4 pairing within groups of 8.

#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

enum class TransposeKind : uint8_t { NONE, TR_B4, TR_B6, TR_B8, TR_B16 };

/// @brief B64 byte-level transpose (TR_B8 only).
///
/// Groups of 4 source lanes, 8 byte iterations per group.
/// Each iteration packs one byte from each of 4 source lanes into a dword
/// and writes to a cross-lane destination computed by compact formula.
inline void transpose_b64(std::vector<uint8_t> &response_data, uint32_t num_elems,
                          uint32_t wf_size) {
  constexpr uint32_t lanes_per_half = 32;
  constexpr uint32_t source_group_size = 4;
  constexpr uint32_t bytes_per_lane = 8;

  const uint32_t bytes_per_lane_total = num_elems * 4;
  const uint32_t num_halves = (wf_size > lanes_per_half) ? 2u : 1u;

  std::vector<uint8_t> output(response_data.size(), 0);

  for (uint32_t half_index = 0; half_index < num_halves; ++half_index) {
    const uint32_t lane_base = half_index * lanes_per_half;

    for (uint32_t group_start = 0; group_start < lanes_per_half; group_start += source_group_size) {
      for (uint32_t byte_index = 0; byte_index < bytes_per_lane; ++byte_index) {
        uint32_t packed_dword = 0;
        for (uint32_t lane_in_group = 0; lane_in_group < source_group_size; ++lane_in_group) {
          uint32_t source_lane = lane_base + group_start + lane_in_group;
          uint32_t source_offset = source_lane * bytes_per_lane_total + byte_index;
          uint8_t source_byte =
              (source_offset < response_data.size()) ? response_data[source_offset] : 0;
          packed_dword |= static_cast<uint32_t>(source_byte) << (lane_in_group * 8);
        }

        uint32_t dest_vgpr_index = (group_start % 16) / 8;
        uint32_t dest_lane = lane_base + (group_start / 16) * 16 +
                             ((group_start / source_group_size) % 2) * bytes_per_lane + byte_index;
        uint32_t dest_offset = dest_lane * bytes_per_lane_total + dest_vgpr_index * 4;

        if (dest_offset + 4 <= output.size())
          std::memcpy(&output[dest_offset], &packed_dword, 4);
      }
    }
  }

  response_data = std::move(output);
}

/// @brief TR_B16: 16-bit element transpose (2 VGPRs, 8 bytes per lane).
///
/// Groups of 8 source lanes with stride-4 pairing: source lanes i and i+4
/// are packed together. Each pair produces 4 dwords (one per halfword).
/// dest_vgpr = group / 8 (0 or 1), dest_lane = lane_within_group * 4 + hw.
inline void transpose_b16(std::vector<uint8_t> &response_data, uint32_t num_elems,
                          uint32_t wf_size) {
  constexpr uint32_t lanes_per_half = 32;
  constexpr uint32_t group_size = 8;
  constexpr uint32_t halfwords_per_source_lane = 4;
  constexpr uint32_t pair_stride = 4;

  const uint32_t bytes_per_lane_total = num_elems * 4;
  const uint32_t num_halves = (wf_size > lanes_per_half) ? 2u : 1u;

  std::vector<uint8_t> output(response_data.size(), 0);

  for (uint32_t half_index = 0; half_index < num_halves; ++half_index) {
    const uint32_t lane_base = half_index * lanes_per_half;

    for (uint32_t group_start = 0; group_start < lanes_per_half; group_start += group_size) {
      for (uint32_t lane_in_group = 0; lane_in_group < pair_stride; ++lane_in_group) {
        uint32_t source_lo = lane_base + group_start + lane_in_group;
        uint32_t source_hi = source_lo + pair_stride;

        for (uint32_t halfword_index = 0; halfword_index < halfwords_per_source_lane;
             ++halfword_index) {
          uint16_t word_lo = 0, word_hi = 0;
          uint32_t offset_lo = source_lo * bytes_per_lane_total + halfword_index * 2;
          uint32_t offset_hi = source_hi * bytes_per_lane_total + halfword_index * 2;
          if (offset_lo + 2 <= response_data.size())
            std::memcpy(&word_lo, &response_data[offset_lo], 2);
          if (offset_hi + 2 <= response_data.size())
            std::memcpy(&word_hi, &response_data[offset_hi], 2);
          uint32_t packed_dword =
              static_cast<uint32_t>(word_lo) | (static_cast<uint32_t>(word_hi) << 16);

          uint32_t dest_vgpr_index = (group_start / group_size) % num_elems;
          uint32_t dest_lane = lane_base + (group_start / (group_size * num_elems)) * 16 +
                               lane_in_group * halfwords_per_source_lane + halfword_index;
          uint32_t dest_offset = dest_lane * bytes_per_lane_total + dest_vgpr_index * 4;

          if (dest_offset + 4 <= output.size())
            std::memcpy(&output[dest_offset], &packed_dword, 4);
        }
      }
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
  switch (kind) {
  case TransposeKind::TR_B8:
    transpose_b64(d.response_data, d.num_elems, d.wf_size);
    break;
  case TransposeKind::TR_B16:
    transpose_b16(d.response_data, d.num_elems, d.wf_size);
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
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_DS_TRANSPOSE_H_
