/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ABCE_DECODE_H_
#define ABCE_DECODE_H_

#include <cstdint>

#include "abce_types.h"
#include "sdma_packets.h"

namespace abce {

struct DecodedPacket {
  const char* name = "UNKNOWN";
  uint32_t dwords = 1;
  uint32_t op = 0;
  uint32_t sub_op = 0;
};

inline DecodedPacket DecodeSdmaPacket(const uint32_t* words, uint32_t remaining_dwords,
                                      const IsaVersion& isa) {
  if (words == nullptr || remaining_dwords == 0) return {};

  const uint32_t header = words[0];
  const uint32_t op = header & 0xffu;
  const uint32_t sub_op = (header >> 8) & 0xffu;
  const bool gfx125plus = isa.major == 12 && isa.minor >= 5;
  DecodedPacket packet{"UNKNOWN", 1, op, sub_op};

  if (header == 0) {
    packet.name = "NOP";
  } else if (op == SDMA_OP_COPY) {
    const bool wait = (header & (1u << 30)) != 0;
    const bool signal = (header & (1u << 31)) != 0;
    if (sub_op == SDMA_SUBOP_COPY_LINEAR_RECT) {
      packet = {"COPY_LINEAR_RECT", 13, op, sub_op};
    } else if (sub_op == SDMA_SUBOP_COPY_SWAP) {
      packet = {gfx125plus && (wait || signal) ? "COPY_LINEAR_SWAP_WAITSIGNAL"
                                               : "COPY_LINEAR_SWAP",
                7 + (gfx125plus && wait ? 7u : 0u) +
                    (gfx125plus && signal ? 5u : 0u),
                op, sub_op};
    } else if (sub_op == SDMA_SUBOP_COPY_MULTICAST) {
      // A fused packet's parameter DWORD sits after the 7-DWORD wait block.
      const uint32_t parameter_dword = gfx125plus && wait ? 9u : 2u;
      // num_of_destination is encoded as count-1, so one destination is always
      // present even when the DWORD is past the end of the buffer.
      const uint32_t num_destinations =
          (remaining_dwords > parameter_dword ? words[parameter_dword] & 0x3ffu : 0) + 1;
      packet = {gfx125plus && (wait || signal) ? "COPY_LINEAR_MULTICAST_WAITSIGNAL"
                                               : "COPY_LINEAR_MULTICAST",
                5 + 2 * num_destinations + (gfx125plus && wait ? 7u : 0u) +
                    (gfx125plus && signal ? 5u : 0u),
                op, sub_op};
    } else if (sub_op == SDMA_SUBOP_COPY_LINEAR) {
      const bool indirect = (header & ((1u << 20) | (1u << 21))) != 0;
      const bool broadcast = (header & (1u << 27)) != 0;
      if (gfx125plus && (wait || signal)) {
        packet = {indirect ? "COPY_LINEAR_WAITSIGNAL_INDIRECT"
                           : "COPY_LINEAR_WAITSIGNAL",
                  7 + (wait ? 7u : 0u) + (signal ? 5u : 0u), op, sub_op};
      } else if (broadcast) {
        packet = {"COPY_LINEAR_BROADCAST", 9, op, sub_op};
      } else {
        packet = {indirect ? "COPY_LINEAR_INDIRECT" : "COPY_LINEAR", 7, op, sub_op};
      }
    }
  } else if (op == SDMA_OP_FENCE) {
    packet = {sub_op == SDMA_SUBOP_FENCE_64B ? "FENCE_64" : "FENCE",
              sub_op == SDMA_SUBOP_FENCE_64B ? 5u : 4u, op, sub_op};
  } else if (op == SDMA_OP_TRAP) {
    packet = {"TRAP", 2, op, sub_op};
  } else if (op == SDMA_OP_POLL_REGMEM) {
    const bool hdp_flush = remaining_dwords >= 6 && words[0] == 0x8u && words[1] == 0 &&
                           words[2] == 0x80000000u && words[3] == 0 && words[4] == 0 &&
                           words[5] == 0;
    packet = {hdp_flush ? "HDP_FLUSH"
                        : sub_op == SDMA_SUBOP_POLL_MEM_64B ? "POLL_MEM_64"
                                                            : "POLL_REGMEM",
              sub_op == SDMA_SUBOP_POLL_MEM_64B ? 8u : 6u, op, sub_op};
  } else if (op == SDMA_OP_ATOMIC) {
    packet = {"ATOMIC", 8, op, sub_op};
  } else if (op == SDMA_OP_CONST_FILL) {
    packet = {"CONST_FILL", 5, op, sub_op};
  } else if (op == SDMA_OP_TIMESTAMP) {
    packet = {"TIMESTAMP_GET_GLOBAL", 3, op, sub_op};
  } else if (op == SDMA_OP_GCR) {
    packet = {"GCR", gfx125plus ? 6u : 5u, op, sub_op};
  }

  if (packet.dwords > remaining_dwords) {
    packet.name = "TRUNCATED";
    packet.dwords = remaining_dwords;
  }
  return packet;
}

}  // namespace abce

#endif  // ABCE_DECODE_H_
