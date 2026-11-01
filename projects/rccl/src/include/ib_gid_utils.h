/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_IB_GID_UTILS_H_
#define NCCL_IB_GID_UTILS_H_

#include <arpa/inet.h>
#include <endian.h>
#include <stdint.h>
#include <string.h>

// Deliberately free of ibverbs types: rocSHMEM's ibv_core.hpp and RCCL's ibvcore.h cannot be
// included in the same translation unit.

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static inline uint16_t ncclIbGidSubnetPrefix16(uint64_t subnet_prefix) {
  return (be64toh(subnet_prefix) & 0xffff);
}

static inline uint16_t ncclIbGidFlid(const uint8_t raw[16]) {
  uint16_t flid;
  memcpy(&flid, raw + 4, sizeof(flid));
  return ntohs(flid);
}

#endif
