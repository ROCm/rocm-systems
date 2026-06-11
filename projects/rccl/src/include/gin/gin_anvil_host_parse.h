/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_GIN_ANVIL_HOST_PARSE_H_
#define RCCL_GIN_ANVIL_HOST_PARSE_H_

#include <cstdint>
#include <cstdlib>

// Host-side parsing for GIN_ANVIL environment knobs (unit-tested; used by gin_host_anvil.cc).

inline int ncclGinAnvilParseNumSdmaChannels(const char* nccl_gin_anvil_env,
                                             const char* rocshmem_sdma_num_channels_env) {
  const char* env = nccl_gin_anvil_env;
  if (env == nullptr) env = rocshmem_sdma_num_channels_env;
  int numChannels = env ? static_cast<int>(std::strtol(env, nullptr, 0)) : 4;
  if (numChannels < 1) numChannels = 1;
  if (numChannels > 8) numChannels = 8;
  return numChannels;
}

// Returns chunk size in bytes from NCCL_GIN_ANVIL_SDMA_CHUNK_MB-style string (megabytes).
inline uint32_t ncclGinAnvilParseSdmaChunkBytes(const char* chunk_mb_env) {
  unsigned long mb = 8;
  if (chunk_mb_env != nullptr && chunk_mb_env[0] != '\0')
    mb = std::strtoul(chunk_mb_env, nullptr, 0);
  if (mb < 1) mb = 1;
  if (mb > 128) mb = 128;
  return static_cast<uint32_t>(mb << 20);
}

#endif  // RCCL_GIN_ANVIL_HOST_PARSE_H_
