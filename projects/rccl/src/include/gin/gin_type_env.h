/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_GIN_TYPE_ENV_H_
#define RCCL_GIN_TYPE_ENV_H_

#include <cerrno>
#include <cstdint>
#include <cstdlib>

// NCCL_GIN_TYPE forces exactly one GIN backend. When it is unset, NCCL_GIN_TYPES
// (RCCL-only) is a comma-separated list of the backend types allowed to be active
// together, e.g. "6,7" for rocSHMEM GDA alongside Anvil SDMA. With neither set,
// each backend keeps its default: rocSHMEM GDA stays off, the others stay on.
//
// Both are read with getenv() on every call, not through NCCL_PARAM, whose cached
// value would ignore a later setenv() from the plugin tests.

// Parses list into a bitmask of GIN types (bit t set for type t). Returns false if
// any entry is not an integer in [1, 63]; valid entries are still recorded in *mask.
static inline bool ncclGinParseTypeList(const char* list, uint64_t* mask) {
  bool valid = true;
  *mask = 0;
  const char* p = list;
  while (*p) {
    while (*p == ' ' || *p == '\t') p++;
    char* end = nullptr;
    errno = 0;
    long type = strtol(p, &end, 10);
    while (*end == ' ' || *end == '\t') end++;
    if (end == p || errno != 0 || type < 1 || type > 63 || (*end != ',' && *end != '\0')) {
      valid = false;
      while (*end != ',' && *end != '\0') end++;
    } else {
      *mask |= (uint64_t)1 << type;
    }
    p = (*end == ',') ? end + 1 : end;
  }
  return valid;
}

// Returns true and fills *mask when NCCL_GIN_TYPE is unset and NCCL_GIN_TYPES names
// at least one valid type. Otherwise the list does not apply.
static inline bool ncclGinTypeListMask(uint64_t* mask) {
  if (getenv("NCCL_GIN_TYPE")) return false;
  const char* list = getenv("NCCL_GIN_TYPES");
  if (!list) return false;
  ncclGinParseTypeList(list, mask);
  return *mask != 0;
}

// Whether the backend of the given type may initialize. enabledByDefault is its
// behavior when neither NCCL_GIN_TYPE nor a usable NCCL_GIN_TYPES is set.
static inline bool ncclGinTypeEnabled(int type, bool enabledByDefault) {
  const char* single = getenv("NCCL_GIN_TYPE");
  if (single) return atoi(single) == type;
  uint64_t mask;
  if (ncclGinTypeListMask(&mask)) return (mask >> type) & 1;
  return enabledByDefault;
}

#endif  // RCCL_GIN_TYPE_ENV_H_
