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

// NCCL_GIN_TYPE forces one backend; if unset, RCCL_GIN_TYPE_LIST lists the types allowed together.
// Uncached getenv() so tests can setenv() between cases.

// Backend types a list entry may name: NCCL_GIN_TYPE_PROXY through NCCL_GIN_MAX_TYPES - 1.
// gin.cc static_asserts both against ncclGinType_t, which this header cannot include.
static constexpr long kRcclGinTypeListMin = 2;
static constexpr long kRcclGinTypeListMax = 7;

// NCCL_GIN_TYPE, or nullptr when it is unset or empty. ncclParamGinType() also reads an
// empty value as unset, so both readers agree on whether the variable is set.
static inline const char* ncclGinTypeEnv() {
  const char* single = getenv("NCCL_GIN_TYPE");
  return (single && single[0] != '\0') ? single : nullptr;
}

// Parses list into a bitmask of GIN types (bit t set for type t). Returns false if any
// entry is not an integer in [kRcclGinTypeListMin, kRcclGinTypeListMax]; valid entries
// are still recorded in *mask.
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
    if (end == p || errno != 0 || type < kRcclGinTypeListMin || type > kRcclGinTypeListMax ||
        (*end != ',' && *end != '\0')) {
      valid = false;
      while (*end != ',' && *end != '\0') end++;
    } else {
      *mask |= (uint64_t)1 << type;
    }
    p = (*end == ',') ? end + 1 : end;
  }
  return valid;
}

// Returns true and fills *mask when NCCL_GIN_TYPE is unset and RCCL_GIN_TYPE_LIST names
// at least one valid type. Otherwise the list does not apply.
static inline bool ncclGinTypeListMask(uint64_t* mask) {
  if (ncclGinTypeEnv()) return false;
  const char* list = getenv("RCCL_GIN_TYPE_LIST");
  if (!list) return false;
  ncclGinParseTypeList(list, mask);
  return *mask != 0;
}

// Whether the backend of the given type may initialize. enabledByDefault is its
// behavior when neither NCCL_GIN_TYPE nor a usable RCCL_GIN_TYPE_LIST is set.
static inline bool ncclGinTypeEnabled(int type, bool enabledByDefault) {
  if (const char* single = ncclGinTypeEnv()) return atoi(single) == type;
  uint64_t mask;
  if (ncclGinTypeListMask(&mask)) return (mask >> type) & 1;
  return enabledByDefault;
}

#endif  // RCCL_GIN_TYPE_ENV_H_
