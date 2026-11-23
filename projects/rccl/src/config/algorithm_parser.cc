/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "config/algorithm_parser.h"
#include "config/algorithm_registry.h"
#include "utils.h"  // strtok_r -> strtok_s shim on Windows
#include <cstring>  // strtok_r
#include <string>

// Return sel with all whitespace removed (case is handled at match time).
static std::string removeWhitespace(const char* sel) {
  std::string out;
  for (const char* p = sel; *p != '\0'; p++) {
    if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') continue;
    out.push_back(*p);
  }
  return out;
}

ncclResult_t ncclAlgParse(const char* sel, rcclTuningMask* outMask) {
  *outMask = rcclTuningMask();
  if (sel == NULL) return ncclSuccess;
  std::string s = removeWhitespace(sel);   // strtok_r mutates this buffer in place
  if (s.empty()) return ncclSuccess;

  const rcclTuningMask all = ncclAlgAllBits();
  rcclTuningMask mask{};
  char* termSave;
  for (char* term = strtok_r(&s[0], ",", &termSave);        // ',' = OR
       term != NULL; term = strtok_r(NULL, ",", &termSave)) {
    rcclTuningMask termMask = all;
    int factorCount = 0;
    char* facSave;
    for (char* factor = strtok_r(term, "+", &facSave);       // '+' = AND
         factor != NULL; factor = strtok_r(NULL, "+", &facSave)) {
      bool neg = factor[0] == '^';                           // '^' negates
      const char* tag = neg ? factor + 1 : factor;
      if (tag[0] == '\0') return ncclInvalidArgument;        // bare '^'
      rcclTuningMask tm = ncclAlgTagMask(tag);
      if (tm.none()) return ncclInvalidArgument;             // unknown tag
      termMask &= neg ? (all & ~tm) : tm;
      factorCount++;
    }
    if (factorCount == 0) return ncclInvalidArgument;        // term was only '+', e.g. "a,+,b"
    if (termMask.none()) return ncclInvalidArgument;         // contradictory term
    mask |= termMask;
  }
  *outMask = mask;
  return ncclSuccess;
}
