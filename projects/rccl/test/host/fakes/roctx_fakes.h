/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// roctx_scoped_range_in (src/misc/roctx.cc), recorded. Every range is kept with its message or schema name
// and a copy of its first payload entry, so a test can read the payload after the range has closed.

#ifndef RCCL_TEST_HOST_ROCTX_FAKES_H_
#define RCCL_TEST_HOST_ROCTX_FAKES_H_

#include <string>
#include <vector>

struct RoctxRange {
  std::string name;
  std::vector<unsigned char> payload;  // empty for a message-only range
};

// Every range opened since the last reset, in order.
std::vector<RoctxRange> RoctxRanges();
// Ranges constructed but not yet destroyed.
int RoctxRangesOpen();
void ResetRoctxFakes();

#endif  // RCCL_TEST_HOST_ROCTX_FAKES_H_
