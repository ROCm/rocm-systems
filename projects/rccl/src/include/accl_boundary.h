// SPDX-License-Identifier: MIT
// Optional RCCL/ACCL protocol carried by existing v5 NetPlugin events.
// No profiler ABI structures or event/state numbers are changed.
#ifndef RCCL_ACCL_BOUNDARY_H_
#define RCCL_ACCL_BOUNDARY_H_
#include <stdint.h>
#define RCCL_ACCL_BOUNDARY_INFO INT64_C(0x4143434c0001)
#define RCCL_ACCL_PROXY_BEGIN   INT64_C(0x4143434c0002)
#define RCCL_ACCL_PROXY_ADD     INT64_C(0x4143434c0003)
#define RCCL_ACCL_BOUNDARY_FINAL INT64_C(0x4143434c0004)
#define RCCL_ACCL_BOUNDARY_VERSION 2
// Data of BOUNDARY_INFO (every kernel launch) and BOUNDARY_FINAL (comm destroy).
struct rcclAcclBoundaryInfo {
  int version;
  int supported; // eager, one collective/plan, one GPU/process; socket OOB
  void* context;
  // Collective across the comm: every rank calls it whenever it is non-NULL,
  // voting locallyValid. Returns 0 only if every rank voted valid. NULL when
  // not every rank launches this plan (P2P-only).
  int (*synchronize)(void* context, int locallyValid);
};
#endif
