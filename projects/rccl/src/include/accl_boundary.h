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
struct rcclAcclBoundaryInfo {
  int version;
  int supported; // eager, one collective/plan, one GPU/process; socket OOB
  void* context;
  // Returns 0 only if every rank reports a valid local boundary.
  int (*synchronize)(void* context, int locallyValid);
};
#endif
