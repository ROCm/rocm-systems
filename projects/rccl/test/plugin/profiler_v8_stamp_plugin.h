/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Query interface of the v8 timestamp plugin (profiler_v8_stamp_plugin.cpp),
// read by ProfilerV8TimestampTests through dlsym(RCCL_TEST_PROF_V8_SNAPSHOT).

#ifndef RCCL_TEST_PROFILER_V8_STAMP_PLUGIN_H_
#define RCCL_TEST_PROFILER_V8_STAMP_PLUGIN_H_

#include <stdint.h>

#define RCCL_TEST_PROF_V8_SNAPSHOT "rcclTestProfV8Snapshot"

struct RcclProfV8Stats {
  uint64_t inits;          // init() calls
  uint64_t initsNoConfig;  // init() calls given a NULL config
  uint64_t starts;         // startEvent() calls
  uint64_t stops;          // stopEvent() calls
  uint64_t states;         // recordEventState() calls
  uint64_t unstamped;      // of the above, carried no tsCpu (or no descriptor)
  uint64_t stopBeforeStart;  // stops stamped earlier than their own start
  uint64_t minTs;          // smallest tsCpu seen, UINT64_MAX when none
  uint64_t maxTs;          // largest tsCpu seen, 0 when none
};

typedef void (*RcclProfV8SnapshotFn)(struct RcclProfV8Stats* out);

#endif // RCCL_TEST_PROFILER_V8_STAMP_PLUGIN_H_
