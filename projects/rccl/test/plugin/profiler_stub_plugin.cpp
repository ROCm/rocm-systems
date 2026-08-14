/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Profiler plugin stub for ProfilerPluginCloseTests. It loads, but exports no
// ncclProfiler_v* symbol, which is what makes ncclProfilerPluginInit() reject it.
//
// Since RCCL rejects it, none of its code is ever called through a plugin entry
// point, so the load is recorded from a constructor instead: it appends a line to
// the file named in RCCL_TEST_PROFILER_STUB_LOAD_FILE. The test needs that record
// to tell an unloaded plugin apart from one that was never loaded at all.

#include <stdio.h>
#include <stdlib.h>

static void recordLine(const char* envVar) {
  const char* path = getenv(envVar);
  if (path == nullptr) return;

  FILE* f = fopen(path, "a");
  if (f == nullptr) return;

  fputs("1\n", f);
  fclose(f);
}

__attribute__((constructor)) static void recordLoad(void) {
  recordLine("RCCL_TEST_PROFILER_STUB_LOAD_FILE");
}
