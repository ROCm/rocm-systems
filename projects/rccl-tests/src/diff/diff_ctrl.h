/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Control API shared by the diff main, the library shim, and the driver.
 * No dependency on the rccl-tests headers.
 ************************************************************************/
#ifndef RCCL_TESTS_DIFF_CTRL_H
#define RCCL_TESTS_DIFF_CTRL_H

enum {
  kRcclDiffInterleaveSize = 0,
  kRcclDiffInterleaveSweep = 1,
  kRcclDiffOrderAB = 0,
  kRcclDiffOrderBA = 1,
  kRcclDiffOrderRandom = 2,
  kRcclDiffStatMedian = 0,
  kRcclDiffStatMean = 1
};

// spec is "A=/path/librccl.so.1,B=/path/librccl.so.1".
// Returns 0 on success. On failure writes a NUL-terminated message to err.
int rcclDiffLoadLibraries(const char* spec, int deepbind, char* err, int errLen);

void rcclDiffConfigure(int reps, int interleave, int order, int stat, int seed,
    const char* jsonPath, const char* csvPath);

void rcclDiffSetActive(int lib);
int rcclDiffActiveLib();

void rcclDiffGetLibInfo(int index, const char** label, const char** path, int* version);

#endif
