/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Declarations for the rccl-tests diff hooks. Include only after common.h.
 * Compiled in only when RCCL_TESTS_DIFF is set, so the stock *_perf binaries
 * do not see this header.
 ************************************************************************/
#ifndef RCCL_TESTS_DIFF_API_H
#define RCCL_TESTS_DIFF_API_H

int rcclDiffShouldTakeover();
testResult_t rcclDiffTimeTest(struct threadArgs* args, ncclDataType_t type,
    const char* typeName, ncclRedOp_t op, const char* opName, int root);

void rcclDiffSetRunCycles(int n);
int rcclDiffGetRunCycles();

// Return 1 when the stock writer should skip its output.
int rcclDiffOnPreamble(size_t nBytes, size_t nElem, const char* typeName,
    const char* opName, int root);
int rcclDiffOnBody(double timeUsec, double algBw, double busBw, bool reportErrors,
    int64_t wrongElts, bool outOfPlace);
int rcclDiffOnNullBody();
int rcclDiffOnTerminator();
int rcclDiffWriteHeader(bool reportCputime, bool reportTimestamps,
    bool enableOutOfPlace, bool enableInPlace, bool outputAlgoProtoChannels);
int rcclDiffWriteFooter(const int errors[], const double bw[], double checkAvgBw,
    const char* programName);

#endif
