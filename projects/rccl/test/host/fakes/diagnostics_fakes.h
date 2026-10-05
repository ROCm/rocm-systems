/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the external-tool runners src/diagnostics.cc owns: no test spawns a process, each returns an exit code.

#ifndef RCCL_TEST_HOST_FAKES_DIAGNOSTICS_FAKES_H_
#define RCCL_TEST_HOST_FAKES_DIAGNOSTICS_FAKES_H_

#include "diagnostics.h"

#include <functional>
#include <string>

extern std::function<int(const char*, int, char*, int, bool*)> g_ncclDiagChildRun;
extern std::function<int(const char*, int, char*, int, ncclDiagChildLineFn, void*, bool*)> g_ncclDiagChildRunStream;

// Hands `text` over as the real runner does: onLine per fgets line, output NUL-terminated and clamped to outputSize-1.
void DeliverChildOutput(const std::string& text, char* output, int outputSize, ncclDiagChildLineFn onLine, void* ctx,
                        bool* outputTruncated);

void ResetDiagnosticsFakes();

#endif  // RCCL_TEST_HOST_FAKES_DIAGNOSTICS_FAKES_H_
