/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/diagnostics_fakes.h"

#include <algorithm>
#include <cstring>

namespace {

constexpr std::size_t kFgetsChunk = 4095;
constexpr int kToolMissingExit = 127;

int DefaultChildRunStream(const char*, int, char* output, int outputSize, ncclDiagChildLineFn, void*,
                          bool* outputTruncated) {
  DeliverChildOutput("", output, outputSize, nullptr, nullptr, outputTruncated);
  return kToolMissingExit;
}

int DefaultChildRun(const char* command, int timeoutSec, char* output, int outputSize, bool* outputTruncated) {
  return DefaultChildRunStream(command, timeoutSec, output, outputSize, nullptr, nullptr, outputTruncated);
}

}  // namespace

std::function<int(const char*, int, char*, int, bool*)> g_ncclDiagChildRun = DefaultChildRun;
std::function<int(const char*, int, char*, int, ncclDiagChildLineFn, void*, bool*)> g_ncclDiagChildRunStream =
    DefaultChildRunStream;

void DeliverChildOutput(const std::string& text, char* output, int outputSize, ncclDiagChildLineFn onLine, void* ctx,
                        bool* outputTruncated) {
  if (output != nullptr && outputSize > 0) {
    output[0] = '\0';
  }
  if (outputTruncated != nullptr) {
    *outputTruncated = false;
  }
  int used = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    const std::size_t newline = text.find('\n', pos);
    const std::size_t end = newline == std::string::npos ? text.size() : newline + 1;
    const std::string chunk = text.substr(pos, std::min(end - pos, kFgetsChunk));
    pos += chunk.size();
    if (onLine != nullptr) {
      onLine(chunk.c_str(), ctx);
    }
    const int got = static_cast<int>(chunk.size());
    int copy = 0;
    if (output != nullptr && used + 1 < outputSize) {
      copy = std::min(got, outputSize - used - 1);
      std::memcpy(output + used, chunk.data(), copy);
      used += copy;
      output[used] = '\0';
    }
    if (outputTruncated != nullptr && copy < got) {
      *outputTruncated = true;
    }
  }
}

void ResetDiagnosticsFakes() {
  g_ncclDiagChildRun = DefaultChildRun;
  g_ncclDiagChildRunStream = DefaultChildRunStream;
}

int ncclDiagChildRunStream(const char* command, int timeoutSec, char* output, int outputSize,
                           ncclDiagChildLineFn onLine, void* onLineCtx, bool* outputTruncated) {
  return g_ncclDiagChildRunStream(command, timeoutSec, output, outputSize, onLine, onLineCtx, outputTruncated);
}

int ncclDiagChildRun(const char* command, int timeoutSec, char* output, int outputSize, bool* outputTruncated) {
  return g_ncclDiagChildRun(command, timeoutSec, output, outputSize, outputTruncated);
}
