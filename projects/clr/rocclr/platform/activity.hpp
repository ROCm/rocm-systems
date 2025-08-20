/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include "top.hpp"

#include <atomic>
#include <array>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace amd {
class Command;
}  // namespace amd

enum OpId { OP_ID_DISPATCH = 0, OP_ID_COPY = 1, OP_ID_BARRIER = 2, OP_ID_NUMBER = 3 };

#include "prof_protocol.h"

namespace amd::activity_prof {

extern std::atomic<int (*)(activity_domain_t domain, uint32_t operation_id, void* data)>
    report_activity;

#if defined(__linux__)
extern __thread activity_correlation_id_t correlation_id __attribute__((tls_model("initial-exec")));
#elif defined(_WIN32)
extern __declspec(thread) activity_correlation_id_t correlation_id;
#endif  // defined(_WIN32)

constexpr OpId OperationId(cl_command_type commandType) {
  switch (commandType) {
    case CL_COMMAND_NDRANGE_KERNEL:
    case CL_COMMAND_TASK:
      return OP_ID_DISPATCH;
    case CL_COMMAND_READ_BUFFER:
    case CL_COMMAND_READ_BUFFER_RECT:
    case CL_COMMAND_WRITE_BUFFER:
    case CL_COMMAND_WRITE_BUFFER_RECT:
    case CL_COMMAND_COPY_BUFFER:
    case CL_COMMAND_COPY_BUFFER_RECT:
    case CL_COMMAND_FILL_BUFFER:
    case CL_COMMAND_READ_IMAGE:
    case CL_COMMAND_WRITE_IMAGE:
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_FILL_IMAGE:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      return OP_ID_COPY;
    case CL_COMMAND_MARKER:
      return OP_ID_BARRIER;
    default:
      return OP_ID_NUMBER;
  }
}

bool IsEnabled(OpId operation_id);
void ReportActivity(const amd::Command& command);


const char* getOclCommandKindString(cl_command_type kind);
}  // namespace amd::activity_prof