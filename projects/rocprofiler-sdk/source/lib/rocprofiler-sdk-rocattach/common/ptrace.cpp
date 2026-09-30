// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "ptrace.hpp"

#include <errno.h>

namespace rocprofiler
{
namespace rocattach
{
// Very limited list of operations for logging only.
const char*
ptrace_op_name(__ptrace_request op)
{
    switch(op)
    {
        case PTRACE_SEIZE: return "PTRACE_SEIZE";
        case PTRACE_DETACH: return "PTRACE_DETACH";
        case PTRACE_POKEDATA: return "PTRACE_POKEDATA";
        case PTRACE_PEEKDATA: return "PTRACE_PEEKDATA";
        case PTRACE_INTERRUPT: return "PTRACE_INTERRUPT";
        case PTRACE_GETREGS: return "PTRACE_GETREGS";
        case PTRACE_SETREGS: return "PTRACE_SETREGS";
        case PTRACE_CONT: return "PTRACE_CONT";
        default: return "unknown op";
    }
}

// Translates ptrace errno into rocattach status errors
rocattach_status_t
convert_ptrace_error(int error)
{
    switch(error)
    {
        case EPERM: return ROCATTACH_STATUS_ERROR_PTRACE_OPERATION_NOT_PERMITTED;
        case ESRCH: return ROCATTACH_STATUS_ERROR_PTRACE_PROCESS_NOT_FOUND;
        default: return ROCATTACH_STATUS_ERROR_PTRACE_ERROR;
    }
}

}  // namespace rocattach
}  // namespace rocprofiler
