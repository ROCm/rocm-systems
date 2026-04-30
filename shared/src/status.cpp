////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "shared/include/status.h"

const char* ErrorCodeToString(const ErrorCode code) {
  switch (code) {
  case ErrorCode::Success:
    return "success";
  case ErrorCode::DeviceLost:
    return "device lost";
  case ErrorCode::NotReady:
    return "not ready";
  case ErrorCode::UnSupported:
    return "unsupported";
  case ErrorCode::OutOfMemory:
    return "out of memory";
  case ErrorCode::InitializationFailed:
    return "initialization failed";
  case ErrorCode::OutOfGpuMemory:
    return "out of gpu memory";
  case ErrorCode::OutOfHandleApeMemory:
    return "out of handle ape memory";
  case ErrorCode::Timeout:
    return "timeout";
  case ErrorCode::SyscallFail:
    return "syscall fail";
  case ErrorCode::InvalidParams:
    return "invalid params";
  case ErrorCode::InvalidPointer:
    return "invalid pointer";
  case ErrorCode::IncompatibleDevice:
    return "incompatible device";
  case ErrorCode::CheckError:
    return "check error";
  case ErrorCode::NotFound:
    return "not found";
  case ErrorCode::SameProcessSameDevice:
    return "same process same device";
  default:
    return "unknown";
  }
}
