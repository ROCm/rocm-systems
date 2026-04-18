////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
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

// Darwin stub for the GPU core-dump harvester.
//
// The Linux implementation (lnx/amd_core_dump.cpp) builds an ELF core file
// from the GPU's VRAM + queue state at the moment of a crashing dispatch. It
// depends on a handful of Linux-only mechanisms:
//   - /proc/self/coredump_filter and the kernel's core-piping behavior
//   - getrlimit(RLIMIT_CORE) semantics matching Linux's core-file workflow
//   - fork() + exec() into the user's configured `kernel.core_pattern` helper
//   - ELF32/ELF64 note generation (PT_NOTE NT_PRSTATUS / NT_PRPSINFO)
//
// macOS has its own crash-reporter pipeline (ReportCrash) that swallows
// SIGSEGV before our code can harvest GPU state, and has no analogue of
// core_pattern. Porting the Linux path would require a Darwin-native
// crash-reporter hook (Mach exception ports) plus a separate delivery
// mechanism. For now, stub dump_gpu_core() as a no-op that returns success;
// higher layers treat this as "core dumps disabled."
//
// Follow-up (tracked separately): wire up a Mach-exception-port based crash
// harvester that writes a .dmp file parallel to the Linux core dump format.

#ifdef __APPLE__
#include "hsa.h"
#include "core/inc/amd_core_dump.hpp"

namespace rocr {
namespace amd {
namespace coredump {

hsa_status_t dump_gpu_core() {
  // No-op on Darwin. See file comment for rationale.
  return HSA_STATUS_SUCCESS;
}

}   //  namespace coredump
}   //  namespace amd
}   //  namespace rocr

#endif  // __APPLE__
