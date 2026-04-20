////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2026, Advanced Micro Devices, Inc. All rights reserved.
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

// Darwin shim for glibc's <link.h>.
//
// ROCR's loader publishes an `_amdgpu_r_debug` / `link_map` chain in the
// same shape that glibc's dynamic linker exports via _r_debug, so a
// debugger (ROCgdb) can walk GPU-side loaded code objects alongside the
// host process's shared-library list. On Linux glibc and musl this
// protocol is standard; Darwin has no analogue (dyld uses a wholly
// different dyld_all_image_infos struct).
//
// For now we provide a source-compatible stub so ROCR compiles and its
// own internal bookkeeping works. A Darwin-aware debugger will need a
// dedicated ROCR↔lldb bridge; that's out of scope for initial bring-up.
//
// Field set is narrowed to exactly what loader/executable.{hpp,cpp},
// core/runtime/runtime.cpp and core/runtime/amd_topology.cpp reference.

#ifndef HSA_RUNTIME_CORE_INC_LINK_DARWIN_H_
#define HSA_RUNTIME_CORE_INC_LINK_DARWIN_H_

#if !defined(__APPLE__)
#error "link_darwin.h should only be included on Darwin — use <link.h> otherwise"
#endif

#include <stdint.h>

struct link_map {
  // l_addr holds the base-relocation delta for the loaded code object —
  // consumed by ROCgdb to translate ELF static VAs into runtime VAs.
  uint64_t       l_addr;
  char*          l_name;  // owned by the loader — strdup'd URI string.
  void*          l_ld;    // unused on Darwin but present for struct parity.
  struct link_map* l_next;
  struct link_map* l_prev;
};

struct r_debug {
  int r_version;
  struct link_map* r_map;
  uintptr_t r_brk;
  enum {
    RT_CONSISTENT,  // Mapping change complete.
    RT_ADD,         // Link map just added.
    RT_DELETE,      // Link map about to be removed.
  } r_state;
  uintptr_t r_ldbase;
};

#endif  // HSA_RUNTIME_CORE_INC_LINK_DARWIN_H_
