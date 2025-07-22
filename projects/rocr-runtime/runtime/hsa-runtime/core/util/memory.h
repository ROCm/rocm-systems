////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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

// Memory related utility functions.

#ifndef HSA_RUNTIME_CORE_UTIL_MEMORY_H_
#define HSA_RUNTIME_CORE_UTIL_MEMORY_H_

#ifdef __linux__
#include "inc/hsa.h"
#include <sys/mman.h>
#endif

namespace rocr {

#ifdef __linux__
/// @brief Converts @ref hsa_access_permission_t to mmap memory protection
///        flags.
__forceinline int PermissionsToMmapFlags(hsa_access_permission_t perms) {
  switch (perms) {
    case HSA_ACCESS_PERMISSION_RO:
      return PROT_READ;
    case HSA_ACCESS_PERMISSION_WO:
      return PROT_WRITE;
    case HSA_ACCESS_PERMISSION_RW:
      return PROT_READ | PROT_WRITE;
    case HSA_ACCESS_PERMISSION_NONE:
    default:
      return PROT_NONE;
  }
}
#endif

}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_MEMORY_H_
