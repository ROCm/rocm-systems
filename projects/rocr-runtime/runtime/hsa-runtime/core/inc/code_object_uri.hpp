////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
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
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
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

// Helpers that build code-object URIs. amd-dbgapi / rocgdb resolve a loaded
// code object's contents by fetching the bytes named by its URI, so a code
// object must be named by where its bytes actually live. Extracted from
// amd_hsa_loader.cpp so consumers other than the loader (e.g. the HotSwap
// rewrite path, which loads a code object from a rewritten in-memory buffer)
// can produce the same URI forms.

#ifndef HSA_RUNTIME_CORE_INC_CODE_OBJECT_URI_HPP_
#define HSA_RUNTIME_CORE_INC_CODE_OBJECT_URI_HPP_

#include <cstddef>
#include <string>

#include "inc/hsa.h"

namespace rocr {
namespace amd {
namespace hsa {
namespace loader {

/// Build a URI that resolves to \p size bytes at \p memory in this process:
/// "memory://<pid>#offset=0x<addr>&size=<size>". This is the fallback form used
/// when the bytes cannot be attributed to a file on disk.
std::string GetUriFromMemoryAddress(const void* memory, size_t size);

/// If \p memory falls inside a loaded executable/shared-object image mapped from
/// a file, return a file:// URI naming that file and the byte offset within it;
/// otherwise fall back to GetUriFromMemoryAddress.
std::string GetUriFromMemoryInExecutableFile(const void* memory, size_t size);

/// If \p memory falls inside a file-backed mapping (per /proc/self/maps on
/// Linux), return a file:// URI naming that file; otherwise fall back to
/// GetUriFromMemoryAddress.
std::string GetUriFromMemoryInMmapedFile(const void* memory, size_t size);

/// Build a file:// URI for the file behind \p file_descriptor (resolved via
/// /proc/self/fd on Linux), naming \p offset / \p size unless \p is_complete_file
/// is set. Falls back to GetUriFromMemoryAddress(\p memory, \p size) if the path
/// cannot be resolved.
std::string GetUriFromFile(hsa_file_t file_descriptor, size_t offset,
                           size_t size, bool is_complete_file,
                           const void* memory);

}  // namespace loader
}  // namespace hsa
}  // namespace amd
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_CODE_OBJECT_URI_HPP_
