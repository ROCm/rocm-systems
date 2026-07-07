////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/amd_hsa_loader.hpp"
#include "core/inc/code_object_uri.hpp"
#include "core/inc/runtime.h"

#include <assert.h>

#if defined(__linux__)
#include <link.h>
#include <linux/limits.h>
#include <sys/mman.h>
#include <unistd.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <cstdint>
#else
#include <cstdint>
#endif
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

#if !defined(_WIN32) && !defined(_WIN64)
uintptr_t PAGE_SIZE_MASK{
    [] () {
      uintptr_t page_size = sysconf(_SC_PAGE_SIZE);
      if (page_size == -1) {
        page_size = 1 << 12; // Default page size to 4KiB.
      }
      return ~(page_size - 1);
    } ()
  };
#endif

}  // namespace

namespace rocr {
namespace amd {
namespace hsa {
namespace loader {

/// @brief Default destructor.
CodeObjectReaderImpl::~CodeObjectReaderImpl() {
  if (is_mmap) {
#if !defined(_WIN32) && !defined(_WIN64)
    uintptr_t address = reinterpret_cast<uintptr_t>(code_object_memory);
    uintptr_t adjusted_address = address & PAGE_SIZE_MASK;
    size_t adjusted_size = code_object_size + (address - adjusted_address);
    munmap(reinterpret_cast<void *>(adjusted_address), adjusted_size);
#else
    if (map_base) UnmapViewOfFile(map_base);
    if (map_handle) CloseHandle(static_cast<HANDLE>(map_handle));
#endif  // !defined(_WIN32) && !defined(_WIN64)
  }
}

hsa_status_t CodeObjectReaderImpl::SetFile(
    hsa_file_t _code_object_file_descriptor,
    size_t _code_object_offset,
    size_t _code_object_size) {
  assert(!code_object_memory && "Code object reader wrapper is already set");

#if !defined(_WIN32) && !defined(_WIN64)
  if (_code_object_file_descriptor == -1) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  off_t file_size = __lseek__(_code_object_file_descriptor, 0, SEEK_END);
  if (file_size == (off_t)-1) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }
  if (file_size <= _code_object_offset) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  if (_code_object_size == 0) {
    _code_object_size = file_size - _code_object_offset;
  }
  bool is_complete_file = _code_object_offset == 0 && _code_object_size == file_size;

  off_t adjusted_offset = _code_object_offset & PAGE_SIZE_MASK;
  size_t adjusted_size = _code_object_size + (_code_object_offset - adjusted_offset);
  void *memory = mmap(nullptr, adjusted_size, PROT_READ, MAP_PRIVATE,
                      _code_object_file_descriptor, adjusted_offset);
  if (memory == (void *) -1) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }
  code_object_memory = reinterpret_cast<unsigned char*>(memory) +
                        (_code_object_offset & ~PAGE_SIZE_MASK);
  code_object_size = _code_object_size;
  is_mmap = true;

  uri = GetUriFromFile(_code_object_file_descriptor, _code_object_offset,
                        _code_object_size, is_complete_file, code_object_memory);
#else
  if (_code_object_file_descriptor == INVALID_HANDLE_VALUE ||
      _code_object_file_descriptor == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  LARGE_INTEGER file_size_li;
  if (!GetFileSizeEx(_code_object_file_descriptor, &file_size_li)) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }
  uint64_t file_size = static_cast<uint64_t>(file_size_li.QuadPart);
  if (file_size <= _code_object_offset) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  if (_code_object_size == 0) {
    _code_object_size = static_cast<size_t>(file_size - _code_object_offset);
  }
  bool is_complete_file = _code_object_offset == 0 && _code_object_size == file_size;

  // MapViewOfFile requires the file offset to be aligned to the system
  // allocation granularity (typically 64 KiB on Windows), not page size.
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  uint64_t granularity = si.dwAllocationGranularity;
  uint64_t aligned_offset = _code_object_offset & ~(granularity - 1);
  size_t slop = static_cast<size_t>(_code_object_offset - aligned_offset);
  size_t adjusted_size = _code_object_size + slop;

  HANDLE mapping = CreateFileMappingW(_code_object_file_descriptor, nullptr,
                                      PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }

  void *base = MapViewOfFile(mapping, FILE_MAP_READ,
                             static_cast<DWORD>(aligned_offset >> 32),
                             static_cast<DWORD>(aligned_offset & 0xFFFFFFFFu),
                             adjusted_size);
  if (base == nullptr) {
    CloseHandle(mapping);
    return HSA_STATUS_ERROR_INVALID_FILE;
  }

  map_base = base;
  map_handle = mapping;
  code_object_memory = reinterpret_cast<unsigned char*>(base) + slop;
  code_object_size = _code_object_size;
  is_mmap = true;

  uri = GetUriFromFile(_code_object_file_descriptor, _code_object_offset,
                       _code_object_size, is_complete_file, code_object_memory);
#endif  // !defined(_WIN32) && !defined(_WIN64)

  return HSA_STATUS_SUCCESS;
}

hsa_status_t CodeObjectReaderImpl::SetMemory(
    const void *_code_object_memory,
    size_t _code_object_size) {
  assert(!code_object_memory && "Code object reader wrapper is already set");

  if (!_code_object_memory || _code_object_size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  code_object_memory = _code_object_memory;
  code_object_size = _code_object_size;

  bool loader_enable_mmap_uri = core::Runtime::runtime_singleton_->flag().loader_enable_mmap_uri();
  if (loader_enable_mmap_uri) {
    uri = GetUriFromMemoryInMmapedFile(_code_object_memory, _code_object_size);
  } else {
    uri = GetUriFromMemoryInExecutableFile(_code_object_memory, _code_object_size);
  }

  return HSA_STATUS_SUCCESS;
}

}  // namespace loader
}  // namespace hsa
}  // namespace amd
}  // namespace rocr
