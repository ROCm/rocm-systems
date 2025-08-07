////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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
#include "core/inc/runtime.h"

#include <assert.h>
#include <link.h>
#include <linux/limits.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>

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

std::string EncodePathname(const char *file_path) {
  std::ostringstream ss;
  unsigned char c;

  ss.fill('0');
  ss << "file://";

  while ((c = *file_path++) != '\0') {
    if (isalnum(c) || c == '/' || c == '-' ||
        c == '_' || c == '.' || c == '~') {
      ss << c;
    } else {
      ss << std::uppercase;
      ss << '%' << std::hex << std::setw(2) << static_cast<int>(c);
      ss << std::nouppercase;
    }
  }

  return ss.str();
}

std::string GetUriFromMemoryAddress(const void *memory, size_t size) {
  pid_t pid = getpid();
  std::ostringstream uri_stream;
  uri_stream << "memory://" << pid
             << "#offset=0x" << std::hex << (uintptr_t)memory << std::dec
             << "&size=" << size;
  return uri_stream.str();
}

std::string GetUriFromMemoryInExecutableFile(const void *memory, size_t size) {
#if !defined(_WIN32) && !defined(_WIN64)
  uintptr_t address = reinterpret_cast<uintptr_t>(memory);
  struct callback_data_s {
    ElfW(Addr) address;
    size_t callback_num;
    const char *file_path;
    size_t file_offset;
  } callback_data{address, 0, nullptr, 0};

  // Iterate the loaded shared objects program headers to see if the ELF binary
  // is allocated in a mapped file.
  if (dl_iterate_phdr([](struct dl_phdr_info *info, size_t size, void *ptr) -> int {
    struct callback_data_s *callback_data = (struct callback_data_s *) ptr;
    const ElfW(Addr) elf_address = callback_data->address - info->dlpi_addr;

    int n = info->dlpi_phnum;
    while (--n >= 0) {
      // Check if lib name is not empty and its not a "vdso.so" lib,
      // The vDSO is a special shared object file that is built into
      // the Linux kernel. It is not a regular shared library and thus
      // does not have all the properties of regular shared libraries.
      // The way the vDSO is loaded and organized in memory is different
      // from regular shared libraries and it's not guaranteed that it
      // will have a specific segment or section. Hence its skipped.
      if (info->dlpi_name[0] != '\0'
          && std::string(info->dlpi_name).find("vdso.so") != std::string::npos) {
        continue;
      }

      if (info->dlpi_phdr[n].p_type == PT_LOAD
          && elf_address - info->dlpi_phdr[n].p_vaddr >= 0
          && elf_address - info->dlpi_phdr[n].p_vaddr < info->dlpi_phdr[n].p_memsz) {
        // The first callback is always the program executable.
        if (!info->dlpi_name[0] && callback_data->callback_num == 0) {
          static char argv0[PATH_MAX] = {0};
          if (!argv0[0] && readlink("/proc/self/exe", argv0, sizeof(argv0)) == -1)
            return 0;
          callback_data->file_path = argv0;
        } else {
          callback_data->file_path = info->dlpi_name;
        }

        callback_data->file_offset =
            elf_address - info->dlpi_phdr[n].p_vaddr + info->dlpi_phdr[n].p_offset;
        return 1;
      }
    }

    ++callback_data->callback_num;
    return 0;
  }, &callback_data)) {
    if (!callback_data.file_path || callback_data.file_path[0] == '\0') {
      return GetUriFromMemoryAddress(memory, size);
    }

    std::ostringstream uri_stream;
    uri_stream << EncodePathname(callback_data.file_path);
    uri_stream << "#offset=" << callback_data.file_offset;
    uri_stream << "&size=" << size;
    return uri_stream.str();
  }
#endif  // !defined(_WIN32) && !defined(_WIN64)
  return GetUriFromMemoryAddress(memory, size);
}

std::string GetUriFromMemoryInMmapedFile(const void *memory, size_t size) {
#if !defined(_WIN32) && !defined(_WIN64)
  std::ifstream proc_maps;
  proc_maps.open("/proc/self/maps", std::ifstream::in);
  if (!proc_maps.is_open() || !proc_maps.good()) {
    return GetUriFromMemoryAddress(memory, size);
  }

  std::string line;
  while (std::getline(proc_maps, line)) {
    std::stringstream tokens(line);

    uintptr_t low_address, high_address;
    char dash;
    tokens >> std::hex >> low_address >> std::dec
           >> dash
           >> std::hex >> high_address >> std::dec;
    if (dash != '-') {
      continue;
    }

    uintptr_t address = reinterpret_cast<uintptr_t>(memory);
    if (!(address >= low_address && (address + size) <= high_address)) {
      continue;
    }

    std::string permissions, device, uri_file_path;
    size_t offset;
    uint64_t inode;
    tokens >> permissions
           >> std::hex >> offset >> std::dec
           >> device
           >> inode
           >> uri_file_path;

    if (inode == 0 || uri_file_path.empty()) {
      return GetUriFromMemoryAddress(memory, size);
    }

    size_t uri_offset = offset + address - low_address;

    bool is_complete_file = false;
    if (uri_offset == 0) {
      std::ifstream uri_file(uri_file_path, std::ios::binary);
      if (uri_file) {
        uri_file.seekg(0, std::ios::end);
        is_complete_file = uri_file.tellg() == size;
      }
    }

    std::ostringstream uri_stream;
    uri_stream << EncodePathname(uri_file_path.c_str());
    if (!is_complete_file) {
      uri_stream << "#offset=" << uri_offset;
      uri_stream << "&size=" << size;
    }
    return uri_stream.str();
  }
#endif  // !defined(_WIN32) && !defined(_WIN64)
  return GetUriFromMemoryAddress(memory, size);
}

std::string GetUriFromFile(int file_descriptor, size_t offset, size_t size,
    bool is_complete_file, const void *memory) {
#if !defined(_WIN32) && !defined(_WIN64)
  std::ostringstream proc_fd_path;
  proc_fd_path << "/proc/self/fd/" << file_descriptor;

  char uri_file_path[PATH_MAX];
  memset(uri_file_path, 0, PATH_MAX);

  if (readlink(proc_fd_path.str().c_str(), uri_file_path, PATH_MAX) == -1) {
    return GetUriFromMemoryAddress(memory, size);
  }

  if (uri_file_path[0] == '\0') {
    return GetUriFromMemoryAddress(memory, size);
  }

  std::ostringstream uri_stream;
  uri_stream << EncodePathname(uri_file_path);
  if (!is_complete_file) {
    uri_stream << "#offset=" << offset;
    uri_stream << "&size=" << size;
  }
  return uri_stream.str();
#else
  return GetUriFromMemoryAddress(memory, size);
#endif  // !defined(_WIN32) && !defined(_WIN64)
}

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
    delete [] code_object_memory;
#endif  // !defined(_WIN32) && !defined(_WIN64)
  }
}

hsa_status_t CodeObjectReaderImpl::SetFile(
    hsa_file_t _code_object_file_descriptor,
    size_t _code_object_offset,
    size_t _code_object_size) {
  assert(!code_object_memory && "Code object reader wrapper is already set");

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

#if !defined(_WIN32) && !defined(_WIN64)
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
#else
  if (__lseek__(_code_object_file_descriptor, 0, SEEK_SET) == (off_t)-1) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }

  std::unique_ptr<unsigned char> memory(new unsigned char[_code_object_size]);
  if (!memory) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (__read__(_code_object_file_descriptor, mmap_memory,
                _code_object_size) != _code_object_size) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }
  mmap_memory = memory.release();
  mmap_size = _code_object_size;
  code_object_memory = memory;
  code_object_size = _code_object_size;
#endif  // !defined(_WIN32) && !defined(_WIN64)

  uri = GetUriFromFile(_code_object_file_descriptor, _code_object_offset,
                        _code_object_size, is_complete_file, code_object_memory);

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
