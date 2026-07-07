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

#include "core/inc/code_object_uri.hpp"

#if defined(__linux__)
#include <link.h>
#include <linux/limits.h>
#include <unistd.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

std::string EncodePathname(const char *file_path) {
  std::ostringstream ss;
  unsigned char c;

  ss.fill('0');
  ss << "file://";

  while ((c = *file_path++) != '\0') {
    if (isalnum(c) || c == '/' || c == '-' ||
        c == '_' || c == '.' || c == '~' ||
	c == ':') {
      ss << c;
    } else {
      ss << std::uppercase;
      ss << '%' << std::hex << std::setw(2) << static_cast<int>(c);
      ss << std::nouppercase;
    }
  }

  return ss.str();
}

#if defined(_WIN32) || defined(_WIN64)
// Convert a Win32 wide-character path into the UTF-8, forward-slash form used
// in file:// URIs. Strips the \\?\ and \\?\UNC\ prefixes that
// GetFinalPathNameByHandleW / GetMappedFileNameW may emit, and prepends a
// leading '/' so the result composes into file:///C:/path/... (RFC 8089).
// Returns an empty string on conversion failure.
std::string NormalizeWindowsPathToUtf8(std::wstring wpath) {
  if (wpath.compare(0, 8, L"\\\\?\\UNC\\") == 0) {
    wpath = L"\\\\" + wpath.substr(8);  // \\?\UNC\srv\share -> \\srv\share
  } else if (wpath.compare(0, 4, L"\\\\?\\") == 0) {
    wpath.erase(0, 4);                  // \\?\C:\... -> C:\...
  }

  int u8len = WideCharToMultiByte(CP_UTF8, 0, wpath.data(),
                                  static_cast<int>(wpath.size()),
                                  nullptr, 0, nullptr, nullptr);
  if (u8len <= 0) {
    return {};
  }
  std::string path(static_cast<size_t>(u8len), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wpath.data(), static_cast<int>(wpath.size()),
                      path.data(), u8len, nullptr, nullptr);

  for (auto& c : path) {
    if (c == '\\') c = '/';
  }
  if (!path.empty() && path.front() != '/') {
    path.insert(path.begin(), '/');
  }
  return path;
}
#endif

}  // namespace

namespace rocr {
namespace amd {
namespace hsa {
namespace loader {

std::string GetUriFromMemoryAddress(const void *memory, size_t size) {
  int pid = getpid();
  std::ostringstream uri_stream;
  uri_stream << "memory://" << pid
             << "#offset=0x" << std::hex << (uintptr_t)memory << std::dec
             << "&size=" << size;
  return uri_stream.str();
}

std::string GetUriFromMemoryInExecutableFile(const void *memory, size_t size) {
#if defined(_WIN32) || defined(_WIN64)
  // Find the loaded module (EXE or DLL) whose image contains `memory`.
  // GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT avoids ref-counting the
  // module so we don't have to FreeLibrary it.
  HMODULE hModule = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(memory), &hModule) ||
      hModule == nullptr) {
    return GetUriFromMemoryAddress(memory, size);
  }

  // Walk the PE headers of the mapped image to translate the memory address
  // into a file offset. hModule == ImageBase, so:
  //   RVA          = memory - hModule
  //   file_offset  = RVA - section.VirtualAddress + section.PointerToRawData
  // for the section that covers the RVA.
  auto* base = reinterpret_cast<const BYTE*>(hModule);
  auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
    return GetUriFromMemoryAddress(memory, size);
  }
  auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE) {
    return GetUriFromMemoryAddress(memory, size);
  }
  uintptr_t rva = reinterpret_cast<const BYTE*>(memory) - base;
  const IMAGE_SECTION_HEADER* sections = IMAGE_FIRST_SECTION(nt);

  size_t file_offset = 0;
  bool found = false;
  for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
    const auto& s = sections[i];
    DWORD vsize = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
    if (rva >= s.VirtualAddress && rva < s.VirtualAddress + vsize) {
      // Sections with no backing on disk (e.g. .bss) can't yield a file URI.
      if (s.PointerToRawData == 0) break;
      file_offset = rva - s.VirtualAddress + s.PointerToRawData;
      found = true;
      break;
    }
  }
  if (!found) {
    return GetUriFromMemoryAddress(memory, size);
  }

  // Resolve the module's on-disk path. Grow the buffer if MAX_PATH overflows
  // (long paths or DLLs loaded via \\?\ prefixed names).
  std::wstring wpath(MAX_PATH, L'\0');
  for (;;) {
    DWORD n = GetModuleFileNameW(hModule, wpath.data(),
                                 static_cast<DWORD>(wpath.size()));
    if (n == 0) {
      return GetUriFromMemoryAddress(memory, size);
    }
    if (n < wpath.size()) {
      wpath.resize(n);
      break;
    }
    if (wpath.size() >= 32768) {
      return GetUriFromMemoryAddress(memory, size);
    }
    wpath.resize(wpath.size() * 2);
  }

  std::string path = NormalizeWindowsPathToUtf8(std::move(wpath));
  if (path.empty()) {
    return GetUriFromMemoryAddress(memory, size);
  }

  std::ostringstream uri_stream;
  uri_stream << EncodePathname(path.c_str())
             << "#offset=" << file_offset
             << "&size=" << size;
  return uri_stream.str();
#else
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

std::string GetUriFromFile(hsa_file_t file_descriptor, size_t offset, size_t size,
    bool is_complete_file, const void *memory) {
  std::string path;

#if defined(_WIN32) || defined(_WIN64)
  // Resolve the HANDLE to a filesystem path via GetFinalPathNameByHandleW.
  // First call sizes the buffer (return value includes the terminating NUL
  // when the supplied buffer is too small).
  DWORD wlen = GetFinalPathNameByHandleW(file_descriptor, nullptr, 0,
                                         FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  if (wlen == 0) {
    return GetUriFromMemoryAddress(memory, size);
  }
  std::wstring wpath(wlen, L'\0');
  DWORD got = GetFinalPathNameByHandleW(file_descriptor, wpath.data(), wlen,
                                        FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  if (got == 0 || got >= wlen) {
    return GetUriFromMemoryAddress(memory, size);
  }
  wpath.resize(got);  // drop the trailing NUL from std::wstring length

  path = NormalizeWindowsPathToUtf8(std::move(wpath));
  if (path.empty()) {
    return GetUriFromMemoryAddress(memory, size);
  }
#else
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
  path = uri_file_path;
#endif

  std::ostringstream uri_stream;
  uri_stream << EncodePathname(path.c_str());
  if (!is_complete_file) {
    uri_stream << "#offset=" << offset;
    uri_stream << "&size=" << size;
  }
  return uri_stream.str();
}

}  // namespace loader
}  // namespace hsa
}  // namespace amd
}  // namespace rocr
