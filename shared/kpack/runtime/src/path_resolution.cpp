// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>

#include "rocm_kpack/kpack.h"

#if defined(__linux__)
#include <fstream>
#include <sstream>
#include <string>
#elif defined(_WIN32)
// Windows headers would go here
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>
#endif

extern "C" {

kpack_error_t kpack_discover_binary_path(const void* address_in_binary,
                                         char* path_out, size_t path_out_size,
                                         size_t* offset_out) {
  // Validate required arguments
  if (address_in_binary == nullptr || path_out == nullptr ||
      path_out_size == 0) {
    return KPACK_ERROR_INVALID_ARGUMENT;
  }

#if defined(__linux__)
  // Linux implementation: parse /proc/self/maps
  //
  // Format of /proc/self/maps:
  // address           perms offset  dev   inode   pathname
  // 7f1234567000-7f1234568000 r-xp 00001000 08:01 12345 /path/to/lib.so

  std::ifstream maps("/proc/self/maps");
  if (!maps.is_open()) {
    return KPACK_ERROR_PATH_DISCOVERY_FAILED;
  }

  uintptr_t target_addr = reinterpret_cast<uintptr_t>(address_in_binary);
  std::string line;

  while (std::getline(maps, line)) {
    // Parse address range
    uintptr_t low_addr = 0;
    uintptr_t high_addr = 0;
    size_t file_offset = 0;

    // Find the dash separating low-high addresses
    size_t dash_pos = line.find('-');
    if (dash_pos == std::string::npos) {
      continue;
    }

    // Parse low address (hex)
    try {
      low_addr = std::stoull(line.substr(0, dash_pos), nullptr, 16);
    } catch (...) {
      continue;
    }

    // Find space after high address
    size_t space_pos = line.find(' ', dash_pos + 1);
    if (space_pos == std::string::npos) {
      continue;
    }

    // Parse high address (hex)
    try {
      high_addr = std::stoull(
          line.substr(dash_pos + 1, space_pos - dash_pos - 1), nullptr, 16);
    } catch (...) {
      continue;
    }

    // Check if target address is in this range
    if (target_addr < low_addr || target_addr >= high_addr) {
      continue;
    }

    // Found the mapping! Now parse the rest of the line.
    // Skip permissions field (e.g., "r-xp")
    size_t next_space = line.find(' ', space_pos + 1);
    if (next_space == std::string::npos) {
      continue;
    }

    // Parse file offset (hex)
    size_t offset_end = line.find(' ', next_space + 1);
    if (offset_end == std::string::npos) {
      continue;
    }

    try {
      file_offset =
          std::stoull(line.substr(next_space + 1, offset_end - next_space - 1),
                      nullptr, 16);
    } catch (...) {
      file_offset = 0;
    }

    // Skip dev and inode fields, find pathname
    // Format: "08:01 12345   /path/to/file"
    // Skip dev field
    size_t dev_end = line.find(' ', offset_end + 1);
    if (dev_end == std::string::npos) {
      continue;
    }

    // Skip inode field (and any whitespace after it)
    size_t path_start = line.find_first_not_of(" \t", dev_end + 1);
    if (path_start == std::string::npos) {
      // Anonymous mapping (no file path)
      continue;
    }

    // Skip inode number
    size_t inode_end = line.find_first_of(" \t", path_start);
    if (inode_end == std::string::npos) {
      // Line ends with inode, no pathname
      continue;
    }

    // Find start of pathname
    path_start = line.find_first_not_of(" \t", inode_end);
    if (path_start == std::string::npos) {
      // No pathname (anonymous mapping)
      continue;
    }

    // Extract pathname (rest of line, trimmed)
    std::string pathname = line.substr(path_start);

    // Skip special mappings like [heap], [stack], [vdso]
    if (!pathname.empty() && pathname[0] == '[') {
      continue;
    }

    // Copy path to output buffer
    if (pathname.size() >= path_out_size) {
      // Path too long for buffer
      return KPACK_ERROR_INVALID_ARGUMENT;
    }

    std::memcpy(path_out, pathname.c_str(), pathname.size());
    path_out[pathname.size()] = '\0';

    // Calculate offset within file
    if (offset_out != nullptr) {
      *offset_out = file_offset + (target_addr - low_addr);
    }

    return KPACK_SUCCESS;
  }

  // Address not found in any mapping
  return KPACK_ERROR_PATH_DISCOVERY_FAILED;

#elif defined(_WIN32)
  // Windows implementation: use GetModuleHandleEx + GetModuleFileName
  //
  // TODO: Implement Windows support
  // Pattern:
  //   HMODULE hm = NULL;
  //   if (GetModuleHandleExA(
  //       GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
  //       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
  //       (LPCSTR)address_in_binary, &hm)) {
  //     char path[MAX_PATH];
  //     if (GetModuleFileNameA(hm, path, sizeof(path))) {
  //       // Copy to path_out
  //       // offset_out = address_in_binary - hm
  //     }
  //   }
  (void)address_in_binary;
  (void)path_out;
  (void)path_out_size;
  (void)offset_out;
  return KPACK_ERROR_NOT_IMPLEMENTED;

#else
  // Unsupported platform
  (void)address_in_binary;
  (void)path_out;
  (void)path_out_size;
  (void)offset_out;
  return KPACK_ERROR_NOT_IMPLEMENTED;
#endif
}

}  // extern "C"
