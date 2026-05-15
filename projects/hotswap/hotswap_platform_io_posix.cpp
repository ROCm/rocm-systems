//===- hotswap_platform_io_posix.cpp - POSIX file I/O shims ---------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_platform_io.hpp"
#include <algorithm>
#include <cerrno>
#include <limits>
#include <unistd.h>

namespace rocr::hotswap::platform_io {

bool read_all(hsa_file_t file, void *buf, size_t count) {
  constexpr size_t kMaxReadChunk =
      static_cast<size_t>(std::numeric_limits<ssize_t>::max());
  size_t total = 0;
  while (total < count) {
    const size_t chunk_size = std::min(count - total, kMaxReadChunk);
    auto *dst = static_cast<uint8_t *>(buf) + total;
    const ssize_t n = read(file, dst, chunk_size);
    if (n > 0) {
      total += static_cast<size_t>(n);
    } else if (n == 0) {
      break;
    } else if (errno != EINTR) {
      return false;
    }
  }
  return total == count;
}

bool get_file_pos(hsa_file_t file, int whence, file_pos_t *pos) {
  const auto raw_pos = lseek(file, 0, whence);
  if (raw_pos < 0) {
    return false;
  }
  *pos = static_cast<file_pos_t>(raw_pos);
  return true;
}

bool get_file_bounds(hsa_file_t file, file_pos_t *saved_pos,
                     file_pos_t *file_size) {
  if (!get_file_pos(file, SEEK_CUR, saved_pos) ||
      !get_file_pos(file, SEEK_END, file_size) || *file_size == 0) {
    return false;
  }
  return true;
}

bool set_file_pos(hsa_file_t file, file_pos_t pos) {
  using pos_type = off_t;
  if (pos > static_cast<file_pos_t>(std::numeric_limits<pos_type>::max())) {
    return false;
  }
  return lseek(file, static_cast<pos_type>(pos), SEEK_SET) >= 0;
}

void restore_file_pos(hsa_file_t file, file_pos_t pos) {
  (void)set_file_pos(file, pos);
}

} // namespace rocr::hotswap::platform_io
