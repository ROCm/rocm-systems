//===- hotswap_platform_io_win.cpp - Windows file I/O shims ---------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_platform_io.hpp"
#include <algorithm>
#include <cerrno>
#include <io.h>
#include <limits>

namespace rocr::hotswap::platform_io {

bool read_all(hsa_file_t file, void *buf, size_t count) {
  constexpr size_t kMaxReadChunk =
      static_cast<size_t>(std::numeric_limits<unsigned int>::max());
  size_t total = 0;
  while (total < count) {
    const size_t chunk_size = std::min(count - total, kMaxReadChunk);
    auto *dst = static_cast<uint8_t *>(buf) + total;
    const int n = _read(file, dst, static_cast<unsigned int>(chunk_size));
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
  const auto raw_pos = _lseeki64(file, 0, whence);
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
  if (pos > static_cast<file_pos_t>(std::numeric_limits<long long>::max())) {
    return false;
  }
  return _lseeki64(file, static_cast<long long>(pos), SEEK_SET) >= 0;
}

void restore_file_pos(hsa_file_t file, file_pos_t pos) {
  (void)set_file_pos(file, pos);
}

} // namespace rocr::hotswap::platform_io
