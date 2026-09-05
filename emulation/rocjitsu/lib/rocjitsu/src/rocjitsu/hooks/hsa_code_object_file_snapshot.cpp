// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/hsa_code_object_file_snapshot.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <new>

#include <sys/stat.h>
#include <unistd.h>

namespace rocjitsu {

CodeObjectFileSnapshot snapshot_code_object_file_range(int file, size_t offset,
                                                       size_t size) noexcept {
  struct stat status{};
  if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 || size == 0 ||
      static_cast<uintmax_t>(status.st_size) > std::numeric_limits<size_t>::max())
    return {};

  const size_t file_size = static_cast<size_t>(status.st_size);
  if (offset > file_size || size > file_size - offset ||
      offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
    return {};

  std::shared_ptr<std::vector<uint8_t>> bytes;
  try {
    bytes = std::make_shared<std::vector<uint8_t>>(size);
  } catch (const std::bad_alloc &) {
    return {};
  }

  size_t done = 0;
  while (done < size) {
    const size_t absolute_offset = offset + done;
    if (absolute_offset > static_cast<size_t>(std::numeric_limits<off_t>::max()))
      return {};
    const size_t chunk =
        std::min(size - done, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t read =
        pread(file, bytes->data() + done, chunk, static_cast<off_t>(absolute_offset));
    if (read > 0) {
      done += static_cast<size_t>(read);
      continue;
    }
    if (read < 0 && errno == EINTR)
      continue;
    return {};
  }
  return bytes;
}

CodeObjectFileSnapshot snapshot_code_object_file(int file) noexcept {
  struct stat status{};
  if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
      static_cast<uintmax_t>(status.st_size) > std::numeric_limits<size_t>::max())
    return {};
  return snapshot_code_object_file_range(file, 0, static_cast<size_t>(status.st_size));
}

} // namespace rocjitsu
