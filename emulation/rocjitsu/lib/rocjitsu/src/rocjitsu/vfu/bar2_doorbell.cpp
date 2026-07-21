// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar2_doorbell.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace rocjitsu::vfu {

Bar2Doorbell::Bar2Doorbell(uint64_t size, RingCallback cb)
    // SealedMemfd: create, truncate, init every byte to 0xFF (the "no packet"
    // sentinel rocjitsu's CP polling thread uses), then seal.
    : memfd_("rocjitsu_doorbell", static_cast<size_t>(size), 0xFF),
      ring_cb_(std::move(cb)) {}

ssize_t Bar2Doorbell::handle_access(char *buf, size_t count, long offset,
                                     bool is_write) {
  if (static_cast<uint64_t>(offset) + count > size()) {
    errno = ERANGE;
    return -1;
  }

  if (!is_write) {
    // Guest reads the doorbell page (unusual but harmless — return zeros).
    std::memset(buf, 0, count);
    return static_cast<ssize_t>(count);
  }

  // Guest wrote a doorbell slot (4 or 8 bytes). Collect as 64-bit and dispatch.
  uint64_t value = 0;
  std::memcpy(&value, buf, count < 8 ? count : 8);

  // Update the shadow memfd so a subsequent guest mmap read is coherent.
  const long page_size = sysconf(_SC_PAGE_SIZE);
  const off_t page_aligned_off =
      static_cast<off_t>(offset & ~(static_cast<long>(page_size) - 1));
  memfd_.with_mapping(static_cast<uint64_t>(page_aligned_off), count,
                      PROT_WRITE, [&](void *p) {
                        std::memcpy(static_cast<char *>(p) +
                                        (offset - page_aligned_off),
                                    buf, count);
                      });

  if (ring_cb_)
    ring_cb_(static_cast<uint32_t>(offset), value);

  return static_cast<ssize_t>(count);
}

} // namespace rocjitsu::vfu
