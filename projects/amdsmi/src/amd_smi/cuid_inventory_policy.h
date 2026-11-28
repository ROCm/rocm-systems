// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_SRC_CUID_INVENTORY_POLICY_H_
#define AMDSMI_SRC_CUID_INVENTORY_POLICY_H_

#include <fcntl.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <string>

namespace amdsmi_cuid_detail {

static inline bool may_be_partition(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return errno != ENOENT;
  }
  char value[16] = {};
  const ssize_t length = read(fd, value, sizeof(value) - 1);
  close(fd);
  return length < 0 || std::string(value, static_cast<size_t>(length)) != "SPX\n";
}

// strcmp with digit runs compared as numbers. A CPU package is named by its
// lowest logical CPU, /sys/devices/system/cpu/cpu<N>, and "cpu96" has to come
// before "cpu192" for `node --cuid` to number the packages in socket order.
static inline int path_compare(const char* a, const char* b) {
  while (*a != '\0' && *b != '\0') {
    if (isdigit(static_cast<unsigned char>(*a)) && isdigit(static_cast<unsigned char>(*b))) {
      char* a_end = nullptr;
      char* b_end = nullptr;
      const unsigned long long x = strtoull(a, &a_end, 10);
      const unsigned long long y = strtoull(b, &b_end, 10);
      if (x != y) return x < y ? -1 : 1;
      a = a_end;
      b = b_end;
    } else {
      if (*a != *b) return static_cast<unsigned char>(*a) < static_cast<unsigned char>(*b) ? -1 : 1;
      ++a;
      ++b;
    }
  }
  return static_cast<unsigned char>(*a) - static_cast<unsigned char>(*b);
}

}  // namespace amdsmi_cuid_detail

#endif  // AMDSMI_SRC_CUID_INVENTORY_POLICY_H_
