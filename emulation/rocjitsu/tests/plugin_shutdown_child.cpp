// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_shutdown_child.cpp
/// @brief Opens the interposed KFD and exits while that descriptor is still live.

#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char **argv) {
  const int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    std::cerr << "plugin_shutdown_child: open /dev/kfd failed\n";
    return 2;
  }
  // SIGKILL skips every finalizer, so on_shutdown must not run.
  if (argc > 1 && std::strcmp(argv[1], "--kill") == 0)
    raise(SIGKILL);
  // Leave fd open. A normal exit() still runs the interposer finalizer, and the
  // driver is not idle because this reference was never closed.
  return 0;
}
