// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vfio.cpp
/// @brief C entry point for the vfio-user server.
///
/// Kept apart from vfio_server.cpp so the C surface is one file and the server
/// stays a C++ function with C++ arguments. The CLI that used to call
/// run_vfio_server() directly was itself C++ and linked this code into its own
/// binary; the CLI that replaced it is not, and resolves this symbol out of
/// librocjitsu.so instead.

#include "rocjitsu/vmm/rj_vfio.h"

#include <string>

#include "rocjitsu/vmm/vfu/vfio_server.h"

extern "C" int rj_run_vfio_server(const char *config_path, const char *socket_path) {
  // The old CLI checked these before dispatching, and there is no useful
  // server to start without them. Reported as the failure exit status this
  // returns everywhere else rather than a distinct code: the caller's job is
  // to pass a config and a socket, and it either did or it did not.
  if (config_path == nullptr || *config_path == '\0' || socket_path == nullptr ||
      *socket_path == '\0') {
    return 1;
  }
  return rocjitsu::run_vfio_server(std::string(config_path), std::string(socket_path));
}
