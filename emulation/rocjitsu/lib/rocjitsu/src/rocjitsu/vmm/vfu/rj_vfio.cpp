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

#include <exception>
#include <string>

#include "rocjitsu/vmm/vfu/vfio_server.h"
#include "util/log.h"

extern "C" int rj_run_vfio_server(const char *config_path, const char *socket_path,
                                  int ready_fd) {
  // The old CLI checked these before dispatching, and there is no useful
  // server to start without them. Reported as the failure exit status this
  // returns everywhere else rather than a distinct code: the caller's job is
  // to pass a config and a socket, and it either did or it did not.
  if (config_path == nullptr || *config_path == '\0' || socket_path == nullptr ||
      *socket_path == '\0') {
    return 1;
  }
  // Nothing may leave this frame as an exception. The caller across it is Rust,
  // which declares this symbol on a non-unwind ABI, so unwinding into it does
  // not become an error the CLI can report -- it terminates the process, and
  // the message that comes out names neither the server nor the config. The
  // server has plenty to throw: the two strings below, the device host, the
  // serving thread, every std::format on the way. Converted here to the status
  // this entry point already uses for "it did not start", the same way
  // rj_vm_create converts a bad config.
  try {
    return rocjitsu::run_vfio_server(std::string(config_path), std::string(socket_path), ready_fd);
  } catch (const std::exception &error) {
    util::Logger::warn(std::string("vfu: the server failed to start: ") + error.what());
    return 1;
  } catch (...) {
    util::Logger::warn("vfu: the server failed to start");
    return 1;
  }
}
