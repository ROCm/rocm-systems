// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_server.h
/// @brief Entry point that serves a PCI function to a VMM until interrupted.
///
/// @details This is the body of the CLI's vfio-user mode, kept out of the CLI so
/// the front end stays a thin argument parser and so nothing outside this
/// library links libvfio-user.

#pragma once

#include <string>

namespace rocjitsu {

/// @brief What the serving loop does about a signal it woke on.
enum class ServerSignalAction {
  KeepServing,      ///< Nothing arrived, or nothing this server acts on.
  Stop,             ///< Shut down and exit.
  DeliverInterrupt, ///< Ask the device to put an entry in its interrupt ring.
};

/// @brief Map a signal the server waits on to what it should do about it.
/// @details Exposed because the loop that consumes it needs a process, a socket
/// and a connected client before it runs a single line, so nothing that tests
/// the loop tests only this. A signal routed to the wrong arm, or an arm
/// deleted, is otherwise invisible.
/// @param[in] signal Signal number, or negative when the wait timed out.
/// @returns What the loop should do next.
[[nodiscard]] ServerSignalAction action_for_signal(int signal);

/// @brief Serve a PCI function on @p socket_path until the process is signalled.
/// @param[in] config_path Simulation config describing the GPU to present.
/// @param[in] socket_path Filesystem path of the AF_UNIX socket to listen on.
/// @returns A process exit status: zero on an orderly shutdown.
/// @details Blocks. A VMM such as QEMU connects to the socket and presents the
/// function to its guest as a real PCI device. Which GPU is presented comes from
/// the config, so a different part is a different config file.
int run_vfio_server(const std::string &config_path, const std::string &socket_path);

} // namespace rocjitsu
