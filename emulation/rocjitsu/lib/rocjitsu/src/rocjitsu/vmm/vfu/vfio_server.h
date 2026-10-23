// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_server.h
/// @brief Entry points that serve local PCI functions to a VMM until interrupted.
///
/// @details This is the body of the CLI's vfio-user mode, kept out of the CLI so
/// the front end stays a thin argument parser and so nothing outside this
/// library links libvfio-user.

#pragma once

#include <string>
#include <vector>

namespace rocjitsu {

/// @brief What the serving loop does about a signal it woke on.
enum class ServerSignalAction {
  KeepServing,      ///< Nothing arrived, or nothing this server acts on.
  Stop,             ///< Shut down and exit.
  DeliverInterrupt, ///< Ask each served device to put an entry in its interrupt ring.
};

/// @brief Map a signal the server waits on to what it should do about it.
/// @details Exposed because the loop that consumes it needs a process, a socket
/// and a connected client before it runs a single line, so nothing that tests
/// the loop tests only this. A signal routed to the wrong arm, or an arm
/// deleted, is otherwise invisible.
/// @param[in] signal Signal number, or negative when the wait timed out.
/// @returns What the loop should do next.
[[nodiscard]] ServerSignalAction action_for_signal(int signal);

/// @brief Serve one PCI function on @p socket_path until the process is signalled.
/// @param[in] config_path Simulation config describing the GPU to present.
/// @param[in] socket_path Filesystem path of the AF_UNIX socket to listen on.
/// @param[in] ready_fd Optional pipe descriptor notified after the socket is ready.
/// @returns A process exit status: zero on an orderly shutdown.
/// @details Blocks. A VMM such as QEMU connects to the socket and presents the
/// function to its guest as a real PCI device. The config must contain one
/// local GPU; use the socket-list overload for a multi-GPU node.
int run_vfio_server(const std::string &config_path, const std::string &socket_path,
                    int ready_fd = -1);

/// @brief Serve every local GPU in one node through its own PCI function.
/// @param[in] config_path Simulation config describing the local GPU node.
/// @param[in] socket_paths One AF_UNIX filesystem path per local GPU, in local
/// GPU ordinal order. Paths must be nonempty, unique after lexical normalization,
/// and short enough for sockaddr_un::sun_path.
/// @param[in] ready_fd Optional pipe descriptor notified once every socket is ready.
/// @returns A process exit status: zero on an orderly shutdown.
/// @details Blocks. All functions share one simulation engine and one process
/// lifetime. A permanently stopped function fails the multi-function session.
/// SIGUSR1 requests an interrupt on every function in the group.
int run_vfio_server(const std::string &config_path, const std::vector<std::string> &socket_paths,
                    int ready_fd = -1);

/// @brief Run the server and request a simulation failure after readiness.
/// @details Test-only entry point for proving that an engine failure after the
/// socket becomes usable terminates the server and propagates its status.
int run_vfio_server_with_engine_exit_for_test(const std::string &config_path,
                                              const std::string &socket_path, int ready_fd,
                                              int exit_code);

/// @brief Request a simulation failure after all node sockets are ready.
/// @details Test-only equivalent of the socket-list server overload.
int run_vfio_server_with_engine_exit_for_test(const std::string &config_path,
                                              const std::vector<std::string> &socket_paths,
                                              int ready_fd, int exit_code);

} // namespace rocjitsu
