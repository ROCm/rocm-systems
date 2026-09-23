// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vfio_server.h
/// @brief Entry point that serves a PCI function to a VMM until interrupted.
///
/// @details This is the body of the CLI's vfio-user mode, kept here so the front
/// end stays a thin argument parser and so nothing outside this library links
/// libvfio-user. Built only when `ROCJITSU_ENABLE_VFIO` is on, which is also
/// what puts it into `librocjitsu.so` for the CLI to resolve.

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
///
/// @note Reached from the CLI through `rj_run_vfio_server` in
/// `rocjitsu/vmm/rj_vfio.h`, which is the C-callable spelling of this: the CLI
/// is not C++ and cannot name a function taking `std::string`.
/// @param[in] config_path Simulation config describing the GPU to present.
/// @param[in] socket_path Filesystem path of the AF_UNIX socket to listen on.
/// @param[in] ready_fd Optional pipe descriptor notified after the socket is ready.
/// @returns A process exit status: zero on an orderly shutdown.
/// @details Blocks. A VMM such as QEMU connects to the socket and presents the
/// function to its guest as a real PCI device. Which GPU is presented comes from
/// the config, so a different part is a different config file.
int run_vfio_server(const std::string &config_path, const std::string &socket_path,
                    int ready_fd = -1);

/// @brief Run the server and request a simulation failure after readiness.
/// @details Test-only entry point for proving that an engine failure after the
/// socket becomes usable terminates the server and propagates its status.
int run_vfio_server_with_engine_exit_for_test(const std::string &config_path,
                                              const std::string &socket_path, int ready_fd,
                                              int exit_code);

/// @brief Make the next @ref run_vfio_server call throw, once.
/// @details Test-only, and the only way to exercise the exception boundary in
/// `rj_run_vfio_server`: that boundary exists because the server can throw and
/// its caller is Rust, which cannot receive an exception, but nothing a test
/// can pass as an argument reaches a throwing path. Without a seam the boundary
/// is asserted by reading it, which is how the same guarantee was wrong once
/// already -- an earlier version of it allocated inside the handler.
///
/// Consumed by the call it arms, so a test that arms it and then fails before
/// calling does not leave the next one poisoned.
void throw_from_next_vfio_server_for_test();

} // namespace rocjitsu
