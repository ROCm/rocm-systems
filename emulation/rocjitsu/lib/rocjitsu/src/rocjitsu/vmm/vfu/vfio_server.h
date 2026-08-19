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

/// @brief Serve a PCI function on @p socket_path until the process is signalled.
/// @param[in] socket_path Filesystem path of the AF_UNIX socket to listen on.
/// @returns A process exit status: zero on an orderly shutdown.
/// @details Blocks. A VMM such as QEMU connects to the socket and presents the
/// function to its guest as a real PCI device. The function served today is a
/// scratch device that exists to exercise the transport; the emulated GPU
/// replaces it once its model is wired in.
int run_vfio_server(const std::string &socket_path);

} // namespace rocjitsu
