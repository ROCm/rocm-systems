// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vfio.h
/// @brief Public C API for serving a rocjitsu GPU to a VMM over vfio-user.

#ifndef ROCJITSU_VMM_RJ_VFIO_H_
#define ROCJITSU_VMM_RJ_VFIO_H_

#include "rocjitsu/base/rj_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup vmm
/// @{

/// @brief Serve a PCI function on @p socket_path until the process is
/// signalled.
///
/// @details A VMM such as QEMU connects to the socket and presents the function
/// to its guest as a real PCI device. Which GPU is presented comes from the
/// config, so a different part is a different config file.
///
/// Exported for callers outside C++. The implementation takes `std::string`
/// and is therefore unreachable by name from any other language; this is the
/// same function, with a C signature, and it is what the CLI resolves.
///
/// @note Present only in a build configured with `-DROCJITSU_ENABLE_VFIO=ON`.
/// The vfio-user transport pulls in libvfio-user, which is not a dependency
/// rocjitsu takes by default, so this symbol is absent from an ordinary
/// `librocjitsu.so`. This header is deliberately not in the `rocjitsu.h`
/// umbrella for that reason: a caller asks for it, and a caller resolving it
/// dynamically must be ready for it not to be there.
///
/// @warning Blocks for the lifetime of the server, and handles signals while it
/// does: `SIGINT` and `SIGTERM` stop it, `SIGUSR1` delivers an interrupt to the
/// emulated device. It blocks those signals on the calling thread and consumes
/// them with `sigtimedwait`, so a caller that has installed its own handlers
/// for them will not see them for as long as this runs. Call it from a process
/// that has not, or from one that has removed them first.
///
/// @param[in] config_path Simulation config describing the GPU to present.
/// @param[in] socket_path Filesystem path of the AF_UNIX socket to listen on.
/// @param[in] ready_fd Descriptor written to once the socket is accepting, so a
/// launcher can wait for readiness rather than poll for the socket. Negative to
/// ask for no notification, which is what a caller that does not supervise the
/// server passes.
/// @returns A process exit status: zero on an orderly shutdown, nonzero if the
/// config could not be parsed, the device could not be built, or the transport
/// failed.
RJ_API_EXPORT int rj_run_vfio_server(const char *config_path, const char *socket_path,
                                     int ready_fd);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_VMM_RJ_VFIO_H_
