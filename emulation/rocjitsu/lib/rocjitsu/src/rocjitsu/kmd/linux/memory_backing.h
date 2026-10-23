// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// @brief Shared backing descriptors, either native memfds or opaque socket tokens.
/// @details Descriptors retain their storage across dup, fork, and SCM_RIGHTS.
/// Tokens additionally depend on their creating keeper; native memfds do not.
/// Keeper loss invalidates tokens; new
/// mappings and reopens report ESTALE. Existing mapped views remain usable.
/// Callers own returned descriptors and must use these adapters for mapping and
/// metadata; tokens do not support file I/O. Failures use errno. Ordinary file
/// descriptors retain host behavior where an operation supports both forms.

#include <cstddef>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>

namespace rocjitsu {

inline constexpr int kMemoryKeeperListenerFd = 3;
inline constexpr int kMemoryKeeperLeaseFd = 4;
inline constexpr int kMemoryKeeperStartupFd = 5;
inline constexpr int kMemoryKeeperFirstUnreservedFd = kMemoryKeeperStartupFd + 1;

/// @brief Create fixed-size shared RAM, independent of RLIMIT_FSIZE.
/// @details Preallocation is best effort for native memfds; token storage stays
/// sparse. RJ_TEST_FORCE_TOKEN_BACKING=1 selects tokens for integration testing
/// without affecting ordinary metadata files or changing application limits.
int create_memory_backing(size_t size, bool preallocate = false);

/// @brief Populate a newly created metadata file using the keeper's inherited hard limit.
/// @details The caller owns this unpublished regular file; its offset stays at zero.
/// Unlike GPU backing, metadata files must fit the keeper's inherited hard file limit.
int populate_backing_file(int fd, const void *data, size_t size);

/// @brief Entry point for the private memory keeper executable.
[[noreturn]] void run_memory_keeper(int listener, int lease, int startup, pid_t owner);

/// @brief Recognize a shared-RAM descriptor without consulting process-local state.
bool is_memory_backing_token(int fd);

/// @brief Map shared RAM, or pass a non-backing descriptor to the host mmap.
/// @details The caller must hold the host mapping lock when replacing mappings
/// visible to the simulator. Ordinary munmap/mprotect/mremap work on the result.
/// Token attachments enforce current System V IPC permissions; retaining a token
/// across an effective UID change does not grant new mapping authority (EACCES).
/// A token whose keeper has exited fails with ESTALE.
void *map_memory_backing(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

/// @brief Inspect RAM size and identity, or pass through for ordinary files.
int stat_memory_backing(int fd, struct stat *result);
/// @brief Overlay token metadata on an already obtained host stat result.
/// @details Returns false for ordinary descriptors; callers can skip non-sockets.
bool stat_memory_backing_token(int fd, struct stat *result);

/// @brief Apply dma-buf seek semantics to a token, returning false for other fds.
/// @details The result is -1 with errno set for unsupported token seek operations.
bool seek_memory_backing_token(int fd, off_t offset, int whence, off_t &result);

/// @brief Query backing metadata or forward a descriptor command to the host.
/// @details Tokens synthesize F_GETFL and F_GET_SEALS, reject F_ADD_SEALS with
/// EPERM, and forward other commands to their underlying socket descriptor.
int fcntl_memory_backing(int fd, int command, uintptr_t argument = 0);

/// @brief Reopen a backing with an independently retained access mode.
int reopen_memory_backing(int fd, int access_mode);

/// @brief Verify that existing backing covers a requested mapping, without growing a file.
int ensure_memory_backing_size(int fd, size_t size);

} // namespace rocjitsu
