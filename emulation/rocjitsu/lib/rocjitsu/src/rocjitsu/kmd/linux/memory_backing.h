// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_MEMORY_BACKING_H_
#define ROCJITSU_KMD_LINUX_MEMORY_BACKING_H_

/// @file
/// @brief Shared backing descriptors, either native memfds or opaque socket tokens.
/// @details Descriptors retain their storage across dup, fork, and SCM_RIGHTS.
/// Callers own returned descriptors and must use these adapters for mapping and
/// metadata; tokens do not support file I/O. Failures use errno. Ordinary file
/// descriptors retain host behavior where an operation supports both forms.

#include <cstddef>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>

namespace rocjitsu::memory_backing {

inline constexpr int kListenerFd = 3;
inline constexpr int kLeaseFd = 4;
inline constexpr int kStartupFd = 5;
inline constexpr int kFirstUnreservedFd = kStartupFd + 1;

/// @brief Create fixed-size shared RAM, independent of RLIMIT_FSIZE.
/// @details Preallocation is best effort for native memfds; token storage stays
/// sparse. RJ_TEST_FORCE_TOKEN_BACKING=1 selects tokens for integration testing
/// without affecting ordinary metadata files or changing application limits.
int create(size_t size, bool preallocate = false);

/// @brief Populate a newly created metadata file using the keeper's inherited hard limit.
/// @details The caller owns this unpublished regular file; its offset stays at zero.
/// Unlike GPU backing, metadata files must fit the keeper's inherited hard file limit.
int populate_file(int fd, const void *data, size_t size);

/// @brief Entry point for the private memory keeper executable.
[[noreturn]] void run_keeper(int listener, int lease, int startup, pid_t owner);

/// @brief Recognize a shared-RAM descriptor without consulting process-local state.
bool contains(int fd);

/// @brief Map shared RAM, or pass a non-backing descriptor to the host mmap.
/// @details The caller must hold the host mapping lock when replacing mappings
/// visible to the simulator. Ordinary munmap/mprotect/mremap work on the result.
void *map(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

/// @brief Inspect RAM size and identity, or pass through for ordinary files.
int stat(int fd, struct stat *result);
/// @brief Overlay token metadata on an already obtained host stat result.
/// @details Returns false for ordinary descriptors; callers can skip non-sockets.
bool stat_token(int fd, struct stat *result);

/// @brief Apply dma-buf seek semantics to a token, returning false for other fds.
/// @details The result is -1 with errno set for unsupported token seek operations.
bool seek_token(int fd, off_t offset, int whence, off_t &result);

/// @brief Query backing metadata or forward a descriptor command to the host.
/// @details Tokens synthesize F_GETFL and F_GET_SEALS, reject F_ADD_SEALS with
/// EPERM, and forward other commands to their underlying socket descriptor.
int fcntl(int fd, int command, uintptr_t argument = 0);

/// @brief Reopen a backing with an independently retained access mode.
int reopen(int fd, int access_mode);

/// @brief Verify that existing backing covers a requested mapping, without growing a file.
int ensure_size(int fd, size_t size);

} // namespace rocjitsu::memory_backing

#endif
