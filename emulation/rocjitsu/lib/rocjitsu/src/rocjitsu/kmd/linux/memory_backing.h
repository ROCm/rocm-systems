// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_MEMORY_BACKING_H_
#define ROCJITSU_KMD_LINUX_MEMORY_BACKING_H_

#include <cstddef>
#include <cstdint>
#include <sys/stat.h>
#include <sys/types.h>

namespace rocjitsu::memory_backing {

/// @brief Create fixed-size shared RAM, independent of RLIMIT_FSIZE.
/// @details The descriptor retains the RAM across dup, fork, and SCM_RIGHTS.
/// Use this interface to map or inspect it: when a memfd cannot fit the
/// inherited hard limit, the result is an opaque shared-RAM handle instead.
/// All functions use errno on failure.
int create(size_t size);

/// @brief Populate a newly created metadata file using the keeper's startup limit.
/// @details The caller owns this unpublished regular file; its offset stays at zero.
/// Unlike GPU backing, metadata files must fit the keeper's inherited file limit.
int populate_file(int fd, const void *data, size_t size);

/// @brief Entry point for the private memory keeper executable.
[[noreturn]] void run_keeper(int listener, pid_t owner);

/// @brief Recognize a shared-RAM descriptor without consulting process-local state.
bool contains(int fd);

/// @brief Map shared RAM, or pass a non-backing descriptor to the host mmap.
/// @details The caller must hold the host mapping lock when replacing mappings
/// visible to the simulator. Ordinary munmap/mprotect/mremap work on the result.
void *map(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

/// @brief Inspect RAM size and identity, or pass through for ordinary files.
int stat(int fd, struct stat *result);
int fcntl(int fd, int command, uintptr_t argument = 0);

/// @brief Reopen a backing with an independently retained access mode.
int reopen(int fd, int access_mode);

/// @brief Verify that existing backing covers a requested mapping, without growing a file.
int ensure_size(int fd, size_t size);

} // namespace rocjitsu::memory_backing

#endif
