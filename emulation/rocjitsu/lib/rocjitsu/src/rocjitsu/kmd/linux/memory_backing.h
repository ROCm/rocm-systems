// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file memory_backing.h
/// @brief Shared backing descriptors, either native memfds or opaque socket tokens.
/// @details Descriptors retain their storage across dup, fork, and SCM_RIGHTS.
/// Tokens additionally depend on their creating keeper; native memfds do not.
/// Keeper loss invalidates tokens; new
/// mappings and reopens report ESTALE. Existing mapped views remain usable.
/// Callers own returned descriptors and must use these adapters for mapping and
/// metadata; tokens do not support file I/O. Failures use errno. Ordinary file
/// descriptors retain host behavior where an operation supports both forms.

#include "rocjitsu/kmd/linux/device_backing.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace rocjitsu {

/// @brief Flags shared by token mapping and the public VM mapper.
/// @details MAP_SHARED is required; the placement flags are optional.
inline constexpr uint32_t kMemoryBackingMappingFlags = MAP_SHARED | MAP_FIXED | MAP_FIXED_NOREPLACE
#ifdef MAP_32BIT
                                                       | MAP_32BIT
#endif
    ;

inline constexpr char kMemoryBackingDefaultName[] = "rocjitsu_backing";

/// @brief Prepare the private memory keeper with the current resource limits.
/// @details The interposer calls this before application code can lower its
/// hard limits. It validates the initial handshake or checks an existing lease.
/// @returns Zero after preparation or -1 with errno set on startup failure.
int prepare_memory_keeper();

/// @brief Create fixed-size shared RAM, independent of RLIMIT_FSIZE.
/// @details Preallocation is best effort for native memfds; token storage stays
/// sparse. RJ_TEST_FORCE_TOKEN_BACKING=1 selects tokens for integration testing
/// without affecting ordinary metadata files or changing application limits.
/// The name controls classification: names beginning with kmd::kDeviceBackingNamePrefix
/// select device backing for both memfds and tokens. All names without that prefix
/// select ordinary host backing, including kMemoryBackingDefaultName and custom
/// diagnostic names such as "rocjitsu_alloc" or "rocjitsu_userptr".
/// Device tokens falling back to IPC_PRIVATE retain
/// that classification but lose the ROCr GPU-core maps marker, as described in
/// docs/memory-backing.md under "GPU core dump classification".
int create_memory_backing(size_t size, bool preallocate = false,
                          const char *name = kMemoryBackingDefaultName);

/// @brief Populate a newly created metadata file using the keeper's inherited hard limit.
/// @details The caller owns this unpublished regular file; its offset is unchanged.
/// Unlike GPU backing, metadata files must fit the keeper's inherited hard file limit.
int populate_backing_file(int fd, const void *data, size_t size);

/// @brief An unpublished metadata file and its individual creation/population error.
struct BackingFile {
  std::string path;
  std::string contents;
  int error = 0;
};

/// @brief Create metadata files in order, sharing a keeper connection for each run of nonempty
/// files.
/// @details Create paths with mode 0666 and truncate existing files. Empty entries are handled
/// locally. Keep the files unpublished until this call returns. Each entry reports zero on success
/// or its errno on failure.
void create_backing_files(std::span<BackingFile> files);

/// @brief Recognize a shared-RAM descriptor without consulting process-local state.
bool is_memory_backing_token(int fd);

/// @brief Map shared RAM, or pass a non-backing descriptor to the host mmap.
/// @details The caller must hold the host mapping lock when replacing mappings
/// visible to the simulator. Ordinary munmap/mprotect/mremap work on the result.
/// Token attachments enforce current System V IPC permissions; retaining a token
/// across an effective UID change does not grant new mapping authority (EACCES).
/// A token whose keeper has exited fails with ESTALE.
/// Tokens accept only kMemoryBackingMappingFlags; other flags fail with EINVAL.
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
int check_memory_backing_size(int fd, size_t size);

} // namespace rocjitsu

namespace rocjitsu::kmd {

/// @brief Recognize device backing independently of a descriptor's number.
/// @details SCM_RIGHTS and DMA-BUF export create independent descriptor numbers for the
/// same backing. Its name preserves the device identity across those transfers.
/// Only allocations associated with published topology carry this marker and
/// receive device dump policy. Standalone allocations made before topology is
/// published retain ordinary host dump eligibility through every alias.
bool is_device_memory_backing(int fd);

} // namespace rocjitsu::kmd
