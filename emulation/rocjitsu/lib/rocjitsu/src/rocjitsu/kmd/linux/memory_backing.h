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
#include <string_view>
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

// Fixed descriptor slots shared by exec_keeper and the rocjitsu-memory entry
// point across exec. Cleanup and descriptor duplication start at
// kMemoryKeeperFirstUnreservedFd to preserve these channels.
inline constexpr int kMemoryKeeperListenerFd = 3;
inline constexpr int kMemoryKeeperLeaseFd = 4;
inline constexpr int kMemoryKeeperStartupFd = 5;
inline constexpr int kMemoryKeeperFirstUnreservedFd = kMemoryKeeperStartupFd + 1;

inline constexpr char kMemoryBackingDefaultName[] = "rocjitsu_backing";

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

inline constexpr std::string_view kDeviceBackingNamePrefix = "rocjitsu_alloc_renderD";

/// @brief Recognize device backing independently of a descriptor's number.
/// @details SCM_RIGHTS and DMA-BUF export create independent descriptor numbers for the
/// same backing. Its name preserves the device identity across those transfers.
/// Only allocations associated with published topology carry this marker and
/// receive device dump policy. Standalone allocations made before topology is
/// published retain ordinary host dump eligibility through every alias.
bool is_device_memory_backing(int fd);

/// @brief Persistent GPU classification in System V VMA names after IPC_RMID.
/// @details The low slot bits select a transient creation key, released after
/// the keeper attaches and marks the segment for removal. Exhausting these keys
/// falls back to IPC_PRIVATE: token metadata retains device classification, but
/// the VMA name can no longer identify the segment for ROCr GPU core dumps.
inline constexpr uint32_t kDeviceTokenKeyPrefix = 0x524a4700;
inline constexpr unsigned kDeviceTokenSlotBits = 8;
inline constexpr uint32_t kDeviceTokenSlotCount = uint32_t{1} << kDeviceTokenSlotBits;
inline constexpr unsigned kDeviceTokenSlotHexDigits = kDeviceTokenSlotBits / 4;
static_assert(kDeviceTokenSlotBits % 4 == 0);
static_assert((kDeviceTokenKeyPrefix & (kDeviceTokenSlotCount - 1)) == 0);

} // namespace rocjitsu::kmd
