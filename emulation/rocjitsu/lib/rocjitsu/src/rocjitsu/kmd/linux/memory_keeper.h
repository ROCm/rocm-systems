// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file memory_keeper.h
/// @brief Private wire protocol and entry point for the memory keeper.

#include <array>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace rocjitsu::kmd {

// Fixed descriptor slots shared by exec_keeper and the rocjitsu-memory entry
// point across exec. Cleanup and descriptor duplication start at
// kKeeperFirstUnreservedFd to preserve these channels.
inline constexpr int kKeeperLeaseFd = 3;
inline constexpr int kKeeperStartupFd = 4;
inline constexpr int kKeeperFirstUnreservedFd = kKeeperStartupFd + 1;

inline constexpr uint64_t kKeeperTokenMagic = 0x314d41525553544a;
/// @brief Abstract socket address payload identifying a shared-RAM token across processes.
struct KeeperTokenIdentity {
  uint64_t magic = kKeeperTokenMagic;
  uint64_t size = 0;
  uint64_t object = 0;
  uint64_t token = 0;
  uint64_t ipc_namespace = 0;
  int32_t shmid = -1;
  int32_t mode = O_RDWR;
  bool device_backing = false;
};

// The abstract socket name includes the native tail padding (56 bytes on
// 64-bit Linux, 52 on i386). Keep existing exported token names compatible.
static_assert(offsetof(KeeperTokenIdentity, device_backing) == 48);
static_assert(sizeof(KeeperTokenIdentity) == 48 + alignof(uint64_t));

/// @brief Requests accepted on the private listener or an exported token.
/// @details Inline BatchFile requests queue their results. FlushFiles must specify the
/// pending count before any other operation. A mapped BatchFile replies immediately.
/// Both keep the channel open. Sizing and allocation requests reply once and close.
/// The inline limit and these reply boundaries are part of the private protocol.
enum class KeeperOperation : uint32_t {
  Backing,
  SizeFile,
  PreallocatedBacking,
  TokenBacking,
  BatchFile,
  FlushFiles,
  Identify
};

// Keep inline metadata in one small seqpacket. Map larger files instead.
inline constexpr size_t kKeeperInlineFileSize = 4096;
inline constexpr size_t kKeeperMaxBatchFiles = 8;

/// @brief Fixed-width header for a single sequenced-packet request.
struct KeeperRequest {
  // Byte count for a file or allocation, or the pending file count for FlushFiles.
  uint64_t size;
  int32_t mode;
  KeeperOperation operation;
  char name[64];
};
// Token reopen requests can cross an exec into a different rocjitsu build.
static_assert(sizeof(KeeperRequest) == 80);

/// @brief Receive buffer for a metadata request with an optional inline payload.
struct KeeperFilePacket {
  KeeperRequest request;
  char data[kKeeperInlineFileSize];
};

static_assert(offsetof(KeeperFilePacket, data) == sizeof(KeeperRequest));

/// @brief Per-file errno values returned when a metadata batch is flushed.
struct KeeperFileResults {
  uint32_t count = 0;
  std::array<int32_t, kKeeperMaxBatchFiles> errors{};
};

// The private protocol uses a fixed-size flush reply.
static_assert(sizeof(KeeperFileResults) == 36);

// The original helper sent only its PID. A distinct ready packet rejects a
// mismatched private protocol before either side waits for an allocation reply.
inline constexpr uint32_t kKeeperProtocolVersion = 3;
using KeeperCookie = std::array<uint64_t, 2>;
/// @brief Private startup result and identity for subsequent owner connections.
struct KeeperStartupReply {
  uint32_t version;
  int32_t error;
  KeeperCookie cookie;
  uint64_t process_inode;
};

static_assert(sizeof(KeeperStartupReply) == 32);

uint64_t keeper_ipc_namespace();

int send_keeper_descriptor(int socket, const void *data, size_t size, int fd,
                           const void *payload = nullptr, size_t payload_size = 0,
                           int flags = MSG_DONTWAIT);
long receive_keeper_descriptor(int socket, void *data, size_t size, int &fd, int flags = 0,
                               bool *missing_descriptor = nullptr, pid_t *sender_pid = nullptr);
int send_keeper_startup_reply(int socket, int error, const KeeperCookie &cookie = {},
                              uint64_t process_inode = 0);

/// @brief Serve backing requests in the private helper until its owner and tokens are gone.
[[noreturn]] void run_memory_keeper(pid_t owner, const char *listener_name);

} // namespace rocjitsu::kmd
