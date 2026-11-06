// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_backing_test.cpp
/// @brief Tests memory backing and its helper in isolated processes.

#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/memory_backing.h"
#include "rocjitsu/kmd/linux/memory_keeper.h"

#include "rocjitsu/kmd/linux/syscall_compat.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <poll.h>
#include <sched.h>
#include <span>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr size_t kSize = 64 * 1024;
// Carry a death-test child's prerequisite skip back to the parent test.
constexpr int kPrerequisiteSkipExitCode = 77;

class MemoryBackingDeathTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Re-exec buffers gtest_main's banner. TSan flushes it on _exit, which
    // would exceed the test's later hard-zero limit when stdout is a file.
    ASSERT_EQ(std::fflush(stdout), 0);
  }
};

ssize_t receive_legacy_keeper_ready(int socket, msghdr *message, int flags) {
  rocjitsu::libc_passthrough().recvmsg = ::recvmsg;
  const ssize_t received = ::recvmsg(socket, message, flags);
  if (received < static_cast<ssize_t>(sizeof(int)))
    return received;
  const int legacy_pid = getpid();
  std::memcpy(message->msg_iov[0].iov_base, &legacy_pid, sizeof(legacy_pid));
  return sizeof(legacy_pid);
}

TEST_F(MemoryBackingDeathTest, LegacyKeeperFailsBeforeSendingRequests) {
  ASSERT_EXIT(
      {
        rocjitsu::libc_passthrough().recvmsg = receive_legacy_keeper_ready;
        EXPECT_EQ(rocjitsu::prepare_memory_keeper(), -1);
        EXPECT_EQ(errno, EPROTO);
        rocjitsu::libc_passthrough().recvmsg = receive_legacy_keeper_ready;
        const int failed = rocjitsu::create_memory_backing(1);
        const int error = errno;
        EXPECT_EQ(failed, -1);
        EXPECT_EQ(error, EPROTO);
        // A subsequent launch can use a matching helper.
        const int fd = rocjitsu::create_memory_backing(1);
        EXPECT_GE(fd, 0);
        close(fd);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_device_backing_identity() {
  const std::string name = std::string(rocjitsu::kmd::kDeviceBackingNamePrefix) + "129";
  const int native = rocjitsu::create_memory_backing(kSize, false, name.c_str());
  ASSERT_GE(native, 0);
  EXPECT_TRUE(rocjitsu::kmd::is_device_memory_backing(native));
  close(native);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1);
  std::vector<int> retained;
  // Keys are released by IPC_RMID, even while more than a full slot count of regions
  // retain the creation marker in their kernel VMA names.
  for (unsigned index = 0; index < rocjitsu::kmd::kDeviceTokenSlotCount + 1; ++index) {
    const int backing = rocjitsu::create_memory_backing(kSize, false, name.c_str());
    ASSERT_GE(backing, 0);
    ASSERT_TRUE(rocjitsu::is_memory_backing_token(backing));
    EXPECT_TRUE(rocjitsu::kmd::is_device_memory_backing(backing));
    retained.push_back(backing);
  }
  const int alias = rocjitsu::reopen_memory_backing(retained.front(), O_RDONLY);
  ASSERT_GE(alias, 0);
  EXPECT_TRUE(rocjitsu::kmd::is_device_memory_backing(alias));
  close(alias);
  const int ordinary = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(ordinary, 0);
  EXPECT_FALSE(rocjitsu::kmd::is_device_memory_backing(ordinary));
  close(ordinary);
  for (const int backing : retained)
    close(backing);
}

TEST_F(MemoryBackingDeathTest, DeviceIdentitySurvivesReopenAndCreationKeysAreReleased) {
  ASSERT_EXIT(
      {
        check_device_backing_identity();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, NativePreallocationAndForcedTokensPreserveMetadata) {
  ASSERT_EXIT(
      {
        const int sparse = rocjitsu::create_memory_backing(kSize);
        const int committed = rocjitsu::create_memory_backing(kSize, true);
        struct stat sparse_info {};
        struct stat committed_info {};
        bool okay = sparse >= 0 && committed >= 0 && fstat(sparse, &sparse_info) == 0 &&
                    fstat(committed, &committed_info) == 0 && sparse_info.st_blocks == 0 &&
                    committed_info.st_blocks >= static_cast<blkcnt_t>(kSize / 512);
        setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1);
        const int token = rocjitsu::create_memory_backing(kSize, true);
        char path[] = "/tmp/rocjitsu-metadata-XXXXXX";
        const int metadata = mkstemp(path);
        unlink(path);
        const char contents[] = "metadata";
        okay = okay && token >= 0 && rocjitsu::is_memory_backing_token(token) && metadata >= 0 &&
               rocjitsu::populate_backing_file(metadata, contents, sizeof(contents)) == 0;
        char observed[sizeof(contents)]{};
        okay = okay && pread(metadata, observed, sizeof(observed), 0) == sizeof(observed) &&
               std::memcmp(contents, observed, sizeof(contents)) == 0;
        close(metadata);
        close(token);
        close(committed);
        close(sparse);
        _exit(okay ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_full_mapping_address_space(bool token) {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", token ? "1" : "0", 1), 0);
  constexpr size_t size = 64ULL << 20;
  const int backing = rocjitsu::create_memory_backing(size);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(rocjitsu::is_memory_backing_token(backing), token);
  uint64_t pages = 0;
  {
    std::ifstream statm("/proc/self/statm");
    ASSERT_TRUE(statm >> pages);
  }
  rlimit original{};
  ASSERT_EQ(getrlimit(RLIMIT_AS, &original), 0);
  const uint64_t virtual_size = pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
  const rlim_t headroom = original.rlim_cur > virtual_size ? original.rlim_cur - virtual_size : 0;
  constexpr rlim_t experiment_headroom = 96ULL << 20;
  if (headroom < experiment_headroom)
    GTEST_SKIP() << "inherited RLIMIT_AS needs 96 MiB of headroom for the experiment";
  if (token && headroom < 2 * size)
    GTEST_SKIP() << "inherited RLIMIT_AS needs 128 MiB of headroom for the token retry";
  rlimit limited = original;
  limited.rlim_cur = virtual_size + experiment_headroom;
  ASSERT_EQ(setrlimit(RLIMIT_AS, &limited), 0);
  void *view =
      rocjitsu::map_memory_backing(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
  const int error = errno;
  ASSERT_EQ(setrlimit(RLIMIT_AS, &original), 0);
  if (token) {
    EXPECT_EQ(view, MAP_FAILED);
    EXPECT_EQ(error, ENOMEM);
    if (view != MAP_FAILED) {
      EXPECT_EQ(munmap(view, size), 0);
    }
    view =
        rocjitsu::map_memory_backing(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
  }
  ASSERT_NE(view, MAP_FAILED);
  auto *bytes = static_cast<unsigned char *>(view);
  bytes[0] = 0x37;
  bytes[size - 1] = 0xA5;
  EXPECT_EQ(bytes[0], 0x37);
  EXPECT_EQ(bytes[size - 1], 0xA5);
  EXPECT_EQ(munmap(view, size), 0);
  EXPECT_EQ(close(backing), 0);
}

void check_full_mapping_address_space_in_child(bool token) {
  bool skipped = false;
  const auto accepted_status = [&skipped](int status) {
    skipped = ::testing::ExitedWithCode(kPrerequisiteSkipExitCode)(status);
    return skipped || ::testing::ExitedWithCode(0)(status);
  };
  ASSERT_EXIT(
      {
        check_full_mapping_address_space(token);
        _exit(::testing::Test::HasFailure()  ? 1
              : ::testing::Test::IsSkipped() ? kPrerequisiteSkipExitCode
                                             : 0);
      },
      accepted_status, "");
  if (skipped)
    GTEST_SKIP() << (token ? "inherited RLIMIT_AS lacks headroom for the 96 MiB experiment or "
                             "128 MiB token retry"
                           : "inherited RLIMIT_AS needs 96 MiB of headroom for the experiment");
}

TEST_F(MemoryBackingDeathTest, NativeFullMappingFitsConstrainedAddressSpace) {
  check_full_mapping_address_space_in_child(false);
}

TEST_F(MemoryBackingDeathTest, TokenFullMappingNeedsTemporaryAttachmentAndView) {
  check_full_mapping_address_space_in_child(true);
}

void check_keeper_startup_error() {
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_pidfd_open, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(std::size(filter)), filter};
  ASSERT_EQ(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
  ASSERT_EQ(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program), 0);
  EXPECT_EQ(rocjitsu::prepare_memory_keeper(), -1);
  EXPECT_EQ(errno, EPERM);
}

TEST_F(MemoryBackingDeathTest, PreparationReportsKeeperStartupError) {
  ASSERT_EXIT(
      {
        check_keeper_startup_error();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

bool is_identify_request(const msghdr *message) {
  if (!message->msg_iovlen || message->msg_iov[0].iov_len != sizeof(rocjitsu::kmd::KeeperRequest))
    return false;
  const auto *request =
      static_cast<const rocjitsu::kmd::KeeperRequest *>(message->msg_iov[0].iov_base);
  return request->operation == rocjitsu::kmd::KeeperOperation::Identify;
}

void check_keeper_probes() {
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  // Retain the same open file description a concurrent fork would inherit.
  static int retained = -1;
  auto &libc = rocjitsu::libc_passthrough();
  libc.connect = [](int fd, const sockaddr *address, socklen_t length) {
    const int result = ::connect(fd, address, length);
    if (result == 0 && (fcntl(fd, F_GETFL) & O_NONBLOCK)) {
      if (retained >= 0)
        close(retained);
      retained = dup(fd);
    }
    return result;
  };
  auto expect_peer_closed = [] {
    ASSERT_GE(retained, 0);
    pollfd peer{retained, POLLRDHUP, 0};
    EXPECT_EQ(poll(&peer, 1, 2000), 1);
    EXPECT_NE(peer.revents & POLLRDHUP, 0);
    close(retained);
    retained = -1;
  };
  EXPECT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  expect_peer_closed();

  // Fail the allocation send so the client probes a live keeper for a reply.
  libc.sendmsg = [](int fd, const msghdr *message, int flags) -> ssize_t {
    if (is_identify_request(message))
      return ::sendmsg(fd, message, flags);
    rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
    errno = EPIPE;
    return -1;
  };
  static unsigned polls = 0;
  libc.poll = [](pollfd *fds, nfds_t count, int timeout) {
    if (++polls == 1) {
      errno = EINTR;
      return -1;
    }
    return ::poll(fds, count, timeout);
  };
  EXPECT_EQ(rocjitsu::create_memory_backing(1), -1);
  EXPECT_EQ(errno, EPIPE);
  EXPECT_GE(polls, 2u);
  expect_peer_closed();
}

TEST_F(MemoryBackingDeathTest, KeeperProbesReleaseInheritedSocketsAndRetryInterruptedPoll) {
  ASSERT_EXIT(
      {
        check_keeper_probes();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_inherited_descriptors() {
  int pipes[2];
  ASSERT_EQ(pipe(pipes), 0);
  const int high = fcntl(pipes[1], F_DUPFD, 200);
  ASSERT_GE(high, 200);
  ASSERT_EQ(close(pipes[1]), 0);
  const rlimit nofile{64, 64};
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &nofile), 0);
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_close_range, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(std::size(filter)), filter};
  ASSERT_EQ(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
  ASSERT_EQ(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program), 0);
  const int memory = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(memory, 0);
  ASSERT_EQ(close(high), 0);
  pollfd pipe_state{pipes[0], POLLIN | POLLHUP, 0};
  ASSERT_EQ(poll(&pipe_state, 1, 1000), 1);
  EXPECT_TRUE(pipe_state.revents & POLLHUP);
  close(memory);
  close(pipes[0]);
}

TEST_F(MemoryBackingDeathTest, ClosesInheritedDescriptorsAboveLoweredHardLimit) {
  ASSERT_EXIT(
      {
        check_inherited_descriptors();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_metadata_population() {
  // Preparation must finish the startup handshake before the hard limit drops.
  // A one-off backing allocation would also open a request connection.
  auto &libc = rocjitsu::libc_passthrough();
  const auto saved_connect = libc.connect;
  libc.connect = [](int, const sockaddr *, socklen_t) {
    errno = ECONNABORTED;
    return -1;
  };
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  libc.connect = saved_connect;
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  for (const bool resolved : {false, true}) {
    if (resolved)
      rocjitsu::libc_passthrough().resolve();
    for (const bool memory_file : {false, true}) {
      SCOPED_TRACE(resolved);
      SCOPED_TRACE(memory_file);
      char path[] = "/tmp/rocjitsu-metadata-XXXXXX";
      const int fd = memory_file ? memfd_create("metadata", MFD_CLOEXEC) : mkstemp(path);
      ASSERT_GE(fd, 0);
      if (!memory_file) {
        ASSERT_EQ(unlink(path), 0);
      }
      // Exercise a caller-owned append-mode descriptor as well as a regular file.
      // Even a small snapshot must replace its contents and preserve the offset.
      if (memory_file) {
        ASSERT_EQ(fcntl(fd, F_SETFL, O_APPEND), 0);
      }
      for (const size_t size : {size_t{2}, size_t{4096}, size_t{4097}, size_t{8193}, size_t{17}}) {
        std::string expected(size, '\0');
        for (size_t byte_index = 0; byte_index < size; ++byte_index)
          expected[byte_index] = static_cast<char>(byte_index * 37);
        ASSERT_EQ(lseek(fd, 7, SEEK_SET), 7);
        ASSERT_EQ(rocjitsu::populate_backing_file(fd, expected.data(), expected.size()), 0);
        struct stat info {};
        ASSERT_EQ(fstat(fd, &info), 0);
        EXPECT_EQ(info.st_size, size);
        EXPECT_EQ(lseek(fd, 0, SEEK_CUR), 7);
        std::string actual(size, '\0');
        ASSERT_EQ(pread(fd, actual.data(), actual.size(), 0), size);
        EXPECT_EQ(actual, expected);
      }
      close(fd);
    }
  }
  rlimit observed{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &observed), 0);
  EXPECT_EQ(observed.rlim_cur, 0);
  EXPECT_EQ(observed.rlim_max, 0);
}

TEST_F(MemoryBackingDeathTest, MetadataPopulationPreservesContentsAfterLimitDrops) {
  ASSERT_EXIT(
      {
        check_metadata_population();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_metadata_limit_errors() {
  const rlimit limited{64, 64};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limited), 0);
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int fd = memfd_create("metadata", MFD_CLOEXEC);
  ASSERT_GE(fd, 0);
  const std::string contents(64, 'm');
  ASSERT_EQ(rocjitsu::populate_backing_file(fd, contents.data(), contents.size()), 0);
  const std::string too_large(65, 'x');
  EXPECT_EQ(rocjitsu::populate_backing_file(fd, too_large.data(), too_large.size()), -1);
  EXPECT_EQ(errno, EFBIG);
  std::string actual(contents.size(), '\0');
  ASSERT_EQ(pread(fd, actual.data(), actual.size(), 0), actual.size());
  EXPECT_EQ(actual, contents);

  const std::string path = "/proc/self/fd/" + std::to_string(fd);
  const int readonly = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(readonly, 0);
  EXPECT_EQ(rocjitsu::populate_backing_file(readonly, contents.data(), contents.size()), -1);
  EXPECT_EQ(errno, EINVAL);
  close(readonly);
  // Errors must not retire the helper or leave a reply queued for the next call.
  ASSERT_EQ(rocjitsu::populate_backing_file(fd, contents.data(), 17), 0);
  struct stat info {};
  ASSERT_EQ(fstat(fd, &info), 0);
  EXPECT_EQ(info.st_size, 17);
  close(fd);
}

TEST_F(MemoryBackingDeathTest, MetadataPopulationReportsHelperLimitsAndAccessErrors) {
  ASSERT_EXIT(
      {
        check_metadata_limit_errors();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

int small_send_buffer_socket(int domain, int type, int protocol) noexcept {
  const int fd = ::socket(domain, type, protocol);
  const int size = 4096;
  if (fd >= 0 && setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) != 0) {
    const int error = errno;
    close(fd);
    errno = error;
    return -1;
  }
  return fd;
}

unsigned queued_files = 0;
unsigned queue_full_count = 0;

// Model the kernel limiting outstanding SCM_RIGHTS messages until a flush.
// This makes descriptor pressure independent of how fast the keeper runs.
ssize_t limit_queued_files(int socket, const msghdr *message, int flags) {
  if (message->msg_iovlen == 2) {
    if (queued_files == 2) {
      ++queue_full_count;
      errno = ETOOMANYREFS;
      return -1;
    }
    ++queued_files;
  } else {
    queued_files = 0;
  }
  return ::sendmsg(socket, message, flags);
}

void check_metadata_batch(bool finite_limit) {
  if (finite_limit) {
    const rlimit limit{64, 64};
    ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
  }
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  char directory[] = "/tmp/rocjitsu-metadata-batch-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  rlimit original{};
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &original), 0);
  rlimit limited = original;
  limited.rlim_cur = std::min<rlim_t>(original.rlim_cur, 16);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &limited), 0);
  std::vector<rocjitsu::BackingFile> files;
  const size_t inlined = rocjitsu::kmd::kKeeperInlineFileSize;
  const size_t sizes[] = {0,       1,       64,      65,          inlined,
                          inlined, inlined, inlined, inlined + 1, 2 * inlined + 1};
  for (size_t file_index = 0; file_index < 42; ++file_index) {
    std::string contents(sizes[file_index % std::size(sizes)], '\0');
    for (size_t byte_index = 0; byte_index < contents.size(); ++byte_index)
      contents[byte_index] = static_cast<char>(file_index + byte_index * 37);
    files.push_back(
        {std::string(directory) + "/" + std::to_string(file_index), std::move(contents)});
  }
  files[20].path = std::string(directory) + "/missing/parent";
  rocjitsu::libc_passthrough().socket = small_send_buffer_socket;
  rocjitsu::libc_passthrough().sendmsg = limit_queued_files;
  rocjitsu::create_backing_files(files);
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  rocjitsu::libc_passthrough().socket = ::socket;
  EXPECT_GT(queue_full_count, 0u);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  for (size_t file_index = 0; file_index < files.size(); ++file_index) {
    const rocjitsu::BackingFile &file = files[file_index];
    if (file_index == 20) {
      EXPECT_EQ(file.error, ENOENT);
      continue;
    }
    const bool too_large = finite_limit && file.contents.size() > 64;
    EXPECT_EQ(file.error, too_large ? EFBIG : 0) << file_index;
    const int fd = open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    struct stat info {};
    ASSERT_EQ(fstat(fd, &info), 0);
    const size_t expected = too_large ? 0 : file.contents.size();
    EXPECT_EQ(info.st_size, expected);
    std::string observed(expected, '\0');
    ASSERT_EQ(read(fd, observed.data(), observed.size()), observed.size());
    if (!too_large) {
      EXPECT_EQ(observed, file.contents);
    }
    close(fd);
    EXPECT_EQ(unlink(file.path.c_str()), 0);
  }
  EXPECT_EQ(rmdir(directory), 0);
}

TEST_F(MemoryBackingDeathTest, MetadataBatchPreservesContentsWithFewDescriptors) {
  ASSERT_EXIT(
      {
        check_metadata_batch(false);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, MetadataBatchReportsIndividualFileErrors) {
  ASSERT_EXIT(
      {
        check_metadata_batch(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

int deny_metadata_connection(int, const sockaddr *, socklen_t) {
  errno = EPERM;
  return -1;
}

int refuse_disconnected_keeper(int, const sockaddr *, socklen_t) {
  // Refuse the health probe, then deny the request to the replacement keeper.
  rocjitsu::libc_passthrough().connect = deny_metadata_connection;
  errno = ECONNREFUSED;
  return -1;
}

void check_unavailable_metadata_connection(bool after_flush) {
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  char directory[] = "/tmp/rocjitsu-failed-connection-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const std::string prefix = std::string(directory) + "/";
  const std::string existing = prefix + "existing";
  const int fd = open(existing.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(write(fd, "old", 3), 3);

  auto &libc = rocjitsu::libc_passthrough();
  std::vector<rocjitsu::BackingFile> files;
  if (after_flush) {
    for (size_t file_index = 0; file_index < rocjitsu::kmd::kKeeperMaxBatchFiles; ++file_index)
      files.push_back({prefix + std::to_string(file_index), "keep"});
    // Inject a disconnect on the first write after this group is acknowledged.
    static size_t writes_until_failure = rocjitsu::kmd::kKeeperMaxBatchFiles + 1;
    static const char *inflight_path;
    inflight_path = existing.c_str();
    libc.sendmsg = [](int socket, const msghdr *message, int flags) -> ssize_t {
      if (message->msg_iovlen == 2 && --writes_until_failure == 0) {
        // Cleanup must not reopen this already-opened destination by name.
        EXPECT_EQ(unlink(inflight_path), 0);
        auto &libc = rocjitsu::libc_passthrough();
        libc.sendmsg = ::sendmsg;
        libc.connect = refuse_disconnected_keeper;
        errno = EPIPE;
        return -1;
      }
      return ::sendmsg(socket, message, flags);
    };
  } else {
    libc.connect = deny_metadata_connection;
  }
  const size_t acknowledged = files.size();
  files.push_back({existing, "x"});
  files.push_back({prefix + "new", "x"});
  files.push_back({prefix + "new/child", "x"});
  files.push_back({prefix + "missing/child", "x"});
  rocjitsu::create_backing_files(files);
  libc.connect = ::connect;
  libc.sendmsg = ::sendmsg;

  for (size_t file_index = 0; file_index < acknowledged; ++file_index) {
    EXPECT_EQ(files[file_index].error, 0);
    std::ifstream input(files[file_index].path);
    std::string contents;
    input >> contents;
    EXPECT_EQ(contents, "keep");
    EXPECT_EQ(unlink(files[file_index].path.c_str()), 0);
  }
  EXPECT_EQ(files[acknowledged].error, EPERM);
  EXPECT_EQ(files[acknowledged + 1].error, EPERM);
  EXPECT_EQ(files[acknowledged + 2].error, ENOTDIR);
  EXPECT_EQ(files.back().error, ENOENT);
  struct stat info {};
  ASSERT_EQ(fstat(fd, &info), 0);
  EXPECT_EQ(info.st_size, 0);
  close(fd);
  if (after_flush) {
    EXPECT_EQ(access(existing.c_str(), F_OK), -1);
    EXPECT_EQ(errno, ENOENT);
  } else {
    EXPECT_EQ(unlink(existing.c_str()), 0);
  }
  EXPECT_EQ(unlink(files[acknowledged + 1].path.c_str()), 0);
  EXPECT_EQ(rmdir(directory), 0);
}

TEST_F(MemoryBackingDeathTest, FailedMetadataConnectionPreservesLocalOpenResults) {
  ASSERT_EXIT(
      {
        check_unavailable_metadata_connection(false);
        check_unavailable_metadata_connection(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

int reject_one_open(int, const char *, int, ...) {
  rocjitsu::libc_passthrough().openat = ::openat;
  errno = EMFILE;
  return -1;
}

void check_metadata_descriptor_pressure(bool transient) {
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  char directory[] = "/tmp/rocjitsu-metadata-fds-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const std::string prefix = std::string(directory) + "/";
  rocjitsu::BackingFile files[] = {{prefix + "queued", "keep"},
                                   {prefix + "missing/child", "x"},
                                   {prefix + "existing", "replacement"},
                                   {prefix + "new", "data"},
                                   {prefix + "new/child", "x"}};
  const int fd = open(files[2].path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(write(fd, "old", 3), 3);
  close(fd);

  rlimit original{};
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &original), 0);
  if (transient) {
    // Lose a descriptor slot after a successful send, while the file is still
    // awaiting its flush. Subsequent opens have room to complete the batch.
    rocjitsu::libc_passthrough().sendmsg = [](int socket, const msghdr *message, int flags) {
      const ssize_t sent = ::sendmsg(socket, message, flags);
      if (sent >= 0 && message->msg_iovlen == 2) {
        auto &libc = rocjitsu::libc_passthrough();
        libc.openat = reject_one_open;
        libc.sendmsg = ::sendmsg;
      }
      return sent;
    };
    rocjitsu::create_backing_files(files);
    EXPECT_EQ(files[0].error, 0);
    std::ifstream queued(files[0].path);
    EXPECT_EQ(queued.peek(), 'k');
    EXPECT_EQ(unlink(files[0].path.c_str()), 0);
  } else {
    rlimit one_free = original;
    one_free.rlim_cur = 0;
    while (fcntl(one_free.rlim_cur++, F_GETFD) >= 0) {
    }
    ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &one_free), 0);
    rocjitsu::create_backing_files(std::span(files).subspan(1));
    ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  }

  EXPECT_EQ(files[1].error, ENOENT);
  EXPECT_EQ(files[2].error, transient ? 0 : EMFILE);
  EXPECT_EQ(files[3].error, transient ? 0 : EMFILE);
  EXPECT_EQ(files[4].error, ENOTDIR);
  for (const rocjitsu::BackingFile &file : std::span(files).subspan(2, 2)) {
    std::ifstream input(file.path);
    ASSERT_TRUE(input.is_open());
    std::string text;
    std::getline(input, text);
    EXPECT_EQ(text, transient ? file.contents : "");
    EXPECT_EQ(unlink(file.path.c_str()), 0);
  }
  EXPECT_EQ(rmdir(directory), 0);
}

TEST_F(MemoryBackingDeathTest, MetadataOpensPathsInOrderWithOneFreeDescriptor) {
  ASSERT_EXIT(
      {
        check_metadata_descriptor_pressure(false);
        check_metadata_descriptor_pressure(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_aliases() {
  const rlimit limit{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
  const int fd = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(fd, 0) << strerror(errno);
  auto *original = static_cast<unsigned char *>(
      rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  ASSERT_NE(original, MAP_FAILED);
  original[4096] = 0x42;
  const int readonly = rocjitsu::reopen_memory_backing(fd, O_RDONLY | O_CLOEXEC);
  ASSERT_GE(readonly, 0);
  EXPECT_EQ(rocjitsu::fcntl_memory_backing(readonly, F_GETFL) & O_ACCMODE, O_RDONLY);
  EXPECT_NE(fcntl(readonly, F_GETFD) & FD_CLOEXEC, 0);
  auto *alias = static_cast<unsigned char *>(
      rocjitsu::map_memory_backing(nullptr, 4096, PROT_READ, MAP_SHARED, readonly, 4096));
  ASSERT_NE(alias, MAP_FAILED);
  EXPECT_EQ(alias[0], 0x42);
  EXPECT_EQ(mprotect(alias, 4096, PROT_READ | PROT_WRITE), -1);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(rocjitsu::map_memory_backing(nullptr, 4096, PROT_WRITE, MAP_SHARED, readonly, 0),
            MAP_FAILED);
  EXPECT_EQ(errno, EACCES);
  struct stat before {
  }, after{};
  ASSERT_EQ(rocjitsu::stat_memory_backing(fd, &before), 0);
  ASSERT_EQ(rocjitsu::stat_memory_backing(readonly, &after), 0);
  EXPECT_EQ(before.st_size, kSize);
  EXPECT_EQ(before.st_ino, after.st_ino);
  const int retained = dup(fd);
  ASSERT_GE(retained, 0);
  close(fd);
  close(readonly);
  ASSERT_EQ(munmap(original, kSize), 0);
  ASSERT_EQ(munmap(alias, 4096), 0);
  original = static_cast<unsigned char *>(rocjitsu::map_memory_backing(
      nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, retained, 0));
  ASSERT_NE(original, MAP_FAILED);
  close(retained);
  EXPECT_EQ(original[4096], 0x42);
  EXPECT_EQ(munmap(original, kSize), 0);
  rlimit observed{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &observed), 0);
  EXPECT_EQ(observed.rlim_cur, 0);
  EXPECT_EQ(observed.rlim_max, 0);
}

TEST_F(MemoryBackingDeathTest, HardZeroLimitPreservesAliasesAndAccessModes) {
  ASSERT_EXIT(
      {
        check_aliases();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_FixedPartialMappingsPreserveDataAndRejectOverlap() {
  const int fd = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(fd, 0);
  auto *original = static_cast<unsigned char *>(
      rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  ASSERT_NE(original, MAP_FAILED);
  original[4096] = 0x57;
  auto *target = static_cast<unsigned char *>(
      mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(target, MAP_FAILED);
  target[0] = 0x11;
  EXPECT_EQ(rocjitsu::map_memory_backing(target, 4096, PROT_READ, MAP_SHARED | MAP_FIXED_NOREPLACE,
                                         fd, 4096),
            MAP_FAILED);
  EXPECT_EQ(errno, EEXIST);
  EXPECT_EQ(target[0], 0x11);
  ASSERT_EQ(rocjitsu::map_memory_backing(target, 4096, PROT_READ | PROT_WRITE,
                                         MAP_SHARED | MAP_FIXED, fd, 4096),
            target);
  EXPECT_EQ(target[0], 0x57);
  target[0] = 0x68;
  EXPECT_EQ(original[4096], 0x68);
  ASSERT_EQ(munmap(target, kSize), 0);
  // The freed hole is also a candidate for the helper's temporary shmat.
  ASSERT_EQ(rocjitsu::map_memory_backing(target, kSize, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0),
            target);
  EXPECT_EQ(target[4096], 0x68);
  ASSERT_EQ(munmap(target, kSize), 0);
  ASSERT_EQ(rocjitsu::map_memory_backing(target, kSize, PROT_READ, MAP_SHARED | MAP_FIXED_NOREPLACE,
                                         fd, 0),
            target);
  EXPECT_EQ(target[4096], 0x68);
  EXPECT_EQ(munmap(target, kSize), 0);
  EXPECT_EQ(munmap(original, kSize), 0);
  close(fd);
}

void check_ConcurrentReopensUseIndependentReplies() {
  const int fd = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(fd, 0);
  std::atomic<unsigned> failures = 0;
  std::vector<std::thread> workers;
  for (unsigned thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&] {
      for (unsigned iteration = 0; iteration < 32; ++iteration) {
        const int alias = rocjitsu::reopen_memory_backing(fd, O_RDWR);
        if (alias < 0) {
          ++failures;
          continue;
        }
        void *view = rocjitsu::map_memory_backing(nullptr, 4096, PROT_READ, MAP_SHARED, alias, 0);
        if (view == MAP_FAILED || (fcntl(alias, F_GETFD) & FD_CLOEXEC))
          ++failures;
        if (view != MAP_FAILED)
          munmap(view, 4096);
        close(alias);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  EXPECT_EQ(failures, 0);
  close(fd);
}

void check_DescriptorSurvivesCreatingProcessExit() {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(sockets[0]);
    const rlimit limit{};
    if (setrlimit(RLIMIT_FSIZE, &limit) != 0)
      _exit(1);
    const int fd = rocjitsu::create_memory_backing(kSize);
    if (fd < 0)
      _exit(2);
    auto *view = static_cast<unsigned char *>(
        rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (view == MAP_FAILED)
      _exit(3);
    view[0] = 0x9A;
    ucred peer{};
    socklen_t peer_size = sizeof(peer);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) != 0)
      _exit(4);
    pid_t helper = peer.pid;
    iovec data{&helper, sizeof(helper)};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(fd))]{};
    msghdr message{};
    message.msg_iov = &data;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    auto *header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(fd));
    memcpy(CMSG_DATA(header), &fd, sizeof(fd));
    _exit(sendmsg(sockets[1], &message, MSG_NOSIGNAL) == sizeof(helper) ? 0 : 5);
  }
  close(sockets[1]);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  pid_t helper = -1;
  iovec data{&helper, sizeof(helper)};
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
  msghdr message{};
  message.msg_iov = &data;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  ASSERT_EQ(recvmsg(sockets[0], &message, MSG_CMSG_CLOEXEC), sizeof(helper));
  auto *header = CMSG_FIRSTHDR(&message);
  ASSERT_NE(header, nullptr);
  int fd = -1;
  memcpy(&fd, CMSG_DATA(header), sizeof(fd));
  ASSERT_GE(fd, 0);
  const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
  ASSERT_GE(pidfd, 0);
  FILE *segments = std::fopen("/proc/sysvipc/shm", "r");
  ASSERT_NE(segments, nullptr);
  char line[512];
  int segment = -1;
  while (std::fgets(line, sizeof(line), segments)) {
    int id = -1, creator = -1;
    unsigned long long size = 0;
    if (std::sscanf(line, "%*u %d %*o %llu %d", &id, &size, &creator) == 3 && creator == helper &&
        size == kSize)
      segment = id;
  }
  std::fclose(segments);
  ASSERT_GE(segment, 0);
  auto *view = static_cast<unsigned char *>(
      rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, fd, 0));
  ASSERT_NE(view, MAP_FAILED);
  close(fd);
  close(sockets[0]);
  pollfd exited{pidfd, POLLIN, 0};
  EXPECT_EQ(poll(&exited, 1, 2000), 1) << "keeper retained its final descriptor";
  close(pidfd);
  // The mapping itself retains the marked-for-removal segment after the keeper
  // exits. Its last unmap must remove the kernel object, too.
  shmid_ds info{};
  EXPECT_EQ(shmctl(segment, IPC_STAT, &info), 0);
  EXPECT_EQ(view[0], 0x9A);
  EXPECT_EQ(munmap(view, kSize), 0);
  EXPECT_EQ(shmctl(segment, IPC_STAT, &info), -1);
  EXPECT_EQ(errno, EINVAL);
}
TEST_F(MemoryBackingDeathTest, FixedPartialMappingsPreserveDataAndRejectOverlap) {
  ASSERT_EXIT(
      {
        const rlimit limit{};
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(2);
        check_FixedPartialMappingsPreserveDataAndRejectOverlap();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, ConcurrentReopensUseIndependentReplies) {
  ASSERT_EXIT(
      {
        const rlimit limit{};
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(2);
        check_ConcurrentReopensUseIndependentReplies();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, DescriptorSurvivesCreatingProcessExit) {
  ASSERT_EXIT(
      {
        const rlimit limit{};
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(2);
        check_DescriptorSurvivesCreatingProcessExit();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_concurrent_limits() {
  rlimit limit{};
  limit.rlim_max = 4096;
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
  std::atomic<bool> stop = false;
  std::atomic<unsigned> failures = 0;
  std::thread changer([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      limit.rlim_cur = limit.rlim_cur ? 0 : 4096;
      if (setrlimit(RLIMIT_FSIZE, &limit) != 0)
        ++failures;
    }
  });
  std::vector<std::thread> workers;
  for (unsigned thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&] {
      for (unsigned iteration = 0; iteration < 16; ++iteration) {
        const int fd = rocjitsu::create_memory_backing(kSize);
        if (fd < 0) {
          ++failures;
          continue;
        }
        auto *view = static_cast<unsigned char *>(rocjitsu::map_memory_backing(
            nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        close(fd);
        if (view == MAP_FAILED) {
          ++failures;
          continue;
        }
        view[kSize - 1] = 0x75;
        munmap(view, kSize);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  stop.store(true, std::memory_order_relaxed);
  changer.join();
  EXPECT_EQ(failures, 0);
  rlimit observed{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &observed), 0);
  EXPECT_EQ(observed.rlim_max, 4096);
  EXPECT_EQ(observed.rlim_cur, limit.rlim_cur);
}

TEST_F(MemoryBackingDeathTest, ConcurrentLimitChangesDoNotAffectAllocations) {
  ASSERT_EXIT(
      {
        check_concurrent_limits();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

pid_t token_keeper(int fd) {
  ucred peer{};
  socklen_t size = sizeof(peer);
  return getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0 ? peer.pid : -1;
}

int bootstrap_child_lifetime[2];
pid_t bootstrap_child = -1;

int fork_during_bootstrap(int domain, int type, int protocol, int sockets[2]) noexcept {
  const int result = ::socketpair(domain, type, protocol, sockets);
  if (result == 0) {
    bootstrap_child = fork();
    if (bootstrap_child == 0) {
      close(bootstrap_child_lifetime[1]);
      char byte;
      _exit(read(bootstrap_child_lifetime[0], &byte, 1) == 0 ? 0 : 1);
    }
  }
  return result;
}

sockaddr_storage private_address{};
socklen_t private_address_size = 0;

int capture_private_address(int fd, const sockaddr *address, socklen_t size) {
  rocjitsu::libc_passthrough().connect = ::connect;
  if (size <= sizeof(private_address)) {
    std::memcpy(&private_address, address, size);
    private_address_size = size;
  }
  return ::connect(fd, address, size);
}

pid_t recycled_pid;

int report_recycled_peer(int fd, int level, int option, void *value, socklen_t *size) noexcept {
  const int status = ::getsockopt(fd, level, option, value, size);
  if (status == 0 && level == SOL_SOCKET && option == SO_PEERCRED && *size == sizeof(ucred)) {
    auto &peer = *static_cast<ucred *>(value);
    if (peer.pid == getpid())
      peer.pid = recycled_pid;
  }
  return status;
}

void check_rebound_keeper(bool prepare_first, bool recycle_credentials = false) {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  rocjitsu::libc_passthrough().connect = capture_private_address;
  const int token = rocjitsu::create_memory_backing(1);
  ASSERT_GE(token, 0);
  const pid_t helper = token_keeper(token);
  ASSERT_GT(helper, 0);
  const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
  ASSERT_GE(pidfd, 0);
  ASSERT_EQ(kill(helper, SIGKILL), 0);
  pollfd exited{pidfd, POLLIN, 0};
  ASSERT_EQ(poll(&exited, 1, 2000), 1);
  close(pidfd);
  close(token);

  // Use the owner's own UID and GID so the process identity must distinguish
  // this listener from the helper that sent the ready packet.
  const int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  ASSERT_GE(listener, 0);
  ASSERT_GT(private_address_size, 0u);
  ASSERT_EQ(
      bind(listener, reinterpret_cast<const sockaddr *>(&private_address), private_address_size),
      0);
  ASSERT_EQ(listen(listener, 2), 0);
  if (recycle_credentials) {
    // Exercise PID reuse without permission to set ns_last_pid, and force
    // the private proof path for the impostor and the genuine replacement.
    recycled_pid = helper;
    rocjitsu::libc_passthrough().getsockopt = report_recycled_peer;
    rocjitsu::libc_passthrough().pidfd_open = [](pid_t, unsigned) {
      errno = EMFILE;
      return -1;
    };
  }
  const int file = memfd_create("rj-rebound", MFD_CLOEXEC);
  ASSERT_GE(file, 0);

  if (prepare_first) {
    ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
    close(listener);
    // A genuine replacement must already hold the original file limit.
    const rlimit zero{};
    ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
    EXPECT_EQ(rocjitsu::populate_backing_file(file, "x", 1), 0);
  } else {
    int leaked_fd = -1;
    bool proof_requested = false;
    std::thread impostor([&] {
      pollfd waiting{listener, POLLIN, 0};
      if (poll(&waiting, 1, 2000) != 1)
        return;
      const int peer = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
      ASSERT_GE(peer, 0);
      rocjitsu::kmd::KeeperRequest request{};
      if (rocjitsu::kmd::receive_keeper_descriptor(peer, &request, sizeof(request), leaked_fd) >
          0) {
        if (recycle_credentials && leaked_fd < 0) {
          proof_requested = true;
          const std::array<uint64_t, 2> wrong_cookie{};
          rocjitsu::kmd::send_keeper_descriptor(peer, &wrong_cookie, sizeof(wrong_cookie), -1);
          // The owner must reject the proof before sending a file.
          rocjitsu::kmd::receive_keeper_descriptor(peer, &request, sizeof(request), leaked_fd);
        } else {
          const int error = EACCES;
          rocjitsu::kmd::send_keeper_descriptor(peer, &error, sizeof(error), -1);
        }
      }
      close(peer);
    });
    const int result = rocjitsu::populate_backing_file(file, "x", 1);
    impostor.join();
    EXPECT_EQ(proof_requested, recycle_credentials);
    EXPECT_EQ(leaked_fd, -1);
    if (leaked_fd >= 0)
      close(leaked_fd);
    EXPECT_EQ(result, 0);
    close(listener);
  }
  char value = 0;
  EXPECT_EQ(pread(file, &value, 1, 0), 1);
  EXPECT_EQ(value, 'x');
  close(file);
}

TEST_F(MemoryBackingDeathTest, ReboundListenerCannotReceiveBackingDescriptors) {
  ASSERT_EXIT(
      {
        alarm(10);
        check_rebound_keeper(false);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(MemoryKeeperProtocolTest, ReceivesCredentialsAndDescriptorInOnePacket) {
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets), 0);
  const int enabled = 1;
  ASSERT_EQ(setsockopt(sockets[0], SOL_SOCKET, SO_PASSCRED, &enabled, sizeof(enabled)), 0);
  const char sent = 'x';
  ASSERT_EQ(rocjitsu::kmd::send_keeper_descriptor(sockets[1], &sent, 1, sockets[0]), 0);
  char received = 0;
  int fd = -1;
  pid_t sender = -1;
  EXPECT_EQ(
      rocjitsu::kmd::receive_keeper_descriptor(sockets[0], &received, 1, fd, 0, nullptr, &sender),
      1);
  EXPECT_EQ(received, sent);
  EXPECT_EQ(sender, getpid());
  EXPECT_GE(fd, 0);
  if (fd >= 0)
    close(fd);
  close(sockets[1]);
  close(sockets[0]);
}

TEST_F(MemoryBackingDeathTest, RecycledCredentialsRequirePrivateProof) {
  ASSERT_EXIT(
      {
        alarm(10);
        check_rebound_keeper(false, true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, PreparationRejectsReboundListenerBeforeLimitsDrop) {
  ASSERT_EXIT(
      {
        alarm(10);
        check_rebound_keeper(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_full_rebound_queue(bool prepare_first) {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  rocjitsu::libc_passthrough().connect = capture_private_address;
  const int token = rocjitsu::create_memory_backing(1);
  ASSERT_GE(token, 0);
  const pid_t helper = token_keeper(token);
  ASSERT_GT(helper, 0);
  const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
  ASSERT_GE(pidfd, 0);
  ASSERT_EQ(kill(helper, SIGKILL), 0);
  pollfd exited{pidfd, POLLIN, 0};
  ASSERT_EQ(poll(&exited, 1, 2000), 1);
  close(pidfd);
  close(token);

  const auto *address = reinterpret_cast<const sockaddr *>(&private_address);
  const int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  ASSERT_GE(listener, 0);
  ASSERT_EQ(bind(listener, address, private_address_size), 0);
  ASSERT_EQ(listen(listener, 0), 0);
  // Linux accepts one connection beyond the requested backlog.
  const int filler = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  ASSERT_GE(filler, 0);
  ASSERT_EQ(connect(filler, address, private_address_size), 0);

  if (prepare_first) {
    ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  }
  const int replacement = rocjitsu::create_memory_backing(1);
  ASSERT_GE(replacement, 0) << std::strerror(errno);
  EXPECT_NE(token_keeper(replacement), helper);
  close(replacement);
  close(filler);
  close(listener);
}

TEST_F(MemoryBackingDeathTest, FullReboundListenerCannotBlockKeeperReplacement) {
  ASSERT_EXIT(
      {
        alarm(6);
        check_full_rebound_queue(false);
        check_full_rebound_queue(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

ssize_t flush_empty_batch(int fd) {
  using namespace rocjitsu::kmd;
  const KeeperRequest flush{0, O_RDWR, KeeperOperation::FlushFiles, {}};
  if (send(fd, &flush, sizeof(flush), MSG_NOSIGNAL) != sizeof(flush))
    return -1;
  KeeperFileResults results{};
  return recv(fd, &results, sizeof(results), 0);
}

void check_retired_channels(std::span<const int> channels, int token) {
  for (const int channel : channels) {
    // An inherited idle channel must receive EOF without sending another request.
    pollfd peer{channel, POLLRDHUP, 0};
    ASSERT_EQ(poll(&peer, 1, 2000), 1);
    EXPECT_NE(peer.revents & POLLRDHUP, 0);
    char byte;
    EXPECT_EQ(recv(channel, &byte, 1, MSG_DONTWAIT), 0);
  }
  const int reopened = rocjitsu::reopen_memory_backing(token, O_RDONLY | O_CLOEXEC);
  ASSERT_GE(reopened, 0) << strerror(errno);
  EXPECT_EQ(token_keeper(reopened), token_keeper(token));
  close(reopened);
}

void check_private_channel_retirement() {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  rocjitsu::libc_passthrough().connect = capture_private_address;
  const int token = rocjitsu::create_memory_backing(1);
  ASSERT_GE(token, 0);
  ASSERT_GT(private_address_size, 0u);
  int channels[2];
  for (int &channel : channels) {
    channel = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    ASSERT_GE(channel, 0);
    const timeval timeout{2, 0};
    ASSERT_EQ(setsockopt(channel, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
    ASSERT_EQ(::connect(channel, reinterpret_cast<const sockaddr *>(&private_address),
                        private_address_size),
              0);
    // Confirm acceptance before exec closes the owner's lease.
    ASSERT_EQ(flush_empty_batch(channel), sizeof(rocjitsu::kmd::KeeperFileResults));
  }

  int result[2];
  ASSERT_EQ(pipe2(result, O_CLOEXEC), 0);
  const pid_t holder = fork();
  ASSERT_GE(holder, 0);
  if (holder == 0) {
    close(result[0]);
    alarm(10);
    check_retired_channels(channels, token);
    _exit(!::testing::Test::HasFailure() && write(result[1], "ok\n", 3) == 3 ? 0 : 1);
  }
  // Keep the same owner PID alive across exec until the holder delivers its verdict.
  close(result[1]);
  ASSERT_EQ(dup2(result[0], STDIN_FILENO), STDIN_FILENO);
  close(result[0]);
  execl("/bin/sh", "sh", "-c", "read result && test \"$result\" = ok", nullptr);
  FAIL() << strerror(errno);
}

TEST_F(MemoryBackingDeathTest, ExecRetiresInheritedPrivateChannelWhileTokenStaysLive) {
  ASSERT_EXIT(
      {
        alarm(15);
        check_private_channel_retirement();
        _exit(1);
      },
      ::testing::ExitedWithCode(0), "");
}

pid_t paused_metadata_writer = -1;

ssize_t resume_at_batch_flush(int socket, const msghdr *message, int flags) {
  auto *request = static_cast<const rocjitsu::kmd::KeeperRequest *>(message->msg_iov[0].iov_base);
  if (request->operation == rocjitsu::kmd::KeeperOperation::BatchFile)
    kill(paused_metadata_writer, SIGSTOP);
  if (request->operation == rocjitsu::kmd::KeeperOperation::FlushFiles) {
    kill(paused_metadata_writer, SIGCONT);
    rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  }
  return ::sendmsg(socket, message, flags);
}

void check_failed_alias_truncation() {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  const int token = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(token, 0);
  paused_metadata_writer = token_keeper(token);
  ASSERT_GT(paused_metadata_writer, 0);
  const rlimit limit{64, 64};
  ASSERT_EQ(prlimit(paused_metadata_writer, RLIMIT_FSIZE, &limit, nullptr), 0);

  char path[] = "/tmp/rocjitsu-failed-alias-XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  const std::string alias = std::string(path) + ".link";
  ASSERT_EQ(link(path, alias.c_str()), 0);
  rocjitsu::BackingFile files[] = {{path, "first"}, {alias, std::string(65, 'x')}};
  rocjitsu::libc_passthrough().sendmsg = resume_at_batch_flush;
  rocjitsu::create_backing_files(files);
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  kill(paused_metadata_writer, SIGCONT);
  EXPECT_EQ(files[0].error, 0);
  EXPECT_EQ(files[1].error, EFBIG);
  struct stat info {};
  ASSERT_EQ(fstat(fd, &info), 0);
  EXPECT_EQ(info.st_size, 0);
  close(fd);
  close(token);
  EXPECT_EQ(unlink(alias.c_str()), 0);
  EXPECT_EQ(unlink(path), 0);
}

TEST_F(MemoryBackingDeathTest, FailedMetadataAliasKeepsItsTruncation) {
  ASSERT_EXIT(
      {
        check_failed_alias_truncation();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_empty_metadata_destinations() {
  char directory[] = "/tmp/rocjitsu-empty-metadata-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const std::string path = std::string(directory) + "/file";
  const std::string alias = std::string(directory) + "/alias";
  int fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  close(fd);
  ASSERT_EQ(link(path.c_str(), alias.c_str()), 0);
  rocjitsu::BackingFile aliases[] = {{path, "nonempty"}, {alias, ""}};
  rocjitsu::create_backing_files(aliases);
  EXPECT_EQ(aliases[0].error, 0);
  EXPECT_EQ(aliases[1].error, 0);
  struct stat info {};
  ASSERT_EQ(stat(path.c_str(), &info), 0);
  EXPECT_EQ(info.st_size, 0);
  ASSERT_EQ(unlink(alias.c_str()), 0);
  ASSERT_EQ(unlink(path.c_str()), 0);

  rlimit initial{};
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &initial), 0);
  rlimit one_free = initial;
  one_free.rlim_cur = 0;
  while (fcntl(one_free.rlim_cur++, F_GETFD) >= 0) {
  }
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &one_free), 0);
  rocjitsu::create_backing_files(std::span(aliases).last(1));
  EXPECT_EQ(aliases[1].error, 0);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &initial), 0);
  EXPECT_EQ(unlink(alias.c_str()), 0);
  EXPECT_EQ(rmdir(directory), 0);
}

TEST_F(MemoryBackingDeathTest, EmptyMetadataTruncatesAliasesAndNeedsOnlyOneFd) {
  ASSERT_EXIT(
      {
        check_empty_metadata_destinations();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_preparation_with_stopped_keeper(bool simulate_old_kernel) {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  if (simulate_old_kernel) {
    // A non-pidfs kernel reports no unique inode on the startup channel.
    rocjitsu::libc_passthrough().recvmsg = [](int fd, msghdr *message, int flags) -> ssize_t {
      rocjitsu::libc_passthrough().recvmsg = ::recvmsg;
      const ssize_t size = ::recvmsg(fd, message, flags);
      if (size == sizeof(rocjitsu::kmd::KeeperStartupReply))
        static_cast<rocjitsu::kmd::KeeperStartupReply *>(message->msg_iov[0].iov_base)
            ->process_inode = 0;
      return size;
    };
  }
  const int token = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(token, 0);
  const pid_t helper = token_keeper(token);
  ASSERT_GT(helper, 0);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  ASSERT_EQ(kill(helper, SIGSTOP), 0);
  // Let a hung probe finish so the test can report a lost lease.
  std::thread resume([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    kill(helper, SIGCONT);
  });
  const int prepared = rocjitsu::prepare_memory_keeper();
  const int error = errno;
  resume.join();
  if (prepared < 0) {
    // Without pidfs a stopped keeper cannot prove its identity until resumed.
    EXPECT_EQ(error, EAGAIN);
    ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  } else {
    EXPECT_FALSE(simulate_old_kernel);
  }
  const int file = memfd_create("rj-live-keeper", MFD_CLOEXEC);
  ASSERT_GE(file, 0);
  EXPECT_EQ(rocjitsu::populate_backing_file(file, "x", 1), 0);
  close(file);
  close(token);
}

TEST_F(MemoryBackingDeathTest, StoppedKeeperRetainsOriginalFileLimit) {
  ASSERT_EXIT(
      {
        check_preparation_with_stopped_keeper(false);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, StoppedKeeperRetainsOriginalFileLimitWithoutPidfs) {
  ASSERT_EXIT(
      {
        check_preparation_with_stopped_keeper(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_live_keeper_rejects_probe() {
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int file = memfd_create("rj-rejected-probe", MFD_CLOEXEC);
  ASSERT_GE(file, 0);

  // An overloaded keeper may accept and close both the request and the
  // health-check connection before their first sends.
  static unsigned attempts = 0;
  rocjitsu::libc_passthrough().sendmsg = [](int fd, const msghdr *message, int flags) -> ssize_t {
    if (is_identify_request(message))
      return ::sendmsg(fd, message, flags);
    ++attempts;
    errno = EPIPE;
    return -1;
  };
  EXPECT_EQ(rocjitsu::populate_backing_file(file, "x", 1), -1);
  EXPECT_EQ(errno, EPIPE);
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  EXPECT_EQ(attempts, 2u);
  // Only the original keeper retained the file limit needed for this write.
  EXPECT_EQ(rocjitsu::populate_backing_file(file, "y", 1), 0);
  close(file);
}

TEST_F(MemoryBackingDeathTest, RejectedProbePreservesPreparedFileLimit) {
  ASSERT_EXIT(
      {
        check_live_keeper_rejects_probe();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

int allocation_channel = -1;

ssize_t retain_allocation_channel(int socket, const msghdr *message, int flags) {
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  allocation_channel = fcntl(socket, F_DUPFD_CLOEXEC, 0);
  return ::sendmsg(socket, message, flags);
}

void check_keeper_descriptor_exhaustion() {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  rocjitsu::libc_passthrough().sendmsg = retain_allocation_channel;
  const int token = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(token, 0);
  const pid_t helper = token_keeper(token);
  ASSERT_GT(helper, 0);
  // The SCM_RIGHTS result can arrive before the helper drops its temporary
  // descriptors. Its close of this one-shot connection completes the request.
  ASSERT_GE(allocation_channel, 0);
  pollfd finished{allocation_channel, POLLRDHUP, 0};
  ASSERT_EQ(poll(&finished, 1, 2000), 1);
  EXPECT_NE(finished.revents & POLLRDHUP, 0);
  close(allocation_channel);
  rlimit original{};
  ASSERT_EQ(prlimit(helper, RLIMIT_NOFILE, nullptr, &original), 0);
  rlimit one_free = original;
  // The existing token keeps this helper alive, so its fd table is stable.
  const std::string prefix = "/proc/" + std::to_string(helper) + "/fd/";
  struct stat info {};
  one_free.rlim_cur = 0;
  while (lstat((prefix + std::to_string(one_free.rlim_cur++)).c_str(), &info) == 0) {
  }
  ASSERT_EQ(errno, ENOENT);
  char path[] = "/tmp/rocjitsu-keeper-nofile-XXXXXX";
  const int file = mkstemp(path);
  ASSERT_GE(file, 0);
  close(file);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  ASSERT_EQ(prlimit(helper, RLIMIT_NOFILE, &one_free, nullptr), 0);
  rocjitsu::BackingFile files[] = {{path, "data"}};
  rocjitsu::create_backing_files(files);
  EXPECT_EQ(files[0].error, EMFILE);
  ASSERT_EQ(prlimit(helper, RLIMIT_NOFILE, &original, nullptr), 0);
  rocjitsu::create_backing_files(files);
  EXPECT_EQ(files[0].error, 0);
  ASSERT_EQ(stat(path, &info), 0);
  EXPECT_EQ(info.st_size, 4);
  EXPECT_EQ(unlink(path), 0);
  close(token);
}

TEST_F(MemoryBackingDeathTest, FullHelperFdTablePreservesPreparedFileLimit) {
  ASSERT_EXIT(
      {
        check_keeper_descriptor_exhaustion();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_prepare_after_keeper_loss(bool fork_during_setup = false) {
  if (fork_during_setup) {
    ASSERT_EQ(pipe2(bootstrap_child_lifetime, O_CLOEXEC), 0);
    rocjitsu::libc_passthrough().socketpair = fork_during_bootstrap;
  }
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  const int token = rocjitsu::create_memory_backing(kSize);
  if (fork_during_setup) {
    rocjitsu::libc_passthrough().socketpair = ::socketpair;
    ASSERT_GT(bootstrap_child, 0);
  }
  ASSERT_GE(token, 0);
  const pid_t helper = token_keeper(token);
  ASSERT_GT(helper, 0);
  const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
  ASSERT_GE(pidfd, 0);
  ASSERT_EQ(kill(helper, SIGKILL), 0);
  pollfd exited{pidfd, POLLIN, 0};
  ASSERT_EQ(poll(&exited, 1, 2000), 1);
  close(pidfd);
  close(token);

  ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  if (fork_during_setup) {
    close(bootstrap_child_lifetime[1]);
    close(bootstrap_child_lifetime[0]);
    int status;
    ASSERT_EQ(waitpid(bootstrap_child, &status, 0), bootstrap_child);
    ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  const int fd = memfd_create("metadata", MFD_CLOEXEC);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(rocjitsu::populate_backing_file(fd, "abc", 3), 0);
  close(fd);
}

TEST_F(MemoryBackingDeathTest, PreparationReplacesDeadKeeperBeforeLimitsDrop) {
  ASSERT_EXIT(
      {
        check_prepare_after_keeper_loss();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, PreparationReplacesDeadKeeperAfterConcurrentFork) {
  ASSERT_EXIT(
      {
        alarm(20);
        check_prepare_after_keeper_loss(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

pid_t metadata_keeper_to_kill = -1;
unsigned requests_until_loss = 0;

ssize_t interrupt_metadata_batch(int socket, const msghdr *message, int flags) {
  if (metadata_keeper_to_kill > 0 && !is_identify_request(message) && --requests_until_loss == 0) {
    kill(metadata_keeper_to_kill, SIGKILL);
    metadata_keeper_to_kill = -1;
  }
  return ::sendmsg(socket, message, flags);
}

void check_metadata_batch_recovery(bool after_flush) {
  ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
  const int retained = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(retained, 0);
  metadata_keeper_to_kill = token_keeper(retained);
  ASSERT_GT(metadata_keeper_to_kill, 0);
  char directory[] = "/tmp/rocjitsu-metadata-recovery-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  std::vector<rocjitsu::BackingFile> files;
  for (unsigned file_index = 0; file_index < 19; ++file_index)
    files.push_back({std::string(directory) + "/" + std::to_string(file_index),
                     std::string(100, static_cast<char>('a' + file_index))});
  // One full batch and its flush are acknowledged. The next
  // helper inherits the lowered hard limit and cannot repeat those writes.
  requests_until_loss = after_flush ? rocjitsu::kmd::kKeeperMaxBatchFiles + 2 : 3;
  if (after_flush) {
    const rlimit zero{};
    ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  }
  rocjitsu::libc_passthrough().sendmsg = interrupt_metadata_batch;
  rocjitsu::create_backing_files(files);
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  EXPECT_EQ(metadata_keeper_to_kill, -1);
  for (size_t file_index = 0; file_index < files.size(); ++file_index) {
    const rocjitsu::BackingFile &file = files[file_index];
    const bool too_large = after_flush && file_index >= rocjitsu::kmd::kKeeperMaxBatchFiles;
    EXPECT_EQ(file.error, too_large ? EFBIG : 0) << file_index;
    const int fd = open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    std::string contents(file.contents.size(), '\0');
    ASSERT_EQ(read(fd, contents.data(), contents.size()), too_large ? 0 : contents.size());
    if (!too_large) {
      EXPECT_EQ(contents, file.contents);
    }
    close(fd);
    EXPECT_EQ(unlink(file.path.c_str()), 0);
  }
  EXPECT_EQ(rmdir(directory), 0);
  close(retained);
}

TEST_F(MemoryBackingDeathTest, MetadataBatchRetriesAfterHelperLoss) {
  ASSERT_EXIT(
      {
        check_metadata_batch_recovery(false);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, MetadataBatchKeepsAcknowledgedFilesAfterHelperLoss) {
  ASSERT_EXIT(
      {
        check_metadata_batch_recovery(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

pid_t metadata_child = -1;

ssize_t fork_with_batch_socket(int socket, const msghdr *message, int flags) {
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  metadata_child = fork();
  if (metadata_child == 0) {
    // A close in the parent alone cannot deliver EOF while we own this copy.
    pollfd peer{socket, POLLRDHUP, 0};
    _exit(poll(&peer, 1, 2000) == 1 && (peer.revents & POLLRDHUP) ? 0 : 1);
  }
  return ::sendmsg(socket, message, flags);
}

void check_fork_during_metadata_batch() {
  const int warmup = rocjitsu::create_memory_backing(1);
  ASSERT_GE(warmup, 0);
  close(warmup);
  char path[] = "/tmp/rocjitsu-batch-fork-XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  close(fd);
  rocjitsu::BackingFile file{path, "data"};
  rocjitsu::libc_passthrough().sendmsg = fork_with_batch_socket;
  rocjitsu::create_backing_files({&file, 1});
  EXPECT_EQ(unlink(path), 0);
  EXPECT_EQ(file.error, 0);
  ASSERT_GT(metadata_child, 0);
  int status = 0;
  ASSERT_EQ(waitpid(metadata_child, &status, 0), metadata_child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(MemoryBackingDeathTest, MetadataBatchReleasesConnectionAcrossFork) {
  ASSERT_EXIT(
      {
        check_fork_during_metadata_batch();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

thread_local bool delay_allocation_send = false;
std::promise<void> allocation_send_entered;
std::atomic<bool> release_allocation_send{false};

ssize_t send_after_release(int socket, const msghdr *message, int flags) {
  if (delay_allocation_send) {
    delay_allocation_send = false;
    allocation_send_entered.set_value();
    release_allocation_send.wait(false);
  }
  return ::sendmsg(socket, message, flags);
}

void check_delayed_allocation() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int retained = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(retained, 0);
  rocjitsu::libc_passthrough().sendmsg = send_after_release;
  auto allocation = std::async(std::launch::async, [] {
    delay_allocation_send = true;
    return rocjitsu::create_memory_backing(kSize);
  });
  if (allocation_send_entered.get_future().wait_for(std::chrono::seconds(2)) !=
      std::future_status::ready) {
    release_allocation_send.store(true);
    release_allocation_send.notify_all();
    const int result = allocation.get();
    rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
    ADD_FAILURE() << "Allocation did not reach sendmsg (result " << result << ")";
    close(result);
    close(retained);
    return;
  }
  // The allocation has connected but deliberately sent nothing. The helper
  // must keep serving existing tokens instead of blocking on that connection.
  auto reopened = std::async(std::launch::async,
                             [&] { return rocjitsu::reopen_memory_backing(retained, O_RDONLY); });
  const auto status = reopened.wait_for(std::chrono::seconds(2));
  release_allocation_send.store(true);
  release_allocation_send.notify_all();
  const int next = allocation.get();
  const int alias = reopened.get();
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_GE(next, 0);
  EXPECT_GE(alias, 0);
  close(next);
  close(alias);
  close(retained);
}

TEST_F(MemoryBackingDeathTest, DelayedAllocationDoesNotBlockTokenReopens) {
  ASSERT_EXIT(
      {
        check_delayed_allocation();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

bool truncated_reply = false;

ssize_t receive_with_full_descriptor_table(int socket, msghdr *message, int flags) {
  if (message->msg_iov[0].iov_len == sizeof(rocjitsu::kmd::KeeperCookie))
    return ::recvmsg(socket, message, flags);
  rocjitsu::libc_passthrough().recvmsg = ::recvmsg;
  rlimit original{};
  if (getrlimit(RLIMIT_NOFILE, &original) != 0)
    return -1;
  rlimit limited = original;
  limited.rlim_cur = std::min<rlim_t>(original.rlim_cur, 256);
  if (setrlimit(RLIMIT_NOFILE, &limited) != 0)
    return -1;
  std::vector<int> fillers;
  int filler;
  while ((filler = open("/dev/null", O_RDONLY | O_CLOEXEC)) >= 0)
    fillers.push_back(filler);
  const ssize_t received = recvmsg(socket, message, flags);
  const int error = errno;
  truncated_reply = (message->msg_flags & MSG_CTRUNC) != 0;
  for (int descriptor : fillers)
    close(descriptor);
  EXPECT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  errno = error;
  return received;
}

TEST_F(MemoryBackingDeathTest, EmptyTokenPacketReleasesReceivedDescriptor) {
  ASSERT_EXIT(
      {
        ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", "1", 1), 0);
        const int token = rocjitsu::create_memory_backing(kSize);
        ASSERT_GE(token, 0);
        int pipe_fds[2];
        ASSERT_EQ(pipe2(pipe_fds, O_CLOEXEC), 0);
        ASSERT_EQ(rocjitsu::kmd::send_keeper_descriptor(token, nullptr, 0, pipe_fds[1]), 0);
        close(pipe_fds[1]);
        pollfd reader{};
        reader.fd = pipe_fds[0];
        reader.events = POLLIN;
        ASSERT_EQ(poll(&reader, 1, 2000), 1);
        EXPECT_EQ(reader.revents, POLLHUP);
        close(pipe_fds[0]);
        close(token);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, MetadataPopulationNeedsNoReplyDescriptor) {
  ASSERT_EXIT(
      {
        const int fd = rocjitsu::create_memory_backing(1);
        ASSERT_GE(fd, 0);
        const int metadata = memfd_create("metadata", MFD_CLOEXEC);
        ASSERT_GE(metadata, 0);
        rocjitsu::libc_passthrough().recvmsg = receive_with_full_descriptor_table;
        ASSERT_EQ(rocjitsu::populate_backing_file(metadata, "data", 4), 0);
        ASSERT_FALSE(truncated_reply);
        char bytes[4];
        ASSERT_EQ(pread(metadata, bytes, sizeof(bytes), 0), sizeof(bytes));
        EXPECT_EQ(std::memcmp(bytes, "data", sizeof(bytes)), 0);
        close(metadata);
        close(fd);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

ssize_t send_unterminated_backing_name(int socket, const msghdr *message, int flags) {
  if (is_identify_request(message))
    return ::sendmsg(socket, message, flags);
  rocjitsu::libc_passthrough().sendmsg = ::sendmsg;
  auto request = *static_cast<rocjitsu::kmd::KeeperRequest *>(message->msg_iov[0].iov_base);
  std::memset(request.name, 'x', sizeof(request.name));
  return ::send(socket, &request, sizeof(request), flags);
}

TEST_F(MemoryBackingDeathTest, RejectsUnterminatedPrivateBackingNames) {
  ASSERT_EXIT(
      {
        ASSERT_EQ(rocjitsu::prepare_memory_keeper(), 0);
        for (int force_token = 0; force_token != 2; ++force_token) {
          ASSERT_EQ(setenv("RJ_TEST_FORCE_TOKEN_BACKING", force_token ? "1" : "0", 1), 0);
          rocjitsu::libc_passthrough().sendmsg = send_unterminated_backing_name;
          EXPECT_EQ(rocjitsu::create_memory_backing(kSize), -1);
          EXPECT_EQ(errno, EINVAL);
        }
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_truncated_backing_reply() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int retained = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(retained, 0);
  const pid_t helper = token_keeper(retained);
  rocjitsu::libc_passthrough().recvmsg = receive_with_full_descriptor_table;
  const int failed = rocjitsu::create_memory_backing(kSize);
  const int error = errno;
  rocjitsu::libc_passthrough().recvmsg = ::recvmsg;
  EXPECT_TRUE(truncated_reply);
  EXPECT_EQ(failed, -1);
  EXPECT_EQ(error, EPROTO);
  if (failed >= 0)
    close(failed);
  const int next = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(next, 0);
  EXPECT_EQ(token_keeper(next), helper);
  const int alias = rocjitsu::reopen_memory_backing(retained, O_RDONLY);
  ASSERT_GE(alias, 0);
  close(alias);
  close(next);
  close(retained);
}

TEST_F(MemoryBackingDeathTest, TruncatedDescriptorReplyPreservesKeeper) {
  ASSERT_EXIT(
      {
        check_truncated_backing_reply();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

int observed_lease_fd = -1;

int record_owner_descriptor(const char *name, unsigned flags) {
  const int fd = memfd_create(name, flags);
  if (fd >= 0 && std::strcmp(name, "rocjitsu-owner") == 0) {
    observed_lease_fd = fd;
  }
  return fd;
}

void check_retirement_descriptor_reuse() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  rocjitsu::libc_passthrough().memfd_create = record_owner_descriptor;
  const int initial = rocjitsu::create_memory_backing(kSize);
  rocjitsu::libc_passthrough().memfd_create = ::memfd_create;
  ASSERT_GE(initial, 0);
  ASSERT_GE(observed_lease_fd, 0);
  const int retained = fcntl(initial, F_DUPFD_CLOEXEC, 256);
  ASSERT_GE(retained, 0);
  close(initial);
  const pid_t helper = token_keeper(retained);
  const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
  ASSERT_GE(pidfd, 0);
  ASSERT_EQ(kill(helper, SIGKILL), 0);
  pollfd exited{pidfd, POLLIN, 0};
  ASSERT_EQ(poll(&exited, 1, 2000), 1);
  close(pidfd);
  const int replacement = memfd_create("lease-replacement", MFD_CLOEXEC);
  ASSERT_GE(replacement, 0);
  const int lease_replacement_fd = fcntl(replacement, F_DUPFD_CLOEXEC, 256);
  ASSERT_GE(lease_replacement_fd, 0);
  close(replacement);
  struct stat expected {
  }, current{};
  ASSERT_EQ(fstat(lease_replacement_fd, &expected), 0);
  ASSERT_EQ(dup2(lease_replacement_fd, observed_lease_fd), observed_lease_fd);
  const int next = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(next, 0);
  EXPECT_NE(token_keeper(next), helper);
  ASSERT_EQ(fstat(observed_lease_fd, &current), 0);
  EXPECT_EQ(current.st_ino, expected.st_ino);
  close(next);
  close(retained);
  close(lease_replacement_fd);
  close(observed_lease_fd);
}

TEST_F(MemoryBackingDeathTest, RecoveryPreservesPreviouslyReusedDescriptor) {
  ASSERT_EXIT(
      {
        check_retirement_descriptor_reuse();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, KeeperLossRejectsRetainedTokens) {
  ASSERT_EXIT(
      {
        const rlimit zero{};
        ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
        const int fd = rocjitsu::create_memory_backing(kSize);
        ASSERT_GE(fd, 0);
        auto *view = static_cast<uint32_t *>(rocjitsu::map_memory_backing(
            nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        ASSERT_NE(view, MAP_FAILED);
        view[0] = 0x12345678;
        ASSERT_EQ(munmap(view, kSize), 0);
        const int retained_fd = rocjitsu::create_memory_backing(kSize);
        ASSERT_GE(retained_fd, 0);
        auto *retained = static_cast<uint32_t *>(rocjitsu::map_memory_backing(
            nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, retained_fd, 0));
        ASSERT_NE(retained, MAP_FAILED);
        retained[0] = 0x87654321;
        const pid_t helper = token_keeper(fd);
        ASSERT_EQ(token_keeper(retained_fd), helper);
        const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
        ASSERT_GE(pidfd, 0);
        ASSERT_EQ(kill(helper, SIGKILL), 0);
        pollfd exited{};
        exited.fd = pidfd;
        exited.events = POLLIN;
        ASSERT_EQ(poll(&exited, 1, 2000), 1);
        EXPECT_EQ(rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, fd, 0),
                  MAP_FAILED);
        EXPECT_EQ(errno, ESTALE);
        EXPECT_EQ(rocjitsu::reopen_memory_backing(fd, O_RDONLY), -1);
        EXPECT_EQ(errno, ESTALE);
        EXPECT_EQ(retained[0], 0x87654321u);
        EXPECT_EQ(
            rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, retained_fd, 0),
            MAP_FAILED);
        EXPECT_EQ(errno, ESTALE);
        munmap(retained, kSize);
        close(retained_fd);
        close(pidfd);
        close(fd);
        const int replacement = rocjitsu::create_memory_backing(kSize);
        ASSERT_GE(replacement, 0);
        EXPECT_NE(token_keeper(replacement), helper);
        close(replacement);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, ChangedEffectiveUidRechecksTokenMappingPermissions) {
  uid_t real, effective, saved;
  ASSERT_EQ(getresuid(&real, &effective, &saved), 0);
  uid_t alternate = real != effective ? real : saved;
  if (alternate == effective) {
    if (effective != 0)
      GTEST_SKIP() << "requires root or a distinct real/saved UID";
    alternate = 65534;
  }
  ASSERT_EXIT(
      {
        const rlimit zero{};
        ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
        const int fd = rocjitsu::create_memory_backing(kSize);
        ASSERT_GE(fd, 0);
        auto *existing = static_cast<uint32_t *>(rocjitsu::map_memory_backing(
            nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        ASSERT_NE(existing, MAP_FAILED);
        existing[0] = 0x12345678;
        ASSERT_EQ(seteuid(alternate), 0);
        void *denied = rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, fd, 0);
        const int error = errno;
        const uint32_t retained = existing[0];
        ASSERT_EQ(seteuid(effective), 0);
        EXPECT_EQ(denied, MAP_FAILED);
        EXPECT_EQ(error, EACCES);
        EXPECT_EQ(retained, 0x12345678u);
        void *restored = rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, fd, 0);
        ASSERT_NE(restored, MAP_FAILED);
        EXPECT_EQ(*static_cast<uint32_t *>(restored), 0x12345678u);
        munmap(restored, kSize);
        munmap(existing, kSize);
        close(fd);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

volatile sig_atomic_t child_signal = 0;
void observe_child_signal(int) { child_signal = 1; }

void check_helper_lifecycle(bool subreaper = false) {
  if (subreaper && getpid() != 1) {
    ASSERT_EQ(prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0), 0);
  }
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  struct sigaction action {};
  action.sa_handler = observe_child_signal;
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGCHLD, &action, nullptr), 0);
  const int fd = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(fd, 0);
  const pid_t helper = token_keeper(fd);
  ASSERT_GT(helper, 0);
  int status = 0;
  if (subreaper) {
    EXPECT_EQ(waitpid(-1, &status, WNOHANG), 0);
  } else {
    EXPECT_EQ(waitpid(-1, &status, WNOHANG), -1);
    EXPECT_EQ(errno, ECHILD);
  }
  EXPECT_EQ(child_signal, 0);

  rlimit original{}, limited{};
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &original), 0);
  limited = original;
  // Exhaust descriptors at connection creation without depending on the
  // transport's temporary descriptor count. Truncated replies are tested above.
  limited.rlim_cur = 0;
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &limited), 0);
  EXPECT_EQ(rocjitsu::create_memory_backing(kSize), -1);
  EXPECT_EQ(errno, EMFILE);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  int next = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(next, 0);
  EXPECT_EQ(token_keeper(next), helper);
  close(next);

  const int pidfd = rocjitsu::host_pidfd_open(helper, 0);
  ASSERT_GE(pidfd, 0);
  ASSERT_EQ(kill(helper, SIGKILL), 0);
  pollfd exited{pidfd, POLLIN, 0};
  ASSERT_EQ(poll(&exited, 1, 2000), 1);
  close(pidfd);
  close(fd);
  if (subreaper) {
    ASSERT_EQ(waitpid(helper, &status, 0), helper);
  }
  EXPECT_EQ(waitpid(-1, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
  EXPECT_EQ(child_signal, subreaper ? 1 : 0);
  next = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(next, 0) << strerror(errno);
  EXPECT_NE(token_keeper(next), helper);
  close(next);
}

TEST_F(MemoryBackingDeathTest, HelperDoesNotChangeChildWaitsAndRecoversAfterFailure) {
  ASSERT_EXIT(
      {
        check_helper_lifecycle();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_subreaper_warning() {
  // GTest captures stderr in a regular file, which cannot accept the warning
  // under hard FSIZE=0. A pipe observes it without relaxing the file limit.
  int diagnostics[2];
  ASSERT_EQ(pipe2(diagnostics, O_CLOEXEC), 0);
  ASSERT_GE(dup2(diagnostics[1], STDERR_FILENO), 0);
  close(diagnostics[1]);
  check_helper_lifecycle(true);
  close(STDERR_FILENO);
  std::string output;
  char buffer[1024];
  ssize_t bytes;
  while ((bytes = read(diagnostics[0], buffer, sizeof(buffer))) > 0)
    output.append(buffer, bytes);
  close(diagnostics[0]);
  const std::string warning = "Memory keeper child-wait isolation is unsupported for child "
                              "subreapers";
  const auto first = output.find(warning);
  ASSERT_NE(first, std::string::npos) << output;
  EXPECT_EQ(output.find(warning, first + warning.size()), std::string::npos) << output;
}

TEST_F(MemoryBackingDeathTest, SubreaperWarnsAboutChildWaitsAndRecoversAfterFailure) {
  ASSERT_EXIT(
      {
        check_subreaper_warning();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

int write_namespace_map(const char *path, const std::string &contents) {
  const int fd = open(path, O_WRONLY | O_CLOEXEC);
  if (fd < 0)
    return errno;
  const ssize_t written = write(fd, contents.data(), contents.size());
  const int error = written < 0                                        ? errno
                    : written == static_cast<ssize_t>(contents.size()) ? 0
                                                                       : EIO;
  close(fd);
  return error;
}

int enter_test_namespace(int flags) {
  if (unshare(flags) == 0)
    return 0;
  if (errno != EPERM && errno != EINVAL)
    return errno;
  const uid_t uid = geteuid();
  const gid_t gid = getegid();
  if (unshare(CLONE_NEWUSER | flags) != 0)
    return errno;
  // Give the execed keeper a valid owning UID/GID in the isolated namespace.
  int error = write_namespace_map("/proc/self/uid_map", "0 " + std::to_string(uid) + " 1\n");
  if (!error)
    error = write_namespace_map("/proc/self/setgroups", "deny\n");
  if (!error)
    error = write_namespace_map("/proc/self/gid_map", "0 " + std::to_string(gid) + " 1\n");
  return error;
}

void check_device_token_private_fallback() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const std::string name = std::string(rocjitsu::kmd::kDeviceBackingNamePrefix) + "129";
  const int backing = rocjitsu::create_memory_backing(kSize, false, name.c_str());
  ASSERT_GE(backing, 0) << strerror(errno);
  ASSERT_TRUE(rocjitsu::is_memory_backing_token(backing));
  EXPECT_TRUE(rocjitsu::kmd::is_device_memory_backing(backing));
  const int alias = rocjitsu::reopen_memory_backing(backing, O_RDONLY);
  ASSERT_GE(alias, 0);
  EXPECT_TRUE(rocjitsu::kmd::is_device_memory_backing(alias));
  auto *view = static_cast<unsigned char *>(
      rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0));
  ASSERT_NE(view, MAP_FAILED);
  auto *reader = static_cast<unsigned char *>(
      rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, alias, 0));
  ASSERT_NE(reader, MAP_FAILED);
  view[0] = 0xA5;
  EXPECT_EQ(reader[0], 0xA5);
  bool found = false;
  std::ifstream maps("/proc/self/maps");
  for (std::string line; std::getline(maps, line);) {
    unsigned long start = 0, end = 0;
    if (std::sscanf(line.c_str(), "%lx-%lx", &start, &end) == 2 &&
        start <= reinterpret_cast<uintptr_t>(view) && reinterpret_cast<uintptr_t>(view) < end) {
      EXPECT_NE(line.find("/SYSV00000000 (deleted)"), std::string::npos) << line;
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
  EXPECT_EQ(close(alias), 0);
  EXPECT_EQ(close(backing), 0);
  EXPECT_EQ(reader[0], 0xA5);
  EXPECT_EQ(munmap(reader, kSize), 0);
  EXPECT_EQ(munmap(view, kSize), 0);
}

void check_device_key_collisions() {
  // The execed keeper inherits this filter; unrelated shmget calls stay native.
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_shmget, 0, 4),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
      BPF_STMT(BPF_ALU | BPF_AND | BPF_K, ~(rocjitsu::kmd::kDeviceTokenSlotCount - 1)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, rocjitsu::kmd::kDeviceTokenKeyPrefix, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EEXIST),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(std::size(filter)), filter};
  ASSERT_EQ(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
  ASSERT_EQ(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program), 0);
  check_device_token_private_fallback();
}

void check_device_key_exhaustion() {
  const int error = enter_test_namespace(CLONE_NEWIPC);
  if (error == EPERM || error == EACCES || error == EINVAL || error == ENOSPC)
    GTEST_SKIP() << "IPC namespace isolation is unavailable: " << strerror(error);
  ASSERT_EQ(error, 0) << strerror(error);
  class OccupiedKeys {
  public:
    ~OccupiedKeys() {
      for (const int segment : segments)
        EXPECT_EQ(shmctl(segment, IPC_RMID, nullptr), 0);
    }
    std::vector<int> segments;
  } occupied;
  for (uint32_t slot = 0; slot < rocjitsu::kmd::kDeviceTokenSlotCount; ++slot) {
    const int segment = shmget(static_cast<key_t>(rocjitsu::kmd::kDeviceTokenKeyPrefix | slot),
                               kSize, IPC_CREAT | IPC_EXCL | 0600);
    ASSERT_GE(segment, 0) << strerror(errno);
    occupied.segments.push_back(segment);
  }
  check_device_token_private_fallback();
  for (const int segment : occupied.segments) {
    shmid_ds info{};
    EXPECT_EQ(shmctl(segment, IPC_STAT, &info), 0);
    EXPECT_EQ(info.shm_nattch, 0u) << "the keeper must not attach an occupied key";
  }
}

TEST_F(MemoryBackingDeathTest, DeviceKeyCollisionsFallBackToPrivateBacking) {
  // Exercise the fallback even on hosts that disallow IPC namespace creation.
  ASSERT_EXIT(
      {
        check_device_key_collisions();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, ExhaustedDeviceKeysUsePrivateBackingInIsolatedNamespace) {
  bool skipped = false;
  const auto completed_or_skipped = [&skipped](int status) {
    skipped = ::testing::ExitedWithCode(kPrerequisiteSkipExitCode)(status);
    return skipped || ::testing::ExitedWithCode(0)(status);
  };
  ASSERT_EXIT(
      {
        check_device_key_exhaustion();
        _exit(::testing::Test::HasFailure()  ? 1
              : ::testing::Test::IsSkipped() ? kPrerequisiteSkipExitCode
                                             : 0);
      },
      completed_or_skipped, "");
  if (skipped)
    GTEST_SKIP() << "Neither privileged nor user-namespace IPC isolation is available";
}

TEST(MemoryBackingNamespaceTest, NamespaceInitWarnsAndReapsKeepers) {
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    const int error = enter_test_namespace(CLONE_NEWPID);
    if (error)
      _exit(error == EPERM || error == EACCES || error == EINVAL || error == ENOSPC
                ? kPrerequisiteSkipExitCode
                : 1);
    const pid_t init = fork();
    if (init == 0) {
      check_subreaper_warning();
      _exit(::testing::Test::HasFailure() ? 1 : 0);
    }
    int status = 0;
    _exit(init > 0 && waitpid(init, &status, 0) == init && WIFEXITED(status) ? WEXITSTATUS(status)
                                                                             : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  if (WEXITSTATUS(status) == kPrerequisiteSkipExitCode)
    GTEST_SKIP() << "Neither privileged nor user-namespace PID isolation is available";
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

void check_group_signal() {
  ASSERT_GE(setsid(), 0);
  ASSERT_NE(signal(SIGINT, SIG_IGN), SIG_ERR);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int fd = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kill(0, SIGINT), 0);
  void *view =
      rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(view, MAP_FAILED);
  const int reopened = rocjitsu::reopen_memory_backing(fd, O_RDONLY);
  ASSERT_GE(reopened, 0);
  const int next = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(next, 0);
  EXPECT_EQ(token_keeper(next), token_keeper(fd));
  close(next);
  close(reopened);
  close(fd);
  munmap(view, kSize);
}

TEST_F(MemoryBackingDeathTest, GroupSignalsPreserveUnmappedTokens) {
  ASSERT_EXIT(
      {
        check_group_signal();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_lease_replacement() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  rocjitsu::libc_passthrough().memfd_create = record_owner_descriptor;
  int fd = rocjitsu::create_memory_backing(kSize);
  rocjitsu::libc_passthrough().memfd_create = ::memfd_create;
  ASSERT_GE(fd, 0);
  const pid_t first_helper = token_keeper(fd);
  const int lease = observed_lease_fd;
  ASSERT_GE(lease, 0);
  const int retained = fcntl(fd, F_DUPFD_CLOEXEC, 256);
  ASSERT_GE(retained, 0);
  close(fd);
  fd = retained;
  const int replacement = memfd_create("lease-replacement", MFD_CLOEXEC);
  ASSERT_GE(replacement, 0);
  const int replacement_copy = fcntl(replacement, F_DUPFD_CLOEXEC, 256);
  ASSERT_GE(replacement_copy, 0);
  struct stat before {
  }, after{};
  ASSERT_EQ(fstat(replacement, &before), 0);
  close(replacement);
  ASSERT_EQ(rocjitsu::host_close_range(lease, lease, 0), 0);
  ASSERT_EQ(dup2(replacement_copy, lease), lease);
  close(replacement_copy);
  const int next = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(next, 0);
  EXPECT_EQ(token_keeper(next), first_helper);
  ASSERT_EQ(fstat(lease, &after), 0);
  EXPECT_EQ(before.st_ino, after.st_ino);
  // Reusing the temporary lease fd cannot retire the helper or invalidate tokens.
  const int alias = rocjitsu::reopen_memory_backing(fd, O_RDONLY);
  ASSERT_GE(alias, 0);
  void *view = rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, alias, 0);
  ASSERT_NE(view, MAP_FAILED);
  munmap(view, kSize);
  close(alias);
  close(next);
  close(fd);
  close(lease);
}

TEST_F(MemoryBackingDeathTest, DescriptorReusePreservesKeeperAndTokens) {
  ASSERT_EXIT(
      {
        check_lease_replacement();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_credential_retirement(gid_t alternate) {
  const gid_t original = getegid();
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int retained = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(retained, 0);
  const int retained_pidfd = rocjitsu::host_pidfd_open(token_keeper(retained), 0);
  ASSERT_GE(retained_pidfd, 0);
  int previous_pidfd = -1;
  for (unsigned iteration = 0; iteration < 4; ++iteration) {
    ASSERT_EQ(setegid(iteration % 2 ? original : alternate), 0);
    const int next = rocjitsu::create_memory_backing(kSize);
    ASSERT_GE(next, 0) << strerror(errno);
    if (previous_pidfd >= 0) {
      pollfd exited{previous_pidfd, POLLIN, 0};
      EXPECT_EQ(poll(&exited, 1, 2000), 1) << "retired credential helper survived";
      close(previous_pidfd);
    }
    previous_pidfd = rocjitsu::host_pidfd_open(token_keeper(next), 0);
    ASSERT_GE(previous_pidfd, 0);
    close(next);
    unsigned leases = 0;
    for (int fd = 3; fd < 256; ++fd) {
      char path[64], target[128]{};
      std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
      if (readlink(path, target, sizeof(target) - 1) > 0 &&
          std::strstr(target, "memfd:rocjitsu-owner"))
        ++leases;
    }
    EXPECT_EQ(leases, 0u);
  }
  close(previous_pidfd);
  // Retiring the owner's lease must preserve independently exported tokens.
  pollfd retained_exited{retained_pidfd, POLLIN, 0};
  EXPECT_EQ(poll(&retained_exited, 1, 0), 0);
  const int alias = rocjitsu::reopen_memory_backing(retained, O_RDONLY);
  ASSERT_GE(alias, 0);
  void *view = rocjitsu::map_memory_backing(nullptr, kSize, PROT_READ, MAP_SHARED, alias, 0);
  ASSERT_NE(view, MAP_FAILED);
  EXPECT_EQ(munmap(view, kSize), 0);
  close(alias);
  close(retained);
  EXPECT_EQ(poll(&retained_exited, 1, 2000), 1);
  close(retained_pidfd);
}

TEST_F(MemoryBackingDeathTest, CredentialChangesRetireLeasesAndPreserveTokens) {
  gid_t real, effective, saved;
  ASSERT_EQ(getresgid(&real, &effective, &saved), 0);
  gid_t alternate = real != effective ? real : saved;
  if (alternate == effective) {
    if (geteuid() != 0)
      GTEST_SKIP() << "requires root or a distinct real/saved group (setgid test executable)";
    alternate = effective == 0 ? 1 : 0;
  }
  ASSERT_EXIT(
      {
        check_credential_retirement(alternate);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_exec_retirement() {
  int control[2], idle_lifetime[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, control), 0);
  ASSERT_EQ(pipe2(idle_lifetime, O_CLOEXEC), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(control[0]);
    close(idle_lifetime[1]);
    const rlimit zero{};
    if (setrlimit(RLIMIT_FSIZE, &zero) != 0)
      _exit(1);
    const int fd = rocjitsu::create_memory_backing(kSize);
    const pid_t helper = token_keeper(fd);
    if (fd < 0 || helper <= 0)
      _exit(2);
    close(fd);
    // An idle child must not retain the parent's lease across the parent's exec.
    // The private lease mapping is excluded from this fork with MADV_DONTFORK.
    const pid_t idle = fork();
    if (idle < 0)
      _exit(3);
    if (idle == 0) {
      close(control[1]);
      char done;
      _exit(read(idle_lifetime[0], &done, 1) == 0 ? 0 : 1);
    }
    close(idle_lifetime[0]);
    const pid_t children[] = {helper, idle};
    if (write(control[1], children, sizeof(children)) != sizeof(children))
      _exit(4);
    char go;
    if (read(control[1], &go, 1) != 1)
      _exit(5);
    execl("/bin/sleep", "sleep", "5", nullptr);
    _exit(6);
  }
  close(control[1]);
  close(idle_lifetime[0]);
  pid_t children[2]{};
  ASSERT_EQ(read(control[0], children, sizeof(children)), sizeof(children));
  const int pidfd = rocjitsu::host_pidfd_open(children[0], 0);
  const int idle_pidfd = rocjitsu::host_pidfd_open(children[1], 0);
  ASSERT_GE(pidfd, 0);
  ASSERT_GE(idle_pidfd, 0);
  ASSERT_EQ(write(control[0], "x", 1), 1);
  close(control[0]);
  pollfd exited{pidfd, POLLIN, 0};
  EXPECT_EQ(poll(&exited, 1, 2000), 1) << "helper survived its owner's exec";
  close(pidfd);
  pollfd idle_exited{idle_pidfd, POLLIN, 0};
  EXPECT_EQ(poll(&idle_exited, 1, 0), 0) << "forked descriptor holder must still be alive";
  close(idle_lifetime[1]);
  EXPECT_EQ(poll(&idle_exited, 1, 2000), 1);
  close(idle_pidfd);
  int status = 0;
  EXPECT_EQ(waitpid(child, &status, WNOHANG), 0) << "owner must still be alive";
  kill(child, SIGKILL);
  ASSERT_EQ(waitpid(child, &status, 0), child);
}

TEST_F(MemoryBackingDeathTest, ExecRetiresUnusedHelperWhileOwnerLives) {
  ASSERT_EXIT(
      {
        check_exec_retirement();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_nofile_headroom() {
  rlimit original{};
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &original), 0);
  ASSERT_GE(original.rlim_max, 256);
  rlimit limited = original;
  limited.rlim_cur = 64;
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &limited), 0);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  std::vector<int> descriptors{rocjitsu::create_memory_backing(kSize)};
  ASSERT_GE(descriptors.front(), 0);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  for (unsigned count = 1; count < 96; ++count) {
    const int fd = rocjitsu::create_memory_backing(kSize);
    ASSERT_GE(fd, 0) << count << ": " << strerror(errno);
    EXPECT_EQ(token_keeper(fd), token_keeper(descriptors.front()));
    descriptors.push_back(fd);
  }
  for (const int fd : descriptors)
    close(fd);
}

TEST_F(MemoryBackingDeathTest, HelperUsesHardDescriptorLimit) {
  ASSERT_EXIT(
      {
        check_nofile_headroom();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(MemoryBackingDeathTest, StartupFailureNamesHelper) {
  if (getuid() == 0)
    GTEST_SKIP() << "RLIMIT_NPROC does not constrain root";
  ASSERT_EXIT(
      {
        const rlimit zero{};
        if (setrlimit(RLIMIT_NPROC, &zero) != 0)
          _exit(2);
        const int fd = rocjitsu::create_memory_backing(kSize);
        _exit(fd == -1 && errno == EAGAIN ? 0 : 1);
      },
      ::testing::ExitedWithCode(0),
      "Cannot start memory keeper '[^']*rocjitsu-memory': Resource temporarily unavailable");
}

void check_seek_contract() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int fd = rocjitsu::create_memory_backing(kSize);
  ASSERT_GE(fd, 0);
  off_t result = -1;
  EXPECT_TRUE(rocjitsu::seek_memory_backing_token(fd, 0, SEEK_END, result));
  EXPECT_EQ(result, kSize);
  EXPECT_TRUE(rocjitsu::seek_memory_backing_token(fd, 0, SEEK_SET, result));
  EXPECT_EQ(result, 0);
  EXPECT_TRUE(rocjitsu::seek_memory_backing_token(fd, 1, SEEK_SET, result));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_TRUE(rocjitsu::seek_memory_backing_token(fd, 0, SEEK_CUR, result));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EINVAL);
  close(fd);
}

TEST_F(MemoryBackingDeathTest, TokenSupportsDmaBufSeekContract) {
  ASSERT_EXIT(
      {
        check_seek_contract();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

} // namespace
