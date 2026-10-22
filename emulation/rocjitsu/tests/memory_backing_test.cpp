// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/memory_backing.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Match the driver's compatibility with the older headers in the CI image.
#ifndef SYS_pidfd_open
#ifdef __NR_pidfd_open
#define SYS_pidfd_open __NR_pidfd_open
#else
#define SYS_pidfd_open 434
#endif
#endif

namespace {
namespace backing = rocjitsu::memory_backing;
constexpr size_t kSize = 64 * 1024;

void check_aliases() {
  const rlimit limit{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
  const int fd = backing::create(kSize);
  ASSERT_GE(fd, 0) << strerror(errno);
  auto *original = static_cast<unsigned char *>(
      backing::map(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  ASSERT_NE(original, MAP_FAILED);
  original[4096] = 0x42;
  const int readonly = backing::reopen(fd, O_RDONLY | O_CLOEXEC);
  ASSERT_GE(readonly, 0);
  EXPECT_EQ(backing::fcntl(readonly, F_GETFL) & O_ACCMODE, O_RDONLY);
  EXPECT_NE(fcntl(readonly, F_GETFD) & FD_CLOEXEC, 0);
  auto *alias = static_cast<unsigned char *>(
      backing::map(nullptr, 4096, PROT_READ, MAP_SHARED, readonly, 4096));
  ASSERT_NE(alias, MAP_FAILED);
  EXPECT_EQ(alias[0], 0x42);
  EXPECT_EQ(mprotect(alias, 4096, PROT_READ | PROT_WRITE), -1);
  EXPECT_EQ(errno, EACCES);
  EXPECT_EQ(backing::map(nullptr, 4096, PROT_WRITE, MAP_SHARED, readonly, 0), MAP_FAILED);
  EXPECT_EQ(errno, EACCES);
  struct stat before {
  }, after{};
  ASSERT_EQ(backing::stat(fd, &before), 0);
  ASSERT_EQ(backing::stat(readonly, &after), 0);
  EXPECT_EQ(before.st_size, kSize);
  EXPECT_EQ(before.st_ino, after.st_ino);
  const int retained = dup(fd);
  ASSERT_GE(retained, 0);
  close(fd);
  close(readonly);
  ASSERT_EQ(munmap(original, kSize), 0);
  ASSERT_EQ(munmap(alias, 4096), 0);
  original = static_cast<unsigned char *>(
      backing::map(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, retained, 0));
  ASSERT_NE(original, MAP_FAILED);
  close(retained);
  EXPECT_EQ(original[4096], 0x42);
  EXPECT_EQ(munmap(original, kSize), 0);
  rlimit observed{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &observed), 0);
  EXPECT_EQ(observed.rlim_cur, 0);
  EXPECT_EQ(observed.rlim_max, 0);
}

TEST(MemoryBackingDeathTest, HardZeroLimitPreservesAliasesAndAccessModes) {
  ASSERT_EXIT(
      {
        check_aliases();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

void check_FixedPartialMappingsPreserveDataAndRejectOverlap() {
  const int fd = backing::create(kSize);
  ASSERT_GE(fd, 0);
  auto *original = static_cast<unsigned char *>(
      backing::map(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  ASSERT_NE(original, MAP_FAILED);
  original[4096] = 0x57;
  auto *target = static_cast<unsigned char *>(
      mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(target, MAP_FAILED);
  target[0] = 0x11;
  EXPECT_EQ(backing::map(target, 4096, PROT_READ, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 4096),
            MAP_FAILED);
  EXPECT_EQ(errno, EEXIST);
  EXPECT_EQ(target[0], 0x11);
  ASSERT_EQ(backing::map(target, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 4096),
            target);
  EXPECT_EQ(target[0], 0x57);
  target[0] = 0x68;
  EXPECT_EQ(original[4096], 0x68);
  ASSERT_EQ(munmap(target, kSize), 0);
  // The freed hole is also a candidate for the helper's temporary shmat.
  ASSERT_EQ(backing::map(target, kSize, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0), target);
  EXPECT_EQ(target[4096], 0x68);
  ASSERT_EQ(munmap(target, kSize), 0);
  ASSERT_EQ(backing::map(target, kSize, PROT_READ, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0),
            target);
  EXPECT_EQ(target[4096], 0x68);
  EXPECT_EQ(munmap(target, kSize), 0);
  EXPECT_EQ(munmap(original, kSize), 0);
  close(fd);
}

void check_ConcurrentReopensUseIndependentReplies() {
  const int fd = backing::create(kSize);
  ASSERT_GE(fd, 0);
  std::atomic<unsigned> failures = 0;
  std::vector<std::thread> workers;
  for (unsigned thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&] {
      for (unsigned iteration = 0; iteration < 32; ++iteration) {
        const int alias = backing::reopen(fd, O_RDWR);
        if (alias < 0) {
          ++failures;
          continue;
        }
        void *view = backing::map(nullptr, 4096, PROT_READ, MAP_SHARED, alias, 0);
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
    const int fd = backing::create(kSize);
    if (fd < 0)
      _exit(2);
    auto *view = static_cast<unsigned char *>(
        backing::map(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (view == MAP_FAILED)
      _exit(3);
    view[0] = 0x9A;
    char children_path[128];
    std::snprintf(children_path, sizeof(children_path), "/proc/self/task/%d/children", getpid());
    FILE *children = std::fopen(children_path, "r");
    pid_t helper = -1;
    if (!children || std::fscanf(children, "%d", &helper) != 1)
      _exit(4);
    std::fclose(children);
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
  const int pidfd = syscall(SYS_pidfd_open, helper, 0);
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
  auto *view =
      static_cast<unsigned char *>(backing::map(nullptr, kSize, PROT_READ, MAP_SHARED, fd, 0));
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
TEST(MemoryBackingDeathTest, FixedPartialMappingsPreserveDataAndRejectOverlap) {
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

TEST(MemoryBackingDeathTest, ConcurrentReopensUseIndependentReplies) {
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

TEST(MemoryBackingDeathTest, DescriptorSurvivesCreatingProcessExit) {
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
        const int fd = backing::create(kSize);
        if (fd < 0) {
          ++failures;
          continue;
        }
        auto *view = static_cast<unsigned char *>(
            backing::map(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
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

TEST(MemoryBackingDeathTest, ConcurrentLimitChangesDoNotAffectAllocations) {
  ASSERT_EXIT(
      {
        check_concurrent_limits();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

} // namespace
