// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/memory_backing.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
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

pid_t token_keeper(int fd) {
  ucred peer{};
  socklen_t size = sizeof(peer);
  return getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0 ? peer.pid : -1;
}

volatile sig_atomic_t child_signal = 0;
void observe_child_signal(int) { child_signal = 1; }

void check_helper_lifecycle(bool subreaper = false) {
  if (subreaper) {
    ASSERT_EQ(prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0), 0);
  }
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  struct sigaction action {};
  action.sa_handler = observe_child_signal;
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGCHLD, &action, nullptr), 0);
  const int fd = backing::create(kSize);
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
  limited.rlim_cur = 5;
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &limited), 0);
  EXPECT_EQ(backing::create(kSize), -1);
  EXPECT_EQ(errno, EMFILE);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  int next = backing::create(kSize);
  ASSERT_GE(next, 0);
  EXPECT_EQ(token_keeper(next), helper);
  close(next);

  const int pidfd = syscall(SYS_pidfd_open, helper, 0);
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
  next = backing::create(kSize);
  ASSERT_GE(next, 0) << strerror(errno);
  EXPECT_NE(token_keeper(next), helper);
  close(next);
}

TEST(MemoryBackingDeathTest, HelperDoesNotChangeChildWaitsAndRecoversAfterFailure) {
  ASSERT_EXIT(
      {
        check_helper_lifecycle();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(MemoryBackingDeathTest, SubreaperWarnsAboutChildWaitsAndRecoversAfterFailure) {
  // GTest captures stderr in a regular file, which cannot accept the warning
  // under hard FSIZE=0. A pipe observes it without relaxing the file limit.
  int diagnostics[2];
  ASSERT_EQ(pipe2(diagnostics, O_CLOEXEC), 0);
  EXPECT_EXIT(
      {
        close(diagnostics[0]);
        if (dup2(diagnostics[1], STDERR_FILENO) < 0)
          _exit(2);
        close(diagnostics[1]);
        check_helper_lifecycle(true);
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
  close(diagnostics[1]);
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

void check_group_signal() {
  ASSERT_GE(setsid(), 0);
  ASSERT_NE(signal(SIGINT, SIG_IGN), SIG_ERR);
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int fd = backing::create(kSize);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kill(0, SIGINT), 0);
  void *view = backing::map(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(view, MAP_FAILED);
  const int reopened = backing::reopen(fd, O_RDONLY);
  ASSERT_GE(reopened, 0);
  const int next = backing::create(kSize);
  ASSERT_GE(next, 0);
  EXPECT_EQ(token_keeper(next), token_keeper(fd));
  close(next);
  close(reopened);
  close(fd);
  munmap(view, kSize);
}

TEST(MemoryBackingDeathTest, GroupSignalsPreserveUnmappedTokens) {
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
  int fd = backing::create(kSize);
  ASSERT_GE(fd, 0);
  const pid_t first_helper = token_keeper(fd);
  int lease = -1;
  for (int candidate = 3; candidate < 256; ++candidate) {
    char path[64], target[128]{};
    std::snprintf(path, sizeof(path), "/proc/self/fd/%d", candidate);
    const auto length = readlink(path, target, sizeof(target) - 1);
    if (length > 0 && std::strstr(target, "memfd:rocjitsu-owner")) {
      lease = candidate;
      break;
    }
  }
  ASSERT_GE(lease, 0);
  const int replacement = syscall(SYS_memfd_create, "lease-replacement", MFD_CLOEXEC);
  ASSERT_GE(replacement, 0);
  struct stat before {
  }, after{};
  ASSERT_EQ(fstat(replacement, &before), 0);
  ASSERT_EQ(dup2(replacement, lease), lease);
  close(replacement);
  const int next = backing::create(kSize);
  ASSERT_GE(next, 0);
  EXPECT_NE(token_keeper(next), first_helper);
  ASSERT_EQ(fstat(lease, &after), 0);
  EXPECT_EQ(before.st_ino, after.st_ino);
  // Closing the owner's lease must not invalidate separately held tokens.
  const int alias = backing::reopen(fd, O_RDONLY);
  ASSERT_GE(alias, 0);
  void *view = backing::map(nullptr, kSize, PROT_READ, MAP_SHARED, alias, 0);
  ASSERT_NE(view, MAP_FAILED);
  munmap(view, kSize);
  close(alias);
  close(next);
  close(fd);
  close(lease);
}

TEST(MemoryBackingDeathTest, ReplacedLeasePreservesOtherDescriptorsAndTokens) {
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
  const int retained = backing::create(kSize);
  ASSERT_GE(retained, 0);
  const int retained_pidfd = syscall(SYS_pidfd_open, token_keeper(retained), 0);
  ASSERT_GE(retained_pidfd, 0);
  int previous_pidfd = -1;
  for (unsigned iteration = 0; iteration < 4; ++iteration) {
    ASSERT_EQ(setegid(iteration % 2 ? original : alternate), 0);
    const int next = backing::create(kSize);
    ASSERT_GE(next, 0) << strerror(errno);
    if (previous_pidfd >= 0) {
      pollfd exited{previous_pidfd, POLLIN, 0};
      EXPECT_EQ(poll(&exited, 1, 2000), 1) << "retired credential helper survived";
      close(previous_pidfd);
    }
    previous_pidfd = syscall(SYS_pidfd_open, token_keeper(next), 0);
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
    EXPECT_EQ(leases, 1u);
  }
  close(previous_pidfd);
  // Retiring the owner's lease must preserve independently exported tokens.
  pollfd retained_exited{retained_pidfd, POLLIN, 0};
  EXPECT_EQ(poll(&retained_exited, 1, 0), 0);
  const int alias = backing::reopen(retained, O_RDONLY);
  ASSERT_GE(alias, 0);
  void *view = backing::map(nullptr, kSize, PROT_READ, MAP_SHARED, alias, 0);
  ASSERT_NE(view, MAP_FAILED);
  EXPECT_EQ(munmap(view, kSize), 0);
  close(alias);
  close(retained);
  EXPECT_EQ(poll(&retained_exited, 1, 2000), 1);
  close(retained_pidfd);
}

TEST(MemoryBackingDeathTest, CredentialChangesRetireLeasesAndPreserveTokens) {
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
    const int fd = backing::create(kSize);
    const pid_t helper = token_keeper(fd);
    if (fd < 0 || helper <= 0)
      _exit(2);
    close(fd);
    // This child keeps the inherited lease descriptor while its parent execs.
    // A socket-only lease would retain the helper until this child also exits.
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
  const int pidfd = syscall(SYS_pidfd_open, children[0], 0);
  const int idle_pidfd = syscall(SYS_pidfd_open, children[1], 0);
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

TEST(MemoryBackingDeathTest, ExecRetiresUnusedHelperWhileOwnerLives) {
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
  std::vector<int> descriptors{backing::create(kSize)};
  ASSERT_GE(descriptors.front(), 0);
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &original), 0);
  for (unsigned count = 1; count < 96; ++count) {
    const int fd = backing::create(kSize);
    ASSERT_GE(fd, 0) << count << ": " << strerror(errno);
    EXPECT_EQ(token_keeper(fd), token_keeper(descriptors.front()));
    descriptors.push_back(fd);
  }
  for (const int fd : descriptors)
    close(fd);
}

TEST(MemoryBackingDeathTest, HelperUsesHardDescriptorLimit) {
  ASSERT_EXIT(
      {
        check_nofile_headroom();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(MemoryBackingDeathTest, StartupFailureNamesHelper) {
  if (getuid() == 0)
    GTEST_SKIP() << "RLIMIT_NPROC does not constrain root";
  ASSERT_EXIT(
      {
        const rlimit zero{};
        if (setrlimit(RLIMIT_NPROC, &zero) != 0)
          _exit(2);
        const int fd = backing::create(kSize);
        _exit(fd == -1 && errno == EAGAIN ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "Cannot start memory keeper.*Resource temporarily unavailable");
}

void check_seek_contract() {
  const rlimit zero{};
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &zero), 0);
  const int fd = backing::create(kSize);
  ASSERT_GE(fd, 0);
  off_t result = -1;
  EXPECT_TRUE(backing::seek_token(fd, 0, SEEK_END, result));
  EXPECT_EQ(result, kSize);
  EXPECT_TRUE(backing::seek_token(fd, 0, SEEK_SET, result));
  EXPECT_EQ(result, 0);
  EXPECT_TRUE(backing::seek_token(fd, 1, SEEK_SET, result));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_TRUE(backing::seek_token(fd, 0, SEEK_CUR, result));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EINVAL);
  close(fd);
}

TEST(MemoryBackingDeathTest, TokenSupportsDmaBufSeekContract) {
  ASSERT_EXIT(
      {
        check_seek_contract();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

} // namespace
