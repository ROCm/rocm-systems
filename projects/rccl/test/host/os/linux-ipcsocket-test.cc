/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/os/linux_ipcsocket.cc on real Unix datagram sockets, abstract and
// path-bound. Its libc calls go through fakes/libc_seam.h, hooked only to fail or to forge a reply.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <inttypes.h>
#include <random>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "ipcsocket.h"
#include "os.h"
#include "param.h"
#include "utils.h"

#include "linux-test-common.h"

#include "fakes/param_redirect.h"
#include "fakes/libc_seam.h"
#include LINUX_IPCSOCKET_CC_PATH
#include "fakes/libc_seam_undef.h"

namespace {

class LinuxIpcsocketMicrotest : public LinuxTestBase {
 protected:
  // Abstract names are shared by every process in the network namespace, and pids repeat across
  // containers, so each process draws a random token.
  void SetUp() override {
    static const uint64_t token = ((uint64_t)std::random_device{}() << 32 | std::random_device{}()) & ~0xffffull;
    hash_ = token | (++counter_ & 0xffff);
    // Close skips descriptor 0, so with stdin closed a socket must not land on it.
    if (!FdOpen(0)) devNull_ = open("/dev/null", O_RDONLY);
    ASSERT_TRUE(FdOpen(0));
  }
  void TearDown() override {
    for (auto* s : {&a_, &b_}) ncclIpcSocketClose(s);
    if (devNull_ >= 0) close(devNull_);
  }
  std::string Name(int rank) {
    char name[NCCL_IPC_SOCKNAME_LEN];
    snprintf(name, sizeof(name), "/tmp/nccl-socket-%d-%lx", rank, (unsigned long)hash_);
    return name;
  }
  // a_ is rank 0 and b_ rank 1, both on hash_.
  void Open(volatile uint32_t* abortFlag = nullptr) {
    ASSERT_EQ(ncclIpcSocketInit(&a_, 0, hash_, abortFlag), ncclSuccess);
    ASSERT_EQ(ncclIpcSocketInit(&b_, 1, hash_, abortFlag), ncclSuccess);
  }
  // Passes a pipe's write end from a_ to b_ and checks the received copy writes into the pipe.
  void PassDescriptor() {
    int p[2];
    ASSERT_EQ(pipe(p), 0);
    ASSERT_EQ(ncclIpcSocketSendFd(&a_, p[1], 1, hash_), ncclSuccess);
    ncclIpcFd got = -1;
    ASSERT_EQ(ncclIpcSocketRecvFd(&b_, &got), ncclSuccess);
    EXPECT_NE(got, p[1]);
    EXPECT_EQ(write(got, "z", 1), 1);
    char c = 0;
    EXPECT_EQ(read(p[0], &c, 1), 1);
    EXPECT_EQ(c, 'z');
    EXPECT_EQ(ncclIpcFdClose(got), 0);
    EXPECT_FALSE(FdOpen(got));
    close(p[0]);
    close(p[1]);
  }
  static uint64_t counter_;
  uint64_t hash_ = 0;
  int devNull_ = -1;
  ncclIpcSocket a_ = {-1, "", nullptr}, b_ = {-1, "", nullptr};
};
uint64_t LinuxIpcsocketMicrotest::counter_ = 0;

TEST_F(LinuxIpcsocketMicrotest, Init_NullHandleIsAnInternalError) {
  EXPECT_EQ(ncclIpcSocketInit(nullptr, 0, hash_, nullptr), ncclInternalError);
}

TEST_F(LinuxIpcsocketMicrotest, Init_AbstractSocketLeavesNoFile) {
  ASSERT_EQ(ncclIpcSocketInit(&a_, 3, hash_, nullptr), ncclSuccess);
  EXPECT_TRUE(FdOpen(a_.fd));
  EXPECT_EQ(a_.socketName, Name(3));
  EXPECT_EQ(a_.abortFlag, nullptr);
  EXPECT_NE(access(a_.socketName, F_OK), 0);
  EXPECT_EQ(fcntl(a_.fd, F_GETFL) & O_NONBLOCK, 0);
}

TEST_F(LinuxIpcsocketMicrotest, Init_PathSocketReplacesAStaleFileAndCloseRemovesIt) {
  params_.Set("IPC_USE_ABSTRACT_SOCKET", 0);
  std::string name = Name(0);
  FILE* stale = fopen(name.c_str(), "w");
  ASSERT_NE(stale, nullptr);
  fclose(stale);
  ASSERT_EQ(ncclIpcSocketInit(&a_, 0, hash_, nullptr), ncclSuccess);
  struct stat st;
  ASSERT_EQ(stat(name.c_str(), &st), 0);
  EXPECT_TRUE(S_ISSOCK(st.st_mode));
  int fd = a_.fd;
  ASSERT_EQ(ncclIpcSocketClose(&a_), ncclSuccess);
  EXPECT_FALSE(FdOpen(fd));
  EXPECT_NE(access(name.c_str(), F_OK), 0);
  a_.fd = -1;
}

TEST_F(LinuxIpcsocketMicrotest, Init_AbortFlagMakesTheSocketNonBlocking) {
  volatile uint32_t abortFlag = 0;
  ASSERT_EQ(ncclIpcSocketInit(&a_, 0, hash_, &abortFlag), ncclSuccess);
  EXPECT_EQ(a_.abortFlag, &abortFlag);
  EXPECT_NE(fcntl(a_.fd, F_GETFL) & O_NONBLOCK, 0);
}

TEST_F(LinuxIpcsocketMicrotest, Init_NameInUseIsASystemError) {
  ASSERT_EQ(ncclIpcSocketInit(&a_, 0, hash_, nullptr), ncclSuccess);
  EXPECT_EQ(ncclIpcSocketInit(&b_, 0, hash_, nullptr), ncclSystemError);
  EXPECT_EQ(b_.fd, -1);
  EXPECT_STREQ(b_.socketName, "");
}

TEST_F(LinuxIpcsocketMicrotest, Init_SocketFailureIsASystemError) {
  ScopedHook socketHook(g_socket, [](int family, int type, int) {
    EXPECT_EQ(family, AF_UNIX);
    EXPECT_EQ(type, SOCK_DGRAM);
    errno = EMFILE;
    return -1;
  });
  EXPECT_EQ(ncclIpcSocketInit(&a_, 0, hash_, nullptr), ncclSystemError);
  EXPECT_EQ(a_.fd, -1);
}

TEST_F(LinuxIpcsocketMicrotest, GetFd) {
  ncclIpcFd fd = -1;
  EXPECT_EQ(ncclIpcSocketGetFd(nullptr, &fd), ncclInvalidArgument);
  Open();
  EXPECT_EQ(ncclIpcSocketGetFd(&a_, nullptr), ncclSuccess);
  ASSERT_EQ(ncclIpcSocketGetFd(&a_, &fd), ncclSuccess);
  EXPECT_EQ(fd, a_.fd);
}

TEST_F(LinuxIpcsocketMicrotest, Close_RejectsANullSocket) {
  EXPECT_EQ(ncclIpcSocketClose(nullptr), ncclInternalError);
}

// Descriptor 0 counts as "not open", like -1, so Close leaves it. Pins AICOMRCCL-2976; a fix flips this pin.
TEST_F(LinuxIpcsocketMicrotest, Close_KnownBug_TreatsDescriptorZeroAsUnopened) {
  ncclIpcSocket zero = {0, "", nullptr};
  EXPECT_EQ(ncclIpcSocketClose(&zero), ncclSuccess);
  EXPECT_TRUE(FdOpen(0));
}

TEST_F(LinuxIpcsocketMicrotest, SendFd_PassesADescriptorOverAnAbstractSocket) {
  Open();
  PassDescriptor();
}

TEST_F(LinuxIpcsocketMicrotest, SendFd_PassesADescriptorOverAPathSocket) {
  params_.Set("IPC_USE_ABSTRACT_SOCKET", 0);
  Open();
  EXPECT_EQ(access(b_.socketName, F_OK), 0);
  PassDescriptor();
  for (auto* s : {&a_, &b_}) {
    ASSERT_EQ(ncclIpcSocketClose(s), ncclSuccess);  // while the param still says path sockets: unlinks
    EXPECT_NE(access(s->socketName, F_OK), 0);
    s->fd = -1;
  }
}

TEST_F(LinuxIpcsocketMicrotest, SendMsg_CarriesAHeaderWithOrWithoutADescriptor) {
  Open();
  char hdr[6] = "hello";
  ASSERT_EQ(ncclIpcSocketSendMsg(&a_, hdr, sizeof(hdr), -1, 1, hash_), ncclSuccess);
  char got[6] = {};
  ASSERT_EQ(ncclIpcSocketRecvMsg(&b_, got, sizeof(got), nullptr), ncclSuccess);
  EXPECT_STREQ(got, "hello");

  ASSERT_EQ(ncclIpcSocketSendMsg(&a_, hdr, sizeof(hdr), STDOUT_FILENO, 1, hash_), ncclSuccess);
  std::memset(got, 0, sizeof(got));
  ncclIpcFd fd = -1;
  ASSERT_EQ(ncclIpcSocketRecvMsg(&b_, got, sizeof(got), &fd), ncclSuccess);
  EXPECT_STREQ(got, "hello");
  EXPECT_TRUE(FdOpen(fd));
  close(fd);
}

TEST_F(LinuxIpcsocketMicrotest, RecvFd_MessageWithoutADescriptorIsASystemError) {
  Open();
  ASSERT_EQ(ncclIpcSocketSendMsg(&b_, nullptr, 0, -1, 0, hash_), ncclSuccess);
  ncclIpcFd fd = -1;
  EXPECT_EQ(ncclIpcSocketRecvFd(&a_, &fd), ncclSystemError);
}

TEST_F(LinuxIpcsocketMicrotest, RecvFd_ForeignControlMessageIsASystemError) {
  Open();
  ScopedHook recvmsgHook(g_recvmsg, [](int, msghdr* msg, int) -> ssize_t {
    cmsghdr* c = CMSG_FIRSTHDR(msg);
    c->cmsg_len = CMSG_LEN(sizeof(int));
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_CREDENTIALS;
    return 1;
  });
  ncclIpcFd fd = -1;
  EXPECT_EQ(ncclIpcSocketRecvFd(&a_, &fd), ncclSystemError);
  ScopedHook wrongLevelHook(g_recvmsg, [](int, msghdr* msg, int) -> ssize_t {
    cmsghdr* c = CMSG_FIRSTHDR(msg);
    c->cmsg_len = CMSG_LEN(sizeof(int));
    c->cmsg_level = SOL_IP;
    c->cmsg_type = SCM_RIGHTS;
    return 1;
  });
  EXPECT_EQ(ncclIpcSocketRecvFd(&a_, &fd), ncclSystemError);
}

TEST_F(LinuxIpcsocketMicrotest, RecvMsg_RetriesTransientErrorsAndFailsOnOthers) {
  Open();
  const int transient[] = {EINTR, EAGAIN, EWOULDBLOCK};
  int failures = 0;
  ScopedHook recvmsgHook(g_recvmsg, [&](int fd, msghdr* msg, int flags) -> ssize_t {
    if (failures < 3) {
      errno = transient[failures++];
      return -1;
    }
    return ::recvmsg(fd, msg, flags);
  });
  char hdr[2] = "x", got[2] = {};
  ASSERT_EQ(ncclIpcSocketSendMsg(&b_, hdr, sizeof(hdr), -1, 0, hash_), ncclSuccess);
  ASSERT_EQ(ncclIpcSocketRecvMsg(&a_, got, sizeof(got), nullptr), ncclSuccess);
  EXPECT_EQ(failures, 3);
  EXPECT_STREQ(got, "x");

  ScopedHook failingHook(g_recvmsg, FailWith(ENOTSOCK));
  EXPECT_EQ(ncclIpcSocketRecvMsg(&a_, got, sizeof(got), nullptr), ncclSystemError);
}

TEST_F(LinuxIpcsocketMicrotest, RecvMsg_AbortStopsWaiting) {
  volatile uint32_t abortFlag = 1;
  Open(&abortFlag);
  ncclIpcFd fd = -1;
  EXPECT_EQ(ncclIpcSocketRecvFd(&a_, &fd), ncclInternalError);  // non-blocking: EAGAIN, then the flag
  EXPECT_TRUE(log_.Text().empty());
}

TEST_F(LinuxIpcsocketMicrotest, SendMsg_NoPeerIsASystemError) {
  Open();
  EXPECT_EQ(ncclIpcSocketSendFd(&a_, STDOUT_FILENO, 7, hash_), ncclSystemError);
}

TEST_F(LinuxIpcsocketMicrotest, SendMsg_RetriesTransientErrorsUntilAborted) {
  volatile uint32_t abortFlag = 0;
  Open(&abortFlag);
  int attempts = 0;
  ScopedHook sendmsgHook(g_sendmsg, [&](int fd, const msghdr* msg, int flags) -> ssize_t {
    if (++attempts < 3) {
      errno = attempts == 1 ? EINTR : EAGAIN;
      return -1;
    }
    return ::sendmsg(fd, msg, flags);
  });
  char hdr[2] = "y", got[2] = {};
  ASSERT_EQ(ncclIpcSocketSendMsg(&a_, hdr, sizeof(hdr), -1, 1, hash_), ncclSuccess);
  EXPECT_EQ(attempts, 3);
  ASSERT_EQ(ncclIpcSocketRecvMsg(&b_, got, sizeof(got), nullptr), ncclSuccess);
  EXPECT_STREQ(got, "y");

  abortFlag = 1;
  ScopedHook wouldBlockHook(g_sendmsg, FailWith(EWOULDBLOCK));
  EXPECT_EQ(ncclIpcSocketSendMsg(&a_, hdr, sizeof(hdr), -1, 1, hash_), ncclInternalError);
}

}  // namespace
