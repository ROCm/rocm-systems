// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/remote_driver.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <fcntl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

class LibcCancellationTest : public ::testing::Test {
protected:
  void SetUp() override {
    rocjitsu::libc_passthrough().resolve();
    saved_ = rocjitsu::libc_passthrough();
  }
  void TearDown() override { rocjitsu::libc_passthrough() = saved_; }

  void check_maps_cancellation(bool during_close) {
    static int maps_fd, cancel_result;
    static bool cancel_on_close;
    static ssize_t (*forward_read)(int, void *, size_t);
    static int (*forward_close)(int);
    maps_fd = -1;
    cancel_result = -1;
    cancel_on_close = during_close;
    forward_read = saved_.read;
    forward_close = saved_.close;
    rocjitsu::libc_passthrough().read = +[](int fd, void *data, size_t size) -> ssize_t {
      maps_fd = fd;
      if (!cancel_on_close)
        cancel_result = pthread_cancel(pthread_self());
      return forward_read(fd, data, size);
    };
    rocjitsu::libc_passthrough().close = +[](int fd) {
      if (cancel_on_close)
        cancel_result = pthread_cancel(pthread_self());
      return forward_close(fd);
    };
    struct Probe {
      bool completed = false;
      rocjitsu::amdgpu::PageWritability answer = rocjitsu::amdgpu::PageWritability::Indeterminate;
    } probe;
    pthread_t worker;
    ASSERT_EQ(pthread_create(
                  &worker, nullptr,
                  +[](void *opaque) -> void * {
                    auto &probe = *static_cast<Probe *>(opaque);
                    const uint8_t page = 0;
                    probe.answer =
                        rocjitsu::amdgpu::LegacyAddressSpaceTestAccess::page_writability(&page);
                    probe.completed = true;
                    // Cleanup must restore deferred cancellation, not discard the request.
                    pthread_testcancel();
                    return nullptr;
                  },
                  &probe),
              0);
    void *result = nullptr;
    ASSERT_EQ(pthread_join(worker, &result), 0);
    rocjitsu::libc_passthrough() = saved_;
    EXPECT_EQ(cancel_result, 0);
    EXPECT_EQ(result, PTHREAD_CANCELED);
    EXPECT_EQ(probe.completed, during_close);
    if (during_close) {
      EXPECT_EQ(probe.answer, rocjitsu::amdgpu::PageWritability::Writable);
    }
    ASSERT_GE(maps_fd, 0);
    const int descriptor_flags = fcntl(maps_fd, F_GETFD);
    EXPECT_EQ(descriptor_flags, -1);
    EXPECT_EQ(errno, EBADF);
    if (descriptor_flags >= 0)
      saved_.close(maps_fd);
  }

  void check_driver_destruction(int initial_state) {
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
    auto *driver = new rocjitsu::RemoteDriver(sockets[0]);
    static std::array<int, 2> closed_fds;
    static size_t close_count;
    static int (*forward_close)(int);
    closed_fds.fill(-1);
    close_count = 0;
    forward_close = saved_.close;
    rocjitsu::libc_passthrough().close = +[](int fd) {
      if (close_count < closed_fds.size())
        closed_fds[close_count] = fd;
      ++close_count;
      return forward_close(fd);
    };
    struct Probe {
      rocjitsu::RemoteDriver *driver;
      int initial_state;
      int restored_state = -1;
      int cancel_result = -1;
      bool destroyed = false;
    } probe{driver, initial_state};
    pthread_t worker;
    const int created = pthread_create(
        &worker, nullptr,
        +[](void *opaque) -> void * {
          auto &probe = *static_cast<Probe *>(opaque);
          pthread_setcancelstate(probe.initial_state, nullptr);
          probe.cancel_result = pthread_cancel(pthread_self());
          delete probe.driver;
          probe.destroyed = true;
          pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &probe.restored_state);
          pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
          pthread_testcancel();
          return nullptr;
        },
        &probe);
    if (created != 0) {
      rocjitsu::libc_passthrough() = saved_;
      delete driver;
      close(sockets[1]);
    }
    ASSERT_EQ(created, 0);
    void *result = nullptr;
    ASSERT_EQ(pthread_join(worker, &result), 0);
    rocjitsu::libc_passthrough() = saved_;
    EXPECT_EQ(result, PTHREAD_CANCELED);
    EXPECT_EQ(probe.cancel_result, 0);
    EXPECT_TRUE(probe.destroyed);
    EXPECT_EQ(probe.restored_state, initial_state);
    EXPECT_EQ(close_count, closed_fds.size());
    EXPECT_EQ(closed_fds[0], sockets[0]);
    for (int fd : closed_fds) {
      ASSERT_GE(fd, 0);
      EXPECT_EQ(fcntl(fd, F_GETFD), -1);
      EXPECT_EQ(errno, EBADF);
    }
    EXPECT_EQ(close(sockets[1]), 0);
  }

  rocjitsu::LibcPassthrough saved_;
};

TEST_F(LibcCancellationTest, ProcMapsReadCancellationClosesDescriptor) {
  check_maps_cancellation(false);
}

TEST_F(LibcCancellationTest, ProcMapsCloseDefersCancellationUntilCleanupCompletes) {
  check_maps_cancellation(true);
}

TEST_F(LibcCancellationTest, RemoteDriverDestructionDefersCancellation) {
  check_driver_destruction(PTHREAD_CANCEL_ENABLE);
}

TEST_F(LibcCancellationTest, RemoteDriverDestructionPreservesDisabledCancellation) {
  check_driver_destruction(PTHREAD_CANCEL_DISABLE);
}
