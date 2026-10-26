// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "legacy_gpu_memory_fixture.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <thread>
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

  void arm_close_cancellation(unsigned cancel_at) {
    closed_fds_.fill(-1);
    close_count_ = 0;
    cancel_at_ = cancel_at;
    forward_close_ = saved_.close;
    rocjitsu::libc_passthrough().close = +[](int fd) {
      if (close_count_ < closed_fds_.size())
        closed_fds_[close_count_] = fd;
      if (++close_count_ == cancel_at_)
        pthread_cancel(pthread_self());
      return forward_close_(fd);
    };
  }

  void queue_allocation(int server, uint64_t handle, uint64_t address) {
    const int memfd = memfd_create("cancellation-test", MFD_CLOEXEC);
    ASSERT_GE(memfd, 0);
    ASSERT_EQ(ftruncate(memfd, 4096), 0);
    rocjitsu::RpcHeader reply{};
    reply.opcode = rocjitsu::RPC_IOCTL;
    reply.payload_bytes = sizeof(kfd_ioctl_alloc_memory_of_gpu_args);
    ASSERT_EQ(rocjitsu::rpc_send_msg(server, &reply, sizeof(reply), &memfd, 1), sizeof(reply));
    kfd_ioctl_alloc_memory_of_gpu_args args{};
    args.handle = handle;
    args.va_addr = address;
    args.size = 4096;
    ASSERT_TRUE(rocjitsu::rpc_send_exact(server, &args, sizeof(args)));
    ASSERT_EQ(close(memfd), 0);
  }

  void check_remote_cleanup(bool free_allocation) {
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
    auto driver = std::make_unique<rocjitsu::RemoteDriver>(sockets[0]);
    for (uint64_t handle = 1; handle <= (free_allocation ? 1u : 2u); ++handle) {
      queue_allocation(sockets[1], handle, handle * 4096);
      ASSERT_FALSE(HasFatalFailure());
      kfd_ioctl_alloc_memory_of_gpu_args args{};
      ASSERT_EQ(driver->ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args), 0);
      ASSERT_EQ(args.handle, handle);
    }
    if (free_allocation) {
      rocjitsu::RpcHeader reply{};
      reply.opcode = rocjitsu::RPC_IOCTL;
      ASSERT_TRUE(rocjitsu::rpc_send_exact(sockets[1], &reply, sizeof(reply)));
    }
    // For close(), interrupt the later close after the socket is already gone.
    arm_close_cancellation(free_allocation ? 1 : 2);
    int result = -1;
    bool survived_cancel = false;
    std::thread worker([&] {
      kfd_ioctl_free_memory_of_gpu_args args{};
      args.handle = 1;
      result =
          free_allocation ? driver->ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &args) : driver->close();
      pthread_testcancel();
      survived_cancel = true;
    });
    worker.join();
    rocjitsu::libc_passthrough() = saved_;
    EXPECT_EQ(result, 0);
    EXPECT_FALSE(survived_cancel);
    const unsigned expected = free_allocation ? 1 : 3;
    ASSERT_EQ(close_count_, expected);
    int descriptor = -1;
    off_t offset = 0;
    EXPECT_EQ(driver->find_memfd_for_addr(reinterpret_cast<void *>(4096), 1, &descriptor, &offset),
              rocjitsu::RemoteDriver::MemfdLookup::kNotFound);
    const int replacement = open("/dev/null", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(replacement, 0);
    for (unsigned index = 0; index < expected; ++index) {
      const int fd = closed_fds_[index];
      // open() may have reused exactly one of these numbers already.
      if (fd != replacement) {
        EXPECT_EQ(fcntl(fd, F_GETFD), -1);
        EXPECT_EQ(errno, EBADF);
      }
      ASSERT_EQ(dup2(replacement, fd), fd);
    }
    driver.reset();
    for (unsigned index = 0; index < expected; ++index) {
      EXPECT_GE(fcntl(closed_fds_[index], F_GETFD), 0);
      if (closed_fds_[index] != replacement) {
        EXPECT_EQ(close(closed_fds_[index]), 0);
      }
    }
    EXPECT_EQ(close(replacement), 0);
    EXPECT_EQ(close(sockets[1]), 0);
  }

  void check_synthetic_creation(unsigned cancel_at) {
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
    rocjitsu::RemoteDriver driver(sockets[0]);
    arm_close_cancellation(cancel_at);
    int descriptor = -1;
    bool survived_cancel = false;
    std::thread worker([&] {
      descriptor = driver.reissue_synthetic_kfd_fd();
      pthread_testcancel();
      survived_cancel = true;
    });
    worker.join();
    rocjitsu::libc_passthrough() = saved_;
    EXPECT_FALSE(survived_cancel);
    EXPECT_EQ(close_count_, 2u);
    ASSERT_GE(descriptor, 256);
    EXPECT_GE(fcntl(descriptor, F_GETFD), 0);
    for (unsigned index = 0; index < close_count_ && index < closed_fds_.size(); ++index) {
      EXPECT_EQ(fcntl(closed_fds_[index], F_GETFD), -1);
      EXPECT_EQ(errno, EBADF);
    }
    EXPECT_EQ(close(descriptor), 0);
    EXPECT_EQ(close(sockets[1]), 0);
  }

  void check_malformed_reply_cancellation(bool mmap_reply) {
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
    rocjitsu::RemoteDriver driver(sockets[0]);
    rocjitsu::UniqueDriverFd server(sockets[1]);
    rocjitsu::UniqueDriverFd descriptor(memfd_create("malformed-reply", MFD_CLOEXEC));
    ASSERT_TRUE(descriptor);
    rocjitsu::RpcHeader reply{};
    reply.opcode = mmap_reply ? rocjitsu::RPC_MMAP : rocjitsu::RPC_IOCTL;
    reply.payload_bytes =
        mmap_reply ? sizeof(rocjitsu::RpcMmapResponse) : rocjitsu::kMaxPayloadBytes + 1;
    const int sent_fd = descriptor.get();
    ASSERT_EQ(rocjitsu::rpc_send_msg(server.get(), &reply, sizeof(reply), &sent_fd, 1),
              sizeof(reply));
    if (!mmap_reply) {
      // An unpoisoned stream would accept these leftover bytes as the next reply.
      rocjitsu::RpcHeader stale{};
      stale.opcode = rocjitsu::RPC_IOCTL;
      ASSERT_TRUE(rocjitsu::rpc_send_exact(server.get(), &stale, sizeof(stale)));
    }
    // End the mmap reply before its payload, while keeping the request side open
    // so an incorrect subsequent send remains observable without hanging.
    ASSERT_EQ(shutdown(server.get(), SHUT_WR), 0);
    descriptor.reset();
    arm_close_cancellation(1);
    int result = 0;
    bool survived_cancel = false;
    std::thread worker([&] {
      if (mmap_reply) {
        void *mapping = driver.mmap(nullptr, 4096, PROT_READ, MAP_SHARED, 0);
        result = mapping == MAP_FAILED ? -errno : 0;
        if (mapping != MAP_FAILED)
          munmap(mapping, 4096);
      } else {
        kfd_ioctl_get_version_args version{};
        result = driver.ioctl(AMDKFD_IOC_GET_VERSION, &version);
      }
      pthread_testcancel();
      survived_cancel = true;
    });
    worker.join();
    rocjitsu::libc_passthrough() = saved_;
    EXPECT_EQ(result, -EPROTO);
    EXPECT_FALSE(survived_cancel);
    ASSERT_EQ(close_count_, 1u);
    ASSERT_GE(closed_fds_[0], 0);
    const int descriptor_flags = fcntl(closed_fds_[0], F_GETFD);
    EXPECT_EQ(descriptor_flags, -1);
    EXPECT_EQ(errno, EBADF);
    if (descriptor_flags >= 0)
      close(closed_fds_[0]);

    std::array<uint8_t, 128> requests{};
    const size_t request_bytes =
        sizeof(rocjitsu::RpcHeader) +
        (mmap_reply ? sizeof(rocjitsu::RpcMmapRequest)
                    : sizeof(rocjitsu::RpcIoctlRequest) + sizeof(kfd_ioctl_get_version_args));
    EXPECT_EQ(recv(server.get(), requests.data(), requests.size(), MSG_DONTWAIT), request_bytes);
    kfd_ioctl_get_version_args version{};
    EXPECT_EQ(driver.ioctl(AMDKFD_IOC_GET_VERSION, &version), -EPROTO);
    EXPECT_EQ(driver.mmap(nullptr, 4096, PROT_READ, MAP_SHARED, 0), MAP_FAILED);
    EXPECT_EQ(errno, EPROTO);
    EXPECT_EQ(driver.munmap(nullptr, 4096), -EPROTO);
    EXPECT_EQ(recv(server.get(), requests.data(), requests.size(), MSG_DONTWAIT), 0);
  }

  static inline std::array<int, 4> closed_fds_;
  static inline unsigned close_count_, cancel_at_;
  static inline int (*forward_close_)(int);

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

TEST_F(LibcCancellationTest, RemoteCloseClearsOwnershipBeforeCancellation) {
  check_remote_cleanup(false);
}

TEST_F(LibcCancellationTest, RemoteFreeClearsOwnershipBeforeCancellation) {
  check_remote_cleanup(true);
}

TEST_F(LibcCancellationTest, SyntheticCreationDefersMarkerCloseCancellation) {
  check_synthetic_creation(1);
}

TEST_F(LibcCancellationTest, SyntheticCreationDefersEventfdCloseCancellation) {
  check_synthetic_creation(2);
}

TEST_F(LibcCancellationTest, MalformedIoctlReplyPoisonsStreamBeforeCancellation) {
  check_malformed_reply_cancellation(false);
}

TEST_F(LibcCancellationTest, MalformedMmapReplyPoisonsStreamBeforeCancellation) {
  check_malformed_reply_cancellation(true);
}
