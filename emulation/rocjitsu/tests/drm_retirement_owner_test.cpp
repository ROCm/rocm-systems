// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/drm_file_state.h"
#include "rocjitsu/kmd/linux/drm_retirement_owner.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>

// This source belongs to a dedicated executable: its allocation wrappers must
// not affect unrelated simulator tests. Only the current thread rejects new,
// and assertions run after restoring allocation so GoogleTest can report errors.
namespace {
thread_local bool reject_allocations = false;

class RejectAllocations {
public:
  RejectAllocations() { reject_allocations = true; }
  ~RejectAllocations() { reject_allocations = false; }
  RejectAllocations(const RejectAllocations &) = delete;
  RejectAllocations &operator=(const RejectAllocations &) = delete;
};
} // namespace

// Keep successful allocations and deallocations paired by the runtime, including ASan.
extern "C" void *__real__Znwm(std::size_t size);
extern "C" void *__real__Znam(std::size_t size);

extern "C" void *__wrap__Znwm(std::size_t size) {
  if (reject_allocations)
    throw std::bad_alloc();
  return __real__Znwm(size);
}
extern "C" void *__wrap__Znam(std::size_t size) {
  if (reject_allocations)
    throw std::bad_alloc();
  return __real__Znam(size);
}

namespace rocjitsu {
namespace {

// Retirement needs the driver's identity and lifetime, not a running GPU. The
// alias below uses its enclosing runtime's control block, just like the interposer.
class RetirementTestDriver final : public LinuxKfd {
public:
  int open() override { return -ENOSYS; }
  int close() override { return 0; }
  int ioctl(unsigned long, void *) override { return -ENOSYS; }
  void *mmap(void *, size_t, int, int, off_t) override { return MAP_FAILED; }
  int munmap(void *, size_t) override { return -ENOSYS; }
  int fd() const override { return -1; }
  bool owns_fd(int) const override { return false; }
  std::string redirect_sysfs_path(const char *) const override { return {}; }
  bool is_doorbell_range(const void *, size_t) const override { return false; }
  void *mmap_replacing_client_doorbell_views(void *, size_t, int, int, int, off_t) override {
    return MAP_FAILED;
  }
  bool handles_drm_render_minor(uint32_t) const override { return false; }
  const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t) const override { return nullptr; }
  std::string topology_path() const override { return {}; }
  std::string drm_path() const override { return {}; }
};

struct RetirementTestRuntime {
  RetirementTestDriver driver;
};

std::unique_ptr<DrmFileState> make_file(uint64_t id, std::function<void()> deleted = {}) {
  auto file = std::make_unique<DrmFileState>();
  file->id = id;
  // Observe actual file destruction independently of the owner's cleanup call.
  // This lease deliberately has no backend route: retirement must use its owner.
  if (deleted)
    file->backend_lease = DrmBackendLease(std::weak_ptr<LinuxKfd>{}, std::move(deleted));
  return file;
}

std::shared_ptr<DrmFileState> shared_file(const std::shared_ptr<DrmRetirementOwner> &owner,
                                          uint64_t id) {
  return std::shared_ptr<DrmFileState>(make_file(id).release(), [owner](DrmFileState *file) {
    owner->enqueue(std::unique_ptr<DrmFileState>(file));
  });
}

TEST(DrmRetirementOwnerTest, BatchDeletesEveryFileBeforeReleasingTheRuntime) {
  std::vector<std::string> events;
  auto owner = std::make_shared<DrmRetirementOwner>([&](DrmFileState &file, LinuxKfd &) {
    events.push_back("cleanup " + std::to_string(file.id));
  });
  auto runtime = std::shared_ptr<RetirementTestRuntime>(
      new RetirementTestRuntime, [owner, &events](RetirementTestRuntime *runtime) {
        owner->begin_teardown();
        owner->finish_teardown(runtime->driver);
        events.push_back("runtime");
        delete runtime;
      });
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  owner->bind_backend(backend);
  owner->enqueue(make_file(1, [&] { events.push_back("delete 1"); }));
  owner->enqueue(make_file(2, [&] { events.push_back("delete 2"); }));
  auto batch = owner->take_batch();
  ASSERT_TRUE(batch);

  backend.reset();
  runtime.reset();
  EXPECT_TRUE(events.empty());
  batch = {};
  EXPECT_EQ(events, (std::vector<std::string>{"cleanup 1", "delete 1", "cleanup 2", "delete 2",
                                              "runtime"}));
}

TEST(DrmRetirementOwnerTest, DetachedBatchesExcludeConcurrentRuntimeDestruction) {
  std::atomic<unsigned> cleaned{0};
  std::atomic<unsigned> deleted{0};
  std::atomic<bool> destroyed{false};
  auto owner = std::make_shared<DrmRetirementOwner>([&](DrmFileState &, LinuxKfd &) {
    EXPECT_FALSE(destroyed.load());
    ++cleaned;
  });
  auto runtime = std::shared_ptr<RetirementTestRuntime>(
      new RetirementTestRuntime, [owner, &destroyed](RetirementTestRuntime *runtime) {
        owner->begin_teardown();
        owner->finish_teardown(runtime->driver);
        destroyed.store(true);
        delete runtime;
      });
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  owner->bind_backend(backend);

  std::binary_semaphore first_detached{0};
  std::binary_semaphore second_detached{0};
  std::binary_semaphore release_first{0};
  std::binary_semaphore release_second{0};
  owner->enqueue(make_file(1, [&] {
    EXPECT_FALSE(destroyed.load());
    ++deleted;
  }));
  std::jthread first([&] {
    auto batch = owner->take_batch();
    EXPECT_TRUE(batch);
    first_detached.release();
    release_first.acquire();
  });
  first_detached.acquire();
  owner->enqueue(make_file(2, [&] {
    EXPECT_FALSE(destroyed.load());
    ++deleted;
  }));
  std::jthread second([&] {
    auto batch = owner->take_batch();
    EXPECT_TRUE(batch);
    second_detached.release();
    release_second.acquire();
  });
  second_detached.acquire();

  // This is the final-close side of the race: it finds no pending files and
  // drops every runtime owner except the two detached application batches.
  std::jthread closer([&] {
    owner->drain();
    backend.reset();
    runtime.reset();
  });
  closer.join();
  EXPECT_FALSE(destroyed.load());
  release_first.release();
  first.join();
  EXPECT_EQ(cleaned.load(), 1u);
  EXPECT_EQ(deleted.load(), 1u);
  EXPECT_FALSE(destroyed.load());
  release_second.release();
  second.join();
  EXPECT_EQ(cleaned.load(), 2u);
  EXPECT_EQ(deleted.load(), 2u);
  EXPECT_TRUE(destroyed.load());
}

TEST(DrmRetirementOwnerTest, ExpiredPinLeavesOwnershipForRuntimeTeardown) {
  RetirementTestDriver driver;
  std::vector<uint64_t> cleaned;
  bool deleted = false;
  DrmRetirementOwner owner([&](DrmFileState &file, LinuxKfd &backend) {
    EXPECT_EQ(&backend, &driver);
    cleaned.push_back(file.id);
  });
  // Model the runtime deleter: its control block has expired, but the driver is
  // still alive until the deleter finishes draining its files.
  auto lifetime = std::make_shared<int>(0);
  auto backend = std::shared_ptr<LinuxKfd>(lifetime, &driver);
  owner.bind_backend(backend);
  backend.reset();
  lifetime.reset();
  owner.enqueue(make_file(7, [&] { deleted = true; }));
  EXPECT_FALSE(owner.take_batch());
  owner.drain();
  EXPECT_TRUE(cleaned.empty());
  EXPECT_FALSE(deleted);
  owner.begin_teardown();
  owner.finish_teardown(driver);
  EXPECT_EQ(cleaned, (std::vector<uint64_t>{7}));
  EXPECT_TRUE(deleted);
}

TEST(DrmRetirementOwnerTest, ConcurrentDrainsRetireEachFileExactlyOnce) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  std::mutex results_mutex;
  std::vector<unsigned> cleaned(128);
  DrmRetirementOwner owner([&](DrmFileState &file, LinuxKfd &) {
    std::lock_guard lock(results_mutex);
    ++cleaned.at(file.id);
  });
  owner.bind_backend(backend);
  for (uint64_t id = 0; id < cleaned.size(); ++id)
    owner.enqueue(make_file(id));
  std::barrier start(3);
  auto drain = [&] {
    start.arrive_and_wait();
    owner.drain();
  };
  std::jthread first(drain);
  std::jthread second(drain);
  start.arrive_and_wait();
  first.join();
  second.join();
  for (auto count : cleaned)
    EXPECT_EQ(count, 1u);
  EXPECT_FALSE(owner.take_batch());
  owner.begin_teardown();
  owner.finish_teardown(runtime->driver);
}

TEST(DrmRetirementOwnerTest, TeardownAcceptsRecursiveReleasesButPreventsApplicationDrain) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  std::vector<uint64_t> cleaned;
  std::shared_ptr<DrmRetirementOwner> owner;
  owner = std::make_shared<DrmRetirementOwner>([&](DrmFileState &file, LinuxKfd &) {
    cleaned.push_back(file.id);
    if (file.id == 1) {
      owner->enqueue(make_file(3));
      // Cleanup runs outside the owner mutex, and Closing denies this attempt
      // to steal work from the runtime's teardown drain.
      EXPECT_FALSE(owner->take_batch());
      owner->drain();
    }
  });
  owner->bind_backend(backend);
  owner->enqueue(make_file(1, [&] { owner->enqueue(make_file(4)); }));
  owner->begin_teardown();
  owner->enqueue(make_file(2)); // An engine callback released during join.
  EXPECT_FALSE(owner->take_batch());
  owner->finish_teardown(runtime->driver);
  EXPECT_EQ(cleaned, (std::vector<uint64_t>{1, 2, 3, 4}));
  EXPECT_FALSE(owner->take_batch());
}

TEST(DrmRetirementOwnerTest, ApplicationDrainRetiresRecursiveReleases) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  std::vector<uint64_t> cleaned;
  std::shared_ptr<DrmRetirementOwner> owner;
  owner = std::make_shared<DrmRetirementOwner>([&](DrmFileState &file, LinuxKfd &) {
    cleaned.push_back(file.id);
    if (file.id == 1)
      owner->enqueue(make_file(2));
  });
  owner->bind_backend(backend);
  // No backend lease was installed, as with failed file publication. The owner
  // must still retire this file and any later releases without getting stuck.
  owner->enqueue(make_file(1));
  owner->drain();
  EXPECT_EQ(cleaned, (std::vector<uint64_t>{1, 2}));
  owner->begin_teardown();
  owner->finish_teardown(runtime->driver);
}

TEST(DrmRetirementOwnerTest, AcquiredAndSubmissionOwnersDelayRetirementAfterDescriptorClose) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  unsigned cleaned = 0;
  auto owner = std::make_shared<DrmRetirementOwner>([&](DrmFileState &, LinuxKfd &) { ++cleaned; });
  owner->bind_backend(backend);
  auto descriptor = shared_file(owner, 1);
  auto acquired = descriptor;
  std::function<void()> submission = [file = descriptor] { EXPECT_EQ(file->id, 1u); };
  descriptor.reset();
  owner->drain();
  EXPECT_EQ(cleaned, 0u);
  acquired.reset();
  owner->drain();
  EXPECT_EQ(cleaned, 0u);
  submission();
  submission = {};
  EXPECT_EQ(cleaned, 0u); // Final release only enqueues, including on an engine thread.
  owner->drain();
  EXPECT_EQ(cleaned, 1u);
  owner->begin_teardown();
  owner->finish_teardown(runtime->driver);
}

TEST(DrmRetirementOwnerTest, MovingBatchPreservesCleanupAndRuntimeOwnership) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  std::vector<uint64_t> cleaned;
  DrmRetirementOwner owner([&](DrmFileState &file, LinuxKfd &) { cleaned.push_back(file.id); });
  owner.bind_backend(backend);
  owner.enqueue(make_file(1));
  auto first = owner.take_batch();
  owner.enqueue(make_file(2));
  auto second = owner.take_batch();
  auto moved = std::move(first);
  EXPECT_FALSE(first);
  second = std::move(moved);
  EXPECT_FALSE(moved);
  EXPECT_EQ(cleaned, (std::vector<uint64_t>{2}));
  second = {};
  EXPECT_EQ(cleaned, (std::vector<uint64_t>{2, 1}));
  owner.begin_teardown();
  owner.finish_teardown(runtime->driver);
}

TEST(DrmRetirementOwnerTest, TeardownReleasesTheRuntimeDeletersOwnerCapture) {
  auto owner = std::make_shared<DrmRetirementOwner>([](DrmFileState &, LinuxKfd &) {});
  std::weak_ptr<DrmRetirementOwner> observer = owner;
  auto runtime = std::shared_ptr<RetirementTestRuntime>(new RetirementTestRuntime,
                                                        [owner](RetirementTestRuntime *runtime) {
                                                          owner->begin_teardown();
                                                          owner->finish_teardown(runtime->driver);
                                                          delete runtime;
                                                        });
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  owner->bind_backend(backend);
  owner.reset();
  EXPECT_FALSE(observer.expired());
  backend.reset();
  runtime.reset();
  EXPECT_TRUE(observer.expired());
}

TEST(DrmRetirementOwnerTest, FinalReleaseAndBothDrainPathsDoNotAllocate) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  std::array<uint64_t, 3> cleaned{};
  size_t count = 0;
  // Force std::function to store its target on the heap. Sharing this callback
  // with an application batch must not copy its allocating target.
  const std::array<uint64_t, 64> payload{10, 20, 30};
  auto owner = std::make_shared<DrmRetirementOwner>(
      [&, payload](DrmFileState &file, LinuxKfd &) { cleaned[count++] = payload[file.id - 1]; });
  owner->bind_backend(backend);
  auto first = shared_file(owner, 1);
  auto second = shared_file(owner, 2);
  auto third = shared_file(owner, 3);
  {
    RejectAllocations failure;
    first.reset();
    owner->drain();
    // Exercise the first release after a drain as well as appending a successor.
    second.reset();
    owner->begin_teardown();
    third.reset();
    owner->finish_teardown(runtime->driver);
  }
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(cleaned, (std::array<uint64_t, 3>{10, 20, 30}));
}

TEST(DrmRetirementOwnerTest, FailedPublicationCanReleaseItsFileDuringAllocationFailureUnwind) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  unsigned cleaned = 0;
  auto owner = std::make_shared<DrmRetirementOwner>([&](DrmFileState &, LinuxKfd &) { ++cleaned; });
  owner->bind_backend(backend);
  auto unpublished = shared_file(owner, 1);
  bool caught = false;
  {
    RejectAllocations failure;
    try {
      auto file = std::move(unpublished);
      // Model fd-table insertion throwing before the backend lease is installed.
      throw std::bad_alloc();
    } catch (const std::bad_alloc &) {
      caught = true;
    }
    owner->drain();
  }
  EXPECT_TRUE(caught);
  EXPECT_EQ(cleaned, 1u);
  owner->begin_teardown();
  owner->finish_teardown(runtime->driver);
}

TEST(DrmRetirementOwnerTest, LongPendingChainRetiresInOrderWithoutRecursiveDeletion) {
  auto runtime = std::make_shared<RetirementTestRuntime>();
  auto backend = std::shared_ptr<LinuxKfd>(runtime, &runtime->driver);
  uint64_t next = 0;
  bool ordered = true;
  DrmRetirementOwner owner([&](DrmFileState &file, LinuxKfd &) { ordered &= file.id == next++; });
  owner.bind_backend(backend);
  constexpr uint64_t kFiles = 16384;
  for (uint64_t id = 0; id < kFiles; ++id)
    owner.enqueue(make_file(id));
  {
    RejectAllocations failure;
    owner.drain();
  }
  EXPECT_TRUE(ordered);
  EXPECT_EQ(next, kFiles);
  owner.begin_teardown();
  owner.finish_teardown(runtime->driver);
}

} // namespace
} // namespace rocjitsu
