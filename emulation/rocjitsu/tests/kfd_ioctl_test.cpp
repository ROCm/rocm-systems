// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/virtual_machine.h"
#include "scoped_temp.h"

#include "embedded_schema.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "simdojo/sim/simulation.h"
#include "util/unique_handle.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_mi355x.json";
constexpr uint32_t kGpuId = 38144;
const std::string CDNA5_CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx1250_mi455x.json";
constexpr uint32_t kCdna5GpuId = 1250;

// A part with no modelled CWSR record layout. gfx1100 is not a debug target:
// kmd::cwsr_layout_modelled() covers gfx942/gfx950/gfx1250, and the driver has to
// decline stops there rather than publish a record rocm-dbgapi would misparse.
const std::string RDNA3_CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx1100_w7900.json";
constexpr uint32_t kRdna3GpuId = 7019;

class ChildProcessGuard {
public:
  explicit ChildProcessGuard(pid_t pid) : pid_(pid) {}
  ~ChildProcessGuard() {
    if (pid_ <= 0)
      return;
    kill(pid_, SIGKILL);
    int status = 0;
    while (waitpid(pid_, &status, 0) == -1 && errno == EINTR) {
    }
  }

  ChildProcessGuard(const ChildProcessGuard &) = delete;
  ChildProcessGuard &operator=(const ChildProcessGuard &) = delete;

  void release() { pid_ = -1; }

private:
  pid_t pid_;
};

int daemon_debug_enable(rocjitsu::SimulatedKfd &daemon, uint32_t debugger,
                        kfd_ioctl_dbg_trap_args &args) {
  const std::string proc_path = std::format("/proc/{}", args.pid);
  const int proc_fd = ::open(proc_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (proc_fd < 0)
    return -errno;
  int mem_fd = ::openat(proc_fd, "mem", O_RDWR | O_CLOEXEC);
  if (mem_fd < 0) {
    const int open_errno = errno;
    ::close(proc_fd);
    return -open_errno;
  }
  const int result = daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &args, &mem_fd, proc_fd);
  if (mem_fd >= 0)
    ::close(mem_fd);
  ::close(proc_fd);
  return result;
}

uint32_t query_gb_addr_config(const std::string &config_path, uint32_t gpu_id) {
  auto loaded = rocjitsu::config::load_config(config_path.c_str(), rocjitsu::kEmbeddedSchema);
  auto root = loaded.take_root();
  auto *soc = dynamic_cast<rocjitsu::SoC *>(root.get());
  if (!soc)
    return 0;
  auto num_xcds = soc->num_xcds();

  loaded.engine_config.max_ticks = 0;
  loaded.engine_config.await_primaries = true;
  simdojo::SimulationEngine engine(loaded.engine_config);

  auto soc_root = std::unique_ptr<rocjitsu::SoC>(static_cast<rocjitsu::SoC *>(root.release()));
  auto vm = std::make_unique<rocjitsu::VirtualMachine>(std::move(soc_root));
  auto *driver = vm->driver();

  engine.topology().set_root(std::move(vm));
  loaded.wire_links(engine.topology());
  soc->wire_backing(engine.topology());
  engine.create();
  engine.register_as_primary();

  driver->setup_topology(loaded.device, num_xcds);
  int fd = driver->open();
  if (fd < 0)
    return 0;

  kfd_ioctl_get_tile_config_args args{};
  args.gpu_id = gpu_id;
  int rc = driver->ioctl(AMDKFD_IOC_GET_TILE_CONFIG, &args);
  driver->close();
  return rc == 0 ? args.gb_addr_config : 0;
}

class KfdIoctlTest : public ::testing::Test {
protected:
  void SetUp() override { SetUpWithConfig(CONFIG_PATH); }

  // Split out so a fixture can bring up a different part. Uses ASSERT_*, so it
  // has to stay void-returning and be called directly from SetUp().
  void SetUpWithConfig(const std::string &config_path) {
    setenv("RJ_CONFIG", config_path.c_str(), 1);
    loaded_ = rocjitsu::config::load_config(config_path.c_str(), rocjitsu::kEmbeddedSchema);
    auto root = loaded_.take_root();
    auto *soc = dynamic_cast<rocjitsu::SoC *>(root.get());
    ASSERT_NE(soc, nullptr);
    soc_ = soc;
    auto num_xcds = soc->num_xcds();

    loaded_.engine_config.max_ticks = 0;
    loaded_.engine_config.await_primaries = true;
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded_.engine_config);

    auto soc_root = std::unique_ptr<rocjitsu::SoC>(static_cast<rocjitsu::SoC *>(root.release()));
    auto vm = std::make_unique<rocjitsu::VirtualMachine>(std::move(soc_root));
    driver_ = vm->driver();

    engine_->topology().set_root(std::move(vm));
    loaded_.wire_links(engine_->topology());
    soc->wire_backing(engine_->topology());
    engine_->create();
    engine_->register_as_primary();

    driver_->setup_topology(loaded_.device, num_xcds);
    int fd = driver_->open();
    ASSERT_GE(fd, 0);
  }

  void TearDown() override {
    if (driver_)
      driver_->close();
    for (int fd : debug_fds_)
      ::close(fd);
    debug_fds_.clear();
  }

  // Returns a real eventfd standing in for a debugger's notification target.
  // kfd_dbg_trap_enable() takes a reference to dbg_fd via fget(), so the driver
  // rejects an unusable descriptor; enable-success tests therefore need a live
  // fd. Tracked here so TearDown closes it.
  int make_debug_fd() {
    int fd = eventfd(0, EFD_CLOEXEC);
    EXPECT_GE(fd, 0);
    debug_fds_.push_back(fd);
    return fd;
  }

  rocjitsu::config::LoadedConfig loaded_;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
  rocjitsu::SoC *soc_ = nullptr;
  rocjitsu::SimulatedKfd *driver_ = nullptr;
  std::vector<int> debug_fds_;
};

// Same driver surface, brought up on a part whose CWSR layout is not modelled.
// Deriving does not inherit KfdIoctlTest's cases: TEST_F registers against the
// fixture it names, so only the cases written against this one run here.
class KfdIoctlRdna3Test : public KfdIoctlTest {
protected:
  void SetUp() override { SetUpWithConfig(RDNA3_CONFIG_PATH); }
};

// A compute queue created through KFD is replicated onto every XCD so its
// dispatches can be spread across the whole device; the XCD that owns the queue
// still reads the ring alone. An SDMA queue is per-engine and is not replicated.
TEST_F(KfdIoctlTest, CreateQueueReplicatesComputeQueueAcrossXcds) {
  const uint32_t num_xcds = soc_->num_xcds();
  ASSERT_GT(num_xcds, 1u);

  auto total_registered = [&]() {
    size_t total = 0;
    for (uint32_t xi = 0; xi < num_xcds; ++xi)
      total += soc_->xcd(xi)->command_processor()->registered_queue_count_for_test();
    return total;
  };
  ASSERT_EQ(total_registered(), 0u);

  alignas(4096) std::array<std::byte, 8192> ring{};
  alignas(64) std::array<uint64_t, 8> ptrs{};

  kfd_ioctl_create_queue_args args{};
  args.gpu_id = kGpuId;
  args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  args.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  args.ring_size = static_cast<uint32_t>(ring.size());
  args.read_pointer_address = reinterpret_cast<uint64_t>(&ptrs[0]);
  args.write_pointer_address = reinterpret_cast<uint64_t>(&ptrs[1]);
  args.queue_percentage = 100;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &args), 0);

  EXPECT_EQ(total_registered(), num_xcds)
      << "compute queue should be registered on every XCD's command processor";
  for (uint32_t xi = 0; xi < num_xcds; ++xi)
    EXPECT_EQ(soc_->xcd(xi)->command_processor()->registered_queue_count_for_test(), 1u)
        << "xcd" << xi;

  kfd_ioctl_destroy_queue_args destroy{};
  destroy.queue_id = args.queue_id;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy), 0);
  EXPECT_EQ(total_registered(), 0u) << "destroying the queue should drop every replica";
}

TEST_F(KfdIoctlTest, CreateQueueDoesNotReplicateSdmaQueue) {
  const uint32_t num_xcds = soc_->num_xcds();
  ASSERT_GT(num_xcds, 1u);

  alignas(4096) std::array<std::byte, 8192> ring{};
  alignas(64) std::array<uint64_t, 8> ptrs{};

  kfd_ioctl_create_queue_args args{};
  args.gpu_id = kGpuId;
  args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA;
  args.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  args.ring_size = static_cast<uint32_t>(ring.size());
  args.read_pointer_address = reinterpret_cast<uint64_t>(&ptrs[0]);
  args.write_pointer_address = reinterpret_cast<uint64_t>(&ptrs[1]);
  args.queue_percentage = 100;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &args), 0);
  ASSERT_NE(args.queue_id, 0u) << "queue ids are process-local and start at one";

  auto registered_on_all_xcds = [&] {
    size_t total = 0;
    for (uint32_t xi = 0; xi < num_xcds; ++xi)
      total += soc_->xcd(xi)->command_processor()->registered_queue_count_for_test();
    return total;
  };
  EXPECT_EQ(registered_on_all_xcds(), 1u)
      << "an SDMA queue belongs to one engine, not to every XCD";

  kfd_ioctl_destroy_queue_args destroy_args{};
  destroy_args.queue_id = args.queue_id;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy_args), 0);
  EXPECT_EQ(registered_on_all_xcds(), 0u) << "destroying the queue should leave nothing behind";
}

// Fan-out is an allowlist of the compute types, not "anything that is not SDMA",
// so both halves need covering: the second compute type must fan out, and a type
// this driver does not recognize must not acquire device-wide replication by
// falling through. Testing only COMPUTE_AQL and one SDMA type would let either
// half regress -- a negation would still pass both.
TEST_F(KfdIoctlTest, CreateQueueFansOutNamedComputeTypesOnly) {
  const uint32_t num_xcds = soc_->num_xcds();
  ASSERT_GT(num_xcds, 1u);

  alignas(4096) std::array<std::byte, 8192> ring{};
  alignas(64) std::array<uint64_t, 8> ptrs{};

  auto registered_on_all_xcds = [&]() {
    size_t total = 0;
    for (uint32_t xi = 0; xi < num_xcds; ++xi)
      total += soc_->xcd(xi)->command_processor()->registered_queue_count_for_test();
    return total;
  };

  auto create = [&](uint32_t queue_type) {
    kfd_ioctl_create_queue_args args{};
    args.gpu_id = kGpuId;
    args.queue_type = queue_type;
    args.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
    args.ring_size = static_cast<uint32_t>(ring.size());
    args.read_pointer_address = reinterpret_cast<uint64_t>(&ptrs[0]);
    args.write_pointer_address = reinterpret_cast<uint64_t>(&ptrs[1]);
    args.queue_percentage = 100;
    EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &args), 0) << "queue_type " << queue_type;
    return args.queue_id;
  };
  auto destroy = [&](uint32_t queue_id) {
    kfd_ioctl_destroy_queue_args args{};
    args.queue_id = queue_id;
    EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &args), 0);
  };

  ASSERT_EQ(registered_on_all_xcds(), 0u);
  const uint32_t compute = create(KFD_IOC_QUEUE_TYPE_COMPUTE);
  ASSERT_NE(compute, 0u) << "CREATE_QUEUE did not return a queue id to destroy";
  EXPECT_EQ(registered_on_all_xcds(), num_xcds)
      << "the PM4 compute type is named in the allowlist and must fan out";
  destroy(compute);
  ASSERT_EQ(registered_on_all_xcds(), 0u);

  // One past the last type the UAPI defines, so it tracks the header rather than
  // being a literal that quietly becomes a real type when the UAPI grows. It is
  // accepted as an ordinary queue, but it is not thereby a compute queue, so it
  // stays on the XCD that owns it.
  const uint32_t unknown = create(KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID + 1);
  ASSERT_NE(unknown, 0u) << "CREATE_QUEUE did not return a queue id to destroy";
  EXPECT_EQ(registered_on_all_xcds(), 1u)
      << "an unrecognized queue type must not acquire device-wide replication";
  destroy(unknown);
}

// Replication makes every CP hold a host-accessible queue, so "does this CP have
// a KFD queue" stops answering "does this CP have a queue of its own". Two
// queues land on different XCDs, and destroying the first leaves the second's
// replica behind on the XCD that owned it. That CP must then be back to owning
// nothing: a predicate that counted the leftover replica would keep its doorbell
// monitor alive for a ring it never reads, and would let the queue be reported
// idle from more than one CP.
TEST_F(KfdIoctlTest, DestroyingOneOwnerLeavesTheOtherQueueUnaffected) {
  const uint32_t num_xcds = soc_->num_xcds();
  ASSERT_GT(num_xcds, 1u);

  alignas(4096) std::array<std::byte, 8192> ring_a{};
  alignas(4096) std::array<std::byte, 8192> ring_b{};
  alignas(64) std::array<uint64_t, 8> ptrs_a{};
  alignas(64) std::array<uint64_t, 8> ptrs_b{};

  auto create = [&](std::array<std::byte, 8192> &ring, std::array<uint64_t, 8> &ptrs) {
    kfd_ioctl_create_queue_args args{};
    args.gpu_id = kGpuId;
    args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    args.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
    args.ring_size = static_cast<uint32_t>(ring.size());
    args.read_pointer_address = reinterpret_cast<uint64_t>(&ptrs[0]);
    args.write_pointer_address = reinterpret_cast<uint64_t>(&ptrs[1]);
    args.queue_percentage = 100;
    EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &args), 0);
    return args.queue_id;
  };

  // Find the CP that owns a queue: exactly one polls, the rest hold replicas.
  auto owning_cp = [&]() -> rocjitsu::amdgpu::CommandProcessor * {
    rocjitsu::amdgpu::CommandProcessor *found = nullptr;
    for (uint32_t xi = 0; xi < num_xcds; ++xi) {
      auto *cp = soc_->xcd(xi)->command_processor();
      if (cp->polled_kfd_queue_count_for_test() > 0) {
        EXPECT_EQ(found, nullptr) << "more than one CP claims to own a queue";
        found = cp;
      }
    }
    return found;
  };

  const uint32_t queue_a = create(ring_a, ptrs_a);
  auto *cp_a = owning_cp();
  ASSERT_NE(cp_a, nullptr);

  create(ring_b, ptrs_b);
  // Both queues are replicated everywhere, so every CP now holds two entries --
  // but only two CPs own one, and the rest own none. That gap between holding a
  // queue and owning one is the whole point.
  size_t total_owned = 0;
  for (uint32_t xi = 0; xi < num_xcds; ++xi) {
    auto *cp = soc_->xcd(xi)->command_processor();
    EXPECT_EQ(cp->registered_queue_count_for_test(), 2u) << "xcd" << xi;
    EXPECT_LE(cp->polled_kfd_queue_count_for_test(), 1u) << "xcd" << xi;
    total_owned += cp->polled_kfd_queue_count_for_test();
  }
  EXPECT_EQ(total_owned, 2u) << "each queue is owned by exactly one XCD";
  EXPECT_EQ(cp_a->polled_kfd_queue_count_for_test(), 1u) << "the first queue's owner still owns it";

  kfd_ioctl_destroy_queue_args destroy{};
  destroy.queue_id = queue_a;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy), 0);

  // The first queue's owner still holds the second queue's replica, so it still
  // has a KFD queue -- but it no longer owns one, which is what its doorbell
  // monitor and the idle walk must key off.
  EXPECT_EQ(cp_a->registered_queue_count_for_test(), 1u) << "the peer's replica is still here";
  EXPECT_EQ(cp_a->polled_kfd_queue_count_for_test(), 0u)
      << "a leftover replica is not a queue of its own";

  // ...and the surviving queue is still owned by exactly one CP, a different one.
  auto *cp_b = owning_cp();
  ASSERT_NE(cp_b, nullptr);
  EXPECT_NE(cp_b, cp_a);

  // The census above is necessary but not sufficient: it would still pass if the
  // monitor kept keying off "has a KFD queue". Observe the behavior the owner-only
  // predicate exists for. cp_a polls a ring it no longer owns unless its monitor
  // retires, while cp_b must keep polling for the queue it does own.
  auto wait_for_monitor = [](rocjitsu::amdgpu::CommandProcessor *cp, bool expected) {
    // The monitor retires on its own cadence; allow generous slack for CI load.
    for (int i = 0; i < 2000 && cp->doorbell_monitor_running_for_test() != expected; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return cp->doorbell_monitor_running_for_test();
  };
  EXPECT_FALSE(wait_for_monitor(cp_a, false))
      << "a CP left holding only a replica must not keep a monitor for a ring it never reads";
  EXPECT_TRUE(cp_b->doorbell_monitor_running_for_test())
      << "the surviving queue's owner must still poll its own ring";
}

// The idle walk skips replicas, so a fanned-out queue raises HQD_IDLE from the
// XCD that owns it and from nowhere else. Each peer's shards drain before the
// owner's, so a report from a peer is not merely duplicated, it is early.
//
// Reaching the skip needs a CP that is *both* running a monitor and holding a
// replica -- a CP with only replicas never starts a monitor at all, so its sweep
// never runs and a test built on one passes no matter what the walk does. Two
// processes give that shape and make it observable: the callback carries the
// process id, so an owner reporting its own queue is distinguishable from a
// replica reporting someone else's. Queue placement is by process-local ordinal,
// so process B's second queue is what lands on a different XCD than process A's
// first.
TEST_F(KfdIoctlTest, IdleNotificationComesOnlyFromTheOwningXcd) {
  const uint32_t num_xcds = soc_->num_xcds();
  ASSERT_GT(num_xcds, 1u);

  // Extra processes on the fixture's own driver, which already has a topology;
  // a freshly constructed SimulatedKfd would not know this GPU.
  const uint32_t pid_a = driver_->open_process();
  const uint32_t pid_b = driver_->open_process();
  ASSERT_NE(pid_a, 0u);
  ASSERT_NE(pid_b, pid_a);

  alignas(4096) std::array<std::byte, 32768> rings{};
  alignas(64) std::array<uint64_t, 32> ptrs{};

  auto create = [&](uint32_t pid, size_t slot) {
    kfd_ioctl_create_queue_args args{};
    args.gpu_id = kGpuId;
    args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    args.ring_base_address = reinterpret_cast<uint64_t>(rings.data() + slot * 8192);
    args.ring_size = 8192;
    args.read_pointer_address = reinterpret_cast<uint64_t>(&ptrs[slot * 4]);
    args.write_pointer_address = reinterpret_cast<uint64_t>(&ptrs[slot * 4 + 1]);
    args.queue_percentage = 100;
    EXPECT_EQ(driver_->ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &args), 0);
    return args.queue_id;
  };

  auto *owner_of_a = soc_->xcd(0)->command_processor();
  auto *owner_of_b2 = soc_->xcd(1)->command_processor();
  ASSERT_NE(owner_of_a, owner_of_b2);

  // Record which process ids each of the two CPs reports idle. Replaces the
  // driver's own callback; nothing here waits on a KFD event.
  std::mutex seen_mutex;
  std::set<uint32_t> seen_by_b2_owner;
  std::set<uint32_t> seen_by_a_owner;

  // Installed before the first CREATE_QUEUE, which is what starts the doorbell
  // monitors. interrupt_cb_ is a plain std::function that the poll loop reads
  // without a lock, so assigning it once a monitor is live is a data race on the
  // function object itself -- one TSan reports.
  owner_of_b2->set_interrupt_callback([&](uint32_t pid, uint32_t) {
    std::lock_guard<std::mutex> lock(seen_mutex);
    seen_by_b2_owner.insert(pid);
  });
  owner_of_a->set_interrupt_callback([&](uint32_t pid, uint32_t) {
    std::lock_guard<std::mutex> lock(seen_mutex);
    seen_by_a_owner.insert(pid);
  });

  // Detaches the observers before the state they capture goes out of scope.
  // Declared after that state so it is destroyed first, and it runs on every exit
  // path including a failed ASSERT: closing both processes destroys their queues,
  // which stops and joins every monitor, so no poll thread can still be inside a
  // callback by the time these locals die.
  struct ObserverGuard {
    rocjitsu::SimulatedKfd *driver;
    rocjitsu::SoC *soc;
    uint32_t pid_a;
    uint32_t pid_b;
    ~ObserverGuard() {
      driver->close(pid_a);
      driver->close(pid_b);
      for (uint32_t xi = 0; xi < soc->num_xcds(); ++xi)
        soc->xcd(xi)->command_processor()->set_interrupt_callback(nullptr);
    }
  } observer_guard{driver_, soc_, pid_a, pid_b};

  // A's only queue takes ordinal 0; B's queues take ordinals 0 and 1, so B's
  // second one is owned by a different XCD than A's.
  create(pid_a, 0);
  create(pid_b, 1);
  create(pid_b, 2);

  // owner_of_b2 must be running a monitor of its own -- otherwise its idle sweep
  // never executes and this test cannot see the walk at all.
  ASSERT_GT(owner_of_b2->polled_kfd_queue_count_for_test(), 0u);
  ASSERT_GT(owner_of_b2->registered_queue_count_for_test(),
            owner_of_b2->polled_kfd_queue_count_for_test())
      << "the XCD under test must also hold a replica it does not own";

  // The observers below are installed only on the two owners, so a replica-only
  // XCD that started a monitor of its own would run the same idle sweep and report
  // queues it does not own without this test ever seeing it. Pin that directly:
  // holding replicas must not make a CP polled, and must not start a monitor.
  for (uint32_t xi = 0; xi < soc_->num_xcds(); ++xi) {
    auto *cp = soc_->xcd(xi)->command_processor();
    if (cp == owner_of_a || cp == owner_of_b2)
      continue;
    EXPECT_GT(cp->registered_queue_count_for_test(), 0u)
        << "xcd" << xi << " should hold replicas of the fanned-out queues";
    EXPECT_EQ(cp->polled_kfd_queue_count_for_test(), 0u)
        << "xcd" << xi << " owns no queue of its own and must poll nothing";
    EXPECT_FALSE(cp->doorbell_monitor_running_for_test())
        << "xcd" << xi << " started a doorbell monitor for replicas alone";
  }

  // Every queue is empty from creation, so each owner re-broadcasts idle on its
  // periodic sweep. Wait for the sweep to have happened rather than for a fixed
  // duration, so a loaded machine does not turn "not yet" into "never".
  auto b2_owner_reported = [&] {
    std::lock_guard<std::mutex> lock(seen_mutex);
    return !seen_by_b2_owner.empty();
  };
  for (int i = 0; i < 4000 && !b2_owner_reported(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_TRUE(b2_owner_reported()) << "the owning XCD never reported its idle queue";
  // Give a wrong reporter the same chance to appear that the right one had.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::lock_guard<std::mutex> lock(seen_mutex);
  EXPECT_TRUE(seen_by_b2_owner.count(pid_b) == 1) << "the owning XCD must report the queue it owns";
  EXPECT_EQ(seen_by_b2_owner.count(pid_a), 0u)
      << "this XCD holds only a replica of process A's queue and must not report it idle";
  EXPECT_EQ(seen_by_a_owner.count(pid_a), 1u) << "process A's queue is idle on its own owner";
}

class KfdIoctlCdna5Test : public KfdIoctlTest {
protected:
  void SetUp() override { SetUpWithConfig(CDNA5_CONFIG_PATH); }
};

TEST_F(KfdIoctlCdna5Test, RuntimeTrapInterruptSignalsQueueExceptionFromM0) {
  constexpr uint64_t kKernelAddress = 0x600000000ULL;
  constexpr uint64_t kTrapHandlerAddress = 0x600001000ULL;
  constexpr uint32_t kExceptionEventId = 37;
  constexpr uint64_t kWaveAbort = KFD_EC_MASK(EC_QUEUE_WAVE_ABORT);

  std::array<uint8_t, 4096> code_page{};
  std::array<uint8_t, 4096> trap_handler_page{};
  alignas(4096) std::array<uint8_t, 4096> cwsr{};
  alignas(4096) std::array<uint8_t, 4096> exception_page{};
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kKernelAddress, code_page.data(), code_page.size());
  process->map_pages(kTrapHandlerAddress, trap_handler_page.data(), trap_handler_page.size());
  const uint64_t exception_status_address = reinterpret_cast<uint64_t>(exception_page.data());
  process->map_pages(exception_status_address, exception_page.data(), exception_page.size());

  // ROCr allocates this header as host-backed identity memory rather than a
  // KFD allocation. Queue creation must still discover the error signal and
  // event from the first-level CWSR header.
  std::memcpy(cwsr.data() + 6 * sizeof(uint32_t), &exception_status_address,
              sizeof(exception_status_address));
  std::memcpy(cwsr.data() + 6 * sizeof(uint32_t) + sizeof(uint64_t), &kExceptionEventId,
              sizeof(kExceptionEventId));

  kfd_ioctl_set_trap_handler_args set_handler{};
  set_handler.gpu_id = kCdna5GpuId;
  set_handler.tba_addr = kTrapHandlerAddress;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_SET_TRAP_HANDLER, &set_handler), 0);

  std::array<uint8_t, 4096> ring{};
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kCdna5GpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = reinterpret_cast<uint64_t>(cwsr.data());
  create.ctx_save_restore_size = static_cast<uint32_t>(cwsr.size());
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  auto *cp = soc_->xcd(0)->command_processor();
  ASSERT_NE(cp, nullptr);
  ASSERT_FALSE(cp->compute_units().empty());
  auto *cu = cp->compute_units().front();
  ASSERT_NE(cu, nullptr);
  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);

  uint64_t delivered_status = 0;
  uint32_t delivered_event_id = 0;
  cp->set_interrupt_callback([&](uint32_t process_id, uint32_t event_id) {
    EXPECT_EQ(process_id, driver_->local_process_id());
    delivered_status = memory->read64(exception_status_address, process_id);
    delivered_event_id = event_id;
    // ROCr consumes and clears the queue error signal before returning from
    // its asynchronous callback.
    memory->write64(exception_status_address, 0, process_id);
  });

  constexpr uint32_t kSTrapAbort = 0xBF900002u;
  constexpr uint32_t kSSendmsgInterrupt = 0xBFB60001u;
  const uint32_t handler[] = {
      0xBEFD00FFu,        0x00000007u,              // s_mov_b32 m0, doorbell_id
      kSSendmsgInterrupt, 0xBEFD00FFu, 0x00000407u, // s_mov_b32 m0, WAVE_ABORT | doorbell_id
      kSSendmsgInterrupt,
  };
  memory->write32(kKernelAddress, kSTrapAbort, driver_->local_process_id());
  for (uint32_t i = 0; i < std::size(handler); ++i)
    memory->write32(kTrapHandlerAddress + i * sizeof(uint32_t), handler[i],
                    driver_->local_process_id());

  auto *wave = cu->dispatch_wf(/*wg_id=*/0, kKernelAddress, /*sgprs=*/16, /*vgprs=*/4);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(driver_->local_process_id());
  wave->set_queue_id(create.queue_id);

  cu->step(); // Enter the configured trap handler.
  cu->step(); // Load a doorbell-only M0 value.
  cu->step(); // A plain interrupt must not become a queue exception.
  EXPECT_EQ(delivered_status, 0u);
  cu->step(); // Add EC_QUEUE_WAVE_ABORT above the doorbell bits.
  cu->step(); // Deliver it after releasing the CU wave-state lock.

  EXPECT_EQ(delivered_status, kWaveAbort);
  EXPECT_EQ(delivered_event_id, kExceptionEventId);
  EXPECT_TRUE(wave->debug_suspended());
}

TEST_F(KfdIoctlTest, SetMemoryPolicy) {
  kfd_ioctl_set_memory_policy_args args{};
  args.gpu_id = kGpuId;
  args.default_policy = KFD_IOC_CACHE_POLICY_COHERENT;
  args.alternate_policy = KFD_IOC_CACHE_POLICY_NONCOHERENT;
  args.alternate_aperture_base = 0x1000;
  args.alternate_aperture_size = 0x2000;

  int rc = driver_->ioctl(AMDKFD_IOC_SET_MEMORY_POLICY, &args);
  EXPECT_EQ(rc, 0);
}

TEST_F(KfdIoctlTest, GetTileConfig) {
  std::array<uint32_t, 40> tile_config;
  std::array<uint32_t, 40> macro_tile_config;
  tile_config.fill(0xdeadbeef);
  macro_tile_config.fill(0xdeadbeef);

  kfd_ioctl_get_tile_config_args args{};
  args.gpu_id = kGpuId;
  args.tile_config_ptr = reinterpret_cast<uint64_t>(tile_config.data());
  args.macro_tile_config_ptr = reinterpret_cast<uint64_t>(macro_tile_config.data());
  args.num_tile_configs = static_cast<uint32_t>(tile_config.size());
  args.num_macro_tile_configs = static_cast<uint32_t>(macro_tile_config.size());

  int rc = driver_->ioctl(AMDKFD_IOC_GET_TILE_CONFIG, &args);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(args.num_tile_configs, 32u);
  EXPECT_EQ(args.num_macro_tile_configs, 16u);
  EXPECT_EQ(args.gb_addr_config, 0u);
  EXPECT_EQ(args.num_banks, 0u);
  EXPECT_EQ(args.num_ranks, 0u);

  for (uint32_t i = 0; i < args.num_tile_configs; ++i)
    EXPECT_EQ(tile_config[i], 0u);
  for (uint32_t i = 0; i < args.num_macro_tile_configs; ++i)
    EXPECT_EQ(macro_tile_config[i], 0u);
  for (uint32_t i = args.num_tile_configs; i < tile_config.size(); ++i)
    EXPECT_EQ(tile_config[i], 0xdeadbeefu);
  for (uint32_t i = args.num_macro_tile_configs; i < macro_tile_config.size(); ++i)
    EXPECT_EQ(macro_tile_config[i], 0xdeadbeefu);
}

TEST_F(KfdIoctlTest, GetTileConfigReportsWrittenCounts) {
  std::array<uint32_t, 4> tile_config;
  std::array<uint32_t, 3> macro_tile_config;
  tile_config.fill(0xdeadbeef);
  macro_tile_config.fill(0xdeadbeef);

  kfd_ioctl_get_tile_config_args args{};
  args.gpu_id = kGpuId;
  args.tile_config_ptr = reinterpret_cast<uint64_t>(tile_config.data());
  args.macro_tile_config_ptr = reinterpret_cast<uint64_t>(macro_tile_config.data());
  args.num_tile_configs = static_cast<uint32_t>(tile_config.size());
  args.num_macro_tile_configs = static_cast<uint32_t>(macro_tile_config.size());

  int rc = driver_->ioctl(AMDKFD_IOC_GET_TILE_CONFIG, &args);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(args.num_tile_configs, static_cast<uint32_t>(tile_config.size()));
  EXPECT_EQ(args.num_macro_tile_configs, static_cast<uint32_t>(macro_tile_config.size()));
  for (auto value : tile_config)
    EXPECT_EQ(value, 0u);
  for (auto value : macro_tile_config)
    EXPECT_EQ(value, 0u);
}

TEST_F(KfdIoctlTest, GetTileConfigRejectsUnknownGpuId) {
  std::array<uint32_t, 4> tile_config;
  std::array<uint32_t, 3> macro_tile_config;
  tile_config.fill(0xdeadbeef);
  macro_tile_config.fill(0xdeadbeef);

  kfd_ioctl_get_tile_config_args args{};
  args.gpu_id = 0xdeadbeef;
  args.tile_config_ptr = reinterpret_cast<uint64_t>(tile_config.data());
  args.macro_tile_config_ptr = reinterpret_cast<uint64_t>(macro_tile_config.data());
  args.num_tile_configs = static_cast<uint32_t>(tile_config.size());
  args.num_macro_tile_configs = static_cast<uint32_t>(macro_tile_config.size());

  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_GET_TILE_CONFIG, &args), -EINVAL);
  EXPECT_EQ(args.num_tile_configs, static_cast<uint32_t>(tile_config.size()));
  EXPECT_EQ(args.num_macro_tile_configs, static_cast<uint32_t>(macro_tile_config.size()));
  for (auto value : tile_config)
    EXPECT_EQ(value, 0xdeadbeefu);
  for (auto value : macro_tile_config)
    EXPECT_EQ(value, 0xdeadbeefu);
}

TEST_F(KfdIoctlTest, GetTileConfigReturnsUnsupportedInDaemonMode) {
  ASSERT_NE(soc_, nullptr);
  rocjitsu::SimulatedKfd daemon_driver(*soc_, true);
  uint32_t process_id = daemon_driver.open_process();
  ASSERT_NE(process_id, 0u);

  std::array<uint32_t, 4> tile_config;
  std::array<uint32_t, 3> macro_tile_config;
  tile_config.fill(0xdeadbeef);
  macro_tile_config.fill(0xdeadbeef);

  kfd_ioctl_get_tile_config_args args{};
  args.gpu_id = kGpuId;
  args.tile_config_ptr = reinterpret_cast<uint64_t>(tile_config.data());
  args.macro_tile_config_ptr = reinterpret_cast<uint64_t>(macro_tile_config.data());
  args.num_tile_configs = static_cast<uint32_t>(tile_config.size());
  args.num_macro_tile_configs = static_cast<uint32_t>(macro_tile_config.size());

  EXPECT_EQ(daemon_driver.ioctl(process_id, AMDKFD_IOC_GET_TILE_CONFIG, &args), -ENOTSUP);
  for (auto value : tile_config)
    EXPECT_EQ(value, 0xdeadbeefu);
  for (auto value : macro_tile_config)
    EXPECT_EQ(value, 0xdeadbeefu);
  EXPECT_EQ(daemon_driver.close(process_id), 0);
}

TEST(KfdIoctlStandaloneTest, GetTileConfigReportsRdnaGbAddrConfig) {
  EXPECT_EQ(query_gb_addr_config(std::string(CONFIG_DIR) + "/gfx1100_w7900.json", 7019),
            rocjitsu::kmd::gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA3));
  EXPECT_EQ(query_gb_addr_config(std::string(CONFIG_DIR) + "/gfx1201_r9700.json", 8716),
            rocjitsu::kmd::gb_addr_config_for_arch(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST_F(KfdIoctlTest, ImportDmabufAndQueryInfo) {
  constexpr size_t kSize = 4096;
  int memfd = static_cast<int>(syscall(SYS_memfd_create, "kfd_dmabuf_test", MFD_CLOEXEC));
  ASSERT_GE(memfd, 0);
  ASSERT_EQ(ftruncate(memfd, kSize), 0);

  void *addr = mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
  ASSERT_NE(addr, MAP_FAILED);
  std::memset(addr, 0xAB, kSize);

  kfd_ioctl_import_dmabuf_args import_args{};
  import_args.dmabuf_fd = memfd;
  import_args.gpu_id = kGpuId;
  import_args.va_addr = reinterpret_cast<uint64_t>(addr);

  int rc = driver_->ioctl(AMDKFD_IOC_IMPORT_DMABUF, &import_args);
  EXPECT_EQ(rc, 0);
  EXPECT_NE(import_args.handle, 0u);

  kfd_ioctl_get_dmabuf_info_args info_args{};
  info_args.dmabuf_fd = memfd;
  info_args.metadata_ptr = 0;
  info_args.metadata_size = 0;

  rc = driver_->ioctl(AMDKFD_IOC_GET_DMABUF_INFO, &info_args);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(info_args.size, kSize);
  EXPECT_EQ(info_args.gpu_id, kGpuId);
  EXPECT_EQ(info_args.flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT, KFD_IOC_ALLOC_MEM_FLAGS_GTT);

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = import_args.handle;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);

  munmap(addr, kSize);
  close(memfd);
}

TEST_F(KfdIoctlTest, SvmSetAndGetAttributes) {
  constexpr uint64_t kStart = 0x4000;
  constexpr uint64_t kSize = 0x2000;

  std::vector<uint8_t> buffer(sizeof(kfd_ioctl_svm_args) + 2 * sizeof(kfd_ioctl_svm_attribute));
  auto *svm_args = reinterpret_cast<kfd_ioctl_svm_args *>(buffer.data());
  auto *attrs = reinterpret_cast<kfd_ioctl_svm_attribute *>(svm_args + 1);

  svm_args->start_addr = kStart;
  svm_args->size = kSize;
  svm_args->op = KFD_IOCTL_SVM_OP_SET_ATTR;
  svm_args->nattr = 2;
  attrs[0].type = KFD_IOCTL_SVM_ATTR_PREFERRED_LOC;
  attrs[0].value = kGpuId;
  attrs[1].type = KFD_IOCTL_SVM_ATTR_SET_FLAGS;
  attrs[1].value = KFD_IOCTL_SVM_FLAG_GPU_EXEC;

  unsigned long svm_request = rocjitsu::ioctl_with_size(AMDKFD_IOC_SVM, buffer.size());
  EXPECT_TRUE(rocjitsu::is_svm_ioctl(svm_request));
  EXPECT_EQ(rocjitsu::canonical_ioctl_request(svm_request), AMDKFD_IOC_SVM);
  int rc = driver_->ioctl(svm_request, svm_args);
  EXPECT_EQ(rc, 0);

  svm_args->op = KFD_IOCTL_SVM_OP_GET_ATTR;
  attrs[0].value = 0;
  attrs[1].value = 0;

  rc = driver_->ioctl(svm_request, svm_args);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(attrs[0].value, kGpuId);
  EXPECT_EQ(attrs[1].value, KFD_IOCTL_SVM_FLAG_GPU_EXEC);
}

TEST_F(KfdIoctlTest, RuntimeEnableAndDisable) {
  kfd_ioctl_runtime_enable_args args{};
  args.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  args.r_debug = 0xfeed'beef;

  int rc = driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &args);
  EXPECT_EQ(rc, 0);
  EXPECT_NE(args.capabilities_mask & KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK, 0u);

  args.mode_mask = 0;
  rc = driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &args);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(args.capabilities_mask, 0u);
}

// Models the interposer's fd lifecycle: the primary KFD fd plus every dup each
// hold one open reference, so the process must survive until the LAST fd is
// closed, not the first. retain_local_open() is what the interposer calls when
// it tracks a dup; close() is what it calls per fd close.
TEST_F(KfdIoctlTest, OpenRefcountSurvivesDupThenPrimaryClose) {
  // SetUp() already performed the primary open().
  EXPECT_EQ(driver_->local_open_ref_count(), 1u);

  // Two dups of the KFD fd. Each retain must succeed while the process is live.
  EXPECT_TRUE(driver_->retain_local_open());
  EXPECT_TRUE(driver_->retain_local_open());
  EXPECT_EQ(driver_->local_open_ref_count(), 3u);

  // Closing the primary fd first must NOT tear the process down.
  driver_->close();
  EXPECT_EQ(driver_->local_open_ref_count(), 2u);

  // Closing the first dup: still alive.
  driver_->close();
  EXPECT_EQ(driver_->local_open_ref_count(), 1u);

  // Closing the last dup: now the process is destroyed.
  driver_->close();
  EXPECT_EQ(driver_->local_open_ref_count(), 0u);

  // Re-open so the fixture's TearDown close() is balanced.
  ASSERT_GE(driver_->open(), 0);
}

// --- AMDKFD_IOC_DBG_TRAP dispatch skeleton (self-debug in local mode) ---

TEST_F(KfdIoctlTest, DbgTrapUnknownPidReturnsESRCH) {
  kfd_ioctl_dbg_trap_args args{};
  args.pid = 0x7fffffff; // a pid that maps to no emulated process
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -ESRCH);
}

TEST_F(KfdIoctlTest, DbgTrapInvalidPidReturnsESRCH) {
  kfd_ioctl_dbg_trap_args args{};
  args.pid = UINT32_MAX;
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -ESRCH);
}

TEST_F(KfdIoctlTest, DbgTrapOpBeforeEnableReturnsEINVAL) {
  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(getpid());
  args.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -EINVAL);
}

TEST_F(KfdIoctlTest, DbgTrapBareDisableReturnsEINVAL) {
  // DISABLE with no active session has nothing to tear down. The by-pid gate's
  // DISABLE exemption only skips the cross-process authorization check, not the
  // session-enabled requirement, so a bare DISABLE is still rejected with
  // EINVAL like any other non-ENABLE op on a disabled session.
  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(getpid());
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &dis), -EINVAL);
}

TEST_F(KfdIoctlTest, DbgTrapEnablePopulatesRuntimeInfoThenDisable) {
  // ROCr's runtime-enable must have run for the session to report ENABLED.
  kfd_ioctl_runtime_enable_args rt{};
  rt.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK | KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK;
  rt.r_debug = 0xcafef00d;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &rt), 0);

  kfd_runtime_info info{};
  info.runtime_state = 0xdeadbeef; // sentinel the driver must overwrite
  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(getpid());
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.dbg_fd = make_debug_fd();
  args.enable.rinfo_ptr = reinterpret_cast<uint64_t>(&info);
  args.enable.rinfo_size = sizeof(info);

  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), 0);
  EXPECT_EQ(args.enable.rinfo_size, sizeof(kfd_runtime_info));
  EXPECT_EQ(info.runtime_state, static_cast<uint32_t>(DEBUG_RUNTIME_STATE_ENABLED));
  EXPECT_EQ(info.r_debug, 0xcafef00dULL);
  EXPECT_EQ(info.ttmp_setup, 1u);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(getpid());
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &dis), 0);
}

TEST_F(KfdIoctlTest, DbgTrapDoubleEnableReturnsEALREADY) {
  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(getpid());
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), 0);
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -EALREADY);
}

// Hammers ENABLE/DISABLE on one session from many threads to exercise
// debug_sessions_mutex_ under ThreadSanitizer. Races are legitimate: a losing ENABLE
// sees EALREADY and a losing DISABLE sees EINVAL. The invariant is that the
// driver serializes them without a data race or torn session state — every call
// returns one of the well-defined codes, never a crash or a bogus errno. Uses
// self-debug (target pid == getpid()) so the whole cycle stays on debug_sessions_mutex_
// and runtime_mutex_. In local mode the session never owns dbg_fd, so a single
// shared eventfd can back every ENABLE.
TEST_F(KfdIoctlTest, DbgTrapConcurrentEnableDisableIsRaceFree) {
  const int fd = make_debug_fd();
  const auto pid = static_cast<uint32_t>(getpid());
  constexpr int kThreads = 8;
  constexpr int kIters = 250;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        if ((t + i) & 1) {
          kfd_ioctl_dbg_trap_args en{};
          en.pid = pid;
          en.op = KFD_IOC_DBG_TRAP_ENABLE;
          en.enable.dbg_fd = fd;
          const int rc = driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en);
          EXPECT_TRUE(rc == 0 || rc == -EALREADY) << "enable rc=" << rc;
        } else {
          kfd_ioctl_dbg_trap_args dis{};
          dis.pid = pid;
          dis.op = KFD_IOC_DBG_TRAP_DISABLE;
          const int rc = driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &dis);
          EXPECT_TRUE(rc == 0 || rc == -EINVAL) << "disable rc=" << rc;
        }
      }
    });
  }
  for (auto &th : threads)
    th.join();
}

TEST_F(KfdIoctlTest, DbgTrapEnableBadFdReturnsEBADF) {
  // kfd_dbg_trap_enable() fails with -EBADF when it cannot fget(dbg_fd); an
  // unusable notification target must not be stored on the session.
  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(getpid());
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.dbg_fd = KFD_INVALID_FD;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -EBADF);

  // The rejected enable left the session disabled, so a follow-up op is refused
  // with -EINVAL rather than admitted against a half-initialized session.
  kfd_ioctl_dbg_trap_args after{};
  after.pid = static_cast<uint32_t>(getpid());
  after.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &after), -EINVAL);
}

// The driver signals the notifier to wake the debugger, so a read-only
// descriptor is an unusable target even though it is a valid open fd. ENABLE
// validates the access mode (fcntl F_GETFL) and rejects a non-writable fd with
// -EBADF, matching a closed one; it must not be stored on the session.
TEST_F(KfdIoctlTest, DbgTrapEnableReadOnlyFdReturnsEBADF) {
  const int ro_fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(ro_fd, 0);
  debug_fds_.push_back(ro_fd); // closed in TearDown

  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(getpid());
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.dbg_fd = static_cast<uint32_t>(ro_fd);
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -EBADF);

  // The rejected enable left the session disabled.
  kfd_ioctl_dbg_trap_args after{};
  after.pid = static_cast<uint32_t>(getpid());
  after.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &after), -EINVAL);
}

TEST_F(KfdIoctlTest, DbgTrapHwOpWithoutRuntimeReturnsEPERM) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // SET_FLAGS is a DBG_HW_OP: it requires AMDKFD_IOC_RUNTIME_ENABLE first.
  kfd_ioctl_dbg_trap_args flags{};
  flags.pid = static_cast<uint32_t>(getpid());
  flags.op = KFD_IOC_DBG_TRAP_SET_FLAGS;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &flags), -EPERM);
}

TEST_F(KfdIoctlTest, DbgTrapWatchBadGpuReturnsENODEV) {
  kfd_ioctl_runtime_enable_args rt{};
  rt.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &rt), 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // Runtime is enabled, so the HW-op gate passes and the gpu-id check runs.
  kfd_ioctl_dbg_trap_args watch{};
  watch.pid = static_cast<uint32_t>(getpid());
  watch.op = KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH;
  watch.set_node_address_watch.gpu_id = 0xdeadbeef;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &watch), -ENODEV);
}

TEST_F(KfdIoctlTest, DbgTrapQueryDebugEventReportsIdle) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // QUERY_DEBUG_EVENT is not a HW-op, so it is available before runtime enable.
  // EAGAIN is the kernel contract when no exception is pending.
  kfd_ioctl_dbg_trap_args q{};
  q.pid = static_cast<uint32_t>(getpid());
  q.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &q), -EAGAIN);

  // SEND_RUNTIME_EVENT acknowledges the runtime-enable debugger handshake.
  kfd_ioctl_dbg_trap_args event{};
  event.pid = static_cast<uint32_t>(getpid());
  event.op = KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &event), 0);
}

TEST_F(KfdIoctlTest, DbgTrapQueryRuntimeExceptionInfoClampsAndPopulates) {
  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK | KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK;
  runtime.r_debug = 0x123456789abcdef0ULL;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::array<uint8_t, sizeof(kfd_runtime_info) + 8> buffer;
  buffer.fill(0xA5);
  kfd_ioctl_dbg_trap_args query{};
  query.pid = static_cast<uint32_t>(getpid());
  query.op = KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO;
  query.query_exception_info.exception_code = EC_PROCESS_RUNTIME;
  query.query_exception_info.info_ptr = reinterpret_cast<uint64_t>(buffer.data());
  query.query_exception_info.info_size = sizeof(kfd_runtime_info);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &query), 0);
  EXPECT_EQ(query.query_exception_info.info_size, sizeof(kfd_runtime_info));

  kfd_runtime_info info{};
  std::memcpy(&info, buffer.data(), sizeof(info));
  EXPECT_EQ(info.r_debug, runtime.r_debug);
  EXPECT_EQ(info.runtime_state, static_cast<uint32_t>(DEBUG_RUNTIME_STATE_ENABLED));
  EXPECT_EQ(info.ttmp_setup, 1u);
  EXPECT_TRUE(std::all_of(buffer.begin() + sizeof(info), buffer.end(),
                          [](uint8_t byte) { return byte == 0xA5; }));
}

// rocdbgapi has to learn that a process's runtime went away. Otherwise it keeps
// driving queues of a process in teardown, and those ops are gated on the
// runtime being enabled: they answer -EPERM, which rocdbgapi escalates to a
// fatal os_driver::resume_queues failure and GDB to an internal error
// (gdb.rocm/multi-inferior-stress.exp).
//
// The target here debugs itself, which is the one shape that reports without
// blocking: the ack would have to come from the thread already inside this
// ioctl. That keeps the test to what a single process can observe -- the event,
// the notifier, and the toggled state rocdbgapi reads back.
TEST_F(KfdIoctlTest, RuntimeDisableReportsProcessRuntimeTransition) {
  const auto pid = static_cast<uint32_t>(getpid());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  const int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = pid;
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_PROCESS_RUNTIME);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  // Attaching to an already-enabled runtime reports it through the returned
  // runtime_info rather than the notifier, but drain anyway so the read below
  // can only be answering the disable.
  uint64_t drained = 0;
  // EAGAIN is the expected answer -- the notifier is EFD_NONBLOCK and the attach
  // does not post to it -- so the result is deliberately not asserted on.
  [[maybe_unused]] const ssize_t drained_rc = ::read(notifier, &drained, sizeof(drained));

  kfd_ioctl_runtime_enable_args disable{};
  disable.mode_mask = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &disable), 0);

  uint64_t notifications = 0;
  ASSERT_EQ(::read(notifier, &notifications, sizeof(notifications)),
            static_cast<ssize_t>(sizeof(notifications)))
      << strerror(errno);
  EXPECT_EQ(notifications, 1u);

  kfd_ioctl_dbg_trap_args query{};
  query.pid = pid;
  query.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
  query.query_debug_event.exception_mask = KFD_EC_MASK(EC_PROCESS_RUNTIME);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &query), 0);
  EXPECT_NE(query.query_debug_event.exception_mask & KFD_EC_MASK(EC_PROCESS_RUNTIME), 0u);

  // The state behind the event has to have toggled: rocdbgapi treats a
  // runtime_state that reads the same as before as a spurious runtime
  // exception and aborts.
  kfd_runtime_info info{};
  kfd_ioctl_dbg_trap_args info_query{};
  info_query.pid = pid;
  info_query.op = KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO;
  info_query.query_exception_info.exception_code = EC_PROCESS_RUNTIME;
  info_query.query_exception_info.info_ptr = reinterpret_cast<uint64_t>(&info);
  info_query.query_exception_info.info_size = sizeof(info);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &info_query), 0);
  EXPECT_EQ(info.runtime_state, static_cast<uint32_t>(DEBUG_RUNTIME_STATE_DISABLED));

  // The gate the report exists to warn about still stands for anything that
  // would touch hardware.
  kfd_ioctl_dbg_trap_args launch_mode{};
  launch_mode.pid = pid;
  launch_mode.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE;
  launch_mode.launch_mode.launch_mode = KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &launch_mode), -EPERM);

  // A resume naming a queue the process does not have is the one shape that is
  // answered instead: it asks nothing of the hardware, and the per-queue INVALID
  // bit is how rocdbgapi learns to retire the queue rather than escalating the
  // refusal to a fatal error.
  constexpr uint32_t kQueueInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  uint32_t queue_id = 7;
  kfd_ioctl_dbg_trap_args resume{};
  resume.pid = pid;
  resume.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  resume.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  resume.resume_queues.num_queues = 1;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &resume), 0);
  EXPECT_NE(queue_id & kQueueInvalid, 0u);
}

TEST_F(KfdIoctlTest, DbgTrapAttachDetachConfigOpsValidateAndResetState) {
  kfd_ioctl_runtime_enable_args rt{};
  rt.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &rt), 0);

  const auto pid = static_cast<uint32_t>(getpid());
  auto enable = [&] {
    kfd_ioctl_dbg_trap_args en{};
    en.pid = pid;
    en.op = KFD_IOC_DBG_TRAP_ENABLE;
    en.enable.dbg_fd = make_debug_fd();
    return driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en);
  };
  ASSERT_EQ(enable(), 0);

  kfd_ioctl_dbg_trap_args flags{};
  flags.pid = pid;
  flags.op = KFD_IOC_DBG_TRAP_SET_FLAGS;
  flags.set_flags.flags = KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &flags), 0);
  EXPECT_EQ(flags.set_flags.flags, 0u);
  flags.set_flags.flags = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &flags), 0);
  EXPECT_EQ(flags.set_flags.flags, static_cast<uint32_t>(KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP));

  kfd_ioctl_dbg_trap_args mode{};
  mode.pid = pid;
  mode.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE;
  mode.launch_mode.launch_mode = 2;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &mode), -EINVAL);
  mode.launch_mode.launch_mode = KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &mode), 0);

  kfd_ioctl_dbg_trap_args override_args{};
  override_args.pid = pid;
  override_args.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE;
  override_args.launch_override.override_mode = KFD_DBG_TRAP_OVERRIDE_OR;
  override_args.launch_override.enable_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  override_args.launch_override.support_request_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &override_args), 0);
  EXPECT_EQ(override_args.launch_override.enable_mask, 0u);
  EXPECT_EQ(override_args.launch_override.support_request_mask,
            static_cast<uint32_t>(KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH));

  override_args.launch_override.override_mode = 99;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &override_args), -EINVAL);

  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = pid;
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);
  EXPECT_EQ(snapshot.queue_snapshot.num_queues, 0u);
  EXPECT_EQ(snapshot.queue_snapshot.entry_size, sizeof(kfd_queue_snapshot_entry));

  kfd_ioctl_dbg_trap_args queues{};
  queues.pid = pid;
  queues.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  std::array<uint32_t, 2> queue_ids{17, 23};
  queues.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(queue_ids.data());
  queues.suspend_queues.num_queues = queue_ids.size();
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &queues), 0);
  constexpr uint32_t kInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  EXPECT_EQ(queue_ids, (std::array<uint32_t, 2>{17 | kInvalid, 23 | kInvalid}));
  queues.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  queues.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(queue_ids.data());
  queues.resume_queues.num_queues = queue_ids.size();
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &queues), 0);
  EXPECT_EQ(queue_ids, (std::array<uint32_t, 2>{17 | kInvalid, 23 | kInvalid}));

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = pid;
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &dis), 0);
  ASSERT_EQ(enable(), 0);

  flags.set_flags.flags = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &flags), 0);
  EXPECT_EQ(flags.set_flags.flags, 0u);
  override_args.launch_override.override_mode = KFD_DBG_TRAP_OVERRIDE_OR;
  override_args.launch_override.enable_mask = 0;
  override_args.launch_override.support_request_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &override_args), 0);
  EXPECT_EQ(override_args.launch_override.enable_mask, 0u);
  EXPECT_EQ(override_args.launch_override.support_request_mask,
            static_cast<uint32_t>(KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH));
}

TEST_F(KfdIoctlTest, DbgTrapQueueControlOnlyChangesRequestedQueue) {
  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::array<std::vector<uint8_t>, 2> rings{std::vector<uint8_t>(4096), std::vector<uint8_t>(4096)};
  std::array<uint64_t, 2> read_pointers{};
  std::array<uint64_t, 2> write_pointers{};
  std::array<kfd_ioctl_create_queue_args, 2> queues{};
  for (size_t index = 0; index < queues.size(); ++index) {
    auto &queue = queues[index];
    queue.gpu_id = kGpuId;
    queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    queue.ring_base_address = reinterpret_cast<uint64_t>(rings[index].data());
    queue.ring_size = static_cast<uint32_t>(rings[index].size());
    queue.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointers[index]);
    queue.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointers[index]);
    ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
  }

  std::array<kfd_queue_snapshot_entry, 2> snapshots{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(snapshots.data());
  snapshot.queue_snapshot.num_queues = snapshots.size();
  snapshot.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  uint32_t queue_id = queues[0].queue_id;
  kfd_ioctl_dbg_trap_args suspend{};
  suspend.pid = static_cast<uint32_t>(getpid());
  suspend.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  suspend.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  suspend.suspend_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &suspend), 1);

  bool requested_suspended = false;
  bool unrelated_suspended = false;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    requested_suspended |=
        cp->queue_debug_suspended_for_test(queues[0].queue_id, driver_->local_process_id());
    unrelated_suspended |=
        cp->queue_debug_suspended_for_test(queues[1].queue_id, driver_->local_process_id());
  });
  EXPECT_TRUE(requested_suspended);
  EXPECT_FALSE(unrelated_suspended);

  suspend.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  suspend.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  suspend.resume_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &suspend), 1);
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    EXPECT_FALSE(
        cp->queue_debug_suspended_for_test(queues[0].queue_id, driver_->local_process_id()));
  });
}

TEST_F(KfdIoctlTest, DbgTrapWaveLaunchOverrideValidatesRequest) {
  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  kfd_ioctl_dbg_trap_args request{};
  request.pid = static_cast<uint32_t>(getpid());
  request.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE;
  request.launch_override.override_mode = KFD_DBG_TRAP_OVERRIDE_REPLACE;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &request), -EINVAL);

  request.launch_override.override_mode = KFD_DBG_TRAP_OVERRIDE_OR;
  request.launch_override.support_request_mask =
      KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH | KFD_DBG_TRAP_MASK_FP_INVALID;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &request), -EACCES);

  request.launch_override.enable_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  request.launch_override.support_request_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &request), 0);
  EXPECT_EQ(request.launch_override.enable_mask, 0u);
  EXPECT_EQ(request.launch_override.support_request_mask,
            static_cast<uint32_t>(KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH));
}

TEST_F(KfdIoctlTest, DbgTrapNodeAddressWatchAllocatesAndFreesSlots) {
  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);
  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::array<uint32_t, 4> slots{};
  for (uint32_t index = 0; index < slots.size(); ++index) {
    kfd_ioctl_dbg_trap_args watch{};
    watch.pid = static_cast<uint32_t>(getpid());
    watch.op = KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH;
    watch.set_node_address_watch.gpu_id = kGpuId;
    watch.set_node_address_watch.address = 0x1000 + index * 0x100;
    watch.set_node_address_watch.mask = ~0xFFu;
    watch.set_node_address_watch.mode = KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL;
    ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &watch), 0);
    slots[index] = watch.set_node_address_watch.id;
  }
  EXPECT_EQ(slots, (std::array<uint32_t, 4>{0, 1, 2, 3}));

  kfd_ioctl_dbg_trap_args exhausted{};
  exhausted.pid = static_cast<uint32_t>(getpid());
  exhausted.op = KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH;
  exhausted.set_node_address_watch.gpu_id = kGpuId;
  exhausted.set_node_address_watch.mode = KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &exhausted), -ENOMEM);

  kfd_ioctl_dbg_trap_args clear{};
  clear.pid = static_cast<uint32_t>(getpid());
  clear.op = KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH;
  clear.clear_node_address_watch.gpu_id = kGpuId;
  clear.clear_node_address_watch.id = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &clear), 0);
  exhausted.set_node_address_watch.address = 0x9000;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &exhausted), 0);
  EXPECT_EQ(exhausted.set_node_address_watch.id, 1u);
  clear.clear_node_address_watch.id = 4;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &clear), -EINVAL);
}

TEST(KfdAddressWatchTest, KfdMaskPreservesImplicitUpperCompareBits) {
  constexpr uint64_t watched_address = 0x00007FFF12345000ULL;
  constexpr auto watch = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      watched_address, 0xFFFFFF80u, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL);

  EXPECT_EQ(watch.mask, 0xFFFFFFFFFFFFFF80ULL);
  EXPECT_TRUE(watch.overlaps(watched_address + 4, sizeof(uint32_t)));
  EXPECT_FALSE(watch.overlaps(watched_address + 128, sizeof(uint32_t)));
  EXPECT_FALSE(watch.overlaps(watched_address ^ (uint64_t{1} << 32), sizeof(uint32_t)))
      << "equal low 32 bits must not alias a different upper address";
}

TEST(KfdAddressWatchTest, OneAccessReportsEveryMatchingHardwareSlot) {
  constexpr uint64_t address = 0x00007FFF12345013ULL;
  rocjitsu::KfdProcess::DebugSession session;
  session.address_watches[0] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address & ~uint64_t{3}, 0xFFFFFFFCu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL);
  session.address_watches[1] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address, 0xFFFFFFFFu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD);
  session.address_watches[2] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address, 0xFFFFFFFFu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL);
  session.address_watches[3] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address, 0xFFFFFFFFu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ);

  // The mode sets the driver hands down, mirroring on_wave_watchpoint().
  constexpr uint32_t kAll = uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL;
  constexpr uint32_t kWriteModes = kAll | (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD);
  constexpr uint32_t kReadModes = kAll | (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ);
  EXPECT_EQ(session.matching_address_watch_slots(address, 1, kWriteModes), 0b0111u);
  EXPECT_EQ(session.matching_address_watch_slots(address, 1, kReadModes), 0b1101u);
}

// kfd_ioctl.h documents NONREAD as "write or atomic operations only", and
// rocdbgapi maps the STORE_AND_RMW kind a plain `(gdb) watch` requests onto it
// (os_driver.h). An atomic read-modify-write must therefore trip a NONREAD
// watch: comparing the programmed mode against the access mode for equality
// silently dropped every watchpoint on a variable updated by atomicAdd().
TEST(KfdAddressWatchTest, AtomicAccessTripsNonreadAndAtomicWatches) {
  constexpr uint64_t address = 0x00007FFF12345013ULL;
  rocjitsu::KfdProcess::DebugSession session;
  session.address_watches[0] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address, 0xFFFFFFFFu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD);
  session.address_watches[1] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address, 0xFFFFFFFFu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC);
  session.address_watches[2] = rocjitsu::KfdProcess::DebugSession::AddressWatch::from_kfd(
      address, 0xFFFFFFFFu, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ);

  constexpr uint32_t kAll = uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL;
  constexpr uint32_t kNonread = uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD;
  constexpr uint32_t kAtomic = uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC;

  EXPECT_EQ(session.matching_address_watch_slots(address, 4, kAll | kNonread | kAtomic), 0b011u)
      << "an atomic RMW must trip both the write watch and the atomic watch";
  // A plain write is not an atomic, so it must not trip the ATOMIC-only watch.
  EXPECT_EQ(session.matching_address_watch_slots(address, 4, kAll | kNonread), 0b001u);
}

// Local mode borrows the debugger's own fd (the session does not own it), so
// DISABLE must leave it open for the debugger to close. Only daemon mode, which
// dup'd the fd via SCM_RIGHTS, releases it on teardown.
TEST_F(KfdIoctlTest, DbgTrapLocalDisableLeavesDebuggerFdOpen) {
  const int fd = make_debug_fd();
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = fd;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(getpid());
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &dis), 0);

  EXPECT_NE(fcntl(fd, F_GETFD), -1) << "local-mode DISABLE must not close the debugger's fd";
}

TEST_F(KfdIoctlTest, DbgTrapDisableResumesStoppedQueue) {
  const int fd = make_debug_fd();
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = fd;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  rocjitsu::amdgpu::Wavefront *wave = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (wave != nullptr || cp->compute_units().empty())
      return;
    wave = cp->compute_units().front()->dispatch_wf(/*wg_id=*/0, /*pc=*/0x600000000ULL,
                                                    /*sgprs=*/16, /*vgprs=*/4);
  });
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(driver_->local_process_id());
  wave->set_queue_id(create.queue_id);
  wave->set_debug_halted(true);
  wave->set_debug_suspended(true);
  wave->set_debug_single_step(true);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(getpid());
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &dis), 0);

  EXPECT_FALSE(wave->debug_halted());
  EXPECT_FALSE(wave->debug_suspended());
  EXPECT_FALSE(wave->debug_single_step());
}

// The kernel copies min(user_size, sizeof(runtime_info)) bytes back and reports
// the full struct size. An undersized buffer must truncate the copy — never
// writing past the caller's buffer — while still reporting sizeof(kfd_runtime_info).
TEST_F(KfdIoctlTest, DbgTrapEnableUndersizedRuntimeInfoTruncates) {
  kfd_ioctl_runtime_enable_args rt{};
  rt.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  rt.r_debug = 0xcafef00d;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &rt), 0);

  // Buffer smaller than kfd_runtime_info, backed by a full-size array so an
  // overrunning copy is caught by the sentinel check below.
  constexpr uint32_t kSmall = 8;
  static_assert(kSmall < sizeof(kfd_runtime_info));
  constexpr uint8_t kSentinel = 0xCD;
  std::array<uint8_t, sizeof(kfd_runtime_info)> buf;
  buf.fill(kSentinel);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  en.enable.rinfo_ptr = reinterpret_cast<uint64_t>(buf.data());
  en.enable.rinfo_size = kSmall;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  EXPECT_EQ(en.enable.rinfo_size, sizeof(kfd_runtime_info)); // full size reported
  for (size_t i = kSmall; i < buf.size(); ++i)
    EXPECT_EQ(buf[i], kSentinel) << "runtime-info copy overran the undersized buffer at byte " << i;
}

// Exercises the RemoteDriver client stub against an in-process server that runs
// the real daemon-mode handler. A debugger may hand kfd_dbg_trap_enable a
// runtime-info buffer larger than kfd_runtime_info; the handler fills only
// sizeof(kfd_runtime_info) and reports that size, so bytes past it must survive
// the RPC round trip (local mode preserves them; the daemon path used to clobber
// them). Routing through RemoteDriver also locks in the DBG_TRAP embedded-pointer
// marshalling — a crash there would take the server thread, and thus this test
// process, down.
TEST_F(KfdIoctlTest, DbgTrapEnableOversizedRuntimeInfoPreservesTailInDaemonMode) {
  ASSERT_NE(soc_, nullptr);

  rocjitsu::SimulatedKfd daemon_driver(*soc_, /*daemon_mode=*/true);
  const pid_t kClientPid = getpid();
  uint32_t process_id = daemon_driver.open_process(kClientPid);
  ASSERT_NE(process_id, 0u);

  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);

  // Minimal stand-in for the daemon's RPC_IOCTL loop: reconstruct the inlined
  // runtime-info pointer exactly as tools/rocjitsu does, run the real handler,
  // then echo the args (plus any inline tail) back to the client. jthread (not
  // thread) so an ASSERT_* failure below unwinds without calling
  // std::terminate() on a still-joinable thread.
  std::jthread server([&, server_fd = sv[1]] {
    for (;;) {
      rocjitsu::RpcHeader hdr{};
      int in_fds[3] = {-1, -1, -1};
      size_t num_in = 3;
      if (rocjitsu::rpc_recv_msg(server_fd, &hdr, sizeof(hdr), in_fds, &num_in) <= 0)
        break;
      if (hdr.opcode != rocjitsu::RPC_IOCTL) {
        rocjitsu::RpcHeader resp{};
        resp.request_id = hdr.request_id;
        rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
        if (hdr.opcode == rocjitsu::RPC_CLOSE)
          break;
        continue;
      }
      std::vector<uint8_t> payload(hdr.payload_bytes);
      if (!rocjitsu::rpc_recv_exact(server_fd, payload.data(), hdr.payload_bytes))
        break;
      auto *ireq = reinterpret_cast<rocjitsu::RpcIoctlRequest *>(payload.data());
      const uint32_t cmd = ireq->ioctl_cmd;
      const size_t buf_size = ireq->args_bytes;
      uint8_t *buf = payload.data() + sizeof(rocjitsu::RpcIoctlRequest);

      const size_t arg_size = rocjitsu::ioctl_arg_size(cmd);
      if (cmd == AMDKFD_IOC_DBG_TRAP && buf_size > arg_size) {
        auto *dbg = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(buf);
        if (dbg->op == KFD_IOC_DBG_TRAP_ENABLE)
          dbg->enable.rinfo_ptr = reinterpret_cast<uint64_t>(buf + arg_size);
      }

      int mem_fd = num_in > 1 ? in_fds[1] : -1;
      const int proc_fd = num_in > 2 ? in_fds[2] : -1;
      const int result = daemon_driver.ioctl(process_id, cmd, buf, &mem_fd, proc_fd);
      if (num_in > 0 && in_fds[0] >= 0)
        ::close(in_fds[0]);
      if (mem_fd >= 0)
        ::close(mem_fd);
      if (proc_fd >= 0)
        ::close(proc_fd);

      rocjitsu::RpcHeader resp{};
      resp.opcode = rocjitsu::RPC_IOCTL;
      resp.request_id = hdr.request_id;
      resp.result = result;
      resp.payload_bytes = static_cast<uint32_t>(buf_size);
      if (!rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp)))
        break;
      if (buf_size > 0 && !rocjitsu::rpc_send_exact(server_fd, buf, buf_size))
        break;
    }
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  // Runtime-enable so the session reports ENABLED and carries r_debug/ttmp.
  kfd_ioctl_runtime_enable_args rt{};
  rt.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK | KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK;
  rt.r_debug = 0xcafef00d;
  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &rt), 0);

  // Oversized runtime-info buffer: the 16-byte struct plus a 32-byte tail,
  // pre-filled with a sentinel the handler must leave untouched.
  constexpr size_t kCapacity = sizeof(kfd_runtime_info) + 32;
  constexpr uint8_t kSentinel = 0xAB;
  std::array<uint8_t, kCapacity> rinfo_buf;
  rinfo_buf.fill(kSentinel);

  // A live fd for the now-active daemon-mode validation; the daemon adopts it on
  // ENABLE and releases it on DISABLE (RAII), so it is not tracked/closed here.
  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);

  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(kClientPid);
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.dbg_fd = notifier;
  args.enable.rinfo_ptr = reinterpret_cast<uint64_t>(rinfo_buf.data());
  args.enable.rinfo_size = static_cast<uint32_t>(kCapacity);

  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &args), 0);

  // Returned info: the handler reports the true struct size, not the capacity,
  // and fills the runtime state the debugger expects.
  EXPECT_EQ(args.enable.rinfo_size, sizeof(kfd_runtime_info));
  kfd_runtime_info info{};
  std::memcpy(&info, rinfo_buf.data(), sizeof(info));
  EXPECT_EQ(info.runtime_state, static_cast<uint32_t>(DEBUG_RUNTIME_STATE_ENABLED));
  EXPECT_EQ(info.r_debug, 0xcafef00dULL);
  EXPECT_EQ(info.ttmp_setup, 1u);

  // Tail: every byte past the struct must retain the sentinel.
  for (size_t i = sizeof(kfd_runtime_info); i < kCapacity; ++i)
    EXPECT_EQ(rinfo_buf[i], kSentinel) << "runtime-info tail clobbered at byte " << i;

  // Daemon liveness: a follow-up ioctl still round-trips, proving the server
  // survived the embedded-pointer marshalling and is still serving requests.
  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(kClientPid);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &dis), 0);

  rd.close(); // sends RPC_CLOSE so the server loop exits
  server.join();
  EXPECT_EQ(daemon_driver.close(process_id), 0);
}

// Serializing an ioctl with an embedded input array grows the request vector.
// Capture the application pointer before that resize: pointers into the vector's
// copied argument area are invalidated when its storage is reallocated.
TEST(RemoteDriverEmbeddedArrayTest, MapMemorySerializesDeviceIdsAcrossBufferGrowth) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);

  constexpr std::array<uint32_t, 2> device_ids{38144, 38145};
  std::jthread server([server_fd = sv[1], device_ids] {
    rocjitsu::RpcHeader hdr{};
    ASSERT_TRUE(rocjitsu::rpc_recv_exact(server_fd, &hdr, sizeof(hdr)));
    ASSERT_EQ(hdr.opcode, rocjitsu::RPC_IOCTL);
    std::vector<uint8_t> payload(hdr.payload_bytes);
    ASSERT_TRUE(rocjitsu::rpc_recv_exact(server_fd, payload.data(), payload.size()));

    const auto *request = reinterpret_cast<const rocjitsu::RpcIoctlRequest *>(payload.data());
    EXPECT_EQ(request->ioctl_cmd, AMDKFD_IOC_MAP_MEMORY_TO_GPU);
    const auto *args = reinterpret_cast<const kfd_ioctl_map_memory_to_gpu_args *>(payload.data() +
                                                                                  sizeof(*request));
    const auto *inline_ids =
        reinterpret_cast<const uint32_t *>(payload.data() + sizeof(*request) + sizeof(*args));
    ASSERT_EQ(args->n_devices, device_ids.size());
    EXPECT_TRUE(std::equal(device_ids.begin(), device_ids.end(), inline_ids));

    rocjitsu::RpcHeader response{};
    response.opcode = rocjitsu::RPC_IOCTL;
    response.request_id = hdr.request_id;
    response.result = -EINVAL;
    ASSERT_TRUE(rocjitsu::rpc_send_exact(server_fd, &response, sizeof(response)));
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  kfd_ioctl_map_memory_to_gpu_args args{};
  args.device_ids_array_ptr = reinterpret_cast<uint64_t>(device_ids.data());
  args.n_devices = device_ids.size();
  EXPECT_EQ(driver.ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &args), -EINVAL);
}

TEST(RemoteDriverEmbeddedArrayTest, WaitEventsSerializesEventsAcrossBufferGrowth) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);

  std::array<kfd_event_data, 2> events{};
  events[0].event_id = 17;
  events[1].event_id = 29;
  std::jthread server([server_fd = sv[1]] {
    rocjitsu::RpcHeader hdr{};
    ASSERT_TRUE(rocjitsu::rpc_recv_exact(server_fd, &hdr, sizeof(hdr)));
    ASSERT_EQ(hdr.opcode, rocjitsu::RPC_IOCTL);
    std::vector<uint8_t> payload(hdr.payload_bytes);
    ASSERT_TRUE(rocjitsu::rpc_recv_exact(server_fd, payload.data(), payload.size()));

    const auto *request = reinterpret_cast<const rocjitsu::RpcIoctlRequest *>(payload.data());
    EXPECT_EQ(request->ioctl_cmd, AMDKFD_IOC_WAIT_EVENTS);
    const auto *args =
        reinterpret_cast<const kfd_ioctl_wait_events_args *>(payload.data() + sizeof(*request));
    const auto *inline_events =
        reinterpret_cast<const kfd_event_data *>(payload.data() + sizeof(*request) + sizeof(*args));
    ASSERT_EQ(args->num_events, 2u);
    EXPECT_EQ(inline_events[0].event_id, 17u);
    EXPECT_EQ(inline_events[1].event_id, 29u);

    rocjitsu::RpcHeader response{};
    response.opcode = rocjitsu::RPC_IOCTL;
    response.request_id = hdr.request_id;
    response.result = -EINVAL;
    ASSERT_TRUE(rocjitsu::rpc_send_exact(server_fd, &response, sizeof(response)));
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  kfd_ioctl_wait_events_args args{};
  args.events_ptr = reinterpret_cast<uint64_t>(events.data());
  args.num_events = events.size();
  args.timeout = 1;
  EXPECT_EQ(driver.ioctl(AMDKFD_IOC_WAIT_EVENTS, &args), -EINVAL);
}

// --- Authorizing the daemon to reach into this address space ---
//
// A client names the daemon a permitted ptracer so the daemon's memory bridge
// can service GPU access to host memory it holds no mapping for. That grant is
// process-wide and outlives the connection, so who receives it, and when, is a
// trust decision rather than a detail. These cover the decision itself; the
// syscall behind it is the kernel's.

namespace {

/// @brief Serve one RPC_HANDSHAKE and stop.
/// @param[in] fd Server end of the socketpair.
/// @param[in] result Header result: nonzero makes the client reject the peer.
void serve_one_handshake(int fd, int32_t result) {
  rocjitsu::RpcHeader request{};
  if (!rocjitsu::rpc_recv_exact(fd, &request, sizeof(request)))
    return;
  rocjitsu::RpcHeader response{};
  response.opcode = rocjitsu::RPC_HANDSHAKE;
  response.request_id = request.request_id;
  response.result = result;
  response.payload_bytes =
      result == 0 ? static_cast<uint32_t>(sizeof(rocjitsu::RpcHandshakeResponse)) : 0;
  if (!rocjitsu::rpc_send_exact(fd, &response, sizeof(response)) || result != 0)
    return;
  rocjitsu::RpcHandshakeResponse payload{};
  payload.version = rocjitsu::kRpcProtocolVersion;
  rocjitsu::rpc_send_exact(fd, &payload, sizeof(payload));
}

} // namespace

// Both ends of a socketpair belong to this process, so SO_PEERCRED reports this
// PID -- which is what makes the launcher's PID the only variable here.
TEST(RemoteDriverPtracerGrantTest, AuthorizesOnlyTheDaemonItsLauncherStarted) {
  const rocjitsu::test::ScopedEnvironmentVariable daemon_pid(rocjitsu::kRpcDaemonPidEnv,
                                                             std::to_string(getpid()));
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);
  std::jthread server([fd = sv[1]] {
    serve_one_handshake(fd, 0);
    ::close(fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  ASSERT_GE(driver.open(), 0);

  ASSERT_TRUE(driver.ptracer_verdict().has_value());
  EXPECT_EQ(*driver.ptracer_verdict(), rocjitsu::PtracerGrantVerdict::Grant);
}

// The socket path is well known, so a same-user process can be listening on it.
// SO_PEERCRED reports that process truthfully; it is simply not the daemon this
// client's launcher started, and Yama's relationship check is exactly what
// would otherwise separate them.
TEST(RemoteDriverPtracerGrantTest, RefusesAListenerThatIsNotTheLaunchedDaemon) {
  const rocjitsu::test::ScopedEnvironmentVariable daemon_pid(rocjitsu::kRpcDaemonPidEnv,
                                                             std::to_string(getpid() + 1));
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);
  std::jthread server([fd = sv[1]] {
    serve_one_handshake(fd, 0);
    ::close(fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  ASSERT_GE(driver.open(), 0);

  ASSERT_TRUE(driver.ptracer_verdict().has_value());
  EXPECT_EQ(*driver.ptracer_verdict(), rocjitsu::PtracerGrantVerdict::RefusedPeerMismatch);
}

// Attach mode: no launcher named a daemon, so there is nothing to compare the
// peer against and the grant is withheld rather than given to whoever answered.
TEST(RemoteDriverPtracerGrantTest, TrustsTheSocketWhenNoLauncherNamedADaemon) {
  // Empty rather than unset: both mean "no launcher named one", and the empty
  // spelling is also what an inherited-but-cleared environment looks like.
  const rocjitsu::test::ScopedEnvironmentVariable daemon_pid(rocjitsu::kRpcDaemonPidEnv, "");
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);
  std::jthread server([fd = sv[1]] {
    serve_one_handshake(fd, 0);
    ::close(fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  ASSERT_GE(driver.open(), 0);

  ASSERT_TRUE(driver.ptracer_verdict().has_value());
  EXPECT_EQ(*driver.ptracer_verdict(), rocjitsu::PtracerGrantVerdict::GrantUnverifiedPeer);
}

// Having named a daemon outranks trusting the socket, so a peer failing the PID
// check is refused rather than falling back to being trusted -- which would
// make the check decorative.
TEST(RemoteDriverPtracerGrantTest, DoesNotFallBackToTrustWhenTheNamedPidMismatches) {
  const uid_t self = geteuid();
  EXPECT_EQ(rocjitsu::ptracer_grant_verdict(4242, 4243, self, self),
            rocjitsu::PtracerGrantVerdict::RefusedPeerMismatch);
}

// A peer that fails the handshake has proved nothing, so the question of
// authorizing it is never reached -- not asked and refused, but not asked.
TEST(RemoteDriverPtracerGrantTest, NeverReachesTheQuestionWhenTheHandshakeFails) {
  const rocjitsu::test::ScopedEnvironmentVariable daemon_pid(rocjitsu::kRpcDaemonPidEnv,
                                                             std::to_string(getpid()));
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);
  std::jthread server([fd = sv[1]] {
    serve_one_handshake(fd, -EPROTO);
    ::close(fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  EXPECT_LT(driver.open(), 0);

  EXPECT_FALSE(driver.ptracer_verdict().has_value())
      << "a peer that never completed the handshake must not be considered for a grant";
}

TEST(RemoteDriverPtracerGrantTest, ReadsYamaScopeAsThePolicyItDenotes) {
  using rocjitsu::PtraceScopePolicy;
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text(std::nullopt), PtraceScopePolicy::Absent);
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text("0"), PtraceScopePolicy::Disabled);
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text("1\n"), PtraceScopePolicy::Relational);
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text(" 2 "), PtraceScopePolicy::AdminOnly);
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text("3"), PtraceScopePolicy::NoAttach);
  // A scope this build does not know, and a value that is not one at all, both
  // resolve to "attempt the grant" rather than to "skip it": skipping is the
  // answer that silently loses the access.
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text("4"), PtraceScopePolicy::Unknown);
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text(""), PtraceScopePolicy::Unknown);
  EXPECT_EQ(rocjitsu::ptrace_scope_policy_from_text("banana"), PtraceScopePolicy::Unknown);
}

TEST(RemoteDriverPtracerGrantTest, RefusesAPeerBelongingToAnotherUser) {
  const uid_t self = geteuid();
  EXPECT_EQ(rocjitsu::ptracer_grant_verdict(4242, 4242, self + 1, self),
            rocjitsu::PtracerGrantVerdict::RefusedForeignUser);
  EXPECT_EQ(rocjitsu::ptracer_grant_verdict(4242, 0, self, self),
            rocjitsu::PtracerGrantVerdict::RefusedNoPeer);
}

// --- Daemon-mode DBG_TRAP notifier-fd transfer via SCM_RIGHTS ---
//
// In daemon mode the debugger's dbg_fd is a number in the *client's* fd table
// and is meaningless to the daemon. The client hands the real fd over
// out-of-band as SCM_RIGHTS ancillary data; the daemon receives it in its own
// fd space and the rj_vm_execute_as() glue substitutes it into DBG_TRAP
// ENABLE's dbg_fd so the debug session can later signal it, releasing it on
// DISABLE. These tests exercise the real rj_vm_execute_as() dispatch path
// (where the substitution and adoption live), not the raw driver ioctl.
class DbgTrapDaemonTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_DAEMON, &vm_), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NE(vm_, nullptr);
    ASSERT_EQ(rj_vm_device_open(vm_, kClientPid, &process_id_), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NE(process_id_, 0u);
  }

  void TearDown() override {
    if (vm_ != nullptr) {
      if (process_id_ != 0)
        rj_vm_device_close(vm_, process_id_);
      rj_vm_destroy(vm_);
    }
  }

  // Runs one ioctl through rj_vm_execute_as() (the daemon dispatch path), with
  // an optional in_handle carried in cmd.in_handle. On return, *in_handle_out
  // (when given) carries cmd.in_handle, which the glue clears to -1 once the
  // debug session has adopted the transferred fd.
  int execute(uint32_t cmd_id, void *buf, size_t buf_size, int in_handle, int *in_handle_out) {
    rj_vm_cmd_t cmd{};
    cmd.cmd = cmd_id;
    cmd.buf = buf;
    cmd.buf_size = buf_size;
    cmd.shared_handle = -1;
    cmd.in_handle = in_handle;
    cmd.in_mem_handle = -1;
    cmd.in_proc_handle = -1;
    if (cmd_id == AMDKFD_IOC_DBG_TRAP && in_handle >= 0) {
      auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(buf);
      if (dbg->op == KFD_IOC_DBG_TRAP_ENABLE) {
        cmd.in_proc_handle = ::open("/proc/self", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (cmd.in_proc_handle < 0)
          return -errno;
        // mem_mismatch_path_ lets a test transfer a writable mem fd belonging to
        // some other process, which is what the identity check exists to catch.
        const char *mem_path =
            mem_mismatch_path_.empty() ? "/proc/self/mem" : mem_mismatch_path_.c_str();
        cmd.in_mem_handle = ::open(mem_path, O_RDWR | O_CLOEXEC);
        if (cmd.in_mem_handle < 0) {
          ::close(cmd.in_proc_handle);
          return -errno;
        }
      }
    }
    rj_vm_execute_as(vm_, process_id_, &cmd);
    if (cmd.in_mem_handle >= 0)
      ::close(cmd.in_mem_handle);
    if (cmd.in_proc_handle >= 0)
      ::close(cmd.in_proc_handle);
    if (in_handle_out != nullptr)
      *in_handle_out = cmd.in_handle;
    return cmd.result;
  }

  int enable_with_notifier(int in_handle, int *in_handle_out) {
    kfd_ioctl_dbg_trap_args en{};
    en.pid = static_cast<uint32_t>(kClientPid);
    en.op = KFD_IOC_DBG_TRAP_ENABLE;
    en.enable.dbg_fd = 0x0BADF00D; // meaningless client-side number; must be replaced
    return execute(AMDKFD_IOC_DBG_TRAP, &en, sizeof(en), in_handle, in_handle_out);
  }

  int disable() {
    kfd_ioctl_dbg_trap_args dis{};
    dis.pid = static_cast<uint32_t>(kClientPid);
    dis.op = KFD_IOC_DBG_TRAP_DISABLE;
    return execute(AMDKFD_IOC_DBG_TRAP, &dis, sizeof(dis), -1, nullptr);
  }

  const rj_client_pid_t kClientPid = getpid();
  rj_vm_t *vm_ = nullptr;
  uint32_t process_id_ = 0;
  std::string mem_mismatch_path_;
};

// The transferred mem fd becomes authoritative for guest reads and CWSR writes,
// so O_RDWR is not enough to accept it -- it also has to name the memory of the
// process whose /proc directory was pinned. A writable mem fd for some other
// live process must be refused, or the daemon would silently redirect both at
// the wrong address space.
TEST_F(DbgTrapDaemonTest, EnableRejectsAMemFdForADifferentProcess) {
  // A live child, so /proc/<pid>/mem exists and is openable for the whole test.
  // ChildProcessGuard rather than a hand-rolled reap: every ASSERT_* below
  // returns from the test, and only RAII kills and waits on all of those paths.
  pid_t other = fork();
  ASSERT_GE(other, 0);
  if (other == 0) {
    for (;;)
      pause();
    _exit(0);
  }
  ChildProcessGuard other_guard(other);

  const std::string other_mem = std::format("/proc/{}/mem", other);
  // Confirm the premise: procfs gives it a different inode from our own.
  struct stat ours {};
  struct stat theirs {};
  const util::UniqueHandle self_mem{::open("/proc/self/mem", O_RDWR | O_CLOEXEC)};
  ASSERT_GE(self_mem.get(), 0);
  const util::UniqueHandle their_mem{::open(other_mem.c_str(), O_RDWR | O_CLOEXEC)};
  if (their_mem.get() < 0) {
    // Hardened kernels can refuse this open outright; the gate is then moot.
    GTEST_SKIP() << "cannot open another process's mem: " << strerror(errno);
  }
  ASSERT_EQ(fstat(self_mem.get(), &ours), 0);
  ASSERT_EQ(fstat(their_mem.get(), &theirs), 0);
  EXPECT_NE(ours.st_ino, theirs.st_ino);

  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);
  util::UniqueHandle notifier_closer{notifier};

  mem_mismatch_path_ = other_mem;
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(kClientPid);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = 0x0BADF00D;
  // -1 rather than a sentinel the assertion would accept: execute() leaves the
  // out-param untouched on its own early returns, and initialising it to
  // anything else lets "never written" pass as "not adopted".
  int in_handle_out = -1;
  const int rejected = execute(AMDKFD_IOC_DBG_TRAP, &en, sizeof(en), notifier, &in_handle_out);
  EXPECT_EQ(rejected, -ESRCH) << "a writable mem fd for another process was accepted";
  EXPECT_NE(in_handle_out, -1) << "the notifier was adopted by a rejected ENABLE";
  // Only a rejected ENABLE leaves the fd ours to close. If the gate regressed
  // and the session took it, dropping ownership here keeps the failure to one
  // assertion instead of adding a close() of a descriptor since reissued.
  if (rejected == 0)
    (void)notifier_closer.release();

  // The matching fd still works, so the gate rejects the mismatch and not the
  // transfer itself.
  mem_mismatch_path_.clear();
  int notifier2 = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier2, 0);
  util::UniqueHandle notifier2_closer{notifier2};
  kfd_ioctl_dbg_trap_args ok{};
  ok.pid = static_cast<uint32_t>(kClientPid);
  ok.op = KFD_IOC_DBG_TRAP_ENABLE;
  ok.enable.dbg_fd = 0x0BADF00D;
  const int accepted = execute(AMDKFD_IOC_DBG_TRAP, &ok, sizeof(ok), notifier2, nullptr);
  EXPECT_EQ(accepted, 0);
  // Adopted by the session on success, so hand it over; still ours to close if
  // the ENABLE failed.
  if (accepted == 0)
    (void)notifier2_closer.release();
  EXPECT_EQ(disable(), 0);
}

// The transferred fd (in_handle) replaces the client-side dbg_fd in the payload
// and the session takes ownership (in_handle cleared so the transport does not
// reclaim it).
TEST_F(DbgTrapDaemonTest, EnableAdoptsTransferredNotifierFd) {
  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(kClientPid);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = 0x0BADF00D; // client-side number the daemon must replace

  int in_handle_out = -2;
  ASSERT_EQ(execute(AMDKFD_IOC_DBG_TRAP, &en, sizeof(en), notifier, &in_handle_out), 0);

  EXPECT_EQ(en.enable.dbg_fd, static_cast<uint32_t>(notifier)); // substituted
  EXPECT_EQ(in_handle_out, -1);                                 // adopted
  EXPECT_NE(fcntl(notifier, F_GETFD), -1);                      // still open (session owns it)

  EXPECT_EQ(disable(), 0); // releases the adopted fd (asserted in its own test)
}

// DISABLE releases the fd the daemon owns; the descriptor is closed afterward.
TEST_F(DbgTrapDaemonTest, DisableClosesAdoptedNotifierFd) {
  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);

  ASSERT_EQ(enable_with_notifier(notifier, nullptr), 0);
  ASSERT_NE(fcntl(notifier, F_GETFD), -1); // open after ENABLE

  ASSERT_EQ(disable(), 0);

  // The daemon owned the transferred fd and closed it on DISABLE.
  EXPECT_EQ(fcntl(notifier, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
}

// Without a transferred fd (in_handle == -1, e.g. the client passed
// KFD_INVALID_FD), nothing is substituted, so the notifier stays invalid and
// daemon-mode ENABLE is rejected with -EBADF (matching the kernel's fget()
// check) rather than adopting a bogus descriptor.
TEST_F(DbgTrapDaemonTest, EnableWithoutTransferredFdReturnsEbadf) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(kClientPid);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = KFD_INVALID_FD;

  int in_handle_out = -2;
  EXPECT_EQ(execute(AMDKFD_IOC_DBG_TRAP, &en, sizeof(en), -1, &in_handle_out), -EBADF);
  EXPECT_EQ(in_handle_out, -1); // nothing adopted
}

// Security: a client can name a small, plausible integer in dbg_fd that happens
// to be a *live descriptor in the daemon* while attaching nothing over
// SCM_RIGHTS. The daemon must never interpret that number in its own fd
// namespace (confused deputy): with no transferred fd the dbg_fd is scrubbed to
// KFD_INVALID_FD, so ENABLE is rejected with -EBADF and the daemon's own
// descriptor is neither adopted nor closed.
TEST_F(DbgTrapDaemonTest, EnableWithClientChosenFdNumberIsNotTrustedInDaemonNamespace) {
  // A real, live fd in *this* (daemon) process. The client names exactly this
  // number in dbg_fd; without the scrub the handler's fcntl() would validate it
  // against the daemon's fd table and adopt it.
  const int daemon_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(daemon_fd, 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(kClientPid);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(daemon_fd); // live in the daemon, not the client

  int in_handle_out = -2;
  EXPECT_EQ(execute(AMDKFD_IOC_DBG_TRAP, &en, sizeof(en), -1, &in_handle_out), -EBADF);
  EXPECT_EQ(in_handle_out, -1); // nothing adopted

  // The daemon's own descriptor was left untouched: not adopted, not closed.
  EXPECT_NE(fcntl(daemon_fd, F_GETFD), -1);
  ::close(daemon_fd);
}

// End-to-end: the RemoteDriver client hands the debugger's notifier fd to an
// in-process daemon over SCM_RIGHTS (mirroring tools/rocjitsu's handle_client),
// and the daemon-side rj_vm_execute_as() adopts it. Proven by having the daemon
// write a sentinel through the *transferred* descriptor and reading it back on
// the client's own eventfd — only possible if SCM_RIGHTS delivered a working
// alias of the same kernel object.
TEST_F(DbgTrapDaemonTest, EnableSendsNotifierFdOverScmRights) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << strerror(errno);

  constexpr uint64_t kSentinel = 0x0102030405060708ULL;
  constexpr uint64_t kMemorySentinel = 0x8877665544332211ULL;
  uint64_t target_memory = 0;
  std::atomic<int> fds_received{0};
  std::atomic<int> in_handle_after{-2};
  std::atomic<int> notifier_cloexec{-1};

  // Minimal stand-in for the daemon's RPC_IOCTL loop: capture an optional
  // SCM_RIGHTS fd on the header (rpc_recv_msg, exactly as tools/rocjitsu does),
  // thread it through cmd.in_handle into the real rj_vm_execute_as() path, and
  // reclaim it only if the session did not adopt it. jthread so an ASSERT_*
  // failure unwinds without std::terminate() on a joinable thread.
  std::jthread server([&, server_fd = sv[1]] {
    for (;;) {
      rocjitsu::RpcHeader hdr{};
      int in_fds[3] = {-1, -1, -1};
      size_t num_in = 3;
      if (rocjitsu::rpc_recv_msg(server_fd, &hdr, sizeof(hdr), in_fds, &num_in) <= 0)
        break;
      int in_fd = (num_in > 0) ? in_fds[0] : -1;
      int in_mem_fd = (num_in > 1) ? in_fds[1] : -1;
      int in_proc_fd = (num_in > 2) ? in_fds[2] : -1;

      if (hdr.opcode == rocjitsu::RPC_CLOSE) {
        rocjitsu::RpcHeader resp{};
        resp.request_id = hdr.request_id;
        rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
        if (in_fd >= 0)
          ::close(in_fd);
        if (in_mem_fd >= 0)
          ::close(in_mem_fd);
        if (in_proc_fd >= 0)
          ::close(in_proc_fd);
        break;
      }
      if (hdr.opcode != rocjitsu::RPC_IOCTL) {
        rocjitsu::RpcHeader resp{};
        resp.request_id = hdr.request_id;
        rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
        if (in_fd >= 0)
          ::close(in_fd);
        if (in_mem_fd >= 0)
          ::close(in_mem_fd);
        if (in_proc_fd >= 0)
          ::close(in_proc_fd);
        continue;
      }

      std::vector<uint8_t> payload(hdr.payload_bytes);
      if (!rocjitsu::rpc_recv_exact(server_fd, payload.data(), hdr.payload_bytes)) {
        if (in_fd >= 0)
          ::close(in_fd);
        if (in_mem_fd >= 0)
          ::close(in_mem_fd);
        if (in_proc_fd >= 0)
          ::close(in_proc_fd);
        break;
      }
      auto *ireq = reinterpret_cast<rocjitsu::RpcIoctlRequest *>(payload.data());

      // Prove the received descriptor is live and aliases the client's eventfd
      // by writing a sentinel through it before the handler adopts it.
      if (in_fd >= 0) {
        fds_received.store(static_cast<int>(num_in));
        // rpc_recv_msg passes MSG_CMSG_CLOEXEC, so the transferred notifier must
        // arrive close-on-exec and cannot leak through a later exec.
        int fd_flags = ::fcntl(in_fd, F_GETFD);
        notifier_cloexec.store((fd_flags >= 0 && (fd_flags & FD_CLOEXEC)) ? 1 : 0);
        uint64_t s = kSentinel;
        [[maybe_unused]] ssize_t w = ::write(in_fd, &s, sizeof(s));
      }
      if (in_mem_fd >= 0) {
        [[maybe_unused]] ssize_t w =
            ::pwrite(in_mem_fd, &kMemorySentinel, sizeof(kMemorySentinel),
                     static_cast<off_t>(reinterpret_cast<uintptr_t>(&target_memory)));
      }

      rj_vm_cmd_t cmd{};
      cmd.cmd = ireq->ioctl_cmd;
      cmd.buf = payload.data() + sizeof(rocjitsu::RpcIoctlRequest);
      cmd.buf_size = ireq->args_bytes;
      cmd.shared_handle = -1;
      cmd.in_handle = in_fd;
      cmd.in_mem_handle = in_mem_fd;
      cmd.in_proc_handle = in_proc_fd;
      rj_vm_execute_as(vm_, process_id_, &cmd);
      in_handle_after.store(cmd.in_handle);
      if (cmd.in_handle >= 0)
        ::close(cmd.in_handle);
      if (cmd.in_mem_handle >= 0)
        ::close(cmd.in_mem_handle);
      if (cmd.in_proc_handle >= 0)
        ::close(cmd.in_proc_handle);

      rocjitsu::RpcHeader resp{};
      resp.opcode = rocjitsu::RPC_IOCTL;
      resp.request_id = hdr.request_id;
      resp.result = cmd.result;
      resp.payload_bytes = static_cast<uint32_t>(cmd.buf_size);
      if (!rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp)))
        break;
      if (cmd.buf_size > 0 && !rocjitsu::rpc_send_exact(server_fd, cmd.buf, cmd.buf_size))
        break;
    }
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  // Non-blocking so a failed transfer fails the read below instead of hanging.
  int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(kClientPid);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(notifier);
  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // The client sent the notifier, target memory and pinned proc directory; the
  // session adopted the first two and the transport reclaimed the identity fd.
  EXPECT_EQ(fds_received.load(), 3);
  EXPECT_EQ(in_handle_after.load(), -1);
  // The transferred notifier was received close-on-exec (MSG_CMSG_CLOEXEC), so
  // it cannot leak through a later exec in the daemon.
  EXPECT_EQ(notifier_cloexec.load(), 1);

  // The daemon's write through the transferred fd is visible on our eventfd,
  // proving the descriptor was really carried across the process boundary.
  uint64_t got = 0;
  ASSERT_EQ(::read(notifier, &got, sizeof(got)), static_cast<ssize_t>(sizeof(got)))
      << "notifier fd was not transferred: " << strerror(errno);
  EXPECT_EQ(got, kSentinel);
  EXPECT_EQ(target_memory, kMemorySentinel);

  // Release the adopted fd through the transport for symmetry.
  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(kClientPid);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &dis), 0);

  rd.close(); // sends RPC_CLOSE so the server loop exits
  server.join();
  ::close(notifier);
}

// A debug session belongs to the debugger that enabled it: a *different* client
// may not drive it (kernel: EPERM). Only the resolved target itself (self-debug)
// or the registered debugger passes the permission gate.
TEST_F(DbgTrapDaemonTest, ForeignClientCannotDriveAnothersSession) {
  // Client A (kClientPid) self-enables debug, becoming its own debugger.
  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);
  ASSERT_EQ(enable_with_notifier(notifier, nullptr), 0);

  // A second, unrelated client B.
  constexpr rj_client_pid_t kOtherPid = 5555;
  uint32_t other_pid = 0;
  ASSERT_EQ(rj_vm_device_open(vm_, kOtherPid, &other_pid), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(other_pid, 0u);

  // B targets A's session with a non-DISABLE op: rejected with -EPERM.
  kfd_ioctl_dbg_trap_args op{};
  op.pid = static_cast<uint32_t>(kClientPid); // target = A
  op.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  rj_vm_cmd_t cmd{};
  cmd.cmd = AMDKFD_IOC_DBG_TRAP;
  cmd.buf = &op;
  cmd.buf_size = sizeof(op);
  cmd.shared_handle = -1;
  rj_vm_execute_as(vm_, other_pid, &cmd); // caller = B
  EXPECT_EQ(cmd.result, -EPERM);

  rj_vm_device_close(vm_, other_pid);
  EXPECT_EQ(disable(), 0); // A tears down its session (closes the notifier)
}

// GET_DEVICE_SNAPSHOT through the production daemon dispatch, over the same
// cases the local path pins (count probe, zero stride, null buffer, an entry at
// the struct stride, and more entries than devices).
//
// The RemoteDriverDbgSnapshotTest suite answers with serve_one_ioctl_reply(), a
// stand-in that never runs reconstruct_embedded_pointers(): it echoes the arg
// struct and whatever tail the test tells it to. That is the right tool for
// pinning what the *client* does with a reply, but it can agree with a daemon
// that does not exist. rj_vm_execute_as() is the real thing -- it repoints
// snapshot_buf_ptr at the inline tail and hands the request to SimulatedKfd --
// so running the local contract through it is what makes the two transports
// comparable rather than separately self-consistent.
TEST_F(DbgTrapDaemonTest, DeviceSnapshotDispatchMatchesTheLocalContract) {
  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);
  ASSERT_EQ(enable_with_notifier(notifier, nullptr), 0);

  constexpr uint32_t kEntryBytes = sizeof(kfd_dbg_device_info_entry);
  constexpr uint32_t kDeviceTotal = 1; // the fixture config declares one GPU
  constexpr uint8_t kSentinel = 0xAB;

  // Lay the request out the way the transport does: the arg struct followed by
  // the inline tail that stands in for the caller's snapshot buffer. A tail is
  // what triggers the daemon's pointer reconstruction, so `tail_bytes == 0`
  // models the requests the client sends without one (count probe, zero
  // stride) -- there snapshot_buf_ptr survives as the client gave it, which is
  // why those cases pass a real buffer and assert it stays untouched.
  struct SnapshotResult {
    int rc;
    kfd_ioctl_dbg_trap_args args;
    std::vector<uint8_t> tail;
  };
  auto run_snapshot = [&](uint32_t num_devices, uint32_t entry_size, size_t tail_bytes,
                          uint64_t buf_ptr) {
    std::vector<uint8_t> payload(sizeof(kfd_ioctl_dbg_trap_args) + tail_bytes, kSentinel);
    kfd_ioctl_dbg_trap_args snap{};
    snap.pid = static_cast<uint32_t>(kClientPid);
    snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
    snap.device_snapshot.num_devices = num_devices;
    snap.device_snapshot.entry_size = entry_size;
    snap.device_snapshot.snapshot_buf_ptr = buf_ptr;
    std::memcpy(payload.data(), &snap, sizeof(snap));

    SnapshotResult result{};
    result.rc = execute(AMDKFD_IOC_DBG_TRAP, payload.data(), payload.size(), -1, nullptr);
    std::memcpy(&result.args, payload.data(), sizeof(result.args));
    result.tail.assign(payload.begin() + sizeof(kfd_ioctl_dbg_trap_args), payload.end());
    return result;
  };
  auto all_sentinel = [](const std::vector<uint8_t> &bytes, size_t from, size_t to) {
    return std::all_of(bytes.begin() + from, bytes.begin() + to,
                       [](uint8_t byte) { return byte == kSentinel; });
  };

  // A null buffer is rejected before any output is written, exactly as
  // DbgTrapDeviceSnapshotRejectsNullBufferWithoutWritingOutputs pins locally.
  // No tail, so the reconstruction leaves the null in place for the driver to
  // find -- this is the case the RemoteDriver rejects client-side, because on
  // the wire a null pointer would otherwise arrive as a reconstructed tail.
  {
    constexpr uint32_t kRequested = 2;
    constexpr uint32_t kStride = kEntryBytes + 16;
    auto null_buf = run_snapshot(kRequested, kStride, 0, 0);
    EXPECT_EQ(null_buf.rc, -EINVAL);
    EXPECT_EQ(null_buf.args.device_snapshot.num_devices, kRequested);
    EXPECT_EQ(null_buf.args.device_snapshot.entry_size, kStride);
  }

  // Count-only probe: no capacity, so the total is reported and nothing is
  // written.
  {
    std::array<uint8_t, kEntryBytes> caller_buf{};
    caller_buf.fill(kSentinel);
    auto probe = run_snapshot(0, kEntryBytes, 0, reinterpret_cast<uint64_t>(caller_buf.data()));
    EXPECT_EQ(probe.rc, 0);
    EXPECT_EQ(probe.args.device_snapshot.num_devices, kDeviceTotal);
    EXPECT_EQ(probe.args.device_snapshot.entry_size, kEntryBytes);
    EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(),
                            [](uint8_t byte) { return byte == kSentinel; }));
  }

  // Zero stride: success, the total reported, entry_size(OUT) zero, no bytes
  // written -- DbgTrapDeviceSnapshotZeroStrideReportsCountAndWritesNothing.
  {
    std::array<uint8_t, kEntryBytes> caller_buf{};
    caller_buf.fill(kSentinel);
    auto zero_stride =
        run_snapshot(kDeviceTotal, 0, 0, reinterpret_cast<uint64_t>(caller_buf.data()));
    EXPECT_EQ(zero_stride.rc, 0);
    EXPECT_EQ(zero_stride.args.device_snapshot.num_devices, kDeviceTotal);
    EXPECT_EQ(zero_stride.args.device_snapshot.entry_size, 0u);
    EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(),
                            [](uint8_t byte) { return byte == kSentinel; }));
  }

  // One entry at the compact stride the client transmits: the agent lands in
  // the tail, which is the buffer the client scatters into the caller's slots.
  {
    auto filled = run_snapshot(kDeviceTotal, kEntryBytes, kEntryBytes, 0xDEADBEEF);
    ASSERT_EQ(filled.rc, 0);
    EXPECT_EQ(filled.args.device_snapshot.num_devices, kDeviceTotal);
    EXPECT_EQ(filled.args.device_snapshot.entry_size, kEntryBytes);
    ASSERT_EQ(filled.tail.size(), kEntryBytes);
    kfd_dbg_device_info_entry entry{};
    std::memcpy(&entry, filled.tail.data(), sizeof(entry));
    EXPECT_EQ(entry.gpu_id, kGpuId);
    // The same fields rocdbgapi's agent_snapshot fatal-errors on if zero.
    EXPECT_NE(entry.simd_count, 0u);
    EXPECT_NE(entry.max_waves_per_simd, 0u);
    EXPECT_NE(entry.array_count, 0u);
    EXPECT_NE(entry.num_xcc, 0u);
  }

  // Capacity beyond the device total: the driver fills what exists and leaves
  // the rest of the buffer as the caller left it.
  {
    auto oversized = run_snapshot(kDeviceTotal + 1, kEntryBytes, 2 * kEntryBytes, 0xDEADBEEF);
    ASSERT_EQ(oversized.rc, 0);
    EXPECT_EQ(oversized.args.device_snapshot.num_devices, kDeviceTotal);
    ASSERT_EQ(oversized.tail.size(), 2 * kEntryBytes);
    kfd_dbg_device_info_entry entry{};
    std::memcpy(&entry, oversized.tail.data(), sizeof(entry));
    EXPECT_EQ(entry.gpu_id, kGpuId);
    EXPECT_TRUE(all_sentinel(oversized.tail, kEntryBytes, 2 * kEntryBytes))
        << "the daemon wrote past the devices it enumerated";
  }

  EXPECT_EQ(disable(), 0);
}

// GET_QUEUE_SNAPSHOT through the same production daemon dispatch, over the
// cases the local path pins: a live queue's metadata reconstructed into the
// inline tail, a null buffer with queues to report, and a request that declares
// entries but carries no tail for them.
TEST_F(DbgTrapDaemonTest, QueueSnapshotReconstructsInlineBufferAndValidatesErrors) {
  kfd_ioctl_create_queue_args queue{};
  queue.gpu_id = kGpuId;
  queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  queue.ring_base_address = 0x100000;
  queue.ring_size = 4096;
  queue.read_pointer_address = 0x200000;
  queue.write_pointer_address = 0x200040;
  queue.ctx_save_restore_address = 0x300000;
  queue.ctx_save_restore_size = 0x8000;
  ASSERT_EQ(execute(AMDKFD_IOC_CREATE_QUEUE, &queue, sizeof(queue), -1, nullptr), 0);

  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);
  ASSERT_EQ(enable_with_notifier(notifier, nullptr), 0);

  std::array<uint8_t, sizeof(kfd_ioctl_dbg_trap_args) + sizeof(kfd_queue_snapshot_entry)> payload{};
  auto *snap = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(payload.data());
  snap->pid = static_cast<uint32_t>(kClientPid);
  snap->op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap->queue_snapshot.num_queues = 1;
  snap->queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  snap->queue_snapshot.snapshot_buf_ptr = 0xDEADBEEF; // daemon must replace this client pointer
  ASSERT_EQ(execute(AMDKFD_IOC_DBG_TRAP, payload.data(), payload.size(), -1, nullptr), 0);

  const auto *entry = reinterpret_cast<const kfd_queue_snapshot_entry *>(
      payload.data() + sizeof(kfd_ioctl_dbg_trap_args));
  EXPECT_EQ(snap->queue_snapshot.num_queues, 1u);
  EXPECT_EQ(entry->queue_id, queue.queue_id);
  EXPECT_EQ(entry->gpu_id, kGpuId);
  EXPECT_EQ(entry->ctx_save_restore_address, queue.ctx_save_restore_address);
  EXPECT_EQ(entry->exception_status, KFD_EC_MASK(EC_QUEUE_NEW));

  // A null buffer survives reconstruction (rj_vm.cpp only repoints a non-null
  // one), so the driver sees the null the caller sent. With a queue to report
  // and nowhere to put it that is -EFAULT, and nothing is written.
  constexpr uint8_t kSentinel = 0xAB;
  std::array<uint8_t, sizeof(kfd_ioctl_dbg_trap_args) + sizeof(kfd_queue_snapshot_entry)>
      null_queue_payload;
  null_queue_payload.fill(kSentinel);
  auto *null_queue = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(null_queue_payload.data());
  *null_queue = {};
  null_queue->pid = static_cast<uint32_t>(kClientPid);
  null_queue->op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  null_queue->queue_snapshot.num_queues = 1;
  null_queue->queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(execute(AMDKFD_IOC_DBG_TRAP, null_queue_payload.data(), null_queue_payload.size(), -1,
                    nullptr),
            -EFAULT);
  EXPECT_EQ(null_queue->queue_snapshot.snapshot_buf_ptr, 0u);
  EXPECT_TRUE(std::all_of(null_queue_payload.begin() + sizeof(kfd_ioctl_dbg_trap_args),
                          null_queue_payload.end(),
                          [](uint8_t byte) { return byte == kSentinel; }));

  // The device op reaches the same null the same way, but answers it with
  // -EINVAL: kfd_dbg_trap_device_snapshot() validates the buffer before writing
  // any output, which is what DeviceSnapshotDispatchMatchesTheLocalContract and
  // DbgTrapDeviceSnapshotRejectsNullBufferWithoutWritingOutputs pin. The two ops
  // differ here because the drivers they mirror do.
  std::array<uint8_t, sizeof(kfd_ioctl_dbg_trap_args) + sizeof(kfd_dbg_device_info_entry)>
      null_device_payload;
  null_device_payload.fill(kSentinel);
  auto *null_device = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(null_device_payload.data());
  *null_device = {};
  null_device->pid = static_cast<uint32_t>(kClientPid);
  null_device->op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  null_device->device_snapshot.num_devices = 1;
  null_device->device_snapshot.entry_size = sizeof(kfd_dbg_device_info_entry);
  ASSERT_EQ(execute(AMDKFD_IOC_DBG_TRAP, null_device_payload.data(), null_device_payload.size(), -1,
                    nullptr),
            -EINVAL);
  EXPECT_EQ(null_device->device_snapshot.snapshot_buf_ptr, 0u);
  EXPECT_EQ(null_device->device_snapshot.num_devices, 1u)
      << "a rejected request must not report a device total";
  EXPECT_EQ(null_device->device_snapshot.entry_size, sizeof(kfd_dbg_device_info_entry));
  EXPECT_TRUE(std::all_of(null_device_payload.begin() + sizeof(kfd_ioctl_dbg_trap_args),
                          null_device_payload.end(),
                          [](uint8_t byte) { return byte == kSentinel; }));

  // Entries declared with no tail to hold them: the payload is one arg struct
  // short of what num_queues * entry_size demands, so the request is malformed.
  kfd_ioctl_dbg_trap_args missing_inline{};
  missing_inline.pid = static_cast<uint32_t>(kClientPid);
  missing_inline.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  missing_inline.queue_snapshot.num_queues = 1;
  missing_inline.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  EXPECT_EQ(execute(AMDKFD_IOC_DBG_TRAP, &missing_inline, sizeof(missing_inline), -1, nullptr),
            -EINVAL);

  kfd_ioctl_destroy_queue_args destroy{};
  destroy.queue_id = queue.queue_id;
  EXPECT_EQ(execute(AMDKFD_IOC_DESTROY_QUEUE, &destroy, sizeof(destroy), -1, nullptr), 0);
  EXPECT_EQ(disable(), 0);
}

TEST_F(DbgTrapDaemonTest, QueueControlReconstructsRequestedIdsAndReportsInvalidQueues) {
  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(execute(AMDKFD_IOC_RUNTIME_ENABLE, &runtime, sizeof(runtime), -1, nullptr), 0);

  kfd_ioctl_create_queue_args queue{};
  queue.gpu_id = kGpuId;
  queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  queue.ring_base_address = 0x100000;
  queue.ring_size = 4096;
  queue.read_pointer_address = 0x200000;
  queue.write_pointer_address = 0x200040;
  queue.ctx_save_restore_address = 0x300000;
  queue.ctx_save_restore_size = 0x8000;
  ASSERT_EQ(execute(AMDKFD_IOC_CREATE_QUEUE, &queue, sizeof(queue), -1, nullptr), 0);

  int notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier, 0);
  ASSERT_EQ(enable_with_notifier(notifier, nullptr), 0);

  // Clear EC_QUEUE_NEW; real KFD rejects suspension until the debugger has
  // observed and cleared that queue lifecycle event.
  std::array<uint8_t, sizeof(kfd_ioctl_dbg_trap_args) + sizeof(kfd_queue_snapshot_entry)>
      snapshot_payload{};
  auto *snapshot = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(snapshot_payload.data());
  snapshot->pid = static_cast<uint32_t>(kClientPid);
  snapshot->op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot->queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot->queue_snapshot.num_queues = 1;
  snapshot->queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  snapshot->queue_snapshot.snapshot_buf_ptr = 1;
  ASSERT_EQ(
      execute(AMDKFD_IOC_DBG_TRAP, snapshot_payload.data(), snapshot_payload.size(), -1, nullptr),
      0);

  constexpr uint32_t kMissingQueue = 0x1234;
  constexpr uint32_t kInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  std::array<uint8_t, sizeof(kfd_ioctl_dbg_trap_args) + 2 * sizeof(uint32_t)> suspend_payload{};
  auto *suspend = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(suspend_payload.data());
  suspend->pid = static_cast<uint32_t>(kClientPid);
  suspend->op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  suspend->suspend_queues.queue_array_ptr = 1;
  suspend->suspend_queues.num_queues = 2;
  auto *suspend_ids = reinterpret_cast<uint32_t *>(suspend_payload.data() + sizeof(*suspend));
  suspend_ids[0] = queue.queue_id;
  suspend_ids[1] = kMissingQueue;
  EXPECT_EQ(
      execute(AMDKFD_IOC_DBG_TRAP, suspend_payload.data(), suspend_payload.size(), -1, nullptr), 1);
  EXPECT_EQ(suspend_ids[0], queue.queue_id);
  EXPECT_EQ(suspend_ids[1], kMissingQueue | kInvalid);

  std::array<uint8_t, sizeof(kfd_ioctl_dbg_trap_args) + sizeof(uint32_t)> resume_payload{};
  auto *resume = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(resume_payload.data());
  resume->pid = static_cast<uint32_t>(kClientPid);
  resume->op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  resume->resume_queues.queue_array_ptr = 1;
  resume->resume_queues.num_queues = 1;
  auto *resume_id = reinterpret_cast<uint32_t *>(resume_payload.data() + sizeof(*resume));
  *resume_id = queue.queue_id;
  EXPECT_EQ(execute(AMDKFD_IOC_DBG_TRAP, resume_payload.data(), resume_payload.size(), -1, nullptr),
            1);
  EXPECT_EQ(*resume_id, queue.queue_id);

  EXPECT_EQ(disable(), 0);
}

// --- RemoteDriver DBG_TRAP snapshot response copy-back ---
//
// The client saves the caller's snapshot buffer pointer and capacity
// (num_devices * entry_size) before serialization, then on the response only
// writes it back on success, clamped to that capacity. This guards daemon mode
// against a failed op (e.g. -ENOSYS) mutating caller memory and against a
// daemon-returned count larger than the caller's buffer.

// Closes an fd however the enclosing scope exits. A daemon stand-in that
// returns early without closing leaves the client blocked in rpc_recv_msg()
// waiting for an EOF that never arrives, which hangs the run instead of
// failing the test. util::UniqueHandle already owns a POSIX fd exactly this
// way, so there is nothing to hand-roll.
using CloseOnScopeExit = util::UniqueHandle;

// Inspect the request header and rewrite the arg struct the reply echoes back,
// so one stand-in can model a daemon that reports different outputs than the
// caller requested.
using IoctlReplyPatch =
    std::function<void(const rocjitsu::RpcHeader &request, kfd_ioctl_dbg_trap_args &echoed)>;

// Lay out the reply's inline tail by hand instead of flooding it with `poison`,
// for tests that must tell one returned entry from another.
using IoctlReplyTailFill = std::function<void(uint8_t *tail, size_t bytes)>;

// One-shot daemon stand-in: read a single RPC_IOCTL, then reply with `result`
// and a response whose inline tail (after the echoed arg struct) is
// `extra_bytes` of `poison`. Does not close `server_fd` (caller owns it).
void serve_one_ioctl_reply(int server_fd, int32_t result, size_t arg_struct_size,
                           size_t extra_bytes, uint8_t poison, const IoctlReplyPatch &patch = {},
                           const IoctlReplyTailFill &fill_tail = {}) {
  rocjitsu::RpcHeader hdr{};
  int in_fds[1] = {-1};
  size_t num_in = 1;
  if (rocjitsu::rpc_recv_msg(server_fd, &hdr, sizeof(hdr), in_fds, &num_in) <= 0)
    return;
  if (in_fds[0] >= 0)
    ::close(in_fds[0]);
  std::vector<uint8_t> req(hdr.payload_bytes);
  if (!rocjitsu::rpc_recv_exact(server_fd, req.data(), hdr.payload_bytes))
    return;

  // Response payload = echoed arg struct + poison tail. The client copies the
  // first arg_struct_size bytes back into its arg and treats the remainder as
  // inline snapshot data to write into the caller's snapshot buffer.
  std::vector<uint8_t> out(arg_struct_size + extra_bytes);
  std::memcpy(out.data(), req.data() + sizeof(rocjitsu::RpcIoctlRequest), arg_struct_size);
  if (fill_tail)
    fill_tail(out.data() + arg_struct_size, extra_bytes);
  else
    std::memset(out.data() + arg_struct_size, poison, extra_bytes);

  if (patch && arg_struct_size >= sizeof(kfd_ioctl_dbg_trap_args)) {
    kfd_ioctl_dbg_trap_args echoed{};
    std::memcpy(&echoed, out.data(), sizeof(echoed));
    patch(hdr, echoed);
    std::memcpy(out.data(), &echoed, sizeof(echoed));
  }

  rocjitsu::RpcHeader resp{};
  resp.opcode = rocjitsu::RPC_IOCTL;
  resp.request_id = hdr.request_id;
  resp.result = result;
  resp.payload_bytes = static_cast<uint32_t>(out.size());
  rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
  rocjitsu::rpc_send_exact(server_fd, out.data(), out.size());
}

TEST(RemoteDriverDbgQueueControlTest, QueueIdsAndStatusBitsRoundTripInline) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  std::array<uint32_t, 2> observed{};
  std::jthread server([&, server_fd = sv[1]] {
    rocjitsu::RpcHeader header{};
    int received_fds[1] = {-1};
    size_t num_fds = 1;
    ASSERT_GT(rocjitsu::rpc_recv_msg(server_fd, &header, sizeof(header), received_fds, &num_fds),
              0);
    std::vector<uint8_t> request(header.payload_bytes);
    ASSERT_TRUE(rocjitsu::rpc_recv_exact(server_fd, request.data(), request.size()));
    ASSERT_GE(request.size(), sizeof(rocjitsu::RpcIoctlRequest) + sizeof(kfd_ioctl_dbg_trap_args) +
                                  sizeof(observed));
    auto *args = request.data() + sizeof(rocjitsu::RpcIoctlRequest);
    std::memcpy(observed.data(), args + sizeof(kfd_ioctl_dbg_trap_args), sizeof(observed));

    std::vector<uint8_t> response(sizeof(kfd_ioctl_dbg_trap_args) + sizeof(observed));
    std::memcpy(response.data(), args, response.size());
    auto *returned_ids =
        reinterpret_cast<uint32_t *>(response.data() + sizeof(kfd_ioctl_dbg_trap_args));
    returned_ids[1] |= kInvalid;
    rocjitsu::RpcHeader response_header{};
    response_header.opcode = rocjitsu::RPC_IOCTL;
    response_header.request_id = header.request_id;
    response_header.payload_bytes = static_cast<uint32_t>(response.size());
    response_header.result = 1;
    ASSERT_TRUE(rocjitsu::rpc_send_exact(server_fd, &response_header, sizeof(response_header)));
    ASSERT_TRUE(rocjitsu::rpc_send_exact(server_fd, response.data(), response.size()));
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver driver(sv[0]);
  std::array<uint32_t, 2> queue_ids{7, 9};
  kfd_ioctl_dbg_trap_args suspend{};
  suspend.pid = 4242;
  suspend.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  suspend.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(queue_ids.data());
  suspend.suspend_queues.num_queues = queue_ids.size();
  const uint64_t original_pointer = suspend.suspend_queues.queue_array_ptr;

  EXPECT_EQ(driver.ioctl(AMDKFD_IOC_DBG_TRAP, &suspend), 1);
  EXPECT_EQ(observed, (std::array<uint32_t, 2>{7, 9}));
  EXPECT_EQ(queue_ids, (std::array<uint32_t, 2>{7, 9 | kInvalid}));
  EXPECT_EQ(suspend.suspend_queues.queue_array_ptr, original_pointer);
  server.join();
}

// A GET_DEVICE_SNAPSHOT that the daemon fails (result != 0) must not copy the
// response tail into the caller's snapshot buffer, even though the daemon
// returned inline bytes.
TEST(RemoteDriverDbgSnapshotTest, FailedSnapshotLeavesCallerBufferUntouched) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kNumDevices = 4;
  constexpr uint32_t kEntrySize = 16;
  constexpr size_t kCap = static_cast<size_t>(kNumDevices) * kEntrySize;
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kCap, kSentinel);

  std::jthread server([&, server_fd = sv[1]] {
    serve_one_ioctl_reply(server_fd, -ENOSYS, arg_struct_size, kCap, kPoison);
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = kNumDevices;
  snap.device_snapshot.entry_size = kEntrySize;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -ENOSYS);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "failed GET_DEVICE_SNAPSHOT mutated caller memory";

  server.join();
}

// On success the copy is clamped to the caller's original capacity
// (num_devices * entry_size); a daemon returning a larger tail cannot overrun
// the caller's buffer.
TEST(RemoteDriverDbgSnapshotTest, SuccessfulSnapshotClampsCopyToCallerCapacity) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kNumDevices = 4;
  constexpr uint32_t kEntrySize = 16;
  constexpr size_t kCap = static_cast<size_t>(kNumDevices) * kEntrySize;
  constexpr size_t kGuard = 32; // tail beyond the declared capacity, must be untouched
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kCap + kGuard, kSentinel);

  std::jthread server([&, server_fd = sv[1]] {
    // Return MORE inline bytes than the caller's capacity to exercise the clamp.
    serve_one_ioctl_reply(server_fd, 0, arg_struct_size, kCap + kGuard, kPoison);
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = kNumDevices;
  snap.device_snapshot.entry_size = kEntrySize;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.begin() + kCap, [](uint8_t b) {
    return b == kPoison;
  })) << "successful snapshot did not copy the daemon payload";
  EXPECT_TRUE(std::all_of(caller_buf.begin() + kCap, caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "snapshot copy overran the caller's declared capacity";

  server.join();
}

// Daemon mode must reach the same verdict as the local path when the caller
// supplies no snapshot buffer, and only the client can deliver it: the daemon
// rewrites snapshot_buf_ptr to its own inline tail before replaying the ioctl
// (rj_vm.cpp, reconstruct_embedded_pointers), so SimulatedKfd never sees the
// null and answers with success plus a device total -- the opposite of the
// -EINVAL DbgTrapDeviceSnapshotRejectsNullBufferWithoutWritingOutputs pins for
// the local path. So the request must never reach the wire at all, and the
// caller's num_devices/entry_size must come back exactly as they went in.
TEST(RemoteDriverDbgSnapshotTest, NullSnapshotBufferIsNeverWrittenThrough) {
  constexpr uint32_t kNumDevices = 2;
  constexpr uint32_t kEntryBytes = sizeof(kfd_dbg_device_info_entry);

  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = kNumDevices;
  snap.device_snapshot.entry_size = kEntryBytes;
  snap.device_snapshot.snapshot_buf_ptr = 0;

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -EINVAL);
  EXPECT_EQ(snap.device_snapshot.snapshot_buf_ptr, 0u);
  EXPECT_EQ(snap.device_snapshot.num_devices, kNumDevices)
      << "a rejected request must not report a device total";
  EXPECT_EQ(snap.device_snapshot.entry_size, kEntryBytes);

  // Nothing was transmitted, so the daemon side of the pair is still empty.
  uint8_t byte = 0;
  EXPECT_EQ(::recv(sv[1], &byte, 1, MSG_DONTWAIT), -1)
      << "a request with no output buffer reached the wire";
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK) << ::strerror(errno);
}

// A zero entry_size reserves no inline tail whatever the device count is, so
// the transmitted num_devices must survive the request clamp untouched.
// num_devices(IN) is what the driver clamps its fill count against, so
// rewriting it to zero would hand the daemon a different request than the
// caller made -- the count-only probe rather than a fill of zero-byte entries.
//
// The stand-in answers the way the driver does, which
// DbgTrapDeviceSnapshotZeroStrideReportsCountAndWritesNothing pins locally:
// success, the device total reported, entry_size(OUT) zero, nothing written.
TEST(RemoteDriverDbgSnapshotTest, ZeroStrideSnapshotKeepsTheRequestedDeviceCount) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kRequested = 1;
  constexpr uint32_t kDeviceTotal = 1;
  constexpr uint8_t kSentinel = 0xAB;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::array<uint8_t, sizeof(kfd_dbg_device_info_entry)> caller_buf{};
  caller_buf.fill(kSentinel);
  std::atomic<uint32_t> observed_num_devices{0};

  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    serve_one_ioctl_reply(server_fd, 0, arg_struct_size, 0, 0,
                          [&](const rocjitsu::RpcHeader &, kfd_ioctl_dbg_trap_args &echoed) {
                            observed_num_devices = echoed.device_snapshot.num_devices;
                            echoed.device_snapshot.num_devices = kDeviceTotal;
                            echoed.device_snapshot.entry_size = 0;
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = kRequested;
  snap.device_snapshot.entry_size = 0;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  server.join();

  EXPECT_EQ(observed_num_devices.load(), kRequested)
      << "a zero stride was clamped into the count-only probe";
  EXPECT_EQ(snap.device_snapshot.num_devices, kDeviceTotal);
  EXPECT_EQ(snap.device_snapshot.entry_size, 0u);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "a zero-stride snapshot wrote entry bytes";
}

// Reply patch modelling a daemon that enumerates fewer GPUs than the caller
// asked for: report the true device total and the driver's clamped entry size.
IoctlReplyPatch snapshot_outputs(uint32_t out_num_devices, uint32_t out_entry_size) {
  return [out_num_devices, out_entry_size](const rocjitsu::RpcHeader &,
                                           kfd_ioctl_dbg_trap_args &echoed) {
    echoed.device_snapshot.num_devices = out_num_devices;
    echoed.device_snapshot.entry_size = out_entry_size;
  };
}

// entry_size(IN) is the caller's buffer stride, and the uapi lets it exceed the
// current struct (kfd_ioctl.h), so a caller may legally declare a stride wider
// than the whole RPC payload budget. Transmitting it verbatim left room for no
// entries at all, and the count-only request that produced came back as success
// over an untouched buffer -- where local mode fills the entry. The wire stride
// is the struct instead, and the packed reply is scattered into the caller's
// own slots.
TEST(RemoteDriverDbgSnapshotTest, StrideWiderThanThePayloadLimitStillFillsEntries) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kEntryBytes = sizeof(kfd_dbg_device_info_entry);
  // A stride the request could never carry: wider than the payload ceiling.
  constexpr size_t kCallerStride = static_cast<size_t>(rocjitsu::kMaxPayloadBytes) + 4096;
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kCallerStride, kSentinel);
  std::atomic<uint32_t> observed_stride{0};
  std::atomic<uint32_t> observed_devices{0};
  std::atomic<uint32_t> observed_payload{0};

  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    serve_one_ioctl_reply(server_fd, 0, arg_struct_size, kEntryBytes, kPoison,
                          [&](const rocjitsu::RpcHeader &hdr, kfd_ioctl_dbg_trap_args &echoed) {
                            observed_payload = hdr.payload_bytes;
                            observed_stride = echoed.device_snapshot.entry_size;
                            observed_devices = echoed.device_snapshot.num_devices;
                            echoed.device_snapshot.num_devices = 1;
                            echoed.device_snapshot.entry_size = kEntryBytes;
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = 1;
  snap.device_snapshot.entry_size = static_cast<uint32_t>(kCallerStride);
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  server.join();

  EXPECT_EQ(observed_stride.load(), kEntryBytes) << "caller stride was transmitted verbatim";
  EXPECT_EQ(observed_devices.load(), 1u) << "the fill request degraded to a count-only probe";
  EXPECT_LE(observed_payload.load(), rocjitsu::kMaxPayloadBytes);

  EXPECT_EQ(snap.device_snapshot.num_devices, 1u);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.begin() + kEntryBytes, [](uint8_t b) {
    return b == kPoison;
  })) << "the entry did not reach the caller's buffer";
  EXPECT_TRUE(std::all_of(caller_buf.begin() + kEntryBytes, caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "copy-back wrote past the one entry the daemon returned";
}

// libhsakmt's hsaKmtDbgGetDeviceDataCtx() asks for UINT32_MAX devices and lets
// the driver report the real total. Serialized literally that reserves
// UINT32_MAX * sizeof(kfd_dbg_device_info_entry) (~480 GiB) and kills the
// process on std::bad_alloc. The request must be clamped to what the transport
// can carry, and the reply must still land in the caller's buffer.
// Snapshot requests are clamped to the payload ceiling, but QUERY_EXCEPTION_INFO
// takes info_size straight from the caller and SUSPEND/RESUME derive their tail
// from num_queues. Those three were bounded only by UINT32_MAX, so a malformed
// ioctl could drive a multi-gigabyte resize() and a copy out of the caller's
// buffer before the frame was ever sent -- the daemon's 16 MiB limit is checked
// on receipt, far too late to matter. They must be refused, not clamped: the
// daemon recomputes the expected tail from the echoed args, so a short tail with
// an unclamped count is a protocol mismatch that drops the connection.
TEST(RemoteDriverDbgTrapTest, OversizedInlineTailIsRefusedBeforeAllocating) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};
  rocjitsu::RemoteDriver rd(sv[0]);

  // Nothing is served: a request this size must be rejected locally, so the
  // call cannot block waiting for a reply.
  kfd_ioctl_dbg_trap_args info{};
  info.pid = 4242;
  info.op = KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO;
  info.query_exception_info.info_size = 3u * 1024u * 1024u * 1024u; // 3 GiB
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &info), -E2BIG);

  kfd_ioctl_dbg_trap_args suspend{};
  suspend.pid = 4242;
  suspend.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  suspend.suspend_queues.num_queues = 0x4000'0000u; // * 4 bytes = 4 GiB
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &suspend), -E2BIG);

  kfd_ioctl_dbg_trap_args resume{};
  resume.pid = 4242;
  resume.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  resume.resume_queues.num_queues = 0x4000'0000u;
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &resume), -E2BIG);
}

TEST(RemoteDriverDbgSnapshotTest, OversizedSnapshotRequestIsClampedToPayloadLimit) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kEntryBytes = sizeof(kfd_dbg_device_info_entry);
  constexpr uint32_t kActualDevices = 1;
  constexpr size_t kGuard = 32;
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kEntryBytes + kGuard, kSentinel);
  std::atomic<uint32_t> observed_payload_bytes{0};
  std::atomic<uint32_t> observed_num_devices{0};

  std::jthread server([&, server_fd = sv[1]] {
    // Close unconditionally: an early return that leaves the socket open would
    // hang the client in rpc_recv_msg() instead of failing the test.
    const CloseOnScopeExit closer{server_fd};
    // Report the true device total and return only the entries that exist —
    // exactly what a daemon with one GPU does.
    serve_one_ioctl_reply(server_fd, 0, arg_struct_size, kEntryBytes, kPoison,
                          [&](const rocjitsu::RpcHeader &request, kfd_ioctl_dbg_trap_args &echoed) {
                            observed_payload_bytes = request.payload_bytes;
                            observed_num_devices = echoed.device_snapshot.num_devices;
                            echoed.device_snapshot.num_devices = kActualDevices;
                            echoed.device_snapshot.entry_size = kEntryBytes;
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = std::numeric_limits<uint32_t>::max();
  snap.device_snapshot.entry_size = kEntryBytes;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  server.join();

  EXPECT_LE(observed_payload_bytes.load(), rocjitsu::kMaxPayloadBytes)
      << "request exceeded the payload limit the daemon enforces";
  EXPECT_LT(observed_num_devices.load(), std::numeric_limits<uint32_t>::max())
      << "UINT32_MAX device count was transmitted verbatim";
  // The clamp is the transport's capacity, so it must still comfortably exceed
  // any device count a daemon could enumerate.
  EXPECT_GT(observed_num_devices.load(), 1000u);

  EXPECT_EQ(snap.device_snapshot.num_devices, kActualDevices) << "true device total not reported";
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.begin() + kEntryBytes, [](uint8_t b) {
    return b == kPoison;
  })) << "the enumerated entry did not reach the caller";
  EXPECT_TRUE(std::all_of(caller_buf.begin() + kEntryBytes, caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "copy-back wrote past the entries the daemon returned";
}

// num_devices and entry_size are inputs the request rewrites to the compact
// wire values, and the daemon echoes back whatever it was sent. On a failed op
// the caller must get its own values back, not those internals: the local path
// validates before writing any output, so it leaves both untouched. Uses the
// production shape -- UINT32_MAX devices at a stride wider than the struct --
// because that is where the mutation is visible: the clamp rewrites the count
// to 4096 and the stride down to the struct size.
TEST(RemoteDriverDbgSnapshotTest, FailedOversizedSnapshotLeavesTheRequestFieldsUntouched) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kRequested = std::numeric_limits<uint32_t>::max();
  constexpr uint32_t kStride = sizeof(kfd_dbg_device_info_entry) + 136;
  constexpr uint8_t kSentinel = 0xAB;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kStride, kSentinel);
  std::atomic<uint32_t> observed_num_devices{0};
  std::atomic<uint32_t> observed_entry_size{0};

  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    // A daemon that fails the op after the clamp has already rewritten the two
    // fields, echoing the wire values back in the reply's arg struct.
    serve_one_ioctl_reply(server_fd, -ENOSYS, arg_struct_size, 0, 0,
                          [&](const rocjitsu::RpcHeader &, kfd_ioctl_dbg_trap_args &echoed) {
                            observed_num_devices = echoed.device_snapshot.num_devices;
                            observed_entry_size = echoed.device_snapshot.entry_size;
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = kRequested;
  snap.device_snapshot.entry_size = kStride;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -ENOSYS);
  server.join();

  // The rewrite did happen on the wire -- otherwise this test would pass for
  // the wrong reason, having never produced the values it guards against.
  EXPECT_LT(observed_num_devices.load(), kRequested) << "the count was transmitted verbatim";
  EXPECT_EQ(observed_entry_size.load(), sizeof(kfd_dbg_device_info_entry))
      << "the caller's stride was transmitted verbatim";

  EXPECT_EQ(snap.device_snapshot.num_devices, kRequested)
      << "failed snapshot overwrote num_devices(IN) with the clamped wire count";
  EXPECT_EQ(snap.device_snapshot.entry_size, kStride)
      << "failed snapshot overwrote entry_size(IN) with the compact wire stride";
  EXPECT_EQ(snap.device_snapshot.snapshot_buf_ptr, reinterpret_cast<uint64_t>(caller_buf.data()));
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "failed snapshot mutated caller memory";
}

// Daemon mode must reproduce the driver's strided write, not overwrite the
// caller's whole declared capacity: amdkfd fills min(num_devices(IN), total)
// entries of entry_size(OUT) bytes at entry_size(IN) stride, so entries past
// the device total and the padding inside each entry stay as the caller left
// them. This is what SimulatedKfd does on the local path.
TEST(RemoteDriverDbgSnapshotTest, SuccessfulSnapshotLeavesUnfilledEntriesAndPaddingIntact) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kRequested = 4;
  constexpr uint32_t kReturned = 2; // daemon enumerates fewer GPUs than asked
  constexpr uint32_t kEntryBytes = sizeof(kfd_dbg_device_info_entry);
  constexpr uint32_t kStride = kEntryBytes + 16; // caller pads each entry
  constexpr size_t kCap = static_cast<size_t>(kRequested) * kStride;
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kUnwritten = 0;
  // Distinct per-entry contents, so scattering the packed tail into the
  // caller's wider slots at the wrong pitch shows up as the wrong entry rather
  // than as identical bytes.
  const auto entry_byte = [](uint32_t i) { return static_cast<uint8_t>(0xC0 + i); };
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kCap, kSentinel);

  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    // Mirror what SimulatedKfd writes into the daemon's tail. The client
    // transmits the compact stride, not the caller's, so the daemon lays
    // kReturned entries down back to back at kEntryBytes.
    serve_one_ioctl_reply(
        server_fd, 0, arg_struct_size, kCap, kUnwritten, snapshot_outputs(kReturned, kEntryBytes),
        [&](uint8_t *tail, size_t bytes) {
          std::memset(tail, kUnwritten, bytes);
          for (uint32_t i = 0; i < kReturned; ++i)
            std::memset(tail + static_cast<size_t>(i) * kEntryBytes, entry_byte(i), kEntryBytes);
        });
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = 4242;
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.num_devices = kRequested;
  snap.device_snapshot.entry_size = kStride;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_EQ(snap.device_snapshot.num_devices, kReturned);

  for (uint32_t i = 0; i < kRequested; ++i) {
    const auto entry = caller_buf.begin() + static_cast<size_t>(i) * kStride;
    const bool filled = i < kReturned;
    const uint8_t want = filled ? entry_byte(i) : kSentinel;
    EXPECT_TRUE(std::all_of(entry, entry + kEntryBytes, [&](uint8_t b) { return b == want; }))
        << "entry " << i
        << (filled ? " holds the wrong entry's bytes" : " was written past the device total");
    EXPECT_TRUE(
        std::all_of(entry + kEntryBytes, entry + kStride, [](uint8_t b) { return b == kSentinel; }))
        << "entry " << i << " padding was clobbered";
  }

  server.join();
}

// The queue op rides the same request clamp and strided copy-back as the device
// op; these pin the parts where its contract differs.

// A failed op must not touch the caller's buffer, exactly as it must not for
// the device op -- the reply's tail is ignored entirely.
TEST(RemoteDriverDbgQueueSnapshotTest, FailedSnapshotLeavesCallerBufferUntouched) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kRequested = 2;
  constexpr uint32_t kEntryBytes = sizeof(kfd_queue_snapshot_entry);
  constexpr size_t kCap = 2 * kEntryBytes;
  constexpr uint8_t kSentinel = 0xAB;

  std::array<uint8_t, kCap> caller_buf;
  caller_buf.fill(kSentinel);
  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    serve_one_ioctl_reply(server_fd, -EFAULT, sizeof(kfd_ioctl_dbg_trap_args), kCap, 0xCD);
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = kRequested;
  snap.queue_snapshot.entry_size = kEntryBytes;
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -EFAULT);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(),
                          [](uint8_t byte) { return byte == kSentinel; }));
  // A failed op leaves the caller's own inputs in place rather than the wire
  // values the request clamp wrote over them.
  EXPECT_EQ(snap.queue_snapshot.num_queues, kRequested);
  EXPECT_EQ(snap.queue_snapshot.entry_size, kEntryBytes);
  server.join();
}

// Unlike the device op, a null output buffer is not rejected on the request
// path: the local path answers it with success when there are no queues to
// report (SimulatedKfd::debug_queue_snapshot), and the client cannot know that
// before it asks. So the request goes out, and the verdict is reconstructed
// from the queue total that comes back.
TEST(RemoteDriverDbgQueueSnapshotTest, SuccessfulReplyWithNullBufferReturnsEfault) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr size_t kCap = sizeof(kfd_queue_snapshot_entry);
  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    serve_one_ioctl_reply(server_fd, 0, sizeof(kfd_ioctl_dbg_trap_args), kCap, 0xCD);
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = 1;
  snap.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -EFAULT);
  EXPECT_EQ(snap.queue_snapshot.snapshot_buf_ptr, 0u);
  server.join();
}

// A null buffer with nothing to report is success, not -EFAULT: that is the
// count-only probe, and it is how a debugger learns how large a buffer to
// allocate. Pinning it alongside the case above is what keeps the -EFAULT
// synthesis conditional on the returned total rather than on the null alone.
TEST(RemoteDriverDbgQueueSnapshotTest, NullBufferWithNoQueuesIsTheCountProbe) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    // A daemon whose process owns no queues: the total comes back zero.
    serve_one_ioctl_reply(server_fd, 0, sizeof(kfd_ioctl_dbg_trap_args), 0, 0,
                          [](const rocjitsu::RpcHeader &, kfd_ioctl_dbg_trap_args &echoed) {
                            echoed.queue_snapshot.num_queues = 0;
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = 1;
  snap.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_EQ(snap.queue_snapshot.num_queues, 0u);
  server.join();
}

// The caller's stride is wider than the entry struct, so the tail arrives packed
// at the struct and has to be scattered into the caller's wider slots. Anything
// past the entries the daemon reported -- including the padding inside the one
// slot that is filled -- stays as the caller left it.
TEST(RemoteDriverDbgQueueSnapshotTest, SuccessfulSnapshotScattersIntoTheCallerStride) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kEntryBytes = sizeof(kfd_queue_snapshot_entry);
  constexpr uint32_t kStride = kEntryBytes + 16;
  constexpr size_t kGuard = 32;
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;

  std::array<uint8_t, kStride + kGuard> caller_buf;
  caller_buf.fill(kSentinel);
  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    // The client transmits the compact stride, so the daemon lays its one entry
    // down at kEntryBytes and reports that as entry_size(OUT).
    serve_one_ioctl_reply(server_fd, 0, sizeof(kfd_ioctl_dbg_trap_args), kEntryBytes, kPoison,
                          [](const rocjitsu::RpcHeader &, kfd_ioctl_dbg_trap_args &echoed) {
                            // Spelled out rather than reaching for kEntryBytes: inside the
                            // enclosing [&] lambda that name is a captured reference, which
                            // gcc then demands this lambda capture too and clang rejects as
                            // an unnecessary capture.
                            echoed.queue_snapshot.num_queues = 1;
                            echoed.queue_snapshot.entry_size =
                                static_cast<uint32_t>(sizeof(kfd_queue_snapshot_entry));
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = 1;
  snap.queue_snapshot.entry_size = kStride;
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.begin() + kEntryBytes, [](uint8_t byte) {
    return byte == kPoison;
  })) << "the entry did not reach the caller's buffer";
  EXPECT_TRUE(std::all_of(caller_buf.begin() + kEntryBytes, caller_buf.end(), [](uint8_t byte) {
    return byte == kSentinel;
  })) << "copy-back wrote past the one entry the daemon returned";
  server.join();
}

// rocdbgapi asks for UINT32_MAX queues the same way libhsakmt asks for
// UINT32_MAX devices, and for the same reason: let the driver report the true
// total. The request clamp has to bound the queue op too, or serializing that
// count reserves a tail no allocator will hand out.
TEST(RemoteDriverDbgQueueSnapshotTest, OversizedRequestIsClampedToPayloadLimit) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr uint32_t kEntryBytes = sizeof(kfd_queue_snapshot_entry);
  constexpr uint32_t kActualQueues = 1;
  constexpr size_t kGuard = 32;
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;

  std::vector<uint8_t> caller_buf(kEntryBytes + kGuard, kSentinel);
  std::atomic<uint32_t> observed_payload_bytes{0};
  std::atomic<uint32_t> observed_num_queues{0};

  std::jthread server([&, server_fd = sv[1]] {
    const CloseOnScopeExit closer{server_fd};
    serve_one_ioctl_reply(server_fd, 0, sizeof(kfd_ioctl_dbg_trap_args), kEntryBytes, kPoison,
                          [&](const rocjitsu::RpcHeader &request, kfd_ioctl_dbg_trap_args &echoed) {
                            observed_payload_bytes = request.payload_bytes;
                            observed_num_queues = echoed.queue_snapshot.num_queues;
                            echoed.queue_snapshot.num_queues = kActualQueues;
                            echoed.queue_snapshot.entry_size = kEntryBytes;
                          });
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = std::numeric_limits<uint32_t>::max();
  snap.queue_snapshot.entry_size = kEntryBytes;
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  ASSERT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  server.join();

  EXPECT_LE(observed_payload_bytes.load(), rocjitsu::kMaxPayloadBytes)
      << "request exceeded the payload limit the daemon enforces";
  EXPECT_LT(observed_num_queues.load(), std::numeric_limits<uint32_t>::max())
      << "UINT32_MAX queue count was transmitted verbatim";
  EXPECT_GT(observed_num_queues.load(), 1000u);

  EXPECT_EQ(snap.queue_snapshot.num_queues, kActualQueues) << "true queue total not reported";
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.begin() + kEntryBytes, [](uint8_t byte) {
    return byte == kPoison;
  })) << "the enumerated entry did not reach the caller";
  EXPECT_TRUE(std::all_of(caller_buf.begin() + kEntryBytes, caller_buf.end(), [](uint8_t byte) {
    return byte == kSentinel;
  })) << "copy-back wrote past the entries the daemon returned";
}

// A closed but positive notifier fd cannot be transferred over SCM_RIGHTS:
// sendmsg() rejects it with EBADF at the client. send_ioctl() must surface that
// errno so the interposer reports EBADF, not the EPERM a bare -1 becomes
// (-EPERM == -1).
TEST(RemoteDriverDbgNotifierTest, EnableWithClosedNotifierFdPreservesEbadf) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  rocjitsu::RemoteDriver rd(sv[0]);

  // A positive fd number that is already closed: a valid-looking dbg_fd the
  // SCM_RIGHTS send must reject. Allocated after the driver so its fd number is
  // not reused by the driver's internal eventfd before the send.
  //
  // Park it far above the lowest-free-fd range first. open() hands out the
  // lowest free descriptor, so a plain close() leaves this number first in line:
  // anything else in the process that opens a descriptor between here and the
  // send below takes it, the send then succeeds, and the test stops testing
  // what it is named for -- it reports whatever the daemon path answers instead,
  // and can block waiting for a reply that no one is serving.
  int dead_fd = ::eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dead_fd, 0);
  constexpr int kParkedFdBase = 4096;
  const int parked_fd = ::fcntl(dead_fd, F_DUPFD_CLOEXEC, kParkedFdBase);
  if (parked_fd >= 0) {
    ASSERT_EQ(::close(dead_fd), 0);
    dead_fd = parked_fd;
  }
  ASSERT_EQ(::close(dead_fd), 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = 4242;
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dead_fd);

  // The transport rejects the closed fd; the caller must see EBADF, not the bare
  // -1 that would surface as EPERM.
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &en), -EBADF);

  ::close(sv[1]);
}

// A daemon-path ENABLE that the daemon fails (result != 0) must not copy the
// response tail into the caller's runtime-info buffer, even though the daemon
// returned inline bytes. Mirrors the GET_DEVICE_SNAPSHOT success gate: a
// rejected notifier fd (-EBADF) leaves caller memory untouched, as local mode does.
TEST(RemoteDriverDbgEnableTest, FailedEnableLeavesCallerRuntimeInfoUntouched) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr size_t kRinfoSize = sizeof(kfd_runtime_info);
  constexpr uint8_t kSentinel = 0xAB;
  constexpr uint8_t kPoison = 0xCD;
  const size_t arg_struct_size = sizeof(kfd_ioctl_dbg_trap_args);

  std::vector<uint8_t> caller_buf(kRinfoSize, kSentinel);

  std::jthread server([&, server_fd = sv[1]] {
    serve_one_ioctl_reply(server_fd, -EBADF, arg_struct_size, kRinfoSize, kPoison);
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);

  // A valid notifier fd so the SCM_RIGHTS send succeeds and the request reaches
  // the daemon, which then fails the op with -EBADF. Allocated after the driver
  // so its fd number is not reused by the driver's internal eventfd.
  int notifier_fd = ::eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(notifier_fd, 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(notifier_fd);
  en.enable.rinfo_size = static_cast<uint32_t>(kRinfoSize);
  en.enable.rinfo_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &en), -EBADF);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.end(), [](uint8_t b) {
    return b == kSentinel;
  })) << "failed ENABLE mutated caller runtime-info memory";

  ::close(notifier_fd);
  server.join();
}

// --- RemoteDriver RPC stream poisoning ---
//
// Every exchange on the socket is length-framed, so a request that stops
// mid-write, or a reply that is not consumed whole, leaves the two ends
// disagreeing about where the next frame starts. From then on any call would
// decode its result -- and the bytes it copies into caller memory -- out of some
// other exchange's remains. The client marks the connection terminal instead,
// and every entry point has to honour that mark, not just the one that tripped
// over it: ioctl, mmap and munmap all share the one stream.

// Framed size of an RPC_IOCTL request carrying `arg_bytes` of ioctl args and no
// inline tail.
constexpr size_t framed_ioctl_request_bytes(size_t arg_bytes) {
  return sizeof(rocjitsu::RpcHeader) + sizeof(rocjitsu::RpcIoctlRequest) + arg_bytes;
}

// A reply header cut in half, then EOF on the daemon's write half. The client's
// receive asks for MSG_WAITALL but comes back short — the same desync a
// signal-interrupted wait produces, without having to time a signal. Queued
// before the client sends anything: the two directions are independent, so the
// reply can already be waiting when the request goes out.
TEST(RemoteDriverStreamPoisonTest, TruncatedIoctlReplyMakesEveryLaterCallFailFast) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};

  const std::vector<uint8_t> half_header(sizeof(rocjitsu::RpcHeader) / 2, 0xA5);
  ASSERT_TRUE(rocjitsu::rpc_send_exact(sv[1], half_header.data(), half_header.size()));
  ASSERT_EQ(::shutdown(sv[1], SHUT_WR), 0) << ::strerror(errno);

  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_GET_VERSION, &version), -EPROTO);

  // Drain the one request that did go out, so anything still readable afterwards
  // is a write the poisoned client should never have made.
  std::vector<uint8_t> request_seen(framed_ioctl_request_bytes(sizeof(version)));
  ASSERT_TRUE(rocjitsu::rpc_recv_exact(sv[1], request_seen.data(), request_seen.size()));

  kfd_ioctl_get_version_args after{};
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_GET_VERSION, &after), -EPROTO);
  EXPECT_EQ(rd.munmap(reinterpret_cast<void *>(0x1000), 0x1000), -EPROTO);
  errno = 0;
  EXPECT_EQ(rd.mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, 0), MAP_FAILED);
  EXPECT_EQ(errno, EPROTO);

  // Nothing more on the wire, and the socket is shut down rather than merely
  // idle: recv() reports EOF instead of blocking or handing back another frame.
  uint8_t unexpected = 0;
  EXPECT_EQ(::recv(sv[1], &unexpected, sizeof(unexpected), MSG_DONTWAIT), 0)
      << "a poisoned client kept talking to the daemon";
}

// The daemon frames every mmap reply as header + RpcMmapResponse. A reply that
// stops after the header must be refused: reading result out of it takes the
// value from whatever the client's receive buffer happened to hold, so the
// caller could be handed a mapping — or a failure — decided by stack garbage.
TEST(RemoteDriverStreamPoisonTest, ShortMmapReplyIsRejectedRatherThanReadOffTheStack) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};

  rocjitsu::RpcHeader reply{};
  reply.opcode = rocjitsu::RPC_MMAP;
  reply.payload_bytes = sizeof(rocjitsu::RpcMmapResponse);
  ASSERT_TRUE(rocjitsu::rpc_send_exact(sv[1], &reply, sizeof(reply)));
  ASSERT_EQ(::shutdown(sv[1], SHUT_WR), 0) << ::strerror(errno);

  rocjitsu::RemoteDriver rd(sv[0]);

  // This is the call that TRIPS the poison, not one that finds the connection
  // already dead, so it is the one whose errno the documented contract is
  // easiest to break: nothing on the path to MAP_FAILED sets errno as a side
  // effect (the receive succeeded -- it just came back short -- and the
  // shutdown that poisons the stream succeeds too).
  errno = 0;
  EXPECT_EQ(rd.mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, 0), MAP_FAILED);
  EXPECT_EQ(errno, EPROTO) << "a failed mmap left errno for the caller to guess at";

  // The mmap and ioctl paths share one stream, so a desync noticed by one is
  // terminal for the other.
  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_GET_VERSION, &version), -EPROTO);
}

// The daemon's own failure has to reach the caller as its errno too, and EPERM
// is the value that tests the encoding rather than the plumbing: EPERM is 1, so
// a daemon result of -EPERM is the bit pattern -1, which the transport used to
// hand back as its "no usable errno" sentinel. Anything that special-cases -1
// on the way out swallows exactly this reply and leaves the caller reading a
// stale errno, which is why the sentinel was removed rather than exempted.
TEST(RemoteDriverStreamPoisonTest, DaemonReportedEpermReachesTheCallerAsEperm) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};

  // A complete, well-formed mmap reply -- header plus RpcMmapResponse -- that
  // simply reports failure. Nothing here is a framing error.
  rocjitsu::RpcHeader reply{};
  reply.opcode = rocjitsu::RPC_MMAP;
  reply.payload_bytes = sizeof(rocjitsu::RpcMmapResponse);
  reply.result = -EPERM;
  rocjitsu::RpcMmapResponse body{};
  ASSERT_TRUE(rocjitsu::rpc_send_exact(sv[1], &reply, sizeof(reply)));
  ASSERT_TRUE(rocjitsu::rpc_send_exact(sv[1], &body, sizeof(body)));

  rocjitsu::RemoteDriver rd(sv[0]);

  // A sentinel no errno takes, so "left untouched" is distinguishable from
  // "reported as 0".
  errno = 12345;
  EXPECT_EQ(rd.mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, 0), MAP_FAILED);
  EXPECT_EQ(errno, EPERM) << "the daemon's EPERM was mistaken for an absent errno";
}

// A munmap whose reply never arrives has already put its request on the wire, so
// the client cannot know where the next frame begins either. It must not unmap
// on the strength of a reply it never read, and must not leave the connection
// looking usable.
TEST(RemoteDriverStreamPoisonTest, UnansweredMunmapPoisonsTheConnection) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};

  ASSERT_EQ(::shutdown(sv[1], SHUT_WR), 0) << ::strerror(errno);

  rocjitsu::RemoteDriver rd(sv[0]);
  EXPECT_EQ(rd.munmap(reinterpret_cast<void *>(0x1000), 0x1000), -EPROTO);

  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_GET_VERSION, &version), -EPROTO);
}

// The other half of the framing contract: a request that stops part way through
// leaves the daemon holding the front of a frame whose tail never comes, and it
// will read the next request as that tail. A send buffer far smaller than the
// request, on a socket that returns instead of blocking once the buffer fills,
// produces exactly that without any timing dependence.
TEST(RemoteDriverStreamPoisonTest, PartiallyWrittenRequestPoisonsTheConnection) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  const CloseOnScopeExit server_closer{sv[1]};

  int send_buffer_bytes = 4096;
  ASSERT_EQ(
      ::setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &send_buffer_bytes, sizeof(send_buffer_bytes)), 0)
      << ::strerror(errno);
  int socket_flags = ::fcntl(sv[0], F_GETFL, 0);
  ASSERT_NE(socket_flags, -1) << ::strerror(errno);
  ASSERT_EQ(::fcntl(sv[0], F_SETFL, socket_flags | O_NONBLOCK), 0) << ::strerror(errno);

  rocjitsu::RemoteDriver rd(sv[0]);

  // Device ids are serialized inline after the arg struct, so this is a request
  // two orders of magnitude larger than the send buffer. Nothing on the daemon
  // side reads, so the buffer cannot drain mid-write.
  std::vector<uint32_t> device_ids(64 * 1024, kGpuId);
  kfd_ioctl_map_memory_to_gpu_args map_args{};
  map_args.handle = 1;
  map_args.n_devices = static_cast<uint32_t>(device_ids.size());
  map_args.device_ids_array_ptr = reinterpret_cast<uint64_t>(device_ids.data());
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map_args), -EPROTO);

  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_GET_VERSION, &version), -EPROTO);
  EXPECT_EQ(rd.munmap(reinterpret_cast<void *>(0x1000), 0x1000), -EPROTO);
}

// Deterministic regression for the close()-vs-in-flight-ioctl teardown ordering.
// After close() fully tears a process down, a subsequent ioctl on that process id
// must FAIL cleanly (-ESRCH) rather than operate on dismantled per-process state
// (allocations/queues/doorbells already cleared). This exercises the lifetime
// invariant that ioctl() must not mutate a torn-down process; the threaded
// SimulatedKfdTest.ConcurrentIoctlAndCloseIsRaceFree covers the racing variant
// under TSan, this one pins the post-teardown contract without timing.
TEST_F(KfdIoctlTest, IoctlAfterCloseFailsCleanly) {
  ASSERT_NE(soc_, nullptr);
  rocjitsu::SimulatedKfd daemon_driver(*soc_, true);
  uint32_t pid = daemon_driver.open_process();
  ASSERT_NE(pid, 0u);

  // A state-touching ioctl works while the process is live.
  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = 0x100000000ULL;
  alloc.size = 0x1000;
  alloc.gpu_id = kGpuId;
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  EXPECT_EQ(daemon_driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  // Tear the process down (single open reference -> full teardown).
  EXPECT_EQ(daemon_driver.close(pid), 0);

  // Any ioctl on the now-closed process id must fail cleanly, not touch freed
  // state. -ESRCH is returned once the process is gone from the table.
  kfd_ioctl_alloc_memory_of_gpu_args after{};
  after.va_addr = 0x200000000ULL;
  after.size = 0x1000;
  after.gpu_id = kGpuId;
  after.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  EXPECT_EQ(daemon_driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &after), -ESRCH);

  kfd_ioctl_get_version_args ver{};
  EXPECT_EQ(daemon_driver.ioctl(pid, AMDKFD_IOC_GET_VERSION, &ver), -ESRCH);
}

// Deterministic regression for destructor teardown of a multiply-opened process.
// close() only tears a process down on the LAST open reference, so a process with
// open_ref_count_ > 1 survives a single close(). ~SimulatedKfd must keep closing
// each snapshotted pid until it is fully drained, otherwise its allocations,
// queues, and CP callbacks leak past the driver. This pins that drain: a daemon
// process opened twice (same client_pid -> shared, refcount 2) plus a live
// allocation, then the driver is destroyed without an explicit close().
TEST_F(KfdIoctlTest, DestructorDrainsMultiplyOpenedProcess) {
  ASSERT_NE(soc_, nullptr);
  uint32_t pid = 0;
  {
    rocjitsu::SimulatedKfd daemon_driver(*soc_, true);
    // Same client_pid twice -> one shared process with open_ref_count_ == 2.
    pid = daemon_driver.open_process(/*client_pid=*/4242);
    ASSERT_NE(pid, 0u);
    uint32_t pid2 = daemon_driver.open_process(/*client_pid=*/4242);
    EXPECT_EQ(pid2, pid);

    kfd_ioctl_alloc_memory_of_gpu_args alloc{};
    alloc.va_addr = 0x100000000ULL;
    alloc.size = 0x1000;
    alloc.gpu_id = kGpuId;
    alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    EXPECT_EQ(daemon_driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

    // A single close() drops one of the two references; the process is still live.
    EXPECT_EQ(daemon_driver.close(pid), 0);

    // Prove the process survived the first close() (open_ref_count_ still 1): a
    // state-touching ioctl must still succeed rather than return -ESRCH. If close()
    // had torn it down on the first reference, this would fail cleanly instead.
    kfd_ioctl_alloc_memory_of_gpu_args live{};
    live.va_addr = 0x200000000ULL;
    live.size = 0x1000;
    live.gpu_id = kGpuId;
    live.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    EXPECT_EQ(daemon_driver.ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &live), 0);

    // daemon_driver goes out of scope here -> ~SimulatedKfd must drain the
    // still-open (refcount 1) process fully. Under ASan/leak checking this fails
    // if the destructor leaks the process's allocation/memfd.
  }
  // No crash / no leak reported == pass. (pid intentionally unused past scope.)
  (void)pid;
}

TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotEnumeratesAgent) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // First call learns the total device count. rocdbgapi's
  // kfd_snapshots::fetch() probes with a real one-entry buffer rather than a
  // null pointer, so model that: the count comes back in num_devices(OUT)
  // whether or not the buffer was big enough.
  kfd_dbg_device_info_entry probe_entry{};
  kfd_ioctl_dbg_trap_args count{};
  count.pid = static_cast<uint32_t>(getpid());
  count.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  count.device_snapshot.entry_size = sizeof(kfd_dbg_device_info_entry);
  count.device_snapshot.num_devices = 1;
  count.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&probe_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count), 0);
  ASSERT_EQ(count.device_snapshot.num_devices, 1u);
  EXPECT_EQ(count.device_snapshot.entry_size, sizeof(kfd_dbg_device_info_entry));

  // Second call fills one entry.
  kfd_dbg_device_info_entry entry{};
  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = static_cast<uint32_t>(getpid());
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.entry_size = sizeof(entry);
  snap.device_snapshot.num_devices = 1;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);

  const rocjitsu::kmd::DebugTopology topology = rocjitsu::kmd::effective_topology_for(
      loaded_.device.gfx_target_version, loaded_.device.capability, loaded_.device.capability2,
      loaded_.device.debug_prop, loaded_.device.revision_id);
  EXPECT_EQ(entry.gpu_id, kGpuId);
  EXPECT_EQ(entry.gfx_target_version, 90500u); // gfx950 fixture config
  EXPECT_EQ(entry.subsystem_vendor_id, loaded_.device.vendor_id);
  EXPECT_EQ(entry.subsystem_device_id, loaded_.device.device_id);
  EXPECT_EQ(entry.capability, topology.capability);
  EXPECT_EQ(entry.debug_prop, topology.debug_prop);
  // rocm-dbgapi's agent_snapshot fatal-errors if any of these are zero.
  EXPECT_NE(entry.simd_count, 0u);
  EXPECT_NE(entry.max_waves_per_simd, 0u);
  EXPECT_NE(entry.array_count, 0u);
  // Fatal: the shader-engine check below divides by this.
  ASSERT_NE(entry.simd_arrays_per_engine, 0u);

  // The snapshot reports shader arrays per XCC, as amdkfd does. rocdbgapi turns
  // that back into shader engines and uses the result for CWSR and scratch
  // addressing, so it has to agree with the SoC the simulator actually runs —
  // non-zero is not enough.
  uint32_t simulated_shader_engines = 0;
  for (uint32_t i = 0; i < soc_->num_xcds(); ++i)
    simulated_shader_engines += soc_->xcd(i)->num_shader_engines();
  EXPECT_EQ(entry.array_count * entry.num_xcc / entry.simd_arrays_per_engine,
            simulated_shader_engines);
  EXPECT_EQ(entry.num_xcc, soc_->num_xcds());
}

// kfd_dbg_trap_device_snapshot() validates its arguments before it writes any
// of them: a null snapshot buffer is a malformed request, rejected with -EINVAL
// and with the caller's num_devices/entry_size left exactly as they were. A
// caller cannot read a device total off a call that failed this way.
TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotRejectsNullBufferWithoutWritingOutputs) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // Ask for more devices than exist, at a wider stride than the struct, so a
  // written-back output would differ from the input and the assertions can tell
  // "reported" apart from "left as the caller set it".
  constexpr uint32_t kRequestedDevices = 2;
  constexpr uint32_t kRequestedStride = sizeof(kfd_dbg_device_info_entry) + 16;
  kfd_ioctl_dbg_trap_args null_buf{};
  null_buf.pid = static_cast<uint32_t>(getpid());
  null_buf.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  null_buf.device_snapshot.num_devices = kRequestedDevices;
  null_buf.device_snapshot.entry_size = kRequestedStride;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &null_buf), -EINVAL);
  EXPECT_EQ(null_buf.device_snapshot.snapshot_buf_ptr, 0u);
  EXPECT_EQ(null_buf.device_snapshot.num_devices, kRequestedDevices)
      << "a rejected request must not report a device total";
  EXPECT_EQ(null_buf.device_snapshot.entry_size, kRequestedStride);
}

// A zero stride is not an error. The driver's per-entry copy_to_user() moves
// entry_size(OUT) == 0 bytes and succeeds, so the call reports the device total
// and writes nothing -- the caller learns the count without providing room for
// a single entry.
TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotZeroStrideReportsCountAndWritesNothing) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  constexpr uint8_t kSentinel = 0xAB;
  std::array<uint8_t, sizeof(kfd_dbg_device_info_entry)> buffer{};
  buffer.fill(kSentinel);

  kfd_ioctl_dbg_trap_args zero_stride{};
  zero_stride.pid = static_cast<uint32_t>(getpid());
  zero_stride.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  zero_stride.device_snapshot.num_devices = 1;
  zero_stride.device_snapshot.entry_size = 0;
  zero_stride.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(buffer.data());
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &zero_stride), 0);
  EXPECT_EQ(zero_stride.device_snapshot.num_devices, 1u);
  EXPECT_EQ(zero_stride.device_snapshot.entry_size, 0u);
  EXPECT_TRUE(std::all_of(buffer.begin(), buffer.end(), [](uint8_t byte) {
    return byte == kSentinel;
  })) << "zero-stride snapshot wrote into the caller's buffer";
}

TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotEnumeratesMultipleAgentsWithCallerStride) {
  constexpr uint32_t kSecondGpuId = kGpuId + 1;
  rocjitsu::SimulatedKfd multi_gpu_driver({soc_, soc_}, {kGpuId, kSecondGpuId});
  std::vector<rocjitsu::config::KfdDeviceConfig> devices(2, loaded_.device);
  devices[1].gpu_id = kSecondGpuId;
  devices[1].location_id++;
  devices[1].drm_render_minor++;
  multi_gpu_driver.setup_topology(devices, soc_->num_xcds());
  ASSERT_GE(multi_gpu_driver.open(), 0);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(multi_gpu_driver.ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  constexpr uint32_t kEntryStride = sizeof(kfd_dbg_device_info_entry) + 16;
  std::array<uint8_t, kEntryStride * 2> buffer{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snapshot.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(buffer.data());
  snapshot.device_snapshot.num_devices = 1;
  snapshot.device_snapshot.entry_size = kEntryStride;
  ASSERT_EQ(multi_gpu_driver.ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);
  EXPECT_EQ(snapshot.device_snapshot.num_devices, 2u);
  EXPECT_EQ(snapshot.device_snapshot.entry_size, sizeof(kfd_dbg_device_info_entry));
  kfd_dbg_device_info_entry first{};
  std::memcpy(&first, buffer.data(), sizeof(first));
  EXPECT_EQ(first.gpu_id, kGpuId);
  EXPECT_TRUE(std::all_of(buffer.begin() + sizeof(first), buffer.end(),
                          [](uint8_t byte) { return byte == 0; }));

  snapshot.device_snapshot.num_devices = 2;
  snapshot.device_snapshot.entry_size = kEntryStride;
  ASSERT_EQ(multi_gpu_driver.ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);
  kfd_dbg_device_info_entry second{};
  std::memcpy(&first, buffer.data(), sizeof(first));
  std::memcpy(&second, buffer.data() + kEntryStride, sizeof(second));
  EXPECT_EQ(first.gpu_id, kGpuId);
  EXPECT_EQ(second.gpu_id, kSecondGpuId);
  EXPECT_EQ(second.location_id, devices[1].location_id);

  std::array<kfd_process_device_apertures, 2> apertures{};
  kfd_ioctl_get_process_apertures_new_args aperture_args{};
  aperture_args.kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(apertures.data());
  aperture_args.num_of_nodes = apertures.size();
  ASSERT_EQ(multi_gpu_driver.ioctl(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &aperture_args), 0);
  EXPECT_EQ(aperture_args.num_of_nodes, 2u);
  EXPECT_EQ(first.lds_base, apertures[0].lds_base);
  EXPECT_EQ(first.scratch_base, apertures[0].scratch_base);
  EXPECT_EQ(first.gpuvm_base, apertures[0].gpuvm_base);
  EXPECT_EQ(second.lds_base, apertures[1].lds_base);
  EXPECT_EQ(second.scratch_base, apertures[1].scratch_base);
  EXPECT_EQ(second.gpuvm_base, apertures[1].gpuvm_base);

  multi_gpu_driver.close();
}

// Only devices the snapshot can actually describe are enumerable, so the device
// total is clamped to the per-GPU metadata captured by setup_topology, not to
// the number of SoCs the driver was built over. An embedder is free to skip
// setup_topology entirely, or to call the single-device overload on a driver
// holding several SoCs, and both leave fewer descriptions than GPUs.
//
// Reporting the GPU count regardless would look like the obvious
// simplification, and every other test in this file keeps passing when you make
// it -- the fill loop then indexes gpu_infos_ past its end and the process dies
// on the first snapshot with no metadata (verified: substituting gpus_.size()
// here turns this test into a SIGSEGV). Short of the crash, a partially
// described set would hand rocdbgapi entries with simd_count or array_count
// zero, which its agent_snapshot treats as fatal rather than as a bad ioctl.
TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotEnumeratesOnlyDescribableDevices) {
  constexpr uint32_t kSecondGpuId = kGpuId + 1;
  constexpr uint32_t kEntryBytes = sizeof(kfd_dbg_device_info_entry);
  constexpr uint8_t kSentinel = 0xAB;

  // Two GPUs, and a snapshot request with room for both, in every case below.
  auto snapshot_two = [&](rocjitsu::SimulatedKfd &driver, std::array<uint8_t, kEntryBytes * 2> &buf,
                          kfd_ioctl_dbg_trap_args &snapshot) {
    ASSERT_GE(driver.open(), 0);
    kfd_ioctl_dbg_trap_args enable{};
    enable.pid = static_cast<uint32_t>(getpid());
    enable.op = KFD_IOC_DBG_TRAP_ENABLE;
    enable.enable.dbg_fd = make_debug_fd();
    ASSERT_EQ(driver.ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

    buf.fill(kSentinel);
    snapshot = {};
    snapshot.pid = static_cast<uint32_t>(getpid());
    snapshot.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
    snapshot.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(buf.data());
    snapshot.device_snapshot.num_devices = 2;
    snapshot.device_snapshot.entry_size = kEntryBytes;
    ASSERT_EQ(driver.ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);
  };

  // No metadata at all: the request succeeds and reports nothing to describe,
  // rather than handing back two zero-filled entries.
  {
    SCOPED_TRACE("setup_topology never called");
    rocjitsu::SimulatedKfd driver({soc_, soc_}, {kGpuId, kSecondGpuId});
    std::array<uint8_t, kEntryBytes * 2> buf{};
    kfd_ioctl_dbg_trap_args snapshot{};
    snapshot_two(driver, buf, snapshot);
    EXPECT_EQ(snapshot.device_snapshot.num_devices, 0u);
    EXPECT_EQ(snapshot.device_snapshot.entry_size, kEntryBytes);
    EXPECT_TRUE(std::all_of(buf.begin(), buf.end(), [](uint8_t byte) { return byte == kSentinel; }))
        << "an undescribable device was written out anyway";
    driver.close();
  }

  // Metadata for one of the two: the described GPU is enumerated, the other is
  // not, and its slot is left as the caller had it.
  {
    SCOPED_TRACE("single-device setup_topology on a two-GPU driver");
    rocjitsu::SimulatedKfd driver({soc_, soc_}, {kGpuId, kSecondGpuId});
    driver.setup_topology(loaded_.device, soc_->num_xcds());
    std::array<uint8_t, kEntryBytes * 2> buf{};
    kfd_ioctl_dbg_trap_args snapshot{};
    snapshot_two(driver, buf, snapshot);
    EXPECT_EQ(snapshot.device_snapshot.num_devices, 1u);
    kfd_dbg_device_info_entry first{};
    std::memcpy(&first, buf.data(), sizeof(first));
    EXPECT_EQ(first.gpu_id, kGpuId);
    EXPECT_NE(first.simd_count, 0u);
    EXPECT_NE(first.array_count, 0u);
    EXPECT_TRUE(std::all_of(buf.begin() + kEntryBytes, buf.end(), [](uint8_t byte) {
      return byte == kSentinel;
    })) << "the undescribed second GPU was enumerated";
    driver.close();
  }
}

// The simulator services a debugger by writing a CWSR record rocm-dbgapi parses
// out of /proc/<pid>/mem, so an agent it cannot produce a record for is not
// debuggable no matter what the driver would advertise for that part. The
// support bit is what rocdbgapi keys agent_t::supports_debugging() on, so
// withholding it declines at attach instead of at every individual wave stop.
TEST(KfdTopologyTest, TrapDebugSupportTracksTheModelledCwsrLayouts) {
  struct Part {
    uint32_t gfx_target_version;
    const char *name;
  };
  // Every part the arch-keyed predicate accepts, and a representative spread of
  // the ones it does not: an older gfx9 and two RDNA generations.
  constexpr Part kModelled[] = {{90402u, "gfx942"}, {90500u, "gfx950"}, {120500u, "gfx1250"}};
  constexpr Part kUnmodelled[] = {{90010u, "gfx90a"}, {110000u, "gfx1100"}, {120000u, "gfx1200"}};

  for (const Part &part : kModelled) {
    const rocjitsu::kmd::DebugTopology topology =
        rocjitsu::kmd::effective_topology_for(part.gfx_target_version, 0, 0, 0, 0);
    EXPECT_NE(topology.capability & HSA_CAP_TRAP_DEBUG_SUPPORT, 0u) << part.name;
    EXPECT_TRUE(rocjitsu::kmd::cwsr_layout_modelled_for_gc_ip_version(
        rocjitsu::kmd::gc_ip_version_for_gfx_target_version(part.gfx_target_version)))
        << part.name;
  }

  for (const Part &part : kUnmodelled) {
    const rocjitsu::kmd::DebugTopology topology =
        rocjitsu::kmd::effective_topology_for(part.gfx_target_version, 0, 0, 0, 0);
    EXPECT_EQ(topology.capability & HSA_CAP_TRAP_DEBUG_SUPPORT, 0u) << part.name;
    // Only that one bit is withheld: the node still describes the device the
    // driver would, so a non-debug consumer sees no difference.
    EXPECT_NE(topology.capability & HSA_CAP_ATS_PRESENT, 0u) << part.name;
    EXPECT_NE(topology.capability & HSA_CAP_WATCH_POINTS_SUPPORTED, 0u) << part.name;
    EXPECT_NE(topology.debug_prop, 0u) << part.name;
  }
}

// The two selectors name the same layout through different identities, so a
// part added to one or assigned the wrong ABI in the other cannot pass as an
// equivalent boolean support gate.
TEST(KfdTopologyTest, ArchAndGcSpellingsOfTheCwsrGateAgree) {
  struct Part {
    rj_code_arch_t arch;
    uint32_t gfx_target_version;
    const char *name;
  };
  constexpr Part kParts[] = {
      {ROCJITSU_CODE_ARCH_CDNA1, 90002u, "gfx908"},
      {ROCJITSU_CODE_ARCH_CDNA2, 90010u, "gfx90a"},
      {ROCJITSU_CODE_ARCH_CDNA3, 90402u, "gfx942"},
      {ROCJITSU_CODE_ARCH_CDNA4, 90500u, "gfx950"},
      {ROCJITSU_CODE_ARCH_CDNA5, 120500u, "gfx1250"},
      {ROCJITSU_CODE_ARCH_RDNA3, 110000u, "gfx1100"},
      {ROCJITSU_CODE_ARCH_RDNA4, 120000u, "gfx1200"},
  };

  for (const Part &part : kParts)
    EXPECT_EQ(rocjitsu::kmd::cwsr_layout_kind(part.arch),
              rocjitsu::kmd::cwsr_layout_kind_for_gc_ip_version(
                  rocjitsu::kmd::gc_ip_version_for_gfx_target_version(part.gfx_target_version)))
        << part.name;
}

// A config may capture a real device's capability word, which carries that
// device's trap-debug bit. Inheriting it would restore exactly the mismatch the
// derived gate closes.
TEST(KfdTopologyTest, CapturedCapabilityCannotReadvertiseUnservicableTrapDebug) {
  const uint32_t captured = rocjitsu::kmd::default_non_debug_capability() |
                            HSA_CAP_TRAP_DEBUG_SUPPORT | HSA_CAP_TRAP_DEBUG_FIRMWARE_SUPPORTED;

  const rocjitsu::kmd::DebugTopology unmodelled =
      rocjitsu::kmd::effective_topology_for(110000u, captured, 0, 0, 0);
  EXPECT_EQ(unmodelled.capability & HSA_CAP_TRAP_DEBUG_SUPPORT, 0u);
  EXPECT_NE(unmodelled.capability & HSA_CAP_ATS_PRESENT, 0u) << "the override was discarded";

  // The same captured word on a part the codec does model passes through.
  const rocjitsu::kmd::DebugTopology modelled =
      rocjitsu::kmd::effective_topology_for(90500u, captured, 0, 0, 0);
  EXPECT_NE(modelled.capability & HSA_CAP_TRAP_DEBUG_SUPPORT, 0u);
}

TEST(KfdTopologyTest, EffectiveTopologyDerivesGfx121Capability2) {
  const rocjitsu::kmd::DebugTopology topology =
      rocjitsu::kmd::effective_topology_for(120100u, 0, 0, 0, 0);

  EXPECT_NE(topology.capability & HSA_CAP_ATS_PRESENT, 0u);
  EXPECT_NE(topology.capability2 & HSA_CAP2_TRAP_DEBUG_LDS_OUT_OF_ADDR_RANGE_SUPPORTED, 0u);
}

TEST_F(KfdIoctlCdna5Test, DbgTrapCwsrRoundTripPreservesGfx1250WaveState) {
  constexpr uint64_t kCwsrAddress = 0x600100000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;
  constexpr uint32_t kStatus = 1u | (5u << 1) | (1u << 7) | (1u << 19);
  constexpr uint32_t kMode = (1u << 25) | (0xA5u << 12) | 0xF0u;
  constexpr uint32_t kTrapsts = 0x7Fu | (1u << 7) | (1u << 8) | (1u << 10) | (1u << 11) |
                                (0x7u << 12) | (0x1Fu << 22) | (1u << 28) | (0x3u << 30);
  constexpr uint32_t kTrapCtrl = 0x2AAu;
  constexpr uint32_t kXnackStatePriv = 0x00057F7Fu;
  constexpr uint32_t kEditedXnackStatePriv = 0x00015555u;
  constexpr uint32_t kXnackMask = 0xA5A55A5Au;
  constexpr uint32_t kNumVgprs = 1024;

  std::vector<uint8_t> cwsr(static_cast<size_t>(kCwsrSize) * soc_->num_xcds());
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  const int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);
  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kCdna5GpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  kfd_queue_snapshot_entry snapshot_entry{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&snapshot_entry);
  snapshot.queue_snapshot.num_queues = 1;
  snapshot.queue_snapshot.entry_size = sizeof(snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  ASSERT_NE(cu, nullptr);
  auto *wave = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0x600000000ULL, /*sgprs=*/16, kNumVgprs);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(driver_->local_process_id());
  wave->set_queue_id(create.queue_id);
  wave->set_dispatch_id(7);
  wave->set_status_raw(kStatus);
  wave->set_mode_raw(kMode);
  wave->set_trapsts(kTrapsts);
  wave->set_gfx12_trap_ctrl_raw(kTrapCtrl);
  wave->set_gfx1250_xnack_state_priv_raw(kXnackStatePriv);
  wave->set_gfx1250_xnack_mask_raw(kXnackMask);
  wave->set_debug_halted(true);

  uint32_t queue_id = create.queue_id;
  kfd_ioctl_dbg_trap_args control{};
  control.pid = static_cast<uint32_t>(getpid());
  control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.suspend_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);

  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  std::vector<rocjitsu::kmd::CwsrWaveState> states(1);
  states[0].num_sgprs = 16;
  states[0].num_vgprs = kNumVgprs;
  ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
      kCwsrAddress, kCwsrSize, states,
      [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
      ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_EQ(states[0].state_priv & ((1u << 19) | (1u << 20)), (1u << 19) | (1u << 20));
  EXPECT_EQ(states[0].mode, kMode);
  EXPECT_EQ(states[0].trap_ctrl, kTrapCtrl);
  EXPECT_EQ(states[0].xnack_state_priv, kXnackStatePriv);
  EXPECT_EQ(states[0].xnack_mask, kXnackMask);

  states[0].wave_stopped = false;
  states[0].xnack_state_priv = kEditedXnackStatePriv;
  ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                  kCwsrAddress, kCwsrSize, states,
                  [&](uint64_t address, uint32_t value) {
                    memory->write32(address, value, driver_->local_process_id());
                  },
                  ROCJITSU_CODE_ARCH_CDNA5)
                  .ok);
  control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.resume_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);

  EXPECT_EQ(wave->status_raw(), kStatus);
  EXPECT_EQ(wave->mode_raw(), kMode);
  EXPECT_EQ(wave->trapsts(), kTrapsts);
  EXPECT_EQ(wave->gfx12_trap_ctrl_raw(), kTrapCtrl);
  EXPECT_EQ(wave->gfx1250_xnack_state_priv_raw(), kEditedXnackStatePriv);
  EXPECT_EQ(wave->gfx1250_xnack_mask_raw(), kXnackMask);
  EXPECT_FALSE(wave->debug_halted());
}

TEST_F(KfdIoctlCdna5Test, ScratchScoreboardSlotsRestartOnEachShaderEngine) {
  auto *xcd = soc_->xcd(0);
  ASSERT_NE(xcd, nullptr);
  ASSERT_EQ(xcd->num_shader_engines(), 2u);
  auto *se0 = xcd->shader_engine(0);
  auto *se1 = xcd->shader_engine(1);
  ASSERT_EQ(se0->num_compute_units(), 16u);
  ASSERT_EQ(se1->num_compute_units(), 16u);

  auto *se0_last_cu = se0->compute_unit(15);
  auto *se1_first_cu = se1->compute_unit(0);
  auto *se0_last_wave = se0_last_cu->dispatch_wf_at(
      /*wf_id=*/63, /*wg_id=*/0, /*pc=*/0x600000000ULL, /*sgprs=*/16, /*vgprs=*/4);
  auto *se1_first_wave = se1_first_cu->dispatch_wf_at(
      /*wf_id=*/0, /*wg_id=*/1, /*pc=*/0x600001000ULL, /*sgprs=*/16, /*vgprs=*/4);
  ASSERT_NE(se0_last_wave, nullptr);
  ASSERT_NE(se1_first_wave, nullptr);

  EXPECT_EQ(se0_last_wave->shader_engine_id(), 0u);
  EXPECT_EQ(se0_last_wave->scratch_scoreboard_id(), 1023u);
  EXPECT_EQ(se1_first_wave->shader_engine_id(), 1u);
  EXPECT_EQ(se1_first_wave->scratch_scoreboard_id(), 0u);
  EXPECT_EQ(se1_first_wave->shader_engine_id() * 1024u + se1_first_wave->scratch_scoreboard_id(),
            1024u);
}

TEST_F(KfdIoctlCdna5Test, DbgTrapPublishesAndRestoresSecondQueueAcrossActiveXccAreas) {
  constexpr uint64_t kQueue0CwsrAddress = 0x610000000ULL;
  constexpr uint64_t kQueue1CwsrAddress = 0x620000000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;
  const uint32_t xcc_count = soc_->num_xcds();
  ASSERT_GT(xcc_count, 1u);

  std::vector<uint8_t> queue0_cwsr(static_cast<size_t>(kCwsrSize) * xcc_count);
  std::vector<uint8_t> queue1_cwsr(static_cast<size_t>(kCwsrSize) * xcc_count);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kQueue0CwsrAddress, queue0_cwsr.data(), queue0_cwsr.size());
  process->map_pages(kQueue1CwsrAddress, queue1_cwsr.data(), queue1_cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::array<std::vector<uint8_t>, 2> rings{std::vector<uint8_t>(4096), std::vector<uint8_t>(4096)};
  std::array<uint64_t, 2> read_pointers{};
  std::array<uint64_t, 2> write_pointers{};
  std::array<kfd_ioctl_create_queue_args, 2> queues{};
  for (size_t index = 0; index < queues.size(); ++index) {
    auto &queue = queues[index];
    queue.gpu_id = kCdna5GpuId;
    queue.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    queue.ring_base_address = reinterpret_cast<uint64_t>(rings[index].data());
    queue.ring_size = static_cast<uint32_t>(rings[index].size());
    queue.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointers[index]);
    queue.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointers[index]);
    queue.ctx_save_restore_address = index == 0 ? kQueue0CwsrAddress : kQueue1CwsrAddress;
    queue.ctx_save_restore_size = kCwsrSize;
    ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &queue), 0);
  }

  std::array<kfd_queue_snapshot_entry, 2> snapshots{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(snapshots.data());
  snapshot.queue_snapshot.num_queues = snapshots.size();
  snapshot.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  auto *xcc0 = soc_->xcd(0);
  auto *xcc1 = soc_->xcd(1);
  ASSERT_NE(xcc0, nullptr);
  ASSERT_NE(xcc1, nullptr);
  std::array<rocjitsu::amdgpu::Wavefront *, 2> waves{};
  for (uint32_t xcc_id = 0; xcc_id < waves.size(); ++xcc_id) {
    auto *cp = soc_->xcd(xcc_id)->command_processor();
    ASSERT_NE(cp, nullptr);
    ASSERT_FALSE(cp->compute_units().empty());
    waves[xcc_id] = cp->compute_units().front()->dispatch_wf(
        /*wg_id=*/xcc_id, /*pc=*/0x600000000ULL + xcc_id * 0x1000, /*sgprs=*/16, /*vgprs=*/4);
    ASSERT_NE(waves[xcc_id], nullptr);
    waves[xcc_id]->set_process_id(driver_->local_process_id());
    waves[xcc_id]->set_queue_id(queues[1].queue_id);
    waves[xcc_id]->set_dispatch_id(7);
    waves[xcc_id]->set_debug_wave_id(100 + xcc_id);
    waves[xcc_id]->set_wg_coord(xcc_id, 0, 0);
    waves[xcc_id]->set_debug_halted(true);
  }

  uint32_t queue_id = queues[1].queue_id;
  kfd_ioctl_dbg_trap_args control{};
  control.pid = static_cast<uint32_t>(getpid());
  control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.suspend_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);

  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  const uint64_t xcc1_address = kQueue1CwsrAddress + kCwsrSize;
  EXPECT_EQ(memory->read32(kQueue1CwsrAddress, driver_->local_process_id()), 0x100u);
  EXPECT_EQ(memory->read32(xcc1_address, driver_->local_process_id()), 0x100u);

  for (uint32_t xcc_id = 0; xcc_id < waves.size(); ++xcc_id) {
    const uint64_t area_address = kQueue1CwsrAddress + static_cast<uint64_t>(xcc_id) * kCwsrSize;
    std::vector<rocjitsu::kmd::CwsrWaveState> states(1);
    states[0].num_sgprs = 16;
    states[0].num_vgprs = 4;
    ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
        area_address, kCwsrSize, states,
        [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
        ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_EQ(states[0].wave_id, 100u + xcc_id);
    EXPECT_TRUE(states[0].wave_stopped);

    states[0].wave_stopped = false;
    ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                    area_address, kCwsrSize, states,
                    [&](uint64_t address, uint32_t value) {
                      memory->write32(address, value, driver_->local_process_id());
                    },
                    ROCJITSU_CODE_ARCH_CDNA5)
                    .ok);
  }
  control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.resume_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
  for (const auto *wave : waves)
    EXPECT_FALSE(wave->debug_halted());
}

// rocm-dbgapi enumerates the target's compute queues to locate each queue's
// CWSR area (from which it walks wave save state). GET_QUEUE_SNAPSHOT must
// report the ctx_save_restore address/size captured at CREATE_QUEUE.
TEST_F(KfdIoctlTest, DbgTrapQueueSnapshotEnumeratesQueues) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // No queues yet: the count call reports zero.
  kfd_ioctl_dbg_trap_args count0{};
  count0.pid = static_cast<uint32_t>(getpid());
  count0.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  count0.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count0), 0);
  EXPECT_EQ(count0.queue_snapshot.num_queues, 0u);

  kfd_ioctl_dbg_trap_args zero_size{};
  zero_size.pid = static_cast<uint32_t>(getpid());
  zero_size.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &zero_size), -EINVAL);

  // Create two compute queues with distinct geometry.
  std::vector<uint8_t> ring1(4096, 0), ring2(8192, 0), rw(4096, 0);
  constexpr uint64_t kCwsrVa = 0x123400000ULL;
  constexpr uint32_t kCwsrSize = 0x8000;
  kfd_ioctl_create_queue_args q1{};
  q1.gpu_id = kGpuId;
  q1.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  q1.ring_base_address = reinterpret_cast<uint64_t>(ring1.data());
  q1.ring_size = static_cast<uint32_t>(ring1.size());
  q1.read_pointer_address = reinterpret_cast<uint64_t>(rw.data());
  q1.write_pointer_address = reinterpret_cast<uint64_t>(rw.data() + 64);
  q1.ctx_save_restore_address = kCwsrVa;
  q1.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &q1), 0);

  kfd_ioctl_create_queue_args q2 = q1;
  q2.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
  q2.ring_base_address = reinterpret_cast<uint64_t>(ring2.data());
  q2.ring_size = static_cast<uint32_t>(ring2.size());
  q2.read_pointer_address = reinterpret_cast<uint64_t>(rw.data() + 128);
  q2.write_pointer_address = reinterpret_cast<uint64_t>(rw.data() + 192);
  q2.ctx_save_restore_address += kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &q2), 0);

  // Count reports the total, independent of caller capacity.
  kfd_ioctl_dbg_trap_args count1{};
  count1.pid = static_cast<uint32_t>(getpid());
  count1.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  count1.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count1), 0);
  ASSERT_EQ(count1.queue_snapshot.num_queues, 2u);
  EXPECT_EQ(count1.queue_snapshot.entry_size, sizeof(kfd_queue_snapshot_entry));

  // A larger input stride is preserved for addressing while entry_size(OUT) is
  // clamped. Only one entry is written and the stride tail remains untouched.
  constexpr size_t kStride = sizeof(kfd_queue_snapshot_entry) + 16;
  std::array<uint8_t, kStride> partial_buf;
  partial_buf.fill(0xA5);
  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = static_cast<uint32_t>(getpid());
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.entry_size = kStride;
  snap.queue_snapshot.num_queues = 1;
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(partial_buf.data());
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_EQ(snap.queue_snapshot.num_queues, 2u);
  EXPECT_EQ(snap.queue_snapshot.entry_size, sizeof(kfd_queue_snapshot_entry));
  const auto *partial = reinterpret_cast<const kfd_queue_snapshot_entry *>(partial_buf.data());
  EXPECT_EQ(partial->queue_id, q1.queue_id);
  EXPECT_TRUE(std::all_of(partial_buf.begin() + sizeof(*partial), partial_buf.end(),
                          [](uint8_t byte) { return byte == 0xA5; }));

  // UPDATE_QUEUE changes the live ring geometry reported by the kernel ABI.
  kfd_ioctl_update_queue_args update{};
  update.queue_id = q1.queue_id;
  update.ring_base_address = reinterpret_cast<uint64_t>(ring2.data());
  update.ring_size = static_cast<uint32_t>(ring2.size());
  update.queue_percentage = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_UPDATE_QUEUE, &update), 0);
  uint32_t suspended_queues = 0;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    suspended_queues +=
        cp->queue_runtime_suspended_for_test(q1.queue_id, driver_->local_process_id());
  });
  // A compute queue is replicated onto every XCD for dispatch fan-out, and the
  // suspension must reach every copy: a replica holds shards of its own, which
  // the CP would otherwise keep draining while the runtime believes the queue is
  // stopped. UPDATE_QUEUE already broadcasts to every command processor, so the
  // count is one per XCD holding the queue rather than one for the owner.
  EXPECT_EQ(suspended_queues, soc_->num_xcds());

  update.queue_percentage = 100;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_UPDATE_QUEUE, &update), 0);
  suspended_queues = 0;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    suspended_queues +=
        cp->queue_runtime_suspended_for_test(q1.queue_id, driver_->local_process_id());
  });
  EXPECT_EQ(suspended_queues, 0u);

  std::array<kfd_queue_snapshot_entry, 2> entries{};
  snap.queue_snapshot.entry_size = sizeof(entries[0]);
  snap.queue_snapshot.num_queues = entries.size();
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(entries.data());
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  const auto &entry = entries[0];
  EXPECT_EQ(entry.queue_id, q1.queue_id);
  EXPECT_EQ(entry.gpu_id, kGpuId);
  EXPECT_EQ(entry.ctx_save_restore_address, kCwsrVa);
  EXPECT_EQ(entry.ctx_save_restore_area_size, kCwsrSize);
  EXPECT_EQ(entry.ring_base_address, reinterpret_cast<uint64_t>(ring2.data()));
  EXPECT_EQ(entry.ring_size, ring2.size());
  EXPECT_EQ(entry.read_pointer_address, q1.read_pointer_address);
  EXPECT_EQ(entry.write_pointer_address, q1.write_pointer_address);
  EXPECT_EQ(entry.queue_type, static_cast<uint32_t>(KFD_IOC_QUEUE_TYPE_COMPUTE_AQL));
  EXPECT_EQ(entry.exception_status, KFD_EC_MASK(EC_QUEUE_NEW));
  EXPECT_EQ(entry.reserved, 0u);
  EXPECT_EQ(entries[1].queue_id, q2.queue_id);
  EXPECT_EQ(entries[1].queue_type, static_cast<uint32_t>(KFD_IOC_QUEUE_TYPE_COMPUTE));

  snap.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_EQ(entries[0].exception_status, KFD_EC_MASK(EC_QUEUE_NEW));
  snap.queue_snapshot.exception_mask = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_EQ(entries[0].exception_status, 0u);
  EXPECT_EQ(entries[1].exception_status, 0u);

  kfd_ioctl_dbg_trap_args null_buffer = snap;
  null_buffer.queue_snapshot.snapshot_buf_ptr = 0;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &null_buffer), -EFAULT);
  EXPECT_EQ(null_buffer.queue_snapshot.num_queues, 2u);

  rocjitsu::amdgpu::Wavefront *halted_wave = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (halted_wave != nullptr || cp->compute_units().empty())
      return;
    halted_wave = cp->compute_units().front()->dispatch_wf(/*wg_id=*/0, /*pc=*/0x600000000ULL,
                                                           /*sgprs=*/16, /*vgprs=*/4);
  });
  ASSERT_NE(halted_wave, nullptr);
  halted_wave->set_process_id(driver_->local_process_id());
  halted_wave->set_queue_id(q1.queue_id);
  halted_wave->set_debug_halted(true);

  // Destroying queues reclaims resident waves and removes their metadata
  // without disturbing creation order.
  kfd_ioctl_destroy_queue_args dq{};
  dq.queue_id = q1.queue_id;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &dq), 0);
  EXPECT_TRUE(halted_wave->is_halted());
  kfd_ioctl_dbg_trap_args count2{};
  count2.pid = static_cast<uint32_t>(getpid());
  count2.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  count2.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count2), 0);
  EXPECT_EQ(count2.queue_snapshot.num_queues, 1u);
  dq.queue_id = q2.queue_id;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &dq), 0);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count2), 0);
  EXPECT_EQ(count2.queue_snapshot.num_queues, 0u);
}

TEST_F(KfdIoctlTest, DbgTrapRealWaveTrapReportsWhilePeerRunsBeforeExplicitCwsrSuspend) {
  constexpr uint64_t kKernelAddress = 0x600000000ULL;
  constexpr uint64_t kPeerAddress = 0x600001000ULL;
  constexpr uint64_t kTrapHandlerAddress = 0x600080000ULL;
  constexpr uint64_t kCwsrAddress = 0x600100000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;
  constexpr uint32_t kSTrapBreakpoint = 0xBF920001u;
  constexpr uint32_t kSNop = 0xBF800000u;

  std::vector<uint8_t> code_page(4096);
  std::vector<uint8_t> peer_page(4096);
  std::vector<uint8_t> trap_handler_page(4096);
  std::vector<uint8_t> cwsr(kCwsrSize);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kKernelAddress, code_page.data(), code_page.size());
  process->map_pages(kPeerAddress, peer_page.data(), peer_page.size());
  process->map_pages(kTrapHandlerAddress, trap_handler_page.data(), trap_handler_page.size());
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  kfd_ioctl_set_trap_handler_args set_handler{};
  set_handler.gpu_id = kGpuId;
  set_handler.tba_addr = kTrapHandlerAddress;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_SET_TRAP_HANDLER, &set_handler), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  ASSERT_NE(cu, nullptr);

  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  memory->write32(kKernelAddress, kSTrapBreakpoint, driver_->local_process_id());
  for (uint32_t offset = 0; offset < peer_page.size(); offset += sizeof(uint32_t))
    memory->write32(kPeerAddress + offset, kSNop, driver_->local_process_id());
  const uint32_t trap_handler[] = {
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEF800FFu, 0x00002000u, // s_mov_b32 ttmp12, STATUS.HALT
      0xBF900001u,              // s_sendmsg sendmsg(MSG_INTERRUPT)
      0xB978F802u,              // s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(trap_handler); ++i)
    memory->write32(kTrapHandlerAddress + i * 4, trap_handler[i], driver_->local_process_id());
  auto *wave = cu->dispatch_wf(/*wg_id=*/0, kKernelAddress, /*sgprs=*/16, /*vgprs=*/4);
  auto *peer = cu->dispatch_wf(/*wg_id=*/1, kPeerAddress, /*sgprs=*/16, /*vgprs=*/4);
  ASSERT_NE(wave, nullptr);
  ASSERT_NE(peer, nullptr);
  for (auto *resident : {wave, peer}) {
    resident->set_process_id(driver_->local_process_id());
    resident->set_queue_id(create.queue_id);
    resident->set_dispatch_id(7);
  }

  kfd_queue_snapshot_entry snapshot_entry{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&snapshot_entry);
  snapshot.queue_snapshot.num_queues = 1;
  snapshot.queue_snapshot.entry_size = sizeof(snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  for (uint32_t i = 0; i < 8 && !wave->debug_halted(); ++i)
    cu->step();
  ASSERT_TRUE(wave->debug_halted());
  ASSERT_FALSE(peer->debug_halted());
  EXPECT_GT(peer->pc, kPeerAddress);

  uint64_t notifications = 0;
  ASSERT_EQ(::read(notifier, &notifications, sizeof(notifications)),
            static_cast<ssize_t>(sizeof(notifications)))
      << strerror(errno);
  EXPECT_EQ(notifications, 1u);

  kfd_ioctl_dbg_trap_args query{};
  query.pid = static_cast<uint32_t>(getpid());
  query.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
  query.query_debug_event.exception_mask = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &query), 0);
  EXPECT_EQ(query.query_debug_event.queue_id, create.queue_id);
  EXPECT_EQ(query.query_debug_event.gpu_id, kGpuId);
  EXPECT_NE(query.query_debug_event.exception_mask & KFD_EC_MASK(EC_QUEUE_WAVE_TRAP), 0u);

  // Notification alone does not publish queue state. ROCdbgapi explicitly
  // suspends the queue before decoding its authoritative CWSR snapshot.
  EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0u);
  uint32_t queue_id = create.queue_id;
  kfd_ioctl_dbg_trap_args control{};
  control.pid = static_cast<uint32_t>(getpid());
  control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.suspend_queues.num_queues = 1;
  control.suspend_queues.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
  EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0x100u);
  EXPECT_NE(memory->read32(kCwsrAddress + 8, driver_->local_process_id()), 0u);
  EXPECT_NE(memory->read32(kCwsrAddress + 12, driver_->local_process_id()), 0u);
  std::vector<rocjitsu::kmd::CwsrWaveState> states(2);
  for (auto &state : states) {
    state.num_sgprs = 16;
    state.num_vgprs = 4;
  }
  ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
      kCwsrAddress, kCwsrSize, states,
      [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
      ROCJITSU_CODE_ARCH_CDNA4));
  auto stopped = std::find_if(states.begin(), states.end(),
                              [](const auto &state) { return state.wave_stopped; });
  ASSERT_NE(stopped, states.end());
  auto running = std::find_if(states.begin(), states.end(),
                              [](const auto &state) { return !state.wave_stopped; });
  ASSERT_NE(running, states.end());
  EXPECT_FALSE(stopped->saved_status_halt);
  EXPECT_NE(stopped->status & (1u << 13), 0u);
  EXPECT_EQ(running->status & (1u << 13), 0u);
}

// An unfetchable PC is the only wave stop that reaches the debug callbacks
// without going through the decoder, which is what makes it usable on a part
// whose instruction encodings this file does not otherwise touch. The cost is
// that a declined fault here halts the wave and frees its slot, so it cannot
// show whether a declined stop was cleanly undone -- the scalar-load fault
// below is used for that. Returns the resident wave, halted or not.
rocjitsu::amdgpu::Wavefront *fault_a_wave_on_an_unmapped_pc(rocjitsu::SoC *soc,
                                                            rocjitsu::SimulatedKfd *driver,
                                                            uint64_t unmapped_pc,
                                                            uint32_t queue_id) {
  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  if (cu == nullptr)
    return nullptr;
  auto *wave = cu->dispatch_wf(/*wg_id=*/0, unmapped_pc, /*sgprs=*/16, /*vgprs=*/4);
  if (wave == nullptr)
    return nullptr;
  wave->set_process_id(driver->local_process_id());
  wave->set_queue_id(queue_id);
  wave->set_dispatch_id(7);
  for (uint32_t i = 0; i < 4 && !wave->is_halted() && !wave->debug_halted(); ++i)
    cu->step();
  return wave;
}

// A data-side fault instead: s_load_dword through a null sbase. A declined
// fault leaves this wave running -- the access is re-issued at its own PC and
// retires -- which is what makes it able to tell a stop that was rolled back
// from one that was never attempted. gfx9 encodings, so CDNA fixtures only.
rocjitsu::amdgpu::Wavefront *fault_a_wave_on_a_scalar_load(rocjitsu::SoC *soc,
                                                           rocjitsu::SimulatedKfd *driver,
                                                           uint64_t kernel_pc, uint32_t queue_id) {
  auto *memory = soc->memory();
  if (memory == nullptr)
    return nullptr;
  const uint32_t pid = driver->local_process_id();
  memory->write32(kernel_pc, 0xC0020000u, pid);     // s_load_dword s0, s[0:1], 0x0
  memory->write32(kernel_pc + 4, 0u, pid);          // ... its immediate offset
  memory->write32(kernel_pc + 8, 0xBF800000u, pid); // s_nop 0

  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  if (cu == nullptr)
    return nullptr;
  auto *wave = cu->dispatch_wf(/*wg_id=*/0, kernel_pc, /*sgprs=*/16, /*vgprs=*/4);
  if (wave == nullptr)
    return nullptr;
  wave->set_process_id(pid);
  wave->set_queue_id(queue_id);
  wave->set_dispatch_id(7);
  // sbase = s[0:1] = 0, an address nothing is mapped at.
  cu->write_sgpr(wave->sgpr_alloc().base, 0);
  cu->write_sgpr(wave->sgpr_alloc().base + 1, 0);
  cu->step();
  return wave;
}

// The CWSR codec reproduces the gfx9.4 record only, so on any other part a
// published record is one rocm-dbgapi would decode against a layout that does
// not match. Declining the stop has to happen before the wave is touched: a
// wave stopped for a record that never appears parks forever, because the only
// thing that could resume it is a debugger reading that record. The fault
// therefore takes its ordinary undebugged path instead.
TEST_F(KfdIoctlRdna3Test, DbgTrapDeclinesWaveStopsOnAPartWithNoModelledCwsrLayout) {
  constexpr uint64_t kUnmappedAddress = 0x600200000ULL;
  constexpr uint64_t kCwsrAddress = 0x600100000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;

  std::vector<uint8_t> cwsr(kCwsrSize);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  ASSERT_NE(driver_->local_process_id(), 0u);
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kRdna3GpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  auto *wave = fault_a_wave_on_an_unmapped_pc(soc_, driver_, kUnmappedAddress, create.queue_id);
  ASSERT_NE(wave, nullptr);

  // Declined, not deferred: the wave halts the way it would with no debugger
  // attached, and the callback leaves no half-applied stop behind it.
  EXPECT_FALSE(wave->debug_halted());
  EXPECT_TRUE(wave->is_halted());
  EXPECT_EQ(wave->trapsts(), 0u);

  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0u);

  uint64_t notifications = 0;
  EXPECT_EQ(::read(notifier, &notifications, sizeof(notifications)), -1);
  EXPECT_EQ(errno, EAGAIN);
}

// The stop gate above is the last line of defence, not the first: rocdbgapi asks
// the device snapshot whether an agent is debuggable before it attaches to one.
// Withholding the support bit there declines at the capability boundary, with
// every other agent in the process still usable, rather than letting an attach
// succeed and then declining each individual wave stop behind it.
TEST_F(KfdIoctlRdna3Test, DbgTrapDeviceSnapshotWithholdsTrapDebugOnUnmodelledArch) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  kfd_dbg_device_info_entry entry{};
  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = static_cast<uint32_t>(getpid());
  snap.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  snap.device_snapshot.entry_size = sizeof(entry);
  snap.device_snapshot.num_devices = 1;
  snap.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  ASSERT_EQ(snap.device_snapshot.num_devices, 1u);

  ASSERT_FALSE(rocjitsu::kmd::cwsr_layout_modelled_for_gc_ip_version(
      rocjitsu::kmd::gc_ip_version_for_gfx_target_version(entry.gfx_target_version)))
      << "this fixture is supposed to be a part the CWSR codec does not model";
  EXPECT_EQ(entry.capability & HSA_CAP_TRAP_DEBUG_SUPPORT, 0u)
      << "an agent whose CWSR record the simulator cannot produce advertised trap debugging";

  // Only that claim is withheld. The agent is still enumerated, still carries
  // its address-watch description, and is still usable for everything a
  // non-debugging client does with it.
  EXPECT_EQ(entry.gpu_id, kRdna3GpuId);
  EXPECT_NE(entry.capability & HSA_CAP_WATCH_POINTS_SUPPORTED, 0u);
  EXPECT_NE(entry.debug_prop, 0u);
  EXPECT_NE(entry.simd_count, 0u);
}

// rocm-dbgapi reaches the same publication through its own request rather than
// through a wave stop, and that route stops the waves first. Without the same
// gate it would strand every resident wave behind the queue error it reports.
TEST_F(KfdIoctlRdna3Test, DbgTrapSuspendQueuesOnUnmodelledArchErrsWithoutStrandingWaves) {
  constexpr uint64_t kUnmappedAddress = 0x600200000ULL;
  constexpr uint64_t kCwsrAddress = 0x600100000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;

  std::vector<uint8_t> cwsr(kCwsrSize);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kRdna3GpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  ASSERT_NE(cu, nullptr);
  auto *wave = cu->dispatch_wf(/*wg_id=*/0, kUnmappedAddress, /*sgprs=*/16, /*vgprs=*/4);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(driver_->local_process_id());
  wave->set_queue_id(create.queue_id);

  // SUSPEND_QUEUES rejects a queue still carrying EC_QUEUE_NEW as invalid, so
  // acknowledge it the way a debugger does before the request under test.
  kfd_queue_snapshot_entry snapshot_entry{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&snapshot_entry);
  snapshot.queue_snapshot.num_queues = 1;
  snapshot.queue_snapshot.entry_size = sizeof(snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  uint32_t queue_id = create.queue_id;
  kfd_ioctl_dbg_trap_args control{};
  control.pid = static_cast<uint32_t>(getpid());
  control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.suspend_queues.num_queues = 1;
  control.suspend_queues.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 0);
  EXPECT_NE(queue_id & KFD_DBG_QUEUE_ERROR_MASK, 0u);

  // Reporting the error and stranding the waves would be strictly worse than
  // reporting it alone: the queue would never run again and never be resumable.
  EXPECT_FALSE(wave->debug_suspended());
  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0u);
}

// The positive control for the two cases below: on a modelled part with an area
// the queue's waves fit in, the same fault stops the wave, publishes a record
// and wakes the debugger. Without this the rollback cases could pass by never
// reaching the callback at all.
TEST_F(KfdIoctlTest, DbgTrapMemoryViolationStopsTheWaveAndPublishesItsRecord) {
  constexpr uint64_t kKernelAddress = 0x600000000ULL;
  constexpr uint64_t kCwsrAddress = 0x600100000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;

  std::vector<uint8_t> code_page(4096);
  std::vector<uint8_t> cwsr(kCwsrSize);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kKernelAddress, code_page.data(), code_page.size());
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  auto *wave = fault_a_wave_on_a_scalar_load(soc_, driver_, kKernelAddress, create.queue_id);
  ASSERT_NE(wave, nullptr);

  constexpr uint32_t kTrapstsXnackError = 1u << 28;
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_NE(wave->trapsts() & kTrapstsXnackError, 0u);
  // Stopped past the access: a wave the debugger resumes must not re-run it.
  EXPECT_EQ(wave->pc, kKernelAddress + 8);

  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0x100u);

  uint64_t notifications = 0;
  ASSERT_EQ(::read(notifier, &notifications, sizeof(notifications)),
            static_cast<ssize_t>(sizeof(notifications)))
      << strerror(errno);
  EXPECT_EQ(notifications, 1u);
}

// The arch gate cannot be the only check: on a modelled part serialization can
// still fail, and the wave is already stopped by then because the serializer
// selects waves by debug_stopped(). An area too small for the queue's waves is
// the reachable case -- ROCr sizes it from the largest dispatch it expects, and
// nothing validates it at CREATE_QUEUE. The stop has to come back off the wave.
TEST_F(KfdIoctlTest, DbgTrapUnpublishableStopRollsBackInsteadOfStrandingTheWave) {
  constexpr uint64_t kKernelAddress = 0x600000000ULL;
  constexpr uint64_t kCwsrAddress = 0x600100000ULL;
  // Enough to map, nowhere near enough for a 16-SGPR, 4-VGPR wave plus header,
  // control stack and LDS.
  constexpr uint32_t kCwsrSize = 0x100;
  constexpr uint32_t kTrapstsXnackError = 1u << 28;

  std::vector<uint8_t> code_page(4096);
  std::vector<uint8_t> cwsr(4096);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kKernelAddress, code_page.data(), code_page.size());
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  auto *wave = fault_a_wave_on_a_scalar_load(soc_, driver_, kKernelAddress, create.queue_id);
  ASSERT_NE(wave, nullptr);

  // Every field the callback set on the way to the failed publication is back
  // where it was, so the wave keeps running exactly as it would with nobody
  // debugging it. Leaving debug_halted set would park it with no record for a
  // debugger to resume from; leaving TRAPSTS.XNACK_ERROR behind would make the
  // next stop report a memory violation that never happened.
  EXPECT_FALSE(wave->debug_halted());
  EXPECT_FALSE(wave->is_halted());
  EXPECT_EQ(wave->trapsts() & kTrapstsXnackError, 0u);
  EXPECT_EQ(wave->pc, kKernelAddress + 8);

  uint64_t notifications = 0;
  EXPECT_EQ(::read(notifier, &notifications, sizeof(notifications)), -1);
  EXPECT_EQ(errno, EAGAIN);
}

// A wave parked between MSG_INTERRUPT and s_rfe is still executing the ROCr
// trap handler, under the handler's own EXEC rather than the application's.
// The CWSR record has to carry the interrupted mask -- publishing the handler's
// makes every lane read as active, which inverts `lane apply -active` and
// `-inactive` (gdb.rocm/lane-info.exp). The record is also the resume payload,
// so the reverse shadow matters just as much: an EXEC the debugger edits has to
// reach the wave through the handler's restore instead of overwriting the mask
// the unfinished handler is still running under.
TEST_F(KfdIoctlTest, DbgTrapCwsrShadowsTrapHandlerRegistersAndRoutesDebuggerEdits) {
  constexpr uint64_t kKernelAddress = 0x600400000ULL;
  constexpr uint64_t kTrapHandlerAddress = 0x600480000ULL;
  constexpr uint64_t kCwsrAddress = 0x600500000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;
  constexpr uint32_t kSTrapBreakpoint = 0xBF920001u;
  constexpr uint32_t kSEndpgm = 0xBF810000u;
  // Lanes 0, 2 and 4 of a five-lane wave: the shape gdb.rocm/lane-info.exp
  // produces, where half the lanes have converged out of the divergent branch.
  constexpr uint64_t kInterruptedExec = 0x15ULL;
  constexpr uint64_t kHandlerExec = 0xFFFFFFFFULL;
  constexpr uint64_t kDebuggerExec = 0x1FULL;
  const auto pid = static_cast<uint32_t>(getpid());

  std::vector<uint8_t> code_page(4096);
  std::vector<uint8_t> trap_handler_page(4096);
  std::vector<uint8_t> cwsr(kCwsrSize);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kKernelAddress, code_page.data(), code_page.size());
  process->map_pages(kTrapHandlerAddress, trap_handler_page.data(), trap_handler_page.size());
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  const int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = pid;
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  kfd_ioctl_set_trap_handler_args set_handler{};
  set_handler.gpu_id = kGpuId;
  set_handler.tba_addr = kTrapHandlerAddress;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_SET_TRAP_HANDLER, &set_handler), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  ASSERT_NE(cu, nullptr);
  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);

  memory->write32(kKernelAddress, kSTrapBreakpoint, driver_->local_process_id());
  memory->write32(kKernelAddress + 4, kSEndpgm, driver_->local_process_id());
  // The clobber stands in for the doorbell exchange the real ROCr handler runs
  // through EXEC_LO, and lands before the halt so the snapshot is taken with
  // the handler's mask live.
  const uint32_t trap_handler[] = {
      0x806C846Cu,              // s_add_u32 ttmp0, ttmp0, 4
      0x826D806Du,              // s_addc_u32 ttmp1, ttmp1, 0
      0xBEFE00FFu, 0xFFFFFFFFu, // s_mov_b32 exec_lo, 0xffffffff
      0xBEF800FFu, 0x00002000u, // s_mov_b32 ttmp12, STATUS.HALT
      0xBF900001u,              // s_sendmsg sendmsg(MSG_INTERRUPT)
      0xB978F802u,              // s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
      0xBE801F6Cu,              // s_rfe_b64 ttmp[0:1]
  };
  for (uint32_t i = 0; i < std::size(trap_handler); ++i)
    memory->write32(kTrapHandlerAddress + i * 4, trap_handler[i], driver_->local_process_id());

  auto *wave = cu->dispatch_wf(/*wg_id=*/0, kKernelAddress, /*sgprs=*/16, /*vgprs=*/4);
  ASSERT_NE(wave, nullptr);
  wave->set_process_id(driver_->local_process_id());
  wave->set_queue_id(create.queue_id);
  wave->set_dispatch_id(7);
  wave->set_exec(kInterruptedExec);

  // Consume EC_QUEUE_NEW the way rocdbgapi does at attach; until it is cleared
  // the queue reads as invalid and cannot be suspended.
  kfd_queue_snapshot_entry snapshot_entry{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = pid;
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&snapshot_entry);
  snapshot.queue_snapshot.num_queues = 1;
  snapshot.queue_snapshot.entry_size = sizeof(snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  // Stop between MSG_INTERRUPT and s_rfe. s_rfe is what clears in_trap_handler
  // and reinstalls trap_saved_exec_, so this is the window where the live mask
  // is the handler's and the debugger can still be handed a snapshot -- the
  // queue is suspended below while the wave sits here.
  for (uint32_t i = 0; i < 10 && !wave->trap_interrupt_sent(); ++i)
    cu->step();
  ASSERT_TRUE(wave->trap_interrupt_sent());
  ASSERT_TRUE(wave->in_trap_handler());
  ASSERT_EQ(wave->exec(), kHandlerExec);

  uint32_t queue_id = create.queue_id;
  kfd_ioctl_dbg_trap_args control{};
  control.pid = pid;
  control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.suspend_queues.num_queues = 1;
  control.suspend_queues.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);

  std::vector<rocjitsu::kmd::CwsrWaveState> states(1);
  states[0].num_sgprs = 16;
  states[0].num_vgprs = 4;
  ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
      kCwsrAddress, kCwsrSize, states,
      [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
      ROCJITSU_CODE_ARCH_CDNA4));
  // The debugger sees the application's lanes, and the live register is
  // untouched by having published them.
  EXPECT_EQ(states[0].exec, kInterruptedExec);
  EXPECT_EQ(wave->exec(), kHandlerExec);

  states[0].exec = kDebuggerExec;
  ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                  kCwsrAddress, kCwsrSize, states,
                  [&](uint64_t address, uint32_t value) {
                    memory->write32(address, value, driver_->local_process_id());
                  },
                  ROCJITSU_CODE_ARCH_CDNA4)
                  .ok);
  control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.resume_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);

  // The edit went to the shadow, so the handler is still running under its own
  // mask and only installs the debugger's on s_rfe.
  EXPECT_EQ(wave->exec(), kHandlerExec);

  // STATUS is the same contract and the costlier half. Step the s_setreg that
  // raises STATUS.HALT -- the handler announcing it wants the wave kept stopped
  // -- and take another snapshot before s_rfe consumes it.
  constexpr uint32_t kStatusHalt = 1u << 13;
  cu->step();
  ASSERT_TRUE(wave->in_trap_handler());
  ASSERT_NE(wave->status_raw() & kStatusHalt, 0u);

  control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
  control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.suspend_queues.num_queues = 1;
  control.suspend_queues.exception_mask = 0;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
  ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
      kCwsrAddress, kCwsrSize, states,
      [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
      ROCJITSU_CODE_ARCH_CDNA4));
  // The wave has not stopped for the debugger yet -- the handler has only asked
  // -- so the record describes it as running, exactly as it did before.
  EXPECT_FALSE(states[0].wave_stopped);
  ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                  kCwsrAddress, kCwsrSize, states,
                  [&](uint64_t address, uint32_t value) {
                    memory->write32(address, value, driver_->local_process_id());
                  },
                  ROCJITSU_CODE_ARCH_CDNA4)
                  .ok);
  control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
  control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
  control.resume_queues.num_queues = 1;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
  // Applying that record must not take the handler's HALT away. It used to, and
  // s_rfe then resumed a wave the debugger was expecting to catch.
  EXPECT_NE(wave->status_raw() & kStatusHalt, 0u);

  for (uint32_t i = 0; i < 4 && wave->in_trap_handler(); ++i)
    cu->step();
  EXPECT_FALSE(wave->in_trap_handler());
  EXPECT_TRUE(wave->debug_halted());
  EXPECT_EQ(wave->exec(), kDebuggerExec);
}

TEST_F(KfdIoctlTest, DbgTrapSingleStepReportsWhilePeerWaveRuns) {
  constexpr uint64_t kSteppingAddress = 0x600200000ULL;
  constexpr uint64_t kPeerAddress = 0x600201000ULL;
  constexpr uint64_t kCwsrAddress = 0x600300000ULL;
  constexpr uint32_t kCwsrSize = 0x40000;
  constexpr uint32_t kSNop = 0xBF800000u;
  constexpr uint32_t kModeDebugEn = 1u << 11;
  constexpr uint32_t kTrapAfterInst = 1u << 25;
  constexpr uint32_t kStepCount = 5;
  constexpr uint64_t kSteppingWaveId = 0x123456789ABCDEF0ULL;

  std::vector<uint8_t> stepping_code(4096);
  std::vector<uint8_t> peer_code(4096);
  std::vector<uint8_t> cwsr(kCwsrSize);
  auto process = driver_->find_process(driver_->local_process_id());
  ASSERT_NE(process, nullptr);
  process->map_pages(kSteppingAddress, stepping_code.data(), stepping_code.size());
  process->map_pages(kPeerAddress, peer_code.data(), peer_code.size());
  process->map_pages(kCwsrAddress, cwsr.data(), cwsr.size());

  kfd_ioctl_runtime_enable_args runtime{};
  runtime.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &runtime), 0);

  const int notifier = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  ASSERT_GE(notifier, 0);
  debug_fds_.push_back(notifier);

  kfd_ioctl_dbg_trap_args enable{};
  enable.pid = static_cast<uint32_t>(getpid());
  enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  enable.enable.dbg_fd = notifier;
  enable.enable.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &enable), 0);

  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  create.ctx_save_restore_address = kCwsrAddress;
  create.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  kfd_queue_snapshot_entry snapshot_entry{};
  kfd_ioctl_dbg_trap_args snapshot{};
  snapshot.pid = static_cast<uint32_t>(getpid());
  snapshot.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snapshot.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);
  snapshot.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&snapshot_entry);
  snapshot.queue_snapshot.num_queues = 1;
  snapshot.queue_snapshot.entry_size = sizeof(snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snapshot), 0);

  rocjitsu::amdgpu::ComputeUnitCore *cu = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (cu == nullptr && !cp->compute_units().empty())
      cu = cp->compute_units().front();
  });
  ASSERT_NE(cu, nullptr);

  auto *memory = soc_->memory();
  ASSERT_NE(memory, nullptr);
  for (uint32_t step = 0; step < kStepCount; ++step) {
    memory->write32(kSteppingAddress + step * sizeof(uint32_t), kSNop, driver_->local_process_id());
    memory->write32(kPeerAddress + step * sizeof(uint32_t), kSNop, driver_->local_process_id());
  }

  auto *stepping = cu->dispatch_wf(/*wg_id=*/0, kSteppingAddress, /*sgprs=*/16, /*vgprs=*/4);
  auto *peer = cu->dispatch_wf(/*wg_id=*/1, kPeerAddress, /*sgprs=*/16, /*vgprs=*/4);
  ASSERT_NE(stepping, nullptr);
  ASSERT_NE(peer, nullptr);
  for (auto *wave : {stepping, peer}) {
    wave->set_process_id(driver_->local_process_id());
    wave->set_queue_id(create.queue_id);
    wave->set_dispatch_id(7);
  }
  stepping->set_debug_single_step(true);

  for (uint32_t step = 0; step < kStepCount; ++step) {
    cu->step();

    EXPECT_TRUE(stepping->debug_halted());
    EXPECT_EQ(stepping->pc, kSteppingAddress + (step + 1) * sizeof(uint32_t));
    EXPECT_FALSE(peer->debug_halted());
    EXPECT_EQ(peer->pc, kPeerAddress + (step + 1) * sizeof(uint32_t));

    uint64_t notifications = 0;
    ASSERT_EQ(::read(notifier, &notifications, sizeof(notifications)),
              static_cast<ssize_t>(sizeof(notifications)))
        << strerror(errno);
    EXPECT_EQ(notifications, 1u);

    kfd_ioctl_dbg_trap_args query{};
    query.pid = static_cast<uint32_t>(getpid());
    query.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
    query.query_debug_event.exception_mask = 0;
    ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &query), 0);
    EXPECT_EQ(query.query_debug_event.queue_id, create.queue_id);
    EXPECT_NE(query.query_debug_event.exception_mask & KFD_EC_MASK(EC_QUEUE_WAVE_TRAP), 0u);

    // A single-step trap signals an interrupt; it does not implicitly save the
    // queue. The debugger's explicit suspension below is the publication point.
    EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0u);

    uint32_t queue_id = create.queue_id;
    kfd_ioctl_dbg_trap_args control{};
    control.pid = static_cast<uint32_t>(getpid());
    control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
    control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
    control.suspend_queues.num_queues = 1;
    control.suspend_queues.exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
    ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
    EXPECT_EQ(memory->read32(kCwsrAddress, driver_->local_process_id()), 0x100u);

    // Model rocdbgapi selecting this wave for another scheduler-locked step.
    // Its CWSR writes are consumed by RESUME_QUEUES each cycle.
    std::vector<rocjitsu::kmd::CwsrWaveState> states(2);
    for (auto &state : states) {
      state.num_sgprs = 16;
      state.num_vgprs = 4;
    }
    ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
        kCwsrAddress, kCwsrSize, states,
        [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
        ROCJITSU_CODE_ARCH_CDNA4));
    auto selected = std::find_if(states.begin(), states.end(), [&](const auto &state) {
      return step == 0 ? state.group_ids[0] == 0 && state.wave_in_group == 0
                       : state.wave_id == kSteppingWaveId;
    });
    ASSERT_NE(selected, states.end());
    EXPECT_EQ(selected->pc, kSteppingAddress + (step + 1) * sizeof(uint32_t));
    selected->wave_id = kSteppingWaveId;
    if (step != 0) {
      selected->queue_packet_id = 100 + step;
      selected->group_ids = {200 + step, 300 + step, 400 + step};
      selected->wave_in_group = 10 + step;
    }
    selected->wave_stopped = false;
    selected->mode |= kModeDebugEn;
    selected->trapsts &= ~kTrapAfterInst;
    ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                    kCwsrAddress, kCwsrSize, states,
                    [&](uint64_t address, uint32_t value) {
                      memory->write32(address, value, driver_->local_process_id());
                    },
                    ROCJITSU_CODE_ARCH_CDNA4)
                    .ok);
    control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
    control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
    control.resume_queues.num_queues = 1;
    ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
    EXPECT_EQ(stepping->debug_wave_id(), kSteppingWaveId);
    if (step != 0) {
      EXPECT_EQ(stepping->aql_packet_id(), 100 + step);
    }
    EXPECT_TRUE(stepping->debug_single_step());
    EXPECT_FALSE(stepping->debug_halted());
    EXPECT_FALSE(peer->debug_halted());

    if (step + 1 == kStepCount) {
      control = {};
      control.pid = static_cast<uint32_t>(getpid());
      control.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
      control.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
      control.suspend_queues.num_queues = 1;
      ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
      ASSERT_TRUE(rocjitsu::kmd::deserialize_queue_cwsr(
          kCwsrAddress, kCwsrSize, states,
          [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); },
          ROCJITSU_CODE_ARCH_CDNA4));
      auto stopped_peer = std::find_if(states.begin(), states.end(), [&](const auto &state) {
        return state.wave_id != kSteppingWaveId;
      });
      ASSERT_NE(stopped_peer, states.end());
      stopped_peer->wave_stopped = true;
      stopped_peer->mode |= kModeDebugEn;
      peer->set_fatal_exception_pending(true);
      ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                      kCwsrAddress, kCwsrSize, states,
                      [&](uint64_t address, uint32_t value) {
                        memory->write32(address, value, driver_->local_process_id());
                      },
                      ROCJITSU_CODE_ARCH_CDNA4)
                      .ok);
      control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
      control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
      control.resume_queues.num_queues = 1;
      ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
      EXPECT_TRUE(peer->debug_halted());
      EXPECT_TRUE(peer->debug_suspended());
      EXPECT_FALSE(peer->debug_single_step());
      bool queue_suspended = false;
      soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
        queue_suspended |=
            cp->queue_debug_suspended_for_test(create.queue_id, driver_->local_process_id());
      });
      EXPECT_TRUE(queue_suspended);
    }

    // Clear the header sentinel only, so the next completion proves that
    // notification remains independent from CWSR serialization.
    memory->write32(kCwsrAddress, 0, driver_->local_process_id());
  }
}

TEST_F(KfdIoctlTest, DbgTrapCrossProcessEnableAuthorizedByPtrace) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
    _exit(0);
  }
  ChildProcessGuard child_guard(child);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  uint32_t inferior = daemon.open_process(child);
  ASSERT_NE(debugger, 0u);
  ASSERT_NE(inferior, 0u);

  auto enable_from_debugger = [&]() {
    kfd_ioctl_dbg_trap_args en{};
    en.pid = static_cast<uint32_t>(child);
    en.op = KFD_IOC_DBG_TRAP_ENABLE;
    en.enable.dbg_fd = KFD_INVALID_FD;
    return daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en);
  };

  EXPECT_EQ(enable_from_debugger(), -EPERM);

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSTOPPED(status));

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);
  const int enable_result = daemon_debug_enable(daemon, debugger, en);
  if (enable_result != 0)
    close(dbg_fd);
  ASSERT_EQ(enable_result, 0);

  kfd_ioctl_dbg_trap_args exceptions{};
  exceptions.pid = static_cast<uint32_t>(child);
  exceptions.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  exceptions.set_exceptions_enabled.exception_mask = 0x1234;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &exceptions), 0);

  EXPECT_EQ(ptrace(PTRACE_DETACH, child, nullptr, nullptr), 0);

  daemon.close(debugger);
  daemon.close(inferior);
}

// A session can be enabled before the inferior opens /dev/kfd. Once that
// inferior exits, DISABLE must release the stale session/notifier but still
// return ESRCH, matching the kernel's target-liveness check.
TEST_F(KfdIoctlTest, DbgTrapExitedTargetDisableReturnsESRCHAndReleasesSession) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
    _exit(0);
  }
  ChildProcessGuard child_guard(child);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);
  // Deliberately do NOT open_process(child): the inferior has not connected to
  // /dev/kfd, so it has no KfdProcess.

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSTOPPED(status));

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);
  ASSERT_EQ(daemon_debug_enable(daemon, debugger, en), 0);

  ASSERT_EQ(kill(child, SIGKILL), 0);
  ASSERT_EQ(waitpid(child, &status, 0), child);

  // A pidfd distinguishes this exited target from a future process that reuses
  // its numeric pid. DISABLE reaps the stale session and owned notifier.
  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(child);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), -ESRCH);
  EXPECT_EQ(fcntl(dbg_fd, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);

  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapIdentityChangeDuringAuthorizationReturnsESRCH) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
  }
  ChildProcessGuard child_guard(child);

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSTOPPED(status));

  bool hook_ran = false;
  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true, [&] {
    hook_ran = true;
    ASSERT_EQ(kill(child, SIGKILL), 0);
    ASSERT_EQ(waitpid(child, &status, 0), child);
  });
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);

  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en), -ESRCH);
  EXPECT_TRUE(hook_ran);
  EXPECT_NE(fcntl(dbg_fd, F_GETFD), -1) << "failed ENABLE must not adopt the notifier";
  ::close(dbg_fd);
  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapExitedTargetReturnsESRCHBeforePtraceAuthorization) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
  }
  ChildProcessGuard child_guard(child);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSTOPPED(status));

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);
  ASSERT_EQ(daemon_debug_enable(daemon, debugger, en), 0);

  ASSERT_EQ(kill(child, SIGKILL), 0);
  ASSERT_EQ(waitpid(child, &status, 0), child);

  for (int attempt = 0; attempt < 100 && fcntl(dbg_fd, F_GETFD) != -1; ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(fcntl(dbg_fd, F_GETFD), -1) << "target-exit reaper did not release notifier";
  EXPECT_EQ(errno, EBADF);

  kfd_ioctl_dbg_trap_args op{};
  op.pid = static_cast<uint32_t>(child);
  op.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &op), -ESRCH);
  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapSessionSurvivesTargetKfdConnectionClose) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
  }
  ChildProcessGuard child_guard(child);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  uint32_t inferior = daemon.open_process(child);
  ASSERT_NE(debugger, 0u);
  ASSERT_NE(inferior, 0u);

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);
  ASSERT_EQ(daemon_debug_enable(daemon, debugger, en), 0);

  ASSERT_EQ(daemon.close(inferior), 0);
  kfd_ioctl_dbg_trap_args op{};
  op.pid = static_cast<uint32_t>(child);
  op.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &op), 0);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(child);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), 0);
  EXPECT_EQ(ptrace(PTRACE_DETACH, child, nullptr, nullptr), 0);
  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapDisableAfterTargetKfdCloseAndExitReturnsESRCH) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
  }
  ChildProcessGuard child_guard(child);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  uint32_t inferior = daemon.open_process(child);
  ASSERT_NE(debugger, 0u);
  ASSERT_NE(inferior, 0u);

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);
  ASSERT_EQ(daemon_debug_enable(daemon, debugger, en), 0);

  ASSERT_EQ(daemon.close(inferior), 0);
  ASSERT_EQ(kill(child, SIGKILL), 0);
  ASSERT_EQ(waitpid(child, &status, 0), child);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(child);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), -ESRCH);
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), -ESRCH);
  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapDisableWithoutSessionReportsZombieAsExited) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0)
    _exit(0);

  siginfo_t info{};
  ASSERT_EQ(waitid(P_PID, child, &info, WEXITED | WNOWAIT), 0);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(child);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), -ESRCH);

  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapSessionSurvivesDebuggerKfdConnectionClose) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
  }
  ChildProcessGuard child_guard(child);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);

  ASSERT_EQ(ptrace(PTRACE_ATTACH, child, nullptr, nullptr), 0) << strerror(errno);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);

  int dbg_fd = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dbg_fd, 0);
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = static_cast<uint32_t>(dbg_fd);
  ASSERT_EQ(daemon_debug_enable(daemon, debugger, en), 0);

  ASSERT_EQ(daemon.close(debugger), 0);
  debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);
  kfd_ioctl_dbg_trap_args op{};
  op.pid = static_cast<uint32_t>(child);
  op.op = KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &op), 0);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(child);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), 0);
  EXPECT_EQ(ptrace(PTRACE_DETACH, child, nullptr, nullptr), 0);
  daemon.close(debugger);
}

TEST_F(KfdIoctlTest, DbgTrapDebuggerExitReapsSessionAndAllowsReenable) {
  int debugger_ready[2];
  ASSERT_EQ(pipe2(debugger_ready, O_CLOEXEC), 0);

  pid_t debugger_pid = fork();
  ASSERT_GE(debugger_pid, 0);
  if (debugger_pid == 0) {
    ::close(debugger_ready[0]);
    int target_ready[2];
    if (pipe2(target_ready, O_CLOEXEC) != 0)
      _exit(2);

    pid_t target_pid = fork();
    if (target_pid < 0)
      _exit(3);
    if (target_pid == 0) {
      ::close(target_ready[0]);
      ::close(debugger_ready[1]);
      if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) != 0)
        _exit(4);
      const char ready = 1;
      if (::write(target_ready[1], &ready, sizeof(ready)) != sizeof(ready))
        _exit(5);
      for (;;)
        pause();
    }

    auto fail = [&](int code) {
      kill(target_pid, SIGKILL);
      while (waitpid(target_pid, nullptr, 0) == -1 && errno == EINTR) {
      }
      _exit(code);
    };
    ::close(target_ready[1]);
    char ready = 0;
    if (::read(target_ready[0], &ready, sizeof(ready)) != sizeof(ready))
      fail(6);
    if (ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr) != 0)
      fail(7);
    int status = 0;
    if (waitpid(target_pid, &status, 0) != target_pid || !WIFSTOPPED(status))
      fail(8);
    if (::write(debugger_ready[1], &target_pid, sizeof(target_pid)) != sizeof(target_pid))
      fail(9);
    for (;;)
      pause();
  }
  ChildProcessGuard debugger_guard(debugger_pid);
  ::close(debugger_ready[1]);

  pid_t target_pid = 0;
  ASSERT_EQ(::read(debugger_ready[0], &target_pid, sizeof(target_pid)),
            static_cast<ssize_t>(sizeof(target_pid)));
  ASSERT_GT(target_pid, 0);
  ChildProcessGuard target_guard(target_pid);
  ::close(debugger_ready[0]);

  rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(debugger_pid);
  ASSERT_NE(debugger, 0u);

  int first_notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(first_notifier, 0);
  kfd_ioctl_dbg_trap_args first_enable{};
  first_enable.pid = static_cast<uint32_t>(target_pid);
  first_enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  first_enable.enable.dbg_fd = static_cast<uint32_t>(first_notifier);
  ASSERT_EQ(daemon_debug_enable(daemon, debugger, first_enable), 0);

  // A wave stopped for this debugger. Erasing the session is only half of the
  // release: a wave left halted with its queue gate shut outlives the debugger
  // that stopped it, and nothing else will ever resume it, so the inferior is
  // stranded for good. Set both halves of the stop -- the scheduler's flag and
  // the architectural STATUS.HALT that s_rfe consults -- because clearing only
  // the former re-halts the wave at the handler return.
  const uint32_t target_process = daemon.open_process(target_pid);
  ASSERT_NE(target_process, 0u);
  std::vector<uint8_t> ring(4096);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  kfd_ioctl_create_queue_args create{};
  // Not kGpuId: this daemon is a bare SimulatedKfd over soc_, with no
  // setup_topology() call to rename its single device, so it keeps the
  // constructor's default id. Only self-consistency matters here -- the release
  // path finds the queue's GPU by the same number this creates it under.
  create.gpu_id = 0;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  create.ring_size = static_cast<uint32_t>(ring.size());
  create.read_pointer_address = reinterpret_cast<uint64_t>(&read_pointer);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&write_pointer);
  ASSERT_EQ(daemon.ioctl(target_process, AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  rocjitsu::amdgpu::Wavefront *wave = nullptr;
  rocjitsu::amdgpu::CommandProcessor *wave_cp = nullptr;
  rocjitsu::amdgpu::ComputeUnitCore *wave_cu = nullptr;
  soc_->for_each_cp([&](rocjitsu::amdgpu::CommandProcessor *cp) {
    if (wave != nullptr || cp->compute_units().empty())
      return;
    wave_cp = cp;
    wave_cu = cp->compute_units().front();
    wave = wave_cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0x600000000ULL,
                                /*sgprs=*/16, /*vgprs=*/4);
  });
  ASSERT_NE(wave, nullptr);
  ASSERT_NE(wave_cp, nullptr);
  ASSERT_NE(wave_cu, nullptr);
  // Under the CU's wave-state lock, the same one release_debuggee_state takes:
  // the reaper thread is already running, so every access to a wave's debug
  // state from here on shares its mutex or races it.
  wave_cu->with_wave_state_locked([&] {
    wave->set_process_id(target_process);
    wave->set_queue_id(create.queue_id);
    wave->set_debug_halted(true);
    wave->set_debug_suspended(true);
    wave->set_debug_single_step(true);
    wave->set_status_halt(true);
    wave->set_self_halted(true);
  });
  wave_cp->set_queue_debug_suspended(create.queue_id, target_process, true);

  ASSERT_EQ(kill(debugger_pid, SIGKILL), 0);
  int status = 0;
  ASSERT_EQ(waitpid(debugger_pid, &status, 0), debugger_pid);
  ASSERT_TRUE(WIFSIGNALED(status));
  debugger_guard.release();

  for (int attempt = 0; attempt < 100 && fcntl(first_notifier, F_GETFD) != -1; ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(fcntl(first_notifier, F_GETFD), -1) << "debugger-exit reaper did not release notifier";
  EXPECT_EQ(errno, EBADF);
  ASSERT_EQ(kill(target_pid, 0), 0) << "target exited with its debugger";

  // The reaper runs the same release path an explicit DISABLE does, so the
  // stopped wave is running again and its queue can launch.
  //
  // Poll for the wave rather than reading it once: the closed notifier above is
  // not a happens-before for this. The reaper drops the session -- and with it
  // the fd -- while still holding debug_sessions_mutex_, then releases the
  // debuggee only after unlocking, so the fd can already be gone while the wave
  // is still stopped. Sample under the CU's wave-state lock, which is what the
  // release itself holds while it writes these fields.
  struct WaveStop {
    bool debug_halted = true;
    bool debug_suspended = true;
    bool debug_single_step = true;
    bool status_halt = true;
    bool self_halted = true;
  };
  auto sample = [&] {
    return wave_cu->with_wave_state_locked([&] {
      return WaveStop{wave->debug_halted(), wave->debug_suspended(), wave->debug_single_step(),
                      wave->status_halt(), wave->self_halted()};
    });
  };
  WaveStop stop = sample();
  for (int attempt = 0; attempt < 100 && stop.debug_halted; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop = sample();
  }
  EXPECT_FALSE(stop.debug_halted) << "debugger-exit reaper stranded a stopped wave";
  EXPECT_FALSE(stop.debug_suspended);
  EXPECT_FALSE(stop.debug_single_step);
  EXPECT_FALSE(stop.status_halt)
      << "the architectural halt outlived the session, so the wave re-halts at the handler return";
  EXPECT_FALSE(stop.self_halted);
  EXPECT_FALSE(wave_cp->queue_debug_suspended_for_test(create.queue_id, target_process))
      << "the queue launch gate stayed shut after the debugger died";

  ASSERT_EQ(ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr), 0) << strerror(errno);
  ASSERT_EQ(waitpid(target_pid, &status, 0), target_pid);
  ASSERT_TRUE(WIFSTOPPED(status));

  uint32_t replacement_debugger = daemon.open_process(getpid());
  ASSERT_NE(replacement_debugger, 0u);
  int replacement_notifier = eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(replacement_notifier, 0);
  kfd_ioctl_dbg_trap_args replacement_enable{};
  replacement_enable.pid = static_cast<uint32_t>(target_pid);
  replacement_enable.op = KFD_IOC_DBG_TRAP_ENABLE;
  replacement_enable.enable.dbg_fd = static_cast<uint32_t>(replacement_notifier);
  EXPECT_EQ(daemon_debug_enable(daemon, replacement_debugger, replacement_enable), 0);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(target_pid);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(replacement_debugger, AMDKFD_IOC_DBG_TRAP, &dis), 0);
  ASSERT_EQ(ptrace(PTRACE_KILL, target_pid, nullptr, nullptr), 0);
  ASSERT_EQ(waitpid(target_pid, &status, 0), target_pid);
  target_guard.release();
  daemon.close(debugger);
  daemon.close(replacement_debugger);
  daemon.close(target_process);
}

#if defined(__SANITIZE_THREAD__)
#define RJ_TEST_THREAD_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RJ_TEST_THREAD_SANITIZER 1
#endif
#endif

TEST_F(KfdIoctlTest, DebugSessionReaperShutdownDoesNotHang) {
#ifdef RJ_TEST_THREAD_SANITIZER
  // The child below starts the reaper thread after forking from this
  // multi-threaded parent, which ThreadSanitizer refuses outright ("starting new
  // threads after multi-threaded fork is not supported") and kills the child
  // for. The shutdown this test exists to measure never runs, so the result
  // would say nothing about it either way.
  GTEST_SKIP() << "starting threads after fork is unsupported under ThreadSanitizer";
#else
  pid_t worker = fork();
  ASSERT_GE(worker, 0);
  if (worker == 0) {
    alarm(5);
    for (int iteration = 0; iteration < 250; ++iteration) {
      rocjitsu::SimulatedKfd daemon(*soc_, /*daemon_mode=*/true);
    }
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(worker, &status, 0), worker);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
#endif
}

} // namespace
