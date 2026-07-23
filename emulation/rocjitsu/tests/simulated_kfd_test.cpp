// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulated_kfd_test.cpp
/// @brief Tests for SimulatedKfd creation, open/close, and topology generation.

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/guest_kfd.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include "embedded_schema.h"
#include "simdojo/sim/simulation.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_cdna4.json";

struct TestVM {
  rocjitsu::config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;

  rocjitsu::SimulatedKfd *driver() {
    auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(engine->topology().root());
    return vm ? vm->driver() : nullptr;
  }
};

TestVM create_test_vm(bool daemon_mode = false) {
  TestVM t;
  t.loaded = rocjitsu::config::load_config(CONFIG_PATH.c_str(), rocjitsu::kEmbeddedSchema);
  auto *soc = t.loaded.soc();

  t.loaded.engine_config.max_ticks = 0;
  t.loaded.engine_config.await_primaries = true;
  t.engine = std::make_unique<simdojo::SimulationEngine>(t.loaded.engine_config);

  auto root = t.loaded.take_root();
  auto *released_root = root.release();
  (void)released_root;
  auto vm =
      std::make_unique<rocjitsu::VirtualMachine>(std::unique_ptr<rocjitsu::SoC>(soc), daemon_mode);
  vm->driver()->setup_topology(t.loaded.device, soc->num_xcds());

  t.engine->topology().set_root(std::move(vm));
  t.loaded.wire_links(t.engine->topology());
  soc->wire_backing(t.engine->topology());
  t.engine->create();
  t.engine->register_as_primary();

  return t;
}

struct ChildProcessGuard {
  pid_t pid = -1;
  int done_fd = -1;
  bool waited = false;

  ~ChildProcessGuard() {
    if (pid <= 0 || waited)
      return;
    char done = 0;
    if (done_fd >= 0)
      (void)::write(done_fd, &done, 1);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
  }

  int finish() {
    char done = 0;
    if (done_fd >= 0)
      EXPECT_EQ(::write(done_fd, &done, 1), 1);
    int status = 0;
    EXPECT_EQ(::waitpid(pid, &status, 0), pid);
    waited = true;
    return status;
  }
};

class SimulatedKfdTest : public ::testing::Test {
protected:
  void SetUp() override { setenv("RJ_CONFIG", CONFIG_PATH.c_str(), 1); }
};

TEST_F(SimulatedKfdTest, CreateDefault) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
}

TEST_F(SimulatedKfdTest, OpenAndClose) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  int fd = t.driver()->open();
  EXPECT_GE(fd, 0);

  int ret = t.driver()->close();
  EXPECT_EQ(ret, 0);
}

TEST_F(SimulatedKfdTest, LocalOpenRegistersClientPidForMemoryFallback) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  auto *memory = vm->memory();
  ASSERT_NE(memory, nullptr);

  int fd = t.driver()->open();
  ASSERT_GE(fd, 0);
  uint32_t process_id = t.driver()->local_process_id();
  ASSERT_NE(process_id, 0u);

  uint32_t host_value = 0x12345678u;
  uint64_t host_va = reinterpret_cast<uint64_t>(&host_value);
  EXPECT_EQ(memory->read32(host_va, process_id), host_value);
  EXPECT_NE(memory->resolve_host_ptr(host_va, process_id), nullptr);

  memory->write32(host_va, 0xA5A55A5Au, process_id);
  EXPECT_EQ(host_value, 0xA5A55A5Au);

  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, DaemonVmidMissDoesNotReturnClientIdentityPointer) {
  constexpr uint32_t kClientValue = 0x5A17C0DEu;
  constexpr uint32_t kUpdatedValue = 0x13579BDFu;
  int ready_pipe[2];
  int done_pipe[2];
  ASSERT_EQ(::pipe(ready_pipe), 0);
  ASSERT_EQ(::pipe(done_pipe), 0);

  pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    (void)::close(ready_pipe[0]);
    (void)::close(done_pipe[1]);
    constexpr size_t kPageSize = 0x1000;
    void *page = ::mmap(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                        -1, 0);
    if (page == MAP_FAILED)
      _exit(2);
    *static_cast<uint32_t *>(page) = kClientValue;
    uint64_t client_va = reinterpret_cast<uint64_t>(page);
    if (::write(ready_pipe[1], &client_va, sizeof(client_va)) !=
        static_cast<ssize_t>(sizeof(client_va)))
      _exit(3);
    char done = 0;
    (void)::read(done_pipe[0], &done, 1);
    (void)::munmap(page, kPageSize);
    _exit(0);
  }

  ChildProcessGuard child_guard{child, done_pipe[1]};
  (void)::close(ready_pipe[1]);
  (void)::close(done_pipe[0]);

  uint64_t client_va = 0;
  ASSERT_EQ(::read(ready_pipe[0], &client_va, sizeof(client_va)),
            static_cast<ssize_t>(sizeof(client_va)));
  (void)::close(ready_pipe[0]);

  auto t = create_test_vm(/*daemon_mode=*/true);
  ASSERT_NE(t.driver(), nullptr);
  auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  auto *memory = vm->memory();
  ASSERT_NE(memory, nullptr);

  uint32_t process_id = t.driver()->open_process(child);
  ASSERT_NE(process_id, 0u);

  EXPECT_EQ(memory->resolve_host_ptr(client_va, process_id), nullptr);
  EXPECT_EQ(memory->read32(client_va, process_id), kClientValue);
  memory->write32(client_va, kUpdatedValue, process_id);
  EXPECT_EQ(memory->read32(client_va, process_id), kUpdatedValue);

  EXPECT_EQ(t.driver()->close(process_id), 0);
  int status = child_guard.finish();
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(SimulatedKfdTest, ProcessVmidProtNoneReservationUsesSparseFallback) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *vm = dynamic_cast<rocjitsu::VirtualMachine *>(t.engine->topology().root());
  ASSERT_NE(vm, nullptr);
  auto *memory = vm->memory();
  ASSERT_NE(memory, nullptr);

  int fd = t.driver()->open();
  ASSERT_GE(fd, 0);
  uint32_t process_id = t.driver()->local_process_id();
  ASSERT_NE(process_id, 0u);
  auto process = t.driver()->find_process(process_id);
  ASSERT_NE(process, nullptr);

  constexpr size_t kPageSize = 0x1000;
  void *reservation =
      ::mmap(nullptr, kPageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(reservation, MAP_FAILED);

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(reservation);
  alloc.size = kPageSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);

  uint32_t map_gpu = alloc.gpu_id;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = alloc.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&map_gpu);
  map.n_devices = 1;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
  EXPECT_EQ(map.n_success, 1u);
  EXPECT_TRUE(process->contains_sparse_reservation(alloc.va_addr, alloc.size));

  memory->reset_client_memory_probe_count();
  for (uint32_t i = 0; i < 2048; ++i) {
    uint64_t addr = alloc.va_addr + static_cast<uint64_t>((i % 64) * sizeof(uint32_t));
    uint32_t expected = 0xA5A50000u + i;
    memory->write32(addr, expected, process_id);
    EXPECT_EQ(memory->read32(addr, process_id), expected);
  }
  EXPECT_EQ(memory->client_memory_probe_count(), 0u);

  kfd_ioctl_unmap_memory_from_gpu_args unmap{};
  unmap.handle = alloc.handle;
  unmap.device_ids_array_ptr = reinterpret_cast<uint64_t>(&map_gpu);
  unmap.n_devices = 1;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap), 0);

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(::munmap(reservation, kPageSize), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, UserptrAllocationIsNotClassifiedAsSparseReservation) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  int fd = t.driver()->open();
  ASSERT_GE(fd, 0);
  uint32_t process_id = t.driver()->local_process_id();
  ASSERT_NE(process_id, 0u);
  auto process = t.driver()->find_process(process_id);
  ASSERT_NE(process, nullptr);

  constexpr size_t kPageSize = 0x1000;
  void *host = ::mmap(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                      -1, 0);
  ASSERT_NE(host, MAP_FAILED);

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(host);
  alloc.size = kPageSize;
  alloc.gpu_id = t.driver()->gpu_id();
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0);
  EXPECT_FALSE(process->contains_sparse_reservation(alloc.va_addr, alloc.size));

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(t.driver()->ioctl(process_id, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0);
  EXPECT_EQ(::munmap(host, kPageSize), 0);
  EXPECT_EQ(t.driver()->close(), 0);
}

TEST_F(SimulatedKfdTest, SparseReservationOverlapPrefersHostBackedPages) {
  rocjitsu::KfdProcess proc(/*process_id=*/7);
  constexpr uint64_t kBaseVa = 0x40000000ULL;
  constexpr uint64_t kPage = rocjitsu::KfdProcess::kPageSize;

  {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);

    rocjitsu::KfdProcess::GpuAllocation sparse{};
    sparse.handle = 1;
    sparse.gpu_va = kBaseVa;
    sparse.size = 2 * kPage;
    sparse.host_ptr = nullptr;
    proc.allocations_[sparse.handle] = sparse;

    rocjitsu::KfdProcess::GpuAllocation host{};
    host.handle = 2;
    host.gpu_va = kBaseVa + kPage;
    host.size = kPage;
    host.host_ptr = reinterpret_cast<void *>(0x100000ULL);
    proc.allocations_[host.handle] = host;

    proc.mark_allocations_dirty();
  }

  EXPECT_TRUE(proc.contains_sparse_reservation(kBaseVa, kPage));
  EXPECT_FALSE(proc.contains_sparse_reservation(kBaseVa + kPage, kPage));
  EXPECT_FALSE(proc.contains_sparse_reservation(kBaseVa, 2 * kPage));

  {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    proc.allocations_.erase(2);
    proc.mark_allocations_dirty();
  }

  EXPECT_TRUE(proc.contains_sparse_reservation(kBaseVa, 2 * kPage));
}

TEST_F(SimulatedKfdTest, LargeSparseReservationClassifiesReservedRange) {
  rocjitsu::KfdProcess proc(/*process_id=*/9);
  constexpr uint64_t kBaseVa = 0x8000000000ULL;
  constexpr uint64_t kLargeSparseSize = 1ULL << 40;
  constexpr uint64_t kPage = rocjitsu::KfdProcess::kPageSize;

  {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    rocjitsu::KfdProcess::GpuAllocation sparse{};
    sparse.handle = 1;
    sparse.gpu_va = kBaseVa;
    sparse.size = kLargeSparseSize;
    sparse.host_ptr = nullptr;
    proc.allocations_[sparse.handle] = sparse;
    proc.mark_allocations_dirty();
  }

  EXPECT_TRUE(proc.contains_sparse_reservation(kBaseVa, kPage));
  EXPECT_TRUE(proc.contains_sparse_reservation(kBaseVa + (kLargeSparseSize / 2), kPage));
  EXPECT_TRUE(proc.contains_sparse_reservation(kBaseVa + kLargeSparseSize - kPage, kPage));
  EXPECT_FALSE(proc.contains_sparse_reservation(kBaseVa - kPage, kPage));
  EXPECT_FALSE(proc.contains_sparse_reservation(kBaseVa + kLargeSparseSize, kPage));
}

TEST_F(SimulatedKfdTest, GuestDiscoveryOpenIsReleasedOnLastClose) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm, nullptr);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);
  ASSERT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "RJ_VM_MODE_LOCAL must provide the bootstrap open adopted by GuestKfd";

  rocjitsu::config::DbtGuestConfig config;
  config.enabled = true;
  config.guest_isa = "gfx950";
  config.host.isa = "gfx950";
  config.host.gpu_id = execution_driver->gpu_id();
  config.host.backend = rocjitsu::config::DbtExecutionBackend::Simulator;
  config.guest_device = vm->loaded.device;
  config.guest_device.gpu_id += 1;
  config.guest_device.drm_render_minor += 1;

  rocjitsu::GuestKfd guest(std::move(config), execution_driver);
  for (int cycle = 0; cycle < 2; ++cycle) {
    ASSERT_TRUE(guest.prepare_for_discovery());

    const int app_fd = guest.open();
    ASSERT_GE(app_fd, 0);
    EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
        << "application open must reuse the discovery-owned reference";

    EXPECT_EQ(::close(app_fd), 0);
    EXPECT_EQ(guest.close(), 0);
    EXPECT_EQ(execution_driver->local_open_ref_count(), 0u)
        << "last guest close must release the simulated process";
  }
}

TEST_F(SimulatedKfdTest, GuestOpenSurvivesExecutionPrimaryOverwrite) {
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create(CONFIG_PATH.c_str(), RJ_VM_MODE_LOCAL, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw_vm, &rj_vm_destroy);
  ASSERT_NE(vm, nullptr);
  ASSERT_NE(vm->vm, nullptr);
  auto *execution_driver = vm->vm->driver();
  ASSERT_NE(execution_driver, nullptr);
  ASSERT_EQ(execution_driver->local_open_ref_count(), 1u);

  rocjitsu::config::DbtGuestConfig config;
  config.enabled = true;
  config.guest_isa = "gfx950";
  config.host.isa = "gfx950";
  config.host.gpu_id = execution_driver->gpu_id();
  config.host.backend = rocjitsu::config::DbtExecutionBackend::Simulator;
  config.guest_device = vm->loaded.device;
  config.guest_device.gpu_id += 1;
  config.guest_device.drm_render_minor += 1;

  rocjitsu::GuestKfd guest(std::move(config), execution_driver);
  ASSERT_TRUE(guest.prepare_for_discovery());
  const int app_fd = guest.open();
  ASSERT_GE(app_fd, 0);
  ASSERT_EQ(execution_driver->local_open_ref_count(), 1u);
  const uint32_t process_id = execution_driver->local_process_id();

  const int hidden_fd = execution_driver->fd();
  ASSERT_GE(hidden_fd, 0);
  int pipefd[2];
  ASSERT_EQ(::pipe(pipefd), 0);
  ASSERT_EQ(::dup2(pipefd[0], hidden_fd), hidden_fd);
  ASSERT_EQ(::close(pipefd[0]), 0);

  EXPECT_EQ(guest.invalidate_primary_fd(hidden_fd),
            rocjitsu::LinuxKfd::PrimaryInvalidation::kClearedKeepRefs);
  EXPECT_EQ(execution_driver->fd(), -1);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "hidden-fd overwrite must not release GuestKfd's backend open";
  EXPECT_EQ(execution_driver->local_process_id(), process_id);

  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(guest.ioctl(AMDKFD_IOC_GET_VERSION, &version), 0)
      << "the surviving app open must keep the simulated process usable";

  const int reopened_fd = guest.open();
  ASSERT_GE(reopened_fd, 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u)
      << "re-minting the simulator primary must not leak another backend open";
  EXPECT_EQ(execution_driver->local_process_id(), process_id);

  EXPECT_EQ(::close(reopened_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 1u);
  EXPECT_EQ(::close(app_fd), 0);
  EXPECT_EQ(guest.close(), 0);
  EXPECT_EQ(execution_driver->local_open_ref_count(), 0u);

  EXPECT_EQ(::close(hidden_fd), 0);
  EXPECT_EQ(::close(pipefd[1]), 0);
}

TEST_F(SimulatedKfdTest, TopologyDirectoryExists) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);

  const auto &path = t.driver()->topology().path();
  EXPECT_FALSE(path.empty());
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::exists(path + "/generation_id"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/0/properties"));
  EXPECT_TRUE(std::filesystem::exists(path + "/nodes/1/properties"));
}

// Regression test for the close()-vs-in-flight-ioctl teardown race. ioctl()
// snapshots the KfdProcess shared_ptr WITHOUT retaining an open reference, so a
// concurrent close() can erase and tear down the process while other threads are
// dispatching ioctls against the same process id. The hardening (op_mutex_ held
// across all teardown + an is_closing() guard taken right after op_mutex_ in
// dispatch_ioctl) must ensure: (a) no crash / use-after-free, and (b) an ioctl
// that loses the race returns a clean error (-ESRCH) instead of operating on a
// dismantled process. This drives that race in-process, many times, under TSan/
// ASan in the sanitizer CI job.
TEST_F(SimulatedKfdTest, ConcurrentIoctlAndCloseIsRaceFree) {
  auto t = create_test_vm();
  ASSERT_NE(t.driver(), nullptr);
  auto *drv = t.driver();

  constexpr int kRounds = 200;
  constexpr int kIoctlThreads = 4;

  for (int round = 0; round < kRounds; ++round) {
    uint32_t pid = drv->open_process(/*client_pid=*/0);
    ASSERT_NE(pid, 0u);

    const uint32_t gpu_id = drv->gpu_id();
    std::atomic<bool> go{false};
    std::atomic<int> bad{0};
    std::vector<std::thread> workers;
    workers.reserve(kIoctlThreads);
    for (int i = 0; i < kIoctlThreads; ++i) {
      workers.emplace_back([&, pid] {
        while (!go.load(std::memory_order_acquire))
          ;
        // Accepted outcomes for every ioctl below: success (process still live),
        // or a clean -ESRCH/-EINVAL once close() has torn it down. Any other value
        // (or a crash / ASAN-TSAN report) means teardown overlapped a live ioctl.
        auto ok = [](int rc) { return rc == 0 || rc == -ESRCH || rc == -EINVAL; };
        for (int k = 0; k < 50; ++k) {
          // Stateless routing probe.
          kfd_ioctl_get_version_args ver{};
          if (!ok(drv->ioctl(pid, AMDKFD_IOC_GET_VERSION, &ver)))
            bad.fetch_add(1, std::memory_order_relaxed);

          // State-touching ioctls that read/write alloc_mutex_-guarded
          // allocations_, so close()'s teardown can actually overlap a handler
          // dereferencing per-process state (not just the op_mutex_/is_closing
          // gate). alloc then free the returned handle.
          kfd_ioctl_alloc_memory_of_gpu_args alloc{};
          alloc.va_addr = 0x100000000ULL + static_cast<uint64_t>(k) * 0x1000ULL;
          alloc.size = 0x1000;
          alloc.gpu_id = gpu_id;
          alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
          int arc = drv->ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc);
          if (!ok(arc))
            bad.fetch_add(1, std::memory_order_relaxed);
          if (arc == 0) {
            kfd_ioctl_free_memory_of_gpu_args freed{};
            freed.handle = alloc.handle;
            if (!ok(drv->ioctl(pid, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &freed)))
              bad.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }

    go.store(true, std::memory_order_release);
    // Race close() against the in-flight ioctls.
    int cret = drv->close(pid);
    EXPECT_EQ(cret, 0);
    for (auto &w : workers)
      w.join();
    EXPECT_EQ(bad.load(), 0) << "round " << round << ": ioctl saw invalid state during close";
  }
}

} // namespace
