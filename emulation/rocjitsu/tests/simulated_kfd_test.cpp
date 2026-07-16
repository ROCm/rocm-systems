// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulated_kfd_test.cpp
/// @brief Tests for SimulatedKfd creation, open/close, and topology generation.

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/virtual_machine.h"

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

TestVM create_test_vm() {
  TestVM t;
  t.loaded = rocjitsu::config::load_config(CONFIG_PATH.c_str(), rocjitsu::kEmbeddedSchema);
  auto *soc = t.loaded.soc();

  t.loaded.engine_config.max_ticks = 0;
  t.loaded.engine_config.await_primaries = true;
  t.engine = std::make_unique<simdojo::SimulationEngine>(t.loaded.engine_config);

  auto root = t.loaded.take_root();
  root.release();
  auto vm = std::make_unique<rocjitsu::VirtualMachine>(std::unique_ptr<rocjitsu::SoC>(soc));
  vm->driver()->setup_topology(t.loaded.device, soc->num_xcds());

  t.engine->topology().set_root(std::move(vm));
  t.loaded.wire_links(t.engine->topology());
  soc->wire_backing(t.engine->topology());
  t.engine->build();
  t.engine->register_as_primary();

  return t;
}

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
