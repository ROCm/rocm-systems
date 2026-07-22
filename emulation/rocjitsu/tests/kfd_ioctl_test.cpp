// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "embedded_schema.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <fcntl.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_cdna4.json";
constexpr uint32_t kGpuId = 38144;

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
  void SetUp() override {
    setenv("RJ_CONFIG", CONFIG_PATH.c_str(), 1);
    loaded_ = rocjitsu::config::load_config(CONFIG_PATH.c_str(), rocjitsu::kEmbeddedSchema);
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
// debug_mutex_ under ThreadSanitizer. Races are legitimate: a losing ENABLE
// sees EALREADY and a losing DISABLE sees EINVAL. The invariant is that the
// driver serializes them without a data race or torn session state — every call
// returns one of the well-defined codes, never a crash or a bogus errno. Uses
// self-debug (target pid == getpid()) so the whole cycle stays on debug_mutex_
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

TEST_F(KfdIoctlTest, DbgTrapAdmittedOpNotYetImplemented) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // QUERY_DEBUG_EVENT is not a HW-op, so the gate ladder admits it; the handler
  // itself is added by a later change and currently reports not-implemented.
  kfd_ioctl_dbg_trap_args q{};
  q.pid = static_cast<uint32_t>(getpid());
  q.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &q), -ENOSYS);
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
  constexpr pid_t kClientPid = 4242;
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
      if (!rocjitsu::rpc_recv_exact(server_fd, &hdr, sizeof(hdr)))
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

      const int result = daemon_driver.ioctl(process_id, cmd, buf);

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
    rj_vm_execute_as(vm_, process_id_, &cmd);
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

  static constexpr rj_client_pid_t kClientPid = 4242;
  rj_vm_t *vm_ = nullptr;
  uint32_t process_id_ = 0;
};

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
      int in_fds[1] = {-1};
      size_t num_in = 1;
      if (rocjitsu::rpc_recv_msg(server_fd, &hdr, sizeof(hdr), in_fds, &num_in) <= 0)
        break;
      int in_fd = (num_in > 0) ? in_fds[0] : -1;

      if (hdr.opcode == rocjitsu::RPC_CLOSE) {
        rocjitsu::RpcHeader resp{};
        resp.request_id = hdr.request_id;
        rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
        if (in_fd >= 0)
          ::close(in_fd);
        break;
      }
      if (hdr.opcode != rocjitsu::RPC_IOCTL) {
        rocjitsu::RpcHeader resp{};
        resp.request_id = hdr.request_id;
        rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
        if (in_fd >= 0)
          ::close(in_fd);
        continue;
      }

      std::vector<uint8_t> payload(hdr.payload_bytes);
      if (!rocjitsu::rpc_recv_exact(server_fd, payload.data(), hdr.payload_bytes)) {
        if (in_fd >= 0)
          ::close(in_fd);
        break;
      }
      auto *ireq = reinterpret_cast<rocjitsu::RpcIoctlRequest *>(payload.data());

      // Prove the received descriptor is live and aliases the client's eventfd
      // by writing a sentinel through it before the handler adopts it.
      if (in_fd >= 0) {
        fds_received.fetch_add(1);
        // rpc_recv_msg passes MSG_CMSG_CLOEXEC, so the transferred notifier must
        // arrive close-on-exec and cannot leak through a later exec.
        int fd_flags = ::fcntl(in_fd, F_GETFD);
        notifier_cloexec.store((fd_flags >= 0 && (fd_flags & FD_CLOEXEC)) ? 1 : 0);
        uint64_t s = kSentinel;
        [[maybe_unused]] ssize_t w = ::write(in_fd, &s, sizeof(s));
      }

      rj_vm_cmd_t cmd{};
      cmd.cmd = ireq->ioctl_cmd;
      cmd.buf = payload.data() + sizeof(rocjitsu::RpcIoctlRequest);
      cmd.buf_size = ireq->args_bytes;
      cmd.shared_handle = -1;
      cmd.in_handle = in_fd;
      rj_vm_execute_as(vm_, process_id_, &cmd);
      in_handle_after.store(cmd.in_handle);
      if (cmd.in_handle >= 0)
        ::close(cmd.in_handle);

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

  // The client sent exactly one fd via SCM_RIGHTS and the session adopted it.
  EXPECT_EQ(fds_received.load(), 1);
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

// --- RemoteDriver DBG_TRAP GET_DEVICE_SNAPSHOT response copy-back ---
//
// The client saves the caller's snapshot buffer pointer and capacity
// (num_devices * entry_size) before serialization, then on the response only
// writes it back on success, clamped to that capacity. This guards daemon mode
// against a failed op (e.g. -ENOSYS) mutating caller memory and against a
// daemon-returned count larger than the caller's buffer.

// One-shot daemon stand-in: read a single RPC_IOCTL, then reply with `result`
// and a response whose inline tail (after the echoed arg struct) is
// `extra_bytes` of `poison`. Does not close `server_fd` (caller owns it).
void serve_one_ioctl_reply(int server_fd, int32_t result, size_t arg_struct_size,
                           size_t extra_bytes, uint8_t poison) {
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
  std::memset(out.data() + arg_struct_size, poison, extra_bytes);

  rocjitsu::RpcHeader resp{};
  resp.opcode = rocjitsu::RPC_IOCTL;
  resp.request_id = hdr.request_id;
  resp.result = result;
  resp.payload_bytes = static_cast<uint32_t>(out.size());
  rocjitsu::rpc_send_exact(server_fd, &resp, sizeof(resp));
  rocjitsu::rpc_send_exact(server_fd, out.data(), out.size());
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
  int dead_fd = ::eventfd(0, EFD_CLOEXEC);
  ASSERT_GE(dead_fd, 0);
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
  en.pid = 4242;
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

} // namespace
