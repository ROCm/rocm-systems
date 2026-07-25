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

#include "embedded_schema.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "simdojo/sim/simulation.h"

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
#include <thread>
#include <vector>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/gfx950_cdna4.json";
constexpr uint32_t kGpuId = 38144;

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

  EXPECT_EQ(session.matching_address_watch_slots(address, 1,
                                                 KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD,
                                                 KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL),
            0b0111u);
  EXPECT_EQ(session.matching_address_watch_slots(address, 1, KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ,
                                                 KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL),
            0b1101u);
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

  const rj_client_pid_t kClientPid = getpid();
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
            -EFAULT);
  EXPECT_EQ(null_device->device_snapshot.snapshot_buf_ptr, 0u);
  EXPECT_TRUE(std::all_of(null_device_payload.begin() + sizeof(kfd_ioctl_dbg_trap_args),
                          null_device_payload.end(),
                          [](uint8_t byte) { return byte == kSentinel; }));

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

// One-shot daemon stand-in: read a single RPC_IOCTL, then reply with `result`
// and a response whose inline tail (after the echoed arg struct) is
// `extra_bytes` of `poison`. Does not close `server_fd` (caller owns it).
void serve_one_ioctl_reply(int server_fd, int32_t result, size_t arg_struct_size,
                           size_t extra_bytes, uint8_t poison,
                           uint32_t returned_snapshot_entry_size = 0) {
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
  if (returned_snapshot_entry_size != 0) {
    auto *dbg = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(out.data());
    if (dbg->op == KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT)
      dbg->device_snapshot.entry_size = returned_snapshot_entry_size;
    else if (dbg->op == KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT)
      dbg->queue_snapshot.entry_size = returned_snapshot_entry_size;
  }
  std::memset(out.data() + arg_struct_size, poison, extra_bytes);

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

TEST(RemoteDriverDbgQueueSnapshotTest, FailedSnapshotLeavesCallerBufferUntouched) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr size_t kCap = 2 * sizeof(kfd_queue_snapshot_entry);
  std::array<uint8_t, kCap> caller_buf;
  caller_buf.fill(0xAB);
  std::jthread server([&, server_fd = sv[1]] {
    serve_one_ioctl_reply(server_fd, -EFAULT, sizeof(kfd_ioctl_dbg_trap_args), kCap, 0xCD);
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = 2;
  snap.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -EFAULT);
  EXPECT_TRUE(
      std::all_of(caller_buf.begin(), caller_buf.end(), [](uint8_t byte) { return byte == 0xAB; }));
  server.join();
}

TEST(RemoteDriverDbgQueueSnapshotTest, SuccessfulReplyWithNullBufferReturnsEfault) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr size_t kCap = sizeof(kfd_queue_snapshot_entry);
  std::jthread server([&, server_fd = sv[1]] {
    serve_one_ioctl_reply(server_fd, 0, sizeof(kfd_ioctl_dbg_trap_args), kCap, 0xCD);
    ::close(server_fd);
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

TEST(RemoteDriverDbgQueueSnapshotTest, SuccessfulSnapshotClampsCopyToCallerCapacity) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);

  constexpr size_t kStride = sizeof(kfd_queue_snapshot_entry) + 16;
  constexpr size_t kCap = kStride;
  constexpr size_t kGuard = 32;
  std::array<uint8_t, kCap + kGuard> caller_buf;
  caller_buf.fill(0xAB);
  std::jthread server([&, server_fd = sv[1]] {
    serve_one_ioctl_reply(server_fd, 0, sizeof(kfd_ioctl_dbg_trap_args), caller_buf.size(), 0xCD,
                          sizeof(kfd_queue_snapshot_entry));
    ::close(server_fd);
  });

  rocjitsu::RemoteDriver rd(sv[0]);
  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = 1;
  snap.queue_snapshot.entry_size = kStride;
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(caller_buf.data());

  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_TRUE(std::all_of(caller_buf.begin(), caller_buf.begin() + sizeof(kfd_queue_snapshot_entry),
                          [](uint8_t byte) { return byte == 0xCD; }));
  EXPECT_TRUE(std::all_of(caller_buf.begin() + sizeof(kfd_queue_snapshot_entry), caller_buf.end(),
                          [](uint8_t byte) { return byte == 0xAB; }));
  server.join();
}

TEST(RemoteDriverDbgQueueSnapshotTest, OversizedCapacityIsRejectedBeforeTransport) {
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0) << ::strerror(errno);
  rocjitsu::RemoteDriver rd(sv[0]);

  kfd_ioctl_dbg_trap_args snap{};
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.num_queues = UINT32_MAX;
  snap.queue_snapshot.entry_size = UINT32_MAX;
  snap.queue_snapshot.snapshot_buf_ptr = 1;
  EXPECT_EQ(rd.ioctl(AMDKFD_IOC_DBG_TRAP, &snap), -E2BIG);

  ::close(sv[1]);
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

TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotEnumeratesAgent) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = make_debug_fd();
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // First call with num_devices=0 reports the total device count.
  kfd_ioctl_dbg_trap_args count{};
  count.pid = static_cast<uint32_t>(getpid());
  count.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  count.device_snapshot.entry_size = sizeof(kfd_dbg_device_info_entry);
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

  EXPECT_EQ(entry.gpu_id, kGpuId);
  EXPECT_EQ(entry.gfx_target_version, 90500u); // gfx950 fixture config
  EXPECT_TRUE(entry.capability & HSA_CAP_TRAP_DEBUG_SUPPORT);
  EXPECT_EQ(entry.debug_prop, 0x5d6u); // gfx950 debug_prop
  // rocm-dbgapi's agent_snapshot fatal-errors if any of these are zero.
  EXPECT_NE(entry.simd_count, 0u);
  EXPECT_NE(entry.max_waves_per_simd, 0u);
  EXPECT_NE(entry.array_count, 0u);
  EXPECT_NE(entry.simd_arrays_per_engine, 0u);
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
  EXPECT_EQ(suspended_queues, 1u);

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
  ASSERT_TRUE(
      rocjitsu::kmd::deserialize_queue_cwsr(kCwsrAddress, kCwsrSize, states, [&](uint64_t address) {
        return memory->read32(address, driver_->local_process_id());
      }));
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
        [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); }));
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
    ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(kCwsrAddress, kCwsrSize, states,
                                                    [&](uint64_t address, uint32_t value) {
                                                      memory->write32(address, value,
                                                                      driver_->local_process_id());
                                                    })
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
          [&](uint64_t address) { return memory->read32(address, driver_->local_process_id()); }));
      auto stopped_peer = std::find_if(states.begin(), states.end(), [&](const auto &state) {
        return state.wave_id != kSteppingWaveId;
      });
      ASSERT_NE(stopped_peer, states.end());
      stopped_peer->wave_stopped = true;
      stopped_peer->mode |= kModeDebugEn;
      ASSERT_TRUE(rocjitsu::kmd::serialize_queue_cwsr(
                      kCwsrAddress, kCwsrSize, states,
                      [&](uint64_t address, uint32_t value) {
                        memory->write32(address, value, driver_->local_process_id());
                      })
                      .ok);
      control.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
      control.resume_queues.queue_array_ptr = reinterpret_cast<uint64_t>(&queue_id);
      control.resume_queues.num_queues = 1;
      ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &control), 1);
      EXPECT_TRUE(peer->debug_halted());
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
  const int enable_result = daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en);
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
  ASSERT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en), 0);

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
  ASSERT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en), 0);

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
  ASSERT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en), 0);

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
  ASSERT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en), 0);

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
  ASSERT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &first_enable), 0);

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
  EXPECT_EQ(daemon.ioctl(replacement_debugger, AMDKFD_IOC_DBG_TRAP, &replacement_enable), 0);

  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(target_pid);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(replacement_debugger, AMDKFD_IOC_DBG_TRAP, &dis), 0);
  ASSERT_EQ(ptrace(PTRACE_KILL, target_pid, nullptr, nullptr), 0);
  ASSERT_EQ(waitpid(target_pid, &status, 0), target_pid);
  target_guard.release();
  daemon.close(debugger);
  daemon.close(replacement_debugger);
}

TEST_F(KfdIoctlTest, DebugSessionReaperShutdownDoesNotHang) {
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
}

} // namespace
