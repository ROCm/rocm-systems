// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "embedded_schema.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstring>
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
  engine.build();
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
    engine_->build();
    engine_->register_as_primary();

    driver_->setup_topology(loaded_.device, num_xcds);
    int fd = driver_->open();
    ASSERT_GE(fd, 0);
  }

  void TearDown() override {
    if (driver_)
      driver_->close();
  }

  rocjitsu::config::LoadedConfig loaded_;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
  rocjitsu::SoC *soc_ = nullptr;
  rocjitsu::SimulatedDriver *driver_ = nullptr;
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
  rocjitsu::SimulatedDriver daemon_driver(*soc_, true);
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

  // Two dups of the KFD fd.
  driver_->retain_local_open();
  driver_->retain_local_open();
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
  args.enable.dbg_fd = 7;
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

TEST_F(KfdIoctlTest, DbgTrapDoubleEnableReturnsEINVAL) {
  kfd_ioctl_dbg_trap_args args{};
  args.pid = static_cast<uint32_t>(getpid());
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.dbg_fd = KFD_INVALID_FD;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), 0);
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &args), -EINVAL);
}

TEST_F(KfdIoctlTest, DbgTrapHwOpWithoutRuntimeReturnsEPERM) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = KFD_INVALID_FD;
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
  en.enable.dbg_fd = KFD_INVALID_FD;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // Runtime is enabled, so the HW-op gate passes and the gpu-id check runs.
  kfd_ioctl_dbg_trap_args watch{};
  watch.pid = static_cast<uint32_t>(getpid());
  watch.op = KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH;
  watch.set_node_address_watch.gpu_id = 0xdeadbeef;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &watch), -ENODEV);
}

TEST_F(KfdIoctlTest, DbgTrapAdmittedOpsBehaveAsSpecified) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = KFD_INVALID_FD;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // QUERY_DEBUG_EVENT is admitted by the gate ladder and reports EAGAIN when no
  // wave/queue exception is pending.
  kfd_ioctl_dbg_trap_args q{};
  q.pid = static_cast<uint32_t>(getpid());
  q.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &q), -EAGAIN);

  // SEND_RUNTIME_EVENT acknowledges the runtime-enable handshake and succeeds
  // even when no inferior is currently blocking on it.
  kfd_ioctl_dbg_trap_args ev{};
  ev.pid = static_cast<uint32_t>(getpid());
  ev.op = KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &ev), 0);
}

TEST_F(KfdIoctlTest, DbgTrapDeviceSnapshotEnumeratesAgent) {
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = KFD_INVALID_FD;
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
  en.enable.dbg_fd = KFD_INVALID_FD;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // No queues yet: the count call reports zero.
  kfd_ioctl_dbg_trap_args count0{};
  count0.pid = static_cast<uint32_t>(getpid());
  count0.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  count0.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count0), 0);
  EXPECT_EQ(count0.queue_snapshot.num_queues, 0u);

  // Create one compute (AQL) queue with a known context-save-restore area.
  std::vector<uint8_t> ring(4096, 0), rw(4096, 0);
  constexpr uint64_t kCwsrVa = 0x123400000ULL;
  constexpr uint32_t kCwsrSize = 0x8000;
  kfd_ioctl_create_queue_args cq{};
  cq.gpu_id = kGpuId;
  cq.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  cq.ring_base_address = reinterpret_cast<uint64_t>(ring.data());
  cq.ring_size = static_cast<uint32_t>(ring.size());
  cq.read_pointer_address = reinterpret_cast<uint64_t>(rw.data());
  cq.write_pointer_address = reinterpret_cast<uint64_t>(rw.data() + 64);
  cq.ctx_save_restore_address = kCwsrVa;
  cq.ctx_save_restore_size = kCwsrSize;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_CREATE_QUEUE, &cq), 0);

  // Count now reports one queue.
  kfd_ioctl_dbg_trap_args count1{};
  count1.pid = static_cast<uint32_t>(getpid());
  count1.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  count1.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count1), 0);
  ASSERT_EQ(count1.queue_snapshot.num_queues, 1u);
  EXPECT_EQ(count1.queue_snapshot.entry_size, sizeof(kfd_queue_snapshot_entry));

  // Fill reports the CWSR geometry rocm-dbgapi needs to find waves.
  kfd_queue_snapshot_entry entry{};
  kfd_ioctl_dbg_trap_args snap{};
  snap.pid = static_cast<uint32_t>(getpid());
  snap.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  snap.queue_snapshot.entry_size = sizeof(entry);
  snap.queue_snapshot.num_queues = 1;
  snap.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(&entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &snap), 0);
  EXPECT_EQ(entry.queue_id, cq.queue_id);
  EXPECT_EQ(entry.gpu_id, kGpuId);
  EXPECT_EQ(entry.ctx_save_restore_address, kCwsrVa);
  EXPECT_EQ(entry.ctx_save_restore_area_size, kCwsrSize);
  EXPECT_EQ(entry.ring_base_address, reinterpret_cast<uint64_t>(ring.data()));
  EXPECT_EQ(entry.queue_type, static_cast<uint32_t>(KFD_IOC_QUEUE_TYPE_COMPUTE_AQL));

  // Destroying the queue removes it from the snapshot.
  kfd_ioctl_destroy_queue_args dq{};
  dq.queue_id = cq.queue_id;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DESTROY_QUEUE, &dq), 0);
  kfd_ioctl_dbg_trap_args count2{};
  count2.pid = static_cast<uint32_t>(getpid());
  count2.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  count2.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &count2), 0);
  EXPECT_EQ(count2.queue_snapshot.num_queues, 0u);
}

// Cross-process authorization: a debugger may only act on a process it has
// ptrace-attached. Uses a real forked child and real ptrace so the check
// exercises the live /proc TracerPid relationship, matching how rocgdb
// launches and traces the inferior.
TEST_F(KfdIoctlTest, DbgTrapCrossProcessEnableAuthorizedByPtrace) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
    _exit(0);
  }

  rocjitsu::SimulatedDriver daemon(*loaded_.soc(), /*daemon_mode=*/true);
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

  // Not yet ptrace-attached: the debugger is not the child's tracer -> EPERM.
  EXPECT_EQ(enable_from_debugger(), -EPERM);

  if (ptrace(PTRACE_ATTACH, child, nullptr, nullptr) == 0) {
    int status = 0;
    waitpid(child, &status, 0); // child stops under ptrace
    // getpid() is now the child's ptrace parent -> ENABLE authorized.
    EXPECT_EQ(enable_from_debugger(), 0);
    ptrace(PTRACE_DETACH, child, nullptr, nullptr);
  } else {
    GTEST_LOG_(INFO) << "PTRACE_ATTACH not permitted; skipped the authorized case";
  }

  daemon.close(debugger);
  daemon.close(inferior);
  kill(child, SIGKILL);
  int status = 0;
  waitpid(child, &status, 0);
}

// The debugger config / query ops rocm-dbgapi issues during attach and detach
// (SET_FLAGS, SET_WAVE_LAUNCH_MODE/OVERRIDE, GET_QUEUE_SNAPSHOT, QUERY_DEBUG_
// EVENT) succeed so the attach/detach lifecycle completes cleanly.
TEST_F(KfdIoctlTest, DbgTrapAttachDetachConfigOps) {
  kfd_ioctl_runtime_enable_args rt{};
  rt.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_RUNTIME_ENABLE, &rt), 0);

  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(getpid());
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = KFD_INVALID_FD;
  ASSERT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &en), 0);

  // SET_FLAGS returns the previously-enabled flags.
  kfd_ioctl_dbg_trap_args f{};
  f.pid = static_cast<uint32_t>(getpid());
  f.op = KFD_IOC_DBG_TRAP_SET_FLAGS;
  f.set_flags.flags = KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &f), 0);
  EXPECT_EQ(f.set_flags.flags, 0u); // nothing was enabled before
  f.set_flags.flags = 0;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &f), 0);
  EXPECT_EQ(f.set_flags.flags, static_cast<uint32_t>(KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP));

  // SET_WAVE_LAUNCH_MODE / OVERRIDE are accepted.
  kfd_ioctl_dbg_trap_args m{};
  m.pid = static_cast<uint32_t>(getpid());
  m.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE;
  m.launch_mode.launch_mode = KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &m), 0);

  kfd_ioctl_dbg_trap_args ov{};
  ov.pid = static_cast<uint32_t>(getpid());
  ov.op = KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE;
  ov.launch_override.override_mode = KFD_DBG_TRAP_OVERRIDE_OR;
  ov.launch_override.enable_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  ov.launch_override.support_request_mask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &ov), 0);
  EXPECT_EQ(ov.launch_override.enable_mask, 0u); // previously enabled = none

  // GET_QUEUE_SNAPSHOT reports an empty snapshot for now.
  kfd_ioctl_dbg_trap_args qs{};
  qs.pid = static_cast<uint32_t>(getpid());
  qs.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
  qs.queue_snapshot.entry_size = sizeof(kfd_queue_snapshot_entry);
  EXPECT_EQ(driver_->ioctl(AMDKFD_IOC_DBG_TRAP, &qs), 0);
  EXPECT_EQ(qs.queue_snapshot.num_queues, 0u);
}

// DISABLE must clean up the session even after the inferior has exited (rocgdb
// disables debug while detaching from a killed process). The kernel likewise
// exempts DISABLE from its liveness checks.
TEST_F(KfdIoctlTest, DbgTrapDisableSucceedsAfterInferiorExits) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
    _exit(0);
  }

  rocjitsu::SimulatedDriver daemon(*loaded_.soc(), /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);

  if (ptrace(PTRACE_ATTACH, child, nullptr, nullptr) != 0) {
    kill(child, SIGKILL);
    int s = 0;
    waitpid(child, &s, 0);
    daemon.close(debugger);
    GTEST_SKIP() << "ptrace not permitted in this environment";
  }
  int status = 0;
  waitpid(child, &status, 0);

  // Enable debug on the (unconnected) ptraced child.
  kfd_ioctl_dbg_trap_args en{};
  en.pid = static_cast<uint32_t>(child);
  en.op = KFD_IOC_DBG_TRAP_ENABLE;
  en.enable.dbg_fd = KFD_INVALID_FD;
  ASSERT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en), 0);

  // The child exits and is reaped: its pid no longer resolves to a process.
  ptrace(PTRACE_DETACH, child, nullptr, nullptr);
  kill(child, SIGKILL);
  waitpid(child, &status, 0);

  // DISABLE still tears the session down cleanly.
  kfd_ioctl_dbg_trap_args dis{};
  dis.pid = static_cast<uint32_t>(child);
  dis.op = KFD_IOC_DBG_TRAP_DISABLE;
  EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), 0);

  daemon.close(debugger);
}

// PR: debug sessions are keyed by the inferior's Linux pid in a driver-level
// table, independent of any KfdProcess. rocgdb enables debug on the inferior
// right after exec, before its ROCr has opened /dev/kfd, so ENABLE must succeed
// for a live, ptraced pid that has NO KfdProcess yet. This is the scenario the
// pid-keyed session unblocked (previously it failed with ESRCH).
TEST_F(KfdIoctlTest, DbgTrapEnableSucceedsForPtracedPidWithoutKfdProcess) {
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    for (;;)
      pause();
    _exit(0);
  }

  rocjitsu::SimulatedDriver daemon(*loaded_.soc(), /*daemon_mode=*/true);
  uint32_t debugger = daemon.open_process(getpid());
  ASSERT_NE(debugger, 0u);
  // Deliberately do NOT open_process(child): the inferior has not connected to
  // /dev/kfd, so it has no KfdProcess.

  auto enable_child = [&]() {
    kfd_ioctl_dbg_trap_args en{};
    en.pid = static_cast<uint32_t>(child);
    en.op = KFD_IOC_DBG_TRAP_ENABLE;
    en.enable.dbg_fd = KFD_INVALID_FD;
    return daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &en);
  };

  if (ptrace(PTRACE_ATTACH, child, nullptr, nullptr) == 0) {
    int status = 0;
    waitpid(child, &status, 0);
    // The session is created in the driver-level table for a pid with no
    // KfdProcess.
    EXPECT_EQ(enable_child(), 0);

    kfd_ioctl_dbg_trap_args dis{};
    dis.pid = static_cast<uint32_t>(child);
    dis.op = KFD_IOC_DBG_TRAP_DISABLE;
    EXPECT_EQ(daemon.ioctl(debugger, AMDKFD_IOC_DBG_TRAP, &dis), 0);
    ptrace(PTRACE_DETACH, child, nullptr, nullptr);
  } else {
    GTEST_LOG_(INFO) << "PTRACE_ATTACH not permitted; skipped";
  }

  daemon.close(debugger);
  kill(child, SIGKILL);
  int status = 0;
  waitpid(child, &status, 0);
}

} // namespace
