// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "embedded_schema.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <vector>

namespace {

const std::string CONFIG_PATH = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";
constexpr uint32_t kGpuId = 38144;

class KfdIoctlTest : public ::testing::Test {
protected:
  void SetUp() override {
    setenv("RJ_CONFIG", CONFIG_PATH.c_str(), 1);
    loaded_ = rocjitsu::config::load_config(CONFIG_PATH.c_str(), rocjitsu::kEmbeddedSchema);
    auto *soc = loaded_.soc();
    auto num_xcds = soc->num_xcds();

    loaded_.engine_config.max_ticks = 0;
    loaded_.engine_config.await_primaries = true;
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded_.engine_config);

    auto root = loaded_.take_root();
    root.release();
    auto vm = std::make_unique<rocjitsu::VirtualMachine>(std::unique_ptr<rocjitsu::SoC>(soc));
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

} // namespace
