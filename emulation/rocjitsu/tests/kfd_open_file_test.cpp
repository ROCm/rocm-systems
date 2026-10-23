// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Public ioctl integration tests run through the LD_PRELOAD interposer.
#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "uapi/linux/kfd_ioctl.h"
#include <uapi/linux/drm/amdgpu_drm.h>
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace {

class File {
public:
  explicit File(int descriptor) : descriptor_(descriptor) {}
  ~File() { close(); }
  File(const File &) = delete;
  File &operator=(const File &) = delete;
  int get() const { return descriptor_; }
  void close() {
    if (descriptor_ >= 0) {
      EXPECT_EQ(::close(descriptor_), 0);
      descriptor_ = -1;
    }
  }

private:
  /// Application descriptor, closed through the production interposer.
  int descriptor_;
};

int open_kfd() { return open("/dev/kfd", O_RDWR | O_CLOEXEC); }

uint32_t gpu_id() {
  uint32_t id = 0;
  std::ifstream input("/sys/class/kfd/kfd/topology/nodes/1/gpu_id");
  input >> id;
  return id;
}

size_t descriptor_count() {
  DIR *directory = opendir("/proc/self/fd");
  EXPECT_NE(directory, nullptr);
  if (!directory)
    return 0;
  size_t count = 0;
  while (const auto *entry = readdir(directory))
    if (entry->d_name[0] != '.')
      ++count;
  EXPECT_EQ(closedir(directory), 0);
  return count;
}

void create_event_backing(int descriptor) {
  kfd_ioctl_create_event_args event{};
  event.event_type = KFD_IOC_EVENT_SIGNAL;
  ASSERT_EQ(ioctl(descriptor, AMDKFD_IOC_CREATE_EVENT, &event), 0);
  void *page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor,
                    static_cast<off_t>(uint64_t{2} << 62));
  ASSERT_NE(page, MAP_FAILED);
  ASSERT_EQ(munmap(page, 4096), 0);
  // The CPU view is gone; KFD retains the backing until context destruction.
}

TEST(KfdOpenFileTest, IndependentOpensSharePrimaryUntilExplicitSelection) {
  File primary(open_kfd());
  File secondary(open_kfd());
  ASSERT_GE(primary.get(), 0);
  ASSERT_GE(secondary.get(), 0);
  ASSERT_NE(primary.get(), secondary.get());
  kfd_ioctl_get_version_args version{};
  ASSERT_EQ(ioctl(primary.get(), AMDKFD_IOC_GET_VERSION, &version), 0);
  EXPECT_EQ(version.minor_version, 19u);

  kfd_ioctl_create_event_args event{};
  event.event_type = KFD_IOC_EVENT_SIGNAL;
  ASSERT_EQ(ioctl(primary.get(), AMDKFD_IOC_CREATE_EVENT, &event), 0);
  kfd_ioctl_set_event_args set{event.event_id, 0};
  ASSERT_EQ(ioctl(secondary.get(), AMDKFD_IOC_SET_EVENT, &set), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(secondary.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  EXPECT_EQ(ioctl(secondary.get(), AMDKFD_IOC_SET_EVENT, &set), -1);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_EQ(ioctl(primary.get(), AMDKFD_IOC_SET_EVENT, &set), 0);
}

TEST(KfdOpenFileTest, DupSharesSelectionAndRetainsSecondaryAfterOriginalClose) {
  File original(open_kfd());
  ASSERT_GE(original.get(), 0);
  File alias(fcntl(original.get(), F_DUPFD_CLOEXEC, 0));
  ASSERT_GE(alias.get(), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(alias.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  EXPECT_EQ(ioctl(original.get(), AMDKFD_IOC_CREATE_PROCESS, &select), -1);
  EXPECT_EQ(errno, EINVAL);
  kfd_ioctl_create_event_args event{};
  event.event_type = KFD_IOC_EVENT_SIGNAL;
  ASSERT_EQ(ioctl(original.get(), AMDKFD_IOC_CREATE_EVENT, &event), 0);
  original.close();
  kfd_ioctl_set_event_args set{event.event_id, 0};
  EXPECT_EQ(ioctl(alias.get(), AMDKFD_IOC_SET_EVENT, &set), 0);
}

TEST(KfdOpenFileTest, SecondaryFinalCloseReleasesEventBacking) {
  File primary(open_kfd());
  ASSERT_GE(primary.get(), 0);
  const size_t before = descriptor_count();
  {
    File secondary(open_kfd());
    ASSERT_GE(secondary.get(), 0);
    kfd_ioctl_create_process_args select{};
    ASSERT_EQ(ioctl(secondary.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
    create_event_backing(secondary.get());
    EXPECT_GT(descriptor_count(), before);
    // Deliberately leave the event live: last-close owns its cleanup.
  }
  EXPECT_EQ(descriptor_count(), before);
  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(ioctl(primary.get(), AMDKFD_IOC_GET_VERSION, &version), 0);
}

TEST(KfdOpenFileTest, ConcurrentSelectionHasOneWinner) {
  File original(open_kfd());
  ASSERT_GE(original.get(), 0);
  File alias(dup(original.get()));
  ASSERT_GE(alias.get(), 0);
  std::barrier ready(2);
  std::array<int, 2> results{};
  std::array<int, 2> errors{};
  auto select = [&](size_t index, int descriptor) {
    kfd_ioctl_create_process_args arguments{};
    ready.arrive_and_wait();
    results[index] = ioctl(descriptor, AMDKFD_IOC_CREATE_PROCESS, &arguments);
    errors[index] = errno;
  };
  std::thread first(select, 0, original.get());
  std::thread second(select, 1, alias.get());
  first.join();
  second.join();
  EXPECT_EQ((results[0] == 0) + (results[1] == 0), 1);
  const size_t loser = results[0] == 0 ? 1 : 0;
  EXPECT_EQ(results[loser], -1);
  EXPECT_EQ(errors[loser], EINVAL);
}

TEST(KfdOpenFileTest, SecondaryAllocationHandlesBelongToTheirFile) {
  File first(open_kfd());
  File second(open_kfd());
  ASSERT_GE(first.get(), 0);
  ASSERT_GE(second.get(), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(first.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  ASSERT_EQ(ioctl(second.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.va_addr = 0x10000000;
  allocation.size = 4096;
  allocation.gpu_id = gpu_id();
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(ioctl(first.get(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  kfd_ioctl_free_memory_of_gpu_args release{allocation.handle};
  EXPECT_EQ(ioctl(second.get(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release), -1);
  EXPECT_EQ(errno, EINVAL);
  EXPECT_EQ(ioctl(first.get(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release), 0);
}

TEST(KfdOpenFileTest, DrmBindingsAndVirtualAddressesAreContextLocal) {
  std::array<File, 2> contexts{File(open_kfd()), File(open_kfd())};
  std::array<File, 2> renders{File(open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC)),
                              File(open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC))};
  std::array<uint32_t, 2> handles{};
  for (size_t index = 0; index < contexts.size(); ++index) {
    ASSERT_GE(contexts[index].get(), 0);
    ASSERT_GE(renders[index].get(), 0);
    kfd_ioctl_create_process_args select{};
    ASSERT_EQ(ioctl(contexts[index].get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
    kfd_ioctl_acquire_vm_args acquire{};
    acquire.drm_fd = static_cast<uint32_t>(renders[index].get());
    acquire.gpu_id = gpu_id();
    ASSERT_EQ(ioctl(contexts[index].get(), AMDKFD_IOC_ACQUIRE_VM, &acquire), 0);
    drm_amdgpu_gem_create allocation{};
    allocation.in.bo_size = 4096;
    allocation.in.alignment = 4096;
    allocation.in.domains = AMDGPU_GEM_DOMAIN_GTT;
    ASSERT_EQ(ioctl(renders[index].get(), DRM_IOCTL_AMDGPU_GEM_CREATE, &allocation), 0);
    handles[index] = allocation.out.handle;
    drm_amdgpu_gem_va mapping{};
    mapping.handle = handles[index];
    mapping.operation = AMDGPU_VA_OP_MAP;
    mapping.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
    mapping.va_address = 0x10000000;
    mapping.map_size = 4096;
    ASSERT_EQ(ioctl(renders[index].get(), DRM_IOCTL_AMDGPU_GEM_VA, &mapping), 0);
  }
  kfd_ioctl_acquire_vm_args foreign{};
  foreign.drm_fd = static_cast<uint32_t>(renders[0].get());
  foreign.gpu_id = gpu_id();
  EXPECT_EQ(ioctl(contexts[1].get(), AMDKFD_IOC_ACQUIRE_VM, &foreign), -1);
  EXPECT_EQ(errno, EBUSY);
  for (size_t index = 0; index < contexts.size(); ++index) {
    drm_amdgpu_gem_va mapping{};
    mapping.handle = handles[index];
    mapping.operation = AMDGPU_VA_OP_UNMAP;
    mapping.va_address = 0x10000000;
    mapping.map_size = 4096;
    EXPECT_EQ(ioctl(renders[index].get(), DRM_IOCTL_AMDGPU_GEM_VA, &mapping), 0);
  }
}

TEST(KfdOpenFileTest, Dup2ReleasesOverwrittenSecondary) {
  File source(open_kfd());
  File target(open_kfd());
  ASSERT_GE(source.get(), 0);
  ASSERT_GE(target.get(), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(target.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  const size_t before = descriptor_count();
  create_event_backing(target.get());
  EXPECT_GT(descriptor_count(), before);
  ASSERT_EQ(dup2(source.get(), target.get()), target.get());
  EXPECT_EQ(descriptor_count(), before);
  EXPECT_EQ(ioctl(target.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  EXPECT_EQ(ioctl(source.get(), AMDKFD_IOC_CREATE_PROCESS, &select), -1);
  EXPECT_EQ(errno, EINVAL);
}

TEST(KfdOpenFileTest, DrmMmapUsesAcquiredSecondary) {
  File context(open_kfd());
  File render(open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC));
  ASSERT_GE(context.get(), 0);
  ASSERT_GE(render.get(), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(context.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  kfd_ioctl_acquire_vm_args acquire{};
  acquire.drm_fd = static_cast<uint32_t>(render.get());
  acquire.gpu_id = gpu_id();
  ASSERT_EQ(ioctl(context.get(), AMDKFD_IOC_ACQUIRE_VM, &acquire), 0);
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.va_addr = 0x10000000;
  allocation.size = 4096;
  allocation.gpu_id = gpu_id();
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(ioctl(context.get(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  void *page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, render.get(),
                    static_cast<off_t>(allocation.mmap_offset));
  ASSERT_NE(page, MAP_FAILED);
  *static_cast<uint32_t *>(page) = 0x12345678;
  EXPECT_EQ(*static_cast<uint32_t *>(page), 0x12345678u);
  EXPECT_EQ(munmap(page, 4096), 0);
  kfd_ioctl_free_memory_of_gpu_args release{allocation.handle};
  EXPECT_EQ(ioctl(context.get(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release), 0);
}

TEST(KfdOpenFileTest, DoorbellClientViewCanUnmapBeforeContextClose) {
  File context(open_kfd());
  ASSERT_GE(context.get(), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(context.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  const uint64_t offset = (uint64_t{3} << 62) | (uint64_t{gpu_id()} << 46);
  for (unsigned mapping = 0; mapping < 2; ++mapping) {
    void *page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, context.get(),
                      static_cast<off_t>(offset));
    ASSERT_NE(page, MAP_FAILED);
    EXPECT_EQ(munmap(page, 4096), 0);
  }
}

TEST(KfdOpenFileTest, SecondaryRejectsPrimaryOnlyOperations) {
  File context(open_kfd());
  ASSERT_GE(context.get(), 0);
  kfd_ioctl_create_process_args select{};
  ASSERT_EQ(ioctl(context.get(), AMDKFD_IOC_CREATE_PROCESS, &select), 0);
  alignas(4096) std::array<uint8_t, 4096> page{};
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.va_addr = reinterpret_cast<uintptr_t>(page.data());
  allocation.mmap_offset = allocation.va_addr;
  allocation.size = page.size();
  allocation.gpu_id = gpu_id();
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  EXPECT_EQ(ioctl(context.get(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), -1);
  EXPECT_EQ(errno, EOPNOTSUPP);
  alignas(kfd_ioctl_svm_args) std::array<uint8_t, sizeof(kfd_ioctl_svm_args)> storage{};
  auto *svm = reinterpret_cast<kfd_ioctl_svm_args *>(storage.data());
  svm->start_addr = allocation.va_addr;
  svm->size = page.size();
  svm->op = KFD_IOCTL_SVM_OP_SET_ATTR;
  EXPECT_EQ(ioctl(context.get(), AMDKFD_IOC_SVM, svm), -1);
  EXPECT_EQ(errno, EOPNOTSUPP);
  kfd_ioctl_dbg_trap_args debug{};
  EXPECT_EQ(ioctl(context.get(), AMDKFD_IOC_DBG_TRAP, &debug), -1);
  EXPECT_EQ(errno, EOPNOTSUPP);
}

} // namespace
