// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <sys/mman.h>

namespace {
thread_local bool trace_allocations = false;
thread_local size_t allocation_count = 0;
thread_local size_t fail_allocation = 0;
thread_local bool injected = false;

class AllocationTrace {
public:
  explicit AllocationTrace(size_t failure_index) {
    allocation_count = 0;
    fail_allocation = failure_index;
    injected = false;
  }
  ~AllocationTrace() { trace_allocations = false; }
  void start() { trace_allocations = true; }
  AllocationTrace(const AllocationTrace &) = delete;
  AllocationTrace &operator=(const AllocationTrace &) = delete;
};
} // namespace

// Isolate the replacement from the main suite, including nothrow new so ASan
// sees matching allocation and deallocation functions.
void *operator new(size_t size) {
  if (trace_allocations && ++allocation_count == fail_allocation) {
    trace_allocations = false;
    injected = true;
    throw std::bad_alloc();
  }
  for (;;) {
    if (void *memory = std::malloc(size ? size : 1)) {
      return memory;
    }
    const auto handler = std::get_new_handler();
    if (!handler) {
      throw std::bad_alloc();
    }
    handler();
  }
}
RJ_NOINLINE void operator delete(void *memory) noexcept { std::free(memory); }
RJ_NOINLINE void operator delete(void *memory, size_t) noexcept { std::free(memory); }
void *operator new(size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}
RJ_NOINLINE void operator delete(void *memory, const std::nothrow_t &) noexcept {
  ::operator delete(memory);
}

namespace {

void check_replacement(uint32_t flags, size_t failure_index, size_t &allocations) {
  SCOPED_TRACE(failure_index);
  constexpr size_t kPage = 4096;
  const auto release_reservation = [](void *address) {
    EXPECT_EQ(::munmap(address, 3 * kPage), 0);
  };
  void *reserved =
      ::mmap(nullptr, 3 * kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(reserved, MAP_FAILED);
  std::unique_ptr<void, decltype(release_reservation)> reservation(reserved, release_reservation);
  void *target = static_cast<std::byte *>(reserved) + kPage;
  const auto target_address = reinterpret_cast<uint64_t>(target);
  const auto unrelated_address = reinterpret_cast<uint64_t>(reserved);

  rj_vm_t *created = nullptr;
  ASSERT_EQ(rj_vm_create((std::string(CONFIG_DIR) + "/gfx1201_r9700.json").c_str(),
                         RJ_VM_MODE_LOCAL, &created),
            ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(created, rj_vm_destroy);
  auto *driver = vm->vm->driver();
  const uint32_t process_id = driver->local_process_id();
  auto process = driver->find_process(process_id);
  ASSERT_NE(process, nullptr);
  uint32_t gpu_id = driver->gpu_id();
  auto access = vm->soc->gpu_vm().snapshot_vmid(process_id);
  ASSERT_TRUE(access);

  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.va_addr = target_address;
  allocation.size = kPage;
  allocation.gpu_id = gpu_id;
  allocation.flags = flags | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  ASSERT_EQ(driver->mmap(target, kPage, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                         static_cast<off_t>(allocation.mmap_offset)),
            target);
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) {
    kfd_ioctl_ipc_export_handle_args exported{};
    exported.handle = allocation.handle;
    exported.gpu_id = gpu_id;
    ASSERT_EQ(driver->ioctl(AMDKFD_IOC_IPC_EXPORT_HANDLE, &exported), 0);
  }
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = allocation.handle;
  map.n_devices = 1;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map), 0);
  kfd_ioctl_free_memory_of_gpu_args free{};
  free.handle = allocation.handle;
  ASSERT_EQ(driver->ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free), 0);
  ASSERT_FALSE(process->gpu_access_grant(gpu_id, target_address, 1).allowed);
  ASSERT_TRUE(process->gpu_access_grant(gpu_id, unrelated_address, 1).allowed);
  ASSERT_EQ(process->retired_identity_views_.size(), 1u);

  const auto publication = process->gpu_grant_publication().load();
  uint64_t observed_publication = 0;
  void *mapped = MAP_FAILED;
  int mapping_error = 0;
  const auto offset =
      static_cast<off_t>(rocjitsu::KFD_MMAP_TYPE_DOORBELL | rocjitsu::kfd_mmap_gpu_id(gpu_id));
  {
    AllocationTrace trace(failure_index);
    mapped = driver->mmap_observing_host_vma_change(
        target, kPage, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, offset, [&] {
          observed_publication = process->gpu_grant_publication().load();
          trace.start();
        });
    mapping_error = errno;
  }
  allocations = allocation_count;
  // Host replacement is irreversible, but it must not publish an intermediate
  // permission change before the complete doorbell transaction succeeds.
  EXPECT_EQ(observed_publication, publication);
  EXPECT_TRUE(process->gpu_access_grant(gpu_id, unrelated_address, 1).allowed);
  if (failure_index) {
    ASSERT_TRUE(injected);
    EXPECT_EQ(mapped, MAP_FAILED);
    EXPECT_EQ(mapping_error, ENOMEM);
    EXPECT_EQ(process->gpu_grant_publication().load(), publication);
    EXPECT_FALSE(process->gpu_access_grant(gpu_id, target_address, 1).allowed);
    EXPECT_EQ(process->retired_identity_views_.size(), 1u);
    mapped = driver->mmap(target, kPage, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, offset);
  }
  ASSERT_EQ(mapped, target) << errno;
  EXPECT_EQ(process->gpu_grant_publication().load(), publication + 1);
  EXPECT_TRUE(process->retired_identity_views_.empty());
  EXPECT_TRUE(process->gpu_access_grant(gpu_id, target_address, 1).allowed);
  EXPECT_TRUE(process->gpu_access_grant(gpu_id, unrelated_address, 1).allowed);
  EXPECT_EQ(access->atomic_store(target_address, sizeof(uint64_t), 97),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(*static_cast<uint64_t *>(target), 97u);
  EXPECT_EQ(access->atomic_store(unrelated_address, sizeof(uint64_t), 91),
            rocjitsu::amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(*static_cast<uint64_t *>(reserved), 91u);
  EXPECT_EQ(driver->munmap(target, kPage), 0);
  access.reset();
  process.reset();
  ASSERT_EQ(rj_vm_destroy_checked(vm.get()), ROCJITSU_STATUS_SUCCESS);
  (void)vm.release();
}

TEST(KfdMmapAllocationFailureTest, FixedDoorbellReplacementPublishesOnceAndRetries) {
  for (uint32_t flags : {KFD_IOC_ALLOC_MEM_FLAGS_VRAM, KFD_IOC_ALLOC_MEM_FLAGS_GTT,
                         KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
                         KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC}) {
    SCOPED_TRACE(flags);
    size_t allocations = 0;
    ASSERT_NO_FATAL_FAILURE(check_replacement(flags, 0, allocations));
    ASSERT_GT(allocations, 0u);
    for (size_t failure_index = 1; failure_index <= allocations; ++failure_index) {
      size_t attempted = 0;
      ASSERT_NO_FATAL_FAILURE(check_replacement(flags, failure_index, attempted));
    }
  }
}

} // namespace
