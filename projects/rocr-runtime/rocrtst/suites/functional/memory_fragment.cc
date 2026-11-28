/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "suites/functional/memory_fragment.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// Helpers local to this translation unit (see repo rule:
// "no new headers for local definitions").
namespace {

// The fragment sub-allocator carves requests out of 2 MB backing blocks.
// Requests at or below this threshold are guaranteed to be sub-allocated
// (rather than promoted to a dedicated BO), which is exactly the path we
// want to exercise here.
constexpr size_t kFragmentBlockSize = 2 * 1024 * 1024;

// A system request above one block is not sub-allocated: it gets a dedicated
// allocation of the requested size. The zero-init and coherence tests also run
// at this size so that the hand-off between the two paths is covered.
constexpr size_t kAboveBlockSize = kFragmentBlockSize + 4096;

// Deterministic, per-buffer/per-offset fingerprint. Distinct (id, offset)
// pairs produce distinct bytes, so if two live fragments ever overlap, the
// later writer clobbers the earlier one and read-back verification fails.
inline uint8_t FragByte(uint32_t id, size_t offset) {
  uint32_t v = id * 2654435761u + static_cast<uint32_t>(offset) * 40503u + 0x9e37u;
  return static_cast<uint8_t>(v ^ (v >> 8) ^ (v >> 16));
}

// The runtime disables the fragment sub-allocator when
// HSA_DISABLE_FRAGMENT_ALLOCATOR=1 (see core/util/flag.h). With it disabled,
// allocations are served by dedicated BOs rather than sub-allocated fragments,
// so the overlap / zero-init / coherence checks below no longer exercise their
// intended path; skip instead of reporting misleading coverage.
inline bool FragmentAllocatorDisabled() {
  const char* v = getenv("HSA_DISABLE_FRAGMENT_ALLOCATOR");
  return v != nullptr && std::string(v) == "1";
}

// Value the coherence kernel atomically adds into each element.
constexpr int kValue = 7;

// Kernarg layout for test_atomic_add (see atomicOperations_kernels.cl):
//   (int* sysMemory, int* gpuMemory, int* oldValues, int value)
// The kernel is grid-size bounded (one work-item per element, no count arg),
// so no element-count field is needed here.
typedef struct __attribute__((aligned(16))) args_t {
  int* a;
  int* b;
  int* c;
  int d;
} args;

}  // namespace

MemoryFragment::MemoryFragment() : TestBase() {
  set_num_iteration(10);
  set_title("RocR System-Memory Fragment Sub-allocation Test");
  set_description(
      "Validates the fine-grain system-memory fragment sub-allocator: that "
      "distinct fragments never overlap or clobber each other (including "
      "across free/re-allocate churn), that fresh and recycled fragments are "
      "zero-initialized, and that GPU writes to a sub-allocated fragment are "
      "coherent and visible to the CPU.");
}

MemoryFragment::~MemoryFragment() {}

void MemoryFragment::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
  return;
}

void MemoryFragment::Run(void) {
  if (!rocrtst::CheckProfile(this)) return;
  TestBase::Run();
}

void MemoryFragment::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void MemoryFragment::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) return;
  return;
}

void MemoryFragment::Close(void) { TestBase::Close(); }

// Find the fine-grain system pool on the first CPU agent, or skip the test
// if the platform does not expose one.
static bool GetFineGrainSystemPool(hsa_amd_memory_pool_t* pool_out) {
  std::vector<hsa_agent_t> cpus;
  hsa_status_t err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  if (cpus.empty()) return false;

  // rocrtst::FindFineGrainedPool selects a fine-grain (coherent), non-kernarg
  // global pool; on the CPU agent that is the fine-grain system pool backing
  // hipHostMalloc. It reports success via HSA_STATUS_INFO_BREAK and leaves the
  // handle untouched if no such pool exists.
  hsa_amd_memory_pool_t pool = {};
  err = hsa_amd_agent_iterate_memory_pools(cpus[0], rocrtst::FindFineGrainedPool,
                                           &pool);
  if (err != HSA_STATUS_INFO_BREAK) {
    if (err != HSA_STATUS_SUCCESS) {
      EXPECT_TRUE(false) << "Error iterating CPU memory pools: " << err;
    }
    return false;
  }
  *pool_out = pool;
  return true;
}

// Find the (coarse-grain) global device/VRAM pool on a GPU agent. Device pools
// are coarse-grain; the sub-allocator has always been enabled for them, so this
// extends coverage to the pre-existing VRAM path in addition to system memory.
static bool GetDevicePool(hsa_agent_t gpuAgent,
                          hsa_amd_memory_pool_t* pool_out) {
  hsa_amd_memory_pool_t pool = {};
  hsa_status_t err = hsa_amd_agent_iterate_memory_pools(
      gpuAgent, rocrtst::GetGlobalMemoryPool, &pool);
  if (err != HSA_STATUS_SUCCESS || pool.handle == 0) return false;
  *pool_out = pool;
  return true;
}

// Whether the given agent may directly access allocations from the pool. Device
// (VRAM) memory is only host-accessible on large-BAR systems; where it is not,
// the CPU-driven overlap/coherence checks below are skipped.
static bool PoolAccessibleByAgent(hsa_agent_t agent,
                                  hsa_amd_memory_pool_t pool) {
  hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  hsa_status_t err = hsa_amd_agent_memory_pool_get_info(
      agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  return err == HSA_STATUS_SUCCESS &&
         access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
}

void MemoryFragment::FragmentOverlapTest(void) {
  if (FragmentAllocatorDisabled()) {
    // The vendored gtest predates GTEST_SKIP(), so emit the conventional
    // grep-able "[ SKIPPED ]" marker instead of recording a real skip.
    std::cout << "[ SKIPPED ] HSA_DISABLE_FRAGMENT_ALLOCATOR=1; fragment "
                 "sub-allocator disabled." << std::endl;
    return;
  }

  std::vector<hsa_agent_t> cpus;
  ASSERT_EQ(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(cpus.size(), 0u);

  // System memory (fine-grain): host-accessible, no CPU-access grant needed.
  hsa_amd_memory_pool_t sys_pool;
  if (GetFineGrainSystemPool(&sys_pool)) {
    FragmentOverlapOnPool("fine-grain system", sys_pool,
                          /*grant_cpu=*/false, cpus[0], /*light=*/false);
  } else if (verbosity() > 0) {
    std::cout << "  No fine-grain system pool; skipping system case."
              << std::endl;
  }

  // Device (VRAM, coarse-grain): only runs where the CPU can access VRAM
  // (large-BAR), since the checks read/write fragments from the host.
  std::vector<hsa_agent_t> gpus;
  ASSERT_EQ(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus),
            HSA_STATUS_SUCCESS);
  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t dev_pool;
    if (!GetDevicePool(gpus[i], &dev_pool)) continue;
    if (!PoolAccessibleByAgent(cpus[0], dev_pool)) {
      if (verbosity() > 0) {
        std::cout << "  Device pool not host-accessible on GPU " << i
                  << " (non-large-BAR); skipping device case." << std::endl;
      }
      continue;
    }
    FragmentOverlapOnPool("device/VRAM", dev_pool, /*grant_cpu=*/true, cpus[0],
                          /*light=*/true);
  }
}

void MemoryFragment::FragmentOverlapOnPool(const char* label,
                                           hsa_amd_memory_pool_t pool,
                                           bool grant_cpu, hsa_agent_t cpu_agent,
                                           bool light) {
  if (verbosity() > 0) {
    std::cout << "  Fragment overlap on " << label << " memory" << std::endl;
  }

  // A spread of sizes, all below the sub-allocator block size, so that many
  // fragments are packed into shared backing blocks. The lighter workload keeps
  // host-side verification over PCIe BAR (device memory) reasonably fast.
  const size_t kSizesFull[] = {64,   128,  256,   512,   1024,
                               2048, 4096, 16384, 65536, 131072};
  const size_t kSizesLight[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
  const size_t* kSizes = light ? kSizesLight : kSizesFull;
  const size_t kNumSizes =
      light ? sizeof(kSizesLight) / sizeof(kSizesLight[0])
            : sizeof(kSizesFull) / sizeof(kSizesFull[0]);
  const int kNumBufs = light ? 128 : 512;

  struct Buf {
    uint8_t* ptr;
    size_t size;
    uint32_t id;
  };
  std::vector<Buf> bufs(kNumBufs);
  uint32_t next_id = 1;

  auto fill = [](const Buf& b) {
    for (size_t i = 0; i < b.size; ++i) b.ptr[i] = FragByte(b.id, i);
  };
  auto verify = [](const Buf& b) {
    for (size_t i = 0; i < b.size; ++i) {
      ASSERT_EQ(b.ptr[i], FragByte(b.id, i))
          << "Fragment content mismatch (overlap/clobber) in buffer id "
          << b.id << " at offset " << i;
    }
  };
  auto check_no_overlap = [](std::vector<Buf> v) {
    std::sort(v.begin(), v.end(), [](const Buf& x, const Buf& y) {
      return x.ptr < y.ptr;
    });
    for (size_t i = 1; i < v.size(); ++i) {
      const uint8_t* prev_end = v[i - 1].ptr + v[i - 1].size;
      ASSERT_LE(prev_end, v[i].ptr)
          << "Overlapping fragments: buffer id " << v[i - 1].id << " ["
          << static_cast<const void*>(v[i - 1].ptr) << ", "
          << static_cast<const void*>(prev_end) << ") overlaps buffer id "
          << v[i].id << " starting at " << static_cast<const void*>(v[i].ptr);
    }
  };

  auto grant = [&](uint8_t* p) {
    if (grant_cpu) {
      ASSERT_EQ(hsa_amd_agents_allow_access(1, &cpu_agent, NULL, p),
                HSA_STATUS_SUCCESS);
    }
  };

  // Phase 1: allocate everything, then fill, then verify.
  for (int i = 0; i < kNumBufs; ++i) {
    bufs[i].size = kSizes[i % kNumSizes];
    ASSERT_LE(bufs[i].size, kFragmentBlockSize);
    bufs[i].id = next_id++;
    hsa_status_t err = hsa_amd_memory_pool_allocate(
        pool, bufs[i].size, 0, reinterpret_cast<void**>(&bufs[i].ptr));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << "Allocation " << i << " failed";
    ASSERT_NE(bufs[i].ptr, nullptr);
    grant(bufs[i].ptr);
  }
  check_no_overlap(bufs);
  for (const auto& b : bufs) fill(b);
  for (const auto& b : bufs) verify(b);

  // Phase 2: churn. Free the even-indexed fragments, then re-allocate the
  // same slots with fresh ids/patterns. This recycles backing storage and is
  // the classic trigger for stale-mapping / overlap regressions. The odd
  // fragments stay live throughout and must remain intact.
  for (int i = 0; i < kNumBufs; i += 2) {
    hsa_status_t err = hsa_amd_memory_pool_free(bufs[i].ptr);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    bufs[i].ptr = nullptr;
  }
  for (int i = 0; i < kNumBufs; i += 2) {
    bufs[i].size = kSizes[(i + 3) % kNumSizes];
    bufs[i].id = next_id++;
    hsa_status_t err = hsa_amd_memory_pool_allocate(
        pool, bufs[i].size, 0, reinterpret_cast<void**>(&bufs[i].ptr));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS) << "Re-allocation " << i << " failed";
    ASSERT_NE(bufs[i].ptr, nullptr);
    grant(bufs[i].ptr);
    fill(bufs[i]);
  }
  check_no_overlap(bufs);
  for (const auto& b : bufs) verify(b);

  for (auto& b : bufs) {
    if (b.ptr) {
      hsa_status_t err = hsa_amd_memory_pool_free(b.ptr);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    }
  }
}

void MemoryFragment::FragmentZeroInitTest(void) {
  if (FragmentAllocatorDisabled()) {
    // The vendored gtest predates GTEST_SKIP(), so emit the conventional
    // grep-able "[ SKIPPED ]" marker instead of recording a real skip.
    std::cout << "[ SKIPPED ] HSA_DISABLE_FRAGMENT_ALLOCATOR=1; fragment "
                 "sub-allocator disabled." << std::endl;
    return;
  }

  // Zero-initialization of recycled fragments is a system-memory guarantee
  // only: the runtime zeroes system fragments to match dedicated-BO semantics,
  // but intentionally leaves device (VRAM) fragments untouched. This test is
  // therefore not extended to device memory.
  hsa_amd_memory_pool_t pool;
  if (!GetFineGrainSystemPool(&pool)) {
    if (verbosity() > 0) {
      std::cout << "  No fine-grain system pool; skipping." << std::endl;
    }
    return;
  }

  // Repeatedly allocate -> assert zeroed -> dirty -> free with a fixed size.
  // A buffer that still carries the previous (0xAB) data fails the zero check,
  // catching missing zero-initialization. Each size reaches its memory by a
  // different path:
  //   - one page: the sub-allocator hands back the same recycled fragment;
  //   - exactly one block: the freed block comes back from the block cache;
  //   - above one block: a dedicated allocation whose pages come from the OS.
  const struct {
    size_t size;
    int iters;
  } kCases[] = {{4096, 128}, {kFragmentBlockSize, 16}, {kAboveBlockSize, 16}};

  for (const auto& c : kCases) {
    for (int it = 0; it < c.iters; ++it) {
      uint8_t* p = nullptr;
      hsa_status_t err = hsa_amd_memory_pool_allocate(
          pool, c.size, 0, reinterpret_cast<void**>(&p));
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      ASSERT_NE(p, nullptr);

      // One assertion per buffer: a per-byte ASSERT is slow at these sizes.
      size_t first_nonzero = 0;
      while (first_nonzero < c.size && p[first_nonzero] == 0) ++first_nonzero;
      ASSERT_EQ(first_nonzero, c.size)
          << "Buffer of " << c.size << " bytes not zero-initialized at iteration "
          << it << ", offset " << first_nonzero;

      memset(p, 0xAB, c.size);

      err = hsa_amd_memory_pool_free(p);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    }
  }
}

void MemoryFragment::FragmentCoherenceOnPool(
    const char* label, hsa_agent_t cpuAgent, hsa_agent_t gpuAgent,
    hsa_amd_memory_pool_t under_test_pool, bool grant_cpu, int num_elements,
    hsa_amd_memory_pool_t kernarg_pool) {
  hsa_status_t err;

  if (verbosity() > 0) {
    std::cout << "  Fragment coherence on " << label << " memory, "
              << num_elements * sizeof(int) << " bytes" << std::endl;
  }

  // A standard (coarse-grain) system pool for scratch buffers the kernel also
  // writes but which we do not verify.
  hsa_amd_memory_pool_t global_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetGlobalMemoryPool,
                                           &global_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  const bool device_kernargs = kernarg_pool.handle != 0;
  if (!device_kernargs) {
    err =
        hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetKernArgMemoryPool, &kernarg_pool);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  uint32_t queue_size = 0;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  hsa_queue_t* queue = NULL;
  err = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                         0, 0, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  const int kNumElements = num_elements;
  const size_t kBufBytes = kNumElements * sizeof(int);

  // sysMemory is the buffer under test, from the pool under test (fine-grain
  // system or device/VRAM): a sub-allocated fragment, or a dedicated
  // allocation when kBufBytes is above one block. The GPU atomically adds
  // kValue to each element; the CPU must observe the result after the packet's
  // system-scope release.
  int* sysMemory = NULL;
  err = hsa_amd_memory_pool_allocate(under_test_pool, kBufBytes, 0,
                                     reinterpret_cast<void**>(&sysMemory));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Device (VRAM) fragments need an explicit CPU-access grant before the host
  // can zero and later verify them; system fragments are already accessible.
  if (grant_cpu) {
    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, sysMemory);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  int* gpuScratch = NULL;
  err = hsa_amd_memory_pool_allocate(global_pool, kBufBytes, 0,
                                     reinterpret_cast<void**>(&gpuScratch));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  int* oldValues = NULL;
  err = hsa_amd_memory_pool_allocate(global_pool, kBufBytes, 0,
                                     reinterpret_cast<void**>(&oldValues));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  set_kernel_file_name("atomicOperations_kernels.hsaco");
  set_kernel_name("test_atomic_add");

  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // The kernarg segment also holds hidden arguments after args_t, such as the
  // global offsets. They must read as zero, and only system fragments come
  // back zeroed, so size the buffer from the kernel and clear all of it.
  const size_t kernarg_bytes = std::max<size_t>(kernarg_size(), sizeof(args_t));
  args* kernArguments = NULL;
  err = hsa_amd_memory_pool_allocate(kernarg_pool, kernarg_bytes, 0,
                                     reinterpret_cast<void**>(&kernArguments));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Start from a known zero state so this test isolates coherence (fragment
  // zero-init is covered separately by FragmentZeroInitTest).
  memset(sysMemory, 0, kBufBytes);
  memset(gpuScratch, 0, kBufBytes);
  memset(oldValues, 0, kBufBytes);

  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, sysMemory);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, gpuScratch);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, oldValues);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArguments);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (device_kernargs) {
    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, kernArguments);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  memset(kernArguments, 0, kernarg_bytes);
  kernArguments->a = sysMemory;
  kernArguments->b = gpuScratch;
  kernArguments->c = oldValues;
  kernArguments->d = kValue;
  if (device_kernargs) {
    // Reading back flushes the posted PCIe writes of the arguments ahead of
    // the dispatch.
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    ASSERT_EQ(*reinterpret_cast<volatile int*>(&kernArguments->d), kValue);
  }

  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  aql().workgroup_size_x = 256;
  aql().workgroup_size_y = 1;
  aql().workgroup_size_z = 1;
  aql().grid_size_x = kNumElements;
  aql().kernarg_address = kernArguments;
  aql().kernel_object = kernel_object();

  const uint32_t queue_mask = queue->size - 1;
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  rocrtst::WriteAQLToQueueLoc(queue, index, &aql());

  aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  aql().header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  aql().header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

  void* q_base = queue->base_address;
  rocrtst::AtomicSetPacketHeader(
      aql().header, aql().setup,
      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(q_base))[index &
                                                                queue_mask]);

  hsa_signal_store_relaxed(queue->doorbell_signal, index);

  while (hsa_signal_wait_scacquire(aql().completion_signal,
                                   HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }
  hsa_signal_store_relaxed(aql().completion_signal, 1);

  // GPU->CPU coherence: every element must reflect the GPU's atomic add, and
  // the returned "old" values must be the zeros the CPU wrote before dispatch.
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(sysMemory[i], kValue)
        << "GPU write not coherent/visible to CPU at element " << i;
    ASSERT_EQ(oldValues[i], 0)
        << "Unexpected pre-add value observed by GPU at element " << i;
  }

  if (sysMemory) {
    err = hsa_amd_memory_pool_free(sysMemory);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (gpuScratch) {
    err = hsa_amd_memory_pool_free(gpuScratch);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (oldValues) {
    err = hsa_amd_memory_pool_free(oldValues);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (kernArguments) {
    err = hsa_amd_memory_pool_free(kernArguments);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (queue) {
    err = hsa_queue_destroy(queue);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
}

void MemoryFragment::FragmentCoherenceTest(hsa_agent_t cpuAgent,
                                           hsa_agent_t gpuAgent) {
  // One workgroup of elements: every buffer is a fragment.
  const int kFragmentElements = 256;

  // System memory (fine-grain): host-accessible, no CPU-access grant needed.
  hsa_amd_memory_pool_t sys_pool;
  if (GetFineGrainSystemPool(&sys_pool)) {
    FragmentCoherenceOnPool("fine-grain system", cpuAgent, gpuAgent, sys_pool,
                            /*grant_cpu=*/false, kFragmentElements);

    // Above one block the system buffers are dedicated allocations that the
    // GPU reaches only through hsa_amd_agents_allow_access. Skipped on
    // emulators, where a grid this large is slow.
    if (!rocrtst::isEmuModeEnabled()) {
      FragmentCoherenceOnPool("fine-grain system (above one block)", cpuAgent,
                              gpuAgent, sys_pool, /*grant_cpu=*/false,
                              static_cast<int>(kAboveBlockSize / sizeof(int)));
    }
  } else if (verbosity() > 0) {
    std::cout << "  No fine-grain system pool; skipping system case."
              << std::endl;
  }

  // Device (VRAM, coarse-grain): only where the CPU can read back VRAM.
  hsa_amd_memory_pool_t dev_pool;
  if (GetDevicePool(gpuAgent, &dev_pool)) {
    if (PoolAccessibleByAgent(cpuAgent, dev_pool)) {
      FragmentCoherenceOnPool("device/VRAM", cpuAgent, gpuAgent, dev_pool,
                              /*grant_cpu=*/true, kFragmentElements);
    } else if (verbosity() > 0) {
      std::cout << "  Device pool not host-accessible (non-large-BAR); "
                   "skipping device case." << std::endl;
    }
  }
}

void MemoryFragment::FragmentCoherenceTest(void) {
  if (FragmentAllocatorDisabled()) {
    // The vendored gtest predates GTEST_SKIP(), so emit the conventional
    // grep-able "[ SKIPPED ]" marker instead of recording a real skip.
    std::cout << "[ SKIPPED ] HSA_DISABLE_FRAGMENT_ALLOCATOR=1; fragment "
                 "sub-allocator disabled." << std::endl;
    return;
  }

  hsa_status_t err;
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(cpus.size(), 0u);

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    FragmentCoherenceTest(cpus[0], gpus[i]);
  }
}

// Find the device kernarg pool on a GPU agent. Discrete GPUs publish it; APUs
// and older runtimes do not.
static bool GetDeviceKernargPool(hsa_agent_t gpuAgent, hsa_amd_memory_pool_t* pool_out) {
  hsa_amd_memory_pool_t pool = {};
  hsa_status_t err = hsa_amd_agent_iterate_memory_pools(
      gpuAgent,
      [](hsa_amd_memory_pool_t p, void* data) -> hsa_status_t {
        uint32_t flags = 0;
        hsa_status_t err =
            hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (err != HSA_STATUS_SUCCESS) return err;
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
          *static_cast<hsa_amd_memory_pool_t*>(data) = p;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &pool);
  if (err != HSA_STATUS_INFO_BREAK) return false;
  *pool_out = pool;
  return true;
}

void MemoryFragment::FragmentDeviceKernargTest(void) {
  if (FragmentAllocatorDisabled()) {
    // The vendored gtest predates GTEST_SKIP(), so emit the conventional
    // grep-able "[ SKIPPED ]" marker instead of recording a real skip.
    std::cout << "[ SKIPPED ] HSA_DISABLE_FRAGMENT_ALLOCATOR=1; fragment "
                 "sub-allocator disabled."
              << std::endl;
    return;
  }

  std::vector<hsa_agent_t> cpus;
  ASSERT_EQ(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus), HSA_STATUS_SUCCESS);
  ASSERT_GT(cpus.size(), 0u);

  std::vector<hsa_agent_t> gpus;
  ASSERT_EQ(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus), HSA_STATUS_SUCCESS);

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t pool;
    if (!GetDeviceKernargPool(gpus[i], &pool)) {
      if (verbosity() > 0) {
        std::cout << "  No device kernarg pool on GPU " << i << "; skipping." << std::endl;
      }
      continue;
    }

    // The pool reports no grain flag so that callers picking a device pool by
    // grain never select it, and it is not published as a region because hsa.h
    // requires KERNARG regions to be fine grained.
    uint32_t flags = 0;
    ASSERT_EQ(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(flags, static_cast<uint32_t>(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT));
    std::pair<uint64_t, bool> region_search = {pool.handle, false};
    hsa_status_t err = hsa_agent_iterate_regions(
        gpus[i],
        [](hsa_region_t region, void* data) -> hsa_status_t {
          auto* search = static_cast<std::pair<uint64_t, bool>*>(data);
          if (region.handle == search->first) search->second = true;
          return HSA_STATUS_SUCCESS;
        },
        &region_search);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    EXPECT_FALSE(region_search.second)
        << "Device kernarg pool is listed by hsa_agent_iterate_regions";

    if (!PoolAccessibleByAgent(cpus[0], pool)) {
      if (verbosity() > 0) {
        std::cout << "  Device kernarg pool not host-accessible on GPU " << i
                  << " (non-large-BAR); skipping." << std::endl;
      }
      continue;
    }

    // A dedicated buffer object costs at least one 2 MB block of device
    // memory, so kCount small requests would charge kCount blocks without
    // sub-allocation. Sub-allocated, they fit in one block, or two if an
    // earlier user left a partly used one.
    const int kCount = 32;
    const size_t kSize = 4096;
    uint64_t avail_before = 0, avail_after = 0;
    ASSERT_EQ(hsa_agent_get_info(gpus[i], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                                 &avail_before),
              HSA_STATUS_SUCCESS);

    std::vector<void*> ptrs(kCount, nullptr);
    for (int n = 0; n < kCount; ++n) {
      ASSERT_EQ(hsa_amd_memory_pool_allocate(pool, kSize, 0, &ptrs[n]), HSA_STATUS_SUCCESS);
      ASSERT_EQ(hsa_amd_agents_allow_access(1, &cpus[0], NULL, ptrs[n]), HSA_STATUS_SUCCESS);

      hsa_amd_pointer_info_t info = {};
      info.size = sizeof(info);
      ASSERT_EQ(hsa_amd_pointer_info(ptrs[n], &info, NULL, NULL, NULL), HSA_STATUS_SUCCESS);
      EXPECT_NE(info.alloc_flags & HSA_AMD_POINTER_INFO_ALLOC_FLAG_EXECUTABLE, 0u)
          << "Device kernarg allocation " << n << " is not executable";
      // The pointer must report the same pool flag as the pool itself.
      EXPECT_NE(info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, 0u)
          << "Device kernarg allocation " << n << " does not report KERNARG_INIT";
    }

    ASSERT_EQ(hsa_agent_get_info(gpus[i], (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                                 &avail_after),
              HSA_STATUS_SUCCESS);
    const uint64_t charged = avail_before > avail_after ? avail_before - avail_after : 0;
    if (verbosity() > 0) {
      std::cout << "  GPU " << i << ": " << kCount << " x " << kSize
                << " B from the device kernarg pool charged " << charged / 1024
                << " KB of device memory" << std::endl;
    }
    EXPECT_LE(charged, 2 * kFragmentBlockSize)
        << kCount << " small device kernarg allocations were not sub-allocated";

    for (void* p : ptrs) ASSERT_EQ(hsa_amd_memory_pool_free(p), HSA_STATUS_SUCCESS);

    FragmentOverlapOnPool("device kernarg", pool, /*grant_cpu=*/true, cpus[0],
                          /*light=*/true);

    hsa_amd_memory_pool_t sys_pool;
    if (GetFineGrainSystemPool(&sys_pool)) {
      FragmentCoherenceOnPool("fine-grain system, device kernargs", cpus[0], gpus[i], sys_pool,
                              /*grant_cpu=*/false, /*num_elements=*/256, pool);
    }
  }
}
