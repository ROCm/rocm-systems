// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file xcd_fanout_kfd_test.cpp
/// @brief XCD fan-out driven through the KFD ioctl surface rather than a
/// directly registered test queue.
///
/// Every other fan-out regression registers an AqlQueue straight against a
/// command processor: vmid 0, no process page table, and a kernel that touches
/// no private memory. This brings the driver up instead, creates a compute queue
/// through CREATE_QUEUE -- the path that sets HwQueue::xcd_fanout -- and
/// dispatches a kernel that really does spill, behind a real process page table.
///
/// What that reaches, and what it does not: the scratch *addressing* contract is
/// covered here, because a wave's slot is indexed by its grid-wide workgroup id
/// and the high-rank peers run the top of the grid. The driver's on-demand
/// scratch-backing allocator is not, and cannot be from a local-mode test:
/// opening a process turns GpuMemory passthrough on, so a wave's scratch address
/// always resolves and the command processor never asks the driver to back it.
/// That allocator -- and the existing-allocation check that keeps eight XCDs of
/// one grid from each mapping the same pool -- is a daemon-mode path, where the
/// guest's addresses are not the daemon's and passthrough stays off. Covering it
/// needs a daemon-mode regression, not this fixture.

#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
#include "hsa/amd_hsa_queue.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <vector>

#ifdef HAS_DEVICE_KERNELS

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = test::config_path("gfx950_mi355x.json");
using test::kernel_path;

constexpr uint32_t kGpuId = 38144;
constexpr uint32_t kTotalXcds = 8;
constexpr uint32_t kCusPerXcd = 36;
constexpr uint32_t kTotalCus = kTotalXcds * kCusPerXcd;
constexpr uint32_t kWavefrontSize = 64;
// One workgroup per CU, so every XCD takes an equal share and the top of the
// grid -- the part a share-sized scratch pool would leave unbacked -- lands on
// the highest-rank peer.
constexpr uint32_t kGridWgs = kTotalCus;
constexpr uint32_t kGridItems = kGridWgs * kWavefrontSize;

/// Driver, VM and engine brought up together, with the engine driven on its own
/// thread so the doorbell poll thread has something to hand packets to.
struct KfdFanoutFixture {
  config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;
  SimulatedKfd *driver = nullptr;
  std::thread sim_thread;

  KfdFanoutFixture() : loaded(config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema)) {
    // Both taken before take_root(): the loader owns the tree until then, and
    // its accessors stop resolving once ownership has moved out.
    soc = loaded.soc();
    memory = loaded.memory();
    auto root = loaded.take_root();
    // No tick budget, and hold the engine open until a primary says otherwise:
    // work arrives from the doorbell poll thread, not from a pre-seeded queue.
    loaded.engine_config.max_ticks = 0;
    loaded.engine_config.await_primaries = true;
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);

    auto soc_root = std::unique_ptr<SoC>(static_cast<SoC *>(root.release()));
    auto vm = std::make_unique<VirtualMachine>(std::move(soc_root));
    driver = vm->driver();
    engine->topology().set_root(std::move(vm));
    loaded.wire_links(engine->topology());
    soc->wire_backing(engine->topology());
    engine->create();
    engine->register_as_primary();
  }

  void start() {
    sim_thread = std::thread([this] { engine->run(); });
  }

  ~KfdFanoutFixture() {
    if (sim_thread.joinable()) {
      engine->request_exit("test finished");
      sim_thread.join();
    }
    if (driver)
      driver->close();
  }
};

// One registration covers everything the dispatch touches. Page-aligned and
// page-granular because the process page table maps pages, not objects: a buffer
// sharing a page with something unmapped would be a mapping hazard rather than a
// property of the code under test.
constexpr size_t kPage = 4096;
constexpr size_t kImageOffset = 0;
constexpr size_t kImageSize = 1u << 20;
constexpr size_t kOutOffset = kImageOffset + kImageSize;
constexpr size_t kOutSize = 128u << 10; // kGridItems * sizeof(uint32_t), rounded up
constexpr size_t kKernargOffset = kOutOffset + kOutSize;
constexpr size_t kRingOffset = kKernargOffset + kPage;
constexpr size_t kRingSize = 8192;
// The queue descriptor is a real amd_queue_t rather than a bare pair of
// pointers. The driver derives the descriptor base by subtracting
// offsetof(amd_queue_t, write_dispatch_id) from the write pointer it is given,
// so a write pointer that is not actually inside one aims the command
// processor's descriptor accesses at whatever happens to precede it.
constexpr size_t kQueueDescOffset = kRingOffset + kRingSize;
constexpr size_t kSignalOffset = kQueueDescOffset + kPage;
constexpr size_t kArenaSize = kSignalOffset + kPage;
static_assert(sizeof(amd_queue_t) <= kPage);
static_assert(kOutSize >= kGridItems * sizeof(uint32_t));

struct alignas(kPage) Arena {
  std::array<uint8_t, kArenaSize> bytes;
};

// Private memory for the whole grid, supplied the way the runtime supplies it
// rather than left to the driver's on-demand fallback. That fallback only fires
// when a wave's scratch address fails to resolve, which cannot happen in local
// mode: opening a process turns GpuMemory passthrough on, so every user-space VA
// resolves whether or not the page table knows it. The fallback -- and the
// existing-allocation check that keeps eight XCDs from each mapping the pool --
// is therefore a daemon-mode path, where the guest's addresses are not the
// daemon's and passthrough stays off.
//
// 64 KiB aligned because SET_SCRATCH_BACKING_VA carries the base shifted by 16.
constexpr size_t kScratchAlign = 1u << 16;
constexpr size_t kScratchSize = 8u << 20;
struct alignas(kScratchAlign) ScratchArena {
  std::array<uint8_t, kScratchSize> bytes;
};

/// Satisfies load_to_memory()'s Memory concept by copying into the arena.
///
/// The arena is mapped 1:1, so a segment's load address is also its host
/// address and the loaded image is reachable at the dispatch's vmid rather than
/// only at vmid 0, where GpuMemory::load_image() would have put it.
struct HostImageLoader {
  uint8_t *base;
  size_t size;
  bool ok = true;

  void load_image(const uint8_t *src, size_t n, uint64_t addr) {
    const uint64_t offset = addr - reinterpret_cast<uint64_t>(base);
    if (offset > size || n > size - offset) {
      ok = false;
      return;
    }
    std::memcpy(base + offset, src, n);
  }
};

/// Register a host buffer with the driver so the GPU can reach it.
///
/// USERPTR with a caller-supplied address maps the process page table 1:1, so
/// the GPU VA and the host pointer are the same number and the test can seed and
/// read back memory directly.
bool map_userptr(SimulatedKfd *driver, uint32_t pid, void *host, size_t size) {
  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = reinterpret_cast<uint64_t>(host);
  alloc.size = size;
  alloc.gpu_id = kGpuId;
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  return driver->ioctl(pid, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc) == 0;
}

// A fanned-out dispatch of a kernel that spills to private memory, submitted
// through the KFD queue-creation path so it runs behind a real process page
// table.
//
// The claim FanoutSizesScratchForTheWholeGridNotOneShare cannot make. That one
// records what the allocator is *asked* for, with a kernel that never touches
// scratch, so it pins the arithmetic and nothing else. Here the shares actually
// spill and read back, and the value each work item produces depends only on its
// own global id -- so a wave landing on someone else's slot, or off the end of
// the pool, changes the answer. The high-index workgroups are the interesting
// ones: their slots are at the far end of a pool sized for the whole grid, which
// is precisely the region a share-sized pool would not cover.
TEST(XcdFanoutKfdTest, FanoutRunsAPrivateMemoryKernelAcrossEveryXcd) {
  Executable exec(kernel_path("scratch_spill_probe"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load scratch_spill_probe.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  KfdFanoutFixture fx;
  ASSERT_NE(fx.soc, nullptr);
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  fx.driver->setup_topology(fx.loaded.device, kTotalXcds);
  ASSERT_GE(fx.driver->open(), 0);
  const uint32_t pid = fx.driver->open_process();
  ASSERT_NE(pid, 0u);

  // Nominate where private memory lives, as the runtime does before its first
  // dispatch. Sized for the whole grid: a wave's slot is indexed by its grid-wide
  // workgroup id, so the shares the high-rank peers run address the far end of
  // this pool and a share-sized one would leave them writing past it.
  static ScratchArena scratch_arena{};
  std::memset(scratch_arena.bytes.data(), 0, scratch_arena.bytes.size());
  ASSERT_TRUE(map_userptr(fx.driver, pid, scratch_arena.bytes.data(), scratch_arena.bytes.size()));
  const auto scratch_pool_va = reinterpret_cast<uint64_t>(scratch_arena.bytes.data());
  ASSERT_EQ(scratch_pool_va % kScratchAlign, 0u);
  kfd_ioctl_set_scratch_backing_va_args scratch{};
  scratch.gpu_id = kGpuId;
  scratch.va_addr = scratch_pool_va >> 16;
  ASSERT_EQ(fx.driver->ioctl(pid, AMDKFD_IOC_SET_SCRATCH_BACKING_VA, &scratch), 0);

  // One page-aligned arena, mapped into the process in a single registration.
  // Everything the dispatch touches lives in it, so every GPU VA below resolves
  // through the process page table -- which is the whole point of running this
  // through KFD rather than against a directly registered queue.
  static Arena arena{};
  std::memset(arena.bytes.data(), 0, arena.bytes.size());
  ASSERT_TRUE(map_userptr(fx.driver, pid, arena.bytes.data(), arena.bytes.size()));

  auto *image = arena.bytes.data() + kImageOffset;
  auto *out = reinterpret_cast<uint32_t *>(arena.bytes.data() + kOutOffset);
  auto *ring = arena.bytes.data() + kRingOffset;
  auto *queue_desc = reinterpret_cast<amd_queue_t *>(arena.bytes.data() + kQueueDescOffset);
  auto *signal = reinterpret_cast<uint64_t *>(arena.bytes.data() + kSignalOffset);
  constexpr uint32_t kSignalValueIndex = 1; // amd_signal_t::value, 8 bytes in.
  constexpr uint64_t kInitialSignal = 1;

  // Load the code object into the arena rather than into GpuMemory at vmid 0:
  // the command processor fetches kernel code at the dispatch's own vmid.
  HostImageLoader loader{image, kImageSize};
  co->load_to_memory(&loader, reinterpret_cast<uint64_t>(image));
  ASSERT_TRUE(loader.ok) << "the code object does not fit the arena's image window";
  const uint64_t kernel_object =
      reinterpret_cast<uint64_t>(image) + co->kernel_descriptor_offset("scratch_spill_probe");
  ASSERT_NE(kernel_object, reinterpret_cast<uint64_t>(image));

  // The premise of the whole test: without a private segment the dispatch never
  // reaches the scratch path at all.
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  std::memcpy(&kd, reinterpret_cast<const void *>(kernel_object), sizeof(kd));
  ASSERT_GT(kd.private_segment_fixed_size, 0u)
      << "scratch_spill_probe was compiled without private memory";

  struct Kernarg {
    uint64_t out;
    uint32_t n;
    uint32_t workgroup_size;
  };
  auto *kernarg = reinterpret_cast<Kernarg *>(arena.bytes.data() + kKernargOffset);
  *kernarg = Kernarg{reinterpret_cast<uint64_t>(out), kGridItems, kWavefrontSize};
  std::atomic_ref<uint64_t>(signal[kSignalValueIndex]).store(kInitialSignal);

  // The doorbell page has to come from the driver: CREATE_QUEUE hands back an
  // offset into it, and the command processor only polls a queue whose doorbell
  // base has been published.
  const long page = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page, 0);
  const auto doorbell_page_size = static_cast<size_t>(page);
  void *doorbell_page =
      fx.driver->mmap(pid, nullptr, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                      static_cast<off_t>(KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(kGpuId)));
  ASSERT_NE(doorbell_page, MAP_FAILED);

  kfd_ioctl_create_queue_args create{};
  create.gpu_id = kGpuId;
  create.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  create.ring_base_address = reinterpret_cast<uint64_t>(ring);
  create.ring_size = static_cast<uint32_t>(kRingSize);
  create.read_pointer_address = reinterpret_cast<uint64_t>(&queue_desc->read_dispatch_id);
  create.write_pointer_address = reinterpret_cast<uint64_t>(&queue_desc->write_dispatch_id);
  create.queue_percentage = 100;
  ASSERT_EQ(fx.driver->ioctl(pid, AMDKFD_IOC_CREATE_QUEUE, &create), 0);

  // A compute queue created this way is replicated onto every XCD; without that
  // the dispatch would never fan out and there would be no race for the pool.
  size_t registered = 0;
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    registered += fx.soc->xcd(xi)->command_processor()->registered_queue_count_for_test();
  ASSERT_EQ(registered, kTotalXcds) << "the compute queue did not fan out across the XCDs";

  fx.start();

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH |
               (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
               (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt.setup = 1;
  pkt.workgroup_size_x = kWavefrontSize;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = kGridItems;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.private_segment_size = kd.private_segment_fixed_size;
  pkt.kernel_object = kernel_object;
  pkt.kernarg_address = kernarg;
  pkt.completion_signal.handle = reinterpret_cast<uint64_t>(signal);
  std::memcpy(ring, &pkt, sizeof(pkt));

  // Publish the packet, then ring the doorbell the driver gave us.
  std::atomic_ref<uint64_t>(const_cast<uint64_t &>(queue_desc->write_dispatch_id))
      .store(1, std::memory_order_release);
  auto *doorbell = reinterpret_cast<uint64_t *>(static_cast<char *>(doorbell_page) +
                                                (create.doorbell_offset % doorbell_page_size));
  std::atomic_ref<uint64_t>(*doorbell).store(1, std::memory_order_release);

  // Wait on the signal rather than on the engine: the dispatch retires when the
  // last XCD's share does, and only then is every result guaranteed visible.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::atomic_ref<uint64_t>(signal[kSignalValueIndex]).load(std::memory_order_acquire) !=
             kInitialSignal - 1 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_EQ(std::atomic_ref<uint64_t>(signal[kSignalValueIndex]).load(std::memory_order_acquire),
            kInitialSignal - 1)
      << "the fanned-out dispatch never completed";

  // The grid really was spread, or nothing above is about fan-out.
  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kGridWgs);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kGridWgs / kTotalXcds) << "xcd" << xi;

  // Each work item sums its own 64 private slots, so the expected value depends
  // only on its global id -- and is wrong for any item whose slots were not
  // backed, or were repointed mid-flight by a second mapping of the pool.
  unsigned mismatches = 0;
  uint32_t first_bad = 0;
  for (uint32_t gid = 0; gid < kGridItems; ++gid) {
    constexpr uint32_t kSlots = 64;
    uint32_t expected = 0;
    for (uint32_t i = 0; i < kSlots; ++i)
      expected += gid * kSlots + ((gid + i) % kSlots);
    if (out[gid] != expected) {
      if (mismatches == 0)
        first_bad = gid;
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " work items read back the wrong private data, first "
                            << "at gid " << first_bad << " (workgroup "
                            << first_bad / kWavefrontSize << ", xcd "
                            << (first_bad / kWavefrontSize) % kTotalXcds << ")";
}

} // namespace

#endif // HAS_DEVICE_KERNELS
