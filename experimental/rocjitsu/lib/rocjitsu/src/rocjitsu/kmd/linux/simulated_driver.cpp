// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/xcd.h"

#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace rocjitsu {

// KFD mmap offset encoding (mirrors kfd_priv.h).
constexpr uint64_t KFD_MMAP_TYPE_SHIFT = 62;
constexpr uint64_t KFD_MMAP_TYPE_MASK = 0x3ULL << KFD_MMAP_TYPE_SHIFT;
constexpr uint64_t KFD_MMAP_TYPE_DOORBELL = 0x3ULL << KFD_MMAP_TYPE_SHIFT;
constexpr uint64_t KFD_MMAP_TYPE_EVENTS = 0x2ULL << KFD_MMAP_TYPE_SHIFT;
constexpr uint64_t KFD_MMAP_GPU_ID_SHIFT = 46;
constexpr const char *const KFD_SYSFS_PREFIX = "/sys/devices/virtual/kfd/kfd/topology";

namespace {

constexpr uint64_t kfd_mmap_gpu_id(uint32_t gpu_id) {
  return (static_cast<uint64_t>(gpu_id) << KFD_MMAP_GPU_ID_SHIFT) &
         ((1ULL << KFD_MMAP_TYPE_SHIFT) - (1ULL << KFD_MMAP_GPU_ID_SHIFT));
}

// Owned state for the default driver (lives as long as the SimulatedDriver).
struct DefaultDriverState {
  config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  std::jthread engine_thread;

  ~DefaultDriverState() {
    if (engine)
      engine->request_exit("driver shutdown");
    // Don't join — the engine thread may be in a state that prevents clean
    // shutdown (e.g., blocked in a KFD ioctl wait). Detach and let the OS
    // clean up on process exit. This is safe because DefaultDriverState is
    // file-static and only destroyed during program termination.
    if (engine_thread.joinable())
      engine_thread.detach();
  }
};

// Intentionally leaked. The driver state must outlive everything — including
// __cxa_finalize which runs during shared library unload. If we use a
// unique_ptr, __cxa_finalize destroys it before ROCR finishes, causing
// topology files to be deleted while ROCR is reading them.
static DefaultDriverState *g_default_state = nullptr;

} // namespace

// -- Static singleton state --
// Intentionally leaked — must survive __cxa_finalize during library unload.
// Both atomics so lookup(), kfd_fd(), and redirect_sysfs_path() can read them
// from any thread without holding g_mutex.
static std::atomic<SimulatedDriver *> g_instance{nullptr};
static std::atomic<int> g_kfd_fd{-1};
static std::mutex g_mutex;
static thread_local bool g_in_construction = false;

SimulatedDriver *SimulatedDriver::get_or_create() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_instance) {
    g_in_construction = true;
    auto driver = create_default();
    g_in_construction = false;
    if (!driver)
      return nullptr;
    g_instance.store(driver.release(), std::memory_order_release); // Intentionally leaked.
    g_instance.load(std::memory_order_relaxed)->open();            // Sets g_kfd_fd via open().
  }
  return g_instance;
}

SimulatedDriver *SimulatedDriver::lookup(int fd) {
  auto *inst = g_instance.load(std::memory_order_acquire);
  return (fd >= 0 && fd == g_kfd_fd.load(std::memory_order_acquire) && inst) ? inst : nullptr;
}

int SimulatedDriver::kfd_fd() { return g_kfd_fd.load(std::memory_order_acquire); }

std::string SimulatedDriver::redirect_sysfs_path(const char *path) {
  auto *inst = g_instance.load(std::memory_order_acquire);
  if (!inst)
    return {};
  size_t prefix_len = std::strlen(KFD_SYSFS_PREFIX);
  if (std::strncmp(path, KFD_SYSFS_PREFIX, prefix_len) != 0)
    return {};
  return inst->topology_path() + (path + prefix_len);
}

bool SimulatedDriver::in_construction() { return g_in_construction; }

std::unique_ptr<SimulatedDriver> SimulatedDriver::create_default() {
  const char *config_path = getenv("RJ_CONFIG");
  const char *schema_path = getenv("RJ_SCHEMA");
  if (!config_path || !schema_path) {
    fprintf(stderr, "rocjitsu: RJ_CONFIG and RJ_SCHEMA env vars required\n");
    return nullptr;
  }

  auto state = std::make_unique<DefaultDriverState>();
  try {
    state->loaded = config::load_config(config_path, schema_path);
  } catch (const std::exception &e) {
    fprintf(stderr, "rocjitsu: config load failed: %s\n", e.what());
    return nullptr;
  }
  auto *soc = state->loaded.soc();
  // Override max_ticks: the KFD driver runs the engine indefinitely, waiting for
  // doorbell events from ROCR. Termination is controlled by open()/close().
  state->loaded.engine_config.max_ticks = 0;
  state->loaded.engine_config.await_primaries = true;
  state->engine = std::make_unique<simdojo::SimulationEngine>(state->loaded.engine_config);
  state->engine->topology().set_root(state->loaded.take_root());
  state->loaded.wire_links(state->engine->topology());
  // Wire L2→HBM backing store links (must happen after set_memory populates
  // the standalone HBM controller, and before the engine runs).
  soc->wire_backing(state->engine->topology());
  state->engine->build();

  // Start the simulation engine in a background thread. The engine runs
  // continuously, processing events (doorbells, CU work) as they arrive.
  // The engine stays alive via max_ticks (set to a large value) and exits
  // when the driver calls request_exit on shutdown.
  state->engine_thread = std::jthread([&engine = *state->engine]() { engine.run(); });

  auto driver = std::make_unique<SimulatedDriver>(*state->engine, *soc);

  // Set up sysfs topology for ROCR discovery.
  // These defaults match gfx950; other ISAs override via config.
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = 0x9500;
  gpu.gfx_target_version = 90500;
  gpu.simd_count = soc->num_xcds() * 32 * 4; // CUs * SIMDs/CU
  gpu.max_waves_per_simd = 8;
  gpu.num_shader_engines = 4;
  gpu.num_cu_per_sh = 8;
  gpu.num_shader_arrays_per_engine = 1;
  gpu.local_mem_size = 64ULL * 1024 * 1024 * 1024; // 64GB
  gpu.device_id = 0x7400;
  gpu.num_cp_queues = 8;
  gpu.l1_size_kb = 32;
  gpu.l2_size_kb = 4096;
  driver->setup_topology(gpu);

  g_default_state = state.release(); // Intentionally leaked — see declaration.
  return driver;
}

SimulatedDriver::SimulatedDriver(simdojo::SimulationEngine &engine, SoC &soc)
    : engine_(engine), soc_(soc) {}

SimulatedDriver::~SimulatedDriver() {
  if (fd_ >= 0)
    close();
  if (event_memfd_ >= 0)
    ::close(event_memfd_);
}

void SimulatedDriver::setup_topology(const Sysfs::GpuInfo &gpu) {
  gpu_id_ = gpu.gpu_id;
  topology_.generate(gpu);
  topology_.setup_environment();
}

int SimulatedDriver::open() {
  // Allocate the synthetic KFD fd only once using memfd_create. The real kernel
  // fd table entry keeps the fd number reserved so other opens can never reuse it,
  // giving ROCR a stable fd value across close/reopen cycles (10-D).
  if (fd_ < 0) {
    fd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_kfd", 0));
    if (fd_ < 0)
      return -1;
  }
  closing_.store(false, std::memory_order_release);
  g_kfd_fd.store(fd_, std::memory_order_release);
  engine_.register_as_primary();
  // Resolve the CommandProcessor for this device once. All queue operations
  // (create, flush, destroy) use cp_ so they never need to know the XCD index.
  // SoC::command_processor() owns the topology decision (single-XCD → xcd(0);
  // future multi-XCD would return a MES dispatcher).
  cp_ = soc_.command_processor();
  // Register the interrupt callback here rather than at queue-creation time.
  // The CP calls this after writing to a completion signal's event mailbox,
  // waking any thread blocked in wait_events_ioctl.
  cp_->set_interrupt_callback([this]() { event_cv_.notify_all(); });
  return fd_;
}

int SimulatedDriver::close() {
  // Do NOT release the engine primary here. ROCR may close and re-open
  // /dev/kfd during its lifetime (e.g., Init() failure → Close() → retry).
  // Releasing the primary causes the engine to terminate, and it cannot
  // be restarted. The engine stays alive for the process lifetime.
  {
    // Hold event_mutex_ while setting closing_ to ensure wait_events_ioctl's
    // predicate sees the closed state before we notify.
    std::lock_guard<std::mutex> lock(event_mutex_);
    closing_.store(true, std::memory_order_release);
    // fd_ is intentionally NOT reset — the memfd stays reserved so the fd number
    // remains stable across close/reopen. lookup() uses g_kfd_fd to gate routing.
    g_kfd_fd.store(-1, std::memory_order_release);
  }
  // Signal every event page slot non-zero. libhsakmt's WaitOnEvent polls
  // signal_page[event_slot_index] directly; a non-zero value breaks the loop
  // immediately, allowing ROCR's background threads to see IsValid()==false
  // and exit cleanly without spinning on the WAIT_EVENTS ioctl.
  if (event_page_) {
    auto *slots = static_cast<uint64_t *>(event_page_);
    size_t count = event_page_size_ / sizeof(uint64_t);
    for (size_t i = 0; i < count; ++i)
      std::atomic_ref<uint64_t>(slots[i]).store(1, std::memory_order_release);
  }
  event_cv_.notify_all();

  // Unregister any CP queues and free host mappings that ROCR left open.
  // In normal operation ROCR calls DESTROY_QUEUE / FREE_MEMORY before close(),
  // but guard against leaks on abnormal shutdown.
  {
    std::lock_guard<std::mutex> lk(alloc_mutex_);
    for (uint32_t qid : active_queue_ids_)
      cp_->unregister_queue(qid);
    active_queue_ids_.clear();

    auto *mem = soc_.memory();
    for (auto &[handle, alloc] : allocations_) {
      if (alloc.host_ptr && !(alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)) {
        if (mem)
          mem->unmap_host_pages(alloc.gpu_va, alloc.size);
        syscall(SYS_munmap, alloc.host_ptr, alloc.size);
        alloc.host_ptr = nullptr;
      }
    }
    allocations_.clear();
  }

  return 0;
}

int SimulatedDriver::ioctl(unsigned long request, void *arg) {
  switch (request) {
  case AMDKFD_IOC_GET_VERSION:
    return get_version_ioctl(arg);
  case AMDKFD_IOC_GET_CLOCK_COUNTERS:
    return get_clock_counters_ioctl(arg);
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    return get_apertures_ioctl(arg);
  case AMDKFD_IOC_ACQUIRE_VM:
    return acquire_vm_ioctl(arg);
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return alloc_memory_ioctl(arg);
  case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
    return free_memory_ioctl(arg);
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return map_memory_ioctl(arg);
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return unmap_memory_ioctl(arg);
  case AMDKFD_IOC_CREATE_QUEUE:
    return create_queue_ioctl(arg);
  case AMDKFD_IOC_UPDATE_QUEUE:
    return update_queue_ioctl(arg);
  case AMDKFD_IOC_DESTROY_QUEUE:
    return destroy_queue_ioctl(arg);
  case AMDKFD_IOC_CREATE_EVENT:
    return create_event_ioctl(arg);
  case AMDKFD_IOC_DESTROY_EVENT:
    return destroy_event_ioctl(arg);
  case AMDKFD_IOC_SET_EVENT:
    return set_event_ioctl(arg);
  case AMDKFD_IOC_RESET_EVENT:
    return reset_event_ioctl(arg);
  case AMDKFD_IOC_WAIT_EVENTS:
    return wait_events_ioctl(arg);
  case AMDKFD_IOC_SET_XNACK_MODE:
    return set_xnack_mode_ioctl(arg);
  case AMDKFD_IOC_AVAILABLE_MEMORY: {
    auto *args = static_cast<kfd_ioctl_get_available_memory_args *>(arg);
    uint64_t allocated = 0;
    {
      std::lock_guard<std::mutex> lk(alloc_mutex_);
      for (auto &[handle, alloc] : allocations_)
        allocated += alloc.size;
    }
    // Report 64 GiB total VRAM minus current allocations (matches sysfs local_mem_size).
    constexpr uint64_t kVramBytes = 64ULL << 30;
    args->available = kVramBytes - std::min(allocated, kVramBytes);
    return 0;
  }
  case AMDKFD_IOC_RUNTIME_ENABLE: {
    auto *args = static_cast<kfd_ioctl_runtime_enable_args *>(arg);
    args->capabilities_mask = 0;
    return 0;
  }
  // Scratch backing VA: ROCR stores the flat-scratch base here so the CP can
  // program the SH_STATIC_MEM_CONFIG register. No-op in simulation — kernels
  // use private segment memory allocated at dispatch time instead.
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
    return 0;
  // Trap handler: ROCR provides TBA/TMA so the CP can install the trap handler.
  // No-op in simulation — we do not emulate the trap handler mechanism.
  case AMDKFD_IOC_SET_TRAP_HANDLER:
    return 0;
  default:
    fprintf(stderr, "rocjitsu: unhandled ioctl 0x%lx\n", request);
    return 0;
  }
}

void *SimulatedDriver::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    // Doorbell page: use anonymous memory mapped at the GPU VA ROCR requested.
    // Both ROCR's FMM mmap and libhsakmt's map_doorbell_dgpu run in the same
    // process, so they share the anonymous page directly — no file backing needed.
    // We avoid memfd here because MAP_FIXED over certain pre-existing VMAs
    // (e.g., /dev/zero shared mappings) produces SIGBUS even with a correctly
    // ftruncated file. Anonymous MAP_FIXED is always safe to write to.
    //
    // If the doorbell was already mapped (second mmap from libhsakmt), the
    // MAP_FIXED simply re-establishes the mapping and preserves the host VA.
    int db_mflags = MAP_ANONYMOUS | MAP_SHARED;
    if (flags & MAP_FIXED)
      db_mflags |= MAP_FIXED;
    long raw = syscall(SYS_mmap, addr, length, PROT_READ | PROT_WRITE, db_mflags, -1, 0);
    void *ptr = (raw < 0) ? MAP_FAILED : reinterpret_cast<void *>(static_cast<uintptr_t>(raw));
    if (ptr != MAP_FAILED) {
      doorbell_page_ = ptr;
      // Sentinel: fill with 0xFF so the initial read is 0xFFFF…FFFF.
      // The CP's last_doorbell is also 0xFFFF…FFFF, so any ROCR write
      // (including write_ptr=0) is detected as a change.
      memset(ptr, 0xFF, length);
      uint64_t gpu_va = reinterpret_cast<uint64_t>(ptr);
      if (auto *mem = soc_.memory())
        mem->map_host_pages(gpu_va, ptr, length);
      // Provide the aperture base to the CP. All previously registered KFD queues
      // (doorbell_offset already set) now become active in the poll loop.
      // Mirrors the kernel's model: doorbell BO allocated before userspace mmap;
      // mmap just exposes the existing aperture to userspace.
      cp_->set_doorbell_base(ptr);
    }
    return ptr;
  }

  if (type == KFD_MMAP_TYPE_EVENTS) {
    // Signal event page: a shared memfd that libhsakmt polls for event signals.
    // libhsakmt checks signal_page[event_slot_index] != 0 on each wait iteration.
    if (event_memfd_ < 0) {
      event_memfd_ =
          static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_events", MFD_ALLOW_SEALING));
      if (event_memfd_ < 0)
        return MAP_FAILED;
      if (ftruncate(event_memfd_, static_cast<off_t>(length)) != 0) {
        ::close(event_memfd_);
        event_memfd_ = -1;
        return MAP_FAILED;
      }
      // Seal against resize: the event page has a fixed layout (one slot per
      // KFD event ID) and must never grow or shrink after creation.
      fcntl(event_memfd_, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    }
    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    long raw = syscall(SYS_mmap, addr, length, PROT_READ | PROT_WRITE, mflags, event_memfd_, 0);
    void *ptr = (raw < 0) ? MAP_FAILED : reinterpret_cast<void *>(static_cast<uintptr_t>(raw));
    if (ptr != MAP_FAILED) {
      event_page_ = ptr;
      event_page_size_ = length;
    }
    return ptr;
  }

  std::lock_guard<std::mutex> lock(alloc_mutex_);

  uint64_t handle = static_cast<uint64_t>(offset) >> 12;
  auto it = allocations_.find(handle);
  if (it == allocations_.end()) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  auto &alloc = it->second;

  // Map anonymous memory at the GPU VA. Preserve MAP_FIXED and MAP_SHARED from
  // the caller: ROCR's FMM expects GPU VA == host VA (coarse unified memory).
  int mflags = MAP_ANONYMOUS;
  mflags |= (flags & MAP_SHARED) ? MAP_SHARED : MAP_PRIVATE;
  if (flags & MAP_FIXED)
    mflags |= MAP_FIXED;
  long raw = syscall(SYS_mmap, addr, length, prot, mflags, -1, 0);
  void *host_ptr = (raw < 0) ? MAP_FAILED : reinterpret_cast<void *>(static_cast<uintptr_t>(raw));
  if (host_ptr == MAP_FAILED)
    return MAP_FAILED;

  alloc.host_ptr = host_ptr;

  if (auto *mem = soc_.memory())
    mem->map_host_pages(alloc.gpu_va, host_ptr, length);

  return host_ptr;
}

int SimulatedDriver::munmap(void *addr, size_t length) {
  if (addr == doorbell_page_) {
    doorbell_page_ = nullptr;
    cp_->set_doorbell_base(nullptr);
    syscall(SYS_munmap, addr, length);
    return 0;
  }
  if (addr == event_page_) {
    event_page_ = nullptr;
    event_page_size_ = 0;
    syscall(SYS_munmap, addr, length);
    return 0;
  }
  std::lock_guard<std::mutex> lock(alloc_mutex_);
  for (auto &[handle, alloc] : allocations_) {
    if (alloc.host_ptr == addr) {
      if (auto *mem = soc_.memory())
        mem->unmap_host_pages(alloc.gpu_va, alloc.size);
      syscall(SYS_munmap, addr, length);
      alloc.host_ptr = nullptr;
      return 0;
    }
  }
  return -ENOENT;
}

int SimulatedDriver::get_version_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_version_args *>(arg);
  args->major_version = KFD_IOCTL_MAJOR_VERSION;
  args->minor_version = KFD_IOCTL_MINOR_VERSION;
  return 0;
}

int SimulatedDriver::get_clock_counters_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_clock_counters_args *>(arg);
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  uint64_t ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  args->system_clock_freq = 1000000000ULL; // 1 GHz — counter increments in nanoseconds
  args->system_clock_counter = ns;
  args->cpu_clock_counter = ns;
  args->gpu_clock_counter = ns;
  return 0;
}

int SimulatedDriver::get_apertures_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);

  if (args->num_of_nodes == 0) {
    args->num_of_nodes = 1;
    return 0;
  }

  auto *apertures =
      reinterpret_cast<kfd_process_device_apertures *>(args->kfd_process_device_apertures_ptr);
  apertures[0] = default_apertures_;
  apertures[0].gpu_id = gpu_id_;

  args->num_of_nodes = 1;
  return 0;
}

int SimulatedDriver::acquire_vm_ioctl(void *arg) {
  (void)arg;
  return 0;
}

int SimulatedDriver::alloc_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);
  std::lock_guard<std::mutex> lock(alloc_mutex_);

  uint64_t va = args->va_addr;
  if (va == 0) {
    va = next_gpu_va_;
    next_gpu_va_ += (args->size + 0xFFF) & ~0xFFFULL;
  }

  GpuAllocation alloc{};
  alloc.gpu_va = va;
  alloc.size = args->size;
  alloc.flags = args->flags;
  alloc.handle = next_handle_++;
  alloc.host_ptr = nullptr;

  // USERPTR: the va_addr IS the host pointer — no mmap follows.
  // Register the mapping immediately so the GPU can access host memory.
  if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) {
    alloc.host_ptr = reinterpret_cast<void *>(va);
    if (auto *mem = soc_.memory())
      mem->map_host_pages(va, reinterpret_cast<void *>(va), args->size);
  }

  allocations_[alloc.handle] = alloc;

  args->handle = alloc.handle;
  args->va_addr = va;
  if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
    args->mmap_offset = KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(gpu_id_);
  else
    args->mmap_offset = alloc.handle << 12;

  return 0;
}

int SimulatedDriver::free_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
  std::lock_guard<std::mutex> lock(alloc_mutex_);
  auto it = allocations_.find(args->handle);
  if (it != allocations_.end()) {
    auto &alloc = it->second;
    if ((alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) && alloc.host_ptr) {
      if (auto *mem = soc_.memory())
        mem->unmap_host_pages(alloc.gpu_va, alloc.size);
    }
    allocations_.erase(it);
  }
  return 0;
}

int SimulatedDriver::map_memory_ioctl(void *arg) {
  (void)arg;
  return 0;
}

int SimulatedDriver::unmap_memory_ioctl(void *arg) {
  (void)arg;
  return 0;
}

int SimulatedDriver::create_queue_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_create_queue_args *>(arg);
  if (args->gpu_id != gpu_id_)
    return -EINVAL;

  // Register the ring buffer and read/write pointers in GpuMemory.
  // ROCR allocates these in system memory (not through KFD ALLOC_MEMORY),
  // so they don't have host_pages mappings. The CP needs to read them
  // through GpuMemory's host_ranges_ to see the AQL packets ROCR writes.
  if (auto *mem = soc_.memory()) {
    mem->map_host_pages(args->ring_base_address, reinterpret_cast<void *>(args->ring_base_address),
                        args->ring_size);
    // Map the read/write pointer pages (typically 4KB each).
    uint64_t rptr_page = args->read_pointer_address & ~0xFFFULL;
    uint64_t wptr_page = args->write_pointer_address & ~0xFFFULL;
    mem->map_host_pages(rptr_page, reinterpret_cast<void *>(rptr_page), 4096);
    if (wptr_page != rptr_page)
      mem->map_host_pages(wptr_page, reinterpret_cast<void *>(wptr_page), 4096);
  }

  uint32_t queue_id = next_queue_id_++;
  uint32_t db_offset = static_cast<uint32_t>(next_doorbell_offset_);
  next_doorbell_offset_ += sizeof(uint64_t);

  // Register the queue with the CP immediately using just the doorbell offset.
  // The CP mirrors real hardware: it holds doorbell_offset (like cp_hqd_pq_doorbell_control
  // in the MQD) and resolves the address as doorbell_base + offset when the driver
  // maps the aperture via set_doorbell_base(). No deferred "pending queue" list needed.
  amdgpu::HwQueue hw{};
  hw.queue_id = queue_id;
  hw.ring_base_va = args->ring_base_address;
  hw.ring_size = args->ring_size;
  hw.read_ptr_va = args->read_pointer_address;
  hw.write_ptr_va = args->write_pointer_address;
  hw.doorbell_offset = db_offset;
  hw.last_doorbell = ~uint64_t(0); // Sentinel matches memset(0xFF) init on the doorbell page.
  hw.host_accessible = true;       // KFD queues use host VAs for pointers.
  cp_->register_queue(std::move(hw));

  args->queue_id = queue_id;
  args->doorbell_offset = KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(gpu_id_) | db_offset;
  {
    std::lock_guard<std::mutex> lk(alloc_mutex_);
    active_queue_ids_.push_back(queue_id);
  }
  return 0;
}

int SimulatedDriver::update_queue_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_update_queue_args *>(arg);
  cp_->update_queue(args->queue_id, args->ring_base_address, args->ring_size);
  return 0;
}

int SimulatedDriver::destroy_queue_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_destroy_queue_args *>(arg);
  cp_->unregister_queue(args->queue_id);
  {
    std::lock_guard<std::mutex> lk(alloc_mutex_);
    std::erase(active_queue_ids_, args->queue_id);
  }
  return 0;
}

int SimulatedDriver::create_event_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_create_event_args *>(arg);
  std::lock_guard<std::mutex> lock(event_mutex_);

  if (next_event_id_ >= KFD_SIGNAL_EVENT_LIMIT)
    return -ENOSPC;

  GpuEvent ev{};
  ev.event_id = next_event_id_++;
  ev.event_type = args->event_type;
  ev.auto_reset = args->auto_reset != 0;
  ev.signaled = false;

  events_[ev.event_id] = ev;

  args->event_id = ev.event_id;
  args->event_trigger_data = ev.event_id;
  args->event_slot_index = ev.event_id; // 1-indexed; slot 0 unused
  args->event_page_offset = KFD_MMAP_TYPE_EVENTS | kfd_mmap_gpu_id(gpu_id_);
  return 0;
}

int SimulatedDriver::destroy_event_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_destroy_event_args *>(arg);
  {
    std::lock_guard<std::mutex> lock(event_mutex_);
    events_.erase(args->event_id);
  }
  // Wake any threads blocked in wait_events_ioctl — the event they were
  // waiting on may have been destroyed. The real KFD driver does the same.
  event_cv_.notify_all();
  return 0;
}

int SimulatedDriver::set_event_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_event_args *>(arg);
  {
    std::lock_guard<std::mutex> lock(event_mutex_);
    auto it = events_.find(args->event_id);
    if (it == events_.end())
      return -EINVAL;
    it->second.signaled = true;
    // Write to the signal page slot so libhsakmt's direct poll detects this event.
    if (event_page_) {
      auto *slots = static_cast<uint64_t *>(event_page_);
      if (args->event_id < event_page_size_ / sizeof(uint64_t))
        std::atomic_ref<uint64_t>(slots[args->event_id]).store(1, std::memory_order_release);
    }
  }
  event_cv_.notify_all();
  return 0;
}

int SimulatedDriver::reset_event_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_reset_event_args *>(arg);
  std::lock_guard<std::mutex> lock(event_mutex_);
  auto it = events_.find(args->event_id);
  if (it == events_.end())
    return -EINVAL;
  it->second.signaled = false;
  if (event_page_) {
    auto *slots = static_cast<uint64_t *>(event_page_);
    if (args->event_id < event_page_size_ / sizeof(uint64_t))
      std::atomic_ref<uint64_t>(slots[args->event_id]).store(0, std::memory_order_release);
  }
  return 0;
}

int SimulatedDriver::set_xnack_mode_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_xnack_mode_args *>(arg);
  // XNACK (recoverable page faults / SVM) is not modeled; always report disabled.
  // Returning 0 prevents libhsakmt from enabling SVM allocation paths that bypass
  // our ALLOC_MEMORY_OF_GPU tracking.
  args->xnack_enabled = 0;
  return 0;
}

int SimulatedDriver::wait_events_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_wait_events_args *>(arg);

  auto timeout_ms = std::chrono::milliseconds(args->timeout);

  std::unique_lock<std::mutex> lock(event_mutex_);

  // Build the list of requested event IDs from the userspace array.
  auto *ev_data = reinterpret_cast<const kfd_event_data *>(args->events_ptr);

  auto is_signaled = [this](uint32_t id) -> bool {
    if (auto it = events_.find(id); it != events_.end() && it->second.signaled)
      return true;
    if (event_page_ && id < event_page_size_ / sizeof(uint64_t)) {
      auto *slots = static_cast<uint64_t *>(event_page_);
      if (std::atomic_ref<uint64_t>(slots[id]).load(std::memory_order_acquire) != 0)
        return true;
    }
    return false;
  };

  auto pred = [this, args, ev_data, &is_signaled]() -> bool {
    if (closing_)
      return true;
    if (args->wait_for_all) {
      // AND semantics: all requested events must be signaled.
      for (uint32_t i = 0; i < args->num_events; ++i) {
        if (!is_signaled(ev_data[i].event_id))
          return false;
      }
      return args->num_events > 0;
    } else {
      // OR semantics: any one event signaled is sufficient.
      for (uint32_t i = 0; i < args->num_events; ++i) {
        if (is_signaled(ev_data[i].event_id))
          return true;
      }
      return false;
    }
  };

  if (closing_)
    return -EBADF;

  if (args->timeout == 0) {
    args->wait_result = pred() ? 0 : 1;
  } else if (args->timeout == ~0u) {
    // Cap infinite waits at 100ms. ROCR's signal watcher uses UINT32_MAX and
    // relies on the kernel's wake_up_interruptible() during shutdown. Simulate
    // this by returning periodically with wait_result=1 (spurious wakeup) so
    // ROCR's thread can check its own termination flag and exit before close()
    // is called. Without this, hsa_shut_down() deadlocks: it joins the watcher
    // while the watcher is stuck in our wait waiting for close() to be called.
    event_cv_.wait_for(lock, std::chrono::milliseconds(100), pred);
    if (closing_)
      return -EBADF;
    args->wait_result = pred() ? 0 : 1;
  } else {
    // Cap the wait at 100ms. ROCR issues long-timeout drain waits (e.g. 30s)
    // during hsa_shut_down() for GPU work that never completes in the simulator.
    // Returning early with wait_result=1 (timeout) is correct: ROCR treats it
    // identically to the deadline expiring and continues with cleanup.
    // close() also notifies event_cv_, so a real close-driven wake still works.
    auto cap = std::min(timeout_ms, std::chrono::milliseconds(100));
    event_cv_.wait_for(lock, cap, pred);
    if (closing_)
      return -EBADF;
    args->wait_result = pred() ? 0 : 1;
  }

  // Auto-reset signaled events from the requested set.
  for (uint32_t i = 0; i < args->num_events; ++i) {
    uint32_t id = ev_data[i].event_id;
    if (auto it = events_.find(id);
        it != events_.end() && it->second.signaled && it->second.auto_reset) {
      it->second.signaled = false;
      if (event_page_ && id < event_page_size_ / sizeof(uint64_t)) {
        auto *slots = static_cast<uint64_t *>(event_page_);
        std::atomic_ref<uint64_t>(slots[id]).store(0, std::memory_order_release);
      }
    }
  }

  return 0;
}

} // namespace rocjitsu
