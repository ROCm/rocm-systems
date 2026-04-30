// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/types.h>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

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

bool vm_trace_enabled() {
  static const bool enabled = (std::getenv("RJ_VMEM_TRACE") != nullptr);
  return enabled;
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
    std::unique_ptr<SimulatedDriver> driver;
    try {
      driver = create_default();
    } catch (const std::exception &e) {
      g_in_construction = false;
      util::Logger::debug_print("rocjitsu: ", e.what());
      return nullptr;
    }
    g_in_construction = false;
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

static constexpr const char *const DRM_SYSFS_PREFIX = "/sys/class/drm";

static constexpr const char *const KFD_SYSFS_PREFIX_ALT = "/sys/class/kfd/kfd/topology";

std::string SimulatedDriver::redirect_sysfs_path(const char *path) {
  auto *inst = g_instance.load(std::memory_order_acquire);
  if (!inst)
    return {};

  // KFD topology redirect. Match both the canonical path and the
  // /sys/class/kfd/ symlink path (rsmi/amdsmi use the symlink).
  std::string_view sv(path);
  std::string_view kfd_prefix(KFD_SYSFS_PREFIX);
  if (sv.starts_with(kfd_prefix)) {
    auto result = inst->topology_path() + std::string(sv.substr(kfd_prefix.size()));
    util::Logger::vm("sysfs redirect: ", path, " -> ", result);
    return result;
  }
  std::string_view kfd_alt_prefix(KFD_SYSFS_PREFIX_ALT);
  if (sv.starts_with(kfd_alt_prefix))
    return inst->topology_path() + std::string(sv.substr(kfd_alt_prefix.size()));

  // DRM sysfs redirect for amdsmi/rocm_smi device discovery.
  const auto &drm = inst->topology().drm_path();
  if (!drm.empty()) {
    std::string_view drm_prefix(DRM_SYSFS_PREFIX);
    if (sv.starts_with(drm_prefix))
      return drm + std::string(sv.substr(drm_prefix.size()));
  }

  return {};
}

bool SimulatedDriver::in_construction() { return g_in_construction; }

std::unique_ptr<SimulatedDriver> SimulatedDriver::create_default() {
  const char *config_path = getenv("RJ_CONFIG");
  const char *schema_path = getenv("RJ_SCHEMA");
  if (!config_path || !schema_path)
    throw util::ConfigError("RJ_CONFIG and RJ_SCHEMA env vars required");

  auto state = std::make_unique<DefaultDriverState>();
  state->loaded = config::load_config(config_path, schema_path);
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

  // Build sysfs GpuInfo from the config's device section.
  Sysfs::GpuInfo gpu{};
  const auto &dev = state->loaded.device;
  if (!dev.present)
    throw util::ConfigError("config missing vm.gpu.device section");

  gpu.gpu_id = dev.gpu_id;
  gpu.gfx_target_version = dev.gfx_target_version;
  gpu.vendor_id = dev.vendor_id;
  gpu.device_id = dev.device_id;
  gpu.family_id = dev.family_id;
  gpu.unique_id = dev.unique_id;
  gpu.marketing_name = dev.marketing_name.c_str();
  gpu.drm_render_minor = dev.drm_render_minor;
  gpu.simd_count = dev.simd_count;
  gpu.max_waves_per_simd = dev.max_waves_per_simd;
  gpu.num_shader_engines = dev.num_shader_engines;
  gpu.num_shader_arrays_per_engine = dev.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = dev.num_cu_per_sh;
  gpu.simd_per_cu = dev.simd_per_cu;
  gpu.wave_front_size = dev.wave_front_size;
  gpu.max_slots_scratch_cu = dev.max_slots_scratch_cu;
  gpu.local_mem_size = dev.local_mem_size;
  gpu.lds_size_kb = dev.lds_size_kb;
  gpu.mem_width = dev.mem_width;
  gpu.mem_clk_max = dev.mem_clk_max;
  gpu.l1_size_kb = dev.l1_size_kb;
  gpu.l1_line_size = dev.l1_line_size;
  gpu.l1_assoc = dev.l1_assoc;
  gpu.l2_size_kb = dev.l2_size_kb;
  gpu.l2_line_size = dev.l2_line_size;
  gpu.l2_assoc = dev.l2_assoc;
  gpu.num_sdma_engines = dev.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = dev.num_sdma_xgmi_engines;
  gpu.num_cp_queues = dev.num_cp_queues;
  gpu.max_engine_clk_fcompute = dev.max_engine_clk_fcompute;
  gpu.num_xcc = soc->num_xcds();

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

bool SimulatedDriver::is_doorbell_range(const void *addr, size_t length) const {
  if (!doorbell_page_ || doorbell_page_size_ == 0 || !addr || length == 0)
    return false;
  auto base = reinterpret_cast<uintptr_t>(doorbell_page_);
  auto end = base + doorbell_page_size_;
  auto query_base = reinterpret_cast<uintptr_t>(addr);
  auto query_end = query_base + length;
  return query_base < end && query_end > base;
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
  cp_->set_interrupt_callback([this](uint32_t event_id) {
    {
      std::lock_guard<std::mutex> lock(event_mutex_);
      if (auto it = events_.find(event_id); it != events_.end())
        it->second.signaled = true;
    }
    event_cv_.notify_all();
  });
  return fd_;
}

int SimulatedDriver::close() {
  // Do NOT release the engine primary here. ROCR may close and re-open
  // /dev/kfd during its lifetime (e.g., Init() failure → Close() → retry).
  // Releasing the primary causes the engine to terminate, and it cannot
  // be restarted. The engine stays alive for the process lifetime.
  const bool trace_enabled = vm_trace_enabled();
  size_t leaked_allocations = 0;
  uint64_t leaked_bytes = 0;
  size_t leaked_queues = 0;
  std::vector<uint64_t> leaked_handles;
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
      std::atomic_ref<uint64_t>(slots[i]).store(KFD_SIGNAL_EVENT_LIMIT, std::memory_order_release);
  }
  event_cv_.notify_all();

  // Unregister any CP queues and free host mappings that ROCR left open.
  // In normal operation ROCR calls DESTROY_QUEUE / FREE_MEMORY before close(),
  // but guard against leaks on abnormal shutdown.
  {
    std::lock_guard<std::mutex> lk(alloc_mutex_);
    leaked_queues = active_queue_ids_.size();
    for (uint32_t qid : active_queue_ids_)
      cp_->unregister_queue(qid);
    active_queue_ids_.clear();

    auto *mem = soc_.memory();
    if (trace_enabled)
      leaked_handles.reserve(allocations_.size());
    for (auto &[handle, alloc] : allocations_) {
      ++leaked_allocations;
      leaked_bytes += alloc.size;
      if (trace_enabled)
        leaked_handles.push_back(handle);
      if (alloc.host_ptr && !(alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)) {
        if (mem)
          mem->unmap_host_pages(alloc.gpu_va, alloc.size);
        syscall(SYS_munmap, alloc.host_ptr, alloc.size);
        alloc.host_ptr = nullptr;
      }
    }
    allocations_.clear();
  }

  if (doorbell_page_ && doorbell_page_size_)
    munmap(doorbell_page_, doorbell_page_size_);

  if (trace_enabled) {
    if (leaked_allocations == 0 && leaked_queues == 0) {
      util::Logger::vm("kfd.close: no outstanding GPUVM allocations or queues");
    } else {
      util::Logger::vm("kfd.close: leaked_allocations=", leaked_allocations,
                       " leaked_bytes=", leaked_bytes, " leaked_queues=", leaked_queues);
      if (!leaked_handles.empty()) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < leaked_handles.size(); ++i) {
          oss << leaked_handles[i];
          if (i + 1 < leaked_handles.size())
            oss << ",";
        }
        oss << "]";
        util::Logger::vm("kfd.close: leaked_handles=", oss.str());
      }
    }
  }

  for (auto &[handle, dmabuf] : imported_dmabufs_) {
    (void)handle;
    if (dmabuf.fd >= 0)
      ::close(dmabuf.fd);
  }
  imported_dmabufs_.clear();
  fd_to_import_handle_.clear();
  svm_ranges_.clear();
  memory_policies_.clear();
  {
    std::lock_guard<std::mutex> lock(runtime_mutex_);
    runtime_state_ = RuntimeState{};
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
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return set_memory_policy_ioctl(arg);
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
    return runtime_enable_ioctl(arg);
  }
  // Scratch backing VA: ROCR stores the flat-scratch base here so the CP can
  // program SH_STATIC_MEM_CONFIG. Store the value for diagnostic use; the CU
  // reads scratch_backing_memory_location from amd_queue_t at dispatch time.
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA: {
    auto *a = static_cast<kfd_ioctl_set_scratch_backing_va_args *>(arg);
    scratch_backing_va_ = a->va_addr;
    return 0;
  }
  // Trap handler: ROCR provides TBA/TMA addresses for the trap handler.
  // Store them for future trap support; no effect on execution currently.
  case AMDKFD_IOC_SET_TRAP_HANDLER: {
    auto *a = static_cast<kfd_ioctl_set_trap_handler_args *>(arg);
    trap_tba_addr_ = a->tba_addr;
    trap_tma_addr_ = a->tma_addr;
    return 0;
  }
  case AMDKFD_IOC_GET_DMABUF_INFO:
    return get_dmabuf_info_ioctl(arg);
  case AMDKFD_IOC_IMPORT_DMABUF:
    return import_dmabuf_ioctl(arg);
  case AMDKFD_IOC_EXPORT_DMABUF:
    return export_dmabuf_ioctl(arg);
  case AMDKFD_IOC_SVM:
    return svm_ioctl(arg);
  default:
    util::Logger::debug_print("rocjitsu: unhandled ioctl 0x", std::hex, request);
    return 0;
  }
}

void *SimulatedDriver::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;
  util::Logger::vm("SimulatedDriver::mmap type=0x", std::hex, type, " offset=0x", offset,
                   " length=", std::dec, length, " addr=", addr);

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
      doorbell_page_size_ = length;
      doorbell_gpu_va_ = reinterpret_cast<uint64_t>(ptr);
      // Sentinel: fill with 0xFF so the initial read is 0xFFFF…FFFF.
      // The CP's last_doorbell is also 0xFFFF…FFFF, so any ROCR write
      // (including write_ptr=0) is detected as a change.
      memset(ptr, 0xFF, length);
      if (auto *mem = soc_.memory())
        mem->map_host_pages(doorbell_gpu_va_, ptr, length);
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

  // ROCR's FMM pre-maps GPU VAs with MAP_ANONYMOUS|MAP_PRIVATE and may write
  // data (code objects) before calling this KFD mmap. In real KFD, the mmap
  // binds the VA to VRAM (preserving data). We must NOT re-mmap with MAP_FIXED
  // when pages already exist, as that destroys the data ROCR wrote.
  //
  // For user-provided VAs (ROCR FMM path): skip the re-mmap entirely. The
  // existing FMM reservation pages already contain any data ROCR wrote. Just
  // register them in GpuMemory and set mprotect to ensure they're accessible.
  //
  // For simulator-assigned VAs (internal tests): create fresh anonymous pages.
  void *host_ptr;
  bool reuse_pages = false;
  if (alloc.user_va && (flags & MAP_FIXED) && addr != nullptr) {
    // Try to keep existing pages (ROCR's FMM may have written code data).
    // mprotect succeeds only if the mapping exists; if ROCR unmapped the
    // reservation before calling this KFD mmap, fall through to fresh alloc.
    long rc = syscall(SYS_mprotect, addr, length, PROT_READ | PROT_WRITE);
    reuse_pages = (rc == 0);
  }
  if (reuse_pages) {
    host_ptr = addr;
  } else {
    int mflags = MAP_ANONYMOUS;
    mflags |= (flags & MAP_SHARED) ? MAP_SHARED : MAP_PRIVATE;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    long raw = syscall(SYS_mmap, addr, length, prot, mflags, -1, 0);
    host_ptr = (raw < 0) ? MAP_FAILED : reinterpret_cast<void *>(static_cast<uintptr_t>(raw));
    if (host_ptr == MAP_FAILED)
      return MAP_FAILED;
  }

  alloc.host_ptr = host_ptr;

  util::Logger::vm([&](auto &os) {
    os << std::format("mmap: gpu_va={:#x} host_ptr={:#x} size={} flags={:#x}"
                      " MAP_FIXED={} reuse_pages={} user_va={}",
                      alloc.gpu_va, reinterpret_cast<uintptr_t>(host_ptr), length, alloc.flags,
                      bool(flags & MAP_FIXED), reuse_pages, alloc.user_va);
  });

  if (auto *mem = soc_.memory())
    mem->map_host_pages(alloc.gpu_va, host_ptr, length);

  return host_ptr;
}

int SimulatedDriver::munmap(void *addr, size_t length) {
  if (addr == doorbell_page_) {
    if (!closing_.load(std::memory_order_acquire)) {
      errno = EPERM;
      return -1;
    }
    if (auto *mem = soc_.memory(); mem && doorbell_gpu_va_ && doorbell_page_size_)
      mem->unmap_host_pages(doorbell_gpu_va_, doorbell_page_size_);
    doorbell_page_ = nullptr;
    doorbell_gpu_va_ = 0;
    doorbell_page_size_ = 0;
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

  bool user_provided_va = (args->va_addr != 0);
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
  alloc.user_va = user_provided_va;

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

  util::Logger::vm("alloc: handle=", alloc.handle, " gpu_va=0x", std::hex, va, " size=", std::dec,
                   args->size, " flags=0x", std::hex, args->flags, std::dec);

  return 0;
}

int SimulatedDriver::free_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
  std::lock_guard<std::mutex> lock(alloc_mutex_);
  auto it = allocations_.find(args->handle);
  if (it != allocations_.end()) {
    auto &alloc = it->second;
    if (alloc.imported && alloc.dmabuf_fd >= 0) {
      ::close(alloc.dmabuf_fd);
      if (auto dmabuf_it = imported_dmabufs_.find(args->handle);
          dmabuf_it != imported_dmabufs_.end()) {
        fd_to_import_handle_.erase(dmabuf_it->second.fd);
        imported_dmabufs_.erase(dmabuf_it);
      }
    }
    // For FMM (user_va) allocations, keep the host-page mapping alive until
    // the process actually munmaps the VA. ROCR's caching allocator and
    // PyTorch's block pool reuse freed handles without unmapping, so dropping
    // the mapping here causes the GPU to read zeros on reuse.
    if (alloc.host_ptr && !alloc.user_va) {
      if (auto *mem = soc_.memory())
        mem->unmap_host_pages(alloc.gpu_va, alloc.size);
    }
    allocations_.erase(it);
  }
  return 0;
}

int SimulatedDriver::map_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
  std::lock_guard<std::mutex> lock(alloc_mutex_);
  auto it = allocations_.find(args->handle);
  if (it == allocations_.end())
    return -EINVAL;
  auto &alloc = it->second;
  util::Logger::vm([&](auto &os) {
    os << std::format("MAP_MEMORY handle={} gpu_va={:#x} size={} flags={:#x} host_ptr={:#x}",
                      alloc.handle, alloc.gpu_va, alloc.size, alloc.flags,
                      reinterpret_cast<uintptr_t>(alloc.host_ptr));
  });
  // Ensure the allocation's host pages are registered in GpuMemory.
  if (alloc.host_ptr && soc_.memory())
    soc_.memory()->map_host_pages(alloc.gpu_va, alloc.host_ptr, alloc.size);
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
  // Detect SDMA queues by queue_type. ROCR creates SDMA queues for memory
  // copies; the CP processes them as direct memcpy operations.
  // KFD queue type constants (from kfd_ioctl.h):
  //   0 = KFD_IOC_QUEUE_TYPE_COMPUTE
  //   1 = KFD_IOC_QUEUE_TYPE_SDMA
  //   2 = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL
  //   3 = KFD_IOC_QUEUE_TYPE_SDMA_XGMI
  //   4 = KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID
  hw.is_sdma = (args->queue_type == 1 /*KFD_IOC_QUEUE_TYPE_SDMA*/ ||
                args->queue_type == 3 /*KFD_IOC_QUEUE_TYPE_SDMA_XGMI*/ ||
                args->queue_type == 4 /*KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID*/);
  // Log and zero-init the SDMA queue's write/read pointers.
  if (hw.is_sdma) {
    auto *wptr = reinterpret_cast<uint64_t *>(args->write_pointer_address);
    auto *rptr = reinterpret_cast<uint64_t *>(args->read_pointer_address);
    util::Logger::vm("SDMA wptr before init: addr=0x", std::hex, args->write_pointer_address,
                     " val=", std::dec, *wptr, " rptr val=", *rptr);
    *wptr = 0;
    *rptr = 0;
  }
  util::Logger::vm("create_queue: id=", queue_id, " type=", args->queue_type,
                   " is_sdma=", hw.is_sdma, " ring=0x", std::hex, args->ring_base_address,
                   " size=", std::dec, args->ring_size);
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

  // dGPU path: libhsakmt pre-allocates the event page as a regular GPUVM
  // allocation and passes its mmap offset in event_page_offset. The real
  // kernel calls kfd_kmap_event_page() to adopt it. We resolve the offset
  // back to the host pointer via our allocation table.
  if (args->event_page_offset != 0 && !event_page_) {
    uint64_t handle = static_cast<uint64_t>(args->event_page_offset) >> 12;
    std::lock_guard<std::mutex> alock(alloc_mutex_);
    auto it = allocations_.find(handle);
    if (it != allocations_.end() && it->second.host_ptr) {
      event_page_ = it->second.host_ptr;
      event_page_size_ = it->second.size;
      util::Logger::vm("CREATE_EVENT: adopted pre-allocated event page handle=", handle,
                       " host_ptr=0x", std::hex, reinterpret_cast<uintptr_t>(event_page_),
                       " size=", std::dec, event_page_size_);
    }
  }

  GpuEvent ev{};
  ev.event_id = next_event_id_++;
  ev.event_type = args->event_type;
  ev.auto_reset = args->auto_reset != 0;
  ev.signaled = false;

  events_[ev.event_id] = ev;

  args->event_id = ev.event_id;
  args->event_trigger_data = ev.event_id;
  args->event_slot_index = ev.event_id;
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

int SimulatedDriver::set_memory_policy_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_memory_policy_args *>(arg);
  if (args->gpu_id != gpu_id_)
    return -EINVAL;
  MemoryPolicy policy{};
  policy.alternate_base = args->alternate_aperture_base;
  policy.alternate_size = args->alternate_aperture_size;
  policy.default_policy = args->default_policy;
  policy.alternate_policy = args->alternate_policy;
  memory_policies_[args->gpu_id] = policy;
  return 0;
}

int SimulatedDriver::import_dmabuf_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_import_dmabuf_args *>(arg);
  if (args->gpu_id != gpu_id_)
    return -EINVAL;

  struct stat st{};
  if (fstat(args->dmabuf_fd, &st) != 0)
    return -errno;
  uint64_t size = static_cast<uint64_t>(st.st_size);

  int dupfd = fcntl(args->dmabuf_fd, F_DUPFD_CLOEXEC, 0);
  if (dupfd < 0)
    return -errno;

  uint64_t handle;
  {
    std::lock_guard<std::mutex> lk(alloc_mutex_);
    handle = next_handle_++;
    GpuAllocation alloc{};
    alloc.gpu_va = args->va_addr;
    alloc.size = size;
    alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    alloc.handle = handle;
    alloc.user_va = true;
    alloc.imported = true;
    alloc.dmabuf_fd = dupfd;
    alloc.host_ptr = reinterpret_cast<void *>(args->va_addr);
    allocations_[handle] = alloc;
  }

  if (auto *mem = soc_.memory(); mem && args->va_addr)
    mem->map_host_pages(args->va_addr, reinterpret_cast<void *>(args->va_addr), size);

  ImportedDmabuf info{};
  info.handle = handle;
  info.fd = dupfd;
  info.size = size;
  info.va = args->va_addr;
  info.gpu_id = args->gpu_id;
  imported_dmabufs_[handle] = info;
  fd_to_import_handle_[dupfd] = handle;

  args->handle = handle;
  return 0;
}

int SimulatedDriver::export_dmabuf_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);
  std::lock_guard<std::mutex> lk(alloc_mutex_);
  auto it = allocations_.find(args->handle);
  if (it == allocations_.end())
    return -EINVAL;
  const auto &alloc = it->second;
  int memfd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_export", MFD_CLOEXEC));
  if (memfd < 0)
    return -errno;
  if (ftruncate(memfd, static_cast<off_t>(alloc.size)) != 0) {
    ::close(memfd);
    return -errno;
  }
  args->dmabuf_fd = memfd;
  return 0;
}

int SimulatedDriver::get_dmabuf_info_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_dmabuf_info_args *>(arg);
  uint64_t size = 0;
  uint32_t gpu_id = gpu_id_;

  bool found = false;
  for (const auto &[handle, info] : imported_dmabufs_) {
    (void)handle;
    if (info.fd >= 0 && static_cast<uint32_t>(info.fd) == args->dmabuf_fd) {
      size = info.size;
      gpu_id = info.gpu_id;
      found = true;
      break;
    }
  }

  if (!found) {
    struct stat st{};
    if (fstat(args->dmabuf_fd, &st) != 0)
      return -errno;
    size = static_cast<uint64_t>(st.st_size);
  }

  args->size = size;
  args->gpu_id = gpu_id;
  args->flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  if (args->metadata_ptr && args->metadata_size) {
    std::memset(reinterpret_cast<void *>(args->metadata_ptr), 0,
                static_cast<size_t>(args->metadata_size));
  }
  args->metadata_size = 0;
  return 0;
}

int SimulatedDriver::svm_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_svm_args *>(arg);
  auto *attrs = reinterpret_cast<kfd_ioctl_svm_attribute *>(args + 1);

  if (args->op == KFD_IOCTL_SVM_OP_SET_ATTR) {
    SvmRange range{};
    range.size = args->size;
    for (uint32_t i = 0; i < args->nattr; ++i)
      range.attributes[attrs[i].type] = attrs[i].value;
    svm_ranges_[args->start_addr] = std::move(range);
    return 0;
  }

  if (args->op == KFD_IOCTL_SVM_OP_GET_ATTR) {
    auto it = svm_ranges_.find(args->start_addr);
    for (uint32_t i = 0; i < args->nattr; ++i) {
      uint32_t type = attrs[i].type;
      uint32_t value = 0;
      if (it != svm_ranges_.end()) {
        if (auto vit = it->second.attributes.find(type); vit != it->second.attributes.end())
          value = vit->second;
      }
      switch (type) {
      case KFD_IOCTL_SVM_ATTR_PREFERRED_LOC:
      case KFD_IOCTL_SVM_ATTR_PREFETCH_LOC:
        attrs[i].value = value ? value : KFD_IOCTL_SVM_LOCATION_UNDEFINED;
        break;
      default:
        attrs[i].value = value;
        break;
      }
    }
    return 0;
  }

  return -EINVAL;
}

int SimulatedDriver::runtime_enable_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_runtime_enable_args *>(arg);
  std::lock_guard<std::mutex> lock(runtime_mutex_);

  if (args->mode_mask & KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK) {
    if (runtime_state_.pending)
      return -EBUSY;
    if (!runtime_state_.enabled && !active_queue_ids_.empty())
      return -EEXIST;
    runtime_state_.enabled = true;
    runtime_state_.pending = false;
    runtime_state_.mode_mask = args->mode_mask;
    runtime_state_.capabilities_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
    runtime_state_.r_debug = args->r_debug;
    args->capabilities_mask = runtime_state_.capabilities_mask;
    return 0;
  }

  runtime_state_ = RuntimeState{};
  args->capabilities_mask = 0;
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
      auto slot_val = std::atomic_ref<uint64_t>(slots[id]).load(std::memory_order_acquire);
      if (slot_val != 0)
        return true;
    }
    return false;
  };

  // One-shot trace: log what WAIT_EVENTS is looking for.
  {
    static thread_local int wait_trace_count = 0;
    if (++wait_trace_count <= 3) {
      for (uint32_t i = 0; i < args->num_events; ++i) {
        uint32_t id = ev_data[i].event_id;
        bool in_map = events_.count(id) > 0;
        bool map_signaled = in_map && events_[id].signaled;
        uint64_t slot_val = 0;
        if (event_page_ && id < event_page_size_ / sizeof(uint64_t))
          slot_val = static_cast<uint64_t *>(event_page_)[id];
        util::Logger::vm("WAIT_EVENTS[", i, "] event_id=", id, " in_map=", in_map,
                         " map_signaled=", map_signaled, " slot_val=", slot_val,
                         " event_page_=", (event_page_ ? 1 : 0), " page_size=", event_page_size_,
                         " wait_for_all=", args->wait_for_all, " timeout=", args->timeout);
      }
    }
  }

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
