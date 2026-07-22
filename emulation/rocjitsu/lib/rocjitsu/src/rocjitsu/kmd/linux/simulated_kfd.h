// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_SIMULATED_KFD_H_
#define ROCJITSU_KMD_LINUX_SIMULATED_KFD_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu {

namespace amdgpu {
class Wavefront;
}

namespace kmd {
struct CwsrWaveState;
}
/// @brief 128-bit IPC share handle key, matching the kernel's random handle.
struct IpcHandleKey {
  uint32_t words[4];
  bool operator==(const IpcHandleKey &) const = default;
};

struct IpcHandleKeyHash {
  size_t operator()(const IpcHandleKey &key) const {
    size_t hash_value = std::hash<uint32_t>{}(key.words[0]);
    for (int idx = 1; idx < 4; idx++)
      hash_value ^= std::hash<uint32_t>{}(key.words[idx]) + 0x9e3779b9 + (hash_value << 6) +
                    (hash_value >> 2);
    return hash_value;
  }
};

/// @brief Exported IPC object stored in the driver's global IPC store.
struct IpcObject {
  uint32_t share_handle[4];
  int backing_memfd = -1;
  uint64_t allocation_size = 0;
  uint32_t allocation_flags = 0;
  uint32_t source_gpu_id = 0;
  uint32_t source_process_id = 0;
  uint64_t source_alloc_handle = 0;
};

/// @brief Simulated kernel-mode driver that routes KFD ioctls to the simulator.
///
/// @details Per-process state (allocations, queues, events, doorbells) is held
/// in KfdProcess instances. The driver maintains a process table and resolves
/// the target process from a process_id parameter — matching the real kernel's
/// kfd_chardev_ioctl which resolves kfd_process from filp->private_data.
///
/// The local-mode virtual interface (open/close/ioctl/mmap/munmap) operates on
/// the process created by open(). The daemon uses the process_id-aware
/// overloads so each client thread identifies itself by connection, not by
/// shared mutable state.
class SimulatedKfd : public LinuxKfd {
public:
  /// @brief Test seam invoked between procfs authorization and pidfd revalidation.
  using DebugIdentityValidationHook = std::function<void()>;

  [[nodiscard]] bool daemon_mode() const { return daemon_mode_; }

  SimulatedKfd(SoC &soc, bool daemon_mode = false,
               DebugIdentityValidationHook debug_identity_validation_hook = {});
  SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode = false,
               DebugIdentityValidationHook debug_identity_validation_hook = {});
  ~SimulatedKfd() override;

  /// @brief Local-mode interface (interposer). Operates on the local process.
  /// @{
  int open() override;
  int close() override;

  /// @brief Add one open reference to the local process without re-opening.
  /// @details Used by the interposer when an existing KFD fd is duplicated
  /// (dup/dup2/dup3/fcntl F_DUPFD). Each live fd holds one reference so the
  /// process is torn down only when the last fd is closed, not the first.
  /// @retval true A reference was added.
  /// @retval false No local process to retain (e.g. it was already torn down, or
  ///         daemon/remote mode); the caller must NOT treat the fd as retained.
  [[nodiscard]] bool retain_local_open() override;
  int ioctl(unsigned long request, void *arg) override;
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;
  int munmap(void *addr, size_t length) override;
  /// @}

  /// @brief Daemon-mode process lifecycle. Thread-safe for concurrent clients.
  /// @{

  /// @brief Atomically create a new process and return its ID.
  /// @details Unlike open(), which sets local_process_id_ (not thread-safe for
  /// concurrent daemon clients), this method returns the ID directly so the
  /// caller can associate it with a specific client connection.
  uint32_t open_process(pid_t client_pid = 0);
  void set_process_client_pid(uint32_t process_id, pid_t client_pid);

  int ioctl(uint32_t process_id, unsigned long request, void *arg);
  void *mmap(uint32_t process_id, void *addr, size_t length, int prot, int flags, off_t offset);
  int munmap(uint32_t process_id, void *addr, size_t length);
  int close(uint32_t process_id);

  /// @brief Close every registered process, waking any parked event waiters.
  /// @details Daemon-teardown helper: iterates all live processes and closes each,
  /// firing notify_closing() so client threads blocked in an infinite-timeout
  /// WAIT_EVENTS unblock and their jthread joins can complete.
  void close_all_processes();

  [[nodiscard]] int get_mmap_memfd(uint32_t process_id, off_t offset) const;
  /// @}

  /// @brief Local-mode get_mmap_memfd (uses local process).
  [[nodiscard]] int get_mmap_memfd(off_t offset) const;

  void setup_topology(const Sysfs::GpuInfo &gpu);
  void setup_topology(const config::KfdDeviceConfig &dev, uint32_t num_xcc);
  void setup_topology(const std::vector<config::KfdDeviceConfig> &devs, uint32_t num_xcc);
  bool is_doorbell_range(const void *addr, size_t length) const override;
  uint32_t gpu_id() const { return gpus_.empty() ? 0 : gpus_[0].gpu_id; }
  uint32_t num_gpus() const { return static_cast<uint32_t>(gpus_.size()); }
  const Sysfs &topology() const { return topology_; }
  std::string topology_path() const override { return topology_.path(); }
  std::string drm_path() const override { return topology_.drm_path(); }
  [[nodiscard]] int fd() const override { return fd_.load(std::memory_order_acquire); }
  [[nodiscard]] uint32_t local_process_id() const { return local_process_id_; }

  /// @brief Forget the primary KFD fd number without touching the process.
  /// @details Used by the interposer when dup2/dup3 atomically overwrites the
  /// primary KFD fd number: the number no longer refers to this driver, so stop
  /// classifying it as the local primary (fd() must no longer match it). Only
  /// clears if @p fd matches the current primary, so a stale call is a no-op.
  /// @returns kClearedDropRef if @p fd matched and was cleared (the local primary
  ///          holds one counted open reference, so the caller drops it via
  ///          close()); kNotPrimary if it did not match (a concurrent overwrite
  ///          won the race), so the caller must not release a reference.
  /// @details Runs under process_mutex_ so the CAS on fd_ is serialized with
  /// open()'s fd creation/selection/return: a racing dup2 can no longer clear
  /// fd_ in the window between open() publishing it and open() returning it, so
  /// open() never hands back -1 or an already-overwritten descriptor. Lock-free
  /// readers (driver_fd()/kfd_backend_of()/is_kfd_primary()) still observe fd_
  /// atomically.
  [[nodiscard]] PrimaryInvalidation invalidate_primary_fd(int fd) override;

  /// @brief Open-reference count of the local process, or 0 if none is alive.
  /// @details Introspection for tests/diagnostics. Each live KFD fd (the primary
  /// plus every dup) holds one reference; the process is destroyed at zero.
  [[nodiscard]] uint32_t local_open_ref_count() const;

  [[nodiscard]] bool owns_fd(int fd) const override;
  std::string redirect_sysfs_path(const char *path) const override;
  [[nodiscard]] bool handles_drm_render_minor(uint32_t minor) const override;
  [[nodiscard]] const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const override;
  [[nodiscard]] int claim_fd(int real_fd);
  [[nodiscard]] bool owns_reserved_fd(int fd) const;

  /// @brief Derive the PTE MTYPE from KFD allocation flags (mirrors amdgpu).
  /// @details Public so the interposer's DRM GEM_VA path can install page-table
  /// entries with the same coherency type the KFD alloc requested.
  static amdgpu::Mtype pte_mtype_for_flags(uint32_t alloc_flags);

  /// @brief Install a host range into the local process's GPU page table.
  /// @details Drives DRM AMDGPU_GEM_VA MAP/REPLACE from the interposer: maps
  /// @p size bytes at @p gpu_va to @p host_ptr with the MTYPE derived from
  /// @p alloc_flags.
  /// @retval true the range was installed.
  /// @retval false the local process is gone, so nothing was mapped (the caller
  ///         must surface an error rather than report a phantom success).
  [[nodiscard]] bool gem_va_map(uint64_t gpu_va, void *host_ptr, size_t size, uint32_t alloc_flags);

  /// @brief Remove a GPU page-table range installed by gem_va_map (GEM_VA UNMAP).
  /// @retval true the range was unmapped.
  /// @retval false the local process is gone, so nothing was unmapped.
  [[nodiscard]] bool gem_va_unmap(uint64_t gpu_va, size_t size);

  /// @brief Look up a KfdProcess by ID. Returns nullptr if not found.
  std::shared_ptr<KfdProcess> find_process(uint32_t process_id) const;

  /// @brief KFD allocation flags for a local-process allocation @p handle, or 0.
  /// @details Locks the process alloc mutex internally, keeping lock discipline for
  /// per-process allocation state inside the driver. Used by the interposer to
  /// capture the GPU PTE MTYPE flags at EXPORT_DMABUF time (before the allocation
  /// may be freed).
  uint32_t alloc_flags_for_handle(uint64_t handle) const;

  /// @brief Per-GPU device state (mirrors kfd_dev in the kernel).
  struct GpuDevice {
    SoC *soc = nullptr;
    uint32_t gpu_id = 0;
    bool cps_initialized = false;
    kfd_process_device_apertures apertures{};
  };

private:
  /// @brief Look up the local-mode process.
  std::shared_ptr<KfdProcess> find_local_process() const;

  /// @brief Look up a KfdProcess by its client (Linux) pid. Used by
  /// AMDKFD_IOC_DBG_TRAP to resolve the debug target, mirroring the kernel's
  /// kfd_lookup_process_by_pid().
  /// @param pid Client (Linux) pid of the target process.
  /// @return The matching KfdProcess, or nullptr if none matches.
  std::shared_ptr<KfdProcess> find_process_by_client_pid(pid_t pid) const;

  /// @brief Look up a GpuDevice by gpu_id. Returns nullptr if not found.
  GpuDevice *find_gpu(uint32_t gpu_id);
  const GpuDevice *find_gpu(uint32_t gpu_id) const;

  /// @brief Get the ordinal (0-based index) for a gpu_id. Returns 0 if not found.
  uint32_t gpu_ordinal(uint32_t gpu_id) const {
    for (uint32_t i = 0; i < gpus_.size(); ++i)
      if (gpus_[i].gpu_id == gpu_id)
        return i;
    return 0;
  }

  void map_to_gpu(KfdProcess &proc, uint64_t gpu_va, void *host_ptr, size_t size,
                  amdgpu::Mtype mtype = amdgpu::Mtype::RW);
  void unmap_from_gpu(KfdProcess &proc, uint64_t gpu_va, size_t size);

  void update_cp_doorbell_base(uint32_t gpu_ordinal, uint32_t process_id, void *base);

  int dispatch_ioctl(KfdProcess &proc, unsigned long request, void *arg);
  void *dispatch_mmap(KfdProcess &proc, void *addr, size_t length, int prot, int flags,
                      off_t offset);
  int dispatch_munmap(KfdProcess &proc, void *addr, size_t length);
  int dispatch_get_mmap_memfd(KfdProcess &proc, off_t offset) const;

  int get_process_apertures_ioctl(void *arg) override;
  int acquire_vm_ioctl(void *arg) override;
  int get_available_memory_ioctl(void *arg) override;
  int set_memory_policy_ioctl(void *arg) override;
  int alloc_memory_ioctl(void *arg) override;
  int free_memory_ioctl(void *arg) override;
  int map_memory_ioctl(void *arg) override;
  int unmap_memory_ioctl(void *arg) override;
  int get_available_memory_ioctl(KfdProcess &proc, void *arg);
  int alloc_memory_ioctl(KfdProcess &proc, void *arg);
  int free_memory_ioctl(KfdProcess &proc, void *arg);
  int map_memory_ioctl(KfdProcess &proc, void *arg);
  int unmap_memory_ioctl(KfdProcess &proc, void *arg);
  int create_queue_ioctl(KfdProcess &proc, void *arg);
  int update_queue_ioctl(KfdProcess &proc, void *arg);
  int destroy_queue_ioctl(KfdProcess &proc, void *arg);
  int create_event_ioctl(KfdProcess &proc, void *arg);
  int set_memory_policy_ioctl(KfdProcess &proc, void *arg);
  int destroy_event_ioctl(KfdProcess &proc, void *arg);
  int set_event_ioctl(KfdProcess &proc, void *arg);
  int reset_event_ioctl(KfdProcess &proc, void *arg);
  int wait_events_ioctl(KfdProcess &proc, void *arg);
  int import_dmabuf_ioctl(KfdProcess &proc, void *arg);
  int export_dmabuf_ioctl(KfdProcess &proc, void *arg);
  int get_dmabuf_info_ioctl(KfdProcess &proc, void *arg);
  int ipc_export_handle_ioctl(KfdProcess &proc, void *arg);
  int ipc_import_handle_ioctl(KfdProcess &proc, void *arg);
  int svm_ioctl(KfdProcess &proc, void *arg);
  int runtime_enable_ioctl(KfdProcess &proc, void *arg);
  int debug_trap_ioctl(KfdProcess &caller, void *arg);
  void reap_exited_debug_sessions(std::stop_token stop);
  int debug_device_snapshot(kfd_ioctl_dbg_trap_device_snapshot_args &args);
  int debug_queue_snapshot(KfdProcess *target, kfd_ioctl_dbg_trap_queue_snapshot_args &args);
  int debug_query_event(pid_t target_pid, KfdProcess *target_proc, uint64_t enabled_mask,
                        kfd_ioctl_dbg_trap_query_debug_event_args &args);
  int debug_query_exception_info(pid_t target_pid,
                                 kfd_ioctl_dbg_trap_query_exception_info_args &args);
  void raise_process_debug_event(pid_t target_pid, uint64_t exception_mask);
  void runtime_enable_debugger_handshake(pid_t target_pid);

  std::optional<amdgpu::ComputeUnitCore::TrapHandlerConfig>
  resolve_trap_handler(const amdgpu::Wavefront &wf, uint32_t gpu_ordinal);
  bool on_wave_sendmsg(amdgpu::Wavefront &wf, uint32_t message);
  void on_wave_trap_complete(amdgpu::Wavefront &wf);

  /// @brief True if any wave of the given queue is still executing (may yet trap
  /// or complete). Used to defer the debugger report until all waves of a
  /// multi-wave dispatch have stopped, so they are serialized atomically.
  bool queue_has_running_waves(uint32_t process_id, uint32_t queue_id, uint32_t gpu_id,
                               const amdgpu::Wavefront *exclude);

  bool on_wave_single_step_complete(amdgpu::Wavefront &wf);
  void report_wave_stopped(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                           uint32_t gpu_id, uint64_t ctx_base, uint32_t ctx_size,
                           uint64_t exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP));
  void resume_debug_queues(KfdProcess *proc);
  void apply_cwsr_to_wave(amdgpu::Wavefront &wf, const kmd::CwsrWaveState &state);

  /// @brief Address-watchpoint handler installed on every compute unit.
  /// @details Runs on the engine thread for each active-lane global-memory
  /// access. If the address matches one of the debugger's address watchpoints
  /// (compared under its mask, with a matching access mode), stops the wave with
  /// the corresponding TRAPSTS.addr_watch bit set and reports it. Returns true
  /// if the wave was stopped. @param addr post-translation global address;
  /// @param bytes access width; @param is_write store/atomic; @param is_atomic
  /// atomic read-modify-write.
  bool on_wave_watchpoint(amdgpu::Wavefront &wf, uint64_t addr, uint32_t bytes, bool is_write,
                          bool is_atomic);
  bool on_wave_illegal_instruction(amdgpu::Wavefront &wf);

  /// @brief Memory-violation handler installed on every compute unit.
  /// @details Runs on the engine thread for a global-memory access to an
  /// unmapped address. If the wave's process is being debugged, sets
  /// TRAPSTS.xnack_error, stops the wave, and reports an
  /// EC_QUEUE_WAVE_MEMORY_VIOLATION exception (rocm-dbgapi
  /// WAVE_STOP_REASON_MEMORY_VIOLATION); returns true, else false.
  bool on_wave_memory_violation(amdgpu::Wavefront &wf, uint64_t addr, bool is_write);

  /// @brief Toggle the per-access debugger checks on every compute unit. Set
  /// true while any debug session is active so undebugged runs pay no
  /// per-access cost (ComputeUnitCore::set_debug_active).
  void set_debug_active_on_all_cus(bool active);

  /// @brief Serialize all debug-halted waves of a queue into its CWSR area.
  void serialize_queue_debug_waves(uint32_t process_id, uint32_t queue_id, uint32_t gpu_id,
                                   uint64_t ctx_base, uint32_t ctx_size);

  /// @brief Stop the target's running waves and refresh their CWSR areas
  /// (KFD_IOC_DBG_TRAP_SUSPEND_QUEUES).
  void suspend_debug_queues(KfdProcess *proc);

  /// @brief Clear the CWSR area of any of the target's queues whose stopped waves
  /// have completed, so rocm-dbgapi prunes them cleanly instead of reading a
  /// stale wave against an advanced read_dispatch_id (KFD_IOC_DBG_TRAP_SUSPEND).
  void clear_completed_debug_queues(KfdProcess *proc);
  /// @brief Record a debug exception on a queue and reflect it on the snapshot.
  void raise_debug_event(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                         uint32_t gpu_id, uint64_t exception_mask);

  /// @brief Compute the LDS/scratch/GPUVM apertures for a GPU ordinal.
  /// @details Each further ordinal shifts the per-GPU LDS/scratch windows by
  /// @ref kApertureStride. Shared by
  /// AMDKFD_IOC_GET_PROCESS_APERTURES_NEW and the debug device snapshot.
  kfd_process_device_apertures gpu_apertures(uint32_t ordinal) const;
  int set_xnack_mode_ioctl(void *arg);
  int get_tile_config_ioctl(void *arg);
  bool allocate_scratch_backing(uint32_t process_id, uint64_t gpu_va, size_t size);

  /// @brief Lazily create the backing memfd exactly once across racing opens.
  /// @details CAS-publishes fd_ so concurrent open()/open_process() callers agree
  /// on a single memfd; losers close their own and adopt the winner's.
  /// @retval true fd_ holds a valid descriptor. @retval false memfd_create failed.
  [[nodiscard]] bool ensure_fd_created();

  /// @brief One-time per-GPU CP setup: apertures + interrupt/scratch callbacks.
  /// @details Idempotent per GpuDevice via the cps_initialized flag. The caller
  /// MUST hold process_mutex_ so the check-and-set of cps_initialized is atomic
  /// against concurrent daemon opens that would otherwise both register callbacks.
  void init_command_processors_locked();

  std::vector<GpuDevice> gpus_;
  bool daemon_mode_ = false;
  std::atomic<int> fd_{-1};

  /// @brief Process table mapping process_id to KfdProcess.
  /// @details Protected by process_mutex_ for concurrent daemon access.
  ///
  /// Global lock ordering (acquire in this order; never the reverse).
  /// op_mutex_ (KfdProcess) is the outermost per-process ioctl lock. Under it,
  /// process_mutex_ and alloc_mutex_ are independent siblings — they are NEVER
  /// held simultaneously (allocate_scratch_backing and close() both release
  /// process_mutex_ before taking alloc_mutex_):
  ///   op_mutex_ < process_mutex_
  ///   op_mutex_ < alloc_mutex_ < {ipc_mutex_, page_table_mutex_, owned_fds_mutex_}
  ///   op_mutex_ < runtime_mutex_ < alloc_mutex_        (runtime_enable_ioctl)
  ///   op_mutex_ < debug_sessions_mutex_ < runtime_mutex_ (debug_trap_ioctl)
  ///   process_mutex_ < interrupt_mutex_                (open()/open_process())
  /// The op_mutex_ in the debug rule is always the CALLER's, while runtime_mutex_
  /// may belong to a DIFFERENT process (the debug target resolved by client pid).
  /// debug_trap_ioctl holds only the caller's op_mutex_ and never acquires the
  /// target's op_mutex_, so a cross-process attach cannot deadlock.
  /// interrupt_mutex_ is a leaf: the CP interrupt callback takes it and only
  /// descends into EventState::mutex_, and close() takes it only after releasing
  /// process_mutex_, so there is no cycle.
  /// The CP engine thread acquires hw_queue_mutex_ first, then reaches
  /// process_mutex_ (scratch resolver) or alloc_mutex_ (scratch allocator) through
  /// the callbacks — hw_queue_mutex_ -> process_mutex_ and, separately,
  /// hw_queue_mutex_ -> alloc_mutex_ (never both nested). To avoid an ABBA against
  /// that thread, an ioctl MUST NOT hold a per-process lock (alloc_mutex_) across a
  /// CommandProcessor call that takes hw_queue_mutex_ (e.g. register_queue) — build
  /// state under alloc_mutex_, release it, then call the CP.
  mutable std::mutex process_mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<KfdProcess>> processes_;
  uint32_t next_process_id_ = 1;

  /// @brief Debugger sessions keyed by the target inferior's Linux pid.
  /// @details Decoupled from KfdProcess so a debugger (rocgdb) can enable a
  /// session on an inferior before the inferior opens /dev/kfd, mirroring the
  /// kernel creating the target kfd_process in the DBG_TRAP_ENABLE path.
  mutable std::mutex debug_sessions_mutex_;
  std::unordered_map<pid_t, KfdProcess::DebugSession> debug_sessions_;
  DebugIdentityValidationHook debug_identity_validation_hook_;
  std::condition_variable_any debug_sessions_cv_;
  std::jthread debug_session_reaper_;

  /// @brief Pending debug exceptions per target, grouped by queue.
  /// @details Populated by wave traps (engine thread) and drained by
  /// KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT (ioctl thread). Never held together
  /// with debug_sessions_mutex_ by the wave-trap path (lock ordering).
  struct DebugQueueException {
    uint32_t gpu_id = 0;
    uint64_t mask = 0;
  };
  mutable std::mutex debug_events_mutex_;
  std::unordered_map<pid_t, std::unordered_map<uint32_t, DebugQueueException>> debug_events_;

  mutable std::mutex runtime_handshake_mutex_;
  std::condition_variable runtime_handshake_cv_;
  std::unordered_set<pid_t> runtime_acked_;

  /// @brief Interrupt dispatch: process_id → EventState*.
  /// @details Protected by interrupt_mutex_. Decoupled from process_mutex_
  /// to avoid ABBA deadlocks with hw_queue_mutex_ in the CP doorbell thread.
  mutable std::mutex interrupt_mutex_;
  std::unordered_map<uint32_t, EventState *> event_dispatch_;

  /// @brief Process ID for local-mode (interposer). Set once in open().
  /// @details Written under process_mutex_ in open(), then read lock-free from the
  /// ioctl/mmap/munmap/gem paths. Safe only under the local-mode contract: the app
  /// opens /dev/kfd before issuing any ioctl and never re-opens concurrently, so the
  /// single publishing write happens-before every read. Not for use outside that
  /// single-primary-fd local path.
  uint32_t local_process_id_ = 0;

  /// @brief Per-ordinal shift applied to the LDS and scratch aperture windows
  /// so each GPU in a multi-GPU process gets a distinct range.
  static constexpr uint64_t kApertureStride = 0x10000000000ULL;

  /// @brief IPC handle store for cross-process memory sharing.
  /// @details Lock ordering: process_mutex_ < alloc_mutex_ < ipc_mutex_.
  mutable std::mutex ipc_mutex_;
  std::unordered_map<IpcHandleKey, IpcObject, IpcHandleKeyHash> ipc_store_;

  mutable std::mutex owned_fds_mutex_;
  std::unordered_set<int> owned_fds_;

  Sysfs topology_;

  /// @brief Per-GPU topology info captured at setup_topology, indexed to match
  /// gpus_. Feeds the AMDKFD_IOC_DBG_TRAP device snapshot.
  std::vector<Sysfs::GpuInfo> gpu_infos_;

  static constexpr int kReservedFdCount = 256;
  int reserved_fd_base_ = 0;
  int next_reserved_fd_ = 0;

  void init_reserved_fd_range();
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_SIMULATED_KFD_H_
