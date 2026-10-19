// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/host_access_guard.h"
#include "rocjitsu/kmd/linux/host_mapping_lock.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/private_gtt_copy_source_gate.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gws_device.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/iod.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/canonical_backing_id.h"
#include "rocjitsu/vm/fabric_backing_kfd.h"
#include "rocjitsu/vm/fabric_endpoint.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <format>
#include <iterator>
#include <limits>
#include <linux/sched.h>
#include <linux/types.h>
#include <new>
#include <poll.h>
#include <sstream>
#include <string_view>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
// pidfd_open(2) landed in Linux 5.3. The build hosts are older than their
// kernels here -- gcc-toolset-13 on the CI base image ships sanitized headers
// that predate it -- so the wrapper and even __NR_pidfd_open can be missing
// while the running kernel implements it fine. 434 is the number on every
// architecture ROCm targets; the syscall was added after new numbers began
// being allocated identically across arches.
#ifndef SYS_pidfd_open
#ifdef __NR_pidfd_open
#define SYS_pidfd_open __NR_pidfd_open
#else
#define SYS_pidfd_open 434
#endif
#endif
#ifndef SYS_clone3
#ifdef __NR_clone3
#define SYS_clone3 __NR_clone3
#else
#define SYS_clone3 435
#endif
#endif
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {

std::optional<amdgpu::MemoryIngressPermit>
PeerCanonicalLineGate::try_acquire(uint64_t backing_id, uint64_t offset, size_t size) {
  constexpr uint64_t kLineBytes = amdgpu::L2Cache::LINE_SIZE;
  static_assert((kLineBytes & (kLineBytes - 1)) == 0);
  if (backing_id == 0 || size == 0 || size - 1 > UINT64_MAX - offset)
    return std::nullopt;
  const uint64_t first = offset & ~(kLineBytes - 1);
  const uint64_t last = (offset + size - 1) & ~(kLineBytes - 1);
  std::vector<Key> keys;
  keys.reserve(static_cast<size_t>((last - first) / kLineBytes + 1));
  for (uint64_t line = first;; line += kLineBytes) {
    keys.emplace_back(backing_id, line);
    if (line == last)
      break;
  }

  std::optional<amdgpu::MemoryIngressPermit> permit;
  {
    std::lock_guard lock(mutex_);
    if (std::ranges::any_of(keys, [&](const Key &key) { return reserved_.contains(key); }))
      return std::nullopt;
    try {
      for (const Key &key : keys)
        reserved_.insert(key);
      // Construct the release closure while holding the lock. If allocation
      // fails, the catch removes every newly inserted key before publication.
      permit.emplace([self = shared_from_this(), keys] { self->release(keys); });
    } catch (...) {
      for (const Key &key : keys)
        reserved_.erase(key);
      throw;
    }
  }
  return permit;
}

bool PeerCanonicalLineGate::overlaps(uint64_t backing_id, uint64_t offset, size_t size) const {
  constexpr uint64_t kLineBytes = amdgpu::L2Cache::LINE_SIZE;
  if (backing_id == 0 || size == 0 || size - 1 > UINT64_MAX - offset)
    return false;
  const uint64_t first = offset & ~(kLineBytes - 1);
  const uint64_t last = (offset + size - 1) & ~(kLineBytes - 1);
  std::lock_guard lock(mutex_);
  const auto found = reserved_.lower_bound({backing_id, first});
  return found != reserved_.end() && found->first == backing_id && found->second <= last;
}

void PeerCanonicalLineGate::release(std::span<const Key> keys) noexcept {
  std::lock_guard lock(mutex_);
  for (const Key &key : keys)
    reserved_.erase(key);
}

namespace {

std::atomic<uint64_t> next_process_incarnation{1};

uint64_t allocate_internal_identity(std::atomic<uint64_t> &next) noexcept {
  uint64_t current = next.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<uint64_t>::max()) {
    if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
      return current;
  }
  // Wrapping would let a retired backing or VM binding match a new lifetime.
  std::terminate();
}

#ifndef CLONE_ARGS_SIZE_VER0
// Older build headers may lack clone_args even when the host supports clone3.
// The original 64-byte ABI is sufficient for the flags and pidfd used below.
struct clone_args {
  __aligned_u64 flags;
  __aligned_u64 pidfd;
  __aligned_u64 child_tid;
  __aligned_u64 parent_tid;
  __aligned_u64 exit_signal;
  __aligned_u64 stack;
  __aligned_u64 stack_size;
  __aligned_u64 tls;
};
static_assert(sizeof(clone_args) == 64);
#endif

bool vm_trace_enabled() {
  static const bool enabled = (std::getenv("RJ_VMEM_TRACE") != nullptr);
  return enabled;
}

constexpr uint32_t kTileConfigCount = 32;
constexpr uint32_t kMacroTileConfigCount = 16;
constexpr uint32_t kMinimumQueueRingSize = 1024;

std::optional<uint32_t> normalize_queue_ring_size(uint32_t ring_size) {
  if (ring_size != 0 && !std::has_single_bit(ring_size))
    return std::nullopt;
  return std::max(ring_size, kMinimumQueueRingSize);
}

/// @brief Return a KFD doorbell slot unless queue publication commits it.
class DoorbellReservation {
public:
  DoorbellReservation(std::mutex &mutex, std::vector<uint32_t> &free_offsets, uint64_t &next_offset,
                      uint32_t offset, bool recycled)
      : mutex_(mutex), free_offsets_(free_offsets), next_offset_(next_offset), offset_(offset),
        recycled_(recycled) {}
  DoorbellReservation(const DoorbellReservation &) = delete;
  DoorbellReservation &operator=(const DoorbellReservation &) = delete;

  ~DoorbellReservation() noexcept {
    if (committed_)
      return;
    std::lock_guard lock(mutex_);
    if (recycled_) {
      assert(free_offsets_.size() < free_offsets_.capacity());
      free_offsets_.push_back(offset_);
    } else {
      assert(next_offset_ == static_cast<uint64_t>(offset_) + sizeof(uint64_t));
      next_offset_ = offset_;
    }
  }

  void commit() { committed_ = true; }

private:
  std::mutex &mutex_;
  std::vector<uint32_t> &free_offsets_;
  uint64_t &next_offset_;
  uint32_t offset_;
  bool recycled_;
  bool committed_ = false;
};

constexpr KfdProcess::GpuGrantDomain grant_domain_for_flags(uint32_t flags) {
  constexpr uint32_t kMemoryKind =
      KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
  return (flags & kMemoryKind) == KFD_IOC_ALLOC_MEM_FLAGS_VRAM &&
                 !(flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
             ? KfdProcess::GpuGrantDomain::Vram
             : KfdProcess::GpuGrantDomain::System;
}

} // namespace

amdgpu::Mtype SimulatedKfd::pte_mtype_for_flags(uint32_t flags, bool shared) {
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    return amdgpu::Mtype::UC;
  if (flags & (KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR))
    return amdgpu::Mtype::UC;
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
    return amdgpu::Mtype::UC;
  if (shared || (flags & KFD_IOC_ALLOC_MEM_FLAGS_COHERENT))
    return amdgpu::Mtype::CC;
  return amdgpu::Mtype::RW;
}

SimulatedKfd::GemVaTransaction::GemVaTransaction(std::shared_ptr<KfdProcess> process)
    : process_(std::move(process)), transaction_(process_->begin_page_table_transaction()) {}

int SimulatedKfd::GemVaTransaction::preflight_map(uint64_t gpu_va, size_t size,
                                                  uint64_t drm_file_id) {
  if (process_->event_state_.is_closing())
    return -ENODEV;
  const auto owner = KfdProcess::VaOwner{KfdProcess::VaOwnerKind::Gem, drm_file_id};
  const auto resident = process_->has_foreign_resident_overlap(transaction_, gpu_va, size, owner);
  if (resident != KfdProcess::PageTableMutationResult::Applied ||
      process_->active_gpu_grant_overlaps(transaction_, gpu_va, size))
    return -EINVAL;
  return 0;
}

int SimulatedKfd::GemVaTransaction::map(uint64_t gpu_va, void *host_ptr, size_t size,
                                        uint32_t alloc_flags, bool sealed_ram,
                                        uint64_t drm_file_id) {
  if (const int preflight = preflight_map(gpu_va, size, drm_file_id); preflight != 0)
    return preflight;
  // GEM_VA uses the interposer's private read-write dmabuf mapping, whose
  // lifetime is tied to the GEM entry. It is not the application's CPU alias.
  const auto result = process_->map_pages(transaction_, gpu_va, host_ptr, size,
                                          SimulatedKfd::pte_mtype_for_flags(alloc_flags),
                                          sealed_ram ? KfdProcess::HostExtentOwner::DriverSealedRam
                                                     : KfdProcess::HostExtentOwner::Driver,
                                          {KfdProcess::VaOwnerKind::Gem, drm_file_id});
  return result == KfdProcess::PageTableMutationResult::Applied ? 0 : -EINVAL;
}

bool SimulatedKfd::GemVaTransaction::unmap(uint64_t gpu_va, size_t size, uint64_t drm_file_id) {
  return process_->unmap_pages(transaction_, gpu_va, size,
                               {KfdProcess::VaOwnerKind::Gem, drm_file_id}) ==
         KfdProcess::PageTableMutationResult::Applied;
}

std::unique_ptr<SimulatedKfd::GemVaTransaction> SimulatedKfd::begin_gem_va_transaction() {
  auto process = find_process(local_process_id_);
  if (!process)
    return nullptr;
  return std::make_unique<GemVaTransaction>(std::move(process));
}

int SimulatedKfd::submit_pm4(uint32_t render_minor, uint64_t queue_key,
                             amdgpu::Pm4Submission submission) {
  const auto process = find_process(local_process_id_);
  if (!process)
    return -ENODEV;
  std::lock_guard op_lock(process->op_mutex_);
  if (process->event_state_.is_closing())
    return -ENODEV;
  const auto ordinal = gpu_ordinal_for_render_minor(render_minor);
  if (!ordinal)
    return -ENODEV;
  std::lock_guard lock(pm4_mutex_);
  auto it = pm4_queues_.find(queue_key);
  if (it == pm4_queues_.end()) {
    auto *cp = gpus_[*ordinal].soc->assign_queue_owner_cp(0);
    if (!cp)
      return -ENODEV;
    amdgpu::ComputeQueueConfig queue;
    queue.address_space = process->gpu(*ordinal).address_space;
    queue.process_id = local_process_id_;
    queue.queue_id = next_pm4_queue_id_++;
    const uint32_t queue_id = queue.queue_id;
    if (!cp->register_drm_queue(std::move(queue)))
      return -EIO;
    it = pm4_queues_.emplace(queue_key, std::pair{cp, queue_id}).first;
  }
  return it->second.first->submit_pm4(it->second.second, local_process_id_, std::move(submission))
             ? 0
             : -EIO;
}

void SimulatedKfd::retire_pm4_queue(uint64_t queue_key) {
  std::lock_guard lock(pm4_mutex_);
  const auto it = pm4_queues_.find(queue_key);
  if (it == pm4_queues_.end())
    return;
  it->second.first->unregister_drm_queue(it->second.second, local_process_id_);
  pm4_queues_.erase(it);
}

namespace {

/// @brief mmap via the real libc, bypassing the interposer.
/// @details Routes through the process-wide libc_passthrough() table so the
/// driver's own mappings never re-enter the interposer's mmap hook. The table is
/// resolved once in the SimulatedKfd constructor.
///
/// Bypassing the interposer also bypasses the interposer's mapping lock, and an
/// ioctl-routed mmap reaches here without ever passing through it. So it takes
/// the lock itself: an emulated atomic holds a raw pointer across a permission
/// check and the store it authorises, and a replacement in between would land
/// that store in whatever took the page's place.
///
/// Held for the syscall alone. The driver's surrounding mapping work re-enters
/// the memory model and takes its VMID lock, which an atomic already holds when
/// it reaches this lock, so widening the region would invert the two orders.
void *safe_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  auto mapping_lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return libc_passthrough().mmap(addr, length, prot, flags, fd, offset);
}

/// @brief munmap via the real libc, with the mapping layout held still.
/// @details Withdrawing a page is the sharper half of the hazard safe_mmap()
/// describes: an atomic that has already validated the page is left storing
/// through a pointer to nothing. Allocation teardown drops the GPU page-table
/// entry first, so a concurrent access can miss the page table, fall through to
/// the identity path, and validate the very host page this call removes.
int safe_munmap(void *addr, size_t length) {
  auto mapping_lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return libc_passthrough().munmap(addr, length);
}

/// @brief mprotect via the real libc, with the mapping layout held still.
/// @details Revoking write permission is as damaging to an in-flight atomic as
/// removing the page, and for the same reason.
int safe_mprotect(void *addr, size_t length, int prot) {
  auto mapping_lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return libc_passthrough().mprotect(addr, length, prot);
}

/// @brief Return whether two non-empty half-open address ranges overlap.
/// @details Uses subtraction instead of computing either end address, avoiding
/// integer overflow for a malformed range near UINTPTR_MAX.
bool ranges_overlap(const void *lhs, size_t lhs_size, const void *rhs, size_t rhs_size) {
  if (lhs_size == 0 || rhs_size == 0)
    return false;
  const auto lhs_base = reinterpret_cast<uintptr_t>(lhs);
  const auto rhs_base = reinterpret_cast<uintptr_t>(rhs);
  return lhs_base <= rhs_base ? rhs_base - lhs_base < lhs_size : lhs_base - rhs_base < rhs_size;
}

std::vector<GpuAliasIngressGate::Key>
allocation_alias_keys(const KfdProcess &proc, const KfdProcess::GpuAllocation &alloc,
                      std::vector<KfdProcess::GpuAliasRecord> *records = nullptr) {
  std::vector<GpuAliasIngressGate::Key> keys;
  const auto owner = alloc.backing_owner ? alloc.backing_owner : alloc.local_backing_owner;
  if (!owner)
    return keys;
  const auto aliases =
      alloc.backing_owner ? proc.gpu_alias_snapshot() : proc.gpu_local_alias_snapshot();
  keys.reserve(alloc.admission_generations.size());
  for (const auto &alias : *aliases) {
    const auto generation = alloc.admission_generations.find(alias.gpu_id);
    if (alias.gpu_va == alloc.gpu_va && alias.extent == alloc.size &&
        alias.backing_id == owner->backing_id() &&
        generation != alloc.admission_generations.end() &&
        alias.admission_generation == generation->second) {
      keys.push_back(GpuAliasIngressGate::key(alias));
      if (records)
        records->push_back(alias);
    }
  }
  return keys;
}

bool allocation_has_closed_alias(const KfdProcess &proc, const KfdProcess::GpuAllocation &alloc,
                                 const GpuAliasIngressGate &gate) {
  const auto owner = alloc.backing_owner ? alloc.backing_owner : alloc.local_backing_owner;
  if (!owner)
    return false;
  const auto aliases =
      alloc.backing_owner ? proc.gpu_alias_snapshot() : proc.gpu_local_alias_snapshot();
  return std::ranges::any_of(*aliases, [&](const auto &alias) {
    const auto generation = alloc.admission_generations.find(alias.gpu_id);
    return alias.gpu_va == alloc.gpu_va && alias.extent == alloc.size &&
           alias.backing_id == owner->backing_id() &&
           generation != alloc.admission_generations.end() &&
           alias.admission_generation == generation->second &&
           gate.closed(GpuAliasIngressGate::key(alias));
  });
}

bool needs_prepared_local_sink(const KfdProcess::GpuAllocation &alloc) {
  // GPU effects require a published grant. A view that predates the first
  // grant can change, but an old grant may leave dirty lines after UNMAP.
  return alloc.host_ptr && !alloc.backing_owner &&
         (alloc.ever_gpu_admitted || alloc.driver_gpu_mapping ||
          !alloc.admission_generations.empty()) &&
         ((alloc.ipc_shared && !alloc.ipc_export_retains_local_policy) ||
          SimulatedKfd::pte_mtype_for_flags(alloc.flags) != amdgpu::Mtype::UC) &&
         (!alloc.local_backing_owner || alloc.memfd < 0 || !alloc.host_ptr_owned);
}

template <typename Check>
int acquire_mmap_op(std::unique_lock<std::mutex> &op_lock, Check &&check) {
  for (unsigned attempts = 0;; ++attempts) {
    if (op_lock.try_lock())
      return 0;
    if (const int result = check(); result != 0)
      return result;
    if (attempts < 64)
      std::this_thread::yield();
    else
      std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

class PendingAliasPublicationWriter {
public:
  PendingAliasPublicationWriter(std::atomic<uint32_t> *process_count,
                                std::atomic<uint32_t> &writers,
                                std::atomic<uint64_t> &release_epoch)
      : writers_(writers), release_epoch_(release_epoch) {
    writers_.fetch_add(1, std::memory_order_acq_rel);
    arm(process_count);
  }
  void arm(std::atomic<uint32_t> *process_count) noexcept {
    if (!process_count || process_count_)
      return;
    process_count_ = process_count;
    process_count_->fetch_add(1, std::memory_order_acq_rel);
    // Readers already waiting for this writer must recheck the process count.
    release_epoch_.fetch_add(1, std::memory_order_acq_rel);
    release_epoch_.notify_all();
  }
  ~PendingAliasPublicationWriter() {
    if (process_count_)
      process_count_->fetch_sub(1, std::memory_order_release);
    writers_.fetch_sub(1, std::memory_order_release);
    release_epoch_.fetch_add(1, std::memory_order_acq_rel);
    release_epoch_.notify_all();
  }
  PendingAliasPublicationWriter(const PendingAliasPublicationWriter &) = delete;
  PendingAliasPublicationWriter &operator=(const PendingAliasPublicationWriter &) = delete;

private:
  std::atomic<uint32_t> *process_count_ = nullptr;
  std::atomic<uint32_t> &writers_;
  std::atomic<uint64_t> &release_epoch_;
};

/// @brief Return whether one non-empty address range fully contains another.
/// @details Like ranges_overlap(), avoids computing an end address so malformed
/// ranges near UINTPTR_MAX cannot wrap around.
bool range_contains(const void *outer, size_t outer_size, const void *inner, size_t inner_size) {
  if (outer_size == 0 || inner_size == 0)
    return false;
  const auto outer_base = reinterpret_cast<uintptr_t>(outer);
  const auto inner_base = reinterpret_cast<uintptr_t>(inner);
  if (inner_base < outer_base)
    return false;
  const auto offset = inner_base - outer_base;
  return offset <= outer_size && inner_size <= outer_size - offset;
}

/// @brief Move-only owner for an mmap until it is published into driver state.
class UniqueMapping {
public:
  UniqueMapping() = default;
  UniqueMapping(void *addr, size_t size) : addr_(addr), size_(size) {}
  ~UniqueMapping() { reset(); }

  UniqueMapping(const UniqueMapping &) = delete;
  UniqueMapping &operator=(const UniqueMapping &) = delete;

  UniqueMapping(UniqueMapping &&other) noexcept
      : addr_(std::exchange(other.addr_, MAP_FAILED)), size_(std::exchange(other.size_, 0)) {}
  UniqueMapping &operator=(UniqueMapping &&other) noexcept {
    if (this != &other) {
      reset();
      addr_ = std::exchange(other.addr_, MAP_FAILED);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  [[nodiscard]] explicit operator bool() const { return addr_ != MAP_FAILED; }
  [[nodiscard]] void *get() const { return addr_; }
  [[nodiscard]] void *release() {
    size_ = 0;
    return std::exchange(addr_, MAP_FAILED);
  }
  void reset(void *addr = MAP_FAILED, size_t size = 0) {
    if (addr_ != MAP_FAILED)
      safe_munmap(addr_, size_);
    addr_ = addr;
    size_ = size;
  }

private:
  void *addr_ = MAP_FAILED;
  size_t size_ = 0;
};

/// @brief fstat via the real libc, bypassing the interposer.
/// @details Like safe_mmap: the interposer exports fstat with default visibility,
/// so a bare fstat() from this TU binds to our own hook (which takes fd_mutex_ via
/// is_drm()). Routing through the passthrough table keeps "the driver never
/// re-enters the interposer" total and avoids acquiring fd_mutex_ under a held
/// per-process lock (alloc_mutex_/etc.).
int safe_fstat(int fd, struct stat *st) { return libc_passthrough().fstat_fn(fd, st); }

/// @brief fcntl via the real libc, bypassing the interposer.
/// @details The interposer's fcntl hook takes fd_mutex_ on F_DUPFD paths; calling
/// it from the driver while holding a per-process lock is a latent lock-order
/// inversion. The passthrough table's fcntl is variadic; the int-arg forms
/// (F_DUPFD_CLOEXEC, F_ADD_SEALS, F_GETFL/no-arg) used here forward cleanly.
template <typename... Args> int safe_fcntl(int fd, int cmd, Args... args) {
  return libc_passthrough().fcntl(fd, cmd, args...);
}

constexpr auto kDebugNotificationWriteTimeout = std::chrono::milliseconds(250);

/// @brief Reap notification helpers that had not exited when their deadline expired.
/// @details A timed-out writer must never make the engine thread wait indefinitely
/// after SIGKILL: a task in uninterruptible kernel sleep can remain alive until the
/// operation completes. The engine performs one nonblocking wait and transfers any
/// still-live child here. This process-owned thread uses only WNOHANG, so shutdown is
/// bounded even if the host operation never returns.
class NotificationWriterReaper {
public:
  NotificationWriterReaper()
      : owner_pid_(getpid()), worker_([this](std::stop_token stop) { reap_until_stopped(stop); }) {}

  [[nodiscard]] pid_t owner_pid() const { return owner_pid_; }

  void adopt(pid_t writer) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      writers_.push_back(writer);
    }
    cv_.notify_one();
  }

private:
  void reap_until_stopped(std::stop_token stop) {
    while (!stop.stop_requested()) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, stop, [this] { return !writers_.empty(); });
        if (stop.stop_requested())
          break;

        // Keep allocations owned by the process-lifetime reaper: a fork child
        // loses this worker's stack and cannot reclaim thread-local vectors.
        std::erase_if(writers_, [](pid_t writer) {
          int status = 0;
          const pid_t result = ::waitpid(writer, &status, WNOHANG | __WCLONE);
          return result != 0 && !(result < 0 && errno == EINTR);
        });
        if (writers_.empty())
          continue;
      }
      static_cast<void>(::poll(nullptr, 0, 10));
    }
  }

  std::mutex mutex_;
  std::condition_variable_any cv_;
  std::vector<pid_t> writers_;
  const pid_t owner_pid_;
  // Declared last so it stops and joins before the state it accesses is destroyed.
  std::jthread worker_;
};

constinit std::atomic<NotificationWriterReaper *> g_notification_writer_reaper{nullptr};

NotificationWriterReaper &notification_writer_reaper() {
  const pid_t owner_pid = getpid();
  NotificationWriterReaper *reaper = g_notification_writer_reaper.load(std::memory_order_acquire);
  if (reaper != nullptr && reaper->owner_pid() == owner_pid)
    return *reaper;

  auto *replacement = new NotificationWriterReaper();
  while (!g_notification_writer_reaper.compare_exchange_weak(
      reaper, replacement, std::memory_order_release, std::memory_order_acquire)) {
    if (reaper != nullptr && reaper->owner_pid() == owner_pid) {
      delete replacement;
      return *reaper;
    }
  }

  // Process-lifetime ownership is intentional. Static destruction cannot safely
  // join a parent thread from a forked child, so each PID publishes its own
  // instance and lets process teardown reclaim it without running a destructor.
  return *replacement;
}

int reap_notification_writer(pid_t writer, int pidfd, int *status, bool defer_reap_for_testing) {
  const auto deadline = std::chrono::steady_clock::now() + kDebugNotificationWriteTimeout;
  bool exited = false;
  int failure = 0;
  while (!exited) {
    if (pidfd >= 0) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline)
        break;
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      pollfd ready{pidfd, POLLIN, 0};
      const int result = ::poll(&ready, 1, std::max(1, static_cast<int>(remaining.count())));
      if (result > 0) {
        exited = true;
      } else if (result < 0 && errno != EINTR) {
        failure = errno;
        break;
      }
    } else {
      const pid_t result = ::waitpid(writer, status, WNOHANG | __WCLONE);
      if (result == writer)
        return 0;
      if (result < 0 && errno != EINTR) {
        failure = errno;
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline)
        break;
      static_cast<void>(::poll(nullptr, 0, 1));
    }
  }

  if (!exited) {
    static_cast<void>(::kill(writer, SIGKILL));
    pid_t result = 0;
    if (!defer_reap_for_testing) {
      do {
        result = ::waitpid(writer, status, WNOHANG | __WCLONE);
      } while (result < 0 && errno == EINTR);
    }
    if (result == 0) {
      try {
        notification_writer_reaper().adopt(writer);
      } catch (...) {
        // The child remains owned by this process and will be reparented at
        // process exit. Allocation failure must not turn a bounded engine path
        // back into a blocking one.
      }
    }
    return failure != 0 ? -failure : -EAGAIN;
  }

  while (::waitpid(writer, status, __WCLONE) < 0) {
    if (errno != EINTR)
      return -errno;
  }
  return 0;
}

/// @brief Write one debugger wakeup without ever waiting indefinitely for capacity.
/// @details File-status flags belong to the shared open-file description, so a
/// competing holder can clear O_NONBLOCK after any defensive F_SETFL and make a
/// write block. Perform the write in a disposable child and kill it at a bounded
/// deadline instead. A raw clone with no exit signal keeps the internal child
/// out of the application's SIGCHLD and ordinary waitpid(-1) contract.
/// CLONE_UNTRACED also keeps it out of a debugger attached to the host process.
/// Prefer clone3's atomic pidfd result, but fall back to the legacy clone syscall
/// on older kernels and seccomp profiles; its CLONE_PIDFD form is atomic too, and
/// a final no-pidfd form is bounded by nonblocking waitpid polling. The child
/// executes only async-signal-safe operations; it never reaches inherited
/// simulator state or locks.
///
/// KFD's userspace test debugger uses an O_RDWR FIFO and consumes one byte per
/// wakeup. The synthetic primary KFD descriptor is an eventfd, whose ABI requires
/// an eight-byte counter write. Preserve both framing contracts.
/// @returns Zero after a complete write, otherwise the negative errno (EIO for a
/// short write, EAGAIN after the delivery deadline).
int write_debug_notification(int fd, const std::function<void()> &before_write = {},
                             std::optional<int> clone3_error_for_testing = std::nullopt,
                             std::optional<int> clone_pidfd_error_for_testing = std::nullopt,
                             bool defer_reap_for_testing = false) {
  struct stat descriptor_stat {};
  if (safe_fstat(fd, &descriptor_stat) != 0)
    return -errno;
  const size_t write_size = S_ISFIFO(descriptor_stat.st_mode) ? sizeof(uint8_t) : sizeof(uint64_t);

  if (before_write)
    before_write();

  // Block application handlers before the child exists, including the window
  // before write(). The parent restores its mask on every clone outcome.
  sigset_t blocked_signals;
  sigset_t previous_signals;
  sigfillset(&blocked_signals);
  const int mask_result = ::pthread_sigmask(SIG_SETMASK, &blocked_signals, &previous_signals);
  if (mask_result != 0)
    return -mask_result;

  int raw_pidfd = -1;
  clone_args clone{};
  clone.flags = CLONE_PIDFD | CLONE_UNTRACED;
  clone.pidfd = reinterpret_cast<uintptr_t>(&raw_pidfd);
  pid_t writer = -1;
  if (clone3_error_for_testing) {
    errno = *clone3_error_for_testing;
  } else {
    writer = static_cast<pid_t>(::syscall(SYS_clone3, &clone, sizeof(clone)));
  }
  if (writer < 0 && (errno == ENOSYS || errno == EPERM)) {
    raw_pidfd = -1;
    if (clone_pidfd_error_for_testing) {
      errno = *clone_pidfd_error_for_testing;
    } else {
      writer = static_cast<pid_t>(::syscall(SYS_clone, CLONE_PIDFD | CLONE_UNTRACED, nullptr,
                                            &raw_pidfd, nullptr, nullptr));
    }
    if (writer < 0 && (errno == EINVAL || errno == EPERM)) {
      raw_pidfd = -1;
      writer = static_cast<pid_t>(
          ::syscall(SYS_clone, CLONE_UNTRACED, nullptr, nullptr, nullptr, nullptr));
    }
  }
  if (writer != 0) {
    const int clone_error = errno;
    static_cast<void>(::pthread_sigmask(SIG_SETMASK, &previous_signals, nullptr));
    if (writer < 0)
      return -clone_error;
  }
  if (writer == 0) {
    const uint64_t one = 1;
    ssize_t written = 0;
    do {
      written = libc_passthrough().write(fd, &one, write_size);
    } while (written < 0 && errno == EINTR);
    const int result = written == static_cast<ssize_t>(write_size) ? 0 : written < 0 ? errno : EIO;
    static_cast<void>(::syscall(SYS_exit_group, result));
    __builtin_unreachable();
  }

  UniqueDriverFd pidfd(raw_pidfd);
  int status = 0;
  const int reap_result =
      reap_notification_writer(writer, pidfd.get(), &status, defer_reap_for_testing);
  if (reap_result != 0)
    return reap_result;
  if (!WIFEXITED(status))
    return -EIO;
  return WEXITSTATUS(status) == 0 ? 0 : -WEXITSTATUS(status);
}

uint64_t begin_notification_claim(KfdProcess::DebugSession &session, uint64_t exception_mask) {
  uint64_t claim_id = 0;
  do {
    claim_id = session.next_notification_claim_id++;
  } while (claim_id == 0 || session.notification_claims.contains(claim_id));
  session.notification_claims.emplace(
      claim_id, KfdProcess::DebugSession::NotificationClaim{exception_mask, true});
  return claim_id;
}

bool finish_notification_claim(KfdProcess::DebugSession &session, uint64_t claim_id,
                               uint64_t *consumed_mask = nullptr) {
  auto claim = session.notification_claims.find(claim_id);
  if (claim == session.notification_claims.end())
    return false;
  const bool continuously_subscribed = claim->second.continuously_subscribed;
  if (consumed_mask != nullptr)
    *consumed_mask = claim->second.consumed_mask;
  session.notification_claims.erase(claim);
  return continuously_subscribed;
}

void update_notification_claims(KfdProcess::DebugSession &session, uint64_t enabled_mask) {
  for (auto &[_, claim] : session.notification_claims)
    if ((claim.exception_mask & enabled_mask) == 0)
      claim.continuously_subscribed = false;
}

int pidfd_is_exited(int pidfd) {
  pollfd pfd{pidfd, POLLIN, 0};
  const int rc = ::poll(&pfd, 1, 0);
  if (rc < 0)
    return -errno;
  return rc == 1 && (pfd.revents & (POLLIN | POLLHUP)) ? 1 : 0;
}

/// @brief Report whether the process behind @p procfd has exited but not been
/// reaped.
///
/// @details A pidfd stays readable for a zombie, so pidfd_is_exited() cannot
/// tell a live debuggee from one that has already run to completion. The state
/// character in /proc/<pid>/stat can. It sits after the last ')' because the
/// comm field is parenthesised and may itself contain spaces and parentheses.
///
/// @retval 1 The target is a zombie (Z) or dead (X).
/// @retval 0 The target is still running.
/// @retval <0 Negative errno; the state could not be determined.
int procfd_is_zombie(int procfd) {
  // Passthrough, not ::openat/::read/::close: these run inside driver ioctls that
  // the interposer dispatched, so a bare ::close() here would re-enter the
  // interposer's own close() hook mid-dispatch. See PassthroughFdTraits.
  UniqueDriverFd stat_fd(libc_passthrough().openat(procfd, "stat", O_RDONLY | O_CLOEXEC, 0));
  if (stat_fd.get() < 0)
    return errno == ENOENT ? 1 : -errno;
  char buffer[4096];
  const ssize_t bytes = libc_passthrough().read(stat_fd.get(), buffer, sizeof(buffer) - 1);
  const int read_error = errno;
  stat_fd.reset();
  if (bytes < 0)
    return -read_error;
  buffer[bytes] = '\0';
  const char *name_end = std::strrchr(buffer, ')');
  if (name_end == nullptr || name_end[1] != ' ' || name_end[2] == '\0')
    return -EIO;
  return name_end[2] == 'Z' || name_end[2] == 'X';
}

int pin_process_identity(pid_t pid, UniqueDriverFd &pidfd, UniqueDriverFd &procfd) {
  const int raw_pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
  if (raw_pidfd < 0)
    return errno == ESRCH ? -ESRCH : -errno;
  pidfd = UniqueDriverFd(raw_pidfd);

  const std::string proc_path = "/proc/" + std::to_string(pid);
  const int raw_procfd =
      libc_passthrough().openat(AT_FDCWD, proc_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
  if (raw_procfd < 0) {
    const int open_error = errno;
    const int exited = pidfd_is_exited(pidfd.get());
    return exited == 1 ? -ESRCH : (exited < 0 ? exited : -open_error);
  }
  procfd = UniqueDriverFd(raw_procfd);

  const int exited = pidfd_is_exited(pidfd.get());
  return exited == 0 ? 0 : (exited == 1 ? -ESRCH : exited);
}

/// @brief Read TracerPid through a procfs directory pinned to the pidfd identity.
int tracer_pid_of(const UniqueDriverFd &pidfd, const UniqueDriverFd &procfd,
                  const SimulatedKfd::DebugIdentityValidationHook &validation_hook,
                  pid_t &tracer_pid) {
  int exited = pidfd_is_exited(pidfd.get());
  if (exited != 0)
    return exited == 1 ? -ESRCH : exited;

  UniqueDriverFd status_fd(
      libc_passthrough().openat(procfd.get(), "status", O_RDONLY | O_CLOEXEC, 0));
  if (status_fd.get() < 0) {
    const int open_error = errno;
    const int exited = pidfd_is_exited(pidfd.get());
    return exited == 1 ? -ESRCH : (exited < 0 ? exited : -open_error);
  }

  std::string status;
  char buffer[4096];
  for (;;) {
    const ssize_t count = libc_passthrough().read(status_fd.get(), buffer, sizeof(buffer));
    if (count > 0) {
      status.append(buffer, static_cast<size_t>(count));
      continue;
    }
    if (count == 0)
      break;
    if (errno == EINTR)
      continue;
    return -errno;
  }

  constexpr std::string_view kKey = "TracerPid:";
  tracer_pid = 0;
  size_t offset = 0;
  while (offset < status.size()) {
    const size_t end = status.find('\n', offset);
    const std::string_view line(status.data() + offset,
                                (end == std::string::npos ? status.size() : end) - offset);
    if (line.substr(0, kKey.size()) == kKey) {
      tracer_pid = static_cast<pid_t>(std::strtol(line.data() + kKey.size(), nullptr, 10));
      break;
    }
    if (end == std::string::npos)
      break;
    offset = end + 1;
  }

  if (validation_hook)
    validation_hook();

  exited = pidfd_is_exited(pidfd.get());
  return exited == 0 ? 0 : (exited == 1 ? -ESRCH : exited);
}

} // namespace

KfdProcess::KfdProcess(uint32_t process_id, uint32_t num_gpus)
    : process_id_(process_id),
      process_incarnation_(allocate_internal_identity(next_process_incarnation)),
      next_gpu_va_(0x1000000000ULL), gpu_state_(num_gpus) {}

KfdProcess::GpuBacking::GpuBacking(size_t size, uint32_t gpu_id, bool controlled_peer_for_testing)
    : size_(size), gpu_id_(gpu_id), controlled_peer_for_testing_(controlled_peer_for_testing),
      backing_id_(allocate_canonical_backing_id()) {}

KfdProcess::GpuBacking::~GpuBacking() {
  if (mapping_)
    safe_munmap(mapping_, size_);
}

std::shared_ptr<const KfdProcess::GpuBacking>
KfdProcess::GpuBacking::create_memfd_alias(int fd, size_t size, uint32_t gpu_id,
                                           bool controlled_peer_for_testing) {
  if (fd < 0 || size == 0 || size > static_cast<size_t>(std::numeric_limits<off_t>::max()))
    return {};
  struct stat statbuf {};
  if (safe_fstat(fd, &statbuf) != 0 || !S_ISREG(statbuf.st_mode) ||
      statbuf.st_size != static_cast<off_t>(size))
    return {};
  // Allocation creation seals against shrink. A missing seal means an external
  // descriptor can invalidate the alias's bounds after this check.
  const int seals = safe_fcntl(fd, F_GET_SEALS);
  if (seals < 0 || !(seals & F_SEAL_SHRINK))
    return {};

  // Allocate the owner before mmap so a failed C++ allocation cannot leak an
  // alias. The alias owns its mapping, while the IPC and allocation paths keep
  // their existing independent descriptors and client mappings.
  auto owner = std::make_shared<GpuBacking>(size, gpu_id, controlled_peer_for_testing);
  void *mapping = safe_mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED)
    return {};
  owner->mapping_ = mapping;
  return owner;
}

std::shared_ptr<KfdProcess> SimulatedKfd::find_process(uint32_t process_id) const {
  std::lock_guard<std::mutex> lk(process_mutex_);
  auto it = processes_.find(process_id);
  return (it != processes_.end()) ? it->second : nullptr;
}

std::shared_ptr<GemObject> SimulatedKfd::retain_bo_state(const struct stat &st,
                                                         std::shared_ptr<GemObject> state) {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return state;
  std::lock_guard<std::mutex> lock(proc->alloc_mutex_);
  std::vector<KfdProcess::GpuAllocation *> backed;
  for (auto &[handle, alloc] : proc->allocations_) {
    const int backing = alloc.memfd >= 0 ? alloc.memfd : alloc.dmabuf_fd;
    struct stat backing_st {};
    if (backing >= 0 && safe_fstat(backing, &backing_st) == 0 && backing_st.st_dev == st.st_dev &&
        backing_st.st_ino == st.st_ino) {
      backed.push_back(&alloc);
      if (alloc.bo_state)
        state = alloc.bo_state;
    }
  }
  // An IPC share handle can predate the PRIME operation that first tags the
  // BO. Retain the same record there so later imports inherit that metadata.
  std::lock_guard ipc_lock(ipc_mutex_);
  std::vector<IpcObject *> shared;
  for (auto &[key, object] : ipc_store_) {
    struct stat backing_st {};
    if (object.backing_memfd >= 0 && safe_fstat(object.backing_memfd, &backing_st) == 0 &&
        backing_st.st_dev == st.st_dev && backing_st.st_ino == st.st_ino) {
      shared.push_back(&object);
      if (object.bo_state)
        state = object.bo_state;
    }
  }
  for (auto *alloc : backed)
    alloc->bo_state = state;
  for (auto *object : shared)
    object->bo_state = state;
  return state;
}

std::shared_ptr<PrivateGttCopySourceGate>
SimulatedKfd::begin_private_gtt_copy_source_mutation(KfdProcess &process, uint64_t handle) {
  std::lock_guard lock(process.alloc_mutex_);
  const auto allocation = process.allocations_.find(handle);
  if (allocation == process.allocations_.end() || !allocation->second.private_gtt_copy_source_gate)
    return {};
  auto gate = allocation->second.private_gtt_copy_source_gate;
  gate->begin_mutation();
  return gate;
}

void SimulatedKfd::finish_private_gtt_copy_source_mutation(
    KfdProcess &process, uint64_t handle,
    const std::shared_ptr<PrivateGttCopySourceGate> &gate) noexcept {
  if (!gate)
    return;
  std::lock_guard lock(process.alloc_mutex_);
  const auto allocation = process.allocations_.find(handle);
  if (allocation == process.allocations_.end() ||
      allocation->second.private_gtt_copy_source_gate != gate) {
    gate->revoke();
    return;
  }
  const auto &source = allocation->second;
  uint64_t map_generation = 0;
  if (source.mapped_device_ids.size() == 1 && source.admission_generations.size() == 1) {
    const auto grant = source.admission_generations.find(*source.mapped_device_ids.begin());
    if (grant != source.admission_generations.end())
      map_generation = grant->second;
  }
  gate->finish_mutation(map_generation, source.host_view_generation);
}

SimulatedKfd::ScopedPrivateGttCopySourceMutation::ScopedPrivateGttCopySourceMutation(
    SimulatedKfd &driver, KfdProcess &process, uint64_t handle)
    : driver_(driver), process_(process), handle_(handle),
      gate_(driver.begin_private_gtt_copy_source_mutation(process, handle)) {
  if (gate_)
    gate_->wait_quiescent();
}

SimulatedKfd::ScopedPrivateGttCopySourceMutation::~ScopedPrivateGttCopySourceMutation() {
  driver_.finish_private_gtt_copy_source_mutation(process_, handle_, gate_);
}

bool SimulatedKfd::enable_cacheable_peer_allocation_for_testing(uint32_t process_id,
                                                                uint64_t handle) {
  const std::shared_ptr<KfdProcess> proc = find_process(process_id);
  if (!proc)
    return false;
  std::lock_guard operation_lock(proc->op_mutex_);
  std::lock_guard allocation_lock(proc->alloc_mutex_);
  if (proc->event_state_.is_closing())
    return false;
  const auto found = proc->allocations_.find(handle);
  if (found == proc->allocations_.end())
    return false;
  KfdProcess::GpuAllocation &alloc = found->second;
  if (alloc.domain != KfdProcess::GpuGrantDomain::Vram || alloc.imported ||
      alloc.driver_gpu_mapping || alloc.ipc_shared || alloc.backing_owner ||
      alloc.local_backing_owner || alloc.retirement_in_progress || alloc.dmabuf_exported ||
      !alloc.mapped_device_ids.empty() || !alloc.admission_generations.empty() || alloc.memfd < 0 ||
      !alloc.host_ptr || !alloc.host_ptr_owned || alloc.host_ptr_mapped_bytes != alloc.size ||
      alloc.page_table_mapped_bytes != alloc.size ||
      pte_mtype_for_flags(alloc.flags) != amdgpu::Mtype::RW)
    return false;

  // An ordinary raw effect admitted before FREE/VA reuse must finish before
  // this allocation can become controlled. Never wait while holding KFD locks.
  auto raw_write_close = proc->raw_write_gate_->try_close(alloc.gpu_va, alloc.size);
  if (!raw_write_close)
    return false;

  // ALLOC already installed the owner PTE in daemon mode. Remap it before
  // the first grant; subsequent export and MAP repairs preserve this policy.
  if (proc->set_page_mtype(alloc.gpu_va, alloc.size, pte_mtype_for_flags(alloc.flags)) !=
      KfdProcess::PageTableMutationResult::Applied)
    return false;

  // Publish the sink guard on every GPU before the controlled grant can
  // become visible. A prior owner MAP leaves local_backing_owner behind even
  // after UNMAP, so the checks above admit only a never-mapped allocation.
  // With no admission generation, no KFD cache victim can race this publish.
  // Each GPU has its own VM service for this shared process VA.
  const auto retire_guards = [&]() noexcept {
    for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal)
      if (gpus_[ordinal].soc)
        gpus_[ordinal].soc->gpu_vm().retire_controlled_cache_backing(
            proc->gpu(ordinal).address_space, alloc.gpu_va, alloc.size);
  };
  try {
    for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
      if (!gpus_[ordinal].soc)
        continue;
      const amdgpu::AddressSpaceHandle binding = proc->gpu(ordinal).address_space;
      if (!binding || !gpus_[ordinal].soc->gpu_vm().protect_controlled_cache_backing(
                          binding, alloc.gpu_va, alloc.size)) {
        retire_guards();
        return false;
      }
    }
  } catch (const std::bad_alloc &) {
    retire_guards();
    return false;
  }
  alloc.cacheable_peer_for_testing = true;
  try {
    proc->publish_gpu_grants_locked();
  } catch (const std::bad_alloc &) {
    alloc.cacheable_peer_for_testing = false;
    retire_guards();
    return false;
  }
  return true;
}

bool SimulatedKfd::has_exclusive_canonical_lines(
    std::span<const KfdProcess::GpuAliasRecord> aliases) const {
  if (aliases.empty())
    return false;
  const auto same = [](const KfdProcess::GpuAliasRecord &left,
                       const KfdProcess::GpuAliasRecord &right) {
    return left.process_id == right.process_id &&
           left.process_incarnation == right.process_incarnation && left.gpu_id == right.gpu_id &&
           left.gpu_va == right.gpu_va && left.backing_id == right.backing_id &&
           left.backing_offset == right.backing_offset && left.extent == right.extent &&
           left.admission_generation == right.admission_generation;
  };
  for (size_t index = 0; index < aliases.size(); ++index) {
    const auto &alias = aliases[index];
    if (alias.backing_id == 0 || alias.extent == 0 ||
        alias.backing_offset > UINT64_MAX - alias.extent)
      return false;
    for (size_t previous = 0; previous < index; ++previous)
      if (same(alias, aliases[previous]))
        return false;
  }

  // A retiring batch may contain several aliases of the same backing line.
  // Every live overlap must be one of those exact generations; the coordinator
  // separately refuses conflicting dirty writers within the selected batch.
  std::lock_guard lock(process_mutex_);
  for (const auto &alias : aliases) {
    const uint64_t first_line = alias.backing_offset / amdgpu::L2Cache::LINE_SIZE;
    const uint64_t last_line =
        (alias.backing_offset + alias.extent - 1) / amdgpu::L2Cache::LINE_SIZE;
    size_t exact_matches = 0;
    const auto inspect_process = [&](const std::shared_ptr<KfdProcess> &process) {
      for (const KfdProcess::GpuAliasRecord &other : *process->gpu_alias_snapshot()) {
        if (other.backing_id != alias.backing_id)
          continue;
        if (other.extent == 0 || other.backing_offset > UINT64_MAX - other.extent)
          return false;
        const uint64_t other_first = other.backing_offset / amdgpu::L2Cache::LINE_SIZE;
        const uint64_t other_last =
            (other.backing_offset + other.extent - 1) / amdgpu::L2Cache::LINE_SIZE;
        if (other_last < first_line || last_line < other_first)
          continue;
        if (std::ranges::none_of(aliases,
                                 [&](const auto &selected) { return same(other, selected); }))
          return false;
        if (same(other, alias))
          ++exact_matches;
      }
      return true;
    };
    for (const auto &[id, process] : processes_) {
      [[maybe_unused]] const auto &_ = id;
      if (!inspect_process(process))
        return false;
    }
    for (const auto &[id, pending] : pending_closes_) {
      [[maybe_unused]] const auto &_ = id;
      if (!inspect_process(pending->process))
        return false;
    }
    if (exact_matches != 1)
      return false;
  }
  return true;
}

std::vector<amdgpu::DeviceCacheDomain> SimulatedKfd::maintenance_domains_for_aliases(
    std::span<const KfdProcess::GpuAliasRecord> aliases) const {
  if (aliases.empty())
    throw std::runtime_error("alias maintenance needs a retired grant");
  // A local VRAM grant exists only on its owner GPU. A peer MAP needs a
  // canonical owner first, so local-only batches cannot have a cacheable
  // writer for the same BO in another GPU domain.
  const bool local_only = std::ranges::all_of(aliases, &KfdProcess::GpuAliasRecord::local);
  std::vector<amdgpu::DeviceCacheDomain> domains;
  domains.reserve(gpus_.size());
  for (const GpuDevice &device : gpus_) {
    if (local_only && std::ranges::none_of(aliases, [&](const auto &alias) {
          return alias.gpu_id == device.gpu_id;
        }))
      continue;
    if (!device.soc)
      throw std::runtime_error("alias maintenance has no selected GPU cache domain");
    domains.push_back({device.soc->cache_coherence(), device.gpu_id});
  }
  if (domains.empty() || (local_only && std::ranges::any_of(aliases, [&](const auto &alias) {
                            return std::ranges::none_of(domains, [&](const auto &domain) {
                              return domain.gpu_id == alias.gpu_id;
                            });
                          })))
    throw std::runtime_error("alias maintenance cannot find an owner GPU cache domain");
  return domains;
}

std::optional<amdgpu::DeviceCacheAliasScope>
SimulatedKfd::scoped_canonical_alias(const KfdProcess &proc,
                                     const KfdProcess::GpuAliasRecord &alias,
                                     bool allow_cacheable_aliases) const {
  if (alias.gpu_va % amdgpu::L2Cache::LINE_SIZE != 0 ||
      alias.extent % amdgpu::L2Cache::LINE_SIZE != 0 || alias.extent == 0 ||
      alias.backing_id == 0 || alias.backing_offset % amdgpu::L2Cache::LINE_SIZE != 0 ||
      alias.backing_offset > UINT64_MAX - alias.extent ||
      alias.gpu_va > UINT64_MAX - alias.extent || !proc.gpu_alias_grant_current(alias))
    return std::nullopt;
  const GpuDevice *device = find_gpu(alias.gpu_id);
  if (!device || !device->soc)
    return std::nullopt;
  const auto grant = proc.gpu_peer_backing_grant(alias.gpu_id, alias.gpu_va, alias.extent);
  if (!grant.admission.allowed || (grant.admission.cacheable && !allow_cacheable_aliases) ||
      !grant.backing || grant.backing.backing_id() != alias.backing_id ||
      grant.backing.file_offset() != alias.backing_offset)
    return std::nullopt;
  const auto live =
      device->soc->gpu_vm().snapshot(proc.gpu(gpu_ordinal(alias.gpu_id)).address_space);
  if (!live || live->revoked() || live->info().vmid == 0 || live->cache_identity() == 0)
    return std::nullopt;
  return amdgpu::DeviceCacheAliasScope{.domain = {device->soc->cache_coherence(), alias.gpu_id},
                                       .vmid = live->info().vmid,
                                       .address = alias.gpu_va,
                                       .size = alias.extent,
                                       .admission_generation = alias.admission_generation,
                                       .cache_identity = live->cache_identity(),
                                       .backing_id = alias.backing_id,
                                       .backing_offset = alias.backing_offset};
}

std::optional<SimulatedKfd::LivePeerAliasSelection> SimulatedKfd::try_select_live_peer_aliases(
    const KfdProcess::GpuAliasRecord &requester, const FabricBacking &backing,
    amdgpu::MemoryIngressPermit driver_pin, bool allow_cacheable_aliases, bool &busy) {
  busy = false;
  constexpr uint64_t kLineBytes = amdgpu::L2Cache::LINE_SIZE;
  static_assert((kLineBytes & (kLineBytes - 1)) == 0);
  if (!driver_pin || !backing || backing.backing_id() == 0 || backing.extent() == 0 ||
      backing.extent() - 1 > UINT64_MAX - backing.offset())
    return std::nullopt;
  const uint64_t first_line = backing.offset() & ~(kLineBytes - 1);
  const uint64_t last_byte = backing.offset() + backing.extent() - 1;
  const uint64_t last_line = last_byte | (kLineBytes - 1);
  if (last_line == UINT64_MAX)
    return std::nullopt;
  const uint64_t end = last_line + 1;

  // MAP/import publication takes the shared side; the owner must take the
  // exclusive side before scanning and hold it until cache acknowledgment.
  std::unique_lock publication(alias_publication_mutex_, std::try_to_lock);
  if (!publication.owns_lock()) {
    busy = true;
    return std::nullopt;
  }

  std::vector<std::shared_ptr<KfdProcess>> processes;
  {
    std::lock_guard lock(process_mutex_);
    processes.reserve(processes_.size() + pending_closes_.size());
    for (const auto &[id, process] : processes_) {
      (void)id;
      processes.push_back(process);
    }
    for (const auto &[id, pending] : pending_closes_) {
      (void)id;
      processes.push_back(pending->process);
    }
  }

  struct SelectedAlias {
    std::shared_ptr<KfdProcess> process;
    KfdProcess::GpuAliasRecord alias;
    amdgpu::DeviceCacheAliasScope scope;
    uint64_t backing_offset = 0;
    bool requester = false;
  };
  std::vector<SelectedAlias> selected;
  std::vector<GpuAliasIngressGate::Key> keys;
  size_t requester_matches = 0;
  for (const std::shared_ptr<KfdProcess> &process : processes) {
    const KfdProcess::GpuAliasSnapshots snapshots = process->gpu_alias_snapshots();
    const auto inspect = [&](const KfdProcess::GpuAliasSnapshot &aliases) {
      for (const KfdProcess::GpuAliasRecord &alias : aliases) {
        if (alias.backing_id != backing.backing_id())
          continue;
        if (alias.extent == 0 || alias.backing_offset > UINT64_MAX - alias.extent)
          return false;
        const uint64_t alias_end = alias.backing_offset + alias.extent;
        if (alias_end <= first_line || end <= alias.backing_offset)
          continue;
        const uint64_t begin = std::max(first_line, alias.backing_offset);
        const uint64_t finish = std::min(end, alias_end);
        if ((begin & (kLineBytes - 1)) != 0 || (finish & (kLineBytes - 1)) != 0 ||
            begin < alias.backing_offset ||
            begin - alias.backing_offset > UINT64_MAX - alias.gpu_va)
          return false;
        const std::optional<amdgpu::DeviceCacheAliasScope> full_scope =
            scoped_canonical_alias(*process, alias, allow_cacheable_aliases);
        if (!full_scope || finish - begin > std::numeric_limits<size_t>::max())
          return false;
        amdgpu::DeviceCacheAliasScope scope = *full_scope;
        scope.address = alias.gpu_va + (begin - alias.backing_offset);
        scope.size = finish - begin;
        scope.backing_offset = begin;
        const bool is_requester =
            alias.process_id == requester.process_id &&
            alias.process_incarnation == requester.process_incarnation &&
            alias.gpu_id == requester.gpu_id && alias.gpu_va == requester.gpu_va &&
            alias.backing_offset == requester.backing_offset && alias.extent == requester.extent &&
            alias.admission_generation == requester.admission_generation &&
            alias.local == requester.local;
        selected.push_back({process, alias, std::move(scope), begin, is_requester});
        keys.push_back(GpuAliasIngressGate::key(alias));
        if (is_requester)
          ++requester_matches;
      }
      return true;
    };
    if (!inspect(*snapshots.canonical) || !inspect(*snapshots.local))
      return std::nullopt;
  }
  if (requester_matches != 1 || selected.empty())
    return std::nullopt;

  std::optional<GpuAliasIngressGate::EffectWindow> window =
      alias_ingress_gate_->try_freeze_for_effect(keys);
  if (!window) {
    busy = true;
    return std::nullopt;
  }

  // The publication lock excludes grant mutation; the freeze excludes cache
  // ingress. Recheck VM/PTE identity after both are held, before any writeback.
  amdgpu::VmCacheNamespace requester_cache_namespace;
  for (const SelectedAlias &entry : selected) {
    const GpuDevice *device = find_gpu(entry.alias.gpu_id);
    if (!device || !device->soc || !device->legacy_vm ||
        !entry.process->gpu_alias_grant_current(entry.alias))
      return std::nullopt;
    const auto live = device->soc->gpu_vm().snapshot(
        entry.process->gpu(gpu_ordinal(entry.alias.gpu_id)).address_space);
    if (!live || live->revoked() || live->info().vmid != entry.scope.vmid ||
        live->info().vmid != entry.alias.process_id ||
        live->cache_identity() != entry.scope.cache_identity ||
        live->classify_data_route(entry.scope.address, entry.scope.size,
                                  amdgpu::VmAccessKind::Write,
                                  false) == amdgpu::VmDataRoute::Faulted)
      return std::nullopt;
    if (entry.requester)
      requester_cache_namespace = live->cache_namespace();
    const std::shared_ptr<amdgpu::LegacyAddressSpace> address_space =
        device->legacy_vm->address_space_snapshot(entry.scope.vmid);
    if (!address_space ||
        !address_space->is_range_mapped(entry.scope.address, entry.scope.size, entry.scope.vmid) ||
        !address_space->has_writable_host_backing(entry.scope.address, entry.scope.vmid,
                                                  entry.scope.size))
      return std::nullopt;
    const KfdProcess::GpuPeerBackingDecision current = entry.process->gpu_peer_backing_grant(
        entry.alias.gpu_id, entry.scope.address, entry.scope.size);
    if (!current.admission.allowed || (current.admission.cacheable && !allow_cacheable_aliases) ||
        !current.backing || current.backing.backing_id() != entry.alias.backing_id ||
        current.backing.file_offset() != entry.backing_offset ||
        current.backing.size() != entry.scope.size)
      return std::nullopt;
  }

  LivePeerAliasSelection result;
  result.scopes.reserve(selected.size());
  for (const SelectedAlias &entry : selected)
    result.scopes.push_back(entry.scope);
  result.requester_cache_namespace = requester_cache_namespace;
  result.lease = std::make_shared<LivePeerEffectLease>(std::move(driver_pin),
                                                       std::move(publication), std::move(*window));
  return result;
}

std::optional<std::shared_lock<std::shared_mutex>>
SimulatedKfd::acquire_alias_publication_reader(const KfdProcess &proc) const {
  std::shared_lock lock(alias_publication_mutex_, std::defer_lock);
  while (!lock.try_lock()) {
    if (proc.canonical_retirements_pending_.load(std::memory_order_acquire) != 0)
      return std::nullopt;
    const uint64_t epoch = alias_publication_release_epoch_.load(std::memory_order_acquire);
    // Retry after observing the epoch: a writer may have released the gate
    // between the failed acquisition and the snapshot.
    if (lock.try_lock())
      break;
    if (proc.canonical_retirements_pending_.load(std::memory_order_acquire) != 0)
      return std::nullopt;
    if (alias_publication_writers_pending_.load(std::memory_order_acquire) == 0) {
      // try_lock_shared may fail spuriously; no writer would wake an epoch wait.
      std::this_thread::yield();
      continue;
    }
    alias_publication_release_epoch_.wait(epoch, std::memory_order_acquire);
  }
  return lock;
}

SimulatedKfd::GpuAliasJoin
SimulatedKfd::join_gpu_aliases(uint32_t process_id, uint32_t gpu_id,
                               const amdgpu::GpuVmAccess &expected,
                               std::span<const amdgpu::MemoryIngressSpan> spans) const {
  return join_gpu_aliases_impl(process_id, gpu_id, expected, spans, false, false);
}

SimulatedKfd::GpuAliasJoin SimulatedKfd::join_gpu_aliases_for_peer_read(
    uint32_t process_id, uint32_t gpu_id, const amdgpu::GpuVmAccess &expected,
    std::span<const amdgpu::MemoryIngressSpan> spans) const {
  return join_gpu_aliases_impl(process_id, gpu_id, expected, spans, false, true);
}

SimulatedKfd::GpuAliasJoin SimulatedKfd::join_gpu_aliases_for_writeback(
    uint32_t process_id, uint32_t gpu_id, const amdgpu::GpuVmAccess &expected,
    std::span<const amdgpu::MemoryIngressSpan> spans,
    const GpuAliasIngressGate::DrainAuthority *authority) const {
  return join_gpu_aliases_impl(process_id, gpu_id, expected, spans, true, false, authority);
}

SimulatedKfd::GpuAliasJoin SimulatedKfd::join_gpu_aliases_impl(
    uint32_t process_id, uint32_t gpu_id, const amdgpu::GpuVmAccess &expected,
    std::span<const amdgpu::MemoryIngressSpan> spans, bool prepared_writeback, bool peer_cache_read,
    const GpuAliasIngressGate::DrainAuthority *authority) const {
  GpuAliasJoin result;
  SoC *soc = nullptr;
  amdgpu::LegacyGpuVmAdapter *legacy_vm = nullptr;
  {
    // Registration writes these handles under process_mutex_; close erases the
    // process from the table under the same lock before clearing its handles.
    std::lock_guard lock(process_mutex_);
    const auto process = processes_.find(process_id);
    const auto pending = authority ? pending_closes_.find(process_id) : pending_closes_.end();
    if (process == processes_.end() && pending == pending_closes_.end()) {
      result.outcome = amdgpu::VmAccessOutcome::Revoked;
      return result;
    }
    uint32_t ordinal = 0;
    while (ordinal < gpus_.size() && gpus_[ordinal].gpu_id != gpu_id)
      ++ordinal;
    if (ordinal == gpus_.size() || !gpus_[ordinal].soc ||
        ((prepared_writeback || peer_cache_read) && !gpus_[ordinal].legacy_vm)) {
      result.outcome = amdgpu::VmAccessOutcome::Faulted;
      return result;
    }
    result.process = process != processes_.end() ? process->second : pending->second->process;
    result.address_space = result.process->gpu(ordinal).address_space;
    const KfdProcess::GpuAliasSnapshots aliases = result.process->gpu_alias_snapshots();
    result.publication = aliases.canonical;
    result.local_publication = aliases.local;
    soc = gpus_[ordinal].soc;
    legacy_vm = gpus_[ordinal].legacy_vm.get();
  }

  if (!result.address_space || expected.revoked()) {
    result.outcome = amdgpu::VmAccessOutcome::Revoked;
    return result;
  }
  const auto live = soc->gpu_vm().snapshot(result.address_space);
  if (!live || live->revoked() || live->cache_namespace() != expected.cache_namespace()) {
    result.outcome = amdgpu::VmAccessOutcome::Revoked;
    return result;
  }
  if (prepared_writeback && (live->cache_identity() != expected.cache_identity() ||
                             live->retain_effect_gate() != expected.retain_effect_gate())) {
    result.outcome = amdgpu::VmAccessOutcome::Revoked;
    return result;
  }
  std::shared_ptr<amdgpu::LegacyAddressSpace> address_space =
      prepared_writeback || peer_cache_read ? legacy_vm->address_space_snapshot(live->info().vmid)
                                            : nullptr;
  if ((prepared_writeback || peer_cache_read) && !address_space) {
    result.outcome = amdgpu::VmAccessOutcome::Revoked;
    return result;
  }

  for (const auto &span : spans) {
    if (span.size == 0)
      continue;
    if (span.size > UINT64_MAX - span.address) {
      result.outcome = amdgpu::VmAccessOutcome::Malformed;
      return result;
    }
    const uint64_t span_end = span.address + span.size;
    const auto join_alias = [&](const KfdProcess::GpuAliasRecord &alias) {
      if (alias.gpu_id != gpu_id)
        return true;
      const uint64_t alias_end = alias.gpu_va + alias.extent;
      if (span.address >= alias_end || alias.gpu_va >= span_end)
        return true;
      const uint64_t begin = std::max(span.address, alias.gpu_va);
      const uint64_t end = std::min(span_end, alias_end);
      if (!result.process->gpu_alias_grant_current(alias)) {
        result.outcome = amdgpu::VmAccessOutcome::Unavailable;
        return false;
      }
      if (alias.local && !prepared_writeback) {
        // SDMA and queued dispatches can use an independently pinned access
        // state for this same binding. Its cache identity and effect gate are
        // intentionally different from a fresh ordinary snapshot.
        if (!legacy_vm || expected.info().vmid != live->info().vmid) {
          result.outcome = amdgpu::VmAccessOutcome::Revoked;
          return false;
        }
        if (!address_space)
          address_space = legacy_vm->address_space_snapshot(live->info().vmid);
        if (!address_space) {
          result.outcome = amdgpu::VmAccessOutcome::Revoked;
          return false;
        }
      }
      if (prepared_writeback || peer_cache_read) {
        // Generic local-only admission rejects a deferred peer route. Require
        // full PTE residency and a readable or writable host extent before
        // admitting selected cache ingress or a prepared writeback.
        const amdgpu::VmDataRoute route =
            live->classify_data_route(begin, end - begin, span.access, false);
        if (!address_space->is_range_mapped(begin, end - begin, live->info().vmid) ||
            !(peer_cache_read
                  ? address_space->has_readable_host_backing(begin, live->info().vmid, end - begin)
                  : address_space->has_writable_host_backing(begin, live->info().vmid,
                                                             end - begin)) ||
            (peer_cache_read ? route != amdgpu::VmDataRoute::Deferred
                             : route == amdgpu::VmDataRoute::Faulted)) {
          result.outcome = amdgpu::VmAccessOutcome::Unavailable;
          return false;
        }
      } else {
        // CU and SDMA admission retains its local-only generic VM check.
        const auto mapped = live->query_access(begin, end - begin, span.access);
        if (mapped != amdgpu::VmAccessOutcome::Complete ||
            (alias.local &&
             !address_space->is_range_mapped(begin, end - begin, live->info().vmid))) {
          result.outcome = mapped == amdgpu::VmAccessOutcome::Revoked
                               ? mapped
                               : amdgpu::VmAccessOutcome::Unavailable;
          return false;
        }
      }
      result.aliases.push_back({.alias = alias,
                                .cache_namespace = live->cache_namespace(),
                                .cache_identity = live->cache_identity(),
                                .vmid = live->info().vmid,
                                .span = {.address = begin,
                                         .size = static_cast<size_t>(end - begin),
                                         .access = span.access}});
      return true;
    };
    for (const auto &alias : *result.publication)
      if (!join_alias(alias))
        return result;
    for (const auto &alias : *result.local_publication)
      if (!join_alias(alias))
        return result;
  }
  return result;
}

amdgpu::MemoryIngressAdmission SimulatedKfd::admit_gpu_alias_ingress(
    uint32_t process_id, uint32_t gpu_id, const amdgpu::GpuVmAccess &expected,
    std::span<const amdgpu::MemoryIngressSpan> spans, bool peer_cache_read) {
  GpuAliasJoin joined = peer_cache_read
                            ? join_gpu_aliases_for_peer_read(process_id, gpu_id, expected, spans)
                            : join_gpu_aliases(process_id, gpu_id, expected, spans);
  if (joined.outcome != amdgpu::VmAccessOutcome::Complete)
    return {.outcome = joined.outcome, .permit = {}};
  if (peer_cache_read &&
      (joined.aliases.size() != 1 || joined.aliases.front().alias.local || spans.size() != 1 ||
       joined.aliases.front().span.address != spans.front().address ||
       joined.aliases.front().span.size != spans.front().size))
    return {.outcome = amdgpu::VmAccessOutcome::Faulted, .permit = {}};

  std::vector<GpuAliasIngressGate::Key> keys;
  keys.reserve(joined.aliases.size());
  for (const auto &binding : joined.aliases)
    keys.push_back(GpuAliasIngressGate::key(binding.alias));
  auto permit = alias_ingress_gate_->try_acquire(keys);
  if (!permit)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};

  // Count ordinary writers against alias selection before checking a held
  // canonical-line reservation. The selected writer cannot enter while this
  // alias permit is live, even if it reserves the line just after our check.
  for (const GpuAliasBinding &binding : joined.aliases) {
    if (binding.alias.local || (binding.span.access != amdgpu::VmAccessKind::Write &&
                                binding.span.access != amdgpu::VmAccessKind::Atomic))
      continue;
    const uint64_t backing_offset =
        binding.alias.backing_offset + (binding.span.address - binding.alias.gpu_va);
    if (peer_line_gate_->overlaps(binding.alias.backing_id, backing_offset, binding.span.size))
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
    // The controlled policy still refuses owner-local cacheable CU/SDMA
    // writes until those writers join selected canonical maintenance.
    const KfdProcess::GpuGrantDecision grant =
        joined.process->gpu_access_grant(gpu_id, binding.span.address, binding.span.size);
    if (grant.allowed && grant.cacheable && grant.domain == KfdProcess::GpuGrantDomain::Vram &&
        grant.owner_gpu_id == gpu_id)
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .permit = {}};
  }

  const bool has_local_alias = std::ranges::any_of(
      joined.aliases, [](const GpuAliasBinding &binding) { return binding.alias.local; });

  SoC *soc = nullptr;
  amdgpu::LegacyGpuVmAdapter *legacy_vm = nullptr;
  {
    std::lock_guard lock(process_mutex_);
    const auto current = processes_.find(process_id);
    if (current == processes_.end() || current->second != joined.process) {
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
    }
    for (const auto &gpu : gpus_)
      if (gpu.gpu_id == gpu_id) {
        soc = gpu.soc;
        legacy_vm = gpu.legacy_vm.get();
      }
    if (!soc || ((has_local_alias || peer_cache_read) && !legacy_vm) ||
        joined.address_space != joined.process->gpu(gpu_ordinal(gpu_id)).address_space)
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  }
  const KfdProcess::GpuAliasSnapshots current = joined.process->gpu_alias_snapshots();
  if (current.canonical != joined.publication || current.local != joined.local_publication)
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
  const auto live = soc->gpu_vm().snapshot(joined.address_space);
  if (!live || live->revoked() || live->cache_namespace() != expected.cache_namespace() ||
      (has_local_alias && live->info().vmid != expected.info().vmid) ||
      (!joined.aliases.empty() && live->cache_identity() != joined.aliases.front().cache_identity))
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  const std::shared_ptr<amdgpu::LegacyAddressSpace> address_space =
      has_local_alias || peer_cache_read ? legacy_vm->address_space_snapshot(live->info().vmid)
                                         : nullptr;
  if ((has_local_alias || peer_cache_read) && !address_space)
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  for (const auto &binding : joined.aliases) {
    if (binding.vmid != live->info().vmid ||
        !joined.process->gpu_alias_grant_current(binding.alias) ||
        (binding.alias.local && !address_space->is_range_mapped(
                                    binding.span.address, binding.span.size, live->info().vmid)))
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
    if (peer_cache_read) {
      if (!address_space->is_range_mapped(binding.span.address, binding.span.size,
                                          live->info().vmid) ||
          !address_space->has_readable_host_backing(binding.span.address, live->info().vmid,
                                                    binding.span.size) ||
          live->classify_data_route(binding.span.address, binding.span.size,
                                    amdgpu::VmAccessKind::Read,
                                    false) != amdgpu::VmDataRoute::Deferred)
        return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
    } else if (live->query_access(binding.span.address, binding.span.size, binding.span.access) !=
               amdgpu::VmAccessOutcome::Complete) {
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
    }
  }
  return {.outcome = amdgpu::VmAccessOutcome::Complete, .permit = std::move(*permit)};
}

amdgpu::VmPrivateUcGttCopySourceAdmission
SimulatedKfd::admit_private_uc_gtt_copy_source(uint32_t process_id, uint32_t gpu_id,
                                               const amdgpu::GpuVmAccess &expected,
                                               uint64_t address, size_t size) {
  if (size == 0 || size - 1 > UINT64_MAX - address)
    return {};
  if (expected.revoked())
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};

  std::shared_ptr<KfdProcess> process;
  SoC *soc = nullptr;
  amdgpu::AddressSpaceHandle binding;
  {
    std::lock_guard lock(process_mutex_);
    const auto found = processes_.find(process_id);
    if (found == processes_.end())
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
    process = found->second;
    for (size_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
      if (gpus_[ordinal].gpu_id == gpu_id) {
        soc = gpus_[ordinal].soc;
        binding = process->gpu(static_cast<uint32_t>(ordinal)).address_space;
        break;
      }
    }
  }
  if (!soc || !binding)
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  const auto live = soc->gpu_vm().snapshot(binding);
  if (!live || live->revoked() || live->cache_namespace() != expected.cache_namespace() ||
      !live->same_root_lineage(expected) || live->info().vmid != expected.info().vmid)
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};

  std::byte *host = nullptr;
  std::shared_ptr<PrivateGttCopySourceGate> gate;
  amdgpu::VmPrivateUcGttCopySourceAdmission admitted;
  {
    std::lock_guard lock(process->alloc_mutex_);
    if (process->event_state_.is_closing())
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
    const KfdProcess::GpuAllocation *source = nullptr;
    for (const auto &entry : process->allocations_) {
      const auto &allocation = entry.second;
      if (address < allocation.gpu_va || address - allocation.gpu_va >= allocation.size)
        continue;
      if (source || size > allocation.size - (address - allocation.gpu_va))
        return {};
      source = &allocation;
    }
    if (!source)
      return {};
    // A retired or changing source must not silently enter the ordinary COPY
    // maintenance path while its old memfd view is being withdrawn.
    if (source->private_gtt_copy_source_gate) {
      if (const auto closed = source->private_gtt_copy_source_gate->closed_outcome())
        return {.outcome = *closed, .permit = {}};
    }
    if (source->private_gtt_copy_source_gate &&
        (source->retirement_in_progress || source->grants_retired ||
         source->private_gtt_copy_source_unstable))
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable, .permit = {}};
    constexpr uint32_t kMemoryKinds = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                      KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    const auto map = source->admission_generations.find(gpu_id);
    if ((source->flags & kMemoryKinds) != KFD_IOC_ALLOC_MEM_FLAGS_GTT ||
        pte_mtype_for_flags(source->flags) != amdgpu::Mtype::UC ||
        source->domain != KfdProcess::GpuGrantDomain::System || source->gpu_id != gpu_id ||
        source->user_va || source->imported || source->driver_gpu_mapping ||
        source->retirement_in_progress || source->ipc_shared || source->dmabuf_exported ||
        source->dmabuf_fd >= 0 || source->backing_owner || source->local_backing_owner ||
        source->memfd < 0 || !source->host_ptr || !source->host_ptr_owned ||
        !(source->host_view_prot & PROT_READ) ||
        source->size > std::numeric_limits<size_t>::max() ||
        source->host_ptr_mapped_bytes != source->size ||
        source->page_table_mapped_bytes != source->size || source->mapped_device_ids.size() != 1 ||
        !source->mapped_device_ids.contains(gpu_id) || source->admission_generations.size() != 1 ||
        map == source->admission_generations.end() || map->second == 0 ||
        source->host_view_generation == 0 || !source->private_gtt_copy_source_gate)
      return {};
    gate = source->private_gtt_copy_source_gate;
    admitted = gate->admit(map->second, source->host_view_generation);
    if (admitted.outcome != amdgpu::VmAccessOutcome::Complete || !admitted.permit)
      return admitted;
    host = static_cast<std::byte *>(source->host_ptr) + (address - source->gpu_va);
  }

  // The permit keeps the exact memfd view and grant in place while both VM
  // snapshots prove every touched PTE is UC and resolves to that same view.
  if (expected.query_access(address, size, amdgpu::VmAccessKind::Read) !=
          amdgpu::VmAccessOutcome::Complete ||
      live->query_access(address, size, amdgpu::VmAccessKind::Read) !=
          amdgpu::VmAccessOutcome::Complete ||
      expected.classify_data_route(address, size, amdgpu::VmAccessKind::Read, false) !=
          amdgpu::VmDataRoute::Local ||
      live->classify_data_route(address, size, amdgpu::VmAccessKind::Read, false) !=
          amdgpu::VmDataRoute::Local)
    return {};
  size_t checked = 0;
  while (checked < size) {
    const uint64_t page_address = address + checked;
    const size_t chunk =
        std::min(size - checked, size_t{4096} - static_cast<size_t>(page_address & 0xFFF));
    if (expected.query_mtype(page_address) != amdgpu::Mtype::UC ||
        live->query_mtype(page_address) != amdgpu::Mtype::UC ||
        expected.resolve_host_pointer(page_address, chunk) != host + checked ||
        live->resolve_host_pointer(page_address, chunk) != host + checked)
      return {};
    checked += chunk;
  }
  {
    std::lock_guard lock(process_mutex_);
    const auto found = processes_.find(process_id);
    if (found == processes_.end() || found->second != process ||
        process->gpu(gpu_ordinal(gpu_id)).address_space != binding)
      return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  }
  const auto current = soc->gpu_vm().snapshot(binding);
  if (!current || current->revoked() || current->cache_namespace() != expected.cache_namespace() ||
      !current->same_root_lineage(expected))
    return {.outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
  return admitted;
}

bool SimulatedKfd::try_gpu_peer_cache_read(uint32_t process_id, uint32_t gpu_id,
                                           const amdgpu::GpuVmAccess &expected, uint64_t address,
                                           std::span<std::byte> bytes,
                                           const CanonicalLineFreshness::ReadStamp *stamp,
                                           amdgpu::ComputeUnitCore &cu, SoC &soc, bool scalar) {
  constexpr uint64_t kLineBytes = amdgpu::L2Cache::LINE_SIZE;
  if (bytes.empty() || bytes.size() > kLineBytes - (address & (kLineBytes - 1)) ||
      expected.revoked() || expected.info().vmid != process_id || !cu.l2())
    return false;
  const std::shared_ptr<KfdProcess> process = find_process(process_id);
  const GpuDevice *device = find_gpu(gpu_id);
  if (!process || !device || device->soc != &soc || !device->legacy_vm)
    return false;

  const KfdProcess::GpuPeerBackingDecision granted =
      process->gpu_peer_backing_grant(gpu_id, address, bytes.size());
  if (!granted.admission.allowed || !granted.admission.cacheable ||
      !granted.admission.first_peer_candidate || granted.admission.mixed_peer_owners ||
      granted.admission.domain != KfdProcess::GpuGrantDomain::Vram ||
      granted.admission.generation == 0 || granted.admission.first_extent_bytes < bytes.size() ||
      !granted.backing || granted.backing.backing_id() == 0 ||
      granted.backing.owner_gpu_id() != granted.admission.owner_gpu_id ||
      (stamp &&
       !stamp->covers(granted.backing.backing_id(), granted.backing.file_offset(), bytes.size())))
    return false;

  const amdgpu::MemoryIngressSpan span{
      .address = address, .size = bytes.size(), .access = amdgpu::VmAccessKind::Read};
  amdgpu::MemoryIngressAdmission ingress =
      admit_gpu_alias_ingress(process_id, gpu_id, expected, std::span(&span, 1), true);
  if (ingress.outcome != amdgpu::VmAccessOutcome::Complete || !ingress.permit)
    return false;

  const KfdProcess::GpuPeerBackingDecision current =
      process->gpu_peer_backing_grant(gpu_id, address, bytes.size());
  if (!current.admission.allowed || !current.admission.cacheable ||
      current.admission.generation != granted.admission.generation || !current.backing ||
      !current.backing.same_backing(granted.backing) ||
      current.backing.file_offset() != granted.backing.file_offset())
    return false;

  const std::shared_ptr<PeerEndpointAccess> endpoint = peer_endpoint_access_;
  {
    std::lock_guard route_lock(endpoint->mutex);
    if (!endpoint->endpoint ||
        !endpoint->endpoint->can_route(gpu_id, granted.admission.owner_gpu_id))
      return false;
  }

  const amdgpu::AddressSpaceHandle handle = process->gpu(gpu_ordinal(gpu_id)).address_space;
  const std::optional<amdgpu::GpuVmAccess> live = soc.gpu_vm().snapshot(handle);
  if (!live || live->revoked() || live->cache_namespace() != expected.cache_namespace() ||
      live->cache_identity() == 0 ||
      live->classify_data_route(address, bytes.size(), amdgpu::VmAccessKind::Read, false) !=
          amdgpu::VmDataRoute::Deferred ||
      live->query_mtype(address) != amdgpu::Mtype::RW)
    return false;

  const std::shared_ptr<util::DistributedSharedMutex> page_requests =
      process->page_table_request_mutex();
  std::shared_lock page_lease(*page_requests, std::try_to_lock);
  if (!page_lease.owns_lock())
    return false;
  const std::shared_ptr<amdgpu::LegacyAddressSpace> address_space =
      device->legacy_vm->address_space_snapshot(process_id);
  if (!address_space || !address_space->is_range_mapped(address, bytes.size(), process_id) ||
      !address_space->has_readable_host_backing(address, process_id, bytes.size()))
    return false;

  amdgpu::GpuVm::CurrentNamespaceLease namespace_lease =
      soc.gpu_vm().try_acquire_current_namespace(handle, expected.cache_namespace());
  if (!namespace_lease)
    return false;
  const uint64_t generation = current.admission.generation;
  const std::shared_ptr<const KfdProcess::GpuAliasSnapshot> aliases = process->gpu_alias_snapshot();
  if (std::ranges::none_of(*aliases, [&](const KfdProcess::GpuAliasRecord &alias) {
        return alias.process_id == process_id &&
               alias.process_incarnation == process->process_incarnation() &&
               alias.gpu_id == gpu_id && !alias.local &&
               alias.backing_id == granted.backing.backing_id() &&
               alias.admission_generation == generation && address >= alias.gpu_va &&
               address - alias.gpu_va < alias.extent &&
               bytes.size() <= alias.extent - (address - alias.gpu_va) &&
               alias.backing_offset <= UINT64_MAX - (address - alias.gpu_va) &&
               alias.backing_offset + (address - alias.gpu_va) == granted.backing.file_offset() &&
               process->gpu_alias_grant_current(alias);
      }))
    return false;

  std::span<uint8_t> data(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size());
  const uint64_t cache_identity = live->cache_identity();
  if (stamp) {
    CanonicalLineFreshness::FillAuthority fill = stamp->try_acquire_fill(
        granted.backing.backing_id(), granted.backing.file_offset(), bytes.size());
    if (!fill)
      return false;
    for (uint32_t iod = 0; iod < soc.num_iods(); ++iod)
      if (!soc.iod(iod)->msc()->try_fill_peer_cached(address, data, process_id, generation,
                                                     cache_identity))
        return false;
    if (!cu.l2()->try_fill_peer_cached(address, data, process_id, generation, cache_identity))
      return false;
    return scalar ? cu.l1_scalar().try_fill_peer_cached(address, data, process_id, generation,
                                                        cache_identity)
                  : cu.l1_vector().try_fill_peer_cached(address, data, process_id, generation,
                                                        cache_identity);
  }

  if (scalar ? cu.l1_scalar().try_read_peer_cached(address, data, process_id, generation,
                                                   cache_identity)
             : cu.l1_vector().try_read_peer_cached(address, data, process_id, generation,
                                                   cache_identity))
    return true;
  if (cu.l2()->try_read_peer_cached(address, data, process_id, generation, cache_identity))
    return true;
  for (uint32_t iod = 0; iod < soc.num_iods(); ++iod)
    if (soc.iod(iod)->msc()->try_read_peer_cached(address, data, process_id, generation,
                                                  cache_identity))
      return true;
  return false;
}

amdgpu::PreparedCanonicalWriteback SimulatedKfd::prepare_gpu_alias_writeback(
    uint32_t process_id, uint32_t gpu_id, const amdgpu::GpuVmAccess &expected, uint64_t address,
    size_t size, const GpuAliasIngressGate::DrainAuthority *authority,
    const GpuAliasIngressGate::EffectWindow *effect_window) {
  using amdgpu::PreparedCanonicalWriteback;
  using amdgpu::VmAccessKind;
  using amdgpu::VmAccessOutcome;

  // Both cache levels use 128-byte lines. Include the siblings of a partial
  // external write so a dirty whole MSC line can be judged before cache locks.
  constexpr uint64_t kCacheLineBytes = 128;
  if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
    return PreparedCanonicalWriteback(VmAccessOutcome::Malformed);
  const uint64_t line_start = address & ~(kCacheLineBytes - 1);
  const uint64_t last_byte = address + size - 1;
  const uint64_t line_last = last_byte | (kCacheLineBytes - 1);
  if (line_last == std::numeric_limits<uint64_t>::max())
    return PreparedCanonicalWriteback(VmAccessOutcome::Malformed);
  const uint64_t envelope_bytes = line_last + 1 - line_start;
  if (envelope_bytes > std::numeric_limits<size_t>::max())
    return PreparedCanonicalWriteback(VmAccessOutcome::Malformed);
  const amdgpu::MemoryIngressSpan span{.address = line_start,
                                       .size = static_cast<size_t>(envelope_bytes),
                                       .access = VmAccessKind::Write};

  GpuAliasJoin joined =
      join_gpu_aliases_for_writeback(process_id, gpu_id, expected, std::span(&span, 1), authority);
  if (joined.outcome != VmAccessOutcome::Complete)
    return PreparedCanonicalWriteback(joined.outcome);
  if (joined.aliases.empty())
    return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);

  const uint64_t cache_identity = expected.cache_identity();
  const amdgpu::VmCacheNamespace cache_namespace = expected.cache_namespace();
  const std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate = expected.retain_effect_gate();
  if (cache_identity == 0 || !cache_namespace || !effect_gate || expected.revoked())
    return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);

  std::vector<GpuAliasIngressGate::Key> keys;
  keys.reserve(joined.aliases.size());
  for (const GpuAliasBinding &binding : joined.aliases) {
    if (binding.alias.admission_generation == 0 || binding.alias.backing_id == 0 ||
        binding.vmid == 0 || binding.cache_namespace != cache_namespace ||
        binding.cache_identity != cache_identity || binding.span.size == 0)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    keys.push_back(GpuAliasIngressGate::key(binding.alias));
  }
  std::optional<amdgpu::MemoryIngressPermit> permit;
  bool controlled_writeback_authorized = false;
  if (effect_window) {
    if (!*effect_window)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    permit = alias_ingress_gate_->try_acquire_for_effect(keys, *effect_window);
    controlled_writeback_authorized = permit && static_cast<bool>(*permit);
    if (!permit) {
      // Scoped preparation resolves unrelated dirty tags for overlap proof.
      // Their open generations take ordinary ingress; any selected key in a
      // mixed line remains frozen and makes that admission fail closed.
      permit = alias_ingress_gate_->try_acquire(keys);
    }
  } else if (authority) {
    permit = alias_ingress_gate_->try_acquire_for_drained_owner(keys, *authority);
    controlled_writeback_authorized = permit && static_cast<bool>(*permit) &&
                                      alias_ingress_gate_->drained_owner_covers(keys, *authority);
  } else {
    permit = alias_ingress_gate_->try_acquire(keys);
  }
  if (!permit || !*permit)
    return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);

  std::vector<PreparedCanonicalWriteback::Entry> entries;
  entries.reserve(joined.aliases.size());
  for (const GpuAliasBinding &binding : joined.aliases) {
    const KfdProcess::GpuPeerBackingDecision granted =
        binding.alias.local ? joined.process->gpu_local_backing_grant(gpu_id, binding.span.address,
                                                                      binding.span.size)
                            : joined.process->gpu_peer_backing_grant(gpu_id, binding.span.address,
                                                                     binding.span.size);
    if (!granted.admission.allowed || !granted.backing ||
        granted.backing.backing_id() != binding.alias.backing_id ||
        binding.span.address < binding.alias.gpu_va)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    const uint64_t relative = binding.span.address - binding.alias.gpu_va;
    if (relative > binding.alias.extent || binding.span.size > binding.alias.extent - relative ||
        relative > std::numeric_limits<uint64_t>::max() - binding.alias.backing_offset)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    const uint64_t backing_offset = binding.alias.backing_offset + relative;
    if (granted.backing.file_offset() != backing_offset ||
        granted.backing.size() != binding.span.size)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    FabricBacking backing = make_fabric_backing(granted.backing);
    if (!backing || backing.backing_id() != binding.alias.backing_id ||
        backing.offset() != backing_offset || backing.extent() != binding.span.size)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    entries.emplace_back(binding.span.address, binding.span.size, binding.vmid,
                         binding.cache_namespace, binding.cache_identity,
                         binding.alias.admission_generation, binding.alias.backing_id,
                         backing_offset, std::move(backing), granted.backing.bytes());
  }

  SoC *soc = nullptr;
  amdgpu::LegacyGpuVmAdapter *legacy_vm = nullptr;
  {
    std::lock_guard lock(process_mutex_);
    const auto current = processes_.find(process_id);
    const auto pending = authority ? pending_closes_.find(process_id) : pending_closes_.end();
    if ((current == processes_.end() || current->second != joined.process) &&
        (pending == pending_closes_.end() || pending->second->process != joined.process))
      return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
    uint32_t ordinal = 0;
    while (ordinal < gpus_.size() && gpus_[ordinal].gpu_id != gpu_id)
      ++ordinal;
    if (ordinal == gpus_.size() || !gpus_[ordinal].soc || !gpus_[ordinal].legacy_vm ||
        joined.process->gpu(ordinal).address_space != joined.address_space)
      return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
    soc = gpus_[ordinal].soc;
    legacy_vm = gpus_[ordinal].legacy_vm.get();
  }
  const KfdProcess::GpuAliasSnapshots current = joined.process->gpu_alias_snapshots();
  if (current.canonical != joined.publication || current.local != joined.local_publication)
    return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);

  const std::optional<amdgpu::GpuVmAccess> live = soc->gpu_vm().snapshot(joined.address_space);
  if (!live || live->revoked() || live->cache_namespace() != cache_namespace ||
      live->cache_identity() != cache_identity || live->retain_effect_gate() != effect_gate ||
      live->info().vmid != joined.aliases.front().vmid)
    return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
  const std::shared_ptr<amdgpu::LegacyAddressSpace> address_space =
      legacy_vm->address_space_snapshot(live->info().vmid);
  if (!address_space)
    return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
  for (const GpuAliasBinding &binding : joined.aliases) {
    if (binding.vmid != live->info().vmid ||
        !joined.process->gpu_alias_grant_current(binding.alias) ||
        !address_space->is_range_mapped(binding.span.address, binding.span.size,
                                        live->info().vmid) ||
        !address_space->has_writable_host_backing(binding.span.address, live->info().vmid,
                                                  binding.span.size) ||
        live->classify_data_route(binding.span.address, binding.span.size, VmAccessKind::Write,
                                  false) == amdgpu::VmDataRoute::Faulted)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    const KfdProcess::GpuPeerBackingDecision current_backing =
        binding.alias.local ? joined.process->gpu_local_backing_grant(gpu_id, binding.span.address,
                                                                      binding.span.size)
                            : joined.process->gpu_peer_backing_grant(gpu_id, binding.span.address,
                                                                     binding.span.size);
    const uint64_t offset =
        binding.alias.backing_offset + (binding.span.address - binding.alias.gpu_va);
    if (!current_backing.admission.allowed || !current_backing.backing ||
        current_backing.backing.backing_id() != binding.alias.backing_id ||
        current_backing.backing.file_offset() != offset ||
        current_backing.backing.size() != binding.span.size)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    // The ordinary grouped resolver and an effect-window fallback are not a
    // selected canonical effect. Neither may drain a controlled dirty writer.
    // Selected effects and drained-owner retirement retain their exact alias
    // authority and continue to use the prepared canonical sink.
    if (!controlled_writeback_authorized &&
        soc->gpu_vm().controlled_cache_backing_overlaps(binding.vmid, binding.span.address,
                                                        binding.span.size))
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
  }
  return PreparedCanonicalWriteback(std::move(entries), std::move(*permit), effect_gate);
}

amdgpu::PreparedCanonicalWriteback SimulatedKfd::prepare_gpu_alias_writeback_for_vmid(
    uint32_t gpu_id, uint32_t vmid, uint64_t address, size_t size,
    const GpuAliasIngressGate::DrainAuthority *authority,
    const GpuAliasIngressGate::EffectWindow *effect_window) {
  using amdgpu::PreparedCanonicalWriteback;
  using amdgpu::VmAccessOutcome;

  // The group scan can find dirty lines from a process unrelated to the
  // current fabric request. Resolve the tag's own process and GPU binding.
  SoC *soc = nullptr;
  amdgpu::AddressSpaceHandle handle;
  {
    std::lock_guard lock(process_mutex_);
    const auto process = processes_.find(vmid);
    const auto pending = authority ? pending_closes_.find(vmid) : pending_closes_.end();
    if (process == processes_.end() && pending == pending_closes_.end())
      return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
    const auto owner = process != processes_.end() ? process->second : pending->second->process;
    if (owner->process_id() != vmid)
      return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
    uint32_t ordinal = 0;
    while (ordinal < gpus_.size() && gpus_[ordinal].gpu_id != gpu_id)
      ++ordinal;
    if (ordinal == gpus_.size() || !gpus_[ordinal].soc || !gpus_[ordinal].legacy_vm)
      return PreparedCanonicalWriteback(VmAccessOutcome::Unavailable);
    soc = gpus_[ordinal].soc;
    handle = owner->gpu(ordinal).address_space;
  }
  if (!handle)
    return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
  const std::optional<amdgpu::GpuVmAccess> live = soc->gpu_vm().snapshot(handle);
  if (!live || live->revoked() || live->info().vmid != vmid)
    return PreparedCanonicalWriteback(VmAccessOutcome::Revoked);
  return prepare_gpu_alias_writeback(vmid, gpu_id, *live, address, size, authority, effect_window);
}

SimulatedKfd::CacheWritebackResolver SimulatedKfd::cache_writeback_resolver() {
  const std::weak_ptr<CuIngressAccess> weak_access = cu_ingress_access_;
  return [weak_access](uint32_t gpu_id, uint32_t vmid, uint64_t address,
                       size_t size) -> amdgpu::PreparedCanonicalWriteback {
    const std::shared_ptr<CuIngressAccess> access = weak_access.lock();
    if (!access)
      return amdgpu::PreparedCanonicalWriteback(amdgpu::VmAccessOutcome::Revoked);
    std::lock_guard lock(access->mutex);
    if (!access->driver)
      return amdgpu::PreparedCanonicalWriteback(amdgpu::VmAccessOutcome::Revoked);
    return access->driver->prepare_gpu_alias_writeback_for_vmid(gpu_id, vmid, address, size);
  };
}

std::optional<amdgpu::CanonicalLineWitness> SimulatedKfd::witness_gpu_alias_line_for_vmid(
    uint32_t gpu_id, uint32_t vmid, uint64_t line_address, size_t line_size,
    uint64_t cache_identity, uint64_t admission_generation,
    const GpuAliasIngressGate::EffectWindow *effect_window) {
  constexpr size_t kLineSize = amdgpu::L2Cache::LINE_SIZE;
  if (vmid == 0 || cache_identity == 0 || admission_generation == 0 || line_size != kLineSize ||
      (line_address & (kLineSize - 1)) != 0 || line_address > UINT64_MAX - kLineSize)
    return std::nullopt;

  std::shared_ptr<KfdProcess> process;
  SoC *soc = nullptr;
  amdgpu::LegacyGpuVmAdapter *legacy_vm = nullptr;
  amdgpu::AddressSpaceHandle handle;
  {
    std::lock_guard lock(process_mutex_);
    const auto found = processes_.find(vmid);
    if (found == processes_.end() || found->second->process_id() != vmid)
      return std::nullopt;
    process = found->second;
    const auto gpu = std::ranges::find_if(
        gpus_, [gpu_id](const GpuDevice &device) { return device.gpu_id == gpu_id; });
    if (gpu == gpus_.end() || !gpu->soc || !gpu->legacy_vm)
      return std::nullopt;
    soc = gpu->soc;
    legacy_vm = gpu->legacy_vm.get();
    handle = process->gpu(static_cast<uint32_t>(gpu - gpus_.begin())).address_space;
  }
  if (!handle)
    return std::nullopt;
  const auto expected = soc->gpu_vm().snapshot(handle);
  if (!expected || expected->revoked() || expected->info().vmid != vmid ||
      expected->cache_identity() != cache_identity)
    return std::nullopt;
  const amdgpu::MemoryIngressSpan span{
      .address = line_address, .size = line_size, .access = amdgpu::VmAccessKind::Write};
  GpuAliasJoin joined =
      join_gpu_aliases_for_writeback(vmid, gpu_id, *expected, std::span(&span, 1), nullptr);
  if (joined.outcome != amdgpu::VmAccessOutcome::Complete || joined.process != process ||
      joined.address_space != handle || joined.aliases.size() != 1)
    return std::nullopt;
  const GpuAliasBinding &binding = joined.aliases.front();
  if (binding.vmid != vmid || binding.cache_identity != cache_identity ||
      binding.alias.admission_generation != admission_generation || binding.alias.backing_id == 0 ||
      binding.span.address != line_address || binding.span.size != line_size ||
      line_address < binding.alias.gpu_va || binding.alias.extent < line_size ||
      line_address - binding.alias.gpu_va > binding.alias.extent - line_size ||
      binding.alias.backing_offset > UINT64_MAX - (line_address - binding.alias.gpu_va))
    return std::nullopt;
  const std::array keys{GpuAliasIngressGate::key(binding.alias)};
  std::optional<amdgpu::MemoryIngressPermit> permit;
  if (effect_window)
    permit = alias_ingress_gate_->try_acquire_for_effect(keys, *effect_window);
  if (!permit) {
    // A dirty line outside the selected effect takes ordinary ingress; only
    // an exact window member may bypass the selected alias freeze.
    permit = alias_ingress_gate_->try_acquire(keys);
  }
  if (!permit || !*permit)
    return std::nullopt;

  {
    std::lock_guard lock(process_mutex_);
    const auto found = processes_.find(vmid);
    if (found == processes_.end() || found->second != process)
      return std::nullopt;
  }
  const KfdProcess::GpuAliasSnapshots current = process->gpu_alias_snapshots();
  if (current.canonical != joined.publication || current.local != joined.local_publication ||
      !process->gpu_alias_grant_current(binding.alias))
    return std::nullopt;
  const auto live = soc->gpu_vm().snapshot(handle);
  const auto address_space = legacy_vm->address_space_snapshot(vmid);
  if (!live || live->revoked() || live->info().vmid != vmid ||
      live->cache_namespace() != expected->cache_namespace() ||
      live->cache_identity() != cache_identity || !address_space ||
      !address_space->is_range_mapped(line_address, line_size, vmid) ||
      !address_space->has_writable_host_backing(line_address, vmid, line_size) ||
      live->classify_data_route(line_address, line_size, amdgpu::VmAccessKind::Write, false) ==
          amdgpu::VmDataRoute::Faulted)
    return std::nullopt;
  const KfdProcess::GpuPeerBackingDecision backing =
      binding.alias.local ? process->gpu_local_backing_grant(gpu_id, line_address, line_size)
                          : process->gpu_peer_backing_grant(gpu_id, line_address, line_size);
  const uint64_t backing_offset =
      binding.alias.backing_offset + (line_address - binding.alias.gpu_va);
  if (!backing.admission.allowed || !backing.backing ||
      backing.backing.backing_id() != binding.alias.backing_id ||
      backing.backing.file_offset() != backing_offset || backing.backing.size() != line_size ||
      (backing_offset & (kLineSize - 1)) != 0 || backing_offset > UINT64_MAX - line_size)
    return std::nullopt;

  struct ReadOnlyRetention {
    std::shared_ptr<KfdProcess> process;
    KfdProcess::GpuBackingLease backing;
    amdgpu::MemoryIngressPermit permit;
  };
  auto retention = std::make_shared<ReadOnlyRetention>(
      ReadOnlyRetention{std::move(process), backing.backing, std::move(*permit)});
  return amdgpu::CanonicalLineWitness{.gpu_id = gpu_id,
                                      .vmid = vmid,
                                      .line_address = line_address,
                                      .line_size = line_size,
                                      .cache_identity = cache_identity,
                                      .admission_generation = admission_generation,
                                      .backing_id = binding.alias.backing_id,
                                      .backing_offset = backing_offset,
                                      .retention = std::move(retention)};
}

std::shared_ptr<KfdProcess> SimulatedKfd::find_local_process() const {
  return find_process(local_process_id_);
}

uint32_t SimulatedKfd::mapping_flags_for_handle(uint64_t handle) const {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return 0;
  std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
  auto it = proc->allocations_.find(handle);
  return it != proc->allocations_.end()
             ? it->second.flags | (it->second.shared ? KFD_IOC_ALLOC_MEM_FLAGS_COHERENT : 0)
             : 0;
}

KfdProcess::PageTableMutationResult
SimulatedKfd::map_to_gpu(KfdProcess &proc, uint64_t gpu_va, void *host_ptr, size_t size,
                         amdgpu::Mtype mtype, KfdProcess::HostExtentOwner owner,
                         KfdProcess::PageTableTransaction *transaction) {
  util::Logger::cp("MAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec, " mtype=", static_cast<int>(mtype));
  return transaction ? proc.map_pages(*transaction, gpu_va, host_ptr, size, mtype, owner)
                     : proc.map_pages(gpu_va, host_ptr, size, mtype, owner);
}

KfdProcess::PageTableMutationResult
SimulatedKfd::unmap_from_gpu(KfdProcess &proc, uint64_t gpu_va, size_t size,
                             KfdProcess::PageTableTransaction *transaction) {
  util::Logger::cp("UNMAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec);
  return transaction ? proc.unmap_pages(*transaction, gpu_va, size)
                     : proc.unmap_pages(gpu_va, size);
}

void SimulatedKfd::update_cp_doorbell_base(uint32_t gpu_ordinal, uint32_t process_id, void *base) {
  if (gpu_ordinal >= gpus_.size())
    return;
  auto &g = gpus_[gpu_ordinal];
  if (!g.soc)
    return;
  g.soc->set_process_doorbell_base(process_id, base);
}

std::string SimulatedKfd::redirect_sysfs_path(const char *path) const {
  auto result = redirect_sysfs_root_path(path, topology_path(), topology().drm_path());
  if (!result.empty()) {
    util::Logger::vm("sysfs redirect: ", path, " -> ", result);
    return result;
  }
  return {};
}

bool SimulatedKfd::handles_drm_render_minor(uint32_t minor) const {
  if (topology().drm_path().empty())
    return false;
  return gpu_ordinal_for_render_minor(minor).has_value();
}

const Sysfs::GpuInfo *SimulatedKfd::gpu_info_for_render_minor(uint32_t minor) const {
  if (topology().drm_path().empty())
    return nullptr;
  const auto ordinal = gpu_ordinal_for_render_minor(minor);
  return ordinal && *ordinal < gpu_infos_.size() ? &gpu_infos_[*ordinal] : nullptr;
}

std::optional<uint32_t> SimulatedKfd::gpu_ordinal_for_render_minor(uint32_t minor) const {
  if (num_gpus() == 0)
    return std::nullopt;
  // A legacy single-GPU driver accepted any render minor. Keep that routing
  // when old configs advertise a different minor from their PM4 clients.
  // Manifest projections require an exact match to avoid claiming a host GPU.
  if (!strict_render_minor_ && num_gpus() == 1)
    return 0;
  if (!gpu_infos_.empty()) {
    const auto it =
        std::find_if(gpu_infos_.begin(), gpu_infos_.end(),
                     [minor](const Sysfs::GpuInfo &gpu) { return gpu.drm_render_minor == minor; });
    if (it == gpu_infos_.end())
      return std::nullopt;
    const auto ordinal = static_cast<uint32_t>(std::distance(gpu_infos_.begin(), it));
    return ordinal < num_gpus() ? std::optional<uint32_t>{ordinal} : std::nullopt;
  }
  // A driver constructed directly by a unit test can submit before setup_topology.
  if (num_gpus() == 1)
    return 0;
  return minor >= 128 && minor - 128 < num_gpus() ? std::optional<uint32_t>{minor - 128}
                                                  : std::nullopt;
}

void SimulatedKfd::setup_topology(const config::KfdDeviceConfig &dev, uint32_t num_xcc) {
  if (!dev.present)
    return;

  setup_topology(gpu_info_from_config(dev, num_xcc));
}

SimulatedKfd::SimulatedKfd(SoC &soc, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : daemon_mode_(daemon_mode),
      debug_identity_validation_hook_(std::move(debug_identity_validation_hook)),
      debug_session_reaper_([this](std::stop_token stop) { reap_exited_debug_sessions(stop); }) {
  // Resolve the real libc entry points once, up front and single-threaded, so no
  // passthrough call site ever triggers a first-time dlsym under a per-process
  // lock. Idempotent: a no-op if the interposer already resolved the table.
  libc_passthrough().resolve();
  cu_ingress_access_->driver = this;
  GpuDevice device;
  device.soc = &soc;
  device.legacy_vm = std::make_unique<amdgpu::LegacyGpuVmAdapter>(soc.gpu_vm(), soc.memory());
  gpus_.push_back(std::move(device));
}

SimulatedKfd::SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : daemon_mode_(daemon_mode),
      debug_identity_validation_hook_(std::move(debug_identity_validation_hook)),
      debug_session_reaper_([this](std::stop_token stop) { reap_exited_debug_sessions(stop); }) {
  libc_passthrough().resolve();
  cu_ingress_access_->driver = this;
  for (size_t i = 0; i < socs.size(); ++i) {
    GpuDevice device;
    device.soc = socs[i];
    device.gpu_id = i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id();
    device.legacy_vm =
        std::make_unique<amdgpu::LegacyGpuVmAdapter>(socs[i]->gpu_vm(), socs[i]->memory());
    gpus_.push_back(std::move(device));
  }
}

SimulatedKfd::GpuDevice *SimulatedKfd::find_gpu(uint32_t gpu_id) {
  for (auto &g : gpus_)
    if (g.gpu_id == gpu_id)
      return &g;
  return nullptr;
}

const SimulatedKfd::GpuDevice *SimulatedKfd::find_gpu(uint32_t gpu_id) const {
  for (auto &g : gpus_)
    if (g.gpu_id == gpu_id)
      return &g;
  return nullptr;
}

SimulatedKfd::~SimulatedKfd() {
  {
    std::unique_lock lock(cu_ingress_access_->mutex);
    cu_ingress_access_->driver = nullptr;
    cu_ingress_access_->effects_drained.wait(
        lock, [&] { return cu_ingress_access_->active_effect_callbacks == 0; });
  }
  {
    std::lock_guard lock(peer_endpoint_access_->mutex);
    peer_endpoint_access_->endpoint = nullptr;
  }
  // CPs may outlive this frontend. Revoke callback admission and drain any
  // callback already executing before touching the state captured by them.
  for (GpuDevice &gpu : gpus_)
    gpu.interrupt_subscription.reset();

  debug_session_reaper_.request_stop();
  debug_session_reaper_.join();

  const int close_result = close_all_processes();
  {
    std::lock_guard lock(cu_ingress_access_->mutex);
    cu_ingress_access_->standalone_after_shutdown = close_result == 0;
  }
}

void SimulatedKfd::reap_exited_debug_sessions(std::stop_token stop) {
  std::unique_lock<std::mutex> lock(debug_sessions_mutex_);
  while (!stop.stop_requested()) {
    if (debug_sessions_.empty()) {
      debug_sessions_cv_.wait(lock, stop, [&] { return !debug_sessions_.empty(); });
    } else {
      debug_sessions_cv_.wait_for(lock, stop, std::chrono::milliseconds(10), [] { return false; });
    }
    if (stop.stop_requested())
      break;
    std::vector<std::pair<pid_t, std::shared_ptr<KfdProcess>>> released;
    std::vector<pid_t> retry_notifications;
    for (auto it = debug_sessions_.begin(); it != debug_sessions_.end();) {
      if (pidfd_is_exited(it->second.debugger_pidfd.get()) == 1) {
        // The debugger is gone and will never ack; release any inferior
        // blocked in the RUNTIME_ENABLE handshake for it.
        cancel_runtime_handshake(it->first);
        // Resolve the debuggee now, while the session's pinned identity still
        // vouches for the pid. After the erase the number alone could name a
        // process that merely reused it.
        released.emplace_back(it->first, find_process_by_client_pid(it->first));
        it = debug_sessions_.erase(it);
      } else if (pidfd_is_exited(it->second.target_pidfd.get()) == 1) {
        if (!it->second.target_exited) {
          if (auto target = find_process_by_client_pid(it->first))
            revoke_target_mem_routing(target->process_id());
          it->second.owned_dbg_fd.reset();
          it->second.target_mem_fd.reset();
          it->second.dbg_fd = -1;
          it->second.enabled = false;
          it->second.target_exited = true;
        }
        ++it;
      } else {
        if (it->second.enabled && it->second.notification_retry_needed)
          retry_notifications.push_back(it->first);
        ++it;
      }
    }
    // A crashed debugger must not strand the inferior. Erasing the session only
    // stops new debug traffic; the waves it left stopped, the closed queue
    // gates and the target-memory routing all have to be undone too, exactly as
    // an explicit detach does. Done outside debug_sessions_mutex_ because the
    // release takes CU wave-state locks in the opposite order to the engine
    // thread.
    lock.unlock();
    // The shared_ptr in `released` is what keeps proc alive across the
    // unlocked release; release_debuggee_state only borrows it.
    for (auto &[pid, proc] : released) {
      release_debuggee_state(pid, proc.get());
    }
    for (pid_t pid : retry_notifications) {
      [[maybe_unused]] const int notification_result =
          retry_debug_notifications(pid, /*invoke_result_hook=*/true);
    }
    lock.lock();
  }
}

void SimulatedKfd::setup_topology(const Sysfs::GpuInfo &gpu) {
  strict_render_minor_ = false;
  if (!gpus_.empty())
    gpus_[0].gpu_id = gpu.gpu_id;
  gpu_infos_ = {gpu};
  topology_.generate(gpu);
  topology_.setup_environment();
}

void SimulatedKfd::setup_topology(const std::vector<config::KfdDeviceConfig> &devs,
                                  uint32_t num_xcc, bool publish_peer_links,
                                  bool strict_render_minor) {
  strict_render_minor_ = strict_render_minor;
  std::vector<Sysfs::GpuInfo> infos;
  infos.reserve(devs.size());
  for (auto &dev : devs) {
    if (!dev.present)
      continue;
    auto info = gpu_info_from_config(dev, num_xcc);
    if (!publish_peer_links) {
      // Executable manifest routes model local effects, but they do not
      // establish native XGMI discovery through the stock driver. Keep the
      // KFD metadata that ROCr uses to infer that path conservative.
      info.hive_id = 0;
      info.num_sdma_xgmi_engines = 0;
    }
    infos.push_back(std::move(info));
  }
  if (infos.empty())
    return;
  for (size_t i = 0; i < infos.size() && i < gpus_.size(); ++i)
    gpus_[i].gpu_id = infos[i].gpu_id;
  gpu_infos_ = std::move(infos);
  topology_.generate(gpu_infos_, publish_peer_links);
  topology_.setup_environment();
}

bool SimulatedKfd::is_doorbell_range(const void *addr, size_t length) const {
  auto p = find_process(local_process_id_);
  if (!p || !addr || length == 0)
    return false;
  // Check every GPU ordinal's doorbell page: dispatch_mmap/dispatch_munmap install
  // and tear down a doorbell page per ordinal, so a multi-GPU process has more than
  // one to guard (checking only ordinal 0 would leave a higher ordinal's page
  // unprotected against a client mprotect). Snapshot each page/size under
  // alloc_mutex_ so a concurrent dispatch_mmap/dispatch_munmap (which mutate these
  // under the same lock) cannot tear the pointer/size read.
  std::lock_guard<std::mutex> lock(p->alloc_mutex_);
  for (const auto &gs : p->gpu_state_) {
    if (ranges_overlap(addr, length, gs.doorbell_monitor_page, gs.doorbell_page_size))
      return true;
    for (const auto &view : gs.doorbell_views)
      if (ranges_overlap(addr, length, view.page, gs.doorbell_page_size))
        return true;
  }
  return false;
}

void *SimulatedKfd::mmap_replacing_client_doorbell_views(void *addr, size_t length, int prot,
                                                         int flags, int fd, off_t offset) {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return safe_mmap(addr, length, prot, flags, fd, offset);

  // Serialize the replacement with KFD teardown and doorbell mmap. Keep
  // alloc_mutex_ held through the real mmap so mprotect/munmap cannot observe
  // the interval between PTE removal and CPU-view replacement.
  std::lock_guard<std::mutex> op_lock(proc->op_mutex_);
  std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
  auto va_transaction = proc->begin_page_table_transaction();

  for (const auto &gs : proc->gpu_state_) {
    if (ranges_overlap(addr, length, gs.doorbell_monitor_page, gs.doorbell_page_size)) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    for (const auto &view : gs.doorbell_views) {
      if (ranges_overlap(addr, length, view.page, gs.doorbell_page_size) &&
          !range_contains(addr, length, view.page, gs.doorbell_page_size)) {
        // Doorbell GPU mappings and ownership are tracked for the complete
        // client view. Replacing only part would leave the remaining CPU range
        // live but no longer tracked or GPU-mapped.
        errno = EINVAL;
        return MAP_FAILED;
      }
    }
  }

  struct RetiredView {
    size_t gpu_ordinal;
    size_t page_size;
    KfdProcess::PerGpuState::DoorbellView view;
  };
  std::vector<RetiredView> retired;
  try {
    for (size_t ord = 0; ord < proc->gpu_state_.size(); ++ord) {
      const auto &gs = proc->gpu_state_[ord];
      for (const auto &view : gs.doorbell_views)
        if (ranges_overlap(addr, length, view.page, gs.doorbell_page_size))
          retired.push_back({.gpu_ordinal = ord, .page_size = gs.doorbell_page_size, .view = view});
    }
  } catch (const std::bad_alloc &) {
    errno = ENOMEM;
    return MAP_FAILED;
  }

  for (const auto &entry : retired)
    if (proc->has_foreign_resident_overlap(va_transaction, entry.view.gpu_va, entry.page_size,
                                           {KfdProcess::VaOwnerKind::Kfd, 0}) !=
        KfdProcess::PageTableMutationResult::Applied) {
      errno = EINVAL;
      return MAP_FAILED;
    }

  // Keep the original CPU views and bookkeeping live until the replacement
  // succeeds. If a PTE removal or the real mmap fails, restore every removed
  // translation before returning the original mapping to its owner.
  const auto restore = [&](size_t count) noexcept {
    bool restored = true;
    for (size_t i = 0; i < count; ++i) {
      const auto &entry = retired[i];
      try {
        restored &= map_to_gpu(*proc, entry.view.gpu_va, entry.view.page, entry.page_size,
                               amdgpu::Mtype::UC, KfdProcess::HostExtentOwner::Application,
                               &va_transaction) == KfdProcess::PageTableMutationResult::Applied;
      } catch (...) {
        restored = false;
      }
    }
    return restored;
  };

  size_t removed = 0;
  for (const auto &entry : retired) {
    try {
      if (unmap_from_gpu(*proc, entry.view.gpu_va, entry.page_size, &va_transaction) !=
          KfdProcess::PageTableMutationResult::Applied) {
        errno = restore(removed) ? EINVAL : EIO;
        return MAP_FAILED;
      }
      ++removed;
    } catch (...) {
      errno = restore(removed + 1) ? ENOMEM : EIO;
      return MAP_FAILED;
    }
  }

  void *mapped = safe_mmap(addr, length, prot, flags, fd, offset);
  if (mapped != MAP_FAILED) {
    for (const auto &entry : retired) {
      auto &views = proc->gpu_state_[entry.gpu_ordinal].doorbell_views;
      std::erase_if(views, [&](const auto &view) { return view.page == entry.view.page; });
    }
    return mapped;
  }

  const int mmap_errno = errno;
  errno = restore(removed) ? mmap_errno : EIO;
  return MAP_FAILED;
}

bool SimulatedKfd::ensure_fd_created() {
  if (fd_.load(std::memory_order_acquire) >= 0)
    return true;
  int new_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (new_fd < 0)
    return false;
  int expected = -1;
  // CAS so only one racing opener publishes the KFD eventfd; a loser closes
  // its own descriptor and adopts the winner's, avoiding a double create / fd leak.
  if (!fd_.compare_exchange_strong(expected, new_fd, std::memory_order_acq_rel,
                                   std::memory_order_acquire))
    libc_passthrough().close(new_fd);
  return true;
}

void SimulatedKfd::init_command_processors_locked() {
  for (size_t i = 0; i < gpus_.size(); ++i) {
    auto &g = gpus_[i];
    if (g.cps_initialized)
      continue;
    if (!g.soc)
      continue;
    // The WAVES field the CP writes into compute_tmpring_size must be a
    // multiple of the per-XCC shader-engine count, or rocdbgapi disables
    // private memory access outright (architecture.cpp,
    // gfx9_architecture_t::scratch_memory_region). It recovers that count from
    // the topology we publish: os_driver_kfd.cpp derives
    // shader_engine_count = array_count * num_xcc / simd_arrays_per_engine and
    // then divides by xcc_count, which inverts Sysfs::GpuInfo's
    // array_count_per_xcc() exactly back to num_shader_engines. Read that same
    // field here rather than recomputing a quotient, so the value the CP rounds
    // to and the divisor rocdbgapi applies cannot drift apart.
    const uint32_t scratch_wave_divisor =
        i < gpu_infos_.size() ? std::max(1u, gpu_infos_[i].num_shader_engines) : 1u;
    const uint32_t xcc_count = g.soc->num_xcds();
    for (uint32_t xcc_id = 0; xcc_id < xcc_count; ++xcc_id) {
      auto *cp = g.soc->xcd(xcc_id)->command_processor();
      if (!cp)
        continue;
      cp->set_scratch_wave_divisor(scratch_wave_divisor);
    }
    // Same source as the apertures GET_PROCESS_APERTURES_NEW and the DBG_TRAP
    // device snapshot advertise: what the shaders translate LDS/scratch against
    // must be what the runtime and the debugger were told.
    const kfd_process_device_apertures ap = gpu_apertures(static_cast<uint32_t>(i));
    g.soc->set_apertures(ap.lds_base, ap.lds_limit, ap.scratch_base, ap.scratch_limit);
    g.interrupt_subscription =
        amdgpu::InterruptSubscription([this](uint32_t process_id, uint32_t event_id) {
          if (const auto override = interrupt_override_for_testing_.sink()) {
            override.deliver(process_id, event_id);
            return;
          }
          std::lock_guard<std::mutex> ilk(interrupt_mutex_);
          const std::unordered_map<uint32_t, EventState *>::iterator event =
              event_dispatch_.find(process_id);
          if (event != event_dispatch_.end()) {
            util::Logger::cp("INTERRUPT_ROUTE: pid=", process_id, " event_id=", event_id,
                             " found=true");
            event->second->signal_interrupt(event_id);
          } else {
            util::Logger::cp("INTERRUPT_ROUTE: pid=", process_id, " event_id=", event_id,
                             " found=false");
          }
        });
    g.soc->for_each_cp([this, i](amdgpu::CommandProcessor *cp) {
      cp->set_scratch_backing_resolver([this](uint32_t process_id) -> uint64_t {
        std::lock_guard<std::mutex> plk(process_mutex_);
        for (auto &[fd, proc] : processes_) {
          if (proc->process_id() == process_id) {
            for (auto &gs : proc->gpu_state_) {
              if (gs.scratch_backing_va != 0)
                return gs.scratch_backing_va << 16;
            }
          }
        }
        return 0;
      });
      cp->set_scratch_backing_allocator(
          [this](uint32_t process_id, uint64_t gpu_va, size_t size) -> bool {
            return allocate_scratch_backing(process_id, gpu_va, size);
          });
      for (auto *cu : cp->compute_units()) {
        const uint32_t gpu_ordinal = static_cast<uint32_t>(i);
        const std::weak_ptr<CuIngressAccess> ingress_access = cu_ingress_access_;
        cu->set_memory_ingress_handler([ingress_access, gpu_id = gpus_[i].gpu_id](
                                           uint32_t process_id, const amdgpu::GpuVmAccess &expected,
                                           std::span<const amdgpu::MemoryIngressSpan> spans) {
          const auto access = ingress_access.lock();
          if (!access)
            return amdgpu::MemoryIngressAdmission{.outcome = amdgpu::VmAccessOutcome::Revoked,
                                                  .permit = {}};
          std::lock_guard lock(access->mutex);
          if (!access->driver)
            return amdgpu::MemoryIngressAdmission{.outcome = amdgpu::VmAccessOutcome::Revoked,
                                                  .permit = {}};
          return access->driver->admit_gpu_alias_ingress(process_id, gpu_id, expected, spans);
        });
        const auto make_peer_read =
            [ingress_access, gpu_id = gpus_[i].gpu_id, cu,
             soc = gpus_[i].soc](bool scalar) -> amdgpu::MemoryPipeline::PeerCacheReadHandler {
          return [ingress_access, gpu_id, cu, soc, scalar](
                     uint32_t process_id, const amdgpu::GpuVmAccess &expected, uint64_t address,
                     std::span<std::byte> bytes, const amdgpu::VmCompletedRead *completed) {
            const std::shared_ptr<CuIngressAccess> access = ingress_access.lock();
            if (!access)
              return false;
            SimulatedKfd *driver = nullptr;
            amdgpu::MemoryIngressPermit driver_pin;
            {
              std::lock_guard lock(access->mutex);
              if (!access->driver || access->active_effect_callbacks == SIZE_MAX)
                return false;
              driver_pin = amdgpu::MemoryIngressPermit([access] {
                std::lock_guard lock(access->mutex);
                --access->active_effect_callbacks;
                access->effects_drained.notify_all();
              });
              ++access->active_effect_callbacks;
              driver = access->driver;
            }
            return driver->try_gpu_peer_cache_read(process_id, gpu_id, expected, address, bytes,
                                                   completed ? &completed->stamp : nullptr, *cu,
                                                   *soc, scalar);
          };
        };
        cu->set_peer_cache_read_handlers(make_peer_read(true), make_peer_read(false));
        const std::shared_ptr<amdgpu::DeviceCacheCoherence> cache_coherence =
            gpus_[i].soc->cache_coherence();
        CacheWritebackResolver resolve_writeback = cache_writeback_resolver();
        cu->set_prepared_atomic_boundary_handler(
            [atomic_access = cu_ingress_access_, cache_coherence, gpu_id = gpus_[i].gpu_id,
             resolve_writeback = std::move(resolve_writeback)](
                uint32_t vmid) -> std::optional<amdgpu::DeviceCacheMaintenanceLease> {
              {
                std::lock_guard lock(atomic_access->mutex);
                if (!atomic_access->driver) {
                  // Only a completed KFD shutdown restores standalone VMID-zero
                  // atomics; an incomplete retirement keeps its caches closed.
                  if (vmid == 0 && atomic_access->standalone_after_shutdown)
                    return std::nullopt;
                  throw util::Exception(std::string_view("KFD CU cache owner is gone"));
                }
              }
              // Resolution reenters CuIngressAccess outside cache locks. Do not
              // hold its mutex while the prepared group takes those locks.
              const std::array domains{amdgpu::DeviceCacheDomain{cache_coherence, gpu_id}};
              auto leases = amdgpu::DeviceCacheCoherence::acquire_prepared_atomic_boundary(
                  domains, resolve_writeback);
              if (leases.size() != domains.size())
                throw util::Exception(std::string_view("KFD CU cache lease is missing"));
              return std::move(leases.front());
            });
        cu->set_trap_handler_resolver([this, gpu_ordinal](const amdgpu::Wavefront &wf) {
          return resolve_trap_handler(wf, gpu_ordinal);
        });
        cu->set_sendmsg_handler([this](amdgpu::Wavefront &wf, uint32_t message) {
          return on_wave_sendmsg(wf, message);
        });
        cu->set_queue_exception_handler([this, gpu_ordinal](
                                            uint32_t queue_id, uint32_t process_id, uint64_t status,
                                            bool retain_failure_for_debugger,
                                            amdgpu::QueueExceptionDeliveryPolicy delivery_policy) {
          const bool delivered =
              gpu_ordinal < gpus_.size() &&
              signal_runtime_queue_exception(
                  gpus_[gpu_ordinal].gpu_id, queue_id, process_id, status,
                  delivery_policy == amdgpu::QueueExceptionDeliveryPolicy::WaitForAcknowledgment);
          if (runtime_exception_result_hook_for_testing_)
            runtime_exception_result_hook_for_testing_(delivered);
          complete_runtime_queue_exception(process_id, queue_id, status, delivered,
                                           retain_failure_for_debugger);
          return delivered;
        });
        cu->set_trap_completion_handler(
            [this](amdgpu::Wavefront &wf) { on_wave_trap_complete(wf); });
        cu->set_single_step_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_single_step_complete(wf); });
        cu->set_watchpoint_handler([this](amdgpu::Wavefront &wf, uint64_t address, uint32_t bytes,
                                          bool is_write, bool is_atomic) {
          return on_wave_watchpoint(wf, address, bytes, is_write, is_atomic);
        });
        cu->set_illegal_inst_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_illegal_instruction(wf); });
        cu->set_execution_failure_handler([this](amdgpu::Wavefront &wf, bool retain_wave) {
          return on_wave_execution_failure(wf, retain_wave);
        });
        cu->set_alu_exception_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_alu_exception(wf); });
        cu->set_memory_violation_handler(
            [this](amdgpu::Wavefront &wf, uint64_t address, bool is_write) {
              return on_wave_memory_violation(wf, address, is_write);
            });
      }
    });
    g.cps_initialized = true;
  }
}

bool SimulatedKfd::register_process_address_spaces(const std::shared_ptr<KfdProcess> &proc,
                                                   pid_t client_pid, bool passthrough) {
  for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
    GpuDevice &gpu = gpus_[ordinal];
    if (gpu.soc == nullptr || gpu.soc->memory() == nullptr)
      continue;

    KfdProcess::PerGpuState &gpu_state = proc->gpu(ordinal);
    auto fault_reporter = std::make_shared<GpuFaultReporter>(this, gpu.gpu_id, proc);
    auto *const fault_reporter_ptr = fault_reporter.get();
    const std::weak_ptr<PeerEndpointAccess> endpoint_access = peer_endpoint_access_;
    const std::weak_ptr<CuIngressAccess> effect_access = cu_ingress_access_;
    const std::shared_ptr<GpuAliasIngressGate> peer_alias_ingress = alias_ingress_gate_;
    const std::shared_ptr<PeerCanonicalLineGate> peer_line_gate = peer_line_gate_;
    gpu_state.address_space = gpu.legacy_vm->register_address_space(
        proc->process_id(),
        {.page_table = &proc->page_table_,
         .page_table_mutex = &proc->page_table_mutex_,
         .page_table_generation = proc->page_table_generation(),
         .request_mutex = proc->page_table_request_mutex(),
         .mutation_epoch = proc->page_table_mutation_epoch(),
         .page_table_cache_state = proc->page_table_cache_state(),
         .client_pid = client_pid,
         .client_mem_fd = -1,
         .passthrough = passthrough,
         .fault_reporter = fault_reporter_ptr,
         .access_admission =
             [proc, gpu_id = gpu.gpu_id](uint64_t address, std::size_t size, amdgpu::VmAccessKind) {
               const auto grant = proc->gpu_access_grant(gpu_id, address, size);
               return amdgpu::VmAccessAdmission{
                   .outcome = grant.allowed ? amdgpu::VmAccessOutcome::Complete
                                            : amdgpu::VmAccessOutcome::Faulted,
                   .generation = grant.generation,
                   .cacheable = grant.cacheable,
                   .direct_backing_write_allowed = grant.direct_backing_write_allowed};
             },
         .raw_write_gate = proc->raw_write_gate_,
         .data_route =
             [proc, gpu_id = gpu.gpu_id, endpoint_access](uint64_t address, std::size_t size,
                                                          amdgpu::VmAccessKind access) {
               const auto grant = proc->gpu_access_grant(gpu_id, address, size);
               amdgpu::LegacyDataRoute route{
                   // An allowed zero-extent grant found no overlapping allocation
                   // in the entire request, as with queue rings and driver mappings.
                   .first_extent_bytes = grant.allowed && grant.first_extent_bytes == 0
                                             ? size
                                             : grant.first_extent_bytes,
                   .candidate = grant.first_peer_candidate
                                    ? amdgpu::LegacyDeferredCandidate::Peer
                                    : (access == amdgpu::VmAccessKind::Write &&
                                               grant.first_controlled_local_write_candidate
                                           ? amdgpu::LegacyDeferredCandidate::ControlledLocalWrite
                                           : amdgpu::LegacyDeferredCandidate::None)};
               if (!grant.allowed) {
                 route.route = amdgpu::VmDataRoute::Faulted;
               } else if (route.candidate ==
                          amdgpu::LegacyDeferredCandidate::ControlledLocalWrite) {
                 route.route = amdgpu::VmDataRoute::Faulted;
                 const auto endpoint = endpoint_access.lock();
                 if (endpoint) {
                   std::lock_guard lock(endpoint->mutex);
                   if (endpoint->endpoint)
                     route.route = amdgpu::VmDataRoute::Deferred;
                 }
                 // A local effect selects every cache alias of this backing
                 // line, including UC imports with a distinct GPU VA.
                 if (route.route == amdgpu::VmDataRoute::Deferred)
                   route.first_extent_bytes = std::min<std::size_t>(
                       route.first_extent_bytes,
                       amdgpu::L2Cache::LINE_SIZE - (address & (amdgpu::L2Cache::LINE_SIZE - 1)));
               } else if (grant.peer_candidate) {
                 route.route = amdgpu::VmDataRoute::Faulted;
                 if (access != amdgpu::VmAccessKind::Read && access != amdgpu::VmAccessKind::Write)
                   return route;
                 if (grant.mixed_peer_owners)
                   return route;
                 const auto endpoint = endpoint_access.lock();
                 if (!endpoint)
                   return route;
                 std::lock_guard lock(endpoint->mutex);
                 if (endpoint->endpoint &&
                     endpoint->endpoint->can_route(gpu_id, grant.peer_owner_gpu_id))
                   route.route = amdgpu::VmDataRoute::Deferred;
                 if (route.route == amdgpu::VmDataRoute::Deferred &&
                     access == amdgpu::VmAccessKind::Write && grant.cacheable)
                   route.first_extent_bytes = std::min<std::size_t>(
                       route.first_extent_bytes,
                       amdgpu::L2Cache::LINE_SIZE - (address & (amdgpu::L2Cache::LINE_SIZE - 1)));
               }
               return route;
             },
         .prepare_deferred_transfer =
             [proc, gpu_id = gpu.gpu_id, endpoint_access, effect_access, peer_alias_ingress,
              peer_line_gate](amdgpu::LegacyDeferredCandidate candidate, uint64_t address,
                              std::size_t size, std::span<const std::byte> bytes, bool write,
                              const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                              const amdgpu::GpuVmAccess &executing_access,
                              std::shared_ptr<const amdgpu::LegacyAddressSpace> address_space)
             -> std::unique_ptr<amdgpu::VmDeferredTransfer> {
           const bool local_write =
               candidate == amdgpu::LegacyDeferredCandidate::ControlledLocalWrite;
           if (candidate != amdgpu::LegacyDeferredCandidate::Peer && !local_write)
             return {};
           if (local_write && !write)
             return {};
           auto granted = proc->gpu_peer_backing_grant(gpu_id, address, size);
           const auto &admission = granted.admission;
           if (executing_access.cache_identity() == 0 ||
               executing_access.retain_effect_gate() != gate || executing_access.revoked() ||
               executing_access.info().vmid != proc->process_id() || !address_space ||
               !admission.allowed ||
               !(local_write ? admission.first_controlled_local_write_candidate
                             : admission.first_peer_candidate) ||
               admission.domain != KfdProcess::GpuGrantDomain::Vram ||
               admission.first_extent_bytes < size || !granted.backing ||
               granted.backing.owner_gpu_id() != admission.owner_gpu_id ||
               (local_write &&
                (admission.owner_gpu_id != gpu_id ||
                 !granted.backing.controlled_peer_for_testing() ||
                 size > amdgpu::L2Cache::LINE_SIZE - (address & (amdgpu::L2Cache::LINE_SIZE - 1)) ||
                 executing_access.classify_data_route(address, size, amdgpu::VmAccessKind::Write,
                                                      false) != amdgpu::VmDataRoute::Deferred ||
                 !address_space->is_range_mapped(address, size, proc->process_id()) ||
                 !address_space->has_writable_host_backing(address, proc->process_id(), size))) ||
               (write &&
                (bytes.size() != size || (!local_write && admission.cacheable &&
                                          !granted.backing.controlled_peer_for_testing()))))
             return {};
           const auto aliases = proc->gpu_alias_snapshot();
           const auto alias_it = std::ranges::find_if(*aliases, [&](const auto &alias) {
             return alias.process_id == proc->process_id() &&
                    alias.process_incarnation == proc->process_incarnation() &&
                    alias.gpu_id == gpu_id && alias.backing_id == granted.backing.backing_id() &&
                    !alias.local && address >= alias.gpu_va &&
                    address - alias.gpu_va < alias.extent &&
                    size <= alias.extent - (address - alias.gpu_va) &&
                    alias.backing_offset <= UINT64_MAX - (address - alias.gpu_va) &&
                    alias.backing_offset + (address - alias.gpu_va) ==
                        granted.backing.file_offset();
           });
           if (alias_it == aliases->end())
             return {};
           const KfdProcess::GpuAliasRecord alias = *alias_it;
           if (local_write && !proc->gpu_alias_grant_current(alias))
             return {};
           const auto endpoint = endpoint_access.lock();
           if (!endpoint)
             return {};
           std::lock_guard lock(endpoint->mutex);
           if (!endpoint->endpoint ||
               (!local_write && !endpoint->endpoint->can_route(gpu_id, admission.owner_gpu_id)))
             return {};
           FabricEndpoint *const selected_endpoint = endpoint->endpoint;
           FabricBacking backing = make_fabric_backing(std::move(granted.backing));
           const bool cacheable_peer = admission.cacheable;
           FabricEndpoint::EffectAdmission admit_effect =
               [proc, gpu_id, address, size, alias, endpoint_access, selected_endpoint,
                effect_access, peer_alias_ingress, peer_line_gate,
                address_space = std::move(address_space), execution = executing_access,
                cacheable_peer, local_write,
                gate](const FabricBacking &effect_backing, bool effect_write,
                      amdgpu::MemoryIngressPermit &request_reservation)
               -> FabricEndpoint::EffectAuthority {
             FabricEndpoint::EffectAuthority result;
             result.outcome = amdgpu::VmAccessOutcome::Revoked;
             if (execution.revoked() || execution.cache_identity() == 0 ||
                 execution.retain_effect_gate() != gate || !effect_backing ||
                 effect_backing.backing_id() != alias.backing_id ||
                 (local_write ? effect_backing.owner_gpu_id() != gpu_id
                              : effect_backing.owner_gpu_id() == gpu_id) ||
                 effect_backing.offset() != alias.backing_offset + (address - alias.gpu_va) ||
                 effect_backing.extent() != size || (local_write && !effect_write))
               return result;
             const auto requester_current = [&] {
               if (!proc->gpu_alias_grant_current(alias) ||
                   execution.classify_data_route(address, size,
                                                 effect_write ? amdgpu::VmAccessKind::Write
                                                              : amdgpu::VmAccessKind::Read,
                                                 false) != amdgpu::VmDataRoute::Deferred)
                 return false;
               const KfdProcess::GpuPeerBackingDecision current =
                   proc->gpu_peer_backing_grant(gpu_id, address, size);
               const KfdProcess::GpuGrantDecision &grant = current.admission;
               if (!grant.allowed ||
                   !(local_write ? grant.first_controlled_local_write_candidate
                                 : grant.first_peer_candidate) ||
                   (!local_write && grant.mixed_peer_owners) ||
                   grant.domain != KfdProcess::GpuGrantDomain::Vram ||
                   grant.owner_gpu_id != effect_backing.owner_gpu_id() ||
                   grant.first_extent_bytes < size || grant.cacheable != cacheable_peer ||
                   !current.backing || current.backing.backing_id() != alias.backing_id ||
                   current.backing.file_offset() != effect_backing.offset() ||
                   current.backing.size() != size ||
                   (local_write && (!current.backing.controlled_peer_for_testing() ||
                                    size > amdgpu::L2Cache::LINE_SIZE -
                                               (address & (amdgpu::L2Cache::LINE_SIZE - 1)))) ||
                   !address_space->is_range_mapped(address, size, proc->process_id()) ||
                   !(effect_write ? address_space->has_writable_host_backing(
                                        address, proc->process_id(), size)
                                  : address_space->has_readable_host_backing(
                                        address, proc->process_id(), size)))
                 return false;
               const std::shared_ptr<PeerEndpointAccess> live_endpoint = endpoint_access.lock();
               if (!live_endpoint)
                 return false;
               std::lock_guard route_lock(live_endpoint->mutex);
               return live_endpoint->endpoint == selected_endpoint &&
                      (local_write ||
                       selected_endpoint->can_route(gpu_id, effect_backing.owner_gpu_id()));
             };

             const std::shared_ptr<CuIngressAccess> live_access = effect_access.lock();
             if (!live_access)
               return result;
             SimulatedKfd *driver = nullptr;
             amdgpu::MemoryIngressPermit driver_pin;
             {
               std::lock_guard lock(live_access->mutex);
               if (!live_access->driver || live_access->active_effect_callbacks == SIZE_MAX)
                 return result;
               // Build the release closure before incrementing the pin count.
               driver_pin = amdgpu::MemoryIngressPermit([live_access] {
                 std::lock_guard lock(live_access->mutex);
                 --live_access->active_effect_callbacks;
                 live_access->effects_drained.notify_all();
               });
               ++live_access->active_effect_callbacks;
               driver = live_access->driver;
             }

             if (effect_write && !request_reservation) {
               if (!requester_current())
                 return result;
               std::optional<amdgpu::MemoryIngressPermit> reservation = peer_line_gate->try_acquire(
                   effect_backing.backing_id(), effect_backing.offset(), effect_backing.extent());
               if (!reservation) {
                 result.outcome = amdgpu::VmAccessOutcome::Unavailable;
                 return result;
               }
               request_reservation = std::move(*reservation);
             }

             bool selection_busy = false;
             std::optional<LivePeerAliasSelection> selected = driver->try_select_live_peer_aliases(
                 alias, effect_backing, std::move(driver_pin), cacheable_peer || effect_write,
                 selection_busy);
             if (selected) {
               const bool requester_scope_current = std::ranges::any_of(
                   selected->scopes, [&](const amdgpu::DeviceCacheAliasScope &scope) {
                     return scope.domain.gpu_id == gpu_id && scope.vmid == proc->process_id() &&
                            selected->requester_cache_namespace == execution.cache_namespace() &&
                            scope.admission_generation == alias.admission_generation &&
                            scope.address <= address && address - scope.address < scope.size &&
                            size <= scope.size - (address - scope.address);
                   });
               if (!requester_scope_current || !requester_current())
                 return result;
               const std::weak_ptr<LivePeerEffectLease> weak_lease = selected->lease;
               result.resolve_writeback =
                   [driver, weak_lease](uint32_t dirty_gpu_id, uint32_t vmid, uint64_t line_addr,
                                        size_t line_size) -> amdgpu::PreparedCanonicalWriteback {
                 const std::shared_ptr<LivePeerEffectLease> lease = weak_lease.lock();
                 if (!lease)
                   return amdgpu::PreparedCanonicalWriteback(amdgpu::VmAccessOutcome::Revoked);
                 return driver->prepare_gpu_alias_writeback_for_vmid(
                     dirty_gpu_id, vmid, line_addr, line_size, nullptr, &lease->window());
               };
               result.resolve_witness = [driver, weak_lease](uint32_t dirty_gpu_id, uint32_t vmid,
                                                             uint64_t line_addr, size_t line_size,
                                                             uint64_t cache_identity,
                                                             uint64_t admission_generation)
                   -> std::optional<amdgpu::CanonicalLineWitness> {
                 const auto lease = weak_lease.lock();
                 if (!lease)
                   return std::nullopt;
                 return driver->witness_gpu_alias_line_for_vmid(
                     dirty_gpu_id, vmid, line_addr, line_size, cache_identity, admission_generation,
                     &lease->window());
               };
               result.cache_aliases = std::move(selected->scopes);
               result.permit = amdgpu::MemoryIngressPermit([lease = std::move(selected->lease)] {});
               result.outcome = amdgpu::VmAccessOutcome::Complete;
               return result;
             }

             if (cacheable_peer || local_write) {
               result.outcome = selection_busy ? amdgpu::VmAccessOutcome::Unavailable
                                               : amdgpu::VmAccessOutcome::Faulted;
               return result;
             }

             // An active ingress or concurrent publication keeps the established
             // UC full-domain path available until selected effects are required.
             const std::array keys{GpuAliasIngressGate::key(alias)};
             std::optional<amdgpu::MemoryIngressPermit> permit =
                 peer_alias_ingress->try_acquire(keys);
             if (!permit || !requester_current())
               return result;
             result.outcome = amdgpu::VmAccessOutcome::Complete;
             result.permit = std::move(*permit);
             return result;
           };
           auto prepared =
               local_write ? selected_endpoint->prepare_local_write(
                                 gpu_id, std::move(backing), bytes, gate, std::move(admit_effect))
               : write ? selected_endpoint->prepare_write(gpu_id, admission.owner_gpu_id,
                                                          std::move(backing), bytes, gate,
                                                          std::move(admit_effect))
                       : selected_endpoint->prepare_read(gpu_id, admission.owner_gpu_id,
                                                         std::move(backing), gate,
                                                         std::move(admit_effect));
           return std::move(prepared.operation);
         }},
        std::move(fault_reporter));
    if (!gpu_state.address_space) {
      for (uint32_t registered = 0; registered <= ordinal; ++registered) {
        GpuDevice &registered_gpu = gpus_[registered];
        if (registered_gpu.soc == nullptr)
          continue;
        KfdProcess::PerGpuState &registered_state = proc->gpu(registered);
        if (registered_state.address_space) {
          (void)registered_gpu.legacy_vm->unregister_address_space(registered_state.address_space);
          registered_state.address_space = {};
        }
      }
      return false;
    }
  }
  return true;
}

int SimulatedKfd::open() {
  std::shared_lock publication_lock(alias_publication_mutex_);
  static std::once_flag raise_nofile_flag;
  std::call_once(raise_nofile_flag, [] {
    struct rlimit rl {};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 8192) {
      rl.rlim_cur = std::min<rlim_t>(rl.rlim_max, 65536);
      setrlimit(RLIMIT_NOFILE, &rl);
    }
  });

  // Hold process_mutex_ across fd creation, process selection/retain, and the
  // returned fd load so a racing dup2 (which invalidates fd_ via
  // invalidate_primary_fd, also under process_mutex_) cannot clear fd_ between
  // publishing it and returning it. Either open() completes and returns a valid
  // fd, or invalidation wins first and ensure_fd_created() re-mints one below.
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (!ensure_fd_created())
    return -1;
  if (!daemon_mode_ && local_process_id_ != 0 && processes_.contains(local_process_id_)) {
    processes_[local_process_id_]->retain_open();
    return fd_.load(std::memory_order_acquire);
  }
  uint32_t pid = next_process_id_++;
  auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
  // client_pid_ caches getpid() at open() time; DBG_TRAP uses it to resolve a
  // self-debug target, so it must match the caller's live pid. A fork() child
  // inherits this cache stale, but under the fork-then-exec contract that child
  // never reaches the driver at all (owner_pid_ gates every interposed entry
  // point), so the stale value is unobservable until exec replaces the image.
  proc->set_client_pid(static_cast<pid_t>(getpid()));
  proc->event_state_.reset();
  if (!register_process_address_spaces(proc, 0, !daemon_mode_))
    return -1;
  processes_[pid] = proc;
  local_process_id_ = pid;

  {
    std::lock_guard<std::mutex> ilk(interrupt_mutex_);
    event_dispatch_[pid] = &proc->event_state_;
  }

  init_command_processors_locked();

  return fd_.load(std::memory_order_acquire);
}

void SimulatedKfd::report_memory_fault(const std::shared_ptr<KfdProcess> &proc, uint64_t addr,
                                       uint32_t gpu_id, amdgpu::MemoryFaultCause cause) {
  if (proc == nullptr)
    return;

  MemoryFault fault{};
  fault.va = addr;
  fault.gpu_id = gpu_id;
  // Indeterminate deliberately sets neither failure bit: the violation is real
  // but its cause was never established, and naming a protection the address
  // may not have would send the runtime after the wrong problem. It does not
  // set `imprecise` either -- that reports an inexact faulting address, which
  // is not what is uncertain here.
  fault.not_present = cause == amdgpu::MemoryFaultCause::NotPresent;
  fault.read_only = cause == amdgpu::MemoryFaultCause::ReadOnly;

  // Reporting is best effort by design: a process that never created a
  // memory-exception event has no channel for this, exactly as on hardware,
  // and the warning is the only remaining place the violation can surface.
  if (!proc->event_state_.signal_memory_fault(fault)) {
    util::Logger::warn("GPU memory violation at 0x", std::hex, addr, std::dec, " (process ",
                       proc->process_id(),
                       ") has no registered memory-exception event to report it on");
  }
}

void SimulatedKfd::set_process_client_pid(uint32_t process_id, pid_t client_pid) {
  std::lock_guard<std::mutex> lk(process_mutex_);
  auto it = processes_.find(process_id);
  if (it != processes_.end()) {
    it->second->set_client_pid(client_pid);
    for (auto &g : gpus_) {
      if (g.soc == nullptr)
        continue;
      const auto ordinal = static_cast<uint32_t>(&g - gpus_.data());
      const amdgpu::AddressSpaceHandle address_space = it->second->gpu(ordinal).address_space;
      if (address_space)
        (void)g.legacy_vm->set_client_pid(address_space, client_pid);
    }
  }
}

uint32_t SimulatedKfd::open_process(pid_t client_pid) {
  std::shared_lock publication_lock(alias_publication_mutex_);
  uint32_t pid;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    // Create the backing fd under process_mutex_ so both entry points
    // (open()/open_process()) serialize fd creation and never publish two
    // different memfds; ensure_fd_created() itself CASes so it is also safe from
    // any lock-free caller.
    if (!ensure_fd_created())
      return 0;
    // Client-PID process reuse (and its matching retain) is a daemon-mode
    // feature: multiple client opens of the same PID share one process and
    // balance against multiple close()/release_open() calls. Gating reuse on
    // daemon_mode_ keeps it symmetric with close() — outside daemon mode every
    // open creates a fresh process so the first close cannot tear down a
    // still-referenced one.
    if (daemon_mode_ && client_pid > 0) {
      for (auto &[id, proc] : processes_) {
        if (proc->client_pid() == client_pid) {
          proc->retain_open();
          return id;
        }
      }
    }
    pid = next_process_id_++;
    auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
    if (client_pid > 0)
      proc->set_client_pid(client_pid);
    proc->event_state_.reset();
    if (!register_process_address_spaces(proc, client_pid, false))
      return 0;
    processes_[pid] = proc;

    {
      std::lock_guard<std::mutex> ilk(interrupt_mutex_);
      event_dispatch_[pid] = &proc->event_state_;
    }

    init_command_processors_locked();
  }

  if (client_pid > 0) {
    const std::shared_ptr<KfdProcess> process = find_process(pid);
    std::lock_guard<std::mutex> debug_lock(debug_sessions_mutex_);
    auto session = debug_sessions_.find(client_pid);
    if (session != debug_sessions_.end() && session->second.target_mem_fd.get() >= 0) {
      for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
        GpuDevice &gpu = gpus_[ordinal];
        if (gpu.soc == nullptr)
          continue;
        const amdgpu::AddressSpaceHandle address_space = process->gpu(ordinal).address_space;
        if (address_space)
          (void)gpu.legacy_vm->set_client_mem_fd(address_space,
                                                 session->second.target_mem_fd.get());
      }
    }
  }

  return pid;
}

LinuxKfd::PrimaryInvalidation SimulatedKfd::invalidate_primary_fd(int fd) {
  if (fd < 0)
    return PrimaryInvalidation::kNotPrimary;
  // Serialize with open()/open_process(), which hold process_mutex_ across fd
  // creation and the returned-fd load, so this cannot clear fd_ mid-open.
  std::lock_guard<std::mutex> lk(process_mutex_);
  int expected = fd;
  // The local primary fd holds one counted open reference, so on a successful
  // clear the caller must drop it (kClearedDropRef). Report kNotPrimary if a
  // concurrent overwrite already cleared fd_, so the caller does not double
  // release.
  if (fd_.compare_exchange_strong(expected, -1, std::memory_order_acq_rel))
    return PrimaryInvalidation::kClearedDropRef;
  return PrimaryInvalidation::kNotPrimary;
}

bool SimulatedKfd::retain_local_open() {
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (local_process_id_ == 0)
    return false;
  auto it = processes_.find(local_process_id_);
  if (it == processes_.end())
    return false;
  it->second->retain_open();
  return true;
}

uint32_t SimulatedKfd::local_open_ref_count() const {
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (local_process_id_ == 0)
    return 0;
  auto it = processes_.find(local_process_id_);
  return it != processes_.end() ? it->second->open_ref_count() : 0;
}

void SimulatedKfd::begin_local_shutdown() {
  // Release an indefinite-timeout WAIT_EVENTS on the local process so it returns
  // and drops the driver snapshot that would otherwise keep this object alive.
  // Snapshot the process, then fire the wake with process_mutex_ RELEASED:
  // begin_wait_cancel() takes the process's own event mutex, and WAIT_EVENTS does
  // not take process_mutex_, so the parked thread can make progress.
  //
  // Deliberately NOT notify_closing()/signal_page_shutdown(): those turn live ioctls
  // into -EBADF/-ESRCH and poison the signal page, which is the caller-visible part
  // of teardown and belongs to close(). What this does instead is release the
  // waiters, so the ordering is "wake, then destroy" rather than two half-teardowns.
  // The wake is one-way for this driver's life: the interposer calls it only after
  // it has committed to teardown, so nothing clears the flags and every subsequent
  // wait reports the same benign timeout.
  if (auto proc = find_local_process()) {
    proc->event_state_.begin_wait_cancel();
    // WAIT_EVENTS is not the only call that parks. RUNTIME_ENABLE waits on the
    // debugger's acknowledgement, and is dispatched outside op_mutex_ precisely
    // because it blocks -- so that thread holds a driver snapshot for the whole
    // wait and teardown would sit behind it until the handshake deadline expired.
    // Cancelling is equally pure: the waiter reports the handshake as cancelled
    // and no event, page or process state is mutated.
    cancel_runtime_handshake(proc->client_pid());
  }
}

int SimulatedKfd::close() { return close(local_process_id_); }

int SimulatedKfd::close_all_processes() {
  // Snapshot both public and pending owners. A daemon process can have several
  // open references, so bound the number of close calls by its snapshotted
  // count plus one pending retry. Continue other processes after an error and
  // return the first failure to the checked shutdown caller.
  std::vector<std::pair<uint32_t, uint32_t>> pids;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    pids.reserve(processes_.size() + pending_closes_.size());
    for (const auto &[pid, proc] : processes_)
      pids.emplace_back(pid, proc->open_ref_count());
    for (const auto &[pid, pending] : pending_closes_) {
      [[maybe_unused]] const auto &_ = pending;
      pids.emplace_back(pid, 1);
    }
  }
  int first_error = 0;
  for (const auto &[pid, maximum_attempts] : pids) {
    bool finished = false;
    for (uint64_t attempt = 0; attempt <= maximum_attempts; ++attempt) {
      const int result = close(pid);
      if (result != 0) {
        if (first_error == 0)
          first_error = result;
        finished = true;
        break;
      }
      std::lock_guard lock(process_mutex_);
      if (!processes_.contains(pid) && !pending_closes_.contains(pid)) {
        finished = true;
        break;
      }
    }
    if (!finished && first_error == 0)
      first_error = -EAGAIN;
  }
  return first_error;
}

int SimulatedKfd::close(uint32_t process_id) {
  std::lock_guard close_lock(close_mutex_);
  std::shared_ptr<PendingClose> pending;
  std::shared_ptr<KfdProcess> candidate;
  std::vector<uint32_t> queue_ids;
  std::vector<KfdProcess::QueueDoorbellInfo> queues;

  {
    std::shared_lock publication_lock(alias_publication_mutex_);
    std::lock_guard<std::mutex> lk(process_mutex_);
    auto it = processes_.find(process_id);
    if (it == processes_.end()) {
      const auto closing = pending_closes_.find(process_id);
      if (closing == pending_closes_.end())
        return 0;
      pending = closing->second;
    } else {
      if (it->second->open_ref_count() > 1) {
        (void)it->second->release_open();
        return 0;
      }
      candidate = it->second;
    }
  }

  try {
    struct PendingPublicationWriter {
      KfdProcess &process;
      std::atomic<uint32_t> &writers;
      std::atomic<uint64_t> &release_epoch;
      PendingPublicationWriter(KfdProcess &process, std::atomic<uint32_t> &writers,
                               std::atomic<uint64_t> &release_epoch)
          : process(process), writers(writers), release_epoch(release_epoch) {
        process.canonical_retirements_pending_.fetch_add(1, std::memory_order_acq_rel);
        writers.fetch_add(1, std::memory_order_acq_rel);
      }
      ~PendingPublicationWriter() {
        process.canonical_retirements_pending_.fetch_sub(1, std::memory_order_release);
        writers.fetch_sub(1, std::memory_order_release);
        release_epoch.fetch_add(1, std::memory_order_acq_rel);
        release_epoch.notify_all();
      }
    } publication_writer(*(candidate ? candidate : pending->process),
                         alias_publication_writers_pending_, alias_publication_release_epoch_);
    std::unique_lock exclusive_publication_lock(alias_publication_mutex_);
    if (candidate) {
      std::lock_guard process_lock(process_mutex_);
      const auto it = processes_.find(process_id);
      if (it == processes_.end() || it->second != candidate)
        return -EAGAIN;
      if (candidate->open_ref_count() > 1) {
        (void)candidate->release_open();
        return 0;
      }
      // A different process's failed batch needs this process to remain
      // resolver-visible. Do not consume its last open reference until that
      // claim has completed or been retried.
      if (alias_ingress_gate_->close_claimed() &&
          !alias_ingress_gate_->close_claimed_by_process(candidate->process_incarnation()))
        return -EAGAIN;
      pending = std::make_shared<PendingClose>(candidate);
      pending_closes_.emplace(process_id, pending);
      (void)candidate->release_open();
      processes_.erase(it);
    }
    std::lock_guard pending_lock(pending->mutex);
    if (pending->complete)
      return 0;
    auto &proc = *pending->process;

    // Serialize ALL teardown against any in-flight ioctl on this process. ioctl()
    // only snapshots a shared_ptr via find_process() and does NOT retain an open
    // reference, so an ioctl that started before this close() removed the process
    // from the table can still be running (or about to run) under proc.op_mutex_.
    // Acquire op_mutex_ BEFORE any teardown step — including event_dispatch_ erase
    // and mem->unregister_process() — so those cannot overlap an active
    // op_mutex_-guarded ioctl handler and break CP interrupt routing / memory
    // translation mid-ioctl. notify_closing() (below, still under op_mutex_) sets
    // the closing flag that dispatch_ioctl checks right after it takes op_mutex_,
    // so any ioctl that was blocked on op_mutex_ behind this close() will observe
    // is_closing() and bail instead of operating on a torn-down process.
    //
    // The process was already erased from processes_ above, so no NEW ioctl can
    // find it. Ordering is safe: process_mutex_ was released before taking
    // op_mutex_, so this does not nest against dispatch_ioctl's op_mutex_ ->
    // process_mutex_ order. WAIT_EVENTS does not take op_mutex_, so notify_closing()
    // / signal_page_shutdown() below still wake any parked waiter.
    //
    // mmap/munmap also take op_mutex_; is_doorbell_range and scratch allocation
    // use alloc_mutex_ independently. Keep allocation teardown under that lock.
    std::vector<std::shared_ptr<const KfdProcess::GpuBacking>> retiring_backings;
    std::unique_lock<std::mutex> op_lock(proc.op_mutex_);

    // Process-table removal prevents new source admissions. Revoke every
    // surviving GTT source generation and drain its synchronous readers before
    // close changes grants, VM bindings, or host mappings.
    if (!pending->started) {
      {
        std::lock_guard alloc_lock(proc.alloc_mutex_);
        for (const auto &entry : proc.allocations_)
          if (entry.second.private_gtt_copy_source_gate)
            entry.second.private_gtt_copy_source_gate->revoke();
      }
      // Walk by handle so no iterator or allocation lock survives the wait.
      // The process has left the table and op_mutex_ excludes new KFD BOs.
      std::optional<uint64_t> previous_handle;
      while (true) {
        uint64_t selected_handle = 0;
        std::shared_ptr<PrivateGttCopySourceGate> gate;
        {
          std::lock_guard alloc_lock(proc.alloc_mutex_);
          for (const auto &[handle, allocation] : proc.allocations_) {
            if (!allocation.private_gtt_copy_source_gate ||
                (previous_handle && handle <= *previous_handle) ||
                (gate && handle >= selected_handle))
              continue;
            selected_handle = handle;
            gate = allocation.private_gtt_copy_source_gate;
          }
        }
        if (!gate)
          break;
        gate->wait_quiescent();
        previous_handle = selected_handle;
      }
    }

    // Imports that already duplicated a backing fd own that allocation. Revoke
    // only this process's export-store descriptors before its allocations are
    // dismantled, so a new import cannot acquire a handle after close begins.
    if (!pending->started) {
      std::lock_guard<std::mutex> alloc_lock(proc.alloc_mutex_);
      // Publish closing in the same VA writer order as GEM. A GEM transaction
      // pinned before process-table removal finishes before this flag is set;
      // one that enters later observes it and cannot install another PTE.
      {
        auto va_transaction = proc.begin_page_table_transaction();
        proc.event_state_.notify_closing();
      }
      {
        std::lock_guard<std::mutex> ipc_lock(ipc_mutex_);
        for (auto it = ipc_store_.begin(); it != ipc_store_.end();) {
          if (it->second.source_process_id == process_id) {
            if (it->second.backing_owner)
              retiring_backings.push_back(it->second.backing_owner);
            if (it->second.backing_memfd >= 0)
              libc_passthrough().close(it->second.backing_memfd);
            it = ipc_store_.erase(it);
          } else {
            ++it;
          }
        }
      }
      pending->started = true;
    }

    // A CU/SDMA request may have passed the final process-table check just before
    // erase. Its permit must leave before any grant, VM binding, or backing is
    // retired. The gate condition scans existing entries without allocating.
    // A runtime-enable waiter that entered before table removal may hold a
    // driver snapshot. Wake its debugger handshake before waiting for ingress.
    cancel_runtime_handshake(proc.client_pid());
    op_lock.unlock();
    alias_ingress_gate_->wait_for_process_quiescent(proc.process_incarnation());
    op_lock.lock();
    proc.event_state_.signal_page_shutdown();

    // A failed FREE owns the gate-wide close claim and may have already written
    // its dirty lines or retired some grants. Resume it through its original
    // marker before beginning a process-wide batch. A failed UNMAP similarly
    // retains one exact generation and must finish through its original retry.
    if (!pending->maintenance_prepared) {
      if (fail_next_alias_close_prepare_.exchange(false, std::memory_order_acq_rel))
        return -ENOMEM;
      try {
        // A failed ordinary mmap/munmap owns an exact closed batch, not a
        // single UNMAP generation. FREE can retry that batch and inherit its
        // completed-writeback marker before close scans single generations.
        std::vector<uint64_t> failed_mappings;
        {
          std::lock_guard lock(ordinary_mapping_mutex_);
          for (const auto &[identity, mapping] : pending_ordinary_mappings_)
            if (identity.first == proc.process_incarnation() && !mapping->keys.empty())
              failed_mappings.push_back(identity.second);
        }
        for (uint64_t handle : failed_mappings) {
          kfd_ioctl_free_memory_of_gpu_args args{};
          args.handle = handle;
          std::shared_ptr<const KfdProcess::GpuBacking> retiring_backing;
          const int result = free_memory_ioctl(proc, &args, retiring_backing);
          if (result != 0)
            return result;
        }

        std::vector<uint64_t> failed_frees;
        {
          std::lock_guard lock(proc.alloc_mutex_);
          for (const auto &[handle, alloc] : proc.allocations_)
            if (alloc.retirement_in_progress)
              failed_frees.push_back(handle);
        }
        for (uint64_t handle : failed_frees) {
          kfd_ioctl_free_memory_of_gpu_args args{};
          args.handle = handle;
          std::shared_ptr<const KfdProcess::GpuBacking> retiring_backing;
          const int result = free_memory_ioctl(proc, &args, retiring_backing);
          if (result != 0)
            return result;
        }

        const KfdProcess::GpuAliasSnapshots live_aliases = proc.gpu_alias_snapshots();
        const auto retry_closed_alias = [&](const KfdProcess::GpuAliasRecord &alias) {
          const auto key = GpuAliasIngressGate::key(alias);
          if (!alias_ingress_gate_->closed(key))
            return 0;
          uint64_t handle = 0;
          {
            std::lock_guard lock(proc.alloc_mutex_);
            for (const auto &[candidate_handle, alloc] : proc.allocations_) {
              const auto generation = alloc.admission_generations.find(alias.gpu_id);
              const auto owner = alias.local ? alloc.local_backing_owner : alloc.backing_owner;
              if (alloc.gpu_va == alias.gpu_va && alloc.size == alias.extent && owner &&
                  owner->backing_id() == alias.backing_id &&
                  generation != alloc.admission_generations.end() &&
                  generation->second == alias.admission_generation) {
                handle = candidate_handle;
                break;
              }
            }
          }
          if (handle == 0)
            return -EIO;
          uint32_t gpu_id = alias.gpu_id;
          kfd_ioctl_unmap_memory_from_gpu_args args{};
          args.handle = handle;
          args.n_devices = 1;
          args.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
          return unmap_memory_ioctl(proc, &args);
        };
        for (const auto &alias : *live_aliases.canonical)
          if (const int result = retry_closed_alias(alias); result != 0)
            return result;
        for (const auto &alias : *live_aliases.local)
          if (const int result = retry_closed_alias(alias); result != 0)
            return result;

        const auto local_aliases = proc.gpu_local_alias_snapshot();
        {
          std::lock_guard lock(proc.alloc_mutex_);
          for (const auto &[handle, alloc] : proc.allocations_) {
            (void)handle;
            if (!alloc.local_backing_owner || alloc.backing_owner)
              continue;
            const size_t matching = std::ranges::count_if(*local_aliases, [&](const auto &alias) {
              const auto generation = alloc.admission_generations.find(alias.gpu_id);
              return alias.gpu_va == alloc.gpu_va && alias.extent == alloc.size &&
                     alias.backing_id == alloc.local_backing_owner->backing_id() &&
                     generation != alloc.admission_generations.end() &&
                     generation->second == alias.admission_generation;
            });
            if (matching != alloc.admission_generations.size() ||
                alloc.admission_generations.size() != alloc.mapped_device_ids.size())
              return -EOPNOTSUPP;
          }
        }
        pending->aliases = *proc.gpu_alias_snapshot();
        pending->aliases.insert(pending->aliases.end(), local_aliases->begin(),
                                local_aliases->end());
        pending->keys.clear();
        pending->keys.reserve(pending->aliases.size());
        for (const auto &alias : pending->aliases)
          pending->keys.push_back(GpuAliasIngressGate::key(alias));
        pending->maintenance_prepared = true;
      } catch (const std::bad_alloc &) {
        return -ENOMEM;
      } catch (...) {
        return -EIO;
      }
    }

    const auto retire = [&]() {
      if (fail_next_alias_close_retire_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();

      // Linux kfd_release() only drops the file's kfd_process reference;
      // kfd_process_notifier_release_internal() disables debug when the process mm
      // actually exits. Preserve live sessions across a /dev/kfd close, but reap
      // targets whose pinned identity has exited.
      std::vector<std::pair<pid_t, std::shared_ptr<KfdProcess>>> released;
      {
        std::lock_guard<std::mutex> debug_lock(debug_sessions_mutex_);
        for (auto it = debug_sessions_.begin(); it != debug_sessions_.end();) {
          if (pidfd_is_exited(it->second.debugger_pidfd.get()) == 1) {
            // The debugger is gone and will never ack; release any inferior
            // blocked in the RUNTIME_ENABLE handshake for it.
            cancel_runtime_handshake(it->first);
            // Resolve the debuggee now, while the session's pinned identity still
            // vouches for the pid. After the erase the number alone could name a
            // process that merely reused it.
            released.emplace_back(it->first, find_process_by_client_pid(it->first));
            it = debug_sessions_.erase(it);
          } else if (pidfd_is_exited(it->second.target_pidfd.get()) == 1) {
            if (auto target = find_process_by_client_pid(it->first))
              revoke_target_mem_routing(target->process_id());
            it->second.owned_dbg_fd.reset();
            it->second.target_mem_fd.reset();
            it->second.dbg_fd = -1;
            it->second.enabled = false;
            it->second.target_exited = true;
            ++it;
          } else {
            ++it;
          }
        }
      }
      // Same obligation as the background reaper: erasing the session only stops
      // new debug traffic, so a debugger that died before this close still has to
      // have its inferior's waves resumed, queue gates reopened and target-memory
      // routing revoked. Done after the debug_sessions_mutex_ scope above, because
      // this takes CU wave-state locks and the engine thread takes those first.
      for (auto &[pid, proc_ref] : released)
        release_debuggee_state(pid, proc_ref.get());

      {
        std::lock_guard<std::mutex> ilk(interrupt_mutex_);
        event_dispatch_.erase(process_id);
      }

      // Release queue binding leases and submission BO references before revoking
      // this process's address spaces or unmapping its allocations.
      if (process_id == local_process_id_) {
        std::lock_guard lock(pm4_mutex_);
        for (const auto &[key, queue] : pm4_queues_) {
          (void)key;
          queue.first->unregister_drm_queues(process_id);
        }
        pm4_queues_.clear();
      }

      const bool trace_enabled = vm_trace_enabled();
      size_t leaked_allocations = 0;
      uint64_t leaked_bytes = 0;
      size_t leaked_queues = 0;
      std::vector<uint64_t> leaked_handles;

      {
        std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
        for (auto &[handle, alloc] : proc.allocations_) {
          (void)handle;
          alloc.mapped_device_ids.clear();
          alloc.admission_generations.clear();
          alloc.driver_gpu_mapping = false;
          alloc.driver_admission_generation = 0;
        }
        proc.publish_gpu_grants_locked();
        queue_ids.assign(proc.active_queue_ids_.begin(), proc.active_queue_ids_.end());
        proc.active_queue_ids_.clear();
        proc.queue_snapshot_map_.clear();
        queues.reserve(proc.queue_doorbell_map_.size());
        for (const std::pair<const uint32_t, KfdProcess::QueueDoorbellInfo> &queue_entry :
             proc.queue_doorbell_map_) {
          queues.push_back(queue_entry.second);
        }
        proc.queue_doorbell_map_.clear();

        if (trace_enabled)
          leaked_handles.reserve(proc.allocations_.size());
        for (auto &[handle, alloc] : proc.allocations_) {
          ++leaked_allocations;
          leaked_bytes += alloc.size;
          if (trace_enabled)
            leaked_handles.push_back(handle);
          if (alloc.backing_owner)
            retiring_backings.push_back(alloc.backing_owner);
          if (alloc.local_backing_owner)
            retiring_backings.push_back(alloc.local_backing_owner);
          if (alloc.page_table_mapped_bytes) {
            unmap_from_gpu(proc, alloc.gpu_va, alloc.page_table_mapped_bytes);
            alloc.page_table_mapped_bytes = 0;
          }
          if (alloc.host_ptr && alloc.host_ptr_owned && !alloc.host_ptr_caller_reserved) {
            safe_munmap(alloc.host_ptr, alloc.host_ptr_mapped_bytes);
            alloc.host_ptr = nullptr;
            alloc.host_ptr_owned = false;
            alloc.host_ptr_mapped_bytes = 0;
          }
          if (alloc.memfd >= 0) {
            {
              std::lock_guard<std::mutex> flk(owned_fds_mutex_);
              owned_fds_.erase(alloc.memfd);
            }
            libc_passthrough().close(alloc.memfd);
            alloc.memfd = -1;
          }
        }
        for (const auto &[handle, alloc] : proc.allocations_) {
          (void)handle;
          proc.retire_gpu_range_locked(alloc.gpu_va, alloc.size);
        }
        proc.allocations_.clear();
        proc.publish_gpu_grants_locked();
      }

      for (const KfdProcess::QueueDoorbellInfo &queue : queues) {
        if (queue.gpu_ordinal < gpus_.size() && gpus_[queue.gpu_ordinal].soc)
          (void)gpus_[queue.gpu_ordinal].soc->queue_registry().unregister_queue(
              queue.queue_handle, amdgpu::QueueCloseMode::ForceCancel);
      }

      for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
        GpuDevice &gpu = gpus_[ordinal];
        if (gpu.soc == nullptr)
          continue;
        KfdProcess::PerGpuState &gpu_state = proc.gpu(ordinal);
        if (gpu_state.address_space) {
          (void)gpu.legacy_vm->unregister_address_space(gpu_state.address_space);
          gpu_state.address_space = {};
        }
      }

      // Doorbell mappings live in gpu_state_, not allocations_. Snapshot and clear
      // every client view plus the stable monitor alias under alloc_mutex_, then
      // release the GPU and CPU mappings outside the lock.
      for (size_t ord = 0; ord < proc.gpu_state_.size(); ++ord) {
        auto &gs = proc.gpu_state_[ord];
        int doorbell_memfd;
        void *doorbell_monitor_page;
        size_t doorbell_page_size;
        std::vector<KfdProcess::PerGpuState::DoorbellView> doorbell_views;
        {
          std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
          doorbell_memfd = gs.doorbell_memfd;
          doorbell_monitor_page = gs.doorbell_monitor_page;
          doorbell_page_size = gs.doorbell_page_size;
          doorbell_views = std::move(gs.doorbell_views);
          gs.doorbell_memfd = -1;
          gs.doorbell_monitor_page = nullptr;
          gs.doorbell_page_size = 0;
        }
        update_cp_doorbell_base(static_cast<uint32_t>(ord), process_id, nullptr);
        for (const auto &view : doorbell_views) {
          if (view.gpu_va && doorbell_page_size)
            unmap_from_gpu(proc, view.gpu_va, doorbell_page_size);
          if (view.page != MAP_FAILED && doorbell_page_size)
            safe_munmap(view.page, doorbell_page_size);
        }
        if (doorbell_monitor_page && doorbell_page_size)
          safe_munmap(doorbell_monitor_page, doorbell_page_size);
        if (doorbell_memfd >= 0) {
          {
            std::lock_guard<std::mutex> flk(owned_fds_mutex_);
            owned_fds_.erase(doorbell_memfd);
          }
          libc_passthrough().close(doorbell_memfd);
        }
      }

      leaked_queues = queue_ids.size();
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

      // Guard the dmabuf teardown under alloc_mutex_ for consistency with the
      // import_dmabuf_ioctl/get_dmabuf_info_ioctl accessors: although this process was
      // already erased from the table under process_mutex_ and its last open reference
      // released, an ioctl that took a shared_ptr snapshot before the erase could still
      // be touching imported_dmabufs_ under alloc_mutex_.
      {
        std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
        for (auto &[handle, dmabuf] : proc.imported_dmabufs_) {
          [[maybe_unused]] auto &_ = handle;
          if (dmabuf.fd >= 0)
            libc_passthrough().close(dmabuf.fd);
        }
        proc.imported_dmabufs_.clear();
        // Clear the reverse fd->handle map too, so it stays consistent with
        // imported_dmabufs_ (both are maintained together under alloc_mutex_ by
        // import_dmabuf_ioctl/free_memory_ioctl); its fds were just closed above.
        proc.fd_to_import_handle_.clear();
      }
    };

    const auto writeback_then_retire = [&](const GpuAliasIngressGate::DrainAuthority &authority) {
      const std::optional<bool> done = alias_ingress_gate_->final_writeback_done(authority);
      if (!done)
        throw std::runtime_error("process close lost its drained alias batch");
      if (!*done) {
        if (fail_next_alias_close_writeback_.exchange(false, std::memory_order_acq_rel))
          throw std::bad_alloc();
        for (const auto &alias : pending->aliases) {
          if (!proc.gpu_alias_grant_current(alias))
            throw std::runtime_error("process-close alias grant changed before writeback");
        }
        const auto domains = maintenance_domains_for_aliases(pending->aliases);
        const auto resolve = [this, &authority](uint32_t domain_gpu_id, uint32_t vmid,
                                                uint64_t address, size_t size) {
          // Other pending owners with open grants still need ordinary permit
          // admission. Only this authority's exact closed keys bypass denial.
          return prepare_gpu_alias_writeback_for_vmid(domain_gpu_id, vmid, address, size,
                                                      &authority);
        };
        std::vector<amdgpu::DeviceCacheAliasScope> selected;
        if (std::ranges::none_of(pending->aliases, &KfdProcess::GpuAliasRecord::local) &&
            has_exclusive_canonical_lines(pending->aliases)) {
          selected.reserve(pending->aliases.size());
          for (const auto &alias : pending->aliases) {
            auto scope = scoped_canonical_alias(proc, alias);
            if (!scope) {
              selected.clear();
              break;
            }
            selected.push_back(std::move(*scope));
          }
        }
        auto leases = [&]() {
          if (!selected.empty()) {
            try {
              return amdgpu::DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
                  domains, selected, resolve);
            } catch (const util::Exception &) {
              // Unknown or overlapping dirty tags require the conservative path.
            }
          }
          return amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
              domains, amdgpu::DeviceCacheOperation::WritebackInvalidate, resolve);
        }();
        while (!leases.empty())
          leases.pop_back();
        if (!alias_ingress_gate_->mark_final_writeback_done(authority))
          throw std::runtime_error("process close lost its writeback marker");
      }
      retire();
    };

    if (pending->keys.empty()) {
      try {
        retire();
      } catch (const std::bad_alloc &) {
        return -ENOMEM;
      } catch (...) {
        return -EIO;
      }
    } else {
      std::function<void()> continuation;
      std::atomic<bool> queued{false};
      const auto owner_post = [&](std::function<void()> ready) noexcept {
        continuation = std::move(ready);
        queued.store(true, std::memory_order_release);
        queued.notify_one();
      };
      bool closed = false;
      try {
        closed = alias_ingress_gate_->close_batch_with_drain_authority(pending->keys, owner_post,
                                                                       writeback_then_retire) ||
                 alias_ingress_gate_->retry_closed_batch_with_drain_authority(
                     pending->keys, owner_post, writeback_then_retire);
      } catch (const std::bad_alloc &) {
        return -ENOMEM;
      } catch (...) {
        return -EIO;
      }
      if (!closed)
        return -EAGAIN;
      queued.wait(false, std::memory_order_acquire);
      try {
        continuation();
      } catch (const std::bad_alloc &) {
        return -ENOMEM;
      } catch (...) {
        return -EIO;
      }
      if (!alias_ingress_gate_->reopen_batch(pending->keys))
        return -EIO;
    }

    pending->complete = true;
    {
      std::lock_guard lock(process_mutex_);
      const auto it = pending_closes_.find(process_id);
      if (it != pending_closes_.end() && it->second == pending)
        pending_closes_.erase(it);
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return -EIO;
  }
}

int SimulatedKfd::ioctl(unsigned long request, void *arg) {
  return ioctl(local_process_id_, request, arg);
}

int SimulatedKfd::ioctl(uint32_t process_id, unsigned long request, void *arg, int *target_mem_fd,
                        int target_proc_fd) {
  auto proc = find_process(process_id);
  if (!proc)
    return -ESRCH;
  return dispatch_ioctl(*proc, request, arg, target_mem_fd, target_proc_fd);
}

int SimulatedKfd::dispatch_ioctl(KfdProcess &proc, unsigned long request, void *arg,
                                 int *target_mem_fd, int target_proc_fd) {
  // kfd_ioctl() accepts older CREATE_QUEUE payloads by zero-extending them to
  // the current kernel structure. In particular, older 88-byte payloads omit
  // the sdma_engine_id tail. Copy back only bytes supplied by the caller.
  // https://github.com/torvalds/linux/blob/d24e8ac715de2e16a53c144005b1863660a5fbea/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c
  if (ioctl_without_size(request) == ioctl_without_size(AMDKFD_IOC_CREATE_QUEUE) &&
      request != AMDKFD_IOC_CREATE_QUEUE) {
    kfd_ioctl_create_queue_args args{};
    const size_t bytes = std::min(ioctl_arg_size(request), sizeof(args));
    std::memcpy(&args, arg, bytes);
    const int result =
        dispatch_ioctl(proc, AMDKFD_IOC_CREATE_QUEUE, &args, target_mem_fd, target_proc_fd);
    std::memcpy(arg, &args, bytes);
    return result;
  }
  util::Logger::driver("IOCTL pid=", proc.process_id(), " ", LinuxKfd::ioctl_name(request));

  unsigned long dispatch_request = canonical_ioctl_request(request);

  if (dispatch_request == AMDKFD_IOC_WAIT_EVENTS)
    return wait_events_ioctl(proc, arg);

  // RUNTIME_ENABLE blocks too: when a debugger is attached it waits for that
  // debugger to acknowledge EC_PROCESS_RUNTIME. Holding op_mutex_ across that
  // wait would serialize the ack behind it whenever the debugger and the target
  // share a KfdProcess (self-debug, and daemon clients that reuse one pid), so
  // it takes the lock itself around the state mutation only.
  if (dispatch_request == AMDKFD_IOC_RUNTIME_ENABLE)
    return runtime_enable_ioctl(proc, arg);

  // A noncanonical retirement cannot affect the canonical alias set. Classify
  // it under its process operation lock so it can finish independently of an
  // unrelated canonical close that is waiting for GPU ingress to drain.
  std::unique_lock<std::mutex> op_lock(proc.op_mutex_, std::defer_lock);
  bool ordinary_retirement = false;
  if ((dispatch_request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU ||
       dispatch_request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) &&
      arg != nullptr) {
    if (!op_lock.try_lock()) {
      if (proc.event_state_.is_closing())
        return -ESRCH;
      if (proc.canonical_retirements_pending_.load(std::memory_order_acquire) != 0)
        return -EAGAIN;
      op_lock.lock();
    }
    if (proc.event_state_.is_closing())
      return -ESRCH;
    const uint64_t handle = dispatch_request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU
                                ? static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arg)->handle
                                : static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg)->handle;
    {
      std::lock_guard allocation_lock(proc.alloc_mutex_);
      const auto allocation = proc.allocations_.find(handle);
      ordinary_retirement =
          allocation == proc.allocations_.end() ||
          (!allocation->second.backing_owner && !allocation->second.local_backing_owner);
    }
    if (!ordinary_retirement)
      op_lock.unlock();
  }

  class PendingCanonicalRetirement {
  public:
    PendingCanonicalRetirement(std::atomic<uint32_t> *count, std::atomic<uint32_t> *writers_pending,
                               std::atomic<uint64_t> *release_epoch)
        : count_(count), writers_pending_(writers_pending), release_epoch_(release_epoch) {
      if (count_) {
        count_->fetch_add(1, std::memory_order_acq_rel);
        writers_pending_->fetch_add(1, std::memory_order_acq_rel);
      }
    }
    ~PendingCanonicalRetirement() {
      if (count_) {
        count_->fetch_sub(1, std::memory_order_release);
        writers_pending_->fetch_sub(1, std::memory_order_release);
      }
      if (release_epoch_) {
        release_epoch_->fetch_add(1, std::memory_order_acq_rel);
        release_epoch_->notify_all();
      }
    }
    PendingCanonicalRetirement(const PendingCanonicalRetirement &) = delete;
    PendingCanonicalRetirement &operator=(const PendingCanonicalRetirement &) = delete;

  private:
    std::atomic<uint32_t> *count_;
    std::atomic<uint32_t> *writers_pending_;
    std::atomic<uint64_t> *release_epoch_;
  };
  const bool publication_writer =
      !ordinary_retirement && (dispatch_request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU ||
                               dispatch_request == AMDKFD_IOC_FREE_MEMORY_OF_GPU ||
                               dispatch_request == AMDKFD_IOC_IPC_EXPORT_HANDLE);
  // The claim outlives the publication locks below, so its destructor notifies
  // waiting readers only after the exclusive writer has released the gate.
  PendingCanonicalRetirement pending_retirement(
      publication_writer ? &proc.canonical_retirements_pending_ : nullptr,
      publication_writer ? &alias_publication_writers_pending_ : nullptr,
      publication_writer ? &alias_publication_release_epoch_ : nullptr);

  // Retirement and IPC promotion need one publication writer while their
  // selected backing lines are drained, written back, and republished. Other
  // public KFD operations keep that proof stable with the shared gate.
  std::unique_lock exclusive_publication_lock(alias_publication_mutex_, std::defer_lock);
  std::shared_lock shared_publication_lock(alias_publication_mutex_, std::defer_lock);
  if (ordinary_retirement) {
    // Its operation lock already stabilizes the noncanonical allocation.
  } else if (publication_writer) {
    if (!exclusive_publication_lock.try_lock()) {
      // This call owns one pending count. Another pending retirement of this
      // incarnation may be the close that needs our ingress permit to leave.
      if (proc.canonical_retirements_pending_.load(std::memory_order_acquire) > 1 ||
          alias_ingress_gate_->close_claimed())
        return -EAGAIN;
      exclusive_publication_lock.lock();
    }
  } else {
    auto reader = acquire_alias_publication_reader(proc);
    if (!reader)
      return -EAGAIN;
    shared_publication_lock = std::move(*reader);
  }

  // An IPC import can lose its export while it is validating, and FREE can
  // release the final allocation. Hold either owner past the ioctl op lock.
  std::shared_ptr<const KfdProcess::GpuBacking> retiring_backing;
  if (!op_lock.owns_lock())
    op_lock.lock();
  // A concurrent close() may have snapshotted-then-erased this process and be
  // tearing it down under op_mutex_. ioctl() holds only a shared_ptr (no open
  // reference), so an ioctl that raced close() can end up here AFTER teardown
  // ran (allocations/queues cleared, event_dispatch_ removed, memory
  // unregistered). close() sets the closing flag under op_mutex_ before any
  // teardown, so once we hold op_mutex_, is_closing() means the process is
  // logically gone — reject rather than operate on dismantled state. WAIT_EVENTS
  // is handled above and is intentionally exempt (it must observe the closing
  // signal to wake).
  if (proc.event_state_.is_closing())
    return -ESRCH;
  auto dispatch_one = [&]() -> int {
    switch (dispatch_request) {
    case AMDKFD_IOC_GET_VERSION:
      return get_version_ioctl(arg);
    case AMDKFD_IOC_GET_CLOCK_COUNTERS:
      return get_clock_counters_ioctl(arg);
    case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
      return get_process_apertures_ioctl(arg);
    case AMDKFD_IOC_ACQUIRE_VM:
      return acquire_vm_ioctl(arg);
    case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
      return alloc_memory_ioctl(proc, arg);
    case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
      return free_memory_ioctl(proc, arg, retiring_backing);
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
      return map_memory_ioctl(proc, arg);
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
      return unmap_memory_ioctl(proc, arg);
    case AMDKFD_IOC_CREATE_QUEUE:
      return create_queue_ioctl(proc, arg);
    case AMDKFD_IOC_UPDATE_QUEUE:
      return update_queue_ioctl(proc, arg);
    case AMDKFD_IOC_DESTROY_QUEUE:
      return destroy_queue_ioctl(proc, arg);
    case AMDKFD_IOC_CREATE_EVENT:
      return create_event_ioctl(proc, arg);
    case AMDKFD_IOC_DESTROY_EVENT:
      return destroy_event_ioctl(proc, arg);
    case AMDKFD_IOC_SET_EVENT:
      return set_event_ioctl(proc, arg);
    case AMDKFD_IOC_RESET_EVENT:
      return reset_event_ioctl(proc, arg);
    // WAIT_EVENTS is handled before op_mutex_ above (it blocks on a condition
    // variable and must not hold the per-process op lock), so it never reaches
    // this switch.
    case AMDKFD_IOC_SET_XNACK_MODE:
      return set_xnack_mode_ioctl(arg);
    case AMDKFD_IOC_SET_MEMORY_POLICY:
      return set_memory_policy_ioctl(proc, arg);
    case AMDKFD_IOC_ALLOC_QUEUE_GWS:
      return alloc_queue_gws_ioctl(proc, arg);
    case AMDKFD_IOC_SET_CU_MASK:
      return set_cu_mask_ioctl(proc, arg);
    case AMDKFD_IOC_AVAILABLE_MEMORY:
      return get_available_memory_ioctl(proc, arg);
    // RUNTIME_ENABLE is handled before op_mutex_ above (it blocks on the
    // debugger handshake), so it never reaches this switch.
    case AMDKFD_IOC_DBG_TRAP:
      return debug_trap_ioctl(proc, arg, target_mem_fd, target_proc_fd);
    case AMDKFD_IOC_SET_SCRATCH_BACKING_VA: {
      auto *a = static_cast<kfd_ioctl_set_scratch_backing_va_args *>(arg);
      uint32_t ord = gpu_ordinal(a->gpu_id);
      {
        std::lock_guard<std::mutex> plk(process_mutex_);
        proc.gpu(ord).scratch_backing_va = a->va_addr;
      }
      util::Logger::vm([&](auto &os) {
        os << "SET_SCRATCH_BACKING_VA pid=" << proc.process_id() << " gpu_id=" << a->gpu_id
           << " va=" << std::hex << a->va_addr << std::dec;
      });
      return 0;
    }
    case AMDKFD_IOC_SET_TRAP_HANDLER: {
      auto *a = static_cast<kfd_ioctl_set_trap_handler_args *>(arg);
      uint32_t ord = gpu_ordinal(a->gpu_id);
      {
        // Trap entry resolves these fields on the engine thread while ioctls may
        // update them, so publish and consume them under process_mutex_.
        std::lock_guard<std::mutex> plk(process_mutex_);
        proc.gpu(ord).trap_tba_addr = a->tba_addr;
        proc.gpu(ord).trap_tma_addr = a->tma_addr;
      }
      return 0;
    }
    case AMDKFD_IOC_GET_TILE_CONFIG:
      return get_tile_config_ioctl(arg);
    case AMDKFD_IOC_GET_DMABUF_INFO:
      return get_dmabuf_info_ioctl(proc, arg);
    case AMDKFD_IOC_IMPORT_DMABUF:
      return import_dmabuf_ioctl(proc, arg);
    case AMDKFD_IOC_EXPORT_DMABUF:
      return export_dmabuf_ioctl(proc, arg);
    case AMDKFD_IOC_IPC_EXPORT_HANDLE:
      return ipc_export_handle_ioctl(proc, arg, op_lock);
    case AMDKFD_IOC_IPC_IMPORT_HANDLE:
      return ipc_import_handle_ioctl(proc, arg, retiring_backing);
    case AMDKFD_IOC_SVM:
      // SVM requests carry a trailing attribute array, so libhsakmt sets _IOC_SIZE
      // to the actual buffer size. canonical_ioctl_request() lets this follow the
      // normal switch-dispatch style while still accepting those runtime-sized
      // request values.
      return svm_ioctl(proc, arg);
    default:
      util::Logger::debug_print("rocjitsu: unhandled ioctl 0x", std::hex, request);
      return 0;
    }
  };
  int ret = dispatch_one();
  if (ret != 0) {
    util::Logger::driver([&](auto &os) {
      os << std::format("IOCTL_ERROR pid={} {} ret={}", proc.process_id(), ioctl_name(request),
                        ret);
    });
  }
  return ret;
}

std::shared_ptr<SimulatedKfd::PendingOrdinaryMapping> SimulatedKfd::begin_ordinary_mapping(
    KfdProcess &proc, uint64_t handle, PendingOrdinaryMapping::Kind kind, void *addr, size_t length,
    int prot, int flags, std::span<const KfdProcess::GpuAliasRecord> aliases,
    std::span<const GpuAliasIngressGate::Key> keys, bool &retry) {
  std::lock_guard lock(ordinary_mapping_mutex_);
  const auto identity = std::pair{proc.process_incarnation(), handle};
  if (const auto it = pending_ordinary_mappings_.find(identity);
      it != pending_ordinary_mappings_.end()) {
    const auto &pending = *it->second;
    if (pending.kind != kind || pending.addr != addr || pending.length != length ||
        pending.prot != prot || pending.flags != flags || !std::ranges::equal(pending.keys, keys)) {
      retry = true;
      return {};
    }
    retry = true;
    return it->second;
  }
  if (keys.empty())
    return {};
  auto pending = std::make_shared<PendingOrdinaryMapping>();
  pending->kind = kind;
  pending->addr = addr;
  pending->length = length;
  pending->prot = prot;
  pending->flags = flags;
  pending->aliases.assign(aliases.begin(), aliases.end());
  pending->keys.assign(keys.begin(), keys.end());
  pending_ordinary_mappings_.emplace(identity, pending);
  retry = false;
  return pending;
}

void SimulatedKfd::finish_ordinary_mapping(const KfdProcess &proc, uint64_t handle) {
  std::lock_guard lock(ordinary_mapping_mutex_);
  pending_ordinary_mappings_.erase({proc.process_incarnation(), handle});
}

int SimulatedKfd::run_ordinary_mapping_drain(KfdProcess &proc, uint64_t handle,
                                             const PendingOrdinaryMapping &pending, bool retry,
                                             const std::function<void()> &work) {
  if (pending.keys.empty()) {
    try {
      work();
      return 0;
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    } catch (...) {
      return -EIO;
    }
  }
  if (pending.aliases.empty() || pending.aliases.size() != pending.keys.size())
    return -EIO;

  const auto writeback_then_mutate = [&](const GpuAliasIngressGate::DrainAuthority &authority) {
    const std::optional<bool> done = alias_ingress_gate_->final_writeback_done(authority);
    if (!done)
      throw std::runtime_error("ordinary mapping lost its drained alias batch");
    if (!*done) {
      if (fail_next_alias_mapping_writeback_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();
      {
        std::lock_guard lock(proc.alloc_mutex_);
        const auto it = proc.allocations_.find(handle);
        const bool local = pending.aliases.front().local;
        const auto owner = it == proc.allocations_.end() ? nullptr
                                                         : (local ? it->second.local_backing_owner
                                                                  : it->second.backing_owner);
        if (!owner || (local && it->second.backing_owner) || it->second.retirement_in_progress ||
            pending.aliases.size() != pending.keys.size())
          throw std::runtime_error("ordinary mapping allocation changed before writeback");
        const auto &alloc = it->second;
        if (local && (pending.aliases.size() != alloc.admission_generations.size() ||
                      alloc.admission_generations.size() != alloc.mapped_device_ids.size() ||
                      owner->gpu_id() != alloc.gpu_id || alloc.memfd < 0 ||
                      (alloc.host_ptr && !alloc.host_ptr_owned)))
          throw std::runtime_error("ordinary local mapping lost a grant or retained sink");
        for (size_t i = 0; i < pending.aliases.size(); ++i) {
          const auto &alias = pending.aliases[i];
          const auto generation = alloc.admission_generations.find(alias.gpu_id);
          if (GpuAliasIngressGate::key(alias) != pending.keys[i] || alias.gpu_va != alloc.gpu_va ||
              alias.extent != alloc.size || alias.backing_id != owner->backing_id() ||
              alias.local != local || (local && alias.gpu_id != alloc.gpu_id) ||
              generation == alloc.admission_generations.end() ||
              generation->second != alias.admission_generation ||
              !proc.gpu_alias_grant_current(alias))
            throw std::runtime_error("ordinary mapping alias changed before writeback");
        }
        if (alloc.page_table_mapped_bytes) {
          auto va_transaction = proc.begin_page_table_transaction();
          if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va,
                                                alloc.page_table_mapped_bytes,
                                                {KfdProcess::VaOwnerKind::Kfd, 0}) !=
              KfdProcess::PageTableMutationResult::Applied)
            throw std::runtime_error("ordinary mapping PTE ownership changed before writeback");
          if (local) {
            // A local cache tag resolves through the retained memfd view.
            // Check every installed page before the resolver writes dirty bytes.
            if (!alloc.host_ptr || alloc.page_table_mapped_bytes > alloc.host_ptr_mapped_bytes)
              throw std::runtime_error("ordinary local mapping lost its CPU view");
            const amdgpu::Mtype expected_mtype =
                alloc.ipc_shared && !alloc.ipc_export_retains_local_policy
                    ? amdgpu::Mtype::CC
                    : pte_mtype_for_flags(alloc.flags, alloc.shared);
            std::shared_lock page_lock(proc.page_table_mutex_);
            auto *host_base =
                (alloc.host_view_prot & (PROT_READ | PROT_WRITE)) == (PROT_READ | PROT_WRITE)
                    ? static_cast<uint8_t *>(alloc.host_ptr)
                    : reinterpret_cast<uint8_t *>(owner->data());
            for (uint64_t offset = 0; offset < alloc.page_table_mapped_bytes;) {
              const uint64_t gpu_va = alloc.gpu_va + offset;
              const size_t page_offset = gpu_va & (KfdProcess::kPageSize - 1);
              const size_t chunk = std::min<uint64_t>(KfdProcess::kPageSize - page_offset,
                                                      alloc.page_table_mapped_bytes - offset);
              const auto page = proc.page_table_.find(gpu_va >> KfdProcess::kPageShift);
              if (page == proc.page_table_.end() || page->second.mtype != expected_mtype ||
                  std::ranges::none_of(page->second.host_extents, [&](const auto &extent) {
                    return extent.host_ptr && extent.gpu_page_offset <= page_offset &&
                           page_offset + chunk <=
                               extent.gpu_page_offset + extent.host_backed_bytes &&
                           extent.host_ptr + page_offset - extent.gpu_page_offset ==
                               host_base + offset &&
                           extent.va_owner.kind == KfdProcess::VaOwnerKind::Kfd;
                  }))
                throw std::runtime_error("ordinary local mapping PTE changed before writeback");
              offset += chunk;
            }
          }
        }
      }
      const auto resolve = [this, &authority](uint32_t gpu_id, uint32_t vmid, uint64_t address,
                                              size_t size) {
        return prepare_gpu_alias_writeback_for_vmid(gpu_id, vmid, address, size, &authority);
      };
      std::vector<amdgpu::DeviceCacheAliasScope> selected;
      if (!pending.aliases.front().local && has_exclusive_canonical_lines(pending.aliases)) {
        selected.reserve(pending.aliases.size());
        for (const auto &alias : pending.aliases) {
          auto scope = scoped_canonical_alias(proc, alias);
          if (!scope) {
            selected.clear();
            break;
          }
          selected.push_back(std::move(*scope));
        }
      }
      const auto domains = maintenance_domains_for_aliases(pending.aliases);
      auto leases = [&]() {
        if (!selected.empty()) {
          try {
            return amdgpu::DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
                domains, selected, resolve);
          } catch (const util::Exception &) {
            // Unknown or overlapping dirty tags require the whole-group path.
          }
        }
        return amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, amdgpu::DeviceCacheOperation::WritebackInvalidate, resolve);
      }();
      while (!leases.empty())
        leases.pop_back();
      if (!alias_ingress_gate_->mark_final_writeback_done(authority))
        throw std::runtime_error("ordinary mapping lost its writeback marker");
    }
    if (fail_next_alias_mapping_retire_.exchange(false, std::memory_order_acq_rel))
      throw std::bad_alloc();
    work();
  };

  std::function<void()> continuation;
  std::atomic<bool> queued{false};
  const auto owner_post = [&](std::function<void()> next) noexcept {
    continuation = std::move(next);
    queued.store(true, std::memory_order_release);
    queued.notify_one();
  };
  bool closed = false;
  try {
    closed = retry ? alias_ingress_gate_->retry_closed_batch_with_drain_authority(
                         pending.keys, owner_post, writeback_then_mutate)
                   : alias_ingress_gate_->close_batch_with_drain_authority(pending.keys, owner_post,
                                                                           writeback_then_mutate);
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return -EIO;
  }
  if (!closed)
    return -EAGAIN;
  queued.wait(false, std::memory_order_acquire);
  try {
    continuation();
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return -EIO;
  }
  return alias_ingress_gate_->reopen_batch(pending.keys) ? 0 : -EIO;
}

void *SimulatedKfd::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  return mmap(local_process_id_, addr, length, prot, flags, offset);
}

void *SimulatedKfd::mmap(uint32_t process_id, void *addr, size_t length, int prot, int flags,
                         off_t offset) {
  auto p = find_process(process_id);
  if (!p) {
    errno = ESRCH;
    return MAP_FAILED;
  }
  const bool ordinary = (static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK) == 0;
  const auto needs_alias_drain = [&] {
    if (!ordinary)
      return false;
    std::lock_guard lock(p->alloc_mutex_);
    const auto it = p->allocations_.find(static_cast<uint64_t>(offset) >> 12);
    return it != p->allocations_.end() &&
           (it->second.backing_owner || it->second.local_backing_owner) &&
           !it->second.admission_generations.empty();
  };
  const bool initially_aliased = needs_alias_drain();
  std::optional<PendingAliasPublicationWriter> pending_writer;
  if (ordinary)
    pending_writer.emplace(initially_aliased ? &p->canonical_retirements_pending_ : nullptr,
                           alias_publication_writers_pending_, alias_publication_release_epoch_);
  std::unique_lock exclusive_lock(alias_publication_mutex_, std::defer_lock);
  std::shared_lock shared_lock(alias_publication_mutex_, std::defer_lock);
  if (ordinary) {
    // A same-process alias retirement may need an ingress permit held by
    // this caller. Other processes wait for the writer to finish.
    if (!exclusive_lock.try_lock()) {
      if (initially_aliased &&
          (p->canonical_retirements_pending_.load(std::memory_order_acquire) > 1 ||
           alias_ingress_gate_->close_claimed_by_process(p->process_incarnation()))) {
        errno = EAGAIN;
        return MAP_FAILED;
      }
      exclusive_lock.lock();
    }
    if (!initially_aliased && needs_alias_drain()) {
      // An IPC export/MAP may have published an alias while this writer was
      // waiting. Arm the reentrant-reader refusal while retaining exclusivity.
      pending_writer->arm(&p->canonical_retirements_pending_);
    }
  } else {
    auto reader = acquire_alias_publication_reader(*p);
    if (!reader) {
      errno = EAGAIN;
      return MAP_FAILED;
    }
    shared_lock = std::move(*reader);
  }
  if (daemon_mode_)
    return dispatch_mmap(*p, nullptr, length, prot, flags & ~MAP_FIXED, offset);
  return dispatch_mmap(*p, addr, length, prot, flags, offset);
}

void *SimulatedKfd::dispatch_mmap(KfdProcess &proc, void *addr, size_t length, int prot, int flags,
                                  off_t offset) {
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;
  util::Logger::vm("SimulatedKfd::mmap type=0x", std::hex, type, " offset=0x", offset,
                   " length=", std::dec, length, " addr=", addr);

  std::unique_lock<std::mutex> op_lock(proc.op_mutex_, std::defer_lock);
  const int op_result = acquire_mmap_op(op_lock, [&] {
    if (proc.event_state_.is_closing())
      return -ENODEV;
    if (type != 0)
      return 0;
    std::unique_lock<std::mutex> lock(proc.alloc_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      return 0;
    const auto it = proc.allocations_.find(static_cast<uint64_t>(offset) >> 12);
    if (it == proc.allocations_.end())
      return 0;
    return it->second.retirement_in_progress ||
                   allocation_has_closed_alias(proc, it->second, *alias_ingress_gate_)
               ? -EAGAIN
               : 0;
  });
  if (op_result != 0) {
    errno = -op_result;
    return MAP_FAILED;
  }
  if (proc.event_state_.is_closing()) {
    errno = ENODEV;
    return MAP_FAILED;
  }
  std::optional<ScopedPrivateGttCopySourceMutation> copy_source_mutation;
  if (type == 0)
    copy_source_mutation.emplace(*this, proc, static_cast<uint64_t>(offset) >> 12);
  if (flags & MAP_FIXED) {
    if (length == 0 || length > std::numeric_limits<size_t>::max() - 0xFFF) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    const size_t fixed_size = (length + 0xFFF) & ~size_t{0xFFF};
    const uint64_t handle = static_cast<uint64_t>(offset) >> 12;
    const uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      for (const auto &[other_handle, view] : proc.allocations_) {
        const size_t view_size = view.host_ptr_owned ? view.host_ptr_mapped_bytes : view.size;
        if (view.host_ptr && ranges_overlap(addr, fixed_size, view.host_ptr, view_size) &&
            (type != 0 || other_handle != handle || addr != view.host_ptr ||
             fixed_size != view_size)) {
          errno = EINVAL;
          return MAP_FAILED;
        }
      }
      for (size_t ord = 0; ord < proc.gpu_state_.size(); ++ord) {
        const auto &gs = proc.gpu_state_[ord];
        if (ranges_overlap(addr, fixed_size, gs.doorbell_monitor_page, gs.doorbell_page_size)) {
          errno = EINVAL;
          return MAP_FAILED;
        }
        for (const auto &view : gs.doorbell_views)
          if (ranges_overlap(addr, fixed_size, view.page, gs.doorbell_page_size) &&
              (type != KFD_MMAP_TYPE_DOORBELL || ord >= gpus_.size() ||
               encoded_gpu != gpus_[ord].gpu_id || addr != view.page ||
               fixed_size != gs.doorbell_page_size)) {
            errno = EINVAL;
            return MAP_FAILED;
          }
      }
    }
    if (proc.event_state_.overlaps_page(addr, fixed_size)) {
      errno = EINVAL;
      return MAP_FAILED;
    }
  }

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    uint32_t db_gpu_id = static_cast<uint32_t>(encoded_gpu);
    if (!find_gpu(db_gpu_id)) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    uint32_t ord = gpu_ordinal(db_gpu_id);

    // Serialize the canonical backing, both mappings, GPU page-table publication,
    // and CP base update against process teardown and another doorbell mmap.
    int doorbell_fd = -1;
    int source_doorbell_fd = -1;
    void *monitor_ptr = nullptr;
    size_t published_size = 0;
    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      auto &gs = proc.gpu(ord);
      doorbell_fd = gs.doorbell_memfd;
      monitor_ptr = gs.doorbell_monitor_page;
      published_size = gs.doorbell_page_size;
      if (doorbell_fd < 0) {
        for (auto &[handle, alloc] : proc.allocations_) {
          if ((alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) && alloc.gpu_id == db_gpu_id) {
            source_doorbell_fd = alloc.memfd;
            break;
          }
        }
      }
    }

    UniqueDriverFd new_doorbell_fd;
    if (doorbell_fd < 0) {
      // Retain a private descriptor even when a KFD doorbell allocation supplied
      // the source. The allocation can be freed independently; this duplicate keeps
      // the canonical backing valid until the per-process doorbell state is torn down.
      UniqueDriverFd created_source;
      if (source_doorbell_fd < 0) {
        // Local-mode ROCr can request a doorbell mmap without first allocating a
        // KFD doorbell object. Give that path the same persistent shared backing.
        created_source.reset(memfd_create("rocjitsu_doorbell", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (!created_source)
          return MAP_FAILED;
        source_doorbell_fd = created_source.get();
      }
      new_doorbell_fd.reset(safe_fcntl(source_doorbell_fd, F_DUPFD_CLOEXEC, kBackingFdMin));
      if (!new_doorbell_fd && created_source)
        new_doorbell_fd = std::move(created_source);
      if (!new_doorbell_fd)
        return MAP_FAILED;
      doorbell_fd = new_doorbell_fd.get();
    }

    // A process/GPU owns exactly one monitor view. Doorbell mappings have a fixed
    // KFD page size; rejecting a conflicting remap avoids invalidating live queue
    // offsets or changing the address polled by the CP.
    if (monitor_ptr && published_size != length) {
      new_doorbell_fd.reset();
      errno = EINVAL;
      return MAP_FAILED;
    }
    // MAP_FIXED replaces every existing mapping in its target range. Never let a
    // client-selected address replace the CP's private alias; that would leave
    // live queues polling an invalid address.
    if ((flags & MAP_FIXED) && ranges_overlap(addr, length, monitor_ptr, published_size)) {
      new_doorbell_fd.reset();
      errno = EINVAL;
      return MAP_FAILED;
    }

    off_t cur_size = 0;
    {
      struct stat st {};
      if (safe_fstat(doorbell_fd, &st) == 0)
        cur_size = st.st_size;
    }
    if (static_cast<off_t>(length) > cur_size &&
        ftruncate(doorbell_fd, static_cast<off_t>(length)) != 0) {
      int saved_errno = errno;
      new_doorbell_fd.reset();
      errno = saved_errno;
      return MAP_FAILED;
    }

    int db_mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      db_mflags |= MAP_FIXED;

    UniqueMapping new_monitor_mapping;
    if (!monitor_ptr) {
      // Initialize doorbell backing to 0xFF via the CP's permanent mapping so
      // every slot matches the HwQueue::last_doorbell sentinel. This also avoids
      // SIGBUS on the final MAP_SHARED mmap on Linux 6.17+ where
      // shmem large folio allocation can fail during a bulk memset on a
      // freshly-mapped region. Writing through a separate PROT_WRITE
      // mapping forces page allocation before the client mapping is published. Keeping this
      // alias gives the simulated CP a stable device-side view even while the
      // runtime is creating or replacing its own mapping of the same memfd.
      void *new_monitor = MAP_FAILED;
      void *forced_monitor_addr =
          next_doorbell_monitor_mmap_addr_.exchange(nullptr, std::memory_order_acq_rel);
      if (fail_next_doorbell_monitor_mmap_.exchange(false, std::memory_order_acq_rel)) {
        errno = ENOMEM;
      } else {
        int monitor_flags = MAP_SHARED;
        if (forced_monitor_addr)
          monitor_flags |= MAP_FIXED_NOREPLACE;
        new_monitor = safe_mmap(forced_monitor_addr, length, PROT_READ | PROT_WRITE, monitor_flags,
                                doorbell_fd, 0);
      }
      new_monitor_mapping.reset(new_monitor, length);
      if (!new_monitor_mapping) {
        int saved_errno = errno;
        new_doorbell_fd.reset();
        errno = saved_errno;
        return MAP_FAILED;
      }

      // Allocate the stable alias before the destructive client MAP_FIXED. If
      // the kernel placed an alias in the requested client range, retain it as
      // a reservation while establishing another. Once an alias lands fully
      // outside the target, the reservations can be released and the final
      // fixed mapping cannot replace the CP's private alias.
      std::vector<UniqueMapping> monitor_reservations;
      while ((flags & MAP_FIXED) &&
             ranges_overlap(addr, length, new_monitor_mapping.get(), length)) {
        monitor_reservations.push_back(std::move(new_monitor_mapping));
        new_monitor_mapping.reset(
            safe_mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, doorbell_fd, 0), length);
        if (!new_monitor_mapping) {
          int saved_errno = errno;
          new_monitor_mapping.reset();
          monitor_reservations.clear();
          new_doorbell_fd.reset();
          errno = saved_errno;
          return MAP_FAILED;
        }
      }
      monitor_ptr = new_monitor_mapping.get();
      std::memset(monitor_ptr, 0xFF, length);
    }

    void *ptr = MAP_FAILED;
    int map_error = 0;
    {
      // Keep the published view and its PTE synchronized for independent
      // mprotect/munmap classification. The established writer order is
      // alloc_mutex_ -> page-table request lock.
      std::unique_lock<std::mutex> alloc_lock(proc.alloc_mutex_);
      auto &gs = proc.gpu(ord);
      try {
        // Bookkeeping and private-fd registration must be ready before a PTE
        // can point into the new CPU view. Neither commit step then allocates.
        gs.doorbell_views.reserve(gs.doorbell_views.size() + 1);
      } catch (const std::bad_alloc &) {
        errno = ENOMEM;
        return MAP_FAILED;
      }
      auto va_transaction = proc.begin_page_table_transaction();
      if ((flags & MAP_FIXED) &&
          proc.has_foreign_resident_overlap(va_transaction, reinterpret_cast<uint64_t>(addr),
                                            length, {KfdProcess::VaOwnerKind::Kfd, 0}) !=
              KfdProcess::PageTableMutationResult::Applied) {
        errno = EINVAL;
        return MAP_FAILED;
      }

      bool fd_registered = false;
      if (new_doorbell_fd) {
        try {
          std::lock_guard<std::mutex> flk(owned_fds_mutex_);
          owned_fds_.insert(new_doorbell_fd.get());
          fd_registered = true;
        } catch (const std::bad_alloc &) {
          errno = ENOMEM;
          return MAP_FAILED;
        }
      }
      const auto discard_new_fd = [&] {
        if (!fd_registered)
          return;
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.erase(new_doorbell_fd.get());
        fd_registered = false;
      };

      UniqueMapping client_mapping(
          safe_mmap(addr, length, PROT_READ | PROT_WRITE, db_mflags, doorbell_fd, 0), length);
      if (!client_mapping) {
        const int mmap_errno = errno;
        discard_new_fd();
        errno = mmap_errno;
        return MAP_FAILED;
      }
      ptr = client_mapping.get();
      if (ranges_overlap(ptr, length, monitor_ptr, length)) {
        discard_new_fd();
        errno = EINVAL;
        return MAP_FAILED;
      }

      const auto publish_view = [&] {
        if (new_doorbell_fd) {
          assert(gs.doorbell_memfd < 0);
          gs.doorbell_memfd = new_doorbell_fd.release();
          fd_registered = false;
        } else {
          assert(gs.doorbell_memfd == doorbell_fd);
        }
        assert(!gs.doorbell_monitor_page || gs.doorbell_monitor_page == monitor_ptr);
        if (std::ranges::none_of(gs.doorbell_views,
                                 [ptr](const auto &view) { return view.page == ptr; }))
          gs.doorbell_views.push_back({.page = ptr, .gpu_va = reinterpret_cast<uint64_t>(ptr)});
        gs.doorbell_monitor_page = monitor_ptr;
        gs.doorbell_page_size = length;
        static_cast<void>(new_monitor_mapping.release());
        static_cast<void>(client_mapping.release());
      };

      try {
        if (map_to_gpu(proc, reinterpret_cast<uint64_t>(ptr), ptr, length, amdgpu::Mtype::UC,
                       KfdProcess::HostExtentOwner::Application,
                       &va_transaction) != KfdProcess::PageTableMutationResult::Applied) {
          discard_new_fd();
          errno = EINVAL;
          return MAP_FAILED;
        }
      } catch (const std::bad_alloc &) {
        // map_pages may have installed a prefix before an allocation failed.
        // Retain its CPU backing and the canonical memfd for close to clean up.
        map_error = ENOMEM;
      } catch (...) {
        map_error = EIO;
      }
      publish_view();
    }

    update_cp_doorbell_base(ord, proc.process_id(), monitor_ptr);
    if (map_error != 0) {
      errno = map_error;
      return MAP_FAILED;
    }
    return ptr;
  }

  if (type == KFD_MMAP_TYPE_EVENTS) {
    // Keep page adoption ordered with the containing-unmap overlap check and teardown.
    // acquire_mmap_op above already holds op_mutex_ through this branch.
    if (proc.event_state_.is_closing()) {
      errno = ENODEV;
      return MAP_FAILED;
    }
    // Create-or-get the backing as ONE locked operation. Two concurrent event-page
    // mmaps would otherwise both observe no backing, each build one, and hand
    // different fds to different callers -- leaving one polling an object that
    // never receives event updates -- besides racing the field itself.
    const int events_fd = proc.event_state_.ensure_backing(length, [&](size_t size) -> int {
      auto raw_events_fd = memfd_create("rocjitsu_events", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (raw_events_fd < 0)
        return -1;
      int backing = safe_fcntl(raw_events_fd, F_DUPFD_CLOEXEC, kBackingFdMin);
      if (backing < 0)
        backing = raw_events_fd;
      else
        libc_passthrough().close(raw_events_fd);
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(backing);
      }
      if (ftruncate(backing, static_cast<off_t>(size)) != 0) {
        const int ftruncate_errno = errno; // preserve across close() below
        {
          std::lock_guard<std::mutex> lk(owned_fds_mutex_);
          owned_fds_.erase(backing);
        }
        libc_passthrough().close(backing);
        errno = ftruncate_errno;
        return -1;
      }
      fallocate(backing, 0, 0, static_cast<off_t>(size));
      {
        auto *init_ptr =
            static_cast<uint8_t *>(safe_mmap(nullptr, size, PROT_WRITE, MAP_SHARED, backing, 0));
        if (init_ptr != MAP_FAILED) {
          libc_passthrough().madvise(init_ptr, size, MADV_POPULATE_WRITE);
          std::memset(init_ptr, 0xFF, size);
          safe_munmap(init_ptr, size);
        }
      }
      safe_fcntl(backing, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
      return backing;
    });
    if (events_fd < 0)
      return MAP_FAILED;
    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    void *ptr = safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, events_fd, 0);
    if (ptr != MAP_FAILED)
      proc.event_state_.adopt_page(ptr, length);
    return ptr;
  }

  // Ordinary allocation mappings share the ioctl/close operation order. The
  // alias gate then excludes entered CU/SDMA effects before a PTE or its CPU
  // view can be replaced. Never carry alloc_mutex_ into the drain wait.
  const uint64_t handle = static_cast<uint64_t>(offset) >> 12;
  std::vector<GpuAliasIngressGate::Key> keys;
  std::vector<KfdProcess::GpuAliasRecord> aliases;
  {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    if (proc.event_state_.is_closing()) {
      errno = ENODEV;
      return MAP_FAILED;
    }
    const auto it = proc.allocations_.find(handle);
    if (it == proc.allocations_.end()) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    const auto &alloc = it->second;
    if (alloc.retirement_in_progress) {
      errno = EAGAIN;
      return MAP_FAILED;
    }
    // Daemon allocations already own their CPU mapping and PTE. This reply
    // does not replace either view, including before IPC export promotion.
    if (daemon_mode_ && alloc.memfd >= 0 && alloc.host_ptr != nullptr) {
      if (allocation_has_closed_alias(proc, alloc, *alias_ingress_gate_)) {
        errno = EAGAIN;
        return MAP_FAILED;
      }
      return alloc.host_ptr;
    }
    // An admitted cacheable allocation can still hold dirty lines after its
    // last grant is retired. Preserve the old view until it has a local sink.
    if (needs_prepared_local_sink(alloc)) {
      errno = EOPNOTSUPP;
      return MAP_FAILED;
    }
    if (alloc.local_backing_owner && !alloc.backing_owner) {
      // Once a local cacheable grant has existed, only replacement of the
      // exact driver-owned view can preserve its retained memfd identity.
      if (alloc.memfd < 0 || length > alloc.size ||
          (alloc.host_ptr && (!alloc.host_ptr_owned || !(flags & MAP_FIXED) ||
                              addr != alloc.host_ptr || length != alloc.host_ptr_mapped_bytes))) {
        errno = EOPNOTSUPP;
        return MAP_FAILED;
      }
    }
    try {
      keys = allocation_alias_keys(proc, alloc, &aliases);
      if (alloc.local_backing_owner && !alloc.backing_owner &&
          (aliases.size() != alloc.admission_generations.size() ||
           alloc.admission_generations.size() != alloc.mapped_device_ids.size())) {
        errno = EOPNOTSUPP;
        return MAP_FAILED;
      }
    } catch (const std::bad_alloc &) {
      errno = ENOMEM;
      return MAP_FAILED;
    }
  }

  bool retry = false;
  std::shared_ptr<PendingOrdinaryMapping> pending;
  try {
    pending = begin_ordinary_mapping(proc, handle, PendingOrdinaryMapping::Kind::Mmap, addr, length,
                                     prot, flags, aliases, keys, retry);
  } catch (const std::bad_alloc &) {
    errno = ENOMEM;
    return MAP_FAILED;
  }
  if ((retry && !pending) || (!keys.empty() && !pending) ||
      (!retry && std::ranges::any_of(
                     keys, [&](const auto &key) { return alias_ingress_gate_->closed(key); }))) {
    if (pending && !retry)
      finish_ordinary_mapping(proc, handle);
    errno = EAGAIN;
    return MAP_FAILED;
  }

  auto map_allocation = [&]() -> void * {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    auto &alloc = proc.allocations_.at(handle);

    if (length == 0 || length > std::numeric_limits<size_t>::max() - 0xFFF) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    const size_t host_mapped_bytes = (length + 0xFFF) & ~size_t{0xFFF};
    const size_t mapped_bytes = std::min(alloc.size, host_mapped_bytes);
    auto va_transaction = proc.begin_page_table_transaction();
    if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va, mapped_bytes,
                                          {KfdProcess::VaOwnerKind::Kfd, 0}) !=
        KfdProcess::PageTableMutationResult::Applied) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    const bool previously_unstable = alloc.private_gtt_copy_source_unstable;

    void *host_ptr = pending && pending->host_mapping_prepared ? alloc.host_ptr : nullptr;
    bool host_ptr_owned = pending && pending->host_mapping_prepared ? alloc.host_ptr_owned : true;

    if (!host_ptr && alloc.memfd >= 0) {
      if (length > alloc.size) {
        if (ftruncate(alloc.memfd, static_cast<off_t>(length)) != 0) {
          errno = ENOMEM;
          return MAP_FAILED;
        }
      }
      // A failed size check or ftruncate has not replaced the CPU view or PTE.
      if (alloc.private_gtt_copy_source_gate)
        alloc.private_gtt_copy_source_unstable = true;
      if (alloc.user_va && (flags & MAP_FIXED) && addr != nullptr) {
        auto prot_rc = safe_mprotect(addr, length, PROT_READ | PROT_WRITE);
        if (prot_rc == 0) {
          constexpr size_t page_size = 4096;
          size_t num_pages = (length + page_size - 1) / page_size;
          std::vector<uint8_t> page_resident(num_pages);
          auto mc_rc = mincore(addr, length, page_resident.data());

          auto *temp_mapping = static_cast<uint8_t *>(
              safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, alloc.memfd, 0));
          if (temp_mapping != MAP_FAILED) {
            if (mc_rc == 0) {
              auto *source = static_cast<uint8_t *>(addr);
              for (size_t i = 0; i < num_pages; ++i) {
                if (page_resident[i] & 1) {
                  size_t off = i * page_size;
                  size_t copy_len = std::min(page_size, length - off);
                  std::memcpy(temp_mapping + off, source + off, copy_len);
                }
              }
            }
            safe_munmap(temp_mapping, length);
          }
        }
      }

      int mflags = MAP_SHARED;
      if (flags & MAP_FIXED)
        mflags |= MAP_FIXED;
      host_ptr = safe_mmap(addr, length, prot, mflags, alloc.memfd, 0);
      if (host_ptr == MAP_FAILED) {
        if (!(flags & MAP_FIXED))
          alloc.private_gtt_copy_source_unstable = previously_unstable;
        return MAP_FAILED;
      }
    } else if (!host_ptr) {
      if (alloc.private_gtt_copy_source_gate)
        alloc.private_gtt_copy_source_unstable = true;
      bool reuse_pages = false;
      if (alloc.user_va && (flags & MAP_FIXED) && addr != nullptr) {
        auto rc = safe_mprotect(addr, length, PROT_READ | PROT_WRITE);
        reuse_pages = (rc == 0);
      }
      if (reuse_pages) {
        host_ptr = addr;
        host_ptr_owned = false;
      } else {
        int mflags = MAP_ANONYMOUS;
        mflags |= (flags & MAP_SHARED) ? MAP_SHARED : MAP_PRIVATE;
        if (flags & MAP_FIXED)
          mflags |= MAP_FIXED;
        host_ptr = safe_mmap(addr, length, prot, mflags, -1, 0);
        if (host_ptr == MAP_FAILED) {
          if (!(flags & MAP_FIXED))
            alloc.private_gtt_copy_source_unstable = previously_unstable;
          return MAP_FAILED;
        }
      }
    }
    if (alloc.private_gtt_copy_source_gate)
      alloc.private_gtt_copy_source_unstable = true;

    alloc.host_ptr = host_ptr;
    alloc.host_ptr_owned = host_ptr_owned;
    alloc.host_ptr_caller_reserved = (flags & MAP_FIXED) && addr != nullptr;
    alloc.host_ptr_mapped_bytes = host_ptr_owned ? host_mapped_bytes : 0;
    alloc.host_view_prot = prot;
    alloc.host_view_generation = proc.next_host_view_generation_locked();
    if (pending)
      pending->host_mapping_prepared = true;
    if (pending &&
        fail_next_alias_mapping_after_host_map_.exchange(false, std::memory_order_acq_rel))
      throw std::bad_alloc();

    util::Logger::vm([&](auto &os) {
      os << std::format("mmap: gpu_va={:#x} host_ptr={:#x} size={} flags={:#x}"
                        " MAP_FIXED={} user_va={} memfd={}",
                        alloc.gpu_va, reinterpret_cast<uintptr_t>(host_ptr), length, alloc.flags,
                        bool(flags & MAP_FIXED), alloc.user_va, alloc.memfd);
    });
    // mmap and KFD BOs cover whole pages, including the last partial page.
    const amdgpu::Mtype mtype =
        alloc.cacheable_peer_for_testing
            ? pte_mtype_for_flags(alloc.flags, alloc.shared)
            : (alloc.backing_owner ? amdgpu::Mtype::UC
                                   : (alloc.ipc_shared && !alloc.ipc_export_retains_local_policy
                                          ? amdgpu::Mtype::CC
                                          : pte_mtype_for_flags(alloc.flags, alloc.shared)));
    // The caller selects mmap protection and can later change it. Ownership of
    // munmap does not make this backing safe to treat as permanently writable.
    // Keep the full span available to FREE/close if map_pages throws after
    // installing a prefix. The failed gate batch remains closed in that case.
    alloc.page_table_mapped_bytes = mapped_bytes;
    const auto gpu_backing = alloc.backing_owner ? alloc.backing_owner : alloc.local_backing_owner;
    const bool private_gpu_view =
        gpu_backing && (prot & (PROT_READ | PROT_WRITE)) != (PROT_READ | PROT_WRITE);
    void *gpu_ptr = private_gpu_view ? gpu_backing->data() : host_ptr;
    if (map_to_gpu(proc, alloc.gpu_va, gpu_ptr, mapped_bytes, mtype,
                   private_gpu_view ? KfdProcess::HostExtentOwner::Driver
                                    : KfdProcess::HostExtentOwner::Application,
                   &va_transaction) != KfdProcess::PageTableMutationResult::Applied) {
      if (pending)
        throw std::runtime_error("ordinary canonical mmap could not replace GPU PTEs");
      if (host_ptr_owned && !alloc.host_ptr_caller_reserved)
        safe_munmap(host_ptr, host_mapped_bytes);
      alloc.host_ptr = nullptr;
      alloc.host_ptr_owned = false;
      alloc.host_ptr_caller_reserved = false;
      alloc.host_ptr_mapped_bytes = 0;
      alloc.host_view_prot = 0;
      alloc.page_table_mapped_bytes = 0;
      alloc.host_view_generation = proc.next_host_view_generation_locked();
      errno = EINVAL;
      return MAP_FAILED;
    }

    // A complete driver-owned memfd remap has replaced every source PTE and
    // the whole tracked host view, so it also repairs a prior partial attempt.
    const bool full_private_view_repaired =
        alloc.private_gtt_copy_source_gate && alloc.memfd >= 0 && host_ptr_owned &&
        host_mapped_bytes == alloc.size && mapped_bytes == alloc.size;
    alloc.private_gtt_copy_source_unstable = previously_unstable && !full_private_view_repaired;
    return host_ptr;
  };
  void *mapped = MAP_FAILED;
  PendingOrdinaryMapping ordinary{};
  const int drain_result = run_ordinary_mapping_drain(proc, handle, pending ? *pending : ordinary,
                                                      retry, [&] { mapped = map_allocation(); });
  if (drain_result != 0) {
    if (pending && !std::ranges::any_of(
                       keys, [&](const auto &key) { return alias_ingress_gate_->closed(key); }))
      finish_ordinary_mapping(proc, handle);
    errno = -drain_result;
    return MAP_FAILED;
  }
  if (pending)
    finish_ordinary_mapping(proc, handle);
  return mapped;
}

int SimulatedKfd::munmap(void *addr, size_t length) {
  return munmap(local_process_id_, addr, length);
}

int SimulatedKfd::munmap(uint32_t process_id, void *addr, size_t length) {
  auto p = find_process(process_id);
  if (!p)
    return -ESRCH;
  const auto needs_alias_drain = [&] {
    std::lock_guard lock(p->alloc_mutex_);
    const auto it = std::ranges::find_if(p->allocations_, [addr](const auto &entry) {
      return addr != nullptr && entry.second.host_ptr == addr;
    });
    return it != p->allocations_.end() &&
           (it->second.backing_owner || it->second.local_backing_owner) &&
           !it->second.admission_generations.empty();
  };
  const bool initially_aliased = needs_alias_drain();
  PendingAliasPublicationWriter pending_writer(
      initially_aliased ? &p->canonical_retirements_pending_ : nullptr,
      alias_publication_writers_pending_, alias_publication_release_epoch_);
  std::unique_lock publication_lock(alias_publication_mutex_, std::defer_lock);
  if (!publication_lock.try_lock()) {
    if (initially_aliased &&
        (p->canonical_retirements_pending_.load(std::memory_order_acquire) > 1 ||
         alias_ingress_gate_->close_claimed_by_process(p->process_incarnation())))
      return -EAGAIN;
    publication_lock.lock();
  }
  if (!initially_aliased && needs_alias_drain()) {
    pending_writer.arm(&p->canonical_retirements_pending_);
  }
  return dispatch_munmap(*p, addr, length);
}

int SimulatedKfd::dispatch_munmap(KfdProcess &proc, void *addr, size_t length) {
  std::unique_lock<std::mutex> op_lock(proc.op_mutex_, std::defer_lock);
  const int op_result = acquire_mmap_op(op_lock, [&] {
    std::unique_lock<std::mutex> lock(proc.alloc_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      return 0;
    // close() publishes this flag under alloc_mutex_ before clearing tracked
    // views. A stale process snapshot must not reach the libc fallback.
    if (proc.event_state_.is_closing())
      return -ENODEV;
    const auto allocation = std::ranges::find_if(proc.allocations_, [addr](const auto &entry) {
      return addr != nullptr && entry.second.host_ptr == addr;
    });
    if (allocation != proc.allocations_.end() &&
        (allocation->second.retirement_in_progress ||
         allocation_has_closed_alias(proc, allocation->second, *alias_ingress_gate_)))
      return -EAGAIN;
    // An in-flight MAP_FIXED mmap can publish a new tracked view at addr.
    // Wait for op_mutex_ before deciding whether munmap belongs to libc.
    return 0;
  });
  if (op_result != 0)
    return op_result;
  {
    // Client views can disappear while KFD and its queues remain live. Keep
    // the private polling alias and canonical backing until process teardown,
    // just as when MAP_FIXED replaces a client view.
    std::lock_guard<std::mutex> alloc_lock(proc.alloc_mutex_);
    for (const auto &gs : proc.gpu_state_) {
      if (ranges_overlap(addr, length, gs.doorbell_monitor_page, gs.doorbell_page_size)) {
        errno = EPERM;
        return -1;
      }
    }
    bool contains_view = false;
    for (const auto &gs : proc.gpu_state_) {
      for (const auto &view : gs.doorbell_views) {
        if (!ranges_overlap(addr, length, view.page, gs.doorbell_page_size))
          continue;
        if (!range_contains(addr, length, view.page, gs.doorbell_page_size)) {
          errno = EINVAL;
          return -1;
        }
        contains_view = true;
      }
    }
    if (contains_view) {
      // This path only retires doorbell views. Reject ranges that would also
      // remove event or allocation mappings while leaving their owners live.
      // munmap rounds its length up to a host page, including any userptr
      // allocation beginning later in the final partially covered page.
      const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
      const size_t remainder = length % page_size;
      const size_t padding = remainder ? page_size - remainder : 0;
      if (padding > SIZE_MAX - length) {
        errno = EINVAL;
        return -1;
      }
      const size_t unmapped_bytes = length + padding;
      if (proc.event_state_.overlaps_page(addr, unmapped_bytes)) {
        errno = EINVAL;
        return -1;
      }
      for (const auto &[handle, alloc] : proc.allocations_) {
        if (alloc.host_ptr && ranges_overlap(addr, unmapped_bytes, alloc.host_ptr, alloc.size)) {
          errno = EINVAL;
          return -1;
        }
      }
      for (const auto &gs : proc.gpu_state_)
        for (const auto &view : gs.doorbell_views)
          if (range_contains(addr, length, view.page, gs.doorbell_page_size))
            unmap_from_gpu(proc, view.gpu_va, gs.doorbell_page_size);
      if (safe_munmap(addr, length) != 0) {
        const int saved_errno = errno;
        for (const auto &gs : proc.gpu_state_)
          for (const auto &view : gs.doorbell_views)
            if (range_contains(addr, length, view.page, gs.doorbell_page_size))
              map_to_gpu(proc, view.gpu_va, view.page, gs.doorbell_page_size, amdgpu::Mtype::UC);
        errno = saved_errno;
        return -1;
      }
      for (auto &gs : proc.gpu_state_) {
        std::erase_if(gs.doorbell_views, [&](const auto &view) {
          return range_contains(addr, length, view.page, gs.doorbell_page_size);
        });
      }
      return 0;
    }
  }
  // release_page() clears page_/page_size_ under EventState::mutex_, the same lock
  // the CP interrupt thread holds when reading them in signal_interrupt, so the
  // munmap below cannot race a concurrent signal writing into the mapping.
  if (proc.event_state_.release_page(addr)) {
    safe_munmap(addr, length);
    return 0;
  }
  uint64_t selected_handle = 0;
  std::vector<GpuAliasIngressGate::Key> keys;
  std::vector<KfdProcess::GpuAliasRecord> aliases;
  {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    if (proc.event_state_.is_closing())
      return -ENODEV;
    const auto it = std::ranges::find_if(proc.allocations_, [addr](const auto &entry) {
      return addr != nullptr && entry.second.host_ptr == addr;
    });
    if (it == proc.allocations_.end())
      return -ENOENT;
    const auto &alloc = it->second;
    if (alloc.retirement_in_progress)
      return -EAGAIN;
    if (alloc.host_ptr_owned &&
        (length == 0 || length > std::numeric_limits<size_t>::max() - 0xFFF ||
         ((length + 0xFFF) & ~size_t{0xFFF}) != alloc.host_ptr_mapped_bytes))
      return -EINVAL;
    if (alloc.local_backing_owner && !alloc.backing_owner && length != alloc.host_ptr_mapped_bytes)
      return -EINVAL;
    selected_handle = it->first;
    // Keep the old CPU view for cache lines retained from this grant or an
    // earlier one, including after the last UNMAP removed its PTE.
    if (needs_prepared_local_sink(alloc))
      return -EOPNOTSUPP;
    if (alloc.local_backing_owner && !alloc.backing_owner &&
        (alloc.memfd < 0 || !alloc.host_ptr_owned))
      return -EOPNOTSUPP;
    try {
      keys = allocation_alias_keys(proc, alloc, &aliases);
      if (alloc.local_backing_owner && !alloc.backing_owner &&
          (aliases.size() != alloc.admission_generations.size() ||
           alloc.admission_generations.size() != alloc.mapped_device_ids.size()))
        return -EOPNOTSUPP;
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    }
  }

  ScopedPrivateGttCopySourceMutation copy_source_mutation(*this, proc, selected_handle);

  bool retry = false;
  std::shared_ptr<PendingOrdinaryMapping> pending;
  try {
    pending = begin_ordinary_mapping(proc, selected_handle, PendingOrdinaryMapping::Kind::Munmap,
                                     addr, length, 0, 0, aliases, keys, retry);
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  }
  if ((retry && !pending) || (!keys.empty() && !pending) ||
      (!retry && std::ranges::any_of(
                     keys, [&](const auto &key) { return alias_ingress_gate_->closed(key); }))) {
    if (pending && !retry)
      finish_ordinary_mapping(proc, selected_handle);
    return -EAGAIN;
  }

  int munmap_error = 0;
  PendingOrdinaryMapping ordinary{};
  const int drain_result =
      run_ordinary_mapping_drain(proc, selected_handle, pending ? *pending : ordinary, retry, [&] {
        std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
        auto &alloc = proc.allocations_.at(selected_handle);
        const bool previously_unstable = alloc.private_gtt_copy_source_unstable;
        if (alloc.private_gtt_copy_source_gate)
          alloc.private_gtt_copy_source_unstable = true;
        if (alloc.page_table_mapped_bytes &&
            unmap_from_gpu(proc, alloc.gpu_va, alloc.page_table_mapped_bytes) !=
                KfdProcess::PageTableMutationResult::Applied) {
          // A typed rejection occurs before the page-table loop begins.
          alloc.private_gtt_copy_source_unstable = previously_unstable;
          munmap_error = -EINVAL;
          if (pending)
            throw std::runtime_error("ordinary canonical munmap could not remove GPU PTEs");
          return;
        }
        if (pending &&
            fail_next_alias_mapping_after_pte_unmap_.exchange(false, std::memory_order_acq_rel))
          throw std::bad_alloc();
        if (safe_munmap(addr, length) != 0) {
          munmap_error = -errno;
          throw std::runtime_error("ordinary allocation munmap failed after PTE removal");
        }
        alloc.host_ptr = nullptr;
        alloc.host_ptr_owned = false;
        alloc.host_ptr_caller_reserved = false;
        alloc.host_ptr_mapped_bytes = 0;
        alloc.host_view_prot = 0;
        alloc.page_table_mapped_bytes = 0;
        alloc.host_view_generation = proc.next_host_view_generation_locked();
        alloc.private_gtt_copy_source_unstable = previously_unstable;
      });
  if (pending && (drain_result == 0 || std::ranges::none_of(keys, [&](const auto &key) {
                    return alias_ingress_gate_->closed(key);
                  })))
    finish_ordinary_mapping(proc, selected_handle);
  return munmap_error != 0 ? munmap_error : drain_result;
}

int SimulatedKfd::get_process_apertures_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
  auto n = static_cast<uint32_t>(gpus_.size());

  if (args->num_of_nodes == 0) {
    args->num_of_nodes = n;
    return 0;
  }

  auto *apertures =
      reinterpret_cast<kfd_process_device_apertures *>(args->kfd_process_device_apertures_ptr);
  const uint32_t filled = std::min(n, args->num_of_nodes);
  for (uint32_t i = 0; i < filled; ++i)
    apertures[i] = gpu_apertures(i);

  // The count written back is how many entries were filled, not how many nodes
  // exist -- kfd_ioctl_get_process_apertures_new() reports the loop index. A
  // caller whose buffer was smaller than the node count would otherwise iterate
  // past its own allocation over entries this call never wrote.
  args->num_of_nodes = filled;
  return 0;
}

kfd_process_device_apertures SimulatedKfd::gpu_apertures(uint32_t ordinal) const {
  const uint64_t offset = static_cast<uint64_t>(ordinal) * kApertureStride;
  const uint64_t lds_base = 0x1000000000000ULL + offset;
  const uint64_t scratch_base = 0x2000000000000ULL + offset;
  return {
      .lds_base = lds_base,
      .lds_limit = lds_base + 0xFFFFFFFFULL,
      .scratch_base = scratch_base,
      .scratch_limit = scratch_base + 0xFFFFFFFFULL,
      // rocjitsu maps GPU VAs directly to host pointers, so the aperture must
      // cover the host addresses accepted by the runtime.
      .gpuvm_base = 0x10000ULL,
      .gpuvm_limit = 0x7FFFFFFFFFFFULL,
      .gpu_id = ordinal < gpus_.size() ? gpus_[ordinal].gpu_id : 0,
      .pad = 0,
  };
}

int SimulatedKfd::get_available_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? get_available_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::get_available_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_get_available_memory_args *>(arg);
  uint64_t allocated = 0;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    for (auto &[handle, alloc] : proc.allocations_)
      allocated += alloc.size;
  }
  constexpr uint64_t kVramBytes = 64ULL << 30;
  args->available = kVramBytes - std::min(allocated, kVramBytes);
  return 0;
}

int SimulatedKfd::get_tile_config_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_tile_config_args *>(arg);
  if (daemon_mode_)
    return -ENOTSUP;

  auto *gpu = find_gpu(args->gpu_id);
  if (!gpu || !gpu->soc)
    return -EINVAL;

  uint32_t tile_write_count = std::min(args->num_tile_configs, kTileConfigCount);
  uint32_t macro_write_count = std::min(args->num_macro_tile_configs, kMacroTileConfigCount);

  // ROCr needs gb_addr_config for swizzled-address calculation. Tile-mode arrays are stubbed until
  // a simulator consumer needs their packed register encodings.
  if (args->tile_config_ptr && tile_write_count > 0) {
    auto *tile_config = reinterpret_cast<uint32_t *>(args->tile_config_ptr);
    std::ranges::fill_n(tile_config, tile_write_count, 0u);
  }
  if (args->macro_tile_config_ptr && macro_write_count > 0) {
    auto *macro_tile_config = reinterpret_cast<uint32_t *>(args->macro_tile_config_ptr);
    std::ranges::fill_n(macro_tile_config, macro_write_count, 0u);
  }

  args->num_tile_configs = tile_write_count;
  args->num_macro_tile_configs = macro_write_count;
  const uint32_t ordinal = gpu_ordinal(args->gpu_id);
  args->gb_addr_config =
      ordinal < gpu_infos_.size()
          ? kmd::gb_addr_config_for_gfx_target_version(gpu_infos_[ordinal].gfx_target_version)
          : kmd::gb_addr_config_for_arch(gpu->soc->arch());
  args->num_banks = 0;
  args->num_ranks = 0;
  return 0;
}

int SimulatedKfd::acquire_vm_ioctl([[maybe_unused]] void *arg) {
  (void)arg;
  return 0;
}

int SimulatedKfd::set_memory_policy_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? set_memory_policy_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::alloc_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? alloc_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::free_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  std::shared_ptr<const KfdProcess::GpuBacking> retiring_backing;
  return proc ? free_memory_ioctl(*proc, arg, retiring_backing) : -ESRCH;
}

int SimulatedKfd::map_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? map_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::unmap_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? unmap_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::alloc_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);
  if (args->size == 0 || args->size > UINT64_MAX - 0xFFF)
    return -EINVAL;
  // The kernel creates a PAGE_ALIGN(size) BO. LLVM can legally widen a scalar
  // load within its mapped tail page even when the logical tensor is smaller.
  const uint64_t allocation_size = (args->size + 0xFFF) & ~uint64_t(0xFFF);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);

  bool user_provided_va = (args->va_addr != 0);
  uint64_t va = args->va_addr;
  if (va == 0) {
    va = proc.next_gpu_va_;
  }
  if (va > UINT64_MAX - allocation_size)
    return -EINVAL;
  const bool is_userptr = (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0;
  const bool is_doorbell = (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) != 0;
  const bool use_userptr_mapping = is_userptr && !daemon_mode_;
  const bool needs_memfd =
      !use_userptr_mapping && (daemon_mode_ || !user_provided_va || !is_doorbell);
  UniqueDriverFd pending_backing;
  if (needs_memfd) {
    if (allocation_size > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
      return -EINVAL;
    // Acquire the backing before publishing a grant or consuming a handle.
    // A failed memfd_create must leave the allocation entirely unpublished.
    const int raw_fd = memfd_create("rocjitsu_alloc", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (raw_fd < 0)
      return -errno;
    pending_backing.reset(raw_fd);
  }
  constexpr uint32_t kMemoryKinds =
      KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
  std::shared_ptr<PrivateGttCopySourceGate> copy_source_gate;
  if ((args->flags & kMemoryKinds) == KFD_IOC_ALLOC_MEM_FLAGS_GTT) {
    try {
      copy_source_gate = std::make_shared<PrivateGttCopySourceGate>();
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    }
  }
  std::optional<KfdProcess::PageTableTransaction> va_transaction;
  if (use_userptr_mapping) {
    va_transaction.emplace(proc.begin_page_table_transaction());
    if (proc.has_foreign_resident_overlap(*va_transaction, va, allocation_size,
                                          {KfdProcess::VaOwnerKind::Kfd, 0}) !=
        KfdProcess::PageTableMutationResult::Applied)
      return -EINVAL;
  }
  auto previous_retired_ranges = proc.retired_gpu_ranges_;
  if (!user_provided_va)
    proc.next_gpu_va_ += allocation_size;

  KfdProcess::GpuAllocation alloc{};
  alloc.gpu_va = va;
  alloc.size = allocation_size;
  alloc.flags = args->flags;
  alloc.domain = grant_domain_for_flags(args->flags);
  alloc.handle = proc.next_handle_++;
  alloc.host_ptr = nullptr;
  alloc.gpu_id = args->gpu_id;
  alloc.user_va = user_provided_va;
  alloc.private_gtt_copy_source_gate = std::move(copy_source_gate);
  proc.claim_gpu_range_locked(va, allocation_size);
  proc.allocations_[alloc.handle] = alloc;
  if (va_transaction)
    proc.publish_gpu_grants_locked(*va_transaction);
  else
    proc.publish_gpu_grants_locked();

  auto rollback_allocation = [&](int error) {
    if (alloc.page_table_mapped_bytes)
      (void)unmap_from_gpu(proc, va, alloc.page_table_mapped_bytes,
                           va_transaction ? &*va_transaction : nullptr);
    if (alloc.host_ptr_owned)
      safe_munmap(alloc.host_ptr, alloc.host_ptr_mapped_bytes);
    if (alloc.memfd >= 0) {
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.erase(alloc.memfd);
      }
      libc_passthrough().close(alloc.memfd);
    }
    proc.allocations_.erase(alloc.handle);
    proc.retired_gpu_ranges_.swap(previous_retired_ranges);
    if (va_transaction)
      proc.publish_gpu_grants_locked(*va_transaction);
    else
      proc.publish_gpu_grants_locked();
    if (!user_provided_va)
      proc.next_gpu_va_ = va;
    return error;
  };

  auto alloc_mtype = pte_mtype_for_flags(args->flags);
  if (use_userptr_mapping) {
    alloc.host_ptr = reinterpret_cast<void *>(va);
    if (map_to_gpu(proc, va, reinterpret_cast<void *>(va), alloc.size, alloc_mtype,
                   KfdProcess::HostExtentOwner::Application,
                   &*va_transaction) != KfdProcess::PageTableMutationResult::Applied)
      return rollback_allocation(-EINVAL);
    alloc.page_table_mapped_bytes = alloc.size;
    va_transaction.reset();
  } else if (needs_memfd) {
    {
      alloc.memfd = safe_fcntl(pending_backing.get(), F_DUPFD_CLOEXEC, kBackingFdMin);
      if (alloc.memfd < 0)
        alloc.memfd = pending_backing.release();
      else
        pending_backing.reset();
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(alloc.memfd);
      }
      if (alloc.memfd >= 0) {
        if (ftruncate(alloc.memfd, static_cast<off_t>(alloc.size)) != 0)
          return rollback_allocation(-errno);
        if (daemon_mode_ || !user_provided_va)
          fallocate(alloc.memfd, 0, 0, static_cast<off_t>(alloc.size));
        safe_fcntl(alloc.memfd, F_ADD_SEALS, F_SEAL_SHRINK);

        if (daemon_mode_ && !is_doorbell) {
          auto *mapped =
              safe_mmap(nullptr, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.memfd, 0);
          if (mapped != MAP_FAILED) {
            alloc.host_ptr = mapped;
            alloc.host_ptr_owned = true;
            alloc.host_ptr_mapped_bytes = alloc.size;
            alloc.host_view_prot = PROT_READ | PROT_WRITE;
            alloc.host_view_generation = proc.next_host_view_generation_locked();
            // Driver-owned: this is our memfd, mapped read-write here and held
            // open, so nothing outside can change its protection or unmap it.
            va_transaction.emplace(proc.begin_page_table_transaction());
            if (proc.has_foreign_resident_overlap(*va_transaction, va, alloc.size,
                                                  {KfdProcess::VaOwnerKind::Kfd, 0}) !=
                KfdProcess::PageTableMutationResult::Applied) {
              va_transaction.reset();
              return rollback_allocation(-EINVAL);
            }
            if (map_to_gpu(proc, va, alloc.host_ptr, alloc.size, alloc_mtype,
                           KfdProcess::HostExtentOwner::Driver,
                           &*va_transaction) != KfdProcess::PageTableMutationResult::Applied) {
              va_transaction.reset();
              return rollback_allocation(-EINVAL);
            }
            alloc.page_table_mapped_bytes = alloc.size;
            va_transaction.reset();
          } else {
            return rollback_allocation(-errno);
          }
        }
      }
    }
  }

  proc.allocations_[alloc.handle] = alloc;

  args->handle = alloc.handle;
  args->va_addr = va;
  if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) {
    args->mmap_offset = KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(args->gpu_id);
  } else {
    args->mmap_offset = alloc.handle << 12;
  }

  util::Logger::cp([&](auto &os) {
    os << std::format("ALLOC_MEMORY handle={} gpu_va={:#x} size={:#x} flags={:#x}", alloc.handle,
                      va, args->size, args->flags);
  });
  util::Logger::vm([&](auto &os) {
    os << std::format(
        "ALLOC pid={} handle={} gpu_va={:#x} size={} flags={:#x} memfd={} host_ptr={}",
        proc.process_id(), alloc.handle, va, args->size, args->flags, alloc.memfd,
        reinterpret_cast<uintptr_t>(alloc.host_ptr));
  });
  return 0;
}

bool SimulatedKfd::allocate_scratch_backing(uint32_t process_id, uint64_t gpu_va, size_t size) {
  if (size == 0 || size > std::numeric_limits<size_t>::max() - 0xFFF)
    return false;

  std::shared_ptr<KfdProcess> proc;
  {
    std::lock_guard<std::mutex> plk(process_mutex_);
    for (auto &[fd, p] : processes_) {
      if (p->process_id() == process_id) {
        proc = p;
        break;
      }
    }
  }
  if (!proc)
    return false;

  const size_t aligned_size = (size + 0xFFF) & ~0xFFFULL;
  if (aligned_size > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
      gpu_va > std::numeric_limits<uint64_t>::max() - aligned_size)
    return false;
  const uint64_t end = gpu_va + aligned_size;

  // Fan-out shards can request the same range concurrently, and non-barrier
  // dispatches can grow it while older waves still use it. Keep every existing
  // allocation in place and back only the gaps: copying or remapping an old
  // prefix would race with live spills. Retain each chunk until process teardown.
  std::lock_guard<std::mutex> scratch_lock(proc->scratch_backing_mutex_);
  std::vector<std::pair<uint64_t, uint64_t>> backed;
  std::vector<uint64_t> incomplete;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (const auto &[handle, alloc] : proc->allocations_) {
      if (!alloc.host_ptr || alloc.gpu_va >= end)
        continue;
      const uint64_t begin = std::max(gpu_va, alloc.gpu_va);
      const uint64_t offset = begin - alloc.gpu_va;
      if (offset < alloc.size) {
        backed.emplace_back(begin, begin + std::min(alloc.size - offset, end - begin));
        if (alloc.driver_gpu_mapping && alloc.scratch_mapping_incomplete)
          incomplete.push_back(handle);
      }
    }
  }
  std::ranges::sort(backed);
  std::vector<std::pair<uint64_t, size_t>> missing;
  uint64_t cursor = gpu_va;
  for (const auto &[begin, backed_end] : backed) {
    if (cursor < begin)
      missing.emplace_back(cursor, static_cast<size_t>(begin - cursor));
    cursor = std::max(cursor, backed_end);
  }
  if (cursor < end)
    missing.emplace_back(cursor, static_cast<size_t>(end - cursor));

  struct PreparedScratchRange {
    uint64_t gpu_va;
    size_t size;
    void *host_ptr;
    UniqueMapping backing;
  };
  std::vector<PreparedScratchRange> prepared;
  prepared.reserve(missing.size());

  // Prepare every backing before publishing any grant. A failed memfd or mmap
  // then leaves neither a partial page table nor driver-owned VA grants.
  for (const auto &[range_va, range_size] : missing) {
    auto raw_fd = memfd_create("rocjitsu_scratch", MFD_CLOEXEC);
    if (raw_fd < 0)
      return false;

    int memfd = safe_fcntl(raw_fd, F_DUPFD_CLOEXEC, kBackingFdMin);
    if (memfd < 0)
      memfd = raw_fd;
    else
      libc_passthrough().close(raw_fd);
    {
      std::lock_guard<std::mutex> lk(owned_fds_mutex_);
      owned_fds_.insert(memfd);
    }

    if (ftruncate(memfd, static_cast<off_t>(range_size)) != 0) {
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.erase(memfd);
      }
      libc_passthrough().close(memfd);
      return false;
    }
    auto *host_ptr = safe_mmap(nullptr, range_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
    if (host_ptr == MAP_FAILED) {
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.erase(memfd);
      }
      libc_passthrough().close(memfd);
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(owned_fds_mutex_);
      owned_fds_.erase(memfd);
    }
    libc_passthrough().close(memfd);
    std::memset(host_ptr, 0, range_size);
    prepared.push_back({range_va, range_size, host_ptr, UniqueMapping(host_ptr, range_size)});
  }

  bool failed = false;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto va_transaction = proc->begin_page_table_transaction();
    failed = proc->event_state_.is_closing();
    for (const uint64_t handle : incomplete) {
      if (failed)
        break;
      auto &alloc = proc->allocations_.at(handle);
      try {
        if (proc->map_pages(va_transaction, alloc.gpu_va, alloc.host_ptr, alloc.size,
                            amdgpu::Mtype::RW, KfdProcess::HostExtentOwner::Driver) !=
            KfdProcess::PageTableMutationResult::Applied) {
          failed = true;
          break;
        }
        alloc.scratch_mapping_incomplete = false;
      } catch (...) {
        failed = true;
      }
    }
    for (const auto &range : prepared) {
      if (failed)
        break;
      if (proc->has_foreign_resident_overlap(va_transaction, range.gpu_va, range.size,
                                             {KfdProcess::VaOwnerKind::Kfd, 0}) !=
          KfdProcess::PageTableMutationResult::Applied) {
        failed = true;
        break;
      }
    }
    if (!failed) {
      // The prepared ranges commit under one VA writer transaction. A GEM map
      // cannot install a foreign PTE between scratch preflight and publication.
      for (auto &range : prepared) {
        KfdProcess::GpuAllocation alloc{};
        alloc.gpu_va = range.gpu_va;
        alloc.size = range.size;
        alloc.host_ptr = range.host_ptr;
        alloc.host_ptr_owned = true;
        alloc.host_ptr_mapped_bytes = range.size;
        alloc.handle = proc->next_handle_++;
        alloc.memfd = -1;
        alloc.page_table_mapped_bytes = range.size;
        alloc.driver_gpu_mapping = true;
        alloc.scratch_mapping_incomplete = true;
        alloc.driver_admission_generation = proc->next_gpu_admission_generation_locked();
        try {
          auto previous_retired_ranges = proc->retired_gpu_ranges_;
          try {
            proc->claim_gpu_range_locked(alloc.gpu_va, alloc.size);
            proc->allocations_.emplace(alloc.handle, alloc);
            proc->publish_gpu_grants_locked(va_transaction);
          } catch (...) {
            proc->allocations_.erase(alloc.handle);
            proc->retired_gpu_ranges_.swap(previous_retired_ranges);
            throw;
          }
          // The descriptor now retains the backing even if a later page-table
          // insertion throws after installing a prefix.
          static_cast<void>(range.backing.release());
          if (proc->map_pages(va_transaction, range.gpu_va, range.host_ptr, range.size,
                              amdgpu::Mtype::RW, KfdProcess::HostExtentOwner::Driver) !=
              KfdProcess::PageTableMutationResult::Applied) {
            failed = true;
            break;
          }
          proc->allocations_.at(alloc.handle).scratch_mapping_incomplete = false;
        } catch (...) {
          failed = true;
          break;
        }
      }
    }
  }
  if (failed)
    return false;
  for (const auto &range : prepared)
    util::Logger::vm([&](auto &os) {
      os << "SCRATCH_BACKING pid=" << process_id << " gpu_va=0x" << std::hex << range.gpu_va
         << " size=0x" << range.size << std::dec << " host=" << range.host_ptr;
    });
  return true;
}

int SimulatedKfd::free_memory_ioctl(
    KfdProcess &proc, void *arg, std::shared_ptr<const KfdProcess::GpuBacking> &retiring_backing) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
  ScopedPrivateGttCopySourceMutation copy_source_mutation(*this, proc, args->handle);
  // A failed local UNMAP owns its exact closed generation. Finish through its
  // original drain and durable writeback marker before FREE starts a batch.
  // A failed ordinary mapping instead owns the whole selected batch; FREE
  // adopts that batch below, so do not retry one of its keys as an UNMAP.
  std::shared_ptr<PendingOrdinaryMapping> pending_mapping;
  {
    std::lock_guard lock(ordinary_mapping_mutex_);
    const auto it = pending_ordinary_mappings_.find({proc.process_incarnation(), args->handle});
    if (it != pending_ordinary_mappings_.end())
      pending_mapping = it->second;
  }
  std::vector<uint32_t> failed_local_unmaps;
  try {
    std::lock_guard lock(proc.alloc_mutex_);
    const auto it = proc.allocations_.find(args->handle);
    if (it != proc.allocations_.end() && !it->second.retirement_in_progress &&
        it->second.local_backing_owner && !it->second.backing_owner) {
      const auto aliases = proc.gpu_local_alias_snapshot();
      for (const auto &alias : *aliases) {
        const auto generation = it->second.admission_generations.find(alias.gpu_id);
        if (alias.gpu_va == it->second.gpu_va && alias.extent == it->second.size &&
            alias.backing_id == it->second.local_backing_owner->backing_id() &&
            generation != it->second.admission_generations.end() &&
            generation->second == alias.admission_generation &&
            alias_ingress_gate_->closed(GpuAliasIngressGate::key(alias)) &&
            (!pending_mapping || std::ranges::none_of(pending_mapping->keys, [&](const auto &key) {
              return key == GpuAliasIngressGate::key(alias);
            })))
          failed_local_unmaps.push_back(alias.gpu_id);
      }
    }
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  }
  for (uint32_t gpu_id : failed_local_unmaps) {
    kfd_ioctl_unmap_memory_from_gpu_args unmap{};
    unmap.handle = args->handle;
    unmap.n_devices = 1;
    unmap.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
    if (const int result = unmap_memory_ioctl(proc, &unmap); result != 0)
      return result;
  }
  std::vector<GpuAliasIngressGate::Key> keys;
  bool newly_started = false;
  {
    std::lock_guard lock(proc.alloc_mutex_);
    const auto it = proc.allocations_.find(args->handle);
    if (it == proc.allocations_.end())
      return 0;
    auto &alloc = it->second;
    if (!alloc.retirement_in_progress) {
      newly_started = true;
      try {
        const auto owner = alloc.backing_owner ? alloc.backing_owner : alloc.local_backing_owner;
        if (owner) {
          const auto aliases =
              alloc.backing_owner ? proc.gpu_alias_snapshot() : proc.gpu_local_alias_snapshot();
          for (const auto &alias : *aliases) {
            const auto generation = alloc.admission_generations.find(alias.gpu_id);
            if (alias.gpu_va == alloc.gpu_va && alias.extent == alloc.size &&
                alias.backing_id == owner->backing_id() &&
                generation != alloc.admission_generations.end() &&
                generation->second == alias.admission_generation)
              alloc.retiring_aliases.push_back(alias);
          }
        }
        if (alloc.local_backing_owner && !alloc.backing_owner &&
            (alloc.retiring_aliases.size() != alloc.admission_generations.size() ||
             alloc.admission_generations.size() != alloc.mapped_device_ids.size())) {
          alloc.retiring_aliases.clear();
          return -EOPNOTSUPP;
        }
      } catch (const std::bad_alloc &) {
        alloc.retiring_aliases.clear();
        return -ENOMEM;
      }
    }
    // Keep the owner alive until allocation, IPC, and grant snapshots have
    // retired and alloc_mutex_/ipc_mutex_ have both been released.
    retiring_backing = alloc.backing_owner ? alloc.backing_owner : alloc.local_backing_owner;
    try {
      keys.reserve(alloc.retiring_aliases.size());
      for (const auto &alias : alloc.retiring_aliases)
        keys.push_back(GpuAliasIngressGate::key(alias));
    } catch (const std::bad_alloc &) {
      if (!alloc.retirement_in_progress)
        alloc.retiring_aliases.clear();
      return -ENOMEM;
    }
    alloc.retirement_in_progress = true;
  }

  // A different allocation may own the gate-wide close claim. If this FREE
  // has not closed any of its generations, restore its newly started marker
  // so MAP, UNMAP, mmap, and export remain usable before the caller retries.
  // An existing FREE or a post failure that closed these keys keeps its marker.
  const auto rollback_unclosed_start = [&]() noexcept {
    if (!newly_started)
      return;
    try {
      if (std::ranges::any_of(keys,
                              [&](const auto &key) { return alias_ingress_gate_->closed(key); }))
        return;
      std::lock_guard lock(proc.alloc_mutex_);
      const auto it = proc.allocations_.find(args->handle);
      if (it == proc.allocations_.end() || !it->second.retirement_in_progress ||
          it->second.grants_retired)
        return;
      it->second.retirement_in_progress = false;
      it->second.retiring_aliases.clear();
    } catch (...) {
      // Retain the fail-closed marker if its gate state cannot be observed.
    }
  };

  const auto retire = [&] {
    std::lock_guard lock(proc.alloc_mutex_);
    auto it = proc.allocations_.find(args->handle);
    if (it == proc.allocations_.end() || !it->second.retirement_in_progress)
      throw std::runtime_error("FREE lost its retiring allocation");
    auto &alloc = it->second;
    std::optional<KfdProcess::PageTableTransaction> va_transaction;
    va_transaction.emplace(proc.begin_page_table_transaction());
    if (!alloc.grants_retired) {
      auto mapped = std::move(alloc.mapped_device_ids);
      auto generations = std::move(alloc.admission_generations);
      const bool driver_mapping = std::exchange(alloc.driver_gpu_mapping, false);
      const uint64_t driver_generation = std::exchange(alloc.driver_admission_generation, 0);
      try {
        if (!keys.empty() &&
            fail_next_alias_free_publish_.exchange(false, std::memory_order_acq_rel))
          throw std::bad_alloc();
        proc.publish_gpu_grants_locked(*va_transaction);
      } catch (...) {
        alloc.mapped_device_ids = std::move(mapped);
        alloc.admission_generations = std::move(generations);
        alloc.driver_gpu_mapping = driver_mapping;
        alloc.driver_admission_generation = driver_generation;
        throw;
      }
      alloc.grants_retired = true;
    }
    // Deny GPU ingress before waiting for a PTE reader to release its lock.
    // Retain the request transaction through the ownership check and unmap.
    if (alloc.page_table_mapped_bytes &&
        proc.has_foreign_resident_overlap(
            *va_transaction, alloc.gpu_va, alloc.page_table_mapped_bytes,
            {KfdProcess::VaOwnerKind::Kfd, 0}) != KfdProcess::PageTableMutationResult::Applied)
      throw std::runtime_error("FREE encountered a foreign GPU VA owner");
    if (alloc.page_table_mapped_bytes) {
      if (unmap_from_gpu(proc, alloc.gpu_va, alloc.page_table_mapped_bytes, &*va_transaction) !=
          KfdProcess::PageTableMutationResult::Applied)
        throw std::runtime_error("FREE could not remove its GPU PTEs");
      alloc.page_table_mapped_bytes = 0;
    }

    // The final grant publication can allocate. Preserve the descriptor and
    // tombstones until that publication succeeds, then retire IPC and backing.
    auto prior_retired_ranges = proc.retired_gpu_ranges_;
    try {
      proc.retire_gpu_range_locked(alloc.gpu_va, alloc.size);
    } catch (...) {
      proc.retired_gpu_ranges_.swap(prior_retired_ranges);
      throw;
    }
    auto retired = proc.allocations_.extract(it);
    try {
      if (!keys.empty() &&
          fail_next_alias_free_final_publish_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();
      proc.publish_gpu_grants_locked(*va_transaction);
    } catch (...) {
      proc.allocations_.insert(std::move(retired));
      proc.retired_gpu_ranges_.swap(prior_retired_ranges);
      throw;
    }
    va_transaction.reset();
    auto &freed = retired.mapped();
    if (freed.cacheable_peer_for_testing ||
        (freed.backing_owner && freed.backing_owner->controlled_peer_for_testing()))
      for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal)
        if (gpus_[ordinal].soc)
          gpus_[ordinal].soc->gpu_vm().retire_controlled_cache_backing(
              proc.gpu(ordinal).address_space, freed.gpu_va, freed.size);
    if (freed.imported && freed.dmabuf_fd >= 0) {
      libc_passthrough().close(freed.dmabuf_fd);
      if (auto dmabuf_it = proc.imported_dmabufs_.find(args->handle);
          dmabuf_it != proc.imported_dmabufs_.end()) {
        proc.fd_to_import_handle_.erase(dmabuf_it->second.fd);
        proc.imported_dmabufs_.erase(dmabuf_it);
      }
    }
    if (freed.host_ptr && freed.host_ptr_owned && !freed.host_ptr_caller_reserved)
      safe_munmap(freed.host_ptr, freed.host_ptr_mapped_bytes);
    if (freed.memfd >= 0) {
      {
        std::lock_guard fd_lock(owned_fds_mutex_);
        owned_fds_.erase(freed.memfd);
      }
      libc_passthrough().close(freed.memfd);
    }
    {
      std::lock_guard ipc_lock(ipc_mutex_);
      for (auto ipc_it = ipc_store_.begin(); ipc_it != ipc_store_.end();) {
        if (ipc_it->second.source_process_id == proc.process_id() &&
            ipc_it->second.source_alloc_handle == args->handle) {
          if (ipc_it->second.backing_memfd >= 0)
            libc_passthrough().close(ipc_it->second.backing_memfd);
          ipc_it = ipc_store_.erase(ipc_it);
        } else {
          ++ipc_it;
        }
      }
    }
  };

  const auto writeback_then_retire = [&](const GpuAliasIngressGate::DrainAuthority &authority) {
    const std::optional<bool> done = alias_ingress_gate_->final_writeback_done(authority);
    if (!done)
      throw std::runtime_error("canonical FREE drain authority expired");
    if (!*done) {
      // Check the retiring grants and any remaining PTE ownership before the
      // cache group publishes dirty bytes. Release both locks before entering
      // the coordinator; preparation rechecks live grants and translations.
      std::vector<KfdProcess::GpuAliasRecord> retiring_aliases;
      {
        std::lock_guard lock(proc.alloc_mutex_);
        const auto it = proc.allocations_.find(args->handle);
        if (it == proc.allocations_.end() || !it->second.retirement_in_progress ||
            it->second.grants_retired ||
            (it->second.backing_owner ? it->second.backing_owner
                                      : it->second.local_backing_owner) != retiring_backing ||
            it->second.retiring_aliases.size() != keys.size())
          throw std::runtime_error("canonical FREE allocation changed before writeback");
        const auto &alloc = it->second;
        for (size_t i = 0; i < keys.size(); ++i) {
          const auto &alias = alloc.retiring_aliases[i];
          const auto generation = alloc.admission_generations.find(alias.gpu_id);
          if (GpuAliasIngressGate::key(alias) != keys[i] || alias.gpu_va != alloc.gpu_va ||
              alias.extent != alloc.size || alias.backing_id != retiring_backing->backing_id() ||
              generation == alloc.admission_generations.end() ||
              generation->second != alias.admission_generation ||
              !proc.gpu_alias_grant_current(alias))
            throw std::runtime_error("canonical FREE alias changed before writeback");
        }
        if (alloc.page_table_mapped_bytes) {
          auto va_transaction = proc.begin_page_table_transaction();
          if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va,
                                                alloc.page_table_mapped_bytes,
                                                {KfdProcess::VaOwnerKind::Kfd, 0}) !=
              KfdProcess::PageTableMutationResult::Applied)
            throw std::runtime_error("canonical FREE PTE ownership changed before writeback");
        }
        retiring_aliases = alloc.retiring_aliases;
      }

      const auto domains = maintenance_domains_for_aliases(retiring_aliases);
      const auto resolve = [this, &authority](uint32_t domain_gpu_id, uint32_t vmid,
                                              uint64_t address, size_t size) {
        return prepare_gpu_alias_writeback_for_vmid(domain_gpu_id, vmid, address, size, &authority);
      };
      std::vector<amdgpu::DeviceCacheAliasScope> selected;
      // The entire drained batch may retire overlapping aliases, but no
      // outsider may retain a live view of its backing lines after FREE.
      if (std::ranges::none_of(retiring_aliases, &KfdProcess::GpuAliasRecord::local) &&
          has_exclusive_canonical_lines(retiring_aliases)) {
        selected.reserve(retiring_aliases.size());
        for (const auto &alias : retiring_aliases) {
          auto scope = scoped_canonical_alias(proc, alias);
          if (!scope) {
            selected.clear();
            break;
          }
          selected.push_back(std::move(*scope));
        }
      }
      auto leases = [&]() {
        if (!selected.empty()) {
          try {
            return amdgpu::DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
                domains, selected, resolve);
          } catch (const util::Exception &) {
            // An orphan or overlapping dirty tag needs the whole-group path.
          }
        }
        return amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            domains, amdgpu::DeviceCacheOperation::WritebackInvalidate, resolve);
      }();
      while (!leases.empty())
        leases.pop_back();
      if (!alias_ingress_gate_->mark_final_writeback_done(authority))
        throw std::runtime_error("canonical FREE lost its drained batch");
    }

    retire();
  };

  // The op lock serializes ioctls and mappings. The retiring marker also lets
  // an mmap/munmap of this allocation fail promptly during the permit drain.
  // No allocation, VM, or IPC lock is held while CU/SDMA permits drain.
  if (keys.empty()) {
    try {
      retire();
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    } catch (...) {
      return -EIO;
    }
    finish_ordinary_mapping(proc, args->handle);
    return 0;
  }

  std::function<void()> queued_continuation;
  std::atomic<bool> queued{false};
  const auto owner_post = [&](std::function<void()> continuation) noexcept {
    queued_continuation = std::move(continuation);
    queued.store(true, std::memory_order_release);
    queued.notify_one();
  };
  bool closed = false;
  try {
    closed = alias_ingress_gate_->close_batch_with_drain_authority(keys, owner_post,
                                                                   writeback_then_retire) ||
             alias_ingress_gate_->retry_closed_batch_with_drain_authority(keys, owner_post,
                                                                          writeback_then_retire);
  } catch (const std::bad_alloc &) {
    rollback_unclosed_start();
    return -ENOMEM;
  } catch (...) {
    rollback_unclosed_start();
    return -EIO;
  }
  if (!closed) {
    rollback_unclosed_start();
    return -EAGAIN;
  }
  queued.wait(false, std::memory_order_acquire);
  try {
    queued_continuation();
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return -EIO;
  }
  if (!alias_ingress_gate_->reopen_batch(keys))
    return -EIO;
  finish_ordinary_mapping(proc, args->handle);
  return 0;
}

int SimulatedKfd::map_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
  if (args->n_success > args->n_devices)
    return -EINVAL;
  ScopedPrivateGttCopySourceMutation copy_source_mutation(*this, proc, args->handle);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end()) {
    util::Logger::cp(
        [&](auto &os) { os << std::format("MAP_MEMORY_FAIL handle={} not found", args->handle); });
    return -EINVAL;
  }
  auto &alloc = it->second;
  if (alloc.retirement_in_progress)
    return -EAGAIN;
  util::Logger::cp([&](auto &os) {
    os << std::format("MAP_MEMORY handle={} gpu_va={:#x} size={:#x} n_devices={} host_ptr={}",
                      alloc.handle, alloc.gpu_va, alloc.size, args->n_devices,
                      alloc.host_ptr != nullptr);
  });
  constexpr uint32_t kMemoryTypes =
      KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
  const uint32_t memory_type = alloc.flags & kMemoryTypes;
  for (uint32_t i = args->n_success; i < args->n_devices; ++i) {
    uint32_t gpu_id = 0;
    const uint64_t offset = uint64_t{i} * sizeof(gpu_id);
    if (args->device_ids_array_ptr > UINT64_MAX - offset ||
        copy_ioctl_user_buffer(&gpu_id, args->device_ids_array_ptr + offset, sizeof(gpu_id)) != 0)
      return -EFAULT;

    const bool system_memory = memory_type == KFD_IOC_ALLOC_MEM_FLAGS_GTT ||
                               memory_type == KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    const GpuDevice *requester = find_gpu(gpu_id);
    const bool known_gpu = requester || (system_memory && gpu_id == 0 && !gpus_.empty());
    if (!known_gpu || (!system_memory && memory_type != KFD_IOC_ALLOC_MEM_FLAGS_VRAM))
      return -EINVAL;

    // A cacheable local grant needs a retained sink before it is visible.
    // UC grants do not leave dirty cache lines; an admitted UC BO cannot be
    // promoted to CC through IPC export until that transition has its own gate.
    const bool cacheable_local = (alloc.ipc_shared && !alloc.ipc_export_retains_local_policy) ||
                                 pte_mtype_for_flags(alloc.flags) != amdgpu::Mtype::UC;
    if (memory_type == KFD_IOC_ALLOC_MEM_FLAGS_VRAM && gpu_id == alloc.gpu_id && cacheable_local &&
        !alloc.backing_owner && !alloc.local_backing_owner) {
      // Caller-owned VRAM pages have no durable alias. UC remains safe, but
      // cacheable access has no bounded destination for dirty retirement.
      if (alloc.memfd < 0 || (alloc.user_va && !alloc.host_ptr) ||
          alloc.size > std::numeric_limits<size_t>::max())
        return -EOPNOTSUPP;
      try {
        alloc.local_backing_owner = KfdProcess::GpuBacking::create_memfd_alias(
            alloc.memfd, static_cast<size_t>(alloc.size), alloc.gpu_id,
            alloc.cacheable_peer_for_testing);
      } catch (const std::bad_alloc &) {
        return -ENOMEM;
      }
      if (!alloc.local_backing_owner)
        return -EOPNOTSUPP;
    }

    if (memory_type == KFD_IOC_ALLOC_MEM_FLAGS_VRAM && gpu_id != alloc.gpu_id) {
      const GpuDevice *owner = find_gpu(alloc.gpu_id);
      // A peer grant must name the same full canonical VRAM object that the
      // endpoint will use. Check the executable route before publishing it;
      // each transfer rechecks the route and grant when it is prepared.
      if (!requester || !requester->soc || !owner || !owner->soc || !alloc.host_ptr ||
          !alloc.host_ptr_owned || alloc.host_ptr_mapped_bytes != alloc.size ||
          alloc.driver_gpu_mapping || !alloc.backing_owner ||
          alloc.backing_owner->gpu_id() != alloc.gpu_id ||
          alloc.backing_owner->size() != alloc.size)
        return -EINVAL;
      std::lock_guard endpoint_lock(peer_endpoint_access_->mutex);
      if (!peer_endpoint_access_->endpoint ||
          !peer_endpoint_access_->endpoint->can_route(gpu_id, alloc.gpu_id))
        return -EINVAL;
    }

    if (const auto generation = alloc.admission_generations.find(gpu_id);
        generation != alloc.admission_generations.end() &&
        (alloc.backing_owner || alloc.local_backing_owner)) {
      const bool local = !alloc.backing_owner;
      const auto owner = local ? alloc.local_backing_owner : alloc.backing_owner;
      const auto aliases = local ? proc.gpu_local_alias_snapshot() : proc.gpu_alias_snapshot();
      const auto alias = std::ranges::find_if(*aliases, [&](const auto &candidate) {
        return candidate.gpu_id == gpu_id && candidate.gpu_va == alloc.gpu_va &&
               candidate.extent == alloc.size && candidate.backing_id == owner->backing_id() &&
               candidate.admission_generation == generation->second;
      });
      // A failed UNMAP can remove the PTE while retaining this grant and its
      // closed gate. Its retry must retire that generation before MAP can act.
      if (alias != aliases->end() && alias_ingress_gate_->closed(GpuAliasIngressGate::key(*alias)))
        return -EAGAIN;
    }

    auto va_transaction = proc.begin_page_table_transaction();
    if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va, alloc.size,
                                          {KfdProcess::VaOwnerKind::Kfd, 0}) !=
        KfdProcess::PageTableMutationResult::Applied)
      return -EINVAL;
    const bool previously_unstable = alloc.private_gtt_copy_source_unstable;
    if (alloc.private_gtt_copy_source_gate)
      alloc.private_gtt_copy_source_unstable = true;
    bool new_grant = false;
    try {
      new_grant = alloc.mapped_device_ids.insert(gpu_id).second;
      if (new_grant)
        alloc.admission_generations[gpu_id] = proc.next_gpu_admission_generation_locked();
    } catch (const std::bad_alloc &) {
      if (!alloc.private_gtt_copy_source_gate)
        throw;
      // Neither node is published yet; discard a newly inserted first node.
      if (new_grant) {
        alloc.mapped_device_ids.erase(gpu_id);
        alloc.admission_generations.erase(gpu_id);
      }
      alloc.private_gtt_copy_source_unstable = previously_unstable;
      return -ENOMEM;
    }
    const size_t available_bytes =
        alloc.host_ptr_owned ? std::min(alloc.size, alloc.host_ptr_mapped_bytes) : alloc.size;
    bool pte_changed = false;
    const auto gpu_backing = alloc.backing_owner ? alloc.backing_owner : alloc.local_backing_owner;
    const bool private_gpu_view =
        gpu_backing &&
        (alloc.host_view_prot & (PROT_READ | PROT_WRITE)) != (PROT_READ | PROT_WRITE);
    if (alloc.host_ptr && (alloc.page_table_mapped_bytes < available_bytes || private_gpu_view)) {
      const amdgpu::Mtype mtype =
          alloc.cacheable_peer_for_testing
              ? pte_mtype_for_flags(alloc.flags, alloc.shared)
              : (alloc.backing_owner ? amdgpu::Mtype::UC
                                     : (alloc.ipc_shared && !alloc.ipc_export_retains_local_policy
                                            ? amdgpu::Mtype::CC
                                            : pte_mtype_for_flags(alloc.flags, alloc.shared)));
      void *gpu_ptr = private_gpu_view ? gpu_backing->data() : alloc.host_ptr;
      if (map_to_gpu(proc, alloc.gpu_va, gpu_ptr, available_bytes, mtype,
                     private_gpu_view ? KfdProcess::HostExtentOwner::Driver
                                      : KfdProcess::HostExtentOwner::Application,
                     &va_transaction) != KfdProcess::PageTableMutationResult::Applied) {
        if (new_grant) {
          alloc.mapped_device_ids.erase(gpu_id);
          alloc.admission_generations.erase(gpu_id);
        }
        alloc.private_gtt_copy_source_unstable = previously_unstable;
        return -EINVAL;
      }
      alloc.page_table_mapped_bytes = available_bytes;
      pte_changed = true;
    }
    try {
      if (alloc.private_gtt_copy_source_gate && !pte_changed &&
          fail_next_private_gtt_map_publish_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();
      proc.publish_gpu_grants_locked(va_transaction);
    } catch (const std::bad_alloc &) {
      if (!alloc.private_gtt_copy_source_gate)
        throw;
      // Grant publication builds a replacement index before its atomic store.
      // Without a PTE change, discarding any new grant restores the old source.
      if (!pte_changed) {
        if (new_grant) {
          alloc.mapped_device_ids.erase(gpu_id);
          alloc.admission_generations.erase(gpu_id);
        }
        alloc.private_gtt_copy_source_unstable = previously_unstable;
      }
      return -ENOMEM;
    }
    alloc.ever_gpu_admitted = true;
    args->n_success = i + 1;
    alloc.private_gtt_copy_source_unstable = previously_unstable;
  }
  return 0;
}

int SimulatedKfd::unmap_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arg);
  if (args->n_success > args->n_devices)
    return -EINVAL;
  ScopedPrivateGttCopySourceMutation copy_source_mutation(*this, proc, args->handle);

  // UNMAP only changes a mapping grant and its shared page-table side effect.
  // FREE_MEMORY_OF_GPU owns the allocation and backing descriptors.
  constexpr uint32_t kMemoryTypes =
      KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
  for (uint32_t i = args->n_success; i < args->n_devices; ++i) {
    uint32_t gpu_id = 0;
    const uint64_t offset = uint64_t{i} * sizeof(gpu_id);
    if (args->device_ids_array_ptr > UINT64_MAX - offset ||
        copy_ioctl_user_buffer(&gpu_id, args->device_ids_array_ptr + offset, sizeof(gpu_id)) != 0)
      return -EFAULT;

    std::optional<uint64_t> generation;
    std::optional<KfdProcess::GpuAliasRecord> alias;
    {
      std::lock_guard lock(proc.alloc_mutex_);
      const auto it = proc.allocations_.find(args->handle);
      if (it == proc.allocations_.end())
        return -EINVAL;
      const auto &alloc = it->second;
      if (alloc.retirement_in_progress)
        return -EAGAIN;
      const uint32_t memory_type = alloc.flags & kMemoryTypes;
      const bool system_memory = memory_type == KFD_IOC_ALLOC_MEM_FLAGS_GTT ||
                                 memory_type == KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
      const bool known_gpu = find_gpu(gpu_id) || (system_memory && gpu_id == 0 && !gpus_.empty());
      if (!known_gpu || (!system_memory && memory_type != KFD_IOC_ALLOC_MEM_FLAGS_VRAM) ||
          !alloc.mapped_device_ids.contains(gpu_id))
        return -EINVAL;

      if (const auto grant = alloc.admission_generations.find(gpu_id);
          grant != alloc.admission_generations.end())
        generation = grant->second;
      if (generation && (alloc.backing_owner || alloc.local_backing_owner)) {
        const bool local = !alloc.backing_owner;
        const auto owner = local ? alloc.local_backing_owner : alloc.backing_owner;
        const auto aliases = local ? proc.gpu_local_alias_snapshot() : proc.gpu_alias_snapshot();
        const auto found = std::ranges::find_if(*aliases, [&](const auto &candidate) {
          return candidate.gpu_id == gpu_id && candidate.gpu_va == alloc.gpu_va &&
                 candidate.extent == alloc.size && candidate.backing_id == owner->backing_id() &&
                 candidate.admission_generation == *generation;
        });
        if (found != aliases->end())
          alias = *found;
        else if (local)
          return -EOPNOTSUPP;
      }
    }

    const auto allocation_matches = [&](const KfdProcess::GpuAllocation &alloc) {
      const auto grant = alloc.admission_generations.find(gpu_id);
      return !alloc.retirement_in_progress && alloc.mapped_device_ids.contains(gpu_id) &&
             (grant == alloc.admission_generations.end()
                  ? !generation.has_value()
                  : generation && grant->second == *generation) &&
             (!alias ||
              ((alias->local ? alloc.local_backing_owner : alloc.backing_owner) &&
               (alias->local ? alloc.local_backing_owner : alloc.backing_owner)->backing_id() ==
                   alias->backing_id &&
               alloc.gpu_va == alias->gpu_va && alloc.size == alias->extent));
    };
    int retirement_result = 0;
    const auto retire = [&] {
      std::lock_guard lock(proc.alloc_mutex_);
      const auto it = proc.allocations_.find(args->handle);
      if (it == proc.allocations_.end()) {
        retirement_result = -EINVAL;
        return;
      }
      auto &alloc = it->second;
      if (!allocation_matches(alloc)) {
        retirement_result = -EAGAIN;
        return;
      }

      auto va_transaction = proc.begin_page_table_transaction();
      const bool last_grant = alloc.mapped_device_ids.size() == 1;
      if (last_grant && alloc.page_table_mapped_bytes &&
          proc.has_foreign_resident_overlap(
              va_transaction, alloc.gpu_va, alloc.page_table_mapped_bytes,
              {KfdProcess::VaOwnerKind::Kfd, 0}) != KfdProcess::PageTableMutationResult::Applied) {
        retirement_result = -EINVAL;
        return;
      }
      const bool previously_unstable = alloc.private_gtt_copy_source_unstable;
      if (alloc.private_gtt_copy_source_gate)
        alloc.private_gtt_copy_source_unstable = true;
      // For a canonical alias, the closed gate already excludes new ingress.
      // Finish a last-grant PTE removal before changing the grant: if a partial
      // page unmap throws, the ioctl leaves its ordered prefix unchanged and
      // the old generation remains closed.
      if (alias && last_grant && alloc.page_table_mapped_bytes) {
        if (unmap_from_gpu(proc, alloc.gpu_va, alloc.page_table_mapped_bytes, &va_transaction) !=
            KfdProcess::PageTableMutationResult::Applied) {
          retirement_result = -EINVAL;
          return;
        }
        alloc.page_table_mapped_bytes = 0;
      }

      // Preserve the old nodes so a failed grant-index allocation can restore
      // the allocation without allocating again. Publication happens only
      // after both nodes have been removed.
      auto mapped_node = alloc.mapped_device_ids.extract(gpu_id);
      auto generation_node = alloc.admission_generations.extract(gpu_id);
      try {
        if (alias && fail_next_alias_unmap_publish_.exchange(false, std::memory_order_acq_rel))
          throw std::bad_alloc();
        proc.publish_gpu_grants_locked(va_transaction);
      } catch (...) {
        alloc.mapped_device_ids.insert(std::move(mapped_node));
        if (!generation_node.empty())
          alloc.admission_generations.insert(std::move(generation_node));
        // Noncanonical UNMAP has not touched PTEs yet, and publication is
        // atomic; restoring its grant nodes restores the complete old view.
        if (!alias)
          alloc.private_gtt_copy_source_unstable = previously_unstable;
        throw;
      }
      if (!alias && last_grant && alloc.page_table_mapped_bytes) {
        if (unmap_from_gpu(proc, alloc.gpu_va, alloc.page_table_mapped_bytes, &va_transaction) !=
            KfdProcess::PageTableMutationResult::Applied) {
          retirement_result = -EINVAL;
          return;
        }
        alloc.page_table_mapped_bytes = 0;
      }
      args->n_success = i + 1;
      alloc.private_gtt_copy_source_unstable = previously_unstable;
    };

    if (!alias) {
      retire();
      if (retirement_result != 0)
        return retirement_result;
      continue;
    }

    const auto writeback_then_retire = [&](const GpuAliasIngressGate::DrainAuthority &authority) {
      // Check the allocation and last-grant PTE ownership before cache effects.
      // No KFD allocation or VM transaction lock crosses maintenance.
      {
        std::lock_guard lock(proc.alloc_mutex_);
        const auto it = proc.allocations_.find(args->handle);
        if (it == proc.allocations_.end()) {
          retirement_result = -EINVAL;
          throw std::runtime_error("canonical UNMAP allocation disappeared before writeback");
        }
        if (!allocation_matches(it->second)) {
          retirement_result = -EAGAIN;
          throw std::runtime_error("canonical UNMAP allocation changed before writeback");
        }
        const auto &alloc = it->second;
        if (alloc.mapped_device_ids.size() == 1 && alloc.page_table_mapped_bytes) {
          auto va_transaction = proc.begin_page_table_transaction();
          if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va,
                                                alloc.page_table_mapped_bytes,
                                                {KfdProcess::VaOwnerKind::Kfd, 0}) !=
              KfdProcess::PageTableMutationResult::Applied) {
            retirement_result = -EINVAL;
            throw std::runtime_error("canonical UNMAP PTE ownership changed before writeback");
          }
        }
      }

      const std::optional<bool> done = alias_ingress_gate_->final_writeback_done(authority);
      if (!done)
        throw std::runtime_error("canonical UNMAP drain authority expired");
      if (!*done) {
        if (alias->local &&
            fail_next_local_unmap_writeback_.exchange(false, std::memory_order_acq_rel))
          throw std::bad_alloc();
        const std::array selected_aliases{*alias};
        const auto domains = maintenance_domains_for_aliases(selected_aliases);
        const auto resolve = [this, &authority](uint32_t domain_gpu_id, uint32_t vmid,
                                                uint64_t address, size_t size) {
          return prepare_gpu_alias_writeback_for_vmid(domain_gpu_id, vmid, address, size,
                                                      &authority);
        };
        std::optional<amdgpu::DeviceCacheAliasScope> selected;
        // Current canonical grants are UC. When this closed generation is the
        // only live alias of its backing lines, no private cache can observe
        // the selected writeback after retirement. Retained dirty tags from
        // retired grants are checked independently by scoped maintenance.
        if (!alias->local && has_exclusive_canonical_lines(selected_aliases))
          selected = scoped_canonical_alias(proc, *alias);
        auto leases = [&]() {
          if (selected) {
            try {
              return amdgpu::DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
                  domains, std::span<const amdgpu::DeviceCacheAliasScope>(&*selected, 1), resolve);
            } catch (const util::Exception &) {
              // An unknown or overlapping dirty tag keeps the existing
              // whole-group path as the conservative maintenance route.
            }
          }
          return amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
              domains, amdgpu::DeviceCacheOperation::WritebackInvalidate, resolve);
        }();
        // Release in reverse lock order while the old grant and PTE remain.
        while (!leases.empty())
          leases.pop_back();
        if (!alias_ingress_gate_->mark_final_writeback_done(authority))
          throw std::runtime_error("canonical UNMAP lost its drained generation");
      }

      if (alias->local && fail_next_local_unmap_retire_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();

      retire();
      if (retirement_result != 0)
        throw std::runtime_error("canonical UNMAP retirement failed");
    };

    // The gate posts exactly one continuation after the last permit leaves.
    // Its poster only moves a prepared function into this one-shot slot; it
    // cannot allocate, throw, or execute maintenance on a CU/SDMA thread.
    std::function<void()> queued_continuation;
    std::atomic<bool> queued{false};
    const auto owner_post = [&](std::function<void()> continuation) noexcept {
      queued_continuation = std::move(continuation);
      queued.store(true, std::memory_order_release);
      queued.notify_one();
    };
    try {
      const auto key = GpuAliasIngressGate::key(*alias);
      if (!alias_ingress_gate_->close_with_drain_authority(key, owner_post,
                                                           writeback_then_retire) &&
          !alias_ingress_gate_->retry_closed_with_drain_authority(key, owner_post,
                                                                  writeback_then_retire))
        return -EAGAIN;
    } catch (const std::bad_alloc &) {
      return -ENOMEM;
    } catch (...) {
      return -EIO;
    }

    // dispatch_ioctl still holds op_mutex_, but neither alloc_mutex_ nor
    // process_mutex_ is held while an entered CU/SDMA operation drains.
    queued.wait(false, std::memory_order_acquire);
    try {
      queued_continuation();
    } catch (const std::bad_alloc &) {
      // The gate deliberately stays closed if maintenance did not complete.
      return -ENOMEM;
    } catch (...) {
      return retirement_result != 0 ? retirement_result : -EIO;
    }
    if (!reopen_gpu_alias_ingress(*alias))
      return -EIO;
    if (retirement_result != 0)
      return retirement_result;
  }
  return 0;
}

int SimulatedKfd::create_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_create_queue_args *>(arg);
  if (args->metadata_ring_size != 0)
    return -EINVAL;
  const uint32_t scheduling_percentage = args->queue_percentage & 0xff;
  if (scheduling_percentage > 100)
    return -EINVAL;
  auto *gpu = find_gpu(args->gpu_id);
  if (!gpu || !gpu->soc)
    return -EINVAL;

  const bool is_pm4_compute = args->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE;
  const bool is_aql_compute = args->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
  const bool is_sdma = args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA ||
                       args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA_XGMI ||
                       args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID;
  if (!is_aql_compute && !is_pm4_compute && !is_sdma)
    return -ENOTSUP;

  const std::optional<uint32_t> ring_size = normalize_queue_ring_size(args->ring_size);
  if (!ring_size)
    return -EINVAL;
  args->ring_size = *ring_size;

  // Queue IDs are process-local and start at one. Equivalent runtime queues in
  // different processes therefore share XCD resources while each process still
  // distributes additional queues across the device.
  const uint32_t queue_ordinal = proc.next_queue_id_ - 1;
  // KFD reserves percentage bits 8..15 for a native PM4 queue's target XCC.
  // An AQL vendor IB inherits the AQL queue's replicas instead of this placement.
  const uint32_t owner_ordinal =
      is_pm4_compute ? (args->queue_percentage >> 8) & 0xff : queue_ordinal;
  if (is_pm4_compute && owner_ordinal >= gpu->soc->num_xcds())
    return -EINVAL;
  amdgpu::CommandProcessor *target_cp =
      is_sdma ? nullptr : gpu->soc->assign_queue_owner_cp(owner_ordinal);
  if (!is_sdma && target_cp == nullptr)
    return -EINVAL;
  const uint32_t target_xcc_id =
      is_sdma ? args->sdma_engine_id : gpu->soc->queue_xcd_id(owner_ordinal);

  // Build the HW queue and reserve all per-process state under alloc_mutex_, then
  // register it with the CommandProcessor with the lock RELEASED. The CP thread
  // takes alloc_mutex_ under hw_queue_mutex_ (allocate_scratch_backing), so holding
  // alloc_mutex_ across register_queue() — which takes hw_queue_mutex_ — would be
  // an alloc_mutex_->hw_queue_mutex_ inversion against that thread and can deadlock.
  // op_mutex_ already serializes all ioctls for this process, so no concurrent
  // ioctl can observe the partially-registered queue in the window between the
  // unlock and register_queue().
  amdgpu::QueueRegistrationRequest queue_request{};
  uint32_t queue_id = 0;
  std::optional<DoorbellReservation> doorbell_reservation;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);

    if (!daemon_mode_) {
      auto va_transaction = proc.begin_page_table_transaction();
      const auto kfd_owner = KfdProcess::VaOwner{KfdProcess::VaOwnerKind::Kfd, 0};
      const auto clear = [&](uint64_t address, size_t bytes) {
        return proc.has_foreign_resident_overlap(va_transaction, address, bytes, kfd_owner) ==
               KfdProcess::PageTableMutationResult::Applied;
      };
      if (!clear(args->ring_base_address, args->ring_size) ||
          !clear(args->read_pointer_address, sizeof(uint64_t)) ||
          (args->write_pointer_address != args->read_pointer_address &&
           !clear(args->write_pointer_address, sizeof(uint64_t))))
        return -EINVAL;
      if (map_to_gpu(proc, args->ring_base_address,
                     reinterpret_cast<void *>(args->ring_base_address), args->ring_size,
                     amdgpu::Mtype::UC, KfdProcess::HostExtentOwner::Application,
                     &va_transaction) != KfdProcess::PageTableMutationResult::Applied)
        return -EINVAL;
      if (map_to_gpu(proc, args->read_pointer_address,
                     reinterpret_cast<void *>(args->read_pointer_address), sizeof(uint64_t),
                     amdgpu::Mtype::UC, KfdProcess::HostExtentOwner::Application,
                     &va_transaction) != KfdProcess::PageTableMutationResult::Applied)
        return -EINVAL;
      if (args->write_pointer_address != args->read_pointer_address)
        if (map_to_gpu(proc, args->write_pointer_address,
                       reinterpret_cast<void *>(args->write_pointer_address), sizeof(uint64_t),
                       amdgpu::Mtype::UC, KfdProcess::HostExtentOwner::Application,
                       &va_transaction) != KfdProcess::PageTableMutationResult::Applied)
          return -EINVAL;
    }

    queue_id = proc.next_queue_id_++;
    uint32_t ord = gpu_ordinal(args->gpu_id);
    auto &gs = proc.gpu(ord);
    uint32_t db_offset;
    bool recycled_offset = false;
    if (!gs.free_doorbell_offsets.empty()) {
      db_offset = gs.free_doorbell_offsets.back();
      gs.free_doorbell_offsets.pop_back();
      recycled_offset = true;
    } else {
      if (gs.doorbell_page_size > 0 &&
          gs.next_doorbell_offset + sizeof(uint64_t) > gs.doorbell_page_size)
        return -ENOSPC;
      db_offset = static_cast<uint32_t>(gs.next_doorbell_offset);
      gs.next_doorbell_offset += sizeof(uint64_t);
    }
    doorbell_reservation.emplace(proc.alloc_mutex_, gs.free_doorbell_offsets,
                                 gs.next_doorbell_offset, db_offset, recycled_offset);

    // Reset a recycled doorbell slot to the ~0 sentinel. The mmap-time 0xFF fill
    // only primes freshly-mapped pages; a slot freed by destroy_queue() still
    // holds the prior queue's last-rung write index (typically a small value like
    // 0). The CP starts every queue with last_doorbell==~0, so if the poll thread
    // scans this slot in the window between register_queue() and the host's first
    // ring, it latches that stale value as last_doorbell. When the host then rings
    // the new queue with the same value (write_index 0 for a one-packet queue),
    // val==last_doorbell, no edge is detected, and the submission is never fetched
    // — a lost doorbell that hangs the waiter in hsa_signal_wait. Restoring the
    // sentinel keeps the "first real ring is always an edge" invariant.
    if (recycled_offset && gs.doorbell_monitor_page &&
        db_offset + sizeof(uint64_t) <= gs.doorbell_page_size) {
      std::atomic_ref<uint64_t>(
          *reinterpret_cast<uint64_t *>(static_cast<char *>(gs.doorbell_monitor_page) + db_offset))
          .store(~uint64_t(0), std::memory_order_release);
    }

    queue_request.identity = {.address_space = gs.address_space,
                              .interrupt_sink = gpu->interrupt_subscription.sink(),
                              .process_id = proc.process_id(),
                              .queue_id = queue_id};
    queue_request.ring = {.base_address = args->ring_base_address,
                          .size_bytes = args->ring_size,
                          .consumer_pointer_address = args->read_pointer_address,
                          .producer_pointer_address = args->write_pointer_address};
    if (is_sdma) {
      const std::weak_ptr<CuIngressAccess> ingress_access = cu_ingress_access_;
      const std::shared_ptr<amdgpu::DeviceCacheCoherence> cache_coherence =
          gpu->soc->cache_coherence();
      CacheWritebackResolver resolve_writeback = cache_writeback_resolver();
      queue_request.binding_factory = amdgpu::make_sdma_queue_binding_factory(
          gpu->soc->sdma_queue_scheduler(), [ingress_access, cache_coherence, gpu_id = gpu->gpu_id,
                                             resolve_writeback = std::move(resolve_writeback)](
                                                const amdgpu::SdmaQueueContext &context) {
            amdgpu::SdmaPacketCallbacks callbacks;
            callbacks.acquire_cache_maintenance =
                [ingress_access, cache_coherence, gpu_id,
                 resolve_writeback](amdgpu::SdmaCacheOperation operation) {
                  const std::shared_ptr<CuIngressAccess> access = ingress_access.lock();
                  if (!access)
                    throw util::Exception(std::string_view("KFD SDMA cache owner is gone"));
                  {
                    std::lock_guard lock(access->mutex);
                    if (!access->driver)
                      throw util::Exception(std::string_view("KFD SDMA cache owner is gone"));
                  }
                  // Resolution reenters CuIngressAccess after dropping all cache
                  // locks. Never hold its mutex across group acquisition.
                  const std::array domains{amdgpu::DeviceCacheDomain{cache_coherence, gpu_id}};
                  auto leases = amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
                      domains, operation, resolve_writeback);
                  if (leases.size() != domains.size())
                    throw util::Exception(std::string_view("KFD SDMA cache lease is missing"));
                  return std::move(leases.front());
                };
            callbacks.admit_memory_ingress = [ingress_access, gpu_id,
                                              process_id = context.process_id](
                                                 const amdgpu::GpuVmAccess &expected,
                                                 std::span<const amdgpu::MemoryIngressSpan> spans) {
              const std::shared_ptr<CuIngressAccess> access = ingress_access.lock();
              if (!access)
                return amdgpu::MemoryIngressAdmission{.outcome = amdgpu::VmAccessOutcome::Revoked,
                                                      .permit = {}};
              std::lock_guard lock(access->mutex);
              if (!access->driver)
                return amdgpu::MemoryIngressAdmission{.outcome = amdgpu::VmAccessOutcome::Revoked,
                                                      .permit = {}};
              return access->driver->admit_gpu_alias_ingress(process_id, gpu_id, expected, spans);
            };
            callbacks.admit_private_uc_gtt_copy_source =
                [ingress_access, gpu_id, process_id = context.process_id](
                    const amdgpu::GpuVmAccess &expected, uint64_t address, std::size_t size) {
                  const std::shared_ptr<CuIngressAccess> access = ingress_access.lock();
                  if (!access)
                    return amdgpu::VmPrivateUcGttCopySourceAdmission{
                        .outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
                  std::lock_guard lock(access->mutex);
                  if (!access->driver)
                    return amdgpu::VmPrivateUcGttCopySourceAdmission{
                        .outcome = amdgpu::VmAccessOutcome::Revoked, .permit = {}};
                  return access->driver->admit_private_uc_gtt_copy_source(process_id, gpu_id,
                                                                          expected, address, size);
                };
            return callbacks;
          });
    } else {
      queue_request.binding_factory = amdgpu::make_compute_queue_binding_factory(*target_cp);
    }
    queue_request.engine_id = args->sdma_engine_id;
    queue_request.scheduling_percentage = scheduling_percentage;
    // doorbell_base is captured here under alloc_mutex_ but register_queue() runs
    // after the lock is released. This is stable because ROCr maps the doorbell
    // page before creating queues, and queue creation for a process is single-
    // threaded (serialized by op_mutex_), so no concurrent dispatch_mmap re-maps
    // the doorbell in the unlock->register window.
    assert(gs.doorbell_views.empty() || gs.doorbell_monitor_page);
    queue_request.doorbell = {.mode = amdgpu::QueueDoorbellMode::HostPolled,
                              .offset = db_offset,
                              .host_base = gs.doorbell_monitor_page,
                              .last_value = ~uint64_t(0)};
    queue_request.type = is_sdma ? amdgpu::QueueType::Sdma : amdgpu::QueueType::Compute;
    queue_request.packet_format = is_sdma          ? amdgpu::QueuePacketFormat::Sdma
                                  : is_pm4_compute ? amdgpu::QueuePacketFormat::Pm4
                                                   : amdgpu::QueuePacketFormat::Aql;
    queue_request.abi = is_aql_compute ? amdgpu::QueueAbi::KfdAql : amdgpu::QueueAbi::Generic;
    // Fresh SDMA and native PM4 MQDs start at zero; the writeback destination
    // need not contain the initial hardware cursor before the first retirement.
    if (is_sdma || is_pm4_compute)
      queue_request.initial_consumer_cursor = 0;
    // The topology advertises every XCD's compute units as one agent, so a
    // compute dispatch must be able to reach all of them. Without this a
    // single-queue application would only ever use the XCD that
    // assign_queue_owner_cp() happened to pick.
    //
    // Named types only, rather than "anything that is not SDMA". SDMA queues are
    // per-engine and are not spread, but an unrecognized queue_type is not
    // thereby a compute queue, and device-wide replication should not be what an
    // unsupported value silently acquires.
    queue_request.xcd_fanout = is_aql_compute;
    // amd_queue_t base: write_pointer_address points to write_dispatch_id.
    if (is_aql_compute)
      queue_request.queue_descriptor_address =
          args->write_pointer_address - offsetof(amd_queue_t, write_dispatch_id);
    if (is_aql_compute && args->ctx_save_restore_address != 0) {
      constexpr uint32_t kErrorReasonOffset = 6 * sizeof(uint32_t);
      const std::optional<amdgpu::GpuVmAccess> access =
          gpu->soc->gpu_vm().snapshot(gs.address_space);
      if (!access)
        return -EFAULT;
      std::array<std::byte, 2 * sizeof(uint64_t)> exception_fields{};
      if (access->read(args->ctx_save_restore_address + kErrorReasonOffset, exception_fields) !=
          amdgpu::VmAccessOutcome::Complete) {
        return -EFAULT;
      }
      uint64_t exception_status = 0;
      uint64_t exception_event = 0;
      std::memcpy(&exception_status, exception_fields.data(), sizeof(exception_status));
      std::memcpy(&exception_event, exception_fields.data() + sizeof(exception_status),
                  sizeof(exception_event));
      queue_request.exception_status_address = exception_status;
      queue_request.exception_event_id = static_cast<uint32_t>(exception_event);
    }
    if (is_sdma && !daemon_mode_) {
      auto *wptr = reinterpret_cast<uint64_t *>(args->write_pointer_address);
      auto *rptr = reinterpret_cast<uint64_t *>(args->read_pointer_address);
      util::Logger::vm("SDMA wptr before init: addr=0x", std::hex, args->write_pointer_address,
                       " val=", std::dec, *wptr, " rptr val=", *rptr);
      *wptr = 0;
      *rptr = 0;
    } else if (is_sdma && daemon_mode_) {
      if (gpu->soc && gs.address_space) {
        const uint64_t zero = 0;
        const auto bytes = std::as_bytes(std::span<const uint64_t>(&zero, 1));
        if (gpu->soc->gpu_vm().write(gs.address_space, args->write_pointer_address, bytes) !=
                amdgpu::VmAccessOutcome::Complete ||
            gpu->soc->gpu_vm().write(gs.address_space, args->read_pointer_address, bytes) !=
                amdgpu::VmAccessOutcome::Complete) {
          return -EFAULT;
        }
      }
    }
  }

  // Register with the shared queue registry OUTSIDE alloc_mutex_ (see note above).
  amdgpu::QueueHandle queue_handle;
  const auto rollback_published_queue = [&] {
    if (queue_handle)
      (void)gpu->soc->queue_registry().unregister_queue(queue_handle,
                                                        amdgpu::QueueCloseMode::ForceCancel);
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    std::erase(proc.active_queue_ids_, queue_id);
    proc.queue_doorbell_map_.erase(queue_id);
    proc.queue_snapshot_map_.erase(queue_id);
  };
  try {
    queue_handle = gpu->soc->queue_registry().register_queue(queue_request);
    if (!queue_handle)
      return -EINVAL;

    // Publish debug metadata only after queue binding registration. A cross-process
    // debugger does not hold the target's op_mutex_, so publishing it earlier
    // could expose a queue before its execution owner can service it.
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    proc.active_queue_ids_.push_back(queue_id);
    proc.queue_doorbell_map_[queue_id] = {
        .gpu_ordinal = gpu_ordinal(args->gpu_id),
        .doorbell_offset = queue_request.doorbell.offset,
        .queue_handle = queue_handle,
        .pm4_target_xcc = is_pm4_compute ? std::optional(owner_ordinal) : std::nullopt,
    };
    proc.queue_snapshot_map_[queue_id] = {
        .ring_base_address = args->ring_base_address,
        .write_pointer_address = args->write_pointer_address,
        .read_pointer_address = args->read_pointer_address,
        .ctx_save_restore_address = args->ctx_save_restore_address,
        .ctx_save_restore_area_size = args->ctx_save_restore_size,
        .ring_size = args->ring_size,
        .queue_type = args->queue_type,
        .gpu_id = args->gpu_id,
        // The gfx1250 debugger ABI publishes each queue in its owning XCC
        // save area. Preserve the legacy single-area model for older targets.
        .xcc_id = gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5 ? target_xcc_id : 0,
        .has_runtime_exception_signal = queue_request.exception_status_address != 0,
        .exception_status = KFD_EC_MASK(EC_QUEUE_NEW),
        .debug_notification_retained_status = KFD_EC_MASK(EC_QUEUE_NEW),
        .debug_notification_events = {KFD_EC_MASK(EC_QUEUE_NEW)},
    };
  } catch (const std::bad_alloc &) {
    rollback_published_queue();
    return -ENOMEM;
  } catch (...) {
    rollback_published_queue();
    return -EIO;
  }

  doorbell_reservation->commit();
  args->queue_id = queue_id;
  args->doorbell_offset =
      KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(gpu->gpu_id) | queue_request.doorbell.offset;
  return 0;
}

int SimulatedKfd::set_cu_mask_ioctl(KfdProcess &proc, void *arg) {
  const kfd_ioctl_set_cu_mask_args *args = static_cast<const kfd_ioctl_set_cu_mask_args *>(arg);
  if (!args || args->num_cu_mask == 0 || args->num_cu_mask % 32 != 0)
    return -EINVAL;
  if (args->cu_mask_ptr == 0)
    return -EFAULT;
  // KFD clips masks to 1024 bits before copying them from userspace.
  const uint32_t bit_count = std::min(args->num_cu_mask, 1024u);
  std::vector<uint32_t> mask(bit_count / 32);
  if (copy_ioctl_user_buffer(mask.data(), args->cu_mask_ptr, mask.size() * sizeof(uint32_t)) != 0)
    return -EFAULT;

  uint32_t gpu_id = 0;
  bool is_sdma = false;
  {
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    const auto queue = proc.queue_snapshot_map_.find(args->queue_id);
    if (queue == proc.queue_snapshot_map_.end())
      return -EFAULT;
    const uint32_t type = queue->second.queue_type;
    is_sdma = type == KFD_IOC_QUEUE_TYPE_SDMA || type == KFD_IOC_QUEUE_TYPE_SDMA_XGMI ||
              type == KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID;
    if (!is_sdma && type != KFD_IOC_QUEUE_TYPE_COMPUTE && type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
      return -EINVAL;
    gpu_id = queue->second.gpu_id;
  }
  GpuDevice *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc)
    return -EINVAL;
  const bool rdna = arch_is_rdna(gpu->soc->arch());
  if (rdna) {
    // KFD requires both CUs in each WGP to have the same enable bit, even
    // beyond the active CU count. Reject partial pairs before updating queues.
    for (uint32_t bit = 0; bit < bit_count; bit += 2) {
      const uint32_t pair = (mask[bit / 32] >> (bit % 32)) & 3u;
      if (pair != 0 && pair != 3)
        return -EINVAL;
    }
  }
  // KFD validates SDMA masks too, but its SDMA MQD update ignores them.
  if (is_sdma)
    return 0;

  const Sysfs::GpuInfo &info = gpu_infos_[gpu_ordinal(gpu_id)];
  const uint32_t arrays = std::max(1u, info.num_shader_arrays_per_engine);
  const uint32_t cu_per_array = info.num_cu_per_sh;
  const uint32_t active_cus = info.simd_count / std::max(1u, info.simd_per_cu);
  const uint32_t step = rdna ? 2u : 1u;
  amdgpu::QueueCuSelection selected(std::in_place);
  uint32_t bit = 0;
  // KFD numbers the logical mask symmetrically: XCC, SE, SH, then CU.
  // RDNA enables sibling CU pairs, so each pair stays adjacent in the mask.
  for (uint32_t cu = 0; cu < cu_per_array && bit < bit_count && bit < active_cus; cu += step) {
    for (uint32_t sh = 0; sh < arrays; ++sh) {
      for (uint32_t se = 0; se < info.num_shader_engines; ++se) {
        for (amdgpu::Xcd *xcd : gpu->soc->xcds()) {
          if (se >= xcd->num_shader_engines())
            continue;
          amdgpu::ShaderEngine *shader = xcd->shader_engine(se);
          const uint32_t local = sh * cu_per_array + cu;
          if (local + step > shader->num_compute_units())
            continue;
          const bool enabled = bit < bit_count && bit < active_cus &&
                               ((mask[bit / 32] >> (bit % 32)) & ((1u << step) - 1));
          for (uint32_t half = 0; half < step; ++half, ++bit) {
            if (enabled)
              selected->push_back(shader->compute_unit(local + half));
          }
        }
      }
    }
  }
  // Never hold alloc_mutex_ while taking a CP's queue lock.
  gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
    cp->set_queue_cu_selection(args->queue_id, proc.process_id(), selected);
  });
  return 0;
}

bool SimulatedKfd::debugger_supports_gws(uint32_t gpu_id) const {
  const auto &info = gpu_infos_[gpu_ordinal(gpu_id)];
  const uint32_t gc = kmd::gc_ip_version_for_gfx_target_version(info.gfx_target_version);
  // Advertising 64 GWS resources assumes firmware supporting GWS and its debugger
  // save/restore. fw_version is an unmodeled topology placeholder, not MEC2 firmware.
  // Retain the firmware-independent CWSR restriction from kfd_dbg_has_gws_support().
  return !(gc >= kmd::make_gc_ip_version(11, 0, 0) && gc <= kmd::make_gc_ip_version(11, 0, 3));
}

int SimulatedKfd::alloc_queue_gws_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_alloc_queue_gws_args *>(arg);
  // Same lock order as debugger attach, making the two admission checks atomic.
  std::lock_guard session_lock(debug_sessions_mutex_);
  std::lock_guard lock(proc.alloc_mutex_);
  auto queue = proc.queue_snapshot_map_.find(args->queue_id);
  if (queue == proc.queue_snapshot_map_.end())
    return -EINVAL;
  auto &info = queue->second;
  if (info.queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE &&
      info.queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
    return -EINVAL;
  const auto session = debug_sessions_.find(proc.client_pid());
  if (session != debug_sessions_.end() && session->second.enabled &&
      !debugger_supports_gws(info.gpu_id))
    return -EBUSY;
  args->first_gws = 0;
  if (args->num_gws == 0) {
    if (!info.gws_allocated)
      return -EINVAL;
    info.gws_allocated = false;
  } else {
    // KFD grants the whole namespace to one queue on each process/GPU pair.
    // GwsDevice documents the instruction model's relaxed ownership checks.
    for (const auto &[_, other] : proc.queue_snapshot_map_) {
      if (other.gpu_id == info.gpu_id && other.gws_allocated)
        return -EBUSY;
    }
    info.gws_allocated = true;
  }
  args->first_gws = 0;
  return 0;
}

int SimulatedKfd::update_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_update_queue_args *>(arg);
  // Decode scheduling independently of the native PM4 placement field.
  const uint32_t scheduling_percentage = args->queue_percentage & 0xff;
  if (scheduling_percentage > 100)
    return -EINVAL;
  const std::optional<uint32_t> ring_size = normalize_queue_ring_size(args->ring_size);
  if (!ring_size)
    return -EINVAL;

  std::optional<KfdProcess::QueueDoorbellInfo> queue;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    const std::unordered_map<uint32_t, KfdProcess::QueueDoorbellInfo>::iterator found =
        proc.queue_doorbell_map_.find(args->queue_id);
    if (found != proc.queue_doorbell_map_.end() &&
        proc.queue_snapshot_map_.contains(args->queue_id)) {
      queue = found->second;
    }
  }
  if (!queue || queue->gpu_ordinal >= gpus_.size() || !gpus_[queue->gpu_ordinal].soc)
    return -EFAULT;

  if (queue->pm4_target_xcc) {
    const uint32_t target_xcc = (args->queue_percentage >> 8) & 0xff;
    if (target_xcc >= gpus_[queue->gpu_ordinal].soc->num_xcds())
      return -EINVAL;
    // Moving a live CP record requires migrating its registers and pending streams.
    if (target_xcc != *queue->pm4_target_xcc)
      return -EOPNOTSUPP;
  }

  amdgpu::QueueReconfigureResult update;
  try {
    update = gpus_[queue->gpu_ordinal].soc->queue_registry().reconfigure_queue(
        queue->queue_handle, {.ring_base_address = args->ring_base_address,
                              .ring_size_bytes = *ring_size,
                              .scheduling_percentage = scheduling_percentage});
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return -EIO;
  }
  if (!update.found || update.status == amdgpu::QueueReconfigureStatus::Stale)
    return -EFAULT;
  if (update.status == amdgpu::QueueReconfigureStatus::Busy)
    return -EBUSY;
  if (update.status == amdgpu::QueueReconfigureStatus::Invalid)
    return -EINVAL;

  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    const std::unordered_map<uint32_t, KfdProcess::QueueSnapshotInfo>::iterator snapshot =
        proc.queue_snapshot_map_.find(args->queue_id);
    assert(snapshot != proc.queue_snapshot_map_.end());
    snapshot->second.ring_base_address = args->ring_base_address;
    snapshot->second.ring_size = *ring_size;
  }
  return 0;
}

int SimulatedKfd::destroy_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_destroy_queue_args *>(arg);
  std::optional<KfdProcess::QueueDoorbellInfo> queue;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    std::erase(proc.active_queue_ids_, args->queue_id);
    proc.queue_snapshot_map_.erase(args->queue_id);
    auto it = proc.queue_doorbell_map_.find(args->queue_id);
    if (it != proc.queue_doorbell_map_.end()) {
      queue = it->second;
      auto &gs = proc.gpu(it->second.gpu_ordinal);
      gs.free_doorbell_offsets.push_back(it->second.doorbell_offset);
      proc.queue_doorbell_map_.erase(it);
    }
  }
  if (queue && queue->gpu_ordinal < gpus_.size() && gpus_[queue->gpu_ordinal].soc) {
    (void)gpus_[queue->gpu_ordinal].soc->queue_registry().unregister_queue(
        queue->queue_handle, amdgpu::QueueCloseMode::ForceCancel);
  }
  // Real CP sends EOP interrupt when queue is deactivated; KFD broadcasts to
  // all type-0 events. This wakes ROCR's signal threads blocked on queue events.
  proc.event_state_.signal_interrupt(0);
  return 0;
}

int SimulatedKfd::set_memory_policy_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_set_memory_policy_args *>(arg);
  if (!find_gpu(args->gpu_id))
    return -EINVAL;
  KfdProcess::MemoryPolicy policy{};
  policy.alternate_base = args->alternate_aperture_base;
  policy.alternate_size = args->alternate_aperture_size;
  policy.default_policy = args->default_policy;
  policy.alternate_policy = args->alternate_policy;
  proc.memory_policies_[args->gpu_id] = policy;
  return 0;
}

int SimulatedKfd::import_dmabuf_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_import_dmabuf_args *>(arg);
  if (!find_gpu(args->gpu_id))
    return -EINVAL;

  struct stat st {};
  if (safe_fstat(args->dmabuf_fd, &st) != 0)
    return -errno;
  if (st.st_size <= 0 || args->va_addr > UINT64_MAX - static_cast<uint64_t>(st.st_size))
    return -EINVAL;
  uint64_t size = static_cast<uint64_t>(st.st_size);

  int dupfd = safe_fcntl(args->dmabuf_fd, F_DUPFD_CLOEXEC, 0);
  if (dupfd < 0)
    return -errno;

  uint64_t handle;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    auto va_transaction = proc.begin_page_table_transaction();
    if (args->va_addr && proc.has_foreign_resident_overlap(va_transaction, args->va_addr, size,
                                                           {KfdProcess::VaOwnerKind::Kfd, 0}) !=
                             KfdProcess::PageTableMutationResult::Applied) {
      libc_passthrough().close(dupfd);
      return -EINVAL;
    }
    handle = proc.next_handle_++;
    KfdProcess::GpuAllocation alloc{};
    alloc.gpu_va = args->va_addr;
    alloc.size = size;
    alloc.gpu_id = args->gpu_id;
    // Import currently models unknown backing as GTT. IPC re-export preserves
    // these placeholder flags; PRIME retains an existing BO's authoritative
    // flags when it recognizes the same backing.
    alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    alloc.handle = handle;
    alloc.gpu_id = args->gpu_id;
    alloc.user_va = true;
    alloc.imported = true;
    alloc.shared = true;
    alloc.dmabuf_fd = dupfd;
    alloc.host_ptr = reinterpret_cast<void *>(args->va_addr);
    proc.allocations_[handle] = alloc;
    proc.claim_gpu_range_locked(alloc.gpu_va, alloc.size);
    proc.publish_gpu_grants_locked(va_transaction);
    if (args->va_addr) {
      if (map_to_gpu(proc, args->va_addr, alloc.host_ptr, size, amdgpu::Mtype::UC,
                     KfdProcess::HostExtentOwner::Application,
                     &va_transaction) != KfdProcess::PageTableMutationResult::Applied) {
        proc.allocations_.erase(handle);
        proc.retire_gpu_range_locked(alloc.gpu_va, alloc.size);
        proc.publish_gpu_grants_locked(va_transaction);
        libc_passthrough().close(dupfd);
        return -EINVAL;
      }
      alloc.page_table_mapped_bytes = size;
    }
    proc.allocations_[handle] = alloc;

    KfdProcess::ImportedDmabuf info{};
    info.handle = handle;
    info.fd = dupfd;
    info.size = size;
    info.va = args->va_addr;
    info.gpu_id = args->gpu_id;
    proc.imported_dmabufs_[handle] = info;
    proc.fd_to_import_handle_[dupfd] = handle;
  }

  args->handle = handle;
  return 0;
}

int SimulatedKfd::retain_export_descriptor_locked(const KfdProcess::GpuAllocation &alloc) {
  const int backing = alloc.memfd >= 0 ? alloc.memfd : (alloc.imported ? alloc.dmabuf_fd : -1);
  if (backing < 0)
    return -EINVAL;
  const int exported =
      alloc.export_access_mode >= 0
          ? libc_passthrough().openat(AT_FDCWD, std::format("/proc/self/fd/{}", backing).c_str(),
                                      alloc.export_access_mode | O_CLOEXEC, 0)
          : safe_fcntl(backing, F_DUPFD_CLOEXEC, 0);
  return exported < 0 ? -errno : exported;
}

int SimulatedKfd::share_allocation_locked(KfdProcess &proc, KfdProcess::GpuAllocation &alloc) {
  // Doorbells use separate backing; USERPTR pages are not owned by the BO.
  if (alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)
    return -EPERM;
  if (alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
    return -EINVAL;
  // IPC may retain a writable private alias for a read-only public dma-buf.
  const int exported = retain_export_descriptor_locked(alloc);
  if (exported < 0)
    return exported;
  // Publish the KFD sharing policy only after retaining the export succeeds.
  // Keep it separate from requested flags returned by IPC_IMPORT_HANDLE.
  alloc.shared = true;
  proc.set_page_mtype(alloc.gpu_va, alloc.size, pte_mtype_for_flags(alloc.flags, alloc.shared));
  return exported;
}

int SimulatedKfd::export_dmabuf_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);
  ScopedPrivateGttCopySourceMutation copy_source_mutation(*this, proc, args->handle);

  std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end())
    return -EINVAL;
  const auto &alloc = it->second;
  if (alloc.retirement_in_progress)
    return -EAGAIN;
  if (alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)
    return -EPERM;
  if (alloc.cacheable_peer_for_testing ||
      (alloc.backing_owner && alloc.backing_owner->controlled_peer_for_testing()))
    return -EOPNOTSUPP;
  // Routed aliases retain their UC PTE policy; ordinary BO exports use the
  // existing sharing policy and public descriptor access mode.
  int dupfd = -EINVAL;
  if (alloc.backing_owner && alloc.memfd >= 0) {
    // A retained peer backing changes the PTE policy, not the public fd's
    // access mode. IPC imports can retain an O_RDONLY export.
    dupfd = retain_export_descriptor_locked(alloc);
  } else if (!alloc.backing_owner) {
    dupfd = share_allocation_locked(proc, it->second);
  }
  if (dupfd < 0)
    return dupfd;
  args->dmabuf_fd = dupfd;
  it->second.dmabuf_exported = true;
  return 0;
}

int SimulatedKfd::promote_local_alias_for_ipc(KfdProcess &proc, uint64_t handle,
                                              const KfdProcess::GpuAliasRecord &alias,
                                              std::unique_lock<std::mutex> &op_lock) {
  int transition_result = 0;
  const auto validate = [&]() {
    const auto it = proc.allocations_.find(handle);
    if (it == proc.allocations_.end()) {
      transition_result = -EINVAL;
      return false;
    }
    const auto &alloc = it->second;
    const auto generation = alloc.admission_generations.find(alias.gpu_id);
    if (alloc.retirement_in_progress || alloc.backing_promotion_attempted || alloc.backing_owner ||
        !alloc.local_backing_owner || alloc.local_backing_owner->backing_id() != alias.backing_id ||
        alloc.gpu_va != alias.gpu_va || alloc.size != alias.extent ||
        alloc.page_table_mapped_bytes != alloc.size || alloc.mapped_device_ids.size() != 1 ||
        !alloc.mapped_device_ids.contains(alias.gpu_id) ||
        generation == alloc.admission_generations.end() ||
        generation->second != alias.admission_generation || !proc.gpu_alias_grant_current(alias)) {
      transition_result = -EAGAIN;
      return false;
    }
    return true;
  };

  const auto writeback_then_publish = [&](const GpuAliasIngressGate::DrainAuthority &authority) {
    {
      std::lock_guard lock(proc.alloc_mutex_);
      if (!validate())
        throw std::runtime_error("IPC promotion lost its local grant before writeback");
      const auto &alloc = proc.allocations_.at(handle);
      auto va_transaction = proc.begin_page_table_transaction();
      if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va, alloc.size,
                                            {KfdProcess::VaOwnerKind::Kfd, 0}) !=
          KfdProcess::PageTableMutationResult::Applied) {
        transition_result = -EINVAL;
        throw std::runtime_error("IPC promotion lost PTE ownership before writeback");
      }
    }

    const std::optional<bool> done = alias_ingress_gate_->final_writeback_done(authority);
    if (!done)
      throw std::runtime_error("IPC promotion drain authority expired");
    if (!*done) {
      if (fail_next_alias_export_writeback_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();
      // Before the first canonical publication, a VRAM peer cannot hold a
      // MAP grant. Only the owner GPU can have cacheable resident lines for
      // this local BO; leave unrelated GPU domains and epochs alone.
      const GpuDevice *device = find_gpu(alias.gpu_id);
      if (!device || !device->soc)
        throw std::runtime_error("IPC promotion has no owner cache domain");
      const std::array domains{
          amdgpu::DeviceCacheDomain{device->soc->cache_coherence(), alias.gpu_id}};
      const auto resolve = [this, &authority](uint32_t gpu_id, uint32_t vmid, uint64_t address,
                                              size_t size) {
        return prepare_gpu_alias_writeback_for_vmid(gpu_id, vmid, address, size, &authority);
      };
      auto leases = amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
          domains, amdgpu::DeviceCacheOperation::WritebackInvalidate, resolve);
      while (!leases.empty())
        leases.pop_back();
      if (!alias_ingress_gate_->mark_final_writeback_done(authority))
        throw std::runtime_error("IPC promotion lost its writeback marker");
    }

    // The old key remains closed while its owner changes category. Both alias
    // categories use the same key, so no CU or SDMA request can enter between
    // the UC PTE update and canonical grant publication.
    std::lock_guard lock(proc.alloc_mutex_);
    if (!validate())
      throw std::runtime_error("IPC promotion lost its local grant before publication");
    auto &alloc = proc.allocations_.at(handle);
    auto va_transaction = proc.begin_page_table_transaction();
    if (proc.has_foreign_resident_overlap(va_transaction, alloc.gpu_va, alloc.size,
                                          {KfdProcess::VaOwnerKind::Kfd, 0}) !=
        KfdProcess::PageTableMutationResult::Applied) {
      transition_result = -EINVAL;
      throw std::runtime_error("IPC promotion lost PTE ownership before publication");
    }
    const amdgpu::Mtype old_mtype = pte_mtype_for_flags(alloc.flags);
    const amdgpu::Mtype new_mtype =
        alloc.cacheable_peer_for_testing ? old_mtype : amdgpu::Mtype::UC;
    if (proc.set_page_mtype(va_transaction, alloc.gpu_va, alloc.size, new_mtype) !=
        KfdProcess::PageTableMutationResult::Applied) {
      transition_result = -EINVAL;
      throw std::runtime_error("IPC promotion could not set the UC PTE");
    }
    const auto previous_backing = alloc.backing_owner;
    const auto previous_local = alloc.local_backing_owner;
    const bool previous_attempted = alloc.backing_promotion_attempted;
    const bool previous_shared = alloc.ipc_shared;
    alloc.backing_owner = previous_local;
    alloc.local_backing_owner.reset();
    alloc.backing_promotion_attempted = true;
    alloc.ipc_shared = true;
    try {
      if (fail_next_alias_export_publish_.exchange(false, std::memory_order_acq_rel))
        throw std::bad_alloc();
      proc.publish_gpu_grants_locked(va_transaction);
    } catch (...) {
      alloc.backing_owner = previous_backing;
      alloc.local_backing_owner = previous_local;
      alloc.backing_promotion_attempted = previous_attempted;
      alloc.ipc_shared = previous_shared;
      if (proc.set_page_mtype(va_transaction, alloc.gpu_va, alloc.size, old_mtype) !=
          KfdProcess::PageTableMutationResult::Applied)
        throw std::runtime_error("IPC promotion could not restore the old PTE");
      throw;
    }
  };

  std::function<void()> queued_continuation;
  std::atomic<bool> queued{false};
  const auto owner_post = [&](std::function<void()> continuation) noexcept {
    queued_continuation = std::move(continuation);
    queued.store(true, std::memory_order_release);
    queued.notify_one();
  };
  const auto key = GpuAliasIngressGate::key(alias);
  try {
    if (!alias_ingress_gate_->close_with_drain_authority(key, owner_post, writeback_then_publish) &&
        !alias_ingress_gate_->retry_closed_with_drain_authority(key, owner_post,
                                                                writeback_then_publish))
      return -EAGAIN;
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return -EIO;
  }
  // The publication writer excludes new KFD operations while the exact key
  // waits. Releasing the process operation lock lets an admitted engine finish
  // work that may need it before dropping its counted ingress permit.
  op_lock.unlock();
  queued.wait(false, std::memory_order_acquire);
  op_lock.lock();
  try {
    queued_continuation();
  } catch (const std::bad_alloc &) {
    return -ENOMEM;
  } catch (...) {
    return transition_result != 0 ? transition_result : -EIO;
  }
  return alias_ingress_gate_->reopen(key) ? 0 : -EIO;
}

int SimulatedKfd::ipc_export_handle_ioctl(KfdProcess &proc, void *arg,
                                          std::unique_lock<std::mutex> &op_lock) {
  auto *args = static_cast<kfd_ioctl_ipc_export_handle_args *>(arg);
  ScopedPrivateGttCopySourceMutation copy_source_mutation(*this, proc, args->handle);

  std::optional<KfdProcess::GpuAliasRecord> local_alias;
  {
    std::lock_guard lock(proc.alloc_mutex_);
    const auto it = proc.allocations_.find(args->handle);
    if (it == proc.allocations_.end())
      return -EINVAL;
    const auto &alloc = it->second;
    if (alloc.retirement_in_progress)
      return -EAGAIN;
    // USERPTR must retain the caller's live CPU/GPU alias. Snapshot promotion
    // would redirect the GPU PTEs to a private memfd.
    if (alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)
      return -EPERM;
    if (alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
      return -EINVAL;
    if (args->gpu_id != 0 && args->gpu_id != alloc.gpu_id)
      return -EINVAL;
    if (!alloc.backing_promotion_attempted && alloc.domain == KfdProcess::GpuGrantDomain::Vram &&
        !alloc.backing_owner && !alloc.mapped_device_ids.empty()) {
      if (alloc.ipc_shared)
        return -EOPNOTSUPP;
      if (pte_mtype_for_flags(alloc.flags) == amdgpu::Mtype::UC) {
        // A caller-owned view would need a snapshot/remap while UC requests
        // can still be in flight. Driver-owned memfd bytes need no such move.
        if (alloc.memfd < 0 && alloc.host_ptr)
          return -EOPNOTSUPP;
      } else {
        if (!alloc.local_backing_owner || alloc.mapped_device_ids.size() != 1 ||
            !alloc.mapped_device_ids.contains(alloc.gpu_id))
          return -EOPNOTSUPP;
        const auto generation = alloc.admission_generations.find(alloc.gpu_id);
        if (generation == alloc.admission_generations.end())
          return -EAGAIN;
        const auto aliases = proc.gpu_local_alias_snapshot();
        const auto found = std::ranges::find_if(*aliases, [&](const auto &alias) {
          return alias.gpu_id == alloc.gpu_id && alias.gpu_va == alloc.gpu_va &&
                 alias.extent == alloc.size &&
                 alias.backing_id == alloc.local_backing_owner->backing_id() &&
                 alias.admission_generation == generation->second;
        });
        if (found == aliases->end())
          return -EAGAIN;

        constexpr uint32_t kMemoryKind = KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                                         KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                         KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
        const bool eligible = alloc.memfd >= 0 && alloc.host_ptr && alloc.host_ptr_owned &&
                              alloc.host_ptr_mapped_bytes == alloc.size && !alloc.imported &&
                              !alloc.driver_gpu_mapping &&
                              (alloc.flags & kMemoryKind) == KFD_IOC_ALLOC_MEM_FLAGS_VRAM &&
                              !(alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) &&
                              alloc.size <= std::numeric_limits<size_t>::max();
        // An ineligible export keeps its local cacheable policy and owner; it
        // never publishes a peer backing that would need a UC transition.
        if (eligible)
          local_alias = *found;
        else if (alias_ingress_gate_->closed(GpuAliasIngressGate::key(*found)))
          return -EAGAIN;
      }
    }
  }
  if (local_alias) {
    const int result = promote_local_alias_for_ipc(proc, args->handle, *local_alias, op_lock);
    if (result != 0)
      return result;
  }

  uint64_t alloc_size = 0;
  uint32_t alloc_flags = 0;
  uint32_t alloc_gpu_id = 0;
  int dup_fd = -1;
  std::shared_ptr<const KfdProcess::GpuBacking> backing_owner;

  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    auto it = proc.allocations_.find(args->handle);
    if (it == proc.allocations_.end())
      return -EINVAL;
    auto &alloc = it->second;
    if (alloc.retirement_in_progress)
      return -EAGAIN;
    if (!alloc.backing_promotion_attempted && alloc.domain == KfdProcess::GpuGrantDomain::Vram &&
        !alloc.backing_owner && !alloc.mapped_device_ids.empty() &&
        pte_mtype_for_flags(alloc.flags) != amdgpu::Mtype::UC && !alloc.local_backing_owner)
      return -EOPNOTSUPP;
    if (args->gpu_id != 0 && args->gpu_id != alloc.gpu_id)
      return -EINVAL;
    const bool had_memfd = alloc.memfd >= 0;

    if (alloc.memfd < 0 && alloc.host_ptr) {
      int promoted_fd = memfd_create("rocjitsu_ipc_promote", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (promoted_fd < 0)
        return -errno;
      if (ftruncate(promoted_fd, static_cast<off_t>(alloc.size)) != 0) {
        libc_passthrough().close(promoted_fd);
        return -errno;
      }
      if (safe_fcntl(promoted_fd, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
        const int seal_errno = errno;
        libc_passthrough().close(promoted_fd);
        return -seal_errno;
      }
      auto *new_host_ptr =
          safe_mmap(nullptr, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED, promoted_fd, 0);
      if (new_host_ptr == MAP_FAILED) {
        libc_passthrough().close(promoted_fd);
        return -ENOMEM;
      }
      std::memcpy(new_host_ptr, alloc.host_ptr, alloc.size);

      if (proc.remap_page_host_ptrs(alloc.gpu_va, alloc.host_ptr, new_host_ptr, alloc.size) !=
          KfdProcess::PageTableMutationResult::Applied) {
        safe_munmap(new_host_ptr, alloc.size);
        libc_passthrough().close(promoted_fd);
        return -EINVAL;
      }

      if (alloc.host_ptr_owned && !alloc.host_ptr_caller_reserved)
        safe_munmap(alloc.host_ptr, alloc.host_ptr_mapped_bytes);

      alloc.host_ptr = new_host_ptr;
      alloc.host_ptr_owned = true;
      alloc.host_ptr_caller_reserved = false;
      alloc.host_ptr_mapped_bytes = alloc.size;
      alloc.memfd = promoted_fd;
      {
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.insert(promoted_fd);
      }
    } else if (alloc.memfd < 0 && !alloc.imported) {
      int new_fd = memfd_create("rocjitsu_ipc_lazy", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (new_fd < 0)
        return -errno;
      if (ftruncate(new_fd, static_cast<off_t>(alloc.size)) != 0) {
        libc_passthrough().close(new_fd);
        return -errno;
      }
      alloc.memfd = new_fd;
      {
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.insert(new_fd);
      }
    }

    std::optional<amdgpu::DeviceCacheMaintenanceLease> peer_transition;
    const auto previous_backing_owner = alloc.backing_owner;
    const bool previous_promotion_attempted = alloc.backing_promotion_attempted;
    const bool previous_local_policy = alloc.ipc_export_retains_local_policy;
    if (!alloc.backing_promotion_attempted) {
      constexpr uint32_t kMemoryKind = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                       KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
      // A legacy export may snapshot a caller pointer into a new memfd. Such a
      // BO is still ineligible: it never had stable shared backing at alloc.
      // The source's driver-owned CPU view must cover the whole BO. A short
      // mmap can leave valid memfd bytes that the source GPU cannot address.
      if (had_memfd && alloc.host_ptr && alloc.host_ptr_owned &&
          alloc.host_ptr_mapped_bytes == alloc.size && !alloc.imported &&
          !alloc.driver_gpu_mapping &&
          (alloc.flags & kMemoryKind) == KFD_IOC_ALLOC_MEM_FLAGS_VRAM &&
          !(alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) &&
          alloc.size <= std::numeric_limits<size_t>::max()) {
        const GpuDevice *owner = find_gpu(alloc.gpu_id);
        if (!owner || !owner->soc)
          return -EINVAL;
        // Publish dirty owner-cache bytes before the canonical alias can be
        // used by a peer. Keep caches excluded until the local PTE is UC and
        // the new peer lease has reached the immutable grant index.
        try {
          peer_transition.emplace(owner->soc->cache_coherence()->acquire_cache_maintenance(
              amdgpu::DeviceCacheOperation::WritebackInvalidate));
        } catch (const std::bad_alloc &) {
          return -ENOMEM;
        } catch (...) {
          return -EIO;
        }
        alloc.backing_owner = KfdProcess::GpuBacking::create_memfd_alias(
            alloc.memfd, static_cast<size_t>(alloc.size), alloc.gpu_id,
            alloc.cacheable_peer_for_testing);
      }
      alloc.backing_promotion_attempted = true;
    }

    // A controlled source must never export a handle whose import lacks the
    // same retained write policy. Local promotion moves the existing owner;
    // direct export creates it above.
    if (alloc.cacheable_peer_for_testing &&
        (!alloc.backing_owner || !alloc.backing_owner->controlled_peer_for_testing())) {
      alloc.backing_owner = previous_backing_owner;
      alloc.backing_promotion_attempted = previous_promotion_attempted;
      return -EOPNOTSUPP;
    }

    if (!previous_promotion_attempted && alloc.domain == KfdProcess::GpuGrantDomain::Vram &&
        !alloc.backing_owner && alloc.ever_gpu_admitted)
      alloc.ipc_export_retains_local_policy = true;

    // An executable peer alias uses uncached local access until P3 supplies
    // the coherence provider. Legacy IPC exports without an alias retain the
    // CC refetch policy.
    const amdgpu::Mtype export_mtype =
        alloc.cacheable_peer_for_testing
            ? pte_mtype_for_flags(alloc.flags)
            : (alloc.backing_owner
                   ? amdgpu::Mtype::UC
                   : (alloc.ipc_export_retains_local_policy ? pte_mtype_for_flags(alloc.flags)
                                                            : amdgpu::Mtype::CC));
    if (proc.set_page_mtype(alloc.gpu_va, alloc.size, export_mtype) !=
        KfdProcess::PageTableMutationResult::Applied) {
      alloc.backing_owner = previous_backing_owner;
      alloc.backing_promotion_attempted = previous_promotion_attempted;
      alloc.ipc_export_retains_local_policy = previous_local_policy;
      return -EINVAL;
    }
    alloc.ipc_shared = true;
    if (peer_transition)
      proc.publish_gpu_grants_locked();
    peer_transition.reset();

    alloc_size = alloc.size;
    alloc_flags = alloc.flags;
    alloc_gpu_id = alloc.gpu_id;
    backing_owner = alloc.backing_owner;
    dup_fd = retain_export_descriptor_locked(alloc);
  }

  if (dup_fd < 0)
    return dup_fd;
  UniqueDriverFd exported_fd(dup_fd);

  IpcHandleKey key{};
  const ssize_t random_bytes = getrandom(key.words, sizeof(key.words), 0);
  if (random_bytes < 0)
    return -errno;
  if (random_bytes != sizeof(key.words))
    return -EIO;

  // Keep the allocation's metadata snapshot and IPC publication atomic with
  // retain_bo_state(). Generate the key before taking either lock.
  std::lock_guard alloc_lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end())
    return -EINVAL;
  IpcObject obj{};
  std::memcpy(obj.share_handle, key.words, sizeof(key.words));
  obj.backing_memfd = exported_fd.get();
  obj.bo_state = it->second.bo_state;
  obj.backing_owner = std::move(backing_owner);
  obj.allocation_size = alloc_size;
  obj.allocation_flags = alloc_flags;
  obj.source_gpu_id = alloc_gpu_id;
  obj.source_process_id = proc.process_id();
  obj.source_alloc_handle = args->handle;
  {
    std::lock_guard ipc_lock(ipc_mutex_);
    if (!ipc_store_.emplace(key, obj).second)
      return -EAGAIN;
  }
  (void)exported_fd.release();

  std::memcpy(args->share_handle, key.words, sizeof(key.words));
  util::Logger::vm("ipc_export: handle=", args->handle, " size=", alloc_size,
                   " gpu_id=", alloc_gpu_id);
  return 0;
}

int SimulatedKfd::ipc_import_handle_ioctl(
    KfdProcess &proc, void *arg, std::shared_ptr<const KfdProcess::GpuBacking> &retiring_backing) {
  auto *args = static_cast<kfd_ioctl_ipc_import_handle_args *>(arg);
  const uint64_t requested_va = args->va_addr;

  IpcHandleKey key{};
  std::memcpy(key.words, args->share_handle, sizeof(key.words));

  UniqueDriverFd backing;
  std::shared_ptr<GemObject> bo_state;
  uint64_t alloc_size = 0;
  uint32_t alloc_flags = 0;
  uint32_t source_gpu_id = 0;
  int dup_fd = -1;

  // Prevent first-time PRIME tagging from racing the metadata snapshot and
  // allocation insertion. Match retain_bo_state()'s alloc -> IPC lock order.
  std::unique_lock alloc_lock(proc.alloc_mutex_);
  {
    std::lock_guard<std::mutex> lk(ipc_mutex_);
    auto it = ipc_store_.find(key);
    if (it == ipc_store_.end())
      return -EINVAL;
    alloc_size = it->second.allocation_size;
    alloc_flags = it->second.allocation_flags;
    source_gpu_id = it->second.source_gpu_id;
    bo_state = it->second.bo_state;
    retiring_backing = it->second.backing_owner;
    if (alloc_size == 0 || alloc_size > UINT64_MAX - 0xFFF ||
        alloc_size > std::numeric_limits<size_t>::max())
      return -EINVAL;
    if (retiring_backing &&
        (retiring_backing->gpu_id() != source_gpu_id || retiring_backing->size() != alloc_size))
      return -EINVAL;
    if (args->gpu_id != 0 && args->gpu_id != source_gpu_id) {
      util::Logger::vm("ipc_import: gpu_id mismatch: requested=", args->gpu_id,
                       " source=", source_gpu_id);
      return -EINVAL;
    }
    backing = UniqueDriverFd(safe_fcntl(it->second.backing_memfd, F_DUPFD_CLOEXEC, 0));
  }

  if (!backing)
    return -errno;
  const int original_flags = safe_fcntl(backing.get(), F_GETFL, 0);
  if (original_flags < 0)
    return -errno;
  const int public_mode = original_flags & O_ACCMODE;
  if (public_mode != O_RDWR) {
    const auto path = std::format("/proc/self/fd/{}", backing.get());
    const int writable = libc_passthrough().openat(AT_FDCWD, path.c_str(), O_RDWR | O_CLOEXEC, 0);
    if (writable < 0)
      return -errno;
    backing = UniqueDriverFd(writable);
  }
  dup_fd = backing.release();

  const bool controlled_backing =
      retiring_backing && retiring_backing->controlled_peer_for_testing();
  const uint64_t aligned_size = (alloc_size + 0xFFF) & ~uint64_t(0xFFF);
  {
    std::lock_guard<std::mutex> flk(owned_fds_mutex_);
    owned_fds_.insert(dup_fd);
  }
  auto release_backing = [&](void *mapped) {
    if (mapped != MAP_FAILED)
      safe_munmap(mapped, alloc_size);
    {
      std::lock_guard<std::mutex> flk(owned_fds_mutex_);
      owned_fds_.erase(dup_fd);
    }
    libc_passthrough().close(dup_fd);
  };

  if (requested_va != 0 && requested_va > UINT64_MAX - alloc_size) {
    release_backing(MAP_FAILED);
    return -EINVAL;
  }

  auto *host_ptr = safe_mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dup_fd, 0);
  if (host_ptr == MAP_FAILED) {
    release_backing(MAP_FAILED);
    return -ENOMEM;
  }

  uint64_t gpu_va;
  uint64_t handle;
  bool invalid_va = false;
  int import_error = -EINVAL;
  bool protected_import = false;
  bool allocation_inserted = false;
  bool mapping_started = false;
  std::optional<amdgpu::VmRawBackingWriteGate::Closed> raw_write_close;
  const auto retire_import_guards = [&]() noexcept {
    if (!protected_import)
      return;
    for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal)
      if (gpus_[ordinal].soc)
        gpus_[ordinal].soc->gpu_vm().retire_controlled_cache_backing(
            proc.gpu(ordinal).address_space, gpu_va, alloc_size);
  };
  try {
    if (requested_va != 0)
      gpu_va = requested_va;
    else {
      gpu_va = proc.next_gpu_va_;
      invalid_va = gpu_va > UINT64_MAX - aligned_size;
      if (!invalid_va)
        proc.next_gpu_va_ += aligned_size;
    }
    if (!invalid_va) {
      auto va_transaction = proc.begin_page_table_transaction();
      const uint64_t last_byte = gpu_va + alloc_size - 1;
      invalid_va = std::ranges::any_of(proc.allocations_, [&](const auto &entry) {
        const auto &existing = entry.second;
        return existing.size != 0 && existing.gpu_va <= last_byte &&
               (existing.size > UINT64_MAX - existing.gpu_va ||
                gpu_va <= existing.gpu_va + existing.size - 1);
      });
      if (!invalid_va) {
        std::shared_lock page_lock(proc.page_table_mutex_);
        const uint64_t first_page = gpu_va >> KfdProcess::kPageShift;
        const uint64_t last_page = last_byte >> KfdProcess::kPageShift;
        invalid_va = std::ranges::any_of(proc.page_table_, [&](const auto &entry) {
          return first_page <= entry.first && entry.first <= last_page;
        });
      }
      if (!invalid_va)
        invalid_va = proc.has_foreign_resident_overlap(va_transaction, gpu_va, alloc_size,
                                                       {KfdProcess::VaOwnerKind::Kfd, 0}) !=
                     KfdProcess::PageTableMutationResult::Applied;
      if (invalid_va) {
        if (requested_va == 0)
          proc.next_gpu_va_ = gpu_va;
      } else {
        // dispatch_ioctl holds this process's operation lock through import,
        // so close cannot remove its VM bindings while these guards publish.
        // An imported owner VA shares the source's controlled backing, even
        // though its PTE stays UC. Exclude admitted raw effects and publish
        // every sink guard before either a grant or PTE becomes visible.
        if (controlled_backing) {
          for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal)
            if (gpus_[ordinal].soc &&
                gpus_[ordinal].soc->gpu_vm().controlled_cache_backing_overlaps(proc.process_id(),
                                                                               gpu_va, alloc_size))
              invalid_va = true;
          if (!invalid_va) {
            raw_write_close = proc.raw_write_gate_->try_close(gpu_va, alloc_size);
            if (!raw_write_close) {
              invalid_va = true;
              import_error = -EAGAIN;
            }
          }
          if (!invalid_va) {
            protected_import = true;
            for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
              if (!gpus_[ordinal].soc)
                continue;
              const amdgpu::AddressSpaceHandle binding = proc.gpu(ordinal).address_space;
              if (!binding || !gpus_[ordinal].soc->gpu_vm().protect_controlled_cache_backing(
                                  binding, gpu_va, alloc_size)) {
                invalid_va = true;
                break;
              }
            }
          }
        }
      }
      if (!invalid_va) {
        handle = proc.next_handle_++;

        KfdProcess::GpuAllocation alloc{};
        alloc.gpu_va = gpu_va;
        alloc.size = alloc_size;
        alloc.flags = alloc_flags;
        alloc.domain = grant_domain_for_flags(alloc_flags);
        alloc.handle = handle;
        alloc.host_ptr = host_ptr;
        alloc.host_ptr_owned = true;
        alloc.host_ptr_mapped_bytes = alloc_size;
        alloc.memfd = dup_fd;
        alloc.bo_state = bo_state;
        alloc.export_access_mode = public_mode;
        alloc.gpu_id = source_gpu_id;
        alloc.imported = true;
        alloc.ipc_shared = true;
        alloc.backing_promotion_attempted = true;
        alloc.backing_owner = retiring_backing;
        proc.allocations_.emplace(handle, alloc);
        allocation_inserted = true;
        proc.claim_gpu_range_locked(alloc.gpu_va, alloc.size);
        proc.publish_gpu_grants_locked(va_transaction);
        // A routed peer alias bypasses caches until the fabric coherence
        // provider exists. Legacy IPC imports keep their CC refetch policy.
        mapping_started = true;
        if (map_to_gpu(proc, gpu_va, host_ptr, alloc_size,
                       alloc.backing_owner ? amdgpu::Mtype::UC : amdgpu::Mtype::CC,
                       KfdProcess::HostExtentOwner::Application,
                       &va_transaction) != KfdProcess::PageTableMutationResult::Applied) {
          if (mapping_started && unmap_from_gpu(proc, gpu_va, alloc_size, &va_transaction) ==
                                     KfdProcess::PageTableMutationResult::Applied) {
            proc.allocations_.erase(handle);
            allocation_inserted = false;
            proc.retire_gpu_range_locked(gpu_va, alloc_size);
            proc.publish_gpu_grants_locked(va_transaction);
            if (requested_va == 0)
              proc.next_gpu_va_ = gpu_va;
          } else {
            import_error = -EIO;
          }
          invalid_va = true;
        } else {
          proc.allocations_.at(handle).page_table_mapped_bytes = alloc_size;
        }
      }
    }
  } catch (const std::bad_alloc &) {
    import_error = -ENOMEM;
    invalid_va = true;
  } catch (...) {
    import_error = -EIO;
    invalid_va = true;
  }
  alloc_lock.unlock();
  if (invalid_va) {
    if (allocation_inserted) {
      // A page-table insertion can throw after publishing a prefix. Keep the
      // controlled guard and backing alive unless the PTE and grant rollback
      // both complete while the importer's operation lock still excludes MAP.
      try {
        std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
        auto va_transaction = proc.begin_page_table_transaction();
        if (!mapping_started || unmap_from_gpu(proc, gpu_va, alloc_size, &va_transaction) ==
                                    KfdProcess::PageTableMutationResult::Applied) {
          proc.allocations_.erase(handle);
          allocation_inserted = false;
          proc.retire_gpu_range_locked(gpu_va, alloc_size);
          proc.publish_gpu_grants_locked(va_transaction);
        }
      } catch (...) {
        // The retained allocation, immutable backing policy, and guard remain
        // owned by process close even after this scoped gate reopens.
      }
    }
    if (!allocation_inserted) {
      retire_import_guards();
      release_backing(host_ptr);
      if (requested_va == 0) {
        std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
        proc.next_gpu_va_ = gpu_va;
      }
    }
    return import_error;
  }

  args->handle = handle;
  args->mmap_offset = handle << 12;
  args->flags = alloc_flags;

  util::Logger::vm("ipc_import: handle=", handle, " gpu_va=0x", std::hex, gpu_va,
                   " size=", std::dec, alloc_size, " gpu_id=", source_gpu_id);
  return 0;
}

int SimulatedKfd::get_dmabuf_info_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_get_dmabuf_info_args *>(arg);
  uint64_t size = 0;
  uint32_t gpu_id = gpus_.empty() ? 0 : gpus_[0].gpu_id;

  bool found = false;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    for (const auto &[handle, info] : proc.imported_dmabufs_) {
      [[maybe_unused]] auto &_ = handle;
      if (info.fd >= 0 && static_cast<uint32_t>(info.fd) == args->dmabuf_fd) {
        size = info.size;
        gpu_id = info.gpu_id;
        found = true;
        break;
      }
    }
  }

  if (!found) {
    struct stat st {};
    if (safe_fstat(args->dmabuf_fd, &st) != 0)
      return -errno;
    size = static_cast<uint64_t>(st.st_size);
  }

  args->size = size;
  args->gpu_id = gpu_id;
  args->flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  // metadata_ptr is a client-process address that cannot be dereferenced in
  // daemon mode. ROCR currently queries with metadata_size == 0; reject
  // metadata-bearing calls rather than risk a cross-process pointer deref.
  if (args->metadata_size > 0 && daemon_mode_)
    return -EINVAL;
  if (args->metadata_ptr && args->metadata_size && !daemon_mode_) {
    std::memset(reinterpret_cast<void *>(args->metadata_ptr), 0,
                static_cast<size_t>(args->metadata_size));
  }
  args->metadata_size = 0;
  return 0;
}

int SimulatedKfd::svm_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_svm_args *>(arg);
  auto *attrs = reinterpret_cast<kfd_ioctl_svm_attribute *>(args + 1);

  if (args->op == KFD_IOCTL_SVM_OP_SET_ATTR) {
    KfdProcess::SvmRange range{};
    range.size = args->size;
    for (uint32_t i = 0; i < args->nattr; ++i)
      range.attributes[attrs[i].type] = attrs[i].value;
    proc.svm_ranges_[args->start_addr] = std::move(range);
    return 0;
  }

  if (args->op == KFD_IOCTL_SVM_OP_GET_ATTR) {
    auto it = proc.svm_ranges_.find(args->start_addr);
    for (uint32_t i = 0; i < args->nattr; ++i) {
      uint32_t type = attrs[i].type;
      uint32_t value = 0;
      if (it != proc.svm_ranges_.end()) {
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

int SimulatedKfd::runtime_enable_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_runtime_enable_args *>(arg);

  const bool enabling = (args->mode_mask & KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK) != 0;
  // Whether a debugger was already attached when the runtime came up, decided
  // under the same lock that DBG_TRAP_ENABLE uses to publish a session.
  bool notify_debugger = false;
  // Same decision for the disable direction, which reports the same way.
  bool notify_disable = false;
  {
    // Scoped so neither lock is held across the handshake wait below.
    std::unique_lock<std::mutex> op_lock(proc.op_mutex_, std::defer_lock);
    if (!op_lock.try_lock()) {
      if (proc.canonical_retirements_pending_.load(std::memory_order_acquire) != 0 ||
          proc.event_state_.is_closing())
        return -EAGAIN;
      op_lock.lock();
    }
    if (proc.event_state_.is_closing())
      return -ESRCH;
    // debug_sessions_mutex_ orders this against DBG_TRAP_ENABLE, which holds it
    // while it reads runtime_state_ to fill in the runtime_info it returns.
    // Exactly one of the two has to take responsibility for telling the
    // debugger the runtime is up: either ENABLE observes enabled==true and
    // reports it (rocdbgapi then calls runtime_enable() straight from its
    // attach path, process.cpp), or this side observes the session and raises
    // EC_PROCESS_RUNTIME. Without a common lock both can read the other's
    // pre-state -- ENABLE reports DISABLED while this side sees no session --
    // and NEITHER fires -- the debugger then waits for an event that is never
    // coming and never learns the runtime came up.
    //
    // Closing this did NOT measurably change gdb.rocm/multi-inferior-stress.exp,
    // which fails for a different reason (see below); it is fixed here because
    // the window is real on its own terms, not because it cured that test.
    // Taking it here also cannot double-report: whichever side takes the lock
    // second observes the first, and rocdbgapi treats a runtime_state that does
    // not toggle as a fatal "spurious runtime exception".
    std::lock_guard<std::mutex> session_lock(debug_sessions_mutex_);
    std::lock_guard<std::mutex> lock(proc.runtime_mutex_);
    if (enabling) {
      if (proc.runtime_state_.pending)
        return -EBUSY;
      bool has_queues = [&] {
        std::lock_guard<std::mutex> alock(proc.alloc_mutex_);
        return !proc.active_queue_ids_.empty();
      }();
      if (!proc.runtime_state_.enabled && has_queues)
        return -EEXIST;
      proc.runtime_state_.enabled = true;
      proc.runtime_state_.pending = false;
      proc.runtime_state_.mode_mask = args->mode_mask;
      proc.runtime_state_.capabilities_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
      proc.runtime_state_.r_debug = args->r_debug;
      args->capabilities_mask = proc.runtime_state_.capabilities_mask;
      auto session = debug_sessions_.find(proc.client_pid());
      notify_debugger = session != debug_sessions_.end() && session->second.enabled;
    } else {
      // Report the !disabled -> disabled transition. rocdbgapi needs it to stop
      // driving the queues of a process whose runtime has gone away: those ops
      // are gated on the runtime being enabled and answer -EPERM, which it
      // turns into a fatal os_driver::resume_queues failure and GDB into an
      // internal error (gdb.rocm/multi-inferior-stress.exp). Only report a real
      // transition -- a second disable is not one, and rocdbgapi rejects an
      // event whose runtime_state has not moved as spurious. Decided against
      // the session table under the same lock the enable branch uses, so an
      // undebugged process exit does not re-take debug_sessions_mutex_ in the
      // callee just to discover there is nobody to tell.
      auto session = debug_sessions_.find(proc.client_pid());
      notify_disable = proc.runtime_state_.enabled && session != debug_sessions_.end() &&
                       session->second.enabled;
      proc.runtime_state_ = KfdProcess::RuntimeState{};
      args->capabilities_mask = 0;
    }
  }

  // Ordered after the state reset above so that the debugger's follow-up
  // QUERY_EXCEPTION_INFO reads the new state, and outside the lock scope
  // because the handshake blocks.
  if (notify_debugger || notify_disable)
    runtime_debugger_handshake(proc.client_pid(), /*enabling=*/notify_debugger);
  return 0;
}

std::shared_ptr<KfdProcess> SimulatedKfd::find_process_by_client_pid(pid_t pid) const {
  if (pid == 0)
    return nullptr;
  std::lock_guard<std::mutex> lk(process_mutex_);
  for (auto &[id, proc] : processes_)
    if (proc->client_pid() == pid)
      return proc;
  return nullptr;
}

namespace {

uint32_t gfx1250_status(uint32_t status) {
  // SCC, priorities and HALT moved to STATE_PRIV. The remaining fields used by
  // dbgapi retain their gfx9 positions.
  return status & ~((0x1Fu << 0) | (1u << 7) | (1u << 13) | (1u << 19));
}

uint32_t internal_status_from_gfx1250(const kmd::CwsrWaveState &state) {
  uint32_t status = amdgpu::update_status_from_gfx12_state_priv(state.status, state.state_priv);
  status &= ~amdgpu::Wavefront::kStatusHaltMask;
  if (state.saved_status_halt)
    status |= amdgpu::Wavefront::kStatusHaltMask;
  return status;
}

uint32_t internal_trapsts_from_gfx1250(const kmd::CwsrWaveState &state) {
  uint32_t trapsts = amdgpu::update_trapsts_from_gfx12_excp_flag_priv(0, state.excp_flag_priv);
  return amdgpu::update_trapsts_from_gfx12_excp_flag_user(trapsts, state.excp_flag_user);
}

kmd::CwsrWaveState build_cwsr_wave_state(amdgpu::Wavefront &wf, rj_code_arch_t arch) {
  kmd::CwsrWaveState state;
  const bool gfx12_5 = kmd::cwsr_layout_kind(arch) == kmd::CwsrLayoutKind::Gfx12_5;
  const uint32_t raw_status = wf.status_raw();
  state.pc = wf.pc;
  // A wave frozen inside the trap handler has the HANDLER's EXEC live, not the
  // application's: the handler runs with its own mask and parks 0x80000000 (and
  // then a doorbell id) in EXEC_LO around MSG_GET_DOORBELL. Publishing that
  // makes the debugger report every lane active, which inverts `lane apply
  // -active/-inactive` (gdb.rocm/lane-info.exp). trap_saved_exec_ holds the
  // interrupted value for the whole handler, so un-shadow it here, as the
  // STATUS field below does with trap_saved_status().
  //
  // in_trap_handler() is the predicate for both, and it opens at handler entry
  // rather than at MSG_INTERRUPT because the doorbell park happens on the way
  // there. It closes at s_rfe, which puts the interrupted values back, so from
  // then on the live registers are the application's again.
  //
  // The record is not write-only -- apply_cwsr_to_wave() feeds it back on
  // resume -- so un-shadowing here is only safe because that path shadows in
  // the other direction under the same predicate: a debugger edit to EXEC is
  // written to trap_saved_exec_, which the handler's s_rfe installs, instead of
  // overwriting the mask a mid-flight handler is still running under.
  state.exec = wf.in_trap_handler() ? wf.trap_saved_exec() : wf.exec();
  state.vcc = wf.vcc();
  state.flat_scratch = wf.scratch_base();
  state.m0 = wf.m0();
  state.mode = wf.mode_raw();
  state.trapsts = wf.trapsts();
  // Whether the application was already halted when it trapped outlives the
  // handler: trap_saved_status_ still describes the interrupted wave after
  // s_rfe, and trap_interrupt_sent() is the flag that says a handler produced
  // this stop. in_trap_handler() has to be part of the predicate too, or the
  // two halves of the published STATUS come from different registers between
  // trap entry and MSG_INTERRUPT -- the body from trap_saved_status_ and the
  // HALT bit from the handler's live one. apply_cwsr_to_wave() recombines them
  // into a word that never existed on the wave, so an identity round-trip in
  // that window rewrites the application's saved HALT with the handler's.
  const bool handler_context = wf.in_trap_handler() || wf.trap_interrupt_sent();
  state.saved_status_halt = ((handler_context ? wf.trap_saved_status() : raw_status) >> 13) & 1u;
  state.wave_stopped = wf.debug_halted();
  // The published STATUS shadows on the narrower predicate, the same one EXEC
  // uses: only while the handler is actually running is the live register the
  // handler's rather than the application's. Once s_rfe has run, the live value
  // is the application's again and is what the debugger should see.
  const uint32_t application_status = wf.in_trap_handler() ? wf.trap_saved_status() : raw_status;
  state.status =
      state.wave_stopped ? application_status | (1u << 13) : application_status & ~(1u << 13);
  if (gfx12_5) {
    uint32_t debug_status = application_status;
    if (state.wave_stopped)
      debug_status |= amdgpu::Wavefront::kStatusHaltMask;
    else
      debug_status &= ~amdgpu::Wavefront::kStatusHaltMask;
    state.state_priv = amdgpu::gfx12_state_priv_from_status(debug_status, state.flat_scratch != 0);
    state.status = gfx1250_status(application_status);
    state.excp_flag_priv = amdgpu::gfx12_excp_flag_priv_from_trapsts(state.trapsts);
    state.excp_flag_user = (state.trapsts & 0x7Fu) | wf.gfx12_excp_flag_user_extra_raw();
    state.trap_ctrl = wf.gfx12_trap_ctrl_raw();
    state.xnack_state_priv = wf.gfx1250_xnack_state_priv_raw();
    state.xnack_mask = wf.gfx1250_xnack_mask_raw();
  }
  state.trap_id = wf.trap_id();
  state.wave_id = wf.debug_wave_id();
  state.group_ids = wf.wg_coord();
  state.wave_in_group = wf.wave_in_group();
  state.queue_packet_id = wf.aql_packet_id() & 0x1FFFFFFu;
  state.scratch_scoreboard_id = wf.scratch_scoreboard_id();
  state.shader_engine_id = wf.shader_engine_id();
  state.spi_ttmps_setup = true;
  state.num_sgprs = wf.num_sgprs();
  state.num_vgprs = wf.num_vgprs();

  state.sgprs.resize(state.num_sgprs);
  for (uint32_t s = 0; s < state.num_sgprs; ++s)
    state.sgprs[s] = wf.debug_read_sgpr(s);
  // The CWSR record's VGPR stride is wave64-shaped (cwsr.cpp kVgprLaneBytes)
  // whatever the wave size is, but the physical register only has wf_size()
  // lanes -- a wave32 VectorReg<32> is exactly 32 wide, so reading lanes 32-63
  // runs off it into the neighbouring register. Read the live lanes and leave
  // the rest of the wave64-shaped slot at the zero resize() already wrote; the
  // restore path is bounded by wf_size() too and never looks at them.
  state.vgprs.resize(static_cast<size_t>(state.num_vgprs) * 64);
  const uint32_t live_lanes = wf.wf_size();
  for (uint32_t r = 0; r < state.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < live_lanes; ++lane)
      state.vgprs[static_cast<size_t>(r) * 64 + lane] = wf.debug_read_vgpr(r, lane);
  state.lds.resize(wf.lds_size());
  if (!state.lds.empty())
    static_cast<const amdgpu::Lds &>(wf.lds()).read(wf.lds_base(), state.lds.data(),
                                                    static_cast<uint32_t>(state.lds.size()));
  return state;
}

void prepare_cwsr_wave_states(std::vector<kmd::CwsrWaveState> &waves) {
  std::ranges::sort(waves, [](const auto &lhs, const auto &rhs) {
    if (lhs.queue_packet_id != rhs.queue_packet_id)
      return lhs.queue_packet_id < rhs.queue_packet_id;
    if (lhs.group_ids != rhs.group_ids)
      return lhs.group_ids < rhs.group_ids;
    return lhs.wave_in_group < rhs.wave_in_group;
  });
  auto same_group = [](const auto &lhs, const auto &rhs) {
    return lhs.queue_packet_id == rhs.queue_packet_id && lhs.group_ids == rhs.group_ids;
  };
  for (size_t index = 0; index < waves.size(); ++index) {
    waves[index].is_first_in_group = index == 0 || !same_group(waves[index], waves[index - 1]);
    waves[index].is_last_in_group =
        index + 1 == waves.size() || !same_group(waves[index], waves[index + 1]);
    if (!waves[index].is_first_in_group)
      waves[index].lds.clear();
  }
}

} // namespace

bool SimulatedKfd::serialize_queue_debug_waves(uint32_t process_id, uint32_t queue_id,
                                               uint32_t gpu_id, uint64_t ctx_base,
                                               uint32_t ctx_size) {
  auto *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc || ctx_base == 0)
    return false;

  // Never publish a record shaped for the wrong architecture. Handlers ask
  // debug_stop_publishable() before claiming a stop, so reaching here on an
  // unmodelled part means a caller skipped the gate; refuse rather than hand
  // rocm-dbgapi an image it would decode against its own layout.
  if (!kmd::cwsr_layout_modelled(gpu->soc->arch()))
    return false;

  const bool per_xcc_areas = gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5;
  const uint32_t area_count = per_xcc_areas ? std::max(1u, gpu->soc->num_xcds()) : 1u;
  std::vector<std::vector<kmd::CwsrWaveState>> waves_by_area(area_count);
  auto collect = [&](amdgpu::CommandProcessor *cp, uint32_t area) {
    for (auto *cu : cp->compute_units()) {
      cu->with_wave_state_locked([&] {
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          auto *wave = cu->wf(i);
          if (wave && wave->debug_stopped() && wave->process_id() == process_id &&
              wave->queue_id() == queue_id)
            waves_by_area[area].push_back(build_cwsr_wave_state(*wave, gpu->soc->arch()));
        }
      });
    }
  };
  if (per_xcc_areas) {
    for (uint32_t xcc_id = 0; xcc_id < area_count; ++xcc_id)
      collect(gpu->soc->xcd(xcc_id)->command_processor(), xcc_id);
  } else {
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) { collect(cp, 0); });
  }
  size_t wave_count = 0;
  for (const auto &waves : waves_by_area)
    wave_count += waves.size();
  if (wave_count == 0)
    return false;

  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const std::optional<amdgpu::GpuVmAccess> vm_access = gpu->soc->gpu_vm().snapshot_vmid(process_id);
  if (!vm_access)
    return false;
  UniqueDriverFd target_mem = duplicate_debug_target_mem(proc->client_pid());
  bool publish_ok = true;
  auto write_block = [&](uint64_t address, std::span<const uint8_t> bytes) {
    if (target_mem.get() < 0) {
      // A refused write publishes nothing, so reporting success would tell the
      // debugger a context is available at an address that holds none of it.
      // The pwrite path below fails publication on a short write for the same
      // reason; this is that failure arriving through the memory model.
      if (vm_access->write(address, std::as_bytes(bytes)) != amdgpu::VmAccessOutcome::Complete) {
        publish_ok = false;
        util::Logger::warn("CWSR target write faulted: addr=0x", std::hex, address, std::dec,
                           " pid=", proc->client_pid());
      }
      return;
    }
    const ssize_t written =
        pwrite(target_mem.get(), bytes.data(), bytes.size(), static_cast<off_t>(address));
    if (written != static_cast<ssize_t>(bytes.size())) {
      publish_ok = false;
      util::Logger::warn("CWSR target write failed: addr=0x", std::hex, address, " pid=", std::dec,
                         proc->client_pid(), " rc=", written, " errno=", errno);
    }
  };
  bool layouts_ok = true;
  constexpr size_t kCwsrHeaderBytes = 10 * sizeof(uint32_t);
  const std::array<uint8_t, kCwsrHeaderBytes> empty_header{};
  for (uint32_t area = 0; area < area_count; ++area) {
    const uint64_t area_base = ctx_base + static_cast<uint64_t>(area) * ctx_size;
    auto &waves = waves_by_area[area];
    if (waves.empty()) {
      write_block(area_base, empty_header);
      continue;
    }
    prepare_cwsr_wave_states(waves);
    layouts_ok &=
        kmd::serialize_queue_cwsr_bulk(area_base, ctx_size, waves, write_block, gpu->soc->arch())
            .ok;
  }
  if (!layouts_ok || !publish_ok)
    return false;

  uint64_t oldest_packet = UINT64_MAX;
  for (const auto &waves : waves_by_area)
    for (const auto &wave : waves)
      oldest_packet = std::min<uint64_t>(oldest_packet, wave.queue_packet_id);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    read_pointer = queue->second.read_pointer_address;
    write_pointer = queue->second.write_pointer_address;
  }
  if (read_pointer != 0 && write_pointer != 0) {
    const amdgpu::AtomicLoadResult write_index =
        vm_access->atomic_load(write_pointer, sizeof(uint64_t));
    if (write_index.outcome != amdgpu::VmAccessOutcome::Complete)
      return false;
    if (oldest_packet < write_index.value &&
        vm_access->atomic_store(read_pointer, sizeof(uint64_t), oldest_packet) !=
            amdgpu::VmAccessOutcome::Complete) {
      return false;
    }
  }
  return true;
}

void SimulatedKfd::on_wave_trap_complete(amdgpu::Wavefront &wave) {
  const uint32_t process_id = wave.process_id();
  const uint32_t queue_id = wave.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return;
  auto fall_back_to_runtime = [&](bool runtime_already_reserved = false) {
    const uint64_t unreported =
        wave.trap_queue_exception_status() & ~wave.trap_runtime_exception_status();
    if (unreported != 0) {
      // Keep the retained debugger stop until ROCr actually accepts the
      // deferred exception. Its completion clears HALT only after successful
      // publication; failure leaves a future debugger able to claim the stop.
      defer_wave_exception_to_runtime(wave, unreported, /*suspend_while_pending=*/false,
                                      /*clear_debug_stop_on_success=*/true,
                                      runtime_already_reserved);
      return;
    }
    wave.set_debug_halted(false);
    wave.set_status_halt(false);
  };

  const uint64_t queue_exception_status = wave.trap_queue_exception_status();
  uint64_t debugger_status = 0;
  bool runtime_reserved = false;
  if ((queue_exception_status & wave.trap_runtime_exception_status()) == 0) {
    debugger_status = debugger_queue_exception_mask(proc, queue_id, queue_exception_status,
                                                    /*reserve_runtime_if_unclaimed=*/true);
    runtime_reserved = queue_exception_status != 0 && debugger_status == 0;
  }
  if (queue_exception_status != 0 && debugger_status == 0) {
    // The handler observed an attached debugger and requested HALT, but this
    // exception belongs to ROCr. Do not strand the wave as a debugger stop;
    // the deferred runtime route will apply the queue-wide fatal suspension.
    fall_back_to_runtime(runtime_reserved);
    return;
  }

  uint64_t ctx_base = 0;
  uint32_t gpu_id = 0;
  bool queue_found = false;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue != proc->queue_snapshot_map_.end()) {
      queue_found = true;
      ctx_base = queue->second.ctx_save_restore_address;
      gpu_id = queue->second.gpu_id;
    }
  }
  if (!queue_found) {
    fall_back_to_runtime();
    return;
  }
  if (ctx_base == 0) {
    fall_back_to_runtime();
    return;
  }
  // The wave is already halted here by the handler's own STATUS.HALT, so there
  // is no stop to decline -- but waking a debugger that can never be given a
  // record only strands it. resolve_trap_handler() also withholds the debug
  // flag on such a part, so a cooperating handler never gets this far; one that
  // raises STATUS.HALT regardless still would.
  if (!debug_stop_publishable(gpu_id)) {
    fall_back_to_runtime();
    return;
  }

  apply_debug_event_publication_hook_for_testing(proc);

  // A trap interrupt is wave-local: hardware reports it without waiting for
  // every peer in the queue to stop. The debugger's ensuing SUSPEND_QUEUES
  // request publishes the authoritative full-queue CWSR snapshot.
  const bool notified = notify_debug_event(
      proc, queue_id,
      queue_exception_status != 0 ? debugger_status : KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
      /*retain_on_rejection=*/queue_exception_status != 0,
      /*reserve_runtime_on_rejection=*/queue_exception_status != 0);
  if (!notified)
    fall_back_to_runtime(/*runtime_already_reserved=*/queue_exception_status != 0);
}

uint64_t SimulatedKfd::debugger_queue_exception_mask(const std::shared_ptr<KfdProcess> &proc,
                                                     uint32_t queue_id, uint64_t exception_mask,
                                                     bool reserve_runtime_if_unclaimed) {
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return 0;
    gpu_id = queue->second.gpu_id;
  }
  if (!debug_stop_publishable(gpu_id))
    return 0;

  bool debugger_owns = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    const uint64_t enabled_mask = session != debug_sessions_.end() && session->second.enabled
                                      ? session->second.exception_enable_mask
                                      : 0;
    debugger_owns = (exception_mask & enabled_mask) != 0;

    std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return 0;
    queue->second.exception_status |= exception_mask;
    if (reserve_runtime_if_unclaimed && !debugger_owns) {
      // KFD records the event for an attached debugger before selecting its
      // delivery owner. A ROCr acknowledgment releases only the runtime claim.
      if (session != debug_sessions_.end() && session->second.enabled)
        queue->second.debug_notification_retained_status |= exception_mask;
      queue->second.begin_runtime_exception(exception_mask);
    }
  }
  // KFD makes one ownership decision for the complete event: any subscribed
  // bit assigns the full decoded mask to the debugger. Splitting a combined
  // event would let the runtime suspension prevent the handler from reaching
  // trap completion and publishing the debugger-owned portion.
  return debugger_owns ? exception_mask : 0;
}

void SimulatedKfd::reserve_runtime_queue_exception(const std::shared_ptr<KfdProcess> &proc,
                                                   uint32_t queue_id, uint64_t exception_mask) {
  std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
  auto queue = proc->queue_snapshot_map_.find(queue_id);
  if (queue == proc->queue_snapshot_map_.end())
    return;
  queue->second.begin_runtime_exception(exception_mask);
}

int SimulatedKfd::duplicate_debug_notifier(int fd) {
  if (debug_notifier_dup_error_for_testing_) {
    errno = *debug_notifier_dup_error_for_testing_;
    return -1;
  }
  return safe_fcntl(fd, F_DUPFD_CLOEXEC, 0);
}

int SimulatedKfd::replace_debug_session_for_testing(pid_t target_pid, int dbg_fd,
                                                    uint64_t exception_mask) {
  {
    std::lock_guard<std::mutex> lock(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end())
      return -EINVAL;
    session->second.generation = next_debug_session_generation_++;
    session->second.dbg_fd = dbg_fd;
    session->second.exception_enable_mask = exception_mask;
    session->second.next_notification_claim_id = 1;
    session->second.notification_claims.clear();
    session->second.notified_process_exception_mask = 0;
    session->second.pending_process_exception_mask = 0;
    session->second.pending_process_exception_counts.fill(0);
    session->second.notification_retry_needed = true;
  }
  return retry_debug_notifications(target_pid);
}

int SimulatedKfd::retry_debug_notifications(pid_t target_pid, bool invoke_result_hook) {
  struct PendingEventClaim {
    uint64_t mask = 0;
    uint64_t claim_id = 0;
    bool continuously_subscribed = true;
    uint64_t consumed_mask = 0;
  };
  struct PendingQueueNotification {
    uint32_t queue_id = 0;
    uint64_t mask = 0;
    std::vector<PendingEventClaim> claims;
  };

  std::shared_ptr<KfdProcess> proc;
  std::vector<PendingQueueNotification> queues;
  UniqueDriverFd notifier;
  uint64_t process_mask = 0;
  std::vector<PendingEventClaim> process_claims;
  uint64_t session_generation = 0;
  std::optional<int> clone3_error_for_testing;
  std::optional<int> clone_pidfd_error_for_testing;
  bool defer_reap_for_testing = false;
  std::function<void(bool)> result_hook;
  std::function<void()> write_hook;
  {
    std::lock_guard<std::mutex> lock(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled || session->second.dbg_fd < 0)
      return 0;
    // This invocation owns the outstanding retry. A failure or superseding
    // generation below re-arms it; a successful or empty scan leaves it clear.
    session->second.notification_retry_needed = false;

    proc = find_process_by_client_pid(target_pid);
    bool has_visible_queue_event = false;
    auto collect_queues = [&] {
      queues.clear();
      has_visible_queue_event = false;
      if (!proc)
        return;
      std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
      for (uint32_t queue_id : proc->active_queue_ids_) {
        auto queue = proc->queue_snapshot_map_.find(queue_id);
        if (queue == proc->queue_snapshot_map_.end())
          continue;
        const uint64_t already_delivered =
            queue->second.debug_notification_session_generation == session->second.generation
                ? queue->second.debug_notification_delivered_status
                : 0;
        const uint64_t visible_status =
            queue->second.debugger_visible_exception_status(session->second.generation);
        has_visible_queue_event |= (visible_status & session->second.exception_enable_mask) != 0;
        PendingQueueNotification pending{};
        pending.queue_id = queue_id;
        for (uint64_t event_mask : queue->second.debug_notification_events) {
          event_mask &= queue->second.exception_status;
          if ((event_mask & visible_status & session->second.exception_enable_mask &
               ~already_delivered) == 0)
            continue;
          pending.mask |= event_mask;
          pending.claims.push_back({.mask = event_mask});
        }
        if (pending.mask != 0)
          queues.push_back(std::move(pending));
      }
    };
    auto collect_process_mask = [&] {
      process_claims.clear();
      std::lock_guard<std::mutex> event_lock(debug_process_events_mutex_);
      auto events = debug_process_events_.find(target_pid);
      if (events == debug_process_events_.end())
        return uint64_t{0};
      uint64_t mask = 0;
      const uint64_t unavailable = session->second.notified_process_exception_mask |
                                   session->second.pending_process_exception_mask;
      for (const auto &[_, source] : events->second) {
        for (uint64_t event_mask : source.events) {
          event_mask &= source.mask;
          if ((event_mask & session->second.exception_enable_mask & ~unavailable) == 0)
            continue;
          mask |= event_mask;
          process_claims.push_back({.mask = event_mask});
        }
      }
      return mask;
    };

    collect_queues();
    process_mask = collect_process_mask();
    const bool retry_query =
        session->second.notification_query_retry_needed && has_visible_queue_event;
    if (queues.empty() && process_mask == 0 && !retry_query)
      return 0;
    notifier = UniqueDriverFd(duplicate_debug_notifier(session->second.dbg_fd));
    if (notifier.get() < 0) {
      session->second.notification_retry_needed = true;
      debug_sessions_cv_.notify_one();
      return -errno;
    }

    if (retry_query) {
      // A debugger that already drained the first wake must get another after
      // the event becomes visible. Serialize this bounded recovery write with
      // QUERY and session changes; it makes no new ownership claim and cannot
      // hide the committed event again. No CU or allocation lock is held.
      const int result =
          write_debug_notification(notifier.get(), {}, debug_notification_clone3_error_for_testing_,
                                   debug_notification_clone_pidfd_error_for_testing_,
                                   debug_notification_deferred_reap_for_testing_);
      if (result != 0) {
        session->second.notification_retry_needed = true;
        debug_sessions_cv_.notify_one();
        return result;
      }
      session->second.notification_query_retry_needed = false;
    }

    // Recompute after duplication so a concurrently resolved queue event is
    // never marked pending merely because it existed before the syscall.
    collect_queues();
    process_mask = collect_process_mask();
    if (queues.empty() && process_mask == 0)
      return 0;
    session_generation = session->second.generation;
    clone3_error_for_testing = debug_notification_clone3_error_for_testing_;
    clone_pidfd_error_for_testing = debug_notification_clone_pidfd_error_for_testing_;
    defer_reap_for_testing = debug_notification_deferred_reap_for_testing_;
    if (invoke_result_hook) {
      result_hook = debug_notification_result_hook_for_testing_;
      write_hook = debug_notification_write_hook_for_testing_;
    }
    if (process_mask != 0) {
      for (auto &claim : process_claims)
        claim.claim_id = begin_notification_claim(session->second, claim.mask);
      session->second.begin_process_notification(process_mask);
    }
    if (proc) {
      std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
      for (auto &pending : queues) {
        auto queue = proc->queue_snapshot_map_.find(pending.queue_id);
        if (queue != proc->queue_snapshot_map_.end()) {
          for (auto &claim : pending.claims)
            claim.claim_id = begin_notification_claim(session->second, claim.mask);
          queue->second.begin_debug_notification(pending.mask, session_generation);
        }
      }
    }
  }

  const int result =
      write_debug_notification(notifier.get(), write_hook, clone3_error_for_testing,
                               clone_pidfd_error_for_testing, defer_reap_for_testing);
  if (result_hook)
    result_hook(result == 0);

  bool retry_needed = false;
  {
    std::lock_guard<std::mutex> lock(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || session->second.generation != session_generation) {
      if (session != debug_sessions_.end() && session->second.enabled)
        session->second.notification_retry_needed = true;
      debug_sessions_cv_.notify_one();
      return result;
    }
    for (auto &claim : process_claims)
      claim.continuously_subscribed =
          claim.claim_id != 0 &&
          finish_notification_claim(session->second, claim.claim_id, &claim.consumed_mask);
    for (auto &pending : queues)
      for (auto &claim : pending.claims)
        claim.continuously_subscribed =
            claim.claim_id != 0 && finish_notification_claim(session->second, claim.claim_id);
    if (!session->second.enabled)
      return result;
    session->second.finish_process_notification(process_mask);
    uint64_t delivered_process_mask = 0;
    for (const auto &claim : process_claims) {
      const bool still_subscribed = (claim.mask & session->second.exception_enable_mask) != 0;
      if (result == 0 && claim.continuously_subscribed && still_subscribed)
        delivered_process_mask |= claim.mask & ~claim.consumed_mask;
      else if (still_subscribed)
        session->second.notification_retry_needed = true;
      // Another event with the same bit may have arrived after QUERY consumed
      // this claim but before its pending count was released.
      if (claim.consumed_mask != 0 && still_subscribed)
        session->second.notification_retry_needed = true;
    }
    session->second.notified_process_exception_mask |= delivered_process_mask;
    if (proc) {
      std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
      for (const auto &pending : queues) {
        auto queue = proc->queue_snapshot_map_.find(pending.queue_id);
        if (queue != proc->queue_snapshot_map_.end() &&
            queue->second.debug_notification_session_generation == session_generation) {
          uint64_t delivered_queue_mask = 0;
          for (const auto &claim : pending.claims) {
            const bool still_subscribed = (claim.mask & session->second.exception_enable_mask) != 0;
            if (result == 0 && claim.continuously_subscribed && still_subscribed)
              delivered_queue_mask |= claim.mask;
            else if (still_subscribed)
              session->second.notification_retry_needed = true;
          }
          queue->second.finish_debug_notification(pending.mask, delivered_queue_mask);
        }
      }
    }
    session->second.notification_retry_needed |= session->second.notification_query_retry_needed;
    retry_needed = session->second.notification_retry_needed;
  }
  if (retry_needed)
    debug_sessions_cv_.notify_one();
  return result;
}

void SimulatedKfd::complete_runtime_queue_exception(uint32_t process_id, uint32_t queue_id,
                                                    uint64_t exception_mask, bool delivered,
                                                    bool retain_failure) {
  auto proc = find_process(process_id);
  if (!proc)
    return;

  bool retry_needed = false;
  {
    std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return;
    const uint64_t visible_before = queue->second.debugger_visible_exception_status();
    const uint64_t failed_mask =
        queue->second.finish_runtime_exception(exception_mask, delivered, retain_failure);
    retry_needed = failed_mask != 0 ||
                   (queue->second.debugger_visible_exception_status() & ~visible_before) != 0;
  }
  if (retry_needed) {
    [[maybe_unused]] const int notification_result =
        retry_debug_notifications(proc->client_pid(), /*invoke_result_hook=*/true);
  }
}

bool SimulatedKfd::signal_runtime_queue_exception(uint32_t gpu_id, uint32_t queue_id,
                                                  uint32_t process_id, uint64_t exception_mask,
                                                  bool wait_for_ack) {
  auto *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc || exception_mask == 0)
    return false;

  // Every XCD owns a replica of a fanned-out queue. Stop every replica before
  // publishing the shared status, but do not hold the per-queue publication
  // mutex while entering a CU: releasing its wave-state guard can flush another
  // exception back through this function.
  amdgpu::CommandProcessor *publisher = nullptr;
  gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
    if (cp->signal_queue_exception(queue_id, process_id, exception_mask,
                                   /*publish_interrupt=*/false) &&
        publisher == nullptr)
      publisher = cp;
  });
  if (publisher == nullptr)
    return false;

  // Concurrent reports for one queue share its ROCr status word. Serialize
  // that read-modify-write, interrupt, and acknowledgement wait without
  // blocking independent processes or queues behind a stalled runtime.
  const uint64_t key = (static_cast<uint64_t>(process_id) << 32) | queue_id;
  std::shared_ptr<QueueExceptionLock> publication_lock;
  {
    std::lock_guard<std::mutex> lock(queue_exception_locks_mutex_);
    auto &entry = queue_exception_locks_[key];
    if (!entry)
      entry = std::make_shared<QueueExceptionLock>();
    publication_lock = entry;
    ++publication_lock->users;
  }
  bool published = false;
  {
    std::lock_guard<std::mutex> lock(publication_lock->publication_mutex);
    published =
        publisher->publish_queue_exception(queue_id, process_id, exception_mask, wait_for_ack);
  }
  if (queue_exception_cleanup_hook_for_testing_)
    queue_exception_cleanup_hook_for_testing_(false);
  // A publisher joins and leaves under the registry lock, so the last user
  // always observes zero and erases the entry. The identity check prevents a
  // stale releaser from erasing a replacement allocated for the same key.
  {
    std::lock_guard<std::mutex> lock(queue_exception_locks_mutex_);
    const bool last_user = --publication_lock->users == 0;
    if (last_user) {
      auto entry = queue_exception_locks_.find(key);
      if (entry != queue_exception_locks_.end() && entry->second == publication_lock)
        queue_exception_locks_.erase(entry);
    }
  }
  if (queue_exception_cleanup_hook_for_testing_)
    queue_exception_cleanup_hook_for_testing_(true);
  return published;
}

std::optional<amdgpu::ComputeUnitCore::TrapHandlerConfig>
SimulatedKfd::resolve_trap_handler(const amdgpu::Wavefront &wave, uint32_t gpu_ordinal) {
  std::shared_ptr<KfdProcess> proc;
  uint64_t tba = 0;
  uint64_t tma = 0;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    auto proc_it = processes_.find(wave.process_id());
    if (proc_it == processes_.end() || gpu_ordinal >= proc_it->second->gpu_state_.size())
      return std::nullopt;
    proc = proc_it->second;
    tba = proc->gpu(gpu_ordinal).trap_tba_addr;
    tma = proc->gpu(gpu_ordinal).trap_tma_addr;
  }
  if (tba == 0)
    return std::nullopt;

  bool debug_enabled = false;
  {
    std::lock_guard<std::mutex> debug_lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    debug_enabled = session != debug_sessions_.end() && session->second.enabled;
  }
  // Do not tell the trap handler a debugger is attached on a part whose stops
  // could never be published; the handler would raise STATUS.HALT and wait for
  // a debugger that can never read it.
  if (debug_enabled && gpu_ordinal < gpus_.size() &&
      !debug_stop_publishable(gpus_[gpu_ordinal].gpu_id))
    debug_enabled = false;
  return amdgpu::ComputeUnitCore::TrapHandlerConfig{tba, tma, debug_enabled};
}

bool SimulatedKfd::on_wave_sendmsg(amdgpu::Wavefront &wave, uint32_t message) {
  constexpr uint32_t kMessageIdMask = 0xFu;
  constexpr uint32_t kMessageInterrupt = 1;
  constexpr uint32_t kMessageGetDoorbell = 10;
  constexpr uint32_t kDoorbellIdBits = 10;
  constexpr uint32_t kRuntimeQueueExceptionMask = 0x3Fu;
  const uint32_t message_id = message & kMessageIdMask;

  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;

  if (message_id == kMessageGetDoorbell) {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto doorbell = proc->queue_doorbell_map_.find(wave.queue_id());
    if (doorbell == proc->queue_doorbell_map_.end())
      return false;
    const uint32_t doorbell_id = (doorbell->second.doorbell_offset / sizeof(uint64_t)) & 0x3FFu;
    // The handler sets EXEC_LO[31] before issuing MSG_GET_DOORBELL and polls
    // until hardware clears it. Return the 10-bit ID with the pending bit clear.
    wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | doorbell_id);
    return true;
  }

  if (message_id != kMessageInterrupt || !wave.in_trap_handler())
    return false;

  const auto arch = wave.cu().arch();
  const auto interrupt_abi = kmd::detail::trap_interrupt_abi(arch);
  bool profiling_interrupt = false;
  if (interrupt_abi == kmd::detail::TrapInterruptAbi::PreGfx12) {
    // ROCr's pre-GFX12 handler has two MSG_INTERRUPT sites. At the profiling
    // site it loads the event id from TTMP7; at the queue-exception site it
    // loads the packed exception from TTMP3. Both insert an s_nop immediately
    // before the message. Classify the actual control-flow site instead of
    // inspecting the TTMP values: a signal pointer can equal saved M0, and a
    // concurrent trap can legitimately rewrite those registers before the
    // second interrupt.
    if (wave.pc >= 2 * sizeof(uint32_t)) {
      const uint32_t payload_move =
          wave.cu().fetch_instruction_word(wave.pc - 2 * sizeof(uint32_t), wave.process_id());
      const uint32_t delay =
          wave.cu().fetch_instruction_word(wave.pc - sizeof(uint32_t), wave.process_id());
      const auto site =
          kmd::detail::classify_pre_gfx12_trap_interrupt_site(arch, payload_move, delay);
      if (site == kmd::detail::TrapInterruptSite::Profiling)
        profiling_interrupt = true;
      else if (site != kmd::detail::TrapInterruptSite::QueueException)
        return false;
    } else {
      return false;
    }
  } else if (interrupt_abi == kmd::detail::TrapInterruptAbi::Gfx12) {
    // GFX12 exposes the host/performance causes in common TRAPSTS bits 22/26.
    profiling_interrupt = (wave.trapsts() & ((1u << 22) | (1u << 26))) != 0;
  } else
    return false;
  // Profiling completion uses MSG_INTERRUPT too, but puts an event id in M0.
  // Do not reinterpret that id as the queue-exception layout below.
  if (profiling_interrupt)
    return true;

  // The ROCr trap-handler ABI packs the 10-bit doorbell id below six KFD queue
  // exception bits in M0. A subscribed debugger owns the complete event and its
  // CWSR publication is deferred until s_rfe applies the handler's STATUS.HALT;
  // otherwise ROCr owns it. The CU defers the CP call until its wave-state lock
  // is released.
  const uint64_t exception_status = (wave.m0() >> kDoorbellIdBits) & kRuntimeQueueExceptionMask;
  if (exception_status != 0) {
    wave.add_trap_queue_exception_status(exception_status);
    const uint64_t unreported = exception_status & ~wave.trap_runtime_exception_status();
    if (unreported != 0) {
      const uint64_t debugger_status = debugger_queue_exception_mask(
          proc, wave.queue_id(), unreported, /*reserve_runtime_if_unclaimed=*/true);
      if (debugger_status == 0)
        defer_wave_exception_to_runtime(wave, unreported, /*suspend_while_pending=*/false,
                                        /*clear_debug_stop_on_success=*/false,
                                        /*runtime_already_reserved=*/true);
    }
  }

  return true;
}

bool SimulatedKfd::debug_stop_publishable(uint32_t gpu_id) {
  auto *gpu = find_gpu(gpu_id);
  if (gpu == nullptr || gpu->soc == nullptr)
    return false;
  // The codec reproduces the gfx9.4 CWSR layout only (kmd/linux/cwsr.h), and
  // the differences elsewhere are not confined to one field -- the control
  // stack, the COMPUTE_RELAUNCH bits, the wave64 VGPR stride, the SGPR alias
  // slots and the dispatch-identity TTMPs all move. rocm-dbgapi would decode
  // the image against its own layout and act on whatever it read.
  //
  // Asked here, before the stop, rather than refused at DBG_TRAP_ENABLE: every
  // errno that path can return except ESRCH and EALREADY becomes
  // AMD_DBGAPI_STATUS_ERROR, which rocm-dbgapi turns into a [[noreturn]]
  // fatal_error, so refusing there aborts the debugger instead of declining.
  // Declining cleanly is the topology's job (HSA_CAP_TRAP_DEBUG_SUPPORT), and
  // that surface is deliberately kept identical to the real KFD driver.
  if (kmd::cwsr_layout_modelled(gpu->soc->arch()))
    return true;
  std::lock_guard<std::mutex> warning_lock(cwsr_layout_warning_mutex_);
  if (!gpu->cwsr_layout_warned) {
    gpu->cwsr_layout_warned = true;
    util::Logger::warn(
        "wave stops not published on gpu_id=", gpu_id,
        ": no CWSR record layout is modelled for arch=", static_cast<int>(gpu->soc->arch()),
        "; GPU debugging is supported on gfx942/gfx950/gfx1250 only");
  }
  return false;
}

bool SimulatedKfd::report_wave_stopped(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                       uint32_t gpu_id, uint64_t ctx_base, uint32_t ctx_size,
                                       uint64_t exception_mask, bool retain_on_rejection,
                                       bool *runtime_failure_debuggable) {
  if (runtime_failure_debuggable != nullptr)
    *runtime_failure_debuggable = false;
  // Serialization must succeed before the debugger is woken. Event publication
  // latches per-queue exception status and queues an event the debugger will
  // answer with SUSPEND_QUEUES; without a record that request can only come
  // back as a queue error, and the wave stays halted with nothing able to
  // resume it. The stop itself cannot be deferred until after serialization --
  // the serializer selects waves by debug_stopped() -- so the caller undoes it.
  if (!serialize_queue_debug_waves(proc->process_id(), queue_id, gpu_id, ctx_base, ctx_size)) {
    if (retain_on_rejection)
      reserve_runtime_queue_exception(proc, queue_id, exception_mask);
    return false;
  }
  if (runtime_failure_debuggable != nullptr)
    *runtime_failure_debuggable = true;
  apply_debug_event_publication_hook_for_testing(proc);
  return notify_debug_event(proc, queue_id, exception_mask, retain_on_rejection,
                            /*reserve_runtime_on_rejection=*/retain_on_rejection);
}

bool SimulatedKfd::notify_debug_event(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                      uint64_t exception_mask, bool retain_on_rejection,
                                      bool reserve_runtime_on_rejection) {
  const pid_t target_pid = proc->client_pid();

  // Latch status whenever debugging is active, like kfd_dbg_ev_raise(). The
  // current subscription controls notification and ownership, while QUERY and
  // a later SET_EXCEPTIONS_ENABLED can still observe retained status.
  UniqueDriverFd notifier;
  bool subscribed = false;
  uint64_t session_generation = 0;
  uint64_t notification_claim_id = 0;
  std::optional<int> clone3_error_for_testing;
  std::optional<int> clone_pidfd_error_for_testing;
  bool defer_reap_for_testing = false;
  bool notification_pending = false;
  std::function<void(bool)> result_hook;
  std::function<void()> write_hook;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled) {
      if (reserve_runtime_on_rejection)
        reserve_runtime_queue_exception(proc, queue_id, exception_mask);
      return false;
    }
    session_generation = session->second.generation;
    clone3_error_for_testing = debug_notification_clone3_error_for_testing_;
    clone_pidfd_error_for_testing = debug_notification_clone_pidfd_error_for_testing_;
    defer_reap_for_testing = debug_notification_deferred_reap_for_testing_;
    subscribed = (session->second.exception_enable_mask & exception_mask) != 0;
    if (subscribed && session->second.dbg_fd >= 0)
      notifier = UniqueDriverFd(duplicate_debug_notifier(session->second.dbg_fd));
    if ((!subscribed || notifier.get() < 0) && !retain_on_rejection)
      return false;
    std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    queue->second.exception_status |= exception_mask;
    queue->second.record_debug_notification_event(exception_mask);
    if (retain_on_rejection)
      queue->second.debug_notification_retained_status |= exception_mask;
    if (subscribed && notifier.get() >= 0) {
      notification_pending = true;
      notification_claim_id = begin_notification_claim(session->second, exception_mask);
      queue->second.begin_debug_notification(exception_mask, session_generation);
      result_hook = debug_notification_result_hook_for_testing_;
      write_hook = debug_notification_write_hook_for_testing_;
    }
    if (!notification_pending && reserve_runtime_on_rejection)
      queue->second.begin_runtime_exception(exception_mask);
    if (!notification_pending && retain_on_rejection && subscribed)
      session->second.notification_retry_needed = true;
  }
  if (!notification_pending) {
    if (retain_on_rejection)
      debug_sessions_cv_.notify_one();
    return false;
  }
  const bool delivered =
      write_debug_notification(notifier.get(), write_hook, clone3_error_for_testing,
                               clone_pidfd_error_for_testing, defer_reap_for_testing) == 0;
  if (result_hook)
    result_hook(delivered);

  bool accepted = false;
  bool retry_needed = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || session->second.generation != session_generation) {
      if (session != debug_sessions_.end() && session->second.enabled)
        session->second.notification_retry_needed = true;
      if (reserve_runtime_on_rejection)
        reserve_runtime_queue_exception(proc, queue_id, exception_mask);
      debug_sessions_cv_.notify_one();
      return false;
    }
    const bool continuously_subscribed =
        finish_notification_claim(session->second, notification_claim_id);
    if (!session->second.enabled) {
      if (reserve_runtime_on_rejection)
        reserve_runtime_queue_exception(proc, queue_id, exception_mask);
      return false;
    }
    std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end() ||
        queue->second.debug_notification_session_generation != session_generation)
      return false;
    const bool still_subscribed = (exception_mask & session->second.exception_enable_mask) != 0;
    queue->second.finish_debug_notification(
        exception_mask,
        delivered && still_subscribed && continuously_subscribed ? exception_mask : 0);
    accepted = delivered && still_subscribed && continuously_subscribed;
    if (!accepted && reserve_runtime_on_rejection)
      queue->second.begin_runtime_exception(exception_mask);
    if ((!accepted && retain_on_rejection && still_subscribed) ||
        session->second.notification_query_retry_needed)
      session->second.notification_retry_needed = true;
    retry_needed = session->second.notification_retry_needed;
  }
  if (retry_needed)
    debug_sessions_cv_.notify_one();
  return accepted;
}

bool SimulatedKfd::defer_wave_exception_to_runtime(
    amdgpu::Wavefront &wave, uint64_t exception_mask, bool suspend_while_pending,
    bool clear_debug_stop_on_success, bool runtime_already_reserved,
    bool runtime_failure_debuggable, amdgpu::QueueExceptionDeliveryPolicy delivery_policy,
    bool retain_wave) {
  auto proc = find_process(wave.process_id());
  if (proc && !runtime_already_reserved)
    reserve_runtime_queue_exception(proc, wave.queue_id(), exception_mask);
  wave.set_fatal_exception_pending(true);
  if (suspend_while_pending)
    wave.set_debug_suspended(true);
  const bool queued = wave.cu().signal_queue_exception(
      wave.queue_id(), wave.process_id(), exception_mask, clear_debug_stop_on_success,
      runtime_failure_debuggable, delivery_policy, retain_wave);
  if (!queued && proc) {
    if (runtime_exception_result_hook_for_testing_)
      runtime_exception_result_hook_for_testing_(false);
    complete_runtime_queue_exception(wave.process_id(), wave.queue_id(), exception_mask,
                                     /*delivered=*/false, runtime_failure_debuggable);
  }
  if (!queued && delivery_policy == amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail) {
    if (auto *sim_engine = wave.cu().engine())
      sim_engine->request_exit("Failed to queue an execution-failure notification", /*code=*/1);
  }
  // The frontend owns the notification even if it cannot be queued. Retained
  // waves stay held until delivery; callers passing retain_wave=false abort the
  // dispatch and must not leave a borrowed wave pointer in the notification.
  return true;
}

void SimulatedKfd::set_debug_event_claim_mask_for_testing(uint64_t exception_mask) {
  std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
  debug_event_claim_mask_for_testing_ = exception_mask;
}

void SimulatedKfd::detach_debug_event_claim_for_testing() {
  std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
  debug_event_claim_detach_for_testing_ = true;
}

void SimulatedKfd::apply_debug_event_publication_hook_for_testing(
    const std::shared_ptr<KfdProcess> &proc) {
  const pid_t target_pid = proc->client_pid();
  std::unique_lock<std::mutex> lk(debug_sessions_mutex_);
  if (debug_event_claim_detach_for_testing_) {
    debug_event_claim_detach_for_testing_ = false;
    debug_sessions_.erase(target_pid);
    lk.unlock();
    {
      std::lock_guard<std::mutex> event_lock(debug_process_events_mutex_);
      debug_process_events_.erase(target_pid);
    }
    std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
    for (auto &entry : proc->queue_snapshot_map_)
      entry.second.clear_debugger_exception_state();
    return;
  }
  if (!debug_event_claim_mask_for_testing_)
    return;
  const uint64_t exception_mask = *debug_event_claim_mask_for_testing_;
  debug_event_claim_mask_for_testing_.reset();
  auto session = debug_sessions_.find(target_pid);
  if (session != debug_sessions_.end())
    session->second.exception_enable_mask = exception_mask;
}

bool SimulatedKfd::on_wave_single_step_complete(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint64_t ctx_base = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id)) {
    // Clear single-step even while declining. The compute unit discards this
    // callback's result and re-enters it for as long as the flag is set, so
    // returning false without clearing it spins on every instruction.
    wave.set_debug_single_step(false);
    return false;
  }
  amdgpu::Wavefront::DebugStopState saved = wave.debug_stop_state();
  // The completed step is consumed even when publication loses its debugger
  // claim. Restoring single_step would make ComputeUnitCore call us after every
  // subsequent instruction because it intentionally ignores our return value.
  saved.single_step = false;
  wave.set_debug_single_step(false);
  wave.debug_trap(0);
  // gfx9.4 reports completed single-step through TRAPSTS.TRAP_AFTER_INST. This
  // is the public stop-reason bit rocm-dbgapi consumes before the next resume.
  constexpr uint32_t kTrapAfterInstMask = 1u << 25;
  wave.set_trapsts(wave.trapsts() | kTrapAfterInstMask);
  // Single-step completion is wave-specific. Report it even if another wave
  // in the queue is running; rocm-dbgapi will suspend the queue before reading
  // its CWSR state. Deferring until every peer stops can lose the event forever
  // when a peer runs to normal completion without entering a debug callback.
  // Like hardware's trap interrupt, notification does not itself save the
  // queue. The debugger's ensuing SUSPEND_QUEUES request publishes one stable,
  // authoritative CWSR snapshot instead of redundantly serializing every
  // resident wave here first.
  apply_debug_event_publication_hook_for_testing(proc);
  if (!notify_debug_event(proc, wave.queue_id(), KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
                          /*retain_on_rejection=*/false)) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_watchpoint(amdgpu::Wavefront &wave, uint64_t address, uint32_t bytes,
                                      bool is_write, bool is_atomic) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint32_t matched_slots = 0;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
    // The watch modes are not disjoint, so an access can satisfy more than one
    // of them: NONREAD is documented as "write or atomic operations only"
    // (kfd_ioctl.h, and rocdbgapi's os_driver.h maps the STORE_AND_RMW
    // watchpoint kind a plain `(gdb) watch` requests onto it), and ALL matches
    // everything. Hand the set of modes this access triggers to the session
    // rather than a single mode, so a write watch still fires on an atomic
    // read-modify-write. KfdProcess deliberately does not include kfd_ioctl.h,
    // so the mode semantics stay here, on the side of the boundary that owns
    // the KFD ABI.
    const uint32_t matching_modes =
        (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL) |
        (is_atomic ? (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC) |
                         (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
         : is_write ? (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
                    : (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ));
    matched_slots = session->second.matching_address_watch_slots(address, bytes, matching_modes);
  }
  if (matched_slots == 0)
    return false;

  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id))
    return false;

  static constexpr uint32_t kTrapstsBits[] = {1u << 7, 1u << 12, 1u << 13, 1u << 14};
  const auto saved = wave.debug_stop_state();
  uint32_t trapsts = wave.trapsts();
  for (uint32_t slot = 0; slot < KfdProcess::DebugSession::kMaxAddressWatches; ++slot)
    if ((matched_slots & (uint32_t{1} << slot)) != 0)
      trapsts |= kTrapstsBits[slot];
  wave.set_trapsts(trapsts);
  if (wave.uses_separate_trap_ctrl()) {
    constexpr uint32_t kTrapCtrlAddrWatch = 1u << 7;
    wave.set_gfx12_trap_ctrl_raw(wave.gfx12_trap_ctrl_raw() | kTrapCtrlAddrWatch);
  } else {
    constexpr uint32_t kModeExcpEnAddrWatch = 1u << 19;
    wave.set_mode_raw(wave.mode_raw() | kModeExcpEnAddrWatch);
  }
  wave.debug_trap(0);
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size,
                           KFD_EC_MASK(EC_QUEUE_WAVE_TRAP),
                           /*retain_on_rejection=*/false)) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_illegal_instruction(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION);
  bool debugger_enabled = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    debugger_enabled = session != debug_sessions_.end() && session->second.enabled;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  bool has_runtime_signal = false;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
    has_runtime_signal = queue->second.has_runtime_exception_signal;
  }
  if (!debugger_enabled || ctx_base == 0 || !debug_stop_publishable(gpu_id)) {
    if (!has_runtime_signal)
      return false;
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/false,
                                           /*runtime_failure_debuggable=*/false,
                                           amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail);
  }
  constexpr uint32_t kTrapstsIllegalInst = 1u << 11;
  const auto saved = wave.debug_stop_state();
  wave.set_trapsts(wave.trapsts() | kTrapstsIllegalInst);
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  bool runtime_failure_debuggable = false;
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size, kException,
                           /*retain_on_rejection=*/true, &runtime_failure_debuggable)) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimulatedKfd::on_wave_execution_failure(amdgpu::Wavefront &wave, bool retain_wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  {
    std::lock_guard<std::mutex> lock(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end() || !queue->second.has_runtime_exception_signal)
      return false;
  }
  // A simulator capability failure or GWS timeout has no architectural wave-stop reason.
  // dbgapi cannot surface or forward a queue abort without such a wave stop;
  // giving it ownership would strand the queue. Deliver directly to ROCr even
  // with a debugger attached, preserving TRAPSTS and the debugger stop state.
  // ROCr acknowledges only once per queue and may run an error callback or
  // core dump first. Successful publication transfers ownership without waiting.
  return defer_wave_exception_to_runtime(
      wave, KFD_EC_MASK(EC_QUEUE_WAVE_ABORT),
      /*suspend_while_pending=*/retain_wave,
      /*clear_debug_stop_on_success=*/false,
      /*runtime_already_reserved=*/false,
      /*runtime_failure_debuggable=*/false,
      retain_wave ? amdgpu::QueueExceptionDeliveryPolicy::PublishOrFail
                  : amdgpu::QueueExceptionDeliveryPolicy::PublishAndCancel,
      retain_wave);
}

bool SimulatedKfd::on_wave_memory_violation(amdgpu::Wavefront &wave, uint64_t, bool) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id))
    return false;
  constexpr uint32_t kTrapstsXnackError = 1u << 28;
  const auto saved = wave.debug_stop_state();
  wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
  wave.debug_trap(0);
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION);
  bool runtime_failure_debuggable = false;
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size, kException,
                           /*retain_on_rejection=*/true, &runtime_failure_debuggable)) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

bool SimulatedKfd::on_wave_alu_exception(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id))
    return false;
  const auto saved = wave.debug_stop_state();
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  constexpr uint64_t kException = KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR);
  bool runtime_failure_debuggable = false;
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size, kException,
                           /*retain_on_rejection=*/true, &runtime_failure_debuggable)) {
    wave.restore_debug_stop_state(saved);
    return defer_wave_exception_to_runtime(wave, kException,
                                           /*suspend_while_pending=*/true,
                                           /*clear_debug_stop_on_success=*/false,
                                           /*runtime_already_reserved=*/true,
                                           runtime_failure_debuggable);
  }
  return true;
}

void SimulatedKfd::release_debuggee_state(pid_t target_pid, KfdProcess *target_proc) {
  // Everything a session imposed on its inferior, undone in one place so that
  // explicit detach and debugger death cannot drift apart. Callers have already
  // erased the session and must not hold debug_sessions_mutex_ here: this takes
  // CU wave-state locks, and the engine thread takes those first and then
  // debug_sessions_mutex_ from its trap/watchpoint callbacks, so holding both
  // in this order closes an AB-BA cycle against a wave that is trapping.
  {
    std::lock_guard<std::mutex> event_lock(debug_process_events_mutex_);
    debug_process_events_.erase(target_pid);
  }

  // Nothing below has a debuggee to release; the event purge above was the
  // whole job. Returning here keeps the rest free of repeated null tests.
  if (target_proc == nullptr) {
    release_debug_checks_if_last_session();
    return;
  }

  std::vector<std::pair<uint32_t, uint32_t>> queues;
  {
    std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
    for (auto &[queue_id, queue] : target_proc->queue_snapshot_map_) {
      queue.clear_debugger_exception_state();
      queues.emplace_back(queue_id, queue.gpu_id);
    }
  }

  for (const auto &[queue_id, gpu_id] : queues) {
    auto *gpu = find_gpu(gpu_id);
    if (!gpu || !gpu->soc)
      continue;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(queue_id, target_proc->process_id(), false);
      for (auto *cu : cp->compute_units()) {
        bool wake = false;
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (!wave || wave->is_halted() || wave->process_id() != target_proc->process_id() ||
                wave->queue_id() != queue_id)
              continue;
            if (wave->fatal_exception_pending()) {
              wave->set_fatal_exception_cwsr_valid(false);
              continue;
            }
            wave->set_debug_single_step(false);
            wave->set_debug_halted(false);
            wave->set_debug_suspended(false);
            // The architectural half of the same stop. s_sendmsghalt raises
            // STATUS.HALT and s_rfe consults it on the way out of the handler,
            // so leaving it set outlives the session: the wave would re-halt at
            // the handler return with no debugger left to resume it, and the
            // queue would never drain.
            wave->set_status_halt(false);
            wave->set_self_halted(false);
            wake = true;
          }
        });
        if (wake)
          cu->schedule_work_async();
      }
    });
  }

  revoke_target_mem_routing(target_proc->process_id());
  release_debug_checks_if_last_session();
}

void SimulatedKfd::revoke_target_mem_routing(uint32_t process_id) {
  std::shared_ptr<KfdProcess> process = find_process(process_id);
  if (process == nullptr)
    return;
  for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
    GpuDevice &gpu = gpus_[ordinal];
    if (gpu.soc == nullptr)
      continue;
    const amdgpu::AddressSpaceHandle address_space = process->gpu(ordinal).address_space;
    if (address_space)
      (void)gpu.legacy_vm->set_client_mem_fd(address_space, -1);
  }
}

void SimulatedKfd::release_debug_checks_if_last_session() {
  // Debug checks stay on only while some session still wants them. Read the
  // answer under the lock, then reach into the CUs after releasing it -- the
  // whole point of this function running unlocked is that debug_sessions_mutex_
  // is never held while touching a CU.
  bool no_sessions_left = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    no_sessions_left = debug_sessions_.empty();
  }
  if (no_sessions_left)
    set_debug_active_on_all_cus(false);
}

void SimulatedKfd::set_debug_active_on_all_cus(bool active) {
  for (auto &gpu : gpus_) {
    if (!gpu.soc)
      continue;
    gpu.soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units())
        cu->set_debug_active(active);
    });
  }
}

UniqueDriverFd SimulatedKfd::duplicate_debug_target_mem(pid_t target_pid) const {
  std::lock_guard<std::mutex> lock(debug_sessions_mutex_);
  auto session = debug_sessions_.find(target_pid);
  if (session == debug_sessions_.end() || session->second.target_mem_fd.get() < 0)
    return {};
  return UniqueDriverFd(safe_fcntl(session->second.target_mem_fd.get(), F_DUPFD_CLOEXEC, 0));
}

void SimulatedKfd::apply_cwsr_to_wave(amdgpu::Wavefront &wave, const kmd::CwsrWaveState &state,
                                      rj_code_arch_t arch) {
  constexpr uint32_t kModeDebugEnMask = 1u << 11;
  constexpr uint32_t kStatusHaltMask = amdgpu::Wavefront::kStatusHaltMask;
  const bool gfx12_5 = kmd::cwsr_layout_kind(arch) == kmd::CwsrLayoutKind::Gfx12_5;
  wave.pc = state.pc;
  // Mirror of the un-shadowing in build_cwsr_wave_state(): while a handler is
  // mid-flight the published EXEC is the interrupted application mask, so the
  // value coming back belongs to the shadow, not to the live register the
  // handler is executing under. s_rfe installs it when the handler returns.
  if (wave.in_trap_handler())
    wave.set_trap_saved_exec(state.exec);
  else
    wave.set_exec(state.exec);
  wave.set_vcc_raw(state.vcc);
  wave.set_m0(state.m0);
  // STATUS shadows the same way EXEC does, and the cost of getting it wrong is
  // higher: the ROCr handler raises STATUS.HALT and only then returns, so a
  // record applied in that window used to clear the HALT the handler had just
  // set. s_rfe then saw a running wave, resumed it, and the breakpoint was
  // silently lost -- one inferior in gdb.rocm/multi-inferior-stress.exp running
  // its kernel to completion and exiting without ever stopping. Route the
  // record's HALT to the interrupted state the handler restores and leave the
  // live register to the handler.
  const uint32_t restored_status = gfx12_5 ? internal_status_from_gfx1250(state)
                                           : (state.status & ~kStatusHaltMask) |
                                                 (state.saved_status_halt ? kStatusHaltMask : 0u);
  if (wave.in_trap_handler())
    wave.set_trap_saved_status(restored_status);
  else
    wave.set_status_raw(restored_status);
  // The live STATUS.HALT stays as it is for a handler that raised it with
  // s_setreg: there the bit is the handler's own request to keep the wave
  // stopped, and clearing it on a resume loses the breakpoint --
  // DbgTrapCwsrShadowsTrapHandlerRegistersAndRoutesDebuggerEdits pins that.
  // A wave halted at s_sendmsghalt sits in the same window and wants the
  // opposite: it has already reported and is waiting to be let go, so leaving
  // the bit set strands it at the handler return. The record cannot tell the
  // two apart, which is why the wave records who raised the bit.
  if (!state.wave_stopped && wave.self_halted()) {
    wave.set_status_halt(false);
    wave.set_self_halted(false);
  }
  wave.set_mode_raw(state.mode);
  wave.set_trapsts(gfx12_5 ? internal_trapsts_from_gfx1250(state) : state.trapsts);
  if (gfx12_5) {
    wave.set_gfx12_excp_flag_user_extra_raw(state.excp_flag_user & (0x3u << 30));
    wave.set_gfx12_trap_ctrl_raw(state.trap_ctrl);
    wave.set_gfx1250_xnack_state_priv_raw(state.xnack_state_priv);
    wave.set_gfx1250_xnack_mask_raw(state.xnack_mask);
  }
  wave.set_debug_wave_id(state.wave_id);
  wave.set_aql_packet_id(state.queue_packet_id);
  const uint32_t stack_pointer = wave.debug_read_sgpr(32);
  const uint32_t stack_frame = wave.debug_read_sgpr(33);
  // rocm-dbgapi may submit a lightweight control-state update with no register
  // payload after displaced stepping. Preserve the SGPR/VGPR values produced
  // by the displaced instruction in that case rather than restoring zeros.
  //
  // Skip exactly the slots the codec treats as aliases, not everything above
  // the architected count: s104/s105 sit between the FLAT_SCRATCH and VCC
  // aliases and are ordinary registers the debugger may edit. Sharing
  // cwsr_sgpr_slot_is_aliased() with the codec is what keeps the two ends of
  // the round trip from disagreeing about which slots carry register state.
  if (!state.sgprs.empty())
    for (uint32_t s = 0; s < state.num_sgprs && s < state.sgprs.size(); ++s)
      if (s != 32 && s != 33 && !kmd::cwsr_sgpr_slot_is_aliased(s, arch))
        wave.debug_write_sgpr(s, state.sgprs[s]);
  if (!state.vgprs.empty())
    for (uint32_t r = 0; r < state.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        const size_t index = static_cast<size_t>(r) * 64 + lane;
        if (index < state.vgprs.size())
          wave.debug_write_vgpr(r, lane, state.vgprs[index]);
      }
  // FLAT_SCRATCH travels in its own field because it aliases two SGPR slots the
  // loop above skips. This used to save and re-install wave.scratch_base()
  // around the loop, which the loop cannot reach anyway (scratch_base_ is a
  // standalone member), so the only effect was discarding the debugger's edit.
  wave.set_scratch_base(state.flat_scratch);
  wave.debug_write_sgpr(32, stack_pointer);
  wave.debug_write_sgpr(33, stack_frame);
  const bool single_step = !state.wave_stopped && (gfx12_5 ? (state.trap_ctrl & (1u << 9)) != 0
                                                           : (state.mode & kModeDebugEnMask) != 0);
  wave.set_debug_single_step(single_step);
  wave.set_debug_halted(state.wave_stopped);
  // debug_suspended is deliberately left set here. Clearing it per wave made
  // each wave runnable the moment its own record was applied, so the engine
  // could execute it while its queue-mates -- and their LDS images -- were
  // still being restored. resume_debug_queues() commits the bit for every wave
  // in the queue in one pass once the whole restore has succeeded.
}

int SimulatedKfd::resume_debug_queues(KfdProcess *proc, uint32_t *queue_ids, uint32_t num_queues) {
  constexpr uint32_t kQueueError = uint32_t{1} << KFD_DBG_QUEUE_ERROR_BIT;
  constexpr uint32_t kQueueInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  constexpr uint32_t kQueueStatus = kQueueError | kQueueInvalid;
  struct QueueContext {
    uint32_t request_index = 0;
    uint32_t queue_id = 0;
    uint64_t base = 0;
    uint32_t size = 0;
    uint32_t gpu_id = 0;
  };
  std::vector<QueueContext> queues;
  if (proc) {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (uint32_t index = 0; index < num_queues; ++index) {
      const uint32_t queue_id = queue_ids[index] & ~kQueueStatus;
      queue_ids[index] = queue_id;
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue == proc->queue_snapshot_map_.end()) {
        queue_ids[index] |= kQueueInvalid;
        continue;
      }
      const auto &info = queue->second;
      // Completion releases the runtime reservation under this same lock.
      // Until then, even a valid CWSR image cannot recover the fatal stop:
      // publication may still fail, and its result does not stop waves again.
      if (info.runtime_exception_pending_status != 0) {
        queue_ids[index] |= kQueueError;
        continue;
      }
      queues.push_back({index, queue_id, info.ctx_save_restore_address,
                        info.ctx_save_restore_area_size, info.gpu_id});
    }
  } else {
    for (uint32_t index = 0; index < num_queues; ++index)
      queue_ids[index] = (queue_ids[index] & ~kQueueStatus) | kQueueInvalid;
  }
  uint32_t resumed = 0;
  for (const auto &context : queues) {
    auto *gpu = find_gpu(context.gpu_id);
    if (!gpu || !gpu->soc) {
      queue_ids[context.request_index] |= kQueueError;
      continue;
    }
    std::vector<amdgpu::Wavefront *> stopped;
    std::vector<kmd::CwsrWaveState> states;
    std::vector<amdgpu::ComputeUnitCore *> owners;
    const bool per_xcc_areas = gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5;
    const uint32_t area_count = per_xcc_areas ? std::max(1u, gpu->soc->num_xcds()) : 1u;
    std::vector<std::vector<kmd::CwsrWaveState>> states_by_area(area_count);
    auto collect = [&](amdgpu::CommandProcessor *cp, uint32_t area) {
      for (auto *cu : cp->compute_units()) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (wave && wave->debug_stopped() && wave->process_id() == proc->process_id() &&
                wave->queue_id() == context.queue_id) {
              stopped.push_back(wave);
              states_by_area[area].push_back(build_cwsr_wave_state(*wave, gpu->soc->arch()));
              owners.push_back(cu);
            }
          }
        });
      }
    };
    if (per_xcc_areas) {
      for (uint32_t xcc_id = 0; xcc_id < area_count; ++xcc_id)
        collect(gpu->soc->xcd(xcc_id)->command_processor(), xcc_id);
    } else {
      gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) { collect(cp, 0); });
    }
    bool restored = stopped.empty();
    if (!stopped.empty() && context.base != 0) {
      const std::optional<amdgpu::GpuVmAccess> vm_access =
          gpu->soc->gpu_vm().snapshot_vmid(proc->process_id());
      if (!vm_access) {
        queue_ids[context.request_index] |= kQueueError;
        continue;
      }
      UniqueDriverFd target_mem = duplicate_debug_target_mem(proc->client_pid());
      bool read_ok = true;
      auto read_block = [&](uint64_t address, std::span<uint8_t> bytes) {
        if (target_mem.get() < 0) {
          // A refused read leaves the span zero-filled, which deserializes as a
          // valid-looking context: waves would resume with cleared registers and
          // a zero program counter rather than staying stopped. The pread path
          // below already treats a short read that way, and this is the same
          // failure arriving through the memory model instead of the kernel.
          if (vm_access->read(address, std::as_writable_bytes(bytes)) !=
              amdgpu::VmAccessOutcome::Complete) {
            read_ok = false;
            util::Logger::warn("CWSR target read faulted: addr=0x", std::hex, address, std::dec,
                               " pid=", proc->client_pid());
          }
          return;
        }
        const ssize_t bytes_read =
            pread(target_mem.get(), bytes.data(), bytes.size(), static_cast<off_t>(address));
        if (bytes_read != static_cast<ssize_t>(bytes.size())) {
          read_ok = false;
          std::ranges::fill(bytes, 0);
          util::Logger::warn("CWSR target read failed: addr=0x", std::hex, address,
                             " pid=", std::dec, proc->client_pid(), " rc=", bytes_read,
                             " errno=", errno);
        }
      };
      restored = true;
      states.reserve(stopped.size());
      for (uint32_t area = 0; area < area_count; ++area) {
        auto &area_states = states_by_area[area];
        if (area_states.empty())
          continue;
        prepare_cwsr_wave_states(area_states);
        const uint64_t area_base = context.base + static_cast<uint64_t>(area) * context.size;
        restored &= kmd::deserialize_queue_cwsr_bulk(area_base, context.size, area_states,
                                                     read_block, gpu->soc->arch());
        states.insert(states.end(), std::make_move_iterator(area_states.begin()),
                      std::make_move_iterator(area_states.end()));
      }
      restored = restored && read_ok;
    }
    std::unordered_set<amdgpu::ComputeUnitCore *> wake;
    if (restored) {
      std::vector<size_t> matches;
      matches.reserve(states.size());
      std::vector<bool> matched(stopped.size());
      for (const auto &state : states) {
        auto find_match = [&](bool by_wave_id) {
          for (size_t index = 0; index < stopped.size(); ++index) {
            if (matched[index])
              continue;
            const auto *candidate = stopped[index];
            if (by_wave_id ? state.wave_id != 0 && candidate->debug_wave_id() == state.wave_id
                           : candidate->debug_wave_id() == 0 &&
                                 candidate->aql_packet_id() == state.queue_packet_id &&
                                 candidate->wg_coord() == state.group_ids &&
                                 candidate->wave_in_group() == state.wave_in_group)
              return index;
          }
          return stopped.size();
        };
        size_t index = find_match(true);
        if (index == stopped.size())
          index = find_match(false);
        if (index == stopped.size()) {
          restored = false;
          break;
        }
        matched[index] = true;
        matches.push_back(index);
      }
      restored = restored && matches.size() == stopped.size();
      if (restored) {
        for (size_t state_index = 0; state_index < states.size(); ++state_index) {
          const size_t wave_index = matches[state_index];
          if (!states[state_index].lds.empty())
            stopped[wave_index]->lds().write(stopped[wave_index]->lds_base(),
                                             states[state_index].lds.data(),
                                             static_cast<uint32_t>(states[state_index].lds.size()));
          owners[wave_index]->with_wave_state_locked([&] {
            apply_cwsr_to_wave(*stopped[wave_index], states[state_index], gpu->soc->arch());
          });
        }
      }
    }
    // This is the single commit point for debug_suspended: apply_cwsr_to_wave()
    // leaves every wave suspended, so none of them runs until the whole queue's
    // CWSR state and LDS have been restored above.
    //
    // Accumulated inside the locked loop below rather than read afterwards.
    // The loop clears debug_suspended, which makes these waves runnable, so
    // from that point the engine thread can be writing debug_halted_ -- a plain
    // bool -- concurrently. This value decides whether the queue gate reopens,
    // so reading it unlocked would race the answer as well as the byte.
    bool keep_dispatch_suspended = false;
    bool saw_runtime_exception = false;
    bool unresolved_runtime_exception = false;
    for (size_t index = 0; index < stopped.size(); ++index) {
      owners[index]->with_wave_state_locked([&] {
        const bool fatal_exception_pending = stopped[index]->fatal_exception_pending();
        const bool resolve_fatal_exception =
            restored && fatal_exception_pending && stopped[index]->fatal_exception_cwsr_valid();
        saw_runtime_exception |= fatal_exception_pending;
        unresolved_runtime_exception |= fatal_exception_pending && !resolve_fatal_exception;
        if (fatal_exception_pending)
          stopped[index]->set_fatal_exception_cwsr_valid(false);
        if (resolve_fatal_exception)
          stopped[index]->set_fatal_exception_pending(false);
        // A malformed or stale CWSR image must not strand a temporarily
        // suspended wave after the queue gate is released. Architecturally
        // halted waves remain halted until their CWSR record says otherwise.
        stopped[index]->set_debug_suspended(fatal_exception_pending && !resolve_fatal_exception);
        const bool halted = stopped[index]->debug_halted();
        keep_dispatch_suspended |= halted;
        if (!halted && !stopped[index]->is_halted()) {
          if (!fatal_exception_pending || resolve_fatal_exception)
            wake.insert(owners[index]);
        }
      });
    }
    if (saw_runtime_exception && !unresolved_runtime_exception) {
      std::lock_guard<std::mutex> lock(proc->alloc_mutex_);
      auto queue = proc->queue_snapshot_map_.find(context.queue_id);
      if (queue != proc->queue_snapshot_map_.end())
        queue->second.resolve_runtime_exceptions();
    }
    // Keep the queue-level launch gate closed until every resident wave has
    // consumed its CWSR state and become runnable. Reopen it before scheduling
    // those waves so CP completion processing cannot observe a queue as still
    // suspended if a resumed wave completes immediately.
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(context.queue_id, proc->process_id(), keep_dispatch_suspended,
                                    saw_runtime_exception && !unresolved_runtime_exception);
    });
    for (auto *cu : wake)
      cu->schedule_work_async();
    if (restored)
      ++resumed;
    else
      queue_ids[context.request_index] |= kQueueError;
  }
  return static_cast<int>(resumed);
}

int SimulatedKfd::suspend_debug_queues(KfdProcess *proc, uint32_t *queue_ids, uint32_t num_queues,
                                       uint64_t exception_mask) {
  constexpr uint32_t kQueueError = uint32_t{1} << KFD_DBG_QUEUE_ERROR_BIT;
  constexpr uint32_t kQueueInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  constexpr uint32_t kQueueStatus = kQueueError | kQueueInvalid;
  struct RequestedQueue {
    uint32_t request_index = 0;
    uint32_t queue_id = 0;
    KfdProcess::QueueSnapshotInfo info{};
  };
  std::vector<RequestedQueue> queues;
  if (!proc) {
    for (uint32_t index = 0; index < num_queues; ++index)
      queue_ids[index] = (queue_ids[index] & ~kQueueStatus) | kQueueInvalid;
    return 0;
  }
  const uint32_t process_id = proc->process_id();
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (uint32_t index = 0; index < num_queues; ++index) {
      const uint32_t queue_id = queue_ids[index] & ~kQueueStatus;
      queue_ids[index] = queue_id;
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue == proc->queue_snapshot_map_.end() ||
          (queue->second.exception_status & KFD_EC_MASK(EC_QUEUE_NEW)) != 0) {
        queue_ids[index] |= kQueueInvalid;
        continue;
      }
      queue->second.clear_debugger_exception_status(exception_mask);
      queues.push_back({index, queue_id, queue->second});
    }
  }
  uint32_t suspended = 0;
  for (const auto &queue : queues) {
    auto *gpu = find_gpu(queue.info.gpu_id);
    if (!gpu || !gpu->soc) {
      queue_ids[queue.request_index] |= kQueueError;
      continue;
    }
    // Ask before suspending anything. This path stops waves first and only then
    // tries to serialize them, so on an unmodelled architecture it would strand
    // every resident wave behind a kQueueError -- the same hole the wave-stop
    // callbacks have, reached through the debugger's own request instead.
    if (!debug_stop_publishable(queue.info.gpu_id)) {
      queue_ids[queue.request_index] |= kQueueError;
      continue;
    }
    std::vector<std::pair<amdgpu::ComputeUnitCore *, amdgpu::Wavefront *>> newly_suspended;
    std::vector<std::pair<amdgpu::ComputeUnitCore *, amdgpu::Wavefront *>> runtime_frozen;
    bool needs_serialization = false;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(queue.queue_id, process_id, true);
      for (auto *cu : cp->compute_units()) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (!wave || wave->is_halted() || wave->process_id() != process_id ||
                wave->queue_id() != queue.queue_id)
              continue;
            // Runtime exception routing freezes the wave with debug_suspended
            // so it cannot run, but it does not publish CWSR. Serialize that
            // retained fatal stop once; repeated SUSPEND_QUEUES calls must
            // preserve debugger edits in the already-published image.
            if (wave->fatal_exception_pending()) {
              runtime_frozen.emplace_back(cu, wave);
              needs_serialization |= !wave->fatal_exception_cwsr_valid();
            }
            if (wave->debug_suspended())
              continue;
            wave->set_debug_suspended(true);
            newly_suspended.emplace_back(cu, wave);
            needs_serialization = true;
          }
        });
      }
    });
    bool serialized = !needs_serialization;
    if (needs_serialization && queue.info.ctx_save_restore_address != 0)
      serialized = serialize_queue_debug_waves(process_id, queue.queue_id, queue.info.gpu_id,
                                               queue.info.ctx_save_restore_address,
                                               queue.info.ctx_save_restore_area_size);
    if (serialized) {
      if (needs_serialization)
        for (auto &[cu, wave] : runtime_frozen)
          cu->with_wave_state_locked([wave] { wave->set_fatal_exception_cwsr_valid(true); });
      ++suspended;
      continue;
    }
    // No record was published, so the debugger has nothing to resume from.
    // Undo the suspension rather than leave the queue wedged: reporting the
    // error and stranding the waves is strictly worse than reporting it alone.
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(queue.queue_id, process_id, false);
    });
    amdgpu::ComputeUnitCore *woken = nullptr;
    for (auto &[cu, wave] : newly_suspended) {
      cu->with_wave_state_locked([w = wave] { w->set_debug_suspended(false); });
      // Entries are appended compute unit by compute unit, so comparing against
      // the last one woken collapses a CU's waves into a single wake.
      if (cu != woken) {
        cu->schedule_work_async();
        woken = cu;
      }
    }
    queue_ids[queue.request_index] |= kQueueError;
  }
  return static_cast<int>(suspended);
}

void SimulatedKfd::clear_completed_debug_queues(KfdProcess *proc, const uint32_t *queue_ids,
                                                uint32_t num_queues) {
  if (!proc)
    return;
  constexpr uint32_t kQueueStatus =
      (uint32_t{1} << KFD_DBG_QUEUE_ERROR_BIT) | (uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT);
  const uint32_t process_id = proc->process_id();
  std::vector<std::pair<uint32_t, KfdProcess::QueueSnapshotInfo>> queues;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (uint32_t index = 0; index < num_queues; ++index) {
      if ((queue_ids[index] & kQueueStatus) != 0)
        continue;
      const uint32_t queue_id = queue_ids[index];
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue != proc->queue_snapshot_map_.end())
        queues.emplace_back(queue_id, queue->second);
    }
  }
  for (const auto &[queue_id, queue] : queues) {
    auto *gpu = find_gpu(queue.gpu_id);
    if (!gpu || !gpu->soc || queue.ctx_save_restore_address == 0)
      continue;
    bool has_stopped_wave = false;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units()) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            const auto *wave = cu->wf(slot);
            if (wave && wave->debug_stopped() && wave->process_id() == process_id &&
                wave->queue_id() == queue_id)
              has_stopped_wave = true;
          }
        });
      }
    });
    if (!has_stopped_wave) {
      const std::optional<amdgpu::GpuVmAccess> vm_access =
          gpu->soc->gpu_vm().snapshot_vmid(process_id);
      if (!vm_access)
        continue;
      const uint32_t area_count =
          gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5 ? std::max(1u, gpu->soc->num_xcds()) : 1u;
      for (uint32_t area = 0; area < area_count; ++area) {
        const uint64_t area_base = queue.ctx_save_restore_address +
                                   static_cast<uint64_t>(area) * queue.ctx_save_restore_area_size;
        for (uint64_t offset = 0; offset < 40; offset += sizeof(uint32_t)) {
          const uint32_t zero = 0;
          if (vm_access->write(area_base + offset, std::as_bytes(std::span(&zero, 1))) !=
              amdgpu::VmAccessOutcome::Complete) {
            break;
          }
        }
      }
    }
  }
}

int SimulatedKfd::debug_query_event(pid_t target_pid, KfdProcess *target_proc,
                                    KfdProcess::DebugSession &session,
                                    kfd_ioctl_dbg_trap_query_debug_event_args &args) {
  const uint64_t clear_mask = args.exception_mask;
  if (target_proc != nullptr) {
    std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
    for (uint32_t queue_id : target_proc->active_queue_ids_) {
      auto queue = target_proc->queue_snapshot_map_.find(queue_id);
      if (queue == target_proc->queue_snapshot_map_.end())
        continue;
      const uint64_t visible_status = queue->second.debugger_visible_exception_status();
      // Runtime ownership needs a replacement wake only if a debugger
      // notification was already sent. Otherwise a failed ROCr delivery gets
      // its first wake through the normal notification retry.
      const uint64_t pending_status = queue->second.exception_status &
                                      (queue->second.debug_notification_pending_status |
                                       (queue->second.debug_notification_delivered_status &
                                        queue->second.runtime_exception_pending_status)) &
                                      ~queue->second.runtime_exception_queried_status;
      if ((pending_status & session.exception_enable_mask) != 0)
        session.notification_query_retry_needed = true;
      if ((visible_status & session.exception_enable_mask) == 0)
        continue;
      args.exception_mask = visible_status;
      args.queue_id = queue_id;
      args.gpu_id = queue->second.gpu_id;
      queue->second.clear_debugger_exception_status(clear_mask);
      return 0;
    }
  }

  std::lock_guard<std::mutex> lk(debug_process_events_mutex_);
  auto process = debug_process_events_.find(target_pid);
  if (process == debug_process_events_.end())
    return -EAGAIN;
  auto &queues = process->second;
  for (auto queue = queues.begin(); queue != queues.end(); ++queue) {
    if ((queue->second.mask & session.exception_enable_mask) == 0)
      continue;
    args.exception_mask = queue->second.mask;
    session.consume_process_notification(clear_mask & args.exception_mask);
    args.queue_id = queue->first;
    args.gpu_id = queue->second.gpu_id;
    queue->second.mask &= ~clear_mask;
    for (uint64_t &event : queue->second.events)
      event &= ~clear_mask;
    std::erase(queue->second.events, uint64_t{0});
    if (queue->second.mask == 0)
      queues.erase(queue);
    return 0;
  }
  return -EAGAIN;
}

void SimulatedKfd::raise_process_debug_event(pid_t target_pid, uint64_t exception_mask,
                                             bool invoke_result_hook) {
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return;
    std::lock_guard<std::mutex> event_lock(debug_process_events_mutex_);
    auto &event = debug_process_events_[target_pid][0];
    event.gpu_id = 0;
    event.mask |= exception_mask;
    if (std::find(event.events.begin(), event.events.end(), exception_mask) == event.events.end())
      event.events.push_back(exception_mask);
  }
  [[maybe_unused]] const int notification_result =
      retry_debug_notifications(target_pid, invoke_result_hook);
  debug_sessions_cv_.notify_one();
}

void SimulatedKfd::cancel_runtime_handshake(pid_t target_pid) {
  {
    std::lock_guard<std::mutex> lk(runtime_handshake_mutex_);
    runtime_handshake_cancelled_.insert(target_pid);
  }
  runtime_handshake_cv_.notify_all();
}

/// @details The inferior must not run a kernel until the debugger attached to
/// it has seen EC_PROCESS_RUNTIME and installed its breakpoints -- that is the
/// whole point of the handshake, and amdkfd's runtime_enable blocks the caller
/// for it. Returning early is not a graceful degradation: the code objects load
/// unobserved, the debugger's breakpoints are never inserted, and the kernel
/// runs to completion. Under a debugger driving many inferiors at once (see
/// gdb.rocm/multi-inferior-stress.exp, 32 of them) a short deadline expires
/// routinely while the debugger is simply busy servicing its queue, which
/// showed up as rare, load-dependent "inferior exited before hitting the
/// kernel" failures.
///
/// So wait for the ack itself, and treat the deadline purely as a liveness
/// backstop against a debugger that has wedged without detaching -- long enough
/// that a merely slow one is never cut off, and noisy when it does fire so the
/// timeout is never again mistaken for correct behaviour. A debugger that
/// detaches or dies cancels the wait explicitly and does not pay it at all.
///
/// The disable direction reports the transition and returns. kfd_chardev.c's
/// runtime_disable does wait there -- it raises EC_PROCESS_RUNTIME and blocks on
/// runtime_enable_sema, the same way runtime_enable does -- and this deliberately
/// does not follow it, because the two are not in the same position. The kernel
/// blocks a task; here the ioctl is served on the daemon's connection thread for
/// that client, and holding it parks the daemon's side of a process that is
/// already exiting. Doing it anyway destabilises the daemon-backed tests
/// (RemoteDriverDbg*) with no measured benefit: the -EPERM storm this was first
/// written for is answered where it is raised, by the queue gate letting a
/// suspend or resume of already-destroyed queues through.
void SimulatedKfd::runtime_debugger_handshake(pid_t target_pid, bool enabling) {
  // The caller already decided, under debug_sessions_mutex_, that a debugger is
  // attached; re-testing here would just re-open the window it closed. Only the
  // self-debug shape still needs distinguishing.
  bool self_debugged = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return;
    self_debugged = session->second.debugger_pid == target_pid;

    // Drop any ack or cancellation left over from an earlier transition for this
    // pid before the event goes out. The disable direction reports without
    // waiting, so nothing consumes the ack it draws; a later enable would
    // otherwise find that stale entry already in runtime_acked_, satisfy its wait
    // immediately, and let the inferior dispatch before the debugger has
    // installed its breakpoints -- the exact failure the deadline below exists to
    // prevent.
    //
    // A cancellation does the same thing and is easier to leave behind, because
    // cancel_runtime_handshake() inserts unconditionally: a detach with no waiter
    // parked -- which every explicit DBG_TRAP_DISABLE performs -- leaves an entry
    // nobody consumes, and the wait predicate below accepts it just as readily as
    // a real ack. Clearing both under the same lock a real cancel would take
    // means a cancel racing this clear necessarily lands on the new wait.
    //
    // That last part only holds while debug_sessions_mutex_ is still held, which
    // is why the clear lives here rather than after the lookup returns. An
    // explicit DBG_TRAP_DISABLE erases the session, drops the mutex, and only
    // then calls cancel_runtime_handshake(). Clearing after releasing the mutex
    // left a window in which the detach's cancellation was recorded after this
    // function had already decided the session was live, and was then erased by
    // this very clear -- so nothing remained to satisfy the wait below, no acker
    // was left to provide one, and the inferior paid the full 60s deadline.
    // Holding the mutex across both means a detach either erases the session
    // before the lookup (which returns early) or blocks until after the clear,
    // in which case its cancellation lands on the wait.
    if (enabling) {
      std::lock_guard<std::mutex> hlk(runtime_handshake_mutex_);
      runtime_acked_.erase(target_pid);
      runtime_handshake_cancelled_.erase(target_pid);
    }
  }
  raise_process_debug_event(target_pid, KFD_EC_MASK(EC_PROCESS_RUNTIME));
  // A process debugging itself cannot answer its own handshake: the ack would
  // have to come from the thread that is, by construction, blocked here. The
  // event is raised so the state is observable; waiting for a reply that can
  // never arrive would just burn the liveness deadline.
  if (self_debugged || !enabling)
    return;

  constexpr auto kHandshakeDeadline = std::chrono::seconds(60);
  std::unique_lock<std::mutex> lk(runtime_handshake_mutex_);
  const bool released = runtime_handshake_cv_.wait_for(lk, kHandshakeDeadline, [&] {
    return runtime_acked_.contains(target_pid) || runtime_handshake_cancelled_.contains(target_pid);
  });
  const bool cancelled = runtime_handshake_cancelled_.erase(target_pid) != 0;
  runtime_acked_.erase(target_pid);
  if (!released) {
    util::Logger::warn("runtime-enable handshake timed out after ", kHandshakeDeadline.count(),
                       "s for pid=", target_pid,
                       "; the debugger never acknowledged EC_PROCESS_RUNTIME, so its "
                       "breakpoints may not be installed before the first dispatch");
  } else if (cancelled) {
    util::Logger::vm("runtime-enable handshake cancelled for pid=", target_pid,
                     " (debugger detached)");
  }
}

int SimulatedKfd::debug_query_exception_info(pid_t target_pid,
                                             kfd_ioctl_dbg_trap_query_exception_info_args &args) {
  if (args.exception_code != EC_PROCESS_RUNTIME)
    return -EINVAL;
  kfd_runtime_info info{};
  if (auto proc = find_process_by_client_pid(target_pid)) {
    std::lock_guard<std::mutex> lk(proc->runtime_mutex_);
    info.r_debug = proc->runtime_state_.r_debug;
    info.runtime_state =
        proc->runtime_state_.enabled ? DEBUG_RUNTIME_STATE_ENABLED : DEBUG_RUNTIME_STATE_DISABLED;
    info.ttmp_setup =
        (proc->runtime_state_.mode_mask & KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK) ? 1u : 0u;
  }
  const uint32_t capacity = args.info_size;
  args.info_size = sizeof(info);
  if (capacity > 0 && args.info_ptr == 0)
    return -EFAULT;
  if (args.info_ptr != 0 && capacity > 0)
    std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(args.info_ptr)), &info,
                std::min(static_cast<size_t>(capacity), sizeof(info)));
  if (args.clear_exception) {
    std::lock_guard<std::mutex> lk(debug_process_events_mutex_);
    auto process = debug_process_events_.find(target_pid);
    if (process != debug_process_events_.end()) {
      auto event = process->second.find(0);
      if (event != process->second.end()) {
        constexpr uint64_t kRuntime = KFD_EC_MASK(EC_PROCESS_RUNTIME);
        auto session = debug_sessions_.find(target_pid);
        if (session != debug_sessions_.end())
          session->second.consume_process_notification(event->second.mask & kRuntime);
        event->second.mask &= ~kRuntime;
        for (uint64_t &mask : event->second.events)
          mask &= ~kRuntime;
        std::erase(event->second.events, uint64_t{0});
        if (event->second.mask == 0)
          process->second.erase(event);
      }
    }
  }
  return 0;
}

namespace {

/// @brief Whether a suspend/resume names only queues the process has destroyed.
/// @details Used to tell a request that cannot touch hardware from one that
/// can, so the runtime-down gate only refuses the latter. Any other op, an
/// absent process, or an empty request answers false, leaving the kernel's
/// behaviour in place.
bool queues_all_dead(KfdProcess *proc, const kfd_ioctl_dbg_trap_args &args) {
  constexpr uint32_t kQueueStatus = KFD_DBG_QUEUE_ERROR_MASK | KFD_DBG_QUEUE_INVALID_MASK;
  if (proc == nullptr)
    return false;
  uint32_t count = 0;
  uint64_t array_ptr = 0;
  if (args.op == KFD_IOC_DBG_TRAP_SUSPEND_QUEUES) {
    count = args.suspend_queues.num_queues;
    array_ptr = args.suspend_queues.queue_array_ptr;
  } else if (args.op == KFD_IOC_DBG_TRAP_RESUME_QUEUES) {
    count = args.resume_queues.num_queues;
    array_ptr = args.resume_queues.queue_array_ptr;
  } else {
    return false;
  }
  // An empty batch names nothing live, so it belongs on the answered side: with
  // the runtime up the handlers return 0 for it, and refusing it here would flip
  // that to the -EPERM rocdbgapi escalates to a fatal purely on runtime state.
  if (count == 0)
    return true;
  // A null array with a non-zero count is malformed; leave it to the handler's
  // -EFAULT rather than inventing an answer for it.
  if (array_ptr == 0)
    return false;
  const auto *queue_ids = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(array_ptr));
  std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
  for (uint32_t index = 0; index < count; ++index)
    if (proc->queue_snapshot_map_.contains(queue_ids[index] & ~kQueueStatus))
      return false;
  return true;
}

} // namespace

// in real kernel, amd/amdkfd/kfd_chardev.c kfd_ioctl_set_debug_trap
int SimulatedKfd::debug_trap_ioctl(KfdProcess &caller, void *arg, int *target_mem_fd,
                                   int target_proc_fd) {
  auto *args = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
  util::Logger::driver("DBG_TRAP pid=", args->pid, " op=", args->op);
  // rocjitsu always models hardware scheduling, so the driver's
  // KFD_SCHED_POLICY_NO_HWS -> EINVAL guard is not applicable.

  // Resolve the target process by Linux pid (kernel: find_get_pid() +
  // kfd_lookup_process_by_pid()). The target's KfdProcess may not exist yet:
  // rocgdb enables debug on the inferior right after exec, before its ROCr has
  // opened /dev/kfd. The debug session is keyed by the target pid
  // (debug_sessions_) independently of the KfdProcess, mirroring the kernel
  // creating the target kfd_process in the ENABLE path. Operations that need
  // live GPU state look the process up lazily.
  const auto target_pid = static_cast<pid_t>(args->pid);
  if (target_pid <= 0)
    return -ESRCH;

  const bool self_debug = caller.client_pid() != 0 && target_pid == caller.client_pid();

  std::unique_lock<std::mutex> lk(debug_sessions_mutex_);
  auto session_it = debug_sessions_.find(target_pid);
  if (session_it != debug_sessions_.end()) {
    if (session_it->second.target_exited) {
      if (args->op != KFD_IOC_DBG_TRAP_ENABLE)
        return -ESRCH;
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
    }
  }
  if (session_it != debug_sessions_.end()) {
    const int target_exited = pidfd_is_exited(session_it->second.target_pidfd.get());
    if (target_exited < 0)
      return target_exited;
    const int debugger_exited = pidfd_is_exited(session_it->second.debugger_pidfd.get());
    if (debugger_exited < 0)
      return debugger_exited;
    if (target_exited == 1) {
      session_it->second.owned_dbg_fd.reset();
      session_it->second.target_mem_fd.reset();
      session_it->second.dbg_fd = -1;
      session_it->second.enabled = false;
      session_it->second.target_exited = true;
      if (args->op != KFD_IOC_DBG_TRAP_ENABLE)
        return -ESRCH;
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
    } else if (debugger_exited == 1) {
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
    }
  }
  const bool enabled = session_it != debug_sessions_.end();

  // Non-ENABLE ops require an active debug session (kernel: EINVAL).
  if (args->op != KFD_IOC_DBG_TRAP_ENABLE && !enabled) {
    UniqueDriverFd probe_pidfd;
    UniqueDriverFd probe_procfd;
    const int probe_result = pin_process_identity(target_pid, probe_pidfd, probe_procfd);
    if (probe_result == -ESRCH)
      return -ESRCH;
    if (probe_result != 0)
      return probe_result;
    const int zombie = procfd_is_zombie(probe_procfd.get());
    if (zombie != 0)
      return zombie == 1 ? -ESRCH : zombie;
    return -EINVAL;
  }

  // Pin the exact Linux task before consulting ptrace state. For an existing
  // session, use the identity captured by ENABLE; otherwise capture both a
  // pidfd and the matching procfs directory now. The pidfd liveness checks on
  // both sides of the procfs read ensure a numeric-pid reuse can never authorize
  // a different task.
  UniqueDriverFd new_target_pidfd;
  UniqueDriverFd new_target_procfd;
  UniqueDriverFd *target_pidfd;
  UniqueDriverFd *target_procfd;
  if (enabled) {
    target_pidfd = &session_it->second.target_pidfd;
    target_procfd = &session_it->second.target_procfd;
  } else {
    const int pin_result = pin_process_identity(target_pid, new_target_pidfd, new_target_procfd);
    if (pin_result != 0)
      return pin_result;
    target_pidfd = &new_target_pidfd;
    target_procfd = &new_target_procfd;
  }

  UniqueDriverFd new_debugger_pidfd;
  UniqueDriverFd new_debugger_procfd;
  if (args->op == KFD_IOC_DBG_TRAP_ENABLE) {
    const int debugger_pin_result =
        pin_process_identity(caller.client_pid(), new_debugger_pidfd, new_debugger_procfd);
    if (debugger_pin_result != 0)
      return debugger_pin_result;
  }

  // PTRACE gate: for any op other than DISABLE, a debugger acting on another
  // process must be that exact process's ptrace parent. This mirrors
  // ptrace_parent(target->lead_thread) == current in kfd_ioctl_set_debug_trap().
  if (!self_debug && args->op != KFD_IOC_DBG_TRAP_DISABLE) {
    pid_t tracer_pid = 0;
    const int tracer_result =
        tracer_pid_of(*target_pidfd, *target_procfd, debug_identity_validation_hook_, tracer_pid);
    if (tracer_result != 0) {
      if (tracer_result == -ESRCH && enabled)
        debug_sessions_.erase(target_pid);
      return tracer_result;
    }
    if (tracer_pid != caller.client_pid())
      return -EPERM;
  }

  // Resolve live GPU state only after pinning and authorizing the OS identity,
  // so a KfdProcess associated with a reused numeric pid is never selected.
  std::shared_ptr<KfdProcess> target_ref =
      self_debug ? nullptr : find_process_by_client_pid(target_pid);
  KfdProcess *target_proc = self_debug ? &caller : target_ref.get();
  if (target_proc != nullptr && session_it != debug_sessions_.end())
    session_it->second.saw_kfd_process = true;

  // Live runtime-enable state, set by ROCr's AMDKFD_IOC_RUNTIME_ENABLE on the
  // inferior; false until the inferior connects and enables its runtime.
  bool runtime_enabled = false;
  if (target_proc != nullptr) {
    std::lock_guard<std::mutex> rlk(target_proc->runtime_mutex_);
    runtime_enabled = target_proc->runtime_state_.enabled;
  }

  // The target may exit after authorization. Revalidate before performing or
  // committing an operation so a reused numeric pid cannot contribute live
  // KfdProcess state to the pinned session.
  const int still_live = pidfd_is_exited(target_pidfd->get());
  if (still_live != 0) {
    if (still_live == 1 && enabled)
      debug_sessions_.erase(target_pid);
    return still_live == 1 ? -ESRCH : still_live;
  }
  if (args->op == KFD_IOC_DBG_TRAP_ENABLE) {
    const int debugger_still_live = pidfd_is_exited(new_debugger_pidfd.get());
    if (debugger_still_live != 0)
      return debugger_still_live == 1 ? -ESRCH : debugger_still_live;
  }

  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L3132-L3142
  switch (args->op) {
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE:
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE:
  case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
  case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
  case KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH:
  case KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH:
  case KFD_IOC_DBG_TRAP_SET_FLAGS:
    if (!runtime_enabled) {
      // A target that had a KfdProcess and no longer has one is a process on
      // its way out, not a live one refusing the op. The real driver cannot
      // reach this gate in that state at all: its pid lookup fails first and
      // returns -ESRCH, which rocdbgapi handles (PROCESS_EXITED -> invalidate
      // the queues and move on). -EPERM is what it escalates to a fatal
      // os_driver::resume_queues failure, so getting this distinction wrong
      // crashes GDB during teardown (gdb.rocm/multi-inferior-stress.exp).
      if (target_proc == nullptr && session_it != debug_sessions_.end() &&
          session_it->second.saw_kfd_process)
        return -ESRCH;
      // A suspend or resume naming only queues the process has already
      // destroyed asks nothing of the hardware, and answering it is the only
      // way rocdbgapi learns to drop them: the per-queue INVALID bit the normal
      // path writes back. It cannot learn it any other way once the runtime is
      // down, because its queue-list sweep is gated on the runtime being up
      // (process.cpp update_queues), and it escalates the -EPERM to a fatal
      // rather than retiring the queue.
      //
      // Upstream refuses this unconditionally, and never has to answer it: its
      // teardown does not leave a debugger holding a suspended queue across
      // runtime shutdown, so the request does not arise. Ours does, and the
      // narrow shape -- no runtime, and not one live queue among those named --
      // is exactly the one where refusing costs information and buys nothing.
      // Anything still live keeps the kernel's answer.
      if (!queues_all_dead(target_proc, *args))
        return -EPERM;
      util::Logger::vm("DBG_TRAP op=", args->op, " for pid=", target_pid,
                       " names only destroyed queues and the runtime is down; reporting them "
                       "invalid instead of -EPERM");
    }
    break;
  default:
    break;
  }

  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L3144
  if (args->op == KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH ||
      args->op == KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH) {
    const uint32_t gpu_id = args->op == KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH
                                ? args->set_node_address_watch.gpu_id
                                : args->clear_node_address_watch.gpu_id;
    if (find_gpu(gpu_id) == nullptr)
      return -ENODEV;
  }

  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L3158
  switch (args->op) {
  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_debug.c#L788-L847
  case KFD_IOC_DBG_TRAP_ENABLE: {
    if (enabled)
      return -EALREADY; // target process is already debug enabled

    if (target_proc) {
      std::lock_guard allocation_lock(target_proc->alloc_mutex_);
      for (const auto &[_, queue] : target_proc->queue_snapshot_map_)
        if (queue.gws_allocated && !debugger_supports_gws(queue.gpu_id))
          return -EBUSY;
    }

    const int dbg_fd = static_cast<int>(args->enable.dbg_fd);
    // Validate the notifier before trusting it. In daemon mode the fd was
    // received via SCM_RIGHTS and already substituted into our fd space; in
    // local mode it is the debugger's own descriptor. Either way the driver
    // *writes* to it to wake the debugger, so it must be a live, writable
    // descriptor. safe_fcntl(F_GETFL) both proves the fd is open (EBADF otherwise)
    // and reports its access mode, so a read-only or otherwise unusable fd —
    // e.g. one a client passed over SCM_RIGHTS that is not a real event target —
    // is rejected instead of being stored on the session.
    const int fl = safe_fcntl(dbg_fd, F_GETFL);
    if (fl == -1 || (fl & O_ACCMODE) == O_RDONLY)
      return -EBADF;
    if (daemon_mode_) {
      if (target_mem_fd == nullptr || *target_mem_fd < 0 || target_proc_fd < 0)
        return -EBADF;
      const int mem_fl = safe_fcntl(*target_mem_fd, F_GETFL);
      if (mem_fl == -1 || (mem_fl & O_ACCMODE) != O_RDWR)
        return -EBADF;
      struct stat daemon_proc_stat {};
      struct stat client_proc_stat {};
      if (fstat(target_procfd->get(), &daemon_proc_stat) != 0 ||
          fstat(target_proc_fd, &client_proc_stat) != 0)
        return -errno;
      if (daemon_proc_stat.st_dev != client_proc_stat.st_dev ||
          daemon_proc_stat.st_ino != client_proc_stat.st_ino)
        return -ESRCH;
      // The proc directory is pinned to the right process by the check above,
      // but nothing yet ties the transferred mem fd to it -- O_RDWR only says
      // the fd is writable, not whose memory it addresses. This fd becomes
      // authoritative for guest reads and CWSR writes, so a mismatched one
      // silently redirects both at another process. procfs gives each
      // /proc/<pid>/mem its own inode, so comparing it against the pinned
      // directory's own "mem" entry settles the question.
      struct stat expected_mem_stat {};
      struct stat client_mem_stat {};
      if (fstatat(target_procfd->get(), "mem", &expected_mem_stat, 0) != 0 ||
          fstat(*target_mem_fd, &client_mem_stat) != 0)
        return -errno;
      if (expected_mem_stat.st_dev != client_mem_stat.st_dev ||
          expected_mem_stat.st_ino != client_mem_stat.st_ino)
        return -ESRCH;
    }

    KfdProcess::DebugSession sess{};
    sess.generation = next_debug_session_generation_++;
    sess.target_pidfd = std::move(new_target_pidfd);
    sess.target_procfd = std::move(new_target_procfd);
    sess.enabled = true;
    sess.debugger_pid = caller.client_pid();
    sess.debugger_pidfd = std::move(new_debugger_pidfd);
    sess.dbg_fd = dbg_fd;
    sess.exception_enable_mask = args->enable.exception_mask;

    // Snapshot the runtime-enable state under a single lock so the marshaled
    // runtime_state, r_debug and ttmp_setup stay mutually consistent: a
    // concurrent RUNTIME_ENABLE/DISABLE must not change them between reads.
    // Lock order debug_sessions_mutex_ -> runtime_mutex_ is already held that
    // way.
    // Kernel: kfd_dbg_trap_enable copies the saved runtime info and returns its
    // size.
    kfd_runtime_info info{};
    if (target_proc != nullptr) {
      // The session does not exist yet when the common path above records this,
      // so ENABLE has to do it itself; otherwise an inferior that opens
      // /dev/kfd, runs and exits before the debugger's first suspend/resume
      // never sets the flag and the gate below answers -EPERM after all.
      sess.saw_kfd_process = true;
      std::lock_guard<std::mutex> rlk(target_proc->runtime_mutex_);
      const auto &rt = target_proc->runtime_state_;
      sess.runtime_state = rt.enabled ? DEBUG_RUNTIME_STATE_ENABLED : DEBUG_RUNTIME_STATE_DISABLED;
      info.r_debug = rt.r_debug;
      info.ttmp_setup = (rt.mode_mask & KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK) ? 1u : 0u;
    }
    info.runtime_state = sess.runtime_state;
    size_t copy_size = std::min(static_cast<size_t>(args->enable.rinfo_size), sizeof(info));
    if (args->enable.rinfo_ptr != 0 && copy_size > 0)
      std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(args->enable.rinfo_ptr)), &info,
                  copy_size);
    args->enable.rinfo_size = sizeof(info);

    const int target_commit_live = pidfd_is_exited(sess.target_pidfd.get());
    if (target_commit_live != 0)
      return target_commit_live == 1 ? -ESRCH : target_commit_live;
    const int debugger_commit_live = pidfd_is_exited(sess.debugger_pidfd.get());
    if (debugger_commit_live != 0)
      return debugger_commit_live == 1 ? -ESRCH : debugger_commit_live;

    // Both transferred fds are adopted here, past every failure check, because
    // each only has to survive a successful ENABLE: on any earlier return the
    // caller still owns it and reclaims it. Adopting the notifier sooner would
    // double-close it, because the transport clears cmd->in_handle only when
    // the ioctl succeeds -- a late liveness failure would destroy sess (closing
    // the fd) and then let the transport close the same number again. With
    // daemon requests running concurrently another thread can be handed that
    // number in between, so the second close would land on an unrelated
    // descriptor. In local mode dbg_fd is the debugger's own descriptor and
    // nothing is owned here.
    if (daemon_mode_) {
      sess.owned_dbg_fd = UniqueDriverFd(dbg_fd);
      sess.target_mem_fd = UniqueDriverFd(*target_mem_fd);
      *target_mem_fd = -1;
    }
    auto [inserted, _] = debug_sessions_.emplace(target_pid, std::move(sess));
    if (target_proc != nullptr && inserted->second.target_mem_fd) {
      for (uint32_t ordinal = 0; ordinal < gpus_.size(); ++ordinal) {
        GpuDevice &gpu = gpus_[ordinal];
        if (gpu.soc == nullptr)
          continue;
        const amdgpu::AddressSpaceHandle address_space = target_proc->gpu(ordinal).address_space;
        if (address_space)
          (void)gpu.legacy_vm->set_client_mem_fd(address_space,
                                                 inserted->second.target_mem_fd.get());
      }
    }
    set_debug_active_on_all_cus(true);
    debug_sessions_cv_.notify_one();
    lk.unlock();
    [[maybe_unused]] const int notification_result =
        retry_debug_notifications(target_pid, /*invoke_result_hook=*/true);
    return 0;
  }
  case KFD_IOC_DBG_TRAP_DISABLE: {
    // Erasing the session releases the debugger notifier: in daemon mode the
    // session's UniqueHandle closes the SCM_RIGHTS-transferred fd it owns; in
    // local mode nothing is owned, so the debugger's own fd is left untouched.
    //
    // The queued debug events and the per-queue exception status go with it.
    // Both are debugger-visible state that only has meaning inside a session:
    // leaving them behind would hand a stale EC_QUEUE_NEW, or an exception the
    // previous debugger already consumed, to whoever attaches next. Any wave
    // the departing debugger left stopped is resumed below, so detaching never
    // strands the inferior's GPU work.
    //
    // Erase the session first, then release the inferior outside the lock.
    // Dropping debug_sessions_mutex_ before touching any CU is mandatory: the
    // engine thread takes these two the other way round -- ComputeUnitCore::step()
    // runs the issue loop under the CU's wave-state lock and calls back into
    // on_wave_trap_complete()/on_wave_watchpoint()/on_wave_illegal_inst(), each
    // of which acquires debug_sessions_mutex_ -- so holding it across
    // with_wave_state_locked() closes an AB-BA cycle and hangs a detach against
    // a wave that is trapping at that moment. Erasing before the release also
    // closes the window in which the session was still enabled while its events
    // had already been cleared, letting a live callback publish into a session
    // on its way out. This is the same invariant SUSPEND_QUEUES observes.
    debug_sessions_.erase(target_pid);
    lk.unlock();
    // target_proc, not a fresh find_process_by_client_pid(): it was resolved
    // above while the pidfd/procfd pin still vouched for the identity, so a
    // numeric pid reused since then cannot be selected here. It is also the
    // only resolution that is correct for a self-debugging process, where the
    // debuggee is `caller` and need not be reachable by client pid at all --
    // looking it up again would return nullptr and silently skip the whole
    // release, stranding the very waves this path exists to free.
    release_debuggee_state(target_pid, target_proc);
    // An explicit detach also has to release a waiter, or the inferior pays the
    // full liveness deadline for a debugger that is deliberately going away.
    cancel_runtime_handshake(target_pid);
    // Deliberately left unlocked: nothing below touches debug_sessions_, and
    // release_debuggee_state() already re-took and released the mutex for the
    // one guarded question it has to ask.
    return 0;
  }
  case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
    return debug_device_snapshot(args->device_snapshot);
  case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
    return debug_queue_snapshot(target_proc, args->queue_snapshot);
  case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
    return debug_query_exception_info(target_pid, args->query_exception_info);
  case KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT: {
    const auto &event = args->send_runtime_event;
    if (event.exception_mask != 0 &&
        (event.exception_mask & KFD_EC_MASK(EC_PROCESS_RUNTIME)) == 0) {
      // Forwarding a queue exception needs the target's live GPU state, which
      // only exists once it has opened /dev/kfd. A debugger may be attached
      // before that, or still attached after the target closed the device, so
      // this op has to guard the lookup the way its siblings above do.
      if (target_proc == nullptr)
        return -ESRCH;
      auto *gpu = find_gpu(event.gpu_id);
      if (!gpu || !gpu->soc)
        return -ENODEV;
      // Drop debug_sessions_mutex_ before entering the CU wave-state locks:
      // trap callbacks acquire these in the opposite order. Nothing below
      // this point reads debug_sessions_ or session_it.
      lk.unlock();
      // KFD's SEND_RUNTIME_EVENT publishes the status and event without
      // waiting for ROCr to acknowledge them. ROCr removes its async exception
      // handler after a fatal error, so later waves forwarding their aborts
      // cannot satisfy such a wait. A debugger can also have the host stopped.
      const bool published = signal_runtime_queue_exception(
          event.gpu_id, event.queue_id, target_proc->process_id(), event.exception_mask,
          /*wait_for_ack=*/false);
      if (published)
        return 0;
      // The runtime may already have destroyed this queue after the first
      // forwarded abort. KFD accepts that stale request too. Keep reporting
      // actual publication failures for queues that still exist.
      std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
      return target_proc->queue_snapshot_map_.contains(event.queue_id) ? -ENOENT : 0;
    }
    std::lock_guard<std::mutex> runtime_lock(runtime_handshake_mutex_);
    runtime_acked_.insert(target_pid);
    runtime_handshake_cv_.notify_all();
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED: {
    const uint64_t enabled_mask = args->set_exceptions_enabled.exception_mask;
    const uint64_t disabled_mask = session_it->second.exception_enable_mask & ~enabled_mask;
    if (target_proc != nullptr && disabled_mask != 0) {
      std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
      for (auto &[_, queue] : target_proc->queue_snapshot_map_) {
        queue.reset_debug_notification_delivery(disabled_mask, session_it->second.generation);
      }
    }
    session_it->second.exception_enable_mask = enabled_mask;
    update_notification_claims(session_it->second, enabled_mask);
    session_it->second.notified_process_exception_mask &= enabled_mask;
    lk.unlock();
    const int result = retry_debug_notifications(target_pid, /*invoke_result_hook=*/true);
    debug_sessions_cv_.notify_one();
    return result;
  }
  case KFD_IOC_DBG_TRAP_SET_FLAGS: {
    const uint32_t previous = session_it->second.flags;
    session_it->second.flags = args->set_flags.flags;
    args->set_flags.flags = previous;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE:
    if (args->launch_mode.launch_mode != KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL &&
        args->launch_mode.launch_mode != KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT &&
        args->launch_mode.launch_mode != KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG)
      return -EINVAL;
    session_it->second.launch_mode = args->launch_mode.launch_mode;
    return 0;
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE: {
    if (args->launch_override.override_mode != KFD_DBG_TRAP_OVERRIDE_OR)
      return -EINVAL;
    constexpr uint32_t kGfx94SupportedTrapMask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
    if ((args->launch_override.support_request_mask & ~kGfx94SupportedTrapMask) != 0)
      return -EACCES;
    const uint32_t previous = session_it->second.launch_override_enable;
    session_it->second.launch_override_enable = args->launch_override.enable_mask;
    args->launch_override.enable_mask = previous;
    args->launch_override.support_request_mask = kGfx94SupportedTrapMask;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH: {
    auto &watches = session_it->second.address_watches;
    uint32_t slot = 0;
    while (slot < KfdProcess::DebugSession::kMaxAddressWatches && watches[slot].active)
      ++slot;
    if (slot == KfdProcess::DebugSession::kMaxAddressWatches)
      return -ENOMEM;
    watches[slot] = KfdProcess::DebugSession::AddressWatch::from_kfd(
        args->set_node_address_watch.address, args->set_node_address_watch.mask,
        args->set_node_address_watch.mode);
    args->set_node_address_watch.id = slot;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH: {
    const uint32_t slot = args->clear_node_address_watch.id;
    if (slot >= KfdProcess::DebugSession::kMaxAddressWatches)
      return -EINVAL;
    session_it->second.address_watches[slot] = {};
    return 0;
  }
  case KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT: {
    return debug_query_event(target_pid, target_proc, session_it->second, args->query_debug_event);
  }
  case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES: {
    if (args->suspend_queues.num_queues != 0 && args->suspend_queues.queue_array_ptr == 0)
      return -EFAULT;
    auto *queue_ids =
        reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args->suspend_queues.queue_array_ptr));
    // Engine callbacks enter with a CU's wave-state lock and briefly acquire
    // debug_sessions_mutex_. Do not invert that order while freezing CUs.
    lk.unlock();
    const int result = suspend_debug_queues(target_proc, queue_ids, args->suspend_queues.num_queues,
                                            args->suspend_queues.exception_mask);
    clear_completed_debug_queues(target_proc, queue_ids, args->suspend_queues.num_queues);
    return result;
  }
  case KFD_IOC_DBG_TRAP_RESUME_QUEUES: {
    if (args->resume_queues.num_queues != 0 && args->resume_queues.queue_array_ptr == 0)
      return -EFAULT;
    auto *queue_ids =
        reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args->resume_queues.queue_array_ptr));
    lk.unlock();
    return resume_debug_queues(target_proc, queue_ids, args->resume_queues.num_queues);
  }
  default:
    return -EINVAL;
  }
}

int SimulatedKfd::debug_device_snapshot(kfd_ioctl_dbg_trap_device_snapshot_args &args) {
  // Mirrors kfd_dbg_trap_device_snapshot() (amd/amdkfd/kfd_debug.c): report the
  // total device count, clamp the per-entry size, and fill up to the caller's
  // buffer capacity. The two-call protocol is to call once with a small buffer
  // and read the true total back, then call again sized for it -- rocdbgapi's
  // kfd_snapshots::fetch() probes with a one-entry buffer, not a null pointer.
  //
  // The buffer is validated first, before any output is written: the driver
  // rejects a malformed request outright (-EINVAL) rather than half-answering
  // it, so a caller cannot read a device total off a call that failed this way.
  if (args.snapshot_buf_ptr == 0)
    return -EINVAL;

  // Only devices we can actually describe are enumerable. gpu_infos_ is filled
  // by setup_topology, which every embedder is free to skip or to call with
  // fewer devices than gpus_ holds (a config whose device block is absent, or
  // the single-GpuInfo overload on a multi-SoC driver). Reporting gpus_.size()
  // regardless would hand rocdbgapi entries with simd_count/array_count zero,
  // which its agent_snapshot treats as a fatal error rather than a bad ioctl.
  const uint32_t total = static_cast<uint32_t>(std::min(gpus_.size(), gpu_infos_.size()));
  const uint32_t in_entry_size = args.entry_size;
  const uint32_t fill = std::min<uint32_t>(args.num_devices, total);

  args.num_devices = total;
  args.entry_size = std::min<uint32_t>(in_entry_size, sizeof(kfd_dbg_device_info_entry));

  if (fill == 0)
    return 0;

  // A zero stride is not an error: the driver's per-entry copy_to_user() moves
  // entry_size(OUT) == 0 bytes and succeeds, so the call reports the device
  // total and writes nothing. Falling through reproduces that exactly.
  auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.snapshot_buf_ptr));
  for (uint32_t i = 0; i < fill; ++i) {
    const Sysfs::GpuInfo &info = gpu_infos_[i];
    const kfd_process_device_apertures ap = gpu_apertures(i);

    kfd_dbg_device_info_entry entry{};
    entry.gpu_id = gpus_[i].gpu_id;
    entry.lds_base = ap.lds_base;
    entry.lds_limit = ap.lds_limit;
    entry.scratch_base = ap.scratch_base;
    entry.scratch_limit = ap.scratch_limit;
    entry.gpuvm_base = ap.gpuvm_base;
    entry.gpuvm_limit = ap.gpuvm_limit;
    entry.location_id = info.location_id;
    entry.vendor_id = info.vendor_id;
    entry.device_id = info.device_id;
    entry.revision_id = info.pci_revision_id;
    entry.subsystem_vendor_id = info.vendor_id;
    entry.subsystem_device_id = info.device_id;
    entry.fw_version = info.fw_version;
    entry.gfx_target_version = info.gfx_target_version;
    entry.simd_count = info.simd_count;
    entry.max_waves_per_simd = info.max_waves_per_simd;
    // KFD array_count is the per-XCC shader-array count (node_props.array_count).
    // Unlike sysfs, kfd_debug.c passes it through unscaled and reports num_xcc
    // alongside, so rocdbgapi recovers the SoC's total shader-engine count as
    // array_count * num_xcc / simd_arrays_per_engine. Normalize the XCC count
    // the same way sysfs does so that quotient cannot come out zero.
    entry.array_count = info.array_count_per_xcc();
    entry.simd_arrays_per_engine = info.effective_arrays_per_engine();
    entry.num_xcc = info.effective_num_xcc();
    const kmd::DebugTopology topology =
        kmd::effective_topology_for(info.gfx_target_version, info.capability, info.capability2,
                                    info.debug_prop, info.revision_id);
    entry.capability = topology.capability;
    // debug_prop is __u32 in the snapshot entry but __u64 in the sysfs node
    // property, so a config that captured a debug_prop above 2^32 would have
    // the two paths report different values. The derived bits all fit; make the
    // narrowing the uapi struct imposes explicit rather than incidental.
    entry.debug_prop = static_cast<uint32_t>(topology.debug_prop);

    std::memcpy(out + static_cast<uint64_t>(i) * in_entry_size, &entry, args.entry_size);
  }
  return 0;
}

int SimulatedKfd::debug_queue_snapshot(KfdProcess *target,
                                       kfd_ioctl_dbg_trap_queue_snapshot_args &args) {
  // Mirrors pqm_get_queue_snapshot(): report the total queue count, fill only
  // the caller's capacity, and use the input entry size as the output stride.
  const uint32_t in_num = args.num_queues;
  const uint32_t in_entry_size = args.entry_size;

  args.num_queues = 0;
  if (in_entry_size == 0)
    return -EINVAL;
  args.entry_size = std::min<uint32_t>(in_entry_size, sizeof(kfd_queue_snapshot_entry));

  std::vector<kfd_queue_snapshot_entry> entries;
  if (target != nullptr) {
    std::lock_guard<std::mutex> lk(target->alloc_mutex_);
    entries.reserve(std::min<size_t>(in_num, target->active_queue_ids_.size()));
    for (uint32_t qid : target->active_queue_ids_) {
      auto it = target->queue_snapshot_map_.find(qid);
      if (it == target->queue_snapshot_map_.end())
        continue;
      if (args.num_queues < in_num) {
        KfdProcess::QueueSnapshotInfo &q = it->second;
        const uint64_t visible_status = q.debugger_visible_exception_status();
        entries.push_back({
            .exception_status = visible_status,
            .ring_base_address = q.ring_base_address,
            .write_pointer_address = q.write_pointer_address,
            .read_pointer_address = q.read_pointer_address,
            .ctx_save_restore_address = q.ctx_save_restore_address,
            .queue_id = qid,
            .gpu_id = q.gpu_id,
            .ring_size = q.ring_size,
            .queue_type = q.queue_type,
            .ctx_save_restore_area_size = q.ctx_save_restore_area_size,
            .reserved = 0,
        });
        q.clear_debugger_exception_status(args.exception_mask);
      }
      ++args.num_queues;
    }
  }

  if (entries.empty())
    return 0;
  if (args.snapshot_buf_ptr == 0)
    return -EFAULT;

  auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.snapshot_buf_ptr));
  for (size_t i = 0; i < entries.size(); ++i)
    std::memcpy(out + static_cast<uint64_t>(i) * in_entry_size, &entries[i], args.entry_size);
  return 0;
}

int SimulatedKfd::set_xnack_mode_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_xnack_mode_args *>(arg);
  args->xnack_enabled = 0;
  return 0;
}

bool SimulatedKfd::owns_fd(int fd) const {
  if (fd < 0)
    return false;
  std::lock_guard<std::mutex> lock(owned_fds_mutex_);
  return owned_fds_.contains(fd);
}

void SimulatedKfd::init_reserved_fd_range() {
  struct rlimit rl {};
  getrlimit(RLIMIT_NOFILE, &rl);
  reserved_fd_base_ = static_cast<int>(rl.rlim_cur) - kReservedFdCount;
  next_reserved_fd_ = reserved_fd_base_;
}

int SimulatedKfd::claim_fd(int real_fd) {
  if (reserved_fd_base_ == 0)
    init_reserved_fd_range();
  int vfd = next_reserved_fd_++;
  assert(vfd < reserved_fd_base_ + kReservedFdCount && "reserved fd range exhausted");
  libc_passthrough().dup2(real_fd, vfd);
  libc_passthrough().close(real_fd);
  return vfd;
}

bool SimulatedKfd::owns_reserved_fd(int fd) const {
  return reserved_fd_base_ > 0 && fd >= reserved_fd_base_ &&
         fd < reserved_fd_base_ + kReservedFdCount;
}

int SimulatedKfd::get_mmap_memfd(off_t offset) const {
  return get_mmap_memfd(local_process_id_, offset);
}

int SimulatedKfd::get_mmap_memfd(uint32_t process_id, off_t offset) const {
  auto p = find_process(process_id);
  if (!p)
    return -1;
  return dispatch_get_mmap_memfd(*p, offset, false).value_or(-1);
}

std::optional<int> SimulatedKfd::dup_mmap_memfd(uint32_t process_id, off_t offset) const {
  auto p = find_process(process_id);
  if (!p)
    return std::nullopt;
  return dispatch_get_mmap_memfd(*p, offset, true);
}

std::optional<int> SimulatedKfd::dispatch_get_mmap_memfd(KfdProcess &proc, off_t offset,
                                                         bool duplicate) const {
  const auto maybe_duplicate = [this, duplicate](int fd) -> std::optional<int> {
    if (fd < 0)
      return std::nullopt;
    if (!duplicate)
      return fd;
    if (fail_next_mmap_memfd_dup_.exchange(false, std::memory_order_acq_rel)) {
      errno = EMFILE;
      return -1;
    }
    return safe_fcntl(fd, F_DUPFD_CLOEXEC, 0);
  };
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;

  // EventState closes its backing only in its destructor. The find_process()
  // shared_ptr held by the caller keeps that owner alive through duplication;
  // backing_fd() locks publication of a newly created event fd.
  if (type == KFD_MMAP_TYPE_EVENTS)
    return maybe_duplicate(proc.event_state_.backing_fd());

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    uint32_t db_gpu_id = static_cast<uint32_t>(encoded_gpu);
    if (!find_gpu(db_gpu_id))
      return std::nullopt;
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    const auto &gs = proc.gpu(gpu_ordinal(db_gpu_id));
    if (gs.doorbell_memfd >= 0) {
      util::Logger::cp("MEMFD_LOOKUP: pid=", proc.process_id(),
                       " DOORBELL canonical gpu_id=", db_gpu_id, " memfd=", gs.doorbell_memfd);
      return maybe_duplicate(gs.doorbell_memfd);
    }
    // Compatibility fallback for callers that query the backing before the
    // mmap path has published the canonical descriptor.
    for (auto &[handle, alloc] : proc.allocations_) {
      if ((alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) && alloc.gpu_id == db_gpu_id) {
        util::Logger::cp("MEMFD_LOOKUP: pid=", proc.process_id(), " DOORBELL match handle=", handle,
                         " gpu_id=", db_gpu_id, " memfd=", alloc.memfd);
        return maybe_duplicate(alloc.memfd);
      }
    }
    util::Logger::cp("MEMFD_LOOKUP: pid=", proc.process_id(),
                     " DOORBELL NO MATCH gpu_id=", db_gpu_id,
                     " allocations=", proc.allocations_.size());
    return std::nullopt;
  }

  uint64_t handle = static_cast<uint64_t>(offset) >> 12;
  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(handle);
  if (it != proc.allocations_.end())
    return maybe_duplicate(it->second.memfd);

  return std::nullopt;
}

} // namespace rocjitsu
