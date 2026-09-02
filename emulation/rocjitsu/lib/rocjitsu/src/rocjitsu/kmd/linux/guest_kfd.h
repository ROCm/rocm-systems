// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file guest_kfd.h
/// @brief KFD discovery driver that appends one synthetic DBT guest GPU.

#ifndef ROCJITSU_KMD_LINUX_GUEST_KFD_H_
#define ROCJITSU_KMD_LINUX_GUEST_KFD_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/sysfs.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

class GuestKfdTestAccess;

/// @brief KFD driver that exposes a guest GPU for DBT while forwarding host GPU work.
///
/// @details This class appends one guest GPU to KFD topology and process
/// apertures, but does not execute guest queues itself. With the hardware
/// backend, host-GPU operations are forwarded to real /dev/kfd. With the
/// simulator backend, they are delegated to the supplied simulated KFD. If an
/// execution ioctl still targets the guest GPU, the driver returns an error so
/// the missing HSA forwarding path is visible.
class GuestKfd : public LinuxKfd {
public:
  /// @brief Construct a guest discovery driver from parsed DBT configuration.
  /// @param execution_driver Optional simulated host KFD. When null, host
  ///        execution is forwarded to the real `/dev/kfd`. When provided,
  ///        GuestKfd adopts its bootstrap open reference (or opens it lazily)
  ///        and releases that reference on the last application close.
  explicit GuestKfd(config::DbtGuestConfig config, LinuxKfd *execution_driver = nullptr);

  /// @brief Close the execution KFD and remove generated overlay state.
  ~GuestKfd() override;

  /// @brief Open the execution KFD and lazily prepare guest discovery.
  int open() override;

  /// @brief Handle close for the /dev/kfd fd represented by this driver.
  int close() override;

  /// @brief Route guest discovery ioctls locally and host ioctls to the execution KFD.
  int ioctl(unsigned long request, void *arg) override;

  /// @brief Map host-backed KFD offsets and reject unsupported guest doorbells.
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;

  /// @brief Forward unmaps for mappings created through this driver.
  int munmap(void *addr, size_t length) override;

  /// @brief Return the underlying hardware or simulated KFD fd.
  [[nodiscard]] int fd() const override;

  /// @brief Return true when @p fd is an internal rocjitsu-owned fd.
  [[nodiscard]] bool owns_fd(int fd) const override;

  /// @brief Redirect KFD topology and guest DRM sysfs paths into the overlay.
  [[nodiscard]] std::string redirect_sysfs_path(const char *path) const override;

  /// @brief Return true if a mapping range overlaps a protected doorbell.
  [[nodiscard]] bool is_doorbell_range(const void *addr, size_t length) const override;

  /// @brief Forward fixed-map replacement to the simulated execution driver when present.
  void *mmap_replacing_client_doorbell_views(void *addr, size_t length, int prot, int flags, int fd,
                                             off_t offset) override;

  /// @brief Return true when @p minor is the configured guest render node.
  [[nodiscard]] bool handles_drm_render_minor(uint32_t minor) const override;

  /// @brief Return synthetic AMDGPU metadata for the guest render node.
  [[nodiscard]] const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const override;

  /// @brief Return the generated KFD topology root.
  [[nodiscard]] std::string topology_path() const override;

  /// @brief Return the simulated host DRM root, or empty for hardware execution.
  [[nodiscard]] std::string drm_path() const override;

  /// @brief Prepare guest discovery without retaining an application open fd.
  bool prepare_for_discovery();

  /// @brief Add one open reference for a duplicated KFD fd.
  /// @retval true A reference was added.
  /// @retval false No live guest process to retain.
  [[nodiscard]] bool retain_local_open() override;

  /// @brief Number of app-facing KFD descriptor references still live.
  /// @details Only application-visible dup fds are counted — the internal real
  /// /dev/kfd fd is not — so this driver's count is zero at publication and its
  /// teardown baseline is zero.
  [[nodiscard]] uint32_t local_open_ref_count() const override;

  /// @brief Release blocking calls before teardown so their driver snapshot drops.
  /// @details Two distinct backends, one contract — in both cases a parked
  /// WAIT_EVENTS returns a benign KFD_IOC_WAIT_RESULT_TIMEOUT and nothing else is
  /// mutated:
  /// - Simulator execution: WAIT_EVENTS is served by the SimulatedKfd execution
  ///   driver, so this DELEGATES to it and its local process's waiters are
  ///   released.
  /// - Hardware execution: WAIT_EVENTS is forwarded to the real kernel, which this
  ///   process cannot interrupt. Instead this sets hw_closing_, which
  ///   forward_wait_events_bounded()'s poll loop observes between its short kernel
  ///   polls and returns on — CANCELLING the wait rather than waking it.
  /// Idempotent.
  void begin_local_shutdown() override;

  /// @brief Stop classifying the hidden real /dev/kfd fd number as KFD after a
  /// dup2/dup3 overwrote it.
  /// @details GuestKfd hands applications ordinary dup fds and keeps the real
  /// /dev/kfd fd internal (real_kfd_fd_). If a dup2/dup3 target reuses that hidden
  /// fd number, fd()/owns_fd() must stop reporting it as KFD so later ioctl/mmap/
  /// close are not routed to whatever now occupies the number. The real fd is NOT
  /// counted in open_refs_ (only app-facing dups are), so a match returns
  /// kClearedKeepRefs — the interposer must clear the classification WITHOUT
  /// dropping an open reference. For simulator execution, the separately owned
  /// backend open also stays pinned until the final app-facing close; a later
  /// primary-fd re-mint is balanced so it does not add another backend reference.
  /// Returns kNotPrimary if @p fd is not the current hidden real fd.
  [[nodiscard]] PrimaryInvalidation invalidate_primary_fd(int fd) override;

private:
  friend class GuestKfdTestAccess;

  class TopologyOverlay;

  /// @brief Work a final close hands back to be destroyed with mutex_ released.
  struct FinalCloseWork {
    int kfd_fd = -1;
    LinuxKfd *execution_driver = nullptr;
    std::unique_ptr<TopologyOverlay> overlay;
  };

  /// @brief Run the whole final-close state transition. Caller holds mutex_.
  /// @details Shared by close() and release_wait_lease() so a close that was deferred
  /// behind a wait lease performs the SAME transition when it finally runs, rather than
  /// only closing the descriptor. Readiness and epoch state can be republished while a
  /// close is pending, so the drain has to retire them again; leaving readiness set with
  /// no host fd wedges a hardware guest permanently, because ensure_ready() short-
  /// circuits on the flag and never reopens.
  /// @returns The descriptor, backend open and overlay to destroy outside mutex_.
  [[nodiscard]] FinalCloseWork
  begin_final_close_locked(std::unique_ptr<TopologyOverlay> fresh_overlay);

  /// @brief Destroy what begin_final_close_locked() returned, with mutex_ released.
  void finish_final_close(FinalCloseWork work);

  /// @brief Open real KFD, generate topology, and select the host GPU.
  bool ensure_ready();

  /// @brief Prepare guest discovery while mutex_ is already held.
  bool ensure_ready_locked();

  /// @brief Open the process's real /dev/kfd fd while mutex_ is held.
  bool ensure_real_kfd_locked();

  /// @brief Forward one ioctl to the real /dev/kfd fd.
  int forward_ioctl(unsigned long request, void *arg);

  /// @brief Forward WAIT_EVENTS to the real kernel as bounded, cancellable polls.
  /// @details A hardware-backed guest has no simulator wait to wake, and the real
  /// kernel WAIT_EVENTS can block indefinitely. The interposer dispatches this
  /// ioctl while holding a driver snapshot, so an indefinite kernel wait
  /// would keep that pin held and deadlock teardown (begin_local_shutdown() cannot
  /// wake a kernel syscall, and closing the fd does not cancel an in-flight wait).
  /// This breaks a long/indefinite wait into short kernel polls that re-check
  /// hw_closing_ between iterations, so begin_local_shutdown() can cancel it and the
  /// pin drains promptly. Mirrors the RemoteDriver client-side WAIT_EVENTS loop.
  int forward_wait_events_bounded(unsigned long request, void *arg);

  /// @brief The bounded-poll algorithm behind forward_wait_events_bounded().
  /// @details Split out and fully injected so its five outcomes — zero-timeout
  /// pass-through, cancellation (with the caller's timeout restored), indefinite
  /// polling until an event completes, finite-deadline expiry, and a failed poll —
  /// are unit testable without a real /dev/kfd. @p poll performs ONE short kernel
  /// poll, reading and writing @p args (the loop has already set args.timeout to the
  /// per-poll slice) and returning the ioctl result. @p cancelled is the flag
  /// begin_local_shutdown() sets; it is re-read between polls, never inside one.
  /// @returns The ioctl result to hand back to the caller. args.timeout is restored
  /// to the caller's original value on every path EXCEPT a failed poll of a finite
  /// wait, where what REMAINS of the budget is reported instead: that is the driver's
  /// restart convention, so a caller retrying on EINTR continues this wait rather
  /// than starting a fresh one. Zero is a legal remainder and means poll-once.
  static int wait_events_poll_loop(kfd_ioctl_wait_events_args &args,
                                   const std::atomic<bool> &cancelled,
                                   const std::function<int()> &poll);

  /// @brief Take a lease that keeps the host connection open for one whole wait.
  /// @details A caller-visible WAIT_EVENTS is served as MANY bounded ioctls, which puts
  /// descriptor-lifetime boundaries inside what the caller issued as one call. The
  /// kernel gives a native blocking ioctl a file reference for the entire syscall, so a
  /// concurrent close cannot revoke an operation already running; a sliced wait has no
  /// such protection, and the final close would otherwise clear the connection out from
  /// under a wait that had already started, failing it ENODEV partway instead of letting
  /// it complete with an event or a timeout. A lease makes that close DEFER: close()
  /// still returns immediately, as it must -- an indefinite wait cannot be allowed to
  /// block it -- and the last lease to drain performs the close that close() handed off.
  /// Cancellation is unaffected: begin_local_shutdown() sets its cancellation flag
  /// first, so shutdown still ends the wait promptly and only then drains.
  /// @retval false There is no host connection to lease; the caller must fail closed.
  [[nodiscard]] bool acquire_wait_lease();

  /// @brief Serve WAIT_EVENTS under a lease held for the caller-visible duration.
  /// @details The single entry point for both backends, so neither can bypass the lease.
  /// It also owns the choice between them: the simulator takes one ordinary blocking
  /// forward, since it serves the wait itself and begin_local_shutdown() can wake it;
  /// only the kernel needs the wait broken into cancellable slices.
  int wait_events(unsigned long request, void *arg);

  /// @brief Drop a wait lease, performing a close that close() deferred.
  /// @details The overlay it installs was prepared when the close was deferred, so this
  /// adds no allocation of its own. It can still reach a full backend teardown, which
  /// allocates, so callers drop the lease explicitly (WaitLease::release()) rather than
  /// leaving it to the destructor.
  void release_wait_lease();

  /// @brief RAII holder for acquire_wait_lease()/release_wait_lease().
  /// @details Callers should release() on the normal path. Dropping the LAST lease can
  /// run a deferred final close, and a backend teardown there allocates -- from a
  /// destructor, which is implicitly noexcept, that turns a bad_alloc into a terminate on
  /// the way out of an ioctl. Releasing explicitly keeps that work off the noexcept path
  /// and leaves the destructor as the unwind-only fallback it should be.
  class WaitLease {
  public:
    explicit WaitLease(GuestKfd &owner) : owner_(owner.acquire_wait_lease() ? &owner : nullptr) {}
    WaitLease(const WaitLease &) = delete;
    WaitLease &operator=(const WaitLease &) = delete;
    ~WaitLease() { release(); }

    /// @brief True if the lease was granted.
    [[nodiscard]] bool held() const { return owner_ != nullptr; }

    /// @brief Drop the lease now. Idempotent.
    void release() {
      if (owner_ != nullptr)
        std::exchange(owner_, nullptr)->release_wait_lease();
    }

  private:
    GuestKfd *owner_ = nullptr;
  };

  /// @brief Remember the device @p kfd_fd was opened on, for later identity checks.
  void record_kfd_identity(int kfd_fd);

  /// @brief True if @p fd is still the KFD character device we opened.
  /// @details An fd number is not identity. Anything holding a descriptor across a
  /// window where a concurrent close could recycle that number checks this before
  /// using it, so a recycled number fails closed rather than receiving driver traffic.
  [[nodiscard]] bool fd_is_expected_kfd(int fd) const;

  /// @brief Return real process apertures plus one synthetic guest aperture.
  int get_process_apertures_ioctl(void *arg) override;

  /// @brief Return guest clock-counter values or forward host requests.
  int get_clock_counters_ioctl(void *arg) override;

  /// @brief Succeed guest VM acquisition without creating a guest execution VM.
  int acquire_vm_ioctl(void *arg) override;

  /// @brief Report the configured guest-visible local memory size.
  int get_available_memory_ioctl(void *arg) override;

  /// @brief Accept guest startup memory policy setup and forward host policy.
  int set_memory_policy_ioctl(void *arg) override;

  /// @brief Allocate a synthetic KFD memory handle for guest startup bookkeeping.
  int alloc_memory_ioctl(void *arg) override;

  /// @brief Release a synthetic KFD memory handle or forward a real handle.
  int free_memory_ioctl(void *arg) override;

  /// @brief Rewrite guest gpu_id entries to the selected host before mapping.
  int map_memory_ioctl(void *arg) override;

  /// @brief Mirror map_memory rewrites for unmap requests.
  int unmap_memory_ioctl(void *arg) override;

  /// @brief Shared guest-to-host device-id rewrite for map/unmap memory ioctls.
  template <typename Args> int map_or_unmap_memory_ioctl(Args *args, unsigned long request);

  /// @brief Fail unsupported guest execution ioctls visibly.
  int reject_guest_execution_ioctl(unsigned long request, void *arg) const;

  /// @brief Return true when an ioctl argument names the synthetic guest GPU.
  bool request_targets_guest(unsigned long request, void *arg) const;

  /// @brief Build the synthetic aperture record appended after real apertures.
  kfd_process_device_apertures guest_apertures() const;

  config::DbtGuestConfig config_;
  /// @brief Non-owning simulated execution driver owned by the local VM.
  LinuxKfd *execution_driver_ = nullptr;
  Sysfs::GpuInfo guest_{};
  std::unique_ptr<TopologyOverlay> overlay_;
  mutable std::mutex mutex_;
  std::atomic<int> real_kfd_fd_{-1};
  /// @brief st_rdev of the device real_kfd_fd_ was opened on, or 0 if unknown.
  /// @details The fd NUMBER is loaded separately from any use of it, so a concurrent
  /// close plus a recycling open can put an unrelated file behind that number. Anything
  /// that re-loads the number across a window -- every slice of a sliced WAIT_EVENTS --
  /// compares the device identity against this before issuing a KFD ioctl, so a recycled
  /// number fails closed instead of silently receiving driver traffic.
  std::atomic<dev_t> real_kfd_rdev_{0};
  uint32_t open_refs_ = 0;
  /// @brief Caller-visible WAIT_EVENTS calls currently being served as sliced polls.
  uint32_t wait_leases_ = 0;
  /// @brief close() found a live wait lease and handed its close off to the drain.
  bool close_deferred_ = false;
  /// @brief Replacement overlay prepared for a deferred close, so the drain can install
  /// it without allocating.
  std::unique_ptr<TopologyOverlay> deferred_overlay_;
  bool owns_execution_driver_open_ = false;
  uint32_t host_gpu_id_ = 0;
  static constexpr uint64_t kSyntheticHandleBase = 1ULL << 63;
  uint64_t next_synthetic_handle_ = kSyntheticHandleBase;
  std::unordered_set<uint64_t> synthetic_handles_;
  std::unordered_set<uint64_t> synthetic_mmap_offsets_;
  std::atomic<bool> ready_{false};
  /// @brief Set by begin_local_shutdown() for a hardware-backed guest so an
  /// in-flight bounded WAIT_EVENTS poll loop returns and drops its driver snapshot
  /// before teardown.
  std::atomic<bool> hw_closing_{false};
  /// @brief The synthetic descriptor applications receive from open().
  /// @details A memfd, NEVER a duplicate of the real /dev/kfd, mirroring what
  /// SimulatedKfd hands out. This is a security boundary, not a convenience: a
  /// forked child inherits the parent's descriptor table, and under the
  /// fork-then-exec contract the interposer passes a child's ioctl/dup/dup2/fcntl
  /// straight to libc. Had the app-facing fd been a real KFD duplicate, that child
  /// could drive real hardware through it and dup2() it to clear FD_CLOEXEC and
  /// carry it across the exec that was supposed to sanitize the process. A memfd
  /// carries no such authority: inherited, it is inert.
  std::atomic<int> app_fd_{-1};

  /// @brief Create the synthetic app-facing descriptor once. Caller holds mutex_.
  bool ensure_app_fd_locked();

  /// @brief The PRIVATE real /dev/kfd descriptor used for host forwarding.
  /// @details Never returned to an application; see app_fd_.
  [[nodiscard]] int host_fd() const;
};

/// @brief Test-only handle to GuestKfd internals a real /dev/kfd would otherwise gate.
/// @details Both members below belong to the hardware-backed path, which needs a real
/// device to reach through GuestKfd::ioctl(); exposing them lets their contracts be
/// tested on any host. Nothing outside tests may use this.
class GuestKfdTestAccess {
public:
  /// @brief The bounded-wait algorithm: a pure function of its poll and cancel flag.
  static int wait_events_poll_loop(kfd_ioctl_wait_events_args &args,
                                   const std::atomic<bool> &cancelled,
                                   const std::function<int()> &poll) {
    return GuestKfd::wait_events_poll_loop(args, cancelled, poll);
  }

  /// @brief The lease a sliced wait holds for its whole caller-visible duration.
  /// @details What it guards is backend-independent -- close() must hand its work to the
  /// drain rather than revoke a wait already running -- so driving it directly tests that
  /// contract without a device, and without a wait long enough to race by hand.
  [[nodiscard]] static bool acquire_wait_lease(GuestKfd &guest) {
    return guest.acquire_wait_lease();
  }
  static void release_wait_lease(GuestKfd &guest) { guest.release_wait_lease(); }
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_GUEST_KFD_H_
