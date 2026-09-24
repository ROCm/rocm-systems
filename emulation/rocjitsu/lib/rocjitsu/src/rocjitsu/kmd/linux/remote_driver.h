// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_REMOTE_DRIVER_H_
#define ROCJITSU_KMD_LINUX_REMOTE_DRIVER_H_

/// @file remote_driver.h
/// @brief Client-side RPC stub that forwards KFD ioctls to the rocjitsu daemon.
///
/// @details Implements the Driver interface by serializing ioctl requests over
/// a Unix domain socket to the daemon process. GPU memory is shared via memfds
/// passed through SCM_RIGHTS. The client mmaps these memfds locally at the
/// addresses ROCR's FMM expects.

#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/driver.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

/// @brief What Yama's ptrace_scope implies for the daemon's cross-process reads.
/// @details Only one of these states makes the grant both necessary and
/// sufficient, and the others are not interchangeable: two of them mean the
/// access already works, and two mean naming a ptracer cannot make it work. The
/// distinction decides whether a failed grant is worth reporting, which is the
/// difference between a diagnosable failure and an SDMA copy that never retires.
enum class PtraceScopePolicy {
  Absent,     ///< Yama is not built in; process_vm_* is already permitted.
  Disabled,   ///< scope 0: unrestricted, so nothing needs restoring.
  Relational, ///< scope 1: descendants only, so the grant is required and works.
  AdminOnly,  ///< scope 2: CAP_SYS_PTRACE only; naming a ptracer changes nothing.
  NoAttach,   ///< scope 3: attaching is off entirely and cannot be re-enabled.
  Unknown,    ///< Present but unparsable; treated as if a grant were needed.
};

/// @brief Classify the contents of /proc/sys/kernel/yama/ptrace_scope.
/// @param[in] contents File contents, or nullopt when the file does not exist.
/// @returns The policy the value denotes.
[[nodiscard]] PtraceScopePolicy
ptrace_scope_policy_from_text(const std::optional<std::string> &contents);

/// @brief Whether a socket peer may be named a permitted ptracer of this process.
/// @details Separated from the syscalls so the policy is testable on its own.
/// This is the whole of the trust decision; everything around it is mechanism.
enum class PtracerGrantVerdict {
  Grant,               ///< The peer is the daemon this client's launcher started.
  GrantUnverifiedPeer, ///< No launcher named one, so the socket's owner is trusted.
  RefusedNoPeer,       ///< SO_PEERCRED gave nothing usable.
  RefusedForeignUser,  ///< The peer belongs to another user.
  RefusedPeerMismatch, ///< Someone other than that daemon is on this socket.
};

/// @brief Whether @p verdict authorizes the peer, however it was reached.
[[nodiscard]] constexpr bool grants(PtracerGrantVerdict verdict) {
  return verdict == PtracerGrantVerdict::Grant ||
         verdict == PtracerGrantVerdict::GrantUnverifiedPeer;
}

/// @brief Decide whether @p peer_pid may reach into this address space.
/// @param[in] launched_daemon_pid PID the launcher forked, or nullopt if none.
/// @param[in] peer_pid PID the kernel reports for the socket peer.
/// @param[in] peer_uid UID the kernel reports for the socket peer.
/// @param[in] self_uid Effective UID of this process.
/// @returns Why the grant is or is not permitted.
/// @details Where a launcher named a daemon, a peer that is not that process is
/// refused -- which is the case a well-known socket makes reachable, since any
/// process of this user can be the one listening on it. Where no launcher named
/// one, there is nothing to check against and the peer is trusted; that grant is
/// distinguished here rather than merged, because it is the weaker of the two
/// and its callers report it differently.
[[nodiscard]] PtracerGrantVerdict ptracer_grant_verdict(std::optional<pid_t> launched_daemon_pid,
                                                        pid_t peer_pid, uid_t peer_uid,
                                                        uid_t self_uid);

/// @brief Client-side driver that forwards ioctls to the rocjitsu daemon.
///
/// @details Connects to the daemon over a Unix domain socket and forwards
/// KFD ioctls via RPC. Owned by InterposerContext, not a global singleton.
class RemoteDriver : public Driver {
public:
  /// @brief Construct from an already-connected Unix socket fd.
  explicit RemoteDriver(int sock_fd);

  ~RemoteDriver() override;

  /// @brief Get the daemon's sysfs topology directory path.
  [[nodiscard]] const std::string &topology_path() const { return topology_path_; }

  /// @brief Get the daemon's DRM sysfs directory path.
  [[nodiscard]] const std::string &drm_path() const { return drm_path_; }

  /// @brief Get GPU metadata received from the daemon handshake.
  [[nodiscard]] const Sysfs::GpuInfo *gpu_info() const {
    return has_gpu_info_ ? &gpu_info_ : nullptr;
  }

  /// @brief What the last open() decided about authorizing its peer.
  /// @details Reported because the decision is otherwise invisible: a refusal
  /// leaves no trace in this object, and the grant itself is process-wide state
  /// no interface can read back -- the kernel offers no PR_GET_PTRACER. Without
  /// this, that a peer was refused, and that a failed handshake never reached
  /// the question at all, are both untestable. nullopt until open() runs.
  [[nodiscard]] std::optional<PtracerGrantVerdict> ptracer_verdict() const {
    return ptracer_verdict_;
  }

  /// @brief Outcome of find_memfd_for_addr(): distinguishes "no matching range"
  /// from "range matched but the memfd dup failed".
  /// @details The caller must treat these differently: kNotFound means fall back
  /// to the normal (anonymous) mapping; kDupFailed means a daemon-shared range
  /// DID cover the address but we could not hand out a descriptor for it (e.g.
  /// EMFILE/ENFILE), so falling back to an anonymous mapping would silently break
  /// the shared-memory invariant — the caller should fail the mmap instead.
  enum class MemfdLookup { kNotFound, kFound, kDupFailed };

  /// @brief Find a stored memfd that covers the given GPUVM address.
  /// @details Used by the interposer to intercept anonymous MAP_FIXED at
  /// addresses that have daemon-shared memfd mappings.
  /// @param addr The target address to look up.
  /// @param length The mapping length.
  /// @param[out] memfd_out On kFound, a NEWLY DUP'd memfd covering this address.
  ///             The caller OWNS this descriptor and MUST close() it after use;
  ///             it is a dup (taken under the RPC lock) so its lifetime is
  ///             independent of this RemoteDriver and a concurrent close() cannot
  ///             invalidate it mid-use.
  /// @param[out] memfd_offset On kFound, the offset within the memfd.
  /// @returns kFound if a range matched and the dup succeeded; kDupFailed if a
  ///          range matched but the dup failed; kNotFound if no range matched.
  [[nodiscard]] MemfdLookup find_memfd_for_addr(void *addr, size_t length, int *memfd_out,
                                                off_t *memfd_offset);

  /// @brief Perform the RPC handshake with the daemon.
  /// @details Sends RPC_HANDSHAKE, receives the topology path and gpu_id,
  /// and creates a synthetic memfd to use as the KFD fd.
  /// @retval >=0 Synthetic KFD fd on success.
  /// @retval -1 Handshake failed (socket error or daemon rejected).
  int open() override;

  /// @brief Mint a fresh synthetic KFD fd WITHOUT reconnecting or re-handshaking.
  /// @details Used when the interposer's cached primary fd number was lost (e.g.
  /// dup2 overwrote it) but the RPC connection is still live and must keep a
  /// valid primary fd number to hand back to open("/dev/kfd"). Unlike open(),
  /// this performs no RPC and does not disturb the connection/metadata.
  /// @retval >=0 A new synthetic KFD fd. @retval -1 memfd creation failed.
  [[nodiscard]] int reissue_synthetic_kfd_fd();

  /// @brief Send RPC_CLOSE to the daemon.
  /// @retval 0 Success.
  /// @retval -1 Socket error.
  int close() override;

  /// @brief Forward a KFD ioctl to the daemon via RPC_IOCTL.
  /// @param request The AMDKFD_IOC_* ioctl number.
  /// @param arg Pointer to the ioctl args struct (read and possibly modified).
  /// @retval 0 Success.
  /// @retval -EPROTO The RPC stream is unusable — this call or an earlier one
  /// failed to frame a request or reply, so the connection is terminal.
  /// @retval negative Negative errno from the daemon's ioctl dispatch, or from
  /// the transport when the request never reached the wire.
  int ioctl(unsigned long request, void *arg) override;

  /// @brief Forward an mmap request to the daemon via RPC_MMAP.
  /// @param addr Requested mapping address (may include MAP_FIXED).
  /// @param length Length in bytes to map.
  /// @param prot Memory protection flags.
  /// @param flags Mapping flags.
  /// @param offset KFD mmap offset encoding.
  /// @retval non-MAP_FAILED Pointer to the locally mapped memory.
  /// @retval MAP_FAILED Mapping failed; errno is set, and is EPROTO when the RPC
  /// stream has been made terminal by a framing failure.
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;

  /// @brief Forward a munmap request to the daemon via RPC_MUNMAP.
  /// @param addr Address of the mapping to unmap.
  /// @param length Length in bytes to unmap.
  /// @retval 0 Success.
  /// @retval -ENOENT Address not found in daemon's mappings.
  /// @retval -EPROTO The RPC stream is unusable — this call or an earlier one
  /// failed to frame a request or reply, so the connection is terminal.
  /// @retval negative Negative transport errno when the request never reached
  /// the wire.
  int munmap(void *addr, size_t length) override;

private:
  int send_ioctl(unsigned long request, void *arg);
  int send_mmap(void *addr, size_t length, int prot, int flags, off_t offset, int *memfd_out);

  /// @brief Mark the RPC stream unusable and make the connection terminal.
  /// @details Called from every framing failure — a half-written request, a
  /// reply we could not claim, a reply we only partly consumed — because all of
  /// them leave the two ends disagreeing about where the next frame starts.
  /// Sets protocol_failed_ so the ioctl/mmap/munmap entry points fail fast, then
  /// shuts the socket down so an in-flight peer write cannot keep feeding a
  /// stream nobody can parse. shutdown() rather than close(): the fd number
  /// stays reserved, so the concurrent readers of sock_ that rpc_mutex_ does not
  /// cover cannot land on a reused fd.
  /// @returns -EPROTO, so callers can `return poison_stream();`.
  /// @note Must be called with rpc_mutex_ held.
  int poison_stream();

  int sock_ = -1;             ///< Unix socket connection to the daemon.
  uint32_t next_id_ = 0;      ///< Monotonic request ID counter (for debugging).
  std::string topology_path_; ///< Daemon's sysfs topology directory path.
  std::string drm_path_;      ///< Daemon's DRM sysfs directory path.
  Sysfs::GpuInfo gpu_info_{}; ///< GPU metadata received from daemon handshake.
  bool has_gpu_info_ = false; ///< True when gpu_info_ is valid.
  std::optional<PtracerGrantVerdict> ptracer_verdict_; ///< Last open()'s trust decision.
  std::atomic<bool> closing_{false}; ///< Set by close() to break WAIT_EVENTS loops.
  /// @brief Set when the RPC stream is known to be unusable.
  /// @details Any frame that is not written or read in full leaves the stream
  /// misaligned — a reply header we refuse to read the body of, a request that
  /// stopped mid-write, a reply left queued because we could not receive it — so
  /// every later call would parse its header out of the stale bytes. Once this
  /// is set the connection is terminal and further ioctl/mmap/munmap calls fail
  /// with -EPROTO rather than returning bogus results from a poisoned stream.
  std::atomic<bool> protocol_failed_{false};
  int shutdown_efd_ = -1;      ///< eventfd written by close() to wake WAIT_EVENTS pollers.
  void *kfd_marker_ = nullptr; ///< Non-readable mapping identifying /dev/kfd in proc maps.
  size_t kfd_marker_size_ = 0;

  /// @brief Serializes all RPC send+recv pairs on sock_.
  /// @details ROCR is multithreaded — concurrent ioctl/mmap calls interleave
  /// socket writes without this lock, corrupting the RPC stream.
  std::mutex rpc_mutex_;

  /// @brief Memfds received from the daemon during ALLOC_MEMORY, keyed by handle.
  std::unordered_map<uint64_t, int> handle_memfds_;

  /// @brief Maps GPUVM addresses to allocation memfds for anonymous MAP_FIXED
  /// interception. When ROCR's FMM does anonymous MAP_FIXED at a GPUVM address
  /// that already has a daemon-shared memfd, we use the memfd instead to
  /// preserve cross-process sharing. Keyed by alloc va_addr, value is memfd.
  struct AllocRange {
    uint64_t va;
    uint64_t size;
    int memfd;
  };
  std::vector<AllocRange> alloc_ranges_;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_REMOTE_DRIVER_H_
