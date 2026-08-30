// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file remote_driver.cpp
/// @brief Client-side RPC stub for the rocjitsu daemon.

#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "util/log.h"
#include "util/unique_handle.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <fstream>
#include <linux/mman.h>
#include <optional>
#include <string_view>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#include <mutex>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#ifndef PR_SET_PTRACER
// Yama's ptrace-scope exception list. Absent from some libc header sets even
// where the running kernel implements it, so the call below is worth keeping
// unconditional; the kernel answers EINVAL when it is genuinely unsupported.
#define PR_SET_PTRACER 0x59616d61
#endif
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu {
namespace {

constexpr bool has_embedded_pointers(unsigned long request) {
  switch (canonical_ioctl_request(request)) {
  case AMDKFD_IOC_WAIT_EVENTS:
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
  case AMDKFD_IOC_DBG_TRAP:
    return true;
  case AMDKFD_IOC_SVM:
    // SVM's variable-length attribute array is part of the ioctl payload, not a
    // client pointer that the daemon has to rewrite.
    return false;
  default:
    return false;
  }
}

/// @brief Negative errno for a transport call that failed without putting a
/// frame on the wire.
///
/// @details Every such site used to fall back to a bare -1 when errno was
/// unset. That collides with -EPERM: EPERM is 1, so the daemon's own -EPERM --
/// which arrives by exactly the same return path -- became indistinguishable
/// from "no errno available", and any consumer trying to tell them apart got
/// one of the two wrong. Substituting EIO keeps every negative return a real
/// errno, so callers can translate unconditionally and -1 means EPERM and
/// nothing else.
int transport_errno() {
  const int err = errno;
  return err > 0 ? -err : -EIO;
}

/// @brief Permit the daemon on @p socket to read and write this address space.
///
/// @details Host memory that never became a daemon mapping is reachable only by
/// the daemon reaching into this process. Its memory bridge already assumes it
/// can, through process_vm_readv/writev, and GPU access to such memory is
/// serviced no other way -- pageable buffers the runtime pinned through the KFD
/// SVM API, above all, since that registration produces no shared backing the
/// way a USERPTR allocation does. Those calls are nonetheless refused by
/// default: the daemon is forked as this process's child, and Yama's default
/// ptrace_scope admits a ptracer only over its own descendants, never over its
/// own ancestor. Naming the daemon a permitted ptracer restores the access the
/// bridge was written to have; without it every such access fails EPERM and the
/// SDMA copy that needed it never completes.
///
/// The peer PID comes from the kernel rather than from the daemon, so this
/// names exactly the process on the other end of this socket. The grant widens
/// nothing that was not already true: the daemon's own client library is this
/// LD_PRELOAD, and is already executing inside this address space.
///
/// @brief The daemon PID this client's launcher forked, if it had one.
/// @details Read from the environment rather than the socket on purpose: the
/// socket answers who is connected, and this answers who was supposed to be.
std::optional<pid_t> launched_daemon_pid() {
  const char *raw = getenv(kRpcDaemonPidEnv);
  if (raw == nullptr || *raw == '\0')
    return std::nullopt;
  pid_t parsed = 0;
  const char *end = raw + std::strlen(raw);
  const auto [stop, error] = std::from_chars(raw, end, parsed);
  if (error != std::errc{} || stop != end || parsed <= 0)
    return std::nullopt;
  return parsed;
}

/// @brief Contents of Yama's ptrace_scope, or nullopt when it does not exist.
std::optional<std::string> read_ptrace_scope() {
  std::ifstream file("/proc/sys/kernel/yama/ptrace_scope");
  if (!file)
    return std::nullopt;
  std::string contents;
  std::getline(file, contents);
  return contents;
}

/// @brief Why a peer was refused, for the log line that reports it.
std::string_view describe(PtracerGrantVerdict verdict) {
  switch (verdict) {
  case PtracerGrantVerdict::Grant:
    return "the peer is the daemon this client's launcher started";
  case PtracerGrantVerdict::GrantUnverifiedPeer:
    return "no launcher named a daemon, so the socket's owner is trusted";
  case PtracerGrantVerdict::RefusedNoPeer:
    return "the socket reports no usable peer credentials";
  case PtracerGrantVerdict::RefusedForeignUser:
    return "the peer belongs to another user";
  case PtracerGrantVerdict::RefusedPeerMismatch:
    return "the peer is not the daemon this client's launcher started";
  }
  return "unrecognized";
}

/// Called only once the handshake has succeeded. Before that this process has
/// no evidence the peer speaks the protocol at all, and a grant is the one step
/// here that outlives the connection.
///
/// @param[in] socket Connected daemon socket.
/// @returns The trust decision, whether or not a syscall followed from it.
PtracerGrantVerdict authorize_daemon_address_space_access(int socket) {
  ucred peer{};
  socklen_t peer_size = sizeof(peer);
  const bool have_peer = getsockopt(socket, SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) == 0 &&
                         peer_size == sizeof(peer) && peer.pid > 0;

  const PtracerGrantVerdict verdict =
      have_peer ? ptracer_grant_verdict(launched_daemon_pid(), peer.pid, peer.uid, geteuid())
                : PtracerGrantVerdict::RefusedNoPeer;
  if (!grants(verdict)) {
    // Reported rather than dropped. A refusal is not a failure in itself, but
    // it decides whether pageable transfers above the pinning threshold can
    // complete, and a caller left wondering why an SDMA copy never retires has
    // no other way to find out.
    util::Logger::warn(
        std::format("daemon: not authorizing peer to read this address space ({}); transfers "
                    "that rely on process_vm_readv/writev will fail",
                    describe(verdict)));
    return verdict;
  }
  if (verdict == PtracerGrantVerdict::GrantUnverifiedPeer) {
    // Said out loud, every time. This is the grant that rests on nothing but the
    // peer having answered this socket, so which process received it belongs in
    // the log rather than only in the kernel's unreadable exception slot.
    util::Logger::warn(
        std::format("daemon: authorizing pid {} to read this address space; no launcher named a "
                    "daemon for this client, so nothing here identifies it further",
                    peer.pid));
  }

  const PtraceScopePolicy policy = ptrace_scope_policy_from_text(read_ptrace_scope());
  if (policy == PtraceScopePolicy::Absent || policy == PtraceScopePolicy::Disabled)
    return verdict; // Already permitted; the grant would be a no-op.
  if (policy == PtraceScopePolicy::AdminOnly || policy == PtraceScopePolicy::NoAttach) {
    util::Logger::warn(
        "daemon: ptrace_scope forbids attaching regardless of a permitted-ptracer grant, so "
        "transfers that rely on process_vm_readv/writev will fail");
    return verdict;
  }

  // Process-wide, single-valued, and unreadable: PR_SET_PTRACER holds one PID
  // with no PR_GET_PTRACER to inspect it, so this necessarily overwrites
  // whatever was there. That is why it is narrowed as far as it is -- issued
  // only for a handshaked peer that the launcher named, and only in the one
  // scope where it changes anything. A reconnect re-issues it for the new
  // daemon, which is correct: the previous one is the process this client is no
  // longer being served by. A debugger or rocSHMEM grant taken beforehand is
  // displaced, and cannot be restored afterwards because it cannot be read.
  if (prctl(PR_SET_PTRACER, static_cast<unsigned long>(peer.pid), 0, 0, 0) != 0) {
    util::Logger::warn(std::format(
        "daemon: could not authorize pid {} to read this address space ({}); transfers that "
        "rely on process_vm_readv/writev will fail",
        peer.pid, std::strerror(errno)));
  }
  return verdict;
}

/// @brief Safe wrapper around syscall(SYS_mmap, ...) that avoids UB from
/// casting negative return values through uintptr_t/pointer types.
void *safe_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  long rc = syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
  if (rc < 0)
    return MAP_FAILED;
  return reinterpret_cast<void *>(static_cast<uintptr_t>(rc));
}

Sysfs::GpuInfo gpu_info_from_rpc(const RpcGpuInfo &src) {
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = src.gpu_id;
  gpu.gfx_target_version = src.gfx_target_version;
  gpu.vendor_id = src.vendor_id;
  gpu.device_id = src.device_id;
  gpu.family_id = src.family_id;
  gpu.unique_id = src.unique_id;
  gpu.location_id = src.location_id;
  gpu.domain = src.domain;
  gpu.hive_id = src.hive_id;
  gpu.drm_render_minor = src.drm_render_minor;
  gpu.revision_id = src.revision_id;
  gpu.pci_revision_id = src.pci_revision_id;
  gpu.simd_count = src.simd_count;
  gpu.max_waves_per_simd = src.max_waves_per_simd;
  gpu.num_shader_engines = src.num_shader_engines;
  gpu.num_shader_arrays_per_engine = src.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = src.num_cu_per_sh;
  gpu.simd_per_cu = src.simd_per_cu;
  gpu.wave_front_size = src.wave_front_size;
  gpu.num_xcc = src.num_xcc;
  gpu.max_slots_scratch_cu = src.max_slots_scratch_cu;
  gpu.local_mem_size = src.local_mem_size;
  gpu.vram_type = src.vram_type;
  gpu.lds_size_kb = src.lds_size_kb;
  gpu.mem_width = src.mem_width;
  gpu.mem_clk_max = src.mem_clk_max;
  gpu.l1_size_kb = src.l1_size_kb;
  gpu.l1_line_size = src.l1_line_size;
  gpu.l1_assoc = src.l1_assoc;
  gpu.l2_size_kb = src.l2_size_kb;
  gpu.l2_line_size = src.l2_line_size;
  gpu.l2_assoc = src.l2_assoc;
  gpu.num_sdma_engines = src.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = src.num_sdma_xgmi_engines;
  gpu.num_cp_queues = src.num_cp_queues;
  gpu.max_engine_clk_fcompute = src.max_engine_clk_fcompute;
  gpu.capability = src.capability;
  gpu.capability2 = src.capability2;
  gpu.debug_prop = src.debug_prop;
  gpu.fw_version = src.fw_version;
  gpu.sdma_fw_version = src.sdma_fw_version;

  auto *name_end =
      static_cast<const char *>(std::memchr(src.marketing_name, '\0', sizeof(src.marketing_name)));
  auto name_len =
      name_end ? static_cast<size_t>(name_end - src.marketing_name) : sizeof(src.marketing_name);
  gpu.marketing_name.assign(src.marketing_name, name_len);
  return gpu;
}

/// @brief Ceiling on the entry count a snapshot request may reserve room for.
///
/// @details Two orders of magnitude above anything a KFD enumerates (the
/// driver's own MAX_GPU_INSTANCE and libhsakmt's NUM_OF_SUPPORTED_GPUS are both
/// 64), so it can never truncate a real answer, while keeping the reserved tail
/// at half a megabyte instead of the whole payload budget. The same ceiling
/// bounds the queue snapshot: a process with more than 4096 live queues has
/// exhausted the driver's own doorbell budget long before it gets here.
inline constexpr uint32_t kMaxSnapshotEntries = 4096;

/// @brief The request fields shared by the two DBG_TRAP snapshot ops.
struct SnapshotFields {
  // __u64/__u32 rather than uint64_t/uint32_t: these alias the uapi struct's
  // own fields, and __u64 is unsigned long long where uint64_t is unsigned long.
  __u64 *buf_ptr;
  __u32 *count;
  __u32 *entry_size;
  uint32_t struct_size;
};

/// @brief Bind the snapshot request fields of @p dbg for its op.
///
/// @details GET_DEVICE_SNAPSHOT and GET_QUEUE_SNAPSHOT differ only in which
/// sub-struct of kfd_ioctl_dbg_trap_args carries the caller's buffer, entry
/// count and stride, and in the entry struct the driver fills. Binding them in
/// one place lets the request clamp and the strided copy-back below serve both
/// ops instead of drifting into two divergent copies of the same reasoning.
///
/// @p op is passed separately rather than read from @p dbg because the reply
/// path runs against an arg struct the daemon echoed back; the op that selects
/// the union member has to be the one *we* sent, not one a peer could vary.
///
/// @pre @p op is GET_DEVICE_SNAPSHOT or GET_QUEUE_SNAPSHOT.
SnapshotFields snapshot_fields(kfd_ioctl_dbg_trap_args *dbg, uint32_t op) {
  if (op == KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT)
    return {&dbg->device_snapshot.snapshot_buf_ptr, &dbg->device_snapshot.num_devices,
            &dbg->device_snapshot.entry_size, sizeof(kfd_dbg_device_info_entry)};
  return {&dbg->queue_snapshot.snapshot_buf_ptr, &dbg->queue_snapshot.num_queues,
          &dbg->queue_snapshot.entry_size, sizeof(kfd_queue_snapshot_entry)};
}

/// @brief Clamp a DBG_TRAP device-snapshot request to a transmittable size.
///
/// @details The snapshot buffer is pure ioctl output, so the request carries an
/// inline tail of @p count * @p entry_size zero bytes purely to reserve room
/// for the daemon's reply. The count is caller-controlled, and callers are
/// entitled to oversize it: the uapi contract is that KFD fills only the
/// entries that exist and reports the true total back, so libhsakmt's
/// hsaKmtDbgGetDeviceDataCtx() just passes UINT32_MAX. Taken literally that
/// reserves UINT32_MAX * sizeof(kfd_dbg_device_info_entry) bytes (120 B/entry,
/// so ~480 GiB); the resize throws std::bad_alloc out of an interposed ioctl()
/// and takes the process down.
///
/// Clamping the transmitted count keeps the ioctl's semantics rather than
/// failing a request the kernel would have served: the daemon still reports the
/// true total in the count(OUT) field, and the ceiling is far above any device
/// or queue count a KFD can enumerate, so it cannot drop an entry the daemon
/// could have returned.
///
/// The ceiling is @ref kMaxSnapshotEntries rather than the transport's raw
/// capacity. Filling the whole payload budget would be correct but ruinous: at
/// 120 B/entry it reserves ~16 MiB of zeros in the request and makes the daemon
/// echo ~16 MiB back (rj_daemon replies with the buffer it received), i.e. ~32
/// MiB moved under rpc_mutex_ on every debugger attach to describe one GPU.
///
/// A zero return for a non-zero @p count means not even one entry fits; the
/// caller must fail the ioctl rather than transmit the count, because
/// count(IN) == 0 is the count-only probe the daemon answers with success.
///
/// @returns the entry count to transmit, whose tail is guaranteed to fit
/// within @ref kMaxPayloadBytes alongside @p arg_size bytes of ioctl args.
uint32_t clamp_snapshot_entries(uint32_t count, uint32_t entry_size, size_t arg_size) {
  // A zero stride reserves nothing whatever the count is, so pass the caller's
  // count through untouched. The driver does not reject a zero stride: its
  // per-entry copy_to_user() moves entry_size(OUT) == 0 bytes and succeeds, so
  // the call reports the device total and writes nothing
  // (DbgTrapDeviceSnapshotZeroStrideReportsCountAndWritesNothing pins that
  // locally). count(IN) is still what the driver clamps its fill count against,
  // and it is echoed back as part of the request, so rewriting it to zero here
  // would hand the daemon a different request than the caller made.
  if (entry_size == 0)
    return count;

  const size_t overhead = sizeof(RpcIoctlRequest) + arg_size;
  if (overhead >= kMaxPayloadBytes)
    return 0;
  const size_t max_entries =
      std::min<size_t>(kMaxSnapshotEntries, (kMaxPayloadBytes - overhead) / entry_size);
  return static_cast<uint32_t>(std::min<size_t>(count, max_entries));
}

} // namespace

PtraceScopePolicy ptrace_scope_policy_from_text(const std::optional<std::string> &contents) {
  if (!contents)
    return PtraceScopePolicy::Absent;
  std::string_view text = *contents;
  while (!text.empty() && (text.front() == ' ' || text.front() == '\t'))
    text.remove_prefix(1);
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\t' || text.back() == '\n' || text.back() == '\r'))
    text.remove_suffix(1);
  int value = 0;
  const auto [stop, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || stop != text.data() + text.size())
    return PtraceScopePolicy::Unknown;
  switch (value) {
  case 0:
    return PtraceScopePolicy::Disabled;
  case 1:
    return PtraceScopePolicy::Relational;
  case 2:
    return PtraceScopePolicy::AdminOnly;
  case 3:
    return PtraceScopePolicy::NoAttach;
  default:
    // A scope this build does not know is assumed to restrict at least as much
    // as the one it does, so the grant is still attempted rather than skipped.
    return PtraceScopePolicy::Unknown;
  }
}

PtracerGrantVerdict ptracer_grant_verdict(std::optional<pid_t> launched_daemon_pid, pid_t peer_pid,
                                          uid_t peer_uid, uid_t self_uid) {
  if (peer_pid <= 0)
    return PtracerGrantVerdict::RefusedNoPeer;
  // A cross-user peer could not attach even if named here, because PR_SET_PTRACER
  // relaxes Yama alone and the ordinary ptrace credential check still stands
  // behind it. Refused anyway, so the exception list never holds a process this
  // client did not expect to be served by.
  if (peer_uid != self_uid)
    return PtracerGrantVerdict::RefusedForeignUser;
  // SO_PEERCRED says who is connected, which is a weaker claim than it appears:
  // the socket path is well known, so any process of this user can be listening
  // on it, and same-user peers are exactly what Yama's relationship check exists
  // to separate. Where a launcher named a process, that name decides.
  if (launched_daemon_pid) {
    return *launched_daemon_pid == peer_pid ? PtracerGrantVerdict::Grant
                                            : PtracerGrantVerdict::RefusedPeerMismatch;
  }
  // No launcher named one: attach mode, or a bare LD_PRELOAD against a running
  // daemon. Nothing here can identify the peer, and no check could be added that
  // would: a secret shared with the daemon authenticates nothing against an
  // attacker of this UID, who reads whatever this client can. The peer is
  // trusted, and every such grant says so.
  return PtracerGrantVerdict::GrantUnverifiedPeer;
}

RemoteDriver::MemfdLookup RemoteDriver::find_memfd_for_addr(void *addr, size_t length,
                                                            int *memfd_out, off_t *offset_out) {
  *memfd_out = -1;
  *offset_out = 0;
  auto target = reinterpret_cast<uint64_t>(addr);
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  for (const auto &r : alloc_ranges_) {
    // Overflow-safe containment test for [target, target+length) within
    // [r.va, r.va+r.size). Rearranged to subtraction so neither target+length
    // nor r.va+r.size can wrap uint64_t and admit an out-of-range address.
    if (target >= r.va && length <= r.size && target - r.va <= r.size - length) {
      // Return a DUP of the memfd, not the stored fd, so the caller owns a
      // descriptor whose lifetime is independent of this RemoteDriver. Without
      // this, a concurrent last-close -> RemoteDriver::close() (which closes the
      // stored handle_memfds_ fds) could close the fd between this return and the
      // caller's ftruncate/fallocate/mmap, operating on a closed/reused fd. The
      // dup is taken under rpc_mutex_, the same lock close() holds, so the stored
      // fd is guaranteed still open here. The caller MUST close the returned fd.
      int dup_fd = static_cast<int>(syscall(SYS_fcntl, r.memfd, F_DUPFD_CLOEXEC, 0));
      if (dup_fd < 0) {
        // The range matched but we could not hand out a descriptor. Report this
        // distinctly so the caller fails the mmap rather than silently falling
        // back to an anonymous mapping and breaking the shared-memory invariant.
        // The dup's errno (EMFILE/ENFILE/...) reaches the caller: the only code
        // that runs before this returns is the rpc_mutex_ lock_guard unlock, and
        // pthread_mutex_unlock does not touch errno on success. Do not add any
        // errno-setting call after this point without saving/restoring it.
        return MemfdLookup::kDupFailed;
      }
      *memfd_out = dup_fd;
      *offset_out = static_cast<off_t>(target - r.va);
      return MemfdLookup::kFound;
    }
  }
  return MemfdLookup::kNotFound;
}

RemoteDriver::RemoteDriver(int sock_fd) : sock_(sock_fd) {
  shutdown_efd_ = static_cast<int>(syscall(SYS_eventfd2, 0, EFD_CLOEXEC | EFD_NONBLOCK));
}

RemoteDriver::~RemoteDriver() {
  for (auto &[handle, fd] : handle_memfds_) {
    if (fd >= 0)
      syscall(SYS_close, fd);
  }
  if (kfd_marker_ != nullptr)
    syscall(SYS_munmap, kfd_marker_, kfd_marker_size_);
  if (sock_ >= 0)
    syscall(SYS_close, sock_);
  if (shutdown_efd_ >= 0)
    syscall(SYS_close, shutdown_efd_);
}

int RemoteDriver::poison_stream() {
  protocol_failed_.store(true, std::memory_order_release);
  if (sock_ >= 0)
    syscall(SYS_shutdown, sock_, SHUT_RDWR);
  return -EPROTO;
}

int RemoteDriver::open() {
  assert(sock_ >= 0 && "open called on disconnected RemoteDriver");
  ptracer_verdict_.reset();
  closing_.store(false, std::memory_order_release);
  has_gpu_info_ = false;
  gpu_info_ = {};
  topology_path_.clear();
  drm_path_.clear();
  // Drain the shutdown eventfd so it doesn't immediately wake pollers.
  if (shutdown_efd_ >= 0) {
    uint64_t val = 0;
    syscall(SYS_read, shutdown_efd_, &val, sizeof(val));
  }
  std::lock_guard<std::mutex> lock(rpc_mutex_);

  RpcHeader hdr = {};
  hdr.opcode = RPC_HANDSHAKE;
  hdr.request_id = next_id_++;
  hdr.payload_bytes = 0;

  // The handshake is framed like every other exchange, so it fails the same way:
  // a request that stopped mid-write, or a reply we abandon part-read or leave
  // undrained, desyncs the stream for whatever the caller does next. Poison
  // before returning so a failed handshake cannot be followed by ioctls that
  // decode their replies out of leftover handshake bytes. The failure code stays
  // -1 because open() returns an fd, not an errno.
  size_t handshake_bytes = 0;
  if (!rpc_send_exact(sock_, &hdr, sizeof(hdr), &handshake_bytes)) {
    if (handshake_bytes > 0)
      poison_stream();
    return -1;
  }

  RpcHeader resp = {};
  if (!rpc_recv_exact(sock_, &resp, sizeof(resp))) {
    poison_stream();
    return -1;
  }

  if (resp.result != 0)
    return resp.result;

  RpcHandshakeResponse hs = {};
  if (!rpc_recv_exact(sock_, &hs, sizeof(hs))) {
    poison_stream();
    return -1;
  }

  // Every rejection from here on abandons path bytes the reply header already
  // declared, which is the same misalignment as a short read.
  if (hs.version != kRpcProtocolVersion) {
    poison_stream();
    return -1;
  }

  has_gpu_info_ = hs.gpu_info.present != 0;
  if (has_gpu_info_)
    gpu_info_ = gpu_info_from_rpc(hs.gpu_info);

  constexpr uint32_t kMaxPathLen = 4096;
  if (hs.topology_path_len > kMaxPathLen) {
    poison_stream();
    return -1;
  }
  if (hs.topology_path_len > 0) {
    topology_path_.resize(hs.topology_path_len);
    if (!rpc_recv_exact(sock_, topology_path_.data(), hs.topology_path_len)) {
      poison_stream();
      return -1;
    }
  }

  if (hs.drm_path_len > kMaxPathLen) {
    poison_stream();
    return -1;
  }
  if (hs.drm_path_len > 0) {
    drm_path_.resize(hs.drm_path_len);
    if (!rpc_recv_exact(sock_, drm_path_.data(), hs.drm_path_len)) {
      poison_stream();
      return -1;
    }
  }

  // Only here, with the handshake complete. Everything above can fail on a peer
  // that never proved it speaks this protocol, and the grant is the one step in
  // open() whose effect outlives the connection that prompted it.
  ptracer_verdict_ = authorize_daemon_address_space_access(sock_);

  return reissue_synthetic_kfd_fd();
}

int RemoteDriver::reissue_synthetic_kfd_fd() {
  // Create a high-numbered synthetic KFD fd to avoid collisions with ROCR's
  // internal fd lifecycle. Use the top of the current rlimit range (same
  // approach as SimulatedKfd::init_reserved_fd_range).
  struct rlimit rl {};
  getrlimit(RLIMIT_NOFILE, &rl);
  int fd_min = static_cast<int>(rl.rlim_cur) - 64;
  if (fd_min < 256)
    fd_min = 256;
  // NOTE: the name here is deliberately NOT "/dev/kfd". Naming it that makes
  // /proc/<pid>/maps carry a second "/memfd:/dev/kfd (deleted)" line, which
  // breaks gdb.rocm/core-no-read-special-files.exp -- that test parses
  // `info proc mappings` for the one real /dev/kfd mapping and finds none once
  // this decoy is present. InterposerDupTest.ProcMapsNamesRemoteKfdMarker
  // asserts the opposite (that maps names /dev/kfd and never this marker); the
  // two expectations are in direct conflict and the upstream ROCgdb test wins,
  // so that unit test currently only passes on a daemon-backed run and needs a
  // design decision about what the marker is actually for.
  auto raw_fd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_remote_kfd", MFD_CLOEXEC));
  if (raw_fd < 0)
    return -1;
  if (kfd_marker_ == nullptr) {
    constexpr size_t kMarkerSize = 4096;
    if (syscall(SYS_ftruncate, raw_fd, kMarkerSize) == 0) {
      void *marker = reinterpret_cast<void *>(
          syscall(SYS_mmap, nullptr, kMarkerSize, PROT_NONE, MAP_SHARED, raw_fd, 0));
      if (marker != MAP_FAILED) {
        kfd_marker_ = marker;
        kfd_marker_size_ = kMarkerSize;
      }
    }
  }
  // Use the raw syscall, not fcntl(): this shared object exports an interposed
  // fcntl with default visibility, so an unqualified call would re-enter the
  // shim (reserve_dup_backend/untrack_dup, fd_mutex_) for a plain memfd dup.
  int fd = static_cast<int>(syscall(SYS_fcntl, raw_fd, F_DUPFD_CLOEXEC, fd_min));
  syscall(SYS_close, raw_fd);
  return fd;
}

int RemoteDriver::close() {
  closing_.store(true, std::memory_order_release);
  if (shutdown_efd_ >= 0) {
    uint64_t val = 1;
    syscall(SYS_write, shutdown_efd_, &val, sizeof(val));
  }

  {
    std::lock_guard<std::mutex> lock(rpc_mutex_);
    if (sock_ >= 0) {
      RpcHeader hdr{};
      hdr.opcode = RPC_CLOSE;
      hdr.request_id = next_id_++;
      rpc_send_exact(sock_, &hdr, sizeof(hdr));
    }

    if (sock_ >= 0) {
      syscall(SYS_close, sock_);
      sock_ = -1;
    }

    for (auto &[handle, fd] : handle_memfds_) {
      if (fd >= 0)
        syscall(SYS_close, fd);
    }
    handle_memfds_.clear();
    alloc_ranges_.clear();
    // Deliberately do NOT clear the handshake metadata (topology_path_,
    // drm_path_, gpu_info_, has_gpu_info_). It is written once during open() and
    // is immutable for the rest of the object's life. The interposer publishes
    // this RemoteDriver via an atomic shared_ptr and lets lock-free readers take
    // snapshots that call the accessors (topology_path(), drm_path(),
    // gpu_info()); a racing teardown clears the atomic and calls close() while
    // such a snapshot may still be live. Clearing these strings/structs here
    // would be a data race against those readers. Keeping the metadata
    // immutable-after-open makes the reads safe without a lock; the storage is
    // reclaimed when the last shared_ptr drops.
  }

  return 0;
}

int RemoteDriver::ioctl(unsigned long request, void *arg) {
  assert(arg && "ioctl called with null arg");
  // Do NOT assert(sock_ >= 0) here: sock_ is guarded by rpc_mutex_ and a
  // concurrent teardown_remote() -> close() can set it to -1 while another
  // thread holds a live shared_ptr snapshot and is entering this call. Reading
  // sock_ unlocked would be a data race (and would abort in Debug on exactly the
  // teardown-vs-in-flight-ioctl race this design tolerates). The locked send
  // path (send_ioctl) handles a closed socket gracefully by returning -1.

  // WAIT_EVENTS is handled client-side to avoid rpc_mutex_ contention.
  // Multiple ROCR threads poll WAIT_EVENTS concurrently during init. If each
  // poll goes through RPC, the rpc_mutex_ is held for the round-trip duration,
  // starving the main init thread's ioctls and mmaps.
  //
  // The signal page IS shared via memfd (same inode in both processes). The
  // daemon's signal_interrupt writes to the signal page slot. The client polls
  // the slot directly — no RPC round-trip, no mutex contention.
  //
  // For non-signal events (e.g., queue-inactive notifications), a single RPC
  // poll checks the daemon's EventState.
  if (request == AMDKFD_IOC_WAIT_EVENTS) {
    auto *wait_args = static_cast<kfd_ioctl_wait_events_args *>(arg);
    uint32_t original_timeout = wait_args->timeout;
    // timeout=0: return immediately without RPC. Signal values live in shared
    // memory (memfd) — ROCR reads them directly, so the WAIT_EVENTS ioctl
    // only needs to check for non-signal KFD events. Avoid the RPC round-trip
    // and rpc_mutex_ contention that starves the init thread.
    if (original_timeout == 0) {
      wait_args->wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
      return 0;
    }

    // Blocking wait: poll the daemon periodically. Between polls, block on
    // the shutdown eventfd via poll(2) instead of sleeping. When close()
    // writes to the eventfd, poll returns immediately and the loop exits.
    // This avoids both the shutdown ordering deadlock (ROCR joins signal
    // threads before calling close) AND arbitrary time-based workarounds.
    auto deadline =
        (original_timeout == kWaitEventsInfiniteMs)
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + std::chrono::milliseconds(original_timeout);

    while (std::chrono::steady_clock::now() < deadline) {
      if (closing_.load(std::memory_order_acquire)) {
        wait_args->timeout = original_timeout;
        wait_args->wait_result = 1;
        return 0;
      }
      wait_args->timeout = 0;
      int rc = send_ioctl(request, arg);
      wait_args->timeout = original_timeout;
      if (rc != 0)
        return rc;
      if (wait_args->wait_result != KFD_IOC_WAIT_RESULT_TIMEOUT)
        return 0;
      // Block on shutdown_efd_ instead of sleeping. poll() returns
      // immediately if close() has written to the eventfd, or after
      // the poll timeout (5ms) for the next daemon poll iteration.
      // rpc_mutex_ is NOT held — other threads can send ioctls.
      if (shutdown_efd_ >= 0) {
        struct pollfd pfd = {shutdown_efd_, POLLIN, 0};
        struct timespec ts = {0, 5'000'000};
        syscall(SYS_ppoll, &pfd, 1, &ts, nullptr, 0);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
    wait_args->wait_result = 1;
    return 0;
  }

  return send_ioctl(request, arg);
}

int RemoteDriver::send_ioctl(unsigned long request, void *arg) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  // The stream is misaligned past a rejected reply; every later header would be
  // parsed out of stale bytes. Fail closed instead of returning bogus results.
  // Tested under rpc_mutex_, not before it: a thread that read the flag first
  // and then blocked on the lock would go on to drive a full round trip against
  // a connection another thread poisoned while it waited.
  if (protocol_failed_.load(std::memory_order_acquire))
    return -EPROTO;

  size_t arg_size = 0;
  if (!validate_ioctl_arg_size(request, arg, arg_size))
    return -EINVAL;

  // Save original embedded pointers before serialization. The daemon rewrites
  // these to point at its own buffer; we must restore the client-side originals
  // before copying inline response data back.
  uint64_t saved_events_ptr = 0;
  uint64_t saved_apertures_ptr = 0;
  uint64_t saved_device_ids_ptr = 0;
  uint64_t saved_dbg_rinfo_ptr = 0;
  uint32_t saved_dbg_rinfo_size = 0;
  uint32_t saved_dbg_op = 0;
  uint64_t saved_dbg_snapshot_ptr = 0;
  // The response overwrites the count/entry_size pair with the daemon's
  // outputs, so the request's own count and stride have to be kept to reproduce
  // the driver's strided write on copy-back. Together they are also the
  // caller's declared buffer capacity.
  uint32_t saved_dbg_snapshot_count = 0;
  uint32_t saved_dbg_snapshot_stride = 0;
  // The stride actually put on the wire, which is the caller's clamped to the
  // struct the daemon fills. Entries arrive packed at this pitch and are
  // scattered out at the caller's, so the two cannot be conflated.
  uint32_t saved_dbg_snapshot_wire_stride = 0;
  // SUSPEND/RESUME_QUEUES carry an inbound array of queue ids the daemon
  // rewrites to its own tail, and report per-queue status back through it.
  uint64_t saved_dbg_queue_array_ptr = 0;
  uint32_t saved_dbg_queue_count = 0;
  if (has_embedded_pointers(request)) {
    switch (request) {
    case AMDKFD_IOC_WAIT_EVENTS:
      saved_events_ptr = static_cast<kfd_ioctl_wait_events_args *>(arg)->events_ptr;
      break;
    case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
      saved_apertures_ptr = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg)
                                ->kfd_process_device_apertures_ptr;
      break;
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
      saved_device_ids_ptr =
          static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg)->device_ids_array_ptr;
      break;
    case AMDKFD_IOC_DBG_TRAP: {
      auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
      saved_dbg_op = dbg->op;
      switch (dbg->op) {
      case KFD_IOC_DBG_TRAP_ENABLE:
        saved_dbg_rinfo_ptr = dbg->enable.rinfo_ptr;
        saved_dbg_rinfo_size = dbg->enable.rinfo_size;
        break;
      case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
        // Reject a missing output buffer here rather than on the wire. The
        // driver validates it before writing anything (-EINVAL, outputs
        // untouched -- kfd_dbg_trap_device_snapshot(), reproduced by
        // SimulatedKfd::debug_device_snapshot), but the daemon cannot: it
        // rewrites snapshot_buf_ptr to its own inline tail before replaying the
        // ioctl (rj_vm.cpp, reconstruct_embedded_pointers), so the driver never
        // sees the null and answers a malformed request with success and a
        // device total. Failing client-side is what keeps the two transports
        // returning the same verdict.
        if (dbg->device_snapshot.snapshot_buf_ptr == 0)
          return -EINVAL;
        saved_dbg_snapshot_ptr = dbg->device_snapshot.snapshot_buf_ptr;
        saved_dbg_snapshot_count = dbg->device_snapshot.num_devices;
        saved_dbg_snapshot_stride = dbg->device_snapshot.entry_size;
        break;
      case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
        // Deliberately no null-buffer rejection here, unlike the device op
        // above. The local path answers a null buffer with success when the
        // process has no queues to report and only -EFAULT when it has some
        // (SimulatedKfd::debug_queue_snapshot), and the client cannot know that
        // count before it asks. The verdict is reconstructed on copy-back
        // instead, from the queue total the daemon reports.
        saved_dbg_snapshot_ptr = dbg->queue_snapshot.snapshot_buf_ptr;
        saved_dbg_snapshot_count = dbg->queue_snapshot.num_queues;
        saved_dbg_snapshot_stride = dbg->queue_snapshot.entry_size;
        break;
      case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
        saved_dbg_snapshot_ptr = dbg->query_exception_info.info_ptr;
        saved_dbg_snapshot_count = 1;
        saved_dbg_snapshot_stride = dbg->query_exception_info.info_size;
        break;
      case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
        saved_dbg_queue_array_ptr = dbg->suspend_queues.queue_array_ptr;
        saved_dbg_queue_count = dbg->suspend_queues.num_queues;
        break;
      case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
        saved_dbg_queue_array_ptr = dbg->resume_queues.queue_array_ptr;
        saved_dbg_queue_count = dbg->resume_queues.num_queues;
        break;
      default:
        break;
      }
      break;
    }
    default:
      break;
    }
  }

  constexpr size_t prefix = sizeof(RpcHeader) + sizeof(RpcIoctlRequest);
  std::vector<uint8_t> buf(prefix + arg_size);

  std::memcpy(buf.data() + prefix, arg, arg_size);

  if (has_embedded_pointers(request)) {
    auto *args_base = buf.data() + prefix;
    switch (request) {
    case AMDKFD_IOC_WAIT_EVENTS: {
      auto *wait_args = reinterpret_cast<kfd_ioctl_wait_events_args *>(args_base);
      const auto *events = reinterpret_cast<const void *>(wait_args->events_ptr);
      const size_t inline_size = wait_args->num_events * sizeof(kfd_event_data);
      size_t inline_offset = buf.size();
      buf.resize(inline_offset + inline_size);
      std::memcpy(buf.data() + inline_offset, events, inline_size);
      break;
    }
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
      auto *map_args = reinterpret_cast<kfd_ioctl_map_memory_to_gpu_args *>(args_base);
      const auto *device_ids = reinterpret_cast<const void *>(map_args->device_ids_array_ptr);
      const size_t inline_size = map_args->n_devices * sizeof(uint32_t);
      size_t inline_offset = buf.size();
      buf.resize(inline_offset + inline_size);
      std::memcpy(buf.data() + inline_offset, device_ids, inline_size);
      break;
    }
    case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
      auto *aperture_args = reinterpret_cast<kfd_ioctl_get_process_apertures_new_args *>(args_base);
      buf.resize(buf.size() + aperture_args->num_of_nodes * sizeof(kfd_process_device_apertures));
      break;
    }
    case AMDKFD_IOC_DBG_TRAP: {
      auto *dbg = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(args_base);
      size_t inline_size = 0;
      switch (dbg->op) {
      case KFD_IOC_DBG_TRAP_ENABLE:
        inline_size =
            std::min(static_cast<size_t>(dbg->enable.rinfo_size), sizeof(kfd_runtime_info));
        break;
      case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
      case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT: {
        // Rewrite the transmitted count, not just the tail length: the daemon
        // recomputes count * entry_size and disconnects any client whose
        // payload does not match it exactly (validate_ioctl_payload). Only the
        // serialized copy is edited — the caller's args keep the original
        // request until the reply overwrites them with the true total.
        //
        // Transmit a compact stride rather than the caller's. entry_size(IN) is
        // documented as the caller's buffer stride and is allowed to exceed the
        // current struct (kfd_ioctl.h), so a caller with wide padding between
        // entries would otherwise reserve -- and make the daemon return -- that
        // padding as wire bytes. Worse, a stride at or above the payload budget
        // left room for no entries at all, and the count-only request that
        // produced would come back as success over an untouched buffer. The
        // daemon fills at this stride; the copy-back scatters each entry into
        // the caller's own wider slot.
        const SnapshotFields snap = snapshot_fields(dbg, dbg->op);
        const uint32_t wire_stride = std::min(*snap.entry_size, snap.struct_size);
        const uint32_t requested = *snap.count;
        const uint32_t entries = clamp_snapshot_entries(requested, wire_stride, arg_size);
        // A single compact entry not fitting would mean kMaxPayloadBytes is
        // smaller than one struct; fail rather than degrade to the count probe.
        if (entries == 0 && requested != 0 && wire_stride != 0)
          return -ENOMEM;
        *snap.count = entries;
        *snap.entry_size = wire_stride;
        saved_dbg_snapshot_wire_stride = wire_stride;
        inline_size = static_cast<size_t>(entries) * wire_stride;
        break;
      }
      case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
        inline_size = dbg->query_exception_info.info_size;
        break;
      case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
        inline_size = static_cast<size_t>(dbg->suspend_queues.num_queues) * sizeof(uint32_t);
        break;
      case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
        inline_size = static_cast<size_t>(dbg->resume_queues.num_queues) * sizeof(uint32_t);
        break;
      default:
        break;
      }
      // QUERY_EXCEPTION_INFO takes info_size straight from the caller, and
      // SUSPEND/RESUME derive theirs from num_queues; none of the three is
      // clamped the way snapshot requests are. Bound them by the same protocol
      // ceiling the daemon enforces, before resize() and the memcpy above run:
      // UINT32_MAX alone still admits a multi-gigabyte allocation, a
      // std::bad_alloc, or a huge read from the caller's buffer, all of which
      // happen before the daemon ever sees the frame and rejects it.
      //
      // Fail rather than clamp. The daemon recomputes the expected tail from
      // the echoed args, so a clamped tail with an unclamped count is a
      // mismatch and drops the connection.
      const size_t payload_so_far = buf.size() - sizeof(RpcHeader);
      if (payload_so_far >= kMaxPayloadBytes || inline_size > kMaxPayloadBytes - payload_so_far)
        return -E2BIG;
      if (inline_size > 0) {
        const size_t inline_offset = buf.size();
        buf.resize(buf.size() + inline_size);
        if (saved_dbg_queue_array_ptr != 0)
          std::memcpy(buf.data() + inline_offset,
                      reinterpret_cast<const void *>(saved_dbg_queue_array_ptr), inline_size);
      }
      break;
    }
    default:
      break;
    }
  }

  auto *hdr = reinterpret_cast<RpcHeader *>(buf.data());
  hdr->opcode = RPC_IOCTL;
  hdr->request_id = next_id_++;
  hdr->payload_bytes = static_cast<uint32_t>(buf.size() - sizeof(RpcHeader));
  hdr->result = 0;

  auto *ireq = reinterpret_cast<RpcIoctlRequest *>(buf.data() + sizeof(RpcHeader));
  ireq->ioctl_cmd = static_cast<uint32_t>(request);
  ireq->args_bytes = static_cast<uint32_t>(buf.size() - prefix);

  // For DBG_TRAP ENABLE, hand the debugger's notifier pipe write-end to the
  // daemon as an SCM_RIGHTS fd. The daemon substitutes it into the ioctl's
  // dbg_fd so the driver can wake the debugger when a wave stops — the same fd
  // the real kernel would receive through the ioctl. KFD_INVALID_FD (0xffffffff)
  // casts to -1 and is not sent.
  int send_fds[3] = {-1, -1, -1};
  size_t num_send_fds = 0;
  // Owned only for the duration of the send: SCM_RIGHTS installs the daemon's
  // own copies, so ours are released on every path out of here, including the
  // early returns below.
  //
  // These remain util::UniqueHandle (closing via ::close, i.e. through the
  // interposer's own close hook) because RemoteDriver descriptors are the daemon
  // client's own and the interposer's close hook classifies them as untracked and
  // passes them through. Driver-owned fds on the LOCAL path use UniqueDriverFd
  // instead; see PassthroughFdTraits.
  util::UniqueHandle target_mem_fd;
  util::UniqueHandle target_proc_fd;
  if (request == AMDKFD_IOC_DBG_TRAP) {
    auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
    if (dbg->op == KFD_IOC_DBG_TRAP_ENABLE && static_cast<int>(dbg->enable.dbg_fd) >= 0) {
      if (::fcntl(static_cast<int>(dbg->enable.dbg_fd), F_GETFD) < 0)
        return transport_errno();
      // The daemon cannot reach the debuggee's address space on its own: it is
      // not the ptrace parent, so process_vm_readv/writev are refused. We are,
      // so open the target's memory here and transfer the authorization along
      // with the notifier. The directory fd pins the identity, so the daemon
      // can tell a target that exited from one whose pid was reused.
      const std::string proc_path = std::format("/proc/{}", dbg->pid);
      target_proc_fd =
          util::UniqueHandle(::open(proc_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
      if (!target_proc_fd)
        return transport_errno();
      target_mem_fd = util::UniqueHandle(::openat(target_proc_fd.get(), "mem", O_RDWR | O_CLOEXEC));
      if (!target_mem_fd)
        return transport_errno();
      send_fds[num_send_fds++] = static_cast<int>(dbg->enable.dbg_fd);
      send_fds[num_send_fds++] = target_mem_fd.get();
      send_fds[num_send_fds++] = target_proc_fd.get();
    }
  }
  // A send that fails without putting a byte on the wire is recoverable: the
  // daemon never saw a frame, so the stream is still aligned and the caller just
  // gets the transport errno. A send that stops part way through is not — the
  // daemon is left waiting on the rest of a frame that will never arrive, and
  // will parse our next request as its tail. Those two have to be told apart,
  // hence the byte counts.
  if (num_send_fds > 0) {
    // sendmsg() on a stream socket may accept only part of the buffer; the
    // ancillary fds ride on the first byte, so a short send is a truncated frame
    // with the descriptors already handed over.
    const auto sent = rpc_send_msg(sock_, buf.data(), buf.size(), send_fds, num_send_fds);
    if (sent > 0 && static_cast<size_t>(sent) != buf.size())
      return poison_stream();
    if (sent <= 0) {
      // Preserve the transport errno — e.g. EBADF when the client handed us a
      // closed notifier fd for SCM_RIGHTS — instead of a bare -1, which the
      // interposer would surface as EPERM (-EPERM == -1).
      return transport_errno();
    }
  } else {
    size_t sent = 0;
    if (!rpc_send_exact(sock_, buf.data(), buf.size(), &sent)) {
      if (sent > 0)
        return poison_stream();
      return transport_errno();
    }
  }

  // Receive response — may include a memfd via SCM_RIGHTS for ALLOC_MEMORY.
  uint8_t resp_header_buf[sizeof(RpcHeader)];
  int received_fds[1] = {-1};
  size_t num_fds = 1;
  auto bytes =
      rpc_recv_msg(sock_, resp_header_buf, sizeof(resp_header_buf), received_fds, &num_fds);

  // Every reply byte we decline to read leaves the stream misaligned: the next
  // call would parse its header out of this one's remains, silently returning a
  // bogus result and copying stale bytes into caller memory. Kill the connection
  // instead of returning a recoverable-looking error. Any descriptor that rode
  // in on the doomed reply is dropped here — nothing downstream runs to adopt
  // it.
  auto poison_reply = [&] {
    if (num_fds > 0 && received_fds[0] >= 0)
      syscall(SYS_close, received_fds[0]);
    return poison_stream();
  };

  // The request is already on the wire, so a reply we cannot take delivery of is
  // not a clean failure: EINTR aborts the receive with the daemon's answer still
  // queued, and the next call would read that answer as its own. Terminal, even
  // though the socket itself may still be healthy.
  if (bytes <= 0)
    return poison_reply();
  // rpc_recv_msg asks for MSG_WAITALL but can still come back short (a signal
  // interrupts the wait), which would leave the rest of the header in the
  // socket and decode result/payload_bytes out of uninitialized stack.
  if (static_cast<size_t>(bytes) != sizeof(RpcHeader))
    return poison_reply();

  auto *resp = reinterpret_cast<RpcHeader *>(resp_header_buf);

  // kMaxPayloadBytes bounds both directions of the protocol; the daemon
  // enforces it on receive. Enforce it here too so a desynced or corrupted
  // reply header cannot make an interposed ioctl() attempt a multi-GiB
  // allocation and throw std::bad_alloc out of a C entry point.
  if (resp->payload_bytes > kMaxPayloadBytes)
    return poison_reply();

  if (resp->payload_bytes > 0) {
    std::vector<uint8_t> payload(resp->payload_bytes);
    // A partial read consumes part of the declared payload and abandons the
    // rest, which is the same misalignment as refusing it outright.
    if (!rpc_recv_exact(sock_, payload.data(), resp->payload_bytes))
      return poison_reply();

    size_t copy_size = std::min(arg_size, static_cast<size_t>(resp->payload_bytes));
    std::memcpy(arg, payload.data(), copy_size);

    // Restore original client-side pointers that were overwritten by the
    // daemon's response (daemon rewrites them to point at its own buffer).
    if (has_embedded_pointers(request)) {
      switch (request) {
      case AMDKFD_IOC_WAIT_EVENTS:
        static_cast<kfd_ioctl_wait_events_args *>(arg)->events_ptr = saved_events_ptr;
        break;
      case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
        static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg)
            ->kfd_process_device_apertures_ptr = saved_apertures_ptr;
        break;
      case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
      case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
        static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg)->device_ids_array_ptr =
            saved_device_ids_ptr;
        break;
      case AMDKFD_IOC_DBG_TRAP: {
        auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
        switch (saved_dbg_op) {
        case KFD_IOC_DBG_TRAP_ENABLE:
          dbg->enable.rinfo_ptr = saved_dbg_rinfo_ptr;
          break;
        case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
        case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT: {
          const SnapshotFields snap = snapshot_fields(dbg, saved_dbg_op);
          *snap.buf_ptr = saved_dbg_snapshot_ptr;
          // The count and entry_size are inputs the request rewrote to the
          // compact wire values, and the daemon echoes back whatever it was
          // sent. On success that echo is the answer (the true total and the
          // bytes filled), but on failure it would hand the caller our internal
          // values instead of leaving the inputs alone: libhsakmt's UINT32_MAX
          // probe comes back as the 4096 clamp and its stride as the struct
          // size. The local path validates before writing any output
          // (SimulatedKfd::debug_device_snapshot), so a failed op leaves both
          // fields exactly as the caller set them; match that.
          if (resp->result != 0) {
            *snap.count = saved_dbg_snapshot_count;
            *snap.entry_size = saved_dbg_snapshot_stride;
            break;
          }
          // The queue op does not reject a null output buffer on the request
          // path, because whether one is required depends on the queue total
          // only the driver knows. Now that the total is back, apply the local
          // path's verdict: entries to report and nowhere to put them is
          // -EFAULT, no entries is success (SimulatedKfd::debug_queue_snapshot).
          if (saved_dbg_op == KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT && saved_dbg_snapshot_ptr == 0 &&
              saved_dbg_snapshot_count > 0 && *snap.count > 0)
            resp->result = -EFAULT;
          break;
        }
        case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
          dbg->query_exception_info.info_ptr = saved_dbg_snapshot_ptr;
          break;
        case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
          dbg->suspend_queues.queue_array_ptr = saved_dbg_queue_array_ptr;
          break;
        case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
          dbg->resume_queues.queue_array_ptr = saved_dbg_queue_array_ptr;
          break;
        default:
          break;
        }
        break;
      }
      default:
        break;
      }
    }

    if (has_embedded_pointers(request) && resp->payload_bytes > arg_size) {
      size_t extra = resp->payload_bytes - arg_size;
      switch (request) {
      case AMDKFD_IOC_WAIT_EVENTS: {
        auto *wait_args = static_cast<kfd_ioctl_wait_events_args *>(arg);
        std::memcpy(reinterpret_cast<void *>(wait_args->events_ptr), payload.data() + arg_size,
                    std::min(wait_args->num_events * sizeof(kfd_event_data), extra));
        break;
      }
      case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
        auto *aperture_args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
        std::memcpy(
            reinterpret_cast<void *>(aperture_args->kfd_process_device_apertures_ptr),
            payload.data() + arg_size,
            std::min(aperture_args->num_of_nodes * sizeof(kfd_process_device_apertures), extra));
        break;
      }
      case AMDKFD_IOC_DBG_TRAP: {
        auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
        void *dst = nullptr;
        size_t copy_len = 0;
        switch (saved_dbg_op) {
        case KFD_IOC_DBG_TRAP_ENABLE:
          // Only propagate runtime-info bytes on success; a failed op (e.g.
          // -EBADF from a rejected notifier fd) must not mutate caller memory
          // or dereference the saved output pointer, matching local mode.
          if (resp->result == 0) {
            dst = reinterpret_cast<void *>(saved_dbg_rinfo_ptr);
            copy_len = std::min(static_cast<size_t>(saved_dbg_rinfo_size),
                                std::min(static_cast<size_t>(dbg->enable.rinfo_size), extra));
          }
          break;
        case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
        case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT: {
          // Only propagate snapshot bytes on success; a failed op (e.g. -ENOSYS)
          // must not mutate caller memory or dereference the saved output
          // pointer.
          if (resp->result != 0)
            break;
          // The device op's request path rejects a null output buffer with
          // -EINVAL, and the queue op's reply path above turns a null with
          // entries to report into -EFAULT. A null that reaches here is
          // therefore a queue snapshot with nothing to report, which writes
          // nothing anyway — but keep the guard so the memcpy below can never
          // run through a null pointer if either check moves.
          if (saved_dbg_snapshot_ptr == 0)
            break;
          // Replay the driver's write pattern instead of bulk-copying the tail.
          // amdkfd fills min(count(IN), true total) entries, writing
          // entry_size(OUT) bytes at entry_size(IN) stride, and leaves the rest
          // of the caller's buffer alone — both the entries past the total and,
          // when the caller's stride is wider than the current struct, the
          // padding inside each entry. A bulk copy of the declared capacity
          // would instead zero all of that (the request tail is sent empty, so
          // the daemon fills only what it enumerates), diverging from what
          // SimulatedKfd writes on the local path. It would also trust
          // count(IN) as a real allocation size, which callers are entitled to
          // oversize.
          //
          // count(IN) is the caller's declared capacity in entries, so capping
          // the loop at it is the same bound the driver applies; the per-entry
          // `extra` test additionally refuses to read past the tail the daemon
          // actually sent.
          if (saved_dbg_snapshot_stride == 0 || saved_dbg_snapshot_wire_stride == 0)
            break;
          const SnapshotFields snap = snapshot_fields(dbg, saved_dbg_op);
          const uint32_t entries = std::min(saved_dbg_snapshot_count, *snap.count);
          // Two pitches: the tail is packed at the stride we transmitted, the
          // caller's buffer is laid out at the stride it declared.
          const size_t src_stride = saved_dbg_snapshot_wire_stride;
          const size_t dst_stride = saved_dbg_snapshot_stride;
          // entry_size(OUT) is the daemon's word, not ours. The local path
          // clamps it to the struct it actually fills, so apply the same bound
          // here: an inflated value would pull the neighbouring entries' bytes
          // into the padding the caller's wider stride leaves between entries.
          const size_t written = std::min<size_t>(std::min(*snap.entry_size, snap.struct_size),
                                                  std::min(src_stride, dst_stride));
          // A zero-width entry writes nothing, so the loop has nothing to do —
          // and the `src + written > extra` bound below degenerates into a bare
          // offset test that lets a daemon-reported entry_size(OUT) of 0 spin
          // the loop over the whole tail forming out-of-range destination
          // pointers (UB) for no effect.
          if (written == 0)
            break;
          auto *snapshot_out = reinterpret_cast<uint8_t *>(saved_dbg_snapshot_ptr);
          for (uint32_t i = 0; i < entries; ++i) {
            const size_t src = static_cast<size_t>(i) * src_stride;
            const size_t dst_offset = static_cast<size_t>(i) * dst_stride;
            if (src + written > extra)
              break;
            std::memcpy(snapshot_out + dst_offset, payload.data() + arg_size + src, written);
          }
          break;
        }
        case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
          if (resp->result == 0) {
            dst = reinterpret_cast<void *>(saved_dbg_snapshot_ptr);
            copy_len = std::min(static_cast<size_t>(saved_dbg_snapshot_stride), extra);
          }
          break;
        case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
        case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
          // KFD returns a non-negative queue count on success and annotates
          // every requested ID with ERROR/INVALID status bits in the same
          // caller-owned array.
          if (resp->result >= 0) {
            dst = reinterpret_cast<void *>(saved_dbg_queue_array_ptr);
            copy_len =
                std::min(static_cast<size_t>(saved_dbg_queue_count) * sizeof(uint32_t), extra);
          }
          break;
        default:
          break;
        }
        if (dst != nullptr && copy_len > 0)
          std::memcpy(dst, payload.data() + arg_size, copy_len);
        break;
      }
      default:
        break;
      }
    }
  }

  auto register_allocation = [&](uint64_t handle, uint64_t va_addr, uint64_t size, int memfd) {
    handle_memfds_[handle] = memfd;
    if (va_addr != 0 && size > 0)
      alloc_ranges_.push_back({va_addr, size, memfd});
  };

  auto promote_userptr = [&](uint64_t va_addr, uint64_t size, int memfd) {
    auto *va = reinterpret_cast<void *>(va_addr);
    auto length = static_cast<size_t>(size);

    [[maybe_unused]] auto ft_rc = ftruncate(memfd, static_cast<off_t>(length));
    fallocate(memfd, 0, 0, static_cast<off_t>(length));

    constexpr size_t page_size = 4096;
    size_t num_pages = (length + page_size - 1) / page_size;
    std::vector<uint8_t> resident(num_pages);
    auto mc_rc = syscall(SYS_mincore, va, length, resident.data());

    auto *temp =
        static_cast<uint8_t *>(safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, memfd, 0));
    if (temp != MAP_FAILED) {
      if (mc_rc == 0) {
        auto *src = static_cast<uint8_t *>(va);
        for (size_t i = 0; i < num_pages; ++i) {
          if (resident[i] & 1) {
            size_t off = i * page_size;
            size_t n = std::min(page_size, length - off);
            std::memcpy(temp + off, src + off, n);
          }
        }
      }
      syscall(SYS_munmap, temp, length);
    }

    auto *mapped = safe_mmap(va, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memfd, 0);
    if (mapped != MAP_FAILED)
      syscall(SYS_madvise, mapped, length, MADV_POPULATE_WRITE);
  };

  // Store memfd received from ALLOC_MEMORY for use in the subsequent mmap().
  // Also register the address range for anonymous MAP_FIXED interception.
  if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU && resp->result == 0) {
    auto *alloc_args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);
    if (num_fds > 0 && received_fds[0] >= 0) {
      register_allocation(alloc_args->handle, alloc_args->va_addr, alloc_args->size,
                          received_fds[0]);

      if (alloc_args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)
        promote_userptr(alloc_args->va_addr, alloc_args->size, received_fds[0]);
    }
  }

  if (request == AMDKFD_IOC_IPC_IMPORT_HANDLE && resp->result == 0) {
    auto *import_args = static_cast<kfd_ioctl_ipc_import_handle_args *>(arg);
    if (num_fds > 0 && received_fds[0] >= 0) {
      uint64_t size = 0;
      struct stat st {};
      if (fstat(received_fds[0], &st) == 0)
        size = static_cast<uint64_t>(st.st_size);
      register_allocation(import_args->handle, import_args->va_addr, size, received_fds[0]);
    }
  }

  if (request == AMDKFD_IOC_EXPORT_DMABUF && resp->result == 0) {
    auto *export_args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);
    if (num_fds > 0 && received_fds[0] >= 0)
      export_args->dmabuf_fd = received_fds[0];
  }

  if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU && resp->result == 0) {
    auto *free_args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
    if (auto it = handle_memfds_.find(free_args->handle); it != handle_memfds_.end()) {
      int freed_memfd = it->second;
      std::erase_if(alloc_ranges_,
                    [freed_memfd](const AllocRange &r) { return r.memfd == freed_memfd; });
      syscall(SYS_close, freed_memfd);
      handle_memfds_.erase(it);
    }
  }

  return resp->result;
}

void *RemoteDriver::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  // Same fail-closed contract as send_ioctl(), and for the same reason: a
  // poisoned stream would hand back some other exchange's result. Answering with
  // a local anonymous mapping instead would be worse than failing — the daemon
  // has no matching mapping, so the GPU would fault on memory the client
  // believes is shared. Tested under rpc_mutex_ so a thread that blocked on the
  // lock cannot proceed against a connection poisoned while it waited.
  if (protocol_failed_.load(std::memory_order_acquire)) {
    errno = EPROTO;
    return MAP_FAILED;
  }

  // Send the mmap RPC so the daemon creates its own mapping for GPU simulation.
  int memfd = -1;
  int rc = send_mmap(addr, length, prot, flags, offset, &memfd);
  if (rc != 0) {
    if (memfd >= 0)
      syscall(SYS_close, memfd);
    // send_mmap() reports a negative errno, but only its transport failures
    // leave errno set as a side effect; the poison paths and the daemon's own
    // result do not, so without this the caller would read whatever errno this
    // thread last happened to set. The call that TRIPS the poison comes through
    // here, not through the fail-fast check above, so this is the one that has
    // to make -EPROTO visible.
    //
    // Unconditional, with no sentinel to exempt: transport_errno() guarantees
    // every negative return is a real errno, so -1 means EPERM and is
    // translated like any other. A positive rc cannot come from the daemon's
    // reply (its result is 0 or negative), so treat it as the desync it is.
    errno = rc < 0 ? -rc : EPROTO;
    return MAP_FAILED;
  }

  // Resolve the memfd for this allocation: prefer the mmap response fd, fall
  // back to the stored ALLOC_MEMORY fd (same underlying file, different fd).
  int mapping_memfd = memfd;
  uint64_t type = static_cast<uint64_t>(offset) & (0x3ULL << 62);
  if (mapping_memfd < 0 && type == 0) {
    uint64_t handle = static_cast<uint64_t>(offset) >> 12;
    if (auto it = handle_memfds_.find(handle); it != handle_memfds_.end())
      mapping_memfd = it->second;
  }
  if (mapping_memfd >= 0) {
    [[maybe_unused]] auto ft_rc2 = ftruncate(mapping_memfd, static_cast<off_t>(length));

    // Pre-copy committed pages (code objects) from the existing anonymous
    // reservation into the memfd before MAP_FIXED replaces them. Uses a temp
    // mapping outside the GPUVM range for the copy target.
    if ((flags & MAP_FIXED) && addr != nullptr) {
      auto prot_rc = syscall(SYS_mprotect, addr, length, PROT_READ | PROT_WRITE);
      if (prot_rc == 0) {
        constexpr size_t page_size = 4096;
        size_t num_pages = (length + page_size - 1) / page_size;
        std::vector<uint8_t> page_resident(num_pages);
        auto mc_rc = syscall(SYS_mincore, addr, length, page_resident.data());
        auto *temp = static_cast<uint8_t *>(
            safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, mapping_memfd, 0));
        if (temp != MAP_FAILED) {
          if (mc_rc == 0) {
            auto *src = static_cast<uint8_t *>(addr);
            for (size_t i = 0; i < num_pages; ++i) {
              if (page_resident[i] & 1) {
                size_t off = i * page_size;
                size_t n = std::min(page_size, length - off);
                std::memcpy(temp + off, src + off, n);
              }
            }
          }
          syscall(SYS_munmap, temp, length);
        }
      }
    }

    // QEMU vhost-user pattern: per-allocation memfd with F_SEAL_SHRINK (set at
    // creation in alloc_memory_ioctl), MAP_SHARED|MAP_FIXED, then
    // MADV_POPULATE_WRITE to pre-fault pages. This surfaces any shmem ENOSPC
    // as errno rather than deferred SIGBUS on page fault.
    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    auto *mapped = safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, mapping_memfd, 0);
    if (mapped != MAP_FAILED)
      syscall(SYS_madvise, mapped, length, MADV_POPULATE_WRITE);

    if (memfd >= 0 && memfd != mapping_memfd)
      syscall(SYS_close, memfd);
    return mapped;
  }

  // No memfd — anonymous fallback for doorbell/event pages.
  if (memfd >= 0)
    syscall(SYS_close, memfd);
  int mflags = MAP_ANONYMOUS | MAP_PRIVATE;
  if (flags & MAP_FIXED)
    mflags |= MAP_FIXED;
  return safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, -1, 0);
}

int RemoteDriver::munmap(void *addr, size_t length) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  // Fail closed on a poisoned stream rather than unmapping on the strength of a
  // reply that belongs to some other exchange.
  if (protocol_failed_.load(std::memory_order_acquire))
    return -EPROTO;

  RpcMunmapRequest req = {};
  req.addr = reinterpret_cast<uint64_t>(addr);
  req.length = length;

  RpcHeader hdr = {};
  hdr.opcode = RPC_MUNMAP;
  hdr.request_id = next_id_++;
  hdr.payload_bytes = sizeof(req);

  uint8_t send_buffer[sizeof(hdr) + sizeof(req)];
  std::memcpy(send_buffer, &hdr, sizeof(hdr));
  std::memcpy(send_buffer + sizeof(hdr), &req, sizeof(req));

  size_t sent = 0;
  if (!rpc_send_exact(sock_, send_buffer, sizeof(send_buffer), &sent)) {
    // A frame the daemon only half received is terminal; a frame that never
    // started is just a failed call on a still-aligned stream.
    if (sent > 0)
      return poison_stream();
    return transport_errno();
  }

  RpcHeader resp = {};
  // The request is on the wire, so any reply we do not consume whole — abandoned
  // outright or read part way — leaves the daemon's answer, or the tail of it,
  // waiting to be misparsed as the next call's header.
  if (!rpc_recv_exact(sock_, &resp, sizeof(resp)))
    return poison_stream();

  if (resp.result == 0)
    syscall(SYS_munmap, addr, length);

  return resp.result;
}

int RemoteDriver::send_mmap(void *addr, size_t length, int prot, int flags, off_t offset,
                            int *memfd_out) {
  // rpc_mutex_ is already held by the caller (mmap()), which has also already
  // rejected a poisoned stream.
  *memfd_out = -1;
  RpcMmapRequest req = {};
  req.addr = reinterpret_cast<uint64_t>(addr);
  req.length = length;
  req.prot = prot;
  req.flags = flags;
  req.offset = offset;

  RpcHeader hdr = {};
  hdr.opcode = RPC_MMAP;
  hdr.request_id = next_id_++;
  hdr.payload_bytes = sizeof(req);

  uint8_t send_buffer[sizeof(hdr) + sizeof(req)];
  std::memcpy(send_buffer, &hdr, sizeof(hdr));
  std::memcpy(send_buffer + sizeof(hdr), &req, sizeof(req));

  size_t sent = 0;
  if (!rpc_send_exact(sock_, send_buffer, sizeof(send_buffer), &sent)) {
    if (sent > 0)
      return poison_stream();
    return transport_errno();
  }

  uint8_t response_buffer[sizeof(RpcHeader) + sizeof(RpcMmapResponse)];
  int received_fds[1] = {-1};
  size_t num_fds = 1;
  auto bytes_received =
      rpc_recv_msg(sock_, response_buffer, sizeof(response_buffer), received_fds, &num_fds);

  // The daemon frames every mmap reply as header + RpcMmapResponse, so anything
  // shorter is a desynced stream — and reading result out of it would take the
  // value from whatever was on the stack, then hand the caller a mapping (or a
  // failure) decided by uninitialized memory. Any memfd that arrived on the
  // truncated reply is dropped: the caller sees rc != 0 with *memfd_out still
  // -1, so nothing downstream would ever close it.
  auto poison_reply = [&] {
    if (num_fds > 0 && received_fds[0] >= 0)
      syscall(SYS_close, received_fds[0]);
    return poison_stream();
  };
  if (bytes_received <= 0)
    return poison_reply();
  if (static_cast<size_t>(bytes_received) != sizeof(response_buffer))
    return poison_reply();

  auto *resp = reinterpret_cast<RpcHeader *>(response_buffer);
  *memfd_out = (num_fds > 0) ? received_fds[0] : -1;
  return resp->result;
}

} // namespace rocjitsu
