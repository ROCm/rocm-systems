/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * hip_capture_writer.cpp — Streaming event serialization for the HRR capture layer.
 *
 * Writes events.bin, blobs/, and manifest.json to the output directory.
 * Binary format is compatible with hrr_reader.h / hrr_replay.cpp.
 *
 * Crash resilience: events.bin is written through a raw file descriptor with a
 * small app-managed buffer (not buffered stdio). The writer checkpoints
 * (flush+fsync) every kCheckpointEvents events to bound how much a crash can
 * lose, and emergency_finalize can be called from CLR's crash callback as a
 * final best-effort flush. A clean shutdown appends an hrr_eof_record trailer;
 * its absence marks the archive as crash-truncated for the reader, which
 * recovers all complete records.
 *
 * Disk space: capture stops, and the archive is marked incomplete, before the
 * file system holding it falls below a reserve; see "Disk space" below.
 *
 * Thread-safety: write_event_raw() and write_blob() acquire the file mutex.
 * open() publishes a new archive under it. A forked child's first record opens
 * its archive under g_reopen_mu, which flush() and close() also take.
 *
 * Access: on POSIX the archive holds the process's host buffers, kernel
 * arguments and code objects, so every archive directory this writer creates is
 * 0700, every file is 0600, no file is opened through a symbolic link in its last
 * path component, and when the archive is opened the per-process directory must
 * be a real directory owned by the effective user. The directories above each
 * file are resolved again on every open.
 */

#include "hip_capture_writer.h"
#include "hip_capture.h"
#include "hip_capture_metadata.h"

#include "os/os.hpp"           // amd::Os::timeNanos()
#include "utils/debug.hpp"     // LogPrintfError, LogPrintfWarning, LogPrintfInfo

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <process.h>
#  include <sys/stat.h>
// Truncate: new archive. Append: resume into existing events.bin (must NOT use _O_TRUNC).
#  define HRR_OPEN(p)          _open((p), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_OPEN_APPEND(p)   _open((p), _O_RDWR | _O_CREAT | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_WRITE(fd,b,n)    _write((fd), (b), (unsigned)(n))
#  define HRR_CLOSE(fd)        _close((fd))
#  define HRR_FSYNC(fd)        _commit((fd))

using hrr_stat_t = struct _stat64;
static int hrr_stat_file(const char* path, hrr_stat_t* st) { return _stat64(path, st); }
static std::int64_t hrr_stat_size(const hrr_stat_t& st) { return st.st_size; }

static int hrr_ftruncate_fd(int fd, std::int64_t len) {
  return _chsize_s(fd, len) == 0 ? 0 : -1;
}

static std::int64_t hrr_seek_end(int fd) {
  return static_cast<std::int64_t>(_lseeki64(fd, 0, SEEK_END));
}

static inline uint64_t current_thread_id() {
  static thread_local uint64_t cached = static_cast<uint64_t>(GetCurrentThreadId());
  return cached;
}

static inline uint64_t current_process_id() {
  return static_cast<uint64_t>(_getpid());
}

static inline uint64_t current_parent_process_id() {
  return 0;
}
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/statvfs.h>
#  include <sys/syscall.h>
#  include <pthread.h>
#  define HRR_OPEN(p)        open_private_fd((p))
#  define HRR_WRITE(fd,b,n)  ::write((fd), (b), (n))
#  define HRR_CLOSE(fd)      ::close((fd))
#  define HRR_FSYNC(fd)      ::fsync((fd))

static int hrr_ftruncate_fd(int fd, std::int64_t len) {
  return ftruncate(fd, static_cast<off_t>(len));
}

static std::int64_t hrr_seek_end(int fd) {
  return static_cast<std::int64_t>(lseek(fd, 0, SEEK_END));
}

static inline uint64_t current_thread_id() {
  static thread_local uint64_t cached = static_cast<uint64_t>(syscall(SYS_gettid));
  return cached;
}

static inline uint64_t current_process_id() {
  return static_cast<uint64_t>(getpid());
}

static inline uint64_t current_parent_process_id() {
  return static_cast<uint64_t>(getppid());
}
#endif

namespace fs = std::filesystem;

namespace hrr_cap {
namespace writer {

// ---------------------------------------------------------------------------
// FNV-1a 128-bit hash (same algorithm as out-of-tree writer)
// ---------------------------------------------------------------------------

static Hash128 hash_buffer(const void* data, size_t len) {
  uint64_t h1 = 0xcbf29ce484222325ULL;
  uint64_t h2 = 0x100000001b3ULL;
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; i++) {
    h1 ^= p[i]; h1 *= 0x100000001b3ULL;
    h2 ^= p[i]; h2 *= 0xcbf29ce484222325ULL;
  }
  return {h1, h2};
}

static void hash_hex(Hash128 h, char buf[33]) {
  snprintf(buf, 33, "%016llx%016llx",
           static_cast<unsigned long long>(h.lo),
           static_cast<unsigned long long>(h.hi));
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

// Buffer must comfortably hold the largest single record. payload_length is a
// uint32_t (v4); in practice the largest record is a kernel-launch event whose
// serialized args/name/by-value structs are well under this. 256 KiB amortizes
// write() syscalls. buffer_append_locked() handles a record larger than kBufCap
// by flushing and writing it directly, so this is a performance bound, not a cap.
static constexpr size_t   kBufCap           = 256u * 1024u;
// Flush+fsync every N events to bound crash data loss.
static constexpr uint64_t kCheckpointEvents = 4096;
// Path buffers are filled at open() so the signal path never touches std::string.
static constexpr size_t   kPathMax          = 4096;
static constexpr size_t   kMetadataJsonMax  = 128u * 1024u;
static constexpr size_t   kEmergencyManifestMax = kMetadataJsonMax + 1024u;

// Lock order: g_reopen_mu, then g_file_mu, then g_blob_mu, then
// g_unreplayable_mu. atfork_prepare is the only path that holds them all.
static std::mutex   g_file_mu;
static int          g_events_fd = -1;
// g_base_dir is the archive path requested via HIP_HRR_CAPTURE_OUTPUT.
// g_output_dir is the per-process archive directory this process writes to:
// g_base_dir/pid-<pid>/.
static std::string  g_base_dir;
static std::string  g_output_dir;
static char         g_manifest_path[kPathMax] = {0};
static uint64_t     g_pid = 0;
static uint64_t     g_parent_pid = 0;
static char         g_metadata_json[kMetadataJsonMax] = {0};
static size_t       g_metadata_json_len = 0;

// App-managed write buffer for events.bin (protected by g_file_mu).
static uint8_t  g_buf[kBufCap];
static size_t   g_buf_len            = 0;
static uint64_t g_events_since_ckpt  = 0;
// Set once flush() finalizes events.bin, with or without the trailer, under
// g_blob_mu as well as g_file_mu. No record is appended and no blob or code
// object claimed after it: the trailer must stay the last record, and the
// manifest already holds the event count. Atomic because emergency_finalize()
// and atfork_child() use it without taking either.
static std::atomic<bool> g_events_finalized{false};

// Set when an event could not be serialized losslessly and had to be dropped
// (e.g. an oversized kernel launch). A capture with this flag set is finalized
// WITHOUT the clean-shutdown trailer and with manifest "complete": false, so the
// reader treats it like a truncated archive rather than a faithful capture.
static std::atomic<bool> g_capture_incomplete{false};

// Crash-callback guard over g_buf / g_buf_len, raised by writer threads while
// they mutate the buffer. If the crash interrupts a writer mid-lock,
// emergency_finalize must not poke g_file_mu's futex from the handler. Writers
// still hold g_file_mu for thread<->thread exclusion; the flag is purely the
// crash-callback <-> writer coordination point.
static std::atomic_flag g_buf_busy = ATOMIC_FLAG_INIT;

// Raise g_buf_busy for a thread that already holds g_file_mu. Writers are
// serialized by the mutex, so the flag can only be up already because the crash
// callback is flushing g_buf; wait for it rather than mutating under it. The
// callback holds the flag only for that flush and never takes g_file_mu, which
// is what bounds this wait.
static void claim_buf_locked() {
  while (g_buf_busy.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
}

// RAII for writer threads: take the thread<->thread mutex AND raise g_buf_busy so
// the crash callback can tell a g_buf mutation is in flight. Member order
// matters: the mutex locks first and unlocks last, with the busy window nested
// strictly inside it.
struct BufWriteGuard {
  std::lock_guard<std::mutex> lk_;
  BufWriteGuard() : lk_(g_file_mu) { claim_buf_locked(); }
  ~BufWriteGuard() { g_buf_busy.clear(std::memory_order_release); }
};

// emergency_finalize builds the crash manifest here rather than on the stack:
// the crash handler can run on a thread with little stack left.
static std::atomic_flag g_emergency_manifest_busy = ATOMIC_FLAG_INIT;
static char g_emergency_manifest_buf[kEmergencyManifestMax];

static std::atomic<uint64_t> g_seq_id{0};
static std::atomic<uint64_t> g_event_count{0};
static std::atomic<uint64_t> g_blob_count{0};

// Blob and code object writes claimed under g_blob_mu and not finished yet.
// flush() waits for them before it decides on the trailer, since one that fails
// leaves events naming a file that does not exist. flush() waits holding
// g_file_mu, so a claimed write must finish without taking it.
static uint64_t                g_blob_writes_in_flight = 0;  // under g_blob_mu
static std::condition_variable g_blob_writes_done;

// Set in a forked child. POSIX allows the child of a multithreaded process only
// async-signal-safe calls until it execs, and open() is far from that, so the
// child's archive is opened by its first record, blob or code object rather
// than in atfork_child. A child that only execs or exits opens none.
static std::atomic<bool> g_reopen_after_fork{false};
static std::mutex        g_reopen_mu;

// In-memory set of blob hex keys already written to disk.
// Eliminates the fs::exists() stat syscall on repeated blobs (common for weight tensors).
// Protected by g_blob_mu (separate from g_file_mu to avoid head-of-line blocking).
// flush() takes g_blob_mu while it holds g_file_mu, so never take them the other way round.
// "co:" prefix for code objects matches the playback-side load_code_object key convention.
static std::mutex                      g_blob_mu;
static std::unordered_set<std::string> g_written_blobs;
#ifndef _WIN32
// blobs/<xx> prefixes already checked with claim_private_dir (xx is two hex chars).
// Atomic because write_blob runs on many threads without a lock held.
static std::atomic<bool> g_blob_prefix_claimed[256];
#endif
// Keys a thread is reserving space for or writing, also under g_blob_mu. Another
// caller with the same bytes waits for the claim to end, so the reserve is
// charged once per file, not once per caller.
static std::unordered_set<std::string> g_blob_claims;

// APIs recorded in this archive that replay cannot reproduce (note_unreplayable).
// Listed in manifest.json so the gap is a property of the archive rather than
// something only visible in a replay log.
static std::mutex                      g_unreplayable_mu;
// api -> distinct reasons it was noted for. One API can be unreplayable for
// several reasons (e.g. a different truncated argument on different calls),
// and each is worth reporting.
static std::map<std::string, std::set<std::string>> g_unreplayable_apis;

// Notes from note_unreplayable() that wait for the event the calling thread
// writes next. Plain pointers and a count, so nothing here has a destructor: a
// shim can still run on the main thread once its thread_local objects are gone.
// A generated shim stages at most five.
static constexpr size_t kMaxStagedNotes = 8;
struct StagedNote {
  const char* api;
  const char* reason;
};
static thread_local StagedNote t_staged_notes[kMaxStagedNotes];
static thread_local size_t     t_staged_count = 0;

// ---------------------------------------------------------------------------
// Low-level fd helpers
// ---------------------------------------------------------------------------

// Write the entire buffer, retrying short writes. Async-signal-safe: uses only
// write(). Returns true if all bytes were written.
static bool write_all_fd(int fd, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t off = 0;
  while (off < len) {
    auto n = HRR_WRITE(fd, p + off, len - off);
    if (n <= 0) {
#ifndef _WIN32
      if (n < 0 && errno == EINTR) continue;
#endif
      return false;
    }
    off += static_cast<size_t>(n);
  }
  return true;
}

// Drain the app buffer to the events fd. Caller must hold g_file_mu (or be the
// crash callback that has claimed g_buf_busy via test_and_set). Does not fsync.
static void flush_buffer_locked() {
  if (g_events_fd < 0 || g_buf_len == 0) { g_buf_len = 0; return; }
  write_all_fd(g_events_fd, g_buf, g_buf_len);
  g_buf_len = 0;
}

// Append `len` bytes of one complete record to the buffer, flushing first if it
// would not fit. A single record larger than the buffer (possible now that
// payload_length is a uint32_t — e.g. a kernel launch with very large by-value
// args) is flushed-then-written directly so it never overruns g_buf. Caller must
// hold g_file_mu.
static void buffer_append_locked(const void* data, size_t len) {
  if (g_buf_len + len > kBufCap) flush_buffer_locked();
  if (len > kBufCap) {
    // Oversized record: buffer is now empty (flushed above); write it straight
    // through rather than memcpy'ing past the end of g_buf.
    if (g_events_fd >= 0) write_all_fd(g_events_fd, data, len);
    return;
  }
  memcpy(g_buf + g_buf_len, data, len);
  g_buf_len += len;
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

// Create `path` and any missing parents: 0700 from the base directory down, and
// 0777 minus the umask above it. `path` must lie under g_base_dir.
// Never throws: this runs inside hip::init, the API shims and atexit, where an
// exception ends the process.
static bool ensure_dir(const std::string& path) {
  if (path.empty()) {
    errno = ENOENT;
    return false;
  }
#ifdef _WIN32
  std::error_code ec;
  fs::create_directories(path, ec);
  if (ec) {
    const std::error_condition cond = ec.default_error_condition();
    errno = cond.category() == std::generic_category() ? cond.value() : EIO;
    return false;
  }
  return true;
#else
  if (::mkdir(path.c_str(), 0700) == 0 || errno == EEXIST) return true;
  const size_t base_last = g_base_dir.find_last_not_of('/');
  const size_t base_len = base_last == std::string::npos ? 0 : base_last + 1;
  std::string cur;
  size_t pos = 0;
  do {
    pos = path.find('/', pos + 1);
    cur.assign(path, 0, pos);
    if (::mkdir(cur.c_str(), cur.size() < base_len ? 0777 : 0700) != 0 && errno != EEXIST) {
      const int err = errno;
      struct stat st{};
      if (::stat(cur.c_str(), &st) != 0) {
        errno = err;
        return false;
      }
      if (!S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        return false;
      }
    }
  } while (pos != std::string::npos);
  return true;
#endif
}

#ifndef _WIN32
static bool owned_by_euid(const struct stat& st) { return st.st_uid == geteuid(); }
#endif

// pid-<pid>, blobs/, code_objects/ and each blobs/<xx> must be a real directory
// owned by the effective user; anything else planted there (a symbolic link,
// another user's directory) is refused. An existing one is tightened to 0700.
static bool claim_private_dir(const std::string& path) {
#ifdef _WIN32
  (void)path;
  return true;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st{};
  int err = 0;
  if (::fstat(fd, &st) != 0) {
    err = errno;
  } else if (!owned_by_euid(st)) {
    err = EPERM;
  } else if ((st.st_mode & 07777) != 0700 && ::fchmod(fd, 0700) != 0) {
    err = errno;
  }
  ::close(fd);
  errno = err;
  return err == 0;
#endif
}

// Open this process's events.bin and report how many bytes it already holds.
// On POSIX an existing file is reused only if it is a regular file with a single
// link owned by the effective user, and a new one is created exclusively, so
// nothing planted at that path is ever truncated or appended to. O_NONBLOCK keeps
// a planted FIFO or device from blocking the open; it changes nothing for the
// regular file that is kept.
static int open_events_file(const std::string& path, std::int64_t* existing_size) {
  *existing_size = 0;
#ifdef _WIN32
  hrr_stat_t st{};
  if (hrr_stat_file(path.c_str(), &st) == 0 && hrr_stat_size(st) > 0) {
    *existing_size = hrr_stat_size(st);
    return HRR_OPEN_APPEND(path.c_str());
  }
  return HRR_OPEN(path.c_str());
#else
  int fd = ::open(path.c_str(), O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0 && errno == ENOENT)
    fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0600);
  if (fd < 0) return -1;
  struct stat st{};
  int err = 0;
  if (::fstat(fd, &st) != 0) {
    err = errno;
  } else if (!S_ISREG(st.st_mode) || st.st_nlink != 1 || !owned_by_euid(st)) {
    err = EPERM;
  } else if ((st.st_mode & 07777) != 0600 && ::fchmod(fd, 0600) != 0) {
    err = errno;
  }
  if (err != 0) {
    ::close(fd);
    errno = err;
    return -1;
  }
  *existing_size = static_cast<std::int64_t>(st.st_size);
  return fd;
#endif
}

#ifndef _WIN32
// Open a file inside the archive for writing and truncate it: 0600, never
// through a symbolic link or a hard link. Truncation happens only after the
// opened inode has been checked, so a planted hard link cannot empty a file
// outside the archive, and O_NONBLOCK keeps a planted FIFO from blocking the
// open. A refused inode fails with EPERM, so the caller's message names the
// reason. Async-signal-safe: emergency_finalize reaches it through HRR_OPEN.
static int open_private_fd(const char* path) {
  const int fd = ::open(path, O_WRONLY | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0600);
  if (fd < 0) return -1;
  struct stat st{};
  int err = 0;
  if (::fstat(fd, &st) != 0) {
    err = errno;
  } else if (!S_ISREG(st.st_mode) || st.st_nlink != 1 || !owned_by_euid(st)) {
    err = EPERM;
  } else if (((st.st_mode & 07777) != 0600 && ::fchmod(fd, 0600) != 0) || ::ftruncate(fd, 0) != 0) {
    err = errno;
  }
  if (err != 0) {
    ::close(fd);
    errno = err;
    return -1;
  }
  return fd;
}
#endif

// fopen(path, "w") for a file inside the archive, opened by open_private_fd on POSIX.
static FILE* fopen_private(const std::string& path) {
#ifdef _WIN32
  return fopen(path.c_str(), "w");
#else
  const int fd = open_private_fd(path.c_str());
  if (fd < 0) return nullptr;
  FILE* f = ::fdopen(fd, "w");
  if (!f) ::close(fd);
  return f;
#endif
}

// fopen(path, "r") for a file inside the archive. On POSIX only a regular file
// is read, never through a symbolic link, and a planted FIFO is refused rather
// than blocking the open. What is trusted on the way in passes the same checks
// as what is written: ours and a single link, or another user's counters and
// manifests would be taken for this archive's.
static FILE* fopen_read_regular(const std::string& path) {
#ifdef _WIN32
  return fopen(path.c_str(), "r");
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return nullptr;
  struct stat st{};
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_nlink != 1 ||
      !owned_by_euid(st)) {
    ::close(fd);
    return nullptr;
  }
  FILE* f = ::fdopen(fd, "r");
  if (!f) ::close(fd);
  return f;
#endif
}

static bool atomic_write_file(const std::string& path, const void* data, size_t len);

static bool is_json_object_fragment(const std::string& json) {
  const auto begin = json.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos || json[begin] != '{') return false;
  const auto end = json.find_last_not_of(" \t\r\n");
  return end != std::string::npos && json[end] == '}';
}

// ---------------------------------------------------------------------------
// manifest writers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Subprocess / resume helpers
//
// vLLM server mode runs GPU work in a spawned EngineCore child. Each HIP-owning
// process must append to the same events.bin instead of truncating it. On resume
// we scan the existing log (or trust writer_state.json from the last checkpoint),
// strip a clean-shutdown trailer if present, and continue sequence IDs.
// ---------------------------------------------------------------------------

struct ScanResult {
  uint64_t max_seq   = 0;
  uint64_t count     = 0;
  std::int64_t append_at = 0;
  bool     had_trailer = false;
  bool     torn_tail   = false;
};

static ScanResult scan_events_for_resume(FILE* f, std::int64_t file_size) {
  ScanResult r;
  const uint16_t hdr_size = static_cast<uint16_t>(sizeof(hrr_event_header));
  if (fseek(f, static_cast<long>(sizeof(hrr_file_header)), SEEK_SET) != 0)
    return r;

  r.append_at = sizeof(hrr_file_header);
  while (true) {
    long pos = ftell(f);
    if (pos < 0 || static_cast<std::int64_t>(pos) >= file_size) break;

    hrr_event_header h{};
    if (fread(&h, hdr_size, 1, f) != 1) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }

    // Valid trailer only if full hrr_eof_record + magic (same rule as hrr_reader).
    if (h.event_type == HRR_EOF_MARKER &&
        h.payload_length == static_cast<uint32_t>(sizeof(hrr_eof_record))) {
      uint64_t total_events = 0;
      uint32_t eof_magic    = 0;
      if (fread(&total_events, sizeof(total_events), 1, f) != 1 ||
          fread(&eof_magic, sizeof(eof_magic), 1, f) != 1) {
        r.torn_tail = true;
        r.append_at = pos;
        break;
      }
      if (eof_magic == HRR_EOF_MAGIC) {
        r.had_trailer = true;
        r.append_at = pos;
        break;
      }
      // Bogus EOF-shaped record: count it and skip past the bytes we read.
      r.max_seq = (h.sequence_id > r.max_seq) ? h.sequence_id : r.max_seq;
      r.count++;
      r.append_at = static_cast<std::int64_t>(ftell(f));
      continue;
    }

    if (h.payload_length < hdr_size) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }

    r.max_seq = (h.sequence_id > r.max_seq) ? h.sequence_id : r.max_seq;
    r.count++;

    long body = static_cast<long>(h.payload_length) - static_cast<long>(hdr_size);
    // fseek past EOF "succeeds" on most platforms (it only fails to read on the
    // next fread), so a record claiming e.g. 65535 bytes of body in a 100-byte
    // file would otherwise be accepted and leave append_at pointing past EOF —
    // corrupting the resuming capture's offset and sequence IDs. Validate that
    // the full record body actually fits in the file before trusting it.
    if (fseek(f, body, SEEK_CUR) != 0 ||
        ftell(f) < 0 ||
        static_cast<std::int64_t>(ftell(f)) > file_size) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }
    r.append_at = static_cast<std::int64_t>(ftell(f));
  }

  if (!r.had_trailer && !r.torn_tail)
    r.append_at = file_size;
  return r;
}

static bool try_load_writer_state(const std::string& path, std::int64_t file_size,
                                  uint64_t* next_seq, uint64_t* ev_count,
                                  uint64_t* bl_count) {
  FILE* f = fopen_read_regular(path);
  if (!f) return false;

  uint64_t ns = 0, ec = 0, bc = 0;
  long long stored_size = -1;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    unsigned long long u = 0;
    long long s = 0;
    if (sscanf(line, " \"next_seq\": %llu", &u) == 1) { ns = u; continue; }
    if (sscanf(line, " \"event_count\": %llu", &u) == 1) { ec = u; continue; }
    if (sscanf(line, " \"blob_count\": %llu", &u) == 1) { bc = u; continue; }
    if (sscanf(line, " \"events_file_size\": %lld", &s) == 1) { stored_size = s; continue; }
  }
  fclose(f);

  if (stored_size != file_size)
    return false;
  *next_seq = ns;
  *ev_count = ec;
  *bl_count = bc;
  return true;
}

static void save_writer_state_locked() {
  if (g_output_dir.empty() || g_events_fd < 0) return;
  std::int64_t sz = hrr_seek_end(g_events_fd);
  if (sz < 0) return;

  FILE* f = fopen_private(g_output_dir + "/writer_state.json");
  if (!f) return;
  fprintf(f,
          "{\n"
          "  \"next_seq\": %llu,\n"
          "  \"event_count\": %llu,\n"
          "  \"blob_count\": %llu,\n"
          "  \"events_file_size\": %lld\n"
          "}\n",
          static_cast<unsigned long long>(g_seq_id.load()),
          static_cast<unsigned long long>(g_event_count.load()),
          static_cast<unsigned long long>(g_blob_count.load()),
          static_cast<long long>(sz));
  fclose(f);
}

// A blob or code object found on resume counts as already written only if this
// user wrote it: a regular file with a single link owned by the effective user.
// The directories were claimed first, so nobody else can add one now, but a
// file another user planted before the claim, or a hard link to a file
// elsewhere, would otherwise be trusted and never rewritten. Left unindexed,
// it is replaced by the next atomic write of that hash, which renames over it.
static bool resumed_file_is_ours(const fs::path& p) {
#ifdef _WIN32
  std::error_code ec;
  return fs::symlink_status(p, ec).type() == fs::file_type::regular;
#else
  struct stat st{};
  return ::lstat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_nlink == 1 &&
         owned_by_euid(st);
#endif
}

static void index_existing_blobs_locked(const std::string& output_dir) {
  std::lock_guard<std::mutex> lk(g_blob_mu);
  g_written_blobs.clear();
  g_blob_claims.clear();  // as in the fresh-archive path of open()

  // error_code overloads throughout: a missing or unreadable directory only
  // means fewer blobs are known to exist, and a blob written twice is harmless.
  // Only a file resumed_file_is_ours() accepts counts; anything else is replaced.
  // A blobs/<xx> prefix is claimed before its files are trusted: one that fails
  // claim_private_dir contributes nothing, and write_blob refuses it later.
  std::error_code ec;
  const fs::path blobs_root = output_dir + "/blobs";
  for (fs::directory_iterator dit(blobs_root, ec), dend; !ec && dit != dend; dit.increment(ec)) {
    std::error_code entry_ec;
    if (dit->symlink_status(entry_ec).type() != fs::file_type::directory) continue;
#ifndef _WIN32
    const std::string name = dit->path().filename().string();
    const auto nibble = [](char c) -> int {
      return (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : -1;
    };
    if (name.size() != 2 || nibble(name[0]) < 0 || nibble(name[1]) < 0) continue;
    if (!claim_private_dir(dit->path().string())) continue;
    g_blob_prefix_claimed[(nibble(name[0]) << 4) | nibble(name[1])].store(
        true, std::memory_order_release);
#endif
    for (fs::directory_iterator it(dit->path(), entry_ec), end; !entry_ec && it != end;
         it.increment(entry_ec)) {
      if (it->path().extension() == ".blob" && resumed_file_is_ours(it->path()))
        g_written_blobs.insert(it->path().stem().string());
    }
  }

  ec.clear();
  const fs::path co_root = output_dir + "/code_objects";
  for (fs::directory_iterator it(co_root, ec), end; !ec && it != end; it.increment(ec)) {
    if (it->path().extension() == ".hsaco" && resumed_file_is_ours(it->path()))
      g_written_blobs.insert(std::string("co:") + it->path().stem().string());
  }
}

static void reopen_after_fork() {
  if (!g_reopen_after_fork.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lk(g_reopen_mu);
  if (!g_reopen_after_fork.load(std::memory_order_relaxed)) return;
  // From the *base* dir, so the child selects its own pid-<pid> sub-archive.
  // open() assigns g_base_dir, so pass it a copy.
  const std::string base = g_base_dir;
  // A child whose archive is refused runs on uncaptured, without the shims.
  if (!open(base.c_str())) hip_capture_uninstall();
  g_reopen_after_fork.store(false, std::memory_order_release);
}

#ifndef _WIN32
// The writer mutexes are held across fork(): a child must not inherit one that
// a thread which does not exist in the child had locked, because the child
// reopens its archive under them.
static void atfork_prepare() {
  g_reopen_mu.lock();
  g_file_mu.lock();
  claim_buf_locked();
  if (g_events_fd >= 0)
    flush_buffer_locked();
  g_buf_busy.clear(std::memory_order_release);
  g_blob_mu.lock();
  // The child's shutdown writes its manifest under this one.
  g_unreplayable_mu.lock();
}

static void atfork_parent() {
  g_unreplayable_mu.unlock();
  g_blob_mu.unlock();
  g_file_mu.unlock();
  g_reopen_mu.unlock();
}

static void atfork_child() {
  // The parent's blob writers are not in the child to finish their claims.
  g_blob_writes_in_flight = 0;
  g_unreplayable_mu.unlock();
  g_blob_mu.unlock();
  g_file_mu.unlock();
  g_reopen_mu.unlock();
  // A crash callback on another thread can raise g_buf_busy after
  // atfork_prepare clears it, or hold g_emergency_manifest_busy, which
  // atfork_prepare does not take. That thread does not exist in the child.
  g_buf_busy.clear(std::memory_order_release);
  g_emergency_manifest_busy.clear(std::memory_order_release);
  // Only async-signal-safe work from here: drop the parent's events fd and
  // forget its paths, so neither shutdown nor the crash path writes into the
  // parent's archive. reopen_after_fork() opens the child's.
  const bool parent_open = g_events_fd >= 0;
  const bool parent_finalized = g_events_finalized;
  if (parent_open) {
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
  g_buf_len = 0;
  g_events_since_ckpt = 0;
  g_events_finalized = false;
  g_output_dir.clear();
  g_manifest_path[0] = '\0';
  // The child's archive is a new one: an event the parent dropped is not
  // missing from it.
  g_capture_incomplete.store(false, std::memory_order_relaxed);
  // After flush() nothing would finalize a child's archive. The fd stays open
  // until close(), but the shutdown that runs both is on the parent's exiting
  // thread and no longer pending in the child. A parent that is itself a child
  // yet to open its archive passes its flag on unchanged.
  if (parent_open && !parent_finalized)
    g_reopen_after_fork.store(true, std::memory_order_relaxed);
}

static void install_atfork_handlers_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    pthread_atfork(atfork_prepare, atfork_parent, atfork_child);
  });
}
#endif

static void write_manifest_stdio(const char* output_dir, bool complete) {
  FILE* mf = fopen_private(std::string(output_dir) + "/manifest.json");
  if (!mf) return;
  fprintf(mf,
          "{\n"
          "  \"pid\": %llu,\n"
          "  \"parent_pid\": %llu,\n"
          "  \"complete\": %s,\n"
          "  \"event_count\": %llu,\n"
          "  \"blob_count\": %llu",
          static_cast<unsigned long long>(g_pid),
          static_cast<unsigned long long>(g_parent_pid),
          complete ? "true" : "false",
          static_cast<unsigned long long>(g_event_count.load()),
          static_cast<unsigned long long>(g_blob_count.load()));
  {
    std::lock_guard<std::mutex> lk(g_unreplayable_mu);
    if (!g_unreplayable_apis.empty()) {
      fprintf(mf, ",\n  \"unreplayable_apis\": {");
      bool first_api = true;
      for (const auto& [api, reasons] : g_unreplayable_apis) {
        fprintf(mf, "%s\n    \"%s\": [", first_api ? "" : ",",
                metadata::json_escape(api.c_str()).c_str());
        bool first_reason = true;
        for (const auto& reason : reasons) {
          fprintf(mf, "%s\"%s\"", first_reason ? "" : ", ",
                  metadata::json_escape(reason.c_str()).c_str());
          first_reason = false;
        }
        fprintf(mf, "]");
        first_api = false;
      }
      fprintf(mf, "\n  }");
    }
  }
  if (g_metadata_json_len > 0) {
    fprintf(mf, ",\n  \"metadata\": %.*s\n",
            static_cast<int>(g_metadata_json_len), g_metadata_json);
  } else {
    fprintf(mf, "\n");
  }
  fprintf(mf, "}\n");
  fclose(mf);
}

struct ProcessManifestEntry {
  uint64_t pid = 0;
  uint64_t parent_pid = 0;
  uint64_t event_count = 0;
  uint64_t blob_count = 0;
  bool complete = false;
};

static bool read_process_manifest(const std::string& path, ProcessManifestEntry* out) {
  FILE* f = fopen_read_regular(path);
  if (!f) return false;

  ProcessManifestEntry e{};
  bool saw_pid = false;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    unsigned long long u = 0;
    char b[8] = {};
    if (sscanf(line, " \"pid\": %llu", &u) == 1) {
      e.pid = static_cast<uint64_t>(u);
      saw_pid = true;
      continue;
    }
    if (sscanf(line, " \"parent_pid\": %llu", &u) == 1) {
      e.parent_pid = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"event_count\": %llu", &u) == 1) {
      e.event_count = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"blob_count\": %llu", &u) == 1) {
      e.blob_count = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"complete\": %7[^,\n ]", b) == 1) {
      e.complete = (strcmp(b, "true") == 0);
      continue;
    }
  }
  fclose(f);
  if (!saw_pid) return false;
  *out = e;
  return true;
}

static uint64_t derive_owner_pid(const std::vector<ProcessManifestEntry>& entries) {
  if (entries.empty()) return 0;
  for (const auto& candidate : entries) {
    for (const auto& child : entries) {
      if (child.parent_pid == candidate.pid)
        return candidate.pid;
    }
  }
  return entries.front().pid;
}

static void update_root_manifest() {
  if (g_base_dir.empty()) return;

  std::vector<ProcessManifestEntry> entries;
  std::error_code ec;
  for (fs::directory_iterator it(g_base_dir, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code entry_ec;
    if (it->symlink_status(entry_ec).type() != fs::file_type::directory) continue;
    const std::string name = it->path().filename().string();
    if (name.rfind("pid-", 0) != 0) continue;
    ProcessManifestEntry entry{};
    const std::string manifest_path = (it->path() / "manifest.json").string();
    if (read_process_manifest(manifest_path, &entry))
      entries.push_back(entry);
  }
  // A scan that stopped on an error would replace the root manifest with one
  // that leaves out the archives it did not reach.
  if (ec) return;

  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.pid < b.pid; });
  const uint64_t owner_pid = derive_owner_pid(entries);

  std::string json;
  json += "{\n";
  json += "  \"version\": 1,\n";
  json += "  \"capture_mode\": \"in-tree\",\n";
  json += "  \"owner_pid\": " + std::to_string(owner_pid) + ",\n";
  json += "  \"processes\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    json += "    { \"pid\": " + std::to_string(e.pid) +
            ", \"parent_pid\": " + std::to_string(e.parent_pid) +
            ", \"complete\": " + (e.complete ? "true" : "false") +
            ", \"event_count\": " + std::to_string(e.event_count) +
            ", \"blob_count\": " + std::to_string(e.blob_count) + " }";
    json += (i + 1 == entries.size()) ? "\n" : ",\n";
  }
  json += "  ]\n";
  json += "}\n";

  (void)atomic_write_file(g_base_dir + "/manifest.json", json.data(), json.size());
}

// ---------------------------------------------------------------------------
// Disk space
//
// Capture stops rather than fill the file system it writes to. It keeps free
// the smaller of 15% of that file system and 4 GiB, the default systemd-journald
// uses for SystemKeepFree. Free space is read when the archive opens, and again
// before any write that brings the bytes counted since the last check to
// g_space_check_bytes, so less than that is ever written unchecked. The interval
// is at most a quarter of the reserve, which bounds how far a capture can go
// into it.
// ---------------------------------------------------------------------------

static constexpr uint64_t kKeepFreeMax   = 4ull << 30;
static constexpr uint64_t kSpaceCheckMax = 64ull << 20;
static constexpr uint64_t kFileBlockDefault = 4096;
// Set by open(); g_keep_free stays 0 when free space cannot be read at all.
static uint64_t              g_keep_free = 0;
static uint64_t              g_space_check_bytes = kSpaceCheckMax;
// Every blob and code object is a file of its own and takes whole blocks, so they
// are counted in blocks of the archive's file system.
static std::atomic<uint64_t> g_file_block_bytes{kFileBlockDefault};
// Events and blobs both count here. Protected by g_file_mu.
static uint64_t g_bytes_since_space_check = 0;
// Bytes already accepted by reserve_space and not yet finished writing. Concurrent
// writers all count against the free-space check until they release.
static std::atomic<uint64_t> g_bytes_reserved{0};
static std::atomic<bool>     g_out_of_space{false};

static bool fs_space(const std::string& dir, uint64_t* avail, uint64_t* total,
                     uint64_t* block = nullptr) {
  uint64_t bs = 0;
#ifdef _WIN32
  ULARGE_INTEGER a{}, t{}, f{};
  // A UNC path is only accepted with a trailing backslash.
  const std::string d = dir.empty() || dir.back() == '\\' ? dir : dir + '\\';
  if (!GetDiskFreeSpaceExA(d.c_str(), &a, &t, &f)) return false;
  *avail = a.QuadPart;
  *total = t.QuadPart;
  // Files take whole clusters, which can be 64 KiB or more on NTFS and ReFS.
  // GetDiskFreeSpaceA() reports the cluster size but wants the volume's root.
  if (block != nullptr) {
    char full[MAX_PATH], root[MAX_PATH];
    DWORD sectors = 0, sector_bytes = 0, free_clusters = 0, clusters = 0;
    const DWORD n = GetFullPathNameA(d.c_str(), MAX_PATH, full, nullptr);
    if (n > 0 && n < MAX_PATH && GetVolumePathNameA(full, root, MAX_PATH) &&
        GetDiskFreeSpaceA(root, &sectors, &sector_bytes, &free_clusters, &clusters))
      bs = static_cast<uint64_t>(sectors) * sector_bytes;
  }
#else
  struct statvfs sv{};
  if (::statvfs(dir.c_str(), &sv) != 0) return false;
  *avail = static_cast<uint64_t>(sv.f_bavail) * sv.f_frsize;
  *total = static_cast<uint64_t>(sv.f_blocks) * sv.f_frsize;
  bs = sv.f_frsize;
#endif
  if (block != nullptr && bs != 0) *block = bs;
  return true;
}

// pid-<pid>/active tells producers outside the runtime that this process's
// capture is on. open() removes a stale one before any step that can fail and
// creates it as its last step. flush() removes it at shutdown, and
// stop_for_space() when the capture stops early. events.bin cannot carry that
// signal: a resume that fails after opening it leaves the earlier run's file in
// place. The marker holds one line naming the process instance (see
// process_instance()). A killed process cannot remove its marker, and a producer
// in a later process with the same pid checks before HIP starts, so it must find
// its own instance in the marker.
static constexpr const char* kActiveMarker = "/active";

// Flush what is buffered, close events.bin and mark the archive incomplete.
// Every write path already treats a closed events fd as "capture off". Both flags
// are raised under g_file_mu, so a writer that sees the stop flag and then takes
// the mutex finds events.bin already closed, and flush() cannot find it closed
// without also seeing the archive marked incomplete.
static void stop_for_space(uint64_t keep_free) {
  char reason[160];
  snprintf(reason, sizeof(reason),
           "stopped writing: less than %llu MiB would stay free on the file system "
           "holding the archive",
           static_cast<unsigned long long>(keep_free >> 20));
  BufWriteGuard lk;
  if (g_out_of_space.exchange(true)) return;
  if (g_events_fd >= 0) {
    flush_buffer_locked();
    HRR_FSYNC(g_events_fd);
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
  // The application runs on, and producers would keep writing sidecars to an
  // archive that records nothing, on a file system already short of space.
  (void)remove((g_output_dir + kActiveMarker).c_str());
  mark_incomplete(reason);
  // Not gated on AMD_LOG_LEVEL, like the refusal in open().
  fprintf(stderr,
          "[HRR capture] Capture stopped: less than %llu MiB would stay free on the file "
          "system holding %s. The archive is incomplete.\n",
          static_cast<unsigned long long>(keep_free >> 20), g_output_dir.c_str());
}

// Size the reserve for the file system holding `dir`, the archive open() is
// about to publish. False when less than the reserve is free to begin with.
static bool init_space_reserve(const std::string& dir) {
  g_keep_free = 0;
  g_bytes_since_space_check = 0;
  g_bytes_reserved.store(0, std::memory_order_relaxed);
  uint64_t avail = 0, total = 0, block = kFileBlockDefault;
  if (!fs_space(dir, &avail, &total, &block)) return true;
  g_file_block_bytes.store(block, std::memory_order_relaxed);
  g_keep_free = std::min<uint64_t>(total / 100 * 15, kKeepFreeMax);
  g_space_check_bytes = std::max<uint64_t>(std::min<uint64_t>(kSpaceCheckMax, g_keep_free / 4), 1);
  return avail >= g_keep_free;
}

// Count `len` bytes toward the next free-space check and return true when one is
// due. Caller must hold g_file_mu.
static bool space_check_due_locked(uint64_t len) {
  g_bytes_since_space_check += len;
  if (len < g_space_check_bytes && g_bytes_since_space_check < g_space_check_bytes) return false;
  g_bytes_since_space_check = 0;
  return true;
}

// Read free space and return false, having stopped the capture, if `pending` more
// bytes, on top of the events still buffered, would eat into the reserve. Must not
// be called with g_file_mu held.
static bool check_space(uint64_t pending) {
  {
    // Read before the free space, so a flush in between is counted twice, not missed.
    std::lock_guard<std::mutex> lk(g_file_mu);
    pending += g_buf_len;
  }
  uint64_t avail = 0, total = 0;
  if (!fs_space(g_output_dir, &avail, &total)) return true;
  if (avail >= g_keep_free && avail - g_keep_free >= pending) return true;
  stop_for_space(g_keep_free);
  return false;
}

static uint64_t file_bytes(uint64_t len) {
  const uint64_t block = g_file_block_bytes.load(std::memory_order_relaxed);
  return (len + block - 1) / block * block;
}

// Account for a file of `len` bytes and return false, having stopped the
// capture, if writing it would eat into the reserve. Must not be called with
// g_file_mu held. On success the bytes stay reserved until release_space(len).
static bool reserve_space(uint64_t len) {
  if (g_out_of_space.load(std::memory_order_relaxed)) return false;
  if (g_keep_free == 0) return true;
  len = file_bytes(len);
  const uint64_t reserved =
      g_bytes_reserved.fetch_add(len, std::memory_order_acq_rel) + len;
  bool due;
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    due = space_check_due_locked(len);
  }
  if (!due || check_space(reserved)) return true;
  g_bytes_reserved.fetch_sub(len, std::memory_order_acq_rel);
  return false;
}

static void release_space(uint64_t len) {
  if (g_keep_free == 0 || len == 0) return;
  g_bytes_reserved.fetch_sub(file_bytes(len), std::memory_order_acq_rel);
}

// Held for the rest of a writer after reserve_space(len) succeeds, so every way
// out of it, early returns and exceptions included, releases the bytes.
struct SpaceReservation {
  uint64_t len;
  ~SpaceReservation() { release_space(len); }
};

// For open() refusing an archive. A forked child inherits these paths and
// atexit(hip_capture_shutdown), so without this its flush() would finalize the
// parent's archive, or one the child never opened. An empty g_base_dir also keeps
// reopen_after_fork() from trying again.
static void clear_archive_paths() {
  g_base_dir.clear();
  g_output_dir.clear();
  g_manifest_path[0] = '\0';
}

// ---------------------------------------------------------------------------
// open / close / flush / checkpoint
// ---------------------------------------------------------------------------

// A failed open() keeps no archive path, so reopen_after_fork() does not open one
// in a forked child. When this attempt created pid-<pid>, its empty directories go
// too, so a refused capture leaves nothing behind. Only empty directories are
// removed, and a link in their place is not followed.
static bool open_failed(const std::string& out_dir, bool created_pid_dir) {
  if (created_pid_dir) {
    for (const char* sub : {"/blobs", "/code_objects", ""}) {
      const std::string dir = out_dir + sub;
#ifdef _WIN32
      std::error_code ec;
      if (fs::is_directory(fs::symlink_status(dir, ec)) && fs::is_empty(dir, ec))
        fs::remove(dir, ec);
#else
      (void)::rmdir(dir.c_str());
#endif
    }
  }
  clear_archive_paths();
  return false;
}

// Names this process instance, so a producer can tell the marker its own writer
// created from one that a killed process with the same pid left behind. On Linux
// it is the boot id and the start time from /proc/self/stat, in clock ticks since
// boot; on Windows the creation time as a FILETIME. Empty when it cannot be read.
static std::string process_instance() {
#if defined(_WIN32)
  FILETIME created{}, exited{}, kernel{}, user{};
  if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return {};
  return std::to_string((static_cast<uint64_t>(created.dwHighDateTime) << 32) |
                        created.dwLowDateTime);
#elif defined(__linux__)
  auto first_line = [](const char* path) {
    std::string line;
    if (FILE* f = fopen(path, "r")) {
      char buf[1024];
      if (fgets(buf, sizeof(buf), f) != nullptr) line = buf;
      fclose(f);
    }
    return line;
  };
  std::string boot = first_line("/proc/sys/kernel/random/boot_id");
  while (!boot.empty() && boot.back() == '\n') boot.pop_back();
  const std::string stat_line = first_line("/proc/self/stat");
  // Field 2, the command name, may hold spaces and parentheses. Every field
  // after the last ')' is one word with one space before it.
  const size_t name_end = stat_line.rfind(')');
  if (boot.empty() || name_end == std::string::npos) return {};
  const char* p = stat_line.c_str() + name_end;
  for (int field = 3; field <= 22; ++field) {
    p = strchr(p, ' ');
    if (p == nullptr) return {};
    ++p;
  }
  const size_t digits = strspn(p, "0123456789");
  if (digits == 0) return {};
  return boot + " " + std::string(p, digits);
#else
  return {};
#endif
}

static bool publish_active_marker(const std::string& out_dir) {
  const std::string marker = out_dir + kActiveMarker;
  const std::string instance = process_instance();
  if (instance.empty())
    LogPrintfWarning("[HRR capture] Cannot name this process in %s, so producers will not "
                     "take the capture for active", marker.c_str());
  const int fd = HRR_OPEN(marker.c_str());
  if (fd < 0) return false;
  const std::string line = instance + "\n";
  const bool written = write_all_fd(fd, line.data(), line.size());
  const int err = errno;
  HRR_CLOSE(fd);
  if (written) return true;
  (void)remove(marker.c_str());
  errno = err;
  return false;
}

bool open(const char* output_dir) {
  if (is_open()) return true;  // already open — guard against double-invocation
  if (g_out_of_space.load(std::memory_order_relaxed)) {  // e.g. a child after fork
    clear_archive_paths();
    return false;
  }
#ifndef _WIN32
  install_atfork_handlers_once();
#endif
  // The archive is prepared in locals and published below under the writer
  // mutex and g_buf_busy. A forked child opens its archive while other threads
  // may checkpoint, crash or finalize, and none of them may see the events fd
  // before the buffer holds the file header. g_base_dir is set first because
  // ensure_dir() reads it; open() runs at init or under g_reopen_mu, as flush()
  // does.
  g_base_dir = output_dir;
  const uint64_t pid = current_process_id();
  const uint64_t parent_pid = current_parent_process_id();
  char sub[64];
  snprintf(sub, sizeof(sub), "/pid-%llu", static_cast<unsigned long long>(pid));
  const std::string out_dir = g_base_dir + sub;

  std::error_code exists_ec;
  const bool created_pid_dir =
      !fs::exists(fs::symlink_status(out_dir, exists_ec));
  bool dirs_ok = ensure_dir(out_dir) && claim_private_dir(out_dir);
  // A marker left by an earlier process with this pid goes first, once the
  // directory it sits in is known to be ours.
  if (dirs_ok) (void)remove((out_dir + kActiveMarker).c_str());
  dirs_ok = dirs_ok && ensure_dir(out_dir + "/blobs") &&
            claim_private_dir(out_dir + "/blobs") &&
            ensure_dir(out_dir + "/code_objects") &&
            claim_private_dir(out_dir + "/code_objects");
  if (!dirs_ok) {
    const int err = errno;
    LogPrintfError("[HRR capture] Cannot use %s as a private archive directory: %s",
                   out_dir.c_str(), strerror(err));
    fprintf(stderr, "[HRR capture] Capture disabled: cannot use %s as a private archive "
            "directory (%s).\n", out_dir.c_str(), strerror(err));
    return open_failed(out_dir, created_pid_dir);
  }
#ifndef _WIN32
  for (auto& claimed : g_blob_prefix_claimed) claimed.store(false, std::memory_order_relaxed);
#endif

  if (!init_space_reserve(out_dir)) {
    LogPrintfError("[HRR capture] Less than %llu MiB free on the file system holding %s",
                   static_cast<unsigned long long>(g_keep_free >> 20), out_dir.c_str());
    fprintf(stderr, "[HRR capture] Capture disabled: less than %llu MiB free on the file "
            "system holding %s.\n", static_cast<unsigned long long>(g_keep_free >> 20),
            out_dir.c_str());
    return open_failed(out_dir, created_pid_dir);
  }

  std::string events_path = out_dir + "/events.bin";
  std::string manifest_path = out_dir + "/manifest.json";

  std::int64_t existing_size = 0;
  const int fd = open_events_file(events_path, &existing_size);
  if (fd < 0) {
    const int err = errno;
    LogPrintfError("[HRR capture] Failed to open %s: %s", events_path.c_str(), strerror(err));
    fprintf(stderr, "[HRR capture] Capture disabled: cannot open %s (%s).\n",
            events_path.c_str(), strerror(err));
    return open_failed(out_dir, created_pid_dir);
  }

  const bool exists = existing_size > 0;
  uint64_t next_seq = 0, ev_count = 0, bl_count = 0;
  bool fast = false;
  ScanResult scan{};
  if (exists) {
    const std::string state_path = out_dir + "/writer_state.json";
    fast = try_load_writer_state(state_path, existing_size, &next_seq, &ev_count, &bl_count);

    // Without the scan a trailer is not stripped and the counters can restart,
    // so an archive that cannot be scanned is not resumed.
    bool scanned = false;
    int scan_err = 0;
#ifdef _WIN32
    if (FILE* rf = fopen(events_path.c_str(), "rb")) {
      scan = scan_events_for_resume(rf, existing_size);
      fclose(rf);
      scanned = true;
    } else {
      scan_err = errno;
    }
#else
    // Reuse the already-validated events descriptor so a pathname swap cannot
    // make the scan follow a different file than open_events_file accepted.
    const int scan_fd = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (scan_fd < 0) {
      scan_err = errno;
    } else if (FILE* rf = ::fdopen(scan_fd, "rb")) {
      scan = scan_events_for_resume(rf, existing_size);
      fclose(rf);
      scanned = true;
    } else {
      scan_err = errno;
      ::close(scan_fd);
    }
#endif
    if (!scanned) {
      LogPrintfError("[HRR capture] Cannot read %s to resume it: %s", events_path.c_str(),
                     strerror(scan_err));
      fprintf(stderr, "[HRR capture] Capture disabled: cannot read %s to resume it (%s).\n",
              events_path.c_str(), strerror(scan_err));
      HRR_CLOSE(fd);
      return open_failed(out_dir, created_pid_dir);
    }
    if (!fast) {
      next_seq = (scan.count > 0) ? (scan.max_seq + 1) : 0;
      ev_count = scan.count;
    }

    // The reader stops at a trailer or a torn tail, so records appended after
    // one that cannot be cut off would be lost: such an archive is not resumed.
    if (scan.append_at > 0 && (scan.had_trailer || scan.torn_tail) &&
        hrr_ftruncate_fd(fd, scan.append_at) != 0) {
      const int err = errno;
      LogPrintfError("[HRR capture] ftruncate resume of %s at %lld failed: %s", events_path.c_str(),
                     (long long)scan.append_at, strerror(err));
      fprintf(stderr, "[HRR capture] Capture disabled: cannot trim %s to resume it (%s).\n",
              events_path.c_str(), strerror(err));
      HRR_CLOSE(fd);
      return open_failed(out_dir, created_pid_dir);
    }
    if (hrr_seek_end(fd) < 0) {
      const int err = errno;
      LogPrintfError("[HRR capture] seek end of %s failed", events_path.c_str());
      fprintf(stderr, "[HRR capture] Capture disabled: cannot seek to the end of %s (%s).\n",
              events_path.c_str(), strerror(err));
      HRR_CLOSE(fd);
      return open_failed(out_dir, created_pid_dir);
    }

    index_existing_blobs_locked(out_dir);
  } else {
    // Fresh per-process archive. Incomplete belongs to the archive, as in
    // atfork_child: a failure recorded against one closed earlier in this
    // process must not cost this one its trailer. A resumed archive keeps it.
    // So does the unreplayable list, which a forked child also drops here.
    g_capture_incomplete.store(false, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(g_unreplayable_mu);
      g_unreplayable_apis.clear();
    }
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.clear();
    // A forked child drops the claims of the parent's writers here, not in
    // atfork_child, which does only async-signal-safe work.
    g_blob_claims.clear();
  }

  // The last step before the archive is published, on resume and on a fresh one.
  if (!publish_active_marker(out_dir)) {
    const int err = errno;
    LogPrintfError("[HRR capture] Cannot create %s%s: %s", out_dir.c_str(), kActiveMarker,
                   strerror(err));
    fprintf(stderr, "[HRR capture] Capture disabled: cannot create %s%s (%s).\n",
            out_dir.c_str(), kActiveMarker, strerror(err));
    HRR_CLOSE(fd);
    return open_failed(out_dir, created_pid_dir);
  }

  {
    BufWriteGuard lk;
    g_output_dir = out_dir;
    g_pid        = pid;
    g_parent_pid = parent_pid;
    snprintf(g_manifest_path, sizeof(g_manifest_path), "%s", manifest_path.c_str());
    // Buffer/checkpoint state is reset here so it is consistent for this
    // process's pid-<pid> sub-archive.
    g_buf_len           = 0;
    g_events_since_ckpt = 0;
    g_events_finalized  = false;
    g_seq_id.store(next_seq, std::memory_order_relaxed);
    g_event_count.store(ev_count, std::memory_order_relaxed);
    if (!exists)
      g_blob_count.store(0, std::memory_order_relaxed);
    else if (fast)
      g_blob_count.store(bl_count, std::memory_order_relaxed);
    g_events_fd = fd;
    if (!exists) {
      // Pitched host copies record only their copied rows (hip_capture.cpp,
      // write_host_rect_blob), which replay reads back by this flag.
      hrr_file_header fh{HRR_MAGIC, HRR_VERSION, HRR_FILE_FLAG_PACKED_HOST_RECTS};
      buffer_append_locked(&fh, sizeof(fh));
    }
  }

  if (exists) {
    LogPrintfInfo("[HRR capture] Resumed archive at %s (events=%llu next_seq=%llu%s%s)",
                  events_path.c_str(),
                  static_cast<unsigned long long>(ev_count),
                  static_cast<unsigned long long>(next_seq),
                  scan.had_trailer ? ", stripped trailer" : "",
                  scan.torn_tail ? ", trimmed torn tail" : "");
  }
  return true;
}

void checkpoint() {
  BufWriteGuard lk;
  if (g_events_fd < 0) return;
  flush_buffer_locked();
  HRR_FSYNC(g_events_fd);
  save_writer_state_locked();
  g_events_since_ckpt = 0;
}

// Caller holds g_blob_mu. A claimed write ends here even after flush() has cut
// off new ones: flush() waits for it before reading the flag, so it still counts.
static void mark_incomplete_locked(const char* reason) {
  // Record once; the loud, AMD_LOG_LEVEL-routed message is emitted by the caller
  // (e.g. serialize_kernel_launch) which has the relevant context. Here we only
  // need the durable flag and a single breadcrumb so a bare run still surfaces
  // it. Error level: the archive is not faithful and replay must not treat it
  // as one. log_printf appends its own newline, so the format string omits it.
  if (!g_capture_incomplete.exchange(true, std::memory_order_relaxed)) {
    LogPrintfError(
        "[HRR capture] Archive marked INCOMPLETE: %s. The clean-shutdown "
        "trailer will be omitted and manifest.complete=false so replay "
        "cannot treat this capture as faithful",
        reason ? reason : "(unspecified)");
  }
}

// Past the trailer decision, an event a shim drops is one nothing would have
// recorded anyway, and flipping the flag then would leave the trailer and the
// manifest disagreeing. flush() sets g_events_finalized under g_blob_mu, so
// taking it here puts the check on the same side of that cut-off as the writes.
// stop_for_space() calls this with g_file_mu held, so it must not take that
// mutex; g_blob_mu comes after g_file_mu, as in flush().
void mark_incomplete(const char* reason) {
  std::lock_guard<std::mutex> lk(g_blob_mu);
  if (g_events_finalized) return;
  mark_incomplete_locked(reason);
}

bool is_incomplete() { return g_capture_incomplete.load(std::memory_order_relaxed); }

// Caller holds g_unreplayable_mu. True the first time (api, reason) is listed.
static bool list_unreplayable_locked(const char* api, const char* reason) {
  return g_unreplayable_apis[api].insert(reason).second;
}

static void warn_unreplayable(const char* api, const char* reason) {
  // Warning, not Error: unlike mark_incomplete() the archive is well-formed and
  // every event is present — only the ability to re-execute this one call is
  // lost. That is a degradation, not a failure.
  // log_printf appends its own newline, so the format string omits it.
  LogPrintfWarning(
      "[HRR capture] %s cannot be replayed: %s. The call is recorded, but "
      "replay will report it as unreplayable rather than reproduce it",
      api, reason);
}

// Staged, not listed: write_event_raw() lists the note under the same lock that
// accepts the event, so a shim still reached through a wrapped slot after
// flush() cannot name an API whose event the cut-off drops.
void note_unreplayable(const char* api, const char* reason) {
  if (!api) return;
  if (!reason) reason = "(unspecified)";
  if (t_staged_count < kMaxStagedNotes) {
    t_staged_notes[t_staged_count++] = {api, reason};
    return;
  }
  bool fresh;
  {
    std::lock_guard<std::mutex> lk(g_unreplayable_mu);
    fresh = list_unreplayable_locked(api, reason);
  }
  if (fresh) warn_unreplayable(api, reason);
}

void flush(const char* /*output_dir*/) {
  // Always finalize the *effective* directory this process actually wrote to
  // (g_output_dir), which is always a pid-<pid> sub-archive. The caller passes
  // the base HIP_HRR_CAPTURE_OUTPUT path.
  //
  // A forked child opens its archive on its first record, under g_reopen_mu.
  // Holding it here keeps the trailer and the manifest from landing before
  // that open has finished, and no archive opens once this one is finalized.
  std::lock_guard<std::mutex> reopen_lk(g_reopen_mu);
  g_reopen_after_fork.store(false, std::memory_order_release);
  bool incomplete;
  std::string out_dir;
  {
    BufWriteGuard lk;
    // Must be read under the lock, see stop_for_space().
    incomplete = g_capture_incomplete.load(std::memory_order_relaxed);
    out_dir = g_output_dir;
    if (g_events_fd >= 0 && !g_events_finalized) {
      {
        // Stop new blob and code object claims, then wait for the claimed writes
        // to end: a failed one marks the capture incomplete, which decides the
        // trailer and the manifest below. Holding g_file_mu keeps events out
        // meanwhile, so none is recorded without its blob.
        std::unique_lock<std::mutex> blk(g_blob_mu);
        g_events_finalized = true;
        g_blob_writes_done.wait(blk, [] { return g_blob_writes_in_flight == 0; });
      }
      incomplete = g_capture_incomplete.load(std::memory_order_relaxed);
      // Skip the clean-shutdown trailer when the capture is known incomplete: its
      // absence is exactly how the reader detects a non-faithful archive.
      if (!incomplete) {
        hrr_eof_record rec = hrr_make_eof_record(
            g_seq_id.fetch_add(1, std::memory_order_relaxed), g_event_count.load());
        rec.hdr.timestamp_ns = amd::Os::timeNanos();
        rec.hdr.thread_id    = current_thread_id();
        buffer_append_locked(&rec, sizeof(rec));
      }
      flush_buffer_locked();
      HRR_FSYNC(g_events_fd);
    }
    incomplete = g_capture_incomplete.load(std::memory_order_relaxed);
    // close() runs later and the fd stays open until then. A thread can still
    // record in between: a forked child's first record finishes opening the
    // archive just before this, and takes the lock after it.
    g_events_finalized = true;
  }

  if (out_dir.empty()) return;
  write_manifest_stdio(out_dir.c_str(), /*complete=*/!incomplete);
  update_root_manifest();
  remove((out_dir + "/writer_state.json").c_str());
  remove((out_dir + kActiveMarker).c_str());
}

void close() {
  // A forked child that has not recorded yet must not open its archive after
  // this: nothing would finalize it. Fat-binary destructors still record then.
  std::lock_guard<std::mutex> reopen_lk(g_reopen_mu);
  g_reopen_after_fork.store(false, std::memory_order_release);
  BufWriteGuard lk;
  if (g_events_fd >= 0) {
    flush_buffer_locked();
    HRR_FSYNC(g_events_fd);
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
}

// ---------------------------------------------------------------------------
// emergency_finalize — best-effort crash callback path
// ---------------------------------------------------------------------------

// Async-signal-safe unsigned-to-decimal. Writes into out (no NUL), returns len.
static size_t u64_to_dec(uint64_t v, char* out) {
  char tmp[20];
  size_t n = 0;
  if (v == 0) { out[0] = '0'; return 1; }
  while (v) { tmp[n++] = static_cast<char>('0' + (v % 10)); v /= 10; }
  for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
  return n;
}

static size_t append_lit(char* out, size_t off, const char* s) {
  size_t i = 0;
  while (s[i]) { out[off + i] = s[i]; i++; }
  return off + i;
}

void emergency_finalize(bool clean_shutdown) {
  // A space stop has already closed events.bin, but a crash after it must still
  // leave a manifest that says the archive is incomplete.
  if (g_events_fd < 0 && (clean_shutdown || !g_out_of_space.load(std::memory_order_relaxed)))
    return;

  // Flush the in-memory buffer only if no writer thread is mid-mutation.
  // We must NOT touch g_file_mu here: if the crash interrupts a writer mid-lock,
  // probing the futex from the crash callback can deadlock or corrupt state.
  // Instead probe g_buf_busy with test_and_set: if it was already set a writer
  // holds g_buf (possibly a torn record in flight) so we only fsync
  // already-written bytes; if it was clear we now own it and can safely flush.
  // clear() releases it on the way out.
  bool locked = !g_buf_busy.test_and_set(std::memory_order_acquire);
  if (locked) {
    flush_buffer_locked();
    // Clean shutdowns append the fixed-size trailer so the reader does not treat
    // the archive as crash-truncated. The CLR crash callback passes
    // clean_shutdown=false; normal shutdown uses flush().
    if (clean_shutdown && !g_events_finalized) {
      hrr_eof_record rec = hrr_make_eof_record(
          g_seq_id.fetch_add(1, std::memory_order_relaxed), g_event_count.load());
      write_all_fd(g_events_fd, &rec, sizeof(rec));
      g_events_finalized = true;
    }
    g_buf_busy.clear(std::memory_order_release);
  }
  // After a space stop there is no descriptor, and on Windows _commit(-1) calls
  // the CRT invalid parameter handler, which ends the process by default.
  if (g_events_fd >= 0) HRR_FSYNC(g_events_fd);

  // Best-effort manifest via raw open/write only. A clean shutdown writes
  // complete:true (the trailer is present); a crash writes complete:false — its
  // absence-of-trailer is how the reader detects truncation.
  if (g_manifest_path[0] == '\0') return;
  // Concurrent crash callbacks must not share the emergency buffer. A second
  // entrant skips the manifest.
  if (g_emergency_manifest_busy.test_and_set(std::memory_order_acquire)) return;
  bool complete = clean_shutdown && locked;
  int mfd = HRR_OPEN(g_manifest_path);
  if (mfd < 0) {
    g_emergency_manifest_busy.clear(std::memory_order_release);
    return;
  }
  auto& buf = g_emergency_manifest_buf;
  size_t p = 0;
  p = append_lit(buf, p,
                 "{\n"
                 "  \"pid\": ");
  p += u64_to_dec(g_pid, buf + p);
  p = append_lit(buf, p, ",\n  \"parent_pid\": ");
  p += u64_to_dec(g_parent_pid, buf + p);
  p = append_lit(buf, p, ",\n  \"complete\": ");
  p = append_lit(buf, p, complete ? "true" : "false");
  p = append_lit(buf, p, ",\n  \"event_count\": ");
  p += u64_to_dec(g_event_count.load(), buf + p);
  p = append_lit(buf, p, ",\n  \"blob_count\": ");
  p += u64_to_dec(g_blob_count.load(), buf + p);
  // g_metadata_json is written once during HRR init before event capture starts.
  // The crash path reads it lock-free to avoid taking g_file_mu from an exception callback.
  static constexpr const char* kMetadataFieldPrefix = ",\n  \"metadata\": ";
  if (g_metadata_json_len > 0 &&
      p + strlen(kMetadataFieldPrefix) + g_metadata_json_len + sizeof("\n}\n") < sizeof(buf)) {
    p = append_lit(buf, p, kMetadataFieldPrefix);
    memcpy(buf + p, g_metadata_json, g_metadata_json_len);
    p += g_metadata_json_len;
  }
  p = append_lit(buf, p, "\n}\n");
  write_all_fd(mfd, buf, p);
  HRR_FSYNC(mfd);
  HRR_CLOSE(mfd);
  g_emergency_manifest_busy.clear(std::memory_order_release);
}

// ---------------------------------------------------------------------------
// write_event_raw — unified write path for all events
//
// hdr points to the hrr_event_header at the front of an hrr_args_* struct.
// payload_len is sizeof the full hrr_args_* struct (header + fields).
// Fills all header fields then copies the whole struct into the app buffer.
// ---------------------------------------------------------------------------

// Assign sequence_id and buffer the record atomically so IDs are only consumed for
// events that are actually written. A full record is always appended under the
// lock, so the buffer never holds a torn record — which is what makes the
// crash-callback flush in emergency_finalize() safe. Caller holds BufWriteGuard and
// has seen g_events_fd open and events.bin not finalized. The `staged` notes of
// this thread are listed under the same lock; fresh[i] says whether note i was new.
//
// The checkpoint flush+fsync happens inside the same lock scope. An earlier
// version released the lock and re-acquired it for the fsync, which let two
// threads that both crossed the kCheckpointEvents boundary race into back-to-
// back fsyncs (a thundering herd at every 4096-event boundary). Doing the
// fsync under the lock blocks other writers for the duration of the syscall,
// but guarantees exactly one fsync per checkpoint and removes the race.
static void append_event_locked(hrr_event_header* hdr, uint32_t payload_len,
                                size_t staged, bool* fresh) {
  hdr->sequence_id = g_seq_id.fetch_add(1, std::memory_order_relaxed);
  buffer_append_locked(hdr, payload_len);
  g_event_count.fetch_add(1, std::memory_order_relaxed);
  if (staged > 0) {
    std::lock_guard<std::mutex> ulk(g_unreplayable_mu);
    for (size_t i = 0; i < staged; ++i)
      fresh[i] = list_unreplayable_locked(t_staged_notes[i].api, t_staged_notes[i].reason);
  }
  if (++g_events_since_ckpt >= kCheckpointEvents) {
    flush_buffer_locked();
    HRR_FSYNC(g_events_fd);
    g_events_since_ckpt = 0;
  }
}

void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint32_t payload_len) {
  reopen_after_fork();
  // Fill fields that don't require the lock (timestamp and thread_id are
  // cheap and per-thread; getting them outside the lock keeps contention low).
  hdr->event_type     = api_id;
  hdr->timestamp_ns   = amd::Os::timeNanos();
  hdr->thread_id      = current_thread_id();
  hdr->payload_length = payload_len;
  memset(hdr->reserved, 0, sizeof(hdr->reserved));

  // The notes staged for this event share its fate.
  const size_t staged = t_staged_count;
  t_staged_count = 0;
  bool fresh[kMaxStagedNotes] = {};

  // A record is counted toward the next free-space check before it is written. When
  // it makes a check due, the check runs first, so it covers the record, as for a
  // blob: the bytes written between two checks stay under one interval.
  bool due;
  {
    BufWriteGuard lk;
    // Shims stay reachable during shutdown through slots another component wrapped,
    // and the reader would replay a record after the trailer as part of the archive.
    if (g_events_fd < 0 || g_events_finalized) return;
    due = g_keep_free != 0 && space_check_due_locked(payload_len);
    if (!due) append_event_locked(hdr, payload_len, staged, fresh);
  }
  if (due) {
    // Outside g_file_mu, since check_space may take it to stop the capture. The
    // record's bytes stay reserved until it is buffered, so a concurrent check counts
    // them.
    const uint64_t reserved =
        g_bytes_reserved.fetch_add(payload_len, std::memory_order_acq_rel) + payload_len;
    if (check_space(reserved)) {
      BufWriteGuard lk;
      if (g_events_fd >= 0 && !g_events_finalized)
        append_event_locked(hdr, payload_len, staged, fresh);
    }
    g_bytes_reserved.fetch_sub(payload_len, std::memory_order_acq_rel);
  }
  for (size_t i = 0; i < staged; ++i)
    if (fresh[i]) warn_unreplayable(t_staged_notes[i].api, t_staged_notes[i].reason);
}

// ---------------------------------------------------------------------------
// Atomic file write: write to a temp file then rename into place.
//
// g_blob_claims ensures only one thread ever reaches here for a given path,
// so there is no concurrent write to the same temp file. The rename makes the
// blob visible to readers only when fully written: a process crash mid-write
// leaves only the temp file, not a partial final blob.
//
// On Windows, rename() fails when the destination already exists (unlike POSIX
// where it is atomic). Use MoveFileExA(MOVEFILE_REPLACE_EXISTING) instead.
//
// On POSIX the temp file comes from mkostemps: an unpredictable name, created
// exclusively with mode 0600. This matters for the root manifest, whose
// directory is whatever HIP_HRR_CAPTURE_OUTPUT names and may be shared.
// ---------------------------------------------------------------------------

static bool atomic_write_file(const std::string& path,
                              const void* data, size_t len) {
#ifdef _WIN32
  std::string tmp = path + "." + std::to_string(current_process_id()) + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  bool ok = (fwrite(data, 1, len, f) == len);
  // fwrite() can leave the data in the stdio buffer, so the write that fails
  // may be the one fclose() makes.
  if (fclose(f) != 0) ok = false;
  if (!ok) { remove(tmp.c_str()); return false; }
  ok = MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
  std::string tmp = path + ".XXXXXX.tmp";
  const int fd = ::mkostemps(&tmp[0], 4, O_CLOEXEC);
  if (fd < 0) return false;
  // mkostemps creates the file 0600 minus the umask.
  bool ok = ::fchmod(fd, 0600) == 0 && write_all_fd(fd, data, len);
  if (::close(fd) != 0) ok = false;
  if (!ok) { remove(tmp.c_str()); return false; }
  ok = (rename(tmp.c_str(), path.c_str()) == 0);
#endif
  if (!ok) remove(tmp.c_str());
  return ok;
}

// Ends a write claimed in write_blob() or write_code_object(). A failed one is
// unpublished so a later call can retry, and marks the capture incomplete: the
// caller already holds the hash of a file that is not there.
static void finish_claimed_write(const std::string& key, bool ok) {
  std::lock_guard<std::mutex> lk(g_blob_mu);
  if (!ok) {
    g_written_blobs.erase(key);
    g_blob_count.fetch_sub(1, std::memory_order_relaxed);
    mark_incomplete_locked("a blob or code object could not be written");
  }
  if (--g_blob_writes_in_flight == 0) g_blob_writes_done.notify_all();
}

// Finishes a claimed write on every way out of the writer, a throw from
// ensure_dir() or an allocation included, so no claim stays counted in flight
// and leaves flush() waiting for it. The write counts as failed unless done()
// reports otherwise.
class ClaimedWrite {
 public:
  explicit ClaimedWrite(const std::string& key) : key_(key) {}
  ~ClaimedWrite() { finish_claimed_write(key_, ok_); }
  ClaimedWrite(const ClaimedWrite&) = delete;
  ClaimedWrite& operator=(const ClaimedWrite&) = delete;
  void done(bool ok) { ok_ = ok; }

 private:
  const std::string& key_;
  bool ok_ = false;
};

// Claim `key` for reserving and writing, waiting while another thread holds it.
// False when the key is written, before the call or by the thread it waited for.
// Polls rather than waiting on a condition variable: one with waiters at fork()
// cannot be used safely in the child, and callers racing on the same bytes are
// rare. Waits holding no lock, so flush() can still take g_blob_mu meanwhile.
static bool claim_blob(const std::string& key) {
  std::unique_lock<std::mutex> lk(g_blob_mu);
  while (g_blob_claims.count(key) != 0) {
    lk.unlock();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    lk.lock();
  }
  if (g_written_blobs.count(key) != 0) return false;
  g_blob_claims.insert(key);
  return true;
}

// Holds a claim from claim_blob() to the end of the writer, whichever way it
// returns. A caller that waited then finds the key written, or claims it itself
// when this one gave up or its write failed.
class BlobClaim {
 public:
  explicit BlobClaim(const std::string& key) : key_(key) {}
  ~BlobClaim() {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_blob_claims.erase(key_);
  }
  BlobClaim(const BlobClaim&) = delete;
  BlobClaim& operator=(const BlobClaim&) = delete;

 private:
  const std::string& key_;
};

// ---------------------------------------------------------------------------
// write_blob
// ---------------------------------------------------------------------------

Hash128 write_blob(const void* data, size_t len) {
  reopen_after_fork();
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0 || g_events_finalized) return {};  // not open, or past the trailer
  }

  Hash128 h = hash_buffer(data, len);

  char hex[33];
  hash_hex(h, hex);
  std::string key(hex);  // no prefix — plain blobs

  // A caller with the same bytes waits here until this one is done, so the space
  // is reserved once. The write counts in flight only once the space is reserved,
  // so flush() never waits on a claim that takes g_file_mu.
  if (!claim_blob(key)) return h;  // already written
  const BlobClaim blob_claim(key);
  if (!reserve_space(len)) return {};
  const SpaceReservation reservation{len};
  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    // The check at entry ran before hashing and reserving; flush() sets the flag
    // under this lock.
    if (g_events_finalized) return {};
    g_written_blobs.insert(key);  // the claim keeps other callers out
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
    ++g_blob_writes_in_flight;
  }
  ClaimedWrite claim(key);

  // blobs/<2-char-prefix>/<fullhash>.blob
  std::string subdir = g_output_dir + "/blobs/" + std::string(hex, 2);
#ifndef _WIN32
  {
    // hash_hex writes lowercase hex digits; two chars index 0..255.
    const auto nibble = [](char c) -> unsigned {
      return (c >= '0' && c <= '9') ? static_cast<unsigned>(c - '0')
                                    : static_cast<unsigned>(c - 'a' + 10);
    };
    const unsigned pref = (nibble(hex[0]) << 4) | nibble(hex[1]);
    if (!g_blob_prefix_claimed[pref].load(std::memory_order_acquire)) {
      if (!ensure_dir(subdir) || !claim_private_dir(subdir)) {
        // `claim` unpublishes the key and marks the capture incomplete.
        LogPrintfWarning("[HRR capture] Failed to claim blob prefix %s",
                         std::string(hex, 2).c_str());
        return h;
      }
      g_blob_prefix_claimed[pref].store(true, std::memory_order_release);
    }
  }
#else
  ensure_dir(subdir);
#endif
  std::string path = subdir + "/" + key + ".blob";

  const bool ok = atomic_write_file(path, data, len);
  if (!ok) LogPrintfWarning("[HRR capture] Failed to write blob %s", hex);
  claim.done(ok);
  return h;
}

// ---------------------------------------------------------------------------
// write_code_object
// ---------------------------------------------------------------------------

Hash128 write_code_object(const void* image, size_t image_size) {
  reopen_after_fork();
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0 || g_events_finalized) return {};  // not open, or past the trailer
  }

  Hash128 h = hash_buffer(image, image_size);
  char hex[33];
  hash_hex(h, hex);
  std::string key = std::string("co:") + hex;  // namespace to match playback load_code_object key

  // As in write_blob(), the key is claimed before the space is reserved.
  if (!claim_blob(key)) return h;  // already written
  const BlobClaim blob_claim(key);
  if (!reserve_space(image_size)) return {};
  const SpaceReservation reservation{image_size};
  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    if (g_events_finalized) return {};  // as in write_blob()
    g_written_blobs.insert(key);  // the claim keeps other callers out
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
    ++g_blob_writes_in_flight;
  }
  ClaimedWrite claim(key);

  std::string path = g_output_dir + "/code_objects/" + hex + ".hsaco";
  const bool ok = atomic_write_file(path, image, image_size);
  if (!ok) LogPrintfWarning("[HRR capture] Failed to write code object %s", hex);
  claim.done(ok);
  return h;
}

// ---------------------------------------------------------------------------
// Counters / state queries
// ---------------------------------------------------------------------------

bool     is_open()      { std::lock_guard<std::mutex> lk(g_file_mu); return g_events_fd >= 0; }
uint64_t event_count()  { return g_event_count.load(); }
uint64_t blob_count()   { return g_blob_count.load(); }

void set_capture_metadata_json(const std::string& metadata_json) {
  if (!is_json_object_fragment(metadata_json)) return;
  if (metadata_json.size() >= kMetadataJsonMax) return;
  std::lock_guard<std::mutex> lk(g_file_mu);
  const size_t n = metadata_json.size();
  memcpy(g_metadata_json, metadata_json.data(), n);
  g_metadata_json[n] = '\0';
  g_metadata_json_len = n;
}

}  // namespace writer
}  // namespace hrr_cap
