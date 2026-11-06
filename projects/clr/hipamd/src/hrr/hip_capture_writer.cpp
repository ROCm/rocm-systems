/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Crash safety: events.bin uses a raw fd, checkpointed every kCheckpointEvents events;
 * a clean shutdown appends an hrr_eof_record trailer, whose absence marks a crash-truncated archive.
 * Access (POSIX): directories 0700, files 0600, opened relative to checked directory descriptors.
 */

#include "hip_capture_writer.h"
#include "hip_capture.h"
#include "hip_capture_metadata.h"
#include "hip_capture_scrub.h"

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
#include <utility>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <process.h>
#  include <sys/stat.h>
#  define HRR_OPEN(p)          _open((p), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_WRITE(fd,b,n)    _write((fd), (b), (unsigned)(n))
#  define HRR_CLOSE(fd)        _close((fd))
#  define HRR_FSYNC(fd)        _commit((fd))

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
#  define HRR_WRITE(fd,b,n)  ::write((fd), (b), (n))
#  define HRR_CLOSE(fd)      ::close((fd))
#  define HRR_FSYNC(fd)      ::fsync((fd))

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


// Buffer must comfortably hold the largest single record. payload_length is a
// uint32_t (v4); in practice the largest record is a kernel-launch event whose
static constexpr size_t   kBufCap           = 256u * 1024u;
// Flush+fsync every N events to bound crash data loss.
static constexpr uint64_t kCheckpointEvents = 4096;
// Path buffers are filled at open() so the signal path never touches std::string.
static constexpr size_t   kPathMax          = 4096;
static constexpr size_t   kMetadataJsonMax  = 128u * 1024u;
static constexpr size_t   kEmergencyManifestMax = kMetadataJsonMax + 1024u;

// Lock order: g_pinned_mu (hip_capture.cpp, held while a launch copies pinned
// host memory or writes its event), then g_reopen_mu, then g_file_mu, then
// g_blob_mu, then g_unreplayable_mu. Only fork() holds them all: the pinned
// prepare handler takes g_pinned_mu, then atfork_prepare takes the rest.
static std::mutex   g_file_mu;
static int          g_events_fd = -1;
// g_base_dir is the archive path requested via HIP_HRR_CAPTURE_OUTPUT.
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
static std::atomic<bool> g_events_finalized{false};

// Set when an event could not be serialized losslessly and had to be dropped
// (e.g. an oversized kernel launch). A capture with this flag set is finalized
static std::atomic<bool> g_capture_incomplete{false};

// Crash-callback guard over g_buf / g_buf_len, raised by writer threads while
// they mutate the buffer. If the crash interrupts a writer mid-lock,
static std::atomic_flag g_buf_busy = ATOMIC_FLAG_INIT;

// Raise g_buf_busy for a thread that already holds g_file_mu. Writers are
// serialized by the mutex, so the flag can only be up already because the crash
static void claim_buf_locked() {
  while (g_buf_busy.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
}

// RAII for writer threads: take the thread<->thread mutex AND raise g_buf_busy so
// the crash callback can tell a g_buf mutation is in flight. Member order
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
static uint64_t                g_blob_writes_in_flight = 0;  // under g_blob_mu
static std::condition_variable g_blob_writes_done;

// Set in a forked child. POSIX allows the child of a multithreaded process only
// async-signal-safe calls until it execs, and open() is far from that, so the
static std::atomic<bool> g_reopen_after_fork{false};
static std::mutex        g_reopen_mu;

// Set when capture stops early, for space or at the size limit; see
// stop_capture_locked(). A forked child keeps a stop for space, as the file
static std::atomic<bool> g_capture_stopped{false};
static bool              g_stopped_at_limit = false;  // under g_file_mu

// Keys of the blobs and code objects written to this archive, so repeated bytes
// (common for weight tensors) are written once.
static std::mutex                      g_blob_mu;
static std::unordered_set<std::string> g_written_blobs;
// Descriptors of the archive directories open() checked, pid-<pid>, blobs/ and
// code_objects/. Every file inside the archive is created, opened, renamed and
static std::atomic<int> g_dir_fd{-1};
static std::atomic<int> g_blobs_fd{-1};
static std::atomic<int> g_code_objects_fd{-1};
// Keys a thread is reserving space for or writing, also under g_blob_mu. Another
// caller with the same bytes waits for the claim to end, so the reserve is
static std::unordered_set<std::string> g_blob_claims;

// APIs recorded in this archive that replay cannot reproduce (note_unreplayable).
static std::mutex                      g_unreplayable_mu;
// api -> distinct reasons it was noted for. One API can be unreplayable for
// several reasons (e.g. a different truncated argument on different calls),
static std::map<std::string, std::set<std::string>> g_unreplayable_apis;

// Notes from note_unreplayable() that wait for the event the calling thread
// writes next. Plain pointers and a count, so nothing here has a destructor: a
static constexpr size_t kMaxStagedNotes = 8;
struct StagedNote {
  const char* api;
  const char* reason;
};
static thread_local StagedNote t_staged_notes[kMaxStagedNotes];
static thread_local size_t     t_staged_count = 0;


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


// Create `path` and any missing parents: 0700 from the base directory down, and
// 0777 minus the umask above it. `path` must lie under g_base_dir.
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

#ifndef _WIN32
// pid-<pid>, blobs/, code_objects/ and each blobs/<xx> must be a real directory
// owned by the effective user; anything else planted there (a symbolic link,
static int open_private_dir(int at, const char* name) {
  if (::mkdirat(at, name, 0700) != 0 && errno != EEXIST) return -1;
  const int fd = ::openat(at, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return -1;
  struct stat st{};
  int err = 0;
  if (::fstat(fd, &st) != 0) {
    err = errno;
  } else if (!owned_by_euid(st)) {
    err = EPERM;
  } else if ((st.st_mode & 07777) != 0700 && ::fchmod(fd, 0700) != 0) {
    err = errno;
  }
  if (err == 0) return fd;
  ::close(fd);
  errno = err;
  return -1;
}

// Closes the archive directory descriptors. Async-signal-safe: atfork_child
// calls it.
static void close_dir_fds() {
  const int dir = g_dir_fd.exchange(-1);
  if (dir < 0) return;
  ::close(dir);
  for (std::atomic<int>* fd : {&g_blobs_fd, &g_code_objects_fd})
    if (const int d = fd->exchange(-1); d >= 0) ::close(d);
}
#endif

// pid-<pid>, created with any missing parents. On POSIX *fd gets its descriptor.
static bool claim_pid_dir(const std::string& out_dir, int* fd) {
  if (!ensure_dir(out_dir)) return false;
#ifdef _WIN32
  (void)fd;
  return true;
#else
  *fd = open_private_dir(AT_FDCWD, out_dir.c_str());
  return *fd >= 0;
#endif
}

// blobs/, code_objects/ or a blobs/<xx>: `name` in the directory `parent_fd` on
// POSIX, where *fd gets its descriptor, and `path` on Windows.
static bool claim_sub_dir(int parent_fd, const std::string& path, const char* name, int* fd) {
#ifdef _WIN32
  (void)parent_fd;
  (void)name;
  (void)fd;
  return ensure_dir(path);
#else
  (void)path;
  *fd = open_private_dir(parent_fd, name);
  return *fd >= 0;
#endif
}

// Directory descriptors open() holds until it publishes them; any still held
// when it returns early are closed.
struct PendingDirs {
  int dir = -1;
  int blobs = -1;
  int code_objects = -1;
  PendingDirs() = default;
  PendingDirs(const PendingDirs&) = delete;
  PendingDirs& operator=(const PendingDirs&) = delete;
  ~PendingDirs() {
    for (int fd : {dir, blobs, code_objects})
      if (fd >= 0) HRR_CLOSE(fd);
  }
};

// True when the claimed pid-<pid> holds an earlier process's archive: an
// events.bin of the kind this writer leaves, a regular file with a single link
static bool holds_earlier_archive(int dir_fd, const std::string& out_dir) {
#ifdef _WIN32
  (void)dir_fd;
  std::error_code ec;
  return fs::is_regular_file(fs::symlink_status(out_dir + "/events.bin", ec));
#else
  (void)out_dir;
  struct stat st{};
  return ::fstatat(dir_fd, "events.bin", &st, AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(st.st_mode) &&
         st.st_nlink == 1 && owned_by_euid(st);
#endif
}

// A process never adds to an archive it did not open: an earlier process with
// the same pid, common in containers, keeps its archive. Renames pid-<pid> to
static bool move_earlier_archive_aside(const std::string& out_dir, std::string* moved_to) {
  for (unsigned n = 1; n <= 0xffff; ++n) {
    const std::string to = out_dir + ".prev-" + std::to_string(n);
    std::error_code ec;
    if (fs::exists(fs::symlink_status(to, ec))) continue;
    if (std::rename(out_dir.c_str(), to.c_str()) == 0) {
      *moved_to = to;
      return true;
    }
    if (errno != EEXIST && errno != ENOTEMPTY) return false;
  }
  errno = EEXIST;
  return false;
}

#ifndef _WIN32
// Open a file inside the archive for writing and truncate it: 0600, never
// through a symbolic link or a hard link. Truncation happens only after the
static int open_private_fd(int dir_fd, const char* name) {
  const int fd = ::openat(dir_fd, name, O_WRONLY | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0600);
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

// Opens `name` inside the archive for writing and truncates it: in the directory
// `dir_fd` through open_private_fd on POSIX, in `dir` by path on Windows.
static int open_in_archive(int dir_fd, const std::string& dir, const char* name) {
#ifdef _WIN32
  (void)dir_fd;
  return HRR_OPEN((dir + "/" + name).c_str());
#else
  (void)dir;
  return open_private_fd(dir_fd, name);
#endif
}

// Removes `name` from the archive directory, as open_in_archive() finds it.
static void remove_in_archive(int dir_fd, const std::string& dir, const char* name) {
#ifdef _WIN32
  (void)dir_fd;
  (void)remove((dir + "/" + name).c_str());
#else
  (void)dir;
  (void)::unlinkat(dir_fd, name, 0);
#endif
}

// fopen(path, "w") for a file inside the archive, opened by open_in_archive().
static FILE* fopen_private(int dir_fd, const std::string& dir, const char* name) {
#ifdef _WIN32
  (void)dir_fd;
  return fopen((dir + "/" + name).c_str(), "w");
#else
  const int fd = open_in_archive(dir_fd, dir, name);
  if (fd < 0) return nullptr;
  FILE* f = ::fdopen(fd, "w");
  if (!f) ::close(fd);
  return f;
#endif
}

// fopen(path, "r") for a file inside the archive. On POSIX only a regular file
// is read, never through a symbolic link, and a planted FIFO is refused rather
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

static bool atomic_write_file(int dir_fd, const std::string& dir, const std::string& name,
                              const void* data, size_t len);

static bool is_json_object_fragment(const std::string& json) {
  const auto begin = json.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos || json[begin] != '{') return false;
  const auto end = json.find_last_not_of(" \t\r\n");
  return end != std::string::npos && json[end] == '}';
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
static void atfork_prepare() {
  g_reopen_mu.lock();
  g_file_mu.lock();
  claim_buf_locked();
  if (g_events_fd >= 0)
    flush_buffer_locked();
  g_buf_busy.clear(std::memory_order_release);
  // A blob write in flight holds its blobs/<xx> descriptor, which the child
  // would keep without the thread that closes it. Wait for those writes, as
  std::unique_lock<std::mutex> blk(g_blob_mu);
  g_blob_writes_done.wait(blk, [] { return g_blob_writes_in_flight == 0; });
  blk.release();
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
  g_unreplayable_mu.unlock();
  g_blob_mu.unlock();
  g_file_mu.unlock();
  g_reopen_mu.unlock();
  // A crash callback on another thread can raise g_buf_busy after
  // atfork_prepare clears it, or hold g_emergency_manifest_busy, which
  g_buf_busy.clear(std::memory_order_release);
  g_emergency_manifest_busy.clear(std::memory_order_release);
  // Only async-signal-safe work from here: drop the parent's events fd and
  // forget its paths, so neither shutdown nor the crash path writes into the
  const bool parent_open = g_events_fd >= 0;
  const bool parent_finalized = g_events_finalized;
  // The child's archive starts with a fresh HIP_HRR_CAPTURE_MAX_MB budget.
  const bool parent_at_limit = g_capture_stopped.load(std::memory_order_relaxed) &&
                               g_stopped_at_limit;
  if (parent_at_limit) {
    g_capture_stopped.store(false, std::memory_order_relaxed);
    g_stopped_at_limit = false;
  }
  if (parent_open) {
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
  close_dir_fds();
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
  if ((parent_open || parent_at_limit) && !parent_finalized)
    g_reopen_after_fork.store(true, std::memory_order_relaxed);
}

static void install_atfork_handlers_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    pthread_atfork(atfork_prepare, atfork_parent, atfork_child);
  });
}
#endif

static void write_manifest_stdio(int dir_fd, const std::string& output_dir, bool complete) {
  FILE* mf = fopen_private(dir_fd, output_dir, "manifest.json");
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
    // pid-<digits> only: an archive moved aside to pid-<pid>.prev-<n> is not
    // one of this capture's processes.
    const std::string name = it->path().filename().string();
    if (name.size() <= 4 || name.compare(0, 4, "pid-") != 0 ||
        name.find_first_not_of("0123456789", 4) != std::string::npos)
      continue;
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

#ifdef _WIN32
  constexpr int kBaseDir = -1;
#else
  constexpr int kBaseDir = AT_FDCWD;
#endif
  (void)atomic_write_file(kBaseDir, g_base_dir, "manifest.json", json.data(), json.size());
}

// Disk space
// Capture stops rather than fill the file system it writes to. It keeps free

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

// The file system holding `dir`: on POSIX the directory `dir_fd`, or `dir` when
// there is none.
static bool fs_space(int dir_fd, const std::string& dir, uint64_t* avail, uint64_t* total,
                     uint64_t* block = nullptr) {
  uint64_t bs = 0;
#ifdef _WIN32
  (void)dir_fd;
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
  if ((dir_fd >= 0 ? ::fstatvfs(dir_fd, &sv) : ::statvfs(dir.c_str(), &sv)) != 0) return false;
  *avail = static_cast<uint64_t>(sv.f_bavail) * sv.f_frsize;
  *total = static_cast<uint64_t>(sv.f_blocks) * sv.f_frsize;
  bs = sv.f_frsize;
#endif
  if (block != nullptr && bs != 0) *block = bs;
  return true;
}

// pid-<pid>/active tells producers outside the runtime that this process's
// capture is on. open() removes a stale one before any step that can fail and
static constexpr const char* kActiveMarker = "active";

// Flush what is buffered, close events.bin and mark the archive incomplete.
static void stop_capture_locked(const char* why, bool at_limit) {
  char reason[256];
  snprintf(reason, sizeof(reason), "stopped writing: %s", why);
  if (g_capture_stopped.exchange(true)) return;
  g_stopped_at_limit = at_limit;
  if (g_events_fd >= 0) {
    flush_buffer_locked();
    HRR_FSYNC(g_events_fd);
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
  // The application runs on, and producers would keep writing sidecars to an
  // archive that records nothing, on a file system already short of space.
  remove_in_archive(g_dir_fd.load(), g_output_dir, kActiveMarker);
  mark_incomplete(reason);
  // Not gated on AMD_LOG_LEVEL, like the refusal in open().
  fprintf(stderr, "[HRR capture] Capture stopped: %s. The archive at %s is incomplete.\n", why,
          g_output_dir.c_str());
}

static void stop_for_space(uint64_t keep_free) {
  char why[128];
  snprintf(why, sizeof(why),
           "less than %llu MiB would stay free on the file system holding the archive",
           static_cast<unsigned long long>(keep_free >> 20));
  BufWriteGuard lk;
  stop_capture_locked(why, /*at_limit=*/false);
}

// Size the reserve for the file system holding `dir`, the archive open() is
// about to publish, open as `dir_fd`. False when less than the reserve is free
static bool init_space_reserve(int dir_fd, const std::string& dir) {
  g_keep_free = 0;
  g_bytes_since_space_check = 0;
  g_bytes_reserved.store(0, std::memory_order_relaxed);
  uint64_t avail = 0, total = 0, block = kFileBlockDefault;
  if (!fs_space(dir_fd, dir, &avail, &total, &block)) return true;
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
static bool check_space(uint64_t pending) {
  {
    // Read before the free space, so a flush in between is counted twice, not missed.
    std::lock_guard<std::mutex> lk(g_file_mu);
    pending += g_buf_len;
  }
  uint64_t avail = 0, total = 0;
  if (!fs_space(g_dir_fd.load(), g_output_dir, &avail, &total)) return true;
  if (avail >= g_keep_free && avail - g_keep_free >= pending) return true;
  stop_for_space(g_keep_free);
  return false;
}

static uint64_t file_bytes(uint64_t len) {
  const uint64_t block = g_file_block_bytes.load(std::memory_order_relaxed);
  return (len + block - 1) / block * block;
}

// Size limits
// HIP_HRR_CAPTURE_MAX_MB caps what a process archive holds: events.bin, and every

static uint64_t              g_max_archive_bytes = 0;  // set by open()
static uint64_t              g_max_blob_bytes = 0;     // set by open()
static std::atomic<uint64_t> g_archive_bytes{0};
static std::atomic<bool>     g_blob_cap_reported{false};

// The limit in bytes that environment variable `name` sets in MiB, 0 unless
// set. The MiB count is parsed as the CLR flags registry parses a size_t.
static uint64_t env_limit_bytes(const char* name) {
  const char* v = std::getenv(name);
  const uint64_t mib = v ? static_cast<size_t>(std::atol(v)) : 0;
  return std::min<uint64_t>(mib, UINT64_MAX >> 20) << 20;
}

// Count `len` bytes toward the archive limit. False when they would take the
// archive past it; the caller then drops the write and calls
static bool charge_archive(uint64_t len) {
  if (g_max_archive_bytes == 0) return true;
  return g_archive_bytes.fetch_add(len, std::memory_order_relaxed) + len <=
         g_max_archive_bytes;
}

// Caller holds a BufWriteGuard.
static void stop_at_limit_locked() {
  char why[128];
  snprintf(why, sizeof(why), "the archive reached its HIP_HRR_CAPTURE_MAX_MB limit of %llu MiB",
           static_cast<unsigned long long>(g_max_archive_bytes >> 20));
  stop_capture_locked(why, /*at_limit=*/true);
}

// Leave out a blob over HIP_HRR_CAPTURE_MAX_BLOB_MB, saying so once per archive.
static void skip_oversized_blob(size_t len) {
  mark_incomplete("a copy over HIP_HRR_CAPTURE_MAX_BLOB_MB was not stored");
  if (g_blob_cap_reported.exchange(true, std::memory_order_relaxed)) return;
  fprintf(stderr,
          "[HRR capture] A copy of %llu bytes is over the HIP_HRR_CAPTURE_MAX_BLOB_MB limit of "
          "%llu MiB and was not stored. The archive is incomplete.\n",
          static_cast<unsigned long long>(len),
          static_cast<unsigned long long>(g_max_blob_bytes >> 20));
}

// Account for a file of `len` bytes and return false, having stopped the
// capture, if writing it would eat into the reserve. Must not be called with
static bool reserve_space(uint64_t len) {
  if (g_capture_stopped.load(std::memory_order_relaxed)) return false;
  len = file_bytes(len);
  if (g_max_archive_bytes != 0) {
    BufWriteGuard lk;
    // Past the trailer the write is dropped anyway, and the archive stays complete.
    if (g_events_finalized) return false;
    if (!charge_archive(len)) {
      stop_at_limit_locked();
      return false;
    }
  }
  if (g_keep_free == 0) return true;
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
static void clear_archive_paths() {
  g_base_dir.clear();
  g_output_dir.clear();
  g_manifest_path[0] = '\0';
}


// A failed open() keeps no archive path, so reopen_after_fork() does not open one
// in a forked child. When this attempt created pid-<pid>, its empty directories go
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

static bool publish_active_marker(int dir_fd, const std::string& out_dir) {
  const std::string instance = process_instance();
  if (instance.empty())
    LogPrintfWarning("[HRR capture] Cannot name this process in %s/%s, so producers will not "
                     "take the capture for active", out_dir.c_str(), kActiveMarker);
  const int fd = open_in_archive(dir_fd, out_dir, kActiveMarker);
  if (fd < 0) return false;
  const std::string line = instance + "\n";
  const bool written = write_all_fd(fd, line.data(), line.size());
  const int err = errno;
  HRR_CLOSE(fd);
  if (written) return true;
  remove_in_archive(dir_fd, out_dir, kActiveMarker);
  errno = err;
  return false;
}

bool open(const char* output_dir) {
  if (is_open()) return true;  // already open — guard against double-invocation
  if (g_capture_stopped.load(std::memory_order_relaxed)) {  // e.g. a child after fork
    clear_archive_paths();
    return false;
  }
#ifndef _WIN32
  install_atfork_handlers_once();
#endif
  // The archive is prepared in locals and published below under the writer
  // mutex and g_buf_busy. A forked child opens its archive while other threads
  g_base_dir = output_dir;
  const uint64_t pid = current_process_id();
  const uint64_t parent_pid = current_parent_process_id();
  char sub[64];
  snprintf(sub, sizeof(sub), "/pid-%llu", static_cast<unsigned long long>(pid));
  const std::string out_dir = g_base_dir + sub;

  std::error_code exists_ec;
  bool created_pid_dir = !fs::exists(fs::symlink_status(out_dir, exists_ec));
  PendingDirs dirs;
  bool dirs_ok = claim_pid_dir(out_dir, &dirs.dir);
  if (dirs_ok && !created_pid_dir && holds_earlier_archive(dirs.dir, out_dir)) {
    std::string moved_to;
    if (!move_earlier_archive_aside(out_dir, &moved_to)) {
      const int err = errno;
      LogPrintfError("[HRR capture] Cannot move the earlier archive at %s aside: %s",
                     out_dir.c_str(), strerror(err));
      fprintf(stderr, "[HRR capture] Capture disabled: cannot move the earlier archive at %s "
              "aside (%s).\n", out_dir.c_str(), strerror(err));
      return open_failed(out_dir, false);
    }
    fprintf(stderr, "[HRR capture] %s held an archive from an earlier process with this pid; "
            "moved it to %s.\n", out_dir.c_str(), moved_to.c_str());
    created_pid_dir = true;
    // On Windows claim_pid_dir() leaves dirs.dir unset.
    if (dirs.dir >= 0) HRR_CLOSE(dirs.dir);
    dirs.dir = -1;
    dirs_ok = claim_pid_dir(out_dir, &dirs.dir);
  }
  // A marker left by an earlier process with this pid goes first, once the
  // directory it sits in is known to be ours.
  if (dirs_ok) remove_in_archive(dirs.dir, out_dir, kActiveMarker);
  dirs_ok = dirs_ok &&
            claim_sub_dir(dirs.dir, out_dir + "/blobs", "blobs", &dirs.blobs) &&
            claim_sub_dir(dirs.dir, out_dir + "/code_objects", "code_objects", &dirs.code_objects);
  if (!dirs_ok) {
    const int err = errno;
    LogPrintfError("[HRR capture] Cannot use %s as a private archive directory: %s",
                   out_dir.c_str(), strerror(err));
    fprintf(stderr, "[HRR capture] Capture disabled: cannot use %s as a private archive "
            "directory (%s).\n", out_dir.c_str(), strerror(err));
    return open_failed(out_dir, created_pid_dir);
  }

  // Read once: a forked child keeps the limits its parent started with.
  static const uint64_t max_archive_bytes = env_limit_bytes("HIP_HRR_CAPTURE_MAX_MB");
  static const uint64_t max_blob_bytes = env_limit_bytes("HIP_HRR_CAPTURE_MAX_BLOB_MB");
  g_max_archive_bytes = max_archive_bytes;
  g_max_blob_bytes = max_blob_bytes;
  // The file header is the archive's first bytes.
  g_archive_bytes.store(sizeof(hrr_file_header), std::memory_order_relaxed);
  g_blob_cap_reported.store(false, std::memory_order_relaxed);

  if (!init_space_reserve(dirs.dir, out_dir)) {
    LogPrintfError("[HRR capture] Less than %llu MiB free on the file system holding %s",
                   static_cast<unsigned long long>(g_keep_free >> 20), out_dir.c_str());
    fprintf(stderr, "[HRR capture] Capture disabled: less than %llu MiB free on the file "
            "system holding %s.\n", static_cast<unsigned long long>(g_keep_free >> 20),
            out_dir.c_str());
    return open_failed(out_dir, created_pid_dir);
  }

  std::string events_path = out_dir + "/events.bin";
  std::string manifest_path = out_dir + "/manifest.json";

  const int fd = open_in_archive(dirs.dir, out_dir, "events.bin");
  if (fd < 0) {
    const int err = errno;
    LogPrintfError("[HRR capture] Failed to open %s: %s", events_path.c_str(), strerror(err));
    fprintf(stderr, "[HRR capture] Capture disabled: cannot open %s (%s).\n",
            events_path.c_str(), strerror(err));
    return open_failed(out_dir, created_pid_dir);
  }

  // open() runs once at init, or in a forked child whose atfork_child() has
  // already cleared the incomplete flag: a flag set since then is for a record
  {
    std::lock_guard<std::mutex> lk(g_unreplayable_mu);
    g_unreplayable_apis.clear();
  }
  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.clear();
    // A forked child drops the claims of the parent's writers here, not in
    // atfork_child, which does only async-signal-safe work.
    g_blob_claims.clear();
  }

  // The last step before the archive is published.
  if (!publish_active_marker(dirs.dir, out_dir)) {
    const int err = errno;
    LogPrintfError("[HRR capture] Cannot create %s/%s: %s", out_dir.c_str(), kActiveMarker,
                   strerror(err));
    fprintf(stderr, "[HRR capture] Capture disabled: cannot create %s/%s (%s).\n",
            out_dir.c_str(), kActiveMarker, strerror(err));
    HRR_CLOSE(fd);
    return open_failed(out_dir, created_pid_dir);
  }

  {
    BufWriteGuard lk;
    g_output_dir = out_dir;
#ifndef _WIN32
    // The previous archive's, if any. No blob writer is left on them: open() runs
    // only while no archive is open.
    close_dir_fds();
#endif
    g_blobs_fd.store(std::exchange(dirs.blobs, -1));
    g_code_objects_fd.store(std::exchange(dirs.code_objects, -1));
    g_dir_fd.store(std::exchange(dirs.dir, -1));
    g_pid        = pid;
    g_parent_pid = parent_pid;
    snprintf(g_manifest_path, sizeof(g_manifest_path), "%s", manifest_path.c_str());
    // Buffer/checkpoint state is reset here so it is consistent for this
    // process's pid-<pid> sub-archive.
    g_buf_len           = 0;
    g_events_since_ckpt = 0;
    g_events_finalized  = false;
    g_seq_id.store(0, std::memory_order_relaxed);
    g_event_count.store(0, std::memory_order_relaxed);
    g_blob_count.store(0, std::memory_order_relaxed);
    g_events_fd = fd;
    // Pitched host copies record only their copied rows (hip_capture.cpp,
    // write_host_rect_blob), which replay reads back by this flag.
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION,
                       static_cast<uint16_t>(HRR_FILE_FLAG_PACKED_HOST_RECTS |
                                             HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS)};
    buffer_append_locked(&fh, sizeof(fh));
  }
  return true;
}

// Caller holds g_blob_mu. A claimed write ends here even after flush() has cut
// off new ones: flush() waits for it before reading the flag, so it still counts.
static void mark_incomplete_locked(const char* reason) {
  // Record once; the loud, AMD_LOG_LEVEL-routed message is emitted by the caller
  // (e.g. serialize_kernel_launch) which has the relevant context. Here we only
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
  LogPrintfWarning(
      "[HRR capture] %s cannot be replayed: %s. The call is recorded, but "
      "replay will report it as unreplayable rather than reproduce it",
      api, reason);
}

// Staged, not listed: write_event_raw() lists the note under the same lock that
// accepts the event, so a shim still reached through a wrapped slot after
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
  std::lock_guard<std::mutex> reopen_lk(g_reopen_mu);
  g_reopen_after_fork.store(false, std::memory_order_release);
  bool incomplete;
  std::string out_dir;
  int dir_fd;
  {
    BufWriteGuard lk;
    // Must be read under the lock, see stop_capture_locked().
    incomplete = g_capture_incomplete.load(std::memory_order_relaxed);
    out_dir = g_output_dir;
    dir_fd = g_dir_fd.load();
    if (g_events_fd >= 0 && !g_events_finalized) {
      {
        // Stop new blob and code object claims, then wait for the claimed writes
        // to end: a failed one marks the capture incomplete, which decides the
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
    g_events_finalized = true;
  }

  if (out_dir.empty()) return;
  write_manifest_stdio(dir_fd, out_dir, /*complete=*/!incomplete);
  update_root_manifest();
  remove_in_archive(dir_fd, out_dir, kActiveMarker);
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
#ifndef _WIN32
  // The archive's directory descriptors go with events.bin, so the application
  // keeps none of them after the capture. Blob and code object writes use them
  {
    std::unique_lock<std::mutex> blk(g_blob_mu);
    g_events_finalized = true;
    g_blob_writes_done.wait(blk, [] { return g_blob_writes_in_flight == 0; });
  }
  while (g_emergency_manifest_busy.test_and_set(std::memory_order_acquire))
    std::this_thread::yield();
  close_dir_fds();
  g_emergency_manifest_busy.clear(std::memory_order_release);
#endif
}


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
  // An early stop has already closed events.bin, but a crash after it must still
  // leave a manifest that says the archive is incomplete.
  if (g_events_fd < 0 && (clean_shutdown || !g_capture_stopped.load(std::memory_order_relaxed)))
    return;

  // Flush the in-memory buffer only if no writer thread is mid-mutation.
  bool locked = !g_buf_busy.test_and_set(std::memory_order_acquire);
  if (locked) {
    flush_buffer_locked();
    // Clean shutdowns append the fixed-size trailer so the reader does not treat
    // the archive as crash-truncated. The CLR crash callback passes
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
  if (g_manifest_path[0] == '\0') return;
  // Concurrent crash callbacks must not share the emergency buffer. A second
  // entrant skips the manifest. close() holds the flag while it closes g_dir_fd.
  if (g_emergency_manifest_busy.test_and_set(std::memory_order_acquire)) return;
  bool complete = clean_shutdown && locked;
#ifdef _WIN32
  int mfd = HRR_OPEN(g_manifest_path);
#else
  int mfd = open_private_fd(g_dir_fd.load(), "manifest.json");
#endif
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

// write_event_raw — unified write path for all events
// hdr points to the hrr_event_header at the front of an hrr_args_* struct.

// Assign sequence_id and buffer the record atomically so IDs are only consumed for
// events that are actually written. A full record is always appended under the
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
  bool due;
  {
    BufWriteGuard lk;
    // Shims stay reachable during shutdown through slots another component wrapped,
    // and the reader would replay a record after the trailer as part of the archive.
    if (g_events_fd < 0 || g_events_finalized) return;
    if (!charge_archive(payload_len)) {
      stop_at_limit_locked();
      return;
    }
    due = g_keep_free != 0 && space_check_due_locked(payload_len);
    if (!due) append_event_locked(hdr, payload_len, staged, fresh);
  }
  if (due) {
    // Outside g_file_mu, since check_space may take it to stop the capture. The
    // record's bytes stay reserved until it is buffered, so a concurrent check counts
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

// Atomic file write: write to a temp file then rename into place.

static bool atomic_write_file(int dir_fd, const std::string& dir, const std::string& name,
                              const void* data, size_t len) {
#ifdef _WIN32
  (void)dir_fd;
  const std::string path = dir + "/" + name;
  std::string tmp = path + "." + std::to_string(current_process_id()) + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  bool ok = (fwrite(data, 1, len, f) == len);
  // fwrite() can leave the data in the stdio buffer, so the write that fails
  // may be the one fclose() makes.
  if (fclose(f) != 0) ok = false;
  if (!ok) { remove(tmp.c_str()); return false; }
  ok = MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
  if (!ok) remove(tmp.c_str());
#else
  // The claim on the key keeps every other thread off <name>.tmp.
  const std::string path = dir_fd == AT_FDCWD ? dir + "/" + name : name;
  std::string tmp = path + (dir_fd == AT_FDCWD ? ".XXXXXX.tmp" : ".tmp");
  const int fd = dir_fd == AT_FDCWD ? ::mkostemps(&tmp[0], 4, O_CLOEXEC)
                                    : open_private_fd(dir_fd, tmp.c_str());
  if (fd < 0) return false;
  // mkostemps creates the file 0600 minus the umask.
  bool ok = ::fchmod(fd, 0600) == 0 && write_all_fd(fd, data, len);
  if (::close(fd) != 0) ok = false;
  if (ok) ok = ::renameat(dir_fd, tmp.c_str(), dir_fd, path.c_str()) == 0;
  if (!ok) (void)::unlinkat(dir_fd, tmp.c_str(), 0);
#endif
  return ok;
}

// Ends a write claimed in write_blob() or write_code_object(). A failed one is
// unpublished so a later call can retry, and marks the capture incomplete: the
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


Hash128 write_blob(const void* data, size_t len, bool capped) {
  reopen_after_fork();
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0 || g_events_finalized) return {};  // not open, or past the trailer
  }
  if (capped && g_max_blob_bytes != 0 && len > g_max_blob_bytes) {
    skip_oversized_blob(len);
    return {};
  }

  Hash128 h = hash_buffer(data, len);

  char hex[33];
  hash_hex(h, hex);
  std::string key(hex);  // no prefix — plain blobs

  // A caller with the same bytes waits here until this one is done, so the space
  // is reserved once. The write counts in flight only once the space is reserved,
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
  const std::string prefix(hex, 2);
  std::string subdir = g_output_dir + "/blobs/" + prefix;
  int subdir_fd = -1;
#ifndef _WIN32
  // Checked and opened relative to blobs/ for this write only, so the capture
  // holds no descriptor per prefix.
  if (!claim_sub_dir(g_blobs_fd.load(), subdir, prefix.c_str(), &subdir_fd)) {
    // `claim` unpublishes the key and marks the capture incomplete.
    LogPrintfWarning("[HRR capture] Failed to claim blob prefix %s", prefix.c_str());
    return h;
  }
  struct CloseDir {
    int fd;
    ~CloseDir() { ::close(fd); }
  } const subdir_close{subdir_fd};
#else
  ensure_dir(subdir);
#endif

  const bool ok = atomic_write_file(subdir_fd, subdir, key + ".blob", data, len);
  if (!ok) LogPrintfWarning("[HRR capture] Failed to write blob %s", hex);
  claim.done(ok);
  return h;
}


Hash128 write_code_object(const void* image, size_t image_size) {
  reopen_after_fork();
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0 || g_events_finalized) return {};  // not open, or past the trailer
  }

  // The stored bytes, and so the hash, are the image with its .comment and
  // debug sections zeroed (see hip_capture_scrub.h).
  std::vector<uint8_t> scrubbed(static_cast<const uint8_t*>(image),
                                static_cast<const uint8_t*>(image) + image_size);
  scrub::scrub_code_object(scrubbed.data(), scrubbed.size());

  Hash128 h = hash_buffer(scrubbed.data(), image_size);
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

  const bool ok = atomic_write_file(g_code_objects_fd.load(), g_output_dir + "/code_objects",
                                    std::string(hex) + ".hsaco", scrubbed.data(), image_size);
  if (!ok) LogPrintfWarning("[HRR capture] Failed to write code object %s", hex);
  claim.done(ok);
  return h;
}


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
