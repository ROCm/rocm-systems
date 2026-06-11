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
 * small app-managed buffer (not buffered stdio), so the buffer can be flushed
 * with a single async-signal-safe write()+fsync from a fatal-signal handler
 * (see emergency_finalize). The writer also checkpoints (flush+fsync) every
 * kCheckpointEvents events to bound how much a crash can lose. A clean shutdown
 * appends an hrr_eof_record trailer; its absence marks the archive as
 * crash-truncated for the reader, which recovers all complete records.
 *
 * Thread-safety: write_event_raw() and write_blob() acquire the file mutex.
 * open()/close()/flush() are called from a single thread (init/shutdown).
 */

#include "hip_capture_writer.h"
#include "hip_capture.h"

#include "os/os.hpp"           // amd::Os::timeNanos()
#include "utils/debug.hpp"     // LogPrintfError, LogPrintfWarning, LogPrintfInfo

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>
#include <algorithm>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
// Truncate: new archive. Append: resume into existing events.bin (must NOT use _O_TRUNC).
#  define HRR_OPEN(p)          _open((p), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_OPEN_APPEND(p)   _open((p), _O_RDWR | _O_CREAT | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_WRITE(fd,b,n)    _write((fd), (b), (unsigned)(n))
#  define HRR_CLOSE(fd)        _close((fd))
#  define HRR_FSYNC(fd)        _commit((fd))

using hrr_stat_t = struct _stat64;
static int hrr_stat_file(const char* path, hrr_stat_t* st) { return _stat64(path, st); }
static int hrr_fstat_file(int fd, hrr_stat_t* st) { return _fstati64(fd, st); }
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
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/file.h>
#  include <sys/syscall.h>
#  include <pthread.h>
#  define HRR_OPEN(p)        ::open((p), O_WRONLY | O_CREAT | O_TRUNC, 0644)
#  define HRR_OPEN_APPEND(p) ::open((p), O_RDWR | O_CREAT, 0644)
#  define HRR_WRITE(fd,b,n)  ::write((fd), (b), (n))
#  define HRR_CLOSE(fd)      ::close((fd))
#  define HRR_FSYNC(fd)      ::fsync((fd))

using hrr_stat_t = struct stat;
static int hrr_stat_file(const char* path, hrr_stat_t* st) { return stat(path, st); }
static int hrr_fstat_file(int fd, hrr_stat_t* st) { return fstat(fd, st); }
static std::int64_t hrr_stat_size(const hrr_stat_t& st) {
  return static_cast<std::int64_t>(st.st_size);
}

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

// Buffer must comfortably hold the largest single record (payload_length is a
// uint16_t, so <= 65535 bytes). 256 KiB amortizes write() syscalls.
static constexpr size_t   kBufCap           = 256u * 1024u;
// Flush+fsync every N events to bound crash data loss.
static constexpr uint64_t kCheckpointEvents = 4096;
// Path buffers are filled at open() so the signal path never touches std::string.
static constexpr size_t   kPathMax          = 4096;

static std::mutex   g_file_mu;
static int          g_events_fd = -1;
// g_base_dir is the archive path requested via HIP_HRR_CAPTURE_OUTPUT.
// g_output_dir is the *effective* directory this process writes to: it equals
// g_base_dir unless another live process already holds the base archive lock,
// in which case this process is isolated into g_base_dir/pid-<pid>/ (see open()).
static std::string  g_base_dir;
static std::string  g_output_dir;
static char         g_manifest_path[kPathMax] = {0};

// App-managed write buffer for events.bin (protected by g_file_mu).
static uint8_t  g_buf[kBufCap];
static size_t   g_buf_len            = 0;
static uint64_t g_events_since_ckpt  = 0;
static bool     g_trailer_written    = false;

static std::atomic<uint64_t> g_seq_id{0};
static std::atomic<uint64_t> g_event_count{0};
static std::atomic<uint64_t> g_blob_count{0};

// In-memory set of blob hex keys already written to disk.
// Eliminates the fs::exists() stat syscall on repeated blobs (common for weight tensors).
// Protected by g_blob_mu (separate from g_file_mu to avoid head-of-line blocking).
// "co:" prefix for code objects matches the playback-side load_code_object key convention.
static std::mutex                      g_blob_mu;
static std::unordered_set<std::string> g_written_blobs;

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
// signal handler that has taken it via try_lock). Does not fsync.
static void flush_buffer_locked() {
  if (g_events_fd < 0 || g_buf_len == 0) { g_buf_len = 0; return; }
  write_all_fd(g_events_fd, g_buf, g_buf_len);
  g_buf_len = 0;
}

// Append `len` bytes of one complete record to the buffer, flushing first if it
// would not fit. Caller must hold g_file_mu.
static void buffer_append_locked(const void* data, size_t len) {
  if (g_buf_len + len > kBufCap) flush_buffer_locked();
  memcpy(g_buf + g_buf_len, data, len);
  g_buf_len += len;
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

static void ensure_dir(const std::string& path) {
  if (path.empty()) return;
  fs::create_directories(path);
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

    if (h.event_type == HRR_EOF_MARKER) {
      r.had_trailer = true;
      r.append_at = pos;
      break;
    }

    if (h.payload_length < hdr_size) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }

    r.max_seq = (h.sequence_id > r.max_seq) ? h.sequence_id : r.max_seq;
    r.count++;

    long body = static_cast<long>(h.payload_length) - static_cast<long>(hdr_size);
    if (fseek(f, body, SEEK_CUR) != 0) {
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
  FILE* f = fopen(path.c_str(), "r");
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

  std::string path = g_output_dir + "/writer_state.json";
  FILE* f = fopen(path.c_str(), "w");
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

static void index_existing_blobs_locked() {
  std::lock_guard<std::mutex> lk(g_blob_mu);
  g_written_blobs.clear();

  fs::path blobs_root = g_output_dir + "/blobs";
  if (fs::exists(blobs_root)) {
    for (const auto& ent : fs::recursive_directory_iterator(blobs_root)) {
      if (ent.is_regular_file() && ent.path().extension() == ".blob")
        g_written_blobs.insert(ent.path().stem().string());
    }
  }

  fs::path co_root = g_output_dir + "/code_objects";
  if (fs::exists(co_root)) {
    for (const auto& ent : fs::directory_iterator(co_root)) {
      if (ent.is_regular_file() && ent.path().extension() == ".hsaco")
        g_written_blobs.insert(std::string("co:") + ent.path().stem().string());
    }
  }
}

#ifndef _WIN32
static void atfork_prepare() {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_events_fd >= 0)
    flush_buffer_locked();
}

static void atfork_child() {
  std::string dir;
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd >= 0) {
      HRR_CLOSE(g_events_fd);
      g_events_fd = -1;
    }
    g_buf_len = 0;
    g_events_since_ckpt = 0;
    g_trailer_written = false;
    // Re-open from the *base* dir so isolation is re-evaluated: the parent
    // still holds the base lock, so this forked child redirects to its own
    // pid-<pid> sub-archive (mirrors the spawn path).
    dir = g_base_dir;
  }
  if (!dir.empty())
    (void)open(dir.c_str());
}

static void install_atfork_handlers_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    pthread_atfork(atfork_prepare, nullptr, atfork_child);
  });
}
#endif

static void write_manifest_stdio(const char* output_dir, bool complete) {
  std::string manifest_path = std::string(output_dir) + "/manifest.json";
  FILE* mf = fopen(manifest_path.c_str(), "w");
  if (!mf) return;
  fprintf(mf,
          "{\n"
          "  \"version\": 1,\n"
          "  \"capture_mode\": \"in-tree\",\n"
          "  \"complete\": %s,\n"
          "  \"event_count\": %llu,\n"
          "  \"blob_count\": %llu\n"
          "}\n",
          complete ? "true" : "false",
          static_cast<unsigned long long>(g_event_count.load()),
          static_cast<unsigned long long>(g_blob_count.load()));
  fclose(mf);
}

// ---------------------------------------------------------------------------
// open / close / flush / checkpoint
// ---------------------------------------------------------------------------

bool open(const char* output_dir) {
  if (g_events_fd >= 0) return true;  // already open — guard against double-invocation
#ifndef _WIN32
  install_atfork_handlers_once();
#endif
  g_base_dir   = output_dir;
  g_output_dir = g_base_dir;
  ensure_dir(g_base_dir);

  // Buffer/checkpoint state is reset here so it is consistent regardless of
  // which directory we end up writing to (base vs isolated subdir).
  g_buf_len           = 0;
  g_events_since_ckpt = 0;
  g_trailer_written   = false;

#ifndef _WIN32
  // ---------------------------------------------------------------------
  // Per-process archive isolation.
  //
  // vLLM server mode runs GPU work in a *concurrently live* EngineCore child.
  // Because HIP is already initialized in the api_server parent, vLLM uses the
  // `spawn` start method, so the child is a fresh process that inherits
  // HIP_HRR_CAPTURE_OUTPUT and finds the parent's events.bin already present.
  // Two live processes must NOT interleave into one events.bin: they keep
  // independent fds (no shared file offset) and independent in-memory event
  // counters, so the on-disk event index ends up reflecting only one writer
  // even though both wrote bytes (the original "58 MB but 4094 events" bug).
  //
  // We take an exclusive advisory lock (flock) on the base archive's
  // events.bin. The first process in wins and writes the base archive. Any
  // process that finds the lock already held by a *live* process redirects to
  // its own private sub-archive  <base>/pid-<pid>/  — a complete, self-
  // consistent archive just like an offline single-process capture.
  //
  // A *dead* prior process (the crash-resilience restart case) has released
  // its lock, so a restart re-acquires the base lock and resumes the base
  // archive exactly as before.
  // ---------------------------------------------------------------------
  {
    std::string base_events = g_base_dir + "/events.bin";
    int fd = ::open(base_events.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0) {
      // Another live process owns the base archive — isolate this one.
      HRR_CLOSE(fd);
      char sub[64];
      snprintf(sub, sizeof(sub), "/pid-%ld", static_cast<long>(getpid()));
      g_output_dir = g_base_dir + sub;
      LogPrintfInfo("[HRR capture] Base archive in use by another process; "
                    "isolating capture to %s", g_output_dir.c_str());
    } else if (fd >= 0) {
      // We own the base archive — hand this locked fd to the writer below.
      g_events_fd = fd;
    }
  }
#endif

  ensure_dir(g_output_dir);
  ensure_dir(g_output_dir + "/blobs");
  ensure_dir(g_output_dir + "/code_objects");

  std::string events_path = g_output_dir + "/events.bin";
  std::string manifest_path = g_output_dir + "/manifest.json";
  snprintf(g_manifest_path, sizeof(g_manifest_path), "%s", manifest_path.c_str());

  hrr_stat_t st{};
  bool exists = false;
  if (g_events_fd >= 0) {
    // We already hold the locked base events.bin fd.
    exists = (hrr_fstat_file(g_events_fd, &st) == 0 && hrr_stat_size(st) > 0);
  } else {
    exists = (hrr_stat_file(events_path.c_str(), &st) == 0 && hrr_stat_size(st) > 0);
  }

  if (exists) {
    if (g_events_fd < 0) {
      g_events_fd = HRR_OPEN_APPEND(events_path.c_str());
    }
    if (g_events_fd < 0) {
      LogPrintfError("[HRR capture] Failed to open %s for append", events_path.c_str());
      return false;
    }

    uint64_t next_seq = 0, ev_count = 0, bl_count = 0;
    const std::string state_path = g_output_dir + "/writer_state.json";
    const bool fast = try_load_writer_state(state_path, hrr_stat_size(st),
                                            &next_seq, &ev_count, &bl_count);

    ScanResult scan{};
    if (!fast) {
      FILE* rf = fopen(events_path.c_str(), "rb");
      if (rf) {
        scan = scan_events_for_resume(rf, hrr_stat_size(st));
        fclose(rf);
      }
      next_seq = (scan.count > 0) ? (scan.max_seq + 1) : 0;
      ev_count = scan.count;
    } else if (hrr_stat_file(events_path.c_str(), &st) == 0) {
      FILE* rf = fopen(events_path.c_str(), "rb");
      if (rf) {
        scan = scan_events_for_resume(rf, hrr_stat_size(st));
        fclose(rf);
      }
    }

    if (scan.append_at > 0 && (scan.had_trailer || scan.torn_tail)) {
      if (hrr_ftruncate_fd(g_events_fd, scan.append_at) != 0) {
        LogPrintfWarning("[HRR capture] ftruncate resume at %lld failed", (long long)scan.append_at);
      }
    }
    if (hrr_seek_end(g_events_fd) < 0) {
      LogPrintfError("[HRR capture] seek end of %s failed", events_path.c_str());
      HRR_CLOSE(g_events_fd);
      g_events_fd = -1;
      return false;
    }

    g_seq_id.store(next_seq, std::memory_order_relaxed);
    g_event_count.store(ev_count, std::memory_order_relaxed);
    if (fast)
      g_blob_count.store(bl_count, std::memory_order_relaxed);
    index_existing_blobs_locked();

    LogPrintfInfo("[HRR capture] Resumed archive at %s (events=%llu next_seq=%llu%s%s)",
                  events_path.c_str(),
                  static_cast<unsigned long long>(ev_count),
                  static_cast<unsigned long long>(next_seq),
                  scan.had_trailer ? ", stripped trailer" : "",
                  scan.torn_tail ? ", trimmed torn tail" : "");
    return true;
  }

  // Fresh archive. g_events_fd may already be the locked base events.bin
  // (size 0); otherwise open the (isolated subdir or Windows) events.bin now.
  if (g_events_fd < 0) {
#ifndef _WIN32
    g_events_fd = ::open(events_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (g_events_fd >= 0) flock(g_events_fd, LOCK_EX | LOCK_NB);
#else
    g_events_fd = HRR_OPEN(events_path.c_str());
#endif
  }
  if (g_events_fd < 0) {
    LogPrintfError("[HRR capture] Failed to open %s for writing", events_path.c_str());
    return false;
  }

  hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
  buffer_append_locked(&fh, sizeof(fh));
  return true;
}

void checkpoint() {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_events_fd < 0) return;
  flush_buffer_locked();
  HRR_FSYNC(g_events_fd);
  save_writer_state_locked();
  g_events_since_ckpt = 0;
}

void flush(const char* /*output_dir*/) {
  // Always finalize the *effective* directory this process actually wrote to
  // (g_output_dir), which may be an isolated pid-<pid> sub-archive. The caller
  // passes the base HIP_HRR_CAPTURE_OUTPUT path, which is only correct for the
  // process that owns the base archive.
  std::string out_dir;
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    out_dir = g_output_dir;
    if (g_events_fd >= 0 && !g_trailer_written) {
      hrr_eof_record rec;
      memset(&rec, 0, sizeof(rec));
      rec.hdr.event_type     = HRR_EOF_MARKER;
      rec.hdr.sequence_id    = g_seq_id.fetch_add(1, std::memory_order_relaxed);
      rec.hdr.timestamp_ns   = amd::Os::timeNanos();
      rec.hdr.thread_id      = current_thread_id();
      rec.hdr.payload_length = static_cast<uint16_t>(sizeof(rec));
      rec.total_events       = g_event_count.load();
      rec.eof_magic          = HRR_EOF_MAGIC;
      buffer_append_locked(&rec, sizeof(rec));
      flush_buffer_locked();
      HRR_FSYNC(g_events_fd);
      g_trailer_written = true;
    }
  }

  if (out_dir.empty()) return;
  write_manifest_stdio(out_dir.c_str(), /*complete=*/true);
  remove((out_dir + "/writer_state.json").c_str());
}

void close() {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_events_fd >= 0) {
    flush_buffer_locked();
    HRR_FSYNC(g_events_fd);
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
}

// ---------------------------------------------------------------------------
// emergency_finalize — async-signal-safe crash path
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

void emergency_finalize() {
  if (g_events_fd < 0) return;

  // Flush the in-memory buffer only if we can take the lock without blocking.
  // If a writer thread holds it (crash mid-append), we must NOT touch g_buf —
  // it may contain a half-formed record — so we only fsync already-written data.
  bool locked = g_file_mu.try_lock();
  if (locked) {
    flush_buffer_locked();
    g_file_mu.unlock();
  }
  HRR_FSYNC(g_events_fd);

  // Best-effort crash manifest (complete:false) via raw open/write only.
  if (g_manifest_path[0] == '\0') return;
  int mfd = HRR_OPEN(g_manifest_path);
  if (mfd < 0) return;
  char buf[256];
  size_t p = 0;
  p = append_lit(buf, p,
                 "{\n"
                 "  \"version\": 1,\n"
                 "  \"capture_mode\": \"in-tree\",\n"
                 "  \"complete\": false,\n"
                 "  \"event_count\": ");
  p += u64_to_dec(g_event_count.load(), buf + p);
  p = append_lit(buf, p, ",\n  \"blob_count\": ");
  p += u64_to_dec(g_blob_count.load(), buf + p);
  p = append_lit(buf, p, "\n}\n");
  write_all_fd(mfd, buf, p);
  HRR_FSYNC(mfd);
  HRR_CLOSE(mfd);
}

// ---------------------------------------------------------------------------
// write_event_raw — unified write path for all events
//
// hdr points to the hrr_event_header at the front of an hrr_args_* struct.
// payload_len is sizeof the full hrr_args_* struct (header + fields).
// Fills all header fields then copies the whole struct into the app buffer.
// ---------------------------------------------------------------------------

void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint16_t payload_len) {
  // Fill fields that don't require the lock (timestamp and thread_id are
  // cheap and per-thread; getting them outside the lock keeps contention low).
  hdr->event_type     = api_id;
  hdr->timestamp_ns   = amd::Os::timeNanos();
  hdr->thread_id      = current_thread_id();
  hdr->payload_length = payload_len;
  memset(hdr->reserved, 0, sizeof(hdr->reserved));

  // Acquire once: assign sequence_id and buffer the record atomically so IDs are
  // only consumed for events that are actually written. A full record is always
  // appended under the lock, so the buffer never holds a torn record — which is
  // what makes the signal-handler flush in emergency_finalize() safe.
  bool do_checkpoint = false;
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0) return;
    hdr->sequence_id = g_seq_id.fetch_add(1, std::memory_order_relaxed);
    buffer_append_locked(hdr, payload_len);
    g_event_count.fetch_add(1, std::memory_order_relaxed);
    if (++g_events_since_ckpt >= kCheckpointEvents) {
      flush_buffer_locked();
      do_checkpoint = true;
      g_events_since_ckpt = 0;
    }
  }
  // fsync outside the lock to avoid blocking other writers on the syscall.
  if (do_checkpoint) {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd >= 0) HRR_FSYNC(g_events_fd);
  }
}

// ---------------------------------------------------------------------------
// Atomic file write: write to a temp file then rename into place.
//
// g_written_blobs ensures only one thread ever reaches here for a given path,
// so there is no concurrent write to the same temp file. The rename makes the
// blob visible to readers only when fully written — a process crash mid-fwrite
// leaves only the temp file, not a partial final blob.
//
// On Windows, rename() fails when the destination already exists (unlike POSIX
// where it is atomic). Use MoveFileExA(MOVEFILE_REPLACE_EXISTING) instead.
// ---------------------------------------------------------------------------

static bool atomic_write_file(const std::string& path,
                              const void* data, size_t len) {
  std::string tmp = path + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  bool ok = (fwrite(data, 1, len, f) == len);
  fclose(f);
  if (!ok) { remove(tmp.c_str()); return false; }
#ifdef _WIN32
  ok = MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
  ok = (rename(tmp.c_str(), path.c_str()) == 0);
#endif
  if (!ok) remove(tmp.c_str());
  return ok;
}

// ---------------------------------------------------------------------------
// write_blob
// ---------------------------------------------------------------------------

Hash128 write_blob(const void* data, size_t len) {
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0) return {};  // writer not open — drop silently
  }

  Hash128 h = hash_buffer(data, len);

  char hex[33];
  hash_hex(h, hex);
  std::string key(hex);  // no prefix — plain blobs

  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    if (!g_written_blobs.insert(key).second) return h;  // already written
  }

  // blobs/<2-char-prefix>/<fullhash>.blob
  std::string subdir = g_output_dir + "/blobs/" + std::string(hex, 2);
  ensure_dir(subdir);
  std::string path = subdir + "/" + key + ".blob";

  if (atomic_write_file(path, data, len)) {
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Write failed — remove from set so a later call can retry.
    LogPrintfWarning("[HRR capture] Failed to write blob %s", hex);
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.erase(key);
  }
  return h;
}

// ---------------------------------------------------------------------------
// write_code_object
// ---------------------------------------------------------------------------

Hash128 write_code_object(const void* image, size_t image_size) {
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd < 0) return {};  // writer not open — drop silently
  }

  Hash128 h = hash_buffer(image, image_size);
  char hex[33];
  hash_hex(h, hex);
  std::string key = std::string("co:") + hex;  // namespace to match playback load_code_object key

  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    if (!g_written_blobs.insert(key).second) return h;  // already written
  }

  std::string path = g_output_dir + "/code_objects/" + hex + ".hsaco";
  if (atomic_write_file(path, image, image_size)) {
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    LogPrintfWarning("[HRR capture] Failed to write code object %s", hex);
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.erase(key);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Counters / state queries
// ---------------------------------------------------------------------------

bool     is_open()      { std::lock_guard<std::mutex> lk(g_file_mu); return g_events_fd >= 0; }
uint64_t event_count()  { return g_event_count.load(); }
uint64_t blob_count()   { return g_blob_count.load(); }

}  // namespace writer
}  // namespace hrr_cap
