// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_a0_hotswap.cpp
/// @brief Minimal eager-only HSA hook for gfx1250 B0-to-A0 translation.
///
/// @section env Environment controls
///
/// - `HSA_HOTSWAP_VERBOSE` -- debug logging to stderr. Changes what is reported,
///   never what is done. Errors are always reported, independently of this flag.
/// - `HSA_HOTSWAP_DUMP_SOURCE` -- when set to anything but `0`, a translation
///   that fails writes the source code object it refused to disk. Off by
///   default: these objects reach 212 MiB, and a failure the memo declines to
///   remember recurs on every load of the same bytes.
/// - `HSA_HOTSWAP_DUMP_DIR` -- where those artifacts go, falling back to
///   `TMPDIR` and then `/tmp`. Naming a destination does not enable capture;
///   `HSA_HOTSWAP_DUMP_SOURCE` does that.
///
/// Capture writes at most one artifact per source identity and covers at most 32
/// distinct sources per process. An out-of-resources failure is never captured:
/// copying a large input adds pressure to a system that just ran out, and the
/// bytes are not what failed.
///
/// `HSA_HOTSWAP_DISABLE` belongs to ROCr rather than this hook: it stops
/// Runtime::LoadHotswapTool() loading this library at all.

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/code_object_identity.h"
#include "rocjitsu/code/dbt/gfx1250_b0_a0_cache.h"
#include "rocjitsu/code/dbt/translation_store.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Blob = std::shared_ptr<std::vector<uint8_t>>;
using VendorReaderCreate = hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);

constexpr uint32_t kAmdAgentInfoAsicRevision = 0xA012;

using rocjitsu::CacheKey;
using rocjitsu::TranslationIdentity;
using rocjitsu::TranslationStore;

/// @brief The single configuration this hook translates under.
/// @details Shared with the ahead-of-time tool, which must derive the same keys.
constexpr TranslationIdentity kTranslationIdentity = rocjitsu::kGfx1250B0A0Identity;

/// @brief Entries produced ahead of time, which this process only reads.
///
/// @details The objects that most need caching are the large device libraries --
/// a few hundred megabytes, minutes to translate -- which no in-process cache can
/// hold across runs. This tier is what a container image or a post-install step
/// populates, on ordinary storage, once.
///
/// Read-only is a property of the tier and not a precaution: a runtime that wrote
/// here would be writing outside its own trust boundary, and the entry it left
/// would be refused by the next reader for exactly that reason.
///
/// The *new is deliberate and never freed: a load can arrive after static
/// destructors would have run, so the store has to outlive them. It stays
/// reachable through this reference, so LeakSanitizer does not report it.
TranslationStore &pretranslation_store() {
  static TranslationStore &store = *new TranslationStore(
      rocjitsu::kGfx1250B0A0Domain, rocjitsu::gfx1250_b0_a0_translator_anchor(),
      rocjitsu::shared_translation_root(rocjitsu::gfx1250_b0_a0_translator_anchor()),
      TranslationStore::Access::kReadOnly, rocjitsu::kGfx1250B0A0KeyMode);
  return store;
}

// Minimal mirror through the sole AMD loader entry intercepted by this hook.
struct VendorLoaderTable {
  void (*query_host_address)();
  void (*query_segment_descriptors)();
  void (*query_executable)();
  void (*iterate_loaded_code_objects)();
  void (*loaded_code_object_get_info)();
  VendorReaderCreate create_reader_from_file;
};
static_assert(offsetof(VendorLoaderTable, create_reader_from_file) == 5 * sizeof(void (*)()));

bool verbose_logging() {
  const char *value = std::getenv("HSA_HOTSWAP_VERBOSE");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void log(const char *format, ...) noexcept;
void log_error(const char *format, ...) noexcept;

#if defined(RJ_HOTSWAP_TEST_HOOKS)
// The state and its lock live in ordinary functions, not in the template below:
// a function-local static inside a template belongs to one instantiation, so a
// second call site with a different callable type would get its own copy.
std::mutex &test_dump_mutex() {
  static std::mutex mutex;
  return mutex;
}
std::vector<std::string> &test_dump_paths() {
  static std::vector<std::string> paths;
  return paths;
}

/// @brief Run @p fn against the recorded capture paths under their own lock.
template <typename Fn> decltype(auto) with_test_dump_paths(Fn &&fn) {
  std::lock_guard lock(test_dump_mutex());
  return fn(test_dump_paths());
}

void remember_test_dump(const char *path) noexcept {
  try {
    with_test_dump_paths([&](std::vector<std::string> &paths) { paths.emplace_back(path); });
  } catch (...) {
  }
}
#else
void remember_test_dump(const char *) noexcept {}
#endif

/// @brief Whether the operator asked for failing inputs to be written to disk.
///
/// @details Capture is opt-in because the objects are large -- RCCL's gfx1250
/// device image is 212 MiB -- and an environmental failure is deliberately not
/// memoized, so it repeats on every load of the same bytes.
bool source_capture_enabled() {
  const char *value = std::getenv("HSA_HOTSWAP_DUMP_SOURCE");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

/// @brief Directory that receives captured sources, most specific setting first.
const char *source_capture_directory() {
  for (const char *name : {"HSA_HOTSWAP_DUMP_DIR", "TMPDIR"}) {
    const char *value = std::getenv(name);
    if (value != nullptr && value[0] != '\0')
      return value;
  }
  return "/tmp";
}

// A distinct failing object is rare, but nothing about a process guarantees it:
// cap what the registries below can grow to, and stop capturing once a run has
// produced enough material to diagnose whatever is wrong.
constexpr size_t kMaxCapturedSources = 32;

/// @brief Everything the capture policy remembers for the life of the process.
struct SourceCaptureState {
  /// Source identity -> whether its artifact was completely written. An entry
  /// present but false is a capture in flight on another thread.
  std::unordered_map<uint64_t, bool> captures;
  /// Sources the disabled-capture hint has already named. Kept apart from
  /// @ref captures so the hint does not consume the artifact's one slot: an
  /// operator who reads it, exports the variable and retries must get the file.
  std::unordered_set<uint64_t> hinted;
  /// Sources whose write failure has been reported. A failed write is retried,
  /// so without this a permanently unwritable destination would report itself on
  /// every load of the same bytes.
  std::unordered_set<uint64_t> reported_write_failures;
  /// Whether the registry has already explained that it is full.
  bool reported_capacity = false;
};

// Held in ordinary functions for the reason given above test_dump_mutex().
std::mutex &capture_state_mutex() {
  static std::mutex mutex;
  return mutex;
}
SourceCaptureState &capture_state() {
  static SourceCaptureState state;
  return state;
}

/// @brief Run @p fn against the capture state under its own lock.
template <typename Fn> decltype(auto) with_capture_state(Fn &&fn) {
  std::lock_guard lock(capture_state_mutex());
  return fn(capture_state());
}

/// @returns True the first time @p source_id is offered, up to the registry cap.
bool claim_once(std::unordered_set<uint64_t> SourceCaptureState::*member,
                uint64_t source_id) noexcept {
  try {
    return with_capture_state([&](SourceCaptureState &state) {
      auto &seen = state.*member;
      if (seen.size() >= kMaxCapturedSources)
        return false;
      return seen.insert(source_id).second;
    });
  } catch (...) {
    return false;
  }
}

/// @brief What became of an attempt to claim the capture slot for one source.
enum class CaptureClaim : uint8_t {
  Granted,        ///< This call owns the write and must settle it.
  AlreadyHandled, ///< Captured already, or being written by another thread.
  RegistryFull,   ///< The per-process source cap is reached.
};

/// @brief Begin the one capture allowed for @p source_id.
///
/// @details A granted claim is settled by finish_source_capture(), so a capture
/// that could not be written -- an unwritable directory, a full filesystem --
/// can be retried once the operator fixes it rather than being refused for the
/// life of the process.
CaptureClaim begin_source_capture(uint64_t source_id) noexcept {
  try {
    return with_capture_state([&](SourceCaptureState &state) {
      if (state.captures.contains(source_id))
        return CaptureClaim::AlreadyHandled;
      if (state.captures.size() >= kMaxCapturedSources)
        return CaptureClaim::RegistryFull;
      state.captures.emplace(source_id, false);
      return CaptureClaim::Granted;
    });
  } catch (...) {
    return CaptureClaim::AlreadyHandled;
  }
}

/// @brief Settle a claim taken by begin_source_capture().
void finish_source_capture(uint64_t source_id, bool written) noexcept {
  with_capture_state([&](SourceCaptureState &state) {
    if (written)
      state.captures[source_id] = true;
    else
      state.captures.erase(source_id);
  });
}

/// @returns True the one time the full registry should explain itself.
bool claim_capacity_report() noexcept {
  return with_capture_state(
      [](SourceCaptureState &state) { return !std::exchange(state.reported_capacity, true); });
}

/// @brief Write @p bytes to a fresh file in the configured directory.
///
/// @returns True only once the artifact is completely written and closed.
bool write_captured_source(uint64_t source_id, std::span<const uint8_t> bytes) noexcept {
  // The write is retried until it succeeds, so its failure is reported once per
  // source: a destination that stays unwritable would otherwise repeat itself on
  // every load of the same bytes.
  const auto report_failure = [source_id] {
    return claim_once(&SourceCaptureState::reported_write_failures, source_id);
  };
  char path[PATH_MAX];
  const int length =
      std::snprintf(path, sizeof(path), "%s/rocjitsu-gfx1250-b0-to-a0-%016" PRIx64 "-XXXXXX.elf",
                    source_capture_directory(), source_id);
  if (length < 0 || static_cast<size_t>(length) >= sizeof(path)) {
    if (report_failure()) {
      log_error(
          "translation failed and its source code object could not be saved: path is too long "
          "source_id=fnv1a64:%016" PRIx64 "; please file a bug report",
          source_id);
    }
    return false;
  }

  const int descriptor = mkstemps(path, 4);
  if (descriptor < 0) {
    if (report_failure()) {
      log_error("translation failed and its source code object could not be saved "
                "source_id=fnv1a64:%016" PRIx64 " errno=%d; please file a bug report",
                source_id, errno);
    }
    return false;
  }
  FILE *output = fdopen(descriptor, "wb");
  if (output == nullptr) {
    const int saved_errno = errno;
    close(descriptor);
    unlink(path);
    if (report_failure()) {
      log_error("translation failed and its source code object could not be saved "
                "source_id=fnv1a64:%016" PRIx64 " path=%s errno=%d; please file a bug report",
                source_id, path, saved_errno);
    }
    return false;
  }
  const size_t written = std::fwrite(bytes.data(), 1, bytes.size(), output);
  const int close_status = std::fclose(output);
  if (written != bytes.size() || close_status != 0) {
    const int saved_errno = errno;
    unlink(path);
    if (report_failure()) {
      log_error("translation failed and its source code object could not be saved "
                "source_id=fnv1a64:%016" PRIx64
                " path=%s written=%zu expected=%zu errno=%d; please file a bug report",
                source_id, path, written, bytes.size(), saved_errno);
    }
    return false;
  }
  log_error("translation failed; source code object saved source_id=fnv1a64:%016" PRIx64
            " input_bytes=%zu path=%s; please file a bug report and attach this code object",
            source_id, bytes.size(), path);
  remember_test_dump(path);
  return true;
}

void dump_failed_source(uint64_t source_id, std::span<const uint8_t> bytes) noexcept {
  if (!source_capture_enabled()) {
    if (claim_once(&SourceCaptureState::hinted, source_id)) {
      log_error("translation failed; set HSA_HOTSWAP_DUMP_SOURCE=1 to save the source code object "
                "source_id=fnv1a64:%016" PRIx64 " input_bytes=%zu; please file a bug report",
                source_id, bytes.size());
    }
    return;
  }
  switch (begin_source_capture(source_id)) {
  case CaptureClaim::Granted:
    finish_source_capture(source_id, write_captured_source(source_id, bytes));
    return;
  case CaptureClaim::AlreadyHandled:
    return;
  case CaptureClaim::RegistryFull:
    // Silence here would look identical to a successful capture the operator
    // then cannot find.
    if (claim_capacity_report()) {
      log_error("translation failed; no source code object was saved because %zu distinct sources "
                "have already been captured source_id=fnv1a64:%016" PRIx64
                "; please file a bug report",
                kMaxCapturedSources, source_id);
    }
    return;
  }
}

void write_log(bool enabled, const char *level, const char *format, va_list args) noexcept {
  if (!enabled)
    return;
  flockfile(stderr);
  std::fputs("[hsa-hotswap-rj] ", stderr);
  if (level != nullptr)
    std::fprintf(stderr, "%s: ", level);
  std::vfprintf(stderr, format, args);
  std::fputc('\n', stderr);
  funlockfile(stderr);
}

void log(const char *format, ...) noexcept {
  va_list args;
  va_start(args, format);
  write_log(verbose_logging(), nullptr, format, args);
  va_end(args);
}

void log_error(const char *format, ...) noexcept {
  va_list args;
  va_start(args, format);
  write_log(true, "error", format, args);
  va_end(args);
}

/// @brief Report one load attempt.
///
/// @param replay True when the outcome is a remembered verdict rather than a
///        fresh one. A refusal is reported unconditionally the one time it is
///        reached, but the reuses that follow are not new failures: a device
///        library registered once per kernel and per device replays the same
///        verdict hundreds of times, and reporting each one buries the original.
void log_translation(uint64_t source_id, const char *outcome, size_t changed, size_t input_bytes,
                     size_t output_bytes, rj_status_t translation_status, hsa_status_t load_status,
                     bool replay = false) noexcept {
  const bool failed =
      translation_status != ROCJITSU_STATUS_SUCCESS || load_status != HSA_STATUS_SUCCESS;
  if ((!failed || replay) && !verbose_logging())
    return;
  flockfile(stderr);
  std::fputs("[hsa-hotswap-rj] ", stderr);
  if (failed)
    std::fputs("error: ", stderr);
  std::fprintf(stderr,
               "eager translation source_id=fnv1a64:%016" PRIx64
               " input_revision=b0 output_revision=a0 outcome=%s changed=%zu"
               " input_bytes=%zu output_bytes=%zu translation_status=%d status=%d\n",
               source_id, outcome, changed, input_bytes, output_bytes,
               static_cast<int>(translation_status), static_cast<int>(load_status));
  funlockfile(stderr);
}

void log_translation_diagnostic(uint64_t source_id,
                                const rj_gfx1250_b0_to_a0_diagnostic_t *diagnostic) noexcept {
  if (diagnostic == nullptr)
    return;
  const char *severity = diagnostic->severity != nullptr ? diagnostic->severity : "unknown";
  const char *kind = diagnostic->kind != nullptr ? diagnostic->kind : "unknown";
  const char *mnemonic = diagnostic->mnemonic != nullptr ? diagnostic->mnemonic : "";
  const char *message = diagnostic->message != nullptr ? diagnostic->message : "";

  flockfile(stderr);
  std::fprintf(stderr,
               "[hsa-hotswap-rj] %s: translation diagnostic "
               "source_id=fnv1a64:%016" PRIx64 " severity=%s kind=%s",
               severity, source_id, severity, kind);
  if (diagnostic->has_guest_offset)
    std::fprintf(stderr, " guest_offset=.text+0x%" PRIx64, diagnostic->guest_offset);
  if (mnemonic[0] != '\0')
    std::fprintf(stderr, " mnemonic=%s", mnemonic);
  if (diagnostic->required_work)
    std::fprintf(stderr, " required=%s\n", message);
  else
    std::fprintf(stderr, " message=%s\n", message);
  funlockfile(stderr);
}

template <typename Fn> hsa_status_t hsa_boundary(const char *operation, Fn &&fn) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc &) {
    log_error("operation=%s exception=std::bad_alloc status=%d", operation,
              static_cast<int>(HSA_STATUS_ERROR_OUT_OF_RESOURCES));
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  } catch (const std::exception &error) {
    log_error("operation=%s exception=std::exception what=%s status=%d", operation, error.what(),
              static_cast<int>(HSA_STATUS_ERROR));
    return HSA_STATUS_ERROR;
  } catch (...) {
    log_error("operation=%s exception=unknown status=%d", operation,
              static_cast<int>(HSA_STATUS_ERROR));
    return HSA_STATUS_ERROR;
  }
}

template <typename Fn> class ScopeGuard {
public:
  explicit ScopeGuard(Fn fn) : fn_(std::move(fn)) {}
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;
  ~ScopeGuard() noexcept {
    try {
      fn_();
    } catch (...) {
    }
  }

private:
  Fn fn_;
};

template <typename Fn> ScopeGuard<Fn> make_scope_guard(Fn fn) {
  return ScopeGuard<Fn>(std::move(fn));
}

hsa_status_t HSA_API reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *reader);
hsa_status_t HSA_API reader_create_from_memory(const void *code_object, size_t size,
                                               hsa_code_object_reader_t *reader);
hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader);
hsa_status_t HSA_API executable_destroy(hsa_executable_t executable);
hsa_status_t HSA_API load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                            hsa_code_object_reader_t reader, const char *options,
                                            hsa_loaded_code_object_t *loaded);
hsa_status_t HSA_API load_program_code_object(hsa_executable_t executable,
                                              hsa_code_object_reader_t reader, const char *options,
                                              hsa_loaded_code_object_t *loaded);
hsa_status_t HSA_API load_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                      hsa_code_object_t code_object, const char *options);
hsa_status_t HSA_API system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                      size_t table_length, void *table);

struct OriginalApi {
  decltype(hsa_code_object_reader_create_from_file) *create_file = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *create_memory = nullptr;
  decltype(hsa_code_object_reader_destroy) *destroy_reader = nullptr;
  decltype(hsa_executable_destroy) *destroy_executable = nullptr;
  decltype(hsa_executable_load_agent_code_object) *load_agent = nullptr;
  decltype(hsa_executable_load_program_code_object) *load_program = nullptr;
  decltype(hsa_executable_load_code_object) *load_deprecated = nullptr;
  decltype(hsa_system_get_major_extension_table) *get_extension_table = nullptr;
  decltype(hsa_iterate_agents) *iterate_agents = nullptr;
  decltype(hsa_agent_get_info) *agent_get_info = nullptr;
  decltype(hsa_agent_iterate_isas) *agent_iterate_isas = nullptr;
  decltype(hsa_isa_get_info_alt) *isa_get_info = nullptr;
};

/// @brief Cheap selector narrowing the memo to a few candidate entries.
///
/// @details This only ever narrows: a candidate is confirmed by comparing the
/// bytes outright, so the fingerprint has to be fast and reasonably spread, and
/// does not have to be collision resistant. Reading every byte would be neither.
/// Measured on the 212 MiB object RCCL loads: SHA-256 over it costs 586 ms
/// (0.38 GB/s) while comparing it against a candidate costs 9 ms (24.5 GB/s), so
/// hashing to avoid a comparison would cost sixty times what the comparison it
/// replaces does. Sampling a fixed 4 KiB regardless of size keeps this in the
/// microseconds and leaves the comparison to establish identity exactly.
uint64_t sample_fingerprint(std::span<const uint8_t> bytes) noexcept {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  constexpr size_t kWindows = 16;
  constexpr size_t kWindowBytes = 256;

  const size_t size = bytes.size();
  uint64_t value = kOffsetBasis ^ size;
  const size_t span = size <= kWindowBytes ? 0 : size - kWindowBytes;
  for (size_t window = 0; window < kWindows; ++window) {
    const size_t start = span * window / (kWindows - 1);
    const size_t count = std::min(kWindowBytes, size - start);
    for (size_t offset = 0; offset < count; ++offset) {
      value ^= bytes[start + offset];
      value *= kPrime;
    }
    if (span == 0)
      break;
  }
  return value;
}

/// @brief The outcome of translating one exact code object.
///
/// @details `output` is null for a refusal, which is recorded deliberately: a
/// caller that loads a bad object once loads it as many times as a good one, and
/// re-running a translation that is known to fail is the same waste as re-running
/// one that is known to succeed. `info` is kept so a reuse can reproduce the
/// original translation record rather than logging a weaker one.
struct TranslationRecord {
  Blob output;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  rj_status_t status = ROCJITSU_STATUS_SUCCESS;
};

/// @brief One remembered translation, with the source that produced it.
///
/// @details The source is retained because identity is established by comparison
/// rather than by a digest strong enough to trust on its own. That costs the
/// object's size on top of the output, which is a small price beside what it
/// buys: translating the object RCCL loads takes 197 s and peaks at 15.9 GB, so
/// an entry costs a fraction of what merely reproducing it would.
struct MemoEntry {
  uint64_t fingerprint = 0;
  Blob source;
  TranslationRecord translation;
};

using MemoList = std::list<MemoEntry>;

/// @brief A translation currently running, and the exact bytes it is running on.
///
/// @details The source is held rather than just its fingerprint because a
/// fingerprint only narrows. Two same-sized objects that differ solely outside
/// the sampled windows collide deterministically, and waiting on a fingerprint
/// would park an unrelated object behind a translation that can take minutes.
/// Whoever claims gets an @c id back so publication removes its own claim and no
/// one else's.
/// @details A finished translation too large for the memo is still handed to the
/// cohort already waiting on it, through @c result, before the claim is torn
/// down. Discarding it instead would wake every waiter to find neither a cached
/// result nor a running claim, so each would translate the object again in turn:
/// N concurrent loads of a 197 s object would cost N × 197 s, which is worse than
/// switching the memo off. @c cohort counts the waiters still owed that result, so
/// the bytes are released as soon as the last of them has taken a reference and
/// never outlive the demand that justified keeping them.
struct InFlightClaim {
  uint64_t id = 0;
  uint64_t fingerprint = 0;
  Blob source;
  TranslationRecord result;
  bool completed = false;
  size_t cohort = 0;
};

size_t memo_entry_bytes(const MemoEntry &entry) noexcept {
  return (entry.source == nullptr ? 0 : entry.source->size()) +
         (entry.translation.output == nullptr ? 0 : entry.translation.output->size());
}

// Strict ceiling on the payload the memo holds: every entry's source plus its
// output, summed, stays at or below this. Nothing is exempt -- an entry that does
// not fit is never admitted, and eviction runs to the ceiling rather than to the
// last entry. The largest real object measured is 212 MiB, so the default holds a
// couple of them, which is the shape the reported workload has.
constexpr size_t kDefaultMemoCapacity = 1024u * 1024u * 1024u;

struct HookState {
  std::mutex lifecycle_mutex;
  CoreApiTable *core = nullptr;
  // The saved lower API is published as an IMMUTABLE, never-overwritten snapshot.
  // Each install() builds a fresh const OriginalApi on the heap and publishes it;
  // uninstall() swaps in nullptr. A hot-path callback loads the shared_ptr into a
  // local, which keeps that exact table alive for the whole call even if a
  // concurrent OnUnload/OnLoad swaps a new generation in underneath -- so a
  // reinstall can never mutate the table an in-flight callback is dereferencing.
  // (Publishing only a raw pointer to a mutated-in-place member, as before, was a
  // data race across the reinstall window.)
  std::atomic<std::shared_ptr<const OriginalApi>> active_api{nullptr};
  std::atomic<VendorReaderCreate> vendor_reader{nullptr};

  std::mutex storage_mutex;
  std::unordered_map<uint64_t, Blob> readers;
  std::unordered_map<uint64_t, std::vector<Blob>> executables;

  // Translation memo. It has its own mutex rather than sharing storage_mutex
  // because a thread that finds a translation already in flight sleeps here until
  // it finishes, and it must not hold reader capture or executable teardown off
  // while it sleeps.
  //
  // `recent` is the LRU list, most recently used at the front, and `index` maps a
  // fingerprint to its candidate nodes -- several, since a fingerprint only
  // narrows. Splicing within a list keeps iterators valid, so a hit can reorder
  // the list without touching the index.
  std::mutex memo_mutex;
  std::condition_variable memo_ready;
  MemoList recent;
  std::unordered_multimap<uint64_t, MemoList::iterator> index;
  // Bounded by the number of concurrent loads, so a scan costs less than the
  // comparison each candidate needs anyway.
  std::vector<InFlightClaim> in_flight;
  uint64_t next_claim_id = 1;
  size_t memo_bytes = 0;
  size_t memo_capacity = kDefaultMemoCapacity;
  size_t memo_waiters = 0;
  std::atomic<uint64_t> translations{0};
};

// ROCr calls OnUnload before releasing this tool, so one DSO-local state owns the
// saved API and the buffers that must outlive code-object load calls.
//
// INTENTIONALLY LEAKED (never destructed): a heap object referenced by a function-
// local reference, so it has no static destructor. A plain namespace-scope object
// would run its map destructors at process exit -- and that ordering is unsafe here.
// A supported profiler startup (rocprofv3 force-configures rocprofiler and registers
// an atexit finalizer BEFORE the application's hsa_init loads this hook) means our
// state is constructed AFTER rocprofiler's. Reverse-order exit teardown would then
// destroy g_state -- freeing translated bytes -- before rocprofiler's finalizer
// processes its code-object records that alias those bytes, a use-after-free.
// RTLD_NODELETE only defers the DSO's unmap to process exit; it does not make a
// static object indestructible. Leaking the state makes the retained bytes truly
// process-lifetime: reclaimed only by the OS at exit, after all consumers are gone.
HookState &g_state = *new HookState();

bool store_reader(hsa_code_object_reader_t reader, Blob bytes) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.readers[reader.handle] = std::move(bytes);
    return true;
  } catch (...) {
    return false;
  }
}

Blob lookup_reader(hsa_code_object_reader_t reader) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    const auto it = g_state.readers.find(reader.handle);
    return it == g_state.readers.end() ? nullptr : it->second;
  } catch (...) {
    return nullptr;
  }
}

void erase_reader(hsa_code_object_reader_t reader) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.readers.erase(reader.handle);
  } catch (...) {
  }
}

bool retain(hsa_executable_t executable, Blob bytes) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.executables[executable.handle].push_back(std::move(bytes));
    return true;
  } catch (...) {
    return false;
  }
}

// Drop a single reserved blob for @p executable (matched by identity), erasing the
// map entry when its last blob is removed. Used to roll back a reservation when the
// lower load fails.
void unretain(hsa_executable_t executable, const Blob &bytes) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    const auto map_it = g_state.executables.find(executable.handle);
    if (map_it == g_state.executables.end())
      return;
    auto &buffers = map_it->second;
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      if (it->get() == bytes.get()) {
        buffers.erase(it);
        break;
      }
    }
    if (buffers.empty())
      g_state.executables.erase(map_it);
  } catch (...) {
  }
}

void release(hsa_executable_t executable) noexcept {
  try {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.executables.erase(executable.handle);
  } catch (...) {
  }
}

#if defined(RJ_HOTSWAP_TEST_HOOKS)
// Test-only: keep a completed claim alive after its cohort has drained. The
// window in which a late load can take a ready transient result is normally over
// in microseconds, so holding it open is the only way to observe what a load
// arriving inside it does.
std::atomic<bool> g_retain_completed_claims{false};
bool retain_completed_claims_for_test() {
  return g_retain_completed_claims.load(std::memory_order_relaxed);
}

// Test-only: fail one of the two allocations that admitting an entry needs, the
// way a short allocation would. Nothing else can provoke that, and what becomes
// of the cohort when it happens is the whole question. 1 is the list node, 2 the
// index node; they fail at different points and have to be exercised separately.
std::atomic<int> g_fail_next_admission{0};
void fail_admission_if_armed_for_test(int stage) {
  int armed = g_fail_next_admission.load(std::memory_order_relaxed);
  if (armed == stage &&
      g_fail_next_admission.compare_exchange_strong(armed, 0, std::memory_order_relaxed))
    throw std::bad_alloc();
}
#else
constexpr bool retain_completed_claims_for_test() { return false; }
constexpr void fail_admission_if_armed_for_test(int) {}
#endif

/// @brief What the memo did while its lock was held, to be reported once it is
/// released.
///
/// @details Fixed and scalar, so recording an event allocates nothing and cannot
/// fail. That matters more than the detail it gives up: gathering these into a
/// container put a throwing call in the middle of the very transitions this data
/// describes. An allocation failure while noting a refusal would abandon the
/// completed result its cohort was waiting for, and one while noting a
/// displacement would abort eviction with the memo still over its ceiling and
/// nothing left to retry it -- both at exactly the moment memory is scarce, which
/// is when the ceiling and the single-flight guarantee are worth the most.
/// Observing must never be able to change what is observed.
struct MemoLogSummary {
  bool refused = false;
  size_t refused_bytes = 0;
  size_t displaced_entries = 0;
  size_t displaced_bytes = 0;
  size_t held = 0;
  size_t capacity = 0;
  size_t remaining = 0;
};

void emit_memo_summary(const MemoLogSummary &summary) noexcept {
  if (summary.refused)
    log("memo refused entry bytes=%zu capacity=%zu: too large to hold", summary.refused_bytes,
        summary.capacity);
  if (summary.displaced_entries != 0)
    log("memo displaced entries=%zu bytes=%zu held=%zu capacity=%zu remaining=%zu",
        summary.displaced_entries, summary.displaced_bytes, summary.held, summary.capacity,
        summary.remaining);
}

/// @brief Drop least-recently-used translations until the memo is under its cap.
///
/// @details Evicting is always safe: a caller that is using a translation holds
/// its own reference, and a later load of the same object simply translates
/// again. Only the memo's claim on the bytes is dropped, never the bytes
/// themselves.
/// @param summary Accumulates what happened, to be reported once the caller has
/// let go of the memo. Writing to stderr here instead would hold every lookup,
/// publication and waiter behind a synchronous write to a sink this process does
/// not control -- and worse, invert against a thread that already holds the
/// stderr lock and enters an intercepted load needing this mutex.
void memo_evict_locked(MemoLogSummary &summary) {
  // A memo that has been switched off has nothing to protect and releases
  // everything, rather than sitting on one entry nothing will ever read.
  if (g_state.memo_capacity == 0) {
    g_state.recent.clear();
    g_state.index.clear();
    g_state.memo_bytes = 0;
    return;
  }
  // Otherwise evict strictly to the ceiling. An entry too large to fit was never
  // admitted (see memo_publish), so there is no case where holding one more entry
  // than the cap allows is the lesser evil -- these bytes live until the process
  // exits, and a limit that the largest entry is exempt from is not a limit.
  while (g_state.memo_bytes > g_state.memo_capacity && !g_state.recent.empty()) {
    const auto victim = std::prev(g_state.recent.end());
    const auto range = g_state.index.equal_range(victim->fingerprint);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == victim) {
        g_state.index.erase(it);
        break;
      }
    }
    const size_t dropped = memo_entry_bytes(*victim);
    g_state.memo_bytes -= dropped;
    g_state.recent.erase(victim);
    // Note it. Displacement is the one thing that silently reintroduces the cost
    // this memo exists to remove, and an operator watching a slow start-up has no
    // other way to tell a cold memo from one that is thrashing.
    ++summary.displaced_entries;
    summary.displaced_bytes += dropped;
    summary.held = g_state.memo_bytes;
    summary.capacity = g_state.memo_capacity;
    summary.remaining = g_state.recent.size();
  }
}

enum class MemoLookup {
  kHit,        ///< @p out holds a previous outcome for these exact bytes.
  kTranslate,  ///< The caller owns the digest and must publish or release it.
  kUnavailable ///< The memo could not be consulted; translate without it.
};

/// @brief Look up a translation of exactly @p source, waiting out one in flight.
///
/// @details On @ref MemoLookup::kTranslate the caller holds a claim on
/// @p fingerprint and owes exactly one @ref memo_publish or @ref memo_release;
/// until then any other thread asking for the same bytes sleeps rather than
/// duplicating the work. On @ref MemoLookup::kUnavailable nothing was claimed and
/// nothing is owed, and the caller translates as if the memo did not exist -- the
/// memo may only cost throughput, never correctness.
///
/// Candidates are compared with the lock held. That serialises other loads behind
/// a comparison, which is 9 ms for the largest object seen and only happens when
/// a fingerprint and a size both match, against a translation of minutes that the
/// comparison exists to avoid. Comparing outside the lock would mean revalidating
/// a node another thread may have evicted meanwhile, for no measurable gain.
MemoLookup memo_acquire(uint64_t fingerprint, const Blob &source, TranslationRecord &out,
                        uint64_t &claim) noexcept {
  const auto same_bytes = [&source](const Blob &other) {
    return other->size() == source->size() &&
           std::memcmp(other->data(), source->data(), source->size()) == 0;
  };
  claim = 0;
  try {
    std::unique_lock lock(g_state.memo_mutex);
    // A capacity of zero disables the memo: nothing is looked up, nothing is
    // claimed, and every load translates as it did before this existed. Only a
    // test sets that; production runs with a fixed ceiling.
    if (g_state.memo_capacity == 0)
      return MemoLookup::kUnavailable;
    for (;;) {
      const auto range = g_state.index.equal_range(fingerprint);
      for (auto it = range.first; it != range.second; ++it) {
        const MemoList::iterator node = it->second;
        if (!same_bytes(node->source))
          continue;
        out = node->translation;
        g_state.recent.splice(g_state.recent.begin(), g_state.recent, node);
        return MemoLookup::kHit;
      }
      // A claim that finished but has not finished draining its cohort still
      // holds exactly what this load wants. Take a copy of it: holding the memo
      // lock makes that safe even if the last enrolled waiter erases the claim an
      // instant later, since the copy carries its own reference to the bytes.
      // Starting a fresh translation instead -- as skipping completed claims
      // entirely would -- costs another 197 s and 15.9 GB for the object RCCL
      // loads, purely because of when this load happened to arrive.
      const auto finished = std::ranges::find_if(g_state.in_flight, [&](const InFlightClaim &e) {
        return e.completed && e.fingerprint == fingerprint && same_bytes(e.source);
      });
      if (finished != g_state.in_flight.end()) {
        out = finished->result;
        return MemoLookup::kHit;
      }

      // Wait only for a translation of exactly these bytes. Matching on the
      // fingerprint alone would park this load behind an unrelated object that
      // merely samples the same, which for a large object is minutes of lost
      // progress for work that could have run alongside it.
      const auto running = std::ranges::find_if(g_state.in_flight, [&](const InFlightClaim &entry) {
        return !entry.completed && entry.fingerprint == fingerprint && same_bytes(entry.source);
      });
      if (running == g_state.in_flight.end())
        break;

      // Enrol on that exact claim, so waking up means checking what became of it
      // rather than re-deciding which translation this load belongs to.
      const uint64_t enrolled = running->id;
      ++running->cohort;
      ++g_state.memo_waiters;
      bool served = false;
      for (;;) {
        g_state.memo_ready.wait(lock);
        const auto entry = std::ranges::find(g_state.in_flight, enrolled, &InFlightClaim::id);
        if (entry == g_state.in_flight.end())
          break; // Published to the memo, or abandoned. Either way, look again.
        if (!entry->completed)
          continue; // A different claim resolved; this one is still running.
        out = entry->result;
        served = true;
        if (--entry->cohort == 0 && !retain_completed_claims_for_test())
          g_state.in_flight.erase(entry);
        break;
      }
      --g_state.memo_waiters;
      if (served)
        return MemoLookup::kHit;
    }
    claim = g_state.next_claim_id++;
    g_state.in_flight.push_back(InFlightClaim{claim, fingerprint, source, {}, false, 0});
    return MemoLookup::kTranslate;
  } catch (...) {
    claim = 0;
    return MemoLookup::kUnavailable;
  }
}

void erase_claim_locked(uint64_t claim) {
  const auto it = std::ranges::find(g_state.in_flight, claim, &InFlightClaim::id);
  if (it != g_state.in_flight.end())
    g_state.in_flight.erase(it);
}

/// @brief Publish the outcome for a claimed fingerprint and wake anyone waiting.
void memo_publish(uint64_t claim, uint64_t fingerprint, Blob source,
                  TranslationRecord record) noexcept {
  MemoLogSummary summary;
  try {
    {
      std::lock_guard lock(g_state.memo_mutex);
      const size_t bytes = source->size() + (record.output == nullptr ? 0 : record.output->size());
      const auto held = std::ranges::find(g_state.in_flight, claim, &InFlightClaim::id);
      const bool has_cohort = held != g_state.in_flight.end() && held->cohort != 0;

      // Give the cohort something to wake onto before anything that can fail.
      // Admitting an entry allocates twice, and if either allocation is short --
      // most likely exactly when this translation has just peaked at 15.9 GB --
      // the claim is all that stands between these waiters and translating the
      // same object again one at a time. Copying the record here shares ownership
      // of bytes that already exist and allocates nothing.
      if (has_cohort) {
        held->result = record;
        held->completed = true;
      }

      // The ceiling is a promise about how much memory this takes and keeps, so an
      // entry that does not fit is declined rather than admitted and then exempted
      // from the rule. Say so: the alternative to holding it is paying the
      // translation on every load, and an operator watching that has no other way
      // to tell the two apart.
      bool admitted = false;
      if (bytes > g_state.memo_capacity) {
        summary.refused = true;
        summary.refused_bytes = bytes;
        summary.capacity = g_state.memo_capacity;
      } else {
        try {
          // The list, the index and the byte count commit together or not at all.
          // A node the index cannot reach is worse than a missing entry: eviction
          // would later subtract bytes that were never added, wrapping memo_bytes
          // and leaving capacity enforcement broken for the life of the process.
          fail_admission_if_armed_for_test(1);
          g_state.recent.push_front(MemoEntry{fingerprint, source, record});
          try {
            fail_admission_if_armed_for_test(2);
            g_state.index.emplace(fingerprint, g_state.recent.begin());
          } catch (...) {
            g_state.recent.pop_front();
            throw;
          }
          g_state.memo_bytes += bytes;
          memo_evict_locked(summary);
          admitted = true;
        } catch (...) {
          // Could not take it. The cohort above still gets its answer; a later
          // load translates again, which is the same cost as never having cached
          // it and nothing like the cost of a convoy.
        }
      }

      // Let the claim go once it has no one left to serve: either the memo now
      // answers for these bytes, or nobody was waiting on them.
      if (admitted || !has_cohort) {
        const auto entry = std::ranges::find(g_state.in_flight, claim, &InFlightClaim::id);
        if (entry != g_state.in_flight.end())
          g_state.in_flight.erase(entry);
      }
    }
  } catch (...) {
    // A claim that never completed must not survive, or every later load of these
    // bytes waits on a translation that is not coming. One that DID complete is
    // holding a result its cohort has not collected yet, and erasing that is what
    // creates the convoy -- so tell the two apart rather than dropping both.
    try {
      std::lock_guard lock(g_state.memo_mutex);
      const auto entry = std::ranges::find(g_state.in_flight, claim, &InFlightClaim::id);
      if (entry != g_state.in_flight.end() && !entry->completed)
        g_state.in_flight.erase(entry);
    } catch (...) {
    }
  }
  g_state.memo_ready.notify_all();
  emit_memo_summary(summary);
}

/// @brief Give up a claim without publishing, so waiters retry rather than hang.
void memo_release(uint64_t claim) noexcept {
  try {
    std::lock_guard lock(g_state.memo_mutex);
    erase_claim_locked(claim);
  } catch (...) {
  }
  g_state.memo_ready.notify_all();
}

#if defined(RJ_HOTSWAP_TEST_HOOKS)
// Test-only: hold a thread that has claimed a translation until it is let go.
// Single-flight is only observable while a translation is in flight, and the test
// fixture translates in under a millisecond, so without a way to stop the claimer
// there the waiters have nothing to wait for and the case would pass whether or
// not they ever blocked.
std::mutex g_gate_mutex;
std::condition_variable g_gate_open;
bool g_gate_closed = false;

// Test-only: replace the next translation's status, to reach the paths that
// depend on WHICH failure occurred. An environmental failure cannot be provoked
// on demand, and the difference between remembering one and not is the difference
// between a transient fault and a permanent refusal of valid bytes.
std::mutex g_forced_status_mutex;
bool g_forced_status_armed = false;
rj_status_t g_forced_status = ROCJITSU_STATUS_SUCCESS;

void hold_claimed_translation_for_test() {
  std::unique_lock lock(g_gate_mutex);
  g_gate_open.wait(lock, [] { return !g_gate_closed; });
}

void override_translation_status_for_test(rj_status_t &status, uint8_t *&data, size_t &size) {
  std::lock_guard lock(g_forced_status_mutex);
  if (!g_forced_status_armed)
    return;
  g_forced_status_armed = false;
  status = g_forced_status;
  if (status == ROCJITSU_STATUS_SUCCESS)
    return;
  rj_gfx1250_b0_to_a0_free(data);
  data = nullptr;
  size = 0;
}
#else
void hold_claimed_translation_for_test() {}
void override_translation_status_for_test(rj_status_t &, uint8_t *&, size_t &) {}
#endif // RJ_HOTSWAP_TEST_HOOKS

bool install(HsaApiTable *table) {
  std::lock_guard lock(g_state.lifecycle_mutex);
  if (g_state.core != nullptr) {
    log_error("OnLoad refused: hook is already installed");
    return false;
  }
  if (table == nullptr || table->core_ == nullptr) {
    log_error("OnLoad refused: missing HSA API table");
    return false;
  }

  CoreApiTable *core = table->core_;
  constexpr size_t required_size =
      offsetof(CoreApiTable, hsa_executable_load_agent_code_object_fn) +
      sizeof(CoreApiTable::hsa_executable_load_agent_code_object_fn);
  if (core->version.minor_id < required_size) {
    log_error("OnLoad refused: HSA core API table is too small size=%u required=%zu",
              core->version.minor_id, required_size);
    return false;
  }

  OriginalApi original{
      core->hsa_code_object_reader_create_from_file_fn,
      core->hsa_code_object_reader_create_from_memory_fn,
      core->hsa_code_object_reader_destroy_fn,
      core->hsa_executable_destroy_fn,
      core->hsa_executable_load_agent_code_object_fn,
      core->hsa_executable_load_program_code_object_fn,
      core->hsa_executable_load_code_object_fn,
      core->hsa_system_get_major_extension_table_fn,
      core->hsa_iterate_agents_fn,
      core->hsa_agent_get_info_fn,
      core->hsa_agent_iterate_isas_fn,
      core->hsa_isa_get_info_alt_fn,
  };
  if (original.create_file == nullptr || original.create_memory == nullptr ||
      original.destroy_reader == nullptr || original.destroy_executable == nullptr ||
      original.load_agent == nullptr || original.load_program == nullptr ||
      original.load_deprecated == nullptr || original.get_extension_table == nullptr ||
      original.iterate_agents == nullptr || original.agent_get_info == nullptr ||
      original.agent_iterate_isas == nullptr || original.isa_get_info == nullptr) {
    log_error("OnLoad refused: HSA core API table has a missing required entry");
    return false;
  }

  // Do NOT free the previous generation's translated backing storage here. Those
  // buffers can still be referenced by consumers whose lifetime is NOT bounded by
  // the HSA runtime generation -- notably rocprofiler-register, which is not in
  // tool_libs_ and finalizes its code-object records (each holding a memory_base
  // into these bytes) at process-exit atexit, not at hsa_shut_down. A reinstall
  // (next hsa_init after a shutdown that left a live executable) would otherwise
  // free bytes an old-generation profiler record still points into. The buffers are
  // therefore process-lifetime: released only at executable destroy (release()) or
  // when the process exits. RTLD_NODELETE keeps this DSO (and g_state) mapped across
  // generations, so retaining across reinstall is sound. This bounds growth by the
  // number of never-destroyed executables across runtime generations, which is
  // small; correctness (no dangling profiler/debugger pointer) takes precedence.
  // Build the immutable snapshot FIRST -- this is the only fallible step. If the
  // allocation throws, we must not have committed any install state: g_state.core
  // stays null so a later install is not permanently rejected by the "already
  // installed" guard above (the earlier order assigned core before this alloc, so a
  // throw left the hook sticky-uninstallable). A callback that already loaded the
  // previous generation's shared_ptr keeps that table alive for its whole call, so
  // publishing a fresh snapshot never mutates a table another thread dereferences.
  auto snapshot = std::make_shared<const OriginalApi>(original);

  g_state.vendor_reader.store(nullptr, std::memory_order_release);
  g_state.core = core;
  g_state.active_api.store(std::move(snapshot), std::memory_order_release);
  core->hsa_code_object_reader_create_from_file_fn = reader_create_from_file;
  core->hsa_code_object_reader_create_from_memory_fn = reader_create_from_memory;
  core->hsa_code_object_reader_destroy_fn = reader_destroy;
  core->hsa_executable_destroy_fn = executable_destroy;
  core->hsa_executable_load_agent_code_object_fn = load_agent_code_object;
  core->hsa_executable_load_program_code_object_fn = load_program_code_object;
  core->hsa_executable_load_code_object_fn = load_code_object;
  core->hsa_system_get_major_extension_table_fn = system_get_major_extension_table;
  log("installed eager gfx1250 B0-to-A0 hook");
  return true;
}

void uninstall() {
  std::lock_guard lock(g_state.lifecycle_mutex);
  CoreApiTable *core = g_state.core;
  // Swap the published snapshot out for nullptr. Any callback still holding the
  // previous shared_ptr keeps its table alive until that call returns; this only
  // stops NEW callbacks from finding a table.
  const std::shared_ptr<const OriginalApi> original =
      g_state.active_api.exchange(nullptr, std::memory_order_acq_rel);
  if (core != nullptr && original != nullptr) {
    if (core->hsa_code_object_reader_create_from_file_fn == reader_create_from_file)
      core->hsa_code_object_reader_create_from_file_fn = original->create_file;
    if (core->hsa_code_object_reader_create_from_memory_fn == reader_create_from_memory)
      core->hsa_code_object_reader_create_from_memory_fn = original->create_memory;
    if (core->hsa_code_object_reader_destroy_fn == reader_destroy)
      core->hsa_code_object_reader_destroy_fn = original->destroy_reader;
    if (core->hsa_executable_destroy_fn == executable_destroy)
      core->hsa_executable_destroy_fn = original->destroy_executable;
    if (core->hsa_executable_load_agent_code_object_fn == load_agent_code_object)
      core->hsa_executable_load_agent_code_object_fn = original->load_agent;
    if (core->hsa_executable_load_program_code_object_fn == load_program_code_object)
      core->hsa_executable_load_program_code_object_fn = original->load_program;
    if (core->hsa_executable_load_code_object_fn == load_code_object)
      core->hsa_executable_load_code_object_fn = original->load_deprecated;
    if (core->hsa_system_get_major_extension_table_fn == system_get_major_extension_table)
      core->hsa_system_get_major_extension_table_fn = original->get_extension_table;
  }
  // ROCr destroys its loader after OnUnload but before closing tool DSOs.
  // Keep code-object backing storage alive until that later DSO close.
  g_state.vendor_reader.store(nullptr, std::memory_order_release);
  g_state.core = nullptr;
}

Blob adopt_bytes(std::vector<uint8_t> &&bytes) {
  if (bytes.empty())
    return nullptr;
  try {
    return std::make_shared<std::vector<uint8_t>>(std::move(bytes));
  } catch (...) {
    return nullptr;
  }
}

Blob copy_bytes(const void *data, size_t size) {
  if (data == nullptr || size == 0)
    return nullptr;
  try {
    const auto *begin = static_cast<const uint8_t *>(data);
    return std::make_shared<std::vector<uint8_t>>(begin, begin + size);
  } catch (...) {
    return nullptr;
  }
}

Blob read_file_region(hsa_file_t file, size_t offset, size_t requested_size) {
  if (file < 0)
    return nullptr;
  struct stat file_info {};
  if (fstat(file, &file_info) != 0 || file_info.st_size <= 0)
    return nullptr;
  const size_t file_size = static_cast<size_t>(file_info.st_size);
  if (offset > file_size)
    return nullptr;
  const size_t remaining = file_size - offset;
  const size_t size = requested_size == 0 ? remaining : requested_size;
  if (size == 0 || size > remaining)
    return nullptr;

  Blob bytes;
  try {
    bytes = std::make_shared<std::vector<uint8_t>>(size);
  } catch (...) {
    return nullptr;
  }
  size_t done = 0;
  while (done < size) {
    const ssize_t count =
        pread(file, bytes->data() + done, size - done, static_cast<off_t>(offset + done));
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return nullptr;
    done += static_cast<size_t>(count);
  }
  return bytes;
}

hsa_status_t capture_reader(const OriginalApi &api, hsa_code_object_reader_t *reader, Blob bytes) {
  if (bytes != nullptr && store_reader(*reader, std::move(bytes)))
    return HSA_STATUS_SUCCESS;
  (void)api.destroy_reader(*reader);
  *reader = {};
  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

bool is_gfx1250(const Blob &bytes) {
  if (bytes == nullptr || bytes->size() < sizeof(rocjitsu::Elf64_Ehdr))
    return false;
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, bytes->data(), sizeof(header));
  return std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) == 0 &&
         header.e_ident[rocjitsu::EI_CLASS] == rocjitsu::ELFCLASS64 &&
         header.e_machine == rocjitsu::EM_AMDGPU &&
         (header.e_flags & rocjitsu::EF_AMDGPU_MACH) == rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250;
}

enum class AgentStepping { kOther, kA0, kB0OrLater, kUnknown };

AgentStepping classify_agent(const OriginalApi &api, hsa_agent_t agent) {
  auto *iterate = api.agent_iterate_isas;
  auto *get_isa_info = api.isa_get_info;
  auto *get_agent_info = api.agent_get_info;
  if (iterate == nullptr || get_isa_info == nullptr || get_agent_info == nullptr)
    return AgentStepping::kUnknown;

  struct IsaData {
    decltype(hsa_isa_get_info_alt) *get_info;
    std::string name;
    bool failed = false;
  } data{get_isa_info, {}, false};
  const hsa_status_t status = iterate(
      agent,
      [](hsa_isa_t isa, void *opaque) -> hsa_status_t {
        auto *out = static_cast<IsaData *>(opaque);
        uint32_t length = 0;
        if (out->get_info(isa, HSA_ISA_INFO_NAME_LENGTH, &length) != HSA_STATUS_SUCCESS ||
            length == 0) {
          out->failed = true;
          return HSA_STATUS_ERROR;
        }
        out->name.resize(length);
        if (out->get_info(isa, HSA_ISA_INFO_NAME, out->name.data()) != HSA_STATUS_SUCCESS) {
          out->failed = true;
          return HSA_STATUS_ERROR;
        }
        if (!out->name.empty() && out->name.back() == '\0')
          out->name.pop_back();
        return HSA_STATUS_INFO_BREAK;
      },
      &data);
  if (data.failed || status != HSA_STATUS_INFO_BREAK)
    return status == HSA_STATUS_SUCCESS ? AgentStepping::kOther : AgentStepping::kUnknown;

  std::string_view target = data.name;
  const size_t dash = target.rfind('-');
  if (dash != std::string_view::npos)
    target.remove_prefix(dash + 1);
  const size_t colon = target.find(':');
  if (colon != std::string_view::npos)
    target = target.substr(0, colon);
  if (target != "gfx1250")
    return AgentStepping::kOther;

  uint32_t revision = 0;
  if (get_agent_info(agent, static_cast<hsa_agent_info_t>(kAmdAgentInfoAsicRevision), &revision) !=
      HSA_STATUS_SUCCESS)
    return AgentStepping::kUnknown;
  return revision == 0 ? AgentStepping::kA0 : AgentStepping::kB0OrLater;
}

/// @brief Whether an agent-less load could land on a gfx1250 A0.
/// @details kUnenumerable is a separate verdict only so the refusal can name agent
/// enumeration as the cause; it is treated exactly like kPossible by callers. A tool
/// loaded ahead of this hook in HSA_TOOLS_LIB owns hsa_iterate_agents_fn (this hook
/// never patches it), so a failing enumeration usually means that tool refused, not
/// that the machine has no agents.
enum class A0Risk { kNone, kPossible, kUnenumerable };

A0Risk assess_a0_risk(const OriginalApi &api) {
  auto *iterate = api.iterate_agents;
  if (iterate == nullptr)
    return A0Risk::kUnenumerable;
  struct IterateData {
    const OriginalApi *api;
    bool found = false;
  } data{&api, false};
  const hsa_status_t status = iterate(
      [](hsa_agent_t agent, void *opaque) -> hsa_status_t {
        auto *data = static_cast<IterateData *>(opaque);
        const AgentStepping stepping = classify_agent(*data->api, agent);
        if (stepping == AgentStepping::kA0 || stepping == AgentStepping::kUnknown) {
          data->found = true;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &data);
  if (data.found)
    return A0Risk::kPossible;
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    return A0Risk::kUnenumerable;
  return A0Risk::kNone;
}

hsa_status_t load_owned_bytes(hsa_executable_t executable, hsa_agent_t agent, const Blob &bytes,
                              const char *options, hsa_loaded_code_object_t *loaded,
                              const OriginalApi &api,
                              decltype(hsa_executable_load_agent_code_object) *original_load) {
  if (bytes == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  auto *create = api.create_memory;
  auto *destroy = api.destroy_reader;
  if (create == nullptr || destroy == nullptr || original_load == nullptr)
    return HSA_STATUS_ERROR;

  hsa_code_object_reader_t owned_reader{};
  const hsa_status_t reader_status = create(bytes->data(), bytes->size(), &owned_reader);
  if (reader_status != HSA_STATUS_SUCCESS)
    return reader_status;
  auto reader_guard = make_scope_guard([&] { (void)destroy(owned_reader); });

  // RESERVE the ownership slot BEFORE entering the lower loader. ROCr aliases (does
  // not copy) the ELF pointer -- CodeObjectReaderImpl::SetMemory stores it directly
  // and the loader hands code->ElfData() to the LoadedCodeObjectImpl -- so once a
  // load succeeds, ROCr references these bytes until the executable is destroyed.
  // The map insertion is fallible (allocation); doing it AFTER a successful load
  // would mean a successful publish could be followed by a failed retain, leaving
  // ROCr with an unowned raw pointer. Reserving first makes retention a no-fail fact
  // by the time the load returns success.
  if (!retain(executable, bytes))
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // On FAILURE, drop the reservation. A failure may be a pre-publication rejection
  // (null/invalid executable, profile/ISA mismatch) where ROCr never referenced the
  // bytes, so keeping the blob would strand it under a handle no destroy releases --
  // unbounded growth on repeated invalid loads. The rejection status is not
  // distinguishable from a post-publication failure, so we drop on all failures.
  // KNOWN LIMITATION: if ROCr publishes the loaded object and then a LATER stage
  // (segment/symbol/relocation/trampoline) fails, it does not roll back the appended
  // object, so it briefly holds a pointer into bytes we drop here. Closing that
  // window needs a transactional lower loader (or a published-object query) upstream
  // in ROCr; it is not fixable from the hook without a status-guessing heuristic that
  // reintroduces the unbounded-growth bug.
  const hsa_status_t status = original_load(executable, agent, owned_reader, options, loaded);
  if (status != HSA_STATUS_SUCCESS)
    unretain(executable, bytes);
  return status;
}

hsa_status_t HSA_API reader_create_from_memory(const void *code_object, size_t size,
                                               hsa_code_object_reader_t *reader) {
  return hsa_boundary("hsa_code_object_reader_create_from_memory", [&] {
    // Reject a null output pointer before forwarding: the lower/vendor API may write
    // through it without checking, so fail fast the way ROCr's own layer does.
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->create_memory;
    const hsa_status_t status = original(code_object, size, reader);
    if (status != HSA_STATUS_SUCCESS)
      return status;

    return capture_reader(*api, reader, copy_bytes(code_object, size));
  });
}

hsa_status_t HSA_API reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t *reader) {
  return hsa_boundary("hsa_code_object_reader_create_from_file", [&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->create_file;
    const hsa_status_t status = original(file, reader);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    return capture_reader(*api, reader, read_file_region(file, 0, 0));
  });
}

hsa_status_t vendor_reader_create(hsa_file_t file, size_t offset, size_t size,
                                  hsa_code_object_reader_t *reader) {
  return hsa_boundary("amd_loader_code_object_reader_create_from_file", [&] {
    if (reader == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto original = g_state.vendor_reader.load(std::memory_order_acquire);
    if (original == nullptr)
      return HSA_STATUS_ERROR;
    const hsa_status_t status = original(file, offset, size, reader);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    return capture_reader(*api, reader, read_file_region(file, offset, size));
  });
}

hsa_status_t HSA_API system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                      size_t table_length, void *table) {
  return hsa_boundary("hsa_system_get_major_extension_table", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->get_extension_table;
    const hsa_status_t status = original(extension, version_major, table_length, table);
    constexpr size_t reader_field_end =
        offsetof(VendorLoaderTable, create_reader_from_file) + sizeof(VendorReaderCreate);
    if (status != HSA_STATUS_SUCCESS || table == nullptr || extension != HSA_EXTENSION_AMD_LOADER ||
        version_major != 1 || table_length < reader_field_end)
      return status;

    auto *loader = static_cast<VendorLoaderTable *>(table);
    auto reader = loader->create_reader_from_file;
    if (reader != nullptr) {
      auto expected = static_cast<VendorReaderCreate>(nullptr);
      g_state.vendor_reader.compare_exchange_strong(expected, reader, std::memory_order_acq_rel);
      loader->create_reader_from_file = vendor_reader_create;
    }
    return status;
  });
}

hsa_status_t HSA_API reader_destroy(hsa_code_object_reader_t reader) {
  return hsa_boundary("hsa_code_object_reader_destroy", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->destroy_reader;
    const hsa_status_t status = original(reader);
    if (status == HSA_STATUS_SUCCESS)
      erase_reader(reader);
    return status;
  });
}

hsa_status_t HSA_API executable_destroy(hsa_executable_t executable) {
  return hsa_boundary("hsa_executable_destroy", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->destroy_executable;
    const hsa_status_t status = original(executable);
    if (status == HSA_STATUS_SUCCESS)
      release(executable);
    return status;
  });
}

hsa_status_t HSA_API load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                            hsa_code_object_reader_t reader, const char *options,
                                            hsa_loaded_code_object_t *loaded) {
  return hsa_boundary("hsa_executable_load_agent_code_object", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original_load = api->load_agent;

    const AgentStepping stepping = classify_agent(*api, agent);
    if (stepping == AgentStepping::kOther || stepping == AgentStepping::kB0OrLater)
      return original_load(executable, agent, reader, options, loaded);
    if (stepping == AgentStepping::kUnknown)
      return HSA_STATUS_ERROR_INVALID_AGENT;

    Blob source = lookup_reader(reader);
    if (source == nullptr)
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
    if (!is_gfx1250(source))
      return load_owned_bytes(executable, agent, source, options, loaded, *api, original_load);

    // gfx1250 A0 and B0 code objects carry the same machine identity. Mode 2 is
    // a B0-input environment, so gfx1250 input targeting an A0 agent uses the
    // fixed B0-to-A0 profile.
    //
    // Translation is a pure function of these bytes, so a caller that registers
    // the same object repeatedly -- which is exactly what a kernel-attribute walk
    // over a large device library does, once per registration and once per device
    // -- should pay for it once. RCCL's gfx1250 device image is 212 MiB and takes
    // 197 s to translate, so a repeat that translates again does not merely run
    // slowly; it exhausts the caller's whole startup budget.
    const uint64_t fingerprint = sample_fingerprint(*source);

    TranslationRecord translation;
    uint64_t claim = 0;
    const MemoLookup lookup = memo_acquire(fingerprint, source, translation, claim);

    // A claim taken by memo_acquire() must be given back exactly once. Anything
    // that leaves this scope without resolving it would leave every later load of
    // these bytes waiting on a translation that is never coming.
    bool claim_resolved = lookup != MemoLookup::kTranslate;
    const auto claim_guard = make_scope_guard([&] {
      if (!claim_resolved)
        memo_release(claim);
    });
    const auto remember = [&](const TranslationRecord &outcome) {
      if (claim_resolved)
        return;
      claim_resolved = true;
      memo_publish(claim, fingerprint, source, outcome);
    };

    // Serve a remembered outcome, whichever way it went. Both reuses carry the
    // same record as a fresh translation so every load stays attributable to its
    // source object, but only the verbose channel sees them: the reuse itself is
    // not news, and almost every load is a reuse.
    if (lookup == MemoLookup::kHit) {
      if (translation.output == nullptr) {
        log_translation(translation.info.source_code_object_id, "reused_failure",
                        translation.info.changed_instruction_count, source->size(), 0,
                        translation.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
                        /*replay=*/true);
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      const hsa_status_t reused_status = load_owned_bytes(executable, agent, translation.output,
                                                          options, loaded, *api, original_load);
      log_translation(translation.info.source_code_object_id, "reused",
                      translation.info.changed_instruction_count, source->size(),
                      translation.output->size(), translation.status, reused_status);
      return reused_status;
    }

    // Only a load this process has not already answered reaches the store, which
    // is what keeps its cost off the repeat path: a digest over the whole source
    // and a file read, once per distinct object rather than once per load.
    //
    // The key names the translator, the configuration and the source -- never
    // where the entry lives -- so the tool that wrote it and this reader agree by
    // construction, which is the whole reason a translation performed ahead of
    // time can be found at all.
    const CacheKey cache_key = pretranslation_store().key_for(*source, kTranslationIdentity);
    std::vector<uint8_t> cached = pretranslation_store().lookup(cache_key, kTranslationIdentity);
    // Adopting the store's buffer rather than copying it again keeps a hit to one
    // allocation; the bytes have already been read once.
    const size_t cached_size = cached.size();
    if (Blob reused = adopt_bytes(std::move(cached)); reused != nullptr) {
      // Remember what the store gave us. Without this, every later load of these
      // bytes would read the file again -- the repetition the memo exists to
      // remove, moved from the translator to the filesystem.
      TranslationRecord stored;
      stored.output = reused;
      // Only when someone is listening. This walks the whole source -- about
      // 237 ms for the 212 MiB device library -- and its only consumer is a field
      // in a log line that log_translation() drops unless verbose output is on.
      // Paying that on the path whose entire purpose is to be fast, for output
      // almost nobody asks for, is the wrong trade.
      if (verbose_logging())
        stored.info.source_code_object_id =
            rocjitsu::stable_code_object_id(source->data(), source->size());
      remember(stored);

      const hsa_status_t reused_status =
          load_owned_bytes(executable, agent, reused, options, loaded, *api, original_load);
      log("reused tier=aot input_bytes=%zu output_bytes=%zu status=%d", source->size(), cached_size,
          static_cast<int>(reused_status));
      return reused_status;
    }

    uint8_t *translated_data = nullptr;
    size_t translated_size = 0;
    g_state.translations.fetch_add(1, std::memory_order_relaxed);
    hold_claimed_translation_for_test();
    translation.status = rj_gfx1250_b0_to_a0_translate(
        source->data(), source->size(), &translated_data, &translated_size, &translation.info,
        [](const rj_gfx1250_b0_to_a0_diagnostic_t *diagnostic, void *user_data) noexcept {
          const auto *info = static_cast<const rj_gfx1250_b0_to_a0_translation_info_t *>(user_data);
          log_translation_diagnostic(info->source_code_object_id, diagnostic);
        },
        &translation.info);
    override_translation_status_for_test(translation.status, translated_data, translated_size);
    if (translation.status != ROCJITSU_STATUS_SUCCESS || translated_data == nullptr ||
        translated_size == 0) {
      rj_gfx1250_b0_to_a0_free(translated_data);
      // Copying a large input to disk after an allocation failure adds pressure
      // to a system that just ran out, and the bytes are not what failed.
      if (translation.status != ROCJITSU_STATUS_OUT_OF_RESOURCES)
        dump_failed_source(translation.info.source_code_object_id, *source);
      // Remember a refusal only when it is a verdict on the bytes. The translator
      // reports an environmental failure -- an allocation it could not make, an
      // exception it could not attribute -- with a status that promises nothing
      // about the input, and recording one of those would turn a single bad moment
      // into a permanent refusal of an object that translates perfectly well.
      if (translation.status == ROCJITSU_STATUS_INVALID_CODE_OBJECT)
        remember(translation);
      log_translation(translation.info.source_code_object_id, "translation_failed",
                      translation.info.changed_instruction_count, source->size(), 0,
                      translation.status, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }

    translation.output = copy_bytes(translated_data, translated_size);
    rj_gfx1250_b0_to_a0_free(translated_data);
    if (translation.output == nullptr) {
      // Deliberately NOT remembered: running out of memory says nothing about
      // these bytes, and recording it would turn one bad moment into a permanent
      // refusal of an object that translates perfectly well.
      log_translation(translation.info.source_code_object_id, "output_copy_failed",
                      translation.info.changed_instruction_count, source->size(), translated_size,
                      translation.status, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    remember(translation);
    const hsa_status_t load_status = load_owned_bytes(executable, agent, translation.output,
                                                      options, loaded, *api, original_load);
    log_translation(translation.info.source_code_object_id, "translated",
                    translation.info.changed_instruction_count, source->size(), translated_size,
                    translation.status, load_status);
    return load_status;
  });
}

hsa_status_t HSA_API load_program_code_object(hsa_executable_t executable,
                                              hsa_code_object_reader_t reader, const char *options,
                                              hsa_loaded_code_object_t *loaded) {
  return hsa_boundary("hsa_executable_load_program_code_object", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->load_program;
    const A0Risk risk = assess_a0_risk(*api);
    // Same status either way -- callers must not have to distinguish these -- but the
    // unenumerable case is logged separately so a co-loaded tool that refuses agent
    // iteration is not misread as an A0 incompatibility.
    if (risk == A0Risk::kUnenumerable) {
      log("agent enumeration failed; refusing agent-less program load");
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    }
    if (risk == A0Risk::kPossible)
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    return original(executable, reader, options, loaded);
  });
}

hsa_status_t HSA_API load_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                      hsa_code_object_t code_object, const char *options) {
  return hsa_boundary("hsa_executable_load_code_object", [&] {
    const std::shared_ptr<const OriginalApi> api =
        g_state.active_api.load(std::memory_order_acquire);
    if (api == nullptr)
      return HSA_STATUS_ERROR;
    auto *original = api->load_deprecated;
    const AgentStepping stepping = classify_agent(*api, agent);
    if (stepping == AgentStepping::kA0 || stepping == AgentStepping::kUnknown)
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    return original(executable, agent, code_object, options);
  });
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;
  try {
    return install(table);
  } catch (const std::bad_alloc &) {
    log_error("OnLoad failed: exception=std::bad_alloc");
    return false;
  } catch (const std::exception &error) {
    log_error("OnLoad failed: exception=std::exception what=%s", error.what());
    return false;
  } catch (...) {
    log_error("OnLoad failed: exception=unknown");
    return false;
  }
}

extern "C" RJ_HOOK_EXPORT void OnUnload() {
  try {
    uninstall();
  } catch (const std::exception &error) {
    log_error("OnUnload failed: exception=std::exception what=%s", error.what());
  } catch (...) {
    log_error("OnUnload failed: exception=unknown");
  }
}

#if defined(RJ_HOTSWAP_TEST_HOOKS)
// Test-only: total number of translated backing buffers currently retained across
// all executables. Lets a unit test assert the storage-retention lifecycle --
// buffers survive OnUnload() AND a runtime-generation reinstall (install() does not
// clear them, because an old-generation profiler record may still alias them), and
// are released only at executable destroy or process exit. Never present in the
// shipped DSO: its version script exports only OnLoad/OnUnload; the testable build
// adds rj_test_*.
extern "C" RJ_HOOK_EXPORT size_t rj_test_retained_executable_buffer_count() {
  std::lock_guard lock(g_state.storage_mutex);
  size_t count = 0;
  for (const auto &[handle, buffers] : g_state.executables) {
    (void)handle;
    count += buffers.size();
  }
  return count;
}

// Test-only: drop all retained storage and every remembered translation.
// Production storage is process-lifetime (not cleared on reinstall), so a test
// fixture needs this to isolate the retention lifecycle between test cases;
// production never calls it. The memo is cleared here too, because a test that
// expects to observe a translation cannot do so once an earlier case has already
// paid for the same bytes.
extern "C" RJ_HOOK_EXPORT void rj_test_clear_retained_storage() {
  std::vector<std::string> dump_paths;
  with_test_dump_paths([&](std::vector<std::string> &paths) { dump_paths.swap(paths); });
  for (const std::string &path : dump_paths)
    (void)unlink(path.c_str());
  // The capture state is process-wide, so a case that leaves its source claimed
  // or its capacity message spent would silently change what a later one sees.
  with_capture_state([](SourceCaptureState &state) { state = {}; });
  {
    std::lock_guard lock(g_state.storage_mutex);
    g_state.readers.clear();
    g_state.executables.clear();
  }
  // Release anything a previous case armed. A gate left closed or a status left
  // forced would wedge or corrupt every case that followed, which is a far more
  // confusing failure than the one that set it.
  {
    std::lock_guard lock(g_gate_mutex);
    g_gate_closed = false;
  }
  g_gate_open.notify_all();
  g_retain_completed_claims.store(false, std::memory_order_relaxed);
  g_fail_next_admission.store(0, std::memory_order_relaxed);
  {
    std::lock_guard lock(g_forced_status_mutex);
    g_forced_status_armed = false;
  }
  std::lock_guard lock(g_state.memo_mutex);
  g_state.recent.clear();
  g_state.index.clear();
  // Claims too. A completed one left behind would answer a later case's load from
  // an earlier case's translation, which is exactly the confusion this exists to
  // prevent. No production caller can reach here with a claim outstanding.
  g_state.in_flight.clear();
  g_state.memo_bytes = 0;
  g_state.memo_capacity = kDefaultMemoCapacity;
  g_state.translations.store(0, std::memory_order_relaxed);
}

// Test-only: number of translations actually performed. The point of the memo is
// that this stays at the number of DISTINCT code objects however many times they
// are loaded, and two loads agreeing is also true when both translated -- only
// this distinguishes reuse from repetition.
extern "C" RJ_HOOK_EXPORT uint64_t rj_test_translation_count() {
  return g_state.translations.load(std::memory_order_relaxed);
}

// Test-only: set the memo's byte cap, so displacement is reachable without
// allocating the working set the production cap is sized for. Zero disables the
// memo. Production takes no configuration; the ceiling is fixed.
extern "C" RJ_HOOK_EXPORT void rj_test_set_translation_memo_capacity(uint64_t bytes) {
  MemoLogSummary summary;
  {
    std::lock_guard lock(g_state.memo_mutex);
    g_state.memo_capacity = static_cast<size_t>(bytes);
    memo_evict_locked(summary);
  }
  emit_memo_summary(summary);
}

// Test-only: stop the next thread that claims a translation, so a test can hold a
// translation in flight and observe that its peers block rather than duplicate it.
extern "C" RJ_HOOK_EXPORT void rj_test_close_translation_gate() {
  std::lock_guard lock(g_gate_mutex);
  g_gate_closed = true;
}

extern "C" RJ_HOOK_EXPORT void rj_test_open_translation_gate() {
  {
    std::lock_guard lock(g_gate_mutex);
    g_gate_closed = false;
  }
  g_gate_open.notify_all();
}

// Test-only: the fingerprint of @p bytes. A test that wants two objects which
// collide has to be able to confirm they do; constructing a pair by hand and
// assuming the sampling misses the difference would silently stop testing the
// collision the day the window layout changes.
extern "C" RJ_HOOK_EXPORT uint64_t rj_test_sample_fingerprint(const void *bytes, size_t size) {
  return sample_fingerprint({static_cast<const uint8_t *>(bytes), size});
}

// Test-only: hold completed claims open past their cohort, so a test can submit a
// load into the window where a finished result is still available to be taken.
extern "C" RJ_HOOK_EXPORT void rj_test_retain_completed_claims(bool retain) {
  g_retain_completed_claims.store(retain, std::memory_order_relaxed);
}

// Test-only: make the next admission to the memo fail as a short allocation
// would. Stage 1 is the LRU list node, stage 2 the index node; zero disarms.
extern "C" RJ_HOOK_EXPORT void rj_test_fail_next_memo_admission(int stage) {
  g_fail_next_admission.store(stage, std::memory_order_relaxed);
}

// Test-only: threads currently asleep waiting for an in-flight translation.
extern "C" RJ_HOOK_EXPORT uint64_t rj_test_translation_waiters() {
  std::lock_guard lock(g_state.memo_mutex);
  return g_state.memo_waiters;
}

// Test-only: make the next translation report @p status instead of its own.
extern "C" RJ_HOOK_EXPORT void rj_test_force_next_translation_status(int status) {
  std::lock_guard lock(g_forced_status_mutex);
  g_forced_status_armed = true;
  g_forced_status = static_cast<rj_status_t>(status);
}

// Test-only: how many source captures this process has written.
extern "C" RJ_HOOK_EXPORT uint64_t rj_test_dump_path_count() {
  return with_test_dump_paths(
      [](const std::vector<std::string> &paths) { return static_cast<uint64_t>(paths.size()); });
}

// Test-only: bytes the memo is currently holding, sources and outputs together.
extern "C" RJ_HOOK_EXPORT uint64_t rj_test_translation_memo_bytes() {
  std::lock_guard lock(g_state.memo_mutex);
  return g_state.memo_bytes;
}

// Test-only: emit a deterministic translation record so concurrent logger tests
// do not need to race the fake HSA loader state.
extern "C" RJ_HOOK_EXPORT void rj_test_log_translation(uint64_t source_id, size_t changed) {
  log_translation(source_id, "translated", changed, 64, 96, ROCJITSU_STATUS_SUCCESS,
                  HSA_STATUS_SUCCESS);
}

// Test-only: render one diagnostic without arranging a translator result solely
// to select its severity.
extern "C" RJ_HOOK_EXPORT void
rj_test_log_translation_diagnostic(uint64_t source_id,
                                   const rj_gfx1250_b0_to_a0_diagnostic_t *diagnostic) {
  log_translation_diagnostic(source_id, diagnostic);
}
// Test-only seam onto the pre-translated tier. Its production root is derived
// from the translator's install prefix, so a test can only reach it by naming
// one -- and only by counting its hits can a test show that a pre-translated
// entry is what served a load.
extern "C" RJ_HOOK_EXPORT void rj_test_set_pretranslation_root(const char *root) {
  pretranslation_store().set_root_for_test(root);
}

extern "C" RJ_HOOK_EXPORT uint64_t rj_test_pretranslation_hits() {
  return pretranslation_store().hits_for_test();
}

#endif // RJ_HOTSWAP_TEST_HOOKS
