////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/hotswap.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "core/inc/amd_hsa_code.hpp"
#include "core/inc/hotswap_gfx_query.hpp"
#include "core/inc/hsa_internal.h"
#include "core/util/os.h"

namespace rocr {
namespace hotswap {
namespace {

std::mutex g_retained_rewritten_elf_buffers_mutex;
std::unordered_map<uint64_t, std::vector<OwnedElfBuffer>> g_retained_rewritten_elf_buffers;
#ifdef ROCR_HOTSWAP_TESTING
std::atomic<bool> g_force_retarget_code_object_failure_for_testing{false};
#endif

// -- Per-code-object retarget cache -------------------------------------------
//
// Caches the output of RetargetCodeObject keyed by (code object content,
// source ISA, target ISA, entry trampoline flag). When the same code object
// is loaded for multiple GPUs of the same stepping, the COMGR retarget runs
// once; subsequent loads return a shared reference to the cached result.
// Only deterministic failures (COMGR returned an error) are cached; transient
// allocation failures are not, so a later attempt can succeed.

uint64_t FnvHash(const void* data, size_t size) {
  constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kFnvPrime;
  }
  return hash;
}

// Compute the retarget cache key from a strong full-content identity. The
// object's digest is computed once at ingestion (CodeObjectReaderImpl) and
// carried on the view, so the per-GPU loads reuse it instead of rehashing.
// Callers without a reader (tests, prewarm) leave content_digest == 0 and we
// hash the bytes here. Sampled/mtime shortcuts are intentionally avoided: they
// can alias distinct objects and serve a stale rewritten ELF.
uint64_t ComputeRetargetCacheKey(const CodeObjectView& code_object, const std::string& source_isa,
                                 const std::string& target_isa, bool entry_trampolines) {
  uint64_t identity = code_object.content_digest != 0 ? code_object.content_digest
                                                      : FnvHash(code_object.data, code_object.size);
  identity ^= FnvHash(source_isa.data(), source_isa.size()) * 31;
  identity ^= FnvHash(target_isa.data(), target_isa.size()) * 37;
  identity ^= entry_trampolines ? 0xDEADBEEF12345678ULL : 0x0ULL;
  return identity;
}

struct CachedRetargetResult {
  bool succeeded = false;
  // When true the payload lives only in the disk tier (elf_bytes is null); the
  // lookup falls through to the disk tier and mmaps it zero-copy instead of
  // holding a second in-memory copy. When elf_bytes is set (disk cache off, or
  // a memory-backed object) callers copy from it directly.
  bool on_disk = false;
  // Ref-counted so callers can grab a cheap handle under the mutex
  // and copy into the output buffer after releasing it.
  std::shared_ptr<std::vector<uint8_t>> elf_bytes;
};

constexpr size_t kMaxRetargetCacheEntries = 256;

std::mutex g_retarget_cache_mutex;
std::unordered_map<uint64_t, CachedRetargetResult> g_retarget_cache;

// -- Single-flight coordination ----------------------------------------------
//
// Multiple threads (e.g. the 4 per-GPU loads of one code object, or a bring-up
// warm-up thread racing the first real load) can miss the cache for the same
// key simultaneously. Without coordination each would run the full COMGR
// retarget. This lets exactly one thread ("the winner") run COMGR while the
// others wait, then re-check the now-populated cache. Coalescing is keyed by
// the same content/identity key as the cache itself.

struct InflightRetarget {
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
};

std::mutex g_inflight_mutex;
std::unordered_map<uint64_t, std::shared_ptr<InflightRetarget>> g_inflight;

// Single-flight winner cleanup: on any scope exit (including exceptions) drop
// the in-flight entry and wake all waiters, so a throwing retarget can't strand
// losers on the timeout-free wait. noexcept to avoid std::terminate on unwind.
struct InflightCompletion {
  uint64_t key;
  std::shared_ptr<InflightRetarget> flight;

  ~InflightCompletion() noexcept {
    {
      std::scoped_lock lock(g_inflight_mutex);
      g_inflight.erase(key);
    }
    {
      std::lock_guard<std::mutex> lk(flight->mutex);
      flight->done = true;
    }
    flight->cv.notify_all();
  }
};

// -- Disk-persistent tier for the retarget cache ------------------------------
//
// The in-memory cache above only lives for the current process, so every fresh
// launch re-pays the full COMGR retarget (dominated by disassembling a large
// .text). This tier persists the retargeted ELF to disk keyed by the same
// content hash, so repeat runs -- and unrelated processes -- skip COMGR
// entirely. Entries are namespaced by a salt derived from the loaded COMGR
// library (path + size + mtime) plus a format version, so a toolchain change
// naturally invalidates stale results. POSIX-only; a no-op on Windows.

constexpr char kDiskCacheMagic[8] = {'H', 'S', 'H', 'O', 'T', 'S', 'W', '2'};
constexpr uint32_t kDiskCacheFormatVersion = 1;

struct DiskCacheHeader {
  char magic[8];
  uint32_t format_version;
  uint32_t reserved;
  uint64_t comgr_salt;
  uint64_t payload_size;
};

bool IsDiskCacheDisabledByEnv() {
  // Default-on; only an explicit false-like value opts out.
  if (!os::IsEnvVarSet("HSA_HOTSWAP_DISK_CACHE")) {
    return false;
  }
  std::string v = os::GetEnvVar("HSA_HOTSWAP_DISK_CACHE");
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
  return v == "0" || v == "off" || v == "false" || v == "no" || v == "n" || v == "f";
}

std::string GetDiskCacheDir() {
#if defined(_WIN32) || defined(_WIN64)
  return {};
#else
  if (IsDiskCacheDisabledByEnv()) {
    return {};
  }
  if (os::IsEnvVarSet("HSA_HOTSWAP_CACHE_DIR")) {
    const std::string d = os::GetEnvVar("HSA_HOTSWAP_CACHE_DIR");
    if (!d.empty()) {
      return d;
    }
  }
  if (os::IsEnvVarSet("XDG_CACHE_HOME")) {
    const std::string d = os::GetEnvVar("XDG_CACHE_HOME");
    if (!d.empty()) {
      return d + "/rocm/hotswap";
    }
  }
  if (os::IsEnvVarSet("HOME")) {
    const std::string h = os::GetEnvVar("HOME");
    if (!h.empty()) {
      return h + "/.cache/rocm/hotswap";
    }
  }
  return {};
#endif
}

#if !defined(_WIN32) && !defined(_WIN64)
struct ComgrApi;
ComgrApi* GetComgrApi();
std::vector<std::string> GetComgrLibraryCandidates();

// Resolve the COMGR library path for the disk-cache salt WITHOUT loading COMGR.
// The salt only needs a toolchain fingerprint that is identical at store time
// and lookup time; it must not force a dlopen, which would drag COMGR (and the
// LLVM it links) onto the warm cache-hit path this cache exists to keep COMGR
// off. So we stat the same candidate library paths (derived from the runtime's
// own directory) and return the first that exists on disk, whether or not COMGR
// is loaded. Bare-name candidates that only resolve via the loader search path
// simply do not stat and are skipped. Returns empty if none is found, in which
// case the salt falls back to the format version alone -- still deterministic
// and consistent across the store and lookup, so warm hits are preserved.
std::string ResolveComgrPathForSalt() {
  for (const std::string& name : GetComgrLibraryCandidates()) {
    struct stat st = {};
    if (::stat(name.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
      return name;
    }
  }
  return {};
}

uint64_t ComgrIdentitySalt() {
  uint64_t salt = static_cast<uint64_t>(kDiskCacheFormatVersion) * 1000003ULL;
  const std::string comgr_path = ResolveComgrPathForSalt();
  if (!comgr_path.empty()) {
    salt ^= FnvHash(comgr_path.data(), comgr_path.size());
    struct stat st = {};
    if (::stat(comgr_path.c_str(), &st) == 0) {
      salt ^= static_cast<uint64_t>(st.st_size) * 2654435761ULL;
      salt ^= static_cast<uint64_t>(st.st_mtime) * 40503ULL;
    }
  }
  return salt;
}

bool MakeDirs(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  for (size_t i = 1; i < path.size(); ++i) {
    if (path[i] == '/') {
      const std::string sub = path.substr(0, i);
      if (::mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
  }
  return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string DiskCacheSubdir(const std::string& dir, uint64_t salt) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/%016llx", static_cast<unsigned long long>(salt));
  return dir + buf;
}

std::string DiskCachePath(const std::string& dir, uint64_t key, uint64_t salt) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/%016llx.co", static_cast<unsigned long long>(key));
  return DiskCacheSubdir(dir, salt) + buf;
}

// Byte budget for the on-disk cache (entries can be hundreds of MB). Spans the
// whole cache dir so stale toolchain-salt namespaces are reclaimed too.
// HSA_HOTSWAP_CACHE_MAX_BYTES overrides the default; 0 disables eviction.
uint64_t GetDiskCacheMaxBytes() {
  constexpr uint64_t kDefaultMaxBytes = 20ULL << 30;  // 20 GiB
  static const uint64_t max_bytes = []() -> uint64_t {
    if (!os::IsEnvVarSet("HSA_HOTSWAP_CACHE_MAX_BYTES")) {
      return kDefaultMaxBytes;
    }
    const std::string v = os::GetEnvVar("HSA_HOTSWAP_CACHE_MAX_BYTES");
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
    if (end == v.c_str()) {
      return kDefaultMaxBytes;  // unparseable: keep the default budget
    }
    return n == 0 ? UINT64_MAX : static_cast<uint64_t>(n);  // 0 => unbounded
  }();
  return max_bytes;
}

struct DiskCacheFileInfo {
  std::string path;
  uint64_t size = 0;
  uint64_t mtime = 0;
};

// Enumerate every published entry (dir/<salt>/<key>.co) with its size and mtime.
void CollectDiskCacheEntries(const std::string& dir, std::vector<DiskCacheFileInfo>* out) {
  DIR* top = ::opendir(dir.c_str());
  if (!top) {
    return;
  }
  for (struct dirent* sub = ::readdir(top); sub != nullptr; sub = ::readdir(top)) {
    if (sub->d_name[0] == '.') {
      continue;
    }
    const std::string subdir = dir + "/" + sub->d_name;
    struct stat sub_st = {};
    if (::stat(subdir.c_str(), &sub_st) != 0 || !S_ISDIR(sub_st.st_mode)) {
      continue;
    }
    DIR* d = ::opendir(subdir.c_str());
    if (!d) {
      continue;
    }
    for (struct dirent* e = ::readdir(d); e != nullptr; e = ::readdir(d)) {
      const std::string name = e->d_name;
      if (name.size() < 3 || name.compare(name.size() - 3, 3, ".co") != 0) {
        continue;
      }
      const std::string path = subdir + "/" + name;
      struct stat st = {};
      if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        continue;
      }
      out->push_back({path, static_cast<uint64_t>(st.st_size), static_cast<uint64_t>(st.st_mtime)});
    }
    ::closedir(d);
  }
  ::closedir(top);
}

// Evict least-recently-modified entries until the cache fits its byte budget.
// Best-effort: unlink races between processes are benign (ENOENT is ignored).
void EnforceDiskCacheQuota(const std::string& dir) {
  const uint64_t max_bytes = GetDiskCacheMaxBytes();
  if (dir.empty() || max_bytes == UINT64_MAX) {
    return;
  }
  std::vector<DiskCacheFileInfo> entries;
  CollectDiskCacheEntries(dir, &entries);
  uint64_t total = 0;
  for (const auto& e : entries) {
    total += e.size;
  }
  if (total <= max_bytes) {
    return;
  }
  std::sort(entries.begin(), entries.end(),
            [](const DiskCacheFileInfo& a, const DiskCacheFileInfo& b) {
              return a.mtime < b.mtime;  // oldest first
            });
  for (const auto& e : entries) {
    if (total <= max_bytes) {
      break;
    }
    if (::unlink(e.path.c_str()) == 0 || errno == ENOENT) {
      total -= e.size;
    }
  }
}

// Returns true iff the entry was atomically published to disk.
bool WriteDiskCache(const std::string& dir, uint64_t key, uint64_t salt, const void* data,
                    size_t size) {
  if (dir.empty() || !data || size == 0) {
    return false;
  }
  if (!MakeDirs(DiskCacheSubdir(dir, salt))) {
    return false;
  }
  const std::string final_path = DiskCachePath(dir, key, salt);
  char suffix[64];
  std::snprintf(suffix, sizeof(suffix), ".%d.%llu.tmp", static_cast<int>(::getpid()),
                static_cast<unsigned long long>(key));
  const std::string tmp_path = final_path + suffix;

  std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
  if (!f) {
    return false;
  }
  DiskCacheHeader hdr = {};
  std::memcpy(hdr.magic, kDiskCacheMagic, sizeof(hdr.magic));
  hdr.format_version = kDiskCacheFormatVersion;
  hdr.reserved = 0;
  hdr.comgr_salt = salt;
  hdr.payload_size = size;
  const bool ok =
      std::fwrite(&hdr, sizeof(hdr), 1, f) == 1 && std::fwrite(data, 1, size, f) == size;
  std::fclose(f);

  // Atomic publish: a partial writer never exposes a truncated entry.
  if (!ok || ::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
    ::remove(tmp_path.c_str());
    return false;
  }
  EnforceDiskCacheQuota(dir);
  return true;
}

// Zero-copy read of a disk-cache entry: mmap the whole file read-only, validate
// the header in place, and return an OwnedElfBuffer whose pointer aims at the
// ELF payload and whose deleter munmaps the mapping. No 578 MB heap copy and no
// eager read -- the OS pages the ELF in on demand and shares the physical pages
// across every GPU load and process that maps the same file. Returns a null
// buffer (with size 0) on any failure so the caller can fall through.
OwnedElfBuffer MmapDiskCache(const std::string& path, uint64_t salt, size_t* out_size) {
  *out_size = 0;
  OwnedElfBuffer none(nullptr, &std::free);

  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return none;
  }
  struct stat st = {};
  if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(DiskCacheHeader))) {
    ::close(fd);
    return none;
  }
  const size_t map_len = static_cast<size_t>(st.st_size);
  void* base = ::mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // mapping keeps its own reference to the file
  if (base == MAP_FAILED) {
    return none;
  }

  const auto* hdr = static_cast<const DiskCacheHeader*>(base);
  if (std::memcmp(hdr->magic, kDiskCacheMagic, sizeof(hdr->magic)) != 0 ||
      hdr->format_version != kDiskCacheFormatVersion || hdr->comgr_salt != salt ||
      hdr->payload_size == 0 || hdr->payload_size > map_len - sizeof(DiskCacheHeader)) {
    ::munmap(base, map_len);
    // Corrupt/mismatched entry: drop it so a later load recomputes.
    ::unlink(path.c_str());
    return none;
  }

  void* payload = static_cast<char*>(base) + sizeof(DiskCacheHeader);
  const size_t payload_size = static_cast<size_t>(hdr->payload_size);
  // Cheap O(1) check: payload must start with ELF magic. A full-payload hash
  // would defeat the zero-copy design, so deeper corruption is caught and
  // invalidated at load time instead.
  constexpr unsigned char kElfMagic[4] = {0x7f, 'E', 'L', 'F'};
  if (payload_size < sizeof(kElfMagic) || std::memcmp(payload, kElfMagic, sizeof(kElfMagic)) != 0) {
    ::munmap(base, map_len);
    ::unlink(path.c_str());
    return none;
  }
  // The deleter ignores its argument (the payload pointer) and unmaps the
  // captured mapping base, so the ELF pointer can be offset past the header.
  OwnedElfBuffer mapped(payload, [base, map_len](void*) { ::munmap(base, map_len); });
  *out_size = payload_size;
  return mapped;
}
#endif  // !_WIN32 && !_WIN64

constexpr char kGfx1250[] = "gfx1250";
constexpr char kGfx1250B0Feature[] = ":gfx1250-b0-specific+";
constexpr char kGfx1250A0Feature[] = ":gfx1250-b0-specific-";

enum class Gfx1250Stepping {
  kB0,
  kA0,
};

bool IsEnvFlagEnabled(const char* name) {
  if (!os::IsEnvVarSet(name)) {
    return false;
  }

  std::string value = os::GetEnvVar(name);
  if (value.empty()) {
    return false;
  }

  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value != "0" && value != "off" && value != "false" && value != "no" && value != "n" &&
      value != "f";
}

bool IsHotswapDisabledByEnv() { return IsEnvFlagEnabled("HSA_HOTSWAP_DISABLE"); }

bool AreEntryTrampolinesRequested() {
  constexpr char kEnvName[] = "AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES";
  if (!os::IsEnvVarSet(kEnvName)) {
    return kDefaultEntryTrampolinesEnabled;
  }
  return IsEnvFlagEnabled(kEnvName);
}

bool IsStrictModeRequested() { return IsEnvFlagEnabled("HSA_HOTSWAP_STRICT_MODE"); }

bool IsVerboseLoggingEnabled() {
  static const bool verbose = IsEnvFlagEnabled("HSA_HOTSWAP_VERBOSE");
  return verbose;
}

#define HOTSWAP_LOG(...)                                                                           \
  do {                                                                                             \
    if (IsVerboseLoggingEnabled()) {                                                               \
      fprintf(stderr, __VA_ARGS__);                                                                \
    }                                                                                              \
  } while (false)

struct ComgrData {
  uint64_t handle;
};

struct ComgrHotswapRewriteOptions {
  size_t size;
  uint64_t flags;
};

constexpr int kComgrStatusSuccess = 0;
constexpr int kComgrDataKindExecutable = 0x8;
constexpr uint64_t kComgrHotswapRewriteFlagEntryTrampolines = 0x1;
constexpr uint64_t kComgrHotswapRewriteFlagStrictMode = 0x2;

struct ComgrApi {
  os::LibHandle lib = nullptr;
  int (*create_data)(int kind, ComgrData* data) = nullptr;
  int (*release_data)(ComgrData data) = nullptr;
  int (*set_data)(ComgrData data, size_t size, const char* bytes) = nullptr;
  int (*get_data)(ComgrData data, size_t* size, char* bytes) = nullptr;
  int (*get_data_isa_name)(ComgrData data, size_t* size, char* isa_name) = nullptr;
  int (*hotswap_rewrite)(ComgrData input, const char* source_isa_name, const char* target_isa_name,
                         ComgrData* output) = nullptr;
  int (*hotswap_rewrite_with_options)(ComgrData input, const char* source_isa_name,
                                      const char* target_isa_name,
                                      const ComgrHotswapRewriteOptions* rewrite_options,
                                      ComgrData* output) = nullptr;
};

template <typename T> bool ResolveComgrSymbol(os::LibHandle lib, const char* name, T* symbol) {
  *symbol = reinterpret_cast<T>(os::GetExportAddress(lib, name));
  return *symbol != nullptr;
}

bool ResolveComgrApi(os::LibHandle lib, ComgrApi* api) {
  api->lib = lib;
  const bool base_api_ready = ResolveComgrSymbol(lib, "amd_comgr_create_data", &api->create_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_release_data", &api->release_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_set_data", &api->set_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_get_data", &api->get_data) &&
      ResolveComgrSymbol(lib, "amd_comgr_get_data_isa_name", &api->get_data_isa_name) &&
      ResolveComgrSymbol(lib, "amd_comgr_hotswap_rewrite", &api->hotswap_rewrite);
  if (!base_api_ready) {
    return false;
  }

  ResolveComgrSymbol(lib, "amd_comgr_hotswap_rewrite_with_options",
                     &api->hotswap_rewrite_with_options);
  return true;
}

std::string GetRuntimeLibraryDirectory() {
#if defined(_WIN32) || defined(_WIN64)
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&GetRuntimeLibraryDirectory), &module)) {
    return {};
  }

  char path[MAX_PATH] = {};
  const DWORD len = GetModuleFileNameA(module, path, sizeof(path));
  if (len == 0 || len >= sizeof(path)) {
    return {};
  }

  std::string path_str(path);
  const std::string::size_type slash = path_str.find_last_of("\\/");
  return slash == std::string::npos ? std::string{} : path_str.substr(0, slash);
#else
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&GetRuntimeLibraryDirectory), &info) == 0 || !info.dli_fname ||
      info.dli_fname[0] == '\0') {
    return {};
  }

  std::string path(info.dli_fname);
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string{} : path.substr(0, slash);
#endif
}

std::vector<std::string> GetComgrLibraryCandidates() {
  std::vector<std::string> names;
  const std::string runtime_dir = GetRuntimeLibraryDirectory();
  if (!runtime_dir.empty()) {
#if defined(_WIN32) || defined(_WIN64)
    names.push_back(runtime_dir + "\\amd_comgr.dll");
    names.push_back(runtime_dir + "\\..\\bin\\amd_comgr.dll");
#else
    names.push_back(runtime_dir + "/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/libamd_comgr.so");
    names.push_back(runtime_dir + "/../lib/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/../lib/libamd_comgr.so");
    names.push_back(runtime_dir + "/../lib64/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/../lib64/libamd_comgr.so");
#endif
  }

#if defined(_WIN32) || defined(_WIN64)
  names.push_back("amd_comgr.dll");
#else
  names.push_back("libamd_comgr.so.3");
  names.push_back("libamd_comgr.so");
#endif
  return names;
}

ComgrApi* GetComgrApi() {
  static std::once_flag once;
  static ComgrApi api;
  static bool ready = false;

  std::call_once(once, [] {
    auto try_load_comgr_api = [](const char* name) {
      if (!name || name[0] == '\0') {
        return false;
      }

      os::LibHandle lib = os::LoadLib(name);
      if (!lib) {
        return false;
      }

      if (ResolveComgrApi(lib, &api)) {
        ready = true;
        HOTSWAP_LOG("hotswap: loaded COMGR from %s\n", name);
        return true;
      }

      os::CloseLib(lib);
      api = ComgrApi{};
      return false;
    };

    for (const std::string& name : GetComgrLibraryCandidates()) {
      if (try_load_comgr_api(name.c_str())) {
        return;
      }
    }
    HOTSWAP_LOG("hotswap: COMGR hotswap entry points unavailable\n");
  });

  return ready ? &api : nullptr;
}

const char* Gfx1250SteppingFeature(Gfx1250Stepping stepping) {
  return stepping == Gfx1250Stepping::kB0 ? kGfx1250B0Feature : kGfx1250A0Feature;
}

std::string WithGfx1250SteppingFeature(const std::string& isa_name, Gfx1250Stepping stepping) {
  if (ExtractGfxTarget(isa_name) != kGfx1250 ||
      isa_name.find(kGfx1250B0Feature) != std::string::npos ||
      isa_name.find(kGfx1250A0Feature) != std::string::npos) {
    return isa_name;
  }
  return isa_name + Gfx1250SteppingFeature(stepping);
}

bool HasCandidateHotswapRewrite(const AgentGfxRevision& gfx, const RewriteOptions& options) {
  return IsHotswapSupportedGfxRevision(gfx) ||
      (options.strict_mode_enabled && gfx.gfx_target == kGfx1250) ||
      (options.entry_trampolines_enabled && IsGfx12_5Target(gfx.gfx_target));
}

std::optional<RewriteDecision> DecideHotswapRewrite(const AgentGfxRevision& gfx,
                                                    const std::string& source_isa,
                                                    const std::string& target_isa,
                                                    const RewriteOptions& options) {
  if (source_isa.empty() || target_isa.empty()) {
    return std::nullopt;
  }

  const std::string source_gfx = ExtractGfxTarget(source_isa);
  const std::string target_gfx = ExtractGfxTarget(target_isa);
  if (IsHotswapSupportedGfxRevision(gfx) && source_gfx == kGfx1250 && target_gfx == kGfx1250) {
    // Keep A0 retargeting on COMGR's legacy rewrite path. The B0 source and A0
    // target ISA features select the required instruction patches without
    // strict mode; B0 strict rewrites use hotswap_rewrite_with_options().
    RewriteDecision decision;
    decision.source_isa = WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
    decision.target_isa = WithGfx1250SteppingFeature(target_isa, Gfx1250Stepping::kA0);
    decision.rewrite_required = true;
    return decision;
  }

  const bool request_entry_trampolines = options.entry_trampolines_enabled &&
      IsGfx12_5Target(gfx.gfx_target) && IsGfx12_5Target(source_gfx);
  const bool request_strict_mode =
      options.strict_mode_enabled && gfx.gfx_target == kGfx1250 && source_gfx == kGfx1250;
  if (!request_entry_trampolines && !request_strict_mode) {
    return std::nullopt;
  }

  RewriteDecision decision;
  decision.source_isa = source_isa;
  decision.target_isa = source_isa;
  decision.request_entry_trampolines = request_entry_trampolines;
  decision.request_strict_mode = request_strict_mode;
  decision.rewrite_required = request_strict_mode;
  if (source_gfx == kGfx1250) {
    decision.source_isa = WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
    decision.target_isa = WithGfx1250SteppingFeature(source_isa, Gfx1250Stepping::kB0);
  }
  return decision;
}

}  // namespace

std::string GetCodeObjectIsaName(const void* elf_data, size_t elf_size) {
  ComgrApi* api = GetComgrApi();
  if (!api || !elf_data || elf_size == 0) {
    return {};
  }

  ComgrData data = {};
  if (api->create_data(kComgrDataKindExecutable, &data) != kComgrStatusSuccess) {
    return {};
  }

  std::string isa;
  if (api->set_data(data, elf_size, static_cast<const char*>(elf_data)) == kComgrStatusSuccess) {
    size_t isa_len = 0;
    if (api->get_data_isa_name(data, &isa_len, nullptr) == kComgrStatusSuccess && isa_len > 0) {
      isa.resize(isa_len);
      if (api->get_data_isa_name(data, &isa_len, isa.data()) == kComgrStatusSuccess) {
        if (!isa.empty() && isa.back() == '\0') {
          isa.pop_back();
        }
      } else {
        isa.clear();
      }
    }
  }

  api->release_data(data);
  return isa;
}

std::string GetCodeObjectIsaNameFromElf(const void* elf_data, size_t elf_size) {
  if (!elf_data || elf_size == 0) {
    return {};
  }
  // AmdHsaCode parses the ELF in place via libelf (no full-buffer copy) and
  // derives the canonical target-id from e_flags + the code object version,
  // exactly as the loader does when matching against the agent ISA.
  amd::hsa::code::AmdHsaCode code;
  if (!code.InitAsBuffer(elf_data, elf_size)) {
    return {};
  }
  std::string isa;
  unsigned generic_version = 0;
  if (!code.GetIsa(isa, &generic_version)) {
    return {};
  }
  return isa;
}

namespace {

bool IsAgentEligibleForHotswap(const AgentGfxRevision& gfx, const RewriteOptions& options) {
  HOTSWAP_LOG("hotswap: agent gfx=%s asic_revision=%u (valid=%s)\n",
              gfx.gfx_target.empty() ? "?" : gfx.gfx_target.c_str(), gfx.asic_revision,
              gfx.has_asic_revision ? "yes" : "no");
  return HasCandidateHotswapRewrite(gfx, options);
}

void LogRewrittenCodeObjectLoadFailure(hsa_status_t status) {
  HOTSWAP_LOG(
      "hotswap: rewritten load failed (status=%d), falling back to "
      "original code object\n",
      static_cast<int>(status));
}

void LogRequiredRewriteFailure() {
  HOTSWAP_LOG(
      "hotswap: required rewrite failed, not falling back to original "
      "code object\n");
}

void LogRequiredRewrittenLoadFailure(hsa_status_t status) {
  HOTSWAP_LOG(
      "hotswap: required rewritten load failed (status=%d), not falling "
      "back to original code object\n",
      static_cast<int>(status));
}

// Run the COMGR retarget for a cache miss and populate both cache tiers.
// Called by exactly one thread per key (single-flight winner). On success,
// when the result was persisted to disk the in-memory tier stores only an
// on-disk marker (no second heap copy) so other loads mmap the shared file.
bool PerformRetargetAndCache(const CodeObjectView& code_object, const RewriteDecision& decision,
                             uint64_t cache_key, const std::string& disk_cache_dir,
                             uint64_t disk_cache_salt, OwnedElfBuffer* out_elf_buffer,
                             size_t* out_elf_size) {
  bool rewritten = false;
  // Transient by default so a forced test failure isn't negative-cached.
  RetargetFailureKind failure = RetargetFailureKind::kTransient;
#ifdef ROCR_HOTSWAP_TESTING
  if (g_force_retarget_code_object_failure_for_testing.load(std::memory_order_relaxed)) {
    HOTSWAP_LOG("hotswap: forcing retarget failure for test\n");
  } else
#endif
  {
    rewritten = RetargetCodeObject(code_object.data, code_object.size, decision.source_isa.c_str(),
                                   decision.target_isa.c_str(), out_elf_buffer, out_elf_size,
                                   decision.request_entry_trampolines, decision.request_strict_mode,
                                   &failure);
  }

  bool persisted_to_disk = false;
#if !defined(_WIN32) && !defined(_WIN64)
  if (rewritten && !disk_cache_dir.empty()) {
    persisted_to_disk = WriteDiskCache(disk_cache_dir, cache_key, disk_cache_salt,
                                       (*out_elf_buffer).get(), *out_elf_size);
    if (persisted_to_disk) {
      HOTSWAP_LOG("hotswap: disk cache store src=%s tgt=%s in=%zu out=%zu\n",
                  decision.source_isa.c_str(), decision.target_isa.c_str(), code_object.size,
                  *out_elf_size);
    }
  }
#else
  (void)disk_cache_dir;
  (void)disk_cache_salt;
#endif

  // Cache successes and deterministic failures; leave transient failures
  // uncached so a later load can retry.
  const bool cacheable = rewritten || failure == RetargetFailureKind::kDeterministic;
  try {
    std::scoped_lock lock(g_retarget_cache_mutex);
    if (cacheable && g_retarget_cache.find(cache_key) == g_retarget_cache.end() &&
        g_retarget_cache.size() < kMaxRetargetCacheEntries) {
      CachedRetargetResult entry;
      entry.succeeded = rewritten;
      if (rewritten && persisted_to_disk) {
        // Bytes live on disk; other loads mmap them zero-copy.
        entry.on_disk = true;
      } else if (rewritten) {
        const auto* data = static_cast<const uint8_t*>((*out_elf_buffer).get());
        entry.elf_bytes = std::make_shared<std::vector<uint8_t>>(data, data + *out_elf_size);
      }
      g_retarget_cache.emplace(cache_key, std::move(entry));
    }
  } catch (const std::bad_alloc&) {
    HOTSWAP_LOG(
        "hotswap: retarget cache store skipped (out of memory); "
        "returning uncached result for src=%s tgt=%s\n",
        decision.source_isa.c_str(), decision.target_isa.c_str());
  }

  HOTSWAP_LOG(
      "hotswap: rewrite src=%s tgt=%s entry_trampolines=%d strict=%d required=%d "
      "in=%zu out=%zu changed=%d\n",
      decision.source_isa.c_str(), decision.target_isa.c_str(), decision.request_entry_trampolines,
      decision.request_strict_mode, decision.rewrite_required, code_object.size,
      rewritten ? *out_elf_size : 0, rewritten ? 1 : 0);
  return rewritten;
}

}  // namespace

bool RetargetCodeObject(const void* elf_data, size_t elf_size, const char* source_isa,
                        const char* target_isa, OwnedElfBuffer* out_elf_buffer,
                        size_t* out_elf_size, bool request_entry_trampolines,
                        bool request_strict_mode, RetargetFailureKind* out_failure) {
  // Transient unless we reach the COMGR rewrite call; only that failure is
  // deterministic (safe to negative-cache).
  const auto set_failure = [out_failure](RetargetFailureKind kind) {
    if (out_failure) {
      *out_failure = kind;
    }
  };
  set_failure(RetargetFailureKind::kTransient);

  ComgrApi* api = GetComgrApi();
  if (!api || !elf_data || elf_size == 0 || !source_isa || !target_isa || !out_elf_buffer ||
      !out_elf_size) {
    return false;
  }

  ComgrData input = {};
  if (api->create_data(kComgrDataKindExecutable, &input) != kComgrStatusSuccess) {
    return false;
  }

  if (api->set_data(input, elf_size, static_cast<const char*>(elf_data)) != kComgrStatusSuccess) {
    api->release_data(input);
    return false;
  }

  ComgrData output = {};
  int status = kComgrStatusSuccess;
  const uint64_t rewrite_flags =
      (request_entry_trampolines ? kComgrHotswapRewriteFlagEntryTrampolines : 0) |
      (request_strict_mode ? kComgrHotswapRewriteFlagStrictMode : 0);
  if (rewrite_flags != 0) {
    if (!api->hotswap_rewrite_with_options) {
      api->release_data(input);
      HOTSWAP_LOG("hotswap: COMGR rewrite-with-options entry point unavailable\n");
      return false;
    }
    const ComgrHotswapRewriteOptions options{sizeof(ComgrHotswapRewriteOptions), rewrite_flags};
    status = api->hotswap_rewrite_with_options(input, source_isa, target_isa, &options, &output);
  } else {
    status = api->hotswap_rewrite(input, source_isa, target_isa, &output);
  }
  api->release_data(input);
  if (status != kComgrStatusSuccess) {
    // Rewrite ran and rejected the object: reproducible, safe to negative-cache.
    set_failure(RetargetFailureKind::kDeterministic);
    HOTSWAP_LOG("hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n", source_isa, target_isa,
                status);
    return false;
  }

  size_t output_size = 0;
  if (api->get_data(output, &output_size, nullptr) != kComgrStatusSuccess || output_size == 0) {
    api->release_data(output);
    return false;
  }

  OwnedElfBuffer output_buffer(std::malloc(output_size), &std::free);
  if (!output_buffer) {
    api->release_data(output);
    return false;
  }

  size_t copy_size = output_size;
  if (api->get_data(output, &copy_size, static_cast<char*>(output_buffer.get())) !=
      kComgrStatusSuccess) {
    api->release_data(output);
    return false;
  }

  api->release_data(output);
  *out_elf_buffer = std::move(output_buffer);
  *out_elf_size = output_size;
  set_failure(RetargetFailureKind::kNone);
  return true;
}

namespace {

// Serve a cache key from the in-memory then disk tiers. Returns:
//   kHitSuccess : *out_elf_buffer / *out_elf_size filled from cache.
//   kHitFailed  : a prior deterministic COMGR failure is cached; skip rewrite.
//   kMiss       : not cached; caller must retarget.
enum class CacheProbe { kMiss, kHitSuccess, kHitFailed };

CacheProbe ProbeRetargetCache(uint64_t cache_key, const RewriteDecision& decision,
                              const CodeObjectView& code_object, const std::string& disk_cache_dir,
                              uint64_t disk_cache_salt, OwnedElfBuffer* out_elf_buffer,
                              size_t* out_elf_size) {
  // In-memory tier.
  std::shared_ptr<std::vector<uint8_t>> cached_bytes;
  {
    std::scoped_lock lock(g_retarget_cache_mutex);
    auto it = g_retarget_cache.find(cache_key);
    if (it != g_retarget_cache.end()) {
      if (!it->second.succeeded) {
        HOTSWAP_LOG("hotswap: cache hit (failed) src=%s tgt=%s entry_trampolines=%d in=%zu\n",
                    decision.source_isa.c_str(), decision.target_isa.c_str(),
                    decision.request_entry_trampolines, code_object.size);
        return CacheProbe::kHitFailed;
      }
      // Succeeded: either bytes are in memory (disk off / memory-backed) or
      // only on disk (fall through to the mmap tier below).
      cached_bytes = it->second.elf_bytes;
    }
  }
  if (cached_bytes) {
    OwnedElfBuffer buf(std::malloc(cached_bytes->size()), &std::free);
    if (buf) {
      std::memcpy(buf.get(), cached_bytes->data(), cached_bytes->size());
      *out_elf_buffer = std::move(buf);
      *out_elf_size = cached_bytes->size();
      HOTSWAP_LOG(
          "hotswap: cache hit (success) src=%s tgt=%s entry_trampolines=%d in=%zu out=%zu\n",
          decision.source_isa.c_str(), decision.target_isa.c_str(),
          decision.request_entry_trampolines, code_object.size, cached_bytes->size());
      return CacheProbe::kHitSuccess;
    }
  }

#if !defined(_WIN32) && !defined(_WIN64)
  // Disk tier: mmap the persisted ELF zero-copy. The OS shares the physical
  // pages across every GPU load and every process mapping the same file.
  if (!disk_cache_dir.empty()) {
    const std::string disk_path = DiskCachePath(disk_cache_dir, cache_key, disk_cache_salt);
    size_t mapped_size = 0;
    OwnedElfBuffer mapped = MmapDiskCache(disk_path, disk_cache_salt, &mapped_size);
    if (mapped) {
      // Leave an on-disk marker so the remaining GPUs skip the in-memory copy
      // and mmap the shared file too.
      try {
        std::scoped_lock lock(g_retarget_cache_mutex);
        if (g_retarget_cache.find(cache_key) == g_retarget_cache.end() &&
            g_retarget_cache.size() < kMaxRetargetCacheEntries) {
          CachedRetargetResult entry;
          entry.succeeded = true;
          entry.on_disk = true;
          g_retarget_cache.emplace(cache_key, std::move(entry));
        }
      } catch (const std::bad_alloc&) {
        // Marker is best-effort; the mmap result is already returned.
      }
      *out_elf_buffer = std::move(mapped);
      *out_elf_size = mapped_size;
      HOTSWAP_LOG(
          "hotswap: disk cache hit (mmap) src=%s tgt=%s entry_trampolines=%d in=%zu "
          "out=%zu\n",
          decision.source_isa.c_str(), decision.target_isa.c_str(),
          decision.request_entry_trampolines, code_object.size, mapped_size);
      return CacheProbe::kHitSuccess;
    }
  }
#else
  (void)disk_cache_dir;
  (void)disk_cache_salt;
#endif

  return CacheProbe::kMiss;
}

// Identifies a cache entry (both tiers) so a failed load can drop the poisoned
// entry and force a recompute next time.
struct RetargetInvalidationRef {
  bool valid = false;
  uint64_t cache_key = 0;
  std::string disk_cache_dir;
  uint64_t disk_cache_salt = 0;
};

void InvalidateRetargetCacheEntry(const RetargetInvalidationRef& ref) {
  if (!ref.valid) {
    return;
  }
  {
    std::scoped_lock lock(g_retarget_cache_mutex);
    g_retarget_cache.erase(ref.cache_key);
  }
#if !defined(_WIN32) && !defined(_WIN64)
  if (!ref.disk_cache_dir.empty()) {
    ::unlink(DiskCachePath(ref.disk_cache_dir, ref.cache_key, ref.disk_cache_salt).c_str());
  }
#endif
}

}  // namespace

static RetargetCodeObjectResult TryRetargetCodeObjectImpl(
    const CodeObjectView& code_object, hsa_agent_t agent, const RewriteOptions& options,
    OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size, RetargetInvalidationRef* out_ref) {
  if (IsHotswapDisabledByEnv() || !code_object.data || code_object.size == 0) {
    return {};
  }

  const AgentGfxRevision gfx = GetAgentGfxRevision(agent);
  if (!IsAgentEligibleForHotswap(gfx, options)) {
    return {};
  }

  // Derive the source ISA straight from the ELF (no COMGR, no dlopen, no
  // full-buffer copy). Only fall back to COMGR if the lightweight parse fails,
  // which keeps COMGR entirely off the cache-hit path.
  std::string source_isa = GetCodeObjectIsaNameFromElf(code_object.data, code_object.size);
  if (source_isa.empty()) {
    source_isa = GetCodeObjectIsaName(code_object.data, code_object.size);
  }
  const std::string target_isa = GetAgentIsaName(agent);
  const std::optional<RewriteDecision> decision =
      DecideHotswapRewrite(gfx, source_isa, target_isa, options);
  if (!decision) {
    HOTSWAP_LOG("hotswap: rewrite skipped, no decision (src='%s' tgt='%s')\n", source_isa.c_str(),
                target_isa.c_str());
    return {};
  }

  uint64_t cache_key = ComputeRetargetCacheKey(
      code_object, decision->source_isa, decision->target_isa, decision->request_entry_trampolines);
  // Strict-mode rewrites can differ from non-strict ones for the same code
  // object, so keep their cache entries distinct.
  if (decision->request_strict_mode) {
    cache_key ^= 0x9E3779B97F4A7C15ULL;
  }

  std::string disk_cache_dir;
  uint64_t disk_cache_salt = 0ULL;
#if !defined(_WIN32) && !defined(_WIN64)
  disk_cache_dir = GetDiskCacheDir();
  disk_cache_salt = disk_cache_dir.empty() ? 0ULL : ComgrIdentitySalt();
#endif

  // Let the caller invalidate this entry if the retargeted bytes fail to load.
  if (out_ref != nullptr) {
    *out_ref = {true, cache_key, disk_cache_dir, disk_cache_salt};
  }

  // Map a completed retarget to the loader's result type, honoring the
  // strict/required semantics: a required (strict) rewrite that fails must
  // surface an error rather than silently falling back to the original object.
  const auto make_result = [&decision](bool rewritten) -> RetargetCodeObjectResult {
    if (rewritten) {
      return {RetargetCodeObjectStatus::kRewritten, decision->rewrite_required};
    }
    if (decision->rewrite_required) {
      return {RetargetCodeObjectStatus::kRequiredRewriteFailed, true};
    }
    return {};
  };

  // Probe the cache, then single-flight the miss so concurrent loads of the
  // same object (e.g. one per GPU, or a prewarm thread) run COMGR only once.
  // Losers wait and re-probe the now-populated cache.
  for (int attempt = 0; attempt < 2; ++attempt) {
    const CacheProbe probe = ProbeRetargetCache(cache_key, *decision, code_object, disk_cache_dir,
                                                disk_cache_salt, out_elf_buffer, out_elf_size);
    if (probe == CacheProbe::kHitSuccess) {
      return {RetargetCodeObjectStatus::kRewritten, decision->rewrite_required};
    }
    if (probe == CacheProbe::kHitFailed) {
      return make_result(false);
    }

    std::shared_ptr<InflightRetarget> inflight;
    bool winner = false;
    {
      std::scoped_lock lock(g_inflight_mutex);
      auto it = g_inflight.find(cache_key);
      if (it == g_inflight.end()) {
        inflight = std::make_shared<InflightRetarget>();
        g_inflight.emplace(cache_key, inflight);
        winner = true;
      } else {
        inflight = it->second;
      }
    }

    if (winner) {
      bool rewritten = false;
      {
        // Wakes waiters on scope exit even if the retarget throws.
        InflightCompletion completion{cache_key, inflight};
        rewritten = PerformRetargetAndCache(code_object, *decision, cache_key, disk_cache_dir,
                                            disk_cache_salt, out_elf_buffer, out_elf_size);
      }
      return make_result(rewritten);
    }

    // Loser: wait for the winner, then re-probe the cache on the next attempt.
    std::unique_lock<std::mutex> lk(inflight->mutex);
    inflight->cv.wait(lk, [&inflight]() { return inflight->done; });
  }

  // Both attempts missed (e.g. the winner's result was not cacheable). Run the
  // retarget directly to guarantee forward progress.
  const bool rewritten = PerformRetargetAndCache(code_object, *decision, cache_key, disk_cache_dir,
                                                 disk_cache_salt, out_elf_buffer, out_elf_size);
  return make_result(rewritten);
}

namespace {

// Rewrite options for the onLoad engine, driven by environment overrides. The
// prepare API takes explicit options instead so its result is deterministic.
RewriteOptions EnvRewriteOptions() {
  RewriteOptions options;
  options.entry_trampolines_enabled = AreEntryTrampolinesRequested();
  options.strict_mode_enabled = IsStrictModeRequested();
  return options;
}

}  // namespace

RetargetCodeObjectResult TryRetargetCodeObject(const CodeObjectView& code_object, hsa_agent_t agent,
                                               OwnedElfBuffer* out_elf_buffer,
                                               size_t* out_elf_size) {
  return TryRetargetCodeObjectImpl(code_object, agent, EnvRewriteOptions(), out_elf_buffer,
                                   out_elf_size, nullptr);
}

PrepareStatus PrepareRetargetedCodeObject(const CodeObjectView& code_object, hsa_agent_t agent,
                                          const RewriteOptions& options,
                                          OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size) {
  const RetargetCodeObjectResult result =
      TryRetargetCodeObjectImpl(code_object, agent, options, out_elf_buffer, out_elf_size, nullptr);
  switch (result.status) {
    case RetargetCodeObjectStatus::kRewritten:
      return PrepareStatus::kPrepared;
    case RetargetCodeObjectStatus::kRequiredRewriteFailed:
      return PrepareStatus::kRequiredRewriteFailed;
    case RetargetCodeObjectStatus::kSkipped:
    default:
      return PrepareStatus::kNotNeeded;
  }
}

RetargetCodeObjectResult TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader,
                                               hsa_agent_t agent, OwnedElfBuffer* out_elf_buffer,
                                               size_t* out_elf_size) {
  if (!reader) {
    return {};
  }

  CodeObjectView code_object;
  code_object.data = reader->GetCodeObjectMemory();
  code_object.size = reader->GetCodeObjectSize();
  code_object.uri = reader->GetUri();
  code_object.content_digest = reader->GetContentDigest();
  return TryRetargetCodeObject(code_object, agent, out_elf_buffer, out_elf_size);
}

hsa_status_t LoadAgentCodeObjectWithHotswap(hsa_executable_t executable, hsa_agent_t agent,
                                            const CodeObjectView& code_object, const char* options,
                                            hsa_loaded_code_object_t* loaded_code_object,
                                            const LoadAgentCodeObjectCallbacks& callbacks) {
  if (!callbacks.load_original_code_object || !callbacks.load_rewritten_code_object) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_code_object_t original_code_object = {reinterpret_cast<uint64_t>(code_object.data)};

  OwnedElfBuffer rewritten_elf_buffer(nullptr, &std::free);
  size_t rewritten_elf_size = 0;
  RetargetInvalidationRef invalidation;
  const RetargetCodeObjectResult retarget_result =
      TryRetargetCodeObjectImpl(code_object, agent, EnvRewriteOptions(), &rewritten_elf_buffer,
                                &rewritten_elf_size, &invalidation);
  if (retarget_result.status == RetargetCodeObjectStatus::kRewritten) {
    hsa_code_object_t rewritten_code_object = {
        reinterpret_cast<uint64_t>(rewritten_elf_buffer.get())};
    hsa_status_t status = callbacks.load_rewritten_code_object(
        callbacks.context, agent, rewritten_code_object, rewritten_elf_size, options,
        code_object.uri, loaded_code_object);
    if (status == HSA_STATUS_SUCCESS) {
      RetainRewrittenElfBuffer(executable, std::move(rewritten_elf_buffer));
      return status;
    }
    // Retargeted bytes did not load: drop the (possibly corrupt) entry so the
    // next attempt recomputes instead of re-serving it (else strict mode could
    // keep failing on the same bad entry).
    InvalidateRetargetCacheEntry(invalidation);
    if (retarget_result.rewrite_required) {
      LogRequiredRewrittenLoadFailure(status);
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    LogRewrittenCodeObjectLoadFailure(status);
  } else if (retarget_result.status == RetargetCodeObjectStatus::kRequiredRewriteFailed) {
    LogRequiredRewriteFailure();
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  return callbacks.load_original_code_object(callbacks.context, agent, original_code_object,
                                             options, code_object.uri, loaded_code_object);
}

void RetainRewrittenElfBuffer(hsa_executable_t executable, OwnedElfBuffer elf_buffer) {
  try {
    std::scoped_lock lock(g_retained_rewritten_elf_buffers_mutex);
    g_retained_rewritten_elf_buffers[executable.handle].push_back(std::move(elf_buffer));
  } catch (const std::bad_alloc&) {
    // If the keepalive container cannot grow, preserve the loaded code object's
    // raw ELF pointer by intentionally leaking this allocation.
    (void)elf_buffer.release();
  }
}

void ReleaseRetainedRewrittenElfBuffers(hsa_executable_t executable) {
  std::scoped_lock lock(g_retained_rewritten_elf_buffers_mutex);
  g_retained_rewritten_elf_buffers.erase(executable.handle);
}

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(const AgentGfxRevision& gfx,
                                                              const std::string& source_isa,
                                                              const std::string& target_isa,
                                                              const RewriteOptions& options) {
  return DecideHotswapRewrite(gfx, source_isa, target_isa, options);
}

size_t RetainedRewrittenElfBufferCountForTesting(hsa_executable_t executable) {
  std::scoped_lock lock(g_retained_rewritten_elf_buffers_mutex);
  const auto it = g_retained_rewritten_elf_buffers.find(executable.handle);
  return it == g_retained_rewritten_elf_buffers.end() ? 0 : it->second.size();
}

bool HotswapRewriteWithOptionsAvailableForTesting() {
  ComgrApi* api = GetComgrApi();
  return api && api->hotswap_rewrite_with_options;
}

size_t RetargetCacheSizeForTesting() {
  std::scoped_lock lock(g_retarget_cache_mutex);
  return g_retarget_cache.size();
}

void ClearRetargetCacheForTesting() {
  std::scoped_lock lock(g_retarget_cache_mutex);
  g_retarget_cache.clear();
}

void ForceRetargetCodeObjectFailureForTesting(bool force) {
  g_force_retarget_code_object_failure_for_testing.store(force, std::memory_order_relaxed);
}
#endif

}  // namespace hotswap
}  // namespace rocr
