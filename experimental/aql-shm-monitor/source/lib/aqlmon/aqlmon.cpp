// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aqlmon/aqlmon.h"
#include "aqlmon/runtime_contract.h"
#include "../runtime_contract/runtime_contract.hpp"

#include "amd_hsa_signal.h"
#include "amd_hsa_queue.h"
#include "hsa.h"
#include "hsa_ext_amd.h"
#include "hsa_ven_amd_loader.h"

#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

static_assert(sizeof(hsa_kernel_dispatch_packet_t) == AQLMON_PACKET_BYTES,
              "Unexpected AQL packet size");

template <typename Tp>
Tp load_atomic(const volatile Tp* ptr) {
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

template <typename Tp>
void store_atomic(volatile Tp* ptr, Tp value) {
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

template <typename Tp>
void atomic_max(std::atomic<Tp>& target, Tp value) {
  Tp current = target.load(std::memory_order_relaxed);
  while(current < value &&
        !target.compare_exchange_weak(current, value, std::memory_order_release,
                                      std::memory_order_relaxed)) {
  }
}

uint64_t monotonic_nsec() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * 1000000000ull) + static_cast<uint64_t>(ts.tv_nsec);
}

uint32_t current_pid() {
  static const uint32_t pid = static_cast<uint32_t>(getpid());
  return pid;
}

uint32_t current_tid() {
  thread_local const uint32_t tid = static_cast<uint32_t>(syscall(SYS_gettid));
  return tid;
}

uint64_t parse_u64_env(const char* name, uint64_t default_value) {
  const char* value = getenv(name);
  if(value == nullptr || *value == '\0') return default_value;

  char* end = nullptr;
  const auto parsed = strtoull(value, &end, 0);
  return (end != nullptr && *end == '\0') ? parsed : default_value;
}

bool parse_bool_env(const char* name, bool default_value = false) {
  const char* value = getenv(name);
  if(value == nullptr || *value == '\0') return default_value;
  if(strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
     strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0) {
    return true;
  }
  if(strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 ||
     strcasecmp(value, "no") == 0 || strcasecmp(value, "off") == 0) {
    return false;
  }
  return default_value;
}

bool async_wrap_enabled() {
  static const bool enabled =
      parse_bool_env("AQLMONITOR_ENABLE_ASYNC_WRAP", false) &&
      !parse_bool_env("AQLMONITOR_DISABLE_ASYNC_WRAP", false);
  return enabled;
}

bool queue_shadow_enabled() {
  static const bool enabled = !parse_bool_env("AQLMONITOR_DISABLE_QUEUE_SHADOW", false);
  return enabled;
}

bool packet_monotonic_enabled() {
  static const bool enabled = parse_bool_env("AQLMONITOR_PACKET_MONOTONIC_NS", false);
  return enabled;
}

bool code_object_tracking_enabled() {
  static const bool enabled = !parse_bool_env("AQLMONITOR_DISABLE_CODE_OBJECT_TRACKING", false);
  return enabled;
}

bool sync_before_executable_destroy_enabled() {
  static const bool enabled = parse_bool_env("AQLMONITOR_SYNC_BEFORE_EXEC_DESTROY", false);
  return enabled;
}

uint64_t executable_destroy_sleep_ms() {
  static const uint64_t value = parse_u64_env("AQLMONITOR_EXEC_DESTROY_SLEEP_MS", 0);
  return value;
}

uint64_t queue_destroy_sleep_ms() {
  static const uint64_t value = parse_u64_env("AQLMONITOR_QUEUE_DESTROY_SLEEP_MS", 0);
  return value;
}

bool debug_retire_enabled() {
  static const bool enabled = parse_bool_env("AQLMONITOR_DEBUG_RETIRE", false);
  return enabled;
}

FILE* debug_log_file() {
  static FILE* file = []() -> FILE* {
    const char* path = getenv("AQLMONITOR_DEBUG_LOG");
    if(path == nullptr || *path == '\0') return stderr;
    FILE* fp = fopen(path, "a");
    return (fp != nullptr) ? fp : stderr;
  }();
  return file;
}

void debug_log(const char* fmt, ...) {
  if(!debug_retire_enabled()) return;

  static std::mutex mutex = {};
  std::lock_guard<std::mutex> lk{mutex};

  FILE* fp = debug_log_file();
  if(fp == nullptr) return;

  va_list args;
  va_start(args, fmt);
  vfprintf(fp, fmt, args);
  va_end(args);
  fflush(fp);
}

void debug_log_backtrace() {
  if(!debug_retire_enabled()) return;

  void* frames[32] = {};
  const int count = backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
  char** symbols = backtrace_symbols(frames, count);
  if(symbols == nullptr) return;

  for(int i = 0; i < count; ++i) {
    debug_log("    bt[%02d] %s\n", i, symbols[i]);
  }
  free(symbols);
}

std::string shm_name_from_env() {
  const char* env_name = getenv("AQLMONITOR_SHM_NAME");
  if(env_name != nullptr && *env_name != '\0') return env_name;
  return "/aqlmon-" + std::to_string(getpid());
}

uint16_t packet_type(uint16_t header) { return static_cast<uint16_t>(header & 0x00ffu); }

bool is_power_of_two(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

uint64_t queue_slot(uint64_t index, uint64_t size, uint64_t mask) {
  return is_power_of_two(size) ? (index & mask) : (index % size);
}

struct SlotState {
  std::atomic<uint64_t> ticket{0};
  std::atomic<uint32_t> tid{0};
};

struct CodeObjectInfo {
  uint64_t code_object_handle = 0;
  uint64_t executable_handle = 0;
  uint64_t agent_handle = 0;
  uint64_t load_base = 0;
  uint64_t load_size = 0;
  std::string uri = {};
};

struct KernelSymbolInfo {
  uint64_t symbol_handle = 0;
  uint64_t executable_handle = 0;
  uint64_t agent_handle = 0;
  uint64_t kernel_object = 0;
  std::string name = {};
};

struct DispatchSignalTracker {
  hsa_signal_t signal = {};
  hsa_agent_t agent = {};
  uint64_t queue_id = 0;
  uint64_t queue_ptr = 0;
  uint64_t queue_base = 0;
  uint64_t queue_size = 0;
  uint64_t dispatch_id = 0;
  uint64_t kernel_object = 0;
  uint32_t pid = 0;
  uint32_t tid = 0;
  bool injected_signal = false;
};

class CompletionTrackQueue {
 public:
  explicit CompletionTrackQueue(uint64_t requested_capacity)
  : capacity_{std::max<uint64_t>(requested_capacity, 2)}
  , mask_{is_power_of_two(capacity_) ? (capacity_ - 1) : 0}
  , entries_(capacity_) {}

  bool push(const DispatchSignalTracker& value) {
    const uint64_t write = write_index_.load(std::memory_order_relaxed);
    const uint64_t read = published_read_index_.load(std::memory_order_acquire);
    if((write - read) >= capacity_) return false;

    entries_[slot(write)] = value;
    write_index_.store(write + 1, std::memory_order_release);
    return true;
  }

  bool pop(DispatchSignalTracker* value) {
    if(value == nullptr) return false;

    const uint64_t read = local_read_index_;
    const uint64_t write = write_index_.load(std::memory_order_acquire);
    if(read == write) return false;

    *value = entries_[slot(read)];
    local_read_index_ = read + 1;
    published_read_index_.store(local_read_index_, std::memory_order_release);
    return true;
  }

 private:
  uint64_t slot(uint64_t index) const {
    return is_power_of_two(capacity_) ? (index & mask_) : (index % capacity_);
  }

  const uint64_t capacity_ = 0;
  const uint64_t mask_ = 0;
  std::vector<DispatchSignalTracker> entries_ = {};
  std::atomic<uint64_t> write_index_{0};
  std::atomic<uint64_t> published_read_index_{0};
  uint64_t local_read_index_ = 0;
};

struct WrappedAsyncHandlerState {
  hsa_signal_t signal = {};
  hsa_amd_signal_handler handler = nullptr;
  void* handler_arg = nullptr;
};

struct QueueState {
  explicit QueueState(hsa_queue_t* input_queue, amd_queue_v2_t* input_amd_queue,
                      hsa_agent_t input_agent)
  : queue{input_queue}
  , amd_queue{input_amd_queue}
  , agent{input_agent}
  , real_doorbell{input_queue ? input_queue->doorbell_signal : hsa_signal_t{}}
  , size{input_queue ? static_cast<uint64_t>(input_queue->size) : 0}
  , mask{size > 0 ? (size - 1) : 0}
  , slots(size) {}

  hsa_queue_t* queue = nullptr;
  amd_queue_v2_t* amd_queue = nullptr;
  hsa_agent_t agent = {};
  hsa_signal_t real_doorbell = {};
  uint64_t size = 0;
  uint64_t mask = 0;
  std::atomic<uint64_t> shadow_reserved_wptr{0};
  std::atomic<uint64_t> shadow_doorbell_wptr{0};
  std::atomic<uint64_t> published_wptr{0};
  std::atomic<uint64_t> last_real_rptr{0};
  std::atomic<bool> profiling_enabled{false};
  std::atomic<bool> active{true};
  std::mutex publish_mutex = {};
  std::vector<SlotState> slots = {};
};

class SharedMemoryTrace {
 public:
  ~SharedMemoryTrace() {
    if(mapping_ != nullptr && mapping_ != MAP_FAILED) {
      munmap(mapping_, mapping_size_);
    }
    if(fd_ >= 0) close(fd_);
  }

  bool init() {
    name_ = shm_name_from_env();
    capacity_ = static_cast<uint32_t>(parse_u64_env("AQLMONITOR_CAPACITY", 4096));
    if(capacity_ == 0) capacity_ = 4096;

    mapping_size_ = sizeof(aqlmon_shm_header_t) + (sizeof(aqlmon_record_t) * capacity_);
    fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
    if(fd_ < 0) return false;
    if(ftruncate(fd_, static_cast<off_t>(mapping_size_)) != 0) return false;

    mapping_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if(mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      return false;
    }

    header_ = static_cast<aqlmon_shm_header_t*>(mapping_);
    records_ = reinterpret_cast<aqlmon_record_t*>(static_cast<uint8_t*>(mapping_) +
                                                  sizeof(aqlmon_shm_header_t));

    std::memset(mapping_, 0, mapping_size_);
    header_->magic = AQLMON_MAGIC;
    header_->version = AQLMON_VERSION;
    header_->header_size = sizeof(aqlmon_shm_header_t);
    header_->record_size = sizeof(aqlmon_record_t);
    header_->capacity = capacity_;
    std::snprintf(header_->shm_name, sizeof(header_->shm_name), "%s", name_.c_str());
    return true;
  }

  void write_record(const aqlmon_record_t& input) {
    if(header_ == nullptr || records_ == nullptr) return;

    const uint64_t seq = __atomic_fetch_add(&header_->write_seq, 1, __ATOMIC_RELAXED);
    auto& slot = records_[seq % capacity_];
    aqlmon_record_t rec = input;
    rec.seq = 0;
    slot = rec;
    __atomic_store_n(&slot.seq, seq + 1, __ATOMIC_RELEASE);
  }

  void add_dropped_records(uint64_t value = 1) {
    if(header_ == nullptr) return;
    __atomic_fetch_add(&header_->dropped_records, value, __ATOMIC_RELAXED);
  }

  void add_dropped_packets(uint64_t value = 1) {
    if(header_ == nullptr) return;
    __atomic_fetch_add(&header_->dropped_packets, value, __ATOMIC_RELAXED);
  }

 private:
  std::string name_ = {};
  uint32_t capacity_ = 0;
  int fd_ = -1;
  size_t mapping_size_ = 0;
  void* mapping_ = nullptr;
  aqlmon_shm_header_t* header_ = nullptr;
  aqlmon_record_t* records_ = nullptr;
};

using hsa_queue_create_fn_t =
    hsa_status_t (*)(hsa_agent_t, uint32_t, hsa_queue_type32_t,
                     void (*)(hsa_status_t, hsa_queue_t*, void*), void*, uint32_t, uint32_t,
                     hsa_queue_t**);
using hsa_shut_down_fn_t = hsa_status_t (*)();
using hsa_queue_destroy_fn_t = hsa_status_t (*)(hsa_queue_t*);
using hsa_queue_inactivate_fn_t = hsa_status_t (*)(hsa_queue_t*);
using hsa_queue_load_index_fn_t = uint64_t (*)(const hsa_queue_t*);
using hsa_queue_store_index_fn_t = void (*)(const hsa_queue_t*, uint64_t);
using hsa_queue_add_write_index_fn_t = uint64_t (*)(const hsa_queue_t*, uint64_t);
using hsa_queue_cas_write_index_fn_t = uint64_t (*)(const hsa_queue_t*, uint64_t, uint64_t);
using hsa_signal_create_fn_t =
    hsa_status_t (*)(hsa_signal_value_t, uint32_t, const hsa_agent_t*, hsa_signal_t*);
using hsa_signal_load_fn_t = hsa_signal_value_t (*)(hsa_signal_t);
using hsa_signal_store_fn_t = void (*)(hsa_signal_t, hsa_signal_value_t);
using hsa_signal_destroy_fn_t = hsa_status_t (*)(hsa_signal_t);
using hsa_amd_signal_async_handler_fn_t =
    hsa_status_t (*)(hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t,
                     hsa_amd_signal_handler, void*);
using hsa_amd_profiling_set_profiler_enabled_fn_t = hsa_status_t (*)(hsa_queue_t*, int);
using hsa_amd_profiling_get_dispatch_time_fn_t =
    hsa_status_t (*)(hsa_agent_t, hsa_signal_t, hsa_amd_profiling_dispatch_time_t*);
using hsa_amd_profiling_convert_tick_to_system_domain_fn_t =
    hsa_status_t (*)(hsa_agent_t, uint64_t, uint64_t*);
using hsa_executable_freeze_fn_t = hsa_status_t (*)(hsa_executable_t, const char*);
using hsa_executable_destroy_fn_t = hsa_status_t (*)(hsa_executable_t);
using hsa_executable_get_symbol_by_name_fn_t =
    hsa_status_t (*)(hsa_executable_t, const char*, const hsa_agent_t*,
                     hsa_executable_symbol_t*);
using hsa_executable_symbol_get_info_fn_t =
    hsa_status_t (*)(hsa_executable_symbol_t, hsa_executable_symbol_info_t, void*);
using hsa_executable_iterate_symbols_fn_t =
    hsa_status_t (*)(hsa_executable_t,
                     hsa_status_t (*)(hsa_executable_t, hsa_executable_symbol_t, void*),
                     void*);
using hsa_loader_iterate_loaded_code_objects_fn_t =
    hsa_status_t (*)(hsa_executable_t,
                     hsa_status_t (*)(hsa_executable_t, hsa_loaded_code_object_t, void*), void*);
using hsa_loader_loaded_code_object_get_info_fn_t =
    hsa_status_t (*)(hsa_loaded_code_object_t, hsa_ven_amd_loader_loaded_code_object_info_t,
                     void*);

struct RealApi {
  hsa_shut_down_fn_t shut_down = nullptr;
  hsa_queue_create_fn_t queue_create = nullptr;
  hsa_queue_destroy_fn_t queue_destroy = nullptr;
  hsa_queue_inactivate_fn_t queue_inactivate = nullptr;
  hsa_queue_load_index_fn_t queue_load_read_index_scacquire = nullptr;
  hsa_queue_load_index_fn_t queue_load_read_index_relaxed = nullptr;
  hsa_queue_load_index_fn_t queue_load_write_index_scacquire = nullptr;
  hsa_queue_load_index_fn_t queue_load_write_index_relaxed = nullptr;
  hsa_queue_store_index_fn_t queue_store_read_index_relaxed = nullptr;
  hsa_queue_store_index_fn_t queue_store_read_index_screlease = nullptr;
  hsa_queue_store_index_fn_t queue_store_write_index_relaxed = nullptr;
  hsa_queue_store_index_fn_t queue_store_write_index_screlease = nullptr;
  hsa_queue_add_write_index_fn_t queue_add_write_index_scacq_screl = nullptr;
  hsa_queue_add_write_index_fn_t queue_add_write_index_scacquire = nullptr;
  hsa_queue_add_write_index_fn_t queue_add_write_index_relaxed = nullptr;
  hsa_queue_add_write_index_fn_t queue_add_write_index_screlease = nullptr;
  hsa_queue_cas_write_index_fn_t queue_cas_write_index_scacq_screl = nullptr;
  hsa_queue_cas_write_index_fn_t queue_cas_write_index_scacquire = nullptr;
  hsa_queue_cas_write_index_fn_t queue_cas_write_index_relaxed = nullptr;
  hsa_queue_cas_write_index_fn_t queue_cas_write_index_screlease = nullptr;
  hsa_signal_create_fn_t signal_create = nullptr;
  hsa_signal_load_fn_t signal_load_relaxed = nullptr;
  hsa_signal_store_fn_t signal_store_relaxed = nullptr;
  hsa_signal_store_fn_t signal_store_screlease = nullptr;
  hsa_signal_destroy_fn_t signal_destroy = nullptr;
  hsa_amd_signal_async_handler_fn_t amd_signal_async_handler = nullptr;
  hsa_amd_profiling_set_profiler_enabled_fn_t amd_profiling_set_profiler_enabled = nullptr;
  hsa_amd_profiling_get_dispatch_time_fn_t amd_profiling_get_dispatch_time = nullptr;
  hsa_amd_profiling_convert_tick_to_system_domain_fn_t
      amd_profiling_convert_tick_to_system_domain = nullptr;
  hsa_executable_freeze_fn_t executable_freeze = nullptr;
  hsa_executable_destroy_fn_t executable_destroy = nullptr;
  hsa_executable_get_symbol_by_name_fn_t executable_get_symbol_by_name = nullptr;
  hsa_executable_symbol_get_info_fn_t executable_symbol_get_info = nullptr;
  hsa_executable_iterate_symbols_fn_t executable_iterate_symbols = nullptr;
  hsa_loader_iterate_loaded_code_objects_fn_t loader_iterate_loaded_code_objects = nullptr;
  hsa_loader_loaded_code_object_get_info_fn_t loader_loaded_code_object_get_info = nullptr;
};

std::atomic<int>& monitor_fini_status() {
  static std::atomic<int> value{0};
  return value;
}

bool monitor_finalizing() { return monitor_fini_status().load(std::memory_order_acquire) != 0; }

void finalize_monitor_atexit();

RealApi& real_api() {
  static RealApi api = {};
  static std::once_flag once = {};
  std::call_once(once, []() {
    auto resolve = [](const char* name) { return dlsym(RTLD_NEXT, name); };

    api.shut_down = reinterpret_cast<hsa_shut_down_fn_t>(resolve("hsa_shut_down"));
    api.queue_create = reinterpret_cast<hsa_queue_create_fn_t>(resolve("hsa_queue_create"));
    api.queue_destroy = reinterpret_cast<hsa_queue_destroy_fn_t>(resolve("hsa_queue_destroy"));
    api.queue_inactivate =
        reinterpret_cast<hsa_queue_inactivate_fn_t>(resolve("hsa_queue_inactivate"));
    api.queue_load_read_index_scacquire = reinterpret_cast<hsa_queue_load_index_fn_t>(
        resolve("hsa_queue_load_read_index_scacquire"));
    api.queue_load_read_index_relaxed =
        reinterpret_cast<hsa_queue_load_index_fn_t>(resolve("hsa_queue_load_read_index_relaxed"));
    api.queue_load_write_index_scacquire = reinterpret_cast<hsa_queue_load_index_fn_t>(
        resolve("hsa_queue_load_write_index_scacquire"));
    api.queue_load_write_index_relaxed = reinterpret_cast<hsa_queue_load_index_fn_t>(
        resolve("hsa_queue_load_write_index_relaxed"));
    api.queue_store_read_index_relaxed = reinterpret_cast<hsa_queue_store_index_fn_t>(
        resolve("hsa_queue_store_read_index_relaxed"));
    api.queue_store_read_index_screlease = reinterpret_cast<hsa_queue_store_index_fn_t>(
        resolve("hsa_queue_store_read_index_screlease"));
    api.queue_store_write_index_relaxed = reinterpret_cast<hsa_queue_store_index_fn_t>(
        resolve("hsa_queue_store_write_index_relaxed"));
    api.queue_store_write_index_screlease = reinterpret_cast<hsa_queue_store_index_fn_t>(
        resolve("hsa_queue_store_write_index_screlease"));
    api.queue_add_write_index_scacq_screl = reinterpret_cast<hsa_queue_add_write_index_fn_t>(
        resolve("hsa_queue_add_write_index_scacq_screl"));
    api.queue_add_write_index_scacquire = reinterpret_cast<hsa_queue_add_write_index_fn_t>(
        resolve("hsa_queue_add_write_index_scacquire"));
    api.queue_add_write_index_relaxed = reinterpret_cast<hsa_queue_add_write_index_fn_t>(
        resolve("hsa_queue_add_write_index_relaxed"));
    api.queue_add_write_index_screlease = reinterpret_cast<hsa_queue_add_write_index_fn_t>(
        resolve("hsa_queue_add_write_index_screlease"));
    api.queue_cas_write_index_scacq_screl = reinterpret_cast<hsa_queue_cas_write_index_fn_t>(
        resolve("hsa_queue_cas_write_index_scacq_screl"));
    api.queue_cas_write_index_scacquire = reinterpret_cast<hsa_queue_cas_write_index_fn_t>(
        resolve("hsa_queue_cas_write_index_scacquire"));
    api.queue_cas_write_index_relaxed = reinterpret_cast<hsa_queue_cas_write_index_fn_t>(
        resolve("hsa_queue_cas_write_index_relaxed"));
    api.queue_cas_write_index_screlease = reinterpret_cast<hsa_queue_cas_write_index_fn_t>(
        resolve("hsa_queue_cas_write_index_screlease"));
    api.signal_create = reinterpret_cast<hsa_signal_create_fn_t>(resolve("hsa_signal_create"));
    api.signal_load_relaxed =
        reinterpret_cast<hsa_signal_load_fn_t>(resolve("hsa_signal_load_relaxed"));
    api.signal_store_relaxed =
        reinterpret_cast<hsa_signal_store_fn_t>(resolve("hsa_signal_store_relaxed"));
    api.signal_store_screlease =
        reinterpret_cast<hsa_signal_store_fn_t>(resolve("hsa_signal_store_screlease"));
    api.signal_destroy = reinterpret_cast<hsa_signal_destroy_fn_t>(resolve("hsa_signal_destroy"));
    api.amd_signal_async_handler = reinterpret_cast<hsa_amd_signal_async_handler_fn_t>(
        resolve("hsa_amd_signal_async_handler"));
    api.amd_profiling_set_profiler_enabled =
        reinterpret_cast<hsa_amd_profiling_set_profiler_enabled_fn_t>(
            resolve("hsa_amd_profiling_set_profiler_enabled"));
    api.amd_profiling_get_dispatch_time =
        reinterpret_cast<hsa_amd_profiling_get_dispatch_time_fn_t>(
            resolve("hsa_amd_profiling_get_dispatch_time"));
    api.amd_profiling_convert_tick_to_system_domain =
        reinterpret_cast<hsa_amd_profiling_convert_tick_to_system_domain_fn_t>(
            resolve("hsa_amd_profiling_convert_tick_to_system_domain"));
    api.executable_freeze =
        reinterpret_cast<hsa_executable_freeze_fn_t>(resolve("hsa_executable_freeze"));
    api.executable_destroy =
        reinterpret_cast<hsa_executable_destroy_fn_t>(resolve("hsa_executable_destroy"));
    api.executable_get_symbol_by_name =
        reinterpret_cast<hsa_executable_get_symbol_by_name_fn_t>(
            resolve("hsa_executable_get_symbol_by_name"));
    api.executable_symbol_get_info =
        reinterpret_cast<hsa_executable_symbol_get_info_fn_t>(
            resolve("hsa_executable_symbol_get_info"));
    api.executable_iterate_symbols =
        reinterpret_cast<hsa_executable_iterate_symbols_fn_t>(
            resolve("hsa_executable_iterate_symbols"));
    api.loader_iterate_loaded_code_objects =
        reinterpret_cast<hsa_loader_iterate_loaded_code_objects_fn_t>(
            resolve("hsa_ven_amd_loader_executable_iterate_loaded_code_objects"));
    api.loader_loaded_code_object_get_info =
        reinterpret_cast<hsa_loader_loaded_code_object_get_info_fn_t>(
            resolve("hsa_ven_amd_loader_loaded_code_object_get_info"));
  });
  return api;
}

class Monitor {
 public:
  static Monitor& instance() {
    static Monitor* value = new Monitor{};
    return *value;
  }

  bool enabled() const { return enabled_; }

  void begin_shutdown() {
    if(!enabled_) return;

    bool expected = false;
    if(!shutting_down_.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
      return;
    }

    debug_log("begin_shutdown this=%p\n", static_cast<void*>(this));
    drain_queue_publication();
    stop_.store(true, std::memory_order_release);
    if(publisher_.joinable()) publisher_.join();
    if(completion_poller_.joinable()) completion_poller_.join();

    {
      std::lock_guard<std::mutex> lk{registry_mutex_};
      for(auto& itr : queues_) {
        if(itr.second) itr.second->active.store(false, std::memory_order_release);
      }
      queues_.clear();
      doorbells_.clear();
    }

    {
      std::lock_guard<std::mutex> lk{signal_mutex_};
      signal_dispatches_.clear();
      wrapped_handlers_.clear();
    }

    {
      std::lock_guard<std::mutex> lk{destroyed_signal_mutex_};
      destroyed_signals_.clear();
    }

    completion_tracks_.reset();
    destroy_free_injected_signals();
  }

  void register_queue(hsa_queue_t* queue, hsa_agent_t agent) {
    if(!enabled_ || queue == nullptr || !queue_shadow_enabled()) return;

    auto* amd_queue = reinterpret_cast<amd_queue_v2_t*>(queue);
    auto state = std::make_shared<QueueState>(queue, amd_queue, agent);

    const uint64_t initial_wptr = amd_queue ? load_atomic(&amd_queue->write_dispatch_id) : 0;
    const uint64_t initial_rptr = amd_queue ? load_atomic(&amd_queue->read_dispatch_id) : 0;
    state->shadow_reserved_wptr.store(initial_wptr, std::memory_order_relaxed);
    state->shadow_doorbell_wptr.store(initial_wptr, std::memory_order_relaxed);
    state->published_wptr.store(initial_wptr, std::memory_order_relaxed);
    state->last_real_rptr.store(initial_rptr, std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> lk{registry_mutex_};
      queues_[reinterpret_cast<uint64_t>(queue)] = state;
      if(state->real_doorbell.handle != 0) doorbells_[state->real_doorbell.handle] = state;
    }
    debug_log("register queue=%p id=%llu base=%p size=%u doorbell=0x%llx\n",
              static_cast<void*>(queue), static_cast<unsigned long long>(queue->id),
              queue->base_address, queue->size,
              static_cast<unsigned long long>(state->real_doorbell.handle));
    enable_queue_profiling(*state);
  }

  void retire_queue(hsa_queue_t* queue, const char* reason) {
    if(queue == nullptr) return;

    auto state = lookup_queue(queue);
    if(!state) {
      debug_log("retire-miss reason=%s queue=%p\n", reason, static_cast<void*>(queue));
      return;
    }

    const uint64_t reserved = state->shadow_reserved_wptr.load(std::memory_order_acquire);
    const uint64_t hinted = state->shadow_doorbell_wptr.load(std::memory_order_acquire);
    const uint64_t published = state->published_wptr.load(std::memory_order_acquire);
    debug_log("retire reason=%s queue=%p id=%llu reserved=%llu hinted=%llu published=%llu active=%d\n",
              reason, static_cast<void*>(queue),
              static_cast<unsigned long long>(state->queue ? state->queue->id : 0),
              static_cast<unsigned long long>(reserved),
              static_cast<unsigned long long>(hinted),
              static_cast<unsigned long long>(published),
              static_cast<int>(state->active.load(std::memory_order_acquire)));
    if(reserved != published || hinted != published) {
      debug_log("retire backlog reason=%s queue=%p reserved=%llu hinted=%llu published=%llu\n",
                reason, static_cast<void*>(queue),
                static_cast<unsigned long long>(reserved),
                static_cast<unsigned long long>(hinted),
                static_cast<unsigned long long>(published));
      debug_log_backtrace();
    }

    state->active.store(false, std::memory_order_release);

    std::lock_guard<std::mutex> lk{registry_mutex_};
    queues_.erase(reinterpret_cast<uint64_t>(queue));
    if(state->real_doorbell.handle != 0) doorbells_.erase(state->real_doorbell.handle);
  }

  uint64_t load_shadow_write_index(const hsa_queue_t* queue, std::memory_order order,
                                   bool* handled = nullptr) {
    auto state = lookup_queue(queue);
    if(!state) {
      if(handled != nullptr) *handled = false;
      return 0;
    }

    if(handled != nullptr) *handled = true;
    return state->shadow_reserved_wptr.load(order);
  }

  uint64_t add_shadow_write_index(const hsa_queue_t* queue, uint64_t value, std::memory_order order,
                                  bool* handled = nullptr) {
    auto state = lookup_queue(queue);
    if(!state) {
      if(handled != nullptr) *handled = false;
      return 0;
    }

    if(handled != nullptr) *handled = true;
    const uint64_t prior = state->shadow_reserved_wptr.fetch_add(value, order);
    reserve_range(*state, prior, prior + value);
    return prior;
  }

  uint64_t cas_shadow_write_index(const hsa_queue_t* queue, uint64_t expected, uint64_t value,
                                  std::memory_order success_order,
                                  std::memory_order failure_order, bool* handled = nullptr) {
    auto state = lookup_queue(queue);
    if(!state) {
      if(handled != nullptr) *handled = false;
      return 0;
    }

    if(handled != nullptr) *handled = true;
    uint64_t observed = expected;
    if(state->shadow_reserved_wptr.compare_exchange_strong(observed, value, success_order,
                                                           failure_order) &&
       value > expected) {
      reserve_range(*state, expected, value);
    }
    return observed;
  }

  void store_shadow_write_index(const hsa_queue_t* queue, uint64_t value, std::memory_order order,
                                bool* handled = nullptr) {
    auto state = lookup_queue(queue);
    if(!state) {
      if(handled != nullptr) *handled = false;
      return;
    }

    if(handled != nullptr) *handled = true;
    const uint64_t prior = state->shadow_reserved_wptr.exchange(value, order);
    if(value > prior) reserve_range(*state, prior, value);
  }

  uint64_t load_real_read_index(const hsa_queue_t* queue, bool scacquire, bool* handled = nullptr) {
    auto state = lookup_queue(queue);
    if(!state) {
      if(handled != nullptr) *handled = false;
      return 0;
    }

    auto& api = real_api();
    const auto fn = scacquire ? api.queue_load_read_index_scacquire : api.queue_load_read_index_relaxed;
    uint64_t value = state->last_real_rptr.load(std::memory_order_relaxed);
    if(fn != nullptr) value = fn(queue);
    else if(state->amd_queue != nullptr)
      value = load_atomic(&state->amd_queue->read_dispatch_id);

    state->last_real_rptr.store(value, std::memory_order_release);
    if(handled != nullptr) *handled = true;
    return value;
  }

  void store_real_read_index(const hsa_queue_t* queue, uint64_t value, bool screlease,
                             bool* handled = nullptr) {
    auto state = lookup_queue(queue);
    if(!state) {
      if(handled != nullptr) *handled = false;
      return;
    }

    auto& api = real_api();
    const auto fn = screlease ? api.queue_store_read_index_screlease : api.queue_store_read_index_relaxed;
    if(fn != nullptr) fn(queue, value);
    state->last_real_rptr.store(value, std::memory_order_release);
    if(handled != nullptr) *handled = true;
  }

  void note_queue_profiling_setting(const hsa_queue_t* queue, bool enabled) {
    auto state = lookup_queue(queue);
    if(!state) return;
    state->profiling_enabled.store(enabled, std::memory_order_release);
  }

  bool note_shadow_doorbell(hsa_signal_t signal, hsa_signal_value_t value) {
    auto state = lookup_doorbell(signal);
    if(!state) return false;

    if(value >= 0) {
      atomic_max(state->shadow_doorbell_wptr, static_cast<uint64_t>(value) + 1);
    }
    return true;
  }

  void note_symbol_by_name(hsa_executable_t executable, const char* symbol_name,
                           const hsa_agent_t* agent, hsa_executable_symbol_t symbol) {
    if(!enabled_ || symbol_name == nullptr || *symbol_name == '\0' || symbol.handle == 0) {
      return;
    }

    KernelSymbolInfo info{};
    info.symbol_handle = symbol.handle;
    info.executable_handle = executable.handle;
    info.agent_handle = (agent != nullptr) ? agent->handle : 0;
    info.name = symbol_name;

    auto& api = real_api();
    if(api.executable_symbol_get_info != nullptr) {
      uint64_t kernel_object = 0;
      if(api.executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                        &kernel_object) == HSA_STATUS_SUCCESS) {
        info.kernel_object = kernel_object;
      }
    }

    record_kernel_symbol(info);
  }

  void note_symbol_kernel_object(hsa_executable_symbol_t symbol, uint64_t kernel_object) {
    if(!enabled_ || symbol.handle == 0 || kernel_object == 0) return;

    KernelSymbolInfo info{};
    {
      std::lock_guard<std::mutex> lk{symbol_mutex_};
      const auto itr = symbols_by_handle_.find(symbol.handle);
      if(itr != symbols_by_handle_.end()) info = itr->second;
    }

    info.symbol_handle = symbol.handle;
    info.kernel_object = kernel_object;
    if(info.name.empty()) info.name = query_symbol_name(symbol);
    record_kernel_symbol(info);
  }

  void note_executable_frozen(hsa_executable_t executable) {
    if(!enabled_) return;

    auto infos = collect_code_objects(executable);
    auto symbols = collect_kernel_symbols(executable);
    {
      std::lock_guard<std::mutex> lk{registry_mutex_};
      executable_code_objects_[executable.handle] = infos;
    }
    for(const auto& symbol : symbols) {
      record_kernel_symbol(symbol);
    }

    for(const auto& info : infos) {
      emit_code_object_record(AQLMON_RECORD_CODE_OBJECT_LIVE, info);
    }
  }

  void note_executable_destroyed(hsa_executable_t executable) {
    if(!enabled_) return;

    std::vector<CodeObjectInfo> infos = {};
    {
      std::lock_guard<std::mutex> lk{registry_mutex_};
      const auto itr = executable_code_objects_.find(executable.handle);
      if(itr != executable_code_objects_.end()) {
        infos = itr->second;
        executable_code_objects_.erase(itr);
      }
    }

    for(const auto& info : infos) {
      emit_code_object_record(AQLMON_RECORD_CODE_OBJECT_DEAD, info);
    }
  }

  void wait_for_all_queues_idle() {
    if(!enabled_) return;

    auto& api = real_api();
    const uint64_t timeout_ns = parse_u64_env("AQLMONITOR_EXEC_DESTROY_SYNC_TIMEOUT_NS", 5000000000ull);
    const uint64_t start_ns = monotonic_nsec();

    while(true) {
      bool all_idle = true;
      for(const auto& queue : snapshot_queues()) {
        uint64_t current_rptr = queue->last_real_rptr.load(std::memory_order_relaxed);
        if(api.queue_load_read_index_relaxed != nullptr && queue->queue != nullptr) {
          current_rptr = api.queue_load_read_index_relaxed(queue->queue);
          queue->last_real_rptr.store(current_rptr, std::memory_order_release);
        } else if(queue->amd_queue != nullptr) {
          current_rptr = load_atomic(&queue->amd_queue->read_dispatch_id);
          queue->last_real_rptr.store(current_rptr, std::memory_order_release);
        }

        const uint64_t published =
            queue->published_wptr.load(std::memory_order_acquire);
        if(current_rptr < published) {
          all_idle = false;
          break;
        }
      }

      if(all_idle) return;
      if(monotonic_nsec() - start_ns >= timeout_ns) {
        debug_log("wait_for_all_queues_idle timeout after %llu ns\n",
                  static_cast<unsigned long long>(timeout_ns));
        return;
      }

      timespec ts{};
      ts.tv_nsec = 1000000;
      nanosleep(&ts, nullptr);
    }
  }

  bool queues_fully_published() {
    for(const auto& queue : snapshot_queues()) {
      const uint64_t hinted = queue->shadow_doorbell_wptr.load(std::memory_order_acquire);
      const uint64_t published = queue->published_wptr.load(std::memory_order_acquire);
      if(hinted > published) return false;
    }
    return true;
  }

  void drain_queue_publication() {
    const uint64_t timeout_ns =
        parse_u64_env("AQLMONITOR_FINAL_DRAIN_TIMEOUT_NS", 5000000000ull);
    const uint64_t start_ns = monotonic_nsec();

    while(true) {
      bool progress = false;
      for(const auto& queue : snapshot_queues()) {
        progress = publish_queue(*queue) || progress;
      }

      if(queues_fully_published()) {
        wait_for_all_queues_idle();
        return;
      }

      if(monotonic_nsec() - start_ns >= timeout_ns) {
        debug_log("final queue drain timeout after %llu ns\n",
                  static_cast<unsigned long long>(timeout_ns));
        trace_.add_dropped_packets();
        return;
      }

      if(progress) continue;

      timespec ts{};
      ts.tv_nsec = 1000000;
      nanosleep(&ts, nullptr);
    }
  }

 private:
  struct CodeObjectCollection {
    std::vector<CodeObjectInfo> infos = {};
  };

  struct KernelSymbolCollection {
    std::vector<KernelSymbolInfo> infos = {};
  };

  Monitor() {
    enabled_ = trace_.init();
    debug_log("monitor ctor enabled=%d this=%p\n", static_cast<int>(enabled_), static_cast<void*>(this));
    if(enabled_) {
      std::atexit(&finalize_monitor_atexit);
      completion_tracks_ = std::make_unique<CompletionTrackQueue>(
          parse_u64_env("AQLMONITOR_COMPLETION_QUEUE_CAPACITY", 65536));
      publisher_ = std::thread{[this]() { publisher_loop(); }};
      completion_poller_ = std::thread{[this]() { completion_loop(); }};
    }
  }

  ~Monitor() {
    debug_log("monitor dtor begin this=%p\n", static_cast<void*>(this));
    begin_shutdown();
    debug_log("monitor dtor end this=%p\n", static_cast<void*>(this));
  }

  std::shared_ptr<QueueState> lookup_queue(const hsa_queue_t* queue) {
    if(!enabled_ || queue == nullptr || shutting_down_.load(std::memory_order_acquire)) return {};

    thread_local const hsa_queue_t* cached_queue = nullptr;
    thread_local std::shared_ptr<QueueState> cached_state = {};
    if(cached_queue == queue && cached_state &&
       cached_state->active.load(std::memory_order_acquire)) {
      return cached_state;
    }

    std::lock_guard<std::mutex> lk{registry_mutex_};
    const auto itr = queues_.find(reinterpret_cast<uint64_t>(queue));
    cached_queue = queue;
    cached_state = (itr != queues_.end() && itr->second &&
                    itr->second->active.load(std::memory_order_acquire))
                       ? itr->second
                       : std::shared_ptr<QueueState>{};
    return cached_state;
  }

  std::shared_ptr<QueueState> lookup_doorbell(hsa_signal_t signal) {
    if(!enabled_ || signal.handle == 0 || shutting_down_.load(std::memory_order_acquire)) return {};

    thread_local uint64_t cached_signal = 0;
    thread_local std::shared_ptr<QueueState> cached_state = {};
    if(cached_signal == signal.handle && cached_state &&
       cached_state->active.load(std::memory_order_acquire) &&
       cached_state->real_doorbell.handle == signal.handle) {
      return cached_state;
    }

    std::lock_guard<std::mutex> lk{registry_mutex_};
    const auto itr = doorbells_.find(signal.handle);
    cached_signal = signal.handle;
    cached_state = (itr != doorbells_.end() && itr->second &&
                    itr->second->active.load(std::memory_order_acquire))
                       ? itr->second
                       : std::shared_ptr<QueueState>{};
    return cached_state;
  }

  void reserve_range(QueueState& queue, uint64_t begin, uint64_t end) {
    const uint32_t tid = current_tid();
    for(uint64_t index = begin; index < end; ++index) {
      auto& slot = queue.slots[queue_slot(index, queue.size, queue.mask)];
      slot.tid.store(tid, std::memory_order_relaxed);
      slot.ticket.store(index + 1, std::memory_order_release);
    }
  }

  void enable_queue_profiling(QueueState& queue) {
    auto& api = real_api();
    if(queue.queue == nullptr || api.amd_profiling_set_profiler_enabled == nullptr) return;

    const auto status = api.amd_profiling_set_profiler_enabled(queue.queue, 1);
    queue.profiling_enabled.store(status == HSA_STATUS_SUCCESS, std::memory_order_release);
  }

  void note_dispatch_signal(hsa_signal_t signal, const DispatchSignalTracker& tracker) {
    if(signal.handle == 0) return;

    if(completion_tracks_ != nullptr && !completion_tracks_->push(tracker)) {
      trace_.add_dropped_packets();
    }

    if(async_wrap_enabled()) {
      std::lock_guard<std::mutex> lk{signal_mutex_};
      signal_dispatches_[signal.handle].emplace_back(tracker);
    }
  }

 public:
  void note_signal_destroyed(hsa_signal_t signal) {
    if(signal.handle == 0) return;

    {
      std::lock_guard<std::mutex> lk{signal_mutex_};
      signal_dispatches_.erase(signal.handle);
      wrapped_handlers_.erase(signal.handle);
    }

    std::lock_guard<std::mutex> lk{destroyed_signal_mutex_};
    destroyed_signals_.emplace_back(signal.handle);
  }
  void note_wrapped_handler(hsa_signal_t signal, hsa_amd_signal_handler handler, void* arg) {
    if(signal.handle == 0 || handler == nullptr) return;

    std::lock_guard<std::mutex> lk{signal_mutex_};
    wrapped_handlers_[signal.handle] = WrappedAsyncHandlerState{signal, handler, arg};
  }

 private:
  bool claim_dispatch_signal(hsa_signal_t signal, DispatchSignalTracker* tracker) {
    if(signal.handle == 0 || tracker == nullptr) return false;

    std::lock_guard<std::mutex> lk{signal_mutex_};
    auto itr = signal_dispatches_.find(signal.handle);
    if(itr == signal_dispatches_.end() || itr->second.empty()) return false;

    *tracker = itr->second.front();
    itr->second.pop_front();
    if(itr->second.empty()) signal_dispatches_.erase(itr);
    return true;
  }

  hsa_signal_t acquire_injected_signal() {
    auto& api = real_api();
    if(api.signal_create == nullptr) return {};

    hsa_signal_t signal{};
    {
      std::lock_guard<std::mutex> lk{injected_signal_mutex_};
      if(!free_injected_signals_.empty()) {
        signal = free_injected_signals_.back();
        free_injected_signals_.pop_back();
      }
    }

    if(signal.handle == 0 &&
       api.signal_create(1, 0, nullptr, &signal) != HSA_STATUS_SUCCESS) {
      return {};
    }

    if(signal.handle != 0) {
      if(api.signal_store_relaxed != nullptr) api.signal_store_relaxed(signal, 1);
      else if(api.signal_store_screlease != nullptr) api.signal_store_screlease(signal, 1);
    }
    return signal;
  }

  void recycle_injected_signal(hsa_signal_t signal) {
    if(signal.handle == 0) return;
    std::lock_guard<std::mutex> lk{injected_signal_mutex_};
    free_injected_signals_.emplace_back(signal);
  }

  void destroy_free_injected_signals() {
    auto& api = real_api();
    if(api.signal_destroy == nullptr) return;

    std::vector<hsa_signal_t> free_signals = {};
    {
      std::lock_guard<std::mutex> lk{injected_signal_mutex_};
      free_signals.swap(free_injected_signals_);
    }

    for(const auto signal : free_signals) {
      api.signal_destroy(signal);
    }
  }

  void handle_dispatch_completion(const DispatchSignalTracker& tracker,
                                  hsa_signal_value_t signal_value) {
    aqlmon_record_t rec{};
    rec.size = sizeof(rec);
    rec.kind = AQLMON_RECORD_DISPATCH_COMPLETE;
    rec.packet_type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    rec.pid = tracker.pid;
    rec.tid = tracker.tid;
    rec.monotonic_ns = monotonic_nsec();
    rec.queue_id = tracker.queue_id;
    rec.queue_ptr = tracker.queue_ptr;
    rec.queue_base = tracker.queue_base;
    rec.queue_size = tracker.queue_size;
    rec.dispatch_id = tracker.dispatch_id;
    rec.write_index = tracker.dispatch_id;
    rec.kernel_object = tracker.kernel_object;
    rec.completion_signal = tracker.signal.handle;
    rec.signal_value = static_cast<uint64_t>(signal_value);
    rec.agent_handle = tracker.agent.handle;

    auto& api = real_api();
    bool have_timestamps = false;
    if(api.amd_profiling_get_dispatch_time != nullptr && tracker.agent.handle != 0) {
      hsa_amd_profiling_dispatch_time_t time{};
      if(api.amd_profiling_get_dispatch_time(tracker.agent, tracker.signal, &time) ==
         HSA_STATUS_SUCCESS) {
        rec.dispatch_start_ns = time.start;
        rec.dispatch_end_ns = time.end;
        have_timestamps = true;
      }
    }

    if(!have_timestamps && tracker.agent.handle != 0) {
      auto* amd_signal = reinterpret_cast<amd_signal_t*>(tracker.signal.handle);
      if(amd_signal != nullptr) {
        const uint64_t start_tick = load_atomic(&amd_signal->start_ts);
        const uint64_t end_tick = load_atomic(&amd_signal->end_ts);
        if(start_tick != 0 && end_tick != 0 &&
           api.amd_profiling_convert_tick_to_system_domain != nullptr) {
          uint64_t start_ns = 0;
          uint64_t end_ns = 0;
          if(api.amd_profiling_convert_tick_to_system_domain(tracker.agent, start_tick,
                                                             &start_ns) == HSA_STATUS_SUCCESS &&
             api.amd_profiling_convert_tick_to_system_domain(tracker.agent, end_tick,
                                                             &end_ns) == HSA_STATUS_SUCCESS) {
            rec.dispatch_start_ns = start_ns;
            rec.dispatch_end_ns = end_ns;
            have_timestamps = true;
          }
        }
      }
    }

    if(have_timestamps) rec.flags |= AQLMON_FLAG_SIGNAL_TIMESTAMPS_VALID;
    if(tracker.injected_signal) rec.flags |= AQLMON_FLAG_INJECTED_SIGNAL;
    trace_.write_record(rec);
  }

 public:
  static bool wrapped_signal_handler(hsa_signal_value_t signal_value, void* data) {
    const auto signal_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data));
    if(signal_handle == 0) return false;

    WrappedAsyncHandlerState wrapped{};
    {
      std::lock_guard<std::mutex> lk{Monitor::instance().signal_mutex_};
      const auto itr = Monitor::instance().wrapped_handlers_.find(signal_handle);
      if(itr == Monitor::instance().wrapped_handlers_.end()) return false;
      wrapped = itr->second;
    }

    DispatchSignalTracker tracker{};
    if(Monitor::instance().claim_dispatch_signal(wrapped.signal, &tracker)) {
      Monitor::instance().handle_dispatch_completion(tracker, signal_value);
    }

    const bool keep_handler =
        (wrapped.handler != nullptr) ? wrapped.handler(signal_value, wrapped.handler_arg) : false;
    if(!keep_handler) {
      std::lock_guard<std::mutex> lk{Monitor::instance().signal_mutex_};
      const auto itr = Monitor::instance().wrapped_handlers_.find(signal_handle);
      if(itr != Monitor::instance().wrapped_handlers_.end() &&
         itr->second.handler == wrapped.handler && itr->second.handler_arg == wrapped.handler_arg) {
        Monitor::instance().wrapped_handlers_.erase(itr);
      }
    }
    return keep_handler;
  }

 private:
  bool note_packet_completion_signal(QueueState& queue, hsa_kernel_dispatch_packet_t* packet,
                                     uint64_t dispatch_id, uint32_t pid, uint32_t tid) {
    if(packet == nullptr) return false;

    hsa_signal_t signal = packet->completion_signal;
    bool injected_signal = false;
    if(signal.handle == 0 &&
       aqlmon::runtime_contract::effective_completion_signal_mode() ==
           AQLMON_COMPLETION_SIGNAL_MODE_MONITOR_PROVIDED) {
      signal = acquire_injected_signal();
      if(signal.handle != 0) {
        packet->completion_signal = signal;
        injected_signal = true;
      }
    }

    if(signal.handle == 0) return false;

    DispatchSignalTracker tracker{};
    tracker.signal = signal;
    tracker.agent = queue.agent;
    tracker.queue_id = queue.queue ? queue.queue->id : 0;
    tracker.queue_ptr = reinterpret_cast<uint64_t>(queue.queue);
    tracker.queue_base = reinterpret_cast<uint64_t>(queue.queue ? queue.queue->base_address : nullptr);
    tracker.queue_size = queue.queue ? queue.queue->size : 0;
    tracker.dispatch_id = dispatch_id;
    tracker.kernel_object = packet->kernel_object;
    tracker.pid = pid;
    tracker.tid = tid;
    tracker.injected_signal = injected_signal;
    note_dispatch_signal(signal, tracker);
    return injected_signal;
  }

  static std::string query_symbol_name(hsa_executable_symbol_t symbol) {
    auto& api = real_api();
    if(api.executable_symbol_get_info == nullptr || symbol.handle == 0) return {};

    uint32_t name_length = 0;
    if(api.executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                                      &name_length) != HSA_STATUS_SUCCESS ||
       name_length == 0) {
      return {};
    }

    std::vector<char> name(name_length + 1, '\0');
    if(api.executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                                      name.data()) != HSA_STATUS_SUCCESS) {
      return {};
    }
    name[name_length] = '\0';
    return name.data();
  }

  void record_kernel_symbol(const KernelSymbolInfo& info) {
    if(info.symbol_handle == 0 && info.kernel_object == 0) return;

    std::lock_guard<std::mutex> lk{symbol_mutex_};
    if(info.symbol_handle != 0) {
      auto& slot = symbols_by_handle_[info.symbol_handle];
      if(info.executable_handle != 0) slot.executable_handle = info.executable_handle;
      if(info.agent_handle != 0) slot.agent_handle = info.agent_handle;
      if(info.kernel_object != 0) slot.kernel_object = info.kernel_object;
      if(!info.name.empty()) slot.name = info.name;
      slot.symbol_handle = info.symbol_handle;

      if(slot.kernel_object != 0 && !slot.name.empty()) {
        kernel_names_by_object_[slot.kernel_object] = slot.name;
      }
      return;
    }

    if(info.kernel_object != 0 && !info.name.empty()) {
      kernel_names_by_object_[info.kernel_object] = info.name;
    }
  }

  std::string kernel_name_for(uint64_t kernel_object) {
    if(kernel_object == 0) return {};
    std::lock_guard<std::mutex> lk{symbol_mutex_};
    const auto itr = kernel_names_by_object_.find(kernel_object);
    return (itr != kernel_names_by_object_.end()) ? itr->second : std::string{};
  }

  static hsa_status_t collect_executable_symbol(hsa_executable_t executable,
                                                hsa_executable_symbol_t symbol, void* data) {
    auto* collection = static_cast<KernelSymbolCollection*>(data);
    auto& api = real_api();
    if(api.executable_symbol_get_info == nullptr || collection == nullptr) {
      return HSA_STATUS_SUCCESS;
    }

    hsa_symbol_kind_t kind{};
    if(api.executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) !=
           HSA_STATUS_SUCCESS ||
       kind != HSA_SYMBOL_KIND_KERNEL) {
      return HSA_STATUS_SUCCESS;
    }

    uint64_t kernel_object = 0;
    if(api.executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                      &kernel_object) != HSA_STATUS_SUCCESS ||
       kernel_object == 0) {
      return HSA_STATUS_SUCCESS;
    }

    KernelSymbolInfo info{};
    info.symbol_handle = symbol.handle;
    info.executable_handle = executable.handle;
    info.kernel_object = kernel_object;
    info.name = query_symbol_name(symbol);

    hsa_agent_t agent{};
    if(api.executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &agent) ==
       HSA_STATUS_SUCCESS) {
      info.agent_handle = agent.handle;
    }

    collection->infos.emplace_back(std::move(info));
    return HSA_STATUS_SUCCESS;
  }

  std::vector<KernelSymbolInfo> collect_kernel_symbols(hsa_executable_t executable) {
    auto& api = real_api();
    if(api.executable_iterate_symbols == nullptr || api.executable_symbol_get_info == nullptr) {
      return {};
    }

    KernelSymbolCollection collection = {};
    const auto status = api.executable_iterate_symbols(
        executable, &Monitor::collect_executable_symbol, &collection);
    return (status == HSA_STATUS_SUCCESS) ? std::move(collection.infos)
                                          : std::vector<KernelSymbolInfo>{};
  }

  static hsa_status_t collect_loaded_code_object(hsa_executable_t executable,
                                                 hsa_loaded_code_object_t loaded_code_object,
                                                 void* data) {
    auto* collection = static_cast<CodeObjectCollection*>(data);
    auto& api = real_api();
    if(api.loader_loaded_code_object_get_info == nullptr) return HSA_STATUS_SUCCESS;

    CodeObjectInfo info = {};
    info.code_object_handle = loaded_code_object.handle;
    info.executable_handle = executable.handle;

    hsa_agent_t agent{};
    api.loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &agent);
    info.agent_handle = agent.handle;

    api.loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &info.load_base);
    api.loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &info.load_size);

    uint32_t uri_length = 0;
    if(api.loader_loaded_code_object_get_info(
           loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
           &uri_length) == HSA_STATUS_SUCCESS &&
       uri_length > 1 && uri_length < (AQLMON_MAX_CODE_OBJECT_URI * 8)) {
      std::vector<char> uri(uri_length, '\0');
      if(api.loader_loaded_code_object_get_info(
             loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
             uri.data()) == HSA_STATUS_SUCCESS) {
        info.uri = uri.data();
      }
    }

    collection->infos.emplace_back(std::move(info));
    return HSA_STATUS_SUCCESS;
  }

  std::vector<CodeObjectInfo> collect_code_objects(hsa_executable_t executable) {
    auto& api = real_api();
    if(api.loader_iterate_loaded_code_objects == nullptr ||
       api.loader_loaded_code_object_get_info == nullptr) {
      return {};
    }

    CodeObjectCollection collection = {};
    const auto status = api.loader_iterate_loaded_code_objects(
        executable, &Monitor::collect_loaded_code_object, &collection);
    return (status == HSA_STATUS_SUCCESS) ? std::move(collection.infos) : std::vector<CodeObjectInfo>{};
  }

  void emit_code_object_record(uint16_t kind, const CodeObjectInfo& info) {
    aqlmon_record_t rec{};
    rec.size = sizeof(rec);
    rec.kind = kind;
    rec.pid = current_pid();
    rec.tid = 0;
    rec.monotonic_ns = monotonic_nsec();
    rec.code_object_handle = info.code_object_handle;
    rec.executable_handle = info.executable_handle;
    rec.agent_handle = info.agent_handle;
    rec.code_object_load_base = info.load_base;
    rec.code_object_load_size = info.load_size;
    if(info.load_base != 0 || info.load_size != 0) {
      rec.flags |= AQLMON_FLAG_CODE_OBJECT_LOAD_RANGE_VALID;
    }
    if(!info.uri.empty()) {
      std::snprintf(rec.code_object_uri, sizeof(rec.code_object_uri), "%s", info.uri.c_str());
      rec.flags |= AQLMON_FLAG_CODE_OBJECT_URI_VALID;
    }
    trace_.write_record(rec);
  }

  bool publish_queue(QueueState& queue) {
    std::lock_guard<std::mutex> publish_lk{queue.publish_mutex};
    if(!queue.active.load(std::memory_order_acquire)) return false;

    const uint64_t hint = queue.shadow_doorbell_wptr.load(std::memory_order_acquire);
    uint64_t published = queue.published_wptr.load(std::memory_order_relaxed);
    if(hint <= published || queue.queue == nullptr || queue.queue->base_address == nullptr ||
       queue.size == 0) {
      return false;
    }

    auto& api = real_api();
    uint64_t current_rptr = queue.last_real_rptr.load(std::memory_order_relaxed);
    if(api.queue_load_read_index_relaxed != nullptr) {
      current_rptr = api.queue_load_read_index_relaxed(queue.queue);
      queue.last_real_rptr.store(current_rptr, std::memory_order_release);
    } else if(queue.amd_queue != nullptr) {
      current_rptr = load_atomic(&queue.amd_queue->read_dispatch_id);
      queue.last_real_rptr.store(current_rptr, std::memory_order_release);
    }

    auto* ring = static_cast<uint8_t*>(queue.queue->base_address);
    const uint64_t begin = published;
    while(published < hint) {
      const uint64_t slot_index = queue_slot(published, queue.size, queue.mask);
      const auto& slot_meta = queue.slots[slot_index];
      if(slot_meta.ticket.load(std::memory_order_acquire) != (published + 1)) break;

      auto* slot = ring + (slot_index * AQLMON_PACKET_BYTES);
      auto* packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(slot);
      const uint16_t header = load_atomic(&packet->header);
      if(packet_type(header) == HSA_PACKET_TYPE_INVALID) break;

      const uint32_t producer_tid = slot_meta.tid.load(std::memory_order_relaxed);
      const uint32_t record_pid = current_pid();
      const uint32_t record_tid = producer_tid;
      bool injected_signal = false;

      if(packet_type(header) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        injected_signal =
            note_packet_completion_signal(queue, packet, published, record_pid, record_tid);
      }

      aqlmon_record_t rec{};
      rec.size = sizeof(rec);
      rec.kind = AQLMON_RECORD_PACKET;
      rec.packet_type = packet_type(header);
      rec.pid = current_pid();
      rec.tid = 0;
      rec.monotonic_ns = packet_monotonic_enabled() ? monotonic_nsec() : 0;
      rec.queue_id = queue.queue->id;
      rec.queue_ptr = reinterpret_cast<uint64_t>(queue.queue);
      rec.queue_base = reinterpret_cast<uint64_t>(queue.queue->base_address);
      rec.queue_size = queue.queue->size;
      rec.dispatch_id = published;
      rec.write_index = published;
      rec.observed_wptr = hint;
      rec.observed_rptr = current_rptr;
      rec.packet_header = header;
      std::memcpy(rec.packet_bytes, slot, AQLMON_PACKET_BYTES);
      if(producer_tid != 0) {
        rec.pid = record_pid;
        rec.tid = producer_tid;
        rec.flags |= AQLMON_FLAG_PRODUCER_VALID;
      }

      if(rec.packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        rec.kernel_object = packet->kernel_object;
        rec.completion_signal = packet->completion_signal.handle;
        rec.agent_handle = queue.agent.handle;
        const auto kernel_name = kernel_name_for(rec.kernel_object);
        if(!kernel_name.empty()) {
          std::snprintf(rec.kernel_name, sizeof(rec.kernel_name), "%s", kernel_name.c_str());
          rec.flags |= AQLMON_FLAG_KERNEL_NAME_VALID;
        }
        if(injected_signal) rec.flags |= AQLMON_FLAG_INJECTED_SIGNAL;
      }

      trace_.write_record(rec);
      ++published;
    }

    if(published == begin) return false;

    queue.published_wptr.store(published, std::memory_order_release);
    if(api.queue_store_write_index_screlease != nullptr) {
      api.queue_store_write_index_screlease(queue.queue, published);
    } else if(queue.amd_queue != nullptr) {
      store_atomic(&queue.amd_queue->write_dispatch_id, published);
    }

    if(api.signal_store_screlease != nullptr && queue.real_doorbell.handle != 0) {
      api.signal_store_screlease(queue.real_doorbell,
                                 static_cast<hsa_signal_value_t>(published - 1));
    }
    return true;
  }

  std::vector<std::shared_ptr<QueueState>> snapshot_queues() {
    std::vector<std::shared_ptr<QueueState>> result = {};
    std::lock_guard<std::mutex> lk{registry_mutex_};
    result.reserve(queues_.size());
    for(const auto& itr : queues_) {
      if(itr.second && itr.second->active.load(std::memory_order_acquire)) {
        result.emplace_back(itr.second);
      }
    }
    return result;
  }

  void completion_loop() {
    const uint64_t idle_ns = parse_u64_env("AQLMONITOR_COMPLETION_IDLE_NS", 50000);
    const uint64_t final_timeout_ns =
        parse_u64_env("AQLMONITOR_FINAL_DRAIN_TIMEOUT_NS", 5000000000ull);
    std::unordered_map<uint64_t, std::deque<DispatchSignalTracker>> pending = {};
    uint64_t stop_start_ns = 0;

    while(true) {
      const bool stop_requested = stop_.load(std::memory_order_acquire);
      if(stop_requested && stop_start_ns == 0) stop_start_ns = monotonic_nsec();
      bool progress = false;

      if(completion_tracks_ != nullptr) {
        DispatchSignalTracker tracker{};
        while(completion_tracks_->pop(&tracker)) {
          pending[tracker.signal.handle].emplace_back(std::move(tracker));
          progress = true;
        }
      }

      {
        std::vector<uint64_t> destroyed = {};
        {
          std::lock_guard<std::mutex> lk{destroyed_signal_mutex_};
          destroyed.swap(destroyed_signals_);
        }
        for(const auto signal_handle : destroyed) {
          pending.erase(signal_handle);
        }
      }

      auto& api = real_api();
      for(auto itr = pending.begin(); itr != pending.end();) {
        auto& queue = itr->second;
        while(!queue.empty()) {
          const auto& tracker = queue.front();
          const auto value =
              (api.signal_load_relaxed != nullptr) ? api.signal_load_relaxed(tracker.signal) : 1;
          if(value > 0) break;

          handle_dispatch_completion(tracker, value);
          if(tracker.injected_signal) recycle_injected_signal(tracker.signal);
          queue.pop_front();
          progress = true;
        }

        if(queue.empty()) {
          itr = pending.erase(itr);
        } else {
          ++itr;
        }
      }

      if(stop_requested && pending.empty()) break;
      if(stop_requested && stop_start_ns != 0 &&
         monotonic_nsec() - stop_start_ns >= final_timeout_ns) {
        uint64_t remaining = 0;
        for(const auto& itr : pending) remaining += itr.second.size();
        trace_.add_dropped_packets(remaining);
        debug_log("completion final drain timeout pending=%llu\n",
                  static_cast<unsigned long long>(remaining));
        break;
      }

      if(progress) continue;

      if(idle_ns == 0) {
        sched_yield();
      } else {
        timespec ts{};
        ts.tv_sec = static_cast<time_t>(idle_ns / 1000000000ull);
        ts.tv_nsec = static_cast<long>(idle_ns % 1000000000ull);
        nanosleep(&ts, nullptr);
      }
    }
  }

  void publisher_loop() {
    const uint64_t idle_ns = parse_u64_env("AQLMONITOR_IDLE_NS", 50000);
    while(!stop_.load(std::memory_order_acquire)) {
      bool progress = false;
      for(const auto& queue : snapshot_queues()) {
        progress = publish_queue(*queue) || progress;
      }

      if(progress) continue;

      if(idle_ns == 0) {
        sched_yield();
      } else {
        timespec ts{};
        ts.tv_sec = static_cast<time_t>(idle_ns / 1000000000ull);
        ts.tv_nsec = static_cast<long>(idle_ns % 1000000000ull);
        nanosleep(&ts, nullptr);
      }
    }
  }

  bool enabled_ = false;
  std::atomic<bool> stop_{false};
  std::atomic<bool> shutting_down_{false};
  SharedMemoryTrace trace_ = {};
  std::mutex registry_mutex_ = {};
  std::mutex signal_mutex_ = {};
  std::mutex destroyed_signal_mutex_ = {};
  std::mutex injected_signal_mutex_ = {};
  std::mutex symbol_mutex_ = {};
  std::unordered_map<uint64_t, std::shared_ptr<QueueState>> queues_ = {};
  std::unordered_map<uint64_t, std::shared_ptr<QueueState>> doorbells_ = {};
  std::unordered_map<uint64_t, std::deque<DispatchSignalTracker>> signal_dispatches_ = {};
  std::unordered_map<uint64_t, WrappedAsyncHandlerState> wrapped_handlers_ = {};
  std::unordered_map<uint64_t, std::vector<CodeObjectInfo>> executable_code_objects_ = {};
  std::unordered_map<uint64_t, KernelSymbolInfo> symbols_by_handle_ = {};
  std::unordered_map<uint64_t, std::string> kernel_names_by_object_ = {};
  std::vector<uint64_t> destroyed_signals_ = {};
  std::vector<hsa_signal_t> free_injected_signals_ = {};
  std::unique_ptr<CompletionTrackQueue> completion_tracks_ = {};
  std::thread publisher_ = {};
  std::thread completion_poller_ = {};
};

}  // namespace

namespace {

void finalize_monitor_atexit() {
  int expected = 0;
  if(!monitor_fini_status().compare_exchange_strong(expected, -1, std::memory_order_acq_rel,
                                                    std::memory_order_acquire)) {
    return;
  }

  Monitor::instance().begin_shutdown();
  monitor_fini_status().store(1, std::memory_order_release);
}

}  // namespace

#if defined(__GNUC__)
#define AQLMON_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define AQLMON_EXPORT extern "C"
#endif

AQLMON_EXPORT hsa_status_t hsa_queue_create(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue) {
  auto& api = real_api();
  auto status = api.queue_create(agent, size, type, callback, data, private_segment_size,
                                 group_segment_size, queue);
  if(status == HSA_STATUS_SUCCESS && queue != nullptr && !monitor_finalizing()) {
    Monitor::instance().register_queue(*queue, agent);
  }
  return status;
}

AQLMON_EXPORT hsa_status_t hsa_shut_down() {
  debug_log("hsa_shut_down enter\n");
  finalize_monitor_atexit();
  const auto status = real_api().shut_down();
  debug_log("hsa_shut_down exit status=%d\n", static_cast<int>(status));
  return status;
}

AQLMON_EXPORT hsa_status_t hsa_queue_destroy(hsa_queue_t* queue) {
  if(!monitor_finalizing()) Monitor::instance().retire_queue(queue, "destroy");
  const uint64_t sleep_ms = queue_destroy_sleep_ms();
  if(sleep_ms > 0) {
    timespec ts{};
    ts.tv_sec = static_cast<time_t>(sleep_ms / 1000ull);
    ts.tv_nsec = static_cast<long>((sleep_ms % 1000ull) * 1000000ull);
    nanosleep(&ts, nullptr);
  }
  return real_api().queue_destroy(queue);
}

AQLMON_EXPORT hsa_status_t hsa_queue_inactivate(hsa_queue_t* queue) {
  if(!monitor_finalizing()) Monitor::instance().retire_queue(queue, "inactivate");
  return real_api().queue_inactivate(queue);
}

AQLMON_EXPORT uint64_t hsa_queue_load_read_index_scacquire(const hsa_queue_t* queue) {
  bool handled = false;
  const auto value = Monitor::instance().load_real_read_index(queue, true, &handled);
  return handled ? value : real_api().queue_load_read_index_scacquire(queue);
}

AQLMON_EXPORT uint64_t hsa_queue_load_read_index_relaxed(const hsa_queue_t* queue) {
  bool handled = false;
  const auto value = Monitor::instance().load_real_read_index(queue, false, &handled);
  return handled ? value : real_api().queue_load_read_index_relaxed(queue);
}

AQLMON_EXPORT uint64_t hsa_queue_load_read_index_acquire(const hsa_queue_t* queue) {
  return hsa_queue_load_read_index_scacquire(queue);
}

AQLMON_EXPORT uint64_t hsa_queue_load_write_index_scacquire(const hsa_queue_t* queue) {
  bool handled = false;
  const auto value =
      Monitor::instance().load_shadow_write_index(queue, std::memory_order_acquire, &handled);
  return handled ? value : real_api().queue_load_write_index_scacquire(queue);
}

AQLMON_EXPORT uint64_t hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue) {
  bool handled = false;
  const auto value =
      Monitor::instance().load_shadow_write_index(queue, std::memory_order_relaxed, &handled);
  return handled ? value : real_api().queue_load_write_index_relaxed(queue);
}

AQLMON_EXPORT uint64_t hsa_queue_load_write_index_acquire(const hsa_queue_t* queue) {
  return hsa_queue_load_write_index_scacquire(queue);
}

AQLMON_EXPORT void hsa_queue_store_read_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  bool handled = false;
  Monitor::instance().store_real_read_index(queue, value, false, &handled);
  if(!handled) real_api().queue_store_read_index_relaxed(queue, value);
}

AQLMON_EXPORT void hsa_queue_store_read_index_screlease(const hsa_queue_t* queue, uint64_t value) {
  bool handled = false;
  Monitor::instance().store_real_read_index(queue, value, true, &handled);
  if(!handled) real_api().queue_store_read_index_screlease(queue, value);
}

AQLMON_EXPORT void hsa_queue_store_read_index_release(const hsa_queue_t* queue, uint64_t value) {
  hsa_queue_store_read_index_screlease(queue, value);
}

AQLMON_EXPORT void hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  bool handled = false;
  Monitor::instance().store_shadow_write_index(queue, value, std::memory_order_relaxed, &handled);
  if(!handled) real_api().queue_store_write_index_relaxed(queue, value);
}

AQLMON_EXPORT void hsa_queue_store_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
  bool handled = false;
  Monitor::instance().store_shadow_write_index(queue, value, std::memory_order_release, &handled);
  if(!handled) real_api().queue_store_write_index_screlease(queue, value);
}

AQLMON_EXPORT void hsa_queue_store_write_index_release(const hsa_queue_t* queue, uint64_t value) {
  hsa_queue_store_write_index_screlease(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_scacq_screl(const hsa_queue_t* queue,
                                                             uint64_t value) {
  bool handled = false;
  const auto prior = Monitor::instance().add_shadow_write_index(
      queue, value, std::memory_order_acq_rel, &handled);
  return handled ? prior : real_api().queue_add_write_index_scacq_screl(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_scacquire(const hsa_queue_t* queue,
                                                           uint64_t value) {
  bool handled = false;
  const auto prior = Monitor::instance().add_shadow_write_index(
      queue, value, std::memory_order_acquire, &handled);
  return handled ? prior : real_api().queue_add_write_index_scacquire(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue,
                                                         uint64_t value) {
  bool handled = false;
  const auto prior = Monitor::instance().add_shadow_write_index(
      queue, value, std::memory_order_relaxed, &handled);
  return handled ? prior : real_api().queue_add_write_index_relaxed(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_screlease(const hsa_queue_t* queue,
                                                           uint64_t value) {
  bool handled = false;
  const auto prior = Monitor::instance().add_shadow_write_index(
      queue, value, std::memory_order_release, &handled);
  return handled ? prior : real_api().queue_add_write_index_screlease(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_acq_rel(const hsa_queue_t* queue,
                                                         uint64_t value) {
  return hsa_queue_add_write_index_scacq_screl(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_acquire(const hsa_queue_t* queue,
                                                         uint64_t value) {
  return hsa_queue_add_write_index_scacquire(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_add_write_index_release(const hsa_queue_t* queue,
                                                         uint64_t value) {
  return hsa_queue_add_write_index_screlease(queue, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_scacq_screl(const hsa_queue_t* queue,
                                                             uint64_t expected, uint64_t value) {
  bool handled = false;
  const auto observed = Monitor::instance().cas_shadow_write_index(
      queue, expected, value, std::memory_order_acq_rel, std::memory_order_acquire, &handled);
  return handled ? observed : real_api().queue_cas_write_index_scacq_screl(queue, expected, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_scacquire(const hsa_queue_t* queue,
                                                           uint64_t expected, uint64_t value) {
  bool handled = false;
  const auto observed = Monitor::instance().cas_shadow_write_index(
      queue, expected, value, std::memory_order_acquire, std::memory_order_acquire, &handled);
  return handled ? observed : real_api().queue_cas_write_index_scacquire(queue, expected, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_relaxed(const hsa_queue_t* queue,
                                                         uint64_t expected, uint64_t value) {
  bool handled = false;
  const auto observed = Monitor::instance().cas_shadow_write_index(
      queue, expected, value, std::memory_order_relaxed, std::memory_order_relaxed, &handled);
  return handled ? observed : real_api().queue_cas_write_index_relaxed(queue, expected, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_screlease(const hsa_queue_t* queue,
                                                           uint64_t expected, uint64_t value) {
  bool handled = false;
  const auto observed = Monitor::instance().cas_shadow_write_index(
      queue, expected, value, std::memory_order_release, std::memory_order_relaxed, &handled);
  return handled ? observed : real_api().queue_cas_write_index_screlease(queue, expected, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_acq_rel(const hsa_queue_t* queue,
                                                         uint64_t expected, uint64_t value) {
  return hsa_queue_cas_write_index_scacq_screl(queue, expected, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_acquire(const hsa_queue_t* queue,
                                                         uint64_t expected, uint64_t value) {
  return hsa_queue_cas_write_index_scacquire(queue, expected, value);
}

AQLMON_EXPORT uint64_t hsa_queue_cas_write_index_release(const hsa_queue_t* queue,
                                                         uint64_t expected, uint64_t value) {
  return hsa_queue_cas_write_index_screlease(queue, expected, value);
}

AQLMON_EXPORT void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  if(monitor_finalizing() || !Monitor::instance().note_shadow_doorbell(signal, value)) {
    real_api().signal_store_relaxed(signal, value);
  }
}

AQLMON_EXPORT void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  if(monitor_finalizing() || !Monitor::instance().note_shadow_doorbell(signal, value)) {
    real_api().signal_store_screlease(signal, value);
  }
}

AQLMON_EXPORT void hsa_signal_store_release(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_signal_store_screlease(signal, value);
}

AQLMON_EXPORT hsa_status_t hsa_signal_destroy(hsa_signal_t signal) {
  if(!monitor_finalizing()) Monitor::instance().note_signal_destroyed(signal);
  return real_api().signal_destroy(signal);
}

AQLMON_EXPORT hsa_status_t hsa_amd_signal_async_handler(hsa_signal_t signal,
                                                        hsa_signal_condition_t cond,
                                                        hsa_signal_value_t value,
                                                        hsa_amd_signal_handler handler,
                                                        void* arg) {
  if(monitor_finalizing() || !async_wrap_enabled()) {
    return real_api().amd_signal_async_handler(signal, cond, value, handler, arg);
  }

  if(handler == nullptr || signal.handle == 0) {
    return real_api().amd_signal_async_handler(signal, cond, value, handler, arg);
  }

  Monitor::instance().note_wrapped_handler(signal, handler, arg);
  return real_api().amd_signal_async_handler(
      signal, cond, value, &Monitor::wrapped_signal_handler,
      reinterpret_cast<void*>(static_cast<uintptr_t>(signal.handle)));
}

AQLMON_EXPORT hsa_status_t hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable) {
  auto status = real_api().amd_profiling_set_profiler_enabled(queue, enable);
  if(status == HSA_STATUS_SUCCESS && !monitor_finalizing()) {
    Monitor::instance().note_queue_profiling_setting(queue, enable != 0);
  }
  return status;
}

AQLMON_EXPORT hsa_status_t hsa_executable_get_symbol_by_name(
    hsa_executable_t executable, const char* symbol_name, const hsa_agent_t* agent,
    hsa_executable_symbol_t* symbol) {
  auto status =
      real_api().executable_get_symbol_by_name(executable, symbol_name, agent, symbol);
  if(status == HSA_STATUS_SUCCESS && symbol != nullptr && !monitor_finalizing()) {
    Monitor::instance().note_symbol_by_name(executable, symbol_name, agent, *symbol);
  }
  return status;
}

AQLMON_EXPORT hsa_status_t hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol, hsa_executable_symbol_info_t attribute,
    void* value) {
  auto status = real_api().executable_symbol_get_info(executable_symbol, attribute, value);
  if(status == HSA_STATUS_SUCCESS && value != nullptr && !monitor_finalizing() &&
     attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
    Monitor::instance().note_symbol_kernel_object(
        executable_symbol, *static_cast<const uint64_t*>(value));
  }
  return status;
}

AQLMON_EXPORT hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                                 const char* options) {
  auto status = real_api().executable_freeze(executable, options);
  if(status == HSA_STATUS_SUCCESS && code_object_tracking_enabled() && !monitor_finalizing()) {
    Monitor::instance().note_executable_frozen(executable);
  }
  return status;
}

AQLMON_EXPORT hsa_status_t hsa_executable_destroy(hsa_executable_t executable) {
  if(sync_before_executable_destroy_enabled()) {
    Monitor::instance().wait_for_all_queues_idle();
  }
  const uint64_t sleep_ms = executable_destroy_sleep_ms();
  if(sleep_ms > 0) {
    timespec ts{};
    ts.tv_sec = static_cast<time_t>(sleep_ms / 1000ull);
    ts.tv_nsec = static_cast<long>((sleep_ms % 1000ull) * 1000000ull);
    nanosleep(&ts, nullptr);
  }
  auto status = real_api().executable_destroy(executable);
  if(status == HSA_STATUS_SUCCESS && code_object_tracking_enabled() && !monitor_finalizing()) {
    Monitor::instance().note_executable_destroyed(executable);
  }
  return status;
}
