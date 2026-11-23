/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// A link-time fake of the HSA functions the ROCr diagnostic module calls, so
// that the module's own sources can be built into a test that needs no GPU and
// no ROCr. It is written against <hsa/hsa.h> and <hsa/hsa_ext_amd.h>, so a
// signature cannot drift from the one the module compiles against, and it
// defines the functions F9's sources call as well as the ones F22's do, so that
// the same file serves the before run.
//
// It also defines get_kfd_node_id(), which F22 maps an entity with, and
// RdcPerfTimer's constructor and destructor, which only F9's sources reach (its
// unused member), as empty stubs: librdc's constructor measures the clock and
// spins for about a quarter of a second for every object.
//
// fake_hsa.h describes what it answers by default and the controls a case sets.
// All of the state is guarded by one mutex, the gates wait on a condition
// variable under it, and the kernel hook and the error threads take it too.

#include "fake_hsa.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "rdc_lib/RdcPerfTimer.h"

namespace {

// Handles are small numbers in a range of their own, so that a handle the fake
// never issued is seen as such without being read.
constexpr uint64_t kAgentBase = 0x1000;
constexpr uint64_t kPoolBase = 0x2000;
constexpr uint64_t kSignalBase = 0x3000;
constexpr uint64_t kExecutableBase = 0x5000;
constexpr uint64_t kReaderBase = 0x6000;
constexpr uint64_t kSymbolBase = 0x7000;
constexpr uint64_t kKernelBase = 0x8000;

constexpr size_t kPage = 4096;
constexpr unsigned char kFillByte = 0x5A;
constexpr unsigned char kRedzoneByte = 0xDB;
constexpr size_t kRedzoneBytes = 4096;
constexpr size_t kCallLogLimit = 10000;
constexpr size_t kFillLimit = size_t{1} << 20;
constexpr size_t kPacketBytes = sizeof(hsa_kernel_dispatch_packet_t);

// How long one pass of a wait gives the mutex up for, which is what a step of the wall clock
// can stretch or cut of a wait of this fake.
constexpr std::chrono::milliseconds kWaitSlice(20);

// One pass of a wait that ends at that deadline, which is read from the steady clock, so that
// a step of the wall clock can move neither a hint nor a gate's wait. The pass itself waits to
// a system_clock deadline, because libstdc++ sends a steady_clock one to
// pthread_cond_clockwait, which GCC 11's ThreadSanitizer does not intercept: it would not see
// the wait give the mutex up, and would call the next thread's lock of it a double lock. The
// answer says whether the deadline is still ahead.
bool wait_a_slice(std::condition_variable& cv, std::unique_lock<std::mutex>& lock,
                  std::chrono::steady_clock::time_point deadline) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    return false;
  }
  const auto slice = std::min(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now),
                              std::chrono::duration_cast<std::chrono::nanoseconds>(kWaitSlice));
  cv.wait_until(lock, std::chrono::system_clock::now() + slice);
  return true;
}

struct Agent {
  bool is_gpu = false;
  uint32_t node_id = 0;
  char name[64] = {};
  uint32_t queue_max = 1024;
  uint32_t profile = HSA_PROFILE_FULL;
  std::vector<uint32_t> pools;
};

struct Pool {
  uint32_t agent = 0;
  uint32_t segment = HSA_AMD_SEGMENT_GLOBAL;
  uint32_t global_flags = 0;
  uint64_t size = 64 * 1024;
  uint64_t max_size = 64 * 1024;
  bool alloc_allowed = true;
  uint64_t granule = kPage;
  uint64_t alignment = kPage;
  bool accessible_by_all = false;
};

struct Block {
  unsigned char* map = nullptr;
  size_t map_size = 0;
  unsigned char* data = nullptr;
  size_t size = 0;
  uint32_t pool = 0;
  uint32_t frees = 0;
};

struct Queue {
  hsa_queue_t* queue = nullptr;
  unsigned char* ring = nullptr;
  size_t ring_size = 0;
  uint32_t agent = 0;
  uint32_t size = 0;
  uint32_t private_segment = 0;
  uint32_t group_segment = 0;
  void (*callback)(hsa_status_t, hsa_queue_t*, void*) = nullptr;
  void* data = nullptr;
  uint32_t doorbell = 0;
  uint64_t write_index = 0;
  bool destroyed = false;
  bool stopped = false;
  // The error callbacks of this queue that have not returned; its destroy waits for them.
  uint32_t errors_running = 0;
};

struct Signal {
  int64_t value = 0;
  bool destroyed = false;
  bool is_doorbell = false;
  bool is_completion = false;
  uint32_t queue = 0;
  uint32_t waits = 0;
};

struct Executable {
  bool destroyed = false;
};

struct Reader {
  uint64_t device = 0;
  uint64_t inode = 0;
  bool destroyed = false;
};

struct Symbol {
  std::string name;
  uint64_t kernel_object = 0;
};

struct KernelInfo {
  uint32_t kernarg_size = 80;
  uint32_t kernarg_align = 16;
  uint32_t private_segment = 16;
  uint32_t group_segment = 64;
};

struct Kernel {
  std::string name;
  KernelInfo info;
};

struct Dispatch {
  uint32_t queue = 0;
  uint32_t signal = UINT32_MAX;
  uint64_t grid = 0;
  uint32_t workgroup = 0;
  uint32_t private_segment = 0;
  uint32_t group_segment = 0;
  uint64_t kernarg_address = 0;
  uint32_t reserved0 = 0;
  uint64_t reserved2 = 0;
  uint32_t header = 0;
  uint32_t setup = 0;
  std::vector<unsigned char> kernarg;
};

struct Failure {
  uint32_t nth = 0;
  int32_t status = 0;
  bool throws = false;
  int value = 0;
};

struct Gate {
  uint64_t caller = 0;
  uint32_t nth = 0;
  uint32_t held = 0;
  bool released = false;
};

struct State {
  std::mutex mutex;
  std::condition_variable cv;

  std::vector<Agent> agents;
  std::vector<Pool> pools;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> access;
  std::map<std::pair<uint32_t, uint32_t>, int32_t> agent_info_failures;
  std::map<std::pair<uint32_t, uint32_t>, int32_t> pool_info_failures;

  std::deque<Block> blocks;
  std::deque<Queue> queues;
  // The queues a reset does not free: the ones the module never destroyed, with the error
  // data it left with them. They stay alive and reachable for the life of the process.
  std::deque<Queue> kept_queues;
  std::deque<Signal> signals;
  std::deque<Executable> executables;
  std::deque<Reader> readers;
  std::deque<Symbol> symbols;
  std::deque<Kernel> kernels;
  std::deque<Dispatch> dispatches;

  std::map<std::string, KernelInfo> kernel_info;
  KernelInfo default_kernel_info;

  std::map<std::string, uint32_t> call_counts;
  std::deque<std::string> call_log;
  std::deque<std::string> destroy_log;
  std::map<std::string, Failure> failures;
  std::map<std::string, Gate> gates;
  std::map<uint32_t, uint32_t> node_ids;
  std::map<uint32_t, int32_t> node_failures;

  // Tells the runs of the registry apart: a reset raises it, so that an error callback
  // still running knows that the queue it was raised for is gone.
  uint64_t generation = 0;

  uint32_t references = 0;
  uint32_t loads = 0;
  uint32_t failed_loads = 0;
  uint32_t unbalanced_shutdowns = 0;
  uint32_t refused_calls = 0;
  uint32_t foreign_handles = 0;
  uint32_t foreign_pools = 0;
  uint32_t access_reads = 0;
  uint32_t double_frees = 0;
  uint32_t unissued_frees = 0;
  uint32_t redzone_failures = 0;
  uint32_t queue_errors_recorded = 0;
  uint32_t copies = 0;
  uint32_t waits = 0;
  uint32_t grid_violations = 0;
  uint32_t node_id_calls = 0;
  bool latched = false;

  std::vector<uint64_t> wait_hints;
  std::vector<int> wait_states;
  std::string last_grid_violation;

  uint64_t timestamp_frequency = 25000000;
  uint32_t kernarg_offset = 0;
  bool alias_pages = false;
  int kernel_mode = RDC_FAKE_KERNEL_NORMAL;
  int copy_mode = RDC_FAKE_COPY_NORMAL;
  uint32_t search_element0 = 0;
  uint32_t search_element3 = 0;
  int64_t dispatch_signal_value = 0;
  uint32_t completion_wait = 0;
};

// The registry lives as long as the process: a case reads it after a run, and an
// abandoned check's objects must stay reachable, so that LeakSanitizer does not
// report them.
State& state() {
  static State* const kState = new State();
  return *kState;
}

thread_local uint64_t t_caller = 0;

uint64_t round_up(uint64_t value, uint64_t to) { return ((value + to - 1) / to) * to; }

void add_pool_locked(State& s, uint32_t agent, uint32_t segment, uint32_t flags, uint64_t size,
                     uint64_t max_size, bool allowed, uint64_t granule, uint64_t alignment,
                     bool accessible) {
  Pool pool;
  pool.agent = agent;
  pool.segment = segment;
  pool.global_flags = flags;
  pool.size = size;
  pool.max_size = max_size;
  pool.alloc_allowed = allowed;
  pool.granule = granule;
  pool.alignment = alignment;
  pool.accessible_by_all = accessible;
  s.pools.push_back(pool);
  s.agents[agent].pools.push_back(static_cast<uint32_t>(s.pools.size() - 1));
}

void set_name(Agent* agent, const char* name, bool terminate) {
  std::memset(agent->name, 0, sizeof(agent->name));
  const size_t length = (name == nullptr) ? 0 : std::strlen(name);
  if (terminate) {
    std::memcpy(agent->name, name, std::min(length, sizeof(agent->name) - 1));
    return;
  }
  // All 64 bytes carry the name, repeated, and none of them is a NUL.
  for (size_t i = 0; i < sizeof(agent->name); i++) {
    agent->name[i] = (length == 0) ? 'x' : name[i % length];
  }
}

uint32_t add_agent_locked(State& s, bool is_gpu, uint32_t node_id, const char* name) {
  Agent agent;
  agent.is_gpu = is_gpu;
  agent.node_id = node_id;
  s.agents.push_back(agent);
  const uint32_t index = static_cast<uint32_t>(s.agents.size() - 1);
  set_name(&s.agents[index], name, true);
  return index;
}

// The agents and the pools fake_hsa.h describes.
void make_default_agents(State& s) {
  const uint32_t cpu = add_agent_locked(s, false, 0, "CPU");
  add_pool_locked(s, cpu, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
                  64 * 1024, 64 * 1024, true, kPage, kPage, true);
  add_pool_locked(
      s, cpu, HSA_AMD_SEGMENT_GLOBAL,
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT | HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
      64 * 1024, 64 * 1024, true, kPage, kPage, true);
  add_pool_locked(s, cpu, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                  64 * 1024, 64 * 1024, true, kPage, kPage, true);

  const uint32_t first = add_agent_locked(s, true, 1, "gfx1030");
  add_pool_locked(s, first, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                  64 * 1024, 64 * 1024, true, kPage, kPage, false);
  const uint32_t second = add_agent_locked(s, true, 2, "gfx1100");
  add_pool_locked(s, second, HSA_AMD_SEGMENT_GLOBAL, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                  64 * 1024, 64 * 1024, true, kPage, kPage, false);
}

void reset_locked(State& s) {
  for (Block& block : s.blocks) {
    ::munmap(block.map, block.map_size);
  }
  // The blocks of a queue the module destroyed go with its record; a queue it never destroyed
  // stays alive and reachable, with the error data the module left with it, for the life of the
  // process, so that LeakSanitizer reports neither.
  for (Queue& record : s.queues) {
    if (record.destroyed) {
      std::free(record.queue);
      std::free(record.ring);
    } else {
      s.kept_queues.push_back(std::move(record));
    }
  }
  s.generation++;
  s.agents.clear();
  s.pools.clear();
  s.access.clear();
  s.agent_info_failures.clear();
  s.pool_info_failures.clear();
  s.blocks.clear();
  s.queues.clear();
  s.signals.clear();
  s.executables.clear();
  s.readers.clear();
  s.symbols.clear();
  s.kernels.clear();
  s.dispatches.clear();
  s.kernel_info.clear();
  s.default_kernel_info = KernelInfo();
  s.call_counts.clear();
  s.call_log.clear();
  s.destroy_log.clear();
  s.failures.clear();
  s.gates.clear();
  s.node_ids.clear();
  s.node_failures.clear();
  s.references = 0;
  s.loads = 0;
  s.failed_loads = 0;
  s.unbalanced_shutdowns = 0;
  s.refused_calls = 0;
  s.foreign_handles = 0;
  s.foreign_pools = 0;
  s.access_reads = 0;
  s.double_frees = 0;
  s.unissued_frees = 0;
  s.redzone_failures = 0;
  s.queue_errors_recorded = 0;
  s.copies = 0;
  s.waits = 0;
  s.grid_violations = 0;
  s.node_id_calls = 0;
  s.latched = false;
  s.wait_hints.clear();
  s.wait_states.clear();
  s.last_grid_violation.clear();
  s.timestamp_frequency = 25000000;
  s.kernarg_offset = 0;
  s.alias_pages = false;
  s.kernel_mode = RDC_FAKE_KERNEL_NORMAL;
  s.copy_mode = RDC_FAKE_COPY_NORMAL;
  s.search_element0 = 0;
  s.search_element3 = 0;
  s.dispatch_signal_value = 0;
  s.completion_wait = 0;
  make_default_agents(s);
}

// Set up the default state the first time anything calls the fake, so that a
// case which forgets rdc_fake_hsa_reset() still finds agents and pools.
State& started() {
  State& s = state();
  static std::once_flag once;
  std::call_once(once, [&s]() { make_default_agents(s); });
  return s;
}

void raise_queue_error(uint32_t queue, int32_t status);

// Records the call, holds it at a gate keyed to this caller, raises an armed
// queue error and applies a failure switch. It returns true when the entry point
// must return *status at once; it throws an int when a case armed that.
bool enter_call(std::unique_lock<std::mutex>& lock, const char* name, bool needs_reference,
                hsa_status_t* status) {
  State& s = started();
  const std::string key(name);
  const uint32_t count = ++s.call_counts[key];
  s.call_log.push_back(key);
  while (s.call_log.size() > kCallLogLimit) {
    s.call_log.pop_front();
  }

  auto gate = s.gates.find(key);
  if (gate != s.gates.end() && !gate->second.released &&
      (gate->second.caller == 0 || gate->second.caller == t_caller) &&
      (gate->second.nth == 0 || gate->second.nth == count)) {
    gate->second.held++;
    s.cv.notify_all();
    while (!s.gates[key].released) {
      s.cv.wait(lock);
    }
    s.gates[key].held--;
    s.cv.notify_all();
  }

  auto failure = s.failures.find(key);
  if (failure != s.failures.end() && (failure->second.nth == 0 || failure->second.nth == count)) {
    const Failure fired = failure->second;
    if (fired.throws) {
      throw fired.value;
    }
    *status = static_cast<hsa_status_t>(fired.status);
    return true;
  }

  if (needs_reference && s.references == 0) {
    s.refused_calls++;
    *status = HSA_STATUS_ERROR_NOT_INITIALIZED;
    return true;
  }
  return false;
}

Agent* find_agent(State& s, hsa_agent_t agent) {
  const uint64_t index = agent.handle - kAgentBase;
  if (agent.handle < kAgentBase || index >= s.agents.size()) {
    s.foreign_handles++;
    return nullptr;
  }
  return &s.agents[static_cast<size_t>(index)];
}

uint32_t agent_index(State& s, hsa_agent_t agent) {
  const uint64_t index = agent.handle - kAgentBase;
  if (agent.handle < kAgentBase || index >= s.agents.size()) {
    return UINT32_MAX;
  }
  return static_cast<uint32_t>(index);
}

Pool* find_pool(State& s, hsa_amd_memory_pool_t pool) {
  const uint64_t index = pool.handle - kPoolBase;
  if (pool.handle < kPoolBase || index >= s.pools.size()) {
    s.foreign_pools++;
    return nullptr;
  }
  return &s.pools[static_cast<size_t>(index)];
}

Signal* find_signal(State& s, hsa_signal_t signal) {
  if (signal.handle == 0) {
    // A handle of 0 is no signal at all, which HSA allows where one is optional.
    return nullptr;
  }
  const uint64_t index = signal.handle - kSignalBase;
  if (signal.handle < kSignalBase || index >= s.signals.size()) {
    s.foreign_handles++;
    return nullptr;
  }
  return &s.signals[static_cast<size_t>(index)];
}

Queue* find_queue(State& s, const hsa_queue_t* queue) {
  if (queue == nullptr) {
    s.foreign_handles++;
    return nullptr;
  }
  for (Queue& known : s.queues) {
    if (known.queue == queue) {
      return &known;
    }
  }
  s.foreign_handles++;
  return nullptr;
}

Executable* find_executable(State& s, hsa_executable_t executable) {
  const uint64_t index = executable.handle - kExecutableBase;
  if (executable.handle < kExecutableBase || index >= s.executables.size()) {
    s.foreign_handles++;
    return nullptr;
  }
  return &s.executables[static_cast<size_t>(index)];
}

Reader* find_reader(State& s, hsa_code_object_reader_t reader) {
  const uint64_t index = reader.handle - kReaderBase;
  if (reader.handle < kReaderBase || index >= s.readers.size()) {
    s.foreign_handles++;
    return nullptr;
  }
  return &s.readers[static_cast<size_t>(index)];
}

Symbol* find_symbol(State& s, hsa_executable_symbol_t symbol) {
  const uint64_t index = symbol.handle - kSymbolBase;
  if (symbol.handle < kSymbolBase || index >= s.symbols.size()) {
    s.foreign_handles++;
    return nullptr;
  }
  return &s.symbols[static_cast<size_t>(index)];
}

Block* find_block(State& s, const void* pointer) {
  const unsigned char* address = static_cast<const unsigned char*>(pointer);
  for (Block& block : s.blocks) {
    if (address >= block.data && address < block.data + block.size) {
      return &block;
    }
  }
  return nullptr;
}

KernelInfo kernel_info_for(State& s, const std::string& symbol) {
  auto exact = s.kernel_info.find(symbol);
  if (exact != s.kernel_info.end()) {
    return exact->second;
  }
  for (const auto& entry : s.kernel_info) {
    if (!entry.first.empty() && symbol.find(entry.first) != std::string::npos) {
      return entry.second;
    }
  }
  return s.default_kernel_info;
}

void note_destroy(State& s, const char* kind, size_t index) {
  s.destroy_log.push_back(std::string(kind) + " " + std::to_string(index));
}

bool condition_holds(hsa_signal_condition_t condition, int64_t value, int64_t compare) {
  switch (condition) {
    case HSA_SIGNAL_CONDITION_EQ:
      return value == compare;
    case HSA_SIGNAL_CONDITION_NE:
      return value != compare;
    case HSA_SIGNAL_CONDITION_LT:
      return value < compare;
    case HSA_SIGNAL_CONDITION_GTE:
      return value >= compare;
    default:
      return true;
  }
}

void fill_fresh(unsigned char* data, size_t size) {
  if (size <= kFillLimit) {
    std::memset(data, kFillByte, size);
    return;
  }
  // A large block gets the fill on its first and last pages only.
  std::memset(data, kFillByte, kPage);
  std::memset(data + size - kPage, kFillByte, kPage);
}

bool redzone_intact(const Block& block) {
  for (size_t i = 0; i < kRedzoneBytes; i++) {
    if (block.data[block.size + i] != kRedzoneByte) {
      return false;
    }
  }
  return true;
}

// Frees a block, or records a free of a pointer the fake never issued. The
// mapping stays, so that a second free is counted rather than a crash.
hsa_status_t free_pointer(State& s, void* pointer) {
  if (pointer == nullptr) {
    return HSA_STATUS_SUCCESS;
  }
  for (size_t i = 0; i < s.blocks.size(); i++) {
    Block& block = s.blocks[i];
    if (block.data != static_cast<unsigned char*>(pointer)) {
      continue;
    }
    if (!redzone_intact(block)) {
      s.redzone_failures++;
    }
    block.frees++;
    if (block.frees > 1) {
      s.double_frees++;
    }
    note_destroy(s, "block", i);
    return HSA_STATUS_SUCCESS;
  }
  s.unissued_frees++;
  return HSA_STATUS_ERROR_INVALID_ALLOCATION;
}

void note_violation(State& s, const std::string& text) {
  s.grid_violations++;
  s.last_grid_violation = text;
}

// How many bytes of a block are left from an address inside it.
size_t bytes_left(State& s, uint64_t address) {
  const Block* block = find_block(s, reinterpret_cast<const void*>(address));
  if (block == nullptr) {
    return 0;
  }
  return static_cast<size_t>(block->data + block->size - reinterpret_cast<unsigned char*>(address));
}

// Runs the kernel the packet names, with the rules the design gives each of the
// two kernels: it checks that the grid fits the buffers the arguments point to,
// and writes what the kernel writes, nothing, or the read-back shifted.
void run_kernel(State& s, Dispatch* record, uint64_t kernel_object) {
  std::string name;
  KernelInfo info = s.default_kernel_info;
  const uint64_t index = kernel_object - kKernelBase;
  if (kernel_object >= kKernelBase && index < s.kernels.size()) {
    name = s.kernels[static_cast<size_t>(index)].name;
    info = s.kernels[static_cast<size_t>(index)].info;
  }

  const size_t left = bytes_left(s, record->kernarg_address);
  const size_t wanted = info.kernarg_size;
  const size_t recorded = std::min(left, wanted);
  if (left < wanted) {
    note_violation(s, "the kernel argument block holds " + std::to_string(left) +
                          " bytes from the argument address; the segment is " +
                          std::to_string(wanted));
  }
  record->kernarg.assign(recorded, 0);
  if (recorded != 0) {
    std::memcpy(record->kernarg.data(), reinterpret_cast<const void*>(record->kernarg_address),
                recorded);
  }

  if (s.kernel_mode == RDC_FAKE_KERNEL_NOTHING || s.kernel_mode == RDC_FAKE_KERNEL_NEVER) {
    return;
  }

  const unsigned char* args = record->kernarg.data();
  const uint64_t grid = record->grid;
  if (name.find("binarySearch") != std::string::npos) {
    if (recorded < 20) {
      return;
    }
    uint64_t output_address = 0;
    uint64_t sorted_address = 0;
    uint32_t find_me = 0;
    std::memcpy(&output_address, args, sizeof(output_address));
    std::memcpy(&sorted_address, args + 8, sizeof(sorted_address));
    std::memcpy(&find_me, args + 16, sizeof(find_me));
    const size_t output_bytes = bytes_left(s, output_address);
    const size_t sorted_bytes = bytes_left(s, sorted_address);
    if (output_bytes < 16) {
      note_violation(s, "the search kernel's output buffer holds " + std::to_string(output_bytes) +
                            " bytes; it needs 16");
    }
    if (sorted_bytes < grid * 8) {
      note_violation(s, "the search kernel's grid of " + std::to_string(grid) + " needs " +
                            std::to_string(grid * 8) +
                            " bytes of sorted values; its buffer holds " +
                            std::to_string(sorted_bytes));
    }
    if (output_bytes < 16) {
      return;
    }
    uint32_t* output = reinterpret_cast<uint32_t*>(output_address);
    if (s.kernel_mode == RDC_FAKE_KERNEL_OUTPUT) {
      output[0] = s.search_element0;
      output[3] = s.search_element3;
      return;
    }
    const uint32_t* sorted = reinterpret_cast<const uint32_t*>(sorted_address);
    const uint64_t pairs = std::min<uint64_t>(grid, sorted_bytes / 8);
    for (uint64_t tid = 0; tid < pairs; tid++) {
      if (sorted[2 * tid] <= find_me && find_me <= sorted[2 * tid + 1]) {
        output[0] = static_cast<uint32_t>(tid);
        output[3] = 1;
      }
    }
    return;
  }

  if (name.find("gpuReadWrite") != std::string::npos) {
    if (recorded < 24) {
      return;
    }
    uint64_t a_address = 0;
    uint64_t b_address = 0;
    uint64_t c_address = 0;
    std::memcpy(&a_address, args, sizeof(a_address));
    std::memcpy(&b_address, args + 8, sizeof(b_address));
    std::memcpy(&c_address, args + 16, sizeof(c_address));
    const size_t wanted_bytes = static_cast<size_t>(grid) * sizeof(int);
    const size_t a_bytes = bytes_left(s, a_address);
    const size_t b_bytes = bytes_left(s, b_address);
    const size_t c_bytes = bytes_left(s, c_address);
    if (a_bytes < wanted_bytes || b_bytes < wanted_bytes || c_bytes < wanted_bytes) {
      note_violation(s, "the read and write kernel's grid of " + std::to_string(grid) + " needs " +
                            std::to_string(wanted_bytes) + " bytes of each buffer; they hold " +
                            std::to_string(a_bytes) + ", " + std::to_string(b_bytes) + " and " +
                            std::to_string(c_bytes));
    }
    const size_t items = std::min(static_cast<size_t>(grid),
                                  std::min(a_bytes, std::min(b_bytes, c_bytes)) / sizeof(int));
    if (items == 0) {
      return;
    }
    const int* source = reinterpret_cast<const int*>(a_address);
    int* written = reinterpret_cast<int*>(b_address);
    int* read_back = reinterpret_cast<int*>(c_address);
    for (size_t i = 0; i < items; i++) {
      const size_t from = (s.kernel_mode == RDC_FAKE_KERNEL_SHIFTED) ? (i + 1) % items : i;
      read_back[i] = source[from];
      written[i] = static_cast<int>(i);
    }
  }
}

// The doorbell: the fake reads the packet the check made ready and runs it.
void ring_doorbell(State& s, Signal* doorbell, int64_t index) {
  if (doorbell->queue >= s.queues.size()) {
    return;
  }
  Queue& queue = s.queues[doorbell->queue];
  if (queue.size == 0 || queue.ring == nullptr) {
    return;
  }
  const uint64_t slot = static_cast<uint64_t>(index) & (queue.size - 1);
  hsa_kernel_dispatch_packet_t packet = {};
  std::memcpy(&packet, queue.ring + slot * kPacketBytes, kPacketBytes);

  Dispatch record;
  record.queue = doorbell->queue;
  record.grid = packet.grid_size_x;
  record.workgroup = packet.workgroup_size_x;
  record.private_segment = packet.private_segment_size;
  record.group_segment = packet.group_segment_size;
  record.kernarg_address = reinterpret_cast<uint64_t>(packet.kernarg_address);
  record.reserved0 = packet.reserved0;
  record.reserved2 = packet.reserved2;
  record.header = packet.header;
  record.setup = packet.setup;
  s.dispatches.push_back(record);
  Dispatch& kept = s.dispatches.back();
  run_kernel(s, &kept, packet.kernel_object);

  Signal* completion = find_signal(s, packet.completion_signal);
  if (completion == nullptr) {
    return;
  }
  kept.signal = static_cast<uint32_t>(packet.completion_signal.handle - kSignalBase);
  completion->is_completion = true;
  if (s.kernel_mode == RDC_FAKE_KERNEL_NEVER || s.completion_wait != 0) {
    return;
  }
  completion->value = s.dispatch_signal_value;
  s.cv.notify_all();
}

// Raises a queue error from a thread of the fake's: the queue's callback runs
// without the mutex, so that it may call back into the fake. The thread is
// detached and reads nothing but the registry, which lives as long as the
// process, and the generation tells it that its queue belongs to a run that is
// over, so a queue a check abandoned leaves no thread behind.
void raise_queue_error_body(uint64_t generation, uint32_t queue, int32_t status) {
  State& s = state();
  void (*callback)(hsa_status_t, hsa_queue_t*, void*) = nullptr;
  void* data = nullptr;
  hsa_queue_t* handle = nullptr;
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    if (generation != s.generation || queue >= s.queues.size()) {
      return;
    }
    Queue& record = s.queues[queue];
    record.stopped = true;
    callback = record.callback;
    data = record.data;
    handle = record.queue;
    if (callback == nullptr) {
      s.queue_errors_recorded++;
    }
    s.cv.notify_all();
  }
  if (callback != nullptr) {
    callback(static_cast<hsa_status_t>(status), handle, data);
  }
  // The queue's hsa_queue_destroy waits for this, as ~AqlQueue waits for its
  // error handler.
  std::lock_guard<std::mutex> guard(s.mutex);
  if (generation == s.generation && queue < s.queues.size() && s.queues[queue].errors_running > 0) {
    s.queues[queue].errors_running--;
  }
  s.cv.notify_all();
}

void raise_queue_error(uint32_t queue, int32_t status) {
  State& s = state();
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    if (queue >= s.queues.size()) {
      return;
    }
    s.queues[queue].errors_running++;
    generation = s.generation;
  }
  std::thread(raise_queue_error_body, generation, queue, status).detach();
}

}  // namespace

// --- the HSA functions ----------------------------------------------------------

hsa_status_t hsa_init() {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_init", false, &status)) {
    if (s.references == 0) {
      s.failed_loads++;
    }
    return status;
  }
  if (s.latched) {
    if (s.references == 0) {
      s.failed_loads++;
    }
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  if (s.references == 0) {
    s.loads++;
  }
  s.references++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_shut_down() {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_shut_down", false, &status)) {
    return status;
  }
  if (s.references == 0) {
    s.unbalanced_shutdowns++;
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  }
  s.references--;
  s.destroy_log.push_back("shut_down");
  if (s.references == 0) {
    for (const Queue& queue : s.queues) {
      if (!queue.destroyed) {
        // A shutdown to zero with a queue alive latches ROCr for the life of the
        // process, as it does on the hardware, while the shutdown succeeds.
        s.latched = true;
        break;
      }
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_status_string(hsa_status_t status, const char** status_string) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t fired = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_status_string", false, &fired)) {
    return fired;
  }
  if (status_string == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  static const struct {
    hsa_status_t status;
    const char* text;
  } kStrings[] = {
      {HSA_STATUS_SUCCESS, "The function has been executed successfully."},
      {HSA_STATUS_INFO_BREAK, "Traversal has been interrupted."},
      {HSA_STATUS_ERROR, "A generic error has occurred."},
      {HSA_STATUS_ERROR_INVALID_ARGUMENT, "One of the actual arguments is invalid."},
      {HSA_STATUS_ERROR_INVALID_QUEUE_CREATION, "The queue could not be created."},
      {HSA_STATUS_ERROR_INVALID_ALLOCATION, "The requested allocation is not valid."},
      {HSA_STATUS_ERROR_INVALID_AGENT, "The agent is invalid."},
      {HSA_STATUS_ERROR_INVALID_REGION, "The memory region is invalid."},
      {HSA_STATUS_ERROR_INVALID_SIGNAL, "The signal is invalid."},
      {HSA_STATUS_ERROR_INVALID_QUEUE, "The queue is invalid."},
      {HSA_STATUS_ERROR_OUT_OF_RESOURCES, "The runtime failed to allocate the resources."},
      {HSA_STATUS_ERROR_INVALID_PACKET_FORMAT, "The packet format is invalid."},
      {HSA_STATUS_ERROR_RESOURCE_FREE, "An error has been detected while releasing a resource."},
      {HSA_STATUS_ERROR_NOT_INITIALIZED, "An API other than hsa_init has been called."},
      {HSA_STATUS_ERROR_REFCOUNT_OVERFLOW, "The maximum reference count has been reached."},
      {HSA_STATUS_ERROR_INVALID_INDEX, "The index is invalid."},
      {HSA_STATUS_ERROR_INVALID_ISA, "The instruction set architecture is invalid."},
      {HSA_STATUS_ERROR_INVALID_CODE_OBJECT, "The code object is invalid."},
      {HSA_STATUS_ERROR_INVALID_EXECUTABLE, "The executable is invalid."},
      {HSA_STATUS_ERROR_FROZEN_EXECUTABLE, "The executable is frozen."},
      {HSA_STATUS_ERROR_INVALID_SYMBOL_NAME, "There is no symbol with that name."},
      {HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL, "The executable symbol is invalid."},
      {HSA_STATUS_ERROR_INVALID_FILE, "The file descriptor is invalid."},
      {HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER, "The code object reader is invalid."},
      {HSA_STATUS_ERROR_EXCEPTION, "An exception was raised while running a kernel."},
      {HSA_STATUS_ERROR_FATAL, "The runtime is in an unrecoverable state."},
  };
  for (const auto& entry : kStrings) {
    if (entry.status == status) {
      *status_string = entry.text;
      return HSA_STATUS_SUCCESS;
    }
  }
  // ROCr leaves the output untouched for a status it has no string for.
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t hsa_system_get_info(hsa_system_info_t attribute, void* value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_system_get_info", true, &status)) {
    return status;
  }
  if (value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (attribute == HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY) {
    const uint64_t frequency = s.timestamp_frequency;
    std::memcpy(value, &frequency, sizeof(frequency));
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data),
                                void* data) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_iterate_agents", true, &status)) {
    return status;
  }
  if (callback == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const size_t count = s.agents.size();
  for (size_t i = 0; i < count; i++) {
    hsa_agent_t agent = {kAgentBase + i};
    hsa_status_t answer = HSA_STATUS_SUCCESS;
    lock.unlock();
    try {
      answer = callback(agent, data);
    } catch (...) {
      // ROCr turns an exception out of a callback into an error.
      lock.lock();
      return HSA_STATUS_ERROR;
    }
    lock.lock();
    if (answer != HSA_STATUS_SUCCESS) {
      return answer;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_agent_get_info", true, &status)) {
    return status;
  }
  const uint32_t index = agent_index(s, agent);
  Agent* known = find_agent(s, agent);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const auto one = s.agent_info_failures.find(std::make_pair(index, uint32_t{attribute}));
  if (one != s.agent_info_failures.end()) {
    return static_cast<hsa_status_t>(one->second);
  }
  const auto every = s.agent_info_failures.find(std::make_pair(UINT32_MAX, uint32_t{attribute}));
  if (every != s.agent_info_failures.end()) {
    return static_cast<hsa_status_t>(every->second);
  }

  switch (static_cast<int>(attribute)) {
    case HSA_AGENT_INFO_DEVICE: {
      const hsa_device_type_t type = known->is_gpu ? HSA_DEVICE_TYPE_GPU : HSA_DEVICE_TYPE_CPU;
      std::memcpy(value, &type, sizeof(type));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AGENT_INFO_NAME:
      std::memcpy(value, known->name, sizeof(known->name));
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_PROFILE: {
      const hsa_profile_t profile = static_cast<hsa_profile_t>(known->profile);
      std::memcpy(value, &profile, sizeof(profile));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AGENT_INFO_NODE:
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID: {
      const uint32_t node = known->node_id;
      std::memcpy(value, &node, sizeof(node));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE: {
      const uint32_t packets = known->queue_max;
      std::memcpy(value, &packets, sizeof(packets));
      return HSA_STATUS_SUCCESS;
    }
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                               void* data),
                              void* data, uint32_t private_segment_size,
                              uint32_t group_segment_size, hsa_queue_t** queue) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_queue_create", true, &status)) {
    return status;
  }
  const uint32_t index = agent_index(s, agent);
  if (find_agent(s, agent) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (queue == nullptr || size == 0 || (size & (size - 1)) != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Queue record;
  record.agent = index;
  record.size = size;
  record.private_segment = private_segment_size;
  record.group_segment = group_segment_size;
  record.callback = callback;
  record.data = data;
  record.ring_size = static_cast<size_t>(size) * kPacketBytes;
  record.ring = static_cast<unsigned char*>(std::malloc(record.ring_size));
  if (record.ring == nullptr) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  // ROCr leaves a new ring's packet contents as they are.
  std::memset(record.ring, kFillByte, record.ring_size);
  record.queue = static_cast<hsa_queue_t*>(std::calloc(1, sizeof(hsa_queue_t)));
  if (record.queue == nullptr) {
    std::free(record.ring);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  s.queues.push_back(std::move(record));
  const uint32_t queue_index = static_cast<uint32_t>(s.queues.size() - 1);
  Signal doorbell;
  doorbell.is_doorbell = true;
  doorbell.queue = queue_index;
  s.signals.push_back(doorbell);
  const uint32_t doorbell_index = static_cast<uint32_t>(s.signals.size() - 1);

  Queue& kept = s.queues[queue_index];
  kept.doorbell = doorbell_index;
  kept.queue->type = type;
  kept.queue->features = 1;
  kept.queue->base_address = kept.ring;
  kept.queue->doorbell_signal.handle = kSignalBase + doorbell_index;
  kept.queue->size = size;
  kept.queue->id = queue_index;
  *queue = kept.queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_queue_destroy(hsa_queue_t* queue) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_queue_destroy", true, &status)) {
    return status;
  }
  Queue* known = find_queue(s, queue);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }
  known->destroyed = true;
  size_t index = s.queues.size();
  for (size_t i = 0; i < s.queues.size(); i++) {
    if (&s.queues[i] == known) {
      index = i;
      note_destroy(s, "queue", i);
      break;
    }
  }
  // ~AqlQueue waits for the queue's error handler before it returns.
  while (index < s.queues.size() && s.queues[index].errors_running > 0) {
    s.cv.wait(lock);
  }
  return HSA_STATUS_SUCCESS;
}

uint64_t hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_queue_load_write_index_relaxed", true, &status)) {
    return 0;
  }
  Queue* known = find_queue(s, queue);
  return (known == nullptr) ? 0 : known->write_index;
}

uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_queue_add_write_index_relaxed", true, &status)) {
    return 0;
  }
  Queue* known = find_queue(s, queue);
  if (known == nullptr) {
    return 0;
  }
  const uint64_t before = known->write_index;
  known->write_index += value;
  return before;
}

void hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_queue_store_write_index_relaxed", true, &status)) {
    return;
  }
  Queue* known = find_queue(s, queue);
  if (known != nullptr) {
    known->write_index = value;
  }
}

hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value, uint32_t,
                               const hsa_agent_t* consumers, hsa_signal_t* signal) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_signal_create", true, &status)) {
    return status;
  }
  (void)consumers;
  if (signal == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  Signal record;
  record.value = initial_value;
  s.signals.push_back(record);
  signal->handle = kSignalBase + (s.signals.size() - 1);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_signal_destroy(hsa_signal_t signal) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_signal_destroy", true, &status)) {
    return status;
  }
  Signal* known = find_signal(s, signal);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_SIGNAL;
  }
  known->destroyed = true;
  note_destroy(s, "signal", static_cast<size_t>(signal.handle - kSignalBase));
  return HSA_STATUS_SUCCESS;
}

namespace {

void store_signal(const char* name, hsa_signal_t signal, hsa_signal_value_t value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, name, true, &status)) {
    return;
  }
  Signal* known = find_signal(s, signal);
  if (known == nullptr) {
    return;
  }
  if (known->is_doorbell) {
    ring_doorbell(s, known, value);
    return;
  }
  known->value = value;
  s.cv.notify_all();
}

}  // namespace

void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  store_signal("hsa_signal_store_relaxed", signal, value);
}

void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  store_signal("hsa_signal_store_screlease", signal, value);
}

hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal, hsa_signal_condition_t condition,
                                             hsa_signal_value_t compare_value,
                                             uint64_t timeout_hint, hsa_wait_state_t wait_state) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_signal_wait_scacquire", true, &status)) {
    return 0;
  }
  Signal* known = find_signal(s, signal);
  if (known == nullptr) {
    return 0;
  }
  s.waits++;
  const int state = static_cast<int>(wait_state);
  if (std::find(s.wait_states.begin(), s.wait_states.end(), state) == s.wait_states.end()) {
    s.wait_states.push_back(state);
  }
  if (std::find(s.wait_hints.begin(), s.wait_hints.end(), timeout_hint) == s.wait_hints.end()) {
    s.wait_hints.push_back(timeout_hint);
  }
  known->waits++;
  if (s.completion_wait != 0 && known->is_completion && known->waits >= s.completion_wait) {
    known->value = s.dispatch_signal_value;
    s.cv.notify_all();
  }
  if (condition_holds(condition, known->value, compare_value)) {
    return known->value;
  }
  if (timeout_hint == UINT64_MAX) {
    while (!condition_holds(condition, known->value, compare_value)) {
      s.cv.wait(lock);
    }
    return known->value;
  }
  const double seconds =
      static_cast<double>(timeout_hint) /
      static_cast<double>(s.timestamp_frequency == 0 ? 1 : s.timestamp_frequency);
  // The hint ends at a deadline of its own, in slices, as wait_a_slice says.
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(seconds));
  while (!condition_holds(condition, known->value, compare_value)) {
    if (!wait_a_slice(s.cv, lock, deadline)) {
      break;
    }
  }
  return known->value;
}

hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_memory_copy", true, &status)) {
    return status;
  }
  if (dst == nullptr || src == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::memcpy(dst, src, size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_memory_free(void* ptr) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_memory_free", true, &status)) {
    return status;
  }
  // The same entry as hsa_amd_memory_pool_free, as it is in ROCr.
  return free_pointer(s, ptr);
}

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t file,
                                                     hsa_code_object_reader_t* code_object_reader) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_code_object_reader_create_from_file", true, &status)) {
    return status;
  }
  if (code_object_reader == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  Reader record;
  struct stat file_stat = {};
  if (::fstat(file, &file_stat) != 0) {
    return HSA_STATUS_ERROR_INVALID_FILE;
  }
  record.device = file_stat.st_dev;
  record.inode = file_stat.st_ino;
  s.readers.push_back(record);
  code_object_reader->handle = kReaderBase + (s.readers.size() - 1);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_code_object_reader_destroy", true, &status)) {
    return status;
  }
  Reader* known = find_reader(s, code_object_reader);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  }
  known->destroyed = true;
  note_destroy(s, "reader", static_cast<size_t>(code_object_reader.handle - kReaderBase));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_create_alt(hsa_profile_t, hsa_default_float_rounding_mode_t,
                                       const char*, hsa_executable_t* executable) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_executable_create_alt", true, &status)) {
    return status;
  }
  if (executable == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  s.executables.push_back(Executable());
  executable->handle = kExecutableBase + (s.executables.size() - 1);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_destroy(hsa_executable_t executable) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_executable_destroy", true, &status)) {
    return status;
  }
  Executable* known = find_executable(s, executable);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  }
  known->destroyed = true;
  note_destroy(s, "executable", static_cast<size_t>(executable.handle - kExecutableBase));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                                   hsa_code_object_reader_t code_object_reader,
                                                   const char*, hsa_loaded_code_object_t*) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_executable_load_agent_code_object", true, &status)) {
    return status;
  }
  if (find_executable(s, executable) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  }
  if (find_agent(s, agent) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (find_reader(s, code_object_reader) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_freeze(hsa_executable_t executable, const char*) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_executable_freeze", true, &status)) {
    return status;
  }
  if (find_executable(s, executable) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_get_symbol(hsa_executable_t executable, const char*,
                                       const char* symbol_name, hsa_agent_t agent, int32_t,
                                       hsa_executable_symbol_t* symbol) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_executable_get_symbol", true, &status)) {
    return status;
  }
  if (find_executable(s, executable) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  }
  if (find_agent(s, agent) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (symbol == nullptr || symbol_name == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  Kernel kernel;
  kernel.name = symbol_name;
  kernel.info = kernel_info_for(s, kernel.name);
  s.kernels.push_back(kernel);
  Symbol record;
  record.name = symbol_name;
  record.kernel_object = kKernelBase + (s.kernels.size() - 1);
  s.symbols.push_back(record);
  symbol->handle = kSymbolBase + (s.symbols.size() - 1);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                            hsa_executable_symbol_info_t attribute, void* value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_executable_symbol_get_info", true, &status)) {
    return status;
  }
  Symbol* known = find_symbol(s, executable_symbol);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL;
  }
  if (value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const KernelInfo info = s.kernels[static_cast<size_t>(known->kernel_object - kKernelBase)].info;
  switch (static_cast<int>(attribute)) {
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: {
      const uint64_t object = known->kernel_object;
      std::memcpy(value, &object, sizeof(object));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
      std::memcpy(value, &info.kernarg_size, sizeof(info.kernarg_size));
      return HSA_STATUS_SUCCESS;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:
      std::memcpy(value, &info.kernarg_align, sizeof(info.kernarg_align));
      return HSA_STATUS_SUCCESS;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
      std::memcpy(value, &info.private_segment, sizeof(info.private_segment));
      return HSA_STATUS_SUCCESS;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
      std::memcpy(value, &info.group_segment, sizeof(info.group_segment));
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_agent_iterate_memory_pools", true, &status)) {
    return status;
  }
  Agent* known = find_agent(s, agent);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (callback == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const std::vector<uint32_t> pools = known->pools;
  for (uint32_t pool : pools) {
    hsa_amd_memory_pool_t handle = {kPoolBase + pool};
    hsa_status_t answer = HSA_STATUS_SUCCESS;
    lock.unlock();
    try {
      answer = callback(handle, data);
    } catch (...) {
      lock.lock();
      return HSA_STATUS_ERROR;
    }
    lock.lock();
    if (answer != HSA_STATUS_SUCCESS) {
      return answer;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute, void* value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_memory_pool_get_info", true, &status)) {
    return status;
  }
  Pool* known = find_pool(s, memory_pool);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint32_t pool = static_cast<uint32_t>(memory_pool.handle - kPoolBase);
  const auto one = s.pool_info_failures.find(std::make_pair(pool, uint32_t{attribute}));
  if (one != s.pool_info_failures.end()) {
    return static_cast<hsa_status_t>(one->second);
  }
  const auto every = s.pool_info_failures.find(std::make_pair(UINT32_MAX, uint32_t{attribute}));
  if (every != s.pool_info_failures.end()) {
    return static_cast<hsa_status_t>(every->second);
  }

  switch (static_cast<int>(attribute)) {
    case HSA_AMD_MEMORY_POOL_INFO_SEGMENT: {
      const hsa_amd_segment_t segment = static_cast<hsa_amd_segment_t>(known->segment);
      std::memcpy(value, &segment, sizeof(segment));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: {
      const uint32_t flags = known->global_flags;
      std::memcpy(value, &flags, sizeof(flags));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_SIZE: {
      const size_t size = static_cast<size_t>(known->size);
      std::memcpy(value, &size, sizeof(size));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED: {
      const bool allowed = known->alloc_allowed;
      std::memcpy(value, &allowed, sizeof(allowed));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE: {
      const size_t granule = static_cast<size_t>(known->granule);
      std::memcpy(value, &granule, sizeof(granule));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT: {
      const size_t alignment = static_cast<size_t>(known->alignment);
      std::memcpy(value, &alignment, sizeof(alignment));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL: {
      const bool accessible = known->accessible_by_all;
      std::memcpy(value, &accessible, sizeof(accessible));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE: {
      const size_t max_size = static_cast<size_t>(known->max_size);
      std::memcpy(value, &max_size, sizeof(max_size));
      return HSA_STATUS_SUCCESS;
    }
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                hsa_amd_memory_pool_t memory_pool,
                                                hsa_amd_agent_memory_pool_info_t attribute,
                                                void* value) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_agent_memory_pool_get_info", true, &status)) {
    return status;
  }
  const uint32_t index = agent_index(s, agent);
  if (find_agent(s, agent) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (find_pool(s, memory_pool) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (value == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (attribute != HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint32_t pool = static_cast<uint32_t>(memory_pool.handle - kPoolBase);
  uint32_t access = HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
  const auto known = s.access.find(std::make_pair(index, pool));
  if (known != s.access.end()) {
    access = known->second;
  }
  s.access_reads++;
  const hsa_amd_memory_pool_access_t answer = static_cast<hsa_amd_memory_pool_access_t>(access);
  std::memcpy(value, &answer, sizeof(answer));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                          uint32_t flags, void** ptr) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_memory_pool_allocate", true, &status)) {
    return status;
  }
  Pool* known = find_pool(s, memory_pool);
  if (known == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (ptr == nullptr || size == 0 || flags != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!known->alloc_allowed || size > known->max_size) {
    // ROCr refuses a request above the pool's maximum with this status.
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }

  const uint32_t pool = static_cast<uint32_t>(memory_pool.handle - kPoolBase);
  uint32_t offset = 0;
  if ((known->global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0) {
    offset = s.kernarg_offset;
    s.kernarg_offset = 0;
  }
  const size_t map_size =
      static_cast<size_t>(round_up(offset + size + kRedzoneBytes, kPage)) + kPage;
  unsigned char* map = static_cast<unsigned char*>(
      ::mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (map == MAP_FAILED) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  // Nothing may be read or written past the redzone.
  ::mprotect(map + map_size - kPage, kPage, PROT_NONE);

  unsigned char* data = map + offset;
  if (s.alias_pages && size >= 2 * kPage && offset == 0) {
    const int memory = ::memfd_create("rdc_fake_hsa", MFD_CLOEXEC);
    if (memory >= 0 && ::ftruncate(memory, static_cast<off_t>(kPage)) == 0) {
      // The second page is the first page: a write to one is read from the other.
      ::mmap(data, kPage, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memory, 0);
      ::mmap(data + kPage, kPage, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memory, 0);
    }
    if (memory >= 0) {
      ::close(memory);
    }
  }
  fill_fresh(data, size);
  std::memset(data + size, kRedzoneByte, kRedzoneBytes);

  Block block;
  block.map = map;
  block.map_size = map_size;
  block.data = data;
  block.size = size;
  block.pool = pool;
  s.blocks.push_back(block);
  *ptr = data;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_free(void* ptr) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_memory_pool_free", true, &status)) {
    return status;
  }
  return free_pointer(s, ptr);
}

hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
                                         const uint32_t*, const void* ptr) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_agents_allow_access", true, &status)) {
    return status;
  }
  if (agents == nullptr || num_agents == 0 || ptr == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  for (uint32_t i = 0; i < num_agents; i++) {
    if (find_agent(s, agents[i]) == nullptr) {
      return HSA_STATUS_ERROR_INVALID_AGENT;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
                                       hsa_agent_t src_agent, size_t size, uint32_t,
                                       const hsa_signal_t*, hsa_signal_t completion_signal) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_memory_async_copy", true, &status)) {
    return status;
  }
  if (find_agent(s, dst_agent) == nullptr || find_agent(s, src_agent) == nullptr) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (dst == nullptr || src == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  s.copies++;
  Signal* completion = find_signal(s, completion_signal);
  if (s.copy_mode == RDC_FAKE_COPY_NEVER) {
    return HSA_STATUS_SUCCESS;
  }
  if (s.copy_mode == RDC_FAKE_COPY_FAILED) {
    if (completion != nullptr) {
      completion->value = -1;
      s.cv.notify_all();
    }
    return HSA_STATUS_SUCCESS;
  }
  std::memcpy(dst, src, size);
  if (completion != nullptr) {
    if (s.completion_wait != 0) {
      // The copy is done, but its completion waits for the nth wait on its signal, as a
      // dispatch's does.
      completion->is_completion = true;
      return HSA_STATUS_SUCCESS;
    }
    completion->value = 0;
    s.cv.notify_all();
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info, void* (*)(size_t),
                                  uint32_t* num_agents_accessible, hsa_agent_t** accessible) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (enter_call(lock, "hsa_amd_pointer_info", true, &status)) {
    return status;
  }
  if (ptr == nullptr || info == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  Block* block = find_block(s, ptr);
  std::memset(info, 0, sizeof(*info));
  info->size = sizeof(*info);
  info->type = (block == nullptr) ? HSA_EXT_POINTER_TYPE_UNKNOWN : HSA_EXT_POINTER_TYPE_HSA;
  info->hostBaseAddress = (block == nullptr) ? nullptr : block->data;
  info->agentBaseAddress = info->hostBaseAddress;
  info->sizeInBytes = (block == nullptr) ? 0 : block->size;
  if (num_agents_accessible != nullptr) {
    *num_agents_accessible = 0;
  }
  if (accessible != nullptr) {
    *accessible = nullptr;
  }
  return HSA_STATUS_SUCCESS;
}

// --- what the module needs besides ROCr ------------------------------------------

namespace amd {
namespace rdc {

// The mapping F22 asks AMD SMI for; SmiUtils.h declares it in F2's tree, and this
// definition keeps the linkage that header gives it without naming it, so that
// this file also compiles in F9's tree, where the declaration is absent.
amdsmi_status_t get_kfd_node_id(uint32_t entity_id, uint32_t* node_id) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.node_id_calls++;
  auto one = s.node_failures.find(entity_id);
  if (one == s.node_failures.end()) {
    one = s.node_failures.find(UINT32_MAX);
  }
  if (one != s.node_failures.end()) {
    return static_cast<amdsmi_status_t>(one->second);
  }
  if (node_id == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  const auto known = s.node_ids.find(entity_id);
  *node_id = (known == s.node_ids.end()) ? entity_id + 1 : known->second;
  return AMDSMI_STATUS_SUCCESS;
}

// Only F9's sources reach these, through a member nothing reads; librdc's
// constructor would measure the clock for about a quarter of a second each time.
RdcPerfTimer::RdcPerfTimer(void) {}
RdcPerfTimer::~RdcPerfTimer() {}

}  // namespace rdc
}  // namespace amd

// --- the controls -----------------------------------------------------------------

extern "C" {

void rdc_fake_hsa_reset(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  reset_locked(s);
}

uint32_t rdc_fake_hsa_add_agent(int is_gpu, uint32_t node_id, const char* name) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return add_agent_locked(s, is_gpu != 0, node_id, name);
}

void rdc_fake_hsa_clear_agents(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.agents.clear();
  s.pools.clear();
  s.access.clear();
}

uint32_t rdc_fake_hsa_agent_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.agents.size());
}

uint64_t rdc_fake_hsa_agent_handle(uint32_t agent) { return kAgentBase + agent; }

uint64_t rdc_fake_hsa_pool_handle(uint32_t pool) { return kPoolBase + pool; }

void rdc_fake_hsa_set_agent_name(uint32_t agent, const char* name, int terminate) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent < s.agents.size()) {
    set_name(&s.agents[agent], name, terminate != 0);
  }
}

void rdc_fake_hsa_set_agent_queue_max(uint32_t agent, uint32_t packets) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent < s.agents.size()) {
    s.agents[agent].queue_max = packets;
  }
}

void rdc_fake_hsa_set_agent_profile(uint32_t agent, uint32_t profile) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent < s.agents.size()) {
    s.agents[agent].profile = profile;
  }
}

void rdc_fake_hsa_fail_agent_info(uint32_t agent, uint32_t attribute, int32_t status) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.agent_info_failures[std::make_pair(agent, attribute)] = status;
}

uint32_t rdc_fake_hsa_add_pool(uint32_t agent, uint32_t segment, uint32_t global_flags,
                               uint64_t size, uint64_t max_size, int alloc_allowed,
                               uint64_t granule, uint64_t alignment, int accessible_by_all) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent >= s.agents.size()) {
    return UINT32_MAX;
  }
  add_pool_locked(s, agent, segment, global_flags, size, max_size, alloc_allowed != 0, granule,
                  alignment, accessible_by_all != 0);
  return static_cast<uint32_t>(s.pools.size() - 1);
}

void rdc_fake_hsa_clear_pools(uint32_t agent) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent < s.agents.size()) {
    s.agents[agent].pools.clear();
  }
}

uint32_t rdc_fake_hsa_pool_count(uint32_t agent) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent >= s.agents.size()) {
    return 0;
  }
  return static_cast<uint32_t>(s.agents[agent].pools.size());
}

uint32_t rdc_fake_hsa_pool_of_agent(uint32_t agent, uint32_t n) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (agent >= s.agents.size() || n >= s.agents[agent].pools.size()) {
    return UINT32_MAX;
  }
  return s.agents[agent].pools[n];
}

void rdc_fake_hsa_set_pool_size(uint32_t pool, uint64_t size) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (pool < s.pools.size()) {
    s.pools[pool].size = size;
  }
}

void rdc_fake_hsa_set_pool_max(uint32_t pool, uint64_t max_size) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (pool < s.pools.size()) {
    s.pools[pool].max_size = max_size;
  }
}

void rdc_fake_hsa_set_pool_flags(uint32_t pool, uint32_t global_flags) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (pool < s.pools.size()) {
    s.pools[pool].global_flags = global_flags;
  }
}

void rdc_fake_hsa_set_pool_alloc_allowed(uint32_t pool, int allowed) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (pool < s.pools.size()) {
    s.pools[pool].alloc_allowed = allowed != 0;
  }
}

void rdc_fake_hsa_set_pool_granule(uint32_t pool, uint64_t granule, uint64_t alignment) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (pool < s.pools.size()) {
    s.pools[pool].granule = granule;
    s.pools[pool].alignment = alignment;
  }
}

void rdc_fake_hsa_set_pool_access(uint32_t agent, uint32_t pool, uint32_t access) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.access[std::make_pair(agent, pool)] = access;
}

void rdc_fake_hsa_fail_pool_info(uint32_t pool, uint32_t attribute, int32_t status) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.pool_info_failures[std::make_pair(pool, attribute)] = status;
}

uint32_t rdc_fake_hsa_access_reads(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.access_reads;
}

uint32_t rdc_fake_hsa_foreign_pools(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.foreign_pools;
}

uint32_t rdc_fake_hsa_alloc_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.blocks.size());
}

const void* rdc_fake_hsa_alloc_address(uint32_t block) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (block < s.blocks.size()) ? s.blocks[block].data : nullptr;
}

uint64_t rdc_fake_hsa_alloc_size(uint32_t block) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (block < s.blocks.size()) ? s.blocks[block].size : 0;
}

uint32_t rdc_fake_hsa_alloc_pool(uint32_t block) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (block < s.blocks.size()) ? s.blocks[block].pool : UINT32_MAX;
}

uint32_t rdc_fake_hsa_alloc_frees(uint32_t block) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (block < s.blocks.size()) ? s.blocks[block].frees : 0;
}

uint32_t rdc_fake_hsa_live_blocks(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  uint32_t live = 0;
  for (const Block& block : s.blocks) {
    if (block.frees == 0) {
      live++;
    }
  }
  return live;
}

uint32_t rdc_fake_hsa_double_frees(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.double_frees;
}

uint32_t rdc_fake_hsa_unissued_frees(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.unissued_frees;
}

uint32_t rdc_fake_hsa_redzone_failures(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.redzone_failures;
}

void rdc_fake_hsa_set_kernarg_offset(uint32_t bytes) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.kernarg_offset = bytes;
}

void rdc_fake_hsa_set_alias_pages(int on) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.alias_pages = on != 0;
}

uint32_t rdc_fake_hsa_queue_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.queues.size());
}

uint32_t rdc_fake_hsa_queue_agent(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size()) ? s.queues[queue].agent : UINT32_MAX;
}

uint32_t rdc_fake_hsa_queue_size(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size()) ? s.queues[queue].size : 0;
}

uint32_t rdc_fake_hsa_queue_private_segment(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size()) ? s.queues[queue].private_segment : 0;
}

uint32_t rdc_fake_hsa_queue_group_segment(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size()) ? s.queues[queue].group_segment : 0;
}

int rdc_fake_hsa_queue_has_callback(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size() && s.queues[queue].callback != nullptr) ? 1 : 0;
}

int rdc_fake_hsa_queue_destroyed(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size() && s.queues[queue].destroyed) ? 1 : 0;
}

const void* rdc_fake_hsa_queue_error_data(uint32_t queue) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (queue < s.queues.size()) ? s.queues[queue].data : nullptr;
}

void rdc_fake_hsa_raise_queue_error(uint32_t queue, int32_t status) {
  raise_queue_error(queue, status);
}

uint32_t rdc_fake_hsa_queue_errors_recorded(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.queue_errors_recorded;
}

uint32_t rdc_fake_hsa_signal_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.signals.size());
}

int64_t rdc_fake_hsa_signal_value(uint32_t signal) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (signal < s.signals.size()) ? s.signals[signal].value : 0;
}

uint32_t rdc_fake_hsa_signal_waits(uint32_t signal) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (signal < s.signals.size()) ? s.signals[signal].waits : 0;
}

void rdc_fake_hsa_set_dispatch_signal_value(int64_t value) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.dispatch_signal_value = value;
}

void rdc_fake_hsa_set_completion_wait(uint32_t nth) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.completion_wait = nth;
}

uint32_t rdc_fake_hsa_wait_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.waits;
}

uint32_t rdc_fake_hsa_wait_hint_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.wait_hints.size());
}

uint64_t rdc_fake_hsa_wait_hint(uint32_t n) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (n < s.wait_hints.size()) ? s.wait_hints[n] : 0;
}

uint32_t rdc_fake_hsa_wait_state_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.wait_states.size());
}

int rdc_fake_hsa_wait_state(uint32_t n) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (n < s.wait_states.size()) ? s.wait_states[n] : 0;
}

void rdc_fake_hsa_set_kernel_info(const char* symbol, uint32_t kernarg_size, uint32_t kernarg_align,
                                  uint32_t private_segment, uint32_t group_segment) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  KernelInfo info;
  info.kernarg_size = kernarg_size;
  info.kernarg_align = kernarg_align;
  info.private_segment = private_segment;
  info.group_segment = group_segment;
  if (symbol == nullptr || *symbol == '\0') {
    s.default_kernel_info = info;
    return;
  }
  s.kernel_info[std::string(symbol)] = info;
}

void rdc_fake_hsa_set_kernel_mode(int mode) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.kernel_mode = mode;
}

void rdc_fake_hsa_set_search_output(uint32_t element0, uint32_t element3) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.search_element0 = element0;
  s.search_element3 = element3;
}

uint32_t rdc_fake_hsa_dispatch_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.dispatches.size());
}

uint32_t rdc_fake_hsa_dispatch_queue(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].queue : UINT32_MAX;
}

uint32_t rdc_fake_hsa_dispatch_signal(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].signal : UINT32_MAX;
}

uint64_t rdc_fake_hsa_dispatch_grid(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].grid : 0;
}

uint32_t rdc_fake_hsa_dispatch_workgroup(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].workgroup : 0;
}

uint32_t rdc_fake_hsa_dispatch_private_segment(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].private_segment : 0;
}

uint32_t rdc_fake_hsa_dispatch_group_segment(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].group_segment : 0;
}

uint64_t rdc_fake_hsa_dispatch_kernarg_address(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].kernarg_address : 0;
}

uint32_t rdc_fake_hsa_dispatch_reserved0(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].reserved0 : 0;
}

uint64_t rdc_fake_hsa_dispatch_reserved2(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].reserved2 : 0;
}

uint32_t rdc_fake_hsa_dispatch_header(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].header : 0;
}

uint32_t rdc_fake_hsa_dispatch_setup(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (dispatch < s.dispatches.size()) ? s.dispatches[dispatch].setup : 0;
}

uint32_t rdc_fake_hsa_dispatch_kernarg_size(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (dispatch >= s.dispatches.size()) {
    return 0;
  }
  return static_cast<uint32_t>(s.dispatches[dispatch].kernarg.size());
}

const unsigned char* rdc_fake_hsa_dispatch_kernarg(uint32_t dispatch) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (dispatch >= s.dispatches.size() || s.dispatches[dispatch].kernarg.empty()) {
    return nullptr;
  }
  return s.dispatches[dispatch].kernarg.data();
}

uint32_t rdc_fake_hsa_grid_violations(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.grid_violations;
}

const char* rdc_fake_hsa_last_grid_violation(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.last_grid_violation.c_str();
}

void rdc_fake_hsa_set_copy_mode(int mode) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.copy_mode = mode;
}

uint32_t rdc_fake_hsa_copy_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.copies;
}

uint32_t rdc_fake_hsa_executable_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.executables.size());
}

uint32_t rdc_fake_hsa_reader_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.readers.size());
}

uint64_t rdc_fake_hsa_reader_device(uint32_t reader) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (reader < s.readers.size()) ? s.readers[reader].device : 0;
}

uint64_t rdc_fake_hsa_reader_inode(uint32_t reader) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (reader < s.readers.size()) ? s.readers[reader].inode : 0;
}

uint32_t rdc_fake_hsa_destroy_log_size(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.destroy_log.size());
}

const char* rdc_fake_hsa_destroy_log_entry(uint32_t entry) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (entry < s.destroy_log.size()) ? s.destroy_log[entry].c_str() : "";
}

uint32_t rdc_fake_hsa_live_references(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.references;
}

uint32_t rdc_fake_hsa_load_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.loads;
}

uint32_t rdc_fake_hsa_failed_load_count(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.failed_loads;
}

uint32_t rdc_fake_hsa_unbalanced_shutdowns(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.unbalanced_shutdowns;
}

uint32_t rdc_fake_hsa_refused_calls(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.refused_calls;
}

uint32_t rdc_fake_hsa_foreign_handles(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.foreign_handles;
}

int rdc_fake_hsa_latched(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.latched ? 1 : 0;
}

void rdc_fake_hsa_set_timestamp_frequency(uint64_t hz) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.timestamp_frequency = hz;
}

void rdc_fake_hsa_fail(const char* name, uint32_t nth, int32_t status) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  Failure failure;
  failure.nth = nth;
  failure.status = status;
  s.failures[std::string(name)] = failure;
}

void rdc_fake_hsa_throw_int(const char* name, uint32_t nth, int value) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  Failure failure;
  failure.nth = nth;
  failure.throws = true;
  failure.value = value;
  s.failures[std::string(name)] = failure;
}

void rdc_fake_hsa_clear_failures(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.failures.clear();
}

uint32_t rdc_fake_hsa_call_count(const char* name) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  const auto known = s.call_counts.find(std::string(name));
  return (known == s.call_counts.end()) ? 0 : known->second;
}

uint32_t rdc_fake_hsa_call_log_size(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return static_cast<uint32_t>(s.call_log.size());
}

const char* rdc_fake_hsa_call_log_entry(uint32_t entry) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return (entry < s.call_log.size()) ? s.call_log[entry].c_str() : "";
}

void rdc_fake_hsa_set_caller(uint64_t token) { t_caller = token; }

void rdc_fake_hsa_gate(const char* name, uint64_t caller, uint32_t nth) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  Gate gate;
  gate.caller = caller;
  gate.nth = nth;
  s.gates[std::string(name)] = gate;
}

uint32_t rdc_fake_hsa_gate_held(const char* name) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  const auto gate = s.gates.find(std::string(name));
  return (gate == s.gates.end()) ? 0 : gate->second.held;
}

int rdc_fake_hsa_gate_wait(const char* name, uint32_t callers, uint32_t timeout_ms) {
  State& s = started();
  std::unique_lock<std::mutex> lock(s.mutex);
  const std::string key(name);
  // The wait ends at a deadline of its own, in slices, as wait_a_slice says.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    const auto gate = s.gates.find(key);
    if (gate != s.gates.end() && gate->second.held >= callers) {
      return 1;
    }
    if (!wait_a_slice(s.cv, lock, deadline)) {
      const auto last = s.gates.find(key);
      return (last != s.gates.end() && last->second.held >= callers) ? 1 : 0;
    }
  }
}

void rdc_fake_hsa_gate_release(const char* name) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  const std::string key(name);
  if (s.gates.count(key) != 0) {
    s.gates[key].released = true;
    s.cv.notify_all();
  }
}

void rdc_fake_hsa_set_node_id(uint32_t entity, uint32_t node) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.node_ids[entity] = node;
}

void rdc_fake_hsa_fail_node_id(uint32_t entity, int32_t status) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  s.node_failures[entity] = status;
}

uint32_t rdc_fake_hsa_node_id_calls(void) {
  State& s = started();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.node_id_calls;
}
}
