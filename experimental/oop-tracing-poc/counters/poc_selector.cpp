// Local POC selector against runtime-contract section 11 (proposals/runtime-contract.md @ 98deff2).
// Loaded by rocprofiler-register (ROCPROFILER_REGISTER_LIBRARY=<this .so>), which passes the
// runtime's {hooks, services} tables to rocprofiler_set_api_table("hip_runtime_contract", ...).
//
// Env: POC_COUNTERS=SQ_WAVES,GRBM_COUNT  POC_POOL=4096  POC_SERIALIZE=1  POC_MODE=user_only
//      POC_SELECTOR_OUT=<csv>. POC shortcut: results go to a CSV, not to a collector socket.
#include "rocm_runtime_contract.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <aqlprofile-sdk/aql_profile_v2.h>

#include "kfd_ioctl.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
struct counter_def {
  const char* name;
  hsa_ven_amd_aqlprofile_block_name_t block;
  uint32_t event;
};
const counter_def kDefs[] = {  // gfx1250, from rocprofiler-sdk config.yaml
    {"SQ_WAVES", HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 4},
    {"SQ_INSTS_VALU", HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 128},
    {"SQ_INSTS_SALU", HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 67},
    {"GRBM_COUNT", HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0},
    {"GRBM_GUI_ACTIVE", HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 2},
};
constexpr uint16_t kVendor = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
constexpr uint16_t kBarrier = 1 << HSA_PACKET_HEADER_BARRIER;

enum slot_state : uint8_t { FREE = 0, IN_FLIGHT = 1 };
struct slot {
  aqlprofile_handle_t handle{};
  hsa_ext_amd_aql_pm4_packet_t pre[1]{};
  hsa_ext_amd_aql_pm4_packet_t post[2]{};
  uint64_t packet_index = 0;  // join_key target
  std::string kernel;
  std::atomic<uint8_t> state{FREE};
};

// Everything lives on the heap and is never freed: the runtime may call dispatch_done_fn during
// its own teardown, after this library's static destructors would have run.
struct state_t {
  rocm_runtime_hooks_t* hooks = nullptr;
  const rocm_runtime_services_t* services = nullptr;
  std::vector<aqlprofile_pmc_event_t> events;
  std::vector<std::string> names;
  bool serialize = false;
  bool user_only = false;
  size_t pool = 4096;

  uint32_t gpu_id = 0;  // POC: one GPU (the first one that dispatches)
  std::atomic<int> ready{0};  // 0 none, 1 requested, 2 ready, -1 failed
  slot* slots = nullptr;
  std::mutex free_mu;
  std::vector<uint32_t> free_list;

  std::mutex q_mu;
  std::condition_variable q_cv;
  std::vector<std::pair<uint64_t, uint32_t>> done_q;  // (cookie, status)
  bool stop = false;
  std::thread worker;

  FILE* out = nullptr;
  std::atomic<uint64_t> emitted{0}, no_slot{0}, not_ready{0}, completed{0}, dropped{0},
      unexpected{0};
};
state_t* S = nullptr;

uint32_t g_alloc_gpu = 0;
hsa_status_t mem_alloc(void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t f, void*) {
  if (size == 0) {
    *ptr = nullptr;
    return HSA_STATUS_SUCCESS;
  }
  uint32_t flags = f.host_access ? ROCM_MEM_HOST_VISIBLE : ROCM_MEM_DEVICE_LOCAL;
  if (f.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED) flags |= ROCM_MEM_UNCACHED;
  *ptr = S->services->gpu_alloc_fn(g_alloc_gpu, size, flags);  // RC-SVC-1
  if (*ptr == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  if (f.host_access) std::memset(*ptr, 0, size);
  return HSA_STATUS_SUCCESS;
}
void mem_free(void* ptr, void*) { S->services->gpu_free_fn(ptr); }
hsa_status_t mem_copy(void* dst, const void* src, size_t size, void*) {
  return hsa_memory_copy(dst, src, size);
}

hsa_status_t find_agent(hsa_agent_t a, void* data) {
  auto* p = static_cast<std::pair<uint32_t, hsa_agent_t>*>(data);
  uint32_t id = 0;
  hsa_agent_get_info(a, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_UID), &id);
  if (id == p->first) {
    p->second = a;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

void pmc_lock(uint32_t gpu_id) {  // counter owner (slow-path HLD 5.8)
  const int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (fd < 0) return;
  kfd_ioctl_profiler_args a{};
  a.op = KFD_IOC_PROFILER_PMC;
  a.pmc.gpu_id = gpu_id;
  a.pmc.lock = 1;
  a.pmc.perfcount_enable = 1;
  if (ioctl(fd, AMDKFD_IOC_PROFILER, &a) != 0)
    std::fprintf(stderr, "[selector] PMC lock gpu %u: %s\n", gpu_id, std::strerror(errno));
}

bool build_pool(uint32_t gpu_id) {  // worker thread, off the dispatch path
  std::pair<uint32_t, hsa_agent_t> f{gpu_id, hsa_agent_t{0}};
  hsa_iterate_agents(find_agent, &f);
  if (f.second.handle == 0) return false;
  hsa_agent_t agent = f.second;
  char name[64] = {};
  hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  aqlprofile_agent_info_t info{};
  info.agent_gfxip = name;
  hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC), &info.xcc_num);
  hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES), &info.se_num);
  hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &info.cu_num);
  hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE), &info.shader_arrays_per_se);
  aqlprofile_agent_handle_t ah{};
  if (aqlprofile_register_agent(&ah, &info) != HSA_STATUS_SUCCESS) return false;
  aqlprofile_pmc_profile_t profile{};
  profile.agent = ah;
  profile.events = S->events.data();
  profile.event_count = static_cast<uint32_t>(S->events.size());
  g_alloc_gpu = gpu_id;
  S->slots = new slot[S->pool];
  for (size_t i = 0; i < S->pool; ++i) {
    aqlprofile_pmc_aql_packets_t pk{};
    if (aqlprofile_pmc_create_packets(&S->slots[i].handle, &pk, profile, mem_alloc, mem_free,
                                      mem_copy, nullptr) != HSA_STATUS_SUCCESS) {
      std::fprintf(stderr, "[selector] create_packets failed at slot %zu\n", i);
      return false;
    }
    S->slots[i].pre[0] = pk.start_packet;
    S->slots[i].pre[0].header = kVendor;
    S->slots[i].post[0] = pk.stop_packet;
    S->slots[i].post[0].header = kVendor | kBarrier;
    S->slots[i].post[1] = pk.read_packet;
    S->slots[i].post[1].header = kVendor | kBarrier;
    for (auto* p : {&S->slots[i].pre[0], &S->slots[i].post[0], &S->slots[i].post[1]})
      p->completion_signal = hsa_signal_t{0};  // the runtime owns the last-slot signal
  }
  {
    std::lock_guard<std::mutex> lk(S->free_mu);
    for (size_t i = S->pool; i-- > 0;) S->free_list.push_back(static_cast<uint32_t>(i));
  }
  pmc_lock(gpu_id);
  std::fprintf(stderr, "[selector] %s gpu %u: %zu counters, pool %zu, serialize %d\n", name,
               gpu_id, S->events.size(), S->pool, S->serialize ? 1 : 0);
  return true;
}

hsa_status_t collect(aqlprofile_pmc_event_t event, uint64_t, uint64_t value, void* data) {
  auto* sums = static_cast<std::vector<uint64_t>*>(data);
  for (size_t i = 0; i < S->events.size(); ++i)
    if (S->events[i].block_name == event.block_name && S->events[i].event_id == event.event_id)
      (*sums)[i] += value;
  return HSA_STATUS_SUCCESS;
}

void worker() {
  for (;;) {
    std::vector<std::pair<uint64_t, uint32_t>> batch;
    {
      std::unique_lock<std::mutex> lk(S->q_mu);
      S->q_cv.wait(lk, [] { return S->stop || !S->done_q.empty() || S->ready.load() == 1; });
      if (S->ready.load() == 1) {
        lk.unlock();
        S->ready.store(build_pool(S->gpu_id) ? 2 : -1);
        continue;
      }
      batch.swap(S->done_q);
      if (batch.empty() && S->stop) return;
    }
    for (auto [cookie, status] : batch) {
      const uint64_t idx = (cookie & 0xffffffffu) - 1;
      if (idx >= S->pool || S->slots[idx].state.load() != IN_FLIGHT) {
        ++S->unexpected;  // second done call for a cookie, or unknown cookie
        continue;
      }
      slot& s = S->slots[idx];
      if (status == ROCM_DISPATCH_COMPLETED) {
        ++S->completed;
        std::vector<uint64_t> sums(S->events.size(), 0);
        aqlprofile_pmc_iterate_data(s.handle, collect, &sums);
        if (S->out != nullptr) {
          std::fprintf(S->out, "%llu,%s", (unsigned long long)s.packet_index, s.kernel.c_str());
          for (auto v : sums) std::fprintf(S->out, ",%llu", (unsigned long long)v);
          std::fprintf(S->out, "\n");
        }
      } else {
        ++S->dropped;
      }
      s.state.store(FREE);
      std::lock_guard<std::mutex> lk(S->free_mu);
      S->free_list.push_back(static_cast<uint32_t>(idx));
    }
  }
}

// ------------------------------------------------------------------ hooks
void emit_dispatch_packets(const rocm_dispatch_info_t* info, rocm_dispatch_packets_t* out) {
  // never allocates, never waits for the worker or the collector (RC-PKT-1)
  const int r = S->ready.load(std::memory_order_acquire);
  if (r != 2) {
    int expected = 0;
    if (r == 0 && S->ready.compare_exchange_strong(expected, 1)) {
      S->gpu_id = info->gpu_id;
      S->q_cv.notify_one();
    }
    ++S->not_ready;
    return;
  }
  if (info->gpu_id != S->gpu_id) return;
  if (S->user_only && (info->flags & ROCM_DISPATCH_INTERNAL)) return;
  uint32_t idx;
  {
    std::lock_guard<std::mutex> lk(S->free_mu);
    if (S->free_list.empty()) {
      ++S->no_slot;
      return;
    }
    idx = S->free_list.back();
    S->free_list.pop_back();
  }
  slot& s = S->slots[idx];
  s.state.store(IN_FLIGHT);
  s.kernel = info->kernel_name != nullptr ? info->kernel_name : "";
  ++S->emitted;
  out->pre = s.pre;
  out->pre_count = 1;
  out->post = s.post;
  out->post_count = 2;
  out->flags = S->serialize ? ROCM_PACKETS_SERIALIZE : 0;
  out->join_key = &s.packet_index;
  out->cookie = idx + 1;
}

void dispatch_done(const uint64_t* cookies, uint32_t count, uint32_t status) {
  // must not block or call the runtime (RC-PKT-18): queue for the worker
  {
    std::lock_guard<std::mutex> lk(S->q_mu);
    for (uint32_t i = 0; i < count; ++i) S->done_q.emplace_back(cookies[i], status);
  }
  S->q_cv.notify_one();
}

void finalize() {
  {
    std::lock_guard<std::mutex> lk(S->q_mu);
    S->stop = true;
  }
  S->q_cv.notify_one();
  if (S->worker.joinable()) S->worker.join();
  uint64_t in_flight = 0;
  if (S->slots != nullptr)
    for (size_t i = 0; i < S->pool; ++i) in_flight += S->slots[i].state.load() == IN_FLIGHT;
  if (S->out != nullptr) std::fclose(S->out);
  S->out = nullptr;
  std::fprintf(stderr,
               "[selector] cookies issued %llu: completed %llu, dropped %llu, still in flight at "
               "exit %llu, unexpected done calls %llu; skipped: pool empty %llu, not ready %llu\n",
               (unsigned long long)S->emitted.load(), (unsigned long long)S->completed.load(),
               (unsigned long long)S->dropped.load(), (unsigned long long)in_flight,
               (unsigned long long)S->unexpected.load(), (unsigned long long)S->no_slot.load(),
               (unsigned long long)S->not_ready.load());
}
}  // namespace

// rocprofiler-register's entry point (RC-PKT-0): called once per registered library.
extern "C" __attribute__((visibility("default"))) int rocprofiler_set_api_table(
    const char* name, uint64_t, uint64_t, void** tables, uint64_t num_tables) {
  if (name == nullptr || std::strcmp(name, "hip_runtime_contract") != 0 || num_tables < 2) return 0;
  auto* hooks = static_cast<rocm_runtime_hooks_t*>(tables[0]);
  auto* services = static_cast<const rocm_runtime_services_t*>(tables[1]);
  constexpr size_t kHooksNeed = offsetof(rocm_runtime_hooks_t, dispatch_done_fn) + sizeof(void*);
  constexpr size_t kSvcNeed = offsetof(rocm_runtime_services_t, gpu_free_fn) + sizeof(void*);
  if (hooks->version != ROCM_RUNTIME_CONTRACT_VERSION || hooks->size < kHooksNeed ||
      services->size < kSvcNeed || services->gpu_alloc_fn == nullptr)
    return 0;

  S = new state_t{};
  S->hooks = hooks;
  S->services = services;
  const char* list_env = std::getenv("POC_COUNTERS");
  std::string list = list_env ? list_env : "SQ_WAVES,GRBM_COUNT";
  size_t pos = 0;
  for (;;) {
    const size_t c = list.find(',', pos);
    const std::string n = list.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    for (const auto& d : kDefs)
      if (n == d.name) {
        aqlprofile_pmc_event_t e{};
        e.event_id = d.event;
        e.block_name = d.block;
        S->events.push_back(e);
        S->names.push_back(n);
      }
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  S->serialize = std::getenv("POC_SERIALIZE") != nullptr;
  const char* mode = std::getenv("POC_MODE");
  S->user_only = mode != nullptr && std::strcmp(mode, "user_only") == 0;
  if (const char* p = std::getenv("POC_POOL")) S->pool = std::strtoul(p, nullptr, 10);
  const char* out = std::getenv("POC_SELECTOR_OUT");
  S->out = std::fopen(out ? out : "poc_counters.csv", "w");
  if (S->out != nullptr) {
    std::fprintf(S->out, "packet_index,kernel");
    for (const auto& n : S->names) std::fprintf(S->out, ",%s", n.c_str());
    std::fprintf(S->out, "\n");
  }
  S->worker = std::thread(worker);
  std::atexit(finalize);

  // RC-PKT-0: done entry first, emit entry last, both with release stores; never cleared.
  __atomic_store_n(&hooks->dispatch_done_fn, &dispatch_done, __ATOMIC_RELEASE);
  __atomic_store_n(&hooks->emit_dispatch_packets_fn, &emit_dispatch_packets, __ATOMIC_RELEASE);
  return 0;
}
