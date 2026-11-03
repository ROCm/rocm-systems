// Local POC: CLR side of runtime-contract section 11. See hip_rtcontract.hpp.
#include "hip_rtcontract.hpp"

#include <hsa/hsa_ext_amd.h>
#include <rocprofiler-register/rocprofiler-register.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

ROCPROFILER_REGISTER_DEFINE_IMPORT(hip_runtime_contract, ROCM_RUNTIME_CONTRACT_VERSION)

namespace {
rocm_runtime_hooks_t g_hooks{};        // owned by the runtime, filled by the selector
rocm_runtime_services_t g_services{};  // owned and filled by the runtime

// ------------------------------------------------------------------ services (RC-SVC)
struct agent_pools {
  hsa_agent_t gpu{};
  hsa_amd_memory_pool_t host{}, kernarg{}, device{};
};
std::mutex g_svc_mu;
std::unordered_map<uint32_t, agent_pools> g_by_gpu;
std::unordered_map<uint64_t, uint32_t> g_gpu_id_of;
hsa_agent_t g_cpu{};

hsa_status_t pool_cb(hsa_amd_memory_pool_t pool, void* data) {
  auto* p = static_cast<agent_pools*>(data);
  hsa_amd_segment_t seg;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
  bool alloc = false;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
  if (seg != HSA_AMD_SEGMENT_GLOBAL || !alloc) return HSA_STATUS_SUCCESS;
  uint32_t f = 0;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &f);
  if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
    if (p->kernarg.handle == 0) p->kernarg = pool;
  } else if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
    if (p->host.handle == 0) p->host = pool;
  } else if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    if (p->device.handle == 0) p->device = pool;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t agent_cb(hsa_agent_t a, void*) {
  hsa_device_type_t t;
  hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_CPU && g_cpu.handle == 0) g_cpu = a;
  if (t != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
  uint32_t id = 0;
  hsa_agent_get_info(a, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_UID), &id);
  agent_pools p{};
  p.gpu = a;
  hsa_amd_agent_iterate_memory_pools(a, pool_cb, &p);  // device pool
  g_by_gpu[id] = p;
  g_gpu_id_of[a.handle] = id;
  return HSA_STATUS_SUCCESS;
}

void scan_agents() {
  hsa_iterate_agents(agent_cb, nullptr);
  agent_pools cpu{};
  hsa_amd_agent_iterate_memory_pools(g_cpu, pool_cb, &cpu);
  for (auto& [id, p] : g_by_gpu) {
    p.host = cpu.host;
    p.kernarg = cpu.kernarg;
  }
}

void* svc_alloc(uint32_t gpu_id, size_t size, uint32_t flags) {
  agent_pools p;
  {
    std::lock_guard<std::mutex> lk(g_svc_mu);
    auto it = g_by_gpu.find(gpu_id);
    if (it == g_by_gpu.end()) return nullptr;
    p = it->second;
  }
  if (flags & ROCM_MEM_EXPORTABLE) return nullptr;  // not supported by this POC: return null
  // The contract has no executable flag; POC_SVC_EXECUTABLE=1 adds it, to test whether one is needed.
  static const bool exec = std::getenv("POC_SVC_EXECUTABLE") != nullptr;
  const uint32_t alloc_flags =
      exec ? HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG : HSA_AMD_MEMORY_POOL_STANDARD_FLAG;
  void* ptr = nullptr;
  if (flags & ROCM_MEM_HOST_VISIBLE) {
    const hsa_amd_memory_pool_t pool = (flags & ROCM_MEM_UNCACHED) ? p.kernarg : p.host;
    if (hsa_amd_memory_pool_allocate(pool, size, alloc_flags, &ptr) != HSA_STATUS_SUCCESS)
      return nullptr;
    if (hsa_amd_agents_allow_access(1, &p.gpu, nullptr, ptr) != HSA_STATUS_SUCCESS) {
      hsa_amd_memory_pool_free(ptr);
      return nullptr;
    }
    return ptr;
  }
  if (hsa_amd_memory_pool_allocate(p.device, size, alloc_flags, &ptr) != HSA_STATUS_SUCCESS)
    return nullptr;
  return ptr;
}

void svc_free(void* ptr) {
  if (ptr != nullptr) hsa_amd_memory_pool_free(ptr);
}

// ------------------------------------------------------------------ signals and cookies
struct entry {
  const void* queue;
  uint32_t gpu;
  hsa_signal_t sig;
  uint64_t cookie;
  bool serialize;
  hsa_signal_t dep;
};

std::mutex g_mu;
std::vector<hsa_signal_t> g_free;
std::unordered_map<const void*, std::deque<entry>> g_outstanding;  // per queue, in submit order
std::unordered_map<uint64_t, int> g_refs;        // signal handle -> references (last + deps)
std::unordered_map<uint64_t, bool> g_completed;  // completed, waiting for refs to drop
std::unordered_map<uint32_t, int> g_ser_count;   // gpu -> outstanding SERIALIZE dispatches
std::unordered_map<uint32_t, hsa_signal_t> g_last;  // gpu -> last kernel signal while serializing

// caller holds g_mu
void maybe_recycle(hsa_signal_t s) {
  if (s.handle == 0) return;
  auto c = g_completed.find(s.handle);
  if (c == g_completed.end()) return;
  if (g_refs[s.handle] > 0) return;
  g_completed.erase(c);
  g_refs.erase(s.handle);
  hsa_signal_store_relaxed(s, 1);
  g_free.push_back(s);
}

void release_ref(hsa_signal_t s) {  // caller holds g_mu
  if (s.handle == 0) return;
  if (--g_refs[s.handle] <= 0) maybe_recycle(s);
}

void call_done(const std::vector<uint64_t>& cookies, uint32_t status) {
  if (cookies.empty()) return;
  auto fn = __atomic_load_n(&g_hooks.dispatch_done_fn, __ATOMIC_ACQUIRE);
  if (fn != nullptr) fn(cookies.data(), static_cast<uint32_t>(cookies.size()), status);
}

std::once_flag g_once;
bool g_registered = false;

void register_tables() {
  g_services.version = ROCM_RUNTIME_CONTRACT_VERSION;
  g_services.size = sizeof(g_services);
  g_services.gpu_alloc_fn = svc_alloc;
  g_services.gpu_free_fn = svc_free;
  g_services.gpu_export_fn = nullptr;  // cannot export (RC-SVC-3 allows a null entry)
  g_hooks.version = ROCM_RUNTIME_CONTRACT_VERSION;
  g_hooks.size = sizeof(g_hooks);
  {
    std::lock_guard<std::mutex> lk(g_svc_mu);
    scan_agents();
  }
  {
    std::lock_guard<std::mutex> lk(g_mu);
    const char* n = std::getenv("POC_RC_SIGNALS");
    const size_t count = n ? std::strtoul(n, nullptr, 10) : 8192;
    g_free.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      hsa_signal_t s{};
      if (hsa_signal_create(1, 0, nullptr, &s) == HSA_STATUS_SUCCESS) g_free.push_back(s);
    }
  }
  void* tables[2] = {&g_hooks, &g_services};
  rocprofiler_register_library_indentifier_t id{};
  const auto st = rocprofiler_register_library_api_table(
      "hip_runtime_contract", &ROCPROFILER_REGISTER_IMPORT_FUNC(hip_runtime_contract),
      ROCM_RUNTIME_CONTRACT_VERSION, tables, 2, &id);
  g_registered = (st == ROCP_REG_SUCCESS);
  if (std::getenv("POC_RC_VERBOSE") != nullptr)
    std::fprintf(stderr, "[hip_runtime_contract] register: %d (%s), %zu signals\n", st,
                 rocprofiler_register_error_string(st), g_free.size());
}
}  // namespace

namespace rtc {
const rocm_runtime_hooks_t* Hooks() {
  std::call_once(g_once, register_tables);
  constexpr size_t kNeed = offsetof(rocm_runtime_hooks_t, dispatch_done_fn) + sizeof(void*);
  if (g_hooks.size < kNeed) return nullptr;
  if (__atomic_load_n(&g_hooks.emit_dispatch_packets_fn, __ATOMIC_ACQUIRE) == nullptr) return nullptr;
  if (__atomic_load_n(&g_hooks.dispatch_done_fn, __ATOMIC_ACQUIRE) == nullptr) return nullptr;
  return &g_hooks;
}

emit_fn_t EmitFn(const rocm_runtime_hooks_t* hooks) {
  return __atomic_load_n(&hooks->emit_dispatch_packets_fn, __ATOMIC_ACQUIRE);
}

uint32_t GpuId(hsa_agent_t agent) {
  std::lock_guard<std::mutex> lk(g_svc_mu);
  auto it = g_gpu_id_of.find(agent.handle);
  return it == g_gpu_id_of.end() ? 0 : it->second;
}

bool AcquireSignal(hsa_signal_t* out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_free.empty()) return false;
  *out = g_free.back();
  g_free.pop_back();
  return true;
}

void Track(const void* queue, uint32_t gpu_id, hsa_signal_t sig, uint64_t cookie, bool serialize,
           hsa_signal_t dep) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_outstanding[queue].push_back(entry{queue, gpu_id, sig, cookie, serialize, dep});
  if (serialize) ++g_ser_count[gpu_id];
}

void ReportNow(uint64_t cookie, uint32_t status) {
  if (cookie == 0) return;
  call_done(std::vector<uint64_t>{cookie}, status);
}

void Drain() {
  std::vector<uint64_t> done;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [q, fifo] : g_outstanding) {
      while (!fifo.empty() && hsa_signal_load_relaxed(fifo.front().sig) <= 0) {
        const entry e = fifo.front();
        fifo.pop_front();
        if (e.cookie != 0) done.push_back(e.cookie);
        if (e.serialize) --g_ser_count[e.gpu];
        g_completed[e.sig.handle] = true;
        release_ref(e.dep);
        maybe_recycle(e.sig);
      }
    }
  }
  call_done(done, ROCM_DISPATCH_COMPLETED);  // outside the lock (RC-PKT-18)
}

void DropQueue(const void* queue) {
  Drain();
  std::vector<uint64_t> dropped;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_outstanding.find(queue);
    if (it == g_outstanding.end()) return;
    for (const auto& e : it->second) {
      if (e.cookie != 0) dropped.push_back(e.cookie);
      if (e.serialize) --g_ser_count[e.gpu];
    }
    g_outstanding.erase(it);  // signals of a destroyed queue are leaked, not reused (POC)
  }
  call_done(dropped, ROCM_DISPATCH_DROPPED);
}

bool SerializeActive(uint32_t gpu_id) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_ser_count.find(gpu_id);
  return it != g_ser_count.end() && it->second > 0;
}

hsa_signal_t TakeLastSignal(uint32_t gpu_id) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto it = g_last.find(gpu_id);
  if (it == g_last.end() || it->second.handle == 0) return hsa_signal_t{0};
  ++g_refs[it->second.handle];
  return it->second;
}

void SetLastSignal(uint32_t gpu_id, hsa_signal_t sig) {
  std::lock_guard<std::mutex> lk(g_mu);
  hsa_signal_t old = g_last[gpu_id];
  ++g_refs[sig.handle];
  g_last[gpu_id] = sig;
  release_ref(old);
}
}  // namespace rtc
