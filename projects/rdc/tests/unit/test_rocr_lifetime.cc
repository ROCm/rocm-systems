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

// GPU-less checks of the ROCr diagnostic module (librdc_rocr.so), which runs the Compute Queue
// and System memory checks of "rdci diag".
//
// The test compiles the module's sources and defines the HSA runtime functions they call. The
// fake runtime counts its references and the objects the checks create, so the tests can check
// without a GPU that:
//  - ROCr stays loaded from the first check until the module is destroyed. rdcd also loads
//    rocprofiler-sdk, which keeps the agents and queues of the first ROCr load it sees; when a
//    check unloaded ROCr and the next one loaded it again, rocprofiler-sdk was left with stale
//    handles and aborted rdcd ("Could not find agent", "HSA Queue is not initialized").
//  - The checks of a GPU share one queue, which the module destroys when it is destroyed.
//    rocprofiler-sdk does not destroy a queue it intercepted, so a queue per check stayed in
//    rdcd until queue creation failed.
//  - Each check releases the signals, memory and code objects it creates, whether it passes,
//    fails or is skipped. With ROCr kept loaded, nothing else releases them.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcDiagnosticLibInterface.h"

namespace {

constexpr uint64_t kCpuAgent = 0x100;
constexpr uint64_t kGpuAgent = 0x200;
constexpr uint64_t kKernargPool = 0x1000;  // CPU, fine grained, kernel arguments
constexpr uint64_t kSystemPool = 0x2000;   // CPU, fine grained
constexpr uint64_t kVramPool = 0x3000;     // GPU, coarse grained
constexpr size_t kGranule = 4096;
constexpr size_t kPoolSize = 256 * kGranule;
constexpr uint32_t kQueueSize = 128;
constexpr char kGpuName[] = "gfx-rdc-fake";

// The runtime state the fake keeps. Every call takes the mutex.
struct FakeHsa {
  std::mutex mutex;
  int refs = 0;           // hsa_init() calls not yet matched by an hsa_shut_down()
  int loads = 0;          // hsa_init() calls that found no reference (ROCr loads)
  int unloads = 0;        // hsa_shut_down() calls that released the last reference
  int shut_downs = 0;     // every hsa_shut_down() call
  int failing_inits = 0;  // the next hsa_init() calls that fail
  int queue_creates = 0;  // hsa_queue_create() calls that created a queue
  uint64_t next_handle = 1;
  std::set<void*> buffers;
  std::map<hsa_queue_t*, uint64_t> queues;  // queue -> write index
  std::set<uint64_t> signals;
  std::set<uint64_t> executables;
  std::set<uint64_t> readers;
};

FakeHsa& fake() {
  static FakeHsa state;
  return state;
}

size_t live_objects(FakeHsa& f) {
  return f.buffers.size() + f.queues.size() + f.signals.size() + f.executables.size() +
         f.readers.size();
}

}  // namespace

extern "C" {

hsa_status_t hsa_init() {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (f.failing_inits > 0) {
    --f.failing_inits;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  if (f.refs++ == 0) {
    ++f.loads;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_shut_down() {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  ++f.shut_downs;
  if (f.refs == 0) {
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  }
  if (--f.refs == 0) {
    ++f.unloads;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_status_string(hsa_status_t, const char** status_string) {
  *status_string = "fake HSA status";
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data),
                                void* data) {
  {
    FakeHsa& f = fake();
    std::lock_guard<std::mutex> lock(f.mutex);
    if (f.refs == 0) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }
  }
  for (uint64_t handle : {kCpuAgent, kGpuAgent}) {
    hsa_status_t status = callback(hsa_agent_t{handle}, data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value) {
  const bool gpu = agent.handle == kGpuAgent;
  switch (attribute) {
    case HSA_AGENT_INFO_NAME:
      memset(value, 0, 64);
      strcpy(static_cast<char*>(value), gpu ? kGpuName : "cpu-rdc-fake");
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_DEVICE:
      *static_cast<hsa_device_type_t*>(value) = gpu ? HSA_DEVICE_TYPE_GPU : HSA_DEVICE_TYPE_CPU;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_NODE:
      *static_cast<uint32_t*>(value) = gpu ? 1 : 0;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE:
      *static_cast<uint32_t*>(value) = kQueueSize;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_PROFILE:
      *static_cast<hsa_profile_t*>(value) = HSA_PROFILE_BASE;
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data) {
  const bool gpu = agent.handle == kGpuAgent;
  for (uint64_t handle :
       gpu ? std::set<uint64_t>{kVramPool} : std::set<uint64_t>{kKernargPool, kSystemPool}) {
    hsa_status_t status = callback(hsa_amd_memory_pool_t{handle}, data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute, void* value) {
  const bool vram = memory_pool.handle == kVramPool;
  switch (attribute) {
    case HSA_AMD_MEMORY_POOL_INFO_SEGMENT:
      *static_cast<hsa_amd_segment_t*>(value) = HSA_AMD_SEGMENT_GLOBAL;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: {
      uint32_t flags = vram ? HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
                            : HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
      if (memory_pool.handle == kKernargPool) {
        flags |= HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
      }
      *static_cast<uint32_t*>(value) = flags;
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AMD_MEMORY_POOL_INFO_SIZE:
    case HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE:
      *static_cast<size_t*>(value) = kPoolSize;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED:
      // The GPU pool reports no runtime allocation, so the checks that size their allocations
      // from it (Max Single Allocation, the CPU's access to GPU memory) report "not applicable"
      // instead of allocating half of it. The GPU access check still allocates from it.
      *static_cast<bool*>(value) = !vram;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE:
    case HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT:
      *static_cast<size_t*>(value) = kGranule;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL:
      *static_cast<bool*>(value) = false;
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                hsa_amd_memory_pool_t memory_pool,
                                                hsa_amd_agent_memory_pool_info_t attribute,
                                                void* value) {
  if (attribute != HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const bool cpu_to_vram = agent.handle == kCpuAgent && memory_pool.handle == kVramPool;
  *static_cast<hsa_amd_memory_pool_access_t*>(value) =
      cpu_to_vram ? HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT
                  : HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t, size_t size, uint32_t,
                                          void** ptr) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (f.refs == 0) {
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  }
  if (size == 0 || size > kPoolSize) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  // Round up to the granule, as ROCr does: the checks use more than they ask for.
  const size_t rounded = (size + kGranule - 1) / kGranule * kGranule;
  void* buffer = aligned_alloc(kGranule, rounded);
  memset(buffer, 0, rounded);
  f.buffers.insert(buffer);
  *ptr = buffer;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_memory_pool_free(void* ptr) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (f.buffers.erase(ptr) == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  free(ptr);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_memory_free(void* ptr) { return hsa_amd_memory_pool_free(ptr); }

hsa_status_t hsa_amd_agents_allow_access(uint32_t, const hsa_agent_t*, const uint32_t*,
                                         const void*) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
  return HSA_STATUS_SUCCESS;
}

// The copy completes at once; hsa_signal_wait_scacquire() then reports it done.
hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t, const void* src, hsa_agent_t,
                                       size_t size, uint32_t, const hsa_signal_t*, hsa_signal_t) {
  memcpy(dst, src, size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_pointer_info(const void*, hsa_amd_pointer_info_t*, void* (*)(size_t),
                                  uint32_t*, hsa_agent_t**) {
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_signal_create(hsa_signal_value_t, uint32_t, const hsa_agent_t*,
                               hsa_signal_t* signal) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (f.refs == 0) {
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  }
  signal->handle = f.next_handle++;
  f.signals.insert(signal->handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_signal_destroy(hsa_signal_t signal) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  return f.signals.erase(signal.handle) == 1 ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_SIGNAL;
}

void hsa_signal_store_relaxed(hsa_signal_t, hsa_signal_value_t) {}

void hsa_signal_store_screlease(hsa_signal_t, hsa_signal_value_t) {}

// Every dispatch and copy is complete when it is waited for.
hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t, hsa_signal_condition_t,
                                             hsa_signal_value_t, uint64_t, hsa_wait_state_t) {
  return 0;
}

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t,
                              void (*)(hsa_status_t, hsa_queue_t*, void*), void*, uint32_t,
                              uint32_t, hsa_queue_t** queue) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (f.refs == 0) {
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  }
  if (agent.handle != kGpuAgent || size == 0 || size > kQueueSize) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto* created = new hsa_queue_t{};
  created->base_address = aligned_alloc(64, size * sizeof(hsa_kernel_dispatch_packet_t));
  created->size = size;
  created->id = f.next_handle++;
  f.queues[created] = 0;
  ++f.queue_creates;
  *queue = created;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_queue_destroy(hsa_queue_t* queue) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (f.queues.erase(queue) == 0) {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }
  free(queue->base_address);
  delete queue;
  return HSA_STATUS_SUCCESS;
}

uint64_t hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  return f.queues.at(const_cast<hsa_queue_t*>(queue));
}

uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  uint64_t& index = f.queues.at(const_cast<hsa_queue_t*>(queue));
  const uint64_t old = index;
  index += value;
  return old;
}

void hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  f.queues.at(const_cast<hsa_queue_t*>(queue)) = value;
}

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t,
                                                     hsa_code_object_reader_t* code_object_reader) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  code_object_reader->handle = f.next_handle++;
  f.readers.insert(code_object_reader->handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  return f.readers.erase(code_object_reader.handle) == 1 ? HSA_STATUS_SUCCESS
                                                         : HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t hsa_executable_create_alt(hsa_profile_t, hsa_default_float_rounding_mode_t,
                                       const char*, hsa_executable_t* executable) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  executable->handle = f.next_handle++;
  f.executables.insert(executable->handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_destroy(hsa_executable_t executable) {
  FakeHsa& f = fake();
  std::lock_guard<std::mutex> lock(f.mutex);
  return f.executables.erase(executable.handle) == 1 ? HSA_STATUS_SUCCESS
                                                     : HSA_STATUS_ERROR_INVALID_EXECUTABLE;
}

hsa_status_t hsa_executable_load_agent_code_object(hsa_executable_t, hsa_agent_t,
                                                   hsa_code_object_reader_t, const char*,
                                                   hsa_loaded_code_object_t*) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_freeze(hsa_executable_t, const char*) { return HSA_STATUS_SUCCESS; }

hsa_status_t hsa_executable_get_symbol(hsa_executable_t, const char*, const char*, hsa_agent_t,
                                       int32_t, hsa_executable_symbol_t* symbol) {
  symbol->handle = 1;
  return HSA_STATUS_SUCCESS;
}

// The kernels take the code object v4 argument layout the checks fill in: 80 bytes.
hsa_status_t hsa_executable_symbol_get_info(hsa_executable_symbol_t,
                                            hsa_executable_symbol_info_t attribute, void* value) {
  switch (attribute) {
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:
      *static_cast<uint64_t*>(value) = 0x1234;
      return HSA_STATUS_SUCCESS;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      return HSA_STATUS_SUCCESS;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
      *static_cast<uint32_t*>(value) = 80;
      return HSA_STATUS_SUCCESS;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:
      *static_cast<uint32_t*>(value) = 16;
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

}  // extern "C"

namespace {

class RocrDiagLifetime : public ::testing::Test {
 protected:
  void SetUp() override {
    FakeHsa& f = fake();
    std::lock_guard<std::mutex> lock(f.mutex);
    ASSERT_EQ(f.refs, 0) << "an earlier test left ROCr loaded";
    f.loads = f.unloads = f.shut_downs = f.failing_inits = f.queue_creates = 0;
    // Drop what an earlier test left behind; that test has already reported it.
    for (void* buffer : f.buffers) {
      free(buffer);
    }
    for (auto& [queue, index] : f.queues) {
      free(queue->base_address);
      delete queue;
    }
    f.buffers.clear();
    f.queues.clear();
    f.signals.clear();
    f.executables.clear();
    f.readers.clear();
  }

  // Release the module's reference, as rdcd does when it destroys the module.
  void TearDown() override { rdc_diag_destroy(); }

  static rdc_diag_result_t Run(rdc_diag_test_cases_t test_case, std::string* info = nullptr) {
    uint32_t gpu_index[RDC_MAX_NUM_DEVICES] = {0};
    auto result = std::make_unique<rdc_diag_test_result_t>();
    EXPECT_EQ(rdc_diag_test_case_run(test_case, gpu_index, 1, nullptr, 0, result.get(), nullptr),
              RDC_ST_OK);
    if (info != nullptr) {
      *info = result->info;
    }
    return result->status;
  }

  template <typename F>
  static auto Locked(F fn) {
    FakeHsa& f = fake();
    std::lock_guard<std::mutex> lock(f.mutex);
    return fn(f);
  }

  static int refs() {
    return Locked([](FakeHsa& f) { return f.refs; });
  }
  static int loads() {
    return Locked([](FakeHsa& f) { return f.loads; });
  }
  static int unloads() {
    return Locked([](FakeHsa& f) { return f.unloads; });
  }
  static int shut_downs() {
    return Locked([](FakeHsa& f) { return f.shut_downs; });
  }
  static size_t live() {
    return Locked([](FakeHsa& f) { return live_objects(f); });
  }
  static size_t queues() {
    return Locked([](FakeHsa& f) { return f.queues.size(); });
  }
  static int queue_creates() {
    return Locked([](FakeHsa& f) { return f.queue_creates; });
  }
};

// Puts empty kernel files in a temporary directory and makes it the working directory, where the
// module looks for its kernels first. The fake runtime loads any file.
class KernelFiles {
 public:
  KernelFiles() {
    char tmpl[] = "/tmp/rdc_rocr_lifetime_XXXXXX";
    dir_ = mkdtemp(tmpl);
    old_cwd_ = getcwd(nullptr, 0);
    for (const char* name : {"binary_search_kernels.hsaco", "gpuReadWrite_kernels.hsaco"}) {
      close(open((dir_ + "/" + name).c_str(), O_CREAT | O_WRONLY, 0600));
    }
    EXPECT_EQ(chdir(dir_.c_str()), 0);
  }
  ~KernelFiles() {
    EXPECT_EQ(chdir(old_cwd_), 0);
    free(old_cwd_);
    for (const char* name : {"binary_search_kernels.hsaco", "gpuReadWrite_kernels.hsaco"}) {
      unlink((dir_ + "/" + name).c_str());
    }
    rmdir(dir_.c_str());
  }

 private:
  std::string dir_;
  char* old_cwd_;
};

// The kernel files of the GPU are missing, as on a GPU no kernels are packaged for. Both checks
// are skipped, each after it has taken the GPU's queue and created memory. Repeated runs load
// ROCr once and create one queue, which the module destroys with its reference.
TEST_F(RocrDiagLifetime, SkippedChecksKeepRocrLoadedAndReleaseTheirObjects) {
  std::string info;
  for (int run = 1; run <= 2; ++run) {
    SCOPED_TRACE("run " + std::to_string(run));
    EXPECT_EQ(Run(RDC_DIAG_COMPUTE_QUEUE, &info), RDC_DIAG_RESULT_SKIP);
    EXPECT_NE(info.find("fail to open binary_search_kernels.hsaco"), std::string::npos) << info;
    EXPECT_EQ(queues(), 1U) << "the GPU has no queue, or more than one";
    EXPECT_EQ(live(), 1U) << "the Compute Queue check left objects behind";

    EXPECT_EQ(Run(RDC_DIAG_SYS_MEM_CHECK, &info), RDC_DIAG_RESULT_SKIP);
    EXPECT_NE(info.find("fail to open gpuReadWrite_kernels.hsaco"), std::string::npos) << info;
    EXPECT_EQ(queues(), 1U) << "the GPU has no queue, or more than one";
    EXPECT_EQ(live(), 1U) << "the System memory check left objects behind";

    EXPECT_EQ(refs(), 1) << "the module does not keep ROCr loaded between runs";
  }
  EXPECT_EQ(queue_creates(), 1) << "the checks do not share the GPU's queue";
  EXPECT_EQ(loads(), 1);
  EXPECT_EQ(unloads(), 0);

  rdc_diag_destroy();
  EXPECT_EQ(live(), 0U) << "the module did not destroy the GPU's queue";
  EXPECT_EQ(refs(), 0);
  EXPECT_EQ(unloads(), 1);
}

// With kernel files, the Compute Queue check dispatches and passes; the System memory check
// dispatches on the same queue and fails its comparison, since the fake runs no kernel. Both
// release what they created, the loaded code objects included.
TEST_F(RocrDiagLifetime, DispatchingChecksReleaseTheirObjects) {
  KernelFiles kernels;
  std::string info;
  EXPECT_EQ(Run(RDC_DIAG_COMPUTE_QUEUE, &info), RDC_DIAG_RESULT_PASS) << info;
  EXPECT_EQ(queues(), 1U);
  EXPECT_EQ(live(), 1U) << "the Compute Queue check left objects behind";

  EXPECT_EQ(Run(RDC_DIAG_SYS_MEM_CHECK, &info), RDC_DIAG_RESULT_FAIL) << info;
  EXPECT_NE(info.find("gpuResult does not match"), std::string::npos) << info;
  EXPECT_EQ(queues(), 1U);
  EXPECT_EQ(live(), 1U) << "the System memory check left objects behind";

  EXPECT_EQ(queue_creates(), 1);
  EXPECT_EQ(refs(), 1);
  EXPECT_EQ(loads(), 1);
}

// The module's own hsa_init() is the first of a run. When it fails, the check runs with its own
// reference but cannot take the GPU's queue, which needs the module's reference, and fails; the
// next run takes the module's reference.
TEST_F(RocrDiagLifetime, FailedInitIsRetriedOnTheNextRun) {
  Locked([](FakeHsa& f) { return f.failing_inits = 1; });
  std::string info;
  EXPECT_EQ(Run(RDC_DIAG_COMPUTE_QUEUE, &info), RDC_DIAG_RESULT_FAIL) << info;
  EXPECT_NE(info.find("Fail to create queue"), std::string::npos) << info;
  EXPECT_EQ(refs(), 0) << "a reference was kept after hsa_init() failed";
  EXPECT_EQ(live(), 0U);

  EXPECT_EQ(Run(RDC_DIAG_COMPUTE_QUEUE), RDC_DIAG_RESULT_SKIP);
  EXPECT_EQ(refs(), 1);
  EXPECT_EQ(queues(), 1U);
  EXPECT_EQ(live(), 1U);
}

// Destroying a module that holds no reference calls no hsa_shut_down().
TEST_F(RocrDiagLifetime, DestroyWithoutRunReleasesNothing) {
  rdc_diag_destroy();
  EXPECT_EQ(shut_downs(), 0);
  EXPECT_EQ(refs(), 0);
}

}  // namespace
