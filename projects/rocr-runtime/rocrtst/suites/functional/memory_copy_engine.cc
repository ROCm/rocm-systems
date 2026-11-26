/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/memory_copy_engine.h"

#include <strings.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr size_t kCopyBytes = 4 * 1024 * 1024;
constexpr size_t kCopyWords = kCopyBytes / sizeof(uint32_t);

// A bounded wait so that a lost completion fails the test instead of hanging it.
constexpr uint64_t kWaitSeconds = 30;

bool WaitForZero(hsa_signal_t signal) {
  uint64_t freq = 0;
  if (hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq) != HSA_STATUS_SUCCESS)
    return false;
  const hsa_signal_value_t value = hsa_signal_wait_scacquire(
      signal, HSA_SIGNAL_CONDITION_EQ, 0, freq * kWaitSeconds, HSA_WAIT_STATE_BLOCKED);
  return value == 0;
}

void FillPattern(uint32_t* buf, size_t words, uint32_t seed) {
  for (size_t i = 0; i < words; ++i) buf[i] = static_cast<uint32_t>(i) * 7u + seed;
}

bool CheckPattern(const uint32_t* buf, size_t words, uint32_t seed) {
  for (size_t i = 0; i < words; ++i) {
    if (buf[i] != static_cast<uint32_t>(i) * 7u + seed) return false;
  }
  return true;
}

// A buffer that every listed agent can access.
class SharedBuffer {
 public:
  ~SharedBuffer() {
    if (ptr_ != nullptr) hsa_amd_memory_pool_free(ptr_);
  }

  hsa_status_t Allocate(hsa_amd_memory_pool_t pool, size_t bytes,
                        const std::vector<hsa_agent_t>& agents) {
    hsa_status_t err = hsa_amd_memory_pool_allocate(pool, bytes, 0, &ptr_);
    if (err != HSA_STATUS_SUCCESS) {
      ptr_ = nullptr;
      return err;
    }
    return hsa_amd_agents_allow_access(static_cast<uint32_t>(agents.size()), agents.data(),
                                       nullptr, ptr_);
  }

  void* get() const { return ptr_; }
  uint32_t* words() const { return static_cast<uint32_t*>(ptr_); }

 private:
  void* ptr_ = nullptr;
};

class Signal {
 public:
  Signal() { created_ = hsa_signal_create(1, 0, nullptr, &signal_) == HSA_STATUS_SUCCESS; }
  ~Signal() {
    if (created_) hsa_signal_destroy(signal_);
  }
  bool ok() const { return created_; }
  hsa_signal_t get() const { return signal_; }
  void Reset() { hsa_signal_store_screlease(signal_, 1); }

 private:
  hsa_signal_t signal_ = {0};
  bool created_ = false;
};

// Picks one engine id that is free and, if the runtime has a preference, preferred.
// Returns HSA_STATUS_ERROR_OUT_OF_RESOURCES when no engine is free.
hsa_status_t PickEngine(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                        hsa_amd_sdma_engine_id_t* engine) {
  uint32_t preferred = 0;
  uint32_t free_mask = 0;
  hsa_status_t err = hsa_amd_memory_get_preferred_copy_engine(dst_agent, src_agent, &preferred);
  if (err != HSA_STATUS_SUCCESS) return err;
  err = hsa_amd_memory_copy_engine_status(dst_agent, src_agent, &free_mask);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (free_mask == 0) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  const uint32_t usable = (preferred & free_mask) ? (preferred & free_mask) : free_mask;
  *engine = static_cast<hsa_amd_sdma_engine_id_t>(1u << (ffs(usable) - 1));
  return HSA_STATUS_SUCCESS;
}

// Copies on an engine, waits for completion, and reports failures through gtest.
bool CopyOnEngineAndWait(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
                         size_t bytes, hsa_amd_sdma_engine_id_t engine, bool force_sdma,
                         const char* label) {
  Signal done;
  if (!done.ok()) {
    ADD_FAILURE() << label << ": cannot create signal";
    return false;
  }
  hsa_status_t err = hsa_amd_memory_async_copy_on_engine(dst, dst_agent, src, src_agent, bytes, 0,
                                                         nullptr, done.get(), engine, force_sdma);
  if (err != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << label << ": copy on engine 0x" << std::hex << engine << " returned " << err;
    return false;
  }
  if (!WaitForZero(done.get())) {
    ADD_FAILURE() << label << ": copy on engine 0x" << std::hex << engine << " timed out";
    return false;
  }
  return true;
}

bool CopyAndWait(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
                 size_t bytes, const char* label) {
  Signal done;
  if (!done.ok()) {
    ADD_FAILURE() << label << ": cannot create signal";
    return false;
  }
  hsa_status_t err =
      hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent, bytes, 0, nullptr, done.get());
  if (err != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << label << ": async copy returned " << err;
    return false;
  }
  if (!WaitForZero(done.get())) {
    ADD_FAILURE() << label << ": async copy timed out";
    return false;
  }
  return true;
}

}  // namespace

MemoryCopyEngineTest::MemoryCopyEngineTest() : TestBase() {
  set_title("RocR Copy Engine Selection Test");
  set_description(
      "Copies memory on an explicitly chosen engine with "
      "hsa_amd_memory_async_copy_on_engine, using the engine status and "
      "preferred engine queries, and checks the profiling timestamps.");
}

MemoryCopyEngineTest::~MemoryCopyEngineTest() {}

void MemoryCopyEngineTest::SetUp() {
  TestBase::SetUp();
  if (test_skipped_) return;
  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));
}

void MemoryCopyEngineTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void MemoryCopyEngineTest::Close() { TestBase::Close(); }

void MemoryCopyEngineTest::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::DisplayResults();
}

void MemoryCopyEngineTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void MemoryCopyEngineTest::CopyOnEngineTest() {
  const hsa_agent_t gpu = *gpu_device1();
  const hsa_agent_t cpu = *cpu_device();
  const std::vector<hsa_agent_t> agents = {gpu, cpu};

  SharedBuffer host_src, host_dst, dev_a, dev_b;
  ASSERT_SUCCESS(host_src.Allocate(cpu_pool(), kCopyBytes, agents));
  ASSERT_SUCCESS(host_dst.Allocate(cpu_pool(), kCopyBytes, agents));
  ASSERT_SUCCESS(dev_a.Allocate(device_pool(), kCopyBytes, agents));
  ASSERT_SUCCESS(dev_b.Allocate(device_pool(), kCopyBytes, agents));

  FillPattern(host_src.words(), kCopyWords, 0x11u);
  memset(host_dst.get(), 0, kCopyBytes);

  // Host to device, then device to host, each on an engine the runtime reports as free.
  hsa_amd_sdma_engine_id_t h2d_engine = HSA_AMD_SDMA_ENGINE_0;
  hsa_status_t err = PickEngine(gpu, cpu, &h2d_engine);
  if (err == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
    printf("  CopyOnEngineTest: SKIPPED (no free copy engine for host to device)\n");
    return;
  }
  ASSERT_SUCCESS(err);
  ASSERT_TRUE(CopyOnEngineAndWait(dev_a.get(), gpu, host_src.get(), cpu, kCopyBytes, h2d_engine,
                                  false, "host to device"));

  hsa_amd_sdma_engine_id_t d2h_engine = HSA_AMD_SDMA_ENGINE_0;
  err = PickEngine(cpu, gpu, &d2h_engine);
  if (err == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
    printf("  CopyOnEngineTest: SKIPPED (no free copy engine for device to host)\n");
    return;
  }
  ASSERT_SUCCESS(err);
  ASSERT_TRUE(CopyOnEngineAndWait(host_dst.get(), cpu, dev_a.get(), gpu, kCopyBytes, d2h_engine,
                                  false, "device to host"));
  ASSERT_TRUE(CheckPattern(host_dst.words(), kCopyWords, 0x11u)) << "host to device to host";

  // Device to device on the same GPU. Without force_copy_on_sdma the runtime uses a blit
  // kernel, with it the copy goes over SDMA on the chosen engine.
  hsa_amd_sdma_engine_id_t d2d_engine = HSA_AMD_SDMA_ENGINE_0;
  err = PickEngine(gpu, gpu, &d2d_engine);
  if (err == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
    printf("  CopyOnEngineTest: SKIPPED (no free copy engine for device to device)\n");
    return;
  }
  ASSERT_SUCCESS(err);

  for (bool force_sdma : {false, true}) {
    const char* label = force_sdma ? "device to device on SDMA" : "device to device on blit kernel";
    const uint32_t seed = force_sdma ? 0x33u : 0x22u;
    FillPattern(host_src.words(), kCopyWords, seed);
    memset(host_dst.get(), 0, kCopyBytes);
    ASSERT_TRUE(CopyAndWait(dev_a.get(), gpu, host_src.get(), cpu, kCopyBytes, label));
    ASSERT_TRUE(CopyOnEngineAndWait(dev_b.get(), gpu, dev_a.get(), gpu, kCopyBytes, d2d_engine,
                                    force_sdma, label));
    ASSERT_TRUE(CopyAndWait(host_dst.get(), cpu, dev_b.get(), gpu, kCopyBytes, label));
    ASSERT_TRUE(CheckPattern(host_dst.words(), kCopyWords, seed)) << label;
  }

  // Dependent copies: the second copy waits on the first copy's signal.
  FillPattern(host_src.words(), kCopyWords, 0x44u);
  memset(host_dst.get(), 0, kCopyBytes);
  Signal first, second;
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_SUCCESS(hsa_amd_memory_async_copy_on_engine(dev_a.get(), gpu, host_src.get(), cpu,
                                                     kCopyBytes, 0, nullptr, first.get(),
                                                     h2d_engine, false));
  hsa_signal_t dep = first.get();
  ASSERT_SUCCESS(hsa_amd_memory_async_copy_on_engine(host_dst.get(), cpu, dev_a.get(), gpu,
                                                     kCopyBytes, 1, &dep, second.get(),
                                                     d2h_engine, false));
  ASSERT_TRUE(WaitForZero(second.get())) << "dependent copy timed out";
  ASSERT_TRUE(CheckPattern(host_dst.words(), kCopyWords, 0x44u)) << "dependent copies";
}

void MemoryCopyEngineTest::CopyOnEngineNegativeTest() {
  const hsa_agent_t gpu = *gpu_device1();
  const hsa_agent_t cpu = *cpu_device();
  const hsa_agent_t invalid_agent = {0};
  const std::vector<hsa_agent_t> agents = {gpu, cpu};

  SharedBuffer host, dev;
  ASSERT_SUCCESS(host.Allocate(cpu_pool(), kCopyBytes, agents));
  ASSERT_SUCCESS(dev.Allocate(device_pool(), kCopyBytes, agents));
  memset(host.get(), 0x5A, kCopyBytes);
  memset(dev.get(), 0, kCopyBytes);

  Signal done;
  ASSERT_TRUE(done.ok());
  hsa_signal_t dep = done.get();

  // Engine ids must have exactly one bit set, and that bit must name an existing engine.
  // The number of SDMA engines differs per GPU, so only ids that are invalid on every GPU
  // are checked here; a high single-bit id can name a real engine on some parts.
  const hsa_amd_sdma_engine_id_t bad_engines[] = {
      static_cast<hsa_amd_sdma_engine_id_t>(0),
      static_cast<hsa_amd_sdma_engine_id_t>(HSA_AMD_SDMA_ENGINE_0 | HSA_AMD_SDMA_ENGINE_1),
  };
  for (hsa_amd_sdma_engine_id_t engine : bad_engines) {
    EXPECT_EQ(hsa_amd_memory_async_copy_on_engine(dev.get(), gpu, host.get(), cpu, kCopyBytes, 0,
                                                  nullptr, done.get(), engine, false),
              HSA_STATUS_ERROR_INVALID_ARGUMENT)
        << "engine 0x" << std::hex << engine;
  }

  const hsa_amd_sdma_engine_id_t engine = HSA_AMD_SDMA_ENGINE_0;
  EXPECT_EQ(hsa_amd_memory_async_copy_on_engine(nullptr, gpu, host.get(), cpu, kCopyBytes, 0,
                                                nullptr, done.get(), engine, false),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(hsa_amd_memory_async_copy_on_engine(dev.get(), gpu, nullptr, cpu, kCopyBytes, 0,
                                                nullptr, done.get(), engine, false),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Dependency signal count and array must agree.
  EXPECT_EQ(hsa_amd_memory_async_copy_on_engine(dev.get(), gpu, host.get(), cpu, kCopyBytes, 0,
                                                &dep, done.get(), engine, false),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(hsa_amd_memory_async_copy_on_engine(dev.get(), gpu, host.get(), cpu, kCopyBytes, 1,
                                                nullptr, done.get(), engine, false),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Invalid agent and signal handles.
  EXPECT_NE(hsa_amd_memory_async_copy_on_engine(dev.get(), invalid_agent, host.get(), cpu,
                                                kCopyBytes, 0, nullptr, done.get(), engine, false),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(hsa_amd_memory_async_copy_on_engine(dev.get(), gpu, host.get(), invalid_agent,
                                                kCopyBytes, 0, nullptr, done.get(), engine, false),
            HSA_STATUS_SUCCESS);
  hsa_signal_t invalid_signal = {0};
  EXPECT_NE(hsa_amd_memory_async_copy_on_engine(dev.get(), gpu, host.get(), cpu, kCopyBytes, 0,
                                                nullptr, invalid_signal, engine, false),
            HSA_STATUS_SUCCESS);

  // Engine queries reject unknown agents.
  uint32_t mask = 0;
  EXPECT_NE(hsa_amd_memory_copy_engine_status(invalid_agent, cpu, &mask), HSA_STATUS_SUCCESS);
  EXPECT_NE(hsa_amd_memory_copy_engine_status(gpu, invalid_agent, &mask), HSA_STATUS_SUCCESS);
  EXPECT_NE(hsa_amd_memory_get_preferred_copy_engine(invalid_agent, cpu, &mask),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(hsa_amd_memory_get_preferred_copy_engine(gpu, invalid_agent, &mask),
            HSA_STATUS_SUCCESS);

  // The destination was untouched by every rejected call, and the signal is still armed.
  EXPECT_EQ(hsa_signal_load_scacquire(done.get()), 1);
  const uint8_t* bytes = static_cast<const uint8_t*>(dev.get());
  for (size_t i = 0; i < kCopyBytes; i += 4096) {
    ASSERT_EQ(bytes[i], 0) << "rejected copy wrote to the destination at offset " << i;
  }
}

void MemoryCopyEngineTest::AsyncCopyProfilingTest() {
  const hsa_agent_t gpu = *gpu_device1();
  const hsa_agent_t cpu = *cpu_device();
  const std::vector<hsa_agent_t> agents = {gpu, cpu};

  SharedBuffer host, dev;
  ASSERT_SUCCESS(host.Allocate(cpu_pool(), kCopyBytes, agents));
  ASSERT_SUCCESS(dev.Allocate(device_pool(), kCopyBytes, agents));
  FillPattern(host.words(), kCopyWords, 0x55u);

  ASSERT_SUCCESS(hsa_amd_profiling_async_copy_enable(true));

  auto check_times = [&](hsa_signal_t signal, const char* label) {
    hsa_amd_profiling_async_copy_time_t time = {};
    ASSERT_SUCCESS(hsa_amd_profiling_get_async_copy_time(signal, &time)) << label;
    EXPECT_NE(time.start, 0u) << label;
    EXPECT_GE(time.end, time.start) << label;
    printf("  %s: %lu ticks\n", label, static_cast<unsigned long>(time.end - time.start));
  };

  // The default engine choice.
  {
    Signal done;
    ASSERT_TRUE(done.ok());
    ASSERT_SUCCESS(hsa_amd_memory_async_copy(dev.get(), gpu, host.get(), cpu, kCopyBytes, 0,
                                             nullptr, done.get()));
    ASSERT_TRUE(WaitForZero(done.get()));
    check_times(done.get(), "async copy");
  }

  // An explicit engine.
  hsa_amd_sdma_engine_id_t engine = HSA_AMD_SDMA_ENGINE_0;
  hsa_status_t err = PickEngine(cpu, gpu, &engine);
  if (err == HSA_STATUS_SUCCESS) {
    Signal done;
    ASSERT_TRUE(done.ok());
    memset(host.get(), 0, kCopyBytes);
    ASSERT_SUCCESS(hsa_amd_memory_async_copy_on_engine(host.get(), cpu, dev.get(), gpu,
                                                       kCopyBytes, 0, nullptr, done.get(), engine,
                                                       false));
    ASSERT_TRUE(WaitForZero(done.get()));
    check_times(done.get(), "async copy on engine");
    EXPECT_TRUE(CheckPattern(host.words(), kCopyWords, 0x55u));
  } else {
    EXPECT_EQ(err, HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    printf("  AsyncCopyProfilingTest: no free engine, skipped the on-engine copy\n");
  }

  ASSERT_SUCCESS(hsa_amd_profiling_async_copy_enable(false));

  // A signal that no copy has used has no copy agent and cannot report times.
  Signal unused;
  ASSERT_TRUE(unused.ok());
  hsa_amd_profiling_async_copy_time_t time = {};
  EXPECT_NE(hsa_amd_profiling_get_async_copy_time(unused.get(), &time), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_profiling_get_async_copy_time(unused.get(), nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

void MemoryCopyEngineTest::PeerCopyTest() {
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  if (gpus.size() < 2) {
    printf("  PeerCopyTest: SKIPPED (needs two GPUs, found %zu)\n", gpus.size());
    return;
  }

  const hsa_agent_t cpu = *cpu_device();
  hsa_agent_t gpu_a = gpus[0];
  hsa_agent_t gpu_b = gpus[1];
  const std::vector<hsa_agent_t> agents = {gpu_a, gpu_b, cpu};

  hsa_amd_memory_pool_t pool_a = {0};
  hsa_amd_memory_pool_t pool_b = {0};
  ASSERT_SUCCESS(hsa_amd_agent_iterate_memory_pools(gpu_a, rocrtst::FindStandardPool, &pool_a));
  ASSERT_SUCCESS(hsa_amd_agent_iterate_memory_pools(gpu_b, rocrtst::FindStandardPool, &pool_b));

  SharedBuffer host_src, host_dst, dev_a, dev_b;
  ASSERT_SUCCESS(host_src.Allocate(cpu_pool(), kCopyBytes, agents));
  ASSERT_SUCCESS(host_dst.Allocate(cpu_pool(), kCopyBytes, agents));
  hsa_status_t err = dev_a.Allocate(pool_a, kCopyBytes, agents);
  if (err == HSA_STATUS_SUCCESS) err = dev_b.Allocate(pool_b, kCopyBytes, agents);
  if (err != HSA_STATUS_SUCCESS) {
    printf("  PeerCopyTest: SKIPPED (GPUs cannot access each other's memory, status %d)\n", err);
    return;
  }

  // Default engine choice.
  FillPattern(host_src.words(), kCopyWords, 0x66u);
  memset(host_dst.get(), 0, kCopyBytes);
  ASSERT_TRUE(CopyAndWait(dev_a.get(), gpu_a, host_src.get(), cpu, kCopyBytes, "host to GPU A"));
  ASSERT_TRUE(CopyAndWait(dev_b.get(), gpu_b, dev_a.get(), gpu_a, kCopyBytes, "GPU A to GPU B"));
  ASSERT_TRUE(CopyAndWait(host_dst.get(), cpu, dev_b.get(), gpu_b, kCopyBytes, "GPU B to host"));
  ASSERT_TRUE(CheckPattern(host_dst.words(), kCopyWords, 0x66u)) << "peer copy";

  // Explicit engine, in both directions.
  const hsa_agent_t pair[2][2] = {{gpu_b, gpu_a}, {gpu_a, gpu_b}};  // {dst, src}
  SharedBuffer* const bufs[2][2] = {{&dev_b, &dev_a}, {&dev_a, &dev_b}};
  for (int dir = 0; dir < 2; ++dir) {
    const uint32_t seed = 0x77u + dir;
    FillPattern(host_src.words(), kCopyWords, seed);
    memset(host_dst.get(), 0, kCopyBytes);
    const hsa_agent_t dst_agent = pair[dir][0];
    const hsa_agent_t src_agent = pair[dir][1];
    SharedBuffer& dst = *bufs[dir][0];
    SharedBuffer& src = *bufs[dir][1];

    hsa_amd_sdma_engine_id_t engine = HSA_AMD_SDMA_ENGINE_0;
    err = PickEngine(dst_agent, src_agent, &engine);
    if (err == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      printf("  PeerCopyTest: no free engine for direction %d, skipped\n", dir);
      continue;
    }
    ASSERT_SUCCESS(err);
    ASSERT_TRUE(CopyAndWait(src.get(), src_agent, host_src.get(), cpu, kCopyBytes, "host to src"));
    ASSERT_TRUE(CopyOnEngineAndWait(dst.get(), dst_agent, src.get(), src_agent, kCopyBytes, engine,
                                    false, "peer copy on engine"));
    ASSERT_TRUE(CopyAndWait(host_dst.get(), cpu, dst.get(), dst_agent, kCopyBytes, "dst to host"));
    ASSERT_TRUE(CheckPattern(host_dst.words(), kCopyWords, seed)) << "peer copy on engine " << dir;
  }
}
