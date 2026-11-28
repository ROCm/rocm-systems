/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "suites/functional/memory_async_batch_copy.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

constexpr uint64_t kCompletionTimeoutSec = 60;
constexpr uint8_t kPoison = 0xA5;
constexpr size_t kPage = 4096;

// -----------------------------------------------------------------------------
// Target capabilities
// -----------------------------------------------------------------------------

// What the runtime is expected to accept on this system. The linear operation
// types fall back to a shader blit when SDMA is off. Everything else needs SDMA.
//
// A capability that the runtime reports is read from the runtime rather than
// predicted here, so that the expectation tracks whichever routing is actually
// in place. Where no such attribute exists the flag records only what MUST
// succeed; a type may still be serviceable when its flag is false, so a test
// gates its skip on the attempt rather than on the flag.
struct Ctx {
  hsa_agent_t cpu;
  hsa_agent_t gpu;
  hsa_amd_memory_pool_t cpu_pool;
  hsa_amd_memory_pool_t dev_pool;
  bool sdma_ops;   // SDMA-only operation types are expected to work
  bool rect;       // HSA_AMD_MEMORY_COPY_OP_RECT (gfx9 and newer)
  bool swap;       // HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP, native SDMA packet
  bool indirect;   // HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_* (gfx125+)
  bool broadcast;  // HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST, either routing
};

bool SdmaDisabledByEnv() {
  const char* env = getenv("HSA_ENABLE_SDMA");
  return env != nullptr && env[0] == '0';
}

int HexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return 0;
}

// "gfx942" -> 9.4.2, "gfx90a" -> 9.0.10, "gfx1250" -> 12.5.0, "gfx1030" -> 10.3.0
void ParseGfx(const char* name, int* major, int* minor, int* step) {
  *major = *minor = *step = 0;
  if (strncmp(name, "gfx", 3) != 0) return;
  std::string s(name + 3);
  // Keep the leading hex digits only: drop ":xnack+" style features and
  // suffixes such as "-strict".
  size_t keep = 0;
  while (keep < s.size() && isxdigit(static_cast<unsigned char>(s[keep]))) ++keep;
  s.resize(keep);
  if (s.size() == 3) {
    *major = HexDigit(s[0]);
    *minor = HexDigit(s[1]);
    *step = HexDigit(s[2]);
  } else if (s.size() == 4) {
    *major = HexDigit(s[0]) * 10 + HexDigit(s[1]);
    *minor = HexDigit(s[2]);
    *step = HexDigit(s[3]);
  }
}

Ctx MakeCtx(rocrtst::BaseRocR* test) {
  Ctx c;
  c.cpu = *test->cpu_device();
  c.gpu = *test->gpu_device1();
  c.cpu_pool = test->cpu_pool();
  c.dev_pool = test->device_pool();

  char name[64] = {0};
  hsa_agent_get_info(c.gpu, HSA_AGENT_INFO_NAME, name);
  int major, minor, step;
  ParseGfx(name, &major, &minor, &step);

  // A full-profile agent does not issue these copies on SDMA.
  c.sdma_ops = !SdmaDisabledByEnv() && test->profile() == HSA_PROFILE_BASE;
  const bool gfx125plus = (major == 12 && minor >= 5);
  c.rect = c.sdma_ops && major >= 9;
  c.indirect = c.sdma_ops && gfx125plus;

  // Only the native SDMA packet is predicted here. A target without it may
  // still serve a swap from a shader blit, which this expectation cannot see,
  // so SwapTest treats a false flag as "not required" rather than "absent".
  c.swap = c.sdma_ops && ((major == 9 && minor >= 4) || gfx125plus);

  // Broadcast is asked rather than predicted. The agent answers zero when
  // nothing can service the operation and the API's own ceiling when something
  // can, whichever routing supplies it, so this stays correct as the set of
  // routings changes.
  uint32_t max_bcast = 0;
  c.broadcast = hsa_agent_get_info(c.gpu,
                    static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_BROADCAST_DESTINATIONS),
                    &max_bcast) == HSA_STATUS_SUCCESS &&
                max_bcast > 0;
  return c;
}

// -----------------------------------------------------------------------------
// Resources that must outlive an unfinished copy
// -----------------------------------------------------------------------------

// Frees its buffers and signals on destruction, unless a copy may still be in
// flight. Freeing under a running SDMA copy turns a test failure into a GPU
// fault, so after a timeout or a failed assertion the resources are leaked.
struct Resources {
  std::vector<void*> bufs;
  std::vector<hsa_signal_t> sigs;
  bool outstanding = false;

  Resources() = default;
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;

  ~Resources() {
    if (outstanding) return;
    for (void* p : bufs) hsa_amd_memory_pool_free(p);
    for (hsa_signal_t s : sigs) hsa_signal_destroy(s);
  }
};

std::string StatusStr(hsa_status_t st) {
  const char* msg = nullptr;
  if (hsa_status_string(st, &msg) != HSA_STATUS_SUCCESS || msg == nullptr) msg = "unknown";
  return std::string(msg) + " (" + std::to_string(static_cast<int>(st)) + ")";
}

void* AllocHost(Resources& r, const Ctx& c, size_t size) {
  void* p = nullptr;
  if (hsa_amd_memory_pool_allocate(c.cpu_pool, size, 0, &p) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << "host allocation of " << size << " bytes failed";
    return nullptr;
  }
  r.bufs.push_back(p);
  memset(p, 0, size);
  if (hsa_amd_agents_allow_access(1, &c.gpu, nullptr, p) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << "granting the GPU access to host memory failed";
    return nullptr;
  }
  return p;
}

void* AllocDev(Resources& r, const Ctx& c, size_t size) {
  void* p = nullptr;
  if (hsa_amd_memory_pool_allocate(c.dev_pool, size, 0, &p) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << "device allocation of " << size << " bytes failed";
    return nullptr;
  }
  r.bufs.push_back(p);
  return p;
}

bool NewSignal(Resources& r, hsa_signal_value_t initial, hsa_signal_t* out) {
  if (hsa_signal_create(initial, 0, nullptr, out) != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << "hsa_signal_create failed";
    return false;
  }
  r.sigs.push_back(*out);
  return true;
}

// Waits for the completion signal to reach exactly zero. A value below zero
// means the runtime signalled the batch more often than it has operations.
bool WaitSignal(hsa_signal_t sig, const char* what) {
  uint64_t freq = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
  const hsa_signal_value_t v = hsa_signal_wait_scacquire(
      sig, HSA_SIGNAL_CONDITION_LT, 1, freq * kCompletionTimeoutSec, HSA_WAIT_STATE_BLOCKED);
  if (v != 0) {
    ADD_FAILURE() << what << ": completion signal is " << v << " after " << kCompletionTimeoutSec
                  << " s (expected 0)";
    return false;
  }
  return true;
}

bool WaitDone(Resources& r, hsa_signal_t sig, const char* what) {
  if (!WaitSignal(sig, what)) return false;
  r.outstanding = false;
  return true;
}

// Submits the batch. On success a copy may be in flight until WaitDone().
hsa_status_t Submit(Resources& r, const hsa_amd_memory_copy_op_t* ops, uint32_t n,
                    uint32_t num_deps = 0, const hsa_signal_t* deps = nullptr) {
  const hsa_status_t st = hsa_amd_memory_async_batch_copy(ops, n, num_deps, deps);
  if (st == HSA_STATUS_SUCCESS) r.outstanding = true;
  return st;
}

bool RunBatch(Resources& r, const std::vector<hsa_amd_memory_copy_op_t>& ops, hsa_signal_t done,
              const char* what) {
  const hsa_status_t st = Submit(r, ops.data(), static_cast<uint32_t>(ops.size()));
  if (st != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << what << ": hsa_amd_memory_async_batch_copy returned " << StatusStr(st);
    return false;
  }
  return WaitDone(r, done, what);
}

// Copies through hsa_amd_memory_async_copy. The batch API under test is only
// used for the direction being checked; staging and readback use the long
// established path so a failure points at the batch copy.
bool Stage(Resources& r, void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
           size_t size) {
  hsa_signal_t s;
  if (!NewSignal(r, 1, &s)) return false;
  const hsa_status_t st =
      hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent, size, 0, nullptr, s);
  if (st != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << "staging copy failed: " << StatusStr(st);
    return false;
  }
  const bool was = r.outstanding;
  r.outstanding = true;
  if (!WaitSignal(s, "staging copy")) return false;
  r.outstanding = was;
  return true;
}

// -----------------------------------------------------------------------------
// Data helpers
// -----------------------------------------------------------------------------

void FillPattern(void* p, size_t n, uint32_t seed) {
  uint8_t* b = static_cast<uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    b[i] = static_cast<uint8_t>((i * 131u + seed * 17u + (i >> 9)) & 0xff);
  }
}

bool SameBytes(const void* got, const void* want, size_t n, const std::string& what) {
  if (memcmp(got, want, n) == 0) return true;
  const uint8_t* g = static_cast<const uint8_t*>(got);
  const uint8_t* w = static_cast<const uint8_t*>(want);
  size_t i = 0;
  while (g[i] == w[i]) ++i;
  ADD_FAILURE() << what << ": first mismatch at byte " << i << " of " << n << " (got "
                << static_cast<int>(g[i]) << ", expected " << static_cast<int>(w[i]) << ")";
  return false;
}

bool IsPoison(const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  return std::all_of(b, b + n, [](uint8_t v) { return v == kPoison; });
}

// -----------------------------------------------------------------------------
// Operation builders
// -----------------------------------------------------------------------------

hsa_amd_memory_copy_op_t NewOp(hsa_amd_memory_copy_op_type_t type, hsa_signal_t done) {
  hsa_amd_memory_copy_op_t op = {};
  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = type;
  op.completion_signal = done;
  return op;
}

hsa_amd_memory_copy_op_t LinearOp(hsa_signal_t done, void* dst, hsa_agent_t dst_agent,
                                  const void* src, hsa_agent_t src_agent, size_t size) {
  hsa_amd_memory_copy_op_t op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done);
  op.src = const_cast<void*>(src);
  op.src_agent = src_agent;
  op.dst = dst;
  op.dst_agent = dst_agent;
  op.size = size;
  return op;
}

// Owns the arrays a multi-entry operation points at. Keep it alive until the
// operation completes.
struct MultiEntry {
  std::vector<void*> src;
  std::vector<void*> dst;
  std::vector<hsa_agent_t> dst_agents;
  std::vector<size_t> sizes;

  void Add(void* s, void* d, hsa_agent_t da, size_t size) {
    src.push_back(s);
    dst.push_back(d);
    dst_agents.push_back(da);
    sizes.push_back(size);
  }
};

hsa_amd_memory_copy_op_t MultiOp(hsa_amd_memory_copy_op_type_t type, hsa_signal_t done,
                                 hsa_agent_t src_agent, MultiEntry& m) {
  hsa_amd_memory_copy_op_t op = NewOp(type, done);
  op.src_list = m.src.data();
  op.src_agent = src_agent;
  op.dst_list = m.dst.data();
  op.dst_agent_list = m.dst_agents.data();
  op.size_list = m.sizes.data();
  op.num_entries = static_cast<uint16_t>(m.sizes.size());
  return op;
}

// One RECT operation: one or two boxes in pitched images. Builds the image the
// destination must hold afterwards, so the check is a plain memcmp.
struct RectJob {
  static constexpr size_t kSrcPitch = 256;
  static constexpr size_t kDstPitch = 512;
  static constexpr size_t kRows = 64;
  static constexpr size_t kSrcBytes = kSrcPitch * kRows;
  static constexpr size_t kDstBytes = kDstPitch * kRows;

  std::vector<hsa_amd_memory_copy_rect_ptr_t> src;
  std::vector<hsa_amd_memory_copy_rect_ptr_t> dst;
  std::vector<hsa_dim3_t> range;
  std::vector<hsa_agent_t> dst_agents;
  std::vector<uint8_t> expected;

  void AddBox(void* src_base, void* dst_base, hsa_agent_t dst_agent, hsa_dim3_t src_off,
              hsa_dim3_t dst_off, hsa_dim3_t box, const uint8_t* src_img) {
    src.push_back({{src_base, kSrcPitch, kSrcBytes}, src_off});
    dst.push_back({{dst_base, kDstPitch, kDstBytes}, dst_off});
    range.push_back(box);
    dst_agents.push_back(dst_agent);
    for (uint32_t y = 0; y < box.y; ++y) {
      memcpy(expected.data() + (dst_off.y + y) * kDstPitch + dst_off.x,
             src_img + (src_off.y + y) * kSrcPitch + src_off.x, box.x);
    }
  }

  // Boxes are 4-byte aligned in x and do not overlap each other.
  void Build(void* src_base, void* dst_base, hsa_agent_t dst_agent, const uint8_t* src_img,
             const uint8_t* dst_init_img, bool two_boxes) {
    expected.assign(dst_init_img, dst_init_img + kDstBytes);
    AddBox(src_base, dst_base, dst_agent, {16, 2, 0}, {32, 4, 0}, {128, 16, 1}, src_img);
    if (two_boxes) {
      AddBox(src_base, dst_base, dst_agent, {64, 24, 0}, {200, 30, 0}, {96, 8, 1}, src_img);
    }
  }

  hsa_amd_memory_copy_op_t Op(hsa_signal_t done, hsa_agent_t src_agent) {
    hsa_amd_memory_copy_op_t op = NewOp(HSA_AMD_MEMORY_COPY_OP_RECT, done);
    op.rect_src = src.data();
    op.rect_dst = dst.data();
    op.range_list = range.data();
    op.dst_agent_list = dst_agents.data();
    op.src_agent = src_agent;
    op.num_entries = static_cast<uint16_t>(range.size());
    return op;
  }
};

void Banner(const char* text) { std::cout << "  *** Batch copy: " << text << " ***" << std::endl; }

void NotSupported(const char* what) {
  // The vendored gtest predates GTEST_SKIP(); emit the conventional marker.
  std::cout << "[ SKIPPED ] " << what << std::endl;
}

}  // namespace

// -----------------------------------------------------------------------------
// Test case plumbing
// -----------------------------------------------------------------------------

MemoryAsyncBatchCopy::MemoryAsyncBatchCopy() : TestBase() {
  set_num_iteration(1);
  set_title("RocR Memory Async Batch Copy Test");
  set_description("This test exercises hsa_amd_memory_async_batch_copy with each operation type, "
                  "dependent signals and invalid arguments.");
}

MemoryAsyncBatchCopy::~MemoryAsyncBatchCopy() {}

void MemoryAsyncBatchCopy::SetUp() {
  TestBase::SetUp();
  if (test_skipped_) return;

  hsa_status_t err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void MemoryAsyncBatchCopy::Run() {
  if (!rocrtst::CheckProfile(this)) return;
  TestBase::Run();
}

void MemoryAsyncBatchCopy::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void MemoryAsyncBatchCopy::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) return;
}

void MemoryAsyncBatchCopy::Close() { TestBase::Close(); }

// -----------------------------------------------------------------------------
// LINEAR, one copy per op
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::LinearSingleTest() {
  Banner("linear, one copy per op");
  const Ctx c = MakeCtx(this);

  const size_t sizes[] = {4, 100, 4096, 65537, 1 << 20};
  constexpr uint32_t kOps = 4;

  for (const size_t size : sizes) {
    Resources res;
    std::vector<uint8_t*> h_src(kOps), h_dst(kOps);
    std::vector<void*> dev(kOps), dev2(kOps);
    for (uint32_t i = 0; i < kOps; ++i) {
      h_src[i] = static_cast<uint8_t*>(AllocHost(res, c, size));
      h_dst[i] = static_cast<uint8_t*>(AllocHost(res, c, size));
      dev[i] = AllocDev(res, c, size);
      dev2[i] = AllocDev(res, c, size);
      ASSERT_TRUE(h_src[i] && h_dst[i] && dev[i] && dev2[i]);
      FillPattern(h_src[i], size, i + 1);
    }
    const std::string tag = "size " + std::to_string(size);

    // Host to device, kOps independent copies sharing one completion signal.
    hsa_signal_t done;
    ASSERT_TRUE(NewSignal(res, kOps, &done));
    std::vector<hsa_amd_memory_copy_op_t> ops;
    for (uint32_t i = 0; i < kOps; ++i) {
      ops.push_back(LinearOp(done, dev[i], c.gpu, h_src[i], c.cpu, size));
    }
    ASSERT_TRUE(RunBatch(res, ops, done, (tag + " H2D").c_str()));

    // Device to device on the same GPU.
    ASSERT_TRUE(NewSignal(res, kOps, &done));
    ops.clear();
    for (uint32_t i = 0; i < kOps; ++i) {
      ops.push_back(LinearOp(done, dev2[i], c.gpu, dev[i], c.gpu, size));
    }
    ASSERT_TRUE(RunBatch(res, ops, done, (tag + " D2D").c_str()));

    // Device to host, from the D2D destination.
    for (uint32_t i = 0; i < kOps; ++i) memset(h_dst[i], kPoison, size);
    ASSERT_TRUE(NewSignal(res, kOps, &done));
    ops.clear();
    for (uint32_t i = 0; i < kOps; ++i) {
      ops.push_back(LinearOp(done, h_dst[i], c.cpu, dev2[i], c.gpu, size));
    }
    ASSERT_TRUE(RunBatch(res, ops, done, (tag + " D2H").c_str()));

    for (uint32_t i = 0; i < kOps; ++i) {
      ASSERT_TRUE(SameBytes(h_dst[i], h_src[i], size, tag + " op " + std::to_string(i)));
    }
  }
}

// -----------------------------------------------------------------------------
// LINEAR, num_entries > 0
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::LinearMultiTest() {
  Banner("linear, multiple entries per op");
  const Ctx c = MakeCtx(this);

  // Slots are page aligned and larger than the largest entry, so a copy that
  // runs past its size lands in the poison between entries.
  constexpr size_t kStride = 65 * kPage;
  const size_t entry_sizes[] = {4, 100, 4096, 5000, 65536 + 17, 262144};
  const uint32_t counts[] = {1, 7, 64};

  for (const uint32_t n : counts) {
    Resources res;
    const size_t total = n * kStride;
    uint8_t* h_src = static_cast<uint8_t*>(AllocHost(res, c, total));
    uint8_t* h_dst = static_cast<uint8_t*>(AllocHost(res, c, total));
    uint8_t* dev = static_cast<uint8_t*>(AllocDev(res, c, total));
    uint8_t* dev2 = static_cast<uint8_t*>(AllocDev(res, c, total));
    ASSERT_TRUE(h_src && h_dst && dev && dev2);
    for (uint32_t e = 0; e < n; ++e) FillPattern(h_src + e * kStride, kStride, e + 1);

    auto entry_size = [&](uint32_t e) { return entry_sizes[e % ARRAY_SIZE(entry_sizes)]; };
    const std::string tag = std::to_string(n) + " entries";

    // Host to device.
    {
      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      MultiEntry m;
      for (uint32_t e = 0; e < n; ++e) {
        m.Add(h_src + e * kStride, dev + e * kStride, c.gpu, entry_size(e));
      }
      std::vector<hsa_amd_memory_copy_op_t> ops = {
          MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, m)};
      ASSERT_TRUE(RunBatch(res, ops, done, (tag + " H2D").c_str()));
    }

    // Device to device, then device to host.
    {
      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      MultiEntry m;
      for (uint32_t e = 0; e < n; ++e) {
        m.Add(dev + e * kStride, dev2 + e * kStride, c.gpu, entry_size(e));
      }
      std::vector<hsa_amd_memory_copy_op_t> ops = {
          MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.gpu, m)};
      ASSERT_TRUE(RunBatch(res, ops, done, (tag + " D2D").c_str()));
    }
    {
      memset(h_dst, kPoison, total);
      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      MultiEntry m;
      for (uint32_t e = 0; e < n; ++e) {
        m.Add(dev2 + e * kStride, h_dst + e * kStride, c.cpu, entry_size(e));
      }
      std::vector<hsa_amd_memory_copy_op_t> ops = {
          MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.gpu, m)};
      ASSERT_TRUE(RunBatch(res, ops, done, (tag + " D2H").c_str()));
    }

    for (uint32_t e = 0; e < n; ++e) {
      const size_t size = entry_size(e);
      const std::string what = tag + " entry " + std::to_string(e);
      ASSERT_TRUE(SameBytes(h_dst + e * kStride, h_src + e * kStride, size, what));
      ASSERT_TRUE(IsPoison(h_dst + e * kStride + size, kStride - size))
          << what << " wrote past its size";
    }
  }
}

// -----------------------------------------------------------------------------
// LINEAR between two GPUs
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::LinearPeerTest() {
  Banner("linear between two GPUs");
  const Ctx c = MakeCtx(this);

  std::vector<hsa_agent_t> gpus;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  if (gpus.size() < 2) {
    NotSupported("fewer than two GPUs");
    return;
  }
  if (gpus.size() > 4) gpus.resize(4);

  constexpr size_t kStride = 2 * kPage;
  constexpr uint32_t kEntries = 4;
  constexpr size_t kEntrySize = 5000;
  unsigned pairs_run = 0;

  for (const hsa_agent_t a : gpus) {
    for (const hsa_agent_t b : gpus) {
      if (a.handle == b.handle) continue;

      hsa_amd_memory_pool_t pool_a = {0}, pool_b = {0};
      hsa_amd_agent_iterate_memory_pools(a, rocrtst::FindStandardPool, &pool_a);
      hsa_amd_agent_iterate_memory_pools(b, rocrtst::FindStandardPool, &pool_b);
      if (pool_a.handle == 0 || pool_b.handle == 0) continue;
      // The copy engine of a writes into memory of b.
      hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_agent_memory_pool_get_info(
                                        a, pool_b, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access));
      if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) continue;

      Resources res;
      const size_t total = kEntries * kStride;
      uint8_t* h_src = static_cast<uint8_t*>(AllocHost(res, c, total));
      uint8_t* h_dst = static_cast<uint8_t*>(AllocHost(res, c, total));
      void* src = nullptr;
      void* dst = nullptr;
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_allocate(pool_a, total, 0, &src));
      res.bufs.push_back(src);
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_memory_pool_allocate(pool_b, total, 0, &dst));
      res.bufs.push_back(dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_agents_allow_access(1, &a, nullptr, dst));
      ASSERT_TRUE(h_src && h_dst);
      // Host staging buffers are read and written by whichever GPU stages them.
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_agents_allow_access(static_cast<uint32_t>(gpus.size()),
                                                                gpus.data(), nullptr, h_src));
      ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_amd_agents_allow_access(static_cast<uint32_t>(gpus.size()),
                                                                gpus.data(), nullptr, h_dst));
      FillPattern(h_src, total, 7);

      const std::string tag = "peer copy";
      ASSERT_TRUE(Stage(res, src, a, h_src, c.cpu, total));

      // Single op.
      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      std::vector<hsa_amd_memory_copy_op_t> ops = {LinearOp(done, dst, b, src, a, total)};
      ASSERT_TRUE(RunBatch(res, ops, done, (tag + " single").c_str()));
      memset(h_dst, kPoison, total);
      ASSERT_TRUE(Stage(res, h_dst, c.cpu, dst, b, total));
      ASSERT_TRUE(SameBytes(h_dst, h_src, total, tag + " single"));

      // Multiple entries.
      memset(h_dst, kPoison, total);
      ASSERT_TRUE(Stage(res, dst, b, h_dst, c.cpu, total));  // poison the destination
      ASSERT_TRUE(NewSignal(res, 1, &done));
      MultiEntry m;
      for (uint32_t e = 0; e < kEntries; ++e) {
        m.Add(static_cast<uint8_t*>(src) + e * kStride, static_cast<uint8_t*>(dst) + e * kStride,
              b, kEntrySize);
      }
      ops = {MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, a, m)};
      ASSERT_TRUE(RunBatch(res, ops, done, (tag + " multi").c_str()));
      memset(h_dst, 0, total);
      ASSERT_TRUE(Stage(res, h_dst, c.cpu, dst, b, total));
      for (uint32_t e = 0; e < kEntries; ++e) {
        ASSERT_TRUE(SameBytes(h_dst + e * kStride, h_src + e * kStride, kEntrySize,
                              tag + " multi entry " + std::to_string(e)));
        ASSERT_TRUE(IsPoison(h_dst + e * kStride + kEntrySize, kStride - kEntrySize))
            << tag << " multi entry " << e << " wrote past its size";
      }
      ++pairs_run;
    }
  }
  if (pairs_run == 0) NotSupported("no GPU pair can access each other's memory");
}

// -----------------------------------------------------------------------------
// BROADCAST
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::BroadcastTest() {
  Banner("broadcast");
  const Ctx c = MakeCtx(this);
  if (!c.broadcast) {
    NotSupported("the agent reports no broadcast destinations");
    return;
  }

  // Sizes straddle the broadcast, linear-B2B and fan-out thresholds. Odd
  // destination counts take the trailing single-copy path.
  const size_t sizes[] = {4, 4096, 32768, 262144, 1 << 20};
  const uint32_t counts[] = {2, 3, 5};

  for (const size_t size : sizes) {
    for (const uint32_t n : counts) {
      Resources res;
      const size_t stride = (size + kPage - 1) & ~(kPage - 1);
      const size_t total = n * stride;
      uint8_t* h_src = static_cast<uint8_t*>(AllocHost(res, c, size));
      uint8_t* h_dst = static_cast<uint8_t*>(AllocHost(res, c, total));
      void* dev_src = AllocDev(res, c, size);
      uint8_t* dev_dst = static_cast<uint8_t*>(AllocDev(res, c, total));
      ASSERT_TRUE(h_src && h_dst && dev_src && dev_dst);
      FillPattern(h_src, size, 3);
      ASSERT_TRUE(Stage(res, dev_src, c.gpu, h_src, c.cpu, size));
      memset(h_dst, kPoison, total);
      ASSERT_TRUE(Stage(res, dev_dst, c.gpu, h_dst, c.cpu, total));

      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      std::vector<void*> dsts;
      std::vector<hsa_agent_t> agents(n, c.gpu);
      for (uint32_t d = 0; d < n; ++d) dsts.push_back(dev_dst + d * stride);
      hsa_amd_memory_copy_op_t op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST, done);
      op.src = dev_src;
      op.src_agent = c.gpu;
      op.dst_list = dsts.data();
      op.dst_agent_list = agents.data();
      op.size = size;
      op.num_entries = static_cast<uint16_t>(n);

      const std::string tag = "size " + std::to_string(size) + " x" + std::to_string(n);
      std::vector<hsa_amd_memory_copy_op_t> ops = {op};
      ASSERT_TRUE(RunBatch(res, ops, done, tag.c_str()));

      memset(h_dst, 0, total);
      ASSERT_TRUE(Stage(res, h_dst, c.cpu, dev_dst, c.gpu, total));
      for (uint32_t d = 0; d < n; ++d) {
        const std::string what = tag + " destination " + std::to_string(d);
        ASSERT_TRUE(SameBytes(h_dst + d * stride, h_src, size, what));
        ASSERT_TRUE(IsPoison(h_dst + d * stride + size, stride - size))
            << what << " was written past the copy size";
      }
    }
  }
}

// -----------------------------------------------------------------------------
// SWAP
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::SwapTest() {
  Banner("swap");
  const Ctx c = MakeCtx(this);
  // No agent attribute reports whether a swap is serviceable, so the operation
  // is attempted rather than skipped on a prediction. A target that cannot
  // serve one rejects the submission and the test reports that it was not
  // supported; c.swap marks only the targets on which a rejection is a defect.
  // Attempting it is what lets a shader-blit routing be covered here at all,
  // since such a routing is reached precisely when SDMA is unavailable.
  bool swap_served = false;

  // Single swap.
  const size_t sizes[] = {64, 4096, 65536, 1 << 20};
  for (const size_t size : sizes) {
    Resources res;
    uint8_t* h_a = static_cast<uint8_t*>(AllocHost(res, c, size));
    uint8_t* h_b = static_cast<uint8_t*>(AllocHost(res, c, size));
    uint8_t* h_out = static_cast<uint8_t*>(AllocHost(res, c, size));
    void* dev_a = AllocDev(res, c, size);
    void* dev_b = AllocDev(res, c, size);
    ASSERT_TRUE(h_a && h_b && h_out && dev_a && dev_b);
    FillPattern(h_a, size, 1);
    FillPattern(h_b, size, 2);
    ASSERT_TRUE(Stage(res, dev_a, c.gpu, h_a, c.cpu, size));
    ASSERT_TRUE(Stage(res, dev_b, c.gpu, h_b, c.cpu, size));

    hsa_signal_t done;
    ASSERT_TRUE(NewSignal(res, 1, &done));
    hsa_amd_memory_copy_op_t op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP, done);
    op.src = dev_a;
    op.src_agent = c.gpu;
    op.dst = dev_b;
    op.dst_agent = c.gpu;
    op.src_size = size;
    op.dst_size = size;

    const std::string tag = "swap size " + std::to_string(size);
    const hsa_status_t st = Submit(res, &op, 1);
    if (st != HSA_STATUS_SUCCESS) {
      EXPECT_FALSE(c.swap) << tag << ": batch copy returned " << StatusStr(st)
                           << " on a target that supports swap";
      NotSupported("swap is not supported on this target");
      return;
    }
    ASSERT_TRUE(WaitDone(res, done, tag.c_str()));
    swap_served = true;

    memset(h_out, 0, size);
    ASSERT_TRUE(Stage(res, h_out, c.cpu, dev_a, c.gpu, size));
    ASSERT_TRUE(SameBytes(h_out, h_b, size, tag + " first buffer"));
    memset(h_out, 0, size);
    ASSERT_TRUE(Stage(res, h_out, c.cpu, dev_b, c.gpu, size));
    ASSERT_TRUE(SameBytes(h_out, h_a, size, tag + " second buffer"));
  }
  // Gated on what the single-entry form was observed to do, not on the
  // prediction, so the multi-entry form is covered on any routing that served
  // the simpler case.
  if (!swap_served) return;

  // Several swaps in one op.
  {
    Resources res;
    constexpr size_t kStride = 16 * kPage;
    const size_t entry_sizes[] = {4096, 8192, 65536};
    constexpr uint32_t n = 3;
    const size_t total = n * kStride;
    uint8_t* h_a = static_cast<uint8_t*>(AllocHost(res, c, total));
    uint8_t* h_b = static_cast<uint8_t*>(AllocHost(res, c, total));
    uint8_t* h_out = static_cast<uint8_t*>(AllocHost(res, c, total));
    uint8_t* dev_a = static_cast<uint8_t*>(AllocDev(res, c, total));
    uint8_t* dev_b = static_cast<uint8_t*>(AllocDev(res, c, total));
    ASSERT_TRUE(h_a && h_b && h_out && dev_a && dev_b);
    FillPattern(h_a, total, 11);
    FillPattern(h_b, total, 12);
    ASSERT_TRUE(Stage(res, dev_a, c.gpu, h_a, c.cpu, total));
    ASSERT_TRUE(Stage(res, dev_b, c.gpu, h_b, c.cpu, total));

    hsa_signal_t done;
    ASSERT_TRUE(NewSignal(res, 1, &done));
    MultiEntry m;
    for (uint32_t e = 0; e < n; ++e) {
      m.Add(dev_a + e * kStride, dev_b + e * kStride, c.gpu, entry_sizes[e]);
    }
    std::vector<hsa_amd_memory_copy_op_t> ops = {
        MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP, done, c.gpu, m)};
    ASSERT_TRUE(RunBatch(res, ops, done, "multi-entry swap"));

    memset(h_out, 0, total);
    ASSERT_TRUE(Stage(res, h_out, c.cpu, dev_a, c.gpu, total));
    for (uint32_t e = 0; e < n; ++e) {
      const size_t off = e * kStride;
      const std::string what = "multi-entry swap entry " + std::to_string(e);
      ASSERT_TRUE(SameBytes(h_out + off, h_b + off, entry_sizes[e], what + " first buffer"));
      // The rest of the slot was not part of the swap.
      ASSERT_TRUE(SameBytes(h_out + off + entry_sizes[e], h_a + off + entry_sizes[e],
                            kStride - entry_sizes[e], what + " tail"));
    }
  }

  // An address that is not aligned for the SDMA swap packet is rejected before
  // anything is queued. The alignment requirement belongs to the packet, not to
  // the operation, so this only holds where the native packet serves the swap.
  // A shader blit addresses memory in 16-byte chunks plus a byte tail and the
  // targets run with unaligned access enabled, so it accepts the same input.
  if (c.swap) {
    Resources res;
    void* dev_a = AllocDev(res, c, 2 * kPage);
    void* dev_b = AllocDev(res, c, 2 * kPage);
    ASSERT_TRUE(dev_a && dev_b);
    hsa_signal_t done;
    ASSERT_TRUE(NewSignal(res, 1, &done));
    hsa_amd_memory_copy_op_t op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP, done);
    op.src = static_cast<uint8_t*>(dev_a) + 4;
    op.src_agent = c.gpu;
    op.dst = dev_b;
    op.dst_agent = c.gpu;
    op.src_size = 64;
    op.dst_size = 64;
    EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr))
        << "misaligned swap source";
    EXPECT_EQ(1, hsa_signal_load_scacquire(done)) << "rejected swap completed its signal";
  }
}

// -----------------------------------------------------------------------------
// RECT
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::RectTest() {
  Banner("rectangular copies");
  const Ctx c = MakeCtx(this);
  if (!c.rect) {
    NotSupported("rect copy needs SDMA on gfx9 or newer");
    return;
  }

  enum Dir { kH2D, kD2H, kD2D };
  const struct {
    Dir dir;
    const char* name;
  } dirs[] = {{kH2D, "H2D"}, {kD2H, "D2H"}, {kD2D, "D2D"}};

  for (const bool two_boxes : {false, true}) {
    for (const auto& d : dirs) {
      Resources res;
      const std::string tag =
          std::string(d.name) + (two_boxes ? " two boxes" : " one box");

      uint8_t* h_src = static_cast<uint8_t*>(AllocHost(res, c, RectJob::kSrcBytes));
      uint8_t* h_dst = static_cast<uint8_t*>(AllocHost(res, c, RectJob::kDstBytes));
      void* dev_src = AllocDev(res, c, RectJob::kSrcBytes);
      void* dev_dst = AllocDev(res, c, RectJob::kDstBytes);
      ASSERT_TRUE(h_src && h_dst && dev_src && dev_dst);
      FillPattern(h_src, RectJob::kSrcBytes, 5);
      std::vector<uint8_t> poison(RectJob::kDstBytes, kPoison);

      void* src_base = (d.dir == kH2D) ? static_cast<void*>(h_src) : dev_src;
      void* dst_base = (d.dir == kD2H) ? static_cast<void*>(h_dst) : dev_dst;
      const hsa_agent_t src_agent = (d.dir == kH2D) ? c.cpu : c.gpu;
      const hsa_agent_t dst_agent = (d.dir == kD2H) ? c.cpu : c.gpu;

      if (d.dir != kH2D) {
        ASSERT_TRUE(Stage(res, dev_src, c.gpu, h_src, c.cpu, RectJob::kSrcBytes));
      }
      memcpy(h_dst, poison.data(), RectJob::kDstBytes);
      if (d.dir != kD2H) {
        ASSERT_TRUE(Stage(res, dev_dst, c.gpu, h_dst, c.cpu, RectJob::kDstBytes));
      }

      RectJob job;
      job.Build(src_base, dst_base, dst_agent, h_src, poison.data(), two_boxes);

      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      hsa_amd_memory_copy_op_t op = job.Op(done, src_agent);
      const hsa_status_t st = Submit(res, &op, 1);
      if (st != HSA_STATUS_SUCCESS) {
        ADD_FAILURE() << tag << ": batch copy returned " << StatusStr(st)
                      << " on a target that supports rect copies";
        return;
      }
      ASSERT_TRUE(WaitDone(res, done, tag.c_str()));

      if (d.dir != kD2H) {
        memset(h_dst, 0, RectJob::kDstBytes);
        ASSERT_TRUE(Stage(res, h_dst, c.cpu, dev_dst, c.gpu, RectJob::kDstBytes));
      }
      ASSERT_TRUE(SameBytes(h_dst, job.expected.data(), RectJob::kDstBytes, tag));
    }
  }
}

// -----------------------------------------------------------------------------
// INDIRECT
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::IndirectTest() {
  Banner("indirect copies");
  const Ctx c = MakeCtx(this);
  if (!c.sdma_ops) {
    NotSupported("indirect copy needs SDMA (HSA_ENABLE_SDMA=0 or a full-profile agent)");
    return;
  }

  const hsa_amd_memory_copy_op_type_t types[] = {HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC,
                                                 HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST,
                                                 HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST};
  const char* names[] = {"INDIRECT_SRC", "INDIRECT_DST", "INDIRECT_SRCDST"};
  constexpr size_t kSize = 4096;
  constexpr uint32_t kEntries = 3;

  for (int multi = 0; multi < 2; ++multi) {
    for (size_t t = 0; t < ARRAY_SIZE(types); ++t) {
      Resources res;
      const bool ind_src = types[t] != HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST;
      const bool ind_dst = types[t] != HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC;
      const uint32_t n = multi ? kEntries : 1;
      const std::string tag = std::string(names[t]) + (multi ? " multi" : " single");

      uint8_t* h_src = static_cast<uint8_t*>(AllocHost(res, c, n * kSize));
      uint8_t* h_dst = static_cast<uint8_t*>(AllocHost(res, c, n * kSize));
      uint8_t* dev_src = static_cast<uint8_t*>(AllocDev(res, c, n * kSize));
      uint8_t* dev_dst = static_cast<uint8_t*>(AllocDev(res, c, n * kSize));
      // The slots hold the addresses the SDMA engine reads at execution time.
      void** src_slots = static_cast<void**>(AllocHost(res, c, n * sizeof(void*)));
      void** dst_slots = static_cast<void**>(AllocHost(res, c, n * sizeof(void*)));
      ASSERT_TRUE(h_src && h_dst && dev_src && dev_dst && src_slots && dst_slots);
      FillPattern(h_src, n * kSize, 9);
      memset(h_dst, kPoison, n * kSize);
      ASSERT_TRUE(Stage(res, dev_src, c.gpu, h_src, c.cpu, n * kSize));
      ASSERT_TRUE(Stage(res, dev_dst, c.gpu, h_dst, c.cpu, n * kSize));
      for (uint32_t e = 0; e < n; ++e) {
        src_slots[e] = dev_src + e * kSize;
        dst_slots[e] = dev_dst + e * kSize;
      }

      hsa_signal_t done;
      ASSERT_TRUE(NewSignal(res, 1, &done));
      hsa_amd_memory_copy_op_t op = NewOp(types[t], done);
      op.src_agent = c.gpu;
      op.wait.scope = HSA_FENCE_SCOPE_SYSTEM;  // scope of the address reads
      MultiEntry m;
      if (multi) {
        for (uint32_t e = 0; e < n; ++e) {
          m.Add(ind_src ? static_cast<void*>(&src_slots[e]) : static_cast<void*>(dev_src + e * kSize),
                ind_dst ? static_cast<void*>(&dst_slots[e]) : static_cast<void*>(dev_dst + e * kSize),
                c.gpu, kSize);
        }
        op.src_list = m.src.data();
        op.dst_list = m.dst.data();
        op.dst_agent_list = m.dst_agents.data();
        op.size_list = m.sizes.data();
        op.num_entries = static_cast<uint16_t>(n);
      } else {
        op.src = ind_src ? static_cast<void*>(&src_slots[0]) : static_cast<void*>(dev_src);
        op.dst = ind_dst ? static_cast<void*>(&dst_slots[0]) : static_cast<void*>(dev_dst);
        op.dst_agent = c.gpu;
        op.size = kSize;
      }

      const hsa_status_t st = Submit(res, &op, 1);
      if (st != HSA_STATUS_SUCCESS) {
        EXPECT_FALSE(c.indirect) << tag << ": batch copy returned " << StatusStr(st)
                                 << " on a target that supports indirect copies";
        NotSupported("indirect copy is not supported on this target");
        return;
      }
      ASSERT_TRUE(WaitDone(res, done, tag.c_str()));

      memset(h_dst, 0, n * kSize);
      ASSERT_TRUE(Stage(res, h_dst, c.cpu, dev_dst, c.gpu, n * kSize));
      ASSERT_TRUE(SameBytes(h_dst, h_src, n * kSize, tag));
    }
  }
}

// -----------------------------------------------------------------------------
// Dependent signals
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::DependencyTest() {
  Banner("dependent signals");
  const Ctx c = MakeCtx(this);

  constexpr size_t kSize = 1 << 20;
  constexpr uint32_t kOps = 2;
  using std::chrono::milliseconds;
  constexpr int kGateHoldMs = 150;
  constexpr int kFirstCheckMs = 75;

  for (const uint32_t num_deps : {1u, 2u}) {
    Resources res;
    const std::string tag = std::to_string(num_deps) + " dependent signal(s)";

    hsa_signal_t gate[2];
    ASSERT_TRUE(NewSignal(res, 1, &gate[0]));
    ASSERT_TRUE(NewSignal(res, 1, &gate[1]));

    std::vector<uint8_t*> h_src(kOps), h_dst(kOps);
    std::vector<void*> dev(kOps);
    for (uint32_t i = 0; i < kOps; ++i) {
      h_src[i] = static_cast<uint8_t*>(AllocHost(res, c, kSize));
      h_dst[i] = static_cast<uint8_t*>(AllocHost(res, c, kSize));
      dev[i] = AllocDev(res, c, kSize);
      ASSERT_TRUE(h_src[i] && h_dst[i] && dev[i]);
      FillPattern(h_src[i], kSize, i + 20);
      ASSERT_TRUE(Stage(res, dev[i], c.gpu, h_src[i], c.cpu, kSize));
      memset(h_dst[i], kPoison, kSize);
    }

    hsa_signal_t done;
    ASSERT_TRUE(NewSignal(res, kOps, &done));
    std::vector<hsa_amd_memory_copy_op_t> ops;
    for (uint32_t i = 0; i < kOps; ++i) {
      ops.push_back(LinearOp(done, h_dst[i], c.cpu, dev[i], c.gpu, kSize));
    }

    // The gates open one after the other. released[k] is set before gate k is
    // opened, so a check that reads "not released" afterwards knows the gate was
    // still closed while it looked.
    std::atomic<bool> released[2];
    released[0] = false;
    released[1] = false;
    const auto t0 = std::chrono::steady_clock::now();
    std::thread opener([&]() {
      for (uint32_t k = 0; k < num_deps; ++k) {
        std::this_thread::sleep_until(t0 + milliseconds(kGateHoldMs * (k + 1)));
        released[k] = true;
        hsa_signal_store_screlease(gate[k], 0);
      }
    });
    MAKE_SCOPE_GUARD([&]() {
      if (opener.joinable()) opener.join();
    });

    const hsa_status_t st = Submit(res, ops.data(), kOps, num_deps, gate);
    ASSERT_EQ(HSA_STATUS_SUCCESS, st) << tag << ": " << StatusStr(st);

    // Without SDMA the runtime waits for the dependencies inside the call, so
    // the checks below only apply when the gates were still closed.
    for (uint32_t k = 0; k < num_deps; ++k) {
      std::this_thread::sleep_until(t0 + milliseconds(kFirstCheckMs + kGateHoldMs * k));
      const hsa_signal_value_t v = hsa_signal_load_scacquire(done);
      bool untouched = true;
      for (uint32_t i = 0; i < kOps; ++i) untouched = untouched && IsPoison(h_dst[i], kSize);
      if (!released[num_deps - 1].load()) {
        EXPECT_EQ(static_cast<hsa_signal_value_t>(kOps), v)
            << tag << ": batch completed while a dependency was still set";
        EXPECT_TRUE(untouched) << tag << ": a copy wrote its destination before the dependency "
                               << "was released";
      }
    }

    ASSERT_TRUE(WaitDone(res, done, tag.c_str()));
    for (uint32_t i = 0; i < kOps; ++i) {
      ASSERT_TRUE(SameBytes(h_dst[i], h_src[i], kSize, tag + " op " + std::to_string(i)));
    }
  }
}

// -----------------------------------------------------------------------------
// Mixed operation types in one call
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::MixedOpsTest() {
  Banner("mixed operation types in one batch");
  const Ctx c = MakeCtx(this);

  Resources res;

  // 0: single linear H2D.
  constexpr size_t kLinSize = 5000;
  uint8_t* h_lin_src = static_cast<uint8_t*>(AllocHost(res, c, kLinSize));
  uint8_t* h_lin_dst = static_cast<uint8_t*>(AllocHost(res, c, kLinSize));
  void* dev_lin = AllocDev(res, c, kLinSize);
  ASSERT_TRUE(h_lin_src && h_lin_dst && dev_lin);
  FillPattern(h_lin_src, kLinSize, 31);

  // 1: multi-entry linear D2D.
  constexpr size_t kStride = 18 * kPage;
  const size_t multi_sizes[] = {4096, 100, 70000};
  constexpr uint32_t kMulti = 3;
  uint8_t* h_multi_src = static_cast<uint8_t*>(AllocHost(res, c, kMulti * kStride));
  uint8_t* h_multi_dst = static_cast<uint8_t*>(AllocHost(res, c, kMulti * kStride));
  uint8_t* dev_multi_a = static_cast<uint8_t*>(AllocDev(res, c, kMulti * kStride));
  uint8_t* dev_multi_b = static_cast<uint8_t*>(AllocDev(res, c, kMulti * kStride));
  ASSERT_TRUE(h_multi_src && h_multi_dst && dev_multi_a && dev_multi_b);
  FillPattern(h_multi_src, kMulti * kStride, 32);
  ASSERT_TRUE(Stage(res, dev_multi_a, c.gpu, h_multi_src, c.cpu, kMulti * kStride));
  memset(h_multi_dst, kPoison, kMulti * kStride);
  ASSERT_TRUE(Stage(res, dev_multi_b, c.gpu, h_multi_dst, c.cpu, kMulti * kStride));

  // 2: broadcast.
  constexpr size_t kBcastSize = 20000;
  constexpr size_t kBcastStride = 5 * kPage;
  constexpr uint32_t kBcast = 3;
  uint8_t* h_bcast_src = static_cast<uint8_t*>(AllocHost(res, c, kBcastSize));
  uint8_t* h_bcast_dst = static_cast<uint8_t*>(AllocHost(res, c, kBcast * kBcastStride));
  void* dev_bcast_src = AllocDev(res, c, kBcastSize);
  uint8_t* dev_bcast_dst = static_cast<uint8_t*>(AllocDev(res, c, kBcast * kBcastStride));
  ASSERT_TRUE(h_bcast_src && h_bcast_dst && dev_bcast_src && dev_bcast_dst);
  FillPattern(h_bcast_src, kBcastSize, 33);
  ASSERT_TRUE(Stage(res, dev_bcast_src, c.gpu, h_bcast_src, c.cpu, kBcastSize));

  // 3: rect H2D.
  uint8_t* h_rect_src = static_cast<uint8_t*>(AllocHost(res, c, RectJob::kSrcBytes));
  uint8_t* h_rect_dst = static_cast<uint8_t*>(AllocHost(res, c, RectJob::kDstBytes));
  void* dev_rect_dst = AllocDev(res, c, RectJob::kDstBytes);
  ASSERT_TRUE(h_rect_src && h_rect_dst && dev_rect_dst);
  FillPattern(h_rect_src, RectJob::kSrcBytes, 34);
  std::vector<uint8_t> rect_poison(RectJob::kDstBytes, kPoison);
  memcpy(h_rect_dst, rect_poison.data(), RectJob::kDstBytes);
  ASSERT_TRUE(Stage(res, dev_rect_dst, c.gpu, h_rect_dst, c.cpu, RectJob::kDstBytes));

  const bool with_bcast = c.broadcast;
  const bool with_rect = c.rect;
  const uint32_t num_ops = 2 + (with_bcast ? 1 : 0) + (with_rect ? 1 : 0);

  hsa_signal_t done;
  ASSERT_TRUE(NewSignal(res, num_ops, &done));

  std::vector<hsa_amd_memory_copy_op_t> ops;
  ops.push_back(LinearOp(done, dev_lin, c.gpu, h_lin_src, c.cpu, kLinSize));

  MultiEntry multi;
  for (uint32_t e = 0; e < kMulti; ++e) {
    multi.Add(dev_multi_a + e * kStride, dev_multi_b + e * kStride, c.gpu, multi_sizes[e]);
  }
  ops.push_back(MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.gpu, multi));

  std::vector<void*> bcast_dsts;
  std::vector<hsa_agent_t> bcast_agents(kBcast, c.gpu);
  for (uint32_t d = 0; d < kBcast; ++d) bcast_dsts.push_back(dev_bcast_dst + d * kBcastStride);
  if (with_bcast) {
    hsa_amd_memory_copy_op_t op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST, done);
    op.src = dev_bcast_src;
    op.src_agent = c.gpu;
    op.dst_list = bcast_dsts.data();
    op.dst_agent_list = bcast_agents.data();
    op.size = kBcastSize;
    op.num_entries = kBcast;
    ops.push_back(op);
  }

  RectJob rect;
  if (with_rect) {
    rect.Build(h_rect_src, dev_rect_dst, c.gpu, h_rect_src, rect_poison.data(), true);
    ops.push_back(rect.Op(done, c.cpu));
  }

  ASSERT_TRUE(RunBatch(res, ops, done, "mixed batch"));

  ASSERT_TRUE(Stage(res, h_lin_dst, c.cpu, dev_lin, c.gpu, kLinSize));
  ASSERT_TRUE(SameBytes(h_lin_dst, h_lin_src, kLinSize, "mixed: linear"));

  memset(h_multi_dst, 0, kMulti * kStride);
  ASSERT_TRUE(Stage(res, h_multi_dst, c.cpu, dev_multi_b, c.gpu, kMulti * kStride));
  for (uint32_t e = 0; e < kMulti; ++e) {
    const std::string what = "mixed: multi-entry linear entry " + std::to_string(e);
    ASSERT_TRUE(SameBytes(h_multi_dst + e * kStride, h_multi_src + e * kStride, multi_sizes[e], what));
    ASSERT_TRUE(IsPoison(h_multi_dst + e * kStride + multi_sizes[e], kStride - multi_sizes[e]))
        << what << " wrote past its size";
  }

  if (with_bcast) {
    memset(h_bcast_dst, 0, kBcast * kBcastStride);
    ASSERT_TRUE(Stage(res, h_bcast_dst, c.cpu, dev_bcast_dst, c.gpu, kBcast * kBcastStride));
    for (uint32_t d = 0; d < kBcast; ++d) {
      ASSERT_TRUE(SameBytes(h_bcast_dst + d * kBcastStride, h_bcast_src, kBcastSize,
                            "mixed: broadcast destination " + std::to_string(d)));
    }
  } else {
    NotSupported("broadcast not part of this batch (the agent reports no destinations)");
  }

  if (with_rect) {
    memset(h_rect_dst, 0, RectJob::kDstBytes);
    ASSERT_TRUE(Stage(res, h_rect_dst, c.cpu, dev_rect_dst, c.gpu, RectJob::kDstBytes));
    ASSERT_TRUE(SameBytes(h_rect_dst, rect.expected.data(), RectJob::kDstBytes, "mixed: rect"));
  } else {
    NotSupported("rect not part of this batch (needs SDMA on gfx9 or newer)");
  }
}

// -----------------------------------------------------------------------------
// Invalid arguments
// -----------------------------------------------------------------------------

void MemoryAsyncBatchCopy::NegativeTest() {
  Banner("invalid arguments");
  const Ctx c = MakeCtx(this);

  Resources res;
  constexpr size_t kSize = kPage;
  uint8_t* host = static_cast<uint8_t*>(AllocHost(res, c, kSize));
  uint8_t* host2 = static_cast<uint8_t*>(AllocHost(res, c, kSize));
  void* dev = AllocDev(res, c, kSize);
  void* dev2 = AllocDev(res, c, kSize);
  ASSERT_TRUE(host && host2 && dev && dev2);
  FillPattern(host, kSize, 41);
  memset(host2, kPoison, kSize);
  ASSERT_TRUE(Stage(res, dev2, c.gpu, host2, c.cpu, kSize));

  hsa_signal_t done, other_done;
  ASSERT_TRUE(NewSignal(res, 1, &done));
  ASSERT_TRUE(NewSignal(res, 1, &other_done));
  const hsa_signal_t bad_signal = {0};
  const hsa_agent_t bad_agent = {0};

  auto valid = [&]() { return LinearOp(done, dev, c.gpu, host, c.cpu, kSize); };
  auto expect = [&](const hsa_amd_memory_copy_op_t& op, hsa_status_t want, const char* what) {
    EXPECT_EQ(want, hsa_amd_memory_async_batch_copy(&op, 1, 0, nullptr)) << what;
  };

  // The batch itself.
  hsa_amd_memory_copy_op_t op = valid();
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_memory_async_batch_copy(nullptr, 1, 0, nullptr))
      << "null copy_ops";
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_memory_async_batch_copy(&op, 0, 0, nullptr))
      << "zero ops";
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_memory_async_batch_copy(&op, 1, 1, nullptr))
      << "dependency count without signals";
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_memory_async_batch_copy(&op, 1, 0, &other_done))
      << "dependency signals without a count";
  const hsa_signal_t invalid_dep[1] = {bad_signal};
  // A null signal handle is rejected by the handle conversion itself.
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, hsa_amd_memory_async_batch_copy(&op, 1, 1, invalid_dep))
      << "invalid dependency signal";

  // Common fields.
  op = valid();
  op.version = 0;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "version 0");
  op = valid();
  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION + 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unknown version");
  op = valid();
  op.type = HSA_AMD_MEMORY_COPY_OP_RECT + 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unknown type");
  op = valid();
  op.completion_signal = bad_signal;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "invalid completion signal");
  op = valid();
  op.src = nullptr;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "null source");
  op = valid();
  op.dst = nullptr;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "null destination");
  op = valid();
  op.src_agent = bad_agent;
  expect(op, HSA_STATUS_ERROR_INVALID_AGENT, "invalid source agent");
  op = valid();
  op.dst_agent = bad_agent;
  expect(op, HSA_STATUS_ERROR_INVALID_AGENT, "invalid destination agent");
  op = valid();
  op.reserved1[0] = 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "reserved1 set");
  op = valid();
  op.unused_size = 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unused_size set on a linear copy");
  expect(LinearOp(done, host2, c.cpu, host, c.cpu, kSize), HSA_STATUS_ERROR_INVALID_AGENT,
         "host to host");

  // Wait-before-copy descriptor.
  op = valid();
  op.wait.function = HSA_AMD_MEMORY_COPY_WAIT_GT + 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unknown wait function");
  op = valid();
  op.wait.function = HSA_AMD_MEMORY_COPY_WAIT_EQ;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "wait without an address");
  op = valid();
  op.wait.addr = host;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "wait address with no wait function");
  op = valid();
  op.wait.reserved = 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "wait.reserved set");
  op = valid();
  op.wait.function = HSA_AMD_MEMORY_COPY_WAIT_EQ;
  op.wait.addr = host;
  op.wait.scope = HSA_FENCE_SCOPE_SYSTEM + 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unknown wait scope");
  op = valid();
  op.wait.scope = HSA_FENCE_SCOPE_SYSTEM;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "wait scope set with no wait function");

  // Signal-after-copy descriptor.
  op = valid();
  op.signal.operation = HSA_AMD_MEMORY_COPY_SIGNAL_SUB + 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unknown signal operation");
  op = valid();
  op.signal.operation = HSA_AMD_MEMORY_COPY_SIGNAL_WRITE;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "signal operation without an address");
  op = valid();
  op.signal.addr = host;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "signal address with no operation");
  op = valid();
  op.signal.data = 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "signal data with no operation");
  op = valid();
  op.signal.reserved = 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "signal.reserved set");
  op = valid();
  op.signal.operation = HSA_AMD_MEMORY_COPY_SIGNAL_WRITE;
  op.signal.addr = host;
  op.signal.scope = HSA_FENCE_SCOPE_SYSTEM + 1;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "unknown signal scope");

  // Multi-entry linear.
  {
    MultiEntry m;
    m.Add(host, dev, c.gpu, kSize);
    op = MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, m);
    op.dst_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear without dst_list");
    op = MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, m);
    op.dst_agent_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear without dst_agent_list");
    op = MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, m);
    op.size_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear without size_list");
    op = MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, m);
    op.reserved0 = 1;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear with reserved0 set");

    MultiEntry zero;
    zero.Add(host, dev, c.gpu, 0);
    expect(MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, zero),
           HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear entry of size 0");

    MultiEntry null_src;
    null_src.Add(nullptr, dev, c.gpu, kSize);
    // src_list is a non-null array, but its first element is null.
    expect(MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, null_src),
           HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear null source entry");

    MultiEntry null_dst;
    null_dst.Add(host, nullptr, c.gpu, kSize);
    expect(MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, null_dst),
           HSA_STATUS_ERROR_INVALID_ARGUMENT, "multi linear null destination entry");

    MultiEntry bad_dst_agent;
    bad_dst_agent.Add(host, dev, bad_agent, kSize);
    expect(MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, bad_dst_agent),
           HSA_STATUS_ERROR_INVALID_AGENT, "multi linear invalid destination agent");

    MultiEntry host_to_host;
    host_to_host.Add(host, host2, c.cpu, kSize);
    expect(MultiOp(HSA_AMD_MEMORY_COPY_OP_LINEAR, done, c.cpu, host_to_host),
           HSA_STATUS_ERROR_INVALID_AGENT, "multi linear host to host");
  }

  // Broadcast.
  {
    std::vector<void*> dsts = {dev2};
    std::vector<hsa_agent_t> agents = {c.gpu};
    auto bcast = [&]() {
      hsa_amd_memory_copy_op_t b = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST, done);
      b.src = dev;
      b.src_agent = c.gpu;
      b.dst_list = dsts.data();
      b.dst_agent_list = agents.data();
      b.size = kSize;
      b.num_entries = 1;
      return b;
    };
    op = bcast();
    op.num_entries = 0;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "broadcast with no entries");
    op = bcast();
    op.dst_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "broadcast without dst_list");
    op = bcast();
    op.dst_agent_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "broadcast without dst_agent_list");
    op = bcast();
    op.unused_size = 1;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "broadcast with unused_size set");
    op = bcast();
    op.src_agent = c.cpu;
    expect(op, HSA_STATUS_ERROR_INVALID_AGENT, "broadcast from a host agent");
    std::vector<void*> null_dsts = {nullptr};
    op = bcast();
    op.dst_list = null_dsts.data();
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "broadcast to a null destination");
  }

  // Swap.
  op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP, done);
  op.src = dev;
  op.src_agent = c.gpu;
  op.dst = dev2;
  op.dst_agent = c.gpu;
  op.src_size = 0;
  op.dst_size = kSize;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "swap with source size 0");
  op.src_size = kSize;
  op.dst_size = 0;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "swap with destination size 0");

  // Indirect.
  op = NewOp(HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC, done);
  op.src = dev;
  op.src_agent = c.gpu;
  op.dst = dev2;
  op.dst_agent = c.gpu;
  op.size = 0;
  expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "indirect with size 0");

  // Rect.
  {
    RectJob job;
    std::vector<uint8_t> src_img(RectJob::kSrcBytes, 1), dst_img(RectJob::kDstBytes, 2);
    job.Build(host, dev, c.gpu, src_img.data(), dst_img.data(), false);
    op = job.Op(done, c.cpu);
    op.num_entries = 0;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "rect with no boxes");
    for (int axis = 0; axis < 3; ++axis) {
      RectJob zero;
      zero.Build(host, dev, c.gpu, src_img.data(), dst_img.data(), false);
      hsa_dim3_t& r = zero.range[0];
      (axis == 0 ? r.x : axis == 1 ? r.y : r.z) = 0;
      expect(zero.Op(done, c.cpu), HSA_STATUS_ERROR_INVALID_ARGUMENT, "rect with an empty axis");
    }
    op = job.Op(done, c.cpu);
    op.range_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "rect without range_list");
    op = job.Op(done, c.cpu);
    op.dst_agent_list = nullptr;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "rect without dst_agent_list");
    op = job.Op(done, c.cpu);
    op.reserved0 = 1;
    expect(op, HSA_STATUS_ERROR_INVALID_ARGUMENT, "rect with reserved0 set");

    RectJob host_to_host;
    host_to_host.Build(host, host2, c.cpu, src_img.data(), dst_img.data(), false);
    expect(host_to_host.Op(done, c.cpu), HSA_STATUS_ERROR_INVALID_AGENT, "rect host to host");
  }

  // A batch is validated as a whole: a valid first op is not started when a
  // later one is rejected.
  {
    std::vector<hsa_amd_memory_copy_op_t> ops;
    ops.push_back(LinearOp(done, dev2, c.gpu, host, c.cpu, kSize));
    hsa_amd_memory_copy_op_t bad = valid();
    bad.version = 0;
    ops.push_back(bad);
    EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT,
              hsa_amd_memory_async_batch_copy(ops.data(), 2, 0, nullptr))
        << "second op invalid";
  }

  // None of the rejected calls may have copied or signalled anything.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(1, hsa_signal_load_scacquire(done)) << "a rejected batch decremented its signal";
  memset(host2, 0, kSize);
  ASSERT_TRUE(Stage(res, host2, c.cpu, dev2, c.gpu, kSize));
  EXPECT_TRUE(IsPoison(host2, kSize)) << "a rejected batch copied data";
}
