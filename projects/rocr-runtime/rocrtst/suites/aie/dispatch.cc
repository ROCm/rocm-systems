/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include <elf.h>

#include "gtest/gtest.h"

#include "aie_test_env.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

// Only the on-disk struct layout is needed (to locate and corrupt the `kind` field for
// AieKindIsValidated below); pure header, no core/inc source file is linked into this binary.
#include "core/inc/amd_aie_section.h"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace {

using namespace aie_test;  // NOLINT -- test translation unit

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Memory pool discovery
// ---------------------------------------------------------------------------

// Parameters and result for find_memory_pool: describes what kind of pool to
// look for and receives the first matching pool handle.
struct find_pool_data {
  hsa_amd_memory_pool_global_flag_t expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  bool expected_allocatable = true;
  hsa_amd_memory_pool_t pool{};
};

// HSA memory-pool iteration callback: stops at the first global pool that
// matches the flags and allocatability recorded in the find_pool_data* stored
// in `data`, storing the result there and returning HSA_STATUS_INFO_BREAK.
//
// Note this keys on RUNTIME_ALLOC_REC_GRANULE where memory.cc's equivalent keys on
// RUNTIME_ALLOC_GRANULE. The two very likely select the same pool, but that has not been
// established, so the two files keep their own predicate rather than sharing one.
hsa_status_t find_memory_pool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment{};
  auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (s != HSA_STATUS_SUCCESS) {
    return s;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_global_flag_t flags{};
  if (auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
      s != HSA_STATUS_SUCCESS) {
    return s;
  }

  auto& d = *static_cast<find_pool_data*>(data);
  if ((flags & d.expected_flags) == 0) {
    return HSA_STATUS_SUCCESS;
  }

  std::size_t alloc_rec_granule = 0;
  if (auto s = hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &alloc_rec_granule);
      s != HSA_STATUS_SUCCESS) {
    return s;
  }

  const bool allocatable = (alloc_rec_granule != 0);
  if (d.expected_allocatable != allocatable) {
    return HSA_STATUS_SUCCESS;
  }

  d.pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

// ---------------------------------------------------------------------------
// Binary loader
// ---------------------------------------------------------------------------

// Open `path` for binary reading and report its size. On success the returned
// stream is positioned at the start; on failure it is in a failed state, so the
// caller can test it with `operator bool`.
std::ifstream open_binary(const std::filesystem::path& path, std::size_t* size_out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (f) {
    *size_out = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
  }
  return f;
}

// Read exactly `size` bytes from `f` into `dst`. Returns false on short read.
bool read_exact(std::ifstream& f, void* dst, std::size_t size) {
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
  return static_cast<std::size_t>(f.gcount()) == size;
}

testing::AssertionResult load_binary(hsa_amd_memory_pool_t pool, const std::filesystem::path& path,
                                     void** buf, std::size_t& size_out) {
  std::size_t size = 0;
  auto f = open_binary(path, &size);
  if (!f) {
    return testing::AssertionFailure() << "failed to open '" << path << "'";
  }

  if (hsa_amd_memory_pool_allocate(pool, size, 0, buf) != HSA_STATUS_SUCCESS) {
    return testing::AssertionFailure()
        << "failed to allocate " << size << " bytes for '" << path << "'";
  }

  if (!read_exact(f, *buf, size)) {
    hsa_amd_memory_pool_free(*buf);
    *buf = nullptr;
    return testing::AssertionFailure()
        << "short read loading '" << path << "' (expected " << size << " bytes)";
  }
  size_out = size;
  return testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Virtual memory (vmem) allocation
// ---------------------------------------------------------------------------

// Owns a vmem allocation: the physical handle, the reserved virtual address, and
// the mapped size. All three are needed to fully release the buffer via vmem_free.
struct vmem_buffer {
  hsa_amd_vmem_alloc_handle_t handle{};
  void* va = nullptr;
  std::size_t size = 0;
};

// Allocate a buffer through the vmem API and make it accessible to every agent
// in `agents`. The allocation is created from `pool` (a coarse-grained,
// allocatable global pool), reserved, mapped, and granted RW access.
testing::AssertionResult vmem_allocate(hsa_amd_memory_pool_t pool, std::size_t size,
                                       const std::vector<hsa_agent_t>& agents, vmem_buffer* out) {
  // The vmem API requires the allocation size to be a multiple of the pool's
  // allocation granule (page size); unlike hsa_amd_memory_pool_allocate it does
  // not round up internally. Round the request up to the granule.
  std::size_t granule = 0;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                   &granule) != HSA_STATUS_SUCCESS ||
      granule == 0) {
    return testing::AssertionFailure() << "failed to query pool allocation granule";
  }
  size = ((size + granule - 1) / granule) * granule;

  vmem_buffer buf{};
  buf.size = size;

  if (hsa_amd_vmem_handle_create(pool, size, MEMORY_TYPE_PINNED, 0, &buf.handle) !=
      HSA_STATUS_SUCCESS) {
    return testing::AssertionFailure()
        << "hsa_amd_vmem_handle_create failed for " << size << " bytes";
  }
  if (hsa_amd_vmem_address_reserve_align(&buf.va, size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER) !=
      HSA_STATUS_SUCCESS) {
    hsa_amd_vmem_handle_release(buf.handle);
    return testing::AssertionFailure()
        << "hsa_amd_vmem_address_reserve_align failed for " << size << " bytes";
  }
  if (hsa_amd_vmem_map(buf.va, size, 0, buf.handle, 0) != HSA_STATUS_SUCCESS) {
    hsa_amd_vmem_address_free(buf.va, size);
    hsa_amd_vmem_handle_release(buf.handle);
    return testing::AssertionFailure() << "hsa_amd_vmem_map failed for " << size << " bytes";
  }

  std::vector<hsa_amd_memory_access_desc_t> desc;
  desc.reserve(agents.size());
  for (const auto& agent : agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  if (hsa_amd_vmem_set_access(buf.va, size, desc.data(), desc.size()) != HSA_STATUS_SUCCESS) {
    hsa_amd_vmem_unmap(buf.va, size);
    hsa_amd_vmem_address_free(buf.va, size);
    hsa_amd_vmem_handle_release(buf.handle);
    return testing::AssertionFailure()
        << "hsa_amd_vmem_set_access failed for " << agents.size() << " agents";
  }

  *out = buf;
  return testing::AssertionSuccess();
}

// Release a vmem buffer: unmap, free the address range, release the handle.
// All steps are attempted even if an earlier one fails, so a single buffer
// cannot leak the rest of its resources.
testing::AssertionResult vmem_free(const vmem_buffer& buf) {
  std::vector<const char*> failures;
  if (hsa_amd_vmem_unmap(buf.va, buf.size) != HSA_STATUS_SUCCESS) {
    failures.push_back("hsa_amd_vmem_unmap");
  }
  if (hsa_amd_vmem_address_free(buf.va, buf.size) != HSA_STATUS_SUCCESS) {
    failures.push_back("hsa_amd_vmem_address_free");
  }
  if (hsa_amd_vmem_handle_release(buf.handle) != HSA_STATUS_SUCCESS) {
    failures.push_back("hsa_amd_vmem_handle_release");
  }
  if (failures.empty()) {
    return testing::AssertionSuccess();
  }

  auto result = testing::AssertionFailure() << "vmem free failed:";
  for (const auto* f : failures) {
    result << ' ' << f;
  }
  return result;
}

#if 0
// Load a file into a freshly vmem-allocated buffer accessible to `agents`.
// Disabled: PDI/instructions must live in the dev heap, which is incompatible
// with the vmem reserve+map path (see docs/bug-vmem-map-dev-heap.md). Kept here,
// guarded out, so the intended vmem code path is preserved for when the
// runtime/driver gains dev-heap vmem support.
testing::AssertionResult load_binary_vmem(hsa_amd_memory_pool_t pool,
                                          const std::vector<hsa_agent_t>& agents,
                                          const std::filesystem::path& path, vmem_buffer* out) {
  std::size_t size = 0;
  auto f = open_binary(path, &size);
  if (!f) {
    return testing::AssertionFailure() << "failed to open '" << path << "'";
  }

  if (auto r = vmem_allocate(pool, size, agents, out); !r) {
    return testing::AssertionFailure() << "loading '" << path << "': " << r.message();
  }

  if (!read_exact(f, out->va, size)) {
    vmem_free(*out);
    *out = {};
    return testing::AssertionFailure()
        << "short read loading '" << path << "' (expected " << size << " bytes)";
  }
  return testing::AssertionSuccess();
}
#endif

// ---------------------------------------------------------------------------
// AIE packet submission
// ---------------------------------------------------------------------------

// Header fields every AIE dispatch packet carries, whatever kernel it names. The per-kernel
// helpers below differ only in the fields they fill in on top of this.
hsa_amd_aie_kernel_dispatch_packet_t make_aie_packet(hsa_signal_t completion_signal) {
  hsa_amd_aie_kernel_dispatch_packet_t pkt{};
  pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
  pkt.count = 24;
  pkt.completion_signal = completion_signal;
  return pkt;
}

// Claim the next queue slot and write `pkt` into it, returning its write index. Does not ring the
// doorbell, so a caller can batch several packets into one command chain.
std::uint64_t enqueue_aie_packet(hsa_queue_t* q, const hsa_amd_aie_kernel_dispatch_packet_t& pkt) {
  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address);

  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
  while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
    // wait for available slot - if it hangs here, then the doorbell was not rung, or the packet
    // was not processed for some reason
  }

  queue[wr_idx % q->size] = pkt;
  return wr_idx;
}

// ---------------------------------------------------------------------------
// Unified hsaco artifacts
//
// The suite's CMake packages each design's artifacts into an AIE hsaco: the PDI + instruction
// sequence build into <design>.hsaco and the full-ELF build into <design>_elf.hsaco. A dispatch
// packet names a kernel loaded from one of these and carries nothing else about it -- which of
// the two shapes it turns into is the loader's business, not the application's.
// ---------------------------------------------------------------------------
const std::filesystem::path kHsacoPath = STRINGIFY(DEFAULT_HSACO_PATH);
constexpr const char* kHsacoKernelName = DEFAULT_HSACO_KERNEL_NAME;
const std::filesystem::path kMulHsacoPath = STRINGIFY(MUL_HSACO_PATH);
constexpr const char* kMulHsacoKernelName = MUL_HSACO_KERNEL_NAME;
const std::filesystem::path kElfHsacoPath = STRINGIFY(DEFAULT_ELF_HSACO_PATH);
constexpr const char* kElfHsacoKernelName = DEFAULT_ELF_HSACO_KERNEL_NAME;
const std::filesystem::path kMulElfHsacoPath = STRINGIFY(MUL_ELF_HSACO_PATH);
// Neither design names its aie.device or its runtime sequence, so aiecc defaults both to the
// same "<device>:<sequence>" key.
constexpr const char* kMulElfHsacoKernelName = DEFAULT_ELF_HSACO_KERNEL_NAME;
// A FullElf hsaco whose kernel table has two entries sharing one blob (the same vsadd ELF,
// referenced twice) and therefore declaring the same kernel name; see
// HsacoFullElfRejectsDuplicateKernelName below.
const std::filesystem::path kDupElfHsacoPath = STRINGIFY(DUP_ELF_HSACO_PATH);

// The build skips packaging an hsaco when the toolchain cannot produce what it is built from, so
// every test that needs one checks first and skips itself rather than failing.
bool hsaco_available() { return std::filesystem::exists(kHsacoPath); }
bool mul_hsaco_available() { return std::filesystem::exists(kMulHsacoPath); }
bool elf_hsaco_available() { return std::filesystem::exists(kElfHsacoPath); }
bool mul_elf_hsaco_available() { return std::filesystem::exists(kMulElfHsacoPath); }
bool dup_elf_hsaco_available() { return std::filesystem::exists(kDupElfHsacoPath); }

// Reads a whole hsaco file into memory. Returns an empty vector on failure; callers are expected
// to have checked the matching *_available() first.
std::vector<std::uint8_t> read_hsaco(const std::filesystem::path& path) {
  std::size_t size = 0;
  auto f = open_binary(path, &size);
  if (!f) return {};
  std::vector<std::uint8_t> buf(size);
  if (!read_exact(f, buf.data(), size)) return {};
  return buf;
}

std::vector<std::uint8_t> hsaco_bytes() { return read_hsaco(kHsacoPath); }
std::vector<std::uint8_t> elf_hsaco_bytes() { return read_hsaco(kElfHsacoPath); }

std::size_t hsaco_size() {
  std::size_t size = 0;
  open_binary(kHsacoPath, &size);
  return size;
}

const char* kernel_name() { return kHsacoKernelName; }

// Owns one loaded, frozen AIE executable and the reader it came from, and hands out the kernel
// object handles its dispatch packets carry. A separate instance per load is what gives the PDI
// cache tests distinct PDI buffer objects from a single hsaco: each load places its own copy.
class loaded_hsaco {
 public:
  loaded_hsaco() = default;
  loaded_hsaco(const loaded_hsaco&) = delete;
  loaded_hsaco& operator=(const loaded_hsaco&) = delete;
  ~loaded_hsaco() { reset(); }

  // Loads and freezes `bytes` against `agent`. Leaves nothing behind on failure.
  testing::AssertionResult load(const std::vector<std::uint8_t>& bytes, hsa_agent_t agent) {
    reset();
    if (bytes.empty()) {
      return testing::AssertionFailure() << "empty code object";
    }
    if (auto s = hsa_code_object_reader_create_from_memory(bytes.data(), bytes.size(), &reader_);
        s != HSA_STATUS_SUCCESS) {
      return testing::AssertionFailure() << "hsa_code_object_reader_create_from_memory: " << s;
    }
    if (auto s = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                           HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                           &executable_);
        s != HSA_STATUS_SUCCESS) {
      hsa_code_object_reader_destroy(reader_);
      reader_ = {};
      return testing::AssertionFailure() << "hsa_executable_create_alt: " << s;
    }
    loaded_ = true;
    agent_ = agent;
    if (auto s = hsa_executable_load_agent_code_object(executable_, agent, reader_, nullptr,
                                                       nullptr);
        s != HSA_STATUS_SUCCESS) {
      reset();
      return testing::AssertionFailure() << "hsa_executable_load_agent_code_object: " << s;
    }
    if (auto s = hsa_executable_freeze(executable_, nullptr); s != HSA_STATUS_SUCCESS) {
      reset();
      return testing::AssertionFailure() << "hsa_executable_freeze: " << s;
    }
    return testing::AssertionSuccess();
  }

  testing::AssertionResult load(const std::filesystem::path& path, hsa_agent_t agent) {
    return load(read_hsaco(path), agent);
  }

  // The handle to put in a dispatch packet, or 0 if `name` is not in this executable.
  std::uint64_t kernel_object(const char* name) const {
    if (!loaded_) return 0;
    hsa_executable_symbol_t symbol{};
    hsa_agent_t agent = agent_;
    if (hsa_executable_get_symbol_by_name(executable_, name, &agent, &symbol) !=
        HSA_STATUS_SUCCESS) {
      return 0;
    }
    std::uint64_t kernel_object = 0;
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &kernel_object) != HSA_STATUS_SUCCESS) {
      return 0;
    }
    return kernel_object;
  }

  void reset() {
    if (loaded_) {
      hsa_executable_destroy(executable_);
      hsa_code_object_reader_destroy(reader_);
      executable_ = {};
      reader_ = {};
      loaded_ = false;
    }
  }

 private:
  hsa_executable_t executable_{};
  hsa_code_object_reader_t reader_{};
  hsa_agent_t agent_{};
  bool loaded_ = false;
};

// Compile-time constants and dispatch helper for the vector-scalar-add AIE
// kernel: adds 1 to every element of a uint32 array of element_count entries.
struct aie_vector_scalar_kernel {
  static const std::filesystem::path pdiPath;
  static const std::filesystem::path instsPath;

  // Number of elements in the input and output buffers for the vector-scalar add kernel.
  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  // Number of kernargs
  static constexpr std::size_t num_kernargs = 2;
  // Number of kernarg sizes (for this kernel, all kernargs are pointer+size pairs)
  static constexpr std::size_t num_kernarg_sizes = num_kernargs;
  // Kernargs and sizes
  static constexpr std::size_t num_kernargs_sizes = num_kernargs + num_kernarg_sizes;
  // Buffer size for kernargs: 2 pointers + 2 sizes
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(uint64_t);

  /**
   * @brief Create a AIE packet payload for vector-scalar add.
   *
   * @param kernel_object kernel object handle of the loaded vsadd kernel
   * @param input source buffer for the packet
   * @param output destination buffer for the packet
   * @param kernargs pointer to the kernel arguments buffer
   * @param completion_signal signal to be used for completion notification
   * @param q HSA queue to which the packet will be submitted
   */
  static std::uint64_t dispatch_packet(std::uint64_t kernel_object, void* input, void* output,
                                       uint64_t* kernargs, hsa_signal_t completion_signal,
                                       hsa_queue_t* q) {
    kernargs[0] = reinterpret_cast<uint64_t>(input);
    kernargs[1] = reinterpret_cast<uint64_t>(output);
    kernargs[2] = element_bytes;  // input size in bytes
    kernargs[3] = element_bytes;  // output size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.kernel_object_low = kernel_object & 0xFFFFFFFF;
    pkt.kernel_object_high = kernel_object >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;

    return enqueue_aie_packet(q, pkt);
  }
};
const std::filesystem::path aie_vector_scalar_kernel::pdiPath = STRINGIFY(DEFAULT_PDI_PATH);
const std::filesystem::path aie_vector_scalar_kernel::instsPath = STRINGIFY(DEFAULT_INSTS_PATH);

// Compile-time constants and dispatch helper for the vector-scalar-mul AIE kernel: multiplies
// every element of a uint32 array by `scale`, in place.
//
// Deliberately unlike aie_vector_scalar_kernel in the two ways a test can observe: the result
// says which design ran, and the single in/out buffer means one kernarg instead of two, so the
// two kernels' commands are not the same shape either.
struct aie_vector_scalar_mul_kernel {
  static const std::filesystem::path pdiPath;
  static const std::filesystem::path instsPath;

  // Injected by the build from VSMUL_SCALE, the same value it passes to the design script.
  static constexpr std::uint32_t scale = MUL_SCALE;

  // Number of elements in the in/out buffer for the vector-scalar mul kernel.
  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  // Number of kernargs: one buffer, read and written.
  static constexpr std::size_t num_kernargs = 1;
  // uint64_t slots one dispatch's kernargs occupy: every kernarg is an address plus a size, so
  // two per argument. Deliberately not spelled as a `num_kernargs` + `num_kernarg_sizes` pair --
  // those two names differ by one character and mistyping the stride silently halves it.
  static constexpr std::size_t num_kernargs_sizes = 2 * num_kernargs;
  // Buffer size for kernargs: 1 pointer + 1 size
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(uint64_t);

  /**
   * @brief Create an AIE packet payload for vector-scalar mul.
   *
   * @param kernel_object kernel object handle of the loaded vsmul kernel
   * @param inout buffer read and written by the packet
   * @param kernargs pointer to the kernel arguments buffer
   * @param completion_signal signal to be used for completion notification
   * @param q HSA queue to which the packet will be submitted
   */
  static std::uint64_t dispatch_packet(std::uint64_t kernel_object, void* inout,
                                       uint64_t* kernargs, hsa_signal_t completion_signal,
                                       hsa_queue_t* q) {
    kernargs[0] = reinterpret_cast<uint64_t>(inout);
    kernargs[1] = element_bytes;  // in/out size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.kernel_object_low = kernel_object & 0xFFFFFFFF;
    pkt.kernel_object_high = kernel_object >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;

    return enqueue_aie_packet(q, pkt);
  }
};
const std::filesystem::path aie_vector_scalar_mul_kernel::pdiPath = STRINGIFY(MUL_PDI_PATH);
const std::filesystem::path aie_vector_scalar_mul_kernel::instsPath = STRINGIFY(MUL_INSTS_PATH);

}  // namespace

// ===========================================================================
// Tests
// ===========================================================================

TEST(Dispatch, QueueCreate) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(min_queue_size, 0u);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, QueueMinMaxSize) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(min_queue_size, 0u);

  std::uint32_t max_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue_size),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(max_queue_size, 0u);

  EXPECT_LE(min_queue_size, max_queue_size);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, DevPoolDiscovery) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, KernargPoolDiscovery) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  // Try KERNARG_INIT pool first
  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  auto ka_status =
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &kernarg_pool_data);

  if (ka_status != HSA_STATUS_INFO_BREAK) {
    // Fall back to allocatable coarse-grained pool
    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    ASSERT_EQ(
        hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
        HSA_STATUS_INFO_BREAK);
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  // Verify we can allocate from the discovered pool
  void* test_buf = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool_data.pool, 64, 0, &test_buf),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(test_buf, nullptr);
  EXPECT_EQ(hsa_amd_memory_pool_free(test_buf), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, LoadPDI) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(
      load_binary(dev_pool_data.pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));
  EXPECT_NE(pdi_buf, nullptr);
  EXPECT_GT(pdi_size, 0u);

  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, LoadInstructions) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(
      load_binary(dev_pool_data.pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));
  EXPECT_NE(insts_buf, nullptr);
  EXPECT_GT(insts_size, 0u);
  EXPECT_EQ(insts_size % sizeof(std::uint32_t), 0u);

  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

// The dispatch tests' environment. The runtime and the AIE agent come from the shared base in
// aie_test_env.h; the pools and the queue sizing below are specific to this binary.
class DispatchTest : public aie_test::AieTestBase {
 protected:
  hsa_amd_memory_pool_t dev_pool{};
  hsa_amd_memory_pool_t data_pool{};
  hsa_amd_memory_pool_t kernarg_pool{};
  std::uint32_t min_queue_size = 0;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(AieTestBase::SetUp());
    ASSERT_FALSE(aie_agents.empty());

    // dev pool: coarse-grained, non-allocatable (for PDI and instructions)
    find_pool_data dev_pool_data{};
    dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    dev_pool_data.expected_allocatable = false;
    ASSERT_EQ(
        hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
        HSA_STATUS_INFO_BREAK);
    dev_pool = dev_pool_data.pool;

    // data pool: coarse-grained, allocatable (for tensor data)
    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    ASSERT_EQ(
        hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
        HSA_STATUS_INFO_BREAK);
    data_pool = data_pool_data.pool;

    // kernarg pool: KERNARG_INIT, allocatable; falls back to data pool
    find_pool_data kernarg_pool_data{};
    kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    kernarg_pool_data.expected_allocatable = true;
    if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool,
                                           &kernarg_pool_data) != HSA_STATUS_INFO_BREAK) {
      kernarg_pool_data.pool = data_pool_data.pool;
    }
    kernarg_pool = kernarg_pool_data.pool;

    ASSERT_EQ(
        hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
        HSA_STATUS_SUCCESS);
  }

  void TearDown() override {
    // Release the executables before the base class shuts the runtime down: destroying one after
    // hsa_shut_down() is a no-op whose error nothing can see, and it owns device memory.
    loaded_.clear();
    AieTestBase::TearDown();
  }

  // Loads `path` and returns the kernel object for `name`, or 0 on failure. The executable is
  // owned by the fixture and lives until TearDown, so every handle handed out stays valid for the
  // whole test. Each call loads a fresh executable, which is what gives a caller distinct PDI
  // buffer objects from one hsaco.
  std::uint64_t LoadKernel(const std::filesystem::path& path, const char* name) {
    auto loaded = std::make_unique<loaded_hsaco>();
    if (auto r = loaded->load(path, aie_agents.front()); !r) {
      ADD_FAILURE() << "loading " << path << ": " << r.message();
      return 0;
    }
    const std::uint64_t kernel_object = loaded->kernel_object(name);
    if (kernel_object == 0) {
      ADD_FAILURE() << "kernel '" << name << "' not published by " << path;
      return 0;
    }
    loaded_.push_back(std::move(loaded));
    return kernel_object;
  }

  // The two PDI + instruction sequence designs most tests use.
  std::uint64_t LoadAddKernel() { return LoadKernel(kHsacoPath, kHsacoKernelName); }
  std::uint64_t LoadMulKernel() { return LoadKernel(kMulHsacoPath, kMulHsacoKernelName); }

 private:
  std::vector<std::unique_ptr<loaded_hsaco>> loaded_;
};

TEST_F(DispatchTest, SingleDispatch) {
  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);

  std::iota(input, input + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output, aie_vector_scalar_kernel::element_count, 0);

  // --- Create payload ---
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch packet ---
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      kernel_object, input, output, kernargs, signal, queue);

  // --- Ring doorbell ---
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // --- Wait for completion ---
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // --- Verify output: output[i] == input[i] + 1 ---
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    EXPECT_EQ(output[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, SingleDispatchVMem) {
  // Same as SingleDispatch, but the I/O buffers and kernargs go through the vmem
  // API. PDI and instructions stay on plain pool allocation because they must
  // live in the dev heap, which is incompatible with the vmem reserve+map path
  // (see docs/bug-vmem-map-dev-heap.md). The vmem buffers are made accessible to
  // both the AIE agent (for execution) and the CPU agent (for filling inputs /
  // verifying outputs).
  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> access_agents;
  access_agents.insert(access_agents.end(), cpu_agents.begin(), cpu_agents.end());
  access_agents.insert(access_agents.end(), aie_agents.begin(), aie_agents.end());

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  // The runtime places the kernel's own buffers in the XDNA dev heap; only the I/O buffers and
  // kernargs below go through the vmem API, which the dev heap is incompatible with (see
  // docs/bug-vmem-map-dev-heap.md).
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  vmem_buffer input{};
  ASSERT_TRUE(
      vmem_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, access_agents, &input));

  vmem_buffer output{};
  ASSERT_TRUE(
      vmem_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, access_agents, &output));

  auto* input_data = static_cast<std::uint32_t*>(input.va);
  auto* output_data = static_cast<std::uint32_t*>(output.va);
  std::iota(input_data, input_data + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output_data, aie_vector_scalar_kernel::element_count, 0);

  // --- Create payload ---
  vmem_buffer kernargs{};
  ASSERT_TRUE(
      vmem_allocate(data_pool, aie_vector_scalar_kernel::kernarg_bytes, access_agents, &kernargs));

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch packet ---
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      kernel_object, input.va, output.va, static_cast<std::uint64_t*>(kernargs.va),
      signal, queue);

  // --- Ring doorbell ---
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // --- Wait for completion ---
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // --- Verify output: output[i] == input[i] + 1 ---
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    EXPECT_EQ(output_data[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(vmem_free(kernargs));
  EXPECT_TRUE(vmem_free(output));
  EXPECT_TRUE(vmem_free(input));
}

TEST_F(DispatchTest, MultiDispatch) {
  const std::uint32_t total_num_dispatches = 100;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for packet payload ---
  const auto total_kernargs_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernargs_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch loop ---
  for (std::uint32_t iter = 0; iter < total_num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input_ptr, output_ptr, kernarg_ptr, signal, queue);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MultiDispatchAsync) {
  const std::uint32_t total_num_dispatches = 40;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for packet payload ---
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch loop ---
  std::uint64_t last_wr_idx = 0;
  for (std::uint32_t iter = 0; iter < total_num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    last_wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input_ptr, output_ptr, kernarg_ptr, signal, queue);
  }

  // Ring doorbell
  hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MultiDispatchWrapAround) {
  const std::uint32_t initial_dispatches = min_queue_size / 2;
  const std::uint32_t num_dispatches = 10 * min_queue_size;
  const std::uint32_t total_num_dispatches = initial_dispatches + num_dispatches;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for kernel arguments ---
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Initial Packets to set-up for wrap-around ---
  for (std::uint32_t iter = 0; iter < initial_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input_ptr, output_ptr, kernarg_ptr, signal, queue);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // --- Dispatch loop ---
  for (std::uint32_t iter = 0; iter < num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto offset = initial_dispatches + iter;
    auto input_ptr = input + offset * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + offset * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + offset * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input_ptr, output_ptr, kernarg_ptr, signal, queue);
    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output[i] == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MultiDispatchWrapAroundAsync) {
  const std::uint32_t initial_dispatches = min_queue_size - 1;
  const std::uint32_t num_dispatches = 40;
  const std::uint32_t total_num_dispatches = initial_dispatches + num_dispatches;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for kernel arguments ---
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Initial Packets to set-up for wrap-around ---
  for (std::uint32_t iter = 0; iter < initial_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input_ptr, output_ptr, kernarg_ptr, signal, queue);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // --- Dispatch loop ---
  std::uint64_t last_wr_idx = 0;
  for (std::uint32_t iter = 0; iter < num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto offset = initial_dispatches + iter;
    auto input_ptr = input + offset * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + offset * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + offset * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    last_wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input_ptr, output_ptr, kernarg_ptr, signal, queue);
  }

  // Ring doorbell
  hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output[i] == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, AgentInfo) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  char name[64] = {};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_NAME, name), HSA_STATUS_SUCCESS);
  EXPECT_GT(std::strlen(name), 0u);

  char product_name[64] = {};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(),
                               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_PRODUCT_NAME),
                               product_name),
            HSA_STATUS_SUCCESS);

  char vendor_name[64] = {};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_VENDOR_NAME, vendor_name),
            HSA_STATUS_SUCCESS);
  EXPECT_STREQ(vendor_name, "AMD");

  // HSA_AMD_AGENT_INFO_UUID is documented (hsa_ext_amd.h) as an Ascii string with a
  // maximum of 21 chars including NUL. Query into an exactly 21-byte buffer flanked by
  // guard bytes to catch any write past it.
  struct {
    char guard_before[8];
    char uuid[21];
    char guard_after[8];
  } uuid_buf;
  std::memset(uuid_buf.guard_before, 0xAB, sizeof(uuid_buf.guard_before));
  std::memset(uuid_buf.uuid, 0xCD, sizeof(uuid_buf.uuid));
  std::memset(uuid_buf.guard_after, 0xAB, sizeof(uuid_buf.guard_after));

  char guard_expected[8];
  std::memset(guard_expected, 0xAB, sizeof(guard_expected));

  ASSERT_EQ(
      hsa_agent_get_info(aie_agents.front(), static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID),
                         uuid_buf.uuid),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(uuid_buf.guard_before, guard_expected, sizeof(guard_expected)), 0);
  EXPECT_EQ(std::memcmp(uuid_buf.guard_after, guard_expected, sizeof(guard_expected)), 0);
  EXPECT_STREQ(uuid_buf.uuid, "AIE-XX");

  hsa_agent_feature_t feature{};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_FEATURE, &feature),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(feature, HSA_AGENT_FEATURE_AGENT_DISPATCH);

  hsa_queue_type32_t queue_type{};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_TYPE, &queue_type),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queue_type, HSA_QUEUE_TYPE_SINGLE);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, CreateDestroyMultipleQueues) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(min_queue_size, 0u);

  std::uint32_t max_queues = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUES_MAX, &max_queues),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(max_queues, 0u);

  const std::uint32_t num_queues = std::min(max_queues, 4u);
  std::vector<hsa_queue_t*> queues(num_queues, nullptr);

  for (std::uint32_t i = 0; i < num_queues; ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                               nullptr, 0, 0, &queues[i]),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(queues[i], nullptr);
  }

  for (std::uint32_t i = 0; i < num_queues; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(hsa_queue_destroy(queues[i]), HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, DestroyQueueWithPendingPacket) {
  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load the kernel ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Allocate I/O buffers ---
  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);

  std::iota(input, input + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output, aie_vector_scalar_kernel::element_count, 0);

  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch and ring the doorbell, then tear the queue down without waiting ---
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      kernel_object, input, output, kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // AieAqlQueue::Inactivate destroys the kernel-mode queue; it does not drain the ring,
  // so this races tear-down against the in-flight packet. The destroy must still return
  // cleanly and leave the hardware detached from the buffers freed below.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, ConcurrentQueuesIndependentExecution) {
  constexpr std::uint32_t num_queues = 2;
  constexpr std::uint32_t num_rounds = 20;

  // --- Create queues ---
  hsa_queue_t* queues[num_queues] = {};
  for (std::uint32_t q = 0; q < num_queues; ++q) {
    SCOPED_TRACE(q);
    ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                               nullptr, 0, 0, &queues[q]),
              HSA_STATUS_SUCCESS);
  }

  // --- Load the kernel (shared by both queues) ---
  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  // --- Per-queue I/O buffers, so a cross-queue bleed shows up as a wrong value ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * num_rounds;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * num_rounds;
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * num_rounds;

  std::uint32_t* input[num_queues] = {};
  std::uint32_t* output[num_queues] = {};
  uint64_t* kernargs[num_queues] = {};
  hsa_signal_t signals[num_queues] = {};

  for (std::uint32_t q = 0; q < num_queues; ++q) {
    SCOPED_TRACE(q);
    ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                           reinterpret_cast<void**>(&input[q])),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                           reinterpret_cast<void**>(&output[q])),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                           reinterpret_cast<void**>(&kernargs[q])),
              HSA_STATUS_SUCCESS);

    // Distinct value ranges per queue.
    std::iota(input[q], input[q] + total_element_count, q * total_element_count);
    std::fill_n(output[q], total_element_count, 0);

    ASSERT_EQ(hsa_signal_create(num_rounds, 0, nullptr, &signals[q]), HSA_STATUS_SUCCESS);
  }

  // --- Interleave submissions so both queues are in flight at the same time ---
  for (std::uint32_t iter = 0; iter < num_rounds; ++iter) {
    for (std::uint32_t q = 0; q < num_queues; ++q) {
      SCOPED_TRACE(testing::Message() << "queue " << q << " iter " << iter);

      auto* input_ptr = input[q] + iter * aie_vector_scalar_kernel::element_count;
      auto* output_ptr = output[q] + iter * aie_vector_scalar_kernel::element_count;
      auto* kernarg_ptr = kernargs[q] + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

      const auto wr_idx =
          aie_vector_scalar_kernel::dispatch_packet(kernel_object, input_ptr,
                                                    output_ptr, kernarg_ptr, signals[q], queues[q]);
      hsa_signal_store_screlease(queues[q]->doorbell_signal, wr_idx);
    }
  }

  // --- Wait and verify each queue saw only its own dispatches ---
  for (std::uint32_t q = 0; q < num_queues; ++q) {
    SCOPED_TRACE(q);
    hsa_signal_wait_scacquire(signals[q], HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    for (std::size_t i = 0; i < total_element_count; ++i) {
      EXPECT_EQ(output[q][i], input[q][i] + 1) << "mismatch at index " << i;
    }
  }

  // --- Cleanup ---
  for (std::uint32_t q = 0; q < num_queues; ++q) {
    EXPECT_EQ(hsa_signal_destroy(signals[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_queue_destroy(queues[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(kernargs[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(output[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(input[q]), HSA_STATUS_SUCCESS);
  }
}


// ===========================================================================
// Full-ELF dispatch
//
// The runtime reads the ELF at load: it places the PDI in device memory, keeps a pristine copy of
// the control code, and records where the arguments go. A dispatch names the kernel object and
// nothing else -- the runtime copies the control code per packet, patches the packet's arguments
// into the copy, and dispatches that. The application cannot tell this apart from a PDI dispatch
// except by which artifacts its hsaco was built from.
//
// Supported on aie2p only.
// ===========================================================================

namespace {

// Owns a device-pool allocation so a test body can bail out with ASSERT_* without leaking.
class pool_buffer {
 public:
  pool_buffer() = default;
  // Takes ownership of a pointer already allocated from a pool.
  explicit pool_buffer(void* p) : ptr_(p) {}
  pool_buffer(const pool_buffer&) = delete;
  pool_buffer& operator=(const pool_buffer&) = delete;
  pool_buffer(pool_buffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  pool_buffer& operator=(pool_buffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~pool_buffer() { reset(); }

  // Releases anything already held: the class has value semantics, so a caller may reasonably
  // reuse one buffer, and overwriting ptr_ would leak the previous allocation silently.
  hsa_status_t allocate(hsa_amd_memory_pool_t pool, std::size_t size) {
    reset();
    return hsa_amd_memory_pool_allocate(pool, size, 0, &ptr_);
  }

  void reset() {
    if (ptr_ != nullptr) hsa_amd_memory_pool_free(ptr_);
    ptr_ = nullptr;
  }

  template <typename T> T* as() const { return static_cast<T*>(ptr_); }
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

// Post-dispatch checks for the two designs.
void VerifyAdd(const std::uint32_t* in, const std::uint32_t* out, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(out[i], in[i] + 1) << "add mismatch at index " << i;
  }
}

// The mul kernel works in place, so there is no untouched input to compare against and the
// expected value has to be reconstructed from the element's position in the iota that seeded the
// buffer. `base_index` is that position for inout[0]; it has no default because a slice of a
// larger buffer that quietly assumed 0 would compare against another chunk's expectations.
void VerifyMul(const std::uint32_t* inout, std::size_t count, std::size_t base_index) {
  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(inout[i],
              static_cast<std::uint32_t>(base_index + i) * aie_vector_scalar_mul_kernel::scale)
        << "mul mismatch at index " << (base_index + i);
  }
}

// Dispatch one add-kernel packet against `kernel_object`, wait for it, and check the whole output
// buffer. Both PDI-cache tests walk the cache one kernel object at a time exactly like this; call
// it under ASSERT_NO_FATAL_FAILURE so a mismatch stops the caller too.
void DispatchAddAndVerify(hsa_queue_t* queue, std::uint64_t kernel_object, std::uint32_t* in,
                          std::uint32_t* out, std::uint64_t* kernargs) {
  std::fill_n(out, aie_vector_scalar_kernel::element_count, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx =
      aie_vector_scalar_kernel::dispatch_packet(kernel_object, in, out, kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  VerifyAdd(in, out, aie_vector_scalar_kernel::element_count);
}

// The mul counterpart. Seeds the in-place buffer itself, since the kernel overwrites what it read
// and a second run over a stale buffer would be checking the wrong expectations.
void DispatchMulAndVerify(hsa_queue_t* queue, std::uint64_t kernel_object, std::uint32_t* inout,
                          std::uint64_t* kernargs) {
  std::iota(inout, inout + aie_vector_scalar_mul_kernel::element_count, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx =
      aie_vector_scalar_mul_kernel::dispatch_packet(kernel_object, inout, kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  VerifyMul(inout, aie_vector_scalar_mul_kernel::element_count, 0);
}

// ---------------------------------------------------------------------------
// Unified hsaco loader
//
// The suite's CMake packages the vsadd PDI+insts artifacts above into a single
// AIE hsaco (kind = PdiInsts) via aie_hsaco.py, so this test can exercise
// hsa_executable_load_agent_code_object end to end. The suite also packages the vsadd full-ELF
// artifact into a second hsaco (kind = FullElf); see the HsacoFullElf* tests below.
// ---------------------------------------------------------------------------
TEST_F(DispatchTest, HsacoKernelObjectIsPublished) {
  if (!hsaco_available()) {
    GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  }

  const auto hsaco = hsaco_bytes();
  ASSERT_EQ(hsaco.size(), hsaco_size());
  ASSERT_FALSE(hsaco.empty()) << "failed to read " << kHsacoPath;

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(hsa_code_object_reader_create_from_memory(hsaco.data(), hsaco.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_executable_t executable{};
  ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &executable),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(
      hsa_executable_load_agent_code_object(executable, aie_agents.front(), reader, nullptr, nullptr),
      HSA_STATUS_SUCCESS);

  // The kernel object handle is zero until the executable is frozen, matching the GPU contract.
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(hsa_executable_get_symbol_by_name(executable, kernel_name(), &aie_agents.front(),
                                              &symbol),
            HSA_STATUS_SUCCESS);
  std::uint64_t kernel_object = 1;
  ASSERT_EQ(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernel_object, 0u) << "kernel object must be zero before freeze";

  // Pins the hardcoded kernarg size aie_hsaco.py packaged this hsaco with (see the --kernel
  // argument in CMakeLists.txt) as a checked invariant, rather than trusting it silently.
  // num_cols=1 is packaged the same way but is not asserted here: unlike kernarg size, it has
  // no HSA_EXECUTABLE_SYMBOL_INFO_* accessor to check it against.
  std::uint32_t kernarg_segment_size = 0;
  ASSERT_EQ(hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &kernarg_segment_size),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernarg_segment_size, aie_vector_scalar_kernel::kernarg_bytes);

  ASSERT_EQ(hsa_executable_freeze(executable, nullptr), HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_executable_get_symbol_by_name(executable, kernel_name(), &aie_agents.front(),
                                              &symbol),
            HSA_STATUS_SUCCESS);
  kernel_object = 0;
  ASSERT_EQ(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(kernel_object, 0u) << "kernel object must be published after freeze";

  EXPECT_EQ(hsa_executable_destroy(executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_code_object_reader_destroy(reader), HSA_STATUS_SUCCESS);
}

// Returns the absolute file offset of the first kernel table entry's `kind` field (offset 28
// within aie_kernel_entry -- see core/inc/amd_aie_section.h) in an AIE hsaco, by walking the raw
// ELF section headers to find the section whose contents start with kAieSectionMagic. Returns 0
// (never a valid offset -- it always falls inside the ELF header) if no such section is found.
//
// This is a minimal, header-only re-scan (not a use of AieCode: that parser is internal to
// hsa-runtime64 and not exported, see amd_aie_code.cpp) good enough to corrupt one field for the
// negative test below; it does not need to be a general-purpose reader.
std::size_t FindAieSectionOffset(const std::vector<std::uint8_t>& hsaco) {
  if (hsaco.size() < sizeof(Elf64_Ehdr)) return 0;
  const auto* base = hsaco.data();
  const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(base);
  for (Elf64_Half i = 0; i < ehdr->e_shnum; ++i) {
    const std::uint64_t shdr_off =
        ehdr->e_shoff + static_cast<std::uint64_t>(i) * ehdr->e_shentsize;
    if (shdr_off + sizeof(Elf64_Shdr) > hsaco.size()) break;
    const auto* shdr = reinterpret_cast<const Elf64_Shdr*>(base + shdr_off);
    if (shdr->sh_size < sizeof(rocr::AMD::aie_section_header)) continue;
    if (shdr->sh_offset + shdr->sh_size > hsaco.size()) continue;
    const auto* sec_hdr =
        reinterpret_cast<const rocr::AMD::aie_section_header*>(base + shdr->sh_offset);
    if (sec_hdr->magic != rocr::AMD::kAieSectionMagic) continue;
    return shdr->sh_offset;
  }
  return 0;
}

std::size_t FindKindFieldOffset(const std::vector<std::uint8_t>& hsaco) {
  const std::size_t section_offset = FindAieSectionOffset(hsaco);
  if (section_offset == 0) return 0;
  const auto* sec_hdr =
      reinterpret_cast<const rocr::AMD::aie_section_header*>(hsaco.data() + section_offset);
  return section_offset + sec_hdr->header_size + offsetof(rocr::AMD::aie_kernel_entry, kind);
}

// Returns a copy of the PdiInsts hsaco with its first kernel's instruction blob overwritten with
// bytes the NPU cannot execute, or an empty vector if the section cannot be located. Everything
// host-side still accepts it -- the blob is the right size and the loader places it in the device
// heap exactly as before -- so a batch naming it builds normally and the device faults on the
// contents.
std::vector<std::uint8_t> BadInstsHsaco() {
  auto hsaco = hsaco_bytes();
  const std::size_t section_offset = FindAieSectionOffset(hsaco);
  if (section_offset == 0) return {};
  const auto* sec_hdr =
      reinterpret_cast<const rocr::AMD::aie_section_header*>(hsaco.data() + section_offset);
  if (sec_hdr->kernel_count == 0) return {};
  const auto* entry = reinterpret_cast<const rocr::AMD::aie_kernel_entry*>(
      hsaco.data() + section_offset + sec_hdr->header_size);
  const std::size_t insts_offset = section_offset + entry->insts_offset;
  if (insts_offset + entry->insts_size > hsaco.size()) return {};
  std::fill_n(hsaco.begin() + insts_offset, entry->insts_size, std::uint8_t{0xFF});
  return hsaco;
}

// Closes I-4/Concern-2: the kind-validation switch in LoadAieCodeObject (executable.cpp) is
// otherwise unreachable via this hsaco, since aie_hsaco.py packages it with kind=PdiInsts.
// kind=FullElf is exercised for real by HsacoFullElfLoads below; here it is reached by corrupting
// an already-built PdiInsts hsaco's bytes directly, which still must fail -- a PdiInsts kernel
// table entry patched to claim kind=FullElf does not contain a nested ELF, so the load now fails
// while trying to parse the (non-ELF) insts blob as one, rather than at the kind-range check
// itself. The unknown-kind (99) half still fails at the kind-range check, unchanged.
TEST_F(DispatchTest, AieKindIsValidated) {
  if (!hsaco_available()) {
    GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  }

  const auto hsaco = hsaco_bytes();
  ASSERT_FALSE(hsaco.empty()) << "failed to read " << kHsacoPath;

  const std::size_t kind_offset = FindKindFieldOffset(hsaco);
  ASSERT_NE(kind_offset, 0u) << "could not locate the AIE section in " << kHsacoPath;
  ASSERT_LE(kind_offset + sizeof(rocr::AMD::AieKernelKind), hsaco.size());

  for (const std::uint32_t bad_kind :
       {static_cast<std::uint32_t>(rocr::AMD::AieKernelKind::FullElf), std::uint32_t{99}}) {
    auto patched = hsaco;
    std::memcpy(patched.data() + kind_offset, &bad_kind, sizeof(bad_kind));

    hsa_code_object_reader_t reader{};
    ASSERT_EQ(hsa_code_object_reader_create_from_memory(patched.data(), patched.size(), &reader),
              HSA_STATUS_SUCCESS);

    hsa_executable_t executable{};
    ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                        nullptr, &executable),
              HSA_STATUS_SUCCESS);

    EXPECT_EQ(hsa_executable_load_agent_code_object(executable, aie_agents.front(), reader,
                                                     nullptr, nullptr),
              HSA_STATUS_ERROR_INVALID_CODE_OBJECT)
        << "kind=" << bad_kind << " must be rejected";

    EXPECT_EQ(hsa_executable_destroy(executable), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_code_object_reader_destroy(reader), HSA_STATUS_SUCCESS);
  }
}

// The full-ELF build of the same vector-scalar-add kernel used above.
//
// Nothing in a full-ELF packet distinguishes it from a PDI one: the dispatch shape follows from
// the kernel object. The runtime copies the kernel's control code per packet and patches these
// argument addresses into the copy, so two packets naming one kernel with different buffers need
// nothing special from the caller.
struct aie_full_elf_kernel {
  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  static constexpr std::size_t num_kernargs = 2;
  static constexpr std::size_t num_kernargs_sizes = 2 * num_kernargs;
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(std::uint64_t);

  // Write a full-ELF packet into the next queue slot. Does not ring the doorbell,
  // so a caller can batch several packets into one command chain.
  static std::uint64_t dispatch_packet(
      std::uint64_t kernel_object, void* input, void* output, std::uint64_t* kernargs,
      hsa_signal_t completion_signal, hsa_queue_t* q,
      const std::function<void(hsa_amd_aie_kernel_dispatch_packet_t&)>& mutate = {}) {
    kernargs[0] = reinterpret_cast<std::uint64_t>(input);
    kernargs[1] = reinterpret_cast<std::uint64_t>(output);
    kernargs[2] = element_bytes;  // input size in bytes
    kernargs[3] = element_bytes;  // output size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.kernel_object_low = kernel_object & 0xFFFFFFFF;
    pkt.kernel_object_high = kernel_object >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;

    // Applied last, so a test can corrupt exactly one field of a packet that is otherwise known
    // to be good.
    if (mutate) mutate(pkt);

    return enqueue_aie_packet(q, pkt);
  }
};

// The full-ELF build of the vector-scalar-mul kernel. Same shape as above, but one argument
// instead of two, since the design reads and writes a single buffer.
struct aie_full_elf_mul_kernel {
  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  static constexpr std::size_t num_kernargs = 1;
  static constexpr std::size_t num_kernargs_sizes = 2 * num_kernargs;
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(std::uint64_t);

  static std::uint64_t dispatch_packet(std::uint64_t kernel_object, void* inout,
                                       std::uint64_t* kernargs, hsa_signal_t completion_signal,
                                       hsa_queue_t* q) {
    kernargs[0] = reinterpret_cast<std::uint64_t>(inout);
    kernargs[1] = element_bytes;  // in/out size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.kernel_object_low = kernel_object & 0xFFFFFFFF;
    pkt.kernel_object_high = kernel_object >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;

    return enqueue_aie_packet(q, pkt);
  }
};

}  // namespace

class FullElfDispatchTest : public DispatchTest {
 protected:
  // Kernel object of the full-ELF vector-scalar-add kernel, loaded once per test.
  std::uint64_t elf_kernel = 0;

  void SetUp() override {
    DispatchTest::SetUp();
    if (::testing::Test::HasFatalFailure()) return;

    // Full-ELF dispatch is aie2p only; aie2 has neither the firmware command nor the
    // preemption support it is built on.
    char agent_name[64] = {};
    ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_NAME, agent_name),
              HSA_STATUS_SUCCESS);
    if (std::strcmp(agent_name, "aie2p") != 0) {
      GTEST_SKIP() << "full-ELF dispatch needs an aie2p agent, found '" << agent_name << "'";
    }

    // The build skips the hsaco when the toolchain cannot produce a full ELF.
    if (!elf_hsaco_available()) {
      GTEST_SKIP() << "full-ELF hsaco was not built: " << kElfHsacoPath;
    }

    elf_kernel = LoadKernel(kElfHsacoPath, kElfHsacoKernelName);
    ASSERT_NE(elf_kernel, 0u);
  }

  // Checks the whole output buffer of an add dispatch.
  static void ExpectVsaddCorrect(const std::uint32_t* in, const std::uint32_t* out,
                                 std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      ASSERT_EQ(out[i], in[i] + 1) << "mismatch at index " << i;
    }
  }

  // Dispatch `num_dispatches` full-ELF packets with distinct buffer pairs as a
  // single command chain, and check every output. The runtime gives each packet its own
  // control-code buffer; sharing one would run them all with the last packet's arguments.
  void RunChain(hsa_queue_t* queue, std::uint32_t num_dispatches) {
    const std::size_t total_elements = aie_full_elf_kernel::element_count * num_dispatches;

    pool_buffer input, output, kernargs;
    ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_dispatches),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_dispatches),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * num_dispatches),
              HSA_STATUS_SUCCESS);

    auto* in = input.as<std::uint32_t>();
    auto* out = output.as<std::uint32_t>();
    std::iota(in, in + total_elements, 0);
    std::fill_n(out, total_elements, 0);

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

    // Enqueue every packet before ringing the doorbell, so the runtime submits them
    // as one chain rather than one at a time.
    std::uint64_t wr_idx = 0;
    for (std::uint32_t i = 0; i < num_dispatches; ++i) {
      wr_idx = aie_full_elf_kernel::dispatch_packet(
          elf_kernel, in + i * aie_full_elf_kernel::element_count,
          out + i * aie_full_elf_kernel::element_count,
          kernargs.as<std::uint64_t>() + i * aie_full_elf_kernel::num_kernargs_sizes, signal,
          queue);
    }
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);

    ASSERT_NO_FATAL_FAILURE(ExpectVsaddCorrect(in, out, total_elements));

    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  }
};

// ---------------------------------------------------------------------------
// Unified hsaco loader: FullElf
//
// The suite's CMake packages the vsadd full-ELF artifact into a single AIE hsaco (kind =
// FullElf) via aie_hsaco.py's "elf:" kernel spec, so these tests can exercise
// hsa_executable_load_agent_code_object on the FullElf path added by Task 5, mirroring the
// PdiInsts coverage above (HsacoKernelObjectIsPublished, AieKindIsValidated).
// ---------------------------------------------------------------------------
// Loads `bytes` as a code object reader against `agent` and returns the load status without
// asserting on it, so a caller can check a specific failure code. On success, `executable` and
// `reader` are left open for the caller to inspect and destroy; on failure, both are already
// destroyed.
hsa_status_t TryLoad(const std::vector<std::uint8_t>& bytes, hsa_agent_t agent,
                     hsa_executable_t* executable, hsa_code_object_reader_t* reader) {
  if (hsa_code_object_reader_create_from_memory(bytes.data(), bytes.size(), reader) !=
      HSA_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                executable) != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(*reader);
    return HSA_STATUS_ERROR;
  }

  const hsa_status_t status =
      hsa_executable_load_agent_code_object(*executable, agent, *reader, nullptr, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(*executable);
    hsa_code_object_reader_destroy(*reader);
  }
  return status;
}

// Loads `bytes` and freezes the resulting executable; asserts on every step, since the tests that
// use this only care about what happens after a known-good load. Leaves `executable` and `reader`
// open for the caller to inspect and destroy.
void LoadAndFreeze(const std::vector<std::uint8_t>& bytes, hsa_agent_t agent,
                   hsa_executable_t* executable, hsa_code_object_reader_t* reader) {
  ASSERT_EQ(TryLoad(bytes, agent, executable, reader), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_freeze(*executable, nullptr), HSA_STATUS_SUCCESS);
}

// Returns a copy of the FullElf hsaco with its (only) kernel table entry's name replaced by a
// name the nested ELF does not contain, so the loader must fail after successfully parsing the
// ELF rather than while locating the AIE section. The replacement is the same length as the
// original name plus its NUL, so no other offset in the section needs to move.
std::vector<std::uint8_t> BadNameHsaco() {
  auto hsaco = elf_hsaco_bytes();
  if (hsaco.empty()) return {};

  const std::string original_name = kElfHsacoKernelName;
  const std::string bad_name(original_name.size(), 'z');

  const auto it = std::search(hsaco.begin(), hsaco.end(), original_name.begin(),
                              original_name.end());
  if (it == hsaco.end()) return {};
  std::copy(bad_name.begin(), bad_name.end(), it);
  return hsaco;
}

TEST_F(FullElfDispatchTest, HsacoFullElfLoads) {
  if (!elf_hsaco_available()) {
    GTEST_SKIP() << "elf hsaco was not built: " << kElfHsacoPath;
  }

  const auto hsaco = elf_hsaco_bytes();
  ASSERT_FALSE(hsaco.empty()) << "failed to read " << kElfHsacoPath;

  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  ASSERT_NO_FATAL_FAILURE(LoadAndFreeze(hsaco, aie_agents.front(), &executable, &reader));

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(hsa_executable_get_symbol_by_name(executable, kElfHsacoKernelName, &aie_agents.front(),
                                              &symbol),
            HSA_STATUS_SUCCESS);
  std::uint64_t kernel_object = 0;
  ASSERT_EQ(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(kernel_object, 0u) << "kernel object must be published after freeze";

  // The packaged hsaco's kernel-table entry carries kernarg_size=0 (aie_hsaco.py's "elf:" form
  // does not know it); the loader must fill it in from the nested ELF's argument count rather
  // than report 0.
  std::uint32_t kernarg_segment_size = 0;
  ASSERT_EQ(hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &kernarg_segment_size),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernarg_segment_size,
            aie_full_elf_kernel::num_kernargs * sizeof(std::uint64_t));

  EXPECT_EQ(hsa_executable_destroy(executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_code_object_reader_destroy(reader), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, HsacoFullElfRejectsUnknownKernelName) {
  if (!elf_hsaco_available()) {
    GTEST_SKIP() << "elf hsaco was not built: " << kElfHsacoPath;
  }

  const auto bad_hsaco = BadNameHsaco();
  ASSERT_FALSE(bad_hsaco.empty()) << "failed to build a corrupted copy of " << kElfHsacoPath;

  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(TryLoad(bad_hsaco, aie_agents.front(), &executable, &reader),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

// The suite's CMake packages this hsaco (kDupElfHsacoPath) by pointing aie_hsaco.py at the vsadd
// full-ELF artifact twice. aie_hsaco.py derives each kernel-table entry's name from the nested
// ELF's own COMDAT symbols, so two references to the same ELF always produce two entries with the
// identical name "main:sequence"; its blob-pool dedup (keyed on raw bytes) then collapses the two
// identical instruction blobs into one shared blob. That is exactly the shape
// AieCode::Parse's `kernels_.count(info.name)` check (core/runtime/amd_aie_code.cpp) exists to
// reject, and nothing exercised it on the FullElf path before this test.
//
// NOTE: this is not the multi-kernel-ELF scenario the design spec describes -- one ELF containing
// two *different* kernels (two distinct COMDAT groups), which would also produce two entries
// sharing one blob, but with two different names, both resolving successfully. That scenario is
// not covered by any test: every full-ELF artifact this suite's toolchain can build
// (kernel_full_elf_vsadd/aie.elf, kernel_full_elf_vsmul/aie.elf) contains exactly one COMDAT
// group, so there is no multi-kernel ELF anywhere in the tree to build such a test from, and
// producing one needs a new aiecc design, not just test code. That gap is real and left open.
TEST_F(FullElfDispatchTest, HsacoFullElfRejectsDuplicateKernelName) {
  if (!dup_elf_hsaco_available()) {
    GTEST_SKIP() << "duplicate-name elf hsaco was not built: " << kDupElfHsacoPath;
  }

  const auto hsaco = read_hsaco(kDupElfHsacoPath);
  ASSERT_FALSE(hsaco.empty()) << "failed to read " << kDupElfHsacoPath;

  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(TryLoad(hsaco, aie_agents.front(), &executable, &reader),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST_F(FullElfDispatchTest, HsacoFullElfDispatches) {
  // End to end: load, dispatch, check the output buffer.
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NO_FATAL_FAILURE(RunChain(queue, 1));
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, SameKernelDifferentArgsInOneBatch) {
  // Two packets, one kernel, different argument buffers, ONE submission. Each needs its own
  // control-code buffer; sharing one runs both with the second packet's arguments, which shows up
  // as the first dispatch's output never being written.
  constexpr std::uint32_t n = aie_full_elf_kernel::element_count;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(queue->size, 2u);

  // Separate allocations rather than two halves of one, so the two dispatches cannot be told
  // apart by an offset the runtime might have got right by accident.
  pool_buffer in_a, out_a, in_b, out_b, kernargs;
  ASSERT_EQ(in_a.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(out_a.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(in_b.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(out_b.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * 2),
            HSA_STATUS_SUCCESS);

  auto* ia = in_a.as<std::uint32_t>();
  auto* oa = out_a.as<std::uint32_t>();
  auto* ib = in_b.as<std::uint32_t>();
  auto* ob = out_b.as<std::uint32_t>();
  std::iota(ia, ia + n, 0);
  std::iota(ib, ib + n, 5000);
  std::fill_n(oa, n, 0);
  std::fill_n(ob, n, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  auto* args = kernargs.as<std::uint64_t>();
  aie_full_elf_kernel::dispatch_packet(elf_kernel, ia, oa, args, signal, queue);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      elf_kernel, ib, ob, args + aie_full_elf_kernel::num_kernargs_sizes, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  ASSERT_NO_FATAL_FAILURE(ExpectVsaddCorrect(ia, oa, n));
  ASSERT_NO_FATAL_FAILURE(ExpectVsaddCorrect(ib, ob, n));

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfFullQueueDispatch) {
  // A chain as long as the queue allows. A full-ELF command occupies 60 bytes of
  // the driver's 4 KiB chain buffer, so 68 would fit; the queue tops out first.
  // A one-packet queue would leave the chain path untested rather than failing.
  ASSERT_GE(min_queue_size, 2u) << "queue too small to batch a chain";

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  RunChain(queue, min_queue_size);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfRepatch) {
  // Dispatching one kernel repeatedly with different arguments has to give the right answer every
  // time. The shim-DMA patch scheme adds to the buffer descriptor already in the control code, so
  // a runtime that patched over the previous result instead of over a pristine copy gets the
  // second dispatch wrong while the first still looks fine.
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  constexpr std::uint32_t rounds = 4;
  for (std::uint32_t round = 0; round < rounds; ++round) {
    SCOPED_TRACE(round);

    // Fresh buffers each round, so every dispatch patches a different address into the control
    // code of the same kernel.
    pool_buffer input, output, kernargs;
    ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
              HSA_STATUS_SUCCESS);

    auto* in = input.as<std::uint32_t>();
    auto* out = output.as<std::uint32_t>();
    std::iota(in, in + aie_full_elf_kernel::element_count, round * 1000);
    std::fill_n(out, aie_full_elf_kernel::element_count, 0);

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

    const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
        elf_kernel, in, out, kernargs.as<std::uint64_t>(), signal, queue);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);

    ASSERT_NO_FATAL_FAILURE(
        ExpectVsaddCorrect(in, out, aie_full_elf_kernel::element_count));
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfNoPdiCeiling) {
  // The PDI + instruction sequence path can hold at most 32 distinct PDIs per hardware context,
  // because each one costs a compute-unit slot. Full-ELF loads the PDI from the control code
  // instead, so there is no such ceiling: use more than 32 separate PDIs on one queue. Each load
  // of the hsaco places its own copy of the PDI, so these are distinct buffer objects holding
  // identical bytes -- which is what the PDI cache would key on.
  constexpr std::uint32_t num_pdis = 40;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  std::vector<std::uint64_t> kernels(num_pdis);
  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    kernels[i] = LoadKernel(kElfHsacoPath, kElfHsacoKernelName);
    ASSERT_NE(kernels[i], 0u);
  }

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);

  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    std::fill_n(out, aie_full_elf_kernel::element_count, 0);

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
    const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
        kernels[i], in, out, kernargs.as<std::uint64_t>(), signal, queue);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

    ASSERT_NO_FATAL_FAILURE(
        ExpectVsaddCorrect(in, out, aie_full_elf_kernel::element_count));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}
// else. The doorbell store is void, so it cannot return a status; and the packet
// never reached the device, so its completion signal is never released. A blocking
// wait on that signal would therefore hang forever -- these tests must not wait on
// it, and instead assert the signal still holds its initial value.
//
// Submission is synchronous on the calling thread for KMQ queues, so the callback
// has already fired by the time the doorbell store returns.
//
// A queue that takes a rejected packet is suspended and submits nothing further, by
// design: the failed packets are deliberately left unconsumed rather than retired.
// So "the refusal did not poison anything" is checked on a *fresh* queue, never by
// reusing the suspended one.
// ===========================================================================

// Asserts the dispatch was refused: reported once through the error callback, and its completion
// signal left untouched because the packet never executed.
void ExpectRejected(const aie_test::dispatch_error& err, hsa_queue_t* queue, hsa_signal_t signal,
                    hsa_signal_value_t initial) {
  EXPECT_TRUE(err.invoked.load(std::memory_order_acquire))
      << "a rejected dispatch did not report through the queue error callback";
  EXPECT_NE(err.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(err.source, queue);
  EXPECT_EQ(hsa_signal_load_scacquire(signal), initial)
      << "completion signal fired for a packet that never executed";
  // The refused packet is still in the ring: only packets that ran are consumed, so the read
  // index cannot have caught up with the write index.
  EXPECT_LT(hsa_queue_load_read_index_scacquire(queue), hsa_queue_load_write_index_scacquire(queue))
      << "a refused packet was consumed from the ring";
}

TEST_F(FullElfDispatchTest, ControlCodeIsAlignedForDispatch) {
  // The driver rejects a control code whose device address is not 16 KiB aligned, so a completed
  // dispatch is the assertion: the runtime's own allocation satisfied it. A single dispatch would
  // only show that one starting address happened to work, and the rounding is arithmetic on
  // whatever base the allocator returned -- so walk a range of bases instead.
  //
  // Each round holds a device buffer of a different, deliberately unaligned size across the
  // dispatch, so the next control-code allocation starts somewhere else in the heap. The sizes
  // step by a dword so the offsets are not all congruent modulo the alignment, and one of them is
  // a whole 16 KiB so an allocator that is already aligned is exercised too.
  constexpr std::size_t kBiasSizes[] = {4, 1024, 4100, 16384, 16388, 32772};

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  for (std::size_t bias : kBiasSizes) {
    SCOPED_TRACE(bias);
    pool_buffer spacer;
    ASSERT_EQ(spacer.allocate(dev_pool, bias), HSA_STATUS_SUCCESS);
    // Two packets per round, so the second control-code buffer starts from wherever the first
    // left off rather than always from the same spacer boundary.
    ASSERT_NO_FATAL_FAILURE(RunChain(queue, 2));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, KernargCountMismatchRejected) {
  // The arguments are patched into a copy of the kernel's control code rather than handed to the
  // hardware, so nothing downstream would notice a short list: the dispatch would run against
  // whatever the ELF's placeholder happened to be. The refusal has to name the reason, so this
  // asserts the status as well as the fact that it was refused.
  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, aie_full_elf_kernel::element_count, sentinel);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      elf_kernel, in, out, kernargs.as<std::uint64_t>(), signal, queue,
      [](hsa_amd_aie_kernel_dispatch_packet_t& p) {
        p.num_kernargs = aie_full_elf_kernel::num_kernargs - 1;
      });
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  ExpectRejected(err, queue, signal, 1);
  EXPECT_EQ(err.status, HSA_STATUS_ERROR_INVALID_PACKET_FORMAT);

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfMalformedPacketsRejected) {
  // Each case corrupts exactly one field of a packet that is otherwise known good, so a failure
  // points at one validation rather than at "the packet was bad somehow". Either way the output
  // buffer must come back untouched and the refusal must reach the error callback.
  //
  // All of them are refused while the batch is being built, before anything is submitted.
  using packet_t = hsa_amd_aie_kernel_dispatch_packet_t;

  struct malformed_case {
    const char* name;
    std::function<void(packet_t&)> mutate;
  };
  const std::vector<malformed_case> cases = {
      {"null kernel object",
       [](packet_t& p) {
         p.kernel_object_low = 0;
         p.kernel_object_high = 0;
       }},
      // The arguments are patched into the kernel's control code rather than handed to the
      // hardware, so a miscounted list would otherwise run against the ELF's placeholders.
      {"too few kernargs", [](packet_t& p) { p.num_kernargs = aie_full_elf_kernel::num_kernargs - 1; }},
      {"too many kernargs",
       [](packet_t& p) { p.num_kernargs = aie_full_elf_kernel::num_kernargs + 1; }},
      {"unrecognised opcode", [](packet_t& p) { p.opcode = 0xFF; }},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);

    // A fresh queue per case: a rejected packet suspends its queue, and reusing one would let an
    // earlier case mask a later one.
    dispatch_error err;
    hsa_queue_t* queue = nullptr;
    ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
              HSA_STATUS_SUCCESS);

    pool_buffer input, output, kernargs;
    ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    // Sized for one more argument than the kernel takes, so the "too many kernargs" case is
    // refused by the count check rather than by the buffer bound.
    ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes +
                                                  2 * sizeof(std::uint64_t)),
              HSA_STATUS_SUCCESS);

    auto* in = input.as<std::uint32_t>();
    auto* out = output.as<std::uint32_t>();
    std::iota(in, in + aie_full_elf_kernel::element_count, 0);
    constexpr std::uint32_t sentinel = 0xD0D0D0D0;
    std::fill_n(out, aie_full_elf_kernel::element_count, sentinel);

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
    const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
        elf_kernel, in, out, kernargs.as<std::uint64_t>(), signal, queue, c.mutate);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    ExpectRejected(err, queue, signal, 1);

    for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
      ASSERT_EQ(out[i], sentinel) << "rejected dispatch wrote output at index " << i;
    }

    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  }
}

TEST_F(DispatchTest, PdiCacheRolledBackOnFailedBatch) {
  // Building a command records its PDI in the queue's cache, but the hardware context is only
  // reconfigured to match after the whole batch is built. A batch that fails in between must not
  // leave the cache claiming compute units the context never got -- otherwise the next submission
  // finds the PDI "cached", skips the reconfigure, and dispatches against a context that cannot
  // run it. Submit a batch whose second packet is malformed, then check a fresh queue still works.
  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  std::uint32_t* input = nullptr;
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output, aie_vector_scalar_kernel::element_count, 0);

  // Two packets in one batch: the first introduces the PDI, the second is rejected because it
  // declares far more kernel arguments than its kernarg buffer holds.
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  aie_vector_scalar_kernel::dispatch_packet(kernel_object, input, output, kernargs,
                                            signal, queue);
  auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(kernel_object, input,
                                                          output, kernargs, signal, queue);
  auto* ring = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(queue->base_address);
  ring[wr_idx % queue->size].num_kernargs = 2000;
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  // The batch is refused while it is still being built, so neither packet runs and the signal
  // keeps both of its counts.
  ExpectRejected(err, queue, signal, 2);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  // A well-formed dispatch on a fresh queue must still work: if the cache kept the rejected
  // batch's entry, this one skips the reconfigure and runs against an unconfigured context.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  wr_idx = aie_vector_scalar_kernel::dispatch_packet(kernel_object, input, output,
                                                     kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    ASSERT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

// The ceiling DispatchTest.PdiCacheHoldsThirtyTwo walks up to, seen from the other side. A
// 33rd distinct PDI has no compute-unit slot left -- the CU mask is 32 bits -- so the runtime
// refuses the packet rather than dispatching it against a context that cannot select it.
//
// The refusal happens while the batch is still being built, before anything is submitted, so
// nothing runs and the output stays as the test left it.
TEST_F(DispatchTest, PdiCacheRejectsThirtyThree) {
  constexpr std::uint32_t cache_capacity = 32;

  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;

  // One more kernel than the cache holds. Each load places its own copy of the PDI, so these are
  // identical bytes in distinct buffer objects -- and the cache keys on the handle.
  std::vector<std::uint64_t> kernels(cache_capacity + 1);
  for (std::uint32_t i = 0; i < kernels.size(); ++i) {
    SCOPED_TRACE(i);
    kernels[i] = LoadAddKernel();
    ASSERT_NE(kernels[i], 0u);
  }

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  auto* args = kernargs.as<std::uint64_t>();
  std::iota(in, in + aie_vector_scalar_kernel::element_count, 0);

  // Fill the cache. Each of these has to succeed, or the test is not measuring what it claims.
  for (std::uint32_t i = 0; i < cache_capacity; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, kernels[i], in, out, args));
  }

  // The 33rd: rejected, so the output keeps the sentinel.
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, aie_vector_scalar_kernel::element_count, sentinel);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(kernels[cache_capacity], in, out,
                                                                args, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  ExpectRejected(err, queue, signal, 1);
  for (std::size_t e = 0; e < aie_vector_scalar_kernel::element_count; ++e) {
    ASSERT_EQ(out[e], sentinel) << "rejected dispatch wrote output at index " << e;
  }
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  // Hitting the ceiling refuses the packet that could not be placed; it does not take the device
  // or the runtime with it. The refused queue is suspended by design, so this is checked on a
  // fresh one -- which starts with an empty cache and so has room for the PDI again.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, kernels[0], in, out, args));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Grouping dispatches by mode is the caller's job. A batch whose packets do not agree is
// refused outright rather than split: splitting would mean rebuilding the hardware context with
// packets of the same batch still in flight, which the two modes' incompatible CU configurations
// make unsafe. Switching between *batches* is supported -- see ModeSwitchAlternatingBatches.
TEST_F(FullElfDispatchTest, MixedModeBatchRejected) {
  static_assert(aie_vector_scalar_kernel::element_count == aie_full_elf_kernel::element_count);
  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;

  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(queue->size, 2u);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t add_kernel = LoadAddKernel();
  ASSERT_NE(add_kernel, 0u);

  pool_buffer input, output, pdi_kernargs, elf_kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * 2),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * 2),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(pdi_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(elf_kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + n * 2, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, n * 2, sentinel);

  // One PDI packet and one full-ELF packet staged together and released with a single doorbell.
  // The first fixes the batch's mode, the second contradicts it, so neither runs.
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  aie_vector_scalar_kernel::dispatch_packet(add_kernel, in, out,
                                            pdi_kernargs.as<std::uint64_t>(), signal, queue);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      elf_kernel, in + n, out + n, elf_kernargs.as<std::uint64_t>(), signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  // Neither packet runs, so the signal keeps both of its counts.
  ExpectRejected(err, queue, signal, 2);

  for (std::size_t i = 0; i < n * 2; ++i) {
    ASSERT_EQ(out[i], sentinel) << "rejected batch wrote output at index " << i;
  }
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  // The refusal does not take the device or the runtime with it: a well-formed batch still runs.
  // The refused queue is suspended by design, so this runs on a fresh one.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NO_FATAL_FAILURE(
      DispatchAddAndVerify(queue, add_kernel, in, out, pdi_kernargs.as<std::uint64_t>()));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// The PDI + instruction sequence path packs a wider chain slot than full-ELF, so
// its chains are shorter. The driver packs each command into a 4 KiB buffer, giving
// floor(4096 / (52 + 4 * arg_cnt)) commands per chain: 44 for this kernel's two
// arguments, against a 64-packet queue. The runtime splits an oversized batch
// across several chains rather than letting the driver reject it.
//
// Full-ELF needs no such split: its commands take 60 bytes, so 68 fit and the queue
// tops out first.
TEST_F(DispatchTest, PdiChainSplit) {
  const std::uint32_t total_num_dispatches = 64;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(min_queue_size, total_num_dispatches);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);

  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                data_pool, aie_vector_scalar_kernel::element_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                data_pool, aie_vector_scalar_kernel::element_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);
  std::fill_n(output, total_element_count, 0);

  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // One doorbell for the whole batch, so the runtime sees all of them at once.
  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < total_num_dispatches; ++i) {
    wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel_object, input + i * aie_vector_scalar_kernel::element_count,
        output + i * aie_vector_scalar_kernel::element_count,
        kernargs + i * aie_vector_scalar_kernel::num_kernargs_sizes, signal, queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  for (std::size_t i = 0; i < total_element_count; ++i) {
    ASSERT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

// A batch too large for one chain is split and submitted as several chains back to back. When a
// later chain fails, the packets in the chains before it have already run on the device and
// written their output, so their completion signals have to fire: the failure is reported once,
// through the queue callback, and a waiter on an earlier packet has no other way to learn that its
// own dispatch succeeded.
//
// The failure is induced with a second kernel whose instruction sequence has been overwritten with
// bytes the NPU cannot execute. It loads and builds normally, and unlike the PDI the instruction
// sequence takes no part in configuring the hardware context -- so the failure lands at the device,
// mid-batch, instead of before submission. Only the driver's ~6 s watchdog catches it, which is
// what makes this test slow.
//
// The last packet is the failing one, so the split point does not have to be hardcoded: whatever
// it is, at least one whole chain ran. The check that every retired packet also produced correct
// output is the load-bearing one -- it is what would catch the runtime crediting a packet that
// never executed.
TEST_F(DispatchTest, PartiallyFailedBatchRetiresCompletedPackets) {
  constexpr std::uint32_t total_num_dispatches = 64;
  constexpr std::uint32_t n = aie_vector_scalar_kernel::element_count;

  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(min_queue_size, total_num_dispatches);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t kernel_object = LoadAddKernel();
  ASSERT_NE(kernel_object, 0u);
  // A second kernel whose instruction sequence the NPU cannot execute.
  const auto bad_bytes = BadInstsHsaco();
  ASSERT_FALSE(bad_bytes.empty()) << "could not build a corrupted copy of " << kHsacoPath;
  loaded_hsaco bad_hsaco;
  ASSERT_TRUE(bad_hsaco.load(bad_bytes, aie_agents.front()));
  const std::uint64_t bad_kernel = bad_hsaco.kernel_object(kHsacoKernelName);
  ASSERT_NE(bad_kernel, 0u);

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool,
                                         aie_vector_scalar_kernel::element_bytes *
                                             total_num_dispatches,
                                         0, reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool,
                                         aie_vector_scalar_kernel::element_bytes *
                                             total_num_dispatches,
                                         0, reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  const std::size_t total_element_count = n * total_num_dispatches;
  std::iota(input, input + total_element_count, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(output, total_element_count, sentinel);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // One doorbell for the whole batch, so the runtime splits it itself.
  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < total_num_dispatches; ++i) {
    const std::uint64_t kernel = (i == total_num_dispatches - 1) ? bad_kernel : kernel_object;
    wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        kernel, input + i * n, output + i * n,
        kernargs + i * aie_vector_scalar_kernel::num_kernargs_sizes, signal, queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  EXPECT_TRUE(err.invoked.load(std::memory_order_acquire))
      << "a partially failed batch did not report through the queue error callback";
  EXPECT_NE(err.status, HSA_STATUS_SUCCESS);

  // Partial: the chains that ran are retired, the failing one is not.
  const hsa_signal_value_t remaining = hsa_signal_load_scacquire(signal);
  ASSERT_LT(remaining, total_num_dispatches)
      << "no packet was retired, so a waiter on a dispatch that did run would block forever";
  ASSERT_GT(remaining, 0) << "the failing packet was credited with a completion it never reached";

  const std::uint32_t retired = total_num_dispatches - static_cast<std::uint32_t>(remaining);

  // The ring agrees with the signal: the packets that ran are consumed, the failing one and
  // everything behind it are not. This queue started empty, so the read index is the count.
  EXPECT_EQ(hsa_queue_load_read_index_scacquire(queue), retired)
      << "read index does not match the packets that were completed";
  EXPECT_LT(hsa_queue_load_read_index_scacquire(queue), hsa_queue_load_write_index_scacquire(queue))
      << "the failing packet was consumed from the ring";

  for (std::uint32_t d = 0; d < retired; ++d) {
    SCOPED_TRACE(d);
    for (std::size_t e = 0; e < n; ++e) {
      ASSERT_EQ(output[d * n + e], input[d * n + e] + 1)
          << "retired a packet that did not run, at element " << e;
    }
  }
  // The failing dispatch never ran, so it wrote nothing.
  for (std::size_t e = 0; e < n; ++e) {
    ASSERT_EQ(output[(total_num_dispatches - 1) * n + e], sentinel)
        << "the failing dispatch wrote output at element " << e;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
}

// The PDI + instruction sequence path gives every distinct PDI a compute-unit slot in the queue's
// hardware context, and the CU mask driving it is 32 bits wide, so 32 distinct PDIs is the
// ceiling. Each new PDI tears the context down and rebuilds it with one more CU configured, so
// walking all the way to the ceiling checks that the rebuild keeps producing a context that can
// still run the kernel -- not just that the cache accepted the entry.
//
// Counterpart to FullElfDispatchTest.ElfNoPdiCeiling, which shows the full-ELF path has no such
// limit because it loads the PDI from the control code instead of spending a CU slot on it.
TEST_F(DispatchTest, PdiCacheHoldsThirtyTwo) {
  constexpr std::uint32_t num_pdis = 32;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;

  // Distinct BOs holding identical PDI bytes: the cache keys on the BO handle, so these count as
  // 32 separate PDIs even though the design is the same one every time. Each load of the hsaco
  // places its own copy.
  std::vector<std::uint64_t> kernels(num_pdis);
  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    kernels[i] = LoadAddKernel();
    ASSERT_NE(kernels[i], 0u);
  }

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  auto* args = kernargs.as<std::uint64_t>();
  std::iota(in, in + aie_vector_scalar_kernel::element_count, 0);

  // One dispatch per PDI, each waited on before the next: every iteration adds a cache entry and
  // forces a context rebuild, and the result shows the rebuilt context still runs the kernel.
  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, kernels[i], in, out, args));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MulSingleDispatch) {
  // The second kernel on its own. Without this, a failure in the interleaving tests below cannot
  // be told apart from the new design simply not working.
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  if (!mul_hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kMulHsacoPath;
  const std::uint64_t mul_kernel = LoadMulKernel();
  ASSERT_NE(mul_kernel, 0u);

  pool_buffer inout, kernargs;
  ASSERT_EQ(inout.allocate(data_pool, aie_vector_scalar_mul_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_mul_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  ASSERT_NO_FATAL_FAILURE(DispatchMulAndVerify(queue, mul_kernel, inout.as<std::uint32_t>(),
                                               kernargs.as<std::uint64_t>()));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Two different designs on one queue. Each holds a compute-unit slot in the shared hardware
// context and is selected per command by its CU mask, so getting this wrong shows up as one
// kernel's packets running the other's design -- which the differing results make visible.
//
// The two also carry different argument counts (two kernargs against one), so the commands are
// not even the same size in the chain the driver packs them into.
TEST_F(DispatchTest, InterleavedKernels) {
  constexpr std::uint32_t num_pairs = 4;
  // One stride `n` walks both designs' buffers below, which is only valid while they agree.
  static_assert(aie_vector_scalar_kernel::element_count ==
                aie_vector_scalar_mul_kernel::element_count);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  // enqueue_aie_packet spins until a slot frees and the doorbell is only rung after the whole
  // batch is staged, so a ring smaller than the batch would hang rather than fail.
  ASSERT_GE(queue->size, 2 * num_pairs);

  if (!hsaco_available() || !mul_hsaco_available()) GTEST_SKIP() << "hsacos were not built";
  const std::uint64_t add_kernel = LoadAddKernel();
  ASSERT_NE(add_kernel, 0u);
  const std::uint64_t mul_kernel = LoadMulKernel();
  ASSERT_NE(mul_kernel, 0u);

  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;
  pool_buffer add_in, add_out, add_kernargs, mul_inout, mul_kernargs;
  ASSERT_EQ(add_in.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_out.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      add_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * num_pairs),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_inout.allocate(data_pool, aie_vector_scalar_mul_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      mul_kernargs.allocate(kernarg_pool, aie_vector_scalar_mul_kernel::kernarg_bytes * num_pairs),
      HSA_STATUS_SUCCESS);

  auto* ain = add_in.as<std::uint32_t>();
  auto* aout = add_out.as<std::uint32_t>();
  auto* mio = mul_inout.as<std::uint32_t>();
  std::iota(ain, ain + n * num_pairs, 0);
  std::fill_n(aout, n * num_pairs, 0);
  std::iota(mio, mio + n * num_pairs, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2 * num_pairs, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // Alternate the two designs in the ring and ring the doorbell once, so the runtime sees the
  // whole mixed batch at one time and packs it into one chain.
  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < num_pairs; ++i) {
    wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        add_kernel, ain + i * n, aout + i * n,
        add_kernargs.as<std::uint64_t>() + i * aie_vector_scalar_kernel::num_kernargs_sizes, signal,
        queue);
    wr_idx = aie_vector_scalar_mul_kernel::dispatch_packet(
        mul_kernel, mio + i * n,
        mul_kernargs.as<std::uint64_t>() + i * aie_vector_scalar_mul_kernel::num_kernargs_sizes,
        signal, queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  ASSERT_NO_FATAL_FAILURE(VerifyAdd(ain, aout, n * num_pairs));
  ASSERT_NO_FATAL_FAILURE(VerifyMul(mio, n * num_pairs, 0));

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// The same two designs alternating across separate doorbells rather than in one chain. The first
// round introduces both PDIs and rebuilds the hardware context twice; every round after that
// finds them cached and must skip the rebuild, still selecting the right one per packet.
TEST_F(DispatchTest, InterleavedKernelsSeparateBatches) {
  constexpr std::uint32_t num_rounds = 4;
  static_assert(aie_vector_scalar_kernel::element_count ==
                aie_vector_scalar_mul_kernel::element_count);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  if (!hsaco_available() || !mul_hsaco_available()) GTEST_SKIP() << "hsacos were not built";
  const std::uint64_t add_kernel = LoadAddKernel();
  ASSERT_NE(add_kernel, 0u);
  const std::uint64_t mul_kernel = LoadMulKernel();
  ASSERT_NE(mul_kernel, 0u);

  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;
  pool_buffer add_in, add_out, add_kernargs, mul_inout, mul_kernargs;
  ASSERT_EQ(add_in.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_out.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_inout.allocate(data_pool, aie_vector_scalar_mul_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_kernargs.allocate(kernarg_pool, aie_vector_scalar_mul_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  auto* ain = add_in.as<std::uint32_t>();
  auto* aout = add_out.as<std::uint32_t>();
  auto* mio = mul_inout.as<std::uint32_t>();
  std::iota(ain, ain + n, 0);

  for (std::uint32_t round = 0; round < num_rounds; ++round) {
    SCOPED_TRACE(round);
    ASSERT_NO_FATAL_FAILURE(
        DispatchAddAndVerify(queue, add_kernel, ain, aout, add_kernargs.as<std::uint64_t>()));
    ASSERT_NO_FATAL_FAILURE(
        DispatchMulAndVerify(queue, mul_kernel, mio, mul_kernargs.as<std::uint64_t>()));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Adds the second design's full-ELF kernel on top of FullElfDispatchTest's.
class FullElfInterleaveTest : public FullElfDispatchTest {
 protected:
  std::uint64_t elf_mul_kernel = 0;

  void SetUp() override {
    FullElfDispatchTest::SetUp();
    if (::testing::Test::HasFatalFailure() || ::testing::Test::IsSkipped()) return;

    if (!mul_elf_hsaco_available()) {
      GTEST_SKIP() << "full-ELF hsaco was not built: " << kMulElfHsacoPath;
    }
    elf_mul_kernel = LoadKernel(kMulElfHsacoPath, kMulElfHsacoKernelName);
    ASSERT_NE(elf_mul_kernel, 0u);
  }
};

// The full-ELF counterpart of DispatchTest.InterleavedKernels. This path has no PDI cache and no
// CU masks -- each packet gets its own control-code copy, which carries both its arguments and the
// PDI it loads -- so what is being checked is the other half of interleaving: that alternating
// designs in one chain each run against their own control code and PDI rather than the previous
// packet's.
TEST_F(FullElfInterleaveTest, ElfInterleavedKernels) {
  constexpr std::uint32_t num_pairs = 4;
  constexpr std::size_t n = aie_full_elf_kernel::element_count;
  static_assert(aie_full_elf_kernel::element_count == aie_full_elf_mul_kernel::element_count);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  // enqueue_aie_packet spins until a slot frees and the doorbell is only rung after the whole
  // batch is staged, so a ring smaller than the batch would hang rather than fail.
  ASSERT_GE(queue->size, 2 * num_pairs);

  pool_buffer add_in, add_out, add_kernargs, mul_inout, mul_kernargs;
  ASSERT_EQ(add_in.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_out.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_inout.allocate(data_pool, aie_full_elf_mul_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_kernargs.allocate(kernarg_pool, aie_full_elf_mul_kernel::kernarg_bytes * num_pairs),
            HSA_STATUS_SUCCESS);

  auto* ain = add_in.as<std::uint32_t>();
  auto* aout = add_out.as<std::uint32_t>();
  auto* mio = mul_inout.as<std::uint32_t>();
  std::iota(ain, ain + n * num_pairs, 0);
  std::fill_n(aout, n * num_pairs, 0);
  std::iota(mio, mio + n * num_pairs, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2 * num_pairs, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < num_pairs; ++i) {
    wr_idx = aie_full_elf_kernel::dispatch_packet(
        elf_kernel, ain + i * n, aout + i * n,
        add_kernargs.as<std::uint64_t>() + i * aie_full_elf_kernel::num_kernargs_sizes, signal,
        queue);
    wr_idx = aie_full_elf_mul_kernel::dispatch_packet(
        elf_mul_kernel, mio + i * n,
        mul_kernargs.as<std::uint64_t>() + i * aie_full_elf_mul_kernel::num_kernargs_sizes, signal,
        queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  ASSERT_NO_FATAL_FAILURE(VerifyAdd(ain, aout, n * num_pairs));
  ASSERT_NO_FATAL_FAILURE(VerifyMul(mio, n * num_pairs, 0));

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Modes alternate across batches, several packets at a time. Each batch is one mode, rung and
// waited on before the next is staged, so the queue is idle whenever the runtime rebuilds the
// hardware context for the incoming mode.
//
// Each batch carries several packets, so the switch is checked with a populated PDI cache to drop
// rather than with a single packet on either side.
TEST_F(FullElfDispatchTest, ModeSwitchAlternatingBatches) {
  static_assert(aie_vector_scalar_kernel::element_count == aie_full_elf_kernel::element_count);
  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;
  constexpr std::uint32_t pkts_per_batch = 2;
  // true = PDI + instruction sequence, false = full ELF. Starts and ends on different modes so
  // both switch directions are exercised twice.
  constexpr bool batch_is_pdi[] = {true, false, true, false};

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(queue->size, pkts_per_batch);

  if (!hsaco_available()) GTEST_SKIP() << "hsaco was not built: " << kHsacoPath;
  const std::uint64_t add_kernel = LoadAddKernel();
  ASSERT_NE(add_kernel, 0u);

  pool_buffer input, output, pdi_kernargs, elf_kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * pkts_per_batch),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * pkts_per_batch),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      pdi_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * pkts_per_batch),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      elf_kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * pkts_per_batch),
      HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + n * pkts_per_batch, 0);

  for (std::uint32_t b = 0; b < std::size(batch_is_pdi); ++b) {
    SCOPED_TRACE(b);
    std::fill_n(out, n * pkts_per_batch, 0);

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(pkts_per_batch, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

    std::uint64_t wr_idx = 0;
    for (std::uint32_t i = 0; i < pkts_per_batch; ++i) {
      wr_idx = batch_is_pdi[b]
          ? aie_vector_scalar_kernel::dispatch_packet(
                add_kernel, in + i * n, out + i * n,
                pdi_kernargs.as<std::uint64_t>() + i * aie_vector_scalar_kernel::num_kernargs_sizes,
                signal, queue)
          : aie_full_elf_kernel::dispatch_packet(
                elf_kernel, in + i * n, out + i * n,
                elf_kernargs.as<std::uint64_t>() + i * aie_full_elf_kernel::num_kernargs_sizes,
                signal, queue);
    }
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

    // Both designs add one, so a batch is checked the same way whichever mode ran it.
    ASSERT_NO_FATAL_FAILURE(VerifyAdd(in, out, n * pkts_per_batch));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}
