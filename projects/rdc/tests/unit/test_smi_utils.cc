/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

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

// Hardware-aware tests for the flat GPU handle table and partition helpers that
// PR "CPX partition mode support" reworked in SmiUtils.cc. They initialize
// amdsmi directly (no rdcd, no grpc) and validate the invariants the CPX rewrite
// depends on:
//   * the flat table enumerates every AMD GPU processor exactly once,
//   * a flat index resolves to the same handle via the fast (table) path and the
//     entity-encoded (socket/proc) path,
//   * get_num_partition() always resolves against the physical GPU and returns a
//     sane count (>= 1) for every entry, and matches across partitions of one GPU,
//   * reset_flat_gpu_table() rebuilds an identical table.
//
// If no AMD GPU is present the hardware tests self-skip so the binary still
// passes on GPU-less CI. The codec tests in test_entity_codec.cc always run.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcEntityCodec.h"
#include "rdc_lib/impl/SmiUtils.h"

using amd::rdc::get_flat_gpu_table;
using amd::rdc::get_num_partition;
using amd::rdc::get_processor_handle_from_id;
using amd::rdc::GpuHandleEntry;
using amd::rdc::reset_flat_gpu_table;

namespace {

// One-time amdsmi init/shutdown for the whole binary. Mirrors RdcEmbeddedHandler's
// smi_initializer (defensive shut_down first, then init with GPUs).
class SmiEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    amdsmi_shut_down();
    amdsmi_status_t ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    initialized_ = (ret == AMDSMI_STATUS_SUCCESS);
    if (!initialized_) {
      // Not fatal: hardware tests will self-skip. Codec tests still run.
      std::cerr << "[  NOTE    ] amdsmi_init failed (" << ret
                << "); GPU-dependent tests will be skipped." << std::endl;
    }
  }
  void TearDown() override {
    if (initialized_) {
      amdsmi_shut_down();
    }
  }
  static bool initialized_;
};
bool SmiEnvironment::initialized_ = false;

// Build the encoded entity index for a partition instance (socket, proc).
uint32_t PartitionEntityIndex(uint32_t socket, uint32_t proc) {
  rdc_entity_info_t info{};
  info.device_index = socket;
  info.instance_index = proc;
  info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
  info.device_type = RDC_DEVICE_TYPE_GPU;
  return rdc_get_entity_index_from_info(info);
}

class SmiUtilsHwTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!SmiEnvironment::initialized_) {
      GTEST_SKIP() << "amdsmi not initialized";
    }
    if (get_flat_gpu_table().empty()) {
      GTEST_SKIP() << "no AMD GPUs present";
    }
  }
};

// The flat table must contain every enumerated AMD GPU processor once, with
// unique handles and consistent socket/proc bookkeeping.
TEST_F(SmiUtilsHwTest, FlatTableEnumeratesUniqueGpus) {
  const auto& table = get_flat_gpu_table();
  ASSERT_FALSE(table.empty());

  std::set<amdsmi_processor_handle> handles;
  for (const auto& e : table) {
    EXPECT_NE(e.handle, nullptr);
    EXPECT_TRUE(handles.insert(e.handle).second) << "duplicate handle in flat table";

    // Each entry must actually be an AMD GPU.
    processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    ASSERT_EQ(amdsmi_get_processor_type(e.handle, &type), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(type, AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  }
  std::cout << "[  INFO    ] flat GPU table size = " << table.size() << std::endl;
}

// A flat index in range must resolve; out of range must be rejected cleanly.
TEST_F(SmiUtilsHwTest, ProcessorHandleFromFlatIndex) {
  const auto& table = get_flat_gpu_table();

  for (uint32_t i = 0; i < table.size(); i++) {
    amdsmi_processor_handle h = nullptr;
    EXPECT_EQ(get_processor_handle_from_id(i, &h), AMDSMI_STATUS_SUCCESS);
    // The fast path returns exactly the table's cached handle.
    EXPECT_EQ(h, table[i].handle);
  }

  amdsmi_processor_handle h = nullptr;
  EXPECT_EQ(get_processor_handle_from_id(static_cast<uint32_t>(table.size()), &h),
            AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
}

// The entity-encoded (socket/proc) path must resolve to the same handle the flat
// table records for that socket/proc. This is the invariant CPX relies on: both
// index encodings name the same physical processor.
TEST_F(SmiUtilsHwTest, EntityEncodedPathMatchesFlatTable) {
  const auto& table = get_flat_gpu_table();

  for (const auto& e : table) {
    uint32_t entity = PartitionEntityIndex(e.socket_index, e.proc_index);
    amdsmi_processor_handle h = nullptr;
    ASSERT_EQ(get_processor_handle_from_id(entity, &h), AMDSMI_STATUS_SUCCESS)
        << "socket=" << e.socket_index << " proc=" << e.proc_index;
    EXPECT_EQ(h, e.handle) << "socket=" << e.socket_index << " proc=" << e.proc_index;
  }
}

// An entity-encoded index with an impossible socket must be rejected, not crash.
TEST_F(SmiUtilsHwTest, EntityEncodedOutOfRangeSocketRejected) {
  uint32_t entity = PartitionEntityIndex(/*socket=*/RDC_ENTITY_DEVICE_MASK, /*proc=*/0);
  amdsmi_processor_handle h = nullptr;
  EXPECT_EQ(get_processor_handle_from_id(entity, &h), AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
}

// get_num_partition() must succeed and report >= 1 for every flat index, and it
// must reject an out-of-range index and a null out-pointer.
TEST_F(SmiUtilsHwTest, NumPartitionSaneForEveryGpu) {
  const auto& table = get_flat_gpu_table();

  for (uint32_t i = 0; i < table.size(); i++) {
    uint16_t n = 0;
    amdsmi_status_t ret = get_num_partition(i, &n);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "index " << i;
    EXPECT_GE(n, 1) << "index " << i << " reported zero partitions";
    EXPECT_LE(n, RDC_MAX_NUM_PARTITIONS) << "index " << i;
  }

  uint16_t n = 0;
  EXPECT_EQ(get_num_partition(static_cast<uint32_t>(table.size()), &n),
            AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
  EXPECT_EQ(get_num_partition(0, nullptr), AMDSMI_STATUS_INVAL);
}

// All flat entries that share a socket describe partitions of one physical GPU,
// so get_num_partition() must return an identical count for all of them. This is
// the core CPX guarantee: querying any partition resolves the physical profile.
TEST_F(SmiUtilsHwTest, NumPartitionConsistentWithinSocket) {
  const auto& table = get_flat_gpu_table();

  // socket_index -> (count of table entries, first partition count seen)
  std::vector<int> per_socket_entries(RDC_ENTITY_DEVICE_MASK + 1, 0);
  std::vector<uint16_t> per_socket_partcount(RDC_ENTITY_DEVICE_MASK + 1, 0);

  for (uint32_t i = 0; i < table.size(); i++) {
    uint16_t n = 0;
    ASSERT_EQ(get_num_partition(i, &n), AMDSMI_STATUS_SUCCESS);
    uint32_t s = table[i].socket_index;
    if (per_socket_entries[s] == 0) {
      per_socket_partcount[s] = n;
    } else {
      EXPECT_EQ(n, per_socket_partcount[s]) << "partition count differs within socket " << s;
    }
    per_socket_entries[s]++;
  }

  // The flat table enumerates one processor per partition, so the number of flat
  // entries in a socket must equal the partition count that socket reports. This
  // is the core CPX guarantee and is verified on real hardware across:
  //   non-partitionable / SPX: 1 processor per socket, count 1  (1 == 1)
  //   MI350X CPX:              8 processors per socket, count 8  (8 == 8)
  // A divergence here means the flat table and the partition profile disagree
  // about how the GPU is split, which is exactly the class of bug this PR fixes.
  for (uint32_t s = 0; s < per_socket_entries.size(); s++) {
    if (per_socket_entries[s] > 0) {
      std::cout << "[  INFO    ] socket " << s << ": " << per_socket_entries[s]
                << " processor(s), get_num_partition=" << per_socket_partcount[s] << std::endl;
      EXPECT_EQ(static_cast<uint16_t>(per_socket_entries[s]), per_socket_partcount[s])
          << "socket " << s << " enumerates " << per_socket_entries[s]
          << " processor(s) but get_num_partition reports " << per_socket_partcount[s];
    }
  }
}

// reset_flat_gpu_table() must invalidate and rebuild an equivalent table.
TEST_F(SmiUtilsHwTest, ResetRebuildsEquivalentTable) {
  std::vector<GpuHandleEntry> before = get_flat_gpu_table();
  reset_flat_gpu_table();
  const auto& after = get_flat_gpu_table();

  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); i++) {
    EXPECT_EQ(before[i].handle, after[i].handle) << "entry " << i;
    EXPECT_EQ(before[i].socket_index, after[i].socket_index) << "entry " << i;
    EXPECT_EQ(before[i].proc_index, after[i].proc_index) << "entry " << i;
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SmiEnvironment());
  return RUN_ALL_TESTS();
}
