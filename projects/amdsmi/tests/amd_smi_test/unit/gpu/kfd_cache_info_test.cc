// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests KFDNode::get_cache_entries() and the legacy get_cache_info() against a
// mock KFD topology (via SetKFDNodesPathRootForTesting()). KFD publishes one
// caches/<idx> record per cache instance. get_cache_entries() folds consecutive
// records into one entry per cache kind (level, type, size, line size) no matter
// how CU harvesting varies the sharing between instances; get_cache_info() keeps
// its released grouping by level, type, size and sharing. No GPU required.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_kfd.h"

namespace {

// Creates a unique directory on construction, recursively removes it on destruction.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string tmpl = (std::filesystem::temp_directory_path() / "kfd_cache_test_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (char* result = mkdtemp(buf.data())) {
      path_ = result;
    }
  }
  ~ScopedTempDir() {
    if (!path_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
    }
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// Points KFDNodesPathRoot() at a mock topology on construction, restores the real
// path on destruction.
class ScopedKfdRootOverride {
 public:
  explicit ScopedKfdRootOverride(const std::string& path) : path_(path) {
    amd::smi::SetKFDNodesPathRootForTesting(path_.c_str());
  }
  ~ScopedKfdRootOverride() { amd::smi::SetKFDNodesPathRootForTesting(nullptr); }

 private:
  std::string path_;
};

// KFD cache "type" bits.
constexpr uint32_t kData = 0x1;
constexpr uint32_t kInst = 0x2;
constexpr uint32_t kHsaCu = 0x8;

struct CacheRecord {
  uint32_t level;
  uint32_t type;
  uint32_t size_kb;
  uint32_t cu_shared;         // number of set bits in sibling_map
  uint32_t line_bytes = 128;  // cache_line_size
};

void WriteFile(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << content;
}

// Writes GPU node `node_idx` with one caches/<idx>/properties file per record,
// in the same format KFD uses (sibling_map is a comma-separated bit list).
void WriteGpuNodeWithCaches(const std::filesystem::path& root, uint32_t node_idx,
                            const std::vector<CacheRecord>& caches) {
  const auto node = root / std::to_string(node_idx);
  std::filesystem::create_directories(node / "io_links");  // Initialize() requires it
  WriteFile(node / "gpu_id", "4242\n");
  WriteFile(node / "properties",
            "hive_id 0\nsimd_arrays_per_engine 1\narray_count 1\n"
            "cu_per_simd_array 1\ncaches_count " +
                std::to_string(caches.size()) + "\n");
  for (size_t i = 0; i < caches.size(); i++) {
    std::ostringstream sibling_map;
    for (uint32_t bit = 0; bit < 32; bit++) {
      sibling_map << (bit < caches[i].cu_shared ? "1" : "0") << (bit < 31 ? "," : "");
    }
    std::ostringstream props;
    props << "processor_id_low 2147487744\n"
          << "level " << caches[i].level << "\n"
          << "size " << caches[i].size_kb << "\n"
          << "cache_line_size " << caches[i].line_bytes << "\n"
          << "cache_lines_per_tag 0\n"
          << "association 0\n"
          << "latency 0\n"
          << "type " << caches[i].type << "\n"
          << "sibling_map " << sibling_map.str() << "\n";
    WriteFile(node / "caches" / std::to_string(i) / "properties", props.str());
  }
}

// Expected summary of one cache kind.
struct Expected {
  uint32_t level;
  uint32_t type;
  uint32_t size_kb;
  uint32_t line_bytes;
  uint32_t min_cu_shared;
  uint32_t max_cu_shared;
  uint32_t instances;
};

void ExpectEntries(const std::vector<amd::smi::KFDCacheEntry>& entries,
                   const std::vector<Expected>& expected) {
  ASSERT_EQ(entries.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    SCOPED_TRACE("cache entry " + std::to_string(i));
    EXPECT_EQ(entries[i].level, expected[i].level);
    EXPECT_EQ(entries[i].type, expected[i].type);
    EXPECT_EQ(entries[i].size_kb, expected[i].size_kb);
    EXPECT_EQ(entries[i].line_size, expected[i].line_bytes);
    EXPECT_EQ(entries[i].min_cu_shared, expected[i].min_cu_shared);
    EXPECT_EQ(entries[i].max_cu_shared, expected[i].max_cu_shared);
    EXPECT_EQ(entries[i].num_instances, expected[i].instances);
  }
}

// Shaped like a harvested node with a GL1: per-CU vector L1, WGP-shared SQC
// instruction/scalar caches with unpaired CUs at the end of each CU block, a
// per-SA GL1 whose sharing follows each SA's active CU count, and one L2.
// Instances that differ only in sharing must be one entry carrying the sharing
// range; the legacy fixed-size form still reports one entry per sharing count,
// and merges the vector L1D with the unpaired scalar L1D of the same size.
TEST(GpuUnit, KfdCacheInfoGroupsInstancesByCacheKind) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuNodeWithCaches(root.path(), 2,
                         {
                             {1, kData | kHsaCu, 16, 1},  // vector L1D, per CU
                             {1, kData | kHsaCu, 16, 1},
                             {1, kData | kHsaCu, 16, 1},
                             {1, kInst | kHsaCu, 32, 2},  // L1I, WGP pair
                             {1, kInst | kHsaCu, 32, 2},
                             {1, kInst | kHsaCu, 32, 1},   // L1I, unpaired CU
                             {1, kData | kHsaCu, 16, 2},   // scalar L1D, WGP pair
                             {1, kData | kHsaCu, 16, 1},   // scalar L1D, unpaired CU
                             {1, kData | kHsaCu, 128, 9},  // GL1, per SA
                             {1, kData | kHsaCu, 128, 7},
                             {1, kData | kHsaCu, 128, 8},
                             {1, kData | kHsaCu, 128, 9},
                             {2, kData | kHsaCu, 4096, 30},  // L2
                         });

  amd::smi::KFDNode node(2);
  ASSERT_EQ(node.Initialize(), 0);

  std::vector<amd::smi::KFDCacheEntry> entries;
  ASSERT_EQ(node.get_cache_entries(&entries), 0);
  ExpectEntries(entries, {
                             {1, kData | kHsaCu, 16, 128, 1, 1, 3},
                             {1, kInst | kHsaCu, 32, 128, 1, 2, 3},
                             {1, kData | kHsaCu, 16, 128, 1, 2, 2},
                             {1, kData | kHsaCu, 128, 128, 7, 9, 4},
                             {2, kData | kHsaCu, 4096, 128, 30, 30, 1},
                         });

  // {level, type, size, max_num_cu_shared, num_cache_instance}
  const std::vector<std::vector<uint32_t>> legacy = {
      {1, kData | kHsaCu, 16, 1, 4},  {1, kInst | kHsaCu, 32, 2, 2},
      {1, kInst | kHsaCu, 32, 1, 1},  {1, kData | kHsaCu, 16, 2, 1},
      {1, kData | kHsaCu, 128, 9, 2}, {1, kData | kHsaCu, 128, 7, 1},
      {1, kData | kHsaCu, 128, 8, 1}, {2, kData | kHsaCu, 4096, 30, 1},
  };
  rsmi_gpu_cache_info_t info = {};
  ASSERT_EQ(node.get_cache_info(&info), 0);
  ASSERT_EQ(info.num_cache_types, legacy.size());
  for (size_t i = 0; i < legacy.size(); i++) {
    SCOPED_TRACE("legacy cache entry " + std::to_string(i));
    EXPECT_EQ(info.cache[i].cache_level, legacy[i][0]);
    EXPECT_EQ(info.cache[i].flags, legacy[i][1]);
    EXPECT_EQ(info.cache[i].cache_size_kb, legacy[i][2]);
    EXPECT_EQ(info.cache[i].max_num_cu_shared, legacy[i][3]);
    EXPECT_EQ(info.cache[i].num_cache_instance, legacy[i][4]);
  }
}

// Harvesting can give one cache kind more distinct sharing counts than the
// legacy struct has entries. get_cache_entries() still folds them into a single
// entry; the fixed-size get_cache_info() keeps failing as released.
TEST(GpuUnit, KfdCacheInfoSharingVarietyDoesNotExhaustEntries) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  std::vector<CacheRecord> caches;
  for (uint32_t cu_shared = 1; cu_shared <= RSMI_MAX_CACHE_TYPES + 2; cu_shared++) {
    caches.push_back({1, kData | kHsaCu, 256, cu_shared});
  }
  caches.push_back({2, kData | kHsaCu, 4096, 30});
  WriteGpuNodeWithCaches(root.path(), 3, caches);

  amd::smi::KFDNode node(3);
  ASSERT_EQ(node.Initialize(), 0);

  std::vector<amd::smi::KFDCacheEntry> entries;
  ASSERT_EQ(node.get_cache_entries(&entries), 0);
  ExpectEntries(entries, {
                             {1, kData | kHsaCu, 256, 128, 1, RSMI_MAX_CACHE_TYPES + 2,
                              RSMI_MAX_CACHE_TYPES + 2},
                             {2, kData | kHsaCu, 4096, 128, 30, 30, 1},
                         });

  rsmi_gpu_cache_info_t info = {};
  EXPECT_NE(node.get_cache_info(&info), 0);
}

// More distinct cache kinds than the legacy struct holds: get_cache_entries()
// returns all of them, while the fixed-size get_cache_info() still refuses
// rather than silently dropping kinds.
TEST(GpuUnit, KfdCacheEntriesHaveNoCountLimit) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  constexpr uint32_t kKinds = RSMI_MAX_CACHE_TYPES + 3;
  std::vector<CacheRecord> caches;
  std::vector<Expected> expected;
  for (uint32_t kind = 0; kind < kKinds; kind++) {
    const uint32_t size_kb = 16 * (kind + 1);
    caches.push_back({1, kData | kHsaCu, size_kb, 2});
    caches.push_back({1, kData | kHsaCu, size_kb, 1});
    expected.push_back({1, kData | kHsaCu, size_kb, 128, 1, 2, 2});
  }
  WriteGpuNodeWithCaches(root.path(), 4, caches);

  amd::smi::KFDNode node(4);
  ASSERT_EQ(node.Initialize(), 0);

  std::vector<amd::smi::KFDCacheEntry> entries;
  ASSERT_EQ(node.get_cache_entries(&entries), 0);
  ExpectEntries(entries, expected);

  rsmi_gpu_cache_info_t info = {};
  EXPECT_NE(node.get_cache_info(&info), 0);
}

// Distinct cache kinds can have the same level, type and size. On gfx1030 the
// vector L1D (128 B lines) and scalar L1D (64 B lines) are both 16 KB DATA at
// level 1, with the instruction cache between them. Records are only folded into
// the entry right before them, and only if the line size also matches, so the
// two stay separate whether or not another kind sits between them.
TEST(GpuUnit, KfdCacheInfoKeepsSameSizeKindsSeparate) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuNodeWithCaches(root.path(), 1,
                         {
                             {1, kData | kHsaCu, 16, 1, 128},  // vector L1D
                             {1, kData | kHsaCu, 16, 1, 128},
                             {1, kInst | kHsaCu, 32, 2, 64},  // L1I
                             {1, kData | kHsaCu, 16, 2, 64},  // scalar L1D
                             {1, kData | kHsaCu, 16, 2, 64},
                         });
  WriteGpuNodeWithCaches(root.path(), 2,
                         {
                             {1, kData | kHsaCu, 16, 1, 128},  // vector L1D
                             {1, kData | kHsaCu, 16, 2, 64},   // scalar L1D, no L1I between
                         });

  amd::smi::KFDNode separated(1);
  ASSERT_EQ(separated.Initialize(), 0);
  std::vector<amd::smi::KFDCacheEntry> entries;
  ASSERT_EQ(separated.get_cache_entries(&entries), 0);
  ExpectEntries(entries, {
                             {1, kData | kHsaCu, 16, 128, 1, 1, 2},
                             {1, kInst | kHsaCu, 32, 64, 2, 2, 1},
                             {1, kData | kHsaCu, 16, 64, 2, 2, 2},
                         });

  amd::smi::KFDNode adjacent(2);
  ASSERT_EQ(adjacent.Initialize(), 0);
  ASSERT_EQ(adjacent.get_cache_entries(&entries), 0);
  ExpectEntries(entries, {
                             {1, kData | kHsaCu, 16, 128, 1, 1, 1},
                             {1, kData | kHsaCu, 16, 64, 2, 2, 1},
                         });
}

}  // namespace
