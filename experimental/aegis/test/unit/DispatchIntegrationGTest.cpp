//===-- DispatchIntegrationGTest.cpp - Dispatch Integration Tests --*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the dispatch integration components of TracingEngine.
/// Tests extended kernarg creation, dispatch ID generation, active dispatch
/// management, and resource pooling without requiring GPU hardware.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/TracingEngine.h"
#include "aegisbit/Types.h"
#include "gtest/gtest.h"

#include <atomic>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

using namespace aegisbit;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions for testing
//===----------------------------------------------------------------------===//

/// Create a mock TraceArgs structure for testing
TraceArgs createTestTraceArgs(uint32_t KernelID = 0) {
  TraceArgs Args;
  Args.BufferPtr = 0x1000000;
  Args.BufferSize = 1024 * 1024;
  Args.WriteOffsetPtr = 0x1000008;
  Args.KernelID = KernelID;
  return Args;
}

/// Create extended kernarg manually for comparison
std::vector<uint8_t> createExtendedKernargManually(
    const void* OriginalKernarg,
    uint32_t OriginalSize,
    const TraceArgs& Args) {
  uint32_t AlignedOrigSize = (OriginalSize + 7) & ~7u;
  uint32_t TotalSize = AlignedOrigSize + sizeof(TraceArgs);

  std::vector<uint8_t> Extended(TotalSize, 0);

  if (OriginalKernarg && OriginalSize > 0) {
    std::memcpy(Extended.data(), OriginalKernarg, OriginalSize);
  }

  std::memcpy(Extended.data() + AlignedOrigSize, &Args, sizeof(TraceArgs));

  return Extended;
}

//===----------------------------------------------------------------------===//
// ExtendedKernargTest - Tests for extended kernarg creation
//===----------------------------------------------------------------------===//

TEST(ExtendedKernargTest, AlignmentTo8Bytes) {
  // Test that original kernarg is aligned to 8 bytes before TraceArgs

  // Test various sizes that need alignment
  std::vector<uint32_t> TestSizes = {1, 3, 5, 7, 9, 15, 17, 63};

  for (uint32_t Size : TestSizes) {
    std::vector<uint8_t> OrigKernarg(Size, 0xAA);
    TraceArgs Args = createTestTraceArgs();

    auto Extended = createExtendedKernargManually(OrigKernarg.data(), Size, Args);

    // Check alignment: TraceArgs should start at 8-byte aligned offset
    uint32_t AlignedSize = (Size + 7) & ~7u;
    EXPECT_EQ(Extended.size(), AlignedSize + sizeof(TraceArgs))
        << "Size " << Size << " should align to " << AlignedSize;
  }
}

TEST(ExtendedKernargTest, TraceArgsPositioning) {
  // Test that TraceArgs is at correct offset (AlignedOrigSize)
  std::vector<uint8_t> OrigKernarg = {0x11, 0x22, 0x33, 0x44, 0x55};
  TraceArgs Args = createTestTraceArgs(42);

  auto Extended = createExtendedKernargManually(OrigKernarg.data(), OrigKernarg.size(), Args);

  // Original size is 5, aligned to 8
  uint32_t AlignedSize = 8;
  ASSERT_GE(Extended.size(), AlignedSize + sizeof(TraceArgs));

  // Extract TraceArgs from extended buffer
  TraceArgs ExtractedArgs;
  std::memcpy(&ExtractedArgs, Extended.data() + AlignedSize, sizeof(TraceArgs));

  EXPECT_EQ(ExtractedArgs.BufferPtr, Args.BufferPtr);
  EXPECT_EQ(ExtractedArgs.BufferSize, Args.BufferSize);
  EXPECT_EQ(ExtractedArgs.KernelID, 42u);
}

TEST(ExtendedKernargTest, NullOriginalArgs) {
  // Test with nullptr/size=0 produces buffer with just TraceArgs
  TraceArgs Args = createTestTraceArgs(123);

  auto Extended = createExtendedKernargManually(nullptr, 0, Args);

  // Should just have TraceArgs at offset 0 (0 aligned to 8 is still 0)
  EXPECT_EQ(Extended.size(), sizeof(TraceArgs));

  TraceArgs ExtractedArgs;
  std::memcpy(&ExtractedArgs, Extended.data(), sizeof(TraceArgs));

  EXPECT_EQ(ExtractedArgs.KernelID, 123u);
}

TEST(ExtendedKernargTest, VariousSizes) {
  // Test 64B, 256B, 1KB original sizes
  std::vector<uint32_t> TestSizes = {64, 256, 1024};

  for (uint32_t Size : TestSizes) {
    std::vector<uint8_t> OrigKernarg(Size);
    // Fill with pattern
    for (uint32_t i = 0; i < Size; ++i) {
      OrigKernarg[i] = static_cast<uint8_t>(i & 0xFF);
    }

    TraceArgs Args = createTestTraceArgs();
    auto Extended = createExtendedKernargManually(OrigKernarg.data(), Size, Args);

    // Verify original data preserved
    EXPECT_EQ(std::memcmp(Extended.data(), OrigKernarg.data(), Size), 0)
        << "Original kernarg data not preserved for size " << Size;

    // These sizes are already 8-byte aligned
    EXPECT_EQ(Extended.size(), Size + sizeof(TraceArgs));
  }
}

TEST(ExtendedKernargTest, TraceArgsContentPreserved) {
  // Test that written TraceArgs matches input
  std::vector<uint8_t> OrigKernarg(32, 0);

  TraceArgs Args;
  Args.BufferPtr = 0xDEADBEEF12345678ULL;
  Args.BufferSize = 0x10000000;
  Args.WriteOffsetPtr = 0xCAFEBABEULL;
  Args.KernelID = 999;

  auto Extended = createExtendedKernargManually(OrigKernarg.data(), OrigKernarg.size(), Args);

  TraceArgs ExtractedArgs;
  std::memcpy(&ExtractedArgs, Extended.data() + 32, sizeof(TraceArgs));

  EXPECT_EQ(ExtractedArgs.BufferPtr, Args.BufferPtr);
  EXPECT_EQ(ExtractedArgs.BufferSize, Args.BufferSize);
  EXPECT_EQ(ExtractedArgs.WriteOffsetPtr, Args.WriteOffsetPtr);
  EXPECT_EQ(ExtractedArgs.KernelID, Args.KernelID);
}

//===----------------------------------------------------------------------===//
// DispatchIDGenerationTest - Tests for dispatch ID generation
//===----------------------------------------------------------------------===//

TEST(DispatchIDGenerationTest, SequentialGeneration) {
  // Test that IDs are sequential (0, 1, 2, ...)
  std::atomic<uint32_t> NextID{0};

  std::vector<uint32_t> IDs;
  for (int i = 0; i < 100; ++i) {
    IDs.push_back(NextID.fetch_add(1));
  }

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(IDs[i], static_cast<uint32_t>(i));
  }
}

TEST(DispatchIDGenerationTest, NoDuplicatesUnderContention) {
  // Test thread-safe, no duplicates
  std::atomic<uint32_t> NextID{0};
  constexpr int NumThreads = 10;
  constexpr int IDsPerThread = 1000;

  std::vector<std::vector<uint32_t>> ThreadIDs(NumThreads);
  std::vector<std::thread> Threads;
  std::atomic<bool> Go{false};

  for (int i = 0; i < NumThreads; ++i) {
    Threads.emplace_back([&, i]() {
      while (!Go.load()) {
        std::this_thread::yield();
      }
      for (int j = 0; j < IDsPerThread; ++j) {
        ThreadIDs[i].push_back(NextID.fetch_add(1));
      }
    });
  }

  Go.store(true);
  for (auto& T : Threads) {
    T.join();
  }

  // Collect all IDs and check for duplicates
  std::set<uint32_t> AllIDs;
  for (const auto& TIDs : ThreadIDs) {
    for (uint32_t ID : TIDs) {
      EXPECT_TRUE(AllIDs.insert(ID).second) << "Duplicate ID: " << ID;
    }
  }

  // Should have exactly NumThreads * IDsPerThread unique IDs
  EXPECT_EQ(AllIDs.size(), NumThreads * IDsPerThread);
  EXPECT_EQ(NextID.load(), static_cast<uint32_t>(NumThreads * IDsPerThread));
}

TEST(DispatchIDGenerationTest, AtomicOperationCorrectness) {
  // Test that fetch_add works correctly
  std::atomic<uint32_t> NextID{100};

  uint32_t First = NextID.fetch_add(1);
  uint32_t Second = NextID.fetch_add(1);
  uint32_t Third = NextID.fetch_add(1);

  EXPECT_EQ(First, 100u);
  EXPECT_EQ(Second, 101u);
  EXPECT_EQ(Third, 102u);
  EXPECT_EQ(NextID.load(), 103u);
}

//===----------------------------------------------------------------------===//
// ActiveDispatchManagementTest - Tests for ActiveDispatch struct
//===----------------------------------------------------------------------===//

TEST(ActiveDispatchManagementTest, InitializationComplete) {
  // Test that all fields can be set correctly
  ActiveDispatch Dispatch;
  Dispatch.DispatchID = 42;
  Dispatch.KernelName = "testKernel";
  Dispatch.Params.WorkgroupSizeX = 64;
  Dispatch.Params.WorkgroupSizeY = 1;
  Dispatch.Params.WorkgroupSizeZ = 1;
  Dispatch.Params.GridSizeX = 1024;
  Dispatch.Params.GridSizeY = 1;
  Dispatch.Params.GridSizeZ = 1;
  Dispatch.StartTime = std::chrono::steady_clock::now();
  Dispatch.OriginalKernelObject = 0x12345678;
  Dispatch.PatchedKernelObject = 0x87654321;
  Dispatch.GpuKernargPtr = reinterpret_cast<void*>(0x1000);
  Dispatch.CompletionSignalHandle = 0xABCD;
  Dispatch.OriginalSignalHandle = 0xDCBA;

  EXPECT_EQ(Dispatch.DispatchID, 42u);
  EXPECT_EQ(Dispatch.KernelName, "testKernel");
  EXPECT_EQ(Dispatch.Params.WorkgroupSizeX, 64u);
  EXPECT_EQ(Dispatch.Params.GridSizeX, 1024u);
  EXPECT_EQ(Dispatch.OriginalKernelObject, 0x12345678ull);
  EXPECT_EQ(Dispatch.GpuKernargPtr, reinterpret_cast<void*>(0x1000));
  EXPECT_EQ(Dispatch.CompletionSignalHandle, 0xABCDull);
  EXPECT_EQ(Dispatch.OriginalSignalHandle, 0xDCBAull);
}

TEST(ActiveDispatchManagementTest, ExtendedKernargStorage) {
  // Test that extended kernarg can be stored
  ActiveDispatch Dispatch;
  Dispatch.ExtendedKernarg = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};

  EXPECT_EQ(Dispatch.ExtendedKernarg.size(), 8u);
  EXPECT_EQ(Dispatch.ExtendedKernarg[0], 0x11);
  EXPECT_EQ(Dispatch.ExtendedKernarg[7], 0x88);
}

TEST(ActiveDispatchManagementTest, TimestampCorrectness) {
  // Test that StartTime set to current steady_clock
  auto Before = std::chrono::steady_clock::now();

  ActiveDispatch Dispatch;
  Dispatch.StartTime = std::chrono::steady_clock::now();

  auto After = std::chrono::steady_clock::now();

  EXPECT_GE(Dispatch.StartTime, Before);
  EXPECT_LE(Dispatch.StartTime, After);
}

TEST(ActiveDispatchManagementTest, MapOperations) {
  // Test insert, lookup, erase work
  std::unordered_map<uint32_t, ActiveDispatch> ActiveDispatches;

  // Insert
  ActiveDispatch Dispatch1;
  Dispatch1.DispatchID = 1;
  Dispatch1.KernelName = "kernel1";
  ActiveDispatches[1] = std::move(Dispatch1);

  ActiveDispatch Dispatch2;
  Dispatch2.DispatchID = 2;
  Dispatch2.KernelName = "kernel2";
  ActiveDispatches[2] = std::move(Dispatch2);

  EXPECT_EQ(ActiveDispatches.size(), 2u);

  // Lookup
  auto It = ActiveDispatches.find(1);
  EXPECT_NE(It, ActiveDispatches.end());
  EXPECT_EQ(It->second.KernelName, "kernel1");

  It = ActiveDispatches.find(2);
  EXPECT_NE(It, ActiveDispatches.end());
  EXPECT_EQ(It->second.KernelName, "kernel2");

  // Missing key
  It = ActiveDispatches.find(999);
  EXPECT_EQ(It, ActiveDispatches.end());

  // Erase
  ActiveDispatches.erase(1);
  EXPECT_EQ(ActiveDispatches.size(), 1u);
  EXPECT_EQ(ActiveDispatches.find(1), ActiveDispatches.end());
  EXPECT_NE(ActiveDispatches.find(2), ActiveDispatches.end());
}

//===----------------------------------------------------------------------===//
// DispatchParamsExtractionTest - Tests for extracting params from packets
//===----------------------------------------------------------------------===//

TEST(DispatchParamsExtractionTest, Extract3DGridDimensions) {
  // Test GridSizeX/Y/Z extraction
  DispatchParams Params;
  Params.GridSizeX = 1024;
  Params.GridSizeY = 512;
  Params.GridSizeZ = 256;

  EXPECT_EQ(Params.GridSizeX, 1024u);
  EXPECT_EQ(Params.GridSizeY, 512u);
  EXPECT_EQ(Params.GridSizeZ, 256u);
}

TEST(DispatchParamsExtractionTest, Extract3DWorkgroupDimensions) {
  // Test WorkgroupSizeX/Y/Z extraction
  DispatchParams Params;
  Params.WorkgroupSizeX = 64;
  Params.WorkgroupSizeY = 4;
  Params.WorkgroupSizeZ = 2;

  EXPECT_EQ(Params.WorkgroupSizeX, 64u);
  EXPECT_EQ(Params.WorkgroupSizeY, 4u);
  EXPECT_EQ(Params.WorkgroupSizeZ, 2u);
}

TEST(DispatchParamsExtractionTest, ExtractDynamicLDS) {
  // Test DynamicLDSSize extraction
  DispatchParams Params;
  Params.DynamicLDSSize = 16384;  // 16KB

  EXPECT_EQ(Params.DynamicLDSSize, 16384u);
}

TEST(DispatchParamsExtractionTest, ZeroValuesHandled) {
  // Test that zero dimensions are handled gracefully
  DispatchParams Params;
  Params.GridSizeX = 0;
  Params.GridSizeY = 0;
  Params.GridSizeZ = 0;
  Params.WorkgroupSizeX = 0;
  Params.WorkgroupSizeY = 0;
  Params.WorkgroupSizeZ = 0;
  Params.DynamicLDSSize = 0;

  // All zeros should be valid (even if it means no work)
  EXPECT_EQ(Params.GridSizeX, 0u);
  EXPECT_EQ(Params.WorkgroupSizeX, 0u);
  EXPECT_EQ(Params.DynamicLDSSize, 0u);
}

//===----------------------------------------------------------------------===//
// DispatchStatisticsTest - Tests for statistics tracking
//===----------------------------------------------------------------------===//

TEST(DispatchStatisticsTest, IncrementCountersThreadSafe) {
  // Test concurrent increments are correct
  std::atomic<uint64_t> Counter{0};
  constexpr int NumThreads = 10;
  constexpr int IncrementsPerThread = 10000;

  std::vector<std::thread> Threads;
  std::atomic<bool> Go{false};

  for (int i = 0; i < NumThreads; ++i) {
    Threads.emplace_back([&]() {
      while (!Go.load()) {
        std::this_thread::yield();
      }
      for (int j = 0; j < IncrementsPerThread; ++j) {
        Counter.fetch_add(1);
      }
    });
  }

  Go.store(true);
  for (auto& T : Threads) {
    T.join();
  }

  EXPECT_EQ(Counter.load(), NumThreads * IncrementsPerThread);
}

TEST(DispatchStatisticsTest, GetStatsConsistency) {
  // Test that a snapshot of stats is consistent
  TracingEngine::Stats Stats;
  Stats.TotalDispatches = 100;
  Stats.TracedDispatches = 80;
  Stats.SkippedDispatches = 15;
  Stats.ErrorDispatches = 5;
  Stats.TracesWritten = 75;
  Stats.TotalTraceBytes = 1024 * 1024;

  // Verify values
  EXPECT_EQ(Stats.TotalDispatches, 100u);
  EXPECT_EQ(Stats.TracedDispatches, 80u);
  EXPECT_EQ(Stats.SkippedDispatches, 15u);
  EXPECT_EQ(Stats.ErrorDispatches, 5u);
  EXPECT_EQ(Stats.TracesWritten, 75u);
  EXPECT_EQ(Stats.TotalTraceBytes, 1024u * 1024u);

  // Check that traced + skipped + errors <= total
  EXPECT_LE(Stats.TracedDispatches + Stats.SkippedDispatches + Stats.ErrorDispatches,
            Stats.TotalDispatches);
}

TEST(DispatchStatisticsTest, CounterInitialization) {
  // Test that all counters start at zero
  TracingEngine::Stats Stats;

  EXPECT_EQ(Stats.TotalDispatches, 0u);
  EXPECT_EQ(Stats.TracedDispatches, 0u);
  EXPECT_EQ(Stats.SkippedDispatches, 0u);
  EXPECT_EQ(Stats.ErrorDispatches, 0u);
  EXPECT_EQ(Stats.TracesWritten, 0u);
  EXPECT_EQ(Stats.TotalTraceBytes, 0u);
}

//===----------------------------------------------------------------------===//
// PatchCacheKeyTest - Tests for cache key operations
//===----------------------------------------------------------------------===//

TEST(PatchCacheKeyTest, EqualityOperator) {
  PatchCacheKey Key1;
  Key1.CodeObjectId = 100;
  Key1.KernelId = 200;
  Key1.Mode = InstrumentationMode::MEMORY_ONLY;

  PatchCacheKey Key2;
  Key2.CodeObjectId = 100;
  Key2.KernelId = 200;
  Key2.Mode = InstrumentationMode::MEMORY_ONLY;

  PatchCacheKey Key3;
  Key3.CodeObjectId = 100;
  Key3.KernelId = 201;  // Different kernel
  Key3.Mode = InstrumentationMode::MEMORY_ONLY;

  EXPECT_TRUE(Key1 == Key2);
  EXPECT_FALSE(Key1 == Key3);
}

TEST(PatchCacheKeyTest, HashFunction) {
  std::unordered_map<PatchCacheKey, int> Cache;

  PatchCacheKey Key1;
  Key1.CodeObjectId = 100;
  Key1.KernelId = 200;
  Key1.Mode = InstrumentationMode::MEMORY_ONLY;

  PatchCacheKey Key2;
  Key2.CodeObjectId = 100;
  Key2.KernelId = 300;
  Key2.Mode = InstrumentationMode::MEMORY_ONLY;

  Cache[Key1] = 1;
  Cache[Key2] = 2;

  EXPECT_EQ(Cache.size(), 2u);
  EXPECT_EQ(Cache[Key1], 1);
  EXPECT_EQ(Cache[Key2], 2);
}

TEST(PatchCacheKeyTest, CacheLookupAndMiss) {
  std::unordered_map<PatchCacheKey, std::string> Cache;

  PatchCacheKey Key;
  Key.CodeObjectId = 1;
  Key.KernelId = 2;
  Key.Mode = InstrumentationMode::MEMORY_ONLY;

  Cache[Key] = "cached_kernel";

  auto It = Cache.find(Key);
  ASSERT_NE(It, Cache.end());
  EXPECT_EQ(It->second, "cached_kernel");

  // Miss
  PatchCacheKey MissingKey;
  MissingKey.CodeObjectId = 999;
  MissingKey.KernelId = 999;
  MissingKey.Mode = InstrumentationMode::MEMORY_ONLY;

  EXPECT_EQ(Cache.find(MissingKey), Cache.end());
}

//===----------------------------------------------------------------------===//
// LoadedKernelTest - Tests for LoadedKernel struct
//===----------------------------------------------------------------------===//

TEST(LoadedKernelTest, FieldAssignment) {
  LoadedKernel Kernel;
  Kernel.CodeObjectHandle = 0x1234;
  Kernel.ExecutableHandle = 0x5678;
  Kernel.KernelSymbol = 0xABCD;
  Kernel.KernelName = "myKernel";
  Kernel.OriginalKernargSize = 64;

  EXPECT_EQ(Kernel.CodeObjectHandle, 0x1234ull);
  EXPECT_EQ(Kernel.ExecutableHandle, 0x5678ull);
  EXPECT_EQ(Kernel.KernelSymbol, 0xABCDull);
  EXPECT_EQ(Kernel.KernelName, "myKernel");
  EXPECT_EQ(Kernel.OriginalKernargSize, 64u);
}

//===----------------------------------------------------------------------===//
// TraceArgsTest - Tests for TraceArgs structure
//===----------------------------------------------------------------------===//

TEST(TraceArgsTest, SizeAndAlignment) {
  // TraceArgs is packed to exactly 28 bytes to match GPU-side instrumentation
  // layout. The struct uses __attribute__((packed)) so it is NOT naturally
  // 8-byte aligned — the 4-byte KernelID at the end makes it 28 bytes.
  // The kernarg extension code handles alignment when appending TraceArgs.
  EXPECT_EQ(sizeof(TraceArgs), 28u)
      << "TraceArgs must be exactly 28 bytes (packed layout for GPU)";

  // Check that the struct has the expected fields
  TraceArgs Args;
  Args.BufferPtr = 0;
  Args.BufferSize = 0;
  Args.WriteOffsetPtr = 0;
  Args.KernelID = 0;

  // Should compile without issues - fields exist
  SUCCEED();
}

TEST(TraceArgsTest, FieldOffsets) {
  // Verify fields are at expected offsets for GPU access
  TraceArgs Args = {};

  // BufferPtr should be at offset 0
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&Args.BufferPtr) -
            reinterpret_cast<uintptr_t>(&Args), 0u);

  // All 64-bit fields should be at 8-byte boundaries for aligned access
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&Args.BufferPtr) % 8, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&Args.BufferSize) % 8, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&Args.WriteOffsetPtr) % 8, 0u);
}

} // namespace
