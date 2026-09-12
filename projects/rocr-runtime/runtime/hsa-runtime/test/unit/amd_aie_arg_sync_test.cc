////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////


/// Unit tests for ForEachArgumentSyncRange: which argument ranges of an AIE
/// dispatch packet the runtime synchronizes, and which it leaves to the caller.
///
/// The selection is tested against a recording callback rather than against
/// FlushCpuCache, so these run anywhere -- no AIE agent, no driver, no device.

#include "core/inc/amd_aie_arg_sync.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace rocr {
namespace AMD {
namespace {

struct Range {
  void* ptr;
  size_t size;
  bool operator==(const Range& o) const { return ptr == o.ptr && size == o.size; }
};

/// A packet over `args`, laid out as the ABI requires: num_kernargs addresses
/// followed by num_kernargs sizes.
class TestPacket {
 public:
  TestPacket(std::initializer_list<Range> args) {
    for (const auto& a : args) kernargs_.push_back(reinterpret_cast<uint64_t>(a.ptr));
    for (const auto& a : args) kernargs_.push_back(a.size);
    pkt_.num_kernargs = static_cast<uint16_t>(args.size());
    pkt_.kernarg_address = kernargs_.empty() ? nullptr : kernargs_.data();
  }

  const hsa_amd_aie_kernel_dispatch_packet_t* get() const { return &pkt_; }

 private:
  std::vector<uint64_t> kernargs_;
  hsa_amd_aie_kernel_dispatch_packet_t pkt_{};
};

std::vector<Range> Synced(const hsa_amd_aie_kernel_dispatch_packet_t* pkt) {
  std::vector<Range> seen;
  ForEachArgumentSyncRange(pkt, [&](void* ptr, size_t size) { seen.push_back({ptr, size}); });
  return seen;
}

void* const kA = reinterpret_cast<void*>(0x1000);
void* const kB = reinterpret_cast<void*>(0x2000);
void* const kC = reinterpret_cast<void*>(0x3000);

TEST(AieArgumentSyncRange, SyncsEveryArgumentByDefault) {
  const TestPacket pkt({{kA, 64}, {kB, 4096}});
  EXPECT_EQ(Synced(pkt.get()), (std::vector<Range>{{kA, 64}, {kB, 4096}}));
}

TEST(AieArgumentSyncRange, SkipsAnArgumentDeclaredZero) {
  const TestPacket pkt({{kA, 0}});
  EXPECT_TRUE(Synced(pkt.get()).empty());
}

/// The case the zero size exists for: a large caller-synchronized argument
/// alongside small ones the runtime still has to keep coherent.
TEST(AieArgumentSyncRange, SkipsOnlyTheArgumentsDeclaredZero) {
  const TestPacket pkt({{kA, 4096}, {kB, 0}, {kC, 128}});
  EXPECT_EQ(Synced(pkt.get()), (std::vector<Range>{{kA, 4096}, {kC, 128}}));
}

TEST(AieArgumentSyncRange, ReadsSizesFromTheSecondHalfOfTheBuffer) {
  // Addresses that would be plausible sizes and sizes that would be plausible
  // addresses, so reading the halves in the wrong order cannot pass.
  const TestPacket pkt({{reinterpret_cast<void*>(8), 0x4000}, {reinterpret_cast<void*>(16), 0}});
  EXPECT_EQ(Synced(pkt.get()), (std::vector<Range>{{reinterpret_cast<void*>(8), 0x4000}}));
}

TEST(AieArgumentSyncRange, HandlesAPacketWithNoArguments) {
  const TestPacket pkt({});
  EXPECT_TRUE(Synced(pkt.get()).empty());
}

/// num_kernargs may be non-zero only with a buffer, but a null one must not be
/// dereferenced if a caller gets that wrong.
TEST(AieArgumentSyncRange, ToleratesANullKernargBuffer) {
  hsa_amd_aie_kernel_dispatch_packet_t pkt{};
  pkt.num_kernargs = 2;
  pkt.kernarg_address = nullptr;
  EXPECT_TRUE(Synced(&pkt).empty());
}

TEST(AieArgumentSyncRange, SyncsEveryArgumentOfALongPacket) {
  std::vector<Range> args;
  for (uint64_t i = 1; i <= 16; ++i) {
    args.push_back({reinterpret_cast<void*>(i * 0x1000), (i % 2) ? size_t{0} : size_t{i * 8}});
  }
  std::vector<Range> expected;
  for (const auto& a : args)
    if (a.size != 0) expected.push_back(a);

  std::vector<uint64_t> kernargs;
  for (const auto& a : args) kernargs.push_back(reinterpret_cast<uint64_t>(a.ptr));
  for (const auto& a : args) kernargs.push_back(a.size);
  hsa_amd_aie_kernel_dispatch_packet_t pkt{};
  pkt.num_kernargs = static_cast<uint16_t>(args.size());
  pkt.kernarg_address = kernargs.data();

  EXPECT_EQ(Synced(&pkt), expected);
  EXPECT_EQ(Synced(&pkt).size(), 8u);
}

}  // namespace
}  // namespace AMD
}  // namespace rocr
