/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/include/cpuset.h: the sysfs hex-mask and CPU-list parsers and the
// range and list printers. The ncclOsCpu* helpers they call are the os/linux.cc CPU_* one-liners.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "cpuset.h"
#include "os.h"

void ncclOsCpuZero(ncclAffinity& affinity) { CPU_ZERO(&affinity); }
void ncclOsCpuSet(ncclAffinity& affinity, int cpu) { CPU_SET(cpu, &affinity); }
bool ncclOsCpuIsSet(const ncclAffinity& affinity, int cpu) { return CPU_ISSET(cpu, &affinity); }

namespace {

constexpr char kGuard = '#';

class CpusetMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    CPU_ZERO(&set_);
    std::memset(buf_, kGuard, sizeof(buf_));
  }

  void Set(std::initializer_list<int> cpus) {
    CPU_ZERO(&set_);
    for (int cpu : cpus) CPU_SET(cpu, &set_);
  }
  std::vector<int> Cpus() {
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
      if (CPU_ISSET(cpu, &set_)) cpus.push_back(cpu);
    }
    return cpus;
  }
  static std::string Words(const std::vector<std::string>& words) {
    std::string s;
    for (const auto& w : words) s += (s.empty() ? "" : ",") + w;
    return s;
  }

  ncclAffinity set_;
  char buf_[64];
};

// ncclStrToCpuset

TEST_F(CpusetMicrotest, StrToCpuset_LastWordIsTheLowest32Cpus) {
  CPU_SET(500, &set_);
  ASSERT_EQ(ncclSuccess, ncclStrToCpuset("0003ff,f0003fff", &set_));
  std::vector<int> expected;
  for (int cpu : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28, 29, 30, 31}) expected.push_back(cpu);
  for (int cpu = 32; cpu < 42; cpu++) expected.push_back(cpu);
  EXPECT_EQ(expected, Cpus());
  ASSERT_EQ(ncclSuccess, ncclStrToCpuset("80000000", &set_));
  EXPECT_EQ((std::vector<int>{31}), Cpus());
}

TEST_F(CpusetMicrotest, StrToCpuset_FullMaskSetsEveryCpu) {
  ASSERT_EQ(ncclSuccess, ncclStrToCpuset(Words(std::vector<std::string>(32, "ffffffff")).c_str(), &set_));
  EXPECT_EQ(CPU_SETSIZE, CPU_COUNT(&set_));
}

TEST_F(CpusetMicrotest, StrToCpuset_OffsetShiftsAndClipsAtCpuSetSize) {
  ASSERT_EQ(ncclSuccess, ncclStrToCpuset("1,00000001", &set_, 64));
  EXPECT_EQ((std::vector<int>{64, 96}), Cpus());
  ASSERT_EQ(ncclSuccess, ncclStrToCpuset("ff", &set_, CPU_SETSIZE - 4));
  EXPECT_EQ((std::vector<int>{1020, 1021, 1022, 1023}), Cpus());
}

TEST_F(CpusetMicrotest, StrToCpuset_RejectsAnOffsetOutsideTheSet) {
  CPU_SET(7, &set_);
  EXPECT_EQ(ncclInvalidArgument, ncclStrToCpuset("1", &set_, -1));
  EXPECT_EQ(ncclInvalidArgument, ncclStrToCpuset("1", &set_, CPU_SETSIZE));
  EXPECT_EQ((std::vector<int>{7}), Cpus());
  EXPECT_EQ(ncclSuccess, ncclStrToCpuset("1", &set_, CPU_SETSIZE - 1));
  EXPECT_EQ((std::vector<int>{CPU_SETSIZE - 1}), Cpus());
}

// Past 32 words the lowest are dropped and the rest land one word low (CPU 33 reads as 1); a fix flips this pin.
TEST_F(CpusetMicrotest, StrToCpuset_KnownBug_MoreThan32WordsShiftsTheMask) {
  std::vector<std::string> words(33, "0");
  words[31] = "2";
  words[32] = "1";
  ASSERT_EQ(ncclSuccess, ncclStrToCpuset(Words(words).c_str(), &set_));
  EXPECT_EQ((std::vector<int>{1}), Cpus());
}

// Ranks of a single-process multi-GPU job parse their NUMA cpumaps concurrently during init. Each thread
// gets a mask with its own word count and bit pattern, so a tokenizer cursor shared across threads
// (strtok()) shows up as a thread receiving another thread's words, or the wrong number of them.
TEST_F(CpusetMicrotest, StrToCpuset_ConcurrentParsesDoNotInterfere) {
  constexpr int kThreads = 8;
  constexpr int kIters = 20000;
  std::vector<std::string> masks(kThreads);
  std::vector<std::vector<int>> expected(kThreads);
  for (int t = 0; t < kThreads; t++) {
    int nWords = 2 + t;
    for (int w = nWords - 1; w >= 0; w--) {
      int bit = (3 * t + w) % 32;
      char word[16];
      snprintf(word, sizeof(word), "%08x", 1u << bit);
      masks[t] += std::string(masks[t].empty() ? "" : ",") + word;
    }
    for (int w = 0; w < nWords; w++) expected[t].push_back(32 * w + (3 * t + w) % 32);
  }

  std::atomic<bool> go{false};
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([&, t] {
      while (!go.load()) std::this_thread::yield();
      for (int i = 0; i < kIters; i++) {
        ncclAffinity set;
        if (ncclStrToCpuset(masks[t].c_str(), &set) != ncclSuccess) {
          mismatches++;
          continue;
        }
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
          if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
        }
        if (cpus != expected[t]) mismatches++;
      }
    });
  }
  go = true;
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(0, mismatches.load()) << "of " << kThreads * kIters << " concurrent parses";
}

// ncclCpusetToRangeStr

TEST_F(CpusetMicrotest, CpusetToRangeStr_CollapsesRunsIncludingTheLastCpu) {
  Set({0, 1, 2, 5, 7, 8, 1023});
  EXPECT_EQ(buf_, ncclCpusetToRangeStr(&set_, buf_, sizeof(buf_)));
  EXPECT_STREQ("0-2,5,7-8,1023", buf_);
  Set({});
  EXPECT_STREQ("", ncclCpusetToRangeStr(&set_, buf_, sizeof(buf_)));
}

TEST_F(CpusetMicrotest, CpusetToRangeStr_StopsWhenTheBufferIsFull) {
  Set({0, 1, 2, 5, 7, 8, 10});
  ncclCpusetToRangeStr(&set_, buf_, 8);
  EXPECT_STREQ("0-2,5,7", buf_);
  EXPECT_EQ(std::string(8, kGuard), std::string(buf_ + 8, 8));
  std::memset(buf_, kGuard, sizeof(buf_));
  ncclCpusetToRangeStr(&set_, buf_, 4);  // "0-2" fills it exactly
  EXPECT_STREQ("0-2", buf_);
  EXPECT_EQ(std::string(8, kGuard), std::string(buf_ + 4, 8));
}

// With len 0, len - 1 wraps and the second range is written past the empty buffer; a fix flips this pin.
TEST_F(CpusetMicrotest, CpusetToRangeStr_KnownBug_ZeroLengthWritesPastTheBuffer) {
  Set({0, 2});
  ncclCpusetToRangeStr(&set_, buf_, 0);
  EXPECT_EQ(kGuard, buf_[0]);
  EXPECT_EQ(',', buf_[1]);
}

// ncclStrListToCpuset

TEST_F(CpusetMicrotest, StrListToCpuset_SetsEachListedCpu) {
  CPU_SET(500, &set_);
  ASSERT_EQ(ncclSuccess, ncclStrListToCpuset("0,5,1023", &set_));
  EXPECT_EQ((std::vector<int>{0, 5, 1023}), Cpus());
  ASSERT_EQ(ncclSuccess, ncclStrListToCpuset("", &set_));
  EXPECT_TRUE(Cpus().empty());
}

// Tokens are C integer literals: 0x is hex and a leading 0 is octal.
TEST_F(CpusetMicrotest, StrListToCpuset_ParsesCIntegerLiterals) {
  ASSERT_EQ(ncclSuccess, ncclStrListToCpuset("0x10,010", &set_));
  EXPECT_EQ((std::vector<int>{8, 16}), Cpus());
}

// Nothing is rejected: a range keeps its first CPU, junk becomes CPU 0, out-of-set CPUs vanish; a fix flips this pin.
TEST_F(CpusetMicrotest, StrListToCpuset_KnownBug_AcceptsAnything) {
  EXPECT_EQ(ncclSuccess, ncclStrListToCpuset("2-4", &set_));
  EXPECT_EQ((std::vector<int>{2}), Cpus());
  EXPECT_EQ(ncclSuccess, ncclStrListToCpuset("abc", &set_));
  EXPECT_EQ((std::vector<int>{0}), Cpus());
  EXPECT_EQ(ncclSuccess, ncclStrListToCpuset("1024,99999999999", &set_));
  EXPECT_TRUE(Cpus().empty());
}

// ncclCpusetToStrList

TEST_F(CpusetMicrotest, CpusetToStrList_ListsEveryCpu) {
  Set({1, 3, 1023});
  ASSERT_EQ(ncclSuccess, ncclCpusetToStrList(&set_, buf_, sizeof(buf_)));
  EXPECT_STREQ("1,3,1023", buf_);
  Set({});
  ASSERT_EQ(ncclSuccess, ncclCpusetToStrList(&set_, buf_, sizeof(buf_)));
  EXPECT_STREQ("", buf_);
}

TEST_F(CpusetMicrotest, CpusetToStrList_TruncatesWithinLen) {
  Set({1, 3, 10});
  ASSERT_EQ(ncclSuccess, ncclCpusetToStrList(&set_, buf_, 4));
  EXPECT_STREQ("1,3", buf_);
  EXPECT_EQ(kGuard, buf_[4]);
  std::memset(buf_, kGuard, sizeof(buf_));
  ASSERT_EQ(ncclSuccess, ncclCpusetToStrList(&set_, buf_, 0));
  EXPECT_EQ(kGuard, buf_[0]);
}

}  // namespace
