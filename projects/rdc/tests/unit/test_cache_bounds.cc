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

// The retention limits of a watched field's cached samples
// (RdcCacheManagerImpl::evict_cache) as rdc.h documents them for
// rdc_field_watch(). Needs no GPU: the test builds RdcCacheManagerImpl.cc, and
// librdc_bootstrap provides the logger.

#include <gtest/gtest.h>
#include <sys/time.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcCacheManagerImpl.h"

namespace {

using amd::rdc::RdcCacheManagerImpl;

constexpr uint32_t kGpu = 0;
constexpr rdc_field_t kField = RDC_FI_GPU_TEMP;
constexpr double kHour = 3600.0;

uint64_t now_ms() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

// Caches samples first..last of kField, one millisecond apart, the last one
// stamped `last_ts`; each sample's value is its number.
void add_samples(RdcCacheManagerImpl* cache, int64_t first, int64_t last, uint64_t last_ts) {
  for (int64_t i = first; i <= last; i++) {
    rdc_field_value value = {};
    value.field_id = kField;
    value.status = RDC_ST_OK;
    value.ts = last_ts - static_cast<uint64_t>(last - i);
    value.type = INTEGER;
    value.value.l_int = i;
    ASSERT_EQ(cache->rdc_update_cache(kGpu, value), RDC_ST_OK);
  }
}

// The newest cached sample's value, or -1 when the field has none.
int64_t latest(RdcCacheManagerImpl* cache) {
  rdc_field_value value = {};
  if (cache->rdc_field_get_latest_value(kGpu, kField, &value) != RDC_ST_OK) {
    return -1;
  }
  return value.value.l_int;
}

// The cached samples' values, oldest first, read through
// rdc_field_get_value_since() as rdci dmon reads them.
std::vector<int64_t> cached(RdcCacheManagerImpl* cache) {
  std::vector<int64_t> out;
  uint64_t since = 0;
  while (out.size() < 1000) {
    uint64_t next = 0;
    rdc_field_value value = {};
    if (cache->rdc_field_get_value_since(kGpu, kField, since, &next, &value) != RDC_ST_OK) {
      break;
    }
    out.push_back(value.value.l_int);
    since = next;
  }
  return out;
}

// rdc_field_watch() takes any uint32_t sample limit. A limit at or above the
// field's sample count drops nothing, also above INT32_MAX.
class SampleLimitTest : public ::testing::TestWithParam<uint64_t> {};

TEST_P(SampleLimitTest, KeepsEverySampleUpToTheLimit) {
  RdcCacheManagerImpl cache;
  add_samples(&cache, 1, 3, now_ms());
  ASSERT_EQ(cache.evict_cache(kGpu, kField, GetParam(), kHour), RDC_ST_OK);
  // The newest sample first: if the eviction went wrong, the walk below would
  // read past the field's samples.
  ASSERT_EQ(latest(&cache), 3);
  EXPECT_EQ(cached(&cache), (std::vector<int64_t>{1, 2, 3}));
}

INSTANTIATE_TEST_SUITE_P(LargeLimits, SampleLimitTest,
                         ::testing::Values(3u, 4u, uint64_t{std::numeric_limits<int32_t>::max()},
                                           uint64_t{std::numeric_limits<int32_t>::max()} + 1,
                                           uint64_t{std::numeric_limits<int32_t>::max()} + 5,
                                           uint64_t{std::numeric_limits<uint32_t>::max()} - 1,
                                           uint64_t{std::numeric_limits<uint32_t>::max()}));

TEST(CacheBoundsTest, SampleLimitDropsTheOldestSamples) {
  RdcCacheManagerImpl cache;
  add_samples(&cache, 1, 5, now_ms());
  ASSERT_EQ(cache.evict_cache(kGpu, kField, 2, kHour), RDC_ST_OK);
  EXPECT_EQ(cached(&cache), (std::vector<int64_t>{4, 5}));
}

// rdc.h: a max_keep_samples of 0 keeps no samples.
TEST(CacheBoundsTest, SampleLimitZeroKeepsNoSample) {
  RdcCacheManagerImpl cache;
  add_samples(&cache, 1, 3, now_ms());
  ASSERT_EQ(cache.evict_cache(kGpu, kField, 0, kHour), RDC_ST_OK);
  EXPECT_EQ(latest(&cache), -1);
}

TEST(CacheBoundsTest, AgeLimitDropsOlderSamples) {
  RdcCacheManagerImpl cache;
  uint64_t now = now_ms();
  add_samples(&cache, 1, 2, now - 60000);
  add_samples(&cache, 3, 4, now);
  ASSERT_EQ(cache.evict_cache(kGpu, kField, 10, 30.0), RDC_ST_OK);
  EXPECT_EQ(cached(&cache), (std::vector<int64_t>{3, 4}));
}

// rdc.h: a max_keep_age of 0 keeps no samples.
TEST(CacheBoundsTest, AgeLimitZeroKeepsNoSample) {
  RdcCacheManagerImpl cache;
  add_samples(&cache, 1, 3, now_ms() - 10);
  ASSERT_EQ(cache.evict_cache(kGpu, kField, 10, 0.0), RDC_ST_OK);
  EXPECT_EQ(latest(&cache), -1);
}

}  // namespace
