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

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

amdsmi_retired_page_record_t MakeRecord(amdsmi_memory_page_status_t status) {
  amdsmi_retired_page_record_t rec{};
  rec.page_size = 4096;
  rec.status = status;
  return rec;
}

}  // namespace

// A healthy GPU reports zero bad pages, which must count as zero pending
// retirements. This is the exact case that previously surfaced as N/A for
// RDC_HEALTH_PENDING_PAGE_NUM.
TEST(BadPageTest, NoBadPagesYieldsZeroPending) {
  EXPECT_EQ(0u, amd::rdc::count_pending_bad_pages(nullptr, 0));
  std::vector<amdsmi_retired_page_record_t> empty;
  EXPECT_EQ(0u, amd::rdc::count_pending_bad_pages(empty.data(), 0));
}

TEST(BadPageTest, CountsOnlyPendingRecords) {
  std::vector<amdsmi_retired_page_record_t> records = {
      MakeRecord(AMDSMI_MEM_PAGE_STATUS_PENDING),
      MakeRecord(AMDSMI_MEM_PAGE_STATUS_RESERVED),
      MakeRecord(AMDSMI_MEM_PAGE_STATUS_PENDING),
      MakeRecord(AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE),
  };
  EXPECT_EQ(
      2u, amd::rdc::count_pending_bad_pages(records.data(), static_cast<uint32_t>(records.size())));
}

TEST(BadPageTest, NoPendingRecordsYieldsZero) {
  std::vector<amdsmi_retired_page_record_t> records = {
      MakeRecord(AMDSMI_MEM_PAGE_STATUS_RESERVED),
      MakeRecord(AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE),
  };
  EXPECT_EQ(
      0u, amd::rdc::count_pending_bad_pages(records.data(), static_cast<uint32_t>(records.size())));
}

TEST(BadPageTest, AllPendingCountsAll) {
  std::vector<amdsmi_retired_page_record_t> records(5, MakeRecord(AMDSMI_MEM_PAGE_STATUS_PENDING));
  EXPECT_EQ(
      5u, amd::rdc::count_pending_bad_pages(records.data(), static_cast<uint32_t>(records.size())));
}
