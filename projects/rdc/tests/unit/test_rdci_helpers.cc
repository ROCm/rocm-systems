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

// Helpers rdci uses for what rdcd sends: rdc_escape_text() and rdc_is_clean_text()
// (rdc_lib/rdc_common.h) keep terminal control characters out of rdci's output and out of new
// group and field group names; GpuPartitions() (common/rdc_utils.h) gives each GPU index the
// socket and partition that RDC's partition IDs (gS.P) name.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/rdc_utils.h"
#include "rdc_lib/rdc_common.h"

namespace {

std::string escape(const std::string& text) { return rdc_escape_text(text, false); }
std::string escape_json(const std::string& text) { return rdc_escape_text(text, true); }
bool clean(const std::string& text) { return rdc_is_clean_text(text.data(), text.size()); }

// Text that is not valid UTF-8 (RFC 3629) or that holds a control character.
const std::vector<std::string> kUnclean = {
    "\x1b[31mred",           // ESC
    "tab\there",             // C0
    std::string("a\0b", 3),  // NUL
    "x\x7f",                 // DEL
    "\xc2\x9b"
    "31m",  // U+009B, the 8-bit CSI
    "\x9b"
    "31m",               // a lone continuation byte
    "latin1 \xe9t\xe9",  // Latin-1, not UTF-8
    "\xff",              // never in UTF-8
    "\xc0\x80",          // overlong NUL
    "\xe0\x80\x9b",      // overlong CSI
    "\xed\xa0\x80",      // a UTF-16 surrogate
    "\xf4\x90\x80\x80",  // above U+10FFFF
    "\xf5\x80\x80\x80",  // a lead byte past U+10FFFF
    "ok\xe2\x82",        // truncated at the end
};

TEST(RdcEscapeText, KeepsPrintableText) {
  EXPECT_EQ(escape("GPU group 1 (rack A)"), "GPU group 1 (rack A)");
  EXPECT_EQ(escape("calf\xc3\xa9"), "calf\xc3\xa9");          // U+00E9, two bytes
  EXPECT_EQ(escape("\xc2\xa0"), "\xc2\xa0");                  // U+00A0 is no control
  EXPECT_EQ(escape("\xe2\x82\xac 5"), "\xe2\x82\xac 5");      // U+20AC, three bytes
  EXPECT_EQ(escape("\xf0\x9f\x98\x80"), "\xf0\x9f\x98\x80");  // U+1F600, four bytes
  EXPECT_EQ(escape("a\"b\\c"), "a\"b\\c");  // quotes and backslashes are text outside JSON
}

TEST(RdcEscapeText, EscapesC0ControlsAndDelete) {
  EXPECT_EQ(escape("\x1b[31mred"), "\\x1b[31mred");
  EXPECT_EQ(escape("a\tb\nc\rd"), "a\\x09b\\x0ac\\x0dd");
  EXPECT_EQ(escape("bell\x07"), "bell\\x07");
  EXPECT_EQ(escape(std::string("a\0b", 3)), "a\\x00b");
  EXPECT_EQ(escape("x\x7f"), "x\\x7f");
}

TEST(RdcEscapeText, EscapesC1ControlsAndEachByteOfInvalidUtf8) {
  EXPECT_EQ(escape("\xc2\x9b"
                   "31m"),
            "\\xc2\\x9b31m");
  EXPECT_EQ(escape("\xc2\x80"), "\\xc2\\x80");
  EXPECT_EQ(escape("\x9b"
                   "31m"),
            "\\x9b31m");
  EXPECT_EQ(escape("latin1 \xe9t\xe9"), "latin1 \\xe9t\\xe9");
  EXPECT_EQ(escape("\xff"), "\\xff");
  EXPECT_EQ(escape("\xc0\x80"), "\\xc0\\x80");
  EXPECT_EQ(escape("\xe0\x80\x9b"), "\\xe0\\x80\\x9b");
  EXPECT_EQ(escape("\xed\xa0\x80"), "\\xed\\xa0\\x80");
  EXPECT_EQ(escape("\xf4\x90\x80\x80"), "\\xf4\\x90\\x80\\x80");
  EXPECT_EQ(escape("\xf5\x80\x80\x80"), "\\xf5\\x80\\x80\\x80");
  EXPECT_EQ(escape("ok\xe2\x82"), "ok\\xe2\\x82");
  EXPECT_EQ(escape("\xe2\x82"
                   "A"),
            "\\xe2\\x82A");
}

TEST(RdcEscapeText, JsonFormEscapesEachCharacterOnce) {
  EXPECT_EQ(escape_json("\x1b[31m"), "\\u001b[31m");
  EXPECT_EQ(escape_json("a\tb"), "a\\u0009b");
  EXPECT_EQ(escape_json("x\x7f"), "x\\u007f");
  EXPECT_EQ(escape_json("\xc2\x9b"), "\\u009b");  // one C1 character, one escape
  EXPECT_EQ(escape_json("\x9b"), "\\u009b");      // an invalid byte reads as U+00HH
  EXPECT_EQ(escape_json("a\"b\\c"), "a\\\"b\\\\c");
  EXPECT_EQ(escape_json("calf\xc3\xa9"), "calf\xc3\xa9");
}

TEST(RdcIsCleanText, AcceptsPrintableUtf8) {
  EXPECT_TRUE(clean(""));
  EXPECT_TRUE(clean("rdci-dmon-group"));
  EXPECT_TRUE(clean("calf\xc3\xa9 \xc2\xa0 \xe2\x82\xac \xf0\x9f\x98\x80"));
  EXPECT_TRUE(clean("a\"b\\c"));
}

TEST(RdcIsCleanText, RejectsControlsAndInvalidUtf8) {
  for (const std::string& text : kUnclean) {
    EXPECT_FALSE(clean(text)) << escape(text);
  }
}

TEST(RdcIsCleanText, ChecksOnlyTheGivenLength) {
  EXPECT_TRUE(rdc_is_clean_text("ab\x1b", 2));
  EXPECT_FALSE(rdc_is_clean_text("ab\x1b", 3));
  EXPECT_TRUE(rdc_is_clean_text("calf\xc3\xa9", 6));
  EXPECT_FALSE(rdc_is_clean_text("calf\xc3\xa9", 5));  // the cut leaves half a character
}

// GpuPartitions() takes, for each GPU index, the partition count rdc_get_num_partition()
// reports: its socket's count, 1 for a partitionable GPU in SPX mode, UINT16_MAX for a GPU that
// cannot be partitioned.
using Slots = std::vector<std::pair<uint32_t, int32_t>>;  // (socket, partition or -1)
constexpr uint16_t kNotPartitionable = UINT16_MAX;

Slots slots(const std::vector<uint16_t>& counts) {
  Slots out;
  for (const amd::rdc::GpuPartition& p : amd::rdc::GpuPartitions(counts)) {
    out.emplace_back(p.socket, p.partition);
  }
  return out;
}

TEST(GpuPartitions, OneCpxSocketThenSpxSockets) {
  // The MI350X of F27's run: socket 0 in CPX holds GPU indexes 0-7, sockets 1-7 one each.
  std::vector<uint16_t> counts(8, 8);
  counts.insert(counts.end(), 7, 1);
  Slots want;
  for (int32_t k = 0; k < 8; ++k) {
    want.emplace_back(0, k);
  }
  for (uint32_t s = 1; s <= 7; ++s) {
    want.emplace_back(s, -1);
  }
  EXPECT_EQ(slots(counts), want);
}

TEST(GpuPartitions, SpxSocketThenPartitionedSockets) {
  EXPECT_EQ(slots({1, 2, 2, 4, 4, 4, 4}),
            (Slots{{0, -1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 3}}));
}

TEST(GpuPartitions, GpusWithoutPartitions) {
  EXPECT_EQ(slots({kNotPartitionable, kNotPartitionable, 0, 1}),
            (Slots{{0, -1}, {1, -1}, {2, -1}, {3, -1}}));
}

TEST(GpuPartitions, ACountPastTheListsEnd) { EXPECT_EQ(slots({4, 4}), (Slots{{0, 0}, {0, 1}})); }

TEST(GpuPartitions, AnotherCountEndsASocket) {
  EXPECT_EQ(slots({2, 4, 4, 4, 4, 2, 2}),
            (Slots{{0, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}}));
}

TEST(GpuPartitions, NoGpus) { EXPECT_TRUE(slots({}).empty()); }

}  // namespace
