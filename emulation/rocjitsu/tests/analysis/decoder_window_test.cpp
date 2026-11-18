// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {

// Exercise the external decoder contract: maximum lookahead can exceed the
// encoded size, and a decoded instruction may borrow its raw encoding.
class BorrowedInstruction final : public Instruction {
public:
  BorrowedInstruction(const uint32_t *words, int size) : Instruction("borrowed", nullptr) {
    raw_encoding_ = words;
    size_ = size;
  }
};

class WindowDecoder final : public Decoder {
public:
  explicit WindowDecoder(size_t maximum_words = 4, int decoded_size = 4)
      : maximum_words_(maximum_words), decoded_size_(decoded_size) {}

  size_t max_instruction_words() const override { return maximum_words_; }
  DecodeResult decode(const uint32_t *words, const DecodeErrorEmitter &) override {
    observed.assign(words, words + maximum_words_);
    input = words;
    return std::unique_ptr<Instruction>(new BorrowedInstruction(words, decoded_size_));
  }

  std::vector<uint32_t> observed;
  const uint32_t *input = nullptr;

private:
  size_t maximum_words_;
  int decoded_size_;
};

TEST(DecoderWindow, PadsTailAndPreservesBorrowedEncodingLifetime) {
  WindowDecoder decoder;
  const std::array<uint32_t, 1> words{0x12345678};
  DecodeResult result = decoder.decode_window(words, 24);
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(decoder.observed, (std::vector<uint32_t>{words[0], 0, 0, 0}));
  EXPECT_EQ(result.value()->src_loc(), 24u);
  EXPECT_EQ(result.value()->raw_encoding(), words.data());
  EXPECT_EQ(result.value()->raw_encoding()[0], words[0]);
}

TEST(DecoderWindow, FullWindowUsesOriginalInput) {
  WindowDecoder decoder;
  const std::array<uint32_t, 5> words{1, 2, 3, 4, 5};
  DecodeResult result = decoder.decode_window(words);
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(decoder.input, words.data());
  EXPECT_EQ(decoder.observed, (std::vector<uint32_t>{1, 2, 3, 4}));
  EXPECT_EQ(result.value()->raw_encoding(), words.data());
}

TEST(DecoderWindow, RejectsEmptyInputBeforeCallingDecoder) {
  WindowDecoder decoder;
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(decoder.decode_window({}, 0, diagnostic.emitter()).failed());
  EXPECT_EQ(decoder.input, nullptr);
  EXPECT_EQ(diagnostic.message(), "empty decode window");
}

TEST(DecoderWindow, RejectsZeroLookaheadBeforeCallingDecoder) {
  WindowDecoder decoder(0);
  const std::array<uint32_t, 1> words{1};
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(decoder.decode_window(words, 0, diagnostic.emitter()).failed());
  EXPECT_EQ(decoder.input, nullptr);
  EXPECT_EQ(diagnostic.message(), "decoder reported a zero-width decode window");
}

TEST(DecoderWindow, RejectsInvalidDecodedSizes) {
  const std::array<uint32_t, 5> words{1, 2, 3, 4, 5};
  for (int size : {0, -4, 3, 20}) {
    SCOPED_TRACE(size);
    WindowDecoder decoder(4, size);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(decoder.decode_window(words, 0, diagnostic.emitter()).failed());
    EXPECT_EQ(diagnostic.message(), "decoder exceeded the maximum instruction size");
  }
}

TEST(DecoderWindow, RejectsTruncationEvenWhenPaddingDecodes) {
  WindowDecoder decoder(4, 8);
  const std::array<uint32_t, 1> words{1};
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(decoder.decode_window(words, 0, diagnostic.emitter()).failed());
  EXPECT_EQ(diagnostic.message(), "truncated instruction encoding");
  EXPECT_TRUE(decoder.decode_window(words).failed());
}

TEST(DecoderWindow, DecodesExactTailAndRejectsMissingLiteral) {
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> nop{0xbf800000};
  EXPECT_TRUE(decoder->decode_window(nop).succeeded());
  const std::array<uint32_t, 2> literal{0xbe8000ff, 0x12345678}; // s_mov_b32 s0, literal.
  DecodeResult result = decoder->decode_window(literal);
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value()->size(), 8);
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(
      decoder->decode_window(std::span(literal).first(1), 0, diagnostic.emitter()).failed());
  EXPECT_EQ(diagnostic.message(), "truncated instruction encoding");
}

TEST(DecoderWindow, PreservesFourWordGfx1250EncodingAtStreamEnd) {
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 4> words{0xcc3a0200, 0x42020d04, 0xcc336008, 0x04223110};
  DecodeResult result = decoder->decode_window(words, 16);
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value()->size(), 16);
  EXPECT_EQ(result.value()->src_loc(), 16u);
  ASSERT_NE(result.value()->raw_encoding(), nullptr);
  EXPECT_EQ(result.value()->raw_encoding()[3], words[3]);
  for (size_t count : {1, 2, 3}) {
    SCOPED_TRACE(count);
    EXPECT_TRUE(decoder->decode_window(std::span(words).first(count)).failed());
  }
}

TEST(DecoderWindow, PreservesDecoderDiagnostics) {
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 4> invalid{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
  util::StringDiagnostic direct;
  util::StringDiagnostic bounded;
  ASSERT_TRUE(decoder->decode(invalid.data(), direct.emitter()).failed());
  EXPECT_TRUE(decoder->decode_window(invalid, 0, bounded.emitter()).failed());
  EXPECT_FALSE(bounded.message().empty());
  EXPECT_EQ(bounded.message(), direct.message());
}

} // namespace
} // namespace rocjitsu
