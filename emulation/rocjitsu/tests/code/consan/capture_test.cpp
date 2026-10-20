// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "capture_fixture.h"
#include "rocjitsu/code/consan/encoding.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

namespace rocjitsu::consan {
namespace {

class ConSanCapture : public ::testing::Test {
protected:
  void SetUp() override {
    header_ = test::capture_header();
    CaptureEncoder encoder;
    auto prefix = encoder.start(header_, &error_);
    ASSERT_TRUE(prefix.succeeded()) << error_.message;
    auto frame = encoder.append(test::execution(header_.manifest, 1), &error_);
    ASSERT_TRUE(frame.succeeded()) << error_.message;
    auto end = encoder.finish(&error_);
    ASSERT_TRUE(end.succeeded());
    bytes_ = prefix.value();
    frame_offset_ = bytes_.size();
    bytes_.insert(bytes_.end(), frame.value().begin(), frame.value().end());
    bytes_.insert(bytes_.end(), end.value().begin(), end.value().end());
    ASSERT_TRUE(decode_capture(bytes_, &error_).succeeded()) << error_.message;
  }

  void reseal(bool frame = false) {
    if (frame) {
      const size_t size = *wire::read_u64(bytes_, frame_offset_ + 8);
      const Digest checksum = digest(std::span(bytes_).subspan(frame_offset_ + 48, size));
      std::copy(checksum.begin(), checksum.end(), bytes_.begin() + frame_offset_ + 16);
    }
    const Digest checksum = digest(std::span(bytes_).first(bytes_.size() - 32));
    std::copy(checksum.begin(), checksum.end(), bytes_.end() - 32);
  }

  CaptureHeader header_;
  ContractError error_;
  std::vector<uint8_t> bytes_;
  size_t frame_offset_ = 0;
};

TEST_F(ConSanCapture, CompleteContainerOwnsFactsArgumentsContextAndRawBytes) {
  auto decoded = decode_capture(bytes_, &error_);
  ASSERT_TRUE(decoded.succeeded()) << error_.message;
  ASSERT_EQ(decoded.value().executions.size(), 1u);
  EXPECT_EQ(decoded.value().header, header_);
  const auto original = test::execution(header_.manifest, 1);
  std::ranges::fill(bytes_, 0xcd);
  const auto &execution = decoded.value().executions.front();
  EXPECT_EQ(execution.serial, 1u);
  EXPECT_EQ(execution.context, original.context);
  EXPECT_EQ(execution.report, original.report);
  EXPECT_EQ(execution.arguments, original.arguments);
  EXPECT_EQ(decoded.value().header.manifest.entry.name, "synthetic_codec_fixture");
}

TEST_F(ConSanCapture, EveryTruncatedPrefixAndTrailingBytesLackACompleteContainer) {
  for (size_t length = 0; length < bytes_.size(); ++length)
    EXPECT_TRUE(decode_capture(std::span(bytes_).first(length)).failed()) << length;
  bytes_.push_back(0);
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
}

TEST_F(ConSanCapture, WrongContractChecksumAndMissingFactsAreDistinctFailures) {
  const auto saved = bytes_;
  bytes_[8] ^= 1;
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IncompatibleContract);
  bytes_ = saved;
  bytes_[frame_offset_ + 48] ^= 1;
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
  EXPECT_NE(error_.message.find("checksum"), std::string::npos);
  bytes_ = saved;
  wire::write_u64(bytes_, 40, 0);
  reseal();
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
}

TEST_F(ConSanCapture, ValidChecksumsDoNotBypassExecutionIdentityAndVisibility) {
  const auto saved = bytes_;
  for (const auto &[field, value] : std::array<std::pair<size_t, uint32_t>, 4>{
           {{48, 0}, {72, 2}, {72 + wire::kContextGrid, 128}, {56, UINT32_MAX}}}) {
    bytes_ = saved;
    wire::write_u32(bytes_, frame_offset_ + 48 + field, value);
    reseal(true);
    EXPECT_TRUE(decode_capture(bytes_, &error_).failed()) << field;
    EXPECT_EQ(error_.message.find("checksum"), std::string::npos) << error_.message;
  }
}

TEST_F(ConSanCapture, ValidContainerPreservesMalformedRowsForIndependentDiagnosis) {
  CaptureEncoder encoder;
  const auto prefix = encoder.start(header_);
  auto execution = test::execution(header_.manifest, 1);
  wire::write_u32(execution.report, kReportHeaderBytes + wire::kRowStatus, UINT32_MAX);
  const auto frame = encoder.append(execution, &error_);
  ASSERT_TRUE(frame.succeeded()) << error_.message;
  const auto end = encoder.finish();
  ASSERT_TRUE(end.succeeded());
  std::vector<uint8_t> bytes = prefix.value();
  bytes.insert(bytes.end(), frame.value().begin(), frame.value().end());
  bytes.insert(bytes.end(), end.value().begin(), end.value().end());
  const auto decoded = decode_capture(bytes, &error_);
  ASSERT_TRUE(decoded.succeeded()) << error_.message;
  EXPECT_EQ(decoded.value().executions[0].report, execution.report);
}

TEST_F(ConSanCapture, EndMembershipBoundsAndReservedFieldsAreCheckedBeyondTheChecksum) {
  const auto saved = bytes_;
  wire::write_u64(bytes_, bytes_.size() - 64 + 8, 2);
  reseal();
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
  EXPECT_NE(error_.message.find("membership"), std::string::npos);
  bytes_ = saved;
  wire::write_u64(bytes_, frame_offset_ + 8, UINT64_MAX);
  reseal();
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
  EXPECT_NE(error_.message.find("truncated execution"), std::string::npos);
  bytes_ = saved;
  bytes_[frame_offset_ + 48 + 136] = 1;
  reseal(true);
  EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
  EXPECT_NE(error_.message.find("reserved"), std::string::npos);
}

TEST_F(ConSanCapture, BudgetsAreInclusiveAndCountsCannotTriggerUnboundedAllocation) {
  CaptureLimits limits;
  limits.file_bytes = bytes_.size();
  EXPECT_TRUE(decode_capture(bytes_, &error_, limits).succeeded());
  --limits.file_bytes;
  EXPECT_TRUE(decode_capture(bytes_, &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  limits = {};
  limits.executions = 0;
  EXPECT_TRUE(decode_capture(bytes_, &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
  limits = {};
  limits.argument_bytes = 15;
  EXPECT_TRUE(decode_capture(bytes_, &error_, limits).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::ResourceLimit);
}

TEST_F(ConSanCapture, UnsupportedExecutionConditionsRemainRejectedBeyondChecksums) {
  const auto saved = bytes_;
  for (const uint32_t unsupported : {0u, 14u, 13u, 11u, 7u, 31u}) {
    CaptureEncoder encoder;
    auto header = header_;
    header.execution_conditions = unsupported;
    EXPECT_TRUE(encoder.start(header, &error_).failed());
    bytes_ = saved;
    wire::write_u32(bytes_, 48, unsupported);
    wire::write_u32(bytes_, bytes_.size() - 64 + 16, unsupported);
    reseal();
    EXPECT_TRUE(decode_capture(bytes_, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    EXPECT_EQ(error_.message.find("checksum"), std::string::npos);
  }
}

TEST_F(ConSanCapture, EncoderRequiresOrderedUniqueExecutionsAndIncreasingStorageGenerations) {
  CaptureEncoder encoder;
  EXPECT_TRUE(encoder.finish(&error_).failed());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 1), &error_).failed());
  auto invalid = header_;
  invalid.execution_conditions = 0;
  EXPECT_TRUE(encoder.start(invalid, &error_).failed());
  EXPECT_TRUE(encoder.start(header_, &error_).succeeded());
  EXPECT_TRUE(encoder.start(header_, &error_).failed());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 2), &error_).failed());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 1), &error_).succeeded());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 1), &error_).failed());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 2), &error_).failed());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 2, 2), &error_).succeeded());
  EXPECT_TRUE(encoder.finish(&error_).succeeded());
  EXPECT_TRUE(encoder.finish(&error_).failed());
  EXPECT_TRUE(encoder.append(test::execution(header_.manifest, 3, 3), &error_).failed());
}

} // namespace
} // namespace rocjitsu::consan
