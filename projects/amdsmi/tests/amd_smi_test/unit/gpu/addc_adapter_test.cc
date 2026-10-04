// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Covers the addc-base adapter behind amdsmi_get_afids_from_cper() and
// amdsmi_get_cper_json(); no GPU required.

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "addc/lib_c.h"
#include "amd_smi/impl/amd_smi_addc.h"
#include "amd_smi/impl/amd_smi_cper.h"

namespace {

constexpr size_t kCperHeaderSize = 128;
constexpr size_t kSectionDescriptorSize = 72;
constexpr size_t kBodyOffset = (kCperHeaderSize + kSectionDescriptorSize);

// Offsets inside the CPER header and section descriptor that addc reads.
constexpr size_t kHeaderSectionCountOffset = 10;
constexpr size_t kHeaderRecordLengthOffset = 20;
constexpr size_t kHeaderSignatureEndOffset = 6;
constexpr size_t kDescriptorBodyLengthOffset = 4;
constexpr size_t kDescriptorRevisionOffset = 8;
constexpr size_t kDescriptorFlagsOffset = 10;
constexpr size_t kDescriptorPlatformOffset = 20;
constexpr size_t kDescriptorFruOffset = 52;

constexpr uint32_t kCperSignature = 0x52455043;  // "CPER", little endian
constexpr uint32_t kSignatureEnd = 0xFFFFFFFF;
constexpr uint16_t kAmdVendorId = 0x1022;

auto put_u16(std::vector<uint8_t>* bytes, size_t offset, uint16_t value) -> void {
  (*bytes)[offset] = static_cast<uint8_t>(value);
  (*bytes)[offset + 1] = static_cast<uint8_t>(value >> 8);
}

auto put_u32(std::vector<uint8_t>* bytes, size_t offset, uint32_t value) -> void {
  for (size_t i = 0; i < sizeof(value); ++i) {
    (*bytes)[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

// One section that addc accepts but reports no event for.
auto make_record_without_events() -> std::vector<uint8_t> {
  const auto size = (kBodyOffset + 1);
  auto bytes = std::vector<uint8_t>(size, 0);
  put_u32(&bytes, 0, kCperSignature);
  put_u32(&bytes, kHeaderSignatureEndOffset, kSignatureEnd);
  put_u16(&bytes, kHeaderSectionCountOffset, 1);
  put_u32(&bytes, kHeaderRecordLengthOffset, static_cast<uint32_t>(size));
  put_u32(&bytes, kCperHeaderSize, static_cast<uint32_t>(kBodyOffset));
  put_u32(&bytes, (kCperHeaderSize + kDescriptorBodyLengthOffset), 1);
  return bytes;
}

// One AMD section whose event addc cannot classify further: it yields
// ADDC_UNCLASSIFIED_AFID with the FRU text below.
constexpr char kUnclassifiedFru[] = "TEST_FRU";
auto make_record_with_unclassified_event() -> std::vector<uint8_t> {
  constexpr size_t kBodySize = 16;
  const auto size = (kBodyOffset + kBodySize);
  auto bytes = std::vector<uint8_t>(size, 0);
  put_u32(&bytes, 0, kCperSignature);
  put_u32(&bytes, kHeaderSignatureEndOffset, kSignatureEnd);
  put_u16(&bytes, kHeaderSectionCountOffset, 1);
  put_u32(&bytes, kHeaderRecordLengthOffset, static_cast<uint32_t>(size));
  put_u32(&bytes, kCperHeaderSize, static_cast<uint32_t>(kBodyOffset));
  put_u32(&bytes, (kCperHeaderSize + kDescriptorBodyLengthOffset),
          static_cast<uint32_t>(kBodySize));
  put_u16(&bytes, (kCperHeaderSize + kDescriptorRevisionOffset), 0x3100);
  bytes[kCperHeaderSize + kDescriptorFlagsOffset] = 0x2;
  put_u16(&bytes, (kCperHeaderSize + kDescriptorPlatformOffset), kAmdVendorId);
  std::memcpy(&bytes[kCperHeaderSize + kDescriptorFruOffset], kUnclassifiedFru,
              (sizeof(kUnclassifiedFru) - 1));
  return bytes;
}

// A whole header with a valid signature but no sections, which addc refuses.
auto make_record_without_sections() -> std::vector<uint8_t> {
  auto bytes = std::vector<uint8_t>(kCperHeaderSize, 0);
  put_u32(&bytes, 0, kCperSignature);
  put_u32(&bytes, kHeaderSignatureEndOffset, kSignatureEnd);
  put_u32(&bytes, kHeaderRecordLengthOffset, static_cast<uint32_t>(kCperHeaderSize));
  return bytes;
}

// Two sections whose bodies overlap. Tolerant parsing moves the second offset and
// reports the repair; strict parsing rejects the record.
auto make_record_needing_repair() -> std::vector<uint8_t> {
  constexpr size_t kSectionCount = 2;
  constexpr size_t kOverlappingBodyLength = 4;
  const auto first_body = (kCperHeaderSize + (kSectionCount * kSectionDescriptorSize));
  const auto size = (first_body + (2 * kOverlappingBodyLength));
  auto bytes = std::vector<uint8_t>(size, 0);
  put_u32(&bytes, 0, kCperSignature);
  put_u32(&bytes, kHeaderSignatureEndOffset, kSignatureEnd);
  put_u16(&bytes, kHeaderSectionCountOffset, static_cast<uint16_t>(kSectionCount));
  put_u32(&bytes, kHeaderRecordLengthOffset, static_cast<uint32_t>(size));
  for (size_t i = 0; i < kSectionCount; ++i) {
    const auto descriptor = (kCperHeaderSize + (i * kSectionDescriptorSize));
    put_u32(&bytes, descriptor, static_cast<uint32_t>(first_body));
    put_u32(&bytes, (descriptor + kDescriptorBodyLengthOffset),
            static_cast<uint32_t>(kOverlappingBodyLength));
  }
  return bytes;
}

// AMD GPU section types, in the mixed-endian byte order CPER stores GUIDs.
constexpr uint8_t kCrashdumpGuidBytes[16] = {0x78, 0x0c, 0xac, 0x32, 0x23, 0x26, 0xf6, 0x48,
                                             0xb0, 0xd0, 0x73, 0x65, 0x72, 0x5f, 0xd6, 0xae};
constexpr uint8_t kBootNotifyGuidBytes[16] = {0x66, 0xa4, 0x61, 0x3d, 0x40, 0xab, 0x9a, 0x40,
                                              0xa6, 0x98, 0xf3, 0x62, 0xd4, 0x64, 0xb3, 0x8f};
constexpr uint8_t kNonStandardGuidBytes[16] = {0x78, 0x0c, 0xac, 0x32, 0x23, 0x26, 0xf6, 0x48,
                                               0x81, 0xa2, 0xac, 0x69, 0x17, 0x80, 0x55, 0x1d};
constexpr uint8_t kProcErrGuidBytes[16] = {0xb0, 0xa0, 0x3e, 0xdc, 0x44, 0xa1, 0x97, 0x47,
                                           0xb9, 0x5b, 0x53, 0xfa, 0x24, 0x2b, 0x6e, 0x1d};

constexpr uint8_t kMi300RevisionMajor = 0x22;
constexpr uint64_t kOneErrorInfoOneContext = ((uint64_t{1} << 2) | (uint64_t{1} << 8));
constexpr uint16_t kAcaRegisterContext = 1;
constexpr uint16_t kBootRegisterContext = 9;
constexpr size_t kAcaRegisterCount = 16;

// Registers whose fields land at distinct indices, so a decoder that reads them from
// the wrong place gives a different AFID.
constexpr uint64_t kFatalRegisters[] = {0xB400000000000000ULL, 0x0000000000001234ULL,
                                        0x0096000000000000ULL, 0x000000000000ABCDULL};
constexpr uint64_t kAcaRegisters[kAcaRegisterCount] = {
    0, 0xB400000000000000ULL, 0x0000000000001234ULL, 0,
    0, 0x0096000000000000ULL, 0x000000000000ABCDULL};

// [header][descriptor][pad][section]: one AMD GPU section of the given type and size.
// A pad of one byte puts the section at an odd offset.
auto make_gpu_record(const uint8_t (&guid)[16], size_t section_size, size_t pad = 0)
    -> std::vector<uint8_t> {
  const auto section_offset = (kBodyOffset + pad);
  const auto size = (section_offset + section_size);
  auto bytes = std::vector<uint8_t>(size, 0);
  put_u32(&bytes, 0, kCperSignature);
  put_u32(&bytes, kHeaderSignatureEndOffset, kSignatureEnd);
  put_u16(&bytes, kHeaderSectionCountOffset, 1);
  put_u32(&bytes, kHeaderRecordLengthOffset, static_cast<uint32_t>(size));
  auto* descriptor = reinterpret_cast<cper_sec_desc*>(&bytes[kCperHeaderSize]);
  descriptor->sec_offset = static_cast<uint32_t>(section_offset);
  descriptor->sec_length = static_cast<uint32_t>(section_size);
  descriptor->revision_major = kMi300RevisionMajor;
  std::memcpy(&descriptor->sec_type, guid, sizeof(guid));
  return bytes;
}

auto make_mi300_crashdump_record(size_t pad = 0, size_t section_size = sizeof(cper_sec_crashdump))
    -> std::vector<uint8_t> {
  auto bytes = make_gpu_record(kCrashdumpGuidBytes, section_size, pad);
  auto* section = reinterpret_cast<cper_sec_crashdump*>(&bytes[kBodyOffset + pad]);
  section->reserved1 = kOneErrorInfoOneContext;
  section->data.reg_ctx_type = kAcaRegisterContext;
  section->data.reg_arr_size = static_cast<uint16_t>(sizeof(section->data.dump.fatal_err));
  static_assert(sizeof(kFatalRegisters) == sizeof(section->data.dump.fatal_err));
  // A section cut short of fatal_err holds only part of the registers.
  const auto dump_offset =
      (offsetof(cper_sec_crashdump, data) + offsetof(cper_sec_crashdump_data, dump));
  std::memcpy(&section->data.dump.fatal_err, kFatalRegisters,
              std::min(sizeof(kFatalRegisters), (section_size - dump_offset)));
  return bytes;
}

constexpr auto kNonStandardSectionSize =
    (sizeof(cper_sec_nonstd_err_hdr) + sizeof(cper_sec_nonstd_err_body));

auto fill_nonstandard_section(uint8_t* section, uint16_t register_context) -> void {
  auto* header = reinterpret_cast<cper_sec_nonstd_err_hdr*>(section);
  header->valid_bits.err_info_cnt = 1;
  header->valid_bits.err_context_cnt = 1;
  auto* body =
      reinterpret_cast<cper_sec_nonstd_err_body*>(section + sizeof(cper_sec_nonstd_err_hdr));
  body->err_ctx.reg_ctx_type = register_context;
  body->err_ctx.reg_arr_size = static_cast<uint16_t>(sizeof(kAcaRegisters));
  static_assert(sizeof(kAcaRegisters) == sizeof(body->err_ctx.reg_dump));
  std::memcpy(body->err_ctx.reg_dump, kAcaRegisters, sizeof(kAcaRegisters));
}

auto make_mi300_nonstandard_record(uint16_t register_context, size_t pad = 0)
    -> std::vector<uint8_t> {
  auto bytes = make_gpu_record(kNonStandardGuidBytes, kNonStandardSectionSize, pad);
  fill_nonstandard_section(&bytes[kBodyOffset + pad], register_context);
  return bytes;
}

// One non-standard section per context, each with its own body.
auto make_mi300_event_record(const std::vector<uint16_t>& contexts) -> std::vector<uint8_t> {
  const auto kSectionCount = contexts.size();
  const auto first_body = (kCperHeaderSize + (kSectionCount * kSectionDescriptorSize));
  const auto size = (first_body + (kSectionCount * kNonStandardSectionSize));
  auto bytes = std::vector<uint8_t>(size, 0);
  put_u32(&bytes, 0, kCperSignature);
  put_u32(&bytes, kHeaderSignatureEndOffset, kSignatureEnd);
  put_u16(&bytes, kHeaderSectionCountOffset, static_cast<uint16_t>(kSectionCount));
  put_u32(&bytes, kHeaderRecordLengthOffset, static_cast<uint32_t>(size));
  for (size_t i = 0; i < kSectionCount; ++i) {
    const auto section_offset = (first_body + (i * kNonStandardSectionSize));
    auto* descriptor =
        reinterpret_cast<cper_sec_desc*>(&bytes[kCperHeaderSize + (i * kSectionDescriptorSize)]);
    descriptor->sec_offset = static_cast<uint32_t>(section_offset);
    descriptor->sec_length = static_cast<uint32_t>(kNonStandardSectionSize);
    descriptor->revision_major = kMi300RevisionMajor;
    std::memcpy(&descriptor->sec_type, kNonStandardGuidBytes, sizeof(kNonStandardGuidBytes));
    fill_nonstandard_section(&bytes[section_offset], contexts[i]);
  }
  return bytes;
}

// An ACA context and a boot context.
auto make_mi300_two_event_record() -> std::vector<uint8_t> {
  return make_mi300_event_record({kAcaRegisterContext, kBootRegisterContext});
}

// Readable bytes that end flush against an unmapped page, so a read past the end
// faults instead of landing in a neighbouring allocation and returning garbage.
class GuardedRegion_t {
 public:
  explicit GuardedRegion_t(size_t readable_size) {
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    m_size = (2 * page);
    m_region =
        mmap(nullptr, m_size, (PROT_READ | PROT_WRITE), (MAP_PRIVATE | MAP_ANONYMOUS), -1, 0);
    if (m_region == MAP_FAILED) {
      return;
    }
    auto* bytes = static_cast<uint8_t*>(m_region);
    if (mprotect(bytes + page, page, PROT_NONE) != 0) {
      return;
    }
    m_data = ((bytes + page) - readable_size);
    m_is_ready = true;
  }
  ~GuardedRegion_t() {
    if (m_region != MAP_FAILED) {
      munmap(m_region, m_size);
    }
  }
  GuardedRegion_t(const GuardedRegion_t&) = delete;
  auto operator=(const GuardedRegion_t&) -> GuardedRegion_t& = delete;

  auto is_ready() const -> bool { return m_is_ready; }
  auto data() const -> uint8_t* { return m_data; }

 private:
  void* m_region = MAP_FAILED;
  size_t m_size = 0;
  uint8_t* m_data = nullptr;
  bool m_is_ready = false;
};

// Runs the call in a forked child: an overread faults, and one fault in-process would
// take the runner down and silently skip every test after it.
auto expect_no_fault_and_status(const uint8_t* record, size_t size, amdsmi_status_t expected)
    -> void {
  EXPECT_EXIT(
      {
        auto afids = std::vector<int>{};
        _exit((cper_get_afids(record, size, &afids) == expected) ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

}  // namespace

TEST(GpuUnit, AddcAfidsRejectsRecordAddcCannotParse) {
  const auto record = make_record_without_sections();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_TRUE(afids.empty());
}

TEST(GpuUnit, AddcAfidsRejectsBufferSmallerThanHeader) {
  const auto record = make_record_without_events();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), (kCperHeaderSize - 1), &afids),
            AMDSMI_STATUS_UNEXPECTED_SIZE);
  EXPECT_TRUE(afids.empty());
}

TEST(GpuUnit, AddcAfidsRejectsRecordLengthBelowHeader) {
  auto record = make_record_without_events();
  put_u32(&record, kHeaderRecordLengthOffset, static_cast<uint32_t>(kCperHeaderSize - 1));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_SIZE);
}

TEST(GpuUnit, AddcAfidsRejectsRecordLengthPastBuffer) {
  auto record = make_record_without_events();
  put_u32(&record, kHeaderRecordLengthOffset, static_cast<uint32_t>(record.size() + 1));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_SIZE);
}

TEST(GpuUnit, AddcAfidsRejectsMissingSignature) {
  auto record = make_record_without_events();
  record[0] = 'X';
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_DATA);
}

// The record is parsed by its own record_length, so bytes the caller left after it
// (a ring read hands over more than one record) cannot turn a valid record into an error.
TEST(GpuUnit, AddcAfidsIgnoresBytesBeyondTheRecord) {
  auto record = make_record_with_unclassified_event();
  const auto record_size = record.size();
  record.resize((record_size + kCperHeaderSize), 0xAB);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids.size(), 1u);
}

TEST(GpuUnit, AddcAfidsRejectsRecordThatNeedsRepair) {
  const auto record = make_record_needing_repair();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_TRUE(afids.empty());
}

TEST(GpuUnit, AddcAfidsAcceptsRecordWithoutEvents) {
  const auto record = make_record_without_events();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(afids.empty());
}

TEST(GpuUnit, AddcAfidsReturnsLibraryAfidUnchanged) {
  const auto record = make_record_with_unclassified_event();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(afids.size(), 1u);
  EXPECT_EQ(afids[0], ADDC_UNCLASSIFIED_AFID);
  EXPECT_EQ(afids[0], 16999);
}

TEST(GpuUnit, AddcReportSizeQueryReturnsRequiredBytes) {
  const auto record = make_record_without_events();
  auto required = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &required),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_GT(required, 0u);
}

TEST(GpuUnit, AddcReportTooSmallBufferReportsRequiredSize) {
  const auto record = make_record_without_events();
  auto required = uint32_t{0};
  ASSERT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &required),
            AMDSMI_STATUS_SUCCESS);

  auto buffer = std::string(required, 'x');
  auto size = (required - 1);
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), buffer.data(), &size),
            AMDSMI_STATUS_INSUFFICIENT_SIZE);
  EXPECT_EQ(size, required);
}

TEST(GpuUnit, AddcReportExactBufferHoldsTerminatedJson) {
  const auto record = make_record_without_events();
  auto size = uint32_t{0};
  ASSERT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &size),
            AMDSMI_STATUS_SUCCESS);

  auto buffer = std::string(size, 'x');
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), buffer.data(), &size),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(buffer[0], '{');
  EXPECT_EQ(buffer[size - 1], '\0');
  EXPECT_NE(buffer.find("schema_version"), std::string::npos);
}

TEST(GpuUnit, AddcReportRejectsRecordThatNeedsRepair) {
  const auto record = make_record_needing_repair();
  auto size = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &size),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AddcReportRejectsRecordAddcCannotParse) {
  const auto record = make_record_without_sections();
  auto size = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &size),
            AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AddcReportRejectsBufferSmallerThanHeader) {
  const auto record = make_record_without_events();
  auto size = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(record.data(), (kCperHeaderSize - 1), nullptr, &size),
            AMDSMI_STATUS_UNEXPECTED_SIZE);
}

// The record_length load at offset 20 of an 8-byte buffer lands on the guard page
// unless the size check runs first.
TEST(GpuUnit, AddcAfidsDoesNotReadPastAShortBuffer) {
  constexpr size_t kReadable = 8;
  auto region = GuardedRegion_t{kReadable};
  ASSERT_TRUE(region.is_ready()) << strerror(errno);
  std::memcpy(region.data(), "CPER", 4);
  expect_no_fault_and_status(region.data(), kReadable, AMDSMI_STATUS_UNEXPECTED_SIZE);
}

// A section descriptor whose FRU text has no terminator, in a record that ends
// against the guard page: an unbounded string read runs off the descriptor and the buffer.
TEST(GpuUnit, AddcAfidsDoesNotReadPastUnterminatedFruText) {
  auto record = make_record_with_unclassified_event();
  record.resize(kBodyOffset);
  put_u32(&record, kHeaderRecordLengthOffset, static_cast<uint32_t>(record.size()));
  std::memset(&record[kCperHeaderSize + kDescriptorFruOffset], 'A',
              (kSectionDescriptorSize - kDescriptorFruOffset));

  auto region = GuardedRegion_t{record.size()};
  ASSERT_TRUE(region.is_ready()) << strerror(errno);
  std::memcpy(region.data(), record.data(), record.size());
  expect_no_fault_and_status(region.data(), record.size(), AMDSMI_STATUS_UNEXPECTED_DATA);
}

// sec_cnt outruns the descriptor table the record carries, so the second
// descriptor would begin on the guard page.
TEST(GpuUnit, AddcAfidsDoesNotReadPastTheDescriptorTable) {
  auto record = make_record_needing_repair();
  record.resize(kBodyOffset);
  put_u16(&record, kHeaderSectionCountOffset, 3);
  put_u32(&record, kHeaderRecordLengthOffset, static_cast<uint32_t>(record.size()));

  auto region = GuardedRegion_t{record.size()};
  ASSERT_TRUE(region.is_ready()) << strerror(errno);
  std::memcpy(region.data(), record.data(), record.size());
  expect_no_fault_and_status(region.data(), record.size(), AMDSMI_STATUS_UNEXPECTED_DATA);
}

// The expected AFIDs are the values users saw before the addc switch, so a change here is
// a change in what users see.
TEST(GpuUnit, AddcAfidsMatchThePreviousDecoderForMi300Crashdump) {
  const auto record = make_mi300_crashdump_record();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{28}));
}

TEST(GpuUnit, AddcAfidsMatchThePreviousDecoderForMi300NonStandardAca) {
  const auto record = make_mi300_nonstandard_record(kAcaRegisterContext);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{28}));
}

TEST(GpuUnit, AddcAfidsMatchThePreviousDecoderForMi300NonStandardBoot) {
  const auto record = make_mi300_nonstandard_record(kBootRegisterContext);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{5}));
}

// The AFID has to come from the registers the record carries, not from the section
// type alone: a record whose registers are all zero reports no event.
TEST(GpuUnit, AddcAfidsDependOnTheRegisterPayload) {
  auto record = make_mi300_crashdump_record();
  auto* section = reinterpret_cast<cper_sec_crashdump*>(&record[kBodyOffset]);
  std::memset(&section->data.dump.fatal_err, 0, sizeof(section->data.dump.fatal_err));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(afids.empty());
}

// An oversized register array is a malformed record, so it is rejected instead of
// being clamped into an AFID the record never earned.
TEST(GpuUnit, AddcAfidsRejectsOversizedRegisterArray) {
  auto record = make_mi300_nonstandard_record(kBootRegisterContext);
  auto* body = reinterpret_cast<cper_sec_nonstd_err_body*>(
      &record[kBodyOffset + sizeof(cper_sec_nonstd_err_hdr)]);
  body->err_ctx.reg_arr_size = 0xFFFF;
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_TRUE(afids.empty());
}

TEST(GpuUnit, AddcAfidsRejectsNullOrEmptyBuffer) {
  const auto record = make_record_without_events();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(nullptr, record.size(), &afids), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(cper_get_afids(record.data(), 0, &afids), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, AddcReportRejectsNullOrEmptyArguments) {
  const auto record = make_record_without_events();
  auto size = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(nullptr, record.size(), nullptr, &size), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(cper_get_report_json(record.data(), 0, nullptr, &size), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, nullptr),
            AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, AddcAfidsReturnsEveryEventInRecordOrder) {
  const auto record = make_mi300_two_event_record();
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{28, 5}));
}

// The section offset comes from the record, so it can be odd. The decode must not depend
// on alignment; under UBSan a misaligned wide load in addc would report here.
TEST(GpuUnit, AddcAfidsDecodesSectionsAtAnUnalignedOffset) {
  const auto crashdump = make_mi300_crashdump_record(/*pad=*/1);
  const auto nonstandard = make_mi300_nonstandard_record(kAcaRegisterContext, /*pad=*/1);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(crashdump.data(), crashdump.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{28}));
  EXPECT_EQ(cper_get_afids(nonstandard.data(), nonstandard.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{28}));
}

TEST(GpuUnit, AddcAfidsDependOnTheNonStandardRegisterPayload) {
  auto record = make_mi300_nonstandard_record(kAcaRegisterContext);
  auto* body = reinterpret_cast<cper_sec_nonstd_err_body*>(
      &record[kBodyOffset + sizeof(cper_sec_nonstd_err_hdr)]);
  std::memset(body->err_ctx.reg_dump, 0, sizeof(body->err_ctx.reg_dump));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(afids.empty());
}

// addc treats the processor-error GUID as a processor record and reports no GPU event,
// even when the body has the GPU non-standard layout.
TEST(GpuUnit, AddcAfidsIgnoreAGpuLayoutUnderTheProcessorErrorGuid) {
  auto record = make_mi300_nonstandard_record(kAcaRegisterContext);
  auto* descriptor = reinterpret_cast<cper_sec_desc*>(&record[kCperHeaderSize]);
  std::memcpy(&descriptor->sec_type, kProcErrGuidBytes, sizeof(kProcErrGuidBytes));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(afids.empty());
}

// A section whose declared length runs past the record, in a record that ends against the
// guard page: an unbounded read of that section faults.
TEST(GpuUnit, AddcAfidsDoesNotReadPastASectionThatOverrunsTheRecord) {
  auto record = make_mi300_crashdump_record();
  auto* descriptor = reinterpret_cast<cper_sec_desc*>(&record[kCperHeaderSize]);
  descriptor->sec_length += kCperHeaderSize;

  auto region = GuardedRegion_t{record.size()};
  ASSERT_TRUE(region.is_ready()) << strerror(errno);
  std::memcpy(region.data(), record.data(), record.size());
  expect_no_fault_and_status(region.data(), record.size(), AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AddcAfidsDoesNotReadPastASectionOffsetOutsideTheRecord) {
  auto record = make_mi300_crashdump_record();
  auto* descriptor = reinterpret_cast<cper_sec_desc*>(&record[kCperHeaderSize]);
  descriptor->sec_offset = 0x7FFFFFFF;

  auto region = GuardedRegion_t{record.size()};
  ASSERT_TRUE(region.is_ready()) << strerror(errno);
  std::memcpy(region.data(), record.data(), record.size());
  expect_no_fault_and_status(region.data(), record.size(), AMDSMI_STATUS_UNEXPECTED_DATA);
}

// The context is created lazily and shared. A threadsafe death test re-executes the binary,
// so the first calls race on a fresh context instead of one an earlier test already created.
// Without ThreadSanitizer this is a smoke test: it checks that the results agree, not that the
// sharing is race-free.
static auto count_concurrent_mismatches() -> size_t {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kCallsPerThread = 50;
  const auto record = make_mi300_crashdump_record();
  auto mismatches = std::atomic<size_t>{0};
  auto workers = std::vector<std::thread>{};
  for (size_t t = 0; t < kThreadCount; ++t) {
    workers.emplace_back([&]() {
      for (size_t i = 0; i < kCallsPerThread; ++i) {
        auto afids = std::vector<int>{};
        const auto status = cper_get_afids(record.data(), record.size(), &afids);
        if ((status != AMDSMI_STATUS_SUCCESS) || (afids != std::vector<int>{28})) {
          ++mismatches;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  return mismatches.load();
}

TEST(GpuUnit, AddcAfidsAreStableUnderConcurrentCalls) {
  const auto saved_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(_exit((count_concurrent_mismatches() == 0) ? 0 : 1), testing::ExitedWithCode(0), "");
  GTEST_FLAG_SET(death_test_style, saved_style);
}

TEST(GpuUnit, AddcReportRejectsRecordLengthBelowHeader) {
  auto record = make_record_without_events();
  put_u32(&record, kHeaderRecordLengthOffset, static_cast<uint32_t>(kCperHeaderSize - 1));
  auto size = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &size),
            AMDSMI_STATUS_UNEXPECTED_SIZE);
}

TEST(GpuUnit, AddcReportRejectsRecordLengthPastBuffer) {
  auto record = make_record_without_events();
  put_u32(&record, kHeaderRecordLengthOffset, static_cast<uint32_t>(record.size() + 1));
  auto size = uint32_t{0};
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &size),
            AMDSMI_STATUS_UNEXPECTED_SIZE);
}

// A buffer larger than needed still reports the size of the text, not its own capacity.
TEST(GpuUnit, AddcReportSizeIsTheTextSizeForALargerBuffer) {
  const auto record = make_record_without_events();
  auto required = uint32_t{0};
  ASSERT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &required),
            AMDSMI_STATUS_SUCCESS);

  auto buffer = std::string((2 * static_cast<size_t>(required)), 'x');
  auto size = static_cast<uint32_t>(buffer.size());
  EXPECT_EQ(cper_get_report_json(record.data(), record.size(), buffer.data(), &size),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(size, required);
}

// The report carries the events themselves, not just a schema header.
TEST(GpuUnit, AddcReportContainsTheDecodedEvent) {
  const auto record = make_record_with_unclassified_event();
  auto size = uint32_t{0};
  ASSERT_EQ(cper_get_report_json(record.data(), record.size(), nullptr, &size),
            AMDSMI_STATUS_SUCCESS);
  auto buffer = std::string(size, '\0');
  ASSERT_EQ(cper_get_report_json(record.data(), record.size(), buffer.data(), &size),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(buffer.find(std::to_string(ADDC_UNCLASSIFIED_AFID)), std::string::npos);
  EXPECT_NE(buffer.find(kUnclassifiedFru), std::string::npos);
}

// amdgpu writes the crashdump section of a runtime fatal record 32 bytes shorter than
// sizeof(cper_sec_crashdump): the dump union stops after fatal_err.
constexpr size_t kAmdgpuFatalCrashdumpSize = 0xB0;
constexpr size_t kAmdgpuBootCrashdumpSize = 0xD0;

TEST(GpuUnit, AddcAfidsDecodeFatalCrashdumpSizedAsAmdgpuWritesIt) {
  const auto record = make_mi300_crashdump_record(0, kAmdgpuFatalCrashdumpSize);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{28}));
}

// One byte short of fatal_err cuts the register array, so the record is not well formed.
TEST(GpuUnit, AddcAfidsRejectFatalCrashdumpOneByteShortOfFatalErr) {
  const auto record = make_mi300_crashdump_record(0, (kAmdgpuFatalCrashdumpSize - 1));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_TRUE(afids.empty());
}

// A boot record declares a boot context whose 64-byte boot_err array fills the full
// section, where a runtime fatal record stops after fatal_err.
auto make_boot_crashdump_record(size_t section_size) -> std::vector<uint8_t> {
  constexpr size_t kBootMessageCount = 8;
  auto record = make_mi300_crashdump_record(0, section_size);
  reinterpret_cast<amdsmi_cper_hdr_t*>(record.data())->notify_type =
      *reinterpret_cast<const amdsmi_cper_guid_t*>(kBootNotifyGuidBytes);
  auto* section = reinterpret_cast<cper_sec_crashdump*>(&record[kBodyOffset]);
  section->data.reg_ctx_type = kBootRegisterContext;
  section->data.reg_arr_size = static_cast<uint16_t>(sizeof(uint64_t) * kBootMessageCount);
  const auto dump_offset =
      (offsetof(cper_sec_crashdump, data) + offsetof(cper_sec_crashdump_data, dump));
  const auto room = (section_size - dump_offset);
  std::memcpy(&record[kBodyOffset + dump_offset], kAcaRegisters,
              std::min(room, sizeof(uint64_t) * kBootMessageCount));
  return record;
}

TEST(GpuUnit, AddcAfidsDecodeBootCrashdumpSizedAsAmdgpuWritesIt) {
  const auto record = make_boot_crashdump_record(kAmdgpuBootCrashdumpSize);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids, (std::vector<int>{5}));
}

// A boot section sized only for fatal_err cannot hold the boot_err array it declares.
TEST(GpuUnit, AddcAfidsRejectBootCrashdumpSizedOnlyForFatalErr) {
  const auto record = make_boot_crashdump_record(kAmdgpuFatalCrashdumpSize);
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_UNEXPECTED_DATA);
  EXPECT_TRUE(afids.empty());
}

TEST(GpuUnit, AddcAfidsDoesNotReadPastANonStandardSectionThatOverrunsTheRecord) {
  auto record = make_mi300_nonstandard_record(kAcaRegisterContext);
  auto* descriptor = reinterpret_cast<cper_sec_desc*>(&record[kCperHeaderSize]);
  descriptor->sec_length += kCperHeaderSize;

  auto region = GuardedRegion_t{record.size()};
  ASSERT_TRUE(region.is_ready()) << strerror(errno);
  std::memcpy(region.data(), record.data(), record.size());
  expect_no_fault_and_status(region.data(), record.size(), AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuUnit, AddcAfidsAcceptEmptyRegisterArrayAndYieldNoAfid) {
  auto record = make_mi300_nonstandard_record(kAcaRegisterContext);
  auto* body = reinterpret_cast<cper_sec_nonstd_err_body*>(
      &record[kBodyOffset + sizeof(cper_sec_nonstd_err_hdr)]);
  body->err_ctx.reg_arr_size = 0;
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(afids.empty());
}

// The declared register count bounds what is decoded: with 8 of the 128 bytes declared, the
// ACA context has too few registers to classify an event, where the full array gives AFID 28.
TEST(GpuUnit, AddcAfidsUseOnlyTheDeclaredRegisters) {
  auto record = make_mi300_nonstandard_record(kAcaRegisterContext);
  auto* body = reinterpret_cast<cper_sec_nonstd_err_body*>(
      &record[kBodyOffset + sizeof(cper_sec_nonstd_err_hdr)]);
  body->err_ctx.reg_arr_size = static_cast<uint16_t>(sizeof(uint64_t));
  auto afids = std::vector<int>{};
  EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(afids.empty());
}

// The dump holds 128 bytes of registers; the full-array tests above are the accept side of
// this edge. One byte more is rejected in both register contexts. The section ends the record
// against a guard page, so reading the extra register faults instead of returning garbage.
TEST(GpuUnit, AddcAfidsRejectRegisterArrayOneByteOverTheDumpCapacity) {
  for (const auto register_context : {kAcaRegisterContext, kBootRegisterContext}) {
    SCOPED_TRACE(register_context);
    auto record = make_mi300_nonstandard_record(register_context);
    auto* body = reinterpret_cast<cper_sec_nonstd_err_body*>(
        &record[kBodyOffset + sizeof(cper_sec_nonstd_err_hdr)]);
    body->err_ctx.reg_arr_size = static_cast<uint16_t>(sizeof(body->err_ctx.reg_dump) + 1);

    auto region = GuardedRegion_t{record.size()};
    ASSERT_TRUE(region.is_ready()) << strerror(errno);
    std::memcpy(region.data(), record.data(), record.size());
    expect_no_fault_and_status(region.data(), record.size(), AMDSMI_STATUS_UNEXPECTED_DATA);
  }
}

// kInitialAfidCapacity in amd_smi_addc.cc. Exactly that many events fit the first call; one
// more, and well past it, run the retry with the true count.
constexpr size_t kAdapterInitialAfidCapacity = 16;

TEST(GpuUnit, AddcAfidsReturnEveryEventAroundTheInitialCapacity) {
  for (const auto event_count :
       {kAdapterInitialAfidCapacity, (kAdapterInitialAfidCapacity + 1), size_t{40}}) {
    SCOPED_TRACE(event_count);
    const auto record =
        make_mi300_event_record(std::vector<uint16_t>(event_count, kAcaRegisterContext));
    auto afids = std::vector<int>{};
    EXPECT_EQ(cper_get_afids(record.data(), record.size(), &afids), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(afids, std::vector<int>(event_count, 28));
  }
}

// The public wrappers need an initialised library; a host that cannot initialise it skips.
namespace {
class LibraryInit_t {
 public:
  LibraryInit_t() : m_status(amdsmi_init(AMDSMI_INIT_AMD_GPUS)) {}
  ~LibraryInit_t() {
    if (is_ready()) {
      amdsmi_shut_down();
    }
  }
  LibraryInit_t(const LibraryInit_t&) = delete;
  auto operator=(const LibraryInit_t&) -> LibraryInit_t& = delete;

  auto is_ready() const -> bool { return (m_status == AMDSMI_STATUS_SUCCESS); }
  auto status() const -> amdsmi_status_t { return m_status; }

 private:
  amdsmi_status_t m_status;
};
}  // namespace

TEST(GpuUnit, AddcPublicAfidsReportFullCountWhenTheArrayIsTooSmall) {
  const auto library = LibraryInit_t{};
  if (!library.is_ready()) {
    GTEST_SKIP() << "amdsmi_init status " << library.status();
  }
  auto record = make_mi300_two_event_record();
  auto afids = std::array<uint64_t, 1>{};
  auto num_afids = uint32_t{1};
  EXPECT_EQ(
      amdsmi_get_afids_from_cper(reinterpret_cast<char*>(record.data()),
                                 static_cast<uint32_t>(record.size()), afids.data(), &num_afids),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(afids[0], 28u);
  EXPECT_EQ(num_afids, 2u);
}

TEST(GpuUnit, AddcPublicAfidsRejectNullArguments) {
  const auto library = LibraryInit_t{};
  if (!library.is_ready()) {
    GTEST_SKIP() << "amdsmi_init status " << library.status();
  }
  auto record = make_mi300_crashdump_record();
  auto afids = std::array<uint64_t, 1>{};
  auto num_afids = uint32_t{1};
  auto* bytes = reinterpret_cast<char*>(record.data());
  const auto size = static_cast<uint32_t>(record.size());
  EXPECT_EQ(amdsmi_get_afids_from_cper(nullptr, size, afids.data(), &num_afids),
            AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_afids_from_cper(bytes, size, nullptr, &num_afids), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_afids_from_cper(bytes, size, afids.data(), nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, AddcPublicJsonSizeQueryThenFill) {
  const auto library = LibraryInit_t{};
  if (!library.is_ready()) {
    GTEST_SKIP() << "amdsmi_init status " << library.status();
  }
  auto record = make_mi300_crashdump_record();
  auto* bytes = reinterpret_cast<char*>(record.data());
  const auto size = static_cast<uint32_t>(record.size());
  auto json_size = uint32_t{0};
  ASSERT_EQ(amdsmi_get_cper_json(bytes, size, nullptr, &json_size), AMDSMI_STATUS_SUCCESS);
  ASSERT_GT(json_size, 1u);
  auto json = std::string(json_size, '\0');
  EXPECT_EQ(amdsmi_get_cper_json(bytes, size, json.data(), &json_size), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(json.back(), '\0');
}
