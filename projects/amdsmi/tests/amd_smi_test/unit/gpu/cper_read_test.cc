// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression tests for the CPER read path via
// amdsmi_get_gpu_cper_entries_by_path(); no GPU required.
// ROCM-25398: a zero-byte CPER node must not abort the process.
// ROCM-25954: an empty ring returns SUCCESS with zero entries, not an error.
//
// Also covers the structural bounds of the parser: a crafted record whose
// sec_cnt, sec_offset, or reg_arr_size points past the buffer, or whose fixed-width
// text fields carry no terminator, must be rejected per section rather than read
// or written through (CWE-125).

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_cper.h"
#include "amd_smi/impl/amd_smi_cper_testing.h"
#include "ras-decode/ras_decode_constants.h"

namespace {

// 4 MiB ring + 12 B header, matching the st_size reported in the ROCM-25954
// field report.
constexpr off_t kRingCapacity = 4194316;

// Runs the CPER read path against a file. Out-params report the final
// entry_count and buf_size.
amdsmi_status_t CallCperByPath(const char* path, uint64_t* out_entry_count = nullptr,
                               uint64_t* out_buf_size = nullptr) {
  std::vector<char> cper_data(4096, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(8, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path, 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count, &cursor,
      /*product_serial=*/0);

  if (out_entry_count) *out_entry_count = entry_count;
  if (out_buf_size) *out_buf_size = buf_size;
  return status;
}

// Restores the production read() seam however a test exits.
struct CperReadFnGuard {
  ~CperReadFnGuard() { cper_set_read_fn_for_testing(nullptr); }
};

ssize_t FakeReadZero(int, void*, size_t) { return 0; }  // empty ring

ssize_t FakeReadPartial(int, void* buf, size_t) {  // short read of non-record bytes
  std::memset(buf, 0, 100);
  return 100;
}

ssize_t FakeReadError(int, void*, size_t) {  // I/O failure
  errno = EIO;
  return -1;
}

// Sparse regular file: advertises `size` via st_size at no disk cost and passes
// the S_ISREG guard, matching the debugfs node's "capacity in st_size" shape.
// Fatal-asserts on setup failure, returning the path via out_path.
void MakeSparseFile(off_t size, std::string* out_path) {
  std::string tmpl = "/tmp/amdsmi_cper_cap_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  int rc = ftruncate(fd, size);
  close(fd);
  if (rc != 0) {
    unlink(tmpl.c_str());
    FAIL() << "failed to size temp file";
  }
  *out_path = tmpl;
}

// Minimal single-record CPER blob the parser accepts: "CPER" signature,
// 0xFFFFFFFF terminator, record_length == header size, severity 0 (matched by a
// full mask).
std::vector<char> MakeOneRecordBlob() {
  amdsmi_cper_hdr_t hdr{};
  std::memcpy(hdr.signature, "CPER", 4);
  hdr.signature_end = 0xFFFFFFFF;
  hdr.error_severity = AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED;
  hdr.record_length = sizeof(hdr);
  std::vector<char> blob(sizeof(hdr));
  std::memcpy(blob.data(), &hdr, sizeof(hdr));
  return blob;
}

// `count` identical single-record blobs concatenated.
std::vector<char> MakeRecordsBlob(size_t count) {
  std::vector<char> one = MakeOneRecordBlob();
  std::vector<char> blob;
  for (size_t i = 0; i < count; ++i) {
    blob.insert(blob.end(), one.begin(), one.end());
  }
  return blob;
}

// Writes bytes to a fresh temp file, returning its path via out_path.
// Fatal-asserts on setup failure.
void WriteTempFile(const std::vector<char>& bytes, std::string* out_path) {
  std::string tmpl = "/tmp/amdsmi_cper_rec_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  ssize_t written = write(fd, bytes.data(), bytes.size());
  close(fd);
  if (written != static_cast<ssize_t>(bytes.size())) {
    unlink(tmpl.c_str());
    FAIL() << "failed to write temp file";
  }
  *out_path = tmpl;
}

// Like CallCperByPath but with a caller-controlled buffer byte size and header
// slot count, to exercise the buffer-exhaustion return paths. out_cursor, when
// provided, reports the returned cursor.
amdsmi_status_t CallCperSized(const char* path, uint64_t buf_bytes, uint64_t slots,
                              uint64_t* out_entry_count, uint64_t* out_buf_size,
                              uint64_t* out_cursor = nullptr) {
  std::vector<char> cper_data(buf_bytes, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(slots, nullptr);
  uint64_t buf_size = buf_bytes;
  uint64_t entry_count = slots;
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path, 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count, &cursor,
      /*product_serial=*/0);

  if (out_entry_count) *out_entry_count = entry_count;
  if (out_buf_size) *out_buf_size = buf_size;
  if (out_cursor) *out_cursor = cursor;
  return status;
}

}  // namespace

// Zero-size regular file: st_size == 0, so read(fd, buf, 0) returns 0 trivially.
// Hits the same empty-ring success branch as the field case (large st_size,
// read() returns 0) and must not abort the process.
TEST(GpuUnit, CperReadZeroSizeFile) {
  std::string tmpl = "/tmp/amdsmi_cper_zero_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  close(fd);  // leave it empty -> st_size == 0

  uint64_t entry_count = 99;  // sentinel; the call must overwrite both to 0
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(tmpl.c_str(), &entry_count, &buf_size);
  unlink(tmpl.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Characterizes how a regular/CI filesystem differs from the debugfs target.
// ftruncate to a non-zero size with no payload gives st_size == 4096; a regular
// filesystem then returns 4096 zero bytes on read() (a full read of the hole),
// not 0 like the empty debugfs ring. Those zero bytes hold no CPER signature, so
// the result is still SUCCESS with no records. This is why the empty-ring shape
// (read() == 0 while st_size advertises ring capacity) needs the injectable read
// seam in CperEmptyRingAdvertisedCapacityShortRead to reproduce faithfully, and
// why CperReadZeroSizeFile pins the st_size == 0 corner with a real read().
TEST(GpuUnit, CperNonZeroFileRealReadHasNoRecords) {
  std::string path;
  MakeSparseFile(4096, &path);  // st_size == 4096, no payload written
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 99;  // sentinels; the call must overwrite both to 0
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Missing path -> NOT_SUPPORTED (stat() fails), no crash. Create then remove a
// temp file so the path is guaranteed absent (no hardcoded path another process
// might have created).
TEST(GpuUnit, CperReadMissingFile) {
  std::string tmpl = "/tmp/amdsmi_cper_missing_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  close(fd);
  unlink(tmpl.c_str());

  amdsmi_status_t status = CallCperByPath(tmpl.c_str());
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
}

// Happy path: a well-formed single-record file parses to one entry. Guards the
// read path against regressions in the empty/error handling around it.
TEST(GpuUnit, CperParsesSingleRecord) {
  std::string path;
  WriteTempFile(MakeOneRecordBlob(), &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 1u);
  EXPECT_GT(buf_size, 0u);
}

// Faithful ROCM-25954 repro: st_size advertises the 4 MiB ring capacity while
// read() returns 0 on an empty ring. Must be SUCCESS with zero entries.
TEST(GpuUnit, CperEmptyRingAdvertisedCapacityShortRead) {
  CperReadFnGuard guard;
  cper_set_read_fn_for_testing(&FakeReadZero);

  std::string path;
  MakeSparseFile(kRingCapacity, &path);
  ASSERT_FALSE(path.empty());
  uint64_t entry_count = 99;
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Partial read (0 < bytes_read < st_size) of non-record bytes: accepted as
// success; pin that no records are parsed and the out-params are zeroed.
TEST(GpuUnit, CperPartialReadNoRecords) {
  CperReadFnGuard guard;
  cper_set_read_fn_for_testing(&FakeReadPartial);

  std::string path;
  MakeSparseFile(kRingCapacity, &path);
  ASSERT_FALSE(path.empty());
  uint64_t entry_count = 99;
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// A real read() failure (returns -1) must still surface FILE_ERROR.
TEST(GpuUnit, CperReadErrorIsFileError) {
  CperReadFnGuard guard;
  cper_set_read_fn_for_testing(&FakeReadError);

  std::string path;
  MakeSparseFile(kRingCapacity, &path);
  ASSERT_FALSE(path.empty());
  amdsmi_status_t status = CallCperByPath(path.c_str());
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_FILE_ERROR);
}

// A record larger than the caller's buffer yields OUT_OF_RESOURCES with nothing
// copied and the out-params zeroed.
TEST(GpuUnit, CperFirstRecordExceedsBufferOutOfResources) {
  std::string path;
  WriteTempFile(MakeOneRecordBlob(), &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  // Non-zero buffer, but smaller than one record (sizeof(amdsmi_cper_hdr_t)).
  static_assert(sizeof(amdsmi_cper_hdr_t) > 64,
                "test assumes a 64-byte buffer is smaller than one CPER record");
  amdsmi_status_t status =
      CallCperSized(path.c_str(), /*buf_bytes=*/64, /*slots=*/8, &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Two records with a buffer that fits only one: the first is copied and
// MORE_DATA is returned with the partial entry_count/buf_size.
TEST(GpuUnit, CperSecondRecordOverflowsBufferMoreData) {
  std::string path;
  WriteTempFile(MakeRecordsBlob(2), &path);
  ASSERT_FALSE(path.empty());

  const uint64_t one_record = sizeof(amdsmi_cper_hdr_t);
  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  uint64_t cursor = 0;
  // Room for one record plus a partial second, with ample header slots so the
  // byte-buffer limit (not the slot count) is what trips.
  amdsmi_status_t status =
      CallCperSized(path.c_str(), one_record + 16, /*slots=*/8, &entry_count, &buf_size, &cursor);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_MORE_DATA);
  EXPECT_EQ(entry_count, 1u);
  EXPECT_EQ(buf_size, one_record);
  EXPECT_EQ(cursor, 1u);
}

// Two records with ample byte budget but only one header slot: the slot count,
// not the byte buffer, is what trips. The first is copied and MORE_DATA is
// returned with one entry and a non-zero buf_size.
TEST(GpuUnit, CperSlotExhaustionMoreData) {
  std::string path;
  WriteTempFile(MakeRecordsBlob(2), &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  uint64_t cursor = 0;
  amdsmi_status_t status = CallCperSized(path.c_str(), /*buf_bytes=*/8192, /*slots=*/1,
                                         &entry_count, &buf_size, &cursor);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_MORE_DATA);
  EXPECT_EQ(entry_count, 1u);
  EXPECT_EQ(buf_size, sizeof(amdsmi_cper_hdr_t));
  EXPECT_EQ(cursor, 1u);
}

// Invalid arguments are rejected with OUT_OF_RESOURCES before any file read.
TEST(GpuUnit, CperByPathRejectsInvalidArgs) {
  std::string path;
  WriteTempFile(MakeOneRecordBlob(), &path);
  ASSERT_FALSE(path.empty());

  std::vector<char> data(256, 0);
  std::vector<amdsmi_cper_hdr_t*> hdrs(4, nullptr);
  const uint32_t mask = 0xFFFFFFFF;
  uint64_t bs = 0;
  uint64_t ec = 0;
  uint64_t cursor = 0;

  // null path: the guard zeroes the valid out-params before returning.
  bs = data.size();
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(nullptr, mask, data.data(), &bs, hdrs.data(), &ec,
                                                &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(bs, 0u);
  EXPECT_EQ(ec, 0u);
  // null cper_data
  bs = data.size();
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, nullptr, &bs, hdrs.data(), &ec,
                                                &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null buf_size
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), nullptr,
                                                hdrs.data(), &ec, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null entry_count
  bs = data.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                nullptr, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // zero buf_size
  bs = 0;
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                &ec, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // zero entry_count
  bs = data.size();
  ec = 0;
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                &ec, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null cper_hdrs
  bs = data.size();
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, nullptr, &ec,
                                                &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null cursor
  bs = data.size();
  ec = hdrs.size();
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                &ec, nullptr, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);

  unlink(path.c_str());
}

namespace {

// Mirrors the GUID_INIT macro in amd_smi_cper.cc (mixed-endian EFI encoding) so
// the literals below diff argument-for-argument against the production ones.
constexpr amdsmi_cper_guid_t MakeGuid(uint32_t a, uint16_t b, uint16_t c, unsigned char d0,
                                      unsigned char d1, unsigned char d2, unsigned char d3,
                                      unsigned char d4, unsigned char d5, unsigned char d6,
                                      unsigned char d7) {
  return {{static_cast<unsigned char>(a), static_cast<unsigned char>(a >> 8),
           static_cast<unsigned char>(a >> 16), static_cast<unsigned char>(a >> 24),
           static_cast<unsigned char>(b), static_cast<unsigned char>(b >> 8),
           static_cast<unsigned char>(c), static_cast<unsigned char>(c >> 8), d0, d1, d2, d3, d4,
           d5, d6, d7}};
}

constexpr amdsmi_cper_guid_t kCrashdumpGuid =  // AMD_OOB_CRASHDUMP
    MakeGuid(0x32AC0C78, 0x2623, 0x48F6, 0xB0, 0xD0, 0x73, 0x65, 0x72, 0x5F, 0xD6, 0xAE);
constexpr amdsmi_cper_guid_t kNonStandardGuid =  // AMD_GPU_NONSTANDARD_ERROR
    MakeGuid(0x32AC0C78, 0x2623, 0x48F6, 0x81, 0xA2, 0xAC, 0x69, 0x17, 0x80, 0x55, 0x1D);
constexpr amdsmi_cper_guid_t kProcErrGuid =  // PROC_ERR_SECTION_TYPE
    MakeGuid(0xDC3EA0B0, 0xA144, 0x4797, 0xB9, 0x5B, 0x53, 0xFA, 0x24, 0x2B, 0x6E, 0x1D);

// Tracks the production static_assert that ties the reg_dump capacity to the
// register count decode_afid accepts, so both stay on the same constant.
constexpr uint16_t kMaxRegArrayBytes =
    static_cast<uint16_t>(RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES * sizeof(uint64_t));
constexpr uint16_t kOneOverRegArrayBytes =
    static_cast<uint16_t>((RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES + 1) * sizeof(uint64_t));
// The guard converts bytes to registers by dividing, so its edge is not at the
// dump's byte capacity: everything short of the next whole register still counts
// as RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES and must be accepted.
constexpr uint16_t kLargestAcceptedRegArrayBytes =
    static_cast<uint16_t>((kMaxRegArrayBytes + sizeof(uint64_t)) - 1);

// The two register context types the decoder routes on. ACA takes only 4 or 16
// registers and drops any other length before reading one; boot reads the count
// it is handed, so the capacity guard is the only bound on that path.
constexpr uint16_t kAcaRegisterContext = 1;
constexpr uint16_t kBootRegisterContext = 9;

// A zeroed dump decodes the same whatever index each byte lands at, so it cannot
// show that a register reaches the decoder where the record put it. These place
// the fields decode_afid keys off at distinct indices for either dump shape.
constexpr uint64_t kAcaRegisterPattern[RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES] = {
    0, 0xB400000000000000ULL, 0x0000000000001234ULL, 0,
    0, 0x0096000000000000ULL, 0x000000000000ABCDULL};
constexpr uint64_t kFatalRegisterPattern[] = {0xB400000000000000ULL, 0x0000000000001234ULL,
                                              0x0096000000000000ULL, 0x000000000000ABCDULL};

constexpr size_t kDescTableOffset = sizeof(amdsmi_cper_hdr_t);

struct cper_sec_desc* DescAt(std::vector<char>* buf, size_t idx) {
  return reinterpret_cast<struct cper_sec_desc*>((buf->data() + kDescTableOffset) +
                                                 (idx * sizeof(struct cper_sec_desc)));
}

// Fills the header fields the parser requires. record_length and sec_cnt stay
// caller-controlled because they are what these tests vary.
void InitHeader(std::vector<char>* buf, uint16_t sec_cnt, uint32_t record_length) {
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf->data());
  std::memcpy(hdr->signature, "CPER", 4);
  hdr->signature_end = 0xFFFFFFFF;
  hdr->error_severity = AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED;
  hdr->sec_cnt = sec_cnt;
  hdr->record_length = record_length;
}

constexpr size_t kCrashdumpSecOffset = (kDescTableOffset + (3 * sizeof(struct cper_sec_desc)));
constexpr size_t kCrashdumpRecordSize = (kCrashdumpSecOffset + sizeof(struct cper_sec_crashdump));

// [header][desc0][desc1][desc2][crashdump]. desc0 keeps an all-zero (unknown)
// section type and yields nothing; desc1 and desc2 both decode the crashdump, so
// the AFID count reports how far the descriptor walk got. sec_pad slides the
// section along by bytes, so an odd value lands dump.fatal_err on an address no
// wider type is aligned for.
std::vector<char> MakeCrashdumpRecord(uint32_t record_length, size_t sec_pad = 0) {
  const size_t sec_offset = (kCrashdumpSecOffset + sec_pad);
  std::vector<char> buf(sec_offset + sizeof(struct cper_sec_crashdump), 0);
  InitHeader(&buf, /*sec_cnt=*/3, record_length);
  for (size_t i = 1; i < 3; ++i) {
    struct cper_sec_desc* desc = DescAt(&buf, i);
    desc->sec_type = kCrashdumpGuid;
    desc->sec_offset = static_cast<uint32_t>(sec_offset);
  }
  auto* crashdump = reinterpret_cast<struct cper_sec_crashdump*>(buf.data() + sec_offset);
  crashdump->data.reg_ctx_type = kAcaRegisterContext;
  static_assert(sizeof(kFatalRegisterPattern) == sizeof(crashdump->data.dump.fatal_err),
                "Pattern must fill the whole fatal_err dump, otherwise part of it stays zero");
  std::memcpy(&crashdump->data.dump.fatal_err, kFatalRegisterPattern,
              sizeof(kFatalRegisterPattern));
  return buf;
}

// [header][desc0][sec_pad][non-standard section]. reg_arr_size is
// caller-controlled so a test can claim more registers than reg_dump holds, and
// reg_ctx_type so a test can pick the branch that reads them. sec_pad slides the
// section along by bytes, so an odd value lands reg_dump on an address no wider
// type is aligned for.
std::vector<char> MakeNonStandardRecord(uint16_t reg_arr_size, size_t sec_pad = 0,
                                        uint16_t reg_ctx_type = kAcaRegisterContext) {
  const size_t sec_offset = (kDescTableOffset + sizeof(struct cper_sec_desc) + sec_pad);
  constexpr size_t kSecSize =
      (sizeof(struct cper_sec_nonstd_err_hdr) + sizeof(struct cper_sec_nonstd_err_body));
  std::vector<char> buf(sec_offset + kSecSize, 0);
  InitHeader(&buf, /*sec_cnt=*/1, static_cast<uint32_t>(buf.size()));

  struct cper_sec_desc* desc = DescAt(&buf, 0);
  desc->sec_type = kNonStandardGuid;
  desc->sec_offset = static_cast<uint32_t>(sec_offset);

  auto* body = reinterpret_cast<struct cper_sec_nonstd_err_body*>(
      buf.data() + sec_offset + sizeof(struct cper_sec_nonstd_err_hdr));
  body->err_ctx.reg_ctx_type = reg_ctx_type;
  body->err_ctx.reg_arr_size = reg_arr_size;
  static_assert(sizeof(kAcaRegisterPattern) == sizeof(body->err_ctx.reg_dump),
                "Pattern must fill the whole reg_dump, otherwise part of it stays zero");
  std::memcpy(body->err_ctx.reg_dump, kAcaRegisterPattern, sizeof(kAcaRegisterPattern));
  return buf;
}

}  // namespace

// Positive control for the truncation test below: with record_length covering
// the whole descriptor table, both crashdump descriptors decode.
TEST(GpuUnit, CperDecodeDecodesEveryDescriptorInsideTheRecord) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 2u);
}

// The same bytes with record_length claiming only one descriptor. sec_cnt still
// says three, so an unbounded walk decodes desc1 and desc2 from past the record
// end (2 AFIDs, as the control above shows); the walk must stop at desc1.
TEST(GpuUnit, CperDecodeStopsAtTruncatedDescriptorTable) {
  std::vector<char> buf =
      MakeCrashdumpRecord(static_cast<uint32_t>(kDescTableOffset + sizeof(struct cper_sec_desc)));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_TRUE(afids.empty());
}

// record_length exactly at the header size is the smallest value that is not a
// reject: the record is well formed and simply carries no descriptor table, so
// the walk must produce nothing rather than read the first descriptor.
TEST(GpuUnit, CperDecodeStopsAtRecordLengthExactlyTheHeaderSize) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(sizeof(amdsmi_cper_hdr_t)));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_TRUE(afids.empty());
}

// sec_cnt of zero ends the walk before it starts, whatever the descriptors that
// follow happen to contain. The control decodes 2 AFIDs from these same bytes.
TEST(GpuUnit, CperDecodeDecodesNothingWhenSectionCountIsZero) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf.data());
  hdr->sec_cnt = 0;

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_TRUE(afids.empty());
}

// A descriptor whose sec_offset points far past the buffer must be skipped: the
// section pointer is out of bounds and must never be dereferenced (CWE-125).
// Unguarded, this offset faults rather than returning garbage, which would take
// the whole runner down and silently skip every case after it. The forked child
// keeps that failure inside one test.
TEST(GpuUnit, CperDecodeSkipsSectionWithOutOfBoundsOffset) {
  std::vector<char> buf(kDescTableOffset + sizeof(struct cper_sec_desc), 0);
  InitHeader(&buf, /*sec_cnt=*/1, static_cast<uint32_t>(buf.size()));
  struct cper_sec_desc* desc = DescAt(&buf, 0);
  desc->sec_type = kCrashdumpGuid;
  desc->sec_offset = 0x7FFFFFFF;  // wildly out of range

  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());
  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, buf.size());
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// sec_offset lands inside the record but the section it names does not fit: the
// crashdump body needs more bytes than remain. Distinct from the wild offset
// above, which fails on the offset alone. Unbounded, the decoder reads a whole
// crashdump struct starting one byte before the record end.
TEST(GpuUnit, CperDecodeSkipsSectionThatStartsInsideButOverrunsTheRecord) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf.data());
  // One byte of room at sec_offset, against a section that needs the full struct.
  hdr->record_length = static_cast<uint32_t>(kCrashdumpSecOffset + 1);

  // Forked for the same reason as the wild-offset case: unguarded, the section
  // pointer is null and the dump faults.
  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, buf.size());
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// The same overrun on the non-standard path, which reaches the bound through an
// explicit extent rather than through at<> because the body is a flexible array
// member.
TEST(GpuUnit, CperDecodeSkipsNonStandardSectionThatOverrunsTheRecord) {
  std::vector<char> buf = MakeNonStandardRecord(kMaxRegArrayBytes);
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf.data());
  const size_t sec_offset = (kDescTableOffset + sizeof(struct cper_sec_desc));
  hdr->record_length = static_cast<uint32_t>(sec_offset + 1);

  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, buf.size());
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// cper_decode takes record_length and buf_size separately, and a caller that
// hands over less than the record claims must be believed over the record. The
// control above decodes 2 AFIDs from these same bytes at a full buf_size; here
// everything past the first descriptor is the caller's, not the record's.
TEST(GpuUnit, CperDecodeBoundsTheRecordByTheCallersBufferSize) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  const size_t short_size = (kDescTableOffset + sizeof(struct cper_sec_desc));
  EXPECT_TRUE(cper_decode(hdr, short_size).empty());
  // Control: the identical record at its full buffer size still decodes.
  EXPECT_EQ(cper_decode(hdr, buf.size()).size(), 2u);
}

// sec_offset comes from the record here too, so the crashdump body lands wherever
// it points while the decoder reads dump.fatal_err as uint64_t. Shifting the
// section by one byte must not change what decodes. Same standing as the
// non-standard case below: x86-64 absorbs the unaligned loads, so this passes
// either way here. It detects the fault under -fsanitize=undefined, which
// ADDRESS_SANITIZER=ON does not turn on, and on targets that trap.
TEST(GpuUnit, CperDecodeDecodesCrashdumpSectionAtAnUnalignedOffset) {
  std::vector<char> aligned = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  std::vector<char> unaligned =
      MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize + 1), /*sec_pad=*/1);

  std::vector<int> expected =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(aligned.data()), aligned.size());
  std::vector<int> actual =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(unaligned.data()), unaligned.size());

  ASSERT_EQ(expected.size(), 2u);
  EXPECT_EQ(actual, expected);
}

// Positive control for the reject and misalignment tests below: the largest
// register array the decoder accepts yields one AFID.
TEST(GpuUnit, CperDecodeDecodesNonStandardSectionWithFullRegisterArray) {
  std::vector<char> buf = MakeNonStandardRecord(kMaxRegArrayBytes);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 1u);
}

// Processor-error sections share the non-standard section layout and decode
// through the same branch. A mistyped GUID on either constant routes the record
// to "unknown error type" and yields nothing, which no other test would catch.
TEST(GpuUnit, CperDecodeDecodesProcErrSectionThroughTheNonStandardPath) {
  std::vector<char> buf = MakeNonStandardRecord(kMaxRegArrayBytes);
  DescAt(&buf, 0)->sec_type = kProcErrGuid;
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 1u);
}

// An empty register array is in bounds and stays in bounds, so the section is
// read; decode_error_info takes only 4 or 16 registers, so it yields no AFID.
// Pins the low end of the range the capacity guard brackets.
TEST(GpuUnit, CperDecodeAcceptsEmptyRegisterArrayAndYieldsNoAfid) {
  std::vector<char> buf = MakeNonStandardRecord(/*reg_arr_size=*/0);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_TRUE(afids.empty());
}

// sec_offset comes from the record, so nothing stops it landing reg_dump on an
// odd address while the decoder reads those bytes as uint64_t. Shifting the
// section by one byte must not change what decodes. x86-64 absorbs unaligned
// loads, so this passes either way here. It detects the fault under
// -fsanitize=undefined, which ADDRESS_SANITIZER=ON does not turn on, and on
// targets that trap.
TEST(GpuUnit, CperDecodeDecodesNonStandardSectionAtAnUnalignedOffset) {
  std::vector<char> aligned = MakeNonStandardRecord(kMaxRegArrayBytes);
  std::vector<char> unaligned = MakeNonStandardRecord(kMaxRegArrayBytes, /*sec_pad=*/1);

  std::vector<int> expected =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(aligned.data()), aligned.size());
  std::vector<int> actual =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(unaligned.data()), unaligned.size());

  ASSERT_EQ(expected.size(), 1u);
  EXPECT_EQ(actual, expected);
}

// reg_arr_size claims 8191 registers against a 16-register reg_dump. The section
// is rejected: unbounded, the decoder reads ~64 KB past reg_dump and then drops
// the AFID anyway because 8191 is not a length decode_error_info accepts, so
// rejecting keeps the decode result identical while removing the overread. The
// one-over case below pins the same guard at its edge; this one pins the far end,
// where an unbounded read leaves the buffer entirely.
//
// Forked for the same reason as the sec_offset cases: unguarded, the overread
// runs off a stack buffer and aborts the runner rather than returning garbage.
TEST(GpuUnit, CperDecodeRejectsOversizedRegisterArraySize) {
  std::vector<char> buf = MakeNonStandardRecord(/*reg_arr_size=*/0xFFFF);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, buf.size());
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// The guard counts registers, not bytes, so its edge sits one whole register
// past the dump's byte capacity rather than at it: 135 bytes still describes 16
// registers and must decode, 136 describes 17 and must not. Only the accept side
// shows in the return value, because 17 is not a length decode_error_info takes
// and an unguarded build discards the AFID after reading 8 bytes past reg_dump.
// The fork is what turns that read into an observable failure.
TEST(GpuUnit, CperDecodeBracketsTheRegisterArrayCapacityByRegisterCount) {
  std::vector<char> accepted = MakeNonStandardRecord(kLargestAcceptedRegArrayBytes);
  const auto* accepted_hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(accepted.data());
  EXPECT_EQ(cper_decode(accepted_hdr, accepted.size()).size(), 1u);

  std::vector<char> rejected = MakeNonStandardRecord(kOneOverRegArrayBytes);
  const auto* rejected_hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(rejected.data());
  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(rejected_hdr, rejected.size());
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// Boot register context is the branch the capacity guard actually protects: ACA
// context drops any length other than 4 or 16 before reading a register, so the
// ACA cases above would still return an empty list unguarded. Boot decodes
// whatever count it is handed. Positive control for the reject below.
TEST(GpuUnit, CperDecodeDecodesNonStandardSectionUnderBootRegisterContext) {
  std::vector<char> buf =
      MakeNonStandardRecord(kMaxRegArrayBytes, /*sec_pad=*/0, kBootRegisterContext);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 1u);
}

// The same 8191-register claim under boot context. Nothing downstream discards
// the result here, so the guard is the only thing standing between the record
// and a ~64 KB read off a 128-byte dump.
TEST(GpuUnit, CperDecodeRejectsOversizedRegisterArraySizeUnderBootRegisterContext) {
  std::vector<char> buf =
      MakeNonStandardRecord(/*reg_arr_size=*/0xFFFF, /*sec_pad=*/0, kBootRegisterContext);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, buf.size());
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// A section decodes to one AFID whether its registers carry data or not, so a
// size assertion cannot tell whether the dump reached the decoder at the indices
// the record wrote it to. Comparing a patterned decode against a zeroed one says
// that without pinning either result to an entry in the error map.
TEST(GpuUnit, CperDecodeNonStandardAfidDependsOnTheRegisterPayload) {
  std::vector<char> patterned = MakeNonStandardRecord(kMaxRegArrayBytes);
  std::vector<char> zeroed = patterned;
  auto* body = reinterpret_cast<struct cper_sec_nonstd_err_body*>(
      (zeroed.data() + kDescTableOffset + sizeof(struct cper_sec_desc)) +
      sizeof(struct cper_sec_nonstd_err_hdr));
  std::memset(body->err_ctx.reg_dump, 0, sizeof(body->err_ctx.reg_dump));

  std::vector<int> from_pattern =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(patterned.data()), patterned.size());
  std::vector<int> from_zero =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(zeroed.data()), zeroed.size());

  ASSERT_EQ(from_pattern.size(), 1u);
  ASSERT_EQ(from_zero.size(), 1u);
  EXPECT_NE(from_pattern, from_zero);
}

// The same payload dependence on the crashdump path, which reaches its registers
// through fatal_err rather than reg_dump.
TEST(GpuUnit, CperDecodeCrashdumpAfidDependsOnTheRegisterPayload) {
  std::vector<char> patterned = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  std::vector<char> zeroed = patterned;
  auto* crashdump =
      reinterpret_cast<struct cper_sec_crashdump*>(zeroed.data() + kCrashdumpSecOffset);
  std::memset(&crashdump->data.dump.fatal_err, 0, sizeof(crashdump->data.dump.fatal_err));

  std::vector<int> from_pattern =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(patterned.data()), patterned.size());
  std::vector<int> from_zero =
      cper_decode(reinterpret_cast<const amdsmi_cper_hdr_t*>(zeroed.data()), zeroed.size());

  ASSERT_EQ(from_pattern.size(), 2u);
  ASSERT_EQ(from_zero.size(), 2u);
  EXPECT_NE(from_pattern, from_zero);
}

// cper_decode carries its own header-size guard rather than assuming
// amdsmi_get_afids_from_cper ran first, so it is safe for any (pointer, size). Pins
// the return contract only; the guard-page test below is what proves the read itself
// never happens.
TEST(GpuUnit, CperDecodeRejectsBufferSmallerThanHeader) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  EXPECT_TRUE(cper_decode(hdr, sizeof(amdsmi_cper_hdr_t) - 1).empty());
  EXPECT_TRUE(cper_decode(hdr, 0).empty());
  // Control: the same record at a full header size still decodes.
  EXPECT_EQ(cper_decode(hdr, buf.size()).size(), 2u);
}

// The signature is checked here too, not only in amdsmi_get_afids_from_cper, so
// unsignatured bytes cannot walk as a record and report AFIDs no device produced.
// Decoding the record before and after flipping the signature pins that the
// signature alone separates the two outcomes.
TEST(GpuUnit, CperDecodeRejectsBufferWithoutTheCperSignature) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf.data());
  ASSERT_EQ(cper_decode(hdr, buf.size()).size(), 2u);

  std::memcpy(hdr->signature, "NOPE", 4);
  EXPECT_TRUE(cper_decode(hdr, buf.size()).empty());
}

// An overread off a heap buffer lands in adjacent allocation and returns garbage
// instead of faulting, so a plain assertion cannot see it. Ending the short
// buffer flush against an unmapped page turns the read into a fault, which the
// forked child reports as a non-zero exit. Without the guard in cper_decode, the
// record_length load at offset 20 of a 8-byte buffer lands on the guard page.
TEST(GpuUnit, CperDecodeDoesNotReadPastAShortBuffer) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  char* region = static_cast<char*>(
      mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(region, MAP_FAILED) << "mmap failed: " << strerror(errno);
  ASSERT_EQ(mprotect(region + page, page, PROT_NONE), 0) << "mprotect failed: " << strerror(errno);

  constexpr size_t kReadable = 8;  // below sizeof(amdsmi_cper_hdr_t)
  char* buf = region + page - kReadable;
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf);
  std::memcpy(hdr->signature, "CPER", 4);

  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, kReadable);
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");

  munmap(region, 2 * page);
}

// fru_id and fru_text are fixed-width record fields with no guaranteed terminator,
// and the section-descriptor dump streams both. Every structural check passes
// here: sec_cnt is 1, record_length is exact, the one descriptor is fully inside
// the record. Only the terminator is missing, so an unbounded stream runs off the
// descriptor and, on a record that ends against an unmapped page, off the buffer.
TEST(GpuUnit, CperDecodeDoesNotReadPastUnterminatedFruFields) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  char* region = static_cast<char*>(
      mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(region, MAP_FAILED) << "mmap failed: " << strerror(errno);
  ASSERT_EQ(mprotect(region + page, page, PROT_NONE), 0) << "mprotect failed: " << strerror(errno);

  const size_t record = (kDescTableOffset + sizeof(struct cper_sec_desc));
  char* buf = region + page - record;  // the record ends flush against the guard page
  std::memset(buf, 0, record);

  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf);
  std::memcpy(hdr->signature, "CPER", 4);
  hdr->signature_end = 0xFFFFFFFF;
  hdr->sec_cnt = 1;
  hdr->record_length = static_cast<uint32_t>(record);

  auto* desc = reinterpret_cast<struct cper_sec_desc*>(buf + kDescTableOffset);
  std::memset(desc->fru_id, 'A', sizeof(desc->fru_id));
  std::memset(desc->fru_text, 'B', sizeof(desc->fru_text));

  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, record);
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");

  munmap(region, 2 * page);
}

// sec_cnt outruns the descriptor table the record carries. Every other bound is
// satisfied, and the section checks cannot intervene: get_sec_desc_type reads
// desc1 before any of them run. The heap-backed truncation tests cannot isolate
// this, because their descriptors also name a section that fails its own bound,
// so a decoder with no descriptor check still returns an empty list. Here desc1
// begins exactly on the guard page, so only the descriptor bound keeps the walk
// from faulting.
TEST(GpuUnit, CperDecodeDoesNotReadPastTheDescriptorTable) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  char* region = static_cast<char*>(
      mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(region, MAP_FAILED) << "mmap failed: " << strerror(errno);
  ASSERT_EQ(mprotect(region + page, page, PROT_NONE), 0) << "mprotect failed: " << strerror(errno);

  const size_t record = (kDescTableOffset + sizeof(struct cper_sec_desc));
  char* buf = region + page - record;  // desc1 would start on the guard page
  std::memset(buf, 0, record);

  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf);
  std::memcpy(hdr->signature, "CPER", 4);
  hdr->signature_end = 0xFFFFFFFF;
  hdr->sec_cnt = 2;  // one more than the record holds
  hdr->record_length = static_cast<uint32_t>(record);

  EXPECT_EXIT(
      {
        std::vector<int> afids = cper_decode(hdr, record);
        _exit(afids.empty() ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");

  munmap(region, 2 * page);
}

// record_length undercutting the header is the input class that walks the serial
// injection off the end of the caller's buffer in the by-path test below. Here it
// must bound the view to nothing rather than reach the descriptor table at all.
TEST(GpuUnit, CperDecodeRejectsRecordLengthBelowHeader) {
  std::vector<char> buf = MakeCrashdumpRecord(/*record_length=*/4);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  EXPECT_TRUE(cper_decode(hdr, buf.size()).empty());
}

// A record whose record_length undercuts the header: the scan must drop it. Left
// in the list, the memcpy copies record_length bytes and then the serial
// injection reads record_length and sec_cnt back from bytes it never wrote,
// deriving a garbage bound and writing fru_id past the caller's buffer.
TEST(GpuUnit, CperByPathRejectsRecordLengthBelowHeader) {
  std::vector<char> blob = MakeOneRecordBlob();
  reinterpret_cast<amdsmi_cper_hdr_t*>(blob.data())->record_length = 4;
  std::string path;
  WriteTempFile(blob, &path);
  ASSERT_FALSE(path.empty());

  constexpr char kGuard = static_cast<char>(0xAB);
  std::vector<char> cper_data(4096, kGuard);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(4, nullptr);
  uint64_t buf_size = 4;  // exactly the bogus record_length
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path.c_str(), 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count,
      &cursor, /*product_serial=*/1234);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
  for (size_t i = 0; i < cper_data.size(); ++i) {
    ASSERT_EQ(cper_data[i], kGuard) << "byte " << i << " written for a rejected record";
  }
}

// A severity of 200 is wider than the 32-bit mask, so no mask can select it. Under
// the old (1 << severity) test the shift wrapped to bit 8, which a full mask has
// set, and the record was accepted.
TEST(GpuUnit, CperByPathRejectsSeverityWiderThanTheMask) {
  std::vector<char> blob = MakeOneRecordBlob();
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(blob.data());
  constexpr uint32_t kOutOfRangeSeverity = 200;
  std::memcpy(&hdr->error_severity, &kOutOfRangeSeverity, sizeof(kOutOfRangeSeverity));
  std::string path;
  WriteTempFile(blob, &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// A header-only record claiming 64 descriptors: the serial injection walks the
// descriptor table of the copy it just made, so it must stop at record_length
// instead of writing fru_id past the copied record.
TEST(GpuUnit, CperInjectSerialStopsAtRecordEnd) {
  std::vector<char> blob = MakeOneRecordBlob();
  reinterpret_cast<amdsmi_cper_hdr_t*>(blob.data())->sec_cnt = 64;
  std::string path;
  WriteTempFile(blob, &path);
  ASSERT_FALSE(path.empty());

  constexpr char kGuard = static_cast<char>(0xAB);
  std::vector<char> cper_data(8192, kGuard);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(4, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path.c_str(), 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count,
      &cursor, /*product_serial=*/1234);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(entry_count, 1u);
  ASSERT_EQ(buf_size, sizeof(amdsmi_cper_hdr_t));
  for (size_t i = buf_size; i < cper_data.size(); ++i) {
    ASSERT_EQ(cper_data[i], kGuard) << "byte " << i << " written past the copied record";
  }
}
