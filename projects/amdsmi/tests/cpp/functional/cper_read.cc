/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Regression tests for the CPER read path via
// amdsmi_get_gpu_cper_entries_by_path(); no GPU required.
// ROCM-25398: a zero-byte CPER node must not abort the process.
// ROCM-25954: an empty ring returns SUCCESS with zero entries, not an error.

#include <gtest/gtest.h>
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
TEST(amdsmitstReadOnly, CperReadZeroSizeFile) {
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
TEST(amdsmitstReadOnly, CperNonZeroFileRealReadHasNoRecords) {
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
TEST(amdsmitstReadOnly, CperReadMissingFile) {
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
TEST(amdsmitstReadOnly, CperParsesSingleRecord) {
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
TEST(amdsmitstReadOnly, CperEmptyRingAdvertisedCapacityShortRead) {
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
TEST(amdsmitstReadOnly, CperPartialReadNoRecords) {
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
TEST(amdsmitstReadOnly, CperReadErrorIsFileError) {
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
TEST(amdsmitstReadOnly, CperFirstRecordExceedsBufferOutOfResources) {
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
TEST(amdsmitstReadOnly, CperSecondRecordOverflowsBufferMoreData) {
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
TEST(amdsmitstReadOnly, CperSlotExhaustionMoreData) {
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
TEST(amdsmitstReadOnly, CperByPathRejectsInvalidArgs) {
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
