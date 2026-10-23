/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR kernel-argument signature check
 * @{
 * @ingroup HRRTest
 * Replay checks a kernel launch record's arguments against the signature of the
 * kernel in the code object before it launches (AIRUNTIME-3070).
 *
 * The handler tests stand in for the code object. This one captures a real
 * launch on a GPU, so the signature replay compares against comes from
 * hipKernelGetParamInfo on the code object the archive carries. It replays the
 * archive unmodified as the control, then replays copies whose launch record
 * is edited to disagree with that signature:
 *
 *   - too few arguments: the last explicit argument is dropped
 *   - too many arguments: one more is appended
 *   - a wrongly sized argument: an int recorded as eight bytes
 *   - a truncated payload: the record ends inside its last explicit argument
 *
 * Each must be refused with a line naming the sequence and the kernel, exit
 * non-zero, and launch nothing.
 */

#include "hrr_test_common.hh"

#include "hrr_reader.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#if defined(HRR_PLAYBACK_EXE) && defined(HRR_TEST_EXE)

namespace {
constexpr int kScale = 3;
constexpr int kBias  = 4;
constexpr int kExpected = kScale * 7 + kBias;

constexpr const char* kProbeName = "hrr_sigcheck_probe";
}  // namespace

// Three explicit arguments of three kinds: a device pointer and two ints.
__global__ void hrr_sigcheck_probe(int* out, int scale, int bias) {
  if (threadIdx.x == 0) out[0] = scale * 7 + bias;
}

// ===========================================================================
// The captured workload: one launch whose result is read back, so a launch
// that never happens is visible in the replayed D2H check as well.
// ===========================================================================
TEST_CASE("Unit_HRR_KernelSignature_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HRR_HIP_CHECK(hipMemset(out, 0, sizeof(int)));

  hipLaunchKernelGGL(hrr_sigcheck_probe, dim3(1), dim3(64), 0, nullptr, out,
                     kScale, kBias);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int host = 0;
  HRR_HIP_CHECK(hipMemcpy(&host, out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host == kExpected);

  HRR_HIP_CHECK(hipFree(out));
}

namespace {

std::vector<uint8_t> read_bytes(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

void write_bytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  REQUIRE(f.good());
}

// One recorded argument: where its 3-byte (kind, size) prefix starts in the
// record, and the whole of it, trailing pointer-offset list included.
struct ArgSpan {
  size_t   at;     // offset of the kind byte within the record
  size_t   len;    // bytes from `at` to the next argument
  uint8_t  kind;
  uint16_t size;
};

// The launch record as the wire format lays it out after the 32-byte header:
//   stream(8) name_len(2) name co_hash(16) grid(12) block(12) shared(4)
//   num_args(2) num_snapshots(2) args... snapshots... launch-attribute tail
struct LaunchRecord {
  std::vector<uint8_t> bytes;  // header included
  size_t num_args_at = 0;      // offset of the u16 argument count
  std::vector<ArgSpan> args;
  uint64_t sequence_id = 0;

  explicit LaunchRecord(std::vector<uint8_t> raw) : bytes(std::move(raw)) {
    const size_t hdr = sizeof(hrr_event_header);
    REQUIRE(bytes.size() > hdr + 10);
    std::memcpy(&sequence_id, bytes.data() + offsetof(hrr_event_header, sequence_id),
                sizeof(sequence_id));
    uint16_t name_len = 0;
    std::memcpy(&name_len, bytes.data() + hdr + 8, 2);
    num_args_at = hdr + 8 + 2 + name_len + 16 + 12 + 12 + 4;
    REQUIRE(bytes.size() >= num_args_at + 4);
    uint16_t num_args = 0;
    std::memcpy(&num_args, bytes.data() + num_args_at, 2);
    size_t p = num_args_at + 4;
    for (uint16_t i = 0; i < num_args; i++) {
      REQUIRE(bytes.size() >= p + 3);
      ArgSpan a{p, 0, bytes[p], 0};
      std::memcpy(&a.size, bytes.data() + p + 1, 2);
      p += 3 + a.size;
      if (a.kind == 3) {  // trailing list of embedded-pointer offsets
        REQUIRE(bytes.size() >= p + 2);
        uint16_t n = 0;
        std::memcpy(&n, bytes.data() + p, 2);
        p += 2 + 2u * n;
      }
      REQUIRE(bytes.size() >= p);
      a.len = p - a.at;
      args.push_back(a);
    }
  }

  // The last argument that is a plain value or a pointer. Hidden arguments
  // (kind 2) are not part of the signature replay compares against.
  size_t last_explicit() const {
    for (size_t i = args.size(); i-- > 0;)
      if (args[i].kind != 2 && args[i].kind != 3) return i;
    FAIL("the launch record has no explicit argument");
    return 0;
  }

  void set_num_args(int delta) {
    uint16_t n = 0;
    std::memcpy(&n, bytes.data() + num_args_at, 2);
    n = static_cast<uint16_t>(n + delta);
    std::memcpy(bytes.data() + num_args_at, &n, 2);
  }

  // payload_length counts the record, header included.
  void fix_length() {
    const uint32_t len = static_cast<uint32_t>(bytes.size());
    std::memcpy(bytes.data() + offsetof(hrr_event_header, payload_length), &len,
                sizeof(len));
  }
};

enum class Mutation { TooFew, TooMany, WrongSize, Truncated };

const char* mutation_name(Mutation m) {
  switch (m) {
    case Mutation::TooFew:    return "too few arguments";
    case Mutation::TooMany:   return "too many arguments";
    case Mutation::WrongSize: return "a wrongly sized argument";
    case Mutation::Truncated: return "a truncated payload";
  }
  return "?";
}

// Every mutation edits the record only; the code object it names is untouched,
// so its signature stays what the kernel was compiled with.
void mutate(LaunchRecord& rec, Mutation m) {
  const size_t last = rec.last_explicit();
  switch (m) {
    case Mutation::TooFew: {
      const ArgSpan& a = rec.args[last];
      rec.bytes.erase(rec.bytes.begin() + a.at, rec.bytes.begin() + a.at + a.len);
      rec.set_num_args(-1);
      break;
    }
    case Mutation::TooMany: {
      const ArgSpan& a = rec.args.back();
      const uint8_t extra[3 + 4] = {0, 4, 0, 0, 0, 0, 0};
      rec.bytes.insert(rec.bytes.begin() + a.at + a.len, extra, extra + sizeof(extra));
      rec.set_num_args(+1);
      break;
    }
    case Mutation::WrongSize: {
      // The int `scale` (the second explicit argument) recorded as 8 bytes.
      const ArgSpan& a = rec.args[last - 1];
      REQUIRE(a.kind == 0);
      REQUIRE(a.size == sizeof(int));
      const uint16_t wide = 8;
      std::memcpy(rec.bytes.data() + a.at + 1, &wide, 2);
      const uint8_t pad[4] = {};
      rec.bytes.insert(rec.bytes.begin() + a.at + a.len, pad, pad + sizeof(pad));
      break;
    }
    case Mutation::Truncated: {
      // End the record two bytes into the last explicit argument's value, after
      // the count already promised every argument.
      const ArgSpan& a = rec.args[last];
      REQUIRE(a.size == sizeof(int));
      rec.bytes.resize(a.at + 3 + 2);
      break;
    }
  }
  rec.fix_length();
}

// Copy the archive and splice `replacement` over the original launch record in
// the copy's events.bin. Returns the copy's root.
void splice_record(const fs::path& copy_root, const std::vector<uint8_t>& original,
                   const std::vector<uint8_t>& replacement) {
  const std::vector<fs::path> archives = hrr_process_archives(copy_root);
  REQUIRE(archives.size() == 1);
  const fs::path events = archives.front() / "events.bin";
  std::vector<uint8_t> file = read_bytes(events);
  const auto found = std::search(file.begin(), file.end(), original.begin(), original.end());
  REQUIRE(found != file.end());
  // The record has to be unique in the stream, or the edit could land elsewhere.
  REQUIRE(std::search(found + 1, file.end(), original.begin(), original.end()) == file.end());
  const auto at = found - file.begin();
  file.erase(file.begin() + at, file.begin() + at + static_cast<ptrdiff_t>(original.size()));
  file.insert(file.begin() + at, replacement.begin(), replacement.end());
  write_bytes(events, file);
}

// The one launch record of the kernel under test.
std::vector<uint8_t> find_launch_record(const fs::path& archive_dir) {
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive_dir.string(), arc));
  const std::vector<uint8_t>* found = nullptr;
  size_t n = 0;
  for (const auto& e : arc.events) {
    const auto& pl = e.raw_payload;
    const auto hit = std::search(pl.begin(), pl.end(), kProbeName,
                                 kProbeName + std::strlen(kProbeName));
    if (hit == pl.end()) continue;
    found = &pl;
    n++;
  }
  INFO("launch records naming " << kProbeName << ": " << n);
  REQUIRE(n == 1);
  return *found;
}

// The summary's launch counter. Replay stdout carries the summary, stderr the
// refusal; both are merged.
size_t kernels_launched(const std::string& out) {
  const size_t at = out.find("Kernels launched:");
  REQUIRE(at != std::string::npos);
  return static_cast<size_t>(std::strtoull(out.c_str() + at + 17, nullptr, 10));
}

}  // namespace

// ---------------------------------------------------------------------------
// One capture, replayed unmodified and then once per mutation.
//
//   - Control: the unmodified archive replays, launches its one kernel, and the
//     read-back check passes. Without it a refusal below could be a replay that
//     refuses everything.
//   - Each mutated copy: replay exits non-zero, prints the refusal naming the
//     record's sequence and the kernel, launches no kernel, and the read-back
//     of the buffer the kernel would have written does not pass.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_KernelSignature_MismatchedArchiveRefused", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_kernel_signature.hrr");
  hrr_capture_direct("Unit_HRR_KernelSignature_Direct", cap.path, 5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  const std::vector<uint8_t> original = find_launch_record(archive);
  const LaunchRecord parsed(original);
  // Pointer, scale, bias and whatever hidden arguments the compiler adds.
  REQUIRE(parsed.args.size() >= 3);

  SECTION("the unmodified archive replays") {
    auto [rc, out] = hrr_playback_merged(cap.path);
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("launch refused") == std::string::npos);
    CHECK(kernels_launched(out) == 1);
    int pass = 0, fail = 0;
    REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
    CHECK(pass >= 1);
    CHECK(fail == 0);
  }

  for (const Mutation m : {Mutation::TooFew, Mutation::TooMany, Mutation::WrongSize,
                           Mutation::Truncated}) {
    DYNAMIC_SECTION("a launch record with " << mutation_name(m) << " is refused") {
      ScopedDir copy(fs::temp_directory_path() / "hrr_kernel_signature_edit.hrr");
      fs::copy(cap.path, copy.path, fs::copy_options::recursive);
      LaunchRecord edited(original);
      mutate(edited, m);
      REQUIRE(edited.bytes != original);
      splice_record(copy.path, original, edited.bytes);

      auto [rc, out] = hrr_playback_merged(copy.path);
      INFO("Replay of the edited archive:\n" << out);
      CHECK(rc != 0);

      const std::string named = "seq=" + std::to_string(parsed.sequence_id) + " kernel '";
      const size_t at = out.find(named);
      REQUIRE(at != std::string::npos);
      const size_t eol = out.find('\n', at);
      const std::string line = out.substr(at, eol - at);
      CHECK(line.find(kProbeName) != std::string::npos);
      CHECK(line.find("launch refused") != std::string::npos);

      // The refusal is fatal, so replay stops before its summary. Carrying on
      // past it gets the counters: nothing ran, and the buffer the kernel would
      // have written reads back unwritten, which replay compares and reports.
      auto [rc2, out2] = hrr_playback_merged(copy.path, "--continue-on-error");
      INFO("Replay of the edited archive, continuing past errors:\n" << out2);
      CHECK(out2.find(named) != std::string::npos);
      CHECK(kernels_launched(out2) == 0);
      int pass = 0, fail = 0;
      REQUIRE(hrr_parse_d2h_summary(out2, pass, fail));
      CHECK(pass == 0);
      CHECK(fail >= 1);
    }
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
