/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR kernel-argument signature check
 * @{
 * @ingroup HRRTest
 * Replay checks a launch record's arguments against the kernel signature in the code object before launching (AIRUNTIME-3070).
 * This captures a real launch on a GPU, replays it unmodified as the control, then replays copies whose record is edited.
 * Only a record with too few arguments is refused: it must name the sequence and kernel, exit non-zero and launch nothing.
 * An extra argument or an int recorded as eight bytes replays like the control, since CLR reads nothing past the record.
 * Reading the signature must also leave no GPU error behind.
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

__global__ void hrr_sigcheck_probe(int* out, int scale, int bias) {
  if (threadIdx.x == 0) out[0] = scale * 7 + bias;
}

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

// The launch record's wire bytes, with where each argument sits in them. The
// reader has already parsed the record (kernel name, arguments), so the offsets
// follow from it; the layout after the 32-byte header is
//   stream(8) name_len(2) name co_hash(16) grid(12) block(12) shared(4)
//   num_args(2) num_snapshots(2) args... snapshots... launch-attribute tail
// and each argument is kind(1) size(2) data, plus a count and 2-byte offsets
// after a kind-3 argument.
struct LaunchRecord {
  struct Arg {
    size_t at;   // offset of the argument's kind byte
    size_t len;  // bytes to the next argument
    uint16_t size;
  };
  std::vector<uint8_t> bytes;  // header included
  size_t num_args_at = 0;      // offset of the u16 argument count
  std::vector<Arg> args;
  uint64_t sequence_id = 0;

  LaunchRecord(const hrr::Event& e) : bytes(e.raw_payload) {
    sequence_id = e.header().sequence_id;
    num_args_at = sizeof(hrr_event_header) + 8 + 2 + e.kernel_launch->kernel_name.size() +
                  16 + 12 + 12 + 4;
    size_t at = num_args_at + 4;
    for (const auto& a : e.kernel_launch->args) {
      const size_t len = 3 + a.size + (a.value_kind == 3 ? 2 + 2 * a.ptr_offsets.size() : 0);
      args.push_back({at, len, a.size});
      at += len;
    }
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

enum class Mutation { TooFew, TooMany, Wide };

const char* mutation_name(Mutation m) {
  switch (m) {
    case Mutation::TooFew:  return "too few arguments";
    case Mutation::TooMany: return "one more argument";
    case Mutation::Wide:    return "a wide argument";
  }
  return "?";
}

// Every mutation edits the record only; the code object it names is untouched,
// so its signature stays what the kernel was compiled with.
void mutate(LaunchRecord& rec, Mutation m) {
  // Capture never records hidden arguments: the pointer, scale and bias.
  REQUIRE(rec.args.size() == 3);
  const auto& last = rec.args[2];
  const auto& scale = rec.args[1];
  switch (m) {
    case Mutation::TooFew: {
      rec.bytes.erase(rec.bytes.begin() + last.at, rec.bytes.begin() + last.at + last.len);
      rec.set_num_args(-1);
      break;
    }
    case Mutation::TooMany: {
      const uint8_t extra[3 + 4] = {0, 4, 0, 0, 0, 0, 0};
      rec.bytes.insert(rec.bytes.begin() + last.at + last.len, extra, extra + sizeof(extra));
      rec.set_num_args(+1);
      break;
    }
    case Mutation::Wide: {
      // The int `scale` recorded as 8 bytes, the extra bytes zero.
      REQUIRE(scale.size == sizeof(int));
      const uint16_t wide = 8;
      std::memcpy(rec.bytes.data() + scale.at + 1, &wide, 2);
      const uint8_t pad[4] = {};
      rec.bytes.insert(rec.bytes.begin() + scale.at + scale.len, pad, pad + sizeof(pad));
      break;
    }
  }
  rec.fix_length();
}

void splice_record(const fs::path& copy_root, const std::vector<uint8_t>& original,
                   const std::vector<uint8_t>& replacement) {
  const std::vector<fs::path> archives = hrr_process_archives(copy_root);
  REQUIRE(archives.size() == 1);
  const fs::path events = archives.front() / "events.bin";
  std::vector<uint8_t> file = read_file(events);
  const auto found = std::search(file.begin(), file.end(), original.begin(), original.end());
  REQUIRE(found != file.end());
  // The record has to be unique in the stream, or the edit could land elsewhere.
  REQUIRE(std::search(found + 1, file.end(), original.begin(), original.end()) == file.end());
  const auto at = found - file.begin();
  file.erase(file.begin() + at, file.begin() + at + static_cast<ptrdiff_t>(original.size()));
  file.insert(file.begin() + at, replacement.begin(), replacement.end());
  write_file(events, file);
}

LaunchRecord find_launch_record(const hrr::Archive& arc) {
  const hrr::Event* found = nullptr;
  size_t n = 0;
  for (const auto& e : arc.events) {
    if (!e.kernel_launch || e.kernel_launch->kernel_name.find(kProbeName) == std::string::npos)
      continue;
    found = &e;
    n++;
  }
  INFO("launch records naming " << kProbeName << ": " << n);
  REQUIRE(n == 1);
  return LaunchRecord(*found);
}

// Replay stdout carries the summary and stderr the refusal; both are merged.
size_t kernels_launched(const std::string& out) {
  const size_t at = out.find("Kernels launched:");
  REQUIRE(at != std::string::npos);
  return static_cast<size_t>(std::strtoull(out.c_str() + at + 17, nullptr, 10));
}

}  // namespace

// The control must replay cleanly, or a refusal below could be a replay that refuses everything.
TEST_CASE("Unit_HRR_KernelSignature_MismatchedArchiveRefused", "[.][hrr]") {
  ScopedDir cap(fs::temp_directory_path() / "hrr_kernel_signature.hrr");
  hrr_capture_direct("Unit_HRR_KernelSignature_Direct", cap.path, 5);
  const fs::path archive = hrr_single_process_archive(cap.path);

  hrr::Archive loaded;
  REQUIRE(hrr::load_archive(archive.string(), loaded));
  const LaunchRecord parsed = find_launch_record(loaded);
  const std::vector<uint8_t>& original = parsed.bytes;
  REQUIRE(parsed.args.size() == 3);

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

  // The signature read ends on a call that answers hipErrorInvalidValue, one
  // index past the last parameter. That must not stay HIP's last error: the
  // warm-up pass --kernel-filter adds would report it as a pending GPU error
  // (and --sync-after-event would abort on it).
  SECTION("the unmodified archive leaves no pending GPU error behind") {
    auto [rc, out] = hrr_playback_merged(cap.path, std::string("--kernel-filter ") + kProbeName);
    INFO("Replay:\n" << out);
    CHECK(rc == 0);
    CHECK(out.find("pending GPU error") == std::string::npos);
    CHECK(kernels_launched(out) == 1);
  }

  SECTION("a launch record with too few arguments is refused") {
    ScopedDir copy(fs::temp_directory_path() / "hrr_kernel_signature_edit.hrr");
    fs::copy(cap.path, copy.path, fs::copy_options::recursive);
    LaunchRecord edited = parsed;
    mutate(edited, Mutation::TooFew);
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

  // CLR reads the kernel's own parameters, each at its own size: an extra
  // argument, or an argument wider than its parameter, is never read past.
  for (const Mutation m : {Mutation::TooMany, Mutation::Wide}) {
    DYNAMIC_SECTION("a launch record with " << mutation_name(m) << " replays") {
      ScopedDir copy(fs::temp_directory_path() / "hrr_kernel_signature_edit.hrr");
      fs::copy(cap.path, copy.path, fs::copy_options::recursive);
      LaunchRecord edited = parsed;
      mutate(edited, m);
      REQUIRE(edited.bytes != original);
      splice_record(copy.path, original, edited.bytes);

      auto [rc, out] = hrr_playback_merged(copy.path);
      INFO("Replay of the edited archive:\n" << out);
      CHECK(rc == 0);
      CHECK(out.find("launch refused") == std::string::npos);
      CHECK(kernels_launched(out) == 1);
      int pass = 0, fail = 0;
      REQUIRE(hrr_parse_d2h_summary(out, pass, fail));
      CHECK(pass >= 1);
      CHECK(fail == 0);
    }
  }
}

#endif  // HRR_PLAYBACK_EXE && HRR_TEST_EXE

/**
 * @}
 */
