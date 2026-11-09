/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Capture/Replay Roundtrip
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include <hip/hiprtc.h>
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


static size_t find_string_end(const std::string& json, size_t quote_pos) {
  bool escape = false;
  for (size_t i = quote_pos + 1; i < json.size(); ++i) {
    const char c = json[i];
    if (escape) {
      escape = false;
    } else if (c == '\\') {
      escape = true;
    } else if (c == '"') {
      return i;
    }
  }
  return std::string::npos;
}

static size_t find_key_value(const std::string& json, const std::string& key) {
  size_t pos = 0;
  while ((pos = json.find('"', pos)) != std::string::npos) {
    const size_t end = find_string_end(json, pos);
    if (end == std::string::npos) return std::string::npos;
    if (json.compare(pos + 1, end - pos - 1, key) == 0) {
      size_t colon = end + 1;
      while (colon < json.size() && std::isspace(static_cast<unsigned char>(json[colon]))) ++colon;
      if (colon < json.size() && json[colon] == ':') {
        size_t value = colon + 1;
        while (value < json.size() && std::isspace(static_cast<unsigned char>(json[value]))) ++value;
        return value;
      }
    }
    pos = end + 1;
  }
  return std::string::npos;
}

static std::string json_string_value(const std::string& json, const std::string& key) {
  size_t pos = find_key_value(json, key);
  if (pos == std::string::npos || pos >= json.size() || json[pos] != '"') return {};
  size_t end = find_string_end(json, pos);
  if (end == std::string::npos) return {};
  return json.substr(pos + 1, end - pos - 1);
}

static long long json_integer_value(const std::string& json, const std::string& key,
                                    long long missing = -1) {
  size_t pos = find_key_value(json, key);
  if (pos == std::string::npos) return missing;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  char* end = nullptr;
  long long value = std::strtoll(json.c_str() + pos, &end, 10);
  return (end == json.c_str() + pos) ? missing : value;
}

static std::string json_object_value(const std::string& json, const std::string& key) {
  size_t pos = find_key_value(json, key);
  if (pos == std::string::npos || pos >= json.size() || json[pos] != '{') return {};

  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = pos; i < json.size(); ++i) {
    const char c = json[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0) return json.substr(pos, i - pos + 1);
    }
  }
  return {};
}

static bool json_array_exists(const std::string& json, const std::string& key) {
  size_t pos = find_key_value(json, key);
  return pos != std::string::npos && pos < json.size() && json[pos] == '[';
}

HRR_TEST_CASE(Unit_HRR_CaptureReplayRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_gpu"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    INFO("Capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs"))
    ++blob_count;
  INFO("Blob count: " << blob_count);
  REQUIRE(blob_count >= 1);

  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    INFO("Event count: " << arc.events.size());
    REQUIRE(arc.events.size() >= 10);  // malloc + H2D + kernel×n + D2H + free minimum
  }

  hrr_run_playback(cap.path);
}

HRR_TEST_CASE(Unit_HRR_CaptureStartNotice) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_capture_notice"};
  const std::string kNotice =
      "[HRR capture] Recording this process's HIP calls, with their host buffers, kernel "
      "arguments and code objects, to ";
  auto count_notices = [&](const std::string& out) {
    size_t n = 0;
    for (size_t at = out.find(kNotice); at != std::string::npos;
         at = out.find(kNotice, at + kNotice.size()))
      ++n;
    return n;
  };

  {
    hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    proc.setEnv("AMD_LOG_LEVEL", "0");
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    const std::string out = proc.getOutput();
    INFO("Capture subprocess output:\n" << out);
    REQUIRE(ret == 0);
    REQUIRE(count_notices(out) == 1);

    const size_t begin = out.find(kNotice);
    std::string line = out.substr(begin, out.find('\n', begin) - begin);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::string base = cap.path.string();
    const std::string pid_dir = hrr_single_process_archive(cap.path).filename().string();
    // The writer joins the pid-<pid> component with '/' on every platform.
    const std::string expected = kNotice + base + "/" + pid_dir +
                                 " (child processes record to their own pid-* directories in " +
                                 base + ")";
    CHECK(line == expected);
  }

  {
    // The notice goes to stderr, so stdout holds none.
    ScopedDir cap_stdout{fs::temp_directory_path() / "hrr_capture_notice_stdout"};
    hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/false);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_stdout.path.string());
    proc.setEnv("AMD_LOG_LEVEL", "0");
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    const std::string out = proc.getOutput();
    INFO("Capture subprocess stdout:\n" << out);
    REQUIRE(ret == 0);
    CHECK(count_notices(out) == 0);
  }

  {
    hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
    proc.setEnv("AMD_LOG_LEVEL", "0");
    // CLR turns an empty HIP_HRR_CAPTURE_OUTPUT into a single space, which still enables capture.
    // Unset it so an inherited value cannot arm capture.
    proc.unsetEnv("HIP_HRR_CAPTURE_OUTPUT");
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    const std::string out = proc.getOutput();
    INFO("Subprocess output without capture:\n" << out);
    REQUIRE(ret == 0);
    CHECK(count_notices(out) == 0);
  }
}

HRR_TEST_CASE(Unit_HRR_CaptureExitSummary) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_capture_exit_summary"};
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
  // The summary logs at LOG_INFO with LOG_ALWAYS, which no mask filters.
  // AMD_LOG_MASK=0 keeps per-API traces out of the output.
  proc.setEnv("AMD_LOG_LEVEL", "3");
  proc.setEnv("AMD_LOG_MASK", "0");
  set_proc_search_path(proc);
  int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
  const std::string out = proc.getOutput();
  INFO("Capture subprocess output:\n" << out);
  REQUIRE(ret == 0);

  const std::string kSummary = "[HRR capture] Wrote ";
  const size_t begin = out.find(kSummary);
  REQUIRE(begin != std::string::npos);
  CHECK(out.find(kSummary, begin + kSummary.size()) == std::string::npos);
  std::string line = out.substr(begin, out.find('\n', begin) - begin);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  INFO("Summary line: " << line);
  const std::string pid_dir = hrr_single_process_archive(cap.path).filename().string();
  const std::string tail = " blobs to: " + cap.path.string() + "/" + pid_dir;
  REQUIRE(line.size() >= tail.size());
  CHECK(line.compare(line.size() - tail.size(), tail.size(), tail) == 0);
}

static void hrr_require_recorded_apis(const fs::path& cap_path,
                                      const std::vector<hrr_api_id_t>& apis);

HRR_TEST_CASE(Unit_HRR_AllApisRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_allapis"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_AllApis_Direct\"");
    INFO("AllApis capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs"))
    ++blob_count;
  INFO("Blob count: " << blob_count);
  REQUIRE(blob_count >= 1);

  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    INFO("Event count: " << arc.events.size());
    REQUIRE(arc.events.size() >= 40);  // ~55 distinct APIs exercised
  }

  // gfx950 (MI350X) reports no image support, so check the array copies only where images exist.
  // Without the check, a capture missing the copies would still replay.
  int images = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&images, hipDeviceAttributeImageSupport, 0));
  if (images)
    hrr_require_recorded_apis(cap.path, {HRR_API_HIPARRAY3DCREATE, HRR_API_HIPARRAYCREATE,
                                         HRR_API_HIPMEMCPY3D, HRR_API_HIPMEMCPY3DASYNC});

  hrr_run_playback(cap.path);
}

HRR_TEST_CASE(Unit_HRR_HostMemRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostmem"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_HostMemWorkload_Direct\"");
    INFO("HostMem capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs"))
    ++blob_count;
  INFO("Blob count: " << blob_count);
  REQUIRE(blob_count >= 1);

  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    INFO("Event count: " << arc.events.size());
    REQUIRE(arc.events.size() >= 5);  // malloc + H2D(init) + kernel + D2H + free minimum
  }

  hrr_run_playback(cap.path, "", /*require_d2h=*/true);
}

HRR_TEST_CASE(Unit_HRR_GraphRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_graph"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_GraphWorkload_Direct\"");
    INFO("Graph capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs"))
    ++blob_count;
  INFO("Blob count: " << blob_count);
  REQUIRE(blob_count >= 1);

  hrr_run_playback(cap.path);
}

HRR_TEST_CASE(Unit_HRR_StressApisRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_stress"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_StressApis_Direct\"");
    INFO("Stress capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs"))
    ++blob_count;
  INFO("Blob count: " << blob_count);
  REQUIRE(blob_count >= 1);

  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    INFO("Event count: " << arc.events.size());
    REQUIRE(arc.events.size() >= 200);  // 500+ API calls: alloc/free loops, memset/memcpy, kernels
  }

  hrr_run_playback(cap.path);
}

static bool hrr_find_peer_accessible_pair(int& src_dev, int& dst_dev, int& ndev) {
  HRR_HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) return false;

  for (int src = 0; src < ndev; ++src) {
    for (int dst = 0; dst < ndev; ++dst) {
      if (src == dst) continue;
      int can_access = 0;
      HRR_HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src, dst));
      if (can_access) {
        src_dev = src;
        dst_dev = dst;
        return true;
      }
    }
  }
  return false;
}

// Workloads call plain API names and rely on -fgpu-default-stream=per-thread to redirect them to _spt ids.
// A lost redirect still replays cleanly, so this asserts the archive holds an event for every id.
static void hrr_require_recorded_apis(const fs::path& cap_path,
                                      const std::vector<hrr_api_id_t>& apis) {
  if (apis.empty()) return;
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap_path.string(), arc));
  for (hrr_api_id_t api : apis) {
    bool found = std::any_of(arc.events.begin(), arc.events.end(),
                             [api](const hrr::Event& e) {
                               return e.header().event_type ==
                                      static_cast<uint16_t>(api);
                             });
    INFO("API not recorded: " << hrr::event_type_name(static_cast<uint16_t>(api)));
    REQUIRE(found);
  }
}


/**
 * Hidden: capture shims install inside hip::init(), so the first HIP call is not recorded.
 * A first-call hipMalloc is then missing and replay aborts with "H2D dst ... not mapped".
 */
TEST_CASE("Unit_HRR_FirstMallocRoundtrip", "[.][hrr-repro]") {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_firstmalloc"};
  hrr_run_roundtrip("Unit_HRR_FirstMalloc_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_EmbeddedPtrRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_embeddedptr"};
  hrr_run_roundtrip("Unit_HRR_EmbeddedPtrStruct_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_ZeroInitRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_zeroinit"};
  hrr_capture_direct("Unit_HRR_ZeroInitRead_Direct", cap.path);

  auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_REPLAY_ZERO_INIT", "1"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  REQUIRE(ret == 0);  // zero-init reproduces the captured all-zero output

  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  INFO("D2H pass=" << d2h_pass << " fail=" << d2h_fail);
  CHECK(d2h_pass >= 1);
  CHECK(d2h_fail == 0);
}

HRR_TEST_CASE(Unit_HRR_D2HMismatchExit) {
  const std::string mismatch =
      "[HRR]   D2H checks     : 3 pass (3 exact, 0 within tol), 1 fail, 0 skipped\n";
  const std::string clean =
      "[HRR]   D2H checks     : 4 pass (4 exact, 0 within tol), 0 fail, 0 skipped\n";

  CHECK(hrr_d2h_mismatch_exit(1, mismatch));
  CHECK(hrr_d2h_mismatch_exit(2, ""));  // divergence guard, no summary

  CHECK_FALSE(hrr_d2h_mismatch_exit(1, ""));     // fatal HIP error
  CHECK_FALSE(hrr_d2h_mismatch_exit(1, clean));  // failed for another reason
  CHECK_FALSE(hrr_d2h_mismatch_exit(0, clean));
  CHECK_FALSE(hrr_d2h_mismatch_exit(134, mismatch));  // crash
}

// Playback reads blobs by name without re-hashing them, so a test can edit one
// and replay compares against the edit.
static std::vector<fs::path> hrr_memcpy_d2h_blobs(const fs::path& cap) {
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.string(), arc));
  std::vector<fs::path> blobs;
  for (const auto& e : arc.events) {
    if (e.header().event_type != static_cast<uint16_t>(HRR_API_HIPMEMCPY)) continue;
    REQUIRE(e.raw_payload.size() >= sizeof(hrr_args_hipMemcpy));
    const auto* a = reinterpret_cast<const hrr_args_hipMemcpy*>(e.raw_payload.data());
    if (a->kind != hipMemcpyDeviceToHost || (!a->blob_hash_lo && !a->blob_hash_hi)) continue;
    const auto it = arc.blobs.find(hrr::hash_hex(a->blob_hash_lo, a->blob_hash_hi));
    REQUIRE(it != arc.blobs.end());
    blobs.push_back(it->second);
  }
  return blobs;
}

struct HrrD2HCounts { int pass = 0, exact = 0, tol = 0, fail = 0, skipped = 0; };
static bool hrr_parse_d2h_counts(const std::string& out, HrrD2HCounts& c) {
  const size_t pos = out.find("D2H checks");
  if (pos == std::string::npos) return false;
  const size_t colon = out.find(':', pos);
  if (colon == std::string::npos) return false;
  return std::sscanf(out.c_str() + colon + 1,
                     " %d pass (%d exact, %d within tol), %d fail, %d skipped",
                     &c.pass, &c.exact, &c.tol, &c.fail, &c.skipped) == 5;
}

HRR_TEST_CASE(Unit_HRR_D2HExactByDefaultRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_d2h_exact"};
  hrr_capture_direct("Unit_HRR_ZeroInitRead_Direct", cap.path);
  const auto blobs = hrr_memcpy_d2h_blobs(cap.path);
  REQUIRE(blobs.size() == 1);
  const auto words = static_cast<size_t>(fs::file_size(blobs[0]) / sizeof(int32_t));
  REQUIRE(words > 0);
  {
    std::vector<int32_t> sentinels(words);
    for (size_t i = 0; i < words; ++i) sentinels[i] = static_cast<int32_t>(i + 1);
    std::ofstream f(blobs[0], std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(sentinels.data()),
            static_cast<std::streamsize>(words * sizeof(int32_t)));
    REQUIRE(f.good());
  }

  SECTION("byte-exact by default") {
    auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_REPLAY_ZERO_INIT", "1"},
                                                  {"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0"}});
    INFO("Playback stdout:\n" << out);
    HrrD2HCounts c;
    REQUIRE(hrr_parse_d2h_counts(out, c));
    CHECK(c.pass == 0);
    CHECK(c.fail == 1);
    REQUIRE(ret == 1);
  }

  SECTION("tolerance as an f32 opt-in") {
    auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_REPLAY_ZERO_INIT", "1"},
                                                  {"HIP_HRR_D2H_TOL", "f32:1e-3:1e-3"}});
    INFO("Playback stdout:\n" << out);
    HrrD2HCounts c;
    REQUIRE(hrr_parse_d2h_counts(out, c));
    CHECK(c.pass == 1);
    CHECK(c.exact == 0);
    CHECK(c.tol == 1);
    REQUIRE(ret == 0);
  }

  SECTION("an opt-in without a float type is refused") {
    auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_REPLAY_ZERO_INIT", "1"},
                                                  {"HIP_HRR_D2H_TOL", "i32:1e-3:1e-3"}});
    INFO("Playback stdout:\n" << out);
    CHECK(out.find("D2H checks") == std::string::npos);
    REQUIRE(ret == 1);
  }
}

HRR_TEST_CASE(Unit_HRR_D2HShortBlobRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_d2h_short"};
  hrr_capture_direct("Unit_HRR_ZeroInitRead_Direct", cap.path);
  const auto blobs = hrr_memcpy_d2h_blobs(cap.path);
  REQUIRE(blobs.size() == 1);
  const auto size = fs::file_size(blobs[0]);
  REQUIRE(size >= 2);
  fs::resize_file(blobs[0], size / 2);

  auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_REPLAY_ZERO_INIT", "1"}});
  INFO("Playback stdout:\n" << out);
  HrrD2HCounts c;
  REQUIRE(hrr_parse_d2h_counts(out, c));
  CHECK(c.pass == 0);
  CHECK(c.fail == 0);
  CHECK(c.skipped == 1);
  REQUIRE(ret == 1);
}

HRR_TEST_CASE(Unit_HRR_DivergenceAbortRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_divergence"};
  // Pinned host snapshots would record the flag, so disable them.
  hrr_capture_direct("Unit_HRR_UncapturedHostWrite_Direct", cap.path, 5,
                     {{"HIP_HRR_HOST_SNAPSHOTS", "0"}});

  SECTION("guard ON -> clean exit 2") {
    auto [ret, out] = hrr_playback_env(
        cap.path,
        {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0.25"},
         {"HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES", "16"}});
    INFO("Playback stdout:\n" << out);
    INFO("Playback exit code: " << ret);
    // The "replay DIVERGED" text is on stderr, which is not captured, so assert the exit code.
    REQUIRE(ret < 128);
    REQUIRE(ret == 2);
  }

  SECTION("guard OFF -> not exit 2") {
    auto [ret, out] = hrr_playback_env(
        cap.path,
        {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0"}});
    INFO("Playback stdout:\n" << out);
    INFO("Playback exit code: " << ret);
    REQUIRE(ret != 2);
    REQUIRE(ret < 128);  // and never a crash
  }
}

/**
 * Hidden: a mistuned guard could let the raw GPU fault through, which can destabilise a shared runner.
 * The guard must trip before the kernel writes null + 0x20000.
 */
TEST_CASE("Unit_HRR_NullOptionalPtrRoundtrip", "[.][hrr-repro]") {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_nulloptional"};
  hrr_capture_direct("Unit_HRR_NullOptionalPtr_Direct", cap.path, 5,
                     {{"HIP_HRR_HOST_SNAPSHOTS", "0"}});

  auto [ret, out] = hrr_playback_env(
      cap.path,
      {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0.25"},
       {"HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES", "16"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  // Exit 2 is the clean divergence stop; a GPU fault exits above 128 and a HIP error exits 1.
  REQUIRE(ret == 2);
}

/**
 * Replay uses byte-exact D2H: a float tolerance would accept a lost 32-bit write as an f64 within tolerance.
 */
HRR_TEST_CASE(Unit_HRR_StreamWriteValueRoundtrip) {
  int canUseStreamValue = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&canUseStreamValue,
                                  hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (!canUseStreamValue) {
    HRR_SKIP("stream wait value unsupported");
  }

  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamwritevalue"};
  hrr_capture_direct("Unit_HRR_StreamWriteValue_Direct", cap.path);

  auto [ret, out] = hrr_playback_env(cap.path, {});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
#ifdef _WIN32
  // Windows consumer-iGPU CI does not guarantee bit-exact device output, so D2H fidelity is best-effort.
  // A crash still fails via ret < 128.
  REQUIRE(ret < 128);
  if (ret != 0) return;
#else
  REQUIRE(ret == 0);  // exact-mode D2H: any differing byte fails the replay
#endif

  // A replay that validated nothing must not pass on the exit code alone.
  size_t pos = out.find("D2H checks");
  REQUIRE(pos != std::string::npos);
  size_t colon = out.find(':', pos);
  REQUIRE(colon != std::string::npos);
  int d2h_pass = 0;
  sscanf(out.c_str() + colon + 1, " %d pass", &d2h_pass);
  INFO("D2H pass=" << d2h_pass);
  CHECK(d2h_pass >= 3);
}

// The workloads are bit-deterministic, so no HIP_HRR_D2H_TOL: a 1e-3 tolerance would accept an all-zero buffer.
static void hrr_run_exact_roundtrip(const std::string& direct_case,
                                    const fs::path& cap_path,
                                    const std::vector<hrr_api_id_t>& expect_apis = {}) {
  hrr_capture_direct(direct_case, cap_path);
  hrr_require_recorded_apis(cap_path, expect_apis);

  auto [ret, out] = hrr_playback_env(cap_path, {});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);

  // A handler error aborts the replay before the summary prints, so the summary line
  // proves the replay ran to completion on every platform.
  size_t pos = out.find("D2H checks");
  REQUIRE(pos != std::string::npos);
  size_t colon = out.find(':', pos);
  REQUIRE(colon != std::string::npos);
  // Summary format: "N pass (E exact, T within tol), F fail, S skipped".
  int d2h_pass = 0;
  REQUIRE(sscanf(out.c_str() + colon + 1, " %d pass", &d2h_pass) == 1);
  INFO("D2H pass=" << d2h_pass);

#ifdef _WIN32
// Windows D2H is best-effort, as in hrr_run_playback(); a crash still fails.
  REQUIRE(ret < 128);
#else
  // hrr-playback exits non-zero on a D2H failure or when every D2H event was skipped.
  // The pass count rules out an archive with no D2H blob.
  REQUIRE(ret == 0);
  REQUIRE(d2h_pass >= 1);
#endif
}

HRR_TEST_CASE(Unit_HRR_MemsetD2DRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetd2d"};
  hrr_run_exact_roundtrip("Unit_HRR_MemsetD2D_Direct", cap.path);
}

/**
 * hipMemAllocPitch is a playback no-op, so the memset destinations have no alloc_map entry at replay.
 * The default unmapped-pointer policy skips those calls; D2H passes only if replay survived.
 */
HRR_TEST_CASE(Unit_HRR_MemsetD2DPitchAllocRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetd2dpitchalloc"};
  hrr_run_exact_roundtrip("Unit_HRR_MemsetD2DPitchAlloc_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_UnmappedPointerPolicy) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_unmapped_policy"};
  hrr_capture_direct("Unit_HRR_MemsetD2DPitchAlloc_Direct", cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  size_t idx = 0;
  uint64_t dst = 0;
  for (size_t i = 0; i < arc.events.size() && dst == 0; ++i) {
    const auto& ev = arc.events[i];
    if (ev.header().event_type != HRR_API_HIPMEMSETD2D8) continue;
    REQUIRE(ev.raw_payload.size() >= sizeof(hrr_args_hipMemsetD2D8));
    dst = reinterpret_cast<const hrr_args_hipMemsetD2D8*>(ev.raw_payload.data())->dst;
    idx = i;
  }
  REQUIRE(dst != 0);
  char diag[128];
  snprintf(diag, sizeof(diag), "Event %zu (hipMemsetD2D8): recorded pointer 0x%llx ",
           idx, static_cast<unsigned long long>(dst));

  SECTION("by default the events are reported, counted and skipped") {
    auto [ret, out] = hrr_playback_merged(cap.path, "", {{"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback output:\n" << out);
    CHECK(out.find(diag) != std::string::npos);
    CHECK(out.find("Unmapped       : 6 event(s) not replayed as recorded") !=
          std::string::npos);
    CHECK(out.find("D2H checks") != std::string::npos);
#ifndef _WIN32
    CHECK(ret == 0);
#endif
  }

  SECTION("--strict-unmapped aborts at the first one") {
    auto [ret, out] = hrr_playback_merged(cap.path, "--strict-unmapped",
                                          {{"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback output:\n" << out);
    CHECK(ret != 0);
    CHECK(out.find(diag) != std::string::npos);
    CHECK(out.find("aborting replay (--strict-unmapped)") != std::string::npos);
    CHECK(out.find("D2H checks") == std::string::npos);
  }

  SECTION("--strict-unmapped names the event in the silent warm-up pass too") {
    auto [ret, out] = hrr_playback_merged(
        cap.path, "--kernel-filter hrr_no_such_kernel --strict-unmapped",
        {{"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback output:\n" << out);
    CHECK(ret != 0);
    CHECK(out.find("Fatal error in warm-up pass") != std::string::npos);
    CHECK(out.find(diag) != std::string::npos);
  }
}

/**
 * The workload calls plain hipMemset* names under -fgpu-default-stream=per-thread, so assert the _spt ids first.
 * A float tolerance would accept a zero-initialised buffer and hide a NOOP _spt handler, so replay byte-exact.
 */
HRR_TEST_CASE(Unit_HRR_MemsetSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetspt"};
  hrr_capture_direct("Unit_HRR_MemsetSpt_Direct", cap.path);

  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    auto recorded = [&arc](hrr_api_id_t api) {
      return std::any_of(arc.events.begin(), arc.events.end(), [api](const hrr::Event& e) {
        return e.header().event_type == static_cast<uint16_t>(api);
      });
    };
    REQUIRE(recorded(HRR_API_HIPMEMSET_SPT));
    REQUIRE(recorded(HRR_API_HIPMEMSETASYNC_SPT));
    REQUIRE(recorded(HRR_API_HIPMEMSET2D_SPT));
    REQUIRE(recorded(HRR_API_HIPMEMSET2DASYNC_SPT));
    // The readbacks must stay on plain hipMemcpy: hipMemcpy_spt records no data blob,
    // so a redirected readback would drop the D2H oracle.
    REQUIRE(recorded(HRR_API_HIPMEMCPY));
  }

  auto [ret, out] = hrr_playback_env(cap.path, {});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
#ifdef _WIN32
  REQUIRE(ret < 128);
#else
  REQUIRE(ret == 0);  // any byte mismatch exits 1

  size_t pos = out.find("D2H checks");
  REQUIRE(pos != std::string::npos);
  size_t colon = out.find(':', pos);
  REQUIRE(colon != std::string::npos);
  int d2h_pass = 0;
  sscanf(out.c_str() + colon + 1, " %d pass", &d2h_pass);
  INFO("D2H pass=" << d2h_pass);
  CHECK(d2h_pass >= 4);  // one validated buffer per _spt memset API
#endif
}

/**
 * All six _spt copies have NOOP playback handlers, so their recorded ids are all this proves about them.
 * A clean exact-mode replay shows the redirect did not swallow the oracle.
 */
HRR_TEST_CASE(Unit_HRR_MemcpySptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memcpyspt"};
  hrr_run_exact_roundtrip("Unit_HRR_MemcpySpt_Direct", cap.path,
                          {HRR_API_HIPMEMCPY_SPT,
                           HRR_API_HIPMEMCPYASYNC_SPT,
                           HRR_API_HIPMEMCPY2D_SPT,
                           HRR_API_HIPMEMCPY2DASYNC_SPT,
                           HRR_API_HIPMEMCPYTOSYMBOL_SPT,
                           HRR_API_HIPMEMCPYTOSYMBOLASYNC_SPT,
                           HRR_API_HIPMEMCPYHTOD,
                           HRR_API_HIPMEMCPYDTOH});
}

HRR_TEST_CASE(Unit_HRR_Memset3DSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memset3dspt"};
  // hipMemset3D_spt and hipMemset3DAsync_spt are NOOP on replay, so this covers the capture path only.
  // The oracle is written by hipMemsetD32 and read by the plain hipMemcpy.
  hrr_run_exact_roundtrip("Unit_HRR_Memset3DSpt_Direct", cap.path,
                          {HRR_API_HIPMEMSET3D_SPT,
                           HRR_API_HIPMEMSET3DASYNC_SPT,
                           HRR_API_HIPMEMCPY});
}

HRR_TEST_CASE(Unit_HRR_Memcpy3DSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memcpy3dspt"};
  // These two have faithful handlers, so each destination is its own oracle:
  // a handler that stopped copying leaves the replayed pre-fill.
  hrr_run_exact_roundtrip("Unit_HRR_Memcpy3DSpt_Direct", cap.path,
                          {HRR_API_HIPMEMCPY3D_SPT,
                           HRR_API_HIPMEMCPY3DASYNC_SPT,
                           HRR_API_HIPMEMCPY});
}

HRR_TEST_CASE(Unit_HRR_StreamQuerySptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamqueryspt"};
  // The per-thread queries write no device memory, so only the recorded ids and a completed replay are checked.
  // The validated buffer gives the replay an oracle.
  hrr_run_exact_roundtrip("Unit_HRR_StreamQuerySpt_Direct", cap.path,
                          {HRR_API_HIPSTREAMISCAPTURING_SPT,
                           HRR_API_HIPSTREAMQUERY_SPT,
                           HRR_API_HIPSTREAMSYNCHRONIZE_SPT,
                           HRR_API_HIPSTREAMGETPRIORITY_SPT,
                           HRR_API_HIPSTREAMGETFLAGS_SPT,
                           HRR_API_HIPEVENTRECORD_SPT,
                           HRR_API_HIPMEMCPY});
}

HRR_TEST_CASE(Unit_HRR_MemsetVariantsRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetvariants"};
  hrr_run_roundtrip("Unit_HRR_MemsetVariants_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_DeviceInfoRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_deviceinfo"};
  hrr_run_roundtrip("Unit_HRR_DeviceInfo_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_MetadataManifest) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_metadata_manifest"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_DeviceInfo_Direct\"");
    INFO("Metadata capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "manifest.json"));

  const std::string proc_manifest = read_text_file(archive_path / "manifest.json");
  const std::string metadata = json_object_value(proc_manifest, "metadata");

  INFO("Process manifest:\n" << proc_manifest);
  INFO("Metadata:\n" << metadata);

  REQUIRE_FALSE(metadata.empty());

  CHECK(json_integer_value(metadata, "schema_version") == 1);
  const std::string runtime_version = json_string_value(metadata, "hip_runtime_version");
  const std::string comgr_version = json_string_value(metadata, "comgr_version");
  INFO("hip_runtime_version=" << runtime_version);
  INFO("comgr_version=" << comgr_version);
  CHECK_FALSE(runtime_version.empty());
  CHECK_FALSE(comgr_version.empty());
  CHECK(std::count(runtime_version.begin(), runtime_version.end(), '.') == 2);
  CHECK(std::count(comgr_version.begin(), comgr_version.end(), '.') == 1);

  const long long device_count = json_integer_value(metadata, "device_count");
  const long long captured_device_count = json_integer_value(metadata, "captured_device_count");
  CHECK(device_count >= 1);
  CHECK(captured_device_count >= 1);
  CHECK(captured_device_count <= device_count);
  CHECK(json_array_exists(metadata, "devices"));

  const long long ordinal = json_integer_value(metadata, "ordinal");
  const long long total_global_mem = json_integer_value(metadata, "total_global_mem");
  const long long multi_processor_count = json_integer_value(metadata, "multi_processor_count");
  const long long compute_mode = json_integer_value(metadata, "compute_mode");
  const std::string name = json_string_value(metadata, "name");
  const std::string gcn_arch_name = json_string_value(metadata, "gcn_arch_name");
  const std::string compute_capability = json_string_value(metadata, "compute_capability");
  const std::string pci = json_string_value(metadata, "pci");
  const std::string uuid = json_string_value(metadata, "uuid");
  const std::string luid = json_string_value(metadata, "luid");

  CHECK(ordinal >= 0);
  CHECK_FALSE(json_object_value(metadata, "properties").empty());
  CHECK_FALSE(name.empty());
  CHECK_FALSE(gcn_arch_name.empty());
  CHECK(total_global_mem > 0);
  CHECK(multi_processor_count > 0);
  CHECK(compute_mode >= 0);
  CHECK_FALSE(compute_capability.empty());
  CHECK(std::count(compute_capability.begin(), compute_capability.end(), '.') == 1);
  // The keys stay, blank: they would identify the individual GPU and replay
  // needs none of them.
  for (const char* key : {"\"pci\": \"\"", "\"uuid\": \"\"", "\"luid\": \"\""}) {
    INFO("Blank field: " << key);
    CHECK(metadata.find(key) != std::string::npos);
  }
  CHECK(pci.empty());
  CHECK(uuid.empty());
  CHECK(luid.empty());

  // The workload hands hipChooseDevice the properties of device 0: the same
  // identifiers are blanked in the copy events.bin keeps, and the rest stays.
  uint16_t version = 0;
  FILE* f = hrr::open_record_stream((archive_path / "events.bin").string(), HRR_MAGIC,
                                    HRR_VERSION, &version);
  REQUIRE(f != nullptr);
  std::vector<uint8_t> raw;
  int choose_records = 0;
  while (hrr::read_raw_record(f, raw) == hrr::RecordStatus::Ok) {
    const auto* hdr = reinterpret_cast<const hrr_event_header*>(raw.data());
    if (hdr->event_type != HRR_API_HIPCHOOSEDEVICE) continue;
    REQUIRE(raw.size() >= sizeof(hrr_args_hipChooseDevice));
    const auto* a = reinterpret_cast<const hrr_args_hipChooseDevice*>(raw.data());
    REQUIRE(a->prop_present == 1);
    hipDeviceProp_t prop;
    std::memcpy(&prop, a->prop_bytes, sizeof(prop));
    ++choose_records;
    CHECK(std::string(prop.name).size() > 0);
    CHECK(prop.multiProcessorCount > 0);
    CHECK(std::all_of(std::begin(prop.uuid.bytes), std::end(prop.uuid.bytes),
                      [](char c) { return c == 0; }));
    CHECK(std::all_of(std::begin(prop.luid), std::end(prop.luid), [](char c) { return c == 0; }));
    CHECK(prop.pciDomainID == 0);
    CHECK(prop.pciBusID == 0);
    CHECK(prop.pciDeviceID == 0);
  }
  fclose(f);
  CHECK(choose_records == 1);
}

HRR_TEST_CASE(Unit_HRR_StreamAdvancedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamadvanced"};
  hrr_run_roundtrip("Unit_HRR_StreamAdvanced_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_DrvMemcpyRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy"};
  hrr_run_roundtrip("Unit_HRR_DrvMemcpy_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_OccupancyRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_occupancy"};
  { hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    { set_proc_search_path(proc); }
    int ret = proc.run("\"Unit_HRR_Occupancy_Direct\"");
    INFO("Capture exit: " << ret); REQUIRE(ret == 0); }
  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs")) ++bc;
  INFO("Blob count: " << bc); REQUIRE(bc >= 1);
// Linux does not capture fat-binary code objects at static init, so kernel replay is a no-op.
// Linux asserts only that capture produced events and playback did not crash.
#ifdef _WIN32
  hrr_run_playback(cap.path);
#else
  hrr_run_playback(cap.path, /*extra_args=*/"", /*require_d2h=*/false);
#endif
}

HRR_TEST_CASE(Unit_HRR_HostAliasesRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostaliases"};
  hrr_run_roundtrip("Unit_HRR_HostAliases_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_MemPoolExtendedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_mempoolext"};
  hrr_run_roundtrip("Unit_HRR_MemPoolExtended_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_DeviceMemPoolRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_devicemempool"};
  hrr_run_roundtrip("Unit_HRR_DeviceMemPool_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_ExtMallocRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_extmalloc"};
  hrr_run_roundtrip("Unit_HRR_ExtMalloc_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_StreamWaitEventSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamwaitspt"};
  // The readback stays on the driver-style D2H, which the per-thread header never redirects,
  // so the oracle survives.
  hrr_run_exact_roundtrip("Unit_HRR_StreamWaitEventSpt_Direct", cap.path,
                          {HRR_API_HIPSTREAMWAITEVENT_SPT,
                           HRR_API_HIPMEMCPYDTOHASYNC});
}

HRR_TEST_CASE(Unit_HRR_GraphLaunchSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_graphlaunchspt"};
  // Only the hand-written hipStreamEndCapture handler records a graph HRR can instantiate,
  // so assert the plain begin and end ids: a frame that slipped to the _spt names would leave nothing to replay.
  hrr_run_exact_roundtrip("Unit_HRR_GraphLaunchSpt_Direct", cap.path,
                          {HRR_API_HIPGRAPHLAUNCH_SPT,
                           HRR_API_HIPSTREAMBEGINCAPTURE,
                           HRR_API_HIPSTREAMENDCAPTURE,
                           HRR_API_HIPMEMCPYDTOHASYNC});
}

HRR_TEST_CASE(Unit_HRR_ExtModuleLaunchKernelRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_extmodulelaunch"};
  hrr_run_roundtrip("Unit_HRR_ExtModuleLaunchKernel_Direct", cap.path);
}

// Offset and event_type of each events.bin record before the EOF trailer.
struct RawEvent {
  size_t   offset;
  uint16_t type;
};

static std::vector<RawEvent> hrr_raw_events(const fs::path& events) {
  uint16_t version = 0;
  FILE* f = hrr::open_record_stream(events.string(), HRR_MAGIC, HRR_VERSION, &version);
  REQUIRE(f != nullptr);
  std::vector<RawEvent> out;
  std::vector<uint8_t> raw;
  long off = ftell(f);
  while (hrr::read_raw_record(f, raw) == hrr::RecordStatus::Ok) {
    const auto* hdr = reinterpret_cast<const hrr_event_header*>(raw.data());
    if (hdr->event_type == HRR_EOF_MARKER) break;
    out.push_back({static_cast<size_t>(off), hdr->event_type});
    off = ftell(f);
  }
  fclose(f);
  return out;
}

static void hrr_poke_u16(const fs::path& file, size_t offset, uint16_t value) {
  std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
  f.seekp(static_cast<std::streamoff>(offset));
  f.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/**
 * The runtime marks both events inside the launch, so the archive holds a hipEventRecord on each side of it.
 * The header flag HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS says so.
 * With the flag set a missing stop marker fails replay; with it cleared, as in older archives, replay warns once.
 */
HRR_TEST_CASE(Unit_HRR_ExtLaunchEventMarkers) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_extlaunch_markers"};
  hrr_capture_direct("Unit_HRR_ExtModuleLaunchKernel_Direct", cap.path);

  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 1);
  const fs::path events = archives.front() / "events.bin";
  const std::string fallback = "no marker on a live pair";

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  REQUIRE((arc.flags & HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS) != 0);

  size_t launch = arc.events.size();
  for (size_t i = 0; i < arc.events.size(); ++i)
    if (arc.events[i].header().event_type == HRR_API_HIPMODULELAUNCHKERNEL)
      launch = i;
  REQUIRE(launch > 0);
  REQUIRE(launch + 1 < arc.events.size());
  auto record_event = [&](size_t i) -> uint64_t {
    const hrr::Event& e = arc.events[i];
    REQUIRE(e.header().event_type == HRR_API_HIPEVENTRECORD);
    REQUIRE(e.raw_payload.size() >= sizeof(hrr_args_hipEventRecord));
    return reinterpret_cast<const hrr_args_hipEventRecord*>(e.raw_payload.data())->event;
  };
  const uint64_t start_handle = record_event(launch - 1);
  const uint64_t stop_handle  = record_event(launch + 1);
  CHECK(start_handle != 0);
  CHECK(stop_handle != 0);
  CHECK(start_handle != stop_handle);

  SECTION("replay applies the recorded markers") {
    auto [ret, out] = hrr_playback_merged(cap.path, "", {{"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback output:\n" << out);
    CHECK(out.find("D2H checks") != std::string::npos);
    CHECK(out.find(fallback) == std::string::npos);
#ifndef _WIN32
    CHECK(ret == 0);
#endif
  }

  const std::vector<RawEvent> raw = hrr_raw_events(events);
  size_t at = raw.size();
  for (size_t i = 0; i < raw.size(); ++i)
    if (raw[i].type == HRR_API_HIPMODULELAUNCHKERNEL) at = i;
  REQUIRE(at + 1 < raw.size());
  REQUIRE(raw[at + 1].type == HRR_API_HIPEVENTRECORD);
  // hipExtStreamCreateWithCUMask has a no-op replay handler that never reads its
  // payload, so the swapped record is skipped and the stop event is never marked.
  hrr_poke_u16(events, raw[at + 1].offset, HRR_API_HIPEXTSTREAMCREATEWITHCUMASK);

  SECTION("a marker missing from a marked archive fails the replay") {
    auto [ret, out] = hrr_playback_merged(cap.path, "", {{"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback output:\n" << out);
    CHECK(ret != 0);
    CHECK(out.find("D2H checks") == std::string::npos);
    CHECK(out.find(fallback) == std::string::npos);
  }

  SECTION("an archive without the flag keeps the fallback") {
    const uint16_t flags = static_cast<uint16_t>(
        arc.flags & ~HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS);
    hrr_poke_u16(events, offsetof(hrr_file_header, reserved), flags);
    auto [ret, out] = hrr_playback_merged(cap.path, "", {{"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback output:\n" << out);
    CHECK(out.find("D2H checks") != std::string::npos);
    CHECK(out.find(fallback) != std::string::npos);
#ifndef _WIN32
    CHECK(ret == 0);
#endif
  }
}

HRR_TEST_CASE(Unit_HRR_HostFreeRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostfree"};
  hrr_run_roundtrip("Unit_HRR_HostFree_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_LoggingRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_logging"};
  hrr_run_roundtrip("Unit_HRR_Logging_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_StreamCaptureQuerySptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_capturequeryspt"};
  hrr_run_exact_roundtrip("Unit_HRR_StreamCaptureQuerySpt_Direct", cap.path,
                          {HRR_API_HIPSTREAMISCAPTURING_SPT,
                           HRR_API_HIPSTREAMGETCAPTUREINFO_SPT,
                           HRR_API_HIPSTREAMBEGINCAPTURE,
                           HRR_API_HIPSTREAMENDCAPTURE,
                           HRR_API_HIPMEMCPYDTOHASYNC});
}

HRR_TEST_CASE(Unit_HRR_StreamCaptureBeginSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_capturebeginspt"};
  // The _spt begin handler omits the ctx.in_graph_capture bookkeeping, which is safe for a memset-only capture region.
  // Asserting _spt begin with plain end proves the workload's macro ordering.
  hrr_run_exact_roundtrip("Unit_HRR_StreamCaptureBeginSpt_Direct", cap.path,
                          {HRR_API_HIPSTREAMBEGINCAPTURE_SPT,
                           HRR_API_HIPSTREAMENDCAPTURE,
                           HRR_API_HIPMEMCPYDTOHASYNC});
}

HRR_TEST_CASE(Unit_HRR_ConfigureCallRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_configurecall"};
  hrr_run_roundtrip("Unit_HRR_ConfigureCall_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_MemcpyPeerRoundtrip) {
  int src_dev = 0;
  int dst_dev = 1;
  int ndev = 0;
  if (!hrr_find_peer_accessible_pair(src_dev, dst_dev, ndev)) {
    if (ndev < 2) {
      HRR_SKIP("fewer than two GPUs");
    } else {
      HRR_SKIP("peer access unavailable");
    }
  }
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memcpypeer"};
  hrr_run_roundtrip("Unit_HRR_MemcpyPeer_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_MemsetExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetextra"};
  hrr_run_roundtrip("Unit_HRR_MemsetExtra_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_MemcpyExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memcpyextra"};
  hrr_run_roundtrip("Unit_HRR_MemcpyExtra_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_DeviceExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_deviceextra"};
  hrr_run_roundtrip("Unit_HRR_DeviceExtra_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_StreamAdvanced2Roundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamadv2"};
  hrr_run_roundtrip("Unit_HRR_StreamAdvanced2_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_ContextRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_context"};
  hrr_run_roundtrip("Unit_HRR_Context_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_ModuleExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_moduleextra"};
  hrr_run_roundtrip("Unit_HRR_ModuleExtra_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_MiscAPIsRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_miscapis"};
  hrr_run_roundtrip("Unit_HRR_MiscAPIs_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_DrvMemcpy3DRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("driver memcpy 3D HRR roundtrip is disabled on Windows");
#else
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy3d"};
  hrr_run_exact_roundtrip("Unit_HRR_DrvMemcpy3D_Direct", cap.path);
#endif
}

HRR_TEST_CASE(Unit_HRR_DrvMemcpy2DUnalignedRoundtrip) {
#ifdef _WIN32
  HRR_SKIP("driver memcpy 2D unaligned HRR roundtrip is disabled on Windows");
#else
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy2dunaligned"};
  hrr_run_exact_roundtrip("Unit_HRR_DrvMemcpy2DUnaligned_Direct", cap.path);
#endif
}

HRR_TEST_CASE(Unit_HRR_TextureRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_texture"};
  // On devices without image support the workload exits early after
  // hipDeviceGetAttribute (no D2H memcpy), so D2H validation is skipped.
  int imageSupport = 0;
  (void)hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport, 0);
  // min_events=1: a near-empty archive (fat-binary events only) is legitimate
  // on no-image-support devices.
  hrr_run_roundtrip("Unit_HRR_Texture_Direct", cap.path,
                    /*min_events=*/1, /*require_d2h=*/imageSupport != 0);
}

HRR_TEST_CASE(Unit_HRR_GraphExplicitRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_graphexplicit"};
  hrr_run_roundtrip("Unit_HRR_GraphExplicit_Direct", cap.path);
}

/**
 * The failed hipStreamEndCapture still ended the capture, so replay needs it to end its own.
 * Otherwise the restore and launch would run into a capture replay never closed.
 */
HRR_TEST_CASE(Unit_HRR_FailedEndCaptureRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_failedendcapture"};
  hrr_run_exact_roundtrip("Unit_HRR_FailedEndCapture_Direct", cap.path,
                          {HRR_API_HIPSTREAMBEGINCAPTURE,
                           HRR_API_HIPSTREAMENDCAPTURE,
                           HRR_API_HIPMEMCPYASYNC});
}

/**
 * The failed hipMalloc is not recorded, so replay's capture stays valid and its EndCapture returns a graph, which replay must destroy.
 */
HRR_TEST_CASE(Unit_HRR_FailedEndCaptureInvalidatedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_failedendcaptureinval"};
  hrr_run_exact_roundtrip("Unit_HRR_FailedEndCaptureInvalidated_Direct", cap.path,
                          {HRR_API_HIPSTREAMBEGINCAPTURE,
                           HRR_API_HIPSTREAMENDCAPTURE,
                           HRR_API_HIPMEMCPYASYNC});
}

HRR_TEST_CASE(Unit_HRR_HostRegLaunchRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostreglch"};
  hrr_run_roundtrip("Unit_HRR_HostRegLaunch_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_ModuleAPIRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_moduleapi"};
  hrr_run_roundtrip("Unit_HRR_ModuleAPI_Direct", cap.path);
}

/**
 * hipModuleLoad must record the device ELF the runtime loaded, not the bundle file,
 * so it hashes the same as the bare ELF's hipModuleLoadData.
 */
HRR_TEST_CASE(Unit_HRR_ModuleLoadBundleRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_moduleloadbundle"};
  hrr_run_roundtrip("Unit_HRR_ModuleLoadBundle_Direct", cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  const hrr::Event* load_data = nullptr;
  const hrr::Event* load_file = nullptr;
  const hrr::Event* launch    = nullptr;
  for (const auto& ev : arc.events) {
    const uint16_t type = ev.header().event_type;
    if (type == HRR_API_HIPMODULELOADDATA) load_data = &ev;
    if (type == HRR_API_HIPMODULELOAD) load_file = &ev;
    if (ev.kernel_launch &&
        ev.kernel_launch->kernel_name.find("rtc_fill") != std::string::npos)
      launch = &ev;
  }
  REQUIRE(load_data);
  REQUIRE(load_file);
  REQUIRE(launch);

  const auto& file_ev = load_file->module_load_ev;
  INFO("hipModuleLoad code object: " << hrr::hash_hex(file_ev.hash_lo, file_ev.hash_hi));
  std::vector<uint8_t> file_co;
  REQUIRE(hrr::read_code_object(arc, file_ev.hash_lo, file_ev.hash_hi, file_co));
  INFO("hipModuleLoad code object size: " << file_co.size());
  REQUIRE(file_co.size() >= 4);
  REQUIRE(std::memcmp(file_co.data(), "\x7f" "ELF", 4) == 0);

  const auto& data_ev = load_data->module_load_ev;
  CHECK(file_ev.hash_lo == data_ev.hash_lo);
  CHECK(file_ev.hash_hi == data_ev.hash_hi);
  CHECK(launch->kernel_launch->co_hash_lo == file_ev.hash_lo);
  CHECK(launch->kernel_launch->co_hash_hi == file_ev.hash_hi);
}

HRR_TEST_CASE(Unit_HRR_VMMRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_vmm"};
  hrr_run_roundtrip("Unit_HRR_VMM_Direct", cap.path);
}

HRR_TEST_CASE(Unit_HRR_ChevronLaunchRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_chevron"};
  hrr_run_roundtrip("Unit_HRR_ChevronLaunch_Direct", cap.path, 5);
}

// Capture records only successful calls, so the swap event exists only where swap runs.
static bool hrr_device_supports_batch_swap() {
  HRR_HIP_CHECK(hipSetDevice(0));
  void *a = nullptr, *b = nullptr;
  HRR_HIP_CHECK(hipMalloc(&a, 256));
  HRR_HIP_CHECK(hipMalloc(&b, 256));
  void* dsts[] = {a};
  void* srcs[] = {b};
  size_t sizes[] = {256};
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpSwap;
  size_t attr_idx[] = {0};
  const hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr, 1,
                                                &attr, attr_idx, 1, nullptr);
  if (err == hipErrorNotSupported) {
    (void)hipGetLastError();
  } else {
    HRR_HIP_CHECK(err);
    HRR_HIP_CHECK(hipDeviceSynchronize());
  }
  HRR_HIP_CHECK(hipFree(a));
  HRR_HIP_CHECK(hipFree(b));
  return err != hipErrorNotSupported;
}

HRR_TEST_CASE(Unit_HRR_ExtMemcpyBatchRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_extmemcpybatch"};
  hrr_run_roundtrip("Unit_HRR_ExtMemcpyBatch_Direct", cap.path);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  const auto is_batch = [](const hrr::Event& ev) {
    return ev.header().event_type == HRR_API_HIPEXTMEMCPYBATCHASYNC;
  };
  REQUIRE(std::any_of(arc.events.begin(), arc.events.end(), is_batch));
  if (!hrr_device_supports_batch_swap()) return;
  REQUIRE(std::any_of(arc.events.begin(), arc.events.end(), [&](const hrr::Event& ev) {
    if (!is_batch(ev)) return false;
    hrr_args_hipExtMemcpyBatchAsync a{};
    if (ev.raw_payload.size() < sizeof(a)) return false;
    std::memcpy(&a, ev.raw_payload.data(), sizeof(a));
    size_t sizes_dst = 0;
    std::memcpy(&sizes_dst, a.sizesDst_bytes, sizeof(sizes_dst));
    return a.sizesDst_present && a.sizesDst_n == 1 && sizes_dst == 1024;
  }));
}

HRR_TEST_CASE(Unit_HRR_MultiThreadRoundtrip) {
  static constexpr const char* raw_trace_exe = HRR_RAW_TRACE_EXE;
  if (!raw_trace_exe || raw_trace_exe[0] == '\0') {
    SUCCEED("hip_raw_trace not found at configure time — skipping "
            "(pass -DHRR_RAW_TRACE_EXE=<path> to cmake)");
    return;
  }

  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_multithread"};

  {
    hrr::test::SpawnProc proc(raw_trace_exe);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("");
    INFO("hip_raw_trace capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs")) ++bc;
  INFO("Blob count: " << bc);
  REQUIRE(bc >= 1);

// Playback only has to start, read the archive and not crash: fat-binary kernels replay only with the capture DLL.
// --skip-device-sync: hipStreamSynchronize returns error 900 on a stream that is still capturing.
  auto run_mt_playback = [&](const std::string& extra_args) {
    hrr::test::SpawnProc proc(hrr_playback_exe(), /*capture_stdout=*/true);
    set_proc_search_path(proc);
#ifdef _WIN32
    std::string mt_path_arg = "\"" + cap.path.string() + "\"";
#else
    std::string mt_path_arg = cap.path.string();
#endif
    int ret = proc.run(mt_path_arg + " --skip-device-sync" +
                       (extra_args.empty() ? "" : " " + extra_args));
    std::string out = proc.getOutput();
    INFO("Playback args: " + extra_args);
    INFO("Playback stdout:\n" << out);
    INFO("Playback exit code: " << ret);
    // Fat-binary kernel replay needs the capture DLL, so CI accepts a D2H mismatch (exit 1).
    // A crash (exit >= 128) always fails.
    REQUIRE(ret < 128);
    REQUIRE(out.find("[HRR] Archive") != std::string::npos);
  };

  run_mt_playback("");               // single-thread
  run_mt_playback("--multi-thread"); // MT dispatch path
}

// --replace-kernel N=PATH matches N against the recorded (mangled) kernel name and loads the same symbol from PATH.
// D2H is validated against recorded blobs, so the happy path uses an identical kernel and asserts the "Replacing kernel" log line.

#ifndef _WIN32  // hiprtc replacement CO build + load is exercised on Linux CI
#define HRR_HIPRTC_CHECK(expr)                                                  \
  do {                                                                          \
    hiprtcResult _r = (expr);                                                   \
    if (_r != HIPRTC_SUCCESS) {                                                 \
      FAIL("HIPRTC error " << static_cast<int>(_r) << " at " #expr);            \
    }                                                                           \
  } while (0)

static const char* k_replace_vectoradd_src = R"(
__global__ void hrr_vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
)";

static std::string build_replacement_co(const fs::path& dir) {
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, k_replace_vectoradd_src,
                                       "hrr_replace.hip", 0, nullptr, nullptr));
  hiprtcResult compile_rc = hiprtcCompileProgram(prog, 0, nullptr);
  if (compile_rc != HIPRTC_SUCCESS) {
    size_t log_sz = 0;
    (void)hiprtcGetProgramLogSize(prog, &log_sz);
    std::string log(log_sz, '\0');
    (void)hiprtcGetProgramLog(prog, log.data());
    (void)hiprtcDestroyProgram(&prog);
    FAIL("hiprtcCompileProgram failed: " + log);
  }
  size_t co_size = 0;
  HRR_HIPRTC_CHECK(hiprtcGetCodeSize(prog, &co_size));
  std::vector<char> co(co_size);
  HRR_HIPRTC_CHECK(hiprtcGetCode(prog, co.data()));
  HRR_HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  fs::create_directories(dir);
  std::string co_path = (dir / "replacement.hsaco").string();
  std::ofstream out(co_path, std::ios::binary);
  out.write(co.data(), static_cast<std::streamsize>(co.size()));
  out.close();
  return co_path;
}

// --replace-kernel matches the mangled name exactly, so look it up from the archive.
static std::string recorded_kernel_name(const fs::path& cap_path,
                                        const std::string& needle) {
  hrr::Archive arc;
  if (!hrr::load_archive(cap_path.string(), arc)) return "";
  for (const auto& ev : arc.events) {
    if (ev.kernel_launch &&
        ev.kernel_launch->kernel_name.find(needle) != std::string::npos)
      return ev.kernel_launch->kernel_name;
  }
  return "";
}

HRR_TEST_CASE(Unit_HRR_ReplaceKernelRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_replace_happy"};
  ScopedDir co_dir{fs::temp_directory_path() / "hrr_replace_co"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    INFO("Capture exit: " << ret);
    REQUIRE(ret == 0);
  }
  REQUIRE(fs::exists(hrr_single_process_archive(cap.path) / "events.bin"));

  std::string kname = recorded_kernel_name(cap.path, "hrr_vectorAdd");
  INFO("Recorded kernel name: " << kname);
  REQUIRE_FALSE(kname.empty());

  std::string co_path = build_replacement_co(co_dir.path);
  REQUIRE(fs::exists(co_path));

  auto [ret, out] = run_playback_raw(cap.path, "--replace-kernel " + kname + "=" + co_path);
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  REQUIRE(ret == 0);
  CHECK(out.find("Replacing kernel") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_ReplaceKernelMissingCO) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_replace_missing"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    INFO("Capture exit: " << ret);
    REQUIRE(ret == 0);
  }
  REQUIRE(fs::exists(hrr_single_process_archive(cap.path) / "events.bin"));

  std::string kname = recorded_kernel_name(cap.path, "hrr_vectorAdd");
  INFO("Recorded kernel name: " << kname);
  REQUIRE_FALSE(kname.empty());

  std::string missing =
      (fs::temp_directory_path() / "hrr_no_such_replacement.hsaco").string();
  fs::remove(missing);  // ensure it really does not exist

  auto [ret, out] = run_playback_raw(cap.path, "--replace-kernel " + kname + "=" + missing);
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  REQUIRE(ret == 0);
}

HRR_TEST_CASE(Unit_HRR_ReplaceKernelBadSpec) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_replace_badspec"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    REQUIRE(ret == 0);
  }
  REQUIRE(fs::exists(hrr_single_process_archive(cap.path) / "events.bin"));

  auto [ret, out] = run_playback_raw(cap.path, "--replace-kernel noequalshere");
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  REQUIRE(ret != 0);   // parser rejects malformed spec
  REQUIRE(ret < 128);  // ...with a clean error, not a crash
}
#endif  // !_WIN32

#ifndef _WIN32
/**
 * The crash handler runs on the thread's 64 KiB stack, and the emergency manifest buffer is larger than it.
 */
HRR_TEST_CASE(Unit_HRR_CaptureCrashOnSmallStack) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_small_stack"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_CaptureCrashSmallStack_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 128 + SIGSEGV);
  }

  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "manifest.json"));
  const std::string manifest = read_text_file(archive_path / "manifest.json");
  INFO("Process manifest:\n" << manifest);
  REQUIRE(manifest.find("\"complete\": false") != std::string::npos);
}

/**
 * A child must not inherit a writer mutex that another recording thread holds locked.
 * A child opens its archive on its first record, so one that exited at once leaves none.
 */
HRR_TEST_CASE(Unit_HRR_ForkWhileRecording) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_fork_while_recording"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkWhileRecording_Direct\"", 600);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  CHECK(hrr_process_archives(cap.path).size() ==
        static_cast<size_t>(kHrrForkWhileRecordingArchives));
}

/**
 * fork() keeps the writer's events mutex locked until it returns, so no other thread can take it in between
 * and leave the child a mutex that only a missing thread could unlock.
 */
HRR_TEST_CASE(Unit_HRR_ForkWhileWriterHoldsLock) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_fork_writer_holds_lock"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkWhileWriterHoldsLock_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  CHECK(hrr_process_archives(cap.path).size() == 2);
}

/**
 * A child forked while the parent's crash callback writes the manifest must not inherit
 * the emergency manifest buffer marked busy.
 */
HRR_TEST_CASE(Unit_HRR_CaptureCrashDuringFork) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_crash_during_fork"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_CaptureCrashDuringFork_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 128 + SIGSEGV);
  }

  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 2);
  for (const fs::path& archive : archives) {
    INFO("Archive: " << archive.string());
    REQUIRE(fs::exists(archive / "manifest.json"));
    const std::string manifest = read_text_file(archive / "manifest.json");
    INFO("Process manifest:\n" << manifest);
    CHECK(manifest.find("\"complete\": false") != std::string::npos);
  }
}

/**
 * fork() waits for a thread noting an unreplayable API, so the child does not inherit that list's mutex locked.
 * The child's manifest omits the API because its archive holds no event for it.
 */
HRR_TEST_CASE(Unit_HRR_ForkWhileNotingUnreplayable) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_fork_while_noting_unreplayable"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkWhileNotingUnreplayable_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 2);
  for (const fs::path& archive : archives) {
    INFO("Archive: " << archive.string());
    REQUIRE(fs::exists(archive / "manifest.json"));
    const std::string manifest = read_text_file(archive / "manifest.json");
    INFO("Process manifest:\n" << manifest);
    const fs::path& other = archive == archives[0] ? archives[1] : archives[0];
    const std::string other_pid = other.filename().string().substr(std::strlen("pid-"));
    const bool child = manifest.find("\"parent_pid\": " + other_pid + ",") != std::string::npos;
    CHECK((manifest.find("\"hipUserObjectCreate\"") != std::string::npos) == !child);
  }
}

/**
 * A forked child drops a record before its first one opens its archive, so its manifest says "complete": false.
 */
HRR_TEST_CASE(Unit_HRR_ForkedChildStaysIncomplete) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_forked_child_stays_incomplete"};
  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkedChildStaysIncomplete_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }
  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 2);
  int children = 0;
  for (const fs::path& archive : archives) {
    const std::string manifest = read_text_file(archive / "manifest.json");
    INFO("Process manifest:\n" << manifest);
    const fs::path& other = archive == archives[0] ? archives[1] : archives[0];
    const std::string other_pid = other.filename().string().substr(std::strlen("pid-"));
    const bool child = manifest.find("\"parent_pid\": " + other_pid + ",") != std::string::npos;
    children += child ? 1 : 0;
    CHECK(manifest.find(child ? "\"complete\": false" : "\"complete\": true") !=
          std::string::npos);
  }
  REQUIRE(children == 1);
}

/**
 * A forked child that records after its capture shutdown, as a fat-binary destructor does, leaves no archive.
 */
HRR_TEST_CASE(Unit_HRR_ForkedChildRecordsAfterShutdown) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_forked_child_records_after_shutdown"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkedChildRecordsAfterShutdown_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  CHECK(hrr_process_archives(cap.path).size() == 1);
}
HRR_TEST_CASE(Unit_HRR_ForkAfterCaptureShutdown) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_fork_after_capture_shutdown"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkAfterCaptureShutdown_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  CHECK(hrr_process_archives(cap.path).size() == 1);
}

// events.bin must end in the clean-shutdown trailer, which counts every event before it.
// load_archive() sets `complete` for a trailer anywhere in the file, so check the tail here.
static void check_ends_in_trailer(const fs::path& archive, const hrr::Archive& arc) {
  const std::string bytes = read_text_file(archive / "events.bin");
  REQUIRE(bytes.size() >= sizeof(hrr_file_header) + sizeof(hrr_eof_record));
  hrr_eof_record eof;
  memcpy(&eof, bytes.data() + bytes.size() - sizeof(eof), sizeof(eof));
  CHECK(eof.hdr.event_type == HRR_EOF_MARKER);
  CHECK(eof.hdr.payload_length == sizeof(hrr_eof_record));
  CHECK(eof.eof_magic == HRR_EOF_MAGIC);
  CHECK(eof.total_events == arc.events.size());
}

/**
 * The child's shutdown waits while another thread opens the child's archive, then finalizes it,
 * so events.bin ends in the trailer with no record after it.
 */
HRR_TEST_CASE(Unit_HRR_ShutdownWhileChildOpensArchive) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_shutdown_while_child_opens_archive"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ShutdownWhileChildOpensArchive_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  const std::vector<fs::path> archives = hrr_process_archives(cap.path);
  REQUIRE(archives.size() == 2);
  for (const fs::path& archive : archives) {
    INFO("Archive: " << archive.string());
    REQUIRE(fs::exists(archive / "manifest.json"));
    const std::string manifest = read_text_file(archive / "manifest.json");
    INFO("Process manifest:\n" << manifest);
    CHECK(manifest.find("\"complete\": true") != std::string::npos);
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(archive.string(), arc));
    CHECK(arc.complete);
    check_ends_in_trailer(archive, arc);
  }
}

/**
 * A record made after the trailer is written and before events.bin closes is dropped.
 */
HRR_TEST_CASE(Unit_HRR_RecordAfterCaptureShutdown) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_record_after_capture_shutdown"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_RecordAfterCaptureShutdown_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  const fs::path archive = hrr_single_process_archive(cap.path);
  const std::string manifest = read_text_file(archive / "manifest.json");
  INFO("Process manifest:\n" << manifest);
  CHECK(manifest.find("\"complete\": true") != std::string::npos);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  CHECK(arc.complete);
  check_ends_in_trailer(archive, arc);
}

HRR_TEST_CASE(Unit_HRR_ForkBetweenCaptureFlushAndClose) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_fork_between_capture_flush_and_close"};

  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.runWithTimeout("\"Unit_HRR_ForkBetweenCaptureFlushAndClose_Direct\"", 120);
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  REQUIRE(hrr_process_archives(cap.path).size() == 1);
  const fs::path archive = hrr_single_process_archive(cap.path);
  hrr::Archive arc;
  REQUIRE(hrr::load_archive(archive.string(), arc));
  CHECK(arc.complete);
  check_ends_in_trailer(archive, arc);
}

/**
 * Shutdown must not close the archive's directory descriptors while a crash callback
 * on another thread writes the manifest through one of them.
 */
HRR_TEST_CASE(Unit_HRR_CloseDuringCrashManifest) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_close_during_crash_manifest"};
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout("\"Unit_HRR_CloseDuringCrashManifest_Direct\"", 120);
  INFO("Capture exit code: " << ret << "\n" << proc.getOutput());
  CHECK(proc.getOutput().find("capture shutdown started while the crash callback was held") !=
        std::string::npos);
  REQUIRE(ret == 128 + SIGSEGV);
}

/**
 * fork() waits for a blob write in flight, so the child holds no descriptor of a blobs/<xx> directory.
 */
HRR_TEST_CASE(Unit_HRR_ForkDuringBlobWrite) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_fork_during_blob_write"};
  hrr::test::SpawnProc proc(hrr_self_exe());
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
  set_proc_search_path(proc);
  const int ret = proc.runWithTimeout("\"Unit_HRR_ForkDuringBlobWrite_Direct\"", 120);
  INFO("Capture exit code: " << ret);
  REQUIRE(ret == 0);
}
#endif  // !_WIN32

/**
 * The copy extent far exceeds both buffers, so capture must not hash extent-many bytes from the 4 KiB host buffer.
 */
HRR_TEST_CASE(Unit_HRR_FailedMemcpy3DNotRecorded) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_failed_memcpy3d"};
  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    const int ret = proc.run("\"Unit_HRR_FailedMemcpy3D_Direct\"");
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }
  const auto counts = hrr_info_api_counts(cap.path);
  const auto it = counts.find("hipMalloc3D");
  REQUIRE(it != counts.end());  // the capture was live
  CHECK(it->second == 1);
  CHECK(counts.count("hipMemcpy3D") == 0);
  CHECK(counts.count("hipMemcpy3DAsync") == 0);
  CHECK(counts.count("hipMemcpy3D_spt") == 0);
  CHECK(counts.count("hipMemcpy3DAsync_spt") == 0);
}

/**
 * The 70,000-character name does not fit the uint16_t length on the wire,
 * so capture must record no launch and mark the archive incomplete.
 */
HRR_TEST_CASE(Unit_HRR_LongKernelNameNotRecorded) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_long_kernel_name"};
  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    const int ret = proc.run("\"Unit_HRR_LongKernelName_Direct\"");
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }
  const auto counts = hrr_info_api_counts(cap.path);
  const auto it = counts.find("hipModuleLoadData");
  REQUIRE(it != counts.end());  // the capture was live
  CHECK(counts.count("hipModuleLaunchKernel") == 0);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  CHECK_FALSE(arc.complete);
}

/**
 * The extent product overflows size_t, so no blob can be sized and the copy must not be recorded.
 * Recorded blobless, it would take the device-to-device replay path with an unmapped host source.
 */
HRR_TEST_CASE(Unit_HRR_OverflowingMemcpy3DNotRecorded) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_overflowing_memcpy3d"};
  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    const int ret = proc.run("\"Unit_HRR_OverflowingMemcpy3D_Direct\"");
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }
  const auto counts = hrr_info_api_counts(cap.path);
  const auto it = counts.find("hipStreamBeginCapture");
  REQUIRE(it != counts.end());  // the capture was live
  CHECK(it->second == 1);
  CHECK(counts.count("hipMemcpy3DAsync") == 0);

  hrr::Archive arc;
  REQUIRE(hrr::load_archive(cap.path.string(), arc));
  CHECK_FALSE(arc.complete);

  auto [rc, out] = hrr_playback_merged(hrr_single_process_archive(cap.path));
  INFO("Replay output:\n" << out);
  CHECK(rc == 0);
}

/**
 * Replay issues every recorded call again, so a recorded rejected call would stop it.
 */
HRR_TEST_CASE(Unit_HRR_FailedShimCallsNotRecorded) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_failed_shim_calls"};
  {
    hrr::test::SpawnProc proc(hrr_self_exe());
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    const int ret = proc.run("\"Unit_HRR_FailedShimCalls_Direct\"");
    INFO("Capture exit code: " << ret);
    REQUIRE(ret == 0);
  }
  const auto counts = hrr_info_api_counts(cap.path);
  const auto pools = counts.find("hipMemPoolCreate");
  REQUIRE(pools != counts.end());  // the capture was live
  CHECK(pools->second == 1);
  const auto attrs = counts.find("hipMemPoolSetAttribute");
  REQUIRE(attrs != counts.end());
  CHECK(attrs->second == 1);
  for (const char* api :
       {"hipMemPoolSetAccess", "hipMemSetAccess", "hipArrayCreate", "hipArray3DCreate",
        "hipStreamSetAttribute", "hipMemGetAllocationGranularity"}) {
    INFO("API: " << api);
    CHECK(counts.count(api) == 0);
  }

  // The accepted reuse policy is an int32_t, so capture copies 4 bytes and the
  // sentinel the workload put after it stays out of the recorded value.
  {
    hrr::Archive arc;
    REQUIRE(hrr::load_archive(cap.path.string(), arc));
    const auto ev = std::find_if(arc.events.begin(), arc.events.end(), [](const hrr::Event& e) {
      return e.header().event_type == static_cast<uint16_t>(HRR_API_HIPMEMPOOLSETATTRIBUTE);
    });
    REQUIRE(ev != arc.events.end());
    hrr_args_hipMemPoolSetAttribute a{};
    REQUIRE(ev->raw_payload.size() >= sizeof(a));
    std::memcpy(&a, ev->raw_payload.data(), sizeof(a));
    CHECK(a.attr == static_cast<int32_t>(hipMemPoolReuseAllowOpportunistic));
    CHECK(a.value_u64 == 1);
  }

  auto [rc, out] = hrr_playback_merged(hrr_single_process_archive(cap.path));
  INFO("Replay output:\n" << out);
  CHECK(rc == 0);
}
