/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Capture/Replay Roundtrip
 * @{
 * @ingroup HRRTest
 * Subprocess-based tests that avoid MSYS2/bash SEH-exception-handling
 * interference with HIP's internal __try/__except frames:
 *
 *   Unit_HRR_CaptureReplayRoundtrip:
 *     1. Sets HIP_HRR_CAPTURE_OUTPUT and spawns Unit_HRR_GpuWorkload_Direct
 *        so the capture layer records all HIP API calls and D2H blobs.
 *        The subprocess exiting 0 also validates GPU correctness (all
 *        REQUIRE(hc[i]==2.0f) passed in the Direct test).
 *     2. Verifies the archive exists and contains at least one blob.
 *     3. Runs hrr-playback on the archive; validates D2H buffers byte-for-byte
 *        against captured blobs. Any mismatch → non-zero exit → REQUIRE fails.
 *     4. Deletes the temp archive directory (via RAII, even on failure).
 *
 *   Unit_HRR_GraphRoundtrip:
 *     Same as Unit_HRR_CaptureReplayRoundtrip but for the HIP graph workload.
 *
 * HRR_TEST_EXE and HRR_PLAYBACK_EXE are required; CMakeLists.txt fails at
 * configure time if HRR_PLAYBACK_EXE is not found. Pass -DHRR_PLAYBACK_EXE=<path>
 * if hrr-playback is not installed under CMAKE_INSTALL_PREFIX or ROCM_PATH.
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <hip/hiprtc.h>
#include "hrr_reader.h"
#include "hrr_api_args.h"
#include "hrr_test_common.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// fs, kPathSep, set_proc_search_path, ScopedDir, hrr_single_process_archive,
// read_text_file, hrr_run_playback, hrr_run_roundtrip, hrr_capture_direct,
// hrr_playback_env and run_playback_raw all live in hrr_test_common.h so the
// API-matrix tests can reuse them.

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

// ---------------------------------------------------------------------------
/**
 * Test Description
 * ----------------
 *   - Spawns HrrTest Unit_HRR_GpuWorkload_Direct as a subprocess with
 *     HIP_HRR_CAPTURE_OUTPUT set to a temp directory.  The capture layer
 *     records all HIP API calls and writes a blob for each D2H memcpy.
 *   - Verifies the archive exists and contains at least one D2H blob.
 *   - Runs hrr-playback on the archive.  It replays every event and validates
 *     each D2H host buffer against the captured blob byte-for-byte.
 *   - REQUIRE(playback exit == 0): any D2H mismatch causes failure.
 *   - Deletes the temp archive directory on scope exit.
 */
HIP_TEST_CASE(Unit_HRR_CaptureReplayRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_gpu"};

  // -------------------------------------------------------------------------
  // Step 1: capture
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    // Prepend ROCm bin to PATH so the subprocess finds amdhip64_7.dll.
    // SpawnProc replaces PATH entirely, so we reconstruct the full value.
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    INFO("Capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  // -------------------------------------------------------------------------
  // Step 2: verify archive structure
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // Step 3: playback + D2H validation
  //   hrr-playback replays every event; for each D2H memcpy it copies the
  //   replayed host buffer into a staging allocation and compares against the
  //   stored blob.  Any mismatch → exit 1.
  // -------------------------------------------------------------------------
  hrr_run_playback(cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Spawns HrrTest Unit_HRR_AllApis_Direct as a subprocess with
 *     HIP_HRR_CAPTURE_OUTPUT set to a temp directory.  Exercises ~55 distinct
 *     HIP APIs covering device queries, streams, events, malloc variants
 *     (Malloc/Async/Pool/Host/Managed), memset, memcpy variants, occupancy,
 *     pointer attributes, cache config, and (conditionally) managed-memory
 *     advise/prefetch.
 *   - Verifies the archive exists and contains at least one D2H blob.
 *   - Runs hrr-playback on the archive; validates d2[i]==94 byte-for-byte.
 *   - REQUIRE(playback exit == 0): any D2H mismatch causes failure.
 *   - Deletes the temp archive directory on scope exit.
 */
HIP_TEST_CASE(Unit_HRR_AllApisRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_allapis"};

  // -------------------------------------------------------------------------
  // Step 1: capture
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_AllApis_Direct\"");
    INFO("AllApis capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  // -------------------------------------------------------------------------
  // Step 2: verify archive structure
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // Step 3: playback + D2H validation (d2[i] == 94)
  // -------------------------------------------------------------------------
  hrr_run_playback(cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Spawns HrrTest Unit_HRR_HostMemWorkload_Direct as a subprocess with
 *     HIP_HRR_CAPTURE_OUTPUT set to a temp directory.
 *   - Verifies the archive exists and contains at least one blob.
 *   - Runs hrr-playback on the archive; validates D2H memcpy buffers byte-for-byte
 *     against the captured expected-output blob (value == 2).
 *   - REQUIRE(playback exit == 0): any D2H mismatch causes failure.
 *   - Deletes the temp archive directory on scope exit.
 */
HIP_TEST_CASE(Unit_HRR_HostMemRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostmem"};

  // -------------------------------------------------------------------------
  // Step 1: capture
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_HostMemWorkload_Direct\"");
    INFO("HostMem capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  // -------------------------------------------------------------------------
  // Step 2: verify archive structure
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // Step 3: playback + D2H validation
  // HostMem workload now includes an explicit D2H memcpy (value == 2), so playback
  // can validate the D2H blob byte-for-byte against the captured expected output.
  // -------------------------------------------------------------------------
  hrr_run_playback(cap.path, "", /*require_d2h=*/true);
}

/**
 * Test Description
 * ----------------
 *   - Spawns HrrTest Unit_HRR_GraphWorkload_Direct as a subprocess with
 *     HIP_HRR_CAPTURE_OUTPUT set to a temp directory.
 *   - Verifies the archive exists and contains at least one D2H blob.
 *   - Runs hrr-playback on the archive; validates D2H buffers byte-for-byte.
 *   - REQUIRE(playback exit == 0): any D2H mismatch causes failure.
 *   - Deletes the temp archive directory on scope exit.
 */
HIP_TEST_CASE(Unit_HRR_GraphRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_graph"};

  // -------------------------------------------------------------------------
  // Step 1: capture
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    // Prepend ROCm bin to PATH so the subprocess finds amdhip64_7.dll.
    // SpawnProc replaces PATH entirely, so we reconstruct the full value.
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_GraphWorkload_Direct\"");
    INFO("Graph capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  // -------------------------------------------------------------------------
  // Step 2: verify archive structure
  // -------------------------------------------------------------------------
  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs"))
    ++blob_count;
  INFO("Blob count: " << blob_count);
  REQUIRE(blob_count >= 1);

  // -------------------------------------------------------------------------
  // Step 3: playback + D2H validation
  // -------------------------------------------------------------------------
  hrr_run_playback(cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Spawns HrrTest Unit_HRR_StressApis_Direct as a subprocess with
 *     HIP_HRR_CAPTURE_OUTPUT set to a temp directory.  Generates 500+
 *     HIP API call events covering stream/event lifecycle, many alloc/free
 *     cycles, memset/memcpy loops, repeated kernel launches, and device
 *     attribute queries.
 *   - Verifies the archive exists and contains at least one D2H blob.
 *   - Runs hrr-playback on the archive; validates D2H buffers byte-for-byte
 *     (h_out[i] == 2.0f).
 *   - REQUIRE(playback exit == 0): any D2H mismatch causes failure.
 *   - Deletes the temp archive directory on scope exit.
 */
HIP_TEST_CASE(Unit_HRR_StressApisRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_stress"};

  // -------------------------------------------------------------------------
  // Step 1: capture
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("\"Unit_HRR_StressApis_Direct\"");
    INFO("Stress capture subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }

  // -------------------------------------------------------------------------
  // Step 2: verify archive structure
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // Step 3: playback + D2H validation (h_out[i] == 2.0f)
  // -------------------------------------------------------------------------
  hrr_run_playback(cap.path);
}

static bool hrr_find_peer_accessible_pair(int& src_dev, int& dst_dev, int& ndev) {
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) return false;

  for (int src = 0; src < ndev; ++src) {
    for (int dst = 0; dst < ndev; ++dst) {
      if (src == dst) continue;
      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src, dst));
      if (can_access) {
        src_dev = src;
        dst_dev = dst;
        return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Repro roundtrips — capture the micro-kernel workloads from
// hrr_repro_workload.cc and validate the playback behaviours fixed in this area.
// ---------------------------------------------------------------------------

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_FirstMalloc_Direct, whose very first HIP call is a
 *     hipMalloc (no hipSetDevice / warm-up allocation first).  Replay and
 *     REQUIRE the D2H validates — i.e. the very first allocation was captured
 *     and translated.
 *
 *   Hidden ([.]) because it reproduces the known dropped-first-hipMalloc
 *   limitation: capture shims are installed from hip_capture_init() (run inside
 *   hip::init()), which is the in-flight first HIP API call, so that first call
 *   itself is not recorded.  When the first call is a hipMalloc the allocation
 *   is missing and replay aborts with "H2D dst ... not mapped".  The accepted
 *   workaround is a warm-up allocation before the real work.  Run explicitly by
 *   name; this case PASSES once first-call capture is fixed (the warm-up is no
 *   longer required), so it doubles as the regression guard for that fix.
 */
TEST_CASE("Unit_HRR_FirstMallocRoundtrip", "[.][hrr-repro]") {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_firstmalloc"};
  hrr_run_roundtrip("Unit_HRR_FirstMalloc_Direct", cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_EmbeddedPtrStruct_Direct (kernel takes a by-value struct
 *     with embedded device pointers).  Replay and REQUIRE D2H pass >= 1, proving
 *     the embedded pointers were detected at capture and translated at replay.
 */
HIP_TEST_CASE(Unit_HRR_EmbeddedPtrRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_embeddedptr"};
  hrr_run_roundtrip("Unit_HRR_EmbeddedPtrStruct_Direct", cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_ZeroInitRead_Direct (uninitialised device buffer copied
 *     to out, then D2H; captured blob is all-zero on a zeroing driver).
 *   - Replay with HIP_HRR_REPLAY_ZERO_INIT=1: the replayed source is zeroed
 *     deterministically, so the D2H validates (exit 0, pass >= 1).  Verifies the
 *     zero-init replay knob.  Note: with the knob off the replay may reuse stale
 *     bytes and diverge — that path is intentionally not asserted here.
 */
HIP_TEST_CASE(Unit_HRR_ZeroInitRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_zeroinit"};
  hrr_capture_direct("Unit_HRR_ZeroInitRead_Direct", cap.path);

  auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_REPLAY_ZERO_INIT", "1"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  REQUIRE(ret == 0);  // zero-init reproduces the captured all-zero output

  int d2h_pass = 0, d2h_fail = 0;
  REQUIRE(hrr_parse_d2h_summary(out, d2h_pass, d2h_fail));
  INFO("D2H pass=" << d2h_pass << " fail=" << d2h_fail);
// With ASAN enabled this won't be true, because inside Unit_HRR_ZeroInitRead_Direct
// fresh device allocations are not zeroed with ASAN enabled
#if !defined(ENABLE_ADDRESS_SANITIZER)
  CHECK(d2h_pass >= 1);
  CHECK(d2h_fail == 0);
#endif
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_UncapturedHostWrite_Direct: a mapped host flag is set by
 *     an uncaptured CPU store, then kDivergeIters kernels each write the flag
 *     value out and read it back.  Every replay D2H diverges (fresh zero flag).
 *   - Section 1 (guard ON): replay with a low divergence-abort threshold,
 *     exact D2H validation, and a small min-samples count.  REQUIRE clean exit
 *     code 2 (replay diverged) — the guard stops the replay deterministically.
 *     Directly validates the divergence-abort guard.
 *   - Section 2 (guard OFF): replay with HIP_HRR_REPLAY_DIVERGENCE_ABORT=0 and
 *     exact D2H validation.  REQUIRE the exit code is NOT 2 (it runs to
 *     completion / D2H-fail), proving the guard is what produces exit 2, not
 *     some unrelated error.
 *
 *   Regression guard for ROCM-27652: the guard-ON path takes hrr-playback's
 *   early divergence-abort exit, which must still tear down every GPU/host
 *   resource tracked in the PlaybackContext.  Under the AddressSanitizer CI
 *   build a leak on this path is reported by LeakSanitizer, so this test is the
 *   guard that the divergence-abort teardown stays leak-free.  The clean exit 2
 *   (not a signal/abort >= 128) is the deterministic contract asserted here.
 */
HIP_TEST_CASE(Unit_HRR_DivergenceAbortRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_divergence"};
  hrr_capture_direct("Unit_HRR_UncapturedHostWrite_Direct", cap.path);

  SECTION("guard ON -> clean exit 2") {
    auto [ret, out] = hrr_playback_env(
        cap.path,
        {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0.25"},
         {"HIP_HRR_D2H_EXACT", "1"},
         {"HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES", "16"}});
    INFO("Playback stdout:\n" << out);
    INFO("Playback exit code: " << ret);
    // Exit 2 == divergence guard tripped and stopped cleanly. The "replay
    // DIVERGED" text is on stderr (not captured), so the exit code is the
    // asserted contract.
    // A clean divergence-abort, never a crash/sanitizer abort (>= 128).
    REQUIRE(ret < 128);
    // Exit 2 == divergence guard tripped and stopped cleanly.
    REQUIRE(ret == 2);
  }

  SECTION("guard OFF -> not exit 2") {
    auto [ret, out] = hrr_playback_env(
        cap.path,
        {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0"},
         {"HIP_HRR_D2H_EXACT", "1"}});
    INFO("Playback stdout:\n" << out);
    INFO("Playback exit code: " << ret);
    // With the guard disabled the replay does not stop early: it either
    // completes (exit 0/1 from D2H-fail) — never the divergence exit 2.
    REQUIRE(ret != 2);
    REQUIRE(ret < 128);  // and never a crash
  }
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_NullOptionalPtr_Direct: a divergence loop (uncaptured
 *     host write) followed by a slotmap kernel whose optional output pointer is
 *     NULL at capture and would be written at byte offset 0x20000 on replay.
 *   - Replay with the divergence guard ON: the guard trips during the readback
 *     loop and stops the replay (exit 2) BEFORE the faulting kernel runs, so the
 *     null + 0x20000 write never happens.  REQUIRE a clean exit 2 rather than a
 *     GPU memory fault.
 *
 *   Hidden ([.]) because a mistuned guard could let the raw GPU fault through,
 *   which can destabilise a shared runner.  Run explicitly by name to validate
 *   the guard converts the fault class into a clean exit.
 */
TEST_CASE("Unit_HRR_NullOptionalPtrRoundtrip", "[.][hrr-repro]") {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_nulloptional"};
  hrr_capture_direct("Unit_HRR_NullOptionalPtr_Direct", cap.path);

  auto [ret, out] = hrr_playback_env(
      cap.path,
      {{"HIP_HRR_REPLAY_DIVERGENCE_ABORT", "0.25"},
       {"HIP_HRR_REPLAY_DIVERGENCE_MIN_SAMPLES", "16"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  // Clean divergence stop (2), NOT a GPU fault (signal -> >128) nor a generic
  // fatal HIP error (1) from the null write.
  REQUIRE(ret == 2);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_StreamWriteValue_Direct (hipStreamWriteValue32 /
 *     hipStreamWriteValue64, including one hipExtStreamWriteValueIncrement
 *     write) and replay it.  Replay must reproduce every written value.
 *   - Replay with HIP_HRR_D2H_EXACT=1.  This is deliberate, not decoration:
 *     with the default tolerant validator a lost 32-bit write is accepted as
 *     "f64 within tolerance" on any blob whose length is a multiple of 8, and
 *     the increment slot differs from its no-increment value by far less than
 *     atol=rtol=1e-3 of the recorded magnitude.  Exact mode makes the playback
 *     exit code (the real gate) a byte-for-byte verdict.
 *   - Gated on hipDeviceAttributeCanUseStreamWaitValue: on a target without
 *     support the workload skips, which would leave too few events / no D2H
 *     blob for the archive assertions, so skip the roundtrip as well.
 */
HIP_TEST_CASE(Unit_HRR_StreamWriteValueRoundtrip) {
  int canUseStreamValue = 0;
  HIP_CHECK(hipDeviceGetAttribute(&canUseStreamValue,
                                  hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (!canUseStreamValue) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamwritevalue"};
  hrr_capture_direct("Unit_HRR_StreamWriteValue_Direct", cap.path);

  auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_D2H_EXACT", "1"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
#ifdef _WIN32
  // Same policy as hrr_run_playback: on the Windows consumer-iGPU CI target
  // replay is not guaranteed to reproduce device output bit-for-bit, so D2H
  // fidelity is best-effort there.  A crash still fails via ret < 128.
  REQUIRE(ret < 128);
  if (ret != 0) return;
#else
  REQUIRE(ret == 0);  // exact-mode D2H: any differing byte fails the replay
#endif

  // Assert the three sentinel blobs were actually compared, so a replay that
  // silently validated nothing cannot pass on the exit code alone.
  size_t pos = out.find("D2H checks");
  REQUIRE(pos != std::string::npos);
  size_t colon = out.find(':', pos);
  REQUIRE(colon != std::string::npos);
  int d2h_pass = 0;
  sscanf(out.c_str() + colon + 1, " %d pass", &d2h_pass);
  INFO("D2H pass=" << d2h_pass);
  CHECK(d2h_pass >= 3);
}

// ---------------------------------------------------------------------------
// Helper: capture a workload and replay it with byte-exact D2H validation.
//
// Used by memset-D2D and driver-memcpy roundtrips.  hrr_run_roundtrip() cannot
// be used because it offers no way to pass playback environment variables.
// These workloads are bit-deterministic (pure fills/copies, no FP arithmetic),
// so the numeric-tolerance fallback in the D2H validator can only weaken the
// oracle: at the default atol=rtol=1e-3 it accepts any small float
// interpretation, including an all-zero buffer against a small-magnitude
// expectation.  HIP_HRR_D2H_EXACT=1 makes any byte difference a failure.
// ---------------------------------------------------------------------------
static void hrr_run_exact_roundtrip(const std::string& direct_case,
                                    const fs::path& cap_path) {
  hrr_capture_direct(direct_case, cap_path);

  auto [ret, out] = hrr_playback_env(cap_path, {{"HIP_HRR_D2H_EXACT", "1"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);

  // A handler returning anything other than hipSuccess aborts the replay pass
  // before the summary block is printed, so the presence of the summary line is
  // a platform-independent proof that the replay ran to completion.
  size_t pos = out.find("D2H checks");
  REQUIRE(pos != std::string::npos);
  size_t colon = out.find(':', pos);
  REQUIRE(colon != std::string::npos);
  // Summary format: "N pass (E exact, T within tol), F fail, S skipped".
  int d2h_pass = 0;
  REQUIRE(sscanf(out.c_str() + colon + 1, " %d pass", &d2h_pass) == 1);
  INFO("D2H pass=" << d2h_pass);

#ifdef _WIN32
  // Same policy as hrr_run_playback(): on the Windows CI target replay is not
  // guaranteed to reproduce device output bit-for-bit, so D2H fidelity is
  // best-effort there.  The completion check above still applies, and a crash
  // still fails via this bound.
  REQUIRE(ret < 128);
#else
  // hrr-playback exits non-zero when any D2H validation fails and also when
  // every D2H event was skipped, so this is the load-bearing fidelity
  // assertion; the pass count rules out the one remaining vacuous case, an
  // archive that carried no D2H blob at all.
  REQUIRE(ret == 0);
  REQUIRE(d2h_pass >= 1);
#endif
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_MemsetD2D_Direct: hipMemsetD2D8 / hipMemsetD2D8Async /
 *     hipMemsetD2D16 / hipMemsetD2D16Async / hipMemsetD2D32 /
 *     hipMemsetD2D32Async over a sub-region of a buffer allocated with a real
 *     row stride, on top of a whole-buffer sentinel.
 *   - Replay with byte-exact D2H: the fills must reproduce the pattern in the
 *     written sub-region AND leave the inter-row padding at the sentinel, so
 *     neither a skipped fill nor a fill that ignores the pitch can pass.
 */
HIP_TEST_CASE(Unit_HRR_MemsetD2DRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetd2d"};
  hrr_run_exact_roundtrip("Unit_HRR_MemsetD2D_Direct", cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_MemsetD2DPitchAlloc_Direct, which aims all six
 *     hipMemsetD2D* variants at a hipMemAllocPitch destination.  That API is a
 *     playback no-op, so the recorded destination has no alloc_map entry at
 *     replay.
 *   - Replay must warn and skip those calls, not hand a null destination to the
 *     real API: a non-success handler return is fatal and would abort the whole
 *     replay.  The archive's other (translatable) D2H blob is only reached and
 *     validated if the replay survived, which is what this asserts.
 */
HIP_TEST_CASE(Unit_HRR_MemsetD2DPitchAllocRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetd2dpitchalloc"};
  hrr_run_exact_roundtrip("Unit_HRR_MemsetD2DPitchAlloc_Direct", cap.path);
}

/**
 * Test Description
 * ----------------
 *   - Capture Unit_HRR_MemsetSpt_Direct: hipMemset_spt / hipMemsetAsync_spt /
 *     hipMemset2D_spt / hipMemset2DAsync_spt each fill their own buffer with
 *     their own byte pattern, and each buffer is read back by its own D2H.
 *     The workload calls the ordinary hipMemset* names and is compiled with
 *     -fgpu-default-stream=per-thread, so the archive is what proves the _spt
 *     entry points were the ones reached; assert the recorded API ids before
 *     replaying.
 *   - Replay with HIP_HRR_D2H_EXACT=1 and REQUIRE exit 0.  Exact mode matters:
 *     the default oracle falls back to float tolerance (atol=rtol=1e-3) and
 *     accepts a zero-initialised replay buffer whenever the captured pattern
 *     decodes to a small magnitude, which would let a NOOP _spt memset handler
 *     pass.  The workload also picks patterns that are out of tolerance in every
 *     candidate encoding, so exact mode is belt and braces, not the only guard.
 *   - REQUIRE at least 4 validated D2H buffers, one per API under test: a NOOP
 *     playback handler for any single _spt memset fails its own buffer and turns
 *     the playback exit code into 1.
 */
HIP_TEST_CASE(Unit_HRR_MemsetSptRoundtrip) {
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
    // The readbacks must stay on the plain hipMemcpy: hipMemcpy_spt records no
    // data blob, so a redirected readback would silently drop the D2H oracle
    // the exit-code and pass-count checks below depend on.
    REQUIRE(recorded(HRR_API_HIPMEMCPY));
  }

  auto [ret, out] = hrr_playback_env(cap.path, {{"HIP_HRR_D2H_EXACT", "1"}});
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
#ifdef _WIN32
  // Device-output fidelity is best-effort on the Windows CI target (same policy
  // as hrr_run_playback), so only a crash fails the test there.
  REQUIRE(ret < 128);
#else
  REQUIRE(ret == 0);  // any byte mismatch in exact mode exits 1

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

HIP_TEST_CASE(Unit_HRR_MemsetVariantsRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetvariants"};
  hrr_run_roundtrip("Unit_HRR_MemsetVariants_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_DeviceInfoRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_deviceinfo"};
  hrr_run_roundtrip("Unit_HRR_DeviceInfo_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_MetadataManifest) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_metadata_manifest"};

  {
    hip::SpawnProc proc(HRR_TEST_EXE);
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

  CHECK(ordinal >= 0);
  CHECK_FALSE(json_object_value(metadata, "properties").empty());
  CHECK_FALSE(name.empty());
  CHECK_FALSE(gcn_arch_name.empty());
  CHECK(total_global_mem > 0);
  CHECK(multi_processor_count > 0);
  CHECK(compute_mode >= 0);
  CHECK_FALSE(compute_capability.empty());
  CHECK(std::count(compute_capability.begin(), compute_capability.end(), '.') == 1);
  CHECK_FALSE(pci.empty());
  CHECK(std::count(pci.begin(), pci.end(), ':') == 2);
  CHECK_FALSE(uuid.empty());
}

HIP_TEST_CASE(Unit_HRR_StreamAdvancedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamadvanced"};
  hrr_run_roundtrip("Unit_HRR_StreamAdvanced_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_DrvMemcpyRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy"};
  hrr_run_roundtrip("Unit_HRR_DrvMemcpy_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_OccupancyRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_occupancy"};
  { hip::SpawnProc proc(HRR_TEST_EXE);
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
  // KNOWN LIMITATION (Linux): fat-binary code objects are not captured at static
  // init time on Linux, so kernel replay is a no-op and D2H bytes will not match.
  // On Linux we only assert that capture produced events and playback did not crash
  // (exit < 128).  D2H correctness is NOT verified on Linux for this workload.
  // This is a regression risk: a Linux-only kernel replay bug would not be caught
  // here.  Tracked as a known gap — fix requires resolving fat-binary static-init
  // capture timing on Linux.
#ifdef _WIN32
  hrr_run_playback(cap.path);
#else
  hrr_run_playback(cap.path, /*extra_args=*/"", /*require_d2h=*/false);
#endif
}

HIP_TEST_CASE(Unit_HRR_HostAliasesRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostaliases"};
  hrr_run_roundtrip("Unit_HRR_HostAliases_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_MemPoolExtendedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_mempoolext"};
  hrr_run_roundtrip("Unit_HRR_MemPoolExtended_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_DeviceMemPoolRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_devicemempool"};
  // Exercises hipDeviceSetMemPool (device stream-ordered pool association),
  // the one device mem-pool API left untested by Unit_HRR_DeviceInfo_Direct.
  hrr_run_roundtrip("Unit_HRR_DeviceMemPool_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ExtMallocRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_extmalloc"};
  // Exercises hipExtMallocWithFlags (device allocation, manual playback handler).
  hrr_run_roundtrip("Unit_HRR_ExtMalloc_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_StreamWaitEventSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamwaitspt"};
  // Exercises hipStreamWaitEvent_spt (per-thread-stream cross-stream ordering).
  hrr_run_roundtrip("Unit_HRR_StreamWaitEventSpt_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_GraphLaunchSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_graphlaunchspt"};
  // Exercises hipGraphLaunch_spt (per-thread-stream graph launch).
  hrr_run_roundtrip("Unit_HRR_GraphLaunchSpt_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ExtModuleLaunchKernelRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_extmodulelaunch"};
  // Exercises hipExtModuleLaunchKernel (manual kernarg + device-ptr replay via
  // the HIPRTC module path, which is captured/replayed on Linux).
  hrr_run_roundtrip("Unit_HRR_ExtModuleLaunchKernel_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_HostFreeRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostfree"};
  // Exercises hipHostFree + hipHostMalloc (pinned-host alloc-map lifecycle).
  hrr_run_roundtrip("Unit_HRR_HostFree_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_LoggingRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_logging"};
  // Exercises hipExtSetLoggingParams / hipExtEnableLogging / hipExtDisableLogging.
  hrr_run_roundtrip("Unit_HRR_Logging_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_StreamCaptureQuerySptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_capturequeryspt"};
  // Exercises hipStreamIsCapturing_spt + hipStreamGetCaptureInfo_spt inside a
  // manual capture frame (graph executes -> D2H validates the whole path).
  hrr_run_roundtrip("Unit_HRR_StreamCaptureQuerySpt_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_StreamCaptureBeginSptRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_capturebeginspt"};
  // Validates hipStreamBeginCapture_spt on GPU (generated handler omits the
  // ctx.in_graph_capture bookkeeping; safe for a memset-only capture region).
  hrr_run_roundtrip("Unit_HRR_StreamCaptureBeginSpt_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ConfigureCallRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_configurecall"};
  // Exercises hipConfigureCall (legacy execution-stack launch configuration).
  hrr_run_roundtrip("Unit_HRR_ConfigureCall_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_MemcpyPeerRoundtrip) {
  int src_dev = 0;
  int dst_dev = 1;
  int ndev = 0;
  if (!hrr_find_peer_accessible_pair(src_dev, dst_dev, ndev)) {
    if (ndev < 2) {
      HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    } else {
      HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
    }
  }
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memcpypeer"};
  // Exercises hipMemcpyPeer across two GPUs: capture on a multi-GPU host, replay
  // must recreate both allocations on their devices and validate the D2H bytes.
  hrr_run_roundtrip("Unit_HRR_MemcpyPeer_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_MemsetExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetextra"};
  hrr_run_roundtrip("Unit_HRR_MemsetExtra_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_MemcpyExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memcpyextra"};
  hrr_run_roundtrip("Unit_HRR_MemcpyExtra_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_DeviceExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_deviceextra"};
  hrr_run_roundtrip("Unit_HRR_DeviceExtra_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_StreamAdvanced2Roundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_streamadv2"};
  hrr_run_roundtrip("Unit_HRR_StreamAdvanced2_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ContextRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_context"};
  hrr_run_roundtrip("Unit_HRR_Context_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ModuleExtraRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_moduleextra"};
  hrr_run_roundtrip("Unit_HRR_ModuleExtra_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_MiscAPIsRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_miscapis"};
  hrr_run_roundtrip("Unit_HRR_MiscAPIs_Direct", cap.path);
}

// Driver-memcpy roundtrips (hipDrvMemcpy3D / 3DAsync / 2DUnaligned).
// Pure copy chains with no floating-point arithmetic; byte-exact D2H via
// hrr_run_exact_roundtrip() is the correct oracle (see helper above).
HIP_TEST_CASE(Unit_HRR_DrvMemcpy3DRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy3d"};
  hrr_run_exact_roundtrip("Unit_HRR_DrvMemcpy3D_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_DrvMemcpy2DUnalignedRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy2dunaligned"};
  hrr_run_exact_roundtrip("Unit_HRR_DrvMemcpy2DUnaligned_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_TextureRoundtrip) {
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

HIP_TEST_CASE(Unit_HRR_GraphExplicitRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_graphexplicit"};
  hrr_run_roundtrip("Unit_HRR_GraphExplicit_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_HostRegLaunchRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_hostreglch"};
  hrr_run_roundtrip("Unit_HRR_HostRegLaunch_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ModuleAPIRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_moduleapi"};
  hrr_run_roundtrip("Unit_HRR_ModuleAPI_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_VMMRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_vmm"};
  hrr_run_roundtrip("Unit_HRR_VMM_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_ChevronLaunchRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_chevron"};
  // Exercises __hipPushCallConfiguration → hipLaunchByPtr path.
  // min_events=5: malloc + stream + push + launch + D2H.
  hrr_run_roundtrip("Unit_HRR_ChevronLaunch_Direct", cap.path, 5);
}

/**
 * Test Description
 * ----------------
 *   - Spawns hip_raw_trace.exe as a subprocess with HIP_HRR_CAPTURE_OUTPUT set.
 *     hip_raw_trace is a genuine multi-threaded workload: 4 threadFunc threads
 *     (each with 2 streams, H2D, vectorAdd×64, D2H), plus graphFunc, pinnedFunc,
 *     and hostRegisterFunc, all running concurrently.  Captured events from all 7
 *     threads are interleaved in a single events.bin.
 *   - Replays the archive single-threaded to establish a D2H baseline.
 *   - Replays again with --multi-thread to exercise the MT dispatch path
 *     (spin-wait ordering, atomic next_seq, per-thread timing events).
 *   - Skips if HRR_RAW_TRACE_EXE was not found at configure time.
 */
HIP_TEST_CASE(Unit_HRR_MultiThreadRoundtrip) {
  static constexpr const char* raw_trace_exe = HRR_RAW_TRACE_EXE;
  if (!raw_trace_exe || raw_trace_exe[0] == '\0') {
    SUCCEED("hip_raw_trace not found at configure time — skipping "
            "(pass -DHRR_RAW_TRACE_EXE=<path> to cmake)");
    return;
  }

  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_multithread"};

  // -------------------------------------------------------------------------
  // Step 1: capture with hip_raw_trace (multi-threaded workload)
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(raw_trace_exe);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    {
      set_proc_search_path(proc);
    }
    int ret = proc.run("");
    INFO("hip_raw_trace capture exit code: " << ret);
    REQUIRE(ret == 0);
  }

  // -------------------------------------------------------------------------
  // Step 2: verify archive
  // -------------------------------------------------------------------------
  fs::path archive_path = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(archive_path / "events.bin"));
  REQUIRE(fs::exists(archive_path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(archive_path / "blobs")) ++bc;
  INFO("Blob count: " << bc);
  REQUIRE(bc >= 1);

  // -------------------------------------------------------------------------
  // Step 3 & 4: playback (single-thread then multi-thread)
  //
  // The MT workload uses fat-binary kernels (hipLaunchByPtr path).  Those are
  // only replayable when the capture DLL is present, which is not guaranteed in
  // CI.  We therefore only assert that hrr-playback starts, reads the archive,
  // and exits without crashing (exit code is not checked here).  D2H correctness
  // is fully validated by the other single-thread roundtrip tests.
  //
  // --skip-device-sync: the graph-capture stream cannot be synchronised during
  // replay (hipStreamSynchronize returns error 900 on an open capture stream).
  // -------------------------------------------------------------------------
  auto run_mt_playback = [&](const std::string& extra_args) {
    hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
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
    // Exit code: fat-binary kernel replay requires the capture DLL to be present,
    // so a D2H mismatch (exit 1) is accepted in CI.  A crash (signal, exit >= 128)
    // is always a hard failure.
    REQUIRE(ret < 128);
    // Must print the archive header line — confirms hrr-playback read events.bin.
    REQUIRE(out.find("[HRR] Archive") != std::string::npos);
  };

  run_mt_playback("");               // single-thread
  run_mt_playback("--multi-thread"); // MT dispatch path
}

// ---------------------------------------------------------------------------
// --replace-kernel tests
//
// The capture layer records each kernel by its device symbol name (mangled C++
// name, e.g. _Z14hrr_vectorAddPKfS0_Pfi). hrr-playback's --replace-kernel N=PATH
// matches N as a substring of the recorded name and, on a match, loads PATH
// (.hsaco) and resolves the SAME recorded symbol from it, substituting the
// function at launch time. These tests exercise:
//   1. Happy path: a functionally identical replacement CO (same hrr_vectorAdd)
//      is loaded and used (stdout reports "Replacing kernel"), and because the
//      replacement is byte-identical in behaviour the D2H validation still
//      passes (exit 0).
//   2. Missing CO: a non-existent path matches the pattern but cannot be read,
//      so playback falls back to the recorded kernel — no crash, D2H still
//      passes (exit 0), and the fallback is reported on stderr/stdout.
//   3. Bad spec: a NAME=PATH argument with no '=' is rejected by the parser
//      (exit 1) before any GPU work.
//
// Note: hrr-playback validates D2H against the *recorded* blobs, so it cannot
// by construction validate that a *behaviourally different* replacement produced
// the replacement's output (there is no recorded ground truth for that). The
// happy-path test therefore uses an identical kernel and asserts the replacement
// code path ran via the "Replacing kernel" log line.
// ---------------------------------------------------------------------------

#ifndef _WIN32  // hiprtc replacement CO build + load is exercised on Linux CI
#define HRR_HIPRTC_CHECK(expr)                                                  \
  do {                                                                          \
    hiprtcResult _r = (expr);                                                   \
    if (_r != HIPRTC_SUCCESS) {                                                 \
      FAIL("HIPRTC error " << static_cast<int>(_r) << " at " #expr);            \
    }                                                                           \
  } while (0)

// Compile a code object that exports a kernel byte-identical to the workload's
// hrr_vectorAdd (so the mangled symbol matches the recorded name) and write it
// to <dir>/replacement.hsaco. Returns the path.
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

// Spawn hrr-playback with arbitrary extra args; return {exit_code, stdout}.
// Return the full recorded name of the first kernel-launch event whose name
// contains `needle`, or "" if none. --replace-kernel matches the recorded name
// EXACTLY, and C++/chevron kernels are recorded under their mangled symbol
// (e.g. _Z14hrr_vectorAddPKfS0_Pfi), so a test cannot hard-code the bare source
// name — it must look up the exact recorded string from the archive.
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

/**
 * Test Description
 * ----------------
 *   - Capture the GPU workload, build a functionally identical replacement code
 *     object for hrr_vectorAdd via HIPRTC, and replay with --replace-kernel
 *     using the EXACT recorded (mangled) kernel name looked up from the archive.
 *   - The replacement symbol matches the recorded name, so playback loads it and
 *     reports "Replacing kernel". Because behaviour is identical, the D2H
 *     validation still passes (exit 0).
 */
HIP_TEST_CASE(Unit_HRR_ReplaceKernelRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_replace_happy"};
  ScopedDir co_dir{fs::temp_directory_path() / "hrr_replace_co"};

  {
    hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.path.string());
    set_proc_search_path(proc);
    int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    INFO("Capture exit: " << ret);
    REQUIRE(ret == 0);
  }
  REQUIRE(fs::exists(hrr_single_process_archive(cap.path) / "events.bin"));

  // --replace-kernel matches the recorded name exactly; resolve the mangled
  // symbol the capture stored for hrr_vectorAdd.
  std::string kname = recorded_kernel_name(cap.path, "hrr_vectorAdd");
  INFO("Recorded kernel name: " << kname);
  REQUIRE_FALSE(kname.empty());

  std::string co_path = build_replacement_co(co_dir.path);
  REQUIRE(fs::exists(co_path));

  auto [ret, out] = run_playback_raw(cap.path, "--replace-kernel " + kname + "=" + co_path);
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  // Identical replacement → D2H still matches the recorded output.
  REQUIRE(ret == 0);
  // Confirm the replacement code path actually ran (not a silent fallback).
  CHECK(out.find("Replacing kernel") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - Replay a captured workload with --replace-kernel naming the exact recorded
 *     kernel but pointing at a path that does not exist. The name matches, but
 *     the file cannot be read, so playback must gracefully fall back to the
 *     recorded kernel: no crash, D2H still passes (exit 0), and a fallback
 *     message is emitted.
 */
HIP_TEST_CASE(Unit_HRR_ReplaceKernelMissingCO) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_replace_missing"};

  {
    hip::SpawnProc proc(HRR_TEST_EXE);
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
  // Graceful fallback to the recorded kernel — must not crash and D2H must pass.
  REQUIRE(ret == 0);
}

/**
 * Test Description
 * ----------------
 *   - A --replace-kernel argument that is not in NAME=PATH form (no '=') must be
 *     rejected by the argument parser with a non-zero exit before any GPU work.
 */
HIP_TEST_CASE(Unit_HRR_ReplaceKernelBadSpec) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_replace_badspec"};

  {
    hip::SpawnProc proc(HRR_TEST_EXE);
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
