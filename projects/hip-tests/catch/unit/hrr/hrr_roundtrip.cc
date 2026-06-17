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
#include "hrr_reader.h"
#include "hrr_api_args.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Platform path separator for setEnv("PATH", ...).
// ';' on Windows, ':' on POSIX.
#ifdef _WIN32
static constexpr char kPathSep = ';';
#else
static constexpr char kPathSep = ':';
#endif

// Set PATH so the subprocess can find the ROCm runtime binaries.
// On Windows: DLLs are found via PATH.
// On Linux:   fork() inherits LD_LIBRARY_PATH from the parent automatically;
//             no explicit setEnv needed.
static void set_proc_search_path(hip::SpawnProc& proc) {
  const char* cur_path = getenv("PATH");
  proc.setEnv("PATH",
              std::string(ROCM_BIN_PATH) + kPathSep + (cur_path ? cur_path : ""));
}

// RAII guard: removes a directory tree on scope exit (even on REQUIRE failure).
struct ScopedDir {
  fs::path path;
  explicit ScopedDir(fs::path p) : path(std::move(p)) { fs::remove_all(path); }
  ~ScopedDir() { fs::remove_all(path); }
};

// ---------------------------------------------------------------------------
// hrr_run_playback — spawn hrr-playback, capture stdout, assert:
//   1. Exit code == 0.
//   2. The "D2H checks" summary line is present and shows >= 1 pass, 0 fail.
//
// The D2H line format (from hrr_playback.cpp):
//   "[HRR]   D2H checks     : N pass, M fail, K skipped"
//
// If require_d2h == true (default) we REQUIRE pass >= 1.
// Workloads with no D2H memcpy (e.g. DeviceInfo, Occupancy) pass require_d2h=false.
// ---------------------------------------------------------------------------
static void hrr_run_playback(const fs::path& cap_path,
                             const std::string& extra_args = "",
                             bool require_d2h = true) {
  hip::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(proc);
  // On Windows, wrap the path in quotes so CreateProcess handles spaces.
  // On Linux, SpawnProc uses execvp (no shell), so quotes are literal characters
  // in the argument — pass the raw path without quoting.
#ifdef _WIN32
  std::string path_arg = "\"" + cap_path.string() + "\"";
#else
  std::string path_arg = cap_path.string();
#endif
  int ret = proc.run(path_arg + (extra_args.empty() ? "" : " " + extra_args));
  std::string out = proc.getOutput();
  INFO("Playback stdout:\n" << out);
  INFO("Playback exit code: " << ret);
  // When require_d2h is false (e.g. no D2H in workload, or Linux fat-binary
  // limitation) we only assert that hrr-playback did not crash (signal).
  // A non-zero exit due to D2H mismatch is accepted.
  if (require_d2h) {
    REQUIRE(ret == 0);
  } else {
    // Treat SIGSEGV/SIGBUS (>128) as hard failure; clean exit or D2H-fail (1) is ok.
    REQUIRE(ret < 128);
    if (ret != 0) return;  // D2H mismatch expected — skip summary parse
  }

  // Parse the D2H summary line.
  size_t pos = out.find("D2H checks");
  if (pos == std::string::npos) {
    // hrr-playback didn't print a summary — treat as failure.
    FAIL("hrr-playback output missing 'D2H checks' summary line");
  }
  size_t colon = out.find(':', pos);
  if (colon == std::string::npos) FAIL("D2H checks line missing ':'");
  std::string rest = out.substr(colon + 1);
  int d2h_pass = 0, d2h_fail = 0;
  // Format: " N pass, M fail, K skipped"
  sscanf(rest.c_str(), " %d pass, %d fail", &d2h_pass, &d2h_fail);
  INFO("D2H pass=" << d2h_pass << " fail=" << d2h_fail);
  if (require_d2h) {
    CHECK(d2h_pass >= 1);
    CHECK(d2h_fail == 0);
  }
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs"))
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs"))
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs"))
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs"))
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));

  int blob_count = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs"))
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

// ---------------------------------------------------------------------------
// Helper: shared roundtrip body — capture → verify archive → playback.
//
// min_events:  minimum number of events expected in events.bin.  Every workload
//   must produce at least a few events (malloc, memcpy, kernel, free) — a value
//   of 5 is a conservative floor that would catch a totally empty capture.
//   Use a higher value for workloads known to emit many events (StressApis, etc.).
// require_d2h: if true (default), asserts that playback validated at least one
//   D2H blob.  Pass false for workloads that conditionally skip D2H (e.g. the
//   texture workload on devices without image support).
// ---------------------------------------------------------------------------
static void hrr_run_roundtrip(const std::string& direct_case,
                               const fs::path& cap_path,
                               size_t min_events = 5,
                               bool require_d2h = true) {
  { hip::SpawnProc proc(HRR_TEST_EXE);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap_path.string());
    { set_proc_search_path(proc); }
    int ret = proc.run("\"" + direct_case + "\"");
    INFO("Capture exit: " << ret); REQUIRE(ret == 0); }
  REQUIRE(fs::exists(cap_path / "events.bin"));
  REQUIRE(fs::exists(cap_path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap_path / "blobs")) ++bc;
  INFO("Blob count: " << bc); REQUIRE(bc >= 1);

  // Load the archive and assert a minimum event count.  This catches generator
  // bugs that silently produce empty or near-empty archives while still writing
  // at least one blob (which would otherwise satisfy the blob_count >= 1 check).
  hrr::Archive arc;
  bool arc_ok = hrr::load_archive(cap_path.string(), arc);
  INFO("Archive event count: " << arc.events.size());
  REQUIRE(arc_ok);
  REQUIRE(arc.events.size() >= min_events);

  hrr_run_playback(cap_path, /*extra_args=*/"", require_d2h);
}

HIP_TEST_CASE(Unit_HRR_MemsetVariantsRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_memsetvariants"};
  hrr_run_roundtrip("Unit_HRR_MemsetVariants_Direct", cap.path);
}

HIP_TEST_CASE(Unit_HRR_DeviceInfoRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_deviceinfo"};
  hrr_run_roundtrip("Unit_HRR_DeviceInfo_Direct", cap.path);
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs")) ++bc;
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

HIP_TEST_CASE(Unit_HRR_DrvMemcpy3DRoundtrip) {
  ScopedDir cap{fs::temp_directory_path() / "hrr_roundtrip_drvmemcpy3d"};
  hrr_run_roundtrip("Unit_HRR_DrvMemcpy3D_Direct", cap.path);
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
  REQUIRE(fs::exists(cap.path / "events.bin"));
  REQUIRE(fs::exists(cap.path / "blobs"));
  int bc = 0;
  for ([[maybe_unused]] const auto& _ :
       fs::recursive_directory_iterator(cap.path / "blobs")) ++bc;
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
