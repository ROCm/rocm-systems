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

#include <filesystem>

namespace fs = std::filesystem;

// RAII guard: removes a directory tree on scope exit (even on REQUIRE failure).
struct ScopedDir {
  fs::path path;
  explicit ScopedDir(fs::path p) : path(std::move(p)) { fs::remove_all(path); }
  ~ScopedDir() { fs::remove_all(path); }
};


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
      const char* cur = getenv("PATH");
      proc.setEnv("PATH", std::string(ROCM_BIN_PATH) + ";" + (cur ? cur : ""));
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

  // -------------------------------------------------------------------------
  // Step 3: playback + D2H validation
  //   hrr-playback replays every event; for each D2H memcpy it copies the
  //   replayed host buffer into a staging allocation and compares against the
  //   stored blob.  Any mismatch → exit 1.
  // -------------------------------------------------------------------------
  {
    hip::SpawnProc proc(HRR_PLAYBACK_EXE);
    int ret = proc.run("\"" + cap.path.string() + "\"");
    INFO("Playback subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }
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
      const char* cur = getenv("PATH");
      proc.setEnv("PATH", std::string(ROCM_BIN_PATH) + ";" + (cur ? cur : ""));
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
  {
    hip::SpawnProc proc(HRR_PLAYBACK_EXE);
    int ret = proc.run("\"" + cap.path.string() + "\"");
    INFO("Graph playback subprocess exit code: " << ret);
    REQUIRE(ret == 0);
  }
}
