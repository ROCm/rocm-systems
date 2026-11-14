/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Compiles inspector.cc for host unit tests, and everything needed to do so in
// isolation: stubs for the plugin's sibling translation units, plus the C
// forwarders InspectorPluginTests.cpp uses to reach the units under test.
//
// Why this TU exists at all. The plugin vendors its own nccl/ headers, and its
// nccl/profiler.h collides with RCCL's hipified profiler.h. Per-source
// COMPILE_FLAGS cannot resolve that: CMake emits them *after* the target's
// include directories, so RCCL's headers always win. Hence this file (and the
// stubs below) live in their own static library, rccl-inspector-under-test,
// whose include path is deliberately only the plugin's own directories -- see
// test/host/CMakeLists.txt.
//
// The same constraint is why InspectorPluginTests.cpp includes none of the
// plugin's headers and talks to the forwarders at the bottom of this file
// instead.

#include "inspector.cc"

#include <cstdint>

// ===========================================================================
// Stubs for the plugin's sibling TUs
// ===========================================================================
// These satisfy the externals inspector.cc reaches so the host tests link
// without the rest of the plugin, librccl, or a HIP runtime. Each is annotated
// with the production TU that defines the real symbol.
//
// They are fixed floors, not controllable seams: none of the units under test
// drive them. Give one a std::function hook when a test needs to steer it.
//
// Not included here: hipGetDevice/hipGetErrorString, which fakes/hip_fakes.cc
// already owns for this binary (defining them again is a duplicate-symbol link
// error), and the JSON writer itself -- json.cc is linked for real so the
// output-file creation path under test is production code end to end.

// --- inspector_prom.cc ---------------------------------------------------
inspectorResult_t inspectorPromCommInfoListDump(struct inspectorCommInfoList*,
                                                const char*,
                                                struct inspectorDumpThread*) {
  return inspectorSuccess;
}

int64_t inspectorPromValidateInterval(int64_t interval) { return interval; }

// --- inspector_otel.cc ---------------------------------------------------
inspectorResult_t inspectorOtelCommInfoListDump(struct inspectorCommInfoList*) {
  return inspectorSuccess;
}

void inspectorOtelInitFromEnv() {}

bool inspectorOtelIsEnabled() { return false; }

// --- inspector_json.cc ---------------------------------------------------
inspectorResult_t inspectorCommInfoListDump(jsonFileOutput*,
                                            struct inspectorCommInfoList*) {
  return inspectorSuccess;
}

// --- inspector_cudawrap.cc -----------------------------------------------
inspectorResult_t inspectorCudaWrapInit(void) { return inspectorSuccess; }

void inspectorCudaWrapCleanup(void) {}

PFN_cuGetErrorString pfn_cuGetErrorString = nullptr;
PFN_cuDeviceGet      pfn_cuDeviceGet      = nullptr;
PFN_cuDeviceGetUuid  pfn_cuDeviceGetUuid  = nullptr;

// --- inspector_event_pool.cc ---------------------------------------------
inspectorResult_t inspectorEventPoolInit(uint32_t, uint32_t, uint32_t) {
  return inspectorSuccess;
}

inspectorResult_t inspectorEventPoolFinalize() { return inspectorSuccess; }

// --- inspector_ring.cc ---------------------------------------------------
inspectorResult_t inspectorRingInit(struct inspectorCompletedRing*,
                                    uint32_t,
                                    size_t) {
  return inspectorSuccess;
}

void inspectorRingFinalize(struct inspectorCompletedRing*) {}

// --- version.cc ----------------------------------------------------------
const char* get_git_version_info() { return "host-unit-tests"; }

// ===========================================================================
// Forwarders for InspectorPluginTests.cpp
// ===========================================================================

namespace {

void NoOpLogger(ncclDebugLogLevel, unsigned long, const char*, int,
                const char*, ...) {}

}  // namespace

extern "C" {

// logFn is a plain global in inspector.cc and starts NULL; every INFO_INSPECTOR
// in the units under test would dereference it.
void test_inspectorInstallNoOpLogger(void) { logFn = NoOpLogger; }

// enableNcclInspectorPromDump is file-static in inspector.cc. The Prometheus
// tests need ~inspectorDumpThread to take its PROM_DUMP branch so the file
// handles they open are closed at teardown.
void test_inspectorSetPromDumpEnabled(int enabled) {
  enableNcclInspectorPromDump = (enabled != 0);
}

void* test_inspectorDumpThreadCreate(const char* outputRoot,
                                     int64_t sampleIntervalUsecs) {
  return new inspectorDumpThread(outputRoot, sampleIntervalUsecs);
}

void test_inspectorDumpThreadDestroy(void* dumper) {
  delete static_cast<inspectorDumpThread*>(dumper);
}

// Returns 1 if the plugin handed back an open stream, 0 if it refused.
int test_inspectorGetOrCreateFileHandle(void* dumper,
                                        const char* deviceUuidStr,
                                        const char* filename,
                                        uint64_t currentTime) {
  FILE* f = static_cast<inspectorDumpThread*>(dumper)->getOrCreateFileHandle(
      deviceUuidStr, filename, currentTime);
  return f != nullptr ? 1 : 0;
}

int test_inspectorStateDumpJSON(void* dumper, const char* outputRoot) {
  return static_cast<int>(
      static_cast<inspectorDumpThread*>(dumper)->inspectorStateDumpJSON(
          outputRoot));
}

// inspectorResult_t values, so the test can assert the specific code without
// including the plugin's headers.
int test_inspectorResultSuccess(void) {
  return static_cast<int>(inspectorSuccess);
}

int test_inspectorResultFileOpenError(void) {
  return static_cast<int>(inspectorFileOpenError);
}

}  // extern "C"
