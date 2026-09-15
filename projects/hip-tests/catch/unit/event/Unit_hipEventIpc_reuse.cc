/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Test cross-process ordering through a re-recorded interprocess event.

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <utils.hh>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
// More records than the 32 signal slots an emulated IPC event cycles through.
constexpr int kRounds = 40;
// Set by the parent for the child: "<read fd>,<write fd>,<reopen>".
constexpr char kChildEnv[] = "HIP_IPC_EVENT_REUSE";
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - One interprocess event is recorded 40 times by the parent, each time behind a delay kernel
 *    and a memset that writes the round number into device memory shared with the child through
 *    hipIpcGetMemHandle. For every record the child waits on the imported event with
 *    hipStreamWaitEvent and then copies the shared value: with correct ordering it reads exactly
 *    the current round.
 *  - Two variants: the child opens the event handle once and keeps it for all records, or it
 *    opens and destroys the imported event in every round.
 *  - Guards the emulated (shared memory) IPC event path against slot overflow after 32 records
 *    and against the importer unlinking the shared memory name on destroy; runs unchanged on the
 *    ROCr IPC signal path.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventIpc_reuse.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipEventIpc_RecordReuse) {
  const bool reopen = GENERATE(false, true);
  INFO((reopen ? "child reopens the event handle every round" : "child opens the event handle once"));

  int to_child[2], to_parent[2];
  REQUIRE(pipe(to_child) == 0);
  REQUIRE(pipe(to_parent) == 0);
  // Only the child's ends survive the exec, so either side sees EOF if the other one dies.
  REQUIRE(fcntl(to_child[1], F_SETFD, FD_CLOEXEC) == 0);
  REQUIRE(fcntl(to_parent[0], F_SETFD, FD_CLOEXEC) == 0);

  int* value = nullptr;
  HIP_CHECK(hipMalloc(&value, sizeof(int)));
  HIP_CHECK(hipMemset(value, 0, sizeof(int)));
  hipEvent_t event;
  HIP_CHECK(hipEventCreateWithFlags(&event, hipEventInterprocess | hipEventDisableTiming));
  hipIpcEventHandle_t event_handle;
  HIP_CHECK(hipIpcGetEventHandle(&event_handle, event));
  hipIpcMemHandle_t mem_handle;
  HIP_CHECK(hipIpcGetMemHandle(&mem_handle, value));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hip::SpawnProc child(getSelfExePath());
  child.setEnv(kChildEnv, std::to_string(to_child[0]) + "," + std::to_string(to_parent[1]) + "," +
                              (reopen ? "1" : "0"));
  REQUIRE(child.spawn("Unit_hipEventIpc_RecordReuse_Child") == 0);
  REQUIRE(close(to_child[0]) == 0);
  REQUIRE(close(to_parent[1]) == 0);
  REQUIRE(hip::writeAll(to_child[1], &event_handle, sizeof(event_handle)));
  REQUIRE(hip::writeAll(to_child[1], &mem_handle, sizeof(mem_handle)));

  int ordered = 0;
  int first_unordered = 0;
  for (int round = 1; round <= kRounds; ++round) {
    // The value changes only after the delay kernel and the event is recorded behind it, so the
    // record is still pending when the child starts waiting on it.
    LaunchDelayKernel(std::chrono::milliseconds(50), stream);
    HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(value), round, 1, stream));
    HIP_CHECK(hipEventRecord(event, stream));
    REQUIRE(hip::writeAll(to_child[1], &round, sizeof(round)));
    char ok = 0;
    REQUIRE(hip::readAll(to_parent[0], &ok, sizeof(ok)));  // the child is done with this record
    ordered += ok;
    if (!ok && first_unordered == 0) first_unordered = round;
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  REQUIRE(close(to_child[1]) == 0);
  REQUIRE(close(to_parent[0]) == 0);
  REQUIRE(child.wait() == 0);

  INFO("first unordered round: " << first_unordered);
  REQUIRE(ordered == kRounds);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipFree(value));
}

/**
 * Test Description
 * ------------------------
 *  - Child process of Unit_hipEventIpc_RecordReuse: imports the event and the shared value,
 *    waits on the event for every record and reports whether it read the current round.
 *    Skipped when not launched by the parent test.
 * Test source
 * ------------------------
 *  - unit/event/Unit_hipEventIpc_reuse.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipEventIpc_RecordReuse_Child) {
  const char* env = std::getenv(kChildEnv);
  if (env == nullptr || env[0] == '\0') {
    HIP_SKIP_TEST("This test must be launched by Unit_hipEventIpc_RecordReuse.");
  }
  int rd = -1, wr = -1, reopen = 0;
  REQUIRE(sscanf(env, "%d,%d,%d", &rd, &wr, &reopen) == 3);

  hipIpcEventHandle_t event_handle;
  hipIpcMemHandle_t mem_handle;
  REQUIRE(hip::readAll(rd, &event_handle, sizeof(event_handle)));
  REQUIRE(hip::readAll(rd, &mem_handle, sizeof(mem_handle)));
  int* value = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void**>(&value), mem_handle,
                                hipIpcMemLazyEnablePeerAccess));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipEvent_t event = nullptr;
  for (int i = 0; i < kRounds; ++i) {
    int round = 0;
    REQUIRE(hip::readAll(rd, &round, sizeof(round)));
    // Errors are reported to the parent instead of failing here, so it sees every round.
    int seen = -1;
    hipError_t err = hipSuccess;
    if (event == nullptr) err = hipIpcOpenEventHandle(&event, event_handle);
    if (err == hipSuccess) err = hipStreamWaitEvent(stream, event, 0);
    if (err == hipSuccess) {
      err = hipMemcpyAsync(&seen, value, sizeof(seen), hipMemcpyDeviceToHost, stream);
    }
    if (err == hipSuccess) err = hipStreamSynchronize(stream);
    const char ok = (err == hipSuccess && seen == round);
    if (!ok) {
      fprintf(stderr, "round %d: %s, read %d\n", round, hipGetErrorString(err), seen);
    }
    if (event != nullptr && (reopen || err != hipSuccess)) {
      static_cast<void>(hipEventDestroy(event));
      event = nullptr;
    }
    REQUIRE(hip::writeAll(wr, &ok, sizeof(ok)));
  }

  if (event != nullptr) HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipIpcCloseMemHandle(value));
}
