/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* Every thunk call here is injected, so this runs with no GPU, no DXCore
 * adapter and no open thunk. That seam is also the limit of what these cases
 * see: KfdDriver::ThunkOps() decides which thunk call lands in which
 * KfdLifecycleOps field, and swapping release_snapshot with close would release
 * a snapshot after the close that invalidates it without failing anything
 * below.
 *
 * That wiring is left to review rather than tested. ThunkOps() is private to
 * KfdDriver, HSAKMT_CALL() resolves through
 * core::Runtime::runtime_singleton_->thunkLoader() whose constructor and
 * thunkLoader_ are both private, and amd_kfd_driver.cpp pulls in the GPU agent,
 * the memory regions and the code-object loader. That is the whole runtime
 * linked in to check three assignments.
 */

#include <string>
#include <vector>

#include "core/inc/amd_kfd_lifecycle.h"
#include "unit_test_harness.h"

using rocr::AMD::KfdLifecycle;
using rocr::AMD::KfdLifecycleOps;

namespace {

struct Recorder {
  std::vector<std::string> calls;
  hsa_status_t disable_status = HSA_STATUS_SUCCESS;
  hsa_status_t release_status = HSA_STATUS_SUCCESS;
  hsa_status_t close_status = HSA_STATUS_SUCCESS;

  /* Assigned by name, the way KfdDriver::ThunkOps() does it, so that reordering
   * the fields of KfdLifecycleOps cannot silently relabel these lambdas and
   * leave the cases below asserting the wrong sequence.
   */
  KfdLifecycleOps Ops() {
    KfdLifecycleOps ops;
    ops.disable_runtime = [this]() {
      calls.emplace_back("disable");
      return disable_status;
    };
    ops.release_snapshot = [this]() {
      calls.emplace_back("release");
      return release_status;
    };
    ops.close = [this]() {
      calls.emplace_back("close");
      return close_status;
    };
    return ops;
  }
};

void AcquireAll(KfdLifecycle& lifecycle) {
  lifecycle.Open([]() { return HSA_STATUS_SUCCESS; });
  lifecycle.AcquireSnapshot([]() { return HSA_STATUS_SUCCESS; });
  lifecycle.EnableRuntime([]() { return HSA_STATUS_SUCCESS; });
}

}  // namespace

TEST_CASE(each_failed_acquire_stage_releases_only_what_preceded_it) {
  {
    Recorder recorder;
    KfdLifecycle lifecycle(recorder.Ops());
    CHECK_EQ(lifecycle.Open([]() { return HSA_STATUS_ERROR; }), HSA_STATUS_ERROR);
    CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
    CHECK(recorder.calls.empty());
  }

  {
    Recorder recorder;
    KfdLifecycle lifecycle(recorder.Ops());
    CHECK_EQ(lifecycle.Open([]() { return HSA_STATUS_SUCCESS; }), HSA_STATUS_SUCCESS);
    CHECK_EQ(lifecycle.AcquireSnapshot([]() { return HSA_STATUS_ERROR; }), HSA_STATUS_ERROR);
    CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
    CHECK_EQ(recorder.calls, (std::vector<std::string>{"close"}));
  }

  {
    Recorder recorder;
    KfdLifecycle lifecycle(recorder.Ops());
    CHECK_EQ(lifecycle.Open([]() { return HSA_STATUS_SUCCESS; }), HSA_STATUS_SUCCESS);
    CHECK_EQ(lifecycle.AcquireSnapshot([]() { return HSA_STATUS_SUCCESS; }), HSA_STATUS_SUCCESS);
    CHECK_EQ(lifecycle.EnableRuntime([]() { return HSA_STATUS_ERROR; }), HSA_STATUS_ERROR);
    CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
    CHECK_EQ(recorder.calls, (std::vector<std::string>{"release", "close"}));
  }
}

TEST_CASE(shutdown_continues_after_errors_and_reports_the_first_one) {
  Recorder recorder;
  recorder.disable_status = HSA_STATUS_ERROR;
  recorder.release_status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  recorder.close_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
  KfdLifecycle lifecycle(recorder.Ops());
  AcquireAll(lifecycle);

  CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_ERROR);
  CHECK_EQ(recorder.calls, (std::vector<std::string>{"disable", "release", "close"}));
}

TEST_CASE(shutdown_is_idempotent) {
  Recorder recorder;
  KfdLifecycle lifecycle(recorder.Ops());
  AcquireAll(lifecycle);

  CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
  const auto first_shutdown = recorder.calls;
  CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
  CHECK_EQ(recorder.calls, first_shutdown);
}

TEST_CASE(close_gives_back_only_the_open_reference) {
  Recorder recorder;
  KfdLifecycle lifecycle(recorder.Ops());
  AcquireAll(lifecycle);

  /* Twice, because a second close must find nothing left to give back rather
   * than hand the thunk a reference it no longer holds.
   */
  CHECK_EQ(lifecycle.Close(), HSA_STATUS_SUCCESS);
  CHECK_EQ(lifecycle.Close(), HSA_STATUS_SUCCESS);
  CHECK_EQ(recorder.calls, (std::vector<std::string>{"close"}));

  /* The runtime enable and the snapshot were never Close()'s to release, so
   * the broad teardown still gives those back afterwards.
   */
  CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
  CHECK_EQ(recorder.calls, (std::vector<std::string>{"close", "disable", "release"}));
}

TEST_CASE(a_forked_child_gives_back_nothing_it_inherited) {
  /* The pid provider is mutable, so the instance records one process at
   * construction and every later check reports another - what an inherited
   * KfdLifecycle sees in a forked child, without needing to fork.
   */
  {
    Recorder recorder;
    int pid = 4242;
    KfdLifecycleOps ops = recorder.Ops();
    ops.get_pid = [&pid]() { return pid; };
    KfdLifecycle lifecycle(std::move(ops));
    AcquireAll(lifecycle);

    pid = 4243;

    CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
    CHECK(recorder.calls.empty());
  }

  /* Close() carries the same guard and has to be checked on its own, because
   * ShutDown() returns above before it would reach it.
   */
  {
    Recorder recorder;
    int pid = 4242;
    KfdLifecycleOps ops = recorder.Ops();
    ops.get_pid = [&pid]() { return pid; };
    KfdLifecycle lifecycle(std::move(ops));
    AcquireAll(lifecycle);

    pid = 4243;

    CHECK_EQ(lifecycle.Close(), HSA_STATUS_SUCCESS);
    CHECK(recorder.calls.empty());
  }

  /* Same process, so the ordinary unwind has to be exactly as it was. */
  {
    Recorder recorder;
    KfdLifecycleOps ops = recorder.Ops();
    ops.get_pid = []() { return 4242; };
    KfdLifecycle lifecycle(std::move(ops));
    AcquireAll(lifecycle);

    CHECK_EQ(lifecycle.ShutDown(), HSA_STATUS_SUCCESS);
    CHECK_EQ(recorder.calls, (std::vector<std::string>{"disable", "release", "close"}));
  }
}

TEST_CASE(open_and_snapshot_acquisition_are_each_idempotent) {
  Recorder recorder;
  KfdLifecycle lifecycle(recorder.Ops());
  int open_calls = 0;
  int snapshot_calls = 0;

  auto open = [&]() {
    ++open_calls;
    return HSA_STATUS_SUCCESS;
  };
  auto acquire = [&]() {
    ++snapshot_calls;
    return HSA_STATUS_SUCCESS;
  };

  CHECK_EQ(lifecycle.Open(open), HSA_STATUS_SUCCESS);
  CHECK_EQ(lifecycle.Open(open), HSA_STATUS_SUCCESS);
  CHECK_EQ(lifecycle.AcquireSnapshot(acquire), HSA_STATUS_SUCCESS);
  CHECK_EQ(lifecycle.AcquireSnapshot(acquire), HSA_STATUS_SUCCESS);
  CHECK_EQ(open_calls, 1);
  CHECK_EQ(snapshot_calls, 1);
}

int main() { return unittest::RunAllTests(); }
