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
#include "cross_process_serialization.h"

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"

// How long the holder process holds the mutex (seconds).
static constexpr unsigned int kHoldSeconds = 5;
// Minimum elapsed time (seconds) the waiter must observe to prove it blocked.
// Set below kHoldSeconds to absorb scheduling jitter.
static constexpr double kMinWaitSeconds = 3.0;
// Nanoseconds the waiter pauses after receiving the run_pipe_ signal before
// calling amdsmi_get_gpu_id. rsmi_test_sleep's mutex acquisition is
// sub-microsecond; 200ms is a 200,000x margin.
static constexpr long kWaiterMutexPauseNs = 200000000L;  // 200 ms

extern amdsmi_status_t rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds);

TestCrossProcessSerialization::TestCrossProcessSerialization() : TestBase() {
  set_title("Cross-Process Serialization Test");
  set_description(
      "Verifies that AMDSMI_MUTEX_CROSS_PROCESS=1 (blocking mode) correctly "
      "serializes GPU API calls across processes. Process A holds the device "
      "mutex for several seconds via rsmi_test_sleep. Process B then calls an "
      "AMDSMI API and measures how long it blocks. A wait time >= " +
      std::to_string(kMinWaitSeconds) +
      " seconds proves Process B was serialized rather than racing"
      " through concurrently.");
}

TestCrossProcessSerialization::~TestCrossProcessSerialization() {}

void TestCrossProcessSerialization::SetUp(void) {
  std::string label;
  amdsmi_status_t ret;

  IF_VERB(STANDARD) {
    MakeHeaderStr(kSetupLabel, &label);
    std::cout << "\n\t" << label << std::endl;
  }

  holder_process_ = false;
  child_ = 0;

  // Cross-process shared memory mutex is required.
  // Must be set before fork() so both processes inherit it and amdsmi_init()
  // uses shm_open/mmap instead of a per-process pthread_mutex_t.
  // Save original value so Close() can restore it after the test.
  const char* orig = getenv("AMDSMI_MUTEX_CROSS_PROCESS");
  orig_cross_process_env_ = orig ? orig : "";
  orig_cross_process_env_was_set_ = (orig != nullptr);
  setenv("AMDSMI_MUTEX_CROSS_PROCESS", "1", 1);

  // Three pipes provide a fully deterministic two-phase handshake:
  //   init_pipe_:         holder → waiter  (holder init complete)
  //   waiter_ready_pipe_: waiter → holder  (waiter init complete)
  //   run_pipe_:          holder → waiter  (holder about to acquire mutex)
  // This guarantees both processes finish amdsmi_init before the holder
  // calls rsmi_test_sleep, which would otherwise block the waiter's own
  // amdsmi_init (device enumeration acquires the shared device mutex).
  if (pipe(init_pipe_) < 0 || pipe(waiter_ready_pipe_) < 0 || pipe(run_pipe_) < 0) {
    std::cout << "pipe() failed: " << strerror(errno) << std::endl;
    setup_failed_ = true;
    return;
  }

  child_ = fork();
  if (child_ < 0) {
    std::cout << "fork() failed: " << strerror(errno) << std::endl;
    close(init_pipe_[0]);
    close(init_pipe_[1]);
    close(waiter_ready_pipe_[0]);
    close(waiter_ready_pipe_[1]);
    close(run_pipe_[0]);
    close(run_pipe_[1]);
    setup_failed_ = true;
    return;
  }

  if (child_ != 0) {
    // Parent = mutex holder process. Init without RESRV_TEST1 so locks block.
    holder_process_ = true;
    // Holder: does not read init_pipe or waiter_ready_pipe write-end,
    // does not read run_pipe.
    close(init_pipe_[0]);
    close(waiter_ready_pipe_[1]);
    close(run_pipe_[0]);

    DISPLAY_AMDSMI_API("[holder] amdsmi_init", "", VERB(STANDARD));
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    // Phase 1: signal waiter that holder's amdsmi_init is complete.
    char ready = 1;
    write(init_pipe_[1], &ready, 1);
    close(init_pipe_[1]);

    // Phase 2: wait for waiter to also finish amdsmi_init before entering
    // Run(). Without this barrier the holder would call rsmi_test_sleep
    // (acquiring the device mutex) while the waiter's amdsmi_init is still
    // trying to enumerate devices — which also acquires that same mutex.
    // Then amdsmi_get_gpu_id would see a free mutex and return instantly.
    char waiter_done = 0;
    if (read(waiter_ready_pipe_[0], &waiter_done, 1) < 0) {
      std::cout << "read(waiter_ready_pipe_) failed: " << strerror(errno) << std::endl;
      setup_failed_ = true;
    }
    close(waiter_ready_pipe_[0]);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  } else {
    // Waiter: does not write to init_pipe, does not read waiter_ready_pipe
    // read-end, does not write to run_pipe.
    close(init_pipe_[1]);
    close(waiter_ready_pipe_[0]);
    close(run_pipe_[1]);

    // Phase 1: block until holder's amdsmi_init is complete.
    char ready = 0;
    if (read(init_pipe_[0], &ready, 1) < 0) {
      std::cout << "read(init_pipe_) failed: " << strerror(errno) << std::endl;
      setup_failed_ = true;
    }
    close(init_pipe_[0]);

    DISPLAY_AMDSMI_API("[waiter] amdsmi_init", "", VERB(STANDARD));
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    // Phase 2: signal holder that waiter's amdsmi_init is also complete.
    char waiter_done = 1;
    write(waiter_ready_pipe_[1], &waiter_done, 1);
    close(waiter_ready_pipe_[1]);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  }

  // Enumerate processor handles in each process after amdsmi_init()
  uint32_t socket_count = 0;
  amdsmi_status_t enum_err = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (enum_err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
    return;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  enum_err = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
  if (enum_err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
    return;
  }
  num_monitor_devs_ = 0;
  for (uint32_t i = 0; i < socket_count; i++) {
    uint32_t device_count = 0;
    amdsmi_status_t status = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS || device_count == 0) {
      continue;
    }
    std::vector<amdsmi_processor_handle> handles(device_count);
    status = amdsmi_get_processor_handles(sockets[i], &device_count, &handles[0]);
    if (status != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (uint32_t j = 0; j < device_count && num_monitor_devs_ < MAX_MONITOR_DEVICES; j++) {
      processor_handles_[num_monitor_devs_++] = handles[j];
    }
  }

  if (num_monitor_devs_ == 0) {
    std::cout << "No monitor devices found on this machine." << std::endl;
    setup_failed_ = true;
  }
}

void TestCrossProcessSerialization::DisplayTestInfo(void) {
  IF_VERB(STANDARD) { TestBase::DisplayTestInfo(); }
}

void TestCrossProcessSerialization::DisplayResults(void) const {
  IF_VERB(STANDARD) { TestBase::DisplayResults(); }
}

void TestCrossProcessSerialization::Close() {
  // Shut down while AMDSMI_MUTEX_CROSS_PROCESS is still set so that
  // shared_mutex_close() correctly munmaps instead of calling delete.
  TestBase::Close();

  if (orig_cross_process_env_was_set_) {
    setenv("AMDSMI_MUTEX_CROSS_PROCESS", orig_cross_process_env_.c_str(), 1);
  } else {
    unsetenv("AMDSMI_MUTEX_CROSS_PROCESS");
  }
}

void TestCrossProcessSerialization::Run(void) {
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    if (!holder_process_) {
      // In the child process, _exit() immediately to avoid running gtest/AMDSMI cleanup
      // cleanup that was never meant to run here.
      std::cout.flush();
      _exit(1);
    }
    // Holder process: reap the child to avoid leaving a zombie.
    if (child_ > 0) {
      int child_status = 0;
      waitpid(child_, &child_status, 0);
    }
    return;
  }

  if (holder_process_) {
    PRINT_VERBOSITY();

    // Block SIGCHLD so it doesn't interrupt sleep inside rsmi_test_sleep.
    sigset_t sigchld_mask, old_mask;
    sigemptyset(&sigchld_mask);
    sigaddset(&sigchld_mask, SIGCHLD);
    sigprocmask(SIG_BLOCK, &sigchld_mask, &old_mask);

    IF_VERB(STANDARD) {
      std::cout << "HOLDER process: acquiring mutex and sleeping for " << kHoldSeconds << "s..."
                << std::endl;
    }
    // Signal waiter that holder is about to enter rsmi_test_sleep (acquire
    // mutex), then immediately call it. The waiter reads this signal and
    // pauses kWaiterMutexPauseNs before calling amdsmi_get_gpu_id.
    char run_ready = 1;
    write(run_pipe_[1], &run_ready, 1);
    close(run_pipe_[1]);
    amdsmi_status_t ret = rsmi_test_sleep(0, kHoldSeconds);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) {
      std::cout << "HOLDER process: released mutex after " << kHoldSeconds << "s." << std::endl;
    }

    sigprocmask(SIG_SETMASK, &old_mask, nullptr);
    int child_status = 0;
    pid_t cpid = wait(&child_status);
    ASSERT_EQ(cpid, child_);
    ASSERT_TRUE(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0)
        << "WAITER process exited with non-zero status " << WEXITSTATUS(child_status)
        << " — child assertions failed (see output above)";

  } else {
    TestBase::Run();

    // Exit the child process with failure. GTEST_FAIL() cannot be used here
    // because gtest state does not cross fork() boundaries — the parent reports
    // child failure by inspecting the exit status via wait().
    auto fail_child = [](const std::string& msg) {
      std::cout << msg << std::endl;
      std::cout.flush();
      _exit(1);
    };

    // Block until holder signals it is about to acquire the mutex, then
    // pause briefly so rsmi_test_sleep's pthread_mutex_lock completes
    // before we attempt to acquire the same lock via amdsmi_get_gpu_id.
    char run_ready = 0;
    read(run_pipe_[0], &run_ready, 1);
    close(run_pipe_[0]);

    // Print after the pipe read so this message always follows the holder's
    // "acquiring mutex" message in the output (the holder writes to the pipe
    // immediately after printing its own message).
    IF_VERB(STANDARD) {
      std::cout << "WAITER process: calling amdsmi_get_gpu_id() — should block until "
                   "HOLDER releases mutex..."
                << std::endl;
    }

    struct timespec pause_ts = {0, kWaiterMutexPauseNs};
    nanosleep(&pause_ts, nullptr);

    uint16_t gpu_id = 0;
    auto t0 = std::chrono::steady_clock::now();

    DISPLAY_AMDSMI_API("[waiter] amdsmi_get_gpu_id(processor_handles_[0], &gpu_id)", "",
                       VERB(STANDARD));
    // This call must wait for the holder's rsmi_test_sleep to finish.
    amdsmi_status_t ret = amdsmi_get_gpu_id(processor_handles_[0], &gpu_id);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    IF_VERB(STANDARD) {
      std::cout << "WAITER process: amdsmi_get_gpu_id() returned after " << elapsed << "s"
                << " (expected >= " << kMinWaitSeconds << "s)" << std::endl;
    }

    if (ret != AMDSMI_STATUS_SUCCESS) {
      fail_child("WAITER process: amdsmi_get_gpu_id() returned " + std::to_string(ret) +
                 " — expected AMDSMI_STATUS_SUCCESS after mutex was released");
    }
    if (elapsed < kMinWaitSeconds) {
      fail_child("WAITER process: elapsed " + std::to_string(elapsed) + "s < " +
                 std::to_string(kMinWaitSeconds) +
                 "s — waiter did not block, cross-process serialization is broken");
    }

    IF_VERB(STANDARD) {
      std::cout << "WAITER process: serialization verified after " << elapsed << "s. gpu_id=0x"
                << std::hex << gpu_id << std::dec << std::endl;
    }

    std::cout.flush();
    _exit(0);
  }
}
