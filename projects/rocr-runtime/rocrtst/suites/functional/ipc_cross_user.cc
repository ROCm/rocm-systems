/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

// Cross-user IPC test.
//
// The exporting process serves IPC buffers only to processes that run as its
// own user unless HSA_ENABLE_IPC_CROSS_USER=1. This test exports a GPU buffer
// from the parent and attaches to it from a child that has switched to a second
// user, then checks the outcome against the exporter's setting.
//
// Opt-in: the test does nothing unless ROCRTST_IPC_SECOND_UID is set to the
// numeric ID of a user other than the one running the test. It must run as root
// so the child can switch users, and the second user must be able to open
// /dev/kfd and the render nodes.
//
// Attach must fail:
//   sudo ROCRTST_IPC_SECOND_UID=1100 ./rocrtst64 --gtest_filter=rocrtstFunc.IPC_Cross_User
// Attach must succeed:
//   sudo ROCRTST_IPC_SECOND_UID=1100 HSA_ENABLE_IPC_CROSS_USER=1 ./rocrtst64
//        --gtest_filter=rocrtstFunc.IPC_Cross_User
//
// Parent: allocate a GPU buffer, create an IPC handle, publish it to the child
//   through shared memory, wait for the child's verdict.
// Child (forked before hsa_init, then switched to the second user): attach to
//   the handle, record the status, detach if the attach worked.

#include <grp.h>
#include <pwd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "suites/functional/ipc_cross_user.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

namespace {

// Upper bound on any single wait for the peer. The rejected attach fails right
// away; the importer's own receive timeout is 10 s, so this leaves headroom.
const int kPeerWaitTimeoutMs = 30000;
const int kPollMs = 10;
const int kChildExitFail = 1;

const char kSecondUidEnv[] = "ROCRTST_IPC_SECOND_UID";

std::string DescribeWaitStatus(int status) {
  std::ostringstream os;
  if (WIFEXITED(status)) {
    os << "child exited with status " << WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    os << "child was terminated by signal " << WTERMSIG(status) << " (" << strsignal(WTERMSIG(status))
       << ")";
  } else {
    os << "child is in an unexpected wait state (raw status " << status << ")";
  }
  return os.str();
}

}  // namespace

IPCCrossUserTest::IPCCrossUserTest() : TestBase() {
  set_num_iteration(1);
  set_title("IPC Cross-User Test");
  set_description(
      "Verifies that an IPC exporter refuses to serve a process running as another user, "
      "unless HSA_ENABLE_IPC_CROSS_USER=1. Opt-in: set ROCRTST_IPC_SECOND_UID to the ID of a "
      "second user and run as root.");
}

IPCCrossUserTest::~IPCCrossUserTest() {}

bool IPCCrossUserTest::DropToUser(uid_t uid, std::string* why) {
  struct passwd* pw = getpwuid(uid);
  if (pw == nullptr) {
    *why = "uid " + std::to_string(uid) +
           " has no passwd entry; its groups (render/video) are needed to open the GPU";
    return false;
  }
  // Order matters: supplementary groups and gid cannot be changed after setuid.
  if (initgroups(pw->pw_name, pw->pw_gid) != 0) {
    *why = std::string("initgroups failed: ") + strerror(errno);
    return false;
  }
  if (setgid(pw->pw_gid) != 0) {
    *why = std::string("setgid failed: ") + strerror(errno);
    return false;
  }
  if (setuid(uid) != 0) {
    *why = std::string("setuid failed: ") + strerror(errno);
    return false;
  }
  if (getuid() != uid || geteuid() != uid) {
    *why = "process did not switch to the requested uid";
    return false;
  }
  if (setuid(0) == 0) {
    *why = "process could still regain root after setuid";
    return false;
  }
  // Switching away from root clears the dumpable flag, which makes /proc/self
  // root-owned and unreadable to this process.
  prctl(PR_SET_DUMPABLE, 1, 0, 0, 0);
  return true;
}

void IPCCrossUserTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;

#ifdef ROCRTST_ASAN
  // The test forks; ASAN's shadow memory is inconsistent in the child.
  std::cout << "Skipping IPC cross-user test under ASAN (fork unsupported)." << std::endl;
  test_skipped_ = true;
  return;
#endif

  const char* uid_env = getenv(kSecondUidEnv);
  if (uid_env == nullptr || *uid_env == '\0') {
    std::cout << "Skipping IPC cross-user test: set " << kSecondUidEnv
              << "=<uid of a second user> to run it." << std::endl;
    test_skipped_ = true;
    return;
  }

  char* end = nullptr;
  errno = 0;
  unsigned long uid_val = strtoul(uid_env, &end, 10);
  if (errno != 0 || end == uid_env || *end != '\0') {
    ADD_FAILURE() << kSecondUidEnv << "=\"" << uid_env << "\" is not a numeric user ID.";
    test_skipped_ = true;
    return;
  }
  second_uid_ = static_cast<uid_t>(uid_val);

  if (geteuid() != 0) {
    ADD_FAILURE() << "The IPC cross-user test must run as root so the importing process can "
                     "switch to uid "
                  << second_uid_ << ".";
    test_skipped_ = true;
    return;
  }
  if (second_uid_ == geteuid()) {
    ADD_FAILURE() << kSecondUidEnv << " must differ from the user running the test (uid "
                  << geteuid() << ").";
    test_skipped_ = true;
    return;
  }

  // Fork before hsa_init: each process must initialize the runtime on its own.
  shared_ = reinterpret_cast<IPCCrossUserShared*>(mmap(nullptr, sizeof(IPCCrossUserShared),
                                                       PROT_READ | PROT_WRITE,
                                                       MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (shared_ == MAP_FAILED) {
    shared_ = nullptr;
    ADD_FAILURE() << "mmap failed to allocate shared memory";
    test_skipped_ = true;
    return;
  }
  memset(static_cast<void*>(shared_), 0, sizeof(IPCCrossUserShared));

  // Flush so the child does not repeat buffered output.
  fflush(stdout);
  fflush(stderr);
  child_ = fork();
  if (child_ < 0) {
    child_ = 0;
    ADD_FAILURE() << "fork failed: " << strerror(errno);
    munmap(shared_, sizeof(IPCCrossUserShared));
    shared_ = nullptr;
    test_skipped_ = true;
    return;
  }

  parentProcess_ = (child_ != 0);
  if (!parentProcess_) {
    parent_pid_ = getppid();
    std::string why;
    if (!DropToUser(second_uid_, &why)) {
      fprintf(stderr, "Child: cannot switch to uid %u: %s\n", static_cast<unsigned>(second_uid_),
              why.c_str());
      shared_->child_status = -1;
      _exit(kChildExitFail);
    }
  }

  // Calls hsa_init() in this process: as root in the parent, as the second user
  // in the child.
  TestBase::SetUp();

  // Failures in the child are not seen by the parent's gtest, so report them
  // through shared memory and leave without running Run().
  auto child_bail = [&](const char* what) {
    fprintf(stderr, "Child (uid %u): %s\n", static_cast<unsigned>(getuid()), what);
    shared_->child_status = -1;
    _exit(kChildExitFail);
  };

  if (!parentProcess_ && (test_skipped_ || ::testing::Test::HasFailure()))
    child_bail("runtime setup failed; the second user may lack access to /dev/kfd or the render "
               "nodes");
  if (test_skipped_) return;

  hsa_status_t err = rocrtst::SetDefaultAgents(this);
  if (err == HSA_STATUS_SUCCESS) err = rocrtst::SetPoolsTypical(this);
  if (err != HSA_STATUS_SUCCESS) {
    if (!parentProcess_) child_bail("failed to find the default GPU agent and pools");
    ADD_FAILURE() << "Parent: failed to find the default GPU agent and pools";
    return;
  }

  if (parentProcess_) {
    err = hsa_amd_memory_pool_get_info(device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                       &gpu_mem_granule_);
    if (err != HSA_STATUS_SUCCESS) {
      ADD_FAILURE() << "Parent: failed to query the pool allocation granule";
      return;
    }
  }

  setup_ok_ = true;
}

bool IPCCrossUserTest::ChildHasExited(int* status) {
  if (child_reaped_) {
    *status = child_wait_status_;
    return true;
  }
  int st = 0;
  pid_t ret = waitpid(child_, &st, WNOHANG);
  if (ret == child_) {
    child_reaped_ = true;
    child_wait_status_ = st;
    *status = st;
    return true;
  }
  return false;
}

void IPCCrossUserTest::TerminateChild() {
  // child_ is 0 in the child and when SetUp() never forked; kill(0) would
  // signal the whole process group.
  if (child_ <= 0 || child_reaped_) return;
  kill(child_, SIGKILL);
  waitpid(child_, &child_wait_status_, 0);
  child_reaped_ = true;
}

int IPCCrossUserTest::ChildProcessImpl() {
  // Wait for the parent to publish the handle.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kPeerWaitTimeoutMs);
  while (shared_->token == 0) {
    if (getppid() != parent_pid_) {
      fprintf(stderr, "Child: parent is gone while waiting for the IPC handle\n");
      return kChildExitFail;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      fprintf(stderr, "Child: timed out after %d ms waiting for the IPC handle\n",
              kPeerWaitTimeoutMs);
      return kChildExitFail;
    }
    usleep(kPollMs * 1000);
  }

  hsa_agent_t ag_list[1] = {*gpu_device1()};
  void* ipc_ptr = nullptr;
  hsa_status_t err = hsa_amd_ipc_memory_attach(&shared_->handle, shared_->size, 1, ag_list, &ipc_ptr);
  fprintf(stdout, "Child (uid %u): hsa_amd_ipc_memory_attach returned %d\n",
          static_cast<unsigned>(getuid()), static_cast<int>(err));
  shared_->attach_status = static_cast<int>(err);

  if (err == HSA_STATUS_SUCCESS) {
    hsa_status_t derr = hsa_amd_ipc_memory_detach(ipc_ptr);
    if (derr != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "Child: hsa_amd_ipc_memory_detach failed (%d)\n", static_cast<int>(derr));
      return kChildExitFail;
    }
  }

  shared_->child_status = 1;
  return 0;
}

void IPCCrossUserTest::ParentProcessImpl() {
  // The exporter's own setting decides the expectation, because the runtime
  // reads it when the exporting process starts.
  const char* cross_env = getenv("HSA_ENABLE_IPC_CROSS_USER");
  const bool cross_user_enabled = (cross_env != nullptr && std::string(cross_env) == "1");

  void* buf = nullptr;
  hsa_status_t err = hsa_amd_memory_pool_allocate(device_pool(), gpu_mem_granule_, 0, &buf);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err) << "Parent: failed to allocate gpu memory";

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, buf);
  if (err == HSA_STATUS_SUCCESS) {
    shared_->size = gpu_mem_granule_;
    err = hsa_amd_ipc_memory_create(buf, gpu_mem_granule_, &shared_->handle);
  }
  if (err != HSA_STATUS_SUCCESS) {
    ADD_FAILURE() << "Parent: failed to export the buffer (" << err << ")";
    hsa_amd_memory_pool_free(buf);
    return;
  }

  std::cout << "Parent (uid " << getuid() << "): exported a buffer; HSA_ENABLE_IPC_CROSS_USER="
            << (cross_user_enabled ? "1" : "unset/0") << ", importer runs as uid " << second_uid_
            << std::endl;
  shared_->token = 1;

  // Wait for the child to finish its attach attempt.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kPeerWaitTimeoutMs);
  int child_exit_status = 0;
  bool child_gone = false;
  while (shared_->child_status == 0) {
    if (ChildHasExited(&child_exit_status)) {
      child_gone = true;
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) break;
    usleep(kPollMs * 1000);
  }

  const int child_status = shared_->child_status;
  if (child_status == 0) {
    // A rejected importer that hangs (rather than failing) lands here.
    if (child_gone) {
      ADD_FAILURE() << "Parent: " << DescribeWaitStatus(child_exit_status)
                    << " before reporting an attach result.";
    } else {
      ADD_FAILURE() << "Parent: the importer did not return from hsa_amd_ipc_memory_attach within "
                    << kPeerWaitTimeoutMs << " ms.";
    }
  } else if (child_status < 0) {
    ADD_FAILURE() << "Parent: the importer failed before it could attach; see its output above.";
  } else {
    const hsa_status_t attach = static_cast<hsa_status_t>(shared_->attach_status.load());
    if (cross_user_enabled) {
      EXPECT_EQ(HSA_STATUS_SUCCESS, attach)
          << "HSA_ENABLE_IPC_CROSS_USER=1: the attach from uid " << second_uid_
          << " should have succeeded.";
    } else {
      EXPECT_NE(HSA_STATUS_SUCCESS, attach)
          << "uid " << second_uid_ << " attached to a buffer exported by uid " << getuid()
          << " although HSA_ENABLE_IPC_CROSS_USER is not set.";
    }
  }

  // The child exits right after reporting; hold it to a clean exit.
  int wait_ms = 0;
  while (!ChildHasExited(&child_exit_status) && wait_ms < kPeerWaitTimeoutMs) {
    usleep(kPollMs * 1000);
    wait_ms += kPollMs;
  }
  if (!ChildHasExited(&child_exit_status)) {
    ADD_FAILURE() << "Parent: child (pid " << child_ << ") did not exit within "
                  << kPeerWaitTimeoutMs << " ms.";
    TerminateChild();
  } else if (!WIFEXITED(child_exit_status) || WEXITSTATUS(child_exit_status) != 0) {
    ADD_FAILURE() << "Parent: " << DescribeWaitStatus(child_exit_status) << ".";
  }

  err = hsa_amd_memory_pool_free(buf);
  EXPECT_EQ(HSA_STATUS_SUCCESS, err) << "Parent: failed to free gpu memory";
}

void IPCCrossUserTest::Run(void) {
  TestBase::Run();
  if (test_skipped_ || !setup_ok_) return;

  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    // The child must shut the runtime down itself, then leave without running
    // the rest of the gtest binary.
    int rc = ChildProcessImpl();
    hsa_shut_down();
    exit(rc);
  }
}

void IPCCrossUserTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void IPCCrossUserTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void IPCCrossUserTest::Close() {
  if (parentProcess_) {
    TerminateChild();
    if (shared_ != nullptr) {
      munmap(shared_, sizeof(IPCCrossUserShared));
      shared_ = nullptr;
    }
  }
  TestBase::Close();
}
