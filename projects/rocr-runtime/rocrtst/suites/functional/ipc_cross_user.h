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

#ifndef ROCRTST_SUITES_FUNCTIONAL_IPC_CROSS_USER_H_
#define ROCRTST_SUITES_FUNCTIONAL_IPC_CROSS_USER_H_

#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <string>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// State shared between the exporting (parent) and importing (child) process.
struct IPCCrossUserShared {
  // Parent sets 1 once the IPC handle is published.
  std::atomic<int> token;
  // Child sets to 1 once it has attempted the attach (see attach_status), or
  // -1 if it failed before it could try.
  std::atomic<int> child_status;
  // hsa_status_t returned by the child's hsa_amd_ipc_memory_attach().
  std::atomic<int> attach_status;
  std::atomic<size_t> size;
  hsa_amd_ipc_memory_t handle;
};

// Verifies the same-user check on the IPC export socket.
//
// The test is opt-in. It runs only when ROCRTST_IPC_SECOND_UID names a second
// user ID, and it needs root so the importing child can switch to that user.
// The second user must be able to open /dev/kfd and the render nodes (e.g. be
// in the render or video group).
//
// The parent exports a GPU buffer. A forked child then drops to the second user
// and attaches to it. The outcome expected depends on the exporter's
// environment, so run the test twice:
//   * HSA_ENABLE_IPC_CROSS_USER unset (or not "1"): the attach must fail.
//   * HSA_ENABLE_IPC_CROSS_USER=1: the attach must succeed.
class IPCCrossUserTest : public TestBase {
 public:
  IPCCrossUserTest();
  virtual ~IPCCrossUserTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

 private:
  // Switch the calling process to |uid| (with that user's groups) and verify
  // that it cannot get root back. On failure returns false and fills |why|.
  static bool DropToUser(uid_t uid, std::string* why);

  // Child: wait for the parent to publish the handle, attach, and record the
  // result in shared_. Returns the child's exit code.
  int ChildProcessImpl();

  // Parent: export a buffer, wait for the child's verdict, and check it against
  // the exporter's HSA_ENABLE_IPC_CROSS_USER setting.
  void ParentProcessImpl();

  // Non-blocking check for child termination; caches the wait status.
  bool ChildHasExited(int* status);

  // Kill the child if still running, then reap it.
  void TerminateChild();

  int child_ = 0;
  IPCCrossUserShared* shared_ = nullptr;
  bool parentProcess_ = true;
  // Set once SetUp() finished in this process; Run() does nothing otherwise.
  bool setup_ok_ = false;
  uid_t second_uid_ = 0;
  pid_t parent_pid_ = 0;
  size_t gpu_mem_granule_ = 0;
  bool child_reaped_ = false;
  int child_wait_status_ = 0;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_IPC_CROSS_USER_H_
