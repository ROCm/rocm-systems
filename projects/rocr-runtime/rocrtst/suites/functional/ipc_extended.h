/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_IPC_EXTENDED_H_
#define ROCRTST_SUITES_FUNCTIONAL_IPC_EXTENDED_H_

#include <sys/types.h>
#include <unistd.h>
#include <atomic>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

/**
 * Extended IPC shared memory structure for complex test scenarios
 */
struct IPCExtendedShared {
  std::atomic<int> token;
  std::atomic<int> count;
  std::atomic<size_t> size;
  std::atomic<int> child_status;
  std::atomic<int> parent_status;
  std::atomic<int> phase;  // Multi-phase sync
  hsa_amd_ipc_memory_t handle;
  hsa_amd_ipc_memory_t handle2;  // Second handle for re-export tests
  hsa_amd_ipc_signal_t signal_handle;
};

/**
 * IPCServerRestartTest - Tests socket-server restart race condition
 *
 * Scenario: When all exports are freed while the old server thread is still
 * blocked in accept(), and a new export comes in, the server needs to restart.
 *
 * Test flow:
 * 1. Parent: Export memory buffer A -> starts IPC socket server
 * 2. Child: Attach to buffer A
 * 3. Child: Detach from buffer A
 * 4. Parent: Free buffer A -> ipc_sock_server_conns_ becomes empty
 * 5. Parent: Export new buffer B -> triggers server restart logic
 * 6. Child: Attach to buffer B -> validates restart succeeded
 * 7. Both: Cleanup
 */
class IPCServerRestartTest : public TestBase {
 public:
  IPCServerRestartTest();
  virtual ~IPCServerRestartTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

 private:
  uint32_t RealIterationNum(void);
  void PrintVerboseMesg(void);
  void ParentProcessImpl();
  void ChildProcessImpl();

  int child_;
  IPCExtendedShared* shared_;
  bool parentProcess_;
  size_t gpu_mem_granule;
  int32_t timeout_ = 0x20000;
};

/**
 * IPCFreeBeforeDetachTest - Tests free-before-detach bookkeeping
 *
 * Scenario: Exporter frees memory while importer hasn't detached yet.
 * Tests that:
 * 1. FreeMemory correctly removes ipc_sock_server_conns_ entry
 * 2. No stale connection entries remain
 * 3. Subsequent export still succeeds
 *
 * Test flow:
 * 1. Parent: Export memory buffer
 * 2. Child: Attach to buffer
 * 3. Parent: Free the buffer (before child detaches)
 * 4. Parent: Export a new buffer (should succeed, no stale entries)
 * 5. Child: Detach from old buffer
 * 6. Child: Attach to new buffer (validates cleanup worked)
 * 7. Both: Normal cleanup
 */
class IPCFreeBeforeDetachTest : public TestBase {
 public:
  IPCFreeBeforeDetachTest();
  virtual ~IPCFreeBeforeDetachTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

 private:
  uint32_t RealIterationNum(void);
  void PrintVerboseMesg(void);
  void ParentProcessImpl();
  void ChildProcessImpl();

  int child_;
  IPCExtendedShared* shared_;
  bool parentProcess_;
  size_t gpu_mem_granule;
  int32_t timeout_ = 0x20000;
};

/**
 * IPCRepeatedHandleTest - Tests repeated IPC handle creation
 *
 * Scenario: Create multiple IPC handles for the same or different memory
 * regions to verify BO handle management doesn't leak.
 *
 * Test flow:
 * 1. Parent: Allocate buffer, create IPC handle
 * 2. Parent: Create IPC handle again for same buffer
 * 3. Child: Attach/detach multiple times
 * 4. Parent: Free and re-allocate
 * 5. Repeat cycle to catch handle leaks
 */
class IPCRepeatedHandleTest : public TestBase {
 public:
  IPCRepeatedHandleTest();
  virtual ~IPCRepeatedHandleTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

 private:
  uint32_t RealIterationNum(void);
  void PrintVerboseMesg(void);
  void ParentProcessImpl();
  void ChildProcessImpl();

  int child_;
  IPCExtendedShared* shared_;
  bool parentProcess_;
  size_t gpu_mem_granule;
  int32_t timeout_ = 0x20000;

  static constexpr int kRepeatCount = 3;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_IPC_EXTENDED_H_
