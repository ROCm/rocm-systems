/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file Rendezvous.hpp
 * @brief File-based READY/GO barrier for the init-pipeline test execution mode.
 */

#ifndef RCCL_TEST_RENDEZVOUS_HPP
#define RCCL_TEST_RENDEZVOUS_HPP

#include <string>

namespace RcclUnitTesting
{
  /**
   * @class Rendezvous
   * @brief C++ side of the init-pipeline READY/GO barrier (opt-in, no-op default).
   *
   * The init-pipeline runner launches many test entries that initialize (warm
   * their RCCL device code) concurrently, parks each at a READY barrier, then
   * releases exactly one at a time (GO) so no two tested executions overlap. This
   * helper is the in-binary side of that barrier and is a no-op unless
   * RCCL_TEST_RENDEZVOUS_DIR is set, so the default (serial) path is unchanged.
   *
   * Tokens live in RCCL_TEST_RENDEZVOUS_DIR (the runner sets it per entry to a
   * fresh path): the binary writes "ready", the runner writes "go". Tokens are
   * created write-temp-then-atomic-rename so a reader never sees a torn file.
   *
   * @note publishReady() must be called only AFTER device-code warmup, and from a
   *       process that has NOT initialized HIP in a fork parent (see
   *       test/common/ForkSafetyInvariant.md). The TestBed parent qualifies (its
   *       warmup runs in the forked children); for MPI, rank 0 qualifies.
   * @note waitForGo() blocks INDEFINITELY -- the runner owns every phase timeout,
   *       so there is no C++ init-scale timeout that could collide with the
   *       runner's queue-wait. It aborts early only on runner death, detected via
   *       an inherited liveness pipe (RCCL_TEST_LIVENESS_FD) when the runner
   *       provides one; process-group teardown from the runner remains the
   *       primary orphan-avoidance mechanism.
   */
  class Rendezvous
  {
  public:
    /// True iff RCCL_TEST_RENDEZVOUS_DIR is set (feature enabled for this process).
    static bool Enabled();

    /// Atomically publish the READY token. No-op when disabled. Returns false on
    /// an I/O error so the caller can fail the entry rather than hang.
    static bool PublishReady();

    /// Block until the GO token appears (or the runner dies). No-op when disabled.
    static void WaitForGo();

  private:
    static std::string Dir();
  };
}

#endif // RCCL_TEST_RENDEZVOUS_HPP
