/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <thread>

#include "TestBed.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  // Stress coverage for the parallelized TestBed::DestroyComms teardown
  // (ROCM-25953): DestroyComms now broadcasts the destroy command to every child
  // process first and collects the acknowledgements in a second pass, so the
  // children tear their communicators down concurrently instead of one at a time.
  // These tests drive that path repeatedly, in multi-process mode (one child per
  // GPU) where the two-pass ordering actually matters, and in both blocking and
  // non-blocking modes (the child-side DestroyComms branches on useBlocking).
  namespace
  {
    // Returns true if ncclFloat32 is available under the current UT_DATATYPES.
    // Checked once in each test body so the test can GTEST_SKIP() explicitly
    // rather than silently passing as a no-op when the datatype is excluded.
    bool float32Supported(TestBed& testBed)
    {
      std::vector<ncclDataType_t> dataTypes;
      testBed.GetSupportedDataTypes(dataTypes, {ncclFloat32});
      return !dataTypes.empty();
    }

    // Run `iterations` init / collective / destroy cycles over `totalRanks`
    // ranks, one child process per rank. Each cycle ends in DestroyComms, the
    // path under test. A leaked pipe fd or child handle, or any mismatch between
    // the broadcast pass and the ack-collection pass, shows up as a hang or a
    // failure once enough cycles accumulate.
    void RunTeardownCycles(TestBed& testBed,
                           int  const totalRanks,
                           bool const useBlocking,
                           int  const iterations,
                           bool&      isCorrect)
    {
      size_t const numElements   = 32 * 1024;
      bool   const inPlace       = false;
      bool   const useManagedMem = false;
      int    const numProcesses  = totalRanks;  // one child process per rank
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();

      for (int iter = 0; iter < iterations && isCorrect; ++iter)
      {
        testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks,
                                                    gpuPriorityOrder),
                          1, 1, 1, useBlocking);

        OptionalColArgs options;
        options.redOp = ncclSum;
        testBed.SetCollectiveArgs(ncclCollAllReduce, ncclFloat32,
                                  numElements, numElements, options);
        testBed.AllocateMem(inPlace, useManagedMem);
        testBed.PrepareData();
        testBed.ExecuteCollectives();
        testBed.ValidateResults(isCorrect);
        testBed.DeallocateMem();
        testBed.DestroyComms();
      }
    }
  }

  // Repeated multi-process teardown in both blocking and non-blocking modes.
  TEST(Teardown, RepeatedDestroyComms)
  {
    TestBed testBed;
    if (testBed.ev.maxGpus < 2)
      GTEST_SKIP() << "Teardown stress requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    if (!(testBed.ev.processMask & (1 << 1)))
      GTEST_SKIP() << "Teardown stress requires multi-process mode (UT_PROCESS_MASK)";
    if (!float32Supported(testBed))
      GTEST_SKIP() << "Teardown stress requires ncclFloat32 (excluded by UT_DATATYPES)";

    bool isCorrect = true;
    for (bool useBlocking : {true, false})
      RunTeardownCycles(testBed, testBed.ev.maxGpus, useBlocking,
                        /*iterations*/ 3, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  // Teardown across a varying number of child processes within one test, so
  // DestroyComms is exercised with different numActiveChildren values back to
  // back. Catches assumptions that only hold for a fixed child count.
  TEST(Teardown, DestroyCommsVaryingChildCount)
  {
    TestBed testBed;
    if (testBed.ev.maxGpus < 2)
      GTEST_SKIP() << "Teardown stress requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    if (!(testBed.ev.processMask & (1 << 1)))
      GTEST_SKIP() << "Teardown stress requires multi-process mode (UT_PROCESS_MASK)";
    if (!float32Supported(testBed))
      GTEST_SKIP() << "Teardown stress requires ncclFloat32 (excluded by UT_DATATYPES)";

    bool isCorrect = true;
    for (int ranks = testBed.ev.maxGpus; ranks >= 2 && isCorrect; ranks /= 2)
      RunTeardownCycles(testBed, ranks, /*useBlocking*/ true,
                        /*iterations*/ 1, isCorrect);
    EXPECT_TRUE(isCorrect);
    testBed.Finalize();
  }

  // AICOMRCCL-2467 / AICOMRCCL-2468: commDestroySync() runs a host-local barrier
  // (bootstrapIntraNodeBarrier) before stopping the proxy, so that a rank cannot
  // stop its proxy while a host-local peer is still establishing a PXN
  // connection. The barrier requires every host-local rank to participate, and
  // nothing cancels it when one of them does not. A rank with a non-zero
  // abortFlag skips the barrier, so a peer that entered it waits forever in
  // bootstrapRecv -- the PyTorch ProcessGroupNCCL watchdog hits exactly this.
  //
  // These run on a single-node communicator at nRanks == 2, where the barrier
  // cannot be protecting anything: RCCL defaults NCCL_PXN_DISABLE=1 and
  // rcclSetPxn() only auto-enables PXN at nRanks >= 64 (gfx942) / >= 32
  // (gfx950), so there is no PXN relay to keep alive.
  namespace
  {
    // EnvVars reports 0 GPUs in a process re-exec'd by ProcessIsolatedTestRunner, and a
    // skipped child is scored as a pass -- so every GPU gate below must run in the parent
    // only, or the coverage silently evaporates into a green run.
    bool isIsolatedChild()
    {
      return std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr;
    }

    // Never call hipGetDeviceCount() here: HIP state does not survive the fork() below.
    int getDetectedGpuCount()
    {
      static const int detectedGpus = EnvVars().GetNumDetectedGpus();
      return detectedGpus;
    }

    enum class PeerTeardown
    {
      Abort,    // peer calls ncclCommAbort -- non-zero abortFlag, skips the barrier
      Destroy   // peer calls ncclCommDestroy -- control, both ranks reach the barrier
    };

    // Two ranks, one process each, one collective, then a deliberately divergent
    // teardown. Returns true if the rank taking the normal destroy path returned
    // from ncclCommDestroy before the deadline.
    //
    // pxnDisable is pinned in each child rather than inherited, because whether
    // the destroy barrier runs at all is derived from it -- an ambient
    // NCCL_PXN_DISABLE would silently change which path is under test.
    //
    // Nothing here may touch HIP or RCCL before fork(): initializing the runtime
    // in the parent leaves the children with unusable state and they die in
    // hipSetDevice.
    bool DestroyReturnsWhenPeer(PeerTeardown peer, char const* pxnDisable, int deadlineSeconds)
    {
      constexpr int    nRanks      = 2;
      constexpr size_t numElements = 32 * 1024;

      int idpipe[2];
      if (pipe(idpipe) != 0) return false;

      pid_t kids[nRanks] = {-1, -1};
      for (int rank = 0; rank < nRanks; ++rank)
      {
        pid_t pid = fork();
        if (pid < 0)
        {
          close(idpipe[0]);
          close(idpipe[1]);
          return false;
        }
        if (pid == 0)
        {
          if (setenv("NCCL_PXN_DISABLE", pxnDisable, 1) != 0) _exit(69);

          ncclUniqueId id;
          if (rank == 0)
          {
            close(idpipe[0]);
            if (ncclGetUniqueId(&id) != ncclSuccess) _exit(70);
            if (write(idpipe[1], &id, sizeof(id)) != (ssize_t)sizeof(id)) _exit(71);
            close(idpipe[1]);
          }
          else
          {
            close(idpipe[1]);
            size_t got = 0;
            while (got < sizeof(id))
            {
              ssize_t n = read(idpipe[0], ((char*)&id) + got, sizeof(id) - got);
              if (n <= 0) _exit(72);
              got += n;
            }
            close(idpipe[0]);
          }

          if (hipSetDevice(rank) != hipSuccess) _exit(73);
          hipStream_t stream;
          if (hipStreamCreate(&stream) != hipSuccess) _exit(74);
          float* buf = nullptr;
          if (hipMalloc(&buf, numElements * sizeof(float)) != hipSuccess) _exit(75);
          if (hipMemset(buf, 0, numElements * sizeof(float)) != hipSuccess) _exit(76);

          ncclComm_t comm;
          if (ncclCommInitRank(&comm, nRanks, id, rank) != ncclSuccess) _exit(77);
          if (ncclAllReduce(buf, buf, numElements, ncclFloat, ncclSum, comm, stream) != ncclSuccess)
            _exit(78);
          if (hipStreamSynchronize(stream) != hipSuccess) _exit(79);

          if (rank == 1 && peer == PeerTeardown::Abort)
          {
            if (ncclCommAbort(comm) != ncclSuccess) _exit(80);
          }
          else
          {
            // rank 0 always takes the barrier path. A non-success rc here is a real
            // failure, not a hang -- without this check the parent reads exit 0 and
            // scores a broken destroy as a pass.
            if (ncclCommDestroy(comm) != ncclSuccess) _exit(81);
          }

          _exit(0);
        }
        kids[rank] = pid;
      }
      close(idpipe[0]);
      close(idpipe[1]);

      // Poll for rank 0 only. It is the one that enters the barrier, so it is
      // the one that hangs on regression.
      bool returned = false;
      auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(deadlineSeconds);
      while (std::chrono::steady_clock::now() < deadline)
      {
        int   status = 0;
        pid_t done   = waitpid(kids[0], &status, WNOHANG);
        if (done == kids[0])
        {
          // WEXITSTATUS is meaningless for a signal-killed child; checking it
          // alone would report a crash as a clean pass.
          returned = !WIFSIGNALED(status) && WEXITSTATUS(status) == 0;
          break;
        }
        if (done < 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      for (int i = 0; i < nRanks; ++i)
      {
        if (kids[i] <= 0) continue;
        kill(kids[i], SIGKILL);
        waitpid(kids[i], nullptr, 0);
      }
      return returned;
    }
  }

  // Control: both ranks destroy normally, so both reach the barrier and it
  // completes. If this fails the harness itself is broken and the companion
  // test below proves nothing.
  TEST(Teardown, DivergentTeardown_PeerDestroys_Completes)
  {
    if (!isIsolatedChild() && getDetectedGpuCount() < 2)
      GTEST_SKIP() << "Divergent teardown requires at least 2 GPUs (detected "
                   << getDetectedGpuCount() << ")";

    RUN_ISOLATED_TESTS(
      ProcessIsolatedTestRunner::TestConfig(
        "DivergentTeardown_PeerDestroys",
        []() {
          EXPECT_TRUE(DestroyReturnsWhenPeer(PeerTeardown::Destroy, "1", /*deadlineSeconds*/ 60))
            << "rank 0 did not return from ncclCommDestroy even though its peer "
               "also destroyed -- the host-local destroy barrier is broken";
        })
        .withTimeout(std::chrono::seconds(120)));
  }

  // Same control with PXN on, so the barrier is actually entered. Guards against
  // "fixing" the hang by disabling the barrier outright: with every rank
  // participating it must still rendezvous and complete.
  TEST(Teardown, DivergentTeardown_PeerDestroys_PxnEnabled_Completes)
  {
    if (!isIsolatedChild() && getDetectedGpuCount() < 2)
      GTEST_SKIP() << "Divergent teardown requires at least 2 GPUs (detected "
                   << getDetectedGpuCount() << ")";

    RUN_ISOLATED_TESTS(
      ProcessIsolatedTestRunner::TestConfig(
        "DivergentTeardown_PeerDestroysPxn",
        []() {
          EXPECT_TRUE(DestroyReturnsWhenPeer(PeerTeardown::Destroy, "0", /*deadlineSeconds*/ 60))
            << "rank 0 did not return from ncclCommDestroy with NCCL_PXN_DISABLE=0 -- "
               "the destroy barrier does not complete even when every rank enters it";
        })
        .withTimeout(std::chrono::seconds(120)));
  }

  // Regression: the peer aborts, so it skips the barrier. Rank 0 must still
  // return instead of blocking forever in bootstrapRecv.
  //
  // Pinned to NCCL_PXN_DISABLE=1, RCCL's default. With PXN genuinely enabled an
  // ncclCommAbort peer still strands the barrier -- that needs a cancellation
  // protocol and is tracked separately; do not widen this case without one.
  TEST(Teardown, DivergentTeardown_PeerAborts_DoesNotHang)
  {
    if (!isIsolatedChild() && getDetectedGpuCount() < 2)
      GTEST_SKIP() << "Divergent teardown requires at least 2 GPUs (detected "
                   << getDetectedGpuCount() << ")";

    RUN_ISOLATED_TESTS(
      ProcessIsolatedTestRunner::TestConfig(
        "DivergentTeardown_PeerAborts",
        []() {
          EXPECT_TRUE(DestroyReturnsWhenPeer(PeerTeardown::Abort, "1", /*deadlineSeconds*/ 60))
            << "rank 0 hung in ncclCommDestroy after its host-local peer aborted "
               "and skipped the destroy barrier (AICOMRCCL-2467 / AICOMRCCL-2468)";
        })
        .withTimeout(std::chrono::seconds(120)));
  }
}
