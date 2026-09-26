/*************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

#include <glob.h>
#include <unistd.h>
#include <sys/wait.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace RcclUnitTesting
{
  // Return true if device 0's architecture is one for which the LL128 P2P send/recv kernel is
  // generated and activated (gfx942/gfx950 only; see reg_values_of("SendRecv") and the enqueue
  // gate). The arch is queried in a forked child so the parent test process does not initialize
  // HIP (mirrors EnvVars' isolated arch detection).
  static bool DeviceSupportsLL128SendRecv()
  {
    int pipefd[2];
    if (pipe(pipefd) != 0) return false;
    pid_t pid = fork();
    if (pid == 0)
    {
      close(pipefd[0]);
      char arch[256] = "";
      hipDeviceProp_t prop;
      if (hipGetDeviceProperties(&prop, 0) == hipSuccess)
        strncpy(arch, prop.gcnArchName, sizeof(arch) - 1);
      ssize_t w = write(pipefd[1], arch, sizeof(arch));
      (void)w;
      close(pipefd[1]);
      _exit(0);
    }
    close(pipefd[1]);
    char arch[256] = "";
    ssize_t r = read(pipefd[0], arch, sizeof(arch));
    (void)r;
    close(pipefd[0]);
    waitpid(pid, nullptr, 0);
    std::string a(arch);
    return a.find("gfx942") != std::string::npos || a.find("gfx950") != std::string::npos;
  }

  // Scan the NCCL_DEBUG=INFO log files matching globPattern for the per-op protocol line emitted by
  // the P2P send/recv enqueue path ("RCCL P2P SendRecv protocol=<proto>"), returning true if any
  // line reports the given protocol ("LL128" or "Simple"). Used to assert that LL128 (or SIMPLE) was
  // actually selected rather than relying on data correctness alone, which both protocols satisfy.
  static bool DebugLogsContainProtocol(const std::string& globPattern, const char* protocol)
  {
    // Trailing space so "LL" does not also match the "LL128" log line (the enqueue log prints
    // "protocol=<proto> dir=...").
    const std::string needle = std::string("RCCL P2P SendRecv protocol=") + protocol + " ";
    glob_t g{};
    bool found = false;
    if (glob(globPattern.c_str(), 0, nullptr, &g) == 0)
    {
      for (size_t i = 0; i < g.gl_pathc && !found; ++i)
      {
        std::ifstream f(g.gl_pathv[i]);
        std::string line;
        while (std::getline(f, line))
        {
          if (line.find(needle) != std::string::npos) { found = true; break; }
        }
      }
    }
    globfree(&g);
    return found;
  }

  // Remove any log files left over from a previous run so scraping only sees the current run.
  static void RemoveGlobbedFiles(const std::string& globPattern)
  {
    glob_t g{};
    if (glob(globPattern.c_str(), 0, nullptr, &g) == 0)
      for (size_t i = 0; i < g.gl_pathc; ++i) remove(g.gl_pathv[i]);
    globfree(&g);
  }

  // Force a lightweight NET loopback for P2P+SHM-disabled tests. IB-CAST on this path treats
  // each GPU as its own node and ibv_create_qp fails (EAGAIN); GIN still opens IB QPs even when
  // NCCL_NET=Socket. Socket + GIN off still goes through net.cc staging-buffer allocation.
  struct NetLoopbackEnvBackup {
    bool        hadNet           = false;
    bool        hadGinEnable     = false;
    bool        hadMinNchannels  = false;
    bool        hadMaxNchannels  = false;
    std::string net;
    std::string ginEnable;
    std::string minNchannels;
    std::string maxNchannels;
  };

  static NetLoopbackEnvBackup netLoopbackEnvBackup;

  static void PinNetLoopbackTransport()
  {
    if (const char* v = ::getenv("NCCL_NET")) {
      netLoopbackEnvBackup.hadNet = true;
      netLoopbackEnvBackup.net    = v;
    } else {
      netLoopbackEnvBackup.hadNet = false;
      netLoopbackEnvBackup.net.clear();
    }

    if (const char* v = ::getenv("NCCL_GIN_ENABLE")) {
      netLoopbackEnvBackup.hadGinEnable = true;
      netLoopbackEnvBackup.ginEnable    = v;
    } else {
      netLoopbackEnvBackup.hadGinEnable = false;
      netLoopbackEnvBackup.ginEnable.clear();
    }

    if (const char* v = ::getenv("NCCL_MIN_NCHANNELS")) {
      netLoopbackEnvBackup.hadMinNchannels = true;
      netLoopbackEnvBackup.minNchannels    = v;
    } else {
      netLoopbackEnvBackup.hadMinNchannels = false;
      netLoopbackEnvBackup.minNchannels.clear();
    }

    if (const char* v = ::getenv("NCCL_MAX_NCHANNELS")) {
      netLoopbackEnvBackup.hadMaxNchannels = true;
      netLoopbackEnvBackup.maxNchannels    = v;
    } else {
      netLoopbackEnvBackup.hadMaxNchannels = false;
      netLoopbackEnvBackup.maxNchannels.clear();
    }

    ::setenv("NCCL_NET", "Socket", 1);
    ::setenv("NCCL_GIN_ENABLE", "0", 1);
    ::setenv("NCCL_MIN_NCHANNELS", "1", 1);
    ::setenv("NCCL_MAX_NCHANNELS", "2", 1);
  }

  static void UnpinNetLoopbackTransport()
  {
    if (netLoopbackEnvBackup.hadMaxNchannels)
      ::setenv("NCCL_MAX_NCHANNELS", netLoopbackEnvBackup.maxNchannels.c_str(), 1);
    else
      ::unsetenv("NCCL_MAX_NCHANNELS");

    if (netLoopbackEnvBackup.hadMinNchannels)
      ::setenv("NCCL_MIN_NCHANNELS", netLoopbackEnvBackup.minNchannels.c_str(), 1);
    else
      ::unsetenv("NCCL_MIN_NCHANNELS");

    if (netLoopbackEnvBackup.hadGinEnable)
      ::setenv("NCCL_GIN_ENABLE", netLoopbackEnvBackup.ginEnable.c_str(), 1);
    else
      ::unsetenv("NCCL_GIN_ENABLE");

    if (netLoopbackEnvBackup.hadNet)
      ::setenv("NCCL_NET", netLoopbackEnvBackup.net.c_str(), 1);
    else
      ::unsetenv("NCCL_NET");
  }

  TEST(SendRecv, SinglePairs)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& testDataTypes   = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024, 0};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;

    OptionalColArgs options;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }
    if (testBed.ev.maxGpus < 2) {
      GTEST_SKIP() << "Skipping... SendRecv.SinglePairs requires at least 2 GPUs (detected " << testBed.ev.maxGpus << ")";
    }

    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank; //self sendrecv group has two coll
          int recvId      = sendRank == recvRank; //where recv will be second coll
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            //set up the collArg slot to make sure AllocateMem is called once and correctly
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            TEST_INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);
          options.root = sendRank;

          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  // Shared sweep body for the LL128 latency-bound P2P send/recv regression tests.
  //
  // RCCL's P2P send/recv uses the LL128 protocol (in place of the legacy LL) for per-channel
  // payloads <= the active latency threshold (NCCL_P2P_LL128_THRESHOLD for the LL128 path,
  // NCCL_P2P_LL_THRESHOLD for the legacy LL path), and SIMPLE above it. The switch requires converting between
  // LL128 "wire" chunk sizes (data lines carry an extra flag word) and "data" chunk sizes in the
  // enqueue path, plus matching proxy byte accounting. Those conversions are size-sensitive, so a
  // bug typically corrupts data only at specific sizes near the protocol boundary or at sizes that
  // are not a multiple of the LL128 line/data granularity.
  //
  // This sweeps awkward element counts straddling the LL128/SIMPLE threshold using datatypes of
  // different element sizes (so the per-channel byte payload crosses the threshold at different
  // element counts, exercising both protocol selections and the wire<->data conversion), and
  // validates end-to-end correctness for every send/recv pair. The caller disables the P2P and SHM
  // transports (disabling P2P alone routes via SHM; disabling both falls through to NET/IB) and pins
  // NCCL_ALLOC_P2P_NET_LL_BUFFERS to 0 or 1 to exercise both branches of the net.cc LL128 staging-
  // buffer allocation for correctness.
  //
  // Protocol selection is asserted (not just data correctness, which SIMPLE also satisfies) by
  // scraping the NCCL_DEBUG=INFO protocol line. Protocol is chosen per channel (payload <=
  // nChannels * <threshold>), so the caller pins NCCL_MAX_P2P_NCHANNELS=1 (single channel) and pins
  // the relevant threshold env var (NCCL_P2P_LL128_THRESHOLD or NCCL_P2P_LL_THRESHOLD) to 16384 so
  // the 16 KiB boundary the element counts below straddle is deterministic and independent of the
  // (differing) default threshold values.
  //
  // The caller pins the transport + NCCL_ALLOC_P2P_NET_LL_BUFFERS to make one latency protocol the
  // deterministic choice for below-threshold sizes, and passes the matching `expect`:
  //   - ExpectProto::LL128 (gfx942/gfx950, ll128Enabled=1, flag=1, NET path): below-threshold ->
  //       LL128, above -> SIMPLE; legacy LL must NOT appear. Skipped on archs without the LL128
  //       send/recv kernel.
  //   - ExpectProto::LegacyLL (intranode path, flag=0, any arch): below-threshold -> legacy LL,
  //       above -> SIMPLE; LL128 must NOT appear. Pins the default latency path (and its restored
  //       LL wire<->data / proxy byte conversions), which correctness alone can't distinguish from a
  //       silent fallback to SIMPLE.
  //   - ExpectProto::NoLL128 (NET path, flag=0): the LL128 send/recv kernel must never be selected
  //       when the flag is off, and above-threshold sizes select SIMPLE. Below-threshold may be
  //       legacy LL or SIMPLE depending on whether the NET connection allocated an LL staging buffer
  //       (shared connections do not; non-shared ones do), so that is intentionally not constrained.
  enum class ExpectProto { LegacyLL, LL128, NoLL128 };
  static void RunLL128BoundarySweep(TestBed& testBed, const std::string& debugGlob, ExpectProto expect)
  {
    // Configuration: int8 (1B) and float32 (4B) so the caller-pinned 16 KiB/channel latency threshold
    // is crossed at 16384 and 4096 elements respectively. With NCCL_MAX_P2P_NCHANNELS=1 (single
    // channel) the per-channel threshold is exactly 16384 bytes, so below-threshold sizes pick the
    // latency protocol and above-threshold sizes pick SIMPLE deterministically:
    //   int8:  16385 B  -> SIMPLE, 4097 B    -> LL128, 1 B -> LL128
    //   fp32:  1 MiB     -> SIMPLE, 65540 B   -> SIMPLE, 16388 B -> SIMPLE, 4 B -> LL128
    std::vector<ncclDataType_t> const& testDataTypes = {ncclInt8, ncclFloat32};
    // Awkward counts around the boundary (non-power-of-2 / prime), plus a clearly-SIMPLE large size.
    // Kept small because these variants force transfers over the (slower) network loopback path.
    std::vector<int>            const  numElements   = {262144, 16385, 4097, 1};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

    OptionalColArgs options;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }
    if (testBed.ev.maxGpus < 2) {
      GTEST_SKIP() << "Skipping... LL128 boundary sweep requires at least 2 GPUs (detected "
                   << testBed.ev.maxGpus << ")";
    }
    if (expect == ExpectProto::LL128 && !DeviceSupportsLL128SendRecv()) {
      GTEST_SKIP() << "Skipping... LL128 P2P send/recv is only enabled on gfx942/gfx950.";
    }

    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    // Use exactly one rank per GPU so every send/recv pair is cross-GPU and (with P2P + SHM
    // disabled) travels the network transport. Multiple ranks per GPU would place some pairs on the
    // same device, whose intranode connections always allocate the LL128 staging buffer regardless
    // of NCCL_ALLOC_P2P_NET_LL_BUFFERS -- that would let LL128 be selected even in the disabled case
    // and defeat the enabled-vs-disabled protocol assertion below.
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = 1;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);
      if (::testing::Test::HasFatalFailure()) return;

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank; //self sendrecv group has two coll
          int recvId      = sendRank == recvRank; //where recv will be second coll
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            TEST_INFO("%s Datatype: %s LL128 boundary SendReceive Rank %d -> Rank %d for %d Elements",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);
          options.root = sendRank;

          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();

    // Finalize() has stopped the child processes, flushing and closing their NCCL_DEBUG_FILE logs;
    // scrape them to confirm the intended protocols were actually selected.
    // "LL " (trailing space in the helper) matches only the legacy-LL line, never "LL128 ".
    bool const sawLL     = DebugLogsContainProtocol(debugGlob, "LL");
    bool const sawLL128  = DebugLogsContainProtocol(debugGlob, "LL128");
    bool const sawSimple = DebugLogsContainProtocol(debugGlob, "Simple");
    switch (expect) {
      case ExpectProto::LL128:
        // Enabled on gfx942/gfx950: below-threshold sizes must select LL128 (a silent fallback to
        // SIMPLE would still pass the correctness checks above) and above-threshold must select SIMPLE.
        EXPECT_TRUE(sawLL128) << "Expected LL128 to be selected for latency-bound (below-threshold) "
                                 "send/recv with NCCL_ALLOC_P2P_NET_LL_BUFFERS=1, but no LL128 protocol "
                                 "log line was found (silent fallback to SIMPLE?).";
        EXPECT_FALSE(sawLL) << "Legacy LL must not be selected when the LL128 send/recv kernel is active.";
        EXPECT_TRUE(sawSimple) << "Expected the above-threshold size to select SIMPLE, but no SIMPLE "
                                  "protocol log line was found.";
        break;
      case ExpectProto::LegacyLL:
        // Default intranode path: below-threshold sizes must select the legacy LL protocol (a silent
        // fallback to SIMPLE would still pass correctness) and above-threshold must select SIMPLE.
        EXPECT_TRUE(sawLL) << "Expected legacy LL to be selected for latency-bound (below-threshold) "
                              "intranode send/recv, but no LL protocol log line was found (silent "
                              "fallback to SIMPLE, or broken LL wire<->data conversion?).";
        EXPECT_FALSE(sawLL128) << "LL128 must not be selected when NCCL_ALLOC_P2P_NET_LL_BUFFERS=0.";
        EXPECT_TRUE(sawSimple) << "Expected the above-threshold size to select SIMPLE, but no SIMPLE "
                                  "protocol log line was found.";
        break;
      case ExpectProto::NoLL128:
        // Disabled: the LL128 kernel must not be selected when the flag is off (or on an unsupported
        // arch); a regression here would silently use LL128. Above-threshold sizes select SIMPLE.
        // (Below-threshold may be legacy LL or SIMPLE depending on NET buffer allocation -- see enum.)
        EXPECT_FALSE(sawLL128) << "LL128 must not be selected when NCCL_ALLOC_P2P_NET_LL_BUFFERS=0 (or "
                                  "on an arch without the LL128 send/recv kernel).";
        EXPECT_TRUE(sawSimple) << "Expected SIMPLE to be selected on the NET path, but no SIMPLE "
                                  "protocol log line was found.";
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // P2P send/recv protocol-selection matrix (host gate in enqueue.cc). LL128 is selected only when
  // ALL of the following hold; otherwise the legacy LL kernel is used (SIMPLE above threshold):
  //
  //     useLL128SendRecv = defined(ENABLE_LL128)          // compile-time (HIP >= 6.1.33591)
  //                        && comm->topo->ll128Enabled    // comm LL128 gate (RCCL_LL128_FORCE_ENABLE)
  //                        && comm->allocP2pNetLLBuffers  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
  //                        && (cudaArch == 940 || 950);   // gfx942 / gfx950 only
  //
  // ll128Enabled is required so P2P stays consistent with the comm's collective protocol choice: if
  // LL128 is not enabled for the comm, send/recv must not use it even with the opt-in flag set. For
  // an op whose staging buffer is present, below-threshold picks the latency protocol and
  // above-threshold picks SIMPLE:
  //     below: useLL128SendRecv ? LL128 (reg=1) : legacy LL (reg=0)
  //     above: SIMPLE (under whichever kernel variant was selected)
  //
  //  ENABLE_LL128 | ll128Enabled | ALLOC_FLAG | size  | protocol  | kernel | covered by
  //  ------------ | ------------ | ---------- | ----- | --------- | ------ | ----------------------------
  //   on          | on           | 1          | below | LL128     | reg=1  | LL128NetBuffersEnabled
  //   on          | on           | 1          | above | SIMPLE    | reg=1  | LL128NetBuffersEnabled
  //   on          | on           | 0          | below | legacy LL | reg=0  | Ll128EnabledDoesNotEnableP2pLL128
  //   on          | on           | 0          | above | SIMPLE    | reg=0  | Ll128EnabledDoesNotEnableP2pLL128
  //   on          | off          | 1          | below | legacy LL | reg=0  | Ll128DisabledDoesNotEnableP2pLL128
  //   on          | off          | 1          | above | SIMPLE    | reg=0  | Ll128DisabledDoesNotEnableP2pLL128
  //   on          | off          | 0          | below | legacy LL | reg=0  | LegacyLLIntranode, LL128NetBuffersDisabled
  //   on          | off          | 0          | above | SIMPLE    | reg=0  | LegacyLLIntranode, LL128NetBuffersDisabled
  //   off         | any          | any        | below | legacy LL | reg=0  | LL128NetBuffersEnabled (#ifdef-guarded)
  //   off         | any          | any        | above | SIMPLE    | reg=0  | LL128NetBuffersEnabled (#ifdef-guarded)
  //
  // Non-gfx942/950 archs behave like the ENABLE_LL128=off rows (useLL128SendRecv is always false);
  // the LL128 cases are skipped there via DeviceSupportsLL128SendRecv().
  // ---------------------------------------------------------------------------

  // Default intranode path (P2P/SHM left enabled), NCCL_ALLOC_P2P_NET_LL_BUFFERS=0: latency-bound
  // send/recv selects the legacy LL protocol (useLL128SendRecv is false on every arch when the flag
  // is off), above-threshold selects SIMPLE. This pins the default P2P latency path and its restored
  // LL wire<->data and proxy byte conversions -- a silent LL->SIMPLE fallback (or a conversion bug
  // that still yields correct-but-SIMPLE transfers) is invisible to correctness alone. Runs on any
  // arch (the legacy LL kernel is built everywhere). See RunLL128BoundarySweep.
  TEST(SendRecv, LegacyLLIntranode)
  {
    // P2P and SHM left ENABLED so cross-GPU pairs use intranode connections, which always allocate
    // the LL staging buffer -> legacy LL is the deterministic below-threshold choice.
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0", 1); // force useLL128SendRecv=false even on gfx942/gfx950
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL_THRESHOLD", "16384", 1);     // pin legacy-LL threshold to the 16 KiB boundary the sweep straddles
    std::string const debugGlob = "/tmp/rccl_legacy_ll_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_legacy_ll_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::LegacyLL);
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
  }

  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=0: exercises the branch of the net.cc staging-buffer allocation
  // that does not add the dedicated LL128 buffer for shared p2p NET connections. The legacy LL
  // send/recv kernel is used; the key invariant is that LL128 is NEVER selected when the flag is off.
  // Verifies end-to-end correctness and that LL128 is not selected. See RunLL128BoundarySweep.
  TEST(SendRecv, LL128NetBuffersDisabled)
  {
    setenv("NCCL_P2P_DISABLE", "1", 1);              // disable P2P/IPC so send/recv does not use it
    setenv("NCCL_SHM_DISABLE", "1", 1);              // disable SHM so send/recv falls through to NET
    PinNetLoopbackTransport();
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0", 1); // disabled case
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL_THRESHOLD", "16384", 1);     // legacy-LL threshold governs when the LL128 path is off
    // Capture the per-op protocol selection so we can assert LL128/SIMPLE were used (see helper).
    std::string const debugGlob = "/tmp/rccl_ll128_disabled_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_ll128_disabled_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::NoLL128);
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
    UnpinNetLoopbackTransport();
    unsetenv("NCCL_SHM_DISABLE");
    unsetenv("NCCL_P2P_DISABLE");
  }

  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 with LL128 enabled for the comm: exercises the branch that
  // allocates a dedicated LL/LL128 staging buffer for shared p2p NET connections (device memory +
  // GDR), enabling LL128 for latency-bound send/recv over the network. Verifies correctness of the
  // changed net allocation + LL128 dispatch, and that LL128 is selected for latency-bound send/recv.
  // RCCL_LL128_FORCE_ENABLE=1 is required because ll128Enabled defaults off on gfx942/gfx950 and the
  // gate now requires it (see the matrix above). See RunLL128BoundarySweep.
  TEST(SendRecv, LL128NetBuffersEnabled)
  {
    setenv("NCCL_P2P_DISABLE", "1", 1);              // disable P2P/IPC so send/recv does not use it
    setenv("NCCL_SHM_DISABLE", "1", 1);              // disable SHM so send/recv falls through to NET
    PinNetLoopbackTransport();
    setenv("RCCL_LL128_FORCE_ENABLE", "1", 1);       // ll128Enabled=true (required by the P2P LL128 gate)
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "1", 1); // enabled case
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL128_THRESHOLD", "16384", 1);  // pin LL128 threshold to the 16 KiB boundary the sweep straddles
    // Capture the per-op protocol selection so we can assert LL128 was actually chosen (see helper).
    std::string const debugGlob = "/tmp/rccl_ll128_enabled_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_ll128_enabled_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
#if defined(ENABLE_LL128)
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::LL128);
#else
      // Built without LL128 support (HIP < 6.1.33591): the reg=1 kernel is not generated, so the
      // host gate forces useLL128SendRecv=false even with the flag on and the arch match -- LL128
      // must never be selected (it would resolve to a null/trap device slot). See enqueue.cc gate.
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::NoLL128);
#endif
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL128_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
    unsetenv("RCCL_LL128_FORCE_ENABLE");
    UnpinNetLoopbackTransport();
    unsetenv("NCCL_SHM_DISABLE");
    unsetenv("NCCL_P2P_DISABLE");
  }

  // Both the P2P opt-in flag AND the comm's ll128Enabled gate are required for P2P LL128. This pins
  // the "ALLOC_FLAG=1 but ll128Enabled=0" rows: with the flag on but ll128Enabled off (its default
  // on gfx942/gfx950), latency-bound send/recv must fall back to legacy LL (reg=0), above-threshold
  // SIMPLE, and LL128 must never appear. Uses the intranode path (P2P/SHM enabled) so the LL staging
  // buffer is always present and legacy LL is the deterministic below-threshold choice. See
  // RunLL128BoundarySweep.
  TEST(SendRecv, Ll128DisabledDoesNotEnableP2pLL128)
  {
    // ll128Enabled left at its gfx942/gfx950 default (off): RCCL_LL128_FORCE_ENABLE is NOT set.
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "1", 1); // P2P opt-in ON, but ll128Enabled off -> no LL128
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL_THRESHOLD", "16384", 1);     // legacy-LL threshold governs the fallback path
    std::string const debugGlob = "/tmp/rccl_ll128_disabled_gate_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_ll128_disabled_gate_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::LegacyLL);
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
  }

  // Both conditions required (mirror of the above): forcing ll128Enabled on with
  // RCCL_LL128_FORCE_ENABLE=1 must NOT by itself make send/recv use LL128 while
  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=0. Same intranode setup as LegacyLLIntranode but with ll128Enabled
  // forced on: below-threshold must still be legacy LL (reg=0), above-threshold SIMPLE, and LL128
  // must never appear -- proving the P2P opt-in flag is also required. Covers the "ll128Enabled=on,
  // ALLOC_FLAG=0" rows of the matrix above. Runs on any arch. See RunLL128BoundarySweep.
  TEST(SendRecv, Ll128EnabledDoesNotEnableP2pLL128)
  {
    // P2P and SHM left ENABLED so cross-GPU pairs use intranode connections (deterministic legacy LL
    // below-threshold). RCCL_LL128_FORCE_ENABLE flips comm->topo->ll128Enabled, but with the flag
    // below off useLL128SendRecv stays false.
    setenv("RCCL_LL128_FORCE_ENABLE", "1", 1);       // force comm->topo->ll128Enabled = true
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0", 1); // P2P LL128 opt-in OFF -> useLL128SendRecv=false
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL_THRESHOLD", "16384", 1);     // legacy-LL threshold governs the fallback path
    std::string const debugGlob = "/tmp/rccl_ll128_indep_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_ll128_indep_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::LegacyLL);
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
    unsetenv("RCCL_LL128_FORCE_ENABLE");
  }

  // ---------------------------------------------------------------------------
  // Independent latency thresholds: the LL128 path is governed solely by
  // NCCL_P2P_LL128_THRESHOLD and the legacy LL path solely by NCCL_P2P_LL_THRESHOLD. The two knobs
  // have different defaults (legacy LL 4 KiB, LL128 16 KiB) because legacy LL's ~2x wire overhead
  // stops beating SIMPLE at smaller sizes than LL128's much lower per-line flag overhead. The two
  // tests below cross the knobs (set the OTHER protocol's threshold to 0) and confirm the active
  // path still selects its latency protocol below its OWN 16 KiB threshold -- proving neither path
  // reads the other's env var. See RunLL128BoundarySweep.
  // ---------------------------------------------------------------------------

  // LL128 path reads NCCL_P2P_LL128_THRESHOLD only: force NCCL_P2P_LL_THRESHOLD=0 (which would push
  // legacy LL to SIMPLE at every size) while keeping NCCL_P2P_LL128_THRESHOLD=16384. LL128 must
  // still be selected below 16 KiB, proving the LL128 path ignores the legacy-LL knob. Same NET +
  // ll128Enabled setup as LL128NetBuffersEnabled. Skipped on archs without the LL128 send/recv
  // kernel (via ExpectProto::LL128).
  TEST(SendRecv, SeparateThresholdLL128UsesLL128Knob)
  {
    setenv("NCCL_P2P_DISABLE", "1", 1);              // disable P2P/IPC so send/recv does not use it
    setenv("NCCL_SHM_DISABLE", "1", 1);              // disable SHM so send/recv falls through to NET
    PinNetLoopbackTransport();
    setenv("RCCL_LL128_FORCE_ENABLE", "1", 1);       // ll128Enabled=true (required by the P2P LL128 gate)
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "1", 1); // enable the LL128 staging buffer
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL_THRESHOLD", "0", 1);         // legacy-LL knob at 0: must NOT affect the LL128 path
    setenv("NCCL_P2P_LL128_THRESHOLD", "16384", 1);  // LL128 knob governs -> LL128 below 16 KiB
    std::string const debugGlob = "/tmp/rccl_sep_ll128_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_sep_ll128_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
#if defined(ENABLE_LL128)
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::LL128);
#else
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::NoLL128);
#endif
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL128_THRESHOLD");
    unsetenv("NCCL_P2P_LL_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
    unsetenv("RCCL_LL128_FORCE_ENABLE");
    UnpinNetLoopbackTransport();
    unsetenv("NCCL_SHM_DISABLE");
    unsetenv("NCCL_P2P_DISABLE");
  }

  // Legacy LL path reads NCCL_P2P_LL_THRESHOLD only: force NCCL_P2P_LL128_THRESHOLD=0 (irrelevant on
  // this path) while keeping NCCL_P2P_LL_THRESHOLD=16384. Legacy LL must still be selected below
  // 16 KiB, proving the legacy-LL path ignores the LL128 knob. Same intranode setup as
  // LegacyLLIntranode (NCCL_ALLOC_P2P_NET_LL_BUFFERS=0 -> useLL128SendRecv=false on every arch).
  TEST(SendRecv, SeparateThresholdLegacyLLUsesLLKnob)
  {
    setenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS", "0", 1); // force useLL128SendRecv=false even on gfx942/gfx950
    setenv("NCCL_MAX_P2P_NCHANNELS", "1", 1);        // single channel -> deterministic per-channel threshold
    setenv("NCCL_P2P_LL128_THRESHOLD", "0", 1);      // LL128 knob at 0: must NOT affect the legacy-LL path
    setenv("NCCL_P2P_LL_THRESHOLD", "16384", 1);     // legacy-LL knob governs -> legacy LL below 16 KiB
    std::string const debugGlob = "/tmp/rccl_sep_legacy_ll_" + std::to_string(getpid()) + ".*";
    RemoveGlobbedFiles(debugGlob);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_DEBUG_FILE", ("/tmp/rccl_sep_legacy_ll_" + std::to_string(getpid()) + ".%p").c_str(), 1);
    {
      TestBed testBed;
      RunLL128BoundarySweep(testBed, debugGlob, ExpectProto::LegacyLL);
    }
    RemoveGlobbedFiles(debugGlob);
    unsetenv("NCCL_DEBUG_FILE");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_DEBUG");
    unsetenv("NCCL_P2P_LL_THRESHOLD");
    unsetenv("NCCL_P2P_LL128_THRESHOLD");
    unsetenv("NCCL_MAX_P2P_NCHANNELS");
    unsetenv("NCCL_ALLOC_P2P_NET_LL_BUFFERS");
  }

  TEST(SendRecv, UserBufferRegister)
  {
    setenv("RCCL_ENABLE_INTRANET", "1", 1);
    TestBed testBed;

    // Configuration
    std::vector<ncclDataType_t> const& testDataTypes   = {ncclInt32, ncclFloat16, ncclFloat64};
    std::vector<int>            const  numElements     = {1048576, 53327, 1024};
    bool                        const  inPlace         = false;
    bool                        const  useManagedMem   = false;
    bool                        const  userRegistered  = true;

    OptionalColArgs options;

    std::vector<ncclDataType_t> dataTypes;
    testBed.GetSupportedDataTypes(dataTypes, testDataTypes);
    if (dataTypes.empty()) {
      GTEST_SKIP() << "Skipping... test datatypes excluded by UT_DATATYPES.";
    }

    bool isCorrect = true;
    int numGpus = testBed.ev.maxGpus;
    for (int rpg=0; rpg < 2 && isCorrect; ++rpg)
    for (int isMultiProcess = 0; isMultiProcess <= 1 && isCorrect; ++isMultiProcess)
    {
      if (!(testBed.ev.processMask & (1 << isMultiProcess))) continue;
      int ranksPerGpu = rpg == 0 ? 1 : testBed.ev.maxRanksPerGpu;
      int totalRanks = numGpus * ranksPerGpu;
      int const numProcesses = isMultiProcess ? numGpus : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, numGpus, ranksPerGpu, gpuPriorityOrder),
                        {1,2}, //two group, second group sendrecv to self, has 2 coll
                        testBed.GetNumStreamsPerGroup(1,2),
                        2);

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      for (int numIdx = 0; numIdx < numElements.size() && isCorrect; ++numIdx)
      for (int sendRank = 0; sendRank < totalRanks; ++sendRank)
      {
        for (int recvRank = 0; recvRank  < totalRanks; ++recvRank)
        {
          options.root = recvRank;
          int groupCallId = sendRank == recvRank;
          int recvId      = sendRank == recvRank;
          testBed.SetCollectiveArgs(ncclCollSend,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    0,
                                    groupCallId,
                                    sendRank);
          if (recvRank == 0)
          {
            testBed.SetCollectiveArgs(ncclCollSend,
                                      dataTypes[dataIdx],
                                      numElements[numIdx],
                                      numElements[numIdx],
                                      options,
                                      0,
                                      !groupCallId,
                                      sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 0, 0, sendRank, userRegistered);
            testBed.PrepareData(0, 0, sendRank);
            testBed.AllocateMem(inPlace, useManagedMem, 1, 0, sendRank, userRegistered);
            testBed.PrepareData(1, 0, sendRank);
          }

          if (testBed.ev.showNames) // Show test names
            TEST_INFO("%s Datatype: %s SendReceive test Rank %d -> Rank %d for %d Elements",
                 isMultiProcess ? "MP" : "SP",
                 ncclDataTypeNames[dataTypes[dataIdx]],
                 sendRank,
                 recvRank,
                 numElements[numIdx]);

          options.root = sendRank;
          testBed.SetCollectiveArgs(ncclCollRecv,
                                    dataTypes[dataIdx],
                                    numElements[numIdx],
                                    numElements[numIdx],
                                    options,
                                    recvId,
                                    groupCallId,
                                    recvRank);
          testBed.AllocateMem(inPlace, useManagedMem, groupCallId, recvId, recvRank, userRegistered);
          testBed.PrepareData(groupCallId, recvId, recvRank);
          testBed.ExecuteCollectives({sendRank, recvRank}, groupCallId);
          testBed.ValidateResults(isCorrect, groupCallId, recvId, recvRank);
          testBed.DeallocateMem(groupCallId, recvId, recvRank);
        }
        testBed.DeallocateMem(0, 0, sendRank);
        testBed.DeallocateMem(1, 0, sendRank);
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
    unsetenv("RCCL_ENABLE_INTRANET");
  }
}
