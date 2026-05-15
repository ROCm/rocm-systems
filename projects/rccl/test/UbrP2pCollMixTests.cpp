/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for the upstream-NCCL bug tracked as
// NVIDIA/nccl issue #1859 (fixed in NCCL v2.29.2):
//   "user-buffer p2p + coll rmtAddr nullptr error"
//
// Bug summary: when the same user buffer is registered via ncclCommRegister
// and used first for a P2P (Send/Recv) op and then for a collective
// (AllReduce/AllGather/...), the device array
// regRecord->regIpcAddrs.devPeerRmtAddrs is not built up.  The first call
// (NCCL_IPC_SENDRECV) populates only the host-side hostPeerRmtAddrs and
// sets needUpdate=true, but the dev-side calloc+memcpy is gated on
// `type == NCCL_IPC_COLLECTIVE` so it is skipped.  The second call
// (NCCL_IPC_COLLECTIVE) hits the "reuse" branch (ipcInfos[peer] already
// non-NULL), short-circuits with regBufFlag=1 and needUpdate=false,
// leaving devPeerRmtAddrs == NULL.  The collective kernel then derefs
// NULL.  See: https://github.com/NVIDIA/nccl/issues/1859 (NVIDIA/nccl).
//
// Repro shape on AMD/RCCL: the buggy lines live on the cross-process
// IPC register path (`ipcRegisterBuffer` in src/transport/p2p.cc).
// ncclCommInitAll in a single process takes the same-process directMode
// shortcut and never enters that path; we therefore fork one child per
// GPU and use ncclCommInitRank, mirroring the existing
// SendRecv.UserBufferRegister test in this tree.  In each child:
//   1) ncclCommRegister(comm, dbuf, ...)       (NCCL_LOCAL_REGISTER=1)
//   2) ncclSend / ncclRecv on dbuf  -> fills hostPeerRmtAddrs but
//      leaves devPeerRmtAddrs NULL on the buggy build.
//   3) ncclAllReduce on the SAME dbuf -> hits the IPC reuse branch and,
//      pre-fix, dereferences NULL devPeerRmtAddrs.
// Pre-fix: SIGSEGV / wrong result in step 3.
// Post-fix: ncclSuccess and numerically correct output.
//
// Status on RCCL @ NCCL-v2.28.9 pin (current baseline):
//   The cross-process IPC code path that contains the buggy lines
//   (`ipcRegisterBuffer` in src/transport/p2p.cc) is not actually entered
//   on AMD with the current HIP runtime: the legacy hipIpc path falls
//   through to a same-process direct shortcut and `ncclParamLocalRegister`
//   short-circuits before reaching the buggy reuse branch.  Coverage
//   measurement confirms `ipcRegisterBuffer` shows 0 hits today, so this
//   test PASSES on the buggy build as well.
//
//   This test is therefore a forward-looking guard: when a future change
//   (e.g. enabling cuMem* on AMD with HIP >= 7.1, the upcoming
//   nccl-sync-v2-29 merge, or any tweak that routes the ncclCommRegister
//   path through `ipcRegisterBuffer`) starts exercising the buggy code,
//   this test will SIGSEGV unless the NCCL v2.29.2 fix is also present.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>

#if defined(RCCL_TEST_CODE_COVERAGE)
extern "C" int  __llvm_profile_write_file(void);
extern "C" void __llvm_profile_set_filename(const char* name);
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/ErrCode.hpp"
#include "common/StandaloneUtils.hpp"

namespace RcclUnitTesting {

namespace {

// Probe device count by forking a tiny child process: we cannot call any
// HIP API in the parent before fork()-ing the worker children, otherwise
// the children inherit a broken HIP context and crash inside hipSetDevice.
int probeDeviceCountViaFork() {
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }
    if (pid == 0) {
        close(pipefd[0]);
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess) n = -1;
        int a = 0, b = 0;
        if (n >= 2) {
            if (hipDeviceCanAccessPeer(&a, 0, 1) != hipSuccess) a = 0;
            if (hipDeviceCanAccessPeer(&b, 1, 0) != hipSuccess) b = 0;
            if (!(a && b)) n = 0;  // signal "no usable peers"
        }
        ssize_t w = write(pipefd[1], &n, sizeof(n));
        (void)w;
        close(pipefd[1]);
        std::_Exit(0);
    }
    close(pipefd[1]);
    int n = -1;
    ssize_t r = read(pipefd[0], &n, sizeof(n));
    (void)r;
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return n;
}

// In-child fatal: print a message to stderr and exit non-zero.  Using
// gtest assertions inside a forked child is fragile, so we mark failure
// via the exit code and let the parent ASSERT on it.
#define CHILD_REQUIRE(cond, msg)                                              \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "[child rank %d] %s (%s:%d)\n",              \
                         rank, (msg), __FILE__, __LINE__);                    \
            std::_Exit(10);                                                   \
        }                                                                    \
    } while (0)

#define CHILD_NCCL(cmd)                                                       \
    do {                                                                      \
        ncclResult_t r__ = (cmd);                                             \
        if (r__ != ncclSuccess) {                                             \
            std::fprintf(stderr, "[child rank %d] NCCL fail %s:%d: %s\n",     \
                         rank, __FILE__, __LINE__, ncclGetErrorString(r__));  \
            std::_Exit(11);                                                   \
        }                                                                    \
    } while (0)

#define CHILD_HIP(cmd)                                                        \
    do {                                                                      \
        hipError_t e__ = (cmd);                                               \
        if (e__ != hipSuccess) {                                              \
            std::fprintf(stderr, "[child rank %d] HIP fail %s:%d: %s\n",      \
                         rank, __FILE__, __LINE__, hipGetErrorString(e__));   \
            std::_Exit(12);                                                   \
        }                                                                    \
    } while (0)

// Body of one child rank.  Performs the two-step "p2p then coll on the
// same registered buffer" sequence and exits 0 on success.
[[noreturn]] void runChild(int rank, int nRanks, ncclUniqueId id) {
    // Force the IPC registration path that contains the bug:
    //   * NCCL_LOCAL_REGISTER=1: take the IPC reg branch in
    //     ncclRegisterP2pIpcBuffer / ncclRegisterCollBuffers.
    //   * NCCL_GRAPH_REGISTER=0: avoid the graph register branch; it is
    //     not what's under test and would confuse the analysis.
    //   * RCCL_ENABLE_INTRANET=1: matches the existing SendRecv.UBR test;
    //     forces the same-node IPC register path even for nearby GPUs.
    //   * NCCL_PROTO=Simple: the buggy branch is reached on the SIMPLE
    //     protocol p2p path.
    setenv("NCCL_LOCAL_REGISTER",  "1",      1);
    setenv("NCCL_GRAPH_REGISTER",  "0",      1);
    setenv("RCCL_ENABLE_INTRANET", "1",      1);
    setenv("NCCL_PROTO",           "Simple", 1);

    constexpr int kNumElems = 1024;
    const size_t bufBytes   = kNumElems * sizeof(int);

    CHILD_HIP(hipSetDevice(rank));
    hipStream_t stream;
    CHILD_HIP(hipStreamCreate(&stream));

    ncclComm_t comm;
    CHILD_NCCL(ncclCommInitRank(&comm, nRanks, id, rank));

    // The single user buffer reused for both the p2p and the collective.
    // dbuf[i] = rank * 1000 + i so each rank's contribution is identifiable.
    int* dbuf = nullptr;
    int* dscratch = nullptr;  // peer-side recv landing for the p2p step
    CHILD_HIP(hipMalloc(&dbuf,     bufBytes));
    CHILD_HIP(hipMalloc(&dscratch, bufBytes));

    std::vector<int> host(kNumElems);
    for (int i = 0; i < kNumElems; ++i) host[i] = rank * 1000 + i;
    CHILD_HIP(hipMemcpy(dbuf, host.data(), bufBytes, hipMemcpyHostToDevice));
    CHILD_HIP(hipMemset(dscratch, 0, bufBytes));
    CHILD_HIP(hipDeviceSynchronize());

    // Register the buffer that will be used by BOTH the p2p step and the
    // collective step.  This is what makes the buggy reuse branch fire.
    void* regHandle = nullptr;
    CHILD_NCCL(ncclCommRegister(comm, dbuf, bufBytes, &regHandle));
    CHILD_REQUIRE(regHandle != nullptr,
                  "ncclCommRegister returned NULL handle "
                  "(NCCL_LOCAL_REGISTER=1 was set)");

    // ---- Step 1: P2P swap on the registered buffer. -----------------------
    // Send dbuf -> peer's dscratch, Recv peer's dbuf -> our dscratch.
    // dbuf is the *registered* buffer -> exercises ipcRegisterBuffer with
    // type=NCCL_IPC_SENDRECV, populating hostPeerRmtAddrs.
    const int peer = (rank + 1) % nRanks;
    CHILD_NCCL(ncclGroupStart());
    CHILD_NCCL(ncclSend(dbuf,     kNumElems, ncclInt, peer, comm, stream));
    CHILD_NCCL(ncclRecv(dscratch, kNumElems, ncclInt, peer, comm, stream));
    CHILD_NCCL(ncclGroupEnd());
    CHILD_HIP(hipStreamSynchronize(stream));

    // Sanity-check the p2p result so we know step 1 actually ran.
    std::vector<int> got(kNumElems);
    CHILD_HIP(hipMemcpy(got.data(), dscratch, bufBytes, hipMemcpyDeviceToHost));
    for (int i = 0; i < kNumElems; ++i) {
        if (got[i] != peer * 1000 + i) {
            std::fprintf(stderr,
                "[child rank %d] p2p payload mismatch at %d: got %d want %d\n",
                rank, i, got[i], peer * 1000 + i);
            std::_Exit(20);
        }
    }

    // ---- Step 2: Collective on the SAME registered buffer. ----------------
    // In-place AllReduce so sendbuff == recvbuff == dbuf, the registered
    // buffer that already has IPC info recorded from the SENDRECV step.
    // This is the call that crashes pre-fix: ipcRegisterBuffer hits its
    // "reuse" branch (ipcInfos[peerLocal] != NULL), short-circuits with
    // regBufFlag=1/needUpdate=false, and the COLLECTIVE branch then reads
    // a NULL devPeerRmtAddrs.
    CHILD_NCCL(ncclAllReduce(dbuf, dbuf, kNumElems, ncclInt, ncclSum,
                             comm, stream));
    CHILD_HIP(hipStreamSynchronize(stream));

    // Expected: each element is the sum across all ranks of (r*1000 + i).
    long expected_const = 0;
    for (int r = 0; r < nRanks; ++r) expected_const += r * 1000;
    CHILD_HIP(hipMemcpy(got.data(), dbuf, bufBytes, hipMemcpyDeviceToHost));
    for (int i = 0; i < kNumElems; ++i) {
        const int want = static_cast<int>(expected_const + nRanks * i);
        if (got[i] != want) {
            std::fprintf(stderr,
                "[child rank %d] AllReduce mismatch at %d: got %d want %d "
                "(likely NVIDIA/nccl issue #1859 regression)\n",
                rank, i, got[i], want);
            std::_Exit(21);
        }
    }

    CHILD_NCCL(ncclCommDeregister(comm, regHandle));
    CHILD_HIP(hipFree(dbuf));
    CHILD_HIP(hipFree(dscratch));
    CHILD_HIP(hipStreamDestroy(stream));
    CHILD_NCCL(ncclCommDestroy(comm));
#if defined(RCCL_TEST_CODE_COVERAGE)
    // Flush both runtimes' coverage data before _Exit (which skips
    // atexit handlers).  Mirrors ProcessIsolatedTestRunner.
    using WriteFn = int (*)(void);
    __llvm_profile_write_file();
    if (auto libWrite = reinterpret_cast<WriteFn>(
            dlsym(RTLD_DEFAULT, "rcclCoverageWriteFile"))) {
        libWrite();
    }
#endif
    std::_Exit(0);
}

#undef CHILD_REQUIRE
#undef CHILD_NCCL
#undef CHILD_HIP

} // namespace

TEST(UbrP2pCollMix, RegBufferSendRecvThenAllReduce) {
    const int probedGpus = probeDeviceCountViaFork();
    if (probedGpus < 2) {
        GTEST_SKIP() << "Test requires >=2 GPUs with mutual peer access "
                        "(IPC path covers this regression). Probed: "
                     << probedGpus;
    }

    constexpr int kNumRanks = 2;

    // One pipe per rank to ship the unique id from rank 0 to the rest
    // (matches the pattern in test/common/CallCollectiveForked.cpp).
    std::vector<std::array<int, 2>> pipes(kNumRanks);
    for (int r = 0; r < kNumRanks; ++r) {
        ASSERT_EQ(pipe(pipes[r].data()), 0) << "pipe() failed for rank " << r;
    }

    std::vector<pid_t> kids(kNumRanks, 0);
    for (int r = 0; r < kNumRanks; ++r) {
        kids[r] = fork();
        ASSERT_NE(kids[r], -1) << "fork() failed for rank " << r;
        if (kids[r] == 0) {
            // Child.
#if defined(RCCL_TEST_CODE_COVERAGE)
            // Per-child profraw filename so children don't clobber each
            // other or the parent.  Same trick as ProcessIsolatedTestRunner:
            // the LLVM runtime resolves %p once in the parent, so we
            // substitute %p ourselves with the child PID.
            using SetFn = void (*)(const char*);
            const char* envPattern = std::getenv("LLVM_PROFILE_FILE");
            std::string pattern    = (envPattern && *envPattern)
                                         ? envPattern
                                         : std::string("ubr_p2p_coll_%m.profraw");
            if (pattern.find("%p") == std::string::npos) {
                auto dot = pattern.rfind('.');
                if (dot == std::string::npos) pattern += "_%p";
                else pattern.insert(dot, "_%p");
            }
            {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%d", (int)getpid());
                std::string out;
                for (size_t i = 0; i < pattern.size();) {
                    if (i + 1 < pattern.size() && pattern[i] == '%' &&
                        pattern[i + 1] == 'p') { out += buf; i += 2; }
                    else out.push_back(pattern[i++]);
                }
                pattern.swap(out);
            }
            __llvm_profile_set_filename(pattern.c_str());
            if (auto libSet = reinterpret_cast<SetFn>(
                    dlsym(RTLD_DEFAULT, "rcclCoverageSetFilename"))) {
                libSet(pattern.c_str());
            }
#endif
            ncclUniqueId id{};
            if (r == 0) {
                ncclResult_t res = ncclGetUniqueId(&id);
                if (res != ncclSuccess) std::_Exit(30);
                close(pipes[r][0]);
                ssize_t w = write(pipes[r][1], &id, sizeof(id));
                close(pipes[r][1]);
                if (w != static_cast<ssize_t>(sizeof(id))) std::_Exit(31);
            } else {
                close(pipes[r][1]);
                ssize_t got = read(pipes[r][0], &id, sizeof(id));
                close(pipes[r][0]);
                if (got != static_cast<ssize_t>(sizeof(id))) std::_Exit(32);
            }
            runChild(r, kNumRanks, id);
            // unreachable
        }
    }

    // Parent: read the id from rank 0 and forward it to the others.
    ncclUniqueId id{};
    close(pipes[0][1]);
    ssize_t got = read(pipes[0][0], &id, sizeof(id));
    close(pipes[0][0]);
    ASSERT_EQ(got, static_cast<ssize_t>(sizeof(id)))
        << "parent failed to read uniqueId from rank 0";
    for (int r = 1; r < kNumRanks; ++r) {
        close(pipes[r][0]);
        ssize_t w = write(pipes[r][1], &id, sizeof(id));
        close(pipes[r][1]);
        ASSERT_EQ(w, static_cast<ssize_t>(sizeof(id)))
            << "parent failed to forward uniqueId to rank " << r;
    }

    // Reap children and require all of them succeeded.  A non-zero exit
    // (in particular SIGSEGV from the NULL devPeerRmtAddrs deref) flags
    // the NVIDIA/nccl issue #1859 regression.
    for (int r = 0; r < kNumRanks; ++r) {
        int status = 0;
        pid_t w = waitpid(kids[r], &status, 0);
        ASSERT_EQ(w, kids[r]) << "waitpid failed for rank " << r;
        if (WIFSIGNALED(status)) {
            FAIL() << "child rank " << r << " killed by signal "
                   << WTERMSIG(status)
                   << " (likely NVIDIA/nccl issue #1859: NULL devPeerRmtAddrs deref "
                      "after p2p+coll on the same registered buffer)";
        }
        ASSERT_TRUE(WIFEXITED(status))
            << "child rank " << r << " did not exit normally";
        ASSERT_EQ(WEXITSTATUS(status), 0)
            << "child rank " << r << " exited with code "
            << WEXITSTATUS(status)
            << " (see child stderr above; non-zero indicates NVIDIA/nccl issue #1859 "
               "regression or a setup failure)";
    }
}

} // namespace RcclUnitTesting
