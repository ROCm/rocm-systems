/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Consistency tests for rcclGetCollImplInfo (AllReduce + AllGather + ReduceScatter).
//
// rcclGetCollImplInfo is the *reporting* path: it routes AR/AG/RS through
// rcclSelectAllReduce/rcclSelectAllGather/rcclSelectReduceScatter with query=true.
// Live dispatch routes the same functions with query=false and logs which backend
// it selected. These tests treat the dispatch log as ground truth: for each size we run the
// real collective, parse what the library logged as *selected* (algo / protocol),
// then call rcclGetCollImplInfo for the same operands and assert the *reported*
// algo/proto/channels agree. No assumption is made about which backend runs at a
// given size -- whatever the host picks, the report must match the log. Channels are
// asserted when the log states them (native-kernel tuning line): the report exposes
// the traffic-packed channel count, which must equal the per-op channel{Lo..Hi} range.
//
// Per-size isolation: NCCL_DEBUG_FILE is re-pointed to a fresh file for every
// size and ncclResetDebugInitInternal() forces the reopen, so each file holds
// exactly one collective's selection lines (line-buffered, so complete lines are
// on disk by the time we read).

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "StandaloneUtils.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/SymmetricBufferHelpers.hpp"  // RCCLTestHelpers::SymBuf (RAII deregister+free)
#include "rccl_common.h"  // rcclGetCollImplInfo, rcclSymKGetInfo, rcclGetAlgoName, rcclGetProtocolName, rcclAddonAlgos_t

// rccl_common.h drags in RCCL's internal NCCLCHECK (which `return`s). These tests
// live in void functions, so use a gtest-friendly, non-returning check instead.
#undef NCCLCHECK
#define NCCLCHECK(cmd)                                                       \
  do {                                                                       \
    ncclResult_t res_ = (cmd);                                              \
    ASSERT_EQ(res_, ncclSuccess) << "NCCL failure: " << ncclGetErrorString(res_); \
  } while (0)

namespace RcclUnitTesting
{
  // Internal (non-deprecated) debug reset: re-reads NCCL_DEBUG* env and reopens
  // NCCL_DEBUG_FILE on the next log. Exported from src/debug.cc.
  extern "C" void ncclResetDebugInitInternal();

  namespace
  {
    // What the dispatch log says was selected for a collective. proto/channels are
    // only set when the log states them literally.
    struct SelectedImpl
    {
      bool        found    = false;
      std::string algoName;              // e.g. "RING", "DDA", "Direct", "Hier", "RING*"
      bool        hasProto = false;
      std::string protoName;             // e.g. "LL", "LL128", "SIMPLE"
      bool        hasChannels = false;
      int         channels = 0;
    };

    std::string ReadFile(const std::string& path)
    {
      std::ifstream f(path);
      std::stringstream ss;
      ss << f.rdbuf();
      return ss.str();
    }

    // Returns the whitespace-delimited token that follows `key` in `line`, or "".
    std::string TokenAfter(const std::string& line, const std::string& key)
    {
      size_t p = line.find(key);
      if (p == std::string::npos) return "";
      p += key.size();
      while (p < line.size() && isspace((unsigned char)line[p])) p++;
      size_t e = p;
      while (e < line.size() && !isspace((unsigned char)line[e])) e++;
      return line.substr(p, e - p);
    }

    // Names as produced by rcclGetProtocolName / ncclProtoToString.
    std::string ProtoIdToName(int proto)
    {
      const char* n = nullptr;
      if (rcclGetProtocolName(proto, &n) == ncclSuccess && n) return n;
      return "";
    }

    // Decode the single selection recorded in a fresh per-size log. Addon-backend
    // lines take precedence over the native-kernel tuning line (addon backends
    // bypass or override the kernel plan). `func` is "AllReduce" / "AllGather" / "ReduceScatter".
    SelectedImpl ParseSelected(const std::string& log, const std::string& func)
    {
      SelectedImpl kernel;    // from enqueue.cc tuning line
      SelectedImpl addon;     // from a backend-specific line, if any

      std::istringstream iss(log);
      std::string        line;
      while (std::getline(iss, line))
      {
        // Canonical addon-backend selection line (CE / DDA / Direct / Hier /
        // symmetric): "<Func> impl selected: algo <NAME>". This is the only line
        // several addon backends emit (DDA IPC, CE registered, symmetric all run
        // through paths with no other recognizable log), so it is what makes the
        // report checkable on cumem systems. Algo only; proto/channels are not
        // computed on the dispatch path for these backends, so they stay unset
        // here and a later, more specific line (below) may refine the protocol.
        if (line.find(func + " impl selected: algo ") != std::string::npos)
        {
          addon = {true, TokenAfter(line, "algo "), false, "", false, 0};
        }

        if (func == "AllGather")
        {
          if (line.find("DDA fabric LL128") != std::string::npos)
          {
            addon = {true, "DDA", true, "LL128", false, 0};
          }
          else if (line.find("DDA fabric LL path") != std::string::npos)
          {
            addon = {true, "DDA", true, "LL", false, 0};
          }
          else if (line.find("DDA fabric (VMM)") != std::string::npos)
          {
            addon = {true, "DDA", false, "", false, 0};  // "VMM" is not an NCCL proto string
          }
          else if (line.find("RCCL DIRECT ALLGATHER") != std::string::npos)
          {
            addon = {true, "Direct", true, "SIMPLE", false, 0};
          }
          else if (line.find("Hierarchical AG inter") != std::string::npos)
          {
            // "Hierarchical AG inter: proto=%d channels=%d, ..."
            int proto = -1, chans = -1;
            size_t pp = line.find("proto=");
            size_t cp = line.find("channels=");
            if (pp != std::string::npos) proto = atoi(line.c_str() + pp + 6);
            if (cp != std::string::npos) chans = atoi(line.c_str() + cp + 9);
            addon = {true, "Hier", proto >= 0, ProtoIdToName(proto), chans >= 0, chans};
          }
        }

        if (func == "ReduceScatter")
        {
          // RS DDA fabric lines carry the protocol; Direct RS reports SIMPLE. These
          // refine the canonical "impl selected" line's protocol (channels unset:
          // RS DDA has no Blocks helper and Direct's p2pnChannels aren't logged).
          if (line.find("DDA fabric LL128") != std::string::npos)
          {
            addon = {true, "DDA", true, "LL128", false, 0};
          }
          else if (line.find("DDA fabric LL path") != std::string::npos)
          {
            addon = {true, "DDA", true, "LL", false, 0};
          }
          else if (line.find("DDA fabric (VMM)") != std::string::npos)
          {
            addon = {true, "DDA", false, "", false, 0};  // "VMM" is not an NCCL proto string
          }
          else if (line.find("RCCL DIRECT REDUCE-SCATTER") != std::string::npos)
          {
            addon = {true, "Direct", true, "SIMPLE", false, 0};
          }
        }

        // Native-kernel tuning line (rank 0):
        //   "<Func>: <bytes> Bytes -> Algo <A> proto <P> channel{Lo..Hi}={lo..hi}"
        if (line.find(func + ":") != std::string::npos &&
            line.find("Bytes -> Algo ") != std::string::npos)
        {
          std::string algo  = TokenAfter(line, "Algo ");
          std::string proto = TokenAfter(line, "proto ");
          int lo = -1, hi = -1;
          size_t cp = line.find("channel{Lo..Hi}={");
          if (cp != std::string::npos)
          {
            if (sscanf(line.c_str() + cp, "channel{Lo..Hi}={%d..%d}", &lo, &hi) != 2)
            {
              lo = hi = -1;
            }
          }
          kernel = {true, algo, !proto.empty(), proto,
                    (lo >= 0 && hi >= lo), (lo >= 0 && hi >= lo) ? hi - lo + 1 : 0};
        }
      }

      if (addon.found) return addon;

      // WarpSpeed self-describes on the tuning line as "RING*" with the physical
      // block count (enqueue.cc), so no post-processing is needed here -- the parsed
      // algo/channels already match what rcclGetCollImplInfo reports.
      return kernel;
    }

    // (algo, proto) reported by rcclGetCollImplInfo, decoded to names.
    struct ReportedImpl
    {
      int         algo = -1, proto = -1, channels = -1;
      std::string algoName, protoName;
    };

    ReportedImpl QueryReport(ncclComm_t comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dt,
                             ncclRedOp_t op, const void* sbuf, void* rbuf)
    {
      ReportedImpl r;
      ncclResult_t res = rcclGetCollImplInfo(comm, coll, count, dt, op, sbuf, rbuf,
                                             /*graphCapturing=*/0, &r.algo, &r.proto, &r.channels);
      EXPECT_EQ(res, ncclSuccess) << "rcclGetCollImplInfo failed: " << ncclGetErrorString(res);
      const char* an = nullptr;
      const char* pn = nullptr;
      if (rcclGetAlgoName(r.algo, &an) == ncclSuccess && an) r.algoName = an;
      if (rcclGetProtocolName(r.proto, &pn) == ncclSuccess && pn) r.protoName = pn;
      return r;
    }

    // (algo, proto) reported by rcclSymKGetInfo -- the reporter rccl-tests calls in
    // -R 2 (symmetric-window-registered) mode. Decoded to names like ReportedImpl.
    struct SymkReport
    {
      ncclResult_t res = ncclSuccess;
      int          algo = -1, proto = -1, channels = -1;
      std::string  algoName, protoName;
    };

    SymkReport QuerySymk(ncclComm_t comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dt, ncclRedOp_t op)
    {
      SymkReport r;
      r.res = rcclSymKGetInfo(comm, coll, count, dt, op, &r.algo, &r.proto, &r.channels);
      const char* an = nullptr;
      const char* pn = nullptr;
      if (r.res == ncclSuccess)
      {
        if (rcclGetAlgoName(r.algo, &an) == ncclSuccess && an) r.algoName = an;
        if (rcclGetProtocolName(r.proto, &pn) == ncclSuccess && pn) r.protoName = pn;
      }
      return r;
    }

    // Options controlling one sweep. Plain sweeps (existing tests) use all-false.
    struct SweepMode
    {
      bool registerSym = false;  // allocate cuMem buffers + register NCCL_WIN_COLL_SYMMETRIC
      bool checkSymk   = false;  // also assert rcclSymKGetInfo agrees with the dispatch log
      bool expectNoSym = false;  // additionally assert the symk reporter never returns SYM
      bool cumemOff    = false;  // force NCCL_CUMEM_ENABLE=0 before comm init
    };

    // Runs one collective across all comms, captures its selection log to a fresh
    // per-size file, then asserts rcclGetCollImplInfo reports what was logged.
    void CheckSizeMatchesLog(const char* funcStr, ncclFunc_t coll, int idx, int nRanks,
                             const std::vector<ncclComm_t>& comms, const std::vector<hipStream_t>& streams,
                             const std::vector<void*>& sbuf, const std::vector<void*>& rbuf, size_t count,
                             ncclDataType_t dt, const SweepMode& mode, bool* sawSym)
    {
      char path[256];
      snprintf(path, sizeof(path), "/tmp/rccl_collimpl_%d_%s_%d.log", (int)getpid(), funcStr, idx);
      remove(path);
      setenv("NCCL_DEBUG_FILE", path, 1);
      ncclResetDebugInitInternal();  // reopen NCCL_DEBUG_FILE on next log

      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        if (coll == ncclFuncAllReduce)
          NCCLCHECK(ncclAllReduce(sbuf[i], rbuf[i], count, dt, ncclSum, comms[i], streams[i]));
        else if (coll == ncclFuncReduceScatter)
          NCCLCHECK(ncclReduceScatter(sbuf[i], rbuf[i], count, dt, ncclSum, comms[i], streams[i]));
        else
          NCCLCHECK(ncclAllGather(sbuf[i], rbuf[i], count, dt, comms[i], streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipStreamSynchronize(streams[i]));
      }

      std::string  log = ReadFile(path);
      SelectedImpl sel = ParseSelected(log, funcStr);

      const size_t bytes = count * ((coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) ? nRanks : 1) *
                           (dt == ncclFloat32 ? 4 : 2);
      SCOPED_TRACE(std::string(funcStr) + " count=" + std::to_string(count) + " (~" +
                   std::to_string(bytes) + "B) dtype=" + (dt == ncclFloat32 ? "f32" : "bf16"));

      ASSERT_TRUE(sel.found) << "No recognizable selection line in dispatch log:\n" << log;

      // DDA is disabled inside a group (rcclDdaEnabled bails when ncclGroupDepth
      // != 0), and the dispatch above runs grouped. Query in the same grouped
      // context so the report reflects the backend that actually ran; querying
      // ungrouped would spuriously report DDA. The query only reads state (no
      // enqueue), so the surrounding group closes empty.
      NCCLCHECK(ncclGroupStart());
      ReportedImpl rep = QueryReport(comms[0], coll, count, dt, ncclSum, sbuf[0], rbuf[0]);
      // rcclSymKGetInfo's non-symk fallback routes through rcclSelect*, which is
      // group-depth sensitive, so query it at the same depth as dispatch.
      SymkReport symk;
      if (mode.checkSymk) symk = QuerySymk(comms[0], coll, count, dt, ncclSum);
      NCCLCHECK(ncclGroupEnd());

      if (sawSym && sel.algoName == "SYM") *sawSym = true;

      EXPECT_EQ(rep.algoName, sel.algoName)
        << "reported algo != logged-selected algo\nLOG:\n" << log;
      if (sel.hasProto)
        EXPECT_EQ(rep.protoName, sel.protoName)
          << "reported proto != logged-selected proto\nLOG:\n" << log;
      // Channels: rcclGetCollImplInfo reports the traffic-packed channel count the
      // kernel actually runs on, so it must equal the per-op channel{Lo..Hi} range
      // the dispatch log records (only checked when the log states it -- i.e. the
      // native-kernel tuning line; addon backends don't always log channels).
      if (sel.hasChannels)
        EXPECT_EQ(rep.channels, sel.channels)
          << "reported channels != logged-selected channels\nLOG:\n" << log;

      if (mode.checkSymk)
      {
        // rcclSymKGetInfo is the -R 2 reporter. It must never fail (the old code
        // returned ncclInvalidArgument, which aborted rccl-tests under TESTCHECK).
        ASSERT_EQ(symk.res, ncclSuccess)
          << "rcclSymKGetInfo failed: " << ncclGetErrorString(symk.res);

        if (mode.expectNoSym)
        {
          // cuMem off => symmetricSupport false => symk falls back to the real
          // backend. No symmetric window exists anywhere (dispatch buffers are plain
          // too), so the null-buffer fallback sees the same state as dispatch and must
          // equal the log -- and must never claim SYM.
          EXPECT_EQ(symk.algoName, sel.algoName)
            << "symk-reported algo != logged-selected algo\nLOG:\n" << log;
          EXPECT_NE(symk.algoName, "SYM")
            << "symk reported SYM with cuMem disabled\nLOG:\n" << log;
        }
        else if (sel.algoName == "SYM")
        {
          // Registered sweep, symmetric actually dispatched: the symk reporter must
          // agree it is SYM. (At non-SYM sizes the dispatch may run a
          // registration-dependent backend like CE that rcclSymKGetInfo's null-buffer
          // fallback cannot observe, so only the SYM case is asserted here; rep above
          // -- queried with the real registered buffers -- covers the rest.)
          EXPECT_EQ(symk.algoName, "SYM")
            << "symk did not report SYM though dispatch ran SYM\nLOG:\n" << log;
        }
      }

      remove(path);
    }

    // Shared driver for both collectives. Sweeps total message size in powers of
    // two over [loBytes, hiBytes] (matching rccl-tests -b/-e). The total maps to a
    // per-dtype element count; for AllGather the total is split across ranks
    // (per-rank sendcount = total / nRanks), as rccl-tests reports all_gather size.
    void RunSweep(const char* funcStr, ncclFunc_t coll, size_t loBytes, size_t hiBytes,
                  const SweepMode& mode = SweepMode{})
    {
      int numDevices = 0;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2)
      {
        GTEST_SKIP() << "This test requires at least 2 GPUs.";
      }
      const int nRanks = std::min(numDevices, 8);

      // Force cuMem off before init so comm->symmetricSupport is false: the -R 2
      // reporter (rcclSymKGetInfo) must then fall back to the real backend and
      // never claim SYM. Set in the isolated child only (fresh param cache).
      if (mode.cumemOff) setenv("NCCL_CUMEM_ENABLE", "0", 1);

      // Ground truth = the library's own selection log. COLL covers the addon
      // backend lines; TUNING covers the native-kernel algo/proto/channel line.
      setenv("NCCL_DEBUG", "INFO", 1);
      setenv("NCCL_DEBUG_SUBSYS", "COLL,TUNING", 1);

      std::vector<ncclComm_t> comms(nRanks);
      ASSERT_EQ(ncclCommInitAll(comms.data(), nRanks, nullptr), ncclSuccess);

      const std::vector<ncclDataType_t> dtypes = {ncclFloat32, ncclBfloat16};

      // Byte-sized buffers are dtype-agnostic and cover every size in the sweep.
      // AllReduce: send == recv == total. AllGather: send == total/nRanks, recv == total.
      std::vector<void*>       sbuf(nRanks), rbuf(nRanks);
      std::vector<hipStream_t> streams(nRanks);
      // For the -R 2 sweep, buffers are cuMem-allocated and registered as symmetric
      // windows (RAII); kept alive for the whole sweep. Released before comm destroy.
      std::vector<RCCLTestHelpers::SymBuf> symSend, symRecv;
      // AllReduce: send == recv == total. AllGather: send == total/nRanks, recv == total.
      // ReduceScatter is the inverse of AllGather: send == total, recv == total/nRanks.
      const size_t recvBytes =
        (coll == ncclFuncReduceScatter) ? (hiBytes + nRanks - 1) / nRanks : hiBytes;
      const size_t sendBytes =
        (coll == ncclFuncAllGather) ? (hiBytes + nRanks - 1) / nRanks : hiBytes;

      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipStreamCreate(&streams[i]));
      }

      if (mode.registerSym)
      {
        symSend.resize(nRanks);
        symRecv.resize(nRanks);

        // cuMem-backed allocation is local per rank -- do it before the group.
        bool allocOk = true;
        for (int i = 0; i < nRanks && allocOk; i++)
        {
          HIPCALL(hipSetDevice(i));
          symSend[i].comm = comms[i];
          symRecv[i].comm = comms[i];
          if (ncclMemAlloc(&symSend[i].ptr, sendBytes) != ncclSuccess ||
              ncclMemAlloc(&symRecv[i].ptr, recvBytes) != ncclSuccess)
            allocOk = false;
        }

        // Symmetric-window registration is a COLLECTIVE op: every rank must register
        // inside a single group, else the first rank blocks forever waiting on peers
        // (in-process multi-GPU). Mirrors tryRegisterInputWindows in DeviceApiTests.
        // In-group calls defer, so the real status is ncclGroupEnd()'s.
        ncclResult_t regRes = allocOk ? ncclSuccess : ncclInternalError;
        if (allocOk)
        {
          std::vector<ncclResult_t> rr(2 * nRanks, ncclSuccess);
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < nRanks; i++)
          {
            HIPCALL(hipSetDevice(i));
            rr[2 * i]     = ncclCommWindowRegister(comms[i], symSend[i].ptr, sendBytes,
                                                   &symSend[i].win, NCCL_WIN_COLL_SYMMETRIC);
            rr[2 * i + 1] = ncclCommWindowRegister(comms[i], symRecv[i].ptr, recvBytes,
                                                   &symRecv[i].win, NCCL_WIN_COLL_SYMMETRIC);
          }
          ncclResult_t ge = ncclGroupEnd();
          for (ncclResult_t r : rr)
            if (r != ncclSuccess) { regRes = r; break; }
          if (regRes == ncclSuccess) regRes = ge;
        }

        if (regRes != ncclSuccess)
        {
          // Symmetric windows unavailable/unsupported (cuMem/VMM off). Release any
          // partial allocations while comms are alive (deregister is per-rank local),
          // then skip.
          symSend.clear();
          symRecv.clear();
          for (int j = 0; j < nRanks; j++)
          {
            HIPCALL(hipSetDevice(j));
            hipStreamDestroy(streams[j]);
          }
          for (auto& c : comms) ncclCommDestroy(c);
          GTEST_SKIP() << "symmetric windows unavailable (cuMem/VMM off or unsupported): "
                       << ncclGetErrorString(regRes);
        }

        for (int i = 0; i < nRanks; i++)
        {
          HIPCALL(hipSetDevice(i));
          sbuf[i] = symSend[i].ptr;
          rbuf[i] = symRecv[i].ptr;
          HIPCALL(hipMemset(sbuf[i], 0, sendBytes));
          HIPCALL(hipMemset(rbuf[i], 0, recvBytes));
        }
      }
      else
      {
        for (int i = 0; i < nRanks; i++)
        {
          HIPCALL(hipSetDevice(i));
          HIPCALL(hipMalloc(&sbuf[i], sendBytes));
          HIPCALL(hipMalloc(&rbuf[i], recvBytes));
          HIPCALL(hipMemset(sbuf[i], 0, sendBytes));
          HIPCALL(hipMemset(rbuf[i], 0, recvBytes));
        }
      }

      // Warmup: some backends initialize lazily on first use (CE allocates its
      // staging buffer via a group task on the first enqueue-path CE collective;
      // DDA sets up IPC/fabric handles). Until that runs, the dispatch path can
      // pick a fallback for one size while the report -- taken after dispatch has
      // triggered the init -- sees the now-eligible backend, a spurious mismatch.
      // Run the full sweep once, discarded, so every lazy init completes first.
      for (ncclDataType_t dt : dtypes)
      {
        const size_t elemSize = (dt == ncclFloat32 ? 4 : 2);
        const size_t denom =
          elemSize * ((coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) ? (size_t)nRanks : 1);
        for (size_t bytes = loBytes; bytes <= hiBytes; bytes <<= 1)
        {
          const size_t count = bytes / denom;
          if (count == 0) continue;
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < nRanks; i++)
          {
            HIPCALL(hipSetDevice(i));
            if (coll == ncclFuncAllReduce)
              NCCLCHECK(ncclAllReduce(sbuf[i], rbuf[i], count, dt, ncclSum, comms[i], streams[i]));
            else if (coll == ncclFuncReduceScatter)
              NCCLCHECK(ncclReduceScatter(sbuf[i], rbuf[i], count, dt, ncclSum, comms[i], streams[i]));
            else
              NCCLCHECK(ncclAllGather(sbuf[i], rbuf[i], count, dt, comms[i], streams[i]));
          }
          NCCLCHECK(ncclGroupEnd());
        }
      }
      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipStreamSynchronize(streams[i]));
      }

      int  idx    = 0;
      bool sawSym = false;
      for (ncclDataType_t dt : dtypes)
      {
        const size_t elemSize = (dt == ncclFloat32 ? 4 : 2);
        // For AllGather/ReduceScatter the swept total is divided among ranks; AllReduce uses it whole.
        const size_t denom =
          elemSize * ((coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) ? (size_t)nRanks : 1);
        for (size_t bytes = loBytes; bytes <= hiBytes; bytes <<= 1)
        {
          const size_t count = bytes / denom;
          if (count == 0) continue;  // total too small to split across ranks for this dtype
          CheckSizeMatchesLog(funcStr, coll, idx++, nRanks, comms, streams, sbuf, rbuf, count, dt,
                              mode, &sawSym);
        }
      }

      // The -R 2 sweep registered symmetric windows. Whether SYM actually dispatches
      // is arch/config dependent (AllGather loses to CE by taskAppend ordering; on
      // some arches/tuner configs AllReduce stays on native RING/TREE), so this is
      // informational only -- the portable guarantees are the per-size assertions
      // above: rcclGetCollImplInfo and the symk reporter both match the dispatch log,
      // and the symk reporter never aborts or falsely claims SYM.
      if (mode.registerSym && !sawSym)
        fprintf(stderr,
                "[ NOTE     ] %s: symmetric windows registered but SYM never dispatched "
                "on this arch/config (reporter still matched the dispatch log at every size)\n",
                funcStr);

      // Restore default debug target before teardown.
      unsetenv("NCCL_DEBUG_FILE");
      ncclResetDebugInitInternal();

      // Release symmetric windows before destroying comms (deregister needs a live
      // comm); RAII owns the cuMem buffers, so no hipFree for those.
      symSend.clear();
      symRecv.clear();

      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        if (!mode.registerSym)
        {
          HIPCALL(hipFree(sbuf[i]));
          HIPCALL(hipFree(rbuf[i]));
        }
        HIPCALL(hipStreamDestroy(streams[i]));
      }
      for (auto& c : comms) NCCLCHECK(ncclCommDestroy(c));
    }

    // 1 KiB .. 2 GiB total message size, powers of two.
    constexpr size_t kLoBytes = 1 << 10;
    constexpr size_t kHiBytes = (size_t)2 << 30;
  }  // namespace

  TEST(CollImplInfo, AllReduceMatchesDispatchLog)
  {
    RUN_ISOLATED_TEST("AllReduceMatchesDispatchLog", []() {
      RunSweep("AllReduce", ncclFuncAllReduce, kLoBytes, kHiBytes);
    });
  }

  TEST(CollImplInfo, AllGatherMatchesDispatchLog)
  {
    RUN_ISOLATED_TEST("AllGatherMatchesDispatchLog", []() {
      RunSweep("AllGather", ncclFuncAllGather, kLoBytes, kHiBytes);
    });
  }

  // -R 2 path: buffers registered as symmetric windows. Both rcclGetCollImplInfo
  // and the symk reporter (rcclSymKGetInfo) must match the dispatch log at every
  // size (whether SYM actually dispatches is arch/config dependent -- see the
  // informational NOTE in RunSweep). Skips if cuMem/VMM is unavailable.
  TEST(CollImplInfo, AllReduceSymmetricMatchesDispatchLog)
  {
    RUN_ISOLATED_TEST("AllReduceSymmetricMatchesDispatchLog", []() {
      SweepMode mode;
      mode.registerSym = true;
      mode.checkSymk   = true;
      RunSweep("AllReduce", ncclFuncAllReduce, kLoBytes, kHiBytes, mode);
    });
  }

  TEST(CollImplInfo, AllGatherSymmetricMatchesDispatchLog)
  {
    RUN_ISOLATED_TEST("AllGatherSymmetricMatchesDispatchLog", []() {
      SweepMode mode;
      mode.registerSym = true;
      mode.checkSymk   = true;
      RunSweep("AllGather", ncclFuncAllGather, kLoBytes, kHiBytes, mode);
    });
  }

  // Regression guard: with cuMem disabled, symmetric memory cannot be used, so the
  // -R 2 reporter must fall back to the real backend -- succeed (not abort) and
  // never claim SYM. Buffers stay plain; comm->symmetricSupport is forced false.
  TEST(CollImplInfo, AllReduceSymkNotReportedWhenCuMemDisabled)
  {
    RUN_ISOLATED_TEST("AllReduceSymkNotReportedWhenCuMemDisabled", []() {
      SweepMode mode;
      mode.cumemOff    = true;
      mode.checkSymk   = true;
      mode.expectNoSym = true;
      RunSweep("AllReduce", ncclFuncAllReduce, kLoBytes, kHiBytes, mode);
    });
  }

  TEST(CollImplInfo, AllGatherSymkNotReportedWhenCuMemDisabled)
  {
    RUN_ISOLATED_TEST("AllGatherSymkNotReportedWhenCuMemDisabled", []() {
      SweepMode mode;
      mode.cumemOff    = true;
      mode.checkSymk   = true;
      mode.expectNoSym = true;
      RunSweep("AllGather", ncclFuncAllGather, kLoBytes, kHiBytes, mode);
    });
  }

  TEST(CollImplInfo, ReduceScatterMatchesDispatchLog)
  {
    RUN_ISOLATED_TEST("ReduceScatterMatchesDispatchLog", []() {
      RunSweep("ReduceScatter", ncclFuncReduceScatter, kLoBytes, kHiBytes);
    });
  }

  TEST(CollImplInfo, ReduceScatterSymmetricMatchesDispatchLog)
  {
    RUN_ISOLATED_TEST("ReduceScatterSymmetricMatchesDispatchLog", []() {
      SweepMode mode;
      mode.registerSym = true;
      mode.checkSymk   = true;
      RunSweep("ReduceScatter", ncclFuncReduceScatter, kLoBytes, kHiBytes, mode);
    });
  }

  TEST(CollImplInfo, ReduceScatterSymkNotReportedWhenCuMemDisabled)
  {
    RUN_ISOLATED_TEST("ReduceScatterSymkNotReportedWhenCuMemDisabled", []() {
      SweepMode mode;
      mode.cumemOff    = true;
      mode.checkSymk   = true;
      mode.expectNoSym = true;
      RunSweep("ReduceScatter", ncclFuncReduceScatter, kLoBytes, kHiBytes, mode);
    });
  }
}  // namespace RcclUnitTesting
