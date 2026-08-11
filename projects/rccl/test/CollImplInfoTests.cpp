/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Consistency tests for rcclGetCollImplInfo (AllReduce + AllGather).
//
// rcclGetCollImplInfo is the *reporting* path: it routes AR/AG through
// rcclSelectAllReduce/rcclSelectAllGather with query=true. Live dispatch routes
// the same functions with query=false and logs which backend it selected. These
// tests treat the dispatch log as ground truth: for each message size we run the
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
#include "rccl_common.h"  // rcclGetCollImplInfo, rcclGetAlgoName, rcclGetProtocolName, rcclAddonAlgos_t

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
    // bypass or override the kernel plan). `func` is "AllReduce" / "AllGather".
    SelectedImpl ParseSelected(const std::string& log, const std::string& func)
    {
      SelectedImpl kernel;    // from enqueue.cc tuning line
      SelectedImpl addon;     // from a backend-specific line, if any
      bool         warpSpeed = false;

      std::istringstream iss(log);
      std::string        line;
      while (std::getline(iss, line))
      {
        // WarpSpeed: RING algorithm reported as "RING*" with scaled channels.
        if (line.find("WarpSpeed enabled") != std::string::npos) warpSpeed = true;

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

      if (kernel.found && warpSpeed)
      {
        // Base tuning line logs "RING"; the backend actually running is WarpSpeed,
        // which the report names "RING*" and runs on a scaled channel count -- so
        // proto still matches the tuning line but channels do not.
        kernel.algoName    = "RING*";
        kernel.hasChannels = false;
      }
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

    // Runs one collective across all comms, captures its selection log to a fresh
    // per-size file, then asserts rcclGetCollImplInfo reports what was logged.
    void CheckSizeMatchesLog(const char* funcStr, ncclFunc_t coll, int idx, int nRanks,
                             const std::vector<ncclComm_t>& comms, const std::vector<hipStream_t>& streams,
                             const std::vector<void*>& sbuf, const std::vector<void*>& rbuf, size_t count,
                             ncclDataType_t dt)
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

      const size_t bytes = count * (coll == ncclFuncAllGather ? nRanks : 1) *
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
      NCCLCHECK(ncclGroupEnd());

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

      remove(path);
    }

    // Shared driver for both collectives. Sweeps total message size in powers of
    // two over [loBytes, hiBytes] (matching rccl-tests -b/-e). The total maps to a
    // per-dtype element count; for AllGather the total is split across ranks
    // (per-rank sendcount = total / nRanks), as rccl-tests reports all_gather size.
    void RunSweep(const char* funcStr, ncclFunc_t coll, size_t loBytes, size_t hiBytes)
    {
      int numDevices = 0;
      HIPCALL(hipGetDeviceCount(&numDevices));
      if (numDevices < 2)
      {
        GTEST_SKIP() << "This test requires at least 2 GPUs.";
      }
      const int nRanks = std::min(numDevices, 8);

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
      const size_t recvBytes = hiBytes;
      const size_t sendBytes =
        (coll == ncclFuncAllGather) ? (hiBytes + nRanks - 1) / nRanks : hiBytes;
      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipMalloc(&sbuf[i], sendBytes));
        HIPCALL(hipMalloc(&rbuf[i], recvBytes));
        HIPCALL(hipMemset(sbuf[i], 0, sendBytes));
        HIPCALL(hipMemset(rbuf[i], 0, recvBytes));
        HIPCALL(hipStreamCreate(&streams[i]));
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
        const size_t denom = elemSize * (coll == ncclFuncAllGather ? (size_t)nRanks : 1);
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

      int idx = 0;
      for (ncclDataType_t dt : dtypes)
      {
        const size_t elemSize = (dt == ncclFloat32 ? 4 : 2);
        // For AllGather the swept total is divided among ranks; AllReduce uses it whole.
        const size_t denom = elemSize * (coll == ncclFuncAllGather ? (size_t)nRanks : 1);
        for (size_t bytes = loBytes; bytes <= hiBytes; bytes <<= 1)
        {
          const size_t count = bytes / denom;
          if (count == 0) continue;  // total too small to split across ranks for this dtype
          CheckSizeMatchesLog(funcStr, coll, idx++, nRanks, comms, streams, sbuf, rbuf, count, dt);
        }
      }

      // Restore default debug target before teardown.
      unsetenv("NCCL_DEBUG_FILE");
      ncclResetDebugInitInternal();

      for (int i = 0; i < nRanks; i++)
      {
        HIPCALL(hipSetDevice(i));
        HIPCALL(hipFree(sbuf[i]));
        HIPCALL(hipFree(rbuf[i]));
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
}  // namespace RcclUnitTesting
