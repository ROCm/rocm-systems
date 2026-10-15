/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/recorder.cc: RCCL_REPLAY_FILE output naming, the JSON and binary
// record formats of every record() overload, graph-capture deferral, and parseJsonEntry.

#include <gtest/gtest.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nccl.h"
#include "debug.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
std::vector<std::string> g_errors;
int g_device = 0;
hipError_t g_addressRangeResult = hipSuccess;
hipError_t g_captureInfoResult = hipSuccess;
hipStreamCaptureStatus g_captureStatus = hipStreamCaptureStatusNone;
unsigned long long g_graphId = 0;
hipError_t g_addHostNodeResult = hipSuccess;
std::vector<hipHostNodeParams> g_hostNodes;
std::vector<hipGraph_t> g_hostNodeGraphs;
const auto kGraph = reinterpret_cast<hipGraph_t>(0x6a);
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (level == NCCL_LOG_ERROR) g_errors.push_back(buf);
}

hipError_t hipGetDevice(int* dev) {
  *dev = g_device;
  return hipSuccess;
}
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }
// Every allocation starts 0x10 below the pointer and spans 0x100 bytes.
hipError_t hipMemGetAddressRange(hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
  if (g_addressRangeResult != hipSuccess) return g_addressRangeResult;
  *base = static_cast<char*>(ptr) - 0x10;
  *size = 0x100;
  return hipSuccess;
}
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t, hipStreamCaptureStatus* status, unsigned long long* id,
                                      hipGraph_t* graph, const hipGraphNode_t**, size_t*) {
  if (g_captureInfoResult != hipSuccess) return g_captureInfoResult;
  *status = g_captureStatus;
  *id = g_graphId;
  *graph = kGraph;
  return hipSuccess;
}
hipError_t hipGraphAddHostNode(hipGraphNode_t*, hipGraph_t graph, const hipGraphNode_t*, size_t,
                               const hipHostNodeParams* params) {
  g_hostNodes.push_back(*params);
  g_hostNodeGraphs.push_back(graph);
  return g_addHostNodeResult;
}

// The constructor reads the environment once per process; the tests re-run it on the singleton.
#define private public
#include RECORDER_CC_PATH
#undef private

namespace {

using rccl::rcclApiCall;
using rccl::Recorder;

constexpr char kContext[] = "context : [";

std::string HostName() {
  char host[256];
  gethostname(host, sizeof(host));
  return host;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  return lines;
}

std::string Fmt(const char* fmt, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

// The part of a JSON entry before its context, which holds the run-dependent time and thread.
std::string Head(const std::string& line) {
  size_t pos = line.find(kContext);
  return pos == std::string::npos ? line : line.substr(0, pos + sizeof(kContext) - 1);
}

struct Context {
  double time = -1;
  int thread = -1, device = -1, captured = -1;
  unsigned long long graphId = 0;
};

Context ParseContext(const std::string& line) {
  Context c;
  size_t pos = line.find(kContext);
  EXPECT_NE(pos, std::string::npos) << line;
  if (pos == std::string::npos) return c;
  EXPECT_EQ(std::sscanf(line.c_str() + pos + sizeof(kContext) - 1, rccl::ctxt_fmt.c_str(), &c.time, &c.thread,
                        &c.device, &c.captured, &c.graphId),
            5)
    << line;
  return c;
}

class RecorderMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    char dirTemplate[] = "/tmp/rccl-recorder-test-XXXXXX";
    ASSERT_NE(mkdtemp(dirTemplate), nullptr);
    dir_ = dirTemplate;
    g_errors.clear();
    g_device = 2;
    g_addressRangeResult = hipSuccess;
    g_captureInfoResult = hipSuccess;
    g_captureStatus = hipStreamCaptureStatusNone;
    g_graphId = 0;
    g_addHostNodeResult = hipSuccess;
    g_hostNodes.clear();
    g_hostNodeGraphs.clear();
    comm_ = std::make_unique<ncclComm>();
    comm_->opCount = 0x2a;
    comm_->nRanks = 3;
    comm_->localRank = 1;
    comm_->localRankToRank = localRankToRank_;
    comm_->planner.nTasksColl = 13;
    comm_->planner.nTasksP2p = 21;
  }

  void TearDown() override {
    Restart(nullptr);
    std::system(("rm -rf " + dir_).c_str());
  }

  // Re-runs the singleton's constructor with RCCL_REPLAY_FILE (and optionally RCCL_LOG_LEVEL) set.
  Recorder& Restart(const char* file, const char* logLevel = nullptr) {
    if (file) setenv("RCCL_REPLAY_FILE", file, 1);
    else unsetenv("RCCL_REPLAY_FILE");
    if (logLevel) setenv("RCCL_LOG_LEVEL", logLevel, 1);
    else unsetenv("RCCL_LOG_LEVEL");
    Recorder& r = Recorder::instance();
    r.~Recorder();
    new (&r) Recorder();
    rccl::lastcall = rccl::rrGroupStart;
    Recorder::depth = 0;
    unsetenv("RCCL_REPLAY_FILE");
    unsetenv("RCCL_LOG_LEVEL");
    return r;
  }

  Recorder& StartJson(const char* logLevel = nullptr) {
    Recorder& r = Restart((dir_ + "/trace.json").c_str(), logLevel);
    path_ = Output("/trace", ".json");
    return r;
  }
  Recorder& StartBinary() {
    Recorder& r = Restart((dir_ + "/trace.bin").c_str());
    path_ = Output("/trace", ".bin");
    return r;
  }

  std::string Output(const std::string& name, const std::string& ext) {
    return dir_ + name + "." + std::to_string(getpid()) + "." + HostName() + ext;
  }

  // JSON entries after the opening "{" and "  version : 1," lines.
  std::vector<std::string> Entries() {
    std::vector<std::string> lines = SplitLines(ReadFile(path_));
    EXPECT_GE(lines.size(), 2u);
    if (lines.size() < 2) return {};
    EXPECT_EQ(lines[0], "{");
    EXPECT_EQ(lines[1], "  version : 1,");
    return std::vector<std::string>(lines.begin() + 2, lines.end());
  }

  std::vector<rcclApiCall> BinaryCalls() {
    std::string bytes = ReadFile(path_);
    EXPECT_EQ(bytes.size() % sizeof(rcclApiCall), 0u);
    std::vector<rcclApiCall> calls(bytes.size() / sizeof(rcclApiCall));
    std::memcpy(static_cast<void*>(calls.data()), bytes.data(), calls.size() * sizeof(rcclApiCall));
    return calls;
  }

  ncclInfo Info(const void* send, void* recv) {
    ncclInfo info = {};
    info.sendbuff = send;
    info.recvbuff = recv;
    info.count = 1024;
    info.datatype = ncclFloat16;
    info.op = ncclMin;
    info.root = 1;
    info.comm = comm_.get();
    info.stream = stream_;
    info.acc = acc_;
    return info;
  }

  std::string dir_;
  std::string path_;
  int localRankToRank_[2] = {7, 9};
  std::unique_ptr<ncclComm> comm_;
  hipStream_t stream_ = reinterpret_cast<hipStream_t>(0x57);
  const void* acc_ = reinterpret_cast<const void*>(0xacc0);
  char send_[64] = {};
  char recv_[64] = {};
};

TEST_F(RecorderMicrotest, Disabled_WithoutReplayFileNothingIsWritten) {
  Recorder& r = Restart(nullptr);
  ncclInfo info = Info(send_, recv_);
  ncclUniqueId id = {};
  int devs[1] = {0};
  EXPECT_EQ(r.record(rccl::rrAllReduce, info), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrSend, send_, recv_, 1, ncclInt8, comm_.get(), stream_), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrRedOpDestroy, ncclSum, comm_.get()), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrGroupStart, 1), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrCommInitRank, 2, 0, &id), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrCommDestroy, comm_.get()), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrCommRegister, comm_.get(), recv_, send_, 8), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrMemFree, recv_), ncclSuccess);
  r.record(1, nullptr);
  r.record(rccl::rrCommInitRankConfig, 2, 0, &id, static_cast<ncclConfig_t*>(nullptr));
  r.record(nullptr, 1, devs);
  r.record("ncclGetVersion");
  rcclApiCall raw(rccl::rrBroadcast);
  EXPECT_EQ(r.record(raw), ncclSuccess);
  r.skip(true);
  EXPECT_FALSE(r.skipped);
  EXPECT_FALSE(r.outputFile.is_open());
  EXPECT_EQ(r.logLevel, -1);
}

TEST_F(RecorderMicrotest, FileName_JsonExtensionGetsPidAndHostBeforeIt) {
  Recorder& r = StartJson();
  EXPECT_EQ(r.output_json, 1);
  EXPECT_EQ(r.logLevel, 1);
  EXPECT_EQ(r.pid, getpid());
  EXPECT_EQ(ReadFile(path_).substr(0, 2), "{\n");
}

TEST_F(RecorderMicrotest, FileName_OtherExtensionIsBinary) {
  Recorder& r = StartBinary();
  EXPECT_EQ(r.output_json, 0);
  EXPECT_TRUE(r.outputFile.is_open());
  EXPECT_EQ(ReadFile(path_), "");
}

TEST_F(RecorderMicrotest, FileName_WithoutExtensionIsBinary) {
  Recorder& r = Restart((dir_ + "/trace").c_str(), "3");
  EXPECT_EQ(r.output_json, 0);
  EXPECT_EQ(r.logLevel, 3);
  path_ = Output("/trace", "");
  EXPECT_TRUE(r.outputFile.is_open());
  EXPECT_EQ(ReadFile(path_), "");
}

// The extension is taken from the first '.' in the whole path, so a dotted directory breaks the name.
TEST_F(RecorderMicrotest, FileName_DotInDirectorySplitsThePath_Observed) {
  std::string dotted = dir_ + "/run.1";
  ASSERT_EQ(mkdir(dotted.c_str(), 0700), 0);
  Recorder& r = Restart((dotted + "/trace.json").c_str());
  EXPECT_EQ(r.output_json, 0);
  EXPECT_FALSE(r.outputFile.is_open());  // tried <dir>/run.<pid>.<host>.1/trace.json
}

TEST_F(RecorderMicrotest, Destructor_ClosesTheJsonObject) {
  StartJson();
  Restart(nullptr);
  EXPECT_EQ(ReadFile(path_), "{\n  version : 1,\n}\n");
}

TEST_F(RecorderMicrotest, Json_CollectiveFromInfoRecordsBuffersCommAndContext) {
  Recorder& r = StartJson();
  ncclInfo info = Info(send_, recv_);

  ASSERT_EQ(r.record(rccl::rrAllReduce, info), ncclSuccess);

  std::vector<std::string> entries = Entries();
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(Head(entries[0]),
            Fmt("  AllReduce : [opCount : 2a, sendbuff : [addr : %p, base : %p, size : 256], recvbuff : [addr : %p, "
                "base : %p, size : 256], acc : %p, count : 1024, datatype : %d, op : %d, root : 1, comm : %p, "
                "nranks : 3, stream : %p, task : 34, globalrank : 9, context : [",
                send_, send_ - 0x10, recv_, recv_ - 0x10, acc_, ncclFloat16, ncclMin, comm_.get(), stream_));
  Context c = ParseContext(entries[0]);
  EXPECT_EQ(c.thread, (int)syscall(SYS_gettid));
  EXPECT_EQ(c.device, 2);
  EXPECT_EQ(c.captured, 0);
  EXPECT_GT(c.time, 0.0);
}

// A receive-only call has no send buffer to look up.
TEST_F(RecorderMicrotest, Json_NoSendBufferLeavesItsRangeEmpty) {
  Recorder& r = StartJson();
  ncclInfo info = Info(nullptr, recv_);

  ASSERT_EQ(r.record(rccl::rrRecv, info), ncclSuccess);

  EXPECT_NE(Entries()[0].find("sendbuff : [addr : (nil), base : (nil), size : 0]"), std::string::npos);
}

TEST_F(RecorderMicrotest, Json_EntriesAreCommaSeparatedAndGroupsNest) {
  Recorder& r = StartJson();

  r.record(rccl::rrGroupStart, 1);
  r.record(rccl::rrCommDestroy, comm_.get());
  r.record(rccl::rrCommAbort, comm_.get());
  r.record(rccl::rrGroupEnd, 0);
  r.record(rccl::rrCommFinalize, comm_.get());

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 5u);
  EXPECT_EQ(e[0], "  {");
  EXPECT_EQ(Head(e[1]), Fmt("    CommDestroy : [comm : %p, context : [", comm_.get()));
  EXPECT_EQ(e[1].back(), ',');
  EXPECT_EQ(Head(e[2]), Fmt("    CommAbort : [comm : %p, context : [", comm_.get()));
  EXPECT_EQ(e[2].back(), ']');
  EXPECT_EQ(e[3], "  },");
  EXPECT_EQ(Head(e[4]), Fmt("  CommFinalize : [comm : %p, context : [", comm_.get()));
}

// A dropped Broadcast does not update the last call, so every Broadcast until another call is dropped.
TEST_F(RecorderMicrotest, Json_BroadcastsAfterBcastAreDropped) {
  Recorder& r = StartJson();
  rcclApiCall bcast(rccl::rrBcast), broadcast(rccl::rrBroadcast), reduce(rccl::rrReduce);

  r.record(bcast);
  r.record(broadcast);
  r.record(broadcast);
  r.record(reduce);
  r.record(broadcast);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 3u);
  EXPECT_EQ(e[0].find("  Bcast : ["), 0u);
  EXPECT_EQ(e[1].find("  Reduce : ["), 0u);
  EXPECT_EQ(e[2].find("  Broadcast : ["), 0u);
}

TEST_F(RecorderMicrotest, Json_InitAndDestroyEntries) {
  Recorder& r = StartJson();
  ncclUniqueId id = {};
  for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) id.internal[i] = (char)i;
  ncclComm_t comms[2];
  int devs[2] = {4, 6};

  r.record(rccl::rrGetUniqueId, 0, 0, &id);
  r.record(rccl::rrCommInitDev, 8, 5, &id, comm_.get(), 3);
  r.record(rccl::rrCommInitRank, 8, 5, &id);
  r.record(comms, 2, devs);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 4u);
  const uint64_t hash = rccl::hashUniqueId(id);
  EXPECT_EQ(Head(e[0]), Fmt("  GetUniqueId : [uniqueID : %llu, context : [", (unsigned long long)hash));
  EXPECT_EQ(Head(e[1]), Fmt("  CommInitDev : [comm : %p, size : 8, uniqueID : %llu, rank : 5, dev : 3, context : [",
                            comm_.get(), (unsigned long long)hash));
  EXPECT_EQ(Head(e[2]), Fmt("  CommInitRank : [size : 8, uniqueID : %llu, rank : 5, context : [",
                            (unsigned long long)hash));
  EXPECT_EQ(Head(e[3]), "  CommInitAll : [# of device : 2, context : [");
  EXPECT_NE(e[3].find("]], devlist : [4, 6]"), std::string::npos) << e[3];
}

TEST_F(RecorderMicrotest, Json_UniqueIdHashIsStable) {
  ncclUniqueId zero = {}, other = {};
  other.internal[NCCL_UNIQUE_ID_BYTES - 1] = 1;
  EXPECT_EQ(rccl::hashUniqueId(zero), 15490093245688404123ull);
  EXPECT_NE(rccl::hashUniqueId(zero), rccl::hashUniqueId(other));
}

TEST_F(RecorderMicrotest, Json_ConfigIsAppendedToInitRankConfig) {
  Recorder& r = StartJson();
  ncclUniqueId id = {};
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  config.cgaClusterSize = 2;
  config.minCTAs = 3;
  config.maxCTAs = 4;
  config.netName = "IB";
  config.splitShare = 1;

  r.record(rccl::rrCommInitRankConfig, 4, 1, &id, &config);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 1u);
  EXPECT_EQ(Head(e[0]), Fmt("  CommInitRankConfig : [size : 4, uniqueID : %llu, rank : 1, context : [",
                            (unsigned long long)rccl::hashUniqueId(id)));
  EXPECT_NE(e[0].find(Fmt("]], ncclConfig : [size : %zu, magic : %u, version : %u, blocking : 0, cgaClusterSize : 2, "
                          "minCTA : 3, maxCTA : 4, netname : IB, splitshare : 1]",
                          config.size, config.magic, config.version)),
            std::string::npos)
    << e[0];
}

// The split format and its arguments are out of step: "color" prints the comm pointer's low bits,
// "key" prints the color and "newcomm" the key.
TEST_F(RecorderMicrotest, Json_SplitFieldsAreShiftedByOne_Observed) {
  Recorder& r = StartJson();
  ncclComm_t parent = reinterpret_cast<ncclComm_t>(0x1234);
  ncclUniqueId* asId = reinterpret_cast<ncclUniqueId*>(0x5678);

  r.record(rccl::rrCommSplit, 6 /*color*/, 9 /*key*/, asId, static_cast<ncclConfig_t*>(nullptr), parent);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 1u);
  int color = -1, key = -1;
  void* comm = nullptr;
  ASSERT_EQ(std::sscanf(e[0].c_str(), "  CommSplit : [comm : %p, color : %d, key : %d,", &comm, &color, &key), 3)
    << e[0];
  EXPECT_EQ(comm, static_cast<void*>(asId));
  EXPECT_EQ(color, 0x1234);
  EXPECT_EQ(key, 6);
}

TEST_F(RecorderMicrotest, Json_MemoryAndRegistrationEntries) {
  Recorder& r = StartJson();
  void* handle = reinterpret_cast<void*>(0x4a);

  r.record(rccl::rrMemAlloc, recv_, 4096);
  r.record(rccl::rrMemFree, recv_, 4096);
  r.record(rccl::rrCommRegister, comm_.get(), handle, send_, 32);
  r.record(rccl::rrCommDeregister, comm_.get(), handle, send_, 32);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 4u);
  EXPECT_EQ(Head(e[0]), Fmt("  MemAlloc : [returned ptr : %p, size : 4096, context : [", recv_));
  EXPECT_EQ(Head(e[1]), Fmt("  MemFree : [ptr : %p, context : [", recv_));
  EXPECT_EQ(Head(e[2]), Fmt("  CommRegister : [comm : %p, buff : [addr : %p, base : %p, size : 256], returned handle "
                            ": %p, count : 32, context : [",
                            comm_.get(), send_, send_ - 0x10, handle));
  EXPECT_EQ(Head(e[3]), Fmt("  CommDeregister : [comm : %p, handle : %p, context : [", comm_.get(), handle));
}

TEST_F(RecorderMicrotest, Json_RedOpEntries) {
  Recorder& r = StartJson();
  float scalar = 2.0f;

  r.record(rccl::rrRedOpCreatePreMulSum, ncclMax, comm_.get(), ncclFloat32, ncclScalarHostImmediate, &scalar);
  r.record(rccl::rrRedOpDestroy, ncclMax, comm_.get(), ncclFloat64, ncclScalarHostImmediate, &scalar);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 2u);
  EXPECT_EQ(Head(e[0]), Fmt("  RedOpCreatePreMulSum : [scalar : %p, datatype : %d, op : %d, residence : %d, comm : %p, "
                            "context : [",
                            &scalar, ncclFloat32, ncclMax, ncclScalarHostImmediate, comm_.get()));
  EXPECT_EQ(Head(e[1]), Fmt("  RedOpDestroy : [op : %d, comm : %p, context : [", ncclMax, comm_.get()));
}

TEST_F(RecorderMicrotest, Json_SendRecvKeepsRootOnlyWhenGiven) {
  Recorder& r = StartJson();

  r.record(rccl::rrSend, send_, nullptr, 5, ncclInt32, comm_.get(), stream_, 2);
  r.record(rccl::rrRecv, nullptr, recv_, 5, ncclInt32, comm_.get(), stream_);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 2u);
  EXPECT_NE(e[0].find(", root : 2, "), std::string::npos) << e[0];
  EXPECT_NE(e[1].find(", root : 0, "), std::string::npos) << e[1];
}

TEST_F(RecorderMicrotest, Json_AlltoAllvAppendsItsCountsAndDisplacements) {
  Recorder& r = StartJson();
  const size_t sc[3] = {1, 2, 3}, sd[3] = {0, 1, 3}, rc[3] = {4, 5, 6}, rd[3] = {0, 4, 9};

  ASSERT_EQ(r.record(rccl::rrAllToAllv, send_, recv_, 0, ncclInt8, comm_.get(), stream_, -1, sc, sd, rc, rd),
            ncclSuccess);

  std::string e = Entries()[0];
  EXPECT_NE(e.find("]], sendcounts : [1, 2, 3], sdispls : [0, 1, 3], recvcounts : [4, 5, 6], rdispls : [0, 4, 9]"),
            std::string::npos)
    << e;
}

TEST_F(RecorderMicrotest, Json_SimulatedGroupEndClosesTheGroupAndAppendsSimInfo) {
  Recorder& r = StartJson();
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
  sim.estimatedTime = 1.5f;

  r.record(rccl::rrGroupStart, 1);
  r.record(0, &sim);
  r.record(0, nullptr);

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 3u);
  EXPECT_EQ(e[0], "  {");
  EXPECT_EQ(e[1].find(Fmt("  }[size : %zu, magic : %u, version : %u, estimated time : 1.500000, timestamp : ", sim.size,
                          sim.magic, sim.version)),
            0u)
    << e[1];
  EXPECT_EQ(e[2], "}");  // an unmatched end still closes a level
}

TEST_F(RecorderMicrotest, Json_NamedCallsNeedLogLevelTwo) {
  StartJson().record("ncclGetVersion");
  EXPECT_EQ(ReadFile(path_).find("ncclGetVersion"), std::string::npos);

  Recorder& r = StartJson("2");
  r.record("ncclGetVersion");
  r.record("ncclCommCount");

  std::vector<std::string> e = Entries();
  ASSERT_EQ(e.size(), 2u);
  double t0 = 0, t1 = 0;
  EXPECT_EQ(std::sscanf(e[0].c_str(), "  ncclGetVersion : [time : %lf],", &t0), 1) << e[0];
  EXPECT_EQ(e[0].back(), ',');
  EXPECT_EQ(std::sscanf(e[1].c_str(), "  ncclCommCount : [time : %lf]", &t1), 1) << e[1];
  EXPECT_LE(t0, t1);
}

TEST_F(RecorderMicrotest, Binary_WritesOneRawRecordPerCall) {
  Recorder& r = StartBinary();
  ncclInfo info = Info(send_, recv_);

  r.record(rccl::rrReduce, info);
  r.record(rccl::rrMemAlloc, recv_, 64);
  r.record(rccl::rrGroupEnd, 0);
  r.record("ignored");

  std::vector<rcclApiCall> calls = BinaryCalls();
  ASSERT_EQ(calls.size(), 3u);
  EXPECT_EQ(calls[0].type, rccl::rrReduce);
  EXPECT_EQ(calls[0].sendbuff, static_cast<const void*>(send_));
  EXPECT_EQ(calls[0].recvPtrBase, static_cast<void*>(recv_ - 0x10));
  EXPECT_EQ(calls[0].nTasks, 34);
  EXPECT_EQ(calls[0].globalRank, 9);
  EXPECT_EQ(calls[0].hipDev, 2);
  EXPECT_EQ(calls[0].pid, getpid());
  EXPECT_EQ(calls[1].type, rccl::rrMemAlloc);
  EXPECT_EQ(calls[1].count, 64u);
  EXPECT_EQ(calls[2].type, rccl::rrGroupEnd);
  EXPECT_EQ(calls[2].groupDepth, 0);
}

TEST_F(RecorderMicrotest, Binary_TrailingArraysFollowTheirRecord) {
  Recorder& r = StartBinary();
  const size_t sc[3] = {1, 2, 3}, sd[3] = {4, 5, 6}, rc[3] = {7, 8, 9}, rd[3] = {10, 11, 12};
  ncclComm_t comms[2];
  int devs[2] = {3, 1};

  r.record(rccl::rrAllToAllv, send_, recv_, 0, ncclInt8, comm_.get(), stream_, -1, sc, sd, rc, rd);
  r.record(comms, 2, devs);
  r.record(comms, 2, nullptr);

  std::string bytes = ReadFile(path_);
  ASSERT_EQ(bytes.size(), 3 * sizeof(rcclApiCall) + 12 * sizeof(size_t) + 2 * sizeof(int));
  size_t arrays[12];
  std::memcpy(arrays, bytes.data() + sizeof(rcclApiCall), sizeof(arrays));
  for (int i = 0; i < 12; ++i) EXPECT_EQ(arrays[i], (size_t)i + 1);
  int gotDevs[2];
  std::memcpy(gotDevs, bytes.data() + 2 * sizeof(rcclApiCall) + sizeof(arrays), sizeof(gotDevs));
  EXPECT_EQ(gotDevs[0], 3);
  EXPECT_EQ(gotDevs[1], 1);
}

TEST_F(RecorderMicrotest, Binary_RecordFieldsPerOverload) {
  Recorder& r = StartBinary();
  ncclUniqueId id = {};
  ncclComm_t comms[1];
  void* handle = reinterpret_cast<void*>(0x4a);

  r.record(rccl::rrCommInitDev, 8, 5, &id, comm_.get(), 3);
  r.record(rccl::rrCommSplit, 6, 9, reinterpret_cast<ncclUniqueId*>(0x5678), comm_.get());
  r.record(rccl::rrCommDeregister, comm_.get(), handle, send_, 32);
  r.record(rccl::rrMemFree, recv_, 99);
  r.record(comms, 1, nullptr);
  r.record(rccl::rrRedOpDestroy, ncclMax, comm_.get(), ncclFloat64, ncclScalarHostImmediate, handle);

  std::vector<rcclApiCall> c = BinaryCalls();
  ASSERT_EQ(c.size(), 6u);
  EXPECT_EQ(c[0].root, 3);
  EXPECT_EQ(c[0].comm, comm_.get());
  EXPECT_EQ(c[0].commId, rccl::hashUniqueId(id));
  EXPECT_EQ(c[0].nRanks, 8);
  EXPECT_EQ(c[0].globalRank, 5);
  EXPECT_EQ(c[1].commId, 0x5678u);  // the split records the id pointer itself
  EXPECT_EQ(c[1].comm, comm_.get());
  EXPECT_EQ(c[1].root, -1);
  EXPECT_EQ(c[2].recvbuff, handle);
  EXPECT_EQ(c[2].sendbuff, nullptr);
  EXPECT_EQ(c[2].count, 0u);
  EXPECT_EQ(c[3].recvbuff, static_cast<void*>(recv_));
  EXPECT_EQ(c[3].count, 0u);
  EXPECT_EQ(c[4].root, 1);
  EXPECT_EQ(c[4].sendbuff, nullptr);
  EXPECT_EQ(c[5].sendbuff, nullptr);  // only PreMulSum records the scalar
  EXPECT_EQ(c[5].datatype, ncclInt8);
}

// Named calls, sim info and configs are JSON-only extras.
TEST_F(RecorderMicrotest, Binary_JsonOnlyExtrasAreLeftOut) {
  Recorder& r = Restart((dir_ + "/trace.bin").c_str(), "2");
  path_ = Output("/trace", ".bin");
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclUniqueId id = {};

  r.record("ncclGetVersion");
  r.record(0, &sim);
  r.record(rccl::rrCommInitRankConfig, 2, 1, &id, &config);

  std::vector<rcclApiCall> c = BinaryCalls();
  ASSERT_EQ(c.size(), 2u);
  EXPECT_EQ(c[0].type, rccl::rrGroupSimulatedEnd);
  EXPECT_EQ(c[1].type, rccl::rrCommInitRankConfig);
}

// skip() hides collectives and group calls only.
TEST_F(RecorderMicrotest, Skip_HidesCollectivesAndGroupsOnly) {
  Recorder& r = StartBinary();
  ncclInfo info = Info(send_, recv_);

  r.skip(true);
  r.record(rccl::rrAllGather, info);
  r.record(rccl::rrGroupStart, 1);
  r.record(rccl::rrSend, send_, nullptr, 5, ncclInt32, comm_.get(), stream_);
  r.skip(false);
  r.record(rccl::rrAllGather, info);

  std::vector<rcclApiCall> c = BinaryCalls();
  ASSERT_EQ(c.size(), 2u);
  EXPECT_EQ(c[0].type, rccl::rrSend);
  EXPECT_EQ(c[1].type, rccl::rrAllGather);
}

TEST_F(RecorderMicrotest, Capture_CollectiveIsWrittenNowAndAgainFromAHostNode) {
  Recorder& r = StartBinary();
  ncclInfo info = Info(send_, recv_);
  g_captureStatus = hipStreamCaptureStatusActive;
  g_graphId = 77;

  ASSERT_EQ(r.record(rccl::rrAllReduce, info), ncclSuccess);

  ASSERT_EQ(g_hostNodes.size(), 1u);
  EXPECT_EQ(g_hostNodeGraphs[0], kGraph);
  EXPECT_EQ(g_hostNodes[0].fn, &Recorder::recordLater);
  EXPECT_EQ(g_hostNodes[0].userData, (void*)0);
  ASSERT_EQ(r.calls.size(), 1u);
  std::vector<rcclApiCall> c = BinaryCalls();
  ASSERT_EQ(c.size(), 1u);
  EXPECT_EQ(c[0].graphCaptured, 1);
  EXPECT_EQ(c[0].graphID, 77u);

  ASSERT_EQ(r.record(rccl::rrReduce, info), ncclSuccess);
  ASSERT_EQ(g_hostNodes.size(), 2u);
  EXPECT_EQ(g_hostNodes[1].userData, (void*)1);

  g_captureStatus = hipStreamCaptureStatusNone;  // graph launch: the stream is no longer capturing
  g_hostNodes[1].fn(g_hostNodes[1].userData);
  g_hostNodes[0].fn(g_hostNodes[0].userData);

  c = BinaryCalls();
  ASSERT_EQ(c.size(), 4u);
  EXPECT_EQ(c[2].type, rccl::rrReduce);
  EXPECT_EQ(c[3].type, rccl::rrAllReduce);
  EXPECT_EQ(c[3].recvbuff, static_cast<void*>(recv_));
  EXPECT_EQ(c[3].graphCaptured, 0);
}

TEST_F(RecorderMicrotest, Capture_CommunicatorCallsAreNotDeferred) {
  Recorder& r = StartBinary();
  g_captureStatus = hipStreamCaptureStatusActive;
  g_captureInfoResult = hipErrorInvalidValue;  // never queried for these

  EXPECT_EQ(r.record(rccl::rrCommDestroy, comm_.get()), ncclSuccess);
  EXPECT_EQ(r.record(rccl::rrGroupStart, 1), ncclSuccess);

  EXPECT_TRUE(g_hostNodes.empty());
  EXPECT_EQ(BinaryCalls().size(), 2u);
}

TEST_F(RecorderMicrotest, Capture_AlltoAllIsCheckedLikeOtherCollectives) {
  Recorder& r = StartBinary();
  g_captureStatus = hipStreamCaptureStatusActive;

  r.record(rccl::rrAllToAll, send_, recv_, 1, ncclInt8, comm_.get(), stream_);

  EXPECT_EQ(g_hostNodes.size(), 1u);
}

TEST_F(RecorderMicrotest, Capture_QueryFailureIsReturnedAndNothingIsWritten) {
  Recorder& r = StartBinary();
  ncclInfo info = Info(send_, recv_);
  g_captureInfoResult = hipErrorInvalidValue;

  EXPECT_EQ(r.record(rccl::rrAllReduce, info), ncclUnhandledCudaError);

  EXPECT_TRUE(BinaryCalls().empty());
  ASSERT_FALSE(g_errors.empty());
  EXPECT_NE(g_errors[0].find("HIP failure"), std::string::npos);
}

TEST_F(RecorderMicrotest, Capture_HostNodeFailureIsReturnedAndNothingIsWritten) {
  Recorder& r = StartBinary();
  ncclInfo info = Info(send_, recv_);
  g_captureStatus = hipStreamCaptureStatusActive;
  g_addHostNodeResult = hipErrorInvalidValue;

  EXPECT_EQ(r.record(rccl::rrAllReduce, info), ncclUnhandledCudaError);

  EXPECT_TRUE(BinaryCalls().empty());
}

TEST_F(RecorderMicrotest, Register_AddressRangeFailureIsReturned) {
  Recorder& r = StartBinary();
  g_addressRangeResult = hipErrorInvalidValue;

  EXPECT_EQ(r.record(rccl::rrCommRegister, comm_.get(), nullptr, send_, 8), ncclUnhandledCudaError);
  EXPECT_EQ(r.record(rccl::rrCommDeregister, comm_.get(), nullptr, send_, 8), ncclSuccess);  // no lookup

  EXPECT_EQ(BinaryCalls().size(), 1u);
}

TEST_F(RecorderMicrotest, Parse_CollectiveRoundTripsEveryField) {
  Recorder& r = StartJson();
  ncclInfo info = Info(send_, recv_);
  r.record(rccl::rrReduceScatter, info);
  std::vector<rcclApiCall> parsed;

  rccl::parseJsonEntry(Entries()[0].c_str(), parsed);

  ASSERT_EQ(parsed.size(), 1u);
  const rcclApiCall& p = parsed[0];
  EXPECT_EQ(p.type, rccl::rrReduceScatter);
  EXPECT_EQ(p.opCount, 0x2au);
  EXPECT_EQ(p.sendbuff, static_cast<const void*>(send_));
  EXPECT_EQ(p.sendPtrBase, static_cast<void*>(send_ - 0x10));
  EXPECT_EQ(p.sendPtrExtent, 0x100u);
  EXPECT_EQ(p.recvbuff, static_cast<void*>(recv_));
  EXPECT_EQ(p.recvPtrBase, static_cast<void*>(recv_ - 0x10));
  EXPECT_EQ(p.recvPtrExtent, 0x100u);
  EXPECT_EQ(p.acc, acc_);
  EXPECT_EQ(p.count, 1024u);
  EXPECT_EQ(p.datatype, ncclFloat16);
  EXPECT_EQ(p.op, ncclMin);
  EXPECT_EQ(p.root, 1);
  EXPECT_EQ(p.comm, comm_.get());
  EXPECT_EQ(p.nRanks, 3);
  EXPECT_EQ(p.stream, stream_);
  EXPECT_EQ(p.nTasks, 34);
  EXPECT_EQ(p.globalRank, 9);
  EXPECT_EQ(p.tid, (int)syscall(SYS_gettid));
  EXPECT_EQ(p.hipDev, 2);
  EXPECT_EQ(p.graphCaptured, 0);
}

// Non-collective formats name the context fields but pass them no destinations, so a full line is
// undefined behaviour to parse; these stop each entry right after "context : [". CommSplit is left
// out: its parse writes the "newcomm" pointer into the int globalRank.
TEST_F(RecorderMicrotest, Parse_OtherCallsRoundTripUpToTheirContext) {
  Recorder& r = StartJson();
  ncclUniqueId id = {};
  id.internal[0] = 5;
  ncclComm_t comms[1];
  float scalar = 1.0f;
  void* handle = reinterpret_cast<void*>(0x4a);
  r.record(rccl::rrCommRegister, comm_.get(), handle, send_, 32);
  r.record(rccl::rrCommDeregister, comm_.get(), handle);
  r.record(rccl::rrGetUniqueId, 0, 0, &id);
  r.record(rccl::rrCommInitDev, 8, 5, &id, comm_.get(), 3);
  r.record(rccl::rrCommInitRank, 8, 6, &id);
  r.record(rccl::rrCommInitRankConfig, 8, 7, &id);
  r.record(comms, 4, nullptr);
  r.record(rccl::rrCommFinalize, comm_.get());
  r.record(rccl::rrCommDestroy, comm_.get());
  r.record(rccl::rrCommAbort, comm_.get());
  r.record(rccl::rrMemAlloc, recv_, 4096);
  r.record(rccl::rrMemFree, recv_);
  r.record(rccl::rrRedOpCreatePreMulSum, ncclMax, comm_.get(), ncclFloat32, ncclScalarHostImmediate, &scalar);
  r.record(rccl::rrRedOpDestroy, ncclProd, comm_.get());
  std::vector<rcclApiCall> p;

  for (const std::string& entry : Entries()) rccl::parseJsonEntry(Head(entry).c_str(), p);

  ASSERT_EQ(p.size(), 14u);
  const uint64_t hash = rccl::hashUniqueId(id);
  EXPECT_EQ(p[0].type, rccl::rrCommRegister);
  EXPECT_EQ(p[0].comm, comm_.get());
  EXPECT_EQ(p[0].sendbuff, static_cast<const void*>(send_));
  EXPECT_EQ(p[0].sendPtrBase, static_cast<void*>(send_ - 0x10));
  EXPECT_EQ(p[0].sendPtrExtent, 0x100u);
  EXPECT_EQ(p[0].recvbuff, handle);
  EXPECT_EQ(p[0].count, 32u);
  EXPECT_EQ(p[1].comm, comm_.get());
  EXPECT_EQ(p[1].recvbuff, handle);
  EXPECT_EQ(p[2].commId, hash);
  EXPECT_EQ(p[3].comm, comm_.get());
  EXPECT_EQ(p[3].nRanks, 8);
  EXPECT_EQ(p[3].commId, hash);
  EXPECT_EQ(p[3].globalRank, 5);
  EXPECT_EQ(p[3].root, 3);
  EXPECT_EQ(p[4].type, rccl::rrCommInitRank);
  EXPECT_EQ(p[4].globalRank, 6);
  EXPECT_EQ(p[5].type, rccl::rrCommInitRankConfig);
  EXPECT_EQ(p[5].nRanks, 8);
  EXPECT_EQ(p[5].commId, hash);
  EXPECT_EQ(p[5].globalRank, 7);
  EXPECT_EQ(p[6].root, 4);
  EXPECT_EQ(p[7].type, rccl::rrCommFinalize);
  EXPECT_EQ(p[7].comm, comm_.get());
  EXPECT_EQ(p[8].type, rccl::rrCommDestroy);
  EXPECT_EQ(p[8].comm, comm_.get());
  EXPECT_EQ(p[9].type, rccl::rrCommAbort);
  EXPECT_EQ(p[9].comm, comm_.get());
  EXPECT_EQ(p[10].recvbuff, static_cast<void*>(recv_));
  EXPECT_EQ(p[10].count, 4096u);
  EXPECT_EQ(p[11].type, rccl::rrMemFree);
  EXPECT_EQ(p[11].recvbuff, static_cast<void*>(recv_));
  EXPECT_EQ(p[12].sendbuff, static_cast<const void*>(&scalar));
  EXPECT_EQ(p[12].datatype, ncclFloat32);
  EXPECT_EQ(p[12].op, ncclMax);
  EXPECT_EQ(p[12].root, ncclScalarHostImmediate);
  EXPECT_EQ(p[12].comm, comm_.get());
  EXPECT_EQ(p[13].op, ncclProd);
  EXPECT_EQ(p[13].comm, comm_.get());
}

TEST_F(RecorderMicrotest, Parse_MalformedEntriesAreSkipped) {
  std::vector<rcclApiCall> p;
  testing::internal::CaptureStdout();
  rccl::parseJsonEntry("", p);
  rccl::parseJsonEntry("   ", p);
  rccl::parseJsonEntry("  {", p);
  rccl::parseJsonEntry(" : CommDestroy", p);
  std::string out = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(p.empty());
  EXPECT_NE(out.find("parseJsonEntry: empty entry string"), std::string::npos) << out;
  EXPECT_NE(out.find("parseJsonEntry: invalid entry format: '   '"), std::string::npos) << out;
  EXPECT_NE(out.find("parseJsonEntry: invalid entry format: '  {'"), std::string::npos) << out;
  EXPECT_NE(out.find("invalid entry format: ' : CommDestroy'"), std::string::npos) << out;
  rccl::parseBinLog();
}

TEST_F(RecorderMicrotest, Parse_UnknownCallNameExits) {
  std::vector<rcclApiCall> p;
  EXPECT_EXIT(rccl::parseJsonEntry("  NotACall : [x", p), ::testing::ExitedWithCode(1), "");
}

}  // namespace
