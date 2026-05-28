/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#include "bootstrap.h"
#include "group.h"
#include "utils.h"
#include <cstring>
#include <string>
#include <iomanip>
#include <sys/syscall.h>
#include "debug.h"
 
using namespace std::chrono;

namespace rccl
{
__thread int Recorder::rcclReplayThreadIdx = -1;
int Recorder::depth = 0;

static char buffer[4096];
static rcclCall_t lastcall = rrGroupStart; // for trailing comma, need modify for multithread case for Bcast
void indent(int depth, std::ofstream& o) {for (int i = 0; i < depth; i++) o << " ";}
void newLine(std::ofstream& o) {if (lastcall != rrGroupStart) o << ","; o << std::endl;}
static uint64_t hashUniqueId(ncclUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for(int i=0; i < (int)sizeof(ncclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

rcclApiCall::rcclApiCall(rcclCall_t type, const ncclInfo& info)://name(rcclCallStr[call.type]), // opName
                                                                type(type),
                                                                opCount(info.comm->opCount),
                                                                sendbuff(info.sendbuff),
                                                                recvbuff(info.recvbuff),
                                                                acc(info.acc),
                                                                count(info.count),
                                                                datatype(info.datatype),
                                                                op(info.op),
                                                                root(info.root),
                                                                nRanks(info.comm->nRanks),
                                                                comm(info.comm),
                                                                stream(info.stream),
                                                                nTasks(info.comm->planner.nTasksP2p + info.comm->planner.nTasksColl),
                                                                globalRank(info.comm->localRankToRank[info.comm->localRank])
{
  hipMemGetAddressRange(&recvPtrBase, &recvPtrExtent, const_cast<void*>(info.recvbuff)); // should always exist for collectives
  if (info.sendbuff) // ncclSend/Recv
  {
    hipMemGetAddressRange(&sendPtrBase, &sendPtrExtent, const_cast<void*>(info.sendbuff));
  }
}

rcclApiCall::rcclApiCall(rcclCall_t type) : type(type){}

std::string siminfo_fmt = "[size : %zu, magic : %u, version : %u, estimated time : %f, timestamp : %f]";
std::string config_fmt = ", ncclConfig : [size : %zu, magic : %u, version : %u, blocking : %d, cgaClusterSize : %d, minCTA : %d, maxCTA : %d, netname : %s, splitshare : %d]";
std::string ctxt_fmt = "time : %lf, thread : %d, device : %d, captured : %d, graphID : %llu ]]"; // implicit context info
std::string ubr_fmt = "%s : [comm : %p, buff : [addr : %p, base : %p, size : %zu], returned handle : %p, count : %zu, context : [";
std::string getId_fmt = "%s : [uniqueID : %llu, context : [";
std::string ubDereg_fmt = "%s : [comm : %p, handle : %p, context : [";
std::string rank_fmt = "%s : [size : %d, uniqueID : %llu, rank : %d, context : [";
std::string init_fmt = "%s : [comm : %p, size : %d, uniqueID : %llu, rank : %d, dev : %d, context : [";
std::string all_fmt = "%s : [# of device : %d, context : [";
std::string destroy_fmt = "%s : [comm : %p, context : [";
std::string split_fmt = "%s : [comm : %p, color : %d, key : %d, newcomm : %p, context : [";
std::string alloc_fmt = "%s : [returned ptr : %p, size : %zu, context : [";
std::string free_fmt = "%s : [ptr : %p, context : [";
std::string redop_fmt = "%s : [scalar : %p, datatype : %d, op : %d, residence : %d, comm : %p, context : [";
std::string redopdestroy_fmt = "%s : [op : %d, comm : %p, context : [";
std::string coll_fmt = "%s : [opCount : %lx, sendbuff : [addr : %p, base : %p, size : %zu], recvbuff : [addr : %p, base : %p, size : %zu], acc : %p, count : %zu, datatype : %d, op : %d, root : %d, comm : %p, nranks : %d, stream : %p, task : %d, globalrank : %d, context : [";

Recorder::Recorder()
{
  filename = getenv("RCCL_REPLAY_FILE") ? getenv("RCCL_REPLAY_FILE") : "";

  if (!filename.size())
  {
    return;
  }

  logLevel = getenv("RCCL_LOG_LEVEL") ? std::stoi(getenv("RCCL_LOG_LEVEL")) : 1;
  char hostname[256];
  gethostname(hostname, 256);
  pid = getpid();
  output_json = 0;

  size_t dot;
  std::string output_name, output_extension;

  if ((dot = std::string(filename).find(".")) != std::string::npos)
  {
    output_name = std::string(filename).substr(0, dot);
    output_extension = std::string(filename).substr(dot);
    if (output_extension.compare(".json") == 0)
    {
      output_json = 1;
    }
  } else {
    output_name = std::string(filename);
  }

  outputFile.open(output_name + "." + std::to_string(pid) + "." + std::string(hostname) + output_extension,
                  output_json ? std::ofstream::out : std::ofstream::binary);
  if (output_json)
  {
    outputFile << "{" << std::endl;
    indent(2, outputFile);
    outputFile << "version : 1,";
  }
}

Recorder& Recorder::instance()
{
  static Recorder _instance;
  return _instance;
}

void Recorder::skip(bool b)
{
  if (filename.size())
  {
    skipped = b;
  }
}

void Recorder::captureGpuContext(rcclApiCall& call) const
{
  call.timestamp = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count() * 1000;

  if (rcclReplayThreadIdx == -1)
  {
    rcclReplayThreadIdx = syscall(SYS_gettid);
  }

  int hipDev;
  hipGetDevice(&hipDev); // need later change to copy from comm

  call.pid = pid;
  call.tid = rcclReplayThreadIdx;
  call.hipDev = hipDev;
  return;
}

// for single process use only, for now
// TODO: potentially need async logging for performance
void Recorder::write(const rcclApiCall &call)
{
  if (!filename.size())
  {
    return ;
  }

  std::unique_lock<std::mutex> lock(writemtx);

  if (lastcall == rrBcast && call.type == rrBroadcast)
  {
    return;
  }

  int len = -1;

  if (output_json)
  {
    if (call.type == rrGroupEnd || call.type == rrGroupSimulatedEnd)
    {
      depth--;
      outputFile << std::endl;
      indent(2 + 2 * depth, outputFile);
      outputFile << "}";
      lastcall = call.type;
      return ;
    }

    newLine(outputFile);
    indent(2 + 2 * depth, outputFile);
    lastcall = call.type;
    switch (call.type) {
    case rrGroupStart:
    {
      outputFile << "{";
      depth++;
      return ;
    }
    case rrCommRegister:
    {
      len = snprintf(buffer, 4096, ubr_fmt.c_str(),
                     rcclCallStr[call.type], call.comm, call.sendbuff, call.sendPtrBase, call.sendPtrExtent, call.recvbuff, call.count);
      break;
    }
    case rrCommDeregister:
    {
      len = snprintf(buffer, 4096, ubDereg_fmt.c_str(),
                     rcclCallStr[call.type], call.comm, call.recvbuff);
      break;
    }
    case rrGetUniqueId:
    {
      len = snprintf(buffer, 4096, getId_fmt.c_str(), rcclCallStr[call.type], call.commId);
      break;
    }
    case rrCommInitDev:
    {
      len = snprintf(buffer, 4096, init_fmt.c_str(), rcclCallStr[call.type], call.comm, call.nRanks, call.commId, call.globalRank, call.root);
      break;
    }
    case rrCommInitRank:
    case rrCommInitRankConfig:
    {
      len = snprintf(buffer, 4096, rank_fmt.c_str(), rcclCallStr[call.type], call.nRanks, call.commId, call.globalRank);
      break; // detail to be provided by init dev or config info
    }
    case rrCommInitAll:
    {
      len = snprintf(buffer, 4096, all_fmt.c_str(), rcclCallStr[call.type], call.root);
      break;
    }
    case rrCommFinalize:
    case rrCommDestroy:
    case rrCommAbort:
    {
      len = snprintf(buffer, 4096, destroy_fmt.c_str(), rcclCallStr[call.type], call.comm);
      break;
    }
    case rrCommSplit:
    {
      len = snprintf(buffer, 4096, split_fmt.c_str(),
                     rcclCallStr[call.type], (void*)(call.commId), call.comm, call.nRanks, call.globalRank, call.comm);
      break;
    }
    case rrMemAlloc:
    {
      len = snprintf(buffer, 4096, alloc_fmt.c_str(),
                     rcclCallStr[call.type], call.recvbuff, call.count);
      break;
    }
    case rrMemFree:
    {
      len = snprintf(buffer, 4096, free_fmt.c_str(),
                     rcclCallStr[call.type], call.recvbuff);
      break;
    }
    case rrRedOpCreatePreMulSum:
    {
      len = snprintf(buffer, 4096, redop_fmt.c_str(),
                     rcclCallStr[call.type], call.sendbuff, call.datatype, call.op, call.root, call.comm);
      break;
    }
    case rrRedOpDestroy:
    {
      len = snprintf(buffer, 4096, redopdestroy_fmt.c_str(),
                     rcclCallStr[call.type], call.op, call.comm);
      break;
    }
    default: // collectives
      len = snprintf(buffer, 4096, coll_fmt.c_str(),
                     rcclCallStr[call.type], call.opCount, call.sendbuff, call.sendPtrBase, call.sendPtrExtent,
                     call.recvbuff, call.recvPtrBase, call.recvPtrExtent, call.acc, call.count, call.datatype,
                     call.op, call.root, call.comm, call.nRanks, call.stream, call.nTasks, call.globalRank);

    }
    outputFile.write(buffer, len);
    len = snprintf(buffer, 4096, ctxt_fmt.c_str(), call.timestamp, call.tid, call.hipDev, call.graphCaptured, call.graphID);
    outputFile.write(buffer, len);
  } else {
    outputFile.write((char*)&call, sizeof(rcclApiCall));
  }
  outputFile.flush();
  return ;
}

// wrapper function for graph launch callback
void Recorder::recordLater(void* idx)
{
  Recorder& recorder = Recorder::instance();
  size_t callidx = (size_t) idx;
  rcclApiCall call = recorder.calls[callidx];
  recorder.record(call);
}

void Recorder::record(const char* name)
{
  if (!filename.size() || logLevel <= 1)
  {
    return;
  }
  double ts = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count() * 1000;
  if (output_json) // will not record for binary replay export
  {
    std::unique_lock<std::mutex> lock(writemtx);
    newLine(outputFile); lastcall = rrOtherCall;
    indent(2 + 2 * depth, outputFile);
    outputFile << name << " : [time : " << std::fixed << std::setprecision(6) << ts << "]";
  }
  //numCall++;
  outputFile.flush();
}

ncclResult_t Recorder::record(rcclApiCall& call)
{
  ncclResult_t ret = ncclSuccess;
  captureGpuContext(call);

  switch (call.type) {
  case rrGroupStart:
  case rrGroupEnd:
  case rrGroupSimulatedEnd:
  case rrGetUniqueId:
  case rrCommInitDev:
  case rrCommInitRank:
  case rrCommInitAll:
  case rrCommInitRankConfig:
  case rrCommSplit:
  case rrCommFinalize:
  case rrCommDestroy:
  case rrCommAbort: // communicator ops just exit and write
  case rrCommRegister: // same with UBR
  case rrCommDeregister:
  case rrMemAlloc:
  case rrMemFree:
  case rrRedOpCreatePreMulSum:
  case rrRedOpDestroy:
    break;

  // collectives, which may be registered with graph
  case rrAllToAll: // should work with rest of the coll with nested sendrecv
  default: // collective other than a2a/a2av
  #if ROCM_VERSION >= 60100
    hipStreamCaptureStatus status;
    hipGraph_t graphCaptured;
    CUDACHECK(hipStreamGetCaptureInfo_v2(call.stream, &status, &(call.graphID), &graphCaptured)); // shouldnt we need dependency?

    if (status == hipStreamCaptureStatusActive) // when graph launched this should be disabled
    {
      call.graphCaptured = 1;
      calls.push_back(call);
      hipGraphNode_t logNode;
      hipHostNodeParams p;
      p.fn = &(Recorder::recordLater);
      p.userData = (void*) (calls.size() - 1);
      CUDACHECK(hipGraphAddHostNode(&logNode, graphCaptured, nullptr, 0, &p));
    } else {
      call.graphCaptured = 0;
    }
  #endif
  }

  write(call); // write immediatelly
  return ret;
}

ncclResult_t Recorder::record(rcclCall_t type, const ncclInfo& info)
{
  if (!filename.size() || skipped)
  {
    return ncclSuccess;
  }

  rcclApiCall call(type, info);
  return record(call);
}

ncclResult_t Recorder::record(rcclCall_t type, const void* sendbuff, void* recvbuff,
                              size_t count, ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream, int root,
                              const size_t sendcounts[], const size_t sdispls[], const size_t recvcounts[], const size_t rdispls[])
{
  if (!filename.size())
  {
    return ncclSuccess;
  }

  rcclApiCall call(type, {.sendbuff = sendbuff, .recvbuff = recvbuff, .count = count,
                          .datatype = datatype, .comm = comm, .stream = stream});
  if (root != -1)
  {
    call.root = root;
  }

  ncclResult_t ret = record(call);
  if (type == rrAllToAllv)
  {
    int size = call.nRanks - 1;
    if (output_json)
    {
      outputFile << ", sendcounts : [";
      for (int i = 0; i < size; i++) outputFile << sendcounts[i] << ", ";
      outputFile << sendcounts[size] << "], sdispls : [";
      for (int i = 0; i < size; i++) outputFile << sdispls[i] << ", ";
      outputFile << sdispls[size] << "], recvcounts : [";
      for (int i = 0; i < size; i++) outputFile << recvcounts[i] << ", ";
      outputFile << recvcounts[size] << "], rdispls : [";
      for (int i = 0; i < size; i++) outputFile << rdispls[i] << ", ";
      outputFile << rdispls[size] << "]";
    } else {
      outputFile.write((char*)sendcounts, sizeof(size_t) * (size + 1));
      outputFile.write((char*)sdispls, sizeof(size_t) * (size + 1));
      outputFile.write((char*)recvcounts, sizeof(size_t) * (size + 1));
      outputFile.write((char*)rdispls, sizeof(size_t) * (size + 1));
    }
    outputFile.flush();
  }
  return ret;
}

ncclResult_t Recorder::record(rcclCall_t type, ncclRedOp_t op, ncclComm_t comm, ncclDataType_t datatype, ncclScalarResidence_t residence, void* scalar)
{
  if (!filename.size())
  {
    return ncclSuccess;
  }

  rcclApiCall call(type, {.op = op, .comm = comm});
  if (type == rrRedOpCreatePreMulSum)
  {
    call.sendbuff = scalar;
    call.datatype = datatype;
    call.root = residence;
    call.sendbuff = scalar;
  }
  return record(call);
  //TODO: printout scalar
}

ncclResult_t Recorder::record(rcclCall_t type, int groupDepth)
{
  if (!filename.size() || skipped)
  {
    return ncclSuccess;
  }
  rcclApiCall gc(type);
  gc.groupDepth = groupDepth;
  return record(gc);
}

ncclResult_t Recorder::record(rcclCall_t type, int size, int rank, ncclUniqueId* commId, ncclComm_t comm, int device)
{
  if (!filename.size())
  {
    return ncclSuccess;
  }

  rcclApiCall initCall(type);

  if (type == rrCommSplit)
  {
    initCall.comm = comm;
    initCall.commId = (uint64_t)commId;
  } else {
    initCall.commId = hashUniqueId(*commId);
  }

  initCall.nRanks = size;
  initCall.globalRank = rank;
  if (type == rrCommInitDev)
  {
    initCall.root = device;
    initCall.comm = comm;
  }

  return record(initCall);
}

// comm destroy
ncclResult_t Recorder::record(rcclCall_t type, ncclComm_t comm)
{
  if (!filename.size())
  {
    return ncclSuccess;
  }

  rcclApiCall call(type);
  call.comm = comm;
  return record(call);
}

ncclResult_t Recorder::record(rcclCall_t type, ncclComm_t comm, void* handle, void* userBuffer, size_t size)
{
  if (!filename.size())
  {
    return ncclSuccess;
  }
  rcclApiCall call(type);
  call.comm = comm;
  call.recvbuff = handle;
  if (type == rrCommRegister)
  {
    CUDACHECK(hipMemGetAddressRange(&call.sendPtrBase, &call.sendPtrExtent, userBuffer));
    call.sendbuff = userBuffer;
    call.count = size;
  }
  return record(call);
}

ncclResult_t Recorder::record(rcclCall_t type, void* ptr, size_t size)
{
  if (!filename.size())
  {
    return ncclSuccess;
  }
  rcclApiCall call(type);
  call.recvbuff = ptr;
  if (type == rrMemAlloc)
  {
    call.count = size;
  }
  return record(call);
}

void Recorder::record(int groupDepth, ncclSimInfo_t *siminfo)
{
  if (!filename.size())
  {
    return;
  }
  rcclApiCall call(rrGroupSimulatedEnd);
  record(call);

  if (output_json && siminfo)
  {
    int len = snprintf(buffer, 4096, siminfo_fmt.c_str(),
                       siminfo->size, siminfo->magic, siminfo->version, siminfo->estimatedTime, call.timestamp);
    outputFile.write(buffer, len);
  } // no tid for groupCall
  // TODO: else flush siminfo in binary
  outputFile.flush();
}

void Recorder::record(rcclCall_t type, int size, int rank, ncclUniqueId* commId, ncclConfig_t* config, ncclComm_t comm)
{
  if (!filename.size())
  {
    return ;
  }

  if (type == rrCommInitRankConfig)
  {
    record(type, size, rank, commId);
  }
  else //rrCommSplit
  {
    record(type, size /*color*/, rank/*key*/, commId, comm);
  }

  if (output_json && config)
  {
    int len = snprintf(buffer, 4096, config_fmt.c_str(), config->size, config->magic, config->version, config->blocking,
                       config->cgaClusterSize, config->minCTAs, config->maxCTAs, config->netName, config->splitShare);
    outputFile.write(buffer, len);
    outputFile.flush();
  }
  // TODO: else flush ncclConfig in binary
}

void Recorder::record(ncclComm_t* comms, int ndev, const int* devlist)
{
  if (!filename.size())
  {
    return ;
  }

  rcclApiCall call(rrCommInitAll);
  call.root = ndev;
  call.sendbuff = devlist;
  record(call);

  if (devlist)
  {
    if (output_json)
    {
      outputFile << ", devlist : [";
      for (int i = 0; i < call.root - 1; i++)
        outputFile << devlist[i] << ", ";
      outputFile << devlist[call.root - 1] << "]";
    } else {
      outputFile.write((char*)devlist, sizeof(int) * ndev);
    }
    outputFile.flush();
  }
}

Recorder::~Recorder()
{
  if (outputFile.is_open())
  {
    if (output_json) outputFile << std::endl << "}" << std::endl;
    outputFile.close();
    calls.clear();
  }
}

static rcclCall_t getFuncType(std::string func)
{
  for (int i = 0; i < sizeof(rcclCallStr) / sizeof(char*); i++)
  {
    if (func == std::string(rcclCallStr[i]))
    {
      return (rcclCall_t)i;
    }
  }
  printf("[ERROR] Unrecognized func %s\n", func.c_str());
  exit(1);
}

void parseJsonEntry(const char* entry, std::vector<rcclApiCall>& calls)
{
  // TODO: parse comma too
  rcclApiCall call;
  std::string str(entry);
  size_t begin = str.find_first_not_of(' ');
  size_t end = str.find(" : ");
  rcclCall_t type = getFuncType(str.substr(begin, end-begin));
  call.type = type;
  switch(type) {
  case rrCommRegister:
  {
    assert(sscanf(str.c_str() + end + 3, (ubr_fmt.substr(5) + ctxt_fmt).c_str(),
                  &call.comm, &call.sendbuff, &call.sendPtrBase, &call.sendPtrExtent, &call.recvbuff, &call.count) == 6);
    break;
  }
  case rrCommDeregister:
  {
    assert(sscanf(str.c_str() + end + 3, (ubDereg_fmt.substr(5) + ctxt_fmt).c_str(),
                  &call.comm, &call.recvbuff) == 2);
    break;
  }
  case rrGetUniqueId:
  {
    assert(sscanf(str.c_str() + end + 3, (getId_fmt.substr(5) + ctxt_fmt).c_str(), &call.commId) == 1);
    break;
  }
  case rrCommInitDev:
  {
    assert(sscanf(str.c_str() + end + 3, (init_fmt.substr(5) + ctxt_fmt).c_str(),
                  &call.comm, &call.nRanks, &call.commId, &call.globalRank, &call.root) == 5);
    break;
  }
  case rrCommInitRank:
  case rrCommInitRankConfig:
  {
    assert(sscanf(str.c_str() + end + 3, (rank_fmt.substr(5) + ctxt_fmt).c_str(),
           &call.nRanks, &call.commId, &call.globalRank) == 3);
    break;
  }
  case rrCommInitAll:
  {
    assert(sscanf(str.c_str() + end + 3, (all_fmt.substr(5) + ctxt_fmt).c_str(), &call.root) == 1);
    break;
  }
  case rrCommFinalize:
  case rrCommDestroy:
  case rrCommAbort:
  {
    assert(sscanf(str.c_str() + end + 3, (destroy_fmt.substr(5) + ctxt_fmt).c_str(), &call.comm) == 1);
    break;
  }
  case rrCommSplit:
  {
    assert(sscanf(str.c_str() + end + 3, (split_fmt.substr(5) + ctxt_fmt).c_str(),
           &call.commId, &call.comm, &call.nRanks, &call.globalRank, &call.comm) == 5);
    break;
  }
  case rrMemAlloc:
  {
    assert(sscanf(str.c_str() + end + 3, (alloc_fmt.substr(5) + ctxt_fmt).c_str(),
           &call.recvbuff, &call.count) == 2);
    break;
  }
  case rrMemFree:
  {
    assert(sscanf(str.c_str() + end + 3, (free_fmt.substr(5) + ctxt_fmt).c_str(), &call.recvbuff) == 1);
    break;
  }
  case rrRedOpCreatePreMulSum:
  {
    assert(sscanf(str.c_str() + end + 3, (redop_fmt.substr(5) + ctxt_fmt).c_str(),
           &call.sendbuff, &call.datatype, &call.op, &call.root, &call.comm) == 5);
    break;
  }
  case rrRedOpDestroy:
  {
    assert(sscanf(str.c_str() + end + 3, (redopdestroy_fmt.substr(5) + ctxt_fmt).c_str(),
           &call.op, &call.comm) == 2);
    break;
  }
  default:
    assert(sscanf(str.c_str() + end + 3, (coll_fmt.substr(5) + ctxt_fmt).c_str(),
                  &call.opCount, &call.sendbuff, &call.sendPtrBase, &call.sendPtrExtent, &call.recvbuff, &call.recvPtrBase, &call.recvPtrExtent,
                  &call.acc, &call.count, &call.datatype, &call.op, &call.root,
                  &call.comm, &call.nRanks, &call.stream, &call.nTasks, &call.globalRank, &call.timestamp, &call.tid,
                  &call.hipDev, &call.graphCaptured, &call.graphID) == 22);
  }
  calls.push_back(call);
}

void parseBinLog()
{
  // TODO: need to handle trailing data such as simInfo, devList, a2av data, etc.
}
};
