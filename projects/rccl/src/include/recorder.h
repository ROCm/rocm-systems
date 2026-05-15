/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>
#include <chrono>

namespace rccl
{
// API opcode covered by rccl replayer
typedef enum {
  rrBroadcast,
  rrReduce,
  rrAllGather,
  rrReduceScatter,
  rrAllReduce,
  rrAllReduceWithBias,
  rrSend,
  rrRecv,
  rrAllToAll,
  rrAllToAllv,
  rrGather,
  rrScatter,
  rrBcast,
  rrGroupStart,
  rrGroupEnd,
  rrGroupSimulatedEnd,
  rrGetUniqueId,
  rrCommInitDev,
  rrCommInitRank,
  rrCommInitAll,
  rrCommInitRankConfig,
  rrCommSplit,
  rrCommFinalize,
  rrCommDestroy,
  rrCommAbort,
  rrCommRegister,
  rrCommDeregister,
  rrMemAlloc,
  rrMemFree,
  rrRedOpCreatePreMulSum,
  rrRedOpDestroy,
  rrOtherCall
} rcclCall_t;

constexpr const char* rcclCallStr[]
{
  "Broadcast",
  "Reduce",
  "AllGather",
  "ReduceScatter",
  "AllReduce",
  "AllReduceWithBias",
  "Send",
  "Recv",
  "AllToAll",
  "AllToAllv",
  "Gather",
  "Scatter",
  "Bcast",
  "GroupStart",
  "GroupEnd",
  "GroupSimulatedEnd",
  "GetUniqueId",
  "CommInitDev",
  "CommInitRank",
  "CommInitAll",
  "CommInitRankConfig",
  "CommSplit",
  "CommFinalize",
  "CommDestroy",
  "CommAbort",
  "CommRegister",
  "CommDeregister",
  "MemAlloc",
  "MemFree",
  "RedOpCreatePreMulSum",
  "RedOpDestroy",
  "OtherCall"
};

struct rcclApiCall {
// implicit data
  int                   pid = -1;
  int                   tid = -1;
  int                   hipDev = -1;
  int                   groupDepth = -1;
  double                timestamp = -1;
  unsigned long long    graphID = 0;
  int                   graphCaptured = -1;

// explicit data from header
  rcclCall_t            type;
  uint64_t              opCount = 0;
  const void*           sendbuff = NULL;
  void*                 recvbuff = NULL;
  const void*           acc = NULL;
  void*                 sendPtrBase = NULL;
  void*                 recvPtrBase = NULL;
  size_t                sendPtrExtent = 0;
  size_t                recvPtrExtent = 0;
  size_t                count = 0;
  ncclDataType_t        datatype;
  ncclRedOp_t           op;
  int                   root = -1;
  int                   nRanks = -1;
  ncclComm_t            comm = NULL;
  hipStream_t           stream = NULL;
  int                   nTasks = -1;
  int                   globalRank = -1;
  uint64_t              commId = 0;

  rcclApiCall(){}
  rcclApiCall(rcclCall_t type, const ncclInfo& info);
  rcclApiCall(rcclCall_t type);
};

class Recorder {
 private:
  std::ofstream         outputFile; // 1 per process
  int                   output_json = 0; // 0 is to binary, 1 to json
  std::string           filename;
  int                   logLevel = -1;

  //std::string           hostname;
  int                   pid = -1;
  int                   numCall = 0; // reserved for future record format/debug
  bool                  skipped = false; // number of sendrecv calls to skip for gather/scatter/a2a(v)
  static __thread int   rcclReplayThreadIdx;
  static int            depth; // for indentation purpose, will need thread safty later

  std::mutex            writemtx;
  std::vector<rcclApiCall> calls;

  void                  captureGpuContext(rcclApiCall& call) const;
  void                  write(const rcclApiCall &call);
  static void           recordLater(void* idx);
  Recorder();
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  ~Recorder();

 public:
  static Recorder&      instance();
  void                  skip(bool b);
  void                  record(const char* name); // non-replayable calls
  ncclResult_t          record(rcclApiCall& call);
  ncclResult_t          record(rcclCall_t type, const ncclInfo& info); // collective
  ncclResult_t          record(rcclCall_t type, const void* sendbuff, void* recvbuff, size_t count, // sendrecv based
                               ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream, int root = -1,
                               const size_t sendcounts[] = NULL, const size_t sdispls[] = NULL, const size_t recvcounts[] = NULL,
                               const size_t rdispls[] = NULL); // for alltoallv
  ncclResult_t          record(rcclCall_t type, ncclRedOp_t op, ncclComm_t comm,
                               ncclDataType_t datatype = ncclInt8, ncclScalarResidence_t residence = ncclScalarDevice,
                               void* scalar = NULL); // redop
  ncclResult_t          record(rcclCall_t type, int groupDepth); // group op
  ncclResult_t          record(rcclCall_t type, int size, int rank, ncclUniqueId* commId,
                               ncclComm_t comm = NULL, int device = 0); // init
  void                  record(ncclComm_t* comms, int ndev, const int* devlist); // CommInitAll
  ncclResult_t          record(rcclCall_t type, ncclComm_t comm); // comm destroy
  void                  record(rcclCall_t type, int size, int rank, ncclUniqueId* commId, ncclConfig_t* config,
                               ncclComm_t comm = NULL); // CommInitConfig OR split
  ncclResult_t          record(rcclCall_t type, void* ptr, size_t size = 0); // mem alloc
  ncclResult_t          record(rcclCall_t type, ncclComm_t comm, void* handle,
                               void* userBuffer = NULL, size_t size = 0); // UBR
  void                  record(int groupDepth, ncclSimInfo_t* siminfo); // SimulatedGroupEnd
};

void parseJsonEntry(const char* entry, std::vector<rcclApiCall>& calls);
void parseBinLog();
} // namespace rccl
