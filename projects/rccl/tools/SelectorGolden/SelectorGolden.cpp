/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// SelectorGolden: records the backend RCCL selects for every point of a
// (collective x dtype x op x count x buffer registration x in-place x capture)
// grid, so a selector refactor can be checked against develop decision by
// decision. It only asks rcclGetCollImplInfo (query mode, no side effects) and
// never changes tuning. See README.md.
//
// One MPI rank per GPU. Every rank queries the same grid; rank 0 writes the CSV
// and flags any point where ranks disagree.

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

extern char** environ;

// Exported by librccl (RCCL_EXPOSE_STATIC is always on), not declared in rccl.h.
// coll is an ncclFunc_t; the values below match src/include/nccl_common.h.
extern "C" ncclResult_t rcclGetCollImplInfo(ncclComm_t comm, int coll, uint64_t count, ncclDataType_t dataType,
                                            ncclRedOp_t op, const void* sendbuff, void* recvbuff,
                                            int graphCapturing, int* algo, int* protocol, int* maxChannels);
extern "C" ncclResult_t rcclGetAlgoName(int algo, const char** algoName);

enum : int { kFuncBroadcast = 0, kFuncAllGather = 2, kFuncReduceScatter = 3, kFuncAllReduce = 4, kFuncAlltoAll = 8 };

#define HIPCHECK(cmd)                                                                              \
  do {                                                                                             \
    hipError_t e_ = (cmd);                                                                         \
    if (e_ != hipSuccess) {                                                                        \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__);       \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)
#define NCCLCHECK(cmd)                                                                             \
  do {                                                                                             \
    ncclResult_t r_ = (cmd);                                                                       \
    if (r_ != ncclSuccess) {                                                                       \
      fprintf(stderr, "RCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__);     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

namespace {

struct Coll {
  const char* name;
  int func;
  bool reduces;
  bool perPeerCount;  // count argument is per rank (AG sendcount, RS recvcount, A2A per peer)
};
const Coll kColls[] = {
  {"AllReduce", kFuncAllReduce, true, false},
  {"AllGather", kFuncAllGather, false, true},
  {"ReduceScatter", kFuncReduceScatter, true, true},
  {"AlltoAll", kFuncAlltoAll, false, true},
  {"Broadcast", kFuncBroadcast, false, false},
};

struct DType {
  const char* name;
  ncclDataType_t type;
  size_t size;
};
const DType kDTypes[] = {
  {"float", ncclFloat32, 4},       {"half", ncclFloat16, 2},       {"bfloat16", ncclBfloat16, 2},
  {"fp8e4m3", ncclFloat8e4m3, 1},  {"fp8e5m2", ncclFloat8e5m2, 1}, {"double", ncclFloat64, 8},
  {"int32", ncclInt32, 4},
};

struct Op {
  const char* name;
  ncclRedOp_t op;
};
const Op kOps[] = {{"sum", ncclSum}, {"prod", ncclProd}, {"max", ncclMax}, {"min", ncclMin}, {"avg", ncclAvg}};

// -R levels, as in rccl-tests: 0 plain hipMalloc, 1 ncclMemAlloc + ncclCommRegister,
// 2 ncclMemAlloc + symmetric window.
const char* kRegNames[] = {"R0", "R1", "R2"};

struct Options {
  std::vector<std::string> colls{"AllReduce", "AllGather", "ReduceScatter", "AlltoAll", "Broadcast"};
  std::vector<std::string> dtypes{"float", "half", "bfloat16"};
  std::vector<std::string> ops{"sum"};
  std::vector<int> regs{0, 1, 2};
  size_t minBytes = 8;
  size_t maxBytes = size_t(4) << 30;  // largest buffer allocated per rank
  bool warmup = true;
  int ctaPolicy = -1;                 // -1 = leave config default
  std::string out = "golden.csv";
};

std::vector<std::string> splitList(const char* s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) v.push_back(item);
  return v;
}

size_t parseBytes(const char* s) {
  char* end = nullptr;
  double v = strtod(s, &end);
  switch (end && *end ? *end : ' ') {
  case 'K': case 'k': v *= 1024.0; break;
  case 'M': case 'm': v *= 1024.0 * 1024.0; break;
  case 'G': case 'g': v *= 1024.0 * 1024.0 * 1024.0; break;
  default: break;
  }
  return (size_t)v;
}

void usage() {
  fprintf(stderr,
          "SelectorGolden [options]\n"
          "  -c, --colls LIST     AllReduce,AllGather,ReduceScatter,AlltoAll,Broadcast (default: all)\n"
          "  -d, --dtypes LIST    float,half,bfloat16,fp8e4m3,fp8e5m2,double,int32 (default: float,half,bfloat16)\n"
          "  -o, --ops LIST       sum,prod,max,min,avg (default: sum)\n"
          "  -R, --regs LIST      0,1,2 (default: 0,1,2)\n"
          "  -b, --min-bytes N    smallest message, K/M/G suffixes (default 8)\n"
          "  -e, --max-bytes N    largest buffer per rank (default 4G)\n"
          "  -p, --cta-policy N   ncclConfig_t CTAPolicy (0 default, 1 efficiency, 2 zero)\n"
          "  -w, --no-warmup      skip the warm-up collectives before querying\n"
          "  -f, --out FILE       output CSV on rank 0 (default golden.csv)\n");
}

Options parseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { usage(); MPI_Abort(MPI_COMM_WORLD, 2); }
      return argv[++i];
    };
    if (a == "-c" || a == "--colls") o.colls = splitList(next());
    else if (a == "-d" || a == "--dtypes") o.dtypes = splitList(next());
    else if (a == "-o" || a == "--ops") o.ops = splitList(next());
    else if (a == "-R" || a == "--regs") {
      o.regs.clear();
      for (auto& r : splitList(next())) o.regs.push_back(atoi(r.c_str()));
    } else if (a == "-b" || a == "--min-bytes") o.minBytes = parseBytes(next());
    else if (a == "-e" || a == "--max-bytes") o.maxBytes = parseBytes(next());
    else if (a == "-p" || a == "--cta-policy") o.ctaPolicy = atoi(next());
    else if (a == "-w" || a == "--no-warmup") o.warmup = false;
    else if (a == "-f" || a == "--out") o.out = next();
    else { usage(); MPI_Abort(MPI_COMM_WORLD, 2); }
  }
  return o;
}

template <typename T, size_t N>
const T* findByName(const T (&table)[N], const std::string& name) {
  for (const T& t : table)
    if (name == t.name) return &t;
  fprintf(stderr, "unknown name: %s\n", name.c_str());
  MPI_Abort(MPI_COMM_WORLD, 2);
  return nullptr;
}

// Counts to query for one collective. Selector thresholds are written against
// either the count argument's bytes or the whole buffer, and compare with <, <=
// or >=, so for each power of two in both measures the grid takes the value, one
// element either side, and the midpoint to the next power.
std::vector<uint64_t> countGrid(const Coll& c, const DType& t, int nRanks, const Options& o) {
  const uint64_t perArgMult = c.perPeerCount ? (uint64_t)nRanks : 1;  // buffer elems per count
  const uint64_t maxCount = o.maxBytes / (t.size * perArgMult);
  std::set<uint64_t> counts;
  auto add = [&](uint64_t cnt) {
    if (cnt == 0 || cnt > maxCount) return;
    if (cnt * t.size * perArgMult < o.minBytes) return;
    counts.insert(cnt);
  };
  for (int k = 0; k < 63; k++) {
    const uint64_t p = uint64_t(1) << k;
    if (p < t.size) continue;
    // Power of two in the count argument's bytes.
    const uint64_t c1 = p / t.size;
    add(c1 - 1); add(c1); add(c1 + 1); add(c1 + c1 / 2);
    // Power of two in the whole buffer's bytes.
    if (perArgMult > 1) {
      const uint64_t c2 = p / (t.size * perArgMult);
      add(c2 - 1); add(c2); add(c2 + 1);
    }
    if (p / (t.size * perArgMult) > maxCount * 2) break;
  }
  return std::vector<uint64_t>(counts.begin(), counts.end());
}

struct Buffers {
  int reg = 0;
  void* send = nullptr;
  void* recv = nullptr;
  void* sendHandle = nullptr;
  void* recvHandle = nullptr;
  ncclWindow_t sendWin = nullptr;
  ncclWindow_t recvWin = nullptr;
};

Buffers allocBuffers(ncclComm_t comm, int reg, size_t bytes) {
  Buffers b;
  b.reg = reg;
  if (reg == 0) {
    HIPCHECK(hipMalloc(&b.send, bytes));
    HIPCHECK(hipMalloc(&b.recv, bytes));
  } else {
    NCCLCHECK(ncclMemAlloc(&b.send, bytes));
    NCCLCHECK(ncclMemAlloc(&b.recv, bytes));
    if (reg == 1) {
      NCCLCHECK(ncclCommRegister(comm, b.send, bytes, &b.sendHandle));
      NCCLCHECK(ncclCommRegister(comm, b.recv, bytes, &b.recvHandle));
    } else {
      NCCLCHECK(ncclCommWindowRegister(comm, b.send, bytes, &b.sendWin, NCCL_WIN_COLL_SYMMETRIC));
      NCCLCHECK(ncclCommWindowRegister(comm, b.recv, bytes, &b.recvWin, NCCL_WIN_COLL_SYMMETRIC));
    }
  }
  HIPCHECK(hipMemset(b.send, 0, bytes));
  HIPCHECK(hipMemset(b.recv, 0, bytes));
  return b;
}

void freeBuffers(ncclComm_t comm, Buffers& b) {
  if (b.reg == 0) {
    HIPCHECK(hipFree(b.send));
    HIPCHECK(hipFree(b.recv));
    return;
  }
  if (b.reg == 1) {
    NCCLCHECK(ncclCommDeregister(comm, b.sendHandle));
    NCCLCHECK(ncclCommDeregister(comm, b.recvHandle));
  } else {
    NCCLCHECK(ncclCommWindowDeregister(comm, b.sendWin));
    NCCLCHECK(ncclCommWindowDeregister(comm, b.recvWin));
  }
  NCCLCHECK(ncclMemFree(b.send));
  NCCLCHECK(ncclMemFree(b.recv));
}

// Pointers the way rccl-tests passes them, so in-place alignment and window
// offsets match a real run.
void operands(const Coll& c, const Buffers& b, uint64_t count, size_t esize, int rank, bool inPlace,
              const void** send, void** recv) {
  *recv = b.recv;
  *send = b.send;
  if (!inPlace) return;
  char* r = (char*)b.recv;
  switch (c.func) {
  case kFuncAllGather: *send = r + (size_t)rank * count * esize; break;
  case kFuncReduceScatter:
    *send = b.recv;
    *recv = r + (size_t)rank * count * esize;
    break;
  default: *send = b.recv; break;
  }
}

// One collective per backend RCCL may set up lazily (CE staging, DDA scratch),
// so queries see the steady state a benchmark sees after its warm-up iterations.
void warmup(ncclComm_t comm, const Buffers& b, size_t bufBytes, int nRanks, hipStream_t s) {
  const size_t sizes[] = {4096, std::min<size_t>(bufBytes, size_t(64) << 20)};
  for (size_t bytes : sizes) {
    const size_t n = bytes / sizeof(float);
    const size_t perPeer = std::max<size_t>(n / nRanks, 1);
    NCCLCHECK(ncclAllReduce(b.send, b.recv, n, ncclFloat32, ncclSum, comm, s));
    NCCLCHECK(ncclAllGather(b.send, b.recv, perPeer, ncclFloat32, comm, s));
    NCCLCHECK(ncclReduceScatter(b.send, b.recv, perPeer, ncclFloat32, ncclSum, comm, s));
    NCCLCHECK(ncclAllToAll(b.send, b.recv, perPeer, ncclFloat32, comm, s));
    NCCLCHECK(ncclBroadcast(b.send, b.recv, n, ncclFloat32, 0, comm, s));
  }
  HIPCHECK(hipStreamSynchronize(s));
}

const char* protoName(int p) {
  switch (p) {
  case 0: return "LL";
  case 1: return "LL128";
  case 2: return "Simple";
  default: return "?";
  }
}

struct Point {
  uint16_t coll, dtype, op, reg, inPlace, capture;
  uint64_t count;
};
struct Result {
  int32_t algo, proto, channels, status;
};

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, nRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  Options o = parseArgs(argc, argv);

  // One GPU per rank on each host.
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  int localRank, localSize;
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localSize);
  int nDev = 0;
  HIPCHECK(hipGetDeviceCount(&nDev));
  HIPCHECK(hipSetDevice(localRank % nDev));
  int nHosts = 0, isLeader = localRank == 0 ? 1 : 0;
  MPI_Allreduce(&isLeader, &nHosts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  if (o.ctaPolicy >= 0) config.CTAPolicy = o.ctaPolicy;
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  std::vector<const Coll*> colls;
  std::vector<const DType*> dtypes;
  std::vector<const Op*> ops;
  for (auto& n : o.colls) colls.push_back(findByName(kColls, n));
  for (auto& n : o.dtypes) dtypes.push_back(findByName(kDTypes, n));
  for (auto& n : o.ops) ops.push_back(findByName(kOps, n));

  std::vector<Point> points;
  std::vector<Result> results;
  for (int reg : o.regs) {
    Buffers b = allocBuffers(comm, reg, o.maxBytes);
    if (o.warmup) warmup(comm, b, o.maxBytes, nRanks, stream);
    for (const Coll* c : colls) {
      for (const DType* t : dtypes) {
        for (const Op* op : ops) {
          if (!c->reduces && op != ops.front()) continue;  // op is not an input there
          for (uint64_t count : countGrid(*c, *t, nRanks, o)) {
            for (int inPlace = 0; inPlace < 2; inPlace++) {
              for (int capture = 0; capture < 2; capture++) {
                const void* send;
                void* recv;
                operands(*c, b, count, t->size, rank, inPlace, &send, &recv);
                Result r = {-1, -1, -1, 0};
                r.status = (int)rcclGetCollImplInfo(comm, c->func, count, t->type, op->op, send, recv, capture,
                                                    &r.algo, &r.proto, &r.channels);
                Point p = {(uint16_t)(c - kColls), (uint16_t)(t - kDTypes), (uint16_t)(op - kOps),
                           (uint16_t)reg, (uint16_t)inPlace, (uint16_t)capture, count};
                points.push_back(p);
                results.push_back(r);
              }
            }
          }
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    freeBuffers(comm, b);
  }

  // Rank 0 compares every rank's decisions; all ranks ran the identical grid.
  const int nInts = (int)(results.size() * 4);
  std::vector<Result> all;
  if (rank == 0) all.resize(results.size() * nRanks);
  MPI_Gather(results.data(), nInts, MPI_INT32_T, all.data(), nInts, MPI_INT32_T, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    FILE* f = fopen(o.out.c_str(), "w");
    if (!f) { perror(o.out.c_str()); MPI_Abort(MPI_COMM_WORLD, 1); }
    int version = 0;
    ncclGetVersion(&version);
    hipDeviceProp_t prop;
    int dev;
    HIPCHECK(hipGetDevice(&dev));
    HIPCHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(f, "# SelectorGolden v1\n");
    fprintf(f, "# rccl_version=%d\n", version);
    fprintf(f, "# arch=%s\n", prop.gcnArchName);
    fprintf(f, "# nranks=%d\n", nRanks);
    fprintf(f, "# nhosts=%d\n", nHosts);
    fprintf(f, "# ranks_per_host=%d\n", localSize);
    fprintf(f, "# cta_policy=%d\n", o.ctaPolicy);
    fprintf(f, "# warmup=%d\n", (int)o.warmup);
    fprintf(f, "# max_bytes=%zu\n", o.maxBytes);
    for (char** e = environ; *e; e++)
      if (!strncmp(*e, "NCCL_", 5) || !strncmp(*e, "RCCL_", 5)) fprintf(f, "# env %s\n", *e);
    fprintf(f, "coll,dtype,op,count,bytes,reg,inplace,capture,algo,algo_name,proto,channels,status,ranks_agree\n");
    size_t disagree = 0;
    for (size_t i = 0; i < points.size(); i++) {
      const Point& p = points[i];
      const Result& r = results[i];
      int agree = 1;
      for (int k = 1; k < nRanks; k++) {
        const Result& q = all[(size_t)k * results.size() + i];
        if (q.algo != r.algo || q.proto != r.proto || q.channels != r.channels || q.status != r.status) agree = 0;
      }
      disagree += !agree;
      const Coll& c = kColls[p.coll];
      const DType& t = kDTypes[p.dtype];
      const uint64_t bytes = p.count * t.size * (c.perPeerCount ? (uint64_t)nRanks : 1);
      const char* aname = "?";
      if (r.status == 0 && rcclGetAlgoName(r.algo, &aname) != ncclSuccess) aname = "?";
      fprintf(f, "%s,%s,%s,%llu,%llu,%s,%d,%d,%d,%s,%s,%d,%d,%d\n", c.name, t.name,
              c.reduces ? kOps[p.op].name : "-", (unsigned long long)p.count, (unsigned long long)bytes,
              kRegNames[p.reg], p.inPlace, p.capture, r.algo, aname, protoName(r.proto), r.channels, r.status,
              agree);
    }
    fclose(f);
    printf("SelectorGolden: wrote %zu decisions to %s (%zu with ranks disagreeing)\n", points.size(),
           o.out.c_str(), disagree);
  }

  HIPCHECK(hipStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));
  MPI_Comm_free(&local);
  MPI_Finalize();
  return 0;
}
