/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl_ab_perf: A/B performance comparison of two RCCL builds inside a single
// process.
//
// Both librccl.so files are dlopen()ed with RTLD_LOCAL (and this binary does
// not link against RCCL), so each library binds to its own copy of every
// nccl* symbol and keeps its own global state.  One set of communicators is
// created per library on the same devices; both sets share the same streams
// and user buffers.  For every message size the timed batches of the two
// libraries are interleaved (A B B A ...) so that clock, thermal and system
// drift affect both libraries equally, and the speedup is computed from the
// paired per-round ratios.

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <getopt.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#ifdef MPI_SUPPORT
#include <mpi.h>
#endif

// Minimal, version-independent subset of the RCCL ABI.  All enums are passed as
// int, which matches the C ABI of the enum types declared in rccl.h.
typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef struct ncclComm *ncclComm_t;
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

static int worldRank = 0, worldSize = 1, localRank = 0;

[[noreturn]] static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  fprintf(stderr, "[%s:rank %d] rccl_ab_perf error: ", host, worldRank);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  fflush(stderr);
#ifdef MPI_SUPPORT
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  exit(1);
}

static void print0(const char *fmt, ...) {
  if (worldRank != 0)
    return;
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
}

#define HIPCHECK(cmd)                                                          \
  do {                                                                         \
    hipError_t e_ = (cmd);                                                     \
    if (e_ != hipSuccess)                                                      \
      die("%s:%d HIP error '%s' in %s", __FILE__, __LINE__,                    \
          hipGetErrorString(e_), #cmd);                                        \
  } while (0)

#define NCCLCHECK(lib, cmd)                                                    \
  do {                                                                         \
    ncclResult_t r_ = (cmd);                                                   \
    if (r_ != 0)                                                               \
      die("%s:%d [lib %s] RCCL error %d '%s' in %s", __FILE__, __LINE__,       \
          (lib).label.c_str(), r_,                                             \
          (lib).GetErrorString ? (lib).GetErrorString(r_) : "?", #cmd);        \
  } while (0)

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

struct RcclLib {
  std::string label;
  std::string spec;
  std::string path;
  std::string commit;
  std::string subject;
  void *handle = nullptr;
  int version = 0;

  ncclResult_t (*GetVersion)(int *) = nullptr;
  ncclResult_t (*GetUniqueId)(ncclUniqueId *) = nullptr;
  ncclResult_t (*CommInitRank)(ncclComm_t *, int, ncclUniqueId, int) = nullptr;
  ncclResult_t (*CommDestroy)(ncclComm_t) = nullptr;
  const char *(*GetErrorString)(ncclResult_t) = nullptr;
  ncclResult_t (*GroupStart)() = nullptr;
  ncclResult_t (*GroupEnd)() = nullptr;
  ncclResult_t (*AllReduce)(const void *, void *, size_t, ncclDataType_t,
                            ncclRedOp_t, ncclComm_t, hipStream_t) = nullptr;
  ncclResult_t (*AllGather)(const void *, void *, size_t, ncclDataType_t,
                            ncclComm_t, hipStream_t) = nullptr;
  ncclResult_t (*ReduceScatter)(const void *, void *, size_t, ncclDataType_t,
                                ncclRedOp_t, ncclComm_t, hipStream_t) = nullptr;
  ncclResult_t (*AllToAll)(const void *, void *, size_t, ncclDataType_t,
                           ncclComm_t, hipStream_t) = nullptr;
  ncclResult_t (*Broadcast)(const void *, void *, size_t, ncclDataType_t, int,
                            ncclComm_t, hipStream_t) = nullptr;
  ncclResult_t (*Reduce)(const void *, void *, size_t, ncclDataType_t,
                         ncclRedOp_t, int, ncclComm_t, hipStream_t) = nullptr;
  ncclResult_t (*Send)(const void *, size_t, ncclDataType_t, int, ncclComm_t,
                       hipStream_t) = nullptr;
  ncclResult_t (*Recv)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                       hipStream_t) = nullptr;
};

static bool isFile(const std::string &p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool isDir(const std::string &p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string dirName(const std::string &p) {
  size_t pos = p.find_last_of('/');
  if (pos == std::string::npos)
    return ".";
  if (pos == 0)
    return "/";
  return p.substr(0, pos);
}

static std::string realPath(const std::string &p) {
  char buf[PATH_MAX];
  if (realpath(p.c_str(), buf))
    return buf;
  return p;
}

// Default location of prebuilt libraries: $RCCL_LIBS_DIR, else the first
// "rccl-libs" directory found walking up from the executable.
static std::string defaultLibsDir() {
  if (const char *e = getenv("RCCL_LIBS_DIR"))
    return e;
  char buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0)
    return "";
  buf[n] = '\0';
  std::string dir = dirName(buf);
  for (int i = 0; i < 5; i++) {
    if (isDir(dir + "/rccl-libs"))
      return dir + "/rccl-libs";
    if (dir == "/")
      break;
    dir = dirName(dir);
  }
  return "";
}

static std::string libInDir(const std::string &dir) {
  for (const char *rel :
       {"/lib/librccl.so", "/lib64/librccl.so", "/librccl.so"})
    if (isFile(dir + rel))
      return dir + rel;
  return "";
}

// A library spec is a path to librccl.so, a directory containing it (an install
// prefix or a build directory), or a commit-hash prefix / name of a directory
// under the libs dir.
static std::string resolveLib(const std::string &spec,
                              const std::string &libsDir) {
  if (isFile(spec))
    return realPath(spec);
  if (isDir(spec)) {
    std::string p = libInDir(spec);
    if (p.empty())
      die("no librccl.so found in directory '%s'", spec.c_str());
    return realPath(p);
  }
  if (libsDir.empty())
    die("'%s' is not a file or directory and no rccl-libs directory was found "
        "(use --libs_dir or RCCL_LIBS_DIR)",
        spec.c_str());
  std::vector<std::string> matches;
  if (DIR *d = opendir(libsDir.c_str())) {
    while (struct dirent *ent = readdir(d)) {
      std::string name = ent->d_name;
      if (name.empty() || name[0] == '.')
        continue;
      if (name.compare(0, spec.size(), spec) == 0 &&
          isDir(libsDir + "/" + name))
        matches.push_back(name);
    }
    closedir(d);
  }
  if (matches.empty())
    die("no prebuilt library matching '%s' in %s (build it with: rccl-ab build "
        "%s)",
        spec.c_str(), libsDir.c_str(), spec.c_str());
  if (matches.size() > 1)
    die("'%s' is ambiguous in %s", spec.c_str(), libsDir.c_str());
  std::string p = libInDir(libsDir + "/" + matches[0]);
  if (p.empty())
    die("no librccl.so found under %s/%s", libsDir.c_str(), matches[0].c_str());
  return realPath(p);
}

// Pick up commit/subject recorded by the rccl-ab build driver, if present.
static void readBuildInfo(RcclLib &lib) {
  std::string libDir = dirName(lib.path);
  for (const std::string &f :
       {libDir + "/../BUILD_INFO", libDir + "/BUILD_INFO"}) {
    FILE *fp = fopen(f.c_str(), "r");
    if (!fp)
      continue;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
        s.pop_back();
      if (s.compare(0, 7, "commit=") == 0)
        lib.commit = s.substr(7);
      if (s.compare(0, 8, "subject=") == 0)
        lib.subject = s.substr(8);
    }
    fclose(fp);
    return;
  }
}

template <typename F>
static void loadSym(RcclLib &lib, F &fn, const char *name) {
  fn = reinterpret_cast<F>(dlsym(lib.handle, name));
  if (!fn)
    die("symbol %s not found in %s", name, lib.path.c_str());
}

static void loadLib(RcclLib &lib, bool deepbind) {
  int flags = RTLD_NOW | RTLD_LOCAL;
  if (deepbind)
    flags |= RTLD_DEEPBIND;
  lib.handle = dlopen(lib.path.c_str(), flags);
  if (!lib.handle)
    die("dlopen(%s) failed: %s", lib.path.c_str(), dlerror());
  loadSym(lib, lib.GetVersion, "ncclGetVersion");
  loadSym(lib, lib.GetUniqueId, "ncclGetUniqueId");
  loadSym(lib, lib.CommInitRank, "ncclCommInitRank");
  loadSym(lib, lib.CommDestroy, "ncclCommDestroy");
  loadSym(lib, lib.GetErrorString, "ncclGetErrorString");
  loadSym(lib, lib.GroupStart, "ncclGroupStart");
  loadSym(lib, lib.GroupEnd, "ncclGroupEnd");
  loadSym(lib, lib.AllReduce, "ncclAllReduce");
  loadSym(lib, lib.AllGather, "ncclAllGather");
  loadSym(lib, lib.ReduceScatter, "ncclReduceScatter");
  loadSym(lib, lib.AllToAll, "ncclAllToAll");
  loadSym(lib, lib.Broadcast, "ncclBroadcast");
  loadSym(lib, lib.Reduce, "ncclReduce");
  loadSym(lib, lib.Send, "ncclSend");
  loadSym(lib, lib.Recv, "ncclRecv");
  NCCLCHECK(lib, lib.GetVersion(&lib.version));
  readBuildInfo(lib);
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

enum Coll {
  AllReduce,
  AllGather,
  ReduceScatter,
  AllToAll,
  Broadcast,
  Reduce,
  SendRecv
};
static const char *kCollNames[] = {"all_reduce", "all_gather", "reduce_scatter",
                                   "alltoall",   "broadcast",  "reduce",
                                   "sendrecv"};

struct TypeInfo {
  const char *name;
  int nccl;
  size_t size;
};
static const TypeInfo kTypes[] = {
    {"int8", 0, 1},     {"uint8", 1, 1},     {"int32", 2, 4},
    {"uint32", 3, 4},   {"int64", 4, 8},     {"uint64", 5, 8},
    {"half", 6, 2},     {"float", 7, 4},     {"double", 8, 8},
    {"bfloat16", 9, 2}, {"fp8_e4m3", 10, 1}, {"fp8_e5m2", 11, 1}};
enum {
  tInt8,
  tUint8,
  tInt32,
  tUint32,
  tInt64,
  tUint64,
  tHalf,
  tFloat,
  tDouble,
  tBf16,
  tF8e4m3,
  tF8e5m2
};

static const char *kOpNames[] = {"sum", "prod", "max", "min", "avg"};
enum { opSum, opProd, opMax, opMin, opAvg };

struct Options {
  std::string libA, libB, libsDir;
  Coll coll = AllReduce;
  int type = tFloat;
  int op = opSum;
  int root = 0;
  size_t minBytes = 32 * 1024 * 1024;
  size_t maxBytes = 32 * 1024 * 1024;
  size_t stepBytes = 1024 * 1024;
  double stepFactor = 1.0;
  int ngpus = 1;
  int iters = 20;
  int warmup = 5;
  int batchWarmup = 1;
  int rounds = 10;
  int discardRounds = 1;
  bool abba = true;
  bool inplace = false;
  bool check = false;
  bool deepbind = false;
  std::string csv, rawCsv;
};

static size_t parseSize(const char *s) {
  char *end = nullptr;
  double v = strtod(s, &end);
  if (end == s || v < 0)
    die("invalid size '%s'", s);
  switch (*end) {
  case 'k':
  case 'K':
    v *= 1024.0;
    end++;
    break;
  case 'm':
  case 'M':
    v *= 1024.0 * 1024;
    end++;
    break;
  case 'g':
  case 'G':
    v *= 1024.0 * 1024 * 1024;
    end++;
    break;
  case 't':
  case 'T':
    v *= 1024.0 * 1024 * 1024 * 1024;
    end++;
    break;
  default:
    break;
  }
  if (*end == 'B' || *end == 'b')
    end++;
  if (*end != '\0')
    die("invalid size '%s'", s);
  return (size_t)v;
}

static int parseInt(const char *s, const char *what, int minVal) {
  char *end = nullptr;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < minVal)
    die("invalid value '%s' for %s", s, what);
  return (int)v;
}

static void usage(const char *prog) {
  print0(
      "Usage: %s -A <lib> -B <lib> [options]\n"
      "\n"
      "Interleaved in-process A/B comparison of two RCCL libraries. <lib> is a "
      "path to\n"
      "librccl.so, a directory containing it, or a commit-hash prefix of a "
      "library\n"
      "prebuilt under the rccl-libs directory (see tools/rccl-ab/rccl-ab "
      "build).\n"
      "Run under mpirun with one process per GPU, or use -g for several GPUs "
      "per process.\n"
      "\n"
      "  -A, --lib_a <lib>          baseline library\n"
      "  -B, --lib_b <lib>          library compared against the baseline\n"
      "  -L, --libs_dir <dir>       prebuilt library directory (default: "
      "$RCCL_LIBS_DIR or the\n"
      "                             rccl-libs directory next to rccl-tests)\n"
      "  -C, --collective <name>    "
      "all_reduce|all_gather|reduce_scatter|alltoall|broadcast|\n"
      "                             reduce|sendrecv (default: all_reduce)\n"
      "  -b, --minbytes <size>      minimum size, K/M/G suffixes (default: "
      "32M)\n"
      "  -e, --maxbytes <size>      maximum size (default: 32M)\n"
      "  -i, --stepbytes <size>     additive step between sizes (default: 1M)\n"
      "  -f, --stepfactor <n>       multiplicative step between sizes "
      "(overrides -i)\n"
      "  -d, --datatype <type>      "
      "int8|uint8|int32|uint32|int64|uint64|half|float|double|\n"
      "                             bfloat16|fp8_e4m3|fp8_e5m2 (default: "
      "float)\n"
      "  -o, --op <op>              sum|prod|max|min|avg (default: sum)\n"
      "  -r, --root <rank>          root for broadcast/reduce (default: 0)\n"
      "  -g, --ngpus <n>            GPUs per process (default: 1)\n"
      "  -n, --iters <n>            iterations per timed batch (default: 20)\n"
      "  -R, --rounds <n>           interleaved A/B batch pairs per size "
      "(default: 10)\n"
      "  -w, --warmup_iters <n>     warmup iterations per library at each size "
      "(default: 5)\n"
      "  -W, --batch_warmup <n>     untimed iterations before every timed "
      "batch (default: 1)\n"
      "      --discard_rounds <n>   untimed A/B rounds run at each size before "
      "measuring\n"
      "                             (default: 1)\n"
      "      --order <abba|ab>      batch order within rounds (default: abba)\n"
      "      --inplace              run in-place (not for alltoall/sendrecv)\n"
      "  -c, --check <0|1>          verify results of both libraries at every "
      "size (default: 0)\n"
      "  -O, --csv <file>           write per-size summary as CSV\n"
      "      --raw_csv <file>       write every per-round batch time as CSV\n"
      "      --deepbind             dlopen with RTLD_DEEPBIND as well as "
      "RTLD_LOCAL\n"
      "  -h, --help                 show this help\n"
      "\n"
      "Speedup is time(A)/time(B), so values above 1 mean B is faster. It is "
      "the median of\n"
      "the paired per-round ratios, with a distribution-free 95%% confidence "
      "interval for\n"
      "that median (needs -R >= 6); '*' marks sizes whose interval excludes 1. "
      "Reported\n"
      "times are medians over rounds of the rank-averaged per-iteration "
      "time.\n",
      prog);
}

static Options parseArgs(int argc, char **argv) {
  Options o;
  enum { optOrder = 1000, optInplace, optRawCsv, optDeepbind, optDiscard };
  static struct option longOpts[] = {
      {"lib_a", required_argument, 0, 'A'},
      {"lib_b", required_argument, 0, 'B'},
      {"libs_dir", required_argument, 0, 'L'},
      {"collective", required_argument, 0, 'C'},
      {"minbytes", required_argument, 0, 'b'},
      {"maxbytes", required_argument, 0, 'e'},
      {"stepbytes", required_argument, 0, 'i'},
      {"stepfactor", required_argument, 0, 'f'},
      {"datatype", required_argument, 0, 'd'},
      {"op", required_argument, 0, 'o'},
      {"root", required_argument, 0, 'r'},
      {"ngpus", required_argument, 0, 'g'},
      {"iters", required_argument, 0, 'n'},
      {"rounds", required_argument, 0, 'R'},
      {"warmup_iters", required_argument, 0, 'w'},
      {"batch_warmup", required_argument, 0, 'W'},
      {"discard_rounds", required_argument, 0, optDiscard},
      {"order", required_argument, 0, optOrder},
      {"inplace", no_argument, 0, optInplace},
      {"check", required_argument, 0, 'c'},
      {"csv", required_argument, 0, 'O'},
      {"raw_csv", required_argument, 0, optRawCsv},
      {"deepbind", no_argument, 0, optDeepbind},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  bool haveStep = false, haveFactor = false;
  int c;
  while ((c = getopt_long(argc, argv, "A:B:L:C:b:e:i:f:d:o:r:g:n:R:w:W:c:O:h",
                          longOpts, nullptr)) != -1) {
    switch (c) {
    case 'A':
      o.libA = optarg;
      break;
    case 'B':
      o.libB = optarg;
      break;
    case 'L':
      o.libsDir = optarg;
      break;
    case 'C': {
      int i = 0;
      for (; i < (int)(sizeof(kCollNames) / sizeof(kCollNames[0])); i++)
        if (strcmp(optarg, kCollNames[i]) == 0)
          break;
      if (i == (int)(sizeof(kCollNames) / sizeof(kCollNames[0])))
        die("unknown collective '%s'", optarg);
      o.coll = (Coll)i;
      break;
    }
    case 'b':
      o.minBytes = parseSize(optarg);
      break;
    case 'e':
      o.maxBytes = parseSize(optarg);
      break;
    case 'i':
      o.stepBytes = parseSize(optarg);
      haveStep = true;
      break;
    case 'f':
      o.stepFactor = strtod(optarg, nullptr);
      if (o.stepFactor <= 1.0)
        die("--stepfactor must be > 1");
      haveFactor = true;
      break;
    case 'd': {
      const char *aliases[][2] = {{"float32", "float"},  {"float16", "half"},
                                  {"float64", "double"}, {"bf16", "bfloat16"},
                                  {"int", "int32"},      {"char", "int8"}};
      const char *name = optarg;
      for (auto &a : aliases)
        if (strcmp(name, a[0]) == 0)
          name = a[1];
      int i = 0;
      for (; i < (int)(sizeof(kTypes) / sizeof(kTypes[0])); i++)
        if (strcmp(name, kTypes[i].name) == 0)
          break;
      if (i == (int)(sizeof(kTypes) / sizeof(kTypes[0])))
        die("unknown datatype '%s'", optarg);
      o.type = i;
      break;
    }
    case 'o': {
      int i = 0;
      for (; i < 5; i++)
        if (strcmp(optarg, kOpNames[i]) == 0)
          break;
      if (i == 5)
        die("unknown op '%s'", optarg);
      o.op = i;
      break;
    }
    case 'r':
      o.root = parseInt(optarg, "--root", 0);
      break;
    case 'g':
      o.ngpus = parseInt(optarg, "--ngpus", 1);
      break;
    case 'n':
      o.iters = parseInt(optarg, "--iters", 1);
      break;
    case 'R':
      o.rounds = parseInt(optarg, "--rounds", 1);
      break;
    case 'w':
      o.warmup = parseInt(optarg, "--warmup_iters", 0);
      break;
    case 'W':
      o.batchWarmup = parseInt(optarg, "--batch_warmup", 0);
      break;
    case optDiscard:
      o.discardRounds = parseInt(optarg, "--discard_rounds", 0);
      break;
    case optOrder:
      if (strcmp(optarg, "abba") == 0)
        o.abba = true;
      else if (strcmp(optarg, "ab") == 0)
        o.abba = false;
      else
        die("--order must be abba or ab");
      break;
    case optInplace:
      o.inplace = true;
      break;
    case 'c':
      o.check = parseInt(optarg, "--check", 0) != 0;
      break;
    case 'O':
      o.csv = optarg;
      break;
    case optRawCsv:
      o.rawCsv = optarg;
      break;
    case optDeepbind:
      o.deepbind = true;
      break;
    case 'h':
      usage(argv[0]);
#ifdef MPI_SUPPORT
      MPI_Finalize();
#endif
      exit(0);
    default:
      usage(argv[0]);
      die("invalid arguments");
    }
  }
  if (optind < argc)
    die("unexpected argument '%s'", argv[optind]);
  if (o.libA.empty() || o.libB.empty()) {
    usage(argv[0]);
    die("both -A and -B are required");
  }
  if (haveStep && haveFactor)
    die("use only one of --stepbytes and --stepfactor");
  if (!haveFactor && o.stepBytes == 0)
    die("--stepbytes must be > 0");
  if (o.minBytes > o.maxBytes)
    die("--minbytes must not exceed --maxbytes");
  if (o.inplace && (o.coll == AllToAll || o.coll == SendRecv))
    die("--inplace is not supported for %s", kCollNames[o.coll]);
  return o;
}

// ---------------------------------------------------------------------------
// Benchmark state
// ---------------------------------------------------------------------------

struct Shape {
  size_t size;  // reported bytes
  size_t count; // count argument passed to RCCL
  size_t sendElems;
  size_t recvElems;
  double busFactor;
};

static Shape shapeFor(Coll coll, size_t bytes, size_t typeSize, int nRanks) {
  Shape s;
  size_t base = bytes / typeSize;
  double n = nRanks;
  switch (coll) {
  case AllReduce:
    s.count = s.sendElems = s.recvElems = base;
    s.busFactor = 2.0 * (n - 1) / n;
    break;
  case AllGather:
    s.count = s.sendElems = base / nRanks;
    s.recvElems = s.count * nRanks;
    s.busFactor = (n - 1) / n;
    break;
  case ReduceScatter:
    s.count = s.recvElems = base / nRanks;
    s.sendElems = s.count * nRanks;
    s.busFactor = (n - 1) / n;
    break;
  case AllToAll:
    s.count = base / nRanks;
    s.sendElems = s.recvElems = s.count * nRanks;
    s.busFactor = (n - 1) / n;
    break;
  default: // Broadcast, Reduce, SendRecv
    s.count = s.sendElems = s.recvElems = base;
    s.busFactor = 1.0;
    break;
  }
  size_t elems = std::max(s.sendElems, s.recvElems);
  s.size = elems * typeSize;
  return s;
}

struct Bench {
  Options opt;
  RcclLib libs[2];
  int nRanks = 0;
  size_t typeSize = 0;
  std::vector<int> devs;
  std::vector<hipStream_t> streams;
  std::vector<char *> sendbufs, recvbufs;
  std::vector<ncclComm_t> comms[2];

  int rankOf(int gpu) const { return worldRank * opt.ngpus + gpu; }

  void ptrs(int gpu, const Shape &sh, const void **sendp, void **recvp) const {
    char *s = sendbufs[gpu];
    char *r = recvbufs[gpu];
    if (opt.inplace) {
      size_t off = (size_t)rankOf(gpu) * sh.count * typeSize;
      if (opt.coll == AllGather)
        s = r + off;
      else if (opt.coll == ReduceScatter)
        r = s + off;
      else
        s = r;
    }
    *sendp = s;
    *recvp = r;
  }

  void launch(int l, int gpu, const Shape &sh) {
    RcclLib &L = libs[l];
    const void *s;
    void *r;
    ptrs(gpu, sh, &s, &r);
    ncclComm_t comm = comms[l][gpu];
    hipStream_t st = streams[gpu];
    int dt = kTypes[opt.type].nccl;
    int rank = rankOf(gpu);
    switch (opt.coll) {
    case AllReduce:
      NCCLCHECK(L, L.AllReduce(s, r, sh.count, dt, opt.op, comm, st));
      break;
    case AllGather:
      NCCLCHECK(L, L.AllGather(s, r, sh.count, dt, comm, st));
      break;
    case ReduceScatter:
      NCCLCHECK(L, L.ReduceScatter(s, r, sh.count, dt, opt.op, comm, st));
      break;
    case AllToAll:
      NCCLCHECK(L, L.AllToAll(s, r, sh.count, dt, comm, st));
      break;
    case Broadcast:
      NCCLCHECK(L, L.Broadcast(s, r, sh.count, dt, opt.root, comm, st));
      break;
    case Reduce:
      NCCLCHECK(L, L.Reduce(s, r, sh.count, dt, opt.op, opt.root, comm, st));
      break;
    case SendRecv:
      NCCLCHECK(L, L.GroupStart());
      NCCLCHECK(L, L.Send(s, sh.count, dt, (rank + 1) % nRanks, comm, st));
      NCCLCHECK(
          L, L.Recv(r, sh.count, dt, (rank - 1 + nRanks) % nRanks, comm, st));
      NCCLCHECK(L, L.GroupEnd());
      break;
    }
  }

  void runIters(int l, const Shape &sh, int iters) {
    RcclLib &L = libs[l];
    for (int it = 0; it < iters; it++) {
      if (opt.ngpus > 1)
        NCCLCHECK(L, L.GroupStart());
      for (int g = 0; g < opt.ngpus; g++)
        launch(l, g, sh);
      if (opt.ngpus > 1)
        NCCLCHECK(L, L.GroupEnd());
    }
  }

  void syncAll() {
    for (hipStream_t st : streams)
      HIPCHECK(hipStreamSynchronize(st));
  }

  void barrier() {
#ifdef MPI_SUPPORT
    if (worldSize > 1)
      MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  // Average per-iteration time in microseconds of one timed batch.
  double timedBatch(int l, const Shape &sh) {
    if (opt.batchWarmup > 0) {
      runIters(l, sh, opt.batchWarmup);
      syncAll();
    }
    barrier();
    auto t0 = std::chrono::steady_clock::now();
    runIters(l, sh, opt.iters);
    syncAll();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() /
           opt.iters;
  }

  long long checkLib(int l, const Shape &sh);
};

// ---------------------------------------------------------------------------
// Result verification
// ---------------------------------------------------------------------------

static uint16_t floatToHalf(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff)
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp >= 0x1f)
    return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10)
      return sign;
    mant |= 0x800000;
    uint32_t shift = 14 - exp;
    uint32_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1)))
      h++;
    return sign | h;
  }
  uint32_t h = sign | (exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1)))
    h++;
  return h;
}

static float halfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      x = sign | (exp << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 0x1f) {
    x = sign | 0x7f800000 | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}

static uint16_t floatToBf16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  if ((x & 0x7fffffff) > 0x7f800000)
    return (x >> 16) | 0x40;
  return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16);
}

static float bf16ToFloat(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  memcpy(&f, &x, 4);
  return f;
}

static void encodeVal(int type, double v, void *buf, size_t i) {
  switch (type) {
  case tInt8:
    ((int8_t *)buf)[i] = (int8_t)(int64_t)v;
    break;
  case tUint8:
    ((uint8_t *)buf)[i] = (uint8_t)(int64_t)v;
    break;
  case tInt32:
    ((int32_t *)buf)[i] = (int32_t)(int64_t)v;
    break;
  case tUint32:
    ((uint32_t *)buf)[i] = (uint32_t)(int64_t)v;
    break;
  case tInt64:
    ((int64_t *)buf)[i] = (int64_t)v;
    break;
  case tUint64:
    ((uint64_t *)buf)[i] = (uint64_t)(int64_t)v;
    break;
  case tHalf:
    ((uint16_t *)buf)[i] = floatToHalf((float)v);
    break;
  case tFloat:
    ((float *)buf)[i] = (float)v;
    break;
  case tDouble:
    ((double *)buf)[i] = v;
    break;
  case tBf16:
    ((uint16_t *)buf)[i] = floatToBf16((float)v);
    break;
  default:
    ((uint8_t *)buf)[i] = 0;
    break;
  }
}

static double decodeVal(int type, const void *buf, size_t i) {
  switch (type) {
  case tInt8:
    return ((const int8_t *)buf)[i];
  case tUint8:
    return ((const uint8_t *)buf)[i];
  case tInt32:
    return ((const int32_t *)buf)[i];
  case tUint32:
    return ((const uint32_t *)buf)[i];
  case tInt64:
    return (double)((const int64_t *)buf)[i];
  case tUint64:
    return (double)((const uint64_t *)buf)[i];
  case tHalf:
    return halfToFloat(((const uint16_t *)buf)[i]);
  case tFloat:
    return ((const float *)buf)[i];
  case tDouble:
    return ((const double *)buf)[i];
  case tBf16:
    return bf16ToFloat(((const uint16_t *)buf)[i]);
  default:
    return 0;
  }
}

static bool checkable(const Options &o) {
  if (o.type == tF8e4m3 || o.type == tF8e5m2)
    return false;
  bool reduces =
      o.coll == AllReduce || o.coll == ReduceScatter || o.coll == Reduce;
  return !reduces || o.op == opSum || o.op == opMax || o.op == opMin;
}

// Input element k of global rank p: small non-negative integers so sums of a
// modest number of ranks are exact in every checked type.
static inline double inputVal(int p, size_t k) {
  return (double)(((size_t)p * 7 + k * 3) % 11);
}

// Returns the number of wrong elements across this process's GPUs.
long long Bench::checkLib(int l, const Shape &sh) {
  const int type = opt.type;
  double tol = 0;
  if (type == tHalf)
    tol = 5e-3;
  else if (type == tBf16)
    tol = 3e-2;
  else if (type == tFloat)
    tol = 1e-5;
  else if (type == tDouble)
    tol = 1e-12;

  std::vector<std::vector<char>> host(opt.ngpus);
  for (int g = 0; g < opt.ngpus; g++) {
    const void *s;
    void *r;
    ptrs(g, sh, &s, &r);
    HIPCHECK(hipSetDevice(devs[g]));
    HIPCHECK(hipMemset(r, 0xff, sh.recvElems * typeSize));
    host[g].resize(std::max(sh.sendElems, sh.recvElems) * typeSize);
    int rank = rankOf(g);
    for (size_t k = 0; k < sh.sendElems; k++)
      encodeVal(type, inputVal(rank, k), host[g].data(), k);
    HIPCHECK(hipMemcpy((void *)s, host[g].data(), sh.sendElems * typeSize,
                       hipMemcpyHostToDevice));
  }
  HIPCHECK(hipDeviceSynchronize());
  runIters(l, sh, 1);
  syncAll();

  long long wrong = 0;
  for (int g = 0; g < opt.ngpus; g++) {
    const void *s;
    void *r;
    ptrs(g, sh, &s, &r);
    int rank = rankOf(g);
    if (opt.coll == Reduce && rank != opt.root)
      continue;
    HIPCHECK(hipMemcpy(host[g].data(), r, sh.recvElems * typeSize,
                       hipMemcpyDeviceToHost));
    std::vector<char> enc(8);
    for (size_t j = 0; j < sh.recvElems; j++) {
      double exp = 0;
      auto reduceOver = [&](size_t k) {
        double acc = inputVal(0, k);
        for (int p = 1; p < nRanks; p++) {
          double v = inputVal(p, k);
          if (opt.op == opSum)
            acc += v;
          else if (opt.op == opMax)
            acc = std::max(acc, v);
          else
            acc = std::min(acc, v);
        }
        return acc;
      };
      switch (opt.coll) {
      case AllReduce:
      case Reduce:
        exp = reduceOver(j);
        break;
      case ReduceScatter:
        exp = reduceOver((size_t)rank * sh.count + j);
        break;
      case AllGather:
        exp = inputVal((int)(j / sh.count), j % sh.count);
        break;
      case AllToAll:
        exp = inputVal((int)(j / sh.count),
                       (size_t)rank * sh.count + j % sh.count);
        break;
      case Broadcast:
        exp = inputVal(opt.root, j);
        break;
      case SendRecv:
        exp = inputVal((rank - 1 + nRanks) % nRanks, j);
        break;
      }
      // Round-trip the expectation through the datatype so integer wraparound
      // and floating-point rounding match what the device produced.
      encodeVal(type, exp, enc.data(), 0);
      double want = decodeVal(type, enc.data(), 0);
      double got = decodeVal(type, host[g].data(), j);
      bool ok =
          (tol == 0)
              ? (got == want)
              : (std::fabs(got - want) <= tol * std::max(1.0, std::fabs(want)));
      if (!ok)
        wrong++;
    }
  }
  return wrong;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void meanStd(const std::vector<double> &v, double *mean, double *sd) {
  double m = 0;
  for (double x : v)
    m += x;
  m /= v.size();
  double ss = 0;
  for (double x : v)
    ss += (x - m) * (x - m);
  *mean = m;
  *sd = v.size() > 1 ? std::sqrt(ss / (v.size() - 1)) : 0.0;
}

// Distribution-free confidence interval (>= 95% coverage) for the median of v:
// [x_(k), x_(n-k+1)] with the largest k such that P(Binomial(n, 1/2) < k)
// <= 2.5%. Requires n >= 6.
static bool medianCI95(std::vector<double> v, double *lo, double *hi) {
  int n = (int)v.size();
  std::sort(v.begin(), v.end());
  double cdf = 0, pmf = std::pow(0.5, n);
  int k = 0;
  for (int i = 0; i < n; i++) {
    cdf += pmf; // P(B <= i)
    if (cdf > 0.025)
      break;
    k = i + 1;
    pmf = pmf * (n - i) / (i + 1);
  }
  if (k == 0)
    return false;
  *lo = v[k - 1];
  *hi = v[n - k];
  return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static std::string fmtVersion(int v) {
  char buf[64];
  if (v >= 10000)
    snprintf(buf, sizeof(buf), "%d.%d.%d", v / 10000, (v % 10000) / 100,
             v % 100);
  else
    snprintf(buf, sizeof(buf), "%d.%d.%d", v / 1000, (v % 1000) / 100, v % 100);
  return buf;
}

int main(int argc, char **argv) {
#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                        MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
  }
#endif
  setvbuf(stdout, nullptr, _IOLBF, 0);

  Bench b;
  b.opt = parseArgs(argc, argv);
  Options &opt = b.opt;
  b.nRanks = worldSize * opt.ngpus;
  b.typeSize = kTypes[opt.type].size;
  if ((opt.coll == Broadcast || opt.coll == Reduce) && opt.root >= b.nRanks)
    die("--root %d out of range for %d ranks", opt.root, b.nRanks);

  std::string libsDir = opt.libsDir.empty() ? defaultLibsDir() : opt.libsDir;
  b.libs[0].label = "A";
  b.libs[0].spec = opt.libA;
  b.libs[1].label = "B";
  b.libs[1].spec = opt.libB;
  for (RcclLib &L : b.libs) {
    L.path = resolveLib(L.spec, libsDir);
    loadLib(L, opt.deepbind);
  }
  bool sameLib = b.libs[0].handle == b.libs[1].handle;

  int nDev = 0;
  HIPCHECK(hipGetDeviceCount(&nDev));
  if (nDev == 0)
    die("no GPUs visible");
  b.devs.resize(opt.ngpus);
  b.streams.resize(opt.ngpus);
  b.sendbufs.resize(opt.ngpus);
  b.recvbufs.resize(opt.ngpus);
  size_t allocBytes = std::max(opt.maxBytes, b.typeSize);
  allocBytes = (allocBytes + 4095) & ~(size_t)4095;
  for (int g = 0; g < opt.ngpus; g++) {
    b.devs[g] = (localRank * opt.ngpus + g) % nDev;
    HIPCHECK(hipSetDevice(b.devs[g]));
    HIPCHECK(hipStreamCreateWithFlags(&b.streams[g], hipStreamNonBlocking));
    HIPCHECK(hipMalloc((void **)&b.sendbufs[g], allocBytes));
    HIPCHECK(hipMalloc((void **)&b.recvbufs[g], allocBytes));
    HIPCHECK(hipMemset(b.sendbufs[g], 0, allocBytes));
    HIPCHECK(hipMemset(b.recvbufs[g], 0, allocBytes));
  }
  HIPCHECK(hipDeviceSynchronize());

  print0("# rccl_ab_perf: interleaved in-process A/B comparison\n");
  for (RcclLib &L : b.libs) {
    print0("#  %s: %s\n", L.label.c_str(), L.path.c_str());
    print0("#     version %s", fmtVersion(L.version).c_str());
    if (!L.commit.empty())
      print0("  commit %s", L.commit.c_str());
    if (!L.subject.empty())
      print0("  \"%s\"", L.subject.c_str());
    print0("\n");
  }
  if (sameLib)
    print0("#  note: A and B are the same file, so both communicator sets "
           "share one library "
           "instance (A/A noise-floor run)\n");
  print0("#  collective %s  datatype %s  op %s  root %d  inplace %d\n",
         kCollNames[opt.coll], kTypes[opt.type].name, kOpNames[opt.op],
         opt.root, (int)opt.inplace);
  print0("#  nranks %d (%d processes x %d GPUs)  iters/batch %d  rounds %d "
         "(%s, %d discarded)  "
         "warmup %d  batch warmup %d  check %d\n",
         b.nRanks, worldSize, opt.ngpus, opt.iters, opt.rounds,
         opt.abba ? "abba" : "ab", opt.discardRounds, opt.warmup,
         opt.batchWarmup, (int)opt.check);

  for (int l = 0; l < 2; l++) {
    RcclLib &L = b.libs[l];
    ncclUniqueId id;
    if (worldRank == 0)
      NCCLCHECK(L, L.GetUniqueId(&id));
#ifdef MPI_SUPPORT
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    b.comms[l].resize(opt.ngpus);
    NCCLCHECK(L, L.GroupStart());
    for (int g = 0; g < opt.ngpus; g++) {
      HIPCHECK(hipSetDevice(b.devs[g]));
      NCCLCHECK(L, L.CommInitRank(&b.comms[l][g], b.nRanks, id, b.rankOf(g)));
    }
    NCCLCHECK(L, L.GroupEnd());
  }
  print0("#  communicators initialized for both libraries\n#\n");

  bool doCheck = opt.check && checkable(opt);
  if (opt.check && !doCheck)
    print0("#  note: --check is not supported for datatype %s / op %s; "
           "skipping verification\n",
           kTypes[opt.type].name, kOpNames[opt.op]);

  FILE *csv = nullptr;
  FILE *raw = nullptr;
  if (worldRank == 0 && !opt.csv.empty()) {
    csv = fopen(opt.csv.c_str(), "w");
    if (!csv)
      die("cannot open %s", opt.csv.c_str());
    fprintf(csv, "collective,size_bytes,count,type,redop,root,nranks,a_time_us,"
                 "a_algbw_gbs,"
                 "a_busbw_gbs,a_cv_pct,b_time_us,b_algbw_gbs,b_busbw_gbs,b_cv_"
                 "pct,speedup_b_over_a,"
                 "ci95_lo,ci95_hi,a_wrong,b_wrong\n");
  }
  if (worldRank == 0 && !opt.rawCsv.empty()) {
    raw = fopen(opt.rawCsv.c_str(), "w");
    if (!raw)
      die("cannot open %s", opt.rawCsv.c_str());
    fprintf(raw, "collective,size_bytes,round,lib,position,time_us\n");
  }

  print0("#%49s|%s|%s|\n", "", "------------------- A --------------------",
         "------------------- B --------------------");
  print0("# %12s %12s %8s %6s %5s | %10s %10s %10s %7s | %10s %10s %10s %7s | "
         "%8s %19s%s\n",
         "size", "count", "type", "redop", "root", "time(us)", "algbw", "busbw",
         "cv(%)", "time(us)", "algbw", "busbw", "cv(%)", "B/A", "95% CI",
         doCheck ? "  #wrong(A,B)" : "");
  print0("# %12s %12s %8s %6s %5s | %10s %10s %10s %7s | %10s %10s %10s %7s | "
         "%8s %19s\n",
         "(B)", "(elements)", "", "", "", "", "(GB/s)", "(GB/s)", "", "",
         "(GB/s)", "(GB/s)", "", "speedup", "");

  std::vector<size_t> sizes;
  for (size_t s = opt.minBytes; s <= opt.maxBytes;) {
    sizes.push_back(s);
    size_t next =
        opt.stepFactor > 1.0 ? (size_t)(s * opt.stepFactor) : s + opt.stepBytes;
    if (next <= s)
      next = s + 1;
    s = next;
  }

  double logSpeedupSum = 0;
  int nSpeedups = 0;
  const int R = opt.rounds;
  for (size_t bytes : sizes) {
    Shape sh = shapeFor(opt.coll, bytes, b.typeSize, b.nRanks);
    if (sh.count == 0)
      continue;

    for (int l = 0; l < 2; l++) {
      b.runIters(l, sh, opt.warmup);
      b.syncAll();
    }

    // times[l][r]: per-iteration time of library l in round r
    std::vector<double> times(2 * R);
    std::vector<int> firstLib(R);
    for (int r = -opt.discardRounds; r < R; r++) {
      int first = (opt.abba && (r & 1)) ? 1 : 0;
      if (r < 0) {
        b.timedBatch(first, sh);
        b.timedBatch(1 - first, sh);
        continue;
      }
      firstLib[r] = first;
      times[first * R + r] = b.timedBatch(first, sh);
      times[(1 - first) * R + r] = b.timedBatch(1 - first, sh);
    }
#ifdef MPI_SUPPORT
    MPI_Allreduce(MPI_IN_PLACE, times.data(), 2 * R, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    for (double &t : times)
      t /= worldSize;

    long long wrong[2] = {0, 0};
    if (doCheck) {
      for (int l = 0; l < 2; l++)
        wrong[l] = b.checkLib(l, sh);
#ifdef MPI_SUPPORT
      MPI_Allreduce(MPI_IN_PLACE, wrong, 2, MPI_LONG_LONG, MPI_SUM,
                    MPI_COMM_WORLD);
#endif
    }

    std::vector<double> tl[2] = {
        std::vector<double>(times.begin(), times.begin() + R),
        std::vector<double>(times.begin() + R, times.end())};
    double med[2], algbw[2], busbw[2], cv[2];
    for (int l = 0; l < 2; l++) {
      double mean, sd;
      med[l] = median(tl[l]);
      meanStd(tl[l], &mean, &sd);
      cv[l] = mean > 0 ? 100.0 * sd / mean : 0.0;
      algbw[l] = (double)sh.size / (med[l] * 1e-6) / 1e9;
      busbw[l] = algbw[l] * sh.busFactor;
    }
    std::vector<double> ratio(R);
    for (int r = 0; r < R; r++)
      ratio[r] = tl[0][r] / tl[1][r];
    double speedup = median(ratio);
    double ciLo = NAN, ciHi = NAN;
    bool haveCI = medianCI95(ratio, &ciLo, &ciHi);
    logSpeedupSum += std::log(speedup);
    nSpeedups++;

    bool signif = haveCI && (ciLo > 1.0 || ciHi < 1.0);
    char ci[64];
    if (haveCI)
      snprintf(ci, sizeof(ci), "[%.4f, %.4f]%s", ciLo, ciHi,
               signif ? "*" : " ");
    else
      snprintf(ci, sizeof(ci), "%s", "n/a (R<6) ");
    char wrongStr[64] = "";
    if (doCheck)
      snprintf(wrongStr, sizeof(wrongStr), "  %lld,%lld", wrong[0], wrong[1]);
    bool hasRoot = opt.coll == Broadcast || opt.coll == Reduce;
    bool hasOp = opt.coll == AllReduce || opt.coll == ReduceScatter ||
                 opt.coll == Reduce;
    char rootStr[16];
    snprintf(rootStr, sizeof(rootStr), "%d", hasRoot ? opt.root : -1);
    print0("  %12zu %12zu %8s %6s %5s | %10.2f %10.2f %10.2f %7.2f | %10.2f "
           "%10.2f %10.2f %7.2f | "
           "%8.4f %19s%s\n",
           sh.size, sh.count, kTypes[opt.type].name,
           hasOp ? kOpNames[opt.op] : "none", rootStr, med[0], algbw[0],
           busbw[0], cv[0], med[1], algbw[1], busbw[1], cv[1], speedup, ci,
           wrongStr);
    if (csv)
      fprintf(csv,
              "%s,%zu,%zu,%s,%s,%s,%d,%.3f,%.4f,%.4f,%.3f,%.3f,%.4f,%.4f,%.3f,%"
              ".5f,%.5f,%.5f,%lld,%lld\n",
              kCollNames[opt.coll], sh.size, sh.count, kTypes[opt.type].name,
              hasOp ? kOpNames[opt.op] : "none", rootStr, b.nRanks, med[0],
              algbw[0], busbw[0], cv[0], med[1], algbw[1], busbw[1], cv[1],
              speedup, ciLo, ciHi, doCheck ? wrong[0] : -1LL,
              doCheck ? wrong[1] : -1LL);
    if (raw)
      for (int r = 0; r < R; r++)
        for (int l = 0; l < 2; l++)
          fprintf(raw, "%s,%zu,%d,%s,%d,%.3f\n", kCollNames[opt.coll], sh.size,
                  r, b.libs[l].label.c_str(), firstLib[r] == l ? 0 : 1,
                  tl[l][r]);
  }

  if (nSpeedups > 0)
    print0("#\n# Geometric mean B/A speedup over %d sizes: %.4f\n", nSpeedups,
           std::exp(logSpeedupSum / nSpeedups));
  if (csv)
    fclose(csv);
  if (raw)
    fclose(raw);

  for (int l = 0; l < 2; l++)
    for (int g = 0; g < opt.ngpus; g++)
      NCCLCHECK(b.libs[l], b.libs[l].CommDestroy(b.comms[l][g]));
  for (int g = 0; g < opt.ngpus; g++) {
    HIPCHECK(hipSetDevice(b.devs[g]));
    HIPCHECK(hipFree(b.sendbufs[g]));
    HIPCHECK(hipFree(b.recvbufs[g]));
    HIPCHECK(hipStreamDestroy(b.streams[g]));
  }
#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif
  return 0;
}
