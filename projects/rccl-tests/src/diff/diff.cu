/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Drives a back-to-back comparison of two already-loaded librccl copies.
 * Both stay resident. --diff_interleave=size alternates libraries inside
 * each message size; --diff_interleave=sweep alternates full size sweeps.
 ************************************************************************/

#include "util.h"
#include "diff/diff_api.h"
#include "diff/diff_ctrl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern int iters;
extern int warmup_iters;
extern int unalign;
extern int deviceImpl;
extern int per_iter_timing;
extern int deviceTimingMode;

extern testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place);
extern testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t opIndex, int root, int in_place, int iter);
extern testResult_t completeColl(struct threadArgs* args);
extern void setupArgs(size_t size, ncclDataType_t type, struct threadArgs* args);

namespace {

int g_enabled = 0;
int g_reps = 3;
int g_interleave = kRcclDiffInterleaveSize;
int g_order = kRcclDiffOrderAB;
int g_stat = kRcclDiffStatMedian;
uint32_t g_seed = 1;
std::string g_jsonPath;
std::string g_csvPath;
thread_local int g_bypass = 0;
thread_local int g_capture = 0;
thread_local int g_captureLib = 0;

struct Sample {
  double timeUsec;
  double algBw;
  double busBw;
  int64_t wrong;
  bool valid;
};

struct Key {
  size_t bytes;
  size_t count;
  int root;
  std::string type;
  std::string op;
  bool operator<(const Key& o) const {
    if (bytes != o.bytes) return bytes < o.bytes;
    if (count != o.count) return count < o.count;
    if (root != o.root) return root < o.root;
    if (type != o.type) return type < o.type;
    return op < o.op;
  }
};

struct LibStat {
  int n;
  double timeStat, timeMean, timeStdev, timeMin, timeMax;
  double busStat, algStat;
  int64_t wrongMax;
};

struct Row {
  std::vector<Sample> sample[2][2];  // [place][lib], place 0 = out of place
  bool printed;
  Row() : printed(false) {}
};

std::map<Key, Row> g_rows;
std::vector<Key> g_rowOrder;
thread_local Key g_key;
thread_local bool g_haveKey = false;

bool g_showOop = true;
bool g_showIp = true;
bool g_headerDone = false;
int g_warnedCycles = 0;

double g_busSum[2] = {0, 0};
int g_busN = 0;
double g_logRatioSum = 0;
int g_ratioN = 0;

struct UnalignGuard {
  threadArgs* args;
  size_t bytes;
  bool on;
  UnalignGuard(threadArgs* a, ncclDataType_t type, bool enable)
      : args(a), bytes((size_t)unalign * wordSize(type)), on(enable && unalign) {
    if (!on) return;
    for (int i = 0; i < args->nGpus; ++i) {
      args->sendbuffs[i] = (char*)args->sendbuffs[i] + bytes;
      args->recvbuffs[i] = (char*)args->recvbuffs[i] + bytes;
    }
  }
  ~UnalignGuard() {
    if (!on) return;
    for (int i = 0; i < args->nGpus; ++i) {
      args->sendbuffs[i] = (char*)args->sendbuffs[i] - bytes;
      args->recvbuffs[i] = (char*)args->recvbuffs[i] - bytes;
    }
  }
};

void orderFor(int rep, int out[2]) {
  if (g_order == kRcclDiffOrderBA) {
    out[0] = 1;
    out[1] = 0;
    return;
  }
  if (g_order == kRcclDiffOrderRandom) {
    uint32_t x = g_seed ^ (uint32_t)(rep + 1) * 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    if (x & 1u) {
      out[0] = 1;
      out[1] = 0;
    } else {
      out[0] = 0;
      out[1] = 1;
    }
    return;
  }
  out[0] = 0;
  out[1] = 1;
}

const char* interleaveName() {
  return g_interleave == kRcclDiffInterleaveSweep ? "sweep" : "size";
}
const char* orderName() {
  if (g_order == kRcclDiffOrderBA) return "ba";
  if (g_order == kRcclDiffOrderRandom) return "random";
  return "ab";
}
const char* statName() {
  return g_stat == kRcclDiffStatMean ? "mean" : "median";
}

void noteKey(size_t bytes, size_t count, const char* type, const char* op, int root) {
  g_key.bytes = bytes;
  g_key.count = count;
  g_key.root = root;
  g_key.type = type ? type : "";
  g_key.op = op ? op : "";
  g_haveKey = true;
  if (!is_main_thread) return;
  if (!g_rows.count(g_key)) g_rowOrder.push_back(g_key);
  g_rows[g_key];
}

void record(bool outOfPlace, bool valid, double timeUsec, double algBw, double busBw, int64_t wrong) {
  if (!is_main_thread || !g_haveKey) return;
  Sample s;
  s.timeUsec = timeUsec;
  s.algBw = algBw;
  s.busBw = busBw;
  s.wrong = wrong;
  s.valid = valid;
  int place = outOfPlace ? 0 : 1;
  g_rows[g_key].sample[place][g_captureLib].push_back(s);
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  double idx = p * (v.size() - 1);
  size_t lo = (size_t)idx;
  size_t hi = std::min(lo + 1, v.size() - 1);
  double f = idx - lo;
  return v[lo] * (1 - f) + v[hi] * f;
}

LibStat summarize(const std::vector<Sample>& samples) {
  LibStat st;
  memset(&st, 0, sizeof(st));
  std::vector<double> times, buses, algs;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (!samples[i].valid) continue;
    times.push_back(samples[i].timeUsec);
    buses.push_back(samples[i].busBw);
    algs.push_back(samples[i].algBw);
    if (samples[i].wrong > st.wrongMax) st.wrongMax = samples[i].wrong;
  }
  st.n = (int)times.size();
  if (!st.n) return st;
  double sum = 0, sum2 = 0;
  st.timeMin = times[0];
  st.timeMax = times[0];
  for (int i = 0; i < st.n; ++i) {
    sum += times[i];
    if (times[i] < st.timeMin) st.timeMin = times[i];
    if (times[i] > st.timeMax) st.timeMax = times[i];
  }
  st.timeMean = sum / st.n;
  if (st.n > 1) {
    for (int i = 0; i < st.n; ++i) {
      double d = times[i] - st.timeMean;
      sum2 += d * d;
    }
    st.timeStdev = sqrt(sum2 / (st.n - 1));
  }
  double timeMed = percentile(times, 0.5);
  double busMed = percentile(buses, 0.5);
  double algMed = percentile(algs, 0.5);
  double busMean = 0, algMean = 0;
  for (int i = 0; i < st.n; ++i) {
    busMean += buses[i];
    algMean += algs[i];
  }
  busMean /= st.n;
  algMean /= st.n;
  if (g_stat == kRcclDiffStatMean) {
    st.timeStat = st.timeMean;
    st.busStat = busMean;
    st.algStat = algMean;
  } else {
    st.timeStat = timeMed;
    st.busStat = busMed;
    st.algStat = algMed;
  }
  return st;
}

void fmtNum(double v, int width, char* out) {
  snprintf(out, 32, "%*.2f", width, v);
  if ((int)strlen(out) > width) snprintf(out, 32, "%*.1e", width, v);
}

bool significant(const LibStat& a, const LibStat& b) {
  if (a.n < 2 || b.n < 2) return false;
  double seA = a.timeStdev / sqrt((double)a.n);
  double seB = b.timeStdev / sqrt((double)b.n);
  double diff = b.timeMean - a.timeMean;
  return fabs(diff) > 2.0 * sqrt(seA * seA + seB * seB);
}

void jsonEscape(FILE* f, const std::string& s) {
  fputc('"', f);
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '"' || c == '\\') fputc('\\', f);
    if ((unsigned char)c < 0x20) continue;
    fputc(c, f);
  }
  fputc('"', f);
}

void printRow(const Key& key, Row& row) {
  const char* places[2] = {"oop", "ip"};
  bool show[2] = {g_showOop, g_showIp};
  for (int place = 0; place < 2; ++place) {
    if (!show[place]) continue;
    LibStat a = summarize(row.sample[place][0]);
    LibStat b = summarize(row.sample[place][1]);
    char ta[32], tb[32], ba[32], bb[32];
    fmtNum(a.n ? a.timeStat : 0, 8, ta);
    fmtNum(b.n ? b.timeStat : 0, 8, tb);
    fmtNum(a.n ? a.busStat : 0, 8, ba);
    fmtNum(b.n ? b.busStat : 0, 8, bb);
    double sdA = (a.n && a.timeMean > 0) ? 100.0 * a.timeStdev / a.timeMean : 0;
    double sdB = (b.n && b.timeMean > 0) ? 100.0 * b.timeStdev / b.timeMean : 0;
    double ratioT = (a.n && b.n && a.timeStat != 0) ? b.timeStat / a.timeStat : 0;
    double ratioB = (a.n && b.n && a.busStat != 0) ? b.busStat / a.busStat : 0;
    bool star = a.n && b.n && significant(a, b);
    PRINT("%12zu  %12zu  %8s  %6s  %6d  %5s | %8s  %5.1f  %8s | %8s  %5.1f  %8s | %7.3f  %7.3f %s\n",
        key.bytes, key.count, key.type.c_str(), key.op.c_str(), key.root, places[place],
        a.n ? ta : "     n/a", sdA, a.n ? ba : "     n/a",
        b.n ? tb : "     n/a", sdB, b.n ? bb : "     n/a",
        (a.n && b.n) ? ratioT : 0, (a.n && b.n) ? ratioB : 0, star ? "*" : "");
    if (a.n) {
      g_busSum[0] += a.busStat;
    }
    if (b.n) {
      g_busSum[1] += b.busStat;
    }
    if (a.n && b.n) g_busN++;
    if (a.n && b.n && ratioB > 0) {
      g_logRatioSum += log(ratioB);
      g_ratioN++;
    }
    if (a.wrongMax || b.wrongMax) {
      PRINT("# WARN: wrong elements at %zu bytes %s: A %ld  B %ld\n",
          key.bytes, places[place], (long)a.wrongMax, (long)b.wrongMax);
    }
  }
  row.printed = true;
}

void printNewRows() {
  if (!is_main_thread) return;
  for (size_t i = 0; i < g_rowOrder.size(); ++i) {
    Row& row = g_rows[g_rowOrder[i]];
    if (!row.printed) printRow(g_rowOrder[i], row);
  }
  fflush(stdout);
}

testResult_t warmupBoth(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root) {
  for (int lib = 0; lib < 2; ++lib) {
    rcclDiffSetActive(lib);
    Barrier(args);
    for (size_t size = args->minbytes; size <= args->maxbytes; size *= 2) {
      setupArgs(size, type, args);
      for (int iter = 0; iter < warmup_iters; ++iter) {
        TESTCHECK(startColl(args, type, op, root, 0, iter));
      }
      TESTCHECK(completeColl(args));
    }
  }
  return testSuccess;
}

testResult_t runSize(struct threadArgs* args, ncclDataType_t type, const char* typeName,
    ncclRedOp_t op, const char* opName, int root) {
  UnalignGuard shift(args, type, true);
  TESTCHECK(warmupBoth(args, type, op, root));
  for (size_t size = args->minbytes; size <= args->maxbytes;
       size = (args->stepfactor > 1) ? size * args->stepfactor : size + args->stepbytes) {
    setupArgs(size, type, args);
    int places[2] = {0, 1};
    int nPlaces = 0;
    if (args->enable_out_of_place) places[nPlaces++] = 0;
    if (args->enable_in_place) places[nPlaces++] = 1;
    for (int p = 0; p < nPlaces; ++p) {
      int inPlace = places[p];
      for (int rep = 0; rep < g_reps; ++rep) {
        int seq[2];
        orderFor(rep, seq);
        for (int s = 0; s < 2; ++s) {
          rcclDiffSetActive(seq[s]);
          g_captureLib = seq[s];
          g_capture = 1;
          noteKey(std::max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type), typeName, opName, root);
          TESTCHECK(BenchTime(args, type, op, root, inPlace));
          g_capture = 0;
          Barrier(args);
        }
      }
    }
    printNewRows();
    Barrier(args);
  }
  return testSuccess;
}

testResult_t runSweep(struct threadArgs* args, ncclDataType_t type, const char* typeName,
    ncclRedOp_t op, const char* opName, int root) {
  int saved = rcclDiffGetRunCycles();
  rcclDiffSetRunCycles(1);
  if (saved != 1 && is_main_thread && !g_warnedCycles) {
    fprintf(stderr, "# rccl-diff: -N %d is not nested; repetitions come from --diff_reps (%d)\n", saved, g_reps);
    g_warnedCycles = 1;
  }
  for (int rep = 0; rep < g_reps; ++rep) {
    int seq[2];
    orderFor(rep, seq);
    for (int s = 0; s < 2; ++s) {
      rcclDiffSetActive(seq[s]);
      g_captureLib = seq[s];
      g_capture = 1;
      g_bypass = 1;
      testResult_t r = TimeTest(args, type, typeName, op, opName, root);
      g_bypass = 0;
      g_capture = 0;
      Barrier(args);
      if (r != testSuccess) {
        rcclDiffSetRunCycles(saved);
        return r;
      }
    }
  }
  rcclDiffSetRunCycles(saved);
  printNewRows();
  Barrier(args);
  return testSuccess;
}

void writeJson(FILE* f) {
  fprintf(f, "{\n  \"interleave\": \"%s\",\n  \"reps\": %d,\n  \"order\": \"%s\",\n  \"stat\": \"%s\",\n  \"iters\": %d,\n",
      interleaveName(), g_reps, orderName(), statName(), iters);
  fprintf(f, "  \"libs\": [\n");
  for (int i = 0; i < 2; ++i) {
    const char *label, *path;
    int ver;
    rcclDiffGetLibInfo(i, &label, &path, &ver);
    fprintf(f, "    {\"label\": \"%s\", \"version\": %d, \"path\": ", label, ver);
    jsonEscape(f, path ? path : "");
    fprintf(f, "}%s\n", i == 0 ? "," : "");
  }
  fprintf(f, "  ],\n  \"rows\": [\n");
  bool firstRow = true;
  for (size_t i = 0; i < g_rowOrder.size(); ++i) {
    const Key& key = g_rowOrder[i];
    Row& row = g_rows[key];
    if (!firstRow) fprintf(f, ",\n");
    firstRow = false;
    fprintf(f, "    {\"size\": %zu, \"count\": %zu, \"type\": \"%s\", \"redop\": \"%s\", \"root\": %d",
        key.bytes, key.count, key.type.c_str(), key.op.c_str(), key.root);
    const char* places[2] = {"out_of_place", "in_place"};
    for (int place = 0; place < 2; ++place) {
      LibStat a = summarize(row.sample[place][0]);
      LibStat b = summarize(row.sample[place][1]);
      fprintf(f, ",\n      \"%s\": {", places[place]);
      if (!a.n && !b.n) {
        fprintf(f, "}");
        continue;
      }
      auto one = [&](const char* name, const LibStat& st, bool comma) {
        fprintf(f, "%s\"%s\": {\"n\": %d, \"time_us_stat\": %.6g, \"time_us_mean\": %.6g, \"time_us_stdev\": %.6g, \"time_us_min\": %.6g, \"time_us_max\": %.6g, \"busbw_stat\": %.6g, \"algbw_stat\": %.6g, \"wrong_max\": %ld}",
            comma ? ", " : "", name, st.n, st.timeStat, st.timeMean, st.timeStdev, st.timeMin, st.timeMax, st.busStat, st.algStat, (long)st.wrongMax);
      };
      one("A", a, false);
      one("B", b, true);
      if (a.n && b.n && a.timeStat != 0 && a.busStat != 0) {
        fprintf(f, ", \"ratio_B_over_A\": {\"time\": %.6g, \"busbw\": %.6g}",
            b.timeStat / a.timeStat, b.busStat / a.busStat);
      }
      fprintf(f, "}");
    }
    fprintf(f, "}");
  }
  double geomean = g_ratioN ? exp(g_logRatioSum / g_ratioN) : 0;
  fprintf(f, "\n  ],\n  \"summary\": {\"rows\": %d, \"A_busbw_mean\": %.6g, \"B_busbw_mean\": %.6g, \"busbw_ratio_geomean\": %.6g}\n}\n",
      g_busN, g_busN ? g_busSum[0] / g_busN : 0, g_busN ? g_busSum[1] / g_busN : 0, geomean);
}

void writeCsv(FILE* f) {
  fprintf(f, "size,count,type,redop,root,place,lib,rep,time_us,algbw,busbw,wrong,valid\n");
  const char* places[2] = {"oop", "ip"};
  const char* libs[2] = {"A", "B"};
  for (size_t i = 0; i < g_rowOrder.size(); ++i) {
    const Key& key = g_rowOrder[i];
    Row& row = g_rows[key];
    for (int place = 0; place < 2; ++place) {
      for (int lib = 0; lib < 2; ++lib) {
        const std::vector<Sample>& v = row.sample[place][lib];
        for (size_t r = 0; r < v.size(); ++r) {
          fprintf(f, "%zu,%zu,%s,%s,%d,%s,%s,%zu,%.8g,%.8g,%.8g,%ld,%d\n",
              key.bytes, key.count, key.type.c_str(), key.op.c_str(), key.root,
              places[place], libs[lib], r, v[r].timeUsec, v[r].algBw, v[r].busBw,
              (long)v[r].wrong, v[r].valid ? 1 : 0);
        }
      }
    }
  }
}

}  // namespace

void rcclDiffConfigure(int reps, int interleave, int order, int stat, int seed,
    const char* jsonPath, const char* csvPath) {
  g_enabled = 1;
  if (reps > 0) g_reps = reps;
  g_interleave = interleave;
  g_order = order;
  g_stat = stat;
  g_seed = (uint32_t)seed;
  if (jsonPath) g_jsonPath = jsonPath;
  if (csvPath) g_csvPath = csvPath;
}

int rcclDiffShouldTakeover() {
  return g_enabled && !g_bypass;
}

testResult_t rcclDiffTimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName,
    ncclRedOp_t op, const char* opName, int root) {
  if (deviceImpl != 0 || per_iter_timing || deviceTimingMode) {
    if (is_main_thread) {
      fprintf(stderr, "rccl-diff: device implementation, per-iteration timing, and device timing are not supported yet\n");
    }
    return testInvalidUsage;
  }
  g_showOop = args->enable_out_of_place;
  g_showIp = args->enable_in_place;
  if (g_interleave == kRcclDiffInterleaveSweep) {
    return runSweep(args, type, typeName, op, opName, root);
  }
  return runSize(args, type, typeName, op, opName, root);
}

int rcclDiffOnPreamble(size_t nBytes, size_t nElem, const char* typeName, const char* opName, int root) {
  if (!g_capture) return 0;
  noteKey(nBytes, nElem, typeName, opName, root);
  return 1;
}

int rcclDiffOnBody(double timeUsec, double algBw, double busBw, bool reportErrors, int64_t wrongElts, bool outOfPlace) {
  if (!g_capture) return 0;
  record(outOfPlace, true, timeUsec, algBw, busBw, reportErrors ? wrongElts : 0);
  return 1;
}

int rcclDiffOnNullBody() {
  if (!g_capture) return 0;
  record(true, false, 0, 0, 0, 0);
  return 1;
}

int rcclDiffOnTerminator() {
  return g_capture ? 1 : 0;
}

int rcclDiffWriteHeader(bool, bool, bool enableOutOfPlace, bool enableInPlace, bool outputAlgo) {
  if (!g_enabled) return 0;
  g_showOop = enableOutOfPlace;
  g_showIp = enableInPlace;
  if (outputAlgo && is_main_thread) {
    fprintf(stderr, "# rccl-diff: algorithm/protocol columns are not supported yet and were ignored\n");
  }
  if (g_headerDone || !is_main_thread) return 1;
  g_headerDone = true;
  const char* la = "A";
  const char* lb = "B";
  const char *pa, *pb;
  int va, vb;
  rcclDiffGetLibInfo(0, &la, &pa, &va);
  rcclDiffGetLibInfo(1, &lb, &pb, &vb);
  PRINT("# rccl-diff  interleave=%s  reps=%d  order=%s  stat=%s  iters=%d  warmup=%d\n",
      interleaveName(), g_reps, orderName(), statName(), iters, warmup_iters);
  PRINT("# A %d.%d.%d  %s\n", va / 10000, (va / 100) % 100, va % 100, pa ? pa : "");
  PRINT("# B %d.%d.%d  %s\n", vb / 10000, (vb / 100) % 100, vb % 100, pb ? pb : "");
  PRINT("# %s marks a time difference larger than the combined noise\n", "*");
  PRINT("# %12s  %12s  %8s  %6s  %6s  %5s | %8s  %5s  %8s | %8s  %5s  %8s | %7s  %7s\n",
      "size", "count", "type", "redop", "root", "place",
      "A time", "sd%", "A busbw", "B time", "sd%", "B busbw", "B/A time", "B/A bw");
  return 1;
}

int rcclDiffWriteFooter(const int errors[], const double*, double, const char* programName) {
  if (!g_enabled) return 0;
  if (is_main_thread) {
    double geomean = g_ratioN ? exp(g_logRatioSum / g_ratioN) : 0;
    PRINT("#\n");
    PRINT("# rccl-diff summary (%s)\n", programName ? programName : "");
    PRINT("# %-22s : %.4g\n", "A busbw mean", g_busN ? g_busSum[0] / g_busN : 0);
    PRINT("# %-22s : %.4g\n", "B busbw mean", g_busN ? g_busSum[1] / g_busN : 0);
    PRINT("# %-22s : %.4f\n", "B/A busbw geomean", geomean);
    PRINT("# %-22s : %d %s\n", "Out of bounds values", errors ? errors[0] : 0,
        (errors && errors[0]) ? "FAILED" : "OK");
    PRINT("#\n");
    if (!g_jsonPath.empty()) {
      FILE* f = fopen(g_jsonPath.c_str(), "w");
      if (!f) {
        fprintf(stderr, "rccl-diff: cannot write %s\n", g_jsonPath.c_str());
      } else {
        writeJson(f);
        fclose(f);
        PRINT("# rccl-diff json: %s\n", g_jsonPath.c_str());
      }
    }
    if (!g_csvPath.empty()) {
      FILE* f = fopen(g_csvPath.c_str(), "w");
      if (!f) {
        fprintf(stderr, "rccl-diff: cannot write %s\n", g_csvPath.c_str());
      } else {
        writeCsv(f);
        fclose(f);
        PRINT("# rccl-diff csv: %s\n", g_csvPath.c_str());
      }
    }
  }
  return 1;
}
