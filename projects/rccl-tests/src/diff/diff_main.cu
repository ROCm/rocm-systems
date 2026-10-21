/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Entry point for *_diff binaries. Strips --diff_* options, loads the two
 * libraries, then calls the original rccl-tests main.
 ************************************************************************/

#include "diff/diff_ctrl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern int rcclTestsOrigMain(int argc, char** argv, char** envp);

static void usage(const char* argv0) {
  fprintf(stderr,
      "Usage: %s [rccl-tests options] --diff_libs A=<librccl.so>,B=<librccl.so> [diff options]\n"
      "\n"
      "Diff options:\n"
      "  --diff_libs A=PATH,B=PATH     Required. Two librccl.so binaries.\n"
      "  --diff_reps N                 Repetitions per library (default 3).\n"
      "  --diff_interleave size|sweep  size: alternate inside each message size (default).\n"
      "                                sweep: alternate full size sweeps. Both keep both\n"
      "                                libraries resident.\n"
      "  --diff_order ab|ba|random     Order within a repetition (default ab).\n"
      "  --diff_stat median|mean       Statistic used for the reported value and the ratio\n"
      "                                (default median). sd%% is always the sample stddev.\n"
      "  --diff_seed N                 Seed for --diff_order=random (default 1).\n"
      "  --diff_json PATH              Write a JSON report.\n"
      "  --diff_csv PATH               Write one CSV row per sample.\n"
      "  --diff_deepbind 0|1           dlopen with RTLD_DEEPBIND (default 0).\n"
      "\n"
      "Not supported yet: -D device implementation, -I per-iteration timing,\n"
      "-B device timing, -A algorithm columns, -J stock json, NCCL_COMM_ID.\n",
      argv0);
}

static int takeValue(int argc, char** argv, int* i, const char* flag, const char** out) {
  const char* a = argv[*i];
  size_t n = strlen(flag);
  if (strncmp(a, flag, n) == 0 && a[n] == '=') {
    *out = a + n + 1;
    return 1;
  }
  if (strcmp(a, flag) == 0) {
    if (*i + 1 >= argc) return -1;
    *i += 1;
    *out = argv[*i];
    return 1;
  }
  return 0;
}

static int rejected(int argc, char** argv, const char* longName, char shortName) {
  for (int i = 1; i < argc; ++i) {
    const char* v = nullptr;
    int r = 0;
    if (longName) r = takeValue(argc, argv, &i, longName, &v);
    if (r < 0) return 1;
    if (r == 1) return v && strcmp(v, "0") != 0;
    if (shortName && argv[i][0] == '-' && argv[i][1] == shortName && argv[i][2] == '\0') {
      if (i + 1 >= argc) return 1;
      return strcmp(argv[i + 1], "0") != 0;
    }
  }
  return 0;
}

int main(int argc, char** argv, char** envp) {
  const char* libs = nullptr;
  const char* json = nullptr;
  const char* csv = nullptr;
  int help = 0;
  int reps = 3;
  int interleave = kRcclDiffInterleaveSize;
  int order = kRcclDiffOrderAB;
  int stat = kRcclDiffStatMedian;
  int seed = 1;
  int deepbind = 0;

  std::vector<char*> kept;
  kept.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    const char* v = nullptr;
    int r;
    r = takeValue(argc, argv, &i, "--diff_libs", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) { libs = v; continue; }
    r = takeValue(argc, argv, &i, "--diff_reps", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) { reps = atoi(v); continue; }
    r = takeValue(argc, argv, &i, "--diff_interleave", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) {
      if (strcmp(v, "size") == 0) interleave = kRcclDiffInterleaveSize;
      else if (strcmp(v, "sweep") == 0) interleave = kRcclDiffInterleaveSweep;
      else { fprintf(stderr, "rccl-diff: --diff_interleave is size or sweep\n"); return 2; }
      continue;
    }
    r = takeValue(argc, argv, &i, "--diff_order", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) {
      if (strcmp(v, "ab") == 0) order = kRcclDiffOrderAB;
      else if (strcmp(v, "ba") == 0) order = kRcclDiffOrderBA;
      else if (strcmp(v, "random") == 0) order = kRcclDiffOrderRandom;
      else { fprintf(stderr, "rccl-diff: --diff_order is ab, ba, or random\n"); return 2; }
      continue;
    }
    r = takeValue(argc, argv, &i, "--diff_stat", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) {
      if (strcmp(v, "median") == 0) stat = kRcclDiffStatMedian;
      else if (strcmp(v, "mean") == 0) stat = kRcclDiffStatMean;
      else { fprintf(stderr, "rccl-diff: --diff_stat is median or mean\n"); return 2; }
      continue;
    }
    r = takeValue(argc, argv, &i, "--diff_seed", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) { seed = atoi(v); continue; }
    r = takeValue(argc, argv, &i, "--diff_json", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) { json = v; continue; }
    r = takeValue(argc, argv, &i, "--diff_csv", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) { csv = v; continue; }
    r = takeValue(argc, argv, &i, "--diff_deepbind", &v);
    if (r < 0) { usage(argv[0]); return 2; }
    if (r == 1) { deepbind = atoi(v); continue; }
    if (strncmp(argv[i], "--diff", 6) == 0) {
      fprintf(stderr, "rccl-diff: unknown option %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      help = 1;
      kept.push_back(argv[i]);
      continue;
    }
    kept.push_back(argv[i]);
  }

  if (help) usage(argv[0]);
  if (!libs) {
    if (!help) usage(argv[0]);
    return help ? 0 : 2;
  }
  if (reps < 1) {
    fprintf(stderr, "rccl-diff: --diff_reps must be >= 1\n");
    return 2;
  }
  if (rejected(argc, argv, "--device_implementation", 'D') ||
      rejected(argc, argv, "--per_iter_timing", 'I') ||
      rejected(argc, argv, "--device_timing", 'B') ||
      rejected(argc, argv, "--output_algo_proto_channels", 'A') ||
      rejected(argc, argv, "--output_file", 'J')) {
    fprintf(stderr, "rccl-diff: -D, -I, -B, -A, and -J are not supported yet\n");
    return 2;
  }
  if (getenv("NCCL_COMM_ID")) {
    fprintf(stderr, "rccl-diff: NCCL_COMM_ID is not supported (both libraries would share one listen address)\n");
    return 2;
  }

  char err[512];
  if (rcclDiffLoadLibraries(libs, deepbind, err, sizeof(err)) != 0) {
    fprintf(stderr, "rccl-diff: %s\n", err);
    return 2;
  }
  rcclDiffConfigure(reps, interleave, order, stat, seed, json, csv);

  kept.push_back(nullptr);
  return rcclTestsOrigMain((int)kept.size() - 1, kept.data(), envp);
}
