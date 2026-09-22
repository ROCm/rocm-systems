/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information. */

#include <assert.h>
#include <time.h>

// Deterministic timestamps exercise the callbacks without sleeps or GPU hardware.
static unsigned long long nowUs = 1000;
static int testClockGettime(clockid_t, struct timespec* ts) {
  ts->tv_sec = nowUs / 1000000;
  ts->tv_nsec = (nowUs % 1000000) * 1000;
  return 0;
}
#define clock_gettime testClockGettime
#include "accl_profiler.cc"
#undef clock_gettime

static int warnings = 0;
static void logger(int level, unsigned long flags, const char*, int, const char*, ...) {
  // RCCL's default subsystem mask excludes PROFILE.
  if (level == 3 && (flags & 1)) warnings++;
}

static void* start(void* ctx, ncclProfilerEventDescr_v5_t d) {
  void* h = NULL;
  assert(acclPluginStartEvent(ctx, &h, &d) == ncclSuccess);
  return h;
}

static void* startColl(void* ctx) {
  ncclProfilerEventDescr_v5_t d = {};
  d.type = ncclProfileColl;
  d.coll.func = "AllReduce"; d.coll.algo = "Ring"; d.coll.proto = "Simple";
  d.coll.datatype = "ncclFloat32"; d.coll.count = 1024; d.coll.nChannels = 1;
  return start(ctx, d);
}

static void finishColl(void* ctx, void* coll) {
  ncclProfilerEventDescr_v5_t d = {};
  d.type = ncclProfileKernelCh; d.parentObj = coll; d.kernelCh.pTimer = 1000;
  void* ch = start(ctx, d);
  ncclProfilerEventStateArgs_v5_t args = {};
  args.kernelCh.pTimer = 21000;
  acclPluginRecordEventState(ch, ncclProfilerKernelChStop, &args);
  acclPluginStopEvent(ch);
  acclPluginStopEvent(coll);
}

int main(int argc, char** argv) {
  assert(argc >= 3);
  setenv("ACCL_PROFILER_OUTPUT_DIR", argv[1], 1);
  void* ctx = NULL; int mask = 0;
  acclPluginInit(&ctx, 1, &mask, "correctness", 1, 2, 0, logger);
  assert(ctx);
  const char* mode = argv[2];
  if (!strcmp(mode, "threshold") || !strcmp(mode, "percomm")) {
    void* second = NULL;
    if (!strcmp(mode, "percomm")) {
      unsetenv("ACCL_PROFILER_MIN_SIZE_BYTES");
      acclPluginInit(&second, 2, &mask, "second", 1, 2, 0, logger);
    }
    void* coll = startColl(ctx);
    printf("accepted=%d\n", coll != NULL);
    if (coll) finishColl(ctx, coll);
    if (second) {
      coll = startColl(second);
      printf("second_accepted=%d\n", coll != NULL);
      if (coll) finishColl(second, coll);
      acclPluginFinalize(second);
    }
  } else if (!strcmp(mode, "ops") || !strcmp(mode, "steps")) {
    bool ops = !strcmp(mode, "ops");
    int count = ops ? ACCL_PROXY_OP_POOL_SIZE : ACCL_PROXY_STEP_POOL_SIZE;
    void* handles[ACCL_PROXY_STEP_POOL_SIZE];
    ncclProfilerEventDescr_v5_t d = {};
    d.type = ops ? ncclProfileProxyOp : ncclProfileProxyStep;
    for (int i = 0; i < count; i++) { handles[i] = start(ctx, d); assert(handles[i]); }
    assert(start(ctx, d) == NULL);
    for (int i = 0; i < count; i++) acclPluginStopEvent(handles[i]);
  } else if (!strcmp(mode, "io") || !strcmp(mode, "prior-io")) {
    auto* c = (acclCommContext*)ctx;
    fclose(c->outputFile);
    c->outputFile = fopen("/dev/full", "w");
    assert(c->outputFile);
    if (!strcmp(mode, "prior-io")) {
      fputs("record", c->outputFile);
      assert(fflush(c->outputFile) != 0);
      // Retain the sticky error but allow the summary to reach a readable file.
      FILE* output = fopen(c->outputPath, "w");
      assert(output && dup2(fileno(output), fileno(c->outputFile)) >= 0);
      fclose(output);
    }
  } else if (!strcmp(mode, "timing") || !strcmp(mode, "send-only") || !strcmp(mode, "overflow")) {
    void* coll = startColl(ctx); assert(coll);
    bool overflow = !strcmp(mode, "overflow");
    int count = overflow ? ACCL_MAX_PROXY_OPS + 1 : (!strcmp(mode, "send-only") ? 2 : 3);
    for (int i = 0; i < count; i++) {
      bool send = i != 2;
      ncclProfilerEventDescr_v5_t d = {};
      d.type = ncclProfileProxyOp; d.parentObj = coll; d.proxyOp.isSend = send;
      void* op = start(ctx, d); assert(op);
      if (!overflow) {
        d = {}; d.type = ncclProfileProxyStep; d.parentObj = op;
        void* step = start(ctx, d); assert(step);
        nowUs += 100; // Before the first named state: must not enter any bucket.
        ncclProfilerEventState_v5_t states[] = {
          send ? ncclProfilerProxyStepSendGPUWait : ncclProfilerProxyStepRecvWait,
          send ? ncclProfilerProxyStepSendPeerWait_v4 : ncclProfilerProxyStepRecvFlushWait,
          send ? ncclProfilerProxyStepSendWait : ncclProfilerProxyStepRecvGPUWait};
        for (int j = 0; j < 3; j++) {
          acclPluginRecordEventState(step, states[j], NULL);
          nowUs += (send ? 10 : 40) + j * 10;
        }
        acclPluginStopEvent(step);
      }
      acclPluginStopEvent(op);
    }
    finishColl(ctx, coll);
  }
  acclPluginFinalize(ctx);
  printf("warnings=%d\n", warnings);
}
