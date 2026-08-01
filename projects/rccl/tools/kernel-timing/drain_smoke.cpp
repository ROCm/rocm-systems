// End-to-end check of RCCL's kernel-timing drain API on a single rank.
//
// Runs a batch of AllReduces, drains the records, and reports whether every
// dispatch was captured, whether the timestamps are ordered and bracketed by
// the host clock, and what the durations look like. Run under rocprofv3
// --kernel-trace to compare against the profiler's own numbers.

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <vector>

typedef struct {
  uint64_t startNs, endNs, seq, commHash, count;
  uint32_t func, datatype, nChannels, nThreads;
  int32_t rank;
  uint32_t nColls;
} Record;

typedef ncclResult_t (*DrainFn)(ncclComm_t, Record*, int, int*, uint64_t*);

#define CK(x)                                                                            \
  do {                                                                                   \
    auto _e = (x);                                                                       \
    if (_e != 0) {                                                                       \
      printf("FAIL %s -> %d\n", #x, (int)_e);                                            \
      return 1;                                                                          \
    }                                                                                    \
  } while (0)

static uint64_t bootNs() {
  timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 200;
  size_t count = argc > 2 ? atol(argv[2]) : (1 << 20);

  DrainFn drain = (DrainFn)dlsym(RTLD_DEFAULT, "ncclKernelTimingDrain");
  if (drain == nullptr) printf("ncclKernelTimingDrain not found; running without drain\n");

  ncclComm_t comm;
  CK(ncclCommInitAll(&comm, 1, nullptr));
  float *sendbuf, *recvbuf;
  CK(hipMalloc(&sendbuf, count * sizeof(float)));
  CK(hipMalloc(&recvbuf, count * sizeof(float)));
  hipStream_t stream;
  CK(hipStreamCreate(&stream));

  CK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
  CK(hipStreamSynchronize(stream));

  const char* mode = getenv("SMOKE_MODE");
  bool p2p = mode && strcmp(mode, "p2p") == 0;

  uint64_t before = bootNs();
  for (int i = 0; i < iters; i++) {
    if (p2p) {
      // A single-rank AllReduce is served by a memcpy shortcut, never reaching
      // ncclLaunchKernel; a self send/recv pair may still take the kernel path.
      CK(ncclGroupStart());
      CK(ncclSend(sendbuf, count, ncclFloat, 0, comm, stream));
      CK(ncclRecv(recvbuf, count, ncclFloat, 0, comm, stream));
      CK(ncclGroupEnd());
    } else {
      CK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
    }
  }
  CK(hipStreamSynchronize(stream));
  uint64_t after = bootNs();

  std::vector<Record> all;
  while (drain != nullptr) {
    Record buf[4096];
    int got = 0;
    uint64_t dropped = 0;
    ncclResult_t r = drain(comm, buf, 4096, &got, &dropped);
    if (r != ncclSuccess) {
      printf("drain returned %d (is RCCL_KERNEL_TIMING=1 set?)\n", (int)r);
      return 1;
    }
    all.insert(all.end(), buf, buf + got);
    if (got == 0) {
      printf("records %zu, dropped %llu\n", all.size(), (unsigned long long)dropped);
      break;
    }
  }

  if (all.empty()) {
    printf("no records captured\n");
    return 1;
  }

  int outOfWindow = 0, outOfOrder = 0;
  std::vector<double> dur;
  uint64_t prevEnd = 0;
  for (size_t i = 0; i < all.size(); i++) {
    const Record& r = all[i];
    if (r.startNs < before || r.endNs > after) outOfWindow++;
    if (r.startNs < prevEnd) outOfOrder++;
    prevEnd = r.endNs;
    dur.push_back((r.endNs - r.startNs) / 1000.0);
  }
  std::sort(dur.begin(), dur.end());

  printf("launched %d, captured %zu\n", iters, all.size());
  printf("outside host window: %d, overlapping previous dispatch: %d\n", outOfWindow, outOfOrder);
  printf("kernel us: min %.3f  median %.3f  max %.3f\n", dur.front(), dur[dur.size() / 2], dur.back());
  printf("host wall per iter: %.3f us\n", (after - before) / 1000.0 / iters);
  printf("first: seq %llu func %u dtype %u count %llu nChannels %u nThreads %u nColls %u\n",
         (unsigned long long)all[0].seq, all[0].func, all[0].datatype, (unsigned long long)all[0].count,
         all[0].nChannels, all[0].nThreads, all[0].nColls);

  CK(ncclCommDestroy(comm));
  return 0;
}
