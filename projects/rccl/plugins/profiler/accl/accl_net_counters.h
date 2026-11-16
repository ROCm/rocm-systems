#ifndef ACCL_NET_COUNTERS_H_
#define ACCL_NET_COUNTERS_H_

// Per-collective NIC counter snapshots (opt-in: ACCL_NET_COUNTERS=1).
//
// Snapshots are chained: S0 -> coll N -> S1 -> coll N+1 -> S2, so
// delta(N) = S1 - S0. A plugin-owned worker thread does every hardware read;
// profiler callbacks only post work and, before a launch, wait for a pending read.

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include "accl_boundary.h"

#define ACCL_HIDDEN __attribute__((visibility("hidden")))

#define ACCL_NET_MAX_DEVS     8
#define ACCL_NET_MAX_RDMA     128
#define ACCL_NET_MAX_ETH      64
#define ACCL_NET_MAX_SW       16
#define ACCL_NET_NAME_LEN     64
#define ACCL_NET_MAX_INFLIGHT 64

struct acclNetDevSnap {
  int      rdmaOk;
  int      ethOk;
  uint64_t rdmaStartUs, rdmaEndUs;
  uint64_t rdma[ACCL_NET_MAX_RDMA];  // indexed by the device's rdmaNames slot
  uint64_t eth[ACCL_NET_MAX_ETH];    // indexed by the device's ethNames slot
};

struct acclNetSwSnap {
  int      deviceId;
  uint64_t txBytes;
  uint64_t rxBytes;
  uint64_t cqErrors;
};

// Refcounted: shared by the chain head, collectives that took it as "before",
// and queued records.
struct acclNetSnapshot {
  int      refCount;
  uint64_t seq;
  uint64_t tsStartUs;
  uint64_t tsEndUs;
  int      stable;       // strict: two consecutive identical hardware reads
  int      readAttempts;
  int      nSw;          // -1: SW counters unavailable
  struct acclNetSwSnap  sw[ACCL_NET_MAX_SW];
  struct acclNetDevSnap dev[ACCL_NET_MAX_DEVS];
};

// Per-collective chain state, embedded in acclCollInfo. Written under the
// net-counter mutex only.
struct acclNetCommStats;
struct acclNetCollState {
  struct acclNetCommStats* owner;
  int    tracked;
  int    completionVerified;
  int    coordinated;
  int    posted;       // Coll stop seen: its proxy ops are posted
  int    overlapped;
  int    unavailable;
  uint64_t startUs;
  uint64_t launchWaitUs; // time the launch waited for this window's "before" read
  struct acclNetSnapshot* before;
};

// Per-communicator totals for the summary line. Written under the net mutex.
struct acclNetCommStats {
  int      boundarySupported;
  int      boundarySucceeded;
  int      strictFailed;
  int      finalBoundaryDone;
  void*    boundaryContext;
  int    (*synchronize)(void*, int);
  int      enabled;      // set at init, constant after
  int      nNodes;
  int      closed;       // set at finalize
  int      queued;       // records waiting for their "after" snapshot
  int      launchSeen;   // KernelLaunch fired since the last Coll start
  int      graceOff;     // this comm pipelines: skip the launch grace
  uint64_t launchWaitUs;
  uint64_t lastReadSeq;
  uint64_t reads;        // distinct "after" reads behind this comm's records
  uint64_t lateProxyOps; // proxy ops that started after their coll completed
  uint64_t records;
  uint64_t overlapped;
  uint64_t unavailable;
  uint64_t readErrors;   // records whose "after" read failed
  uint64_t readUsSum;
  uint64_t readUsMax;
  uint64_t totals[ACCL_NET_MAX_DEVS][ACCL_NET_MAX_RDMA];
};

// Test seams. NULL means the real call; snapshotFn replaces every device read.
struct acclNetHooks {
  int     (*nlOpenFn)(void);
  ssize_t (*sendFn)(int fd, const void* buf, size_t len);
  ssize_t (*recvFn)(int fd, void* buf, size_t len);
  int     (*ioctlFn)(int fd, unsigned long req, void* arg);
  int     (*swCaptureFn)(void* out, int maxDevs);
  int     (*snapshotFn)(struct acclNetSnapshot* snap);
};

struct acclCommContext;
struct acclCompletedRecord;

// Provided by accl_profiler.cc.
ACCL_HIDDEN void acclWriteRecord(struct acclCommContext* ctx,
                                 struct acclCompletedRecord* rec, const char* netJson);
ACCL_HIDDEN void acclCtxRetain(struct acclCommContext* ctx);
ACCL_HIDDEN void acclCtxRelease(struct acclCommContext* ctx);

// Lifecycle. acclNetInit returns 1 if this communicator records net counters.
ACCL_HIDDEN int  acclNetInit(struct acclNetCommStats* st, int rank, int nRanks, int nNodes);
// Writes this comm's queued records and returns ",\"net_counters\":{...}" for
// the summary line (malloc'd; NULL when off). *lost: records not written in time.
ACCL_HIDDEN char* acclNetFinalize(struct acclNetCommStats* st, int* lost);

// Event hooks.
ACCL_HIDDEN void acclNetBoundaryInfo(struct acclNetCommStats*, const struct rcclAcclBoundaryInfo*);
ACCL_HIDDEN void acclNetFinalBoundary(struct acclNetCommStats*, const struct rcclAcclBoundaryInfo*);
ACCL_HIDDEN void acclNetRegistered(struct acclNetCollState*);
ACCL_HIDDEN void acclNetKernelLaunch(struct acclNetCommStats* st);
ACCL_HIDDEN void acclNetCollStart(struct acclNetCommStats* st, struct acclNetCollState* cs);
ACCL_HIDDEN void acclNetCollPosted(struct acclNetCollState* cs);
// Work no record owns (P2P, filtered or symmetric colls): flags the windows it
// can reach. st may be NULL.
ACCL_HIDDEN void acclNetUntrackedStart(struct acclNetCommStats* st);
// A proxy op no tracked record owns, live until acclNetUntrackedOpStop. late:
// its collective had already completed.
ACCL_HIDDEN void acclNetUntrackedOpStart(struct acclNetCommStats* st, int late);
ACCL_HIDDEN void acclNetUntrackedOpStop(void);
// Traffic of this collective the plugin cannot follow (e.g. proxy-op pool full).
ACCL_HIDDEN void acclNetCollFlag(struct acclNetCollState* cs);
// Queues rec for the worker, which writes it once the "after" snapshot is read.
// Returns 0 if queued; otherwise the caller writes rec itself.
// nProxyOps: proxy ops that started; 0 on a multi-node comm delays the read
// briefly, since a send-only rank's ops can start after its kernel finished.
ACCL_HIDDEN int  acclNetCollComplete(struct acclCommContext* ctx, struct acclNetCommStats* st,
                                     struct acclNetCollState* cs, struct acclCompletedRecord* rec,
                                     int nProxyOps);
ACCL_HIDDEN void acclNetCollDrop(struct acclNetCollState* cs);

// Parsers, exposed for unit tests.
// RDMA_NLDEV_CMD_GET dump: index of device `name`, -1 if absent, -2 on
// NLMSG_ERROR. Sets *done once NLMSG_DONE is seen.
ACCL_HIDDEN int acclNlFindDevIndex(const void* buf, size_t len, const char* name, int* done);
// RDMA_NLDEV_CMD_STAT_GET reply: calls cb once per hw counter, in reply order.
// Returns the number of counters, or -1 on NLMSG_ERROR, a malformed entry, or
// no STAT_GET message.
ACCL_HIDDEN int acclNlParseHwCounters(const void* buf, size_t len,
                                      void (*cb)(const char* name, uint64_t value, int pos, void* arg),
                                      void* arg);
// ethtool stats kept in records (PFC / pause).
ACCL_HIDDEN int acclEthKeepName(const char* name);

#endif
