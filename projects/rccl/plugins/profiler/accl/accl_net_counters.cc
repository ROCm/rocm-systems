/*
 * ACCL Profiler — per-collective NIC counter snapshots.
 *
 * Readers: RDMA hw counters over RDMA netlink (sysfs hw_counters are cached for
 * 10 ms), ethtool stats over SIOCETHTOOL, and RCCL's SW telemetry counters.
 * Only the worker thread calls a reader.
 */

#include "accl_net_counters.h"

#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <linux/ethtool.h>
#include <linux/netlink.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <pthread.h>
#include <rdma/rdma_netlink.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "accl_profiler.h"
#include "accl_shim.h"
#include "net_telemetry.h"

extern ncclDebugLogger_t gLogFn;
#define ACCL_NET_INFO(...) do { if (gLogFn) gLogFn(4, 0x4000, __func__, __LINE__, __VA_ARGS__); } while (0)
#define ACCL_NET_WARN(...) do { if (gLogFn) gLogFn(3, ~0UL, __func__, __LINE__, __VA_ARGS__); } while (0)

#define ACCL_NL_BUF_SIZE    32768
#define ACCL_ETH_HEADROOM   256   // old kernels ignore n_stats/len and copy the current count
#define ACCL_NET_MAX_QUEUED 256   // force a read if overlapping collectives never drain
#define ACCL_NET_SETTLE_US  200   // lets a late-waking proxy thread start its ops
#define ACCL_NET_STUCK_US   10000000  // in flight this long: never going to complete

static const char* const kAcclVolumeCounters[] = {
  "tx_rdma_ucast_bytes", "tx_rdma_ucast_pkts", "rx_rdma_ucast_bytes", "rx_rdma_ucast_pkts",
};

struct acclNetDev {
  char     name[ACCL_NET_NAME_LEN];
  // RDMA netlink
  int      nlFd;
  int      nlIndex;
  uint32_t nlSeq;
  int      nRdma;
  char     rdmaNames[ACCL_NET_MAX_RDMA][ACCL_NET_NAME_LEN];
  // ethtool
  char     netdev[IFNAMSIZ];
  int      ethFd;
  uint32_t ethNStats;
  int      nEth;
  char     ethNames[ACCL_NET_MAX_ETH][ETH_GSTRING_LEN + 1];
  uint32_t ethIdx[ACCL_NET_MAX_ETH];
  struct ethtool_stats* ethBuf;
};

struct acclNetJob {
  struct acclNetJob*       next;
  struct acclCommContext*  ctx;
  struct acclNetCommStats* st;
  struct acclNetSnapshot*  before;
  int                      overlapped;
  int                      unavailable;
  int                      settle;
  int                      completionVerified;
  int                      coordinated;
  uint64_t                 startUs, completeUs;
  uint64_t                 launchWaitUs;
  struct acclCompletedRecord rec;
};

static pthread_mutex_t gNetMu = PTHREAD_MUTEX_INITIALIZER;

// Shared by every communicator in the process: the NIC counters are too.
static struct {
  pthread_cond_t  workCv;   // worker: a read was requested
  pthread_cond_t  doneCv;   // waiters: a read finished, a coll completed, a job was written
  int             condInit;
  pthread_t       thread;
  int             users;
  int             strict;
  int             unknownTraffic;
  int             stop;
  uint64_t        waitUs;
  uint64_t        graceUs;
  int             nDevs;
  struct acclNetDev devs[ACCL_NET_MAX_DEVS];
  int           (*swCapture)(RcclTelemetrySwSnapshot*, int);
  uint64_t        snapSeq;
  char            selectedDev[ACCL_NET_NAME_LEN];
  struct acclNetSnapshot* cur;      // chain head: the next collective's "before"
  int             readPending;      // requested or in progress
  uint64_t        readSeq;          // bumped per requested read
  uint64_t        timedOutSeq;      // a read some caller already gave up on
  int             stopping;         // last finalize is joining the worker
  int             untrackedLive;    // untracked proxy ops started, not yet stopped
  uint64_t        untrackedLastUs;
  struct acclNetCollState* inflight[ACCL_NET_MAX_INFLIGHT];
  int             nInflight;
  int             dirty;            // untracked work launched since the last tracked start
  struct acclNetJob* head;
  struct acclNetJob* tail;
  int             nJobs;
  struct acclNetJob* reading;       // jobs whose "after" read is in progress
} g;

ACCL_HIDDEN struct acclNetHooks gAcclNetHooks;

// ============================================================================
// Small helpers
// ============================================================================
static uint64_t acclNetNowUs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

// Wait on doneCv until `deadlineUs` (CLOCK_MONOTONIC). Returns 0 on timeout.
static int acclNetWaitUntil(uint64_t deadlineUs) {
  uint64_t now = acclNetNowUs();
  if (now >= deadlineUs) return 0;
  struct timespec ts;
  ts.tv_sec = (time_t)(deadlineUs / 1000000ULL);
  ts.tv_nsec = (long)((deadlineUs % 1000000ULL) * 1000ULL);
  pthread_cond_timedwait(&g.doneCv, &gNetMu, &ts);
  return 1;
}

static struct acclNetSnapshot* acclSnapRef(struct acclNetSnapshot* s) {
  if (s) __atomic_add_fetch(&s->refCount, 1, __ATOMIC_SEQ_CST);
  return s;
}

static void acclSnapUnref(struct acclNetSnapshot* s) {
  if (s && __atomic_sub_fetch(&s->refCount, 1, __ATOMIC_SEQ_CST) == 0) free(s);
}

struct acclStr {
  char*  p;
  size_t n;
  size_t cap;
  int    oom;
};

static void acclStrAppend(struct acclStr* s, const char* fmt, ...) {
  if (s->oom) return;
  for (;;) {
    va_list ap;
    va_start(ap, fmt);
    size_t room = s->cap - s->n;
    int k = vsnprintf(s->p ? s->p + s->n : NULL, s->p ? room : 0, fmt, ap);
    va_end(ap);
    if (k < 0) { s->oom = 1; return; }
    if (s->p && (size_t)k < room) { s->n += (size_t)k; return; }
    size_t cap = s->cap ? s->cap * 2 : 4096;
    while (cap < s->n + (size_t)k + 1) cap *= 2;
    char* np = (char*)realloc(s->p, cap);
    if (!np) { s->oom = 1; return; }
    s->p = np;
    s->cap = cap;
  }
}

// Names go into JSON unescaped, so accept only the characters they use.
static int acclNetNameOk(const char* s) {
  if (!s[0]) return 0;
  for (; *s; s++) {
    if (!isalnum((unsigned char)*s) && *s != '_' && *s != '-' && *s != '.' && *s != ':') return 0;
  }
  return 1;
}

static int acclNetIsVolume(const char* name) {
  for (size_t i = 0; i < sizeof(kAcclVolumeCounters) / sizeof(kAcclVolumeCounters[0]); i++) {
    if (strcmp(name, kAcclVolumeCounters[i]) == 0) return 1;
  }
  return 0;
}

// ============================================================================
// RDMA netlink
// ============================================================================
static int acclNlaOk(const struct nlattr* a, int rem) {
  return rem >= (int)sizeof(*a) && a->nla_len >= sizeof(*a) && (int)a->nla_len <= rem;
}

static const struct nlattr* acclNlaNext(const struct nlattr* a, int* rem) {
  int adv = NLA_ALIGN(a->nla_len);
  *rem -= adv;
  return (const struct nlattr*)((const char*)a + adv);
}

static const void* acclNlaData(const struct nlattr* a) { return (const char*)a + NLA_HDRLEN; }
static int acclNlaLen(const struct nlattr* a) { return (int)a->nla_len - NLA_HDRLEN; }
static int acclNlaType(const struct nlattr* a) { return a->nla_type & NLA_TYPE_MASK; }

// A NUL-terminated string attribute, or NULL.
static const char* acclNlaStr(const struct nlattr* a) {
  int n = acclNlaLen(a);
  const char* s = (const char*)acclNlaData(a);
  return (n > 0 && memchr(s, '\0', n)) ? s : NULL;
}

int acclNlFindDevIndex(const void* buf, size_t len, const char* name, int* done) {
  const uint16_t getType = RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_GET);
  int found = -1;
  int rem = (int)len;
  for (const struct nlmsghdr* nh = (const struct nlmsghdr*)buf; NLMSG_OK(nh, rem); nh = NLMSG_NEXT(nh, rem)) {
    if (nh->nlmsg_type == NLMSG_DONE) { *done = 1; break; }
    if (nh->nlmsg_type == NLMSG_ERROR) {
      const struct nlmsgerr* e = (const struct nlmsgerr*)NLMSG_DATA(nh);
      if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(*e)) || e->error != 0) return -2;
      continue;
    }
    if (nh->nlmsg_type != getType) continue;
    int idx = -1;
    const char* devName = NULL;
    int arem = (int)nh->nlmsg_len - NLMSG_HDRLEN;
    for (const struct nlattr* a = (const struct nlattr*)NLMSG_DATA(nh); acclNlaOk(a, arem); a = acclNlaNext(a, &arem)) {
      if (acclNlaType(a) == RDMA_NLDEV_ATTR_DEV_INDEX && acclNlaLen(a) >= 4) {
        uint32_t v;
        memcpy(&v, acclNlaData(a), 4);
        idx = (int)v;
      } else if (acclNlaType(a) == RDMA_NLDEV_ATTR_DEV_NAME) {
        devName = acclNlaStr(a);
      }
    }
    if (idx >= 0 && devName && strcmp(devName, name) == 0) found = idx;
  }
  return found;
}

int acclNlParseHwCounters(const void* buf, size_t len,
                          void (*cb)(const char* name, uint64_t value, int pos, void* arg),
                          void* arg) {
  const uint16_t statType = RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_STAT_GET);
  int seen = -1;
  int rem = (int)len;
  for (const struct nlmsghdr* nh = (const struct nlmsghdr*)buf; NLMSG_OK(nh, rem); nh = NLMSG_NEXT(nh, rem)) {
    if (nh->nlmsg_type == NLMSG_ERROR) {
      const struct nlmsgerr* e = (const struct nlmsgerr*)NLMSG_DATA(nh);
      if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(*e)) || e->error != 0) return -1;
      continue;
    }
    if (nh->nlmsg_type != statType) continue;
    int pos = 0;
    int arem = (int)nh->nlmsg_len - NLMSG_HDRLEN;
    for (const struct nlattr* a = (const struct nlattr*)NLMSG_DATA(nh); acclNlaOk(a, arem); a = acclNlaNext(a, &arem)) {
      if (acclNlaType(a) != RDMA_NLDEV_ATTR_STAT_HWCOUNTERS) continue;
      int trem = acclNlaLen(a);
      for (const struct nlattr* e = (const struct nlattr*)acclNlaData(a); acclNlaOk(e, trem); e = acclNlaNext(e, &trem)) {
        if (acclNlaType(e) != RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY) continue;
        const char* name = NULL;
        int haveValue = 0;
        uint64_t value = 0;
        int erem = acclNlaLen(e);
        for (const struct nlattr* f = (const struct nlattr*)acclNlaData(e); acclNlaOk(f, erem); f = acclNlaNext(f, &erem)) {
          if (acclNlaType(f) == RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME) {
            name = acclNlaStr(f);
          } else if (acclNlaType(f) == RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_VALUE && acclNlaLen(f) >= 8) {
            memcpy(&value, acclNlaData(f), 8);
            haveValue = 1;
          }
        }
        if (!name || !haveValue) return -1;
        cb(name, value, pos++, arg);
      }
    }
    seen = pos;
  }
  return seen;
}

static int acclNlOpenDefault(void) {
  int fd = socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_RDMA);
  if (fd < 0) return -1;
  struct sockaddr_nl sa;
  memset(&sa, 0, sizeof(sa));
  sa.nl_family = AF_NETLINK;
  // A stuck firmware command must not hang the worker forever.
  struct timeval tv = {1, 0};
  if (bind(fd, (struct sockaddr*)&sa, sizeof(sa)) != 0 ||
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

static ssize_t acclNlSend(int fd, const void* buf, size_t len) {
  if (gAcclNetHooks.sendFn) return gAcclNetHooks.sendFn(fd, buf, len);
  struct sockaddr_nl sa;
  memset(&sa, 0, sizeof(sa));
  sa.nl_family = AF_NETLINK;
  return sendto(fd, buf, len, 0, (struct sockaddr*)&sa, sizeof(sa));
}

static ssize_t acclNlRecv(int fd, void* buf, size_t len) {
  if (gAcclNetHooks.recvFn) return gAcclNetHooks.recvFn(fd, buf, len);
  return recv(fd, buf, len, 0);
}

static void acclNlPutU32(char* buf, struct nlmsghdr* nh, uint16_t type, uint32_t v) {
  struct nlattr* a = (struct nlattr*)(buf + NLMSG_ALIGN(nh->nlmsg_len));
  a->nla_type = type;
  a->nla_len = NLA_HDRLEN + sizeof(v);
  memcpy((char*)a + NLA_HDRLEN, &v, sizeof(v));
  nh->nlmsg_len = NLMSG_ALIGN(nh->nlmsg_len) + NLA_ALIGN(a->nla_len);
}

static int acclNlRequest(struct acclNetDev* d, uint16_t cmd, uint16_t flags, int withDev) {
  char buf[64];
  memset(buf, 0, sizeof(buf));
  struct nlmsghdr* nh = (struct nlmsghdr*)buf;
  nh->nlmsg_len = NLMSG_LENGTH(0);
  nh->nlmsg_type = RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, cmd);
  nh->nlmsg_flags = flags;
  nh->nlmsg_seq = ++d->nlSeq;
  if (withDev) {
    acclNlPutU32(buf, nh, RDMA_NLDEV_ATTR_DEV_INDEX, (uint32_t)d->nlIndex);
    acclNlPutU32(buf, nh, RDMA_NLDEV_ATTR_PORT_INDEX, 1);
  }
  return acclNlSend(d->nlFd, buf, nh->nlmsg_len) == (ssize_t)nh->nlmsg_len ? 0 : -1;
}

static int acclRdmaResolve(struct acclNetDev* d, char* rx) {
  if (acclNlRequest(d, RDMA_NLDEV_CMD_GET, NLM_F_REQUEST | NLM_F_DUMP, 0) != 0) return -1;
  int done = 0, idx = -1;
  for (int i = 0; i < 64 && !done; i++) {
    ssize_t n = acclNlRecv(d->nlFd, rx, ACCL_NL_BUF_SIZE);
    if (n <= 0) return -1;
    int r = acclNlFindDevIndex(rx, (size_t)n, d->name, &done);
    if (r == -2) return -1;
    if (r >= 0) idx = r;
  }
  if (!done || idx < 0) return -1;
  d->nlIndex = idx;
  return 0;
}

static int acclRdmaOpen(struct acclNetDev* d, char* rx) {
  d->nlFd = gAcclNetHooks.nlOpenFn ? gAcclNetHooks.nlOpenFn() : acclNlOpenDefault();
  if (d->nlFd < 0) return -1;
  if (acclRdmaResolve(d, rx) != 0) {
    close(d->nlFd);
    d->nlFd = -1;
    return -1;
  }
  return 0;
}

struct acclRdmaParse {
  struct acclNetDev* d;
  uint64_t*          out;
  int                learn;
  int                filled;  // distinct slots
  uint8_t            seen[ACCL_NET_MAX_RDMA];
};

static void acclRdmaCounterCb(const char* name, uint64_t value, int pos, void* arg) {
  struct acclRdmaParse* p = (struct acclRdmaParse*)arg;
  struct acclNetDev* d = p->d;
  int slot = -1;
  if (p->learn) {
    if (d->nRdma >= ACCL_NET_MAX_RDMA || strlen(name) >= ACCL_NET_NAME_LEN || !acclNetNameOk(name)) return;
    slot = d->nRdma++;
    strcpy(d->rdmaNames[slot], name);
  } else if (pos < d->nRdma && strcmp(d->rdmaNames[pos], name) == 0) {
    slot = pos;
  } else {
    // Reply order moved: map by name. Counters added since the first read are ignored.
    for (int i = 0; i < d->nRdma; i++) {
      if (strcmp(d->rdmaNames[i], name) == 0) { slot = i; break; }
    }
  }
  if (slot < 0) return;
  p->out[slot] = value;
  if (!p->seen[slot]) p->filled++;
  p->seen[slot] = 1;
}

// Parses one STAT_GET reply into out[] by slot. Only the first read (at init,
// before the worker starts) learns the names. Returns 0 on success.
static int acclRdmaParseReply(struct acclNetDev* d, const void* buf, size_t len, uint64_t* out, int learn) {
  if (!learn && d->nRdma == 0) return -1;
  struct acclRdmaParse p;
  memset(&p, 0, sizeof(p));
  p.d = d;
  p.out = out;
  p.learn = learn;
  int n = acclNlParseHwCounters(buf, len, acclRdmaCounterCb, &p);
  // A counter that disappeared would leave a stale slot.
  if (n < 0 || p.filled < d->nRdma || d->nRdma == 0) {
    if (learn) d->nRdma = 0;
    return -1;
  }
  return 0;
}

static int acclRdmaRead(struct acclNetDev* d, char* rx, uint64_t* out, int learn) {
  if (d->nlFd < 0) return -1;
  if (acclNlRequest(d, RDMA_NLDEV_CMD_STAT_GET, NLM_F_REQUEST, 1) != 0) return -1;
  // Skip replies to an earlier request that timed out.
  for (int i = 0; i < 8; i++) {
    ssize_t n = acclNlRecv(d->nlFd, rx, ACCL_NL_BUF_SIZE);
    if (n <= 0) return -1;
    if ((size_t)n < sizeof(struct nlmsghdr)) return -1;
    if (((struct nlmsghdr*)rx)->nlmsg_seq != d->nlSeq) continue;
    return acclRdmaParseReply(d, rx, (size_t)n, out, learn);
  }
  return -1;
}

// ============================================================================
// ethtool
// ============================================================================
int acclEthKeepName(const char* name) {
  return strstr(name, "pause") != NULL || strstr(name, "pfc") != NULL;
}

static int acclEthIoctl(struct acclNetDev* d, void* data) {
  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  memcpy(ifr.ifr_name, d->netdev, sizeof(ifr.ifr_name) - 1);
  ifr.ifr_data = (char*)data;
  if (gAcclNetHooks.ioctlFn) return gAcclNetHooks.ioctlFn(d->ethFd, SIOCETHTOOL, &ifr);
  return ioctl(d->ethFd, SIOCETHTOOL, &ifr);
}

// Fetches the stat names. The first fetch picks the kept names and fixes their
// slots; later fetches (after the stat count changed) re-map them by name.
static int acclEthFetchNames(struct acclNetDev* d) {
  struct ethtool_drvinfo info;
  memset(&info, 0, sizeof(info));
  info.cmd = ETHTOOL_GDRVINFO;
  if (acclEthIoctl(d, &info) != 0 || info.n_stats == 0) return -1;
  uint32_t n = info.n_stats;
  struct ethtool_gstrings* gs = (struct ethtool_gstrings*)calloc(
    1, sizeof(*gs) + ((size_t)n + ACCL_ETH_HEADROOM) * ETH_GSTRING_LEN);
  if (!gs) return -1;
  gs->cmd = ETHTOOL_GSTRINGS;
  gs->string_set = ETH_SS_STATS;
  gs->len = n;
  if (acclEthIoctl(d, gs) != 0) { free(gs); return -1; }
  if (gs->len < n) n = gs->len;

  int learn = d->nEth == 0;
  int ok = 1;
  uint32_t idx[ACCL_NET_MAX_ETH];
  if (learn) {
    for (uint32_t i = 0; i < n && d->nEth < ACCL_NET_MAX_ETH; i++) {
      char name[ETH_GSTRING_LEN + 1];
      memcpy(name, gs->data + (size_t)i * ETH_GSTRING_LEN, ETH_GSTRING_LEN);
      name[ETH_GSTRING_LEN] = '\0';
      if (!acclEthKeepName(name) || !acclNetNameOk(name)) continue;
      memcpy(d->ethNames[d->nEth], name, sizeof(name));
      d->ethIdx[d->nEth++] = i;
    }
  } else {
    for (int k = 0; k < d->nEth && ok; k++) {
      ok = 0;
      for (uint32_t i = 0; i < n; i++) {
        if (strncmp(d->ethNames[k], (const char*)gs->data + (size_t)i * ETH_GSTRING_LEN, ETH_GSTRING_LEN) == 0) {
          idx[k] = i;
          ok = 1;
          break;
        }
      }
    }
  }
  free(gs);
  if (!ok || d->nEth == 0) return -1;
  if (!learn) memcpy(d->ethIdx, idx, sizeof(uint32_t) * d->nEth);

  struct ethtool_stats* buf = (struct ethtool_stats*)realloc(
    d->ethBuf, sizeof(*buf) + ((size_t)n + ACCL_ETH_HEADROOM) * sizeof(uint64_t));
  if (!buf) return -1;
  d->ethBuf = buf;
  d->ethNStats = n;
  return 0;
}

static int acclEthRead(struct acclNetDev* d, uint64_t* out) {
  if (d->ethFd < 0 || !d->ethBuf) return -1;
  for (int attempt = 0; attempt < 2; attempt++) {
    d->ethBuf->cmd = ETHTOOL_GSTATS;
    d->ethBuf->n_stats = d->ethNStats;
    if (acclEthIoctl(d, d->ethBuf) != 0) return -1;
    // n_stats == 0: the stat count changed since GSTRINGS.
    if (d->ethBuf->n_stats != d->ethNStats) {
      if (acclEthFetchNames(d) != 0) return -1;
      continue;
    }
    for (int k = 0; k < d->nEth; k++) out[k] = d->ethBuf->data[d->ethIdx[k]];
    return 0;
  }
  return -1;
}

static void acclEthOpen(struct acclNetDev* d) {
  char path[256];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/net", d->name);
  DIR* dir = opendir(path);
  if (dir) {
    struct dirent* e;
    while ((e = readdir(dir)) != NULL) {
      if (e->d_name[0] == '.' || strlen(e->d_name) >= sizeof(d->netdev)) continue;
      strcpy(d->netdev, e->d_name);
      break;
    }
    closedir(dir);
  }
  if (!d->netdev[0]) return;
  d->ethFd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (d->ethFd >= 0 && acclEthFetchNames(d) != 0) {
    ACCL_NET_INFO("ACCL Profiler: no ethtool PFC/pause stats on %s (%s)", d->netdev, d->name);
  }
}

// ============================================================================
// Snapshot
// ============================================================================
static void acclNetReadSw(struct acclNetSnapshot* s) {
  s->nSw = -1;
  // Headroom: a librccl with a larger struct writes past our ACCL_NET_MAX_SW entries.
  RcclTelemetrySwSnapshot sw[4 * ACCL_NET_MAX_SW];
  int n = 0;
  if (gAcclNetHooks.swCaptureFn) n = gAcclNetHooks.swCaptureFn(sw, ACCL_NET_MAX_SW);
  else if (g.swCapture) n = g.swCapture(sw, ACCL_NET_MAX_SW);
  if (n <= 0) return;
  if (n > ACCL_NET_MAX_SW) n = ACCL_NET_MAX_SW;
  for (int i = 0; i < n; i++) {
    s->sw[i].deviceId = sw[i].device_id;
    s->sw[i].txBytes = sw[i].tx_bytes;
    s->sw[i].rxBytes = sw[i].rx_bytes;
    s->sw[i].cqErrors = sw[i].num_cq_errors;
  }
  s->nSw = n;
}

static void acclNetRequestReadLocked(void);
static int acclNetUntrackedActive(void);
static void acclNetFlagPendingJobs(void);

// Worker thread only.
static struct acclNetSnapshot* acclNetReadAll(char* rx) {
  struct acclNetSnapshot* s = (struct acclNetSnapshot*)calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->refCount = 1;
  s->seq = ++g.snapSeq;
  s->tsStartUs = acclNetNowUs();
  if (gAcclNetHooks.snapshotFn) {
    if (gAcclNetHooks.snapshotFn(s) != 0) {
      for (int i = 0; i < ACCL_NET_MAX_DEVS; i++) s->dev[i].rdmaOk = s->dev[i].ethOk = 0;
      s->nSw = -1;
    }
  } else {
    for (int i = 0; i < g.nDevs; i++) {
      s->dev[i].rdmaStartUs = acclNetNowUs();
      s->dev[i].rdmaOk = acclRdmaRead(&g.devs[i], rx, s->dev[i].rdma, 0) == 0;
      s->dev[i].rdmaEndUs = acclNetNowUs();
      s->dev[i].ethOk = acclEthRead(&g.devs[i], s->dev[i].eth) == 0;
    }
    acclNetReadSw(s);
  }
  s->tsEndUs = acclNetNowUs();
  return s;
}

// Preserve the raw endpoints and read intervals so attribution can be audited.
static void acclNetRawSnapshot(struct acclStr* js, const struct acclNetSnapshot* s) {
  if (!s) { acclStrAppend(js, "null"); return; }
  acclStrAppend(js, "{\"id\":%lu,\"read_start_us\":%lu,\"read_end_us\":%lu,\"devices\":[",
                (unsigned long)s->seq, (unsigned long)s->tsStartUs, (unsigned long)s->tsEndUs);
  for (int i = 0; i < g.nDevs; i++) {
    acclStrAppend(js, "%s{\"name\":\"%s\",\"ok\":%s,\"read_start_us\":%lu,\"read_end_us\":%lu,\"rdma\":{",
      i ? "," : "", g.devs[i].name, s->dev[i].rdmaOk ? "true" : "false",
      (unsigned long)s->dev[i].rdmaStartUs, (unsigned long)s->dev[i].rdmaEndUs);
    for (int k = 0; k < g.devs[i].nRdma; k++) acclStrAppend(js, "%s\"%s\":%lu",
      k ? "," : "", g.devs[i].rdmaNames[k], (unsigned long)s->dev[i].rdma[k]);
    acclStrAppend(js, "}}");
  }
  acclStrAppend(js, "],\"stable\":%s,\"read_attempts\":%d}", s->stable ? "true" : "false", s->readAttempts);
}

// Counter values must settle while every rank remains behind the boundary.
// This validates observed quiescence; it does not claim a firmware atomic latch.
static int acclNetSameHardware(const struct acclNetSnapshot* a, const struct acclNetSnapshot* b) {
  if (!a || !b) return 0;
  for (int i = 0; i < g.nDevs; i++) {
    if (!a->dev[i].rdmaOk || !b->dev[i].rdmaOk) return 0;
    for (int k = 0; k < g.devs[i].nRdma; k++)
      if (a->dev[i].rdma[k] != b->dev[i].rdma[k]) return 0;
  }
  return 1;
}

// Builds the record's "net_counters" object and updates the comm totals.
// Called with gNetMu held.
static void acclNetFormat(struct acclStr* js, struct acclNetJob* job, struct acclNetSnapshot* a) {
  struct acclNetSnapshot* b = job->before;
  struct acclNetCommStats* st = job->st;
  int unavailable = job->unavailable || !b || !a || g.nDevs == 0;
  // A late "before" read already holds the previous collective's tail.
  int trusted = !job->unavailable && b && a && (!g.strict || (b->stable && a->stable));
  int readError = !a;
  int usable[ACCL_NET_MAX_DEVS] = {0};
  for (int i = 0; i < g.nDevs && a; i++) {
    if (!a->dev[i].rdmaOk) readError = 1;
    if (g.devs[i].ethBuf && !a->dev[i].ethOk) readError = 1;
    usable[i] = trusted && b->dev[i].rdmaOk && a->dev[i].rdmaOk;
    if (usable[i]) for (int k = 0; k < g.devs[i].nRdma; k++) {
      if (a->dev[i].rdma[k] < b->dev[i].rdma[k]) { usable[i] = 0; break; }
    }
    if (!usable[i]) unavailable = 1;
  }

  acclStrAppend(js, "\"net_counters\":{\"overlapped\":%s,\"unavailable\":%s",
                job->overlapped ? "true" : "false", unavailable ? "true" : "false");
  acclStrAppend(js, ",\"completion_verified\":%s,\"coordinated\":%s,\"collective_start_cpu_us\":%lu,\"collective_complete_cpu_us\":%lu",
    job->completionVerified ? "true" : "false", job->coordinated ? "true" : "false",
    (unsigned long)job->startUs, (unsigned long)job->completeUs);
  acclStrAppend(js, ",\"counter_discontinuities\":[");
  int discontinuity = 0;
  if (b && a) for (int i = 0; i < g.nDevs; i++) {
    if (!b->dev[i].rdmaOk || !a->dev[i].rdmaOk) continue;
    for (int k = 0; k < g.devs[i].nRdma; k++) if (a->dev[i].rdma[k] < b->dev[i].rdma[k]) {
      acclStrAppend(js, "%s\"%s/%s\"", discontinuity++ ? "," : "", g.devs[i].name, g.devs[i].rdmaNames[k]);
    }
  }
  acclStrAppend(js, "]");
  if (b && a) {
    acclStrAppend(js, ",\"before_ts_us\":%lu,\"after_ts_us\":%lu,\"after_read_us\":%lu",
                  (unsigned long)b->tsStartUs, (unsigned long)a->tsStartUs,
                  (unsigned long)(a->tsEndUs - a->tsStartUs));
  }
  acclStrAppend(js, ",\"launch_wait_us\":%lu", (unsigned long)job->launchWaitUs);
  acclStrAppend(js, ",\"devices\":[");
  for (int i = 0; i < g.nDevs; i++) {
    struct acclNetDev* d = &g.devs[i];
    acclStrAppend(js, "%s{\"name\":\"%s\",\"rdma\":", i ? "," : "", d->name);
    if (usable[i]) {
      acclStrAppend(js, "{");
      int first = 1;
      for (int k = 0; k < d->nRdma; k++) {
        // Narrow firmware counters wrap; a counter that went backwards is omitted.
        if (a->dev[i].rdma[k] < b->dev[i].rdma[k]) continue;
        uint64_t delta = a->dev[i].rdma[k] - b->dev[i].rdma[k];
        if (!job->overlapped) st->totals[i][k] += delta;
        if (delta == 0 && !acclNetIsVolume(d->rdmaNames[k])) continue;
        acclStrAppend(js, "%s\"%s\":%lu", first ? "" : ",", d->rdmaNames[k], (unsigned long)delta);
        first = 0;
      }
      acclStrAppend(js, "}");
    } else {
      acclStrAppend(js, "null");
    }
    acclStrAppend(js, ",\"ethtool\":");
    if (trusted && !unavailable && b && a && b->dev[i].ethOk && a->dev[i].ethOk) {
      // ionic refreshes these about every 5 s: context only, never per-collective.
      acclStrAppend(js, "{\"stale\":true");
      for (int k = 0; k < d->nEth; k++) {
        if (a->dev[i].eth[k] <= b->dev[i].eth[k]) continue;
        acclStrAppend(js, ",\"%s\":%lu", d->ethNames[k],
                      (unsigned long)(a->dev[i].eth[k] - b->dev[i].eth[k]));
      }
      acclStrAppend(js, "}");
    } else {
      acclStrAppend(js, "null");
    }
    acclStrAppend(js, "}");
  }
  acclStrAppend(js, "],\"sw\":");
  int swUsable = trusted && !unavailable && b && a && b->nSw > 0 && b->nSw == a->nSw;
  if (swUsable) for (int i = 0; i < a->nSw; i++) {
    if (a->sw[i].deviceId != b->sw[i].deviceId || a->sw[i].txBytes < b->sw[i].txBytes ||
        a->sw[i].rxBytes < b->sw[i].rxBytes || a->sw[i].cqErrors < b->sw[i].cqErrors) swUsable = 0;
  }
  if (swUsable) {
    acclStrAppend(js, "[");
    for (int i = 0; i < a->nSw; i++) {
      acclStrAppend(js, "%s{\"device_id\":%d,\"tx_bytes\":%lu,\"rx_bytes\":%lu,\"cq_errors\":%lu}",
                    i ? "," : "", a->sw[i].deviceId,
                    (unsigned long)(a->sw[i].txBytes - b->sw[i].txBytes),
                    (unsigned long)(a->sw[i].rxBytes - b->sw[i].rxBytes),
                    (unsigned long)(a->sw[i].cqErrors - b->sw[i].cqErrors));
    }
    acclStrAppend(js, "]");
  } else {
    acclStrAppend(js, "null");
  }
  acclStrAppend(js, ",\"snapshot_before\":"); acclNetRawSnapshot(js, b);
  acclStrAppend(js, ",\"snapshot_after\":"); acclNetRawSnapshot(js, a);
  acclStrAppend(js, "}");

  st->records++;
  if (job->overlapped) st->overlapped++;
  if (unavailable) st->unavailable++;
  if (readError) st->readErrors++;
  // Overlapping records share one "after" read: count each read once.
  if (a && a->seq != st->lastReadSeq) {
    uint64_t us = a->tsEndUs - a->tsStartUs;
    st->lastReadSeq = a->seq;
    st->reads++;
    st->readUsSum += us;
    if (us > st->readUsMax) st->readUsMax = us;
  }
}

// Writes and frees one job; drops gNetMu around the write. snap NULL: no counters.
static void acclNetWriteJobLocked(struct acclNetJob* job, struct acclNetSnapshot* snap) {
  struct acclStr js = {NULL, 0, 0, 0};
  if (snap) acclNetFormat(&js, job, snap);
  pthread_mutex_unlock(&gNetMu);
  acclWriteRecord(job->ctx, &job->rec, js.oom ? NULL : js.p);
  free(js.p);
  acclSnapUnref(job->before);
  struct acclCommContext* ctx = job->ctx;
  pthread_mutex_lock(&gNetMu);
  job->st->queued--;
  pthread_cond_broadcast(&g.doneCv);
  pthread_mutex_unlock(&gNetMu);
  // Last: st lives inside ctx.
  acclCtxRelease(ctx);
  free(job);
  pthread_mutex_lock(&gNetMu);
}

static void* acclNetWorker(void* arg) {
  char* rx = (char*)arg;
  pthread_mutex_lock(&gNetMu);
  for (;;) {
    while (!g.readPending && !g.stop) pthread_cond_wait(&g.workCv, &gNetMu);
    if (!g.readPending) break;
    // Records queued now complete before this read starts, so it is their "after".
    g.reading = g.head;
    g.head = g.tail = NULL;
    g.nJobs = 0;
    if (acclNetUntrackedActive()) acclNetFlagPendingJobs();
    int settle = 0;
    for (struct acclNetJob* j = g.reading; j; j = j->next) settle |= j->settle;
    pthread_mutex_unlock(&gNetMu);
    if (settle) usleep(ACCL_NET_SETTLE_US);

    struct acclNetSnapshot* snap = acclNetReadAll(rx);
    if (snap) snap->readAttempts = 1;
    if (g.strict && snap) {
      uint64_t verificationStartUs = snap->tsStartUs;
      for (int attempt = 2; attempt <= 6; attempt++) {
        struct acclNetSnapshot* check = acclNetReadAll(rx);
        if (!check) break;
        int stable = acclNetSameHardware(snap, check);
        acclSnapUnref(snap);
        snap = check;
        snap->readAttempts = attempt;
        snap->tsStartUs = verificationStartUs;
        snap->stable = stable;
        if (stable) break;
      }
    }

    pthread_mutex_lock(&gNetMu);
    acclSnapUnref(g.cur);
    g.cur = snap;
    g.readPending = 0;
    // Records that completed during this read need the next one.
    if (!g.strict && g.head && g.nInflight == 0) acclNetRequestReadLocked();
    struct acclNetJob* jobs = g.reading;
    g.reading = NULL;
    pthread_cond_broadcast(&g.doneCv);
    while (jobs) {
      struct acclNetJob* job = jobs;
      jobs = job->next;
      acclNetWriteJobLocked(job, snap);
    }
  }
  pthread_mutex_unlock(&gNetMu);
  return rx;
}

// ============================================================================
// Chain bookkeeping (all under gNetMu)
// ============================================================================
static void acclNetRequestReadLocked(void) {
  if (!g.readPending) g.readSeq++;
  g.readPending = 1;
  pthread_cond_signal(&g.workCv);
}

// Untracked proxy ops still sending. Only net transports emit ProxyOp stop
// (src/transport/net.cc), so an op that never stops belongs to a transport
// without NIC traffic (shm/p2p memcpy, collnet): it stops counting after
// ACCL_NET_STUCK_US without a new one.
static int acclNetUntrackedActive(void) {
  return g.untrackedLive > 0 && acclNetNowUs() - g.untrackedLastUs < ACCL_NET_STUCK_US;
}

static void acclNetFlagInflight(void) {
  for (int i = 0; i < g.nInflight; i++) g.inflight[i]->overlapped = 1;
}

// A pending read will now also see new traffic.
static void acclNetFlagPendingJobs(void) {
  for (struct acclNetJob* j = g.reading; j; j = j->next) j->overlapped = 1;
  for (struct acclNetJob* j = g.head; j; j = j->next) j->overlapped = 1;
}

// A collective the plugin never sees complete (e.g. no KernelCh events) would
// otherwise mark every later record overlapped.
static void acclNetEvictStuckLocked(void) {
  if (g.strict) return;
  uint64_t now = acclNetNowUs();
  for (int i = 0; i < g.nInflight;) {
    struct acclNetCollState* cs = g.inflight[i];
    if (now - cs->startUs < ACCL_NET_STUCK_US) { i++; continue; }
    cs->overlapped = 1;
    g.unknownTraffic = 1;
    g.dirty = 1;
    g.inflight[i] = g.inflight[--g.nInflight];
  }
}

// Queued records whose window has closed must not wait for a later trigger.
static void acclNetKickLocked(void) {
  acclNetEvictStuckLocked();
  if (!g.strict && g.head && g.nInflight == 0 && !g.readPending) acclNetRequestReadLocked();
}

// This comm's collectives whose proxy ops are posted, i.e. from an earlier plan.
static int acclNetInflightOf(struct acclNetCommStats* st) {
  int n = 0;
  for (int i = 0; i < g.nInflight; i++) n += g.inflight[i]->owner == st && g.inflight[i]->posted;
  return n;
}

// Wait for the read that opens the next collective's window. Never waits while
// tracked collectives or untracked proxy ops are still running: that work is
// already overlapped, and waiting would serialize an application that does
// not serialize itself. A read that already timed out once is not waited for
// again. Returns 0 if the read is still pending.
static int acclNetWaitReadLocked(void) {
  acclNetKickLocked();
  if (g.nInflight > 0 || acclNetUntrackedActive()) return 1;
  if (!g.cur && !g.readPending) acclNetRequestReadLocked();
  if (g.readPending && g.readSeq == g.timedOutSeq) return 0;
  uint64_t deadline = acclNetNowUs() + g.waitUs;
  while (g.readPending && acclNetWaitUntil(deadline)) {}
  if (g.readPending) g.timedOutSeq = g.readSeq;
  return !g.readPending;
}

static void acclNetAdoptBoundaryLocked(struct acclNetCommStats* st, const struct rcclAcclBoundaryInfo* info) {
  int ok = info && info->version == RCCL_ACCL_BOUNDARY_VERSION;
  st->boundarySupported = ok && info->supported && info->synchronize;
  st->boundaryContext = ok ? info->context : NULL;
  st->synchronize = ok ? info->synchronize : NULL;
  st->boundarySucceeded = 0;
}

void acclNetBoundaryInfo(struct acclNetCommStats* st, const struct rcclAcclBoundaryInfo* info) {
  pthread_mutex_lock(&gNetMu);
  acclNetAdoptBoundaryLocked(st, info);
  pthread_mutex_unlock(&gNetMu);
}

void acclNetRegistered(struct acclNetCollState* cs) {
  pthread_mutex_lock(&gNetMu);
  cs->completionVerified = 1;
  pthread_mutex_unlock(&gNetMu);
}

// Strict mode: drain locally, establish global quiescence, take one shared
// after/before snapshot, then release only after all ranks finish reading.
// No GPU/transport thread waits here; the supported eager launch thread does.
static void acclNetStrictBoundary(struct acclNetCommStats* st) {
  pthread_mutex_lock(&gNetMu);
  uint64_t deadline = acclNetNowUs() + g.waitUs;
  while ((g.nInflight || acclNetUntrackedActive()) && acclNetWaitUntil(deadline)) {}
  int valid = g.nInflight == 0 && !acclNetUntrackedActive() && !g.unknownTraffic &&
              g.users == 1 && st->boundarySupported && !st->strictFailed;
  int (*sync)(void*, int) = st->synchronize;
  void* context = st->boundaryContext;
  pthread_mutex_unlock(&gNetMu);
  // Every rank must enter each exchange the others enter, so an unsupported
  // rank still calls in and votes invalid; skipping would hang its peers.
  if (sync) valid = sync(context, valid) == 0 && valid;
  else valid = 0;
  pthread_mutex_lock(&gNetMu);
  if (!valid) {
    st->strictFailed = 1;
    for (int i = 0; i < g.nInflight; i++) g.inflight[i]->unavailable = 1;
    for (struct acclNetJob* j = g.head; j; j = j->next) j->unavailable = 1;
    for (struct acclNetJob* j = g.reading; j; j = j->next) j->unavailable = 1;
  }
  acclNetRequestReadLocked();
  deadline = acclNetNowUs() + g.waitUs;
  while (g.readPending && acclNetWaitUntil(deadline)) {}
  int readOk = !g.readPending && g.cur && g.cur->stable;
  if (readOk) for (int i = 0; i < g.nDevs; i++) readOk &= g.cur->dev[i].rdmaOk;
  // A rank may release subsequent traffic after the failed coordination.
  // Invalidate jobs already being read before entering the release exchange;
  // a later Coll start cannot retroactively repair a record already written.
  if (!readOk) {
    for (struct acclNetJob* j = g.reading; j; j = j->next) j->unavailable = 1;
    for (struct acclNetJob* j = g.head; j; j = j->next) j->unavailable = 1;
  }
  pthread_mutex_unlock(&gNetMu);
  if (sync) readOk = sync(context, valid && readOk) == 0 && readOk;
  else readOk = 0;
  pthread_mutex_lock(&gNetMu);
  st->boundarySucceeded = valid && readOk;
  if (!st->boundarySucceeded) st->strictFailed = 1;
  pthread_mutex_unlock(&gNetMu);
}

void acclNetFinalBoundary(struct acclNetCommStats* st, const struct rcclAcclBoundaryInfo* info) {
  if (!g.strict || st->finalBoundaryDone) return;
  // The last launch may have been a P2P-only plan with no synchronize.
  if (info) {
    pthread_mutex_lock(&gNetMu);
    acclNetAdoptBoundaryLocked(st, info);
    pthread_mutex_unlock(&gNetMu);
  }
  acclNetStrictBoundary(st);
  st->finalBoundaryDone = 1;
}

// User thread, before the kernel launch: the only place that waits.
void acclNetKernelLaunch(struct acclNetCommStats* st) {
  if (g.strict) {
    uint64_t t0 = acclNetNowUs();
    acclNetStrictBoundary(st);
    pthread_mutex_lock(&gNetMu);
    st->launchWaitUs = acclNetNowUs() - t0;
    st->launchSeen = 1;
    pthread_mutex_unlock(&gNetMu);
    return;
  }
  pthread_mutex_lock(&gNetMu);
  uint64_t t0 = acclNetNowUs();
  // Grace for the previous collective's proxy tail. Once it times out this
  // comm is pipelining, so stop waiting until a launch finds nothing of its
  // own in flight.
  if (acclNetInflightOf(st) == 0) {
    st->graceOff = 0;
  } else if (!st->graceOff) {
    uint64_t deadline = acclNetNowUs() + g.graceUs;
    while (acclNetInflightOf(st) > 0 && acclNetWaitUntil(deadline)) {}
    if (acclNetInflightOf(st) > 0) st->graceOff = 1;
  }
  acclNetWaitReadLocked();
  st->launchWaitUs = acclNetNowUs() - t0;
  st->launchSeen = 1;
  pthread_mutex_unlock(&gNetMu);
}

// Never blocks: on the eager path the kernel is already running, and on the
// host-callback path this is a HIP callback.
void acclNetCollStart(struct acclNetCommStats* st, struct acclNetCollState* cs) {
  pthread_mutex_lock(&gNetMu);
  acclNetKickLocked();
  int readDone = !g.readPending;
  cs->launchWaitUs = st->launchSeen ? st->launchWaitUs : 0;
  cs->coordinated = g.strict && st->boundarySucceeded && !st->strictFailed && st->launchSeen;
  int launchSeen = st->launchSeen;
  st->launchSeen = 0;
  cs->tracked = 1;
  cs->owner = st;
  cs->startUs = acclNetNowUs();
  cs->before = acclSnapRef(g.cur);
  cs->unavailable = !readDone || !cs->before || !launchSeen || (g.strict && !cs->coordinated);
  // The pending read will land inside this collective.
  if (!readDone) acclNetFlagPendingJobs();
  if (g.dirty || g.unknownTraffic || g.nInflight > 0 || acclNetUntrackedActive()) {
    cs->overlapped = 1;
    acclNetFlagInflight();
  }
  g.dirty = 0;
  if (g.nInflight < ACCL_NET_MAX_INFLIGHT) {
    g.inflight[g.nInflight++] = cs;
  } else {
    // Untracked from here on: whatever runs next cannot be called clean.
    g.unknownTraffic = 1;
    cs->overlapped = 1;
    g.dirty = 1;
  }
  pthread_mutex_unlock(&gNetMu);
}

void acclNetCollPosted(struct acclNetCollState* cs) {
  pthread_mutex_lock(&gNetMu);
  cs->posted = 1;
  pthread_mutex_unlock(&gNetMu);
}

static void acclNetUntrackedLocked(struct acclNetCommStats* st) {
  g.dirty = 1;
  acclNetFlagInflight();
  if (g.readPending) acclNetFlagPendingJobs();
  // A KernelLaunch for untracked work must not be charged to the next coll.
  if (st) st->launchSeen = 0;
}

void acclNetUntrackedStart(struct acclNetCommStats* st) {
  pthread_mutex_lock(&gNetMu);
  acclNetUntrackedLocked(st);
  pthread_mutex_unlock(&gNetMu);
}

void acclNetUntrackedOpStart(struct acclNetCommStats* st, int late) {
  pthread_mutex_lock(&gNetMu);
  if (late) st->lateProxyOps++;
  g.untrackedLive++;
  g.untrackedLastUs = acclNetNowUs();
  acclNetUntrackedLocked(NULL);
  pthread_mutex_unlock(&gNetMu);
}

void acclNetUntrackedOpStop(void) {
  pthread_mutex_lock(&gNetMu);
  if (g.untrackedLive > 0) g.untrackedLive--;
  // Its traffic may have reached a window that is already being read.
  if (g.readPending) acclNetFlagPendingJobs();
  if (g.untrackedLive == 0) acclNetKickLocked();
  pthread_mutex_unlock(&gNetMu);
}

void acclNetCollFlag(struct acclNetCollState* cs) {
  pthread_mutex_lock(&gNetMu);
  if (cs->tracked) cs->overlapped = 1;
  acclNetUntrackedLocked(NULL);
  pthread_mutex_unlock(&gNetMu);
}

static void acclNetRemoveInflightLocked(struct acclNetCollState* cs) {
  for (int i = 0; i < g.nInflight; i++) {
    if (g.inflight[i] == cs) {
      g.inflight[i] = g.inflight[--g.nInflight];
      break;
    }
  }
  pthread_cond_broadcast(&g.doneCv);
}

int acclNetCollComplete(struct acclCommContext* ctx, struct acclNetCommStats* st,
                        struct acclNetCollState* cs, struct acclCompletedRecord* rec,
                        int nProxyOps) {
  if (!cs->tracked) return -1;
  struct acclNetJob* job = (struct acclNetJob*)malloc(sizeof(*job));
  pthread_mutex_lock(&gNetMu);
  acclNetRemoveInflightLocked(cs);
  if (!job || st->closed) {
    // Written without counters; the chain still has to move past its traffic.
    free(job);
    acclSnapUnref(cs->before);
    cs->before = NULL;
    if (g.nInflight == 0 && !g.readPending && !g.stop) acclNetRequestReadLocked();
    pthread_mutex_unlock(&gNetMu);
    return -1;
  }
  memcpy(&job->rec, rec, sizeof(*rec));
  job->next = NULL;
  job->ctx = ctx;
  job->st = st;
  job->before = cs->before;
  job->overlapped = cs->overlapped;
  // Unverified completion (RCCL without the protocol) is reported, not
  // invalidated: the report keeps such windows out of CLEAN.
  job->unavailable = cs->unavailable;
  job->completionVerified = cs->completionVerified;
  job->coordinated = cs->coordinated;
  job->startUs = cs->startUs;
  job->completeUs = acclNetNowUs();
  job->settle = !cs->completionVerified && nProxyOps == 0 && st->nNodes > 1;
  job->launchWaitUs = cs->launchWaitUs;
  cs->before = NULL;
  acclCtxRetain(ctx);
  if (g.tail) g.tail->next = job; else g.head = job;
  g.tail = job;
  g.nJobs++;
  st->queued++;
  if (!g.strict && (g.nInflight == 0 || g.nJobs > ACCL_NET_MAX_QUEUED)) acclNetRequestReadLocked();
  pthread_mutex_unlock(&gNetMu);
  return 0;
}

// A collective dropped without completing may still be sending.
void acclNetCollDrop(struct acclNetCollState* cs) {
  if (!cs->tracked) return;
  pthread_mutex_lock(&gNetMu);
  acclNetRemoveInflightLocked(cs);
  acclSnapUnref(cs->before);
  cs->before = NULL;
  cs->tracked = 0;
  acclNetUntrackedLocked(NULL);
  acclNetKickLocked();
  pthread_mutex_unlock(&gNetMu);
}

// ============================================================================
// Lifecycle
// ============================================================================
static uint64_t acclNetEnvU64(const char* name, uint64_t def, uint64_t max) {
  const char* v = getenv(name);
  if (!v || !v[0]) return def;
  char* end = NULL;
  errno = 0;
  unsigned long long x = strtoull(v, &end, 10);
  if (errno || end == v || *end || v[0] == '-' || x > max) {
    ACCL_NET_WARN("ACCL Profiler: %s=\"%s\" is not an integer in [0, %lu]; using %lu",
                  name, v, (unsigned long)max, (unsigned long)def);
    return def;
  }
  return x;
}

// rcclTelemetrySwCapture from the librccl that loaded us: logfn lives there, and
// RTLD_DEFAULT misses a librccl that was itself dlopen'd RTLD_LOCAL.
static void* acclNetFindSwCapture(void) {
  Dl_info info;
  if (gLogFn && dladdr((void*)gLogFn, &info) && info.dli_fname) {
    void* h = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if (h) {
      void* fn = dlsym(h, "rcclTelemetrySwCapture");
      dlclose(h);
      if (fn) return fn;
    }
  }
  return dlsym(RTLD_DEFAULT, "rcclTelemetrySwCapture");
}

// Parses ACCL_NET_DEVICES and applies ACCL_NET_DEVICE_SELECT. Returns device count.
static int acclNetParseDevices(int rank, int nRanks, int nNodes) {
  const char* env = getenv("ACCL_NET_DEVICES");
  if (!env || !env[0]) return 0;
  char names[ACCL_NET_MAX_DEVS][ACCL_NET_NAME_LEN];
  int n = 0;
  char buf[ACCL_NET_MAX_DEVS * ACCL_NET_NAME_LEN];
  snprintf(buf, sizeof(buf), "%s", env);
  char* save = NULL;
  for (char* tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    if (!acclNetNameOk(tok) || strlen(tok) >= ACCL_NET_NAME_LEN) {
      ACCL_NET_WARN("ACCL Profiler: ignoring bad device name \"%s\" in ACCL_NET_DEVICES", tok);
      continue;
    }
    if (n == ACCL_NET_MAX_DEVS) {
      ACCL_NET_WARN("ACCL Profiler: ACCL_NET_DEVICES has more than %d devices; using the first %d",
                    ACCL_NET_MAX_DEVS, ACCL_NET_MAX_DEVS);
      break;
    }
    strcpy(names[n++], tok);
  }
  if (n == 0) return 0;

  const char* sel = getenv("ACCL_NET_DEVICE_SELECT");
  if (sel && strcmp(sel, "local_rank") == 0) {
    if (nNodes > 0 && nRanks % nNodes == 0) {
      int local = rank % (nRanks / nNodes);
      if (local >= n) {
        ACCL_NET_WARN("ACCL Profiler: local rank %d has no entry in ACCL_NET_DEVICES (%d devices); "
                      "using %s", local, n, names[local % n]);
      }
      if (local % n) strcpy(names[0], names[local % n]);
      n = 1;
    } else {
      ACCL_NET_WARN("ACCL Profiler: ACCL_NET_DEVICE_SELECT=local_rank needs the same rank count on "
                    "every node (nRanks=%d nNodes=%d); reading every listed device", nRanks, nNodes);
    }
  } else if (sel && sel[0]) {
    ACCL_NET_WARN("ACCL Profiler: unknown ACCL_NET_DEVICE_SELECT=\"%s\"; reading every listed device", sel);
  }
  for (int i = 0; i < n; i++) {
    memset(&g.devs[i], 0, sizeof(g.devs[i]));
    strcpy(g.devs[i].name, names[i]);
    g.devs[i].nlFd = -1;
    g.devs[i].ethFd = -1;
  }
  return n;
}

static void acclNetCloseDevices(void) {
  for (int i = 0; i < g.nDevs; i++) {
    struct acclNetDev* d = &g.devs[i];
    if (d->nlFd >= 0) close(d->nlFd);
    if (d->ethFd >= 0) close(d->ethFd);
    free(d->ethBuf);
    memset(d, 0, sizeof(*d));
  }
  g.nDevs = 0;
}

int acclNetInit(struct acclNetCommStats* st, int rank, int nRanks, int nNodes) {
  const char* env = getenv("ACCL_NET_COUNTERS");
  if (!env || strcmp(env, "1") != 0) return 0;

  pthread_mutex_lock(&gNetMu);
  if (!g.condInit) {
    pthread_condattr_t attr;
    pthread_condattr_init(&attr);
    pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    pthread_cond_init(&g.doneCv, &attr);
    pthread_condattr_destroy(&attr);
    pthread_cond_init(&g.workCv, NULL);
    g.condInit = 1;
  }
  // The last comm's finalize is still tearing down the previous worker.
  while (g.stopping) pthread_cond_wait(&g.doneCv, &gNetMu);
  st->nNodes = nNodes;
  if (g.users > 0) {
    g.users++;
    st->enabled = 1;
    const char* sel = getenv("ACCL_NET_DEVICE_SELECT");
    const char* devs = getenv("ACCL_NET_DEVICES");
    if (sel && strcmp(sel, "local_rank") == 0 && devs && nNodes > 0 && nRanks % nNodes == 0) {
      ACCL_NET_INFO("ACCL Profiler: comm rank %d shares the net counter chain on %s chosen by the "
                    "first communicator", rank, g.selectedDev);
    }
    pthread_mutex_unlock(&gNetMu);
    return 1;
  }
  const char* strictEnv = getenv("ACCL_NET_STRICT");
  g.strict = strictEnv && strcmp(strictEnv, "1") == 0;
  g.unknownTraffic = 0;
  g.waitUs = acclNetEnvU64("ACCL_NET_WAIT_MS", 1000, 600000) * 1000;
  g.graceUs = acclNetEnvU64("ACCL_NET_GRACE_US", 1000, 1000000);
  g.nDevs = acclNetParseDevices(rank, nRanks, nNodes);
  snprintf(g.selectedDev, sizeof(g.selectedDev), "%s", g.nDevs ? g.devs[0].name : "");
  if (g.nDevs == 0 && !gAcclNetHooks.snapshotFn) {
    ACCL_NET_WARN("ACCL Profiler: ACCL_NET_COUNTERS=1 but ACCL_NET_DEVICES lists no usable device; "
                  "net counters are off");
    pthread_mutex_unlock(&gNetMu);
    return 0;
  }
  char* rx = (char*)malloc(ACCL_NL_BUF_SIZE);
  if (!rx) {
    acclNetCloseDevices();
    pthread_mutex_unlock(&gNetMu);
    return 0;
  }
  // One-time probe under the lock: no worker or other comm is active yet.
  for (int i = 0; i < g.nDevs && !gAcclNetHooks.snapshotFn; i++) {
    struct acclNetDev* d = &g.devs[i];
    uint64_t probe[ACCL_NET_MAX_RDMA];
    // The first read also learns the counter names.
    if (acclRdmaOpen(d, rx) != 0 || acclRdmaRead(d, rx, probe, 1) != 0) {
      ACCL_NET_WARN("ACCL Profiler: cannot read RDMA netlink counters for %s port 1; its records "
                    "will be marked unavailable", d->name);
    }
    acclEthOpen(d);
  }
  g.swCapture = (int (*)(RcclTelemetrySwSnapshot*, int))acclNetFindSwCapture();
  g.stop = 0;
  g.dirty = 0;
  g.untrackedLive = 0;
  g.nInflight = 0;
  g.head = g.tail = g.reading = NULL;
  g.nJobs = 0;
  g.readPending = 0;
  g.readSeq = g.timedOutSeq = 0;
  if (pthread_create(&g.thread, NULL, acclNetWorker, rx) != 0) {
    ACCL_NET_WARN("ACCL Profiler: cannot start the net counter thread; net counters are off");
    free(rx);
    acclNetCloseDevices();
    pthread_mutex_unlock(&gNetMu);
    return 0;
  }
  g.users = 1;
  st->enabled = 1;
  ACCL_NET_INFO("ACCL Profiler: net counters on, %d device(s) %s%s, wait=%lums grace=%luus",
                g.nDevs, g.nDevs ? g.devs[0].name : "", g.nDevs > 1 ? ",..." : "",
                (unsigned long)(g.waitUs / 1000), (unsigned long)g.graceUs);
  pthread_mutex_unlock(&gNetMu);
  return 1;
}

// Called with gNetMu held, before the devices are closed.
static char* acclNetSummaryJsonLocked(struct acclNetCommStats* st, int lost) {
  struct acclStr js = {NULL, 0, 0, 0};
  acclStrAppend(&js, ",\"net_counters\":{\"records\":%lu,\"overlapped\":%lu,\"unavailable\":%lu,"
                "\"records_with_read_error\":%lu,\"queued_at_finalize\":%d,"
                "\"late_proxy_ops\":%lu,\"reads\":%lu,"
                "\"read_us_mean\":%.1f,\"read_us_max\":%lu,\"totals\":{",
                (unsigned long)st->records, (unsigned long)st->overlapped,
                (unsigned long)st->unavailable, (unsigned long)st->readErrors, lost,
                (unsigned long)st->lateProxyOps, (unsigned long)st->reads,
                st->reads ? (double)st->readUsSum / (double)st->reads : 0.0,
                (unsigned long)st->readUsMax);
  for (int i = 0; i < g.nDevs; i++) {
    acclStrAppend(&js, "%s\"%s\":{", i ? "," : "", g.devs[i].name);
    int first = 1;
    for (int k = 0; k < g.devs[i].nRdma; k++) {
      if (st->totals[i][k] == 0 && !acclNetIsVolume(g.devs[i].rdmaNames[k])) continue;
      acclStrAppend(&js, "%s\"%s\":%lu", first ? "" : ",", g.devs[i].rdmaNames[k],
                    (unsigned long)st->totals[i][k]);
      first = 0;
    }
    acclStrAppend(&js, "}");
  }
  acclStrAppend(&js, "}}");
  if (js.oom) {
    free(js.p);
    return NULL;
  }
  return js.p;
}

char* acclNetFinalize(struct acclNetCommStats* st, int* lost) {
  *lost = 0;
  if (!st->enabled) return NULL;
  pthread_mutex_lock(&gNetMu);
  // Write this comm's queued records before its summary line. Re-kick on every
  // wakeup: another comm's collective may have held the window open.
  if (g.strict && !st->finalBoundaryDone) {
    for (struct acclNetJob* j = g.head; j; j = j->next) if (j->st == st) j->unavailable = 1;
  }
  uint64_t deadline = acclNetNowUs() + 2 * g.waitUs + 500000;
  while (st->queued > 0) {
    if (g.head && !g.readPending) acclNetRequestReadLocked();
    if (!acclNetWaitUntil(deadline)) break;
  }
  int last = --g.users == 0;
  pthread_t worker = g.thread;
  if (last) {
    g.stop = 1;
    g.stopping = 1;
    // No new jobs once the worker is going away.
    st->closed = 1;
    if (g.head) acclNetRequestReadLocked();
    pthread_cond_signal(&g.workCv);
    pthread_mutex_unlock(&gNetMu);
    void* rx = NULL;
    pthread_join(worker, &rx);
    free(rx);
    pthread_mutex_lock(&gNetMu);
    // Queued after the worker's last read: write them without counters.
    struct acclNetJob* left = g.head;
    g.head = g.tail = NULL;
    g.nJobs = 0;
    while (left) {
      struct acclNetJob* job = left;
      left = job->next;
      if (job->st == st) (*lost)++;
      acclNetWriteJobLocked(job, NULL);
    }
  }
  *lost += st->queued;
  if (*lost > 0) {
    ACCL_NET_WARN("ACCL Profiler: %d net-counter record(s) not written before the summary", *lost);
  }
  char* summary = acclNetSummaryJsonLocked(st, *lost);
  st->closed = 1;
  if (last) {
    acclSnapUnref(g.cur);
    g.cur = NULL;
    g.readPending = 0;
    acclNetCloseDevices();
    g.stopping = 0;
    pthread_cond_broadcast(&g.doneCv);
  }
  pthread_mutex_unlock(&gNetMu);
  return summary;
}
