/*************************************************************************
 * RCCL Bootstrap Tier-2 Deep Profiling — implementation
 *************************************************************************/

#include "bootstrap_trace.h"
#include "param.h"
#include "debug.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ncclBootstrapTrace {

NCCL_PARAM(BootstrapTrace, "BOOTSTRAP_TRACE", 0);

static std::atomic<int> s_initState{0};        // 0=uninit, 1=initing, 2=ready
static bool s_enabled = false;
static char s_outDir[256] = {0};               // empty => no binary dump

static void ensureGlobalInit() {
  int s = s_initState.load(std::memory_order_acquire);
  if (s == 2) return;
  int expected = 0;
  if (s_initState.compare_exchange_strong(expected, 1)) {
    s_enabled = ncclParamBootstrapTrace() != 0;
    // Binary dump is opt-in: only when NCCL_BOOTSTRAP_TRACE_DIR is explicitly
    // set. Default behaviour is to accumulate in-memory and flush via INFO()
    // into the user's existing NCCL log (stderr or NCCL_DEBUG_FILE) — no /tmp
    // pollution, no separate files to chase down.
    const char* env = getenv("NCCL_BOOTSTRAP_TRACE_DIR");
    if (env && env[0]) {
      snprintf(s_outDir, sizeof(s_outDir), "%s", env);
    } else {
      s_outDir[0] = '\0';
    }
    if (s_enabled) {
      if (s_outDir[0]) {
        // rec_26 M3: a discarded mkdir() makes the binary-dump path look
        // enabled even when the directory can't be created. Check it; on
        // failure (other than EEXIST) clear s_outDir so we skip binary dumps
        // and warn the user instead of silently dropping every fopen().
        if (mkdir(s_outDir, 0755) != 0 && errno != EEXIST) {
          WARN("BootstrapTrace: cannot create dump dir %s (%s); binary dump disabled, log-dump only",
               s_outDir, strerror(errno));
          s_outDir[0] = '\0';
        } else {
          INFO(NCCL_BOOTSTRAP, "BootstrapTrace: enabled, log-dump + binary dump dir=%s", s_outDir);
        }
      }
      if (!s_outDir[0]) {
        INFO(NCCL_BOOTSTRAP, "BootstrapTrace: enabled, log-dump only (set NCCL_BOOTSTRAP_TRACE_DIR for binary dump)");
      }
    }
    s_initState.store(2, std::memory_order_release);
  } else {
    while (s_initState.load(std::memory_order_acquire) != 2) {
      sched_yield();
    }
  }
}

bool isEnabled() {
  // rec_26 M4: fast path on hot callers (kernel launch). Once initialised this
  // is a single acquire atomic load + predicted-not-taken branch; no spin.
  // Acquire (not relaxed) is required so that observing state==2 also makes the
  // s_enabled store visible on weak-memory architectures (paired with the
  // release store in ensureGlobalInit).
  if (__builtin_expect(s_initState.load(std::memory_order_acquire) == 2, 1))
    return s_enabled;
  ensureGlobalInit();
  return s_enabled;
}

const char* outputDir() {
  ensureGlobalInit();
  return s_outDir;
}

// rec_26 C1: a bare `thread_local PerThreadBuffer*` allocated with new is never
// freed, leaking 216 KB per thread in worker-pool / thread-cycling frameworks.
// A thread_local unique_ptr ties the buffer lifetime to thread teardown.
static thread_local std::unique_ptr<PerThreadBuffer> tl_buf;

PerThreadBuffer* getBuffer() {
  if (!tl_buf && isEnabled()) {
    tl_buf.reset(new PerThreadBuffer());
    tl_buf->idx = 0;
    tl_buf->netIdx = 0;
    tl_buf->rank = -1;
    tl_buf->isRootThread = 0;
    tl_buf->eventOverflow = 0;
    tl_buf->netOverflow = 0;
  }
  return tl_buf.get();
}

void initThreadBuffer(int rank, int isRootThread) {
  if (!isEnabled()) return;
  PerThreadBuffer* b = getBuffer();
  if (!b) return;
  b->rank = rank;
  b->isRootThread = isRootThread;
}

// rec_26 M2: warn exactly once (process-wide) when a ring fills, so a truncated
// trace is never mistaken for a complete one. Per-thread overflow counts are
// also written to the binary dump header.
static std::atomic<int> s_overflowWarned{0};
static void warnOverflowOnce() {
  int expected = 0;
  if (s_overflowWarned.compare_exchange_strong(expected, 1))
    WARN("BootstrapTrace: ring buffer full, events dropped (per-thread overflow counts in dump header)");
}

void recordEvent(uint16_t phase, uint16_t md, uint64_t startNs, uint32_t bytes) {
  if (!isEnabled()) return;
  PerThreadBuffer* b = getBuffer();
  if (!b) return;
  if (b->idx >= RING_BUFFER_SIZE) { b->eventOverflow++; warnOverflowOnce(); return; }
  uint64_t now = nowNs();
  Event& e = b->events[b->idx++];
  e.t_ns   = startNs;
  e.rank   = (uint32_t)b->rank;
  e.phase  = phase;
  e.md     = md;
  e.dur_us = (uint32_t)((now > startNs ? now - startNs : 0ULL) / 1000ULL);
  e.bytes  = bytes;
}

void recordInstant(uint16_t phase, uint16_t md) {
  if (!isEnabled()) return;
  PerThreadBuffer* b = getBuffer();
  if (!b) return;
  if (b->idx >= RING_BUFFER_SIZE) { b->eventOverflow++; warnOverflowOnce(); return; }
  Event& e = b->events[b->idx++];
  e.t_ns   = nowNs();
  e.rank   = (uint32_t)b->rank;
  e.phase  = phase;
  e.md     = md;
  e.dur_us = 0;
  e.bytes  = 0;
}

void recordNetStat(uint16_t phase, uint16_t md, int fd) {
  if (!isEnabled() || fd < 0) return;
  PerThreadBuffer* b = getBuffer();
  if (!b) return;
  struct tcp_info ti;
  socklen_t len = sizeof(ti);
  memset(&ti, 0, sizeof(ti));
  // Non-TCP sockets / closed fds simply yield no sample.
  if (getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &len) != 0) return;
  if (b->netIdx >= NET_SAMPLE_SIZE) { b->netOverflow++; warnOverflowOnce(); return; }
  NetSample& s = b->netSamples[b->netIdx++];
  s.t_ns          = nowNs();
  s.rank          = (uint32_t)b->rank;
  s.phase         = phase;
  s.md            = md;
  s.rtt_us        = ti.tcpi_rtt;
  s.rttvar_us     = ti.tcpi_rttvar;
  s.rto_us        = ti.tcpi_rto;
  s.snd_cwnd      = ti.tcpi_snd_cwnd;
  s.total_retrans = ti.tcpi_total_retrans;
  s.lost          = ti.tcpi_lost;
  s.unacked       = ti.tcpi_unacked;
  s.ca_state      = ti.tcpi_ca_state;
  s.reserved      = 0;
}

static const char* phaseName(uint16_t p) {
  switch (p) {
    case PHASE_INIT_TOTAL:        return "init.total";
    case PHASE_LISTEN_FWD:        return "listen.fwd";
    case PHASE_LISTEN_REV:        return "listen.rev";
    case PHASE_LISTEN_ROOT:       return "listen.root";
    case PHASE_SEND_TO_ROOT_CONN: return "send_to_root.conn";
    case PHASE_SEND_TO_ROOT_DATA: return "send_to_root.data";
    case PHASE_RECV_FROM_ROOT:    return "recv_from_root";
    case PHASE_FORWARD_CONNECT:   return "forward_connect";
    case PHASE_REVERSE_CONNECT:   return "reverse_connect";
    case PHASE_PROXY_LISTEN:      return "proxy_listen";
    case PHASE_PEER_LISTEN:       return "peer_listen";
    case PHASE_RAS_INIT:          return "ras_init";
    case PHASE_RING_ALLGATHER:    return "ring_allgather";
    case PHASE_RING_STEP:         return "ring_step";
    case PHASE_PROXY_INIT:        return "proxy_init";
    case PHASE_DEPLOY_COMM_INIT_TOTAL:      return "deploy.comm_init.total";
    case PHASE_DEPLOY_HIP_CTX:              return "deploy.hip_ctx";
    case PHASE_DEPLOY_KERNEL_LOAD:          return "deploy.kernel_load";
    case PHASE_DEPLOY_COMM_SPLIT_ALLGATHER: return "deploy.comm_split_allgather";
    case PHASE_DEPLOY_BOOTSTRAP:            return "deploy.bootstrap";
    case PHASE_DEPLOY_ALLGATHER_PEER:       return "deploy.allgather.peer";
    case PHASE_DEPLOY_TOPO_DETECT:          return "deploy.topo.detect";
    case PHASE_DEPLOY_TOPO_PATHS:           return "deploy.topo.paths";
    case PHASE_DEPLOY_GRAPH_SEARCH:         return "deploy.graph_search";
    case PHASE_DEPLOY_ALLGATHER3:           return "deploy.allgather3";
    case PHASE_DEPLOY_TOPO_POSTSET:         return "deploy.topo.postset";
    case PHASE_DEPLOY_BUFFERS:              return "deploy.buffers";
    case PHASE_DEPLOY_PROXY_CREATE:         return "deploy.proxy_create";
    case PHASE_DEPLOY_TRANSPORT_CONNECT:    return "deploy.transport_connect";
    case PHASE_DEPLOY_PROXY_CONNECT:        return "deploy.proxy_connect";
    case PHASE_DEPLOY_TUNER_LOAD:           return "deploy.tuner_load";
    case PHASE_DEPLOY_DEV_COMM_SETUP:       return "deploy.dev_comm_setup";
    case PHASE_DEPLOY_INTRANODE_BARRIER:    return "deploy.intranode_barrier";
    case PHASE_DEPLOY_KERNEL_LAUNCH:        return "deploy.kernel_launch";
    case PHASE_ROOT_TOTAL:        return "root.total";
    case PHASE_ROOT_WAIT_FIRST:   return "root.wait_first";
    case PHASE_ROOT_ACCEPT:       return "root.accept";
    case PHASE_ROOT_RECV_INFO:    return "root.recv_info";
    case PHASE_ROOT_INLINE_SEND:  return "root.inline_send";
    case PHASE_ROOT_FINAL_SEND:   return "root.final_send";
    case PHASE_TCP_CONNECT:       return "tcp.connect";
    case PHASE_TCP_ACCEPT:        return "tcp.accept";
    case PHASE_TCP_READY:         return "tcp.ready";
    default:                      return "unknown";
  }
}

// Compact text dump into the user's NCCL log. Format per event line:
//   BTRACE rank=R root=0|1 p=PID name=NAME t_ns=ABS dur_us=DUR md=MD bytes=B
// Grep with `grep BTRACE` on the user's NCCL log; one INFO call per line keeps
// each event on its own logger line for easy filtering.
static void logDump(PerThreadBuffer* b) {
  INFO(NCCL_BOOTSTRAP, "BTRACE dump begin rank=%d root=%d events=%d net=%d ev_overflow=%d net_overflow=%d",
       b->rank, b->isRootThread, b->idx, b->netIdx, b->eventOverflow, b->netOverflow);
  for (int i = 0; i < b->idx; ++i) {
    const Event& e = b->events[i];
    INFO(NCCL_BOOTSTRAP,
         "BTRACE rank=%d root=%d p=%u name=%s t_ns=%llu dur_us=%u md=%u bytes=%u",
         b->rank, b->isRootThread, (unsigned)e.phase, phaseName(e.phase),
         (unsigned long long)e.t_ns, e.dur_us, (unsigned)e.md, e.bytes);
  }
  // TCP_INFO samples on their own grep-able line prefix (BNETSTAT).
  for (int i = 0; i < b->netIdx; ++i) {
    const NetSample& s = b->netSamples[i];
    INFO(NCCL_BOOTSTRAP,
         "BNETSTAT rank=%d root=%d p=%u name=%s t_ns=%llu md=%u rtt_us=%u rttvar_us=%u rto_us=%u cwnd=%u retrans=%u lost=%u unacked=%u ca_state=%u",
         b->rank, b->isRootThread, (unsigned)s.phase, phaseName(s.phase),
         (unsigned long long)s.t_ns, (unsigned)s.md, s.rtt_us, s.rttvar_us, s.rto_us,
         s.snd_cwnd, s.total_retrans, s.lost, s.unacked, (unsigned)s.ca_state);
  }
  INFO(NCCL_BOOTSTRAP, "BTRACE dump end rank=%d root=%d", b->rank, b->isRootThread);
}

static void binaryDump(PerThreadBuffer* b) {
  char path[600];
  if (b->isRootThread) {
    snprintf(path, sizeof(path), "%s/root_pid%d_tid%lu.bin",
             s_outDir, (int)getpid(), (unsigned long)pthread_self());
  } else {
    snprintf(path, sizeof(path), "%s/rank%05d_pid%d.bin",
             s_outDir, b->rank, (int)getpid());
  }
  // Append, not truncate: dumpThreadBuffer() is called multiple times per
  // thread across the deployment timeline (bootstrap end, deploy init end,
  // kernel launch). Each call flushes and resets the ring, so truncating here
  // would keep only the last batch and silently drop the bootstrap phases and
  // net samples. Appending writes one self-describing [header|events|nets]
  // segment per dump; the reader iterates segments to EOF. (pid is in the
  // filename, so a fresh process never appends to a previous run's file.)
  // rec_27 N4: today there are O(3) flushes/process so the file stays small; if
  // dumps are ever added per-iteration, cap the count or rotate the file.
  FILE* f = fopen(path, "ab");
  if (!f) {
    WARN("BootstrapTrace: cannot open %s for write", path);
    return;
  }
  // Header (all uint32, native byte order):
  //   [0] magic        0xB007F00D
  //   [1] version      FORMAT_VERSION
  //   [2] endian_mark  0x01020304  (rec_26 C2: reader detects byte order and
  //                                 swaps if it reads it as 0x04030201)
  //   [3] rank
  //   [4] isRoot
  //   [5] eventCount
  //   [6] netCount
  //   [7] eventOverflow
  //   [8] netOverflow
  //   [9] reserved
  // Followed by eventCount Event records, then netCount NetSample records.
  uint32_t hdr[10] = {
      0xB007F00Du,
      FORMAT_VERSION,
      0x01020304u,
      (uint32_t)b->rank,
      (uint32_t)b->isRootThread,
      (uint32_t)b->idx,
      (uint32_t)b->netIdx,
      (uint32_t)b->eventOverflow,
      (uint32_t)b->netOverflow,
      0u};
  fwrite(hdr, sizeof(hdr), 1, f);
  fwrite(b->events, sizeof(Event), b->idx, f);
  fwrite(b->netSamples, sizeof(NetSample), b->netIdx, f);
  fclose(f);
}

void dumpThreadBuffer() {
  if (!isEnabled()) return;
  PerThreadBuffer* b = getBuffer();
  if (!b || b->idx == 0) return;

  // Always emit text dump into user's NCCL log.
  logDump(b);

  // Optional binary dump when NCCL_BOOTSTRAP_TRACE_DIR was set.
  if (s_outDir[0]) binaryDump(b);

  // Reset after dump to avoid double-write if init is re-entered.
  b->idx = 0;
  b->netIdx = 0;
}

}  // namespace ncclBootstrapTrace
