/*************************************************************************
 * RCCL Bootstrap Tier-2 Deep Profiling — implementation
 *************************************************************************/

#include "bootstrap_trace.h"
#include "param.h"
#include "debug.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
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
        mkdir(s_outDir, 0755);
        INFO(NCCL_BOOTSTRAP, "BootstrapTrace: enabled, log-dump + binary dump dir=%s", s_outDir);
      } else {
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
  ensureGlobalInit();
  return s_enabled;
}

const char* outputDir() {
  ensureGlobalInit();
  return s_outDir;
}

static thread_local PerThreadBuffer* tl_buf = nullptr;

PerThreadBuffer* getBuffer() {
  if (tl_buf == nullptr && isEnabled()) {
    tl_buf = new PerThreadBuffer();
    tl_buf->idx = 0;
    tl_buf->rank = -1;
    tl_buf->isRootThread = 0;
  }
  return tl_buf;
}

void initThreadBuffer(int rank, int isRootThread) {
  if (!isEnabled()) return;
  PerThreadBuffer* b = getBuffer();
  if (!b) return;
  b->rank = rank;
  b->isRootThread = isRootThread;
  b->idx = 0;
}

void recordEvent(uint16_t phase, uint16_t md, uint64_t startNs, uint32_t bytes) {
  if (!isEnabled()) return;
  PerThreadBuffer* b = getBuffer();
  if (!b) return;
  if (b->idx >= RING_BUFFER_SIZE) return;
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
  if (b->idx >= RING_BUFFER_SIZE) return;
  Event& e = b->events[b->idx++];
  e.t_ns   = nowNs();
  e.rank   = (uint32_t)b->rank;
  e.phase  = phase;
  e.md     = md;
  e.dur_us = 0;
  e.bytes  = 0;
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
  INFO(NCCL_BOOTSTRAP, "BTRACE dump begin rank=%d root=%d events=%d",
       b->rank, b->isRootThread, b->idx);
  for (int i = 0; i < b->idx; ++i) {
    const Event& e = b->events[i];
    INFO(NCCL_BOOTSTRAP,
         "BTRACE rank=%d root=%d p=%u name=%s t_ns=%llu dur_us=%u md=%u bytes=%u",
         b->rank, b->isRootThread, (unsigned)e.phase, phaseName(e.phase),
         (unsigned long long)e.t_ns, e.dur_us, (unsigned)e.md, e.bytes);
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
  FILE* f = fopen(path, "wb");
  if (!f) {
    WARN("BootstrapTrace: cannot open %s for write", path);
    return;
  }
  // Header: magic(4), version(4), rank(4), isRoot(4), count(4), reserved(4)
  uint32_t hdr[6] = {
      0xB007F00D,
      1u,
      (uint32_t)b->rank,
      (uint32_t)b->isRootThread,
      (uint32_t)b->idx,
      0u};
  fwrite(hdr, sizeof(hdr), 1, f);
  fwrite(b->events, sizeof(Event), b->idx, f);
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

  // Reset idx after dump to avoid double-write if init is re-entered.
  b->idx = 0;
}

}  // namespace ncclBootstrapTrace
