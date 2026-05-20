/*************************************************************************
 * RCCL Bootstrap Tier-2 Deep Profiling
 *
 * Per-thread in-memory event ring buffer. On bootstrap exit each thread's
 * buffer is flushed to the same NCCL log stream the user is already using
 * (stderr or NCCL_DEBUG_FILE) via INFO(NCCL_BOOTSTRAP, ...) — no separate
 * files, no /tmp directory by default. Binary dump remains available as an
 * opt-in for offline processing with scripts/merge_bootstrap_trace.py.
 *
 * Activation:
 *   NCCL_BOOTSTRAP_TRACE=1                       # accumulate + log-dump
 *   NCCL_BOOTSTRAP_TRACE_DIR=/path/to/dump_dir   # ALSO write binary dump
 *
 * No measurable overhead when disabled (single bool check + early return).
 *************************************************************************/
#ifndef BOOTSTRAP_TRACE_H_
#define BOOTSTRAP_TRACE_H_

#include <cstdint>
#include <ctime>

namespace ncclBootstrapTrace {

// Phase IDs. Keep stable: post-process scripts depend on these names.
enum Phase : uint16_t {
  PHASE_INIT_TOTAL          = 0,
  PHASE_LISTEN_FWD          = 1,
  PHASE_LISTEN_REV          = 2,
  PHASE_LISTEN_ROOT         = 3,
  PHASE_SEND_TO_ROOT_CONN   = 4,  // md = root index
  PHASE_SEND_TO_ROOT_DATA   = 5,  // md = root index, bytes = sizeof(extInfo)
  PHASE_RECV_FROM_ROOT      = 6,
  PHASE_FORWARD_CONNECT     = 7,  // socketRingConnectStart..Finish
  PHASE_REVERSE_CONNECT     = 8,  // bootstrapBidirRingSetupStart..Finish
  PHASE_PROXY_LISTEN        = 9,
  PHASE_PEER_LISTEN         = 10,
  PHASE_RAS_INIT            = 11,
  PHASE_RING_ALLGATHER      = 12, // ringAllInfo total
  PHASE_RING_STEP           = 13, // md = step index, bytes = bytes/dir
  PHASE_PROXY_INIT          = 14,
  // Root-thread events
  PHASE_ROOT_TOTAL          = 100,
  PHASE_ROOT_WAIT_FIRST     = 101,
  PHASE_ROOT_ACCEPT         = 102, // md = peer rank
  PHASE_ROOT_RECV_INFO      = 103, // md = peer rank, bytes = sizeof(extInfo)
  PHASE_ROOT_INLINE_SEND    = 104, // md = peer rank
  PHASE_ROOT_FINAL_SEND     = 105, // md = peer rank
  // TCP socket lifecycle (instrumented in bootstrap.cc, not socket.cc)
  PHASE_TCP_CONNECT         = 200, // md = peer rank
  PHASE_TCP_ACCEPT          = 201, // md = peer rank
  PHASE_TCP_READY           = 202, // md = peer rank, dur = time from start to ready
  PHASE_COUNT
};

struct Event {
  uint64_t t_ns;     // CLOCK_MONOTONIC_RAW absolute ns
  uint32_t rank;     // global rank (or root pid for root thread)
  uint16_t phase;    // Phase enum
  uint16_t md;       // metadata: step index, peer rank, etc.
  uint32_t dur_us;   // duration in microseconds (0 for instant)
  uint32_t bytes;    // payload bytes (0 if N/A)
};
static_assert(sizeof(Event) == 24, "Event must be packed to 24B");

constexpr int RING_BUFFER_SIZE = 8192;  // 192 KB per thread

struct PerThreadBuffer {
  Event events[RING_BUFFER_SIZE];
  int idx;
  int rank;
  int isRootThread;  // 0 = main rank thread, 1 = bootstrapRoot thread
};

// Global state
bool isEnabled();
const char* outputDir();  // "" when no binary dump requested

// Per-thread buffer access
PerThreadBuffer* getBuffer();

// Initialise current thread's buffer (sets rank/role).
void initThreadBuffer(int rank, int isRootThread);

// Raw event recording.
void recordEvent(uint16_t phase, uint16_t md, uint64_t startNs, uint32_t bytes);
void recordInstant(uint16_t phase, uint16_t md);

// Flush current thread's buffer. Always emits a text dump via NCCL INFO into
// the user's RCCL log; additionally writes a .bin dump if outputDir() is set.
// Idempotent.
void dumpThreadBuffer();

inline uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

}  // namespace ncclBootstrapTrace

// Convenience macros. All no-op (compile-time) when trace is disabled at runtime.
// BTRACE_BEGIN declares the timer without initializer to permit `goto label`
// to jump over it (C++17 forbids jumping over a scalar with initializer); the
// assignment is a separate statement.
#define BTRACE_BEGIN(name) \
  uint64_t name; \
  name = ncclBootstrapTrace::isEnabled() ? ncclBootstrapTrace::nowNs() : 0

#define BTRACE_END(phase, md, name, bytes) \
  do { \
    if (ncclBootstrapTrace::isEnabled() && (name)) \
      ncclBootstrapTrace::recordEvent((phase), (md), (name), (bytes)); \
  } while (0)

#define BTRACE_INSTANT(phase, md) \
  do { \
    if (ncclBootstrapTrace::isEnabled()) \
      ncclBootstrapTrace::recordInstant((phase), (md)); \
  } while (0)

#endif  // BOOTSTRAP_TRACE_H_
