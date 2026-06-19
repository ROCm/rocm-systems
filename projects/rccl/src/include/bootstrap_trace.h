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
 * Overhead when disabled: one relaxed atomic load on the fast path of
 * isEnabled() (branch-predicted false). Not literally zero, but negligible;
 * callers on hot paths (e.g. kernel launch) still gate every record behind
 * this single predicted-not-taken branch.
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

  // Deployment-wide ncclCommInitRank events (instrumented outside bootstrap.cc).
  PHASE_DEPLOY_COMM_INIT_TOTAL      = 300,
  PHASE_DEPLOY_HIP_CTX              = 301,
  PHASE_DEPLOY_KERNEL_LOAD          = 302,
  PHASE_DEPLOY_COMM_SPLIT_ALLGATHER = 303,
  PHASE_DEPLOY_BOOTSTRAP            = 304,
  PHASE_DEPLOY_ALLGATHER_PEER       = 305,
  PHASE_DEPLOY_TOPO_DETECT          = 306,
  PHASE_DEPLOY_TOPO_PATHS           = 307,
  PHASE_DEPLOY_GRAPH_SEARCH         = 308, // md = graph id
  PHASE_DEPLOY_ALLGATHER3           = 309,
  PHASE_DEPLOY_TOPO_POSTSET         = 310,
  PHASE_DEPLOY_BUFFERS              = 311,
  PHASE_DEPLOY_PROXY_CREATE         = 312,
  PHASE_DEPLOY_TRANSPORT_CONNECT    = 313,
  PHASE_DEPLOY_PROXY_CONNECT        = 314,
  PHASE_DEPLOY_TUNER_LOAD           = 315,
  PHASE_DEPLOY_DEV_COMM_SETUP       = 316,
  PHASE_DEPLOY_INTRANODE_BARRIER    = 317,
  PHASE_DEPLOY_KERNEL_LAUNCH        = 318,
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
  PHASE_COUNT               = 319
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

// Kernel TCP_INFO sample, captured from a connected socket fd at the end of a
// network phase. Lets offline analysis attribute bootstrap latency variance to
// transport behaviour (RTT spikes, retransmits, congestion-control state) vs.
// CPU/scheduling. Recorded into a separate, smaller ring so the common Event
// hot path stays 24B and untouched.
struct NetSample {
  uint64_t t_ns;          // capture time (CLOCK_MONOTONIC_RAW abs ns)
  uint32_t rank;          // global rank
  uint16_t phase;         // Phase this sample is attached to
  uint16_t md;            // peer rank (or 0)
  uint32_t rtt_us;        // tcpi_rtt (smoothed RTT, microseconds)
  uint32_t rttvar_us;     // tcpi_rttvar
  uint32_t rto_us;        // tcpi_rto (retransmission timeout)
  uint32_t snd_cwnd;      // tcpi_snd_cwnd (segments)
  uint32_t total_retrans; // tcpi_total_retrans (cumulative on this connection)
  uint32_t lost;          // tcpi_lost
  uint32_t unacked;       // tcpi_unacked
  uint16_t ca_state;      // tcpi_ca_state (kernel uint8; widened to u16 here for
                          // 8-byte struct alignment) (0=Open, !=0 => loss/recovery)
  uint16_t reserved;      // explicit pad to a 48B (8-byte multiple) record
};
static_assert(sizeof(NetSample) == 48, "NetSample must be packed to 48B");

constexpr int RING_BUFFER_SIZE = 8192;     // 192 KB per thread (Event)
constexpr int NET_SAMPLE_SIZE  = 512;      // 24 KB per thread (NetSample)

// Binary dump format version. Bump on ANY change to Event/NetSample layout or
// header so the Python reader rejects stale/mismatched files (rec_26 m3).
constexpr uint32_t FORMAT_VERSION = 2;

struct PerThreadBuffer {
  Event events[RING_BUFFER_SIZE];
  NetSample netSamples[NET_SAMPLE_SIZE];
  int idx;
  int netIdx;
  // Root thread uses rank = -1; stored into the uint32 Event/NetSample.rank as
  // the sentinel 0xFFFFFFFF, which post-processors render as the ROOT track.
  int rank;
  int isRootThread;     // 0 = main rank thread, 1 = bootstrapRoot thread
  int eventOverflow;    // count of Event records dropped after ring filled
  int netOverflow;      // count of NetSample records dropped after ring filled
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

// Capture kernel TCP_INFO from a connected TCP socket fd and store as a
// NetSample attached to `phase` (md = peer rank). No-op if fd < 0, the trace
// is disabled, or the getsockopt fails (e.g. non-TCP socket).
void recordNetStat(uint16_t phase, uint16_t md, int fd);

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

// Capture kernel TCP_INFO from a connected socket fd at the end of a network
// phase. Only meaningful for the socket OOB path (fd is a real TCP socket).
#define BTRACE_NETSTAT(phase, md, fd) \
  do { \
    if (ncclBootstrapTrace::isEnabled()) \
      ncclBootstrapTrace::recordNetStat((phase), (md), (fd)); \
  } while (0)

#endif  // BOOTSTRAP_TRACE_H_
