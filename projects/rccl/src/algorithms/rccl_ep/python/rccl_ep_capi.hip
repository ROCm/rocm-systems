/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C ABI for the rccl_ep kernels, consumed from Python via ctypes
 * rather than a torch extension, which keeps the binding independent of the
 * torch version. Owns the communicator and the symmetric window; every
 * per-call structure belongs to the caller, which is what makes a cached
 * replay exact.
 *
 * The stream belongs to the caller too. Every data-path entry point takes one
 * and launches on it: torch creates its streams non-blocking, so the legacy
 * null stream does NOT implicitly order against them, and a null-stream launch
 * would read inputs still being produced on the caller's stream. Setup
 * (ep_create, ep_configure) stays on the null stream -- it runs once, before
 * any caller work exists to race with.
 *
 * Every entry point validates its own arguments and returns -1 (or a null
 * handle) with one "[rccl_ep] <fn>: <reason>" line before it launches anything
 * or changes any state, since the ABI is reachable without the Python layer.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "device/combine.h"
#include "device/dispatch.h"
#include "device/ep_common.h"
#include "device/expand.h"
#include "device/hip_prims.h"
#include "device/internode.h"
#include "include/ep_arch.h"
#include "include/ep_layout.h"

using namespace rccl_ep;

namespace {

struct EpHandle {
  EpConfig cfg{};
  ncclComm_t comm{};
  ncclWindow_t win{};
  void* win_buf{};
  size_t win_bytes{};
  EpWindowLayout lay{};
  ncclDevComm devComm{};
  WindowView* d_views{};
  WindowView self_view{};
  int32_t *d_recvoff{}, *d_total{};
  // The HIP device ordinal, kept separately from cfg.rank: the caller is free to
  // map ranks onto devices any way it likes, so querying device properties by
  // rank reads the wrong GPU on every non-identity mapping.
  int device{-1};
  bool devcomm{false};
  bool configured{false};
  // A transfer failed after part of it was posted to the peers (see mark_broken): no later call can
  // pair with what the peers expect, so every call but ep_destroy refuses.
  bool broken{false};

  // Internode (ep_create_nodes with node_size < num_ranks). Ranks [node_base, node_base +
  // node_size) form the LSA team: the window, its devComm and every LSA barrier live on
  // node_comm, a split of `comm`. Writes to a remote peer go through d_views_inter (see
  // InterView) into `outbox`, which has the window's layout, and RCCL send/recv on `comm`
  // moves each region; `inbox` receives returned combine rows. ep_inter_bind sets `bound`.
  bool inter{false};
  int node_size{0}, node_base{0};
  ncclComm_t node_comm{};
  ncclComm_t lsa{};  // node_comm internode, comm otherwise: the window and its devComm live on it
  uint8_t* outbox{};
  uint8_t* inbox{};
  WindowView* d_views_inter{};
  int32_t* d_src_inter{};
  int32_t* d_scratch{};  // count exchange and barrier, kScratchInts
  int32_t* host_scratch{};
  InterCounts bound{};
  const int32_t* bound_send_list{};
  int bound_num_tokens{0};
  bool is_bound{false};
};

// Count exchange scratch: every rank's sendc (R ints each).
constexpr int kScratchInts = kMaxWorldRanks * kMaxWorldRanks;

// Row strides differ by dispatch dtype (x) and by combine mode (y), so a remote peer
// needs one outbox view per stride; the local views are the same in all four.
enum InterView {
  kViewBf16 = 0,
  kViewFp8 = 1,
  kViewCombGrouped = 2,
  kViewCombFlat = 3,
  kNumInterViews = 4
};

inline bool is_remote(const EpHandle* h, int r) {
  return h->inter && (r < h->node_base || r >= h->node_base + h->node_size);
}

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {          \
    fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__,          \
            hipGetErrorString(e_)); return -1; } } while (0)

// Prints "[rccl_ep] <fn>: <reason>" and returns -1: how an entry point rejects a call.
__attribute__((format(printf, 2, 3))) int ep_fail(const char* fn, const char* fmt, ...) {
  char msg[320];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  fprintf(stderr, "[rccl_ep] %s: %s\n", fn, msg);
  return -1;
}

#define REQUIRE_HANDLE(h, fn) \
  do { \
    if ((h) == nullptr) return ep_fail(fn, "null handle"); \
    if ((h)->broken) return ep_fail(fn, "a transfer failed part-way; the handle can only be destroyed"); \
  } while (0)

// A pointer the call reads or writes, required when `needed`. Buffers sized by a token or row
// count are needed only when that count is positive: an empty tensor's data pointer is null.
#define REQUIRE_PTR(fn, p, needed) \
  do { \
    if ((needed) && (p) == 0) return ep_fail(fn, "%s must not be null", #p); \
  } while (0)

// A token count of this rank's (num_tokens) or a received row count (num_recv), within the
// window's capacity: T per source rank, R * T received.
#define REQUIRE_TOKENS(h, fn, n) \
  do { \
    if ((n) < 0 || (n) > (h)->cfg.num_max_tokens_per_rank) { \
      return ep_fail(fn, "%s %d outside [0, num_max_tokens_per_rank=%d]", #n, (int)(n), \
                     (h)->cfg.num_max_tokens_per_rank); \
    } \
  } while (0)
#define REQUIRE_RECV(h, fn, n) \
  do { \
    const int64_t cap_ = (int64_t)(h)->cfg.num_ranks * (h)->cfg.num_max_tokens_per_rank; \
    if ((n) < 0 || (int64_t)(n) > cap_) { \
      return ep_fail(fn, "%s %d outside [0, %lld] (num_ranks x num_max_tokens_per_rank)", #n, (int)(n), \
                     (long long)cap_); \
    } \
  } while (0)

// num_sms is a CTA budget, 0 for the default. Past the grid's y limit a launch would fail
// without stopping the barriers around it.
constexpr int kMaxNumSms = 65535;
#define REQUIRE_SMS(fn, n) \
  do { \
    if ((n) < 0 || (n) > kMaxNumSms) { \
      return ep_fail(fn, "num_sms %d outside [0, %d]", (int)(n), kMaxNumSms); \
    } \
  } while (0)

inline int sms_or(int num_sms, int dflt) { return num_sms > 0 ? num_sms : dflt; }

// `num_sms` is a TOTAL CTA budget, but the dispatch grid is
// (num_ranks, ctas_per_rank), so it has to be divided down. Treating it as a
// per-rank count would quietly take num_ranks times the CUs the caller asked
// for, which defeats the entire point of the knob.
inline int ctas_per_rank(int num_sms, int dflt, int num_ranks) {
  const int total = sms_or(num_sms, dflt);
  const int per = total / num_ranks;
  return per > 0 ? per : 1;
}

// Release the device communicator and the symmetric window. ep_configure's
// failure paths run with `configured` still false, so ep_destroy would never
// reach resources they had already taken; they have to hand them back
// themselves. Every step is guarded by its own flag, so a partially built
// handle unwinds through the same path as a fully built one.
//
// The devComm is torn down first: it holds the LSA barrier state that lives in
// the window, so deregistering the window under it would leave it dangling.
inline void release_window(EpHandle* h, bool registered) {
  if (h->devcomm) {
    ncclDevCommDestroy(h->lsa, &h->devComm);
    h->devcomm = false;
  }
  if (registered && h->win != nullptr) ncclCommWindowDeregister(h->lsa, h->win);
  h->win = nullptr;
  // The internode buffers are sized from the layout, so they go with the window.
  for (void** p : {(void**)&h->outbox, (void**)&h->inbox, (void**)&h->d_views_inter, (void**)&h->d_src_inter}) {
    if (*p) {
      (void)hipFree(*p);
      *p = nullptr;
    }
  }
  h->is_bound = false;
  if (h->win_buf) {
    ncclMemFree(h->win_buf);
    h->win_buf = nullptr;
  }
}

// The node communicator and the count-exchange buffers (ep_create_nodes).
inline void release_inter_comm(EpHandle* h) {
  if (h->node_comm) (void)ncclCommDestroy(h->node_comm);
  h->node_comm = nullptr;
  (void)hipFree(h->d_scratch);
  h->d_scratch = nullptr;
  if (h->host_scratch) (void)hipHostFree(h->host_scratch);
  h->host_scratch = nullptr;
}

// Every data-path entry point is reachable without the Python layer -- the unit
// tests include this ABI directly -- so an unconfigured handle has to say so
// rather than fault on a null window or an uninitialised devComm.
#define REQUIRE_CONFIGURED(h, fn) \
  do { \
    REQUIRE_HANDLE(h, fn); \
    if (!(h)->configured) return ep_fail(fn, "called before a successful ep_configure"); \
  } while (0)

// Makes `device` current for a scope and restores the caller's device when it ends. Configure and teardown allocate,
// free, synchronize and launch on the null stream, all on the current device, which the calling thread may have pointed
// at another GPU since ep_create.
class DeviceScope {
public:
  explicit DeviceScope(int device) {
    if (hipGetDevice(&prev_) != hipSuccess) prev_ = -1;
    if (device < 0 || device == prev_) return;
    const hipError_t he = hipSetDevice(device);
    if (he != hipSuccess) {
      fprintf(stderr, "[rccl_ep] hipSetDevice(%d): %s\n", device, hipGetErrorString(he));
      ok_ = false;
      return;
    }
    switched_ = prev_ >= 0;
  }
  ~DeviceScope() {
    if (switched_) (void)hipSetDevice(prev_);
  }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;
  bool ok() const {
    return ok_;
  }

private:
  int prev_{-1};
  bool switched_{false};
  bool ok_{true};
};

// For a transfer that failed after some of it was posted: the peers expect the rest, which no
// later call can supply, and a kernel of this handle may still be running, so the handle refuses
// everything but ep_destroy from now on (which then releases no device memory; see there).
int mark_broken(EpHandle* h, const char* fn) {
  h->broken = true;
  return ep_fail(fn, "a transfer failed part-way; the handle can only be destroyed");
}

// Waves per block for every kernel that moves payload. The parallelism these
// kernels get is CTAs x waves, and only the CTA count follows the caller's
// num_sms budget, so a narrow block starves them at every budget below the
// default. 16 waves is 1024 threads, the gfx950 ceiling.
//
// Widening changes which wave owns which token and which lane owns which
// element, never the order of additions applied to one element, so results
// stay bit-exact.
//
// Caveat: the combine reduce is 1-D over tokens at one wave per token, so a
// wide block over-subscribes when tokens are few -- the decode regime, which
// this path does not serve.
constexpr int kEpWaves = 16;

#define NC(x) \
  do { \
    ncclResult_t r_ = (x); \
    if (r_ != ncclSuccess) { \
      fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r_)); \
      return -1; \
    } \
  } while (0)

// Inside ncclGroupStart/End, NC would return with the group still open, and every later NCCL
// call on this thread would then only be queued: a hang instead of an error. NCG keeps the first
// failure in `err` and skips the remaining posts; NCG_END always closes the group, then fails if
// anything did.
#define NCG(err, x) \
  do { \
    if ((err) == ncclSuccess && ((err) = (x)) != ncclSuccess) { \
      fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(err)); \
    } \
  } while (0)
#define NCG_END(err) \
  do { \
    ncclResult_t e_ = ncclGroupEnd(); \
    if (e_ != ncclSuccess) { \
      fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(e_)); \
    } \
    if ((err) != ncclSuccess || e_ != ncclSuccess) return -1; \
  } while (0)

// The layout a remote peer q is seen through: every region offset moves by (q - rank)
// rows of that region, so the kernels' slot(rank, i) = rank * T + i lands on row
// q * T + i, in outbox region q. A negative shift is exact under size_t wrap-around.
EpWindowLayout remote_layout(const EpWindowLayout& l, const EpConfig& c, int q, int kind) {
  const int64_t d = (int64_t)q - c.rank;
  const int64_t T = c.num_max_tokens_per_rank, H = c.hidden, K = c.num_topk;
  auto sh = [d](size_t off, int64_t row_bytes) { return off + (size_t)(d * row_bytes); };
  EpWindowLayout r = l;
  r.off_counts = sh(l.off_counts, sizeof(int32_t));
  r.off_x = sh(l.off_x, T * H * (kind == kViewFp8 ? 1 : 2));
  r.off_sf = sh(l.off_sf, T * c.hidden_sf() * (int64_t)sizeof(float));
  r.off_topk_idx = sh(l.off_topk_idx, T * K * (int64_t)sizeof(int32_t));
  r.off_topk_w = sh(l.off_topk_w, T * K * (int64_t)sizeof(float));
  r.off_src_idx = sh(l.off_src_idx, T * (int64_t)sizeof(int32_t));
  r.off_y = sh(l.off_y, T * H * 2 * (kind == kViewCombFlat ? K : 1));
  r.off_cw = sh(l.off_cw, T * K * (int64_t)sizeof(float));
  r.off_arch = sh(l.off_arch, kArchLen);
  return r;
}

// One region moved per peer: its offset in any buffer with the window's layout, and its
// row size. Region q starts at off + q * T * row in both the outbox and the window.
struct InterRegion {
  size_t off, row;
};

// Send outbox region q and receive region q of `dst` for every remote q, in one group.
// Both sides post ranks and regions in the same order, so every send meets its receive;
// a zero count is skipped on both sides, since one's sendc[q] is the other's recvc[q].
int inter_exchange(EpHandle* h, const InterRegion* regs, int nreg, const int32_t* send_rows, const int32_t* recv_rows,
                   uint8_t* dst, hipStream_t s) {
  const size_t T = h->cfg.num_max_tokens_per_rank;
  NC(ncclGroupStart());
  ncclResult_t gr = ncclSuccess;
  for (int q = 0; q < h->cfg.num_ranks; ++q) {
    if (!is_remote(h, q)) continue;
    for (int g = 0; g < nreg; ++g) {
      const size_t base = regs[g].off + (size_t)q * T * regs[g].row;
      if (send_rows[q]) {
        NCG(gr, ncclSend(h->outbox + base, (size_t)send_rows[q] * regs[g].row, ncclUint8, q, h->comm, s));
      }
      if (recv_rows[q]) {
        NCG(gr, ncclRecv(dst + base, (size_t)recv_rows[q] * regs[g].row, ncclUint8, q, h->comm, s));
      }
    }
  }
  NCG_END(gr);
  return 0;
}

// Dispatch's remote leg, after dispatch_impl has filled the outbox: each region goes to
// its peer, and each remote source's rows and count land in this window for the epilogue.
int inter_dispatch(EpHandle* h, int use_fp8, hipStream_t s) {
  const EpConfig& c = h->cfg;
  const EpWindowLayout& l = h->lay;
  const size_t H = c.hidden, K = c.num_topk;
  InterRegion regs[5];
  int n = 0;
  regs[n++] = {l.off_x, H * (use_fp8 ? 1 : sizeof(bf16_t))};
  if (use_fp8) regs[n++] = {l.off_sf, (size_t)c.hidden_sf() * sizeof(float)};
  regs[n++] = {l.off_topk_idx, K * sizeof(int32_t)};
  regs[n++] = {l.off_topk_w, K * sizeof(float)};
  regs[n++] = {l.off_src_idx, sizeof(int32_t)};
  uint8_t* win = (uint8_t*)h->win_buf;
  if (inter_exchange(h, regs, n, h->bound.sendc, h->bound.recvc, win, s) != 0) return -1;
  hipLaunchKernelGGL(k_inter_set_counts, dim3(1), dim3(kMaxWorldRanks), 0, s, c.num_ranks, h->bound,
                     (int32_t*)(win + l.off_counts));
  return 0;
}

// Combine's remote leg, after combine_impl has filled the outbox: rows held for a remote
// owner go back to it, and rows returned by each remote rank are scattered from the inbox
// into this window's combine region for the reduce epilogue. The weight region always travels
// with the rows (K floats per row): whether to pass weights is each rank's own choice, and a
// rank that received no rows may pass none, but every rank must post the same transfers.
int inter_combine(EpHandle* h, int grouped, int cta, hipStream_t s) {
  const EpConfig& c = h->cfg;
  const EpWindowLayout& l = h->lay;
  const size_t H = c.hidden, K = c.num_topk;
  const int y_row_elems = (int)(grouped ? H : K * H);
  InterRegion regs[2] = {{l.off_y, (size_t)y_row_elems * sizeof(bf16_t)}, {l.off_cw, K * sizeof(float)}};
  // Owner-side: this rank received recvc[o] rows from o, and gets back sendc[r] from r.
  if (inter_exchange(h, regs, 2, h->bound.recvc, h->bound.sendc, h->inbox, s) != 0) return -1;
  hipLaunchKernelGGL(k_inter_unpack_combine, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s, c, h->bound,
                     h->bound_send_list, h->bound_num_tokens, (const bf16_t*)(h->inbox + l.off_y),
                     (const float*)(h->inbox + l.off_cw), y_row_elems, h->self_view);
  return 0;
}

#define REQUIRE_BOUND(h, fn) \
  do { \
    if ((h)->inter && !(h)->is_bound) return ep_fail(fn, "an internode handle needs ep_inter_bind first"); \
  } while (0)

// The arch check over the global communicator: an all-gather of every rank's arch string,
// in place of the LSA window the intranode check travels over, which spans one node here.
int check_uniform_arch_global(EpHandle* h, char* err, size_t err_len) {
  const int R = h->cfg.num_ranks;
  char mine[kArchLen];
  if (local_arch_string(h->cfg, h->device, mine, err, err_len) != 0) return -1;
  char* d = nullptr;
  if (hipMalloc(&d, (size_t)(R + 1) * kArchLen) != hipSuccess) {
    snprintf(err, err_len, "hipMalloc for the arch check failed");
    return -1;
  }
  char* all = (char*)malloc((size_t)R * kArchLen);
  int rc = -1;
  if (all != nullptr && hipMemcpy(d + (size_t)R * kArchLen, mine, kArchLen, hipMemcpyHostToDevice) == hipSuccess &&
      ncclAllGather(d + (size_t)R * kArchLen, d, kArchLen, ncclUint8, h->comm, 0) == ncclSuccess &&
      hipDeviceSynchronize() == hipSuccess &&
      hipMemcpy(all, d, (size_t)R * kArchLen, hipMemcpyDeviceToHost) == hipSuccess) {
    rc = compare_arch_table(all, R, err, err_len);
  } else {
    snprintf(err, err_len, "arch all-gather failed");
  }
  free(all);
  (void)hipFree(d);
  return rc;
}

}  // namespace

__global__ void k_capi_fill_views(ncclWindow_t w, EpWindowLayout l, int n, WindowView* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int p = 0; p < n; ++p) { out[p].base = (uint8_t*)ncclGetLsaPointer(w, 0, p); out[p].l = l; }
  }
}

// Internode views: a local peer's is its LSA window; a remote peer's aims at the outbox through
// remote_layout, one table per InterView. d_views gets the bf16 table. Also allocates the
// internode buffers.
static int inter_fill_views(EpHandle* h) {
  const EpConfig& c = h->cfg;
  const int R = c.num_ranks;
  uint8_t** d_bases = nullptr;
  uint8_t* bases[kMaxRanks] = {};
  if (hipMalloc(&d_bases, h->node_size * sizeof(uint8_t*)) != hipSuccess) return -1;
  hipLaunchKernelGGL(k_inter_lsa_bases, dim3(1), dim3(1), 0, 0, h->win, h->node_size, d_bases);
  const bool ok = hipMemcpy(bases, d_bases, h->node_size * sizeof(uint8_t*), hipMemcpyDeviceToHost) == hipSuccess;
  (void)hipFree(d_bases);
  // The outbox and the inbox each take a full window. Most of each goes unused, but sharing the
  // window's layout is what lets the intranode kernels write the outbox.
  const size_t lay_bytes = h->lay.total_bytes;
  if (!ok || hipMalloc(&h->outbox, lay_bytes) != hipSuccess || hipMalloc(&h->inbox, lay_bytes) != hipSuccess ||
      hipMalloc(&h->d_src_inter, (size_t)R * c.num_max_tokens_per_rank * sizeof(int32_t)) != hipSuccess ||
      hipMalloc(&h->d_views_inter, kNumInterViews * R * sizeof(WindowView)) != hipSuccess) {
    fprintf(stderr, "[rccl_ep] internode buffers (2 x %zu bytes) failed\n", lay_bytes);
    return -1;
  }
  WindowView views[kNumInterViews * kMaxWorldRanks];
  for (int kind = 0; kind < kNumInterViews; ++kind) {
    for (int p = 0; p < R; ++p) {
      views[kind * R + p] = is_remote(h, p) ? WindowView{h->outbox, remote_layout(h->lay, c, p, kind)} :
                                              WindowView{bases[p - h->node_base], h->lay};
    }
  }
  if (hipMemcpy(h->d_views_inter, views, kNumInterViews * R * sizeof(WindowView), hipMemcpyHostToDevice) !=
        hipSuccess ||
      hipMemcpy(h->d_views, views + kViewBf16 * R, R * sizeof(WindowView), hipMemcpyHostToDevice) != hipSuccess) {
    return -1;
  }
  return 0;
}

// librccl is built -fvisibility=hidden. Without this the entry points below
// become STB_LOCAL, the library still links, and the ctypes binding fails at
// import with "undefined symbol: ep_create".
#pragma GCC visibility push(default)
extern "C" {

int ep_unique_id_size() { return (int)sizeof(ncclUniqueId); }

int ep_get_unique_id(char* out) {
  REQUIRE_PTR("ep_get_unique_id", out, true);
  ncclUniqueId id;
  if (ncclGetUniqueId(&id) != ncclSuccess) return -1;
  memcpy(out, &id, sizeof(id));
  return 0;
}

void* ep_create_nodes(int rank, int num_ranks, const char* unique_id, int max_tokens_per_rank, int hidden, int device,
                      int node_size);

// unique_id must be ncclUniqueId bytes broadcast from rank 0 by the caller.
void* ep_create(int rank, int num_ranks, const char* unique_id,
                int max_tokens_per_rank, int hidden, int device) {
  return ep_create_nodes(rank, num_ranks, unique_id, max_tokens_per_rank, hidden, device, num_ranks);
}

// ep_create for num_ranks / node_size nodes, node n holding ranks [n * node_size,
// (n + 1) * node_size); node_size == num_ranks is ep_create. A peer on another node is
// reached by RCCL send/recv (see EpHandle).
void* ep_create_nodes(int rank, int num_ranks, const char* unique_id, int max_tokens_per_rank, int hidden, int device,
                      int node_size) {
  // Every argument check below runs before ncclCommInitRank and gives the same answer on every rank given the same
  // arguments, so a rejected call returns on each rank instead of leaving its peers waiting in the init. The device
  // check, which can differ between ranks, is agreed over the new communicator instead (see below).
  if (unique_id == nullptr) {
    fprintf(stderr, "[rccl_ep] ep_create: unique_id must not be null\n");
    return nullptr;
  }
  // Payload copies move 16B per lane from a 256B-aligned region base, so the
  // vector paths are only aligned when hidden is a multiple of 8. The scalar
  // tails cover a leftover element count, not a misaligned base, so reject an
  // odd hidden here rather than reach a misaligned dwordx4.
  if (hidden <= 0 || hidden % 8 != 0) {
    fprintf(stderr, "[rccl_ep] ep_create: hidden %d must be a positive multiple of 8 (16B vector alignment)\n", hidden);
    return nullptr;
  }
  // kMaxRanks bounds the scale-up domain the window layout is written for, kMaxWorldRanks the
  // internode group, and ctas_per_rank divides by num_ranks, so zero is a division by zero.
  const int max_ranks = node_size < num_ranks ? kMaxWorldRanks : kMaxRanks;
  if (num_ranks < 1 || num_ranks > max_ranks || node_size > kMaxRanks) {
    fprintf(stderr, "[rccl_ep] ep_create: num_ranks %d must be in [1, %d], node_size %d at most kMaxRanks=%d\n",
            num_ranks, max_ranks, node_size, kMaxRanks);
    return nullptr;
  }
  // EpWindowLayout::make widens this to size_t, so a negative value sign-extends
  // and wraps every offset in the window; zero reserves empty regions that the
  // slot arithmetic then indexes anyway.
  if (max_tokens_per_rank < 1) {
    fprintf(stderr, "[rccl_ep] ep_create: max_tokens_per_rank %d must be positive\n", max_tokens_per_rank);
    return nullptr;
  }
  if (node_size < 1 || num_ranks % node_size != 0) {
    fprintf(stderr, "[rccl_ep] ep_create: node_size %d must divide num_ranks %d\n", node_size, num_ranks);
    return nullptr;
  }
  if (num_ranks / node_size > kMaxNodes) {
    fprintf(stderr, "[rccl_ep] ep_create: %d nodes exceed kMaxNodes=%d\n", num_ranks / node_size, kMaxNodes);
    return nullptr;
  }
  if (rank < 0 || rank >= num_ranks) {
    fprintf(stderr, "[rccl_ep] ep_create: rank %d outside [0, num_ranks=%d)\n", rank, num_ranks);
    return nullptr;
  }
  // Bind explicitly. The caller's torch context may have selected a different
  // device than this thread's current one.
  hipError_t he = hipSetDevice(device);
  if (he != hipSuccess) {
    fprintf(stderr, "[rccl_ep] ep_create: hipSetDevice(%d): %s\n", device, hipGetErrorString(he));
    return nullptr;
  }
  // The kernels are wave64 throughout (kWarpSize lanes per wave, 64-bit ballots): a wave32
  // device would compute wrong lane and wave indices rather than fail. Checked once every rank
  // has joined the communicator, below.
  int warp_size = 0;
  if (hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device) != hipSuccess) warp_size = -1;
  // Test hook, as RCCL_EP_FAKE_ARCH is for the architecture check: RCCL_EP_FAKE_WAVE_SIZE replaces the
  // wave size on the rank RCCL_EP_FAKE_WAVE_SIZE_RANK names (every rank without it), so the failing
  // path can be tested on wave64 GPUs.
  const char* fake_wave = getenv("RCCL_EP_FAKE_WAVE_SIZE");
  const char* fake_wave_rank = getenv("RCCL_EP_FAKE_WAVE_SIZE_RANK");
  if (fake_wave && *fake_wave && (!fake_wave_rank || atoi(fake_wave_rank) == rank)) warp_size = atoi(fake_wave);
  const bool wave_ok = warp_size == kWarpSize;

  auto* h = new EpHandle();
  h->cfg.num_ranks = num_ranks;
  h->cfg.rank = rank;
  h->cfg.hidden = hidden;
  h->cfg.num_max_tokens_per_rank = max_tokens_per_rank;
  h->device = device;

  ncclUniqueId id;
  memcpy(&id, unique_id, sizeof(id));
  ncclResult_t nr = ncclCommInitRank(&h->comm, num_ranks, id, rank);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclCommInitRank: %s\n", ncclGetErrorString(nr));
    delete h;
    return nullptr;
  }
  h->lsa = h->comm;
  h->node_size = node_size;
  if (node_size < num_ranks) {
    h->inter = true;
    h->node_base = rank / node_size * node_size;
    nr = ncclCommSplit(h->comm, rank / node_size, rank, &h->node_comm, nullptr);
    if (nr != ncclSuccess || hipMalloc(&h->d_scratch, kScratchInts * sizeof(int32_t)) != hipSuccess ||
        hipHostMalloc((void**)&h->host_scratch, kScratchInts * sizeof(int32_t)) != hipSuccess) {
      fprintf(stderr, "[rccl_ep] internode setup failed: %s\n",
              nr != ncclSuccess ? ncclGetErrorString(nr) : "scratch allocation");
      release_inter_comm(h);
      ncclCommDestroy(h->comm);
      delete h;
      return nullptr;
    }
    h->lsa = h->node_comm;
  }
  // Unchecked, a failure here returns a handle of null device pointers that
  // faults inside a kernel later, leaking the comm.
  const bool alloced =
      hipMalloc(&h->d_recvoff, num_ranks * sizeof(int32_t)) == hipSuccess &&
      hipMalloc(&h->d_total, sizeof(int32_t)) == hipSuccess &&
      hipMalloc(&h->d_views, num_ranks * sizeof(WindowView)) == hipSuccess;
  auto release = [h]() -> void* {
    (void)hipFree(h->d_recvoff);
    (void)hipFree(h->d_total);
    (void)hipFree(h->d_views);
    release_inter_comm(h);
    ncclCommDestroy(h->comm);
    delete h;
    return nullptr;
  };
  if (!alloced) {
    fprintf(stderr, "[rccl_ep] hipMalloc for scratch failed\n");
    return release();
  }
  // The device check, agreed over the communicator: the number of ranks whose device fails it, summed in d_total
  // (scratch until the first dispatch). Every rank sees the same sum, so a group with any such device fails on every
  // rank together and releases its communicator, rather than one rank returning while the others wait for it in their
  // next collective.
  int bad_ranks = wave_ok ? 0 : 1;
  if (hipMemcpy(h->d_total, &bad_ranks, sizeof(int32_t), hipMemcpyHostToDevice) != hipSuccess ||
      ncclAllReduce(h->d_total, h->d_total, 1, ncclInt32, ncclSum, h->comm, nullptr) != ncclSuccess ||
      hipDeviceSynchronize() != hipSuccess ||
      hipMemcpy(&bad_ranks, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
    fprintf(stderr, "[rccl_ep] ep_create: the cross-rank device check failed\n");
    return release();
  }
  if (bad_ranks > 0) {
    if (!wave_ok) {
      fprintf(stderr, "[rccl_ep] ep_create: device %d has a wave size of %d; rccl_ep needs a wave64 (gfx9) device\n",
              device, warp_size);
    } else {
      fprintf(stderr, "[rccl_ep] ep_create: %d rank(s) of this group have no wave64 (gfx9) device; failing on all\n",
              bad_ranks);
    }
    return release();
  }
  return h;
}

// Size and register the symmetric window. Separate from ep_create because the
// window size depends on num_topk and num_experts, which the API only
// reveals at the first dispatch, not at buffer construction.
int ep_configure(void* handle, int num_experts, int num_topk) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_configure";
  REQUIRE_HANDLE(h, fn);
  if (h->configured && h->cfg.num_experts == num_experts && h->cfg.num_topk == num_topk) {
    return 0;
  }
  // Validated before the teardown below, so a rejected reconfigure leaves the
  // existing window intact rather than freed.
  if (num_topk < 1 || num_topk > kMaxTopk) {
    return ep_fail(fn, "num_topk %d must be in [1, kMaxTopk=%d]", num_topk, kMaxTopk);
  }
  // experts_per_rank() truncates, so a non-multiple leaves the tail experts
  // owned by no rank: k_ep_plan never routes their tokens, and the epilogue's
  // owner = e / experts_per_rank() reaches num_ranks, indexing one whole
  // rank-region past the end of `y`. num_experts <= 0 is the epr == 0 case.
  if (num_experts <= 0 || num_experts % h->cfg.num_ranks != 0) {
    return ep_fail(fn, "num_experts %d must be a positive multiple of num_ranks %d", num_experts, h->cfg.num_ranks);
  }
  // Everything below allocates, frees and launches on the current device (see DeviceScope).
  const DeviceScope dev(h->device);
  if (!dev.ok()) return ep_fail(fn, "cannot select device %d", h->device);
  if (h->configured) {
    release_window(h, true);
    h->configured = false;
  }
  h->cfg.num_experts = num_experts;
  h->cfg.num_topk = num_topk;
  h->lay = EpWindowLayout::make(h->cfg, sizeof(bf16_t));
  h->win_bytes = h->lay.total_bytes;

  ncclResult_t nr = ncclMemAlloc(&h->win_buf, h->win_bytes);
  if (nr != ncclSuccess) {
    h->win_buf = nullptr;
    return ep_fail(fn, "ncclMemAlloc(%zu): %s", h->win_bytes, ncclGetErrorString(nr));
  }
  // Spelled out rather than wrapped in HC: the macro returns before the window
  // can be handed back.
  if (hipMemset(h->win_buf, 0, h->win_bytes) != hipSuccess) {
    release_window(h, false);
    return ep_fail(fn, "clearing the %zu-byte window failed", h->win_bytes);
  }
  h->win = nullptr;
  nr = ncclCommWindowRegister(h->lsa, h->win_buf, h->win_bytes, &h->win, NCCL_WIN_COLL_SYMMETRIC);
  if (nr != ncclSuccess) {
    release_window(h, false);
    return ep_fail(fn, "WindowRegister: %s (needs NCCL_WIN_ENABLE=1 and NCCL_CUMEM_ENABLE=1)", ncclGetErrorString(nr));
  }
  // RCCL can decline a registration and still return success, with a null window, when the
  // communicator has no symmetric memory (window support disabled, or cuMem off).
  if (h->win == nullptr) {
    release_window(h, false);
    return ep_fail(fn, "symmetric memory is unavailable (the window registration was declined); "
                       "needs NCCL_CUMEM_ENABLE=1 and window support");
  }

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = kEpBarrierCtas;  // LSA only: no GIN contexts on the intranode path
  nr = ncclDevCommCreate(h->lsa, &reqs, &h->devComm);
  if (nr != ncclSuccess) {
    release_window(h, true);
    return ep_fail(fn, "ncclDevCommCreate: %s", ncclGetErrorString(nr));
  }
  h->devcomm = true;

  // k_capi_fill_views resolves peer windows by LSA index while every data-path
  // kernel indexes d_views by world rank, so the two agree only when the LSA
  // team is this rank's node under the identity mapping. A comm that spans
  // nodes without ep_create_nodes, or one whose ranks are reordered within the
  // node, would otherwise push payloads into the wrong peer's window with no error.
  if (h->devComm.lsaSize != h->node_size || h->devComm.lsaRank != h->cfg.rank - h->node_base) {
    release_window(h, true);
    return ep_fail(fn,
                   "LSA team is rank %d of %d but this node is ranks [%d, %d) of %d, this rank %d; "
                   "rccl_ep needs every node's ranks contiguous, with the identity LSA mapping "
                   "(a communicator spanning nodes needs ep_create_nodes)",
                   h->devComm.lsaRank, h->devComm.lsaSize, h->node_base, h->node_base + h->node_size, h->cfg.num_ranks,
                   h->cfg.rank);
  }

  if (h->inter) {
    if (inter_fill_views(h) != 0) {
      release_window(h, true);
      return -1;
    }
  } else {
    hipLaunchKernelGGL(k_capi_fill_views, dim3(1), dim3(1), 0, 0, h->win, h->lay, h->cfg.num_ranks, h->d_views);
  }
  // Spelled out rather than wrapped in HC: the macro returns before the window
  // can be handed back.
  hipError_t he = hipDeviceSynchronize();
  if (he != hipSuccess) {
    release_window(h, true);
    return ep_fail(fn, "view fill: %s", hipGetErrorString(he));
  }
  h->self_view = WindowView{(uint8_t*)h->win_buf, h->lay};

  // Before any payload moves: a mixed-architecture communicator has no correct
  // behaviour, only a slower way to be wrong. This runs here rather than in
  // ep_create because it travels over the LSA window, which does not exist
  // until now.
  char err[256];
  if ((h->inter ? check_uniform_arch_global(h, err, sizeof(err)) :
                  check_uniform_arch(h->cfg, h->device, h->d_views, h->self_view, h->devComm, err, sizeof(err))) != 0) {
    release_window(h, true);
    return ep_fail(fn, "%s", err);
  }

  h->configured = true;
  return 0;
}

size_t ep_window_bytes(void* handle) {
  return handle ? static_cast<EpHandle*>(handle)->win_bytes : 0;
}

// A broken handle (see mark_broken) releases none of its device memory, window or communicators:
// a kernel of its own or a peer's transfer may still use them, and releasing them would wait for
// that kernel anyway (hipFree synchronizes the device). It leaks them, with one line saying so,
// and frees only the host-side handle.
void ep_destroy(void* handle) {
  auto* h = static_cast<EpHandle*>(handle);
  if (!h) return;
  if (h->broken) {
    fprintf(stderr,
            "[rccl_ep] rank %d: ep_destroy: the handle is broken; leaking its device memory, window and "
            "communicators, which a kernel still running may use\n",
            h->cfg.rank);
    delete h;
    return;
  }
  const DeviceScope dev(h->device);
  // Unconditional rather than gated on `configured`: a handle whose ep_configure
  // failed part-way has already handed back what it took, and release_window is
  // a no-op on the flags it cleared, so this covers both without double-freeing.
  release_window(h, h->configured);
  (void)hipFree(h->d_recvoff);
  (void)hipFree(h->d_total);
  (void)hipFree(h->d_views);
  release_inter_comm(h); // the split goes before its parent
  ncclCommDestroy(h->comm);
  delete h;
}

int ep_barrier(void* handle, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  REQUIRE_CONFIGURED(h, "ep_barrier");
  hipStream_t s = (hipStream_t)stream;
  if (h->inter) {
    // The LSA team is one node: rendezvous over the global communicator instead.
    NC(ncclAllReduce(h->d_scratch, h->d_scratch, 1, ncclInt32, ncclSum, h->comm, s));
  } else {
    hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  }
  HC(hipStreamSynchronize(s));
  return 0;
}

// Routing plan. Written into caller-owned tensors so a cached dispatch replays
// the identical plan rather than depending on library state.
int ep_plan(void* handle, uintptr_t topk_idx, int num_tokens,
            uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc,
            uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_plan";
  REQUIRE_CONFIGURED(h, fn);
  REQUIRE_TOKENS(h, fn, num_tokens);
  REQUIRE_PTR(fn, topk_idx, num_tokens > 0);
  REQUIRE_PTR(fn, out_slot, num_tokens > 0);
  REQUIRE_PTR(fn, out_list, num_tokens > 0);
  REQUIRE_PTR(fn, out_sendc, true);
  hipStream_t s = (hipStream_t)stream;
  // A full block per destination; k_ep_plan sizes its per-wave scan from blockDim.
  static_assert(kEpWaves <= kPlanMaxWaves, "k_ep_plan's wave scan is sized by kPlanMaxWaves");
  hipLaunchKernelGGL(k_ep_plan, dim3(h->cfg.num_ranks), dim3(kEpWaves * kWarpSize), 0, s, h->cfg,
                     (const int32_t*)topk_idx, num_tokens, (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc);
  HC(hipStreamSynchronize(s));
  return 0;
}

// Returns the number of received tokens, or -1 on error.
int ep_dispatch(void* handle,
                uintptr_t x, uintptr_t x_sf,
                uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf,
                uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_dispatch";
  hipStream_t s = (hipStream_t)stream;
  REQUIRE_CONFIGURED(h, fn);
  const EpConfig& c = h->cfg;
  // Slot index is rank * num_max_tokens_per_rank + i, with i bounded only by
  // num_tokens in k_ep_plan. Past capacity the push runs into the next rank's
  // region of the peer's window, and combine's src_tok = g % T then aliases
  // two tokens onto one slot.
  REQUIRE_TOKENS(h, fn, num_tokens);
  REQUIRE_SMS(fn, num_sms);
  // fp8 payload is 1 byte/element, so the 16B vector copies need hidden % 16 == 0;
  // ep_create's hidden % 8 guard only covers bf16 (2 bytes/element).
  if (use_fp8 && c.hidden % 16 != 0) {
    return ep_fail(fn, "hidden %d must be a multiple of 16 for fp8 (16B vector alignment)", c.hidden);
  }
  // Per-token inputs are read only when this rank has tokens; a rank with none passes null
  // pointers (empty tensors). The fp8 paths read x_sf for every local token and write out_sf
  // unconditionally, so a missing scale tensor would be a device-side fault, not a skipped copy.
  const bool tok = num_tokens > 0;
  REQUIRE_PTR(fn, sendc, true);
  REQUIRE_PTR(fn, send_list, tok);
  REQUIRE_PTR(fn, x, tok);
  REQUIRE_PTR(fn, x_sf, tok && use_fp8);
  REQUIRE_PTR(fn, topk_idx, tok);
  REQUIRE_PTR(fn, topk_w, tok);
  // The epilogue writes every output, metadata included, for each row this rank receives, which
  // is not known here, so all of them are required.
  REQUIRE_PTR(fn, out_x, true);
  REQUIRE_PTR(fn, out_sf, use_fp8);
  REQUIRE_PTR(fn, out_topk, true);
  REQUIRE_PTR(fn, out_tw, true);
  REQUIRE_PTR(fn, out_src, true);

  // An internode dispatch posts its receives at the counts the caller bound for it.
  REQUIRE_BOUND(h, fn);

  // 128 total CTAs is the measured bandwidth peak on gfx950; sweeping the CTA
  // budget shows dispatch flat from 32 upward on 8x MI350X.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);
  WindowView* views = h->inter ? h->d_views_inter + (use_fp8 ? kViewFp8 : kViewBf16) * c.num_ranks : h->d_views;

  HC(hipMemsetAsync((uint8_t*)h->win_buf + h->lay.off_counts, 0, c.num_ranks * sizeof(int32_t), s));

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  const dim3 grid(c.num_ranks, cta), block(kEpWaves * kWarpSize);
  if (use_fp8) {
    hipLaunchKernelGGL(dispatch_impl<fp8_t>, grid, block, 0, s, c, (const fp8_t*)x, (const float*)x_sf,
                       (const int32_t*)topk_idx, (const float*)topk_w, num_tokens, (const int32_t*)send_list,
                       (const int32_t*)sendc, views);
  } else {
    hipLaunchKernelGGL(dispatch_impl<bf16_t>, grid, block, 0, s, c, (const bf16_t*)x, (const float*)nullptr,
                       (const int32_t*)topk_idx, (const float*)topk_w, num_tokens, (const int32_t*)send_list,
                       (const int32_t*)sendc, views);
  }
  // Same stream, so the remote rows and counts are in this window before barrier B. A remote
  // source writes this window only through this rank's own receive, so barrier A only has local
  // writers to hold off.
  if (h->inter && inter_dispatch(h, use_fp8, s) != 0) return mark_broken(h, fn);
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  hipLaunchKernelGGL(k_ep_scan_counts, dim3(1), dim3(1), 0, s,
                     c, h->self_view, h->d_recvoff, h->d_total);

  // The epilogue reads the receive offsets on the device, so it follows the scan on `s` without
  // the host knowing the receive total; one sync after it returns the total.
  if (use_fp8) {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl<fp8_t>, grid, block, 0, s, c, h->self_view, h->d_recvoff,
                       (fp8_t*)out_x, (float*)out_sf, (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src);
  } else {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl<bf16_t>, grid, block, 0, s, c, h->self_view, h->d_recvoff,
                       (bf16_t*)out_x, (float*)nullptr, (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src);
  }
  int total = 0;
  HC(hipMemcpyAsync(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  return total;
}

// Per-source received counts, so the caller can report the per-rank prefix sum.
int ep_recv_counts(void* handle, uintptr_t out_counts, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_recv_counts";
  REQUIRE_CONFIGURED(h, fn);
  REQUIRE_PTR(fn, out_counts, true);
  hipStream_t s = (hipStream_t)stream;
  // No sync: the destination is a caller tensor read back on the same stream.
  HC(hipMemcpyAsync((void*)out_counts, (uint8_t*)h->win_buf + h->lay.off_counts,
                    h->cfg.num_ranks * sizeof(int32_t), hipMemcpyDeviceToDevice, s));
  return 0;
}

int ep_expert_counts(void* handle, uintptr_t recv_topk, int num_recv, uintptr_t out_counts,
                     uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_expert_counts";
  REQUIRE_CONFIGURED(h, fn);
  REQUIRE_RECV(h, fn, num_recv);
  REQUIRE_PTR(fn, recv_topk, num_recv > 0);
  REQUIRE_PTR(fn, out_counts, true);
  hipStream_t s = (hipStream_t)stream;
  hipLaunchKernelGGL(k_expand_count, dim3(h->cfg.experts_per_rank()), dim3(256), 0, s,
                     h->cfg, (const int32_t*)recv_topk, num_recv, (int32_t*)out_counts);
  HC(hipStreamSynchronize(s));
  return 0;
}

// Build the expanded-layout row map. Returns total rows (including padding).
int ep_expand_build(void* handle, uintptr_t recv_topk, int num_recv, int expert_alignment,
                    uintptr_t counts, uintptr_t offsets, uintptr_t psum,
                    uintptr_t hist, uintptr_t row_map, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_expand_build";
  REQUIRE_CONFIGURED(h, fn);
  REQUIRE_RECV(h, fn, num_recv);
  if (expert_alignment < 1) return ep_fail(fn, "expert_alignment %d must be positive", expert_alignment);
  REQUIRE_PTR(fn, recv_topk, num_recv > 0);
  REQUIRE_PTR(fn, hist, num_recv > 0);
  REQUIRE_PTR(fn, row_map, num_recv > 0);
  REQUIRE_PTR(fn, counts, true);
  REQUIRE_PTR(fn, offsets, true);
  REQUIRE_PTR(fn, psum, true);
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  const int epr = c.experts_per_rank();
  const int blocks = num_recv > 0 ? (num_recv + 255) / 256 : 1;

  hipLaunchKernelGGL(k_expand_count, dim3(epr), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)counts);
  hipLaunchKernelGGL(k_expand_scan, dim3(1), dim3(1), 0, s,
                     c, expert_alignment, (const int32_t*)counts,
                     (int32_t*)offsets, (int32_t*)psum, h->d_total);
  hipLaunchKernelGGL(k_expand_hist, dim3(blocks), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_base, dim3(epr), dim3(kWarpSize), 0, s,
                     c, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_map, dim3(blocks), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv,
                     (const int32_t*)offsets, (const int32_t*)hist, (int32_t*)row_map);

  // Row count reaches the host to size the expanded tensors, so this sync is
  // structural. The copy is already ordered behind k_expand_scan on `s`.
  int total = 0;
  HC(hipMemcpyAsync(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  return total;
}

int ep_expand_scatter(void* handle, int use_fp8,
                      uintptr_t src_x, uintptr_t src_sf, uintptr_t src_w,
                      uintptr_t row_map, int num_recv,
                      uintptr_t out_x, uintptr_t out_sf, int sf_row_stride, int sf_col_stride,
                      uintptr_t out_w,
                      int zero_pad, int expert_alignment,
                      uintptr_t counts, uintptr_t offsets, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_expand_scatter";
  REQUIRE_CONFIGURED(h, fn);
  const EpConfig& c = h->cfg;
  REQUIRE_RECV(h, fn, num_recv);
  REQUIRE_SMS(fn, num_sms);
  // The same 16-byte vector bound dispatch applies to an fp8 row.
  if (use_fp8 && c.hidden % 16 != 0) {
    return ep_fail(fn, "hidden %d must be a multiple of 16 for fp8 (16B vector alignment)", c.hidden);
  }
  const bool rows = num_recv > 0;
  REQUIRE_PTR(fn, src_x, rows);
  REQUIRE_PTR(fn, src_w, rows && out_w != 0);
  REQUIRE_PTR(fn, row_map, rows);
  // With no rows every count is 0, so the padding pass writes nothing either.
  REQUIRE_PTR(fn, out_x, rows);
  REQUIRE_PTR(fn, src_sf, rows && use_fp8);
  REQUIRE_PTR(fn, out_sf, rows && use_fp8);
  if (use_fp8 && (sf_row_stride < 0 || sf_col_stride < 0)) {
    return ep_fail(fn, "scale strides %d / %d must not be negative", sf_row_stride, sf_col_stride);
  }
  if (zero_pad) {
    if (expert_alignment < 1) return ep_fail(fn, "expert_alignment %d must be positive", expert_alignment);
    REQUIRE_PTR(fn, counts, true);
    REQUIRE_PTR(fn, offsets, true);
  }
  hipStream_t s = (hipStream_t)stream;
  const int cta = sms_or(num_sms, 64);

  if (use_fp8) {
    hipLaunchKernelGGL(k_expand_scatter<fp8_t>, dim3(cta), dim3(kEpWaves * kWarpSize), 0, s, c, (const fp8_t*)src_x,
                       (const float*)src_sf, (const float*)src_w, (const int32_t*)row_map, num_recv, (fp8_t*)out_x,
                       (float*)out_sf, sf_row_stride, sf_col_stride, (float*)out_w);
  } else {
    hipLaunchKernelGGL(k_expand_scatter<bf16_t>, dim3(cta), dim3(kEpWaves * kWarpSize), 0, s, c, (const bf16_t*)src_x,
                       (const float*)nullptr, (const float*)src_w, (const int32_t*)row_map, num_recv, (bf16_t*)out_x,
                       (float*)nullptr, 0, 0, (float*)out_w);
  }
  if (zero_pad) {
    hipLaunchKernelGGL(k_expand_zero_pad, dim3(c.experts_per_rank()), dim3(kEpWaves * kWarpSize), 0, s,
                       c, expert_alignment, (const int32_t*)counts, (const int32_t*)offsets,
                       use_fp8 ? 1 : (int)sizeof(bf16_t), (void*)out_x,
                       (float*)out_sf, sf_row_stride, sf_col_stride, (float*)out_w);
  }
  HC(hipStreamSynchronize(s));
  return 0;
}

// `grouped` selects what each rank writes into the window: 1 for one reduced
// row per received token, with this rank's owned rows summed here in ascending
// k; 0 for one row per (slot, k). In either mode a non-null `row_map` means the
// caller's input is the expanded layout and is addressed through it.
int ep_combine(void* handle,
               uintptr_t y, uintptr_t in_w, uintptr_t row_map,
               uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
               uintptr_t topk_idx, int num_tokens,
               uintptr_t bias0, uintptr_t bias1, int grouped,
               uintptr_t out, uintptr_t out_w, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_combine";
  hipStream_t s = (hipStream_t)stream;
  REQUIRE_CONFIGURED(h, fn);
  const EpConfig& c = h->cfg;
  // Same capacity bound ep_dispatch enforces: combine writes slot
  // rank * num_max_tokens_per_rank + t, and src_tok = g % T aliases two tokens
  // onto one slot once t runs past capacity.
  REQUIRE_TOKENS(h, fn, num_tokens);
  REQUIRE_RECV(h, fn, num_recv);
  REQUIRE_SMS(fn, num_sms);
  REQUIRE_PTR(fn, y, num_recv > 0);
  REQUIRE_PTR(fn, recv_topk, num_recv > 0);
  REQUIRE_PTR(fn, recv_src, num_recv > 0);
  REQUIRE_PTR(fn, topk_idx, num_tokens > 0);
  REQUIRE_PTR(fn, out, num_tokens > 0);
  grouped = grouped ? 1 : 0;
  // Two different grids. The transport is (num_ranks, per_rank) so a CTA owns
  // one peer; the reduce epilogue is purely local and 1-D over tokens, so it
  // takes the whole budget. Sharing one value silently starved the epilogue of
  // CTAs and cost it 7x.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);
  const int cta_reduce = sms_or(num_sms, 128);
  // An internode combine returns rows at the counts and plan the caller bound for its dispatch.
  REQUIRE_BOUND(h, fn);

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  WindowView* views = h->d_views;
  const int32_t* src = (const int32_t*)recv_src;
  if (h->inter) {
    views = h->d_views_inter + (grouped ? kViewCombGrouped : kViewCombFlat) * c.num_ranks;
    hipLaunchKernelGGL(k_inter_remap_src, dim3(num_recv > 0 ? (num_recv + 255) / 256 : 1), dim3(256), 0, s,
                       (const int32_t*)recv_src, num_recv, c.num_max_tokens_per_rank, h->bound, h->d_src_inter);
    src = h->d_src_inter;
  }
  hipLaunchKernelGGL(combine_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s, c, (const bf16_t*)y,
                     (const float*)in_w, (const int32_t*)row_map, (const int32_t*)recv_topk, src, num_recv, grouped,
                     h->self_view, views);
  // As in dispatch: the remote rows are in place before barrier B, on the same stream.
  if (h->inter && inter_combine(h, grouped, cta, s) != 0) return mark_broken(h, fn);
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(kEpBarrierCtas), dim3(kWarpSize), 0, s, h->devComm);
  hipLaunchKernelGGL(combine_reduce_epilogue_impl, dim3(cta_reduce), dim3(kEpWaves * kWarpSize), 0, s,
                     c, h->self_view, (const int32_t*)topk_idx, num_tokens,
                     (const bf16_t*)bias0, (const bf16_t*)bias1, grouped,
                     (bf16_t*)out, (float*)out_w);
  HC(hipStreamSynchronize(s));
  return 0;
}

// Internode count exchange for an ep_plan plan: all-gathers every rank's `sendc` (device
// int32 [num_ranks]) over the global communicator and writes [sendc, recvc] (host int32
// [2 * num_ranks]) to `out`. Synchronizes: each posted receive needs its size on the host.
int ep_inter_counts(void* handle, uintptr_t sendc, int32_t* out, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_inter_counts";
  REQUIRE_CONFIGURED(h, fn);
  if (!h->inter) return ep_fail(fn, "needs an internode handle (ep_create_nodes)");
  REQUIRE_PTR(fn, sendc, true);
  REQUIRE_PTR(fn, out, true);
  const int R = h->cfg.num_ranks, me = h->cfg.rank;
  hipStream_t s = (hipStream_t)stream;
  NC(ncclAllGather((const void*)sendc, h->d_scratch, R, ncclInt32, h->comm, s));
  HC(hipMemcpyAsync(h->host_scratch, h->d_scratch, (size_t)R * R * sizeof(int32_t), hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  for (int q = 0; q < R; ++q) {
    out[q] = h->host_scratch[me * R + q];
    out[R + q] = h->host_scratch[q * R + me];
  }
  return 0;
}

// Bind a dispatch's `sendc` / `recvc` (host int32 [num_ranks], from ep_inter_counts) and
// send_list (device int32 [num_ranks, num_tokens]) for the internode dispatch or combine
// that follows. Receives are posted at these sizes and the combine scatters by send_list,
// so a cached replay rebinds its handle's. Stays bound until replaced, or until ep_configure
// changes the shape, which drops it. A no-op on an intranode handle.
int ep_inter_bind(void* handle, const int32_t* sendc, const int32_t* recvc, uintptr_t send_list, int num_tokens) {
  auto* h = static_cast<EpHandle*>(handle);
  constexpr const char* fn = "ep_inter_bind";
  REQUIRE_CONFIGURED(h, fn);
  if (!h->inter) return 0;
  REQUIRE_TOKENS(h, fn, num_tokens);
  REQUIRE_PTR(fn, sendc, true);
  REQUIRE_PTR(fn, recvc, true);
  REQUIRE_PTR(fn, send_list, num_tokens > 0);
  const int R = h->cfg.num_ranks, T = h->cfg.num_max_tokens_per_rank;
  InterCounts ic{};
  int acc = 0;
  for (int r = 0; r < R; ++r) {
    // A plan sends a token at most once per destination, and the combine reads send_list's row r
    // up to sendc[r] entries, so a send count is bounded by this call's num_tokens.
    if (sendc[r] < 0 || sendc[r] > num_tokens) {
      return ep_fail(fn, "rank %d send count %d outside [0, num_tokens=%d]", r, sendc[r], num_tokens);
    }
    if (recvc[r] < 0 || recvc[r] > T) {
      return ep_fail(fn, "rank %d receive count %d outside [0, %d]", r, recvc[r], T);
    }
    ic.sendc[r] = sendc[r];
    ic.recvc[r] = recvc[r];
    ic.prefix[r] = acc;
    acc += recvc[r];
    ic.remote[r] = is_remote(h, r) ? 1 : 0;
  }
  h->bound = ic;
  h->bound_send_list = (const int32_t*)send_list;
  h->bound_num_tokens = num_tokens;
  h->is_bound = true;
  return 0;
}

}  // extern "C"
#pragma GCC visibility pop
