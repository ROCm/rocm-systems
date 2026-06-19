/*************************************************************************
 * bootstrap_net_probe — standalone reproducer for RCCL socket-bootstrap
 * variance, with kernel TCP_INFO capture.
 *
 * Purpose: isolate the TCP OOB bootstrap critical path (forward ring
 * connect/accept + small-message ring allgather) that the David10
 * instrumented analysis flagged as the source of bootstrap latency
 * variance, and attach kernel-level TCP counters (rtt, retransmits, rto,
 * cwnd, ca_state) to each measured iteration. This lets us prove whether
 * the variance is transport/network induced (RTT spikes, retransmits,
 * delayed-ACK stalls) rather than CPU/scheduling.
 *
 * No GPU / ROCm / MPI dependency. Rank/size are taken from SLURM env.
 * Rendezvous is done over a shared filesystem directory; all *measured*
 * phases are pure TCP on the chosen interface.
 *
 * Build:  cc -O2 -Wall -pthread -o bootstrap_net_probe bootstrap_net_probe.c
 * Launch: srun -N4 --ntasks-per-node=8 ./bootstrap_net_probe
 *
 * Knobs (env):
 *   PROBE_IFACE      network interface to bind (default eno8303)
 *   PROBE_ITERS      measured ring-allgather iterations (default 50)
 *   PROBE_WARMUP     warmup iterations, not recorded (default 3)
 *   PROBE_MSG_BYTES  bytes per ring step per direction (default 480)
 *   PROBE_NODELAY    set TCP_NODELAY 1/0 (default 1)
 *   PROBE_QUICKACK   set TCP_QUICKACK 1/0 each step (default 0)
 *   PROBE_OUT        output dir for per-rank CSV (default $HOME/.bnp_out)
 *   PROBE_RDV        rendezvous dir (default $HOME/.bnp_rdv/<jobid>)
 *   PROBE_TAG        label written into CSV (default "run")
 *************************************************************************/
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static int g_rank = 0, g_size = 1;
static int g_iter = -1;  // current measured iteration (for diagnostics)
static int g_busypoll = 0, g_sndbuf = 0, g_rcvbuf = 0;  // extra knobs

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// CPU/OS scheduling-jitter baseline: back-to-back monotonic-clock reads. The
// largest gaps are preemptions/IRQs — the noise floor below which we cannot
// attribute latency to the network. Returns p99 and max gap in microseconds.
static void cpu_jitter(int samples, double* p99_us, double* max_us) {
  static uint64_t* g = NULL;
  g = (uint64_t*)malloc(sizeof(uint64_t) * samples);
  uint64_t prev = now_ns();
  for (int i = 0; i < samples; ++i) {
    uint64_t n = now_ns();
    g[i] = n - prev;
    prev = n;
  }
  // simple insertion of max + p99 via partial sort
  for (int i = 0; i < samples; ++i)
    for (int j = i + 1; j < samples; ++j)
      if (g[j] < g[i]) { uint64_t t = g[i]; g[i] = g[j]; g[j] = t; }
  *p99_us = g[(int)(0.99 * samples)] / 1000.0;
  *max_us = g[samples - 1] / 1000.0;
  free(g);
}

static int env_int(const char* name, int def) {
  const char* v = getenv(name);
  return (v && v[0]) ? atoi(v) : def;
}
static const char* env_str(const char* name, const char* def) {
  const char* v = getenv(name);
  return (v && v[0]) ? v : def;
}

static void die(const char* msg) {
  fprintf(stderr, "[rank %d] FATAL: %s: %s\n", g_rank, msg, strerror(errno));
  exit(1);
}

// Resolve the IPv4 address of the named interface.
static int iface_ip(const char* iface, struct in_addr* out) {
  struct ifaddrs* ifap = NULL;
  if (getifaddrs(&ifap) != 0) return -1;
  int found = -1;
  for (struct ifaddrs* p = ifap; p; p = p->ifa_next) {
    if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
    if (strcmp(p->ifa_name, iface) != 0) continue;
    *out = ((struct sockaddr_in*)p->ifa_addr)->sin_addr;
    found = 0;
    break;
  }
  freeifaddrs(ifap);
  return found;
}

static void set_sockopts(int fd, int nodelay, int quickack) {
  int one = 1;
  if (nodelay) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (quickack) setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#ifdef SO_BUSY_POLL
  if (g_busypoll > 0) setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &g_busypoll, sizeof(g_busypoll));
#endif
  if (g_sndbuf > 0) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &g_sndbuf, sizeof(g_sndbuf));
  if (g_rcvbuf > 0) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &g_rcvbuf, sizeof(g_rcvbuf));
  // Bound every blocking op so a lost peer aborts with context instead of
  // hanging the whole allocation to the Slurm time limit.
  struct timeval tv = {20, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

// Returns 0 ok, -1 on error (errno set), -2 on clean EOF (peer closed).
static int recv_all(int fd, void* buf, int n) {
  char* p = (char*)buf;
  int got = 0;
  while (got < n) {
    int r = (int)recv(fd, p + got, n - got, 0);
    if (r == 0) return -2;
    if (r < 0) { if (errno == EINTR) continue; return -1; }
    got += r;
  }
  return 0;
}
static int send_all(int fd, const void* buf, int n) {
  const char* p = (const char*)buf;
  int sent = 0;
  while (sent < n) {
    int r = (int)send(fd, p + sent, n - sent, 0);
    if (r < 0) { if (errno == EINTR) continue; return -1; }
    sent += r;
  }
  return 0;
}

struct accept_arg { int lfd; int rfd; int nodelay; int quickack; };

// Blocking accept() of the single inbound (prev) connection, run on its own
// thread so it overlaps the main thread's blocking connect() to next.
void* accept_thread(void* a) {
  struct accept_arg* arg = (struct accept_arg*)a;
  int fd = accept(arg->lfd, NULL, NULL);
  arg->rfd = fd;
  return NULL;
}

// Like die(), but for a peer-context failure inside the measured loop. rc==-2
// means the peer closed the connection (EOF), which errno wouldn't describe.
static void die_ctx(const char* what, int peer, int rc) {
  if (rc == -2)
    fprintf(stderr, "[rank %d] FATAL: %s (peer=%d, iter=%d): peer closed connection (EOF)\n",
            g_rank, what, peer, g_iter);
  else
    fprintf(stderr, "[rank %d] FATAL: %s (peer=%d, iter=%d): %s\n",
            g_rank, what, peer, g_iter, strerror(errno));
  exit(1);
}

struct addr_rec { uint32_t ip; uint16_t port; };

// Atomic-ish shared-fs publish: write tmp then rename.
static void publish_addr(const char* rdv, int rank, struct addr_rec* a) {
  char tmp[1024], fin[1024];
  snprintf(tmp, sizeof(tmp), "%s/.tmp_rank_%d_%d", rdv, rank, (int)getpid());
  snprintf(fin, sizeof(fin), "%s/rank_%d", rdv, rank);
  FILE* f = fopen(tmp, "wb");
  if (!f) die("publish fopen");
  if (fwrite(a, sizeof(*a), 1, f) != 1) die("publish fwrite");
  fclose(f);
  if (rename(tmp, fin) != 0) die("publish rename");
}

static int try_read_addr(const char* rdv, int rank, struct addr_rec* a) {
  char fin[1024];
  snprintf(fin, sizeof(fin), "%s/rank_%d", rdv, rank);
  FILE* f = fopen(fin, "rb");
  if (!f) return -1;
  int ok = (fread(a, sizeof(*a), 1, f) == 1) ? 0 : -1;
  fclose(f);
  return ok;
}

// Wait until all peer address files are present (shared-fs barrier).
static void gather_addrs(const char* rdv, struct addr_rec* all) {
  for (int r = 0; r < g_size; ++r) {
    int waited_ms = 0;
    while (try_read_addr(rdv, r, &all[r]) != 0) {
      struct timespec ts = {0, 2 * 1000 * 1000};  // 2ms
      nanosleep(&ts, NULL);
      waited_ms += 2;
      if (waited_ms > 120000) { fprintf(stderr, "[rank %d] timeout waiting rank %d\n", g_rank, r); exit(1); }
    }
  }
}

struct tcp_snap {
  uint32_t rtt, rttvar, rto, snd_cwnd, total_retrans, lost, unacked, ca_state;
};
static void tcp_snapshot(int fd, struct tcp_snap* s) {
  struct tcp_info ti;
  socklen_t len = sizeof(ti);
  memset(&ti, 0, sizeof(ti));
  memset(s, 0, sizeof(*s));
  if (getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &len) != 0) return;
  s->rtt = ti.tcpi_rtt;
  s->rttvar = ti.tcpi_rttvar;
  s->rto = ti.tcpi_rto;
  s->snd_cwnd = ti.tcpi_snd_cwnd;
  s->total_retrans = ti.tcpi_total_retrans;
  s->lost = ti.tcpi_lost;
  s->unacked = ti.tcpi_unacked;
  s->ca_state = ti.tcpi_ca_state;
}

int main(void) {
  g_rank = env_int("SLURM_PROCID", 0);
  g_size = env_int("SLURM_NTASKS", 1);
  int local = env_int("SLURM_LOCALID", 0);
  if (g_size < 2) { fprintf(stderr, "need >=2 ranks (SLURM_NTASKS)\n"); return 1; }

  const char* iface = env_str("PROBE_IFACE", "eno8303");
  int iters = env_int("PROBE_ITERS", 50);
  int warmup = env_int("PROBE_WARMUP", 3);
  int msg = env_int("PROBE_MSG_BYTES", 480);
  int nodelay = env_int("PROBE_NODELAY", 1);
  int quickack = env_int("PROBE_QUICKACK", 0);
  g_busypoll = env_int("PROBE_BUSYPOLL", 0);
  g_sndbuf = env_int("PROBE_SNDBUF", 0);
  g_rcvbuf = env_int("PROBE_RCVBUF", 0);
  const char* tag = env_str("PROBE_TAG", "run");

  // CPU/OS noise floor (all ranks measure; rank 0 reports). Establishes the
  // baseline jitter that is NOT network-attributable.
  { double j99, jmax; cpu_jitter(200000, &j99, &jmax);
    fprintf(stderr, "[rank %d] cpu_jitter p99=%.1fus max=%.1fus\n", g_rank, j99, jmax); }
  const char* home = env_str("HOME", "/tmp");
  const char* jobid = env_str("SLURM_JOB_ID", "0");

  char rdv_def[1024], out_def[1024];
  snprintf(rdv_def, sizeof(rdv_def), "%s/.bnp_rdv/%s", home, jobid);
  snprintf(out_def, sizeof(out_def), "%s/.bnp_out", home);
  const char* rdv = env_str("PROBE_RDV", rdv_def);
  const char* out = env_str("PROBE_OUT", out_def);

  if (g_rank == 0) {
    char mk[1100];
    snprintf(mk, sizeof(mk), "%s/.bnp_rdv", home); mkdir(mk, 0755);
    mkdir(rdv, 0755);
    mkdir(out, 0755);
  }

  struct in_addr ip;
  if (iface_ip(iface, &ip) != 0) { fprintf(stderr, "[rank %d] iface %s not found\n", g_rank, iface); return 1; }

  // Listen socket for the ring (accept from prev).
  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  if (lfd < 0) die("listen socket");
  int one = 1; setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in la; memset(&la, 0, sizeof(la));
  // Bind to INADDR_ANY so same-node peers can reach us over loopback and
  // cross-node peers over the chosen interface IP. (On this cluster, same-node
  // traffic to the interface IP is dropped by rp_filter=2 with multiple NICs;
  // routing same-node links over loopback sidesteps that and keeps intra-node
  // bootstrap links local, as in the real stack.)
  la.sin_family = AF_INET; la.sin_addr.s_addr = htonl(INADDR_ANY); la.sin_port = 0;
  if (bind(lfd, (struct sockaddr*)&la, sizeof(la)) != 0) die("bind");
  if (listen(lfd, 64) != 0) die("listen");
  socklen_t alen = sizeof(la);
  if (getsockname(lfd, (struct sockaddr*)&la, &alen) != 0) die("getsockname");

  struct addr_rec mine = { (uint32_t)ip.s_addr, la.sin_port };

  // Rank 0 ensures dirs exist before anyone publishes.
  if (g_rank != 0) {
    struct stat st; int waited = 0;
    while (stat(rdv, &st) != 0) { struct timespec ts={0,2000000}; nanosleep(&ts,NULL); if(++waited>60000){fprintf(stderr,"rdv missing\n");return 1;} }
  }
  publish_addr(rdv, g_rank, &mine);
  struct addr_rec* all = (struct addr_rec*)calloc(g_size, sizeof(struct addr_rec));
  gather_addrs(rdv, all);

  int next = (g_rank + 1) % g_size;
  // prev connects to us; we accept on lfd (no explicit prev address needed).

  // === forward ring connect/accept (measured once) ===
  // Robust, deadlock-free setup: a dedicated thread runs the blocking accept()
  // for the inbound connection from prev, while the main thread does a blocking
  // connect() to next (with retries on ECONNREFUSED in case next hasn't called
  // listen() yet). This avoids the non-blocking-connect / select dance whose
  // half-open edge cases caused same-node deliveries to stall.
  uint64_t t0 = now_ns();
  struct accept_arg aarg;
  aarg.lfd = lfd; aarg.rfd = -1; aarg.nodelay = nodelay; aarg.quickack = quickack;
  pthread_t athr;
  if (pthread_create(&athr, NULL, accept_thread, &aarg) != 0) die("pthread_create");

  uint64_t tc0 = now_ns();
  int sfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sfd < 0) die("send socket");
  struct sockaddr_in na; memset(&na, 0, sizeof(na));
  na.sin_family = AF_INET;
  // Same-node next (same iface IP) → connect over loopback to avoid the
  // rp_filter same-IP drop; cross-node next → connect over the interface IP.
  na.sin_addr.s_addr = (all[next].ip == mine.ip) ? htonl(INADDR_LOOPBACK) : all[next].ip;
  na.sin_port = all[next].port;
  int connected = 0, attempts = 0;
  while (!connected) {
    if (connect(sfd, (struct sockaddr*)&na, sizeof(na)) == 0) { connected = 1; break; }
    if (errno == ECONNREFUSED || errno == EAGAIN || errno == ETIMEDOUT) {
      if (++attempts > 50000) die("connect retries exhausted");
      struct timespec ts = {0, 1 * 1000 * 1000}; nanosleep(&ts, NULL);  // 1ms
      close(sfd);
      sfd = socket(AF_INET, SOCK_STREAM, 0);
      if (sfd < 0) die("send socket retry");
      continue;
    }
    die("connect");
  }
  set_sockopts(sfd, nodelay, quickack);
  uint64_t ta0 = now_ns();
  if (pthread_join(athr, NULL) != 0) die("pthread_join");
  int rfd = aarg.rfd;
  if (rfd < 0) die("accept (thread)");
  set_sockopts(rfd, nodelay, quickack);
  uint64_t ta1 = now_ns();
  uint64_t tready = now_ns();

  if (env_int("PROBE_DEBUG", 0)) {
    struct sockaddr_in pa; socklen_t pl = sizeof(pa);
    char nb[64] = "?", sp[64] = "?", rp[64] = "?";
    inet_ntop(AF_INET, &na.sin_addr, nb, sizeof(nb));
    if (getpeername(sfd, (struct sockaddr*)&pa, &pl) == 0) snprintf(sp, sizeof(sp), "%s:%d", inet_ntoa(pa.sin_addr), ntohs(pa.sin_port));
    pl = sizeof(pa);
    if (getpeername(rfd, (struct sockaddr*)&pa, &pl) == 0) snprintf(rp, sizeof(rp), "%s:%d", inet_ntoa(pa.sin_addr), ntohs(pa.sin_port));
    fprintf(stderr, "[rank %d] next=%d want=%s:%d sfd_peer=%s rfd_peer=%s\n",
            g_rank, next, nb, ntohs(na.sin_port), sp, rp);
    fflush(stderr);
  }

  // Identity handshake: confirm the ring is wired as expected (rfd really is
  // prev, sfd really is next). Catches stale-rendezvous / scrambled-ring bugs
  // immediately with a clear message instead of a downstream deadlock.
  {
    int prev = (g_rank - 1 + g_size) % g_size;
    int myid = g_rank, gotid = -1, rc;
    int dbg = env_int("PROBE_DEBUG", 0);
    if (dbg) { fprintf(stderr, "[rank %d] setup ok sfd=%d rfd=%d, handshake start\n", g_rank, sfd, rfd); fflush(stderr); }
    if (g_rank == 0) {
      if (send_all(sfd, &myid, sizeof(myid)) != 0) die_ctx("id send", next, -1);
      if (dbg) { fprintf(stderr, "[rank %d] sent id to next=%d\n", g_rank, next); fflush(stderr); }
      if ((rc = recv_all(rfd, &gotid, sizeof(gotid))) != 0) die_ctx("id recv", prev, rc);
    } else {
      if ((rc = recv_all(rfd, &gotid, sizeof(gotid))) != 0) die_ctx("id recv", prev, rc);
      if (send_all(sfd, &myid, sizeof(myid)) != 0) die_ctx("id send", next, -1);
    }
    if (gotid != prev) {
      fprintf(stderr, "[rank %d] FATAL: ring scrambled: expected prev=%d got=%d\n", g_rank, prev, gotid);
      exit(1);
    }
  }

  double connect_us = (ta0 - tc0) / 1000.0;
  double accept_us = (ta1 - ta0) / 1000.0;
  double ready_us = (tready - t0) / 1000.0;

  // Per-rank CSV.
  char path[1200];
  snprintf(path, sizeof(path), "%s/rank_%05d.csv", out, g_rank);
  FILE* csv = fopen(path, "w");
  if (!csv) die("csv fopen");
  fprintf(csv, "tag,rank,size,local,iface,nodelay,quickack,msg,iter,phase,"
               "allgather_us,max_step_us,connect_us,accept_us,ready_us,"
               "rtt_us,rttvar_us,rto_us,snd_cwnd,retrans_delta,lost,ca_state\n");
  fprintf(csv, "%s,%d,%d,%d,%s,%d,%d,%d,-1,setup,0,0,%.1f,%.1f,%.1f,0,0,0,0,0,0,0\n",
          tag, g_rank, g_size, local, iface, nodelay, quickack, msg,
          connect_us, accept_us, ready_us);
  fflush(csv);  // ensure setup row survives even if a later phase aborts

  char* sbuf = (char*)malloc(msg);
  char* rbuf = (char*)malloc(msg);
  memset(sbuf, g_rank & 0xff, msg);

  struct tcp_snap snap_prev, snap_cur;
  tcp_snapshot(sfd, &snap_prev);

  int prev = (g_rank - 1 + g_size) % g_size;
  // Ring barrier helper over the established ring: token passes around once.
  // rank 0 sends, waits to receive back; others recv then send.
  #define RING_BARRIER() do { \
    char b = 0; int rc; \
    if (g_rank == 0) { if (send_all(sfd, &b, 1) != 0) die_ctx("barrier send", next, -1); \
                       if ((rc = recv_all(rfd, &b, 1)) != 0) die_ctx("barrier recv", prev, rc); } \
    else { if ((rc = recv_all(rfd, &b, 1)) != 0) die_ctx("barrier recv", prev, rc); \
           if (send_all(sfd, &b, 1) != 0) die_ctx("barrier send", next, -1); } \
  } while (0)

  // Per-step latency buckets accumulated over STEADY iterations (warmup excluded):
  // <1ms, 1-5, 5-35, 35-60(dACK~40), 60-150, 150-700(RTO), >700ms.
  long bkt[7] = {0};
  double first_step_sum = 0; int first_step_n = 0;     // slow-start probe
  double rest_step_sum = 0; long rest_step_n = 0;

  int total = warmup + iters;
  const char* mode = env_str("PROBE_MODE", "ring");
  if (strcmp(mode, "dack") == 0) {
    // Delayed-ACK provocation: even rank = initiator (sends TWO small msgs then
    // waits a 1-byte app-ack); odd rank = responder (recvs both, then acks).
    // With Nagle on (NODELAY=0) the 2nd send is held until the responder TCP-ACKs
    // the 1st, and the responder delays that ACK ~40ms -> ~40ms round time.
    // TCP_QUICKACK on the responder, or TCP_NODELAY on the initiator, defeats it.
    int is_init = (g_rank % 2) == 0;
    char ack = 1;
    for (int it = 0; it < total; ++it) {
      g_iter = it - warmup;
      RING_BARRIER();
      uint64_t s0 = now_ns();
      int rc;
      if (is_init) {
        if (send_all(sfd, sbuf, msg) != 0) die_ctx("dack send1", next, -1);
        if (send_all(sfd, sbuf, msg) != 0) die_ctx("dack send2", next, -1);
        if ((rc = recv_all(sfd, &ack, 1)) != 0) die_ctx("dack ack", next, rc);
      } else {
        if (quickack) { int q=1; setsockopt(rfd, IPPROTO_TCP, TCP_QUICKACK, &q, sizeof(q)); }
        if ((rc = recv_all(rfd, rbuf, msg)) != 0) die_ctx("dack r1", prev, rc);
        if ((rc = recv_all(rfd, rbuf, msg)) != 0) die_ctx("dack r2", prev, rc);
        if (send_all(rfd, &ack, 1) != 0) die_ctx("dack ack send", prev, -1);
      }
      double dt = (now_ns() - s0) / 1000.0;
      if (is_init && it >= warmup) {
        double m = dt / 1000.0;
        int b = m<1?0 : m<5?1 : m<35?2 : m<60?3 : m<150?4 : m<700?5 : 6;
        bkt[b]++;
        fprintf(csv, "%s,%d,%d,%d,%s,%d,%d,%d,%d,iter,%.1f,%.1f,0,0,0,0,0,0,0,0,0,0\n",
                tag, g_rank, g_size, local, iface, nodelay, quickack, msg, it - warmup, dt, dt);
      }
    }
  } else
  for (int it = 0; it < total; ++it) {
    g_iter = it - warmup;
    RING_BARRIER();
    uint64_t it0 = now_ns();
    double max_step = 0;
    int rc;
    // ring allgather: size-1 steps, each send to next / recv from prev (msg bytes/dir)
    for (int step = 0; step < g_size - 1; ++step) {
      uint64_t s0 = now_ns();
      if (g_rank == 0) {
        if (send_all(sfd, sbuf, msg) != 0) die_ctx("send step", next, -1);
        if ((rc = recv_all(rfd, rbuf, msg)) != 0) die_ctx("recv step", prev, rc);
      } else {
        if ((rc = recv_all(rfd, rbuf, msg)) != 0) die_ctx("recv step", prev, rc);
        if (send_all(sfd, sbuf, msg) != 0) die_ctx("send step", next, -1);
      }
      if (quickack) { int q=1; setsockopt(rfd, IPPROTO_TCP, TCP_QUICKACK, &q, sizeof(q)); }
      double step_us = (now_ns() - s0) / 1000.0;
      if (step_us > max_step) max_step = step_us;
      if (it >= warmup) {
        double m = step_us / 1000.0;  // ms
        int b = m<1?0 : m<5?1 : m<35?2 : m<60?3 : m<150?4 : m<700?5 : 6;
        bkt[b]++;
        if (step == 0) { first_step_sum += step_us; first_step_n++; }
        else { rest_step_sum += step_us; rest_step_n++; }
      }
    }
    double ag_us = (now_ns() - it0) / 1000.0;
    tcp_snapshot(sfd, &snap_cur);
    uint32_t retrans_delta = snap_cur.total_retrans - snap_prev.total_retrans;
    if (it >= warmup) {
      fprintf(csv, "%s,%d,%d,%d,%s,%d,%d,%d,%d,iter,%.1f,%.1f,0,0,0,%u,%u,%u,%u,%u,%u,%u\n",
              tag, g_rank, g_size, local, iface, nodelay, quickack, msg, it - warmup,
              ag_us, max_step,
              snap_cur.rtt, snap_cur.rttvar, snap_cur.rto, snap_cur.snd_cwnd,
              retrans_delta, snap_cur.lost, snap_cur.ca_state);
    }
    snap_prev = snap_cur;
    if ((it & 0x1f) == 0) fflush(csv);
  }
  // Per-step distribution + slow-start (first step vs rest) for this rank.
  fprintf(csv, "%s,%d,%d,%d,%s,%d,%d,%d,-2,buckets,%ld,%ld,%ld,%ld,%ld,%ld,%ld,0,0,0,0,0,0\n",
          tag, g_rank, g_size, local, iface, nodelay, quickack, msg,
          bkt[0],bkt[1],bkt[2],bkt[3],bkt[4],bkt[5],bkt[6]);
  fprintf(csv, "%s,%d,%d,%d,%s,%d,%d,%d,-3,slowstart,%.1f,%.1f,0,0,0,0,0,0,0,0,0,0\n",
          tag, g_rank, g_size, local, iface, nodelay, quickack, msg,
          first_step_n?first_step_sum/first_step_n:0.0,
          rest_step_n?rest_step_sum/rest_step_n:0.0);
  fclose(csv);
  RING_BARRIER();
  if (g_rank == 0) fprintf(stderr, "[probe] done tag=%s size=%d iters=%d nodelay=%d quickack=%d msg=%d busypoll=%d sndbuf=%d rcvbuf=%d\n",
                           tag, g_size, iters, nodelay, quickack, msg, g_busypoll, g_sndbuf, g_rcvbuf);
  close(sfd); close(rfd); close(lfd);
  free(sbuf); free(rbuf); free(all);
  return 0;
}
