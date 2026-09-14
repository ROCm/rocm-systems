/* SPDX-License-Identifier: MIT */
/*
 * Copyright (c) OdinLink-Five contributors
 * https://github.com/Geramy/OdinLink-Five
 *
 * OdinLink Thunderbolt 5 - NCCL/RCCL Net Plugin
 *
 * Implements the NCCL net plugin ABI (versions v6 through v12) backed by the
 * OdinLink TB5 stream API. Uses the host-pointer DMA path
 * (odl_tb5_stream_send / odl_tb5_stream_recv); RCCL stages GPU buffers
 * to host before calling us (ptrSupport = NCCL_PTR_HOST).
 *
 * Each (sendComm, recvComm) carries a single TB5 stream. Per-comm worker
 * threads pump the blocking stream ioctls so isend/irecv return promptly
 * and avoid the bidirectional deadlock that would result from calling
 * the blocking ioctl directly on RCCL's proxy thread.
 *
 * Stats are exported at /run/odl_tb5/rccl_stats for the daemon.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>

#include "net.h"
#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_rccl_stats.h>

#define ODL_MAX_DEVICES        16
#define ODL_HANDLE_MAGIC       0x4F444C00u  /* 'O','D','L',0 */
#define ODL_HANDLE_VERSION     1
#define ODL_WAIT_PEER_MS       10000
/*
 * Windowed ACK on the reverse of the data stream. The 1-byte 0xA5 ACK used
 * to be sent to recvComm->dst_id, but 2-node (one USB4 cable, nports==1)
 * skipped the connect handshake and left dst_id=0, so the sender blocked
 * in wait_rx forever. Handshake now always carries the sender stream id;
 * ACK is a 4-byte magic on that stream so 2-node and 4-node share one path.
 */
#define ODL_ACK_MIN_MESSAGE    (64 * 1024)
#define ODL_ACK_WINDOW_BYTES   (4 * 1024 * 1024)
#define ODL_ACK_HDR           0x4F444C41u  /* 'O','D','L','A' */
/* Stay below both the vendored and protocol-v2 4096-frame message limits. */
#define ODL_WIRE_CHUNK_BYTES   (8 * 1024 * 1024)
#define ODL_FORCE_DEV_ENV      "ODL_TB5_FORCE_DEV_INDEX"
#define ODL_PROFILE_ENV        "ODL_TB5_PROFILE"
#define ODL_PROFILE_INTERVAL_ENV "ODL_TB5_PROFILE_INTERVAL"
#define ODL_RECV_TIMEOUT_ENV   "ODL_TB5_RECV_TIMEOUT_MS"
#define ODL_VIRTUAL_DEV_ENV    "ODL_TB5_VIRTUAL_SINGLE_DEV"
#define ODL_HANDLE_MAX_PORTS   4
#define ODL_CONNECT_MAGIC      0x4F444C43u  /* 'O','D','L','C' */
#define ODL_NET_NAME           "OdinLink_TB5"
#define ODL_LINK_SPEED_MBPS    20000   /* TB4/USB4 v1 negotiated rate */
#define ODL_NET_LATENCY_US     20.0f   /* CLI ping-pong RTT/2 ~ 6us; round up */
#define ODL_EXPORT             __attribute__((visibility("default")))

/* Debug tracing to stderr — toggled by ODL_TB5_TRACE env var. */
static int g_trace;
#define TRACE(fmt, ...) do { if (g_trace) \
	fprintf(stderr, "[odl_tb5] " fmt "\n", ##__VA_ARGS__); } while (0)

static int env_flag_enabled(const char *name)
{
	const char *value = getenv(name);

	if (!value || strcmp(value, "0") == 0 ||
	    strcasecmp(value, "false") == 0 ||
	    strcasecmp(value, "off") == 0 ||
	    strcasecmp(value, "no") == 0)
		return 0;
	return 1;
}

static int g_profile;
static uint64_t g_profile_interval = 1024;
static int g_virtual_single_dev = 1;
static uint32_t g_recv_timeout_ms = 30000;

struct profile_counters {
	uint64_t submit_ops;
	uint64_t submit_block_ns;
	uint64_t worker_ops;
	uint64_t bytes;
	uint64_t hdr_ops;
	uint64_t hdr_ns;
	uint64_t pay_ops;
	uint64_t pay_ns;
	uint64_t test_ops;
	uint64_t test_pending;
	uint64_t max_size;
};

static struct profile_counters g_prof_tx;
static struct profile_counters g_prof_rx;

struct odl_port {
	int                 dev_index;   /* kernel /dev/odl_tb5_N index */
	char                path[64];
	char                name[64];
	char                pci_path[96];
	uint64_t            guid;
	uint8_t             local_uuid[16];
	uint8_t             peer_uuid[16];
	struct odl_tb5_link_info info;
	odl_tb5_t           handle;
	int                 refs;
};

/* Discovered TB5 links. RCCL dev ids index this compact port table. */
static struct odl_port g_ports[ODL_MAX_DEVICES];
static int             g_num_devices;

/* Stats shared-memory region (populated by stats_init). */
static struct odl_rccl_stats *g_stats;
static int                    g_stats_fd = -1;

static pthread_mutex_t  g_port_lock = PTHREAD_MUTEX_INITIALIZER;

/* Wire format for the handle blob exchanged between listen() and connect(). */
struct odl_handle_port {
	int32_t  dev_id;
	uint8_t  listener_uuid[16];
	uint8_t  stream_id;
	uint8_t  pad[3];
};

struct odl_handle {
	uint32_t magic;
	uint32_t version;
	uint32_t nports;
	uint64_t guid;
	struct odl_handle_port ports[ODL_HANDLE_MAX_PORTS];
};

struct odl_connect_msg {
	uint32_t magic;
	uint32_t version;
	uint8_t  src_stream; /* sender local stream; ACK reverse path */
	uint8_t  pad[3];
};

struct odl_comm;

/* Shared work-item between proxy thread (producer) and worker (consumer). */
struct odl_request {
	atomic_bool done;
	int         size;     /* bytes sent / received, -1 on error */
	bool        is_send;
	struct odl_comm *comm;
	int         pool_idx;
};

/* One pending op carried into the worker. */
struct odl_job {
	struct odl_request *req;
	void               *data;
	int                 size;
};

/* Queue depth must accommodate NCCL_NET_MAX_REQUESTS (=32) outstanding ops
 * per comm; otherwise comm_submit blocks the proxy thread and deadlocks
 * with the worker stuck in a blocking ioctl. */
#define ODL_JOB_QUEUE_DEPTH 64
#define ODL_REQ_POOL_DEPTH 64

/* One per send-comm or recv-comm. */
struct odl_comm {
	odl_tb5_t       handle;       /* shared device, ref held */
	int             dev_id;
	uint8_t         stream_id;    /* local stream */
	uint8_t         dst_id;       /* peer stream id for data or ACKs */
	bool            is_send;
	bool            stream_owned; /* true => closeXxx must close stream */
	uint8_t         peer_stream; /* last observed peer stream (ACK dest) */

	pthread_t       worker;
	pthread_mutex_t mu;
	pthread_cond_t  cv;
	bool            shutdown;
	uint64_t        ack_bytes;    /* worker-owned flow-control counter */

	/* Ring buffer of pending jobs. Producer = proxy thread (isend/irecv),
	 * consumer = worker thread. comm_submit blocks only if the ring is
	 * full, which under NCCL's ABI cap of 32 outstanding reqs cannot happen
	 * with depth 64. */
	struct odl_job  q[ODL_JOB_QUEUE_DEPTH];
	unsigned        q_head; /* next slot to consume */
	unsigned        q_tail; /* next slot to produce */

	struct odl_request req_pool[ODL_REQ_POOL_DEPTH];
	bool               req_used[ODL_REQ_POOL_DEPTH];
};

/* Listen state: holds an opened stream that accept() inherits into recvComm. */
struct odl_listen_port {
	int      dev_id;
	odl_tb5_t handle;
	uint8_t  stream_id;
	bool     stream_owned;  /* true until accept() takes ownership */
	pthread_t probe_thread;
	bool     probe_started;
};

struct odl_listen {
	int      nports;
	struct odl_listen_port ports[ODL_HANDLE_MAX_PORTS];
	pthread_mutex_t mu;
	pthread_cond_t  cv;
	bool     accepted;
	int      accepted_port;
	uint8_t  accepted_src;
	int      finished_probes;
};

/* ── Stats ────────────────────────────────────────────────────────────── */

static uint64_t mono_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void prof_add_u64(uint64_t *p, uint64_t v)
{
	__atomic_add_fetch(p, v, __ATOMIC_RELAXED);
}

static uint64_t prof_load_u64(uint64_t *p)
{
	return __atomic_load_n(p, __ATOMIC_RELAXED);
}

static void prof_max_u64(uint64_t *p, uint64_t v)
{
	uint64_t old = prof_load_u64(p);
	while (v > old &&
	       !__atomic_compare_exchange_n(p, &old, v, true,
					    __ATOMIC_RELAXED,
					    __ATOMIC_RELAXED)) {
	}
}

static double prof_avg_us(uint64_t ns, uint64_t ops)
{
	return (double)ns / (double)(ops ? ops : 1) / 1000.0;
}

static void prof_print_if_needed(const char *reason)
{
	uint64_t tx_ops, rx_ops, total_ops;
	uint64_t tx_hdr_ops, tx_pay_ops, rx_hdr_ops, rx_pay_ops;

	if (!g_profile)
		return;
	tx_ops = prof_load_u64(&g_prof_tx.worker_ops);
	rx_ops = prof_load_u64(&g_prof_rx.worker_ops);
	total_ops = tx_ops + rx_ops;
	if (total_ops == 0 || total_ops % g_profile_interval != 0)
		return;

	tx_hdr_ops = prof_load_u64(&g_prof_tx.hdr_ops);
	tx_pay_ops = prof_load_u64(&g_prof_tx.pay_ops);
	rx_hdr_ops = prof_load_u64(&g_prof_rx.hdr_ops);
	rx_pay_ops = prof_load_u64(&g_prof_rx.pay_ops);
	fprintf(stderr,
		"[odl_tb5_profile] reason=%s "
		"tx_ops=%llu tx_bytes=%llu tx_submit_avg_us=%.1f "
		"tx_hdr_avg_us=%.1f tx_pay_avg_us=%.1f tx_max=%llu "
		"rx_ops=%llu rx_bytes=%llu rx_submit_avg_us=%.1f "
		"rx_hdr_avg_us=%.1f rx_pay_avg_us=%.1f rx_max=%llu "
		"test_ops=%llu test_pending=%llu\n",
		reason,
		(unsigned long long)tx_ops,
		(unsigned long long)prof_load_u64(&g_prof_tx.bytes),
		prof_avg_us(prof_load_u64(&g_prof_tx.submit_block_ns),
			    prof_load_u64(&g_prof_tx.submit_ops)),
		prof_avg_us(prof_load_u64(&g_prof_tx.hdr_ns), tx_hdr_ops),
		prof_avg_us(prof_load_u64(&g_prof_tx.pay_ns), tx_pay_ops),
		(unsigned long long)prof_load_u64(&g_prof_tx.max_size),
		(unsigned long long)rx_ops,
		(unsigned long long)prof_load_u64(&g_prof_rx.bytes),
		prof_avg_us(prof_load_u64(&g_prof_rx.submit_block_ns),
			    prof_load_u64(&g_prof_rx.submit_ops)),
		prof_avg_us(prof_load_u64(&g_prof_rx.hdr_ns), rx_hdr_ops),
		prof_avg_us(prof_load_u64(&g_prof_rx.pay_ns), rx_pay_ops),
		(unsigned long long)prof_load_u64(&g_prof_rx.max_size),
		(unsigned long long)(prof_load_u64(&g_prof_tx.test_ops) +
				     prof_load_u64(&g_prof_rx.test_ops)),
		(unsigned long long)(prof_load_u64(&g_prof_tx.test_pending) +
				     prof_load_u64(&g_prof_rx.test_pending)));
}

static void stats_init(void)
{
	mkdir(ODL_RCCL_STATS_DIR, 0755);
	g_stats_fd = open(ODL_RCCL_STATS_PATH, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (g_stats_fd < 0)
		return;
	if (ftruncate(g_stats_fd, sizeof(*g_stats)) < 0) {
		close(g_stats_fd);
		g_stats_fd = -1;
		return;
	}
	g_stats = mmap(NULL, sizeof(*g_stats), PROT_READ | PROT_WRITE,
				MAP_SHARED, g_stats_fd, 0);
	if (g_stats == MAP_FAILED) {
		g_stats = NULL;
		close(g_stats_fd);
		g_stats_fd = -1;
		return;
	}
	memset(g_stats, 0, sizeof(*g_stats));
	g_stats->magic = ODL_RCCL_STATS_MAGIC;
	g_stats->version = ODL_RCCL_STATS_VERSION;
	g_stats->start_time_ns = mono_ns();
	__atomic_store_n(&g_stats->active, 1, __ATOMIC_RELEASE);
}

static void stats_cleanup(void)
{
	if (g_stats) {
		__atomic_store_n(&g_stats->active, 0, __ATOMIC_RELEASE);		munmap(g_stats, sizeof(*g_stats));		g_stats = NULL;	}
	if (g_stats_fd >= 0) {
		close(g_stats_fd);		g_stats_fd = -1;	}
}

static void stats_tx(int n)
{
	if (!g_stats || n <= 0) return;
	__atomic_add_fetch(&g_stats->tx_bytes, (uint64_t)n, __ATOMIC_RELAXED);
	__atomic_add_fetch(&g_stats->tx_ops, 1, __ATOMIC_RELAXED);
	__atomic_store_n(&g_stats->last_update_ns, mono_ns(), __ATOMIC_RELAXED);
}

static void stats_rx(int n)
{
	if (!g_stats || n <= 0) return;
	__atomic_add_fetch(&g_stats->rx_bytes, (uint64_t)n, __ATOMIC_RELAXED);
	__atomic_add_fetch(&g_stats->rx_ops, 1, __ATOMIC_RELAXED);
	__atomic_store_n(&g_stats->last_update_ns, mono_ns(), __ATOMIC_RELAXED);
}

/* ── Port discovery / device handles ─────────────────────────────────── */

static bool uuid_is_zero(const uint8_t uuid[16])
{
	int i;

	for (i = 0; i < 16; i++) {
		if (uuid[i] != 0)			return false;	}
	return true;
}

static bool uuid_equal_bytes(const uint8_t a[16], const uint8_t b[16])
{
	return memcmp(a, b, 16) == 0;
}

static void uuid_to_str(const uint8_t uuid[16], char *buf, size_t len)
{
	snprintf(buf, len,
		"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",		uuid[0], uuid[1], uuid[2], uuid[3],		uuid[4], uuid[5], uuid[6], uuid[7],		uuid[8], uuid[9], uuid[10], uuid[11],		uuid[12], uuid[13], uuid[14], uuid[15]);}

static uint64_t hash_uuid_pair(const uint8_t local[16], const uint8_t peer[16],
				int dev_index)
{
	uint64_t h = 1469598103934665603ULL;
	int i;

	for (i = 0; i < 16; i++) {
		h ^= local[i];		h *= 1099511628211ULL;	}
	for (i = 0; i < 16; i++) {
		h ^= peer[i];		h *= 1099511628211ULL;	}
	h ^= (uint64_t)(uint32_t)dev_index;
	h *= 1099511628211ULL;
	return h ? h : 1;
}

static int refresh_ports_locked(void)
{
	struct odl_port new_ports[ODL_MAX_DEVICES];
	int count = 0;
	int i;

	memset(new_ports, 0, sizeof(new_ports));
	for (i = 0; i < ODL_MAX_DEVICES; i++) {
		odl_tb5_t handle = NULL;		struct odl_tb5_link_info info;		int rc;
		snprintf(new_ports[count].path, sizeof(new_ports[count].path),			"/dev/odl_tb5_%d", i);		if (access(new_ports[count].path, F_OK) != 0)			continue;		if (access(new_ports[count].path, R_OK | W_OK) != 0) {			TRACE("devices skip %s: no rw permission (%s)",			new_ports[count].path, strerror(errno));			continue;		}
		rc = odl_tb5_open(&handle, i);		if (rc < 0) {			TRACE("devices skip %s: open rc=%d (%s)",			new_ports[count].path, rc, strerror(-rc));			continue;		}		memset(&info, 0, sizeof(info));		rc = odl_tb5_get_link_info(handle, &info);		odl_tb5_close(handle);		if (rc < 0) {			TRACE("devices skip %s: get_link_info rc=%d (%s)",			new_ports[count].path, rc, strerror(-rc));			continue;		}		if (info.state != ODL_TB5_STATE_READY) {			TRACE("devices skip %s: state=%u not ready",			new_ports[count].path, info.state);			continue;		}		if (uuid_is_zero(info.local_uuid) || uuid_is_zero(info.peer_uuid)) {			TRACE("devices skip %s: missing UUIDs", new_ports[count].path);			continue;		}
		new_ports[count].dev_index = i;		new_ports[count].info = info;		memcpy(new_ports[count].local_uuid, info.local_uuid, 16);		memcpy(new_ports[count].peer_uuid, info.peer_uuid, 16);		snprintf(new_ports[count].name, sizeof(new_ports[count].name),			"OdinLink_TB5_%d", count);		snprintf(new_ports[count].pci_path, sizeof(new_ports[count].pci_path),			"/sys/class/odl_tb5/odl_tb5_%d", i);		new_ports[count].guid = hash_uuid_pair(info.local_uuid,							info.peer_uuid, i);
		for (int old = 0; old < g_num_devices; old++) {			if (g_ports[old].dev_index == i) {				new_ports[count].handle = g_ports[old].handle;				new_ports[count].refs = g_ports[old].refs;				g_ports[old].handle = NULL;				g_ports[old].refs = 0;				break;			}		}
		TRACE("devices port=%d dev=%d guid=0x%llx peer=%02x%02x%02x%02x...",		count, i, (unsigned long long)new_ports[count].guid,		info.peer_uuid[0], info.peer_uuid[1],		info.peer_uuid[2], info.peer_uuid[3]);		count++;	}

	for (i = 0; i < ODL_MAX_DEVICES; i++) {
		if (g_ports[i].handle && g_ports[i].refs == 0)			odl_tb5_close(g_ports[i].handle);	}
	memcpy(g_ports, new_ports, sizeof(g_ports));
	return count;
}

static int device_acquire(int dev_id, odl_tb5_t *out)
{
	int rc;
	struct odl_port *p;

	pthread_mutex_lock(&g_port_lock);
	if (dev_id < 0 || dev_id >= g_num_devices) {
		pthread_mutex_unlock(&g_port_lock);		return -EINVAL;	}
	p = &g_ports[dev_id];
	if (!p->handle) {
		rc = odl_tb5_open(&p->handle, p->dev_index);		if (rc < 0) {			TRACE("device_acquire: open dev=%d rc=%d (%s)",				dev_id, rc, strerror(-rc));			p->handle = NULL;			pthread_mutex_unlock(&g_port_lock);			return rc;		}		TRACE("device_acquire: opened port=%d dev=%d", dev_id, p->dev_index);		rc = odl_tb5_wait_peer(p->handle, ODL_WAIT_PEER_MS);		if (rc < 0) {			TRACE("device_acquire: wait_peer rc=%d (%s)",				rc, strerror(-rc));			odl_tb5_close(p->handle);			p->handle = NULL;			pthread_mutex_unlock(&g_port_lock);			return rc;		}	}
	p->refs++;
	 *out = p->handle;
	pthread_mutex_unlock(&g_port_lock);
	return 0;
}

static void device_release(int dev_id)
{
	struct odl_port *p;

	pthread_mutex_lock(&g_port_lock);
	if (dev_id >= 0 && dev_id < g_num_devices) {
		p = &g_ports[dev_id];		if (p->refs > 0)			p->refs--;		if (p->refs == 0 && p->handle) {			odl_tb5_close(p->handle);			p->handle = NULL;		}	}
	pthread_mutex_unlock(&g_port_lock);
}

static int forced_device_from_env(int requested_dev, bool *forced)
{
	const char *env = getenv(ODL_FORCE_DEV_ENV);
	char *end = NULL;
	long want;
	int i;

	*forced = false;
	if (!env || !*env)
		return requested_dev;
	errno = 0;
	want = strtol(env, &end, 10);
	if (errno != 0 || end == env || *end != '\0' || want < 0) {
		TRACE("forced device: invalid %s=%s", ODL_FORCE_DEV_ENV, env);
		*forced = true;
		return -1;
	}
	*forced = true;
	for (i = 0; i < g_num_devices; i++) {
		if (g_ports[i].dev_index == (int)want) {
			TRACE("forced device: kernel dev=%ld -> port=%d", want, i);
			return i;
		}
	}
	TRACE("forced device: kernel dev=%ld not found", want);
	return -1;
}

static uint64_t virtual_guid(void)
{
	uint64_t guid = 1469598103934665603ULL;

	for (int i = 0; i < g_num_devices; i++) {
		guid ^= g_ports[i].guid;
		guid *= 1099511628211ULL;
	}
	return guid ? guid : 1;
}

static int virtual_speed_mbps(void)
{
	int speed = 0;

	for (int i = 0; i < g_num_devices; i++) {
		int port_speed = g_ports[i].info.link_speed ?
			(int)g_ports[i].info.link_speed * 1000 :
			ODL_LINK_SPEED_MBPS;
		if (port_speed > speed)
			speed = port_speed;
	}
	return speed ? speed : ODL_LINK_SPEED_MBPS;
}

/* ── Worker thread ────────────────────────────────────────────────────── */

static int stream_recv_timed(struct odl_comm *c, void *buf, uint32_t len,
			     uint8_t *src_id, uint32_t *actual)
{
	int ret;

	ret = odl_tb5_stream_wait_rx(c->handle, c->stream_id,
				     g_recv_timeout_ms);
	if (ret != 0) {
		fprintf(stderr,
			"[odl_tb5] RX wait failed sid=%u timeout_ms=%u rc=%d (%s)\n",
			c->stream_id, g_recv_timeout_ms, ret,
			strerror(-ret));
		return ret;
	}

	return odl_tb5_stream_recv(c->handle, c->stream_id, buf, len,
				   src_id, actual);
}

static int stream_send_payload(struct odl_comm *c, const void *data,
			       uint32_t len)
{
	uint32_t sent = 0;

	while (sent < len) {
		uint32_t chunk = len - sent;
		int ret;

		if (chunk > ODL_WIRE_CHUNK_BYTES)
			chunk = ODL_WIRE_CHUNK_BYTES;
		ret = odl_tb5_stream_send(c->handle, c->stream_id, c->dst_id,
					  (const uint8_t *)data + sent, chunk);
		if (ret != 0)
			return ret;
		sent += chunk;
	}
	return 0;
}

static int stream_recv_exact(struct odl_comm *c, void *data, uint32_t len,
			      uint8_t *src_id)
{
	uint32_t recvd = 0;
	uint8_t last_src = 0;

	while (recvd < len) {
		uint32_t want = len - recvd;
		uint32_t actual = 0;
		uint8_t src = 0;
		int ret;

		if (want > ODL_WIRE_CHUNK_BYTES)
			want = ODL_WIRE_CHUNK_BYTES;
		ret = stream_recv_timed(c, (uint8_t *)data + recvd, want,
					 &src, &actual);
		if (ret != 0)
			return ret;
		if (actual == 0 || actual > want) {
			fprintf(stderr,
				"[odl_tb5] payload boundary mismatch sid=%u expected=%u actual=%u\n",
				c->stream_id, want, actual);
			return -EBADMSG;
		}
		recvd += actual;
		if (src)
			last_src = src;
	}
	if (src_id && last_src)
		*src_id = last_src;
	return 0;
}

static int stream_recv_payload(struct odl_comm *c, void *data, uint32_t len)
{
	return stream_recv_exact(c, data, len, NULL);
}

static int stream_drain_payload(struct odl_comm *c, uint32_t len)
{
	uint8_t scratch[4096];
	uint32_t drained = 0;

	while (drained < len) {
		uint32_t remain = len - drained;
		uint32_t copy_len, actual = 0;
		int ret;

		if (remain > ODL_WIRE_CHUNK_BYTES)
			remain = ODL_WIRE_CHUNK_BYTES;
		copy_len = remain < sizeof(scratch) ? remain : sizeof(scratch);
		ret = stream_recv_timed(c, scratch, copy_len, NULL, &actual);
		if (ret != 0)
			return ret;
		if (actual == 0 || actual > copy_len)
			return -EBADMSG;
		drained += actual;
	}
	return 0;
}

static uint8_t ack_dst(struct odl_comm *c)
{
	if (c->peer_stream)
		return c->peer_stream;
	return c->dst_id;
}

static int stream_send_ack(struct odl_comm *c)
{
	uint32_t ack = ODL_ACK_HDR;
	uint8_t dst = ack_dst(c);

	if (dst == 0) {
		fprintf(stderr,
			"[odl_tb5] ACK skipped sid=%u: no peer stream\n",
			c->stream_id);
		return -EINVAL;
	}
	TRACE("TX ack sid=%u dst=%u", c->stream_id, dst);
	return odl_tb5_stream_send(c->handle, c->stream_id, dst,
				   &ack, sizeof(ack));
}

static int stream_recv_ack(struct odl_comm *c)
{
	uint32_t ack = 0;
	uint32_t actual = 0;
	int ret;

	TRACE("RX ack_pre sid=%u", c->stream_id);
	ret = stream_recv_timed(c, &ack, sizeof(ack), NULL, &actual);
	if (ret != 0)
		return ret;
	if (actual != sizeof(ack) || ack != ODL_ACK_HDR) {
		fprintf(stderr,
			"[odl_tb5] invalid RX ack sid=%u actual=%u value=0x%x\n",
			c->stream_id, actual, ack);
		return -EBADMSG;
	}
	TRACE("RX ack_post sid=%u", c->stream_id);
	return 0;
}

static void *worker_fn(void *arg)
{
	struct odl_comm *c = arg;

	pthread_mutex_lock(&c->mu);
	for (;;) {
		while (c->q_head == c->q_tail && !c->shutdown)
			pthread_cond_wait(&c->cv, &c->mu);
		if (c->shutdown && c->q_head == c->q_tail)
			break;

		struct odl_job j = c->q[c->q_head % ODL_JOB_QUEUE_DEPTH];
		c->q_head++;
		pthread_cond_signal(&c->cv);
		pthread_mutex_unlock(&c->mu);

		int xferred = -1;
		struct profile_counters *prof = c->is_send ? &g_prof_tx : &g_prof_rx;

		if (c->is_send) {
			int ret = 0;
			uint64_t t0 = 0, t1 = 0;
			uint32_t hdr = (uint32_t)j.size;
			if (g_profile)
				t0 = mono_ns();
			TRACE("TX hdr_pre sid=%u dst=%u size=%d",
			      c->stream_id, c->dst_id, j.size);
			ret = odl_tb5_stream_send(c->handle, c->stream_id,
						  c->dst_id, &hdr,
						  sizeof(hdr));
			if (g_profile) {
				t1 = mono_ns();
				prof_add_u64(&prof->hdr_ops, 1);
				prof_add_u64(&prof->hdr_ns, t1 - t0);
			}
			TRACE("TX hdr_post sid=%u dst=%u size=%d ret=%d",
			      c->stream_id, c->dst_id, j.size, ret);
			if (ret == 0 && j.size > 0) {
				if (g_profile)
					t0 = mono_ns();
				TRACE("TX pay_pre sid=%u dst=%u size=%d",
				      c->stream_id, c->dst_id, j.size);
				ret = stream_send_payload(c, j.data,
							  (uint32_t)j.size);
				if (g_profile) {
					t1 = mono_ns();
					prof_add_u64(&prof->pay_ops, 1);
					prof_add_u64(&prof->pay_ns, t1 - t0);
				}
				TRACE("TX pay_post sid=%u dst=%u size=%d ret=%d",
				      c->stream_id, c->dst_id, j.size, ret);
			}
			if (ret == 0 && j.size >= ODL_ACK_MIN_MESSAGE)
				c->ack_bytes += (uint64_t)j.size;
			if (ret == 0 &&
			    c->ack_bytes >= ODL_ACK_WINDOW_BYTES) {
				c->ack_bytes = 0;
				ret = stream_recv_ack(c);
			}
			if (ret == 0) {
				xferred = j.size;
				stats_tx(xferred);
			}
		} else {
			uint32_t hdr = 0, actual = 0;
			int ret = 0;
			uint64_t t0 = 0, t1 = 0;

			if (g_profile)
				t0 = mono_ns();
			TRACE("RX hdr_pre sid=%u jsize=%d",
			      c->stream_id, j.size);
			{
				uint8_t src = 0;

				ret = stream_recv_exact(c, &hdr, sizeof(hdr),
							 &src);
				actual = ret == 0 ? (uint32_t)sizeof(hdr) : 0;
				if (ret == 0 && src)
					c->peer_stream = src;
			}
			if (g_profile) {
				t1 = mono_ns();
				prof_add_u64(&prof->hdr_ops, 1);
				prof_add_u64(&prof->hdr_ns, t1 - t0);
			}
			TRACE("RX hdr_post sid=%u ret=%d actual=%u hdr=%u",
			      c->stream_id, ret, actual, hdr);

			if (ret == 0 && hdr > 32u * 1024u * 1024u) {
				fprintf(stderr,
					"[odl_tb5] FATAL: hdr insane sid=%u hdr=%u (>32MB) framing broken\n",
					c->stream_id, hdr);
				ret = -EBADMSG;
			} else if (ret == 0 && hdr > 0) {
				if ((int)hdr > j.size) {
					int drain_ret = stream_drain_payload(c, hdr);

					if (drain_ret != 0)
						fprintf(stderr,
							"[odl_tb5] payload drain failed sid=%u size=%u rc=%d\n",
							c->stream_id, hdr,
							drain_ret);
					ret = -EMSGSIZE;
				} else {
					if (g_profile)
						t0 = mono_ns();
					TRACE("RX pay_pre sid=%u hdr=%u jsize=%d",
					      c->stream_id, hdr, j.size);
					ret = stream_recv_payload(c, j.data, hdr);
					if (g_profile) {
						t1 = mono_ns();
						prof_add_u64(&prof->pay_ops, 1);
						prof_add_u64(&prof->pay_ns, t1 - t0);
					}
					TRACE("RX pay_post sid=%u ret=%d size=%u",
					      c->stream_id, ret, hdr);
					actual = ret == 0 ? hdr : 0;
				}
			} else if (ret == 0 && hdr == 0) {
				actual = 0;
			}
			if (ret == 0 && hdr >= ODL_ACK_MIN_MESSAGE)
				c->ack_bytes += hdr;
			if (ret == 0 &&
			    c->ack_bytes >= ODL_ACK_WINDOW_BYTES) {
				c->ack_bytes = 0;
				ret = stream_send_ack(c);
			}
			if (ret == 0) {
				xferred = (int)actual;
				stats_rx(xferred);
			}
		}

		if (g_profile) {
			prof_add_u64(&prof->worker_ops, 1);
			if (xferred > 0)
				prof_add_u64(&prof->bytes, (uint64_t)xferred);
			if (j.size > 0)
				prof_max_u64(&prof->max_size, (uint64_t)j.size);
			prof_print_if_needed(c->is_send ? "tx_worker" : "rx_worker");
		}

		j.req->size = xferred;
		atomic_store_explicit(&j.req->done, true, memory_order_release);
		pthread_mutex_lock(&c->mu);
	}
	pthread_mutex_unlock(&c->mu);
	return NULL;
}

/* ── Comm allocation / teardown ───────────────────────────────────────── */

static struct odl_comm *comm_new(odl_tb5_t handle, int dev_id, uint8_t stream_id,
				uint8_t dst_id, bool is_send, bool stream_owned)
{
	struct odl_comm *c = calloc(1, sizeof(*c));
	if (!c)
		return NULL;
	c->handle = handle;
	c->dev_id = dev_id;
	c->stream_id = stream_id;
	c->dst_id = dst_id;
	c->is_send = is_send;
	c->stream_owned = stream_owned;
	c->peer_stream = dst_id;
	pthread_mutex_init(&c->mu, NULL);
	pthread_cond_init(&c->cv, NULL);
	if (pthread_create(&c->worker, NULL, worker_fn, c) != 0) {
		pthread_cond_destroy(&c->cv);
		pthread_mutex_destroy(&c->mu);
		free(c);
		return NULL;
	}
	return c;
}

static struct odl_request *req_alloc(struct odl_comm *c)
{
	struct odl_request *req = NULL;

	pthread_mutex_lock(&c->mu);
	for (int i = 0; i < ODL_REQ_POOL_DEPTH; i++) {
		if (!c->req_used[i]) {
			c->req_used[i] = true;
			req = &c->req_pool[i];
			req->comm = c;
			req->pool_idx = i;
			break;
		}
	}
	pthread_mutex_unlock(&c->mu);

	if (!req) {
		req = calloc(1, sizeof(*req));
		if (!req)
			return NULL;
		req->comm = NULL;
		req->pool_idx = -1;
	}
	atomic_init(&req->done, false);
	req->size = -1;
	req->is_send = c->is_send;
	return req;
}

static void req_free(struct odl_request *req)
{
	struct odl_comm *c;

	if (!req)
		return;
	c = req->comm;
	if (!c || req->pool_idx < 0) {
		free(req);
		return;
	}
	pthread_mutex_lock(&c->mu);
	c->req_used[req->pool_idx] = false;
	pthread_mutex_unlock(&c->mu);
}

static void comm_free(struct odl_comm *c)
{
	int dev_id;

	if (!c)
		return;
	pthread_mutex_lock(&c->mu);
	c->shutdown = true;
	pthread_cond_broadcast(&c->cv);
	pthread_mutex_unlock(&c->mu);
	pthread_join(c->worker, NULL);

	if (c->stream_owned)
		odl_tb5_stream_close(c->handle, c->stream_id);
	pthread_cond_destroy(&c->cv);
	pthread_mutex_destroy(&c->mu);
	dev_id = c->dev_id;
	free(c);
	device_release(dev_id);
}

/* Enqueue one job. Blocks only if the ring is full, which under NCCL's
 * 32-outstanding cap and our depth of 64 cannot happen in practice. */
static ncclResult_t comm_submit(struct odl_comm *c, struct odl_request *req,
				void *data, int size){
	uint64_t t0 = g_profile ? mono_ns() : 0;
	struct profile_counters *prof = c->is_send ? &g_prof_tx : &g_prof_rx;

	pthread_mutex_lock(&c->mu);
	while ((c->q_tail - c->q_head) >= ODL_JOB_QUEUE_DEPTH && !c->shutdown)
		pthread_cond_wait(&c->cv, &c->mu);
	if (c->shutdown) {
		pthread_mutex_unlock(&c->mu);
		return ncclInternalError;
	}
	struct odl_job *j = &c->q[c->q_tail % ODL_JOB_QUEUE_DEPTH];
	j->req = req;
	j->data = data;
	j->size = size;
	c->q_tail++;
	pthread_cond_signal(&c->cv);
	pthread_mutex_unlock(&c->mu);
	if (g_profile) {
		prof_add_u64(&prof->submit_ops, 1);
		prof_add_u64(&prof->submit_block_ns, mono_ns() - t0);
	}
	return ncclSuccess;
}

struct accept_probe_arg {
	struct odl_listen *listen;
	int idx;
};

static void *accept_probe_fn(void *arg)
{
	struct accept_probe_arg *probe = arg;
	struct odl_listen *l = probe->listen;
	int idx = probe->idx;
	struct odl_listen_port *p = &l->ports[idx];
	struct odl_connect_msg msg;
	uint8_t src = 0;
	uint32_t actual = 0;
	free(probe);

	for (;;) {
		int rc;

		pthread_mutex_lock(&l->mu);
		if (l->accepted) {
			pthread_mutex_unlock(&l->mu);
			break;
		}
		pthread_mutex_unlock(&l->mu);

		rc = odl_tb5_stream_wait_rx(p->handle, p->stream_id, 100);
		if (rc == -ETIMEDOUT)
			continue;
		if (rc < 0) {
			TRACE("accept probe wait failed idx=%d rc=%d", idx, rc);
			break;
		}

		memset(&msg, 0, sizeof(msg));
		rc = odl_tb5_stream_recv(p->handle, p->stream_id, &msg,
					 sizeof(msg), &src, &actual);
		if (rc < 0) {
			TRACE("accept probe recv failed idx=%d rc=%d", idx, rc);
			break;
		}
		if (actual != sizeof(msg) || msg.magic != ODL_CONNECT_MAGIC ||
		    msg.version != ODL_HANDLE_VERSION) {
			TRACE("accept probe ignored idx=%d actual=%u magic=0x%x version=%u",
			      idx, actual, msg.magic, msg.version);
			continue;
		}

		pthread_mutex_lock(&l->mu);
		if (!l->accepted) {
			l->accepted = true;
			l->accepted_port = idx;
			l->accepted_src = msg.src_stream ? msg.src_stream : src;
			pthread_cond_broadcast(&l->cv);
		}
		pthread_mutex_unlock(&l->mu);
		break;
	}

	pthread_mutex_lock(&l->mu);
	l->finished_probes++;
	pthread_cond_broadcast(&l->cv);
	pthread_mutex_unlock(&l->mu);
	return NULL;
}

/* ── Common (version-independent) plugin entrypoints ──────────────────── */

static ncclResult_t odl_init(ncclDebugLogger_t logFunction)
{
	const char *virtual_env;

	(void)logFunction;
	g_trace = env_flag_enabled("ODL_TB5_TRACE");
	virtual_env = getenv(ODL_VIRTUAL_DEV_ENV);
	g_virtual_single_dev = !(virtual_env && strcmp(virtual_env, "0") == 0);
	g_profile = (getenv(ODL_PROFILE_ENV) != NULL &&
		     strcmp(getenv(ODL_PROFILE_ENV), "0") != 0);
	if (getenv(ODL_PROFILE_INTERVAL_ENV)) {
		char *end = NULL;
		unsigned long long v = strtoull(getenv(ODL_PROFILE_INTERVAL_ENV),
						&end, 10);
		if (end && *end == '\0' && v > 0)
			g_profile_interval = (uint64_t)v;
	}
	if (getenv(ODL_RECV_TIMEOUT_ENV)) {
		char *end = NULL;
		unsigned long v = strtoul(getenv(ODL_RECV_TIMEOUT_ENV), &end, 10);

		if (end && *end == '\0' && v <= UINT32_MAX)
			g_recv_timeout_ms = (uint32_t)v;
	}
	stats_init();
	atexit(stats_cleanup);
	TRACE("init pid=%d virtual_single_dev=%d recv_timeout_ms=%u",
	      (int)getpid(), g_virtual_single_dev, g_recv_timeout_ms);
	if (g_profile)
		fprintf(stderr,
			"[odl_tb5_profile] enabled interval=%llu pid=%d\n",
			(unsigned long long)g_profile_interval, (int)getpid());
	return ncclSuccess;
}

static ncclResult_t odl_devices(int *ndev)
{
	int physical;

	pthread_mutex_lock(&g_port_lock);
	g_num_devices = refresh_ports_locked();
	physical = g_num_devices;
	 *ndev = (g_virtual_single_dev && physical > 0) ? 1 : physical;
	pthread_mutex_unlock(&g_port_lock);
	TRACE("devices ndev=%d physical=%d virtual=%d",
	      *ndev, physical, g_virtual_single_dev);
	return ncclSuccess;
}

static ncclResult_t odl_listen(int dev, void *opaqueHandle, void **listenComm)
{
	struct odl_listen *l;
	struct odl_handle *wire = opaqueHandle;
	bool forced = false;
	int selected_dev = -1, nlisten, i;

	if (sizeof(*wire) > NCCL_NET_HANDLE_MAXSIZE)
		return ncclInternalError;
	l = calloc(1, sizeof(*l));
	if (!l)
		return ncclSystemError;
	pthread_mutex_init(&l->mu, NULL);
	pthread_cond_init(&l->cv, NULL);
	l->accepted_port = -1;

	if (g_virtual_single_dev && dev == 0 && !getenv(ODL_FORCE_DEV_ENV)) {
		nlisten = g_num_devices < ODL_HANDLE_MAX_PORTS ?
			g_num_devices : ODL_HANDLE_MAX_PORTS;
		selected_dev = 0;
	} else {
		selected_dev = forced_device_from_env(dev, &forced);
		TRACE("listen dev=%d selected=%d forced=%d",
		      dev, selected_dev, forced);
		if (selected_dev < 0 || selected_dev >= g_num_devices) {
			pthread_cond_destroy(&l->cv);
			pthread_mutex_destroy(&l->mu);
			free(l);
			return ncclInvalidArgument;
		}
		nlisten = 1;
	}

	memset(wire, 0, NCCL_NET_HANDLE_MAXSIZE);
	wire->magic = ODL_HANDLE_MAGIC;
	wire->version = ODL_HANDLE_VERSION;
	wire->nports = (uint32_t)nlisten;
	wire->guid = g_virtual_single_dev ? virtual_guid() : g_ports[selected_dev].guid;

	for (i = 0; i < nlisten; i++) {
		odl_tb5_t handle;
		uint8_t stream_id = 0;
		int port_dev = (nlisten == 1) ? selected_dev : i;

		if (device_acquire(port_dev, &handle) < 0)
			goto fail;
		if (odl_tb5_stream_open(handle, 0, &stream_id) < 0) {
			TRACE("listen: stream_open failed dev=%d errno=%d",
			      port_dev, errno);
			device_release(port_dev);
			goto fail;
		}
		l->ports[i].dev_id = port_dev;
		l->ports[i].handle = handle;
		l->ports[i].stream_id = stream_id;
		l->ports[i].stream_owned = true;
		l->nports++;

		wire->ports[i].dev_id = port_dev;
		memcpy(wire->ports[i].listener_uuid,
		       g_ports[port_dev].local_uuid,
		       sizeof(wire->ports[i].listener_uuid));
		wire->ports[i].stream_id = stream_id;
		TRACE("listen ok logical=%d port=%d kernel=%d stream_id=%u",
		      dev, port_dev, g_ports[port_dev].dev_index, stream_id);
	}

	 *listenComm = l;
	return ncclSuccess;

fail:
	for (i = 0; i < l->nports; i++) {
		if (l->ports[i].stream_owned)
			odl_tb5_stream_close(l->ports[i].handle,
					     l->ports[i].stream_id);
		device_release(l->ports[i].dev_id);
	}
	pthread_cond_destroy(&l->cv);
	pthread_mutex_destroy(&l->mu);
	free(l);
	return ncclSystemError;
}

static ncclResult_t odl_accept(void *listenComm, void **recvComm)
{
	struct odl_listen *l = listenComm;
	odl_tb5_t handle;
	struct odl_comm *c;
	int idx;

	if (l->nports <= 0)
		return ncclInternalError;
	/* 2-node (nports==1) and 4-node mesh share the connect handshake so
	 * recvComm->dst_id is the sender stream ACK must target. */
	for (int i = 0; i < l->nports; i++) {
		struct accept_probe_arg *arg = calloc(1, sizeof(*arg));
		if (!arg)
			return ncclSystemError;
		arg->listen = l;
		arg->idx = i;
		if (pthread_create(&l->ports[i].probe_thread, NULL,
				   accept_probe_fn, arg) != 0) {
			free(arg);
			return ncclSystemError;
		}
		l->ports[i].probe_started = true;
	}
	pthread_mutex_lock(&l->mu);
	while (!l->accepted && l->finished_probes < l->nports)
		pthread_cond_wait(&l->cv, &l->mu);
	pthread_mutex_unlock(&l->mu);
	for (int i = 0; i < l->nports; i++) {
		if (l->ports[i].probe_started) {
			pthread_join(l->ports[i].probe_thread, NULL);
			l->ports[i].probe_started = false;
		}
	}

	idx = l->accepted_port;
	if (idx < 0 || idx >= l->nports)
		return ncclSystemError;
	TRACE("accept port=%d dev=%d stream_id=%u src=%u",
	      idx, l->ports[idx].dev_id, l->ports[idx].stream_id,
	      l->accepted_src);
	/* Bump device ref independently of listen — listen's ref is dropped
	 * by closeListen, recvComm's is dropped by closeRecv. */
	if (device_acquire(l->ports[idx].dev_id, &handle) < 0)
		return ncclSystemError;
	c = comm_new(handle, l->ports[idx].dev_id, l->ports[idx].stream_id,
		     l->accepted_src, false, true);
	if (!c) {
		device_release(l->ports[idx].dev_id);
		return ncclSystemError;
	}
	/* Stream ownership transfers from listen to recvComm. */
	l->ports[idx].stream_owned = false;

	 *recvComm = c;
	return ncclSuccess;
}

static ncclResult_t odl_connect(int dev, void *opaqueHandle, void **sendComm)
{
	odl_tb5_t handle;
	struct odl_comm *c;
	struct odl_handle *wire = opaqueHandle;
	uint8_t local_stream = 0;
	uint8_t dst_stream = 0;
	bool forced = false;
	bool route_found = false;
	int selected_dev = -1;

	TRACE("connect dev=%d nports=%u",
	      dev, wire ? wire->nports : 0);
	if (!wire || wire->magic != ODL_HANDLE_MAGIC ||
	    wire->version != ODL_HANDLE_VERSION) {
		TRACE("connect: bad handle magic=0x%x ver=%u",
		      wire ? wire->magic : 0, wire ? wire->version : 0);
		return ncclInvalidArgument;
	}
	if (wire->nports == 0 || wire->nports > ODL_HANDLE_MAX_PORTS)
		return ncclInvalidArgument;

	selected_dev = forced_device_from_env(dev, &forced);
	for (int h = 0; h < (int)wire->nports; h++) {
		for (int i = 0; i < g_num_devices; i++) {
			if (forced && i != selected_dev)
				continue;
			if (uuid_equal_bytes(g_ports[i].peer_uuid,
					     wire->ports[h].listener_uuid)) {
				selected_dev = i;
				dst_stream = wire->ports[h].stream_id;
				route_found = true;
				goto route_search_done;
			}
		}
	}
route_search_done:
	if (!route_found || selected_dev < 0 || selected_dev >= g_num_devices) {
		char want[40];
		uuid_to_str(wire->ports[0].listener_uuid, want, sizeof(want));
		TRACE("connect: no local port reaches listener uuid=%s nports=%u",
		      want, wire->nports);
		return ncclSystemError;
	}

	if (device_acquire(selected_dev, &handle) < 0)
		return ncclSystemError;
	/* Sender needs *some* local stream to call stream_send from. The
	 * dst_id field is what the receiver filters on. */
	if (odl_tb5_stream_open(handle, 0, &local_stream) < 0) {
		TRACE("connect: stream_open failed errno=%d", errno);
		device_release(selected_dev);
		return ncclSystemError;
	}
	TRACE("connect ok dev=%d local=%u dst=%u", selected_dev,
	local_stream, dst_stream);

	/* Always handshake, including 2-node nports==1: accept() waits for
	 * this so recvComm knows the sender stream for windowed ACKs. */
	{
		struct odl_connect_msg msg = {
			.magic = ODL_CONNECT_MAGIC,
			.version = ODL_HANDLE_VERSION,
			.src_stream = local_stream,
		};
		int rc = odl_tb5_stream_send(handle, local_stream, dst_stream,
					     &msg, sizeof(msg));
		if (rc < 0) {
			TRACE("connect: control send failed rc=%d", rc);
			odl_tb5_stream_close(handle, local_stream);
			device_release(selected_dev);
			return ncclSystemError;
		}
		TRACE("connect handshake sent local=%u dst=%u",
		      local_stream, dst_stream);
	}

	c = comm_new(handle, selected_dev, local_stream, dst_stream, true, true);
	if (!c) {
		odl_tb5_stream_close(handle, local_stream);
		device_release(selected_dev);
		return ncclSystemError;
	}
	 *sendComm = c;
	return ncclSuccess;
}

static ncclResult_t odl_closeSend(void *sendComm)
{
	comm_free(sendComm);
	return ncclSuccess;
}

static ncclResult_t odl_closeRecv(void *recvComm)
{
	comm_free(recvComm);
	return ncclSuccess;
}

static ncclResult_t odl_closeListen(void *listenComm)
{
	struct odl_listen *l = listenComm;
	if (!l)
		return ncclSuccess;
	pthread_mutex_lock(&l->mu);
	l->accepted = true;
	pthread_cond_broadcast(&l->cv);
	pthread_mutex_unlock(&l->mu);
	for (int i = 0; i < l->nports; i++) {
		if (l->ports[i].probe_started)
			pthread_join(l->ports[i].probe_thread, NULL);
		if (l->ports[i].stream_owned)
			odl_tb5_stream_close(l->ports[i].handle,
					     l->ports[i].stream_id);
		device_release(l->ports[i].dev_id);
	}
	pthread_cond_destroy(&l->cv);
	pthread_mutex_destroy(&l->mu);
	free(l);
	return ncclSuccess;
}

/* Memory registration: we run on host pointers, no IOMMU mapping needed.
 * Return a sentinel mhandle so RCCL is happy. */
static ncclResult_t odl_regMr(void *comm, void *data, size_t size,
				int type, void **mhandle){
	(void)comm; (void)data;
	TRACE("regMr size=%zu type=%d", size, type);
	if (type != NCCL_PTR_HOST)
		return ncclInternalError;
	 *mhandle = (void *)(uintptr_t)0x1; /* opaque, non-NULL */
	return ncclSuccess;
}

static ncclResult_t odl_regMrDmaBuf(void *comm, void *data, size_t size,
					int type, uint64_t offset, int fd,					void **mhandle){
	(void)comm; (void)data; (void)size; (void)type;
	(void)offset; (void)fd; (void)mhandle;
	return ncclInternalError;
}

static ncclResult_t odl_deregMr(void *comm, void *mhandle)
{
	(void)comm; (void)mhandle;
	return ncclSuccess;
}

static ncclResult_t odl_isend(void *sendComm, void *data, int size, int tag,
				void *mhandle, void **request){
	struct odl_comm *c = sendComm;
	struct odl_request *req;
	(void)tag; (void)mhandle;

	TRACE("isend size=%d tag=%d", size, tag);
	if (size < 0) {
		 *request = NULL;
		return ncclInvalidArgument;	}
	req = req_alloc(c);
	if (!req)
		return ncclSystemError;

	ncclResult_t r = comm_submit(c, req, data, size);
	if (r != ncclSuccess) {
		req_free(req);
		return r;
	}
	 *request = req;
	return ncclSuccess;
}

static ncclResult_t odl_irecv(void *recvComm, int n, void **data, int *sizes,
				int *tags, void **mhandles, void **request){
	struct odl_comm *c = recvComm;
	struct odl_request *req;
	(void)tags; (void)mhandles;

	TRACE("irecv n=%d size0=%d tag0=%d", n, n>0?sizes[0]:-1, n>0?tags[0]:-1);
	if (n != 1)
		return ncclInvalidArgument;
	if (sizes[0] < 0) {
		*request = NULL;
		return ncclInvalidArgument;
	}
	req = req_alloc(c);
	if (!req)
		return ncclSystemError;

	ncclResult_t r = comm_submit(c, req, data[0], sizes[0]);
	if (r != ncclSuccess) {
		req_free(req);
		return r;
	}
	 *request = req;
	return ncclSuccess;
}

static ncclResult_t odl_iflush(void *recvComm, int n, void **data, int *sizes,
					void **mhandles, void **request)
{
	(void)recvComm; (void)n; (void)data; (void)sizes; (void)mhandles;
	 *request = NULL; /* host pointers: nothing to flush */
	return ncclSuccess;
}

static ncclResult_t odl_test(void *request, int *done, int *size)
{
	struct odl_request *req = request;
	if (!req) {
		 *done = 1;
		if (size)
			*size = 0;
		return ncclSuccess;
	}
	if (g_profile) {
		struct profile_counters *prof =
			req->is_send ? &g_prof_tx : &g_prof_rx;
		prof_add_u64(&prof->test_ops, 1);
	}
	if (atomic_load_explicit(&req->done, memory_order_acquire)) {
		int xferred = req->size;
		*done = 1;
		if (size)
			*size = xferred < 0 ? 0 : xferred;
		req_free(req);
		if (xferred < 0)
			return ncclSystemError;
		return ncclSuccess;
	}
	if (g_profile) {
		struct profile_counters *prof =
			req->is_send ? &g_prof_tx : &g_prof_rx;
		prof_add_u64(&prof->test_pending, 1);
	}
	 *done = 0;
	return ncclSuccess;
}

static ncclResult_t odl_getDeviceMr(void *comm, void *mhandle,
					void **dptr_mhandle){
	(void)comm; (void)mhandle; (void)dptr_mhandle;
	return ncclInternalError;
}

static ncclResult_t odl_irecvConsumed(void *recvComm, int n, void *request)
{
	(void)recvComm; (void)n; (void)request;
	return ncclSuccess;
}

/* ── Per-version property fillers ─────────────────────────────────────── */

static ncclResult_t odl_getProperties_v8(int dev, ncclNetProperties_v8_t *p)
{
	static char virtual_name[] = "OdinLink_TB5_virtual";

	TRACE("getProperties_v8 dev=%d (n=%d)", dev, g_num_devices);
	if (g_virtual_single_dev) {
		if (dev != 0 || g_num_devices <= 0)
			return ncclInvalidArgument;
		memset(p, 0, sizeof(*p));
		p->name = virtual_name;
		p->pciPath = NULL;
		p->guid = virtual_guid();
		p->ptrSupport = NCCL_PTR_HOST;
		p->regIsGlobal = 0;
		p->speed = virtual_speed_mbps();
		p->port = 0;
		p->latency = ODL_NET_LATENCY_US;
		p->maxComms = 8;
		p->maxRecvs = 1;
		p->netDeviceType = NCCL_NET_DEVICE_HOST;
		p->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
		return ncclSuccess;
	}
	if (dev < 0 || dev >= g_num_devices)
		return ncclInvalidArgument;
	memset(p, 0, sizeof(*p));
	p->name = g_ports[dev].name;
	/*
	 * OdinLink devices are virtual character devices, not NIC PCI functions.
	 * Returning /sys/devices/virtual/... here makes RCCL try to treat a
	 * non-PCI node as topology data. Match NCCL's socket transport behavior:
	 * when no real PCI path exists, leave pciPath NULL.
	 */
	p->pciPath = NULL;
	p->guid = g_ports[dev].guid;
	p->ptrSupport = NCCL_PTR_HOST;
	p->regIsGlobal = 0;
	p->speed = g_ports[dev].info.link_speed ?
		(int)g_ports[dev].info.link_speed * 1000 :
		ODL_LINK_SPEED_MBPS;
	p->port = dev;
	p->latency = ODL_NET_LATENCY_US;
	p->maxComms = 8;
	p->maxRecvs = 1;
	p->netDeviceType = NCCL_NET_DEVICE_HOST;
	p->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	TRACE("getProperties_v8 done dev=%d props=%p", dev, (void *)p);
	return ncclSuccess;
}

static ncclResult_t odl_getProperties_v7(int dev, ncclNetProperties_v7_t *p)
{
	ncclNetProperties_v8_t v8;
	ncclResult_t r = odl_getProperties_v8(dev, &v8);
	if (r != ncclSuccess) return r;
	memset(p, 0, sizeof(*p));
	p->name = v8.name;
	p->pciPath = v8.pciPath;
	p->guid = v8.guid;
	p->ptrSupport = v8.ptrSupport;
	p->speed = v8.speed;
	p->port = v8.port;
	p->latency = v8.latency;
	p->maxComms = v8.maxComms;
	p->maxRecvs = v8.maxRecvs;
	p->netDeviceType = v8.netDeviceType;
	p->netDeviceVersion = v8.netDeviceVersion;
	return ncclSuccess;
}

static ncclResult_t odl_getProperties_v6(int dev, ncclNetProperties_v6_t *p)
{
	ncclNetProperties_v8_t v8;
	ncclResult_t r = odl_getProperties_v8(dev, &v8);
	if (r != ncclSuccess) return r;
	memset(p, 0, sizeof(*p));
	p->name = v8.name;
	p->pciPath = v8.pciPath;
	p->guid = v8.guid;
	p->ptrSupport = v8.ptrSupport;
	p->speed = v8.speed;
	p->port = v8.port;
	p->latency = v8.latency;
	p->maxComms = v8.maxComms;
	p->maxRecvs = v8.maxRecvs;
	return ncclSuccess;
}

/* ── ABI adapters ─────────────────────────────────────────────────────── */

/* v6 regMr takes int size; v8 takes size_t. Adapt. */
static ncclResult_t odl_regMr_v6(void *comm, void *data, int size,
				int type, void **mhandle){
	return odl_regMr(comm, data, (size_t)size, type, mhandle);
}

/* v7/v8 connect/accept include device-handle outparams (must NULL them). */
static ncclResult_t odl_connect_v7(int dev, void *handle, void **sendComm,
					ncclNetDeviceHandle_v7_t **sendDevComm)
{
	if (sendDevComm) *sendDevComm = NULL;
	return odl_connect(dev, handle, sendComm);
}

static ncclResult_t odl_accept_v7(void *listenComm, void **recvComm,
				ncclNetDeviceHandle_v7_t **recvDevComm){
	if (recvDevComm) *recvDevComm = NULL;
	return odl_accept(listenComm, recvComm);
}

/* ── Plugin tables ────────────────────────────────────────────────────── */

ODL_EXPORT const ncclNet_v8_t ncclNetPlugin_v8 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v8,
	.listen         = odl_listen,
	.connect        = odl_connect_v7,
	.accept         = odl_accept_v7,
	.regMr          = odl_regMr,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend,
	.irecv          = odl_irecv,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
	.getDeviceMr    = odl_getDeviceMr,
	.irecvConsumed  = odl_irecvConsumed,
};

ODL_EXPORT const ncclNet_v7_t ncclNetPlugin_v7 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v7,
	.listen         = odl_listen,
	.connect        = odl_connect_v7,
	.accept         = odl_accept_v7,
	.regMr          = odl_regMr_v6,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend,
	.irecv          = odl_irecv,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
	.getDeviceMr    = odl_getDeviceMr,
	.irecvConsumed  = odl_irecvConsumed,
};

ODL_EXPORT const ncclNet_v6_t ncclNetPlugin_v6 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v6,
	.listen         = odl_listen,
	.connect        = odl_connect,
	.accept         = odl_accept,
	.regMr          = odl_regMr_v6,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend,
	.irecv          = odl_irecv,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
};

/* ── v9–v12 ABI adapters (current RCCL prefers ncclNetPlugin_v12) ───── */

static ncclResult_t odl_isend_sz(void *sendComm, void *data, size_t size, int tag,
				void *mhandle, void **request)
{
	if (size > (size_t)INT_MAX)
		return ncclInternalError;
	return odl_isend(sendComm, data, (int)size, tag, mhandle, request);
}

static ncclResult_t odl_irecv_sz(void *recvComm, int n, void **data, size_t *sizes,
				int *tags, void **mhandles, void **request)
{
	int sizesInt[NCCL_NET_MAX_REQUESTS];
	int i;

	if (n < 0 || n > NCCL_NET_MAX_REQUESTS)
		return ncclInvalidArgument;
	for (i = 0; i < n; i++) {
		if (sizes[i] > (size_t)INT_MAX)
			return ncclInternalError;
		sizesInt[i] = (int)sizes[i];
	}
	return odl_irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
}

static ncclResult_t odl_getProperties_v9(int dev, ncclNetProperties_v9_t *p)
{
	ncclNetProperties_v8_t v8;
	ncclResult_t r = odl_getProperties_v8(dev, &v8);
	if (r != ncclSuccess)
		return r;
	memset(p, 0, sizeof(*p));
	p->name = v8.name;
	p->pciPath = v8.pciPath;
	p->guid = v8.guid;
	p->ptrSupport = v8.ptrSupport;
	p->regIsGlobal = v8.regIsGlobal;
	p->forceFlush = 0;
	p->speed = v8.speed;
	p->port = v8.port;
	p->latency = v8.latency;
	p->maxComms = v8.maxComms;
	p->maxRecvs = v8.maxRecvs;
	p->netDeviceType = v8.netDeviceType;
	p->netDeviceVersion = v8.netDeviceVersion;
	p->vProps.ndevs = 1;
	p->vProps.devs[0] = dev;
	p->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
	p->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
	return ncclSuccess;
}

static ncclResult_t odl_getProperties_v10(int dev, ncclNetProperties_v10_t *p)
{
	return odl_getProperties_v9(dev, (ncclNetProperties_v9_t *)p);
}

static ncclResult_t odl_getProperties_v11(int dev, ncclNetProperties_v11_t *p)
{
	ncclNetProperties_v9_t v9;
	ncclResult_t r = odl_getProperties_v9(dev, &v9);
	if (r != ncclSuccess)
		return r;
	memset(p, 0, sizeof(*p));
	p->name = v9.name;
	p->pciPath = v9.pciPath;
	p->guid = v9.guid;
	p->ptrSupport = v9.ptrSupport;
	p->regIsGlobal = v9.regIsGlobal;
	p->forceFlush = v9.forceFlush;
	p->speed = v9.speed;
	p->port = v9.port;
	p->latency = v9.latency;
	p->maxComms = v9.maxComms;
	p->maxRecvs = v9.maxRecvs;
	p->netDeviceType = v9.netDeviceType;
	p->netDeviceVersion = v9.netDeviceVersion;
	p->vProps.ndevs = v9.vProps.ndevs;
	p->vProps.devs[0] = v9.vProps.devs[0];
	p->maxP2pBytes = v9.maxP2pBytes;
	p->maxCollBytes = v9.maxCollBytes;
	p->maxMultiRequestSize = 1;
	return ncclSuccess;
}

static ncclResult_t odl_getProperties_v12(int dev, ncclNetProperties_v12_t *p)
{
	ncclNetProperties_v11_t v11;
	ncclResult_t r = odl_getProperties_v11(dev, &v11);
	if (r != ncclSuccess)
		return r;
	memset(p, 0, sizeof(*p));
	p->name = v11.name;
	p->pciPath = v11.pciPath;
	p->guid = v11.guid;
	p->ptrSupport = v11.ptrSupport;
	p->regIsGlobal = v11.regIsGlobal;
	p->forceFlush = v11.forceFlush;
	p->speed = v11.speed;
	p->port = v11.port;
	p->latency = v11.latency;
	p->maxComms = v11.maxComms;
	p->maxRecvs = v11.maxRecvs;
	p->netDeviceType = v11.netDeviceType;
	p->netDeviceVersion = v11.netDeviceVersion;
	p->vProps.ndevs = v11.vProps.ndevs;
	p->vProps.devs[0] = v11.vProps.devs[0];
	p->maxP2pBytes = v11.maxP2pBytes;
	p->maxCollBytes = v11.maxCollBytes;
	p->maxMultiRequestSize = v11.maxMultiRequestSize;
	p->railId = NCCL_NET_ID_UNDEF;
	p->planeId = NCCL_NET_ID_UNDEF;
	return ncclSuccess;
}

static ncclResult_t odl_init_v10(ncclDebugLogger_t logFunction,
				ncclProfilerCallback_t profFunction)
{
	(void)profFunction;
	return odl_init(logFunction);
}

static ncclResult_t odl_init_v11(void **ctx, uint64_t commId,
				ncclNetCommConfig_v11_t *config,
				ncclDebugLogger_t logFunction,
				ncclProfilerCallback_t profFunction)
{
	(void)commId;
	(void)config;
	(void)profFunction;
	if (ctx)
		*ctx = NULL;
	return odl_init(logFunction);
}

static ncclResult_t odl_init_v12(void **ctx, uint64_t commId,
				ncclNetCommConfig_v12_t *config,
				ncclDebugLogger_t logFunction,
				ncclProfilerCallback_t profFunction)
{
	(void)commId;
	(void)config;
	(void)profFunction;
	if (ctx)
		*ctx = NULL;
	return odl_init(logFunction);
}

static ncclResult_t odl_listen_v11(void *ctx, int dev, void *handle, void **listenComm)
{
	(void)ctx;
	return odl_listen(dev, handle, listenComm);
}

static ncclResult_t odl_connect_v10(int dev, ncclNetCommConfig_v10_t *config,
				void *handle, void **sendComm,
				ncclNetDeviceHandle_v10_t **sendDevComm)
{
	(void)config;
	if (sendDevComm)
		*sendDevComm = NULL;
	return odl_connect(dev, handle, sendComm);
}

static ncclResult_t odl_connect_v11(void *ctx, int dev, void *handle, void **sendComm,
				ncclNetDeviceHandle_v11_t **sendDevComm)
{
	(void)ctx;
	if (sendDevComm)
		*sendDevComm = NULL;
	return odl_connect(dev, handle, sendComm);
}

static ncclResult_t odl_connect_v12(void *ctx, int dev, void *handle, void **sendComm,
				ncclNetDeviceHandle_v12_t **sendDevComm)
{
	(void)ctx;
	if (sendDevComm)
		*sendDevComm = NULL;
	return odl_connect(dev, handle, sendComm);
}

static ncclResult_t odl_accept_v11(void *listenComm, void **recvComm,
				ncclNetDeviceHandle_v11_t **recvDevComm)
{
	if (recvDevComm)
		*recvDevComm = NULL;
	return odl_accept(listenComm, recvComm);
}

static ncclResult_t odl_isend_v10(void *sendComm, void *data, size_t size, int tag,
				void *mhandle, void *phandle, void **request)
{
	(void)phandle;
	return odl_isend_sz(sendComm, data, size, tag, mhandle, request);
}

static ncclResult_t odl_irecv_v10(void *recvComm, int n, void **data, size_t *sizes,
				int *tags, void **mhandles, void **phandles,
				void **request)
{
	(void)phandles;
	return odl_irecv_sz(recvComm, n, data, sizes, tags, mhandles, request);
}

static ncclResult_t odl_finalize(void *ctx)
{
	(void)ctx;
	return ncclSuccess;
}

ODL_EXPORT const ncclNet_v9_t ncclNetPlugin_v9 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v9,
	.listen         = odl_listen,
	.connect        = odl_connect_v7,
	.accept         = odl_accept_v7,
	.regMr          = odl_regMr,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend_sz,
	.irecv          = odl_irecv_sz,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
	.getDeviceMr    = odl_getDeviceMr,
	.irecvConsumed  = odl_irecvConsumed,
	.makeVDevice    = NULL,
};

ODL_EXPORT const ncclNet_v10_t ncclNetPlugin_v10 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init_v10,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v10,
	.listen         = odl_listen,
	.connect        = odl_connect_v10,
	.accept         = odl_accept_v7,
	.regMr          = odl_regMr,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend_v10,
	.irecv          = odl_irecv_v10,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
	.getDeviceMr    = odl_getDeviceMr,
	.irecvConsumed  = odl_irecvConsumed,
	.makeVDevice    = NULL,
};

ODL_EXPORT const ncclNet_v11_t ncclNetPlugin_v11 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init_v11,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v11,
	.listen         = odl_listen_v11,
	.connect        = odl_connect_v11,
	.accept         = odl_accept_v11,
	.regMr          = odl_regMr,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend_v10,
	.irecv          = odl_irecv_v10,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
	.getDeviceMr    = odl_getDeviceMr,
	.irecvConsumed  = odl_irecvConsumed,
	.makeVDevice    = NULL,
	.finalize       = odl_finalize,
	.setNetAttr     = NULL,
};

ODL_EXPORT const ncclNet_v12_t ncclNetPlugin_v12 = {
	.name           = ODL_NET_NAME,
	.init           = odl_init_v12,
	.devices        = odl_devices,
	.getProperties  = odl_getProperties_v12,
	.listen         = odl_listen_v11,
	.connect        = odl_connect_v12,
	.accept         = odl_accept_v11,
	.regMr          = odl_regMr,
	.regMrDmaBuf    = odl_regMrDmaBuf,
	.deregMr        = odl_deregMr,
	.isend          = odl_isend_v10,
	.irecv          = odl_irecv_v10,
	.iflush         = odl_iflush,
	.test           = odl_test,
	.closeSend      = odl_closeSend,
	.closeRecv      = odl_closeRecv,
	.closeListen    = odl_closeListen,
	.getDeviceMr    = odl_getDeviceMr,
	.irecvConsumed  = odl_irecvConsumed,
	.makeVDevice    = NULL,
	.finalize       = odl_finalize,
	.setNetAttr     = NULL,
};
