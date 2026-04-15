/*
 * shim_ipc.c — memfd+sock IPC implementation.
 *
 * Target side: memfd_create + seal + abstract socket + bg thread.
 * Consumer side: connect + SO_PEERCRED + SCM_RIGHTS + mmap.
 * Ring buffer: SPSC, drop-on-full, eventfd wake at watermark.
 *
 * Matches SHIM_MEMFD_SOCK_DESIGN.md §4, §5, §6, §10.4, §11.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "shim_ipc.h"

/* ---- memfd compat ---- */
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC       0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif
#ifndef F_ADD_SEALS
#define F_ADD_SEALS 1033
#endif
#ifndef F_SEAL_SHRINK
#define F_SEAL_SHRINK 0x0002
#endif
#ifndef F_SEAL_GROW
#define F_SEAL_GROW   0x0004
#endif
#ifndef F_SEAL_SEAL
#define F_SEAL_SEAL   0x0001
#endif

static int compat_memfd_create(const char* name, unsigned int flags)
{
#ifdef __NR_memfd_create
    return (int)syscall(__NR_memfd_create, name, flags);
#else
    (void)name; (void)flags; errno = ENOSYS; return -1;
#endif
}

/* ---- SCM_RIGHTS helpers ---- */

static int send_fds(int sock, int fd1, int fd2, const void* data, size_t len)
{
    struct iovec iov = { .iov_base = (void*)data, .iov_len = len };
    union {
        char buf[CMSG_SPACE(2 * sizeof(int))];
        struct cmsghdr align;
    } cmsg_buf;
    memset(&cmsg_buf, 0, sizeof(cmsg_buf));

    struct msghdr msg = {0};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;
    cmsg->cmsg_len   = CMSG_LEN(2 * sizeof(int));
    int* fdptr = (int*)CMSG_DATA(cmsg);
    fdptr[0] = fd1;
    fdptr[1] = fd2;

    return sendmsg(sock, &msg, MSG_NOSIGNAL) >= 0 ? 0 : -1;
}

static int recv_fds(int sock, int* fd1, int* fd2, void* data, size_t len)
{
    struct iovec iov = { .iov_base = data, .iov_len = len };
    union {
        char buf[CMSG_SPACE(2 * sizeof(int))];
        struct cmsghdr align;
    } cmsg_buf;
    memset(&cmsg_buf, 0, sizeof(cmsg_buf));

    struct msghdr msg = {0};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    ssize_t n = recvmsg(sock, &msg, 0);
    if (n <= 0) return -1;

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS) return -1;
    int* fdptr = (int*)CMSG_DATA(cmsg);
    *fd1 = fdptr[0];
    *fd2 = fdptr[1];
    return 0;
}

/* ---- Read start_time from /proc/self/stat ---- */
static uint64_t get_start_time(void)
{
    FILE* f = fopen("/proc/self/stat", "r");
    if (!f) return 0;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    /* Field 22 (1-indexed) is starttime. Skip past the comm field (in parens). */
    char* p = strrchr(buf, ')');
    if (!p) return 0;
    p += 2; /* skip ") " */
    uint64_t val = 0;
    for (int field = 3; field <= 22 && p; field++) {
        val = strtoull(p, &p, 10);
        while (*p == ' ') p++;
    }
    return val;
}

/* ---- Ring buffer ---- */

static size_t get_ring_size(void)
{
    const char* env = getenv("ROCP_SHIM_RING_SIZE_MB");
    size_t mb = env ? (size_t)atoi(env) : 1;
    if (mb < 1) mb = 1;
    if (mb > 256) mb = 256;
    /* Round up to next power of two */
    size_t s = mb * 1024 * 1024;
    s--;
    s |= s >> 1; s |= s >> 2; s |= s >> 4;
    s |= s >> 8; s |= s >> 16; s |= s >> 32;
    s++;
    return s;
}

int shim_ring_write(shim_ipc_target_t* ipc, const shim_record_t* rec)
{
    shim_ring_header_t* h = ipc->ring_hdr;
    uint64_t head = atomic_load_explicit(&h->head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&h->tail, memory_order_acquire);
    uint64_t next = (head + h->record_size) & h->mask;

    if (next == (tail & h->mask)) {
        atomic_fetch_add_explicit(&ipc->ctrl->events_dropped, 1,
                                  memory_order_relaxed);
        return -1;
    }

    memcpy(ipc->ring_data + (head & h->mask), rec, h->record_size);
    atomic_store_explicit(&h->head, head + h->record_size,
                          memory_order_release);
    atomic_fetch_add_explicit(&ipc->ctrl->events_traced, 1,
                              memory_order_relaxed);

    /* Watermark kick */
    uint64_t wm = ipc->ctrl->watermark_bytes;
    if (wm > 0 && ((head + h->record_size) % wm) == 0) {
        uint64_t one = 1;
        ssize_t rc = write(ipc->eventfd, &one, sizeof(one));
        (void)rc;
    }
    return 0;
}

/* ---- Background thread (§4, §10.4) ---- */

static void* shim_bg_main(void* arg)
{
    shim_ipc_target_t* ipc = (shim_ipc_target_t*)arg;

    sigset_t mask;
    sigfillset(&mask);
    pthread_sigmask(SIG_SETMASK, &mask, NULL);

    while (!atomic_load(&ipc->shutdown)) {
        int client = accept(ipc->listen_sock, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }

        /* SO_PEERCRED auth (§12) */
        struct ucred cred;
        socklen_t clen = sizeof(cred);
        if (getsockopt(client, SOL_SOCKET, SO_PEERCRED, &cred, &clen) < 0
            || cred.uid != geteuid()) {
            close(client);
            continue;
        }

        /* Send handshake + memfd + eventfd via SCM_RIGHTS (§6.2) */
        shim_hello_t hello;
        memcpy(hello.magic, SHIM_HELLO_MAGIC, 4);
        hello.struct_version  = SHIM_CTRL_VERSION;
        hello.n_registrations = ipc->ctrl->n_registrations;
        hello.total_ops       = ipc->ctrl->total_ops;
        hello.watermark_bytes = ipc->ctrl->watermark_bytes;
        hello.start_time      = ipc->ctrl->start_time;

        if (send_fds(client, ipc->memfd, ipc->eventfd,
                     &hello, sizeof(hello)) < 0) {
            close(client);
            continue;
        }

        /* Store client fd for POLLHUP liveness monitoring (§10.4).
         * Close any previous client (single-controller model §14.1). */
        if (ipc->client_sock >= 0) close(ipc->client_sock);
        ipc->client_sock = client;

        /* Monitor consumer liveness via POLLHUP. When the consumer
         * dies or disconnects, zero all op_mode slots immediately. */
        struct pollfd pfd = { .fd = client, .events = 0 };
        while (!atomic_load(&ipc->shutdown)) {
            int r = poll(&pfd, 1, 1000);
            if (r > 0 && (pfd.revents & (POLLHUP | POLLERR))) {
                /* Consumer died — zero all modes (§10.4). */
                for (uint32_t i = 0; i < ipc->ctrl->total_ops; i++)
                    atomic_store_explicit(&ipc->ctrl->op_mode[i],
                                          ROCP_SHIM_MODE_OFF,
                                          memory_order_release);
                fprintf(stderr, "[shim] consumer disconnected, slots zeroed\n");
                break;
            }
        }
        close(client);
        ipc->client_sock = -1;
        /* Back to accept() for next consumer. */
    }
    return NULL;
}

/* ---- Target-side init ---- */

int shim_ipc_init(shim_ipc_target_t* ipc)
{
    memset(ipc, 0, sizeof(*ipc));
    ipc->memfd       = -1;
    ipc->eventfd     = -1;
    ipc->listen_sock = -1;
    ipc->client_sock = -1;

    size_t ring_data_size = get_ring_size();
    size_t ring_offset    = sizeof(shim_ctrl_t);
    /* Align ring to page boundary */
    ring_offset = (ring_offset + 4095) & ~(size_t)4095;
    size_t total_size = ring_offset + sizeof(shim_ring_header_t) + ring_data_size;

    /* 1. memfd_create */
    ipc->memfd = compat_memfd_create("rocp-shim-ctrl",
                                      MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (ipc->memfd < 0) {
        perror("[shim] memfd_create");
        return -1;
    }
    if (ftruncate(ipc->memfd, (off_t)total_size) != 0) {
        perror("[shim] ftruncate");
        goto fail;
    }

    /* 2. mmap */
    void* map = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, ipc->memfd, 0);
    if (map == MAP_FAILED) { perror("[shim] mmap"); goto fail; }
    ipc->ctrl = (shim_ctrl_t*)map;

    /* 3. Initialize header */
    memset(ipc->ctrl, 0, sizeof(shim_ctrl_t));
    ipc->ctrl->magic          = SHIM_CTRL_MAGIC;
    ipc->ctrl->struct_version = SHIM_CTRL_VERSION;
    ipc->ctrl->pid            = (uint32_t)getpid();
    ipc->ctrl->start_time     = get_start_time();
    ipc->ctrl->ring_offset    = (uint32_t)ring_offset;
    ipc->ctrl->watermark_bytes = (uint32_t)(ring_data_size / 2);

    /* 4. Ring buffer header */
    ipc->ring_hdr  = (shim_ring_header_t*)((uint8_t*)map + ring_offset);
    ipc->ring_data = (uint8_t*)ipc->ring_hdr + sizeof(shim_ring_header_t);
    ipc->ring_hdr->mask        = ring_data_size - 1;
    ipc->ring_hdr->record_size = sizeof(shim_record_t);
    atomic_store(&ipc->ring_hdr->head, 0);
    atomic_store(&ipc->ring_hdr->tail, 0);

    /* 5. Seal (§5: SHRINK + GROW + SEAL) */
    if (fcntl(ipc->memfd, F_ADD_SEALS,
              F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
        perror("[shim] F_ADD_SEALS (non-fatal)");
    }

    /* 6. eventfd */
    ipc->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (ipc->eventfd < 0) { perror("[shim] eventfd"); goto fail; }

    /* 7. Abstract socket */
    ipc->listen_sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (ipc->listen_sock < 0) { perror("[shim] socket"); goto fail; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    addr.sun_path[0] = '\0';
    int n = snprintf(addr.sun_path + 1, sizeof(addr.sun_path) - 1,
                     "rocprof-shim_%d", getpid());
    socklen_t addrlen = (socklen_t)(offsetof(struct sockaddr_un, sun_path)
                                    + 1 + n);
    if (bind(ipc->listen_sock, (struct sockaddr*)&addr, addrlen) < 0) {
        perror("[shim] bind"); goto fail;
    }
    if (listen(ipc->listen_sock, 1) < 0) {
        perror("[shim] listen"); goto fail;
    }

    /* 8. Background thread */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int rc = pthread_create(&ipc->bg_thread, &attr, shim_bg_main, ipc);
    pthread_attr_destroy(&attr);
    if (rc != 0) { fprintf(stderr, "[shim] pthread_create: %d\n", rc); goto fail; }
    ipc->bg_thread_ok = 1;

    fprintf(stderr, "[shim] IPC ready: memfd=%d eventfd=%d sock=\\0rocprof-shim_%d\n",
            ipc->memfd, ipc->eventfd, getpid());
    return 0;

fail:
    shim_ipc_destroy(ipc);
    return -1;
}

void shim_ipc_destroy(shim_ipc_target_t* ipc)
{
    atomic_store(&ipc->shutdown, 1);
    if (ipc->listen_sock >= 0) {
        shutdown(ipc->listen_sock, SHUT_RDWR);
        close(ipc->listen_sock);
        ipc->listen_sock = -1;
    }
    if (ipc->client_sock >= 0) {
        close(ipc->client_sock);
        ipc->client_sock = -1;
    }
    if (ipc->bg_thread_ok) {
        pthread_join(ipc->bg_thread, NULL);
        ipc->bg_thread_ok = 0;
    }
    if (ipc->ctrl) {
        size_t ring_data_size = ipc->ring_hdr ? (ipc->ring_hdr->mask + 1) : 0;
        size_t total = ipc->ctrl->ring_offset + sizeof(shim_ring_header_t)
                       + ring_data_size;
        munmap(ipc->ctrl, total);
        ipc->ctrl = NULL;
    }
    if (ipc->eventfd >= 0)  { close(ipc->eventfd);  ipc->eventfd = -1; }
    if (ipc->memfd >= 0)    { close(ipc->memfd);    ipc->memfd = -1; }
}

/* ---- Consumer-side attach (§4) ---- */

int shim_consumer_attach(pid_t target_pid, shim_ipc_consumer_t* con)
{
    memset(con, 0, sizeof(*con));
    con->memfd = -1;
    con->eventfd = -1;
    con->sock = -1;

    /* Connect to abstract socket */
    con->sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (con->sock < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    addr.sun_path[0] = '\0';
    int n = snprintf(addr.sun_path + 1, sizeof(addr.sun_path) - 1,
                     "rocprof-shim_%d", (int)target_pid);
    socklen_t addrlen = (socklen_t)(offsetof(struct sockaddr_un, sun_path)
                                    + 1 + n);

    /* Retry for up to ~2s (target may not have finished bind yet) */
    for (int attempt = 0; attempt < 40; attempt++) {
        if (connect(con->sock, (struct sockaddr*)&addr, addrlen) == 0)
            goto connected;
        if (errno != ECONNREFUSED && errno != ENOENT) break;
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 50000000 };
        nanosleep(&ts, NULL);
    }
    fprintf(stderr, "[consumer] connect failed: %s\n", strerror(errno));
    goto fail;

connected:;
    /* Receive handshake + fds */
    shim_hello_t hello;
    if (recv_fds(con->sock, &con->memfd, &con->eventfd,
                 &hello, sizeof(hello)) < 0) {
        fprintf(stderr, "[consumer] recv_fds failed\n");
        goto fail;
    }

    /* Verify magic */
    if (memcmp(hello.magic, SHIM_HELLO_MAGIC, 4) != 0) {
        fprintf(stderr, "[consumer] bad magic\n");
        goto fail;
    }

    /* mmap the memfd */
    off_t size = lseek(con->memfd, 0, SEEK_END);
    if (size <= 0) { fprintf(stderr, "[consumer] lseek\n"); goto fail; }
    con->mmap_size = (size_t)size;

    void* map = mmap(NULL, con->mmap_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, con->memfd, 0);
    if (map == MAP_FAILED) { perror("[consumer] mmap"); goto fail; }

    con->ctrl = (shim_ctrl_t*)map;

    /* Verify header */
    if (con->ctrl->magic != SHIM_CTRL_MAGIC) {
        fprintf(stderr, "[consumer] header magic mismatch\n");
        goto fail;
    }

    uint32_t ring_off = con->ctrl->ring_offset;
    con->ring_hdr  = (shim_ring_header_t*)((uint8_t*)map + ring_off);
    con->ring_data = (uint8_t*)con->ring_hdr + sizeof(shim_ring_header_t);

    fprintf(stderr, "[consumer] attached to pid=%d, total_ops=%u, ring_mask=0x%lx\n",
            con->ctrl->pid, con->ctrl->total_ops,
            (unsigned long)con->ring_hdr->mask);
    return 0;

fail:
    shim_consumer_detach(con);
    return -1;
}

void shim_consumer_detach(shim_ipc_consumer_t* con)
{
    /* Zero all modes before disconnecting (clean detach §10.3) */
    if (con->ctrl) {
        for (uint32_t i = 0; i < con->ctrl->total_ops; i++)
            atomic_store_explicit(&con->ctrl->op_mode[i],
                                  ROCP_SHIM_MODE_OFF,
                                  memory_order_release);
        munmap(con->ctrl, con->mmap_size);
        con->ctrl = NULL;
    }
    if (con->sock >= 0)    { close(con->sock);    con->sock = -1; }
    if (con->eventfd >= 0) { close(con->eventfd); con->eventfd = -1; }
    if (con->memfd >= 0)   { close(con->memfd);   con->memfd = -1; }
}

/* ---- Consumer-side poll (§11) ---- */

int shim_consumer_poll(shim_ipc_consumer_t* con,
                       shim_record_cb_t cb, void* user_data,
                       int timeout_ms)
{
    /* Wait for eventfd wake or timeout */
    struct pollfd pfd = { .fd = con->eventfd, .events = POLLIN };
    int pr = poll(&pfd, 1, timeout_ms);
    if (pr > 0 && (pfd.revents & POLLIN)) {
        uint64_t val;
        ssize_t rc = read(con->eventfd, &val, sizeof(val));
        (void)rc;
    }

    /* Drain ring */
    shim_ring_header_t* h = con->ring_hdr;
    uint64_t head = atomic_load_explicit(&h->head, memory_order_acquire);
    uint64_t tail = atomic_load_explicit(&h->tail, memory_order_relaxed);
    int count = 0;

    while (tail != head) {
        shim_record_t* rec = (shim_record_t*)(con->ring_data + (tail & h->mask));
        if (cb) cb(rec, user_data);
        tail += h->record_size;
        count++;
    }
    atomic_store_explicit(&h->tail, tail, memory_order_release);
    return count;
}
