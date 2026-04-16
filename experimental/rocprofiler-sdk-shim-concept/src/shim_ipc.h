/*
 * shim_ipc.h — memfd+sock IPC layer for the shim mock.
 *
 * Target-side: shim_ipc_init() creates the memfd, maps it, seals it,
 *   binds the abstract socket, and spawns the bg thread.
 * Consumer-side: shim_consumer_attach() connects, receives the memfd
 *   via SCM_RIGHTS, mmaps it, and returns a handle.
 *
 * Matches SHIM_MEMFD_SOCK_DESIGN.md §4, §5, §6, §10.4, §11.
 */
#ifndef SHIM_IPC_H
#define SHIM_IPC_H

#include <pthread.h>
#include <sys/types.h>

#include "shim_protocol.h"

/* Default ring size (overridden by ROCP_SHIM_RING_SIZE_MB env var). */
#define SHIM_DEFAULT_RING_SIZE (1 << 20) /* 1 MiB */

/* ---- Target-side (in-process shim) ---- */

typedef struct {
    shim_ctrl_t*       ctrl;         /* mmap'd memfd */
    shim_ring_header_t* ring_hdr;    /* points into the mmap after ctrl */
    uint8_t*           ring_data;    /* record data region */
    int                memfd;
    int                eventfd;
    int                listen_sock;
    int                client_sock;  /* current consumer's socket (-1 if none) */
    pthread_t          bg_thread;
    int                bg_thread_ok;
    pthread_mutex_t    ring_write_lock;
    int                ring_write_lock_ok;
    _Atomic int        shutdown;
} shim_ipc_target_t;

int  shim_ipc_init(shim_ipc_target_t* ipc);
void shim_ipc_destroy(shim_ipc_target_t* ipc);

/* Write a record to the ring. Returns 0 on success, -1 if ring full (dropped). */
int  shim_ring_write(shim_ipc_target_t* ipc, const shim_record_t* rec);

/* ---- Consumer-side (external process) ---- */

typedef struct {
    shim_ctrl_t*       ctrl;         /* mmap'd memfd (shared with target) */
    shim_ring_header_t* ring_hdr;
    uint8_t*           ring_data;
    int                memfd;
    int                eventfd;
    int                sock;
    size_t             mmap_size;
} shim_ipc_consumer_t;

int  shim_consumer_attach(pid_t target_pid, shim_ipc_consumer_t* out);
void shim_consumer_detach(shim_ipc_consumer_t* con);

/* Read records from the ring. Calls cb for each. Returns count read. */
typedef void (*shim_record_cb_t)(const shim_record_t* rec, void* user_data);
int  shim_consumer_poll(shim_ipc_consumer_t* con,
                        shim_record_cb_t cb, void* user_data,
                        int timeout_ms);

#endif /* SHIM_IPC_H */
