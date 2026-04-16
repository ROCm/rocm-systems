/*
 * shim_protocol.h — shared types between the rocprofiler-sdk tool shim and
 * the out-of-process consumer.
 *
 * In the current concept, rocprofiler-sdk owns tracing and record production.
 * The shim only snapshots SDK-produced buffer records into a shared-memory ring
 * and exposes that ring over memfd + Unix socket transport.
 */
#ifndef SHIM_PROTOCOL_H
#define SHIM_PROTOCOL_H

#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

#define SHIM_RECORD_PAYLOAD_BYTES 1024
#define SHIM_RUNTIME_NAME_MAX     32
#define SHIM_MAX_RUNTIME_REGISTRATIONS 16

#define SHIM_MAX_TOTAL_OPS 1
#define SHIM_FILTER_WORDS  1

typedef union
{
    uint64_t words[SHIM_RECORD_PAYLOAD_BYTES / sizeof(uint64_t)];
    uint8_t  bytes[SHIM_RECORD_PAYLOAD_BYTES];
} shim_payload_storage_t;

typedef struct
{
    uint64_t sequence;
    uint32_t category;
    uint32_t kind;
    uint32_t payload_bytes;
    uint32_t payload_truncated;
    shim_payload_storage_t payload;
} shim_record_t;

typedef struct
{
    char     common_name[SHIM_RUNTIME_NAME_MAX];
    uint32_t major_version;
    uint32_t minor_version;
    uint32_t patch_version;
    uint32_t reserved;
    uint64_t instance;
} shim_runtime_registration_t;

typedef struct
{
    _Atomic uint64_t head;
    _Atomic uint64_t tail;
    uint64_t         mask;
    uint32_t         record_size;
    uint32_t         reserved;
} shim_ring_header_t;

#define SHIM_CTRL_MAGIC   0x4D494853
#define SHIM_CTRL_VERSION 2

typedef struct
{
    uint32_t magic;
    uint32_t struct_version;
    uint32_t pid;
    uint32_t pad0;
    uint64_t start_time;
    uint32_t n_runtime_registrations;
    uint32_t active_services;
    uint32_t watermark_bytes;
    uint32_t ring_offset;

    _Atomic uint32_t capture_enabled;
    uint32_t         pad1[3];

    _Atomic uint64_t events_traced;
    _Atomic uint64_t events_dropped;
    _Atomic uint64_t sdk_drop_count;
    uint64_t         reserved0;

    shim_runtime_registration_t runtime_registrations[SHIM_MAX_RUNTIME_REGISTRATIONS];
} shim_ctrl_t;

#define SHIM_HELLO_MAGIC "SHIM"

typedef struct
{
    char     magic[4];
    uint32_t struct_version;
    uint32_t n_runtime_registrations;
    uint32_t active_services;
    uint32_t watermark_bytes;
    uint64_t start_time;
} shim_hello_t;

static inline uint64_t
shim_rdtsc(void)
{
#if defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t) hi << 32) | lo;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#endif
}

#endif /* SHIM_PROTOCOL_H */
