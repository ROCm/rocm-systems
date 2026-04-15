/*
 * shim_protocol.h — Shared types between target-side shim and consumer.
 *
 * Defines the memfd layout, record format, mode selectors, filter
 * structures, and handshake protocol. Both the in-process shim and the
 * external consumer link against this header.
 *
 * Matches SHIM_MEMFD_SOCK_DESIGN.md §5, §6, §11, §13.5, §13.6.
 */
#ifndef SHIM_PROTOCOL_H
#define SHIM_PROTOCOL_H

#include <stdatomic.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Mode selectors (§5.1)                                               */
/* ------------------------------------------------------------------ */

#define ROCP_SHIM_MODE_OFF         0
#define ROCP_SHIM_MODE_RECORD      1
#define ROCP_SHIM_MODE_RECORD_ARGS 2
#define ROCP_SHIM_MODE_RECORD_FULL 3

/* ------------------------------------------------------------------ */
/* Record format (§7A.5 + §7B.1)                                       */
/* ------------------------------------------------------------------ */

#define SHIM_RECORD_ARG_BYTES 128
#define SHIM_OP_NAME_MAX      64

/* Per-op metadata (name + arg count). Stored in the memfd header
 * alongside registrations, so the consumer can print meaningful
 * function names and decode args without linking the target's headers. */
typedef struct {
    char     name[SHIM_OP_NAME_MAX];
    uint32_t n_args;
    uint32_t arg_total_bytes;   /* sizeof the packed arg struct */
} shim_op_info_t;

#define SHIM_MAX_OP_INFOS SHIM_MAX_TOTAL_OPS

typedef struct {
    uint64_t internal;
    uint64_t external;
    uint64_t ancestor;
} shim_correlation_id_t;

typedef struct {
    uint64_t                tsc;
    uint32_t                kind;
    uint32_t                op;
    uint32_t                phase;       /* 0=ENTER, 1=EXIT, 2=EXIT_UNREACHED */
    uint32_t                cpu;
    uint64_t                thread_id;
    shim_correlation_id_t   correlation_id;
    uint32_t                slot_idx;
    uint32_t                arg_overflow; /* 0 = ok, 1 = ARG_TRUNCATED */
    uint32_t                arg_bytes;
    uint32_t                reserved;
    uint8_t                 args[SHIM_RECORD_ARG_BYTES];
} shim_record_t;

#define SHIM_PHASE_ENTER          0
#define SHIM_PHASE_EXIT           1
#define SHIM_PHASE_EXIT_UNREACHED 2

/* ------------------------------------------------------------------ */
/* Per-table registration (§13.6)                                      */
/* ------------------------------------------------------------------ */

#define SHIM_MAX_REGISTRATIONS 16
#define SHIM_TABLE_NAME_MAX    32

typedef struct {
    char     name[SHIM_TABLE_NAME_MAX];
    uint32_t lib_instance;
    uint32_t major_version;
    uint32_t minor_version;
    uint32_t slot_base;
    uint32_t n_ops;
    uint32_t reserved;
} shim_table_registration_t;

/* ------------------------------------------------------------------ */
/* Value-filter rule (§13.5 Phase 3)                                   */
/* ------------------------------------------------------------------ */

#define SHIM_CMP_EQ      0
#define SHIM_CMP_NEQ     1
#define SHIM_CMP_GT      2
#define SHIM_CMP_LT      3
#define SHIM_CMP_BITMASK 4

#define SHIM_MAX_VALUE_RULES_PER_OP 4

typedef struct {
    uint16_t arg_index;
    uint16_t comparison;
    uint64_t operand;
} shim_value_rule_t;

/* ------------------------------------------------------------------ */
/* Filter bitmap (§13.5 Phase 1 — _Atomic uint64_t per 64 ops)        */
/* ------------------------------------------------------------------ */

#define SHIM_MAX_TOTAL_OPS   1024
#define SHIM_FILTER_WORDS    (SHIM_MAX_TOTAL_OPS / 64)

/* ------------------------------------------------------------------ */
/* Ring buffer header (§11)                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    _Atomic uint64_t head;
    _Atomic uint64_t tail;
    uint64_t         mask;         /* capacity - 1, power of two */
    uint32_t         record_size;  /* sizeof(shim_record_t) */
    uint32_t         reserved;
} shim_ring_header_t;

/* ------------------------------------------------------------------ */
/* Memfd shared-memory header (§5)                                     */
/* ------------------------------------------------------------------ */

#define SHIM_CTRL_MAGIC    0x4D494853   /* 'S''H''I''M' */
#define SHIM_CTRL_VERSION  1

typedef struct {
    /* Fixed header — const after init */
    uint32_t magic;
    uint32_t struct_version;
    uint32_t pid;
    uint32_t pad0;
    uint64_t start_time;
    uint32_t n_registrations;
    uint32_t total_ops;
    uint32_t watermark_bytes;
    uint32_t ring_offset;          /* byte offset to shim_ring_header_t */

    /* Counters — atomic, written by target */
    _Atomic uint64_t events_traced;
    _Atomic uint64_t events_dropped;

    /* Generation counter — atomic, written by consumer */
    _Atomic uint32_t gen_counter;
    uint32_t pad1;

    /* Per-table registrations — const after init */
    shim_table_registration_t registrations[SHIM_MAX_REGISTRATIONS];

    /* Name-filter bitmaps — atomic word updates (§5 P1 fix) */
    _Atomic uint64_t name_filter[SHIM_FILTER_WORDS];

    /* Per-op mode selectors — atomic, written by consumer */
    _Atomic uint32_t op_mode[SHIM_MAX_TOTAL_OPS];

    /* Per-op value-filter rules + counts */
    uint32_t         value_filter_count[SHIM_MAX_TOTAL_OPS];
    shim_value_rule_t value_rules[SHIM_MAX_TOTAL_OPS][SHIM_MAX_VALUE_RULES_PER_OP];

    /* Per-op arg policy */
    uint32_t         arg_policy[SHIM_MAX_TOTAL_OPS];

    /* Per-op metadata — function names + arg layout info.
     * Consumer reads this to print meaningful names and decode args. */
    shim_op_info_t   op_info[SHIM_MAX_OP_INFOS];

    /* Ring buffer follows at ring_offset */
} shim_ctrl_t;

/* ------------------------------------------------------------------ */
/* Socket handshake (§6.2)                                             */
/* ------------------------------------------------------------------ */

#define SHIM_HELLO_MAGIC "SHIM"

typedef struct {
    char     magic[4];
    uint32_t struct_version;
    uint32_t n_registrations;
    uint32_t total_ops;
    uint32_t watermark_bytes;
    uint64_t start_time;
    /* Per-table versions follow in the mmap'd header after attach */
} shim_hello_t;

/* Query commands (§6.3) */
#define SHIM_Q_STATS  1
#define SHIM_Q_DETACH 2
#define SHIM_Q_FLUSH  3

typedef struct {
    uint32_t type;
    uint32_t reserved;
    uint64_t arg0;
} shim_query_t;

typedef struct {
    uint32_t type;
    uint32_t status;
    uint64_t arg0;
    uint64_t events_traced;
    uint64_t events_dropped;
} shim_response_t;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static inline int shim_filter_test(const _Atomic uint64_t* bitmap,
                                   uint32_t slot_idx)
{
    uint32_t word = slot_idx / 64;
    uint64_t bit  = 1ULL << (slot_idx % 64);
    if (word >= SHIM_FILTER_WORDS) return 1; /* out of range = pass */
    return (atomic_load_explicit(&bitmap[word], memory_order_relaxed) & bit) != 0;
}

static inline void shim_filter_set(volatile _Atomic uint64_t* bitmap,
                                   uint32_t slot_idx)
{
    uint32_t word = slot_idx / 64;
    uint64_t bit  = 1ULL << (slot_idx % 64);
    if (word < SHIM_FILTER_WORDS)
        atomic_fetch_or_explicit(&bitmap[word], bit, memory_order_relaxed);
}

static inline void shim_filter_clear(volatile _Atomic uint64_t* bitmap,
                                     uint32_t slot_idx)
{
    uint32_t word = slot_idx / 64;
    uint64_t bit  = 1ULL << (slot_idx % 64);
    if (word < SHIM_FILTER_WORDS)
        atomic_fetch_and_explicit(&bitmap[word], ~bit, memory_order_relaxed);
}

static inline uint64_t shim_rdtsc(void)
{
#if defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

#endif /* SHIM_PROTOCOL_H */
