// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Phase hooks: report T1/T2 to the bench runner over the pipe fd in $BENCH_FD.
 * Record: 24 bytes {uint32 id; uint32 reserved; uint64 ns; uint64 cpu_ns}; ns is
 * CLOCK_MONOTONIC, cpu_ns is CLOCK_PROCESS_CPUTIME_ID (CPU time of all threads of
 * the calling process since its start).
 * No-op if BENCH_FD is unset/invalid. Records are dropped when the pipe is full. */
#ifndef BENCH_HOOKS_H
#define BENCH_HOOKS_H

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"
{
#endif

    enum
    {
        BENCH_HOOK_WORK_BEGIN = 1,
        BENCH_HOOK_WORK_END   = 2
    };

    typedef struct
    {
        uint32_t id;
        uint32_t reserved;
        uint64_t ns;
        uint64_t cpu_ns;
    } bench_hook_record;

#define BENCH_HOOK_FD_UNINIT (-2)

    static int      bench_hook_fd_      = BENCH_HOOK_FD_UNINIT; /* -1: disabled */
    static uint64_t bench_hook_dropped_ = 0;

    /* Forget the cached fd so the next hook re-reads BENCH_FD (used by tests). */
    static inline void bench_hook_reset(void)
    {
        bench_hook_fd_      = BENCH_HOOK_FD_UNINIT;
        bench_hook_dropped_ = 0;
    }

    static inline uint64_t bench_hook_dropped(void) { return bench_hook_dropped_; }

    static inline int bench_hook_resolve_fd_(void)
    {
        const char* env = getenv("BENCH_FD");
        char*       end = NULL;
        long        fd  = -1;
        int         fl  = 0;
        if(env != NULL && *env != '\0')
        {
            errno = 0;
            fd    = strtol(env, &end, 10);
            if(errno != 0 || *end != '\0' || fd < 0 || fd > 0x7fffffffL) fd = -1;
        }
        if(fd >= 0)
        {
            fl = fcntl((int) fd, F_GETFL);
            if(fl < 0 || fcntl((int) fd, F_SETFL, fl | O_NONBLOCK) < 0) fd = -1;
        }
        return (int) fd;
    }

    static inline void bench_hook(uint32_t id)
    {
        bench_hook_record rec;
        struct timespec   ts;
        struct timespec   cpu;
        if(bench_hook_fd_ == BENCH_HOOK_FD_UNINIT)
            bench_hook_fd_ = bench_hook_resolve_fd_();
        if(bench_hook_fd_ < 0) return;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu);
        rec.id       = id;
        rec.reserved = 0;
        rec.ns       = (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
        rec.cpu_ns   = (uint64_t) cpu.tv_sec * 1000000000ULL + (uint64_t) cpu.tv_nsec;
        if(write(bench_hook_fd_, &rec, sizeof(rec)) != (ssize_t) sizeof(rec))
            ++bench_hook_dropped_;
    }

#ifdef __cplusplus
}
#endif

#endif /* BENCH_HOOKS_H */
