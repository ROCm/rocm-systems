/*
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

/// \file Windows compatibility layer for missing libraries and functions

#ifndef ROCRTST_COMMON_WINDOWS_COMPAT_H_
#define ROCRTST_COMMON_WINDOWS_COMPAT_H_

#ifdef _WIN32

#include <windows.h>
#include <io.h>
#include <direct.h>
#include <process.h>
#include <cstdint>
#include <cstdlib>  // For rand, srand
#include "sys/mman.h"  // Windows mmap implementation
#include "unistd.h"    // Windows unistd implementation
#include "sys/socket.h" // Windows socket implementation

typedef int64_t ssize_t;

// Unix socket compatibility structures and constants
#define AF_UNIX 1
#define SOCK_DGRAM 2
#define SOL_SOCKET 1
#define SCM_RIGHTS 1

struct msghdr {
    void* msg_name;
    int msg_namelen;
    struct iovec* msg_iov;
    int msg_iovlen;
    void* msg_control;
    int msg_controllen;
    int msg_flags;
};

struct iovec {
    void* iov_base;
    size_t iov_len;
};

struct cmsghdr {
    int cmsg_len;
    int cmsg_level;
    int cmsg_type;
};

// Unix socket function stubs
#define CMSG_SPACE(len) (sizeof(struct cmsghdr) + (len))
#define CMSG_LEN(len) (sizeof(struct cmsghdr) + (len))
#define CMSG_FIRSTHDR(msg) (reinterpret_cast<struct cmsghdr*>((msg)->msg_control))
#define CMSG_DATA(cmsg) (reinterpret_cast<unsigned char*>(cmsg) + sizeof(struct cmsghdr))

#ifdef __cplusplus
extern "C" {
#endif
pid_t waitpid(pid_t pid, int* status, int options);
int sched_yield(void);
int is_child_process_win32(int argc, char** argv);  // Returns 1 if child, 0 if parent

// Unix socket compatibility functions
int socketpair(int domain, int type, int protocol, int sv[2]);
ssize_t sendmsg(int sockfd, const struct msghdr* msg, int flags);
ssize_t recvmsg(int sockfd, struct msghdr* msg, int flags);
#ifdef __cplusplus
}
#endif

// Environment variable functions for Windows - inline implementation
inline int setenv(const char* name, const char* value, int overwrite) {
    if (!name || !value) return -1;

    // Check if variable exists and overwrite is 0
    if (!overwrite && GetEnvironmentVariableA(name, NULL, 0) > 0) {
        return 0; // Variable exists and we shouldn't overwrite
    }

    return SetEnvironmentVariableA(name, value) ? 0 : -1;
}

inline int unsetenv(const char* name) {
    if (!name) return -1;
    return SetEnvironmentVariableA(name, NULL) ? 0 : -1;
}

// Random number functions for Windows
inline int rand_r(unsigned int* seed) {
    // Use Windows thread-safe random number generation
    // Note: This is a simple implementation, not as robust as POSIX rand_r
    srand(*seed);
    int result = rand();
    *seed = (unsigned int)result;
    return result;
}

// Windows atomic definitions (replace GCC atomics with Windows equivalents)
#include <intrin.h>
#include <atomic>

// GCC attribute compatibility for MSVC
#ifndef __attribute__
#define __attribute__(x) /* ignore */
#endif

// Replace GCC builtin functions with MSVC equivalents
#define __builtin_popcount __popcnt
#define __builtin_popcountl __popcnt64

// Replace GCC atomic functions with standard C++ atomics for Windows
#define __ATOMIC_RELEASE std::memory_order_release
#define __ATOMIC_ACQUIRE std::memory_order_acquire
#define __ATOMIC_SEQ_CST std::memory_order_seq_cst

template<typename T>
inline void __atomic_store_n(T* ptr, T val, std::memory_order order) {
    std::atomic<T>* atomic_ptr = reinterpret_cast<std::atomic<T>*>(ptr);
    atomic_ptr->store(val, order);
}

template<typename T, typename U>
inline void __atomic_store_n(T* ptr, U val, std::memory_order order) {
    std::atomic<T>* atomic_ptr = reinterpret_cast<std::atomic<T>*>(ptr);
    atomic_ptr->store(static_cast<T>(val), order);
}

template<typename T>
inline T __atomic_load_n(const T* ptr, std::memory_order order) {
    const std::atomic<T>* atomic_ptr = reinterpret_cast<const std::atomic<T>*>(ptr);
    return atomic_ptr->load(order);
}

// POSIX sleep function for Windows (seconds to milliseconds conversion)
inline unsigned int sleep(unsigned int seconds) {
    Sleep(seconds * 1000);
    return 0;
}

// POSIX time functions for Windows compatibility (timespec is already defined in Windows SDK)
// Clock types
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC     1
#endif
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME      0
#endif

// High-resolution timer for Windows
inline int clock_gettime(int clk_id, struct timespec* tp) {
    (void)clk_id; // Ignore clock type for simplicity

    if (!tp) return -1;

    // Use QueryPerformanceCounter for high-resolution timing
    static LARGE_INTEGER frequency = {0};
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    // Convert to seconds and nanoseconds
    tp->tv_sec = static_cast<long>(counter.QuadPart / frequency.QuadPart);
    tp->tv_nsec = static_cast<long>(((counter.QuadPart % frequency.QuadPart) * 1000000000) / frequency.QuadPart);

    return 0;
}

// POSIX ffs() function - Find First Set bit (1-indexed)
// Returns the position of the first (least significant) bit set in value.
// Returns 0 if no bits are set.
inline int ffs(int value) {
    if (value == 0) return 0;
    unsigned long index;
    _BitScanForward(&index, static_cast<unsigned long>(value));
    return static_cast<int>(index) + 1;  // ffs is 1-indexed (bit 1 is LSB)
}

#endif // _WIN32

#endif // ROCRTST_COMMON_WINDOWS_COMPAT_H_
