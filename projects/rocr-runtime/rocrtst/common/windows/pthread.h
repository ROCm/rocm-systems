/*
 * =============================================================================
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

// Windows pthread.h stub header
// This provides minimal pthread definitions for compilation compatibility

#ifndef _PTHREAD_H_STUB
#define _PTHREAD_H_STUB

#ifndef _WIN32
#include_next <pthread.h>
#else
#include <windows.h>
#include <process.h>

#define PTHREAD_CREATE_JOINABLE 0
#define PTHREAD_CREATE_DETACHED 1
#define PTHREAD_MUTEX_NORMAL 0
#define PTHREAD_MUTEX_RECURSIVE 1
#define PTHREAD_MUTEX_ERRORCHECK 2
#define PTHREAD_MUTEX_DEFAULT PTHREAD_MUTEX_NORMAL

// Error codes for pthread
#ifndef ESRCH
#define ESRCH 3
#endif

typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef int pthread_condattr_t;
typedef int pthread_mutexattr_t;
typedef struct {
    int detachstate;
    int schedpolicy;
    int schedparam;
    int inheritsched;
    int scope;
    size_t guardsize;
    void* stackaddr;
    size_t stacksize;
} pthread_attr_t;

inline int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start_routine)(void*), void* arg) {
    (void)attr;
    unsigned threadId;
    // Use _beginthreadex instead of CreateThread for better C runtime compatibility
    uintptr_t handle = _beginthreadex(NULL, 0, reinterpret_cast<unsigned (__stdcall *)(void*)>(start_routine), arg, 0, &threadId);
    *thread = reinterpret_cast<HANDLE>(handle);
    return (handle == 0) ? -1 : 0;
}

inline int pthread_join(pthread_t thread, void** retval) {
    WaitForSingleObject(thread, INFINITE);
    if (retval) {
        // Note: Windows thread exit codes are DWORD, not pointers
        // For safety, always return nullptr since rocrtst doesn't use return values
        *retval = nullptr;
    }
    CloseHandle(thread);
    return 0;
}

inline int pthread_detach(pthread_t thread) {
    CloseHandle(thread);
    return 0;
}

inline pthread_t pthread_self(void) {
    return GetCurrentThread();
}

inline int pthread_equal(pthread_t t1, pthread_t t2) {
    return (t1 == t2) ? 1 : 0;
}

inline void pthread_exit(void* retval) {
    // Note: _endthreadex takes unsigned (32-bit), but rocrtst passes nullptr as if Linux
    (void)retval; // Suppress unused parameter warning
    _endthreadex(0);
}

inline int pthread_kill(pthread_t thread, int sig) {
    (void)thread;
    (void)sig;
    return ESRCH; // Thread not found (stub)
}

inline int pthread_cancel(pthread_t thread) {
    (void)thread;
    return 0; // Success (stub)
}

inline int pthread_attr_init(pthread_attr_t* attr) {
    (void)attr;
    return 0;
}

inline int pthread_attr_destroy(pthread_attr_t* attr) {
    (void)attr;
    return 0;
}

inline int pthread_attr_setdetachstate(pthread_attr_t* attr, int detachstate) {
    (void)attr;
    (void)detachstate;
    return 0;
}

inline int pthread_mutex_init(pthread_mutex_t* mutex, const pthread_mutexattr_t* attr) {
    (void)attr;
    InitializeCriticalSection(mutex);
    return 0;
}

inline int pthread_mutex_destroy(pthread_mutex_t* mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

inline int pthread_mutex_lock(pthread_mutex_t* mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

inline int pthread_mutex_unlock(pthread_mutex_t* mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}

inline int pthread_mutex_trylock(pthread_mutex_t* mutex) {
    return TryEnterCriticalSection(mutex) ? 0 : -1;
}

inline int pthread_cond_init(pthread_cond_t* cond, const pthread_condattr_t* attr) {
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

inline int pthread_cond_destroy(pthread_cond_t* cond) {
    (void)cond; // Windows condition variables don't need explicit cleanup
    return 0;
}

inline int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
    BOOL result = SleepConditionVariableCS(cond, mutex, INFINITE);
    return result ? 0 : -1;
}

inline int pthread_cond_signal(pthread_cond_t* cond) {
    WakeConditionVariable(cond);
    return 0;
}

inline int pthread_cond_broadcast(pthread_cond_t* cond) {
    WakeAllConditionVariable(cond);
    return 0;
}
#endif // _WIN32
#endif // _PTHREAD_H_STUB
