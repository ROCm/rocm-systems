// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define _GNU_SOURCE

#define ROCPROFV3_PUBLIC_API   __attribute__((visibility("default")));
#define ROCPROFV3_INTERNAL_API __attribute__((visibility("internal")));

#if defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define ROCPROFV3_THREAD_SANITIZER 1
#    endif
#elif defined(__SANITIZE_THREAD__)
#    define ROCPROFV3_THREAD_SANITIZER 1
#endif

#include <dlfcn.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//
// local type definitions
//
typedef int (*main_func_t)(int, char**, char**);
typedef int (*start_main_t)(int (*)(int, char**, char**),
                            int,
                            char**,
                            int (*)(int, char**, char**),
                            void (*)(void),
                            void (*)(void),
                            void*);
int
rocprofv3_libc_start_main(int (*)(int, char**, char**),
                          int,
                          char**,
                          int (*)(int, char**, char**),
                          void (*)(void),
                          void (*)(void),
                          void*) ROCPROFV3_INTERNAL_API;

int
__libc_start_main(int (*)(int, char**, char**),
                  int,
                  char**,
                  int (*)(int, char**, char**),
                  void (*)(void),
                  void (*)(void),
                  void*) ROCPROFV3_PUBLIC_API;

sighandler_t
signal(int signum, sighandler_t handler) ROCPROFV3_PUBLIC_API;

#if !defined(ROCPROFV3_THREAD_SANITIZER)
// breaks thread sanitizer
int
sigaction(int                              signum,
          const struct sigaction* restrict act,
          struct sigaction* restrict       oldact) ROCPROFV3_PUBLIC_API;
#endif

extern void
rocprofv3_set_main(main_func_t main_func) ROCPROFV3_INTERNAL_API;

extern int
rocprofv3_main(int argc, char** argv, char** envp) ROCPROFV3_INTERNAL_API;

extern sighandler_t
rocprofv3_signal(int signum, sighandler_t handler) ROCPROFV3_INTERNAL_API;

extern int
rocprofv3_sigaction(int                              signum,
                    const struct sigaction* restrict act,
                    struct sigaction* restrict       oldact) ROCPROFV3_INTERNAL_API;

int
rocprofv3_libc_start_main(int (*_main)(int, char**, char**),
                          int    _argc,
                          char** _argv,
                          int (*_init)(int, char**, char**),
                          void (*_fini)(void),
                          void (*_rtld_fini)(void),
                          void* _stack_end)
{
    // prevent re-entry
    static int _reentry = 0;
    if(_reentry > 0)
    {
        fprintf(stderr,
                "[%i][%s:%i] recursive call into %s\n",
                getpid(),
                basename(__FILE__),
                __LINE__,
                __FUNCTION__);
        fflush(stderr);
        return -1;
    }
    _reentry = 1;

    // Save the real main function address
    rocprofv3_set_main(_main);

    // Find the real __libc_start_main
    start_main_t next_main = (start_main_t) dlsym(RTLD_NEXT, "__libc_start_main");

    if(next_main)
    {
        // call rocprofv3 main function wrapper
        return next_main(rocprofv3_main, _argc, _argv, _init, _fini, _rtld_fini, _stack_end);
    }

    // grab address __libc_start_main overload
    start_main_t dflt_main = (start_main_t) dlsym(RTLD_DEFAULT, "__libc_start_main");

    // get the address of this function (approximately)
    void* this_func = __builtin_extract_return_addr(__builtin_return_address(0));

    fprintf(stderr,
            "[%s:%i][%s] Error! rocprofv3 could not find __libc_start_main! "
            "__builtin_return_address(0)=%p, RTLD_DEFAULT=%p, RTLD_NEXT=%p\n",
            basename(__FILE__),
            __LINE__,
            __FUNCTION__,
            this_func,
            (void*) dflt_main,
            (void*) next_main);
    fflush(stderr);

    return -1;
}

sighandler_t
signal(int signum, sighandler_t handler)
{
    return rocprofv3_signal(signum, handler);
}

int
sigaction(int signum, const struct sigaction* restrict act, struct sigaction* restrict oldact)
{
    return rocprofv3_sigaction(signum, act, oldact);
}

int
__libc_start_main(int (*_main)(int, char**, char**),
                  int    _argc,
                  char** _argv,
                  int (*_init)(int, char**, char**),
                  void (*_fini)(void),
                  void (*_rtld_fini)(void),
                  void* _stack_end)
{
    return rocprofv3_libc_start_main(_main, _argc, _argv, _init, _fini, _rtld_fini, _stack_end);
}
