// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#define OMNITRACE_PUBLIC_API   __attribute__((visibility("default")));
#define OMNITRACE_HIDDEN_API   __attribute__((visibility("hidden")));
#define OMNITRACE_INTERNAL_API __attribute__((visibility("internal")));

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//
// local type definitions
//
typedef int (*main_func_t)(int, char**, char**);
typedef int (*start_main_t)(int (*)(int, char**, char**), int, char**,
                            int (*)(int, char**, char**), void (*)(void), void (*)(void),
                            void*);

//
// local function declarations
//
int
omnitrace_libc_start_main(int (*)(int, char**, char**), int, char**,
                          int (*)(int, char**, char**), void (*)(void), void (*)(void),
                          void*) OMNITRACE_INTERNAL_API;

int
__libc_start_main(int (*)(int, char**, char**), int, char**, int (*)(int, char**, char**),
                  void (*)(void), void (*)(void), void*) OMNITRACE_PUBLIC_API;

//
// external function declarations
//
extern int
omnitrace_preload_library(void);

extern void
omnitrace_finalize(void);

extern void
omnitrace_push_trace(const char* name);

extern void
omnitrace_pop_trace(const char* name);

extern void
omnitrace_init_tooling(void);

extern void
omnitrace_init(const char*, bool, const char*);

extern char*
basename(const char*);

extern void omnitrace_set_main(main_func_t) OMNITRACE_INTERNAL_API;

extern int
omnitrace_main(int argc, char** argv, char** envp) OMNITRACE_INTERNAL_API;

int
omnitrace_libc_start_main(int (*_main)(int, char**, char**), int _argc, char** _argv,
                          int (*_init)(int, char**, char**), void (*_fini)(void),
                          void (*_rtld_fini)(void), void* _stack_end)
{
    int _preload = omnitrace_preload_library();

    // prevent re-entry
    static int _reentry = 0;
    if(_reentry > 0) return -1;
    _reentry = 1;

    // get the address of this function
    void* _this_func = __builtin_return_address(0);

    // Save the real main function address
    omnitrace_set_main(_main);

    // Find the real __libc_start_main()
    start_main_t user_main = dlsym(RTLD_NEXT, "__libc_start_main");

    // disable future LD_PRELOADs
    setenv("OMNITRACE_PRELOAD", "0", 1);

    if(user_main && user_main != _this_func)
    {
        if(_preload == 0)
        {
            // call original main
            return user_main(_main, _argc, _argv, _init, _fini, _rtld_fini, _stack_end);
        }
        else
        {
            // call omnitrace main function wrapper
            return user_main(omnitrace_main, _argc, _argv, _init, _fini, _rtld_fini,
                             _stack_end);
        }
    }
    else
    {
        fputs("Error! omnitrace could not find __libc_start_main!", stderr);
        return -1;
    }
}

int
__libc_start_main(int (*_main)(int, char**, char**), int _argc, char** _argv,
                  int (*_init)(int, char**, char**), void (*_fini)(void),
                  void (*_rtld_fini)(void), void* _stack_end)
{
    return omnitrace_libc_start_main(_main, _argc, _argv, _init, _fini, _rtld_fini,
                                     _stack_end);
}
