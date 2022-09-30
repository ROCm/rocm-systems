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

#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

extern int
omnitrace_preload_library(void);

extern void
omnitrace_finalize(void);

extern void
omnitrace_push_trace(const char* name);

extern void
omnitrace_pop_trace(const char* name);

// extern void
// omnitrace_update_env(char*** envp);

extern void
omnitrace_init_tooling(void);

extern void
omnitrace_init(const char*, bool, const char*);

// Trampoline for the real main()
static int (*main_real)(int, char**, char**);

int
omnitrace_main(int argc, char** argv, char** envp)
{
    // prevent re-entry
    static int _reentry = 0;
    if(_reentry > 0) return -1;
    _reentry = 1;

    // set the relevant environment variables
    // omnitrace_update_env(&envp);

    const char* mode = getenv("OMNITRACE_MODE");
    omnitrace_init(mode ? mode : "sampling", false, argv[0]);
    omnitrace_init_tooling();
    omnitrace_push_trace("main(int argc, char** argv)");

    int ret = main_real(argc, argv, envp);

    omnitrace_pop_trace("main(int argc, char** argv)");
    omnitrace_finalize();

    return ret;
}

typedef int
(*omnitrace_libc_start_main)(int (*)(int, char**, char**), int, char**,
                  int (*)(int, char**, char**), void (*)(void),
                  void (*)(void), void*);

int
__libc_start_main(int (*_main)(int, char**, char**), int _argc, char** _argv,
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
    main_real = _main;

    // Find the real __libc_start_main()
    omnitrace_libc_start_main user_main = dlsym(RTLD_NEXT, "__libc_start_main");

    // disable future LD_PRELOADs
    setenv("OMNITRACE_PRELOAD", "0", 1);

    if(user_main && user_main != _this_func)
    {
        if(_preload == 0)
        {
            // call original main
            return user_main(main_real, _argc, _argv, _init, _fini, _rtld_fini,
                             _stack_end);
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
