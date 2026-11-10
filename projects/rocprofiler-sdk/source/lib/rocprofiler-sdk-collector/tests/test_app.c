// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Test target for the collector unit tests:
//   collector-test-app <events> <exit status> [child]
// writes <events> rocprofiler_collector_test:ping events, then exits with <exit status>.
// With "child", it first runs itself as a child process (fork and exec), which writes as
// many events.

#define LTTNG_UST_TRACEPOINT_DEFINE
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#include "test_provider.h"

#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void
emit(int count)
{
    for(int i = 0; i < count; ++i)
        lttng_ust_tracepoint(rocprofiler_collector_test, ping, i);
}

int
main(int argc, char** argv)
{
    int events      = (argc > 1) ? atoi(argv[1]) : 1;
    int exit_status = (argc > 2) ? atoi(argv[2]) : 0;
    int with_child  = (argc > 3) && strcmp(argv[3], "child") == 0;

    if(with_child)
    {
        pid_t child = fork();
        if(child == 0)
        {
            char* child_argv[] = {argv[0], argv[1], "0", NULL};
            execv(argv[0], child_argv);
            _exit(127);
        }
        if(child > 0)
        {
            int status = 0;
            waitpid(child, &status, 0);
        }
    }
    emit(events);
    return exit_status;
}
