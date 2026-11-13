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
//   collector-test-app <calls> <exit status> [child] [threads=<n>]
// Each of <n> threads (default 1) makes <calls> "outer" calls, each with a nested "inner"
// call, written as rocprofiler_collector_test:<api>_enter / <api>_exit events, and a
// rocprofiler_collector_test:ping event per call. Then the process exits with <exit status>.
// With "child", it first runs itself as a child process (fork and exec), which makes as
// many calls.

#define LTTNG_UST_TRACEPOINT_DEFINE
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#include "test_provider.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int calls = 1;

static void*
make_calls(void* arg)
{
    (void) arg;
    for(int i = 0; i < calls; ++i)
    {
        char label[32];
        snprintf(label, sizeof(label), "call-%d", i);
        lttng_ust_tracepoint(rocprofiler_collector_test, ping, i);
        lttng_ust_tracepoint(rocprofiler_collector_test, outer_enter, i, label, 0.5 * i);
        lttng_ust_tracepoint(rocprofiler_collector_test, inner_enter, (uint64_t) 0xABCD0000 + i);
        lttng_ust_tracepoint(rocprofiler_collector_test, inner_exit, 2 * i);
        lttng_ust_tracepoint(rocprofiler_collector_test, outer_exit, i);
    }
    return NULL;
}

int
main(int argc, char** argv)
{
    calls           = (argc > 1) ? atoi(argv[1]) : 1;
    int exit_status = (argc > 2) ? atoi(argv[2]) : 0;
    int with_child  = 0;
    int threads     = 1;
    for(int i = 3; i < argc; ++i)
    {
        if(strcmp(argv[i], "child") == 0) with_child = 1;
        if(strncmp(argv[i], "threads=", 8) == 0) threads = atoi(argv[i] + 8);
    }
    if(threads < 1) threads = 1;

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

    pthread_t* workers = calloc((size_t) threads, sizeof(pthread_t));
    for(int i = 1; i < threads; ++i)
        pthread_create(&workers[i], NULL, make_calls, NULL);
    make_calls(NULL);
    for(int i = 1; i < threads; ++i)
        pthread_join(workers[i], NULL);
    free(workers);
    return exit_status;
}
