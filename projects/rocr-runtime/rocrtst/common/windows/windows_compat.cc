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

/// \file Windows stub implementations for missing libraries

#ifdef _WIN32

#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdio>  // For snprintf
#include <windows.h>
#include <process.h>
#include <gtest/gtest.h>  // For getting current test info

#include "windows_compat.h"
#include "pthread.h"
#include "unistd.h"
#include "getopt.h"

extern "C" {
// POSIX process APIs
// For IPC test
pid_t fork(void) {
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);

    // Get current executable path
    char exe_path[MAX_PATH];
    if (GetModuleFileNameA(NULL, exe_path, MAX_PATH) == 0) {
        return -1;
    }

    // Build command line for child process
    // Get the current test name from GoogleTest
    const testing::TestInfo* test_info = testing::UnitTest::GetInstance()->current_test_info();
    char cmd_line[4096];
    if (test_info) {
        // Pass --gtest_filter to run only the current test in the child
        snprintf(cmd_line, sizeof(cmd_line), "\"%s\" --gtest_filter=%s.%s",
                exe_path, test_info->test_case_name(), test_info->name());
    } else {
        // Fallback if we can't get test info
        snprintf(cmd_line, sizeof(cmd_line), "\"%s\"", exe_path);
    }

    // Set environment variables for child process to inherit
    DWORD parent_pid = GetCurrentProcessId();
    char parent_pid_str[32];
    snprintf(parent_pid_str, sizeof(parent_pid_str), "%lu", parent_pid);

    SetEnvironmentVariableA("ROCR_IPC_CHILD_PROCESS", "1");
    SetEnvironmentVariableA("ROCR_IPC_PARENT_PID", parent_pid_str);

    // Create child process with command line arguments
    if (!CreateProcessA(NULL, cmd_line, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        return -1;
    }

    // Close thread handle, return process handle as pid
    CloseHandle(pi.hThread);
    return reinterpret_cast<pid_t>(pi.hProcess);
}

pid_t waitpid(pid_t pid, int* status, int options) {
    (void)options;
    HANDLE process = reinterpret_cast<HANDLE>(static_cast<intptr_t>(pid));
    DWORD result = WaitForSingleObject(process, INFINITE);
    if (status) {
        *status = (result == WAIT_OBJECT_0) ? 0 : -1;
    }
    CloseHandle(process);
    return pid;
}

// POSIX scheduling API
int sched_yield(void) {
    SwitchToThread();
    return 0;
}

// Helper function for IPC tests
int is_child_process_win32(int argc, char** argv) {
    (void)argc; (void)argv;  // Parameters not used with environment variable approach
    // Check if we're a child process using the same environment variable that fork() sets
    const char* child_marker = getenv("ROCR_IPC_CHILD_PROCESS");
    return (child_marker != nullptr) ? 1 : 0;  // Return 1 if child, 0 if parent
}

// getopt implementation for Windows
// Global variables used by getopt functions (matches POSIX getopt interface)
char *optarg = nullptr;         // Points to the argument of the current option
int optind = 1;                 // Index of the next element to be processed in argv

// Internal state variables for option parsing (must be static to persist between calls)
static int optpos = 0;          // Position within current argument string
static int optreset = 0;        // Flag to reset option parsing state

int getopt(int argc, char * const argv[], const char *optstring) {
    const char *oli;
    char *place;
    char optopt;

    if (optreset || !optpos) {
        optreset = 0;
        if (optind >= argc || argv[optind][0] != '-') {
            optpos = 0;
            return -1;
        }
        place = argv[optind];
        if (place[1] && place[1] == '-') {  // found "--"
            ++optind;
            optpos = 0;
            return -1;
        }
        optpos = 1;
    }

    place = argv[optind];
    optopt = place[optpos++];

    if (optopt == ':' || !(oli = strchr(optstring, optopt))) {
        /*
         * if the user didn't specify '-' as an option,
         * assume it means -1.
         */
        if (optopt == '-')
            return -1;
        if (!place[optpos])
            ++optind;
        if (*optstring != ':') {
            char str[] = "illegal option\n";
            write(STDERR_FILENO, str, sizeof(str) - 1);
        }
        return '?';
    }

    if (oli[1] != ':') {            /* don't need argument */
        optarg = nullptr;
        if (!place[optpos])
            ++optind;
    } else {                        /* need an argument */
        if (place[optpos])          /* no white space */
            optarg = &place[optpos];
        else if (argc <= ++optind) {   /* no arg */
            optpos = 0;
            if (*optstring == ':')
                return ':';
            {
                char str[] = "option requires an argument\n";
                write(STDERR_FILENO, str, sizeof(str) - 1);
            }
            return '?';
        } else                      /* white space */
            optarg = argv[optind];
        optpos = 0;
        ++optind;
    }
    return static_cast<int>(optopt);
}

int getopt_long(int argc, char * const argv[], const char *optstring,
                const struct option *longopts, int *longindex) {
    const char *tmp;
    int idx, offs;

    if (longindex) *longindex = -1;

    /* short circuit getopt functionality */
    if (!longopts)
        return getopt(argc, argv, optstring);

    /*
     * getopt_long permutes argv as it scans arguments; we have to
     * return to the original state when reentering, so we just don't
     * permute at all
     */
    if (optreset || !optpos) {
        optreset = 0;
        if (optind >= argc || argv[optind][0] != '-') {
            optpos = 0;
            return -1;
        }
        if (argv[optind][1] == '-' && argv[optind][2] == 0) {
            optind++;
            optpos = 0;
            return -1;
        }
    }

    if (argv[optind][0] == '-' && argv[optind][1] == '-') {
        /* long option */
        char *arg = argv[optind] + 2;
        const struct option *o = longopts;
        idx = 0;

        while (o->name) {
            tmp = o->name;
            offs = 0;
            while (*tmp && *tmp == arg[offs])
                tmp++, offs++;
            if (!*tmp) {
                /* exact match found */
                if (longindex)
                    *longindex = idx;
                if (!arg[offs]) {
                    /* no argument needed or provided */
                    if (o->has_arg == required_argument) {
                        if (++optind >= argc) {
                            {
                                char str[] = "option requires an argument\n";
                                write(STDERR_FILENO, str, sizeof(str) - 1);
                            }
                            optind--;
                            return '?';
                        }
                        optarg = argv[optind];
                    } else {
                        optarg = nullptr;
                    }
                } else if (arg[offs] == '=') {
                    /* argument provided inline */
                    if (o->has_arg == no_argument) {
                        {
                            char str[] = "option doesn't allow an argument\n";
                            write(STDERR_FILENO, str, sizeof(str) - 1);
                        }
                        optind++;
                        return '?';
                    }
                    optarg = arg + offs + 1;
                } else {
                    /* partial match or extra characters */
                    continue;
                }

                optind++;
                optpos = 0;
                if (o->flag) {
                    *o->flag = o->val;
                    return 0;
                }
                return o->val;
            }
            o++;
            idx++;
        }

        /* no match found */
        {
            char str[] = "unrecognized option\n";
            write(STDERR_FILENO, str, sizeof(str) - 1);
        }
        optind++;
        optpos = 0;
        return '?';
    }

    /* short option */
    return getopt(argc, argv, optstring);
}

// Unix socket compatibility stubs
// These socket need to work with dmabuf in Linux.
// We may / may not need to implement this in Widnows to test dmabuf
int socketpair(int domain, int type, int protocol, int sv[2]) {
    (void)domain; (void)type; (void)protocol;
    // Create a pair of connected sockets using localhost
    // This is a simplified implementation for testing
    sv[0] = sv[1] = -1;
    return -1;  // Not supported on Windows for now
}

ssize_t sendmsg(int sockfd, const struct msghdr* msg, int flags) {
    (void)sockfd; (void)msg; (void)flags;
    return -1;  // Not supported on Windows for now
}

ssize_t recvmsg(int sockfd, struct msghdr* msg, int flags) {
    (void)sockfd; (void)msg; (void)flags;
    return -1;  // Not supported on Windows for now
}

} // extern "C"

#endif // _WIN32
