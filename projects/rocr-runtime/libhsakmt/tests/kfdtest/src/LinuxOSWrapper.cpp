/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef WIN32

#include "OSWrapper.hpp"

#include <gtest/gtest.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <drm.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

static int protection_flags[8] = {PROT_NONE, PROT_READ, PROT_WRITE, PROT_READ | PROT_WRITE,
                                  PROT_EXEC, PROT_EXEC | PROT_READ, PROT_EXEC | PROT_WRITE,
                                  PROT_EXEC | PROT_WRITE | PROT_READ};

void SetConsoleTextColor(TEXTCOLOR color) {
    // TODO: Complete
}

void Delay(int delayCount) {
    // usleep accepts time in microseconds
    usleep(delayCount * 1000);
}

void *VirtualAllocMemory(void *address, unsigned int size, int memProtection ) {
    void *ptr;

    ptr = mmap(address, size, protection_flags[memProtection], MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

    if (ptr == MAP_FAILED)
        ptr = NULL;
    return ptr;
}

bool VirtualFreeMemory(void *address, unsigned int size) {
    if (munmap(address, size) == 0)
        return true;
    else
        return false;
}

HSAuint64 GetLastErrorNo() {
    return errno;
}

bool MultiProcessTest(const char *testToRun, int numOfProcesses, int runsPerProcess) {
    // TODO: Implement
    return false;
}

bool SuspendAndWakeUp() {
    printf("Please press any key after the system suspends....\n");

    // Use "sudo apt install pm-utils" to install the "pm-suspend"
    int ret = system("sudo pm-suspend");

    if (ret == -1) {
        printf("The system linux command could not be run!\n");
        return false;
    } else {
        if (WEXITSTATUS(ret)) {
            printf("Use 'sudo apt install pm-utils' to install 'pm-suspend' on Ubuntu\n");
            return false;
        }
    }

    return true;
}

bool ReadDriverConfigValue(CONFIG_VALUE config, unsigned int& rValue) {
    return false;
}

void ComandLineArgumentsUsage() {
    printf("Invalid option value\n");
    printf("\t--hws arg\t - Force HW capability\n");
    printf("\t--profile arg\t - Test profile\n");
    printf("\t--child arg\t - Child Process\n");
    printf("\t--timeout arg\t - Time Out\n");
    printf("\t--dst_node\t - For testing multiple nodes");
    printf("\t--sleep_time\t - For testing CRIU, etc");
}

bool GetCommandLineArguments(int argc, char **argv, CommandLineArguments& rArgs) {
    int option_index = 0;

    /* Make getop silent */
    opterr = 0;
    static struct option long_options[] = {
        { "hws", required_argument, 0, 0 },
        { "profile", required_argument, 0, 0},
        { "child", required_argument, 0, 0},
        { "timeout", required_argument, 0, 0},
        { "node", required_argument, 0, 0 },
        { "dst_node", required_argument, 0, 0 },
        { "sleep_time", required_argument, 0, 0 },
        { 0, 0, 0, 0 }
    };

    rArgs.HwsEnabled = HWCAP__DEFAULT;
    rArgs.TestProfile = TESTPROFILE_RUNALL;
    rArgs.ChildProcess = false;
    rArgs.TimeOut = 0;
    rArgs.NodeId = -1;
    rArgs.DstNodeId = -1;
    rArgs.SleepTime = 0;

    while (true) {
        int c = getopt_long(argc, argv, "", long_options, &option_index);

        /* Detect the end of the options. */
        if (c != 0)
            break;

        /* If this option sets a flag, do nothing else. */
        if (long_options[option_index].flag != 0)
            continue;

        if (optarg == NULL) {
            ComandLineArgumentsUsage();
            return false;
        }

        switch (option_index) {
        /* HWS case */
        case 0:
            if (!strcmp(optarg, "disable")) {
                rArgs.HwsEnabled = HWCAP__FORCE_DISABLED;
            } else if (!strcmp(optarg, "enable")) {
                rArgs.HwsEnabled = HWCAP__FORCE_ENABLED;
            } else {
                ComandLineArgumentsUsage();
                return false;
            }
            break;
        /* TEST PROFILE */
        case 1:
            if (!strcmp(optarg, "dev")) {
                rArgs.TestProfile = TESTPROFILE_DEV;
            } else if (!strcmp(optarg, "promo")) {
                rArgs.TestProfile = TESTPROFILE_PROMO;
            } else if (!strcmp(optarg, "all")) {
                rArgs.TestProfile = TESTPROFILE_RUNALL;
            } else {
                ComandLineArgumentsUsage();
                return false;
            }
            break;

        case 2:
            rArgs.ChildProcess = true;
            break;

        case 3:
            {
                int timeOut = atoi(optarg);
                if (timeOut > 0)
                    rArgs.TimeOut = timeOut;
            }
            break;
        case 4:
            {
                int nodeId = atoi(optarg);
                if (nodeId >= 0)
                    rArgs.NodeId = nodeId;
            }
            break;
        case 5:
            {
                int dstNodeId = atoi(optarg);
                if (dstNodeId >= 0)
                    rArgs.DstNodeId = dstNodeId;
            }
            break;
        /* Sleep time - used in testing CRIU */
        case 6:
            {
                int sleepTime = atoi(optarg);
                if (sleepTime >= 0)
                    rArgs.SleepTime = sleepTime;
            }
            break;
        }
    }

    return true;
}

void HWMemoryBarrier() {
    __sync_synchronize();
}

bool StartThread(unsigned int (*thread_func)(void*), void* param, uint64_t& thread_id) {
    pthread_t id;
    bool ret = false;
    typedef void* (*pthread_func_t)(void*);

    if (!pthread_create(&id, NULL, (pthread_func_t)thread_func, param)) {
        thread_id = (pthread_t)id;
        ret = true;
    }
    return ret;
}

bool WaitForThread(uint64_t threadId) {
    return 0 == pthread_join((pthread_t)threadId, NULL);
}

HSAint64 AtomicInc(volatile HSAint64* pValue) {
    return __sync_add_and_fetch(pValue, 1);
}

void MemoryBarrier() {
       __sync_synchronize();
}

#endif  // !WIN32
