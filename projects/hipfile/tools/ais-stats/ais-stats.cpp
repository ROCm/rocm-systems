/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "include_internal/hipfile-stats.h"
#include <iostream>
#include <unistd.h>
#include <cstring>

static void
printUsage()
{
    std::cerr << "Usage:\n"
              << "  ais-stats <program> [args...]\n"
              << "  ais-stats -p <PID> [-i]\n"
              << "  -p <PID> Process to collect data from\n"
              << "  -i Generate report immediately instead of waiting for process to exit\n";
}

int
main(int argc, char *argv[])
{
    pid_t pid{-1};
    bool  imm{false};
    if (argc < 2) {
        printUsage();
        return 1;
    }
    if (std::strcmp(argv[1], "-p") == 0) {
        if (argc < 3) {
            printUsage();
            return 1;
        }
        pid = std::atoi(argv[2]);
        imm = argc == 4 && std::strcmp(argv[3], "-i") == 0;
    }
    else {
        pid = fork();
        if (pid < 0) {
            std::cerr << "Failed to launch " << argv[1] << '\n';
            return 1;
        }
        if (pid == 0) {
            execvp(argv[1], &argv[1]);
            std::cerr << "Failed to launch " << argv[1] << '\n';
            return 1;
        }
    }

    hipFileStatsContext_t *context;
    if (hipFileStatsCreateContext(&context, pid) != hipFileStatsSuccess) {
        std::cerr << "Failed to create context for target process.\n";
        return 1;
    }
    if (hipFileStatsConnectToTargetProcess(context) != hipFileStatsSuccess) {
        std::cerr << "Failed to connect to target process.\n";
        hipFileStatsCloseContext(context);
        return 1;
    }
    if (!imm) {
        if (hipFileStatsPollTargetProcess(context, true) != hipFileStatsSuccess) {
            std::cerr << "Failed to poll target process.\n";
            hipFileStatsCloseContext(context);
            return 1;
        }
    }
    int fd{dup(STDOUT_FILENO)};
    if (fd < 0) {
        std::cerr << "Failed to duplicate stdout file descriptor.\n";
        hipFileStatsCloseContext(context);
        return 1;
    }
    if (hipFileStatsGenerateReport(context, fd) != hipFileStatsSuccess) {
        std::cerr << "Failed to generate report from target process.\n";
        close(fd);
        hipFileStatsCloseContext(context);
        return 1;
    }
    close(fd);
    hipFileStatsCloseContext(context);
    return 0;
}
