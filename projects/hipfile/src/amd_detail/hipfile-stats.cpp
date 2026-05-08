/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "include_internal/hipfile-stats.h"
#include "stats.h"

#include <sstream>
#include <unistd.h>

using namespace hipFile;

hipFileStatsError_t
hipFileStatsCreateContext(hipFileStatsContext_t **context, int targetPid)
{
    if (!context) {
        return hipFileStatsInvalidArgument;
    }
    if (targetPid <= 0) {
        *context = nullptr;
        return hipFileStatsInvalidArgument;
    }
    try {
        *context = reinterpret_cast<hipFileStatsContext_t *>(new StatsClient{targetPid});
    }
    catch (...) {
        *context = nullptr;
        return hipFileStatsTargetProcessNotFound;
    }
    return hipFileStatsSuccess;
}

void
hipFileStatsCloseContext(hipFileStatsContext_t *context)
{
    if (!context) {
        return;
    }
    delete reinterpret_cast<IStatsClient *>(context);
}

hipFileStatsError_t
hipFileStatsConnectToTargetProcess(hipFileStatsContext_t *context)
{
    if (!context) {
        return hipFileStatsInvalidArgument;
    }
    IStatsClient *client{reinterpret_cast<IStatsClient *>(context)};
    if (!client->connectServer()) {
        return hipFileStatsTargetProcessNotAccessible;
    }
    return hipFileStatsSuccess;
}

hipFileStatsError_t
hipFileStatsPollTargetProcess(const hipFileStatsContext_t *context, bool block)
{
    if (!context) {
        return hipFileStatsInvalidArgument;
    }
    const IStatsClient *client{reinterpret_cast<const IStatsClient *>(context)};
    if (!client->pollProcess(block ? -1 : 0)) {
        return hipFileStatsTargetProcessNotAccessible;
    }
    return hipFileStatsSuccess;
}

hipFileStatsError_t
hipFileStatsGenerateReport(const hipFileStatsContext_t *context, int fd)
{
    if (!context || fd < 0) {
        return hipFileStatsInvalidArgument;
    }
    const IStatsClient *client{reinterpret_cast<const IStatsClient *>(context)};
    std::ostringstream  stream;
    if (!client->generateReport(stream)) {
        return hipFileStatsReportGenerationFailed;
    }
    std::string report{stream.str()};
    const char *data{report.c_str()};
    size_t      total_size{report.size()};
    size_t      written{0};
    while (written < total_size) {
        ssize_t n{write(fd, data + written, total_size - written)};
        if (n <= 0) {
            if (errno == EINTR) {
                continue;
            }
            return hipFileStatsReportGenerationFailed;
        }
        written += static_cast<size_t>(n);
    }
    return hipFileStatsSuccess;
}
