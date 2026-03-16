/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef OCLLOG_H_
#define OCLLOG_H_

#ifdef _WIN32

#ifdef OCLTST_LOG_BUILD
#define DLLIMPORT __declspec(dllexport)
#else
#define DLLIMPORT __declspec(dllimport)
#endif  // OCLTST_ENV_BUILD

#else
#define DLLIMPORT

#endif  // _WIN32

enum oclLoggingLevel {
  OCLTEST_LOG_ALWAYS,
  OCLTEST_LOG_VERBOSE,
};

extern DLLIMPORT void oclTestLog(oclLoggingLevel logLevel, const char* fmt, ...);
extern DLLIMPORT void oclTestSetLogLevel(int level);
extern DLLIMPORT void oclTestEnableLogToFile(const char* filename);

#endif  // OCLLOG_H_
