/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
