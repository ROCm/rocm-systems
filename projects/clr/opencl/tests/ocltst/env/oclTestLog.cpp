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

#include "oclTestLog.h"

#include <cassert>
#include <cstring>

#include "OCLLog.h"

oclLog::oclLog() : m_stdout_fp(stdout), m_filename(""), m_writeToFileIsEnabled(false) {}

oclLog::~oclLog() { disable_write_to_file(); }

void oclLog::enable_write_to_file(std::string filename) {
  m_writeToFileIsEnabled = true;
  m_filename = filename;
  FILE* fp = fopen(m_filename.c_str(), "w");
  if (fp == NULL) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "ERROR: Cannot open file %s. Disabling logging to file.\n",
               filename.c_str());
    m_writeToFileIsEnabled = false;
  } else {
    fclose(fp);
  }
}

void oclLog::disable_write_to_file() { m_writeToFileIsEnabled = false; }

void oclLog::vprint(char const* fmt, va_list args) {
  // hack for fixing the lnx64bit segfault and
  // garbage printing in file. XXX 2048 a magic number
  char buffer[4096];

  memset(buffer, 0, sizeof(buffer));
  int rc = vsnprintf(buffer, sizeof(buffer), fmt, args);
  assert(rc >= 0 && rc != sizeof(buffer));

  fputs(buffer, m_stdout_fp);
  if (m_writeToFileIsEnabled) {
    FILE* fp = fopen(m_filename.c_str(), "a");
    if (fp == NULL) {
      oclTestLog(OCLTEST_LOG_ALWAYS, "ERROR: Cannot open file %s. Disabling logging to file.\n",
                 m_filename.c_str());
      m_writeToFileIsEnabled = false;
    }
    fputs(buffer, fp);
    fclose(fp);
  }
}

void oclLog::flush() { fflush(m_stdout_fp); }

static oclLog& theLog() {
  static oclLog Log;
  return Log;
}

static oclLoggingLevel currentLevel = OCLTEST_LOG_ALWAYS;
static float logcount = 0.0f;

void oclTestLog(oclLoggingLevel logLevel, const char* fmt, ...) {
  logcount += 1.0f;

  if (logLevel <= currentLevel) {
    va_list args;
    va_start(args, fmt);

    theLog().vprint(fmt, args);
    theLog().flush();

    va_end(args);
  }
}

void oclTestEnableLogToFile(const char* filename) { theLog().enable_write_to_file(filename); }

void oclTestSetLogLevel(int level) {
  if (level >= 0) {
    currentLevel = static_cast<oclLoggingLevel>(level);
  }
}
