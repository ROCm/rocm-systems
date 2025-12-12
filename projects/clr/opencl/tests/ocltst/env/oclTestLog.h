/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef CALTESTLOG_H_
#define CALTESTLOG_H_

#include <stdarg.h>
#include <stdio.h>

#include <string>

class oclLog {
 public:
  oclLog();
  virtual ~oclLog();
  virtual void vprint(char const* fmt, va_list args);
  virtual void flush();
  virtual void enable_write_to_file(std::string filename);
  virtual void disable_write_to_file();

 private:
  FILE* m_stdout_fp;
  std::string m_filename;
  bool m_writeToFileIsEnabled;
};

#endif  // CALTESTLOG_H_
