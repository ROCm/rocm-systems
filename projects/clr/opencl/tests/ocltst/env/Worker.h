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

#ifndef OCL_TEST_WORKER_H
#define OCL_TEST_WORKER_H

/////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdio.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "Module.h"
#include "OCLTest.h"
#include "OCLTestList.h"
#include "ResultStruct.h"
#include "Timer.h"
#include "getopt.h"
#include "pfm.h"

/////////////////////////////////////////////////////////////////////////////

typedef void* (*TestMethod)(void* param);

/////////////////////////////////////////////////////////////////////////////

class Worker {
 public:
  Worker()
      : m_wrapper(0),
        m_module(0),
        m_run(0),
        m_id(0),
        m_subtest(0),
        m_testindex(0),
        m_dump(false),
        m_display(false),
        m_useCPU(false),
        m_window(0),
        m_width(0),
        m_height(0),
        m_buffer(0),
        m_perflab(false),
        m_deviceId(0),
        m_platform(0) {
    // EMPTY!
  }

  Worker(OCLWrapper* wrapper, Module* module, TestMethod run, unsigned int id, unsigned int subtest,
         unsigned int testindex, bool dump, bool view, bool useCPU, void* window, unsigned int x,
         unsigned int y, bool perflab, unsigned int deviceId = 0, unsigned int platform = 0)
      : m_wrapper(wrapper),
        m_module(module),
        m_run(run),
        m_id(id),
        m_subtest(subtest),
        m_testindex(testindex),
        m_dump(dump),
        m_display(view),
        m_useCPU(useCPU),
        m_window(window),
        m_width(x),
        m_height(y),
        m_buffer(0),
        m_perflab(perflab),
        m_deviceId(deviceId),
        m_platform(platform) {
    if (m_dump == true || m_display == true) {
      m_buffer = new float[4 * m_width * m_height];
      if (m_buffer != 0) {
        memset(m_buffer, 0, 4 * m_width * m_height * sizeof(float));
      } else {
        m_dump = false;
        m_display = false;
      }
    }
    m_result = new TestResult(0.0f);
  }

  Worker(const Worker& w) {
    if (this == &w) return;

    if (m_buffer) delete[] m_buffer;
    m_buffer = 0;

    m_wrapper = w.m_wrapper;
    m_module = w.m_module;
    m_run = w.m_run;
    m_id = w.m_id;
    m_subtest = w.m_subtest;
    m_testindex = w.m_testindex;
    m_dump = w.m_dump;
    m_display = w.m_display;
    m_useCPU = w.m_useCPU;
    m_window = w.m_window;
    m_width = w.m_width;
    m_height = w.m_height;
    m_perflab = w.m_perflab;
    m_deviceId = w.m_deviceId;
    m_result = w.m_result;
    m_platform = w.m_platform;

    if (w.m_buffer) {
      m_buffer = new float[4 * m_width * m_height];
      if (m_buffer != 0) {
        memcpy(m_buffer, w.m_buffer, 4 * m_width * m_height * sizeof(float));
      }
    }
  }

  ~Worker() {
    if (m_buffer) delete[] m_buffer;
    m_buffer = 0;
    delete m_result;
    m_result = 0;
  }

  OCLWrapper* getOCLWrapper() { return m_wrapper; }
  Module* getModule() { return m_module; }
  TestMethod getTestMethod() { return m_run; }
  unsigned int getId() { return m_id; }
  unsigned int getSubTest() { return m_subtest; }
  unsigned int getTestIndex() { return m_testindex; }
  bool isDumpEnabled() { return m_dump; }
  bool isDisplayEnabled() { return m_display; }
  bool isCPUEnabled() { return m_useCPU; }
  void* getWindow() { return m_window; }
  unsigned int getWidth() { return m_width; }
  unsigned int getHeight() { return m_height; }
  float* getBuffer() { return m_buffer; }
  bool getPerflab() { return m_perflab; }
  unsigned int getDeviceId() { return m_deviceId; }
  TestResult* getResult() { return m_result; }
  unsigned int getPlatformID() { return m_platform; }

 private:
  OCLWrapper* m_wrapper;
  Module* m_module;
  TestMethod m_run;
  unsigned int m_id;
  unsigned int m_subtest;
  unsigned int m_testindex;
  bool m_dump;
  bool m_display;
  bool m_useCPU;
  void* m_window;
  unsigned int m_width;
  unsigned int m_height;
  float* m_buffer;
  bool m_perflab;
  unsigned int m_deviceId;
  unsigned int m_platform;
  TestResult* m_result;
};

/////////////////////////////////////////////////////////////////////////////

#endif  // OCL_TEST_WORKER_H
