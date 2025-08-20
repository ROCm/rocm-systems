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

#include "OCLPerfDoubleDMA.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>

#include <cmath>
#include <sstream>
#include <string>

#include "CL/cl.h"
#include "CL/cl_ext.h"

const size_t blockX = 256;
const size_t blockY = 256;
const size_t blockZ = 512;

const size_t chunk = 16;
const size_t size_S = blockX * blockY * blockZ * sizeof(cl_float4);
const size_t size_s = blockX * blockY * chunk * sizeof(cl_float4);
static const int WindowWidth = 80;

const size_t MaxQueues = 3;
bool profEnable = false;

static const char* strKernel =
    "__kernel void dummy(__global float4* out)  \n"
    "{                                          \n"
    "   uint id = get_global_id(0);             \n"
    "   float4 value = (float4)(1.0f, 2.0f, 3.0f, 4.0f);  \n"
    "   uint factorial = 1;                     \n"
    "   for (uint i = 1; i < (id / 0x400); ++i)\n"
    "   {                                       \n"
    "       factorial *= i;                     \n"
    "   }                                       \n"
    "   out[id] = value * factorial;            \n"
    "}                                          \n";

class ProfileQueue {
 public:
  enum Operation { Write = 0, Execute, Read, Total };

  static const char* OperationName[Total];
  static const char StartCommand[Total];
  static const char ExecCommand[Total];

  ProfileQueue() {}
  ~ProfileQueue() {
    for (size_t op = 0; op < Total; ++op) {
      for (size_t idx = 0; idx < events_[op].size(); ++idx) {
        clReleaseEvent(events_[op][idx]);
      }
    }
  }

  void addEvent(Operation op, cl_event event) { events_[op].push_back(event); }

  void findMinMax(cl_long* min, cl_long* max) {
    // Find time min/max ranges for the frame scaling
    for (size_t op = 0; (op < ProfileQueue::Total); ++op) {
      cl_long time;
      if (events_[op].size() == 0) continue;
      clGetEventProfilingInfo(events_[op][0], CL_PROFILING_COMMAND_START, sizeof(cl_long), &time,
                              NULL);
      if (0 == *min) {
        *min = time;
      } else {
        *min = std::min(*min, time);
      }
      clGetEventProfilingInfo(events_[op][events_[op].size() - 1], CL_PROFILING_COMMAND_END,
                              sizeof(cl_long), &time, NULL);
      if (0 == *max) {
        *max = time;
      } else {
        *max = std::max(*max, time);
      }
    }
  }

  void display(cl_long start, cl_long finish) {
    std::string graph;
    graph.resize(WindowWidth + 1);
    graph[WindowWidth] = '\x0';
    cl_long timeFrame = finish - start;
    cl_long interval = timeFrame / WindowWidth;

    // Find time min/max ranges for the frame scaling
    for (size_t op = 0; (op < Total); ++op) {
      if (events_[op].size() == 0) continue;
      cl_long timeStart, timeEnd;
      int begin = 0, end = 0;
      for (size_t idx = 0; idx < events_[op].size(); ++idx) {
        bool cutStart = false;
        clGetEventProfilingInfo(events_[op][idx], CL_PROFILING_COMMAND_START, sizeof(cl_long),
                                &timeStart, NULL);
        clGetEventProfilingInfo(events_[op][idx], CL_PROFILING_COMMAND_END, sizeof(cl_long),
                                &timeEnd, NULL);

        // Continue if out of the frame scope
        if (timeStart >= finish) continue;
        if (timeEnd <= start) continue;

        if (timeStart <= start) {
          timeStart = start;
          cutStart = true;
        }

        if (timeEnd >= finish) {
          timeEnd = finish;
        }

        // Readjust time to the frame
        timeStart -= start;
        timeEnd -= start;
        timeStart = static_cast<cl_long>(floor(static_cast<float>(timeStart) / interval + 0.5f));
        timeEnd = static_cast<cl_long>(floor(static_cast<float>(timeEnd) / interval + 0.5f));
        begin = static_cast<int>(timeStart);
        // Idle from end to begin
        for (int c = end; c < begin; ++c) {
          graph[c] = '-';
        }
        end = static_cast<int>(timeEnd);
        for (int c = begin; c < end; ++c) {
          if ((c == begin) && !cutStart) {
            graph[c] = StartCommand[op];
          } else {
            graph[c] = ExecCommand[op];
          }
        }
        if ((begin == end) && (end < WindowWidth)) {
          graph[begin] = '+';
        }
      }
      if (end < WindowWidth) {
        for (int c = end; c < WindowWidth; ++c) {
          graph[c] = '-';
        }
      }
      printf("%s\n", graph.c_str());
    }
  }

 private:
  // Profiling events
  std::vector<cl_event> events_[Total];
};

const char* ProfileQueue::OperationName[Total] = {"BufferWrite", "KernelExecution", "BufferRead"};
const char ProfileQueue::StartCommand[Total] = {'W', 'X', 'R'};
const char ProfileQueue::ExecCommand[Total] = {'>', '#', '<'};

class Profile {
 public:
  Profile(bool profEna, int numQueues)
      : profileEna_(profEna), numQueues_(numQueues), min_(0), max_(0), execTime_(0) {}

  ~Profile() {}

  void addEvent(int queue, ProfileQueue::Operation op, cl_event event) {
    if (profileEna_) {
      profQueue[queue].addEvent(op, event);
    }
  }

  cl_long findExecTime() {
    if (execTime_ != 0) return execTime_;
    for (int q = 0; q < numQueues_; ++q) {
      profQueue[q].findMinMax(&min_, &max_);
    }
    execTime_ = max_ - min_;
    return execTime_;
  }

  void display(cl_long start, cl_long finish) {
    if (!profileEna_) return;
    printf("\n ----------- Time frame %.3f (us), scale 1:%.0f\n", (float)(finish - start) / 1000,
           (float)(finish - start) / (1000 * WindowWidth));
    for (size_t op = 0; (op < ProfileQueue::Total); ++op) {
      printf("%s - %c%c; ", ProfileQueue::OperationName[op], ProfileQueue::StartCommand[op],
             ProfileQueue::ExecCommand[op]);
    }
    printf("\n");
    for (int q = 0; q < numQueues_; ++q) {
      printf("CommandQueue #%d\n", q);
      profQueue[q].display(min_ + start, min_ + finish);
    }
  }

 private:
  bool profileEna_;
  int numQueues_;     //!< Total number of queues
  cl_long min_;       //!< Min HW timestamp
  cl_long max_;       //!< Max HW timestamp
  cl_long execTime_;  //!< Profile time
  ProfileQueue profQueue[MaxQueues];
};

OCLPerfDoubleDMA::OCLPerfDoubleDMA() {
  _numSubTests = 2 * MaxQueues * 2;
  failed_ = false;
}

OCLPerfDoubleDMA::~OCLPerfDoubleDMA() {}

void OCLPerfDoubleDMA::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  _deviceId = deviceId;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  test_ = test;
  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");
  kernel_ = _wrapper->clCreateKernel(program_, "dummy", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  size_t bufSize = size_s;
  cl_mem buffer;
  if (test_ >= (2 * MaxQueues)) {
    profEnable = true;
  }
  test_ %= 2 * MaxQueues;
  size_t numBufs = (test_ % MaxQueues) + 1;
  for (size_t b = 0; b < numBufs; ++b) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, bufSize, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size_S,
                                    NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfDoubleDMA::run(void) {
  if (failed_) {
    return;
  }
  CPerfCounter timer;
  const int numQueues = (test_ % MaxQueues) + 1;
  const bool useKernel = ((test_ / MaxQueues) > 0);
  const int numBufs = numQueues;
  Profile profile(profEnable, numQueues);

  std::vector<cl_command_queue> cmdQueues(numQueues);
  int q;
  cl_command_queue_properties qProp = (profEnable) ? CL_QUEUE_PROFILING_ENABLE : 0;
  for (q = 0; q < numQueues; ++q) {
    cl_command_queue cmdQueue =
        _wrapper->clCreateCommandQueue(context_, devices_[_deviceId], qProp, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed");
    cmdQueues[q] = cmdQueue;
  }

  float* Data_s = (float*)_wrapper->clEnqueueMapBuffer(cmdQueues[0], buffers_[numBufs], CL_TRUE,
                                                       CL_MAP_READ | CL_MAP_WRITE, 0, size_S, 0,
                                                       NULL, NULL, &error_);

  size_t gws[1] = {size_s / (4 * sizeof(float))};
  size_t lws[1] = {256};

  // Warm-up
  for (q = 0; q < numQueues; ++q) {
    error_ |= _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[q], CL_FALSE, 0, size_s,
                                             (char*)Data_s, 0, NULL, NULL);
    error_ |= _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[q]);
    error_ |=
        _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 0, NULL, NULL);
    error_ |= _wrapper->clEnqueueReadBuffer(cmdQueues[q], buffers_[q], CL_FALSE, 0, size_s,
                                            (char*)Data_s, 0, NULL, NULL);
    error_ |= _wrapper->clFinish(cmdQueues[q]);
  }

  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "Execution failed");

  size_t s_done = 0;
  cl_event r[MaxQueues] = {0}, w[MaxQueues] = {0}, x[MaxQueues] = {0};

  /*----------  pass2:  copy Data_s to and from GPU Buffers ----------*/
  s_done = 0;
  timer.Reset();
  timer.Start();
  int idx = numBufs - 1;
  // Start from the last so read/write won't go to the same DMA when kernel is
  // executed
  q = numQueues - 1;
  size_t iter = 0;
  while (1) {
    if (0 == r[idx]) {
      error_ |= _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                               (char*)Data_s + s_done, 0, NULL, &w[idx]);
    } else {
      error_ |= _wrapper->clEnqueueWriteBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                               (char*)Data_s + s_done, 1, &r[idx], &w[idx]);
      if (!profEnable) {
        error_ |= _wrapper->clReleaseEvent(r[idx]);
      }
    }
    _wrapper->clFlush(cmdQueues[q]);
    profile.addEvent(q, ProfileQueue::Write, w[idx]);

    if (useKernel) {
      // Change the queue
      ++q %= numQueues;
      // Implicit flush of DMA engine on kernel start, because memory dependency
      error_ |= _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[idx]);
      error_ |= _wrapper->clEnqueueNDRangeKernel(cmdQueues[q], kernel_, 1, NULL, gws, lws, 1,
                                                 &w[idx], &x[idx]);
      if (!profEnable) {
        error_ |= _wrapper->clReleaseEvent(w[idx]);
      }
      profile.addEvent(q, ProfileQueue::Execute, x[idx]);
    }
    _wrapper->clFlush(cmdQueues[q]);

    // Change the queue
    ++q %= numQueues;
    error_ |= _wrapper->clEnqueueReadBuffer(cmdQueues[q], buffers_[idx], CL_FALSE, 0, size_s,
                                            (char*)Data_s + s_done, 1,
                                            (useKernel) ? &x[idx] : &w[idx], &r[idx]);
    if (!profEnable) {
      error_ |= _wrapper->clReleaseEvent((useKernel) ? x[idx] : w[idx]);
    }
    profile.addEvent(q, ProfileQueue::Read, r[idx]);
    _wrapper->clFlush(cmdQueues[q]);

    if ((s_done += size_s) >= size_S) {
      if (!profEnable) {
        error_ |= _wrapper->clReleaseEvent(r[idx]);
      }
      break;
    }
    ++iter;
    ++idx %= numBufs;
    ++q %= numQueues;
  }

  for (q = 0; q < numQueues; ++q) {
    error_ |= _wrapper->clFinish(cmdQueues[q]);
  }
  timer.Stop();

  error_ =
      _wrapper->clEnqueueUnmapMemObject(cmdQueues[0], buffers_[numBufs], Data_s, 0, NULL, NULL);

  error_ |= _wrapper->clFinish(cmdQueues[0]);
  CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "Execution failed");

  cl_long gpuTimeFrame = profile.findExecTime();
  cl_long oneIter = gpuTimeFrame / iter;

  // Display 4 iterations in the middle
  cl_long startFrame = oneIter * (iter / 2 - 2);
  cl_long finishFrame = oneIter * (iter / 2 + 2);
  profile.display(startFrame, finishFrame);

  for (q = 0; q < numQueues; ++q) {
    error_ = _wrapper->clReleaseCommandQueue(cmdQueues[q]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
  }

  double GBytes = (double)(2 * size_S) / (double)(1000 * 1000 * 1000);
  _perfInfo = static_cast<float>(GBytes / timer.GetElapsedTime());

  std::stringstream stream;
  if (useKernel) {
    stream << "Write/Kernel/Read operation ";
  } else {
    stream << "Write/Read operation ";
  }
  stream << numQueues << " queues; profiling " << ((profEnable) ? "enabled" : "disabled")
         << " [GB/s]";

  stream.flags(std::ios::right | std::ios::showbase);
  testDescString = stream.str();
}

unsigned int OCLPerfDoubleDMA::close(void) { return OCLTestImp::close(); }
