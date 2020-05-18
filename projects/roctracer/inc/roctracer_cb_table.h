/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef CB_TABLE_H_
#define CB_TABLE_H_

#include <ext/prof_protocol.h>

#include <mutex>

namespace roctracer {

// Generic callbacks table
template <int N>
class CbTable {
  public:
  typedef std::mutex mutex_t;

  CbTable() {
    std::lock_guard<mutex_t> lck(mutex_);
    for (int i = 0; i < N; i++) {
      callback_[i] = NULL;
      arg_[i] = NULL;
    }
  }

  bool set(uint32_t id, activity_rtapi_callback_t callback, void* arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    bool ret = false;
    if (id < N) {
      callback_[id] = callback;
      arg_[id] = arg;
      ret = true;
    }
    return ret;
  }

  bool get(uint32_t id, activity_rtapi_callback_t* callback, void** arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    bool ret = false;
    if (id < N) {
      *callback = callback_[id];
      *arg = arg_[id];
      ret = true;
    }
    return ret;
  }

  private:
  activity_rtapi_callback_t callback_[N];
  void* arg_[N];
  mutex_t mutex_;
};

}  // namespace roctracer

#endif  // CB_TALE_H_
