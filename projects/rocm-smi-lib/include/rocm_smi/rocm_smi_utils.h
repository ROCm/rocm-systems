/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_

#include <pthread.h>

#include <string>
#include <cstdint>

#ifdef NDEBUG
#define debug_print(fmt, ...)               \
  do {                                      \
  } while (false)
#else
#define debug_print(fmt, ...)               \
  do {                                      \
    fprintf(stderr, fmt, ##__VA_ARGS__);    \
  } while (false)
#endif

namespace amd {
namespace smi {

int SameFile(const std::string fileA, const std::string fileB);
bool FileExists(char const *filename);
int isRegularFile(std::string fname, bool *is_reg);

int ReadSysfsStr(std::string path, std::string *retStr);
int WriteSysfsStr(std::string path, std::string val);

struct pthread_wrap {
 public:
        explicit pthread_wrap(pthread_mutex_t &p_mut) : mutex_(p_mut) {}

        void Acquire() { pthread_mutex_lock(&mutex_);   }
        void Release() { pthread_mutex_unlock(&mutex_); }
 private:
        pthread_mutex_t& mutex_;
};
struct ScopedPthread {
     explicit ScopedPthread(pthread_wrap& mutex) : pthrd_ref_(mutex) {
       pthrd_ref_.Acquire();
     }

     ~ScopedPthread() {
       pthrd_ref_.Release();
     }
 private:
     ScopedPthread(const ScopedPthread&);

     pthread_wrap& pthrd_ref_;
};
}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_
