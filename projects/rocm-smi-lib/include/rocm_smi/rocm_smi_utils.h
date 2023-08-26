/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018-2023, Advanced Micro Devices, Inc.
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
#include <vector>
#include <sstream>
#include <iomanip>
#include <type_traits>

#include "rocm_smi/rocm_smi_device.h"

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

pthread_mutex_t *GetMutex(uint32_t dv_ind);
int SameFile(const std::string fileA, const std::string fileB);
bool FileExists(char const *filename);
std::vector<std::string> globFilesExist(const std::string& filePattern);
int isRegularFile(std::string fname, bool *is_reg);
int ReadSysfsStr(std::string path, std::string *retStr);
int WriteSysfsStr(std::string path, std::string val);
bool IsInteger(const std::string & n_str);
std::pair<bool, std::string> executeCommand(std::string command,
                                            bool stdOut = true);
rsmi_status_t storeTmpFile(uint32_t dv_ind, std::string parameterName,
                           std::string stateName, std::string storageData);
std::vector<std::string> getListOfAppTmpFiles();
bool containsString(std::string originalString, std::string substring);
std::tuple<bool, std::string> readTmpFile(
                                          uint32_t dv_ind,
                                          std::string stateName,
                                          std::string parameterName);
void displayAppTmpFilesContent(void);
std::string debugVectorContent(std::vector<std::string> v);
std::string displayAllDevicePaths(std::vector<std::shared_ptr<Device>> v);
rsmi_status_t handleException();
rsmi_status_t
GetDevValueVec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<std::string> *val_vec);
rsmi_status_t
GetDevBinaryBlob(amd::smi::DevInfoTypes type,
           uint32_t dv_ind, std::size_t b_size, void* p_binary_data);
rsmi_status_t ErrnoToRsmiStatus(int err);
std::string getRSMIStatusString(rsmi_status_t ret);
std::tuple<bool, std::string, std::string, std::string, std::string,
           std::string, std::string, std::string, std::string,
           std::string, std::string, std::string>
           getSystemDetails(void);
void logSystemDetails(void);
rsmi_status_t getBDFString(uint64_t bdf_id, std::string& bfd_str);
void logHexDump(const char *desc, const void *addr, const size_t len,
             size_t perLine);
bool isSystemBigEndian();
std::string getBuildType();
std::string getMyLibPath();
int subDirectoryCountInPath(const std::string path);
template <typename T>
std::string print_int_as_hex(T i, bool showHexNotation=true) {
  std::stringstream ss;
  if (showHexNotation) {
    ss << "0x" << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex;
  } else {
    ss << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex;
  }

  if (std::is_same<std::uint8_t, T>::value) {
    ss << static_cast<unsigned int>(i|0);
  } else if (std::is_same<std::int8_t, T>::value) {
    ss << static_cast<int>(static_cast<uint8_t>(i|0));
  } else if (std::is_signed<T>::value) {
    ss << static_cast<long long int>(i | 0);
  } else {
    ss << static_cast<unsigned long long int>(i | 0);
  }
  ss << std::dec;
  return ss.str();
};

template <typename T>
std::string print_unsigned_int(T i) {
  std::stringstream ss;
  ss << static_cast<unsigned long long int>(i | 0);
  return ss.str();
}

template <typename T>
std::string print_unsigned_hex_and_int(T i, std::string heading="") {
  std::stringstream ss;
  if (heading.empty() == false) {
    ss << "\n" << heading << " = ";
  }
  ss << "Hex (MSB): " << print_int_as_hex(i) << ", "
     << "Unsigned int: " << print_unsigned_int(i) << ", "
     << "Byte Size: " << sizeof(T);
  return ss.str();
}

struct pthread_wrap {
 public:
        explicit pthread_wrap(pthread_mutex_t &p_mut) : mutex_(p_mut) {}

        void Acquire() {pthread_mutex_lock(&mutex_);}
        int AcquireNB() {return pthread_mutex_trylock(&mutex_);}
        void Release() {pthread_mutex_unlock(&mutex_);}
 private:
        pthread_mutex_t& mutex_;
};
struct ScopedPthread {
     explicit ScopedPthread(pthread_wrap& mutex, bool blocking = true) : //NOLINT
                               pthrd_ref_(mutex), mutex_not_acquired_(false) {
       if (blocking) {
         pthrd_ref_.Acquire();
       } else {
         int ret = pthrd_ref_.AcquireNB();
         if (ret == EBUSY) {
           mutex_not_acquired_ = true;
         }
       }
     }

     ~ScopedPthread() {
       pthrd_ref_.Release();
     }

     bool mutex_not_acquired() {return mutex_not_acquired_;}

 private:
     ScopedPthread(const ScopedPthread&);
     pthread_wrap& pthrd_ref_;
     bool mutex_not_acquired_;  // Use for AcquireNB (not for Aquire())
};


#define PASTE2(x, y) x##y
#define PASTE(x, y) PASTE2(x, y)

#define __forceinline __inline__ __attribute__((always_inline))

template <typename lambda>
class ScopeGuard {
 public:
  explicit __forceinline ScopeGuard(const lambda& release)
      : release_(release), dismiss_(false) {}

  ScopeGuard(const ScopeGuard& rhs) {*this = rhs; }

  __forceinline ~ScopeGuard() {
    if (!dismiss_) release_();
  }
  __forceinline ScopeGuard& operator=(const ScopeGuard& rhs) {
    dismiss_ = rhs.dismiss_;
    release_ = rhs.release_;
    rhs.dismiss_ = true;
  }
  __forceinline void Dismiss() { dismiss_ = true; }

 private:
  lambda release_;
  bool dismiss_;
};

template <typename lambda>
static __forceinline ScopeGuard<lambda> MakeScopeGuard(lambda rel) {
  return ScopeGuard<lambda>(rel);
}

#define MAKE_SCOPE_GUARD_HELPER(lname, sname, ...) \
  auto lname = __VA_ARGS__;                        \
  amd::smi::ScopeGuard<decltype(lname)> sname(lname);
#define MAKE_SCOPE_GUARD(...)                                   \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), \
                          PASTE(scopeGuard, __COUNTER__), __VA_ARGS__)
#define MAKE_NAMED_SCOPE_GUARD(name, ...)                             \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), name, \
                          __VA_ARGS__)


// A macro to disallow the copy and move constructor and operator= functions
#define DISALLOW_COPY_AND_ASSIGN(TypeName)   \
  TypeName(const TypeName&) = delete;        \
  TypeName(TypeName&&) = delete;             \
  void operator=(const TypeName&) = delete;  \
  void operator=(TypeName&&) = delete;

template <class LockType>
class ScopedAcquire {
 public:
  /// @brief: When constructing, acquire the lock.
  /// @param: lock(Input), pointer to an existing lock.
  explicit ScopedAcquire(LockType* lock) : lock_(lock), doRelease(true) {
                                                            lock_->Acquire();}

  /// @brief: when destructing, release the lock.
  ~ScopedAcquire() {
    if (doRelease) lock_->Release();
  }

  /// @brief: Release the lock early.  Avoid using when possible.
  void Release() {
    lock_->Release();
    doRelease = false;
  }

 private:
  LockType* lock_;
  bool doRelease;
  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(ScopedAcquire);
};

}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_
