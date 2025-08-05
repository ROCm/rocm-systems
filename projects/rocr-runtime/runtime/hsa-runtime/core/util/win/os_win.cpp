////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32  // Are we compiling for windows?
#define NOMINMAX

#include "core/util/os.h"

#include <algorithm>
#include <process.h>
#include <string>
#include <windows.h>

#include <emmintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#undef Yield
#undef CreateMutex

namespace rocr {
namespace os {

static_assert(sizeof(LibHandle) == sizeof(HMODULE),
              "OS abstraction size mismatch");
static_assert(sizeof(LibHandle) == sizeof(::HANDLE),
              "OS abstraction size mismatch");
static_assert(sizeof(Semaphore) == sizeof(::HANDLE),
              "OS abstraction size mismatch");
static_assert(sizeof(Mutex) == sizeof(::HANDLE),
              "OS abstraction size mismatch");
static_assert(sizeof(Thread) == sizeof(::HANDLE),
              "OS abstraction size mismatch");
static_assert(sizeof(EventHandle) == sizeof(::HANDLE),
              "OS abstraction size mismatch");

LibHandle LoadLib(std::string filename) {
  HMODULE ret = LoadLibrary(filename.c_str());
  return *(LibHandle*)&ret;
}

void* GetExportAddress(LibHandle lib, std::string export_name) {
  return GetProcAddress(*(HMODULE*)&lib, export_name.c_str());
}

void CloseLib(LibHandle lib) { FreeLibrary(*(::HMODULE*)&lib); }

std::vector<LibHandle> GetLoadedLibs() {
  // Use EnumProcessModulesEx
  static_assert(false, "Not implemented.");
}

std::string GetLibraryName(LibHandle lib) {
  static_assert(false, "Not implemented.");
}

Semaphore CreateSemaphore() {
  sem = static_cast<void*>(CreateSemaphore(NULL, 0, LONG_MAX, NULL));
  assert(sem != NULL && "CreateSemaphore failed");

  return *(Semaphore*)&sem;
}

bool WaitSemaphore(Semaphore sem) {
  return WaitForSingleObject(*(::HANDLE*)&lock, INFINITE) == WAIT_OBJECT_0;
}

void PostSemaphore(Semaphore sem) {
  ReleaseSemaphore(static_cast<HANDLE>(*sem), 1, NULL);
}

void DestroySemaphore(Semaphore sem) {
  if (!CloseHandle(static_cast<HANDLE>(*sem))) {
    assert("CloseHandle() failed");
  }
  *sem = NULL;
}

Mutex CreateMutex() { return CreateEvent(NULL, false, true, NULL); }

bool TryAcquireMutex(Mutex lock) {
  return WaitForSingleObject(*(::HANDLE*)&lock, 0) == WAIT_OBJECT_0;
}

bool AcquireMutex(Mutex lock) {
  return WaitForSingleObject(*(::HANDLE*)&lock, INFINITE) == WAIT_OBJECT_0;
}

void ReleaseMutex(Mutex lock) { SetEvent(*(::HANDLE*)&lock); }

void DestroyMutex(Mutex lock) { CloseHandle(*(::HANDLE*)&lock); }

void Sleep(int delay_in_millisecond) { ::Sleep(delay_in_millisecond); }

void uSleep(int delayInUs) { ::Sleep(delayInUs / 1000); }

void YieldThread() { ::Sleep(0); }

struct ThreadArgs {
  void* entry_args;
  ThreadEntry entry_function;
};

unsigned __stdcall ThreadTrampoline(void* arg) {
  ThreadArgs* thread_args = (ThreadArgs*)arg;
  ThreadEntry entry = thread_args->entry_function;
  void* data = thread_args->entry_args;
  delete thread_args;
  entry(data);
  _endthreadex(0);
  return 0;
}

Thread CreateThread(ThreadEntry entry_function, void* entry_argument,
                    uint stack_size, int priority_unused) {
  ThreadArgs* thread_args = new ThreadArgs();
  thread_args->entry_args = entry_argument;
  thread_args->entry_function = entry_function;
  uintptr_t ret =
      _beginthreadex(NULL, stack_size, ThreadTrampoline, thread_args, 0, NULL);
  return *(Thread*)&ret;
}

void CloseThread(Thread thread) { CloseHandle(*(::HANDLE*)&thread); }

bool WaitForThread(Thread thread) {
  return WaitForSingleObject(*(::HANDLE*)&thread, INFINITE) == WAIT_OBJECT_0;
}

bool WaitForAllThreads(Thread* threads, uint thread_count) {
  return WaitForMultipleObjects(thread_count, threads, TRUE, INFINITE) ==
         WAIT_OBJECT_0;
}

void SetEnvVar(std::string env_var_name, std::string env_var_value) {
  SetEnvironmentVariable(env_var_name.c_str(), env_var_value.c_str());
}

std::string GetEnvVar(std::string env_var_name) {
  char* buff;
  DWORD char_count = GetEnvironmentVariable(env_var_name.c_str(), NULL, 0);
  if (char_count == 0) return "";
  buff = (char*)alloca(sizeof(char) * char_count);
  GetEnvironmentVariable(env_var_name.c_str(), buff, char_count);
  buff[char_count - 1] = '\0';
  std::string ret = buff;
  return ret;
}

size_t GetUserModeVirtualMemorySize() {
  SYSTEM_INFO system_info = {0};
  GetSystemInfo(&system_info);
  return ((size_t)system_info.lpMaximumApplicationAddress + 1);
}

size_t GetUsablePhysicalHostMemorySize() {
  MEMORYSTATUSEX memory_status = {0};
  memory_status.dwLength = sizeof(memory_status);
  if (GlobalMemoryStatusEx(&memory_status) == 0) {
    return 0;
  }

  const size_t physical_size = static_cast<size_t>(memory_status.ullTotalPhys);
  return std::min(GetUserModeVirtualMemorySize(), physical_size);
}

uintptr_t GetUserModeVirtualMemoryBase() { return (uintptr_t)0; }

// Os event wrappers
EventHandle CreateOsEvent(bool auto_reset, bool init_state) {
  EventHandle evt = reinterpret_cast<EventHandle>(
      CreateEvent(NULL, (BOOL)(!auto_reset), (BOOL)init_state, NULL));
  return evt;
}

int DestroyOsEvent(EventHandle event) {
  if (event == NULL) {
    return -1;
  }
  return CloseHandle(reinterpret_cast<::HANDLE>(event));
}

int WaitForOsEvent(EventHandle event, unsigned int milli_seconds) {
  if (event == NULL) {
    return -1;
  }

  int ret_code =
      WaitForSingleObject(reinterpret_cast<::HANDLE>(event), milli_seconds);
  if (ret_code == WAIT_TIMEOUT) {
    ret_code = 0x14003;  // 0x14003 indicates timeout
  }
  return ret_code;
}

int SetOsEvent(EventHandle event) {
  if (event == NULL) {
    return -1;
  }
  return SetEvent(reinterpret_cast<::HANDLE>(event));
}

int ResetOsEvent(EventHandle event) {
  if (event == NULL) {
    return -1;
  }
  return ResetEvent(reinterpret_cast<::HANDLE>(event));
}

uint64_t ReadAccurateClock() {
  uint64_t ret;
  QueryPerformanceCounter((LARGE_INTEGER*)&ret);
  return ret;
}

uint64_t AccurateClockFrequency() {
  uint64_t ret;
  QueryPerformanceFrequency((LARGE_INTEGER*)&ret);
  return ret;
}

SharedMutex CreateSharedMutex() {
  assert(false && "Not implemented.");
  abort();
  return nullptr;
}

bool TryAcquireSharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
  return false;
}

bool AcquireSharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
  return false;
}

void ReleaseSharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
}

bool TrySharedAcquireSharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
  return false;
}

bool SharedAcquireSharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
  return false;
}

void SharedReleaseSharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
}

void DestroySharedMutex(SharedMutex lock) {
  assert(false && "Not implemented.");
  abort();
}

uint64_t ReadSystemClock() {
  assert(false && "Not implemented.");
  abort();
  return 0;
}

uint64_t SystemClockFrequency() {
  assert(false && "Not implemented.");
  abort();
  return 0;
}

bool ParseCpuID(cpuid_t* cpuinfo) {
  assert(false && "Not implemented.");
  abort();
  return false;
}

}   //  namespace os
}   //  namespace rocr

#endif
