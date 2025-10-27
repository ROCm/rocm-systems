////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/util/os.h"

#include <algorithm>
#include <process.h>
#include <string>
#include <windows.h>
#include <ntstatus.h>
#include <psapi.h>

#include <emmintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <shared_mutex>

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

bool CloseLib(LibHandle lib) { return FreeLibrary(*(::HMODULE*)&lib); }

std::vector<LibHandle> GetLoadedLibs() {
  // Use EnumProcessModulesEx
  assert(!"Not implemented.");
  return std::vector<LibHandle>{};
}

std::string GetLibraryName(LibHandle lib) {
  assert(!"Not implemented.");
  return std::string{};
}

Semaphore CreateSemaphore() {
  auto sem = static_cast<void*>(CreateSemaphoreA(nullptr, 0, LONG_MAX, nullptr));
  assert(sem != nullptr && "CreateSemaphore failed");
  return *(Semaphore*)&sem;
}

bool WaitSemaphore(Semaphore sem) {
  return WaitForSingleObject(sem, INFINITE) == WAIT_OBJECT_0;
}

void PostSemaphore(Semaphore sem) {
  ReleaseSemaphore(sem, 1, nullptr);
}

void DestroySemaphore(Semaphore sem) {
  if (!CloseHandle(sem)) {
    assert("CloseHandle() failed");
  }
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
  return reinterpret_cast<SharedMutex>(new std::shared_mutex());
}

bool TryAcquireSharedMutex(SharedMutex lock) {
  return reinterpret_cast<std::shared_mutex*>(lock)->try_lock();
}

bool AcquireSharedMutex(SharedMutex lock) {
  reinterpret_cast<std::shared_mutex*>(lock)->lock();
  return true;
}

void ReleaseSharedMutex(SharedMutex lock) {
  reinterpret_cast<std::shared_mutex*>(lock)->unlock();
}

bool TrySharedAcquireSharedMutex(SharedMutex lock) {
  return reinterpret_cast<std::shared_mutex*>(lock)->try_lock_shared();
}

bool SharedAcquireSharedMutex(SharedMutex lock) {
  reinterpret_cast<std::shared_mutex*>(lock)->lock_shared();
  return true;
}

void SharedReleaseSharedMutex(SharedMutex lock) {
  reinterpret_cast<std::shared_mutex*>(lock)->unlock_shared();
}

void DestroySharedMutex(SharedMutex lock) {
  delete reinterpret_cast<std::shared_mutex*>(lock);
}

uint64_t ReadSystemClock() {
  assert(false && "Not implemented.");
  abort();
  return 0;
}

uint64_t SystemClockFrequency() {
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  return frequency.QuadPart;
}

bool ParseCpuID(cpuid_t* cpuinfo) {
  int regs[4] = {};
  int info{};

  __cpuid(regs, info);
  memset(cpuinfo->ManufacturerID, 0, sizeof(cpuinfo->ManufacturerID));
  *reinterpret_cast<int*>(cpuinfo->ManufacturerID) = regs[1];
  *reinterpret_cast<int*>(cpuinfo->ManufacturerID + 4) = regs[3];
  *reinterpret_cast<int*>(cpuinfo->ManufacturerID + 8) = regs[2];
  // @todo fill the rest of CPU info
  return true;
}

bool IsEnvVarSet(std::string env_var_name) {
  char* buff = NULL;
  buff = getenv(env_var_name.c_str());
  return (buff != NULL);
}

std::vector<LibHandle> GetLoadedToolsLib() {
  std::vector<LibHandle> ret;
  std::vector<std::string> names;
  HMODULE hMods[1024];
  HANDLE hProcess = GetCurrentProcess();
  DWORD cbNeeded;
  unsigned int i;

  if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
    for (i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
      TCHAR szModName[MAX_PATH];

      // Get the full path to the module's file.

      if (GetModuleFileNameEx(hProcess, hMods[i], szModName, sizeof(szModName) / sizeof(TCHAR))) {
        // Print the module name and handle value.
        names.push_back(szModName);
      }
    }
  }

  if (!names.empty()) {
    for (auto& name : names) ret.push_back(LoadLib(name));
  }

  return ret;
}

int GetProcessId() { return ::_getpid(); }

uint64_t TimeNanos() {
  static double PerformanceFrequency = 0.f;
  if (PerformanceFrequency == 0) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    PerformanceFrequency = (double)frequency.QuadPart;
  }
  LARGE_INTEGER current;
  QueryPerformanceCounter(&current);
  return (uint64_t)((double)current.QuadPart / PerformanceFrequency * 1e9);
}

static inline int memProtToOsProt(MemProt prot) {
  switch (prot) {
    case MEM_PROT_NONE:
      return PAGE_NOACCESS;
    case MEM_PROT_READ:
      return PAGE_READONLY;
    case MEM_PROT_RW:
      return PAGE_READWRITE;
    case MEM_PROT_RWX:
      return PAGE_EXECUTE_READWRITE;
    default:
      break;
  }
  return -1;
}

static size_t g_page_size_ = 0;   //!< The default os page size
static int processorCount_;       //!< The number of active processors
static size_t allocationGranularity_;

//! Return the default os page size.
size_t PageSize() {
  if (g_page_size_ == 0) {
    SYSTEM_INFO si{};
    ::GetSystemInfo(&si);
    g_page_size_ = si.dwPageSize;
  }
  return g_page_size_;
}

void* ReserveMemory(void* start, size_t size, size_t alignment, MemProt prot) {
  size = AlignUp(size, PageSize());
  if (allocationGranularity_ == 0) {
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    g_page_size_ = si.dwPageSize;
    allocationGranularity_ = (size_t)si.dwAllocationGranularity;
  }
  alignment = std::max(allocationGranularity_, AlignUp(alignment, allocationGranularity_));
  assert(IsPowerOfTwo(alignment) && "not a power of 2");

  size_t requested = size + alignment - allocationGranularity_;
  address mem, aligned;
  do {
    mem = reinterpret_cast<address>(VirtualAlloc(start, requested, MEM_RESERVE, memProtToOsProt(prot)));

    // check for out of memory.
    if (mem == NULL) return NULL;

    aligned = AlignUp(mem, alignment);

    // check for already aligned memory.
    if (aligned == mem && size == requested) {
      return mem;
    }

    // try to reserve the aligned address.
    if (VirtualFree(mem, 0, MEM_RELEASE) == 0) {
      assert(!"VirtualFree failed");
    }

    mem = (address)VirtualAlloc(aligned, size, MEM_RESERVE, memProtToOsProt(prot));
    assert((mem == NULL || mem == aligned) && "VirtualAlloc failed");

  } while (mem != aligned);

  return mem;
}
bool ReleaseMemory(void* addr, size_t size) { return VirtualFree(addr, 0, MEM_RELEASE) != 0; }

bool CommitMemory(void* addr, size_t size, MemProt prot) {
  return VirtualAlloc(addr, size, MEM_COMMIT, memProtToOsProt(prot)) != NULL;
}

bool UncommitMemory(void* addr, size_t size) { return VirtualFree(addr, size, MEM_DECOMMIT) != 0; }

uint64_t HostTotalPhysicalMemory() {
  static uint64_t totalPhys = 0;

  if (totalPhys != 0) {
    return totalPhys;
  }

  MEMORYSTATUSEX mstatus;
  mstatus.dwLength = sizeof(mstatus);

  ::GlobalMemoryStatusEx(&mstatus);

  totalPhys = mstatus.ullTotalPhys;
  return totalPhys;
}

int Ffs(int i) {
  int res = 0;
  unsigned long index;
  if (_BitScanForward(&index, i) != 0) {
    res = index + 1;
  }
  return res;
}

int Ctz(uint64_t i) {
  unsigned long index;
  if (_BitScanReverse64(&index, i)) {
    return sizeof(i) * 8 - 1 - index;
  } else {
    return sizeof(i) * 8;
  }
}

char* DlError() { return nullptr; }
}   //  namespace os
}   //  namespace rocr

#endif
