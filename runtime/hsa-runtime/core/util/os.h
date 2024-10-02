////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
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

// Minimal operating system abstraction interfaces.

#ifndef HSA_RUNTIME_CORE_UTIL_OS_H_
#define HSA_RUNTIME_CORE_UTIL_OS_H_

#include <string>
#include <vector>
#include "utils.h"

namespace rocr {
namespace os {
typedef void* LibHandle;
typedef void* Semaphore;
typedef void* Mutex;
typedef void* SharedMutex;
typedef void* Thread;
typedef void* EventHandle;

typedef enum {
  OS_THREAD_PRIORITY_DEFAULT    = -1,
  OS_THREAD_PRIORITY_HIGH       = 254,
  OS_THREAD_PRIORITY_MAX        = 255,
} ThreadPriority;

enum class os_t { OS_WIN = 0, OS_LINUX, COUNT };
static __forceinline std::underlying_type<os_t>::type os_index(os_t val) {
  return std::underlying_type<os_t>::type(val);
}

#ifdef _WIN32
static const os_t current_os = os_t::OS_WIN;
#elif __linux__
static const os_t current_os = os_t::OS_LINUX;
#else
static_assert(false, "Operating System not detected!");
#endif

/// @brief: Loads dynamic library based on file name. Return value will be NULL
/// if failed.
/// @param: filename(Input), file name of the library.
/// @return: LibHandle.
LibHandle LoadLib(std::string filename);

/// @brief: Gets the address of exported symbol. Return NULl if failed.
/// @param: lib(Input), library handle which exporting from.
/// @param: export_name(Input), the name of the exported symbol.
/// @return: void*.
void* GetExportAddress(LibHandle lib, std::string export_name);

/// @brief: Unloads the dynamic library.
/// @param: lib(Input), library handle which will be unloaded.
void CloseLib(LibHandle lib);

/// @brief: Lists loaded tool libraries that contain
/// symbol HSA_AMD_TOOL_PRIORITY
/// @return: List of library handles
std::vector<LibHandle> GetLoadedToolsLib();

/// @brief: Returns the library's path name.
/// @param: lib(Input), libray handle
/// @return: Path name of library
std::string GetLibraryName(LibHandle lib);

/// @brief: Creates a Semaphore, will return NULL if failed.
/// @param: void.
/// @return: Semaphore.
Semaphore CreateSemaphore();

/// @brief: Waits for the semaphore. This is a blocking wait.
/// If the Semaphore is signalled, this function will return.
/// @param: sem(Input), handle to the semaphore.
/// @return: void.
bool WaitSemaphore(Semaphore sem);

/// @brief: Post/Signal/Wake-up the semaphore
/// @param: sem(Input), handle to the semaphore.
/// @return: void.
void PostSemaphore(Semaphore sem);

/// @brief: Destroys the semaphore.
/// @param: sem(Input), handle to the semaphore.
/// @return: void.
void DestroySemaphore(Semaphore sem);

/// @brief: Creates a mutex, will return NULL if failed.
/// @param: void.
/// @return: Mutex.
Mutex CreateMutex();

/// @brief: Tries to acquire the mutex once, if successed, return true.
/// @param: lock(Input), handle to the mutex.
/// @return: bool.
bool TryAcquireMutex(Mutex lock);

/// @brief: Aquires the mutex, if the mutex is locked, it will wait until it is
/// released. If the mutex is acquired successfully, it will return true.
/// @param: lock(Input), handle to the mutex.
/// @return: bool.
bool AcquireMutex(Mutex lock);

/// @brief: Releases the mutex.
/// @param: lock(Input), handle to the mutex.
/// @return: void.
void ReleaseMutex(Mutex lock);

/// @brief: Destroys the mutex.
/// @param: lock(Input), handle to the mutex.
/// @return: void.
void DestroyMutex(Mutex lock);

/// @brief: Creates a shared mutex, will return NULL if failed.
/// @param: void.
/// @return: SharedMutex.
SharedMutex CreateSharedMutex();

/// @brief: Tries to acquire the mutex in exclusive mode once, if successed, return true.
/// @param: lock(Input), handle to the shared mutex.
/// @return: bool.
bool TryAcquireSharedMutex(SharedMutex lock);

/// @brief: Aquires the mutex in exclusive mode, if the mutex is locked, it will wait until it is
/// released. If the mutex is acquired successfully, it will return true.
/// @param: lock(Input), handle to the mutex.
/// @return: bool.
bool AcquireSharedMutex(SharedMutex lock);

/// @brief: Releases the mutex from exclusive mode.
/// @param: lock(Input), handle to the mutex.
/// @return: void.
void ReleaseSharedMutex(SharedMutex lock);

/// @brief: Tries to acquire the mutex in shared mode once, if successed, return true.
/// @param: lock(Input), handle to the mutex.
/// @return: bool.
bool TrySharedAcquireSharedMutex(SharedMutex lock);

/// @brief: Aquires the mutex in shared mode, if the mutex in exclusive mode, it will wait until it
/// is released. If the mutex is acquired successfully, it will return true.
/// @param: lock(Input), handle to the mutex.
/// @return: bool.
bool SharedAcquireSharedMutex(SharedMutex lock);

/// @brief: Releases the mutex from shared mode.
/// @param: lock(Input), handle to the mutex.
/// @return: void.
void SharedReleaseSharedMutex(SharedMutex lock);

/// @brief: Destroys the mutex.
/// @param: lock(Input), handle to the mutex.
/// @return: void.
void DestroySharedMutex(SharedMutex lock);

/// @brief: Puts current thread to sleep.
/// @param: delayInMs(Input), time in millisecond for sleeping.
/// @return: void.
void Sleep(int delayInMs);

/// @brief: Puts current thread to sleep.
/// @param: delayInMs(Input), time in millisecond for sleeping.
/// @return: void.
void uSleep(int delayInUs);

/// @brief: Yields current thread.
/// @param: void.
/// @return: void.
void YieldThread();

typedef void (*ThreadEntry)(void*);

/// @brief: Creates a thread will return NULL if failed.
/// @param: entry_function(Input), a pointer to the function which the thread
/// starts from.
/// @param: entry_argument(Input), a pointer to the argument of the thread
/// function.
/// @param: stack_size(Input), size of the thread's stack, 0 by default.
/// @param: priority(Input), thread priority.
/// @return: Thread, a handle to thread created.
Thread CreateThread(ThreadEntry entry_function, void* entry_argument,
                    uint stack_size = 0, int priority = OS_THREAD_PRIORITY_DEFAULT);

/// @brief: Destroys the thread.
/// @param: thread(Input), thread handle to what will be destroyed.
/// @return: void.
void CloseThread(Thread thread);

/// @brief: Waits for specific thread to finish, if successful, return true.
/// @param: thread(Input), handle to waiting thread.
/// @return: bool.
bool WaitForThread(Thread thread);

/// @brief: Waits for multiple threads to finish, if successful, return true.
/// @param; threads(Input), a pointer to a list of thread handle.
/// @param: thread_count(Input), number of threads to be waited on.
/// @return: bool.
bool WaitForAllThreads(Thread* threads, uint thread_count);

/// @brief: Determines if environment key is set.
/// @param: env_var_name(Input), name of the environment value.
/// @return: bool, true for binding any value to environment key,
/// including an empty string. False otherwise
bool IsEnvVarSet(std::string env_var_name);

/// @brief: Sets the environment value.
/// @param: env_var_name(Input), name of the environment value.
/// @param: env_var_value(Input), value of the environment value.s
/// @return: void.
void SetEnvVar(std::string env_var_name, std::string env_var_value);

/// @brief: Gets the value of environment value.
/// @param: env_var_name(Input), name of the environment value.
/// @return: std::string, value of the environment value, returned as string.
std::string GetEnvVar(std::string env_var_name);

/// @brief: Gets the process ID.
/// @param: void
/// @return: int, process ID returned as int.
int GetProcessId();

/// @brief: Gets the max virtual memory size accessible to the application.
/// @param: void.
/// @return: size_t, size of the accessible memory to the application.
size_t GetUserModeVirtualMemorySize();

/// @brief: Gets the max physical host system memory size.
/// @param: void.
/// @return: size_t, size of the physical host system memory.
size_t GetUsablePhysicalHostMemorySize();

/// @brief: Gets the virtual memory base address. It is hardcoded to 0.
/// @param: void.
/// @return: uintptr_t, always 0.
uintptr_t GetUserModeVirtualMemoryBase();

/// @brief os event api, create an event
/// @param: auto_reset whether an event can reset the status automatically
/// @param: init_state initial state of the event
/// @return: event handle
EventHandle CreateOsEvent(bool auto_reset, bool init_state);

/// @brief os event api, destroy an event
/// @param: event handle
/// @return: whether destroy is correct
int DestroyOsEvent(EventHandle event);

/// @brief os event api, wait on event
/// @param: event Event handle
/// @param: milli_seconds wait time
/// @return: Indicate success or timeout
int WaitForOsEvent(EventHandle event, unsigned int milli_seconds);

/// @brief os event api, set event state
/// @param: event Event handle
/// @return: Whether event set is correct
int SetOsEvent(EventHandle event);

/// @brief os event api, reset event state
/// @param: event Event handle
/// @return: Whether event reset is correct
int ResetOsEvent(EventHandle event);

/// @brief reads a clock which is deemed to be accurate for elapsed time
/// measurements, though not necessarilly fast to query
/// @return clock counter value
uint64_t ReadAccurateClock();

/// @brief retrieves the frequency in Hz of the unit used in ReadAccurateClock.
/// It does not necessarilly reflect the resolution of the clock, but is the
/// value needed to convert a difference in the clock's counter value to elapsed
/// seconds.  This frequency does not change at runtime.
/// @return returns the frequency
uint64_t AccurateClockFrequency();

/// @brief read the system clock which serves as the HSA system clock
/// counter in KFD.
uint64_t ReadSystemClock();

/// @brief read the system clock frequency
uint64_t SystemClockFrequency();

typedef struct cpuid_s {
  char ManufacturerID[13];  // 12 char, NULL terminated
  bool mwaitx;
} cpuid_t;

/// @brief parse CPUID
/// @param: cpuinfo struct to be filled
bool ParseCpuID(cpuid_t* cpuinfo);

}   //  namespace os
}   //  namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_OS_H_
