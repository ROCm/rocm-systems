////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2026, Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef __APPLE__
#include "core/util/os.h"
#include "core/util/utils.h"

#include <dlfcn.h>
#include <pthread.h>
#include <limits.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include <unistd.h>
#include <errno.h>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace rocr {
namespace os {

struct ThreadArgs {
  void* entry_args;
  ThreadEntry entry_function;
};

void* __stdcall ThreadTrampoline(void* arg) {
  ThreadArgs* ar = (ThreadArgs*)arg;
  ThreadEntry CallMe = ar->entry_function;
  void* Data = ar->entry_args;
  CallMe(Data);
  return nullptr;
}

// Thread container allows multiple waits and separate close (destroy).
// Darwin notes:
//   - pthread_setaffinity_np / cpu_set_t are not portable API on Darwin.
//     Affinity is a hint only (thread_policy_set / THREAD_AFFINITY_POLICY);
//     we skip the explicit CPU set. Thread scheduling still honors priority.
class os_thread {
 public:
  explicit os_thread(ThreadEntry function,
                      void* threadArgument,
                      uint stackSize,
                      int priority)
      : thread(0), lock(nullptr), state(RUNNING) {
    int err;
    lock = CreateMutex();
    if (lock == nullptr) return;

    args.entry_args = threadArgument;
    args.entry_function = function;

    pthread_attr_t attrib;
    err = pthread_attr_init(&attrib);
    if (err != 0) {
      fprintf(stderr, "pthread_attr_init failed: %s\n", strerror(err));
      return;
    }

    MAKE_SCOPE_GUARD([&]() {
      if (pthread_attr_destroy(&attrib))
        fprintf(stderr, "pthread_attr_destroy failed: %s\n", strerror(err));
    });

    if (stackSize != 0) {
      stackSize = Max(uint(PTHREAD_STACK_MIN), stackSize);
      stackSize = AlignUp(stackSize, 4096);
      err = pthread_attr_setstacksize(&attrib, stackSize);
      if (err != 0) {
        fprintf(stderr, "pthread_attr_setstacksize failed: %s\n", strerror(err));
        return;
      }
    }

    do {
      err = pthread_create(&thread, &attrib, ThreadTrampoline, &args);
      if (!err) break;

      if (err != EINVAL || stackSize == 0) {
        fprintf(stderr, "pthread_create failed %d (%s)\n", errno, strerror(errno));
        thread = 0;
        return;
      }

      // Probably a stack size error since system limits can be different from PTHREAD_STACK_MIN
      // Attempt to grow the stack within reason.
      stackSize *= 2;
      if (pthread_attr_setstacksize(&attrib, stackSize)) {
        fprintf(stderr, "pthread_attr_setstacksize failed: %s\n", strerror(err));
        thread = 0;
        return;
      }
    } while (stackSize < 20 * 1024 * 1024);

    struct sched_param param = {};
    if (priority != OS_THREAD_PRIORITY_DEFAULT) {
      int set_priority;
      int max_priority = sched_get_priority_max(SCHED_FIFO);

      if (priority == OS_THREAD_PRIORITY_MAX)
        set_priority = max_priority;
      else if (priority == OS_THREAD_PRIORITY_HIGH)
        set_priority = max_priority - 1;
      else if (priority > max_priority)
        set_priority = max_priority;
      else
        set_priority = priority;

      param.sched_priority = set_priority;
      if (pthread_setschedparam(thread, SCHED_FIFO, &param)) {
        fprintf(stderr, "pthread_setschedparam failed\n");
        return;
      }

      int policy = 0;
      if (pthread_getschedparam(thread, &policy, &param))
        fprintf(stderr, "pthread_getschedparam failed: %s\n", strerror(err));

      if (policy != SCHED_FIFO || param.sched_priority != set_priority)
        fprintf(stderr, "Failed to adjust thread priority (policy:%s requested:%d current:%d)\n",
                          policy == SCHED_FIFO ? "FIFO" :
                          policy == SCHED_OTHER ? "OTHER" :
                          policy == SCHED_RR ? "RR" : "Unknown",
                          set_priority, param.sched_priority);
    }
  }

  os_thread(os_thread&& rhs) {
    thread = rhs.thread;
    args = rhs.args;
    lock = rhs.lock;
    state = int(rhs.state);
    rhs.thread = 0;
    rhs.lock = nullptr;
  }

  os_thread(os_thread&) = delete;

  ~os_thread() {
    if (lock != nullptr) DestroyMutex(lock);
    if ((state == RUNNING) && (thread != 0)) {
      int err = pthread_detach(thread);
      if (err != 0) fprintf(stderr, "pthread_detach failed: %s\n", strerror(err));
    }
  }

  bool Valid() { return (lock != nullptr) && (thread != 0); }

  bool Wait() {
    if (state == FINISHED) return true;
    AcquireMutex(lock);
    if (state == FINISHED) {
      ReleaseMutex(lock);
      return true;
    }
    int err = pthread_join(thread, NULL);
    bool success = (err == 0);
    if (success) state = FINISHED;
    ReleaseMutex(lock);
    return success;
  }

 private:
  pthread_t thread;
  struct ThreadArgs args;
  Mutex lock;
  std::atomic<int> state;
  enum { FINISHED = 0, RUNNING = 1 };
};

static_assert(sizeof(LibHandle) == sizeof(void*), "OS abstraction size mismatch");
// On Darwin, POSIX unnamed semaphores are deprecated (sem_init returns ENOSYS
// since macOS 10.10). We use GCD dispatch_semaphore_t instead; the Semaphore
// opaque handle maps to a dispatch_semaphore_t.
static_assert(sizeof(Semaphore) == sizeof(dispatch_semaphore_t),
              "OS abstraction size mismatch");
static_assert(sizeof(Mutex) == sizeof(pthread_mutex_t*), "OS abstraction size mismatch");
static_assert(sizeof(SharedMutex) == sizeof(pthread_rwlock_t*), "OS abstraction size mismatch");
static_assert(sizeof(Thread) == sizeof(os_thread*), "OS abstraction size mismatch");

LibHandle LoadLib(std::string filename) {
  void* ret = dlopen(filename.c_str(), RTLD_LAZY);
  if (ret == nullptr) debug_print("LoadLib(%s) failed: %s\n", filename.c_str(), dlerror());
  return *(LibHandle*)&ret;
}

void* GetExportAddress(LibHandle lib, std::string export_name) {
  void* ret = dlsym(*(void**)&lib, export_name.c_str());

  // dlsym searches the given library and all the library's load dependencies.
  // Remaining code limits symbol lookup to only the library handle given.
  // This lookup pattern matches Windows.
  if (ret == NULL) return ret;

  // On Darwin there's no dlinfo(RTLD_DI_LINKMAP). Use dladdr on the returned
  // symbol address to find the filename of the image that provides it, and
  // compare against the filename the caller's handle resolves to (via the
  // dyld image list).
  Dl_info info;
  if (dladdr(ret, &info) == 0 || info.dli_fname == nullptr) {
    fprintf(stderr, "dladdr failed.\n");
    return nullptr;
  }

  // Find the filename of the image the caller's handle refers to.
  for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
    const char* image_name = _dyld_get_image_name(i);
    if (!image_name) continue;
    // Re-open the image without loading (bumps refcount; drop after).
    void* h = dlopen(image_name, RTLD_NOLOAD | RTLD_LAZY);
    if (!h) continue;
    dlclose(h);
    if (h == *(void**)&lib) {
      if (strcmp(info.dli_fname, image_name) == 0) return ret;
      return NULL;
    }
  }
  // Caller's handle didn't match any dyld image; fall back to accepting it.
  return ret;
}

bool CloseLib(LibHandle lib) { return (dlclose(*(void**)&lib) == 0) ? true : false; }

// Enumerate loaded Mach-O images and find any that export a
// "HSA_AMD_TOOL_PRIORITY" symbol. Mirrors the Linux dl_iterate_phdr path.
std::vector<LibHandle> GetLoadedToolsLib() {
  std::vector<LibHandle> ret;
  const char* kSymbol = "HSA_AMD_TOOL_PRIORITY";

  for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
    const char* name = _dyld_get_image_name(i);
    if (!name || name[0] == '\0') continue;

    // dlopen with RTLD_NOLOAD returns a handle to an already-loaded image
    // without creating a new mapping. If the image isn't already loaded, we
    // get NULL and skip.
    void* h = dlopen(name, RTLD_NOLOAD | RTLD_LAZY);
    if (!h) continue;

    if (dlsym(h, kSymbol) != nullptr) {
      // Record as a library handle; caller holds the reference.
      ret.push_back(*(LibHandle*)&h);
    } else {
      // Not a tool library — drop the refcount we just took.
      dlclose(h);
    }
  }
  return ret;
}

std::string GetLibraryName(LibHandle lib) {
  // Darwin has no dlinfo(RTLD_DI_LINKMAP). Iterate dyld images and compare
  // their RTLD_NOLOAD handles to the caller's handle.
  for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
    const char* name = _dyld_get_image_name(i);
    if (!name) continue;
    void* h = dlopen(name, RTLD_NOLOAD | RTLD_LAZY);
    if (!h) continue;
    dlclose(h);
    if (h == *(void**)&lib) return name;
  }
  return "";
}

Semaphore CreateSemaphore() {
  // GCD dispatch_semaphore; initial value 0 matches sem_init(..., 0, 0).
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  return *(Semaphore*)&sem;
}

bool WaitSemaphore(Semaphore sem) {
  // dispatch_semaphore_wait returns 0 on success; DISPATCH_TIME_FOREVER never
  // times out and is not interrupted by signals, so no EINTR retry is needed.
  long rc = dispatch_semaphore_wait(*(dispatch_semaphore_t*)&sem,
                                    DISPATCH_TIME_FOREVER);
  return rc == 0;
}

void PostSemaphore(Semaphore sem) {
  // Mirror Linux behavior: only post if there are no waiters to avoid value
  // build-up. dispatch_semaphore_signal wakes one waiter and returns non-zero
  // if it actually unblocked a thread, otherwise it increments the value.
  // There's no public "get value" API for dispatch_semaphore, so we always
  // signal; downstream code treats Semaphore as edge-triggered so the extra
  // value is benign.
  dispatch_semaphore_signal(*(dispatch_semaphore_t*)&sem);
}

void DestroySemaphore(Semaphore sem) {
  // dispatch objects are ARC-managed in ObjC; in C/C++ we release manually.
  dispatch_release(*(dispatch_semaphore_t*)&sem);
}

Mutex CreateMutex() {
  pthread_mutex_t* mutex = new pthread_mutex_t;
  pthread_mutex_init(mutex, NULL);
  return *(Mutex*)&mutex;
}

bool TryAcquireMutex(Mutex lock) {
  return pthread_mutex_trylock(*(pthread_mutex_t**)&lock) == 0;
}

bool AcquireMutex(Mutex lock) {
  return pthread_mutex_lock(*(pthread_mutex_t**)&lock) == 0;
}

void ReleaseMutex(Mutex lock) {
  pthread_mutex_unlock(*(pthread_mutex_t**)&lock);
}

void DestroyMutex(Mutex lock) {
  pthread_mutex_destroy(*(pthread_mutex_t**)&lock);
  delete *(pthread_mutex_t**)&lock;
}

void Sleep(int delay_in_millisec) { usleep(delay_in_millisec * 1000); }

void uSleep(int delayInUs) { usleep(delayInUs); }

void YieldThread() { sched_yield(); }

Thread CreateThread(ThreadEntry function, void* threadArgument, uint stackSize, int priority) {
  os_thread* result = new os_thread(function, threadArgument, stackSize, priority);
  if (!result->Valid()) {
    delete result;
    return nullptr;
  }

  return reinterpret_cast<Thread>(result);
}

void CloseThread(Thread thread) { delete reinterpret_cast<os_thread*>(thread); }

bool WaitForThread(Thread thread) { return reinterpret_cast<os_thread*>(thread)->Wait(); }

bool WaitForAllThreads(Thread* threads, uint threadCount) {
  for (uint i = 0; i < threadCount; i++) WaitForThread(threads[i]);
  return true;
}

bool IsEnvVarSet(std::string env_var_name) {
  char* buff = NULL;
  buff = getenv(env_var_name.c_str());
  return (buff != NULL);
}

void SetEnvVar(std::string env_var_name, std::string env_var_value) {
  setenv(env_var_name.c_str(), env_var_value.c_str(), 1);
}

int GetProcessId() {
  return ::getpid();
}

std::string GetEnvVar(std::string env_var_name) {
  char* buff;
  buff = getenv(env_var_name.c_str());
  std::string ret;
  if (buff) {
    ret = buff;
  }
  return ret;
}

size_t GetUserModeVirtualMemorySize() {
#ifdef _LP64
  // Darwin (x86_64 + arm64) follows the same user/kernel split as Linux for
  // practical purposes — 47 bits of usable user VA on both.
  return (size_t)(0x800000000000);
#else
  return (size_t)(0xffffffff);  // ~4GB
#endif
}

size_t GetUsablePhysicalHostMemorySize() {
  // Darwin: query total RAM via sysctl(HW_MEMSIZE). There's no sysinfo().
  uint64_t mem_bytes = 0;
  size_t len = sizeof(mem_bytes);
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  if (sysctl(mib, 2, &mem_bytes, &len, NULL, 0) != 0) {
    return 0;
  }
  const size_t physical_size = static_cast<size_t>(mem_bytes);
  return std::min(GetUserModeVirtualMemorySize(), physical_size);
}

uintptr_t GetUserModeVirtualMemoryBase() { return (uintptr_t)0; }

// Os event implementation
typedef struct EventDescriptor_ {
  pthread_cond_t event;
  pthread_mutex_t mutex;
  bool state;
  bool auto_reset;
} EventDescriptor;

EventHandle CreateOsEvent(bool auto_reset, bool init_state) {
  EventDescriptor* eventDescrp;
  eventDescrp = (EventDescriptor*)malloc(sizeof(EventDescriptor));

  if(!eventDescrp) { return nullptr; }

  pthread_mutex_init(&eventDescrp->mutex, NULL);
  pthread_cond_init(&eventDescrp->event, NULL);
  eventDescrp->auto_reset = auto_reset;
  eventDescrp->state = init_state;

  EventHandle handle = reinterpret_cast<EventHandle>(eventDescrp);

  return handle;
}

int DestroyOsEvent(EventHandle event) {
  if (event == NULL) {
    return -1;
  }

  EventDescriptor* eventDescrp = reinterpret_cast<EventDescriptor*>(event);
  int ret_code = pthread_cond_destroy(&eventDescrp->event);
  ret_code |= pthread_mutex_destroy(&eventDescrp->mutex);
  free(eventDescrp);
  return ret_code;
}

int WaitForOsEvent(EventHandle event, unsigned int milli_seconds) {
  if (event == NULL) {
    return -1;
  }

  EventDescriptor* eventDescrp = reinterpret_cast<EventDescriptor*>(event);
  // Event wait time is 0 and state is non-signaled, return directly
  if (milli_seconds == 0) {
    int tmp_ret = pthread_mutex_trylock(&eventDescrp->mutex);
    if (tmp_ret == EBUSY) {
      // Timeout
      return 1;
    }
  } else {
      pthread_mutex_lock(&eventDescrp->mutex);
  }

  int ret_code = 0;

  if (!eventDescrp->state) {
    if (milli_seconds == 0) {
      ret_code = 1;
    } else {
      struct timespec ts;
      struct timeval tp;

      ret_code = gettimeofday(&tp, NULL);
      ts.tv_sec = tp.tv_sec;
      ts.tv_nsec = tp.tv_usec * 1000;

      unsigned int sec = milli_seconds / 1000;
      unsigned int mSec = milli_seconds % 1000;

      ts.tv_sec += sec;
      ts.tv_nsec += mSec * 1000000;

      // More then one second, add 1 sec to the tv_sec elem
      if (ts.tv_nsec > 1000000000) {
        ts.tv_sec += 1;
        ts.tv_nsec = ts.tv_nsec - 1000000000;
      }

      ret_code =
          pthread_cond_timedwait(&eventDescrp->event, &eventDescrp->mutex, &ts);
      // Darwin uses ETIMEDOUT (60) rather than Linux's 110; map both to the
      // HSA "timeout" encoding for parity with the Linux path.
      if (ret_code == ETIMEDOUT) {
        ret_code = 0x14003;
      }

      if (ret_code == 0 && eventDescrp->auto_reset) {
        eventDescrp->state = false;
      }
    }
  } else if (eventDescrp->auto_reset) {
    eventDescrp->state = false;
  }
  pthread_mutex_unlock(&eventDescrp->mutex);

  return ret_code;
}

int SetOsEvent(EventHandle event) {
  if (event == NULL) {
    return -1;
  }

  EventDescriptor* eventDescrp = reinterpret_cast<EventDescriptor*>(event);
  int ret_code = 0;
  ret_code = pthread_mutex_lock(&eventDescrp->mutex);
  eventDescrp->state = true;
  ret_code = pthread_mutex_unlock(&eventDescrp->mutex);
  ret_code |= pthread_cond_signal(&eventDescrp->event);

  return ret_code;
}

int ResetOsEvent(EventHandle event) {
  if (event == NULL) {
    return -1;
  }

  EventDescriptor* eventDescrp = reinterpret_cast<EventDescriptor*>(event);
  int ret_code = 0;
  ret_code = pthread_mutex_lock(&eventDescrp->mutex);
  eventDescrp->state = false;
  ret_code = pthread_mutex_unlock(&eventDescrp->mutex);

  return ret_code;
}

static double invPeriod = 0.0;

uint64_t ReadAccurateClock() {
  if (invPeriod == 0.0) AccurateClockFrequency();
  timespec time;
  // CLOCK_MONOTONIC_RAW is available on Darwin since macOS 10.12.
  int err = clock_gettime(CLOCK_MONOTONIC_RAW, &time);
  if (err != 0) {
    perror("clock_gettime(CLOCK_MONOTONIC_RAW,...) failed");
    abort();
  }
  return (uint64_t(time.tv_sec) * 1000000000ull + uint64_t(time.tv_nsec)) * invPeriod;
}

uint64_t AccurateClockFrequency() {
  // Darwin's CLOCK_MONOTONIC_RAW is reliable; no kernel-version check needed.
  timespec time;
  int err = clock_getres(CLOCK_MONOTONIC_RAW, &time);
  if (err != 0) {
    perror("clock_getres failed");
    abort();
  }
  if (time.tv_sec != 0 || time.tv_nsec >= 0xFFFFFFFF) {
    fprintf(stderr,
            "clock_getres(CLOCK_MONOTONIC_RAW,...) returned very low "
            "frequency (<1Hz).\n");
    abort();
  }
  if (invPeriod == 0.0) invPeriod = 1.0 / double(time.tv_nsec);
  return 1000000000ull / uint64_t(time.tv_nsec);
}

SharedMutex CreateSharedMutex() {
  // Darwin has neither pthread_rwlockattr_setkind_np nor writer-preferred
  // rwlock attributes. Fall through to the default attribute set; the "HSA
  // system clock" path does not depend on writer-preference for correctness.
  pthread_rwlockattr_t attrib;
  int err = pthread_rwlockattr_init(&attrib);
  if (err != 0) {
    fprintf(stderr, "rw lock attribute init failed: %s\n", strerror(err));
    return nullptr;
  }

  std::unique_ptr<pthread_rwlock_t> lock(new pthread_rwlock_t);
  err = pthread_rwlock_init(lock.get(), &attrib);
  if (err != 0) {
    fprintf(stderr, "rw lock init failed: %s\n", strerror(err));
    return nullptr;
  }

  pthread_rwlockattr_destroy(&attrib);
  return lock.release();
}

bool TryAcquireSharedMutex(SharedMutex lock) {
  int err = pthread_rwlock_trywrlock(*(pthread_rwlock_t**)&lock);
  return err == 0;
}

bool AcquireSharedMutex(SharedMutex lock) {
  int err = pthread_rwlock_wrlock(*(pthread_rwlock_t**)&lock);
  return err == 0;
}

void ReleaseSharedMutex(SharedMutex lock) {
  int err = pthread_rwlock_unlock(*(pthread_rwlock_t**)&lock);
  if (err != 0) {
    fprintf(stderr, "SharedMutex unlock failed: %s\n", strerror(err));
    abort();
  }
}

bool TrySharedAcquireSharedMutex(SharedMutex lock) {
  int err = pthread_rwlock_tryrdlock(*(pthread_rwlock_t**)&lock);
  return err == 0;
}

bool SharedAcquireSharedMutex(SharedMutex lock) {
  int err = pthread_rwlock_rdlock(*(pthread_rwlock_t**)&lock);
  return err == 0;
}

void SharedReleaseSharedMutex(SharedMutex lock) {
  int err = pthread_rwlock_unlock(*(pthread_rwlock_t**)&lock);
  if (err != 0) {
    fprintf(stderr, "SharedMutex unlock failed: %s\n", strerror(err));
    abort();
  }
}

void DestroySharedMutex(SharedMutex lock) {
  pthread_rwlock_destroy(*(pthread_rwlock_t**)&lock);
  delete *(pthread_rwlock_t**)&lock;
}

static uint64_t sys_clock_period_ = 0;

uint64_t ReadSystemClock() {
  // Linux uses CLOCK_BOOTTIME (counts across suspend). Darwin does not have
  // that clock ID; mach_continuous_time() is the equivalent (continues across
  // sleep). Convert ticks → nanoseconds via mach_timebase_info.
  static mach_timebase_info_data_t tb = {0, 0};
  if (tb.denom == 0) mach_timebase_info(&tb);
  uint64_t ticks = mach_continuous_time();
  uint64_t ns = ticks * tb.numer / tb.denom;
  if (sys_clock_period_ != 1 && sys_clock_period_ != 0)
    return ns / sys_clock_period_;
  return ns;
}

uint64_t SystemClockFrequency() {
  // mach_continuous_time() has nanosecond-equivalent resolution once scaled
  // through mach_timebase_info, so the effective period is 1 ns → 1 GHz.
  static mach_timebase_info_data_t tb = {0, 0};
  if (tb.denom == 0) mach_timebase_info(&tb);
  // The reported "tick" period in ns is tb.numer / tb.denom.
  sys_clock_period_ = tb.numer / tb.denom;
  if (sys_clock_period_ == 0) sys_clock_period_ = 1;
  return 1000000000ull / sys_clock_period_;
}

bool ParseCpuID(cpuid_t* cpuinfo) {
  // Darwin on Apple Silicon is arm64; no CPUID. Darwin on Intel does have
  // CPUID but lacks GCC's <cpuid.h> helpers — would need raw inline asm.
  // Not worth carrying; ROCR only uses cpuinfo.mwaitx as an AMD-CPU hint
  // for the platform host agent, which isn't an AMD CPU on Macs.
  memset(cpuinfo, 0, sizeof(*cpuinfo));
  return false;
}

uint64_t TimeNanos() {
  struct timespec tp;
  ::clock_gettime(CLOCK_MONOTONIC, &tp);
  return (uint64_t)tp.tv_sec * (1000ULL * 1000ULL * 1000ULL) + (uint64_t)tp.tv_nsec;
}

static inline int MemProtToOsProt(MemProt prot) {
  switch (prot) {
    case MEM_PROT_NONE:
      return PROT_NONE;
    case MEM_PROT_READ:
      return PROT_READ;
    case MEM_PROT_RW:
      return PROT_READ | PROT_WRITE;
    case MEM_PROT_RWX:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      break;
  }
  return -1;
}

size_t PageSize() {
  static size_t g_page_size_ = 0;  //!< The default os page size
  if (g_page_size_ == 0) {
    g_page_size_ = (size_t)::sysconf(_SC_PAGESIZE);
  }
  return g_page_size_;
}

bool UnmapMemory(void* va, size_t size) { return ::munmap(va, size) == 0; }

bool MapMemory(void* va, size_t size, MemProt perms, int fd, uint64_t cpu_addr) {
  void* mapped_ptr = ::mmap(va, size, MemProtToOsProt(perms),
                            MAP_SHARED | MAP_FIXED, fd, cpu_addr);
  if (mapped_ptr != va)
      return false;
  return true;
}

void* ReserveMemory(void* start, size_t size, size_t alignment, MemProt prot) {
  size = AlignUp(size, PageSize());
  // check for invalid input size
  if (size == 0) {
    return NULL;
  }
  alignment = std::max(PageSize(), AlignUp(alignment, PageSize()));
  assert(IsPowerOfTwo(alignment) && "not a power of 2");

  size_t requested = size + alignment - PageSize();
  // Darwin's MAP_NORESERVE is a no-op (accepted but ignored); MAP_ANONYMOUS
  // is aliased to MAP_ANON in <sys/mman.h>.
  address mem = (address)::mmap(start, requested, MemProtToOsProt(prot),
                                MAP_PRIVATE | MAP_NORESERVE | MAP_ANONYMOUS, -1, 0);

  // check for out of memory
  if (mem == MAP_FAILED) return NULL;

  address aligned = AlignUp(mem, alignment);

  // return the unused leading pages to the free state
  if (&aligned[0] != &mem[0]) {
    assert(&aligned[0] > &mem[0] && "check this code");
    if (::munmap(&mem[0], &aligned[0] - &mem[0]) != 0) {
      assert(!"::munmap failed");
    }
  }
  // return the unused trailing pages to the free state
  if (&aligned[size] != &mem[requested]) {
    assert(&aligned[size] < &mem[requested] && "check this code");
    if (::munmap(&aligned[size], &mem[requested] - &aligned[size]) != 0) {
      assert(!"::munmap failed");
    }
  }

  // Darwin has no MADV_HUGEPAGE; large pages are controlled via VM_FLAGS_SUPERPAGE_*
  // on mach_vm_* APIs, which this reservation path doesn't exercise. Skip.

  return aligned;
}

bool ReleaseMemory(void* addr, size_t size) {
  assert(IsMultipleOf(addr, PageSize()) && "not page aligned!");
  size = AlignUp(size, PageSize());

  return 0 == ::munmap(addr, size);
}

bool CommitMemory(void* addr, size_t size, MemProt prot) {
  assert(IsMultipleOf(addr, PageSize()) && "not page aligned!");
  size = AlignUp(size, PageSize());

  return ::mmap(addr, size, MemProtToOsProt(prot), MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, -1,
                0) != MAP_FAILED;
}

bool UncommitMemory(void* addr, size_t size) {
  assert(IsMultipleOf(addr, PageSize()) && "not page aligned!");
  size = AlignUp(size, PageSize());

  return ::mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE | MAP_ANONYMOUS, -1,
                0) != MAP_FAILED;
}

bool ProtectMemory(void* va, size_t size, MemProt perms) {
  return ::mprotect(va, size, MemProtToOsProt(perms)) == 0;
}

uint64_t HostTotalPhysicalMemory() {
  // Linux path uses sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE); Darwin
  // doesn't define _SC_PHYS_PAGES. Use sysctl(HW_MEMSIZE) which returns
  // total installed RAM in bytes.
  static uint64_t totalPhys = 0;

  if (totalPhys != 0) {
    return totalPhys;
  }

  uint64_t mem_bytes = 0;
  size_t len = sizeof(mem_bytes);
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  if (sysctl(mib, 2, &mem_bytes, &len, NULL, 0) == 0) {
    totalPhys = mem_bytes;
  }
  return totalPhys;
}

int Ffs(int i) { return ffs(i); }

int Ctz(uint64_t i) { return __builtin_ctz(i); }

char* DlError() { return dlerror(); }

// --- IPC sockets ---
// Darwin does NOT support the Linux abstract-namespace convention of a
// leading NUL byte in sun_path. Use a filesystem-path socket under $TMPDIR
// (or /tmp) instead. Callers still see an opaque name; we hash/transform it
// to a filesystem path behind the scenes.
static std::string IPCNameToPath(const char* name) {
  const char* tmp = getenv("TMPDIR");
  if (!tmp || !*tmp) tmp = "/tmp";
  std::string path = tmp;
  // Avoid trailing-slash double-slash, strip trailing '/' from TMPDIR.
  if (!path.empty() && path.back() == '/') path.pop_back();
  path += "/rocm-ipc-";
  path += name ? name : "";
  // sockaddr_un.sun_path is 104 bytes on Darwin; truncate safely.
  if (path.size() >= sizeof(sockaddr_un::sun_path))
    path.resize(sizeof(sockaddr_un::sun_path) - 1);
  return path;
}

static inline int IPCSockToFd(IPCSocket sock) {
  return static_cast<int>(sock);
}

static inline IPCSocket FdToIPCSock(int fd) {
  return static_cast<IPCSocket>(fd);
}

IPCSocket CreateIPCServer(const char* name, int backlog) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd == -1) return INVALID_SOCKET_VALUE;

  std::string path = IPCNameToPath(name);
  // Filesystem-path sockets persist stale inodes across crashes — unlink first.
  ::unlink(path.c_str());

  struct sockaddr_un address;
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);

  if (bind(fd, (struct sockaddr*)&address, sizeof(address)) != 0) {
    close(fd);
    return INVALID_SOCKET_VALUE;
  }
  if (listen(fd, backlog) != 0) {
    close(fd);
    return INVALID_SOCKET_VALUE;
  }
  return FdToIPCSock(fd);
}

IPCSocket AcceptIPCConnection(IPCSocket server) {
  int fd = accept(IPCSockToFd(server), NULL, NULL);
  if (fd == -1) return INVALID_SOCKET_VALUE;
  return FdToIPCSock(fd);
}

IPCSocket ConnectToIPCServer(const char* name, std::chrono::milliseconds timeout,
                             std::chrono::milliseconds retryInterval) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd == -1) return INVALID_SOCKET_VALUE;

  std::string path = IPCNameToPath(name);

  struct sockaddr_un address;
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);

  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0)
      return FdToIPCSock(fd);
    usleep(static_cast<useconds_t>(retryInterval.count()) * 1000);
  }

  close(fd);
  return INVALID_SOCKET_VALUE;
}

void SetIPCSocketRecvTimeout(IPCSocket sock, std::chrono::seconds timeout) {
  struct timeval tv;
  tv.tv_sec = static_cast<time_t>(timeout.count());
  tv.tv_usec = 0;
  setsockopt(IPCSockToFd(sock), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

int IPCSocketRead(IPCSocket conn, void* buf, size_t len) {
  return static_cast<int>(read(IPCSockToFd(conn), buf, len));
}

int IPCSocketWrite(IPCSocket conn, const void* buf, size_t len) {
  return static_cast<int>(write(IPCSockToFd(conn), buf, len));
}

int IPCSendHandle(IPCSocket conn, intptr_t handle) {
  int fd = static_cast<int>(handle);
  char iov_buf[1] = {'y'};
  struct iovec io = {.iov_base = iov_buf, .iov_len = 1};

  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  memset(cmsg_buf, 0, sizeof(cmsg_buf));

  struct msghdr msg = {};
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg) return -1;
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

  msg.msg_controllen = CMSG_SPACE(sizeof(int));

  return (sendmsg(IPCSockToFd(conn), &msg, 0) < 0) ? -1 : 0;
}

intptr_t IPCRecvHandle(IPCSocket conn) {
  char m_buffer[1];
  struct iovec io = {.iov_base = m_buffer, .iov_len = sizeof(m_buffer)};

  char c_buffer[256];
  struct msghdr msg = {};
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = c_buffer;
  msg.msg_controllen = sizeof(c_buffer);

  ssize_t rcv = recvmsg(IPCSockToFd(conn), &msg, MSG_WAITALL);
  if (rcv < 0) return -1;

  while (!rcv)
    rcv = recvmsg(IPCSockToFd(conn), &msg, MSG_WAITALL);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg) return -1;
  int fd;
  memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
  return fd;
}

void CloseIPCSocket(IPCSocket sock) {
  if (sock != INVALID_SOCKET_VALUE)
    close(IPCSockToFd(sock));
}

}   //  namespace os
}   //  namespace rocr

#endif  // __APPLE__
