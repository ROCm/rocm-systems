/* Copyright (c) 2008 - 2023 Advanced Micro Devices, Inc.

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

#if !defined(_WIN32) && !defined(__CYGWIN__)

#include "os/os.hpp"
#include "thread/thread.hpp"
#include "utils/util.hpp"
#include "utils/flags.hpp"

#include <iostream>
#include <stdarg.h>

#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <signal.h>
#include <cxxabi.h>

#include <sys/prctl.h>
#include <sys/resource.h>

#include <link.h>
#include <time.h>
#ifndef DT_GNU_HASH
#define DT_GNU_HASH 0x6ffffef5
#endif  // DT_GNU_HASH

#ifdef ROCCLR_SUPPORT_NUMA_POLICY
#include <numa.h>
#endif  // ROCCLR_SUPPORT_NUMA_POLICY

#include <atomic>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>  // for strncmp
#include <cstdlib>
#include <cstdio>  // for tempnam
#include <limits.h>
#include <memory>
#include <algorithm>
#include <mutex>
#include <fstream>

namespace amd {

static struct sigaction oldSigAction;

static bool callOldSignalHandler(int sig, siginfo_t* info, void* ptr) {
  if (oldSigAction.sa_handler == SIG_DFL) {
    // no signal handler was previously installed.
    return false;
  } else if (oldSigAction.sa_handler != SIG_IGN) {
    if ((oldSigAction.sa_flags & SA_NODEFER) == 0) {
      sigaddset(&oldSigAction.sa_mask, sig);
    }

    void (*handler)(int) = oldSigAction.sa_handler;
    if (oldSigAction.sa_flags & SA_RESETHAND) {
      oldSigAction.sa_handler = SIG_DFL;
    }

    sigset_t savedSigSet;
    pthread_sigmask(SIG_SETMASK, &oldSigAction.sa_mask, &savedSigSet);

    if (oldSigAction.sa_flags & SA_SIGINFO) {
      oldSigAction.sa_sigaction(sig, info, ptr);
    } else {
      handler(sig);
    }

    pthread_sigmask(SIG_SETMASK, &savedSigSet, NULL);
  }

  return true;
}

static void divisionErrorHandler(int sig, siginfo_t* info, void* ptr) {
  assert(info != NULL && ptr != NULL && "just checking");
  ucontext_t* uc = (ucontext_t*)ptr;
  address insn;

#if defined(ATI_ARCH_X86)
  insn = (address)uc->uc_mcontext.gregs[LP64_SWITCH(REG_EIP, REG_RIP)];
#else
  assert(!"Unimplemented");
#endif

  if (Thread::current()->isWorkerThread()) {
    if (Os::skipIDIV(insn)) {
#if defined(ATI_ARCH_X86)
      uc->uc_mcontext.gregs[LP64_SWITCH(REG_EIP, REG_RIP)] = (greg_t)insn;
#else
      assert(!"Unimplemented");
#endif
      return;
    }
  }

  // Call the chained signal handler
  if (callOldSignalHandler(sig, info, ptr)) {
    return;
  }

  std::cerr << "Unhandled signal in divisionErrorHandler()" << std::endl;
  ::abort();
}

typedef int (*pthread_setaffinity_fn)(pthread_t, size_t, const cpu_set_t*);
static pthread_setaffinity_fn pthread_setaffinity_fptr;

static void init() __attribute__((constructor(101)));
static void init() { Os::init(); }
static cpu_set_t nativeMask_;

bool Os::installSigfpeHandler() {
  // Install a SIGFPE signal handler @todo: Chain the handlers
  struct sigaction sa;
  sigfillset(&sa.sa_mask);
  sa.sa_handler = SIG_DFL;
  sa.sa_sigaction = divisionErrorHandler;
  sa.sa_flags = SA_SIGINFO | SA_RESTART;

  if (sigaction(SIGFPE, &sa, &oldSigAction) != 0) {
    return false;
  }
  return true;
}

void Os::uninstallSigfpeHandler() {}

bool Os::init() {
  static bool initialized_ = false;

  // We could use pthread_once here:
  if (initialized_) {
    return true;
  }
  initialized_ = true;

  pageSize_ = (size_t)::sysconf(_SC_PAGESIZE);
  processorCount_ = ::sysconf(_SC_NPROCESSORS_CONF);

  pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &nativeMask_);
  pthread_setaffinity_fptr = (pthread_setaffinity_fn)dlsym(RTLD_NEXT, "pthread_setaffinity_np");

  return Thread::init();
}

static void __exit() __attribute__((destructor(101)));
static void __exit() { Os::tearDown(); }

void Os::tearDown() { Thread::tearDown(); }

void* Os::loadLibrary_(const char* filename) {
  return (*filename == '\0') ? NULL : ::dlopen(filename, RTLD_LAZY);
}

void Os::unloadLibrary(void* handle) { ::dlclose(handle); }

void* Os::getSymbol(void* handle, const char* name) { return ::dlsym(handle, name); }

static inline int memProtToOsProt(Os::MemProt prot) {
  switch (prot) {
    case Os::MEM_PROT_NONE:
      return PROT_NONE;
    case Os::MEM_PROT_READ:
      return PROT_READ;
    case Os::MEM_PROT_RW:
      return PROT_READ | PROT_WRITE;
    case Os::MEM_PROT_RWX:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      break;
  }
  ShouldNotReachHere();
  return -1;
}

address Os::reserveMemory(address start, size_t size, size_t alignment, MemProt prot) {
  size = alignUp(size, pageSize());
  // check for invalid input size
  if (size == 0) {
    return NULL;
  }
  alignment = std::max(pageSize(), alignUp(alignment, pageSize()));
  assert(isPowerOfTwo(alignment) && "not a power of 2");

  size_t requested = size + alignment - pageSize();
  address mem = (address)::mmap(start, requested, memProtToOsProt(prot),
                                MAP_PRIVATE | MAP_NORESERVE | MAP_ANONYMOUS, 0, 0);

  // check for out of memory
  if (mem == MAP_FAILED) return NULL;

  address aligned = alignUp(mem, alignment);

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

  // Hint to enable THP for large host allocations which can help in performance gain
  constexpr size_t kLargePageSize = 2 * Mi;
  if (size >= kLargePageSize) {
    int status = madvise(aligned, size, MADV_HUGEPAGE);
    if (status) {
      ClPrint(amd::LOG_DEBUG, amd::LOG_CODE,
              "madvise with advice MADV_HUGEPAGE"
              " starting at address %p and page size 0x%zx, returned %d, errno: %s",
              aligned, size, status, strerror(errno));
    }
  }

  return aligned;
}

bool Os::releaseMemory(void* addr, size_t size) {
  assert(isMultipleOf(addr, pageSize()) && "not page aligned!");
  size = alignUp(size, pageSize());

  return 0 == ::munmap(addr, size);
}

bool Os::commitMemory(void* addr, size_t size, MemProt prot) {
  assert(isMultipleOf(addr, pageSize()) && "not page aligned!");
  size = alignUp(size, pageSize());

  return ::mmap(addr, size, memProtToOsProt(prot), MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, -1,
                0) != MAP_FAILED;
}

bool Os::uncommitMemory(void* addr, size_t size) {
  assert(isMultipleOf(addr, pageSize()) && "not page aligned!");
  size = alignUp(size, pageSize());

  return ::mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE | MAP_ANONYMOUS, -1,
                0) != MAP_FAILED;
}

bool Os::protectMemory(void* addr, size_t size, MemProt prot) {
  assert(isMultipleOf(addr, pageSize()) && "not page aligned!");
  size = alignUp(size, pageSize());

  return 0 == ::mprotect(addr, size, memProtToOsProt(prot));
}

uint64_t Os::hostTotalPhysicalMemory() {
  static uint64_t totalPhys = 0;

  if (totalPhys != 0) {
    return totalPhys;
  }

  totalPhys = sysconf(_SC_PAGESIZE) * sysconf(_SC_PHYS_PAGES);
  return totalPhys;
}

void* Os::alignedMalloc(size_t size, size_t alignment) {
  void* ptr = NULL;
  if (0 == ::posix_memalign(&ptr, alignment, size)) {
    return ptr;
  }
  return NULL;
}

void Os::alignedFree(void* mem) { ::free(mem); }

void Os::currentStackInfo(address* base, size_t* size) {
  // There could be some issue trying to get the pthread_attr of
  // the primordial thread if the pthread library is not present
  // at load time (a binary loads the OpenCL/HIP app/runtime dynamically.
  // We should look into this... -laurent

  pthread_t self = ::pthread_self();

  pthread_attr_t threadAttr;
  if (0 != ::pthread_getattr_np(self, &threadAttr)) {
    fatal("pthread_getattr_np() failed");
  }

  if (0 != ::pthread_attr_getstack(&threadAttr, (void**)base, size)) {
    fatal("pthread_attr_getstack() failed");
  }
  *base += *size;

  ::pthread_attr_destroy(&threadAttr);

  assert(Os::currentStackPtr() >= *base - *size && Os::currentStackPtr() < *base &&
         "just checking");
}

void Os::setCurrentThreadName(const char* name) { ::prctl(PR_SET_NAME, name); }

void Os::setPreferredNumaNode(uint32_t node) {
#ifdef ROCCLR_SUPPORT_NUMA_POLICY
  if (AMD_CPU_AFFINITY && (numa_available() >= 0)) {
    bitmask* bm = numa_allocate_cpumask();
    numa_node_to_cpus(node, bm);
    if (numa_sched_setaffinity(0, bm) < 0) {
      assert(0 && "failed to set affinity");
    }

    numa_free_cpumask(bm);
  }
#endif  // ROCCLR_SUPPORT_NUMA_POLICY
}

void* Thread::entry(Thread* thread) {
  sigset_t set;

  sigfillset(&set);
  pthread_sigmask(SIG_BLOCK, &set, NULL);

  sigemptyset(&set);
  sigaddset(&set, SIGFPE);
  pthread_sigmask(SIG_UNBLOCK, &set, NULL);

  return thread->main();
}

bool Os::isThreadAlive(const Thread& thread) {
  return ::pthread_kill((pthread_t)thread.handle(), 0) == 0;
}

static size_t tlsSize = 0;

// Try to guess the size of TLS (plus some frames)
void* guessTlsSizeThread(void* param) {
  address stackBase;
  address currentFrame;
  size_t stackSize;
  Os::currentStackInfo(&stackBase, &stackSize);
  currentFrame = reinterpret_cast<address>(&stackSize);
  tlsSize = stackBase - currentFrame;
  // align up to page boundary
  tlsSize = alignUp(tlsSize, amd::Os::pageSize());
  return NULL;
}

static void guessTlsSize(void) {
  int retval;
  pthread_t handle;
  pthread_attr_t threadAttr;

  ::pthread_attr_init(&threadAttr);
  retval = ::pthread_create(&handle, &threadAttr, guessTlsSizeThread, NULL);
  if (retval == 0) {
    pthread_join(handle, NULL);
  } else {
    fatal("pthread_create() failed with default stack size");
  }
  ::pthread_attr_destroy(&threadAttr);
}

const void* Os::createOsThread(amd::Thread* thread) {
  pthread_attr_t threadAttr;
  ::pthread_attr_init(&threadAttr);

  if (thread->stackSize_ != 0) {
    size_t guardsize = 0;
    if (0 != ::pthread_attr_getguardsize(&threadAttr, &guardsize)) {
      fatal("pthread_attr_getguardsize() failed");
    }

    static std::once_flag initOnce;
    std::call_once(initOnce, guessTlsSize);
    ::pthread_attr_setstacksize(&threadAttr, thread->stackSize_ + guardsize + tlsSize);
  }

  // We never plan the use join, so free the resources now.
  ::pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_DETACHED);

  pthread_t handle = 0;
  if (0 != ::pthread_create(&handle, &threadAttr, (void* (*)(void*)) & Thread::entry, thread)) {
    thread->setState(Thread::FAILED);
  }

  ::pthread_attr_destroy(&threadAttr);
  return reinterpret_cast<const void*>(handle);
}

void Os::setThreadAffinity(const void* handle, const Os::ThreadAffinityMask& mask) {
  if (pthread_setaffinity_fptr != NULL) {
    pthread_setaffinity_fptr((pthread_t)handle, sizeof(cpu_set_t), &mask.mask_);
  }
}

bool Os::setThreadAffinityToMainThread() {
  if (AMD_CPU_AFFINITY) {
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Setting Affinity to the main thread's affinity");
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &nativeMask_);
  }
  return true;
}

void Os::yield() { ::sched_yield(); }

uint64_t Os::timeNanos() {
  struct timespec tp;
  ::clock_gettime(CLOCK_MONOTONIC, &tp);
  return (uint64_t)tp.tv_sec * (1000ULL * 1000ULL * 1000ULL) + (uint64_t)tp.tv_nsec;
}

uint64_t Os::timerResolutionNanos() {
  static uint64_t resolution = 0;
  if (resolution == 0) {
    struct timespec tp;
    ::clock_getres(CLOCK_MONOTONIC, &tp);
    resolution = (uint64_t)tp.tv_sec * (1000ULL * 1000ULL * 1000ULL) + (uint64_t)tp.tv_nsec;
  }
  return resolution;
}


const char* Os::libraryExtension() { return MACOS_SWITCH(".dylib", ".so"); }

const char* Os::libraryPrefix() { return "lib"; }

const char* Os::objectExtension() { return ".o"; }

char Os::fileSeparator() { return '/'; }

char Os::pathSeparator() { return ':'; }

bool Os::pathExists(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  return S_ISDIR(st.st_mode);
}

bool Os::createPath(const std::string& path) {
  mode_t mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
  size_t pos = 0;
  while (true) {
    pos = path.find(fileSeparator(), pos);
    const std::string currPath = path.substr(0, pos);
    if (!currPath.empty() && !pathExists(currPath)) {
      int ret = mkdir(currPath.c_str(), mode);
      if (ret == -1) return false;
    }
    if (pos == std::string::npos) break;
    ++pos;
  }
  return true;
}

bool Os::removePath(const std::string& path) {
  size_t pos = std::string::npos;
  bool removed = false;
  while (true) {
    const std::string currPath = path.substr(0, pos);
    if (!currPath.empty()) {
      int ret = rmdir(currPath.c_str());
      if (ret == -1) return removed;
      removed = true;
    }
    if (pos == 0) break;
    pos = path.rfind(fileSeparator(), pos == std::string::npos ? pos : pos - 1);
    if (pos == std::string::npos) break;
  }
  return true;
}

int Os::printf(const char* fmt, ...) {
  va_list ap;

  va_start(ap, fmt);
  int len = ::vprintf(fmt, ap);
  va_end(ap);

  return len;
}

// Os::systemCall()
// ================
// Execute a program and return the program exitcode or -1 if there were problems.
// The input argument 'command' is expected to be a space separated string of
// command-line arguments with arguments containing spaces between double-quotes.
//
// In order to avoid duplication of memory, we use vfork()+exec(). vfork() has
// potiential security risks;
//
// In spite of these risks, the alternatives (system() or fork()) create resource
// issues when running conformance test_allocation which stretches the system
// memory to its limits. Thus we will accept this compromise under the condition
// that the runtime will soon remove any need to call out to external commands.
//
// Note that stdin/stdout/stderr of the command are sent to /dev/null.
//
int Os::systemCall(const std::string& command) {
#if 1
  size_t len = command.size();
  char* cmd = new char[len + 1];
  std::memcpy(cmd, command.c_str(), len);
  cmd[len] = 0;

  // Split the command into arguments. This is a very
  // simple parser that only takes care of quotes and
  // doesn't support escaping with back-slash. In
  // the future, Os::systemCall() will either
  // disappear or it will be replaced with an
  // argc/argv interface. This parser also assumes
  // that if an argument is quoted, the whole
  // argument starts and ends with a double-quote.
  bool inQuote = false;
  int argLength = 0;
  int n = 0;
  char* cp = cmd;
  while (*cp) {
    switch (static_cast<int>(*cp)) {
      case ' ':
        if (inQuote) {
          ++argLength;
        } else {
          *cp = '\0';
          argLength = 0;
        }
        break;
      case '"':
        if (inQuote) {
          inQuote = false;
          *cp = '\0';
        } else {
          inQuote = true;
          *cp = '\0';
          argLength = 1;
          ++n;
        }
        break;
      default:
        if (++argLength == 1) {
          ++n;
        }
        break;
    }
    ++cp;
  }

  char** argv = new char*[n + 1];
  int argc = 0;
  cp = cmd;
  do {
    while ('\0' == *cp) {
      ++cp;
    }
    argv[argc++] = cp;
    while ('\0' != *cp) {
      ++cp;
    }
  } while (argc < n);
  argv[argc] = NULL;

  int ret = -1;
  pid_t pid = vfork();
  if (0 == pid) {
    // Child. Redirect stdin/stdout/stderr to /dev/null
    int fdIn = open("/dev/null", O_RDONLY);
    int fdOut = open("/dev/null", O_WRONLY);
    if (0 <= fdIn || 0 <= fdOut) {
      dup2(fdIn, 0);
      dup2(fdOut, 1);
      dup2(fdOut, 2);

      // Execute the program
      execvp(argv[0], argv);
    }
    _exit(-1);
  } else if (0 > pid) {
    // Can't vfork
  } else {
    // Parent - wait for program to complete and get exit code.
    int exitCode;
    if (0 <= waitpid(pid, &exitCode, 0)) {
      ret = exitCode;
    }
  }
  delete[] argv;
  delete[] cmd;

  return ret;
#else
  return ::system(command.c_str());
#endif
}

std::string Os::getEnvironment(const std::string& name) {
  char* dstBuf;

  dstBuf = ::getenv(name.c_str());
  if (dstBuf == NULL) {
    return std::string("");
  }
  return std::string(dstBuf);
}

std::string Os::getTempPath() {
  std::string tempFolder = amd::Os::getEnvironment("TEMP");
  if (tempFolder.empty()) {
    tempFolder = amd::Os::getEnvironment("TMP");
  }

  if (tempFolder.empty()) {
    tempFolder = "/tmp";
    ;
  }
  return tempFolder;
}

std::string Os::getTempFileName() {
  static std::atomic_size_t counter(0);

  std::string tempPath = getTempPath();
  std::stringstream tempFileName;

  tempFileName << tempPath << "/OCL" << ::getpid() << 'T' << counter++;
  return tempFileName.str();
}

int Os::unlink(const std::string& path) { return ::unlink(path.c_str()); }
int Os::shm_unlink(const std::string& path) { return ::shm_unlink(path.c_str()); }

#if defined(ATI_ARCH_X86)
void Os::cpuid(int regs[4], int info) {
#ifdef _LP64
  __asm__ __volatile__(
      "movq %%rbx, %%rsi;"
      "cpuid;"
      "xchgq %%rbx, %%rsi;"
      : "=a"(regs[0]), "=S"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
      : "a"(info));
#else
  __asm__ __volatile__(
      "movl %%ebx, %%esi;"
      "cpuid;"
      "xchgl %%ebx, %%esi;"
      : "=a"(regs[0]), "=S"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
      : "a"(info));
#endif
}

uint64_t Os::xgetbv(uint32_t ecx) {
  uint32_t eax, edx;

  __asm__ __volatile__(".byte 0x0f,0x01,0xd0"  // in case assembler doesn't recognize xgetbv
                       : "=a"(eax), "=d"(edx)
                       : "c"(ecx));

  return ((uint64_t)edx << 32) | (uint64_t)eax;
}
#endif  // ATI_ARCH_X86

uint64_t Os::offsetToEpochNanos() {
  static uint64_t offset = 0;

  if (offset != 0) {
    return offset;
  }

  struct timeval now;
  if (::gettimeofday(&now, NULL) != 0) {
    return 0;
  }

  offset = (now.tv_sec * UINT64_C(1000000) + now.tv_usec) * UINT64_C(1000) - timeNanos();

  return offset;
}

void Os::setCurrentStackPtr(address sp) {
  sp -= sizeof(void*);
  *(void**)sp = __builtin_return_address(0);

#if defined(ATI_ARCH_X86)
  __asm__ __volatile__(
#if !defined(OMIT_FRAME_POINTER)
      LP64_SWITCH("movl (%%ebp),%%ebp;", "movq (%%rbp),%%rbp;")
#endif  // !OMIT_FRAME_POINTER
          LP64_SWITCH("movl %0,%%esp; ret;", "movq %0,%%rsp; ret;")::"r"(sp));
#else
  assert(!"Unimplemented");
#endif
}

size_t Os::getPhysicalMemSize() {
  struct ::sysinfo si;

  if (::sysinfo(&si) != 0) {
    return 0;
  }

  if (si.mem_unit == 0) {
    // Linux kernels prior to 2.3.23 return sizes in bytes.
    si.mem_unit = 1;
  }

  return (size_t)si.totalram * si.mem_unit;
}

void Os::getAppPathAndFileName(std::string& appName, std::string& appPathAndName) {
  std::unique_ptr<char[]> buff(new char[FILE_PATH_MAX_LENGTH]());

  if (readlink("/proc/self/exe", buff.get(), FILE_PATH_MAX_LENGTH) > 0) {
    // Get filename without path and extension.
    appName = std::string(basename(buff.get()));
    appPathAndName = std::string(buff.get());
  } else {
    appName = "";
    appPathAndName = "";
  }
  return;
}

bool Os::GetURIFromMemory(const void* image, size_t image_size, std::string& uri) {
  pid_t pid = getpid();
  std::ostringstream uri_stream;
  // Create a unique resource indicator to the memory address
  uri_stream << "memory://" << pid << "#offset=0x" << std::hex << (uintptr_t)image << std::dec
             << "&size=" << image_size;
  uri = uri_stream.str();
  return true;
}

bool Os::CloseFileHandle(FileDesc fdesc) {
  // Return false if close system call fails
  if (close(fdesc) < 0) {
    return false;
  }

  return true;
}

bool Os::GetFileHandle(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr) {
  if ((fd_ptr == nullptr) || (sz_ptr == nullptr)) {
    return false;
  }

  // open system function call, return false on fail
  struct stat stat_buf;
  *fd_ptr = open(fname, O_RDONLY);
  if (*fd_ptr < 0) {
    return false;
  }

  // Retrieve stat info and size
  if (fstat(*fd_ptr, &stat_buf) != 0) {
    close(*fd_ptr);
    return false;
  }

  *sz_ptr = stat_buf.st_size;

  return true;
}

bool amd::Os::FindFileNameFromAddress(const void* image, std::string* fname_ptr,
                                      size_t* foffset_ptr) {
  // Get the list of mapped file list
  bool ret_value = false;
  std::ifstream proc_maps;
  proc_maps.open("/proc/self/maps", std::ifstream::in);
  if (!proc_maps.is_open() || !proc_maps.good()) {
    return ret_value;
  }

  // For every line on the list map find out low, high address
  std::string line;
  while (std::getline(proc_maps, line)) {
    char dash;
    std::stringstream tokens(line);
    uintptr_t low_address, high_address;
    tokens >> std::hex >> low_address >> std::dec >> dash >> std::hex >> high_address >> std::dec;
    if (dash != '-') {
      continue;
    }

    // If address is > low_address and < high_address, then this
    // is the mapped file. Get the URI path and offset.
    uintptr_t address = reinterpret_cast<uintptr_t>(image);
    if ((address >= low_address) && (address < high_address)) {
      std::string permissions, device, uri_file_path;
      size_t offset;
      uint64_t inode;
      tokens >> permissions >> std::hex >> offset >> std::dec >> device >> inode;
      std::getline(tokens >> std::ws, uri_file_path);

      if (inode == 0 || uri_file_path.empty()) {
        return ret_value;
      }

      *fname_ptr = uri_file_path;
      *foffset_ptr = offset + address - low_address;
      ret_value = true;
      break;
    }
  }

  return ret_value;
}

bool Os::MemoryMapFileDesc(FileDesc fdesc, size_t fsize, size_t foffset, const void** mmap_ptr) {
  if (fdesc <= 0) {
    return false;
  }

  // If the offset is not aligned then align it
  // and recalculate the new size
  if (foffset > 0) {
    size_t old_foffset = foffset;
    foffset = alignUp(foffset, pageSize());
    fsize += (foffset - old_foffset);
  }

  *mmap_ptr = mmap(NULL, fsize, PROT_READ, MAP_SHARED, fdesc, foffset);
  return (*mmap_ptr == MAP_FAILED) ? false : true;
}

bool Os::MemoryUnmapFile(const void* mmap_ptr, size_t mmap_size) {
  if (munmap(const_cast<void*>(mmap_ptr), mmap_size) != 0) {
    return false;
  }

  return true;
}

bool Os::MemoryMapFile(const char* fname, const void** mmap_ptr, size_t* mmap_size) {
  if ((mmap_ptr == nullptr) || (mmap_size == nullptr)) {
    return false;
  }

  struct stat stat_buf;
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    return false;
  }

  if (fstat(fd, &stat_buf) != 0) {
    close(fd);
    return false;
  }

  *mmap_size = stat_buf.st_size;
  *mmap_ptr = mmap(NULL, stat_buf.st_size, PROT_READ, MAP_SHARED, fd, 0);

  close(fd);

  if (*mmap_ptr == MAP_FAILED) {
    return false;
  }

  return true;
}

bool Os::MemoryMapFileTruncated(const char* fname, const void** mmap_ptr, size_t mmap_size) {
  if (mmap_ptr == nullptr) {
    return false;
  }

  struct stat stat_buf;
  int fd = shm_open(fname, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
  if (fd < 0) {
    return false;
  }

  if (ftruncate(fd, mmap_size) != 0) {
    return false;
  }
  *mmap_ptr = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  close(fd);

  if (*mmap_ptr == MAP_FAILED) {
    return false;
  }

  return true;
}

int Os::getProcessId() { return ::getpid(); }

// ================================================================================================
void* Os::CreateIpcMemory(const char* fname, size_t size, FileDesc* desc) {
  *desc = shm_open(fname, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
  if (*desc < 0) {
    return nullptr;
  }

  int status = ftruncate(*desc, size);
  if (status != 0) {
    return nullptr;
  }

  auto addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *desc, 0);
  return addr;
}

// ================================================================================================
void* Os::OpenIpcMemory(const char* fname, const FileDesc desc, size_t size) {
  FileDesc handle = desc;
  if (fname != nullptr) {
    handle = shm_open(fname, O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
  }

  if (handle < 0) {
    return nullptr;
  }

  auto addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, handle, 0);
  return addr;
}

// ================================================================================================
void Os::CloseIpcMemory(const FileDesc desc, const void* ptr, size_t size) {
  if (ptr != nullptr) {
    munmap(const_cast<void*>(ptr), size);
  }
  if (desc != 0) {
    close(desc);
  }
}

// ================================================================================================
void Os::PrintLibraryLocation() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void*>(Os::loadLibrary), &dl_info) && dl_info.dli_fname) {
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HIP Library Path: %s", dl_info.dli_fname);
  } else {
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HIP Library Path: <unknown>");
  }
}

// ================================================================================================
bool Os::DumpCoreFile() {
  // Execute the default handler if a GPU core file should be generated ...
  struct rlimit rlimit;
  return (getrlimit(RLIMIT_CORE, &rlimit) == 0 && rlimit.rlim_cur != 0);
}

// ================================================================================================
void Os::CxaDemangle(const std::string& name, std::string* result) {
  int status = 0;
  char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
  *result = (status == 0 && demangled != nullptr) ? demangled : name;
  free(demangled);
}

}  // namespace amd

#endif  // !defined(_WIN32) && !defined(__CYGWIN__)
