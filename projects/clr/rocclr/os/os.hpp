/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#ifndef OS_HPP_
#define OS_HPP_

#include "top.hpp"
#include "utils/util.hpp"

#include <vector>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

#ifdef _WIN32
#include <Basetsd.h>  // For KAFFINITY
#endif                // _WIN32

// Smallest supported VM page size.
#define MIN_PAGE_SHIFT 12
#define MIN_PAGE_SIZE (1UL << MIN_PAGE_SHIFT)

namespace amd {

/*! \addtogroup Os Operating System Abstraction
 *
 *  \copydoc amd::Os
 *
 *  @{
 */

class Thread;  // For Os::createOsThread()

class Os : AllStatic {
 public:
// File Desc abstraction between OS
#if defined(_WIN32)
  typedef void* FileDesc;
#else
  typedef int FileDesc;
#endif

  enum MemProt { MEM_PROT_NONE = 0, MEM_PROT_READ, MEM_PROT_RW, MEM_PROT_RWX };

  class ThreadAffinityMask {
    friend class Os;

   private:
#if defined(__linux__)
    cpu_set_t mask_;
#else  // _WIN32
#if !defined(_WIN32)
    typedef uint KAFFINITY;
#endif
    KAFFINITY mask_[512 / sizeof(KAFFINITY)];
#endif

   public:
    ThreadAffinityMask() { init(); }

    inline void init();
    inline void set(uint cpu);
    inline void clear(uint cpu);
    inline bool isSet(uint cpu) const;
    inline bool isEmpty() const;
    inline uint countSet() const;

    inline uint getFirstSet() const;
    inline uint getNextSet(uint cpu) const;

#if defined(__linux__)
    inline void set(const cpu_set_t& mask);
    inline void clear(const cpu_set_t& mask);
    inline void adjust(cpu_set_t& mask) const;
    inline cpu_set_t& getNative() { return mask_; }
#else
    inline void set(size_t group, KAFFINITY affinity);
    inline void adjust(size_t group, KAFFINITY& affinity) const;
#endif
  };

  static FileDesc FDescInit() {
#if defined(__linux__)
    return -1;
#else
    return reinterpret_cast<void*>(-1);
#endif
  }

  // Returns unique resource indicator for a particular memory
  static bool GetURIFromMemory(const void* image, size_t image_size, std::string& uri);

  // Closes the file Handle
  static bool CloseFileHandle(FileDesc fdesc);
  // Given a valid file name, returns file descriptor and file size
  static bool GetFileHandle(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr);

  // Returns the file name & file offset of mapped memory if the file is mapped.
  static bool FindFileNameFromAddress(const void* image, std::string* fname_ptr,
                                      size_t* foffset_ptr);

  // Given a valid file descriptor returns mmaped memory for size and offset
  static bool MemoryMapFileDesc(FileDesc fdesc, size_t fsize, size_t foffset,
                                const void** mmap_ptr);
  // Given a valid file name, returns mmapped memory with the mapped size.
  static bool MemoryMapFile(const char* fname, const void** mmap_ptr, size_t* mmap_size);

  // Given a valid file name amd mapped size, returns ftruncated mmaped memory
  static bool MemoryMapFileTruncated(const char* fname, const void** mmap_ptr, size_t mmap_size);

  // Given a valid mmaped ptr with correct size, unmaps the ptr from memory
  static bool MemoryUnmapFile(const void* mmap_ptr, size_t mmap_size);

  // Given a valid filename create system memory that can be shared between processes
  static void* CreateIpcMemory(const char* fname, size_t size, FileDesc* desc);

  // Given a valid file descriptor open IPC memory
  static void* OpenIpcMemory(const char* fname, const FileDesc desc, size_t size);

  // Given a valid file descriptor close IPC memory
  static void CloseIpcMemory(const FileDesc desc, const void* ptr, size_t size);

 private:
  static constexpr size_t FILE_PATH_MAX_LENGTH = 1024;

  static size_t pageSize_;     //!< The default os page size.
  static int processorCount_;  //!< The number of active processors.

 private:
  //! Load the shared library named by \a filename
  static void* loadLibrary_(const char* filename);

 public:
  //! Initialize the Os package.
  static bool init();
  //! Tear down the Os package.
  static void tearDown();

  // Topology helper routines:
  //

  //! Return the number of active processors in the system.
  inline static int processorCount();

#if defined(ATI_ARCH_X86)
  //! Query the processor information about supported features and CPU type.
  static void cpuid(int regs[4], int info);
  //! Get value of extended control register
  static uint64_t xgetbv(uint32_t which);
#endif  // ATI_ARCH_X86

  // Stack helper routines:
  //

  //! Return the current stack base and size information.
  static void currentStackInfo(address* base, size_t* size);

  //! Return the value of the current stack pointer.
  static NOT_WIN64(inline) address currentStackPtr();
  //! Set the value of the current stack pointer.
  static WIN64_ONLY(inline) void WINDOWS_ONLY(__stdcall /*callee cleanup*/)
      setCurrentStackPtr(address sp);
  //! Touches all stack pages between [bottom,top[
  static void touchStackPages(address bottom, address top);

  // Thread routines:
  //

  //! Create a native thread and link it to the given OsThread.
  static const void* createOsThread(Thread* osThread);
  //! Set the thread's affinity to the given cpu ordinal.
  static void setThreadAffinity(const void* handle, unsigned int cpu);
  //! Set the thread's affinity to the given cpu mask.
  static void setThreadAffinity(const void* handle, const ThreadAffinityMask& mask);
  //! Set the currently running thread's name.
  static void setCurrentThreadName(const char* name);
  //! Set current threads affinity to that of main thread
  static bool setThreadAffinityToMainThread();

  //! Check if the thread is alive
  static bool isThreadAlive(const Thread& osThread);

  //! Sleep for n milli-seconds.
  static void sleep(long n);
  //! Yield to threads of the same or lower priority
  static void yield();
  //! Execute a pause instruction (for spin loops).
  static void spinPause();

  // Memory routines:
  //

  //! Return the default os page size.
  inline static size_t pageSize();
  //! Return the amount of host total physical memory in bytes.
  static uint64_t hostTotalPhysicalMemory();

  //! Reserve a chunk of memory (priv | anon | noreserve).
  static address reserveMemory(address start, size_t size, size_t alignment = 0,
                               MemProt prot = MEM_PROT_NONE);
  //! Release a chunk of memory reserved with reserveMemory.
  static bool releaseMemory(void* addr, size_t size);
  //! Commit a chunk of memory previously reserved with reserveMemory.
  static bool commitMemory(void* addr, size_t size, MemProt prot = MEM_PROT_NONE);
  //! Uncommit a chunk of memory previously committed with commitMemory.
  static bool uncommitMemory(void* addr, size_t size);
  //! Set the page protections for the given memory region.
  static bool protectMemory(void* addr, size_t size, MemProt prot);

  //! Allocate an aligned chunk of memory.
  static void* alignedMalloc(size_t size, size_t alignment);
  //! Deallocate an aligned chunk of memory.
  static void alignedFree(void* mem);

  //! NUMA related settings
  static void setPreferredNumaNode(uint32_t node);

  // File/Path helper routines:
  //

  //! Return the shared library extension string.
  static const char* libraryExtension();
  //! Return the shared library prefix string.
  static const char* libraryPrefix();
  //! Return the object extension string.
  static const char* objectExtension();
  //! Return the file separator char.
  static char fileSeparator();
  //! Return the path separator char.
  static char pathSeparator();
  //! Return whether the path exists
  static bool pathExists(const std::string& path);
  //! Create the path if it does not exist
  static bool createPath(const std::string& path);
  //! Remove the path if it is empty
  static bool removePath(const std::string& path);
  //! Printf re-implementation (due to MS CRT problem)
  static int printf(const char* fmt, ...);
  /*! \brief Invokes the command processor for the command execution
   *
   *  \result Returns the operation result
   */
  static int systemCall(const std::string& command);  //!< command for execution

  /*! \brief Retrieves a string containing the value
   *  of the environment variable
   *
   *  \result Returns the environment variable value
   */
  static std::string getEnvironment(const std::string& name);  //!< the environment variable's name

  /*! \brief Retrieves the path of the directory designated for temporary
   *  files
   *
   *  \result Returns the temporary path
   */
  static std::string getTempPath();

  /*! \brief Creates a name for a temporary file
   *
   *  \result Returns the name of temporary file
   */
  static std::string getTempFileName();

  //! Deletes file
  static int unlink(const std::string& path);

  //! Removes the shared memory object name
  static int shm_unlink(const std::string& path);

  // Library routines:
  //
  typedef bool (*SymbolCallback)(std::string, const void*, void*);

  //! checks if file descriptor is valid
  static bool isValidFileDesc(const amd::Os::FileDesc& desc);
  //! Load the shared library named by \a filename
  static void* loadLibrary(const char* filename);
  //! Unload the shared library.
  static void unloadLibrary(void* handle);
  //! Return the address of the function identified by \a name.
  static void* getSymbol(void* handle, const char* name);

  // Time routines:
  //

  //! Return the current system time counter in nanoseconds.
  static uint64_t timeNanos();
  //! Return the system timer's resolution in nanoseconds.
  static uint64_t timerResolutionNanos();
  //! Return the timeNanos starting point offset to Epoch.
  static uint64_t offsetToEpochNanos();

  // X86 Instructions helpers:
  //

  //! Skip an IDIV (F6/F7) instruction and return a pointer to the next insn.
  static bool skipIDIV(address& insn);

  // return gloabal memory size to be assigned to device info
  static size_t getPhysicalMemSize();

  //! get Application file name and path
  static void getAppPathAndFileName(std::string& appName, std::string& appPathAndName);

  //! Install SIGFPE handler for CPU device
  static bool installSigfpeHandler();

  //! Uninstall SIGFPE handler for CPU device
  static void uninstallSigfpeHandler();

  //! Return the current process id
  static int getProcessId();

  //! Prints the location of the currently loaded library (shared object or DLL)
  static void PrintLibraryLocation();

  //! Checks if a core dump must be generated (rocgdb detection). Returns false in Windows
  static bool DumpCoreFile();

  //! Demangle a C++ name. The function will return the same name if couldn't demangle
  static void CxaDemangle(const std::string& name, std::string* demangle);
};

/*@}*/

inline size_t Os::pageSize() {
  assert(pageSize_ != 0 && "runtime is not initialized");
  return pageSize_;
}

inline int Os::processorCount() { return processorCount_; }

#if defined(_WIN64)

extern "C" void _Os_setCurrentStackPtr(address sp);

ALWAYSINLINE void Os::setCurrentStackPtr(address sp) { _Os_setCurrentStackPtr(sp); }

#else  // !_WIN64

ALWAYSINLINE address Os::currentStackPtr() {
  intptr_t value;

#if defined(__GNUC__)
  __asm__ __volatile__(
#if defined(ATI_ARCH_X86)
      LP64_SWITCH("movl %%esp", "movq %%rsp") ",%0"
      : "=r"(value)
#elif defined(ATI_ARCH_ARM)
      "mov %0,sp"
      : "=r"(value)
#else
      ""
#endif
  );
#else   // !__GNUC__
  __asm mov value, esp;
#endif  // !__GNUC__

  return (address)value;
}

#endif  // !_WIN64


#if defined(__linux__)

inline void Os::ThreadAffinityMask::init() { CPU_ZERO(&mask_); }

inline void Os::ThreadAffinityMask::set(uint cpu) { CPU_SET(cpu, &mask_); }

inline void Os::ThreadAffinityMask::clear(uint cpu) { CPU_CLR(cpu, &mask_); }

inline bool Os::ThreadAffinityMask::isSet(uint cpu) const { return CPU_ISSET(cpu, &mask_); }

inline bool Os::ThreadAffinityMask::isEmpty() const {
  for (__cpu_mask bits : mask_.__bits) {
    if (bits != 0) {
      return false;
    }
  }
  return true;
}

inline void Os::ThreadAffinityMask::set(const cpu_set_t& mask) { mask_ = mask; }

inline void Os::ThreadAffinityMask::clear(const cpu_set_t& mask) {
  for (uint i = 0; i < sizeof(mask_.__bits) / sizeof(mask_.__bits[0]); ++i) {
    mask_.__bits[i] &= ~mask.__bits[i];
  }
}

inline void Os::ThreadAffinityMask::adjust(cpu_set_t& mask) const {
  for (uint i = 0; i < sizeof(mask_.__bits) / sizeof(mask_.__bits[0]); ++i) {
    mask.__bits[i] &= mask_.__bits[i];
  }
}

inline uint Os::ThreadAffinityMask::countSet() const {
  uint count = 0;
  for (__cpu_mask bits : mask_.__bits) {
    count += countBitsSet(bits);
  }
  return count;
}

inline uint Os::ThreadAffinityMask::getFirstSet() const {
  uint i = 0;
  for (__cpu_mask bits : mask_.__bits) {
    if (bits != 0) {
      return leastBitSet(bits) + (i * (8 * sizeof(__cpu_mask)));
    }

    ++i;
  }
  return (uint)-1;
}

inline uint Os::ThreadAffinityMask::getNextSet(uint cpu) const {
  const uint32_t* bits = (const uint32_t*)mask_.__bits;
  ++cpu;
  uint j = cpu % (8 * sizeof(uint32_t));
  for (uint i = cpu / (8 * sizeof(uint32_t)); i < sizeof(mask_.__bits) / (sizeof(uint32_t)); ++i) {
    if (bits[i] != 0) {
      for (; j < (8 * sizeof(uint32_t)); ++j) {
        if (0 != (bits[i] & ((uint32_t)1 << j))) {
          return i * (8 * sizeof(uint32_t)) + j;
        }
      }
    }
    j = 0;
  }
  return (uint)-1;
}

#else

inline void Os::ThreadAffinityMask::init() {
  for (uint i = 0; i < sizeof(mask_) / sizeof(KAFFINITY); ++i) {
    mask_[i] = (KAFFINITY)0;
  }
}

inline void Os::ThreadAffinityMask::set(uint cpu) {
  mask_[cpu / (8 * sizeof(KAFFINITY))] |= (KAFFINITY)1 << (cpu % (8 * sizeof(KAFFINITY)));
}

inline void Os::ThreadAffinityMask::clear(uint cpu) {
  mask_[cpu / (8 * sizeof(KAFFINITY))] &= ~((KAFFINITY)1 << (cpu % (8 * sizeof(KAFFINITY))));
}

inline bool Os::ThreadAffinityMask::isSet(uint cpu) const {
  return (KAFFINITY)0 !=
         (mask_[cpu / (8 * sizeof(KAFFINITY))] & ((KAFFINITY)1 << (cpu % (8 * sizeof(KAFFINITY)))));
}

inline bool Os::ThreadAffinityMask::isEmpty() const {
  for (uint i = 0; i < sizeof(mask_) / sizeof(KAFFINITY); ++i) {
    if (mask_[i] != (KAFFINITY)0) {
      return false;
    }
  }
  return true;
}

inline void Os::ThreadAffinityMask::set(size_t group, KAFFINITY affinity) {
  mask_[group] |= affinity;
}

inline void Os::ThreadAffinityMask::adjust(size_t group, KAFFINITY& affinity) const {
  affinity &= mask_[group];
}

inline uint Os::ThreadAffinityMask::countSet() const {
  uint count = 0;
  for (uint i = 0; i < sizeof(mask_) / sizeof(KAFFINITY); ++i) {
    count += countBitsSet(mask_[i]);
  }
  return count;
}

inline uint Os::ThreadAffinityMask::getFirstSet() const {
  for (uint i = 0; i < sizeof(mask_) / sizeof(KAFFINITY); ++i) {
    if (mask_[i] != 0) {
      return leastBitSet(mask_[i]) + (i * (8 * sizeof(KAFFINITY)));
    }
  }
  return (uint)-1;
}

inline uint Os::ThreadAffinityMask::getNextSet(uint cpu) const {
  ++cpu;
  uint j = cpu % (8 * sizeof(KAFFINITY));
  for (uint i = cpu / (8 * sizeof(KAFFINITY)); i < sizeof(mask_) / sizeof(KAFFINITY); ++i) {
    if (mask_[i] != 0) {
      for (; j < (8 * sizeof(KAFFINITY)); ++j) {
        if (0 != (mask_[i] & ((KAFFINITY)1 << j))) {
          return i * (8 * sizeof(KAFFINITY)) + j;
        }
      }
    }
    j = 0;
  }
  return (uint)-1;
}

#endif

}  // namespace amd

#endif /*OS_HPP_*/
