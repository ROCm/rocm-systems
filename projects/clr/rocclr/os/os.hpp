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
#include "utils/flags.hpp"

#include <vector>
#include <string>

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
  static address currentStackPtr();

  //! Touches all stack pages between [bottom,top[
  static void touchStackPages(address bottom, address top);

  // Thread routines:
  //

  //! Create a native thread and link it to the given OsThread.
  static const void* createOsThread(Thread* osThread);
  //! Set the currently running thread's name.
  static void setCurrentThreadName(const char* name);

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
  inline static void setPreferredNumaNode(uint32_t node);

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

/* Mini numa interface instead of numa lib apis */
namespace numa {

static constexpr uint32_t kBitsPerUInt64 = 8 * sizeof(uint64_t);

/*! \brief Manage Numa policy.
 *
 *  \note Works in Linux only, dummy in Windows.
 */
class NumaPolicy final {
public:
  enum class Policy {
    kDefault = 0,
    kPrefered = 1,
    kBind = 2,
    kInterleave = 3,
    kLocal = 4,
    kPreferedMany = 5,
    kWeightedInterleave = 6,
    kMax = 7
  };
  NumaPolicy(uint32_t numa_node_count);

  //! Query memory policy and node bitmask from Linux kernel
  bool GetMemPolicy();

  //! Check whether node_index is in bitmask for kPrefered and kBind modes
  bool IsPolicySetAt(uint32_t node_index) const;

  //! Return the queried policy
  Policy GetPolicy() const { return policy_; }
private:
  std::vector<uint64_t> node_map_{};  //!< Node bitmask for kPrefered and kBind modes
  Policy policy_{Policy::kDefault};  //!< The policy
};

/*! \brief Manage Numa node.
 *
 *  \note Works in Linux and Windows.
 */
class NumaNode final {
public:
  NumaNode (uint32_t node_index): node_index_(node_index) {}
  ~NumaNode();
  //! Apply the CPU affinity mask of the node onto the current thread
  bool SchedSetAffinity();
private:
  uint32_t node_index_; //! Index of the Numa node
  void* affinity_ = nullptr;  //!< Affinity mask of logical CPUs on this node
  uint32_t size_ = 0;  //!< Number of valid bits
  //! Guery the affinity mask of logical CPUs on this node
  bool GetAffinity();
};

}  // namespace numa

inline void Os::setPreferredNumaNode(uint32_t node) {
  if (AMD_CPU_AFFINITY) {
    numa::NumaNode numaNode(node);
    numaNode.SchedSetAffinity();
  }
}

}  // namespace amd

#endif /*OS_HPP_*/
