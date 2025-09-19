/* Copyright (c) 2008 - 2022 Advanced Micro Devices, Inc.

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

#if defined(_WIN32) || defined(__CYGWIN__)

#include "os/os.hpp"
#include "thread/thread.hpp"
#include "utils/flags.hpp"
#include <windows.h>
#include <process.h>
#include <tchar.h>
#include <time.h>
#include <intrin.h>

#include <atomic>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <algorithm>

#ifndef WINAPI
#define WINAPI
#endif


BOOL(WINAPI* pfnGetNumaNodeProcessorMaskEx)(USHORT, PGROUP_AFFINITY) = NULL;

namespace amd {

static size_t allocationGranularity_;

static LONG WINAPI divExceptionFilter(struct _EXCEPTION_POINTERS* ep);

#ifdef _WIN64
PVOID divExceptionHandler = NULL;
#endif  // _WIN64

static double PerformanceFrequency;

typedef BOOL(WINAPI* SetThreadGroupAffinity_fn)(__in HANDLE, __in CONST GROUP_AFFINITY*,
                                                __out_opt PGROUP_AFFINITY);
static SetThreadGroupAffinity_fn pfnSetThreadGroupAffinity = NULL;

#pragma section(".CRT$XCU", long, read)
__declspec(allocate(".CRT$XCU")) bool (*__init)(void) = Os::init;

bool Os::init() {
  static bool initialized_ = false;

  // We could use InitOnceExecuteOnce here:
  if (initialized_) {
    return true;
  }
  initialized_ = true;

  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  pageSize_ = si.dwPageSize;
  allocationGranularity_ = (size_t)si.dwAllocationGranularity;
  processorCount_ = si.dwNumberOfProcessors;

  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  PerformanceFrequency = (double)frequency.QuadPart;

  HMODULE handle = ::LoadLibrary("kernel32.dll");
  if (handle != NULL) {
    pfnSetThreadGroupAffinity =
        (SetThreadGroupAffinity_fn)::GetProcAddress(handle, "SetThreadGroupAffinity");
    pfnGetNumaNodeProcessorMaskEx = (BOOL(WINAPI*)(USHORT, PGROUP_AFFINITY))::GetProcAddress(
        handle, "GetNumaNodeProcessorMaskEx");
  }

  return Thread::init();
}

#pragma section(".CRT$XTU", long, read)
__declspec(allocate(".CRT$XTU")) void (*__exit)(void) = Os::tearDown;

void Os::tearDown() { Thread::tearDown(); }

void* Os::loadLibrary_(const char* filename) {
  if (filename != NULL) {
    HMODULE hModule = ::LoadLibrary(filename);
    return hModule;
  }
  return NULL;
}

void Os::unloadLibrary(void* handle) { ::FreeLibrary((HMODULE)handle); }

void* Os::getSymbol(void* handle, const char* name) {
  return ::GetProcAddress((HMODULE)handle, name);
}

static inline int memProtToOsProt(Os::MemProt prot) {
  switch (prot) {
    case Os::MEM_PROT_NONE:
      return PAGE_NOACCESS;
    case Os::MEM_PROT_READ:
      return PAGE_READONLY;
    case Os::MEM_PROT_RW:
      return PAGE_READWRITE;
    case Os::MEM_PROT_RWX:
      return PAGE_EXECUTE_READWRITE;
    default:
      break;
  }
  ShouldNotReachHere();
  return -1;
}

address Os::reserveMemory(address start, size_t size, size_t alignment, MemProt prot) {
  size = alignUp(size, pageSize());
  alignment = std::max(allocationGranularity_, alignUp(alignment, allocationGranularity_));
  assert(isPowerOfTwo(alignment) && "not a power of 2");

  size_t requested = size + alignment - allocationGranularity_;
  address mem, aligned;
  do {
    mem = (address)VirtualAlloc(start, requested, MEM_RESERVE, memProtToOsProt(prot));

    // check for out of memory.
    if (mem == NULL) return NULL;

    aligned = alignUp(mem, alignment);

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

bool Os::releaseMemory(void* addr, size_t size) { return VirtualFree(addr, 0, MEM_RELEASE) != 0; }

bool Os::commitMemory(void* addr, size_t size, MemProt prot) {
  return VirtualAlloc(addr, size, MEM_COMMIT, memProtToOsProt(prot)) != NULL;
}

bool Os::uncommitMemory(void* addr, size_t size) {
  return VirtualFree(addr, size, MEM_DECOMMIT) != 0;
}

bool Os::protectMemory(void* addr, size_t size, MemProt prot) {
  DWORD OldProtect;
  return VirtualProtect(addr, size, memProtToOsProt(prot), &OldProtect) != 0;
}


uint64_t Os::hostTotalPhysicalMemory() {
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

void* Os::alignedMalloc(size_t size, size_t alignment) {
  return ::_aligned_malloc(size, alignment);
}

void Os::alignedFree(void* mem) { ::_aligned_free(mem); }


void Os::currentStackInfo(address* base, size_t* size) {
  MEMORY_BASIC_INFORMATION mbInfo;

  address currentStackPage = (address)alignDown((intptr_t)currentStackPtr(), pageSize());

  ::VirtualQuery(currentStackPage, &mbInfo, sizeof(mbInfo));

  address stackBottom = (address)mbInfo.AllocationBase;
  size_t stackSize = 0;

  do {
    stackSize += mbInfo.RegionSize;
    ::VirtualQuery(stackBottom + stackSize, &mbInfo, sizeof(mbInfo));
  } while (stackBottom == (address)mbInfo.AllocationBase);

  *base = stackBottom + stackSize;
  *size = stackSize;

  assert(currentStackPtr() >= *base - *size && currentStackPtr() < *base && "just checking");
}

#define MS_VC_EXCEPTION 0x406D1388
#pragma pack(push, 8)
struct THREADNAME_INFO {
  DWORD dwType;      // Must be 0x1000.
  LPCSTR szName;     // Pointer to name (in user addr space).
  DWORD dwThreadID;  // Thread ID (-1=caller thread).
  DWORD dwFlags;     // Reserved for future use, must be zero.
};
#pragma pack(pop)

static void SetThreadName(DWORD threadId, const char* name) {
  if (name == NULL || *name == '\0') {
    return;
  }

  THREADNAME_INFO info;
  info.dwType = 0x1000;
  info.szName = name;
  info.dwThreadID = threadId;
  info.dwFlags = 0;

  __try {
    ::RaiseException(0x406D1388, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
  }
}

void Os::setCurrentThreadName(const char* name) { SetThreadName(GetCurrentThreadId(), name); }

void Os::setPreferredNumaNode(uint32_t node) {};

static LONG WINAPI divExceptionFilter(struct _EXCEPTION_POINTERS* ep) {
  DWORD code = ep->ExceptionRecord->ExceptionCode;

  if ((code == EXCEPTION_INT_DIVIDE_BY_ZERO || code == EXCEPTION_INT_OVERFLOW) &&
      Thread::current()->isWorkerThread()) {
    address insn = (address)ep->ContextRecord->LP64_SWITCH(Eip, Rip);

    if (Os::skipIDIV(insn)) {
      ep->ContextRecord->LP64_SWITCH(Eip, Rip) = (uintptr_t)insn;
      return EXCEPTION_CONTINUE_EXECUTION;
    }
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

bool Os::installSigfpeHandler() {
#ifdef _WIN64
  divExceptionHandler = AddVectoredExceptionHandler(1, divExceptionFilter);
#endif  // _WIN64
  return true;
}

void Os::uninstallSigfpeHandler() {
#ifdef _WIN64
  if (divExceptionHandler != NULL) {
    RemoveVectoredExceptionHandler(divExceptionHandler);
    divExceptionHandler = NULL;
  }
#endif  // _WIN64
}

void* Thread::entry(Thread* thread) {
  void* ret = NULL;
#if !defined(_WIN64)
  __try {
    ret = thread->main();
  } __except (divExceptionFilter(GetExceptionInformation())) {
    // nothing to do here.
  }
#else   // _WIN64
  ret = thread->main();
#endif  // _WIN64

// The current thread exits, thus clear the pointer
#if defined(USE_DECLSPEC_THREAD)
  details::thread_ = NULL;
#else   // !USE_DECLSPEC_THREAD
  TlsSetValue(details::threadIndex_, NULL);
#endif  // !USE_DECLSPEC_THREAD
  return ret;
}

bool Os::isThreadAlive(const Thread& thread) {
  HANDLE handle = (HANDLE)(thread.handle());

  DWORD exitCode = 0;
  if (GetExitCodeThread(handle, &exitCode)) {
    return exitCode == STILL_ACTIVE;
  } else {
    // Could not get thread's exitcode
    return false;
  }
}

const void* Os::createOsThread(Thread* thread) {
  HANDLE handle = ::CreateThread(NULL, thread->stackSize_, (LPTHREAD_START_ROUTINE)Thread::entry,
                                 thread, 0, NULL);
  if (handle == NULL) {
    thread->setState(Thread::FAILED);
  }
  return reinterpret_cast<const void*>(handle);
}

void Os::setThreadAffinity(const void* handle, const Os::ThreadAffinityMask& mask) {
  if (pfnSetThreadGroupAffinity != NULL) {
    GROUP_AFFINITY group = {0};
    for (WORD i = 0; i < sizeof(mask.mask_) / sizeof(KAFFINITY); ++i) {
      group.Mask = mask.mask_[i];
      group.Group = i;
      if (group.Mask != 0) {
        pfnSetThreadGroupAffinity((HANDLE)handle, &group, NULL);
      }
    }
  } else {  // pfnSetThreadGroupAffinity == NULL
    DWORD_PTR threadAffinityMask = (DWORD_PTR)mask.mask_[0];
    if (threadAffinityMask != 0) {
      ::SetThreadAffinityMask((HANDLE)handle, threadAffinityMask);
    }
  }
}

bool Os::setThreadAffinityToMainThread() { return true; }
void Os::yield() { ::SwitchToThread(); }

uint64_t Os::timeNanos() {
  LARGE_INTEGER current;
  QueryPerformanceCounter(&current);
  return (uint64_t)((double)current.QuadPart / PerformanceFrequency * 1e9);
}

uint64_t Os::timerResolutionNanos() { return (uint64_t)(1e9 / PerformanceFrequency); }


const char* Os::libraryExtension() { return ".DLL"; }

const char* Os::libraryPrefix() { return NULL; }

const char* Os::objectExtension() { return ".OBJ"; }

char Os::fileSeparator() { return '\\'; }

char Os::pathSeparator() { return ';'; }

bool Os::pathExists(const std::string& path) {
  return GetFileAttributes(path.c_str()) != INVALID_FILE_ATTRIBUTES;
}

bool Os::createPath(const std::string& path) {
  size_t pos = 0;
  while (true) {
    pos = path.find(fileSeparator(), pos);
    const std::string currPath = path.substr(0, pos);
    if (!currPath.empty() && !pathExists(currPath)) {
      if (!CreateDirectory(currPath.c_str(), NULL)) return false;
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
      if (!RemoveDirectory(currPath.c_str())) return removed;
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
  DWORD dwBytesWritten;

  va_start(ap, fmt);
  int len = ::_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (len <= 0) return len;

  va_start(ap, fmt);
  char* str = static_cast<char*>(alloca(len + 1));
  len = ::_vsnprintf(str, len + 1, fmt, ap);
  va_end(ap);
  if (len <= 0) return len;

  ::WriteFile(::GetStdHandle(STD_OUTPUT_HANDLE), str, len, &dwBytesWritten, NULL);

  return len;
}

int Os::systemCall(const std::string& command) {
#if 1
  char* cmd = new char[command.size() + 1];
  std::memcpy(cmd, command.c_str(), command.size());
  cmd[command.size()] = 0;

  STARTUPINFO si = {0};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi;

  if (::CreateProcess(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi) == 0) {
    delete[] cmd;
    return -1;  // failed
  };

  // Wait until child process exits.
  ::WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD ExitCode = 0;
  ::GetExitCodeProcess(pi.hProcess, &ExitCode);

  // Close process and thread handles.
  ::CloseHandle(pi.hProcess);
  ::CloseHandle(pi.hThread);

  delete[] cmd;
  return (int)ExitCode;
#else
  std::stringstream str;
  str << "\"" << command << "\"";
  return ::system(str.str().c_str());
#endif
}

std::string Os::getEnvironment(const std::string& name) {
  char dstBuf[MAX_PATH];
  size_t dstSize;

  if (::getenv_s(&dstSize, dstBuf, MAX_PATH, name.c_str())) {
    return std::string("");
  }
  return std::string(dstBuf);
}

std::string Os::getTempPath() {
  char tempPath[MAX_PATH];
  uint ret = GetTempPath(MAX_PATH, tempPath);
  if (ret == 0 || (ret == 1 && tempPath[0] == '?')) {
    return std::string(".");
  }

  // If the app was started from an UNC path instead of a DOS path,
  // the temp env var won't be set correctly and will point to windows
  // system directory instead (usually c:/windows/temp), which will be
  // blocked. So we check if the temp path returned by GetTempPath is
  // under windows directory, use . instead
  std::string tempPathStr(tempPath);
  char winPath[MAX_PATH];
  if (GetWindowsDirectory(winPath, MAX_PATH) > 0) {
    // Need to check if tempPath is C:\Windows or C:\Windows\ //
    if (tempPath[strlen(tempPath) - 1] == '\\') {
      tempPath[strlen(tempPath) - 1] = '\0';
      ret--;
    }
    if (_memicmp(tempPath, winPath, ret) == 0) {
      return std::string(".");
    }
  }
  return tempPathStr;
}


std::string Os::getTempFileName() {
  static std::atomic_size_t counter(0);

  std::string tempPath = getTempPath();
  std::stringstream tempFileName;

  tempFileName << tempPath << "\\OCL" << ::_getpid() << 'T' << counter++;
  return tempFileName.str();
}

int Os::unlink(const std::string& path) { return ::_unlink(path.c_str()); }

// shm_unlink is a nop on windows
int Os::shm_unlink(const std::string& path) { return 0; }

void Os::cpuid(int regs[4], int info) { return __cpuid(regs, info); }

uint64_t Os::xgetbv(uint32_t ecx) { return (uint64_t)_xgetbv(ecx); }

uint64_t Os::offsetToEpochNanos() {
  static uint64_t offset = 0;

  if (offset != 0) {
    return offset;
  }

  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);

  LARGE_INTEGER li;
  li.LowPart = ft.dwLowDateTime;
  li.HighPart = ft.dwHighDateTime;

  uint64_t now = (li.QuadPart - 116444736000000000ull) * 100;
  offset = now - timeNanos();

  return offset;
}

#ifdef _WIN64

address Os::currentStackPtr() { return (address)_AddressOfReturnAddress() + sizeof(void*); }

#else  // !_WIN64

#pragma warning(disable : 4731)

void __stdcall Os::setCurrentStackPtr(address newSp) {
  newSp -= sizeof(void*);
  *(void**)newSp = *(void**)_AddressOfReturnAddress();
  __asm {
        mov esp,newSp
        mov ebp,[ebp]
        ret
  }
}

#endif  // !_WIN64

size_t Os::getPhysicalMemSize() {
  MEMORYSTATUSEX statex;

  statex.dwLength = sizeof(statex);

  if (GlobalMemoryStatusEx(&statex) == 0) {
    return 0;
  }

  return (size_t)statex.ullTotalPhys;
}

void Os::getAppPathAndFileName(std::string& appName, std::string& appPathAndName) {
  char* buff = new char[FILE_PATH_MAX_LENGTH];

  if (GetModuleFileNameA(NULL, buff, FILE_PATH_MAX_LENGTH) != 0) {
    // Get filename without path and extension.
    appPathAndName = buff;
    appName = strrchr(buff, '\\') ? strrchr(buff, '\\') + 1 : buff;
  } else {
    appPathAndName = "";
    appName = "";
  }

  delete[] buff;
  return;
}

bool Os::GetURIFromMemory(const void* image, size_t image_size, std::string& uri_) {
  // Not implemented yet for windows
  uri_ = std::string();
  return true;
}

bool Os::CloseFileHandle(FileDesc fdesc) {
  // return false on failure
  if (CloseHandle(fdesc) < 0) {
    return false;
  }
  return true;
}

bool Os::GetFileHandle(const char* fname, FileDesc* fd_ptr, size_t* sz_ptr) {
  if ((fd_ptr == nullptr) || (sz_ptr == nullptr)) {
    return false;
  }

  *fd_ptr = INVALID_HANDLE_VALUE;
  *fd_ptr =
      CreateFileA(fname, GENERIC_READ, 0x1, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL);
  if (*fd_ptr == INVALID_HANDLE_VALUE) {
    return false;
  }

  *sz_ptr = GetFileSize(*fd_ptr, NULL);
  return true;
}

bool Os::MemoryMapFileDesc(FileDesc fdesc, size_t fsize, size_t foffset, const void** mmap_ptr) {
  if (fdesc == INVALID_HANDLE_VALUE) {
    return false;
  }

  HANDLE map_handle = CreateFileMappingA(fdesc, NULL, PAGE_READONLY, 0, 0, NULL);
  if (map_handle == INVALID_HANDLE_VALUE) {
    CloseHandle(map_handle);
    return false;
  }

  *mmap_ptr = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);

  return (*mmap_ptr == NULL) ? false : true;
}

bool Os::MemoryUnmapFile(const void* mmap_ptr, size_t mmap_size) {
  if (!UnmapViewOfFile(mmap_ptr)) {
    return false;
  }

  return true;
}

bool Os::MemoryMapFile(const char* fname, const void** mmap_ptr, size_t* mmap_size) {
  if ((mmap_ptr == nullptr) || (mmap_size == nullptr)) {
    return false;
  }

  HANDLE file_handle =
      CreateFileA(fname, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL);
  if (file_handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  HANDLE map_handle = CreateFileMappingA(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
  if (map_handle == INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle);
    return false;
  }

  *mmap_size = GetFileSize(file_handle, NULL);
  *mmap_ptr = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);

  CloseHandle(file_handle);
  CloseHandle(map_handle);

  if (*mmap_ptr == nullptr) {
    return false;
  }

  return true;
}

bool Os::MemoryMapFileTruncated(const char* fname, const void** mmap_ptr, size_t mmap_size) {
  if (mmap_ptr == nullptr) {
    return false;
  }

  HANDLE map_handle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, fname);

  if (map_handle == nullptr) {
    map_handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, mmap_size, fname);
    if (map_handle == nullptr) {
      return false;
    }
  }

  *mmap_ptr = MapViewOfFile(map_handle, FILE_MAP_ALL_ACCESS, 0, 0, mmap_size);

  CloseHandle(map_handle);

  if (*mmap_ptr == nullptr) {
    return false;
  }

  return true;
}

bool Os::FindFileNameFromAddress(const void* image, std::string* fname_ptr, size_t* foffset_ptr) {
  // TODO: Implementation on windows side pending.
  return false;
}

int Os::getProcessId() { return ::_getpid(); }

// ================================================================================================
void* Os::CreateIpcMemory(const char* fname, size_t size, FileDesc* desc) {
  void* addr = nullptr;
  *desc = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, static_cast<DWORD>(size),
                            fname);
  if (*desc != 0) {
    addr = MapViewOfFile(*desc, FILE_MAP_ALL_ACCESS, 0, 0, size);
  }

  return addr;
}

// ================================================================================================
void* Os::OpenIpcMemory(const char* fname, const FileDesc desc, size_t size) {
  void* addr = nullptr;
  FileDesc handle = desc;
  if (fname != nullptr) {
    handle = CreateFileMapping(desc, NULL, PAGE_READWRITE, 0, static_cast<DWORD>(size), fname);
  }
  if (handle != 0) {
    addr = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
  }

  return addr;
}

// ================================================================================================
void Os::CloseIpcMemory(const FileDesc desc, const void* ptr, size_t size) {
  if (ptr != nullptr) {
    UnmapViewOfFile(ptr);
  }
  if (desc != nullptr) {
    CloseHandle(desc);
  }
}

// ================================================================================================
void Os::PrintLibraryLocation() {
  HMODULE hm = NULL;
  if (GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)&Os::loadLibrary, &hm)) {
    char cszDllPath[1024] = {0};
    if (GetModuleFileNameA(hm, cszDllPath, sizeof(cszDllPath))) {
      ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HIP Library Path: %s", cszDllPath);
      return;
    }
  }
  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HIP Library Path: <unknown>");
}

// ================================================================================================
bool Os::DumpCoreFile() { return false; }

// ================================================================================================
void Os::CxaDemangle(const std::string& name, std::string* result) { *result = name; }

}  // namespace amd

#endif  // _WIN32 || __CYGWIN__
