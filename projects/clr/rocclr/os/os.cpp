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

#include "os/os.hpp"
#include "thread/thread.hpp"

#include <string>
#include <cstring>

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else  // !_WIN32
#include <time.h>
#include <unistd.h>
#endif  // !_WIN32

#if defined(ATI_ARCH_X86)
#include <xmmintrin.h>  // for _mm_pause
#endif                  // ATI_ARCH_X86

namespace amd {

// ================================================================================================
bool Os::isValidFileDesc(const amd::Os::FileDesc& desc) {
#if IS_WINDOWS
  return desc != nullptr;
#else
  return desc > 0;
#endif
  return false;
}

// ================================================================================================
void* Os::loadLibrary(const char* libraryname) {
  void* handle;

  // Try with the system library prefix and extension instead.
  std::string str = libraryname;

  size_t namestart = str.rfind(fileSeparator());
  namestart = (namestart != std::string::npos) ? namestart + 1 : 0;

  if (namestart == 0) {
#if IS_WINDOWS
    // Try with the path of the current loaded dll(OCL runtime) first
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&loadLibrary, &hm))
      return NULL;

    char cszDllPath[1024] = {0};
    if (!GetModuleFileNameA(hm, cszDllPath, sizeof(cszDllPath))) return NULL;

    LPSTR cszFileName;
    char buffer[1024] = {0};
    if (!GetFullPathNameA(cszDllPath, sizeof(buffer), buffer, &cszFileName)) return NULL;

    std::string newPath;
    newPath = cszDllPath;
    newPath.replace(newPath.find(cszFileName), strlen(libraryname), libraryname);

    handle = Os::loadLibrary_(newPath.c_str());
    if (handle != NULL) {
      return handle;
    }
#endif
  }

  handle = Os::loadLibrary_(libraryname);
  if (handle != NULL) {
    return handle;
  }

  const char* prefix = Os::libraryPrefix();
  if (prefix != NULL && str.compare(namestart, strlen(prefix), prefix) == 0) {
    // It is alread present, not need to prepend it.
    prefix = NULL;
  }
  size_t dot = str.rfind('.');
  if (dot != std::string::npos) {
    // check that the dot was on the filename not a dir name.
    if (namestart < dot) {
      // strip the previous extension.
      str.resize(dot);
    }
  }
  if (prefix != NULL && prefix[0] != '\0') {
    str.insert(namestart, prefix);
  }
  str.append(Os::libraryExtension());

  handle = Os::loadLibrary_(str.c_str());
  if (handle != NULL || str.find(fileSeparator()) != std::string::npos) {
    return handle;
  }

  // Try to find the lib in the current directory.
  return Os::loadLibrary((std::string(".") + fileSeparator() + std::string(libraryname)).c_str());
}

size_t Os::pageSize_ = 0;

int Os::processorCount_ = 0;

void Os::spinPause() {
#if defined(ATI_ARCH_X86)
  _mm_pause();
#elif defined(ATI_ARCH_ARM)
  __asm__ __volatile__("yield");
#endif
}

void Os::sleep(long n) {
// FIXME_lmoriche: Should be nano-seconds not seconds.
#ifdef _WIN32
  ::Sleep(n);
#else   // !_WIN32
  time_t seconds = (time_t)n / 1000;
  long nanoseconds = ((long)n - seconds * 1000) * 1000000;
  timespec ts = {seconds, nanoseconds};
  ::nanosleep(&ts, NULL);
#endif  // !_WIN32
}

void Os::touchStackPages(address bottom, address top) {
  top = alignDown(top, pageSize_) - pageSize_;
  while (top >= bottom) {
    *top = 0;
    top -= pageSize_;
  }
}

bool Os::skipIDIV(address& pc) {
  address insn = pc;
  if (insn[0] == 0x66) {  // LCP prefix
    insn += 1;
  }
  if ((insn[0] & 0xf0) == 0x40) {  // REX prefix
    insn += 1;
  }
  if (insn[0] == 0xf6 || insn[0] == 0xf7) {  // IDIV
    // This is a DivisionError: skip the insn and resume execution
    char mod = insn[1] >> 6;
    char rm = insn[1] & 0x7;
    insn += 2;  // skip opcode and mod/rm

    if (rm == 0x4 && mod != 0x3) {
      insn += 1;  // sib follows mod/rm
    }

    if ((mod == 0x0 && rm == 0x5) || mod == 0x2) {
      insn += 4;  // disp32
    } else if (mod == 0x1) {
      insn += 1;  // disp8
    }
    pc = insn;
    return true;
  }
  return false;
}

void Os::setThreadAffinity(const void* handle, unsigned int cpu) {
  ThreadAffinityMask mask;
  mask.set(cpu);
  setThreadAffinity(handle, mask);
}

}  // namespace amd
