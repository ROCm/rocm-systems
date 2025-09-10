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

#ifndef DEBUG_HPP_
#define DEBUG_HPP_


#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdint>
//! \addtogroup Utils
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace amd { /*@{*/

enum LogLevel {
  LOG_NONE = 0,
  LOG_ERROR = 1,
  LOG_WARNING = 2,
  LOG_INFO = 3,
  LOG_DEBUG = 4,
  LOG_DETAIL_DEBUG = 5,
  LOG_EXTRA_DEBUG = 6
};

enum LogMask {
  LOG_API = 1,            //!< (0x1)     API call
  LOG_CMD = 2,            //!< (0x2)     Kernel and Copy Commands and Barriers
  LOG_WAIT = 4,           //!< (0x4)     Synchronization and waiting for commands to finish
  LOG_AQL = 8,            //!< (0x8)     Decode and display AQL packets
  LOG_QUEUE = 16,         //!< (0x10)    Queue commands and queue contents
  LOG_SIG = 32,           //!< (0x20)    Signal creation, allocation, pool
  LOG_LOCK = 64,          //!< (0x40)    Locks and thread-safety code.
  LOG_KERN = 128,         //!< (0x80)    Kernel creations and arguments, etc.
  LOG_COPY = 256,         //!< (0x100)   Copy debug
  LOG_COPY2 = 512,        //!< (0x200)   Detailed copy debug
  LOG_RESOURCE = 1024,    //!< (0x400)   Resource allocation, performance-impacting events.
  LOG_INIT = 2048,        //!< (0x800)   Initialization and shutdown
  LOG_MISC = 4096,        //!< (0x1000)  Misc debug, not yet classified
  LOG_AQL2 = 8192,        //!< (0x2000)  Show raw bytes of AQL packet
  LOG_CODE = 16384,       //!< (0x4000)  Show code creation debug
  LOG_CMD2 = 32768,       //!< (0x8000)  More detailed command info, including barrier commands
  LOG_LOCATION = 65536,   //!< (0x10000) Log message location
  LOG_MEM = 131072,       //!< (0x20000) Memory allocation
  LOG_MEM_POOL = 262144,  //!< (0x40000) Memory pool allocation, including memory in graphs
  LOG_TS = 524288,        //!< (0x80000) Timestamp details
  LOG_COMGR = 1048576,    //!< (0x100000) Comgr path information print
  LOG_ALWAYS = -1         //!< (0xFFFFFFFF) Log always even mask flag is zero
};

//! \brief log file output
extern FILE* outFile;

//! \brief Display a warning message.
extern void report_warning(const char* message);

//! \brief Insert a log entry.
extern void log_entry(LogLevel level, const char* file, int line, const char* messsage);

//! \brief Insert a timestamped log entry.
extern void log_timestamped(LogLevel level, const char* file, int line, const char* messsage);

//! \brief Insert a printf-style log entry.
extern void log_printf(LogLevel level, const char* file, int line, const char* format, ...);
extern void log_printf(LogLevel level, const char* file, int line, uint64_t* start,
                       const char* format, ...);

/*@}*/  // namespace amd
}  // namespace amd

#if __INTEL_COMPILER

// Disable ICC's warning #279: controlling expression is constant
// (0!=1 && "msg")
//          ^
#pragma warning(disable : 279)

#endif  // __INTEL_COMPILER

//! \brief Abort the program if the invariant \a cond is false.
#define guarantee(cond, format, ...)                                                               \
  if (!(cond)) {                                                                                   \
    amd::log_printf(amd::LOG_NONE, __FILE__, __LINE__, format, ##__VA_ARGS__);                     \
    ::abort();                                                                                     \
  }

#define fixme_guarantee(cond, ...) guarantee(cond, __VA_ARGS__)

//! \brief Abort the program with a fatal error message.
#define fatal(msg)                                                                                 \
  do {                                                                                             \
    assert(false && msg);                                                                          \
  } while (0)


//! \brief Display a warning message.
inline void warning(const char* msg) { amd::report_warning(msg); }

/*! \brief Abort the program with a "ShouldNotReachHere" message.
 *  \hideinitializer
 */
#define ShouldNotReachHere() fatal("ShouldNotReachHere()")

/*! \brief Abort the program with a "ShouldNotCallThis" message.
 *  \hideinitializer
 */
#define ShouldNotCallThis() fatal("ShouldNotCallThis()")

/*! \brief Abort the program with an "Unimplemented" message.
 *  \hideinitializer
 */
#define Unimplemented() fatal("Unimplemented()")

/*! \brief Display an "Untested" warning message.
 *  \hideinitializer
 */
#ifndef NDEBUG
#define Untested(msg) warning("Untested(\"" msg "\")")
#else /*NDEBUG*/
#define Untested(msg) (void)(0)
#endif /*NDEBUG*/

#ifdef _WIN32
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define Log(level, msg)                                                                            \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level) {                                                                  \
      amd::log_entry(level, __FILE__, __LINE__, msg);                                              \
    }                                                                                              \
  } while (false)

#define LogTS(level, msg)                                                                          \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level) {                                                                  \
      amd::log_timestamped(level, __FILE__, __LINE__, msg);                                        \
    }                                                                                              \
  } while (false)

#define Logf(level, format, ...)                                                                   \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level) {                                                                  \
      amd::log_printf(level, __FILE__, __LINE__, format, __VA_ARGS__);                             \
    }                                                                                              \
  } while (false)

#define CondLog(cond, msg)                                                                         \
  do {                                                                                             \
    if (false DEBUG_ONLY(|| (cond))) {                                                             \
      Log(amd::LOG_INFO, msg);                                                                     \
    }                                                                                              \
  } while (false)

#define LogGuarantee(cond, level, msg)                                                             \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level) {                                                                  \
      guarantee(cond);                                                                             \
    }                                                                                              \
  } while (false)


#define LogTSInfo(msg) LogTS(amd::LOG_INFO, msg)
#define LogTSError(msg) LogTS(amd::LOG_ERROR, msg)
#define LogTSWarning(msg) LogTS(amd::LOG_WARNING, msg)

#define DebugInfoGuarantee(cond) LogGuarantee(cond, amd::LOG_INFO, "Warning")

/* backend and compiler use AMD_LOG_LEVEL macro from makefile. Define AMD_LOG_MASK for them. */
#if defined(AMD_LOG_LEVEL)
#define AMD_LOG_MASK 0x7FFFFFFF
#endif

// You may define CL_LOG to enable following log functions even for release build
#define CL_LOG

#ifdef CL_LOG
#define ClPrint(level, mask, format, ...)                                                          \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level) {                                                                  \
      if (AMD_LOG_MASK & mask || mask == amd::LOG_ALWAYS) {                                        \
        if (AMD_LOG_MASK & amd::LOG_LOCATION) {                                                    \
          amd::log_printf(level, __FILENAME__, __LINE__, format, ##__VA_ARGS__);                   \
        } else {                                                                                   \
          amd::log_printf(level, "", 0, format, ##__VA_ARGS__);                                    \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } while (false)

#define IsLogEnabled(level, mask)                                                                  \
  (AMD_LOG_LEVEL >= level && (AMD_LOG_MASK & mask || AMD_LOG_MASK == amd::LOG_ALWAYS))

// called on entry and exit, calculates duration with local starttime variable defined in
// HIP_INIT_API
#define HIPPrintDuration(level, mask, startTimeUs, format, ...)                                    \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level) {                                                                  \
      if (AMD_LOG_MASK & mask || mask == amd::LOG_ALWAYS) {                                        \
        if (AMD_LOG_MASK & amd::LOG_LOCATION) {                                                    \
          amd::log_printf(level, __FILENAME__, __LINE__, startTimeUs, format, ##__VA_ARGS__);      \
        } else {                                                                                   \
          amd::log_printf(level, "", 0, startTimeUs, format, ##__VA_ARGS__);                       \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } while (false)

#define ClCondPrint(level, mask, condition, format, ...)                                           \
  do {                                                                                             \
    if (AMD_LOG_LEVEL >= level && (condition)) {                                                   \
      if (AMD_LOG_MASK & mask || mask == amd::LOG_ALWAYS) {                                        \
        amd::log_printf(level, __FILE__, __LINE__, format, ##__VA_ARGS__);                         \
      }                                                                                            \
    }                                                                                              \
  } while (false)

#else /*CL_LOG*/
#define ClPrint(level, mask, format, ...) (void)(0)
#define ClCondPrint(level, mask, condition, format, ...) (void)(0)
#define HIPPrintDuration(level, mask, startTimeUs, format, ...) (void)(0)
#endif /*CL_LOG*/

#define ClTrace(level, mask) ClPrint(level, mask, "%s", __func__)

#define LogInfo(msg) ClPrint(amd::LOG_INFO, amd::LOG_ALWAYS, msg)
#define LogError(msg) ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, msg)
#define LogWarning(msg) ClPrint(amd::LOG_WARNING, amd::LOG_ALWAYS, msg)

#define LogPrintfDebug(format, ...) ClPrint(amd::LOG_DEBUG, amd::LOG_ALWAYS, format, ##__VA_ARGS__)
#define LogPrintfError(format, ...) ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, format, ##__VA_ARGS__)
#define LogPrintfWarning(format, ...)                                                              \
  ClPrint(amd::LOG_WARNING, amd::LOG_ALWAYS, format, ##__VA_ARGS__)
#define LogPrintfInfo(format, ...) ClPrint(amd::LOG_INFO, amd::LOG_ALWAYS, format, ##__VA_ARGS__)

#if (defined(DEBUG) || defined(DEV_LOG_ENABLE))
#define DevLogPrintfError(format, ...) LogPrintfError(format, ##__VA_ARGS__)
#define DevLogError(msg) LogError(msg)
#else
#define DevLogPrintfError(format, ...)
#define DevLogError(msg)
#endif

#endif /*DEBUG_HPP_*/
