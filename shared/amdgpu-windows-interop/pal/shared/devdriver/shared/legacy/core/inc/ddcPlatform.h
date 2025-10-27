/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/

#pragma once

#include <stdarg.h>
// <new> can not be used in the kernel
#if !DD_PLATFORM_IS_KM
#include <new>
#endif

#include <ddDefs.h>
#include <ddTemplate.h>

#define DD_CACHE_LINE_BYTES 64

#define DD_MALLOC(size, alignment, allocCb) allocCb.Alloc(size, alignment, false)
#define DD_CALLOC(size, alignment, allocCb) allocCb.Alloc(size, alignment, true)
#define DD_FREE(memory, allocCb)            allocCb.Free(memory)

#define DD_NEW(className, allocCb) new(allocCb, alignof(className), true, DD_FILE, __LINE__, __FUNCTION__) className
#define DD_DELETE(memory, allocCb) DevDriver::Platform::Destructor(memory); DD_FREE(memory, allocCb)

#define DD_NEW_ARRAY(className, numElements, allocCb) DevDriver::Platform::NewArray<className>(numElements, allocCb)
#define DD_DELETE_ARRAY(memory, allocCb)              DevDriver::Platform::DeleteArray(memory, allocCb)

// Always enable asserts in Debug builds
#if !defined(NDEBUG)
    #if !defined(DD_OPT_ASSERTS_ENABLE)
        #define DD_OPT_ASSERTS_ENABLE
    #endif
    #if !defined(DD_OPT_ASSERTS_DEBUGBREAK)
        #define DD_OPT_ASSERTS_DEBUGBREAK
    #endif
#endif

#define DD_PTR_TO_HANDLE(x) ((DevDriver::Handle)(uintptr_t)(x))

#define DD_SANITIZE_RESULT(x) ((x != Result::Success) ? Result::Error : x)

namespace DevDriver
{

////////////////////////////
// Common logging levels
enum struct LogLevel : uint8
{
    Debug = 0,
    Verbose,
    Info,
    Warn,
    Error,
    Always,
    Count,

    // Backwards compatibility for old alert log level
    Alert = Warn,

    Never = 0xFF
};

typedef void*(*AllocFunc)(void* pUserdata, size_t size, size_t alignment, bool zero);
typedef void(*FreeFunc)(void* pUserdata, void* pMemory);

struct AllocCb
{
    void*     pUserdata;
    AllocFunc pfnAlloc;
    FreeFunc  pfnFree;

    void* Alloc(size_t size, size_t alignment, bool zero) const;
    void* Alloc(size_t size, bool zero) const;
    void Free(void* pMemory) const;
};

namespace Platform
{

// Used by the Platform::Thread implementation.
typedef void (*ThreadFunction)(void* pThreadParameter);

} // namespace Platform

} // namespace DevDriver

#if defined(DD_PLATFORM_WINDOWS_UM)
    #include <platforms/ddWinPlatform.h>
#elif defined(DD_PLATFORM_WINDOWS_KM)
    #include <platforms/ddWinKernelPlatform.h>
#elif defined(DD_PLATFORM_DARWIN_UM)
    #include <platforms/ddPosixPlatform.h>
#elif defined(DD_PLATFORM_LINUX_UM)
    #include <platforms/ddPosixPlatform.h>
#else
    // Legacy system for Ati Make
    #if defined(_WIN32) && !defined(_KERNEL_MODE)
        #define DD_PLATFORM_WINDOWS_UM
        #include <platforms/ddWinPlatform.h>
    #elif defined(__linux__)
        #define DD_PLATFORM_LINUX_UM
        #include <platforms/ddPosixPlatform.h>
    #else
        #error "Unknown Platform - please configure your build system"
    #endif

    #if __x86_64__
        #define DD_ARCH_BITS 64
    #else
        #define DD_ARCH_BITS 32
    #endif
#endif

#if !defined(DD_RESTRICT)
    #error "DD_RESTRICT not defined by platform!"
#endif

#if !defined(DD_DEBUG_BREAK)
    #error "DD_DEBUG_BREAK not defined by platform!"
#endif

// This only exists for 32bit Windows to specificy callbacks as __stdcall.
#if !defined(DD_APIENTRY)
    #define DD_APIENTRY
#endif

// TODO: remove this and make kDebugLogLevel DD_STATIC_CONST when we use a version of visual studio that supports it
#ifdef DD_OPT_LOG_LEVEL
    #define DD_OPT_LOG_LEVEL_VALUE static_cast<LogLevel>(DD_OPT_LOG_LEVEL)
#else
    #if defined(NDEBUG)
        // In non-debug builds, default to printing asserts, Error, and Always log messages
        #define DD_OPT_LOG_LEVEL_VALUE LogLevel::Error
    #else
        // In debug builds, default to more messages
        #define DD_OPT_LOG_LEVEL_VALUE LogLevel::Verbose
    #endif
#endif

#define DD_WILL_PRINT(lvl) ((lvl >= DD_OPT_LOG_LEVEL_VALUE) && (lvl < DevDriver::LogLevel::Count))
#define DD_PRINT(lvl, ...) DevDriver::LogString<lvl>(__VA_ARGS__)

#if defined(DD_OPT_ASSERTS_DEBUGBREAK)
    #define DD_ASSERT_DEBUG_BREAK() DD_DEBUG_BREAK()
#else
    #define DD_ASSERT_DEBUG_BREAK()
#endif

#include <dd_crc32.h>

// Calling `check_expr_is_bool(x)` when `x` is not exactly a bool will create a compile error.
// When it is a bool, it's a no-op.
// This allows us to enforce bool arguments to DD_ASSERT() macros
namespace DevDriver
{
    inline void check_expr_is_bool(bool) {}

    template <typename T>
    void check_expr_is_bool(const T&) = delete;
}

#if !defined(DD_OPT_ASSERTS_ENABLE)
    #define DD_WARN(statement)       DD_UNUSED(0)
    #define DD_WARN_REASON(reason)   DD_UNUSED(0)

    #ifndef DD_ASSERT
        #define DD_ASSERT(statement)     DD_UNUSED(0)
    #endif

    #define DD_ASSERT_REASON(reason) DD_UNUSED(0)
#else
    #define DD_WARN(statement) do                                                         \
    {                                                                                     \
        DevDriver::check_expr_is_bool(statement);                                         \
        if (!(statement))                                                                 \
        {                                                                                 \
            DD_PRINT(DevDriver::LogLevel::Warn, "%s (%d): Warning triggered in %s: %s",   \
                DD_FILE, __LINE__, __func__, DD_STRINGIFY(statement));                   \
        }                                                                                 \
    } while (0)

    #define DD_WARN_REASON(reason) do                                                 \
    {                                                                                 \
        DD_PRINT(DevDriver::LogLevel::Warn, "%s (%d): Warning triggered in %s: %s",   \
            DD_FILE, __LINE__, __func__, reason);                                    \
    } while (0)

    #ifndef DD_ASSERT
        #define DD_ASSERT(statement) do                                                       \
        {                                                                                     \
            DevDriver::check_expr_is_bool(statement);                                         \
            if (!(statement))                                                                 \
            {                                                                                 \
                DD_PRINT(DevDriver::LogLevel::Error, "%s (%d): Assertion failed in %s: %s",   \
                    DD_FILE, __LINE__, __func__, DD_STRINGIFY(statement));                   \
                DD_ASSERT_DEBUG_BREAK();                                                      \
            }                                                                                 \
        } while (0)
    #endif

    #define DD_ASSERT_REASON(reason) do                                               \
    {                                                                                 \
        DD_PRINT(DevDriver::LogLevel::Error, "%s (%d): Assertion failed in %s: %s",   \
            DD_FILE, __LINE__, __func__, reason);                                    \
        DD_ASSERT_DEBUG_BREAK();                                                      \
    } while (0)
#endif

/// Convenience macro that always warns.
#define DD_WARN_ALWAYS() DD_WARN_REASON("Unconditional Warning")

/// Convenience macro that always asserts.
#define DD_ASSERT_ALWAYS() DD_ASSERT_REASON("Unconditional Assertion")

/// Convenience macro that asserts if something has not been implemented.
#define DD_NOT_IMPLEMENTED() DD_ASSERT_REASON("Code not implemented!")

/// Convenience macro that asserts if an area of code that shouldn't be executed is reached.
#define DD_UNREACHABLE() DD_ASSERT_REASON("Unreachable code has been reached!")

// Backwards compatibility for old alert macro
#define DD_ALERT(statement)      DD_WARN(statement)
#define DD_ALERT_REASON(reason)  DD_WARN_REASON(reason)
#define DD_ALERT_ALWAYS()        DD_WARN_ALWAYS()

// Debug utility to log an expression
//
// This works by taking the format specifier for a local variable, and an expression.
// The expression is evaluated once.
// It then prints that expression and its value:
// ```cpp
//      int x = 5;
//      int y = 10;
//      int z = 0xf0;
//      DD_DBG("0x%x", x + y + z); // Prints: foo/file.cpp:5   "x + y + z" == 0xff
// ```
#define DD_DBG(level, fmt, expr) DD_PRINT(              \
        level,                                          \
        "%s:%d:\t\"" DD_STRINGIFY(expr) "\" == " fmt,   \
        DD_FILE,                                       \
        __LINE__,                                       \
        (expr)                                          \
    )

// Allocates memory using an AllocCb.
// This overload is declared noexcept, and will correctly handle AllocCb::pfnAlloc() returning NULL.
void* operator new(
    size_t                    size,
    const DevDriver::AllocCb& allocCb,
    size_t                    align,
    bool                      zero,
    const char*               pFilename,
    int                       lineNumber,
    const char*               pFunction
) noexcept;

#if DD_PLATFORM_IS_KM
// Provide a placement new function if <new> is not available
inline void* operator new(size_t size, void *pMemory)
{
    return pMemory;
};
#endif

// Overload of operator delete that matches the previously declared operator new.
// The compiler can call this version automatically in the case of exceptions thrown in the Constructor
// ... even though we turn them off?
// Compilers are fussy.
void operator delete(
    void*                     pObject,
    const DevDriver::AllocCb& allocCb,
    size_t                    align,
    bool                      zero,
    const char*               pFilename,
    int                       lineNumber,
    const char*               pFunction
) noexcept;

namespace DevDriver
{

namespace Platform
{

template<typename T>
inline void static Destructor(T* p)
{
    if (p != nullptr)
    {
        p->~T();
    }
}

template<typename T>
static T* NewArray(size_t numElements, const AllocCb& allocCb)
{
    size_t allocSize = (sizeof(T) * numElements) + DD_CACHE_LINE_BYTES;
    size_t allocAlign = DD_CACHE_LINE_BYTES;

    T* pMem = reinterpret_cast<T*>(DD_MALLOC(allocSize, allocAlign, allocCb));
    if (pMem != nullptr)
    {
        pMem = reinterpret_cast<T*>(reinterpret_cast<char*>(pMem) + DD_CACHE_LINE_BYTES);
        size_t* pNumElements = reinterpret_cast<size_t*>(reinterpret_cast<char*>(pMem) - sizeof(size_t));
        *pNumElements = numElements;
        T* pCurrentElement = pMem;
        for (size_t elementIndex = 0; elementIndex < numElements; ++elementIndex)
        {
            new(pCurrentElement) T;
            ++pCurrentElement;
        }
    }

    return pMem;
}

template<typename T>
static void DeleteArray(T* pElements, const AllocCb& allocCb)
{
    if (pElements != nullptr)
    {
        size_t numElements = *reinterpret_cast<size_t*>(reinterpret_cast<char*>(pElements) - sizeof(size_t));
        T* pCurrentElement = pElements;
        for (size_t elementIndex = 0; elementIndex < numElements; ++elementIndex)
        {
            pCurrentElement->~T();
            ++pCurrentElement;
        }

        pElements = reinterpret_cast<T*>(reinterpret_cast<char*>(pElements) - DD_CACHE_LINE_BYTES);
    }

    DD_FREE(pElements, allocCb);
}

// Get the number of elements in a statically sized array
// Usage:
//      char buffer[1024];
//      size_t size = ArraySize(buffer); // size == 1024
//
//  With a cast:
//      char buffer[1024];
//      uint32 size = ArraySize<uint32>(buffer);
//
template <
    typename SizeT = size_t,    // Type to return
    typename T,                 // Inferred type of array elements - you should not need to supply this argument
    size_t   Size               // Inferred length of array (in elements) - you should not need to supply this argument
>
constexpr SizeT ArraySize(const T(&)[Size])
{
    return static_cast<SizeT>(Size);
}

// Log to consoles and attached debuggers
void DebugPrint(LogLevel lvl, const char* pFormat, ...);

// Platform-specific loggers, this is called from DebugPrint.
void PlatformDebugPrint(LogLevel lvl, const char* pString);

/// Get the absolute path to a file or directory that already exists
/// If ppAbsPathFilePart is non-NULL, *ppAbsPathFilePart will point into absPath at the beginning of the Filename
/// This is recommended to do whenever you need to display a path to a user.
Result GetAbsPathName(
    const char*  pPath,
    char         (&absPath)[256]
);

/* platform functions for performing atomic operations */

int32 AtomicIncrement(Atomic* pVariable);
int32 AtomicDecrement(Atomic* pVariable);
int32 AtomicAdd(Atomic* pVariable, int32 num);
int32 AtomicSubtract(Atomic* pVariable, int32 num);

int64 AtomicIncrement(Atomic64* pVariable);
int64 AtomicDecrement(Atomic64* pVariable);
int64 AtomicAdd(Atomic64* pVariable, int64 num);
int64 AtomicSubtract(Atomic64* pVariable, int64 num);

// A generic AllocCb that defers allocation to Platform::AllocateMemory()
// Suitable for memory allocation if you don't care about it.
extern AllocCb GenericAllocCb;

void* AllocateMemory(size_t size, size_t alignment, bool zero);
void FreeMemory(void* pMemory);

/* fast locks */
class AtomicLock
{
public:
    AtomicLock() : m_lock(0) {};
    ~AtomicLock() {};
    void Lock();
    bool TryLock();
    void Unlock();
    bool IsLocked() { return (m_lock != 0); };
private:
    Atomic m_lock;
};

class Mutex
{
public:
    Mutex();
    ~Mutex();
    void Lock();
    void Unlock();
private:
    MutexStorage m_mutex;
};

class Semaphore
{
public:
    explicit Semaphore(uint32 initialCount, uint32 maxCount);
    ~Semaphore();
    Result Signal();
    Result Wait(uint32 millisecTimeout);
private:
    SemaphoreStorage m_semaphore;
};

class Event
{
public:
    explicit Event(bool signaled);
    ~Event();
    void Clear();
    void Signal();
    Result Wait(uint32 timeoutInMs);
private:
    EventStorage m_event;
};

class Thread
{
public:
    Thread() = default;

    Thread(Thread&& other) noexcept = default;
    Thread& operator=(Thread&& other) noexcept = default;

    // Copying a thread doesn't make sense
    Thread(const Thread&) = delete;
    Thread& operator= (const Thread& other) = delete;

    ~Thread();

    Result Start(ThreadFunction pFnThreadFunc, void* pThreadParameter);

    // Set the user-visible name for the thread using printf-style formatters
    // This should only be called on valid thread objects. (Threads that have been started)
    // This function will return Result::Error if it's called on an invalid thread.
    // Note: This change is global to the thread and can be changed by other means
    //       Treat this as an aid for people
    Result SetName(const char* pFmt, ...);

    Result Join(uint32 timeoutInMs);

    bool IsJoinable() const;

private:
    static ThreadReturnType DD_APIENTRY ThreadShim(void* pShimParam);

    // Reset our object to a default state
    void Reset()
    {
        pFnFunction = nullptr;
        pParameter  = nullptr;
        hThread     = kInvalidThreadHandle;

        onExit.Clear();
    }

    // Set the thread name to a hard-coded string.
    // The thread name passed to this function must be no larger than kThreadNameMaxLength including the NULL byte.
    // If a larger string is passed, errors may occur on some platforms.
    Result SetNameRaw(const char* pThreadName);

    ThreadFunction pFnFunction = nullptr;
    void*          pParameter  = nullptr;
    ThreadHandle   hThread     = kInvalidThreadHandle;
    Event          onExit      = Event(false); // Start unsignaled
};

class Random
{
public:
    // Algorithm Constants
    static constexpr uint64 kModulus    = (uint64(1) << 48);
    static constexpr uint64 kMultiplier = 0X5DEECE66Dull;
    static constexpr uint16 kIncrement  = 0xB;

    Random();
    Random(uint64 seed)
    {
        Reseed(seed);
    }
    ~Random() {}

    uint32 Generate();
    void Reseed(uint64 seed);
private:
    uint64 m_prevState = 0;

    // Sanity checks.
    static_assert(0 < kModulus,           "Invalid modulus");
    static_assert(0 < kMultiplier,        "Invalid multiplier");
    static_assert(kMultiplier < kModulus, "Invalid multiplier");
    static_assert(kIncrement < kModulus,  "Invalid increment");
};

class Library
{
public:
    Library() : m_hLib(nullptr) { }
    ~Library() { Close(); }

    Result Load(const char* pLibraryName);

    void Close();

    bool IsLoaded() const { return (m_hLib != nullptr); }

    void Swap(Library* pLibrary)
    {
        m_hLib = pLibrary->m_hLib;
        pLibrary->m_hLib = nullptr;
    }

    // Retrieve a function address from the dynamic library object. Returns true if successful, false otherwise.
    template <typename Func_t>
    bool GetFunction(const char* pName, Func_t* ppfnFunc) const
    {
        (*ppfnFunc) = reinterpret_cast<Func_t>(GetFunctionHelper(pName));
        return ((*ppfnFunc) != nullptr);
    }

private:
    void* GetFunctionHelper(const char* pName) const;

    LibraryHandle m_hLib;

    DD_DISALLOW_COPY_AND_ASSIGN(Library);
};

enum struct MkdirStatus
{
    Unknown,
    Created,
    Existed,
};

// Create a directory with default permissions
//      On Windows, this uses NULL for LPSECURITY_ATTRIBUTES
//      On Unix, this uses 0777 for the mode.
// When pStatus is non-NULL, *pStatus is set to
//      MkdirStatus::Created if the directory did not exist and was created
//      MkdirStatus::Existed if the directory already existed
// Returns:
//      - Result::Success,     if the directory already exists or was created
//      - Result::FileIoError, if the directory failed to be created
Result Mkdir(const char* pDir, MkdirStatus* pStatus = nullptr);

ProcessId GetProcessId();

uint64 GetCurrentTimeInMs();

uint64 QueryTimestampFrequency();
uint64 QueryTimestamp();

// Todo: Remove Sleep() entirely from our platform API. It cannot be used in the KMD and should not be used
// anywhere else either.
void Sleep(uint32 millisecTimeout);

void GetProcessName(char* buffer, size_t bufferSize);

void Strncpy(char* pDst, const char* pSrc, size_t dstSize);

template <size_t DstSize>
void Strncpy(char(&dst)[DstSize], const char* pSrc)
{
    Strncpy(dst, pSrc, DstSize);
}

char* Strtok(char* pDst, const char* pDelimiter, char** ppContext);

void Strncat(char* pDst, const char* pSrc, size_t dstSize);

template <size_t DstSize>
void Strncat(char(&dst)[DstSize], const char* pSrc)
{
    Strncat(dst, pSrc, DstSize);
}

int32 Strcmpi(const char* pSrc1, const char* pSrc2);

int32 Snprintf(char* pDst, size_t dstSize, const char* pFormat, ...);
int32 Vsnprintf(char* pDst, size_t dstSize, const char* pFormat, va_list args);

template <size_t DstSize, typename... Args>
int32 Snprintf(char(&dst)[DstSize], const char* pFormat, Args&&... args)
{
    return Snprintf(dst, DstSize, pFormat, args...);
}

struct OsInfo
{
    DD_STATIC_CONST const char* kOsTypeWindows = "Windows";
    DD_STATIC_CONST const char* kOsTypeLinux   = "Linux";
    DD_STATIC_CONST const char* kOsTypeDarwin  = "Darwin";

    char type[16];         /// The type of the OS, either "Windows", "Linux", or "Darwin".

    char name[32];         /// A human-readable string to identify the version of the OS running
    char description[256]; /// A human-readable string to identify the detailed version of the OS running
    char hostname[128];    /// The hostname for the machine

    struct UserInfo {
        char name[32];     /// Username for the current user
        char homeDir[128]; /// Path to the current user's home directory
                           //< This is typically stored in $HOME or %HOMEPATH% and looks like one of:
                           //<     C:\Users\BobMarley
                           //<     /home/bob_ross
                           //<     /Users/BobTheBuilder
    } user;

    uint64 physMemory; /// Total amount of memory available on host in bytes
    uint64 swapMemory; /// Total amount of swap memory available on host in bytes
};

 Result QueryOsInfo(OsInfo* pInfo);

struct EtwSupportInfo
 {
     bool   isSupported;            ///< If true, indicates that the OS platform supports system monitoring, false otherwise.
     bool   hasPermission;          ///< If true, indicates the account has the required permissions, false otherwise.
     uint32 statusCode;             ///< The status result returned when attempting to open a monitoring session.
     char   statusDescription[256]; ///< The textual status result returned when attempting to open a monitoring.
 };

 Result QueryEtwInfo(EtwSupportInfo* pInfo);

} // Platform

#ifndef DD_PRINT_FUNC
#define DD_PRINT_FUNC Platform::DebugPrint
#else
void DD_PRINT_FUNC(LogLevel logLevel, const char* format, ...);
#endif

template <LogLevel logLevel = LogLevel::Info, class ...Ts>
inline void LogString(const char *format, Ts&&... args)
{
    if (DD_WILL_PRINT(logLevel))
    {
        DD_PRINT_FUNC(logLevel, format, Platform::Forward<Ts>(args)...);
    }
}

// Increments a const pointer by numBytes by first casting it to a const uint8*.
DD_NODISCARD
constexpr const void* VoidPtrInc(
    const void* pPtr,
    size_t      numBytes)
{
    return (static_cast<const uint8*>(pPtr) + numBytes);
}

// Increments a pointer by numBytes by first casting it to a uint8*.
DD_NODISCARD
constexpr void* VoidPtrInc(
    void*  pPtr,
    size_t numBytes)
{
    return (static_cast<uint8*>(pPtr) + numBytes);
}

// Decrements a const pointer by numBytes by first casting it to a const uint8*.
DD_NODISCARD
constexpr const void* VoidPtrDec(
    const void* pPtr,
    size_t      numBytes)
{
    return (static_cast<const uint8*>(pPtr) - numBytes);
}

// Decrements a pointer by numBytes by first casting it to a uint8*.
DD_NODISCARD
constexpr void* VoidPtrDec(
    void*  pPtr,
    size_t numBytes)
{
    return (static_cast<uint8*>(pPtr) - numBytes);
}

/// Convert a `DevDriver::Result` into a human recognizable string.
static inline const char* ResultToString(Result result)
{
    switch (result)
    {
        //// Generic Result Code  ////
        case Result::Success:            return "Success";
        case Result::Error:              return "Error";
        case Result::NotReady:           return "NotReady";
        case Result::VersionMismatch:    return "VersionMismatch";
        case Result::Unavailable:        return "Unavailable";
        case Result::Rejected:           return "Rejected";
        case Result::EndOfStream:        return "EndOfStream";
        case Result::Aborted:            return "Aborted";
        case Result::InsufficientMemory: return "InsufficientMemory";
        case Result::InvalidParameter:   return "InvalidParameter";
        case Result::InvalidClientId:    return "InvalidClientId";
        case Result::ConnectionExists:   return "ConnectionExists";
        case Result::FileNotFound:       return "FileNotFound";
        case Result::FunctionNotFound:   return "FunctionNotFound";
        case Result::InterfaceNotFound:  return "InterfaceNotFound";
        case Result::EntryExists:        return "EntryExists";
        case Result::FileAccessError:    return "FileAccessError";
        case Result::FileIoError:        return "FileIoError";
        case Result::LimitReached:       return "LimitReached";
        case Result::MemoryOverLimit:    return "MemoryOverLimit";

        //// URI PROTOCOL  ////
        case Result::UriServiceRegistrationError:  return "UriServiceRegistrationError";
        case Result::UriStringParseError:          return "UriStringParseError";
        case Result::UriInvalidParameters:         return "UriInvalidParameters";
        case Result::UriInvalidPostDataBlock:      return "UriInvalidPostDataBlock";
        case Result::UriInvalidPostDataSize:       return "UriInvalidPostDataSize";
        case Result::UriFailedToAcquirePostBlock:  return "UriFailedToAcquirePostBlock";
        case Result::UriFailedToOpenResponseBlock: return "UriFailedToOpenResponseBlock";
        case Result::UriRequestFailed:             return "UriRequestFailed";
        case Result::UriPendingRequestError:       return "UriPendingRequestError";
        case Result::UriInvalidChar:               return "UriInvalidChar";
        case Result::UriInvalidJson:               return "UriInvalidJson";

        //// Settings URI Service  ////
        case Result::SettingsUriInvalidComponent:        return "SettingsUriInvalidComponent";
        case Result::SettingsUriInvalidSettingName:      return "SettingsUriInvalidSettingName";
        case Result::SettingsUriInvalidSettingValue:     return "SettingsUriInvalidSettingValue";
        case Result::SettingsUriInvalidSettingValueSize: return "SettingsUriInvalidSettingValueSize";

        //// Info URI Service ////
        case Result::InfoUriSourceNameInvalid:       return "InfoUriSourceNameInvalid";
        case Result::InfoUriSourceCallbackInvalid:   return "InfoUriSourceCallbackInvalid";
        case Result::InfoUriSourceAlreadyRegistered: return "InfoUriSourceAlreadyRegistered";
        case Result::InfoUriSourceWriteFailed:       return "InfoUriSourceWriteFailed";

        //// Settings Service  ////
        case Result::SettingsInvalidComponent:        return "SettingsInvalidComponent";
        case Result::SettingsInvalidSettingName:      return "SettingsInvalidSettingName";
        case Result::SettingsInvalidSettingValue:     return "SettingsInvalidSettingValue";
        case Result::SettingsInsufficientValueSize:   return "SettingsInsufficientValueSize";
        case Result::SettingsInvalidSettingValueSize: return "SettingsInvalidSettingValueSize";
    }

    DD_PRINT(LogLevel::Warn, "Result code %u is not handled", static_cast<uint32>(result));
    return "Unrecognized DevDriver::Result";
}

// Helper function for converting bool values into Result enums
// Useful for cases where Results and bools are interleaved in logic
static inline Result BoolToResult(bool value)
{
    return (value ? Result::Success : Result::Error);
}

// Use this macro to mark Result values that have not been or cannot be handled correctly.
#define DD_UNHANDLED_RESULT(x) DevDriver::MarkUnhandledResultImpl((x), DD_STRINGIFY(x), DD_FILE, __LINE__, __func__)

// Implementation for DD_UNHANDLED_RESULT.
// This is a specialized assert that should be used through the macro, and not called directly.
// This is implemented in ddPlatform.h, so that it has access to DD_ASSERT.
static inline void MarkUnhandledResultImpl(
    Result      result,
    const char* pExpr,
    const char* pFile,
    int         lineNumber,
    const char* pFunc)
{
#if defined(DD_OPT_ASSERTS_ENABLE)
    if (result != Result::Success)
    {
        DD_PRINT(DevDriver::LogLevel::Error,
            "%s (%d): Unchecked Result in %s: \"%s\" == \"%s\" (0x%X)\n",
            pFile,
            lineNumber,
            pFunc,
            pExpr,
            ResultToString(result),
            result);
    }
#else
    DD_UNUSED(result);
    DD_UNUSED(pExpr);
    DD_UNUSED(pFile);
    DD_UNUSED(lineNumber);
    DD_UNUSED(pFunc);
#endif
}

} // DevDriver
