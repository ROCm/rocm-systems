/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2014-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
/**
 ***********************************************************************************************************************
 * @file  palSysUtil.h
 * @brief PAL utility collection system functions.
 ***********************************************************************************************************************
 */

#pragma once

// pal
#include "palAssert.h"
#include "palAutoBuffer.h"
#include "palFile.h"
#include "palSpan.h"
#include "palStringView.h"
#include "palTime.h"
#include "palUtil.h"
#include "palVector.h"

// stl
#include <cerrno>
#include <cstring>

// platform
#if defined(_WIN32)
#define PAL_HAS_CPUID (_M_IX86 || _M_X64)
#include <intrin.h>
#include <winerror.h>
typedef void* HANDLE;
#elif defined(__unix__)
#define PAL_HAS_CPUID (__i386__ || __x86_64__)
#if PAL_HAS_CPUID
#include <cpuid.h>
#endif
#endif

namespace Util
{

static constexpr uint32 RyzenMaxCcxCount = 4;
static constexpr uint32 CpuVendorAmd     = 0x01000000;
static constexpr uint32 CpuVendorIntel   = 0x02000000;

/// Specifies a keyboard key for detecting key presses.
enum class KeyCode : uint32
{
    Esc,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    Backtick,   // ` ~
    Minus,      // - _
    Equal,      // = +
    LBrace,     // [ {
    RBrace,     // ] }
    Backslash,  // \ |
    Semicolon,  // ; :
    Apostrophe, // " '
    Comma,      // , <
    Dot,        // . >
    Slash,      // / ?
    Enter,
    Space,
    Backspace,
    Tab,
    Capslock,
    Shift,
    LShift,
    RShift,
    Control,
    LControl,
    RControl,
    Alt,
    LAlt,
    RAlt,
    Scroll,
    Insert,
    Delete,
    Home,
    End,
    PageUp,
    PageDown,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Numlock,
    NumSlash,
    NumAsterisk,
    NumMinus,
    NumPlus,
    NumDot,
    NumEnter,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Shift_F10,
    Shift_F11,
    Undefined // Used as value where key code is not defined in the enum
};

/// Enum to identify possible configurations
enum class CpuType : uint32
{
    Unknown         = 0,                       ///< No capabilites set
    AmdK5           = (CpuVendorAmd + 0),      ///< No MMX, no cmov, no 3DNow
    AmdK6           = (CpuVendorAmd + 1),      ///< No MMX, no cmov, 3DNow (models 6 and 7)
    AmdK6_2         = (CpuVendorAmd + 2),      ///< MMX, no cmov, 3DNow (model 8, no HW WC but not part of cpuid)
    AmdK6_3         = (CpuVendorAmd + 3),      ///< MMX, no cmov, 3DNow (model 9)
    AmdK7Basic      = (CpuVendorAmd + 4),      ///< K7 missing one of the features of K7
    AmdK7           = (CpuVendorAmd + 5),      ///< MMX, MMX Ext, cmov, 3DNow, 3DNow Ext
    AmdK7Sse        = (CpuVendorAmd + 6),      ///< MMX, MMX Ext, cmov, 3DNow, 3DNow Ext, SSE
    AmdK8           = (CpuVendorAmd + 7),      ///< Athlon 64, Athlon 64 FX, and Opteron
    AmdK10          = (CpuVendorAmd + 8),      ///< Barcelona, Phenom, Greyhound
    AmdFamily12h    = (CpuVendorAmd + 9),      ///< Family 12h - Llano
    AmdBobcat       = (CpuVendorAmd + 10),     ///< Bobcat
    AmdFamily15h    = (CpuVendorAmd + 11),     ///< Family 15h - Orochi, Trinity, Komodo, Kaveri, Basilisk
    AmdFamily16h    = (CpuVendorAmd + 12),     ///< Family 16h - Kabini
    AmdRyzen        = (CpuVendorAmd + 13),     ///< Ryzen
    IntelOld        = (CpuVendorIntel + 0),    ///< Inidicate cpu type befor Intel Pentium III
    IntelP3         = (CpuVendorIntel + 1),    ///< Generic Pentium III
    IntelP3Model7   = (CpuVendorIntel + 2),    ///< PIII-7, PIII Xeon-7
    IntelP3Model8   = (CpuVendorIntel + 3),    ///< PIII-8, PIII Xeon-8, Celeron-8
    IntelPMModel9   = (CpuVendorIntel + 4),    ///< Pentium M Model 9 (Banias)
    IntelXeonModelA = (CpuVendorIntel + 5),    ///< Xeon-A
    IntelP3ModelB   = (CpuVendorIntel + 6),    ///< PIII-B
    IntelPMModelD   = (CpuVendorIntel + 7),    ///< Pentium M Model D (Dothan)
    IntelP4         = (CpuVendorIntel + 8),    ///< Pentium 4, Pentium 4-M, Xenon, Celeron
    IntelPMModelE   = (CpuVendorIntel + 9),    ///< Pentium M Model E (Yonah)
    IntelCoreModelF = (CpuVendorIntel + 10),   ///< Core F (Conroe)
};

/// Specifies a struct that contains information about the system.
struct SystemInfo
{
    CpuType cpuType;              ///< Cpu type
    char    cpuVendorString[16];  ///< Null-terminated cpu vendor string
    char    cpuBrandString[48];   ///< Null-terminated cpu brand string
    uint32  cpuLogicalCoreCount;  ///< Number of logical cores on the cpu
    uint32  cpuPhysicalCoreCount; ///< Number of physical cores on the cpu
    uint32  totalSysMemSize;      ///< Total system memory (RAM) size in megabytes
    uint32  cpuFrequency;         ///< Reports CPU clock speed in MHz.
                                  ///  (From Registry for Windows, current average processor speed for Linux.)
    uint32  displayFamily;        ///< Display Family of cpu
    uint32  displayModel;         ///< Display Model of cpu

    union
    {
        struct
        {
            uint32 affinityMask[RyzenMaxCcxCount]; ///< Affinity mask for each core complex (CCX).
        } amdRyzen; ///< Properties specific to AMD Ryzen CPU's.
    } cpuArchInfo;  ///< This member should be used only for Ryzen for now.
};

/// Returns an appropriate result from the given errno
///
/// @param errno_in Value from 'errno' (or functions that return errno_t)
///
/// @returns Relevent Result value for the given errno-- never Success.
inline Result ConvertErrno(
    int32 errnoIn)
{
    Result result = Result::ErrorUnknown;
    switch (errnoIn)
    {
        case EAGAIN:
        case EBUSY:
        case EINTR:
            result = Result::NotReady;
            break;
        case ETIMEDOUT:
        case ETIME:
            result = Result::Timeout;
            break;
        case EEXIST:
            result = Result::AlreadyExists;
            break;
        case ENOENT:
        case ENOTDIR:
            result = Result::NotFound;
            break;
        case EACCES:
        case EPERM:
        case EROFS:
            result = Result::ErrorPermissionDenied;
            break;
        case ENOSPC:
            result = Result::ErrorDiskFull;
            break;
        case EISDIR:
        case EINVAL:
        case EBADF:
        case ENAMETOOLONG:
        case ELOOP:
            result = Result::ErrorInvalidValue;
            break;
        case ENOMEM:
        case EOVERFLOW:
            result = Result::ErrorOutOfMemory;
            break;
        default:
            PAL_ALERT_ALWAYS_MSG("Unknown result generated from errno %d (%s)", errnoIn, strerror(errnoIn));
            break;
    }
    return result;
}

#if defined(_WIN32)
/// Helper function to convert Pal::Result to HRESULT
///
/// @param result Pal::Result to convert
///
/// @returns the equivalent HRESULT
extern HRESULT PalToHResult(Result result);

/// Helper function to convert HRESULT to Pal::Result
///
/// @param hr HRESULT to convert
///
/// @returns the equivalent Pal::Result
extern Result HResultToPal(HRESULT hr);
#endif

#if defined(_WIN32)
/// Returns an appropriate result from the given error from 'GetLastError'.
/// Certain OS functions may also return a 'ERROR_SUCCESS' which this helper also translates.
///
/// @param errno_in System error code from 'GetLastError'
///
/// @returns Relevent Result value for the given system error code.
inline Result ConvertWinError(
    uint32 winError)
{
    Result result = Result::ErrorUnknown;
    switch (winError)
    {
        // File already exists
        case ERROR_ALREADY_EXISTS:
        case ERROR_FILE_EXISTS:
            result = Result::AlreadyExists;
            break;

        // File or path does not exist
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:
        case ERROR_INVALID_NAME:
            result = Result::NotFound;
            break;

        // File locked for specified access
        case ERROR_SHARING_VIOLATION:
        case ERROR_LOCK_VIOLATION:
            result = Result::ErrorNotShareable;
            break;
        case ERROR_ACCESS_DENIED:
            result = Result::ErrorPermissionDenied;
            break;

        // Temporary Errors
        case ERROR_NOT_READY:
        case ERROR_BUSY:
            result = Result::NotReady;
            break;
        case WAIT_TIMEOUT:
            result = Result::Timeout;
            break;

        // Usage errors
        case ERROR_BUFFER_OVERFLOW:
        case ERROR_INVALID_HANDLE:
            result = Result::ErrorInvalidValue;
            break;

        // Various expected but rare error cases
        case ERROR_TOO_MANY_OPEN_FILES:
        case ERROR_FILE_TOO_LARGE:
        case ERROR_VIRUS_INFECTED:
        case ERROR_VIRUS_DELETED:
            result = Result::ErrorUnavailable;
            break;
        case ERROR_HANDLE_DISK_FULL:
        case ERROR_DISK_FULL:
            result = Result::ErrorDiskFull;
            break;
        case ERROR_NOT_ENOUGH_MEMORY:
        case ERROR_OUTOFMEMORY:
            result = Result::ErrorOutOfMemory;
            break;

        case ERROR_SUCCESS:
            result = Result::Success;
            break;
        default:
            PAL_ALERT_ALWAYS_MSG("Unknown result generated from Windows error code %d", winError);
            break;
    }
    return result;
}

/// Windows Process Integrity Levels
enum class ProcessIntegrityLevel
{
    Untrusted,  //< Untrusted Process (No System Resource Access)
    Low,        //< Restricted Process
    Medium,     //< Normal Process
    High,       //< Elevated Process
    System      //< System-Level or higher Process
};

/// Get the local storage path based on process integrity level
///
/// @returns the Integrity Level of the current process
extern ProcessIntegrityLevel GetProcessIntegrityLevel();

/// Detects if we're inside an App Container sandbox (ie, a Windows Store or UWP app)
///
/// @returns whether the current process is in an App Container
extern bool IsProcessInAppContainer();

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 921
/// Tests whether the passed handle is valid or not.
/// A handle could be either nullptr or INVALID_HANDLE_VALUE.
/// INVALID_HANDLE_VALUE is the proper value to set a handle to when you intend to error out.
/// However, handles default-initialize to nullptr. So that's also a common value you see for an unset handle.
/// A common error is to check for one value but not the other. So PAL prescribes that clients use this helper.
constexpr bool IsValidHandle(HANDLE handle)
{
#ifdef INVALID_HANDLE_VALUE
    return (handle != nullptr) && (handle != INVALID_HANDLE_VALUE);
#else
    return (handle != nullptr) && (handle != HANDLE(size_t(-1)));
#endif
}
#endif
#endif

/// Queries system information.
///
/// @param [out] pSystemInfo SystemInfo struct containing information about the system.
///
/// @returns Success if querying the system info was successful. Otherwise, the following results will be returned:
///          + ErrorInvalidPointer returned if pSystemInfo is nullptr.
///          + ErrorOutOfMemory returned if the system ran out of memory during the function call.
///          + ErrorUnavailable returned if querying the system info is not supported.
///          + ErrorUnknown returned if an error occurs while calling OS functions.
extern Result QuerySystemInfo(SystemInfo* pSystemInfo);

/// Query cpu type for AMD processor.
///
/// @param [out] pSystemInfo SystemInfo struct containing information about the system.
///
/// @returns none.
extern void QueryAMDCpuType(SystemInfo* pSystemInfo);

/// Query cpu type for Intel processor.
///
/// @param [out] pSystemInfo SystemInfo struct containing information about the system.
///
/// @returns none.
extern void QueryIntelCpuType(SystemInfo* pSystemInfo);

/// Gets the frequency of performance-related queries.
///
/// @returns Current CPU performance counter frequency in Hz.
extern int64 GetPerfFrequency();

/// Gets the current time of a performance-related query.
///
/// This is a high resolution time stamp that can be used in conjunction with GetPerfFrequency to measure time
/// intervals.
///
/// @param [in]      raw        Whether to use a 'monotonic raw' clock which ignores smoothing. Ignored on Windows.
///
/// @returns Current value of the CPU performance counter.
extern int64 GetPerfCpuTime(bool raw=false);

/// Determines if a specific key is pressed down.
///
/// @param [in]      key        Specified which key to check.
/// @param [in, out] pPrevState The previous state of the key.
///
/// @returns True if the specified key is currently pressed down.
extern bool IsKeyPressed(KeyCode key, bool* pPrevState = nullptr);

/// Determines if profiling is restricted
///
/// @returns true if the process is not restricted for profiling, otherwise, false will be returned.
extern bool IsProfileRestricted();

/// Retrieves the fully resolved file name of the application binary.
///
/// @param [out] pBuffer      Character buffer to contain the application's executable and (fully-resolved) path
///                           string.
/// @param [out] ppFilename   Pointer to the location within the output buffer where the executable name begins.
/// @param [in]  bufferLength Length of the output buffer, in bytes.
/// @returns Result::Success if GetModuleFileNameA succeeds. Otherwise, the following result codes would be returned:
///          + Result::ErrorInvalidMemorySize returned if pBuffer is not sufficiently large.
extern Result GetExecutableName(
    char*  pBuffer,
    char** ppFilename,
    size_t bufferLength);

/// Retrieves the fully resolved wchar_t file name of the application binary.
///
/// @param [out] pWcBuffer    wchar_t buffer to contain the application's executable and (fully-resolved) path
///                           string.
/// @param [out] ppWcFilename Pointer to the location within the wchar_t output buffer where the executable name begins.
/// @param [in]  bufferLength Length of the output buffer, in bytes.
/// @returns Result::Success if GetModuleFileNameW succeeds. Otherwise, the following result codes would be returned:
///          + Result::ErrorInvalidMemorySize returned if pBuffer is not sufficiently large.
extern Result GetExecutableName(
    wchar_t*  pWcBuffer,
    wchar_t** ppWcFilename,
    size_t    bufferLength);

/// Gets the current library name. ie: the name of the library containing the function
/// 'GetCurrentLibraryName'. Optionally, it will also return the extension if the input
/// buffer for extension is valid.
///
/// @param [out] pLibBuffer      Character buffer where the library name will be stored.
/// @param [in]  libBufferLength Length of the output buffer that will hold the library name, in bytes.
/// @param [out] pExtBuffer      Character buffer where the extension will be stored.
/// @param [in]  extBufferLength Length of the output buffer that will hold the extension, in bytes.
/// @returns Result::Success if no error. Otherwise, returns one of the following codes:
///          Result::ErrorInvalidMemorySize - if incoming buffer is too small.
///          Result::ErrorUnknown - for all other types of errors.
extern Result GetCurrentLibraryName(
    char*  pLibBuffer,
    size_t libBufferLength,
    char*  pExtBuffer,
    size_t extBufferLength);

/// Opaque build ID obtained with GetCurrentLibraryBuildId
struct BuildId {
    uint8 data[16];
};

/// Gets build-unique identifier for the executable or shared library PAL was built into. This will be some opaque
/// hash or timestamp embedded in the code if present and falls back to random bytes (constant until exit) for a
/// 'usuable' build id to always exist.
///
/// @param [out] pBuildId Will contain unique id for build
///
/// @returns true if build id will be persistent, false if temporary
///
/// @note We fallback to using a random build id here instead of failing so that even if this fails and nobody checks,
///       a new driver will always invalidate caches. It is better, of course, not to persist these on disk.
/// @note This may be heavy on first invocation but can be assumed to always return quickly after that.
extern bool GetCurrentLibraryBuildId(
    BuildId* pBuildId);

/// Splits a filename into its path and file components.
///
/// @param [in]  pFullPath  Buffer containing the full path & file name.
/// @param [out] pPathBuf   Optional.  If non-null, will contain the path to the file name.  On Windows, this will also
///                         include the drive letter.
/// @param [in]  pathLen    Length of the pPathBuf buffer.  Must be zero when pPathBuf is null.
/// @param [out] pFileBuf   Optional.  If non-null, will contain the base file name, and extension.
/// @param [in]  fileLen    Length of the pFileBuf buffer.  Must be zero when pFileBuf is null.
extern void SplitFilePath(
    const char* pFullPath,
    char*       pPathBuf,
    size_t      pathLen,
    char*       pFileBuf,
    size_t      fileLen);

/// Creates a new directory at the specified path.
///
/// @param [in] pPathName String specifying the new path to create.  Note that this method can only create one
///                       directory, if you specify "foo/bar" the "bar" directory can only be created if "foo" already
///                       exists.
/// @returns Result::Success if the directory was successfully created, otherwise an appropriate error.  Otherwise, the
///          following result codes may be returned:
///          + Result::AlreadyExists if the specified directory already exists.
///          + Result::ErrorInvalidValue if the parent directory does not exist.
extern Result MkDir(
    const char* pPathName);

/// Creates a new directory at the specified path and all intermediate directories.
///
/// @param [in] pPathName String specifying the new path to create.n
///
/// @returns Result::Success if the directory was successfully created, otherwise an appropriate error.  Otherwise, the
///          following result codes may be returned:
///          + Result::AlreadyExists if the specified directory already exists.
///          + Result::ErrorInvalidValue if the parent directory does not exist.
extern Result MkDirRecursively(
    const char* pPathName);

/// A tuple containing a file's name and statistics
struct StatName
{
    File::Stat stat;
    char name[MaxPathStrLen];
};

/// Gets file information for the files in a directory
///
/// @param [in]  dirPath    string specifying the directory
/// @param [out] pFileInfos list of information on every file in the directory
///
/// @returns Result::ErrorInvalidPointer if any of the input pointers are null
/// @returns Util::ConvertWinError(GetLastError()) if there are any file I/O errors on Windows
/// @returns Result::ErrorInvalidValue if there for all file I/O errors on Linux
/// @returns Result::Success if the dir is empty
/// @returns Result::Success otherwise
extern Result GetFileInfoInDir(
    StringView<char>                       dirPath,
    Vector<StatName, 1, GenericAllocator>* pFileInfos);

/// Counts the number of files found within the directory.
///
/// @param [in]  pDirPath   string specifying the directory
/// @param [out] pFileCount the number of files in the directory
/// @param [out] pCharCount the number of characters in the names of all the files
///
/// @returns Result::ErrorInvalidPointer if any of the input pointers are null
/// @returns Value of Util::ConvertWinError(GetLastError()) if there are any file I/O errors on Windows
/// @returns Result::ErrorInvalidValue if there for all file I/O errors on Linux
/// @returns Result::Success if the dir is empty (pFileCount and pCharCount will be 0)
/// @returns Result::Success otherwise
extern Result CountFilesInDir(
    Util::StringView<char>  dirPath,
    size_t*                 pFileCount,
    size_t*                 pCharCount);

/// Lists the contents of the specified directory in an array of strings
///
/// @param [in]  dirPath    String specifying the directory
/// @param [out] fileNames  An array where pointers the file names will be written.
/// @param [out] buffer     Memory where the file names can be stored.
///
/// @returns Result::ErrorInvalidPointer if any of the inputs are null or empty
/// @returns Result::ErrorInvalidValue if there are any file I/O errors
/// @returns Result::Success otherwise
extern Result GetFileNamesInDir(
    Util::StringView<char>              dirPath,
    Util::Span<Util::StringView<char>>  fileNames,
    Util::Span<char>                    buffer);

/// Non-recursively delete the least-recently-accesssed files from a directory until the directory reaches size in bytes.
///
/// @param [in] pPathName   string specifying the absolute path to the directory you want to remove files from
/// @param      desiredSize the size you want to shrink the directory to
///
/// @returns Result::ErrorUnknown on File I/O error.
///          Result::Success otherwise.
Result RemoveOldestFilesOfDirUntilSize(
    const char* pPathName,
    uint64      desiredSize);

/// Non-recursively delete the least-recently-accesssed files until the total reaches size in bytes.
///
/// @param [in]     pDirPath    String specifying the directory.
/// @param [in/out] pFileInfos  List of files. This span will be modified to remove the files which were deleted.
/// @param          desiredSize The size you want to shrink the list of files to.
///
/// @returns Result::ErrorUnknown on File I/O error.
///          Result::Success otherwise.
Result RemoveOldestFilesOfDirUntilSize(
    StringView<char> dirPath,
    Span<StatName>*  pFileInfos,
    uint64           desiredSize);

/// Remove all files below threshold of a directory at the specified path.
///
/// @param [in] pPathName String specifying the absolute path to remove.
/// @param [in] threshold The file time(from 1970/01/01 00:00:00) older(smaller) than threshold will be removed.
///
/// @returns Result::Success if all files are successfully removed. Otherwise, the
///          following result codes may be returned:
///          + Result::ErrorUnknown if the specified directory is failed to open/remove.
///          + Result::ErrorInvalidValue if the parent directory does not exist.
Result RemoveFilesOfDirOlderThan(
    const char*       pPathName,
    SecondsSinceEpoch threshold);

/// Get status of a directory at the specified path.
///
/// @param [in] pPathName String specifying the absolute path.
/// @param [out] pTotalSize Size(byte) of all files
/// @param [out] pOldestTime The oldest time(seconds from 1970/01/01 00:00:00) of all files
///
/// @returns Result::Success if all files are successfully removed. Otherwise, the
///          following result codes may be returned:
///          + Result::ErrorUnknown if the specified directory is failed to open.
Result GetStatusOfDir(
    const char*        pPathName,
    uint64*            pTotalSize,
    SecondsSinceEpoch* pOldestTime);

/// Almost-Posix-style rename file or directory: replaces already-existing file.
/// Posix says this operation is atomic; Windows does not specify.
///
/// @param [in] pOldName Old file or directory name
/// @param [in] pNewName Name to rename to
///
/// @returns Result::Success if file/directory successfully moved.
Result Rename(
    const char* pOldName,
    const char* pNewName);

/// Get the Process ID of the current process
///
/// @returns The Process ID of the current process
extern uint32 GetIdOfCurrentProcess();

/// OS-specific wrapper for printing stack trace information.
///
/// @param [out] pOutput    Output string. If buffer is a nullptr it returns the length of the string that would be
///                         printed had a buffer with enough space been provided.
/// @param [in]  bufSize    Available space in pOutput.
/// @param [in]  skipFrames Number of stack frames to skip. Implied skip of 1 (0 is 1).
///
/// @returns The resultant length of the stack trace string.
extern size_t DumpStackTrace(
    char*   pOutput,
    size_t  bufSize,
    uint32  skipFrames);

/// Puts the calling thread to sleep for a specified number of milliseconds.
///
/// @param [in] duration  Amount of time to sleep for, in milliseconds.
extern void Sleep(std::chrono::milliseconds duration);

/// Create Directory and SubDirectory of Executable's Name at pBaseDir
/// Like if pBaseDir="amdPal/" then final pLogDir will be created like this: amdpal/app.exe/
///
/// @param [in] pBaseDir        Base Dir Path
/// @param [out] pLogDir        Pointer to Memory where Final Path will be saved
/// @param [in] logDirSize      Size of pLogDir in chars
///
/// @returns Result::Success if the directories were created and pLogDir was populated.
extern Result CreateLogDir(
    const char* pBaseDir,
    char*       pLogDir,
    size_t      logDirSize);

/// Check if the requested key is combo key.
///
/// @param [in]  key    The requested key value
/// @param [out] pKeys  The array of keys the combo key composed of
///
/// @returns If the requested key is a combo key.
inline bool IsComboKey(
    KeyCode key,
    KeyCode* pKeys)
{
    bool ret = false;

    if (key == KeyCode::Shift_F10)
    {
        ret = true;
        pKeys[0] = KeyCode::Shift;
        pKeys[1] = KeyCode::F10;
    }
    else if (key == KeyCode::Shift_F11)
    {
        ret = true;
        pKeys[0] = KeyCode::Shift;
        pKeys[1] = KeyCode::F11;
    }
    else
    {
        pKeys[0] = key;
    }

    return ret;
}

#if PAL_HAS_CPUID
/// Issue the cpuid instruction.
///
/// @param [out]  pRegValues  EAX/EBX/ECX/EDX values
/// @param [in]   level       CpuId instruction feature level.
inline void CpuId(
    uint32* pRegValues,
    uint32 level)
{
#if defined(_WIN32)
    __cpuid(reinterpret_cast<int32*>(pRegValues), static_cast<int32>(level));
#elif defined(__unix__)
    __get_cpuid(level, pRegValues, pRegValues + 1, pRegValues + 2, pRegValues + 3);
#else
#error "Not implemented for the current platform"
#endif
}

/// Issue the cpuid instruction, with an additional sublevel code.
///
/// @param [out]  pRegValues  EAX/EBX/ECX/EDX values
/// @param [in]   level       CpuId instruction feature level.
/// @param [in]   sublevel    CpuId instruction feature sublevel.
inline void CpuId(
    uint32* pRegValues,
    uint32 level,
    uint32 sublevel)
{
#if defined(_WIN32)
    __cpuidex(reinterpret_cast<int32*>(pRegValues), static_cast<int32>(level), static_cast<int32>(sublevel));
#elif defined(__unix__)
    __cpuid_count(level, sublevel, *pRegValues, *(pRegValues + 1), *(pRegValues + 2), *(pRegValues + 3));
#else
#error "Not implemented for the current platform"
#endif
}
#endif

/// Play beep sound. Currently function implemented only for WIN platform.
///
/// @param [in]  frequency  Frequency in hertz of the beep sound.
/// @param [in]  duration   Duration in milliseconds of the beep sound.
extern void BeepSound(
    uint32 frequency,
    uint32 duration);

/// Detect if a debugger is attached to the current process.
///
/// @note This function is conservative, declaring the debugger not attached on error.
///
/// @returns true if there is a debugger attached.
extern bool IsDebuggerAttached();

/// Set path to be accessible by everyone.
///
/// In Linux world that is 777. For Windows it does nothing.
/// To be used for debug folders mainly. Use carefully for release paths as this will open them wide
/// and possibly create a security hole.
///
/// @param [in] fileName  Path to a file or folder, it must exist.
///
/// @returns Returns success if permissions were set successfully, Result::ErrorUnknown otherwise
extern Result SetRwxFilePermissions(const char* pFileName);

} // Util
