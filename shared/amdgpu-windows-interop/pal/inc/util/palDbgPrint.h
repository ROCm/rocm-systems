/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  palDbgPrint.h
 * @brief Defines PAL utility collection debug print functionality.
 ***********************************************************************************************************************
 */

#pragma once

#include "palUtil.h"
#include <stdarg.h>

#if PAL_ENABLE_LOGGING
#include "palDbgLogHelper.h"
#endif

namespace Util
{
#if (PAL_ENABLE_PRINTS_ASSERTS || PAL_ENABLE_LOGGING)
// Forward declarations.
class File;
enum  FileAccessMode : uint32;

/// Specifies the category of a debug print.
///
/// Driver developer can enable/disable each category separately through settings.
enum DbgPrintCategory : uint32
{
    DbgPrintCatInfoMsg = 0,             ///< Generic informational messages.
    DbgPrintCatWarnMsg,                 ///< Warning messages.
    DbgPrintCatErrorMsg,                ///< Error messages.
    DbgPrintCatScMsg,                   ///< Generic shader compiler messages.
    DbgPrintCatEventPrintMsg,           ///< System event messages.
    DbgPrintCatEventPrintCallbackMsg,   ///< System event messages via callback.
    DbgPrintCatMsgFile,                 ///< All Messages via file.
    DbgPrintCatCount
};

/// Specifies the debug print mode: disabled, print to debugger, or print to file.
enum class DbgPrintMode : uint32
{
    Disable,         ///< Debug print is ignored.
    Print,           ///< Debug print is routed to the debug window or stdout.
    File,            ///< Debug print is routed to a file.
    PrintCallback,   ///< Debug print is routed to the print callback only
};

/// Flags specifying style controls for a debug print.
enum DbgPrintStyle : uint32
{
    DbgPrintStyleDefault        = 0x0,  ///< Normal mode: has a prefix and a CR-LF.
    DbgPrintStyleNoPrefix       = 0x1,  ///< Skip the prefix.
    DbgPrintStyleNoCrLf         = 0x2,  ///< Skip the CR-LF.
    DbgPrintStyleNoPrefixNoCrLf = 0x3,  ///< Skip both the prefix and the CR-LF.
};

/// Definition for debug print callback.
///
/// @param [in] pUserData User data that is installed with the callback for use by the installer.
/// @param [in] category  Debug print category that the message belongs to.
/// @param [in] pText     Text data to be printed by the callback.
typedef void (PAL_STDCALL *DbgPrintCallbackFunc)(
    void*            pUserdata,
    DbgPrintCategory category,
    const char*      pText);

/// Debug print callback struct that bundles the callback function and its userdata pointer.
struct DbgPrintCallback
{
    DbgPrintCallbackFunc pCallbackFunc;
    void*                pUserdata;
};

/// Generic debug printf function to be used when the caller wishes to specify the output category and style.  Clients
/// should use the PAL_DPF macro instead of calling this function directly.
///
/// @param [in] category Message category (e.g., CS dumps, SC output, etc.).
/// @param [in] style    Text output style (i.e., has prefix and/or CR-LF).
/// @param [in] pFormat  Printf-style format string.
extern void DbgPrintf(
    DbgPrintCategory category,
    DbgPrintStyle    style,
    const char*      pFormat,
    ...);

/// Generic printf function to be used when the caller wishes to specify the output category and style, and has
/// pre-started the variable arg list (va_list argument instead of ...).
///
/// @param [in] category Message category (e.g., CS dumps, SC output, etc.).
/// @param [in] style    Text output style (i.e., has prefix and/or CR-LF).
/// @param [in] pFormat  Printf-style format string.
/// @param [in] argList  Variable argument list.
extern void DbgVPrintf(
    DbgPrintCategory category,
    DbgPrintStyle    style,
    const char*      pFormat,
    va_list          argList);

#endif

#if PAL_ENABLE_PRINTS_ASSERTS
/// Sets the debug print mode (output to debugger, write to file, or disabled) for the specified category of messages.
///
/// Probably controlled by a setting and set during initialization.
///
/// @param [in] category Message category to control (e.g., CS dumps, SC output, etc.).
/// @param [in] mode     New mode to be used for this message category (print to file, etc.).
extern void SetDbgPrintMode(
    DbgPrintCategory category,
    DbgPrintMode     mode);

/// Opens a file that resides in the selected log directory.
///
/// This function exists in all build configurations.
///
/// @param [in,out] pFile     File object to represent the opened file.
/// @param [in]     pFilename Filename to open.
/// @param [in]     flags     ORed mask of FileAccessMode values specifying how this file will be accessed.
///
/// @returns Success if successful, otherwise an appropriate error.
extern Result OpenLogFile(
    File*       pFile,
    const char* pFilename,
    uint32      flags);

/// Sets the global debug print callback.
///
/// @param [in] callback  Debug print callback struct that contains the callback function and a userdata pointer
extern void SetDbgPrintCallback(
    const DbgPrintCallback& callback);

#endif

// Forward declarations.
template<typename CharT> class StringView;

/// Logs a text string via client callback when provided.
///
/// @param [in] pClientData  Pointer to client-defined data.  The pClientData value specified in the pLogCbInfo
///                          parameter to CreatePlatform() will be passed back to the client on every log callback.
/// @param [in] level        Log priority level associated with the message.
/// @param [in] categoryMask Log category mask that represents what category fields the message relates to.
/// @param [in] pFormat      Format string for the log message.
/// @param [in] args         Variable arguments that correspond to the format string.
typedef void (PAL_STDCALL *LogCallbackFunc)(
    void*       pClientData,
    uint32      level,
    uint64      categoryMask,
    const char* pFormat,
    va_list     args);

/// Specifies client-provided logging callbacks.  Used as a parameter to Pal::CreatePlatform().
///
/// @ingroup LibInit
struct LogCallbackInfo
{
    void*           pClientData;  ///< Opaque pointer to data of client's choosing.  This pointer will be passed back to
                                  ///  every @ref LogCallbackFunc call made by PAL.
    LogCallbackFunc pfnLogCb;     ///< Debug print logging callback.  @see LogCallbackFunc.
};

/// Compiler-specific wrapper of the standard snprintf implementation.
///
/// @param [out] pOutput Output string.
/// @param [in]  bufSize Available space in pOutput.
/// @param [in]  pFormat Printf-style format string.
///
/// @returns The resultant length of the formatted string.
extern int32 Snprintf(
    char*       pOutput,
    size_t      bufSize,
    const char* pFormat,
    ...);

/// Compiler-specific wrapper of the standard vsnprintf implementation.
///
/// @param [out] pOutput Output string. If buffer is a nullptr it returns the length of the string that would be
///                      printed had a buffer with enough space been provided.
/// @param [in]  bufSize Available space in pOutput.
/// @param [in]  pFormat Printf-style format string.
/// @param [in]  argList variable argument list.
///
/// @returns The resultant length of the formatted string.
extern int32 Vsnprintf(
    char*       pOutput,
    size_t      bufSize,
    const char* pFormat,
    va_list     argList);

/// Compiler-specific wrapper of the standard snprintf implementation.
///
/// @param [out] pOutput Output string.
/// @param [in]  bufSize Available space in pOutput.
/// @param [in]  pFormat Printf-style format string.
///
/// @returns The resultant length of the formatted string.
extern int32 Snprintf(
    wchar_t*       pOutput,
    size_t         bufSize,
    const wchar_t* pFormat,
    ...);

/// Compiler-specific wrapper of the standard vsnprintf implementation.
///
/// @param [out] pOutput Output string. If buffer is a nullptr it returns the length of the string that would be
///                      printed had a buffer with enough space been provided.
/// @param [in]  bufSize Available space in pOutput.
/// @param [in]  pFormat Printf-style format string.
/// @param [in]  argList variable argument list.
///
/// @returns The resultant length of the formatted string.
extern int32 Vsnprintf(
    wchar_t*       pOutput,
    size_t         bufSize,
    const wchar_t* pFormat,
    va_list        argList);

/// Copy an arbitrary string into the provided buffer, encoding as necessary to avoid characters that are illegal
/// in filenames (assuming the more restrictive Windows rules, even on non-Windows OSs).
///
/// Any byte that would be illegal is encoded as % then two hex digits, like in a URL.
///
/// @param [out] pOutput           Output string.
/// @param       bufSize           Available space in pOutput.
/// @param [in]  input             Input string
/// @param       allowSpace        Allow (do not % encode) space
/// @param       allowDirSeparator Allow (do not % encode) / and \ characters
///
/// @returns Works like C++ standard snprintf:
///          - If the provided buffer is big enough, it returns the number of bytes written, excluding the
///            terminating \0.
///          - If the provided buffer is not big enough, then the result string is truncated to fit, and the
///            function returns the number of bytes that would have been written if the buffer had been long
///            enough, excluding the terminating \0.
///          - Passing 0 buffer length is allowed as a special case of that, and nullptr pOutput is then allowed.
extern size_t EncodeAsFilename(
    char*                   pOutput,
    size_t                  bufSize,
    const StringView<char>& input,
    bool                    allowSpace,
    bool                    allowDirSeparator);

/// Generate a log filename.
///
/// @param [inout] pFilenameBuffer   Buffer to hold the filename.
/// @param         maxSize           Max size of the pFilenameBuffer.
/// @param         nextPost          The next write position.
/// @param [in]    pExt              The filename extension.
/// @param         logDuplicate      Log duplicate objects.
extern void GenLogFilename(
    char*             pFilenameBuffer,
    size_t            maxSize,
    size_t            nextPos,
    const char* const pExt,
    bool              logDuplicate);

} // Util

/// PAL_ENABLE_LOGGING enables the new logging code. At this time, both, the current and new logging
/// code will be active for development purpose if both macros are enabled.
#if (PAL_ENABLE_PRINTS_ASSERTS && PAL_ENABLE_LOGGING)
/// Debug printf macro.
#define PAL_DPF ::Util::DbgPrintf
/// Debug info printf macro.
#define PAL_DPINFO(_pFormat, ...)                                                                                   \
{                                                                                                                   \
    ::Util::DbgPrintf(::Util::DbgPrintCatInfoMsg, ::Util::DbgPrintStyleDefault, _pFormat " (%s:%d:%s)",             \
                    ##__VA_ARGS__, __FILE__, __LINE__, __func__);                                                   \
    ::Util::DbgLog(::Util::SeverityLevel::Info, ::Util::OriginationType::DebugPrint,                                \
                 "AMD-PAL", _pFormat " (%s:%d:%s)",  ##__VA_ARGS__, __FILE__, __LINE__, __func__);                  \
}
/// Debug warning printf macro.
#define PAL_DPWARN(_pFormat, ...)                                                                                   \
{                                                                                                                   \
    ::Util::DbgPrintf(::Util::DbgPrintCatWarnMsg, ::Util::DbgPrintStyleDefault, _pFormat " (%s:%d:%s)",             \
                    ##__VA_ARGS__, __FILE__, __LINE__, __func__);                                                   \
    ::Util::DbgLog(::Util::SeverityLevel::Warning, ::Util::OriginationType::DebugPrint,                             \
                 "AMD-PAL", _pFormat " (%s:%d:%s)", ##__VA_ARGS__, __FILE__, __LINE__, __func__);                   \
}
/// Debug error printf macro.
#define PAL_DPERROR(_pFormat, ...)                                                                                  \
{                                                                                                                   \
    ::Util::DbgPrintf(::Util::DbgPrintCatErrorMsg, ::Util::DbgPrintStyleDefault, _pFormat " (%s:%d:%s)",            \
                    ##__VA_ARGS__, __FILE__, __LINE__, __func__);                                                   \
    ::Util::DbgLog(::Util::SeverityLevel::Error, ::Util::OriginationType::DebugPrint,                               \
                 "AMD-PAL", _pFormat " (%s:%d:%s)", ##__VA_ARGS__, __FILE__, __LINE__, __func__);                   \
}
#elif PAL_ENABLE_PRINTS_ASSERTS
/// Debug printf macro.
#define PAL_DPF ::Util::DbgPrintf
/// Debug info printf macro.
#define PAL_DPINFO(_pFormat, ...)                                                                                   \
{                                                                                                                   \
    ::Util::DbgPrintf(::Util::DbgPrintCatInfoMsg, ::Util::DbgPrintStyleDefault, _pFormat " (%s:%d:%s)",             \
                    ##__VA_ARGS__, __FILE__, __LINE__, __func__);                                                   \
}
/// Debug warning printf macro.
#define PAL_DPWARN(_pFormat, ...)                                                                                   \
{                                                                                                                   \
    ::Util::DbgPrintf(::Util::DbgPrintCatWarnMsg, ::Util::DbgPrintStyleDefault, _pFormat " (%s:%d:%s)",             \
                    ##__VA_ARGS__, __FILE__, __LINE__, __func__);                                                   \
}
/// Debug error printf macro.
#define PAL_DPERROR(_pFormat, ...)                                                                                  \
{                                                                                                                   \
    ::Util::DbgPrintf(::Util::DbgPrintCatErrorMsg, ::Util::DbgPrintStyleDefault, _pFormat " (%s:%d:%s)",            \
                    ##__VA_ARGS__, __FILE__, __LINE__, __func__);                                                   \
}
#elif PAL_ENABLE_LOGGING
/// Debug printf macro.
#define PAL_DPF ::Util::DbgPrintf
/// Debug info printf macro.
#define PAL_DPINFO(_pFormat, ...)                                                                                   \
{                                                                                                                   \
    ::Util::DbgLog(::Util::SeverityLevel::Info, ::Util::OriginationType::DebugPrint,                                \
                 "AMD-PAL", _pFormat " (%s:%d:%s)", ##__VA_ARGS__, __FILE__, __LINE__, __func__);                   \
}
/// Debug warning printf macro.
#define PAL_DPWARN(_pFormat, ...)                                                                                   \
{                                                                                                                   \
    ::Util::DbgLog(::Util::SeverityLevel::Warning, ::Util::OriginationType::DebugPrint,                             \
                 "AMD-PAL", _pFormat " (%s:%d:%s)", ##__VA_ARGS__, __FILE__, __LINE__, __func__);                   \
}
/// Debug error printf macro.
#define PAL_DPERROR(_pFormat, ...)                                                                                  \
{                                                                                                                   \
    ::Util::DbgLog(::Util::SeverityLevel::Error, ::Util::OriginationType::DebugPrint,                               \
                 "AMD-PAL", _pFormat " (%s:%d:%s)", ##__VA_ARGS__, __FILE__, __LINE__, __func__);                   \
}
#else
/// Debug printf macro.
#define PAL_DPF(...)     ((void)0)
/// Debug info printf macro.
#define PAL_DPINFO(...)  ((void)0)
/// Debug warning printf macro.
#define PAL_DPWARN(...)  ((void)0)
/// Debug error printf macro.
#define PAL_DPERROR(...) ((void)0)
#endif

