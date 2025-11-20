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
 * @file  palFile.h
 * @brief PAL utility collection File class declaration.
 ***********************************************************************************************************************
 */

#pragma once

// pal
#include "palUtil.h"
#include "palInlineFuncs.h"

// stl
#include <chrono>
#include <cstdio>

#if defined(_WIN32)
/// Macro for wide string literal concatenation.
#define PAL_PATH_SEPW L"\\"
/// Macro for narrow string literal concatenation.
#define PAL_PATH_SEP "\\"
#else
/// Macro for wide string literal concatenation.
#define PAL_PATH_SEPW L"/"
/// Macro for narrow string literal concatenation.
#define PAL_PATH_SEP "/"
#endif

namespace Util
{
#if defined(_WIN32)
/// Wide-character of the platform's prefered path separator.
static constexpr wchar_t PathSepW = L'\\';
/// Narrow-character of the platform's prefered path separator.
static constexpr  char   PathSep = '\\';
#else
/// Wide-character of the platform's prefered path separator.
static constexpr wchar_t PathSepW = L'/';
/// Narrow-character of the platform's prefered path separator.
static constexpr  char   PathSep = '/';
#endif

static constexpr uint32 MaxPathStrLen = 512;
static constexpr uint32 MaxFileNameStrLen = 256;

/// Enumerates access modes that may be required on an opened file.
/// Can be bitwise ORed together to specify multiple simultaneous modes.
enum FileAccessMode : uint32
{
    FileAccessRead      = 0x1,  ///< Read access.
    FileAccessWrite     = 0x2,  ///< Write access.
    FileAccessAppend    = 0x4,  ///< Append access.
    FileAccessBinary    = 0x8,  ///< Binary access.
    FileAccessNoDiscard = 0x10, ///< Don't discard existing file.
    FileAccessShared    = 0x20, ///< Require shared file access (simultaneous reading/writing by more than one process)
};

/**
 ***********************************************************************************************************************
 * @brief Exposes simple file I/O functionality by encapsulating standard C runtime file I/O functions like fopen,
 *        fwrite, etc.
 ***********************************************************************************************************************
 */
class File
{
public:
    // Platform-agnostic 64-bit stat structure.
    struct Stat
    {
        uint64                                size;  // Size of the file in bytes.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 922
        uint64                                ctime; // Time of creation of the file (not valid on FAT).
        uint64                                atime; // Time of last access to the file (not valid on FAT).
        uint64                                mtime; // Time of last modification to the file.
#else
        std::chrono::system_clock::time_point ctime; // Time of creation of the file (not valid on FAT).
        std::chrono::system_clock::time_point atime; // Time of last access to the file (not valid on FAT).
        std::chrono::system_clock::time_point mtime; // Time of last modification to the file.
#endif
        uint32                                nlink; // Number of hard links (always 1 on FAT on Windows).
        uint32                                mode;  // Bitmask for the file-mode information.
        uint32                                dev;   // Drive number of the disk containing the file.

        union
        {
            struct
            {
                uint32  isDir       :  1;
                uint32  isRegular   :  1;
                uint32  reserved    : 30;
            };
            uint32 u32All;
        } flags;

        // Common stat members omitted from this structure:
        // uid, gid, and ino because it's not used on Windows
        // rdev because it's a duplicate of dev
    };

    // Where in the file to start seeking from.
    enum class SeekPosition : int32
    {
        // start of the file
        Start   = SEEK_SET,
        // current file pointer position
        Current = SEEK_CUR,
        // end of the file
        End     = SEEK_END
    };

    File() : m_pFileHandle(nullptr), m_ownsHandle(false) {}

    /// Closes the file if it is still open.
    ~File() { Close(); }

    /// Opens a file stream for read, write or append access.
    ///
    /// @param [in] pFilename   Name of file to open.
    /// @param [in] accessFlags Bitmask of FileAccessMode values indicating the usage of the file.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result Open(const char* pFilename, uint32 accessFlags);

    /// Borrows an externally opened C runtime file handle for use by a File object.
    ///
    /// The caller is still responsible for closing this handle after the File object is destroyed.
    ///
    /// @param [in] pFile Externally opened C runtime file handle to borrow.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result FromNative(std::FILE* pFile);

    /// Closes the file handle.
    void Close();

    /// Writes a stream of bytes to the file.
    ///
    /// @param [in] pBuffer    Byte stream to be written to the file.
    /// @param [in] bufferSize Number of bytes to write.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result Write(const void* pBuffer, size_t bufferSize);

    /// Reads a stream of bytes from the file.
    ///
    /// @param [out] pBuffer    Buffer to be written with data read from file.
    /// @param [in]  bufferSize Size of the output buffer.
    /// @param [out] pBytesRead Number of bytes actually read (can be null).
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result Read(void* pBuffer, size_t bufferSize, size_t* pBytesRead);

    /// Reads a single line of bytes from the file.
    ///
    /// @param [out] pBuffer    Buffer to be written with data read from file.
    /// @param [in]  bufferSize Size of the output buffer.
    /// @param [out] pBytesRead Number of bytes actually read (can be null).
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result ReadLine(void* pBuffer, size_t bufferSize, size_t* pBytesRead);

    /// Prints a formatted string to the file.
    ///
    /// @param [in] pFormatStr Printf-style format string.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result Printf(const char* pFormatStr, ...) const;

    /// Prints a formatted string to the file.
    ///
    /// @param [in] pFormatStr Printf-style format string.
    /// @param [in] argList    Variable argument list.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result VPrintf(const char* pFormatStr, va_list argList);

    /// Flushes pending I/O to the file.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    Result Flush() const;

    /// Sets the file position to the beginning of the file.
    void Rewind();

    /// Sets the position indicator to a new position.
    ///
    /// @param  offset      Number of bytes to offset
    /// @param  pos         File position to seek from
    void Seek(int64 offset, SeekPosition pos);

    /// Sets the position indicator to a new position relative to the beginning of the file.
    ///
    /// @param  offset      Number of bytes to offset
    void Seek(size_t offset) { Seek(offset, SeekPosition::Start); }

    /// Sets the position indicator to a new position relative to the end of the file
    ///
    /// @param  offset      Number of bytes to offset
    void Rseek(size_t offset) { Seek(-static_cast<int64>(offset), SeekPosition::End); }

    /// Sets the file position to the end of the file.
    void FastForward() { Rseek(0); }

    /// Returns true if the file is presently open.
    bool IsOpen() const { return (m_pFileHandle != nullptr); }

    /// Gets the size of the file contents in bytes
    ///
    /// @param [in] pFilename Name of the file to check.
    ///
    /// @returns Size of the file in bytes, or std::numeric_limits<size_t>::max() on failure.
    static size_t GetFileSize(const char* pFilename);

    /// Checks if a file with the specified name exists.
    ///
    /// @param [in] pFilename Name of the file to check.
    ///
    /// @returns True if the specified file exists.
    static bool Exists(const char* pFilename);

    /// Platform-agnostic 64-bit stat() function.
    ///
    /// @param [in]     pFilename   Name of the file to check.
    /// @param [out]    pStatus     The status of that file, if it exists.
    ///
    /// @returns Success if the structure was retrieved, error otherwise.
    static Result GetStat(const char* pFilename, Stat* pStatus);

    /// Removes/erases a file, if it exists.
    ///
    /// @param [in] pFilename  Name of file to remove.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    static Result Remove(const char* pFilename);

    /// Reads a file into memory.
    ///
    /// @param [in]  pFilename  Name of the file to read.
    /// @param [in]  pData      Buffer where the file contents are written to.
    /// @param [in]  dataSize   Size of the buffer in bytes.
    /// @param [out] pBytesRead Number of bytes successfully read into the input buffer (can be null).
    /// @param [in]  binary     True for binary mode, false for text. Defaults to binary.
    ///
    /// @returns Success if successful, otherwise an appropriate error.
    ///
    /// @note The input buffer must be large enough to hold the file's contents. If the buffer is larger than the file,
    /// then the region of the buffer beyond the file size is _not_ modified by this function. It is the caller's
    /// responsibility to _not_ read uninitialized portions of the supplied buffer after this call returns.
    ///
    /// @note In binary mode, the number of bytes read is equal to the file size in bytes upon a successful return.
    /// In text mode, newline conversion is performed on Windows, in which case the number of bytes read may not equal
    /// the file size in bytes.
    ///
    /// @note In text mode, should the caller treat the resulting data as a C string, it is the caller's responsibility
    /// to null-terminate the buffer.
    static Result ReadFile(
        const char* pFilename,
        void*       pData,
        size_t      dataSize,
        size_t*     pBytesRead = nullptr,
        bool        binary = true);

    /// Gets the handle associated with this file.
    ///
    /// @returns A pointer to the file handle
    const std::FILE* GetHandle() const { return m_pFileHandle; }

private:
    std::FILE* m_pFileHandle;
    bool       m_ownsHandle; // This object owns the file handle and will close it on destruction.

    PAL_DISALLOW_COPY_AND_ASSIGN(File);
};

} // Util
