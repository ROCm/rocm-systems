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
 * @file  palUtil.h
 * @brief Common include for the PAL utility collection.  Defines common types, macros, enums, etc.
 ***********************************************************************************************************************
 */

#pragma once

/// Utility macro for turning another macro into a string literal.
#define _PAL_STRINGIFY(_x) #_x
#define PAL_STRINGIFY(_x) _PAL_STRINGIFY(_x)

/// C++11 standard version.
#define PAL_CPLUSPLUS_11 201103L
/// C++14 standard version.
#define PAL_CPLUSPLUS_14 201402L
/// C++17 standard version.
#define PAL_CPLUSPLUS_17 201703L
/// C++ feature version from September 2017 contains a few C++20 features.
#define PAL_CPLUSPLUS_1709 201709L
/// C++20 standard version.
#define PAL_CPLUSPLUS_20 202002L

/// C++ standard version used to compile PAL.
# ifdef _MSVC_LANG
// MSVC requires the /Zc:__cplusplus flag, otherwise __cplusplus is that of 199711L (C++98). _MSVC_LANG has the correct
// C++ standard version.
#  define PAL_CPLUSPLUS _MSVC_LANG
# else
#  define PAL_CPLUSPLUS __cplusplus
# endif

/// Checks if PAL is compiled with C++ of at least version @p v.
#define PAL_CPLUSPLUS_AT_LEAST(v) (PAL_CPLUSPLUS >= (v))

static_assert(
    PAL_CPLUSPLUS_AT_LEAST(PAL_CPLUSPLUS_1709),
    "C++ standard version " PAL_STRINGIFY(PAL_CPLUSPLUS_1709) " is required to build PAL. "
    "Found " PAL_STRINGIFY(PAL_CPLUSPLUS) ".");

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 878
/// We already declare NOMINMAX publicly, but that won't stop clients from defining their own min/max macros.
/// These macros confuse the compiler when using functions named min/max, leading to build errors.
#if defined(min) || defined(max)
static_assert(false, "Clients may not define macros named \"min\" or \"max\".");
#endif
#endif

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 873
#include <chrono>
#endif
#include <cstddef>

/// stdint is included instead of cstdint to allow Visual Studio Intellisense to work for Linux builds. This can be
/// removed if the error caused by including cstdint is figured out.
#include "stdint.h"

/// Include in the class declaration in order to disallow use of the copy constructor and assignment operator for that
/// class.
#define PAL_DISALLOW_COPY_AND_ASSIGN(_typename) \
    _typename(const _typename&) = delete;       \
    _typename& operator=(const _typename&) = delete;

/// Include in the declaration in order to disallow use of the default constructor for a class.
#define PAL_DISALLOW_DEFAULT_CTOR(_typename)    \
    _typename() = delete;

#if !defined(__GNUC__)

// Equates to the [__stdcall](https://github.com/MicrosoftDocs/cpp-docs/blob/master/docs/cpp/stdcall.md) convention on Windows.
#define PAL_STDCALL __stdcall
// Equates to the [__cdecl](https://github.com/MicrosoftDocs/cpp-docs/blob/master/docs/cpp/cdecl.md) convention on Windows.
#define PAL_CDECL __cdecl
// Equates to [__declspec(align(__x))](https://github.com/MicrosoftDocs/cpp-docs/blob/master/docs/cpp/align-cpp.md) on Windows.
#define PAL_ALIGN(__x) __declspec(align(__x))
#define PAL_FORCE_INLINE __forceinline
#else
/// Undefined on GCC platforms.
#define PAL_STDCALL
/// Undefined on GCC platforms.
#define PAL_CDECL
/// Undefined on GCC platforms.
#define PAL_ALIGN(__x)
#define PAL_FORCE_INLINE __attribute__((always_inline)) inline
#endif

/// Platform cache line size in bytes.
#define PAL_CACHE_LINE_BYTES 64
/// Platform system memory page size in bytes.
#define PAL_PAGE_BYTES       4096

/// Force cache line alignment.
#define PAL_ALIGN_CACHE_LINE PAL_ALIGN(PAL_CACHE_LINE_BYTES)

#if !defined(_WIN32)
/// Value representing an invalid file descriptor on Linux systems.
constexpr int32_t InvalidFd = -1;
#endif

#ifdef __has_builtin
/// A macro that checks for the presence of builtin functions. Will default to false if the compiler does not have
/// support for doing this check.
#define PAL_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#define PAL_HAS_BUILTIN(builtin) 0
#endif

#if defined(__has_cpp_attribute)
#define PAL_HAS_CPP_ATTR(attr) __has_cpp_attribute(attr)
#else
#define PAL_HAS_CPP_ATTR(attr) 0
#endif

/// Library-wide namespace encapsulating all PAL utility collection entities.
namespace Util
{

typedef int8_t   int8;    ///< 8-bit integer.
typedef int16_t  int16;   ///< 16-bit integer.
typedef int32_t  int32;   ///< 32-bit integer.
typedef int64_t  int64;   ///< 64-bit integer.
typedef uint8_t  uint8;   ///< Unsigned 8-bit integer.
typedef uint16_t uint16;  ///< Unsigned 16-bit integer.
typedef uint32_t uint32;  ///< Unsigned 32-bit integer.
typedef uint64_t uint64;  ///< Unsigned 64-bit integer.
typedef uint64_t gpusize; ///< Used to specify GPU addresses and sizes of GPU allocations.  This differs from
                          ///  size_t since the GPU still uses 64-bit addresses on a 32-bit OS.

/// Error and return codes indicating outcome of a requested operation.  Success result codes are greater than or equal
/// to 0, and error results codes are less than 0.
enum class Result : int32
{
    /// @internal The operation completed successfully.
    _Success                        = 0x00000000,

    // Unfortunately for Linux clients, X.h includes a "#define Success 0" macro.  Clients have their choice of either
    // undefing Success before including this header or using _Success when dealing with PAL.
#ifndef Success
    /// The operation completed successfully.
    Success                         = _Success,
#endif

    /// The operation is not supported.
    Unsupported                     = 0x00000001,

    /// The operation completed successfully but the result is not ready.  This result code normally applies to
    /// situations where results of queued GPU operations such as queries and fences have not been written to memory
    /// yet.
    NotReady                        = 0x00000002,

    /// The wait operation completed due to a client-specified timeout condition.
    Timeout                         = 0x00000003,

    /// The event is in the "set" state.  @see IGpuEvent::GetStatus.
    EventSet                        = 0x00000004,

    /// The event is in the "reset" state.  @see IGpuEvent::GetStatus.
    EventReset                      = 0x00000005,

    /// The operation was successful, but the client has reached the maximum allowable number of flippable GPU memory
    /// objects.  Future requests to create presentable Images or flippable GPU memory objects may fail due to
    /// limitations within the underlying OS.
    /// @see IDevice::CreateGpuMemory.
    /// @see IDevice::CreatePresentableImage.
    TooManyFlippableAllocations     = 0x00000006,

    /// The present was successful, but some portion of the window is currently occluded by another window.
    PresentOccluded                 = 0x00000007,

    /// The directory/file/etc. being created already exists.
    AlreadyExists                   = 0x00000008,

    /// A warning indicates an operation is successful (supported by H/W) but out of a certain spec (e.g. VESA).
    OutOfSpec                       = 0x00000009,

    /// The value being searched for was not found.
    NotFound                        = 0x0000000A,

    /// End of file reached successfully.
    Eof                             = 0x0000000B,

    /// If ReserveEntryOnMiss was specified, the entry was not found, and the entry was successfully reserved.
    Reserved                        = 0x0000000C,

    /// If an operation is purposefully terminated early, rather than from an error.
    Aborted                         = 0x0000000D,

    /// The operation encountered an unknown error.
    ErrorUnknown                    = -(0x00000001),

    /// The requested operation is unavailable at this time.
    ErrorUnavailable                = -(0x00000002),

    /// The initialization operation failed for unknown reasons.
    ErrorInitializationFailed       = -(0x00000003),

    /// The operation could not complete due to insufficient system memory.
    ErrorOutOfMemory                = -(0x00000004),

    /// The operation could not complete due to insufficient GPU memory.
    ErrorOutOfGpuMemory             = -(0x00000005),

    /// The device was lost due to its removal or a possible hang and recovery condition.  The client should destroy all
    /// devices (and objects attached to them) and re-enumerate the available devices be calling EnumerateDevices().
    ErrorDeviceLost                 = -(0x00000007),

    /// A required input pointer passed to the call was invalid (probably null).
    ErrorInvalidPointer             = -(0x00000008),

    /// An invalid value was passed to the call.
    ErrorInvalidValue               = -(0x00000009),

    /// An invalid ordinal was passed to the call.
    ErrorInvalidOrdinal             = -(0x0000000A),

    /// An invalid memory size was passed to the call.
    ErrorInvalidMemorySize          = -(0x0000000B),

    /// Invalid flags were passed to the call.
    ErrorInvalidFlags               = -(0x0000000C),

    /// An invalid alignment parameter was specified
    ErrorInvalidAlignment           = -(0x0000000D),

    /// An invalid resource format was specified.
    ErrorInvalidFormat              = -(0x0000000E),

    /// The requested operation cannot be performed on the provided @ref Pal::IImage object.
    ErrorInvalidImage               = -(0x0000000F),

    /// The descriptor set data is invalid or does not match the related pipeline.
    ErrorInvalidDescriptorSetData   = -(0x00000010),

    /// An invalid queue type was specified.
    ErrorInvalidQueueType           = -(0x00000011),

    /// An invalid object type was specified.
    ErrorInvalidObjectType          = -(0x00000012),

    /// The specified shader uses an unsupported version of AMD IL.
    ErrorUnsupportedShaderIlVersion = -(0x00000013),

    /// The specified shader code is invalid or corrupt.
    ErrorBadShaderCode              = -(0x00000014),

    /// The specified serialized pipeline data is invalid or corrupt.
    ErrorBadPipelineData            = -(0x00000015),

    /// The queue operation specified more GPU memory references than are supported.
    /// @see Pal::IQueue::Submit
    /// @see Pal::IDevice::AddGpuMemoryReferences
    /// @see Pal::DeviceProperties::maxGpuMemoryRefsResident
    ErrorTooManyMemoryReferences    = -(0x00000016),

    /// The memory object cannot be mapped because it does not reside in a CPU visible heap.
    ErrorNotMappable                = -(0x00000017),

    /// The map operation failed due to an unknown or system reason.
    ErrorGpuMemoryMapFailed         = -(0x00000018),

    /// The unmap operation failed due to an unknown or system reason.
    ErrorGpuMemoryUnmapFailed       = -(0x00000019),

    /// The serialized pipeline load operation failed due to an incompatible device.
    ErrorIncompatibleDevice         = -(0x0000001A),

    /// The serialized pipeline load operation failed due to an incompatible PAL library.
    ErrorIncompatibleLibrary        = -(0x0000001B),

    /// The requested operation (such as command buffer submission) can't be completed because command buffer
    /// construction is not complete.
    ErrorIncompleteCommandBuffer    = -(0x0000001C),

    /// The specified command buffer failed to build correctly.  This error can be delayed from the original source of
    /// the error since the command buffer building methods do not return error codes.
    ErrorBuildingCommandBuffer      = -(0x0000001D),

    /// The operation cannot complete since not all objects have valid GPU memory bound to them.
    ErrorGpuMemoryNotBound          = -(0x0000001E),

    /// The requested operation is not supported on the specified queue type.
    ErrorIncompatibleQueue          = -(0x0000001F),

    /// The object cannot be created or opened for sharing between multiple GPU devices.
    ErrorNotShareable               = -(0x00000020),

    /// The operation failed because the specified fullscreen mode was unavailable.  This could be a failure while
    /// attempting to take fullscreen ownership, or when attempting to perform a fullscreen present and the user has
    /// left fullscreen mode.
    ErrorFullscreenUnavailable      = -(0x00000021),

    /// The targeted screen of the operation has been removed from the system.
    ErrorScreenRemoved              = -(0x00000022),

    /// Present failed because the screen mode is no longer compatible with the source image.
    ErrorIncompatibleScreenMode     = -(0x00000023),

    /// The cross-GPU present failed, possibly due to a lack of system bus bandwidth to accommodate the transfer.
    ErrorMultiDevicePresentFailed   = -(0x00000024),

    /// The slave GPU(s) in an MGPU system cannot create BLTable present images.
    ErrorWindowedPresentUnavailable = -(0x00000025),

    /// The attempt to enter fullscreen exclusive mode failed because the specified image doesn't properly match the
    /// screen's current dimensions.
    ErrorInvalidResolution          = -(0x00000026),

    /// The shader specifies a thread group size that is bigger than what is supported by this device.
    ErrorThreadGroupTooBig          = -(0x00000027),

    /// Invalid image create info: Specified both color target and depth usage
    ErrorInvalidImageTargetUsage    = -(0x00000028),

    /// Invalid image create info: Specified a 1D type for a color target
    ErrorInvalidColorTargetType     = -(0x00000029),

    /// Invalid image create info: Specified a non-2D type for a depth/stencil target
    ErrorInvalidDepthTargetType     = -(0x0000002A),

    /// Invalid image create info: The image format supports depth/stencil but depth/stencil usage was not specified
    ErrorMissingDepthStencilUsage   = -(0x0000002B),

    /// Invalid image create info: Specified MSAA and multiple mip levels
    ErrorInvalidMsaaMipLevels       = -(0x0000002C),

    /// Invalid image create info: The image format is incompatible with MSAA
    ErrorInvalidMsaaFormat          = -(0x0000002D),

    /// Invalid image create info: The image type is incompatible with MSAA
    ErrorInvalidMsaaType            = -(0x0000002E),

    /// The sample count is invalid
    ErrorInvalidSampleCount         = -(0x0000002F),

    /// Invalid image create info: Invalid block compressed image type
    ErrorInvalidCompressedImageType = -(0x00000030),

    /// Invalid image create info: Format is incompatible with the specified image usage
    ErrorInvalidUsageForFormat      = -(0x00000032),

    /// Invalid image create info: Array size is invalid
    ErrorInvalidImageArraySize      = -(0x00000033),

    /// Invalid image create info: Array size is invalid for a 3D image
    ErrorInvalid3dImageArraySize    = -(0x00000034),

    /// Invalid image create info: Image width is invalid
    ErrorInvalidImageWidth          = -(0x00000035),

    /// Invalid image create info: Image height is invalid
    ErrorInvalidImageHeight         = -(0x00000036),

    /// Invalid image create info: Image depth is invalid
    ErrorInvalidImageDepth          = -(0x00000037),

    /// Invalid image create info: Mip count is invalid
    ErrorInvalidMipCount            = -(0x00000038),

    /// Invalid image create info: Image format is incompatible with the image usage specified.
    ErrorFormatIncompatibleWithImageUsage   = -(0x00000039),

    /// Operation requested an image plane that is not available on the image.
    ErrorImagePlaneUnavailable              = -(0x0000003A),

    /// Another format is incompatible with an image's format.
    ErrorFormatIncompatibleWithImageFormat  = -(0x0000003B),

    /// Another format is incompatible with an image plane's format.
    ErrorFormatIncompatibleWithImagePlane   = -(0x0000003C),

    /// Operation requires a shader readable or writable image usage but the image does not support it.
    ErrorImageNotShaderAccessible           = -(0x0000003D),

    /// Format is paired with a channel mapping that contains invalid components.
    ErrorInvalidFormatSwizzle               = -(0x0000003E),

    /// A base mip level that is out of bounds or otherwise invalid was specified.
    ErrorInvalidBaseMipLevel                = -(0x0000003F),

    /// A view array size that was zero or otherwise invalid was specified.
    ErrorInvalidViewArraySize               = -(0x00000040),

    /// A view base array slice that was out of bounds or otherwise invalid was specified.
    ErrorInvalidViewBaseSlice               = -(0x00000041),

    /// A view image type was specified that is incompatible with the image's type.
    ErrorViewTypeIncompatibleWithImageType  = -(0x00000042),

    /// A view specifies an array slice range that is larger than what is supported by the image.
    ErrorInsufficientImageArraySize         = -(0x00000043),

    /// It is illegal to create a cubemap view into an MSAA image.
    ErrorCubemapIncompatibleWithMsaa        = -(0x00000044),

    /// A cubemap view was created to an image that does not have square width and height.
    ErrorCubemapNonSquareFaceSize           = -(0x00000045),

    /// An fmask view was created to an image that does not support an fmask.
    ErrorImageFmaskUnavailable              = -(0x00000046),

    /// A private screen was removed.
    ErrorPrivateScreenRemoved               = -(0x00000047),

    /// A private screen was already in exclusive use.
    ErrorPrivateScreenUsed                  = -(0x00000048),

    /// The image count created or opened on this private display exceed maximum.
    ErrorTooManyPrivateDisplayImages        = -(0x00000049),

    /// The private screen is not enabled.
    ErrorPrivateScreenNotEnabled            = -(0x0000004A),

    /// The private screen count exceeds the maximum (including emulated and physical ones).
    ErrorTooManyPrivateScreens              = -(0x0000004B),

    /// Invalid image create info: Image rowPitch does not equal the image's actual row pitch.
    ErrorMismatchedImageRowPitch            = -(0x0000004C),

    /// Invalid image create info: Image depthPitch does not equal the image's actual depth pitch.
    ErrorMismatchedImageDepthPitch          = -(0x0000004D),

    /// The given swap chain cannot be associated with any more presentable images.
    ErrorTooManyPresentableImages           = -(0x0000004E),

    /// A fence was used in GetStatus() or WaitForFences() without being used in any submission.
    ErrorFenceNeverSubmitted                = -(0x0000004F),

    /// The image used on the specified private screen has an invalid format.
    ErrorPrivateScreenInvalidFormat         = -(0x00000050),

    /// The timing data set on the specified private screen was invalid.
    ErrorPrivateScreenInvalidTiming         = -(0x00000051),

    /// The resolution set on the specified private screen was invalid.
    ErrorPrivateScreenInvalidResolution     = -(0x00000052),

    /// The scaling parameter set on the specified private screen was invalid.
    ErrorPrivateScreenInvalidScaling        = -(0x00000053),

    /// Invalid image create info: Invalid YUV image type
    ErrorInvalidYuvImageType                = -(0x00000054),

    /// The external shader cache found a matching hash but the with different key data.
    ErrorShaderCacheHashCollision           = -(0x00000055),

    /// The external shader cache is full
    ErrorShaderCacheFull                    = -(0x00000056),

    /// The operation caused a pagefault.
    ErrorGpuPageFaultDetected               = -(0x00000057),

    /// The provided pipeline ELF uses an unsupported ABI version.
    ErrorUnsupportedPipelineElfAbiVersion   = -(0x00000058),

    /// The provided pipeline ELF is invalid.
    ErrorInvalidPipelineElf                 = -(0x00000059),

    /// The returned results were incomplete.
    ErrorIncompleteResults                  = -(0x00000060),

    /// The display mode is imcompatible with framebuffer or CRTC.
    ErrorIncompatibleDisplayMode            = -(0x00000061),

    /// Implicit fullscreen exclusive mode is not safe because the specified window size doesn't match the
    /// screen's current dimensions.
    ErrorIncompatibleWindowSize             = -(0x00000062),

    /// A semaphore was used in WaitForSemaphores() without being signaled.
    ErrorSemaphoreNeverSignaled             = -(0x00000063),

    /// Invalid image create info: specified metadataMode is invalid for the Image.
    ErrorInvalidImageMetadataMode           = -(0x00000064),

    /// Invalid external handle detected for the Image.
    ErrorInvalidExternalHandle              = -(0x00000065),

    /// The permission of operation is denied.
    ErrorPermissionDenied                   = -(0x00000066),

    /// The operation failed because the disk is full.
    ErrorDiskFull                           = -(0x00000067),

    /// The static VMID acquire/release operation failed.
    ErrorStaticVmidOpFailed                 = -(0x00000068),

};

///Specifies a ratio of two unsigned integers.
struct Rational
{
    uint32 numerator;   ///< Numerator
    uint32 denominator; ///< Denominator
};

/// Implements operator== for PAL's Rational struct where similar ratios like 2/3 and 4/6 are treated as equal.
///
/// @param left Rational to be compared
/// @param right Rational to be compared
///
/// @return true if the ratios are logically equal.
constexpr bool operator==(
    const Rational& lhs,
    const Rational& rhs)
{
    // Any ratio with a zero denominator is illegal/undefined, for example: "3/0 == 5/0" or "5/3 == 0/0". We must pick
    // either "true" or "false" for these illegal cases. "true" seems like the most wrong option so we use "false".
    if ((lhs.denominator == 0) || (rhs.denominator == 0))
    {
        return false;
    }
    // Otherwise, our equality check is: lhs_n / lhs_d == rhs_n / rhs_d
    // Multiply both sides by lhs_d: lhs_n == rhs_n * lhs_d / rhs_d
    // Multiply both sides by rhs_d: lhs_n * rhs_d == rhs_n * lhs_d
    // This trick avoids dealing with common factors or remainders and uses no slow division instructions.
    return (static_cast<uint64>(lhs.numerator) * static_cast<uint64>(rhs.denominator) ==
            static_cast<uint64>(rhs.numerator) * static_cast<uint64>(lhs.denominator));
}

// Flags to be passed to store operations.
struct StoreFlags
{
    union
    {
        struct
        {
            uint32 enableFileCache      : 1;    ///< If we should skip the file cache layer when we get to it.
            uint32 enableCompression    : 1;    ///< If we should skip the compression layer when we get to it.
            uint32 reserved             : 30;
        };
        uint32 all;
    };
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 873
/// Seconds stored as a float instead of an integer.
using fseconds      = std::chrono::duration<float>;
/// Milliseconds stored as a float instead of an integer.
using fmilliseconds = std::chrono::duration<float, std::milli>;
/// Microseconds stored as a float instead of an integer.
using fmicroseconds = std::chrono::duration<float, std::micro>;
/// Nanoseconds stored as a float instead of an integer.
using fnanoseconds  = std::chrono::duration<float, std::nano>;

/// A time_point who's epoch is January 1st 1970 and uses seconds for the duration.
/// C++20 guarantees us that system_clock's epoch is always January 1st 1970 on all platforms.
/// system_clock's internal duration is still implementation defined.
/// On Windows it's hundreds of nanoseconds and on Linux it's seconds.
/// However time_point has it's own duration type.
/// As long as we go through the time_point to interpret the duration then everything should be in terms of seconds.
using SecondsSinceEpoch = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;

/// Like std::chrono::duration_cast, but it preserves the special 'infinite' value used in timeouts.
template <class ToDuration, class Rep, class Period>
constexpr ToDuration TimeoutCast(
    const std::chrono::duration<Rep, Period>& d)
{
    if (d == (std::chrono::duration<Rep,Period>::max)())
    {
        return (ToDuration::max)();
    }
    else
    {
        return std::chrono::duration_cast<ToDuration, Rep, Period>(d);
    }
}
#endif

/// Inline function to determine if a Result enum is considered an error.
constexpr bool IsErrorResult(Result result) { return (static_cast<int32>(result) < 0); }

/// Inline function to collapse two Result enums into the most useful Result code.  It considers errors to be more
/// interesting than success codes and considers "Success" to be the least interesting success code. If both Results
/// are errors, the first Result is returned.
constexpr Result CollapseResults(Result lhs, Result rhs)
    { return (IsErrorResult(lhs) || (static_cast<uint32>(lhs) > static_cast<uint32>(rhs))) ? lhs : rhs; }

/**
 ***********************************************************************************************************************
 * @page UtilOverview Utility Collection
 *
 * In addition to its GPU-specific core functionality, PAL provides a lot of generic, OS-abstracted software utilities
 * in the @ref Util namespace.  The PAL core relies on these utilities, but they are also available for use by its
 * clients.  In fact, it is possible to build and use PAL only for its utility collection by building PAL with the
 * PAL_BUILD_CORE build option set to 0.
 *
 * All available PAL utilities are defined in the @ref Util namespace, and are briefly summarized below.  See the
 * Reference topics for more detailed information on specific classes, enums, etc.
 *
 * ### System Memory Management
 * palSysMemory.h defines a handful of macros that can be used for allocating and freeing system heap memory.  These
 * macros will use the client-specified allocation callbacks specified by the client at CreatePlatform() if specified.
 * These macros are:
 *
 * - PAL_MALLOC: Equivalent to malloc().
 * - PAL_CALLOC: Equivalent to calloc().
 * - PAL_FREE: Equivalent to free().
 * - PAL_SAFE_FREE: Equivalent to free(), then nulls out the specified pointer.
 * - PAL_NEW: Equivalent to C++ new.
 * - PAL_NEW_ARRAY: Equivalent to C++ new[].
 * - PAL_PLACEMENT_NEW: Equivalent to C++ placement new.
 * - PAL_DELETE: Equivalent to C++ delete.
 * - PAL_DELETE_THIS: Special version of PAL_DELETE that effectively does "delete this;"  This is necessary for
 *   classes that have non-public destructors.
 * - PAL_DELETE_ARRAY: Equivalent to C++ delete[].
 * - PAL_SAFE_DELETE_ARRAY: Equivalent to C++ delete, then nulls out the specified pointer.
 * - PAL_SAFE_DELETE: Equivalent to C++ delete[], then nulls out the specified pointer.
 *
 * ### Allocators
 * All of the memory management macros take in a templated allocator, which is required to have the following two
 * functions defined:
 *
 *     void* Alloc(const Util::AllocInfo)
 *     void  Free(const Util::FreeInfo)
 *
 * It is expected that clients that specify their own allocators will handle cases that require specific alignments
 * and/or zeroing the returned memory.
 *
 * Some allocators can be created for use by clients:
 * - VirtualLinearAllocator: A linear allocator that allocates virtual memory and backs it with physical memory
 *   when needed.
 *
 * ### Debug Prints and Asserts
 * palDbgPrint.h and palAssert.h provide a number of macros used widely by the PAL core and also available for use
 * by clients.
 *
 * The PAL_DPF, PAL_DPINFO, PAL_DPERROR, and PAL_DPWARN can be used to issue debug prints.  These macros will be nulled
 * out if PAL_ENABLE_PRINTS_ASSERTS is not defined to be 1.  SetDbgPrintMode() can be called to configure how the
 * different categories of debug prints will be handled (e.g., print to the debugger, print to file, etc.).
 *
 * The PAL_ASSERT and PAL_ALERT macros can be used to verify expected states of the program at runtime.  PAL_ASSERT
 * should be used for verifying expected invariants and assumptions, while PAL_ALERT should be used to alert the
 * developer of a condition that is allowed, but not typically expected (i.e., failure of a system memory allocation).
 * Note that the polarity of the condition check is different between assert and alert.  Asserts "assert" that the
 * specified condition is true (and complain if it's not), while alerts "alert" a developer if an unexpected condition
 * is true.  These macros will be nulled out if PAL_ENABLE_PRINTS_ASSERTS is not defined to be 1.  EnableAssertMode()
 * can be called to enable/disable asserts or alerts at runtime.
 *
 * ### Generic Containers
 * Util includes a number of generic container data structure implementations.  Note that most of these are broken up
 * into two header files - for example, list.h and listImpl.h.  The intention is that list.h will be included from
 * other header files that need a full list definition, while listImpl.h will be included by .cpp files that actually
 * interact with the list.  This should keep build times down versus putting all implementations directly in list.h.
 * - AutoBuffer: Allows dynamic arrays to be placed on the stack without a heap allocation in situations where a
 *   maximum reasonable expected size is known.
 * - Deque: Double ended queue.
 * - HashMap: Fast map implementation.  Note that this implementation has some non-standard restrictions on the key
 *   (can't be 0) and value size (must fit in a cache line).
 * - HashSet: Fast set implementation.  Note the similar restrictions to HashMap.
 * - IntervalTree: [Interval tree] implementation.
 * - RingBuffer: A ringed buffer of variable length and size.
 *
 * ### Multithreading and Synchronization
 * Util includes a number of OS-abstracted multithreading and CPU synchronization constructs:
 *
 * - Thread
 * - Mutex
 * - Semaphore
 * - ConditionVariable
 * - Event
 *
 * ### Files
 * The File class provides an OS-abstracted interface for opening files and reading/writing data in those files.
 * Further, the ElfReadContext and ElfWriteContext classes provide functionality for reading and writing buffers in the
 * [Executable and Linkable Format (ELF)]
 * The ELF utilities can be used in conjunction with File in order to read/write ELF files on disk.
 *
 * ### Inline Functions
 * palInlineFuncs.h defines a bunch of simple inline functions that are used throughout PAL and might be useful to
 * clients.  Some examples include VoidPtrInc(), Pow2Pad(), Min(), Max(), Strncpy(), etc.
 *
 * palMath.h defines a Math namespace with various constants and functions related to floating point conversions and
 * basic math rouintes like Sqrt().
 *
 * Additionally, palInlineFuncs.h defines a template metaprogramming string hash implementation that can produce
 * a FNV1A hash for a string specified in the source code without the string showing up in a compiled release build.
 *
 * ### System Utilities
 * palSysUtil.h defines a few functions providing abstracted system-specific functionality:
 * - Access to the high resolution CPU performance counters with GetPerfFrequency() and GetPerfCpuTime().
 * - Support for asynchronously querying if a particular keyboard key is currently pressed with IsKeyPressed().
 *
 * ### Cryptographic Algorithm Implementations
 * Util provides the crypto algorithm Md5
 *
 * Next: @ref GpuUtilOverview
 ***********************************************************************************************************************
 */

} // Util
