/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
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
 * @file  palLib.h
 * @brief Defines the Platform Abstraction Library (PAL) initialization and destruction functions.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palSysMemory.h"
#include "palDbgPrint.h"

namespace Pal
{

// Forward declarations
class      IPlatform;

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 987
/// @deprecated Use AsicRevision instead.
/// This is a list of GPUs that the NULL OS layer can compile shaders to in offline mode.
enum class NullGpuId : uint32
{
    Default = 0,   ///< PAL gives the client an arbitrary supported null device.
    Navi10,        ///< 10.1.0
    Navi12,        ///< 10.1.1
    Navi14,        ///< 10.1.2
    Navi21,        ///< 10.3.0
    Navi22,        ///< 10.3.1
    Navi23,        ///< 10.3.2
    Navi24,        ///< 10.3.4
    Rembrandt,     ///< 10.3.5
    Raphael,       ///< 10.3.6
    Navi31,        ///< 11.0.0
    Navi32,        ///< 11.0.1
    Navi33,        ///< 11.0.2
    Phoenix1,      ///< 11.0.3
    Phoenix2,      ///< 11.0.3
    Strix1,        ///< 11.5.0
    StrixHalo,     ///< 11.5.1
    Krackan1,      ///< 11.5.2
    Krackan2,      ///< 11.5.3
    Navi44,        ///< 12.0.0
    Navi48,        ///< 12.0.1
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 958
#endif
    Max,           ///< The maximum count of null devices.
    All,           ///< If you want to enumerate all null devices.
};
#endif

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 989
/// Specifies which graphics IP level (GFXIP) this device has.
enum class GfxIpLevel : uint32
{
    _None = 0,     ///< @internal The device does not have an GFXIP block, or its level cannot be determined

    // Unfortunately for Linux clients, X.h includes a "#define None 0" macro.  Clients have their choice of either
    // undefing None before including this header or using _None when dealing with PAL.
#ifndef None
    None  = _None, ///< The device does not have an GFXIP block, or its level cannot be determined
#endif
    GfxIp10_1,     ///< GFXIP 10.1 (Navi1x)
    GfxIp10_3,     ///< GFXIP 10.3 (Navi2x, Rembrandt, Raphael, Mendocino)
    GfxIp11_0,     ///< GFXIP 11.0 (Navi3x, Phoenix)
    GfxIp11_5,     ///< GFXIP 11.5 (Strix)
    GfxIp12,       ///< GFXIP 12.0 (Navi4x)
};
#endif

/// The version of a particular hardware IP in a specific ASIC.
///
/// @note The fields are purposefully ordered such that the uint32() operator compiles to a single "MOV" instruction
///       on little-endian platforms.
struct IpTriple
{
    uint32 stepping : 16; ///< Stepping value
    uint32 minor    : 8;  ///< Minor revision value
    uint32 major    : 8;  ///< Major revision value

    /// We define a custom constructor solely to avoid using designated initializers, which force us to declare const
    /// IpTriples in field order, which is too confusing. For example: IpTriple{.stepping = 1, .minor = 5, .major = 11}
    ///
    /// Note that C++'s arcane rules cause "IpTriple{11, 5, 1}" to call this constructor with major = 11. minor = 5,
    /// and stepping = 1. So if you see code that looks like classic aggregate initialization, you don't need to worry
    /// that it might actually be setting the values backwards.
    constexpr IpTriple(uint32 major, uint32 minor, uint32 stepping) : stepping{stepping}, minor{minor}, major{major} {}

    /// And this custom zeroing constructor guarantees that "IpTriple{}" still zeros all fields.
    constexpr IpTriple() : IpTriple(0, 0, 0) {}

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 982
    /// These constructors only exist for backcompat, where clients are using partial aggregate initialization.
    /// This is typically done where they're using IpTriple to express a GfxIpLevel which is a misuse of IpTriple.
    /// The new IpLevel struct declared below should be used instead going forwards.
    constexpr IpTriple(uint32 major, uint32 minor) : IpTriple(major, minor, 0) {}
    constexpr IpTriple(uint32 major) : IpTriple(major, 0, 0) {}
#endif

    /// This conversion is designed such that logically higher version numbers will always convert into higher values.
    /// For example, this should evaluate to true: uint32(IpTriple(12, 0, 1)) > uint32(IpTriple(11, 5, 0))
    ///
    /// @returns A single uint32 which uniquely represents this IpTriple.
    constexpr uint32 Bits() const { return stepping | (minor << 16) | (major << 24); }
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 989
    constexpr operator uint32() const { return Bits(); }
#endif
};

///@{
/// Some convenient "IpTriple <> IpTriple" comparison overloads. The inequality operators have been excluded because:
/// 1. Steppings are more like identifiers than ordered verison numbers so it's not intuitive why one ASIC's stepping
///    would sort higher than another's stepping.
/// 2. We want to prevent people from writing code like "if (gfxTriple >= IpTriple(11, 5, 0))", which would result in
///    incorrect behavior if the author actually meant "if (gfxTriple >= IpLevel(11, 5))".
/// So the risk/reward ratio doesn't make IpTriple inequalities seem worth supporting at the moment.
///
/// The definition of the "Bits()" conversion was designed such that these operators should compile to the minimal
/// number of instructions: a CMP + SET pair.
///
/// @param [in] lhs  The left IpTriple in the comparison.
/// @param [in] rhs  The right IpTriple in the comparison.
///
/// @returns True if the comparison is satisfied.
constexpr bool operator==(IpTriple lhs, IpTriple rhs) { return lhs.Bits() == rhs.Bits(); }
constexpr bool operator!=(IpTriple lhs, IpTriple rhs) { return lhs.Bits() != rhs.Bits(); }
///@}

/// The version of a particular hardware IP across a group of ASICs which only differ by stepping values.
///
/// @note The fields are purposefully ordered such that the uint32() operator compiles to a single "MOV" instruction
///       on little-endian platforms. The @ref reserved field is required to make comparisons between IpLevel and
///       @ref IpTriple as trivial as possible.
struct IpLevel
{
    uint32 reserved : 16; ///< Reserved. Must *always* be set to zero!
    uint32 minor    : 8;  ///< Minor revision value
    uint32 major    : 8;  ///< Major revision value

    /// We define a custom constructor solely to avoid using designated initializers, which force us to declare const
    /// IpLevels in field order, which is too confusing. For example: IpLevel{.minor = 5, .major = 11}
    ///
    /// Note that C++'s arcane rules cause "IpLevel{11, 5}" to call this constructor with major = 11. minor = 5. So if
    /// you see code that looks like classic aggregate initialization, you don't need to worry that it might actually
    /// be setting the values backwards.
    constexpr IpLevel(uint32 major, uint32 minor) : reserved{0}, minor{minor}, major{major} {}

    /// And this custom zeroing constructor guarantees that "IpLevel{}" still zeros all fields.
    constexpr IpLevel() : IpLevel(0, 0) {}

    /// This converting constructor is purposefully marked explicit to avoid accidental conversions.
    constexpr explicit IpLevel(IpTriple triple) : IpLevel(triple.major, triple.minor) {}

    /// This conversion is designed such that logically higher version numbers will always convert into higher values.
    /// For example, this should evaluate to true: uint32(IpLevel(12, 0)) > uint32(IpLevel(11, 5)).
    ///
    /// @returns A single uint32 which uniquely represents this IpLevel.
    constexpr uint32 Bits() const { return reserved | (minor << 16) | (major << 24); }
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 989
    constexpr operator uint32() const { return Bits(); }
#endif
};

///@{
/// Some convenient "IpLevel <> IpLevel" comparison overloads.
///
/// The definition of the "Bits()" conversion was designed such that these operators should compile to the minimal
/// number of instructions: a CMP + SET pair.
///
/// @param [in] lhs  The left IpLevel in the comparison.
/// @param [in] rhs  The right IpLevel in the comparison.
///
/// @returns True if the comparison is satisfied.
constexpr bool operator<(IpLevel lhs, IpLevel rhs)  { return lhs.Bits() < rhs.Bits(); }
constexpr bool operator>(IpLevel lhs, IpLevel rhs)  { return lhs.Bits() > rhs.Bits(); }
constexpr bool operator<=(IpLevel lhs, IpLevel rhs) { return lhs.Bits() <= rhs.Bits(); }
constexpr bool operator>=(IpLevel lhs, IpLevel rhs) { return lhs.Bits() >= rhs.Bits(); }
constexpr bool operator==(IpLevel lhs, IpLevel rhs) { return lhs.Bits() == rhs.Bits(); }
constexpr bool operator!=(IpLevel lhs, IpLevel rhs) { return lhs.Bits() != rhs.Bits(); }
///@}

///@{
/// Some convenient "IpTriple <> IpLevel" comparison overloads. These are specifically intended for cases like
/// "if (triple >= IpLevel(11, 5))" where we have a dynamic IpTriple that we want to test against hard-coded levels.
///
/// The definitions of both structs' "Bits()" conversions were designed such that these operators should compile to
/// the minimal number of instructions; at most one ALU and a CMP + SET pair. If the @ref level parameter is a
/// compile-time constant the ALU should be eliminated from all inequalities.
///
/// @param [in] triple  The left IpTriple in the comparison.
/// @param [in] level   The right IpLevel in the comparison.
///
/// @returns True if the comparison is satisfied.
constexpr bool operator<(IpTriple triple, IpLevel level)  { return triple.Bits() <  level.Bits(); }
constexpr bool operator>=(IpTriple triple, IpLevel level) { return triple.Bits() >= level.Bits(); }
constexpr bool operator>(IpTriple triple, IpLevel level)  { return triple.Bits() >  (level.Bits() | 0xFFFFu); }
constexpr bool operator<=(IpTriple triple, IpLevel level) { return triple.Bits() <= (level.Bits() | 0xFFFFu); }
constexpr bool operator==(IpTriple triple, IpLevel level) { return (triple.Bits() & ~0xFFFFu) == level.Bits(); }
constexpr bool operator!=(IpTriple triple, IpLevel level) { return (triple.Bits() & ~0xFFFFu) != level.Bits(); }
constexpr bool operator<(IpLevel level, IpTriple triple)  { return triple > level; }
constexpr bool operator>=(IpLevel level, IpTriple triple) { return triple <= level; }
constexpr bool operator>(IpLevel level, IpTriple triple)  { return triple < level; }
constexpr bool operator<=(IpLevel level, IpTriple triple) { return triple >= level; }
constexpr bool operator==(IpLevel level, IpTriple triple) { return triple == level; }
constexpr bool operator!=(IpLevel level, IpTriple triple) { return triple != level; }
///@}

// The comparisons above only work if both structs are perfectly arranged in a uint32.
static_assert(sizeof(IpTriple) == sizeof(uint32));
static_assert(sizeof(IpLevel)  == sizeof(uint32));

/// Specifies the hardware revision. Values in this enum are intentionally dense so they can be used as direct table
/// indices for null-backend GPU info.
enum class AsicRevision : uint32
{
    Unknown = 0,
    Navi10,               ///< 10.1.0
    Navi12,               ///< 10.1.1
    Navi14,               ///< 10.1.2
    Navi21,               ///< 10.3.0
    Navi22,               ///< 10.3.1
    Navi23,               ///< 10.3.2
    Navi24,               ///< 10.3.4
    Navi31,               ///< 11.0.0
    Navi32,               ///< 11.0.1
    Navi33,               ///< 11.0.2
    Rembrandt,            ///< 10.3.5
    Strix1,               ///< 11.5.0
    Raphael,              ///< 10.3.6
    Phoenix1,             ///< 11.0.3
    Phoenix2,             ///< 11.0.3
    HawkPoint1,           ///< 11.0.3
    HawkPoint2,           ///< 11.0.3
    StrixHalo,            ///< 11.5.1
    Krackan1,             ///< 11.5.2
    Krackan2,             ///< 11.5.3
    Navi44,               ///< 12.0.0
    Navi48,               ///< 12.0.1
    Count,
// These are not included in Count, since Count is the number of unique entries
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 987
/// @deprecated Use GpuInfo directly.
/// Maps a null GPU ID to its associated text name.
struct NullGpuInfo
{
    NullGpuId   nullGpuId;  ///< ID of an ASIC that PAL supports for override purposes
    const char* pGpuName;   ///< Text name of the ASIC specified by nullGpuId
};
#endif

/// Various IDs and info associated with a particular GPU.
struct GpuInfo
{
    AsicRevision asicRev;     ///< PAL specific ASIC revision identifier.
    IpTriple     gfxTriple;   ///< Full GFX IP version (major.minor.stepping) of this GPU.
    uint32       familyId;    ///< Hardware family ID. Driver-defined identifier for a particular family of devices.
    uint32       eRevId;      ///< GPU emulation/internal revision ID.
    uint32       revisionId;  ///< GPU revision. HW-specific value differentiating between different SKUs or revisions.
    uint32       gfxEngineId; ///< Coarse-grain GFX engine ID (R800, SI, etc.).
    uint32       deviceId;    ///< PCI device ID (e.g., Hawaii XT = 0x67B0).
    const char*  pGpuName;    ///< ASIC name and AMDGPU target name (e.g., "NAVI31:gfx1100").
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 987
    NullGpuId    nullId;      ///< @deprecated PAL specific GPU ID supported by the NULL OS layer.
    GfxIpLevel   gfxIpLevel;  ///< @deprecated Use gfxTriple instead.
#endif
};

/// Table of null-device GPU information indexed by AsicRevision.
/// Entries with nullptr pGpuName are unsupported for null device use.
extern const GpuInfo NullGpuInfoTable[static_cast<uint32>(AsicRevision::Count)];

/// Default ASIC revision used when AsicRevision::Unknown is passed to CreateNullDevice.
/// Matches the legacy NullGpuId::Default behavior (Navi31 was the first device in the old lookup table).
constexpr AsicRevision DefaultNullDeviceRevision = AsicRevision::Navi31;

/// The client UMD must identify its API using this enum. Some UMD builds may implement multiple APIs so they must
/// specify which API they're implementing at runtime. Note that the PAL_CLIENT macros are the preferred way to
/// implement client-specific behavior; runtime ClientApi checks should only be used when necessary.
enum class ClientApi : uint32
{
    OpenCl,
    Hip
};

/// Specifies properties for @ref IPlatform creation. Input structure to Pal::CreatePlatform().
struct PlatformCreateInfo
{
    const Util::AllocCallbacks*  pAllocCb;      ///< Optional client-provided callbacks. If non-null, PAL will call the
                                                ///  specified callbacks to allocate and free all internal system
                                                ///  memory. If null, PAL will manage memory on its own through the C
                                                ///  runtime library.
    const Util::LogCallbackInfo* pLogInfo;      ///< Optional client-provided callback info.  If non-null, Pal will
                                                ///  call the callback to pass debug prints to the client.

    const char*                  pSettingsPath; ///< A null-terminated string describing the path to where settings are
                                                ///  located on the system. For example, on Windows, this will refer to
                                                ///  which UMD subkey to look in under a device's key. For Linux, this
                                                ///  is the path to the settings file.

    union
    {
        struct
        {
            uint32 disableGpuTimeout              :  1; ///< Disables GPU timeout detection (Windows only)
            uint32 force32BitVaSpace              :  1; ///< Forces 32bit VA space for the flat address with 32bit ISA
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 987
            uint32 useNullBackend                 :  1; ///< Set to use the null backend. EnumerateDevices() returns
                                                        ///  Result::Unsupported. Use IPlatform::CreateNullDevice()
                                                        ///  to create specific null devices for offline compilation.
#else
            uint32 createNullDevice               :  1; ///< @deprecated Use useNullBackend instead.
#endif
            uint32 enableSvmMode                  :  1; ///< Enable SVM mode. When this bit is set, PAL will reserve
                                                        ///  cpu va range with size "maxSvmSize", and allow client to
                                                        ///  to create gpu or pinned memory for use of Svm.
                                                        ///  For detail of SVM, please refer to CreateSvmGpuMemory
            uint32 requestShadowDescriptorVaRange :  1; ///< Requests that PAL provides support for the client to use
                                                        ///  the @ref VaRange::ShadowDescriptorTable virtual-address
                                                        ///  range. Some GPU's may not be capable of supporting this,
                                                        ///  even when requested by the client.
            uint32 disableInternalResidencyOpts   :  1; ///< Disables residency optimizations for internal GPU memory
                                                        ///  allocations.  Some clients may wish to have them turned
                                                        ///  off to save on system resources.
            uint32 supportRgpTraces               :  1; ///< Indicates that the client supports RGP tracing. PAL will
                                                        ///  use this flag and the hardware support flag to setup the
                                                        ///  DevDriver RgpServer.
            uint32 dontOpenPrimaryNode            :  1; ///< No primary node is needed (Linux only)
            uint32 disableDevDriver               :  1; ///< If no DevDriverMgr should be created with this Platform.
            uint32 reserved                       : 23; ///< Reserved for future use.
        };
        uint32 u32All;                                  ///< Flags packed as 32-bit uint.
    } flags;                                            ///< Platform-wide creation flags.

    ClientApi clientApiId; ///< Client API ID.
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 987
    NullGpuId nullGpuId;   ///< @deprecated ID for the null device. Ignored unless the above flags.createNullDevice
                           ///  bit is set. Use IPlatform::CreateNullDevice(AsicRevision) instead.
#endif
    uint16    apiMajorVer; ///< Major API version number to be used by RGP. Should be set by client based on their
                           ///  contract with RGP.
    uint16    apiMinorVer; ///< Minor API version number to be used by RGP. Should be set by client based on their
                           ///  contract with RGP.
    uint32    instrApiVer; ///  Instrumentation specification version for API-specific SQTT instrumentation fields.
                           ///  Should be set by client based on the SQTT instrumentation spec version being targeted.
    gpusize   maxSvmSize;  ///< Maximum amount of virtual address space that will be reserved for SVM
};

/**
************************************************************************************************************************
* @brief Determines the amount of system memory required for a Platform object.
*
* This function must be called before any other interaction with PAL. An allocation of this amount of memory must be
* provided in the pPlacementAddr parameter of Pal::CreatePlatform.
*
* @ingroup LibInit
*
* @returns Size, in bytes, of system memory required for an IPlatform object.
************************************************************************************************************************
*/
size_t PAL_STDCALL GetPlatformSize();

/**
 ***********************************************************************************************************************
 * @brief Creates the Platform Abstraction Library.
 *
 * On execution of CreatePlatform(), PAL will establish a connection for OS and KMD communication, install the specified
 * system memory allocation callbacks, and initialize any global internal services.  Finally, the client will be
 * returned an object pointer to the instantiated platform object, which is used to query the capabilities of the
 * system.
 *
 * @ingroup LibInit
 *
 * @param [in]  createInfo     Parameters indicating the client requirements for the platform such as allocation
                               callbacks or the settings path.
 * @param [in]  pPlacementAddr Pointer to the location where PAL should construct this object.  There must be as
 *                             much size available here as reported by calling GetPlatformSize().
 * @param [out] ppPlatform     Platform object pointer to the instantiated platform. Must not be null.
 *
 * @returns Success if the initialization completed successfully.  Otherwise, one of the following error codes may be
 *          returned:
 *          + ErrorInvalidPointer will be returned if:
 *              - pPlatform is null.
 *              - pPlacementAddr is null.
 *              - createInfo.pAllocCb is non-null but pfnAlloc and/or pfnFree is null.
 *              - createInfo.pSettingsPath is null.
 *          + ErrorInitializationFailed will be returned if PAL is unable to open a connection to the OS.
 *          + ErrorUnavailable will be returned if none of the GPUs in this system are supported.
 ***********************************************************************************************************************
 */
Result PAL_STDCALL CreatePlatform(
    const PlatformCreateInfo&   createInfo,
    void*                       pPlacementAddr,
    IPlatform**                 ppPlatform);

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 987
/**
 ***********************************************************************************************************************
 * @brief @deprecated Provides an association of NULL devices and their associated text name.
 *
 * @deprecated Use NullGpuInfoTable indexed by AsicRevision instead.
 *
 * @param [in,out] pNullDeviceCount   On input, this is the size of the "pNullDevices" array.  On output, this
 *                                    reflects the number of valid entries in the "pNullDevices" array.
 * @param [out]    pNullDevices       Includes information on the valid NULL devices supported by the system.  If
 *                                    this is NULL, then pNullDeviceCount reflects the maximum possible size of the
 *                                    null-devices array.
 *
 * @returns Success if the initialization completed successfully.  Otherwise, one of the following error codes may be
 *          returned:
 *          + ErrorInvalidPointer will be returned if either input is NULL.
 ***********************************************************************************************************************
 */
Result PAL_STDCALL EnumerateNullDevices(
    uint32*       pNullDeviceCount,
    NullGpuInfo*  pNullDevices);

/**
 ***********************************************************************************************************************
 * @brief @deprecated Provides the NULL device GpuInfo data for the specified NullGpuId.
 *
 * @deprecated Use NullGpuInfoTable[static_cast<uint32>(asicRevision)] instead.
 *
 * @param [in]  nullGpuId Null GPU ID to lookup.
 * @param [out] pGpuInfo  GpuInfo data on successful lookup. Must not be null.
 *
 * @returns Success if the lookup completed successfully. Otherwise, one of the following error codes may be returned:
 *          + ErrorInvalidPointer will be returned if pGpuInfo is NULL.
 *          + NotFound will be returned if the Null GPU ID was not found.
 ***********************************************************************************************************************
 */
Result PAL_STDCALL GetNullGpuInfoForNullGpuId(
    NullGpuId nullGpuId,
    GpuInfo*  pGpuInfo);

/**
 ***********************************************************************************************************************
 * @brief @deprecated Provides the NULL device GpuInfo data for the specified GPU name string.
 *
 * @deprecated Use NullGpuInfoTable and iterate over entries with non-null pGpuName instead.
 *
 * @param [in]  pGpuName Name string of the GPU to lookup (e.g., "NAVI10").
 * @param [out] pGpuInfo GpuInfo data on successful lookup. Must not be null.
 *
 * @returns Success if the lookup completed successfully. Otherwise, one of the following error codes may be returned:
 *          + ErrorInvalidPointer will be returned if pGpuName or pGpuInfo are NULL.
 *          + NotFound will be returned if the Name string was not found.
 ***********************************************************************************************************************
 */
Result PAL_STDCALL GetNullGpuInfoForName(
    const char* pGpuName,
    GpuInfo*    pGpuInfo);

/**
 ***********************************************************************************************************************
 * @brief @deprecated Provides the NULL device GpuInfo data for the specified hardware revision.
 *
 * @deprecated Use NullGpuInfoTable[static_cast<uint32>(asicRevision)] instead.
 *
 * @param [in]  asicRevision Hardware revision to lookup.
 * @param [out] pGpuInfo     GpuInfo data on successful lookup. Must not be null.
 *
 * @returns Success if the lookup completed successfully. Otherwise, one of the following error codes may be returned:
 *          + ErrorInvalidPointer will be returned if pGpuInfo is NULL.
 *          + NotFound will be returned if the hardware revision was not found.
 ***********************************************************************************************************************
 */
Result PAL_STDCALL GetNullGpuInfoForAsicRevision(
    AsicRevision asicRevision,
    GpuInfo*     pGpuInfo);
#endif

/**
 ***********************************************************************************************************************
 * @defgroup LibInit Library Initialization and Destruction
 *
 * Before initializing PAL, it is important to make sure that the interface version is consistent with the client's
 * expectations.  The client should check @ref PAL_INTERFACE_MAJOR_VERSION to ensure the major interface version has not
 * changed since the last PAL integration.  Ideally, this should be performed with a compile-time assert comparing
 * @ref PAL_INTERFACE_MAJOR_VERSION against a client-maintained expected major version.   Minor interface version
 * changes should be backward compatible, and do not require a client change to maintain previous levels of
 * functionality.
 *
 * On startup, the client's first call to PAL must be GetPlatformSize() followed by CreatePlatform().  This function
 * gives an opportunity for PAL to perform any necessary platform-wide initialization such as opening a connection for
 * communication with the operating system and kernel mode driver or initializing tracking facilities for system memory
 * management.  CreatePlatform() returns a created IPlatform object for future interaction with PAL.
 *
 * PAL optionally allows the client to specify a set of memory management callbacks during initialization.  If
 * specified, PAL will not allocate or free any memory directly from the runtime, instead calling back to the client.
 * The client (or application, if the client forwards on the requests) may be able to implement a more efficient
 * allocation scheme.
 *
 * After a successful call to CreatePlatform(), the client should call @ref IPlatform::EnumerateDevices() in order to
 * get a list of supported devices attached to the system.  This function returns an array of @ref IDevice objects
 * which are used by the client to query properties of the devicess and eventually execute work on those devices.
 * IPlatform::EnumerateDevices() is not available to util-only clients (PAL_BUILD_CORE=0).
 *
 * The client may re-enumerate devices at any time by calling IPlatform::EnumerateDevices().  The client must make sure
 * there is no active work on any device and that all objects associated with those devices have been destroyed.
 * IPlatform::EnumerateDevices() will destroy all previously reported @ref IDevice objects and return a fresh set.
 * The client is required to re-enumerate devices when it receives a ErrorDeviceLost error from PAL.
 *
 * After enumerating devices, either during start-up or when recovering from an ErrorDeviceLost error, the client must
 * setup and finalize PAL's per-device settings.  See IDevice::GetPublicSettings(), IDevice::SetDxRuntimeData(),
 * IDevice::CommitSettingsAndInit(), and IDevice::Finalize() for details.
 *
 * After enumerating devices and finalizing them, the client may query the set of available screens. This is done by
 * calling the @ref IPlatform::GetScreens() function.  Note that screens are not available for DX clients.  Each screen
 * is accessible by zero or more of the enumerated devices. Most screens are accessible from a "main" device as well as
 * several other devices which can perform cross-display Flip presents to the screen. In some configurations, screens
 * may not be directly to any of PAL's devices, in which case fullscreen presents are unavailable to that screen. (This
 * typically only occurs in PowerExpress configurations.) Note that when IPlatform::EnumerateDevices() is called, any
 * enumerated @ref IScreen objects which existed prior to that call are invalidated for the specified platform and
 * IPlatform::GetScreens() needs to be called again to get the updated list of screens.
 *
 * On shutdown, the client should call @ref IPlatform::Destroy() to allow PAL to cleanup and free any remaining
 * platform-wide resources.  The client must ensure this call is not made until all other created objects are idle and
 * destroyed (if destroyable).
 *
 * When the client is asked to destroy a device it may call IDevice::Cleanup() to explicitly clean up the device. Some
 * clients will find it necessary to call Cleanup(), for example, if their devices have OS handles that become invalid.
 * Note that Cleanup() doesn't destroy the device; it will return to its initial state, as if it was newly enumerated.
 ***********************************************************************************************************************
 */

} // Pal
