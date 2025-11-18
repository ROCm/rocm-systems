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
 * @file  palShaderLibrary.h
 * @brief Defines the Platform Abstraction Library (PAL) IShaderLibrary interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palDestroyable.h"
#include "palStringView.h"
#include "palSpan.h"

namespace Pal
{

struct GpuMemSubAllocInfo;

/// Common flags controlling creation of shader libraries.
union LibraryCreateFlags
{
    struct
    {
        uint32 clientInternal  : 1;  ///< Internal library not created by the application.
        uint32 isGraphics      : 1;  ///< Whether it is a graphics library
        uint32 reserved        : 30; ///< Reserved for future use.
    };
    uint32 u32All;                  ///< Flags packed as 32-bit uint.
};

/// Specifies properties about an indirect function belonging to a @ref IShaderLibrary object.  Part of the input
/// structure to IDevice::CreateShaderLibrary().
struct ShaderLibraryFunctionInfo
{
    Util::StringView<char> symbolName;  ///< ELF Symbol name for the associated function.
    gpusize                gpuVirtAddr; ///< [out] GPU virtual address of the function.  This is computed by PAL during
                                        ///  library creation.
};

/// Specifies a shader sub type / ShaderKind.
enum class ShaderSubType : uint32
{
    Unknown = 0,
    Traversal,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    LaunchKernel,           ///< Raytracing launch kernel
    Count
};

/// Specifies properties for creation of a compute @ref IShaderLibrary object.  Input structure to
/// IDevice::CreateShaderLibrary().
struct ShaderLibraryCreateInfo
{
    LibraryCreateFlags  flags;      ///< Library creation flags

    const void*  pCodeObject;       ///< Pointer to code-object ELF binary implementing the Pipeline ABI interface.
                                    ///  The code-object ELF contains pre-compiled shaders, register values, and
                                    ///  additional metadata.
    size_t       codeObjectSize;    ///< Size of code object in bytes.
};

/// Reports properties of a compiled library.
struct LibraryInfo
{
    PipelineHash internalLibraryHash;  ///< 128-bit identifier extracted from this library's ELF binary, composed of
                                       ///  the state the compiler decided was appropriate to identify the compiled
                                       ///  library.  The lower 64 bits are "stable"; the upper 64 bits are "unique".
};

/// Reports shader stats. Multiple bits set in the shader stage mask indicates that multiple shaders have been combined
/// due to HW support. The same information will be repeated for both the constituent shaders in this case.
struct ShaderLibStats
{
    ShaderHash         shaderHash;             ///< Shader hash.
    CommonShaderStats  common;                 ///< The shader compilation parameters for this shader.
    /// Maximum number of VGPRs the compiler was allowed to use for this shader.  This limit will be the minimum
    /// of any architectural restriction and any client-requested limit intended to increase the number of waves in
    /// flight.
    uint32             numAvailableVgprs;
    /// Maximum number of SGPRs the compiler was allowed to use for this shader.  This limit will be the minimum
    /// of any architectural restriction and any client-requested limit intended to increase the number of waves in
    /// flight.
    uint32             numAvailableSgprs;
    size_t             isaSizeInBytes;          ///< Size of the shader ISA disassembly for this shader.
    PipelineHash       palInternalLibraryHash;  ///< Internal hash of the shader compilation data used by PAL.
    uint32             stackFrameSizeInBytes;   ///< Shader function stack frame size
    ShaderSubType      shaderSubType;           ///< ShaderSubType / Shader Kind
    CompilerStackSizes cpsStackSizes;           ///< Stack used in Continuation
};

/**
 ***********************************************************************************************************************
 * @interface IShaderLibrary
 * @brief     Object containing one or more shader functions stored in GPU memory.  These shader functions are callable
 *            from the shaders contained within IPipeline objects.
 *
 * Before a pipeline which calls into this library is bound to a command buffer (using @ref ICmdBuffer::BindPipeline),
 * the client must call @ref IPipeline::LinkWithLibraries() and specify this library in the list of linked libraries.
 * Failure to comply with this requirement is an error and will result in undefined behavior.
 *
 * @see IDevice::CreateShaderLibrary()
 * @see IPipeline::LinkWithLibraries()
 ***********************************************************************************************************************
 */
class IShaderLibrary : public IDestroyable
{
public:
    /// Returns properties of this library and its corresponding shader functions.
    ///
    /// @returns Property structure describing this library.
    virtual const LibraryInfo& GetInfo() const = 0;

    /// Returns a list of GPU memory allocations used by this library.
    ///
    /// @param [in,out] pNumEntries    Input value specifies the available size in pAllocInfoList; output value
    ///                                reports the number of GPU memory allocations.
    /// @param [out]    pAllocInfoList If pAllocInfoList=nullptr, then pNumEntries is ignored on input.  On output it
    ///                                will reflect the number of allocations that make up this pipeline.  If
    ///                                pAllocInfoList!=nullptr, then on input pNumEntries is assumed to be the number
    ///                                of entries in the pAllocInfoList array.  On output, pNumEntries reflects the
    ///                                number of entries in pAllocInfoList that are valid.
    /// @returns Success if the allocation info was successfully written to the buffer.
    ///          + ErrorInvalidValue if the caller provides a buffer size that is different from the size needed.
    ///          + ErrorInvalidPointer if pNumEntries is nullptr.
    virtual Result QueryAllocationInfo(
        size_t*                    pNumEntries,
        GpuMemSubAllocInfo* const  pAllocInfoList) const = 0;

    /// Gives the client access to the resource ID used for internal Pal events.
    /// EX: Resource Create, Resource Bind, Resource Destroy.
    ///
    /// @returns The Resource ID.
    virtual const void* GetResourceId() const = 0;

    /// Obtains the binary code object for this library.
    ///
    /// @param [in, out] pSize  Represents the size of the shader ISA code.
    ///
    /// @param [out] pBuffer    If non-null, the library ELF is written in the buffer. If null, the size required
    ///                         for the library ELF is given out in the location pSize.
    ///
    /// @returns Success if the library binary was fetched successfully.
    ///          +ErrorUnavailable if the library binary was not fetched successfully.
    virtual Result GetCodeObject(
        uint32*  pSize,
        void*    pBuffer) const = 0;

    /// Returns the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @returns Pointer to client data.
    void* GetClientData() const { return m_pClientData; }

    /// Sets the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @param  [in]    pClientData     A pointer to arbitrary client data.
    void SetClientData(
        void* pClientData)
    {
        m_pClientData = pClientData;
    }

    /// Obtains the compiled shader ISA code for the shader function specified.
    ///
    /// @param [in]  pShaderExportName The shader exported name
    ///
    /// @param [in, out] pSize  Represents the size of the shader ISA code.
    ///
    /// @param [out] pBuffer    If non-null, the shader ISA code is written in the buffer. If null, the size required
    ///                         for the shader ISA is given out in the location pSize.
    ///
    /// @returns Success if the shader ISA code was fetched successfully.
    ///          +ErrorUnavailable if the shader ISA code was not fetched successfully.

    virtual Result GetShaderFunctionCode(
        Util::StringView<char> shaderExportName,
        size_t*                pSize,
        void*                  pBuffer) const = 0;

    /// Obtains the shader pre and post compilation stats/params for the specified shader.
    ///
    /// @param [in]  pShaderExportName The shader exported name
    ///
    /// @param [out] pShaderStats Pointer to the ShaderStats structure which will be filled with the shader stats for
    ///                           the shader stage mentioned in shaderType. This cannot be nullptr.
    /// @param [in]  getDisassemblySize If set to true performs disassembly on the shader binary code and reports the
    ///                                 size of the disassembly string in ShaderStats::isaSizeInBytes. Else reports 0.
    /// @returns Success if the stats were successfully obtained for this shader, including the shader disassembly size.
    ///          +ErrorUnavailable if a wrong shader stage for this pipeline was specified, or if some internal error
    ///                           occured.
    virtual Result GetShaderFunctionStats(
        Util::StringView<char> shaderExportName,
        ShaderLibStats*        pShaderStats) const = 0;

    /// Returns the function list owned by this shader library
    ///
    /// @returns A list of ShaderLibraryFunctionInfo.
    virtual const Util::Span<const ShaderLibraryFunctionInfo> GetShaderLibFunctionInfos() const = 0;

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IShaderLibrary() : m_pClientData(nullptr) { }

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IShaderLibrary() { }

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void*  m_pClientData;

    IShaderLibrary(const IShaderLibrary&) = delete;
    IShaderLibrary& operator=(const IShaderLibrary&) = delete;
};

} // Pal
