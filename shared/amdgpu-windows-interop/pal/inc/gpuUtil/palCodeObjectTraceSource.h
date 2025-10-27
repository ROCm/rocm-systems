/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "palGpaSession.h"
#include "palGpuUtil.h"
#include "palTraceSession.h"
#include "palVector.h"
#include "palHashSet.h"
#include "palMutex.h"

namespace Pal
{
class IPlatform;
class IDevice;
class IShaderLibrary;
} // namespace Pal

namespace GpuUtil
{
class GpaSession;
} // namespace GpuUtil

namespace GpuUtil
{

namespace TraceChunk
{

/// "CodeObject" RDF chunk identifier & version
constexpr char        CodeObjectChunkId[TextIdentifierSize] = "CodeObject";
constexpr Pal::uint32 CodeObjectChunkVersion                = 2;

/// Header for the "CodeObject" RDF chunk
struct CodeObjectHeader
{
    Pal::uint32     pciId;          /// The ID of the GPU the trace was run on
    Pal::ShaderHash codeObjectHash; /// Hash of the Code Object binary
};

/// "COLoadEvent" RDF chunk identifier & version
constexpr char        CodeObjectLoadEventChunkId[TextIdentifierSize] = "COLoadEvent";
constexpr Pal::uint32 CodeObjectLoadEventChunkVersion                = 3;

struct CodeObjectLoadEventHeader
{
    Pal::uint32 count; /// Number of load events in this chunk
};

/// Describes whether a load event was into GPU memory or from.
enum class CodeObjectLoadEventType : Pal::uint32
{
    LoadToGpuMemory     = 0, /// Code Object was loaded into GPU memory
    UnloadFromGpuMemory = 1  /// Code Object was unloaded from GPU memory
};

/// Describes one or more GPU load/unload(s) of a Code Object. Payload for "COLoadEvent" RDF chunk.
struct CodeObjectLoadEvent
{
    Pal::uint32             pciId;          /// The ID of the GPU the trace was run on
    CodeObjectLoadEventType eventType;      /// Type of loader event
    Pal::uint64             baseAddress;    /// Base address where the Code Object was loaded
    Pal::ShaderHash         codeObjectHash; /// Hash of the (un)loaded Code Object binary
    Pal::uint64             timestamp;      /// CPU timestamp of this event being triggered
};

/// "PsoCorrelation" RDF chunk identifier & version
constexpr char        PsoCorrelationChunkId[TextIdentifierSize] = "PsoCorrelation";
constexpr Pal::uint32 PsoCorrelationChunkVersion                = 3;

struct PsoCorrelationHeader
{
    Pal::uint32 count;  /// Number of PSO correlations in this chunk
};

/// Payload for the "PsoCorrelation" RDF chunks
struct PsoCorrelation
{
    Pal::uint32       pciId;                  /// The ID of the GPU the trace was run on
    Pal::uint64       apiPsoHash;             /// Hash of the API-level Pipeline State Object
    Pal::PipelineHash internalPipelineHash;   /// Hash of all inputs to the pipeline compiler
    char              apiLevelObjectName[64]; /// Debug object name (null-terminated)
};

/// "COCorrelation" RDF chunk identifier & version
constexpr char     CodeObjectCorrelationChunkId[TextIdentifierSize] = "COCorrelation";
constexpr uint32_t CodeObjectCorrelationChunkVersion                = 4;

struct CodeObjectCorrelationHeader
{
    Pal::uint32 count; /// Number of Code Object Correlations in this chunk
};

/// Payload for the "CodeObjectCorrelation" RDF chunks
struct CodeObjectCorrelation
{
    Pal::PipelineHash internalPipelineHash;  /// Hash of all inputs to the pipeline compiler
    Pal::ShaderHash   codeObjectHash;        /// Hash of the Code Object binary in the CO Database
    Pal::uint32       containsMetadata : 1;  /// 1 if the code object contains metadata, 0 otherwise
    Pal::uint32       reserved         : 31; /// Bitflags reserved for future use
};

} // namespace TraceChunk

/// CodeObject Trace Source name & version
constexpr char        CodeObjectTraceSourceName[]  = "codeobject";
constexpr Pal::uint32 CodeObjectTraceSourceVersion = 3;

// =====================================================================================================================
class CodeObjectTraceSource : public ITraceSource
{
public:
    CodeObjectTraceSource(Pal::IPlatform* pPlatform);
    ~CodeObjectTraceSource();

    // ==== TraceSource Native Functions ========================================================================== //
    Pal::Result RegisterPipeline(const Pal::IPipeline* pPipeline, const RegisterPipelineInfo& clientInfo);
    Pal::Result UnregisterPipeline(const Pal::IPipeline* pPipeline);

    Pal::Result RegisterLibrary(const Pal::IShaderLibrary* pLibrary, const RegisterLibraryInfo& clientInfo);
    Pal::Result UnregisterLibrary(const Pal::IShaderLibrary* pLibrary);

    Pal::Result RegisterElfBinary(const ElfBinaryInfo& elfBinaryInfo);
    Pal::Result UnregisterElfBinary(const ElfBinaryInfo& elfBinaryInfo);

    // ==== Base Class Overrides =================================================================================== //
    virtual void OnConfigUpdated(DevDriver::StructuredValue* pJsonConfig) override { }

    virtual Pal::uint64 QueryGpuWorkMask() const override { return 0; }

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 908
    virtual void OnTraceAccepted(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) override { }
#else
    virtual void OnTraceAccepted() override { }
#endif
    virtual void OnTraceBegin(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) override { }
    virtual void OnTraceEnd(Pal::uint32 gpuIndex, Pal::ICmdBuffer* pCmdBuf) override { }
#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 939
    virtual void OnPostambleEnd(
        Pal::uint32      gpuIndex,
        Pal::ICmdBuffer* pCmdBuf) override { }
#endif
    virtual void OnTraceFinished() override;

    virtual const char* GetName()    const override { return CodeObjectTraceSourceName; }
    virtual Pal::uint32 GetVersion() const override { return CodeObjectTraceSourceVersion; }

private:
    Pal::Result RegisterSinglePipeline(const Pal::IPipeline* pPipeline, const RegisterPipelineInfo& clientInfo);
    Pal::Result UnregisterSinglePipeline(const Pal::IPipeline* pPipeline);

    Pal::Result AddCodeObjectLoadEvent(
        const Pal::IShaderLibrary*          pLibrary,
        TraceChunk::CodeObjectLoadEventType eventType);
    Pal::Result AddCodeObjectLoadEvent(
        const Pal::IPipeline*               pLibrary,
        TraceChunk::CodeObjectLoadEventType eventType);
    Pal::Result AddCodeObjectLoadEvent(
        const ElfBinaryInfo&                elfBinaryInfo,
        TraceChunk::CodeObjectLoadEventType eventType);

    Pal::Result WriteCodeObjectChunks();
    Pal::Result WriteLoaderEventsChunk();
    Pal::Result WritePsoCorrelationChunk();
    Pal::Result WriteCoCorrelationChunk();

    struct CodeObjectDatabaseRecord
    {
        Pal::uint32     recordSize;
        Pal::ShaderHash codeObjectHash;
    };

    Pal::IPlatform* const m_pPlatform;

    Util::RWLock                                                        m_registerPipelineLock;
    Util::Vector<CodeObjectDatabaseRecord*,         1, Pal::IPlatform>  m_codeObjectRecords;
    Util::Vector<TraceChunk::CodeObjectLoadEvent,   1, Pal::IPlatform>  m_loadEventRecords;
    Util::Vector<TraceChunk::PsoCorrelation,        1, Pal::IPlatform>  m_psoCorrelationRecords;
    Util::Vector<TraceChunk::CodeObjectCorrelation, 1, Pal::IPlatform>  m_coCorrelationRecords;

    // API hashes -> internal pipeline hash (-> child code object hashes)
    Util::HashSet<Pal::uint64, Pal::IPlatform, Util::JenkinsHashFunc>   m_registeredApiHashes;
    Util::HashSet<Pal::uint64, Pal::IPlatform, Util::JenkinsHashFunc>   m_registeredPipelines;
    Util::HashSet<Pal::uint64, Pal::IPlatform, Util::JenkinsHashFunc>   m_registeredCoHashes;

};

} // namespace GpuUtil

