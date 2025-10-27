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
 * @file  palQueryPool.h
 * @brief Defines the Platform Abstraction Library (PAL) IQueryPool interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palGpuMemoryBindable.h"

namespace Pal
{

/// Specifies a category of GPU query pool.
enum class QueryPoolType : uint32
{
    Occlusion        = 0x0,  ///< Occlusion query pool. Supports queries based on the Z test.
    PipelineStats    = 0x1,  ///< Pipeline stats query pool. Supports queries based on statistics from the GPU's execution
                             ///  such as a count of prims generated, shader invocations, etc.
    StreamoutStats   = 0x2,  ///< Streamout query pool. Supports queries based on statistics from the GPU's execution
                             ///  such as number of primitives written to SO buffer and storage needed.
    Count,
};

/// Specifies what data a query slot must produce. Some query pool types support multiple query types.
enum class QueryType : uint32
{
    Occlusion        = 0x0, ///< The total passes recorded by the Z test.
    BinaryOcclusion  = 0x1, ///< One if there were one or more Z test passes, zero otherwise.
    PipelineStats    = 0x2, ///< The total statistics selected by the given pipeline stats query pool.
    StreamoutStats   = 0x3, ///< SO statistics tracked by CP/VGT including primitives written and storage needed.
    StreamoutStats1  = 0x4, ///< SO1 statistics tracked by CP/VGT including primitives written and storage needed.
    StreamoutStats2  = 0x5, ///< SO2 statistics tracked by CP/VGT including primitives written and storage needed.
    StreamoutStats3  = 0x6, ///< SO3 statistics tracked by CP/VGT including primitives written and storage needed.
    Count,
};

/// Specifies which pipeline stats should be tracked by a pipeline stats query pool.
enum QueryPipelineStatsFlags : uint32
{
    QueryPipelineStatsIaVertices    = 0x1,    ///< Input vertices.
    QueryPipelineStatsIaPrimitives  = 0x2,    ///< Input primitives.
    QueryPipelineStatsVsInvocations = 0x4,    ///< Vertex shader invocations.
    QueryPipelineStatsGsInvocations = 0x8,    ///< Geometry shader invocations.
    QueryPipelineStatsGsPrimitives  = 0x10,   ///< Geometry shader primitives.
    QueryPipelineStatsCInvocations  = 0x20,   ///< Clipper invocations.
    QueryPipelineStatsCPrimitives   = 0x40,   ///< Clipper primitives.
    QueryPipelineStatsPsInvocations = 0x80,   ///< Pixel shader invocations.
    QueryPipelineStatsHsInvocations = 0x100,  ///< Hull shader invocations.
    QueryPipelineStatsDsInvocations = 0x200,  ///< Domain shader invocations.
    QueryPipelineStatsCsInvocations = 0x400,  ///< Compute shader invocations.
    QueryPipelineStatsTsInvocations = 0x800,  ///< Task shader invocations.
    QueryPipelineStatsMsInvocations = 0x1000, ///< Mesh shader invocations.
    QueryPipelineStatsMsPrimitives  = 0x2000, ///< Mesh shader primitives.
    QueryPipelineStatsAll           = 0x3FFF  ///< All of the above stats.
};

/// Specifies properties for @ref IQueryPool creation.  Input structure to IDevice::CreateQueryPool().
struct QueryPoolCreateInfo
{
    QueryPoolType queryPoolType;    ///< Type of query pool to create (i.e., occlusion vs. pipeline stats).
    uint32        numSlots;         ///< Number of slots in the query pool.
    uint32        enabledStats;     ///< An ORed mask of stats flags specific to the query pool type.
                                    ///  @see QueryPipelineStatsFlags for PipelineStats query pools.
    union
    {
        struct
        {
            /// If true, this query pool can have results retrieved using the CPU (using @ref IQueryPool::GetResults)
            /// and can be reset using the CPU (using @ref IQueryPool::Reset).  Otherwise, the client must use command
            /// buffers to perform these operations (using @ref ICmdBuffer::CmdResetQueryPool and
            /// @ref ICmdBuffer::CmdResolveQuery).
            uint32  enableCpuAccess :  1;
            uint32  reserved        : 31;   ///< Reserved for future use.
        };
        uint32  u32All; ///< Flags packed together as a uint32.
    } flags;            ///< Flags controlling QueryPool behavior.
};

/// Controls operations that compute query results.
enum QueryResultFlags : uint32
{
    QueryResultDefault          =  0x0, ///< Default to 32-bit results with no waiting.
    QueryResult64Bit            =  0x1, ///< Store all results as 64-bit values.
    QueryResultWait             =  0x2, ///< Wait for the queries to finish when computing the results.
    QueryResultAvailability     =  0x4, ///< If the results of a query are available at computation time a one will be
                                        ///  written as a separate value after the result value, if the results were not
                                        ///  available a zero will be written.
    QueryResultPartial          =  0x8, ///< If the final result of a query would be unavailable, then return a
                                        ///  result for that query between 0 and what the final result would be.
    QueryResultAccumulate       = 0x10, ///< Results are added to the values present in the destination, if availability
                                        ///  data is enabled it will be ANDed with the present availability data.
    QueryResultPreferShaderPath = 0x20, ///< Prefer a shader resolve path over a command processor path.
    QueryResultOnlyPrimNeeded   = 0x40, ///< Select only primitives storage needed in Streamout query results
    QueryResultAll              = 0x7F  ///< Clients should NOT use it, for internal static_assert purpose only.
};

/**
 ***********************************************************************************************************************
 * @interface IQueryPool
 * @brief     Represents a set of queries that can be used to retrieve detailed info about the GPU's execution of a
 *            particular range of a command buffer.
 *
 * Currently, only occlusion queries and pipeline statistic queries are supported.  All queries in a pool are the same
 * type.
 *
 * @see IDevice::CreateQueryPool()
 ***********************************************************************************************************************
 */
class IQueryPool : public IGpuMemoryBindable
{
public:
    /// Retrieves query results from a query pool.
    ///
    /// Multiple consecutive query results can be retrieved with one call.
    ///
    /// @param [in]     flags      Flags that control the result data layout and how the results are retrieved.
    /// @param [in]     queryType  Specifies what data the query slots must produce.
    /// @param [in]     startQuery First query pool slot to retrieve data for.
    /// @param [in]     queryCount Number of query pool slots to retrieve data for.
    /// @param [in]     pMappedGpuAddr Specify the query buffer mapped address. If the parameter equals nullptr,
    //                                 this method will use Map\UnMap to access the data.
    /// @param [in,out] pDataSize  Input value specifies the available size in pData in bytes; output value reports the
    ///                            number of bytes required to hold all result data.
    /// @param [out]    pData      Location where the query results should be written. Can be null in order to query the
    ///                            required size. The data returned depends on the query pool type and flags. All data
    ///                            entries are either uint32 or uint64 integers. One or more type-specific entries will
    ///                            be optionally followed by one entry for availability. The type-specific data is:<br>
    ///                            + QueryOcclusion: One entry to store the zPass count.
    ///                            + QueryPipelineStats: One entry per statistic enabled in the create info. The stats
    ///                              will be written in the appropriate order for each PAL client.
    /// @param [in]     stride     Stride in bytes between subsequent query result data or zero to request tightly
    ///                            packed result data.
    ///
    /// @returns Success if query results were successfully returned in pData, or NotReady if any of the requested query
    ///          slots does not yet have results available.  Otherwise, one of the following error codes may be
    ///          returned:
    ///          + ErrorInvalidValue if the range defined by startQuery and queryCount is not valid for this query pool.
    ///          + ErrorGpuMemoryNotBound if the query pool requires GPU memory but none is bound.
    ///          + ErrorInvalidMemorySize if pData is non-null and the value stored in pDataSize is too small.
    virtual Result GetResults(
        QueryResultFlags flags,
        QueryType        queryType,
        uint32           startQuery,
        uint32           queryCount,
        const void*      pMappedGpuAddr,
        size_t*          pDataSize,
        void*            pData,
        size_t           stride) = 0;

    /// Use CPU to reset the query pool slots.
    ///
    /// Supported for occlusion and video decode statistics query pools.
    ///
    /// @param [in]     startQuery     First query pool slot to reset.
    /// @param [in]     queryCount     Number of query pool slots to reset.
    /// @param [in]     pMappedCpuAddr Specify the query buffer mapped address. If the parameter equals nullptr,
    //                                 this method will use Map/UnMap to access the data.
    ///
    /// @returns Success if the reset was successfully performed.
    virtual Result Reset(
        uint32  startQuery,
        uint32  queryCount,
        void*   pMappedCpuAddr) = 0;

    /// Returns the distance, in bytes, between successive query slots in the bound GPU memory.
    /// This method is only supported for @ref QueryPoolType::VideoDecodeStats
    ///
    /// @returns the distance, in bytes, between successive query slots in the bound GPU memory.
    virtual gpusize GetQuerySlotStride() const = 0;

    /// Returns the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @returns Pointer to client data.
    void* GetClientData() const
    {
        return m_pClientData;
    }

    /// Sets the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @param  [in]    pClientData     A pointer to arbitrary client data.
    void SetClientData(
        void* pClientData)
    {
        m_pClientData = pClientData;
    }

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IQueryPool() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IQueryPool() { }

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;
};

} // Pal
