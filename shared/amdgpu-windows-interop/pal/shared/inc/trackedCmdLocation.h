/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  trackedCmdLocation.h
 * @brief Defines the format used for correlation buffers reported through
 *      ICmdBufferReporting::CorrelationReportOnSubmit,
 *      - enum class TrackedCmdLocationMode
 *      - struct TrackedCmdLocation
 *
 *      Plus the helper functions
 *      - TrackedCmdLocationGetDeltaInDwords
 *      - TrackedCmdLocationGetDeltaInBytes
 ***********************************************************************************************************************
 */

#pragma once

namespace Pal
{

namespace CmdDisassembly
{

/// @brief  enum class TrackedCmdLocationMode
///     Defines how to interpret the unions within struct TrackedCmdLocation
///
enum class TrackedCmdLocationMode : uint8_t
{
    Invalid = 0,
    Before,
    After,
    Delta, // before and after
    ClientId,
    ClientEventId
};

/// @brief  struct TrackedCmdLocation defines the format used for correlation data submitted through
///     ICmdBufferReporting::CorrelationReportOnSubmit, and is two DWORDs in size (uint64_t)
///
/// @detail struct TrackedCmdLocation has a number of flavors interpreted by its member m_mode
///
///     For m_mode == TrackedCmdLocationMode::Before, TrackedCmdLocationMode::After or
///         TrackedCmdLocationMode::Delta, m_correlateInternal will be used
///
///     For m_mode == TrackedCmdLocationMode::ClientId, m_clientId will be used
///
///     For m_mode == TrackedCmdLocationMode::ClientEventId, m_clientEvent will be used
///
///
///     For use as m_correlateInternal
///         m_correlateInternal.m_event refers to an internal function that can be converted to a name via
///             Pal::CmdDisassembly::TrackedCmdSupportBase
///         m_correlateInternal.m_ptr is address within the cmdList being correlated by this
///             TrackedCmdLocation
///             For a cmdList with baseAddress and sizeInBytes, m_ptr is in the range
///             [baseAddress, baseAddress+sizeInBytes)
///         m_correlateInternal.m_deltaInDWords is only used when m_mode == TrackedCmdLocationMode::Delta
///             And describes a TrackedCmdLocationMode::Before, TrackedCmdLocationMode::After pair
///             when the m_ptr corresponding to TrackedCmdLocationMode::Before is m_ptr
///             and for TrackedCmdLocationMode::After is m_ptr + m_deltaInDWords * sizeof(DWORD)
///             m_deltaInDWords = 0 if no DWORDS/PM4Packets were written between to the corresponding cmdList
///             between TrackedCmdLocationMode::Before and TrackedCmdLocationMode::After for the
///             event described by m_event
///
///     For use as m_clientEvent
///         m_clientEvent.m_clientEventId is a number provided by the client, provided by a call to
///             IAmdExtCmdDisassembly::IssueClientEvent(clientId, clientEventId)
///             The m_ptr for this event will be the next TrackedCmdLocation, which will have
///                 m_correlateInternal.m_mode == TrackedCmdLocationMode::Delta
///                 m_correlateInternal.m_event == PostClientEvent
///                 m_correlateInternal.m_deltaInDWords == 0
///
///     For use as m_clientId
///         m_clientId.m_clientId is an identifier use by the client for the cmdList that corresponds to
///             this array of correlation data. This will have been set through a call to
///             IAmdExtCmdDisassembly::IssueClientEvent(clientId, clientEventId)
///             When internal correlation is not active, this will be the first tracked location. Otherwise
///             it will not appear until what tracking occurs during Reset is complete.
struct TrackedCmdLocation
{
    static constexpr uint32_t DeltaBitCount = 5;
    static constexpr uint32_t MaxDelta = (1LL << DeltaBitCount) - 1;
    static constexpr uint32_t DwordDeltaShift = 3;
    static constexpr uint8_t PostClientEvent = 0xff;

    static constexpr uint64_t PtrBitCount = 48;
    /// NoCorrespondingBaseAddress is set to an impossible pointer value, that still fits in to the 48 fits
    /// used for m_correlateInternal.m_ptr;
    static constexpr uint64_t NoCorrespondingBaseAddress = (1LL << PtrBitCount) - 1;

    union
    {
        struct
        {
            uint64_t m_mode : 3;
        };

        struct
        {
            uint64_t m_mode : 3;    // TrackedCmdLocationMode::Before/After/Delta
            uint64_t m_event : 8;   // TrackedEvents
            uint64_t m_ptr : PtrBitCount;    // Note, can probably use two bits fewer,
                                    // since these addresses appear to be at a minimum 4-byte aligned.
            uint64_t m_deltaInDWords : DeltaBitCount;
        } m_correlateInternal;

        struct
        {
            uint64_t m_mode : 3;
            uint64_t m_clientId : 61;
        } m_clientId;

        struct
        {
            uint64_t m_mode : 3;
            uint64_t m_clientEventId : 61;

        } m_clientEvent;

        uint64_t m_all;
    };
};

// =====================================================================================================================
/// @brief  Helper funcion to obtain DeltaInDwords from TrackedCmdLocation
///
/// @detail m_correlateInternal.m_deltaInDWords is only used when m_mode == TrackedCmdLocationMode::Delta
///             And describes a TrackedCmdLocationMode::Before, TrackedCmdLocationMode::After pair
///             when the m_ptr corresponding to TrackedCmdLocationMode::Before is m_ptr
///             and for TrackedCmdLocationMode::After is m_ptr + m_deltaInDWords * sizeof(DWORD)
///             m_deltaInDWords = 0 if no DWORDS/PM4Packets were written between to the corresponding cmdList
///             between TrackedCmdLocationMode::Before and TrackedCmdLocationMode::After for the
///             event described by m_event
///
///
/// @returns 0 in m_mode != TrackedCmdLocationMode::Delta
///         m_correlateInternal.m_deltaInDWords otherwise
constexpr uint64_t TrackedCmdLocationGetDeltaInDwords(
    const TrackedCmdLocation location)
{
    const TrackedCmdLocationMode mode = static_cast<TrackedCmdLocationMode>(location.m_mode);
    if (mode == TrackedCmdLocationMode::Delta)
    {
        return location.m_correlateInternal.m_deltaInDWords;
    }
    else
    {
        return 0;
    }
}

// =====================================================================================================================
/// @brief  Helper funcion to convert DeltaInDwords from TrackedCmdLocation to "InBytes"
///
/// @returns 0 in m_mode != TrackedCmdLocationMode::Delta
///         m_correlateInternal.m_deltaInDWords * sizeof(DWORD) otherwise - where DWORD is uint32_t
constexpr uint64_t TrackedCmdLocationGetDeltaInBytes(
    const TrackedCmdLocation location)
{
    return TrackedCmdLocationGetDeltaInDwords(location) << TrackedCmdLocation::DwordDeltaShift;
}

} // namespace CmdDisassembly
} // namespace Pal
