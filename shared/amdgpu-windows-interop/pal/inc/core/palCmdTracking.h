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
 * @file  palCmdTracking.h
 * @brief Defines a number of support classes used for construction and storage of struct TrackedCmdLocation
 *      defined in trackedCmdLocation.h
 *
 *      - struct TrackingEventInfo:             A single from uint8 to name, used for logging
 *      - class TrackedCmdSupportBase      A set of TrackingEventInfo, maintained outside of Pal
 *      - class TrackedCmdLocationArray     The arrays for TrackedCmdLocation's used for reporting
 *                                              correlation data through ICmdBufferReporting::CorrelationReportOnSubmit
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palVector.h"

#include "trackedCmdLocation.h"

namespace Pal
{

// forward decl
class Platform;

namespace CmdDisassembly
{

// forward definition
class TrackedCmdLocationArray;

/**
************************************************************************************************************************
* @brief    class TrackedCmdLocationRef
*           A copyable reference to a member in a TrackedCmdLocationArray, invariant to that array be
*           re-allocated.
*
* @detail   Is simply a pointer to a TrackedCmdLocationArray, and an index in to that array
*
************************************************************************************************************************
*/
class TrackedCmdLocationRef
{
public:
    TrackedCmdLocationRef()
        : m_pSourceArray(nullptr),
        m_index(0)
    {
    }

    TrackedCmdLocationRef(
        TrackedCmdLocationArray*    pSourceArray,
        Util::uint32                index)
        : m_pSourceArray(pSourceArray),
        m_index(index)
    {
    }

    TrackedCmdLocationRef(
        TrackedCmdLocationRef&& other) = default;
    TrackedCmdLocationRef(
        TrackedCmdLocationRef const& other) = default;
    TrackedCmdLocationRef& operator=(
        TrackedCmdLocationRef&& other) = default;
    TrackedCmdLocationRef& operator=(
        TrackedCmdLocationRef const& other) = default;

    bool operator==(
        TrackedCmdLocationRef const& other) const
        { return (this->m_pSourceArray == other.m_pSourceArray) && (this->m_index == other.m_index); }
    bool operator!=(
        TrackedCmdLocationRef const& other) const
        { return (this->m_pSourceArray != other.m_pSourceArray) || (this->m_index != other.m_index); }

    TrackedCmdLocation* Use();
    const TrackedCmdLocation* Get() const;

    Util::uint32 GetIndex() const
    {
        return m_index;
    }

    /// Helper functions
    ///

    /// Clears the TrackedCmdLocation referred to by this TrackedCmdLocationRef
    ///
    /// @returns
    ///     Result::ErrorInvalidPointer if (IsValid() == false)
    ///     Result::Success if successful
    Result Clear();

    /// @returns
    ///     TrackedCmdLocationMode::Invalid if (IsValid() == false)
    ///     Get()->m_mode otherwise
    TrackedCmdLocationMode GetMode() const;

    /// Sets the TrackedCmdLocation referred to by this TrackedCmdLocationRef
    ///     to mode TrackedCmdLocationMode::Before
    ///
    /// @param [in] eventId     Refers to an uint8 event that has a begin and/or an end associated with it
    ///                         Most likely, a value registered to a TrackedCmdSupportBase
    /// @param [in] beforePtr   The end pointer for the cmdList being tracked before the event referred to by eventId
    ///                         Only 48-bits of beforePtr are used
    ///
    /// @returns
    ///     Result::ErrorInvalidPointer if (IsValid() == false)
    ///     Result::Success if successful
    Result SetAsBefore(
        uint8   eventId,
        uint64  beforePtr);

    /// Sets the TrackedCmdLocation referred to by this TrackedCmdLocationRef
    ///     to mode TrackedCmdLocationMode::After
    ///
    /// @param [in] eventId     Refers to an uint8 event that has a begin and/or an end associated with it
    ///                         Most likely, a value registered to a TrackedCmdSupportBase
    /// @param [in] afterPtr   The end pointer for the cmdList being tracked after the event referred to by eventId
    ///                        Only 48-bits of afterPtr are used
    ///
    /// @returns
    ///     Result::ErrorInvalidPointer if (IsValid() == false)
    ///     Result::Success if successful
    Result SetAsAfter(
        uint8   eventId,
        uint64  afterPtr);

    /// Sets the TrackedCmdLocation referred to by this TrackedCmdLocationRef
    ///     to mode TrackedCmdLocationMode::Delta, with no begin or end (ie, no data can be written to
    ///     the cmdList being tracked "during" the event referred to be eventId
    ///
    /// @param [in] eventId     Refers to an uint8 event that does not have a begin and/or an end associated with it
    ///                         Such as Pal::CmdDisassembly::TrackedCmdLocation::PostClientEvent
    /// @param [in] ptr         The end pointer for the cmdList being tracked after the event referred to by eventId
    ///                         Only 48-bits of ptr are used
    ///
    /// @returns
    ///     Result::ErrorInvalidPointer if (IsValid() == false)
    ///     Result::Success if successful
    Result SetAsEmptyDelta(
        uint8   eventId,
        uint64  ptr);

    /// Sets the TrackedCmdLocation referred to by this TrackedCmdLocationRef
    ///     to mode TrackedCmdLocationMode::ClientId
    ///
    /// @param [in] clientId    A 61-bit bit value used by the client application to identify which cmdList is being
    ///                         tracked
    ///
    /// @returns
    ///     Result::ErrorInvalidPointer if (IsValid() == false)
    ///     Result::Success if successful
    Result SetAsClientId(
        uint64 clientId);

    /// Sets the TrackedCmdLocation referred to by this TrackedCmdLocationRef
    ///     to mode TrackedCmdLocationMode::ClientEventId
    ///
    /// @param [in] clientEventId   A 61-bit bit value used by the client application to identify
    ///                             a client event  relative to the current end position of the cmdList being tracked
    ///
    /// @returns
    ///     Result::ErrorInvalidPointer if (IsValid() == false)
    ///     Result::Success if successful
    Result SetAsClientEvent(
        uint64 clientEventId);

    /// @brief  bool TrackedCmdLocation::TrySetAsDelta(uint64 afterPtr)
    ///         Will attempt to set this TrackedCmdLocation to type TrackedCmdLocationMode::Delta
    ///
    /// @detail If GetMode() == TrackedCmdLocationMode::Before and afterPtr - m_correlateInternal.m_ptr is small
    ///         enough to be encoded in m_correlateInternal.m_deltaInDWords, the mode will be altered to
    ///         TrackedCmdLocationMode::Delta, with afterPtr - m_correlateInternal.m_ptr encoded in
    ///         m_correlateInternal.m_deltaInDWords.
    ///         If this attempt fails, the calling function should instead create a TrackedCmdLocationMode::After
    ///         TrackedCmdLocation
    ///
    /// @param  [in] afterPtr, the value a TrackedCmdLocationMode::After would have for m_correlateInternal.m_ptr
    /// @return Result::Success if it was possible to set this TrackedCmdLocation to type
    ///             TrackedCmdLocationMode::Delta
    ///         Result::Unsupported if the conditions described above are not met.
    Result TrySetAsDelta(
        uint64 afterPtr);

private:
    TrackedCmdLocationArray*    m_pSourceArray;
    Util::uint32                m_index;

    Result SetMode(
        TrackedCmdLocationMode mode);
};

/// @brief  struct TrackingEventInfo
///     Essentially just a name, plus a boolean to indicate whether the name is valid / has been set
struct TrackingEventInfo
{
    Util::StringView<char>  name;
    bool                    isValid;

    TrackingEventInfo()
        : isValid(false)
    {}
};

/**
************************************************************************************************************************
* @brief    class TrackedCmdSupportBase translates eventId's to strings for internal correlation events
*
* @detail   For use in Pal::Queue when dumping to text files. Corresponds to
*           TrackedCmdLocation::m_correlateInternal.m_event for the cases where TrackedCmdLocation::m_mode
*           is not TrackedCmdLocationMode::ClientEvent
*
*           The implementation for this is in whatever client of Pal that is creating the internal correlation events,
*
************************************************************************************************************************
*/
class TrackedCmdSupportBase
{
public:
    virtual ~TrackedCmdSupportBase() = default;

    void SetEventIdName(
        uint8           eventId,
        const char*     name)
    {
        PAL_ASSERT(static_cast<uint32>(eventId) < NumUInt8Values);
        m_allEventsMap[eventId].name = name;
        m_allEventsMap[eventId].isValid = true;
    }

    TrackingEventInfo const& GetEventInfo(
        uint8 eventId) const
    {
        PAL_ASSERT(static_cast<uint32>(eventId) < NumUInt8Values);
        return m_allEventsMap[eventId];
    }

protected:
    static constexpr uint32 NumUInt8Values = UINT8_MAX + 1;

    TrackingEventInfo m_allEventsMap[NumUInt8Values];

    TrackedCmdSupportBase() = default;
};

/**
************************************************************************************************************************
* @brief    class TrackedCmdLocationArray is simple a TrackedCmdLocationVec together with a clientId
*           and some helpers. TrackedCmdLocationArray live on Pal::GfxCmdBuffer
*
* @detail Each Pal::GfxCmdBuffer has at most CmdDisassembly::MaxNumSubCmdBuffers TrackedCmdLocationArray's
*       corresponding to Pal::GfxCmdBuffer::NumCmdStreams();
*
*       The clientId used for TrackedCmdLocationArray::m_clientId, corresponds to the client Id used in
*       TrackedCmdLocation::m_clientId.m_clientId
*
*       For the moment, the underlying implementation used is
*       Util::Vector<TrackedCmdLocation, DefaultCapacity, Pal::Platform>, but could be changed to use a Chunk
*       scheme, especially as sizes of cmdLists can become very large.
*       The only requirement to a change, is for TrackedCmdLocationRef continues to function as an accessor
*
*       Note that the functions in TrackedCmdLocationArray are not designed for thread-safety, as they are
*       issued from command-list-building functions that are, in their turn, not thread safe. Adding mutex
*       behavior here would potentially hide issues relating to thread-safety.
*
************************************************************************************************************************
*/
class TrackedCmdLocationArray
{
public:
    static constexpr uint32 DefaultCapacity = 1024;
    static constexpr uint32 BadIndex = UINT32_MAX;
    static constexpr uint64 InvalidClientId = UINT64_MAX;

    typedef Util::Vector<TrackedCmdLocation, DefaultCapacity, Pal::Platform> TrackedCmdLocationVec;

    static uint32 GetTrackedCmdLocationArraySizeInBytes()
    {
        return sizeof(TrackedCmdLocationArray);
    }

    static TrackedCmdLocationArray* CreateTrackedCmdLocationArray(
        void*           pMemory,
        Pal::Platform*  pPlatform);

    void Reset()
    {
        m_lastLocation = TrackedCmdLocationRef(this, BadIndex);
        m_clientId = InvalidClientId;
        m_locations.Clear();
    }

    void Destroy();

    uint64 GetClientId() const
    {
        return m_clientId;
    }

    Result SetClientId(
        uint64 clientId);

    Util::uint32 GetTotalSize() const
    {
        return m_locations.size();
    }

    const TrackedCmdLocationVec& GetLocationsVec() const
    {
        return m_locations;
    }

    TrackedCmdLocationVec& UseLocationsVec()
    {
        return m_locations;
    }

    Pal::Result MakeNext(
        TrackedCmdLocationRef* pResult);

    const TrackedCmdLocationRef GetLast() const
    {
        return m_lastLocation;
    }

    bool IsLast(
        TrackedCmdLocationRef const& location) const
    {
        return location == m_lastLocation;
    }

private:
    TrackedCmdLocationVec   m_locations;
    Pal::Platform*          m_pPlatform;
    uint64                  m_clientId;
    TrackedCmdLocationRef   m_lastLocation;

    TrackedCmdLocationArray(
        Pal::Platform* pPlatform);

    ~TrackedCmdLocationArray() = default;
};

} // namespace CmdDisassembly
} // namespace Pal
