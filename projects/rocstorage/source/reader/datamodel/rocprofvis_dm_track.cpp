// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_track.h"
#include "rocprofvis_dm_trace.h"
#include "rocprofvis_dm_pmc_track_slice.h"
#include "rocprofvis_dm_event_track_slice.h"

namespace RocProfVis
{
namespace DataModel
{

Track::Track( Trace* ctx,
                rocprofvis_dm_track_params_t* params) :
                m_ctx(ctx),
                m_track_params(params) {
                    rocprofvis_dm_event_id_t id = { 0 };
                    m_ext_data = std::make_unique<ExtData>(ctx,id);
                    params->extdata = m_ext_data.get();
                };

std::shared_mutex* Track::Mutex()
{
    return m_ctx->Mutex();
}

rocprofvis_dm_slice_t  Track::AddSlice(rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED, nullptr);
    if(m_track_params->process.category == kRocProfVisDmPmcTrack)
    {
        try{
            m_slices.push_back(std::make_shared<PmcTrackSlice>(this, start, end));
        }
        catch(std::exception ex)
        {
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Falure allocating PMC track time slice!", nullptr);
        }
        return m_slices.back().get();
    } else
    {
        try{
            m_slices.push_back(std::make_shared<EventTrackSlice>(this, start, end));
        }
        catch(std::exception ex)
        {
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Falure allocating event track time slice!", nullptr);
        }
        return m_slices.back().get();
    }
}

rocprofvis_dm_result_t Track::DeleteSliceAtTime(rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end)
{
    // To delete single vector array element thread-safe, we must retain a local copy 
    // The element is protected by its own mutex and will be deleted outside the scope when mutex is unlocked
    std::shared_ptr<TrackSlice>  slice = nullptr;
    {
        TimedLock<std::unique_lock<std::shared_mutex>> lock(*Mutex(), __func__, this);
        rocprofvis_dm_index_t                          index = 0;
        rocprofvis_dm_result_t result = GetSliceIndexAtTime(start, end, index);
        if(result != kRocProfVisDmResultSuccess) return result;
        slice = m_slices[index];
        m_slices.erase(m_slices.begin() + index);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t
Track::DeleteSliceByHandle(rocprofvis_dm_slice_t slice)
{
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*Mutex(), __func__, this);
    std::shared_ptr<TrackSlice>                    item;
    auto                                           it = std::find_if(
        m_slices.begin(), m_slices.end(),
        [slice](const std::shared_ptr<TrackSlice>& ptr) { return ptr.get() == slice; });
    if(it != m_slices.end())
    {
        item = *it;
        m_slices.erase(it);
        return kRocProfVisDmResultSuccess;
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t Track::DeleteAllSlices()
{
    // To delete all vector array elements thread-safe, we must swap its content with local array while protected by main mutex
    // The elements of local array are protected by their own mutexes and will be deleted outside the scope when mutexes are unlocked 
    std::vector<std::shared_ptr<TrackSlice>> slices;
    {
        TimedLock<std::unique_lock<std::shared_mutex>> lock(*Mutex(), __func__, this);
        slices.swap(m_slices);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Track::GetSliceAtIndex(rocprofvis_dm_property_index_t index,  rocprofvis_dm_slice_t & slice)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_slices.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultInvalidParameter);
    slice = m_slices[index].get();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Track::GetSliceAtTime(rocprofvis_dm_timestamp_t time,  rocprofvis_dm_slice_t & slice)
{
    rocprofvis_dm_result_t result = kRocProfVisDmResultNotLoaded;
    for (int i = 0; i < m_slices.size(); i++)
    {
        uint64_t hash_time =
            hash_combine(m_slices[i]->StartTime(), m_slices[i]->EndTime()); 
        if(time == hash_time)
        {
            if (result == kRocProfVisDmResultSuccess)
            {
                return result;
            }
            slice = m_slices[i].get();
            result = kRocProfVisDmResultSuccess;
        }
    }
    return result;
}

rocprofvis_dm_result_t Track::GetSliceIndexAtTime(rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end, 
                                                                                     rocprofvis_dm_index_t& index)
{
    for (int i = 0; i < m_slices.size(); i++)
    {
        if(start == m_slices[i]->StartTime() && end == m_slices[i]->EndTime())
        {
            index = i;
            return kRocProfVisDmResultSuccess;
        }
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_size_t   Track::GetMemoryFootprint()
{
    rocprofvis_dm_size_t size = 0;
    for (int i = 0; i < m_slices.size(); i++)
    {
        size+=m_slices[i].get()->GetMemoryFootprint();
    }
    return size;
}


rocprofvis_dm_charptr_t  Track::CategoryString(){
    switch(m_track_params->process.category)
    {
        case kRocProfVisDmPmcTrack: return "Counter";
        case kRocProfVisDmRegionTrack: return "CPU Thread";
        case kRocProfVisDmRegionMainTrack: return "Thread";
        case kRocProfVisDmRegionSampleTrack: return "Sample Thread";
        case kRocProfVisDmKernelDispatchTrack: return "GPU Queue";
        case kRocProfVisDmMemoryAllocationTrack: return "Memory allocation";
        case kRocProfVisDmMemoryCopyTrack: return "Memory copy";
        case kRocProfVisDmSQTTTrack: return "Shader Execution";
        case kRocProfVisDmNICTrack: return "Network Activity";
        case kRocProfVisDmStreamTrack: return "GPU Stream";
    }
    return "Invalid track";
}

rocprofvis_dm_value_t Track::GetHistogramBucketValueAt(size_t index)
{ 
    auto it = m_track_params->histogram.find(index);
    if (it == m_track_params->histogram.end())
    {
        return 0;
    }
    else
    {
        return it->second;
    }
}

rocprofvis_dm_result_t  Track::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params->extdata, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED, kRocProfVisDmResultNotLoaded);
    switch(property)
    {
        case kRPVDMTrackCategoryEnumUInt64:           
            *value = Category();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackIdUInt64:
            *value = TrackId();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackNodeIdUInt64:
            *value = NodeId();
            return kRocProfVisDmResultSuccess;
        case kRPVDMNumberOfSlicesUInt64:
            *value = NumberOfSlices();
            return kRocProfVisDmResultSuccess;
        case kRPVDMNumberOfTrackExtDataRecordsUInt64:
            *value = m_ext_data.get()->GetNumberOfRecords();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackMemoryFootprintUInt64:
            *value = GetMemoryFootprint();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackNumRecordsUInt64:
            *value = NumRecords();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackMinimumTimestampUInt64:
            *value = MinTimestamp();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackMaximumTimestampUInt64:
            *value = MaxTimestamp();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackHistogramEventDensityUInt64Indexed:
            *value = GetHistogramBucketValueAt(index);
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

 rocprofvis_dm_result_t   Track::GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params->extdata, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED, kRocProfVisDmResultNotLoaded);
    switch(property)
    {
        case kRPVDMTrackExtDataCategoryCharPtrIndexed:
            return m_ext_data.get()->GetPropertyAsCharPtr(kRPVDMExtDataCategoryCharPtrIndexed, index, value);
	    case kRPVDMTrackExtDataNameCharPtrIndexed:
            return m_ext_data.get()->GetPropertyAsCharPtr(kRPVDMExtDataNameCharPtrIndexed, index, value);
        case kRPVDMTrackExtDataValueCharPtrIndexed:
            return m_ext_data.get()->GetPropertyAsCharPtr(kRPVDMExtDataValueCharPtrIndexed, index, value);
        case kRPVDMTrackMainProcessNameCharPtr:
            *value = (char*)Process();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackSubProcessNameCharPtr:
            *value = (char*)SubProcess();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackCategoryEnumCharPtr:
            *value = (char*)CategoryString();
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

 
rocprofvis_dm_result_t
Track::GetPropertyAsDouble(rocprofvis_dm_property_t       property,
                           rocprofvis_dm_property_index_t index,
                           double*        value)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED,
                                 kRocProfVisDmResultNotLoaded);
    switch(property)
    {
        case kRPVDMTrackMinimumValueDouble:
            *value = MinValue();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackMaximumValueDouble:
            *value = MaxValue();
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER,
                                                kRocProfVisDmResultInvalidProperty);
    }
}


rocprofvis_dm_result_t  Track::GetPropertyAsHandle(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(m_track_params, ERROR_TRACK_PARAMETERS_NOT_ASSIGNED, kRocProfVisDmResultNotLoaded);
    switch(property)
    {
        case kRPVDMSliceHandleIndexed:
            return GetSliceAtIndex(index, *value);
        case kRPVDMSliceHandleTimed:
            return GetSliceAtTime(index, *value);
        case kRPVDMTrackDatabaseHandle:
            ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
            *value = Ctx()->Database();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTrackTraceHandle:
            ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
            *value = Ctx();
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}



#ifdef TEST
const char*  Track::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMTrackCategoryEnumUInt64:
            return "kRPVDMTrackCategoryEnumUInt64";        
        case kRPVDMTrackIdUInt64:
            return "kRPVDMTrackIdUInt64";
        case kRPVDMTrackNodeIdUInt64:
            return "kRPVDMTrackNodeIdUInt64";
        case kRPVDMNumberOfSlicesUInt64:
            return "kRPVDMNumberOfSlicesUInt64";
        case kRPVDMNumberOfTrackExtDataRecordsUInt64:
            return "kRPVDMNumberOfTrackExtDataRecordsUInt64";
        case kRPVDMTrackMemoryFootprintUInt64:
            return "kRPVDMTrackMemoryFootprintUInt64";
        case kRPVDMTrackExtDataCategoryCharPtrIndexed:
            return "kRPVDMTrackExtDataCategoryCharPtrIndexed";
        case kRPVDMTrackExtDataNameCharPtrIndexed:
            return "kRPVDMTrackExtDataNameCharPtrIndexed";
        case kRPVDMTrackExtDataValueCharPtrIndexed:
            return "kRPVDMTrackExtDataValueCharPtrIndexed";
        case kRPVDMTrackInfoJsonCharPtr:
            return "kRPVDMTrackInfoJsonCharPtr";
        case kRPVDMTrackMainProcessNameCharPtr:
            return "kRPVDMTrackMainProcessNameCharPtr";
        case kRPVDMTrackSubProcessNameCharPtr:
            return "kRPVDMTrackSubProcessNameCharPtr";
        case kRPVDMTrackCategoryEnumCharPtr:
            return "kRPVDMTrackCategoryEnumCharPtr";
        case kRPVDMSliceHandleIndexed:
            return "kRPVDMSliceHandleIndexed";
        case kRPVDMSliceHandleTimed:
            return "kRPVDMSliceHandleTimed";
        case kRPVDMTrackDatabaseHandle:
            return "kRPVDMTrackDatabaseHandle";
        case kRPVDMTrackTraceHandle:
        return "kRPVDMTrackTraceHandle";
        case kRPVDMTrackNumRecordsUInt64:
            return "kRPVDMTrackNumRecordsUInt64";
        case kRPVDMTrackMinimumTimestampUInt64:
            return "kRPVDMTrackMinimumTimestampUInt64";
        case kRPVDMTrackMaximumTimestampUInt64:
            return "kRPVDMTrackMaximumTimestampUInt64";
        case kRPVDMTrackMinimumLevelDouble:
            return "kRPVDMTrackMinimumValueDouble";
        case kRPVDMTrackMaximumLevelDouble:
            return "kRPVDMTrackMaximumValueDouble";
        default:
            return "Unknown property";
    }   
}
#endif

}  // namespace DataModel
}  // namespace RocProfVis