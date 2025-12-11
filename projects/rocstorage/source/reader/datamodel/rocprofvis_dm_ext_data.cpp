// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_ext_data.h"
#include "rocprofvis_dm_trace.h"
#include <cstdint>
#include <sstream>

namespace RocProfVis
{
namespace DataModel
{

rocprofvis_dm_size_t  ExtData::GetMemoryFootprint(){
    return sizeof(ExtData) + (sizeof(ExtDataRecord) * m_extdata_records.size());
}

rocprofvis_dm_result_t  ExtData::AddRecord( rocprofvis_db_ext_data_t & data){
    try{
        m_extdata_records.push_back(ExtDataRecord(data.category, data.name, data.data, data.type, data.category_enum));
    }
    catch(std::exception ex)
    {
        return kRocProfVisDmResultAllocFailure;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t ExtData::GetRecordNameAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & name){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_extdata_records.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    name = m_extdata_records[index].Name();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t ExtData::GetRecordDataAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t &data){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_extdata_records.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    if (m_id.value != 0) {
        data = m_extdata_records[index].Data();
        return kRocProfVisDmResultSuccess;
    }
    else
    {
        rocprofvis_dm_id_t id  = std::stoll(m_extdata_records[index].Data());
        return m_ctx->BindingInfo()->FuncFindCachedTableValue(m_ctx->Database(), m_extdata_records[index].Category(), id, m_extdata_records[index].Name(), &data);
    }
}

rocprofvis_dm_result_t ExtData::GetRecordCategoryAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t& category) {
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_extdata_records.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    category = m_extdata_records[index].Category();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t ExtData::GetRecordTypeAt(const rocprofvis_dm_property_index_t index, uint64_t & type){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_extdata_records.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    type = m_extdata_records[index].Type();
    return kRocProfVisDmResultSuccess;
}
rocprofvis_dm_result_t ExtData::GetRecordCategoryEnumAt(const rocprofvis_dm_property_index_t index, uint64_t& category_enum) {
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_extdata_records.size(),
                                 ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    category_enum = m_extdata_records[index].CategoryEnum();
    return kRocProfVisDmResultSuccess;
}
rocprofvis_dm_result_t ExtData::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMNumberOfExtDataRecordsUInt64:
            *value = GetNumberOfRecords();
            return kRocProfVisDmResultSuccess;
        case kRPVDMExtDataTypeUint64Indexed:
            return GetRecordTypeAt(index, *value);
        case kRPVDMExtDataEnumUint64Indexed:
            return GetRecordCategoryEnumAt(index, *value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}


 rocprofvis_dm_result_t ExtData::GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMExtDataCategoryCharPtrIndexed:
            return GetRecordCategoryAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMExtDataNameCharPtrIndexed:
            return GetRecordNameAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMExtDataValueCharPtrIndexed:
            return GetRecordDataAt(index, *(rocprofvis_dm_charptr_t*)value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

#ifdef TEST
const char*  ExtData::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMNumberOfExtDataRecordsUInt64:
            return "kRPVDMNumberOfExtDataRecordsUInt64";
        case kRPVDMExtDataCategoryCharPtrIndexed:
            return "kRPVDMExtDataCategoryCharPtrIndexed";
        case kRPVDMExtDataNameCharPtrIndexed:
            return "kRPVDMExtDataNameCharPtrIndexed";
        case kRPVDMExtDataValueCharPtrIndexed:
            return "kRPVDMExtDataCharPtrIndexed";
        case kRPVDMExtDataTypeUint64Indexed:
            return "kRPVDMExtDataTypeUint64Indexed";
        default:
            return "Unknown property";
    }
}
#endif

bool ExtData::HasRecord(rocprofvis_db_ext_data_t & data)
{
    auto it = find_if(m_extdata_records.begin(), m_extdata_records.end(), [&](ExtDataRecord & ext){
        return ext.Equal(data.category, data.name);
    });
    return it != m_extdata_records.end();
}

}  // namespace DataModel
}  // namespace RocProfVis