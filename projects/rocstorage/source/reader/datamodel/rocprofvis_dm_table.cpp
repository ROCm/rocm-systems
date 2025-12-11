// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_table.h"

namespace RocProfVis
{
namespace DataModel
{

Table::Table(   Trace* ctx, 
                rocprofvis_dm_charptr_t description,
                rocprofvis_dm_charptr_t query)
: m_ctx(ctx)
, m_query(query)
, m_description(description)
{
    m_id = std::hash<std::string>{}(m_query);
}

rocprofvis_dm_size_t Table::GetMemoryFootprint(){
    size_t size = sizeof(Table);
    for (int i=0; i < m_rows.size(); i++) size+=m_rows[i].get()->GetMemoryFootprint();
    for (int i=0; i < m_columns.size(); i++) size+=m_columns[i].length();
    return size;
}

rocprofvis_dm_result_t Table::AddColumn(rocprofvis_dm_charptr_t column_name){
    try{
        m_columns.push_back(column_name);
    }
    catch(std::exception ex)
    {
        return kRocProfVisDmResultAllocFailure;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_table_row_t Table::AddRow(){
    try{
        m_rows.push_back(std::make_shared<TableRow>(this));
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Failure allocating table row object", nullptr);
    }
    return m_rows.back().get();
}

rocprofvis_dm_result_t Table::GetColumnNameAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & name){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_columns.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    name = m_columns[index].c_str();
    return kRocProfVisDmResultSuccess;
}


rocprofvis_dm_result_t   Table::GetRowHandleAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t & row){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_rows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    row = m_rows[index].get();
    return kRocProfVisDmResultSuccess;    
}


rocprofvis_dm_result_t Table::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMNumberOfTableIdUInt64: 
            *value = Id();
            return kRocProfVisDmResultSuccess;
        case kRPVDMNumberOfTableColumnsUInt64:
            *value = GetNumberOfColumns();
            return kRocProfVisDmResultSuccess;
        case kRPVDMNumberOfTableRowsUInt64:
            *value = GetNumberOfRows();
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

 rocprofvis_dm_result_t Table::GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMExtTableColumnNameCharPtrIndexed:
            return GetColumnNameAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMExtTableDescriptionCharPtr:
            *value = (char*)Description();
            return kRocProfVisDmResultSuccess;
        case kRPVDMExtTableQueryCharPtr:
            *value = (char*)Query();
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

rocprofvis_dm_result_t  Table::GetPropertyAsHandle(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMExtTableRowHandleIndexed:
            return GetRowHandleAt(index, *value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

#ifdef TEST
const char*  Table::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMNumberOfTableIdUInt64:
            return "kRPVDMNumberOfTableIdUInt64";
        case kRPVDMNumberOfTableColumnsUInt64:
            return "kRPVDMNumberOfTableColumnsUInt64";        
        case kRPVDMNumberOfTableRowsUInt64:
            return "kRPVDMNumberOfTableRowsUInt64";
        case kRPVDMExtTableColumnNameCharPtrIndexed:
            return "kRPVDMExtTableColumnNameCharPtrIndexed";
        case kRPVDMExtTableDescriptionCharPtr:
            return "kRPVDMExtTableDescriptionCharPtr";
        case kRPVDMExtTableQueryCharPtr:
            return "kRPVDMExtTableQueryCharPtr";
        case kRPVDMExtTableRowHandleIndexed:
            return "kRPVDMExtTableRowHandleIndexed";
        default:
            return "Unknown property";
    }   
}
#endif

}  // namespace DataModel
}  // namespace RocProfVis