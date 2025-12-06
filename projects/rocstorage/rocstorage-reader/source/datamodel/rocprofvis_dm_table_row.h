// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"
#include "rocprofvis_dm_base.h"
#include <vector>

namespace RocProfVis
{
namespace DataModel
{

class Table;

// TableRow is class of table row object
class TableRow : public DmBase {
    public:
        // TableRow class constructor
        // @param  ctx - Table object context
        TableRow(  Table* ctx) : m_ctx(ctx) {}; 
        // TableRow class destructor, not required unless declared as virtual
        ~TableRow(){}
        // Method to add a cell to a row
        // @param value pointer to table value string
        // @return status of operation
        rocprofvis_dm_result_t          AddCellValue(rocprofvis_dm_charptr_t value);
        // Returns number of cells in a row
        rocprofvis_dm_size_t            GetNumberOfCells() {return m_values.size();};
        // Method to get amount of memory used by TableRow class object
        // @return used memory size
        rocprofvis_dm_size_t            GetMemoryFootprint();

        // Return table object context pointer
        // @return context pointer
        Table*                          Ctx() {return m_ctx;};

        std::shared_mutex*              Mutex() override;

        // Method to read TableRow object property as uint64
        // @param property - property enumeration rocprofvis_dm_table_row_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation   
        rocprofvis_dm_result_t          GetPropertyAsUint64(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            uint64_t* value) override;  
        // Method to read TableRow object property as char*
        // @param property - property enumeration rocprofvis_dm_table_row_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to char* return value
        // @return status of operation   
        rocprofvis_dm_result_t          GetPropertyAsCharPtr(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            char** value) override;
#ifdef TEST
        // Method to get property symbol for testing/debugging
        // @param property - property enumeration rocprofvis_dm_table_row_property_t
        // @return pointer to property symbol string 
        const char*                     GetPropertySymbol(rocprofvis_dm_property_t property) override;
#endif

    private:
        // Context pointer
        Table*                     m_ctx;
        // Vector array of string values
        std::vector<std::string>        m_values;

        // Method to get pointer to table value string at provided index
        // @param index - record index
        // @param value - reference to table value string
        // @return status of operation
        rocprofvis_dm_result_t          GetValueAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & value);

};

}  // namespace DataModel
}  // namespace RocProfVis