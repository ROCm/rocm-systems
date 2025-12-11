// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"
#include "rocprofvis_dm_table_row.h"
#include <vector>

namespace RocProfVis
{
namespace DataModel
{

class Trace;

// Table is class of table object, container of TableRow objects
class Table : public DmBase {
    public:
        // Table class constructor
        // @param  ctx - Trace object context
        // @param description - pointer to table description string
        // @param query - pointer to table query string

        Table(  Trace* ctx, rocprofvis_dm_charptr_t description, rocprofvis_dm_charptr_t query); 
        // Table class destructor, not required unless declared as virtual
        ~Table(){}
        // Method to add column name to a vector array of columns
        // @param column_name - pointer to column name string
        // @return status of operation
        rocprofvis_dm_result_t                  AddColumn(rocprofvis_dm_charptr_t column_name);
        // Method to create and add empty row object to a vector array of rows
        // @param column_name - pointer to column name string
        // @return status of operation
        rocprofvis_dm_table_row_t               AddRow();
        // Returns number of columns
        rocprofvis_dm_size_t                    GetNumberOfColumns() {return m_columns.size();};
        // Returns number of rows
        rocprofvis_dm_size_t                    GetNumberOfRows() {return m_rows.size();};
        // Method to get amount of memory used by Table class object
        // @return used memory size
        rocprofvis_dm_size_t                    GetMemoryFootprint();

        // Returns table id
        uint64_t                                Id() { return m_id; };
        // Returns trace context pointer
        Trace*                                  Ctx() {return m_ctx;};
        // Returns pointer to query string
        rocprofvis_dm_charptr_t                 Query() {return m_query.c_str();}
        // Returns pointer to description string
        rocprofvis_dm_charptr_t                 Description() {return m_description.c_str();}
        // Returns class mutex
        std::shared_mutex*                      Mutex() override { return &m_lock; }

        // Method to read Table object property as uint64
        // @param property - property enumeration rocprofvis_dm_table_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation   
        rocprofvis_dm_result_t                  GetPropertyAsUint64(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            uint64_t* value) override;  
        // Method to read Table object property as char*
        // @param property - property enumeration rocprofvis_dm_table_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to char* return value
        // @return status of operation  
        rocprofvis_dm_result_t                  GetPropertyAsCharPtr(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            char** value) override;
        // Method to read Table object property as handle (aka void*)
        // @param property - property enumeration rocprofvis_dm_table_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to handle (aka void*) return value
        // @return status of operation
        rocprofvis_dm_result_t                  GetPropertyAsHandle(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            rocprofvis_dm_handle_t* value) override;
#ifdef TEST
        // Method to get property symbol for testing/debugging
        // @param property - property enumeration rocprofvis_dm_table_property_t
        // @return pointer to property symbol string 
        const char*                             GetPropertySymbol(rocprofvis_dm_property_t property) override;
#endif

    private:
        // table id
        uint64_t                                       m_id;
        // pointer to Trace context object
        Trace*                                         m_ctx;
        // vector array of column names
        std::vector<std::string>                            m_columns;
        // vector array of row objects
        std::vector<std::shared_ptr<TableRow>>              m_rows;
        // table query string
        std::string                                         m_query;
        // table description string
        std::string                                         m_description;
        // object mutex, for shared access
        mutable std::shared_mutex                       m_lock;

        // Method to get pointer to column name at provided index
        // @param index - column index
        // @param column - reference to column string
        // @return status of operation
        rocprofvis_dm_result_t          GetColumnNameAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & column);
        // Method to get handle to row (TableRow) object at provided index
        // @param index - row index
        // @param row - reference to row (TableRow) object
        // @return status of operation
        rocprofvis_dm_result_t          GetRowHandleAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t & row);

};

}  // namespace DataModel
}  // namespace RocProfVis