// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once 

/*
* When built with CFFI, headers must not include directives
* Also CFFI does not support function override with different set of parameters
* This header can only be include by C/C++ code. CFFI build must use Interface.h and rocprofvis_interface_types.h
* This header makes interface functions declarations pure C
* This header also adds property access methods, which return values using pointer references
* and status of the operation as return value
*/

#include "rocprofvis_c_interface_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
 #include "rocprofvis_interface.h" 
#ifdef __cplusplus
}
#endif

/***********************************Universal property getters**************************************
*
* There are 10 property getter methods. Two methods to get property of each type:
*       uint64, int64, double, char* and rocprofvis_dm_handle_t
* First set of methods returns values as pointer references and result of operation as return value
* Second set returns property values directly. No result of operation is returned.
* In second case if operation fails, 0 or nullptr will be returned.
****************************************************************************************************/

// Method to get object property value as uint64. Value is returned as pointer reference.
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_uint64(
                                    rocprofvis_dm_handle_t, 	            // object handle
                                    rocprofvis_dm_property_t,               // property enumeration
                                    rocprofvis_dm_property_index_t,         // indexed property index
                                    uint64_t*                               // return value as uint64_t* 
                                    );                                     

// Method to get object property value as int64. Value is returned as pointer reference.
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_int64(
                                    rocprofvis_dm_handle_t,                 // object handle 	
                                    rocprofvis_dm_property_t,               // property enumeration
                                    rocprofvis_dm_property_index_t,         // indexed property index
                                    int64_t*                                // return value as int64_t*
                                    ); 

// Method to get object property value as double. Value is returned as pointer reference.
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_double(
                                    rocprofvis_dm_handle_t,                 // object handle 	
                                    rocprofvis_dm_property_t,               // property enumeration               
                                    rocprofvis_dm_property_index_t,         // indexed property index
                                    double*                                 // return value as double*
                                    ); 

// Method to get object property value as char*. Value is returned as pointer reference.
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_charptr(
                                    rocprofvis_dm_handle_t,                 // object handle	
                                    rocprofvis_dm_property_t,               // property enumeration
                                    rocprofvis_dm_property_index_t,         // indexed property index
                                    char**                                  // return value as char**
                                    ); 

// Method to get object property value as handle (void*). Value is returned as pointer reference.
rocprofvis_dm_result_t  rocprofvis_dm_get_property_as_handle(
                                    rocprofvis_dm_handle_t,                 // object handle 	
                                    rocprofvis_dm_property_t,               // property enumeration
                                    rocprofvis_dm_property_index_t,         // indexed property index
                                    rocprofvis_dm_handle_t*                 // return value as void**
                                    ); 

