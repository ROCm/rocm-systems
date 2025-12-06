// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"

namespace RocProfVis
{
namespace DataModel
{

/*  RpvPmcRecord is a data storage class for performance counters data
**  It's POD (plain old data) class for memory usage optimization.
**  PMC (Performance metric counter) event represents a point on timeline
**  where X-dimension is timestamp anf Y-dimension is value.
*/ 
class PmcRecord 
{
    public:
        // PmcRecord class constructor
        // @param timestamp - PMC sample timestamp
        // @param value - PMC sample value
        PmcRecord(const rocprofvis_dm_timestamp_t timestamp, const rocprofvis_dm_value_t value):
            m_timestamp(timestamp), m_value(value) {};
        // Returns PMC sample timestamp
        const rocprofvis_dm_timestamp_t     Timestamp() {return m_timestamp;}
        // Returns PMC sample value
        const rocprofvis_dm_value_t         Value() {return m_value;}
    private:
        // PMC sample timestamp
        rocprofvis_dm_timestamp_t           m_timestamp;
        // PMC sample value
        rocprofvis_dm_value_t               m_value;
};

}  // namespace DataModel
}  // namespace RocProfVis