// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"

namespace RocProfVis
{
namespace DataModel
{

// StackFrameRecord  is a data storage class for stack trace parameters
class StackFrameRecord 
{
    public:
        // FlowRecord class constructor. Object stores data flow endpoint parameters
        // @param symbol - stack frame method symbol
        // @param args - stack frame method arguments
        // @param line - stack frame code line
        // @param depth - stack frame depth
        StackFrameRecord(rocprofvis_dm_charptr_t symbol, rocprofvis_dm_charptr_t args, rocprofvis_dm_charptr_t line, uint32_t depth):
            m_symbol(symbol), m_args(args), m_code_line(line), m_depth(depth) {};
        // Returns pointer to symbol string
        rocprofvis_dm_charptr_t         Symbol() {return m_symbol.c_str();}
        // Returns pointer to argument string
        rocprofvis_dm_charptr_t         Args() {return m_args.c_str();}
        // Returns pointer to code line string
        rocprofvis_dm_charptr_t         CodeLine() {return m_code_line.c_str();}
        // returns stack frame depth
        uint32_t                        Depth() {return m_depth;}
    private:
        // stack frame symbol string
        rocprofvis_dm_string_t           m_symbol;
        // stack frame arguments string
        rocprofvis_dm_string_t           m_args;
        // stack frame code line  string
        rocprofvis_dm_string_t           m_code_line;
        // stack frame depth
        uint32_t                         m_depth;
};

}  // namespace DataModel
}  // namespace RocProfVis