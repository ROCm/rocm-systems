/* Copyright (c) 2021-2024 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "exception.h"
#include "logging.h"
#include "memory.h"

#include <cinttypes>

namespace amd::dbgapi
{

void
exception_t::print_message () const noexcept
{
  if (const char *message = what (); message != nullptr && *message)
    dbgapi_log (AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR, "%s", message);
}

memory_access_error_t::memory_access_error_t (
  const address_space_t &address_space,
  amd_dbgapi_segment_address_t segment_address, std::string message)
  : api_error_t (AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS,
                 string_printf ("Cannot access memory at %s#%#" PRIx64,
                                address_space.name ().c_str (),
                                segment_address)
                   + (message.empty () ? "" : (": " + message))),
    m_address (std::make_pair (std::cref (address_space), segment_address))
{
}

} /* namespace amd::dbgapi */
