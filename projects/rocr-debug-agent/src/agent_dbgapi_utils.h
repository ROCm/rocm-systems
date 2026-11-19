// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef ROCM_DEBUG_AGENT_DBGAPI_UTILS_H
#define ROCM_DEBUG_AGENT_DBGAPI_UTILS_H

#include <amd-dbgapi/amd-dbgapi.h>

#include <type_traits>

namespace amd::debug_agent
{

/* Map wave stop reason bits to exception bitmask for resume.
   Handles multi-bit stop reasons by processing each bit individually.
   \param stop_reason Wave stop reason bitmask (may contain multiple bits).
   \return Exception bitmask suitable for amd_dbgapi_wave_resume.  */
std::underlying_type_t<amd_dbgapi_exceptions_t> map_stop_reason_to_exceptions (
  std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason);

/* Convert a single wave stop reason to its string name.
   \param reason Stop reason enumeration value (single bit expected).
   \return String name of the stop reason (e.g., "BREAKPOINT"), or empty string
           if unknown.  */
const char *stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t reason);

} /* namespace amd::debug_agent */

#endif /* ROCM_DEBUG_AGENT_DBGAPI_UTILS_H */
