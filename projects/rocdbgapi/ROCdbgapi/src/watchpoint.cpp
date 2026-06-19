/* Copyright (c) 2019-2024 Advanced Micro Devices, Inc.

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

#include "watchpoint.h"
#include "amd-dbgapi.h"
#include "debug.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "process.h"
#include "utils.h"

#include <limits>

namespace amd::dbgapi
{

watchpoint_t::watchpoint_t (amd_dbgapi_watchpoint_id_t watchpoint_id,
                            process_t &process,
                            amd_dbgapi_global_address_t requested_address,
                            amd_dbgapi_size_t requested_size,
                            amd_dbgapi_watchpoint_kind_t kind)
  : handle_object (watchpoint_id), m_requested_address (requested_address),
    m_requested_size (requested_size), m_kind (kind), m_process (process)
{
  dbgapi_assert (requested_size && "requested size cannot be 0");

  /* The mask used to match an address range is in the form:

             47       39        Y       23       15        X     0
     Mask:   11111111 11111111 11xxxxxx xxxxxxxx xxxxxxxx xx000000

     Only the bits in mask[Y-1:X] (x`s) are user programmable. The x`s are what
     this routine is computing before passing the mask to
     agent_t::insert_watchpoint ().

     architecture_t::watchpoint_mask_bits () returns a mask (XBits) with 1`s
     where the x`s are located:

             47       39        Y       23       15        X     0
     XBits:  00000000 00000000 00111111 11111111 11111111 11000000
             [        A         ][             B           ][  C ]

     With the x`s determined for a given range, an address watch match is
     checked with:

     Match := (AccessAddress & Mask) == MatchedAddress

     The mask required to match a given address range is obtained by replacing
     the "Stable" bits between the first and last addresses of the range with
     ones, e.g.:

             47       39        Y       23       15        X     0
     First:  01111111 11111110 11100111 00000100 00000000 01000100
     Last:   01111111 11111110 11100111 00000100 00000000 01001000

     Stable := ~(next_power_of_2 (Start ^ End) - 1)

             47       39        Y       23       15        X     0
     Stable: 11111111 11111111 11111111 11111111 11111111 11110000

     If (Stable[47:Y] contains any 0 bits, a match cannot happen, and the
     watchpoint is rejected.

     The smallest adjusted_size is (1 << X).

     The adjusted mask (aMask) and adjusted address (aAddr) sent to
     agent_t::insert_watchpoint () are:

             47       39        Y       23       15        X     0
     aMask:  11111111 11111111 11111111 11111111 11111111 11000000
     aAddr:  01111111 11111110 11100111 00000100 00000000 01000000
   */

  amd_dbgapi_global_address_t first_address = requested_address;
  amd_dbgapi_global_address_t last_address
    = first_address + requested_size - 1;

  amd_dbgapi_global_address_t stable_bits
    = -utils::next_power_of_two ((first_address ^ last_address) + 1);

  /* programmable_mask_bits is the intersection of all the process' agents
     capabilities.  architecture_t::watchpoint_mask_bits returns a mask
     with 1 bits in the positions that can be programmed (x`s).  */
  amd_dbgapi_global_address_t programmable_mask_bits{
    std::numeric_limits<decltype (programmable_mask_bits)>::max ()
  };
  for (auto &&agent : process.range<agent_t> ())
    programmable_mask_bits &= agent.os_info ().address_watch_mask_bits;

  amd_dbgapi_global_address_t field_B = programmable_mask_bits;
  amd_dbgapi_global_address_t field_A = ~(field_B | (field_B - 1));
  amd_dbgapi_global_address_t field_C = ~(field_A | field_B);

  /* Check that the required mask is within the agents capabilities.  */
  if (stable_bits < field_A)
    {
      /* Set the mask to the smallest range that includes first_address and
         covers as much of first_address..last_address as possible.  The
         smallest range that includes the first_address extends up to the end
         of the largest range that covers it.  So set last_address to that
         boundary and compute the stable_bits again.  This time the stable_bits
         mask must be in the agent capabilities.  */
      last_address = ((first_address + ~field_A) & field_A) - 1;
      stable_bits
        = -utils::next_power_of_two ((first_address ^ last_address) + 1);
      dbgapi_assert (stable_bits >= field_A);
    }

  m_size = -(stable_bits & ~field_C);
  m_address = requested_address & -m_size;
}

void
watchpoint_t::get_info (amd_dbgapi_watchpoint_info_t query, size_t value_size,
                        void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_WATCHPOINT_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;
    case AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS:
      utils::get_info (value_size, value, address ());
      return;
    case AMD_DBGAPI_WATCHPOINT_INFO_SIZE:
      utils::get_info (value_size, value, size ());
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_set_watchpoint (amd_dbgapi_process_id_t process_id,
                           amd_dbgapi_global_address_t address,
                           amd_dbgapi_size_t size,
                           amd_dbgapi_watchpoint_kind_t kind,
                           amd_dbgapi_watchpoint_id_t *watchpoint_id)
{
  TRACE_BEGIN (param_in (process_id), make_hex (param_in (address)),
               param_in (size), param_in (kind), param_in (watchpoint_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    process_t *process = process_t::find (process_id);

    if (process == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);

    if (process->is_frozen ())
      THROW (AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);

    if (process->watchpoint_shared_kind ()
        == AMD_DBGAPI_WATCHPOINT_SHARE_KIND_UNSUPPORTED)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);

    if (!size || watchpoint_id == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    switch (kind)
      {
      case AMD_DBGAPI_WATCHPOINT_KIND_LOAD:
      case AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW:
      case AMD_DBGAPI_WATCHPOINT_KIND_RMW:
      case AMD_DBGAPI_WATCHPOINT_KIND_ALL:
        break;
      default:
        THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
      }

    watchpoint_t &watchpoint
      = process->create<watchpoint_t> (*process, address, size, kind);
    auto destroy_watchpoint
      = utils::make_scope_fail ([&] () { process->destroy (&watchpoint); });

    process->insert_watchpoint (watchpoint);
    *watchpoint_id = watchpoint.id ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE,
         AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
  TRACE_END (make_ref (param_out (watchpoint_id)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_remove_watchpoint (amd_dbgapi_watchpoint_id_t watchpoint_id)
{
  TRACE_BEGIN (param_in (watchpoint_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    watchpoint_t *watchpoint = find (watchpoint_id);

    if (watchpoint == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID);

    process_t &process = watchpoint->process ();
    if (process.is_frozen ())
      THROW (AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);

    process.remove_watchpoint (*watchpoint);
    process.destroy (watchpoint);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID,
         AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
  TRACE_END ();
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_watchpoint_get_info (amd_dbgapi_watchpoint_id_t watchpoint_id,
                                amd_dbgapi_watchpoint_info_t query,
                                size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (watchpoint_id), param_in (query),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    watchpoint_t *watchpoint = find (watchpoint_id);

    if (watchpoint == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID);

    watchpoint->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}
