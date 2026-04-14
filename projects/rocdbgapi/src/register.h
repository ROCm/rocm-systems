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

#ifndef AMD_DBGAPI_REGISTER_H
#define AMD_DBGAPI_REGISTER_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"
#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

namespace amd::dbgapi
{

class architecture_t;

/* Registers  */

enum class amdgpu_regnum_t : uint32_t
{
  first_regnum = 0,
  first_raw = first_regnum,

  /* Raw registers.  */
  first_vgpr = first_raw,
  first_vgpr_64 = first_vgpr,

  /* 32-bit Vector registers (vgprs) for wave64 wavefronts.  */
  v0_64 = first_vgpr_64,
  v255_64 = v0_64 + 255,

  last_vgpr_64 = v255_64,
  first_accvgpr_64 = last_vgpr_64 + 1,

  /* 64-bit Accumulation Vector registers (accvgprs) for wave64 wavefronts  */
  a0_64 = first_accvgpr_64,
  a255_64 = a0_64 + 255,

  last_accvgpr_64 = a255_64,
  first_vgpr_32 = last_accvgpr_64 + 1,

  /* 32-bit Vector registers (vgprs) for wave32 wavefronts.  */
  v0_32 = first_vgpr_32,
  v255_32 = v0_32 + 255,

  last_vgpr_32 = v255_32,
  first_accvgpr_32 = last_vgpr_32 + 1,

  /* 32-bit Accumulation Vector registers (accvgprs) for wave64 wavefronts  */
  a0_32 = first_accvgpr_32,
  a255_32 = a0_32 + 255,

  last_accvgpr_32 = a255_32,
  last_vgpr = last_accvgpr_32,
  first_sgpr = last_vgpr + 1,

  /* 32-bit Scalar registers (sgprs).  */
  s0 = first_sgpr,
  s127 = s0 + 127,

  last_sgpr = s127,

  /* Shadow Scalar registers.  The shadow sgprs are the scalar registers mapped
     under flat_scratch, vcc, and xnack_mask.  */
  first_shadow_sgpr = last_sgpr + 1,
  last_shadow_sgpr = first_shadow_sgpr + 127,

  first_hwreg = last_shadow_sgpr + 1,

  /* Hardware registers (hwregs).  */

  last_hwreg = first_hwreg + 15,
  first_ttmp = last_hwreg + 1,

  /* Trap temporary registers (ttmps).  */
  ttmp4 = first_ttmp + 4,
  ttmp5,
  ttmp6,
  ttmp7,
  ttmp8,
  ttmp9,
  ttmp10,
  ttmp11,
  ttmp13 = ttmp11 + 2,

  last_ttmp = ttmp13 + 2,

  lds_0, /* First dword of the LDS backing store.  */

  last_raw = lds_0,
  first_aliased = last_raw + 1,

  /* Aliased registers:  Registers whose index depends on the wave
     configuration or on the architecture.  For example, vcc is the last
     register pair of the wave's scalar registers allocation.  For a wave with
     32 scalar registers, vcc aliases to [s30,s31].  */

  m0 = first_aliased, /* Memory Descriptor.  */
  pc,                 /* Program counter.  */
  status,             /* Status register.  */
  state_priv,         /* Status register, writable only by trap handler
                         (gfx12+).  */
  mode,               /* Mode register.  */
  trapsts,            /* Exception status registers.  */
  trap_ctrl,          /* Mask of which exceptions cause trap (gfx12+).  */
  excp_flag_priv,     /* Mask exceptions which have occured (gfx12+). */
  excp_flag_user,     /* User writable mask of which exceptions have occured
                         (gfx12+).  */
  exec_32,            /* Execution mask for wave32 wavefronts.  */
  exec_64,            /* Execution mask for wave64 wavefronts.  */
  vcc_32,             /* Vector Condition Code for wave32 wavefronts.  */
  vcc_64,             /* Vector Condition Code for wave64 wavefronts.  */
  xnack_mask_32,      /* XNACK mask for wave32 wavefronts.  */
  xnack_mask_64,      /* XNACK mask for wave64 wavefronts.  */
  flat_scratch,       /* Flat scratch.  */
  scratch_base = flat_scratch,

  /* SHADER_CYCLES_LO / SHADER_CYCLES_HI / DVGPR_ALLOC_LO / DVGPR_ALLOC_HI ?*/

  /* Align the base of the following register pairs.  */
  aligned_block = utils::align_up (flat_scratch + 1, 2),

  flat_scratch_lo = aligned_block,   /* Flat scratch lower 32 bits.  */
  scratch_base_lo = flat_scratch_lo, /* Alias in gfx12.  */
  flat_scratch_hi,                   /* Flat scratch higher 32 bits.  */
  scratch_base_hi = flat_scratch_hi, /* alias in gfx12. */
  exec_lo,                           /* Execution mask lower 32bits.  */
  exec_hi,                           /* Execution mask higher 32bits.  */
  vcc_lo,                            /* Vector Condition Code lower 32
                                        bits.  */
  vcc_hi,                            /* Vector Condition Code higher 32
                                        bits.  */
  xnack_mask_lo,                     /* XNACK mask lower 32 bits.  */
  xnack_mask_hi,                     /* XNACK mask higher 32 bits.  */

  last_aliased = xnack_mask_hi,
  first_pseudo = last_aliased + 1,

  /* Pseudo registers.  Registers that are computed from other registers. They
     cannot be directly loaded from or stored to the context save area.  For
     example, pseudo_vcc is a composite of vcc and status.vccz (when vcc is
     written, status.vccz is set to 1 if vcc == 0, and 0 otherwise).  */

  pseudo_status = first_pseudo, /* The client visible status register.  */
  pseudo_state_priv,

  pseudo_exec_32, /* Read from exec_32, write to exec_32 and status.execz.  */
  pseudo_exec_64, /* Read from exec_64, write to exec_64 and status.execz.  */
  pseudo_vcc_32,  /* Read from vcc_32, write to vcc_32 and status.vccz.  */
  pseudo_vcc_64,  /* Read from vcc_64, write to vcc_64 and status.vccz.  */

  wave_id, /* The wave's globally unique identifier.  */
  csp,     /* Conditional-branch Stack Pointer.  */

  null, /* Special register: read returns 0, write is ignored.  */

  last_pseudo = null,
  last_regnum = last_pseudo
};

constexpr size_t
operator- (amdgpu_regnum_t lhs, amdgpu_regnum_t rhs)
{
  return static_cast<std::underlying_type_t<decltype (lhs)>> (lhs)
         - static_cast<std::underlying_type_t<decltype (rhs)>> (rhs);
}

constexpr amdgpu_regnum_t
operator+ (amdgpu_regnum_t lhs, int rhs)
{
  return static_cast<amdgpu_regnum_t> (
    static_cast<std::underlying_type_t<decltype (lhs)>> (lhs) + rhs);
}

constexpr amdgpu_regnum_t
operator- (amdgpu_regnum_t lhs, int rhs)
{
  return static_cast<amdgpu_regnum_t> (
    static_cast<std::underlying_type_t<decltype (lhs)>> (lhs) - rhs);
}

constexpr std::underlying_type_t<amdgpu_regnum_t>
operator& (amdgpu_regnum_t lhs, int rhs)
{
  return static_cast<std::underlying_type_t<decltype (lhs)>> (lhs) & rhs;
}

constexpr amdgpu_regnum_t &
operator++ (amdgpu_regnum_t &regnum)
{
  regnum = regnum + 1;
  return regnum;
}

constexpr amdgpu_regnum_t
operator++ (amdgpu_regnum_t &regnum, int)
{
  amdgpu_regnum_t prev = regnum;
  regnum = regnum + 1;
  return prev;
}

constexpr bool
is_pseudo_register (amdgpu_regnum_t regnum)
{
  return regnum >= amdgpu_regnum_t::first_pseudo
         && regnum <= amdgpu_regnum_t::last_pseudo;
}

constexpr size_t amdgpu_vgprs_32_count
  = amdgpu_regnum_t::last_vgpr_32 - amdgpu_regnum_t::first_vgpr_32 + 1;
constexpr size_t amdgpu_vgprs_64_count
  = amdgpu_regnum_t::last_vgpr_64 - amdgpu_regnum_t::first_vgpr_64 + 1;
constexpr size_t amdgpu_accvgprs_64_count
  = amdgpu_regnum_t::last_accvgpr_64 - amdgpu_regnum_t::first_accvgpr_64 + 1;
constexpr size_t amdgpu_sgprs_count
  = amdgpu_regnum_t::last_sgpr - amdgpu_regnum_t::first_sgpr + 1;
constexpr size_t amdgpu_hwregs_count
  = amdgpu_regnum_t::last_hwreg - amdgpu_regnum_t::first_hwreg + 1;
constexpr size_t amdgpu_ttmps_count
  = amdgpu_regnum_t::last_ttmp - amdgpu_regnum_t::first_ttmp + 1;

/* Register class.  */

class register_class_t
  : public detail::handle_object<amd_dbgapi_register_class_id_t>
{
public:
  using register_map_t = std::map<amdgpu_regnum_t, amdgpu_regnum_t>;

  register_class_t (amd_dbgapi_register_class_id_t register_class_id,
                    const architecture_t &architecture, std::string name)
    : handle_object (register_class_id), m_name (std::move (name)),
      m_architecture (architecture)
  {
  }

  /* Add a range of registers to this register class.  Return true if all
     registers within that range are successfully inserted.  */
  bool add_registers (amdgpu_regnum_t first, amdgpu_regnum_t last);

  /* Remove a range of registers from this register class.  Return true if all
     registers within that range are successfully removed.  */
  bool remove_registers (amdgpu_regnum_t first, amdgpu_regnum_t last);

  const std::string &name () const { return m_name; }

  bool contains (amdgpu_regnum_t regnum) const;
  std::set<amdgpu_regnum_t> register_set () const;

  void get_info (amd_dbgapi_register_class_info_t query, size_t value_size,
                 void *value) const;

  const architecture_t &architecture () const { return m_architecture; }

private:
  std::string const m_name;
  register_map_t m_register_map;
  const architecture_t &m_architecture;
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_REGISTER_H */
