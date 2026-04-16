#include "canonicalize.hpp"

using namespace llvm;

namespace transpiler {

std::string canonicalizeMnemonic(StringRef mn) {
  // GFX12 global memory: _bNN → _dword[xN]
  if (mn == "global_load_b32")   return "global_load_dword";
  if (mn == "global_load_b64")   return "global_load_dwordx2";
  if (mn == "global_load_b128")  return "global_load_dwordx4";
  if (mn == "global_load_b96")   return "global_load_dwordx3";
  if (mn == "global_store_b32")  return "global_store_dword";
  if (mn == "global_store_b64")  return "global_store_dwordx2";
  if (mn == "global_store_b128") return "global_store_dwordx4";
  if (mn == "global_store_b96")  return "global_store_dwordx3";
  if (mn == "global_store_b16")  return "global_store_short";
  if (mn == "global_store_b8")   return "global_store_byte";
  if (mn == "global_load_u16")   return "global_load_ushort";
  if (mn == "global_load_i16")   return "global_load_sshort";
  if (mn == "global_load_u8")    return "global_load_ubyte";
  if (mn == "global_load_i8")    return "global_load_sbyte";
  if (mn == "global_load_d16_hi_b16") return "global_load_short_d16_hi";

  // GFX12 SMEM: s_load_bNN → s_load_dword[xN]
  if (mn == "s_load_b32")   return "s_load_dword";
  if (mn == "s_load_b64")   return "s_load_dwordx2";
  if (mn == "s_load_b96")   return "s_load_dwordx3";
  if (mn == "s_load_b128")  return "s_load_dwordx4";
  if (mn == "s_load_b256")  return "s_load_dwordx8";

  // GFX12 scalar carry ops — same semantics as the CDNA names
  if (mn == "s_add_co_i32")  return "s_add_u32";
  if (mn == "s_add_co_u32")  return "s_add_u32";
  if (mn == "s_sub_co_i32")  return "s_sub_u32";
  if (mn == "s_sub_co_u32")  return "s_sub_u32";
  if (mn == "s_addc_co_i32") return "s_addc_u32";
  if (mn == "s_addc_co_u32") return "s_addc_u32";
  if (mn == "s_add_co_ci_u32") return "s_addc_u32";
  if (mn == "s_subb_co_i32") return "s_subb_u32";
  if (mn == "s_sub_co_ci_u32") return "s_subb_u32";

  // GFX12 flat memory: flat_load/flat_store _bNN renames
  if (mn == "flat_load_b32")   return "flat_load_dword";
  if (mn == "flat_load_b64")   return "flat_load_dwordx2";
  if (mn == "flat_load_b128")  return "flat_load_dwordx4";
  if (mn == "flat_load_b96")   return "flat_load_dwordx3";
  if (mn == "flat_store_b32")  return "flat_store_dword";
  if (mn == "flat_store_b64")  return "flat_store_dwordx2";
  if (mn == "flat_store_b128") return "flat_store_dwordx4";
  if (mn == "flat_store_b96")  return "flat_store_dwordx3";
  if (mn == "flat_load_u8")    return "flat_load_ubyte";
  if (mn == "flat_load_i8")    return "flat_load_sbyte";
  if (mn == "flat_load_u16")   return "flat_load_ushort";
  if (mn == "flat_load_i16")   return "flat_load_sshort";
  if (mn == "flat_store_b8")   return "flat_store_byte";
  if (mn == "flat_store_b16")  return "flat_store_short";

  // GFX12 logic renames: s_and_not1/s_or_not1 → s_andn2/s_orn2
  if (mn == "s_and_not1_b32")  return "s_andn2_b32";
  if (mn == "s_and_not1_b64")  return "s_andn2_b64";
  if (mn == "s_or_not1_b32")   return "s_orn2_b32";
  if (mn == "s_or_not1_b64")   return "s_orn2_b64";
  if (mn == "s_not1_b32")      return "s_not_b32";
  if (mn == "s_not1_b64")      return "s_not_b64";

  // GFX12 DS: ds_load/ds_store use _b suffix
  if (mn == "ds_load_b32")   return "ds_read_b32";
  if (mn == "ds_load_b64")   return "ds_read_b64";
  if (mn == "ds_load_b128")  return "ds_read_b128";
  if (mn == "ds_store_b32")  return "ds_write_b32";
  if (mn == "ds_store_b64")  return "ds_write_b64";
  if (mn == "ds_store_b128") return "ds_write_b128";
  if (mn == "ds_load_u16")   return "ds_read_u16";
  if (mn == "ds_load_i16")   return "ds_read_i16";
  if (mn == "ds_load_u8")    return "ds_read_u8";
  if (mn == "ds_load_i8")    return "ds_read_i8";
  if (mn == "ds_store_b16")  return "ds_write_b16";
  if (mn == "ds_store_b8")   return "ds_write_b8";

  // GFX12 buffer: buffer_load/store _bNN → _dword[xN]
  if (mn == "buffer_load_b32")   return "buffer_load_dword";
  if (mn == "buffer_load_b64")   return "buffer_load_dwordx2";
  if (mn == "buffer_load_b128")  return "buffer_load_dwordx4";
  if (mn == "buffer_store_b32")  return "buffer_store_dword";
  if (mn == "buffer_store_b64")  return "buffer_store_dwordx2";
  if (mn == "buffer_store_b128") return "buffer_store_dwordx4";

  // GFX12 IEEE "num" renames → standard min/max
  if (mn == "v_max_num_f32")  return "v_max_f32";
  if (mn == "v_min_num_f32")  return "v_min_f32";
  if (mn == "v_max3_num_f32") return "v_max3_f32";
  if (mn == "v_min3_num_f32") return "v_min3_f32";

  // GFX12 SOPK carry rename
  if (mn == "s_addk_co_i32") return "s_addk_i32";

  return mn.str();
}

} // namespace transpiler
