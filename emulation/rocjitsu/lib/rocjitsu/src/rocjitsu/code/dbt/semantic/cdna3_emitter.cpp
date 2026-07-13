// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/semantic/cdna3_emitter.h"

#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"

#include <cstring>

namespace rocjitsu {
namespace {

template <typename MachineInst>
[[nodiscard]] Cdna3Emitter::WordPair encode_pair(const MachineInst &inst) {
  static_assert(sizeof(MachineInst) == sizeof(uint32_t) * 2,
                "CDNA3 emitted instruction encodings are 64-bit");
  uint32_t words[2]{};
  std::memcpy(words, &inst, sizeof(inst));
  return {words[0], words[1]};
}

} // namespace

Cdna3Emitter::WordPair Cdna3Emitter::ds(uint16_t op, uint8_t vdst, uint8_t addr, uint8_t data0,
                                        uint8_t data1, uint8_t offset0, uint8_t offset1) {
  cdna3::DsMachineInst dst{};
  dst.encoding = 0x36;
  dst.op = op & 0xFF;
  dst.offset0 = offset0;
  dst.offset1 = offset1;
  dst.addr = addr;
  dst.data0 = data0;
  dst.data1 = data1;
  dst.vdst = vdst;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::mubuf(const MubufOperands &src, uint16_t op, uint8_t vdata) {
  cdna3::MubufMachineInst dst{};
  dst.encoding = 0x38;
  dst.op = op & 0x7F;
  dst.offset = src.offset;
  dst.offen = src.offen;
  dst.idxen = src.idxen;
  dst.sc0 = src.sc0;
  dst.sc1 = src.sc1;
  dst.lds = 0;
  dst.nt = src.nt;
  dst.vaddr = src.vaddr;
  dst.vdata = vdata;
  dst.srsrc = src.srsrc;
  dst.acc = 0;
  dst.soffset = src.soffset;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::flat_global_load(const FlatGlobalOperands &src, uint16_t op,
                                                      uint8_t vdst) {
  cdna3::FlatMachineInst dst{};
  dst.encoding = 0x37;
  dst.offset = src.signed_offset13 & 0x0FFF;
  dst.pad_12 = (src.signed_offset13 >> 12) & 0x1;
  dst.lds = 0;
  dst.seg = 2;
  dst.sc0 = src.sc0;
  dst.nt = src.nt;
  dst.op = op & 0x7F;
  dst.sc1 = src.sc1;
  dst.addr = src.addr;
  dst.saddr = src.saddr;
  dst.acc = src.acc;
  dst.vdst = vdst;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::flat_global_store(const FlatGlobalOperands &src, uint16_t op,
                                                       uint8_t data) {
  cdna3::FlatMachineInst dst{};
  dst.encoding = 0x37;
  dst.offset = src.signed_offset13 & 0x0FFF;
  dst.pad_12 = (src.signed_offset13 >> 12) & 0x1;
  dst.lds = 0;
  dst.seg = 2;
  dst.sc0 = src.sc0;
  dst.nt = src.nt;
  dst.op = op & 0x7F;
  dst.sc1 = src.sc1;
  dst.addr = src.addr;
  dst.data = data;
  dst.saddr = src.saddr;
  dst.acc = src.acc;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::flat_scratch_dword(uint16_t op, uint8_t vgpr,
                                                        uint32_t byte_offset, bool is_load) {
  cdna3::FlatScratchMachineInst dst{};
  dst.encoding = 0x37;
  dst.op = op & 0x7F;
  dst.offset = byte_offset & 0x1FFF;
  dst.seg = 1;
  dst.sve = 0;
  dst.saddr = 0x7F;
  if (is_load)
    dst.vdst = vgpr;
  else
    dst.data = vgpr;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::smem_load_dwordx2(uint8_t dst_sgpr, uint8_t sbase_sgpr,
                                                       uint32_t byte_offset) {
  cdna3::SmemMachineInst dst{};
  dst.encoding = 0x30;
  dst.op = cdna3::kSLoadDwordx2Smem;
  dst.sbase = (sbase_sgpr / 2) & 0x3F;
  dst.sdata = dst_sgpr & 0x7F;
  dst.imm = 1;
  dst.offset = byte_offset & 0x1FFFFF;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::smem_load_dword(uint8_t dst_sgpr, uint8_t sbase_sgpr,
                                                     uint32_t byte_offset) {
  cdna3::SmemMachineInst dst{};
  dst.encoding = 0x30;
  dst.op = cdna3::kSLoadDwordSmem;
  dst.sbase = (sbase_sgpr / 2) & 0x3F;
  dst.sdata = dst_sgpr & 0x7F;
  dst.imm = 1;
  dst.offset = byte_offset & 0x1FFFFF;
  return encode_pair(dst);
}

Cdna3Emitter::WordPair Cdna3Emitter::vop2_literal(uint16_t op, uint8_t vdst, uint8_t vsrc1,
                                                  uint32_t literal) {
  cdna3::Vop2InstLiteralMachineInst dst{};
  dst.src0 = 0xFF;
  dst.vsrc1 = vsrc1;
  dst.vdst = vdst;
  dst.op = op & 0x3F;
  dst.encoding = 0;
  dst.simm32 = literal;
  return encode_pair(dst);
}

} // namespace rocjitsu
