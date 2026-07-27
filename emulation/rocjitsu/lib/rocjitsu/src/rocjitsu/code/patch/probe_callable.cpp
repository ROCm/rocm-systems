// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/probe_callable.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <cstring>
#include <memory>
#include <span>

namespace rocjitsu {

namespace {

// Bounds-checked POD read out of the raw image. memcpy (not reinterpret_cast)
// because the image buffer is only byte-aligned and T may be stricter.
//
// TODO: read_pod / read_shdr here duplicate the same helpers in
// probe_symbol.cpp; both want the shared ELF visitor noted there.
template <typename T>
[[nodiscard]] bool read_pod(std::span<const uint8_t> image, uint64_t offset, T *out) {
  if (!detail::fits_in_bounds(offset, sizeof(T), image.size()))
    return false;
  std::memcpy(out, image.data() + offset, sizeof(T));
  return true;
}

[[nodiscard]] bool read_shdr(std::span<const uint8_t> image, const Elf64_Ehdr &ehdr, uint64_t index,
                             Elf64_Shdr *out) {
  return read_pod(image, ehdr.e_shoff + index * ehdr.e_shentsize, out);
}

// Reject if any relocation in the object applies inside the probe body's value
// range [st_value, st_value + body_size). Relocation r_offset and Elf64_Sym
// st_value share a coordinate space per object type (vaddr for ET_DYN,
// section-relative for ET_REL), so the same range comparison works for both.
//
// We only consult relocation sections whose sh_info names the body's owning
// section, plus dynamic-style sections (sh_info == 0) that apply globally by
// address. A reloc inside the body means the body is not self-contained.
[[nodiscard]] bool body_has_relocation(std::span<const uint8_t> image, const Elf64_Ehdr &ehdr,
                                       const ResolvedProbeSymbol &sym, bool *out_has,
                                       std::string *error_out) {
  *out_has = false;
  const uint64_t body_begin = sym.st_value;
  const uint64_t body_end = sym.st_value + sym.body_size; // caller guarded the sum vs overflow.

  for (uint64_t i = 0; i < ehdr.e_shnum; ++i) {
    Elf64_Shdr shdr;
    if (!read_shdr(image, ehdr, i, &shdr)) {
      report(error_out, "malformed ELF: section header out of range");
      return false;
    }
    const bool is_rela = shdr.sh_type == SHT_RELA;
    const bool is_rel = shdr.sh_type == SHT_REL;
    if (!is_rela && !is_rel)
      continue;
    if (shdr.sh_info != 0 && shdr.sh_info != sym.section_index)
      continue;

    const uint64_t entsize = is_rela ? sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
    if (shdr.sh_entsize != 0 && shdr.sh_entsize != entsize) {
      report(error_out, "malformed ELF: unexpected relocation entry size");
      return false;
    }
    if (!detail::fits_in_bounds(shdr.sh_offset, shdr.sh_size, image.size())) {
      report(error_out, "malformed ELF: relocation table extends past end of image");
      return false;
    }
    const uint64_t count = shdr.sh_size / entsize;
    for (uint64_t e = 0; e < count; ++e) {
      uint64_t r_offset = 0;
      if (is_rela) {
        Elf64_Rela rela;
        if (!read_pod(image, shdr.sh_offset + e * entsize, &rela)) {
          report(error_out, "malformed ELF: relocation entry out of range");
          return false;
        }
        r_offset = rela.r_offset;
      } else {
        Elf64_Rel rel;
        if (!read_pod(image, shdr.sh_offset + e * entsize, &rel)) {
          report(error_out, "malformed ELF: relocation entry out of range");
          return false;
        }
        r_offset = rel.r_offset;
      }
      if (r_offset >= body_begin && r_offset < body_end) {
        *out_has = true;
        return true;
      }
    }
  }
  return true;
}

[[nodiscard]] bool is_call(const Instruction &inst) { return (inst.flags() & INDIRECT_CALL) != 0; }
// Quick check for explicit scratch access: FLAT scratch_* and SMEM s_scratch_*.
[[nodiscard]] bool is_scratch_access(std::string_view m) {
  return m.starts_with("scratch_") || m.starts_with("s_scratch_");
}

} // namespace

std::optional<ProbeCallable> build_probe_callable(const AmdGpuCodeObject &probe_obj,
                                                  const ResolvedProbeSymbol &sym,
                                                  rj_code_arch_t arch, std::string *error_out) {
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(probe_obj.image_data()),
                                       probe_obj.image_size());

  if (sym.body_size == 0 || sym.body_size % sizeof(uint32_t) != 0) {
    report(error_out, "probe body size is not a positive multiple of 4");
    return std::nullopt;
  }
  // Re-validate the body extent against the image even though the resolver
  // already did since it is taken by value
  if (!detail::fits_in_bounds(sym.body_file_offset, sym.body_size, image.size())) {
    report(error_out, "probe body extends past end of image");
    return std::nullopt;
  }
  // The relocation scan compares against [st_value, st_value + body_size). Guard
  // the sum so a hostile st_value cannot wrap it small and hide an in-body reloc.
  if (sym.st_value > UINT64_MAX - sym.body_size) {
    report(error_out, "probe symbol value would overflow the body range");
    return std::nullopt;
  }

  bool has_reloc = false;
  Elf64_Ehdr ehdr;
  if (!read_pod(image, 0, &ehdr)) {
    report(error_out, "ELF image too small for an ELF header");
    return std::nullopt;
  }
  // The resolver already validated the section header table extent to reach
  // this symbol; re-check defensively so the reloc scan's read_shdr is safe.
  if (ehdr.e_shentsize < sizeof(Elf64_Shdr) ||
      !detail::fits_in_bounds(ehdr.e_shoff, static_cast<uint64_t>(ehdr.e_shnum) * ehdr.e_shentsize,
                              image.size())) {
    report(error_out, "malformed ELF: section header table out of range");
    return std::nullopt;
  }
  if (!body_has_relocation(image, ehdr, sym, &has_reloc, error_out))
    return std::nullopt;
  if (has_reloc) {
    report(error_out, "probe body has a relocation and is not self-contained");
    return std::nullopt;
  }

  // Copy the body into an aligned word buffer. One extra zero word of slack so
  // the decoder succeeds in the event of a malformed input
  const size_t num_words = sym.body_size / sizeof(uint32_t);
  std::vector<uint32_t> words(num_words + 1, 0);
  std::memcpy(words.data(), image.data() + sym.body_file_offset, sym.body_size);

  auto decoder = Decoder::create(arch);
  if (decoder == nullptr) {
    report(error_out, "no decoder for probe architecture");
    return std::nullopt;
  }

  // Use strings to hold info between iterations (currently deleting inst)
  std::string_view last_mnemonic;
  std::string last_src0_name;
  bool last_has_src0 = false;
  size_t w = 0;
  while (w < num_words) {
    std::unique_ptr<Instruction> inst(decoder->decode(&words[w]));
    if (inst == nullptr) {
      report(error_out, "probe body failed to decode");
      return std::nullopt;
    }
    const int size = inst->size();
    if (size != 4 && size != 8) {
      report(error_out, "probe body has an unsupported instruction size");
      return std::nullopt;
    }
    const size_t inst_words = static_cast<size_t>(size) / sizeof(uint32_t);
    if (w + inst_words > num_words) {
      report(error_out, "probe body's last instruction is truncated");
      return std::nullopt;
    }
    const std::string_view m = inst->mnemonic();
    if (is_call(*inst)) {
      report(error_out, "probe body contains a call and is not self-contained");
      return std::nullopt;
    }
    if (is_scratch_access(m)) {
      report(error_out, "probe body accesses scratch / private segment");
      return std::nullopt;
    }
    last_mnemonic = m;
    const Operand *src0 = inst->src_operand(0);
    last_has_src0 = src0 != nullptr;
    last_src0_name = last_has_src0 ? src0->name() : std::string();
    w += inst_words;
  }

  // Must return through the s[30:31] link pair
  if (last_mnemonic != "s_setpc_b64" || !last_has_src0 || last_src0_name != "s[30:31]") {
    report(error_out, "probe body does not return via s_setpc_b64 s[30:31]");
    return std::nullopt;
  }

  ProbeCallable callable;
  callable.symbol = sym.name;
  callable.arch = arch;
  callable.body_words.assign(words.begin(), words.begin() + num_words);
  callable.cc = ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31;
  // output_text_offset stays 0 — assigned by the later layout step.
  return callable;
}

} // namespace rocjitsu
