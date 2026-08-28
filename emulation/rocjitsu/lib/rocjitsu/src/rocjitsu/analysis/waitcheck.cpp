// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/waitcheck.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"
#include "util/except.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <charconv>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

constexpr size_t kCounterCount = static_cast<size_t>(WaitCounterKind::Count);
using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;

static_assert(sizeof(KernelDescriptor) == 64, "AMDHSA kernel descriptor size changed");

[[nodiscard]] bool is_supported_waitcheck_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] uint32_t default_wavefront_size(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                 arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5
             ? 32
             : 64;
}

[[nodiscard]] bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] bool equals_ignore_ascii_case(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto to_lower = [](char c) {
      return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
    };
    if (to_lower(lhs[i]) != to_lower(rhs[i]))
      return false;
  }
  return true;
}

enum class WaitcntModel { LegacyNoVscnt, LegacyVscnt, SplitGfx12 };

[[nodiscard]] WaitcntModel waitcnt_model(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return WaitcntModel::LegacyNoVscnt;
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return WaitcntModel::LegacyVscnt;
  default:
    return WaitcntModel::SplitGfx12;
  }
}

[[nodiscard]] bool uses_legacy_waitcnt(rj_code_arch_t arch) {
  return waitcnt_model(arch) != WaitcntModel::SplitGfx12;
}

[[nodiscard]] bool has_legacy_vscnt(rj_code_arch_t arch) {
  return waitcnt_model(arch) == WaitcntModel::LegacyVscnt;
}

[[nodiscard]] bool supports_expert_scheduling(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] bool tracks_committed_vgpr_generations(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool tracks_gfx12_sgpr_hazards(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] WaitCounterKind smem_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Ds : WaitCounterKind::Km;
}

[[nodiscard]] WaitCounterKind vmem_store_wait_counter(rj_code_arch_t arch) {
  return waitcnt_model(arch) == WaitcntModel::LegacyNoVscnt ? WaitCounterKind::Load
                                                            : WaitCounterKind::Store;
}

[[nodiscard]] WaitCounterKind image_sample_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Load : WaitCounterKind::Sample;
}

[[nodiscard]] WaitCounterKind image_bvh_wait_counter(rj_code_arch_t arch) {
  return uses_legacy_waitcnt(arch) ? WaitCounterKind::Load : WaitCounterKind::Bvh;
}

struct LegacyWaitcnt {
  uint32_t vmcnt = 0;
  uint32_t expcnt = 0;
  uint32_t lgkmcnt = 0;
};

[[nodiscard]] LegacyWaitcnt decode_legacy_waitcnt(uint32_t value) {
  return {
      (value & 0xFu) | (((value >> 14u) & 0x3u) << 4u),
      (value >> 4u) & 0x7u,
      (value >> 8u) & 0xFu,
  };
}

[[nodiscard]] LegacyWaitcnt decode_gfx11_waitcnt(uint32_t value) {
  return {
      (value >> 10u) & 0x3Fu,
      value & 0x7u,
      (value >> 4u) & 0x3Fu,
  };
}

[[nodiscard]] LegacyWaitcnt decode_legacy_waitcnt(uint32_t value, rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5
             ? decode_gfx11_waitcnt(value)
             : decode_legacy_waitcnt(value);
}

[[nodiscard]] std::string wait_expression(WaitCounterKind counter, uint32_t required_count,
                                          rj_code_arch_t arch) {
  std::ostringstream os;
  if (uses_legacy_waitcnt(arch)) {
    switch (counter) {
    case WaitCounterKind::Load:
      os << "s_waitcnt vmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Ds:
      os << "s_waitcnt lgkmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Exp:
      os << "s_waitcnt expcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Store:
      if (has_legacy_vscnt(arch)) {
        os << "s_waitcnt_vscnt null, " << required_count;
        return os.str();
      }
      break;
    default:
      break;
    }
  }
  if (counter == WaitCounterKind::X) {
    os << "s_wait_xcnt " << required_count;
  } else if (counter == WaitCounterKind::VmVsrc) {
    os << "s_wait_alu depctr_vm_vsrc(" << required_count << ")";
  } else {
    os << "s_wait_" << wait_counter_name(counter) << " <= " << required_count;
  }
  return os.str();
}

[[nodiscard]] bool fits_in_image(uint64_t offset, uint64_t size, size_t image_size) {
  return offset <= image_size && size <= image_size - offset;
}

[[nodiscard]] bool is_kernel_descriptor_symbol(const Elf64_Sym &sym, const char *strtab,
                                               size_t strtab_size) {
  if (sym.st_size != sizeof(KernelDescriptor))
    return false;
  if (elf_symbol_type(sym.st_info) != kElfSymbolTypeObject ||
      elf_symbol_bind(sym.st_info) != kElfSymbolBindGlobal)
    return false;
  if (strtab == nullptr || strtab_size == 0 || sym.st_name >= strtab_size)
    return false;

  const char *name = strtab + sym.st_name;
  const size_t max_len = strtab_size - sym.st_name;
  const size_t len = strnlen(name, max_len);
  return len > 3 && std::string_view(name + len - 3, 3) == ".kd";
}

[[nodiscard]] std::map<uint64_t, uint64_t>
find_waitcheck_function_sizes(const CodeObject &code_object) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section &text = *code_object.text_sections().front();
  const auto *image = reinterpret_cast<const uint8_t *>(code_object.image_data());
  const size_t image_size = code_object.image_size();
  if (image == nullptr || image_size < sizeof(Elf64_Ehdr))
    return {};

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
  if (std::memcmp(ehdr->e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 || ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
      !fits_in_image(ehdr->e_shoff, static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr),
                     image_size)) {
    return {};
  }

  const auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(image + ehdr->e_shoff);
  const uint64_t text_vaddr = text.vaddr();
  const uint64_t text_size = text.size();
  std::map<uint64_t, uint64_t> function_sizes;
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr &symtab_shdr = shdrs[i];
    if ((symtab_shdr.sh_type != SHT_SYMTAB && symtab_shdr.sh_type != SHT_DYNSYM) ||
        symtab_shdr.sh_entsize != sizeof(Elf64_Sym) ||
        !fits_in_image(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_size)) {
      continue;
    }

    const auto *syms = reinterpret_cast<const Elf64_Sym *>(image + symtab_shdr.sh_offset);
    const size_t symbol_count = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    for (size_t sym_index = 0; sym_index < symbol_count; ++sym_index) {
      const Elf64_Sym &sym = syms[sym_index];
      if (elf_symbol_type(sym.st_info) != kElfSymbolTypeFunc || sym.st_size == 0 ||
          sym.st_shndx >= ehdr->e_shnum || (shdrs[sym.st_shndx].sh_flags & SHF_EXECINSTR) == 0 ||
          sym.st_value < text_vaddr || sym.st_value >= text_vaddr + text_size) {
        continue;
      }
      function_sizes.insert_or_assign(sym.st_value - text_vaddr, sym.st_size);
    }
  }
  return function_sizes;
}

[[nodiscard]] std::vector<WaitcheckKernelInfo>
find_waitcheck_kernels(const CodeObject &code_object) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section &text = *code_object.text_sections().front();
  const auto *image = reinterpret_cast<const uint8_t *>(code_object.image_data());
  const size_t image_size = code_object.image_size();
  if (image == nullptr || image_size < sizeof(Elf64_Ehdr))
    return {};

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
  if (std::memcmp(ehdr->e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 || ehdr->e_shentsize != sizeof(Elf64_Shdr))
    return {};
  if (!fits_in_image(ehdr->e_shoff, static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr),
                     image_size))
    return {};

  const auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(image + ehdr->e_shoff);
  const uint64_t text_vaddr = text.vaddr();
  const uint64_t text_size = text.size();
  if (text_size == 0)
    return {};

  std::vector<WaitcheckKernelInfo> kernels;
  const auto function_sizes = find_waitcheck_function_sizes(code_object);
  std::set<uint64_t> seen_descriptors;
  std::set<std::pair<uint64_t, uint32_t>> seen_entry_modes;
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr &symtab_shdr = shdrs[i];
    if (symtab_shdr.sh_type != SHT_SYMTAB && symtab_shdr.sh_type != SHT_DYNSYM)
      continue;
    if (symtab_shdr.sh_entsize != sizeof(Elf64_Sym))
      continue;
    if (!fits_in_image(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_size))
      continue;
    if (symtab_shdr.sh_link >= ehdr->e_shnum)
      continue;

    const Elf64_Shdr &strtab_shdr = shdrs[symtab_shdr.sh_link];
    if (!fits_in_image(strtab_shdr.sh_offset, strtab_shdr.sh_size, image_size))
      continue;

    const char *strtab = reinterpret_cast<const char *>(image + strtab_shdr.sh_offset);
    const auto *syms = reinterpret_cast<const Elf64_Sym *>(image + symtab_shdr.sh_offset);
    const size_t symbol_count = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    for (size_t sym_index = 0; sym_index < symbol_count; ++sym_index) {
      const Elf64_Sym &sym = syms[sym_index];
      if (!is_kernel_descriptor_symbol(sym, strtab, strtab_shdr.sh_size))
        continue;
      if (sym.st_shndx >= ehdr->e_shnum)
        continue;

      const Elf64_Shdr &descriptor_section = shdrs[sym.st_shndx];
      if (sym.st_value < descriptor_section.sh_addr)
        continue;

      const uint64_t descriptor_file_offset =
          descriptor_section.sh_offset + (sym.st_value - descriptor_section.sh_addr);
      if (!fits_in_image(descriptor_file_offset, sizeof(KernelDescriptor), image_size))
        continue;
      if (!seen_descriptors.insert(descriptor_file_offset).second)
        continue;

      KernelDescriptor descriptor{};
      std::memcpy(&descriptor, image + descriptor_file_offset, sizeof(descriptor));
      const int64_t entry_vaddr_signed =
          static_cast<int64_t>(sym.st_value) + descriptor.kernel_code_entry_byte_offset;
      if (entry_vaddr_signed < 0)
        continue;

      const uint64_t entry_vaddr = static_cast<uint64_t>(entry_vaddr_signed);
      if (entry_vaddr < text_vaddr || entry_vaddr >= text_vaddr + text_size)
        continue;
      const char *symbol_name = strtab + sym.st_name;
      const size_t symbol_name_size = strnlen(symbol_name, strtab_shdr.sh_size - sym.st_name);
      const uint32_t wavefront_size =
          (descriptor.kernel_code_properties &
           rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0
              ? 32
              : 64;
      const uint64_t entry_offset = entry_vaddr - text_vaddr;
      const auto function_it = function_sizes.find(entry_offset);
      const uint64_t code_size = function_it == function_sizes.end() ? 0 : function_it->second;
      // Link-time identical-code folding can leave many kernel descriptors
      // naming the same executable entry.  They are aliases, not independent
      // code to decode and analyze.  Preserve distinct wave modes because the
      // same bytes can have different implicit-register semantics in wave32
      // and wave64, but visit each entry/mode pair only once.
      if (!seen_entry_modes.emplace(entry_offset, wavefront_size).second)
        continue;
      kernels.push_back({std::string(symbol_name, symbol_name_size - 3), sym.st_value, entry_offset,
                         code_size, wavefront_size});
    }
  }

  std::ranges::sort(kernels, {}, &WaitcheckKernelInfo::entry_offset);
  return kernels;
}

[[nodiscard]] std::vector<uint64_t> find_waitcheck_function_entries(const CodeObject &code_object) {
  const auto function_sizes = find_waitcheck_function_sizes(code_object);
  std::vector<uint64_t> entries;
  entries.reserve(function_sizes.size());
  for (const auto &[entry, size] : function_sizes) {
    (void)size;
    entries.push_back(entry);
  }
  return entries;
}

enum class WaitEventKind {
  Unknown,
  VmemNoSamplerLoad,
  FlatLoad,
  VmemStore,
  FlatStore,
  Ds,
  Smem,
  Sample,
  Bvh,
  Export,
  SccWrite,
  SqMessage,
  GlobalInv,
  GlobalWb,
  LdsDirect,
  AsyncLdsLoad,
  AsyncLdsStore,
  AsyncBarrier,
  TensorLdsLoad,
  TensorLdsStore,
  Count,
};

enum class OrderedCounterDomain {
  None,
  Vmem,
  Lds,
};

inline constexpr size_t kWaitEventKindCount = static_cast<size_t>(WaitEventKind::Count);
inline constexpr uint8_t kNoPendingEventAge = std::numeric_limits<uint8_t>::max();

struct PendingEventAges {
  std::array<uint8_t, kWaitEventKindCount> values = [] {
    std::array<uint8_t, kWaitEventKindCount> result;
    result.fill(kNoPendingEventAge);
    return result;
  }();

  bool operator==(const PendingEventAges &) const = default;
};

enum class TrackedRegisterSource {
  None,
  Defs,
  Uses,
  VectorUses,
  StoreDataUses,
};

struct ClassifiedEvent {
  ClassifiedEvent(WaitCounterKind counter = WaitCounterKind::Load,
                  WaitEventKind kind = WaitEventKind::Unknown,
                  TrackedRegisterSource registers = TrackedRegisterSource::Defs,
                  bool check_uses = true, bool check_defs = true, bool check_exec_defs = false,
                  std::optional<RegisterRef> special_reg = std::nullopt,
                  std::optional<int64_t> barrier_id = std::nullopt, bool check_memory_order = false,
                  bool check_program_end = false, bool check_counter_parity_order = false)
      : counter(counter), kind(kind), registers(registers), check_uses(check_uses),
        check_defs(check_defs), check_exec_defs(check_exec_defs), special_reg(special_reg),
        barrier_id(barrier_id), check_memory_order(check_memory_order),
        check_program_end(check_program_end),
        check_counter_parity_order(check_counter_parity_order) {}

  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  TrackedRegisterSource registers = TrackedRegisterSource::Defs;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool check_memory_order = false;
  bool check_program_end = false;
  bool check_counter_parity_order = false;
};

inline constexpr uint8_t kVgprLow16Mask = 0x1;
inline constexpr uint8_t kVgprHigh16Mask = 0x2;
inline constexpr uint8_t kVgprFull32Mask = kVgprLow16Mask | kVgprHigh16Mask;

struct PartialRegisterAccess {
  RegisterRef reg;
  uint8_t mask = kVgprFull32Mask;
};

struct PendingEvent {
  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  OrderedCounterDomain ordered_domain = OrderedCounterDomain::None;
  RegisterSet regs;
  RegisterSet old_value_regs;
  // LLVM tracks the low/high 16-bit physical subregisters used by D16 memory
  // operations independently. Keep the exceptional partial destination
  // sparse: almost every event still covers whole 32-bit register lanes.
  std::optional<RegisterRef> partial_reg;
  uint8_t partial_reg_mask = kVgprFull32Mask;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool produces_regs = false;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  bool check_memory_order = false;
  bool check_program_end = false;
  bool check_counter_parity_order = false;
  uint32_t min_younger = 0;
  // Minimum number of later operations known to complete after this event on
  // the same ordered hardware queue.  Once this reaches the counter's maximum
  // outstanding value, issue backpressure proves that this event has retired.
  uint32_t min_ordered_younger = 0;

  bool operator==(const PendingEvent &) const = default;
};

struct LdsInterval {
  uint64_t begin = 0;
  uint64_t end = 0;

  [[nodiscard]] bool overlaps(const LdsInterval &other) const {
    return begin < other.end && other.begin < end;
  }

  bool operator==(const LdsInterval &) const = default;
};

struct DtlVisibilityEvent {
  std::optional<LdsInterval> interval;
  bool active = true;
  uint32_t min_younger = 0;
  uint32_t min_ordered_younger = 0;
  std::optional<uint32_t> barrier_required_count;
  uint64_t barrier_section_offset = 0;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;

  bool operator==(const DtlVisibilityEvent &) const = default;
};

struct LdsConstantState {
  std::optional<uint32_t> m0;
  std::map<uint16_t, uint32_t> uniform_vgprs;

  bool operator==(const LdsConstantState &) const = default;
};

struct SgprHazardProducer {
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;

  bool operator==(const SgprHazardProducer &) const = default;
};

constexpr uint8_t kSgprHazardSalu = 1u << 0u;
constexpr uint8_t kSgprHazardValu = 1u << 1u;

struct SgprHazardState {
  std::bitset<64> tracked_pairs;
  bool tracked_vcc = false;
  std::bitset<128> salu_hazards;
  std::bitset<128> valu_hazards;
  uint8_t vcc_hazard = 0;
  std::unordered_map<uint16_t, SgprHazardProducer> salu_producers;
  std::unordered_map<uint16_t, SgprHazardProducer> valu_producers;
  std::optional<SgprHazardProducer> salu_vcc_producer;
  std::optional<SgprHazardProducer> valu_vcc_producer;
  uint8_t consecutive_ds_nops = 0;

  bool operator==(const SgprHazardState &) const = default;
};

struct VaVdstHazard {
  uint8_t age = 0;
  bool trans_since = false;
  SgprHazardProducer producer;

  bool operator==(const VaVdstHazard &) const = default;
};

struct VaVdstHazardState {
  std::unordered_map<uint16_t, VaVdstHazard> hazards;

  bool operator==(const VaVdstHazardState &) const = default;
};

struct VgprMsbState {
  uint8_t mode = 0;
  bool known = true;

  [[nodiscard]] uint32_t for_role(amdgpu::VgprMsbRole role) const {
    switch (role) {
    case amdgpu::VgprMsbRole::Src0:
      return mode & 0x3u;
    case amdgpu::VgprMsbRole::Src1:
      return (mode >> 2u) & 0x3u;
    case amdgpu::VgprMsbRole::Src2:
      return (mode >> 4u) & 0x3u;
    case amdgpu::VgprMsbRole::Dst:
      return (mode >> 6u) & 0x3u;
    case amdgpu::VgprMsbRole::None:
      return 0;
    }
    return 0;
  }

  bool operator==(const VgprMsbState &) const = default;
};

enum class DelayAluEffect : uint8_t {
  None,
  Valu,
  Salu,
};

struct PendingDelayAlu {
  uint8_t countdown = 0;
  DelayAluEffect effect = DelayAluEffect::None;

  bool operator==(const PendingDelayAlu &) const = default;
};

struct ExpertSchedulingState {
  bool enabled = false;
  bool known = true;

  bool operator==(const ExpertSchedulingState &) const = default;
};

struct PendingState {
  // Vectors stay sorted by static event identity so CFG equality is stable.
  // min_younger, rather than vector position, represents hardware issue order.
  std::array<std::vector<PendingEvent>, kCounterCount> pending;
  // LLVM keeps the newest score for every hardware-event kind, including
  // counter-only operations with no register or ordering payload.  Their
  // presence determines whether a counter may retire out of order.  Store the
  // equivalent age rather than materializing one PendingEvent per token.
  std::array<PendingEventAges, kCounterCount> pending_event_ages;
  // Keep scalar-memory presence even for counter-only requests that do not
  // need a full PendingEvent. Scalar memory makes its counter out of order.
  std::array<bool, kCounterCount> pending_smem{};
  std::array<bool, kCounterCount> uncertain_order{};
  RegisterSet ready_regs;
  SgprHazardState sgpr_hazards;
  VaVdstHazardState va_vdst_hazards;
  VgprMsbState vgpr_msb;
  bool vgpr_msb_setreg_hazard = false;
  bool previous_vm_vsrc_zero_wait = false;
  std::optional<SgprHazardProducer> async_barrier_post_wait;
  std::vector<PendingDelayAlu> delay_alu;
  ExpertSchedulingState expert_scheduling;
  std::vector<DtlVisibilityEvent> dtl_visibility;
  LdsConstantState lds_constants;

  bool operator==(const PendingState &) const = default;
};

static_assert(sizeof(PendingState) < 4096,
              "waitcheck CFG states must keep architecture-specific hazard storage sparse");

struct CounterParityRequirement {
  PendingEvent event;
  RegisterRef reg{RegClass::VGPR, 0, 1};
  WaitcheckAccessKind access = WaitcheckAccessKind::Use;
  uint32_t required_count = 0;
  const Instruction *consumer = nullptr;
  uint64_t consumer_section_offset = 0;
  uint64_t consumer_file_offset = 0;
};

enum class DependencyView {
  // Model which committed VGPR generation an ISA instruction can observe.
  RuntimeVisibleGeneration,
  // Mirror LLVM's post-RA wait insertion: every pending physical-register
  // definition that intersects an operand contributes to the required wait.
  CompilerPendingDefinition,
};

struct PendingWaitGroup {
  PendingState state_before;
  std::array<std::optional<uint32_t>, kCounterCount> emitted_counts;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instructions;
  bool non_entry_function_prolog = false;
};

struct CfgInstructionView {
  uint64_t section_offset = 0;
  const Instruction *instruction = nullptr;
};

struct CfgBlockView {
  const BasicBlock *block = nullptr;
  std::vector<CfgInstructionView> instructions;
  std::vector<size_t> predecessors;
  std::vector<size_t> successors;
};

struct ScalarValueConstraint {
  std::optional<uint32_t> equal;
  std::set<uint32_t> not_equal;
  std::optional<uint32_t> unsigned_min;
  std::optional<uint32_t> unsigned_max;
};

using ScalarConstraints = std::map<uint16_t, ScalarValueConstraint>;

struct SccPredicate {
  enum class Kind : uint8_t { EqImm, GeUnsignedImm };

  Kind kind = Kind::EqImm;
  uint16_t sgpr = 0;
  uint32_t value = 0;
};

struct Analyzer {
  Analyzer(WaitcheckReport &report, WaitcheckOptions options)
      : report_(report), options_(options) {}

  void analyze_stream(std::span<const uint32_t> words, rj_code_arch_t arch,
                      std::string section_name, uint64_t file_offset_base) {
    current_kernel_ = nullptr;
    wavefront_size_ = default_wavefront_size(arch);
    dtl_straight_line_model_ = true;
    auto decoder = Decoder::create(arch);
    if (!decoder) {
      report_.supported = false;
      return;
    }

    PendingState state;
    state.expert_scheduling.enabled = supports_expert_scheduling(arch);
    RegisterSet local_ready_regs;
    std::optional<PendingWaitGroup> pending_wait_group;
    size_t word_index = 0;
    while (word_index < words.size()) {
      std::unique_ptr<Instruction> inst;
      try {
        inst.reset(
            decoder->decode_window(words.subspan(word_index), word_index * sizeof(uint32_t)));
      } catch (const util::Exception &ex) {
        set_analysis_error(section_name, word_index * sizeof(uint32_t), ex);
        return;
      }
      if (!inst || inst->size() <= 0) {
        break;
      }

      const size_t inst_words = static_cast<size_t>(inst->size()) / sizeof(uint32_t);
      if (inst_words == 0 || word_index + inst_words > words.size()) {
        break;
      }

      const auto section_offset = static_cast<uint64_t>(word_index * sizeof(uint32_t));
      const auto file_offset = file_offset_base + section_offset;
      update_counter_parity_group(pending_wait_group, state, *inst, section_name, section_offset,
                                  file_offset, arch);
      analyze_instruction(state, local_ready_regs, *inst, section_name, section_offset, file_offset,
                          arch, true);
      ++report_.instructions_analyzed;
      word_index += inst_words;
      if (should_stop_after_diagnostic())
        break;
    }
    finish_counter_parity_block(pending_wait_group, arch);
  }

  void analyze_cfg(std::vector<std::unique_ptr<BasicBlock>> &blocks,
                   const std::string &section_name, uint64_t file_offset_base, rj_code_arch_t arch,
                   uint32_t wavefront_size, std::span<const uint64_t> entry_offsets = {},
                   bool non_entry_function = false) {
    if (blocks.empty())
      return;
    wavefront_size_ = wavefront_size;
    dtl_straight_line_model_ = blocks.size() == 1 && blocks.front()->successors().empty() &&
                               blocks.front()->call_edges().empty();

    std::unordered_map<const BasicBlock *, size_t> block_index;
    block_index.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i)
      block_index.emplace(blocks[i].get(), i);

    // A shared helper may be called from many sites. Keep the active return
    // continuation in the analysis-node identity so an s_setpc_b64 return can
    // resume only the call that created that context. Without this expansion,
    // later calls can return into earlier, mutually exclusive continuations and
    // manufacture impossible pending-event paths.
    // Keep the callee entry as well as the return continuation. Generated
    // device helpers can recurse; re-entering an already active callee is
    // summarized at that call's continuation instead of enumerating an
    // exponential family of finite stacks up to the hard depth limit.
    // The optional third component identifies the first block executed after
    // one specific call returns. Keeping it in the node identity consumes the
    // link-register seed after that block, so a later loop backedge does not
    // manufacture another return from the call.
    using CallFrame = std::tuple<size_t, uint16_t, size_t, uint64_t>;
    using CallContinuationSeed = std::pair<uint16_t, uint64_t>;
    using AnalysisNodeKey =
        std::tuple<size_t, std::vector<CallFrame>, std::optional<CallContinuationSeed>>;
    constexpr size_t kMaxCallDepth = 32;
    // Real code objects can have tens of thousands of distinct depth-two
    // contexts when many wrappers share a small set of helpers. Keep a hard
    // bound, but do not reject that finite graph merely because the underlying
    // CFG is small.
    constexpr size_t kCallContextNodeAllowance = 65536;
    const size_t max_analysis_nodes =
        std::max(blocks.size() * 32, blocks.size() + kCallContextNodeAllowance);

    std::map<AnalysisNodeKey, size_t> analysis_node_index;
    std::vector<AnalysisNodeKey> analysis_node_keys;
    std::vector<BasicBlock *> analysis_blocks;
    std::vector<std::vector<size_t>> cfg_successors;
    std::vector<uint8_t> conservative_sgpr_entries;
    std::vector<size_t> expansion_worklist;

    auto get_or_add_node = [&](AnalysisNodeKey key) -> std::optional<size_t> {
      if (const auto it = analysis_node_index.find(key); it != analysis_node_index.end())
        return it->second;
      if (analysis_node_keys.size() >= max_analysis_nodes)
        return std::nullopt;
      const size_t index = analysis_node_keys.size();
      analysis_node_index.emplace(key, index);
      analysis_blocks.push_back(blocks[std::get<0>(key)].get());
      analysis_node_keys.push_back(std::move(key));
      cfg_successors.emplace_back();
      conservative_sgpr_entries.push_back(0);
      expansion_worklist.push_back(index);
      return index;
    };

    auto set_call_context_limit_error = [&] {
      size_t deepest_stack = 0;
      size_t setpc_nodes = 0;
      size_t matching_return_nodes = 0;
      for (const auto &key : analysis_node_keys)
        deepest_stack = std::max(deepest_stack, std::get<1>(key).size());
      for (size_t i = 0; i < analysis_node_keys.size(); ++i) {
        const Instruction *term = analysis_blocks[i]->terminator();
        if (term == nullptr ||
            (term->mnemonic() != "s_setpc_b64" && term->mnemonic() != "s_set_pc_i64"))
          continue;
        ++setpc_nodes;
        if (!std::get<1>(analysis_node_keys[i]).empty() && term->raw_encoding() != nullptr &&
            static_cast<uint16_t>(term->raw_encoding()[0] & 0xffu) ==
                std::get<1>(std::get<1>(analysis_node_keys[i]).back()))
          ++matching_return_nodes;
      }
      report_.analysis_error = "waitcheck call-context graph exceeded node limit: nodes=" +
                               std::to_string(analysis_node_keys.size()) +
                               " blocks=" + std::to_string(blocks.size()) +
                               " deepest_call_stack=" + std::to_string(deepest_stack) +
                               " setpc_nodes=" + std::to_string(setpc_nodes) +
                               " matching_return_nodes=" + std::to_string(matching_return_nodes);
    };

    std::vector<size_t> root_blocks;
    if (entry_offsets.empty()) {
      // Symbol-less section analysis has no distinguished entry point.
      root_blocks.resize(blocks.size());
      std::iota(root_blocks.begin(), root_blocks.end(), 0);
    } else {
      // Reachable kernel CFGs must start with an empty call stack only at the
      // real entry points. Seeding every helper as another empty-stack root
      // duplicates its graph and analyzes returns without their callers.
      for (uint64_t entry_offset : entry_offsets) {
        const auto it = std::ranges::find_if(
            blocks, [&](const auto &block) { return block->start_offset() == entry_offset; });
        if (it == blocks.end()) {
          report_.supported = false;
          report_.analysis_error = "waitcheck CFG entry block is missing";
          return;
        }
        root_blocks.push_back(static_cast<size_t>(std::distance(blocks.begin(), it)));
      }
    }
    for (size_t i : root_blocks) {
      const auto root = get_or_add_node({i, {}, std::nullopt});
      if (!root) {
        report_.supported = false;
        set_call_context_limit_error();
        return;
      }
      if (tracks_gfx12_sgpr_hazards(arch) && (entry_offsets.empty() || non_entry_function))
        conservative_sgpr_entries[*root] = 1;
    }

    for (size_t work_index = 0; work_index < expansion_worklist.size(); ++work_index) {
      const size_t node_index = expansion_worklist[work_index];
      const AnalysisNodeKey node_key = analysis_node_keys[node_index];
      const BasicBlock &block = *blocks[std::get<0>(node_key)];

      auto add_context_edge = [&](size_t target_block_index, std::vector<CallFrame> call_stack,
                                  bool conservative_sgpr_entry = false,
                                  std::optional<CallContinuationSeed> call_continuation =
                                      std::nullopt) -> bool {
        const auto target = get_or_add_node(
            {target_block_index, std::move(call_stack), std::move(call_continuation)});
        if (!target)
          return false;
        if (conservative_sgpr_entry && tracks_gfx12_sgpr_hazards(arch))
          conservative_sgpr_entries[*target] = 1;
        auto &successors = cfg_successors[node_index];
        if (std::ranges::find(successors, *target) == successors.end())
          successors.push_back(*target);
        return true;
      };

      const Instruction *term = block.terminator();
      if (!std::get<1>(node_key).empty() && term != nullptr &&
          term->size() == static_cast<int>(sizeof(uint32_t)) &&
          (term->mnemonic() == "s_setpc_b64" || term->mnemonic() == "s_set_pc_i64") &&
          term->raw_encoding() != nullptr &&
          static_cast<uint16_t>(term->raw_encoding()[0] & 0xffu) ==
              std::get<1>(std::get<1>(node_key).back())) {
        std::vector<CallFrame> caller_stack = std::get<1>(node_key);
        const size_t continuation_index = std::get<0>(caller_stack.back());
        const uint16_t return_sreg = std::get<1>(caller_stack.back());
        const uint64_t source_call_offset = std::get<3>(caller_stack.back());
        caller_stack.pop_back();
        if (!add_context_edge(continuation_index, std::move(caller_stack),
                              /*conservative_sgpr_entry=*/true,
                              CallContinuationSeed{return_sreg, source_call_offset})) {
          report_.supported = false;
          set_call_context_limit_error();
          return;
        }
        continue;
      }

      for (const BasicBlock *successor : block.successors()) {
        const auto successor_it = block_index.find(successor);
        if (successor_it == block_index.end())
          continue;
        const bool is_call_fallthrough =
            std::ranges::any_of(block.call_edges(), [&](const BasicBlock::CallEdge &call) {
              return call.continuation == successor;
            });
        std::optional<CallContinuationSeed> unresolved_call_continuation;
        if (!is_call_fallthrough && term != nullptr && (term->flags() & INDIRECT_CALL) != 0) {
          const Operand *return_operand = term->dst_operand(0);
          const auto return_ref =
              return_operand == nullptr ? std::nullopt : return_operand->to_register_ref();
          if (return_ref && return_ref->cls == RegClass::SGPR) {
            const uint64_t source_call_offset =
                block.end_offset() - static_cast<uint64_t>(term->size());
            unresolved_call_continuation =
                CallContinuationSeed{return_ref->index, source_call_offset};
          }
        }
        if (!is_call_fallthrough && !add_context_edge(successor_it->second, std::get<1>(node_key),
                                                      unresolved_call_continuation.has_value(),
                                                      unresolved_call_continuation)) {
          report_.supported = false;
          set_call_context_limit_error();
          return;
        }
      }

      for (const BasicBlock::CallEdge &call : block.call_edges()) {
        const auto callee_it = block_index.find(call.callee);
        const auto continuation_it = block_index.find(call.continuation);
        if (callee_it == block_index.end() || continuation_it == block_index.end())
          continue;
        if (std::get<1>(node_key).size() >= kMaxCallDepth) {
          report_.supported = false;
          report_.analysis_error = "waitcheck call depth exceeds supported limit";
          return;
        }
        const bool recursive_reentry =
            std::ranges::any_of(std::get<1>(node_key), [&](const CallFrame &frame) {
              return std::get<2>(frame) == callee_it->second;
            });
        if (recursive_reentry) {
          if (!add_context_edge(continuation_it->second, std::get<1>(node_key),
                                /*conservative_sgpr_entry=*/true,
                                CallContinuationSeed{call.return_sreg, call.source_call_offset})) {
            report_.supported = false;
            set_call_context_limit_error();
            return;
          }
          continue;
        }
        std::vector<CallFrame> callee_stack = std::get<1>(node_key);
        callee_stack.emplace_back(continuation_it->second, call.return_sreg, callee_it->second,
                                  call.source_call_offset);
        if (!add_context_edge(callee_it->second, std::move(callee_stack),
                              /*conservative_sgpr_entry=*/true)) {
          report_.supported = false;
          set_call_context_limit_error();
          return;
        }
      }
    }

    std::vector<std::vector<size_t>> cfg_predecessors(analysis_blocks.size());
    for (size_t from = 0; from < cfg_successors.size(); ++from) {
      for (size_t to : cfg_successors[from])
        cfg_predecessors[to].push_back(from);
    }

    std::vector<PendingState> in(analysis_blocks.size());
    std::vector<PendingState> out(analysis_blocks.size());
    std::vector<uint8_t> out_initialized(analysis_blocks.size());
    std::unordered_map<uint64_t, const Instruction *> instruction_by_offset;
    for (const auto &block : blocks) {
      for (const Instruction &inst : block->instructions())
        instruction_by_offset.emplace(inst.src_loc(), &inst);
    }

    // Revisit only successors whose merged input may have changed. Large
    // generated kernels otherwise spend most of their time rescanning stable
    // blocks on every fixed-point iteration.
    std::deque<size_t> dataflow_worklist;
    std::vector<uint8_t> queued(analysis_blocks.size(), 1);
    for (size_t i = 0; i < analysis_blocks.size(); ++i)
      dataflow_worklist.push_back(i);
    size_t node_visits = 0;
    size_t last_changed_node = 0;
    std::string last_changed_components;
    auto differing_components = [](const PendingState &lhs, const PendingState &rhs) {
      std::string result;
      auto append = [&](std::string_view component) {
        if (!result.empty())
          result += ',';
        result += component;
      };
      for (size_t counter = 0; counter < kCounterCount; ++counter) {
        if (lhs.pending[counter] != rhs.pending[counter])
          append(wait_counter_name(static_cast<WaitCounterKind>(counter)));
        if (lhs.pending_event_ages[counter] != rhs.pending_event_ages[counter])
          append("pending-event-kinds");
        if (lhs.pending_smem[counter] != rhs.pending_smem[counter])
          append("pending-smem");
        if (lhs.uncertain_order[counter] != rhs.uncertain_order[counter])
          append("uncertain-order");
      }
      if (lhs.ready_regs != rhs.ready_regs)
        append("ready-regs");
      if (lhs.sgpr_hazards != rhs.sgpr_hazards)
        append("sgpr-hazards");
      if (lhs.va_vdst_hazards != rhs.va_vdst_hazards)
        append("va-vdst");
      if (lhs.vgpr_msb != rhs.vgpr_msb)
        append("vgpr-msb");
      if (lhs.vgpr_msb_setreg_hazard != rhs.vgpr_msb_setreg_hazard)
        append("vgpr-msb-setreg-hazard");
      if (lhs.previous_vm_vsrc_zero_wait != rhs.previous_vm_vsrc_zero_wait)
        append("previous-vm-vsrc-zero-wait");
      if (lhs.async_barrier_post_wait != rhs.async_barrier_post_wait)
        append("async-barrier-post-wait");
      if (lhs.delay_alu != rhs.delay_alu)
        append("delay-alu");
      if (lhs.expert_scheduling != rhs.expert_scheduling)
        append("expert-scheduling");
      if (lhs.dtl_visibility != rhs.dtl_visibility)
        append("dtl-visibility");
      if (lhs.lds_constants != rhs.lds_constants)
        append("lds-constants");
      return result;
    };
    const size_t max_node_visits = analysis_blocks.size() * 64 + 1024;
    while (!dataflow_worklist.empty() && node_visits++ < max_node_visits) {
      const size_t i = dataflow_worklist.front();
      dataflow_worklist.pop_front();
      queued[i] = 0;

      PendingState merged = merge_predecessors(cfg_predecessors[i], out, out_initialized);
      if (conservative_sgpr_entries[i] != 0) {
        // Linked code objects do not retain whether LLVM's boundary-cull
        // option was enabled. Model its default disabled behavior: every
        // tracked pair and VCC can be live across an externally visible
        // function boundary or a call whose target cannot be recovered.
        merged.sgpr_hazards.tracked_pairs.set();
        merged.sgpr_hazards.tracked_vcc = true;
      }
      if (const auto &continuation = std::get<2>(analysis_node_keys[i])) {
        const auto [return_sreg, source_call_offset] = *continuation;
        const auto source = instruction_by_offset.find(source_call_offset);
        const SgprHazardProducer producer{
            .section_name = section_name,
            .section_offset = source_call_offset,
            .file_offset = file_offset_base + source_call_offset,
            .instruction = source == instruction_by_offset.end() ? std::string{}
                                                                 : source->second->disassemble(),
        };
        set_sgpr_hazard(merged.sgpr_hazards, RegisterRef{RegClass::SGPR, return_sreg, 1},
                        /*is_valu=*/false, producer);
        if (return_sreg < 127) {
          set_sgpr_hazard(merged.sgpr_hazards,
                          RegisterRef{RegClass::SGPR, static_cast<uint16_t>(return_sreg + 1), 1},
                          /*is_valu=*/false, producer);
        }
      }
      PendingState next_out =
          analyze_block(*analysis_blocks[i], merged, section_name, file_offset_base, arch, false);
      const bool input_changed = !(merged == in[i]);
      const bool output_changed = out_initialized[i] == 0 || !(next_out == out[i]);
      if (!input_changed && !output_changed)
        continue;

      last_changed_node = i;
      last_changed_components.clear();
      if (input_changed)
        last_changed_components = "in:" + differing_components(merged, in[i]);
      if (output_changed) {
        if (!last_changed_components.empty())
          last_changed_components += ';';
        last_changed_components += "out:" + differing_components(next_out, out[i]);
      }
      in[i] = std::move(merged);
      out[i] = std::move(next_out);
      out_initialized[i] = 1;

      if (output_changed) {
        for (size_t successor : cfg_successors[i]) {
          if (queued[successor] != 0)
            continue;
          queued[successor] = 1;
          dataflow_worklist.push_back(successor);
        }
      }
    }

    if (!dataflow_worklist.empty()) {
      report_.supported = false;
      std::ostringstream os;
      os << "waitcheck CFG dataflow did not converge at .text+0x" << std::hex
         << analysis_blocks[last_changed_node]->start_offset();
      if (!last_changed_components.empty())
        os << " (" << last_changed_components << ')';
      report_.analysis_error = os.str();
      return;
    }

    prepare_cfg_path_filter(analysis_blocks, cfg_predecessors, cfg_successors);
    for (size_t i = 0; i < analysis_blocks.size(); ++i) {
      current_cfg_view_index_ = i;
      current_in_callee_context_ = !std::get<1>(analysis_node_keys[i]).empty();
      current_call_return_sreg_ =
          current_in_callee_context_
              ? std::optional{std::get<1>(std::get<1>(analysis_node_keys[i]).back())}
              : std::nullopt;
      (void)analyze_block(*analysis_blocks[i], in[i], section_name, file_offset_base, arch, true);
      if (should_stop_after_diagnostic())
        break;
    }
    current_cfg_view_index_.reset();
    current_in_callee_context_ = false;
    current_call_return_sreg_.reset();
    clear_cfg_path_filter();
  }

  void set_kernel_context(const WaitcheckKernelInfo *kernel) { current_kernel_ = kernel; }

private:
  void record_incomplete_analysis(std::string reason) {
    report_.analysis_complete = false;
    ++report_.incomplete_observations;
    if (report_.incomplete_reason.empty())
      report_.incomplete_reason = std::move(reason);
  }

  [[nodiscard]] static size_t counter_index(WaitCounterKind counter) {
    return static_cast<size_t>(counter);
  }

  [[nodiscard]] static uint32_t maximum_dependency_wait(rj_code_arch_t arch,
                                                        WaitCounterKind counter) {
    // LLVM caps a dependency score at the largest non-sentinel wait value.
    // The all-ones encoding means "no wait", so the largest useful value is
    // one less than the hardware counter mask.
    switch (counter) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
      return 62;
    case WaitCounterKind::Ds:
      return uses_legacy_waitcnt(arch) && arch != ROCJITSU_CODE_ARCH_RDNA3 &&
                     arch != ROCJITSU_CODE_ARCH_RDNA3_5
                 ? 14
                 : 62;
    case WaitCounterKind::Km:
      return 30;
    case WaitCounterKind::Sample:
      return 62;
    case WaitCounterKind::Bvh:
    case WaitCounterKind::Exp:
    case WaitCounterKind::VmVsrc:
      return 6;
    case WaitCounterKind::X:
    case WaitCounterKind::Async:
    case WaitCounterKind::Tensor:
      return 62;
    case WaitCounterKind::VaVdst:
      return 14;
    case WaitCounterKind::Depctr:
    case WaitCounterKind::Count:
      return std::numeric_limits<uint32_t>::max();
    }
    return std::numeric_limits<uint32_t>::max();
  }

  [[nodiscard]] static OrderedCounterDomain ordered_counter_domain(rj_code_arch_t arch,
                                                                   const ClassifiedEvent &event,
                                                                   const Instruction &inst) {
    // CDNA prevents a wait counter from overflowing by stalling the issue of
    // the instruction that would overflow it. Keep completion order separate
    // from the counter itself: mixed event types need not complete in order.
    if (arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4)
      return OrderedCounterDomain::None;
    if (event.counter == WaitCounterKind::Ds && event.kind == WaitEventKind::Ds &&
        inst.mnemonic().starts_with("ds_")) {
      // CDNA3/CDNA4 encode the DS GDS modifier in bit 16.
      return (inst.raw_encoding()[0] & (1u << 16u)) == 0 ? OrderedCounterDomain::Lds
                                                         : OrderedCounterDomain::None;
    }
    if (event.counter == WaitCounterKind::Load) {
      switch (event.kind) {
      case WaitEventKind::VmemNoSamplerLoad:
      case WaitEventKind::VmemStore:
      case WaitEventKind::Sample:
      case WaitEventKind::Bvh:
      case WaitEventKind::GlobalInv:
      case WaitEventKind::GlobalWb:
      case WaitEventKind::LdsDirect:
        // CDNA has no separate VSCNT. Its non-FLAT VMEM events share one
        // in-order VMCNT stream.
        return OrderedCounterDomain::Vmem;
      default:
        break;
      }
    }
    return OrderedCounterDomain::None;
  }

  [[nodiscard]] static std::optional<uint32_t> ordered_counter_capacity(rj_code_arch_t arch,
                                                                        const PendingEvent &event) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4)
      return std::nullopt;
    switch (event.ordered_domain) {
    case OrderedCounterDomain::Vmem:
      return 63;
    case OrderedCounterDomain::Lds:
      return 15;
    case OrderedCounterDomain::None:
      return std::nullopt;
    }
    return std::nullopt;
  }

  [[nodiscard]] static bool same_ordered_capacity_domain(rj_code_arch_t arch,
                                                         const PendingEvent &older,
                                                         const PendingEvent &younger) {
    const auto older_capacity = ordered_counter_capacity(arch, older);
    return older_capacity && ordered_counter_capacity(arch, younger) == older_capacity &&
           older.ordered_domain == younger.ordered_domain;
  }

  static void apply_ordered_counter_backpressure(PendingState &state,
                                                 std::span<const ClassifiedEvent> current_events,
                                                 const Instruction &inst, rj_code_arch_t arch) {
    for (const ClassifiedEvent &classification : current_events) {
      PendingEvent current;
      current.counter = classification.counter;
      current.kind = classification.kind;
      current.ordered_domain = ordered_counter_domain(arch, classification, inst);

      if (arch == ROCJITSU_CODE_ARCH_CDNA4 &&
          current.ordered_domain == OrderedCounterDomain::Vmem) {
        const auto capacity = ordered_counter_capacity(arch, current);
        if (capacity) {
          for (DtlVisibilityEvent &visibility : state.dtl_visibility) {
            if (!visibility.active)
              continue;
            if (visibility.min_ordered_younger < *capacity)
              ++visibility.min_ordered_younger;
            if (visibility.min_ordered_younger >= *capacity)
              visibility.active = false;
          }
        }
      }

      auto &pending = state.pending[counter_index(classification.counter)];
      for (PendingEvent &older : pending) {
        const auto capacity = ordered_counter_capacity(arch, older);
        if (capacity && same_ordered_capacity_domain(arch, older, current) &&
            older.min_ordered_younger < *capacity) {
          ++older.min_ordered_younger;
        }
      }
      retire_events(state, pending, [&](const PendingEvent &older) {
        const auto capacity = ordered_counter_capacity(arch, older);
        return capacity && older.min_ordered_younger >= *capacity;
      });
    }
  }

  [[nodiscard]] static bool is_counter_token_only(const PendingEvent &event) {
    // Events without a dependency payload exist only to advance the age of
    // older events on the same hardware counter. Once those ages have been
    // advanced, retaining one static PendingEvent per token adds no
    // information and makes large store-heavy CFGs quadratic in memory.
    return event.regs.size() == 0 && event.old_value_regs.size() == 0 && !event.special_reg &&
           !event.barrier_id && !event.produces_regs && !event.check_uses && !event.check_defs &&
           !event.check_exec_defs && !event.check_memory_order && !event.check_program_end &&
           !event.check_counter_parity_order;
  }

  [[nodiscard]] static bool same_event_identity(const PendingEvent &lhs, const PendingEvent &rhs) {
    return lhs.counter == rhs.counter && lhs.kind == rhs.kind &&
           lhs.ordered_domain == rhs.ordered_domain && lhs.regs == rhs.regs &&
           lhs.partial_reg == rhs.partial_reg && lhs.partial_reg_mask == rhs.partial_reg_mask &&
           lhs.special_reg == rhs.special_reg && lhs.barrier_id == rhs.barrier_id &&
           lhs.produces_regs == rhs.produces_regs && lhs.check_uses == rhs.check_uses &&
           lhs.check_defs == rhs.check_defs && lhs.check_exec_defs == rhs.check_exec_defs &&
           lhs.section_name == rhs.section_name && lhs.section_offset == rhs.section_offset &&
           lhs.file_offset == rhs.file_offset && lhs.instruction == rhs.instruction &&
           lhs.check_memory_order == rhs.check_memory_order &&
           lhs.check_program_end == rhs.check_program_end &&
           lhs.check_counter_parity_order == rhs.check_counter_parity_order;
  }

  [[nodiscard]] static auto register_ref_key(const std::optional<RegisterRef> &ref) {
    return std::make_tuple(ref.has_value(), ref ? static_cast<uint8_t>(ref->cls) : uint8_t{0},
                           ref ? ref->index : uint16_t{0}, ref ? ref->width : uint8_t{0});
  }

  [[nodiscard]] static bool register_set_less(const RegisterSet &lhs, const RegisterSet &rhs) {
    if (lhs == rhs)
      return false;

    std::vector<RegisterRef> lhs_regs;
    std::vector<RegisterRef> rhs_regs;
    lhs_regs.reserve(lhs.size());
    rhs_regs.reserve(rhs.size());
    lhs.for_each([&](RegisterRef ref) { lhs_regs.push_back(ref); });
    rhs.for_each([&](RegisterRef ref) { rhs_regs.push_back(ref); });
    auto less = [](RegisterRef lhs_ref, RegisterRef rhs_ref) {
      return std::make_tuple(static_cast<uint8_t>(lhs_ref.cls), lhs_ref.index, lhs_ref.width) <
             std::make_tuple(static_cast<uint8_t>(rhs_ref.cls), rhs_ref.index, rhs_ref.width);
    };
    return std::lexicographical_compare(lhs_regs.begin(), lhs_regs.end(), rhs_regs.begin(),
                                        rhs_regs.end(), less);
  }

  [[nodiscard]] static bool event_identity_less(const PendingEvent &lhs, const PendingEvent &rhs) {
    const auto lhs_key =
        std::tie(lhs.section_name, lhs.section_offset, lhs.file_offset, lhs.instruction,
                 lhs.counter, lhs.kind, lhs.ordered_domain, lhs.barrier_id, lhs.produces_regs,
                 lhs.check_uses, lhs.check_defs, lhs.check_exec_defs, lhs.check_memory_order,
                 lhs.check_program_end, lhs.check_counter_parity_order);
    const auto rhs_key =
        std::tie(rhs.section_name, rhs.section_offset, rhs.file_offset, rhs.instruction,
                 rhs.counter, rhs.kind, rhs.ordered_domain, rhs.barrier_id, rhs.produces_regs,
                 rhs.check_uses, rhs.check_defs, rhs.check_exec_defs, rhs.check_memory_order,
                 rhs.check_program_end, rhs.check_counter_parity_order);
    if (lhs_key != rhs_key)
      return lhs_key < rhs_key;
    if (register_ref_key(lhs.special_reg) != register_ref_key(rhs.special_reg))
      return register_ref_key(lhs.special_reg) < register_ref_key(rhs.special_reg);
    if (register_ref_key(lhs.partial_reg) != register_ref_key(rhs.partial_reg))
      return register_ref_key(lhs.partial_reg) < register_ref_key(rhs.partial_reg);
    if (lhs.partial_reg_mask != rhs.partial_reg_mask)
      return lhs.partial_reg_mask < rhs.partial_reg_mask;
    return register_set_less(lhs.regs, rhs.regs);
  }

  [[nodiscard]] static auto find_event(std::vector<PendingEvent> &events,
                                       const PendingEvent &event) {
    auto position = std::ranges::lower_bound(events, event, event_identity_less);
    if (position != events.end() && same_event_identity(*position, event))
      return position;
    return events.end();
  }

  [[nodiscard]] static bool contains_event(const std::vector<PendingEvent> &events,
                                           const PendingEvent &event) {
    return std::ranges::any_of(
        events, [&](const PendingEvent &existing) { return same_event_identity(existing, event); });
  }

  [[nodiscard]] bool diagnostics_available() const { return !report_.diagnostics_truncated; }

  [[nodiscard]] bool should_stop_after_diagnostic() const { return report_.stopped_early; }

  void prepare_cfg_path_filter(const std::vector<BasicBlock *> &blocks,
                               const std::vector<std::vector<size_t>> &predecessors,
                               const std::vector<std::vector<size_t>> &successors) {
    cfg_views_.clear();
    feasible_path_cache_.clear();
    reverse_reachability_cache_.clear();
    forward_reachability_cache_.clear();
    reachability_cache_bytes_ = 0;
    dominator_preorder_.clear();
    dominator_subtree_end_.clear();

    cfg_views_.reserve(blocks.size());
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      BasicBlock *block = blocks[block_index];
      CfgBlockView view;
      view.block = block;
      view.predecessors = predecessors[block_index];
      view.successors = successors[block_index];
      uint64_t section_offset = block->start_offset();
      for (const Instruction &inst : block->instructions()) {
        view.instructions.push_back({section_offset, &inst});
        section_offset += static_cast<uint64_t>(inst.size());
      }
      cfg_views_.push_back(std::move(view));
    }

    // A pending event whose block dominates the current wait is structurally
    // guaranteed to have a path to that wait. Cache a compact dominator tree
    // so the path-sensitive scalar-predicate search remains reserved for
    // branch-correlated events. Large generated kernels otherwise repeat the
    // same whole-CFG predicate scan for thousands of ordinary wait fields.
    const size_t node_count = cfg_views_.size();
    const size_t virtual_root = node_count;
    std::vector<std::vector<size_t>> augmented_successors(node_count + 1);
    std::vector<uint8_t> virtual_root_child(node_count);
    for (size_t node = 0; node < node_count; ++node) {
      augmented_successors[node] = cfg_views_[node].successors;
      if (cfg_views_[node].predecessors.empty()) {
        augmented_successors[virtual_root].push_back(node);
        virtual_root_child[node] = 1;
      }
    }

    std::vector<uint8_t> visited(node_count + 1);
    std::vector<size_t> postorder;
    auto append_postorder = [&](size_t root) {
      std::vector<std::pair<size_t, size_t>> stack{{root, 0}};
      visited[root] = 1;
      while (!stack.empty()) {
        auto &[node, successor_index] = stack.back();
        if (successor_index < augmented_successors[node].size()) {
          const size_t successor = augmented_successors[node][successor_index++];
          if (!visited[successor]) {
            visited[successor] = 1;
            stack.emplace_back(successor, 0);
          }
          continue;
        }
        postorder.push_back(node);
        stack.pop_back();
      }
    };
    append_postorder(virtual_root);
    // Defensive support for malformed/disconnected executable CFG islands.
    // Treat each island as another child of the virtual root.
    bool added_island_root = false;
    for (size_t node = 0; node < node_count; ++node) {
      if (!visited[node]) {
        augmented_successors[virtual_root].push_back(node);
        virtual_root_child[node] = 1;
        added_island_root = true;
      }
    }
    if (added_island_root) {
      std::ranges::fill(visited, 0);
      postorder.clear();
      append_postorder(virtual_root);
    }

    std::vector<size_t> reverse_postorder(postorder.rbegin(), postorder.rend());
    std::vector<size_t> rpo_index(node_count + 1, node_count + 1);
    for (size_t index = 0; index < reverse_postorder.size(); ++index)
      rpo_index[reverse_postorder[index]] = index;
    std::vector<size_t> immediate_dominator(node_count + 1, node_count + 1);
    immediate_dominator[virtual_root] = virtual_root;
    auto intersect = [&](size_t lhs, size_t rhs) {
      while (lhs != rhs) {
        while (rpo_index[lhs] > rpo_index[rhs])
          lhs = immediate_dominator[lhs];
        while (rpo_index[rhs] > rpo_index[lhs])
          rhs = immediate_dominator[rhs];
      }
      return lhs;
    };
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t node : reverse_postorder) {
        if (node == virtual_root)
          continue;
        size_t new_dominator = node_count + 1;
        auto merge_predecessor = [&](size_t predecessor) {
          if (immediate_dominator[predecessor] == node_count + 1)
            return;
          new_dominator =
              new_dominator == node_count + 1 ? predecessor : intersect(new_dominator, predecessor);
        };
        if (virtual_root_child[node])
          merge_predecessor(virtual_root);
        for (size_t predecessor : cfg_views_[node].predecessors)
          merge_predecessor(predecessor);
        if (new_dominator != node_count + 1 && immediate_dominator[node] != new_dominator) {
          immediate_dominator[node] = new_dominator;
          changed = true;
        }
      }
    }

    std::vector<std::vector<size_t>> dominator_children(node_count + 1);
    for (size_t node = 0; node < node_count; ++node) {
      const size_t parent = immediate_dominator[node];
      if (parent <= node_count)
        dominator_children[parent].push_back(node);
    }
    dominator_preorder_.resize(node_count);
    dominator_subtree_end_.resize(node_count);
    size_t next_preorder = 0;
    std::vector<std::tuple<size_t, size_t, bool>> stack{{virtual_root, 0, false}};
    while (!stack.empty()) {
      auto &[node, child_index, entered] = stack.back();
      if (!entered) {
        entered = true;
        if (node != virtual_root)
          dominator_preorder_[node] = next_preorder++;
      }
      if (child_index < dominator_children[node].size()) {
        stack.emplace_back(dominator_children[node][child_index++], 0, false);
        continue;
      }
      if (node != virtual_root)
        dominator_subtree_end_[node] = next_preorder;
      stack.pop_back();
    }
  }

  void clear_cfg_path_filter() {
    cfg_views_.clear();
    feasible_path_cache_.clear();
    reverse_reachability_cache_.clear();
    forward_reachability_cache_.clear();
    reachability_cache_bytes_ = 0;
    dominator_preorder_.clear();
    dominator_subtree_end_.clear();
  }

  [[nodiscard]] bool cfg_block_dominates(size_t dominator, size_t node) const {
    if (dominator >= dominator_preorder_.size() || node >= dominator_preorder_.size())
      return false;
    return dominator_preorder_[dominator] <= dominator_preorder_[node] &&
           dominator_preorder_[node] < dominator_subtree_end_[dominator];
  }

  void record_diagnostic(WaitcheckDiagnostic diag) {
    if (current_kernel_) {
      diag.has_kernel = true;
      diag.kernel_name = current_kernel_->name;
      diag.kernel_entry_offset = current_kernel_->entry_offset;
    }
    const auto key =
        std::make_tuple(diag.counter, diag.access, diag.reg.cls, diag.reg.index, diag.reg.width,
                        diag.section_name, diag.section_offset, diag.file_offset,
                        diag.producer_section_offset, diag.producer_file_offset,
                        diag.required_count, diag.instruction, diag.producer_instruction);
    if (!diagnostic_keys_.insert(key).second)
      return;

    ++report_.diagnostics_observed;
    if (options_.stop_after_first_diagnostic) {
      report_.stopped_early = true;
      report_.diagnostics_truncated = true;
    }
    if (report_.diagnostics.size() < options_.max_diagnostics) {
      report_.diagnostics.push_back(std::move(diag));
    } else {
      report_.diagnostics_truncated = true;
      return;
    }

    if (options_.max_diagnostics != std::numeric_limits<size_t>::max() &&
        report_.diagnostics.size() >= options_.max_diagnostics) {
      report_.diagnostics_truncated = true;
    }
  }

  [[nodiscard]] static std::optional<uint32_t> counter_no_wait_value(rj_code_arch_t arch,
                                                                     WaitCounterKind counter) {
    if (uses_legacy_waitcnt(arch)) {
      switch (counter) {
      case WaitCounterKind::Load:
        return 0x3fu;
      case WaitCounterKind::Store:
        return has_legacy_vscnt(arch) ? std::optional<uint32_t>{0x3fu} : std::nullopt;
      case WaitCounterKind::Ds:
        return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ? 0x3fu
                                                                                      : 0x0fu;
      case WaitCounterKind::Exp:
        return 0x07u;
      default:
        return std::nullopt;
      }
    }

    switch (counter) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
    case WaitCounterKind::Ds:
    case WaitCounterKind::Sample:
    case WaitCounterKind::X:
    case WaitCounterKind::Async:
    case WaitCounterKind::Tensor:
      return 0x3fu;
    case WaitCounterKind::Km:
      return 0x1fu;
    case WaitCounterKind::Bvh:
    case WaitCounterKind::Exp:
    case WaitCounterKind::VmVsrc:
      return 0x07u;
    case WaitCounterKind::VaVdst:
      return 0x0fu;
    default:
      return std::nullopt;
    }
  }

  [[nodiscard]] static std::optional<std::array<std::optional<uint32_t>, kCounterCount>>
  explicit_wait_fields(const Instruction &inst, rj_code_arch_t arch) {
    std::array<std::optional<uint32_t>, kCounterCount> fields;
    auto operand_value = [&](int index) -> std::optional<uint32_t> {
      const Operand *op = inst.src_operand(index);
      if (!op)
        return std::nullopt;
      return static_cast<uint32_t>(op->encoding_value());
    };
    auto set_field = [&](WaitCounterKind counter, uint32_t value) {
      const auto no_wait = counter_no_wait_value(arch, counter);
      if (no_wait && value < *no_wait)
        fields[counter_index(counter)] = value;
    };

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic == "s_wait_idle") {
      for (size_t counter_idx = 0; counter_idx < kCounterCount; ++counter_idx) {
        const auto counter = static_cast<WaitCounterKind>(counter_idx);
        if (counter_no_wait_value(arch, counter))
          set_field(counter, 0);
      }
      return fields;
    }
    if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt") {
      const auto value = operand_value(0);
      if (!value)
        return std::nullopt;
      const LegacyWaitcnt wait = decode_legacy_waitcnt(*value, arch);
      set_field(WaitCounterKind::Load, wait.vmcnt);
      set_field(WaitCounterKind::Exp, wait.expcnt);
      set_field(WaitCounterKind::Ds, wait.lgkmcnt);
      return fields;
    }

    if (uses_legacy_waitcnt(arch)) {
      const bool sopk_wait = mnemonic == "s_waitcnt_vmcnt" || mnemonic == "s_waitcnt_vscnt" ||
                             mnemonic == "s_waitcnt_expcnt" || mnemonic == "s_waitcnt_lgkmcnt";
      if (!sopk_wait)
        return std::nullopt;
      const auto value = operand_value(1);
      if (!value)
        return std::nullopt;
      if (mnemonic == "s_waitcnt_vmcnt")
        set_field(WaitCounterKind::Load, *value);
      else if (mnemonic == "s_waitcnt_vscnt")
        set_field(WaitCounterKind::Store, *value);
      else if (mnemonic == "s_waitcnt_expcnt")
        set_field(WaitCounterKind::Exp, *value);
      else
        set_field(WaitCounterKind::Ds, *value);
      return fields;
    }

    const auto value = operand_value(0);
    if (!value)
      return std::nullopt;
    if (mnemonic == "s_wait_loadcnt")
      set_field(WaitCounterKind::Load, *value);
    else if (mnemonic == "s_wait_storecnt")
      set_field(WaitCounterKind::Store, *value);
    else if (mnemonic == "s_wait_dscnt")
      set_field(WaitCounterKind::Ds, *value);
    else if (mnemonic == "s_wait_kmcnt")
      set_field(WaitCounterKind::Km, *value);
    else if (mnemonic == "s_wait_samplecnt")
      set_field(WaitCounterKind::Sample, *value);
    else if (mnemonic == "s_wait_bvhcnt")
      set_field(WaitCounterKind::Bvh, *value);
    else if (mnemonic == "s_wait_expcnt")
      set_field(WaitCounterKind::Exp, *value);
    else if (mnemonic == "s_wait_xcnt")
      set_field(WaitCounterKind::X, *value);
    else if (mnemonic == "s_wait_asynccnt")
      set_field(WaitCounterKind::Async, *value);
    else if (mnemonic == "s_wait_tensorcnt")
      set_field(WaitCounterKind::Tensor, *value);
    else if (mnemonic == "s_wait_loadcnt_dscnt") {
      set_field(WaitCounterKind::Load, (*value >> 8u) & 0x3fu);
      set_field(WaitCounterKind::Ds, *value & 0x3fu);
    } else if (mnemonic == "s_wait_storecnt_dscnt") {
      set_field(WaitCounterKind::Store, (*value >> 8u) & 0x3fu);
      set_field(WaitCounterKind::Ds, *value & 0x3fu);
    } else if (mnemonic == "s_wait_alu") {
      constexpr uint32_t kDepctrVmVsrcShift = 2;
      constexpr uint32_t kDepctrVmVsrcWidth = 3;
      set_field(WaitCounterKind::VmVsrc,
                depctr_field(*value, kDepctrVmVsrcShift, kDepctrVmVsrcWidth));
    } else {
      return std::nullopt;
    }
    return fields;
  }

  [[nodiscard]] static std::optional<std::array<std::optional<uint32_t>, kCounterCount>>
  embedded_wait_fields(const Instruction &inst, rj_code_arch_t arch) {
    if (uses_legacy_waitcnt(arch)) {
      if (const auto wait_exp = vinterp_wait_exp(inst); wait_exp && *wait_exp < 0x7u) {
        std::array<std::optional<uint32_t>, kCounterCount> fields;
        fields[counter_index(WaitCounterKind::Exp)] = *wait_exp;
        return fields;
      }
      return std::nullopt;
    }

    std::array<std::optional<uint32_t>, kCounterCount> fields;
    bool has_wait = false;
    if (const auto wait_exp = vinterp_wait_exp(inst); wait_exp && *wait_exp < 0x7u) {
      fields[counter_index(WaitCounterKind::Exp)] = *wait_exp;
      has_wait = true;
    }
    if (const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst); wait_vm_vsrc && *wait_vm_vsrc == 0) {
      fields[counter_index(WaitCounterKind::VmVsrc)] = 0;
      has_wait = true;
    }
    if (const auto wait_va_vdst = dsdir_wait_va_vdst(inst); wait_va_vdst && *wait_va_vdst < 0xfu) {
      fields[counter_index(WaitCounterKind::VaVdst)] = *wait_va_vdst;
      has_wait = true;
    }
    return has_wait ? std::optional{fields} : std::nullopt;
  }

  [[nodiscard]] static bool instruction_emits_counter_wait(const Instruction &inst,
                                                           WaitCounterKind counter,
                                                           rj_code_arch_t arch) {
    const size_t index = counter_index(counter);
    if (const auto fields = explicit_wait_fields(inst, arch); fields && (*fields)[index])
      return true;
    if (const auto fields = embedded_wait_fields(inst, arch); fields && (*fields)[index])
      return true;
    return false;
  }

  static void apply_counter_parity_wait(PendingState &state, WaitCounterKind counter,
                                        uint32_t count, rj_code_arch_t arch) {
    switch (counter) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
    case WaitCounterKind::Ds:
    case WaitCounterKind::Sample:
    case WaitCounterKind::Bvh:
      apply_memory_wait(state, counter, count, arch);
      break;
    case WaitCounterKind::Km:
      apply_kmcnt_wait(state, count, arch);
      break;
    case WaitCounterKind::X:
      apply_xcnt_wait(state, count);
      break;
    case WaitCounterKind::Exp:
    case WaitCounterKind::Async:
    case WaitCounterKind::Tensor:
    case WaitCounterKind::VmVsrc:
      apply_counter_wait(state, counter, count, arch);
      break;
    case WaitCounterKind::VaVdst:
    case WaitCounterKind::Depctr:
    case WaitCounterKind::Count:
      break;
    }
  }

  [[nodiscard]] static bool has_counter_parity_candidate(const PendingState &state,
                                                         WaitCounterKind counter) {
    if (counter == WaitCounterKind::VaVdst)
      return !state.va_vdst_hazards.hazards.empty();
    if (counter == WaitCounterKind::Depctr || counter == WaitCounterKind::Count)
      return false;
    if (!state.pending[counter_index(counter)].empty())
      return true;

    // Split memory-counter waits may also retire an older VM_VSRC event, and
    // LOADCNT/KMCNT may imply an X_CNT wait. A future dependency can justify
    // the current wait only when one of those events already existed before
    // the wait. Events issued after it cannot retroactively make it useful.
    switch (counter) {
    case WaitCounterKind::Load:
      return !state.pending[counter_index(WaitCounterKind::VmVsrc)].empty() ||
             !state.pending[counter_index(WaitCounterKind::X)].empty();
    case WaitCounterKind::Store:
    case WaitCounterKind::Ds:
    case WaitCounterKind::Sample:
    case WaitCounterKind::Bvh:
      return !state.pending[counter_index(WaitCounterKind::VmVsrc)].empty();
    case WaitCounterKind::Km:
      return !state.pending[counter_index(WaitCounterKind::X)].empty();
    case WaitCounterKind::Exp:
    case WaitCounterKind::X:
    case WaitCounterKind::Async:
    case WaitCounterKind::Tensor:
    case WaitCounterKind::VmVsrc:
    case WaitCounterKind::VaVdst:
    case WaitCounterKind::Depctr:
    case WaitCounterKind::Count:
      return false;
    }
    return false;
  }

  [[nodiscard]] static bool counter_may_retire_event(WaitCounterKind emitted_counter,
                                                     const PendingEvent &event) {
    if (event.counter == emitted_counter)
      return true;
    if (event.counter == WaitCounterKind::VmVsrc) {
      return emitted_counter == WaitCounterKind::Load ||
             emitted_counter == WaitCounterKind::Store || emitted_counter == WaitCounterKind::Ds ||
             emitted_counter == WaitCounterKind::Sample || emitted_counter == WaitCounterKind::Bvh;
    }
    return event.counter == WaitCounterKind::X &&
           (emitted_counter == WaitCounterKind::Load || emitted_counter == WaitCounterKind::Km);
  }

  [[nodiscard]] static std::optional<uint32_t>
  implied_counter_requirement(const PendingState &state, const PendingEvent &event,
                              WaitCounterKind emitted_counter, rj_code_arch_t arch) {
    const auto no_wait = counter_no_wait_value(arch, emitted_counter);
    if (!no_wait || event.counter == emitted_counter ||
        !counter_may_retire_event(emitted_counter, event))
      return std::nullopt;

    auto is_sufficient = [&](uint32_t count) {
      PendingState after = state;
      apply_counter_parity_wait(after, emitted_counter, count, arch);
      const auto &pending = after.pending[counter_index(event.counter)];
      return !std::ranges::any_of(pending, [&](const PendingEvent &candidate) {
        return same_event_identity(candidate, event);
      });
    };
    if (!is_sufficient(0))
      return std::nullopt;

    // Sufficiency is monotonic: a lower counter is always at least as strong.
    // Find the weakest sufficient value without copying a potentially large
    // pending state once for every representable counter value.
    uint32_t lower = 0;
    uint32_t upper = *no_wait;
    while (lower + 1 < upper) {
      const uint32_t middle = lower + (upper - lower) / 2;
      if (is_sufficient(middle))
        lower = middle;
      else
        upper = middle;
    }
    return lower;
  }

  void record_counter_underaccounting(WaitcheckCounterUnderaccountingDiagnostic diag) {
    if (current_kernel_) {
      diag.has_kernel = true;
      diag.kernel_name = current_kernel_->name;
      diag.kernel_entry_offset = current_kernel_->entry_offset;
    }
    const auto key =
        std::make_tuple(diag.kernel_name, diag.counter, diag.wait_section_offset,
                        diag.consumer_section_offset, diag.producer_section_offset,
                        diag.emitted_count, diag.required_count, diag.has_required_dependency);
    if (!counter_underaccounting_keys_.insert(key).second)
      return;

    ++report_.counter_underaccounting_observed;
    if (!diag.has_required_dependency)
      ++report_.counter_unmodeled_wait_observed;
    if (report_.counter_underaccounting_diagnostics.size() <
        options_.max_counter_parity_diagnostics) {
      report_.counter_underaccounting_diagnostics.push_back(std::move(diag));
    } else {
      report_.counter_parity_diagnostics_truncated = true;
    }
  }

  void collect_counter_parity_requirements(
      const PendingWaitGroup &group, const Instruction &consumer, uint64_t consumer_section_offset,
      uint64_t consumer_file_offset, rj_code_arch_t arch,
      std::array<std::optional<CounterParityRequirement>, kCounterCount> &requirements) {
    const PendingState *requirement_state = &group.state_before;
    const InstDefUse du =
        inst_def_use_for_waitcheck(consumer, requirement_state->vgpr_msb, arch, wavefront_size_);
    auto current_events = classify_events(consumer, arch);
    if (!expert_waits_enabled(*requirement_state)) {
      std::erase_if(current_events, [](const ClassifiedEvent &event) {
        return event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst;
      });
    }
    std::optional<PendingState> xcnt_adjusted_state;
    if (std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
          return event.counter == WaitCounterKind::X;
        })) {
      xcnt_adjusted_state.emplace(group.state_before);
      apply_implicit_xcnt_ordering(*xcnt_adjusted_state, du, current_events);
      requirement_state = &*xcnt_adjusted_state;
    }
    visit_dependencies(
        *requirement_state, consumer, du, current_events, arch,
        DependencyView::CompilerPendingDefinition,
        [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access,
            uint32_t required_count) {
          bool can_improve =
              !requirements[counter_index(event.counter)] ||
              required_count < requirements[counter_index(event.counter)]->required_count;
          for (size_t counter_idx = 0; !can_improve && counter_idx < group.emitted_counts.size();
               ++counter_idx) {
            if (!group.emitted_counts[counter_idx])
              continue;
            const auto emitted_counter = static_cast<WaitCounterKind>(counter_idx);
            const auto &current = requirements[counter_idx];
            can_improve = counter_may_retire_event(emitted_counter, event) &&
                          (!current || current->required_count != 0);
          }
          if (!can_improve)
            return;
          if (!event_can_reach_wait(event, group.section_offset, arch))
            return;
          auto record_requirement = [&](WaitCounterKind counter, uint32_t count) {
            auto &strongest = requirements[counter_index(counter)];
            if (!strongest || count < strongest->required_count) {
              strongest = CounterParityRequirement{event,
                                                   reg,
                                                   access,
                                                   count,
                                                   &consumer,
                                                   consumer_section_offset,
                                                   consumer_file_offset};
            }
          };
          record_requirement(event.counter, required_count);

          // A split GFX12 memory-counter wait may also retire the
          // corresponding VM_VSRC or X_CNT event. Attribute that
          // dependency to the emitted field so a compiler using
          // the implied form compares against its effective
          // semantic strength rather than looking redundant.
          for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
            if (!group.emitted_counts[counter_idx])
              continue;
            const auto emitted_counter = static_cast<WaitCounterKind>(counter_idx);
            if (const auto implied =
                    implied_counter_requirement(*requirement_state, event, emitted_counter, arch)) {
              record_requirement(emitted_counter, *implied);
            }
          }
        });

    if (expert_waits_enabled(*requirement_state) && dsdir_wait_va_vdst(consumer)) {
      auto &strongest = requirements[counter_index(WaitCounterKind::VaVdst)];
      du.defs.for_each([&](RegisterRef ref) {
        if (ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
          return;
        const auto hazard_it = requirement_state->va_vdst_hazards.hazards.find(ref.index);
        if (hazard_it == requirement_state->va_vdst_hazards.hazards.end())
          return;
        const VaVdstHazard &hazard = hazard_it->second;
        const uint32_t required_count = hazard.trans_since ? 0u : hazard.age;
        if (strongest && strongest->required_count <= required_count)
          return;

        PendingEvent producer;
        producer.counter = WaitCounterKind::VaVdst;
        producer.section_name = hazard.producer.section_name;
        producer.section_offset = hazard.producer.section_offset;
        producer.file_offset = hazard.producer.file_offset;
        producer.instruction = hazard.producer.instruction;
        strongest =
            CounterParityRequirement{std::move(producer), ref,       WaitcheckAccessKind::Def,
                                     required_count,      &consumer, consumer_section_offset,
                                     consumer_file_offset};
      });
    }

    if (uses_legacy_waitcnt(arch) && consumer.mnemonic() == "s_barrier") {
      // LLVM synchronization lowering may attach a local-memory ordering
      // requirement to a barrier even when the final instruction has no
      // explicit register operands. Attribute a legacy LGKM drain to the
      // strongest outstanding LGKM event instead of losing that final-ISA
      // intent as an unmodeled wait.
      auto &strongest = requirements[counter_index(WaitCounterKind::Ds)];
      for (const PendingEvent &event :
           requirement_state->pending[counter_index(WaitCounterKind::Ds)]) {
        if (!event_can_reach_wait(event, group.section_offset, arch))
          continue;
        const uint32_t required_count = dependency_required_count(*requirement_state, event, arch);
        if (!strongest || required_count < strongest->required_count) {
          strongest = CounterParityRequirement{event,
                                               RegisterRef{RegClass::PC, 0, 1},
                                               WaitcheckAccessKind::MemoryOrder,
                                               required_count,
                                               &consumer,
                                               consumer_section_offset,
                                               consumer_file_offset};
        }
      }
    }

    if (!uses_legacy_waitcnt(arch) && starts_with(consumer.mnemonic(), "s_barrier_signal")) {
      // Split barriers do not encode an implicit DS drain. LLVM-produced
      // synchronization sequences therefore place s_wait_dscnt 0 before the
      // signal when LDS operations are outstanding. Attribute that field to
      // the barrier's memory-order requirement; otherwise forward lookahead
      // can incorrectly attach the deliberately strong wait to a register
      // dependency after the matching barrier wait.
      auto &strongest = requirements[counter_index(WaitCounterKind::Ds)];
      for (const PendingEvent &event :
           requirement_state->pending[counter_index(WaitCounterKind::Ds)]) {
        if (!event_can_reach_wait(event, group.section_offset, arch))
          continue;
        strongest = CounterParityRequirement{event,
                                             RegisterRef{RegClass::PC, 0, 1},
                                             WaitcheckAccessKind::MemoryOrder,
                                             0,
                                             &consumer,
                                             consumer_section_offset,
                                             consumer_file_offset};
        break;
      }
    }

    if (!uses_legacy_waitcnt(arch) && consumer.mnemonic() == "global_inv") {
      // SIMemoryLegalizer lowers an acquire fence that crosses LDS and global
      // address spaces to soft all-zero waits followed by GLOBAL_INV.  The
      // cache instruction itself is not an ordinary register dependency, but
      // it is the object-visible marker that gives the preceding wait group
      // its memory-order meaning.  Attribute only counter fields that the
      // GFX12 fence lowering can emit; otherwise forward lookahead can attach
      // a fence drain to an unrelated register dependency after GLOBAL_INV.
      constexpr std::array fence_counters{WaitCounterKind::Load, WaitCounterKind::Store,
                                          WaitCounterKind::Ds, WaitCounterKind::Sample,
                                          WaitCounterKind::Bvh};
      for (WaitCounterKind counter : fence_counters) {
        const size_t counter_idx = counter_index(counter);
        if (!group.emitted_counts[counter_idx])
          continue;
        auto &strongest = requirements[counter_idx];
        for (const PendingEvent &event : requirement_state->pending[counter_idx]) {
          if (!event_can_reach_wait(event, group.section_offset, arch))
            continue;
          strongest = CounterParityRequirement{event,
                                               RegisterRef{RegClass::PC, 0, 1},
                                               WaitcheckAccessKind::MemoryOrder,
                                               0,
                                               &consumer,
                                               consumer_section_offset,
                                               consumer_file_offset};
          break;
        }
      }
    }

    if (consumer.mnemonic() == "s_barrier" || starts_with(consumer.mnemonic(), "ds_")) {
      for (size_t counter_idx = 0; counter_idx < requirement_state->pending.size(); ++counter_idx) {
        auto &strongest = requirements[counter_idx];
        for (const PendingEvent &event : requirement_state->pending[counter_idx]) {
          if (!event.check_counter_parity_order ||
              !event_can_reach_wait(event, group.section_offset, arch)) {
            continue;
          }
          const uint32_t required_count =
              dependency_required_count(*requirement_state, event, arch);
          if (!strongest || required_count < strongest->required_count) {
            strongest = CounterParityRequirement{event,
                                                 RegisterRef{RegClass::PC, 0, 1},
                                                 WaitcheckAccessKind::MemoryOrder,
                                                 required_count,
                                                 &consumer,
                                                 consumer_section_offset,
                                                 consumer_file_offset};
          }
        }
      }
    }
  }

  [[nodiscard]] std::optional<CounterParityRequirement> find_forward_counter_parity_requirement(
      const PendingWaitGroup &group, const Instruction &first_consumer,
      uint64_t first_consumer_section_offset, WaitCounterKind counter, uint32_t emitted_count,
      rj_code_arch_t arch, size_t max_lookahead_positions, bool &complete) {
    if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size())
      return std::nullopt;

    const CfgBlockView &start_block = cfg_views_[*current_cfg_view_index_];
    const auto start_it = std::ranges::find_if(start_block.instructions, [&](const auto &view) {
      return view.instruction == &first_consumer &&
             view.section_offset == first_consumer_section_offset;
    });
    if (start_it == start_block.instructions.end())
      return std::nullopt;

    RegisterSet possible_uses;
    RegisterSet possible_defs;
    bool may_define_exec = false;
    bool may_use_special = false;
    bool may_define_special = false;
    bool may_require_memory_order = false;
    bool may_require_program_end = false;
    bool may_require_counter_order = false;
    for (const auto &events : group.state_before.pending) {
      for (const PendingEvent &event : events) {
        if (!counter_may_retire_event(counter, event))
          continue;
        if (event.check_uses)
          possible_uses |= event.regs;
        if (event.check_defs)
          possible_defs |= event.regs;
        may_define_exec |= event.check_exec_defs;
        may_use_special |= event.check_uses && event.special_reg.has_value();
        may_define_special |= event.check_defs && event.special_reg.has_value();
        may_require_memory_order |= event.check_memory_order;
        may_require_program_end |= event.check_program_end;
        may_require_counter_order |= event.check_counter_parity_order;
      }
    }

    using Position = std::pair<size_t, size_t>;
    std::vector<Position> worklist{{
        *current_cfg_view_index_,
        static_cast<size_t>(std::distance(start_block.instructions.begin(), start_it)),
    }};
    std::set<Position> visited;
    std::set<const Instruction *> dependencies_checked;
    std::optional<CounterParityRequirement> strongest;
    while (!worklist.empty() && visited.size() < max_lookahead_positions) {
      const auto [block_index, instruction_index] = worklist.back();
      worklist.pop_back();
      if (!visited.insert({block_index, instruction_index}).second)
        continue;

      const CfgBlockView &block = cfg_views_[block_index];
      if (instruction_index >= block.instructions.size()) {
        for (size_t successor : block.successors)
          worklist.emplace_back(successor, 0);
        continue;
      }

      const CfgInstructionView &view = block.instructions[instruction_index];
      const Instruction &instruction = *view.instruction;
      if (instruction_emits_counter_wait(instruction, counter, arch)) {
        continue;
      }

      if (dependencies_checked.insert(&instruction).second) {
        const InstDefUse du = inst_def_use_for_waitcheck(instruction, group.state_before.vgpr_msb,
                                                         arch, wavefront_size_);
        const bool may_depend_on_register =
            possible_uses.intersects(du.uses) || possible_defs.intersects(du.defs);
        const bool may_depend_on_special =
            (may_use_special &&
             instruction_uses_special(instruction, RegisterRef{RegClass::SCC, 0, 1})) ||
            (may_define_special &&
             instruction_defines_special(instruction, RegisterRef{RegClass::SCC, 0, 1}));
        const bool may_depend_on_exec = may_define_exec && instruction_defines_exec(instruction);
        const bool is_barrier = instruction.mnemonic() == "s_barrier";
        const bool is_ds = starts_with(instruction.mnemonic(), "ds_");
        const bool may_depend_on_order =
            may_require_memory_order ||
            (may_require_program_end && is_program_end(instruction.mnemonic())) ||
            (may_require_counter_order && (is_barrier || is_ds)) ||
            (uses_legacy_waitcnt(arch) && counter == WaitCounterKind::Ds && is_barrier) ||
            (counter == WaitCounterKind::VaVdst && dsdir_wait_va_vdst(instruction));
        if (may_depend_on_register || may_depend_on_special || may_depend_on_exec ||
            may_depend_on_order) {
          std::array<std::optional<CounterParityRequirement>, kCounterCount> requirements;
          collect_counter_parity_requirements(
              group, instruction, view.section_offset,
              group.file_offset - group.section_offset + view.section_offset, arch, requirements);
          if (const auto &requirement = requirements[counter_index(counter)]) {
            if (!strongest || requirement->required_count < strongest->required_count)
              strongest = requirement;
            // Counter-parity auditing is intentionally one-sided. As soon as
            // one guarded dependency requires a value no weaker than the
            // emitted field, this wait cannot be an N > M under-accounting
            // candidate. Avoid scanning the rest of a large CFG solely to
            // discover an even stronger ordinary-hazard requirement.
            if (strongest->required_count <= emitted_count)
              return strongest;
          }
        }
      }
      if (is_program_end(instruction.mnemonic()))
        continue;
      worklist.emplace_back(block_index, instruction_index + 1);
    }
    if (!worklist.empty())
      complete = false;
    return strongest;
  }

  [[nodiscard]] std::optional<CounterParityRequirement> find_staged_counter_parity_requirement(
      const PendingWaitGroup &group, const Instruction &first_consumer,
      uint64_t first_consumer_section_offset, WaitCounterKind counter, uint32_t emitted_count,
      rj_code_arch_t arch, bool &complete) {
    if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size() ||
        counter == WaitCounterKind::VaVdst) {
      return std::nullopt;
    }

    const CfgBlockView &start_block = cfg_views_[*current_cfg_view_index_];
    const auto start_it = std::ranges::find_if(start_block.instructions, [&](const auto &view) {
      return view.instruction == &first_consumer &&
             view.section_offset == first_consumer_section_offset;
    });
    if (start_it == start_block.instructions.end())
      return std::nullopt;

    PendingState initial_state = group.state_before;
    for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
      if (counter_idx == counter_index(counter) || !group.emitted_counts[counter_idx])
        continue;
      apply_counter_parity_wait(initial_state, static_cast<WaitCounterKind>(counter_idx),
                                *group.emitted_counts[counter_idx], arch);
    }

    const std::vector<PendingEvent> original_events = initial_state.pending[counter_index(counter)];
    if (original_events.empty())
      return std::nullopt;

    auto original_event = [&](const PendingEvent &event) -> const PendingEvent * {
      const auto found = std::ranges::find_if(original_events, [&](const PendingEvent &original) {
        return same_event_identity(original, event);
      });
      return found == original_events.end() ? nullptr : &*found;
    };

    using Position = std::pair<size_t, size_t>;
    const Position start_position{
        *current_cfg_view_index_,
        static_cast<size_t>(std::distance(start_block.instructions.begin(), start_it)),
    };
    std::map<Position, PendingState> states;
    states.emplace(start_position, initial_state);
    std::vector<Position> worklist{start_position};
    std::optional<CounterParityRequirement> strongest;
    constexpr size_t kMaxSimulationSteps = 200000;
    size_t steps = 0;

    auto enqueue = [&](Position position, const PendingState &state) {
      auto [it, inserted] = states.try_emplace(position, state);
      if (inserted) {
        worklist.push_back(position);
        return;
      }
      PendingState merged = it->second;
      merge_into(merged, state);
      if (merged != it->second) {
        it->second = std::move(merged);
        worklist.push_back(position);
      }
    };

    while (!worklist.empty() && steps++ < kMaxSimulationSteps) {
      const Position position = worklist.back();
      worklist.pop_back();
      const auto [block_index, instruction_index] = position;
      PendingState state = states.at(position);
      const CfgBlockView &block = cfg_views_[block_index];
      if (instruction_index >= block.instructions.size()) {
        for (size_t successor : block.successors)
          enqueue({successor, 0}, state);
        continue;
      }

      const CfgInstructionView &view = block.instructions[instruction_index];
      const Instruction &instruction = *view.instruction;
      if (view.section_offset == group.section_offset && position != start_position &&
          instruction_emits_counter_wait(instruction, counter, arch)) {
        continue;
      }

      const PendingState *dependency_state = &state;
      const InstDefUse du = inst_def_use_for_waitcheck(instruction, dependency_state->vgpr_msb,
                                                       arch, wavefront_size_);
      auto current_events = classify_events(instruction, arch);
      if (!expert_waits_enabled(*dependency_state)) {
        std::erase_if(current_events, [](const ClassifiedEvent &event) {
          return event.counter == WaitCounterKind::VmVsrc ||
                 event.counter == WaitCounterKind::VaVdst;
        });
      }
      std::optional<PendingState> xcnt_adjusted_state;
      if (std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
            return event.counter == WaitCounterKind::X;
          })) {
        xcnt_adjusted_state.emplace(state);
        apply_implicit_xcnt_ordering(*xcnt_adjusted_state, du, current_events);
        dependency_state = &*xcnt_adjusted_state;
      }
      auto record_if_original = [&](const PendingEvent &event, RegisterRef reg,
                                    WaitcheckAccessKind access) {
        const PendingEvent *original = original_event(event);
        if (!original)
          return;
        const uint32_t required_count = dependency_required_count(initial_state, *original, arch);
        if (!strongest || required_count < strongest->required_count) {
          strongest = CounterParityRequirement{*original,
                                               reg,
                                               access,
                                               required_count,
                                               &instruction,
                                               view.section_offset,
                                               group.file_offset - group.section_offset +
                                                   view.section_offset};
        }
      };
      visit_dependencies(
          *dependency_state, instruction, du, current_events, arch,
          DependencyView::CompilerPendingDefinition,
          [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access, uint32_t) {
            if (event.counter == counter)
              record_if_original(event, reg, access);
          });

      if (uses_legacy_waitcnt(arch) && instruction.mnemonic() == "s_barrier" &&
          counter == WaitCounterKind::Ds) {
        for (const PendingEvent &event :
             dependency_state->pending[counter_index(WaitCounterKind::Ds)]) {
          if (event.kind != WaitEventKind::Smem) {
            record_if_original(event, RegisterRef{RegClass::PC, 0, 1},
                               WaitcheckAccessKind::MemoryOrder);
          }
        }
      }
      if (instruction.mnemonic() == "s_barrier" || starts_with(instruction.mnemonic(), "ds_")) {
        for (const PendingEvent &event : dependency_state->pending[counter_index(counter)]) {
          if (event.check_counter_parity_order) {
            record_if_original(event, RegisterRef{RegClass::PC, 0, 1},
                               WaitcheckAccessKind::MemoryOrder);
          }
        }
      }
      if (strongest && strongest->required_count <= emitted_count)
        return strongest;

      RegisterSet local_ready_regs;
      analyze_instruction(state, local_ready_regs, instruction, group.section_name,
                          view.section_offset,
                          group.file_offset - group.section_offset + view.section_offset, arch,
                          /*emit_diagnostics=*/false);
      if (is_program_end(instruction.mnemonic()))
        continue;

      const auto &pending = state.pending[counter_index(counter)];
      const bool has_original = std::ranges::any_of(
          pending, [&](const PendingEvent &event) { return original_event(event) != nullptr; });
      if (!has_original)
        continue;
      enqueue({block_index, instruction_index + 1}, state);
    }
    if (!worklist.empty())
      complete = false;
    return strongest;
  }

  void audit_counter_parity_group(const PendingWaitGroup &group, const Instruction &consumer,
                                  uint64_t consumer_section_offset, uint64_t consumer_file_offset,
                                  rj_code_arch_t arch) {
    ++report_.counter_parity_wait_groups;

    std::array<std::optional<CounterParityRequirement>, kCounterCount> strongest_requirements;
    collect_counter_parity_requirements(group, consumer, consumer_section_offset,
                                        consumer_file_offset, arch, strongest_requirements);

    bool group_indeterminate = false;
    for (size_t counter_idx = 0; counter_idx < group.emitted_counts.size(); ++counter_idx) {
      if (!group.emitted_counts[counter_idx])
        continue;
      const auto counter = static_cast<WaitCounterKind>(counter_idx);
      const auto no_wait = counter_no_wait_value(arch, counter);
      if (!no_wait)
        continue;
      const uint32_t emitted_count = *group.emitted_counts[counter_idx];

      ++report_.counter_parity_fields_checked;
      auto &requirement = strongest_requirements[counter_idx];
      // LLVM inserts unconditional all-zero waits at both boundaries of every
      // non-entry function. Do not walk the entire callee/caller continuation
      // trying to rediscover a dependency for a boundary-policy field: on
      // large outlined helpers that turns a linear audit into a scan of the
      // same CFG once per counter. An immediately attributable dependency can
      // still establish exact agreement; otherwise the field is deliberately
      // retained as unmodeled policy below.
      const bool function_boundary_policy =
          group.non_entry_function_prolog || is_non_entry_function_return(consumer);
      bool boundary_policy_fallback = false;
      if (has_counter_parity_candidate(group.state_before, counter) &&
          (!requirement || requirement->required_count > emitted_count)) {
        bool lookahead_complete = true;
        constexpr size_t kMaxGeneralLookaheadPositions = 200000;
        constexpr size_t kMaxFunctionBoundaryLookaheadPositions = 256;
        const auto forward_requirement = find_forward_counter_parity_requirement(
            group, consumer, consumer_section_offset, counter, emitted_count, arch,
            function_boundary_policy ? kMaxFunctionBoundaryLookaheadPositions
                                     : kMaxGeneralLookaheadPositions,
            lookahead_complete);
        if (!lookahead_complete) {
          if (function_boundary_policy) {
            boundary_policy_fallback = true;
          } else {
            group_indeterminate = true;
            continue;
          }
        }
        if (forward_requirement &&
            (!requirement || forward_requirement->required_count < requirement->required_count)) {
          requirement = forward_requirement;
        }
        if (!function_boundary_policy && emitted_count != 0 &&
            (!requirement || requirement->required_count > emitted_count)) {
          const auto staged_requirement = find_staged_counter_parity_requirement(
              group, consumer, consumer_section_offset, counter, emitted_count, arch,
              lookahead_complete);
          if (!lookahead_complete) {
            group_indeterminate = true;
            continue;
          }
          if (staged_requirement &&
              (!requirement || staged_requirement->required_count < requirement->required_count)) {
            requirement = staged_requirement;
          }
        }
      }
      // A stronger boundary wait is ABI policy rather than evidence that LLVM
      // calculated a tighter dependency count, so keep it in the unmodeled
      // catalog. The fast path above intentionally permits an immediate exact
      // dependency to remain an ordinary parity match.
      // SIInsertWaitcnts reaches a fixed point over loop score brackets while
      // retaining waits inserted by an earlier iteration. At a cyclic block,
      // that process can conservatively strengthen a partial wait (often by
      // one, or all the way to zero) even when the final-ISA dependency has a
      // larger safe count. The encoded count is then loop policy, not a direct
      // dependency score. Exact loop waits remain ordinary parity matches.
      const bool cyclic_cfg_fixed_point_policy = is_cyclic_cfg_wait();
      const bool needs_boundary_policy =
          function_boundary_policy &&
          (boundary_policy_fallback || !requirement || requirement->required_count > emitted_count);
      if (needs_boundary_policy || (cyclic_cfg_fixed_point_policy && requirement &&
                                    requirement->required_count > emitted_count)) {
        requirement.reset();
      }
      const uint32_t required_count = requirement ? requirement->required_count : *no_wait;
      if (requirement && required_count == emitted_count)
        ++report_.counter_parity_exact;
      if (required_count <= emitted_count)
        continue;

      WaitcheckCounterUnderaccountingDiagnostic diag;
      diag.counter = counter;
      diag.emitted_count = emitted_count;
      diag.required_count = required_count;
      diag.has_required_dependency = requirement.has_value();
      diag.section_name = group.section_name;
      diag.wait_section_offset = group.section_offset;
      diag.wait_file_offset = group.file_offset;
      diag.wait_instruction = group.instructions;
      diag.consumer_section_offset =
          requirement ? requirement->consumer_section_offset : consumer_section_offset;
      diag.consumer_file_offset =
          requirement ? requirement->consumer_file_offset : consumer_file_offset;
      diag.consumer_instruction =
          requirement ? requirement->consumer->disassemble() : consumer.disassemble();
      if (requirement) {
        diag.access = requirement->access;
        diag.reg = requirement->reg;
        diag.producer_section_offset = requirement->event.section_offset;
        diag.producer_file_offset = requirement->event.file_offset;
        diag.producer_instruction = requirement->event.instruction;
      } else {
        diag.reg = RegisterRef{RegClass::PC, 0, 1};
      }
      std::ostringstream message;
      message << "emitted " << wait_expression(counter, emitted_count, arch)
              << " is stronger than waitcheck requirement "
              << wait_expression(counter, required_count, arch) << " before "
              << diag.consumer_instruction;
      diag.message = message.str();
      record_counter_underaccounting(std::move(diag));
    }
    if (group_indeterminate)
      ++report_.counter_parity_indeterminate_groups;
  }

  void update_counter_parity_group(std::optional<PendingWaitGroup> &pending_group,
                                   const PendingState &state, const Instruction &inst,
                                   const std::string &section_name, uint64_t section_offset,
                                   uint64_t file_offset, rj_code_arch_t arch) {
    if (!options_.check_counter_parity)
      return;

    const auto emitted_fields = explicit_wait_fields(inst, arch);
    if (emitted_fields) {
      const bool has_wait =
          std::ranges::any_of(*emitted_fields, [](const auto &value) { return value.has_value(); });
      if (!pending_group && has_wait) {
        pending_group = PendingWaitGroup{state,
                                         {},
                                         section_name,
                                         section_offset,
                                         file_offset,
                                         {},
                                         is_non_entry_function_prolog(section_offset)};
      }
      if (!pending_group)
        return;
      for (size_t i = 0; i < emitted_fields->size(); ++i) {
        if (!(*emitted_fields)[i])
          continue;
        auto &combined = pending_group->emitted_counts[i];
        combined = combined ? std::min(*combined, *(*emitted_fields)[i]) : (*emitted_fields)[i];
      }
      if (!pending_group->instructions.empty())
        pending_group->instructions += "; ";
      pending_group->instructions += inst.disassemble();
      return;
    }

    if (const auto embedded_fields = embedded_wait_fields(inst, arch)) {
      if (!pending_group) {
        pending_group = PendingWaitGroup{state,
                                         {},
                                         section_name,
                                         section_offset,
                                         file_offset,
                                         {},
                                         is_non_entry_function_prolog(section_offset)};
      }
      for (size_t i = 0; i < embedded_fields->size(); ++i) {
        if (!(*embedded_fields)[i])
          continue;
        auto &combined = pending_group->emitted_counts[i];
        combined = combined ? std::min(*combined, *(*embedded_fields)[i]) : (*embedded_fields)[i];
      }
      if (!pending_group->instructions.empty())
        pending_group->instructions += "; ";
      pending_group->instructions += inst.disassemble();
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, arch);
      pending_group.reset();
      return;
    }

    if (!pending_group)
      return;
    const std::string_view mnemonic = inst.mnemonic();
    const bool known_non_counter_wait =
        mnemonic == "s_waitcnt_depctr" || mnemonic == "s_wait_event";
    if (inst.is_waitcnt() && !known_non_counter_wait) {
      ++report_.counter_parity_indeterminate_groups;
    } else {
      audit_counter_parity_group(*pending_group, inst, section_offset, file_offset, arch);
    }
    pending_group.reset();
  }

  void finish_counter_parity_block(std::optional<PendingWaitGroup> &pending_group,
                                   rj_code_arch_t arch) {
    if (!pending_group)
      return;

    // A basic block can end at a label even when the final instruction is not
    // a control-flow instruction. In that case an emitted wait at the block
    // boundary still guards the first instruction in its sole fallthrough
    // successor. Continue the pending wait group across that structural split.
    // A block ending in a wait cannot itself branch, so multiple successors
    // remain explicitly indeterminate.
    if (current_cfg_view_index_ && *current_cfg_view_index_ < cfg_views_.size()) {
      const CfgBlockView &block = cfg_views_[*current_cfg_view_index_];
      if (block.successors.size() == 1) {
        const size_t successor_index = block.successors.front();
        if (successor_index < cfg_views_.size() &&
            !cfg_views_[successor_index].instructions.empty()) {
          const CfgInstructionView consumer = cfg_views_[successor_index].instructions.front();
          bool overlapping_counter_wait = false;
          if (const auto fields = explicit_wait_fields(*consumer.instruction, arch)) {
            for (size_t i = 0; i < fields->size(); ++i) {
              overlapping_counter_wait |=
                  (*fields)[i].has_value() && pending_group->emitted_counts[i].has_value();
            }
          }
          if (!overlapping_counter_wait && !embedded_wait_fields(*consumer.instruction, arch)) {
            const size_t saved_block_index = *current_cfg_view_index_;
            current_cfg_view_index_ = successor_index;
            audit_counter_parity_group(*pending_group, *consumer.instruction,
                                       consumer.section_offset,
                                       pending_group->file_offset - pending_group->section_offset +
                                           consumer.section_offset,
                                       arch);
            current_cfg_view_index_ = saved_block_index;
            pending_group.reset();
            return;
          }
        }
      }
    }
    ++report_.counter_parity_indeterminate_groups;
    pending_group.reset();
  }

  [[nodiscard]] bool is_non_entry_function_prolog(uint64_t section_offset) const {
    if (!current_in_callee_context_ || !current_cfg_view_index_ ||
        *current_cfg_view_index_ >= cfg_views_.size()) {
      return false;
    }
    return cfg_views_[*current_cfg_view_index_].block->start_offset() == section_offset;
  }

  [[nodiscard]] bool is_non_entry_function_return(const Instruction &instruction) const {
    if (!current_call_return_sreg_ || instruction.size() != static_cast<int>(sizeof(uint32_t)) ||
        (instruction.mnemonic() != "s_setpc_b64" && instruction.mnemonic() != "s_set_pc_i64") ||
        instruction.raw_encoding() == nullptr) {
      return false;
    }
    return static_cast<uint16_t>(instruction.raw_encoding()[0] & 0xffu) ==
           *current_call_return_sreg_;
  }

  [[nodiscard]] bool is_cyclic_cfg_wait() {
    if (!current_cfg_view_index_ || *current_cfg_view_index_ >= cfg_views_.size())
      return false;
    const size_t block_index = *current_cfg_view_index_;
    const SharedReachability reachable = blocks_reachable_from(block_index);
    return std::ranges::any_of(cfg_views_[block_index].predecessors, [&](size_t predecessor) {
      return predecessor < reachable->size() && (*reachable)[predecessor] != 0;
    });
  }

  static void merge_va_vdst_hazards(VaVdstHazardState &dst, const VaVdstHazardState &src) {
    for (const auto &[index, src_hazard] : src.hazards) {
      auto [dst_it, inserted] = dst.hazards.try_emplace(index, src_hazard);
      if (inserted)
        continue;
      VaVdstHazard &dst_hazard = dst_it->second;
      if (dst_hazard == src_hazard)
        continue;

      dst_hazard.trans_since = dst_hazard.trans_since || src_hazard.trans_since;
      dst_hazard.age = dst_hazard.trans_since ? 0 : std::min(dst_hazard.age, src_hazard.age);
    }
  }

  static void merge_into(PendingState &dst, const PendingState &src) {
    for (size_t i = 0; i < kCounterCount; ++i) {
      for (size_t kind = 0; kind < kWaitEventKindCount; ++kind) {
        const uint8_t src_age = src.pending_event_ages[i].values[kind];
        uint8_t &dst_age = dst.pending_event_ages[i].values[kind];
        if (src_age != kNoPendingEventAge && (dst_age == kNoPendingEventAge || src_age < dst_age)) {
          dst_age = src_age;
        }
      }
      dst.pending_smem[i] = dst.pending_smem[i] || src.pending_smem[i];
      dst.uncertain_order[i] = dst.uncertain_order[i] || src.uncertain_order[i];
      for (const auto &event : src.pending[i]) {
        auto position = std::ranges::lower_bound(dst.pending[i], event, event_identity_less);
        if (position == dst.pending[i].end() || !same_event_identity(*position, event)) {
          dst.pending[i].insert(position, event);
        } else {
          position->min_younger = std::min(position->min_younger, event.min_younger);
          position->min_ordered_younger =
              std::min(position->min_ordered_younger, event.min_ordered_younger);
          position->old_value_regs &= event.old_value_regs;
        }
      }
    }
    merge_sgpr_hazards(dst.sgpr_hazards, src.sgpr_hazards);
    merge_va_vdst_hazards(dst.va_vdst_hazards, src.va_vdst_hazards);
    dst.vgpr_msb_setreg_hazard = dst.vgpr_msb_setreg_hazard || src.vgpr_msb_setreg_hazard;
    if (!dst.async_barrier_post_wait && src.async_barrier_post_wait)
      dst.async_barrier_post_wait = src.async_barrier_post_wait;
    for (const PendingDelayAlu &delay : src.delay_alu) {
      if (std::find(dst.delay_alu.begin(), dst.delay_alu.end(), delay) == dst.delay_alu.end())
        dst.delay_alu.push_back(delay);
    }
  }

  [[nodiscard]] static PendingState
  merge_predecessors(std::span<const size_t> predecessors, const std::vector<PendingState> &outputs,
                     std::span<const uint8_t> output_initialized) {
    PendingState merged;
    if (predecessors.empty())
      return merged;

    std::array<std::optional<std::vector<PendingEvent>>, kCounterCount> first_source;
    std::optional<RegisterSet> ready_regs;
    std::optional<VgprMsbState> first_vgpr_msb;
    std::optional<ExpertSchedulingState> first_expert_scheduling;
    std::optional<bool> all_previous_vm_vsrc_zero_wait;
    for (size_t predecessor : predecessors) {
      if (predecessor >= outputs.size() || predecessor >= output_initialized.size() ||
          output_initialized[predecessor] == 0)
        continue;
      const PendingState &pred_out = outputs[predecessor];
      if (!all_previous_vm_vsrc_zero_wait) {
        all_previous_vm_vsrc_zero_wait = pred_out.previous_vm_vsrc_zero_wait;
      } else {
        *all_previous_vm_vsrc_zero_wait &= pred_out.previous_vm_vsrc_zero_wait;
      }
      if (!ready_regs) {
        ready_regs = pred_out.ready_regs;
      } else {
        *ready_regs &= pred_out.ready_regs;
      }
      if (!first_expert_scheduling) {
        first_expert_scheduling = pred_out.expert_scheduling;
      } else if (*first_expert_scheduling != pred_out.expert_scheduling) {
        first_expert_scheduling->known = false;
        first_expert_scheduling->enabled = true;
      }
      if (!first_vgpr_msb) {
        first_vgpr_msb = pred_out.vgpr_msb;
      } else if (*first_vgpr_msb != pred_out.vgpr_msb) {
        first_vgpr_msb->known = false;
        first_vgpr_msb->mode = 0;
      }
      for (size_t i = 0; i < kCounterCount; ++i) {
        if (!pred_out.pending[i].empty()) {
          if (!first_source[i]) {
            first_source[i] = pred_out.pending[i];
          } else if (*first_source[i] != pred_out.pending[i]) {
            merged.uncertain_order[i] = true;
          }
        }
      }
      merge_into(merged, pred_out);
    }
    if (first_vgpr_msb)
      merged.vgpr_msb = *first_vgpr_msb;
    if (first_expert_scheduling)
      merged.expert_scheduling = *first_expert_scheduling;
    if (ready_regs)
      merged.ready_regs = *ready_regs;
    if (all_previous_vm_vsrc_zero_wait)
      merged.previous_vm_vsrc_zero_wait = *all_previous_vm_vsrc_zero_wait;
    return merged;
  }

  [[nodiscard]] PendingState analyze_block(BasicBlock &block, const PendingState &input,
                                           const std::string &section_name,
                                           uint64_t file_offset_base, rj_code_arch_t arch,
                                           bool emit_diagnostics) {
    PendingState state = input;
    if (!tracks_committed_vgpr_generations(arch))
      state.ready_regs = {};
    RegisterSet local_ready_regs;
    std::optional<PendingWaitGroup> pending_wait_group;
    uint64_t section_offset = block.start_offset();
    for (const Instruction &inst : block.instructions()) {
      if (emit_diagnostics) {
        update_counter_parity_group(pending_wait_group, state, inst, section_name, section_offset,
                                    file_offset_base + section_offset, arch);
      }
      analyze_instruction(state, local_ready_regs, inst, section_name, section_offset,
                          file_offset_base + section_offset, arch, emit_diagnostics);
      if (emit_diagnostics)
        ++report_.instructions_analyzed;
      section_offset += static_cast<uint64_t>(inst.size());
      if (emit_diagnostics && should_stop_after_diagnostic())
        break;
    }
    if (emit_diagnostics)
      finish_counter_parity_block(pending_wait_group, arch);
    if (!tracks_committed_vgpr_generations(arch))
      state.ready_regs = {};
    return state;
  }

  [[nodiscard]] static bool same_register_generation(const PendingEvent &lhs,
                                                     const PendingEvent &rhs) {
    return lhs.produces_regs && rhs.produces_regs && lhs.regs == rhs.regs &&
           lhs.section_name == rhs.section_name && lhs.section_offset == rhs.section_offset &&
           lhs.file_offset == rhs.file_offset && lhs.instruction == rhs.instruction;
  }

  static void make_retired_generations_ready(PendingState &state,
                                             std::span<const PendingEvent> retired_events) {
    for (const PendingEvent &retired : retired_events) {
      if (!retired.produces_regs)
        continue;
      retired.regs.for_each([&](RegisterRef reg) {
        if (reg.cls != RegClass::VGPR && reg.cls != RegClass::ACC_VGPR)
          return;
        const bool generation_still_pending =
            std::ranges::any_of(state.pending, [&](const std::vector<PendingEvent> &events) {
              return std::ranges::any_of(events, [&](const PendingEvent &pending) {
                return pending.regs.contains(reg) && same_register_generation(retired, pending);
              });
            });
        if (generation_still_pending)
          return;

        state.ready_regs.expand(reg);
        for (auto &events : state.pending) {
          for (PendingEvent &pending : events) {
            if (pending.produces_regs && pending.regs.contains(reg))
              pending.old_value_regs.expand(reg);
          }
        }
      });
    }
  }

  template <typename Predicate>
  static void retire_events(PendingState &state, std::vector<PendingEvent> &events,
                            Predicate should_retire) {
    std::vector<PendingEvent> retired_events;
    const auto retained =
        std::remove_if(events.begin(), events.end(), [&](const PendingEvent &event) {
          if (!should_retire(event))
            return false;
          retired_events.push_back(event);
          return true;
        });
    events.erase(retained, events.end());
    make_retired_generations_ready(state, retired_events);
  }

  static void apply_wait_to_event_ages(PendingState &state, WaitCounterKind counter,
                                       uint32_t count) {
    auto &ages = state.pending_event_ages[counter_index(counter)].values;
    for (uint8_t &age : ages) {
      if (age != kNoPendingEventAge && (count == 0 || age >= count))
        age = kNoPendingEventAge;
    }
  }

  static void apply_wait(PendingState &state, WaitCounterKind counter, uint32_t count) {
    const size_t idx = counter_index(counter);
    auto &pending = state.pending[idx];
    apply_wait_to_event_ages(state, counter, count);
    if (count == 0) {
      retire_events(state, pending, [](const PendingEvent &) { return true; });
      state.pending_smem[idx] = false;
      state.uncertain_order[idx] = false;
      return;
    }
    retire_events(state, pending,
                  [count](const PendingEvent &event) { return event.min_younger >= count; });
    if (pending.empty())
      state.uncertain_order[idx] = false;
  }

  static void apply_dtl_vmcnt_wait(PendingState &state, uint32_t count) {
    for (DtlVisibilityEvent &event : state.dtl_visibility) {
      if (event.active && (count == 0 || event.min_younger >= count))
        event.active = false;
    }
  }

  static void apply_kmcnt_wait(PendingState &state, uint32_t count, rj_code_arch_t arch) {
    // LLVM cannot use a partial wait to advance any part of a counter whose
    // pending event kinds may complete out of order.
    if (count != 0 && counter_out_of_order(state, WaitCounterKind::Km, arch))
      return;
    apply_wait(state, WaitCounterKind::Km, count);
    apply_xcnt_wait_implied_by_kmcnt(state, count);
  }

  [[nodiscard]] static bool vm_vsrc_event_implied_by_wait(WaitEventKind kind,
                                                          WaitCounterKind counter) {
    switch (counter) {
    case WaitCounterKind::Load:
      return kind == WaitEventKind::VmemNoSamplerLoad || kind == WaitEventKind::FlatLoad;
    case WaitCounterKind::Store:
      return kind == WaitEventKind::VmemStore || kind == WaitEventKind::FlatStore;
    case WaitCounterKind::Ds:
      return kind == WaitEventKind::Ds || kind == WaitEventKind::FlatLoad ||
             kind == WaitEventKind::FlatStore;
    case WaitCounterKind::Sample:
      return kind == WaitEventKind::Sample;
    case WaitCounterKind::Bvh:
      return kind == WaitEventKind::Bvh;
    default:
      return false;
    }
  }

  template <typename Predicate>
  static void apply_filtered_vm_vsrc_wait(PendingState &state, Predicate is_implied,
                                          uint32_t count) {
    const size_t idx = counter_index(WaitCounterKind::VmVsrc);
    auto &pending = state.pending[idx];
    if (pending.empty())
      return;

    const size_t matching = static_cast<size_t>(std::ranges::count_if(pending, is_implied));
    if (matching == 0)
      return;

    if (count == 0) {
      retire_events(state, pending, is_implied);
      if (pending.empty())
        state.uncertain_order[idx] = false;
      return;
    }

    if (state.uncertain_order[idx] || count >= matching)
      return;

    // Pending vectors are canonicalized by event identity for stable CFG
    // equality, so their physical order is not the hardware issue order.  An
    // event's counter age is the ordering fact: keep the `count` youngest
    // matching events and retire the older ones.  Equal ages at the boundary
    // mean the merge lost their relative order, in which case a partial wait
    // cannot prove that either one retired.
    std::vector<uint32_t> matching_ages;
    matching_ages.reserve(matching);
    for (const PendingEvent &event : pending) {
      if (is_implied(event))
        matching_ages.push_back(event.min_younger);
    }
    std::ranges::sort(matching_ages);
    if (matching_ages[count - 1] == matching_ages[count]) {
      state.uncertain_order[idx] = true;
      return;
    }
    const uint32_t youngest_retired_age = matching_ages[count];
    retire_events(state, pending, [&](const PendingEvent &event) {
      return is_implied(event) && event.min_younger >= youngest_retired_age;
    });
  }

  static void apply_implied_vm_vsrc_wait(PendingState &state, WaitCounterKind counter,
                                         uint32_t count) {
    apply_filtered_vm_vsrc_wait(
        state,
        [&](const PendingEvent &event) {
          return vm_vsrc_event_implied_by_wait(event.kind, counter);
        },
        count);
  }

  [[nodiscard]] static bool is_xcnt_smem_event(const PendingEvent &event) {
    return event.counter == WaitCounterKind::X && event.kind == WaitEventKind::Smem;
  }

  [[nodiscard]] static bool is_xcnt_vmem_kind(WaitEventKind kind) {
    switch (kind) {
    case WaitEventKind::VmemNoSamplerLoad:
    case WaitEventKind::FlatLoad:
    case WaitEventKind::VmemStore:
    case WaitEventKind::FlatStore:
    case WaitEventKind::Sample:
    case WaitEventKind::Bvh:
      return true;
    default:
      return false;
    }
  }

  [[nodiscard]] static bool is_xcnt_vmem_event(const PendingEvent &event) {
    return event.counter == WaitCounterKind::X && is_xcnt_vmem_kind(event.kind);
  }

  [[nodiscard]] static bool is_xcnt_store_event(const PendingEvent &event) {
    return event.counter == WaitCounterKind::X &&
           (event.kind == WaitEventKind::VmemStore || event.kind == WaitEventKind::FlatStore);
  }

  static void apply_xcnt_wait(PendingState &state, uint32_t count) {
    const size_t idx = counter_index(WaitCounterKind::X);
    const auto &pending = state.pending[idx];
    // SIInsertWaitcnts treats X_CNT as out of order while an SMEM
    // translation is pending. Only xcnt(0) proves that a particular scalar
    // source has been released.
    if (count != 0 && std::ranges::any_of(pending, is_xcnt_smem_event))
      return;
    apply_wait(state, WaitCounterKind::X, count);
  }

  template <typename Predicate>
  static void retire_xcnt_group(PendingState &state, Predicate belongs_to_group) {
    const size_t idx = counter_index(WaitCounterKind::X);
    auto &pending = state.pending[idx];
    retire_events(state, pending, belongs_to_group);
    auto &ages = state.pending_event_ages[idx].values;
    for (size_t kind_index = 0; kind_index < ages.size(); ++kind_index) {
      PendingEvent probe;
      probe.counter = WaitCounterKind::X;
      probe.kind = static_cast<WaitEventKind>(kind_index);
      if (belongs_to_group(probe))
        ages[kind_index] = kNoPendingEventAge;
    }
    PendingEvent smem_probe;
    smem_probe.counter = WaitCounterKind::X;
    smem_probe.kind = WaitEventKind::Smem;
    if (belongs_to_group(smem_probe))
      state.pending_smem[idx] = false;
    if (pending.empty())
      state.uncertain_order[idx] = false;
  }

  static void apply_xcnt_wait_implied_by_kmcnt(PendingState &state, uint32_t count) {
    if (count != 0)
      return;
    const size_t idx = counter_index(WaitCounterKind::X);
    const auto &pending = state.pending[idx];
    if (!std::ranges::any_of(pending, is_xcnt_smem_event))
      return;
    if (std::ranges::any_of(pending, is_xcnt_vmem_event)) {
      retire_xcnt_group(state, is_xcnt_smem_event);
      return;
    }
    apply_xcnt_wait(state, 0);
  }

  static void apply_xcnt_wait_implied_by_loadcnt(PendingState &state, uint32_t count) {
    const size_t idx = counter_index(WaitCounterKind::X);
    const auto &pending = state.pending[idx];
    if (!std::ranges::any_of(pending, is_xcnt_vmem_event) ||
        std::ranges::any_of(pending, is_xcnt_store_event))
      return;
    if (std::ranges::any_of(pending, is_xcnt_smem_event)) {
      if (count == 0)
        retire_xcnt_group(state, is_xcnt_vmem_event);
      return;
    }
    apply_xcnt_wait(state, count);
  }

  [[nodiscard]] static bool scalar_memory_makes_counter_out_of_order(const PendingState &state,
                                                                     WaitCounterKind counter,
                                                                     rj_code_arch_t arch) {
    return counter == smem_wait_counter(arch) && state.pending_smem[counter_index(counter)];
  }

  [[nodiscard]] static bool counter_has_event_kind(const PendingState &state,
                                                   WaitCounterKind counter, WaitEventKind kind) {
    return state.pending_event_ages[counter_index(counter)].values[static_cast<size_t>(kind)] !=
           kNoPendingEventAge;
  }

  [[nodiscard]] static std::optional<WaitEventKind>
  normalized_hardware_event_kind(WaitCounterKind counter, WaitEventKind kind, rj_code_arch_t arch) {
    switch (counter) {
    case WaitCounterKind::Load:
      // Generic FLAT and ordinary VMEM loads both raise VMEM_READ_ACCESS.
      // GLOBAL_INV is explicitly ignored by LLVM's LOAD_CNT out-of-order
      // test. Pre-gfx12 image event kinds share that same hardware event.
      if (kind == WaitEventKind::GlobalInv)
        return std::nullopt;
      if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::LdsDirect ||
          (uses_legacy_waitcnt(arch) &&
           (kind == WaitEventKind::Sample || kind == WaitEventKind::Bvh))) {
        return WaitEventKind::VmemNoSamplerLoad;
      }
      return kind;
    case WaitCounterKind::Ds:
      // A generic FLAT access raises the same LDS_ACCESS event as native DS.
      if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore)
        return WaitEventKind::Ds;
      return kind;
    case WaitCounterKind::Store:
      if (kind == WaitEventKind::GlobalWb)
        return WaitEventKind::VmemStore;
      return kind;
    case WaitCounterKind::X:
      // X_CNT distinguishes VMEM_GROUP from SMEM_GROUP, not the underlying
      // load/store/image operation.
      if (kind == WaitEventKind::Smem)
        return WaitEventKind::Smem;
      if (is_xcnt_vmem_kind(kind))
        return WaitEventKind::VmemNoSamplerLoad;
      return kind;
    case WaitCounterKind::VmVsrc:
      if (kind == WaitEventKind::Ds)
        return WaitEventKind::Ds;
      if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore)
        return WaitEventKind::FlatLoad;
      if (is_xcnt_vmem_kind(kind))
        return WaitEventKind::VmemNoSamplerLoad;
      return kind;
    case WaitCounterKind::Async:
      // Load, store, and barrier forms all raise ASYNC_ACCESS.
      return WaitEventKind::AsyncLdsLoad;
    case WaitCounterKind::Tensor:
      return WaitEventKind::TensorLdsLoad;
    default:
      return kind;
    }
  }

  [[nodiscard]] static bool flat_memory_makes_counter_out_of_order(const PendingState &state,
                                                                   WaitCounterKind counter,
                                                                   rj_code_arch_t arch) {
    if ((arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4) ||
        (counter != WaitCounterKind::Load && counter != WaitCounterKind::Ds)) {
      return false;
    }
    return counter_has_event_kind(state, counter, WaitEventKind::FlatLoad) ||
           counter_has_event_kind(state, counter, WaitEventKind::FlatStore);
  }

  [[nodiscard]] static bool counter_out_of_order(const PendingState &state, WaitCounterKind counter,
                                                 rj_code_arch_t arch) {
    // Match WaitcntBrackets::counterOutOfOrder. Scalar memory can always
    // complete out of order on its accounting counter (and on X_CNT).
    if (scalar_memory_makes_counter_out_of_order(state, counter, arch) ||
        (counter == WaitCounterKind::X &&
         counter_has_event_kind(state, counter, WaitEventKind::Smem)) ||
        flat_memory_makes_counter_out_of_order(state, counter, arch)) {
      return true;
    }

    // Before VScnt, VMEM loads and stores share LOAD_CNT but are mutually
    // ordered, so LLVM deliberately does not treat their distinct event bits
    // as an out-of-order mixture.
    if (counter == WaitCounterKind::Load && waitcnt_model(arch) == WaitcntModel::LegacyNoVscnt) {
      return false;
    }

    std::bitset<kWaitEventKindCount> kinds;
    const auto &ages = state.pending_event_ages[counter_index(counter)].values;
    for (size_t kind_index = 0; kind_index < ages.size(); ++kind_index) {
      if (ages[kind_index] == kNoPendingEventAge)
        continue;
      const auto normalized =
          normalized_hardware_event_kind(counter, static_cast<WaitEventKind>(kind_index), arch);
      if (normalized)
        kinds.set(static_cast<size_t>(*normalized));
    }
    return kinds.count() > 1;
  }

  static void apply_counter_wait(PendingState &state, WaitCounterKind counter, uint32_t count,
                                 rj_code_arch_t arch) {
    if (count != 0 && counter_out_of_order(state, counter, arch))
      return;
    apply_wait(state, counter, count);
  }

  [[nodiscard]] static uint32_t dependency_required_count(const PendingState &state,
                                                          const PendingEvent &event,
                                                          rj_code_arch_t arch) {
    if (state.uncertain_order[counter_index(event.counter)] ||
        counter_out_of_order(state, event.counter, arch))
      return 0;
    return event.min_younger;
  }

  static void apply_memory_wait(PendingState &state, WaitCounterKind counter, uint32_t count,
                                rj_code_arch_t arch) {
    // SIInsertWaitcnts cannot use a nonzero wait to retire a particular event
    // while scalar memory is pending on this counter. This notably covers
    // legacy LGKMCNT shared by SMEM and DS operations.
    if (count != 0 && counter_out_of_order(state, counter, arch))
      return;
    if (counter == WaitCounterKind::Load)
      apply_dtl_vmcnt_wait(state, count);
    apply_wait(state, counter, count);
    apply_implied_vm_vsrc_wait(state, counter, count);
    if (counter == WaitCounterKind::Load)
      apply_xcnt_wait_implied_by_loadcnt(state, count);
  }

  [[nodiscard]] static bool expert_waits_enabled(const PendingState &state) {
    return !state.expert_scheduling.known || state.expert_scheduling.enabled;
  }

  static void clear_expert_wait_state(PendingState &state) {
    state.pending[counter_index(WaitCounterKind::VmVsrc)].clear();
    state.uncertain_order[counter_index(WaitCounterKind::VmVsrc)] = false;
    clear_va_vdst_hazards(state.va_vdst_hazards);
  }

  static void clear_salu_sgpr_hazards(SgprHazardState &state) {
    state.salu_hazards.reset();
    state.salu_producers.clear();
    state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardSalu);
    state.salu_vcc_producer.reset();
  }

  static void clear_valu_sgpr_hazards(SgprHazardState &state) {
    state.valu_hazards.reset();
    state.valu_producers.clear();
  }

  static void clear_valu_vcc_hazard(SgprHazardState &state) {
    state.vcc_hazard = static_cast<uint8_t>(state.vcc_hazard & ~kSgprHazardValu);
    state.valu_vcc_producer.reset();
  }

  static void clear_all_sgpr_hazards(SgprHazardState &state) {
    clear_salu_sgpr_hazards(state);
    clear_valu_sgpr_hazards(state);
    clear_valu_vcc_hazard(state);
  }

  static void merge_lane_producers(std::unordered_map<uint16_t, SgprHazardProducer> &dst,
                                   const std::unordered_map<uint16_t, SgprHazardProducer> &src) {
    for (const auto &[index, producer] : src)
      dst.try_emplace(index, producer);
  }

  static void merge_sgpr_hazards(SgprHazardState &dst, const SgprHazardState &src) {
    dst.tracked_pairs |= src.tracked_pairs;
    dst.tracked_vcc = dst.tracked_vcc || src.tracked_vcc;
    dst.salu_hazards |= src.salu_hazards;
    dst.valu_hazards |= src.valu_hazards;
    dst.vcc_hazard |= src.vcc_hazard;
    merge_lane_producers(dst.salu_producers, src.salu_producers);
    merge_lane_producers(dst.valu_producers, src.valu_producers);
    if (!dst.salu_vcc_producer && src.salu_vcc_producer)
      dst.salu_vcc_producer = src.salu_vcc_producer;
    if (!dst.valu_vcc_producer && src.valu_vcc_producer)
      dst.valu_vcc_producer = src.valu_vcc_producer;
  }

  [[nodiscard]] static uint32_t depctr_field(uint32_t value, uint32_t shift, uint32_t width) {
    return (value >> shift) & ((1u << width) - 1u);
  }

  static void apply_sgpr_hazard_wait(SgprHazardState &state, uint32_t depctr) {
    constexpr uint32_t kDepctrSaSdstShift = 0;
    constexpr uint32_t kDepctrVaVccShift = 1;
    constexpr uint32_t kDepctrVaSdstShift = 9;
    constexpr uint32_t kDepctrSaSdstWidth = 1;
    constexpr uint32_t kDepctrVaVccWidth = 1;
    constexpr uint32_t kDepctrVaSdstWidth = 3;

    if (depctr_field(depctr, kDepctrSaSdstShift, kDepctrSaSdstWidth) == 0)
      clear_salu_sgpr_hazards(state);
    if (depctr_field(depctr, kDepctrVaVccShift, kDepctrVaVccWidth) == 0)
      clear_valu_vcc_hazard(state);
    if (depctr_field(depctr, kDepctrVaSdstShift, kDepctrVaSdstWidth) == 0)
      clear_valu_sgpr_hazards(state);
  }

  [[nodiscard]] static DelayAluEffect decode_delay_alu_effect(uint32_t instid) {
    if (instid >= 1 && instid <= 8)
      return DelayAluEffect::Valu;
    if (instid >= 9 && instid <= 11)
      return DelayAluEffect::Salu;
    return DelayAluEffect::None;
  }

  static void append_delay_alu(PendingState &state, uint32_t countdown, DelayAluEffect effect) {
    if (effect == DelayAluEffect::None)
      return;
    PendingDelayAlu delay{static_cast<uint8_t>(countdown), effect};
    if (std::find(state.delay_alu.begin(), state.delay_alu.end(), delay) == state.delay_alu.end())
      state.delay_alu.push_back(delay);
  }

  static void apply_s_delay_alu(PendingState &state, const Instruction &inst) {
    constexpr uint32_t kInstidMask = 0xfu;
    constexpr uint32_t kInstskipShift = 4;
    constexpr uint32_t kInstskipMask = 0x7u;
    constexpr uint32_t kInstid1Shift = 7;

    const Operand *op = inst.src_operand(0);
    const uint32_t value = op ? static_cast<uint32_t>(op->encoding_value()) : 0;
    append_delay_alu(state, 0, decode_delay_alu_effect(value & kInstidMask));
    append_delay_alu(state, (value >> kInstskipShift) & kInstskipMask,
                     decode_delay_alu_effect((value >> kInstid1Shift) & kInstidMask));
    state.sgpr_hazards.consecutive_ds_nops = 0;
  }

  static void apply_due_delay_alu(PendingState &state) {
    for (const PendingDelayAlu &delay : state.delay_alu) {
      if (delay.countdown != 0)
        continue;
      switch (delay.effect) {
      case DelayAluEffect::Valu:
        clear_valu_sgpr_hazards(state.sgpr_hazards);
        clear_valu_vcc_hazard(state.sgpr_hazards);
        break;
      case DelayAluEffect::Salu:
        clear_salu_sgpr_hazards(state.sgpr_hazards);
        break;
      case DelayAluEffect::None:
        break;
      }
    }
    std::erase_if(state.delay_alu,
                  [](const PendingDelayAlu &delay) { return delay.countdown == 0; });
  }

  static void advance_delay_alu(PendingState &state) {
    for (PendingDelayAlu &delay : state.delay_alu) {
      if (delay.countdown > 0)
        --delay.countdown;
    }
  }

  [[nodiscard]] static bool is_vm_vsrc_zero_wait(const Instruction &inst) {
    if (inst.mnemonic() != "s_wait_alu")
      return false;
    const Operand *op = inst.src_operand(0);
    if (!op)
      return false;
    constexpr uint32_t kDepctrVmVsrcShift = 2;
    constexpr uint32_t kDepctrVmVsrcWidth = 3;
    return depctr_field(static_cast<uint32_t>(op->encoding_value()), kDepctrVmVsrcShift,
                        kDepctrVmVsrcWidth) == 0;
  }

  void apply_waitcnt(PendingState &state, const Instruction &inst, rj_code_arch_t arch) {
    const auto mnemonic = inst.mnemonic();
    if (mnemonic == "s_wait_idle") {
      VgprMsbState vgpr_msb = state.vgpr_msb;
      ExpertSchedulingState expert_scheduling = state.expert_scheduling;
      state = {};
      state.vgpr_msb = vgpr_msb;
      state.expert_scheduling = expert_scheduling;
      return;
    }

    auto src_encoding = [&](int operand_index) {
      const Operand *op = inst.src_operand(operand_index);
      return op ? static_cast<uint32_t>(op->encoding_value()) : 0;
    };
    const bool legacy_sopk_wait = mnemonic == "s_waitcnt_vscnt" || mnemonic == "s_waitcnt_vmcnt" ||
                                  mnemonic == "s_waitcnt_expcnt" || mnemonic == "s_waitcnt_lgkmcnt";
    const uint32_t value = legacy_sopk_wait ? src_encoding(1) : src_encoding(0);

    if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt") {
      const LegacyWaitcnt wait = decode_legacy_waitcnt(value, arch);
      apply_memory_wait(state, WaitCounterKind::Load, wait.vmcnt, arch);
      apply_counter_wait(state, WaitCounterKind::Exp, wait.expcnt, arch);
      apply_memory_wait(state, WaitCounterKind::Ds, wait.lgkmcnt, arch);
    } else if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt_vmcnt") {
      apply_memory_wait(state, WaitCounterKind::Load, value, arch);
    } else if (has_legacy_vscnt(arch) && mnemonic == "s_waitcnt_vscnt") {
      apply_memory_wait(state, WaitCounterKind::Store, value, arch);
    } else if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt_expcnt") {
      apply_counter_wait(state, WaitCounterKind::Exp, value, arch);
    } else if (uses_legacy_waitcnt(arch) && mnemonic == "s_waitcnt_lgkmcnt") {
      apply_memory_wait(state, WaitCounterKind::Ds, value, arch);
    } else if (mnemonic == "s_wait_alu") {
      apply_sgpr_hazard_wait(state.sgpr_hazards, value);
      constexpr uint32_t kDepctrVmVsrcShift = 2;
      constexpr uint32_t kDepctrVmVsrcWidth = 3;
      apply_counter_wait(state, WaitCounterKind::VmVsrc,
                         depctr_field(value, kDepctrVmVsrcShift, kDepctrVmVsrcWidth), arch);
    } else if (mnemonic == "s_wait_loadcnt") {
      apply_memory_wait(state, WaitCounterKind::Load, value, arch);
    } else if (mnemonic == "s_wait_storecnt") {
      apply_memory_wait(state, WaitCounterKind::Store, value, arch);
    } else if (mnemonic == "s_wait_dscnt") {
      apply_memory_wait(state, WaitCounterKind::Ds, value, arch);
    } else if (mnemonic == "s_wait_kmcnt") {
      apply_kmcnt_wait(state, value, arch);
    } else if (mnemonic == "s_wait_samplecnt") {
      apply_memory_wait(state, WaitCounterKind::Sample, value, arch);
    } else if (mnemonic == "s_wait_bvhcnt") {
      apply_memory_wait(state, WaitCounterKind::Bvh, value, arch);
    } else if (mnemonic == "s_wait_expcnt") {
      apply_counter_wait(state, WaitCounterKind::Exp, value, arch);
    } else if (mnemonic == "s_wait_xcnt") {
      apply_xcnt_wait(state, value);
    } else if (mnemonic == "s_wait_asynccnt") {
      apply_counter_wait(state, WaitCounterKind::Async, value, arch);
    } else if (mnemonic == "s_wait_tensorcnt") {
      apply_counter_wait(state, WaitCounterKind::Tensor, value, arch);
    } else if (mnemonic == "s_wait_loadcnt_dscnt") {
      apply_memory_wait(state, WaitCounterKind::Load, (value >> 8) & 0x3Fu, arch);
      apply_memory_wait(state, WaitCounterKind::Ds, value & 0x3Fu, arch);
    } else if (mnemonic == "s_wait_storecnt_dscnt") {
      apply_memory_wait(state, WaitCounterKind::Store, (value >> 8) & 0x3Fu, arch);
      apply_memory_wait(state, WaitCounterKind::Ds, value & 0x3Fu, arch);
    }
  }

  [[nodiscard]] static std::optional<int64_t> parse_operand_immediate(const Operand *op) {
    if (op == nullptr || op->to_register_ref())
      return std::nullopt;

    std::string name = op->name();
    int base = 10;
    std::string_view text(name);
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
      text.remove_prefix(2);
      base = 16;
    }
    int64_t value = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (ec != std::errc{} || ptr != text.data() + text.size())
      return std::nullopt;
    return value;
  }

  [[nodiscard]] static std::optional<int64_t> first_operand_value(const Instruction &inst) {
    const Operand *op = inst.src_operand(0);
    if (!op)
      return std::nullopt;
    return static_cast<int64_t>(static_cast<int32_t>(op->encoding_value()));
  }

  [[nodiscard]] static std::optional<int64_t> first_barrier_id(const Instruction &inst) {
    const Operand *op = inst.src_operand(0);
    if (!op)
      return std::nullopt;
    if (const auto value = op->const_value())
      return static_cast<int64_t>(*value);
    return static_cast<int64_t>(static_cast<int32_t>(op->encoding_value()));
  }

  [[nodiscard]] static bool is_xcnt_drain(const Instruction &inst,
                                          bool immediately_after_mode_setreg = false) {
    if ((inst.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR)) != 0)
      return true;

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_trap" || mnemonic == "s_getreg_b32" ||
           starts_with(mnemonic, "s_setreg") || starts_with(mnemonic, "s_sendmsg") ||
           starts_with(mnemonic, "s_barrier_wait") || starts_with(mnemonic, "s_barrier_signal") ||
           (mnemonic == "s_set_vgpr_msb" && !immediately_after_mode_setreg);
  }

  static void apply_xcnt_drain(PendingState &state, const Instruction &inst, rj_code_arch_t arch,
                               bool immediately_after_mode_setreg) {
    if (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_xcnt_drain(inst, immediately_after_mode_setreg))
      apply_xcnt_wait(state, 0);
  }

  static bool apply_vgpr_msb_mode(PendingState &state, const Instruction &inst, rj_code_arch_t arch,
                                  bool immediately_after_mode_setreg) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic == "s_set_vgpr_msb") {
      // GFX1250 silently drops this instruction when it immediately follows
      // S_SETREG_IMM32_B32 targeting MODE. AMDGPULowerVGPREncoding inserts an
      // S_NOP for exactly this hazard.
      if (!immediately_after_mode_setreg) {
        const auto value = first_operand_value(inst);
        state.vgpr_msb.mode = value ? static_cast<uint8_t>(*value) : 0;
        state.vgpr_msb.known = true;
      }
      return true;
    }

    if (mnemonic != "s_setreg_imm32_b32" && mnemonic != "s_setreg_b32")
      return false;
    const Operand *hwreg = inst.dst_operand(0);
    if (hwreg == nullptr)
      return false;
    constexpr uint32_t kHwRegMode = 1;
    if ((static_cast<uint32_t>(hwreg->encoding_value()) & 0x3fu) != kHwRegMode)
      return false;

    if (mnemonic != "s_setreg_imm32_b32" || inst.raw_encoding() == nullptr ||
        inst.size() < 2 * static_cast<int>(sizeof(uint32_t))) {
      state.vgpr_msb.known = false;
      state.vgpr_msb.mode = 0;
      return false;
    }

    // MODE stores the eight VGPR MSB bits as dst/src0/src1/src2, while
    // S_SET_VGPR_MSB and VgprMsbState use src0/src1/src2/dst.
    const uint8_t mode_bits = static_cast<uint8_t>((inst.raw_encoding()[1] >> 12u) & 0xffu);
    state.vgpr_msb.mode = std::rotr(mode_bits, 2);
    state.vgpr_msb.known = true;
    state.vgpr_msb_setreg_hazard = true;
    return false;
  }

  static void apply_expert_scheduling_mode(PendingState &state, const Instruction &inst,
                                           rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 && arch != ROCJITSU_CODE_ARCH_RDNA4)
      return;
    if (inst.mnemonic() != "s_setreg_imm32_b32" && inst.mnemonic() != "s_setreg_b32")
      return;

    constexpr uint32_t kHwRegWaveSchedMode = 26;
    constexpr uint32_t kSchedModeValue = 2;
    const Operand *hwreg = inst.dst_operand(0);
    if (hwreg == nullptr)
      return;

    const uint32_t value = static_cast<uint32_t>(hwreg->encoding_value());
    const uint32_t reg_id = value & 0x3fu;
    const uint32_t offset = (value >> 6u) & 0x1fu;
    const uint32_t size = ((value >> 11u) & 0x1fu) + 1u;
    if (reg_id != kHwRegWaveSchedMode || offset >= 2)
      return;

    if (inst.mnemonic() != "s_setreg_imm32_b32" || inst.raw_encoding() == nullptr ||
        inst.size() < 2 * static_cast<int>(sizeof(uint32_t)) || offset != 0 || size < 2) {
      state.expert_scheduling.known = false;
      state.expert_scheduling.enabled = true;
      return;
    }

    state.expert_scheduling.known = true;
    state.expert_scheduling.enabled =
        ((inst.raw_encoding()[1] >> offset) & 0x3u) == kSchedModeValue;
    if (!state.expert_scheduling.enabled)
      clear_expert_wait_state(state);
  }

  [[nodiscard]] static bool is_exec_masked_def(RegisterRef ref) {
    return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
  }

  static void expand_vgpr_msb_ref(RegisterSet &regs, RegisterRef ref, const Operand &op,
                                  const VgprMsbState &state, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 || ref.cls != RegClass::VGPR ||
        op.vgpr_msb_role() == amdgpu::VgprMsbRole::None) {
      regs.expand(ref);
      return;
    }

    if (!state.known) {
      for (uint32_t high = 0; high < 4; ++high) {
        RegisterRef banked = ref;
        banked.index = static_cast<uint16_t>(ref.index + high * 256u);
        regs.expand(banked);
      }
      return;
    }

    ref.index = static_cast<uint16_t>(ref.index + state.for_role(op.vgpr_msb_role()) * 256u);
    regs.expand(ref);
  }

  static void add_def(InstDefUse &du, RegisterRef ref, const Operand &op, const VgprMsbState &state,
                      rj_code_arch_t arch) {
    expand_vgpr_msb_ref(du.defs, ref, op, state, arch);
    if (is_exec_masked_def(ref))
      du.has_exec_masked_vector_def = true;
  }

  [[nodiscard]] static bool is_vop3_carry_in(std::string_view mnemonic) {
    return mnemonic == "v_add_co_ci_u32" || mnemonic == "v_sub_co_ci_u32" ||
           mnemonic == "v_subrev_co_ci_u32";
  }

  [[nodiscard]] static bool add_adjusted_destination_def(InstDefUse &du, const Instruction &inst,
                                                         const Operand &op, int operand_index,
                                                         const VgprMsbState &state,
                                                         rj_code_arch_t arch,
                                                         uint32_t wavefront_size) {
    if (auto ref = wave_mode_destination_ref(inst, op, operand_index, wavefront_size)) {
      add_def(du, *ref, op, state, arch);
      return true;
    }
    return false;
  }

  [[nodiscard]] static bool has_enabled_scalar_address(const Instruction &inst) {
    const std::string_view mnemonic = inst.mnemonic();
    if (!starts_with(mnemonic, "global_") && !starts_with(mnemonic, "scratch_"))
      return false;

    for (int i = 1; i < inst.num_src_operands(); ++i) {
      const Operand *op = inst.src_operand(i);
      if (!op)
        continue;
      if (auto ref = op->to_register_ref(); ref && ref->cls == RegClass::SGPR)
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool add_adjusted_source_use(InstDefUse &du, const Instruction &inst,
                                                    const Operand &op, int operand_index,
                                                    const VgprMsbState &state, rj_code_arch_t arch,
                                                    uint32_t wavefront_size) {
    const std::string_view mnemonic = inst.mnemonic();
    const bool is_vop3_mask =
        (starts_with(mnemonic, "v_cndmask_b32") || starts_with(mnemonic, "v_cndmask_b16")) &&
        operand_index == 2;
    const bool is_vopd_mask = mnemonic.find("v_dual_cndmask_b32") != std::string_view::npos &&
                              op.vgpr_msb_role() == amdgpu::VgprMsbRole::Src2;
    const bool is_vop3_carry_mask = operand_index == 2 && is_vop3_carry_in(mnemonic);
    if (is_vop3_mask || is_vopd_mask || is_vop3_carry_mask) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::SGPR) {
        // LLVM's BoolRC predicate is one physical SGPR in Wave32 and a pair in
        // Wave64. VOP3/VOPD keep the maximum-width encoded operand in the
        // generated decoder, just like VOPC destinations above.
        ref->width = wavefront_size == 32 ? 1 : 2;
        du.uses.expand(*ref);
        return true;
      }
    }
    if (operand_index == 0 && starts_with(mnemonic, "scratch_") && op.encoding_value() == 0)
      return true;

    if (operand_index == 0 && starts_with(mnemonic, "s_buffer_")) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::SGPR) {
        ref->width = 4;
        du.uses.expand(*ref);
      }
      return true;
    }

    if (operand_index == 0 && has_enabled_scalar_address(inst)) {
      if (auto ref = op.to_register_ref(); ref && ref->cls == RegClass::VGPR) {
        ref->width = 1;
        expand_vgpr_msb_ref(du.uses, *ref, op, state, arch);
        return true;
      }
    }

    return false;
  }

  [[nodiscard]] static bool is_cdna4_mubuf_lds_load(const Instruction &inst, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4 || !starts_with(inst.mnemonic(), "buffer_load") ||
        inst.raw_encoding() == nullptr || inst.size() < 2 * static_cast<int>(sizeof(uint32_t)))
      return false;

    constexpr uint32_t kCdna4MubufLdsBit = 1u << 16u;
    return (inst.raw_encoding()[0] & kCdna4MubufLdsBit) != 0;
  }

  [[nodiscard]] static InstDefUse inst_def_use_for_waitcheck(const Instruction &inst,
                                                             const VgprMsbState &state,
                                                             rj_code_arch_t arch,
                                                             uint32_t wavefront_size) {
    InstDefUse du(inst);
    du.defs = {};
    du.uses = {};
    du.has_exec_masked_vector_def = false;
    du.has_predicated_def = inst.flags() & PREDICATED_DEF;

    if (!is_cdna4_mubuf_lds_load(inst, arch)) {
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        const Operand *op = inst.dst_operand(i);
        if (op == nullptr)
          continue;
        if (add_adjusted_destination_def(du, inst, *op, i, state, arch, wavefront_size))
          continue;
        if (auto ref = op->to_register_ref())
          add_def(du, *ref, *op, state, arch);
      }
    }
    inst.implicit_defs(du.defs);

    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const Operand *op = inst.src_operand(i);
      if (op == nullptr)
        continue;
      if (add_adjusted_source_use(du, inst, *op, i, state, arch, wavefront_size))
        continue;
      if (auto ref = op->to_register_ref())
        expand_vgpr_msb_ref(du.uses, *ref, *op, state, arch);
    }
    RegisterSet implicit_uses;
    inst.implicit_uses(implicit_uses);
    // D16 memory loads preserve the opposite half for liveness, but the two
    // asynchronous half writes do not consume one another at issue time.
    if (const auto partial = partial_d16_load_def(inst, arch))
      implicit_uses.erase(partial->reg);
    du.uses |= implicit_uses;
    return du;
  }

  [[nodiscard]] static std::optional<uint32_t> operand_u32_immediate(const Operand *operand) {
    const auto value = parse_operand_immediate(operand);
    if (!value || *value < std::numeric_limits<int32_t>::min() ||
        *value > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
    return static_cast<uint32_t>(*value);
  }

  [[nodiscard]] static std::optional<LdsInterval>
  cdna4_dtl_interval(const PendingState &state, const Instruction &inst, uint32_t wavefront_size) {
    constexpr uint64_t kBytesPerLane = 16;
    if (inst.mnemonic() != "buffer_load_dwordx4" || wavefront_size != 64 ||
        !state.lds_constants.m0) {
      return std::nullopt;
    }

    const uint64_t begin = *state.lds_constants.m0;
    const uint64_t end = begin + wavefront_size * kBytesPerLane;
    if (end > uint64_t{1} << 32u)
      return std::nullopt;
    return LdsInterval{begin, end};
  }

  void update_lds_constant_state(PendingState &state, const Instruction &inst, const InstDefUse &du,
                                 rj_code_arch_t arch) const {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4 || !dtl_straight_line_model_) {
      state.lds_constants = {};
      return;
    }

    du.defs.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::VGPR)
        state.lds_constants.uniform_vgprs.erase(ref.index);
    });

    bool defines_m0 = false;
    for (int index = 0; index < inst.num_dst_operands(); ++index) {
      const Operand *operand = inst.dst_operand(index);
      const auto ref = operand == nullptr ? std::nullopt : operand->to_register_ref();
      defines_m0 = defines_m0 || (ref && ref->cls == RegClass::M0);
    }
    if (defines_m0)
      state.lds_constants.m0.reset();

    if (inst.mnemonic() == "s_mov_b32") {
      const Operand *destination = inst.dst_operand(0);
      const auto destination_ref =
          destination == nullptr ? std::nullopt : destination->to_register_ref();
      if (destination_ref && destination_ref->cls == RegClass::M0) {
        state.lds_constants.m0 = operand_u32_immediate(inst.src_operand(0));
      }
    }

    if (starts_with(inst.mnemonic(), "v_mov_b32")) {
      const Operand *destination = inst.dst_operand(0);
      const auto destination_ref =
          destination == nullptr ? std::nullopt : destination->to_register_ref();
      const auto value = operand_u32_immediate(inst.src_operand(0));
      if (destination_ref && destination_ref->cls == RegClass::VGPR &&
          destination_ref->width == 1 && value) {
        state.lds_constants.uniform_vgprs.insert_or_assign(destination_ref->index, *value);
      }
    }
  }

  static void clear_matching_barrier_scc_write(PendingState &state, const Instruction &inst) {
    if (inst.mnemonic() != "s_barrier_wait")
      return;
    const auto barrier_id = first_barrier_id(inst);
    if (!barrier_id)
      return;

    auto &events = state.pending[counter_index(WaitCounterKind::Km)];
    std::erase_if(events, [&](const PendingEvent &event) {
      return event.kind == WaitEventKind::SccWrite && event.barrier_id == barrier_id;
    });
  }

  [[nodiscard]] static std::optional<uint32_t> vinterp_wait_exp(const Instruction &inst) {
    if (!starts_with(inst.mnemonic(), "v_interp_") || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;
    return (inst.raw_encoding()[0] >> 8u) & 0x7u;
  }

  [[nodiscard]] static bool is_dsdir(std::string_view mnemonic) {
    return mnemonic == "ds_param_load" || mnemonic == "ds_direct_load";
  }

  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_va_vdst(const Instruction &inst) {
    if (!is_dsdir(inst.mnemonic()) || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;
    return (inst.raw_encoding()[0] >> 16u) & 0xFu;
  }

  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_vm_vsrc(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    if (!is_dsdir(mnemonic) || inst.raw_encoding() == nullptr ||
        inst.size() < static_cast<int>(sizeof(uint32_t)))
      return std::nullopt;
    return (inst.raw_encoding()[0] >> 23u) & 0x1u;
  }

  static void apply_embedded_waitcnt(PendingState &state, const Instruction &inst,
                                     rj_code_arch_t arch) {
    if (const auto wait_exp = vinterp_wait_exp(inst))
      apply_counter_wait(state, WaitCounterKind::Exp, *wait_exp, arch);
    if (const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst); wait_vm_vsrc && *wait_vm_vsrc == 0)
      apply_counter_wait(state, WaitCounterKind::VmVsrc, 0, arch);
  }

  [[nodiscard]] static bool is_scalar_memory_op(std::string_view mnemonic) {
    return starts_with(mnemonic, "s_load") || starts_with(mnemonic, "s_buffer_load") ||
           starts_with(mnemonic, "s_store") || starts_with(mnemonic, "s_buffer_store") ||
           starts_with(mnemonic, "s_prefetch_") || starts_with(mnemonic, "s_atc_probe") ||
           mnemonic == "s_dcache_inv";
  }

  [[nodiscard]] static bool is_nonflat_vmem_op(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_") || starts_with(mnemonic, "buffer_") ||
           starts_with(mnemonic, "tbuffer_") || starts_with(mnemonic, "image_");
  }

  static void apply_sgpr_hazard_memory_cull(SgprHazardState &state, std::string_view mnemonic) {
    if (!is_scalar_memory_op(mnemonic) && !is_nonflat_vmem_op(mnemonic))
      return;
    clear_all_sgpr_hazards(state);
  }

  [[nodiscard]] static bool apply_sgpr_hazard_ds_nop_cull(SgprHazardState &state,
                                                          std::string_view mnemonic) {
    constexpr uint8_t kWave32DsNopCullCount = 4;
    if (mnemonic != "ds_nop") {
      state.consecutive_ds_nops = 0;
      return false;
    }

    if (state.consecutive_ds_nops < kWave32DsNopCullCount)
      ++state.consecutive_ds_nops;
    if (state.consecutive_ds_nops >= kWave32DsNopCullCount) {
      state.tracked_pairs.reset();
      state.tracked_vcc = false;
    }
    return true;
  }

  [[nodiscard]] static bool is_scalar_alu(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "s_") && !inst.is_waitcnt() && !inst.is_barrier() &&
           !inst.is_branch() && !is_scalar_memory_op(mnemonic);
  }

  [[nodiscard]] static bool instruction_has_register_operand(const Instruction &inst) {
    if (inst.has_implicit_register_operand())
      return true;
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *operand = inst.dst_operand(i);
      if (operand != nullptr && operand->is_register())
        return true;
    }
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      const Operand *operand = inst.src_operand(i);
      if (operand != nullptr && operand->is_register())
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool is_sgpr_hazard_control_transfer(const Instruction &inst) {
    constexpr uint64_t kControlTransferFlags = INDIRECT_CALL | INDIRECT_BRANCH;
    return (inst.flags() & kControlTransferFlags) != 0;
  }

  [[nodiscard]] static bool instruction_waits_for_valu_sgpr_writes(const Instruction &inst) {
    // Scalar memory is handled by apply_sgpr_hazard_memory_cull(). For SALU,
    // model the issue stall used by LLVM's
    // AMDGPUInsertDelayAlu::instructionWaitsForSGPRWrites: any explicit or
    // implicit register operand waits for VA_SDST==0, even when unrelated to
    // the outstanding destination. This deliberately differs from the more
    // conservative data-dependency repair in AMDGPUWaitSGPRHazards; see
    // llvm-project#131111 and llvm-project#145728.
    return !is_scalar_memory_op(inst.mnemonic()) && starts_with(inst.mnemonic(), "s_") &&
           instruction_has_register_operand(inst);
  }

  [[nodiscard]] static bool is_vector_alu(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "v_") && !inst.is_memory_op();
  }

  [[nodiscard]] static bool is_trans_valu(const Instruction &inst) {
    const auto mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "v_exp_") || starts_with(mnemonic, "v_log_") ||
           starts_with(mnemonic, "v_rcp_") || starts_with(mnemonic, "v_rsq_") ||
           starts_with(mnemonic, "v_sqrt_") || starts_with(mnemonic, "v_sin_") ||
           starts_with(mnemonic, "v_cos_");
  }

  [[nodiscard]] static bool is_vmem_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_store") || starts_with(mnemonic, "scratch_store") ||
           starts_with(mnemonic, "buffer_store") || starts_with(mnemonic, "tbuffer_store") ||
           starts_with(mnemonic, "image_store");
  }

  [[nodiscard]] static bool is_vmem_atomic(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_atomic") || starts_with(mnemonic, "flat_atomic") ||
           starts_with(mnemonic, "buffer_atomic") || starts_with(mnemonic, "image_atomic");
  }

  [[nodiscard]] static bool is_image_atomic(std::string_view mnemonic) {
    return starts_with(mnemonic, "image_atomic");
  }

  [[nodiscard]] static bool is_gfx1250_smem(std::string_view mnemonic) {
    return is_scalar_memory_op(mnemonic);
  }

  [[nodiscard]] static bool is_scratch_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "scratch_store");
  }

  [[nodiscard]] static bool is_ds_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "ds_store") || starts_with(mnemonic, "ds_write") ||
           starts_with(mnemonic, "ds_cmpstore") || starts_with(mnemonic, "ds_mskor");
  }

  [[nodiscard]] static bool uses_ds_wait_counter(std::string_view mnemonic) {
    // Match LLVM's TII.isDS && TII.usesLGKM_CNT classification. All VDS
    // instructions use DS_CNT except the gfx1250 async-barrier arrive, which
    // uses ASYNC_CNT instead. LDS-direct instructions have separate EXP_CNT
    // semantics below.
    return starts_with(mnemonic, "ds_") && !is_dsdir(mnemonic) &&
           mnemonic != "ds_atomic_async_barrier_arrive_b64";
  }

  [[nodiscard]] static bool is_async_lds_load(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_load_async_to_lds") ||
           starts_with(mnemonic, "cluster_load_async_to_lds");
  }

  [[nodiscard]] static bool is_async_lds_store(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_store_async_from_lds");
  }

  [[nodiscard]] static bool is_tensor_lds_load(std::string_view mnemonic) {
    return mnemonic == "tensor_load_to_lds";
  }

  [[nodiscard]] static bool is_tensor_lds_store(std::string_view mnemonic) {
    return mnemonic == "tensor_store_from_lds";
  }

  [[nodiscard]] static bool is_async_or_tensor_producer(std::string_view mnemonic) {
    return is_async_lds_load(mnemonic) || is_async_lds_store(mnemonic) ||
           mnemonic == "ds_atomic_async_barrier_arrive_b64" || is_tensor_lds_load(mnemonic) ||
           is_tensor_lds_store(mnemonic);
  }

  [[nodiscard]] static bool is_async_ordering_consumer(const PendingEvent &event,
                                                       std::string_view mnemonic) {
    if (event.kind == WaitEventKind::AsyncBarrier)
      return starts_with(mnemonic, "s_barrier_wait");
    if (is_async_or_tensor_producer(mnemonic))
      return false;
    return starts_with(mnemonic, "ds_") || starts_with(mnemonic, "s_barrier_wait");
  }

  [[nodiscard]] static bool is_tensor_ordering_consumer(const PendingEvent &event,
                                                        std::string_view mnemonic) {
    (void)event;
    if (is_async_or_tensor_producer(mnemonic))
      return false;
    return starts_with(mnemonic, "ds_") || starts_with(mnemonic, "s_barrier_wait");
  }

  [[nodiscard]] static bool is_memory_ordering_consumer(std::string_view mnemonic) {
    return starts_with(mnemonic, "global_") || starts_with(mnemonic, "flat_") ||
           starts_with(mnemonic, "scratch_") || starts_with(mnemonic, "buffer_") ||
           starts_with(mnemonic, "tbuffer_") || starts_with(mnemonic, "image_") ||
           starts_with(mnemonic, "ds_") || starts_with(mnemonic, "s_load") ||
           starts_with(mnemonic, "s_buffer_load") || mnemonic == "export";
  }

  [[nodiscard]] static bool is_va_vdst_expiring_instruction(std::string_view mnemonic) {
    const bool vmem = starts_with(mnemonic, "flat_") || starts_with(mnemonic, "global_") ||
                      starts_with(mnemonic, "scratch_") || starts_with(mnemonic, "buffer_") ||
                      starts_with(mnemonic, "tbuffer_") || starts_with(mnemonic, "image_");
    const bool ds = starts_with(mnemonic, "ds_") && !is_dsdir(mnemonic);
    return vmem || ds || mnemonic == "export";
  }

  [[nodiscard]] static bool is_program_end(std::string_view mnemonic) {
    return mnemonic == "s_endpgm" || mnemonic == "s_endpgm_saved";
  }

  [[nodiscard]] static bool is_cdna4_lds_ds_access(const Instruction &inst, rj_code_arch_t arch) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA4 || !starts_with(inst.mnemonic(), "ds_") ||
        is_dsdir(inst.mnemonic()) || inst.raw_encoding() == nullptr ||
        inst.size() < 2 * static_cast<int>(sizeof(uint32_t))) {
      return false;
    }
    constexpr uint32_t kGdsBit = 1u << 16u;
    return (inst.raw_encoding()[0] & kGdsBit) == 0;
  }

  [[nodiscard]] static std::optional<LdsInterval>
  cdna4_ds_read_b128_interval(const PendingState &state, const Instruction &inst,
                              rj_code_arch_t arch) {
    if (!is_cdna4_lds_ds_access(inst, arch) || inst.mnemonic() != "ds_read_b128")
      return std::nullopt;

    const Operand *address = inst.src_operand(0);
    const auto address_ref = address == nullptr ? std::nullopt : address->to_register_ref();
    if (!address_ref || address_ref->cls != RegClass::VGPR || address_ref->width != 1)
      return std::nullopt;
    const auto value = state.lds_constants.uniform_vgprs.find(address_ref->index);
    if (value == state.lds_constants.uniform_vgprs.end())
      return std::nullopt;

    const uint64_t offset = inst.raw_encoding()[0] & 0xffffu;
    const uint64_t begin = static_cast<uint64_t>(value->second) + offset;
    const uint64_t end = begin + 16;
    if (end > uint64_t{1} << 32u)
      return std::nullopt;
    return LdsInterval{begin, end};
  }

  static void apply_dtl_barrier(PendingState &state, uint64_t section_offset) {
    std::erase_if(state.dtl_visibility,
                  [](const DtlVisibilityEvent &event) { return !event.active; });
    for (DtlVisibilityEvent &event : state.dtl_visibility) {
      event.barrier_required_count = event.min_younger;
      event.barrier_section_offset = section_offset;
    }
  }

  void check_dtl_lds_access(const PendingState &state, const Instruction &inst,
                            const std::string &section_name, uint64_t section_offset,
                            uint64_t file_offset, rj_code_arch_t arch) {
    if (state.dtl_visibility.empty() || !is_cdna4_lds_ds_access(inst, arch))
      return;

    const auto consumer_interval = cdna4_ds_read_b128_interval(state, inst, arch);
    if (!consumer_interval) {
      record_incomplete_analysis("direct-to-LDS consumer range is not statically known");
      return;
    }

    for (const DtlVisibilityEvent &event : state.dtl_visibility) {
      if (!event.interval) {
        record_incomplete_analysis("direct-to-LDS producer range is not statically known");
        continue;
      }
      if (!event.interval->overlaps(*consumer_interval))
        continue;
      if (!event.active && !event.barrier_required_count) {
        record_incomplete_analysis(
            "completed direct-to-LDS access requires launch-aware cross-wave visibility analysis");
        continue;
      }

      const uint32_t required_count = event.barrier_required_count.value_or(event.min_younger);
      emit_dtl_diagnostic(inst, event, required_count, section_name, section_offset, file_offset,
                          arch);
    }
  }

  [[nodiscard]] static std::vector<ClassifiedEvent> classify_events(const Instruction &inst,
                                                                    rj_code_arch_t arch) {
    std::vector<ClassifiedEvent> events;
    const bool expert = supports_expert_scheduling(arch);
    const auto mnemonic = inst.mnemonic();
    auto add_xcnt_event = [&](WaitEventKind kind) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA5)
        events.emplace_back(WaitCounterKind::X, kind, TrackedRegisterSource::Uses,
                            /*check_uses=*/false, /*check_defs=*/true,
                            /*check_exec_defs=*/is_xcnt_vmem_kind(kind));
    };
    if (is_async_lds_load(mnemonic) || is_async_lds_store(mnemonic)) {
      const bool is_load = is_async_lds_load(mnemonic);
      const WaitEventKind kind =
          is_load ? WaitEventKind::AsyncLdsLoad : WaitEventKind::AsyncLdsStore;
      // These operations update both their ordinary VMEM counter and the
      // independent async counter in LLVM's hardware-event model.
      events.emplace_back(is_load ? WaitCounterKind::Load : vmem_store_wait_counter(arch),
                          is_load ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore,
                          TrackedRegisterSource::None, /*check_uses=*/false,
                          /*check_defs=*/false);
      events.emplace_back(WaitCounterKind::Async, kind, TrackedRegisterSource::None,
                          /*check_uses=*/false, /*check_defs=*/false,
                          /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                          /*check_memory_order=*/true, /*check_program_end=*/false);
      // LLVM excludes ASYNC_CNT/TENSOR_CNT from blanket function-boundary
      // waits. Their waits are derived from object-invisible ASYNCMARK pairs,
      // so conservatively check observable LDS consumers but not program end.
      add_xcnt_event(is_load ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore);
      return events;
    }

    if (mnemonic == "ds_atomic_async_barrier_arrive_b64") {
      events.emplace_back(WaitCounterKind::Async, WaitEventKind::AsyncBarrier,
                          TrackedRegisterSource::None, /*check_uses=*/false,
                          /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt,
                          std::nullopt, /*check_memory_order=*/true,
                          /*check_program_end=*/false);
      return events;
    }

    if (is_tensor_lds_load(mnemonic) || is_tensor_lds_store(mnemonic)) {
      const bool is_load = is_tensor_lds_load(mnemonic);
      const WaitEventKind kind =
          is_load ? WaitEventKind::TensorLdsLoad : WaitEventKind::TensorLdsStore;
      events.emplace_back(WaitCounterKind::Tensor, kind, TrackedRegisterSource::None,
                          /*check_uses=*/false, /*check_defs=*/false,
                          /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                          /*check_memory_order=*/true, /*check_program_end=*/false);
      // AMDGPU::getEventsFor classifies tensor operations solely as
      // TENSOR_ACCESS, before the generic VMEM/X_CNT path.
      return events;
    }

    if (starts_with(mnemonic, "flat_load")) {
      events.push_back({WaitCounterKind::Load, WaitEventKind::FlatLoad});
      // Keep the instruction kind distinct: generic FLAT and native DS can
      // complete out of order even though both contribute to the DS counter.
      events.push_back({WaitCounterKind::Ds, WaitEventKind::FlatLoad});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatLoad,
                          TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(WaitEventKind::FlatLoad);
      return events;
    }

    if (mnemonic == "global_inv") {
      // LLVM tracks this in LOAD_CNT so it changes the wait threshold for an
      // older load.  It has no result and does not by itself make the next
      // memory instruction a wait consumer.
      events.emplace_back(WaitCounterKind::Load, WaitEventKind::GlobalInv,
                          TrackedRegisterSource::None, false, false);
      return events;
    }

    if (mnemonic == "global_wb" || mnemonic == "global_wbinv") {
      events.emplace_back(vmem_store_wait_counter(arch), WaitEventKind::GlobalWb,
                          TrackedRegisterSource::None, false, false);
      return events;
    }

    if (is_cdna4_mubuf_lds_load(inst, arch)) {
      events.emplace_back(WaitCounterKind::Load, WaitEventKind::LdsDirect,
                          TrackedRegisterSource::None, /*check_uses=*/false,
                          /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt,
                          std::nullopt, /*check_memory_order=*/false,
                          /*check_program_end=*/false,
                          /*check_counter_parity_order=*/true);
      return events;
    }

    if (starts_with(mnemonic, "global_load") || starts_with(mnemonic, "scratch_load") ||
        starts_with(mnemonic, "buffer_load") || starts_with(mnemonic, "tbuffer_load") ||
        starts_with(mnemonic, "image_load")) {
      events.push_back({WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemNoSamplerLoad,
                          TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(WaitEventKind::VmemNoSamplerLoad);
      return events;
    }

    if (is_image_atomic(mnemonic)) {
      const bool returns_value = inst.num_dst_operands() != 0;
      const WaitCounterKind counter =
          returns_value ? WaitCounterKind::Load : vmem_store_wait_counter(arch);
      const WaitEventKind kind =
          returns_value ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore;
      events.emplace_back(counter, kind,
                          returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                          /*check_uses=*/returns_value, /*check_defs=*/returns_value);
      if (!uses_legacy_waitcnt(arch))
        events.push_back({WaitCounterKind::Exp, WaitEventKind::VmemStore,
                          TrackedRegisterSource::StoreDataUses, false, true});
      if (expert)
        events.push_back(
            {WaitCounterKind::VmVsrc, kind, TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(kind);
      return events;
    }

    if (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_vmem_atomic(mnemonic)) {
      const bool returns_value = inst.num_dst_operands() != 0;
      const WaitCounterKind counter =
          returns_value ? WaitCounterKind::Load : vmem_store_wait_counter(arch);
      const WaitEventKind kind =
          returns_value ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore;
      events.emplace_back(counter, kind,
                          returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                          /*check_uses=*/returns_value, /*check_defs=*/returns_value);
      if (starts_with(mnemonic, "flat_atomic"))
        events.emplace_back(WaitCounterKind::Ds, WaitEventKind::Ds,
                            returns_value ? TrackedRegisterSource::Defs
                                          : TrackedRegisterSource::None,
                            /*check_uses=*/returns_value, /*check_defs=*/returns_value);
      if (expert)
        events.emplace_back(WaitCounterKind::VmVsrc, kind, TrackedRegisterSource::VectorUses,
                            /*check_uses=*/false, /*check_defs=*/true);
      add_xcnt_event(kind);
      return events;
    }

    if (starts_with(mnemonic, "flat_store")) {
      events.push_back({vmem_store_wait_counter(arch), WaitEventKind::FlatStore,
                        TrackedRegisterSource::None, false, false});
      events.push_back(
          {WaitCounterKind::Ds, WaitEventKind::Ds, TrackedRegisterSource::None, false, false});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatStore,
                          TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(WaitEventKind::FlatStore);
      return events;
    }

    if (is_vmem_store(mnemonic)) {
      events.push_back({vmem_store_wait_counter(arch), WaitEventKind::VmemStore,
                        TrackedRegisterSource::None, false, false});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemStore,
                          TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(WaitEventKind::VmemStore);
      return events;
    }

    if (uses_ds_wait_counter(mnemonic)) {
      events.push_back({WaitCounterKind::Ds, WaitEventKind::Ds});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Ds,
                          TrackedRegisterSource::VectorUses, false, true});
      return events;
    }

    if (mnemonic == "ds_param_load" || mnemonic == "ds_direct_load") {
      events.push_back({WaitCounterKind::Exp, WaitEventKind::LdsDirect});
      return events;
    }

    if (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_gfx1250_smem(mnemonic)) {
      const bool produces_result = inst.num_dst_operands() != 0;
      events.emplace_back(smem_wait_counter(arch), WaitEventKind::Smem,
                          produces_result ? TrackedRegisterSource::Defs
                                          : TrackedRegisterSource::None,
                          /*check_uses=*/produces_result, /*check_defs=*/produces_result);
      add_xcnt_event(WaitEventKind::Smem);
      return events;
    }

    if (starts_with(mnemonic, "s_load") || starts_with(mnemonic, "s_buffer_load")) {
      events.push_back({smem_wait_counter(arch), WaitEventKind::Smem});
      add_xcnt_event(WaitEventKind::Smem);
      return events;
    }

    if (starts_with(mnemonic, "s_store") || starts_with(mnemonic, "s_buffer_store")) {
      events.emplace_back(smem_wait_counter(arch), WaitEventKind::Smem, TrackedRegisterSource::None,
                          /*check_uses=*/false,
                          /*check_defs=*/false);
      add_xcnt_event(WaitEventKind::Smem);
      return events;
    }

    if (mnemonic == "s_sendmsg_rtn_b32" || mnemonic == "s_sendmsg_rtn_b64") {
      events.push_back({smem_wait_counter(arch), WaitEventKind::SqMessage});
      return events;
    }

    if (mnemonic == "image_msaa_load" || starts_with(mnemonic, "image_sample") ||
        starts_with(mnemonic, "image_gather")) {
      events.push_back({image_sample_wait_counter(arch), WaitEventKind::Sample});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Sample,
                          TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(WaitEventKind::Sample);
      return events;
    }

    if (starts_with(mnemonic, "image_bvh")) {
      events.push_back({image_bvh_wait_counter(arch), WaitEventKind::Bvh});
      if (expert)
        events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Bvh,
                          TrackedRegisterSource::VectorUses, false, true});
      add_xcnt_event(WaitEventKind::Bvh);
      return events;
    }

    if (mnemonic == "export") {
      events.push_back({WaitCounterKind::Exp, WaitEventKind::Export, TrackedRegisterSource::Uses,
                        false, true, true});
      return events;
    }

    if (mnemonic == "s_barrier_signal_isfirst") {
      if (!uses_legacy_waitcnt(arch))
        events.push_back({WaitCounterKind::Km, WaitEventKind::SccWrite, TrackedRegisterSource::None,
                          true, true, false, RegisterRef{RegClass::SCC, 0, 1},
                          first_barrier_id(inst)});
      return events;
    }

    return events;
  }

  [[nodiscard]] static bool tracks_d16_subregisters(rj_code_arch_t arch) {
    // GFX11 uses LLVM's VGPR_LO16/VGPR_HI16 register units for memory
    // operations. The D16Writes32BitVgpr feature still makes VALU and some
    // mixed-event cases conservative, but ordinary memory operations on
    // disjoint halves remain independent.
    return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
  }

  [[nodiscard]] static std::optional<PartialRegisterAccess>
  partial_d16_load_def(const Instruction &inst, rj_code_arch_t arch) {
    if (!tracks_d16_subregisters(arch) || !inst.is_memory_op())
      return std::nullopt;

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic.find("load") == std::string_view::npos ||
        mnemonic.find("d16") == std::string_view::npos)
      return std::nullopt;

    const Operand *dst = inst.dst_operand(0);
    if (!dst)
      return std::nullopt;
    auto ref = dst->to_register_ref();
    if (!ref || ref->cls != RegClass::VGPR || ref->width != 1)
      return std::nullopt;

    const uint8_t mask =
        mnemonic.find("d16_hi") != std::string_view::npos ? kVgprHigh16Mask : kVgprLow16Mask;
    return PartialRegisterAccess{*ref, mask};
  }

  [[nodiscard]] static bool is_low_half_memory_store(std::string_view mnemonic) {
    return mnemonic.find("store_b8") != std::string_view::npos ||
           mnemonic.find("store_b16") != std::string_view::npos ||
           mnemonic.find("store_byte") != std::string_view::npos ||
           mnemonic.find("store_short") != std::string_view::npos ||
           mnemonic.find("write_b8") != std::string_view::npos ||
           mnemonic.find("write_b16") != std::string_view::npos;
  }

  [[nodiscard]] static std::optional<PartialRegisterAccess>
  partial_memory_store_use(const Instruction &inst, rj_code_arch_t arch) {
    if (!tracks_d16_subregisters(arch) || !inst.is_memory_op())
      return std::nullopt;

    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic.find("store") == std::string_view::npos &&
        mnemonic.find("write") == std::string_view::npos)
      return std::nullopt;

    uint8_t mask = kVgprFull32Mask;
    if (mnemonic.find("d16_hi") != std::string_view::npos)
      mask = kVgprHigh16Mask;
    else if (is_low_half_memory_store(mnemonic))
      mask = kVgprLow16Mask;
    else
      return std::nullopt;

    const int data_operand_index = starts_with(mnemonic, "buffer_store") ||
                                           starts_with(mnemonic, "tbuffer_store") ||
                                           starts_with(mnemonic, "image_store")
                                       ? 0
                                       : 1;
    const Operand *data = inst.src_operand(data_operand_index);
    if (!data)
      return std::nullopt;
    auto ref = data->to_register_ref();
    if (!ref || ref->cls != RegClass::VGPR || ref->width != 1)
      return std::nullopt;
    return PartialRegisterAccess{*ref, mask};
  }

  [[nodiscard]] static uint8_t pending_register_mask(const PendingEvent &event, RegisterRef ref) {
    return event.partial_reg && *event.partial_reg == ref ? event.partial_reg_mask
                                                          : kVgprFull32Mask;
  }

  [[nodiscard]] static uint8_t instruction_register_mask(const Instruction &inst, RegisterRef ref,
                                                         WaitcheckAccessKind access,
                                                         rj_code_arch_t arch) {
    const auto partial = access == WaitcheckAccessKind::Def   ? partial_d16_load_def(inst, arch)
                         : access == WaitcheckAccessKind::Use ? partial_memory_store_use(inst, arch)
                                                              : std::nullopt;
    return partial && partial->reg == ref ? partial->mask : kVgprFull32Mask;
  }

  [[nodiscard]] static std::optional<RegisterRef>
  first_dependency_intersection(const PendingEvent &event, const RegisterSet &pending_regs,
                                const Instruction &inst, const RegisterSet &current_regs,
                                WaitcheckAccessKind access, rj_code_arch_t arch) {
    std::optional<RegisterRef> result;
    pending_regs.for_each([&](RegisterRef ref) {
      if (!result && current_regs.contains(ref) &&
          (pending_register_mask(event, ref) &
           instruction_register_mask(inst, ref, access, arch)) != 0)
        result = ref;
    });
    return result;
  }

  [[nodiscard]] static std::optional<RegisterRef> first_intersection(const RegisterSet &lhs,
                                                                     const RegisterSet &rhs) {
    std::optional<RegisterRef> result;
    lhs.for_each([&](RegisterRef ref) {
      if (!result && rhs.contains(ref))
        result = ref;
    });
    return result;
  }

  static void apply_implicit_xcnt_ordering(PendingState &state, const InstDefUse &du,
                                           std::span<const ClassifiedEvent> current_events) {
    const bool current_vmem = std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
      return event.counter == WaitCounterKind::X && is_xcnt_vmem_kind(event.kind);
    });
    const bool current_smem = std::ranges::any_of(current_events, [](const ClassifiedEvent &event) {
      return event.counter == WaitCounterKind::X && event.kind == WaitEventKind::Smem;
    });
    if (!current_vmem && !current_smem)
      return;

    const size_t idx = counter_index(WaitCounterKind::X);
    const auto &pending = state.pending[idx];
    const bool pending_vmem = std::ranges::any_of(pending, is_xcnt_vmem_event);
    const bool pending_smem = std::ranges::any_of(pending, is_xcnt_smem_event);

    // Hardware places an implicit X_CNT drain between interleaved SMEM and
    // VMEM translations. LLVM models the switch by retaining only the new
    // group's event.
    if ((current_vmem && pending_smem) || (current_smem && pending_vmem)) {
      apply_xcnt_wait(state, 0);
      return;
    }

    if (!current_vmem)
      return;

    // VMEM address translations are ordered. If this VMEM instruction would
    // overwrite a source of an older VMEM operation, LLVM internally applies
    // just the X_CNT wait required for that dependency instead of emitting an
    // instruction. Preserve younger, unrelated translations.
    std::optional<uint32_t> required_count;
    for (const PendingEvent &event : state.pending[idx]) {
      if (!is_xcnt_vmem_event(event) || !first_intersection(event.regs, du.defs))
        continue;
      required_count = required_count ? std::min(*required_count, event.min_younger)
                                      : std::optional<uint32_t>(event.min_younger);
    }
    if (required_count)
      apply_xcnt_wait(state, *required_count);
  }

  [[nodiscard]] static bool
  creates_immediate_overlay_generation(rj_code_arch_t arch, const Instruction &inst,
                                       std::span<const ClassifiedEvent> current_events) {
    if (!tracks_committed_vgpr_generations(arch) || inst.is_memory_op())
      return false;
    // CDNA software pipelines can schedule ordinary VALU and AccVGPR transfers
    // into the currently visible generation while an asynchronous replacement
    // of the same VGPR is pending. A memory instruction creates another pending
    // generation and therefore does not qualify for this synchronous-overlay
    // exception.
    return std::ranges::none_of(current_events, [](const ClassifiedEvent &event) {
      return event.registers == TrackedRegisterSource::Defs;
    });
  }

  [[nodiscard]] static std::string reg_name(RegisterRef ref) {
    const char *prefix = "?";
    switch (ref.cls) {
    case RegClass::SGPR:
      prefix = "s";
      break;
    case RegClass::VGPR:
      prefix = "v";
      break;
    case RegClass::ACC_VGPR:
      prefix = "acc";
      break;
    case RegClass::EXEC:
      return "exec";
    case RegClass::VCC:
      return "vcc";
    case RegClass::SCC:
      return "scc";
    case RegClass::PC:
      return "memory operation";
    default:
      break;
    }
    if (ref.width <= 1)
      return std::string(prefix) + std::to_string(ref.index);
    return std::string(prefix) + "[" + std::to_string(ref.index) + ":" +
           std::to_string(ref.index + ref.width - 1) + "]";
  }

  void emit_diagnostic(const Instruction &inst, const PendingEvent &event, RegisterRef reg,
                       WaitcheckAccessKind access, uint32_t required_count, uint64_t section_offset,
                       uint64_t file_offset, rj_code_arch_t arch) {
    if (!should_emit_cfg_diagnostic(event, section_offset, arch))
      return;

    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::WaitCounter;
    diag.counter = event.counter;
    diag.access = access;
    diag.reg = reg;
    diag.section_name = event.section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = event.section_offset;
    diag.producer_file_offset = event.file_offset;
    diag.producer_instruction = event.instruction;
    diag.required_count = required_count;

    std::ostringstream msg;
    msg << "missing " << wait_expression(event.counter, required_count, arch) << " before ";
    if (access == WaitcheckAccessKind::MemoryOrder) {
      msg << "memory operation";
    } else if (access == WaitcheckAccessKind::ProgramEnd) {
      msg << "program end";
    } else {
      msg << (access == WaitcheckAccessKind::Use ? "use" : "def") << " of " << reg_name(reg);
    }
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void emit_dtl_diagnostic(const Instruction &inst, const DtlVisibilityEvent &event,
                           uint32_t required_count, const std::string &section_name,
                           uint64_t section_offset, uint64_t file_offset, rj_code_arch_t arch) {
    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::WaitCounter;
    diag.counter = WaitCounterKind::Load;
    diag.access = WaitcheckAccessKind::MemoryOrder;
    diag.reg = RegisterRef{RegClass::PC, 0, 1};
    diag.section_name = section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = event.section_offset;
    diag.producer_file_offset = event.file_offset;
    diag.producer_instruction = event.instruction;
    diag.required_count = required_count;

    std::ostringstream message;
    message << "missing " << wait_expression(WaitCounterKind::Load, required_count, arch);
    if (event.barrier_required_count) {
      message << " before workgroup barrier at " << section_name << "+0x" << std::hex
              << event.barrier_section_offset << std::dec;
    }
    message << " for LDS access overlapping direct-to-LDS data";
    diag.message = message.str();
    record_diagnostic(std::move(diag));
  }

  void emit_sgpr_hazard_diagnostic(const Instruction &inst, const SgprHazardProducer *producer,
                                   RegisterRef reg, std::string_view depctr_field,
                                   uint64_t section_offset, uint64_t file_offset,
                                   WaitcheckAccessKind access = WaitcheckAccessKind::Use) {
    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::SgprDepctr;
    diag.counter = WaitCounterKind::Depctr;
    diag.access = access;
    diag.reg = reg;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.required_count = 0;
    if (producer) {
      diag.section_name = producer->section_name;
      diag.producer_section_offset = producer->section_offset;
      diag.producer_file_offset = producer->file_offset;
      diag.producer_instruction = producer->instruction;
    }

    std::ostringstream msg;
    msg << "missing s_wait_alu " << depctr_field << "(0) before ";
    if (access == WaitcheckAccessKind::ControlTransfer)
      msg << "control transfer with outstanding " << reg_name(reg);
    else
      msg << "use of " << reg_name(reg);
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void emit_async_barrier_pipe_diagnostic(const Instruction &inst,
                                          const SgprHazardProducer &barrier,
                                          bool missing_after_barrier, uint64_t section_offset,
                                          uint64_t file_offset) {
    WaitcheckDiagnostic diag;
    diag.kind = missing_after_barrier ? WaitcheckDiagnosticKind::AsyncBarrierPostWait
                                      : WaitcheckDiagnosticKind::AsyncBarrierPreWait;
    diag.counter = WaitCounterKind::VmVsrc;
    diag.access = WaitcheckAccessKind::MemoryOrder;
    diag.reg = RegisterRef{RegClass::PC, 0, 1};
    diag.section_name = barrier.section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = barrier.section_offset;
    diag.producer_file_offset = barrier.file_offset;
    diag.producer_instruction = barrier.instruction;
    diag.required_count = 0;
    diag.message = std::string("missing s_wait_alu depctr_vm_vsrc(0) immediately ") +
                   (missing_after_barrier ? "after" : "before") +
                   " ds_atomic_async_barrier_arrive_b64";
    record_diagnostic(std::move(diag));
  }

  void emit_va_vdst_hazard_diagnostic(const Instruction &inst, const VaVdstHazard &hazard,
                                      RegisterRef reg, uint32_t encoded_wait,
                                      uint32_t required_wait, uint64_t section_offset,
                                      uint64_t file_offset) {
    WaitcheckDiagnostic diag;
    diag.kind = WaitcheckDiagnosticKind::VaVdst;
    diag.counter = WaitCounterKind::VaVdst;
    diag.access = WaitcheckAccessKind::Def;
    diag.reg = reg;
    diag.section_name = hazard.producer.section_name;
    diag.section_offset = section_offset;
    diag.file_offset = file_offset;
    diag.instruction = inst.disassemble();
    diag.producer_section_offset = hazard.producer.section_offset;
    diag.producer_file_offset = hazard.producer.file_offset;
    diag.producer_instruction = hazard.producer.instruction;
    diag.required_count = required_wait;

    std::ostringstream msg;
    msg << "missing wait_va_vdst <= " << required_wait << " before def of " << reg_name(reg)
        << " (encoded wait_va_vdst=" << encoded_wait << ")";
    diag.message = msg.str();
    record_diagnostic(std::move(diag));
  }

  void check_va_vdst_hazard(const VaVdstHazardState &state, const Instruction &inst,
                            const InstDefUse &du, uint64_t section_offset, uint64_t file_offset) {
    const auto encoded_wait = dsdir_wait_va_vdst(inst);
    if (!encoded_wait)
      return;

    bool emitted = false;
    du.defs.for_each([&](RegisterRef ref) {
      if (emitted || ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
        return;
      const auto hazard_it = state.hazards.find(ref.index);
      if (hazard_it == state.hazards.end())
        return;
      const VaVdstHazard &hazard = hazard_it->second;

      const uint32_t required_wait = hazard.trans_since ? 0u : hazard.age;
      if (*encoded_wait <= required_wait)
        return;

      emit_va_vdst_hazard_diagnostic(inst, hazard, ref, *encoded_wait, required_wait,
                                     section_offset, file_offset);
      emitted = true;
    });
  }

  static void age_va_vdst_hazards(VaVdstHazardState &state, bool trans_seen) {
    constexpr uint8_t kNoHazardWaitStates = 15;
    for (auto it = state.hazards.begin(); it != state.hazards.end();) {
      VaVdstHazard &hazard = it->second;
      if (hazard.age < kNoHazardWaitStates)
        ++hazard.age;
      if (hazard.age >= kNoHazardWaitStates) {
        it = state.hazards.erase(it);
        continue;
      }
      hazard.trans_since = hazard.trans_since || trans_seen;
      ++it;
    }
  }

  static void clear_va_vdst_hazards(VaVdstHazardState &state) { state.hazards.clear(); }

  static void set_va_vdst_hazard_for_regs(VaVdstHazardState &state, const RegisterSet &regs,
                                          bool trans_seen, const SgprHazardProducer &producer) {
    regs.for_each([&](RegisterRef ref) {
      if (ref.cls != RegClass::VGPR || ref.index >= REGISTER_SET_MAX_VGPRS)
        return;
      state.hazards.insert_or_assign(ref.index, VaVdstHazard{0, trans_seen, producer});
    });
  }

  static void update_va_vdst_hazards(VaVdstHazardState &state, const Instruction &inst,
                                     const InstDefUse &du, const std::string &section_name,
                                     uint64_t section_offset, uint64_t file_offset) {
    if (is_va_vdst_expiring_instruction(inst.mnemonic())) {
      clear_va_vdst_hazards(state);
      return;
    }
    if (!is_vector_alu(inst))
      return;

    const bool trans_seen = is_trans_valu(inst);
    age_va_vdst_hazards(state, trans_seen);

    const SgprHazardProducer producer{section_name, section_offset, file_offset,
                                      inst.disassemble()};
    set_va_vdst_hazard_for_regs(state, du.uses, trans_seen, producer);
    set_va_vdst_hazard_for_regs(state, du.defs, trans_seen, producer);
  }

  [[nodiscard]] static bool ordered_waw(const PendingEvent &event,
                                        std::span<const ClassifiedEvent> current_events) {
    // Match LLVM SIInsertWaitcnts' asymmetric generic-FLAT WAW rule.  Once a
    // wait has retired the LDS-counter facet of an older FLAT load, its
    // remaining VMEM result is ordered with a newer non-sampler VMEM load.
    // The reverse is not true: a newer generic FLAT load may complete through
    // LDS and overwrite an older VMEM result out of order.
    if (event.kind == WaitEventKind::FlatLoad) {
      if (event.counter != WaitCounterKind::Load)
        return false;
      return std::ranges::any_of(current_events, [&](const ClassifiedEvent &current_event) {
        if (current_event.counter != WaitCounterKind::Load)
          return false;
        // Generic FLAT is deliberately excluded here because it is not a
        // VMEM-only instruction.
        return current_event.kind == WaitEventKind::VmemNoSamplerLoad;
      });
    }

    switch (event.kind) {
    case WaitEventKind::VmemNoSamplerLoad:
    case WaitEventKind::Sample:
    case WaitEventKind::Bvh:
      break;
    default:
      return false;
    }
    return std::ranges::any_of(current_events, [&](const ClassifiedEvent &current_event) {
      return event.kind == current_event.kind;
    });
  }

  using Reachability = std::vector<uint8_t>;
  using SharedReachability = std::shared_ptr<const Reachability>;

  template <typename Cache>
  [[nodiscard]] SharedReachability cache_reachability(Cache &cache, size_t block_index,
                                                      Reachability reachable) {
    const size_t bytes = reachable.size() * sizeof(Reachability::value_type);
    auto result = std::make_shared<const Reachability>(std::move(reachable));
    if (bytes <= options_.max_reachability_cache_bytes -
                     std::min(reachability_cache_bytes_, options_.max_reachability_cache_bytes)) {
      reachability_cache_bytes_ += bytes;
      cache.emplace(block_index, result);
    }
    return result;
  }

  [[nodiscard]] SharedReachability blocks_reaching(size_t target_block_index) {
    if (auto cached = reverse_reachability_cache_.find(target_block_index);
        cached != reverse_reachability_cache_.end())
      return cached->second;

    Reachability reachable(cfg_views_.size());
    std::vector<size_t> worklist{target_block_index};
    while (!worklist.empty()) {
      const size_t block_index = worklist.back();
      worklist.pop_back();
      if (reachable[block_index] != 0)
        continue;
      reachable[block_index] = 1;
      for (size_t predecessor : cfg_views_[block_index].predecessors)
        worklist.push_back(predecessor);
    }
    return cache_reachability(reverse_reachability_cache_, target_block_index,
                              std::move(reachable));
  }

  [[nodiscard]] SharedReachability blocks_reachable_from(size_t source_block_index) {
    if (auto cached = forward_reachability_cache_.find(source_block_index);
        cached != forward_reachability_cache_.end())
      return cached->second;

    Reachability reachable(cfg_views_.size());
    std::vector<size_t> worklist{source_block_index};
    while (!worklist.empty()) {
      const size_t block_index = worklist.back();
      worklist.pop_back();
      if (reachable[block_index] != 0)
        continue;
      reachable[block_index] = 1;
      for (size_t successor : cfg_views_[block_index].successors)
        worklist.push_back(successor);
    }
    return cache_reachability(forward_reachability_cache_, source_block_index,
                              std::move(reachable));
  }

  [[nodiscard]] std::vector<size_t> cfg_block_indices_containing(uint64_t section_offset) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if (block != nullptr && section_offset >= block->start_offset() &&
          section_offset < block->end_offset())
        result.push_back(i);
    }
    return result;
  }

  [[nodiscard]] bool instruction_adds_younger_event(const Instruction &inst,
                                                    const PendingEvent &event,
                                                    rj_code_arch_t arch) const {
    const auto events = classify_events(inst, arch);
    return std::ranges::any_of(events, [&](const ClassifiedEvent &classification) {
      return classification.counter == event.counter;
    });
  }

  void apply_path_pre_dependency_waits(const Instruction &inst, const PendingEvent &event,
                                       rj_code_arch_t arch, bool &pending, uint32_t &age) {
    if (!pending)
      return;

    PendingEvent path_event = event;
    path_event.min_younger = age;
    PendingState state;
    state.pending[counter_index(event.counter)].push_back(path_event);

    if (inst.is_waitcnt()) {
      apply_waitcnt(state, inst, arch);
    } else {
      apply_embedded_waitcnt(state, inst, arch);
    }

    auto &events = state.pending[counter_index(event.counter)];
    auto found = find_event(events, path_event);
    if (found == events.end()) {
      pending = false;
      age = 0;
      return;
    }
    age = found->min_younger;
  }

  [[nodiscard]] static std::optional<SccPredicate> scalar_scc_predicate(const Instruction &inst) {
    SccPredicate::Kind kind;
    if (inst.mnemonic() == "s_cmp_eq_u32")
      kind = SccPredicate::Kind::EqImm;
    else if (inst.mnemonic() == "s_cmp_ge_u32")
      kind = SccPredicate::Kind::GeUnsignedImm;
    else
      return std::nullopt;
    const Operand *lhs = inst.src_operand(0);
    const Operand *rhs = inst.src_operand(1);
    if (lhs == nullptr || rhs == nullptr)
      return std::nullopt;

    auto lhs_ref = lhs->to_register_ref();
    auto rhs_imm = parse_operand_immediate(rhs);
    if (!lhs_ref || lhs_ref->cls != RegClass::SGPR || lhs_ref->width != 1 || !rhs_imm ||
        *rhs_imm < std::numeric_limits<int32_t>::min() ||
        *rhs_imm > std::numeric_limits<uint32_t>::max())
      return std::nullopt;

    return SccPredicate{kind, lhs_ref->index, static_cast<uint32_t>(*rhs_imm)};
  }

  [[nodiscard]] static bool is_scc_conditional_branch(std::string_view mnemonic) {
    return mnemonic == "s_cbranch_scc0" || mnemonic == "s_cbranch_scc1";
  }

  [[nodiscard]] static bool
  apply_scc_branch_constraint(ScalarConstraints &constraints,
                              const std::optional<SccPredicate> &predicate,
                              bool branch_condition_true) {
    if (!predicate)
      return true;

    ScalarValueConstraint &constraint = constraints[predicate->sgpr];
    auto is_satisfiable = [&]() {
      if (constraint.unsigned_min && constraint.unsigned_max &&
          *constraint.unsigned_min > *constraint.unsigned_max)
        return false;
      if (constraint.equal) {
        if (constraint.not_equal.contains(*constraint.equal))
          return false;
        if (constraint.unsigned_min && *constraint.equal < *constraint.unsigned_min)
          return false;
        if (constraint.unsigned_max && *constraint.equal > *constraint.unsigned_max)
          return false;
      }
      if (constraint.unsigned_min && constraint.unsigned_max &&
          *constraint.unsigned_min == *constraint.unsigned_max &&
          constraint.not_equal.contains(*constraint.unsigned_min))
        return false;
      return true;
    };

    switch (predicate->kind) {
    case SccPredicate::Kind::EqImm:
      if (branch_condition_true) {
        if (constraint.equal && *constraint.equal != predicate->value)
          return false;
        if (constraint.not_equal.contains(predicate->value))
          return false;
        constraint.equal = predicate->value;
        return is_satisfiable();
      }
      if (constraint.equal && *constraint.equal == predicate->value)
        return false;
      constraint.not_equal.insert(predicate->value);
      return is_satisfiable();
    case SccPredicate::Kind::GeUnsignedImm:
      if (branch_condition_true) {
        if (!constraint.unsigned_min || *constraint.unsigned_min < predicate->value)
          constraint.unsigned_min = predicate->value;
      } else {
        if (predicate->value == 0)
          return false;
        const uint32_t upper_bound = predicate->value - 1;
        if (!constraint.unsigned_max || *constraint.unsigned_max > upper_bound)
          constraint.unsigned_max = upper_bound;
      }
      return is_satisfiable();
    }
    return true;
  }

  [[nodiscard]] static std::string constraints_key(const ScalarConstraints &constraints,
                                                   const std::optional<SccPredicate> &predicate) {
    std::ostringstream os;
    for (const auto &[sgpr, constraint] : constraints) {
      os << 's' << sgpr << '=';
      if (constraint.equal)
        os << *constraint.equal;
      os << '!';
      for (uint32_t value : constraint.not_equal)
        os << value << ',';
      if (constraint.unsigned_min)
        os << ">=" << *constraint.unsigned_min;
      if (constraint.unsigned_max)
        os << "<=" << *constraint.unsigned_max;
      os << ';';
    }
    if (predicate)
      os << "scc:" << static_cast<unsigned>(predicate->kind) << ":s" << predicate->sgpr << ':'
         << predicate->value;
    return os.str();
  }

  static void invalidate_redefined_scalar_constraints(ScalarConstraints &constraints,
                                                      const InstDefUse &du) {
    std::vector<uint16_t> redefined;
    du.defs.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::SGPR)
        redefined.push_back(ref.index);
    });
    for (uint16_t sgpr : redefined)
      constraints.erase(sgpr);
  }

  [[nodiscard]] bool has_cfg_path_with_event_pending(const PendingEvent &event,
                                                     uint64_t consumer_offset, rj_code_arch_t arch,
                                                     bool before_target = false) {
    if (cfg_views_.empty())
      return true;

    using CacheKey = std::tuple<uint64_t, uint64_t, WaitCounterKind, bool>;
    const CacheKey cache_key{event.section_offset, consumer_offset, event.counter, before_target};
    if (auto cached = feasible_path_cache_.find(cache_key); cached != feasible_path_cache_.end())
      return cached->second;

    // The dataflow state passed to the current instruction already proves
    // that an earlier event in the same straight-line block is still pending.
    // Avoid launching the path-sensitive CFG search for the overwhelmingly
    // common compiler pattern of a wait shortly after its producer.
    if (current_cfg_view_index_ && *current_cfg_view_index_ < cfg_views_.size()) {
      const BasicBlock *current_block = cfg_views_[*current_cfg_view_index_].block;
      if (current_block != nullptr && event.section_offset < consumer_offset &&
          event.section_offset >= current_block->start_offset() &&
          consumer_offset < current_block->end_offset()) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }
    }

    const std::vector<size_t> producer_block_indices =
        cfg_block_indices_containing(event.section_offset);
    if (before_target && current_cfg_view_index_ && *current_cfg_view_index_ < cfg_views_.size() &&
        std::ranges::any_of(producer_block_indices, [&](size_t producer_block_index) {
          return cfg_block_dominates(producer_block_index, *current_cfg_view_index_);
        })) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    const std::vector<size_t> consumer_block_indices =
        cfg_block_indices_containing(consumer_offset);
    if (producer_block_indices.empty()) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }
    if (consumer_block_indices.empty()) {
      feasible_path_cache_[cache_key] = true;
      return true;
    }

    std::vector<uint8_t> can_reach_consumer(cfg_views_.size());
    for (size_t consumer_block_index : consumer_block_indices) {
      const SharedReachability reachable = blocks_reaching(consumer_block_index);
      for (size_t i = 0; i < reachable->size(); ++i)
        can_reach_consumer[i] = static_cast<uint8_t>(can_reach_consumer[i] | (*reachable)[i]);
    }
    std::vector<uint8_t> can_reach_producer(cfg_views_.size());
    std::vector<uint8_t> reachable_from_producer(cfg_views_.size());
    for (size_t producer_block_index : producer_block_indices) {
      const SharedReachability reaching = blocks_reaching(producer_block_index);
      const SharedReachability reachable = blocks_reachable_from(producer_block_index);
      for (size_t i = 0; i < cfg_views_.size(); ++i) {
        can_reach_producer[i] = static_cast<uint8_t>(can_reach_producer[i] | (*reaching)[i]);
        reachable_from_producer[i] =
            static_cast<uint8_t>(reachable_from_producer[i] | (*reachable)[i]);
      }
    }
    const bool producer_can_reach_consumer =
        std::ranges::any_of(producer_block_indices, [&](size_t producer_block_index) {
          return can_reach_consumer[producer_block_index] != 0;
        });
    if (!producer_can_reach_consumer) {
      feasible_path_cache_[cache_key] = false;
      return false;
    }

    // Only scalar predicates repeated after the producer can correlate the path
    // that selected a producer with the path to its consumer. Tracking unrelated
    // comparisons across large dispatch CFGs creates a combinatorial number of
    // equivalent search states without improving feasibility precision.
    using PredicateKey = std::tuple<SccPredicate::Kind, uint16_t, uint32_t>;
    std::set<PredicateKey> predicates_before_producer;
    std::set<PredicateKey> predicates_after_producer;
    for (size_t block_index = 0; block_index < cfg_views_.size(); ++block_index) {
      for (const CfgInstructionView &inst_view : cfg_views_[block_index].instructions) {
        const auto predicate = scalar_scc_predicate(*inst_view.instruction);
        if (!predicate)
          continue;
        const PredicateKey key{predicate->kind, predicate->sgpr, predicate->value};
        if (can_reach_producer[block_index] != 0 &&
            inst_view.section_offset < event.section_offset) {
          predicates_before_producer.insert(key);
        }
        if (reachable_from_producer[block_index] != 0 && can_reach_consumer[block_index] != 0 &&
            inst_view.section_offset >= event.section_offset) {
          predicates_after_producer.insert(key);
        }
      }
    }
    std::set<PredicateKey> relevant_predicates;
    std::ranges::set_intersection(predicates_before_producer, predicates_after_producer,
                                  std::inserter(relevant_predicates, relevant_predicates.end()));
    if (relevant_predicates.empty()) {
      // Without a supported repeated scalar predicate there is no correlation
      // for the path filter to prove. The converged dataflow state already
      // established a structural path on which this event remains pending.
      feasible_path_cache_[cache_key] = true;
      return true;
    }

    struct SearchState {
      size_t block_index = 0;
      size_t inst_index = 0;
      bool pending = false;
      uint32_t age = 0;
      ScalarConstraints constraints;
      std::optional<SccPredicate> scc_predicate;
    };

    constexpr uint32_t kMaxTrackedAge = 255;
    constexpr size_t kMaxPathSearchStates = 200000;
    std::vector<SearchState> worklist;
    for (size_t i = 0; i < cfg_views_.size(); ++i) {
      const BasicBlock *block = cfg_views_[i].block;
      if (block != nullptr && cfg_views_[i].predecessors.empty() && can_reach_producer[i] != 0)
        worklist.push_back({i, 0, false, 0, {}, std::nullopt});
    }
    if (worklist.empty()) {
      for (size_t producer_block_index : producer_block_indices)
        worklist.push_back({producer_block_index, 0, false, 0, {}, std::nullopt});
    }
    std::set<std::tuple<size_t, size_t, bool, uint32_t, std::string>> visited;
    size_t states_processed = 0;

    auto can_continue_from = [&](size_t block_index, bool pending) {
      return pending ? can_reach_consumer[block_index] != 0 : can_reach_producer[block_index] != 0;
    };

    while (!worklist.empty()) {
      SearchState state = worklist.back();
      worklist.pop_back();
      state.age = std::min(state.age, kMaxTrackedAge);
      const std::string constraint_key = constraints_key(state.constraints, state.scc_predicate);
      if (!visited
               .insert(
                   {state.block_index, state.inst_index, state.pending, state.age, constraint_key})
               .second)
        continue;
      if (++states_processed > kMaxPathSearchStates) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }

      const CfgBlockView &block_view = cfg_views_[state.block_index];
      if (state.inst_index >= block_view.instructions.size()) {
        if (block_view.block == nullptr)
          continue;
        for (size_t successor : block_view.successors) {
          if (can_continue_from(successor, state.pending))
            worklist.push_back(
                {successor, 0, state.pending, state.age, state.constraints, state.scc_predicate});
        }
        continue;
      }

      const CfgInstructionView &inst_view = block_view.instructions[state.inst_index];
      const Instruction &inst = *inst_view.instruction;
      if (before_target && inst_view.section_offset == consumer_offset && state.pending) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }
      apply_path_pre_dependency_waits(inst, event, arch, state.pending, state.age);

      if (!before_target && inst_view.section_offset == consumer_offset && state.pending) {
        feasible_path_cache_[cache_key] = true;
        return true;
      }

      if (inst_view.section_offset == event.section_offset) {
        state.pending = true;
        state.age = 0;
      } else if (state.pending && !inst.is_waitcnt() && is_program_end(inst.mnemonic())) {
        state.pending = false;
        state.age = 0;
      } else if (state.pending && !inst.is_waitcnt() &&
                 instruction_adds_younger_event(inst, event, arch)) {
        state.age = std::min<uint32_t>(state.age + 1, kMaxTrackedAge);
      }

      const InstDefUse du = inst_def_use_for_waitcheck(inst, VgprMsbState{}, arch, wavefront_size_);
      invalidate_redefined_scalar_constraints(state.constraints, du);

      if (const auto predicate = scalar_scc_predicate(inst)) {
        if (relevant_predicates.contains({predicate->kind, predicate->sgpr, predicate->value}))
          state.scc_predicate = predicate;
        else
          state.scc_predicate.reset();
      } else if (is_scc_conditional_branch(inst.mnemonic())) {
        const auto branch_delta = inst.branch_offset_bytes();
        const auto fallthrough_offset =
            inst_view.section_offset + static_cast<uint64_t>(inst.size());
        if (branch_delta) {
          const int64_t target =
              static_cast<int64_t>(fallthrough_offset) + static_cast<int64_t>(*branch_delta);
          if (target >= 0) {
            for (size_t successor : block_view.successors) {
              const BasicBlock *successor_block = cfg_views_[successor].block;
              if (successor_block == nullptr ||
                  successor_block->start_offset() != static_cast<uint64_t>(target) ||
                  !can_continue_from(successor, state.pending))
                continue;
              ScalarConstraints taken_constraints = state.constraints;
              const bool branch_takes_on_scc_true = inst.mnemonic() == "s_cbranch_scc1";
              if (apply_scc_branch_constraint(taken_constraints, state.scc_predicate,
                                              branch_takes_on_scc_true)) {
                worklist.push_back({successor, 0, state.pending, state.age,
                                    std::move(taken_constraints), std::nullopt});
              }
            }
          }
        }

        ScalarConstraints fallthrough_constraints = state.constraints;
        const bool fallthrough_on_scc_true = inst.mnemonic() == "s_cbranch_scc0";
        if (apply_scc_branch_constraint(fallthrough_constraints, state.scc_predicate,
                                        fallthrough_on_scc_true)) {
          bool found_fallthrough = false;
          for (size_t successor : block_view.successors) {
            const BasicBlock *successor_block = cfg_views_[successor].block;
            if (successor_block == nullptr || successor_block->start_offset() != fallthrough_offset)
              continue;
            found_fallthrough = true;
            if (can_continue_from(successor, state.pending)) {
              worklist.push_back(
                  {successor, 0, state.pending, state.age, fallthrough_constraints, std::nullopt});
            }
          }
          if (!found_fallthrough && state.inst_index + 1 < block_view.instructions.size()) {
            worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.age,
                                std::move(fallthrough_constraints), std::nullopt});
          }
        }
        continue;
      } else if (instruction_defines_special(inst, RegisterRef{RegClass::SCC, 0, 1}) ||
                 instruction_uses_special(inst, RegisterRef{RegClass::SCC, 0, 1})) {
        state.scc_predicate.reset();
      }

      worklist.push_back({state.block_index, state.inst_index + 1, state.pending, state.age,
                          std::move(state.constraints), state.scc_predicate});
    }

    feasible_path_cache_[cache_key] = false;
    return false;
  }

  [[nodiscard]] bool should_emit_cfg_diagnostic(const PendingEvent &event, uint64_t consumer_offset,
                                                rj_code_arch_t arch) {
    return has_cfg_path_with_event_pending(event, consumer_offset, arch);
  }

  [[nodiscard]] bool event_can_reach_wait(const PendingEvent &event, uint64_t wait_offset,
                                          rj_code_arch_t arch) {
    return has_cfg_path_with_event_pending(event, wait_offset, arch, true);
  }

  [[nodiscard]] static bool instruction_uses_special(const Instruction &inst, RegisterRef ref) {
    if (ref.cls != RegClass::SCC)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_cselect_b32" || mnemonic == "s_cselect_b64" ||
           mnemonic == "s_cbranch_scc0" || mnemonic == "s_cbranch_scc1" ||
           mnemonic == "s_addc_u32" || mnemonic == "s_subb_u32";
  }

  [[nodiscard]] static bool instruction_defines_special(const Instruction &inst, RegisterRef ref) {
    if (ref.cls != RegClass::SCC)
      return false;

    const std::string_view mnemonic = inst.mnemonic();
    return mnemonic == "s_barrier_signal_isfirst" || starts_with(mnemonic, "s_cmp_") ||
           starts_with(mnemonic, "s_bitcmp") || starts_with(mnemonic, "s_addk_co_") ||
           starts_with(mnemonic, "s_add_co_") || starts_with(mnemonic, "s_sub_co_") ||
           starts_with(mnemonic, "s_addc_") || starts_with(mnemonic, "s_subb_");
  }

  template <typename Visitor>
  void visit_dependencies(const PendingState &state, const Instruction &inst, const InstDefUse &du,
                          std::span<const ClassifiedEvent> current_events, rj_code_arch_t arch,
                          DependencyView dependency_view, Visitor &&visit) {
    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        std::optional<RegisterRef> reg;
        WaitcheckAccessKind access = WaitcheckAccessKind::Use;
        if (event.check_uses) {
          // An asynchronous load creates a pending register generation.  Until
          // the matching wait retires it, instructions may still consume a
          // committed generation that was available when the load issued.
          // Restrict RAW diagnostics to lanes for which no such old generation
          // exists. Definitions are handled separately below because CDNA also
          // supports synchronous overlays of the visible generation.
          const RegisterSet dependency_regs =
              dependency_view == DependencyView::CompilerPendingDefinition
                  ? event.regs
                  : event.regs - event.old_value_regs;
          reg = first_dependency_intersection(event, dependency_regs, inst, du.uses,
                                              WaitcheckAccessKind::Use, arch);
        }
        if (!reg && event.check_defs) {
          RegisterSet dependency_regs = dependency_view == DependencyView::CompilerPendingDefinition
                                            ? event.regs
                                            : event.regs - event.old_value_regs;
          if (dependency_view == DependencyView::RuntimeVisibleGeneration &&
              creates_immediate_overlay_generation(arch, inst, current_events))
            dependency_regs -= du.defs;
          reg = first_dependency_intersection(event, dependency_regs, inst, du.defs,
                                              WaitcheckAccessKind::Def, arch);
          access = WaitcheckAccessKind::Def;
        }
        if (!reg && event.special_reg) {
          if (event.check_uses && instruction_uses_special(inst, *event.special_reg)) {
            reg = event.special_reg;
            access = WaitcheckAccessKind::Use;
          } else if (event.check_defs && instruction_defines_special(inst, *event.special_reg)) {
            reg = event.special_reg;
            access = WaitcheckAccessKind::Def;
          }
        }
        if (!reg && event.check_memory_order) {
          const bool is_consumer = event.counter == WaitCounterKind::Async
                                       ? is_async_ordering_consumer(event, inst.mnemonic())
                                   : event.counter == WaitCounterKind::Tensor
                                       ? is_tensor_ordering_consumer(event, inst.mnemonic())
                                       : is_memory_ordering_consumer(inst.mnemonic());
          if (is_consumer) {
            reg = RegisterRef{RegClass::PC, 0, 1};
            access = WaitcheckAccessKind::MemoryOrder;
          }
        }
        if (!reg && event.check_program_end && is_program_end(inst.mnemonic())) {
          reg = RegisterRef{RegClass::PC, 0, 1};
          access = WaitcheckAccessKind::ProgramEnd;
        }
        if (!reg)
          continue;
        if (access == WaitcheckAccessKind::Def && ordered_waw(event, current_events))
          continue;

        const auto required_count = dependency_required_count(state, event, arch);
        visit(event, *reg, access, required_count);
      }
    }

    if (!instruction_defines_exec(inst))
      return;

    for (size_t counter_idx = 0; counter_idx < state.pending.size(); ++counter_idx) {
      const auto &events = state.pending[counter_idx];
      for (size_t i = 0; i < events.size(); ++i) {
        const PendingEvent &event = events[i];
        if (!event.check_exec_defs)
          continue;
        const auto required_count = dependency_required_count(state, event, arch);
        visit(event, RegisterRef{RegClass::EXEC, 0, 1}, WaitcheckAccessKind::Def, required_count);
      }
    }
  }

  void check_dependencies(const PendingState &state, const Instruction &inst, const InstDefUse &du,
                          std::span<const ClassifiedEvent> current_events, uint64_t section_offset,
                          uint64_t file_offset, rj_code_arch_t arch) {
    visit_dependencies(state, inst, du, current_events, arch,
                       DependencyView::RuntimeVisibleGeneration,
                       [&](const PendingEvent &event, RegisterRef reg, WaitcheckAccessKind access,
                           uint32_t required_count) {
                         emit_diagnostic(inst, event, reg, access, required_count, section_offset,
                                         file_offset, arch);
                       });
  }

  [[nodiscard]] static RegisterSet
  registers_for_event(const Instruction &inst, const InstDefUse &du, ClassifiedEvent classification,
                      const VgprMsbState &state, rj_code_arch_t arch) {
    switch (classification.registers) {
    case TrackedRegisterSource::None:
      return {};
    case TrackedRegisterSource::Defs:
      return du.defs;
    case TrackedRegisterSource::Uses:
      return du.uses;
    case TrackedRegisterSource::VectorUses:
      return vector_use_registers(du);
    case TrackedRegisterSource::StoreDataUses:
      return store_data_registers(inst, state, arch);
    }
    return {};
  }

  [[nodiscard]] static RegisterSet vector_use_registers(const InstDefUse &du) {
    RegisterSet regs;
    du.uses.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR)
        regs.expand(ref);
    });
    return regs;
  }

  [[nodiscard]] static RegisterSet
  store_data_registers(const Instruction &inst, const VgprMsbState &state, rj_code_arch_t arch) {
    RegisterSet regs;
    const auto mnemonic = inst.mnemonic();
    if (is_ds_store(mnemonic)) {
      for (int i = 0; i < inst.num_src_operands(); ++i) {
        const Operand *op = inst.src_operand(i);
        if (!op)
          continue;
        if (auto ref = op->to_register_ref())
          expand_vgpr_msb_ref(regs, *ref, *op, state, arch);
      }
      return regs;
    }

    const int data_operand_index =
        starts_with(mnemonic, "buffer_store") || starts_with(mnemonic, "tbuffer_store") ||
                starts_with(mnemonic, "image_store") || is_image_atomic(mnemonic)
            ? 0
            : 1;
    const Operand *op = inst.src_operand(data_operand_index);
    if (op) {
      if (auto ref = op->to_register_ref())
        expand_vgpr_msb_ref(regs, *ref, *op, state, arch);
    }
    return regs;
  }

  [[nodiscard]] static bool instruction_defines_exec(const Instruction &inst) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *op = inst.dst_operand(i);
      if (!op)
        continue;
      const std::string name = op->name();
      if (equals_ignore_ascii_case(name, "exec") || equals_ignore_ascii_case(name, "exec_lo") ||
          equals_ignore_ascii_case(name, "exec_hi"))
        return true;
    }

    const std::string_view mnemonic = inst.mnemonic();
    return starts_with(mnemonic, "v_cmpx_") ||
           mnemonic.find("_saveexec_") != std::string_view::npos ||
           mnemonic.find("_wrexec_") != std::string_view::npos;
  }

  [[nodiscard]] static uint16_t sgpr_pair(RegisterRef ref) {
    return static_cast<uint16_t>((ref.index >> 1u) & 0x3fu);
  }

  static void track_sgpr_uses(SgprHazardState &state, const RegisterSet &uses) {
    uses.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::SGPR)
        state.tracked_pairs.set(sgpr_pair(ref));
    });
  }

  [[nodiscard]] static bool operand_is_vcc(const Operand *op) {
    if (op == nullptr)
      return false;
    const std::string name = op->name();
    return name == "vcc" || name == "vcc_lo" || name == "vcc_hi";
  }

  [[nodiscard]] static bool instruction_uses_vcc_explicit(const Instruction &inst) {
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (operand_is_vcc(inst.src_operand(i)))
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool instruction_defines_vcc_explicit(const Instruction &inst) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      if (operand_is_vcc(inst.dst_operand(i)))
        return true;
    }
    return false;
  }

  [[nodiscard]] static bool has_sgpr_def(const RegisterSet &defs) {
    bool result = false;
    defs.for_each([&](RegisterRef ref) {
      if (ref.cls == RegClass::SGPR)
        result = true;
    });
    return result;
  }

  [[nodiscard]] static bool instruction_uses_vcc_implicit(const Instruction &inst) {
    const std::string_view mnemonic = inst.mnemonic();
    constexpr uint16_t kVopdXyEncodingId = 0x32;
    constexpr uint16_t kVopdCndmaskOp = 9;
    const bool vopd_xy_uses_vcc =
        inst.encoding_id() == kVopdXyEncodingId &&
        ((inst.opcode() >> 8u) == kVopdCndmaskOp || (inst.opcode() & 0xffu) == kVopdCndmaskOp);
    return mnemonic == "v_cndmask_b32_e32" || mnemonic == "v_add_co_ci_u32_e32" ||
           mnemonic == "v_sub_co_ci_u32_e32" || mnemonic == "v_subrev_co_ci_u32_e32" ||
           vopd_xy_uses_vcc;
  }

  [[nodiscard]] static bool instruction_defines_vcc_implicit(const Instruction &inst,
                                                             const InstDefUse &du) {
    const std::string_view mnemonic = inst.mnemonic();
    if (starts_with(mnemonic, "v_cmp_") && !has_sgpr_def(du.defs))
      return true;
    return mnemonic == "v_add_co_ci_u32_e32" || mnemonic == "v_sub_co_ci_u32_e32" ||
           mnemonic == "v_subrev_co_ci_u32_e32";
  }

  [[nodiscard]] static bool instruction_uses_vcc(const Instruction &inst) {
    return instruction_uses_vcc_explicit(inst) || instruction_uses_vcc_implicit(inst);
  }

  [[nodiscard]] static bool instruction_defines_vcc(const Instruction &inst, const InstDefUse &du) {
    return instruction_defines_vcc_explicit(inst) || instruction_defines_vcc_implicit(inst, du);
  }

  static void set_sgpr_hazard(SgprHazardState &state, RegisterRef ref, bool is_valu,
                              const SgprHazardProducer &producer) {
    if (ref.cls != RegClass::SGPR || ref.index >= 128)
      return;
    if (!state.tracked_pairs.test(sgpr_pair(ref)))
      return;

    if (is_valu) {
      state.valu_hazards.set(ref.index);
      state.valu_producers.insert_or_assign(ref.index, producer);
    } else {
      state.salu_hazards.set(ref.index);
      state.salu_producers.insert_or_assign(ref.index, producer);
    }
  }

  static void set_vcc_hazard(SgprHazardState &state, bool is_valu,
                             const SgprHazardProducer &producer) {
    if (!state.tracked_vcc)
      return;

    state.vcc_hazard = is_valu ? kSgprHazardValu : kSgprHazardSalu;
    if (is_valu) {
      state.valu_vcc_producer = producer;
      state.salu_vcc_producer.reset();
    } else {
      state.salu_vcc_producer = producer;
      state.valu_vcc_producer.reset();
    }
  }

  [[nodiscard]] static std::optional<uint16_t>
  first_sgpr_hazard_lane(const std::bitset<128> &hazards) {
    for (uint16_t lane = 0; lane < hazards.size(); ++lane) {
      if (hazards.test(lane))
        return lane;
    }
    return std::nullopt;
  }

  void check_sgpr_control_transfer_hazards(const SgprHazardState &state, const Instruction &inst,
                                           uint64_t section_offset, uint64_t file_offset) {
    // One all-zero depctr field repairs every register in that hazard class.
    // Emit its lowest-numbered representative instead of one diagnostic per
    // outstanding lane; VCC represents the field only when no SGPR lane does.
    if (const auto lane = first_sgpr_hazard_lane(state.salu_hazards)) {
      const auto producer = state.salu_producers.find(*lane);
      emit_sgpr_hazard_diagnostic(
          inst, producer == state.salu_producers.end() ? nullptr : &producer->second,
          RegisterRef{RegClass::SGPR, *lane, 1}, "depctr_sa_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    } else if ((state.vcc_hazard & kSgprHazardSalu) != 0) {
      emit_sgpr_hazard_diagnostic(
          inst, state.salu_vcc_producer ? &*state.salu_vcc_producer : nullptr,
          RegisterRef{RegClass::VCC, 0, 1}, "depctr_sa_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    }

    if (const auto lane = first_sgpr_hazard_lane(state.valu_hazards)) {
      const auto producer = state.valu_producers.find(*lane);
      emit_sgpr_hazard_diagnostic(
          inst, producer == state.valu_producers.end() ? nullptr : &producer->second,
          RegisterRef{RegClass::SGPR, *lane, 1}, "depctr_va_sdst", section_offset, file_offset,
          WaitcheckAccessKind::ControlTransfer);
    }
    if ((state.vcc_hazard & kSgprHazardValu) != 0) {
      emit_sgpr_hazard_diagnostic(inst,
                                  state.valu_vcc_producer ? &*state.valu_vcc_producer : nullptr,
                                  RegisterRef{RegClass::VCC, 0, 1}, "depctr_va_vcc", section_offset,
                                  file_offset, WaitcheckAccessKind::ControlTransfer);
    }
  }

  void check_sgpr_hazard_uses(const SgprHazardState &state, const Instruction &inst,
                              const RegisterSet &uses, bool is_valu, uint64_t section_offset,
                              uint64_t file_offset) {
    uses.for_each([&](RegisterRef ref) {
      if (ref.cls != RegClass::SGPR || ref.index >= 128)
        return;
      if (!state.tracked_pairs.test(sgpr_pair(ref)))
        return;

      if (is_valu && state.salu_hazards.test(ref.index)) {
        const auto producer = state.salu_producers.find(ref.index);
        emit_sgpr_hazard_diagnostic(
            inst, producer == state.salu_producers.end() ? nullptr : &producer->second, ref,
            "depctr_sa_sdst", section_offset, file_offset);
      }
      if (is_valu && state.valu_hazards.test(ref.index)) {
        const auto producer = state.valu_producers.find(ref.index);
        emit_sgpr_hazard_diagnostic(
            inst, producer == state.valu_producers.end() ? nullptr : &producer->second, ref,
            "depctr_va_sdst", section_offset, file_offset);
      }
    });
  }

  void check_vcc_hazard_use(const SgprHazardState &state, const Instruction &inst, bool is_valu,
                            uint64_t section_offset, uint64_t file_offset) {
    if (!state.tracked_vcc)
      return;

    RegisterRef vcc{RegClass::VCC, 0, 1};
    if (is_valu && (state.vcc_hazard & kSgprHazardSalu))
      emit_sgpr_hazard_diagnostic(inst,
                                  state.salu_vcc_producer ? &*state.salu_vcc_producer : nullptr,
                                  vcc, "depctr_sa_sdst", section_offset, file_offset);
    if (is_valu && (state.vcc_hazard & kSgprHazardValu))
      emit_sgpr_hazard_diagnostic(inst,
                                  state.valu_vcc_producer ? &*state.valu_vcc_producer : nullptr,
                                  vcc, "depctr_va_vcc", section_offset, file_offset);
  }

  void update_sgpr_hazards(SgprHazardState &state, const Instruction &inst, const InstDefUse &du,
                           const std::string &section_name, uint64_t section_offset,
                           uint64_t file_offset, bool emit_diagnostics) {
    if (is_sgpr_hazard_control_transfer(inst)) {
      if (emit_diagnostics)
        check_sgpr_control_transfer_hazards(state, inst, section_offset, file_offset);
      // Calls, returns, and indirect branches do not carry the hardware hazard
      // chain across the transfer. Report every required field at the source,
      // then begin the target context without stale producer state. CFG edge
      // construction separately marks callees and return continuations as
      // conservatively tracked and injects a call's link-register write only
      // into that call's matching continuation.
      clear_all_sgpr_hazards(state);
      return;
    }

    if (instruction_waits_for_valu_sgpr_writes(inst))
      clear_valu_sgpr_hazards(state);

    const bool is_salu = is_scalar_alu(inst);
    const bool is_valu = is_vector_alu(inst);
    if (!is_salu && !is_valu)
      return;

    const bool uses_vcc = instruction_uses_vcc(inst);
    const bool defines_vcc = instruction_defines_vcc(inst, du);
    if (is_salu && uses_vcc)
      clear_valu_vcc_hazard(state);
    if (emit_diagnostics) {
      check_sgpr_hazard_uses(state, inst, du.uses, is_valu, section_offset, file_offset);
      if (uses_vcc)
        check_vcc_hazard_use(state, inst, is_valu, section_offset, file_offset);
    }

    if (is_valu) {
      track_sgpr_uses(state, du.uses);
      if (uses_vcc)
        state.tracked_vcc = true;
    }

    SgprHazardProducer producer{section_name, section_offset, file_offset, inst.disassemble()};
    du.defs.for_each([&](RegisterRef ref) { set_sgpr_hazard(state, ref, is_valu, producer); });
    if (defines_vcc)
      set_vcc_hazard(state, is_valu, producer);
  }

  void add_event(PendingState &state, const RegisterSet &local_ready_regs,
                 ClassifiedEvent classification, const Instruction &inst, const InstDefUse &du,
                 std::string section_name, uint64_t section_offset, uint64_t file_offset,
                 std::string instruction, rj_code_arch_t arch, bool record_stats) {
    if (arch == ROCJITSU_CODE_ARCH_CDNA4 && classification.counter == WaitCounterKind::Load) {
      const uint32_t max_wait = maximum_dependency_wait(arch, WaitCounterKind::Load);
      for (DtlVisibilityEvent &visibility : state.dtl_visibility) {
        if (visibility.active && visibility.min_younger < max_wait)
          ++visibility.min_younger;
      }
      if (classification.kind == WaitEventKind::LdsDirect) {
        if (!dtl_straight_line_model_) {
          if (record_stats) {
            record_incomplete_analysis("direct-to-LDS visibility crosses nontrivial control flow");
          }
        } else {
          state.dtl_visibility.push_back(DtlVisibilityEvent{
              .interval = cdna4_dtl_interval(state, inst, wavefront_size_),
              .active = true,
              .min_younger = 0,
              .min_ordered_younger = 0,
              .barrier_required_count = std::nullopt,
              .barrier_section_offset = 0,
              .section_name = section_name,
              .section_offset = section_offset,
              .file_offset = file_offset,
              .instruction = instruction,
          });
        }
      }
    }

    PendingEvent event;
    event.counter = classification.counter;
    event.kind = classification.kind;
    event.ordered_domain = ordered_counter_domain(arch, classification, inst);
    event.regs = registers_for_event(inst, du, classification, state.vgpr_msb, arch);
    if (classification.registers == TrackedRegisterSource::Defs) {
      if (const auto partial = partial_d16_load_def(inst, arch);
          partial && event.regs.contains(partial->reg)) {
        event.partial_reg = partial->reg;
        event.partial_reg_mask = partial->mask;
      }
    }
    event.produces_regs = tracks_committed_vgpr_generations(arch) &&
                          classification.registers == TrackedRegisterSource::Defs;
    event.special_reg = classification.special_reg;
    event.barrier_id = classification.barrier_id;
    event.check_uses = classification.check_uses;
    event.check_defs = classification.check_defs;
    event.check_exec_defs = classification.check_exec_defs;
    event.section_name = std::move(section_name);
    event.section_offset = section_offset;
    event.file_offset = file_offset;
    event.instruction = std::move(instruction);
    event.check_memory_order = classification.check_memory_order;
    event.check_program_end = classification.check_program_end;
    event.check_counter_parity_order = classification.check_counter_parity_order;
    const size_t idx = counter_index(classification.counter);
    if (classification.kind == WaitEventKind::Smem)
      state.pending_smem[idx] = true;
    const uint32_t max_wait = maximum_dependency_wait(arch, classification.counter);
    auto &event_ages = state.pending_event_ages[idx].values;
    for (uint8_t &age : event_ages) {
      if (age != kNoPendingEventAge && age < max_wait)
        ++age;
    }
    event_ages[static_cast<size_t>(classification.kind)] = 0;
    for (PendingEvent &pending_event : state.pending[idx]) {
      if (pending_event.min_younger < max_wait)
        ++pending_event.min_younger;
    }
    if (event.produces_regs)
      event.old_value_regs = event.regs & (state.ready_regs | local_ready_regs);
    if (record_stats)
      ++report_.memory_events_tracked;
    if (is_counter_token_only(event))
      return;
    auto position = std::ranges::lower_bound(state.pending[idx], event, event_identity_less);
    if (position != state.pending[idx].end() && same_event_identity(*position, event)) {
      position->min_younger = 0;
      position->min_ordered_younger = 0;
      position->old_value_regs &= event.old_value_regs;
    } else {
      state.pending[idx].insert(position, std::move(event));
    }
  }

  void analyze_instruction(PendingState &state, RegisterSet &local_ready_regs,
                           const Instruction &inst, const std::string &section_name,
                           uint64_t section_offset, uint64_t file_offset, rj_code_arch_t arch,
                           bool emit_diagnostics) {
    const bool record_stats = emit_diagnostics;
    const bool emit_report_diagnostics = emit_diagnostics && diagnostics_available();
    constexpr uint64_t kNontrivialControlFlow =
        BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL;
    if (arch == ROCJITSU_CODE_ARCH_CDNA4 && (inst.flags() & kNontrivialControlFlow) != 0) {
      if (!state.dtl_visibility.empty() && emit_diagnostics) {
        record_incomplete_analysis("direct-to-LDS visibility crosses nontrivial control flow");
      }
      dtl_straight_line_model_ = false;
      state.dtl_visibility.clear();
      state.lds_constants = {};
    }
    const bool immediately_after_mode_setreg = std::exchange(state.vgpr_msb_setreg_hazard, false);
    const bool previous_vm_vsrc_zero_wait = std::exchange(state.previous_vm_vsrc_zero_wait, false);
    const bool current_vm_vsrc_zero_wait = is_vm_vsrc_zero_wait(inst);
    if (state.async_barrier_post_wait) {
      if (!current_vm_vsrc_zero_wait && emit_report_diagnostics)
        emit_async_barrier_pipe_diagnostic(inst, *state.async_barrier_post_wait,
                                           /*missing_after_barrier=*/true, section_offset,
                                           file_offset);
      state.async_barrier_post_wait.reset();
    }
    if (inst.mnemonic() == "s_delay_alu") {
      apply_s_delay_alu(state, inst);
      return;
    }
    apply_due_delay_alu(state);
    auto finish_instruction = [&]() { advance_delay_alu(state); };
    if (apply_sgpr_hazard_ds_nop_cull(state.sgpr_hazards, inst.mnemonic())) {
      clear_va_vdst_hazards(state.va_vdst_hazards);
      finish_instruction();
      return;
    }
    apply_xcnt_drain(state, inst, arch, immediately_after_mode_setreg);
    if (inst.is_waitcnt()) {
      apply_waitcnt(state, inst, arch);
      state.previous_vm_vsrc_zero_wait = current_vm_vsrc_zero_wait;
      finish_instruction();
      return;
    }
    if (apply_vgpr_msb_mode(state, inst, arch, immediately_after_mode_setreg)) {
      finish_instruction();
      return;
    }
    apply_expert_scheduling_mode(state, inst, arch);
    clear_matching_barrier_scc_write(state, inst);
    // LLVM's AMDGPUWaitSGPRHazards pass models this RDNA4 hardware hazard
    // independently of expert scheduling mode. Do not couple SA_SDST,
    // VA_SDST, or VA_VCC tracking to the expert-only VM/VALU counters.
    if (tracks_gfx12_sgpr_hazards(arch))
      apply_sgpr_hazard_memory_cull(state.sgpr_hazards, inst.mnemonic());
    apply_embedded_waitcnt(state, inst, arch);

    InstDefUse du = inst_def_use_for_waitcheck(inst, state.vgpr_msb, arch, wavefront_size_);
    auto events = classify_events(inst, arch);
    const bool is_async_barrier =
        arch == ROCJITSU_CODE_ARCH_CDNA5 && inst.mnemonic() == "ds_atomic_async_barrier_arrive_b64";
    if (is_async_barrier && !previous_vm_vsrc_zero_wait && emit_report_diagnostics) {
      const SgprHazardProducer barrier{section_name, section_offset, file_offset,
                                       inst.disassemble()};
      emit_async_barrier_pipe_diagnostic(inst, barrier, /*missing_after_barrier=*/false,
                                         section_offset, file_offset);
    }
    if (!expert_waits_enabled(state)) {
      std::erase_if(events, [](const ClassifiedEvent &event) {
        return event.counter == WaitCounterKind::VmVsrc || event.counter == WaitCounterKind::VaVdst;
      });
    }
    // A memory instruction cannot issue if incrementing its wait counter would
    // overflow it. Apply that backpressure before checking the instruction's
    // register and memory dependencies: the instruction that fills the next
    // slot may itself safely consume the oldest completed result.
    apply_ordered_counter_backpressure(state, events, inst, arch);
    apply_implicit_xcnt_ordering(state, du, events);
    if (emit_diagnostics)
      check_dtl_lds_access(state, inst, section_name, section_offset, file_offset, arch);
    if (emit_report_diagnostics) {
      check_dependencies(state, inst, du, events, section_offset, file_offset, arch);
      if (expert_waits_enabled(state))
        check_va_vdst_hazard(state.va_vdst_hazards, inst, du, section_offset, file_offset);
    }
    if (tracks_gfx12_sgpr_hazards(arch)) {
      update_sgpr_hazards(state.sgpr_hazards, inst, du, section_name, section_offset, file_offset,
                          emit_report_diagnostics);
    }
    if (expert_waits_enabled(state))
      update_va_vdst_hazards(state.va_vdst_hazards, inst, du, section_name, section_offset,
                             file_offset);

    if (arch == ROCJITSU_CODE_ARCH_CDNA4 && inst.mnemonic() == "s_barrier")
      apply_dtl_barrier(state, section_offset);

    if (is_program_end(inst.mnemonic())) {
      state = {};
      return;
    }

    RegisterSet asynchronous_defs;
    for (const ClassifiedEvent &event : events) {
      if (event.registers == TrackedRegisterSource::Defs)
        asynchronous_defs |= registers_for_event(inst, du, event, state.vgpr_msb, arch);
      add_event(state, local_ready_regs, event, inst, du, section_name, section_offset, file_offset,
                inst.disassemble(), arch, record_stats);
    }
    update_lds_constant_state(state, inst, du, arch);
    if (is_async_barrier) {
      state.async_barrier_post_wait =
          SgprHazardProducer{section_name, section_offset, file_offset, inst.disassemble()};
    }
    if (tracks_committed_vgpr_generations(arch)) {
      RegisterSet unavailable_regs;
      for (const auto &pending : state.pending) {
        for (const PendingEvent &pending_event : pending) {
          if (pending_event.produces_regs)
            unavailable_regs |= pending_event.regs - pending_event.old_value_regs;
        }
      }
      // A register that is consumed without an outstanding producer already
      // has a committed generation, including ABI/live-in VGPRs such as the
      // workitem id. Remember it so a later asynchronous replacement does not
      // make that old value appear unavailable. Do not bless a first use of a
      // pending load result: it remains unavailable and is diagnosed above.
      const RegisterSet committed_uses = du.uses - unavailable_regs;
      committed_uses.for_each([&](RegisterRef reg) {
        if (reg.cls == RegClass::VGPR || reg.cls == RegClass::ACC_VGPR)
          local_ready_regs.expand(reg);
      });
      const RegisterSet synchronous_defs = du.defs - asynchronous_defs;
      synchronous_defs.for_each([&](RegisterRef reg) {
        if (reg.cls != RegClass::VGPR && reg.cls != RegClass::ACC_VGPR)
          return;
        state.ready_regs.expand(reg);
        for (auto &pending : state.pending) {
          for (PendingEvent &pending_event : pending) {
            if (pending_event.produces_regs && pending_event.regs.contains(reg))
              pending_event.old_value_regs.expand(reg);
          }
        }
      });
    }
    finish_instruction();
  }

  void set_analysis_error(std::string_view section_name, uint64_t section_offset,
                          const util::Exception &ex) {
    report_.supported = false;
    std::ostringstream os;
    os << "decode failed in " << section_name << "+0x" << std::hex << section_offset << ": "
       << ex.what();
    report_.analysis_error = os.str();
  }

  WaitcheckReport &report_;
  WaitcheckOptions options_;
  const WaitcheckKernelInfo *current_kernel_ = nullptr;
  uint32_t wavefront_size_ = 64;
  bool dtl_straight_line_model_ = true;
  std::set<
      std::tuple<WaitCounterKind, WaitcheckAccessKind, RegClass, uint16_t, uint16_t, std::string,
                 uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, std::string, std::string>>
      diagnostic_keys_;
  std::set<std::tuple<std::string, WaitCounterKind, uint64_t, uint64_t, uint64_t, uint32_t,
                      uint32_t, bool>>
      counter_underaccounting_keys_;
  std::vector<CfgBlockView> cfg_views_;
  std::optional<size_t> current_cfg_view_index_;
  bool current_in_callee_context_ = false;
  std::optional<uint16_t> current_call_return_sreg_;
  std::map<std::tuple<uint64_t, uint64_t, WaitCounterKind, bool>, bool> feasible_path_cache_;
  std::map<size_t, SharedReachability> reverse_reachability_cache_;
  std::map<size_t, SharedReachability> forward_reachability_cache_;
  size_t reachability_cache_bytes_ = 0;
  std::vector<size_t> dominator_preorder_;
  std::vector<size_t> dominator_subtree_end_;
};

} // namespace

std::string_view wait_counter_name(WaitCounterKind counter) {
  switch (counter) {
  case WaitCounterKind::Load:
    return "loadcnt";
  case WaitCounterKind::Store:
    return "storecnt";
  case WaitCounterKind::Ds:
    return "dscnt";
  case WaitCounterKind::Km:
    return "kmcnt";
  case WaitCounterKind::Sample:
    return "samplecnt";
  case WaitCounterKind::Bvh:
    return "bvhcnt";
  case WaitCounterKind::Exp:
    return "expcnt";
  case WaitCounterKind::X:
    return "xcnt";
  case WaitCounterKind::Async:
    return "asynccnt";
  case WaitCounterKind::Tensor:
    return "tensorcnt";
  case WaitCounterKind::VmVsrc:
    return "depctr_vm_vsrc";
  case WaitCounterKind::VaVdst:
    return "wait_va_vdst";
  case WaitCounterKind::Depctr:
    return "depctr";
  case WaitCounterKind::Count:
    break;
  }
  return "unknown";
}

rj_code_arch_t waitcheck_arch_for_target(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX942:
    return ROCJITSU_CODE_ARCH_CDNA3;
  case ROCJITSU_CODE_TARGET_GFX1100:
    return ROCJITSU_CODE_ARCH_RDNA3;
  case ROCJITSU_CODE_TARGET_GFX1150:
  case ROCJITSU_CODE_TARGET_GFX1151:
    return ROCJITSU_CODE_ARCH_RDNA3_5;
  case ROCJITSU_CODE_TARGET_GFX1200:
  case ROCJITSU_CODE_TARGET_GFX1201:
    return ROCJITSU_CODE_ARCH_RDNA4;
  case ROCJITSU_CODE_TARGET_GFX1250:
    return ROCJITSU_CODE_ARCH_CDNA5;
  case ROCJITSU_CODE_TARGET_GFX950:
    return ROCJITSU_CODE_ARCH_CDNA4;
  default:
    return ROCJITSU_CODE_ARCH_INVALID;
  }
}

std::vector<WaitcheckKernelInfo> waitcheck_kernels(const CodeObject &code_object) {
  return find_waitcheck_kernels(code_object);
}

WaitcheckReport analyze_waitcnts(std::span<const uint32_t> words, rj_code_arch_t arch,
                                 WaitcheckOptions options) {
  WaitcheckReport report;
  report.arch = arch;
  if (!is_supported_waitcheck_arch(arch)) {
    report.supported = false;
    report.analysis_error = "unsupported architecture";
    return report;
  }

  Analyzer analyzer(report, options);
  analyzer.analyze_stream(words, arch, ".text", 0);
  return report;
}

namespace {

WaitcheckReport analyze_code_object(const CodeObject &code_object, rj_code_arch_t arch,
                                    std::optional<uint64_t> selected_kernel_entry,
                                    std::span<const WaitcheckKernelInfo> known_kernels,
                                    bool kernels_known, WaitcheckOptions options) {
  WaitcheckReport report;
  report.arch = arch;
  if (!is_supported_waitcheck_arch(arch)) {
    report.supported = false;
    report.analysis_error = "unsupported architecture";
    return report;
  }

  if (code_object.image_size() >= sizeof(Elf64_Ehdr)) {
    Elf64_Ehdr ehdr{};
    std::memcpy(&ehdr, code_object.image_data(), sizeof(ehdr));
    if (std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE) == 0 && ehdr.e_type == ET_REL) {
      report.supported = false;
      report.analysis_error = "relocatable AMDGPU objects are not final loadable code objects";
      return report;
    }
  }

  Analyzer analyzer(report, options);
  auto decoder = Decoder::create(arch);
  if (!decoder) {
    report.supported = false;
    report.analysis_error = "failed to create decoder";
    return report;
  }

  const auto text_file_offset = code_object.text_sections().size() == 1
                                    ? code_object.text_sections().front()->sectionOffset()
                                    : 0;
  try {
    std::vector<WaitcheckKernelInfo> discovered_kernels;
    if (!kernels_known)
      discovered_kernels = find_waitcheck_kernels(code_object);
    const std::span<const WaitcheckKernelInfo> kernels =
        kernels_known ? known_kernels : std::span<const WaitcheckKernelInfo>{discovered_kernels};
    report.kernels_discovered = kernels.size();
    const auto function_entries = !kernels_known && kernels.empty()
                                      ? find_waitcheck_function_entries(code_object)
                                      : std::vector<uint64_t>{};
    if (kernels_known && kernels.empty())
      return report;
    if (selected_kernel_entry) {
      const auto kernel_it =
          std::ranges::find(kernels, *selected_kernel_entry, &WaitcheckKernelInfo::entry_offset);
      const bool is_kernel = kernel_it != kernels.end();
      if (!is_kernel &&
          std::ranges::find(function_entries, *selected_kernel_entry) == function_entries.end()) {
        report.supported = false;
        report.analysis_error = "kernel entry offset is not present in the code object";
        return report;
      }
      const std::array<uint64_t, 1> entry{*selected_kernel_entry};
      const std::array<uint64_t, 1> entry_size{is_kernel ? kernel_it->code_size : 0};
      const uint32_t wavefront_size =
          is_kernel ? kernel_it->wavefront_size : default_wavefront_size(arch);
      const auto start = std::chrono::steady_clock::now();
      std::vector<std::unique_ptr<BasicBlock>> blocks = BasicBlock::build_reachable(
          code_object, *decoder, arch, entry, entry_size, wavefront_size);
      analyzer.set_kernel_context(is_kernel ? &*kernel_it : nullptr);
      analyzer.analyze_cfg(blocks, ".text", text_file_offset, arch, wavefront_size, entry,
                           /*non_entry_function=*/!is_kernel);
      if (is_kernel && report.supported && !report.stopped_early) {
        ++report.kernels_analyzed;
        if (options.kernel_analyzed_callback)
          options.kernel_analyzed_callback();
        if (options.kernel_timing_callback)
          options.kernel_timing_callback(*kernel_it,
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - start));
      }
    } else if (kernels.empty() && function_entries.empty()) {
      std::vector<std::unique_ptr<BasicBlock>> blocks =
          BasicBlock::build(code_object, *decoder, arch);
      analyzer.set_kernel_context(nullptr);
      analyzer.analyze_cfg(blocks, ".text", text_file_offset, arch, default_wavefront_size(arch));
    } else {
      // Analyze each kernel independently. Building one combined CFG for a
      // multi-kernel library keeps two large wait states per basic block alive
      // at once; generated Tensile libraries can contain hundreds of kernels.
      const auto analyze_entry = [&](uint64_t entry_offset, const WaitcheckKernelInfo *kernel,
                                     uint32_t wavefront_size) {
        const std::array<uint64_t, 1> entry{entry_offset};
        const std::array<uint64_t, 1> entry_size{kernel ? kernel->code_size : 0};
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::unique_ptr<BasicBlock>> blocks = BasicBlock::build_reachable(
            code_object, *decoder, arch, entry, entry_size, wavefront_size);
        analyzer.set_kernel_context(kernel);
        analyzer.analyze_cfg(blocks, ".text", text_file_offset, arch, wavefront_size, entry,
                             /*non_entry_function=*/kernel == nullptr);
        if (kernel && report.supported && !report.stopped_early) {
          ++report.kernels_analyzed;
          if (options.kernel_analyzed_callback)
            options.kernel_analyzed_callback();
          if (options.kernel_timing_callback)
            options.kernel_timing_callback(*kernel,
                                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - start));
        }
      };
      for (const WaitcheckKernelInfo &kernel : kernels) {
        analyze_entry(kernel.entry_offset, &kernel, kernel.wavefront_size);
        if (!report.supported || report.stopped_early)
          break;
      }
      if (report.supported && !report.stopped_early) {
        for (uint64_t entry_offset : function_entries) {
          analyze_entry(entry_offset, nullptr, default_wavefront_size(arch));
          if (!report.supported || report.stopped_early)
            break;
        }
      }
    }
  } catch (const util::Exception &ex) {
    report.supported = false;
    report.analysis_error = std::string("decode failed while building CFG: ") + ex.what();
    return report;
  }
  if (!report.supported)
    return report;
  if (report.stopped_early)
    return report;
  if (selected_kernel_entry || kernels_known)
    return report;

  for (const auto &owned_section : code_object.all_sections()) {
    const Section *section = owned_section.get();
    if ((section->flags() & SHF_EXECINSTR) == 0)
      continue;
    if (std::ranges::find(code_object.text_sections(), section) !=
        code_object.text_sections().end())
      continue;
    const auto *words = reinterpret_cast<const uint32_t *>(section->data());
    const size_t word_count = section->size() / sizeof(uint32_t);
    analyzer.analyze_stream(std::span<const uint32_t>(words, word_count), arch, section->name(),
                            section->sectionOffset());
  }
  return report;
}

} // namespace

WaitcheckReport analyze_waitcnts(const CodeObject &code_object, rj_code_arch_t arch,
                                 WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, std::nullopt, {}, false, options);
}

WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object, rj_code_arch_t arch,
                                            uint64_t kernel_entry_offset,
                                            WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, kernel_entry_offset, {}, false, options);
}

WaitcheckReport analyze_waitcnts_for_kernel(const CodeObject &code_object, rj_code_arch_t arch,
                                            const WaitcheckKernelInfo &kernel,
                                            WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, kernel.entry_offset,
                             std::span<const WaitcheckKernelInfo>{&kernel, 1}, true, options);
}

WaitcheckReport analyze_waitcnts_for_kernels(const CodeObject &code_object, rj_code_arch_t arch,
                                             std::span<const WaitcheckKernelInfo> kernels,
                                             WaitcheckOptions options) {
  return analyze_code_object(code_object, arch, std::nullopt, kernels, true, options);
}

} // namespace rocjitsu
