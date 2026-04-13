// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/kernel_metadata.h"
#include "rocjitsu/code/amdgpu_elf.h"

#include <msgpack.hpp>

#include <cstring>

namespace rocjitsu {

/// Parse kernel argument metadata from the .note section of an AMDGPU device
/// ELF. The .note section contains an ELF note with type NT_AMDGPU_METADATA
/// whose payload is msgpack-encoded AMDHSA metadata. This function
/// navigates the msgpack structure to extract the argument list of the
/// named kernel from amdhsa.kernels[].args[].
///
/// For each argument, the fields .offset, .size, .value_kind, .name, and
/// .address_space are extracted when present. Arguments without a .value_kind
/// field are skipped.
///
/// Returns std::variant<KernelArgs, std::string>: a KernelArgs on success,
/// or an error string describing why parsing failed. Failure reasons include:
/// truncated ELF, missing .note section, wrong note type, malformed msgpack,
/// or missing amdhsa.kernels / .args keys.
///
/// Limitations:
///   - Only the first SHT_NOTE section is examined. If the ELF has multiple
///     note sections, only the first is used.
///   - The function performs no validation of the ELF beyond basic header
///     bounds checks. Malformed ELFs may produce incorrect results.
///   - Kernel name matching uses the ".name" field in the msgpack metadata.
///     If the field is absent for all kernels, name-based selection will fail.
///
/// Complexity: O(S + K) where S is the number of ELF section headers and K is
/// the total number of msgpack key-value pairs in the kernel metadata.
KernelArgs::ParseResult KernelArgs::parse(const uint8_t *elf_data,
                                          size_t elf_size,
                                          const std::string &kernel_name) {
  auto fail = [](std::string msg) -> ParseResult { return msg; };

  if (elf_size < sizeof(Elf64_Ehdr)) {
    return fail("ELF too small (" + std::to_string(elf_size) + " bytes)");
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, elf_data, sizeof(ehdr));
  if (ehdr.e_shoff + ehdr.e_shnum * sizeof(Elf64_Shdr) > elf_size) {
    return fail("section header table extends beyond ELF image");
  }

  const uint8_t *note_data = nullptr;
  for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
    Elf64_Shdr shdr;
    std::memcpy(&shdr, elf_data + ehdr.e_shoff + i * sizeof(Elf64_Shdr),
                sizeof(shdr));
    if (shdr.sh_type == SHT_NOTE &&
        shdr.sh_offset + shdr.sh_size <= elf_size) {
      note_data = elf_data + shdr.sh_offset;
      break;
    }
  }
  if (!note_data) {
    return fail("no SHT_NOTE section found in ELF "
                "(only the first SHT_NOTE section is examined)");
  }

  uint32_t namesz, descsz, ntype;
  std::memcpy(&namesz, note_data, 4);
  std::memcpy(&descsz, note_data + 4, 4);
  std::memcpy(&ntype, note_data + 8, 4);
  if (ntype != NT_AMDGPU_METADATA) {
    return fail("first note has type " + std::to_string(ntype) +
                ", expected NT_AMDGPU_METADATA (32) "
                "(only the first note is examined)");
  }

  uint32_t name_padded = (namesz + 3) & ~3u;
  const char *msgpack_data =
      reinterpret_cast<const char *>(note_data + 12 + name_padded);

  auto handle = msgpack::unpack(msgpack_data, descsz);
  auto top = handle.get();
  if (top.type != msgpack::type::MAP) {
    return fail("top-level msgpack value is not a map");
  }

  auto top_map = top.via.map;
  for (uint32_t i = 0; i < top_map.size; ++i) {
    auto &key = top_map.ptr[i].key;
    if (key.type != msgpack::type::STR ||
        std::string_view(key.via.str.ptr, key.via.str.size) !=
            "amdhsa.kernels") {
      continue;
    }

    auto &kernels = top_map.ptr[i].val;
    if (kernels.type != msgpack::type::ARRAY ||
        kernels.via.array.size == 0) {
      return fail("amdhsa.kernels is not a non-empty array");
    }

    // Find the kernel whose .name matches kernel_name.
    const msgpack::object *kernel_obj = nullptr;
    for (uint32_t ki = 0; ki < kernels.via.array.size; ++ki) {
      auto &candidate = kernels.via.array.ptr[ki];
      if (candidate.type != msgpack::type::MAP) {
        continue;
      }
      auto cmap = candidate.via.map;
      for (uint32_t ci = 0; ci < cmap.size; ++ci) {
        auto &ck = cmap.ptr[ci].key;
        auto &cv = cmap.ptr[ci].val;
        if (ck.type == msgpack::type::STR &&
            std::string_view(ck.via.str.ptr, ck.via.str.size) == ".name" &&
            cv.type == msgpack::type::STR &&
            std::string_view(cv.via.str.ptr, cv.via.str.size) == kernel_name) {
          kernel_obj = &candidate;
          break;
        }
      }
      if (kernel_obj) {
        break;
      }
    }
    if (!kernel_obj) {
      return fail("kernel '" + kernel_name + "' not found in amdhsa.kernels");
    }

    auto kernel_map = kernel_obj->via.map;
    for (uint32_t k = 0; k < kernel_map.size; ++k) {
      auto &kkey = kernel_map.ptr[k].key;
      if (kkey.type != msgpack::type::STR ||
          std::string_view(kkey.via.str.ptr, kkey.via.str.size) != ".args") {
        continue;
      }

      auto &args_array = kernel_map.ptr[k].val;
      if (args_array.type != msgpack::type::ARRAY) {
        return fail(".args is not an array");
      }

      KernelArgs result;
      for (uint32_t a = 0; a < args_array.via.array.size; ++a) {
        auto &arg = args_array.via.array.ptr[a];
        if (arg.type != msgpack::type::MAP) {
          continue;
        }

        Arg entry;
        auto arg_map = arg.via.map;
        for (uint32_t f = 0; f < arg_map.size; ++f) {
          auto &fkey = arg_map.ptr[f].key;
          auto &fval = arg_map.ptr[f].val;
          if (fkey.type != msgpack::type::STR) {
            continue;
          }
          std::string_view fname(fkey.via.str.ptr, fkey.via.str.size);
          if (fname == ".offset") {
            entry.offset = fval.as<uint32_t>();
          } else if (fname == ".size") {
            entry.size = fval.as<uint32_t>();
          } else if (fname == ".value_kind" &&
                     fval.type == msgpack::type::STR) {
            entry.value_kind =
                std::string(fval.via.str.ptr, fval.via.str.size);
          } else if (fname == ".name" && fval.type == msgpack::type::STR) {
            entry.name = std::string(fval.via.str.ptr, fval.via.str.size);
          } else if (fname == ".address_space" &&
                     fval.type == msgpack::type::STR) {
            entry.address_space =
                std::string(fval.via.str.ptr, fval.via.str.size);
          }
        }

        if (!entry.value_kind.empty()) {
          result.add(std::move(entry));
        }
      }
      return result;
    }
    return fail("kernel '" + kernel_name + "' has no .args field");
  }
  return fail("amdhsa.kernels key not found in metadata");
}

} // namespace rocjitsu
