// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Kernel argument metadata extracted from AMDGPU code objects.

#ifndef ROCJITSU_CODE_KERNEL_METADATA_H_
#define ROCJITSU_CODE_KERNEL_METADATA_H_

#include "rocjitsu/code/amdgpu_elf.h"

#include <msgpack.hpp>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>

namespace rocjitsu {

/// A single kernel argument descriptor.
struct KernelArg {
  uint32_t offset;
  uint32_t size;
};

/// All kernel arguments, keyed by value_kind (e.g. "global_buffer",
/// "by_value", "hidden_block_count_x", "hidden_group_size_x").
/// Parsed from the AMDGPU code object's .note metadata.
using KernelArgMap = std::unordered_map<std::string, KernelArg>;

/// Parse kernel argument metadata from an AMDGPU device ELF image.
/// Returns the argument map for the first kernel, or nullopt on failure.
inline std::optional<KernelArgMap>
parseKernelArgs(const uint8_t *elf_data, size_t elf_size) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    return std::nullopt;
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, elf_data, sizeof(ehdr));
  if (ehdr.e_shoff + ehdr.e_shnum * sizeof(Elf64_Shdr) > elf_size) {
    return std::nullopt;
  }

  // Find the SHT_NOTE section.
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
    return std::nullopt;
  }

  // Parse the ELF note header.
  uint32_t namesz, descsz, ntype;
  std::memcpy(&namesz, note_data, 4);
  std::memcpy(&descsz, note_data + 4, 4);
  std::memcpy(&ntype, note_data + 8, 4);
  if (ntype != NT_AMDGPU_METADATA) {
    return std::nullopt;
  }

  uint32_t name_padded = (namesz + 3) & ~3u;
  const char *msgpack_data =
      reinterpret_cast<const char *>(note_data + 12 + name_padded);

  // Parse msgpack.
  auto handle = msgpack::unpack(msgpack_data, descsz);
  auto top = handle.get();
  if (top.type != msgpack::type::MAP) {
    return std::nullopt;
  }

  auto top_map = top.via.map;
  for (uint32_t i = 0; i < top_map.size; ++i) {
    auto &key = top_map.ptr[i].key;
    if (key.type != msgpack::type::STR) {
      continue;
    }
    std::string key_str(key.via.str.ptr, key.via.str.size);
    if (key_str != "amdhsa.kernels") {
      continue;
    }

    auto &kernels = top_map.ptr[i].val;
    if (kernels.type != msgpack::type::ARRAY || kernels.via.array.size == 0) {
      return std::nullopt;
    }

    auto &kernel = kernels.via.array.ptr[0];
    if (kernel.type != msgpack::type::MAP) {
      return std::nullopt;
    }

    auto kernel_map = kernel.via.map;
    for (uint32_t k = 0; k < kernel_map.size; ++k) {
      auto &kkey = kernel_map.ptr[k].key;
      if (kkey.type != msgpack::type::STR) {
        continue;
      }
      std::string kkey_str(kkey.via.str.ptr, kkey.via.str.size);
      if (kkey_str != ".args") {
        continue;
      }

      auto &args = kernel_map.ptr[k].val;
      if (args.type != msgpack::type::ARRAY) {
        return std::nullopt;
      }

      KernelArgMap result;
      for (uint32_t a = 0; a < args.via.array.size; ++a) {
        auto &arg = args.via.array.ptr[a];
        if (arg.type != msgpack::type::MAP) {
          continue;
        }

        uint32_t offset = 0;
        uint32_t size = 0;
        std::string value_kind;

        auto arg_map = arg.via.map;
        for (uint32_t f = 0; f < arg_map.size; ++f) {
          auto &fkey = arg_map.ptr[f].key;
          auto &fval = arg_map.ptr[f].val;
          if (fkey.type != msgpack::type::STR) {
            continue;
          }
          std::string fkey_str(fkey.via.str.ptr, fkey.via.str.size);
          if (fkey_str == ".offset") {
            offset = fval.as<uint32_t>();
          } else if (fkey_str == ".size") {
            size = fval.as<uint32_t>();
          } else if (fkey_str == ".value_kind") {
            value_kind = std::string(fval.via.str.ptr, fval.via.str.size);
          }
        }

        if (!value_kind.empty()) {
          result[value_kind] = {offset, size};
        }
      }
      return result;
    }
  }
  return std::nullopt;
}

} // namespace rocjitsu

#endif // ROCJITSU_CODE_KERNEL_METADATA_H_
