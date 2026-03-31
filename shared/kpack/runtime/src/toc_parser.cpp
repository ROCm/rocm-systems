// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <msgpack.hpp>
#include <vector>

#include "kpack_internal.h"

namespace kpack {

// Helper to find key in msgpack map
static msgpack::object* find_key(const msgpack::object_map& map,
                                 const char* key) {
  for (uint32_t i = 0; i < map.size; ++i) {
    const auto& kv = map.ptr[i];
    if (kv.key.type == msgpack::type::STR) {
      std::string k(kv.key.via.str.ptr, kv.key.via.str.size);
      if (k == key) {
        return const_cast<msgpack::object*>(&kv.val);
      }
    }
  }
  return nullptr;
}

kpack_error_t parse_toc(FILE* file, uint64_t toc_offset, uint64_t file_size,
                        kpack_archive* archive) {
  // Validate TOC offset is within file bounds
  if (toc_offset >= file_size) {
    return KPACK_ERROR_INVALID_FORMAT;
  }

  // Seek to TOC
  if (fseek(file, toc_offset, SEEK_SET) != 0) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Read TOC data
  size_t toc_size = file_size - toc_offset;
  std::vector<char> toc_buf(toc_size);
  if (fread(toc_buf.data(), 1, toc_size, file) != toc_size) {
    return KPACK_ERROR_IO_ERROR;
  }

  // Unpack MessagePack
  msgpack::object_handle oh;
  try {
    oh = msgpack::unpack(toc_buf.data(), toc_buf.size());
  } catch (...) {
    return KPACK_ERROR_MSGPACK_PARSE_FAILED;
  }

  msgpack::object obj = oh.get();
  if (obj.type != msgpack::type::MAP) {
    return KPACK_ERROR_MSGPACK_PARSE_FAILED;
  }

  const auto& map = obj.via.map;

  // Extract compression_scheme
  auto* val = find_key(map, "compression_scheme");
  if (val && val->type == msgpack::type::STR) {
    std::string scheme(val->via.str.ptr, val->via.str.size);
    if (scheme == "none") {
      archive->compression_scheme = KPACK_COMPRESSION_NOOP;
    } else if (scheme == "zstd-per-kernel") {
      archive->compression_scheme = KPACK_COMPRESSION_ZSTD_PER_KERNEL;
    }
  }

  // Extract gfx_arches array
  val = find_key(map, "gfx_arches");
  if (val && val->type == msgpack::type::ARRAY) {
    const auto& arr = val->via.array;
    for (uint32_t i = 0; i < arr.size; ++i) {
      if (arr.ptr[i].type == msgpack::type::STR) {
        archive->gfx_arches.push_back(
            std::string(arr.ptr[i].via.str.ptr, arr.ptr[i].via.str.size));
      }
    }
  }

  // Extract NoOp blobs
  if (archive->compression_scheme == KPACK_COMPRESSION_NOOP) {
    val = find_key(map, "blobs");
    if (val && val->type == msgpack::type::ARRAY) {
      const auto& arr = val->via.array;
      for (uint32_t i = 0; i < arr.size; ++i) {
        if (arr.ptr[i].type == msgpack::type::MAP) {
          BlobInfo blob;
          auto* offset_obj = find_key(arr.ptr[i].via.map, "offset");
          auto* size_obj = find_key(arr.ptr[i].via.map, "size");
          if (offset_obj) blob.offset = offset_obj->as<uint64_t>();
          if (size_obj) blob.size = size_obj->as<uint64_t>();
          archive->blobs.push_back(blob);
        }
      }
    }
  }

  // Extract Zstd metadata
  if (archive->compression_scheme == KPACK_COMPRESSION_ZSTD_PER_KERNEL) {
    val = find_key(map, "zstd_offset");
    if (val) archive->zstd_offset = val->as<uint64_t>();
    val = find_key(map, "zstd_size");
    if (val) archive->zstd_size = val->as<uint64_t>();
  }

  // Parse nested TOC
  val = find_key(map, "toc");
  if (val && val->type == msgpack::type::MAP) {
    const auto& toc_map = val->via.map;
    for (uint32_t i = 0; i < toc_map.size; ++i) {
      const auto& binary_kv = toc_map.ptr[i];
      if (binary_kv.key.type != msgpack::type::STR) continue;
      std::string binary_path(binary_kv.key.via.str.ptr,
                              binary_kv.key.via.str.size);

      if (binary_kv.val.type == msgpack::type::MAP) {
        const auto& arch_map = binary_kv.val.via.map;
        for (uint32_t j = 0; j < arch_map.size; ++j) {
          const auto& arch_kv = arch_map.ptr[j];
          if (arch_kv.key.type != msgpack::type::STR) continue;
          std::string arch(arch_kv.key.via.str.ptr, arch_kv.key.via.str.size);

          if (arch_kv.val.type == msgpack::type::MAP) {
            KernelMetadata km;
            const auto& meta_map = arch_kv.val.via.map;
            auto* ordinal_obj = find_key(meta_map, "ordinal");
            auto* size_obj = find_key(meta_map, "original_size");
            auto* type_obj = find_key(meta_map, "type");

            if (ordinal_obj) km.ordinal = ordinal_obj->as<uint32_t>();
            if (size_obj) km.original_size = size_obj->as<uint64_t>();
            if (type_obj && type_obj->type == msgpack::type::STR) {
              km.type =
                  std::string(type_obj->via.str.ptr, type_obj->via.str.size);
            }

            archive->toc[binary_path][arch] = km;
          }
        }
      }
    }
  }

  return KPACK_SUCCESS;
}

}  // namespace kpack
