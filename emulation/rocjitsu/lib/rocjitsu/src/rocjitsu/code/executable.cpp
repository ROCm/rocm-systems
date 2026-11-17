// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/executable.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"
#include "util/log.h"

#include <zlib.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <utility>

namespace rocjitsu {

namespace {

constexpr char CLANG_OFFLOAD_KIND_HOST[] = "host";
constexpr char CLANG_OFFLOAD_KIND_HIP[] = "hip";
constexpr char CLANG_OFFLOAD_KIND_HIPV4[] = "hipv4";

class HsaHeader : public Header {
public:
  explicit HsaHeader(const Elf64_Ehdr &ehdr) : ehdr_(ehdr) {}

  uint64_t programHeaderOff() const override { return ehdr_.e_phoff; }
  int numProgramHeaders() const override { return static_cast<int>(ehdr_.e_phnum); }
  uint64_t sectionHeaderOff() const override { return ehdr_.e_shoff; }
  int numSectionHeaders() const override { return static_cast<int>(ehdr_.e_shnum); }
  uint64_t sectionHeaderEntrySize() const override { return ehdr_.e_shentsize; }
  int sectionHeaderStrIdx() const override { return static_cast<int>(ehdr_.e_shstrndx); }
  uint32_t flags() const override { return ehdr_.e_flags; }

private:
  Elf64_Ehdr ehdr_;
};

class HsaSection : public Section {
public:
  HsaSection(std::string name, std::unique_ptr<char[]> data, const Elf64_Shdr &shdr,
             size_t section_index)
      : Section(std::move(name), std::move(data)), shdr_(shdr), section_index_(section_index) {}

  std::size_t size() const override { return shdr_.sh_size; }
  uint64_t flags() const override { return shdr_.sh_flags; }
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  std::optional<size_t> sectionHeaderIndex() const override { return section_index_; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
  size_t section_index_;
};

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

using detail::fits_in_bounds;
using detail::read_value;

struct DecompressedOffloadBundle {
  size_t compressed_file_size = 0;
  std::vector<char> data;
};

[[nodiscard]] std::optional<DecompressedOffloadBundle>
decompress_offload_bundle(const char *input, size_t input_size) {
  constexpr char magic[] = {'C', 'C', 'O', 'B'};
  constexpr uint16_t kZlibMethod = 0;
  constexpr uint16_t kZstdMethod = 1;

  if (input == nullptr || input_size < sizeof(magic) ||
      std::memcmp(input, magic, sizeof(magic)) != 0)
    return std::nullopt;

  size_t cursor = sizeof(magic);
  uint16_t version = 0;
  uint16_t method = 0;
  if (!read_value(input, input_size, cursor, version) ||
      !read_value(input, input_size, cursor, method))
    return std::nullopt;

  uint64_t total_file_size = input_size;
  uint64_t uncompressed_size = 0;
  switch (version) {
  case 1: {
    uint32_t size = 0;
    if (!read_value(input, input_size, cursor, size))
      return std::nullopt;
    uncompressed_size = size;
    break;
  }
  case 2: {
    uint32_t file_size = 0;
    uint32_t size = 0;
    if (!read_value(input, input_size, cursor, file_size) ||
        !read_value(input, input_size, cursor, size))
      return std::nullopt;
    total_file_size = file_size;
    uncompressed_size = size;
    break;
  }
  case 3:
    if (!read_value(input, input_size, cursor, total_file_size) ||
        !read_value(input, input_size, cursor, uncompressed_size))
      return std::nullopt;
    break;
  default:
    return std::nullopt;
  }

  uint64_t hash = 0;
  if (!read_value(input, input_size, cursor, hash) || total_file_size < cursor ||
      total_file_size > input_size || uncompressed_size > std::numeric_limits<size_t>::max())
    return std::nullopt;
  (void)hash;

  const size_t compressed_size = static_cast<size_t>(total_file_size) - cursor;
  // Grow the result only as the codec produces bytes. The container's claimed
  // size can be wrong. Version 1 also relies on the codec for the frame end.
  std::vector<char> decompressed;
  std::array<char, 64 * 1024> chunk;
  auto append = [&](size_t count) {
    if (count > uncompressed_size - decompressed.size())
      return false;
    decompressed.insert(decompressed.end(), chunk.data(), chunk.data() + count);
    return true;
  };
  size_t consumed = 0;
  if (method == kZstdMethod) {
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context(ZSTD_createDCtx(), ZSTD_freeDCtx);
    if (!context)
      return std::nullopt;
    ZSTD_inBuffer source{input + cursor, compressed_size, 0};
    size_t remaining;
    do {
      ZSTD_outBuffer output{chunk.data(), chunk.size(), 0};
      const size_t previous = source.pos;
      remaining = ZSTD_decompressStream(context.get(), &output, &source);
      if (ZSTD_isError(remaining) || !append(output.pos) ||
          (remaining && previous == source.pos && output.pos == 0))
        return std::nullopt;
    } while (remaining);
    consumed = source.pos;
  } else if (method == kZlibMethod) {
    z_stream stream{};
    if (inflateInit(&stream) != Z_OK)
      return std::nullopt;
    const std::unique_ptr<z_stream, decltype(&inflateEnd)> cleanup(&stream, inflateEnd);
    int status;
    do {
      const auto available = static_cast<uInt>(
          std::min<size_t>(compressed_size - consumed, std::numeric_limits<uInt>::max()));
      stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(input + cursor + consumed));
      stream.avail_in = available;
      stream.next_out = reinterpret_cast<Bytef *>(chunk.data());
      stream.avail_out = chunk.size();
      status = inflate(&stream, Z_NO_FLUSH);
      if ((status != Z_OK && status != Z_STREAM_END) || !append(chunk.size() - stream.avail_out))
        return std::nullopt;
      consumed += available - stream.avail_in;
    } while (status != Z_STREAM_END);
  } else {
    return std::nullopt;
  }
  if (decompressed.size() != uncompressed_size || (version != 1 && consumed != compressed_size))
    return std::nullopt;
  return DecompressedOffloadBundle{cursor + consumed, std::move(decompressed)};
}

} // namespace

Executable::Executable(const std::string &path) {
  util::FailureOr<std::vector<char>> maybeImage = detail::read_file_bytes(path);
  if (maybeImage.failed()) {
    read_failed_ = true;
    is_valid_ = false;
    return;
  }
  image_ = std::move(maybeImage.value());

  parse_image();
}

Executable::Executable(std::span<const uint8_t> image) {
  if (!image.empty())
    image_.assign(reinterpret_cast<const char *>(image.data()),
                  reinterpret_cast<const char *>(image.data()) + image.size());
  parse_image();
}

void Executable::parse_image() {
  if (image_.size() < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr)) {
    is_valid_ = false;
    return;
  }

  if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
    is_valid_ = false;
    return;
  }
  gpu_payload_recognized_ = ehdr.e_machine == EM_AMDGPU;

  // Standalone AMD GPU HSA device ELF.
  if (ehdr.e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA) {
    if (!load_program_headers(ehdr)) {
      is_valid_ = false;
      return;
    }
    load_device_elf();
    return;
  }

  // x86 host ELF is a container for .hip_fatbin, not a GPU load image. Its
  // host program headers do not describe the extracted device objects.
  if (ehdr.e_ident[EI_OSABI] == ELFOSABI_NONE || ehdr.e_ident[EI_OSABI] == ELFOSABI_GNU) {
    header_ = std::make_unique<HsaHeader>(ehdr);
    load_fat_binary();
    return;
  }

  is_valid_ = false;
}

Executable::~Executable() = default;

uint32_t Executable::num_code_objects(rj_code_target_id_t target) const {
  auto it = code_objs_by_target_.find(target);
  if (it == code_objs_by_target_.end())
    return 0;
  return static_cast<uint32_t>(it->second.size());
}

AmdGpuCodeObject *Executable::code_object(rj_code_target_id_t target, uint32_t index) {
  auto it = code_objs_by_target_.find(target);
  if (it == code_objs_by_target_.end() || index >= it->second.size())
    return nullptr;
  return it->second[index];
}

const AmdGpuCodeObject *Executable::code_object(rj_code_target_id_t target, uint32_t index) const {
  auto it = code_objs_by_target_.find(target);
  if (it == code_objs_by_target_.end() || index >= it->second.size())
    return nullptr;
  return it->second[index];
}

void Executable::load_device_elf() {
  auto co = std::make_unique<AmdGpuCodeObject>(reinterpret_cast<const uint8_t *>(image_.data()),
                                               image_.size());
  if (!co->is_valid()) {
    is_valid_ = false;
    return;
  }
  register_code_object(std::move(co));
}

void Executable::load_fat_binary() {
  // Parse section headers to find .hip_fatbin.
  const int num_shdrs = header_->numSectionHeaders();
  if (num_shdrs < 0) {
    is_valid_ = false;
    return;
  }
  const std::span image(reinterpret_cast<const uint8_t *>(image_.data()), image_.size());
  std::optional<std::vector<Elf64_Shdr>> section_hdrs = read_elf_table<Elf64_Shdr>(
      image, header_->sectionHeaderOff(), static_cast<uint64_t>(num_shdrs),
      header_->sectionHeaderEntrySize());
  if (!section_hdrs) {
    is_valid_ = false;
    return;
  }

  int shstrndx = header_->sectionHeaderStrIdx();
  if (shstrndx < 0 || static_cast<size_t>(shstrndx) >= section_hdrs->size()) {
    is_valid_ = false;
    return;
  }

  const Elf64_Shdr &shstrtab = (*section_hdrs)[shstrndx];
  if (!fits_in_bounds(shstrtab.sh_offset, shstrtab.sh_size, image_.size())) {
    is_valid_ = false;
    return;
  }
  const char *shstrtab_data = image_.data() + shstrtab.sh_offset;

  for (size_t section_index = 0; section_index < section_hdrs->size(); ++section_index) {
    const Elf64_Shdr &shdr = (*section_hdrs)[section_index];
    if (shdr.sh_type == SHT_NULL)
      continue;
    if (shdr.sh_name >= shstrtab.sh_size)
      continue;
    size_t max_len = shstrtab.sh_size - shdr.sh_name;
    std::string sec_name(shstrtab_data + shdr.sh_name,
                         strnlen(shstrtab_data + shdr.sh_name, max_len));

    if (sec_name == ".hip_fatbin") {
      gpu_payload_recognized_ = true;
      if (shdr.sh_type == SHT_NOBITS ||
          !fits_in_bounds(shdr.sh_offset, shdr.sh_size, image_.size())) {
        is_valid_ = false;
        return;
      }
      std::unique_ptr<char[]> sec_data(std::make_unique<char[]>(shdr.sh_size));
      if (shdr.sh_size != 0)
        std::memcpy(sec_data.get(), image_.data() + shdr.sh_offset, shdr.sh_size);
      auto section =
          std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr, section_index);
      load_hip_fatbin(*section);
      sections_.emplace_back(std::move(section));
      if (!is_valid_)
        return;
    }
  }
}

void Executable::load_hip_fatbin(const Section &fatbin_section) {
  const char *fatbin_data = fatbin_section.data();
  const size_t fatbin_size = fatbin_section.size();
  size_t offset = 0;
  while (offset < fatbin_size) {
    const char *data = fatbin_data + offset;
    const size_t remaining = fatbin_size - offset;
    if (remaining >= COMPRESSED_OFFLOAD_MAGIC_STR_SIZE &&
        std::memcmp(data, COMPRESSED_OFFLOAD_MAGIC_STR, COMPRESSED_OFFLOAD_MAGIC_STR_SIZE) == 0) {
      auto bundle = decompress_offload_bundle(data, remaining);
      if (!bundle || !load_clang_offload_bundle(bundle->data.data(), bundle->data.size())) {
        is_valid_ = false;
        return;
      }
      offset += bundle->compressed_file_size;
    } else {
      const size_t consumed = load_clang_offload_bundle(data, remaining);
      if (!consumed)
        return;
      offset += consumed;
    }
    while (offset < fatbin_size && fatbin_data[offset] == 0)
      ++offset;
  }
}

size_t Executable::load_clang_offload_bundle(const char *bundle_data, size_t bundle_size) {
  ClangOffloadBundleHeader bundle_hdr;
  size_t cursor = 0;
  if (!fits_in_bounds(cursor, sizeof(bundle_hdr.magic), bundle_size)) {
    is_valid_ = false;
    return 0;
  }
  std::memcpy(bundle_hdr.magic, bundle_data + cursor, sizeof(bundle_hdr.magic));
  cursor += sizeof(bundle_hdr.magic);

  if (std::memcmp(bundle_hdr.magic, CLANG_OFFLOAD_MAGIC_STR, CLANG_OFFLOAD_MAGIC_STR_SIZE)) {
    is_valid_ = false;
    return 0;
  }

  if (!read_value(bundle_data, bundle_size, cursor, bundle_hdr.num_code_objs)) {
    is_valid_ = false;
    return 0;
  }

  // The fixed prefix is offset, size, and bundle_entry_id_size, before the
  // variable-length bundle identifier.
  constexpr uint64_t kBundleInfoFixedBytes = 3 * sizeof(uint64_t);
  const uint64_t remaining = bundle_size - cursor;
  if (bundle_hdr.num_code_objs > remaining / kBundleInfoFixedBytes) {
    is_valid_ = false;
    return 0;
  }

  util::Logger::debug_print(__func__, ": Found .hip_fatbin with ", bundle_hdr.num_code_objs,
                            " code objects");

  std::vector<ClangOffloadBundleInfo> infos(bundle_hdr.num_code_objs);
  size_t bundle_end = 0;
  for (auto &info : infos) {
    if (!read_value(bundle_data, bundle_size, cursor, info.offset)) {
      is_valid_ = false;
      return 0;
    }
    if (!read_value(bundle_data, bundle_size, cursor, info.size)) {
      is_valid_ = false;
      return 0;
    }
    if (!read_value(bundle_data, bundle_size, cursor, info.bundle_entry_id_size)) {
      is_valid_ = false;
      return 0;
    }
    if (!fits_in_bounds(cursor, info.bundle_entry_id_size, bundle_size)) {
      is_valid_ = false;
      return 0;
    }
    info.bundle_entry_id.resize(info.bundle_entry_id_size, '\0');
    std::memcpy(info.bundle_entry_id.data(), bundle_data + cursor, info.bundle_entry_id_size);
    cursor += static_cast<size_t>(info.bundle_entry_id_size);

    if (!fits_in_bounds(info.offset, info.size, bundle_size)) {
      is_valid_ = false;
      return 0;
    }
    bundle_end = std::max(bundle_end, static_cast<size_t>(info.offset + info.size));
    auto parts_view = info.bundle_entry_id | std::views::split('-');
    std::vector<std::string> parts;
    for (const auto &v : parts_view)
      parts.emplace_back(v.begin(), v.end());

    if (parts.size() <= 5) {
      is_valid_ = false;
      return 0;
    }

    if (parts[0] == CLANG_OFFLOAD_KIND_HIP || parts[0] == CLANG_OFFLOAD_KIND_HIPV4) {
      util::Logger::debug_print(__func__, ": Fatbin has target triple ", parts[5]);
      auto co = std::make_unique<AmdGpuCodeObject>(
          reinterpret_cast<const uint8_t *>(bundle_data + info.offset),
          static_cast<size_t>(info.size), parts[0], parts[5]);
      if (!co->is_valid()) {
        is_valid_ = false;
        return 0;
      }
      register_code_object(std::move(co));
    } else if (parts[0] == CLANG_OFFLOAD_KIND_HOST) {
      continue;
    } else {
      is_valid_ = false;
      return 0;
    }
  }
  return std::max(cursor, bundle_end);
}

void Executable::register_code_object(std::unique_ptr<AmdGpuCodeObject> co) {
  auto target = co->target_id();
  code_objs_by_target_[target].push_back(co.get());
  owned_code_objs_.push_back(std::move(co));
}

} // namespace rocjitsu
