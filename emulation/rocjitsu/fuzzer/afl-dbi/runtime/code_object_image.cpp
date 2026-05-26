#include "code_object_image.h"

#include "md5_util.h"
#include "rocjitsu/code/amdgpu_elf.h"

#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <zstd.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

namespace rocjitsu::fuzzer::afl_dbi {
namespace {

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t kBundleMagicSize = sizeof(kBundleMagic) - 1;
constexpr char kCcobMagic[] = "CCOB";
constexpr size_t kCcobMagicSize = sizeof(kCcobMagic) - 1;
constexpr uint64_t kMaxInMemoryImageSize = 256ull * 1024ull * 1024ull;
constexpr uint64_t kMaxBundleEntryCount = 256;
constexpr uint64_t kMaxBundleIdSize = 4096;
constexpr uint16_t kCcobVersion3 = 3;
constexpr uint16_t kCcobZstdMethod = 1;
constexpr uint32_t kHipfMagic = 0x48495046; // HIPF, stored as FPIH on little-endian hosts.
constexpr uint32_t kHipkMagic = 0x4b504948; // HIPK, stored as HIPK on little-endian hosts.
constexpr uint32_t kHipFatBinaryWrapperVersion = 1;
constexpr unsigned kMaxWrapperUnwrapDepth = 2;

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t *out) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  *out = lhs + rhs;
  return true;
}

uint32_t read_u32(const uint8_t *p) {
  uint32_t v = 0;
  memcpy(&v, p, sizeof(v));
  return v;
}

uint64_t read_u64(const uint8_t *p) {
  uint64_t v = 0;
  memcpy(&v, p, sizeof(v));
  return v;
}

void append_u32(std::vector<uint8_t> *data, uint32_t value) {
  for (unsigned i = 0; i < sizeof(value); ++i)
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffu));
}

void append_u64(std::vector<uint8_t> *data, uint64_t value) {
  for (unsigned i = 0; i < sizeof(value); ++i)
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffu));
}

struct ClangBundleEntry {
  std::string id;
  std::span<const uint8_t> image;
};

struct OwnedBundleEntry {
  std::string id;
  std::vector<uint8_t> image;
};

struct DeviceImageSelector {
  std::string_view id;
  std::optional<uint32_t> index;
  uint32_t next_index = 0;
};

struct HipFatBinaryWrapper {
  uint32_t magic = 0;
  uint32_t version = 0;
  const void *binary = nullptr;
  const void *reserved = nullptr;
};

std::vector<uint8_t> copy_raw_elf_from_memory(const uint8_t *base) {
  if (base == nullptr || !is_raw_elf_image({base, 4}))
    return {};

  rocjitsu::Elf64_Ehdr ehdr{};
  memcpy(&ehdr, base, sizeof(ehdr));
  if (ehdr.e_ehsize != sizeof(rocjitsu::Elf64_Ehdr))
    return {};

  uint64_t size = sizeof(rocjitsu::Elf64_Ehdr);
  auto include_range = [&](uint64_t offset, uint64_t range_size) {
    uint64_t end = 0;
    if (!checked_add(offset, range_size, &end) || end > kMaxInMemoryImageSize)
      return false;
    size = std::max(size, end);
    return true;
  };

  if (ehdr.e_phnum != 0) {
    if (ehdr.e_phentsize != sizeof(rocjitsu::Elf64_Phdr) || ehdr.e_phnum > 1024)
      return {};
    uint64_t phdr_bytes = 0;
    if (!checked_add(0, static_cast<uint64_t>(ehdr.e_phnum) * ehdr.e_phentsize, &phdr_bytes) ||
        !include_range(ehdr.e_phoff, phdr_bytes))
      return {};

    for (uint16_t i = 0; i < ehdr.e_phnum; ++i) {
      rocjitsu::Elf64_Phdr phdr{};
      memcpy(&phdr, base + ehdr.e_phoff + static_cast<uint64_t>(i) * ehdr.e_phentsize,
             sizeof(phdr));
      if (phdr.p_filesz != 0 && !include_range(phdr.p_offset, phdr.p_filesz))
        return {};
    }
  }

  if (ehdr.e_shnum != 0) {
    if (ehdr.e_shentsize != sizeof(rocjitsu::Elf64_Shdr))
      return {};
    uint64_t shdr_bytes = 0;
    if (!checked_add(0, static_cast<uint64_t>(ehdr.e_shnum) * ehdr.e_shentsize, &shdr_bytes) ||
        !include_range(ehdr.e_shoff, shdr_bytes))
      return {};

    for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
      rocjitsu::Elf64_Shdr shdr{};
      memcpy(&shdr, base + ehdr.e_shoff + static_cast<uint64_t>(i) * ehdr.e_shentsize,
             sizeof(shdr));
      if (shdr.sh_type != rocjitsu::SHT_NOBITS && shdr.sh_size != 0 &&
          !include_range(shdr.sh_offset, shdr.sh_size))
        return {};
    }
  }

  return {base, base + size};
}

std::optional<uint64_t> clang_bundle_memory_size(const uint8_t *base) {
  if (base == nullptr || memcmp(base, kBundleMagic, kBundleMagicSize) != 0)
    return std::nullopt;

  uint64_t size = kBundleMagicSize + sizeof(uint64_t);
  uint64_t pos = kBundleMagicSize;
  const uint64_t count = read_u64(base + pos);
  pos += sizeof(uint64_t);
  if (count > kMaxBundleEntryCount)
    return std::nullopt;

  auto include_range = [&](uint64_t offset, uint64_t range_size) {
    uint64_t end = 0;
    if (!checked_add(offset, range_size, &end) || end > kMaxInMemoryImageSize)
      return false;
    size = std::max(size, end);
    return true;
  };

  for (uint64_t i = 0; i < count; ++i) {
    if (!include_range(pos, 3 * sizeof(uint64_t)))
      return std::nullopt;
    const uint64_t offset = read_u64(base + pos);
    pos += sizeof(uint64_t);
    const uint64_t image_size = read_u64(base + pos);
    pos += sizeof(uint64_t);
    const uint64_t id_size = read_u64(base + pos);
    pos += sizeof(uint64_t);
    if (id_size > kMaxBundleIdSize || !include_range(pos, id_size) ||
        !include_range(offset, image_size))
      return std::nullopt;
    pos += id_size;
  }

  return size;
}

std::optional<uint64_t> clang_bundle_image_size(std::span<const uint8_t> image) {
  if (!is_clang_offload_bundle(image))
    return std::nullopt;

  uint64_t size = kBundleMagicSize + sizeof(uint64_t);
  size_t pos = kBundleMagicSize;
  const uint64_t count = read_u64(image.data() + pos);
  pos += sizeof(uint64_t);
  if (count > kMaxBundleEntryCount)
    return std::nullopt;

  auto include_range = [&](uint64_t offset, uint64_t range_size) {
    uint64_t end = 0;
    if (!checked_add(offset, range_size, &end) || end > image.size())
      return false;
    size = std::max(size, end);
    return true;
  };

  for (uint64_t i = 0; i < count; ++i) {
    if (!include_range(pos, 3 * sizeof(uint64_t)))
      return std::nullopt;
    const uint64_t offset = read_u64(image.data() + pos);
    pos += sizeof(uint64_t);
    const uint64_t image_size = read_u64(image.data() + pos);
    pos += sizeof(uint64_t);
    const uint64_t id_size = read_u64(image.data() + pos);
    pos += sizeof(uint64_t);
    if (id_size > kMaxBundleIdSize || !include_range(pos, id_size) ||
        !include_range(offset, image_size))
      return std::nullopt;
    pos += static_cast<size_t>(id_size);
  }

  return size;
}

struct CcobHeader {
  uint16_t version = 0;
  uint16_t method = 0;
  uint64_t header_size = 0;
  uint64_t total_size = 0;
  uint64_t uncompressed_size = 0;
  uint64_t hash = 0;
};

std::optional<CcobHeader> parse_ccob_header(std::span<const uint8_t> image) {
  if (!is_ccob_image(image) || image.size() < 24)
    return std::nullopt;

  const uint32_t version_method = read_u32(image.data() + 4);
  CcobHeader header;
  header.version = static_cast<uint16_t>(version_method & 0xffffu);
  header.method = static_cast<uint16_t>(version_method >> 16);

  if (header.version == kCcobVersion3) {
    if (image.size() < 32)
      return std::nullopt;
    header.header_size = 32;
    header.total_size = read_u64(image.data() + 8);
    header.uncompressed_size = read_u64(image.data() + 16);
    header.hash = read_u64(image.data() + 24);
  } else {
    header.header_size = 24;
    header.total_size = read_u32(image.data() + 8);
    header.uncompressed_size = read_u32(image.data() + 12);
    header.hash = read_u64(image.data() + 16);
  }

  if (header.method != kCcobZstdMethod || header.total_size < header.header_size ||
      header.total_size > image.size() || header.total_size > kMaxInMemoryImageSize ||
      header.uncompressed_size > kMaxInMemoryImageSize)
    return std::nullopt;
  return header;
}

std::optional<uint64_t> ccob_memory_size(const uint8_t *base) {
  if (base == nullptr || memcmp(base, kCcobMagic, kCcobMagicSize) != 0)
    return std::nullopt;

  const uint32_t version_method = read_u32(base + 4);
  const uint16_t version = static_cast<uint16_t>(version_method & 0xffffu);
  uint64_t total_size = 0;
  if (version == kCcobVersion3)
    total_size = read_u64(base + 8);
  else
    total_size = read_u32(base + 8);
  if (total_size == 0 || total_size > kMaxInMemoryImageSize)
    return std::nullopt;
  return total_size;
}

const uint8_t *hip_fatbin_wrapper_payload(const uint8_t *base) {
  if (base == nullptr)
    return nullptr;

  HipFatBinaryWrapper wrapper{};
  memcpy(&wrapper, base, sizeof(wrapper));
  if ((wrapper.magic != kHipfMagic && wrapper.magic != kHipkMagic) ||
      wrapper.version != kHipFatBinaryWrapperVersion || wrapper.binary == nullptr)
    return nullptr;
  if (wrapper.binary == base)
    return nullptr;
  return static_cast<const uint8_t *>(wrapper.binary);
}

std::optional<std::vector<uint8_t>> decompress_ccob(std::span<const uint8_t> image) {
  const std::optional<CcobHeader> header = parse_ccob_header(image);
  if (!header)
    return std::nullopt;

  std::vector<uint8_t> decompressed(static_cast<size_t>(header->uncompressed_size));
  const uint8_t *compressed = image.data() + header->header_size;
  const size_t compressed_size = static_cast<size_t>(header->total_size - header->header_size);
  const size_t written =
      ZSTD_decompress(decompressed.data(), decompressed.size(), compressed, compressed_size);
  if (ZSTD_isError(written) || written != decompressed.size())
    return std::nullopt;
  return decompressed;
}

bool is_amdgpu_bundle_id(std::string_view id) {
  return id.find("amdgcn") != std::string_view::npos;
}

std::string extract_target_id_from_bundle_id(std::string_view id) {
  const size_t gfx_pos = id.rfind("gfx");
  if (gfx_pos == std::string_view::npos)
    return {};

  size_t end = gfx_pos;
  while (end < id.size()) {
    const unsigned char c = static_cast<unsigned char>(id[end]);
    if (std::isalnum(c) || c == ':' || c == '+' || c == '-' || c == '_') {
      ++end;
      continue;
    }
    break;
  }
  return normalize_amdgpu_target_id(id.substr(gfx_pos, end - gfx_pos));
}

std::vector<std::string> compatible_target_ids(std::string_view agent_target_id) {
  const std::string normalized = normalize_amdgpu_target_id(agent_target_id);
  if (normalized.empty())
    return {};

  const size_t first_colon = normalized.find(':');
  const std::string processor = normalized.substr(0, first_colon);
  if (processor.empty())
    return {};
  if (first_colon == std::string::npos)
    return {processor};

  std::vector<std::string> features;
  size_t pos = first_colon + 1;
  while (pos < normalized.size()) {
    const size_t colon = normalized.find(':', pos);
    std::string feature = normalized.substr(pos, colon - pos);
    if (!feature.empty())
      features.push_back(std::move(feature));
    if (colon == std::string::npos)
      break;
    pos = colon + 1;
  }

  std::vector<std::string> out;
  const size_t n = features.size();
  if (n == 0)
    return {processor};
  if (n >= sizeof(unsigned) * 8)
    return {normalized, processor};

  const unsigned full_mask = (1u << n) - 1;
  for (unsigned mask = full_mask;; --mask) {
    std::string candidate = processor;
    for (size_t i = 0; i < n; ++i) {
      if (mask & (1u << (n - 1 - i))) {
        candidate += ':';
        candidate += features[i];
      }
    }
    out.push_back(std::move(candidate));
    if (mask == 0)
      break;
  }
  return out;
}

std::optional<std::vector<ClangBundleEntry>>
parse_clang_bundle_entries(std::span<const uint8_t> image) {
  if (!is_clang_offload_bundle(image))
    return std::nullopt;

  size_t pos = kBundleMagicSize;
  const uint64_t count = read_u64(image.data() + pos);
  pos += sizeof(uint64_t);
  if (count > kMaxBundleEntryCount)
    return std::nullopt;

  std::vector<ClangBundleEntry> entries;
  entries.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    if (pos + 3 * sizeof(uint64_t) > image.size())
      return std::nullopt;
    const uint64_t offset = read_u64(image.data() + pos);
    pos += sizeof(uint64_t);
    const uint64_t size = read_u64(image.data() + pos);
    pos += sizeof(uint64_t);
    const uint64_t id_size = read_u64(image.data() + pos);
    pos += sizeof(uint64_t);
    if (id_size > image.size() || id_size > kMaxBundleIdSize || pos + id_size > image.size())
      return std::nullopt;
    std::string id(reinterpret_cast<const char *>(image.data() + pos),
                   static_cast<size_t>(id_size));
    pos += static_cast<size_t>(id_size);
    if (offset > image.size() || size > image.size() - offset)
      return std::nullopt;
    entries.push_back(
        {std::move(id), {image.data() + offset, static_cast<size_t>(size)}});
  }

  return entries;
}

std::vector<uint8_t> make_clang_bundle(std::span<const OwnedBundleEntry> entries) {
  std::vector<uint8_t> bundle;
  bundle.insert(bundle.end(), kBundleMagic, kBundleMagic + kBundleMagicSize);
  append_u64(&bundle, entries.size());

  uint64_t offset = bundle.size();
  for (const OwnedBundleEntry &entry : entries)
    offset += 3 * sizeof(uint64_t) + entry.id.size();

  uint64_t data_offset = offset;
  for (const OwnedBundleEntry &entry : entries) {
    append_u64(&bundle, data_offset);
    append_u64(&bundle, entry.image.size());
    append_u64(&bundle, entry.id.size());
    bundle.insert(bundle.end(), entry.id.begin(), entry.id.end());
    data_offset += entry.image.size();
  }

  for (const OwnedBundleEntry &entry : entries)
    bundle.insert(bundle.end(), entry.image.begin(), entry.image.end());
  return bundle;
}

std::optional<std::vector<uint8_t>> make_ccob(const CcobHeader &header,
                                              std::span<const uint8_t> uncompressed) {
  if (header.method != kCcobZstdMethod)
    return std::nullopt;
  if (header.version != kCcobVersion3 && uncompressed.size() > std::numeric_limits<uint32_t>::max())
    return std::nullopt;

  std::vector<uint8_t> compressed(ZSTD_compressBound(uncompressed.size()));
  const size_t compressed_size =
      ZSTD_compress(compressed.data(), compressed.size(), uncompressed.data(),
                    uncompressed.size(), /*compressionLevel=*/1);
  if (ZSTD_isError(compressed_size))
    return std::nullopt;
  compressed.resize(compressed_size);

  // CCOB stores the first 64 bits of the MD5 digest of the uncompressed bundle
  // for verification and loader caching.
  std::optional<uint64_t> hash = truncated_md5_hash64(uncompressed);
  if (!hash)
    return std::nullopt;

  const uint64_t header_size = header.version == kCcobVersion3 ? 32 : 24;
  const uint64_t total_size = header_size + compressed.size();
  if (header.version != kCcobVersion3 && total_size > std::numeric_limits<uint32_t>::max())
    return std::nullopt;

  std::vector<uint8_t> ccob;
  ccob.insert(ccob.end(), kCcobMagic, kCcobMagic + kCcobMagicSize);
  append_u32(&ccob,
             static_cast<uint32_t>(header.version) |
                 (static_cast<uint32_t>(header.method) << 16));
  if (header.version == kCcobVersion3) {
    append_u64(&ccob, total_size);
    append_u64(&ccob, uncompressed.size());
    append_u64(&ccob, *hash);
  } else {
    append_u32(&ccob, static_cast<uint32_t>(total_size));
    append_u32(&ccob, static_cast<uint32_t>(uncompressed.size()));
    append_u64(&ccob, *hash);
  }
  ccob.insert(ccob.end(), compressed.begin(), compressed.end());
  return ccob;
}

bool append_clang_bundle_entries(std::span<const uint8_t> image, std::string_view prefix,
                                 std::vector<DeviceImage> *out) {
  if (!is_clang_offload_bundle(image) || out == nullptr)
    return false;

  std::optional<std::vector<ClangBundleEntry>> entries = parse_clang_bundle_entries(image);
  if (!entries)
    return false;

  for (const ClangBundleEntry &entry : *entries) {
    const std::string &id = entry.id;
    const std::string full_id = prefix.empty() ? id : std::string(prefix) + ":" + id;
    std::string target_id = extract_target_id_from_bundle_id(id);
    if (!is_amdgpu_bundle_id(id))
      continue;

    if (is_raw_elf_image(entry.image)) {
      out->push_back({full_id, target_id, {entry.image.begin(), entry.image.end()}});
      continue;
    }

    if (is_clang_offload_bundle(entry.image)) {
      std::vector<DeviceImage> nested = extract_device_images(entry.image);
      for (DeviceImage &device_image : nested) {
        device_image.id = full_id + ":" + device_image.id;
        if (device_image.target_id.empty())
          device_image.target_id = target_id;
        out->push_back(std::move(device_image));
      }
      continue;
    }

    if (is_ccob_image(entry.image)) {
      std::vector<DeviceImage> nested = extract_device_images(entry.image);
      for (DeviceImage &device_image : nested) {
        device_image.id = full_id + ":" + device_image.id;
        if (device_image.target_id.empty())
          device_image.target_id = target_id;
        out->push_back(std::move(device_image));
      }
    }
  }
  return true;
}

std::optional<size_t> find_next_container_magic(std::span<const uint8_t> image, size_t pos) {
  while (pos < image.size()) {
    if (pos + kCcobMagicSize <= image.size() &&
        memcmp(image.data() + pos, kCcobMagic, kCcobMagicSize) == 0)
      return pos;
    if (pos + kBundleMagicSize <= image.size() &&
        memcmp(image.data() + pos, kBundleMagic, kBundleMagicSize) == 0)
      return pos;
    ++pos;
  }
  return std::nullopt;
}

std::string prefixed_id(std::string_view prefix, std::string_view id) {
  if (prefix.empty())
    return std::string(id);
  std::string out(prefix);
  out.push_back(':');
  out.append(id.data(), id.size());
  return out;
}

bool selector_matches(DeviceImageSelector *selector, std::string_view id) {
  if (selector == nullptr)
    return false;

  const uint32_t current_index = selector->next_index++;
  if (id != selector->id)
    return false;
  return !selector->index || current_index == *selector->index;
}

bool append_fatbin_container_images(std::span<const uint8_t> image, std::string_view prefix,
                                    std::vector<DeviceImage> *out) {
  if (out == nullptr || image.empty())
    return false;

  bool parsed_any = false;
  size_t pos = 0;
  while (std::optional<size_t> magic = find_next_container_magic(image, pos)) {
    std::span<const uint8_t> tail = image.subspan(*magic);
    if (is_clang_offload_bundle(tail)) {
      std::optional<uint64_t> size = clang_bundle_image_size(tail);
      if (!size) {
        pos = *magic + 1;
        continue;
      }
      std::span<const uint8_t> bundle = tail.first(static_cast<size_t>(*size));
      parsed_any = append_clang_bundle_entries(bundle, prefix, out) || parsed_any;
      pos = *magic + static_cast<size_t>(*size);
      continue;
    }

    std::optional<CcobHeader> header = parse_ccob_header(tail);
    if (!header) {
      pos = *magic + 1;
      continue;
    }
    std::span<const uint8_t> ccob = tail.first(static_cast<size_t>(header->total_size));
    std::optional<std::vector<uint8_t>> decompressed = decompress_ccob(ccob);
    if (decompressed) {
      const std::string ccob_prefix = prefixed_id(prefix, "ccob");
      if (is_raw_elf_image(*decompressed)) {
        out->push_back({prefixed_id(prefix, "ccob:raw-elf"), "", std::move(*decompressed)});
        parsed_any = true;
      } else {
        parsed_any =
            append_fatbin_container_images(*decompressed, ccob_prefix, out) || parsed_any;
      }
    }
    pos = *magic + static_cast<size_t>(header->total_size);
  }
  return parsed_any;
}

std::optional<std::vector<uint8_t>>
rebuild_image_graph(std::span<const uint8_t> image, std::string_view prefix,
                    DeviceImageSelector *selector, std::span<const uint8_t> replacement,
                    bool *replaced);

std::optional<std::vector<uint8_t>>
rebuild_clang_bundle(std::span<const uint8_t> image, std::string_view prefix,
                     DeviceImageSelector *selector, std::span<const uint8_t> replacement,
                     bool *replaced) {
  if (replaced == nullptr || *replaced)
    return std::nullopt;

  std::optional<std::vector<ClangBundleEntry>> entries =
      parse_clang_bundle_entries(image);
  if (!entries)
    return std::nullopt;

  const bool replaced_before = *replaced;
  std::vector<OwnedBundleEntry> rebuilt_entries;
  rebuilt_entries.reserve(entries->size());
  for (const ClangBundleEntry &entry : *entries) {
    OwnedBundleEntry rebuilt;
    rebuilt.id = entry.id;
    const std::string full_id = prefixed_id(prefix, entry.id);
    if (!*replaced && is_amdgpu_bundle_id(entry.id) && is_raw_elf_image(entry.image)) {
      if (selector_matches(selector, full_id)) {
        rebuilt.image.assign(replacement.begin(), replacement.end());
        *replaced = true;
      } else {
        rebuilt.image.assign(entry.image.begin(), entry.image.end());
      }
    } else if (!*replaced && is_amdgpu_bundle_id(entry.id)) {
      std::optional<std::vector<uint8_t>> nested =
          rebuild_image_graph(entry.image, full_id, selector, replacement, replaced);
      if (nested)
        rebuilt.image = std::move(*nested);
      else
        rebuilt.image.assign(entry.image.begin(), entry.image.end());
    } else {
      rebuilt.image.assign(entry.image.begin(), entry.image.end());
    }
    rebuilt_entries.push_back(std::move(rebuilt));
  }

  if (*replaced == replaced_before)
    return std::nullopt;
  return make_clang_bundle(rebuilt_entries);
}

std::optional<std::vector<uint8_t>>
rebuild_ccob_container(std::span<const uint8_t> image, std::string_view prefix,
                       DeviceImageSelector *selector, std::span<const uint8_t> replacement,
                       bool *replaced) {
  if (replaced == nullptr || *replaced)
    return std::nullopt;

  std::optional<CcobHeader> header = parse_ccob_header(image);
  if (!header)
    return std::nullopt;
  const std::span<const uint8_t> ccob = image.first(static_cast<size_t>(header->total_size));
  std::optional<std::vector<uint8_t>> decompressed = decompress_ccob(ccob);
  if (!decompressed)
    return std::nullopt;

  const bool replaced_before = *replaced;
  const std::string ccob_prefix = prefixed_id(prefix, "ccob");
  std::optional<std::vector<uint8_t>> rebuilt =
      rebuild_image_graph(*decompressed, ccob_prefix, selector, replacement, replaced);
  if (!rebuilt || *replaced == replaced_before)
    return std::nullopt;
  return make_ccob(*header, *rebuilt);
}

std::optional<size_t> parsed_container_size(std::span<const uint8_t> image) {
  if (is_clang_offload_bundle(image)) {
    std::optional<uint64_t> size = clang_bundle_image_size(image);
    if (size && *size <= image.size())
      return static_cast<size_t>(*size);
    return std::nullopt;
  }

  if (std::optional<CcobHeader> header = parse_ccob_header(image))
    return static_cast<size_t>(header->total_size);
  return std::nullopt;
}

std::optional<std::vector<uint8_t>>
rebuild_container(std::span<const uint8_t> image, std::string_view prefix,
                  DeviceImageSelector *selector, std::span<const uint8_t> replacement,
                  bool *replaced) {
  if (is_clang_offload_bundle(image))
    return rebuild_clang_bundle(image, prefix, selector, replacement, replaced);
  if (is_ccob_image(image))
    return rebuild_ccob_container(image, prefix, selector, replacement, replaced);
  return std::nullopt;
}

std::optional<std::vector<uint8_t>>
rebuild_image_graph(std::span<const uint8_t> image, std::string_view prefix,
                    DeviceImageSelector *selector, std::span<const uint8_t> replacement,
                    bool *replaced) {
  if (replaced == nullptr || *replaced)
    return std::nullopt;

  if (is_raw_elf_image(image)) {
    if (!selector_matches(selector, prefixed_id(prefix, "raw-elf")))
      return std::nullopt;
    *replaced = true;
    return std::vector<uint8_t>{replacement.begin(), replacement.end()};
  }

  const bool replaced_before = *replaced;
  std::vector<uint8_t> rebuilt;
  size_t copied = 0;
  size_t scan = 0;
  bool parsed_any = false;
  while (std::optional<size_t> magic = find_next_container_magic(image, scan)) {
    std::span<const uint8_t> tail = image.subspan(*magic);
    std::optional<size_t> container_size = parsed_container_size(tail);
    if (!container_size) {
      scan = *magic + 1;
      continue;
    }

    parsed_any = true;
    rebuilt.insert(rebuilt.end(), image.begin() + copied, image.begin() + *magic);
    std::span<const uint8_t> container = tail.first(*container_size);
    const bool replaced_before_container = *replaced;
    std::optional<std::vector<uint8_t>> rebuilt_container =
        rebuild_container(container, prefix, selector, replacement, replaced);
    if (rebuilt_container && *replaced != replaced_before_container)
      rebuilt.insert(rebuilt.end(), rebuilt_container->begin(), rebuilt_container->end());
    else
      rebuilt.insert(rebuilt.end(), container.begin(), container.end());

    copied = *magic + *container_size;
    scan = copied;
  }

  if (!parsed_any || *replaced == replaced_before)
    return std::nullopt;

  rebuilt.insert(rebuilt.end(), image.begin() + copied, image.end());
  return rebuilt;
}

std::vector<uint8_t> copy_module_data_image_impl(const uint8_t *base, unsigned depth) {
  if (base == nullptr || depth > kMaxWrapperUnwrapDepth)
    return {};

  std::vector<uint8_t> copied = copy_raw_elf_from_memory(base);
  if (!copied.empty())
    return copied;
  if (std::optional<uint64_t> size = clang_bundle_memory_size(base))
    return {base, base + *size};
  if (std::optional<uint64_t> size = ccob_memory_size(base))
    return {base, base + *size};
  if (const uint8_t *payload = hip_fatbin_wrapper_payload(base))
    return copy_module_data_image_impl(payload, depth + 1);
  return {};
}

} // namespace

bool is_raw_elf_image(std::span<const uint8_t> image) {
  return image.size() >= 4 && image[0] == 0x7f && image[1] == 'E' && image[2] == 'L' &&
         image[3] == 'F';
}

bool is_clang_offload_bundle(std::span<const uint8_t> image) {
  return image.size() >= kBundleMagicSize + sizeof(uint64_t) &&
         memcmp(image.data(), kBundleMagic, kBundleMagicSize) == 0;
}

bool is_ccob_image(std::span<const uint8_t> image) {
  return image.size() >= kCcobMagicSize && memcmp(image.data(), kCcobMagic, kCcobMagicSize) == 0;
}

bool is_ccob_memory(const uint8_t *base) {
  return base != nullptr && memcmp(base, kCcobMagic, kCcobMagicSize) == 0;
}

std::vector<uint8_t> read_file_bytes(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    return {};
  const auto size = file.tellg();
  if (size <= 0)
    return {};
  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!file)
    return {};
  return data;
}

std::vector<uint8_t> read_fd_bytes(int fd, uint64_t offset, uint64_t size) {
  if (fd < 0 || size == 0 || size > kMaxInMemoryImageSize ||
      offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
    return {};

  std::vector<uint8_t> data(static_cast<size_t>(size));
  size_t read_bytes = 0;
  while (read_bytes < data.size()) {
    const uint64_t next_offset = offset + read_bytes;
    if (next_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
      return {};
    const ssize_t chunk =
        pread(fd, data.data() + read_bytes, data.size() - read_bytes,
              static_cast<off_t>(next_offset));
    if (chunk < 0)
      return {};
    if (chunk == 0)
      break;
    read_bytes += static_cast<size_t>(chunk);
  }
  data.resize(read_bytes);
  return data;
}

std::vector<uint8_t> copy_module_data_image(const void *image) {
  return copy_module_data_image_impl(static_cast<const uint8_t *>(image), /*depth=*/0);
}

std::vector<DeviceImage> extract_device_images(std::span<const uint8_t> image) {
  if (is_raw_elf_image(image))
    return {DeviceImage{"raw-elf", "", {image.begin(), image.end()}}};

  std::vector<DeviceImage> out;
  (void)append_fatbin_container_images(image, "", &out);
  for (size_t i = 0; i < out.size(); ++i)
    out[i].index = static_cast<uint32_t>(
        std::min<size_t>(i, std::numeric_limits<uint32_t>::max()));
  return out;
}

CodeObjectImageSummary summarize_code_object_image(std::span<const uint8_t> image) {
  CodeObjectImageSummary summary;
  summary.top_level_raw_elf = is_raw_elf_image(image);
  summary.top_level_bundle = is_clang_offload_bundle(image);
  summary.top_level_ccob = is_ccob_image(image);

  const std::vector<DeviceImage> device_images = extract_device_images(image);
  summary.device_image_count = static_cast<uint32_t>(
      std::min<size_t>(device_images.size(), std::numeric_limits<uint32_t>::max()));
  if (summary.top_level_ccob) {
    if (std::optional<CcobHeader> header = parse_ccob_header(image)) {
      std::span<const uint8_t> ccob = image.first(static_cast<size_t>(header->total_size));
      if (std::optional<std::vector<uint8_t>> decompressed = decompress_ccob(ccob)) {
        const std::vector<DeviceImage> ccob_images = extract_device_images(*decompressed);
        summary.top_level_ccob_device_image_count = static_cast<uint32_t>(
            std::min<size_t>(ccob_images.size(), std::numeric_limits<uint32_t>::max()));
      }
    }
  }
  summary.raw_elf_bypass_drops_sibling_payloads =
      !summary.top_level_raw_elf && summary.device_image_count > 1;
  return summary;
}

std::optional<std::vector<uint8_t>>
rebuild_code_object_image_with_replaced_device_image(std::span<const uint8_t> image,
                                                     std::string_view device_image_id,
                                                     std::span<const uint8_t> replacement) {
  if (image.empty() || device_image_id.empty() || replacement.empty())
    return std::nullopt;

  DeviceImageSelector selector{device_image_id, std::nullopt};
  bool replaced = false;
  return rebuild_image_graph(image, "", &selector, replacement, &replaced);
}

std::optional<std::vector<uint8_t>>
rebuild_code_object_image_with_replaced_device_image(std::span<const uint8_t> image,
                                                     const DeviceImage &device_image,
                                                     std::span<const uint8_t> replacement) {
  if (image.empty() || device_image.id.empty() || replacement.empty())
    return std::nullopt;

  DeviceImageSelector selector{device_image.id, device_image.index};
  bool replaced = false;
  return rebuild_image_graph(image, "", &selector, replacement, &replaced);
}

std::optional<std::vector<uint8_t>>
rebuild_ccob_with_replaced_device_image(std::span<const uint8_t> image,
                                        std::string_view device_image_id,
                                        std::span<const uint8_t> replacement) {
  if (!is_ccob_image(image))
    return std::nullopt;
  return rebuild_code_object_image_with_replaced_device_image(image, device_image_id,
                                                             replacement);
}

std::string normalize_amdgpu_target_id(std::string_view target_id) {
  constexpr std::string_view kPrefix = "amdgcn-amd-amdhsa--";
  if (target_id.substr(0, kPrefix.size()) == kPrefix)
    target_id.remove_prefix(kPrefix.size());

  std::string out(target_id);
  for (size_t pos = out.find("-xnack"); pos != std::string::npos;
       pos = out.find("-xnack", pos + 1)) {
    out[pos] = ':';
  }
  for (size_t pos = out.find("-sramecc"); pos != std::string::npos;
       pos = out.find("-sramecc", pos + 1)) {
    out[pos] = ':';
  }
  return out;
}

std::vector<DeviceImage> order_device_images_for_target(std::span<const DeviceImage> images,
                                                        std::string_view agent_target_id) {
  std::vector<DeviceImage> ordered;
  if (images.empty())
    return ordered;

  const std::vector<std::string> compatible = compatible_target_ids(agent_target_id);
  std::vector<bool> used(images.size(), false);
  for (const std::string &candidate : compatible) {
    for (size_t i = 0; i < images.size(); ++i) {
      if (used[i])
        continue;
      if (normalize_amdgpu_target_id(images[i].target_id) == candidate) {
        ordered.push_back(images[i]);
        used[i] = true;
      }
    }
  }

  for (size_t i = 0; i < images.size(); ++i) {
    if (!used[i])
      ordered.push_back(images[i]);
  }
  return ordered;
}

} // namespace rocjitsu::fuzzer::afl_dbi
