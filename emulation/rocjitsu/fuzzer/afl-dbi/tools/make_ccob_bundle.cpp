#include "md5_util.h"

#include <zstd.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";

void append_u32(std::vector<uint8_t> *data, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffu));
}

void append_u64(std::vector<uint8_t> *data, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffu));
}

uint64_t read_u64(const uint8_t *p) {
  uint64_t value = 0;
  memcpy(&value, p, sizeof(value));
  return value;
}

std::vector<uint8_t> read_file(const char *path) {
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

bool write_file(const char *path, const std::vector<uint8_t> &data) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    return false;
  file.write(reinterpret_cast<const char *>(data.data()),
             static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(file);
}

std::vector<uint8_t>
make_bundle(const std::vector<std::pair<std::string, std::vector<uint8_t>>> &entries) {
  std::vector<uint8_t> bundle;
  bundle.insert(bundle.end(), kBundleMagic, kBundleMagic + sizeof(kBundleMagic) - 1);
  append_u64(&bundle, entries.size());

  uint64_t offset = bundle.size();
  for (const auto &[id, _] : entries)
    offset += 3 * sizeof(uint64_t) + id.size();

  uint64_t data_offset = offset;
  for (const auto &[id, image] : entries) {
    append_u64(&bundle, data_offset);
    append_u64(&bundle, image.size());
    append_u64(&bundle, id.size());
    bundle.insert(bundle.end(), id.begin(), id.end());
    data_offset += image.size();
  }

  for (const auto &[_, image] : entries)
    bundle.insert(bundle.end(), image.begin(), image.end());
  return bundle;
}

bool is_bundle(const std::vector<uint8_t> &image) {
  return image.size() >= sizeof(kBundleMagic) - 1 &&
         memcmp(image.data(), kBundleMagic, sizeof(kBundleMagic) - 1) == 0;
}

bool append_device_entries_from_bundle(
    const std::vector<uint8_t> &bundle,
    std::vector<std::pair<std::string, std::vector<uint8_t>>> *entries) {
  constexpr size_t kBundleMagicSize = sizeof(kBundleMagic) - 1;
  if (!is_bundle(bundle) || bundle.size() < kBundleMagicSize + sizeof(uint64_t))
    return false;

  size_t pos = kBundleMagicSize;
  const uint64_t count = read_u64(bundle.data() + pos);
  pos += sizeof(uint64_t);
  if (count > 256)
    return false;

  for (uint64_t i = 0; i < count; ++i) {
    if (pos + 3 * sizeof(uint64_t) > bundle.size())
      return false;
    const uint64_t offset = read_u64(bundle.data() + pos);
    pos += sizeof(uint64_t);
    const uint64_t size = read_u64(bundle.data() + pos);
    pos += sizeof(uint64_t);
    const uint64_t id_size = read_u64(bundle.data() + pos);
    pos += sizeof(uint64_t);
    if (id_size > bundle.size() || pos + id_size > bundle.size() || offset > bundle.size() ||
        size > bundle.size() - offset)
      return false;
    std::string id(reinterpret_cast<const char *>(bundle.data() + pos),
                   static_cast<size_t>(id_size));
    pos += static_cast<size_t>(id_size);
    if (id.find("amdgcn") == std::string::npos)
      continue;
    entries->emplace_back(
        id, std::vector<uint8_t>(bundle.begin() + offset, bundle.begin() + offset + size));
  }
  return true;
}

std::vector<uint8_t> make_ccob_v3(const std::vector<uint8_t> &bundle) {
  std::vector<uint8_t> compressed(ZSTD_compressBound(bundle.size()));
  const size_t compressed_size =
      ZSTD_compress(compressed.data(), compressed.size(), bundle.data(), bundle.size(),
                    /*compressionLevel=*/1);
  if (ZSTD_isError(compressed_size)) {
    fprintf(stderr, "make_ccob_bundle: zstd compression failed: %s\n",
            ZSTD_getErrorName(compressed_size));
    return {};
  }
  compressed.resize(compressed_size);

  constexpr uint16_t kVersion = 3;
  constexpr uint16_t kZstdMethod = 1;
  constexpr uint32_t kHeaderSize = 32;
  const uint64_t total_size = kHeaderSize + compressed.size();
  std::optional<uint64_t> hash = rocjitsu::fuzzer::afl_dbi::truncated_md5_hash64(bundle);
  if (!hash) {
    fprintf(stderr, "make_ccob_bundle: MD5 hash computation failed\n");
    return {};
  }

  std::vector<uint8_t> ccob;
  ccob.insert(ccob.end(), {'C', 'C', 'O', 'B'});
  append_u32(&ccob, static_cast<uint32_t>(kVersion) | (static_cast<uint32_t>(kZstdMethod) << 16));
  append_u64(&ccob, total_size);
  append_u64(&ccob, bundle.size());
  append_u64(&ccob, *hash);
  ccob.insert(ccob.end(), compressed.begin(), compressed.end());
  return ccob;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4 && !(argc >= 5 && strcmp(argv[1], "--multi") == 0 && argc % 2 == 1)) {
    fprintf(stderr,
            "usage: %s <input-hsaco-or-bundle> <output-co> <bundle-id>\n"
            "       %s --multi <output-co> <input-hsaco> <bundle-id> "
            "[<input-hsaco> <bundle-id> ...]\n",
            argv[0], argv[0]);
    return 2;
  }

  if (argc == 4) {
    std::vector<uint8_t> input = read_file(argv[1]);
    if (input.empty()) {
      fprintf(stderr, "make_ccob_bundle: failed to read %s\n", argv[1]);
      return 1;
    }

    const std::vector<uint8_t> bundle =
        is_bundle(input) ? input : make_bundle({{std::string(argv[3]), std::move(input)}});
    std::vector<uint8_t> ccob = make_ccob_v3(bundle);
    if (ccob.empty() || !write_file(argv[2], ccob)) {
      fprintf(stderr, "make_ccob_bundle: failed to write %s\n", argv[2]);
      return 1;
    }
    return 0;
  }

  std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
  for (int arg = 3; arg < argc; arg += 2) {
    std::vector<uint8_t> input = read_file(argv[arg]);
    if (input.empty()) {
      fprintf(stderr, "make_ccob_bundle: failed to read %s\n", argv[arg]);
      return 1;
    }
    if (is_bundle(input)) {
      if (!append_device_entries_from_bundle(input, &entries)) {
        fprintf(stderr, "make_ccob_bundle: failed to parse bundle %s\n", argv[arg]);
        return 1;
      }
      continue;
    }
    entries.emplace_back(argv[arg + 1], std::move(input));
  }
  if (entries.empty()) {
    fprintf(stderr, "make_ccob_bundle: no device payloads found\n");
    return 1;
  }
  std::vector<uint8_t> ccob = make_ccob_v3(make_bundle(entries));
  if (ccob.empty() || !write_file(argv[2], ccob)) {
    fprintf(stderr, "make_ccob_bundle: failed to write %s\n", argv[2]);
    return 1;
  }
  return 0;
}
