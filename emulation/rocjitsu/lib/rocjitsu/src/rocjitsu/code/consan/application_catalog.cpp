// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/application_catalog.h"

#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/target.h"
#include "util/utf8.h"

#include <algorithm>
#include <fstream>
#include <set>

namespace rocjitsu::consan {
namespace {

constexpr std::array<uint8_t, 8> kCatalogMagic = {'R', 'J', 'C', 'S', 'A', 'P', 'P', '0'};

bool zero_digest(const Digest &value) {
  return std::ranges::all_of(value, [](uint8_t byte) { return byte == 0; });
}

bool valid_path(std::string_view path) {
  return !path.empty() && path.size() <= kApplicationCatalogPathBytes &&
         path.find('\0') == std::string_view::npos && util::is_valid_utf8(path) &&
         std::filesystem::path(path).is_absolute() &&
         std::filesystem::path(path).lexically_normal().string() == path;
}

Result validate_catalog(const ApplicationCatalog &catalog, util::DiagnosticEmitter diagnostic) {
  if (catalog.images.empty() || catalog.images.size() > kApplicationCatalogImages)
    return diagnostic.emit() << "application catalog image count is empty or oversized";
  uint64_t image_bytes = 0;
  std::set<Digest> identities;
  std::set<std::string> paths;
  std::set<std::string> captures;
  for (const ApplicationImage &image : catalog.images) {
    if (image.original.bytes == 0 || image.original.bytes > ContractLimits{}.image_bytes ||
        image.original.bytes > kApplicationCatalogImageBytes - image_bytes ||
        zero_digest(image.original.sha256) || target_profile(image.target) == nullptr ||
        !valid_path(image.path) || !identities.insert(image.original.sha256).second ||
        !paths.insert(image.path).second)
      return diagnostic.emit() << "invalid, duplicate or oversized application image";
    image_bytes += image.original.bytes;
    if (image.kind == ApplicationImageKind::Selected) {
      if (zero_digest(image.selection_profile) || !valid_path(image.capture_path) ||
          !captures.insert(image.capture_path).second)
        return diagnostic.emit() << "invalid selected profile or duplicate capture path";
    } else if (image.kind == ApplicationImageKind::Native) {
      if (!zero_digest(image.selection_profile) || !image.capture_path.empty())
        return diagnostic.emit() << "native image declares selected profile or capture";
    } else {
      return diagnostic.emit() << "unknown application image kind";
    }
  }
  for (const std::string &capture : captures)
    if (paths.contains(capture))
      return diagnostic.emit() << "application capture aliases an input path";
  return Result::success();
}

} // namespace

FailureOr<std::vector<uint8_t>> encode_application_catalog(const ApplicationCatalog &catalog,
                                                           util::DiagnosticEmitter diagnostic) {
  if (validate_catalog(catalog, diagnostic).failed())
    return Result::failure();
  wire::Encoder encoder;
  encoder.bytes(kCatalogMagic);
  encoder.bytes(kProducerContract);
  encoder.u32(static_cast<uint32_t>(catalog.images.size()));
  for (const ApplicationImage &image : catalog.images) {
    encoder.bytes(image.original.sha256);
    encoder.u64(image.original.bytes);
    encoder.u32(image.target);
    encoder.u32(static_cast<uint32_t>(image.kind));
    encoder.string(image.path);
    encoder.bytes(image.selection_profile);
    encoder.string(image.capture_path);
  }
  std::vector<uint8_t> bytes = encoder.take();
  const Digest checksum = digest(bytes);
  bytes.insert(bytes.end(), checksum.begin(), checksum.end());
  if (bytes.size() > kApplicationCatalogBytes)
    return diagnostic.emit() << "application catalog exceeds its byte budget";
  return bytes;
}

FailureOr<ApplicationCatalog> decode_application_catalog(std::span<const uint8_t> bytes,
                                                         util::DiagnosticEmitter diagnostic) {
  if (bytes.size() < kCatalogMagic.size() + kProducerContract.size() + 4 + Digest{}.size() ||
      bytes.size() > kApplicationCatalogBytes)
    return diagnostic.emit() << "application catalog has invalid extent";
  const std::span<const uint8_t> payload = bytes.first(bytes.size() - Digest{}.size());
  if (!std::ranges::equal(bytes.last(Digest{}.size()), digest(payload)))
    return diagnostic.emit() << "application catalog checksum differs";
  wire::Decoder decoder(payload);
  std::array<uint8_t, 8> magic{};
  std::array<uint8_t, 16> producer{};
  uint32_t count = 0;
  if (!decoder.bytes(magic) || magic != kCatalogMagic || !decoder.bytes(producer) ||
      producer != kProducerContract || !decoder.u32(count) || count == 0 ||
      count > kApplicationCatalogImages)
    return diagnostic.emit() << "unsupported application catalog header";
  ApplicationCatalog catalog;
  for (uint32_t index = 0; index < count; ++index) {
    ApplicationImage image;
    uint32_t kind = 0;
    if (!decoder.bytes(image.original.sha256) || !decoder.u64(image.original.bytes) ||
        !decoder.u32(image.target) || !decoder.u32(kind) ||
        !decoder.string(image.path, kApplicationCatalogPathBytes) ||
        !decoder.bytes(image.selection_profile) ||
        !decoder.string(image.capture_path, kApplicationCatalogPathBytes))
      return diagnostic.emit() << "truncated application image record";
    image.kind = static_cast<ApplicationImageKind>(kind);
    catalog.images.push_back(std::move(image));
  }
  if (decoder.remaining() != 0 || validate_catalog(catalog, diagnostic).failed())
    return diagnostic.emit() << "invalid application catalog records or trailing bytes";
  return catalog;
}

FailureOr<ApplicationCatalog> load_application_catalog(const std::filesystem::path &path,
                                                       util::DiagnosticEmitter diagnostic) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || static_cast<uint64_t>(file.tellg()) > kApplicationCatalogBytes)
    return diagnostic.emit() << "missing or oversized application catalog";
  std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file || file.peek() != std::char_traits<char>::eof())
    return diagnostic.emit() << "incomplete or changing application catalog";
  return decode_application_catalog(bytes, diagnostic);
}

} // namespace rocjitsu::consan
