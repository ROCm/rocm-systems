// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/application_catalog.h"
#include "rocjitsu/code/consan/encoding.h"

#include <gtest/gtest.h>

namespace rocjitsu::consan {
namespace {

ApplicationCatalog example_catalog() {
  return {.images = {{.original = {.sha256 = Digest{1}, .bytes = 4096},
                      .target = 1250,
                      .kind = ApplicationImageKind::Selected,
                      .path = "/application/artifact-1",
                      .selection_profile = Digest{3},
                      .capture_path = "/application/captures/1.bin"},
                     {.original = {.sha256 = Digest{2}, .bytes = 2048},
                      .target = 1250,
                      .kind = ApplicationImageKind::Native,
                      .path = "/application/native-2.hsaco",
                      .selection_profile = {},
                      .capture_path = {}}}};
}

TEST(ConSanApplicationCatalog, RoundTripsSelectedAndNativeIdentityWithoutRuntimeJson) {
  const ApplicationCatalog catalog = example_catalog();
  const FailureOr<std::vector<uint8_t>> encoded = encode_application_catalog(catalog);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ApplicationCatalog> decoded = decode_application_catalog(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), catalog);
  for (size_t extent = 0; extent < encoded.value().size(); ++extent)
    EXPECT_TRUE(decode_application_catalog(std::span(encoded.value()).first(extent)).failed());
}

TEST(ConSanApplicationCatalog, ChecksumAndRecordEnumsAreBothValidated) {
  const FailureOr<std::vector<uint8_t>> encoded = encode_application_catalog(example_catalog());
  ASSERT_TRUE(encoded.succeeded());
  std::vector<uint8_t> corrupted = encoded.value();
  corrupted[40] ^= 1;
  EXPECT_TRUE(decode_application_catalog(corrupted).failed());
  corrupted = encoded.value();
  // Header28 + original digest32 + byte extent8 + target4 precede the kind.
  ASSERT_TRUE(wire::write_u32(corrupted, 72, 99));
  const Digest checksum = digest(std::span(corrupted).first(corrupted.size() - 32));
  std::copy(checksum.begin(), checksum.end(), corrupted.end() - 32);
  EXPECT_TRUE(decode_application_catalog(corrupted).failed());
}

TEST(ConSanApplicationCatalog, DuplicateImagesAndCapturePathsNeverAliasBindings) {
  ApplicationCatalog catalog = example_catalog();
  catalog.images[1].original = catalog.images[0].original;
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[1].kind = ApplicationImageKind::Selected;
  catalog.images[1].selection_profile = Digest{4};
  catalog.images[1].capture_path = catalog.images[0].capture_path;
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[0].capture_path = catalog.images[1].path;
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
}

TEST(ConSanApplicationCatalog, NativeImagesCannotClaimSelectedCaptureEvidence) {
  ApplicationCatalog catalog = example_catalog();
  catalog.images[1].selection_profile = Digest{1};
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[1].capture_path = "/application/captures/native.bin";
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[0].selection_profile = {};
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
}

TEST(ConSanApplicationCatalog, RejectsInvalidPathsTargetsKindsAndExtentBudgets) {
  for (const std::string &path :
       {std::string(), std::string("relative/path"), std::string("/application/../artifact"),
        std::string("/application/\0hidden", 20), std::string("/application/\xff", 14),
        std::string(kApplicationCatalogPathBytes + 1, '/')}) {
    ApplicationCatalog catalog = example_catalog();
    catalog.images[0].path = path;
    EXPECT_TRUE(encode_application_catalog(catalog).failed());
  }
  ApplicationCatalog catalog = example_catalog();
  catalog.images[0].target = 9999;
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[0].kind = static_cast<ApplicationImageKind>(99);
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[0].original.bytes = UINT64_MAX;
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  catalog = example_catalog();
  catalog.images[0].original.sha256 = {};
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
  EXPECT_TRUE(encode_application_catalog({}).failed());
  catalog.images.resize(kApplicationCatalogImages + 1);
  EXPECT_TRUE(encode_application_catalog(catalog).failed());
}

} // namespace
} // namespace rocjitsu::consan
