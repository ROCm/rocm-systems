#include "md5_util.h"

#include <openssl/evp.h>

#include <array>
#include <string.h>

namespace rocjitsu::fuzzer::afl_dbi {
namespace {

uint64_t read_le_u64(const unsigned char *data) {
  uint64_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

} // namespace

std::optional<uint64_t> truncated_md5_hash64(std::span<const uint8_t> data) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  const unsigned char *input =
      data.empty() ? reinterpret_cast<const unsigned char *>("") : data.data();
  if (EVP_Digest(input, data.size(), digest.data(), &digest_size, EVP_md5(), nullptr) != 1 ||
      digest_size < sizeof(uint64_t))
    return std::nullopt;
  return read_le_u64(digest.data());
}

} // namespace rocjitsu::fuzzer::afl_dbi
