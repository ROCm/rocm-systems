// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm/sha2/sha256.h"

#include <cstring>
#include <iomanip>
#include <sstream>

namespace rocm {
namespace sha2 {

/// FIPS 180-4 section 5.3.3 initial hash value.
constexpr std::array<uint32_t, 8> sha256_initial_state = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

void secure_zero(void *p, size_t n) {
  auto *volatile q = static_cast<volatile unsigned char *>(p);
  while (n-- > 0)
    *q++ = 0;
}

sha256::sha256() { reset(); }

sha256::sha256(const std::string &data) {
  reset();
  update(data);
  finalize();
}

void sha256::update(const uint8_t *data, size_t len) {
  if (m_finalized)
    return;
  for (size_t i = 0; i < len; ++i) {
    m_data[m_datalen++] = data[i];
    if (m_datalen == 64) {
      transform();
      m_bitlen += 512;
      m_datalen = 0;
    }
  }
}

void sha256::update(const std::string &data) {
  if (m_finalized)
    return;
  update(reinterpret_cast<const uint8_t *>(data.data()), data.size());
}

void sha256::finalize() {
  if (m_finalized)
    return;

  uint32_t idx = m_datalen;
  if (m_datalen < 56) {
    m_data[idx++] = 0x80;
    while (idx < 56)
      m_data[idx++] = 0x00;
  } else {
    m_data[idx++] = 0x80;
    while (idx < 64)
      m_data[idx++] = 0x00;
    transform();
    std::memset(m_data.data(), 0, 56);
  }

  m_bitlen += static_cast<uint64_t>(m_datalen) * 8;
  for (int j = 0; j < 8; ++j)
    m_data[63 - j] = static_cast<uint8_t>((m_bitlen >> (8 * j)) & 0xFF);

  transform();

  m_finalized = true;
}

std::string sha256::hexdigest() {
  finalize();

  auto oss = std::ostringstream{};
  for (int j = 0; j < 8; ++j)
    oss << std::hex << std::setfill('0') << std::setw(8) << m_state[j];
  return oss.str();
}

std::array<uint32_t, 8> sha256::rawdigest() {
  finalize();

  return m_state;
}

std::array<uint8_t, SHA256_DIGEST_SIZE> sha256::digest() {
  finalize();

  auto out = std::array<uint8_t, SHA256_DIGEST_SIZE>{};
  for (size_t j = 0; j < m_state.size(); ++j) {
    out[j * 4 + 0] = static_cast<uint8_t>((m_state[j] >> 24) & 0xFF);
    out[j * 4 + 1] = static_cast<uint8_t>((m_state[j] >> 16) & 0xFF);
    out[j * 4 + 2] = static_cast<uint8_t>((m_state[j] >> 8) & 0xFF);
    out[j * 4 + 3] = static_cast<uint8_t>((m_state[j]) & 0xFF);
  }
  return out;
}

uint32_t sha256::rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

uint32_t sha256::ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}

uint32_t sha256::maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

uint32_t sha256::sig0(uint32_t x) {
  return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

uint32_t sha256::sig1(uint32_t x) {
  return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

uint32_t sha256::theta0(uint32_t x) {
  return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

uint32_t sha256::theta1(uint32_t x) {
  return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

void sha256::transform() {
  uint32_t m[64];
  for (int i = 0; i < 16; ++i) {
    m[i] =
        (static_cast<uint32_t>(m_data[static_cast<size_t>(i) * 4]) << 24) |
        (static_cast<uint32_t>(m_data[static_cast<size_t>(i) * 4 + 1]) << 16) |
        (static_cast<uint32_t>(m_data[static_cast<size_t>(i) * 4 + 2]) << 8) |
        (static_cast<uint32_t>(m_data[static_cast<size_t>(i) * 4 + 3]));
  }
  for (int i = 16; i < 64; ++i) {
    m[i] = theta1(m[i - 2]) + m[i - 7] + theta0(m[i - 15]) + m[i - 16];
  }

  uint32_t a = m_state[0];
  uint32_t b = m_state[1];
  uint32_t c = m_state[2];
  uint32_t d = m_state[3];
  uint32_t e = m_state[4];
  uint32_t f = m_state[5];
  uint32_t g = m_state[6];
  uint32_t h = m_state[7];

  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + sig1(e) + ch(e, f, g) + m_k[i] + m[i];
    uint32_t t2 = sig0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  m_state[0] += a;
  m_state[1] += b;
  m_state[2] += c;
  m_state[3] += d;
  m_state[4] += e;
  m_state[5] += f;
  m_state[6] += g;
  m_state[7] += h;

  secure_zero(m, sizeof(m));
}

void sha256::reset() {
  m_state = sha256_initial_state;
  m_data = {};
  m_datalen = 0;
  m_bitlen = 0;
  m_finalized = false;
}

void sha256_digest(const uint8_t *data, size_t data_len,
                   uint8_t out[SHA256_DIGEST_SIZE]) {
  if (!out)
    return;
  if (!data && data_len > 0) {
    secure_zero(out, SHA256_DIGEST_SIZE);
    return;
  }
  sha256 h;
  if (data_len > 0)
    h.update(data, data_len);
  const auto d = h.digest();
  std::memcpy(out, d.data(), d.size());
}

void hmac_sha256(const uint8_t *key, size_t key_len, const uint8_t *msg,
                 size_t msg_len, uint8_t out[SHA256_DIGEST_SIZE]) {
  uint8_t k0[SHA256_BLOCK_SIZE] = {};

  if (key_len > SHA256_BLOCK_SIZE) {
    sha256_digest(key, key_len, k0);
  } else if (key_len > 0) {
    std::memcpy(k0, key, key_len);
  }

  uint8_t pad[SHA256_BLOCK_SIZE];
  uint8_t inner[SHA256_DIGEST_SIZE];

  for (size_t i = 0; i < SHA256_BLOCK_SIZE; ++i)
    pad[i] = static_cast<uint8_t>(k0[i] ^ 0x36);
  {
    sha256 h;
    h.update(pad, SHA256_BLOCK_SIZE);
    if (msg_len > 0)
      h.update(msg, msg_len);
    const auto d = h.digest();
    std::memcpy(inner, d.data(), d.size());
  }

  for (size_t i = 0; i < SHA256_BLOCK_SIZE; ++i)
    pad[i] = static_cast<uint8_t>(k0[i] ^ 0x5c);
  {
    sha256 h;
    h.update(pad, SHA256_BLOCK_SIZE);
    h.update(inner, SHA256_DIGEST_SIZE);
    const auto d = h.digest();
    std::memcpy(out, d.data(), d.size());
  }

  secure_zero(k0, sizeof(k0));
  secure_zero(pad, sizeof(pad));
  secure_zero(inner, sizeof(inner));
}

} // namespace sha2
} // namespace rocm
