//===-- aegisbit/Endian.h - Little-Endian Byte Utilities --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities for reading and writing little-endian values from byte arrays.
/// AMD GPU code objects are always little-endian.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_ENDIAN_H
#define AEGISBIT_ENDIAN_H

#include <cstdint>

namespace aegisbit {

/// Read a little-endian uint16_t from a byte array.
/// \param Bytes Pointer to at least 2 bytes
/// \return The decoded value
inline uint16_t readLE16(const uint8_t* Bytes) {
  return static_cast<uint16_t>(Bytes[0]) |
         (static_cast<uint16_t>(Bytes[1]) << 8);
}

/// Read a little-endian uint32_t from a byte array.
/// \param Bytes Pointer to at least 4 bytes
/// \return The decoded value
inline uint32_t readLE32(const uint8_t* Bytes) {
  return static_cast<uint32_t>(Bytes[0]) |
         (static_cast<uint32_t>(Bytes[1]) << 8) |
         (static_cast<uint32_t>(Bytes[2]) << 16) |
         (static_cast<uint32_t>(Bytes[3]) << 24);
}

/// Read a little-endian uint64_t from a byte array.
/// \param Bytes Pointer to at least 8 bytes
/// \return The decoded value
inline uint64_t readLE64(const uint8_t* Bytes) {
  return static_cast<uint64_t>(readLE32(Bytes)) |
         (static_cast<uint64_t>(readLE32(Bytes + 4)) << 32);
}

/// Write a little-endian uint16_t to a byte array.
/// \param Bytes Pointer to at least 2 bytes
/// \param Value The value to encode
inline void writeLE16(uint8_t* Bytes, uint16_t Value) {
  Bytes[0] = static_cast<uint8_t>(Value);
  Bytes[1] = static_cast<uint8_t>(Value >> 8);
}

/// Write a little-endian uint32_t to a byte array.
/// \param Bytes Pointer to at least 4 bytes
/// \param Value The value to encode
inline void writeLE32(uint8_t* Bytes, uint32_t Value) {
  Bytes[0] = static_cast<uint8_t>(Value);
  Bytes[1] = static_cast<uint8_t>(Value >> 8);
  Bytes[2] = static_cast<uint8_t>(Value >> 16);
  Bytes[3] = static_cast<uint8_t>(Value >> 24);
}

/// Write a little-endian uint64_t to a byte array.
/// \param Bytes Pointer to at least 8 bytes
/// \param Value The value to encode
inline void writeLE64(uint8_t* Bytes, uint64_t Value) {
  writeLE32(Bytes, static_cast<uint32_t>(Value));
  writeLE32(Bytes + 4, static_cast<uint32_t>(Value >> 32));
}

} // namespace aegisbit

#endif // AEGISBIT_ENDIAN_H
