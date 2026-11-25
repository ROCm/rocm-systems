#ifndef RCCL_TEST_HOST_DEVCOMMTEST_H_
#define RCCL_TEST_HOST_DEVCOMMTEST_H_

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>

namespace devcomm_test {

inline bool AllBytesAre(const void* memory, std::size_t size, unsigned char value) {
  const auto* bytes = static_cast<const unsigned char*>(memory);
  for (std::size_t index = 0; index < size; ++index) {
    if (bytes[index] != value) return false;
  }
  return true;
}

inline std::ptrdiff_t FirstDiff(const void* actual, const void* expected, std::size_t size) {
  const auto* actualBytes = static_cast<const unsigned char*>(actual);
  const auto* expectedBytes = static_cast<const unsigned char*>(expected);
  for (std::size_t index = 0; index < size; ++index) {
    if (actualBytes[index] != expectedBytes[index]) return static_cast<std::ptrdiff_t>(index);
  }
  return -1;
}

template <class Value, std::size_t GuardLength = 64>
class PoisonedArena {
 public:
  static constexpr unsigned char kGuardByte = 0x5A;

  explicit PoisonedArena(unsigned char poison = 0xA5) { Reset(poison); }

  void Reset(unsigned char poison) {
    static_assert(std::is_trivially_default_constructible_v<Value>,
                  "placement-new below would overwrite the poison the oracles depend on");
    std::memset(bytes_, kGuardByte, sizeof(bytes_));
    std::memset(bytes_ + GuardLength, poison, sizeof(Value));
    new (bytes_ + GuardLength) Value;
  }

  unsigned char* raw() { return bytes_ + GuardLength; }
  const unsigned char* raw() const { return bytes_ + GuardLength; }
  Value* obj() { return reinterpret_cast<Value*>(raw()); }
  unsigned char byteAt(std::size_t index) const { return bytes_[GuardLength + index]; }

  bool GuardsIntact() const {
    return AllBytesAre(bytes_, GuardLength, kGuardByte) &&
           AllBytesAre(bytes_ + GuardLength + sizeof(Value), GuardLength, kGuardByte);
  }

 private:
  static constexpr std::size_t kAlign = alignof(Value) > 16 ? alignof(Value) : 16;
  static_assert(GuardLength % kAlign == 0, "guard band would misalign the object it flanks");
  alignas(kAlign) unsigned char bytes_[GuardLength + sizeof(Value) + GuardLength];
};

}

#endif
