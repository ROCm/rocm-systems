// Auto-generated CTS test data for rdna3_5 bitop3.
// Do not edit — regenerate with cts_generator.py.

#ifndef ROCJITSU_CTS_DATA_RDNA3_5_VECTOR_BITOP3_H_
#define ROCJITSU_CTS_DATA_RDNA3_5_VECTOR_BITOP3_H_

#include <array>
#include <cstdint>
#include <string_view>

namespace rocjitsu::cts::rdna3_5 {

struct VectorBitop3TestCase {
  std::string_view mnemonic;
  std::array<uint32_t, 2> encoding;
  std::array<uint32_t, 3> inputs;
  uint32_t expected_output;
  uint32_t truth_table;
};

inline constexpr VectorBitop3TestCase *VECTOR_BITOP3_TESTS = nullptr;

inline constexpr size_t NUM_VECTOR_BITOP3_TESTS = 0;

} // namespace rocjitsu::cts::rdna3_5

#endif // ROCJITSU_CTS_DATA_RDNA3_5_VECTOR_BITOP3_H_
