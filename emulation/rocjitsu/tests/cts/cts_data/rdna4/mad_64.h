// Auto-generated CTS test data for rdna4 mad_u64_u32/mad_i64_i32.
// Do not edit — regenerate with cts_generator.py.

#ifndef ROCJITSU_CTS_DATA_RDNA4_MAD_64_H_
#define ROCJITSU_CTS_DATA_RDNA4_MAD_64_H_

#include <array>
#include <cstdint>
#include <string_view>

namespace rocjitsu::cts::rdna4 {

struct Mad64TestCase {
  std::string_view mnemonic;
  std::array<uint32_t, 2> encoding;
  std::array<uint32_t, 4> inputs;
  uint32_t expected_lo;
  uint32_t expected_hi;
};

inline constexpr Mad64TestCase MAD_64_TESTS[] = {
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U},
     0x00000000U,
     0x00000000U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x00000001U, 0x00000001U, 0x00000000U, 0x00000000U},
     0x00000001U,
     0x00000000U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0xFFFFFFFFU, 0xFFFFFFFFU, 0x00000000U, 0x00000000U},
     0x00000001U,
     0xFFFFFFFEU},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x7FFFFFFFU, 0x00000002U, 0x00000000U, 0x00000000U},
     0xFFFFFFFEU,
     0x00000000U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x80000000U, 0x00000002U, 0x00000000U, 0x00000000U},
     0x00000000U,
     0x00000001U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x00000064U, 0x000000C8U, 0x0000FFFFU, 0x00000000U},
     0x00014E1FU,
     0x00000000U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0xDEADBEEFU, 0xCAFEBABEU, 0x12345678U, 0x9ABCDEF0U},
     0x9B03B1DAU,
     0x4B4F8A6BU},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x00000000U, 0xFFFFFFFFU, 0x00000001U, 0x00000000U},
     0x00000001U,
     0x00000000U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x00000002U, 0x00000003U, 0xFFFFFFFFU, 0xFFFFFFFFU},
     0x00000005U,
     0x00000000U},
    {"v_mad_co_u64_u32",
     {0xD6FE0008U, 0x04120500U},
     {0x00010000U, 0x00010000U, 0x00000000U, 0x00000000U},
     0x00000000U,
     0x00000001U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U},
     0x00000000U,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x00000001U, 0x00000001U, 0x00000000U, 0x00000000U},
     0x00000001U,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0xFFFFFFFFU, 0xFFFFFFFFU, 0x00000000U, 0x00000000U},
     0x00000001U,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x7FFFFFFFU, 0x00000002U, 0x00000000U, 0x00000000U},
     0xFFFFFFFEU,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x80000000U, 0x00000002U, 0x00000000U, 0x00000000U},
     0x00000000U,
     0xFFFFFFFFU},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x00000064U, 0x000000C8U, 0x0000FFFFU, 0x00000000U},
     0x00014E1FU,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0xDEADBEEFU, 0xCAFEBABEU, 0x12345678U, 0x9ABCDEF0U},
     0x9B03B1DAU,
     0xA1A310BEU},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x00000000U, 0xFFFFFFFFU, 0x00000001U, 0x00000000U},
     0x00000001U,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x00000002U, 0x00000003U, 0xFFFFFFFFU, 0xFFFFFFFFU},
     0x00000005U,
     0x00000000U},
    {"v_mad_co_i64_i32",
     {0xD6FF0008U, 0x04120500U},
     {0x00010000U, 0x00010000U, 0x00000000U, 0x00000000U},
     0x00000000U,
     0x00000001U},
};

inline constexpr size_t NUM_MAD_64_TESTS = 20;

} // namespace rocjitsu::cts::rdna4

#endif // ROCJITSU_CTS_DATA_RDNA4_MAD_64_H_
