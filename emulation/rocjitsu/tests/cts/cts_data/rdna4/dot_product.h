// Auto-generated CTS test data for rdna4 dot products.
// Do not edit — regenerate with cts_generator.py.

#ifndef ROCJITSU_CTS_DATA_RDNA4_DOT_PRODUCT_H_
#define ROCJITSU_CTS_DATA_RDNA4_DOT_PRODUCT_H_

#include <array>
#include <cstdint>
#include <string_view>

namespace rocjitsu::cts::rdna4 {

struct DotProductTestCase {
  std::string_view mnemonic;
  std::array<uint32_t, 2> encoding;
  std::array<uint32_t, 3> inputs;
  uint32_t expected_output;
};

inline constexpr DotProductTestCase DOT_PRODUCT_TESTS[] = {
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x01020304U, 0x05060708U, 0x00000000U},
     0x00000046U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x01010101U, 0x01010101U, 0x00000001U},
     0x00000005U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x7F7F7F7FU, 0x7F7F7F7FU, 0x00000000U},
     0x0000FC04U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x80808080U, 0x01010101U, 0x00000064U},
     0x00000264U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0xFFFFFFFFU, 0xFFFFFFFFU, 0x00000000U},
     0x0003F804U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x00FF00FFU, 0x00010001U, 0x0000000AU},
     0x00000208U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x12345678U, 0x9ABCDEF0U, 0x00001000U},
     0x0000FC18U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x55555555U, 0xAAAAAAAAU, 0xFFFFFFFFU},
     0x0000E1C7U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x00010002U, 0x00030004U, 0x00000005U},
     0x00000010U},
    {"v_dot4_u32_u8",
     {0xCC170006U, 0x1C120500U},
     {0x80008000U, 0x7FFF7FFFU, 0x00000000U},
     0x00007F00U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x01020304U, 0x05060708U, 0x00000000U},
     0x00000046U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x01010101U, 0x01010101U, 0x00000001U},
     0x00000005U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x7F7F7F7FU, 0x7F7F7F7FU, 0x00000000U},
     0x00000448U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x80808080U, 0x01010101U, 0x00000064U},
     0x00000064U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0xFFFFFFFFU, 0xFFFFFFFFU, 0x00000000U},
     0x00000708U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x00FF00FFU, 0x00010001U, 0x0000000AU},
     0x00000028U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x12345678U, 0x9ABCDEF0U, 0x00001000U},
     0x0000116CU},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x55555555U, 0xAAAAAAAAU, 0xFFFFFFFFU},
     0x0000018FU},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x00010002U, 0x00030004U, 0x00000005U},
     0x00000010U},
    {"v_dot8_u32_u4",
     {0xCC190006U, 0x1C120500U},
     {0x80008000U, 0x7FFF7FFFU, 0x00000000U},
     0x00000070U},
};

inline constexpr size_t NUM_DOT_PRODUCT_TESTS = 20;

} // namespace rocjitsu::cts::rdna4

#endif // ROCJITSU_CTS_DATA_RDNA4_DOT_PRODUCT_H_
