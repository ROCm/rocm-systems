// Auto-generated CTS test data for rdna2 MFMA/WMMA.
// Do not edit — regenerate with cts_generator.py.

#ifndef ROCJITSU_CTS_DATA_RDNA2_MFMA_H_
#define ROCJITSU_CTS_DATA_RDNA2_MFMA_H_

#include <cstdint>
#include <string_view>

namespace rocjitsu::cts::rdna2 {

struct MfmaTestCase {
  std::string_view mnemonic;
  uint32_t encoding[2];
  uint16_t num_src0_vgprs;
  uint16_t num_src1_vgprs;
  uint16_t num_dst_vgprs;
  uint16_t src0_base;
  uint16_t src1_base;
  uint16_t dst_base;
  uint16_t acc_base;
  uint16_t wf_size;
  bool has_acc;
  const uint32_t *src0_data;
  const uint32_t *src1_data;
  const uint32_t *acc_data;
  const uint32_t *expected;
};

inline constexpr MfmaTestCase *MFMA_TESTS = nullptr;

inline constexpr size_t NUM_MFMA_TESTS = 0;

} // namespace rocjitsu::cts::rdna2

#endif // ROCJITSU_CTS_DATA_RDNA2_MFMA_H_
