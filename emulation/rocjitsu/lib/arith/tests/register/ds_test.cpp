// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for DS gather/scatter and lane swizzling.

#include "register/helpers.h"

namespace {
static const uint16_t swizzle_offsets[] = {31, 0, 0x7c1f, 0x80b1, 0xc020, 0xc47f, 0xe000, 0xe015};

// Independent bit-by-bit routing reference. The FFT lookup is the manual's
// published mask=0 order (converted from one-based to zero-based lane indices).
unsigned source_lane(int lane, unsigned offset) {
  static const unsigned fft[] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                                 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
  int local = lane % 32, source = 0;
  if (offset >= 0xe000) {
    unsigned divisor = 1;
    for (int bit = 0; bit < 5; ++bit) {
      if (offset & (1U << bit)) {
        divisor *= 2;
      }
    }
    source = fft[local] / divisor;
    for (int bit = 0; bit < 5; ++bit) {
      if ((offset & local) & (1U << bit)) {
        source |= 1U << bit;
      }
    }
  } else if (offset >= 0xc000) {
    int delta = (offset / 32) % 32;
    if (offset & 1024) {
      delta = -delta;
    }
    unsigned rotated = (int(local) + delta + 32) % 32;
    for (int bit = 0; bit < 5; ++bit) {
      source |= (((offset >> bit) & 1 ? local : rotated) & (1U << bit));
    }
  } else if (offset & 0x8000) {
    source = (local / 4) * 4 + ((offset >> (2 * (local % 4))) & 3);
  } else {
    for (int bit = 0; bit < 5; ++bit) {
      bool value = ((local >> bit) & 1) && ((offset >> bit) & 1);
      value = value || ((offset >> (bit + 5)) & 1);
      value = value != bool((offset >> (bit + 10)) & 1);
      source |= unsigned(value) << bit;
    }
  }
  return lane / 32 * 32 + source;
}

} // namespace

// Check register routing and preservation of unselected destination data.
TEST(ds_permute_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_permute_b32_w32(f, e, m, r.v + d, r.v, r.v + 1, uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 32; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 32) + lane / 32 * 32;
          if (((e >> lane) & 1) && ((e >> address) & 1)) {
            out.v[d][address] = in.v[1][lane];
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(ds_bpermute_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_bpermute_b32_w32(f, e, m, r.v + d, r.v, r.v + 1, uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 32; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 32) + lane / 32 * 32;
          if ((e >> lane) & 1) {
            out.v[d][lane] = (((e >> address) & 1)) ? in.v[1][address] : 0;
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(ds_bpermute_fi_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_bpermute_fi_b32_w32(f, e, m, r.v + d, r.v, r.v + 1, uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 32; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 32) + lane / 32 * 32;
          if ((e >> lane) & 1) {
            out.v[d][lane] = (true) ? in.v[1][address] : 0;
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(ds_swizzle_b32_w32, Arithmetic) {
  rocjitsu::arith::test::check_register<32>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_swizzle_b32_w32(f, e, m, r.v + d, r.v, swizzle_offsets[s % 8]);
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 32; ++lane) {
          if ((e >> lane) & 1) {
            int source = source_lane(lane, swizzle_offsets[s % 8]);
            out.v[d][lane] = ((e >> source) & 1) ? in.v[0][source] : 0;
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterDsPermuteB32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_permute_b32_w64(f, e, m, r.v + d, r.v, r.v + 1, uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 64; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 32) + lane / 32 * 32;
          if (((e >> lane) & 1) && ((e >> address) & 1)) {
            out.v[d][address] = in.v[1][lane];
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterDsPermuteB32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_permute_b32_w64_gfx12(f, e, m, r.v + d, r.v, r.v + 1,
                                                 uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 64; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 64) + lane / 64 * 64;
          if (((e >> lane) & 1) && ((e >> address) & 1)) {
            out.v[d][address] = in.v[1][lane];
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterDsBpermuteB32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_bpermute_b32_w64(f, e, m, r.v + d, r.v, r.v + 1, uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 64; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 32) + lane / 32 * 32;
          if ((e >> lane) & 1) {
            out.v[d][lane] = (((e >> address) & 1)) ? in.v[1][address] : 0;
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(RegisterDsBpermuteB32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_bpermute_b32_w64_gfx12(f, e, m, r.v + d, r.v, r.v + 1,
                                                  uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 64; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 64) + lane / 64 * 64;
          if ((e >> lane) & 1) {
            out.v[d][lane] = (((e >> address) & 1)) ? in.v[1][address] : 0;
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(ds_bpermute_fi_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_bpermute_fi_b32_w64(f, e, m, r.v + d, r.v, r.v + 1, uint16_t(s * 1973));
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            out.v[d][lane] = 0;
          }
        }
        for (int lane = 0; lane < 64; ++lane) {
          int address = int(((in.v[0][lane] + uint16_t(s * 1973)) >> 2) % 64) + lane / 64 * 64;
          if ((e >> lane) & 1) {
            out.v[d][lane] = (true) ? in.v[1][address] : 0;
          }
        }
      });
}

// Check register routing and preservation of unselected destination data.
TEST(ds_swizzle_b32_w64, Arithmetic) {
  rocjitsu::arith::test::check_register<64>(
      [](auto f, auto e, auto m, auto &r, int d, int s) {
        return rj_arith_ds_swizzle_b32_w64(f, e, m, r.v + d, r.v, swizzle_offsets[s % 8]);
      },
      [](auto &out, const auto &in, auto e, auto, int d, int s) {
        for (int lane = 0; lane < 64; ++lane) {
          if ((e >> lane) & 1) {
            int source = source_lane(lane, swizzle_offsets[s % 8]);
            out.v[d][lane] = ((e >> source) & 1) ? in.v[0][source] : 0;
          }
        }
      });
}
