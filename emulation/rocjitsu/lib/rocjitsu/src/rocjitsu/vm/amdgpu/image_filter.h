// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IMAGE_FILTER_H_
#define ROCJITSU_VM_AMDGPU_IMAGE_FILTER_H_

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// Resource performance modulation, in units of 1/16.
inline constexpr std::array<uint32_t, 8> image_perf_scales{0, 2, 5, 7, 9, 11, 14, 16};

/// Remap a fixed8 mip fraction, snapping the outer transition intervals to
/// one mip. Unlike LOD conversion, the remapped fraction is truncated.
inline uint32_t image_mip_fraction(uint32_t fraction, uint32_t perf_mip, uint32_t perf_mod) {
  constexpr std::array<int32_t, 16> gains{16, 17, 18, 20, 24, 32, 34,  36,
                                          40, 48, 64, 72, 80, 96, 128, 192};
  const uint32_t index = perf_mip * image_perf_scales[perf_mod] >> 4;
  const int32_t scaled = 2048 + (static_cast<int32_t>(fraction) - 128) * gains[index];
  return static_cast<uint32_t>(std::clamp(scaled, 0, 4096)) / 16;
}

/// Texture footprint logarithm in signed units of 1/256. The GFX11/12
/// mantissa is rounded to ten fractional bits before an eight-segment log.
inline int32_t image_log2_fixed8(double value) {
  if (value <= 0)
    return -8192;
  int exponent;
  const double mantissa = std::frexp(value, &exponent);
  uint32_t fraction = static_cast<uint32_t>((2 * mantissa - 1) * 1024 + 0.5);
  if (fraction == 1024) {
    fraction = 0;
    ++exponent;
  }
  constexpr std::array<uint32_t, 8> bases{0, 175, 330, 471, 599, 717, 827, 929};
  constexpr std::array<uint32_t, 8> slopes{176, 155, 142, 128, 119, 110, 102, 95};
  const uint32_t segment = fraction >> 7;
  return (exponent - 1) * 256 +
         static_cast<int32_t>((bases[segment] * 128 + slopes[segment] * (fraction & 127) + 256) /
                              512);
}

/// GFX11/12 reciprocal mantissas for a ten-fraction-bit norm input in [1, 2).
/// Entries are fixed10 values recovered from independent physical direction
/// and phase controls across the complete mantissa domain.
inline constexpr std::array<uint16_t, 1024> image_norm_reciprocals{
    1024, 1023, 1022, 1021, 1020, 1019, 1018, 1017, 1016, 1015, 1014, 1013, 1012, 1011, 1010, 1009,
    1008, 1008, 1007, 1006, 1005, 1004, 1003, 1002, 1001, 1000, 999,  998,  997,  996,  995,  994,
    993,  992,  992,  991,  990,  989,  988,  987,  986,  985,  984,  983,  982,  981,  980,  979,
    978,  977,  976,  976,  975,  974,  973,  972,  971,  970,  969,  968,  967,  966,  965,  964,
    963,  962,  962,  961,  960,  959,  958,  957,  957,  956,  955,  954,  953,  952,  952,  951,
    950,  949,  948,  947,  947,  946,  945,  944,  943,  942,  942,  941,  940,  939,  938,  937,
    936,  936,  935,  934,  933,  932,  931,  931,  930,  929,  928,  927,  926,  926,  925,  924,
    923,  922,  921,  921,  920,  919,  918,  917,  916,  916,  915,  914,  913,  912,  911,  911,
    910,  909,  908,  908,  907,  906,  905,  905,  904,  903,  902,  902,  901,  900,  899,  899,
    898,  897,  896,  896,  895,  894,  893,  893,  892,  891,  890,  890,  889,  888,  887,  887,
    886,  885,  884,  884,  883,  882,  881,  881,  880,  879,  878,  878,  877,  876,  875,  875,
    874,  873,  872,  872,  871,  870,  869,  869,  868,  867,  866,  866,  865,  864,  863,  863,
    862,  861,  861,  860,  859,  859,  858,  857,  856,  856,  855,  854,  854,  853,  852,  852,
    851,  850,  850,  849,  848,  848,  847,  846,  846,  845,  844,  844,  843,  842,  842,  841,
    840,  840,  839,  838,  838,  837,  836,  836,  835,  834,  834,  833,  832,  832,  831,  830,
    830,  829,  828,  828,  827,  826,  825,  825,  824,  823,  823,  822,  821,  821,  820,  819,
    819,  818,  818,  817,  816,  816,  815,  814,  814,  813,  813,  812,  811,  811,  810,  810,
    809,  808,  808,  807,  807,  806,  805,  805,  804,  804,  803,  802,  802,  801,  800,  800,
    799,  799,  798,  797,  797,  796,  796,  795,  794,  794,  793,  793,  792,  791,  791,  790,
    789,  789,  788,  788,  787,  786,  786,  785,  785,  784,  783,  783,  782,  782,  781,  780,
    780,  779,  779,  778,  778,  777,  776,  776,  775,  775,  774,  774,  773,  773,  772,  771,
    771,  770,  770,  769,  769,  768,  768,  767,  766,  766,  765,  765,  764,  764,  763,  763,
    762,  761,  761,  760,  760,  759,  759,  758,  758,  757,  756,  756,  755,  755,  754,  754,
    753,  753,  752,  751,  751,  750,  750,  749,  749,  748,  748,  747,  746,  746,  745,  745,
    744,  744,  743,  743,  742,  742,  741,  741,  740,  740,  739,  739,  738,  738,  737,  737,
    736,  736,  735,  735,  734,  734,  733,  733,  732,  732,  731,  731,  730,  730,  729,  729,
    728,  728,  727,  727,  726,  726,  725,  725,  724,  724,  723,  722,  722,  721,  721,  720,
    720,  719,  719,  718,  718,  717,  717,  716,  716,  715,  715,  714,  714,  713,  713,  712,
    712,  711,  711,  710,  710,  710,  709,  709,  708,  708,  707,  707,  706,  706,  705,  705,
    704,  704,  704,  703,  703,  702,  702,  701,  701,  700,  700,  699,  699,  698,  698,  698,
    697,  697,  696,  696,  695,  695,  694,  694,  693,  693,  692,  692,  692,  691,  691,  690,
    690,  689,  689,  688,  688,  687,  687,  686,  686,  685,  685,  685,  684,  684,  683,  683,
    682,  682,  681,  681,  680,  680,  680,  679,  679,  678,  678,  677,  677,  677,  676,  676,
    675,  675,  674,  674,  674,  673,  673,  672,  672,  671,  671,  671,  670,  670,  669,  669,
    668,  668,  668,  667,  667,  666,  666,  666,  665,  665,  664,  664,  663,  663,  663,  662,
    662,  661,  661,  660,  660,  660,  659,  659,  658,  658,  657,  657,  657,  656,  656,  655,
    655,  654,  654,  654,  653,  653,  653,  652,  652,  651,  651,  651,  650,  650,  649,  649,
    649,  648,  648,  647,  647,  647,  646,  646,  645,  645,  645,  644,  644,  643,  643,  643,
    642,  642,  641,  641,  641,  640,  640,  639,  639,  639,  638,  638,  638,  637,  637,  636,
    636,  636,  635,  635,  634,  634,  634,  633,  633,  632,  632,  632,  631,  631,  630,  630,
    630,  629,  629,  629,  628,  628,  627,  627,  627,  626,  626,  626,  625,  625,  625,  624,
    624,  623,  623,  623,  622,  622,  622,  621,  621,  620,  620,  620,  619,  619,  619,  618,
    618,  618,  617,  617,  616,  616,  616,  615,  615,  615,  614,  614,  614,  613,  613,  612,
    612,  612,  611,  611,  611,  610,  610,  610,  609,  609,  608,  608,  608,  607,  607,  607,
    606,  606,  606,  605,  605,  605,  604,  604,  604,  603,  603,  603,  602,  602,  602,  601,
    601,  601,  600,  600,  600,  599,  599,  599,  598,  598,  598,  597,  597,  597,  596,  596,
    596,  595,  595,  595,  594,  594,  594,  593,  593,  593,  592,  592,  592,  591,  591,  590,
    590,  590,  589,  589,  589,  588,  588,  588,  587,  587,  587,  586,  586,  586,  585,  585,
    585,  584,  584,  584,  583,  583,  583,  582,  582,  582,  581,  581,  581,  581,  580,  580,
    580,  579,  579,  579,  578,  578,  578,  577,  577,  577,  576,  576,  576,  576,  575,  575,
    575,  574,  574,  574,  573,  573,  573,  572,  572,  572,  571,  571,  571,  570,  570,  570,
    570,  569,  569,  569,  568,  568,  568,  567,  567,  567,  566,  566,  566,  565,  565,  565,
    565,  564,  564,  564,  563,  563,  563,  562,  562,  562,  562,  561,  561,  561,  560,  560,
    560,  559,  559,  559,  559,  558,  558,  558,  557,  557,  557,  557,  556,  556,  556,  555,
    555,  555,  554,  554,  554,  554,  553,  553,  553,  553,  552,  552,  552,  551,  551,  551,
    550,  550,  550,  549,  549,  549,  549,  548,  548,  548,  547,  547,  547,  547,  546,  546,
    546,  545,  545,  545,  545,  544,  544,  544,  543,  543,  543,  543,  542,  542,  542,  541,
    541,  541,  541,  540,  540,  540,  540,  539,  539,  539,  538,  538,  538,  538,  537,  537,
    537,  537,  536,  536,  536,  535,  535,  535,  535,  534,  534,  534,  534,  533,  533,  533,
    532,  532,  532,  532,  531,  531,  531,  530,  530,  530,  530,  529,  529,  529,  529,  528,
    528,  528,  527,  527,  527,  527,  526,  526,  526,  526,  525,  525,  525,  525,  524,  524,
    524,  524,  523,  523,  523,  523,  522,  522,  522,  522,  521,  521,  521,  521,  520,  520,
    520,  519,  519,  519,  519,  518,  518,  518,  518,  517,  517,  517,  517,  516,  516,  516,
    516,  515,  515,  515,  515,  514,  514,  514,  514,  513,  513,  513,  513,  512,  512,  512,
};

struct ImageReciprocal {
  uint32_t mantissa;
  int exponent;
};

/// Keep the reciprocal's carry bit in the mantissa at exact powers of two.
/// Renormalizing it changes the precision of subsequent aligned products.
inline ImageReciprocal image_reciprocal(uint32_t value) {
  if (value == 0)
    return {0, 0};
  const uint32_t leading = std::bit_width(value) - 1;
  const uint32_t index = ((value - (1u << leading)) * 1024) >> leading;
  return {image_norm_reciprocals[index], -static_cast<int>(leading)};
}

inline uint32_t image_polygon_norm(int32_t x, int32_t y) {
  const uint32_t a = std::max(std::abs(x), std::abs(y));
  const uint32_t b = std::min(std::abs(x), std::abs(y));
  const uint32_t projection = std::max(
      {256 * a + 16 * b, 250 * a + 56 * b, 238 * a + 96 * b, 221 * a + 130 * b, 197 * a + 164 * b});
  return (projection + 128) >> 8;
}

struct ImageFootprint {
  std::array<int32_t, 4> gradients{};
  std::array<uint32_t, 2> norms{};
  std::array<int32_t, 2> lods{-8192, -8192};

  /// Normalize the transformed gradient vectors before combining their
  /// polygonal norms. Products retain separate exponents until alignment;
  /// signed ties round upward at both stages. The final norms truncate to
  /// eleven significant bits, including a carry from vector addition.
  std::array<double, 2> direction(uint32_t width = 1, uint32_t height = 1) const {
    std::array<int32_t, 4> products{};
    std::array<int, 2> exponents{};
    const std::array transformed{gradients[0] + gradients[3], gradients[2] - gradients[1],
                                 gradients[0] - gradients[3], gradients[2] + gradients[1]};
    for (uint32_t pair = 0; pair < 2; ++pair) {
      if (norms[pair] == 0)
        continue;
      const auto [reciprocal, reciprocal_exponent] = image_reciprocal(norms[pair]);
      exponents[pair] = 11 + reciprocal_exponent;
      for (uint32_t i = pair * 2; i < pair * 2 + 2; ++i)
        products[i] = (transformed[i] * static_cast<int32_t>(reciprocal) + 512) >> 10;
    }
    const int exponent = norms[0] == 0   ? exponents[1]
                         : norms[1] == 0 ? exponents[0]
                                         : std::max(exponents[0], exponents[1]);
    for (uint32_t i = 0; i < products.size(); ++i) {
      if (norms[i / 2] == 0)
        continue;
      const uint32_t shift = exponent - exponents[i / 2];
      if (shift)
        products[i] = (products[i] + (1 << (shift - 1))) >> shift;
    }
    const auto component = [exponent](int32_t x, int32_t y, uint32_t extent) {
      uint32_t norm = image_polygon_norm(x, y);
      const auto [reciprocal, reciprocal_exponent] = image_reciprocal(extent);
      // Convert back to normalized coordinates before truncating the norm.
      norm = (norm * reciprocal + 512) >> 10;
      if (const uint32_t bits = std::bit_width(norm); bits > 11)
        norm = (norm >> (bits - 11)) << (bits - 11);
      return std::ldexp(double(norm), exponent + reciprocal_exponent - 12);
    };
    const double u = component(products[0] + products[2], products[1] + products[3], width);
    const double v = component(products[0] - products[2], products[1] - products[3], height);
    const int64_t covariance =
        int64_t(gradients[0]) * gradients[1] + int64_t(gradients[2]) * gradients[3];
    return {u, covariance < 0 ? -v : v};
  }
};

/// Major and minor footprint LODs from the principal axes of the gradient matrix.
/// Align each coordinate's screen derivatives before scaling by the image
/// extent, then align U/V for the polygonal norms. Preserve multiplication and
/// norm carry bits through the logarithm. Extents are one for unnormalized input.
inline ImageFootprint image_footprint(double xu, double xv, double yu, double yv, uint32_t width,
                                      uint32_t height) {
  const std::array<double, 4> gradients{xu, xv, yu, yv};
  const std::array largest{std::max(std::abs(xu), std::abs(yu)),
                           std::max(std::abs(xv), std::abs(yv))};
  if (largest[0] == 0 && largest[1] == 0)
    return {};
  const std::array extents{width, height};
  std::array<int, 2> exponents{};
  std::array<double, 4> scaled{};
  for (uint32_t coordinate = 0; coordinate < 2; ++coordinate) {
    if (largest[coordinate] == 0)
      continue;
    int coordinate_exponent;
    std::frexp(largest[coordinate], &coordinate_exponent);
    const int extent_exponent = std::bit_width(extents[coordinate]) - 1;
    // The extent multiplier retains ten fractional bits, without rounding.
    const double extent_mantissa =
        std::floor(std::ldexp(double(extents[coordinate]), 10 - extent_exponent)) / 1024;
    exponents[coordinate] = coordinate_exponent + extent_exponent;
    for (uint32_t i : {coordinate, coordinate + 2}) {
      const double mantissa = std::ldexp(std::abs(gradients[i]), 11 - coordinate_exponent);
      const double significand = std::min(2047.0, std::floor(mantissa + 0.5));
      // Input conversion rounds magnitudes; multiplication rounds signed values.
      scaled[i] = std::floor(std::copysign(significand, gradients[i]) * extent_mantissa + 0.5);
    }
  }
  const int exponent = largest[0] == 0   ? exponents[1]
                       : largest[1] == 0 ? exponents[0]
                                         : std::max(exponents[0], exponents[1]);
  std::array<int32_t, 4> fixed{};
  for (uint32_t i = 0; i < gradients.size(); ++i)
    fixed[i] =
        static_cast<int32_t>(std::floor(std::ldexp(scaled[i], exponents[i % 2] - exponent) + 0.5));
  const uint32_t first_norm = image_polygon_norm(fixed[0] + fixed[3], fixed[1] - fixed[2]);
  const uint32_t second_norm = image_polygon_norm(fixed[0] - fixed[3], fixed[1] + fixed[2]);
  const uint32_t major = (first_norm + second_norm) >> 1;
  // Round the signed difference before taking its magnitude. Negative odd
  // differences therefore produce a larger minor axis than positive ones.
  const uint32_t minor = std::abs((int32_t(first_norm) - int32_t(second_norm)) >> 1);
  const auto logarithm = [exponent](uint32_t axis) -> int32_t {
    if (axis == 0)
      return -8192;
    const uint32_t leading = std::bit_width(axis) - 1;
    const uint32_t denominator = 1u << leading;
    const uint32_t fraction = (axis - denominator) * 1024;
    const uint32_t segment = fraction / (128 * denominator);
    constexpr std::array<uint32_t, 8> bases{0, 175, 330, 471, 599, 717, 827, 929};
    constexpr std::array<uint32_t, 8> slopes{176, 155, 142, 129, 119, 110, 102, 95};
    const uint32_t logarithm =
        (bases[segment] * 128 * denominator +
         slopes[segment] * (fraction - segment * 128 * denominator) + 256 * denominator) /
        (512 * denominator);
    return (exponent - 11 + static_cast<int32_t>(leading)) * 256 + static_cast<int32_t>(logarithm);
  };
  return {fixed, {first_norm, second_norm}, {logarithm(major), logarithm(minor)}};
}

/// Add a footprint offset with the coordinate adder's 24-bit significand.
/// Operand alignment truncates magnitudes; a sum carry drops the low signed
/// bit. This differs from rounding the final sum in either host FP32 mode.
inline double image_sample_coordinate(double coordinate, double offset) {
  if (offset == 0)
    return coordinate;
  int coordinate_exponent, offset_exponent;
  std::frexp(coordinate, &coordinate_exponent);
  std::frexp(offset, &offset_exponent);
  int exponent = coordinate == 0 ? offset_exponent : std::max(coordinate_exponent, offset_exponent);
  const int64_t coordinate_mantissa =
      static_cast<int64_t>(std::trunc(std::ldexp(coordinate, 24 - exponent)));
  const int64_t offset_mantissa =
      static_cast<int64_t>(std::trunc(std::ldexp(offset, 24 - exponent)));
  int64_t sum = coordinate_mantissa + offset_mantissa;
  if (sum >= (1ll << 24) || sum <= -(1ll << 24)) {
    sum >>= 1;
    ++exponent;
  }
  return std::ldexp(double(sum), exponent - 24);
}

/// Select the filter count from the quantized footprint and sampler controls.
/// Bias enlarges only the footprint not already covered by the anisotropy limit.
inline uint32_t image_anisotropic_filter_count(const ImageFootprint &footprint,
                                               uint32_t max_anisotropy, uint32_t threshold,
                                               uint32_t bias, uint32_t perf_mod) {
  if (max_anisotropy == 1)
    return 1;
  const auto [major_lod, minor_lod] = footprint.lods;
  const int32_t footprint_lod =
      std::max(minor_lod, major_lod - int32_t(std::countr_zero(max_anisotropy) * 256));
  const uint32_t perf_scale = image_perf_scales[perf_mod];
  const int32_t bias_lod = static_cast<int32_t>((bias * perf_scale >> 4) * 8);
  // The anisotropy limit can already enlarge the footprint. Apply only the
  // additional enlargement due to bias, after clamping the footprint to one
  // texel. Applying all bias after that clamp miscounts sub-texel footprints.
  const int32_t biased_lod =
      major_lod - std::max(0, footprint_lod) - std::max(0, minor_lod + bias_lod - footprint_lod);
  if (biased_lod <= 0)
    return 1;
  const double fraction_threshold = (threshold * perf_scale >> 4) * 0.25;
  const double ratio = std::exp2(biased_lod / 256.0);
  // Counts above one are even. The threshold extends each lower even count's interval.
  const double count = 2 * std::ceil((ratio - fraction_threshold) / 2);
  return static_cast<uint32_t>(std::clamp(count, 1.0, double(max_anisotropy)));
}

/// Truncate a signed filter step to eleven significant bits before multiplying
/// by the filter index. Rounding each individual offset gives different taps.
inline double image_filter_step_truncate(double value) {
  int exponent;
  const double mantissa = std::frexp(value, &exponent);
  return std::ldexp(std::trunc(mantissa * 2048), exponent - 11);
}

/// GFX11/12 spacing from the footprint's logarithms. The base span and the
/// count coefficient have separate tables and opposite log-index rounding.
inline double image_anisotropic_filter_step(const ImageFootprint &footprint,
                                            uint32_t max_anisotropy, uint32_t count,
                                            uint32_t unbiased_count, bool linear_mip) {
  if (count <= 1)
    return 0;
  const auto [major_lod, minor_lod] = footprint.lods;
  const int32_t rho =
      std::max({0, minor_lod, major_lod - int32_t(std::countr_zero(max_anisotropy) * 256)});
  constexpr std::array<uint16_t, 64> spans{
      128, 129, 131, 132, 134, 135, 137, 138, 140, 141, 143, 144, 146, 147, 149, 151,
      152, 154, 156, 157, 159, 161, 162, 164, 166, 168, 170, 171, 173, 175, 177, 179,
      182, 183, 185, 187, 190, 192, 194, 196, 198, 200, 202, 204, 207, 209, 211, 213,
      216, 218, 220, 223, 225, 228, 230, 233, 235, 238, 240, 243, 246, 248, 251, 254};
  // Linear mip filtering retains one more span bit in the upper half of the
  // logarithm, even for a single-level image or a clamped LOD.
  const uint32_t unit = linear_mip || (rho % 256) < 128 ? 1 : 2;
  uint32_t span = spans[(rho / 4) % 64] / unit * unit;
  if (count != unbiased_count) {
    // Bias redistributes the original footprint over fewer filters. Round the
    // count ratio in fixed7, then truncate at the ORIGINAL span's precision.
    // Normalizing the ratio or the product changes non-power-of-two counts.
    const uint32_t ratio = (128 * unbiased_count + count / 2) / count;
    span = span * ratio / (128 * unit) * unit;
  }
  if (unbiased_count == 2) {
    const uint32_t difference = std::clamp(major_lod - rho, 0, 256);
    if (difference == 0)
      return 0;
    const int taper = difference < 16 ? -6 : std::min(0, int(std::bit_width(difference)) - 8);
    const int fractional_bits = linear_mip || (rho % 256) < 128 ? 12 : 11;
    const uint32_t fixed =
        static_cast<uint32_t>(std::ldexp(double(span), taper + fractional_bits - 7));
    return std::ldexp(double(fixed), rho / 256 - fractional_bits);
  }
  constexpr std::array<int32_t, 9> count_lods{0, 256, 512, 662, 768, 850, 918, 975, 1024};
  constexpr std::array<uint16_t, 65> coefficients{
      1024, 1035, 1046, 1057, 1069, 1080, 1092, 1104, 1116, 1128, 1141, 1153, 1166,
      1178, 1191, 1204, 1217, 1231, 1244, 1257, 1271, 1285, 1299, 1313, 1327, 1342,
      1357, 1371, 1386, 1401, 1417, 1432, 1448, 1463, 1479, 1495, 1512, 1528, 1545,
      1562, 1579, 1596, 1613, 1631, 1649, 1667, 1685, 1703, 1722, 1740, 1759, 1779,
      1798, 1817, 1837, 1857, 1878, 1898, 1919, 1940, 1961, 1982, 2004, 2025, 2048};
  // Division of a nonpositive difference rounds toward zero, giving the ceil.
  const int32_t index =
      64 + std::clamp((major_lod - rho - count_lods[unbiased_count / 2]) / 4, -64, 0);
  return image_filter_step_truncate(std::ldexp(double(span) * coefficients[index], rho / 256 - 18));
}

/// Quantized, symmetric footprint weights. Residual units go to the center
/// filters, preserving a total weight of one for non-power-of-two counts.
inline double image_anisotropic_filter_weight(uint32_t count, uint32_t index) {
  const uint32_t denominator = 128 * std::bit_ceil(count);
  const uint32_t residual = denominator % count;
  const uint32_t first = (count - residual) / 2;
  const uint32_t weight = denominator / count + (index >= first && index < first + residual);
  return static_cast<double>(weight) / denominator;
}

} // namespace rocjitsu::amdgpu

#endif
