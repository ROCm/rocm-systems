// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Fixed-point sine/cosine polynomial coefficients shared by the empirical model.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_TRIG_COEFFICIENTS_H_
#define RJ_ARITH_TRIG_COEFFICIENTS_H_

#include "common/bits.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trig {

// Captured rocjitsu RDNA3/4 coefficients in units of 2^-26, transposed for
// SIMD gathers. Rows hold constant, linear, quadratic and cubic terms.
// Columns 0..15 approximate sine and 16..31 cosine over the first octant.
static const int32_t coefficients[4][32] = {
    {0,        3292874,  6577818,  9846914,  13092288, 16306122, 19480674, 22608294,
     25681450, 28692736, 31634898, 34500850, 37283686, 39976702, 42573412, 45067558,
     67108864, 67028028, 66785716, 66382510, 65819384, 65097694, 64219178, 63185952,
     62000504, 60665694, 59184734, 57561192, 55798980, 53902344, 51875852, 49724388},
    {3294200,  3290232,  3278340,  3258548,  3230908,  3195480,  3152356,  3101636,
     3043444,  2977924,  2905224,  2825532,  2739032,  2645928,  2546452,  2440844,
     0,        -161636,  -322884,  -483352,  -642664,  -800416,  -956248,  -1109776,
     -1260628, -1408448, -1552872, -1693552, -1830152, -1962348, -2089812, -2212244},
    {0,      -3968,  -7928,  -11864, -15778, -19650, -23476, -27246, -30948, -34576, -38122,
     -41576, -44928, -48176, -51304, -54310, -80872, -80774, -80482, -79996, -79318, -78448,
     -77390, -76144, -74716, -73104, -71320, -69364, -67242, -64956, -62514, -59922},
    {-1322, -1320, -1312, -1304, -1290, -1274, -1256, -1234, -1210, -1182, -1152,
     -1118, -1082, -1042, -1002, -958,  32,    98,    162,   226,   290,   352,
     414,   476,   536,   594,   652,   708,   762,   814,   864,   912}};

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::trig

#endif
