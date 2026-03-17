/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
namespace math_reference {

inline float sin_pi(float x) {
    return static_cast<float>(sin_pi(static_cast<double>(x)));
}

inline float cos_pi(float x) {
    return static_cast<float>(cos_pi(static_cast<double>(x)));
}

inline double sin_pi(double x) {
    if (std::isnan(x)) return x;
    if (std::isinf(x)) return std::numeric_limits<double>::quiet_NaN();

    double int_part;
    double frac_part = std::modf(x, &int_part);
    if (frac_part == 0.0) {
        return std::copysign(0.0, x);
    }
    if (std::fabs(frac_part) == 0.5) {
        int n = static_cast<int>(int_part);
        double result = (n % 2 == 0) ? 1.0 : -1.0;
        return x < 0.0 ? -result : result;
    }

    double sign = x < 0.0 ? -1.0 : 1.0;
    double r = std::fmod(std::fabs(x), 2.0);
    double result;
    if (r <= 1.0) {
        result = std::sin(r * M_PI);
    } else {
        result = -std::sin((r - 1.0) * M_PI);
    }
    return result * sign;
}

inline double cos_pi(double x) {
    if (std::isnan(x)) return x;
    if (std::isinf(x)) return std::numeric_limits<double>::quiet_NaN();

    double xd = std::fabs(x);
    double int_part;
    double frac_part = std::modf(xd, &int_part);
    if (frac_part == 0.0) {
        int n = static_cast<int>(int_part);
        return (n % 2 == 0) ? 1.0 : -1.0;
    }
    if (std::fabs(frac_part) == 0.5) {
        return 0.0;
    }

    double r = std::fmod(xd, 2.0);
    double result;
    if (r <= 1.0) {
        result = std::cos(r * M_PI);
    } else {
        result = -std::cos((r - 1.0) * M_PI);
    }
    return result;
}

inline long double sin_pi(long double x) {
    return static_cast<long double>(sin_pi(static_cast<double>(x)));
}

inline long double cos_pi(long double x) {
    return static_cast<long double>(cos_pi(static_cast<double>(x)));
}

// =============================================================================
// Inverse Error Function (erfinv, erfcinv) Reference Implementations
// Ported from ROCm device-libs OCML
// =============================================================================

/*
 * Ported from ROCm device-libs OCML
 * Source: llvm-project/amd/device-libs/ocml/src/erfinv{F,D}.cl
 *         llvm-project/amd/device-libs/ocml/src/erfcinv{F,D}.cl
 */

float erfcinv_ref(float y);
double erfcinv_ref(double y);

inline float erfinv_ref(float x) {
    float ax = std::fabs(x);
    float p;

    if (ax < 0.375f) {
        float t = ax * ax;
        p = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
            std::fma(t, std::fma(t,
                0x1.48b6cap-3f, -0x1.a2930ap-6f), 0x1.65b0b4p-4f), 0x1.5581aep-4f),
                0x1.05aa56p-3f), 0x1.db2748p-3f), 0x1.c5bf8ap-1f);
    } else {
        float w = std::fma(-ax, ax, 1.0f);
        w = -std::log(w);

        if (w < 5.0f) {
            w = w - 2.5f;
            p = std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                    0x1.e2cb10p-26f, 0x1.70966cp-22f), -0x1.d8e6aep-19f), -0x1.26b582p-18f),
                    0x1.ca65b6p-13f), -0x1.48a810p-10f), -0x1.11c9dep-8f), 0x1.f91ec6p-3f),
                    0x1.805c5ep+0f);
        } else {
            w = std::sqrt(w) - 3.0f;
            p = std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                    -0x1.a3e136p-13f, 0x1.a76ad6p-14f), 0x1.61b8e4p-10f), -0x1.e17bcep-9f),
                    0x1.7824f6p-8f), -0x1.f38baep-8f), 0x1.354afcp-7f), 0x1.006db6p+0f),
                    0x1.6a9efcp+1f);
        }
    }

    float ret = p * ax;

    ret = ax > 1.0f ? std::numeric_limits<float>::quiet_NaN() : ret;
    ret = ax == 1.0f ? std::numeric_limits<float>::infinity() : ret;

    return std::copysign(ret, x);
}

inline double erfinv_ref(double x) {
    double ax = std::fabs(x);
    double ret;

    if (ax < 0.375) {
        double t = ax * ax;
        ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t,
                  0x1.c5ec06cd8002bp-2, -0x1.bb7dd47aef0d6p-1), 0x1.d189992eccdb6p-1), -0x1.10ec180cde957p-1),
                  0x1.05cce379dd66fp-2), -0x1.6b9067e3dae74p-5), 0x1.5f7f0487c11a3p-5), 0x1.e0fbf22b2350cp-6),
                  0x1.2ce26322b7f90p-5), 0x1.5ebeeee81dd31p-5), 0x1.a7cacb897f0d4p-5), 0x1.0a130d62cba32p-4),
                  0x1.62847c8653359p-4), 0x1.053c2c0a5e083p-3), 0x1.db29fb2feec72p-3), 0x1.c5bf891b4ef6ap-1);
        ret = ax * ret;
    } else if (ax < 0x1.fffep-1) {
        double w = -std::log(std::fma(-ax, ax, 1.0));

        if (w < 6.25) {
            w = w - 3.125;
            ret = std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w,
                      -0x1.135d2e746e627p-68, -0x1.8ddf93324d327p-63),  0x1.7b83eef0b7c9fp-60), 0x1.9ba72cd589b91p-57),
                      -0x1.33689090a6b96p-53), 0x1.82e11898132e0p-56),  0x1.de4acfd9e26bap-48), -0x1.6d33eed66c487p-45),
                      -0x1.6f2167040d8e2p-44), 0x1.72a22c2d77e20p-39), -0x1.c8859c4e5c0afp-37), -0x1.dc583d118a561p-35),
                      0x1.20f47ccf46b3cp-30), -0x1.1a9e38dc84d60p-28), -0x1.f36cd6d3d46a9p-26), 0x1.c6b4f5d03b787p-22),
                      -0x1.6e8a5434ae8a2p-20), -0x1.d1d1f7b8736f6p-17),  0x1.879c2a212f024p-13), -0x1.845769484fca8p-11),
                      -0x1.8b6c33114f909p-8), 0x1.ebd80d9b13e28p-3),   0x1.a755e7c99ae86p+0);
        } else if (w < 16.0) {
            w = std::sqrt(w) - 3.25;
            ret = std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w,
                       0x1.3040f87dbd932p-29, 0x1.85cbe52878635p-24), -0x1.2777453dd3955p-22), 0x1.395abcd554c6cp-26),
                       0x1.936388a3790adp-20), -0x1.0d5db812b5083p-18),  0x1.8860cd5d652f6p-19), 0x1.a29a0cacdfb23p-17),
                       -0x1.8cef1f80281f2p-15), 0x1.1e684d0b9188ap-14),  0x1.932cd54c8a222p-16), -0x1.7448a89ef8aa3p-12),
                       0x1.f3cc55ad40c25p-11), -0x1.ba924132f38b1p-10),  0x1.468eeca533cf8p-9), -0x1.ebadabb891bbdp-9),
                       0x1.5ffcfe5b76afcp-8), 0x1.0158a6d641d39p+0),   0x1.8abcc380d5a48p+1);
        } else {
            w = std::sqrt(w) - 5.0;
            ret = std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                  std::fma(w, std::fma(w, std::fma(w, std::fma(w,
                      -0x1.dcec3a7785389p-36, -0x1.18feec0e38727p-32),  0x1.9e6bf2dda45e3p-30), -0x1.0468fb24e2f5fp-28),
                      0x1.05ac6a8fba182p-27), -0x1.0102e495fb9c0p-26),  0x1.f4c20e1334af8p-26), -0x1.22d220fdf9c3ep-24),
                      0x1.ebc8bb824cb54p-23), -0x1.0a8d40ea372ccp-20),  0x1.2fbd29d093d2bp-18), -0x1.4a3497e1e0facp-16),
                      0x1.3ebf4eb00938fp-14), -0x1.c2f36a8fc5d53p-13), -0x1.22ea5df04047cp-13), 0x1.02a30d1fba0dcp+0),
                      0x1.3664ddd1ad7fbp+2);
        }
        ret = ax * ret;
    } else {
        double s = std::sqrt(-std::log(1.0 - ax));
        double t = 1.0 / s;

        if (ax < 0x1.fffffffep-1) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t,
                      0x1.c4bd831a51669p+7, -0x1.66af45b757c26p+9), 0x1.061b293ee1671p+10), -0x1.d4aa0fd7248e9p+9),
                      0x1.1eebb0088748dp+9), -0x1.ff4cb6c165efep+7), 0x1.59c379a609255p+6), -0x1.762b2677680c6p+4),
                      0x1.7626132cf7c5ap+2), -0x1.a298cc231a949p+0), -0x1.9fa2d429b22cap-6), 0x1.00131c4b15d15p+0);
        } else {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t,
                      0x1.e1f462cc8e58ap+7, -0x1.dd260d25bee8dp+8), 0x1.af7dab6c206e6p+8), -0x1.d97c75a0f5809p+7),
                      0x1.632c20bf45d30p+6), -0x1.8e4908179a727p+4), 0x1.89538a73a2c3cp+2), -0x1.aad8569b3607dp+0),
                      -0x1.80d1bec4b54cbp-6), 0x1.001006f90ea2cp+0);
        }

        ret = s * ret;
    }

    ret = ax > 1.0 ? std::numeric_limits<double>::quiet_NaN() : ret;
    ret = ax == 1.0 ? std::numeric_limits<double>::infinity() : ret;

    return std::copysign(ret, x);
}

inline long double erfinv_ref(long double x) {
    return static_cast<long double>(erfinv_ref(static_cast<double>(x)));
}

inline float erfcinv_ref(float y) {
    float ret;

    if (y > 0.625f) {
        ret = erfinv_ref(1.0f - y);
    } else if (y > 0x1.0p-10f) {
        // Medium range
        float t = -std::log(y * (2.0f - y)) - 3.125f;
        ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t,
                  0x1.7ee662p-31f, -0x1.3f5a80p-28f), -0x1.b638f0p-26f), 0x1.c9ccc6p-22f),
                  -0x1.72f8aep-20f), -0x1.d21aa6p-17f), 0x1.87aebcp-13f), -0x1.8455d4p-11f),
                  -0x1.8b6ca4p-8f), 0x1.ebd80cp-3f), 0x1.a755e8p+0f);
        ret = std::fma(-y, ret, ret);
    } else {
        float s = std::sqrt(-std::log(y));
        float t = 1.0f / s;

        if (y > 0x1.0p-42f) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t,
                      -0x1.57221ep+0f, 0x1.7f6144p+1f), -0x1.98dd40p+1f), 0x1.2c9066p+1f),
                      -0x1.3a07eap+0f), -0x1.ba546cp-5f), 0x1.004e66p+0f);
        } else {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t,
                      -0x1.649c6ap+4f, 0x1.8fa8fap+4f), -0x1.a112d8p+3f), 0x1.309d98p+2f),
                      -0x1.919488p+0f), -0x1.c084ecp-6f), 0x1.00143ep+0f);
        }
        ret = s * ret;
    }

    ret = ((y < 0.0f) || (y > 2.0f)) ? std::numeric_limits<float>::quiet_NaN() : ret;
    ret = y == 0.0f ? std::numeric_limits<float>::infinity() : ret;
    ret = y == 2.0f ? -std::numeric_limits<float>::infinity() : ret;

    return ret;
}

inline double erfcinv_ref(double y) {
    double ret;

    if (y > 0.625) {
        ret = erfinv_ref(1.0 - y);
    } else if (y > 0x1.0p-10) {
        double t = -std::log(y * (2.0 - y)) - 3.125;

        ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t, std::fma(t, std::fma(t,
              std::fma(t, std::fma(t,
                  0x1.1267a785a1166p-69, -0x1.a6581051dd484p-63), 0x1.2b2956fc047a4p-60), 0x1.ad835aed5cc07p-57),
                  -0x1.25e0612eae68fp-53), 0x1.a0cab63f02a91p-57), 0x1.d9227af501adbp-48), -0x1.6c3ad559a9b4ep-45),
                  -0x1.6cafa36036318p-44), 0x1.72879641e158fp-39), -0x1.c89d755f7fff8p-37), -0x1.dc51171ddae3ap-35),
                  0x1.20f512744ae65p-30), -0x1.1a9e5f4bcfcd8p-28), -0x1.f36ce926b83e8p-26), 0x1.c6b4f6c7cfa1ep-22),
                  -0x1.6e8a53e0c2026p-20), -0x1.d1d1f7bf4570bp-17), 0x1.879c2a20cc3e2p-13), -0x1.8457694844d14p-11),
                  -0x1.8b6c33114edadp-8), 0x1.ebd80d9b13e14p-3), 0x1.a755e7c99ae86p+0);
        ret = std::fma(-y, ret, ret);
    } else {
        double s = std::sqrt(-std::log(y));
        double t = 1.0 / s;

        if (y > 0x1.0p-19) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t,
                      0x1.8b3cfc98a5212p+4, -0x1.907bcdab54a4ep+6), 0x1.7659cf8216d7dp+7), -0x1.ac222777f664dp+7),
                      0x1.4f2f8e33151acp+7), -0x1.7d7d1eb301c4cp+6), 0x1.48e630c1c77e7p+5), -0x1.c63e7d0e327f6p+3),
                      0x1.225b286aeb0dfp+2), -0x1.82a4acc22b05dp+0), -0x1.0a88271680e57p-5), 0x1.001f6acebb122p+0);
        } else if (y > 0x1.0p-40) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t,
                      0x1.0fdcb40bf066dp+9, -0x1.870ddeaa832dbp+10), 0x1.035c39e0428c4p+11), -0x1.a4d3c54a3ec14p+10),
                      0x1.d382aee6efae8p+9), -0x1.79f9e26565bc1p+8), 0x1.d00e058ce9abap+6), -0x1.c7d1e01821eb3p+4),
                      0x1.9d930ba7a3111p+2), -0x1.af47941dd2baap+0), -0x1.787ecc823998bp-6), 0x1.000fae5fb73e3p+0);
        } else if (y > 0x1.0p-82) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t,
                      0x1.c9e5b8e31c18ep+13, -0x1.c866153b1bce6p+14), 0x1.a386b3b4fb25cp+14), -0x1.d7bf378e7b5fbp+13),
                      0x1.6b416de0a7a75p+12), -0x1.9757c1cf44e90p+10), 0x1.5b56ededbaa8cp+8), -0x1.da79924b4d155p+5),
                      0x1.2ba25315d612bp+3), -0x1.de5808fbd786dp+0), -0x1.04e014b9fc507p-6), 0x1.000788df1c89fp+0);
        } else if (y > 0x1.0p-200) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t,
                      0x1.ff518aae00301p+18, -0x1.5781ef98c6aa9p+19), 0x1.a9511b21c7715p+18), -0x1.41d8f1455b21ep+17),
                      0x1.4d4a3d4025a4cp+15), -0x1.f640fe7077996p+12), 0x1.1faf674f42181p+10), -0x1.080c5cd81d791p+7),
                      0x1.c0ae370098ef4p+3), -0x1.08ebd67dc005ap+1), -0x1.5cf3329e72289p-7), 0x1.00035e75f27e2p+0);
        } else if (y > 0x1.0p-400) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t,
                      -0x1.d554f00bf9d81p+20, 0x1.8456711ff3627p+20), -0x1.26c90acc5daafp+19), 0x1.106501cdef815p+17),
                      -0x1.57a4c95601c04p+14), 0x1.3ca627cbaede6p+11), -0x1.c716e091922fbp+7), 0x1.292f8f6e8bc75p+4),
                      -0x1.1b469c212bd5fp+1), -0x1.04977fb6d0462p-7), 0x1.0001dc9f52f8ap+0);
        } else if (y > 0x1.0p-900) {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t,
                      -0x1.21913925f3a73p+25, 0x1.4aa2fba282b9bp+24), -0x1.5a2a3f9742896p+22), 0x1.b8ee3895772e8p+19),
                      -0x1.7f2ce0b036be4p+16), 0x1.e62ab1bcbb738p+12), -0x1.e0ed2965d2a06p+8), 0x1.b0c16705263e5p+4),
                      -0x1.334f9a732ecc7p+1), -0x1.65f60412f9578p-8), 0x1.0000e0bda43b5p+0);
        } else {
            ret = std::fma(t, std::fma(t, std::fma(t, std::fma(t,
                  std::fma(t, std::fma(t,
                      -0x1.e3d70f1fdc7bep+11, 0x1.28d9acd5b9596p+10), -0x1.554c1ce591414p+7), 0x1.15b1e5a1fe7f5p+4),
                      -0x1.1aa8e6f616c69p+1), -0x1.f6803b3b4d6ccp-8), 0x1.00019ac5bed2ap+0);
        }
        ret = s * ret;
    }

    ret = ((y < 0.0) || (y > 2.0)) ? std::numeric_limits<double>::quiet_NaN() : ret;
    ret = y == 0.0 ? std::numeric_limits<double>::infinity() : ret;
    ret = y == 2.0 ? -std::numeric_limits<double>::infinity() : ret;

    return ret;
}

inline long double erfcinv_ref(long double y) {
    return static_cast<long double>(erfcinv_ref(static_cast<double>(y)));
}

} // namespace math_reference
