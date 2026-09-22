/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
The Functions defined in RtcFunctions.cpp are declared here in RtcFunctions.h.
*/

#ifndef CATCH_UNIT_RTC_HEADERS_RTCFUNCTIONS_H_
#define CATCH_UNIT_RTC_HEADERS_RTCFUNCTIONS_H_
#include <string>

bool check_architecture(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                        int fast_math_present);

bool check_rdc(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
               int fast_math_present);

bool check_denormals_enabled(const char** Combination_CO, int Combination_CO_size,
                             int max_thread_pos, int fast_math_present);

bool check_denormals_disabled(const char** Combination_CO, int Combination_CO_size,
                              int max_thread_pos, int fast_math_present);

bool check_ffp_contract_off(const char** Combination_CO, int Combination_CO_size,
                            int max_thread_pos, int fast_math_present);

bool check_ffp_contract_on(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                           int fast_math_present);

bool check_ffp_contract_fast(const char** Combination_CO, int Combination_CO_size,
                             int max_thread_pos, int fast_math_present);

bool check_fast_math_enabled(const char** Combination_CO, int Combination_CO_size,
                             int max_thread_pos, int fast_math_present);

bool check_fast_math_disabled(const char** Combination_CO, int Combination_CO_size,
                              int max_thread_pos, int fast_math_present);

bool check_slp_vectorize_enabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present);

bool check_slp_vectorize_disabled(const char** Combination_CO, int Combination_CO_size,
                                  int max_thread_pos, int fast_math_present);

bool check_macro(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                 int fast_math_present);

bool check_undef_macro(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                       int fast_math_present);

bool check_header_dir(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                      int fast_math_present);

bool check_warning(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                   int fast_math_present);

bool check_Rpass_inline(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                        int fast_math_present);

bool check_conversionerror_enabled(const char** Combination_CO, int Combination_CO_size,
                                   int max_thread_pos, int fast_math_present);

bool check_conversionerror_disabled(const char** Combination_CO, int Combination_CO_size,
                                    int max_thread_pos, int fast_math_present);

bool check_conversionwarning_enabled(const char** Combination_CO, int Combination_CO_size,
                                     int max_thread_pos, int fast_math_present);

bool check_conversionwarning_disabled(const char** Combination_CO, int Combination_CO_size,
                                      int max_thread_pos, int fast_math_present);

bool check_max_thread(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                      int fast_math_present);

bool check_unsafe_atomic_enabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present);

bool check_unsafe_atomic_disabled(const char** Combination_CO, int Combination_CO_size,
                                  int max_thread_pos, int fast_math_present);

bool check_infinite_num_enabled(const char** Combination_CO, int Combination_CO_size,
                                int max_thread_pos, int fast_math_present);

bool check_infinite_num_disabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present);

bool check_NAN_num_enabled(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                           int fast_math_present);

bool check_NAN_num_disabled(const char** Combination_CO, int Combination_CO_size,
                            int max_thread_pos, int fast_math_present);

bool check_finite_math_enabled(const char** Combination_CO, int Combination_CO_size,
                               int max_thread_pos, int fast_math_present);

bool check_finite_math_disabled(const char** Combination_CO, int Combination_CO_size,
                                int max_thread_pos, int fast_math_present);

bool check_associative_math_enabled(const char** Combination_CO, int Combination_CO_size,
                                    int max_thread_pos, int fast_math_present);

bool check_associative_math_disabled(const char** Combination_CO, int Combination_CO_size,
                                     int max_thread_pos, int fast_math_present);

bool check_signed_zeros_enabled(const char** Combination_CO, int Combination_CO_size,
                                int max_thread_pos, int fast_math_present);

bool check_signed_zeros_disabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present);

bool check_trapping_math_enabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present);

bool check_trapping_math_disabled(const char** Combination_CO, int Combination_CO_size,
                                  int max_thread_pos, int fast_math_present);

std::string checking_IR(const char* kername, const char** extra_CO_IRadded,
                        int extra_CO_IRadded_size, const char** Combination_CO,
                        int Combination_CO_size);

#endif  // CATCH_UNIT_RTC_HEADERS_RTCFUNCTIONS_H_
