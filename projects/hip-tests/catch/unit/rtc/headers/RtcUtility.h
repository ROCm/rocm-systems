/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
The Functions defined in RtcUtility.cpp are declared here in RtcUtility.h.
*/

#ifndef CATCH_UNIT_RTC_HEADERS_RTCUTILITY_H_
#define CATCH_UNIT_RTC_HEADERS_RTCUTILITY_H_
#include <picojson.h>
#include <vector>
#include <string>

std::vector<std::string> get_combi_string_vec();

int split_comb_string(std::string option);

int calling_combination_function(std::vector<std::string> combi_vec_list);

int check_positive_CO_present(std::string find_string);

int check_negative_CO_present(std::string find_string);

bool calling_resp_function(const std::string block_name, const char** Combination_CO,
                           int Combination_CO_size, int max_thread_position, int fast_math_present);

picojson::array getblock_fromconfig();

std::string get_string_parameters(std::string para_name_to_retrieve, std::string block_name);

picojson::array get_array_parameters(std::string para_name_to_retrieve, std::string block_name);

#endif  // CATCH_UNIT_RTC_HEADERS_RTCUTILITY_H_
