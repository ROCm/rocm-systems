/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
