/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sindxl
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
This file has definition of functions for the following functinality:

1) get_combi_string_vec() : Retrieve the combination string which contains
contains the combination of block name which indicate the respective compiler
option seperated by ':' from RtcConfig.jason file and returns them in the
form of vectors.

2) split_comb_string() : The combination of blockname which are seperated by
':' has to split so that their respective compiler option can be retrieved
from the json file. This functn internally calls calling_combination_function()
for each of the combination of compiler options. This function returns a
int value i.e the total failed cases in that combination which is obtained
by calling_combination_function() function.

3) calling_combination_function() : This function takes the combination of
blockname as the input. The respective compiler option for that block name is
retrieved from the json file and store the compiler options in a array.
calling_resp_function() is called which mapps the compiler option function
which has to be called with a set of required parameters
(combination of compiler options is one among them). this function returns
the status of execution ie 1 or 0 (bool).

4) getblock_fromconfig() : This function is used to open the RtcConfig.json
file and return the blocks.

5) get_string_parameters() and get_array_parameters() : retrieved the
parameters of the respective block name.

*/

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <picojson.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "headers/RtcUtility.h"
#include "headers/RtcFunctions.h"
#include "headers/RtcKernels.h"
#include <hip_test_common.hh>
#include "headers/printf_common.h"

#pragma clang diagnostic ignored "-Wunused-but-set-variable"

std::vector<std::string> get_combi_string_vec() {
  picojson::array combi_string = get_array_parameters("Combi_CO", "all_compier_options");
  std::vector<std::string> combi_string_list;
  for (auto& indx : combi_string) {
    combi_string_list.push_back(indx.get<std::string>());
  }
  return combi_string_list;
}

int split_comb_string(std::string option) {
  int start_collon_index = option.find(':');
  int start_index = 0;
  std::vector<std::string> combi_block_name;
  while (start_collon_index != std::string::npos) {
    std::string singleoption = option.substr(start_index, start_collon_index - start_index);
    combi_block_name.push_back(singleoption);
    start_index = start_collon_index + 1;
    start_collon_index = option.find(':', start_index);
  }
  std::string last_option = option.substr(start_index, option.length() - start_index);
  combi_block_name.push_back(last_option);
  return calling_combination_function(combi_block_name);
}

int calling_combination_function(std::vector<std::string> combi_vec_list) {
  int combi_size = combi_vec_list.size();
  int fast_math_present = -1, undef_present = 0;
  int max_thread_position;
  std::vector<std::string> hold_CO(combi_size, "");
  const char** Combination_CO = new const char*[combi_size];
  picojson::array undef_compiler_option = get_array_parameters("compiler_option", "undef_macro");
  std::vector<std::string> undef_CO_vec;
  for (auto& indx : undef_compiler_option) {
    undef_CO_vec.push_back(indx.get<std::string>());
  }
  for (int i = 0; i < combi_size; i++) {
    if (combi_vec_list[i] == "max_thread") {
      std::string ready_CO = get_string_parameters("ready_compiler_option", combi_vec_list[i]);
      hold_CO[i] = ready_CO;
      if (combi_vec_list[i] == "max_thread") {
        max_thread_position = i;
      }
    } else if (combi_vec_list[i] == "header_dir") {
      std::string retrived_CO = get_string_parameters("compiler_option", "header_dir");
      std::string str = "pwd";
      const char* cmd = str.c_str();
      CaptureStream capture(stdout);
      capture.Begin();
      system(cmd);
      capture.End();
      std::string wor_dir = capture.getData();
      std::string break_dir = wor_dir.substr(0, wor_dir.find("build"));
      std::string append_str = "catch/unit/rtc/headers";
      std::string CO = retrived_CO + " " + break_dir + append_str;
      hold_CO[i] = CO;
    } else if (combi_vec_list[i] == "architecture") {
      std::string retrived_CO = get_string_parameters("compiler_option", "architecture");
      hipDeviceProp_t prop;
      HIP_CHECK(hipGetDeviceProperties(&prop, 0));
      std::string actual_architecture = prop.gcnArchName;
      std::string complete_CO = retrived_CO + actual_architecture;
      hold_CO[i] = complete_CO;
    } else if (check_positive_CO_present(combi_vec_list[i]) == 1) {
      std::string positive_CO = get_string_parameters("compiler_option", combi_vec_list[i]);
      hold_CO[i] = positive_CO;
      if (combi_vec_list[i] == "fast_math") fast_math_present = 1;
    } else if (check_negative_CO_present(combi_vec_list[i]) == 1) {
      std::string split_block_name = combi_vec_list[i].substr(3, combi_vec_list[i].length() - 3);
      std::string negative_CO = get_string_parameters("reverse_compiler_option", split_block_name);
      hold_CO[i] = negative_CO;
      if (split_block_name == "fast_math") fast_math_present = 0;
    } else if (combi_vec_list[i] == "conversion_error" ||
               combi_vec_list[i] == "conversion_no_error" ||
               combi_vec_list[i] == "conversion_no_warning" ||
               combi_vec_list[i] == "conversion_warning") {
      picojson::array compiler_option = get_array_parameters("compiler_option", "error");
      std::vector<std::string> CO_vec;
      for (auto& indx : compiler_option) {
        CO_vec.push_back(indx.get<std::string>());
      }
      if (combi_vec_list[i] == "conversion_error") {
        hold_CO[i] = CO_vec[0];
      } else if (combi_vec_list[i] == "conversion_no_error") {
        hold_CO[i] = CO_vec[1];
      } else if (combi_vec_list[i] == "conversion_warning") {
        hold_CO[i] = CO_vec[2];
      } else if (combi_vec_list[i] == "conversion_no_warning") {
        hold_CO[i] = CO_vec[3];
      }
    } else if (combi_vec_list[i] == "off_ffp_contract" || combi_vec_list[i] == "on_ffp_contract" ||
               combi_vec_list[i] == "fast_ffp_contract" ||
               combi_vec_list[i] == "pragmas_ffp_contract") {
      picojson::array compiler_option = get_array_parameters("compiler_option", "ffp_contract");
      std::vector<std::string> CO_vec;
      for (auto& indx : compiler_option) {
        CO_vec.push_back(indx.get<std::string>());
      }
      if (combi_vec_list[i] == "off_ffp_contract") {
        hold_CO[i] = CO_vec[0];
      } else if (combi_vec_list[i] == "on_ffp_contract") {
        hold_CO[i] = CO_vec[1];
      } else if (combi_vec_list[i] == "fast_ffp_contract") {
        hold_CO[i] = CO_vec[2];
      } else if (combi_vec_list[i] == "pragmas_ffp_contract") {
        hold_CO[i] = CO_vec[3];
      }
    } else if (combi_vec_list[i] == "undef_macro") {
      hold_CO[i] = undef_CO_vec[1].c_str();
      undef_present = 1;
    } else {
      WARN("BLOCK NAME " << combi_vec_list[i] << " NOT PRESENT");
    }
    Combination_CO[i] = hold_CO[i].c_str();
  }
  int errors = 0;
  for (int j = 0; j < combi_size; j++) {
    std::string block_name = combi_vec_list[j].c_str();
    if (!calling_resp_function(block_name, Combination_CO, combi_size, max_thread_position,
                               fast_math_present)) {
      errors++;
    }
    Combination_CO[j] = hold_CO[j].c_str();
  }
  return errors;
}

int check_positive_CO_present(std::string find_string) {
  static std::vector<std::string> positive_CO = {
      "macro",        "warning",     "rdc",           "denormals",   "fp32_div_sqrt",
      "Rpass_inline", "fast_math",   "slp_vectorize", "amdgpu_ieee", "unsafe_atomic",
      "infinite_num", "NAN_num",     "slp_vectorize", "math_errno",  "associative_math",
      "signed_zeros", "finite_math", "trapping_math"};
  if (std::find(positive_CO.begin(), positive_CO.end(), find_string) != positive_CO.end())
    return 1;
  else
    return 0;
}

int check_negative_CO_present(std::string find_string) {
  static std::vector<std::string> negative_CO = {
      "no_fast_math",   "no_fp32_div_sqrt", "no_denormals",        "no_slp_vectorize",
      "no_amdgpu_ieee", "no_unsafe_atomic", "no_infinite_num",     "no_slp_vectorize",
      "no_NAN_num",     "no_math_errno",    "no_associative_math", "no_signed_zeros",
      "no_finite_math", "no_trapping_math"};
  if (std::find(negative_CO.begin(), negative_CO.end(), find_string) != negative_CO.end())
    return 1;
  else
    return 0;
}

bool calling_resp_function(const std::string block_name, const char** Combination_CO,
                           int Combination_CO_size, int max_thread_position,
                           int fast_math_present) {
  if (block_name == "max_thread") {
    return check_max_thread(Combination_CO, Combination_CO_size, max_thread_position,
                            fast_math_present);
  } else if (block_name == "architecture") {
    return check_architecture(Combination_CO, Combination_CO_size, max_thread_position,
                              fast_math_present);
  } else if (block_name == "rdc") {
    return check_rdc(Combination_CO, Combination_CO_size, max_thread_position, fast_math_present);
  } else if (block_name == "denormals") {
    return check_denormals_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                   fast_math_present);
  } else if (block_name == "no_denormals") {
    return check_denormals_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                    fast_math_present);
  } else if (block_name == "warning") {
    return check_warning(Combination_CO, Combination_CO_size, max_thread_position,
                         fast_math_present);
  } else if (block_name == "conversion_error") {
    return check_conversionerror_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                         fast_math_present);
  } else if (block_name == "conversion_no_error") {
    return check_conversionerror_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                          fast_math_present);
  } else if (block_name == "conversion_warning") {
    return check_conversionwarning_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                           fast_math_present);
  } else if (block_name == "conversion_no_warning") {
    return check_conversionwarning_disabled(Combination_CO, Combination_CO_size,
                                            max_thread_position, fast_math_present);
  } else if (block_name == "Rpass_inline") {
    return check_Rpass_inline(Combination_CO, Combination_CO_size, max_thread_position,
                              fast_math_present);
  } else if (block_name == "macro") {
    return check_macro(Combination_CO, Combination_CO_size, max_thread_position, fast_math_present);
  } else if (block_name == "undef_macro") {
    return check_undef_macro(Combination_CO, Combination_CO_size, max_thread_position,
                             fast_math_present);
  } else if (block_name == "header_dir") {
    return check_header_dir(Combination_CO, Combination_CO_size, max_thread_position,
                            fast_math_present);
  } else if (block_name == "no_fast_math") {
    return check_fast_math_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                    fast_math_present);
  } else if (block_name == "fast_math") {
    return check_fast_math_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                   fast_math_present);
  } else if (block_name == "off_ffp_contract") {
    return check_ffp_contract_off(Combination_CO, Combination_CO_size, max_thread_position,
                                  fast_math_present);
  } else if (block_name == "on_ffp_contract") {
    return check_ffp_contract_on(Combination_CO, Combination_CO_size, max_thread_position,
                                 fast_math_present);
  } else if (block_name == "fast_ffp_contract") {
    return check_ffp_contract_fast(Combination_CO, Combination_CO_size, max_thread_position,
                                   fast_math_present);
  } else if (block_name == "no_unsafe_atomic") {
    return check_unsafe_atomic_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                        fast_math_present);
  } else if (block_name == "unsafe_atomic") {
    return check_unsafe_atomic_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                       fast_math_present);
  } else if (block_name == "no_slp_vectorize") {
    return check_slp_vectorize_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                        fast_math_present);
  } else if (block_name == "slp_vectorize") {
    return check_slp_vectorize_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                       fast_math_present);
  } else if (block_name == "infinite_num") {
    return check_infinite_num_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                      fast_math_present);
  } else if (block_name == "no_infinite_num") {
    return check_infinite_num_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                       fast_math_present);
  } else if (block_name == "NAN_num") {
    return check_NAN_num_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                 fast_math_present);
  } else if (block_name == "no_NAN_num") {
    return check_NAN_num_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                  fast_math_present);
  } else if (block_name == "finite_math") {
    return check_finite_math_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                     fast_math_present);
  } else if (block_name == "no_finite_math") {
    return check_finite_math_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                      fast_math_present);
  } else if (block_name == "associative_math") {
    return check_associative_math_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                          fast_math_present);
  } else if (block_name == "no_associative_math") {
    return check_associative_math_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                           fast_math_present);
  } else if (block_name == "signed_zeros") {
    return check_signed_zeros_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                      fast_math_present);
  } else if (block_name == "no_signed_zeros") {
    return check_signed_zeros_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                       fast_math_present);
  } else if (block_name == "trapping_math") {
    return check_trapping_math_enabled(Combination_CO, Combination_CO_size, max_thread_position,
                                       fast_math_present);
  } else if (block_name == "no_trapping_math") {
    return check_trapping_math_disabled(Combination_CO, Combination_CO_size, max_thread_position,
                                        fast_math_present);
  } else {
    WARN("BLOCK NAME '" << block_name << "' not found");
    return 0;
  }
}

picojson::array getblock_fromconfig() {
  std::string str = "pwd";
  const char* cmd = str.c_str();
  CaptureStream capture(stdout);
  capture.Begin();
  system(cmd);
  capture.End();
  std::string wor_dir = capture.getData();
  std::string break_dir = wor_dir.substr(0, wor_dir.find("build"));
  std::string append_str = "catch/unit/rtc/RtcConfig.json";
  std::string config_path = break_dir + append_str;
  std::string returnValue = "";
  std::ifstream json_file(config_path.c_str());
  if (!json_file.is_open()) {
    WARN("Error loading config.jason");
    exit(0);
  }
  std::string json_str((std::istreambuf_iterator<char>(json_file)),
                       std::istreambuf_iterator<char>());
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    WARN("empty config.jason");
    exit(0);
  }
  picojson::array& blocks = v.get<picojson::array>();
  return blocks;
}

std::string get_string_parameters(std::string para_name_to_retrieve, std::string block_name) {
  std::string returnValue = "";
  picojson::array blocks = getblock_fromconfig();
  for (picojson::value& block : blocks) {
    picojson::object& block_obj = block.get<picojson::object>();
    std::string blk_name = block_obj.at("block_name").get<std::string>();
    if (blk_name == block_name) {
      if (para_name_to_retrieve == "compiler_option") {
        std::string compiler_opt = block_obj.at("compiler_option").get<std::string>();
        returnValue += compiler_opt;
      } else if (para_name_to_retrieve == "Target_Vals") {
        std::string Target_Vals = block_obj.at("Target_Vals").get<std::string>();
        returnValue += Target_Vals;
      } else if (para_name_to_retrieve == "kernel_name") {
        std::string ker_name = block_obj.at("kernel_name").get<std::string>();
        returnValue += ker_name;
      } else if (para_name_to_retrieve == "reverse_compiler_option") {
        std::string reverse = block_obj.at("reverse_compiler_option").get<std::string>();
        returnValue += reverse;
      } else if (para_name_to_retrieve == "ready_compiler_option") {
        std::string ready_CO = block_obj.at("ready_compiler_option").get<std::string>();
        returnValue += ready_CO;
      } else {
        WARN("REQUESTED FIELD not present : " << para_name_to_retrieve);
      }
    } else {
      continue;
    }
  }
  return returnValue;
}

picojson::array get_array_parameters(std::string para_name_to_retrieve, std::string block_name) {
  std::string returnValue = "";
  picojson::array blocks = getblock_fromconfig();
  for (picojson::value& block : blocks) {
    picojson::object& block_obj = block.get<picojson::object>();
    std::string blk_name = block_obj.at("block_name").get<std::string>();
    if (blk_name == block_name) {
      if (para_name_to_retrieve == "Target_Vals") {
        picojson::array& Target_Vals = block_obj.at("Target_Vals").get<picojson::array>();
        return Target_Vals;
      } else if (para_name_to_retrieve == "single_CO") {
        picojson::array& single_CO = block_obj.at("single_CO").get<picojson::array>();
        return single_CO;
      } else if (para_name_to_retrieve == "Combi_CO") {
        picojson::array& Combi_CO = block_obj.at("Combi_CO").get<picojson::array>();
        return Combi_CO;
      } else if (para_name_to_retrieve == "Input_Vals") {
        picojson::array& Input_Vals = block_obj.at("Input_Vals").get<picojson::array>();
        return Input_Vals;
      } else if (para_name_to_retrieve == "Expected_Results") {
        picojson::array& Expected = block_obj.at("Expected_Results").get<picojson::array>();
        return Expected;
      } else if (para_name_to_retrieve == "Expected_Results_for_no") {
        picojson::array& Expected_for_no =
            block_obj.at("Expected_Results_for_no").get<picojson::array>();
        return Expected_for_no;
      } else if (para_name_to_retrieve == "compiler_option") {
        picojson::array& compiler_option = block_obj.at("compiler_option").get<picojson::array>();
        return compiler_option;
      } else if (para_name_to_retrieve == "reverse_compiler_option") {
        picojson::array& reverse_compiler_option =
            block_obj.at("reverse_compiler_option").get<picojson::array>();
        return reverse_compiler_option;
      } else if (para_name_to_retrieve == "Headers") {
        picojson::array& Headers = block_obj.at("Headers").get<picojson::array>();
        return Headers;
      } else if (para_name_to_retrieve == "Src_headers") {
        picojson::array& Src_headers = block_obj.at("Src_headers").get<picojson::array>();
        return Src_headers;
      } else if (para_name_to_retrieve == "depending_comp_optn") {
        picojson::array& depending_comp_optn =
            block_obj.at("depending_comp_optn").get<picojson::array>();
        return depending_comp_optn;
      } else {
        WARN("REQUESTED FIELD not present : " << para_name_to_retrieve);
        return picojson::array();
      }
    } else {
      continue;
    }
  }
  WARN("REQUESTED BLOCK " << block_name << " is not present ");
  return picojson::array();
}
