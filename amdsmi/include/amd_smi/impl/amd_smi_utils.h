/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_
#define AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_

#include <dirent.h>
#include <limits>
#include <type_traits>
#include <iterator>
#include <ostream>
#include <string>

#include "amd_smi/amdsmi.h"



extern "C" {
    void amdsmi_free_name_value_pairs(void *p);
}

amdsmi_status_t smi_amdgpu_get_pcie_speed_from_pcie_type(uint16_t pcie_type, uint32_t *pcie_speed);
std::string smi_split_string(std::string str, char delim);
std::vector<std::string> split_string(const std::string& line, char delim);
std::string smi_amdgpu_get_status_string(amdsmi_status_t ret, bool fullStatus);

uint32_t smi_brcm_get_value_u32(const std::string &folder, const std::string &file_name);
std::string smi_brcm_get_value_string(const std::string &folder, const std::string &file_name);
amdsmi_status_t smi_brcm_execute_cmd_get_data(const std::string &command, std::string *data);

amdsmi_status_t smi_clear_char_and_reinitialize(char buffer[], uint32_t len,
                                                    std::string newString);





/**
 *  @brief Get an int environment var or return default if does not exist
 *
 *  @details Given a const char* @p name and a default int @p def
 *  and call getenv with name. On any error, return default int
 *
 *  @param[in] name a const char* containing ENV var name
 *
 *  @param[in] def default int in case of error
 *
 *  @retval int of environment variable
 */
int read_env_ms(const char* name, int def);

template<typename>
constexpr bool is_dependent_false_v = false;

template<typename T>
inline constexpr bool is_supported_type_v = (
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint8_t>  ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint16_t> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint32_t> ||
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint64_t>
);

template<typename T>
constexpr T get_std_num_limit()
{
    if constexpr (is_supported_type_v<T>) {
        return std::numeric_limits<T>::max();
    } else {
        return std::numeric_limits<T>::min();
        static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
    }
}

template<typename T>
constexpr bool is_std_num_limit(T value)
{
    return (value == get_std_num_limit<T>());
}

template<typename T, typename U,  typename V = T>
constexpr T translate_umax_or_assign_value(U source_value, V target_value)
{
    T result{};
    if constexpr (is_supported_type_v<T> && is_supported_type_v<U>) {
        // If the source value is uint<U>::max(), then return is uint<T>::max()
        if (is_std_num_limit(source_value)) {
            result = get_std_num_limit<T>();
        } else {
            result = static_cast<T>(target_value);
        }

        return result;
    } else {
        static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
    }

    return result;
}

template<typename A, typename T>
void fill_2d_array(A& arr, T value) {
    for (auto& row : arr) {
        std::fill(std::begin(row), std::end(row), value);
    }
}

/**
 *  @brief Get the product serial number given the processor handle.
 *
 *  @param[in] processor_handle a pointer to amdsmi_processor_handle
 *  which the corresponding processor_handle will be stored
 *
 *  @retval ::The serial number
 *          ::0 if it cannot be determined
 */
uint64_t get_product_serial_number(amdsmi_processor_handle processor_handle);

/**
 *  @brief Tokenize bdfid into components.
 *
 *  @param[in] bdfid a uint64_t containing the bdfid
 *
 *  @retval ::Tuple of domain, bus, device, function
 */
std::tuple<uint64_t,uint64_t,uint64_t,uint64_t> parse_bdfid(uint64_t bdfid);

amdsmi_status_t smi_amdgpu_get_device_index(amdsmi_processor_handle processor_handle,
                                            uint32_t* device_index);
amdsmi_status_t smi_amdgpu_get_device_count(uint32_t *total_num_devices);
amdsmi_status_t smi_amdgpu_get_processor_handle_by_index(uint32_t device_index,
                                        amdsmi_processor_handle *processor_handle);

namespace amd::smi {

template<typename DelimiterType, typename CharType = char,
         typename TraitsType = std::char_traits<CharType>>
class ostream_joiner {
 public:
  using Char_t = CharType;
  using Traits_t = TraitsType;
  using Ostream_t = std::basic_ostream<Char_t, Traits_t>;
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = void;
  using pointer = void;
  using reference = void;

  ostream_joiner(Ostream_t* outstream,
                const DelimiterType& delimiter) noexcept
      (std::is_nothrow_copy_constructible_v<DelimiterType>)
        : m_outstream(outstream), m_delimiter(delimiter) {}

  ostream_joiner(Ostream_t* outstream, DelimiterType&& delimiter) noexcept
      (std::is_nothrow_move_constructible_v<DelimiterType>)
      : m_outstream(outstream), m_delimiter(std::move(delimiter)) {}

  template<typename ValueType> ostream_joiner& operator=(const ValueType& value) {
    if (!m_is_first) {
      *m_outstream << m_delimiter;
    }
    this->m_is_first = false;
    this->m_value_count++;
    if ((m_value_count % kMAX_VALUES_PER_LINE) == 0) {
      *m_outstream << "\n" << value;
      this->m_value_count = 0;
    } else {
      *m_outstream << value;
    }
    return *this;
  }

  ostream_joiner& operator*() noexcept { return *this; }
  ostream_joiner& operator++() noexcept { return *this; }
  ostream_joiner& operator++(int) noexcept { return *this; }

 private:
  Ostream_t* m_outstream;
  DelimiterType m_delimiter;
  bool m_is_first = true;
  uint32_t m_value_count = 0;
  const uint32_t kMAX_VALUES_PER_LINE = 9;
};

template<typename CharType, typename TraitsType, typename DelimiterType>
inline ostream_joiner<std::decay_t<DelimiterType>, CharType, TraitsType>
make_ostream_joiner(std::basic_ostream<CharType, TraitsType>* outstream,
                    DelimiterType&& delimiter) {
  return { outstream, std::forward<DelimiterType>(delimiter) };
}

bool is_vm_guest();
bool is_sudo_user();

} // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_
