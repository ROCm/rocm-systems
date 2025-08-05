/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018-2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_

#include <pthread.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <iterator>
#include <limits>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <utility>

#include "rocm_smi/rocm_smi_device.h"

#ifdef NDEBUG
#define debug_print(fmt, ...)               \
  do {                                      \
  } while (false)
#else
#define debug_print(fmt, ...)               \
  do {                                      \
    fprintf(stderr, fmt, ##__VA_ARGS__);    \
  } while (false)
#endif

namespace amd {
namespace smi {

pthread_mutex_t *GetMutex(uint32_t dv_ind);
int SameFile(const std::string fileA, const std::string fileB);
bool FileExists(char const *filename);
std::vector<std::string> globFilesExist(const std::string& filePattern);
int isRegularFile(std::string fname, bool *is_reg);
int isReadOnlyForAll(const std::string& fname, bool *is_read_only);
int ReadSysfsStr(std::string path, std::string *retStr);
int WriteSysfsStr(std::string path, std::string val);
bool IsInteger(const std::string & n_str);
std::pair<bool, std::string> executeCommand(std::string command,
                                            bool stdOut = true);
rsmi_status_t storeTmpFile(uint32_t dv_ind, std::string parameterName,
                           std::string stateName, std::string storageData);
std::vector<std::string> getListOfAppTmpFiles();
bool containsString(std::string originalString, std::string substring,
                    bool displayComparisons = false);
std::tuple<bool, std::string> readTmpFile(
                                          uint32_t dv_ind,
                                          std::string stateName,
                                          std::string parameterName);
void displayAppTmpFilesContent(void);
std::string debugVectorContent(std::vector<std::string> v);
std::string displayAllDevicePaths(std::vector<std::shared_ptr<Device>> v);
rsmi_status_t handleException();
rsmi_status_t
GetDevValueVec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<std::string> *val_vec);
rsmi_status_t
GetDevBinaryBlob(amd::smi::DevInfoTypes type,
           uint32_t dv_ind, std::size_t b_size, void* p_binary_data);
rsmi_status_t ErrnoToRsmiStatus(int err);
std::string getRSMIStatusString(rsmi_status_t ret, bool fullStatus = true);
std::tuple<bool, std::string, std::string, std::string, std::string,
           std::string, std::string, std::string, std::string,
           std::string, std::string, std::string, std::string, std::string>
           getSystemDetails(void);
void logSystemDetails(void);
rsmi_status_t getBDFString(uint64_t bdf_id, std::string& bfd_str);
void logHexDump(const char *desc, const void *addr, const size_t len,
             size_t perLine);
bool isSystemBigEndian();
std::string getBuildType();
std::string getMyLibPath();
std::string getFileCreationDate(std::string path);
int subDirectoryCountInPath(const std::string path);
std::queue<std::string> getAllDeviceGfxVers();
std::string monitor_type_string(amd::smi::MonitorTypes type);
std::string power_type_string(RSMI_POWER_TYPE type);
std::string splitString(std::string str, char delim);
std::string print_rsmi_od_volt_freq_data_t(rsmi_od_volt_freq_data_t *odv);
std::string print_rsmi_od_volt_freq_regions(uint32_t num_regions,
                                            rsmi_freq_volt_region_t *regions);
bool is_sudo_user();
rsmi_status_t rsmi_get_gfx_target_version(uint32_t dv_ind,
  std::string *gfx_version);

std::string leftTrim(const std::string &s);
std::string rightTrim(const std::string &s);
std::string trim(const std::string &s);
std::string trimAllWhiteSpace(const std::string &s);
std::string removeWhitespace(const std::string &s);
std::string removeNewLines(const std::string &s);

std::string removeString(const std::string origStr,
                        const std::string &removeMe);
void system_wait(int milli_seconds);
int countDigit(uint64_t n);
std::string find_file_in_folder(const std::string& folder,
               const std::string& regex);
uint64_t get_multiplier_from_char(char units_char);
template <typename T>
  std::string print_int_as_hex(T i, bool showHexNotation = true,
  int overloadBitSize = 0) {
  std::stringstream ss;
  if (showHexNotation) {
    if (overloadBitSize == 0) {
      ss << "0x" << std::hex << std::setw(sizeof(T) * 2) << std::setfill('0');
    } else {
      // 8 bits per 1 byte
      int byteSize = (overloadBitSize / 8) * 2;
      ss << "0x" << std::hex << std::setw(byteSize) << std::setfill('0');
    }
  } else {
    if (overloadBitSize == 0) {
      ss << std::hex << std::setw(sizeof(T) * 2) << std::setfill('0');
    } else {
      int byteSize = (overloadBitSize / 8) * 2;
      ss << std::hex << std::setw(byteSize) << std::setfill('0');
    }
  }

  if (std::is_same<std::uint8_t, T>::value) {
    ss << static_cast<unsigned int>(i|0);
  } else if (std::is_same<std::int8_t, T>::value) {
    ss << static_cast<int>(static_cast<uint8_t>(i|0));
  } else if (std::is_signed<T>::value) {
    ss << static_cast<long long int>(i | 0);
  } else {
    ss << static_cast<unsigned long long int>(i | 0);
  }
  ss << std::dec;
  return ss.str();
}

template <typename T>
std::string print_unsigned_int(T i) {
  std::stringstream ss;
  ss << static_cast<unsigned long long int>(i | 0);
  return ss.str();
}

template <typename T>
std::string print_unsigned_hex_and_int(T i, std::string heading="") {
  std::stringstream ss;
  if (heading.empty() == false) {
    ss << "\n" << heading << " = ";
  }
  ss << "Hex (MSB): " << print_int_as_hex(i) << ", "
     << "Unsigned int: " << print_unsigned_int(i) << ", "
     << "Byte Size: " << sizeof(T) << ", "
     << "Bits: " << sizeof(T) * 8;  // 8 bits per 1 byte
  return ss.str();
}

struct pthread_wrap {
 public:
        explicit pthread_wrap(pthread_mutex_t &p_mut) : mutex_(p_mut) {}

        void Acquire() {pthread_mutex_lock(&mutex_);}
        int AcquireNB() {return pthread_mutex_trylock(&mutex_);}
        void Release() {pthread_mutex_unlock(&mutex_);}
 private:
        pthread_mutex_t& mutex_;
};
struct ScopedPthread {
     explicit ScopedPthread(pthread_wrap& mutex, bool blocking = true) : //NOLINT
                               pthrd_ref_(mutex), mutex_not_acquired_(false) {
       if (blocking) {
         pthrd_ref_.Acquire();
       } else {
         int ret = pthrd_ref_.AcquireNB();
         if (ret == EBUSY) {
           mutex_not_acquired_ = true;
         }
       }
     }

     ~ScopedPthread() {
       pthrd_ref_.Release();
     }

     bool mutex_not_acquired() {return mutex_not_acquired_;}

 private:
     ScopedPthread(const ScopedPthread&);
     pthread_wrap& pthrd_ref_;
     bool mutex_not_acquired_;  // Use for AcquireNB (not for Aquire())
};


#define PASTE2(x, y) x##y
#define PASTE(x, y) PASTE2(x, y)

#define __forceinline __inline__ __attribute__((always_inline))

template <typename lambda>
class ScopeGuard {
 public:
  explicit __forceinline ScopeGuard(const lambda& release)
      : release_(release), dismiss_(false) {}

  ScopeGuard(const ScopeGuard& rhs) {*this = rhs; }

  __forceinline ~ScopeGuard() {
    if (!dismiss_) release_();
  }
  __forceinline ScopeGuard& operator=(ScopeGuard& rhs) {
    dismiss_ = rhs.dismiss_;
    release_ = rhs.release_;
    rhs.dismiss_ = true;
  }
  __forceinline void Dismiss() { dismiss_ = true; }

 private:
  lambda release_;
  bool dismiss_;
};

template <typename lambda>
static __forceinline ScopeGuard<lambda> MakeScopeGuard(lambda rel) {
  return ScopeGuard<lambda>(rel);
}

#define MAKE_SCOPE_GUARD_HELPER(lname, sname, ...) \
  auto lname = __VA_ARGS__;                        \
  amd::smi::ScopeGuard<decltype(lname)> sname(lname);
#define MAKE_SCOPE_GUARD(...)                                   \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), \
                          PASTE(scopeGuard, __COUNTER__), __VA_ARGS__)
#define MAKE_NAMED_SCOPE_GUARD(name, ...)                             \
  MAKE_SCOPE_GUARD_HELPER(PASTE(scopeGuardLambda, __COUNTER__), name, \
                          __VA_ARGS__)


// A macro to disallow the copy and move constructor and operator= functions
#define DISALLOW_COPY_AND_ASSIGN(TypeName)   \
  TypeName(const TypeName&) = delete;        \
  TypeName(TypeName&&) = delete;             \
  void operator=(const TypeName&) = delete;  \
  void operator=(TypeName&&) = delete;

template <class LockType>
class ScopedAcquire {
 public:
  /// @brief: When constructing, acquire the lock.
  /// @param: lock(Input), pointer to an existing lock.
  explicit ScopedAcquire(LockType* lock) : lock_(lock), doRelease(true) {
                                                            lock_->Acquire();}

  /// @brief: when destructing, release the lock.
  ~ScopedAcquire() {
    if (doRelease) lock_->Release();
  }

  /// @brief: Release the lock early.  Avoid using when possible.
  void Release() {
    lock_->Release();
    doRelease = false;
  }

 private:
  LockType* lock_;
  bool doRelease;
  /// @brief: Disable copiable and assignable ability.
  DISALLOW_COPY_AND_ASSIGN(ScopedAcquire)
};

// The best effort way to decide whether it is in VM guest environment:
// In VM environment, the /proc/cpuinfo set hypervisor flag by default
bool is_vm_guest();


//
enum class TagSplitterPositional_t
{
    kFIRST,
    kBETWEEN,
    kLAST,
    kNONE,
};

template <typename PrimaryKeyType = std::string, typename PrimaryDataType = std::string,
          typename SecondaryKeyType = PrimaryKeyType, typename SecondaryDataType = PrimaryDataType>
class TagTextContents_t
{
    public:
        using TextLines_t = std::vector<std::string>;
        using PrimaryList_t = std::vector<PrimaryDataType>;
        using SecondaryList_t = std::vector<SecondaryDataType>;
        using PrimaryKeyTbl_t = std::map<PrimaryKeyType, PrimaryList_t>;
        using SecondaryKeyTbl_t = std::map<SecondaryKeyType, SecondaryList_t>;
        using StructuredKeysTbl_t = std::map<PrimaryDataType, std::map<SecondaryKeyType, SecondaryDataType>>;

        //
        TagTextContents_t() = default;
        TagTextContents_t(const TagTextContents_t&) = delete;
        TagTextContents_t(TagTextContents_t&&) = delete;
        TagTextContents_t& operator=(const TagTextContents_t&) = delete;
        TagTextContents_t& operator=(TagTextContents_t&&) = delete;

        explicit TagTextContents_t(const TextLines_t& text_content)
            : m_text_content(text_content) {}

        TagTextContents_t& set_text_content(const TextLines_t& text_content)
        {
            m_text_content = text_content;
        }

        TagTextContents_t& set_title_terminator(const std::string& title_mark,
                                                TagSplitterPositional_t title_mark_position) {
            m_title_mark = title_mark;
            m_title_mark_position = title_mark_position;

            return *this;
        }

        TagTextContents_t& set_key_data_splitter(const std::string& line_splitter_mark,
                                                 TagSplitterPositional_t line_mark_position) {
            m_line_splitter_mark = line_splitter_mark;
            m_line_mark_position = line_mark_position;

            return *this;
        }

        TagTextContents_t& structure_content() {
            // Sanitizes the content.
            if (!m_text_content.empty()) {
                std::for_each(m_text_content.begin(), m_text_content.end(), trim);
                section_title_lookup();
                section_data_lookup();
            }

            return *this;
        }

        decltype(auto) get_title_size() {
            return m_primary.size();
        }

        decltype(auto) get_structured_subkeys_size(const PrimaryKeyType& prim_key) {
            return m_structured[prim_key].size();
        }

        decltype(auto) contains_title_key(const PrimaryKeyType& key) {
            return (m_primary.find(key) != m_primary.end());
        }

        decltype(auto) contains_structured_key(const PrimaryKeyType& prim_key,
                                               const SecondaryKeyType& sec_key) {
            if (auto first_key_itr = m_structured.find(prim_key);
                first_key_itr != m_structured.end()) {
                if (auto sec_key_itr = first_key_itr->second.find(sec_key);
                    sec_key_itr != first_key_itr->second.end()) {
                    return true;
                }
            }

            return false;
        }

        decltype(auto) get_structured_value_by_keys(const PrimaryKeyType& prim_key,
                                                    const SecondaryKeyType& sec_key,
                                                    bool is_value_id = true) {
            if (auto first_key_itr = m_structured.find(prim_key);
                first_key_itr != m_structured.end()) {
                if (auto sec_key_itr = first_key_itr->second.find(sec_key);
                    sec_key_itr != first_key_itr->second.end()) {
                    SecondaryDataType key_value{};
                    if (is_value_id) {
                        key_value = SecondaryDataType(sec_key_itr->first) + " ";
                    }
                    key_value += sec_key_itr->second;
                    return key_value;
                }
            }

            return SecondaryDataType{};
        }

        decltype(auto) get_structured_data_subkey_by_position(const PrimaryKeyType& prim_key,
                                                              uint32_t key_position) {
            auto key_counter = uint32_t(0);
            SecondaryKeyType data_key{};
            if (key_position < (get_structured_subkeys_size(prim_key))) {
                for (const auto& [sec_key, sec_value] : m_structured[prim_key]) {
                    if (key_counter == key_position) {
                        data_key = static_cast<SecondaryKeyType>(sec_key);
                        return data_key;
                    }
                    ++key_counter;
                }
            }

            return data_key;
        }

        decltype(auto) get_structured_data_subkey_first(const PrimaryKeyType& prim_key) {
            return (get_structured_value_by_keys(prim_key,
                                                 get_structured_data_subkey_by_position(prim_key, 0)));
        }

        decltype(auto) get_structured_data_subkey_last(const PrimaryKeyType& prim_key) {
            return (get_structured_value_by_keys(prim_key, get_structured_data_subkey_by_position(prim_key,
                                                                                                  (get_structured_subkeys_size(prim_key) - 1))));
        }

        void reset() {
            m_text_content.clear();
            m_primary.clear();
            m_structured.clear();
            m_title_mark.clear();
            m_line_splitter_mark.clear();
            m_title_mark_position = TagSplitterPositional_t::kNONE;
            m_line_mark_position = TagSplitterPositional_t::kNONE;
        }

        decltype(auto) dump_structured_content() {
            std::ostringstream ostrstream;
            ostrstream << __PRETTY_FUNCTION__ << "| ======= start =======" << "\n";
            ostrstream << "** Primary Table **" << "\n";
            for (const auto& [key, values] : m_primary) {
                ostrstream << "key: " << key << " values: " << values.size() << "\n";
                for (const auto& value : values) {
                    ostrstream << "\t value: " << value << "\n";
                }
            }

            ostrstream << "\n ** Structured Table **" << "\n";
            for (const auto& [prim_key, prim_values] : m_structured) {
                ostrstream << "key: " << prim_key << "\n";
                for (const auto& [sec_key, sec_value] : prim_values) {
                    ostrstream << "\t key: " << sec_key << " -> " << sec_value << "\n";
                }
            }
            ostrstream << "\n\n";

            return ostrstream.str();
        }


    private:
        TextLines_t m_text_content;
        PrimaryKeyTbl_t m_primary;
        StructuredKeysTbl_t m_structured;
        std::string m_title_mark;
        std::string m_line_splitter_mark;
        TagSplitterPositional_t m_title_mark_position;
        TagSplitterPositional_t m_line_mark_position;

        //
        //  Note: Organizes table with Title as a Key, and a list of values.
        //
        decltype(auto) section_title_lookup() {
            if (m_title_mark.empty() ||
                m_title_mark_position == TagSplitterPositional_t::kNONE) {
                return;
            }

            //
            //  Note:
            //      - top_title_line: Left pointer for the sliding window
            //      - bottom_title_line: Right pointer for the sliding window
            //
            auto top_title_line = uint32_t(std::numeric_limits<uint32_t>::max());
            auto bottom_title_line = uint32_t(std::numeric_limits<uint32_t>::max());
            auto line_counter = uint32_t(0);

            //
            //  Note:   This whole interval/window where the section/title starts, and where it ends.
            //
            auto update_primary_tbl = [&](const uint32_t& from_line, const uint32_t& to_line) {
                auto key = static_cast<PrimaryKeyType>(m_text_content[from_line]);
                for (auto line_num(from_line + 1); line_num < to_line; ++line_num) {
                    if ((line_num < m_text_content.size()) && !m_text_content[line_num].empty()) {
                        m_primary[key].push_back(m_text_content[line_num]);
                    }
                }
            };

            auto adjust_sliding_window = [&](const uint32_t& title_line) {
                // First time top_title_line gets adjusted.
                if (top_title_line == uint32_t(std::numeric_limits<uint32_t>::max())) {
                    top_title_line = title_line;
                    bottom_title_line = top_title_line;
                    return;
                }
                if (title_line > bottom_title_line) {
                    bottom_title_line = title_line;
                    update_primary_tbl(top_title_line, bottom_title_line);
                    top_title_line = bottom_title_line;
                }
            };

            for (const auto& line : m_text_content) {
                auto was_title_found{false};
                switch (m_title_mark_position) {
                    case TagSplitterPositional_t::kFIRST:
                        // Section/Title Mark was found at the first position
                        if (line.find_first_of(m_title_mark.c_str()) == 0) {
                            was_title_found = true;
                        }
                        break;

                    case TagSplitterPositional_t::kLAST:
                        // Section/Title Mark was found at the last position
                        if ((line.find_last_of(m_title_mark.c_str()) + 1) == line.size()) {
                            was_title_found = true;
                        }
                        break;

                    default:
                      break;
                }

                if (was_title_found) {
                    adjust_sliding_window(line_counter);
                }
                ++line_counter;
            }

            // Any remaining elements? If so, the data belongs to the last found section title
            if (line_counter > bottom_title_line) {
                update_primary_tbl(bottom_title_line, line_counter);
            }
        }

        decltype(auto) section_data_lookup() {
            if (m_line_splitter_mark.empty() ||
                m_line_mark_position == TagSplitterPositional_t::kNONE) {
                return;
            }

            //
            //  Note: Organizes table with Title as a Key, a Key/ID for values and values.
            //        It takes into consideration the initial constraints were all good and
            //        that the primary table has been populated.
            auto sec_key = std::string();
            auto sec_data = std::string();
            auto auto_key = uint32_t(0);
            for (const auto& [prim_key, prim_values] : m_primary) {
                for (const auto& value : prim_values) {
                    if (auto mark_pos = value.find_first_of(m_line_splitter_mark.c_str());
                        mark_pos != std::string::npos) {
                        sec_key = trim(value.substr(0, mark_pos + 1));
                        sec_data = trim(value.substr((mark_pos + 1), value.size()));
                    }
                    // In case there is no 'key' based on the data token marker, generate one.
                    else {
                        sec_key = std::to_string(auto_key) + m_line_splitter_mark;
                        sec_data = trim(value.substr(0, value.size()));
                        ++auto_key;
                    }
                    if (!sec_key.empty()) {
                        m_structured[prim_key].insert(std::make_pair(sec_key, sec_data));
                    }
                }
            }
        }

};

using TextFileTagContents_t = TagTextContents_t<std::string, std::string,
                                                std::string, std::string>;


//
//  Note: Output iterator that inserts a delimiter between elements.
//
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

/// Object generator for ostream_joiner.
template<typename CharType, typename TraitsType, typename DelimiterType>
inline ostream_joiner<std::decay_t<DelimiterType>, CharType, TraitsType>
  make_ostream_joiner(std::basic_ostream<CharType, TraitsType>* outstream,
    DelimiterType&& delimiter) {
    return {
      outstream,
      std::forward<DelimiterType>(delimiter)
    };
}


}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_UTILS_H_
