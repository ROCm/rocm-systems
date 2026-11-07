/*
Copyright (C) 2001-2015 by Serge Lamikhov-Center
Modifications Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ELFIO_SYMBOLS_HPP
#define ELFIO_SYMBOLS_HPP

#include <cstring>
#include <limits>

namespace amd {
namespace ELFIO {

//------------------------------------------------------------------------------
template <class S> class symbol_section_accessor_template {
 public:
  //------------------------------------------------------------------------------
  symbol_section_accessor_template(const elfio& elf_file_, S* symbol_section_)
      : elf_file(elf_file_), symbol_section(symbol_section_) {
    find_hash_section();
  }

  //------------------------------------------------------------------------------
  Elf_Xword get_symbols_num() const {
    // A section whose data was not loaded (e.g. sh_size beyond the file) has no
    // records, whatever its header claims.
    if (symbol_section == nullptr || symbol_section->get_data() == nullptr) {
      return 0;
    }

    Elf_Xword minimum_size;
    if (elf_file.get_class() == ELFCLASS32) {
      minimum_size = sizeof(Elf32_Sym);
    } else if (elf_file.get_class() == ELFCLASS64) {
      minimum_size = sizeof(Elf64_Sym);
    } else {
      return 0;
    }

    const Elf_Xword entry_size = symbol_section->get_entry_size();
    const Elf_Xword section_size = symbol_section->get_size();
    if (entry_size < minimum_size || !fits_in_size_t(section_size)) {
      return 0;
    }
    return section_size / entry_size;
  }

  //------------------------------------------------------------------------------
  bool get_symbol(Elf_Xword index, std::string& name, Elf64_Addr& value, Elf_Xword& size,
                  unsigned char& bind, unsigned char& type, Elf_Half& section_index,
                  unsigned char& other) const {
    bool ret = false;

    if (elf_file.get_class() == ELFCLASS32) {
      ret =
          generic_get_symbol<Elf32_Sym>(index, name, value, size, bind, type, section_index, other);
    } else {
      ret =
          generic_get_symbol<Elf64_Sym>(index, name, value, size, bind, type, section_index, other);
    }

    return ret;
  }

  //------------------------------------------------------------------------------
  /* Search in terms of symbol name */
  bool get_symbol(const std::string& name, Elf64_Addr& value, Elf_Xword& size, unsigned char& bind,
                  unsigned char& type, Elf_Half& section_index, unsigned char& other) const {
    return lookup_symbol(name, nullptr, value, size, bind, type, section_index, other);
  }
  //------------------------------------------------------------------------------
  /* Search in terms of symbol name and section name */
  bool get_symbol(const std::string& name, const std::string& section_name, Elf64_Addr& value,
                  Elf_Xword& size, unsigned char& bind, unsigned char& type,
                  Elf_Half& section_index, unsigned char& other) const {
    return lookup_symbol(name, &section_name, value, size, bind, type, section_index, other);
  }

  //------------------------------------------------------------------------------
  /* Search in terms of value */
  bool get_symbol(const Elf64_Addr& value, std::string& name, Elf_Xword& size, unsigned char& bind,
                  unsigned char& type, Elf_Half& section_index, unsigned char& other) const {
    const endianess_convertor& convertor = elf_file.get_convertor();

    auto search = [&](Elf_Xword& idx) {
      if (elf_file.get_class() == ELFCLASS32) {
        return generic_search_symbols<Elf32_Sym>(
            [&convertor, &value](const Elf32_Sym* sym) {
              return convertor(sym->st_value) == value;
            },
            idx);
      }
      return generic_search_symbols<Elf64_Sym>(
          [&convertor, &value](const Elf64_Sym* sym) { return convertor(sym->st_value) == value; },
          idx);
    };

    // A matching symbol whose name cannot be resolved does not end the search.
    Elf64_Addr v = 0;
    for (Elf_Xword idx = 0; search(idx); ++idx) {
      if (get_symbol(idx, name, v, size, bind, type, section_index, other)) {
        return true;
      }
    }

    return false;
  }

  //------------------------------------------------------------------------------
  Elf_Word add_symbol(Elf_Word name, Elf64_Addr value, Elf_Xword size, unsigned char info,
                      unsigned char other, Elf_Half shndx) {
    Elf_Word nRet;

    if (symbol_section->get_size() == 0) {
      if (elf_file.get_class() == ELFCLASS32) {
        nRet = generic_add_symbol<Elf32_Sym>(0, 0, 0, 0, 0, 0);
      } else {
        nRet = generic_add_symbol<Elf64_Sym>(0, 0, 0, 0, 0, 0);
      }
    }

    if (elf_file.get_class() == ELFCLASS32) {
      nRet = generic_add_symbol<Elf32_Sym>(name, value, size, info, other, shndx);
    } else {
      nRet = generic_add_symbol<Elf64_Sym>(name, value, size, info, other, shndx);
    }

    return nRet;
  }

  //------------------------------------------------------------------------------
  Elf_Word add_symbol(Elf_Word name, Elf64_Addr value, Elf_Xword size, unsigned char bind,
                      unsigned char type, unsigned char other, Elf_Half shndx) {
    return add_symbol(name, value, size, ELF_ST_INFO(bind, type), other, shndx);
  }

  //------------------------------------------------------------------------------
  Elf_Word add_symbol(string_section_accessor& pStrWriter, const char* str, Elf64_Addr value,
                      Elf_Xword size, unsigned char info, unsigned char other, Elf_Half shndx) {
    Elf_Word index = pStrWriter.add_string(str);
    return add_symbol(index, value, size, info, other, shndx);
  }

  //------------------------------------------------------------------------------
  Elf_Word add_symbol(string_section_accessor& pStrWriter, const char* str, Elf64_Addr value,
                      Elf_Xword size, unsigned char bind, unsigned char type, unsigned char other,
                      Elf_Half shndx) {
    return add_symbol(pStrWriter, str, value, size, ELF_ST_INFO(bind, type), other, shndx);
  }

  //------------------------------------------------------------------------------
 private:
  //------------------------------------------------------------------------------
  void find_hash_section() {
    hash_section = 0;
    hash_section_index = 0;
    if (symbol_section == nullptr) {
      return;
    }
    Elf_Half nSecNo = elf_file.sections.size();
    for (Elf_Half i = 0; i < nSecNo && 0 == hash_section_index; ++i) {
      const section* sec = elf_file.sections[i];
      if (sec->get_type() == SHT_HASH && sec->get_link() == symbol_section->get_index()) {
        hash_section = sec;
        hash_section_index = i;
      }
    }
  }

  //------------------------------------------------------------------------------
  enum class hash_result { found, not_found, invalid };

  //------------------------------------------------------------------------------
  // Always true where size_t is 64-bit; guards casts on 32-bit hosts.
  static bool fits_in_size_t(Elf_Xword value) {
    return sizeof(size_t) >= sizeof(Elf_Xword) ||
           value <= static_cast<Elf_Xword>((std::numeric_limits<size_t>::max)());
  }

  //------------------------------------------------------------------------------
  Elf_Word get_string_table_index() const { return symbol_section->get_link(); }

  //------------------------------------------------------------------------------
  Elf_Half get_hash_table_index() const { return hash_section_index; }

  //------------------------------------------------------------------------------
  bool matches_symbol(const std::string& name, const std::string* section_name,
                      const std::string& symbol_name, Elf_Half section_index) const {
    if (symbol_name != name) {
      return false;
    }
    if (section_name == nullptr) {
      return true;
    }
    // SHN_UNDEF and reserved indices (SHN_ABS, SHN_XINDEX, ...) never name a section.
    if (section_index == SHN_UNDEF || section_index >= SHN_LORESERVE) {
      return false;
    }
    const section* target_section = elf_file.sections[section_index];
    return target_section != nullptr && *section_name == target_section->get_name();
  }

  //------------------------------------------------------------------------------
  bool lookup_symbol(const std::string& name, const std::string* section_name, Elf64_Addr& value,
                     Elf_Xword& size, unsigned char& bind, unsigned char& type,
                     Elf_Half& section_index, unsigned char& other) const {
    // Search into locals so the caller's outputs are only written on a match.
    Elf64_Addr found_value = 0;
    Elf_Xword found_size = 0;
    unsigned char found_bind = 0;
    unsigned char found_type = 0;
    Elf_Half found_section_index = 0;
    unsigned char found_other = 0;
    bool found = false;

    hash_result result = hash_result::invalid;
    // Unnamed symbols are never reachable through the hash table.
    if (get_hash_table_index() != 0 && !name.empty()) {
      result = hash_lookup(name, section_name, found_value, found_size, found_bind, found_type,
                           found_section_index, found_other);
      found = result == hash_result::found;
    }
    // An unusable table needs a full scan. After a miss only local symbols are
    // searched, since linkers may add them to the table without hashing them.
    if (!found) {
      const bool locals_only = result == hash_result::not_found;
      const Elf_Xword symbol_count = get_symbols_num();
      for (Elf_Xword i = 0; i < symbol_count && !found; ++i) {
        if (locals_only && !is_local_symbol(i)) {
          continue;
        }
        std::string symbol_name;
        found = get_symbol(i, symbol_name, found_value, found_size, found_bind, found_type,
                           found_section_index, found_other) &&
                matches_symbol(name, section_name, symbol_name, found_section_index);
      }
    }
    if (!found) {
      return false;
    }

    value = found_value;
    size = found_size;
    bind = found_bind;
    type = found_type;
    section_index = found_section_index;
    other = found_other;
    return true;
  }

  //------------------------------------------------------------------------------
  bool read_hash_word(Elf_Xword index, Elf_Word& value) const {
    if (hash_section == nullptr) {
      return false;
    }
    const char* data = hash_section->get_data();
    const Elf_Xword data_size = hash_section->get_size();
    if (data == nullptr || !fits_in_size_t(data_size) || index >= data_size / sizeof(Elf_Word)) {
      return false;
    }

    Elf_Word raw_value;
    std::memcpy(&raw_value, data + index * sizeof(Elf_Word), sizeof(raw_value));
    value = elf_file.get_convertor()(raw_value);
    return true;
  }

  //------------------------------------------------------------------------------
  hash_result hash_lookup(const std::string& name, const std::string* section_name,
                          Elf64_Addr& value, Elf_Xword& size, unsigned char& bind,
                          unsigned char& type, Elf_Half& section_index,
                          unsigned char& other) const {
    Elf_Word nbucket;
    Elf_Word nchain;
    if (!read_hash_word(0, nbucket) || !read_hash_word(1, nchain) || nbucket == 0) {
      return hash_result::invalid;
    }

    // A SysV hash table has one chain entry per symbol, so a miss is only
    // authoritative when the table covers the whole symbol table.
    const Elf_Xword word_count = hash_section->get_size() / sizeof(Elf_Word);
    if (word_count < 2 || nbucket > word_count - 2 || nchain > word_count - 2 - nbucket ||
        nchain != get_symbols_num()) {
      return hash_result::invalid;
    }

    const Elf_Word val = elf_hash(reinterpret_cast<const unsigned char*>(name.c_str()));
    Elf_Word y;
    if (!read_hash_word(2 + static_cast<Elf_Xword>(val % nbucket), y)) {
      return hash_result::invalid;
    }

    // A well-formed chain ends at STN_UNDEF within nchain steps; anything else
    // (out-of-range index, unreadable symbol, cycle) marks the table unusable.
    for (Elf_Word steps = 0; steps < nchain; ++steps) {
      if (y == STN_UNDEF) {
        return hash_result::not_found;
      }
      if (y >= nchain) {
        return hash_result::invalid;
      }
      std::string symbol_name;
      if (!get_symbol(y, symbol_name, value, size, bind, type, section_index, other)) {
        return hash_result::invalid;
      }
      if (matches_symbol(name, section_name, symbol_name, section_index)) {
        return hash_result::found;
      }
      if (!read_hash_word(2 + static_cast<Elf_Xword>(nbucket) + y, y)) {
        return hash_result::invalid;
      }
    }

    return y == STN_UNDEF ? hash_result::not_found : hash_result::invalid;
  }

  //------------------------------------------------------------------------------
  template <class T> const T* generic_get_symbol_ptr(Elf_Xword index) const {
    if (index >= get_symbols_num() || symbol_section->get_entry_size() < sizeof(T)) {
      return nullptr;
    }
    const char* data = symbol_section->get_data();
    if (data == nullptr) {
      return nullptr;
    }
    // index < size / entry_size makes the multiplication and full record fit.
    return reinterpret_cast<const T*>(data + index * symbol_section->get_entry_size());
  }

  //------------------------------------------------------------------------------
  bool is_local_symbol(Elf_Xword index) const {
    if (elf_file.get_class() == ELFCLASS32) {
      return generic_is_local_symbol<Elf32_Sym>(index);
    }
    return generic_is_local_symbol<Elf64_Sym>(index);
  }

  //------------------------------------------------------------------------------
  template <class T> bool generic_is_local_symbol(Elf_Xword index) const {
    const T* record = generic_get_symbol_ptr<T>(index);
    if (record == nullptr) {
      return false;
    }
    T symbol;
    std::memcpy(&symbol, record, sizeof(symbol));
    return ELF_ST_BIND(symbol.st_info) == STB_LOCAL;
  }

  //------------------------------------------------------------------------------
  // Searches from idx onwards; on a match idx holds the matching index.
  template <class T>
  bool generic_search_symbols(std::function<bool(const T*)> match, Elf_Xword& idx) const {
    for (Elf_Xword i = idx; i < get_symbols_num(); i++) {
      const T* symPtr = generic_get_symbol_ptr<T>(i);

      if (symPtr == nullptr) return false;

      T symbol;
      std::memcpy(&symbol, symPtr, sizeof(symbol));
      if (match(&symbol)) {
        idx = i;
        return true;
      }
    }

    return false;
  }

  //------------------------------------------------------------------------------
  template <class T> bool generic_get_symbol(Elf_Xword index, std::string& name, Elf64_Addr& value,
                                             Elf_Xword& size, unsigned char& bind,
                                             unsigned char& type, Elf_Half& section_index,
                                             unsigned char& other) const {
    const T* data = generic_get_symbol_ptr<T>(index);
    if (data == nullptr) {
      return false;
    }

    T symbol;
    std::memcpy(&symbol, data, sizeof(symbol));
    const endianess_convertor& convertor = elf_file.get_convertor();
    const section* strings = elf_file.sections[get_string_table_index()];
    if (strings == nullptr || strings->get_type() != SHT_STRTAB) {
      return false;
    }
    const Elf_Word name_offset = convertor(symbol.st_name);
    const Elf_Xword string_size = strings->get_size();
    const char* string_data = strings->get_data();
    if (string_data == nullptr || name_offset >= string_size || !fits_in_size_t(string_size)) {
      return false;
    }
    const char* begin = string_data + name_offset;
    const char* end = static_cast<const char*>(
        std::memchr(begin, '\0', static_cast<size_t>(string_size - name_offset)));
    if (end == nullptr) {
      return false;
    }

    name.assign(begin, end);
    value = convertor(symbol.st_value);
    size = convertor(symbol.st_size);
    bind = ELF_ST_BIND(symbol.st_info);
    type = ELF_ST_TYPE(symbol.st_info);
    section_index = convertor(symbol.st_shndx);
    other = symbol.st_other;
    return true;
  }

  //------------------------------------------------------------------------------
  template <class T> Elf_Word generic_add_symbol(Elf_Word name, Elf64_Addr value, Elf_Xword size,
                                                 unsigned char info, unsigned char other,
                                                 Elf_Half shndx) {
    const endianess_convertor& convertor = elf_file.get_convertor();

    T entry;
    entry.st_name = convertor(name);
    entry.st_value = value;
    entry.st_value = convertor(entry.st_value);
    entry.st_size = size;
    entry.st_size = convertor(entry.st_size);
    entry.st_info = convertor(info);
    entry.st_other = convertor(other);
    entry.st_shndx = convertor(shndx);

    symbol_section->append_data(reinterpret_cast<char*>(&entry), sizeof(entry));

    Elf_Word nRet = symbol_section->get_size() / sizeof(entry) - 1;

    return nRet;
  }

  //------------------------------------------------------------------------------
 private:
  const elfio& elf_file;
  S* symbol_section;
  Elf_Half hash_section_index;
  const section* hash_section;
};

using symbol_section_accessor = symbol_section_accessor_template<section>;
using const_symbol_section_accessor = symbol_section_accessor_template<const section>;

}  // namespace ELFIO
}  // namespace amd

#endif  // ELFIO_SYMBOLS_HPP
