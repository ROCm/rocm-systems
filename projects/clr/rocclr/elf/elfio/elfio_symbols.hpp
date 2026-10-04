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
    if (symbol_section == nullptr) {
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
    if (entry_size < minimum_size || section_size > std::numeric_limits<size_t>::max()) {
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

    Elf_Xword idx = 0;
    bool match = false;
    Elf64_Addr v = 0;

    if (elf_file.get_class() == ELFCLASS32) {
      match = generic_search_symbols<Elf32_Sym>(
          [&convertor, &value](const Elf32_Sym* sym) { return convertor(sym->st_value) == value; },
          idx);
    } else {
      match = generic_search_symbols<Elf64_Sym>(
          [&convertor, &value](const Elf64_Sym* sym) { return convertor(sym->st_value) == value; },
          idx);
    }

    if (match) {
      return get_symbol(idx, name, v, size, bind, type, section_index, other);
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
    const section* target_section = elf_file.sections[section_index];
    return target_section != nullptr && *section_name == target_section->get_name();
  }

  //------------------------------------------------------------------------------
  bool lookup_symbol(const std::string& name, const std::string* section_name, Elf64_Addr& value,
                     Elf_Xword& size, unsigned char& bind, unsigned char& type,
                     Elf_Half& section_index, unsigned char& other) const {
    if (get_hash_table_index() != 0 &&
        hash_lookup(name, section_name, value, size, bind, type, section_index, other)) {
      return true;
    }
    const Elf_Xword symbol_count = get_symbols_num();
    for (Elf_Xword i = 0; i < symbol_count; ++i) {
      std::string symbol_name;
      if (get_symbol(i, symbol_name, value, size, bind, type, section_index, other) &&
          matches_symbol(name, section_name, symbol_name, section_index)) {
        return true;
      }
    }
    return false;
  }

  //------------------------------------------------------------------------------
  bool read_hash_word(Elf_Xword index, Elf_Word& value) const {
    if (hash_section == nullptr) {
      return false;
    }
    const char* data = hash_section->get_data();
    if (data == nullptr || index >= hash_section->get_size() / sizeof(Elf_Word)) {
      return false;
    }

    Elf_Word raw_value;
    std::memcpy(&raw_value, data + index * sizeof(Elf_Word), sizeof(raw_value));
    value = elf_file.get_convertor()(raw_value);
    return true;
  }

  //------------------------------------------------------------------------------
  bool hash_lookup(const std::string& name, const std::string* section_name, Elf64_Addr& value,
                   Elf_Xword& size, unsigned char& bind, unsigned char& type,
                   Elf_Half& section_index, unsigned char& other) const {
    Elf_Word nbucket;
    Elf_Word nchain;
    if (!read_hash_word(0, nbucket) || !read_hash_word(1, nchain) || nbucket == 0) {
      return false;
    }

    const Elf_Xword word_count = hash_section->get_size() / sizeof(Elf_Word);
    if (word_count < 2 || nbucket > word_count - 2 || nchain > word_count - 2 - nbucket ||
        nchain > get_symbols_num()) {
      return false;
    }

    const Elf_Word val = elf_hash(reinterpret_cast<const unsigned char*>(name.c_str()));
    Elf_Word y;
    if (!read_hash_word(2 + static_cast<Elf_Xword>(val % nbucket), y)) {
      return false;
    }

    for (Elf_Word steps = 0; y != STN_UNDEF && y < nchain && steps < nchain; ++steps) {
      std::string symbol_name;
      if (!get_symbol(y, symbol_name, value, size, bind, type, section_index, other)) {
        return false;
      }
      if (matches_symbol(name, section_name, symbol_name, section_index)) {
        return true;
      }
      if (!read_hash_word(2 + static_cast<Elf_Xword>(nbucket) + y, y)) {
        return false;
      }
    }

    return false;
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
  template <class T>
  bool generic_search_symbols(std::function<bool(const T*)> match, Elf_Xword& idx) const {
    for (Elf_Xword i = 0; i < get_symbols_num(); i++) {
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
    if (string_data == nullptr || name_offset >= string_size ||
        string_size > std::numeric_limits<size_t>::max()) {
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
