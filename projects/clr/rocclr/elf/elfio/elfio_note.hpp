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

#ifndef ELFIO_NOTE_HPP
#define ELFIO_NOTE_HPP

#include <cstring>

namespace amd {
namespace ELFIO {

//------------------------------------------------------------------------------
// There are discrepancies in documentations. SCO documentation
// (http://www.sco.com/developers/gabi/latest/ch5.pheader.html#note_section)
// requires 8 byte entries alignment for 64-bit ELF file,
// but Oracle's definition uses the same structure
// for 32-bit and 64-bit formats.
// (https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-18048.html)
//
// It looks like EM_X86_64 Linux implementation is similar to Oracle's
// definition. Therefore, the same alignment works for both formats
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
template <class S> class note_section_accessor_template {
 public:
  //------------------------------------------------------------------------------
  note_section_accessor_template(const elfio& elf_file_, S* section_)
      : elf_file(elf_file_), note_section(section_) {
    process_section();
  }

  //------------------------------------------------------------------------------
  Elf_Word get_notes_num() const { return (Elf_Word)note_start_positions.size(); }

  //------------------------------------------------------------------------------
  bool get_note(Elf_Word index, Elf_Word& type, std::string& name, void*& desc,
                Elf_Word& descSize) const {
    if (index >= note_start_positions.size()) {
      return false;
    }

    const char* data = note_section->get_data();
    const Elf_Xword data_size = note_section->get_size();
    const Elf_Xword position = note_start_positions[index];
    const Elf_Xword align = sizeof(Elf_Word);
    const Elf_Xword header_size = 3 * align;
    if (data == nullptr || position > data_size || header_size > data_size - position) {
      return false;
    }

    const endianess_convertor& convertor = elf_file.get_convertor();
    const char* pData = data + position;
    Elf_Word raw_namesz;
    Elf_Word raw_descsz;
    Elf_Word raw_type;
    std::memcpy(&raw_namesz, pData, sizeof(raw_namesz));
    std::memcpy(&raw_descsz, pData + align, sizeof(raw_descsz));
    std::memcpy(&raw_type, pData + 2 * align, sizeof(raw_type));
    const Elf_Word namesz = convertor(raw_namesz);
    const Elf_Word parsed_descsz = convertor(raw_descsz);
    const Elf_Word parsed_type = convertor(raw_type);
    const Elf_Xword padded_name = (static_cast<Elf_Xword>(namesz) + align - 1) / align * align;
    const Elf_Xword padded_desc =
        (static_cast<Elf_Xword>(parsed_descsz) + align - 1) / align * align;
    const Elf_Xword remaining = data_size - position - header_size;
    if (padded_name > remaining || padded_desc > remaining - padded_name) {
      return false;
    }

    type = parsed_type;
    descSize = parsed_descsz;
    if (namesz == 0) {
      name.clear();
    } else {
      name.assign(pData + header_size, namesz - 1);
    }
    if (0 == parsed_descsz) {
      desc = 0;
    } else {
      desc = const_cast<char*>(pData + header_size + padded_name);
    }

    return true;
  }

  //------------------------------------------------------------------------------
  void add_note(Elf_Word type, const std::string& name, const void* desc, Elf_Word descSize) {
    const endianess_convertor& convertor = elf_file.get_convertor();

    int align = sizeof(Elf_Word);
    Elf_Word nameLen = (Elf_Word)name.size() + 1;
    Elf_Word nameLenConv = convertor(nameLen);
    std::string buffer(reinterpret_cast<char*>(&nameLenConv), align);
    Elf_Word descSizeConv = convertor(descSize);
    buffer.append(reinterpret_cast<char*>(&descSizeConv), align);
    type = convertor(type);
    buffer.append(reinterpret_cast<char*>(&type), align);
    buffer.append(name);
    buffer.append(1, '\x00');
    const char pad[] = {'\0', '\0', '\0', '\0'};
    if (nameLen % align != 0) {
      buffer.append(pad, align - nameLen % align);
    }
    if (desc != 0 && descSize != 0) {
      buffer.append(reinterpret_cast<const char*>(desc), descSize);
      if (descSize % align != 0) {
        buffer.append(pad, align - descSize % align);
      }
    }

    note_start_positions.push_back(note_section->get_size());
    note_section->append_data(buffer);
  }

 private:
  //------------------------------------------------------------------------------
  void process_section() {
    const endianess_convertor& convertor = elf_file.get_convertor();
    const char* data = note_section->get_data();
    Elf_Xword size = note_section->get_size();
    Elf_Xword current = 0;

    note_start_positions.clear();

    // Is it empty?
    if (0 == data || 0 == size) {
      return;
    }

    const Elf_Xword align = sizeof(Elf_Word);
    const Elf_Xword header_size = 3 * align;
    while (current <= size && header_size <= size - current) {
      Elf_Word raw_namesz;
      Elf_Word raw_descsz;
      std::memcpy(&raw_namesz, data + current, sizeof(raw_namesz));
      std::memcpy(&raw_descsz, data + current + sizeof(raw_namesz), sizeof(raw_descsz));
      const Elf_Word namesz = convertor(raw_namesz);
      const Elf_Word descsz = convertor(raw_descsz);
      const Elf_Xword padded_name = (static_cast<Elf_Xword>(namesz) + align - 1) / align * align;
      const Elf_Xword padded_desc = (static_cast<Elf_Xword>(descsz) + align - 1) / align * align;
      const Elf_Xword remaining = size - current - header_size;
      if (padded_name > remaining || padded_desc > remaining - padded_name) {
        break;
      }

      note_start_positions.push_back(current);
      current += header_size + padded_name + padded_desc;
    }
  }

  //------------------------------------------------------------------------------
 private:
  const elfio& elf_file;
  S* note_section;
  std::vector<Elf_Xword> note_start_positions;
};

using note_section_accessor = note_section_accessor_template<section>;
using const_note_section_accessor = note_section_accessor_template<const section>;

}  // namespace ELFIO
}  // namespace amd

#endif  // ELFIO_NOTE_HPP
