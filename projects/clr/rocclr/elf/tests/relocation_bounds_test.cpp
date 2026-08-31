#include <elfio/elfio.hpp>

#include <cstring>
#include <string>

using namespace amd::ELFIO;

bool rejects_missing_data(unsigned char file_class, Elf_Word section_type) {
  elfio file;
  file.create(file_class, ELFDATA2LSB);

  section* relocations = file.sections.add(".relocations.test");
  const Elf_Xword entry_size = file.get_default_entry_size(section_type);
  relocations->set_type(SHT_NOBITS);
  relocations->set_data(nullptr, static_cast<Elf_Word>(entry_size));
  relocations->set_type(section_type);
  relocations->set_entry_size(entry_size);

  relocation_section_accessor reader(file, relocations);
  Elf64_Addr offset = 0;
  Elf_Word symbol = 0;
  Elf_Word type = 0;
  Elf_Sxword addend = 0;
  return reader.get_entries_num() == 1 && !reader.get_entry(0, offset, symbol, type, addend);
}

bool rejects_undersized_entry(unsigned char file_class, Elf_Word section_type) {
  elfio file;
  file.create(file_class, ELFDATA2LSB);
  section* relocations = file.sections.add(".relocations.test");
  const Elf_Xword entry_size = file.get_default_entry_size(section_type) - 1;
  relocations->set_type(section_type);
  relocations->set_entry_size(entry_size);
  relocations->set_data(std::string(entry_size, '\0'));

  relocation_section_accessor reader(file, relocations);
  Elf64_Addr offset = 0;
  Elf_Word symbol = 0;
  Elf_Word type = 0;
  Elf_Sxword addend = 0;
  return reader.get_entries_num() == 1 && !reader.get_entry(0, offset, symbol, type, addend);
}

bool supports_extended_stride(unsigned char file_class, unsigned char encoding,
                              Elf_Word section_type) {
  elfio file;
  file.create(file_class, encoding);
  section* relocations = file.sections.add(".relocations.test");
  const Elf_Xword entry_size = file.get_default_entry_size(section_type) + 1;
  std::string bytes(2 * entry_size, static_cast<char>(0x5a));
  const endianess_convertor& convertor = file.get_convertor();

  if (file_class == ELFCLASS32 && section_type == SHT_REL) {
    Elf32_Rel entry{};
    entry.r_offset = convertor(static_cast<Elf32_Addr>(0x1234));
    entry.r_info = convertor(ELF32_R_INFO(7, 3));
    std::memcpy(bytes.data() + entry_size, &entry, sizeof(entry));
  } else if (file_class == ELFCLASS32) {
    Elf32_Rela entry{};
    entry.r_offset = convertor(static_cast<Elf32_Addr>(0x1234));
    entry.r_info = convertor(ELF32_R_INFO(7, 3));
    entry.r_addend = convertor(static_cast<Elf32_Sword>(-9));
    std::memcpy(bytes.data() + entry_size, &entry, sizeof(entry));
  } else if (section_type == SHT_REL) {
    Elf64_Rel entry{};
    entry.r_offset = convertor(static_cast<Elf64_Addr>(0x1234));
    entry.r_info = convertor(ELF64_R_INFO(7, 3));
    std::memcpy(bytes.data() + entry_size, &entry, sizeof(entry));
  } else {
    Elf64_Rela entry{};
    entry.r_offset = convertor(static_cast<Elf64_Addr>(0x1234));
    entry.r_info = convertor(ELF64_R_INFO(7, 3));
    entry.r_addend = convertor(static_cast<Elf64_Sxword>(-9));
    std::memcpy(bytes.data() + entry_size, &entry, sizeof(entry));
  }

  relocations->set_type(section_type);
  relocations->set_entry_size(entry_size);
  relocations->set_data(bytes);
  relocation_section_accessor reader(file, relocations);
  Elf64_Addr offset = 0;
  Elf_Word symbol = 0;
  Elf_Word type = 0;
  Elf_Sxword addend = 0;
  return reader.get_entry(1, offset, symbol, type, addend) && offset == 0x1234 && symbol == 7 &&
         type == 3 && addend == (section_type == SHT_RELA ? -9 : 0);
}

bool rejects_invalid_symbol_link() {
  elfio file;
  file.create(ELFCLASS64, ELFDATA2LSB);
  section* relocations = file.sections.add(".rela.test");
  relocations->set_type(SHT_RELA);
  relocations->set_entry_size(file.get_default_entry_size(SHT_RELA));
  relocation_section_accessor accessor(file, relocations);
  accessor.add_entry(1, 2, 3, 4);
  relocations->set_link(0xffff);

  Elf64_Addr offset = 0;
  Elf64_Addr symbol_value = 0;
  std::string symbol_name;
  Elf_Word type = 0;
  Elf_Sxword addend = 0;
  Elf_Sxword calculated = 0;
  return !accessor.get_entry(0, offset, symbol_value, symbol_name, type, addend, calculated);
}

int main() {
  for (unsigned char file_class : {ELFCLASS32, ELFCLASS64}) {
    if (!rejects_missing_data(file_class, SHT_REL) || !rejects_missing_data(file_class, SHT_RELA) ||
        !rejects_undersized_entry(file_class, SHT_REL) ||
        !rejects_undersized_entry(file_class, SHT_RELA)) {
      return 1;
    }
    for (unsigned char encoding : {ELFDATA2LSB, ELFDATA2MSB}) {
      if (!supports_extended_stride(file_class, encoding, SHT_REL) ||
          !supports_extended_stride(file_class, encoding, SHT_RELA)) {
        return 2;
      }
    }
  }
  return rejects_invalid_symbol_link() ? 0 : 3;
}
