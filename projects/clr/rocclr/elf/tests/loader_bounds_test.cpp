#include <elfio/elfio.hpp>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

using namespace amd::ELFIO;

unsigned char host_encoding() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const unsigned char*>(&value) == 1 ? ELFDATA2LSB : ELFDATA2MSB;
}

template <class Header> class test_section : public section_impl<Header> {
 public:
  using section_impl<Header>::section_impl;

  bool load_from(std::istream& stream) { return this->load(stream, 0); }
};

template <class Header> class test_segment : public segment_impl<Header> {
 public:
  using segment_impl<Header>::segment_impl;

  bool load_from(std::istream& stream) { return this->load(stream, 0); }
};

template <class ElfHeader, class TableHeader>
bool rejects_truncated_table(bool section_table, unsigned char file_class) {
  ElfHeader header{};
  header.e_ident[EI_MAG0] = ELFMAG0;
  header.e_ident[EI_MAG1] = ELFMAG1;
  header.e_ident[EI_MAG2] = ELFMAG2;
  header.e_ident[EI_MAG3] = ELFMAG3;
  header.e_ident[EI_CLASS] = file_class;
  header.e_ident[EI_DATA] = host_encoding();
  header.e_ident[EI_VERSION] = EV_CURRENT;
  header.e_version = EV_CURRENT;
  header.e_ehsize = sizeof(ElfHeader);

  if (section_table) {
    header.e_shoff = sizeof(ElfHeader);
    header.e_shentsize = sizeof(TableHeader);
    header.e_shnum = 1;
    header.e_shstrndx = SHN_UNDEF;
  } else {
    header.e_phoff = sizeof(ElfHeader);
    header.e_phentsize = sizeof(TableHeader);
    header.e_phnum = 1;
  }

  std::string bytes(sizeof(ElfHeader) + sizeof(TableHeader) - 1, '\0');
  std::memcpy(bytes.data(), &header, sizeof(header));
  std::istringstream stream(bytes, std::ios::in | std::ios::binary);
  elfio reader;
  return !reader.load(stream) && reader.sections.size() == 0 && reader.segments.size() == 0;
}

template <class Header> bool section_loader_rejects_truncation() {
  endianess_convertor convertor;
  convertor.setup(host_encoding());
  test_section<Header> reader(&convertor);

  std::string short_header(sizeof(Header) - 1, '\0');
  std::istringstream short_header_stream(short_header, std::ios::in | std::ios::binary);
  if (reader.load_from(short_header_stream)) {
    return false;
  }

  Header header{};
  header.sh_type = SHT_PROGBITS;
  header.sh_offset = sizeof(Header);
  header.sh_size = 4;
  std::string short_payload(sizeof(Header) + 3, '\0');
  std::memcpy(short_payload.data(), &header, sizeof(header));
  std::istringstream short_payload_stream(short_payload, std::ios::in | std::ios::binary);
  return !reader.load_from(short_payload_stream) && reader.get_data() == nullptr;
}

template <class Header> bool segment_loader_rejects_truncation() {
  endianess_convertor convertor;
  convertor.setup(host_encoding());
  test_segment<Header> reader(&convertor);

  std::string short_header(sizeof(Header) - 1, '\0');
  std::istringstream short_header_stream(short_header, std::ios::in | std::ios::binary);
  if (reader.load_from(short_header_stream)) {
    return false;
  }

  Header header{};
  header.p_type = PT_LOAD;
  header.p_offset = sizeof(Header);
  header.p_filesz = 4;
  std::string short_payload(sizeof(Header) + 3, '\0');
  std::memcpy(short_payload.data(), &header, sizeof(header));
  std::istringstream short_payload_stream(short_payload, std::ios::in | std::ios::binary);
  return !reader.load_from(short_payload_stream) && reader.get_data() == nullptr;
}

int main() {
  if (!rejects_truncated_table<Elf32_Ehdr, Elf32_Shdr>(true, ELFCLASS32) ||
      !rejects_truncated_table<Elf64_Ehdr, Elf64_Shdr>(true, ELFCLASS64) ||
      !rejects_truncated_table<Elf32_Ehdr, Elf32_Phdr>(false, ELFCLASS32) ||
      !rejects_truncated_table<Elf64_Ehdr, Elf64_Phdr>(false, ELFCLASS64)) {
    return 1;
  }
  if (!section_loader_rejects_truncation<Elf32_Shdr>() ||
      !section_loader_rejects_truncation<Elf64_Shdr>()) {
    return 2;
  }
  if (!segment_loader_rejects_truncation<Elf32_Phdr>() ||
      !segment_loader_rejects_truncation<Elf64_Phdr>()) {
    return 3;
  }
  return 0;
}
