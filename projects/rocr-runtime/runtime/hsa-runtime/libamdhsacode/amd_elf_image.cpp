////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_elf_image.hpp"
#include "amd_hsa_code_util.hpp"
#include <gelf.h>
#include <errno.h>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <memory>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#ifdef _WIN32
#include <Windows.h>
#define alignof __alignof
#endif // _WIN32
#include <libelf.h>

#ifndef _WIN32
#define _open open
#define _close close
#define _tempnam tempnam
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(USE_MEMFILE)

#include "memfile.h"
#define OpenTemp(f)           mem_open(NULL, 0, 0)
#define CloseTemp(f)          mem_close(f)
#define _read(f, b, l)        mem_read((f), (b), (l))
#define _write(f, b, l)       mem_write((f), (b), (l))
#define _lseek(f, l, w)       mem_lseek((f), (l), (w))
#define _ftruncate(f, l)      mem_ftruncate((f), (size_t)(l))
#define sendfile(o, i, p, s)  mem_sendfile((o), (i), (p), (s))

#else // USE_MEMFILE

#define OpenTemp(f) amd::hsa::OpenTempFile(f);
#define CloseTemp(f) amd::hsa::CloseTempFile(f);

#ifndef _WIN32
#define _read read
#define _write write
#define _lseek lseek
#define _ftruncate ftruncate
#include <sys/sendfile.h>
#else
#define _ftruncate _chsize
#endif // !_WIN32

#endif // !USE_MEMFILE

#if !defined(BSD_LIBELF)
  #define elf_setshstrndx elfx_update_shstrndx
#endif

#define NOTE_RECORD_ALIGNMENT 4

using rocr::amd::hsa::alignUp;

namespace rocr {
namespace amd {
namespace elf {

    class FileImage {
    public:
      FileImage();
      ~FileImage();
      bool create();
      bool readFrom(const std::string& filename);
      bool copyFrom(const void* data, size_t size);
      bool writeTo(const std::string& filename);
      bool copyTo(void** buffer, size_t* size = 0);
      bool copyTo(void* buffer, size_t size);
      size_t getSize();

      std::string output() { return out.str(); }

      int fd() { return d; }

    private:
      int d;
      std::ostringstream out;

      bool error(const char* msg);
      bool perror(const char *msg);
      std::string werror();
    };

    FileImage::FileImage()
      : d(-1)
    {
    }

    FileImage::~FileImage()
    {
      if (d != -1) { CloseTemp(d); }
    }

    bool FileImage::error(const char* msg)
    {
      out << "Error: " << msg << std::endl;
      return false;
    }

    bool FileImage::perror(const char* msg)
    {
      out << "Error: " << msg << ": " << strerror(errno) << std::endl;
      return false;
    }

#ifdef _WIN32
    std::string FileImage::werror()
    {
      LPVOID lpMsgBuf;
      DWORD dw = GetLastError();

      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);
      std::string result((LPTSTR)lpMsgBuf);
      LocalFree(lpMsgBuf);
      return result;
    }
#endif // _WIN32

    bool FileImage::create()
    {
      d = OpenTemp("amdelf");
      if (d == -1) { return error("Failed to open temporary file for elf image"); }
      return true;
    }

    bool FileImage::readFrom(const std::string& filename)
    {
#ifdef _WIN32
      std::unique_ptr<char> buffer(new char[32 * 1024 * 1024]);
      HANDLE in = CreateFile(filename.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if (in == INVALID_HANDLE_VALUE) { out << "Failed to open " << filename << ": " << werror() << std::endl; return false; }
      DWORD read;
      unsigned write;
      int written;
      do {
        if (!ReadFile(in, buffer.get(), sizeof(buffer), &read, NULL)) {
          out << "Failed to read " << filename << ": " << werror() << std::endl;
          CloseHandle(in);
          return false;
        }
        if (read > 0) {
          write = read;
          do {
            written = _write(d, buffer.get(), write);
            if (written < 0) {
              out << "Failed to write image file: " << werror() << std::endl;
              CloseHandle(in);
            }
            write -= written;
          } while (write > 0);
        }
      } while (read > 0);
      if (_lseek(d, 0L, SEEK_SET) < 0) { return perror("lseek(0) failed"); }
      CloseHandle(in);
      return true;
#else // _WIN32
      int in = _open(filename.c_str(), O_RDONLY);
      if (in < 0) { return perror("open failed"); }
      if (_lseek(in, 0L, SEEK_END) < 0) { 
        _close(in);
        return perror("lseek failed"); 
      }
      off_t size;
      if ((size = _lseek(in, 0L, SEEK_CUR)) < 0) { 
        _close(in);
        return perror("lseek(2) failed"); 
      }
      if (_lseek(in, 0L, SEEK_SET) < 0) { 
        _close(in);
        return perror("lseek(3) failed"); 
      }
      if (_lseek(d, 0L, SEEK_SET) < 0) { return perror("lseek(3) failed"); }
      ssize_t written;
      do {
        written = sendfile(d, in, NULL, size);
        if (written < 0) {
          _close(in);
          return perror("sendfile failed");
        }
        size -= written;
      } while (size > 0);
      _close(in);
      if (_lseek(d, 0L, SEEK_SET) < 0) { return perror("lseek(0) failed"); }
      return true;
#endif // _WIN32
    }

    bool FileImage::copyFrom(const void* data, size_t size)
    {
      assert(d != -1);
      if (_lseek(d, 0L, SEEK_SET) < 0) { return perror("lseek failed"); }
      if (_ftruncate(d, 0) < 0) { return perror("ftruncate failed"); }
      int written, offset = 0;
      while (size > 0) {
        written = _write(d, (const char*) data + offset, size);
        if (written < 0) {
          return perror("write failed");
        }
        size -= written;
        offset += written;
      }
      if (_lseek(d, 0L, SEEK_SET) < 0) { return perror("lseek failed"); }
      return true;
    }

    size_t FileImage::getSize()
    {
      assert(d != -1);
      if (_lseek(d, 0L, SEEK_END) < 0) { return perror("lseek failed"); }
      long seek = 0;
      if ((seek = _lseek(d, 0L, SEEK_CUR)) < 0) { return perror("lseek(2) failed"); }
      if (_lseek(d, 0L, SEEK_SET) < 0) { return perror("lseek(3) failed"); }
      return seek;
    }

    bool FileImage::copyTo(void** buffer, size_t* size)
    {
      size_t size1 = getSize();
      void* buffer1 = malloc(size1);
      if (_read(d, buffer1, size1) < 0) { free(buffer1); return perror("read failed"); }
      *buffer = buffer1;
      if (size) { *size = size1; }
      return true;
    }

    bool FileImage::copyTo(void* buffer, size_t size)
    {
      size_t size1 = getSize();
      if (size < size1) { return error("Buffer size is not enough"); }
      if (_read(d, buffer, size1) < 0) { return perror("read failed"); }
      return true;
    }

    bool FileImage::writeTo(const std::string& filename)
    {
      bool res = false;
      size_t size = 0;
      void *buffer = nullptr;
      if (copyTo(&buffer, &size)) {
        res = true;
        std::ofstream out(filename.c_str(), std::ios::binary);
        out.write((char*)buffer, size);
      }
      free(buffer);
      return res;
    }

    class Buffer {
    public:
      typedef unsigned char byte_type;
      typedef size_t size_type;

      Buffer();
      Buffer(const byte_type *src, size_type size, size_type align = 0);
      virtual ~Buffer();

      const byte_type* raw() const
        { return this->isConst() ? ptr_ : data_.data(); }
      size_type align() const
        { return align_; }
      size_type size() const
        { return this->isConst() ? size_ : data_.size(); }
      bool isConst() const
        { return 0 != size_; }
      bool isEmpty()
        { return size() == 0; }
      bool hasRaw(const byte_type *src) const
        { return (src >= this->raw()) && (src < this->raw() + this->size()); }
      template<typename T>
      bool has(const T *src) const
        { return this->hasRaw((const byte_type*)src); }
      bool has(size_type offset) const
        { return offset < this->size(); }

      template<typename T>
      size_type getOffset(const T *src) const
        { return this->getRawOffset((const byte_type*)src); }
      template<typename T>
      T get(size_type offset) const
        { return (T)this->getRaw(offset); }
      size_type addString(const std::string &str, size_type align = 0);
      size_type addStringLength(const std::string &str, size_type align = 0);
      size_type nextOffset(size_type align) const { return alignUp(this->size(), align); }
      template<typename T>
      size_type add(const T *src, size_type size, size_type align)
        { return this->addRaw((const byte_type*)src, size, align); }
      template<typename T>
      size_type add(const T &src, size_type align = 0)
        { return this->addRaw((const byte_type*)&src, sizeof(T), align == 0 ? alignof(T) : align); }
      size_type align(size_type align);

      template<typename T>
      size_type reserve()
      {
        Buffer::size_type offset = this->align(alignof(T));
        data_.insert(data_.end(), sizeof(T), 0x0);
        return offset;
      }

    private:
      size_type getRawOffset(const byte_type *src) const;
      const byte_type* getRaw(size_type offset) const;
      size_type addRaw(const byte_type *src, size_type size, size_type align);

      std::vector<byte_type> data_;
      const byte_type *ptr_;
      size_type size_;
      size_type align_;
    };

    Buffer::Buffer()
      : ptr_(nullptr)
      , size_(0)
      , align_(0)
    {
    }

    Buffer::Buffer(const Buffer::byte_type *src, Buffer::size_type size, Buffer::size_type align)
      : ptr_(src)
      , size_(size)
      , align_(align)
    {
    }

    Buffer::~Buffer()
    {
    }

    Buffer::size_type Buffer::getRawOffset(const Buffer::byte_type *src) const
    {
      assert(this->has(src));
      return src - this->raw();
    }

    const Buffer::byte_type* Buffer::getRaw(Buffer::size_type offset) const
    {
      assert(this->has(offset));
      return this->raw() + offset;
    }

    Buffer::size_type Buffer::addRaw(const Buffer::byte_type *src, Buffer::size_type size, Buffer::size_type align)
    {
      assert(!this->isConst());
      assert(nullptr != src);
      assert(0 != size);
      assert(0 != align);
      Buffer::size_type offset = this->align(align);
      data_.insert(data_.end(), src, src + size);
      return offset;
    }

    Buffer::size_type Buffer::addString(const std::string &str, size_type align)
    {
      return this->add(str.c_str(), str.length() + 1, align == 0 ? alignof(char) : align);
    }

    Buffer::size_type Buffer::addStringLength(const std::string &str, size_type align)
    {
      return this->add((uint32_t)(str.length() + 1), align == 0 ? alignof(uint32_t) : align);
    }

    Buffer::size_type Buffer::align(Buffer::size_type align)
    {
      assert(!this->isConst());
      assert(0 != align);
      Buffer::size_type offset = alignUp(this->size(), align);
      align_ = (std::max)(align_, align);
      data_.insert(data_.end(), offset - this->size(), 0x0);
      return offset;
    }

    class GElfImage;
    class GElfSegment;

    class GElfSection : public virtual Section {
    public:
      GElfSection(GElfImage* elf);

      bool push(const char* name, uint32_t shtype, uint64_t shflags, uint16_t shlink, uint32_t info, uint32_t align, uint64_t entsize = 0);
      bool pull0();
      bool pull(uint16_t ndx);
      virtual bool pullData() { return true; }
      bool push();
      uint16_t getSectionIndex() const override;
      uint32_t type() const override { return hdr.sh_type; }
      std::string Name() const override;
      uint64_t offset() const override { return hdr.sh_offset; }
      uint64_t addr() const override { return hdr.sh_addr; }
      bool updateAddr(uint64_t addr) override;
      uint64_t addralign() const override { return data0.size() == 0 ? data.align() : data0.align(); }
      uint64_t flags() const override { return hdr.sh_flags; }
      uint64_t size() const override { return data0.size() == 0 ? data.size() : data0.size(); }
      uint64_t nextDataOffset(uint64_t align) const override;
      uint64_t addData(const void *src, uint64_t size, uint64_t align) override;
      bool getData(uint64_t offset, void* dest, uint64_t size) override;
      bool hasRelocationSection() const override { return reloc_sec != 0; }
      RelocationSection* relocationSection(SymbolTable* symtab = 0) override;
      Segment* segment() override { return seg; }
      RelocationSection* asRelocationSection() override { return 0; }
      bool setMemSize(uint64_t s) override { memsize_ = s; return true; }
      uint64_t memSize() const override { return memsize_ ? memsize_ : size(); }
      bool setAlign(uint64_t a) override { align_ = a; return true; }
      uint64_t memAlign() const override { return align_ ? align_ : addralign(); }

    protected:
      GElfImage* elf;
      Segment* seg;
      GElf_Shdr hdr;
      Buffer data0, data;
      uint64_t memsize_;
      uint64_t align_;
      RelocationSection *reloc_sec;

      size_t ndxscn;

      friend class GElfSymbol;
      friend class GElfSegment;
      friend class GElfImage;
    };

    class GElfSegment : public Segment {
    public:
      GElfSegment(GElfImage* elf, uint16_t index);
      GElfSegment(GElfImage* elf, uint16_t index, uint32_t type, uint32_t flags, uint64_t paddr = 0);
      bool push(uint64_t vaddr);
      bool pull();
      uint64_t type() const override { return phdr.p_type; }
      uint64_t memSize() const override { return phdr.p_memsz; }
      uint64_t align() const override { return phdr.p_align; }
      uint64_t imageSize() const override { return phdr.p_filesz; }
      uint64_t vaddr() const override { return phdr.p_vaddr; }
      uint64_t flags() const override { return phdr.p_flags; }
      uint64_t offset() const override { return phdr.p_offset; }
      const char* data() const override;
      uint16_t getSegmentIndex() override;
      bool updateAddSection(Section *section) override;

    private:
      GElfImage* elf;
      uint16_t index;
      GElf_Phdr phdr;
      std::vector<Section*> sections;
    };

    class GElfStringTable : public GElfSection, public StringTable {
    public:
      GElfStringTable(GElfImage* elf);
      bool push(const char* name, uint32_t shtype, uint64_t shflags);
      bool pullData() override;
      const char* addString(const std::string& s) override;
      size_t addString1(const std::string& s) override;
      const char* getString(size_t ndx) override;
      size_t getStringIndex(const char* name) override;

      uint16_t getSectionIndex() const override { return GElfSection::getSectionIndex(); }
      uint32_t type() const override { return GElfSection::type(); }
      std::string Name() const override { return GElfSection::Name(); }
      uint64_t addr() const override { return GElfSection::addr(); }
      uint64_t offset() const override { return GElfSection::offset(); }
      bool updateAddr(uint64_t addr) override { return GElfSection::updateAddr(addr); }
      uint64_t addralign() const override { return GElfSection::addralign(); }
      uint64_t flags() const override { return GElfSection::flags(); }
      uint64_t size() const override { return GElfSection::size(); }
      Segment* segment() override { return GElfSection::segment(); }
      uint64_t nextDataOffset(uint64_t align) const override { return GElfSection::nextDataOffset(align); }
      uint64_t addData(const void *src, uint64_t size, uint64_t align) override { return GElfSection::addData(src, size, align); }
      bool getData(uint64_t offset, void* dest, uint64_t size) override { return GElfSection::getData(offset, dest, size); }
      bool hasRelocationSection() const override { return GElfSection::hasRelocationSection(); }
      RelocationSection* relocationSection(SymbolTable* symtab) override { return GElfSection::relocationSection(); }
      RelocationSection* asRelocationSection() override { return 0; }
      uint64_t memSize() const override { return GElfSection::memSize(); }
      bool setMemSize(uint64_t s) override { return GElfSection::setMemSize(s); }
      uint64_t memAlign() const override { return GElfSection::memAlign(); }
      bool setAlign(uint64_t a) override { return GElfSection::setAlign(a); }
    };

    class GElfSymbolTable;

    class GElfSymbol : public Symbol {
    public:
      GElfSymbol(GElfSymbolTable* symtab, Buffer &data, size_t index);

      bool push(const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, uint16_t shndx, unsigned char other);

      uint32_t index() override { return eindex / sizeof(GElf_Rela); }
      uint32_t type() override { return GELF_ST_TYPE(Sym()->st_info); }
      uint32_t binding() override { return GELF_ST_BIND(Sym()->st_info); }
      uint64_t size() override { return Sym()->st_size; }
      uint64_t value() override { return Sym()->st_value; }
      unsigned char other() override { return Sym()->st_other; }
      std::string name() override;
      Section* section() override;

      void setValue(uint64_t value) override { Sym()->st_value = value; }
      void setSize(uint64_t size) override { Sym()->st_size = size; }

    private:
      GElf_Sym* Sym() { return edata.get<GElf_Sym*>(eindex); }
      GElfSymbolTable* symtab;
      Buffer &edata;
      size_t eindex;
      friend class GElfSymbolTable;
    };

    class GElfSymbolTable : public GElfSection, public SymbolTable {
    private:
      Symbol* addSymbolInternal(Section* section, const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, unsigned char other = 0);

      GElfStringTable* strtab;
      std::vector<std::unique_ptr<GElfSymbol>> symbols;
      friend class GElfSymbol;

    public:
      GElfSymbolTable(GElfImage* elf);
      bool push(const char* name, GElfStringTable* strtab);
      bool pullData() override;
      uint16_t getSectionIndex() const override { return GElfSection::getSectionIndex(); }
      uint32_t type() const override { return GElfSection::type(); }
      std::string Name() const override { return GElfSection::Name(); }
      uint64_t offset() const override { return GElfSection::offset(); }
      uint64_t addr() const override { return GElfSection::addr(); }
      bool updateAddr(uint64_t addr) override { return GElfSection::updateAddr(addr); }
      uint64_t addralign() const override { return GElfSection::addralign(); }
      uint64_t flags() const override { return GElfSection::flags(); }
      uint64_t size() const override { return GElfSection::size(); }
      Segment* segment() override { return GElfSection::segment(); }
      uint64_t nextDataOffset(uint64_t align) const override { return GElfSection::nextDataOffset(align); }
      uint64_t addData(const void *src, uint64_t size, uint64_t align) override { return GElfSection::addData(src, size, align); }
      bool getData(uint64_t offset, void* dest, uint64_t size) override { return GElfSection::getData(offset, dest, size); }
      bool hasRelocationSection() const override { return GElfSection::hasRelocationSection(); }
      RelocationSection* relocationSection(SymbolTable* symtab) override { return GElfSection::relocationSection(); }
      Symbol* addSymbol(Section* section, const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, unsigned char other = 0) override;
      size_t symbolCount() override;
      Symbol* symbol(size_t i) override;
      RelocationSection* asRelocationSection() override { return 0; }
      uint64_t memSize() const override { return GElfSection::memSize(); }
      bool setMemSize(uint64_t s) override { return GElfSection::setMemSize(s); }
      uint64_t memAlign() const override { return GElfSection::memAlign(); }
      bool setAlign(uint64_t a) override { return GElfSection::setAlign(a); }
    };

    class GElfNoteSection : public GElfSection, public NoteSection {
    public:
      GElfNoteSection(GElfImage* elf);
      bool push(const std::string& name);
      uint16_t getSectionIndex() const override { return GElfSection::getSectionIndex(); }
      uint32_t type() const override { return GElfSection::type(); }
      std::string Name() const override { return GElfSection::Name(); }
      uint64_t addr() const override { return GElfSection::addr(); }
      bool updateAddr(uint64_t addr) override { return GElfSection::updateAddr(addr); }
      uint64_t offset() const override { return GElfSection::offset(); }
      uint64_t addralign() const override { return GElfSection::addralign(); }
      uint64_t flags() const override { return GElfSection::flags(); }
      uint64_t size() const override { return GElfSection::size(); }
      Segment* segment() override { return GElfSection::segment(); }
      uint64_t nextDataOffset(uint64_t align) const override { return GElfSection::nextDataOffset(align); }
      uint64_t addData(const void *src, uint64_t size, uint64_t align) override { return GElfSection::addData(src, size, align); }
      bool getData(uint64_t offset, void* dest, uint64_t size) override { return GElfSection::getData(offset, dest, size); }
      bool hasRelocationSection() const override { return GElfSection::hasRelocationSection(); }
      RelocationSection* relocationSection(SymbolTable* symtab) override { return GElfSection::relocationSection(); }
      bool addNote(const std::string& name, uint32_t type, const void* desc, uint32_t desc_size) override;
      bool getNote(const std::string& name, uint32_t type, void** desc, uint32_t* desc_size) override;
      RelocationSection* asRelocationSection() override { return 0; }
      uint64_t memSize() const override { return GElfSection::memSize(); }
      bool setMemSize(uint64_t s) override { return GElfSection::setMemSize(s); }
      uint64_t memAlign() const override { return GElfSection::memAlign(); }
      bool setAlign(uint64_t a) override { return GElfSection::setAlign(a); }
    };

    class GElfRelocationSection;

    class GElfRelocation : public Relocation {
    private:
      GElf_Rela *Rela() { return edata.get<GElf_Rela*>(eindex); }

      GElfRelocationSection* rsection;
      Buffer &edata;
      size_t eindex;

    public:
      GElfRelocation(GElfRelocationSection* rsection_, Buffer &edata_, size_t eindex_)
        : rsection(rsection_),
          edata(edata_), eindex(eindex_)
      {
      }

      bool push(uint32_t type, Symbol* symbol, uint64_t offset, int64_t addend);

      RelocationSection* section() override;
      uint32_t type() override { return GELF_R_TYPE(Rela()->r_info); }
      uint32_t symbolIndex() override { return GELF_R_SYM(Rela()->r_info); }
      Symbol* symbol() override;
      uint64_t offset() override { return Rela()->r_offset; }
      int64_t addend() override { return Rela()->r_addend; }
    };

    class GElfRelocationSection : public GElfSection, public RelocationSection {
    private:
      Section* section;
      GElfSymbolTable* symtab;
      std::vector<std::unique_ptr<GElfRelocation>> relocations;

    public:
      GElfRelocationSection(GElfImage* elf, Section* targetSection = 0, GElfSymbolTable* symtab_ = 0);
      bool push(const std::string& name);
      bool pullData() override;
      uint16_t getSectionIndex() const override { return GElfSection::getSectionIndex(); }
      uint32_t type() const override { return GElfSection::type(); }
      std::string Name() const override { return GElfSection::Name(); }
      uint64_t addr() const override { return GElfSection::addr(); }
      uint64_t offset() const override { return GElfSection::offset(); }
      bool updateAddr(uint64_t addr) override { return GElfSection::updateAddr(addr); }
      uint64_t addralign() const override { return GElfSection::addralign(); }
      uint64_t flags() const override { return GElfSection::flags(); }
      uint64_t size() const override { return GElfSection::size(); }
      Segment* segment() override { return GElfSection::segment(); }
      uint64_t nextDataOffset(uint64_t align) const override { return GElfSection::nextDataOffset(align); }
      uint64_t addData(const void *src, uint64_t size, uint64_t align) override { return GElfSection::addData(src, size, align); }
      bool getData(uint64_t offset, void* dest, uint64_t size) override { return GElfSection::getData(offset, dest, size); }
      bool hasRelocationSection() const override { return GElfSection::hasRelocationSection(); }
      RelocationSection* relocationSection(SymbolTable* symtab) override { return GElfSection::relocationSection(); }
      RelocationSection* asRelocationSection() override { return this; }

      size_t relocationCount() const override { return relocations.size(); }
      Relocation* relocation(size_t i) override { return relocations[i].get(); }
      Relocation* addRelocation(uint32_t type, Symbol* symbol, uint64_t offset, int64_t addend) override;
      Section* targetSection() override { return section; }
      uint64_t memSize() const override { return GElfSection::memSize(); }
      bool setMemSize(uint64_t s) override { return GElfSection::setMemSize(s); }
      uint64_t memAlign() const override { return GElfSection::memAlign(); }
      bool setAlign(uint64_t a) override { return GElfSection::setAlign(a); }
      friend class GElfRelocation;
    };

    class GElfImage : public Image {
    public:
      GElfImage(int elfclass);
      ~GElfImage();
      bool initNew(uint16_t machine, uint16_t type, uint8_t os_abi = 0, uint8_t abi_version = 0, uint32_t e_flags = 0) override;
      bool loadFromFile(const std::string& filename) override;
      bool saveToFile(const std::string& filename) override;
      bool initFromBuffer(const void* buffer, size_t size) override;
      bool initAsBuffer(const void* buffer, size_t size) override;
      bool close();
      bool writeTo(const std::string& filename) override;
      bool copyToBuffer(void** buf, size_t* size = 0) override;
      bool copyToBuffer(void* buf, size_t size) override;

      const char* data() override { assert(buffer); return buffer; }
      uint64_t size() override;

      bool push();

      bool Freeze() override;
      bool Validate() override;

      uint16_t Machine() override { return ehdr.e_machine; }
      uint16_t Type() override { return ehdr.e_type; }
      uint32_t EFlags() override { return ehdr.e_flags; }
      uint32_t ABIVersion() override { return (uint32_t)(ehdr.e_ident[EI_ABIVERSION]); }
      uint32_t EClass() override { return (uint32_t)(ehdr.e_ident[EI_CLASS]); }
      uint32_t OsAbi() override { return (uint32_t)(ehdr.e_ident[EI_OSABI]); }

      GElfStringTable* shstrtab() override;
      GElfStringTable* strtab() override;
      GElfSymbolTable* getReferencedSymbolTable(uint16_t index)
      {
        return static_cast<GElfSymbolTable*>(section(index));
      }
      GElfSymbolTable* getSymtab(uint16_t index) override
      {
        if (section(index)->type() == SHT_SYMTAB)
          return static_cast<GElfSymbolTable*>(section(index));
        return nullptr;
      }
      GElfSymbolTable* getDynsym(uint16_t index) override
      {
        if (section(index)->type() == SHT_DYNSYM)
          return static_cast<GElfSymbolTable*>(section(index));
        return nullptr;
      }

      GElfSymbolTable* getSymbolTable() override;
      GElfSymbolTable* getSymbolTable(uint16_t index) override
      {
        const char *UseDynsym = getenv("LOADER_USE_DYNSYM");
        if (UseDynsym && std::strncmp(UseDynsym, "0", 1) != 0)
          return getDynsym(index);
        return getSymtab(index);
      }

      GElfStringTable* addStringTable(const std::string& name) override;
      GElfStringTable* getStringTable(uint16_t index) override;

      GElfSymbolTable* addSymbolTable(const std::string& name, StringTable* stab = 0) override;
      GElfSymbolTable* symtab() override;
      GElfSymbolTable* dynsym() override;

      GElfSegment* segment(size_t i) override { return segments[i].get(); }
      Segment* segmentByVAddr(uint64_t vaddr) override;
      size_t sectionCount() override { return sections.size(); }
      GElfSection* section(size_t i) override { return sections[i].get(); }
      Section* sectionByVAddr(uint64_t vaddr) override;
      uint16_t machine() const;
      uint16_t etype() const;
      int eclass() const { return elfclass; }
      bool elfError(const char* msg);

      GElfNoteSection* note() override;
      GElfNoteSection* addNoteSection(const std::string& name) override;

      size_t segmentCount() override { return segments.size(); }
      Segment* initSegment(uint32_t type, uint32_t flags, uint64_t paddr = 0) override;
      bool addSegments() override;

      Section* addSection(const std::string &name,
                          uint32_t type,
                          uint64_t flags = 0,
                          uint64_t entsize = 0,
                          Segment* segment = 0) override;

      RelocationSection* addRelocationSection(Section* sec, SymbolTable* symtab);
      RelocationSection* relocationSection(Section* sec, SymbolTable* symtab = 0) override;

    private:
      bool frozen;
      int elfclass;
      FileImage img;
      const char* buffer;
      size_t bufferSize;
      Elf* e;
      GElf_Ehdr ehdr;
      GElfStringTable* shstrtabSection;
      GElfStringTable* strtabSection;
      GElfSymbolTable* symtabSection;
      GElfSymbolTable* dynsymSection;
      GElfNoteSection* noteSection;
      std::vector<std::unique_ptr<GElfSegment>> segments;
      std::vector<std::unique_ptr<GElfSection>> sections;

      bool imgError();
      const char *elfError();
      bool elfBegin(Elf_Cmd cmd);
      bool elfEnd();
      bool push0();
      bool pullElf();

      friend class GElfSection;
      friend class GElfSymbolTable;
      friend class GElfNoteSection;
      friend class GElfRelocationSection;
      friend class GElfSegment;
      friend class GElfSymbol;
    };

    GElfSegment::GElfSegment(GElfImage* elf_, uint16_t index_)
      : elf(elf_),
        index(index_)
    {
      memset(&phdr, 0, sizeof(phdr));
    }

    GElfSegment::GElfSegment(GElfImage* elf_, uint16_t index_,
      uint32_t type, uint32_t flags, uint64_t paddr)
      : elf(elf_),
        index(index_)
    {
      memset(&phdr, 0, sizeof(phdr));
      phdr.p_type = type;
      phdr.p_flags = flags;
      phdr.p_paddr = paddr;
    }

    const char* GElfSegment::data() const
    {
      return (const char*) elf->data() + phdr.p_offset;
    }

    bool GElfImage::Freeze()
    {
      assert(!frozen);
      if (!push()) { return false; }
      frozen = true;
      return true;
    }

    bool GElfImage::Validate()
    {
      if (ELFMAG0 != ehdr.e_ident[EI_MAG0] ||
          ELFMAG1 != ehdr.e_ident[EI_MAG1] ||
          ELFMAG2 != ehdr.e_ident[EI_MAG2] ||
          ELFMAG3 != ehdr.e_ident[EI_MAG3]) {
        out << "Invalid ELF magic" << std::endl;
        return false;
      }
      if (EV_CURRENT != ehdr.e_version) {
        out << "Invalid ELF version" << std::endl;
        return false;
      }
      return true;
    }

    bool GElfSegment::push(uint64_t vaddr)
    {
      phdr.p_align = 0;
      phdr.p_offset = 0;
      if (!sections.empty()) {
        phdr.p_offset = sections[0]->offset();
      }
      for (Section* section : sections) {
        phdr.p_align = (std::max)(phdr.p_align, section->memAlign());
      }
      phdr.p_vaddr = alignUp(vaddr, (std::max)(phdr.p_align, (uint64_t) 1));
      phdr.p_filesz = 0;
      phdr.p_memsz = 0;
      for (Section* section : sections) {
        phdr.p_memsz = alignUp(phdr.p_memsz, (std::max)(section->memAlign(), (uint64_t) 1));
        phdr.p_filesz = alignUp(phdr.p_filesz, (std::max)(section->memAlign(), (uint64_t) 1));
        if (!section->updateAddr(phdr.p_vaddr + phdr.p_memsz)) { return false; }
        phdr.p_filesz += (section->type() == SHT_NOBITS) ? 0 : section->size();
        phdr.p_memsz += section->memSize();
      }
      if (!gelf_update_phdr(elf->e, index, &phdr)) { return elf->elfError("gelf_update_phdr failed"); }
      return true;
    }

    bool GElfSegment::pull()
    {
      if (!gelf_getphdr(elf->e, index, &phdr)) { return elf->elfError("gelf_getphdr failed"); }
      return true;
    }

    uint16_t GElfSegment::getSegmentIndex()
    {
      return index;
    }

    bool GElfSegment::updateAddSection(Section *section)
    {
      sections.push_back(section);
      return true;
    }

    GElfSection::GElfSection(GElfImage* elf_)
      : elf(elf_),
        seg(nullptr),
        hdr{},
        memsize_(0),
        align_(0),
        reloc_sec(nullptr),
        ndxscn(0)
    {
    }

    uint16_t GElfSection::getSectionIndex() const
    {
      return (uint16_t)ndxscn;
    }

    std::string GElfSection::Name() const
    {
      return std::string(elf->shstrtab()->getString(hdr.sh_name));
    }

    bool GElfSection::updateAddr(uint64_t addr)
    {
      Elf_Scn *scn = elf_getscn(elf->e, ndxscn);
      assert(scn);
      if (!gelf_getshdr(scn, &hdr)) { return elf->elfError("gelf_get_shdr failed"); }
      hdr.sh_addr = addr;
      if (!gelf_update_shdr(scn, &hdr)) { return elf->elfError("gelf_update_shdr failed"); }
      return true;
    }

    bool GElfSection::push(const char* name, uint32_t shtype, uint64_t shflags, uint16_t shlink, uint32_t info, uint32_t align, uint64_t entsize)
    {
      Elf_Scn *scn = elf_newscn(elf->e);
      if (!scn) { return false; }
      ndxscn = elf_ndxscn(scn);
      if (!gelf_getshdr(scn, &hdr)) { return elf->elfError("gelf_get_shdr failed"); }
      align = (std::max)(align, (uint32_t) 8);
      hdr.sh_name = elf->shstrtab()->addString1(name);
      hdr.sh_type = shtype;
      hdr.sh_flags = shflags;
      hdr.sh_link = shlink;
      hdr.sh_addr = 0;
      hdr.sh_info = info;
      hdr.sh_addralign = align;
      hdr.sh_entsize = entsize;
      if (!gelf_update_shdr(scn, &hdr)) { return elf->elfError("gelf_update_shdr failed"); }
      return true;
    }

    bool GElfSection::pull0()
    {
      Elf_Scn *scn = elf_getscn(elf->e, ndxscn);
      if (!scn) { return false; }
      if (!gelf_getshdr(scn, &hdr)) { return elf->elfError("gelf_get_shdr failed"); }
      return true;
    }

    bool GElfSection::pull(uint16_t ndx)
    {
      ndxscn = (size_t) ndx;
      if (!pull0()) { return false; }
      Elf_Scn *scn = elf_getscn(elf->e, ndx);
      if (!scn) { return false; }
      Elf_Data *edata0 = elf_getdata(scn, NULL);
      if (edata0) {
        data0 = Buffer((const Buffer::byte_type*)edata0->d_buf, edata0->d_size, edata0->d_align);
      }
      seg = elf->segmentByVAddr(hdr.sh_addr);
      return true;
    }

    bool GElfSection::push()
    {
      Elf_Scn *scn = elf_getscn(elf->e, ndxscn);
      assert(scn);
      Elf_Data *edata = nullptr;
      edata = elf_newdata(scn);
      if (!edata) { return elf->elfError("elf_newdata failed"); }
      if (hdr.sh_type == SHT_NOBITS) {
        edata->d_buf = 0;
        edata->d_size = memsize_;
        if (align_ != 0) {
          edata->d_align = align_;
        }
      } else {
        edata->d_buf = (void*)data.raw();
        edata->d_size = data.size();
        if (data.align() != 0) {
          edata->d_align = data.align();
        }
      }
      edata->d_align = (std::max)(edata->d_align, (uint64_t) 8);
      switch (hdr.sh_type) {
      case SHT_RELA:
        edata->d_type = ELF_T_RELA;
        break;
      case SHT_SYMTAB:
        edata->d_type = ELF_T_SYM;
        break;
      default:
        edata->d_type = ELF_T_BYTE;
        break;
      }
      edata->d_version = EV_CURRENT;
      if (!gelf_getshdr(scn, &hdr)) { return elf->elfError("gelf_get_shdr failed"); }
      hdr.sh_size = edata->d_size;
      hdr.sh_addralign = edata->d_align;
      if (!gelf_update_shdr(scn, &hdr)) { return elf->elfError("gelf_update_shdr failed"); }
      return true;
    }

    uint64_t GElfSection::nextDataOffset(uint64_t align) const
    {
      return data.nextOffset(align);
    }

    uint64_t GElfSection::addData(const void *src, uint64_t size, uint64_t align)
    {
      return data.add(src, size, align);
    }

    bool GElfSection::getData(uint64_t offset, void* dest, uint64_t size)
    {
      Elf_Data* edata = 0;
      uint64_t coffset = 0;
      uint64_t csize = 0;
      Elf_Scn *scn = elf_getscn(elf->e, ndxscn);
      assert(scn);
      if ((edata = elf_getdata(scn, edata)) != 0) {
        if (coffset <= offset && offset <= coffset + edata->d_size) {
          csize = (std::min)(size, edata->d_size - offset);
          memcpy(dest, (const char*) edata->d_buf + offset - coffset, csize);
          dest = (char*) dest + csize;
          size -= csize;
          if (!size) { return true; }
        }
      }
      return false;
    }

    RelocationSection* GElfSection::relocationSection(SymbolTable* symtab)
    {
      if (!reloc_sec) {
        reloc_sec = elf->addRelocationSection(this, symtab);
      }
      return reloc_sec;
    }

    GElfStringTable::GElfStringTable(GElfImage* elf)
      : GElfSection(elf)
    {
    }

    bool GElfStringTable::push(const char* name, uint32_t shtype, uint64_t shflags)
    {
      if (!GElfSection::push(name, shtype, shflags, SHN_UNDEF, 0, 0)) { return false; }
      return true;
    }

    bool GElfStringTable::pullData()
    {
      return true;
    }

    const char* GElfStringTable::addString(const std::string& s)
    {
      if (data0.size() == 0 && data.size() == 0) {
        data.add('\0');
      }
      return data.get<const char*>(data.addString(s));
    }

    size_t GElfStringTable::addString1(const std::string& s)
    {
      if (data0.size() == 0 && data.size() == 0) {
        data.add('\0');
      }
      return data.addString(s);
    }

    const char* GElfStringTable::getString(size_t ndx)
    {
      if (data0.has(ndx)) { return data0.get<const char*>(ndx); }
      else if (data.has(ndx)) { return data.get<const char*>(ndx); }
      return nullptr;
    }

    size_t GElfStringTable::getStringIndex(const char* s)
    {
      if (data0.has(s)) {
        return data0.getOffset(s);
      } else if (data.has(s)) {
        return data.getOffset(s);
      } else {
        assert(false);
        return 0;
      }
    }

    GElfSymbol::GElfSymbol(GElfSymbolTable* symtab_, Buffer &data_, size_t index_)
      : symtab(symtab_),
        edata(data_),
        eindex(index_)
    {
    }

    Section* GElfSymbol::section()
    {
      if (Sym()->st_shndx != SHN_UNDEF) {
        return symtab->elf->section(Sym()->st_shndx);
      }
      return 0;
    }

    bool GElfSymbol::push(const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, uint16_t shndx, unsigned char other)
    {
      Sym()->st_name = symtab->strtab->addString1(name.c_str());
      Sym()->st_value = value;
      Sym()->st_size = size;
      Sym()->st_info = GELF_ST_INFO(binding, type);
      Sym()->st_shndx = shndx;
      Sym()->st_other = other;
      return true;
    }

    std::string GElfSymbol::name()
    {
      return symtab->strtab->getString(Sym()->st_name);
    }

    GElfSymbolTable::GElfSymbolTable(GElfImage* elf)
      : GElfSection(elf),
        strtab(0)
    {
    }

    bool GElfSymbolTable::push(const char* name, GElfStringTable* strtab)
    {
      if (!strtab) { strtab = elf->strtab(); }
      this->strtab = strtab;
      if (!GElfSection::push(name, SHT_SYMTAB, 0, strtab->getSectionIndex(), 0, 0, sizeof(Elf64_Sym))) { return false;  }
      return true;
    }

    bool GElfSymbolTable::pullData()
    {
      strtab = elf->getStringTable(hdr.sh_link);
      for (size_t i = 0; i < data0.size() / sizeof(GElf_Sym); ++i) {
        symbols.push_back(std::unique_ptr<GElfSymbol>(new GElfSymbol(this, data0, i * sizeof(GElf_Sym))));
      }
      return true;
    }

    Symbol* GElfSymbolTable::addSymbolInternal(Section* section, const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, unsigned char other)
    {
      GElfSymbol *sym = new (std::nothrow) GElfSymbol(this, data, data.reserve<GElf_Sym>());
      uint16_t shndx = section ? section->getSectionIndex() : (uint16_t) SHN_UNDEF;
      if (!sym->push(name, value, size, type, binding, shndx, other)) {
        delete sym;
        return nullptr;
      }
      symbols.push_back(std::unique_ptr<GElfSymbol>(sym));
      return sym;
    }

    Symbol* GElfSymbolTable::addSymbol(Section* section, const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, unsigned char other)
    {
      if (symbols.size() == 0) {
        this->addSymbolInternal(nullptr, "", 0, 0, 0, 0, 0);
      }
      return this->addSymbolInternal(section, name, value, size, type, binding, other);
    }

    size_t GElfSymbolTable::symbolCount()
    {
      return symbols.size();
    }

    Symbol* GElfSymbolTable::symbol(size_t i)
    {
      return symbols[i].get();
    }

    GElfNoteSection::GElfNoteSection(GElfImage* elf)
      : GElfSection(elf)
    {
    }

    bool GElfNoteSection::push(const std::string& name)
    {
      return GElfSection::push(name.c_str(), SHT_NOTE, 0, 0, 0, 8);
    }

    bool GElfNoteSection::addNote(const std::string& name, uint32_t type, const void* desc, uint32_t desc_size)
    {
      data.addStringLength(name, NOTE_RECORD_ALIGNMENT);
      data.add(desc_size, NOTE_RECORD_ALIGNMENT);
      data.add(type, NOTE_RECORD_ALIGNMENT);
      data.addString(name, NOTE_RECORD_ALIGNMENT);
      data.align(NOTE_RECORD_ALIGNMENT);
      if (desc_size > 0) {
        assert(desc);
        data.add(desc, desc_size, NOTE_RECORD_ALIGNMENT);
        data.align(NOTE_RECORD_ALIGNMENT);
      }
      return true;
    }

    bool GElfNoteSection::getNote(const std::string& name, uint32_t type, void** desc, uint32_t* desc_size)
    {
      Elf_Data* data = 0;
      Elf_Scn *scn = elf_getscn(elf->e, ndxscn);
      assert(scn);
      while ((data = elf_getdata(scn, data)) != 0) {
        uint32_t note_offset = 0;
        while (note_offset < data->d_size) {
          char* notec = (char *) data->d_buf + note_offset;
          Elf64_Nhdr* note = (Elf64_Nhdr*) notec;
          if (type == note->n_type) {
            std::string note_name = GetNoteString(note->n_namesz, notec + sizeof(Elf64_Nhdr));
            if (name == note_name) {
              *desc = notec + sizeof(Elf64_Nhdr) + alignUp(note->n_namesz, 4);
              *desc_size = note->n_descsz;
              return true;
            }
          }
          note_offset += sizeof(Elf64_Nhdr) + alignUp(note->n_namesz, 4) + alignUp(note->n_descsz, 4);
        }
      }
      return false;
    }

    bool GElfRelocation::push(uint32_t type, Symbol* symbol, uint64_t offset, int64_t addend)
    {
      Rela()->r_info = GELF_R_INFO((uint64_t) symbol->index(), type);
      Rela()->r_offset = offset;
      Rela()->r_addend = addend;
      return true;
    }

    RelocationSection* GElfRelocation::section()
    {
      return rsection;
    }

    Symbol* GElfRelocation::symbol()
    {
      return rsection->symtab->symbol(symbolIndex());
    }

    GElfRelocationSection::GElfRelocationSection(GElfImage* elf, Section* section_, GElfSymbolTable* symtab_)
      : GElfSection(elf),
        section(section_),
        symtab(symtab_)
    {
    }

    bool GElfRelocationSection::push(const std::string& name)
    {
      return GElfSection::push(name.c_str(), SHT_RELA, 0, symtab->getSectionIndex(), section->getSectionIndex(), 0, sizeof(Elf64_Rela));
    }

    Relocation* GElfRelocationSection::addRelocation(uint32_t type, Symbol* symbol, uint64_t offset, int64_t addend)
    {
      GElfRelocation *rela = new (std::nothrow) GElfRelocation(this, data, data.reserve<GElf_Rela>());
      if (!rela || !rela->push(type, symbol, offset, addend)) {
        delete rela;
        return nullptr;
      }
      relocations.push_back(std::unique_ptr<GElfRelocation>(rela));
      return rela;
    }

    bool GElfRelocationSection::pullData()
    {
      section = elf->section(hdr.sh_info);
      symtab = elf->getReferencedSymbolTable(hdr.sh_link);
      Elf_Scn *lScn = elf_getscn(elf->e, ndxscn);
      assert(lScn);
      Elf_Data *lData = elf_getdata(lScn, nullptr);
      assert(lData);
      data0 = Buffer((const Buffer::byte_type*)lData->d_buf, lData->d_size, lData->d_align);
      for (size_t i = 0; i < data0.size() / sizeof(GElf_Rela); ++i) {
        relocations.push_back(std::unique_ptr<GElfRelocation>(new GElfRelocation(this, data0, i * sizeof(GElf_Rela))));
      }
      return true;
    }

    GElfImage::GElfImage(int elfclass_)
      : frozen(true),
        elfclass(elfclass_),
        buffer(0), bufferSize(0),
        e(0),
        shstrtabSection(0), strtabSection(0),
        symtabSection(0),
        dynsymSection(0),
        noteSection(0)
    {
      if (EV_NONE == elf_version(EV_CURRENT)) {
        assert(false);
      }
    }

    GElfImage::~GElfImage()
    {
      elf_end(e);
    }

    bool GElfImage::imgError()
    {
      out << img.output();
      return false;
    }

    const char *GElfImage::elfError()
    {
      return elf_errmsg(-1);
    }

    bool GElfImage::elfBegin(Elf_Cmd cmd)
    {
      if ((e = elf_begin(img.fd(), cmd, NULL
#ifdef AMD_LIBELF
                       , NULL
#endif
        )) == NULL) {
        out << "elf_begin failed: " << elfError() << std::endl;
        return false;
      }
      return true;
    }

    bool GElfImage::initNew(uint16_t machine, uint16_t type, uint8_t os_abi, uint8_t abi_version, uint32_t e_flags)
    {
      if (!img.create()) { return imgError(); }
      if (!elfBegin(ELF_C_WRITE)) { return false; }
      if (!gelf_newehdr(e, elfclass)) { return elfError("gelf_newehdr failed"); }
      if (!gelf_getehdr(e, &ehdr)) { return elfError("gelf_getehdr failed"); }
      ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
      ehdr.e_ident[EI_VERSION] = EV_CURRENT;
      ehdr.e_ident[EI_OSABI] = os_abi;
      ehdr.e_ident[EI_ABIVERSION] = abi_version;
      ehdr.e_machine = machine;
      ehdr.e_type = type;
      ehdr.e_version = EV_CURRENT;
      ehdr.e_flags = e_flags;
      if (!gelf_update_ehdr(e, &ehdr)) { return elfError("gelf_updateehdr failed"); }
      sections.push_back(std::unique_ptr<GElfSection>());
      if (!shstrtab()->push(".shstrtab", SHT_STRTAB, SHF_STRINGS)) { return elfError("Failed to create shstrtab"); }
      ehdr.e_shstrndx = shstrtab()->getSectionIndex();
      if (!gelf_update_ehdr(e, &ehdr)) { return elfError("gelf_updateehdr failed"); }
      if (!strtab()->push(".strtab", SHT_STRTAB, SHF_STRINGS)) { return elfError("Failed to create strtab"); }
      frozen = false;
      return true;
    }

    bool GElfImage::loadFromFile(const std::string& filename)
    {
      if (!img.create()) { return imgError(); }
      if (!img.readFrom(filename)) { return imgError(); }
      if (!elfBegin(ELF_C_RDWR)) { return false; }
      return pullElf();
    }

    bool GElfImage::saveToFile(const std::string& filename)
    {
      if (buffer) {
        std::ofstream out(filename.c_str(), std::ios::binary);
        if (out.fail()) { return false; }
        out.write(buffer, bufferSize);
        return !out.fail();
      } else {
        if (!push()) { return false; }
        return img.writeTo(filename);
      }
    }

    bool GElfImage::initFromBuffer(const void* buffer, size_t size)
    {
      if (size == 0) { size = ElfSize(buffer); }
      if (!img.create()) { return imgError(); }
      if (!img.copyFrom(buffer, size)) { return imgError(); }
      if (!elfBegin(ELF_C_RDWR)) { return false; }
      return pullElf();
    }

    bool GElfImage::initAsBuffer(const void* buffer, size_t size)
    {
      if (size == 0) { size = ElfSize(buffer); }
      if ((e = elf_memory(reinterpret_cast<char*>(const_cast<void*>(buffer)), size
#ifdef AMD_LIBELF
                       , NULL
#endif
        )) == NULL) {
        out << "elf_begin(buffer) failed: " << elfError() << std::endl;
        return false;
      }
      this->buffer = reinterpret_cast<const char*>(buffer);
      this->bufferSize = size;
      return pullElf();
    }

    bool GElfImage::pullElf()
    {
      if (!gelf_getehdr(e, &ehdr)) { return elfError("gelf_getehdr failed"); }
      segments.reserve(ehdr.e_phnum);
      for (size_t i = 0; i < ehdr.e_phnum; ++i) {
        GElfSegment* segment = new GElfSegment(this, i);
        segment->pull();
        segments.push_back(std::unique_ptr<GElfSegment>(segment));
      }

      shstrtabSection = new GElfStringTable(this);
      if (!shstrtabSection->pull(ehdr.e_shstrndx)) { return false; }
      Elf_Scn* scn = 0;
      for (unsigned n = 0; n < ehdr.e_shnum; ++n) {
        scn = elf_getscn(e, n);
        if (n == ehdr.e_shstrndx) {
          sections.push_back(std::unique_ptr<GElfSection>(shstrtabSection));
          continue;
        }
        GElf_Shdr shdr;
        if (!gelf_getshdr(scn, &shdr)) { return elfError("Failed to get shdr"); }
        GElfSection* section = 0;
        if (shdr.sh_type == SHT_NOTE) {
          section = new GElfNoteSection(this);
        } else if (shdr.sh_type == SHT_RELA) {
          section = new GElfRelocationSection(this);
        } else if (shdr.sh_type == SHT_STRTAB) {
          section = new GElfStringTable(this);
        } else if (shdr.sh_type == SHT_SYMTAB || shdr.sh_type == SHT_DYNSYM) {
          section = new GElfSymbolTable(this);
        } else if (shdr.sh_type == SHT_NULL) {
          section = 0;
          sections.push_back(std::unique_ptr<GElfSection>());
        } else {
          section = new GElfSection(this);
        }
        if (section) {
          sections.push_back(std::unique_ptr<GElfSection>(section));
          if (!section->pull(n)) { return false; }
        }
      }

      for (size_t n = 1; n < sections.size(); ++n) {
        GElfSection* section = sections[n].get();
        if (section->type() == SHT_STRTAB) {
          if (!section->pullData()) { return false; }
        }
      }

      for (size_t n = 1; n < sections.size(); ++n) {
        GElfSection* section = sections[n].get();
        if (section->type() == SHT_SYMTAB || section->type() == SHT_DYNSYM) {
          if (!section->pullData()) { return false; }
        }
      }

      for (size_t n = 1; n < sections.size(); ++n) {
        GElfSection* section = sections[n].get();
        if (section->type() != SHT_STRTAB && section->type() != SHT_SYMTAB && section->type() != SHT_DYNSYM) {
          if (!section->pullData()) { return false; }
        }
      }

      for (size_t i = 1; i < sections.size(); ++i) {
        if (i == ehdr.e_shstrndx) { continue; }
        std::unique_ptr<GElfSection>& section = sections[i];
        if (section->type() == SHT_STRTAB) { strtabSection = static_cast<GElfStringTable*>(section.get()); }
        if (section->type() == SHT_SYMTAB) { symtabSection = static_cast<GElfSymbolTable*>(section.get()); }
        if (section->type() == SHT_NOTE) { noteSection = static_cast<GElfNoteSection*>(section.get()); }
        if (section->type() == SHT_DYNSYM) { dynsymSection = static_cast<GElfSymbolTable*>(section.get()); }
      }

      size_t phnum;
      if (elf_getphdrnum(e, &phnum) < 0) { return elfError("elf_getphdrnum failed"); }
      for (size_t i = 0; i < phnum; ++i) {
        segments.push_back(std::unique_ptr<GElfSegment>(new GElfSegment(this, i)));
        if (!segments[i]->pull()) { return false; }
      }

      return true;
    }

    bool GElfImage::elfError(const char* msg)
    {
      out << "Error: " << msg << ": " << elfError() << std::endl;
      return false;
    }

    uint64_t GElfImage::size()
    {
      if (buffer) {
        return ElfSize(buffer);
      } else {
        return img.getSize();
      }
    }

    bool GElfImage::push0()
    {
      assert(e);
      for (std::unique_ptr<GElfSection>& section : sections) {
        if (section && !section->push()) { return false; }
      }

      for (std::unique_ptr<GElfSection>& section : sections) {
        if (section && !section->pull0()) { return false; }
      }

      if (!segments.empty()) {
        if (!gelf_newphdr(e, segments.size())) { return elfError("gelf_newphdr failed"); }
      }
      if (elf_update(e, ELF_C_NULL) < 0) { return elfError("elf_update (1.1) failed"); }
      if (!segments.empty()) {
        for (std::unique_ptr<GElfSection>& section : sections) {
          // Update section offsets.
          if (section && !section->pull0()) { return false; }
        }
        uint64_t vaddr = 0;
        for (std::unique_ptr<GElfSegment>& segment : segments) {
          if (!segment->push(vaddr)) { return false; }
          vaddr = segment->vaddr() + segment->memSize();
        }
      }
      return true;
    }

    bool GElfImage::push()
    {
      if (!push0()) { return false; }
      if (elf_update(e, ELF_C_WRITE) < 0) { return elfError("elf_update (2) failed"); }
      return true;
    }

    Segment* GElfImage::segmentByVAddr(uint64_t vaddr)
    {
      for (std::unique_ptr<GElfSegment>& seg : segments) {
        if (seg->vaddr() <= vaddr && vaddr < seg->vaddr() + seg->memSize()) {
          return seg.get();
        }
      }
      return 0;
    }

    Section* GElfImage::sectionByVAddr(uint64_t vaddr)
    {
      for (size_t n = 1; n < sections.size(); ++n) {
        if (sections[n]->addr() <= vaddr && vaddr < sections[n]->addr() + sections[n]->size()) {
          return sections[n].get();
        }
      }
      return nullptr;
    }

    bool GElfImage::elfEnd()
    {
      return false;
    }

    bool GElfImage::writeTo(const std::string& filename)
    {
      if (!img.writeTo(filename)) { return imgError(); }
      return true;
    }

    bool GElfImage::copyToBuffer(void** buf, size_t* size)
    {
      if (buffer) {
        *buf = malloc(bufferSize);
        memcpy(*buf, buffer, bufferSize);
        if (size) { *size = bufferSize; }
        return true;
      } else {
        return img.copyTo(buf, size);
      }
    }

    bool GElfImage::copyToBuffer(void* buf, size_t size)
    {
      if (buffer) {
        if (size < bufferSize) { return false; }
        memcpy(buf, buffer, bufferSize);
        return true;
      } else {
        return img.copyTo(buf, size);
      }
    }

    GElfStringTable* GElfImage::addStringTable(const std::string& name)
    {
      GElfStringTable* stab = new GElfStringTable(this);
      sections.push_back(std::unique_ptr<GElfStringTable>(stab));
      return stab;
    }

    GElfStringTable* GElfImage::getStringTable(uint16_t index)
    {
      return static_cast<GElfStringTable*>(sections[index].get());
    }

    GElfSymbolTable* GElfImage::addSymbolTable(const std::string& name, StringTable* stab)
    {
      if (!stab) { stab = strtab(); }
      const char* name0 = shstrtab()->addString(name);
      GElfSymbolTable* symtab = new GElfSymbolTable(this);
      symtab->push(name0, static_cast<GElfStringTable*>(stab));
      sections.push_back(std::unique_ptr<GElfSection>(symtab));
      return symtab;
    }

    GElfStringTable* GElfImage::shstrtab() {
      if (!shstrtabSection) {
        shstrtabSection = addStringTable(".shstrtab");
      }
      return shstrtabSection;
    }

    GElfStringTable* GElfImage::strtab() {
      if (!strtabSection) {
        strtabSection = addStringTable(".shstrtab");
      }
      return strtabSection;
    }

    GElfSymbolTable* GElfImage::symtab()
    {
      if (!symtabSection) {
        symtabSection = addSymbolTable(".symtab", strtab());
      }
      return symtabSection;
    }

    GElfSymbolTable* GElfImage::dynsym()
    {
      if (!dynsymSection) {
        dynsymSection = addSymbolTable(".dynsym", strtab());
      }
      return dynsymSection;
    }

    GElfSymbolTable* GElfImage::getSymbolTable()
    {
      const char *UseDynsym = getenv("LOADER_USE_DYNSYM");
      if (UseDynsym && std::strncmp(UseDynsym, "0", 1) != 0)
        return dynsym();
      return symtab();
    }

    GElfNoteSection* GElfImage::note()
    {
      if (!noteSection) { noteSection = addNoteSection(".note"); }
      return noteSection;
    }

    GElfNoteSection* GElfImage::addNoteSection(const std::string& name)
    {
      GElfNoteSection* note = new GElfNoteSection(this);
      note->push(name);
      sections.push_back(std::unique_ptr<GElfSection>(note));
      return note;
    }

    Segment* GElfImage::initSegment(uint32_t type, uint32_t flags, uint64_t paddr)
    {
      GElfSegment *seg = new (std::nothrow) GElfSegment(this, segments.size(), type, flags, paddr);
      segments.push_back(std::unique_ptr<GElfSegment>(seg));
      return seg;
    }

    bool GElfImage::addSegments()
    {
      return true;
    }

    Section* GElfImage::addSection(const std::string &name,
                                   uint32_t type,
                                   uint64_t flags,
                                   uint64_t entsize, Segment* segment)
    {
      GElfSection *section = new (std::nothrow) GElfSection(this);
      if (!section || !section->push(name.c_str(), type, flags, 0, 0, 0, entsize)) {
        delete section;
        return nullptr;
      }
      if (segment) {
        if (!segment->updateAddSection(section)) {
          delete section;
          return nullptr;
        }
      }
      sections.push_back(std::unique_ptr<GElfSection>(section));
      return section;
    }

    RelocationSection* GElfImage::addRelocationSection(Section* sec, SymbolTable* symtab)
    {
      std::string section_name = ".rela" + sec->Name();
      if (!symtab) { symtab = this->symtab(); }
      GElfRelocationSection *rsec = new GElfRelocationSection(this, sec, (GElfSymbolTable*) symtab);
      if (!rsec || !rsec->push(section_name)) {
        delete rsec;
        return nullptr;
      }
      sections.push_back(std::unique_ptr<GElfRelocationSection>(rsec));
      return rsec;
    }

    RelocationSection* GElfImage::relocationSection(Section* sec, SymbolTable* symtab)
    {
      return sec->relocationSection(symtab);
    }

    uint16_t GElfImage::machine() const
    {
      return ehdr.e_machine;
    }

    uint16_t GElfImage::etype() const
    {
      return ehdr.e_type;
    }

    Image* NewElf32Image() { return new GElfImage(ELFCLASS32); }
    Image* NewElf64Image() { return new GElfImage(ELFCLASS64); }

    uint64_t ElfSize(const void* emi)
    {
      const Elf64_Ehdr *ehdr = (const Elf64_Ehdr*) emi;
      if (NULL == ehdr || EV_CURRENT != ehdr->e_version) {
        return false;
      }

      const Elf64_Shdr *shdr = (const Elf64_Shdr*)((char*)emi + ehdr->e_shoff);
      if (NULL == shdr) {
        return false;
      }

      uint64_t max_offset = ehdr->e_shoff;
      uint64_t total_size = max_offset + static_cast<uint64_t>(ehdr->e_shentsize) * static_cast<uint64_t>(ehdr->e_shnum);

      for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
        uint64_t cur_offset = static_cast<uint64_t>(shdr[i].sh_offset);
        if (max_offset < cur_offset) {
          max_offset = cur_offset;
          total_size = max_offset;
          if (SHT_NOBITS != shdr[i].sh_type) {
            total_size += static_cast<uint64_t>(shdr[i].sh_size);
          }
        }
      }

      return total_size;
    }

    std::string GetNoteString(uint32_t s_size, const char* s)
    {
      if (!s_size) { return ""; }
      if (s[s_size-1] == '\0') {
        return std::string(s, s_size-1);
      } else {
        return std::string(s, s_size);
      }
    }

}   //  namespace elf
}   //  namespace amd
}   //  namespace rocr
