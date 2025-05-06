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

#ifndef AMD_ELF_IMAGE_HPP_
#define AMD_ELF_IMAGE_HPP_

#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <cstdint>

namespace rocr {
namespace amd {
namespace elf {
    class Symbol;
    class SymbolTable;
    class Section;
    class RelocationSection;

    class Segment {
    public:
      virtual ~Segment() { }
      virtual uint64_t type() const = 0;
      virtual uint64_t memSize() const = 0;
      virtual uint64_t align() const = 0;
      virtual uint64_t imageSize() const = 0;
      virtual uint64_t vaddr() const = 0;
      virtual uint64_t flags() const = 0;
      virtual uint64_t offset() const = 0;
      virtual const char* data() const = 0;
      virtual uint16_t getSegmentIndex() = 0;
      virtual bool updateAddSection(Section *section) = 0;
    };

    class Section {
    public:
      virtual ~Section() { }
      virtual uint16_t getSectionIndex() const = 0;
      virtual uint32_t type() const = 0;
      virtual std::string Name() const = 0;
      virtual uint64_t offset() const = 0;
      virtual uint64_t addr() const = 0;
      virtual bool updateAddr(uint64_t addr) = 0;
      virtual uint64_t addralign() const = 0;
      virtual uint64_t flags() const = 0;
      virtual uint64_t size() const = 0;
      virtual uint64_t nextDataOffset(uint64_t align) const = 0;
      virtual uint64_t addData(const void *src, uint64_t size, uint64_t align) = 0;
      virtual bool getData(uint64_t offset, void* dest, uint64_t size) = 0;
      virtual Segment* segment() = 0;
      virtual RelocationSection* asRelocationSection() = 0;
      virtual bool hasRelocationSection() const = 0;
      virtual RelocationSection* relocationSection(SymbolTable* symtab = 0) = 0;
      virtual bool setMemSize(uint64_t s) = 0;
      virtual uint64_t memSize() const = 0;
      virtual bool setAlign(uint64_t a) = 0;
      virtual uint64_t memAlign() const = 0;
    };

    class Relocation {
    public:
      virtual ~Relocation() { }
      virtual RelocationSection* section() = 0;
      virtual uint32_t type() = 0;
      virtual uint32_t symbolIndex() = 0;
      virtual Symbol* symbol() = 0;
      virtual uint64_t offset() = 0;
      virtual int64_t addend() = 0;
    };

    class RelocationSection : public virtual Section {
    public:
      virtual Relocation* addRelocation(uint32_t type, Symbol* symbol, uint64_t offset, int64_t addend) = 0;
      virtual size_t relocationCount() const = 0;
      virtual Relocation* relocation(size_t i) = 0;
      virtual Section* targetSection() = 0;
    };

    class StringTable : public virtual Section {
    public:
      virtual const char* addString(const std::string& s) = 0;
      virtual size_t addString1(const std::string& s) = 0;
      virtual const char* getString(size_t ndx) = 0;
      virtual size_t getStringIndex(const char* name) = 0;
    };

    class Symbol {
    public:
      virtual ~Symbol() { }
      virtual uint32_t index() = 0;
      virtual uint32_t type() = 0;
      virtual uint32_t binding() = 0;
      virtual uint64_t size() = 0;
      virtual uint64_t value() = 0;
      virtual unsigned char other() = 0;
      virtual std::string name() = 0;
      virtual Section* section() = 0;
      virtual void setValue(uint64_t value) = 0;
      virtual void setSize(uint64_t size) = 0;
    };

    class SymbolTable : public virtual Section {
    public:
      virtual Symbol* addSymbol(Section* section, const std::string& name, uint64_t value, uint64_t size, unsigned char type, unsigned char binding, unsigned char other = 0) = 0;
      virtual size_t symbolCount() = 0;
      virtual Symbol* symbol(size_t i) = 0;
    };

    class NoteSection : public virtual Section {
    public:
      virtual bool addNote(const std::string& name, uint32_t type, const void* desc = 0, uint32_t desc_size = 0) = 0;
      virtual bool getNote(const std::string& name, uint32_t type, void** desc, uint32_t* desc_size) = 0;
    };

    class Image {
    public:
      virtual ~Image() { }

      virtual bool initNew(uint16_t machine, uint16_t type, uint8_t os_abi = 0, uint8_t abi_version = 0, uint32_t e_flags = 0) = 0;
      virtual bool loadFromFile(const std::string& filename) = 0;
      virtual bool saveToFile(const std::string& filename) = 0;
      virtual bool initFromBuffer(const void* buffer, size_t size) = 0;
      virtual bool initAsBuffer(const void* buffer, size_t size) = 0;
      virtual bool writeTo(const std::string& filename) = 0;
      virtual bool copyToBuffer(void** buf, size_t* size = 0) = 0; // Copy to new buffer allocated with malloc
      virtual bool copyToBuffer(void* buf, size_t size) = 0; // Copy to existing buffer of given size.

      virtual const char* data() = 0;
      virtual uint64_t size() = 0;

      virtual uint16_t Machine() = 0;
      virtual uint16_t Type() = 0;
      virtual uint32_t EFlags() = 0;
      virtual uint32_t ABIVersion() = 0;
      virtual uint32_t EClass() = 0;
      virtual uint32_t OsAbi() = 0;

      std::string output() { return out.str(); }

      virtual bool Freeze() = 0;
      virtual bool Validate() = 0;

      virtual StringTable* shstrtab() = 0;
      virtual StringTable* strtab() = 0;
      virtual SymbolTable* symtab() = 0;
      virtual SymbolTable* getSymtab(uint16_t index) = 0;
      virtual SymbolTable* dynsym() = 0;
      virtual SymbolTable* getDynsym(uint16_t index) = 0;
      virtual SymbolTable* getSymbolTable() = 0;
      virtual SymbolTable* getSymbolTable(uint16_t index) = 0;

      virtual StringTable* addStringTable(const std::string& name) = 0;
      virtual StringTable* getStringTable(uint16_t index) = 0;

      virtual SymbolTable* addSymbolTable(const std::string& name, StringTable* stab = 0) = 0;

      virtual size_t segmentCount() = 0;
      virtual Segment* segment(size_t i) = 0;
      virtual Segment* segmentByVAddr(uint64_t vaddr) = 0;

      virtual size_t sectionCount() = 0;
      virtual Section* section(size_t i) = 0;
      virtual Section* sectionByVAddr(uint64_t vaddr) = 0;

      virtual NoteSection* note() = 0;
      virtual NoteSection* addNoteSection(const std::string& name) = 0;

      virtual Segment* initSegment(uint32_t type, uint32_t flags, uint64_t paddr = 0) = 0;
      virtual bool addSegments() = 0;

      virtual Section* addSection(const std::string &name,
                                  uint32_t type,
                                  uint64_t flags = 0,
                                  uint64_t entsize = 0,
                                  Segment* segment = 0) = 0;

      virtual RelocationSection* relocationSection(Section* sec, SymbolTable* symtab = 0) = 0;

    protected:
      std::ostringstream out;
    };

    Image* NewElf32Image();
    Image* NewElf64Image();

    uint64_t ElfSize(const void* buffer);

    std::string GetNoteString(uint32_t s_size, const char* s);

}   //  namespace elf
}   //  namespace amd
}   //  namespace rocr

#endif // AMD_ELF_IMAGE_HPP_
