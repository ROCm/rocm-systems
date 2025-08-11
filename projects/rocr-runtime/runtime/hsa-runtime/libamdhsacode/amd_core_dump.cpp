////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
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

#include <unistd.h>
#include <elf.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <cstring>
#include <vector>
#include <sstream>
#include <fstream>
#include <memory>
#include "core/util/utils.h"
#include "core/inc/runtime.h"
#include "./amd_hsa_code_util.hpp"
#include "core/inc/amd_core_dump.hpp"
#include "hsakmt/hsakmt.h"

constexpr char SNAPSHOT_INFO_ALIGNMENT = 0x8;
constexpr uint32_t LOAD_ALIGNMENT_SHIFT = 4;
constexpr uint32_t NOTE_ALIGNMENT_SHIFT = 2;
const std::string PREFIX_FILE_NAME = "gpucore";
constexpr size_t MAX_BUFFER_SIZE = 4 * 1024 * 1024;

namespace rocr {
namespace amd {
namespace coredump {
/* Implementation details */
namespace impl {
class PackageBuilder {
 public:
  PackageBuilder() : st_(std::stringstream::out | std::stringstream::binary) {}
  size_t Size() const { return st_.str().size(); }
  template <typename T, typename = typename std::enable_if<!std::is_pointer<T>::value>::type>
  void Write(const T& v) {
    st_.write((char*)&v, sizeof(T));
  }
  void Write(const std::vector<uint8_t>& v) { st_.write((const char*)v.data(), v.size()); }
  void Write(void* data, uint32_t size) { st_.write((const char*)data, size); }
  bool GetBuffer(void* out) {
    size_t sz = Size();

    if (!sz) return false;
    std::memcpy(out, st_.str().c_str(), sz);
    return true;
  }
  void Print(void* buf, uint64_t size) {
    int i;
    for (i = 0; i < size; i++) debug_print("%02x ", 0xFF & ((uint8_t*)buf)[i]);
    debug_print("\n");
  }
 private:
  std::stringstream st_;
};

enum SegmentType { LOAD, NOTE };
struct SegmentBuilder;

struct SegmentInfo {
  SegmentType stype;
  uint64_t vaddr = 0;
  uint64_t size = 0;
  uint32_t flags = 0;
  SegmentBuilder* builder;
};

using SegmentsInfo = std::vector<SegmentInfo>;
using rocr::amd::hsa::alignUp;
struct SegmentBuilder {
  virtual ~SegmentBuilder() = default;
  /* Find which segments needs to be created.  */
  virtual hsa_status_t Collect(SegmentsInfo& segments) = 0;
  /* Called to read a given SegmentInfo's data.  */
  virtual hsa_status_t Read(void* buf, size_t buf_size, off_t offset) = 0;
};

struct NoteSegmentBuilder : public SegmentBuilder {
  hsa_status_t Collect(SegmentsInfo& segments) override {
    void *runtime_ptr, *agents_ptr = NULL, *queues_ptr = NULL;
    uint32_t runtime_size, agents_size, queue_size, n_entries, entry_size;
    HsaVersionInfo versionInfo = {0};

    if (HSAKMT_CALL(hsaKmtDbgEnable(&runtime_ptr, &runtime_size))) {
      fprintf(stderr, "Failed to enable debug interface, "
              "debugger might be already attached.\n");
      return HSA_STATUS_ERROR;
    }
    std::unique_ptr<void, decltype(std::free) *> runtime_info(runtime_ptr, std::free);

    if (HSAKMT_CALL(hsaKmtGetVersion(&versionInfo))) {
      HSAKMT_CALL(hsaKmtDbgDisable());
      fprintf(stderr, "Failed to fetch driver ABI version.\n");
      return HSA_STATUS_ERROR;
    }
    /* Note version */
    note_package_builder_.Write<uint64_t>(1);
    /* Store version_major in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(versionInfo.KernelInterfaceMajorVersion);
    /* Store version_minor in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(versionInfo.KernelInterfaceMinorVersion);
    /* Store runtime_info_size in PT_NOTE package */
    note_package_builder_.Write<uint64_t>(runtime_size);

    if (HSAKMT_CALL(hsaKmtDbgGetDeviceData(&agents_ptr, &n_entries, &entry_size))) {
       HSAKMT_CALL(hsaKmtDbgDisable());
       fprintf(stderr, "Failed to fetch agents snapshot.\n");
       return HSA_STATUS_ERROR;
    }
    agents_size = n_entries * entry_size;
    std::unique_ptr<void, decltype(std::free) *> agents_info(agents_ptr, std::free);
    /* Store n_agents in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(n_entries);
    /* Store agent_info_entry_size in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(entry_size);

    if (HSAKMT_CALL(hsaKmtDbgGetQueueData(&queues_ptr, &n_entries, &entry_size, true))) {
       HSAKMT_CALL(hsaKmtDbgDisable());
       fprintf(stderr, "Failed to fetch queues snapshot.\n");
       return HSA_STATUS_ERROR;
    }
    queue_size = n_entries * entry_size;
    std::unique_ptr<void, decltype(std::free) *> queues_info(queues_ptr, std::free);
    /* Store n_queues in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(n_entries);
    /* Store queue_info_entry_size in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(entry_size);

    PushInfo(runtime_info.get(), runtime_size);
    PushInfo(agents_info.get(), agents_size);
    PushInfo(queues_info.get(), queue_size);
    if (HSAKMT_CALL(hsaKmtDbgDisable())) {
      fprintf(stderr, "Failed to disable debug interface.\n");
      return HSA_STATUS_ERROR;
    }

    /* With note content, package this in the PT_NOTE.  */
    PackageBuilder noteHeaderBuilder;
    noteHeaderBuilder.Write<uint32_t> (7);  /* namesz */
    noteHeaderBuilder.Write<uint32_t> (note_package_builder_.Size());
    noteHeaderBuilder.Write<uint32_t> (NT_AMDGPU_CORE_STATE);  /* type.  */
    noteHeaderBuilder.Write<char[8]> ("AMDGPU\0");

    raw_.resize(noteHeaderBuilder.Size() + note_package_builder_.Size());
    if (!(noteHeaderBuilder.GetBuffer(raw_.data())
          && note_package_builder_.GetBuffer(&raw_[noteHeaderBuilder.Size()]))) {
      fprintf(stderr, "Failed to build the NT_AMDGPU_CORE_STATE note.\n");
      return HSA_STATUS_ERROR;
    }

    SegmentInfo s;
    s.stype = NOTE;
    s.vaddr = 0;
    s.size = raw_.size();
    s.flags = 0;
    s.builder = this;
    segments.push_back(s);

    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t Read(void* buf, size_t buf_size, off_t offset) override {
    if (offset + buf_size >raw_.size ()) return HSA_STATUS_ERROR;
    memcpy(buf, raw_.data() + offset, buf_size);
    return HSA_STATUS_SUCCESS;
  }

 private:
  PackageBuilder note_package_builder_;
  std::vector<unsigned char> raw_;

  void PushInfo(void *data, uint32_t size) {
    note_package_builder_.Write(data, size);
    size = alignUp(size, SNAPSHOT_INFO_ALIGNMENT) - size;
    for (int i = 0; i < size; i++)
      note_package_builder_.Write<uint8_t>(0);
  }
};

struct LoadSegmentBuilder : public SegmentBuilder {
  LoadSegmentBuilder() : fd_(open("/proc/self/mem", O_RDONLY)) {}

  ~LoadSegmentBuilder() {
    if (fd_ != -1) close(fd_);
  }

  hsa_status_t Collect(SegmentsInfo& segments) override {
    const std::string maps_path = "/proc/self/maps";
    std::ifstream maps(maps_path);
    if (!maps.is_open()) {
      fprintf(stderr, "Could not open '%s'", maps_path.c_str());
      return HSA_STATUS_ERROR;
    }

    std::string line;
    while (std::getline(maps, line)) {
      std::istringstream isl{ line };
      std::string address, perms, offset, dev, inode, path;
      if (!(isl >> address >> perms >> offset >> dev >> inode)) {
        fprintf(stderr, "Failed to parse '%s'", maps_path.c_str());
        return HSA_STATUS_ERROR;
      }

      std::getline(isl >> std::ws, path);

      /* Look for the /dev/dri/renderD* files.  */
      if (path.rfind("/dev/dri/renderD", 0) == 0) {
        uint64_t start, end;
        if (sscanf(address.c_str(), "%lx-%lx", &start, &end) != 2) {
          fprintf(stderr, "Failed to parse '%s'", maps_path.c_str());
          return HSA_STATUS_ERROR;
        }
        uint32_t flags = SHF_ALLOC;
        flags |= (perms.find('w', 0) != std::string::npos) ? SHF_WRITE : 0;
        flags |= (perms.find('x', 0) != std::string::npos) ? SHF_EXECINSTR : 0;
        uint64_t size = end - start;

        debug_print("LOAD 0x%lx size: %ld\n", start, size);
        SegmentInfo s;
        s.stype = LOAD;
        s.vaddr = start;
        s.size = size;
        s.flags = flags;
        s.builder = this;
        segments.push_back(s);
       }
     }
     return HSA_STATUS_SUCCESS;
  }

  hsa_status_t Read(void* buf, size_t buf_size, off_t offset) override {
    if (fd_ == -1) return HSA_STATUS_ERROR;

    size_t done = 0;
    ssize_t read;
    do {
      read = pread(fd_, static_cast<char *>(buf) + done, buf_size - done,
                   offset + done);

      if (read == -1 && errno != EINTR) {
        perror("Failed to read GPU memory");
        return HSA_STATUS_ERROR;
      }
      else if (read > 0)
        done += read;
    } while (read != 0 && done < buf_size);

    if (read == 0 && done < buf_size) {
      fprintf(stderr, "Reached unexpected EOF while reading VRAM.\n");
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

 private:
  int fd_ = -1;
};

hsa_status_t build_core_dump(const std::string& filename, const SegmentsInfo& segments, size_t size_limit) {
  std::unique_ptr<unsigned char[]> copy_buffer(new unsigned char[MAX_BUFFER_SIZE]);
  if (!segments.size()) return HSA_STATUS_SUCCESS;
  SegmentInfo front = segments.front();
  off_t offset = sizeof(Elf64_Ehdr) + segments.size() * sizeof(Elf64_Phdr);

  if (size_limit != -1 && (offset + front.size > size_limit)) {
    debug_print("Core file size over limit\n");
    return HSA_STATUS_SUCCESS;
  }
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    perror("Failed to create GPU coredump");
    return HSA_STATUS_ERROR;
  }
  Elf64_Ehdr ehdr{};
  ehdr.e_ident[EI_MAG0] = ELFMAG0;
  ehdr.e_ident[EI_MAG1] = ELFMAG1;
  ehdr.e_ident[EI_MAG2] = ELFMAG2;
  ehdr.e_ident[EI_MAG3] = ELFMAG3;
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr.e_ident[EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = 0;
  ehdr.e_type = ET_CORE;
  ehdr.e_machine = ELF::EM_AMDGPU;
  ehdr.e_version = EV_CURRENT;
  ehdr.e_entry = 0;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = 0;
  ehdr.e_flags = 0;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = segments.size();
  ehdr.e_shentsize = 0;
  ehdr.e_shnum = 0;
  ehdr.e_shstrndx = 0;

  if (write(fd, &ehdr, sizeof(ehdr)) == -1) {
    perror("Failed to write ELF header");
    close(fd);
    return HSA_STATUS_ERROR;
  }

  /* Make sure that the underlying file has enough space for the file headers. */
  int error = posix_fallocate(fd, sizeof(Elf64_Ehdr), segments.size() * sizeof(Elf64_Phdr));
  if (error != 0) {
    fprintf(stderr, "Failed to allocate file: %s\n", strerror(error));
    close(fd);
    return HSA_STATUS_ERROR;
  }
  size_t idx = 0;
  for (SegmentInfo seg : segments) {
    Elf64_Phdr phdr{};
    phdr.p_type = [](SegmentType s) {
      switch (s) {
        case LOAD:
          return PT_LOAD;
        case NOTE:
          return PT_NOTE;
        default:
          assert(false);
          return PT_NULL;
      }
    }(seg.stype);
    phdr.p_flags = seg.flags;
    phdr.p_vaddr = seg.vaddr;
    phdr.p_paddr = 0;
    phdr.p_memsz = seg.size;
    phdr.p_filesz = seg.size;
    phdr.p_align = [](SegmentType s) {
      switch (s) {
        case LOAD:
          return LOAD_ALIGNMENT_SHIFT;
        case NOTE:
          return NOTE_ALIGNMENT_SHIFT;
        default:
          assert(false);
          return (uint32_t)0;
      }
    }(seg.stype);
    if (size_limit != -1 && (offset + seg.size > size_limit)) {
      printf("Core limit file reached. GPU core dump created: %s\n", filename.c_str());
      close(fd);
      return HSA_STATUS_SUCCESS;
    }
    phdr.p_offset = alignUp(offset, (uint64_t)1 << phdr.p_align);
    if (pwrite(fd, &phdr, sizeof(phdr), sizeof(Elf64_Ehdr) + idx * sizeof(Elf64_Phdr)) == -1) {
      perror("Failed to write ELF header");
      close(fd);
      return HSA_STATUS_ERROR;
    }
    /* Allocate stace for the segment on the file, and write the segment
       content.  */
    error = posix_fallocate(fd, phdr.p_offset, phdr.p_filesz);
    if (error != 0) {
      fprintf(stderr, "Failed to allocate file: %s\n", strerror(error));
      close(fd);
      return HSA_STATUS_ERROR;
    }
    size_t remaining = phdr.p_filesz;
    while (remaining > 0) {
      size_t curr_chunk = std::min(remaining, MAX_BUFFER_SIZE);
      try {
        hsa_status_t st = seg.builder->Read(copy_buffer.get(), curr_chunk,
                                                    phdr.p_vaddr + phdr.p_filesz - remaining);
        if (st != HSA_STATUS_SUCCESS) {
          close(fd);
          return st;
        }
        if (pwrite(fd, copy_buffer.get(), curr_chunk, phdr.p_offset + phdr.p_filesz - remaining) ==
            -1) {
          perror("Failed to white core dump");
          close(fd);
          return HSA_STATUS_ERROR;
        }
      } catch (...) {
        close(fd);
        return HSA_STATUS_ERROR;
      }
      remaining -= curr_chunk;
    }
    offset += phdr.p_filesz;
    idx++;
  }
  printf("GPU core dump created: %s\n", filename.c_str());
  close(fd);
  return HSA_STATUS_SUCCESS;
}
}   //  namespace impl

hsa_status_t dump_gpu_core() {
  impl::NoteSegmentBuilder nbuilder;
  impl::LoadSegmentBuilder lbuilder;
  impl::SegmentsInfo segments;

  struct rlimit rlimit;

  if (getrlimit(RLIMIT_CORE, &rlimit)) {
    perror("Could not get core file size\n");
    return HSA_STATUS_ERROR;
  }
  debug_print("core file size: %ld\n", rlimit.rlim_cur);

  if (rlimit.rlim_cur == 0)
    return HSA_STATUS_SUCCESS;

  hsa_status_t status = nbuilder.Collect(segments);
  if (status != HSA_STATUS_SUCCESS) return status;

  status = lbuilder.Collect(segments);
  if (status != HSA_STATUS_SUCCESS) return status;

  std::stringstream st;
  st << PREFIX_FILE_NAME << "." << getpid();
  return build_core_dump(st.str(), segments, rlimit.rlim_cur);
}
}   //  namespace coredump
}   //  namespace amd
}   //  namespace rocr
