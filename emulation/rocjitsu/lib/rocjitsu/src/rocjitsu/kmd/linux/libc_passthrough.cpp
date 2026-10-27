// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file libc_passthrough.cpp
/// @brief Named libc wrappers and allocation-free mapping bootstrap.

#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include "util/dynamic_loader.h"

#include <cassert>
#include <cstdint>
#include <dlfcn.h>
#include <link.h>
#include <string_view>

namespace rocjitsu {

void *bootstrap_mapping_symbol(const char *name) {
  struct Lookup {
    const char *name;
    void *address;
  } lookup{name, nullptr};
  // Unlike dlsym's error-state initialization on older glibc, this walk does
  // not allocate. Its recursive loader lock also permits allocator re-entry
  // from a lookup that is already in progress.
  dl_iterate_phdr(
      +[](dl_phdr_info *info, size_t, void *opaque) {
        auto &lookup = *static_cast<Lookup *>(opaque);
        const ElfW(Dyn) *dynamic = nullptr;
        for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index)
          if (info->dlpi_phdr[index].p_type == PT_DYNAMIC)
            dynamic = reinterpret_cast<const ElfW(Dyn) *>(info->dlpi_addr +
                                                          info->dlpi_phdr[index].p_vaddr);
        if (!dynamic)
          return 0;
        const ElfW(Sym) *symbols = nullptr;
        const char *strings = nullptr;
        const uint32_t *hash = nullptr, *gnu_hash = nullptr;
        const ElfW(Half) *versions = nullptr;
        ElfW(Xword) soname = 0;
        const auto loaded_address = [info](ElfW(Addr) address) {
          // glibc relocates dynamic pointers in ordinary DSOs, while objects
          // such as the vDSO can retain relative virtual addresses.
          for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
            const ElfW(Phdr) &segment = info->dlpi_phdr[index];
            const ElfW(Addr) begin = info->dlpi_addr + segment.p_vaddr;
            if (segment.p_type == PT_LOAD && address >= begin && address - begin < segment.p_memsz)
              return address;
          }
          return info->dlpi_addr + address;
        };
        for (const ElfW(Dyn) *entry = dynamic; entry->d_tag != DT_NULL; ++entry) {
          const ElfW(Addr) address = loaded_address(entry->d_un.d_ptr);
          switch (entry->d_tag) {
          case DT_SONAME:
            soname = entry->d_un.d_val;
            break;
          case DT_SYMTAB:
            symbols = reinterpret_cast<const ElfW(Sym) *>(address);
            break;
          case DT_STRTAB:
            strings = reinterpret_cast<const char *>(address);
            break;
          case DT_HASH:
            hash = reinterpret_cast<const uint32_t *>(address);
            break;
          case DT_GNU_HASH:
            gnu_hash = reinterpret_cast<const uint32_t *>(address);
            break;
          case DT_VERSYM:
            versions = reinterpret_cast<const ElfW(Half) *>(address);
            break;
          }
        }
        // A preload or symlink can give libc any pathname; its ELF identity
        // remains libc.so.6 regardless of the name retained in dlpi_name.
        if (!strings || !soname || std::string_view(strings + soname) != "libc.so.6")
          return 0;
        if (!symbols)
          return 1;
        uint32_t count = hash ? hash[1] : 0;
        if (!count && gnu_hash) {
          const uint32_t bucket_count = gnu_hash[0], first_symbol = gnu_hash[1];
          const auto *buckets = reinterpret_cast<const uint32_t *>(
              reinterpret_cast<const ElfW(Addr) *>(gnu_hash + 4) + gnu_hash[2]);
          const auto *chains = buckets + bucket_count;
          count = first_symbol;
          for (uint32_t index = 0; index < bucket_count; ++index) {
            uint32_t symbol = buckets[index];
            if (symbol < first_symbol)
              continue;
            while (!(chains[symbol - first_symbol] & 1))
              ++symbol;
            if (count <= symbol)
              count = symbol + 1;
          }
        }
        for (uint32_t index = 0; index < count; ++index) {
          const ElfW(Sym) &symbol = symbols[index];
          if (symbol.st_shndx == SHN_UNDEF || ELF64_ST_TYPE(symbol.st_info) != STT_FUNC ||
              (versions && (versions[index] & 0x8000)))
            continue;
          if (std::string_view(strings + symbol.st_name) == lookup.name) {
            lookup.address = reinterpret_cast<void *>(info->dlpi_addr + symbol.st_value);
            break;
          }
        }
        return 1;
      },
      &lookup);
  return lookup.address;
}

void LibcPassthrough::resolve() {
  if (initialized_)
    return;

  auto *handle = RTLD_NEXT;
  openat = util::lookup_symbol<decltype(openat)>(handle, "openat");
  close = util::lookup_symbol<decltype(close)>(handle, "close");
  read = util::lookup_symbol<decltype(read)>(handle, "read");
  write = util::lookup_symbol<decltype(write)>(handle, "write");
  ioctl = util::lookup_symbol<decltype(ioctl)>(handle, "ioctl");
  mmap = util::lookup_symbol<decltype(mmap)>(handle, "mmap");
  munmap = util::lookup_symbol<decltype(munmap)>(handle, "munmap");
  mremap = util::lookup_symbol<decltype(mremap)>(handle, "mremap");
  mprotect = util::lookup_symbol<decltype(mprotect)>(handle, "mprotect");
  madvise = util::lookup_symbol<decltype(madvise)>(handle, "madvise");
  memfd_create = util::lookup_symbol<decltype(memfd_create)>(handle, "memfd_create");
  dup = util::lookup_symbol<decltype(dup)>(handle, "dup");
  dup2 = util::lookup_symbol<decltype(dup2)>(handle, "dup2");
  dup3 = util::lookup_symbol<decltype(dup3)>(handle, "dup3");
  fcntl = util::lookup_symbol<decltype(fcntl)>(handle, "fcntl");
  fopen = util::lookup_symbol<decltype(fopen)>(handle, "fopen");
  freopen = util::lookup_symbol<decltype(freopen)>(handle, "freopen");
  opendir = util::lookup_symbol<decltype(opendir)>(handle, "opendir");
  readdir = util::lookup_symbol<decltype(readdir)>(handle, "readdir");
  closedir = util::lookup_symbol<decltype(closedir)>(handle, "closedir");
  stat = util::lookup_symbol<decltype(stat)>(handle, "stat");
  lstat = util::lookup_symbol<decltype(lstat)>(handle, "lstat");
  access = util::lookup_symbol<decltype(access)>(handle, "access");
  fstat_fn = util::lookup_symbol<decltype(fstat_fn)>(handle, "fstat");
  // Legacy aliases, resolved eagerly so no wrapper needs a lazy static (see the
  // declarations for why that matters across fork).
  fstat64_fn = util::lookup_symbol<decltype(fstat64_fn)>(handle, "fstat64");
  fxstat_fn = util::lookup_symbol<decltype(fxstat_fn)>(handle, "__fxstat");
  fxstat64_fn = util::lookup_symbol<decltype(fxstat64_fn)>(handle, "__fxstat64");
  stat64_fn = util::lookup_symbol<decltype(stat64_fn)>(handle, "stat64");
  lstat64_fn = util::lookup_symbol<decltype(lstat64_fn)>(handle, "lstat64");
  xstat_fn = util::lookup_symbol<decltype(xstat_fn)>(handle, "__xstat");
  xstat64_fn = util::lookup_symbol<decltype(xstat64_fn)>(handle, "__xstat64");
  lxstat_fn = util::lookup_symbol<decltype(lxstat_fn)>(handle, "__lxstat");
  lxstat64_fn = util::lookup_symbol<decltype(lxstat64_fn)>(handle, "__lxstat64");
  readlink_fn = util::lookup_symbol<decltype(readlink_fn)>(handle, "readlink");
  realpath_fn = util::lookup_symbol<decltype(realpath_fn)>(handle, "realpath");
  realpath_chk_fn = util::lookup_symbol<decltype(realpath_chk_fn)>(handle, "__realpath_chk");
  fork = util::lookup_symbol<decltype(fork)>(handle, "fork");
  clock_gettime = util::lookup_symbol<decltype(clock_gettime)>(handle, "clock_gettime");
  // rocjitsu does not interpose mutex operations. Match std::mutex's global
  // binding so a preloaded TSan runtime observes both acquisition and release.
  pthread_mutex_lock =
      util::lookup_symbol<decltype(pthread_mutex_lock)>(RTLD_DEFAULT, "pthread_mutex_lock");
  sem_init = util::lookup_symbol<decltype(sem_init)>(handle, "sem_init");
  sem_destroy = util::lookup_symbol<decltype(sem_destroy)>(handle, "sem_destroy");
  sem_post = util::lookup_symbol<decltype(sem_post)>(handle, "sem_post");
  sem_clockwait = util::lookup_symbol<decltype(sem_clockwait)>(handle, "sem_clockwait");
  pidfd_open = util::lookup_symbol<decltype(pidfd_open)>(handle, "pidfd_open");
  getdents64 = util::lookup_symbol<decltype(getdents64)>(handle, "getdents64");
  // Keep ready() false unless every required interposed libc entry point was
  // resolved. In release builds the asserts disappear, so the boolean must not
  // claim readiness while any later call would dereference a null function
  // pointer.
  initialized_ = openat && close && read && write && ioctl && mmap && munmap && mprotect &&
                 madvise && memfd_create && dup && dup2 && dup3 && fcntl && fopen && freopen &&
                 opendir && readdir && closedir && stat && lstat && access && fstat_fn &&
                 readlink_fn && realpath_fn && fork && clock_gettime && sem_init && sem_destroy &&
                 sem_post && pthread_mutex_lock;
  assert(initialized_);
}

LibcPassthrough &libc_passthrough() {
  static LibcPassthrough real;
  return real;
}

} // namespace rocjitsu
