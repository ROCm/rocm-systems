#ifndef __BFD_BACKTRACE__
#define __BFD_BACKTRACE__

/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Modification Copyright (C) Advanced Micro Devices, Inc, 2022. ALL RIGHTS RESERVED
 *
 * This code is based on the UCX library's mechanism to extract the call stack
 * using the BFD library (ucx/src/ucs/debug/debug.c).
 */

#include <dirent.h>
#include <link.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <bfd.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#ifdef HAVE_CPLUS_DEMANGLE
#define HAVE_DECL_BASENAME 1
#include <demangle.h>
#endif

struct dl_address_search {
    unsigned long            address;
    const char               *filename;
    unsigned long            base;
};

struct backtrace_file {
    struct dl_address_search dl;
    bfd                      *abfd;
    asymbol                  **syms;
};

struct backtrace_line {
    unsigned long            address;
    char                     *file;
    char                     *function;
    unsigned                 lineno;
};

#define BACKTRACE_MAX            64

struct backtrace {
    struct backtrace_line    lines[BACKTRACE_MAX];
    int                      size;
    int                      position;
};
typedef struct backtrace backtrace_h;

struct backtrace_search {
    int                      count;
    struct backtrace_file    *file;
    int                      backoff; /* search the line where the function call
                                         took place, instead of return address */
    struct backtrace_line    *lines;
    int                      max_lines;
};

static const char *get_exe()
{
    static char exe[1024];
    int ret;

    ret = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (ret < 0) {
        exe[0] = '\0';
    } else {
        exe[ret] = '\0';
    }

    return exe;
}

static int dl_match_address(struct dl_phdr_info *info, size_t size, void *data)
{
    struct dl_address_search *dl = (struct dl_address_search *) data;
    const ElfW(Phdr) *phdr;
    ElfW(Addr) load_base = info->dlpi_addr;
    long n;

    phdr = info->dlpi_phdr;
    for (n = info->dlpi_phnum; --n >= 0; phdr++) {
        if (phdr->p_type == PT_LOAD) {
            ElfW(Addr) vbaseaddr = phdr->p_vaddr + load_base;
            if (dl->address >= vbaseaddr && dl->address < vbaseaddr + phdr->p_memsz) {
                dl->filename = info->dlpi_name;
                dl->base     = info->dlpi_addr;
            }
        }
    }
    return 0;
}

static int dl_lookup_address(struct dl_address_search *dl)
{
    dl->filename = NULL;
    dl->base     = 0;

    dl_iterate_phdr(dl_match_address, dl);
    if (dl->filename == NULL) {
        return 0;
    }
    if (strlen(dl->filename) == 0) {
        dl->filename = get_exe();
    }
    return 1;
}

static int load_file(struct backtrace_file *file)
{
    long symcount;
    unsigned int size;
    char **matching;

    file->syms = NULL;
    file->abfd = bfd_openr(file->dl.filename, NULL);
    if (!file->abfd) {
        goto err;
    }

    if (bfd_check_format(file->abfd, bfd_archive)) {
        goto err_close;
    }

    if (!bfd_check_format_matches(file->abfd, bfd_object, &matching)) {
        goto err_close;
    }
    if ((bfd_get_file_flags(file->abfd) & HAS_SYMS) == 0) {
        goto err_close;
    }

    symcount = bfd_read_minisymbols(file->abfd, 0, (void**)&file->syms, &size);
    if (symcount == 0) {
        free(file->syms);
        symcount = bfd_read_minisymbols(file->abfd, 1, (void**)&file->syms, &size);
    }
    if (symcount < 0) {
        goto err_close;
    }

    return 1;

err_close:
    bfd_close(file->abfd);
err:
    return 0;
}

static void unload_file(struct backtrace_file *file)
{
    free(file->syms);
    bfd_close(file->abfd);
}

static void find_address_in_section(bfd *abfd, asection *section, void *data)
{
    struct backtrace_search *search = (backtrace_search *)data;
    bfd_size_type size;
    bfd_vma vma;
    unsigned long address;
    const char *filename, *function;
    unsigned lineno;
    int found;

    if ((search->count > 0) || (search->max_lines == 0) ||
#ifdef HAVE_DECL_BFD_GET_SECTION_FLAGS
        ((bfd_get_section_flags(abfd, section) & SEC_ALLOC) == 0)) {
#else
        ((bfd_section_flags(section) & SEC_ALLOC) == 0)) {
#endif
        return;
    }

    address = search->file->dl.address - search->file->dl.base;
#ifdef HAVE_DECL_BFD_GET_SECTION_VMA
    vma = bfd_get_section_vma(abfd, section);
#else
    vma = bfd_section_vma(section);
#endif

    if (address < vma) {
        return;
    }
#ifdef HAVE_TWO_ARG_BFD_SECTION_SIZE
    size = bfd_section_size(abfd, section);
#else
    size = bfd_section_size(section);
#endif
    if (address >= vma + size) {
        return;
    }

    /* Search in address-1 to get the calling line instead of return address */
    found = bfd_find_nearest_line(abfd, section, search->file->syms,
                                  address - vma - search->backoff,
                                  &filename, &function, &lineno);
   do {
        search->lines[search->count].address  = address;
        search->lines[search->count].file     = strdup(filename ? filename :
                                                       "UNKNOWN_FILE");
	search->lines[search->count].function = function ?
#ifdef HAVE_CPLUS_DEMANGLE
	  cplus_demangle(function, 0) : strdup("UNKNOWN_FUNCTION");
#else
	  strdup(function) : strdup("UNKNOWN_FUNCTION");
#endif
        search->lines[search->count].lineno   = lineno;
        if (search->count == 0) {
            /* To get the inliner info, search at the original address */
            bfd_find_nearest_line(abfd, section, search->file->syms, address - vma,
                                  &filename, &function, &lineno);
        }

        ++search->count;
        found = bfd_find_inliner_info(abfd, &filename, &function, &lineno);
    } while (found && (search->count < search->max_lines));
}


static int get_line_info(struct backtrace_file *file, int backoff,
                         struct backtrace_line *lines, int max)
{
    struct backtrace_search search;

    search.file      = file;
    search.backoff   = backoff;
    search.count     = 0;
    search.lines     = lines;
    search.max_lines = max;
    bfd_map_over_sections(file->abfd, find_address_in_section, &search);
    return search.count;
}

#endif
