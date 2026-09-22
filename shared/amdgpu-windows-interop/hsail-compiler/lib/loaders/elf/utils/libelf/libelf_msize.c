/*-
 * Copyright (c) 2006,2008 Joseph Koshy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS `AS IS' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <sys/cdefs.h>

#include <assert.h>
#include <libelf.h>
#include <string.h>

#include "_libelf.h"

LIBELF_VCSID("$Id: libelf_msize.m4 311 2009-02-26 16:46:31Z jkoshy $");

/* WARNING: GENERATED FROM libelf_msize.m4. */

struct msize {
	size_t	msz32;
	size_t	msz64;
};

static struct msize msize[ELF_T_NUM] = {
#if defined(__GNUC__)
#if	LIBELF_CONFIG_ADDR
    [ELF_T_ADDR] = { .msz32 = sizeof(Elf32_Addr), .msz64 = sizeof(Elf64_Addr) },
#endif
#if	LIBELF_CONFIG_BYTE
    [ELF_T_BYTE] = { .msz32 = 1, .msz64 = 1 },
#endif
#if	LIBELF_CONFIG_CAP
    [ELF_T_CAP] = { .msz32 = sizeof(Elf32_Cap), .msz64 = sizeof(Elf64_Cap) },
#endif
#if	LIBELF_CONFIG_DYN
    [ELF_T_DYN] = { .msz32 = sizeof(Elf32_Dyn), .msz64 = sizeof(Elf64_Dyn) },
#endif
#if	LIBELF_CONFIG_EHDR
    [ELF_T_EHDR] = { .msz32 = sizeof(Elf32_Ehdr), .msz64 = sizeof(Elf64_Ehdr) },
#endif
#if	LIBELF_CONFIG_GNUHASH
    [ELF_T_GNUHASH] = { .msz32 = 1, .msz64 = 1 },
#endif
#if	LIBELF_CONFIG_HALF
    [ELF_T_HALF] = { .msz32 = sizeof(Elf32_Half), .msz64 = sizeof(Elf64_Half) },
#endif
#if	LIBELF_CONFIG_LWORD
    [ELF_T_LWORD] = { .msz32 = sizeof(Elf32_Lword), .msz64 = sizeof(Elf64_Lword) },
#endif
#if	LIBELF_CONFIG_MOVE
    [ELF_T_MOVE] = { .msz32 = sizeof(Elf32_Move), .msz64 = sizeof(Elf64_Move) },
#endif
#if	LIBELF_CONFIG_MOVEP
    [ELF_T_MOVEP] = { .msz32 = 0, .msz64 = 0 },
#endif
#if	LIBELF_CONFIG_NOTE
    [ELF_T_NOTE] = { .msz32 = 1, .msz64 = 1 },
#endif
#if	LIBELF_CONFIG_OFF
    [ELF_T_OFF] = { .msz32 = sizeof(Elf32_Off), .msz64 = sizeof(Elf64_Off) },
#endif
#if	LIBELF_CONFIG_PHDR
    [ELF_T_PHDR] = { .msz32 = sizeof(Elf32_Phdr), .msz64 = sizeof(Elf64_Phdr) },
#endif
#if	LIBELF_CONFIG_REL
    [ELF_T_REL] = { .msz32 = sizeof(Elf32_Rel), .msz64 = sizeof(Elf64_Rel) },
#endif
#if	LIBELF_CONFIG_RELA
    [ELF_T_RELA] = { .msz32 = sizeof(Elf32_Rela), .msz64 = sizeof(Elf64_Rela) },
#endif
#if	LIBELF_CONFIG_SHDR
    [ELF_T_SHDR] = { .msz32 = sizeof(Elf32_Shdr), .msz64 = sizeof(Elf64_Shdr) },
#endif
#if	LIBELF_CONFIG_SWORD
    [ELF_T_SWORD] = { .msz32 = sizeof(Elf32_Sword), .msz64 = sizeof(Elf64_Sword) },
#endif
#if	LIBELF_CONFIG_SXWORD
    [ELF_T_SXWORD] = { .msz32 = 0, .msz64 = sizeof(Elf64_Sxword) },
#endif
#if	LIBELF_CONFIG_SYMINFO
    [ELF_T_SYMINFO] = { .msz32 = sizeof(Elf32_Syminfo), .msz64 = sizeof(Elf64_Syminfo) },
#endif
#if	LIBELF_CONFIG_SYM
    [ELF_T_SYM] = { .msz32 = sizeof(Elf32_Sym), .msz64 = sizeof(Elf64_Sym) },
#endif
#if	LIBELF_CONFIG_VDEF
    [ELF_T_VDEF] = { .msz32 = sizeof(Elf32_Verdef), .msz64 = sizeof(Elf64_Verdef) },
#endif
#if	LIBELF_CONFIG_VNEED
    [ELF_T_VNEED] = { .msz32 = sizeof(Elf32_Verneed), .msz64 = sizeof(Elf64_Verneed) },
#endif
#if	LIBELF_CONFIG_WORD
    [ELF_T_WORD] = { .msz32 = sizeof(Elf32_Word), .msz64 = sizeof(Elf64_Word) },
#endif
#if	LIBELF_CONFIG_XWORD
    [ELF_T_XWORD] = { .msz32 = 0, .msz64 = sizeof(Elf64_Xword) },
#endif
#elif defined(_MSC_VER)
   {4, 8}, {1, 1}, {0, 0}, {8, 16}, {52, 64},
   {2, 2}, {0, 0}, {0, 0}, {0, 0}, {1, 1},
   {4, 8}, {32, 56}, {8, 16}, {12, 24}, {40, 64},
   {4, 4}, {0, 8}, {0, 0}, {16, 24}, {20, 20},
   {16, 16}, {4, 4}, {0, 8}, {1, 1}
#else
#error
#endif
};

size_t
_libelf_msize(Elf_Type t, int elfclass, unsigned int version)
{
	size_t sz;

	assert(elfclass == ELFCLASS32 || elfclass == ELFCLASS64);
	assert((signed) t >= ELF_T_FIRST && t <= ELF_T_LAST);

	if (version != EV_CURRENT) {
		LIBELF_SET_ERROR(VERSION, 0);
		return (0);
	}

	sz = (elfclass == ELFCLASS32) ? msize[t].msz32 : msize[t].msz64;

	return (sz);
}
