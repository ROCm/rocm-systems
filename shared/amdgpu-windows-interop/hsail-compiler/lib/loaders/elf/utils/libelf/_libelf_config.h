/*-
 * Copyright (c) 2008-2011 Joseph Koshy
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
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
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
 *
 * $Id: _libelf_config.h 2032 2011-10-23 09:07:00Z jkoshy $
 */

#ifdef __FreeBSD__

#define	LIBELF_VCSID(ID)	__FBSDID(ID)

/*
 * Define LIBELF_{ARCH,BYTEORDER,CLASS} based on the machine architecture.
 * See also: <machine/elf.h>.
 */

#if	defined(__amd64__)

#define	LIBELF_ARCH		EM_X86_64
#define	LIBELF_BYTEORDER	ELFDATA2LSB
#define	LIBELF_CLASS		ELFCLASS64

#elif	defined(__arm__)

#define	LIBELF_ARCH		EM_ARM
#if	defined(__ARMEB__)	/* Big-endian ARM. */
#define	LIBELF_BYTEORDER	ELFDATA2MSB
#else
#define	LIBELF_BYTEORDER	ELFDATA2LSB
#endif
#define	LIBELF_CLASS		ELFCLASS32

#elif	defined(__i386__)

#define	LIBELF_ARCH		EM_386
#define	LIBELF_BYTEORDER	ELFDATA2LSB
#define	LIBELF_CLASS		ELFCLASS32

#elif	defined(__ia64__)

#define	LIBELF_ARCH		EM_IA_64
#define	LIBELF_BYTEORDER	ELFDATA2LSB
#define	LIBELF_CLASS		ELFCLASS64

#elif	defined(__mips__)

#define	LIBELF_ARCH		EM_MIPS
#if	defined(__MIPSEB__)
#define	LIBELF_BYTEORDER	ELFDATA2MSB
#else
#define	LIBELF_BYTEORDER	ELFDATA2LSB
#endif
#define	LIBELF_CLASS		ELFCLASS32

#elif	defined(__powerpc__)

#define	LIBELF_ARCH		EM_PPC
#define	LIBELF_BYTEORDER	ELFDATA2MSB
#define	LIBELF_CLASS		ELFCLASS32

#elif	defined(__sparc__)

#define	LIBELF_ARCH		EM_SPARCV9
#define	LIBELF_BYTEORDER	ELFDATA2MSB
#define	LIBELF_CLASS		ELFCLASS64

#else
#error	Unknown FreeBSD architecture.
#endif
#endif  /* __FreeBSD__ */

#ifdef __NetBSD__

#include <machine/elf_machdep.h>

#define	LIBELF_VCSID(ID)	__RCSID(ID)

#if	!defined(ARCH_ELFSIZE)
#error	ARCH_ELFSIZE is not defined.
#endif

#if	ARCH_ELFSIZE == 32
#define	LIBELF_ARCH		ELF32_MACHDEP_ID
#define	LIBELF_BYTEORDER	ELF32_MACHDEP_ENDIANNESS
#define	LIBELF_CLASS		ELFCLASS32
#define	Elf_Note		Elf32_Nhdr
#else
#define	LIBELF_ARCH		ELF64_MACHDEP_ID
#define	LIBELF_BYTEORDER	ELF64_MACHDEP_ENDIANNESS
#define	LIBELF_CLASS		ELFCLASS64
#define	Elf_Note		Elf64_Nhdr
#endif

#endif	/* __NetBSD__ */

/*
 * GNU & Linux compatibility.
 *
 * `__linux__' is defined in an environment runs the Linux kernel and glibc.
 * `__GNU__' is defined in an environment runs a GNU kernel (Hurd) and glibc.
 * `__GLIBC__' is defined for an environment that runs glibc over a non-GNU
 *     kernel such as GNU/kFreeBSD.
 */

#if defined(__linux__) || defined(__GNU__) || defined(__GLIBC__)

#if defined(__linux__)

#include "native-elf-format.h"

#define	LIBELF_CLASS		ELFTC_CLASS
#define	LIBELF_ARCH		ELFTC_ARCH
#define	LIBELF_BYTEORDER	ELFTC_BYTEORDER

#endif	/* defined(__linux__) */

#define	LIBELF_VCSID(ID)

#if	LIBELF_CLASS == ELFCLASS32
#define	Elf_Note		Elf32_Nhdr
#elif   LIBELF_CLASS == ELFCLASS64
#define	Elf_Note		Elf64_Nhdr
#else
#error  LIBELF_CLASS needs to be one of ELFCLASS32 or ELFCLASS64
#endif

#define	roundup2	roundup

#endif /* defined(__linux__) || defined(__GNU__) || defined(__GLIBC__) */
/*
 * Common configuration for the GNU environment.
 */

#define	LIBELF_CONFIG_ADDR	1
#define	LIBELF_CONFIG_BYTE	1
#define	LIBELF_CONFIG_DYN	1
#define	LIBELF_CONFIG_EHDR	1
#define	LIBELF_CONFIG_HALF	1
#define	LIBELF_CONFIG_MOVEP	1
#define	LIBELF_CONFIG_NOTE	1
#define	LIBELF_CONFIG_OFF	1
#define	LIBELF_CONFIG_PHDR	1
#define	LIBELF_CONFIG_REL	1
#define	LIBELF_CONFIG_RELA	1
#define	LIBELF_CONFIG_SHDR	1
#define	LIBELF_CONFIG_SWORD	1
#define	LIBELF_CONFIG_SXWORD	1
#define	LIBELF_CONFIG_SYM	1
#define	LIBELF_CONFIG_VDEF	1
#define	LIBELF_CONFIG_VNEED	1
#define	LIBELF_CONFIG_WORD	1
#define	LIBELF_CONFIG_XWORD	1

#if defined(WIN32)

#include "native-elf-format.h"

#define	LIBELF_CLASS		ELFTC_CLASS
#define	LIBELF_ARCH		ELFTC_ARCH
#define	LIBELF_BYTEORDER	ELFTC_BYTEORDER

#define	LIBELF_CONFIG_ADDR	1
#define	LIBELF_CONFIG_BYTE	1
#define	LIBELF_CONFIG_DYN	1
#define	LIBELF_CONFIG_EHDR	1
#define	LIBELF_CONFIG_HALF	1
#define	LIBELF_CONFIG_MOVEP	1
#define	LIBELF_CONFIG_OFF	1
#define	LIBELF_CONFIG_PHDR	1
#define	LIBELF_CONFIG_REL	1
#define	LIBELF_CONFIG_RELA	1
#define	LIBELF_CONFIG_SHDR	1
#define	LIBELF_CONFIG_SWORD	1
#define	LIBELF_CONFIG_SXWORD	1
#define	LIBELF_CONFIG_SYM	1
#define	LIBELF_CONFIG_WORD	1
#define	LIBELF_CONFIG_XWORD	1

#define	LIBELF_VCSID(ID)

#define	roundup2	roundup
#endif // defined(WIN32)
#ifndef	LIBELF_CONFIG_GNUHASH
#define	LIBELF_CONFIG_GNUHASH	1

/*
 * The header for GNU-style hash sections.
 */

typedef struct {
	u_int32_t	gh_nbuckets;	/* Number of hash buckets. */
	u_int32_t	gh_symndx;	/* First visible symbol in .dynsym. */
	u_int32_t	gh_maskwords;	/* #maskwords used in bloom filter. */
	u_int32_t	gh_shift2;	/* Bloom filter shift count. */
} Elf_GNU_Hash_Header;
#endif

#if defined(USE_MEMFILE)
#include "memfile.h"

#if !defined(read)
#define read(f, b, l)     mem_read((f), (b), (l))
#endif

#if !defined(write)
#define write(f, b, l)    mem_write((f), (b), (l))
#endif

#if !defined(lseek)
#define lseek(f, l, w)    mem_lseek((f), (l), (w))
#endif

#if !defined(fstat)
#define fstat(f, b)       mem_fstat((f), (struct stat*)(b))
#endif

#if !defined(_fstat64i32)
#define _fstat64i32(f, b) mem_fstat((f), (struct stat*)(b))
#endif

#if !defined(_fstat32i64)
#define _fstat32i64(f, b) mem_fstat((f), (struct stat*)(b))
#endif

#if !defined(_fstat32)
#define _fstat32(f, b)    mem_fstat((f), (struct stat*)(b))
#endif

#if !defined(_fstat64)
#define _fstat64(f, b)    mem_fstat((f), (struct stat*)(b))
#endif

#if !defined(ftruncate)
#define ftruncate(f, l)   mem_ftruncate((f), (size_t)(l))
#endif

#if !defined(_chsize)
#define _chsize(f, l)     mem_ftruncate((f), (size_t)(l))
#endif

#if !defined(mmap)
#define mmap              mem_mmap
#endif

#if !defined(mem_munmap)
#define munmap            mem_munmap
#endif

#else // !USE_MEMFILE

#if !defined(mmap)
#if defined(WIN32)
#define mmap              w32_mmap
#else
#define mmap              mmap
#endif
#endif

#if !defined(mem_munmap)
#if defined(WIN32)
#define munmap            w32_munmap
#else
#define munmap            munmap
#endif
#endif

#endif //USE_MEMFILE
