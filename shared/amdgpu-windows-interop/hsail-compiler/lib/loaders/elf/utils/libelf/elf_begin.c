/*-
 * Copyright (c) 2006,2008-2011 Joseph Koshy
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
 */

#include <sys/cdefs.h>

#include <sys/types.h>
#if !defined(WIN32)
#include <sys/errno.h>
#include <sys/mman.h>
#else
#ifndef PROT_READ
#define PROT_READ FILE_MAP_READ
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE FILE_MAP_COPY
#endif
#ifndef MAP_FAILED
#define MAP_FAILED NULL
#endif
#include <Windows.h>
#endif
#include <sys/stat.h>

#include <ar.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <libelf.h>
#include <stdlib.h>
#include <stdio.h>
#if !defined(WIN32)
#include <unistd.h>
#else
#include "compat.h"
#endif

#include "_libelf.h"

LIBELF_VCSID("$Id: elf_begin.c 1923 2011-09-23 09:01:13Z jkoshy $");

#define	_LIBELF_INITSIZE	(64*1024)

/*
 * Read from a device file, pipe or socket.
 */
static void *
_libelf_read_special_file(int fd, size_t *fsz)
{
	ssize_t readsz;
	size_t bufsz, datasz;
	unsigned char *buf, *t;

	datasz = 0;
	readsz = 0;
	bufsz = _LIBELF_INITSIZE;
	if ((buf = malloc(bufsz)) == NULL)
		goto resourceerror;

	/*
	 * Read data from the file descriptor till we reach EOF, or
	 * till an error is encountered.
	 */
	do {
		/* Check if we need to expand the data buffer. */
		if (datasz == bufsz) {
			bufsz *= 2;
			if ((t = realloc(buf, bufsz)) == NULL)
				goto resourceerror;
			buf = t;
		}

		do {
			readsz = bufsz - datasz;
			t = buf + datasz;
			if ((readsz = read(fd, t, readsz)) <= 0)
				break;
			datasz += readsz;
		} while (datasz < bufsz);

	} while (readsz > 0);

	if (readsz < 0) {
		LIBELF_SET_ERROR(IO, errno);
		goto error;
	}

	assert(readsz == 0);

	/*
	 * Free up extra buffer space.
	 */
	if (bufsz > datasz) {
		if (datasz > 0) {
			if ((t = realloc(buf, datasz)) == NULL)
				goto resourceerror;
			buf = t;
		} else {	/* Zero bytes read. */
			LIBELF_SET_ERROR(ARGUMENT, 0);
			free(buf);
			buf = NULL;
		}
	}

	*fsz = datasz;
	return (buf);

resourceerror:
	LIBELF_SET_ERROR(RESOURCE, 0);
error:
	if (buf != NULL)
		free(buf);
	return (NULL);
}

static Elf *
_libelf_open_object(int fd, Elf_Cmd c, Elf_Mem *mem)
{
	Elf *e;
	void *m;
	mode_t mode;
	size_t fsize;
	struct stat sb;
	unsigned int flags;

	assert(c == ELF_C_READ || c == ELF_C_RDWR || c == ELF_C_WRITE);

	if (fstat(fd, &sb) < 0) {
		LIBELF_SET_ERROR(IO, errno);
		return (NULL);
	}

	mode = sb.st_mode;
	fsize = (size_t) sb.st_size;

	/*
	 * Reject unsupported file types.
	 */
	if (!S_ISREG(mode) && !S_ISCHR(mode)
#if !defined(WIN32)
      && !S_ISFIFO(mode) &&
	    !S_ISSOCK(mode)
#endif
      ) {
		LIBELF_SET_ERROR(ARGUMENT, 0);
		return (NULL);
	}

	/*
	 * For ELF_C_WRITE mode, allocate and return a descriptor.
     * For ELF_C_RDWR mode, if the file is empty, allocate and return.
	 */
	if (c == ELF_C_WRITE || (c == ELF_C_RDWR && !fsize)) {
		if ((e = _libelf_allocate_elf(mem)) != NULL) {
			_libelf_init_elf(e, ELF_K_ELF);
			e->e_byteorder = LIBELF_PRIVATE(byteorder);
			e->e_fd = fd;
			e->e_cmd = c;
			if (!S_ISREG(mode))
				e->e_flags |= LIBELF_F_SPECIAL_FILE;
		}

		return (e);
	}

	/*
	 * ELF_C_READ and ELF_C_RDWR mode.
	 */
	m = NULL;
	flags = 0;
	if (S_ISREG(mode)) {
		/*
		 * Always map regular files in with 'PROT_READ'
		 * permissions.
		 *
		 * For objects opened in ELF_C_RDWR mode, when
		 * elf_update(3) is called, we remove this mapping,
		 * write file data out using write(2), and map the new
		 * contents back.
		 */
		if ((m = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd,
	    (off_t) 0)) == MAP_FAILED) {
		LIBELF_SET_ERROR(IO, errno);
		return (NULL);
	    }
		flags = LIBELF_F_RAWFILE_MMAP;
#if 0
        m = mem.alloc(fsize);
        if (!fread(m, 1, fsize, _fdopen(fd, "w+b"))) {
            LIBELF_SET_ERROR(IO, errno);
            mem.dealloc(m);
            return(NULL);
        }
        flags = LIBELF_F_RAWFILE_MALLOC;
#endif
	} else if ((m = _libelf_read_special_file(fd, &fsize)) != NULL)
		flags = LIBELF_F_RAWFILE_MALLOC | LIBELF_F_SPECIAL_FILE;
	else
		return (NULL);

	if ((e = elf_memory(m, fsize, mem)) == NULL) {
		assert((flags & LIBELF_F_RAWFILE_MALLOC) ||
		    (flags & LIBELF_F_RAWFILE_MMAP));
		if (flags & LIBELF_F_RAWFILE_MMAP)
			(void) munmap(m, fsize);
		else
			e->e_mem.dealloc(m);
		return (NULL);
	}

	/* ar(1) archives aren't supported in RDWR mode. */
	if (c == ELF_C_RDWR && e->e_kind == ELF_K_AR) {
		(void) elf_end(e);
		LIBELF_SET_ERROR(ARGUMENT, 0);
		return (NULL);
	}

	e->e_flags |= flags;
	e->e_fd = fd;
	e->e_cmd = c;

	return (e);
}

Elf *
elf_begin(int fd, Elf_Cmd c, Elf *a, Elf_Mem *mem)
{
	Elf *e;

	e = NULL;

	if (LIBELF_PRIVATE(version) == EV_NONE) {
		LIBELF_SET_ERROR(SEQUENCE, 0);
		return (NULL);
	}

	switch (c) {
	case ELF_C_NULL:
		return (NULL);

	case ELF_C_WRITE:
		/*
		 * The ELF_C_WRITE command is required to ignore the
		 * descriptor passed in.
		 */
		a = NULL;
		break;

	case ELF_C_RDWR:
		if (a != NULL && a->e_kind == ELF_K_AR) { /* not allowed for ar(1) archives. */
			LIBELF_SET_ERROR(ARGUMENT, 0);
			return (NULL);
		}
		/*FALLTHROUGH*/
	case ELF_C_READ:
		/*
		 * Descriptor `a' could be for a regular ELF file, or
		 * for an ar(1) archive.  If descriptor `a' was opened
		 * using a valid file descriptor, we need to check if
		 * the passed in `fd' value matches the original one.
		 */
		if (a &&
		    ((a->e_fd != -1 && a->e_fd != fd) || c != a->e_cmd)) {
			LIBELF_SET_ERROR(ARGUMENT, 0);
			return (NULL);
		}
		break;

	default:
		LIBELF_SET_ERROR(ARGUMENT, 0);
		return (NULL);

	}

	if (a == NULL)
		e = _libelf_open_object(fd, c, mem);
	else if (a->e_kind == ELF_K_AR)
		e = _libelf_ar_open_member(a->e_fd, c, a, mem);
	else
		(e = a)->e_activations++;

	return (e);
}
#if defined(WIN32)
// This code taken from:
// http://git.661346.n2.nabble.com/PATCH-mmap-implementation-for-mingw-td1560056.html
// This code is in public domain according to the FAQ here:
// http://www.mingw.org/wiki/FAQ
// http://www.mingw.org/license
// FIXME: This needs to be more robust to the protection and flag options.
void *w32_mmap(void *start, size_t length, int prot, int flags, int fd,
unsigned offset)
{
	HANDLE handle;

	if (start != NULL || !(flags & MAP_PRIVATE))
	 assert(!"Invalid usage of mingw_mmap");

	handle = CreateFileMapping((HANDLE)_get_osfhandle(fd), NULL, PAGE_READONLY, 0, 0, NULL);
	if (handle != NULL) {
	 start = MapViewOfFile(handle, flags, 0, offset,
length);
	 CloseHandle(handle);
    }
	return start;
}

int w32_munmap(void *start, size_t length) {
  UnmapViewOfFile(start);
  return 0;
}
#endif
