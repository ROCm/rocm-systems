/*-
 * Copyright (c) 2006,2008,2010 Joseph Koshy
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

/*
 * Internal APIs
 */

#include <sys/cdefs.h>

#if !defined(WIN32)
#include <sys/errno.h>
#endif

#include <assert.h>
#include <errno.h>
#include <libelf.h>
#include <stdlib.h>
#include <string.h>

#include "_libelf.h"

LIBELF_VCSID("$Id: libelf_allocate.c 1341 2011-01-01 04:28:29Z jkoshy $");

Elf *
_libelf_allocate_elf(Elf_Mem *mem)
{
	Elf *e;

  if (mem != NULL) {
    if ((e = mem->alloc(sizeof(*e))) == NULL) {
      LIBELF_SET_ERROR(RESOURCE, errno);
      return NULL;
    }
    e->e_mem.dealloc = mem->dealloc;
    e->e_mem.alloc   = mem->alloc;
  } else {
    if ((e = malloc(sizeof(*e))) == NULL) {
      LIBELF_SET_ERROR(RESOURCE, errno);
      return NULL;
    }
    e->e_mem.dealloc = &free;
    e->e_mem.alloc   = &malloc;
  }

	e->e_activations = 1;
	e->e_hdr.e_rawhdr = NULL;
	e->e_byteorder   = ELFDATANONE;
	e->e_class       = ELFCLASSNONE;
	e->e_cmd         = ELF_C_NULL;
	e->e_fd          = -1;
	e->e_flags	     = 0;
	e->e_kind        = ELF_K_NONE;
	e->e_parent      = NULL;
	e->e_rawfile     = NULL;
	e->e_rawsize     = 0;
	e->e_version     = LIBELF_PRIVATE(version);

	(void) memset(&e->e_u, 0, sizeof(e->e_u));

	return (e);
}

void
_libelf_init_elf(Elf *e, Elf_Kind kind)
{
	assert(e != NULL);
	assert(e->e_kind == ELF_K_NONE);

	e->e_kind = kind;

	switch (kind) {
	case ELF_K_ELF:
		STAILQ_INIT(&e->e_u.e_elf.e_scn);
		break;
	default:
		break;
	}
}

#define	FREE(E, P)		do {				\
		if (P)					\
			E->e_mem.dealloc(P);			\
	} while (0)

Elf *
_libelf_release_elf(Elf *e)
{
	Elf_Arhdr *arh;

	switch (e->e_kind) {
	case ELF_K_AR:
		FREE(e, e->e_u.e_ar.e_symtab);
		break;

	case ELF_K_ELF:
		switch (e->e_class) {
		case ELFCLASS32:
			FREE(e, e->e_u.e_elf.e_ehdr.e_ehdr32);
			FREE(e, e->e_u.e_elf.e_phdr.e_phdr32);
			break;
		case ELFCLASS64:
			FREE(e, e->e_u.e_elf.e_ehdr.e_ehdr64);
			FREE(e, e->e_u.e_elf.e_phdr.e_phdr64);
			break;
		}

		assert(STAILQ_EMPTY(&e->e_u.e_elf.e_scn));

		if (e->e_flags & LIBELF_F_AR_HEADER) {
			arh = e->e_hdr.e_arhdr;
			FREE(e, arh->ar_name);
			FREE(e, arh->ar_rawname);
			e->e_mem.dealloc(arh);
		}

		break;

	default:
		break;
	}

	e->e_mem.dealloc(e);

	return (NULL);
}

Elf_Data *
_libelf_allocate_data(Elf_Scn *s)
{
	Elf_Data *d;

	if ((d = s->s_elf->e_mem.alloc(sizeof(Elf_Data))) == NULL) {
		LIBELF_SET_ERROR(RESOURCE, 0);
		return (NULL);
	}
    memset(d, 0, sizeof(*d));

	d->d_scn = s;

	return (d);
}

Elf_Data *
_libelf_release_data(Elf_Data *d)
{

	if (d->d_flags & LIBELF_F_DATA_MALLOCED)
		d->d_scn->s_elf->e_mem.dealloc(d->d_buf);

	d->d_scn->s_elf->e_mem.dealloc(d);

	return (NULL);
}

Elf_Scn *
_libelf_allocate_scn(Elf *e, size_t ndx)
{
	Elf_Scn *s;

	if ((s = e->e_mem.alloc(sizeof(Elf_Scn))) == NULL) {
		LIBELF_SET_ERROR(RESOURCE, errno);
		return (NULL);
	}
    memset(s, 0, sizeof(*s));

	s->s_elf = e;
	s->s_ndx = ndx;

	STAILQ_INIT(&s->s_data);
	STAILQ_INIT(&s->s_rawdata);

	STAILQ_INSERT_TAIL(&e->e_u.e_elf.e_scn, s, s_next);

	return (s);
}

Elf_Scn *
_libelf_release_scn(Elf_Scn *s)
{
	Elf *e;
	Elf_Data *d, *td;

	assert(s != NULL);

	STAILQ_FOREACH_SAFE(d, &s->s_data, d_next, td) {
		STAILQ_REMOVE(&s->s_data, d, _Elf_Data, d_next);
		d = _libelf_release_data(d);
	}

	STAILQ_FOREACH_SAFE(d, &s->s_rawdata, d_next, td) {
		assert((d->d_flags & LIBELF_F_DATA_MALLOCED) == 0);
		STAILQ_REMOVE(&s->s_rawdata, d, _Elf_Data, d_next);
		d = _libelf_release_data(d);
	}

	e = s->s_elf;

	assert(e != NULL);

	STAILQ_REMOVE(&e->e_u.e_elf.e_scn, s, _Elf_Scn, s_next);

	e->e_mem.dealloc(s);

	return (NULL);
}
