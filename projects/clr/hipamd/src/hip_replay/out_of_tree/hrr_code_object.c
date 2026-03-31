/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* Standalone ELF/msgpack parser for AMD GPU code object metadata.
 *
 * Parses:
 *   1. ELF64 headers to find .note sections
 *   2. NT_AMDGPU_METADATA note (type 32, name "AMDGPU")
 *   3. Msgpack map containing amdhsa.kernels[].args[].value_kind
 *
 * No external dependencies — pure C, no COMGR, no libelf. */

#include "hrr_code_object.h"
#include <string.h>
#include <stdio.h>

/* ---- ELF64 structures (minimal, avoiding elf.h dependency) ---- */

typedef struct {
  uint8_t  e_ident[16];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize;
  uint16_t e_phentsize;
  uint16_t e_phnum;
  uint16_t e_shentsize;
  uint16_t e_shnum;
  uint16_t e_shstrndx;
} Elf64_Ehdr;

typedef struct {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
} Elf64_Shdr;

typedef struct {
  uint32_t n_namesz;
  uint32_t n_descsz;
  uint32_t n_type;
  /* followed by name (padded to 4), then desc (padded to 4) */
} Elf64_Nhdr;

#define SHT_NOTE 7
#define NT_AMDGPU_METADATA 32

/* ---- Minimal msgpack decoder ---- */

/* Msgpack format bytes */
#define MP_FIXMAP_MIN   0x80
#define MP_FIXMAP_MAX   0x8f
#define MP_FIXARRAY_MIN 0x90
#define MP_FIXARRAY_MAX 0x9f
#define MP_FIXSTR_MIN   0xa0
#define MP_FIXSTR_MAX   0xbf
#define MP_NIL          0xc0
#define MP_FALSE        0xc2
#define MP_TRUE         0xc3
#define MP_UINT8        0xcc
#define MP_UINT16       0xcd
#define MP_UINT32       0xce
#define MP_UINT64       0xcf
#define MP_INT8         0xd0
#define MP_INT16        0xd1
#define MP_INT32        0xd2
#define MP_INT64        0xd3
#define MP_STR8         0xd9
#define MP_STR16        0xda
#define MP_STR32        0xdb
#define MP_ARRAY16      0xdc
#define MP_ARRAY32      0xdd
#define MP_MAP16        0xde
#define MP_MAP32        0xdf

typedef struct {
  const uint8_t* data;
  size_t pos;
  size_t len;
} mp_reader_t;

static int mp_eof(mp_reader_t* r) { return r->pos >= r->len; }
static uint8_t mp_peek(mp_reader_t* r) { return r->data[r->pos]; }
static uint8_t mp_read8(mp_reader_t* r) { return r->data[r->pos++]; }

static uint16_t mp_read16(mp_reader_t* r) {
  uint16_t v = ((uint16_t)r->data[r->pos] << 8) | r->data[r->pos + 1];
  r->pos += 2;
  return v;
}

static uint32_t mp_read32(mp_reader_t* r) {
  uint32_t v = ((uint32_t)r->data[r->pos] << 24) |
               ((uint32_t)r->data[r->pos + 1] << 16) |
               ((uint32_t)r->data[r->pos + 2] << 8) |
               r->data[r->pos + 3];
  r->pos += 4;
  return v;
}

static uint64_t mp_read_uint(mp_reader_t* r) {
  uint8_t tag = mp_read8(r);
  if (tag <= 0x7f) return tag;
  if (tag == MP_UINT8) return mp_read8(r);
  if (tag == MP_UINT16) return mp_read16(r);
  if (tag == MP_UINT32) return mp_read32(r);
  if (tag == MP_UINT64) {
    uint64_t hi = mp_read32(r);
    uint64_t lo = mp_read32(r);
    return (hi << 32) | lo;
  }
  if (tag == MP_INT8) return (int8_t)mp_read8(r);
  if (tag == MP_INT16) return (int16_t)mp_read16(r);
  if (tag == MP_INT32) return (int32_t)mp_read32(r);
  return 0;
}

/* Read a msgpack string into buf. Returns string length. */
static int mp_read_str(mp_reader_t* r, char* buf, size_t buf_size) {
  if (mp_eof(r)) return -1;
  uint8_t tag = mp_read8(r);
  uint32_t len = 0;

  if (tag >= MP_FIXSTR_MIN && tag <= MP_FIXSTR_MAX) {
    len = tag & 0x1f;
  } else if (tag == MP_STR8) {
    len = mp_read8(r);
  } else if (tag == MP_STR16) {
    len = mp_read16(r);
  } else if (tag == MP_STR32) {
    len = mp_read32(r);
  } else {
    return -1;
  }

  uint32_t copy = len < buf_size - 1 ? len : (uint32_t)(buf_size - 1);
  memcpy(buf, r->data + r->pos, copy);
  buf[copy] = '\0';
  r->pos += len;
  return (int)len;
}

/* Read map size */
static int mp_read_map_size(mp_reader_t* r) {
  if (mp_eof(r)) return -1;
  uint8_t tag = mp_read8(r);
  if (tag >= MP_FIXMAP_MIN && tag <= MP_FIXMAP_MAX) return tag & 0x0f;
  if (tag == MP_MAP16) return mp_read16(r);
  if (tag == MP_MAP32) return (int)mp_read32(r);
  return -1;
}

/* Read array size */
static int mp_read_array_size(mp_reader_t* r) {
  if (mp_eof(r)) return -1;
  uint8_t tag = mp_read8(r);
  if (tag >= MP_FIXARRAY_MIN && tag <= MP_FIXARRAY_MAX) return tag & 0x0f;
  if (tag == MP_ARRAY16) return mp_read16(r);
  if (tag == MP_ARRAY32) return (int)mp_read32(r);
  return -1;
}

/* Skip a msgpack value (recursive) */
static void mp_skip(mp_reader_t* r) {
  if (mp_eof(r)) return;
  uint8_t tag = mp_peek(r);

  /* Positive fixint */
  if (tag <= 0x7f) { r->pos++; return; }
  /* Negative fixint */
  if (tag >= 0xe0) { r->pos++; return; }
  /* Fixstr */
  if (tag >= MP_FIXSTR_MIN && tag <= MP_FIXSTR_MAX) {
    r->pos++;
    r->pos += tag & 0x1f;
    return;
  }
  /* Fixmap */
  if (tag >= MP_FIXMAP_MIN && tag <= MP_FIXMAP_MAX) {
    int n = mp_read_map_size(r);
    for (int i = 0; i < n * 2; i++) mp_skip(r);
    return;
  }
  /* Fixarray */
  if (tag >= MP_FIXARRAY_MIN && tag <= MP_FIXARRAY_MAX) {
    int n = mp_read_array_size(r);
    for (int i = 0; i < n; i++) mp_skip(r);
    return;
  }

  r->pos++;
  switch (tag) {
    case MP_NIL: case MP_FALSE: case MP_TRUE: break;
    case MP_UINT8: case MP_INT8: r->pos++; break;
    case MP_UINT16: case MP_INT16: r->pos += 2; break;
    case MP_UINT32: case MP_INT32: r->pos += 4; break;
    case MP_UINT64: case MP_INT64: r->pos += 8; break;
    case MP_STR8: { uint8_t n = r->data[r->pos++]; r->pos += n; break; }
    case MP_STR16: { uint16_t n = mp_read16(r); r->pos += n; break; }
    case MP_STR32: { uint32_t n = mp_read32(r); r->pos += n; break; }
    case MP_ARRAY16: { int n = mp_read16(r); for (int i=0;i<n;i++) mp_skip(r); break; }
    case MP_ARRAY32: { int n = (int)mp_read32(r); for (int i=0;i<n;i++) mp_skip(r); break; }
    case MP_MAP16: { int n = mp_read16(r); for (int i=0;i<n*2;i++) mp_skip(r); break; }
    case MP_MAP32: { int n = (int)mp_read32(r); for (int i=0;i<n*2;i++) mp_skip(r); break; }
    default: break;
  }
}

/* ---- Code object parser ---- */

static int parse_kernel_args(mp_reader_t* r, hrr_kernel_meta_t* km) {
  int nargs = mp_read_array_size(r);
  if (nargs < 0) return -1;

  km->num_args = 0;
  for (int i = 0; i < nargs && km->num_args < 64; i++) {
    int map_size = mp_read_map_size(r);
    if (map_size < 0) return -1;

    hrr_arg_desc_t arg = {HRR_ARG_VALUE, 0, 0};
    char value_kind[64] = "";

    for (int j = 0; j < map_size; j++) {
      char key[64];
      if (mp_read_str(r, key, sizeof(key)) < 0) { mp_skip(r); mp_skip(r); continue; }

      if (strcmp(key, ".value_kind") == 0) {
        mp_read_str(r, value_kind, sizeof(value_kind));
      } else if (strcmp(key, ".size") == 0) {
        arg.size = (uint16_t)mp_read_uint(r);
      } else if (strcmp(key, ".offset") == 0) {
        arg.offset = (uint16_t)mp_read_uint(r);
      } else {
        mp_skip(r);
      }
    }

    /* Classify argument kind */
    if (strncmp(value_kind, "hidden_", 7) == 0) {
      arg.kind = HRR_ARG_HIDDEN;
    } else if (strcmp(value_kind, "global_buffer") == 0) {
      arg.kind = HRR_ARG_GLOBAL_BUFFER;
    } else {
      arg.kind = HRR_ARG_VALUE;
    }

    km->args[km->num_args++] = arg;
  }
  return 0;
}

static int parse_kernels_array(mp_reader_t* r, hrr_kernel_meta_t* kernels,
                               int max_kernels) {
  int nkernels = mp_read_array_size(r);
  if (nkernels < 0) return 0;

  int count = 0;
  for (int i = 0; i < nkernels && count < max_kernels; i++) {
    int map_size = mp_read_map_size(r);
    if (map_size < 0) break;

    hrr_kernel_meta_t* km = &kernels[count];
    memset(km, 0, sizeof(*km));

    for (int j = 0; j < map_size; j++) {
      char key[64];
      if (mp_read_str(r, key, sizeof(key)) < 0) { mp_skip(r); mp_skip(r); continue; }

      if (strcmp(key, ".name") == 0) {
        mp_read_str(r, km->name, sizeof(km->name));
      } else if (strcmp(key, ".args") == 0) {
        parse_kernel_args(r, km);
      } else {
        mp_skip(r);
      }
    }

    if (km->name[0] != '\0') count++;
  }
  return count;
}

static int parse_metadata(const uint8_t* data, size_t len,
                          hrr_kernel_meta_t* kernels, int max_kernels) {
  mp_reader_t r = {data, 0, len};

  /* Top level is a map */
  int map_size = mp_read_map_size(&r);
  if (map_size < 0) return 0;

  for (int i = 0; i < map_size; i++) {
    char key[64];
    if (mp_read_str(&r, key, sizeof(key)) < 0) { mp_skip(&r); mp_skip(&r); continue; }

    if (strcmp(key, "amdhsa.kernels") == 0) {
      return parse_kernels_array(&r, kernels, max_kernels);
    } else {
      mp_skip(&r);
    }
  }
  return 0;
}

int hrr_parse_code_object(const void* image, size_t image_size,
                          hrr_kernel_meta_t* kernels, int max_kernels) {
  const uint8_t* base = (const uint8_t*)image;

  /* Validate ELF magic */
  if (image_size < sizeof(Elf64_Ehdr)) return 0;
  if (base[0] != 0x7f || base[1] != 'E' || base[2] != 'L' || base[3] != 'F')
    return 0;

  const Elf64_Ehdr* ehdr = (const Elf64_Ehdr*)base;

  /* Iterate section headers looking for SHT_NOTE */
  for (uint16_t i = 0; i < ehdr->e_shnum; i++) {
    size_t sh_off = ehdr->e_shoff + (size_t)i * ehdr->e_shentsize;
    if (sh_off + sizeof(Elf64_Shdr) > image_size) break;

    const Elf64_Shdr* shdr = (const Elf64_Shdr*)(base + sh_off);
    if (shdr->sh_type != SHT_NOTE) continue;
    if (shdr->sh_offset + shdr->sh_size > image_size) continue;

    /* Iterate notes in this section */
    const uint8_t* note_data = base + shdr->sh_offset;
    size_t note_end = (size_t)shdr->sh_size;
    size_t pos = 0;

    while (pos + sizeof(Elf64_Nhdr) <= note_end) {
      const Elf64_Nhdr* nhdr = (const Elf64_Nhdr*)(note_data + pos);
      size_t name_off = pos + sizeof(Elf64_Nhdr);
      size_t name_padded = (nhdr->n_namesz + 3) & ~3u;
      size_t desc_off = name_off + name_padded;
      size_t desc_padded = (nhdr->n_descsz + 3) & ~3u;

      if (desc_off + nhdr->n_descsz > note_end) break;

      if (nhdr->n_type == NT_AMDGPU_METADATA && nhdr->n_namesz >= 6 &&
          memcmp(note_data + name_off, "AMDGPU", 6) == 0) {
        /* Found metadata — parse msgpack */
        return parse_metadata(note_data + desc_off, nhdr->n_descsz,
                              kernels, max_kernels);
      }

      pos = desc_off + desc_padded;
    }
  }

  return 0;
}

const hrr_kernel_meta_t* hrr_find_kernel(const hrr_kernel_meta_t* kernels,
                                         int num_kernels, const char* name) {
  for (int i = 0; i < num_kernels; i++) {
    if (strcmp(kernels[i].name, name) == 0) return &kernels[i];
  }
  return NULL;
}
