/*
************************************************************************************************************************
*
*  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
*
***********************************************************************************************************************/

//
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//

#include "memfile.h"
#include <errno.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <vector>
#include <cstring>

#if defined(__GNUC__)
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sendfile.h>
#else
#include <io.h>
#if !defined(PROT_READ)
#define PROT_READ 0x0004   // FILE_MAP_READ
#endif
#if !defined(MAP_PRIVATE)
#define MAP_PRIVATE 0x0001 // FILE_MAP_COPY
#endif
#endif

// Allocation granularity
#define ALLOC_G          512
#define is_file(fd)      ((fd) >= 0)

#if defined(_WIN32)

#define OPEN             ::_open
#define READ(f, b, l)    ::_read((f), (b), (unsigned int)(l))
#define WRITE(f, b, l)   ::_write((f), (b), (unsigned int)(l))
#define CLOSE            ::_close
#define LSEEK            ::_lseek
#define FSTAT            ::fstat
#define FTRUNC(f, l)     ::_chsize((f), (long)(l))
#define MMAP             ::w32_mmap
#define MUNMAP           ::w32_munmap

#else

#define OPEN             ::open
#define READ(f, b, l)    ::read((f), (b), (size_t)(l))
#define WRITE            ::write
#define CLOSE            ::close
#define LSEEK            ::lseek
#define FSTAT            ::fstat
#define FTRUNC(f, l)     ::ftruncate((f), (off_t)(l))
#define MMAP             ::mmap
#define MUNMAP           ::munmap

#endif

#if defined(_WIN32)
extern "C" void* w32_mmap(void* start, size_t length, int prot, int flags, int fd, unsigned offset);
extern "C" int   w32_munmap(void* start, size_t length);
#endif

namespace amd {

// A structure which either maintains in memory file or uses a real file.
class memfile_t {
public:
  memfile_t() : buf(nullptr), curp(nullptr), size(0) {}

  bool reserve(size_t new_size) {
    if (!new_size)
      new_size = 1;
    new_size = (new_size + ALLOC_G - 1) & ~(ALLOC_G - 1);
    size_t pos = tell();
    void *p = realloc(buf, new_size);
    if (!p)
      return false;
    buf = p;
    setpos(pos);
    return true;
  }

  bool open(int oflag, int pmode)
  {
    size = 0;
    buf = curp = nullptr;
    return reserve(1);
  }

  off_t read(void *buffer, size_t count)
  {
    if (!buffer) {
      errno = EINVAL;
      return -1;
    }

    size_t pos = tell();
    if (pos >= size)
      return 0;

    size_t ret = size - pos;
    ret = std::min(ret, count);
    memcpy(buffer, curp, ret);
    advance(ret);
    return (off_t)ret;
  }

  off_t write(const void *buffer, size_t count)
  {
    if (!buffer) {
      errno = EINVAL;
      return -1;
    }

    size_t pos = tell();
    size_t new_size = std::max(pos + count, size);
    if (new_size > size) {
      if (!reserve(new_size))
        return -1;
      if (pos > size)
        memset((char*)buf + size, 0, pos - size);
      size = new_size;
    }

    memcpy(curp, buffer, count);
    advance(count);
    return (off_t)count;
  }

  int close() {
    if (is_open()) {
      free(buf);
      buf = nullptr;
      size = 0;
      return 0;
    }
    errno = EBADF;
    return -1;
  }

  off_t lseek(off_t offset, int origin)
  {
    switch (origin) {
    default:
      errno = EINVAL;
      return -1;
    case SEEK_SET:
      break;
    case SEEK_CUR:
      offset += (off_t)tell();
      break;
    case SEEK_END:
      offset += (off_t)size;
      break;
    }

    if (offset < 0) {
      errno = EOVERFLOW;
      return -1;
    }

    setpos((size_t)offset);
    return offset;
  }

  bool fstat(struct stat *buf) const
  {
    if (!is_open()) {
      errno = EBADF;
      return false;
    }

    memset(buf, 0, sizeof(struct stat));
    buf->st_mode = S_IFREG;
    buf->st_size = (off_t)size;
    return true;
  }

  bool ftruncate(size_t len)
  {
    if (len > size) {
      size_t pos = tell();
      lseek(0, SEEK_END);
      while(len--)
        write("", 1u);
      setpos(len);
    } else {
      reserve(len);
      size = len;
    }
    return true;
  }

  bool   is_open() const { return buf != nullptr; }
  size_t tell()    const { return size_t((char*)curp - (char*)buf); }
  void*  get()     const { return buf; }

protected:

  void setpos(size_t new_pos) { curp = (char*)buf + new_pos; }
  void advance(off_t offset)  { curp = (char*)curp + offset; }
  void advance(size_t offset) { curp = (char*)curp + offset; }

private:
  void*  buf;
  void*  curp;
  size_t size;
};

} // namespace amd

using namespace amd;

static std::vector<memfile_t> Files;

static size_t fd2idx(int fd)
{
  return (unsigned)-fd - 2;
}

static int idx2fd(size_t idx)
{
  return -(int)idx - 2;
}

static memfile_t* get_memfile(int fd)
{
  if (fd >= -1) {
    errno = EBADF;
    return nullptr;
  }

  size_t fno = fd2idx(fd);

  if (fno >= Files.size()) {
    errno = EBADF;
    return nullptr;
  }

  memfile_t &m = Files[fno];
  if (!m.is_open()) {
    errno = EBADF;
    return nullptr;
  }

  return &m;
}

// Acts the same as open(), but path can be NULL, which is a request for in memory file
int mem_open(const char *path, int oflag, int pmode)
{
  if (path && path[0]) // Filename provided, real file requested
    return OPEN(path, oflag, pmode);

  memfile_t m;
  if (!m.open(oflag, pmode))
    return -1;

  for (size_t i = 0; i < Files.size(); ++i) {
    if (!Files[i].is_open()) {
      Files[i] = m;
      return idx2fd(i);
    }
  }

  Files.push_back(m);
  return idx2fd(Files.size() - 1);
}

off_t mem_read(int fd, void *buffer, size_t count)
{
  if (is_file(fd))
    return READ(fd, buffer, count);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return -1;

  return m->read(buffer, count);
}

off_t mem_write(int fd, const void *buffer, size_t count)
{
  if (is_file(fd))
    return WRITE(fd, buffer, count);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return -1;

  return m->write(buffer, count);
}

int mem_close(int fd)
{
  if (is_file(fd))
    return CLOSE(fd);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return -1;

  int ret = m->close();

  if ((size_t)fd == (Files.size() - 1))
    Files.pop_back();

  return ret;
}

off_t mem_lseek(int fd, off_t offset, int origin)
{
  if (is_file(fd))
    return LSEEK(fd, offset, origin);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return -1;

  return m->lseek(offset, origin);
}

int mem_fstat(int fd, struct stat *buf)
{
  if (is_file(fd))
    return FSTAT(fd, buf);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return -1;

  return m->fstat(buf) ? 0 : -1;
}

int mem_ftruncate(int fd, size_t len)
{
  if (is_file(fd))
    return FTRUNC(fd, len);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return -1;

  return m->ftruncate(len) ? 0 : -1;
}

off_t mem_sendfile(int out_fd, int in_fd, off_t *offset, size_t count)
{
#if defined(__GNUC__)
  if (is_file(in_fd) && is_file(out_fd))
    return sendfile(out_fd, in_fd, offset, count);
#endif

  off_t start = offset ? *offset : mem_lseek(in_fd, 0, SEEK_CUR);
  struct stat sb;
  if (mem_fstat(in_fd, &sb) == -1)
    return -1;
  if (start < 0 || sb.st_size <= start)
    return 0;
  count = std::min(count, (size_t)(sb.st_size - start));
  void *in = mem_mmap(NULL, count, PROT_READ, MAP_PRIVATE, in_fd, start);
  if ((void*)-1 == in)
    return -1;

  off_t written = mem_write(out_fd, in, count);
  mem_munmap(in, count);
  if (written < 0)
    return -1;

  if (offset) {
    *offset += written;
  } else {
    if (mem_lseek(in_fd, written, SEEK_CUR) < 0)
      return -1;
  }

  return written;
}

void* mem_mmap(void* start, size_t length, int prot, int flags, int fd, unsigned offset)
{
  if (is_file(fd))
    return MMAP(start, length, prot, flags, fd, offset);

  memfile_t *m = get_memfile(fd);
  if (!m)
    return (void*)-1;

  return (char*)m->get()  + offset;
}

int mem_munmap(void* start, size_t length)
{
  MUNMAP(start, length);
  return 0;
}
