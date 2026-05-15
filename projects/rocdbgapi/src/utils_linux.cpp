/* Copyright (c) 2019-2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "utils.h"

#include "logging.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace amd::dbgapi
{

namespace utils
{

const char *
get_self_name ()
{
  Dl_info dl_info{};
  if (!dladdr (static_cast<void *> (&amd::dbgapi::log_level), &dl_info))
    return "";
  return dl_info.dli_fname;
}

} /* namespace amd::dbgapi::utils.  */

class pipe_t
{
private:
  std::optional<std::array<file_desc_t, 2>> m_pipe_fd{};

public:
  pipe_t () = default;
  ~pipe_t () { close (); }

  bool open ();
  void close ();

  /* Return true if the pipe is valid and ready for use.  */
  bool is_valid () const { return m_pipe_fd.has_value (); }

  /* Return the read-end file descriptor of the pipe.  */
  file_desc_t read_fd () const
  {
    dbgapi_assert (is_valid () && "this pipe is not valid");
    return (*m_pipe_fd)[0];
  }

  /* Return the write-end file descriptor of the pipe.  */
  file_desc_t write_fd () const
  {
    dbgapi_assert (is_valid () && "this pipe is not valid");
    return (*m_pipe_fd)[1];
  }

  /* Write a single char, '+', to the pipe.  Return 0 if successful, errno
     otherwise.  */
  int mark ();

  /* Consume all the data in the pipe.  Return 0 if successful, errno
     otherwise.  */
  int flush ();
};

bool
pipe_t::open ()
{
  std::array<file_desc_t, 2> pipe;
  if (::pipe2 (pipe.data (), O_CLOEXEC))
    {
      warning ("pipe_t::open: pipe2 failed: %s", strerror (errno));
      return false;
    }

  m_pipe_fd.emplace (pipe);

  if (::fcntl (read_fd (), F_SETFL, O_NONBLOCK)
      || ::fcntl (write_fd (), F_SETFL, O_NONBLOCK))
    {
      warning ("pipe_t::open: fcntl failed: %s", strerror (errno));
      close ();
      return false;
    }

  return true;
}

void
pipe_t::close ()
{
  if (is_valid ())
    {
      ::close (read_fd ());
      ::close (write_fd ());
    }

  m_pipe_fd.reset ();
}

int
pipe_t::flush ()
{
  int ret;

  do
    {
      char buf;
      ret = ::read (read_fd (), &buf, 1);
    }
  while (ret >= 0 || (ret == -1 && errno == EINTR));

  if (ret == -1 && errno != EAGAIN)
    fatal_error ("read: %s", strerror (errno));

  return ret == -1 ? -errno : 0;
}

int
pipe_t::mark ()
{
  int ret;

  /* First, flush the pipe.  */
  flush ();

  do
    {
      ret = ::write (write_fd (), "+", 1);
    }
  while (ret == -1 && errno == EINTR);

  if (ret == -1 && errno != EAGAIN)
    fatal_error ("write: %s", strerror (errno));

  return ret == -1 ? -errno : 0;
}

class pipe_notifier_t : public notifier_t
{
public:
  pipe_notifier_t () {}
  ~pipe_notifier_t () override {}

  void open () override { m_pipe.open (); }
  void close () override { m_pipe.close (); }
  bool is_valid () const override { return m_pipe.is_valid (); }

  amd_dbgapi_notifier_t consumer_end () const override
  {
    return m_pipe.read_fd ();
  }
  amd_dbgapi_notifier_t producer_end () const override
  {
    return m_pipe.write_fd ();
  }
  bool mark () override
  {
    const int ret = m_pipe.mark ();
    return ret > 0 || ret == -EAGAIN;
  }
  bool clear () override
  {
    const int ret = m_pipe.flush ();
    return ret == 0 || ret == -EAGAIN;
  }

private:
  pipe_t m_pipe;
};

std::unique_ptr<notifier_t>
notifier_t::create ()
{
  return std::make_unique<pipe_notifier_t> ();
}

} /* namespace amd::dbgapi */
