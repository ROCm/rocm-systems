/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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

#include "utils_windows.h"
#include "utils.h"
#include <io.h>
#include <windows.h>

namespace amd::dbgapi
{

namespace utils
{

std::string
convert_to_string (std::wstring_view ws)
{
  if (ws.empty ())
    return {};

  int size_needed
    = ::WideCharToMultiByte (CP_ACP, 0, ws.data (), (int)ws.size (),
                             nullptr, 0, nullptr, nullptr);
  if (size_needed <= 0)
    return {};

  std::string result (static_cast<size_t> (size_needed), '\0');

  [[maybe_unused]] int size
    = ::WideCharToMultiByte (CP_ACP, 0, ws.data (), (int)ws.size (),
                             &result[0], size_needed, "?", nullptr);

  dbgapi_assert (size == size_needed);

  return result;
}

const char *
get_self_name ()
{
  static const auto self_name = [] () -> std::string
  {
    HMODULE h;
    if (::GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCWSTR)&get_self_name, &h))
      {
        std::wstring wself_name;
        DWORD size = MAX_PATH;

        /* Loop to find the true length, thus handling long paths.  */
        while (true)
          {
            wself_name.resize (size);
            DWORD len = ::GetModuleFileNameW (h, &wself_name[0], size);
            if (len == 0)
              break;

            if (len < size)
              {
                wself_name.resize (len);
                return convert_to_string (wself_name);
              }

            /* Buffer was too small; double it and try again.  */
            size *= 2;
          }
      }

    return {};
  } ();

  return self_name.c_str ();
}

} /* namespace amd::dbgapi::utils.  */

class event_notifier_t : public notifier_t
{
public:
  event_notifier_t ();
  ~event_notifier_t () override;

  void open () override;
  void close () override;
  bool is_valid () const override;

  amd_dbgapi_notifier_t producer_end () const override;
  amd_dbgapi_notifier_t consumer_end () const override;
  bool mark () override;
  bool clear () override;

private:
  HANDLE m_event = nullptr;
};

event_notifier_t::event_notifier_t () {}

event_notifier_t::~event_notifier_t ()
{
  if (is_valid ())
    close ();
}

void
event_notifier_t::open ()
{
  m_event = ::CreateEvent (nullptr, true, false, nullptr);

  if (m_event == nullptr)
    warning ("notifier_t::notifier_t: CreateEvent failed: %ld",
             ::GetLastError ());
}

void
event_notifier_t::close ()
{
  if (!is_valid ())
    return;

  if (::CloseHandle (m_event) == 0)
    warning ("notifier_t::~notifier_t: CloseHandle failed: %ld",
             ::GetLastError ());

  m_event = nullptr;
}

bool
event_notifier_t::is_valid () const
{
  return m_event != nullptr;
}

amd_dbgapi_notifier_t
event_notifier_t::producer_end () const
{
  return m_event;
}

amd_dbgapi_notifier_t
event_notifier_t::consumer_end () const
{
  return m_event;
}

bool
event_notifier_t::clear ()
{
  if (::ResetEvent (m_event) == 0)
    fatal_error ("notifier_t::clear: ResetEvent failed: %ld",
                 ::GetLastError ());
  return true;
}

bool
event_notifier_t::mark ()
{
  if (::SetEvent (m_event) == 0)
    fatal_error ("notifier_t::mark: SetEvent failed: %ld", ::GetLastError ());
  return true;
}

std::unique_ptr<notifier_t>
notifier_t::create ()
{
  return std::make_unique<event_notifier_t> ();
}

} /* namespace amd::dbgapi */
