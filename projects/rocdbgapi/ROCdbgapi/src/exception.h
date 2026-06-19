/* Copyright (c) 2021-2024 Advanced Micro Devices, Inc.

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

#ifndef AMD_DBGAPI_EXCEPTION_H
#define AMD_DBGAPI_EXCEPTION_H 1

#include "amd-dbgapi.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace amd::dbgapi
{

class address_space_t;
class process_t;

/* AMD Debugger API exception.  */

class exception_t : public std::runtime_error
{
protected:
  exception_t (std::string message = {})
    : std::runtime_error (std::move (message))
  {
  }

public:
  void print_message () const noexcept;
};

class process_exited_exception_t : public exception_t
{
private:
  process_t &m_process;

public:
  process_exited_exception_t (process_t &process, std::string message = {})
    : exception_t (std::move (message)), m_process (process)
  {
  }

  process_t &process () const noexcept { return m_process; }
};

class api_error_t : public exception_t
{
private:
  amd_dbgapi_status_t m_code; /* The error code for this exception.  */

public:
  api_error_t (amd_dbgapi_status_t code, std::string message = {})
    : exception_t (std::move (message)), m_code (code)
  {
  }

public:
  amd_dbgapi_status_t code () const noexcept { return m_code; }
};

class memory_access_error_t : public api_error_t
{
private:
  std::optional<
    std::pair<const address_space_t &, amd_dbgapi_segment_address_t>>
    m_address;

public:
  memory_access_error_t (const address_space_t &address_space,
                         amd_dbgapi_segment_address_t segment_address,
                         std::string message = {});

  const auto &address () const noexcept { return m_address; }
};

class fatal_error_t : public api_error_t
{
public:
  fatal_error_t (std::string message = {})
    : api_error_t (AMD_DBGAPI_STATUS_FATAL, std::move (message))
  {
  }
};

namespace detail
{

void update_process_handles (process_t *process);

} /* namespace detail */

#define TRY                                                                   \
  for (process_t *_exited_process = {};;)                                     \
    {                                                                         \
      try                                                                     \
        {                                                                     \
          if (_exited_process)                                                \
            {                                                                 \
              _exited_process->update_waves ();                               \
              _exited_process->update_queues ();                              \
              _exited_process->update_code_objects ();                        \
              _exited_process->update_agents ();                              \
            }

#define CATCH(/* allowed codes  */...)                                        \
  }                                                                           \
  catch (const amd::dbgapi::process_exited_exception_t &_error)               \
  {                                                                           \
    /* Process exited exceptions are never returned to the caller, instead,   \
       the function is re-tried one more time after process tear-down.  If a  \
       process exited exception is thrown again after retrying then return a  \
       fatal error.  */                                                       \
    if (!_exited_process)                                                     \
      {                                                                       \
        _exited_process = &_error.process ();                                 \
        continue;                                                             \
      }                                                                       \
    return AMD_DBGAPI_STATUS_FATAL;                                           \
  }                                                                           \
  catch (const amd::dbgapi::api_error_t &_error)                              \
  {                                                                           \
    /* If the error code is one that is allowed, simply return it.  */        \
    if ([&] (auto &&..._code) { return ((_error.code () == _code) || ...); }( \
          AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED, ##__VA_ARGS__))            \
      return _error.code ();                                                  \
                                                                              \
    /* Everything else is a fatal error.  */                                  \
    _error.print_message ();                                                  \
    return AMD_DBGAPI_STATUS_FATAL;                                           \
  }                                                                           \
  catch (...) { return AMD_DBGAPI_STATUS_FATAL; }                             \
                                                                              \
  return AMD_DBGAPI_STATUS_SUCCESS;                                           \
  }

#define THROW(error_code) throw amd::dbgapi::api_error_t (error_code)

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_EXCEPTION_H */
