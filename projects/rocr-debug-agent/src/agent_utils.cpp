/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "agent_utils.h"
#include "debug.h"

#include <getopt.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

namespace amd::debug_agent
{

/* Hex and register value formatting functions. */

std::string
hex_string (const std::vector<uint8_t> &value)
{
  std::string value_string;
  value_string.reserve (2 * value.size ());

  for (size_t pos = value.size (); pos > 0; --pos)
    {
      static constexpr char hex_digits[] = "0123456789abcdef";
      value_string.push_back (hex_digits[value[pos - 1] >> 4]);
      value_string.push_back (hex_digits[value[pos - 1] & 0xF]);
    }

  return value_string;
}

std::string
register_value_string (const std::string &register_type,
                       const std::vector<uint8_t> &register_value)
{
  /* Handle vector types. */
  if (size_t pos = register_type.find_last_of ('['); pos != std::string::npos)
    {
      const std::string element_type = register_type.substr (0, pos);
      const size_t element_count = std::stoi (register_type.substr (pos + 1));
      const size_t element_size = register_value.size () / element_count;

      agent_assert ((register_value.size () % element_size) == 0);

      std::stringstream ss;
      for (size_t i = 0; i < element_count; ++i)
        {
          if (i != 0)
            ss << " ";
          ss << "[" << i << "] ";

          std::vector<uint8_t> element_value (
              &register_value[element_size * i],
              &register_value[element_size * (i + 1)]);

          ss << register_value_string (element_type, element_value);
        }
      return ss.str ();
    }

  return hex_string (register_value);
}

/* URI parsing and sanitization. */

parsed_uri_t
parse_code_object_uri (const std::string &uri)
{
  parsed_uri_t result;

  const std::string protocol_delim{ "://" };

  size_t protocol_end = uri.find (protocol_delim);
  result.protocol = uri.substr (0, protocol_end);
  protocol_end += protocol_delim.length ();

  std::transform (result.protocol.begin (), result.protocol.end (),
                  result.protocol.begin (),
                  [] (unsigned char c) { return std::tolower (c); });

  std::string path;
  size_t path_end = uri.find_first_of ("#?", protocol_end);
  if (path_end != std::string::npos)
    path = uri.substr (protocol_end, path_end++ - protocol_end);
  else
    path = uri.substr (protocol_end);

  /* %-decode the string. */
  result.decoded_path.reserve (path.length ());
  for (size_t i = 0; i < path.length (); ++i)
    if (path[i] == '%' && i + 2 < path.length ()
        && std::isxdigit (path[i + 1]) && std::isxdigit (path[i + 2]))
      {
        result.decoded_path += std::stoi (path.substr (i + 1, 2), 0, 16);
        i += 2;
      }
    else
      result.decoded_path += path[i];

  /* Tokenize the query/fragment. */
  std::vector<std::string> tokens;
  size_t pos, last = path_end;
  while ((pos = uri.find ('&', last)) != std::string::npos)
    {
      tokens.emplace_back (uri.substr (last, pos - last));
      last = pos + 1;
    }
  if (last != std::string::npos)
    tokens.emplace_back (uri.substr (last));

  /* Create a tag-value map from the tokenized query/fragment. */
  std::for_each (tokens.begin (), tokens.end (), [&] (std::string &token) {
    size_t delim = token.find ('=');
    if (delim != std::string::npos)
      result.params.emplace (token.substr (0, delim), token.substr (delim + 1));
  });

  return result;
}

std::string
sanitize_uri_for_filename (const std::string &uri)
{
  std::string name{ uri };

  size_t pos{};
  while ((pos = name.find_first_of (":/#?&="), pos) != std::string::npos)
    name[pos] = '_';

  return name;
}

/* Debug agent options parsing. */

void
print_usage ()
{
  std::cerr << "ROCdebug-agent usage:" << std::endl;
  std::cerr << "  -a, --all                   "
               "Print all wavefronts."
            << std::endl;
  std::cerr << "  -s, --save-code-objects[=DIR]   "
               "Save all loaded code objects. If the directory"
            << std::endl
            << "                              "
               "is not specified, the code objects are saved in"
            << std::endl
            << "                              "
               "the current directory."
            << std::endl;
  std::cerr << "  -c, --load-all-code-objects "
               "Load all code objects as soon as they are loaded"
            << std::endl
            << "                              "
            << "by the runtime.";
  std::cerr << "  -z, --lazy                  "
               "Delay inspecting the content of all loaded code "
            << std::endl
            << "                              "
            << "obects until after an exception is reported."
            << std::endl
            << "                              "
            << "Note that the application must not free the code "
            << std::endl
            << "                              "
            << "objects' memory while they are loaded on the device."
            << std::endl
            << "                              "
            << "This option is incompatible with -c."
            << std::endl;
  std::cerr << "  -p, --precise-memory        "
            << "Enable precise memory mode which ensures that " << std::endl
            << "                              "
               "when an exception is reported, the PC points to"
            << std::endl
            << "                              "
               "the instruction immediately after the one that"
            << std::endl
            << "                              "
               "caused the exception."
            << std::endl;
  std::cerr << "  -o, --output=FILE           "
               "Save the output in FILE. By default, the output"
            << std::endl
            << "                              "
               "is redirected to stderr."
            << std::endl;
  std::cerr << "  -d, --disable-linux-signals "
               "Disable installing a SIGQUIT signal handler, so"
            << std::endl
            << "                              "
               "that the default Linux handler may dump a core"
            << std::endl
            << "                              "
               "file."
            << std::endl;
  std::cerr << "  -l, --log-level={none|error|warning|info|verbose}"
            << std::endl
            << "                              "
               "Change the Debug Agent and Debugger API log"
            << std::endl
            << "                              "
               "level. The default log level is 'none'."
            << std::endl;
  std::cerr << "  -h, --help                  "
               "Display a usage message and abort the process."
            << std::endl;
}

std::variant<std::string, debug_agent_options_t>
parse_debug_agent_options (const char *env_options)
{
  debug_agent_options_t options;

  if (!env_options)
    return options;

  std::istringstream args_stream (env_options);

  /* RAII wrapper for argv strings to prevent leaks. */
  std::vector<std::unique_ptr<char[], decltype (&std::free)>> arg_storage;
  std::vector<char *> args;

  arg_storage.emplace_back (strdup ("rocm-debug-agent"), &std::free);
  args.push_back (arg_storage.back ().get ());

  for (std::istream_iterator<std::string> it (args_stream),
       end = std::istream_iterator<std::string> ();
       it != end; ++it)
    {
      arg_storage.emplace_back (strdup (it->c_str ()), &std::free);
      args.push_back (arg_storage.back ().get ());
    }

  char *const *argv = const_cast<char *const *> (args.data ());
  int argc = args.size ();

  static struct option long_options[]
      = { { "all", no_argument, nullptr, 'a' },
          { "disable-linux-signals", no_argument, nullptr, 'd' },
          { "log-level", required_argument, nullptr, 'l' },
          { "output", required_argument, nullptr, 'o' },
          { "save-code-objects", optional_argument, nullptr, 's' },
          { "lazy", no_argument, nullptr, 'z' },
          { "load-all-code-objects", no_argument, nullptr, 'c' },
          { "precise-memory", no_argument, nullptr, 'p' },
          { "precise-alu-exceptions", no_argument, nullptr, 'e' },
          { "help", no_argument, nullptr, 'h' },
          { 0 } };

  /* We use getopt_long locally, so preserve and reset the global optind. */
  int saved_optind = optind;
  optind = 1;

  while (int c
         = getopt_long (argc, argv, ":as::o:dpezcl:h", long_options, nullptr))
    {
      if (c == -1)
        break;

      std::optional<std::string> argument;

      if (!optarg && optind < argc && *argv[optind] != '-')
        optarg = argv[optind++];

      if (optarg)
        argument.emplace (optarg);

      switch (c)
        {
        case 'a': /* -a or --all */
          options.all_wavefronts = true;
          break;

        case 'd': /* -d or --disable-linux-signals */
          options.disable_sigquit = true;
          break;

        case 'p': /* -p or --precise-memory */
          options.precise_memory = true;
          break;

        case 'e': /* -e or --precise-alu-exceptions */
          options.precise_alu_exceptions = true;
          break;

        case 'l': /* -l or --log-level */
          if (!argument)
            return std::string ("error: --log-level requires an argument");

          if (argument == "none")
            options.log_level = log_level_t::none;
          else if (argument == "verbose")
            options.log_level = log_level_t::verbose;
          else if (argument == "info")
            options.log_level = log_level_t::info;
          else if (argument == "warning")
            options.log_level = log_level_t::warning;
          else if (argument == "error")
            options.log_level = log_level_t::error;
          else
            return std::string ("error: Invalid log level '" + *argument + "'");

          break;

        case 'c': /* -c or --load-all-code-objects */
          options.lazy = false;
          break;

        case 'z': /* -z or --lazy */
          options.delay_loading = true;
          break;

        case 's': /* -s or --save-code-objects */
          if (argument)
            {
              struct stat path_stat;
              if (stat (argument->c_str (), &path_stat) == -1
                  || !S_ISDIR (path_stat.st_mode))
                return std::string ("error: Cannot access code object save "
                                    "directory '" + *argument + "'");

              options.code_objects_dir = *argument;
            }
          else
            {
              options.code_objects_dir = ".";
            }
          break;

        case 'o': /* -o or --output */
          if (!argument)
            return std::string ("error: --output requires an argument");

          options.output_file = *argument;
          break;

        case '?': /* Unrecognized option */
          return std::string ("error: Unrecognized option");

        case ':': /* Missing required argument */
          if (optopt == 'l')
            return std::string ("error: --log-level requires an argument");
          else if (optopt == 'o')
            return std::string ("error: --output requires an argument");
          else
            return std::string ("error: Option requires an argument");
          break;

        case 'h': /* -h or --help */
          return std::string ("help");

        default:
          return std::string ("error: Unknown option");
        }
    }

  if (!options.lazy && options.delay_loading)
    return std::string ("error: \"--load-all-code-objects\" and \"--lazy\" are "
                        "mutually exclusive");

  /* Restore the global optind. */
  optind = saved_optind;

  return options;
}

/* Exception bitmask mapping. */

std::underlying_type_t<amd_dbgapi_exceptions_t>
map_stop_reason_to_exceptions (
    std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason)
{
  std::underlying_type_t<amd_dbgapi_exceptions_t> resume_exceptions = 0;
  auto stop_reason_bits{ stop_reason };

  do
    {
      auto one_bit
          = stop_reason_bits ^ (stop_reason_bits & (stop_reason_bits - 1));
      stop_reason_bits ^= one_bit;

      switch (one_bit)
        {
        case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
        case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_NONE;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
        case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
        case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
        case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_TRAP;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
          /* Is this even possible? */
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_NONE;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
        case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ADDRESS_ERROR;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
        case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ABORT;
          break;

#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
        case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
          break;
#endif
        }
    }
  while (stop_reason_bits != 0);

  return resume_exceptions;
}

const char *
stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t reason)
{
  switch (reason)
    {
    case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
      return "NONE";
    case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
      return "BREAKPOINT";
    case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
      return "WATCHPOINT";
    case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
      return "SINGLE_STEP";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
      return "FP_INPUT_DENORMAL";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
      return "FP_DIVIDE_BY_0";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
      return "FP_OVERFLOW";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
      return "FP_UNDERFLOW";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
      return "FP_INEXACT";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
      return "FP_INVALID_OPERATION";
    case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
      return "INT_DIVIDE_BY_0";
    case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
      return "DEBUG_TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
      return "ASSERT_TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
      return "TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
      return "MEMORY_VIOLATION";
    case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
      return "ADDRESS_ERROR";
    case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
      return "ILLEGAL_INSTRUCTION";
    case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
      return "ECC_ERROR";
    case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
      return "FATAL_HALT";
#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
    case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
      return "RESERVED";
#endif
    }
  return "";
}

} /* namespace amd::debug_agent */
