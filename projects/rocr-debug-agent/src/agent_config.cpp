// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_config.h"
#include "agent_utils.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace amd::debug_agent
{

void
print_usage ()
{
  std::cerr << R"(ROCdebug-agent usage:
  -a, --all                   Print all wavefronts.
  -s, --save-code-objects[=DIR]   Save all loaded code objects. If the directory
                              is not specified, the code objects are saved in
                              the current directory.
  -c, --load-all-code-objects Load all code objects as soon as they are loaded
                              by the runtime.
  -z, --lazy                  Delay inspecting the content of all loaded code
                              obects until after an exception is reported.
                              Note that the application must not free the code
                              objects' memory while they are loaded on the device.
                              This option is incompatible with -c.
  -p, --precise-memory        Enable precise memory mode which ensures that
                              when an exception is reported, the PC points to
                              the instruction immediately after the one that
                              caused the exception.
  -o, --output=FILE           Save the output in FILE. By default, the output
                              is redirected to stderr.
  -d, --disable-linux-signals Disable installing a SIGQUIT signal handler, so
                              that the default Linux handler may dump a core
                              file.
  -l, --log-level={none|error|warning|info|verbose}
                              Change the Debug Agent and Debugger API log
                              level. The default log level is 'none'.
  -h, --help                  Display a usage message and abort the process.
)";
}

std::variant<std::string, debug_agent_options_t>
parse_debug_agent_options (const char *env_options)
{
  debug_agent_options_t options;

  if (!env_options)
    return options;

  std::istringstream args_stream (env_options);

  /* RAII wrapper for argv strings to prevent leaks.  */
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

  /* We use getopt_long locally, so preserve and reset the global optind.  */
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
              std::string dir = expand_format_tokens (*argument);

              struct stat path_stat;
              if (stat (dir.c_str (), &path_stat) == -1
                  || !S_ISDIR (path_stat.st_mode))
                return std::string ("error: Cannot access code object save "
                                    "directory '" + dir + "'");

              options.code_objects_dir = std::move (dir);
            }
          else
            {
              options.code_objects_dir = ".";
            }
          break;

        case 'o': /* -o or --output */
          if (!argument)
            return std::string ("error: --output requires an argument");

          options.output_file = expand_format_tokens (*argument);
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

  /* Restore the global optind.  */
  optind = saved_optind;

  return options;
}

} /* namespace amd::debug_agent */
