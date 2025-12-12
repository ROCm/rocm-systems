/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef HAVE_BFD
#include "BfdBacktrace.hpp"
#endif

#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
#include <string.h>
#include "param.h"
#include "debug.h"
#include <vector>

void sig_handler(int signum)
{
  printf("\n[Process: %d] Inside handler function signal: %s (%d)\n", getpid(), strsignal(signum), signum);

#ifdef HAVE_BFD
  void *addresses[BACKTRACE_MAX];
  int num_addresses = backtrace(addresses, BACKTRACE_MAX);
  struct backtrace_file file;
  backtrace_line line;
  backtrace_h bckt;
  bckt.size = 0;

  for (int i = 0; i < num_addresses; ++i)
  {
    file.dl.address = (unsigned long)addresses[i];
    if (dl_lookup_address(&file.dl) && load_file(&file))
    {
      bckt.size += get_line_info(&file, 1,
                                 bckt.lines + bckt.size,
                                 BACKTRACE_MAX - bckt.size);
      unload_file(&file);
    }
  }

  for (int i=0; i<BACKTRACE_MAX; i++ )
  {
    if ((char*)bckt.lines[i].address == NULL) break;
    printf("%p %s : %s line %u\n", (char*)bckt.lines[i].address,
           bckt.lines[i].file, bckt.lines[i].function, bckt.lines[i].lineno);
  }
#else
#define BT_BUF_SIZE 1024
  void *buffer[BT_BUF_SIZE];
  char **strings;

  int nptrs = backtrace(buffer, BT_BUF_SIZE);
  strings = backtrace_symbols(buffer, nptrs);
  for (int j = 0; j < nptrs; j++)
    printf("%s\n", strings[j]);
  free (strings);
#endif

  if (signum == SIGUSR2) {
    return;
  }

  exit (-1);
}

RCCL_PARAM(EnableSignalHandler, "ENABLE_SIGNALHANDLER", 0); // Opt-in environment variable for enabling custom signal handler

void RegisterSignalHandlers()
{
  if (rcclParamEnableSignalHandler())
  {
    INFO(NCCL_INIT, "Enabling custom signal handler");

    std::vector<int> signalsToCatch = {SIGILL, SIGBUS, SIGFPE, SIGSEGV, SIGUSR2};

    for (auto signum : signalsToCatch)
    {
      if (signal(signum, sig_handler) == SIG_ERR)
      {
        INFO(NCCL_INIT, "Unable to register signal handler for %s\n", strsignal(signum));
      }
    }
  }
}
