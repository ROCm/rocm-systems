#include "param.h"
#include "debug.h"

RCCL_PARAM(EnableSignalHandler, "ENABLE_SIGNALHANDLER", 0);

void RegisterSignalHandlers() {
  if (rcclParamEnableSignalHandler()) {
    WARN("RCCL_ENABLE_SIGNALHANDLER is unavailable on Windows; use the Windows crash debugger.");
  }
}
