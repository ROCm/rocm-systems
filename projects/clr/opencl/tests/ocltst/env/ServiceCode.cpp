/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include <windows.h>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 1
#include "CL/cl.hpp"

SERVICE_STATUS serviceStatus = {0};
SERVICE_STATUS_HANDLE serviceStatusHandle = 0;

const wchar_t* CrossProcessEventName = L"Global\\OpenCL_Test_serviceEvent";
const wchar_t* successMessage = L"OpenCL Service Test Success\n";
const wchar_t* serviceName = L"OpenCL Test service";
// this event is set whenever the service thread is finished executing
// all it's tasks
HANDLE RetireServiceEvent = 0;

DWORD WINAPI ThreadProc(LPVOID lpdwThreadParam);

//////////////////////////
// log relate functions //
//////////////////////////
void getLogFileName(wchar_t fileName[MAX_PATH]) {
  DWORD dwSize = GetModuleFileNameW(NULL, fileName, MAX_PATH);
  wchar_t* p = fileName + dwSize;
  while (*p != '\\' && p > fileName) p--;
  p++;
  wcscpy(p, L"result.txt");
}

VOID WriteLog(const wchar_t* pMsg) {
  static wchar_t fileName[MAX_PATH] = {0};
  if (fileName[0] == 0) getLogFileName(fileName);

  FILE* pLog = _wfopen(fileName, L"w");
  if (NULL != pLog) {
    fwprintf(pLog, pMsg);
    fclose(pLog);
  }
}

VOID AppendLog(const wchar_t* pMsg) {
  static wchar_t fileName[MAX_PATH] = {0};
  if (fileName[0] == 0) getLogFileName(fileName);
  FILE* pLog = _wfopen(fileName, L"a");
  if (NULL != pLog) {
    fwprintf(pLog, pMsg);
    fclose(pLog);
  }
}

VOID AppendLog(const char* pMsg) {
  static wchar_t fileName[MAX_PATH] = {0};
  if (fileName[0] == 0) getLogFileName(fileName);
  FILE* pLog = _wfopen(fileName, L"a");
  if (NULL != pLog) {
    fprintf(pLog, pMsg);
    fclose(pLog);
  }
}
///////////////////////////////
// service related functions //
///////////////////////////////
void WINAPI ServiceControlHandler(DWORD controlCode) {
  switch (controlCode) {
    case SERVICE_CONTROL_INTERROGATE:
      break;

    case SERVICE_CONTROL_SHUTDOWN:
    case SERVICE_CONTROL_STOP:
      serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
      if (!SetServiceStatus(serviceStatusHandle, &serviceStatus))
        AppendLog(L"SetServiceStatus SERVICE_STOP_PENDING failed\n");

      if (RetireServiceEvent) SetEvent(RetireServiceEvent);
      return;

    case SERVICE_CONTROL_PAUSE:
      break;

    case SERVICE_CONTROL_CONTINUE:
      break;

    default:
      if (controlCode >= 128 && controlCode <= 255)
        // user defined control code
        break;
      else
        // unrecognised control code
        break;
  }

  if (!SetServiceStatus(serviceStatusHandle, &serviceStatus))
    AppendLog(L"SetServiceStatus SERVICE_STOP_PENDING failed\n");
}

void WINAPI ServiceMain(DWORD /*argc*/, wchar_t* /*argv*/[]) {
  // initialise service status
  serviceStatus.dwServiceType = SERVICE_WIN32;
  serviceStatus.dwCurrentState = SERVICE_START_PENDING;
  serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_SHUTDOWN;
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = NO_ERROR;
  serviceStatus.dwCheckPoint = 0;
  serviceStatus.dwWaitHint = 0;

  serviceStatusHandle = RegisterServiceCtrlHandlerW(serviceName, ServiceControlHandler);

  if (serviceStatusHandle) {
    // service is starting
    serviceStatus.dwCurrentState = SERVICE_START_PENDING;
    if (!SetServiceStatus(serviceStatusHandle, &serviceStatus))
      AppendLog(L"SetServiceStatus SERVICE_START_PENDING failed\n");

    // do initialisation here
    RetireServiceEvent = CreateEvent(0, FALSE, FALSE, 0);

    // running
    serviceStatus.dwControlsAccepted |= (SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN);
    serviceStatus.dwCurrentState = SERVICE_RUNNING;
    if (!SetServiceStatus(serviceStatusHandle, &serviceStatus))
      AppendLog(L"SetServiceStatus SERVICE_RUNNING failed\n");

    // Create the thread that actually does the CL testing
    CreateThread(NULL, 0, ThreadProc, NULL, 0, NULL);
    // wait for the thread to finish
    WaitForSingleObject(RetireServiceEvent, 60000);

    HANDLE crossProcessEvent = OpenEventW(EVENT_ALL_ACCESS, FALSE, CrossProcessEventName);
    if (NULL != crossProcessEvent) {
      SetEvent(crossProcessEvent);
    } else {
      AppendLog(L"cross process Event could not be openned\n");
    }

    // service was stopped
    serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
    if (!SetServiceStatus(serviceStatusHandle, &serviceStatus))
      AppendLog(L"SetServiceStatus SERVICE_STOP_PENDING failed\n");

    // do cleanup here
    CloseHandle(crossProcessEvent);
    CloseHandle(RetireServiceEvent);
    RetireServiceEvent = 0;

    // service is now stopped
    serviceStatus.dwControlsAccepted &= ~(SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN);
    serviceStatus.dwCurrentState = SERVICE_STOPPED;
    if (!SetServiceStatus(serviceStatusHandle, &serviceStatus))
      AppendLog(L"SetServiceStatus SERVICE_STOPPED failed\n");
  }
}

// This function services ocltst as a service when launched
// by the OS. It registers the service routines.
void serviceStubCall() {
  wchar_t serviceName[MAX_PATH];
  wcscpy(serviceName, ::serviceName);
  SERVICE_TABLE_ENTRYW serviceTable[] = {{serviceName, ServiceMain}, {0, 0}};
  DWORD session_id;
  BOOL retVal = ProcessIdToSessionId(GetCurrentProcessId(), &session_id);
  if (0 == session_id) {
    StartServiceCtrlDispatcherW(serviceTable);
  }
}
/////////////////////
// CL related code //
/////////////////////
const char c_kernelCode[] =
    " __kernel void hello(__global char * theArray)"
    "{"
    " size_t i =  get_global_id(0);"
    "if ( i < get_global_size(0))"
    "theArray[i] = 78;"
    "}";

const unsigned int c_bufferSize = 1024;

DWORD WINAPI ThreadProc(LPVOID lpdwThreadParam) {
  cl_int err;
  // Platform info
  std::vector<cl::Platform> platforms;

  err = cl::Platform::get(&platforms);
  if (err != CL_SUCCESS) {
    AppendLog(L"Platform::get() failed\n");
    return -1;
  }

  std::vector<cl::Platform>::iterator i;
  if (platforms.size() > 0) {
    for (i = platforms.begin(); i != platforms.end(); ++i) {
      if (!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc.")) {
        break;
      }
    }
  }
  if (err != CL_SUCCESS) {
    AppendLog(L"Platform::getInfo() failed \n");
    return -1;
  }

  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(*i)(), 0};

  cl::Context context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    AppendLog(L"Context::Context() failed \n");
    return -1;
  }

  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  if (err != CL_SUCCESS) {
    AppendLog(L"Context::getInfo() failed \n");
    return -1;
  }
  if (devices.size() == 0) {
    AppendLog(L"No device available\n");
    return -1;
  }

  cl::Program::Sources sources(1, std::make_pair(c_kernelCode, sizeof(c_kernelCode)));

  cl::Program program = cl::Program(context, sources, &err);
  if (err != CL_SUCCESS) {
    AppendLog(L"Program::Program() failed\n");
  }

  err = program.build(devices);
  if (err != CL_SUCCESS) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
      std::string str((char*)program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());

      AppendLog(L" \n\t\t\tBUILD LOG\n\n");
      AppendLog(L" ************************************************\n");
      AppendLog(str.c_str());
      AppendLog(L" ************************************************\n");
    }

    AppendLog(L"Program::build() failed\n");
    return -1;
  }

  cl::Kernel kernel(program, "hello", &err);
  if (err != CL_SUCCESS) {
    AppendLog(L"Kernel::Kernel() failed\n");
    return -1;
  }

  cl::Buffer buffer = cl::Buffer(context, CL_MEM_READ_WRITE, c_bufferSize, 0, &err);
  if (err != CL_SUCCESS) {
    AppendLog(L"Kernel::setArg() failed \n");
  }

  cl::CommandQueue queue(context, devices[0], 0, &err);
  if (err != CL_SUCCESS) {
    AppendLog(L"CommandQueue::CommandQueue() failed \n");
    return -1;
  }

  err = kernel.setArg(0, buffer);
  if (err != CL_SUCCESS) {
    AppendLog(L"Kernel::setArg() failed \n");
    return -1;
  }

  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(c_bufferSize), cl::NullRange);

  if (err != CL_SUCCESS) {
    AppendLog(L"CommandQueue::enqueueNDRangeKernel()\n");
    return -1;
  }

  err = queue.finish();
  if (err != CL_SUCCESS) {
    AppendLog(L"Event::wait() failed \n");
  }
  char* ptr = (char*)malloc(c_bufferSize);
  err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, c_bufferSize, ptr, NULL, NULL);
  if (err != CL_SUCCESS) {
    AppendLog(L"CommandQueue::enqueueReadBuffer()\n");
    return -1;
  }

  bool validateSuccess = true;
  // validate the results
  for (int i = 0; i < c_bufferSize; i++) {
    if (ptr[i] != 78) validateSuccess = false;
  }

  free(ptr);
  if (validateSuccess) {
    WriteLog(successMessage);
    AppendLog(L"validate success");
  } else {
    AppendLog(L"Validate fail");
    return -1;
  }

  SetEvent(RetireServiceEvent);
  return 0;
}
