////////////////////////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//
////////////////////////////////////////////////////////////////////////////////
#include <cstring>
#include <cstdlib>
#include <climits> /* PATH_MAX */
#include <cstdio>
#if !defined(_WIN32) && !defined(__CYGWIN__)
#include <link.h>
#include <dlfcn.h>
#endif
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define PATH_MAX MAX_PATH
#elif !defined(PATH_MAX)
#define PATH_MAX FILENAME_MAX
#endif
#define RC_PATH_MAX (PATH_MAX > 1024 && FILENAME_MAX > 1024 ? 1024 : (PATH_MAX < FILENAME_MAX ? PATH_MAX : FILENAME_MAX))

#include "rocm_getpath.h"

/* Macro for NULL CHECK */
#define NULL_CHECK(ptr) if(!ptr) return PathIncorrecPararmeters;


/* Target Library Install Dir */
#define TARGET_LIB_INSTALL_DIR TARGET_LIBRARY_INSTALL_DIR

/* Target Library Name Buf Size */
#define LIBRARY_FILENAME_BUFSZ RC_PATH_MAX+1

/* Internal Function to get Base Path - Ref from Icarus Logic*/
static int getROCmBase(char *buf);

/* Public Function to get the ROCm Install Base Path
//  Argument1 (out) :  InstallPath (char** pointer which will return InstallPath found)
//  Argument2 (out) :  InstallPathLen (Pointer to integer (size of InstallPath) returned)
//  Usage :
//      char *installPath=NULL;
//      int installPathLen = 0;
//      installStatus = getROCmInstallPath( &installPath, &installPathLen );
//      if(installStatus !=PathSuccess ){  // error occured
//	...
//	}
//      free(installPath); //caller must free allocated memory after usage.
//    ...

*/
PathErrors_t getROCmInstallPath( char** InstallPath, unsigned int *InstallPathLen ) {

        NULL_CHECK(InstallPath);
        NULL_CHECK(InstallPathLen);
        int ret = PathErrorMAX;
        char *bufPtr = (char *)NULL;
        unsigned int bufSz = 0;

        bufPtr = (char *)malloc( LIBRARY_FILENAME_BUFSZ * sizeof(char) );
        memset( bufPtr, 0, LIBRARY_FILENAME_BUFSZ );
        *InstallPathLen = 0;
        *InstallPath = NULL;

        ret = getROCmBase(bufPtr);
        if (0 > ret){
          free(bufPtr);
          return (PathErrors_t)ret;
        }
        else if (0 == ret){
          free(bufPtr);
          return PathFailedToGetBase;
        }
        else{
          bufSz = ret;//additional char for null termination
        }

        *InstallPath = bufPtr;
        *InstallPathLen = bufSz;
        return  PathSuccess;
}

/* General purpose function that fills the directory to find rocm related stuff */
/* returns the offset into the buffer for the terminating NUL or -1 for error */
/* The buffer should be at least PATH_MAX */
static int getROCmBase(char *buf)
{
  int len=0;
  char *envStr=NULL;
  char libFileName[LIBRARY_FILENAME_BUFSZ];
  char *end=NULL;

  // Check Environment Variable is set for ROCM
  // install base path, then use it directly.
  if ((envStr = getenv("ROCM_PATH"))) {
    /* User space override, essentially just copied through as long as it is not too long */
    len = strlen(envStr);
    if (len > 0) {
      if (envStr[len] == '/') {
         /* Already has at least one terminating */
         len--;
      }
      if (len > LIBRARY_FILENAME_BUFSZ-1 ) {
         return PathValuesTooLong;
      }
      strncpy(buf, envStr, len);
      buf[len]='/';
      buf[len+1]='\0';

      /* Length of string including trailing '/' */
      return len+1;
    }
  }

  // If Environment Variable is not set
  // use platform-specific APIs to get target lib path
  // and get rocm base install path using the lib Path.
#if defined(_WIN32) || defined(_WIN64)
  // Windows implementation using Windows APIs
  sprintf(libFileName, "%s.dll", TARGET_LIBRARY_NAME);

  HMODULE hModule = LoadLibraryA(libFileName);
  if (!hModule) {
    // Try common ROCm installation paths
    const char* commonPaths[] = {
      "C:\\Program Files\\AMD\\ROCm\\",
      "C:\\ROCm\\",
      "C:\\AMD\\ROCm\\"
    };

    for (int i = 0; i < sizeof(commonPaths)/sizeof(commonPaths[0]); i++) {
      char testPath[LIBRARY_FILENAME_BUFSZ];
      sprintf(testPath, "%s%s\\%s", commonPaths[i], TARGET_LIB_INSTALL_DIR, libFileName);
      hModule = LoadLibraryA(testPath);
      if (hModule) {
        strcpy(buf, commonPaths[i]);
        len = strlen(buf);
        FreeLibrary(hModule);
        return len;
      }
    }
    return PathLinuxRuntimeErrors;
  }

  // Get the full path of the loaded module
  char modulePath[LIBRARY_FILENAME_BUFSZ];
  DWORD pathLen = GetModuleFileNameA(hModule, modulePath, LIBRARY_FILENAME_BUFSZ);
  FreeLibrary(hModule);

  if (pathLen == 0 || pathLen >= LIBRARY_FILENAME_BUFSZ) {
    return PathLinuxRuntimeErrors;
  }

  // Convert backslashes to forward slashes for consistency
  for (DWORD i = 0; i < pathLen; i++) {
    if (modulePath[i] == '\\') {
      modulePath[i] = '/';
    }
  }

  // Find the library install directory and strip it to get ROCm base
  char* libDirPos = strstr(modulePath, TARGET_LIB_INSTALL_DIR);
  if (libDirPos == NULL) {
    // If we can't find the expected lib directory, try to get parent directory
    char* lastSlash = strrchr(modulePath, '/');
    if (lastSlash && lastSlash > modulePath) {
      *lastSlash = '\0';
      // Try one more level up
      lastSlash = strrchr(modulePath, '/');
      if (lastSlash && lastSlash > modulePath) {
        *(lastSlash + 1) = '\0'; // Keep the trailing slash
        strcpy(buf, modulePath);
        len = strlen(buf);
        return len;
      }
    }
    return PathLinuxRuntimeErrors;
  }

  // Terminate string at the lib directory to get ROCm base path
  *libDirPos = '\0';
  strcpy(buf, modulePath);
  len = strlen(buf);

  // Ensure trailing slash
  if (len > 0 && buf[len-1] != '/') {
    buf[len] = '/';
    buf[len+1] = '\0';
    len++;
  }

#elif BUILD_SHARED_LIBS
  sprintf(libFileName, "lib%s.so", TARGET_LIBRARY_NAME);
  void *handle=dlopen(libFileName,RTLD_NOW);
  if (!handle){
    /* We can't find the library */
    return PathLinuxRuntimeErrors;
  }
  /* Variable to hold the return value from dlinfo */
  struct link_map *map = (struct link_map*)NULL;
  /* Query the runtime linker */
  dlinfo(handle,RTLD_DI_LINKMAP,&map);
  if (map ->l_name && realpath(map ->l_name,buf)) {
    /* Get Library Directory Path */
    char *end = strrchr(buf, '/');
    if (end && end > buf) {
      *end = '\0';
    }
  }
  else{
    /* If l_name is NULL or realpath() failed
     * Close handle before return error */
    dlclose(handle);
    return PathLinuxRuntimeErrors;
  }

  dlclose(handle);
  /* find the start of substring TARGET_LIB_INSTALL_DIR
   * To strip down Path up to Parent Directory of TARGET_LIB_INSTALL_DIR. */
  end=strstr(buf, TARGET_LIB_INSTALL_DIR);
  if( NULL == end ){
    /* We can't find the library install directory*/
    return PathLinuxRuntimeErrors;
  }
  *end = '\0';
#else
  // BUILD_SHARED_LIBS not defined
  return PathLinuxRuntimeErrors;
#endif

  /* Length of Path String up to Parent Directoy (ROCm Base Path)
   * with trailing '/'.*/
  len = strlen(buf);
  return len;
}

