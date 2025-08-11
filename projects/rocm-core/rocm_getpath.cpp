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

#include <string.h>
#include <stdlib.h>
#include <limits.h> /* PATH_MAX */
#include <stdio.h>
#include <link.h>
#include <dlfcn.h>
#include "rocm_getpath.h"

/* Macro for NULL CHECK */
#define NULL_CHECK(ptr) if(!ptr) return PathIncorrecPararmeters;


/* Target Library Install Dir */
#define TARGET_LIB_INSTALL_DIR TARGET_LIBRARY_INSTALL_DIR

/* Target Library Name Buf Size */
#define LIBRARY_FILENAME_BUFSZ (PATH_MAX+1)

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
      if (len > PATH_MAX-1 ) {
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
  // use dl APIs to get target lib path
  // and get rocm base install path using the lib Path.
#if BUILD_SHARED_LIBS
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
#endif

  /* Length of Path String up to Parent Directoy (ROCm Base Path)
   * with trailing '/'.*/
  len = strlen(buf);
  return len;
}

