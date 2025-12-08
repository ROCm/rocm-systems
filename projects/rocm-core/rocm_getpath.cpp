//Copyright © Advanced Micro Devices, Inc., or its affiliates.
//SPDX-License-Identifier: MIT

#include <cstring> /* String Operations */
#include <cstdlib> /* Dynamic Memory Mgmt */
#include <cstdio>  /* FILENAME_MAX */
#if !defined(_WIN32) && !defined(_WIN64)
  #include <climits> /* PATH_MAX */
  #include <link.h>  /* ELF Dynamic Linking DS */
  #include <dlfcn.h> /* Dynamic linker Operations */
  #define RC_PATH_MAX (PATH_MAX+1)
#else
  #if defined(_WIN32) || defined(_WIN64)
    #include <windows.h> /* MAX_PATH */
    #define PATH_MAX MAX_PATH
  #elif !defined(PATH_MAX)
    #define PATH_MAX FILENAME_MAX
  #endif
  #define RC_PATH_MAX ((PATH_MAX > 1024 && FILENAME_MAX > 1024 ? 1024 : (PATH_MAX < FILENAME_MAX ? PATH_MAX : FILENAME_MAX))+1)
#endif

#include "rocm_getpath.h"

/* Macro for NULL CHECK */
#define NULL_CHECK(ptr) if(!ptr) return PathIncorrecPararmeters;

/* Target Library Install Dir */
#define TARGET_LIB_INSTALL_DIR TARGET_LIBRARY_INSTALL_DIR

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

        bufPtr = (char *)malloc( RC_PATH_MAX * sizeof(char) );
        memset( bufPtr, 0, RC_PATH_MAX );
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
  char libFileName[RC_PATH_MAX];
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
      if (len > RC_PATH_MAX-1 ) {
         return PathValuesTooLong;
      }
      strncpy(buf, envStr, len);
      buf[len]='/';
      buf[len+1]='\0';

      /* Length of string including trailing '/' */
      return len+1;
    }
  }

  /* If Environment Variable is not set
   * use platform-specific APIs to get target lib path
   * and get rocm base install path using the lib Path. */
#if defined(_WIN32) || defined(_WIN64)
  /* Limited support for Windows:
   * getROCmBase() Needs ROCM_PATH environment variable set on Windows. */
  return PathWindowsNotSet;
#else
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
  #else
    // BUILD_SHARED_LIBS not defined
    return PathLinuxRuntimeErrors;
  #endif
#endif
  /* Length of Path String up to Parent Directoy (ROCm Base Path)
   * with trailing '/'.*/
  len = strlen(buf);
  return len;
}

