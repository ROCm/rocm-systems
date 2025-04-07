#ifdef _WIN32 // Compiling for Windows Platform

#include <stdlib.h>
#include <Windows.h>
#include "os.h"
#include <stdio.h>

void SetEnv(const char* env_var_name, const char* env_var_value) {
  BOOL err = SetEnvironmentVariable(env_var_name, env_var_value);
  if(FALSE == err){
	  printf("Set environment variable failed!\n");
	  exit(1);
  }
  return;
}

char* GetEnv(const char* env_var_name){
  char* buff;
  DWORD char_count = GetEnvironmentVariable(env_var_name, NULL, 0);
  if (char_count == 0) return NULL;
  buff = (char*)malloc(sizeof(char) * char_count);
  GetEnvironmentVariable(env_var_name, buff, char_count);
  buff[char_count - 1] = '\0';
  return buff;
}

#elif defined(__linux__)

#include "os.h"
#include <stdlib.h>

void SetEnv(const char* env_var_name, const char* env_var_value){
	int err = setenv(env_var_name, env_var_value, 1);
	if(0 != err){
		printf("Set environment variable failed!\n");
		exit(1);
	}
	return;
}

char* GetEnv(const char* env_var_name) {
  return getenv(env_var_name);
}

#endif
