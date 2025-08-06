#ifndef HSA_PERF_SRC_UTILS_OS_H_
#define HSA_PERF_SRC_UTILS_OS_H_

#include <stdio.h>

// Set envriroment variable
void SetEnv(const char* env_var_name, const char* env_var_value);

// Get the value of enviroment
char* GetEnv(const char* env_var_name);

#endif
