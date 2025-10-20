//Copyright © Advanced Micro Devices, Inc., or its affiliates.
 
//SPDX-License-Identifier: MIT


#include "rocm_version.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#define NULL_CHECK(ptr)	if(!ptr) return VerIncorrectParameters;


#define CHECK_AND_REPORT_API_RESULT(val)	do {					\
							if(VerSuccess != val) {	\
								const char *ErrStrings[VerErrorMAX]= { \
                                                "VerSuccess", \
                                                "VerIncorrectParameters", \
                                                "VerMemoryAllocationFailed", \
                                                "VerValuesNotDefined" \
                                            };  \
								fprintf(stderr, " API returned : %s \n", ErrStrings[val]);	\
								fflush(stderr);		\
								return val;		\
							}				\
						}while(0);



VerErrors getROCmVersion(unsigned int* Major, unsigned int* Minor, unsigned int* Patch) {

	NULL_CHECK(Major)
	NULL_CHECK(Minor)
	NULL_CHECK(Patch)

	*Major=ROCM_VERSION_MAJOR;
	*Minor=ROCM_VERSION_MINOR;
	*Patch=ROCM_VERSION_PATCH;

	return	VerSuccess;
}

static VerErrors getBuildInfo( char* InfoString, int length_of_the_buffer ) {
	NULL_CHECK(InfoString);
#if defined(ROCM_BUILD_INFO)
	if(length_of_the_buffer<=strlen(ROCM_BUILD_INFO)){
		fprintf(stderr, "\n Error :: Buffer is less than adequate size required\n");
		fflush(stderr);
		return	VerIncorrectParameters;
	}
	snprintf(InfoString, length_of_the_buffer, "%s", ROCM_BUILD_INFO);
	return	VerSuccess;
#else
	return VerValuesNotDefined;
#endif //end definition checker
}

VerErrors printBuildInfo() {
	int len_of_buffer_to_be_created = 0;
	VerErrors apiret=VerSuccess;

#if defined(ROCM_BUILD_INFO)
	len_of_buffer_to_be_created = 1 + strlen(ROCM_BUILD_INFO);//additional char for null termination
#else
	return VerValuesNotDefined;
#endif //end definition checker

	// len_of_buffer_to_be_created is now strlen(ROCM_BUILD_INFO) + 1
	char* cstr=(char*) malloc(len_of_buffer_to_be_created);
	if(cstr){
		apiret=getBuildInfo(cstr,len_of_buffer_to_be_created);
		if(VerSuccess != apiret) {
			free(cstr);
		}
		CHECK_AND_REPORT_API_RESULT(apiret);
		printf("\n Build Info of lib = [%s] \n",cstr);
		//All good lets free
		free(cstr);
		return VerSuccess;
	}
	fprintf(stderr, "\n malloc failed for length [%d] \n",len_of_buffer_to_be_created);
	fflush(stderr);
	return VerMemoryAllocationFailed;
}
