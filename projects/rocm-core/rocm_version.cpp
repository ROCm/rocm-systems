#include "rocm_version.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#define NULL_CHECK(ptr)	if(!ptr) return VerIncorrecPararmeters;


#define CHECK_AND_REPORT_API_RESULT(val)	do {					\
							if(VerSuccess != val) {	\
								const char *ErrStrings[VerErrorMAX]= { "VerSuccess", "VerIncorrecPararmeters", "VerValuesNotDefined" }; 	\
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



static VerErrors getBuildInfoLen( int* InfoStrlen ) {

	NULL_CHECK(InfoStrlen);
#if defined(ROCM_BUILD_INFO)
	*InfoStrlen = 1 + strlen(ROCM_BUILD_INFO);//additional char for null termination
#else
	return VerValuesNotDefined;
#endif //end defination checker
	return	VerSuccess;
}

static VerErrors getBuildInfo( char* InfoString, int len ) {

	NULL_CHECK(InfoString);
#if defined(ROCM_BUILD_INFO)

	strcpy(InfoString,ROCM_BUILD_INFO);
	InfoString[len]='\0';
#else
	return VerValuesNotDefined;
#endif //end defination checker
	return	VerSuccess;
}

VerErrors printBuildInfo() {

	int lenstr=0;
        VerErrors apiret=VerSuccess;

	apiret=getBuildInfoLen(&lenstr);
	CHECK_AND_REPORT_API_RESULT(apiret);

	char* cstr=(char*) malloc(lenstr*sizeof(char));
        apiret=getBuildInfo(cstr,lenstr);
	CHECK_AND_REPORT_API_RESULT(apiret);

	printf("\n Build Info of lib = [%s] \n",cstr);

        free(cstr);

	return VerSuccess;
}
