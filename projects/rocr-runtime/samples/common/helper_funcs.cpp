/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "helper_funcs.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif


/*
 * Prints no more than 256 elements of the given array.
 * Prints full array if length is less than 256.
 * Prints Array name followed by elements.
 */
template<typename T> 
void printArray(
    const std::string header, 
    const T * data, 
    const int width,
    const int height)
{
    std::cout<<"\n"<<header<<"\n";
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            std::cout<<data[i*width+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

template<typename T> 
int fillRandom(
         T * arrayPtr, 
         const int width,
         const int height,
         const T rangeMin,
         const T rangeMax,
         unsigned int seed)
{
    if(!arrayPtr)
    {
        error("Cannot fill array. NULL pointer.");
        return HSA_SDK_FAILURE;
    }

    if(!seed)
        seed = (unsigned int)time(NULL);

    srand(seed);
    double range = double(rangeMax - rangeMin) + 1.0; 

    /* random initialisation of input */
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int index = i*width + j;
            arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0)); 
        }

    return HSA_SDK_SUCCESS;
}

template<typename T> 
int fillPos(
         T * arrayPtr, 
         const int width,
         const int height)
{
    if(!arrayPtr)
    {
        error("Cannot fill array. NULL pointer.");
        return HSA_SDK_FAILURE;
    }

    /* initialisation of input with positions*/
    for(T i = 0; i < height; i++)
        for(T j = 0; j < width; j++)
        {
            T index = i*width + j;
            arrayPtr[index] = index;
        }

    return HSA_SDK_SUCCESS;
}

template<typename T> 
int fillConstant(
         T * arrayPtr, 
         const int width,
         const int height,
         const T val)
{
    if(!arrayPtr)
    {
        error("Cannot fill array. NULL pointer.");
        return HSA_SDK_FAILURE;
    }

    /* initialisation of input with constant value*/
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int index = i*width + j;
            arrayPtr[index] = val;
        }

    return HSA_SDK_SUCCESS;
}

template<typename T>
T roundToPowerOf2(T val)
{
    int bytes = sizeof(T);

    val--;
    for(int i = 0; i < bytes; i++)
        val |= val >> (1<<i);  
    val++;

    return val;
}

template<typename T>
int isPowerOf2(T val)
{
    long long _val = val;
    if((_val & (-_val))-_val == 0 && _val != 0)
        return HSA_SDK_SUCCESS;
    else
        return HSA_SDK_FAILURE;
}



template<typename T>
bool checkVal(
    T input, 
    T reference, 
    std::string message,
    bool isAPIerror)
{
    if(input==reference)
    {
        return true;
    }
    else
    {
        error(message);   
        return false;
    }
}


template<typename T>
std::string toString(T t, std::ios_base &(*r)(std::ios_base&))
{
  std::ostringstream output;
  output << r << t;
  return output.str();
}


bool
compare(const float *refData, const float *data, 
                        const int length, const float epsilon)
{
    float error = 0.0f;
    float ref = 0.0f;

    for(int i = 1; i < length; ++i) 
    {
        float diff = refData[i] - data[i];
        error += diff * diff;
        ref += refData[i] * refData[i];
    }

    float normRef =::sqrtf((float) ref);
    if (::fabs((float) ref) < 1e-7f) {
        return false;
    }
    float normError = ::sqrtf((float) error);
    error = normError / normRef;

    return error < epsilon;
}

bool
compare(const double *refData, const double *data, 
                        const int length, const double epsilon)
{
    double error = 0.0;
    double ref = 0.0;

    for(int i = 1; i < length; ++i) 
    {
        double diff = refData[i] - data[i];
        error += diff * diff;
        ref += refData[i] * refData[i];
    }

    double normRef =::sqrt((double) ref);
    if (::fabs((double) ref) < 1e-7) {
        return false;
    }
    double normError = ::sqrt((double) error);
    error = normError / normRef;

    return error < epsilon;
}

void 
error(const char* errorMsg)
{
    std::cout<<"Error: "<<errorMsg<<std::endl;
}

void 
error(std::string errorMsg)
{
    std::cout<<"Error: "<<errorMsg<<std::endl;
}

void 
expectedError(const char* errorMsg)
{
    std::cout<<"Expected Error: "<<errorMsg<<std::endl;
}

void 
expectedError(std::string errorMsg)
{
    std::cout<<"Expected Error: "<<errorMsg<<std::endl;
}


/////////////////////////////////////////////////////////////////
// Template Instantiations 
/////////////////////////////////////////////////////////////////
template 
void printArray<short>(const std::string, 
        const short*, int, int);
template 
void printArray<unsigned char>(const std::string, 
        const unsigned char *, int, int);
template 
void printArray<unsigned int>(const std::string, 
        const unsigned int *, int, int);
template 
void printArray<int>(const std::string, 
        const int *, int, int);
template 
void printArray<long>(const std::string, 
        const long*, int, int);
template 
void printArray<float>(const std::string, 
        const float*, int, int);
template 
void printArray<double>(const std::string, 
        const double*, int, int);

template 
int fillRandom<unsigned char>(unsigned char* arrayPtr, 
        const int width, const int height, 
        unsigned char rangeMin, unsigned char rangeMax, unsigned int seed);	
template 
int fillRandom<unsigned int>(unsigned int* arrayPtr, 
        const int width, const int height, 
        unsigned int rangeMin, unsigned int rangeMax, unsigned int seed);	
template 
int fillRandom<int>(int* arrayPtr, 
        const int width, const int height, 
        int rangeMin, int rangeMax, unsigned int seed);	
template 
int fillRandom<long>(long* arrayPtr, 
        const int width, const int height, 
        long rangeMin, long rangeMax, unsigned int seed);	
template 
int fillRandom<float>(float* arrayPtr, 
        const int width, const int height, 
        float rangeMin, float rangeMax, unsigned int seed);	
template 
int fillRandom<double>(double* arrayPtr, 
        const int width, const int height, 
        double rangeMin, double rangeMax, unsigned int seed);	

template 
short roundToPowerOf2<short>(short val);
template 
unsigned int roundToPowerOf2<unsigned int>(unsigned int val);
template 
int roundToPowerOf2<int>(int val);
template 
long roundToPowerOf2<long>(long val);

template
int isPowerOf2<short>(short val);
template
int isPowerOf2<unsigned int>(unsigned int val);
template
int isPowerOf2<int>(int val);
template
int isPowerOf2<long>(long val);

template<> 
int fillPos<short>(short * arrayPtr, const int width, const int height);
template<> 
int fillPos<unsigned int>(unsigned int * arrayPtr, const int width, const int height);
template<> 
int fillPos<int>(int * arrayPtr, const int width, const int height);
template<> 
int fillPos<long>(long * arrayPtr, const int width, const int height);

template<> 
int fillConstant<short>(short * arrayPtr, 
        const int width, const int height, 
        const short val);
template<> 
int fillConstant(unsigned int * arrayPtr, 
        const int width, const int height, 
        const unsigned int val);
template<> 
int fillConstant(int * arrayPtr, 
        const int width, const int height, 
        const int val);
template<> 
int fillConstant(long * arrayPtr, 
        const int width, const int height, 
        const long val);
template<> 
int fillConstant(long * arrayPtr, 
        const int width, const int height, 
        const long val);
template<> 
int fillConstant(long * arrayPtr, 
        const int width, const int height, 
        const long val);


template
bool checkVal<char>(char input, char reference, std::string message, bool isAPIerror);
template
bool checkVal<bool>(bool input, bool reference, std::string message, bool isAPIerror);
template
bool checkVal<std::string>(std::string input, std::string reference, std::string message, bool isAPIerror);
template
bool checkVal<short>(short input, short reference, std::string message, bool isAPIerror);
template
bool checkVal<unsigned int>(unsigned int  input, unsigned int  reference, std::string message, bool isAPIerror);
template
bool checkVal<int>(int input, int reference, std::string message, bool isAPIerror);
template
bool checkVal<long>(long input, long reference, std::string message, bool isAPIerror);


template
std::string toString<char>(char t, std::ios_base &(*r)(std::ios_base&));
template
std::string toString<short>(short t, std::ios_base &(*r)(std::ios_base&));
template
std::string toString<unsigned int>(unsigned int t, std::ios_base &(*r)(std::ios_base&));
template
std::string toString<int>(int t, std::ios_base &(*r)(std::ios_base&));
template
std::string toString<long>(long t, std::ios_base &(*r)(std::ios_base&));
template
std::string toString<float>(float t, std::ios_base &(*r)(std::ios_base&));
template
std::string toString<double>(double t, std::ios_base &(*r)(std::ios_base&));

