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
#ifndef HELPER_FUNCS_HPP_
#define HELPER_FUNCS_HPP_

#define HSA_SDK_SUCCESS 0
#define HSA_SDK_FAILURE 1
#define HSA_SDK_EXPECTED_FAILURE 2

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <ctime>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <malloc.h>

/**
 * error
 * constant function, Prints error messages 
 * @param errorMsg char* message
 */
void error(const char* errorMsg);	

/**
 * error
 * constant function, Prints error messages 
 * @param errorMsg std::string message
 */
void error(std::string errorMsg);

/**
 * expectedError
 * constant function, Prints error messages 
 * @param errorMsg char* message
 */
void expectedError(const char* errorMsg);	

/**
 * expectedError
 * constant function, Prints error messages 
 * @param errorMsg string message
 */
void expectedError(std::string errorMsg);

/**
 * compare template version
 * compare data to check error
 * @param refData templated input
 * @param data templated input
 * @param length number of values to compare
 * @param epsilon errorWindow
 */
bool compare(const float *refData, const float *data, 
        const int length, const float epsilon = 1e-6f); 
bool compare(const double *refData, const double *data, 
        const int length, const double epsilon = 1e-6); 

/**
 * printArray
 * displays a array on std::out
 */
template<typename T> 
void printArray(
     const std::string header,
     const T * data, 
     const int width,
     const int height);


/**
 * fillRandom
 * fill array with random values
 */
template<typename T> 
int fillRandom(
     T * arrayPtr, 
     const int width,
     const int height,
     const T rangeMin,
     const T rangeMax,
     unsigned int seed=123);	
  
/**
 * fillPos
 * fill the specified positions
 */
template<typename T> 
int fillPos(
     T * arrayPtr, 
     const int width,
     const int height);
  
/**
 * fillConstant
 * fill the array with constant value
 */
template<typename T> 
int fillConstant(
     T * arrayPtr, 
     const int width,
     const int height,
     const T val);

  
/**
 * roundToPowerOf2
 * rounds to a power of 2
 */
template<typename T>
T roundToPowerOf2(T val);

/**
 * isPowerOf2
 * checks if input is a power of 2
 */
template<typename T>
int isPowerOf2(T val);
  
/**
 * checkVal
 * Set default(isAPIerror) parameter to false 
 * if checkVaul is used to check otherthan OpenCL API error code 
 */
template<typename T> 
bool checkVal(
  T input, 
  T reference, 
  std::string message, bool isAPIerror = true);

/**
 * toString
 * convert a T type to string
 */
template<typename T>
std::string toString(T t, std::ios_base & (*r)(std::ios_base&)); 




#endif
