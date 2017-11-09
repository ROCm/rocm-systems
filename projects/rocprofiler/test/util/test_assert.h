/******************************************************************************

Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

#ifndef TEST_CTRL_TEST_ASSERT_H_
#define TEST_CTRL_TEST_ASSERT_H_

#define TEST_ASSERT(cond)                                                                          \
  {                                                                                                \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assert failed(" << #cond << ") at " << __FILE__ << ", line " << __LINE__       \
                << std::endl;                                                                      \
      exit(-1);                                                                                    \
    }                                                                                              \
  }

#define TEST_STATUS(cond)                                                                          \
  {                                                                                                \
    if (!(cond)) {                                                                                 \
      std::cerr << "Test error at " << __FILE__ << ", line " << __LINE__                           \
                << std::endl;                                                                      \
      const char* message;                                                                         \
      rocprofiler_error_string(&message);                                                          \
      std::cerr << "ERROR: " << message << std::endl;                                              \
      exit(-1);                                                                                    \
    }                                                                                              \
  }

#endif  // TEST_CTRL_TEST_ASSERT_H_
