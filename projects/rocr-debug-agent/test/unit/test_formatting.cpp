/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "agent_utils.h"

#include <gtest/gtest.h>

using namespace amd::debug_agent;

TEST (HexStringTest, EmptyVector)
{
  std::vector<uint8_t> empty;
  EXPECT_EQ (hex_string (empty), "");
}

TEST (HexStringTest, SingleByte)
{
  std::vector<uint8_t> data = { 0xAB };
  EXPECT_EQ (hex_string (data), "ab");
}

TEST (HexStringTest, MultipleBytes_LittleEndian)
{
  std::vector<uint8_t> data = { 0x12, 0x34, 0x56 };
  /* Little-endian: bytes are reversed in output. */
  EXPECT_EQ (hex_string (data), "563412");
}

TEST (HexStringTest, AllZeros)
{
  std::vector<uint8_t> data = { 0x00, 0x00, 0x00 };
  EXPECT_EQ (hex_string (data), "000000");
}

TEST (HexStringTest, AllOnes)
{
  std::vector<uint8_t> data = { 0xFF, 0xFF };
  EXPECT_EQ (hex_string (data), "ffff");
}

TEST (RegisterValueStringTest, ScalarType_TwoBytes)
{
  std::vector<uint8_t> data = { 0xAB, 0xCD };
  EXPECT_EQ (register_value_string ("int16_t", data), "cdab");
}

TEST (RegisterValueStringTest, ScalarType_FourBytes)
{
  std::vector<uint8_t> data = { 0x12, 0x34, 0x56, 0x78 };
  EXPECT_EQ (register_value_string ("int32_t", data), "78563412");
}

TEST (RegisterValueStringTest, VectorType_Int2)
{
  std::vector<uint8_t> data = { 0x12, 0x34, 0x56, 0x78 };
  std::string result = register_value_string ("int[2]", data);
  EXPECT_TRUE (result.find ("[0]") != std::string::npos);
  EXPECT_TRUE (result.find ("[1]") != std::string::npos);
  EXPECT_TRUE (result.find ("3412") != std::string::npos);
  EXPECT_TRUE (result.find ("7856") != std::string::npos);
}

TEST (RegisterValueStringTest, VectorType_Int4)
{
  std::vector<uint8_t> data = { 0x01, 0x02, 0x03, 0x04,
                                0x05, 0x06, 0x07, 0x08 };
  std::string result = register_value_string ("int[4]", data);
  EXPECT_TRUE (result.find ("[0]") != std::string::npos);
  EXPECT_TRUE (result.find ("[1]") != std::string::npos);
  EXPECT_TRUE (result.find ("[2]") != std::string::npos);
  EXPECT_TRUE (result.find ("[3]") != std::string::npos);
}

TEST (RegisterValueStringTest, EmptyVector)
{
  std::vector<uint8_t> empty;
  EXPECT_EQ (register_value_string ("int", empty), "");
}
