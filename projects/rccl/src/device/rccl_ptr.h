#pragma once

/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Defines a series of global address space pointers.  Casting to these
// pointers in hot code paths should improve performance since global
// aperture vector instrutions like global_store_dwordx4 can be used.
// These are cheaper than flat loads and stores.  
// Verify the intended effect by inspecting assembly.  If you see 
// flat in the name of the emitted instruction, something is wrong.
using u64_gptr = __attribute__((address_space(1))) uint64_t*;
using u32_gptr = __attribute__((address_space(1))) uint32_t*;
using u16_gptr = __attribute__((address_space(1))) uint16_t*;
using u8_gptr = __attribute__((address_space(1))) uint8_t*;

