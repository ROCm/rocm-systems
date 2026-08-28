//---------------------------------------------------------------------
// CRC32
//
// Calculate a 32bit crc using a the Sarwate look up table method. The original algorithm was created by
// Dilip V. Sarwate, and is based off of Stephan Brumme's implementation. See also:
// https://dl.acm.org/citation.cfm?doid=63030.63037
// http://create.stephan-brumme.com/crc32/#sarwate
//
//// Copyright (c) 2011-2016 Stephan Brumme. All rights reserved.
//*****************************************************************************************************************
// * This software is provided 'as-is', without any express or implied warranty. In no event will the author be held
// * liable for any damages arising from the use of this software. Permission is granted to anyone to use this
// * software for any purpose, including commercial applications, and to alter it and redistribute it freely,
// * subject to the following restrictions:
// *    1. The origin of this software must not be misrepresented; you must not claim that you wrote the original
// *         software
// *    2. If you use this software in a product, an acknowledgment in the product documentation would be
// *         appreciated but is not required.
// *    3. Altered source versions must be plainly marked as such, and must not be misrepresented as being the
// *         original software.
// *****************************************************************************************************************
//
// Copyright (c) 2004-2006 Intel Corporation - All Rights Reserved
//
// This software program is licensed subject to the BSD License,
// available at http://www.opensource.org/licenses/bsd-license.html.
//
//
// Tables for software CRC generation
//

#include <stddef.h>
#include <stdint.h>

uint32_t CRC32(const void* pData, size_t length, uint32_t lastCRC = 0);
