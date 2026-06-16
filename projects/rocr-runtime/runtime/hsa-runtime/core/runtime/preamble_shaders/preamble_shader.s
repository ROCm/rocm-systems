/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

 /* Currently using a blank preamble shader that just jumps to the actual shader */

.text

.set JUMP_ADDRESS_HI , 0x12345678
.set JUMP_ADDRESS_LO , 0x9ABCDEF0

Main:
    s_mov_b32 s100, JUMP_ADDRESS_LO
    s_mov_b32 s101, JUMP_ADDRESS_HI
    s_setpc_b64 s[100:101]
    // Add s_code_end padding so instruction prefetch always has something to read.
    .rept (256 - ((. - Main) % 64)) / 4
        s_code_end
    .endr

// This generates a HEX dump of:
// 00000000: ff 00 e4 be f0 de bc 9a ff 00 e5 be 78 56 34 12
// 00000010: 64 48 80 be 00 00 9f bf 00 00 9f bf 00 00 9f bf
// :
// 000000f0: 00 00 9f bf 00 00 9f bf 00 00 9f bf 00 00 9f bf
// 
//
// So Address-HI is at offset 12 bytes
// And Address-LO is at offset 4 bytes
//