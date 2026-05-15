/*****************************************************************************\
* 
* *** NEED TO REPLACE THIS WITH OPEN SOURCE HEADER ***
* 
*  
*  Copyright (c) 2013 Advanced Micro Devices, Inc.
*
*
\*****************************************************************************/

#ifndef _CZ_ID_H
#define _CZ_ID_H

enum {
    CARRIZO_A1      = 0x02,
    CARRIZO_BRISTOL_A0 = 0x10,
    STONEY_A0       = 0x61,
    CZ_UNKNOWN      = 0xFF
};


// for the original CARRIZO + other CARRIZO kickers
#define ASICREV_IS_CARRIZO(eChipRev)            (eChipRev == CARRIZO_A1)

//
// Carrizo device IDs 
//
#define DEVICE_ID_CZ_9874                   0x9874  

// CARRIZO ASIC internal revision number
#define INTERNAL_REV_CARRIZO_A1             0x01

// CARRIZO PCI Revision IDs
#define PRID_CZ_C4                          0xC4  // Client B10
#define PRID_CZ_C5                          0xC5  // Client B8
#define PRID_CZ_C6                          0xC6  // Client B6
#define PRID_CZ_C7                          0xC7  // Client B4

// eCARRIZO PCI Revision IDs
#define PRID_eCZ_84                          0x84  // eCZ B10
#define PRID_eCZ_81                          0x81  // eCZ B8/iTemp
#define PRID_eCZ_85                          0x85  // eCZ B8
#define PRID_eCZ_87                          0x87  // eCZ B4

#endif // _CZ_ID_H

