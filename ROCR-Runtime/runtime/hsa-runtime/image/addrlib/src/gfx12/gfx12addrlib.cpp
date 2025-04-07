/*
************************************************************************************************************************
*
*  Copyright (C) 2023 Advanced Micro Devices, Inc.  All rights reserved.
*  SPDX-License-Identifier: MIT
*
***********************************************************************************************************************/

/**
************************************************************************************************************************
* @file  gfx12addrlib.cpp
* @brief Contain the implementation for the Gfx12Lib class.
************************************************************************************************************************
*/

#include "gfx12addrlib.h"
#include "gfx12_gb_reg.h"

#include "amdgpu_asic_addr.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace rocr {
namespace Addr
{
/**
************************************************************************************************************************
*   Gfx12HwlInit
*
*   @brief
*       Creates an Gfx12Lib object.
*
*   @return
*       Returns an Gfx12Lib object pointer.
************************************************************************************************************************
*/
Addr::Lib* Gfx12HwlInit(
    const Client* pClient)
{
    return V3::Gfx12Lib::CreateObj(pClient);
}

namespace V3
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               Static Const Member
////////////////////////////////////////////////////////////////////////////////////////////////////
const SwizzleModeFlags Gfx12Lib::SwizzleModeTable[ADDR3_MAX_TYPE] =
{//Linear 2d   3d  256B  4KB  64KB  256KB  Reserved
    {{1,   0,   0,    0,   0,    0,     0,    0}}, // ADDR3_LINEAR
    {{0,   1,   0,    1,   0,    0,     0,    0}}, // ADDR3_256B_2D
    {{0,   1,   0,    0,   1,    0,     0,    0}}, // ADDR3_4KB_2D
    {{0,   1,   0,    0,   0,    1,     0,    0}}, // ADDR3_64KB_2D
    {{0,   1,   0,    0,   0,    0,     1,    0}}, // ADDR3_256KB_2D
    {{0,   0,   1,    0,   1,    0,     0,    0}}, // ADDR3_4KB_3D
    {{0,   0,   1,    0,   0,    1,     0,    0}}, // ADDR3_64KB_3D
    {{0,   0,   1,    0,   0,    0,     1,    0}}, // ADDR3_256KB_3D
};

const ADDR_EXTENT3D Gfx12Lib::Block4K_Log2_3d[]   = {{4, 4, 4}, {3, 4, 4}, {3, 4, 3}, {3, 3, 3}, {2, 3, 3}};
const ADDR_EXTENT3D Gfx12Lib::Block64K_Log2_3d[]  = {{6, 5, 5}, {5, 5, 5}, {5, 5, 4}, {5, 4, 4}, {4, 4, 4}};
const ADDR_EXTENT3D Gfx12Lib::Block256K_Log2_3d[] = {{6, 6, 6}, {5, 6, 6}, {5, 6, 5}, {5, 5, 5}, {4, 5, 5}};

/**
************************************************************************************************************************
*   Gfx12Lib::Gfx12Lib
*
*   @brief
*       Constructor
*
************************************************************************************************************************
*/
Gfx12Lib::Gfx12Lib(
    const Client* pClient)
    :
    Lib(pClient),
    m_numSwizzleBits(0)
{
    memset(&m_settings, 0, sizeof(m_settings));
    memcpy(m_swizzleModeTable, SwizzleModeTable, sizeof(SwizzleModeTable));
}

/**
************************************************************************************************************************
*   Gfx12Lib::~Gfx12Lib
*
*   @brief
*       Destructor
************************************************************************************************************************
*/
Gfx12Lib::~Gfx12Lib()
{
}

/**
************************************************************************************************************************
*   Gfx12Lib::ConvertSwizzlePatternToEquation
*
*   @brief
*       Convert swizzle pattern to equation.
*
*   @return
*       N/A
************************************************************************************************************************
*/
VOID Gfx12Lib::ConvertSwizzlePatternToEquation(
    UINT_32                elemLog2,  ///< [in] element bytes log2
    Addr3SwizzleMode       swMode,    ///< [in] swizzle mode
    const ADDR_SW_PATINFO* pPatInfo,  ///< [in] swizzle pattern info
    ADDR_EQUATION*         pEquation) ///< [out] equation converted from swizzle pattern
    const
{
    ADDR_BIT_SETTING fullSwizzlePattern[Log2Size256K];
    GetSwizzlePatternFromPatternInfo(pPatInfo, fullSwizzlePattern);

    const ADDR_BIT_SETTING* pSwizzle = fullSwizzlePattern;
    const UINT_32           blockSizeLog2 = GetBlockSizeLog2(swMode, TRUE);

    pEquation->numBits = blockSizeLog2;
    pEquation->stackedDepthSlices = FALSE;

    for (UINT_32 i = 0; i < elemLog2; i++)
    {
        pEquation->addr[i].channel = 0;
        pEquation->addr[i].valid = 1;
        pEquation->addr[i].index = i;
    }

    for (UINT_32 i = elemLog2; i < blockSizeLog2; i++)
    {
        ADDR_ASSERT(IsPow2(pSwizzle[i].value));

        if (pSwizzle[i].x != 0)
        {
            ADDR_ASSERT(IsPow2(static_cast<UINT_32>(pSwizzle[i].x)));

            pEquation->addr[i].channel = 0;
            pEquation->addr[i].valid = 1;
            pEquation->addr[i].index = Log2(pSwizzle[i].x) + elemLog2;
        }
        else if (pSwizzle[i].y != 0)
        {
            ADDR_ASSERT(IsPow2(static_cast<UINT_32>(pSwizzle[i].y)));

            pEquation->addr[i].channel = 1;
            pEquation->addr[i].valid = 1;
            pEquation->addr[i].index = Log2(pSwizzle[i].y);
        }
        else if (pSwizzle[i].z != 0)
        {
            ADDR_ASSERT(IsPow2(static_cast<UINT_32>(pSwizzle[i].z)));

            pEquation->addr[i].channel = 2;
            pEquation->addr[i].valid = 1;
            pEquation->addr[i].index = Log2(pSwizzle[i].z);
        }
        else if (pSwizzle[i].s != 0)
        {
            ADDR_ASSERT(IsPow2(static_cast<UINT_32>(pSwizzle[i].s)));

            pEquation->addr[i].channel = 3;
            pEquation->addr[i].valid = 1;
            pEquation->addr[i].index = Log2(pSwizzle[i].s);
        }
        else
        {
            ADDR_ASSERT_ALWAYS();
        }
    }
}

/**
************************************************************************************************************************
*   Gfx12Lib::InitEquationTable
*
*   @brief
*       Initialize Equation table.
*
*   @return
*       N/A
************************************************************************************************************************
*/
VOID Gfx12Lib::InitEquationTable()
{
    memset(m_equationTable, 0, sizeof(m_equationTable));

    for (UINT_32 swModeIdx = 0; swModeIdx < ADDR3_MAX_TYPE; swModeIdx++)
    {
        const Addr3SwizzleMode swMode = static_cast<Addr3SwizzleMode>(swModeIdx);

        if (IsLinear(swMode))
        {
            // Skip linear equation (data table is not useful for 2D/3D images-- only contains x-coordinate bits)
            continue;
        }

        const UINT_32 maxMsaa = Is2dSwizzle(swMode) ? MaxMsaaRateLog2 : 1;

        for (UINT_32 msaaIdx = 0; msaaIdx < maxMsaa; msaaIdx++)
        {
            for (UINT_32 elemLog2 = 0; elemLog2 < MaxElementBytesLog2; elemLog2++)
            {
                UINT_32                equationIndex = ADDR_INVALID_EQUATION_INDEX;
                const ADDR_SW_PATINFO* pPatInfo = GetSwizzlePatternInfo(swMode, elemLog2, 1 << msaaIdx);

                if (pPatInfo != NULL)
                {
                    ADDR_ASSERT(IsValidSwMode(swMode));

                    ADDR_EQUATION equation = {};

                    ConvertSwizzlePatternToEquation(elemLog2, swMode, pPatInfo, &equation);

                    equationIndex = m_numEquations;
                    ADDR_ASSERT(equationIndex < NumSwizzlePatterns);

                    m_equationTable[equationIndex] = equation;
                    m_numEquations++;
                }
                SetEquationTableEntry(swMode, msaaIdx, elemLog2, equationIndex);
            }
        }
    }
}

/**
************************************************************************************************************************
*   Gfx12Lib::GetBlockPixelDimensions
*
*   @brief
*       Returns the pixel dimensions of one block.
*
************************************************************************************************************************
*/
ADDR_EXTENT3D  Gfx12Lib::GetBlockPixelDimensions(
    Addr3SwizzleMode  swizzleMode,
    UINT_32           log2BytesPerPixel
    ) const
{
    ADDR_EXTENT3D  log2Dim = {};

    switch (swizzleMode)
    {
        case ADDR3_4KB_3D:
            log2Dim = Block4K_Log2_3d[log2BytesPerPixel];
            break;
        case ADDR3_64KB_3D:
            log2Dim = Block64K_Log2_3d[log2BytesPerPixel];
            break;
        case ADDR3_256KB_3D:
            log2Dim = Block256K_Log2_3d[log2BytesPerPixel];
            break;
        default:
            ADDR_ASSERT_ALWAYS();
            break;
    }

    return { 1u << log2Dim.width, 1u << log2Dim.height, 1u << log2Dim.depth };
}

/**
************************************************************************************************************************
*   Gfx12Lib::GetMipOrigin
*
*   @brief
*       Internal function to calculate origins of the mip levels
*
*   @return
*       ADDR_E_RETURNCODE
************************************************************************************************************************
*/
VOID Gfx12Lib::GetMipOrigin(
     const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn,        ///< [in] input structure
     const ADDR_EXTENT3D&                    mipExtentFirstInTail,
     ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*      pOut        ///< [out] output structure
     ) const
{
    const BOOL_32        is3d           = Is3dSwizzle(pIn->swizzleMode);
    const UINT_32        bytesPerPixel  = pIn->bpp >> 3;
    const UINT_32        log2Bpp        = Log2(bytesPerPixel);
    const ADDR_EXTENT3D  pixelBlockDims = GetBlockPixelDimensions(ADDR3_4KB_3D, log2Bpp);
    const ADDR_EXTENT3D  tailMaxDim     = GetMipTailDim(pIn->swizzleMode,
                                                        pOut->blockExtent);
    const UINT_32        blockSizeLog2  = GetBlockSizeLog2(pIn->swizzleMode);
    const UINT_32        maxMipsInTail  = GetMaxNumMipsInTail(pIn->swizzleMode, blockSizeLog2);

    UINT_32 pitch  = tailMaxDim.width;
    UINT_32 height = tailMaxDim.height;

    UINT_32 depth  = (is3d ? PowTwoAlign(mipExtentFirstInTail.depth, pixelBlockDims.depth) : 1);

    const UINT_32 tailMaxDepth   = (is3d ? (depth / pixelBlockDims.depth) : 1);

    for (UINT_32 i = pOut->firstMipIdInTail; i < pIn->numMipLevels; i++)
    {
        INT_32  mipInTail = static_cast<INT_32>(i) - static_cast<INT_32>(pOut->firstMipIdInTail);
        if ((mipInTail < 0) || (pIn->numMipLevels == 1))
        {
            mipInTail = MaxMipLevels;
        }

        // "m" can be negative
        const INT_32  signedM   = static_cast<INT_32>(maxMipsInTail) - static_cast<INT_32>(1) - mipInTail;
        const UINT_32 m         = Max(0, signedM);
        const UINT_32 mipOffset = (m > 6) ? (16 << m) : (m << 8);

        pOut->pMipInfo[i].offset           = mipOffset * tailMaxDepth;
        pOut->pMipInfo[i].mipTailOffset    = mipOffset;
        pOut->pMipInfo[i].macroBlockOffset = 0;

        pOut->pMipInfo[i].pitch  = pitch;
        pOut->pMipInfo[i].height = height;
        pOut->pMipInfo[i].depth  = depth;

        if (IsLinear(pIn->swizzleMode))
        {
            pOut->pMipInfo[i].mipTailCoordX = mipOffset >> 8;
            pOut->pMipInfo[i].mipTailCoordY = 0;
            pOut->pMipInfo[i].mipTailCoordZ = 0;

            pitch = Max(pitch >> 1, 1u);
        }
        else
        {
            UINT_32 mipX = ((mipOffset >> 9)  & 1)  |
                           ((mipOffset >> 10) & 2)  |
                           ((mipOffset >> 11) & 4)  |
                           ((mipOffset >> 12) & 8)  |
                           ((mipOffset >> 13) & 16) |
                           ((mipOffset >> 14) & 32);
            UINT_32 mipY = ((mipOffset >> 8)  & 1)  |
                           ((mipOffset >> 9)  & 2)  |
                           ((mipOffset >> 10) & 4)  |
                           ((mipOffset >> 11) & 8)  |
                           ((mipOffset >> 12) & 16) |
                           ((mipOffset >> 13) & 32);

            if (is3d == FALSE)
            {
                pOut->pMipInfo[i].mipTailCoordX = mipX * Block256_2d[log2Bpp].w;
                pOut->pMipInfo[i].mipTailCoordY = mipY * Block256_2d[log2Bpp].h;
                pOut->pMipInfo[i].mipTailCoordZ = 0;

                pitch  = Max(pitch  >> 1, Block256_2d[log2Bpp].w);
                height = Max(height >> 1, Block256_2d[log2Bpp].h);
                depth  = 1;
            }
            else
            {
                pOut->pMipInfo[i].mipTailCoordX = mipX * pixelBlockDims.width;
                pOut->pMipInfo[i].mipTailCoordY = mipY * pixelBlockDims.height;
                pOut->pMipInfo[i].mipTailCoordZ = 0;

                pitch  = Max(pitch  >> 1, pixelBlockDims.width);
                height = Max(height >> 1, pixelBlockDims.height);
                depth  = PowTwoAlign(Max(depth >> 1, 1u), pixelBlockDims.depth);
            }
        }
    }
}

/**
************************************************************************************************************************
*   Gfx12Lib::GetMipOffset
*
*   @brief
*       Internal function to calculate alignment for a surface
*
*   @return
*       ADDR_E_RETURNCODE
************************************************************************************************************************
*/
VOID Gfx12Lib::GetMipOffset(
     const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn,    ///< [in] input structure
     ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*      pOut    ///< [out] output structure
     ) const
{
    const UINT_32        bytesPerPixel = pIn->bpp >> 3;
    const UINT_32        log2Bpp       = Log2(bytesPerPixel);
    const UINT_32        blockSizeLog2 = GetBlockSizeLog2(pIn->swizzleMode);
    const UINT_32        blockSize     = 1 << blockSizeLog2;
    const ADDR_EXTENT3D  tailMaxDim    = GetMipTailDim(pIn->swizzleMode,
                                                       pOut->blockExtent);
    const ADDR_EXTENT3D  mip0Dims      = GetBaseMipExtents(pIn);
    const UINT_32        maxMipsInTail = GetMaxNumMipsInTail(pIn->swizzleMode, blockSizeLog2);

    UINT_32       firstMipInTail    = pIn->numMipLevels;
    UINT_64       mipChainSliceSize = 0;
    UINT_64       mipSize[MaxMipLevels];
    UINT_64       mipSliceSize[MaxMipLevels];

    const ADDR_EXTENT3D fixedTailMaxDim = tailMaxDim;

    for (UINT_32 mipIdx = 0; mipIdx < pIn->numMipLevels; mipIdx++)
    {
        const ADDR_EXTENT3D  mipExtents = GetMipExtent(mip0Dims, mipIdx);

        if (SupportsMipTail(pIn->swizzleMode) &&
            IsInMipTail(fixedTailMaxDim, mipExtents, maxMipsInTail, pIn->numMipLevels - mipIdx))
        {
            firstMipInTail     = mipIdx;
            mipChainSliceSize += blockSize / pOut->blockExtent.depth;
            break;
        }
        else
        {
            const UINT_32 pitch  = UseCustomPitch(pIn)
                                        ? pOut->pitch
                                        : ((mipIdx == 0) && CanTrimLinearPadding(pIn))
                                          ? PowTwoAlign(mipExtents.width,  128u / bytesPerPixel)
                                          : PowTwoAlign(mipExtents.width,  pOut->blockExtent.width);
            const UINT_32 height = UseCustomHeight(pIn)
                                        ? pOut->height
                                        : PowTwoAlign(mipExtents.height, pOut->blockExtent.height);
            const UINT_32 depth  = PowTwoAlign(mipExtents.depth,  pOut->blockExtent.depth);

            // The original "blockExtent" calculation does subtraction of logs (i.e., division) to get the
            // sizes.  We aligned our pitch and height to those sizes, which means we need to multiply the various
            // factors back together to get back to the slice size.
            const UINT_64 sliceSize = static_cast<UINT_64>(pitch) * height * pIn->numSamples * (pIn->bpp >> 3);

            mipSize[mipIdx]       = sliceSize * depth;
            mipSliceSize[mipIdx]  = sliceSize * pOut->blockExtent.depth;
            mipChainSliceSize    += sliceSize;

            if (pOut->pMipInfo != NULL)
            {
                pOut->pMipInfo[mipIdx].pitch  = pitch;
                pOut->pMipInfo[mipIdx].height = height;
                pOut->pMipInfo[mipIdx].depth  = depth;

                // The slice size of a linear image was calculated above as if the "pitch" is 256 byte aligned.
                // However, the rendering pitch is aligned to 128 bytes, and that is what needs to be reported
                // to our clients.
                if (IsLinear(pIn->swizzleMode))
                {
                    pOut->pMipInfo[mipIdx].pitch = PowTwoAlign(mipExtents.width,  128u / bytesPerPixel);
                }
            }
        }
    }

    pOut->sliceSize        = mipChainSliceSize;
    pOut->surfSize         = mipChainSliceSize * pOut->numSlices;
    pOut->mipChainInTail   = (firstMipInTail == 0) ? TRUE : FALSE;
    pOut->firstMipIdInTail = firstMipInTail;

    if (pOut->pMipInfo != NULL)
    {
       if (IsLinear(pIn->swizzleMode))
        {
            // 1. Linear swizzle mode doesn't have miptails.
            // 2. The organization of linear 3D mipmap resource is same as GFX11, we should use mip slice size to
            // caculate mip offset.
            ADDR_ASSERT(firstMipInTail == pIn->numMipLevels);

            UINT_64 sliceSize = 0;

            for (INT_32 i = static_cast<INT_32>(pIn->numMipLevels) - 1; i >= 0; i--)
            {
                pOut->pMipInfo[i].offset           = sliceSize;
                pOut->pMipInfo[i].macroBlockOffset = sliceSize;
                pOut->pMipInfo[i].mipTailOffset    = 0;

                sliceSize += mipSliceSize[i];
            }
        }
        else
        {
           UINT_64 offset         = 0;
           UINT_64 macroBlkOffset = 0;
           UINT_32 tailMaxDepth   = 0;

           ADDR_EXTENT3D  mipExtentFirstInTail = {};
           if (firstMipInTail != pIn->numMipLevels)
           {
              mipExtentFirstInTail = GetMipExtent(mip0Dims, firstMipInTail);

              offset         = blockSize *
                 PowTwoAlign(mipExtentFirstInTail.depth,
                             pOut->blockExtent.depth) / pOut->blockExtent.depth;
              macroBlkOffset = blockSize;
           }

           for (INT_32 i = firstMipInTail - 1; i >= 0; i--)
           {
              pOut->pMipInfo[i].offset           = offset;
              pOut->pMipInfo[i].macroBlockOffset = macroBlkOffset;
              pOut->pMipInfo[i].mipTailOffset    = 0;

              offset         += mipSize[i];
              macroBlkOffset += mipSliceSize[i];
           }

           GetMipOrigin(pIn, mipExtentFirstInTail, pOut);
        }
    }
}

/**
************************************************************************************************************************
*   Gfx12Lib::HwlComputeSurfaceInfo
*
*   @brief
*       Internal function to calculate alignment for a surface
*
*   @return
*       VOID
************************************************************************************************************************
*/
ADDR_E_RETURNCODE Gfx12Lib::HwlComputeSurfaceInfo(
     const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn,    ///< [in] input structure
     ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*      pOut    ///< [out] output structure
     ) const
{
    ComputeBlockDimensionForSurf(&pOut->blockExtent,
                                 pIn->bpp,
                                 pIn->numSamples,
                                 pIn->swizzleMode);

    ADDR_E_RETURNCODE  returnCode = ApplyCustomizedPitchHeight(pIn, pOut);

    if (returnCode == ADDR_OK)
    {
        pOut->numSlices = PowTwoAlign(pIn->numSlices, pOut->blockExtent.depth);
        pOut->baseAlign = 1 << GetBlockSizeLog2(pIn->swizzleMode);

        GetMipOffset(pIn, pOut);

        SanityCheckSurfSize(pIn, pOut);

        // Slices must be exact multiples of the block sizes.  However:
        // - with 3D images, one block will contain multiple slices, so that needs to be taken into account.
        // - with linear images that have only once slice, we may trim and use the pitch alignment for size.
        ADDR_ASSERT(((pOut->sliceSize * pOut->blockExtent.depth) %
                     GetBlockSize(pIn->swizzleMode, CanTrimLinearPadding(pIn))) == 0);
    }

    return returnCode;
}

/**
************************************************************************************************************************
*   Gfx12Lib::GetBaseMipExtents
*
*   @brief
*       Return the size of the base mip level in a nice cozy little structure.
*
************************************************************************************************************************
*/
ADDR_EXTENT3D Gfx12Lib::GetBaseMipExtents(
    const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn
    ) const
{
    return { pIn->width,
             pIn->height,
             (IsTex3d(pIn->resourceType) ? pIn->numSlices : 1) }; // slices is depth for 3d
}

/**
************************************************************************************************************************
*   Gfx12Lib::GetMaxNumMipsInTail
*
*   @brief
*       Return max number of mips in tails
*
*   @return
*       Max number of mips in tails
************************************************************************************************************************
*/
UINT_32 Gfx12Lib::GetMaxNumMipsInTail(
    Addr3SwizzleMode  swizzleMode,
    UINT_32           blockSizeLog2     ///< block size log2
    ) const
{
    UINT_32 effectiveLog2 = blockSizeLog2;
    UINT_32 mipsInTail    = 1;

    if (Is3dSwizzle(swizzleMode))
    {
        effectiveLog2 -= (blockSizeLog2 - 8) / 3;
    }

    if (effectiveLog2 > 8)
    {
        mipsInTail = (effectiveLog2 <= 11) ? (1 + (1 << (effectiveLog2 - 9))) : (effectiveLog2 - 4);
    }

    return mipsInTail;
}

/**
************************************************************************************************************************
*   Gfx12Lib::HwlComputeSurfaceAddrFromCoordTiled
*
*   @brief
*       Internal function to calculate address from coord for tiled swizzle surface
*
*   @return
*       ADDR_E_RETURNCODE
************************************************************************************************************************
*/
ADDR_E_RETURNCODE Gfx12Lib::HwlComputeSurfaceAddrFromCoordTiled(
     const ADDR3_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT* pIn,    ///< [in] input structure
     ADDR3_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT*      pOut    ///< [out] output structure
     ) const
{
    // 256B block cannot support 3D image.
    ADDR_ASSERT((IsTex3d(pIn->resourceType) && IsBlock256b(pIn->swizzleMode)) == FALSE);

    ADDR3_COMPUTE_SURFACE_INFO_INPUT  localIn = {};
    ADDR3_COMPUTE_SURFACE_INFO_OUTPUT localOut = {};
    ADDR3_MIP_INFO                    mipInfo[MaxMipLevels];

    localIn.size         = sizeof(localIn);
    localIn.flags        = pIn->flags;
    localIn.swizzleMode  = pIn->swizzleMode;
    localIn.resourceType = pIn->resourceType;
    localIn.format       = ADDR_FMT_INVALID;
    localIn.bpp          = pIn->bpp;
    localIn.width        = Max(pIn->unAlignedDims.width, 1u);
    localIn.height       = Max(pIn->unAlignedDims.height, 1u);
    localIn.numSlices    = Max(pIn->unAlignedDims.depth, 1u);
    localIn.numMipLevels = Max(pIn->numMipLevels, 1u);
    localIn.numSamples   = Max(pIn->numSamples, 1u);

    localOut.size        = sizeof(localOut);
    localOut.pMipInfo    = mipInfo;

    ADDR_E_RETURNCODE ret = ComputeSurfaceInfo(&localIn, &localOut);

    if (ret == ADDR_OK)
    {
        const UINT_32 elemLog2    = Log2(pIn->bpp >> 3);
        const UINT_32 blkSizeLog2 = GetBlockSizeLog2(pIn->swizzleMode);
        const UINT_32 eqIndex     = GetEquationTableEntry(pIn->swizzleMode, Log2(localIn.numSamples), elemLog2);

        if (eqIndex != ADDR_INVALID_EQUATION_INDEX)
        {
            const BOOL_32 inTail     = ((mipInfo[pIn->mipId].mipTailOffset != 0) && (blkSizeLog2 != Log2Size256));
            const BOOL_32 is3dNoMsaa = ((IsTex3d(pIn->resourceType) == TRUE) && (localIn.numSamples == 1));
            const UINT_64 sliceSize  = is3dNoMsaa ? (localOut.sliceSize * localOut.blockExtent.depth)
                                                  : localOut.sliceSize;
            const UINT_32 sliceId    = is3dNoMsaa ? (pIn->slice / localOut.blockExtent.depth) : pIn->slice;
            const UINT_32 x          = inTail ? (pIn->x + mipInfo[pIn->mipId].mipTailCoordX) : pIn->x;
            const UINT_32 y          = inTail ? (pIn->y + mipInfo[pIn->mipId].mipTailCoordY) : pIn->y;
            const UINT_32 z          = inTail ? (pIn->slice + mipInfo[pIn->mipId].mipTailCoordZ) : pIn->slice;
            const UINT_32 pb         = mipInfo[pIn->mipId].pitch / localOut.blockExtent.width;
            const UINT_32 yb         = pIn->y / localOut.blockExtent.height;
            const UINT_32 xb         = pIn->x / localOut.blockExtent.width;
            const UINT_64 blkIdx     = yb * pb + xb;
            const UINT_32 blkOffset  = ComputeOffsetFromEquation(&m_equationTable[eqIndex],
                                                                 x << elemLog2,
                                                                 y,
                                                                 z,
                                                                 pIn->sample);
            pOut->addr = sliceSize * sliceId +
                         mipInfo[pIn->mipId].macroBlockOffset +
                         (blkIdx << blkSizeLog2) +
                         blkOffset;
        }
        else
        {
            ret = ADDR_INVALIDPARAMS;
        }
    }

    return ret;
}

/**
************************************************************************************************************************
*   Gfx12Lib::HwlComputePipeBankXor
*
*   @brief
*       Generate a PipeBankXor value to be ORed into bits above numSwizzleBits of address
*
*   @return
*       PipeBankXor value
************************************************************************************************************************
*/
ADDR_E_RETURNCODE Gfx12Lib::HwlComputePipeBankXor(
    const ADDR3_COMPUTE_PIPEBANKXOR_INPUT* pIn,     ///< [in] input structure
    ADDR3_COMPUTE_PIPEBANKXOR_OUTPUT*      pOut     ///< [out] output structure
    ) const
{
    if ((m_numSwizzleBits != 0)               && // does this configuration support swizzling
        //         base address XOR in GFX12 will be applied to all blk_size = 4KB, 64KB, or 256KB swizzle modes,
        //         Note that Linear and 256B are excluded.
        (IsLinear(pIn->swizzleMode) == FALSE) &&
        (IsBlock256b(pIn->swizzleMode) == FALSE))
    {
        pOut->pipeBankXor = pIn->surfIndex % (1 << m_numSwizzleBits);
    }
    else
    {
        pOut->pipeBankXor = 0;
    }

    return ADDR_OK;
}

/**
************************************************************************************************************************
*   Gfx12Lib::ComputeOffsetFromEquation
*
*   @brief
*       Compute offset from equation
*
*   @return
*       Offset
************************************************************************************************************************
*/
UINT_32 Gfx12Lib::ComputeOffsetFromEquation(
    const ADDR_EQUATION* pEq,   ///< Equation
    UINT_32              x,     ///< x coord in bytes
    UINT_32              y,     ///< y coord in pixel
    UINT_32              z,     ///< z coord in slice
    UINT_32              s      ///< MSAA sample index
    ) const
{
    UINT_32 offset = 0;

    for (UINT_32 i = 0; i < pEq->numBits; i++)
    {
        UINT_32 v = 0;

        if (pEq->addr[i].valid)
        {
            if (pEq->addr[i].channel == 0)
            {
                v ^= (x >> pEq->addr[i].index) & 1;
            }
            else if (pEq->addr[i].channel == 1)
            {
                v ^= (y >> pEq->addr[i].index) & 1;
            }
            else if (pEq->addr[i].channel == 2)
            {
                v ^= (z >> pEq->addr[i].index) & 1;
            }
            else if (pEq->addr[i].channel == 3)
            {
                v ^= (s >> pEq->addr[i].index) & 1;
            }
            else
            {
                ADDR_ASSERT_ALWAYS();
            }
        }

        offset |= (v << i);
    }

    return offset;
}

/**
************************************************************************************************************************
*   Gfx12Lib::GetSwizzlePatternInfo
*
*   @brief
*       Get swizzle pattern
*
*   @return
*       Swizzle pattern information
************************************************************************************************************************
*/
const ADDR_SW_PATINFO* Gfx12Lib::GetSwizzlePatternInfo(
    Addr3SwizzleMode swizzleMode,       ///< Swizzle mode
    UINT_32          elemLog2,          ///< Element size in bytes log2
    UINT_32          numFrag            ///< Number of fragment
    ) const
{
    const ADDR_SW_PATINFO* patInfo = NULL;

    if (Is2dSwizzle(swizzleMode) == FALSE)
    {
        ADDR_ASSERT(numFrag == 1);
    }

    switch (swizzleMode)
    {
    case ADDR3_256KB_2D:
        switch (numFrag)
        {
        case 1:
            patInfo = GFX12_SW_256KB_2D_1xAA_PATINFO;
            break;
        case 2:
            patInfo = GFX12_SW_256KB_2D_2xAA_PATINFO;
            break;
        case 4:
            patInfo = GFX12_SW_256KB_2D_4xAA_PATINFO;
            break;
        case 8:
            patInfo = GFX12_SW_256KB_2D_8xAA_PATINFO;
            break;
        default:
            ADDR_ASSERT_ALWAYS();
        }
        break;
    case ADDR3_256KB_3D:
        patInfo = GFX12_SW_256KB_3D_PATINFO;
        break;
    case ADDR3_64KB_2D:
        switch (numFrag)
        {
        case 1:
            patInfo = GFX12_SW_64KB_2D_1xAA_PATINFO;
            break;
        case 2:
            patInfo = GFX12_SW_64KB_2D_2xAA_PATINFO;
            break;
        case 4:
            patInfo = GFX12_SW_64KB_2D_4xAA_PATINFO;
            break;
        case 8:
            patInfo = GFX12_SW_64KB_2D_8xAA_PATINFO;
            break;
        default:
            ADDR_ASSERT_ALWAYS();
        }
        break;
    case ADDR3_64KB_3D:
        patInfo = GFX12_SW_64KB_3D_PATINFO;
        break;
    case ADDR3_4KB_2D:
        switch (numFrag)
        {
        case 1:
            patInfo = GFX12_SW_4KB_2D_1xAA_PATINFO;
            break;
        case 2:
            patInfo = GFX12_SW_4KB_2D_2xAA_PATINFO;
            break;
        case 4:
            patInfo = GFX12_SW_4KB_2D_4xAA_PATINFO;
            break;
        case 8:
            patInfo = GFX12_SW_4KB_2D_8xAA_PATINFO;
            break;
        default:
            ADDR_ASSERT_ALWAYS();
        }
        break;
    case ADDR3_4KB_3D:
        patInfo = GFX12_SW_4KB_3D_PATINFO;
        break;
    case ADDR3_256B_2D:
        switch (numFrag)
        {
        case 1:
            patInfo = GFX12_SW_256B_2D_1xAA_PATINFO;
            break;
        case 2:
            patInfo = GFX12_SW_256B_2D_2xAA_PATINFO;
            break;
        case 4:
            patInfo = GFX12_SW_256B_2D_4xAA_PATINFO;
            break;
        case 8:
            patInfo = GFX12_SW_256B_2D_8xAA_PATINFO;
            break;
        default:
            break;
        }
        break;
    default:
        ADDR_ASSERT_ALWAYS();
        break;
    }

    return (patInfo != NULL) ? &patInfo[elemLog2] : NULL;
}
/**
************************************************************************************************************************
*   Gfx12Lib::HwlInitGlobalParams
*
*   @brief
*       Initializes global parameters
*
*   @return
*       TRUE if all settings are valid
*
************************************************************************************************************************
*/
BOOL_32 Gfx12Lib::HwlInitGlobalParams(
    const ADDR_CREATE_INPUT* pCreateIn) ///< [in] create input
{
    BOOL_32              valid = TRUE;
    GB_ADDR_CONFIG_GFX12 gbAddrConfig;

    gbAddrConfig.u32All = pCreateIn->regValue.gbAddrConfig;

    switch (gbAddrConfig.bits.NUM_PIPES)
    {
        case ADDR_CONFIG_1_PIPE:
            m_pipesLog2 = 0;
            break;
        case ADDR_CONFIG_2_PIPE:
            m_pipesLog2 = 1;
            break;
        case ADDR_CONFIG_4_PIPE:
            m_pipesLog2 = 2;
            break;
        case ADDR_CONFIG_8_PIPE:
            m_pipesLog2 = 3;
            break;
        case ADDR_CONFIG_16_PIPE:
            m_pipesLog2 = 4;
            break;
        case ADDR_CONFIG_32_PIPE:
            m_pipesLog2 = 5;
            break;
        case ADDR_CONFIG_64_PIPE:
            m_pipesLog2 = 6;
            break;
        default:
            ADDR_ASSERT_ALWAYS();
            valid = FALSE;
            break;
    }

    switch (gbAddrConfig.bits.PIPE_INTERLEAVE_SIZE)
    {
        case ADDR_CONFIG_PIPE_INTERLEAVE_256B:
            m_pipeInterleaveLog2 = 8;
            break;
        case ADDR_CONFIG_PIPE_INTERLEAVE_512B:
            m_pipeInterleaveLog2 = 9;
            break;
        case ADDR_CONFIG_PIPE_INTERLEAVE_1KB:
            m_pipeInterleaveLog2 = 10;
            break;
        case ADDR_CONFIG_PIPE_INTERLEAVE_2KB:
            m_pipeInterleaveLog2 = 11;
            break;
        default:
            ADDR_ASSERT_ALWAYS();
            valid = FALSE;
            break;
    }

    m_numSwizzleBits = ((m_pipesLog2 >= 3) ? m_pipesLog2 - 2 : 0);

    if (valid)
    {
        InitEquationTable();
    }

    return valid;
}

/**
************************************************************************************************************************
*   Gfx12Lib::HwlComputeNonBlockCompressedView
*
*   @brief
*       Compute non-block-compressed view for a given mipmap level/slice.
*
*   @return
*       ADDR_E_RETURNCODE
************************************************************************************************************************
*/
ADDR_E_RETURNCODE Gfx12Lib::HwlComputeNonBlockCompressedView(
    const ADDR3_COMPUTE_NONBLOCKCOMPRESSEDVIEW_INPUT* pIn,    ///< [in] input structure
    ADDR3_COMPUTE_NONBLOCKCOMPRESSEDVIEW_OUTPUT*      pOut    ///< [out] output structure
    ) const
{
    ADDR_E_RETURNCODE returnCode = ADDR_OK;

    if (((pIn->format < ADDR_FMT_ASTC_4x4) || (pIn->format > ADDR_FMT_ETC2_128BPP)) &&
        ((pIn->format < ADDR_FMT_BC1) || (pIn->format > ADDR_FMT_BC7)))
    {
        // Only support BC1~BC7, ASTC, or ETC2 for now...
        returnCode = ADDR_NOTSUPPORTED;
    }
    else
    {
        UINT_32 bcWidth, bcHeight;
        const UINT_32 bpp = GetElemLib()->GetBitsPerPixel(pIn->format, NULL, &bcWidth, &bcHeight);

        ADDR3_COMPUTE_SURFACE_INFO_INPUT infoIn = {};
        infoIn.size         = sizeof(infoIn);
        infoIn.flags        = pIn->flags;
        infoIn.swizzleMode  = pIn->swizzleMode;
        infoIn.resourceType = pIn->resourceType;
        infoIn.format       = pIn->format;
        infoIn.bpp          = bpp;
        infoIn.width        = RoundUpQuotient(pIn->unAlignedDims.width, bcWidth);
        infoIn.height       = RoundUpQuotient(pIn->unAlignedDims.height, bcHeight);
        infoIn.numSlices    = pIn->unAlignedDims.depth;
        infoIn.numMipLevels = pIn->numMipLevels;
        infoIn.numSamples   = 1;

        ADDR3_MIP_INFO mipInfo[MaxMipLevels] = {};

        ADDR3_COMPUTE_SURFACE_INFO_OUTPUT infoOut = {};
        infoOut.size     = sizeof(infoOut);
        infoOut.pMipInfo = mipInfo;

        returnCode = HwlComputeSurfaceInfo(&infoIn, &infoOut);

        if (returnCode == ADDR_OK)
        {
            ADDR3_COMPUTE_SUBRESOURCE_OFFSET_FORSWIZZLEPATTERN_INPUT subOffIn = {};
            subOffIn.size             = sizeof(subOffIn);
            subOffIn.swizzleMode      = infoIn.swizzleMode;
            subOffIn.resourceType     = infoIn.resourceType;
            subOffIn.pipeBankXor      = pIn->pipeBankXor;
            subOffIn.slice            = pIn->slice;
            subOffIn.sliceSize        = infoOut.sliceSize;
            subOffIn.macroBlockOffset = mipInfo[pIn->mipId].macroBlockOffset;
            subOffIn.mipTailOffset    = mipInfo[pIn->mipId].mipTailOffset;

            ADDR3_COMPUTE_SUBRESOURCE_OFFSET_FORSWIZZLEPATTERN_OUTPUT subOffOut = {};
            subOffOut.size = sizeof(subOffOut);

            // For any mipmap level, move nonBc view base address by offset
            HwlComputeSubResourceOffsetForSwizzlePattern(&subOffIn, &subOffOut);
            pOut->offset = subOffOut.offset;

            ADDR3_COMPUTE_SLICE_PIPEBANKXOR_INPUT slicePbXorIn = {};
            slicePbXorIn.size            = sizeof(slicePbXorIn);
            slicePbXorIn.swizzleMode     = infoIn.swizzleMode;
            slicePbXorIn.resourceType    = infoIn.resourceType;
            slicePbXorIn.bpe             = infoIn.bpp;
            slicePbXorIn.basePipeBankXor = pIn->pipeBankXor;
            slicePbXorIn.slice           = pIn->slice;
            slicePbXorIn.numSamples      = 1;

            ADDR3_COMPUTE_SLICE_PIPEBANKXOR_OUTPUT slicePbXorOut = {};
            slicePbXorOut.size = sizeof(slicePbXorOut);

            // For any mipmap level, nonBc view should use computed pbXor
            HwlComputeSlicePipeBankXor(&slicePbXorIn, &slicePbXorOut);
            pOut->pipeBankXor = slicePbXorOut.pipeBankXor;

            const BOOL_32 tiled            = (pIn->swizzleMode != ADDR3_LINEAR);
            const BOOL_32 inTail           = tiled && (pIn->mipId >= infoOut.firstMipIdInTail);
            const UINT_32 requestMipWidth  =
                    RoundUpQuotient(Max(pIn->unAlignedDims.width  >> pIn->mipId, 1u), bcWidth);
            const UINT_32 requestMipHeight =
                    RoundUpQuotient(Max(pIn->unAlignedDims.height >> pIn->mipId, 1u), bcHeight);

            if (inTail)
            {
                // For mipmap level that is in mip tail block, hack a lot of things...
                // Basically all mipmap levels in tail block will be viewed as a small mipmap chain that all levels
                // are fit in tail block:

                // - mipId = relative mip id (which is counted from first mip ID in tail in original mip chain)
                pOut->mipId = pIn->mipId - infoOut.firstMipIdInTail;

                // - at least 2 mipmap levels (since only 1 mipmap level will not be viewed as mipmap!)
                pOut->numMipLevels = Max(infoIn.numMipLevels - infoOut.firstMipIdInTail, 2u);

                // - (mip0) width = requestMipWidth << mipId, the value can't exceed mip tail dimension threshold
                pOut->unAlignedDims.width  = Min(requestMipWidth << pOut->mipId, infoOut.blockExtent.width / 2);

                // - (mip0) height = requestMipHeight << mipId, the value can't exceed mip tail dimension threshold
                pOut->unAlignedDims.height = Min(requestMipHeight << pOut->mipId, infoOut.blockExtent.height);
            }
            // This check should cover at least mipId == 0
            else if ((requestMipWidth << pIn->mipId) == infoIn.width)
            {
                // For mipmap level [N] that is not in mip tail block and downgraded without losing element:
                // - only one mipmap level and mipId = 0
                pOut->mipId        = 0;
                pOut->numMipLevels = 1;

                // (mip0) width = requestMipWidth
                pOut->unAlignedDims.width  = requestMipWidth;

                // (mip0) height = requestMipHeight
                pOut->unAlignedDims.height = requestMipHeight;
            }
            else
            {
                // For mipmap level [N] that is not in mip tail block and downgraded with element losing,
                // We have to make it a multiple mipmap view (2 levels view here), add one extra element if needed,
                // because single mip view may have different pitch value than original (multiple) mip view...
                // A simple case would be:
                // - 64KB block swizzle mode, 8 Bytes-Per-Element. Block dim = [0x80, 0x40]
                // - 2 mipmap levels with API mip0 width = 0x401/mip1 width = 0x200 and non-BC view
                //   mip0 width = 0x101/mip1 width = 0x80
                // By multiple mip view, the pitch for mip level 1 would be 0x100 bytes, due to rounding up logic in
                // GetMipSize(), and by single mip level view the pitch will only be 0x80 bytes.

                // - 2 levels and mipId = 1
                pOut->mipId        = 1;
                pOut->numMipLevels = 2;

                const UINT_32 upperMipWidth  =
                    RoundUpQuotient(Max(pIn->unAlignedDims.width  >> (pIn->mipId - 1), 1u), bcWidth);
                const UINT_32 upperMipHeight =
                    RoundUpQuotient(Max(pIn->unAlignedDims.height >> (pIn->mipId - 1), 1u), bcHeight);

                const BOOL_32 needToAvoidInTail = tiled                                              &&
                                                  (requestMipWidth <= infoOut.blockExtent.width / 2) &&
                                                  (requestMipHeight <= infoOut.blockExtent.height);

                const UINT_32 hwMipWidth  =
                    PowTwoAlign(ShiftCeil(infoIn.width, pIn->mipId), infoOut.blockExtent.width);
                const UINT_32 hwMipHeight =
                    PowTwoAlign(ShiftCeil(infoIn.height, pIn->mipId), infoOut.blockExtent.height);

                const BOOL_32 needExtraWidth =
                    ((upperMipWidth < requestMipWidth * 2) ||
                     ((upperMipWidth == requestMipWidth * 2) &&
                      ((needToAvoidInTail == TRUE) ||
                       (hwMipWidth > PowTwoAlign(requestMipWidth, infoOut.blockExtent.width)))));

                const BOOL_32 needExtraHeight =
                    ((upperMipHeight < requestMipHeight * 2) ||
                     ((upperMipHeight == requestMipHeight * 2) &&
                      ((needToAvoidInTail == TRUE) ||
                       (hwMipHeight > PowTwoAlign(requestMipHeight, infoOut.blockExtent.height)))));

                // (mip0) width = requestLastMipLevelWidth
                pOut->unAlignedDims.width  = upperMipWidth + (needExtraWidth ? 1: 0);

                // (mip0) height = requestLastMipLevelHeight
                pOut->unAlignedDims.height = upperMipHeight + (needExtraHeight ? 1: 0);
            }

            // Assert the downgrading from this mip[0] width would still generate correct mip[N] width
            ADDR_ASSERT(ShiftRight(pOut->unAlignedDims.width, pOut->mipId)  == requestMipWidth);
            // Assert the downgrading from this mip[0] height would still generate correct mip[N] height
            ADDR_ASSERT(ShiftRight(pOut->unAlignedDims.height, pOut->mipId) == requestMipHeight);
        }
    }

    return returnCode;
}

/**
************************************************************************************************************************
*   Gfx12Lib::HwlComputeSubResourceOffsetForSwizzlePattern
*
*   @brief
*       Compute sub resource offset to support swizzle pattern
*
*   @return
*       VOID
************************************************************************************************************************
*/
VOID Gfx12Lib::HwlComputeSubResourceOffsetForSwizzlePattern(
    const ADDR3_COMPUTE_SUBRESOURCE_OFFSET_FORSWIZZLEPATTERN_INPUT* pIn,    ///< [in] input structure
    ADDR3_COMPUTE_SUBRESOURCE_OFFSET_FORSWIZZLEPATTERN_OUTPUT*      pOut    ///< [out] output structure
    ) const
{
    pOut->offset = pIn->slice * pIn->sliceSize + pIn->macroBlockOffset;
}

/**
************************************************************************************************************************
*   Gfx12Lib::HwlComputeSlicePipeBankXor
*
*   @brief
*       Generate slice PipeBankXor value based on base PipeBankXor value and slice id
*
*   @return
*       PipeBankXor value
************************************************************************************************************************
*/
ADDR_E_RETURNCODE Gfx12Lib::HwlComputeSlicePipeBankXor(
    const ADDR3_COMPUTE_SLICE_PIPEBANKXOR_INPUT* pIn,   ///< [in] input structure
    ADDR3_COMPUTE_SLICE_PIPEBANKXOR_OUTPUT*      pOut   ///< [out] output structure
    ) const
{
    ADDR_E_RETURNCODE returnCode = ADDR_OK;

    // PipeBankXor is only applied to 4KB, 64KB and 256KB on GFX12.
    if ((IsLinear(pIn->swizzleMode) == FALSE) && (IsBlock256b(pIn->swizzleMode) == FALSE))
    {
        if (pIn->bpe == 0)
        {
            // Require a valid bytes-per-element value passed from client...
            returnCode = ADDR_INVALIDPARAMS;
        }
        else
        {
            const ADDR_SW_PATINFO* pPatInfo = GetSwizzlePatternInfo(pIn->swizzleMode,
                                                                    Log2(pIn->bpe >> 3),
                                                                    1);

            if (pPatInfo != NULL)
            {
                const UINT_32 elemLog2    = Log2(pIn->bpe >> 3);
                const UINT_32 eqIndex     = GetEquationTableEntry(pIn->swizzleMode, Log2(pIn->numSamples), elemLog2);

                const UINT_32 pipeBankXorOffset = ComputeOffsetFromEquation(&m_equationTable[eqIndex],
                                                                            0,
                                                                            0,
                                                                            pIn->slice,
                                                                            0);

                const UINT_32 pipeBankXor = pipeBankXorOffset >> m_pipeInterleaveLog2;

                // Should have no bit set under pipe interleave
                ADDR_ASSERT((pipeBankXor << m_pipeInterleaveLog2) == pipeBankXorOffset);

                pOut->pipeBankXor = pIn->basePipeBankXor ^ pipeBankXor;
            }
            else
            {
                // Should never come here...
                ADDR_NOT_IMPLEMENTED();

                returnCode = ADDR_NOTSUPPORTED;
            }
        }
    }
    else
    {
        pOut->pipeBankXor = 0;
    }

    return returnCode;
}

/**
************************************************************************************************************************
*   Gfx12Lib::SanityCheckSurfSize
*
*   @brief
*       Calculate the surface size via the exact hardware algorithm to see if it matches.
*
*   @return
************************************************************************************************************************
*/
void Gfx12Lib::SanityCheckSurfSize(
    const ADDR3_COMPUTE_SURFACE_INFO_INPUT*   pIn,
    const ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*  pOut
    ) const
{
#if DEBUG
    // Verify that the requested image size is valid for the below algorithm.  The below code includes
    // implicit assumptions about the surface dimensions being less than "MaxImageDim"; otherwise, it can't
    // calculate "firstMipInTail" accurately and the below assertion will trip incorrectly.
    //
    // Surfaces destined for use only on the SDMA engine can exceed the gfx-engine-imposed limitations of
    // the "maximum" image dimensions.
    if ((pIn->width  <= MaxImageDim)        &&
        (pIn->height <= MaxImageDim)        &&
        (pIn->numMipLevels <= MaxMipLevels) &&
        (UseCustomPitch(pIn)  == FALSE)     &&
        (UseCustomHeight(pIn) == FALSE)     &&
        // HiZS surfaces have a reduced image size (i.e,. each pixel represents an 8x8 region of the parent
        // image, at least for single samples) but they still have the same number of mip levels as the
        // parent image.  This disconnect produces false assertions below as the image size doesn't apparently
        // support the specified number of mip levels.
        ((pIn->flags.hiZHiS == 0) || (pIn->numMipLevels == 1))   &&
        !(pIn->flags.view3dAs2dArray))
    {
        UINT_32  lastMipSize   = 1;
        UINT_32  dataChainSize = 0;

        const ADDR_EXTENT3D  mip0Dims      = GetBaseMipExtents(pIn);
        const UINT_32        blockSizeLog2 = GetBlockSizeLog2(pIn->swizzleMode);
        const ADDR_EXTENT3D  tailMaxDim    = GetMipTailDim(pIn->swizzleMode, pOut->blockExtent);
        const UINT_32        maxMipsInTail = GetMaxNumMipsInTail(pIn->swizzleMode, blockSizeLog2);

        UINT_32  firstMipInTail = 0;
        for (INT_32 mipIdx = MaxMipLevels - 1; mipIdx >= 0; mipIdx--)
        {
            const ADDR_EXTENT3D  mipExtents = GetMipExtent(mip0Dims, mipIdx);

            if ((mipExtents.width  <= tailMaxDim.width)  &&
                (mipExtents.height <= tailMaxDim.height) &&
                ((static_cast<INT_32>(pIn->numMipLevels) - mipIdx) < static_cast<INT_32>(maxMipsInTail)))
            {
                firstMipInTail = mipIdx;
            }
        }

        for (INT_32  mipIdx = firstMipInTail - 1; mipIdx >= -1; mipIdx--)
        {
            const ADDR_EXTENT3D  mipExtents     = GetMipExtent(mip0Dims, mipIdx);
            const UINT_32        mipBlockWidth  = ShiftCeil(mipExtents.width,  Log2(pOut->blockExtent.width));
            const UINT_32        mipBlockHeight = ShiftCeil(mipExtents.height, Log2(pOut->blockExtent.height));

            if (mipIdx < (static_cast<INT_32>(pIn->numMipLevels) - 1))
            {
                dataChainSize += lastMipSize;
            }

            if (mipIdx >= 0)
            {
                lastMipSize = 4 * lastMipSize
                    - ((mipBlockWidth  & 1) ? mipBlockHeight : 0)
                    - ((mipBlockHeight & 1) ? mipBlockWidth  : 0)
                    - ((mipBlockWidth  & mipBlockHeight & 1) ? 1 : 0);
            }
        }

        if (CanTrimLinearPadding(pIn))
        {
            ADDR_ASSERT((pOut->sliceSize * pOut->blockExtent.depth) <= (dataChainSize << blockSizeLog2));
        }
        else
        {
            ADDR_ASSERT((pOut->sliceSize * pOut->blockExtent.depth) == (dataChainSize << blockSizeLog2));
        }
    }
#endif
}

} // V3
} // Addr
} // namespace rocr
