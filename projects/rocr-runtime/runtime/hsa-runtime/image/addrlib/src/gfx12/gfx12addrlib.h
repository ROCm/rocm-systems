/*
************************************************************************************************************************
*
*  Copyright (C) 2023 Advanced Micro Devices, Inc.  All rights reserved.
*  SPDX-License-Identifier: MIT
*
***********************************************************************************************************************/

/**
************************************************************************************************************************
* @file  gfx12addrlib.h
* @brief Contains the Gfx12Lib class definition.
************************************************************************************************************************
*/

#ifndef __GFX12_ADDR_LIB_H__
#define __GFX12_ADDR_LIB_H__

#include "addrlib3.h"
#include "coord.h"
#include "gfx12SwizzlePattern.h"

namespace rocr {
namespace Addr
{
namespace V3
{

/**
************************************************************************************************************************
* @brief GFX12 specific settings structure.
************************************************************************************************************************
*/
struct Gfx12ChipSettings
{
    struct
    {
        // Misc configuration bits
        UINT_32 reserved : 32;
    };
};

/**
************************************************************************************************************************
* @brief GFX12 data surface type.
************************************************************************************************************************
*/

/**
************************************************************************************************************************
* @brief This class is the GFX12 specific address library
*        function set.
************************************************************************************************************************
*/
class Gfx12Lib : public Lib
{
public:
    /// Creates Gfx12Lib object
    static Addr::Lib* CreateObj(const Client* pClient)
    {
        VOID* pMem = Object::ClientAlloc(sizeof(Gfx12Lib), pClient);
        return (pMem != NULL) ? new (pMem) Gfx12Lib(pClient) : NULL;
    }

protected:
    Gfx12Lib(const Client* pClient);
    virtual ~Gfx12Lib();

    // Meta surfaces such as Hi-S/Z are essentially images on GFX12, so just return the max
    // image alignment.
    virtual UINT_32 HwlComputeMaxMetaBaseAlignments() const { return 256 * 1024; }

    UINT_32 GetMaxNumMipsInTail(
        Addr3SwizzleMode  swizzleMode,
        UINT_32           blockSizeLog2) const;

    BOOL_32 IsInMipTail(
        const ADDR_EXTENT3D&  mipTailDim,
        const ADDR_EXTENT3D&  mipDims,
        UINT_32               maxNumMipsInTail,
        UINT_32               numMipsToTheEnd) const
    {
        BOOL_32 inTail = ((mipDims.width   <= mipTailDim.width)  &&
                          (mipDims.height  <= mipTailDim.height) &&
                          (numMipsToTheEnd <= maxNumMipsInTail));

        return inTail;
    }

    virtual ADDR_E_RETURNCODE HwlComputeSurfaceAddrFromCoordTiled(
        const ADDR3_COMPUTE_SURFACE_ADDRFROMCOORD_INPUT* pIn,
        ADDR3_COMPUTE_SURFACE_ADDRFROMCOORD_OUTPUT*      pOut) const;

    virtual ADDR_E_RETURNCODE HwlComputeNonBlockCompressedView(
        const ADDR3_COMPUTE_NONBLOCKCOMPRESSEDVIEW_INPUT* pIn,
        ADDR3_COMPUTE_NONBLOCKCOMPRESSEDVIEW_OUTPUT*      pOut) const;

    virtual VOID HwlComputeSubResourceOffsetForSwizzlePattern(
        const ADDR3_COMPUTE_SUBRESOURCE_OFFSET_FORSWIZZLEPATTERN_INPUT* pIn,
        ADDR3_COMPUTE_SUBRESOURCE_OFFSET_FORSWIZZLEPATTERN_OUTPUT*      pOut) const;

    virtual ADDR_E_RETURNCODE HwlComputeSlicePipeBankXor(
        const ADDR3_COMPUTE_SLICE_PIPEBANKXOR_INPUT* pIn,
        ADDR3_COMPUTE_SLICE_PIPEBANKXOR_OUTPUT*      pOut) const;

    virtual UINT_32 HwlGetEquationTableInfo(const ADDR_EQUATION** ppEquationTable) const
    {
        *ppEquationTable = m_equationTable;

        return m_numEquations;
    }

private:
    Gfx12ChipSettings m_settings;
    static const SwizzleModeFlags SwizzleModeTable[ADDR3_MAX_TYPE];

    virtual ADDR_E_RETURNCODE HwlComputePipeBankXor(
        const ADDR3_COMPUTE_PIPEBANKXOR_INPUT* pIn,
        ADDR3_COMPUTE_PIPEBANKXOR_OUTPUT*      pOut) const override;

    virtual BOOL_32 HwlInitGlobalParams(const ADDR_CREATE_INPUT* pCreateIn) override;

    void SanityCheckSurfSize(
        const ADDR3_COMPUTE_SURFACE_INFO_INPUT*   pIn,
        const ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*  pOut) const;

    UINT_32           m_numSwizzleBits;

    static const ADDR_EXTENT3D Block4K_Log2_3d[];
    static const ADDR_EXTENT3D Block64K_Log2_3d[];
    static const ADDR_EXTENT3D Block256K_Log2_3d[];

    // Initialize equation table
    VOID InitEquationTable();

    VOID GetSwizzlePatternFromPatternInfo(
        const ADDR_SW_PATINFO* pPatInfo,
        ADDR_BIT_SETTING       (&pSwizzle)[Log2Size256K]) const
    {
        memcpy(pSwizzle,
               GFX12_SW_PATTERN_NIBBLE1[pPatInfo->nibble1Idx],
               sizeof(GFX12_SW_PATTERN_NIBBLE1[pPatInfo->nibble1Idx]));

        memcpy(&pSwizzle[8],
               GFX12_SW_PATTERN_NIBBLE2[pPatInfo->nibble2Idx],
               sizeof(GFX12_SW_PATTERN_NIBBLE2[pPatInfo->nibble2Idx]));

        memcpy(&pSwizzle[12],
               GFX12_SW_PATTERN_NIBBLE3[pPatInfo->nibble3Idx],
               sizeof(GFX12_SW_PATTERN_NIBBLE3[pPatInfo->nibble3Idx]));

        memcpy(&pSwizzle[16],
               GFX12_SW_PATTERN_NIBBLE4[pPatInfo->nibble4Idx],
               sizeof(GFX12_SW_PATTERN_NIBBLE4[pPatInfo->nibble4Idx]));
    }

    VOID ConvertSwizzlePatternToEquation(
        UINT_32                elemLog2,
        Addr3SwizzleMode       swMode,
        const ADDR_SW_PATINFO* pPatInfo,
        ADDR_EQUATION* pEquation) const;

    ADDR_EXTENT3D GetBaseMipExtents(
        const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn) const;

    ADDR_EXTENT3D GetBlockPixelDimensions(
        Addr3SwizzleMode  swizzleMode,
        UINT_32           log2BytesPerPixel) const;

    virtual ADDR_E_RETURNCODE HwlComputeSurfaceInfo(
         const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn,
         ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*      pOut) const override;

    static ADDR_EXTENT3D GetMipExtent(
        const ADDR_EXTENT3D&  mip0,
        UINT_32               mipId)
    {
        return {
            ShiftCeil(Max(mip0.width, 1u),  mipId),
            ShiftCeil(Max(mip0.height, 1u), mipId),
            ShiftCeil(Max(mip0.depth, 1u),  mipId)
        };
    }

    //# See 6.3 in //gfxip/gfx10/doc/architecture/ImageAddressing/gfx10_image_addressing.docx
    // miptail is applied to only larger block size (4kb, 64kb, 256kb), so there is no miptail in linear and
    // 256b_2d addressing since they are both 256b block.
    BOOL_32 SupportsMipTail(Addr3SwizzleMode swizzleMode) const
    {
        return GetBlockSize(swizzleMode) > 256u;
    }

    UINT_32 ComputeOffsetFromEquation(
        const ADDR_EQUATION* pEq,
        UINT_32              x,
        UINT_32              y,
        UINT_32              z,
        UINT_32              s) const;

    const ADDR_SW_PATINFO* GetSwizzlePatternInfo(
        Addr3SwizzleMode swizzleMode,
        UINT_32          log2Elem,
        UINT_32          numFrag) const;

    VOID GetMipOffset(
         const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn,
         ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*      pOut) const;

    VOID GetMipOrigin(
         const ADDR3_COMPUTE_SURFACE_INFO_INPUT* pIn,
         const ADDR_EXTENT3D&                    mipExtentFirstInTail,
         ADDR3_COMPUTE_SURFACE_INFO_OUTPUT*      pOut) const;
};

} // V3
} // Addr
} // namespace rocr
#endif
