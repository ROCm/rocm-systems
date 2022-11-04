a = AVG(
    (
        (
            (
                (
                    (
                        (
                            (64 * (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16))
                            + (SQ_INSTS_VALU_MFMA_MOPS_F16 * 512)
                        )
                        + (SQ_INSTS_VALU_MFMA_MOPS_BF16 * 512)
                    )
                    + (64 * (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32))
                )
                + (SQ_INSTS_VALU_MFMA_MOPS_F32 * 512)
            )
            + (64 * (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64))
        )
        / (EndNs - BeginNs)
    )
)
